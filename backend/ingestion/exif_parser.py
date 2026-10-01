from __future__ import annotations
import os
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from PIL import Image, ExifTags


class ExifParser:
    """
    Extracts embedded EXIF and camera metadata from image files.
    Identifies camera make/model, original capture timestamp, lens parameters,
    editing software signatures, and flags platform-stripping.
    """

    KNOWN_EDITING_SOFTWARE = [
        "photoshop", "lightroom", "gimp", "snapseed", "canva",
        "picsart", "vsco", "afterlight", "facetune", "procreate", "pixelmator",
        "affinity", "paint.net", "photopea", "fotor", "meitu", "faceapp", "remini", "lensa"
    ]

    SOCIAL_PLATFORMS = [
        "whatsapp", "twitter", "instagram", "facebook", "reddit", "telegram", "tiktok"
    ]

    @classmethod
    def extract_metadata(cls, image_path: str) -> Dict[str, Any]:
        """
        Parses EXIF tags from the image. Returns a normalized metadata dict.
        """
        result: Dict[str, Any] = {
            "has_exif": False,
            "platform_wash_detected": True,  # Default to washed until camera tags are proven
            "make": None,
            "model": None,
            "lens": None,
            "software": None,
            "datetime_original": None,
            "parsed_timestamp": None,
            "is_edited": False,
            "editing_software_detected": None,
            "raw_tags": {},
        }

        if not os.path.exists(image_path):
            return result

        try:
            with Image.open(image_path) as img:
                exif_data = img.getexif()
                if not exif_data:
                    return result

                result["has_exif"] = True
                readable_tags = {}

                # Map numeric tag IDs to tag names
                for tag_id, value in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    # Handle binary / un-serializable values
                    if isinstance(value, bytes):
                        try:
                            value = value.decode("utf-8", errors="ignore").strip("\x00")
                        except Exception:
                            value = f"<bytes len={len(value)}>"
                    readable_tags[tag_name] = value

                # Check Exif sub-IFD (DateTimeOriginal is often inside IFD 34665)
                try:
                    for ifd_id in [ExifTags.IFD.Exif, ExifTags.IFD.GPSInfo]:
                        sub_ifd = exif_data.get_ifd(ifd_id)
                        for tag_id, value in sub_ifd.items():
                            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                            if isinstance(value, bytes):
                                try:
                                    value = value.decode("utf-8", errors="ignore").strip("\x00")
                                except Exception:
                                    value = f"<bytes len={len(value)}>"
                            readable_tags[tag_name] = value
                except Exception:
                    pass

                result["raw_tags"] = {k: cls._json_safe(v) for k, v in readable_tags.items()}

                # Camera details
                result["make"] = readable_tags.get("Make")
                result["model"] = readable_tags.get("Model")
                result["software"] = readable_tags.get("Software")
                result["lens"] = readable_tags.get("LensModel") or readable_tags.get("LensMake")

                # Parse original capture date
                dt_str = readable_tags.get("DateTimeOriginal") or readable_tags.get("DateTime")
                if dt_str:
                    result["datetime_original"] = str(dt_str)
                    result["parsed_timestamp"] = cls._parse_exif_date(str(dt_str))

                # Camera indicator means file has not been completely washed by a social messaging app
                if result["make"] or result["model"]:
                    result["platform_wash_detected"] = False

                # Check for editing software tags
                soft = str(result["software"] or "").lower()
                for editor in cls.KNOWN_EDITING_SOFTWARE:
                    if editor in soft:
                        result["is_edited"] = True
                        result["editing_software_detected"] = editor
                        break

        except Exception as e:
            result["error"] = str(e)

        return result

    @classmethod
    def _json_safe(cls, value, depth: int = 0):
        """EXIF values include IFDRational, tuples and nested IFDs; make them JSON-serialisable."""
        if isinstance(value, (str, int, bool)) or value is None:
            return value.replace("\x00", "") if isinstance(value, str) else value
        if isinstance(value, float):
            return value if value == value and abs(value) != float("inf") else None
        if isinstance(value, bytes):
            return f"<bytes len={len(value)}>"
        if depth < 3 and isinstance(value, (tuple, list)):
            return [cls._json_safe(v, depth + 1) for v in value[:32]]
        if depth < 3 and isinstance(value, dict):
            return {str(k): cls._json_safe(v, depth + 1) for k, v in list(value.items())[:32]}
        try:
            f = float(value)  # IFDRational and similar numeric wrappers
            return f if f == f and abs(f) != float("inf") else None
        except Exception:
            return str(value)[:200]

    @staticmethod
    def _parse_exif_date(date_str: str) -> Optional[datetime]:
        """
        Parses standard EXIF format 'YYYY:MM:DD HH:MM:SS' or ISO formats.
        """
        clean_str = date_str.strip()
        for fmt in (
            "%Y:%m:%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y:%m:%d %H:%M",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
        ):
            try:
                return datetime.strptime(clean_str, fmt)
            except ValueError:
                continue
        return None
