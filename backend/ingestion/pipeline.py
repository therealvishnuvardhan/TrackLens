from __future__ import annotations
import os
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from backend.engine.models import ImageVersion
from backend.ingestion.exif_parser import ExifParser
from backend.ingestion.c2pa_verifier import C2PAVerifier
from backend.ingestion.video_keyframe import VideoKeyframeExtractor
from backend.ingestion.embedder import ImageEmbedder


class IngestionPipeline:
    """
    Unified Ingestion & Fingerprinting Pipeline (Steps 1 & 2):
    Handles user media uploads (images and videos).
    1. Extracts EXIF, camera model, and platform-stripping clues.
    2. Validates C2PA cryptographic Content Credentials.
    3. Extracts scene-change keyframes for video uploads.
    4. Computes invariant perceptual fingerprints (pHash, feature vectors).
    """

    VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}

    def __init__(self, keyframe_extractor: Optional[VideoKeyframeExtractor] = None):
        self.keyframe_extractor = keyframe_extractor or VideoKeyframeExtractor()

    def process_upload(
        self,
        file_path: str,
        platform_hint: str = "User Upload",
        author_hint: Optional[str] = None,
        caption_hint: Optional[str] = None,
    ) -> List[ImageVersion]:
        """
        Processes an uploaded media file and returns one or more ImageVersion instances.
        For an image, returns [ImageVersion].
        For a video, returns [ImageVersion for each keyframe].
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Uploaded file does not exist: {file_path}")

        _, ext = os.path.splitext(file_path.lower())

        if ext in self.VIDEO_EXTENSIONS:
            return self._process_video(file_path, platform_hint, author_hint, caption_hint)
        elif ext in self.IMAGE_EXTENSIONS or self._is_readable_image(file_path):
            return [self._process_single_image(file_path, platform_hint, author_hint, caption_hint)]
        else:
            raise ValueError(f"Unsupported media format: {ext or 'unknown'}")

    @staticmethod
    def _is_readable_image(file_path: str) -> bool:
        try:
            from PIL import Image
            with Image.open(file_path) as im:
                im.verify()
            return True
        except Exception:
            return False

    def _process_single_image(
        self,
        image_path: str,
        platform: str,
        author: Optional[str],
        caption: Optional[str],
    ) -> ImageVersion:
        # 1. EXIF Metadata
        exif_info = ExifParser.extract_metadata(image_path)

        # 2. C2PA Content Credentials
        c2pa_info = C2PAVerifier.verify_content_credentials(image_path)

        # 3. Perceptual Fingerprints
        fingerprint = ImageEmbedder.compute_fingerprint(image_path)

        # 4. Determine timestamp: EXIF timestamp if present, otherwise file mtime
        ts = exif_info.get("parsed_timestamp")
        if not ts:
            mtime = os.path.getmtime(image_path)
            ts = datetime.utcfromtimestamp(mtime)

        # 5. Base Source Trust Score
        trust = 0.50
        if c2pa_info.get("is_signed"):
            trust = 0.95
        elif exif_info.get("make") or exif_info.get("model"):
            trust = 0.85
        elif exif_info.get("platform_wash_detected"):
            trust = 0.40

        dims = tuple(fingerprint["dimensions"]) if fingerprint.get("dimensions") else None

        return ImageVersion(
            id=f"upload_{uuid.uuid4().hex[:8]}",
            platform=platform,
            author=author,
            url=f"file://{os.path.abspath(image_path)}",
            timestamp=ts,
            caption=caption or "User submitted media",
            image_path=image_path,
            dimensions=dims,
            phash=fingerprint.get("phash"),
            c2pa=c2pa_info if c2pa_info.get("is_signed") else None,
            exif=exif_info.get("raw_tags", {}),
            source_trust_score=trust,
        )

    def _process_video(
        self,
        video_path: str,
        platform: str,
        author: Optional[str],
        caption: Optional[str],
    ) -> List[ImageVersion]:
        keyframes = self.keyframe_extractor.extract_keyframes(video_path)
        versions = []

        for kf in keyframes:
            kf_img_path = kf["image_path"]
            idx = kf["frame_index"]
            ts_sec = kf["timestamp_sec"]

            fingerprint = ImageEmbedder.compute_fingerprint(kf_img_path)
            dims = tuple(fingerprint["dimensions"]) if fingerprint.get("dimensions") else None

            v = ImageVersion(
                id=f"video_kf_{idx:04d}_{uuid.uuid4().hex[:6]}",
                platform=f"{platform} (Keyframe @ {ts_sec}s)",
                author=author,
                url=f"file://{os.path.abspath(video_path)}#t={ts_sec}",
                timestamp=datetime.utcnow(),
                caption=f"{caption or 'Video segment'} [Scene Change Score: {kf.get('transition_score')}]",
                image_path=kf_img_path,
                dimensions=dims,
                phash=fingerprint.get("phash"),
                c2pa=None,
                exif={},
                source_trust_score=0.50,
            )
            versions.append(v)

        return versions
