from __future__ import annotations
import io

from PIL import Image, ImageChops, ImageEnhance


def compute_ela_map(image_path: str, output_path: str, quality: int = 90) -> str:
    """
    Computes an Error Level Analysis (ELA) map by re-compressing at the given JPEG quality
    and amplifying the per-pixel difference. Returns output_path, or "" on failure.
    """
    try:
        with Image.open(image_path) as src:
            im = src.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality)
        buf.seek(0)
        resaved = Image.open(buf).convert("RGB")
        ela_im = ImageChops.difference(im, resaved)
        max_diff = max(ex[1] for ex in ela_im.getextrema())
        ela_im = ImageEnhance.Brightness(ela_im).enhance(255.0 / max(max_diff, 1))
        ela_im.save(output_path, "PNG")
        return output_path
    except Exception as e:
        print(f"[ELA] Failed to compute ELA: {e}")
        return ""
