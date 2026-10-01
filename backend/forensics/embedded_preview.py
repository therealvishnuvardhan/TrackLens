from __future__ import annotations
import io
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from backend.search.match_verifier import MatchVerifier

COMPARE_WIDTH = 480
MIN_PREVIEW_SIDE = 64


class EmbeddedPreviewCheck:
    """
    Cameras and phones store a small preview (EXIF thumbnail, MPF preview, Photoshop
    thumbnail) inside the JPEG at capture time. Many editors rewrite the main image but
    leave that preview untouched, so a preview that no longer matches the visible picture
    is direct evidence of editing — and the preview itself shows the pre-edit content.
    """

    def check(self, image_path: str, out_dir: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "status": "absent",  # absent | consistent | cropped | mismatch | unreadable
            "previews_found": 0,
            "preview_path": None,
            "diff_path": None,
            "diff_fraction": None,
            "preview_size": None,
            "detail": "No embedded camera preview in this file (common once an image has been shared online).",
        }
        try:
            with open(image_path, "rb") as f:
                data = f.read()
            main = self._load_main(image_path)
        except Exception as e:
            result.update(status="unreadable", detail=f"Could not read file: {e}")
            return result

        previews = self._find_previews(data, main.size)
        result["previews_found"] = len(previews)
        if not previews:
            return result

        # Prefer the largest preview: it carries the most detail.
        preview = max(previews, key=lambda im: im.size[0] * im.size[1])
        preview = self._strip_letterbox(preview)
        result["preview_size"] = list(preview.size)
        os.makedirs(out_dir, exist_ok=True)
        preview_path = os.path.join(out_dir, "embedded_preview.jpg")
        preview.save(preview_path, "JPEG", quality=95)
        result["preview_path"] = preview_path

        main_aspect = main.size[0] / float(main.size[1])
        prev_aspect = preview.size[0] / float(preview.size[1])
        if abs(main_aspect - prev_aspect) / main_aspect > 0.04:
            result.update(
                status="cropped",
                detail=(
                    f"The embedded camera preview has a different shape ({preview.size[0]}×{preview.size[1]}) "
                    f"from the visible image ({main.size[0]}×{main.size[1]}): the photo was cropped or "
                    "re-framed after it was taken. The preview shows the original framing."
                ),
            )
            return result

        diff_fraction, diff_vis = self._compare(main, preview)
        result["diff_fraction"] = round(diff_fraction, 4)
        if diff_fraction >= 0.01:
            diff_path = os.path.join(out_dir, "embedded_preview_diff.png")
            cv2.imwrite(diff_path, diff_vis)
            result.update(
                status="mismatch",
                diff_path=diff_path,
                detail=(
                    f"The camera preview stored inside the file differs from the visible image in "
                    f"{diff_fraction * 100:.1f}% of the frame. The pixels were changed after capture "
                    "while the original preview was left behind — it shows the image before editing."
                ),
            )
        else:
            result.update(
                status="consistent",
                detail="The embedded camera preview matches the visible image — no edits since capture were detected this way.",
            )
        return result

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _load_main(path: str) -> Image.Image:
        with Image.open(path) as im:
            im.load()
            return im.convert("RGB")

    @staticmethod
    def _find_previews(data: bytes, main_size: Tuple[int, int]) -> List[Image.Image]:
        """Finds every additional JPEG stream embedded in the file (EXIF IFD1, MPF, APP13...)."""
        previews: List[Image.Image] = []
        start = 2  # skip the main image's own SOI
        seen_sizes = set()
        while True:
            idx = data.find(b"\xff\xd8\xff", start)
            if idx == -1:
                break
            start = idx + 3
            end = data.find(b"\xff\xd9", idx)
            if end == -1:
                break
            # Thumbnails can contain FFD9 inside their own entropy data rarely; try a few ends.
            for _ in range(4):
                chunk = data[idx:end + 2]
                try:
                    im = Image.open(io.BytesIO(chunk))
                    im.load()
                    im = im.convert("RGB")
                    w, h = im.size
                    if min(w, h) >= MIN_PREVIEW_SIDE and (w, h) != tuple(main_size) and (w, h) not in seen_sizes:
                        previews.append(im)
                        seen_sizes.add((w, h))
                    break
                except Exception:
                    nxt = data.find(b"\xff\xd9", end + 2)
                    if nxt == -1:
                        break
                    end = nxt
        return previews

    @staticmethod
    def _strip_letterbox(im: Image.Image) -> Image.Image:
        """Some cameras pad the 160×120 thumbnail with black bars to fit a 4:3 box."""
        arr = np.asarray(im.convert("L"))
        rows = np.where(arr.max(axis=1) > 24)[0]
        cols = np.where(arr.max(axis=0) > 24)[0]
        if len(rows) == 0 or len(cols) == 0:
            return im
        top, bottom, left, right = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        if (bottom - top) < arr.shape[0] * 0.98 or (right - left) < arr.shape[1] * 0.98:
            return im.crop((left, top, right, bottom))
        return im

    @staticmethod
    def _compare(main: Image.Image, preview: Image.Image) -> Tuple[float, np.ndarray]:
        w = COMPARE_WIDTH
        h = max(8, int(round(w * preview.size[1] / float(preview.size[0]))))
        a = np.asarray(main.resize((w, h), Image.LANCZOS))
        b = np.asarray(preview.resize((w, h), Image.BICUBIC))
        a_lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB)
        b_lab = cv2.cvtColor(b, cv2.COLOR_RGB2LAB)
        # Previews are tiny and heavily compressed; blur both to compare content, not detail.
        a_lab = cv2.GaussianBlur(a_lab, (7, 7), 0)
        b_lab = cv2.GaussianBlur(b_lab, (7, 7), 0)
        mask = MatchVerifier._difference_mask(a_lab, b_lab, np.eye(3), (h, w))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)).astype(bool)
        fraction = float(mask.mean())

        # Visual: preview (left) | edited image with changed regions outlined (right)
        vis_main = cv2.cvtColor(a, cv2.COLOR_RGB2BGR).copy()
        contours, _ = cv2.findContours(cv2.dilate(mask.astype(np.uint8) * 255, np.ones((9, 9), np.uint8)),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_main, contours, -1, (60, 60, 255), 2)
        vis_prev = cv2.cvtColor(b, cv2.COLOR_RGB2BGR)
        gap = np.full((h, 8, 3), 12, np.uint8)
        return fraction, np.hstack([vis_prev, gap, vis_main])
