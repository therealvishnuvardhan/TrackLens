from __future__ import annotations
import io
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageChops

from backend.ingestion.exif_parser import ExifParser
from backend.ingestion.c2pa_verifier import C2PAVerifier

ANALYSIS_MAX_SIDE = 1024
BLOCK = 16


def _noisy_or(probabilities: List[float]) -> float:
    acc = 1.0
    for p in probabilities:
        acc *= 1.0 - max(0.0, min(1.0, p))
    return 1.0 - acc


def _robust_z(values: np.ndarray) -> np.ndarray:
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med))) * 1.4826 + 1e-6
    return (values - med) / mad


class ForensicClassifier:
    """
    Classifies an image as ORIGINAL or TAMPERED by fusing independent forensic signals:

      - ELA (error level analysis): regions whose JPEG error level is inconsistent with
        their texture, a typical sign of pasted/retouched content.
      - Noise residual consistency: spliced regions usually carry a different sensor-noise level.
      - Copy-move detection: ORB keypoints duplicated elsewhere in the same frame.
      - Metadata: editing-software signatures, C2PA content credentials.
      - Overlay detection: solid-colour banners carrying text (meme / "BREAKING" overlays).
      - Gemini multimodal verdict (when the API is reachable), weighted highest.

    These are heuristics: they surface evidence, not proof. Every signal is returned so the
    UI can show why a verdict was reached.
    """

    THRESHOLD = 0.5

    def classify(
        self,
        image_path: str,
        gemini_analysis: Optional[Dict[str, Any]] = None,
        overlay_output_path: Optional[str] = None,
        extra_signals: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        rgb = self._load_rgb(image_path)
        signals: List[Dict[str, Any]] = []
        suspicious_boxes: List[Tuple[int, int, int, int]] = []

        ela_signal, ela_boxes = self._ela_signal(rgb)
        signals.append(ela_signal)
        suspicious_boxes += ela_boxes

        noise_signal, noise_boxes = self._noise_signal(rgb)
        signals.append(noise_signal)
        suspicious_boxes += noise_boxes

        overlay_signal, overlay_boxes = self._overlay_signal(rgb)
        clone_signal, clone_boxes = self._copy_move_signal(rgb, exclude_boxes=overlay_boxes)
        signals.append(clone_signal)
        suspicious_boxes += clone_boxes
        signals.append(overlay_signal)
        suspicious_boxes += overlay_boxes
        # Text banners are evidence, but not worth reverse-searching as "pasted content"
        searchable_boxes = [b for b in suspicious_boxes if b not in overlay_boxes]

        meta_signal, c2pa_signed = self._metadata_signal(image_path)
        signals.append(meta_signal)

        # Learned / file-structure detectors computed outside this class (TruFor, embedded preview)
        for extra in extra_signals or []:
            signals.append(dict(extra))

        local_p = _noisy_or([s["score"] * s["weight"] for s in signals])
        if c2pa_signed:
            local_p *= 0.4

        gemini_used = bool(gemini_analysis and gemini_analysis.get("source", "gemini") == "gemini")
        if gemini_used:
            conf = float(gemini_analysis.get("confidence_score") or 0.7)
            detected = bool(gemini_analysis.get("tampering_detected"))
            p_gemini = conf if detected else 1.0 - conf
            signals.append({
                "key": "gemini",
                "name": "Gemini multimodal inspection",
                "score": round(p_gemini, 3),
                "weight": 1.0,
                "detail": gemini_analysis.get("forensic_details") or "",
            })
            final_p = 0.65 * p_gemini + 0.35 * local_p
        else:
            final_p = local_p

        verdict = "TAMPERED" if final_p >= self.THRESHOLD else "ORIGINAL"
        confidence = final_p if verdict == "TAMPERED" else 1.0 - final_p

        stages = self._manipulation_stages(signals, gemini_analysis if gemini_used else None)

        overlay_url = None
        if overlay_output_path and suspicious_boxes:
            overlay_url = self._render_overlay(rgb, suspicious_boxes, overlay_output_path)

        for s in signals:
            s["score"] = round(float(s["score"]), 3)

        return {
            "verdict": verdict,
            "tamper_probability": round(final_p, 3),
            "confidence": round(confidence, 3),
            "signals": signals,
            "gemini_used": gemini_used,
            "manipulation_stages": stages,
            "suspicious_regions": len(suspicious_boxes),
            "suspicious_boxes": [
                [round(x0 / float(rgb.shape[1]), 4), round(y0 / float(rgb.shape[0]), 4),
                 round(x1 / float(rgb.shape[1]), 4), round(y1 / float(rgb.shape[0]), 4)]
                for (x0, y0, x1, y1) in searchable_boxes
            ],
            "overlay_path": overlay_url,
        }

    # ------------------------------------------------------------------ loading

    @staticmethod
    def _load_rgb(image_path: str) -> np.ndarray:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = min(1.0, ANALYSIS_MAX_SIDE / max(w, h))
            if scale < 1.0:
                im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            return np.array(im)

    @staticmethod
    def _blocks(arr: np.ndarray, block: int = BLOCK) -> np.ndarray:
        h, w = arr.shape[:2]
        hb, wb = h // block, w // block
        trimmed = arr[: hb * block, : wb * block]
        return trimmed.reshape(hb, block, wb, block).mean(axis=(1, 3))

    @staticmethod
    def _boxes_from_mask(mask: np.ndarray, block: int, min_blocks: int = 3) -> List[Tuple[int, int, int, int]]:
        mask_u8 = mask.astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        boxes = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area >= min_blocks:
                boxes.append((x * block, y * block, (x + w) * block, (y + h) * block))
        return boxes

    # ------------------------------------------------------------------ signals

    def _ela_signal(self, rgb: np.ndarray):
        im = Image.fromarray(rgb)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
        buf.seek(0)
        resaved = Image.open(buf).convert("RGB")
        ela = np.asarray(ImageChops.difference(im, resaved), dtype=np.float32).mean(axis=2)

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        texture = np.abs(cv2.Laplacian(gray, cv2.CV_32F))

        e = self._blocks(ela)
        t = self._blocks(texture)
        if e.size < 16:
            return self._signal("ela", "Error level analysis", 0.0, 0.55, "Image too small for ELA."), []

        # ELA is expected to scale with texture. A pasted or retouched region shifts that
        # relationship for a whole area, so compare region-averaged (5x5 block) ratios.
        valid = t > np.percentile(t, 20)
        ratio = np.log((e + 0.5) / (t + 2.0))
        fill = float(np.median(ratio[valid])) if valid.any() else 0.0
        ratio = np.where(valid, ratio, fill).astype(np.float32)
        smooth = cv2.blur(ratio, (5, 5))
        z = _robust_z(smooth.ravel()).reshape(smooth.shape)
        outliers = (np.abs(z) > 3.5) & valid
        frac = float(outliers.mean())

        mask = cv2.morphologyEx(outliers.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        boxes = self._boxes_from_mask(mask, BLOCK, min_blocks=6)
        frame = float(rgb.shape[0] * rgb.shape[1])
        boxes = [b for b in boxes if (b[2] - b[0]) * (b[3] - b[1]) < 0.5 * frame]
        largest = max(((b[2] - b[0]) * (b[3] - b[1]) for b in boxes), default=0) / frame

        score = min(1.0, largest * 12.0)
        detail = (
            f"{len(boxes)} region(s) whose compression error level is inconsistent with the rest of the image "
            f"(largest covers {largest * 100:.1f}% of the frame)."
            if boxes else "Compression error levels are consistent across the frame."
        )
        return self._signal("ela", "Error level analysis (ELA)", score, 0.55, detail), boxes

    def _noise_signal(self, rgb: np.ndarray):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        residual = gray - cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
        texture = np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (5, 5), 0), cv2.CV_32F))

        n = self._blocks(np.abs(residual), 32)
        t = self._blocks(texture, 32)
        if n.size < 16:
            return self._signal("noise", "Noise consistency", 0.0, 0.35, "Image too small."), []

        ratio = np.log((n + 0.1) / (t + 0.5))
        z = _robust_z(ratio.ravel()).reshape(ratio.shape)
        outliers = np.abs(z) > 5.5
        mask = cv2.morphologyEx(outliers.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        boxes = self._boxes_from_mask(mask, 32, min_blocks=3)
        largest = max(((b[2] - b[0]) * (b[3] - b[1]) for b in boxes), default=0) / float(rgb.shape[0] * rgb.shape[1])
        score = min(1.0, largest * 8.0)
        detail = (
            f"{len(boxes)} region(s) whose sensor-noise level differs from the rest of the image."
            if boxes else "Noise residual is uniform across the frame."
        )
        return self._signal("noise", "Noise residual consistency", score, 0.3, detail), boxes

    def _copy_move_signal(self, rgb: np.ndarray, exclude_boxes=()):
        name = "Copy-move (clone) detection"
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mask = np.full(gray.shape, 255, np.uint8)
        for (x0, y0, x1, y1) in exclude_boxes:
            mask[max(0, y0):y1, max(0, x0):x1] = 0  # repeated glyphs in captions are not clones
        orb = cv2.ORB_create(nfeatures=3000)
        kps, desc = orb.detectAndCompute(gray, mask)
        if desc is None or len(kps) < 20:
            return self._signal("copy_move", name, 0.0, 0.8, "Too few keypoints."), []

        knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc, desc, k=3)
        min_dist = max(gray.shape) * 0.05
        src, dst = [], []
        for m in knn:
            # m[0] is the keypoint itself; m[1] is the nearest *other* keypoint
            if len(m) < 3:
                continue
            best, second = m[1], m[2]
            if best.distance > 40 or best.distance > 0.7 * second.distance:
                continue
            p1 = np.array(kps[best.queryIdx].pt)
            p2 = np.array(kps[best.trainIdx].pt)
            if np.linalg.norm(p1 - p2) < min_dist:
                continue
            src.append(p1)
            dst.append(p2)

        if len(src) < 8:
            return self._signal("copy_move", name, 0.0, 0.8, "No duplicated regions found."), []

        src_a, dst_a = np.float32(src), np.float32(dst)
        _, inlier_mask = cv2.estimateAffinePartial2D(src_a, dst_a, method=cv2.RANSAC, ransacReprojThreshold=4.0)
        if inlier_mask is None:
            return self._signal("copy_move", name, 0.0, 0.8, "No duplicated regions found."), []
        keep = inlier_mask.ravel() == 1
        inliers = int(keep.sum())
        ratio = inliers / float(len(src))
        frame = float(gray.shape[0] * gray.shape[1])
        s_pts, d_pts = src_a[keep], dst_a[keep]
        spread = float(np.ptp(s_pts, axis=0).prod()) / frame if inliers else 0.0

        # A clone is one compact region copied with a single displacement. Repetitive
        # textures (windows, tiles, letters) give many matches with inconsistent geometry.
        is_clone = inliers >= 12 and ratio >= 0.35 and 0.002 <= spread <= 0.08
        if not is_clone:
            return self._signal("copy_move", name, 0.0, 0.8,
                                "No consistent duplicated region found."), []

        score = min(1.0, 0.5 + (inliers - 12) / 20.0)
        boxes = []
        for pts in (s_pts, d_pts):
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            boxes.append((int(x0) - 8, int(y0) - 8, int(x1) + 8, int(y1) + 8))
        detail = f"{inliers} keypoints duplicated with one consistent displacement — a region appears to be cloned."
        return self._signal("copy_move", name, score, 0.8, detail), boxes

    def _overlay_signal(self, rgb: np.ndarray):
        h, w = rgb.shape[:2]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        quant = (rgb // 32).astype(np.int32)
        codes = quant[..., 0] * 64 + quant[..., 1] * 8 + quant[..., 2]
        sat = hsv[..., 1].astype(np.float32) / 255.0
        val = hsv[..., 2].astype(np.float32) / 255.0

        row_flags = np.zeros(h, dtype=bool)
        for y in range(h):
            row = codes[y]
            counts = np.bincount(row, minlength=512)
            dom = int(counts.argmax())
            frac = counts[dom] / float(w)
            is_dom = row == dom
            vivid = float(np.mean(sat[y][is_dom])) > 0.45 or float(np.mean(val[y][is_dom])) < 0.12
            row_flags[y] = frac > 0.5 and vivid

        # Text glyphs break row uniformity inside a banner; bridge short gaps.
        gap = max(3, int(h * 0.015))
        closed = cv2.morphologyEx(row_flags.astype(np.uint8).reshape(-1, 1), cv2.MORPH_CLOSE,
                                  np.ones((gap, 1), np.uint8)).ravel().astype(bool)
        row_flags = closed | row_flags

        boxes = []
        y = 0
        min_band = max(6, int(h * 0.035))
        while y < h:
            if row_flags[y]:
                start = y
                while y < h and row_flags[y]:
                    y += 1
                if y - start >= min_band and self._band_has_hard_edge(rgb, start, y):
                    band = cv2.cvtColor(rgb[start:y], cv2.COLOR_RGB2GRAY)
                    edges = cv2.Canny(band, 80, 200)
                    # Text inside the band produces dense edges even though the band is a solid colour.
                    if edges.mean() / 255.0 > 0.01:
                        boxes.append((0, start, w, y))
            y += 1

        # Text drawn inside a band may break row uniformity; merge bands separated by small gaps.
        score = 0.85 if boxes else 0.0
        detail = (
            f"{len(boxes)} solid-colour banner(s) with text overlaid on the photo."
            if boxes else "No banner or caption overlay detected."
        )
        return self._signal("overlay", "Text / banner overlay", score, 0.75, detail), boxes

    @staticmethod
    def _band_has_hard_edge(rgb: np.ndarray, start: int, end: int) -> bool:
        """A pasted banner has an abrupt, full-width boundary with the photo; sky gradients do not."""
        h = rgb.shape[0]
        img = rgb.astype(np.int16)

        def step(inside: int, outside: int) -> float:
            return float(np.mean(np.abs(img[inside] - img[outside]).sum(axis=1) > 60))

        top = step(start, start - 2) if start >= 2 else None
        bottom = step(end - 1, end + 1) if end + 1 < h else None
        edges = [v for v in (top, bottom) if v is not None]
        return bool(edges) and all(v > 0.6 for v in edges)

    def _metadata_signal(self, image_path: str):
        exif = ExifParser.extract_metadata(image_path)
        c2pa = C2PAVerifier.verify_content_credentials(image_path)
        software = exif.get("editing_software_detected") or ""
        raw = exif.get("raw_tags", {}) or {}
        notes = []
        score = 0.0
        if software:
            score = 0.85
            notes.append(f"Saved by editing software: {software}.")
        elif raw.get("Software"):
            notes.append(f"Software tag: {raw.get('Software')}.")
        dt, dto = raw.get("DateTime"), raw.get("DateTimeOriginal")
        if dt and dto and str(dt) != str(dto):
            score = max(score, 0.35)
            notes.append("File modified after capture (DateTime differs from DateTimeOriginal).")
        if exif.get("make") or exif.get("model"):
            notes.append(f"Camera: {exif.get('make') or ''} {exif.get('model') or ''}".strip() + ".")
        elif not exif.get("has_exif"):
            notes.append("No EXIF metadata (stripped — common for social media re-uploads).")
        signed = bool(c2pa.get("is_signed"))
        if signed:
            notes.append("Valid C2PA content credentials present.")
        return self._signal("metadata", "Metadata & content credentials", score, 0.6,
                            " ".join(notes) or "No notable metadata."), signed

    @staticmethod
    def _signal(key: str, name: str, score: float, weight: float, detail: str) -> Dict[str, Any]:
        return {"key": key, "name": name, "score": float(max(0.0, min(1.0, score))), "weight": weight, "detail": detail}

    # ------------------------------------------------------------------ outputs

    @staticmethod
    def _manipulation_stages(signals: List[Dict[str, Any]], gemini: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
        by_key = {s["key"]: s for s in signals}
        stages: List[Dict[str, str]] = []

        if gemini and gemini.get("manipulation_steps"):
            for step in gemini["manipulation_steps"]:
                if step:
                    stages.append({"label": str(step), "evidence": "Gemini multimodal inspection"})

        def add(key: str, label: str, threshold: float = 0.35):
            s = by_key.get(key)
            if s and s["score"] >= threshold:
                stages.append({"label": label, "evidence": s["detail"]})

        add("metadata", "Opened and re-saved in editing software")
        add("copy_move", "Region cloned within the image (copy-move)")
        add("ela", "Foreign content pasted or retouched (splice)")
        add("noise", "Region with mismatched sensor noise inserted")
        add("overlay", "Text banner / caption overlay added")
        return stages

    @staticmethod
    def _render_overlay(rgb: np.ndarray, boxes: List[Tuple[int, int, int, int]], output_path: str) -> Optional[str]:
        try:
            canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            shade = canvas.copy()
            for (x0, y0, x1, y1) in boxes:
                cv2.rectangle(shade, (x0, y0), (x1, y1), (40, 40, 240), -1)
            canvas = cv2.addWeighted(shade, 0.28, canvas, 0.72, 0)
            for (x0, y0, x1, y1) in boxes:
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (40, 40, 255), 2)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, canvas)
            return output_path
        except Exception as e:
            print(f"[ForensicClassifier] overlay render failed: {e}")
            return None
