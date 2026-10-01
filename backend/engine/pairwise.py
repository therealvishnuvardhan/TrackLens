from __future__ import annotations
import math
from typing import Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image
import imagehash
import cv2

from backend.engine.models import (
    ImageVersion,
    EditDirectionMetrics,
    EdgeScore,
)


class PairwiseEngine:
    """
    Computes directed relationship score S(A -> B), representing the likelihood
    that version B was directly derived or copied from version A.
    
    Weights (from TraceLens spec):
      - Visual Match: 35%
      - Time Order: 20%
      - Edit Direction: 15%
      - Metadata / C2PA: 15%
      - Source Trust: 15%
    """

    def __init__(
        self,
        weight_vis: float = 0.35,
        weight_time: float = 0.20,
        weight_edit: float = 0.15,
        weight_meta: float = 0.15,
        weight_trust: float = 0.15,
    ):
        self.w_vis = weight_vis
        self.w_time = weight_time
        self.w_edit = weight_edit
        self.w_meta = weight_meta
        self.w_trust = weight_trust

    def compute_edge_score(
        self,
        source: ImageVersion,
        target: ImageVersion,
        source_img_cache: Optional[np.ndarray] = None,
        target_img_cache: Optional[np.ndarray] = None,
    ) -> EdgeScore:
        if source.id == target.id:
            return EdgeScore(source_id=source.id, target_id=target.id, total_score=0.0)

        # Preload OpenCV images
        img_a = source_img_cache if source_img_cache is not None else cv2.imread(source.image_path)
        img_b = target_img_cache if target_img_cache is not None else cv2.imread(target.image_path)

        # 1. Visual Match Score (35%)
        vis_score = self._compute_visual_similarity(source, target, img_a, img_b)

        # Fast prune: if images are completely unrelated visually, they cannot be related
        if vis_score < 0.25:
            return EdgeScore(
                source_id=source.id,
                target_id=target.id,
                visual_score=round(vis_score, 4),
                total_score=0.0,
            )

        # 2. Time Order Score (20%)
        time_score = self._compute_time_order(source, target)

        # 3. Edit Direction Score (15%)
        edit_metrics, edit_score = self._compute_edit_direction(
            source, target, img_a, img_b
        )

        # 4. Metadata / C2PA Score (15%)
        meta_score = self._compute_metadata_score(source, target, vis_score)

        # 5. Source Trust Score (15%)
        trust_score = self._compute_trust_score(source)

        # Total Weighted Score
        total = (
            self.w_vis * vis_score
            + self.w_time * time_score
            + self.w_edit * edit_score
            + self.w_meta * meta_score
            + self.w_trust * trust_score
        )
        total = max(0.0, min(1.0, total))

        return EdgeScore(
            source_id=source.id,
            target_id=target.id,
            visual_score=round(vis_score, 4),
            time_score=round(time_score, 4),
            edit_direction_score=round(edit_score, 4),
            metadata_score=round(meta_score, 4),
            trust_score=round(trust_score, 4),
            total_score=round(total, 4),
            edit_metrics=edit_metrics,
        )

    def _compute_visual_similarity(
        self,
        source: ImageVersion,
        target: ImageVersion,
        img_a: Optional[np.ndarray],
        img_b: Optional[np.ndarray],
    ) -> float:
        """
        Combines pHash Hamming similarity (broad structure) and structural pixel similarity (exact match).
        """
        try:
            # 1. pHash similarity
            if not source.phash:
                with Image.open(source.image_path) as im:
                    source.phash = str(imagehash.phash(im))
            if not target.phash:
                with Image.open(target.image_path) as im:
                    target.phash = str(imagehash.phash(im))

            h1 = imagehash.hex_to_hash(source.phash)
            h2 = imagehash.hex_to_hash(target.phash)
            hamming_dist = h1 - h2
            phash_sim = max(0.0, 1.0 - (hamming_dist / 32.0))

            # 2. Structural pixel similarity, crop alignment & color histogram
            pixel_sim = phash_sim
            crop_sim = 0.0
            color_sim = 0.5
            if img_a is not None and img_b is not None:
                # 3D Color Histogram correlation
                h_a = cv2.calcHist([img_a], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                cv2.normalize(h_a, h_a)
                h_b = cv2.calcHist([img_b], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                cv2.normalize(h_b, h_b)
                raw_color_corr = float(cv2.compareHist(h_a, h_b, cv2.HISTCMP_CORREL))
                color_sim = max(0.0, min(1.0, (raw_color_corr + 0.2) / 1.2))

                g_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
                g_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
                s_a = cv2.resize(g_a, (256, 256), interpolation=cv2.INTER_AREA)
                s_b = cv2.resize(g_b, (256, 256), interpolation=cv2.INTER_AREA)

                # Normalized mean absolute error
                mae = float(np.mean(np.abs(s_a.astype(float) - s_b.astype(float))) / 255.0)
                pixel_sim = max(0.0, 1.0 - (mae / 0.15))

                # Check if one is a crop of another
                _, conf = self._check_crop_containment(g_a, g_b)
                crop_sim = float(conf)

            # Strongest visual correlation tempered by color filter consistency
            structural_match = max(crop_sim, 0.40 * phash_sim + 0.60 * pixel_sim)
            if structural_match < 0.25:
                vis_score = structural_match
            else:
                vis_score = 0.60 * structural_match + 0.40 * color_sim
            return float(min(1.0, vis_score))
        except Exception:
            return 0.50

    def _compute_time_order(self, source: ImageVersion, target: ImageVersion) -> float:
        """
        Calculates chronological feasibility.
        A copy cannot predate its source.
        """
        if not source.timestamp or not target.timestamp:
            return 0.50

        if source.timestamp < target.timestamp:
            delta_days = (target.timestamp - source.timestamp).total_seconds() / 86400.0
            # Decay for time hops: immediate hops (1-5 days) receive full confidence (1.0)
            decay = math.exp(-delta_days / 120.0)
            return float(0.85 + 0.15 * decay)
        elif source.timestamp == target.timestamp:
            return 0.50
        else:
            # Reverse time order is heavily penalized
            return 0.05

    def _compute_edit_direction(
        self,
        source: ImageVersion,
        target: ImageVersion,
        img_a: Optional[np.ndarray],
        img_b: Optional[np.ndarray],
    ) -> Tuple[EditDirectionMetrics, float]:
        """
        Analyzes directional asymmetry:
        - Crop containment (B fits in A => A is parent)
        - Resolution drop (downscaling indicates derivation)
        - Additions & localized manipulations
        """
        if img_a is None or img_b is None:
            return EditDirectionMetrics(), 0.50

        ha, wa = img_a.shape[:2]
        hb, wb = img_b.shape[:2]

        area_a = float(wa * ha)
        area_b = float(wb * hb)
        res_ratio = area_b / max(1.0, area_a)

        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

        # 1. Sharpness ratio
        var_a = cv2.Laplacian(gray_a, cv2.CV_64F).var()
        var_b = cv2.Laplacian(gray_b, cv2.CV_64F).var()
        sharpness_ratio = float(var_b / max(1e-5, var_a))

        # 2. Crop containment
        containment, conf = self._check_crop_containment(gray_a, gray_b)

        # 3. Additions / Localized Alterations
        addition_score = self._detect_additions(gray_a, gray_b)

        # 4. Check for localized modification on top of shared base
        # (e.g. WhatsApp to Reddit: same banner, but Reddit has localized face distortion)
        localized_edit = self._detect_localized_alteration(gray_a, gray_b)

        # Direction score computation
        dir_score = 0.65  # Base favorable prior if time order already holds

        if containment > 0.5:
            dir_score += 0.20 * conf
        elif containment < -0.5:
            dir_score -= 0.35 * conf

        if res_ratio < 0.95:
            dir_score += 0.10
        elif res_ratio > 1.05:
            dir_score -= 0.15

        if addition_score > 0.20:
            dir_score += 0.15

        if localized_edit > 0.5:
            # Source has clean base, target has localized alteration
            dir_score += 0.15

        dir_score = max(0.05, min(0.98, dir_score))

        metrics = EditDirectionMetrics(
            is_crop_containment=containment,
            containment_confidence=round(conf, 4),
            resolution_ratio=round(res_ratio, 4),
            sharpness_ratio=round(sharpness_ratio, 4),
            addition_mask_score=round(addition_score, 4),
            direction_score=round(dir_score, 4),
        )
        return metrics, dir_score

    def _check_crop_containment(
        self, gray_a: np.ndarray, gray_b: np.ndarray
    ) -> Tuple[float, float]:
        ha, wa = gray_a.shape[:2]
        hb, wb = gray_b.shape[:2]

        if wb <= wa and hb <= ha:
            score = self._template_match_score(gray_a, gray_b)
            if score > 0.75:
                return 1.0, score

        scales = [0.4, 0.6, 0.8, 1.0]
        for s in scales:
            target_w = int(wa * s)
            if target_w < 50:
                continue
            aspect = float(hb) / max(1.0, float(wb))
            target_h = int(target_w * aspect)
            if target_h <= ha and target_w <= wa:
                resized_b = cv2.resize(gray_b, (target_w, target_h), interpolation=cv2.INTER_AREA)
                score = self._template_match_score(gray_a, resized_b)
                if score > 0.75:
                    return 1.0, score

        if wa <= wb and ha <= hb:
            rev_score = self._template_match_score(gray_b, gray_a)
            if rev_score > 0.75:
                return -1.0, rev_score

        return 0.0, 0.0

    def _template_match_score(self, large_img: np.ndarray, small_img: np.ndarray) -> float:
        try:
            if small_img.shape[0] >= large_img.shape[0] and small_img.shape[1] >= large_img.shape[1]:
                res = cv2.matchTemplate(large_img, small_img, cv2.TM_CCOEFF_NORMED)
                return float(res[0][0])
            res = cv2.matchTemplate(large_img, small_img, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            return float(max_val)
        except Exception:
            return 0.0

    def _detect_additions(self, gray_a: np.ndarray, gray_b: np.ndarray) -> float:
        """
        Detects if image B has added text overlays, banners, or stickers absent in A.
        """
        try:
            sobel_b = cv2.Sobel(gray_b, cv2.CV_64F, 0, 1, ksize=3)
            edge_energy_b = np.mean(np.abs(sobel_b))

            resized_a = cv2.resize(gray_a, (gray_b.shape[1], gray_b.shape[0]))
            sobel_a = cv2.Sobel(resized_a, cv2.CV_64F, 0, 1, ksize=3)
            edge_energy_a = np.mean(np.abs(sobel_a))

            diff_ratio = (edge_energy_b - edge_energy_a) / max(1.0, edge_energy_a)
            if diff_ratio > 0.15:
                return float(min(1.0, diff_ratio))
            return 0.0
        except Exception:
            return 0.0

    def _detect_localized_alteration(self, gray_a: np.ndarray, gray_b: np.ndarray) -> float:
        """
        Checks if B is identical to A across most of the frame except for a localized cluster
        (e.g., face edit / AI tampering on an already existing meme image).
        """
        try:
            if gray_a.shape != gray_b.shape:
                resized_a = cv2.resize(gray_a, (gray_b.shape[1], gray_b.shape[0]))
            else:
                resized_a = gray_a

            diff = np.abs(resized_a.astype(float) - gray_b.astype(float))
            # Fraction of pixels that have substantial change (> 30 intensity)
            altered_fraction = float(np.mean(diff > 30))
            # If 0.1% to 15% of pixels are altered while the rest are identical,
            # this represents a classic localized edit / face swap on a parent template.
            if 0.001 <= altered_fraction <= 0.15:
                return 1.0
            return 0.0
        except Exception:
            return 0.0

    def _compute_metadata_score(
        self, source: ImageVersion, target: ImageVersion, visual_sim: float = 1.0
    ) -> float:
        """
        C2PA cryptographic assertions and EXIF camera indicators.
        Camera EXIF supports direct export to immediate children (high visual sim),
        but does not indicate a direct link to distant derivative generations.
        """
        if source.c2pa and source.c2pa.get("is_signed"):
            if not target.c2pa or not target.c2pa.get("is_signed"):
                return float(0.50 + 0.45 * min(1.0, visual_sim))

        has_cam_a = bool(source.exif and source.exif.get("Make"))
        has_cam_b = bool(target.exif and target.exif.get("Make"))

        if has_cam_a and not has_cam_b:
            return float(0.50 + 0.40 * min(1.0, visual_sim))
        elif not has_cam_a and has_cam_b:
            return 0.10
        else:
            return 0.50

    def _compute_trust_score(self, source: ImageVersion) -> float:
        """
        Trust score of the source. Credible sources (photographer portfolio, news) score high (0.90-0.95).
        For social platforms, trust provides a stable baseline so downstream derivation can still form.
        """
        base = float(source.source_trust_score)
        return float(0.60 + 0.35 * base)
