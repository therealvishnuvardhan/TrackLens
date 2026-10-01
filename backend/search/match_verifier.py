from __future__ import annotations
import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import imagehash
import numpy as np
import requests
from PIL import Image

from backend.search.page_scraper import BROWSER_UA

MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024
STORE_MAX_SIDE = 1000
COMPARE_MAX_SIDE = 640


class MatchVerifier:
    """
    Downloads reverse-search candidates and verifies them against the query image:

      - pHash similarity (near-duplicate detection)
      - ORB feature matching + RANSAC homography (robust to crops, resizes, overlays)
      - Aligned pixel difference: after warping the candidate onto the query, how much
        of the overlapping frame differs. A near-identical copy that differs in one region
        is strong evidence that one of the two was edited.
    """

    def __init__(self, timeout: float = 6.0, workers: int = 16):
        self.timeout = timeout
        self.workers = workers
        self._orb = None

    # ------------------------------------------------------------------ public

    def verify_all(
        self,
        query_path: str,
        candidates: List[Dict[str, Any]],
        out_dir: str,
        on_progress: Optional[Callable[[int, int], None]] = None,
        fetch_full: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Screens every candidate using its thumbnail (fast). With fetch_full, real matches
        are re-fetched at full resolution (needed for true dimensions and lineage analysis);
        otherwise call upgrade_full() later.
        """
        os.makedirs(out_dir, exist_ok=True)
        query = self._prepare(self._load_rgb_from_path(query_path))

        def work(idx_cand: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
            idx, cand = idx_cand
            result = dict(cand)
            result.update({"verified": False, "local_path": None, "similarity": 0.0,
                           "full_resolution": False})
            # Screen with the thumbnail only; the full image is fetched later for real matches.
            img, used_url = self._download_first(cand.get("thumb_url") or cand.get("image_url"))
            if img is None:
                result["error"] = "download failed"
                return result
            result.update(self._compare(query, self._prepare(np.array(img))))
            self._store(result, img, used_url, os.path.join(out_dir, f"match_{idx:02d}.jpg"))
            result["verified"] = True
            if fetch_full and result["related"]:
                self._upgrade_one(query, result)
            return result

        results = self._run_parallel(work, list(enumerate(candidates)), on_progress)
        results.sort(key=lambda r: r.get("similarity", 0.0), reverse=True)
        return results

    def upgrade_full(
        self,
        query_path: str,
        results: List[Dict[str, Any]],
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Downloads the full-resolution image for already-screened matches and re-compares."""
        query = self._prepare(self._load_rgb_from_path(query_path))
        return self._run_parallel(lambda r: self._upgrade_one(query, r), results, on_progress)

    def _upgrade_one(self, query: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        if result.get("full_resolution") or not result.get("image_url"):
            return result
        if result.get("downloaded_from") == result.get("image_url"):
            result["full_resolution"] = True
            return result
        full, full_url = self._download_first(result.get("image_url"))
        thumb_side = max(result.get("dimensions") or [0])
        if full is not None and max(full.size) > thumb_side:
            comparison = self._compare(query, self._prepare(np.array(full)))
            # Keep the full image only if it is still the same picture (pages can serve other images)
            if comparison["similarity"] >= result["similarity"] - 0.15:
                result.update(comparison)
                self._store(result, full, full_url, result["local_path"])
                result["full_resolution"] = True
        return result

    def _store(self, result: Dict[str, Any], img: Image.Image, url: Optional[str], local_path: str):
        stored = img.copy()
        stored.thumbnail((STORE_MAX_SIDE, STORE_MAX_SIDE))
        stored.save(local_path, "JPEG", quality=90)
        result["local_path"] = local_path
        result["downloaded_from"] = url
        result["dimensions"] = list(img.size)
        if result.get("reported_dimensions") and url != result.get("image_url"):
            # thumbnail stands in for an image whose real size the engine reported
            result["dimensions"] = list(result["reported_dimensions"])

    def _run_parallel(self, fn, items: list, on_progress) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(fn, item) for item in items]
            for fut in as_completed(futures):
                results.append(fut.result())
                if on_progress:
                    on_progress(len(results), len(items))
        return results

    def compare_paths(self, path_a: str, path_b: str) -> Dict[str, Any]:
        a = self._prepare(self._load_rgb_from_path(path_a))
        b = self._prepare(self._load_rgb_from_path(path_b))
        return self._compare(a, b)

    def render_difference(self, edited_path: str, reference_path: str, out_path: str) -> Optional[str]:
        """
        Aligns reference onto the edited image and writes a visual where regions that differ
        are outlined in red. Returns out_path, or None if the images could not be aligned.
        """
        q_rgb = self._load_rgb_from_path(edited_path)
        q = self._prepare(q_rgb)
        c = self._prepare(self._load_rgb_from_path(reference_path))
        H = self._homography(c, q)
        if H is None:
            return None
        h, w = q["lab"].shape[:2]
        mask = self._difference_mask(q["lab"], cv2.warpPerspective(c["lab"], H, (w, h)), H, c["lab"].shape[:2])
        mask = cv2.dilate(mask.astype(np.uint8) * 255, np.ones((9, 9), np.uint8))
        canvas = cv2.cvtColor(cv2.resize(q_rgb, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
        dimmed = (canvas * 0.45).astype(np.uint8)
        out = np.where(mask[..., None] > 0, canvas, dimmed)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (40, 40, 255), 2)
        cv2.imwrite(out_path, out)
        return out_path

    def difference_boxes(self, edited_path: str, reference_path: str, min_area: float = 0.004) -> List[List[float]]:
        """
        Regions (normalised x0, y0, x1, y1 in the edited image) whose content differs from the
        aligned reference. Small specks are dropped; nearby pieces are merged.
        """
        q = self._prepare(self._load_rgb_from_path(edited_path))
        c = self._prepare(self._load_rgb_from_path(reference_path))
        H = self._homography(c, q)
        if H is None:
            return []
        h, w = q["lab"].shape[:2]
        mask = self._difference_mask(q["lab"], cv2.warpPerspective(c["lab"], H, (w, h)), H, c["lab"].shape[:2])
        mask = cv2.dilate(mask.astype(np.uint8) * 255, np.ones((15, 15), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        boxes = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area / float(w * h) >= min_area:
                boxes.append([x / w, y / h, (x + bw) / w, (y + bh) / h])
        return sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)

    def _homography(self, c: Dict[str, Any], q: Dict[str, Any]) -> Optional[np.ndarray]:
        if q["desc"] is None or c["desc"] is None or len(q["kps"]) < 8 or len(c["kps"]) < 8:
            return None
        good = []
        for pair in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(c["desc"], q["desc"], k=2):
            if len(pair) == 2 and pair[0].distance < 0.78 * pair[1].distance:
                good.append(pair[0])
        if len(good) < 15:
            return None
        src = np.float32([c["kps"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([q["kps"][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None or mask is None or int(mask.sum()) < 15 or not self._sane_homography(H):
            return None
        return H

    # ------------------------------------------------------------------ download

    def _download_first(self, *urls: Optional[str]) -> Tuple[Optional[Image.Image], Optional[str]]:
        for url in urls:
            if not url or not url.startswith("http"):
                continue
            try:
                resp = requests.get(
                    url, timeout=self.timeout, stream=True,
                    headers={"User-Agent": BROWSER_UA, "Referer": "https://www.bing.com/"},
                )
                if resp.status_code != 200:
                    continue
                data = resp.raw.read(MAX_DOWNLOAD_BYTES + 1, decode_content=True)
                resp.close()
                if not data or len(data) > MAX_DOWNLOAD_BYTES:
                    continue
                img = Image.open(io.BytesIO(data))
                img.draft("RGB", (2048, 2048))  # fast JPEG downscale-on-decode
                img.load()
                img = img.convert("RGB")
                if min(img.size) < 48:
                    continue
                return img, url
            except Exception:
                continue
        return None, None

    # ------------------------------------------------------------------ comparison

    @staticmethod
    def _load_rgb_from_path(path: str) -> np.ndarray:
        with Image.open(path) as im:
            return np.array(im.convert("RGB"))

    def _prepare(self, rgb: np.ndarray) -> Dict[str, Any]:
        h, w = rgb.shape[:2]
        scale = min(1.0, COMPARE_MAX_SIDE / float(max(h, w)))
        small = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
        orb = cv2.ORB_create(nfeatures=1500)
        kps, desc = orb.detectAndCompute(gray, None)
        return {
            "orig_size": (w, h),
            "gray": gray,
            "lab": lab,
            "kps": kps,
            "desc": desc,
            "phash": imagehash.phash(Image.fromarray(small)),
        }

    def _compare(self, q: Dict[str, Any], c: Dict[str, Any]) -> Dict[str, Any]:
        phash_dist = int(q["phash"] - c["phash"])
        phash_sim = max(0.0, 1.0 - phash_dist / 32.0)

        inliers, homography = 0, None
        if q["desc"] is not None and c["desc"] is not None and len(q["kps"]) >= 8 and len(c["kps"]) >= 8:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            good = []
            for pair in matcher.knnMatch(c["desc"], q["desc"], k=2):
                if len(pair) == 2 and pair[0].distance < 0.78 * pair[1].distance:
                    good.append(pair[0])
            if len(good) >= 8:
                src = np.float32([c["kps"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst = np.float32([q["kps"][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                if mask is not None:
                    inliers = int(mask.sum())
                if homography is not None and not self._sane_homography(homography):
                    homography, inliers = None, min(inliers, 6)

        orb_score = min(1.0, max(0.0, (inliers - 8) / 40.0))
        similarity = max(phash_sim if phash_dist <= 12 else phash_sim * 0.6, orb_score)

        diff_fraction, overlap, scale_ratio = None, None, None
        if homography is not None and inliers >= 15:
            diff_fraction, overlap = self._aligned_difference(q["lab"], c["lab"], homography)
            scale_ratio = (c["orig_size"][0] * c["orig_size"][1]) / float(q["orig_size"][0] * q["orig_size"][1])

        qa = q["orig_size"][0] / float(q["orig_size"][1])
        ca = c["orig_size"][0] / float(c["orig_size"][1])
        same_aspect = abs(qa - ca) / qa < 0.04

        if similarity >= 0.85 and same_aspect and (diff_fraction is None or diff_fraction < 0.003):
            match_type = "FULL_MATCH"
        elif similarity >= 0.55 and diff_fraction is not None and diff_fraction >= 0.003:
            match_type = "DERIVATIVE"
        elif similarity >= 0.55:
            match_type = "PARTIAL_CROP" if not same_aspect else "FULL_MATCH"
        else:
            match_type = "VISUALLY_SIMILAR"

        return {
            "similarity": round(float(similarity), 4),
            "phash_distance": phash_dist,
            "feature_inliers": inliers,
            "diff_fraction": None if diff_fraction is None else round(float(diff_fraction), 4),
            "overlap_fraction": None if overlap is None else round(float(overlap), 4),
            "scale_ratio": None if scale_ratio is None else round(float(scale_ratio), 3),
            "same_aspect": same_aspect,
            "match_type": match_type,
            "related": similarity >= 0.55,
        }

    @staticmethod
    def _sane_homography(H: np.ndarray) -> bool:
        # Reject degenerate warps (extreme perspective or mirror) from repetitive textures
        det = float(np.linalg.det(H[:2, :2]))
        return 0.05 < det < 20.0 and abs(H[2, 0]) < 0.002 and abs(H[2, 1]) < 0.002

    @classmethod
    def _aligned_difference(cls, q_lab: np.ndarray, c_lab: np.ndarray, H: np.ndarray) -> Tuple[float, float]:
        h, w = q_lab.shape[:2]
        warped = cv2.warpPerspective(c_lab, H, (w, h))
        diff, valid = cls._difference_mask(q_lab, warped, H, c_lab.shape[:2], return_valid=True)
        overlap = float(valid.mean())
        if overlap < 0.05:
            return 0.0, overlap
        return float(diff.sum()) / float(valid.sum()), overlap

    @staticmethod
    def _difference_mask(q_lab: np.ndarray, warped: np.ndarray, H: np.ndarray, ref_shape, return_valid: bool = False):
        h, w = q_lab.shape[:2]
        valid = cv2.warpPerspective(np.full(ref_shape, 255, np.uint8), H, (w, h)) > 0
        valid = cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        diff = np.zeros((h, w), bool)
        if valid.mean() >= 0.05:
            a = cv2.GaussianBlur(q_lab, (5, 5), 0).astype(np.float32)
            b = cv2.GaussianBlur(warped, (5, 5), 0).astype(np.float32)
            # Normalise each channel so global colour filters / exposure changes are not
            # counted as content edits. Robust (median/IQR) statistics keep an edited
            # region from skewing the normalisation.
            for ch in range(3):
                av, bv = a[..., ch][valid], b[..., ch][valid]
                a_q1, a_med, a_q3 = np.percentile(av, [25, 50, 75])
                b_q1, b_med, b_q3 = np.percentile(bv, [25, 50, 75])
                scale = (a_q3 - a_q1) / (b_q3 - b_q1) if (b_q3 - b_q1) > 2 else 1.0
                b[..., ch] = (b[..., ch] - b_med) * float(np.clip(scale, 0.5, 2.0)) + a_med
            lum = np.abs(a[..., 0] - b[..., 0]) > 32
            chroma = np.hypot(a[..., 1] - b[..., 1], a[..., 2] - b[..., 2]) > 22
            diff = (lum | chroma) & valid
            diff = cv2.morphologyEx(diff.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
        return (diff, valid) if return_valid else diff
