from __future__ import annotations
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from backend.search.match_verifier import MatchVerifier
from backend.search.reverse_search import ReverseImageSearch

FACE_MODEL = Path(__file__).resolve().parents[1] / "models_data" / "face_detection_yunet_2023mar.onnx"
MAX_REGIONS = 3
MIN_CROP_PX = 96
# A pasted region is the *same pixels* as its source, so require a geometric (homography)
# match — other photos of the same person or place do not pass this.
SOURCE_MIN_SIMILARITY = 0.7
SOURCE_MIN_INLIERS = 20

PRIORITY = {"changed_vs_copy": 0, "model": 1, "forensic": 2, "face": 3}
LABELS = {
    "changed_vs_copy": "Region that differs from the copy online",
    "model": "Region flagged by the forensic model",
    "forensic": "Region with inconsistent compression/noise",
    "face": "Face",
}


def _iou(a: List[float], b: List[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class RegionSourceSearch:
    """
    Composite fakes are rarely created from nothing: a head, an object or a sky is pasted
    in from another photo. Searching the whole image only finds copies of the composite;
    searching each suspicious region separately finds where the pasted content came from.

    Region proposals (best first): areas that differ from the closest copy online, areas
    the forensic model localises, local forensic outliers, and faces.
    """

    def __init__(self, searcher: Optional[ReverseImageSearch] = None, verifier: Optional[MatchVerifier] = None):
        self.searcher = searcher or ReverseImageSearch()
        self.verifier = verifier or MatchVerifier()
        self._face_model = str(FACE_MODEL)

    # ------------------------------------------------------------------ proposals

    def detect_faces(self, image_path: str) -> List[List[float]]:
        """YuNet face detector (OpenCV model zoo); returns normalised boxes."""
        if not Path(self._face_model).exists():
            return []
        with Image.open(image_path) as im:
            rgb = np.asarray(im.convert("RGB"))
        h, w = rgb.shape[:2]
        scale = min(1.0, 1024 / float(max(h, w)))
        small = cv2.resize(rgb, (int(w * scale), int(h * scale))) if scale < 1 else rgb
        sh, sw = small.shape[:2]
        detector = cv2.FaceDetectorYN.create(self._face_model, "", (sw, sh), 0.75, 0.3, 50)
        _, faces = detector.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        boxes = []
        for f in faces if faces is not None else []:
            x, y, fw, fh = [float(v) for v in f[:4]]
            if fw * fh / float(sw * sh) < 0.002:
                continue
            boxes.append([max(0.0, x / sw), max(0.0, y / sh), min(1.0, (x + fw) / sw), min(1.0, (y + fh) / sh)])
        return boxes

    @staticmethod
    def build_proposals(candidates: List[Tuple[str, List[float]]]) -> List[Dict[str, Any]]:
        """Pads boxes for context, drops duplicates, keeps the most informative few."""
        padded: List[Dict[str, Any]] = []
        for kind, (x0, y0, x1, y1) in sorted(candidates, key=lambda c: PRIORITY.get(c[0], 9)):
            bw, bh = x1 - x0, y1 - y0
            if bw <= 0 or bh <= 0 or bw * bh > 0.7:  # nearly the whole frame: whole-image search covers it
                continue
            pad = 0.35 if kind == "face" else 0.2
            box = [max(0.0, x0 - bw * pad), max(0.0, y0 - bh * pad), min(1.0, x1 + bw * pad), min(1.0, y1 + bh * pad)]
            if any(_iou(box, p["box"]) > 0.4 for p in padded):
                continue
            padded.append({"kind": kind, "label": LABELS.get(kind, kind), "box": [round(v, 4) for v in box]})
            if len(padded) >= MAX_REGIONS:
                break
        return padded

    # ------------------------------------------------------------------ search

    def search_regions(
        self,
        image_path: str,
        proposals: List[Dict[str, Any]],
        out_dir: str,
        whole_image_match_urls: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        os.makedirs(out_dir, exist_ok=True)
        with Image.open(image_path) as im:
            img = im.convert("RGB")
        W, H = img.size
        jobs = []
        for i, p in enumerate(proposals):
            x0, y0, x1, y1 = p["box"]
            crop = img.crop((int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)))
            if min(crop.size) < MIN_CROP_PX:
                scale = MIN_CROP_PX / float(min(crop.size))
                crop = crop.resize((int(crop.size[0] * scale), int(crop.size[1] * scale)), Image.LANCZOS)
            crop_path = os.path.join(out_dir, f"region_{i}.png")
            crop.save(crop_path)
            jobs.append((i, p, crop_path))

        def run(job):
            i, p, crop_path = job
            return self._search_one(image_path, p, crop_path, os.path.join(out_dir, f"region_{i}"),
                                    whole_image_match_urls or set())

        with ThreadPoolExecutor(max_workers=MAX_REGIONS) as pool:
            return list(pool.map(run, jobs))

    def _search_one(self, image_path: str, proposal: Dict[str, Any], crop_path: str, out_dir: str,
                    known_copies: set) -> Dict[str, Any]:
        region = {**proposal, "crop_path": crop_path, "sources": [], "searched": 0, "error": None}
        try:
            found = self.searcher.search(crop_path, use_keyword=False, max_results=30)
        except Exception as e:
            region["error"] = str(e)
            return region
        candidates = [r for r in found["results"] if (r.get("image_url") or r.get("page_url")) not in known_copies]
        region["searched"] = len(candidates)
        region["engine_errors"] = found.get("errors", [])
        if not candidates:
            return region

        verified = self.verifier.verify_all(crop_path, candidates, out_dir, fetch_full=False)
        sources = []
        for m in verified:
            if not m.get("verified"):
                continue
            if m.get("similarity", 0) < SOURCE_MIN_SIMILARITY or (m.get("feature_inliers") or 0) < SOURCE_MIN_INLIERS:
                continue
            # A copy of the whole composite also contains this region; that is not a source.
            whole = self.verifier.compare_paths(m["local_path"], image_path)
            if whole.get("related"):
                continue
            m["whole_image_similarity"] = whole.get("similarity")
            sources.append(m)
        # Clearest sources first: least like the whole image, most matching detail in the region
        sources.sort(key=lambda m: (round(m.get("whole_image_similarity") or 0, 1), -(m.get("feature_inliers") or 0)))
        region["sources"] = sources[:4]
        return region
