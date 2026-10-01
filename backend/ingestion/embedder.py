from __future__ import annotations
import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image
import imagehash
import cv2


class ImageEmbedder:
    """
    Computes invariant perceptual fingerprints for images:
    1. pHash (Perceptual Hash): High-frequency structural fingerprint to detect near-exact copies.
    2. dHash (Difference Hash): Gradient fingerprint capturing edge orientations.
    3. Semantic Color & Spatial Frequency Vector: Invariant representation for heavily filtered / edited relatives.
    """

    @classmethod
    def compute_fingerprint(cls, image_path: str) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        result: Dict[str, Any] = {
            "phash": None,
            "dhash": None,
            "dimensions": None,
            "aspect_ratio": None,
            "feature_vector": [],
        }

        try:
            with Image.open(image_path) as im:
                # 1. pHash
                p_hash = imagehash.phash(im)
                result["phash"] = str(p_hash)

                # 2. dHash
                d_hash = imagehash.dhash(im)
                result["dhash"] = str(d_hash)

                w, h = im.size
                result["dimensions"] = [w, h]
                result["aspect_ratio"] = round(float(w) / max(1, float(h)), 3)

            # 3. Semantic invariant feature vector
            cv_img = cv2.imread(image_path)
            if cv_img is not None:
                result["feature_vector"] = cls._extract_spatial_feature_vector(cv_img)

        except Exception as e:
            result["error"] = str(e)

        return result

    @classmethod
    def _extract_spatial_feature_vector(cls, img: np.ndarray) -> List[float]:
        """
        Extracts a normalized 64-dimensional feature vector combining
        spatial block color moments and edge frequency distributions.
        """
        # Resize to standard 64x64 grid
        resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        # 4x4 spatial blocks
        features = []
        for r in range(4):
            for c in range(4):
                block = resized[r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16]
                mean_bgr = np.mean(block, axis=(0, 1)) / 255.0
                gray_block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
                std_edge = np.std(gray_block) / 128.0
                features.extend([float(mean_bgr[0]), float(mean_bgr[1]), float(mean_bgr[2]), float(std_edge)])

        # Normalize vector
        vec = np.array(features, dtype=float)
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return [round(float(x), 5) for x in vec]

    @staticmethod
    def hamming_similarity(hash1: str, hash2: str, hash_bits: int = 64) -> float:
        """
        Computes normalized Hamming similarity in [0, 1].
        """
        try:
            h1 = imagehash.hex_to_hash(hash1)
            h2 = imagehash.hex_to_hash(hash2)
            dist = h1 - h2
            return max(0.0, 1.0 - (dist / float(hash_bits / 2.0)))
        except Exception:
            return 0.50

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Computes cosine similarity between two feature vectors in [0, 1].
        """
        try:
            v1 = np.array(vec1, dtype=float)
            v2 = np.array(vec2, dtype=float)
            dot = np.dot(v1, v2)
            norm = (np.linalg.norm(v1) * np.linalg.norm(v2))
            if norm < 1e-6:
                return 0.50
            sim = float(dot / norm)
            return max(0.0, min(1.0, (sim + 1.0) / 2.0))
        except Exception:
            return 0.50
