from __future__ import annotations
import os
from typing import List, Dict, Any, Optional
import cv2
import numpy as np


class VideoKeyframeExtractor:
    """
    Extracts representative scene-transition keyframes from video files using OpenCV.
    Detects shot boundaries and scene transitions via color histogram correlation
    and frame difference analysis.
    """

    def __init__(
        self,
        scene_threshold: float = 0.55,
        max_keyframes: int = 6,
        sample_rate: int = 5,  # Sample every N frames for speed
    ):
        self.scene_threshold = scene_threshold
        self.max_keyframes = max_keyframes
        self.sample_rate = sample_rate

    def extract_keyframes(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Processes a video file and saves keyframe images to output_dir.
        Returns a list of extracted keyframe descriptors.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(video_path),
                f"{os.path.splitext(os.path.basename(video_path))[0]}_keyframes"
            )
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        prev_hist = None
        frame_idx = 0
        candidate_keyframes = []

        # Always capture the initial frame (frame 0) as baseline
        ret, first_frame = cap.read()
        if ret and first_frame is not None:
            first_path = os.path.join(output_dir, "keyframe_0000.png")
            cv2.imwrite(first_path, first_frame)
            prev_hist = self._compute_frame_hist(first_frame)
            candidate_keyframes.append({
                "frame_index": 0,
                "timestamp_sec": 0.0,
                "image_path": first_path,
                "transition_score": 1.0,
                "dimensions": [first_frame.shape[1], first_frame.shape[0]],
            })

        while cap.isOpened():
            ret, frame = cap.read()
            frame_idx += 1
            if not ret:
                break

            # Skip frames to accelerate processing
            if frame_idx % self.sample_rate != 0:
                continue

            curr_hist = self._compute_frame_hist(frame)
            if prev_hist is not None:
                # Compare histogram correlation (1.0 = identical, < 0.5 = major scene change)
                correlation = float(cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL))
                transition_score = 1.0 - max(0.0, correlation)

                if correlation < self.scene_threshold:
                    # Significant scene transition detected
                    timestamp_sec = round(frame_idx / fps, 2)
                    kf_path = os.path.join(output_dir, f"keyframe_{frame_idx:04d}.png")
                    cv2.imwrite(kf_path, frame)

                    candidate_keyframes.append({
                        "frame_index": frame_idx,
                        "timestamp_sec": timestamp_sec,
                        "image_path": kf_path,
                        "transition_score": round(transition_score, 4),
                        "dimensions": [frame.shape[1], frame.shape[0]],
                    })

                    prev_hist = curr_hist

                    if len(candidate_keyframes) >= self.max_keyframes:
                        break

        cap.release()
        return candidate_keyframes

    def _compute_frame_hist(self, frame: np.ndarray) -> np.ndarray:
        """Computes 3D normalized color histogram for scene comparison."""
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist
