from __future__ import annotations
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image

TRUFOR_DIR = Path(__file__).resolve().parent / "trufor"
WEIGHTS = Path(__file__).resolve().parents[1] / "models_data" / "weights" / "trufor.pth.tar"
MAX_SIDE = 1536  # bounds GPU memory / CPU time; TruFor is fully convolutional


class TruForDetector:
    """
    Learned forgery detection & localization (TruFor, CVPR 2023, GRIP-UNINA).

    Combines the RGB image with a learned camera-noise fingerprint (Noiseprint++) and
    returns, without any reference image:
      - score: probability that the image was manipulated (0..1)
      - map:   per-pixel localization of the manipulated region
      - conf:  reliability of the localization map

    The model is loaded lazily on first use and runs on CUDA when available.
    License: GRIP-UNINA research licence — informational, non-profit use only.
    """

    _lock = threading.Lock()

    def __init__(self):
        self._model = None
        self._device = None
        self.error: Optional[str] = None

    @property
    def available(self) -> bool:
        if not WEIGHTS.exists():
            self.error = f"weights not found at {WEIGHTS}"
            return False
        try:
            import torch  # noqa: F401
            import timm  # noqa: F401
            import yacs  # noqa: F401
        except ImportError as e:
            self.error = f"missing dependency: {e.name}"
            return False
        return True

    def _load(self):
        if self._model is not None:
            return
        import torch
        from yacs.config import CfgNode
        from backend.forensics.trufor.config import _C as base_config
        from backend.forensics.trufor.models.cmx.builder_np_conf import myEncoderDecoder

        cfg: CfgNode = base_config.clone()
        cfg.defrost()
        cfg.merge_from_file(str(TRUFOR_DIR / "trufor.yaml"))
        cfg.TEST.MODEL_FILE = str(WEIGHTS)
        cfg.freeze()

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(str(WEIGHTS), map_location=self._device, weights_only=False)
        model = myEncoderDecoder(cfg=cfg)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self._device).eval()
        self._model = model

    def analyze(self, image_path: str, out_dir: Optional[str] = None) -> Dict[str, Any]:
        if not self.available:
            return {"available": False, "error": self.error}
        import torch
        from torch.nn import functional as F

        with self._lock:  # one inference at a time keeps GPU memory predictable
            self._load()
            with Image.open(image_path) as im:
                rgb = im.convert("RGB")
            w, h = rgb.size
            scale = min(1.0, MAX_SIDE / float(max(w, h)))
            if scale < 1.0:
                rgb = rgb.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            arr = np.asarray(rgb)
            tensor = torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float)[None] / 256.0
            tensor = tensor.to(self._device)
            with torch.no_grad():
                pred, conf, det, _ = self._model(tensor)
            loc = F.softmax(torch.squeeze(pred, 0), dim=0)[1].cpu().numpy()
            conf_map = torch.sigmoid(torch.squeeze(conf, 0))[0].cpu().numpy() if conf is not None else None
            score = float(torch.sigmoid(det).item()) if det is not None else float(loc.max())
            if self._device == "cuda":
                torch.cuda.empty_cache()

        # Region summary: pixels the model marks as manipulated, weighted by reliability
        reliable = conf_map if conf_map is not None else np.ones_like(loc)
        mask = (loc > 0.5) & (reliable > 0.5)
        area = float(mask.mean())

        mh, mw = mask.shape
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        boxes = [
            [round(x / mw, 4), round(y / mh, 4), round((x + bw) / mw, 4), round((y + bh) / mh, 4)]
            for x, y, bw, bh, a in (stats[i] for i in range(1, n))
            if a / float(mw * mh) >= 0.004
        ]
        result: Dict[str, Any] = {
            "available": True,
            "device": self._device,
            "score": round(score, 4),
            "manipulated_area": round(area, 4),
            "boxes": sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)[:4],
            "map_path": None,
        }
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            result["map_path"] = self._render(arr, loc, mask, os.path.join(out_dir, "trufor_map.png"))
        return result

    @staticmethod
    def _render(rgb: np.ndarray, loc: np.ndarray, mask: np.ndarray, path: str) -> str:
        """Dimmed image with the localization map overlaid in red and the detected regions outlined."""
        base = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
        dimmed = base * 0.4
        heat = np.zeros_like(base)
        heat[..., 2] = 255
        alpha = np.clip(loc, 0, 1)[..., None] * 0.75
        out = dimmed * (1 - alpha) + heat * alpha
        contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = out.astype(np.uint8)
        cv2.drawContours(out, contours, -1, (255, 255, 255), 2)
        cv2.imwrite(path, out)
        return path
