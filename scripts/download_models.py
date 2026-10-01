"""
Downloads the model files TraceLens uses into backend/models_data/:

  - TruFor weights (GRIP-UNINA, ~260 MB). Licence: informational / non-profit use only.
    https://grip-unina.github.io/TruFor/
  - YuNet face detector (OpenCV model zoo, ~230 KB, MIT).

Usage:  python scripts/download_models.py
TruFor also needs PyTorch, timm and yacs (see requirements.txt).
"""
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

DEST = Path(__file__).resolve().parents[1] / "backend" / "models_data"
TRUFOR_URL = "https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip"
TRUFOR_MD5 = "7bee48f3476c75616c3c5721ab256ff8"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


def download(url: str, target: Path):
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as resp, open(target, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                sys.stdout.write(f"\r  {done / total * 100:5.1f}%  ({done >> 20} / {total >> 20} MB)")
                sys.stdout.flush()
    print()


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    DEST.mkdir(parents=True, exist_ok=True)

    yunet = DEST / "face_detection_yunet_2023mar.onnx"
    if not yunet.exists():
        download(YUNET_URL, yunet)

    weights = DEST / "weights" / "trufor.pth.tar"
    if weights.exists():
        print(f"TruFor weights already present: {weights}")
        return
    archive = DEST / "TruFor_weights.zip"
    if not archive.exists() or md5(archive) != TRUFOR_MD5:
        download(TRUFOR_URL, archive)
    if md5(archive) != TRUFOR_MD5:
        sys.exit("TruFor archive checksum mismatch — delete it and try again.")
    with zipfile.ZipFile(archive) as z:
        z.extractall(DEST)
    print(f"TruFor weights ready: {weights}")


if __name__ == "__main__":
    main()
