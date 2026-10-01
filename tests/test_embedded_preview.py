import io
import struct

from PIL import Image, ImageDraw

from backend.forensics.embedded_preview import EmbeddedPreviewCheck
from tests.test_investigation_units import _photo


def _exif_with_thumbnail(thumb: Image.Image) -> bytes:
    """Builds a minimal little-endian EXIF block: IFD0 (Make) -> IFD1 (JPEG thumbnail)."""
    buf = io.BytesIO()
    thumb.save(buf, "JPEG", quality=85)
    jpeg = buf.getvalue()
    make = b"TestCam\x00"

    ifd0_off = 8
    ifd0_size = 2 + 1 * 12 + 4
    make_off = ifd0_off + ifd0_size
    ifd1_off = make_off + len(make)
    ifd1_size = 2 + 3 * 12 + 4
    thumb_off = ifd1_off + ifd1_size

    tiff = b"II*\x00" + struct.pack("<I", ifd0_off)
    tiff += struct.pack("<H", 1) + struct.pack("<HHII", 0x010F, 2, len(make), make_off) + struct.pack("<I", ifd1_off)
    tiff += make
    tiff += struct.pack("<H", 3)
    tiff += struct.pack("<HHII", 0x0103, 3, 1, 6)
    tiff += struct.pack("<HHII", 0x0201, 4, 1, thumb_off)
    tiff += struct.pack("<HHII", 0x0202, 4, 1, len(jpeg))
    tiff += struct.pack("<I", 0)
    tiff += jpeg
    return b"Exif\x00\x00" + tiff


def _camera_jpeg(path, img: Image.Image, thumb_source: Image.Image):
    thumb = thumb_source.copy()
    thumb.thumbnail((240, 240))
    img.save(path, "JPEG", quality=92, exif=_exif_with_thumbnail(thumb))


def test_preview_matches_unedited_photo(tmp_path):
    src = Image.open(_photo(str(tmp_path / "base.jpg"), seed=5)).convert("RGB")
    path = str(tmp_path / "camera.jpg")
    _camera_jpeg(path, src, src)
    res = EmbeddedPreviewCheck().check(path, str(tmp_path / "out"))
    assert res["previews_found"] >= 1
    assert res["status"] == "consistent", res


def test_preview_reveals_edit(tmp_path):
    src = Image.open(_photo(str(tmp_path / "base.jpg"), seed=6)).convert("RGB")
    edited = src.copy()
    ImageDraw.Draw(edited).ellipse((380, 60, 600, 280), fill=(240, 240, 40))
    path = str(tmp_path / "edited.jpg")
    _camera_jpeg(path, edited, src)  # editor rewrote pixels, kept the capture-time preview
    res = EmbeddedPreviewCheck().check(path, str(tmp_path / "out"))
    assert res["status"] == "mismatch", res
    assert res["diff_fraction"] > 0.05
    assert res["preview_path"] and res["diff_path"]


def test_preview_reveals_crop(tmp_path):
    src = Image.open(_photo(str(tmp_path / "base.jpg"), seed=7)).convert("RGB")
    cropped = src.crop((0, 0, 400, 400))
    path = str(tmp_path / "cropped.jpg")
    _camera_jpeg(path, cropped, src)
    assert EmbeddedPreviewCheck().check(path, str(tmp_path / "out"))["status"] == "cropped"


def test_no_preview(tmp_path):
    path = _photo(str(tmp_path / "plain.jpg"), seed=8)
    assert EmbeddedPreviewCheck().check(path, str(tmp_path / "out"))["status"] == "absent"
