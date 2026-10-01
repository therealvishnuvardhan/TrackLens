from PIL import Image, ImageDraw

from backend.search.match_verifier import MatchVerifier
from backend.search.region_search import RegionSourceSearch, MAX_REGIONS
from tests.test_investigation_units import _photo


def test_proposals_prioritise_changed_regions_and_dedupe():
    props = RegionSourceSearch.build_proposals([
        ("face", [0.40, 0.10, 0.50, 0.25]),
        ("forensic", [0.41, 0.11, 0.51, 0.26]),          # same place as the face -> dropped
        ("changed_vs_copy", [0.10, 0.50, 0.30, 0.80]),
        ("model", [0.0, 0.0, 0.95, 0.95]),              # nearly the whole frame -> skipped
        ("forensic", [0.70, 0.70, 0.80, 0.80]),
        ("face", [0.85, 0.05, 0.95, 0.15]),
    ])
    kinds = [p["kind"] for p in props]
    assert kinds[0] == "changed_vs_copy"
    assert len(props) <= MAX_REGIONS
    assert all(0.0 <= v <= 1.0 for p in props for v in p["box"])


def test_difference_boxes_locate_the_edit(tmp_path):
    ref = _photo(str(tmp_path / "ref.jpg"), seed=11)
    im = Image.open(ref).convert("RGB")
    ImageDraw.Draw(im).rectangle((420, 80, 580, 220), fill=(30, 220, 240))
    edited = str(tmp_path / "edited.jpg")
    im.save(edited, quality=92)

    boxes = MatchVerifier().difference_boxes(edited, ref)
    assert boxes, "edited area should be found"
    x0, y0, x1, y1 = boxes[0]
    # The edit spans x 420-580 / 640 and y 80-220 / 420
    assert x0 < 0.70 < x1 and y0 < 0.35 < y1
