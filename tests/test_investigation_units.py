from datetime import datetime
from fractions import Fraction

import numpy as np
from PIL import Image, ImageDraw

from backend.engine.models import ImageVersion
from backend.engine.tree_builder import TreeBuilder
from backend.forensics.classifier import ForensicClassifier
from backend.ingestion.exif_parser import ExifParser
from backend.search.match_verifier import MatchVerifier
from backend.search.reverse_search import _clean_url, _blocked

FIXTURE = "backend/fixtures/scenario_title_slide/images/v1_photographer_blog.png"


def _photo(path, seed=0, size=(640, 420)):
    """A textured, photo-like image (smooth gradients + noise) so keypoints exist."""
    rng = np.random.default_rng(seed)
    h, w = size[1], size[0]
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([
        120 + 60 * np.sin(xx / 37.0) + 30 * np.cos(yy / 23.0),
        110 + 50 * np.cos(xx / 29.0 + yy / 41.0),
        100 + 40 * np.sin((xx + yy) / 53.0),
    ], axis=-1)
    blobs = rng.integers(0, 255, size=(h // 8 + 1, w // 8 + 1, 3)).repeat(8, 0).repeat(8, 1)
    img = np.clip(0.7 * base + 0.3 * blobs[:h, :w] + rng.normal(0, 6, (h, w, 3)), 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=92)
    return path


def test_overlay_banner_is_flagged(tmp_path):
    src = _photo(str(tmp_path / "photo.jpg"))
    im = Image.open(src).convert("RGB")
    d = ImageDraw.Draw(im)
    d.rectangle((0, im.height - 60, im.width, im.height), fill=(200, 20, 20))
    d.text((20, im.height - 45), "BREAKING: FAKE CLAIM", fill="white")
    banner = str(tmp_path / "banner.jpg")
    im.save(banner, quality=90)

    c = ForensicClassifier()
    assert c.classify(banner)["verdict"] == "TAMPERED"
    assert c.classify(src)["verdict"] == "ORIGINAL"


def test_verifier_measures_local_edit(tmp_path):
    src = _photo(str(tmp_path / "a.jpg"), seed=3)
    im = Image.open(src).convert("RGB")
    ImageDraw.Draw(im).rectangle((400, 50, 560, 200), fill=(250, 250, 30))
    edited = str(tmp_path / "b.jpg")
    im.save(edited, quality=92)

    v = MatchVerifier()
    same = v.compare_paths(src, src)
    diff = v.compare_paths(edited, src)
    assert same["diff_fraction"] == 0.0 and same["match_type"] == "FULL_MATCH"
    assert diff["related"] and diff["diff_fraction"] > 0.05
    assert diff["match_type"] == "DERIVATIVE"


def test_tree_builder_keeps_upload_as_leaf_and_connects_everything(tmp_path):
    paths = []
    base = Image.open(FIXTURE).convert("RGB")
    for i, scale in enumerate([1.0, 0.8, 0.6, 0.5]):
        p = str(tmp_path / f"v{i}.png")
        base.resize((int(base.width * scale), int(base.height * scale))).save(p)
        paths.append(p)
    versions = [
        ImageVersion(id="upload", platform="Your upload", image_path=paths[3], timestamp=datetime(2026, 1, 1)),
        ImageVersion(id="a", platform="News", image_path=paths[0], timestamp=datetime(2020, 1, 1), source_trust_score=0.9),
        ImageVersion(id="b", platform="Social", image_path=paths[1], timestamp=None),
        ImageVersion(id="c", platform="Blog", image_path=paths[2], timestamp=None),
    ]
    tree = TreeBuilder().build_tree(
        versions, exclude_root_ids={"upload"}, live_inspection_ids=set(),
        connect_all=True, leaf_ids={"upload"},
    )
    assert tree.root_id != "upload"
    assert all(n.parent_id != "upload" for n in tree.nodes.values())
    assert all(n.is_root or n.parent_id for n in tree.nodes.values())


def test_exif_values_are_json_safe():
    assert ExifParser._json_safe("a\x00b") == "ab"
    assert ExifParser._json_safe((1, Fraction(1, 4))) == [1, 0.25]
    assert ExifParser._json_safe(b"\x01\x02").startswith("<bytes")


def test_search_url_cleanup_and_filtering():
    assert _clean_url("https://x.com/a?utm_source=y&id=3") == "https://x.com/a?id=3"
    assert _clean_url("//cdn.example.com/i.jpg") == "https://cdn.example.com/i.jpg"
    assert _blocked("https://some-porn-site.example", None)
    assert not _blocked("https://en.wikipedia.org/wiki/X", "Eiffel Tower")
