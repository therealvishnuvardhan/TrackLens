import os
import json
import pytest
from datetime import datetime

from backend.engine.models import ImageVersion
from backend.engine.pairwise import PairwiseEngine
from backend.engine.gemini_inspector import GeminiInspector
from backend.main import compute_ela_map


def test_test_pairs_exist():
    """Verify that all image pairs and metadata exist in tests/test_pairs/"""
    pairs_dir = "tests/test_pairs"
    meta_path = os.path.join(pairs_dir, "metadata.json")
    assert os.path.exists(meta_path), "metadata.json must exist in tests/test_pairs/"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert "test_pairs" in meta
    assert len(meta["test_pairs"]) >= 3

    for p in meta["test_pairs"]:
        orig_file = os.path.join(pairs_dir, p["original"]["file"])
        manip_file = os.path.join(pairs_dir, p["manipulated"]["file"])
        assert os.path.exists(orig_file), f"Original file {orig_file} not found"
        assert os.path.exists(manip_file), f"Manipulated file {manip_file} not found"
        assert os.path.getsize(orig_file) > 1000
        assert os.path.getsize(manip_file) > 1000


def test_pairwise_direction_and_ela_on_pairs():
    """Verify pairwise scoring and Error Level Analysis (ELA) generation on test pairs"""
    engine = PairwiseEngine()
    pairs_dir = "tests/test_pairs"
    ela_out = os.path.join(pairs_dir, "test_ela_reddit.png")

    # Generate ELA map on manipulated image
    ela_res = compute_ela_map(
        os.path.join(pairs_dir, "pair2_reddit_ai_tampered.png"),
        ela_out
    )
    assert os.path.exists(ela_out)
    assert os.path.getsize(ela_out) > 5000

    # Test pair 2: clean original -> AI face tampered
    v_clean = ImageVersion(
        id="clean",
        platform="Elena Rossi Blog",
        image_path=os.path.join(pairs_dir, "pair2_valencia_original.png"),
        timestamp=datetime(2024, 3, 2, 10, 0, 0),
        caption="Rescue volunteers wading through flash flood waters",
        source_trust_score=0.95
    )
    v_tampered = ImageVersion(
        id="tampered",
        platform="Reddit",
        image_path=os.path.join(pairs_dir, "pair2_reddit_ai_tampered.png"),
        timestamp=datetime(2024, 3, 9, 18, 45, 0),
        caption="AI or real? Rescuer with terrifying glowing eyes in California floodwaters",
        source_trust_score=0.40
    )

    edge_forward = engine.compute_edge_score(v_clean, v_tampered)
    edge_reverse = engine.compute_edge_score(v_tampered, v_clean)

    # Clean -> Tampered should have a much higher score than reverse
    assert edge_forward.total_score > edge_reverse.total_score
    assert edge_forward.total_score >= 0.70
    assert edge_forward.visual_score >= 0.70


def test_gemini_detection_on_manipulated_pairs():
    """Verify Gemini inspector detects tampering on the manipulated copies"""
    inspector = GeminiInspector()
    pairs_dir = "tests/test_pairs"

    # Single-image inspection of pair 1 manipulated (Lincoln composite)
    manip_path = os.path.join(pairs_dir, "pair1_lincoln_manipulated.jpg")
    analysis = inspector.inspect_single_image(
        manip_path,
        caption="Famous 19th-century portrait"
    )
    assert "tampering_detected" in analysis
    assert "visual_summary" in analysis
