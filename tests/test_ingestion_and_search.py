import os
import sys
import tempfile
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.ingestion.exif_parser import ExifParser
from backend.ingestion.c2pa_verifier import C2PAVerifier
from backend.ingestion.video_keyframe import VideoKeyframeExtractor
from backend.ingestion.embedder import ImageEmbedder
from backend.ingestion.pipeline import IngestionPipeline
from backend.search.vision_client import VisionWebDetectionClient
from backend.search.page_scraper import PageScraper
from backend.search.discovery_orchestrator import DiscoveryOrchestrator
from backend.engine.tree_builder import TreeBuilder


def test_exif_and_c2pa():
    print("\n--- 1. Testing EXIF & C2PA Ingestion ---")
    img_path = "backend/fixtures/scenario_title_slide/images/v1_photographer_blog.png"
    assert os.path.exists(img_path)

    # Test EXIF
    exif = ExifParser.extract_metadata(img_path)
    print(f"EXIF Parsed: has_exif={exif['has_exif']}, wash_detected={exif['platform_wash_detected']}")

    # Test C2PA
    c2pa = C2PAVerifier.verify_content_credentials(img_path)
    print(f"C2PA Status: validation_status={c2pa['validation_status']}, is_signed={c2pa['is_signed']}")
    assert "validation_status" in c2pa


def test_embedder_fingerprints():
    print("\n--- 2. Testing Image Fingerprinting (pHash & Embeddings) ---")
    img1 = "backend/fixtures/scenario_title_slide/images/v1_photographer_blog.png"
    img2 = "backend/fixtures/scenario_title_slide/images/v2_twitter.png"

    fp1 = ImageEmbedder.compute_fingerprint(img1)
    fp2 = ImageEmbedder.compute_fingerprint(img2)

    assert fp1["phash"] is not None
    assert fp2["phash"] is not None
    assert len(fp1["feature_vector"]) == 64

    hamming_sim = ImageEmbedder.hamming_similarity(fp1["phash"], fp2["phash"])
    cos_sim = ImageEmbedder.cosine_similarity(fp1["feature_vector"], fp2["feature_vector"])
    print(f"pHash v1: {fp1['phash']} | pHash v2: {fp2['phash']}")
    print(f"pHash Hamming Similarity: {hamming_sim:.4f}")
    print(f"Semantic Vector Cosine Similarity: {cos_sim:.4f}")
    assert hamming_sim > 0.40
    assert cos_sim > 0.50


def test_video_keyframe_extractor():
    print("\n--- 3. Testing Video Keyframe Extraction ---")
    # Synthesize a temporary test video with two distinct color scenes
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "test_flood_clip.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (320, 240))

    # Scene 1: Blue floodwater frames (15 frames)
    for _ in range(15):
        frame = np.full((240, 320, 3), (120, 80, 40), dtype=np.uint8)
        out.write(frame)

    # Scene 2: Bright orange rescue boat transition (15 frames)
    for _ in range(15):
        frame = np.full((240, 320, 3), (20, 100, 240), dtype=np.uint8)
        out.write(frame)

    out.release()
    assert os.path.exists(video_path)

    extractor = VideoKeyframeExtractor(scene_threshold=0.50, max_keyframes=4, sample_rate=2)
    keyframes = extractor.extract_keyframes(video_path, output_dir=os.path.join(temp_dir, "keyframes"))

    print(f"Extracted {len(keyframes)} scene keyframes from video:")
    for kf in keyframes:
        print(f"  • Frame {kf['frame_index']} @ {kf['timestamp_sec']}s (Transition: {kf.get('transition_score')}) -> {kf['image_path']}")

    assert len(keyframes) >= 2, f"Expected at least 2 keyframes across scene transition, got {len(keyframes)}"


def test_web_discovery_and_end_to_end():
    print("\n--- 4. Testing Web Discovery & End-to-End Orchestrator ---")
    query_img = "backend/fixtures/scenario_title_slide/images/v3_whatsapp.png"

    orchestrator = DiscoveryOrchestrator()
    relatives = orchestrator.discover_relatives(query_img, query_caption="Suspicious WhatsApp California flood forward")

    print(f"Discovered {len(relatives)} related versions online across platforms:")
    for v in relatives:
        print(f"  • {v.platform} ({v.id}) | Date: {v.timestamp} | Trust: {v.source_trust_score}")

    assert len(relatives) >= 5, f"Expected at least 5 relatives, got {len(relatives)}"

    # Run tree builder on discovered corpus
    builder = TreeBuilder()
    tree = builder.build_tree(relatives, scenario_id="discovered_web_lineage")

    print("\nReconstructed Tree from Web Discovery:")
    print(f"  Root: {tree.root_id} ({tree.nodes[tree.root_id].version.platform})")
    assert tree.root_id == "v1_photographer_blog", f"Expected root v1_photographer_blog, got {tree.root_id}"
    print(f"  Total Edges: {len(tree.edges)}")
    assert len(tree.edges) >= 4

    print("\n✓ ALL PHASE 2 & PHASE 3 INGESTION AND DISCOVERY CHECKS PASSED!")


if __name__ == "__main__":
    test_exif_and_c2pa()
    test_embedder_fingerprints()
    test_video_keyframe_extractor()
    test_web_discovery_and_end_to_end()
