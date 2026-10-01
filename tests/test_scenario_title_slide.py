import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Force UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.engine.models import ImageVersion, ClaimDriftType
from backend.engine.pairwise import PairwiseEngine
from backend.engine.gemini_inspector import GeminiInspector
from backend.engine.tree_builder import TreeBuilder


def load_fixture_scenario():
    base_dir = Path(__file__).resolve().parent.parent / "backend" / "fixtures" / "scenario_title_slide"
    gt_file = base_dir / "ground_truth.json"

    with open(gt_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    versions = []
    for item in data["versions"]:
        img_full_path = str(base_dir / item["image_path"])
        ts = datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00")) if item.get("timestamp") else None
        v = ImageVersion(
            id=item["id"],
            platform=item["platform"],
            author=item.get("author"),
            url=item.get("url"),
            timestamp=ts,
            caption=item.get("caption"),
            image_path=img_full_path,
            dimensions=tuple(item["dimensions"]) if item.get("dimensions") else None,
            c2pa=item.get("c2pa"),
            exif=item.get("exif", {}),
            source_trust_score=item.get("source_trust_score", 0.5),
        )
        versions.append(v)

    return data, versions


def test_pairwise_and_tree():
    data, versions = load_fixture_scenario()
    assert len(versions) == 6, f"Expected 6 versions, got {len(versions)}"

    pairwise = PairwiseEngine()
    inspector = GeminiInspector()
    builder = TreeBuilder(pairwise_engine=pairwise, gemini_inspector=inspector)

    # 1. Run pairwise engine
    print("\n" + "=" * 60)
    print("STEP 4: PAIRWISE DIRECTED EVIDENCE SCORES")
    print("=" * 60)

    # Test key directed edges
    v_map = {v.id: v for v in versions}
    v1 = v_map["v1_photographer_blog"]
    v2 = v_map["v2_twitter"]
    v3 = v_map["v3_whatsapp"]
    v4 = v_map["v4_reddit"]
    v5 = v_map["v5_instagram"]
    v6 = v_map["v6_news_outlet"]

    s12 = pairwise.compute_edge_score(v1, v2)
    s21 = pairwise.compute_edge_score(v2, v1)
    print(f"Edge v1 -> v2 (Original -> Twitter): {s12.total_score:.4f} (Visual: {s12.visual_score}, Time: {s12.time_score}, Edit: {s12.edit_direction_score})")
    print(f"Edge v2 -> v1 (Twitter -> Original): {s21.total_score:.4f} (Violates time & crop)")
    assert s12.total_score > s21.total_score, "v1 -> v2 must score higher than reverse"

    s23 = pairwise.compute_edge_score(v2, v3)
    print(f"Edge v2 -> v3 (Twitter -> WhatsApp): {s23.total_score:.4f} (Additions: {s23.edit_metrics.addition_mask_score})")

    s34 = pairwise.compute_edge_score(v3, v4)
    print(f"Edge v3 -> v4 (WhatsApp -> Reddit): {s34.total_score:.4f}")

    s15 = pairwise.compute_edge_score(v1, v5)
    print(f"Edge v1 -> v5 (Original -> Instagram): {s15.total_score:.4f} (Crop containment: {s15.edit_metrics.is_crop_containment})")

    s56 = pairwise.compute_edge_score(v5, v6)
    print(f"Edge v5 -> v6 (Instagram -> News): {s56.total_score:.4f}")

    # 2. Build Tree
    print("\n" + "=" * 60)
    print("STEP 5: RECONSTRUCTED PROVENANCE TREE")
    print("=" * 60)
    tree = builder.build_tree(versions, scenario_id="scenario_title_slide")

    print(f"Identified Root: {tree.root_id} ({tree.nodes[tree.root_id].version.platform})")
    assert tree.root_id == "v1_photographer_blog", f"Expected root v1_photographer_blog, got {tree.root_id}"

    # Print Family Tree
    print("\nFamily Tree:")
    for edge in tree.edges:
        src = tree.nodes[edge.source_id].version
        tgt = tree.nodes[edge.target_id].version
        drift_tag = f"[{edge.gemini_evidence.claim_drift.value}]" if edge.gemini_evidence else ""
        tamper_tag = "⚠ MANIPULATED" if (edge.gemini_evidence and edge.gemini_evidence.tampering_detected) else ""
        print(
            f"  {src.platform} ({src.id}) ─── {edge.total_score:.2f} ───> {tgt.platform} ({tgt.id})  "
            f"{drift_tag} {tamper_tag}"
        )
        if edge.gemini_evidence:
            print(f"      Diff: {edge.gemini_evidence.visual_diff_summary}")
            print(f"      Drift: {edge.gemini_evidence.claim_drift_reasoning}")
            if edge.gemini_evidence.tampering_details:
                print(f"      Tampering: {edge.gemini_evidence.tampering_details}")

    # Assert expected edges are present
    edge_pairs = {(e.source_id, e.target_id) for e in tree.edges}
    expected_pairs = {
        ("v1_photographer_blog", "v2_twitter"),
        ("v2_twitter", "v3_whatsapp"),
        ("v3_whatsapp", "v4_reddit"),
        ("v1_photographer_blog", "v5_instagram"),
        ("v5_instagram", "v6_news_outlet"),
    }

    print("\nEdge Topology Validation:")
    for ep in expected_pairs:
        status = "MATCHED" if ep in edge_pairs else "MISSING"
        print(f"  {ep[0]} -> {ep[1]}: [{status}]")

    assert ("v1_photographer_blog", "v2_twitter") in edge_pairs
    assert ("v2_twitter", "v3_whatsapp") in edge_pairs
    assert ("v3_whatsapp", "v4_reddit") in edge_pairs
    assert ("v1_photographer_blog", "v5_instagram") in edge_pairs

    # Verify tampering was detected on Reddit node
    reddit_node = tree.nodes["v4_reddit"]
    assert reddit_node.tampering_flagged is True, "Reddit node must have tampering_flagged = True"
    print(f"\nForensic Localization:")
    print(f"  Tampering pinpointed at node '{reddit_node.version.id}': {reddit_node.tampering_summary}")

    print("\n✓ ALL PHASE 0 & PHASE 1 VALIDATION CHECKS PASSED!")


if __name__ == "__main__":
    test_pairwise_and_tree()
