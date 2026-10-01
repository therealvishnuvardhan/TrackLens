#!/usr/bin/env python3
import sys
import os
import json
from pathlib import Path
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.engine.models import ImageVersion
from backend.engine.pairwise import PairwiseEngine
from backend.engine.gemini_inspector import GeminiInspector
from backend.engine.tree_builder import TreeBuilder


def print_banner():
    banner = """
==================================================================
                 TraceLens: Provenance Detective Engine
==================================================================
"""
    print(banner)


def run_scenario(scenario_name: str = "scenario_title_slide"):
    print_banner()
    fixtures_dir = Path(__file__).resolve().parent / "backend" / "fixtures" / scenario_name
    gt_file = fixtures_dir / "ground_truth.json"

    if not gt_file.exists():
        print(f"[Error] Scenario directory not found: {fixtures_dir}")
        sys.exit(1)

    with open(gt_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loading Scenario: {data.get('title')}")
    print(f"Description: {data.get('description')}\n")

    versions = []
    for item in data["versions"]:
        img_path = str(fixtures_dir / item["image_path"])
        ts = datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00")) if item.get("timestamp") else None
        v = ImageVersion(
            id=item["id"],
            platform=item["platform"],
            author=item.get("author"),
            url=item.get("url"),
            timestamp=ts,
            caption=item.get("caption"),
            image_path=img_path,
            dimensions=tuple(item["dimensions"]) if item.get("dimensions") else None,
            c2pa=item.get("c2pa"),
            exif=item.get("exif", {}),
            source_trust_score=item.get("source_trust_score", 0.5),
        )
        versions.append(v)

    pairwise = PairwiseEngine()
    inspector = GeminiInspector()
    builder = TreeBuilder(pairwise_engine=pairwise, gemini_inspector=inspector)

    print("[Step 4 & 5] Computing pairwise evidence matrix & solving provenance tree...")
    tree = builder.build_tree(versions, scenario_id=scenario_name)

    print("\n" + "=" * 65)
    print("RECONSTRUCTED PROVENANCE TREE (FAMILY TREE)")
    print("=" * 65)
    print(f"Root (Earliest Occurrence): {tree.root_id}")
    print(f"Earliest Date: {tree.earliest_date}")
    print(f"Executive Summary: {tree.executive_summary}\n")

    # Group edges by parent
    from collections import defaultdict
    children_map = defaultdict(list)
    for e in tree.edges:
        children_map[e.source_id].append(e)

    def print_node(node_id: str, prefix: str = "", is_last: bool = True):
        node = tree.nodes[node_id]
        v = node.version
        branch = "└── " if is_last else "├── "
        if node.is_root:
            print(f"ROOT: {v.platform} [{v.id}] ({v.timestamp.strftime('%d %b %Y') if v.timestamp else 'No Date'})")
        else:
            tamper_str = " ⚠ MANIPULATION DETECTED" if node.tampering_flagged else ""
            print(f"{prefix}{branch}{v.platform} [{v.id}] (Score: {node.incoming_edge_score:.2f}){tamper_str}")

        children = children_map.get(node_id, [])
        new_prefix = prefix + ("    " if is_last else "│   ")
        for i, child_edge in enumerate(children):
            is_last_child = (i == len(children) - 1)
            print_node(child_edge.target_id, new_prefix, is_last_child)

    print_node(tree.root_id)

    print("\n" + "=" * 65)
    print("EDGE EVIDENCE & FORENSIC BREAKDOWN")
    print("=" * 65)
    for e in tree.edges:
        print(f"\nHop: {e.source_id} -> {e.target_id} (Total Confidence: {e.total_score:.2f})")
        print(f"  • Score Factors: Visual={e.visual_score} (35%) | Time={e.time_score} (20%) | Edit={e.edit_direction_score} (15%) | Meta={e.metadata_score} (15%) | Trust={e.trust_score} (15%)")
        if e.gemini_evidence:
            print(f"  • Visual Diff: {e.gemini_evidence.visual_diff_summary}")
            print(f"  • Claim Drift: [{e.gemini_evidence.claim_drift.value}] {e.gemini_evidence.claim_drift_reasoning}")
            if e.gemini_evidence.tampering_detected:
                print(f"  • Forensic Flag: {e.gemini_evidence.tampering_details}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "scenario_title_slide"
    run_scenario(scenario)
