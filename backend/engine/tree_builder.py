from __future__ import annotations
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set
import cv2
import networkx as nx

from backend.engine.models import (
    ImageVersion,
    EdgeScore,
    ProvenanceNode,
    ProvenanceTree,
)
from backend.engine.pairwise import PairwiseEngine
from backend.engine.gemini_inspector import GeminiInspector


class TreeBuilder:
    """
    Constructs the directed provenance tree (family tree) from a set of image versions.
    1. Computes pairwise directed edge scores S(A -> B).
    2. Identifies the most credible root candidate (earliest occurrence, uncropped, highest trust).
    3. Solves the Maximum Weight Arborescence (Edmonds' Algorithm) to select the optimal parent for every version.
    4. Attaches Gemini multimodal explanations, claim drift flags, and tampering markers.
    """

    def __init__(
        self,
        pairwise_engine: Optional[PairwiseEngine] = None,
        gemini_inspector: Optional[GeminiInspector] = None,
        min_edge_threshold: float = 0.50,
    ):
        self.pairwise = pairwise_engine or PairwiseEngine()
        self.inspector = gemini_inspector or GeminiInspector()
        self.min_edge_threshold = min_edge_threshold

    def build_tree(
        self,
        versions: List[ImageVersion],
        scenario_id: Optional[str] = None,
        exclude_root_ids: Optional[Set[str]] = None,
        live_inspection_ids: Optional[Set[str]] = None,
        connect_all: bool = False,
        leaf_ids: Optional[Set[str]] = None,
    ) -> ProvenanceTree:
        """
        exclude_root_ids: versions that may not be chosen as the origin (e.g. a tampered upload
            when copies of it were found online).
        live_inspection_ids: when given, only edges touching these versions are sent to the
            live Gemini API (the rest use deterministic heuristics) to conserve quota.
        connect_all: attach versions that have no credible parent directly to the origin
            instead of leaving them disconnected.
        leaf_ids: versions that cannot be the parent of anything (e.g. the investigator's
            upload, which is the newest observation of the image).
        """
        leaf_ids = leaf_ids or set()
        if not versions:
            raise ValueError("Cannot build provenance tree with 0 versions.")

        version_map: Dict[str, ImageVersion] = {v.id: v for v in versions}

        if len(versions) == 1:
            v = versions[0]
            root_node = ProvenanceNode(
                version=v,
                depth=0,
                is_root=True,
                incoming_edge_score=1.0,
            )
            return ProvenanceTree(
                scenario_id=scenario_id,
                root_id=v.id,
                earliest_date=v.timestamp,
                nodes={v.id: root_node},
                edges=[],
                executive_summary=f"Single isolated version identified on {v.platform}.",
            )

        # 1. Compute pairwise comparison matrix (decode every image once)
        images = self._load_scaled_images(versions)
        edge_scores: Dict[Tuple[str, str], EdgeScore] = {}
        for src in versions:
            for tgt in versions:
                if src.id == tgt.id:
                    continue
                score = self.pairwise.compute_edge_score(src, tgt, images[src.id], images[tgt.id])
                edge_scores[(src.id, tgt.id)] = score

        # 2. Identify candidate root (earliest date + highest credibility)
        root_candidates = [v for v in versions if v.id not in (exclude_root_ids or set())] or versions
        root_id = self._select_best_root(root_candidates, edge_scores)

        # 3. Build directed graph for Maximum Weight Arborescence
        G = nx.DiGraph()
        for v in versions:
            G.add_node(v.id)

        for (src_id, tgt_id), score in edge_scores.items():
            # A candidate root shouldn't have incoming edges from later nodes
            if tgt_id == root_id or src_id in leaf_ids:
                continue
            if score.total_score >= self.min_edge_threshold:
                G.add_edge(src_id, tgt_id, weight=score.total_score)

        # 4. Solve Maximum Weight Branching / Arborescence
        # Use networkx Edmonds branching solver
        branching = nx.maximum_branching(G, preserve_attrs=True)

        # Ensure every non-root node has a parent from the best available edge
        final_edges_map: Dict[str, Tuple[str, float]] = {}
        for u, v, data in branching.edges(data=True):
            final_edges_map[v] = (u, data.get("weight", 0.0))

        def descendants_of(node_id: str) -> Set[str]:
            children: Dict[str, List[str]] = {}
            for child, (parent, _) in final_edges_map.items():
                children.setdefault(parent, []).append(child)
            seen, stack = set(), [node_id]
            while stack:
                for c in children.get(stack.pop(), []):
                    if c not in seen:
                        seen.add(c)
                        stack.append(c)
            return seen

        # Fallback for any disconnected nodes: pick the maximum scoring legal predecessor.
        # A node's own descendants are never legal parents (that would create a cycle).
        for v in versions:
            if v.id == root_id or v.id in final_edges_map:
                continue
            blocked = descendants_of(v.id) | {v.id} | leaf_ids
            best_parent = None
            best_s = -1.0
            for candidate in versions:
                if candidate.id in blocked:
                    continue
                s = edge_scores.get((candidate.id, v.id))
                if s and s.total_score > best_s:
                    best_s = s.total_score
                    best_parent = candidate.id
            if best_parent and best_s > 0.3:
                final_edges_map[v.id] = (best_parent, best_s)
            elif connect_all:
                # Attach to the origin so every version appears in one family tree
                final_edges_map[v.id] = (root_id, max(0.0, best_s))

        # 5. Build selected edge objects and run Gemini Multimodal Inspection
        chosen_edges: List[EdgeScore] = []
        for target_id, (source_id, _) in final_edges_map.items():
            edge = edge_scores[(source_id, target_id)]
            # Run Gemini inspection for the confirmed tree link
            src_ver = version_map[source_id]
            tgt_ver = version_map[target_id]
            allow_live = live_inspection_ids is None or bool({source_id, target_id} & live_inspection_ids)
            gemini_result = self.inspector.inspect_pair(
                src_ver, tgt_ver, edge_metrics=edge.edit_metrics, allow_live=allow_live
            )
            edge.gemini_evidence = gemini_result
            chosen_edges.append(edge)

        # 6. Assemble provenance nodes & calculate depths
        tree_graph = nx.DiGraph()
        tree_graph.add_nodes_from(v.id for v in versions)  # unconnected versions must still exist
        for edge in chosen_edges:
            tree_graph.add_edge(edge.source_id, edge.target_id)

        nodes: Dict[str, ProvenanceNode] = {}
        for v in versions:
            is_root = (v.id == root_id)
            depth = 0
            if not is_root and nx.has_path(tree_graph, root_id, v.id):
                depth = nx.shortest_path_length(tree_graph, root_id, v.id)

            parent_id = final_edges_map.get(v.id, (None, None))[0]
            incoming_score = final_edges_map.get(v.id, (None, None))[1]

            # Check for tampering on incoming edge
            tamper_flag = False
            tamper_summary = None
            for e in chosen_edges:
                if e.target_id == v.id and e.gemini_evidence:
                    if e.gemini_evidence.tampering_detected:
                        tamper_flag = True
                        tamper_summary = e.gemini_evidence.tampering_details

            nodes[v.id] = ProvenanceNode(
                version=v,
                depth=depth,
                is_root=is_root,
                parent_id=parent_id,
                incoming_edge_score=incoming_score,
                tampering_flagged=tamper_flag,
                tampering_summary=tamper_summary,
            )

        root_ver = version_map[root_id]
        executive_summary = (
            f"Earliest known occurrence identified at '{root_ver.platform}' "
            f"on {root_ver.timestamp.strftime('%d %b %Y') if root_ver.timestamp else 'Unknown Date'}. "
            f"Lineage contains {len(versions)} versions spanning {len(chosen_edges)} verified propagation hops."
        )

        return ProvenanceTree(
            scenario_id=scenario_id,
            root_id=root_id,
            earliest_date=root_ver.timestamp,
            nodes=nodes,
            edges=chosen_edges,
            executive_summary=executive_summary,
        )

    @staticmethod
    def _load_scaled_images(versions: List[ImageVersion], max_side: int = 520):
        """
        Decodes every image once and downscales all of them by one common factor, so pairwise
        template matching stays fast while relative resolution (used for edit direction) is kept.
        """
        images = {v.id: cv2.imread(v.image_path) for v in versions}
        largest = max((max(im.shape[:2]) for im in images.values() if im is not None), default=0)
        if largest > max_side:
            scale = max_side / float(largest)
            for vid, im in images.items():
                if im is not None:
                    h, w = im.shape[:2]
                    images[vid] = cv2.resize(im, (max(8, int(w * scale)), max(8, int(h * scale))),
                                             interpolation=cv2.INTER_AREA)
        return images

    def _select_best_root(
        self,
        versions: List[ImageVersion],
        edge_scores: Dict[Tuple[str, str], EdgeScore],
    ) -> str:
        """
        Picks the version with earliest timestamp, highest trust, and least crop containment.
        """
        # Sort candidate roots
        candidates = []
        for v in versions:
            # Score factors for being root
            score = 0.0

            # 1. Trust
            score += v.source_trust_score * 3.0

            # 2. C2PA / EXIF
            if v.c2pa and v.c2pa.get("is_signed"):
                score += 4.0
            if v.exif and v.exif.get("Make"):
                score += 2.0

            # 3. In-degree penalty: if v is strongly derived from others, lower root score
            in_degree_weight = 0.0
            for u in versions:
                if u.id == v.id:
                    continue
                s = edge_scores.get((u.id, v.id))
                if s and s.total_score > 0.70:
                    in_degree_weight += s.total_score
            score -= in_degree_weight * 1.5

            candidates.append((score, v.timestamp, v.id))

        # Sort: highest root score, then earliest timestamp
        epoch = datetime(1970, 1, 1)
        def sort_key(item):
            root_sc, ts, _ = item
            # If ts is None, place at end
            if ts:
                ts_naive = ts.replace(tzinfo=None) if hasattr(ts, 'tzinfo') and ts.tzinfo else ts
                ts_val = (ts_naive - epoch).total_seconds()
            else:
                ts_val = 9999999999
            return (-root_sc, ts_val)

        candidates.sort(key=sort_key)
        return candidates[0][2]
