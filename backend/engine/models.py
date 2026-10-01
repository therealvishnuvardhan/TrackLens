from __future__ import annotations
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from pydantic import BaseModel, Field


class ClaimDriftType(str, Enum):
    SAME = "SAME"
    REFRAMED = "REFRAMED"
    CONTRADICTED = "CONTRADICTED"


class ImageVersion(BaseModel):
    id: str
    platform: str
    author: Optional[str] = None
    url: Optional[str] = None
    timestamp: Optional[datetime] = None
    caption: Optional[str] = None
    image_path: str
    dimensions: Optional[Tuple[int, int]] = None
    phash: Optional[str] = None
    c2pa: Optional[Dict[str, Any]] = None
    exif: Optional[Dict[str, Any]] = Field(default_factory=dict)
    source_trust_score: float = 0.5
    image_url: Optional[str] = None         # browser-servable URL of image_path
    source_image_url: Optional[str] = None  # original image URL on the web (for discovered copies)
    domain: Optional[str] = None
    category: Optional[str] = None          # social | news | fact_check | stock | reference | blog | upload


class EditDirectionMetrics(BaseModel):
    is_crop_containment: float = 0.0  # +1 if B is inside A (A is parent), -1 if A inside B
    containment_confidence: float = 0.0
    resolution_ratio: float = 1.0  # target_area / source_area (< 1 means target is downscaled/cropped)
    sharpness_ratio: float = 1.0   # target_var / source_var (< 1 means target is blurrier/compressed)
    addition_mask_score: float = 0.0  # > 0 indicates added text/stickers/watermarks in B not in A
    direction_score: float = 0.5   # [0, 1] probability that B was produced from A


class GeminiPairInspection(BaseModel):
    visual_diff_summary: str
    edit_types: List[str] = Field(default_factory=list)
    claim_drift: ClaimDriftType = ClaimDriftType.SAME
    claim_drift_reasoning: str
    tampering_detected: bool = False
    tampering_details: Optional[str] = None


class EdgeScore(BaseModel):
    source_id: str
    target_id: str
    visual_score: float = 0.0         # 35% weight
    time_score: float = 0.5           # 20% weight
    edit_direction_score: float = 0.5 # 15% weight
    metadata_score: float = 0.5       # 15% weight
    trust_score: float = 0.5          # 15% weight
    total_score: float = 0.0          # Weighted sum [0, 1]
    gemini_evidence: Optional[GeminiPairInspection] = None
    edit_metrics: Optional[EditDirectionMetrics] = None


class ProvenanceNode(BaseModel):
    version: ImageVersion
    depth: int = 0
    is_root: bool = False
    parent_id: Optional[str] = None
    incoming_edge_score: Optional[float] = None
    tampering_flagged: bool = False
    tampering_summary: Optional[str] = None
    ela_path: Optional[str] = None
    forensic_analysis: Optional[Dict[str, Any]] = None
    kind: str = "version"                   # version | upload | web_copy | inferred_original
    altered_here: bool = False              # content changed relative to its parent at this hop
    diff_fraction: Optional[float] = None   # share of the frame that differs from the parent


class WebMatchItem(BaseModel):
    match_id: str
    image_url: str
    page_url: Optional[str] = None
    page_title: Optional[str] = None
    platform: str
    match_type: str = "VISUALLY_SIMILAR" # FULL_MATCH | PARTIAL_CROP | VISUALLY_SIMILAR | DERIVATIVE
    similarity_score: float = 0.85
    published_at: Optional[datetime] = None
    dimensions: Optional[Tuple[int, int]] = None
    is_earlier_than_query: bool = False
    is_higher_res_than_query: bool = False
    is_broader_crop_than_query: bool = False
    verdict_relationship: str = "POTENTIAL_PARENT"
    thumbnail_url: Optional[str] = None
    source_image_url: Optional[str] = None
    domain: Optional[str] = None
    category: Optional[str] = None
    diff_fraction: Optional[float] = None


class ProvenanceTree(BaseModel):
    scenario_id: Optional[str] = None
    root_id: str
    earliest_date: Optional[datetime] = None
    nodes: Dict[str, ProvenanceNode] = Field(default_factory=dict)
    edges: List[EdgeScore] = Field(default_factory=list)
    executive_summary: Optional[str] = None
    web_matches: List[WebMatchItem] = Field(default_factory=list)
    authenticity_verdict: Optional[str] = None # AUTHENTIC_ORIGINAL | MANIPULATED_DERIVATIVE | ISOLATED_MEDIA
    verdict_confidence: float = 0.90
    verdict_explanation: Optional[str] = None
    classification: Optional[Dict[str, Any]] = None
    upload_node_id: Optional[str] = None
    lineage_path: List[str] = Field(default_factory=list)   # root ... upload
    stage_count: int = 0                                     # hops from the original to the upload
    platforms: List[Dict[str, Any]] = Field(default_factory=list)
    search_summary: Optional[Dict[str, Any]] = None
    proof: List[Dict[str, Any]] = Field(default_factory=list)    # evidence that the image was edited
    regions: List[Dict[str, Any]] = Field(default_factory=list)  # per-region reverse searches
