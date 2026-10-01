from __future__ import annotations
import json
import shutil
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.engine.models import (
    ImageVersion,
    EdgeScore,
    ProvenanceTree,
)
from backend.engine.pairwise import PairwiseEngine
from backend.engine.gemini_inspector import GeminiInspector
from backend.engine.tree_builder import TreeBuilder
from backend.forensics.ela import compute_ela_map
from backend.ingestion.pipeline import IngestionPipeline
from backend.jobs import JobManager
from backend.pipeline.investigation import InvestigationPipeline, safe_filename, to_url
from backend.search.discovery_orchestrator import DiscoveryOrchestrator
from backend.search.vision_client import compute_authenticity_verdict

app = FastAPI(
    title="TraceLens API",
    description="Provenance and Mutation Lineage Reconstruction Engine for Viral Media",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pairwise_engine = PairwiseEngine()
gemini_inspector = GeminiInspector()
tree_builder = TreeBuilder(pairwise_engine=pairwise_engine, gemini_inspector=gemini_inspector)
ingestion_pipeline = IngestionPipeline()
discovery_orchestrator = DiscoveryOrchestrator()
investigation_pipeline = InvestigationPipeline(gemini_inspector, tree_builder, ingestion_pipeline)
job_manager = JobManager()

BACKEND_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = BACKEND_DIR / "fixtures"
UPLOADS_DIR = BACKEND_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Mount uploads directory (includes per-investigation folders under uploads/jobs/)
app.mount("/static/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

# Serve fixture images statically
if (FIXTURES_DIR / "scenario_title_slide" / "images").exists():
    app.mount(
        "/static/fixtures",
        StaticFiles(directory=str(FIXTURES_DIR / "scenario_title_slide" / "images")),
        name="fixtures",
    )

# Serve test pairs statically
TEST_PAIRS_DIR = BACKEND_DIR.parent / "tests" / "test_pairs"
if TEST_PAIRS_DIR.exists():
    app.mount(
        "/static/test_pairs",
        StaticFiles(directory=str(TEST_PAIRS_DIR)),
        name="test_pairs",
    )


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "gemini_api_configured": bool(gemini_inspector.client),
        "gemini_available": gemini_inspector.available,
        "gemini_models": gemini_inspector.models,
        "reverse_search_engines": ["Yandex reverse image search", "Bing visual search"],
        "engine": "TraceLens v1.1",
    }


# ---------------------------------------------------------------------------
# Live investigations (stage-by-stage pipeline the UI polls)
# ---------------------------------------------------------------------------

@app.post("/api/investigations")
async def start_investigation(
    file: UploadFile = File(...),
    caption: Optional[str] = Form(None),
    force_trace: bool = Form(False),
):
    """
    Starts the investigation pipeline for an uploaded image and returns immediately.
    Poll GET /api/investigations/{id} for per-stage progress and the final result.
    """
    job = job_manager.create(file.filename or "upload", caption or None, force_trace)
    upload_path = InvestigationPipeline.prepare_job_dir(job, file.file, file.filename or "upload")
    job_manager.submit(job, lambda j: investigation_pipeline.run(j, upload_path))
    return job.to_dict()


@app.get("/api/investigations/{job_id}")
def get_investigation(job_id: str):
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Investigation not found (the server may have restarted).")
    return job.to_dict()


@app.get("/api/scenarios")
def list_scenarios():
    """Lists available pre-packaged evaluation scenarios."""
    scenarios = []
    for p in FIXTURES_DIR.iterdir():
        gt_file = p / "ground_truth.json"
        if gt_file.exists():
            with open(gt_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                scenarios.append({
                    "id": data.get("scenario_id", p.name),
                    "title": data.get("title", p.name),
                    "description": data.get("description", ""),
                    "versions_count": len(data.get("versions", [])),
                })
    return {"scenarios": scenarios}


@app.get("/api/scenarios/{scenario_id}/tree", response_model=ProvenanceTree)
def get_scenario_tree(scenario_id: str):
    """
    Executes Steps 4 & 5 to build the directed provenance tree for a scenario.
    """
    scenario_path = FIXTURES_DIR / scenario_id
    gt_file = scenario_path / "ground_truth.json"
    if not gt_file.exists():
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found.")

    with open(gt_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    versions = []
    for item in data["versions"]:
        img_path = str(scenario_path / item["image_path"])
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

    tree = tree_builder.build_tree(versions, scenario_id=scenario_id)
    root_ver = next((v for v in versions if v.id == tree.root_id), versions[0] if versions else None)
    if root_ver:
        web_matches = discovery_orchestrator.vision_client.get_30_similar_web_matches(root_ver.image_path)
        verdict, confidence, explanation = compute_authenticity_verdict(root_ver.image_path, web_matches)
        tree.web_matches = web_matches
        tree.authenticity_verdict = verdict
        tree.verdict_confidence = confidence
        tree.verdict_explanation = explanation
    return tree


class PairwiseRequest(BaseModel):
    source: ImageVersion
    target: ImageVersion


@app.post("/api/analyze/pairwise", response_model=EdgeScore)
def analyze_pair(req: PairwiseRequest):
    """
    Step 4: Compares two versions and returns the 5-factor weighted score + Gemini multimodal diff.
    """
    score = pairwise_engine.compute_edge_score(req.source, req.target)
    gemini_res = gemini_inspector.inspect_pair(req.source, req.target, edge_metrics=score.edit_metrics)
    score.gemini_evidence = gemini_res
    return score


class DiscoverRequest(BaseModel):
    image_path: str
    caption: Optional[str] = None


@app.post("/api/search/discover", response_model=List[ImageVersion])
def discover_relatives(req: DiscoverRequest):
    """
    Step 3 (benchmark scenarios): returns the curated version pool for the demo fixtures.
    """
    return discovery_orchestrator.discover_relatives(req.image_path, req.caption)


class EndToEndPipelineRequest(BaseModel):
    image_path: str
    caption: Optional[str] = None


@app.post("/api/pipeline/end-to-end", response_model=ProvenanceTree)
def run_end_to_end_pipeline(req: EndToEndPipelineRequest):
    """
    Benchmark presets (Steps 1 -> 7) for the bundled fixture images:
    Input image -> Ingest -> Curated Discovery -> Pairwise Comparison -> Reconstructed Tree.
    """
    image_path = Path(req.image_path)
    if not image_path.is_absolute():
        image_path = BACKEND_DIR.parent / image_path
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {req.image_path}")
    path = str(image_path)

    forensic_analysis = gemini_inspector.inspect_single_image(path, req.caption)
    versions = discovery_orchestrator.discover_relatives(path, req.caption)
    tree = tree_builder.build_tree(versions, scenario_id="end_to_end_investigation")

    web_matches = discovery_orchestrator.vision_client.get_30_similar_web_matches(
        path, caption=req.caption, forensic_analysis=forensic_analysis
    )
    verdict, confidence, explanation = compute_authenticity_verdict(path, web_matches, forensic_analysis)
    tree.web_matches = web_matches
    tree.authenticity_verdict = verdict
    tree.verdict_confidence = confidence
    tree.verdict_explanation = explanation
    conf_pct = int(confidence * 100)
    tree.executive_summary = f"[{verdict} | {conf_pct}% CONFIDENCE] {explanation}"
    return tree


@app.post("/api/upload/investigate")
def upload_and_investigate(
    file: UploadFile = File(...),
    caption: Optional[str] = Form(None),
):
    """
    Synchronous variant of /api/investigations for scripts and the CLI:
    runs the full pipeline (always tracing lineage) and returns the provenance tree.
    """
    job = job_manager.create(file.filename or "upload", caption or None, force_trace=True)
    upload_path = InvestigationPipeline.prepare_job_dir(job, file.file, file.filename or "upload")
    try:
        return investigation_pipeline.run(job, upload_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Investigation failed: {e}")


@app.post("/api/upload/batch", response_model=ProvenanceTree)
def upload_batch_and_trace(
    files: List[UploadFile] = File(...),
):
    """
    Multi-Image Batch Upload:
    The user uploads multiple suspicious copies to trace who copied whom.
    """
    if len(files) < 1:
        raise HTTPException(status_code=400, detail="At least 1 file required.")

    batch_dir = UPLOADS_DIR / "batches" / uuid.uuid4().hex[:12]
    batch_dir.mkdir(parents=True, exist_ok=True)

    versions: List[ImageVersion] = []
    analyses: Dict[str, Any] = {}
    ela_urls: Dict[str, str] = {}

    for idx, f in enumerate(files):
        save_path = str(batch_dir / f"{idx}_{safe_filename(f.filename)}")
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)

        ela_save_path = compute_ela_map(save_path, str(batch_dir / f"ela_{idx}.png"))
        if ela_save_path:
            ela_urls[save_path] = to_url(ela_save_path)

        try:
            ingested_list = ingestion_pipeline.process_upload(
                save_path,
                platform_hint=f"Uploaded Copy #{idx + 1} ({f.filename})",
                caption_hint=f"File: {f.filename}",
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"{f.filename}: {e}")
        for v in ingested_list:
            v.image_url = to_url(v.image_path)
            v.url = v.image_url
            analysis = gemini_inspector.inspect_single_image(v.image_path, v.caption)
            analyses[v.image_path] = analysis
            versions.append(v)

    tree = tree_builder.build_tree(versions, scenario_id="batch_user_investigation")

    for node in tree.nodes.values():
        img_p = node.version.image_path
        if img_p in analyses:
            node.forensic_analysis = analyses[img_p]
            if analyses[img_p].get("tampering_detected"):
                node.tampering_flagged = True
                node.tampering_summary = analyses[img_p].get("forensic_details")
        if img_p in ela_urls:
            node.ela_path = ela_urls[img_p]

    primary_path = versions[0].image_path
    primary_analysis = analyses.get(primary_path) or {}
    tampered = [n for n in tree.nodes.values() if n.tampering_flagged]
    tree.authenticity_verdict = "MANIPULATED_DERIVATIVE" if tampered else "AUTHENTIC_ORIGINAL"
    tree.verdict_confidence = float(primary_analysis.get("confidence_score") or 0.6)
    tree.verdict_explanation = (
        f"{len(tampered)} of {len(versions)} uploaded copies show signs of manipulation. "
        f"Earliest/most credible copy: {tree.nodes[tree.root_id].version.platform}."
        if tampered else
        f"No uploaded copy was flagged as manipulated. Earliest/most credible copy: "
        f"{tree.nodes[tree.root_id].version.platform}."
    )
    conf_pct = int(tree.verdict_confidence * 100)
    tree.executive_summary = f"[{tree.authenticity_verdict} | {conf_pct}% CONFIDENCE] {tree.verdict_explanation}"
    return tree
