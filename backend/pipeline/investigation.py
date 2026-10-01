from __future__ import annotations
import os
import re
import shutil
import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from backend.engine.gemini_inspector import GeminiInspector
from backend.engine.models import (
    EdgeScore,
    GeminiPairInspection,
    ImageVersion,
    ProvenanceNode,
    ProvenanceTree,
    WebMatchItem,
)
from backend.engine.tree_builder import TreeBuilder
from backend.forensics.classifier import ForensicClassifier
from backend.forensics.ela import compute_ela_map
from backend.forensics.embedded_preview import EmbeddedPreviewCheck
from backend.forensics.trufor_detector import TruForDetector
from backend.search.region_search import RegionSourceSearch
from backend.ingestion.pipeline import IngestionPipeline
from backend.jobs import Job
from backend.search.match_verifier import MatchVerifier
from backend.search.page_scraper import PageScraper
from backend.search.reverse_search import ReverseImageSearch

UPLOADS_DIR = Path(__file__).resolve().parents[1] / "uploads"
JOBS_DIR = UPLOADS_DIR / "jobs"
UPLOADS_URL = "/static/uploads"

MAX_LINEAGE_VERSIONS = 6       # most similar web copies placed in the tree (besides the upload)
MAX_SIMILAR_SHOWN = 8          # most similar verified copies listed in the results
RELATED_SIMILARITY = 0.55      # verified as the same picture (possibly cropped/edited)
EDIT_DIFF_THRESHOLD = 0.003    # share of the frame that must differ to count as an edit


def safe_filename(name: str) -> str:
    base = os.path.basename(name or "upload")
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._") or "upload"
    return base[-80:]


def short_platform(name: Optional[str]) -> str:
    """'News: bbc.co.uk' -> 'bbc.co.uk' for use in sentences."""
    if not name:
        return "another site"
    return re.sub(r"^(?:Website|News|Fact-check|Institution|Photographer|Wallpaper/meme site):\s*", "", name)


def to_url(path: Optional[str]) -> Optional[str]:
    """Maps a file under backend/uploads to the URL it is served from."""
    if not path:
        return None
    try:
        rel = Path(path).resolve().relative_to(UPLOADS_DIR.resolve())
    except ValueError:
        return None
    return f"{UPLOADS_URL}/{rel.as_posix()}"


class InvestigationPipeline:
    """
    End-to-end investigation for one uploaded image, reported stage by stage on a Job:

      ingest -> forensics -> web_check -> regions -> verdict -> discovery -> provenance -> lineage -> report

    The verdict is built from explicit proof items (a differing earlier copy online, an embedded
    camera preview that no longer matches, a region found in a different photo, the TruFor
    model's localisation) rather than a single opaque score.
    """

    def __init__(
        self,
        gemini: GeminiInspector,
        tree_builder: TreeBuilder,
        ingestion: Optional[IngestionPipeline] = None,
    ):
        self.gemini = gemini
        self.tree_builder = tree_builder
        self.ingestion = ingestion or IngestionPipeline()
        self.classifier = ForensicClassifier()
        self.searcher = ReverseImageSearch()
        self.verifier = MatchVerifier()
        self.preview_check = EmbeddedPreviewCheck()
        self.trufor = TruForDetector()
        self.region_search = RegionSourceSearch(self.searcher, self.verifier)

    # ------------------------------------------------------------------ entry points

    @staticmethod
    def prepare_job_dir(job: Job, upload_stream, filename: str) -> str:
        job_dir = JOBS_DIR / job.id
        job_dir.mkdir(parents=True, exist_ok=True)
        save_path = job_dir / safe_filename(filename)
        with open(save_path, "wb") as out:
            shutil.copyfileobj(upload_stream, out)
        return str(save_path)

    def run(self, job: Job, upload_path: str) -> Dict[str, Any]:
        job_dir = Path(upload_path).parent
        ctx: Dict[str, Any] = {"job_dir": job_dir, "upload_path": upload_path}

        self._stage_ingest(job, ctx)
        self._stage_forensics(job, ctx)
        self._stage_web_check(job, ctx)
        self._stage_regions(job, ctx)
        self._stage_verdict(job, ctx)

        # Lineage is traced for every upload: an original image still has a spread history
        # worth showing, and the verdict only decides how the tree is interpreted.
        self._stage_discovery(job, ctx)
        self._stage_provenance(job, ctx)
        tree = self._stage_lineage(job, ctx)

        result = self._stage_report(job, ctx, tree)
        job.result = result
        return result

    # ------------------------------------------------------------------ stages

    def _stage_ingest(self, job: Job, ctx: Dict[str, Any]):
        job.start("ingest", "Reading metadata and fingerprinting…")
        upload_path = ctx["upload_path"]
        versions = self.ingestion.process_upload(
            upload_path, platform_hint="Your upload", caption_hint=job.caption
        )
        primary = versions[0]
        is_video = primary.image_path != upload_path
        analysis_source = primary.image_path

        # A bounded-size working copy keeps forensics, search and pairwise scoring fast.
        analysis_path = str(ctx["job_dir"] / "analysis.png")
        with Image.open(analysis_source) as im:
            im = im.convert("RGB")
            original_size = im.size
            im.thumbnail((1400, 1400))
            im.save(analysis_path, "PNG")

        digest = hashlib.sha1(open(upload_path, "rb").read()).hexdigest()
        ela_path = compute_ela_map(analysis_source, str(ctx["job_dir"] / "ela.png"))

        primary.id = f"upload_{digest[:10]}"
        primary.platform = "Your upload"
        primary.author = None
        primary.image_url = to_url(analysis_source if is_video else upload_path)
        primary.url = primary.image_url
        primary.dimensions = original_size
        primary.category = "upload"
        primary.caption = job.caption or None
        ctx.update(
            analysis_path=analysis_path,
            analysis_source=analysis_source,
            upload_version=primary,
            ela_url=to_url(ela_path) if ela_path else None,
            digest=digest,
        )
        exif_make = (primary.exif or {}).get("Make")
        job.preview.update(
            image_url=primary.image_url,
            ela_url=ctx["ela_url"],
            dimensions=list(original_size),
            phash=primary.phash,
            exif_camera=f"{exif_make} {(primary.exif or {}).get('Model', '')}".strip() if exif_make else None,
            c2pa_signed=bool(primary.c2pa and primary.c2pa.get("is_signed")),
            is_video=is_video,
        )
        job.finish(
            "ingest",
            f"{original_size[0]}×{original_size[1]} · pHash {primary.phash or 'n/a'} · "
            + ("camera EXIF present" if exif_make else "no camera EXIF"),
        )

    def _stage_forensics(self, job: Job, ctx: Dict[str, Any]):
        job.start("forensics", "Running Gemini vision inspection…")
        gemini = self.gemini.inspect_single_image(ctx["analysis_source"], job.caption)
        ctx["gemini"] = gemini
        extra_signals = []

        job.progress("forensics", "Checking the embedded camera preview…")
        preview = self.preview_check.check(ctx["upload_path"], str(ctx["job_dir"]))
        preview["preview_url"] = to_url(preview.get("preview_path"))
        preview["diff_url"] = to_url(preview.get("diff_path"))
        ctx["embedded_preview"] = preview
        if preview["status"] != "absent":
            extra_signals.append({
                "key": "embedded_preview",
                "name": "Embedded camera preview",
                "score": {"mismatch": 0.97, "cropped": 0.2}.get(preview["status"], 0.0),
                "weight": 1.0,
                "detail": preview["detail"],
            })

        job.progress("forensics", "Running the TruFor forgery-localisation model…")
        try:
            trufor = self.trufor.analyze(ctx["analysis_source"], str(ctx["job_dir"]))
        except Exception as e:  # never let the model take the whole pipeline down
            trufor = {"available": False, "error": str(e)}
            print(f"[TruFor] failed: {e}")
        trufor["map_url"] = to_url(trufor.get("map_path"))
        ctx["trufor"] = trufor
        if trufor.get("available"):
            extra_signals.append({
                "key": "trufor",
                "name": "TruFor forgery model",
                "score": trufor["score"],
                "weight": 0.9,
                "detail": (
                    f"Learned detector (camera-noise fingerprint + image content) rates this image "
                    f"{trufor['score'] * 100:.0f}% likely manipulated"
                    + (f" and localises {trufor['manipulated_area'] * 100:.1f}% of the frame."
                       if trufor["manipulated_area"] >= 0.003 else ".")
                ),
            })

        job.progress("forensics", "Running ELA, noise, clone, overlay & metadata checks…")
        overlay_path = str(ctx["job_dir"] / "suspicious_regions.png")
        classification = self.classifier.classify(
            ctx["analysis_path"], gemini, overlay_output_path=overlay_path, extra_signals=extra_signals
        )
        classification["overlay_url"] = to_url(classification.pop("overlay_path", None))
        classification["gemini_summary"] = gemini.get("visual_summary")
        classification["gemini_status"] = gemini.get("source")
        classification["ela_url"] = ctx["ela_url"]
        ctx["classification"] = classification
        job.classification = {**classification, "stage": "forensics"}

        notes = [f"Tamper probability {classification['tamper_probability'] * 100:.0f}%"]
        notes.append(f"TruFor {trufor['score'] * 100:.0f}% on {trufor['device'].upper()}" if trufor.get("available")
                     else "TruFor unavailable")
        notes.append({"absent": "no camera preview", "consistent": "camera preview matches",
                      "cropped": "camera preview shows a different crop",
                      "mismatch": "camera preview DIFFERS", "unreadable": "preview unreadable"}[preview["status"]])
        notes.append("Gemini used" if classification["gemini_used"] else "Gemini unavailable")
        job.finish("forensics", " · ".join(notes))

    def _stage_web_check(self, job: Job, ctx: Dict[str, Any]):
        job.start("web_check", "Uploading to reverse image search engines…")
        gemini = ctx.get("gemini") or {}
        hint = job.caption or (gemini.get("search_query") if gemini.get("source") == "gemini" else None)
        search = self.searcher.search(ctx["analysis_path"], query_hint=hint)
        candidates = search["results"]
        job.search = {
            "engines": search["engines"],
            "best_query": search["best_query"],
            "search_url": search["search_url"],
            "yandex_url": search["yandex_url"],
            "keyword_search_url": search["keyword_search_url"],
            "candidates": len(candidates),
            "verified": 0,
            "related": 0,
            "errors": search["errors"],
        }
        if not candidates:
            ctx["matches"] = []
            ctx["search"] = search
            job.finish("web_check", "No results from the search engines" +
                       (f" ({'; '.join(search['errors'])})" if search["errors"] else "."))
            return

        job.progress("web_check", f"Comparing {len(candidates)} candidates with your image…", 0.0)

        def on_progress(done: int, total: int):
            job.progress("web_check", f"Compared {done}/{total} candidates…", done / float(total))

        matches = self.verifier.verify_all(
            ctx["analysis_path"], candidates, str(ctx["job_dir"] / "matches"),
            on_progress=on_progress, fetch_full=False,
        )
        related = [m for m in matches if m.get("verified") and m.get("related")]
        ctx["matches"] = matches
        ctx["search"] = search
        job.search.update(verified=sum(1 for m in matches if m.get("verified")), related=len(related))
        job.finish(
            "web_check",
            f"{len(candidates)} candidates from {', '.join(search['engines']) or 'no engine'} · "
            f"{len(related)} verified copies of this image",
        )

    def _closest_differing_copy(self, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The strongest geometric match online whose content differs from the upload."""
        cands = [m for m in ctx.get("matches", []) if m.get("verified") and m.get("local_path")
                 and (m.get("feature_inliers") or 0) >= 25 and (m.get("diff_fraction") or 0) >= 0.01]
        return max(cands, key=lambda m: m.get("feature_inliers") or 0) if cands else None

    def _stage_regions(self, job: Job, ctx: Dict[str, Any]):
        job.start("regions", "Looking for faces and edited areas to search separately…")
        candidates = []
        ref = self._closest_differing_copy(ctx)
        ctx["reference_copy"] = ref
        if ref:
            for b in self.verifier.difference_boxes(ctx["analysis_path"], ref["local_path"]):
                candidates.append(("changed_vs_copy", b))
        for b in (ctx.get("trufor") or {}).get("boxes", []):
            candidates.append(("model", b))
        for b in ctx["classification"].get("suspicious_boxes", []):
            candidates.append(("forensic", b))
        # Faces are the most commonly swapped region, but only worth the searches when
        # something already looks off (otherwise every portrait would trigger them).
        suspicion = max(ctx["classification"]["tamper_probability"],
                        self._web_evidence(ctx.get("matches", []))["probability"])
        if suspicion >= 0.35:
            for b in self.region_search.detect_faces(ctx["analysis_path"]):
                candidates.append(("face", b))

        proposals = self.region_search.build_proposals(candidates)
        if not proposals:
            ctx["regions"] = []
            job.skip("regions", "Nothing to search separately — no edited areas or suspicious faces found.")
            return

        job.progress("regions", f"Reverse-searching {len(proposals)} region(s) on their own…")
        known = {m.get("image_url") or m.get("page_url") for m in ctx.get("matches", [])
                 if m.get("verified") and m.get("related")}
        regions = self.region_search.search_regions(
            ctx["analysis_path"], proposals, str(ctx["job_dir"] / "regions"), whole_image_match_urls=known
        )
        for r in regions:
            r["crop_url"] = to_url(r.pop("crop_path", None))
            for m in r["sources"]:
                m["thumbnail_url"] = to_url(m.get("local_path"))
        ctx["regions"] = regions
        found = [r for r in regions if r["sources"]]
        job.finish(
            "regions",
            (f"Found the source of {len(found)} of {len(regions)} region(s) in other images online"
             if found else f"Searched {len(regions)} region(s); no separate source found online"),
        )

    def _stage_verdict(self, job: Job, ctx: Dict[str, Any]) -> bool:
        job.start("verdict", "Collecting the proof…")
        classification = ctx["classification"]
        web = self._web_evidence(ctx.get("matches", []))
        ctx["web_evidence"] = web
        proof = self._collect_proof(ctx, web)
        ctx["proof"] = proof

        p_local = classification["tamper_probability"]
        p_web = web["probability"]
        p_final = 1.0 - (1.0 - p_local) * (1.0 - p_web)
        decisive = [p for p in proof if p["strength"] == "decisive"]
        if decisive:
            p_final = max(p_final, 0.97 if len(decisive) > 1 else 0.93)
        verdict = "TAMPERED" if p_final >= 0.5 else "ORIGINAL"

        signals = list(classification["signals"])
        signals.append({
            "key": "web",
            "name": "Comparison with copies found online",
            "score": round(p_web, 3),
            "weight": 1.0,
            "detail": web["detail"],
        })
        stages = list(classification["manipulation_stages"])
        if web["probability"] >= 0.5 and web.get("diff_fraction"):
            stages.append({
                "label": f"Content differs from online copies in {web['diff_fraction'] * 100:.1f}% of the frame",
                "evidence": web["detail"],
            })
        for r in ctx.get("regions", []):
            if r["sources"]:
                stages.append({
                    "label": f"{r['label']} was pasted in from another image",
                    "evidence": f"Found in a different picture at {short_platform(r['sources'][0].get('platform'))}.",
                })
        classification.update(
            verdict=verdict,
            tamper_probability=round(p_final, 3),
            confidence=round(p_final if verdict == "TAMPERED" else 1.0 - p_final, 3),
            signals=signals,
            manipulation_stages=stages,
            web_evidence=web,
            proof_count={"decisive": len(decisive), "total": len(proof)},
        )
        job.classification = {**classification, "stage": "verdict"}
        if verdict == "TAMPERED":
            msg = (f"EDITED — {len(decisive)} piece(s) of direct proof" if decisive
                   else f"EDITED — {classification['confidence'] * 100:.0f}% confidence (no direct proof)")
        else:
            msg = f"No edit proven — {classification['confidence'] * 100:.0f}% confidence"
        job.finish("verdict", msg)
        return verdict == "TAMPERED"

    def _collect_proof(self, ctx: Dict[str, Any], web: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evidence that the image was edited, strongest first.
          decisive   — a reference shows different content (earlier copy, camera preview, pasted source)
          strong     — a trained detector localises a manipulation
          supporting — heuristic signals
        """
        proof: List[Dict[str, Any]] = []
        job_dir = ctx["job_dir"]

        ref = ctx.get("reference_copy") or self._closest_differing_copy(ctx)
        if ref and web["probability"] >= 0.5:
            diff_path = str(job_dir / "diff_vs_copy.png")
            ok = self.verifier.render_difference(ctx["analysis_path"], ref["local_path"], diff_path)
            diff_url = to_url(diff_path) if ok else None
            proof.append({
                "key": "web_copy",
                "strength": "decisive" if web["differing"] >= 2 and not web["identical"] else "strong",
                "title": "Differs from the same photo published online",
                "detail": (f"{web['detail']} The outlined area is what changed compared with the copy on "
                           f"{short_platform(ref.get('platform'))}."),
                "images": [i for i in [{"url": diff_url, "label": "Your image — changed area outlined"},
                                       {"url": to_url(ref.get("local_path")), "label": f"Copy on {short_platform(ref.get('platform'))}"}]
                           if i["url"]],
                "links": [{"url": ref.get("page_url"), "label": "Open that copy"}] if ref.get("page_url") else [],
            })

        preview = ctx.get("embedded_preview") or {}
        if preview.get("status") == "mismatch":
            proof.append({
                "key": "embedded_preview",
                "strength": "decisive",
                "title": "The file's own camera preview shows different content",
                "detail": preview["detail"],
                "images": [i for i in [{"url": preview.get("preview_url"), "label": "Preview stored in the file (before editing)"},
                                       {"url": preview.get("diff_url"), "label": "Preview vs. visible image"}] if i["url"]],
                "links": [],
            })
        elif preview.get("status") == "cropped":
            proof.append({
                "key": "embedded_preview",
                "strength": "supporting",
                "title": "Cropped after it was taken",
                "detail": preview["detail"],
                "images": ([{"url": preview.get("preview_url"), "label": "Original framing (embedded preview)"}]
                           if preview.get("preview_url") else []),
                "links": [],
            })

        for r in ctx.get("regions", []):
            if not r["sources"]:
                continue
            src = r["sources"][0]
            proof.append({
                "key": "region_source",
                "strength": "decisive",
                "title": f"{r['label']} comes from a different photo",
                "detail": (f"This part of your image appears, pixel for pixel, in a different picture on "
                           f"{short_platform(src.get('platform'))}, while the rest of that picture does not match yours. "
                           "It was cut from there and pasted in."),
                "images": [i for i in [{"url": r.get("crop_url"), "label": "Region in your image"},
                                       {"url": src.get("thumbnail_url"), "label": f"Source on {short_platform(src.get('platform'))}"}]
                           if i["url"]],
                "links": [{"url": m.get("page_url"), "label": short_platform(m.get("platform"))} for m in r["sources"] if m.get("page_url")],
            })

        trufor = ctx.get("trufor") or {}
        if trufor.get("available") and trufor["score"] >= 0.5:
            proof.append({
                "key": "trufor",
                "strength": "strong" if trufor["score"] >= 0.8 else "supporting",
                "title": "Forgery model localises an edited area",
                "detail": (f"TruFor rates the image {trufor['score'] * 100:.0f}% likely manipulated, based on "
                           "inconsistencies in the camera-noise fingerprint. Red marks where it localises the edit."),
                "images": [{"url": trufor.get("map_url"), "label": "TruFor localisation map"}] if trufor.get("map_url") else [],
                "links": [],
            })

        classification = ctx["classification"]
        for sig in classification["signals"]:
            if sig["key"] in ("trufor", "embedded_preview", "gemini") or sig["score"] < 0.5:
                continue
            proof.append({
                "key": sig["key"],
                "strength": "supporting",
                "title": sig["name"],
                "detail": sig["detail"],
                "images": ([{"url": classification.get("overlay_url"), "label": "Suspicious regions"}]
                           if classification.get("overlay_url") and sig["key"] in ("ela", "noise", "copy_move", "overlay")
                           else []),
                "links": [],
            })
        gem = ctx.get("gemini") or {}
        if gem.get("source") == "gemini" and gem.get("tampering_detected"):
            proof.append({
                "key": "gemini",
                "strength": "supporting",
                "title": "Gemini vision analysis",
                "detail": gem.get("forensic_details") or "",
                "images": [],
                "links": [],
            })
        order = {"decisive": 0, "strong": 1, "supporting": 2}
        return sorted(proof, key=lambda p: order[p["strength"]])

    def _stage_discovery(self, job: Job, ctx: Dict[str, Any]):
        related = [m for m in ctx.get("matches", []) if m.get("verified") and m.get("related")]
        related.sort(key=lambda m: (m.get("similarity", 0), m.get("feature_inliers", 0)), reverse=True)
        chosen = self._diverse_selection(related, MAX_LINEAGE_VERSIONS)
        ctx["lineage_matches"] = chosen
        if not chosen:
            job.skip("discovery", "No verified copies online — the original source could not be located.")
            return
        job.start("discovery", f"Fetching full-resolution images for {len(chosen)} copies…", 0.0)

        def on_progress(done: int, total: int):
            job.progress("discovery", f"Fetched {done}/{total} full-resolution copies…", done / float(total))

        self.verifier.upgrade_full(ctx["analysis_path"], chosen, on_progress=on_progress)
        full = sum(1 for m in chosen if m.get("full_resolution"))
        platforms = sorted({m["platform"] for m in chosen})
        job.finish("discovery", f"{len(chosen)} copies across {len(platforms)} platform(s) · {full} at full resolution")

    def _stage_provenance(self, job: Job, ctx: Dict[str, Any]):
        chosen = ctx.get("lineage_matches") or []
        if not chosen:
            job.skip("provenance", "Nothing to date.")
            return
        job.start("provenance", f"Reading publication dates from {len(chosen)} pages…", 0.0)
        done = [0]

        def scrape(m: Dict[str, Any]):
            info = PageScraper.scrape_page(m["page_url"]) if m.get("page_url") else {}
            done[0] += 1
            job.progress("provenance", f"Checked {done[0]}/{len(chosen)} pages…", done[0] / float(len(chosen)))
            return m, info

        with ThreadPoolExecutor(max_workers=8) as pool:
            for m, info in pool.map(scrape, chosen):
                m["published_at"] = info.get("published_at")
                m["page_reachable"] = info.get("reachable")
                if info.get("headline") and not m.get("title"):
                    m["title"] = info["headline"]
                m["page_caption"] = info.get("caption")
        dated = sum(1 for m in chosen if m.get("published_at"))
        job.finish("provenance", f"{dated}/{len(chosen)} pages expose a publication date")

    def _stage_lineage(self, job: Job, ctx: Dict[str, Any]) -> ProvenanceTree:
        job.start("lineage", "Scoring every pair of versions…")
        upload: ImageVersion = ctx["upload_version"]
        chosen = ctx.get("lineage_matches") or []
        if not chosen:
            if ctx["classification"]["verdict"] == "TAMPERED":
                tree = self._inferred_original_tree(ctx)
                job.finish("lineage", "No online copies found — showing the inferred original → your upload.")
            else:
                tree = self._single_node_tree(ctx)
                job.finish("lineage", "No online copies found — your upload is the only known version.")
            return tree

        # Pairwise scoring needs every version as a local file; the upload uses its analysis copy.
        upload_for_tree = upload.model_copy(update={"image_path": ctx["analysis_path"]})
        upload_for_tree.timestamp = datetime.utcnow()
        upload_for_tree.source_trust_score = 0.3 if ctx["classification"]["verdict"] == "TAMPERED" else 0.5

        versions: List[ImageVersion] = [upload_for_tree]
        for i, m in enumerate(chosen):
            versions.append(ImageVersion(
                id=f"web_{hashlib.sha1((m.get('downloaded_from') or m.get('page_url') or '').encode()).hexdigest()[:8]}_{i:02d}",
                platform=m["platform"],
                author=m.get("domain"),
                url=m.get("page_url"),
                timestamp=m.get("published_at"),
                caption=m.get("title") or m.get("page_caption"),
                image_path=m["local_path"],
                dimensions=tuple(m["dimensions"]) if m.get("dimensions") else None,
                source_trust_score=float(m.get("trust", 0.5)),
                image_url=to_url(m["local_path"]),
                source_image_url=m.get("image_url"),
                domain=m.get("domain"),
                category=m.get("category"),
            ))

        exclude = {upload_for_tree.id}
        tree = self.tree_builder.build_tree(
            versions,
            scenario_id=f"investigation_{job.id}",
            exclude_root_ids=exclude,
            live_inspection_ids={upload_for_tree.id},
            connect_all=True,
            leaf_ids={upload_for_tree.id},
        )
        job.progress("lineage", "Locating where the content was altered…")

        # Attach every node and hop to the real files and mark where content changed.
        for node in tree.nodes.values():
            v = node.version
            node.kind = "upload" if v.id == upload_for_tree.id else "web_copy"
            if node.kind == "upload":
                v.image_path = upload.image_path
                v.image_url = upload.image_url
                node.ela_path = ctx["ela_url"]
                node.forensic_analysis = ctx.get("gemini")
                if ctx["classification"]["verdict"] == "TAMPERED":
                    node.tampering_flagged = True
                    node.tampering_summary = self._tamper_summary(ctx["classification"])
        path_by_id = {v.id: v.image_path for v in versions}
        for edge in tree.edges:
            src_p, tgt_p = path_by_id.get(edge.source_id), path_by_id.get(edge.target_id)
            if not (src_p and tgt_p):
                continue
            try:
                cmp = self.verifier.compare_paths(tgt_p, src_p)
            except Exception:
                continue
            node = tree.nodes[edge.target_id]
            node.diff_fraction = cmp.get("diff_fraction")
            if cmp.get("diff_fraction") is not None and cmp["diff_fraction"] >= EDIT_DIFF_THRESHOLD:
                node.altered_here = True
                is_tampered_upload = node.kind == "upload" and ctx["classification"]["verdict"] == "TAMPERED"
                if is_tampered_upload or (node.kind != "upload" and cmp["diff_fraction"] >= 0.01):
                    node.tampering_flagged = True
                    node.tampering_summary = node.tampering_summary or (
                        f"Content differs from its parent ({tree.nodes[edge.source_id].version.platform}) "
                        f"in {cmp['diff_fraction'] * 100:.1f}% of the frame."
                    )
            if node.kind == "upload":
                diff_path = str(ctx["job_dir"] / "diff_vs_parent.png")
                if self.verifier.render_difference(tgt_p, src_p, diff_path):
                    ctx["diff_url"] = to_url(diff_path)

        tree.upload_node_id = upload_for_tree.id
        upload_node = tree.nodes[upload_for_tree.id]
        upload_node.version.image_path = upload.image_path
        job.finish(
            "lineage",
            f"Root: {tree.nodes[tree.root_id].version.platform} · {len(tree.edges)} hop(s) · "
            f"your upload is {upload_node.depth} stage(s) from the earliest copy found",
        )
        return tree

    def _stage_report(self, job: Job, ctx: Dict[str, Any], tree: ProvenanceTree) -> Dict[str, Any]:
        job.start("report", "Compiling report…")
        classification = ctx["classification"]
        classification["diff_url"] = ctx.get("diff_url")
        tree.classification = classification
        tree.proof = ctx.get("proof", [])
        tree.regions = [
            {**{k: v for k, v in r.items() if k != "sources"},
             "sources": [{"page_url": m.get("page_url"), "platform": m.get("platform"), "title": m.get("title"),
                          "thumbnail_url": m.get("thumbnail_url"), "source_image_url": m.get("image_url"),
                          "similarity": m.get("similarity")} for m in r["sources"]]}
            for r in ctx.get("regions", [])
        ]
        tree.upload_node_id = tree.upload_node_id or ctx["upload_version"].id

        # Path from the origin to the upload, and the hop count ("stages")
        path: List[str] = []
        cursor: Optional[str] = tree.upload_node_id
        while cursor and cursor in tree.nodes and cursor not in path:
            path.append(cursor)
            cursor = tree.nodes[cursor].parent_id
        tree.lineage_path = list(reversed(path))
        tree.stage_count = max(0, len(tree.lineage_path) - 1)

        tree.web_matches = self._web_match_items(ctx)
        tree.platforms = self._platform_summary(ctx)
        # "Edited" means the lineage shows content changing at a copy on that platform
        # (relative to its parent), not merely that it differs from the upload.
        edited_by_platform: Dict[str, int] = {}
        for node in tree.nodes.values():
            if node.kind == "web_copy" and node.altered_here:
                edited_by_platform[node.version.platform] = edited_by_platform.get(node.version.platform, 0) + 1
        for g in tree.platforms:
            g["altered"] = edited_by_platform.get(g["platform"], 0)
        search = ctx.get("search") or {}
        tree.search_summary = {**(job.search or {}), "query": search.get("keyword_query") or search.get("best_query")}

        verdict = classification["verdict"]
        conf = classification["confidence"]
        tree.authenticity_verdict = "MANIPULATED_DERIVATIVE" if verdict == "TAMPERED" else "AUTHENTIC_ORIGINAL"
        tree.verdict_confidence = conf
        tree.verdict_explanation = self._verdict_explanation(classification, tree)
        tree.executive_summary = f"[{verdict} | {int(conf * 100)}% CONFIDENCE] {tree.verdict_explanation}"
        job.finish("report", "Done")
        return tree.model_dump(mode="json")

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _web_evidence(matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        strong = [m for m in matches if m.get("verified") and m.get("similarity", 0) >= 0.8
                  and (m.get("feature_inliers") or 0) >= 25 and m.get("diff_fraction") is not None]
        if not strong:
            return {"probability": 0.0, "copies": 0, "identical": 0, "differing": 0, "diff_fraction": None,
                    "detail": "No near-identical copy online to compare against."}
        identical = [m for m in strong if m["diff_fraction"] < EDIT_DIFF_THRESHOLD]
        differing = [m for m in strong if 0.01 <= m["diff_fraction"] <= 0.6]
        diffs = sorted(m["diff_fraction"] for m in differing)
        median_diff = diffs[len(diffs) // 2] if diffs else None

        if differing and not identical:
            # Every copy online differs from the upload in the same kind of way.
            p = 0.9 if len(differing) >= 2 else 0.65
            detail = (f"Your image differs from all {len(differing)} matching copies found online "
                      f"(median {median_diff * 100:.1f}% of the frame changed).")
        elif differing and identical:
            # Two variants circulate; which one came first is decided by the lineage stage.
            # The more copies match the upload exactly, the more likely it is the common original.
            p = 0.6 * len(differing) / float(len(differing) + len(identical))
            detail = (f"Two variants circulate online: {len(identical)} {'copy' if len(identical) == 1 else 'copies'} "
                      f"identical to yours and {len(differing)} that differ in a small region "
                      f"({median_diff * 100:.1f}% of the frame).")
        else:
            p = 0.0
            detail = (f"Identical to {len(identical)} {'copy' if len(identical) == 1 else 'copies'} "
                      "found online — no local edits relative to them.")
        return {"probability": p, "copies": len(strong), "identical": len(identical),
                "differing": len(differing), "diff_fraction": median_diff, "detail": detail}

    @staticmethod
    def _diverse_selection(related: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """Keeps the strongest matches but at most two per domain, so the tree spans platforms."""
        per_domain: Dict[str, int] = {}
        chosen = []
        for m in related:
            if not m.get("local_path"):
                continue
            d = m.get("domain") or ""
            if per_domain.get(d, 0) >= 2:
                continue
            per_domain[d] = per_domain.get(d, 0) + 1
            chosen.append(m)
            if len(chosen) >= limit:
                break
        return chosen

    @staticmethod
    def _tamper_summary(classification: Dict[str, Any]) -> str:
        stages = [s["label"] for s in classification.get("manipulation_stages", [])]
        if stages:
            return "; ".join(dict.fromkeys(stages)) + "."
        top = max(classification["signals"], key=lambda s: s["score"] * s.get("weight", 1.0))
        return top["detail"]

    def _single_node_tree(self, ctx: Dict[str, Any]) -> ProvenanceTree:
        upload: ImageVersion = ctx["upload_version"]
        node = ProvenanceNode(version=upload, depth=0, is_root=True, kind="upload",
                              ela_path=ctx["ela_url"], forensic_analysis=ctx.get("gemini"))
        return ProvenanceTree(root_id=upload.id, earliest_date=upload.timestamp,
                              nodes={upload.id: node}, upload_node_id=upload.id)

    def _inferred_original_tree(self, ctx: Dict[str, Any]) -> ProvenanceTree:
        upload: ImageVersion = ctx["upload_version"]
        original = ImageVersion(
            id="inferred_original",
            platform="Original (not found online)",
            caption="The unedited source was not located on the web. Its existence is inferred "
                    "from the manipulation evidence in your upload.",
            image_path="",
            category="inferred",
        )
        root = ProvenanceNode(version=original, depth=0, is_root=True, kind="inferred_original")
        child = ProvenanceNode(
            version=upload, depth=1, parent_id=original.id, kind="upload",
            ela_path=ctx["ela_url"], forensic_analysis=ctx.get("gemini"),
            tampering_flagged=True, altered_here=True,
            tampering_summary=self._tamper_summary(ctx["classification"]),
        )
        stages = [s["label"] for s in ctx["classification"].get("manipulation_stages", [])]
        edge = EdgeScore(
            source_id=original.id,
            target_id=upload.id,
            total_score=ctx["classification"]["tamper_probability"],
            gemini_evidence=GeminiPairInspection(
                visual_diff_summary="; ".join(stages) or "Manipulation inferred from forensic signals.",
                edit_types=[s["label"] for s in ctx["classification"].get("manipulation_stages", [])],
                claim_drift_reasoning="No source caption available to compare.",
                tampering_detected=True,
                tampering_details=child.tampering_summary,
            ),
        )
        return ProvenanceTree(root_id=original.id, nodes={original.id: root, upload.id: child},
                              edges=[edge], upload_node_id=upload.id)

    @staticmethod
    def _web_match_items(ctx: Dict[str, Any]) -> List[WebMatchItem]:
        upload: ImageVersion = ctx["upload_version"]
        q_area = (upload.dimensions[0] * upload.dimensions[1]) if upload.dimensions else 0
        items = []
        related = [x for x in ctx.get("matches", []) if x.get("verified") and x.get("related")]
        related.sort(key=lambda x: (x.get("similarity", 0), x.get("feature_inliers") or 0), reverse=True)
        for i, m in enumerate(related[:MAX_SIMILAR_SHOWN]):
            dims = tuple(m["dimensions"]) if m.get("dimensions") else None
            published = m.get("published_at")
            rel = "ALTERED_DERIVATIVE" if (m.get("diff_fraction") or 0) >= 0.01 else (
                "SAME_IMAGE" if m.get("related") else "VISUALLY_SIMILAR")
            if m.get("category") == "fact_check":
                rel = "FACT_CHECK_DEBUNK"
            items.append(WebMatchItem(
                match_id=f"match_{i + 1:02d}",
                image_url=m.get("image_url") or m.get("thumb_url") or "",
                thumbnail_url=to_url(m.get("local_path")) or m.get("thumb_url"),
                source_image_url=m.get("image_url"),
                page_url=m.get("page_url"),
                page_title=m.get("title"),
                platform=m.get("platform") or "Web",
                domain=m.get("domain"),
                category=m.get("category"),
                match_type=m.get("match_type", "VISUALLY_SIMILAR"),
                similarity_score=float(m.get("similarity", 0.0)),
                published_at=published,
                dimensions=dims,
                is_earlier_than_query=bool(published),
                is_higher_res_than_query=bool(dims and q_area and dims[0] * dims[1] > q_area * 1.05),
                is_broader_crop_than_query=bool(m.get("scale_ratio") and m.get("overlap_fraction", 0) > 0.95
                                                and not m.get("same_aspect")),
                verdict_relationship=rel,
                diff_fraction=m.get("diff_fraction"),
            ))
        return items

    @staticmethod
    def _platform_summary(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        groups: Dict[str, Dict[str, Any]] = {}
        for m in ctx.get("matches", []):
            if not (m.get("verified") and m.get("related")):
                continue
            g = groups.setdefault(m["platform"], {
                "platform": m["platform"], "category": m.get("category"), "count": 0,
                "altered": 0, "first_seen": None, "urls": [],
            })
            g["count"] += 1
            if m.get("published_at") and (g["first_seen"] is None or m["published_at"] < g["first_seen"]):
                g["first_seen"] = m["published_at"]
            if m.get("page_url") and len(g["urls"]) < 5:
                g["urls"].append(m["page_url"])
        return sorted(groups.values(), key=lambda g: (-g["count"], g["platform"]))

    @staticmethod
    def _verdict_explanation(classification: Dict[str, Any], tree: ProvenanceTree) -> str:
        verdict = classification["verdict"]
        top = sorted(
            (s for s in classification["signals"] if s["score"] >= 0.35),
            key=lambda s: s["score"] * s.get("weight", 1.0), reverse=True,
        )[:3]
        evidence = " ".join(f"{s['name']}: {s['detail']}" for s in top)
        if verdict == "TAMPERED":
            root = tree.nodes.get(tree.root_id)
            if root and root.kind == "web_copy":
                origin = f" Earliest/most credible copy found: {root.version.platform}"
                if root.version.timestamp:
                    origin += f" ({root.version.timestamp.strftime('%d %b %Y')})"
                origin += f"; your upload is {tree.stage_count} stage(s) downstream."
            else:
                origin = " The unedited original could not be located online."
            return f"Digital manipulation detected. {evidence}{origin}".strip()
        if not top:
            evidence = "No forensic signal crossed its threshold."
        return f"No manipulation detected. {evidence} Forensic checks are heuristic; absence of evidence is not proof of authenticity.".strip()
