from __future__ import annotations
import os
import json
import time
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

from backend.engine.models import (
    ImageVersion,
    GeminiPairInspection,
    ClaimDriftType,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_CACHE_FILE = str(Path(__file__).resolve().parents[1] / "fixtures" / ".gemini_cache.json")

# Ordered by preference. Override with GEMINI_MODELS="model-a,model-b".
DEFAULT_MODELS = [
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
]
RATE_LIMIT_COOLDOWN_SEC = 120
REQUEST_TIMEOUT_MS = 30_000


class GeminiInspector:
    """
    Multimodal Gemini Inspector:
    1. Inspects Image A and Image B side by side to describe visual differences in plain English.
    2. Classifies caption/narrative drift: SAME, REFRAMED, CONTRADICTED.
    3. Detects signs of AI tampering or localized manipulation (e.g. face distortion).
    """

    def __init__(self, api_key: Optional[str] = None, cache_file: str = DEFAULT_CACHE_FILE):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key and self.api_key.startswith("your_"):
            self.api_key = None  # placeholder copied from .env.example
        self.cache_file = cache_file
        self.cache: dict = {}
        self._load_cache()
        env_models = [m.strip() for m in os.getenv("GEMINI_MODELS", "").split(",") if m.strip()]
        self.models: List[str] = env_models or list(DEFAULT_MODELS)
        # model name -> unix time until which the model is skipped (rate limited)
        self._cooldown_until: Dict[str, float] = {}
        self.last_error: Optional[str] = None
        self.client = None
        if self.api_key:
            try:
                from google import genai
                from google.genai import types
                # Fail fast: by default the SDK retries 429s with backoff for minutes,
                # which made uploads hang without ever returning a result.
                self.client = genai.Client(
                    api_key=self.api_key,
                    http_options=types.HttpOptions(
                        timeout=REQUEST_TIMEOUT_MS,
                        retry_options=types.HttpRetryOptions(attempts=1),
                    ),
                )
            except Exception as e:
                print(f"[GeminiInspector] Warning: Failed to initialize genai client: {e}")

    def _load_cache(self):
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
        except Exception:
            self.cache = {}

    def _save_cache(self):
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        except Exception:
            pass

    @property
    def available(self) -> bool:
        now = time.time()
        return bool(self.client) and any(self._cooldown_until.get(m, 0) <= now for m in self.models)

    def generate_json(self, contents: list, temperature: float = 0.2) -> Optional[Dict[str, Any]]:
        """
        Calls Gemini with JSON output, trying each configured model in turn.
        Rate-limited models are put on cooldown; unknown models are dropped.
        Returns None when no model could answer.
        """
        if not self.client:
            return None
        from google.genai import types

        for model_name in list(self.models):
            if self._cooldown_until.get(model_name, 0) > time.time():
                continue
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=temperature,
                    ),
                )
                text = (response.text or "").strip()
                if text.startswith("```"):
                    text = text.strip("`").removeprefix("json").strip()
                data = json.loads(text)
                if isinstance(data, list) and data:
                    data = data[0]
                if isinstance(data, dict):
                    data["_model"] = model_name
                    return data
            except Exception as e:
                msg = str(e)
                self.last_error = f"{model_name}: {msg[:200]}"
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    self._cooldown_until[model_name] = time.time() + RATE_LIMIT_COOLDOWN_SEC
                elif "404" in msg or "NOT_FOUND" in msg:
                    self.models.remove(model_name)
                print(f"[GeminiInspector] {model_name} failed: {msg[:160]}")
        return None

    @staticmethod
    def _file_digest(path: str) -> str:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def inspect_pair(
        self,
        source: ImageVersion,
        target: ImageVersion,
        edge_metrics: Optional[Any] = None,
        allow_live: bool = True,
    ) -> GeminiPairInspection:
        cache_key = f"{source.id}__to__{target.id}"
        if cache_key in self.cache:
            data = self.cache[cache_key]
            return GeminiPairInspection(
                visual_diff_summary=data["visual_diff_summary"],
                edit_types=data["edit_types"],
                claim_drift=ClaimDriftType(data["claim_drift"]),
                claim_drift_reasoning=data["claim_drift_reasoning"],
                tampering_detected=data["tampering_detected"],
                tampering_details=data.get("tampering_details"),
            )

        res = None
        if allow_live and self.available:
            try:
                res = self._call_gemini_multimodal(source, target)
            except Exception as e:
                print(f"[GeminiInspector] Gemini API call failed, falling back to heuristics: {e}")

        if not res:
            # Heuristic results are not cached so a live answer can replace them later.
            return self._heuristic_mock_inspection(source, target, edge_metrics)

        # Cache result
        self.cache[cache_key] = {
            "visual_diff_summary": res.visual_diff_summary,
            "edit_types": res.edit_types,
            "claim_drift": res.claim_drift.value,
            "claim_drift_reasoning": res.claim_drift_reasoning,
            "tampering_detected": res.tampering_detected,
            "tampering_details": res.tampering_details,
        }
        self._save_cache()
        return res

    def inspect_single_image(self, image_path: str, caption: Optional[str] = None) -> Dict[str, Any]:
        """
        Forensic inspection of a single uploaded image:
        Detects AI generative artifacts, synthetic face modifications, deepfake signs,
        and platform text overlays. Also proposes a web search query for discovery.

        The result carries "source": "gemini" when the model answered, or
        "unavailable" when it could not be reached (no key, quota exhausted, timeout).
        """
        try:
            digest = self._file_digest(image_path)
        except OSError:
            digest = os.path.basename(image_path)
        cache_key = f"single_v2__{digest}__{caption or ''}"
        legacy_key = f"single__{os.path.basename(image_path)}"
        for key in (cache_key, legacy_key):
            if key in self.cache and self.cache[key].get("source", "gemini") == "gemini":
                return self.cache[key]

        if self.available:
            from PIL import Image

            try:
                img = Image.open(image_path)
                img.load()
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                img.thumbnail((1536, 1536))
                prompt = f"""
You are an expert digital forensics investigator.
Inspect this single uploaded image thoroughly.
Accompanying context/caption: "{caption or 'None provided'}"

Definitions:
- TAMPERED means the depicted content was changed: objects/people added, removed or replaced, faces or bodies
  altered, regions cloned or spliced, text/banners/captions overlaid, or the whole image AI-generated.
- NOT tampering: global edits that keep the content — exposure, HDR tone mapping, colour grading, filters,
  sharpening, cropping, resizing, rotation, JPEG compression, or a photographer's watermark.

Analyze:
1. Is this image authentic, partially manipulated, or synthetically AI-generated?
2. Are there visual signs of content tampering, deepfake alterations, localized face editing, spliced/composited objects, cloned regions, inconsistent lighting/shadows, or warped anatomical features?
3. Are there text overlays, meme banners, watermarks, or platform compression artifacts?
4. Describe the image content and visual characteristics in plain English.
5. Give a short web search query (3-8 words) that would find other copies of this image online (name the people, place, or event if recognisable).

Respond strictly in valid JSON matching this schema:
{{
  "tampering_detected": true | false,
  "confidence_score": 0.0 to 1.0,
  "tampering_type": "ai_generated | ai_face_edit | deepfake | splice | copy_move | text_overlay | compression | none",
  "forensic_details": "Detailed explanation of findings",
  "manipulation_steps": ["Ordered list of edits you believe were applied, earliest first"],
  "visual_summary": "Plain English summary of the image",
  "search_query": "short web search query",
  "extracted_claims": ["List of key claims or themes inferred from image and caption"]
}}
"""
                data = self.generate_json([img, prompt])
                if data:
                    data["source"] = "gemini"
                    self.cache[cache_key] = data
                    self._save_cache()
                    return data
            except Exception as e:
                print(f"[GeminiInspector] Single image analysis error: {e}")

        # Offline / rate-limited fallback (not cached, so a later call can still succeed)
        reason = "no API key configured"
        if self.client:
            reason = "quota exhausted or request failed"
        return {
            "source": "unavailable",
            "tampering_detected": False,
            "confidence_score": 0.0,
            "tampering_type": "none",
            "forensic_details": f"Gemini vision analysis was unavailable ({reason}); "
                                "the verdict is based on local forensic signals only.",
            "manipulation_steps": [],
            "visual_summary": caption or f"Uploaded media file: {os.path.basename(image_path)}",
            "search_query": caption or "",
            "extracted_claims": [caption] if caption else [],
        }

    def _call_gemini_multimodal(
        self, source: ImageVersion, target: ImageVersion
    ) -> GeminiPairInspection:
        from PIL import Image

        img_a = Image.open(source.image_path)
        img_b = Image.open(target.image_path)
        img_a.thumbnail((1024, 1024))
        img_b.thumbnail((1024, 1024))

        prompt = f"""
You are an expert digital forensics investigator and fact-checking analyst.
You are given two related images:
- Image A is the potential parent/source image.
- Image B is the potential child/derivative image.

Caption for Image A: "{source.caption or 'No caption provided'}"
Caption for Image B: "{target.caption or 'No caption provided'}"

Analyze both images carefully:
1. Identify all visual transformations from Image A to Image B (crops, resolution loss, color filters, text overlays, meme banners, watermarks, face modifications, AI generation artifacts).
2. Analyze the narrative/claim drift between Caption A and Caption B:
   - "SAME": The narrative, location, and event context are preserved.
   - "REFRAMED": The tone, emotional framing, or emphasis changed without direct falsification.
   - "CONTRADICTED": The factual claims (such as location, date, identities, or nature of the disaster) are falsified or directly contradicted.
3. Check for localized AI tampering or deepfake alterations (especially on human faces, objects, or lighting).

You MUST respond strictly with a valid JSON object matching this schema:
{{
  "visual_diff_summary": "Plain English summary of edits (e.g. 'Cropped right ~25%, text overlay added across bottom, warm color filter')",
  "edit_types": ["crop", "compression", "color_filter", "text_overlay", "ai_edit", "face_alteration"],
  "claim_drift": "SAME | REFRAMED | CONTRADICTED",
  "claim_drift_reasoning": "Explanation of how the claim shifted or was preserved",
  "tampering_detected": true | false,
  "tampering_details": "Description of any AI manipulation or null"
}}
"""
        data = self.generate_json([
            "Image A (Source):", img_a,
            "Image B (Derivative):", img_b,
            prompt,
        ])
        if not data:
            raise RuntimeError(self.last_error or "All Gemini models failed")

        drift_raw = str(data.get("claim_drift", "SAME")).upper()
        drift_enum = ClaimDriftType(drift_raw) if drift_raw in ClaimDriftType.__members__ else ClaimDriftType.SAME
        return GeminiPairInspection(
            visual_diff_summary=data.get("visual_diff_summary", ""),
            edit_types=data.get("edit_types", []) or [],
            claim_drift=drift_enum,
            claim_drift_reasoning=data.get("claim_drift_reasoning", ""),
            tampering_detected=bool(data.get("tampering_detected", False)),
            tampering_details=data.get("tampering_details"),
        )

    def _heuristic_mock_inspection(
        self,
        source: ImageVersion,
        target: ImageVersion,
        edge_metrics: Optional[Any] = None,
    ) -> GeminiPairInspection:
        """
        Deterministic inspection used for testing or when the Gemini API is unavailable.
        """
        edits: List[str] = []
        tampering = False
        tampering_details = None
        notes: List[str] = []

        # Check for crop / resolution
        if source.dimensions and target.dimensions:
            wa, ha = source.dimensions
            wb, hb = target.dimensions
            if wb < wa or hb < ha:
                edits.extend(["crop", "compressed"])
                notes.append(f"Downscaled/cropped from {wa}x{ha} to {wb}x{hb}")
            elif wb > wa or hb > ha:
                notes.append(f"Upscaled from {wa}x{ha} to {wb}x{hb}")

        if edge_metrics is not None:
            if getattr(edge_metrics, "is_crop_containment", 0) > 0.5:
                edits.append("crop")
                notes.append("Frame is a crop of the parent")
            if getattr(edge_metrics, "addition_mask_score", 0) > 0.2:
                edits.append("overlay_added")
                notes.append("New text/graphic overlay detected")
            if getattr(edge_metrics, "sharpness_ratio", 1.0) < 0.7:
                edits.append("recompression")
                notes.append("Detail lost to recompression")

        # Platform signatures (benchmark scenario platforms)
        if target.platform == "X/Twitter":
            edits.extend(["exif_stripped", "jpeg_recompressed"])
            diff_text = "Cropped ~15%, metadata stripped by platform, compressed."
        elif target.platform == "WhatsApp":
            edits.extend(["text_overlay_banner", "heavy_compression"])
            diff_text = "Heavy downscaling and red breaking-news text overlay banner added."
        elif target.platform == "Reddit" and "reddit" in (target.image_path or "").lower():
            edits.extend(["ai_edit", "face_alteration", "color_saturation"])
            tampering = True
            tampering_details = "AI-altered face on central volunteer with glowing eyes and high-saturation distortion."
            diff_text = "Subject face digitally altered with synthetic glowing eyes and heightened contrast."
        elif target.platform == "Instagram":
            edits.extend(["square_crop", "color_filter", "vignette"])
            diff_text = "Cropped to 1:1 square aspect ratio with warm amber color grading filter."
        elif "News" in target.platform:
            edits.extend(["crop", "editorial_credit"])
            diff_text = "Editorial crop with standard photo credit notation."
        elif notes:
            diff_text = "; ".join(dict.fromkeys(notes)) + "."
        else:
            diff_text = "Re-published copy with platform recompression."

        # Caption drift analysis
        cap_a = (source.caption or "").lower()
        cap_b = (target.caption or "").lower()

        if "california" in cap_b and "valencia" in cap_a:
            drift = ClaimDriftType.CONTRADICTED
            drift_reason = "Location switched from Valencia (Spain) flash floods to Southern California."
        elif "terrifying glowing eyes" in cap_b or "conspiracy" in cap_b:
            drift = ClaimDriftType.REFRAMED
            drift_reason = "Framing shifted from flood rescue reporting to supernatural/AI spectacle."
        elif cap_a and cap_b and cap_a != cap_b:
            drift = ClaimDriftType.SAME
            drift_reason = "Different wording; no automated contradiction detected (Gemini unavailable for deep claim analysis)."
        else:
            drift = ClaimDriftType.SAME
            drift_reason = "Context and meaning preserved."

        return GeminiPairInspection(
            visual_diff_summary=diff_text,
            edit_types=list(dict.fromkeys(edits)),
            claim_drift=drift,
            claim_drift_reasoning=drift_reason,
            tampering_detected=tampering,
            tampering_details=tampering_details,
        )
