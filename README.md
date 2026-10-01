# TraceLens

**Prove an image was edited — even without the original.**

Upload a picture and TraceLens finds its copies online, outlines exactly what changed, traces
pasted-in parts back to the photo they were cut from, and reconstructs the path from the
earliest known copy to yours. Every verdict comes with the evidence behind it.

---

## What it does

| | |
|---|---|
| **Copies online** | Reverse-searches the image on Yandex and Bing, downloads every hit and aligns it pixel-by-pixel with yours to outline what changed. |
| **Pasted regions** | Faces and edited areas are searched *on their own*, so a head or object cut from another photo is traced to its source. |
| **Camera preview** | Cameras hide a small preview inside the JPEG. If an editor left it untouched, it shows the picture before it was changed. |
| **Forgery model** | [TruFor](https://grip-unina.github.io/TruFor/) (CVPR 2023) reads the camera-noise fingerprint and localises spliced or retouched areas. *(optional)* |
| **Lineage tree** | Copies are dated, attributed to platforms and ordered into a family tree from the origin to your copy. |
| **Proof, not a score** | Evidence is ranked: **direct proof** (a reference shows different content), **strong evidence** (a trained detector localises an edit), **supporting signals** (heuristics, Gemini). |

### Pipeline

Each upload runs as a background job whose progress is streamed to the UI:

1. **Ingest & fingerprint** — EXIF, C2PA credentials, perceptual hash, ELA map
2. **Forensic analysis** — embedded camera preview, TruFor, ELA / noise / clone / banner checks, Gemini vision
3. **Reverse image search** — Yandex + Bing search-by-image; every candidate is visually verified
4. **Trace edited regions** — faces and changed areas reverse-searched separately
5. **Original or edited?** — the proof is collected and a verdict is reached
6. **Collect similar images** — full-resolution copies of the closest matches
7. **Dates & platforms** — publication dates scraped from each page
8. **Build lineage tree** — pairwise scoring + maximum-weight arborescence
9. **Compile report**

---

## Quick start

**Requirements:** Python 3.11+, Node 18+, and optionally an NVIDIA GPU for TruFor.

### 1. Backend

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r backend/requirements.txt

cp .env.example .env          # then add your GEMINI_API_KEY (optional)
python -m uvicorn backend.main:app --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev                   # http://localhost:5173 (proxies /api and /static to :8000)
```

### 3. Optional: TruFor forgery model

TruFor needs PyTorch and ~270 MB of weights. Without it, everything else still works and the
pipeline reports TruFor as unavailable.

```bash
# Install PyTorch for your platform first, e.g. CUDA 12.4 (or use .../whl/cpu for CPU-only):
pip install torch --index-url https://download.pytorch.org/whl/cu124

python scripts/download_models.py   # TruFor weights (MD5-verified) + YuNet face detector
```

The face detector used for region search is also fetched by this script; without it, region
search still runs on edited areas but skips faces.

---

## Configuration

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Enables Gemini vision analysis (semantic edits, AI-generation checks). Optional — the app degrades gracefully when it is missing or its quota is exhausted. |
| `GEMINI_MODELS` | Optional comma-separated model list to try in order (default: `gemini-3.5-flash, gemini-flash-latest, gemini-3.1-flash-lite, gemini-flash-lite-latest`). |

Reverse image search uses Yandex and Bing directly and needs no API key.

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/investigations` | Upload an image (`file`, optional `caption`); starts a job and returns its id |
| `GET` | `/api/investigations/{id}` | Job status, per-stage progress and, when done, the full result |
| `POST` | `/api/upload/investigate` | Synchronous version of the above (scripts / CLI) |
| `POST` | `/api/upload/batch` | Upload several copies to work out who copied whom |
| `GET` | `/api/health` | Backend, Gemini and search-engine status |
| `GET` | `/api/scenarios/{id}/tree` | Bundled benchmark scenario (fictional demo data) |

The result includes the verdict, `proof` items, per-region searches, verified `web_matches`,
the lineage `nodes`/`edges`, the origin-to-upload `lineage_path`, and per-platform summaries.

---

## Project structure

```
backend/
  main.py                    FastAPI app and routes
  jobs.py                    Background job registry and pipeline stage definitions
  pipeline/investigation.py  The end-to-end investigation pipeline and proof collection
  forensics/
    classifier.py            Local signals: ELA, noise, copy-move, banners, metadata
    embedded_preview.py      Camera-preview (EXIF/MPF thumbnail) comparison
    trufor_detector.py       TruFor wrapper (lazy-loaded, CUDA when available)
    trufor/                  Vendored TruFor inference code (GRIP-UNINA licence)
    ela.py                   Error level analysis map
  search/
    reverse_search.py        Yandex + Bing search-by-image
    match_verifier.py        Download, align and diff candidates against the upload
    region_search.py         Per-region reverse search for pasted-in content
    page_scraper.py          Publication dates and platform classification
  engine/                    Data models, pairwise edge scoring, lineage tree, Gemini client
  ingestion/                 EXIF, C2PA, video keyframes, perceptual fingerprints
  fixtures/                  Benchmark scenario (fictional demo data) and Gemini response cache
frontend/
  src/App.jsx                Landing page, live pipeline, results
  src/components/            Proof, LineageTree, Timeline, SimilarImages, PipelineRunner, …
scripts/download_models.py   Fetch TruFor weights and the face detector
tests/                       pytest suite and sample image pairs
cli.py                       Run the benchmark scenario from the terminal
```

---

## Tests

```bash
python -m pytest tests
```

The suite covers the forensic classifier, the embedded-preview check (with generated camera
JPEGs), change localisation, region proposals, the lineage tree builder and the benchmark
scenario. It runs offline.

---

## Limitations

- **Absence of evidence is not proof of authenticity.** "No edit found" means none of the checks
  found one; careful edits can still slip through.
- **If the original was never published**, its pixels cannot be recovered. TraceLens proves the
  edit instead — by localising it, checking the embedded preview and tracing pasted regions.
- **Reverse search is unofficial.** Yandex and Bing are used without an API and may rate-limit or
  change their pages; the UI reports engine failures.
- **Embedded previews** survive only in files straight from a camera or phone — most social
  platforms strip them.
- **Publication dates** are read from page metadata and are missing on many sites.

---

## Credits & licences

- **TruFor** — F. Guillaro, D. Cozzolino, A. Sud, N. Dufour, L. Verdoliva, *TruFor: Leveraging
  all-round clues for trustworthy image forgery detection and localization*, CVPR 2023.
  © GRIP-UNINA. Code and weights may be used **for informational and non-profit purposes only** —
  see `backend/forensics/trufor/LICENSE.txt`. This restriction applies to any deployment that
  enables TruFor.
- **YuNet** face detector — OpenCV model zoo (MIT).
- Reverse image search via Yandex and Bing; vision analysis via Google Gemini.
