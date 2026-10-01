from __future__ import annotations
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

# (id, label, description)
PIPELINE_STAGES: List[Tuple[str, str, str]] = [
    ("ingest", "Ingest & fingerprint", "EXIF, C2PA credentials, perceptual hash, ELA map"),
    ("forensics", "Forensic analysis", "TruFor model, embedded camera preview, pixel checks and Gemini vision"),
    ("web_check", "Reverse image search", "Search by image on Yandex & Bing, compare with copies online"),
    ("regions", "Trace edited regions", "Search faces and edited areas separately to find pasted-in sources"),
    ("verdict", "Original or edited?", "Collect the proof and decide"),
    ("discovery", "Collect similar images", "Fetch full-resolution copies of every verified match"),
    ("provenance", "Dates & platforms", "Read publication dates and identify each platform"),
    ("lineage", "Build lineage tree", "Pairwise scoring + maximum-weight arborescence"),
    ("report", "Compile report", "Verdict, mutation trail and evidence"),
]


class Job:
    def __init__(self, filename: str, caption: Optional[str], force_trace: bool):
        self.id = uuid.uuid4().hex[:12]
        self.created_at = time.time()
        self.filename = filename
        self.caption = caption
        self.force_trace = force_trace
        self.status = "queued"  # queued | running | done | failed
        self.error: Optional[str] = None
        self.preview: Dict[str, Any] = {}
        self.classification: Optional[Dict[str, Any]] = None
        self.search: Optional[Dict[str, Any]] = None
        self.result: Optional[Dict[str, Any]] = None
        self.stages: List[Dict[str, Any]] = [
            {"id": sid, "label": label, "description": desc, "status": "pending",
             "detail": None, "started_at": None, "finished_at": None, "progress": None}
            for sid, label, desc in PIPELINE_STAGES
        ]
        self._lock = threading.Lock()

    def _stage(self, stage_id: str) -> Dict[str, Any]:
        return next(s for s in self.stages if s["id"] == stage_id)

    def start(self, stage_id: str, detail: Optional[str] = None, fraction: Optional[float] = None):
        with self._lock:
            st = self._stage(stage_id)
            st.update(status="running", detail=detail, started_at=time.time(), progress=fraction)

    def progress(self, stage_id: str, detail: Optional[str] = None, fraction: Optional[float] = None):
        with self._lock:
            st = self._stage(stage_id)
            if detail is not None:
                st["detail"] = detail
            if fraction is not None:
                st["progress"] = round(max(0.0, min(1.0, fraction)), 3)

    def finish(self, stage_id: str, detail: Optional[str] = None, status: str = "done"):
        with self._lock:
            st = self._stage(stage_id)
            st.update(status=status, finished_at=time.time())
            if st["started_at"] is None:
                st["started_at"] = st["finished_at"]
            if detail is not None:
                st["detail"] = detail
            st["progress"] = None

    def skip(self, stage_id: str, reason: str):
        self.finish(stage_id, reason, status="skipped")

    def fail_running(self, message: str):
        with self._lock:
            for st in self.stages:
                if st["status"] == "running":
                    st.update(status="failed", detail=message, finished_at=time.time())
                elif st["status"] == "pending":
                    st["status"] = "skipped"

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "status": self.status,
                "filename": self.filename,
                "caption": self.caption,
                "force_trace": self.force_trace,
                "created_at": self.created_at,
                "error": self.error,
                "preview": dict(self.preview),
                "classification": self.classification,
                "search": self.search,
                "stages": [dict(s) for s in self.stages],
                "result": self.result,
            }


class JobManager:
    """In-memory job registry; investigations run on a small background thread pool."""

    def __init__(self, max_workers: int = 2, max_jobs: int = 50):
        self._jobs: Dict[str, Job] = {}
        self._order: List[str] = []
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="investigation")
        self._max_jobs = max_jobs

    def create(self, filename: str, caption: Optional[str], force_trace: bool) -> Job:
        job = Job(filename, caption, force_trace)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            while len(self._order) > self._max_jobs:
                self._jobs.pop(self._order.pop(0), None)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def submit(self, job: Job, fn: Callable[[Job], None]):
        def runner():
            job.status = "running"
            try:
                fn(job)
                job.status = "done"
            except Exception as e:  # surface the failure to the UI instead of hanging
                import traceback
                traceback.print_exc()
                job.error = str(e)
                job.fail_running(str(e))
                job.status = "failed"

        self._pool.submit(runner)
