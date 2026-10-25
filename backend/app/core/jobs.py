"""Background jobs with progress and cancellation. PLAN.md D6."""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .errors import JobNotFound
from .pipeline import CancelToken


@dataclass
class Job:
    job_id: str
    kind: str
    status: str = "pending"  # pending | running | done | error | cancelled
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    token: CancelToken = field(default_factory=CancelToken)

    def summary(self) -> dict:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "progress": round(self.progress, 4),
            "message": self.message,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "elapsed": round((self.finished_at or time.time()) - self.created_at, 3),
            "result": self.result,
        }


class JobManager:
    def __init__(self, max_jobs: int = 64):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        self.max_jobs = max_jobs
        self._listeners: dict[str, list[Callable[[dict], None]]] = {}

    def create(self, kind: str) -> Job:
        job = Job(job_id=uuid.uuid4().hex, kind=kind)
        with self._lock:
            self._jobs[job.job_id] = job
            if len(self._jobs) > self.max_jobs:
                done = sorted(
                    (j for j in self._jobs.values()
                     if j.status in ("done", "error", "cancelled")),
                    key=lambda j: j.created_at,
                )
                for old in done[: len(self._jobs) - self.max_jobs]:
                    self._jobs.pop(old.job_id, None)
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            j = self._jobs.get(job_id)
        if j is None:
            raise JobNotFound(f"No job with id '{job_id}'.")
        return j

    def list(self) -> list[dict]:
        with self._lock:
            return [j.summary() for j in sorted(
                self._jobs.values(), key=lambda j: -j.created_at)]

    def cancel(self, job_id: str) -> None:
        j = self.get(job_id)
        j.token.cancel()
        if j.status in ("pending", "running"):
            j.status = "cancelled"
            j.finished_at = time.time()

    def run(self, job: Job, fn: Callable[[Job], Any]) -> None:
        """Run fn(job) on a worker thread, recording progress and failures."""
        def target():
            job.status = "running"
            try:
                job.result = fn(job)
                job.status = "done" if not job.token.cancelled else "cancelled"
                job.progress = 1.0
            except Exception as exc:
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.message = traceback.format_exc(limit=3)
            finally:
                job.finished_at = time.time()

        threading.Thread(target=target, daemon=True, name=f"job-{job.kind}").start()

    def progress(self, job: Job, value: float, message: str = "") -> None:
        job.progress = max(0.0, min(1.0, float(value)))
        if message:
            job.message = message


jobs = JobManager()
