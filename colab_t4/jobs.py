"""Thread-backed management jobs with bounded, redacted in-memory history."""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
import queue
import threading
from typing import Any, Callable, Optional
from uuid import uuid4

from .config import redact


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {redact(str(key)): _redact_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    try:
        return redact(repr(value))
    except Exception:
        return redact(f"<{type(value).__name__}>")


class Job:
    def __init__(self, name: str) -> None:
        now = _timestamp()
        self.id = str(uuid4())
        self.name = redact(str(name))
        self.status = "queued"
        self.progress: dict[str, Any] = {"message": "queued", "percent": 0}
        self.result: Any = None
        self.error: Optional[str] = None
        self.created_at = now
        self.updated_at = now
        self._cancel_event = threading.Event()
        self._cancellation_observed = False
        self._lock = threading.RLock()

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "name": redact(str(self.name)),
                "status": self.status,
                "progress": _redact_value(self.progress),
                "result": _redact_value(self.result),
                "error": _redact_value(self.error),
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            }


class JobContext:
    def __init__(self, job: Job) -> None:
        self._job = job

    @property
    def cancelled(self) -> bool:
        cancelled = self._job._cancel_event.is_set()
        if cancelled:
            with self._job._lock:
                self._job._cancellation_observed = True
        return cancelled

    def progress(self, message: str, percent: Optional[int] = None) -> None:
        with self._job._lock:
            self._job.progress = {
                "message": redact(str(message)),
                "percent": percent,
            }
            self._job.updated_at = _timestamp()


class JobManager:
    def __init__(self) -> None:
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()
        self._lock = threading.RLock()
        self._work: "queue.Queue[tuple[Job, Callable[[JobContext], Any]]]" = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, name: str, target: Callable[[JobContext], Any]) -> Job:
        job = Job(name)
        with self._lock:
            self._jobs[job.id] = job
            self._trim_terminal_jobs()
        self._work.put((job, target))
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        """Return a snapshot of retained jobs in submission order."""
        with self._lock:
            return list(self._jobs.values())

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            with job._lock:
                if job.status in {"succeeded", "failed", "cancelled"}:
                    return False
                job._cancel_event.set()
                if job.status == "queued":
                    job.status = "cancelled"
                    job.progress = {"message": "cancelled", "percent": None}
                    job.updated_at = _timestamp()
                return True

    def _trim_terminal_jobs(self) -> None:
        """Retain all active work while bounding completed job history to 50."""
        terminal_ids = []
        for job_id, job in self._jobs.items():
            with job._lock:
                if job.status in {"succeeded", "failed", "cancelled"}:
                    terminal_ids.append(job_id)
        for job_id in terminal_ids[:-50]:
            self._jobs.pop(job_id, None)

    def _worker(self) -> None:
        while True:
            job, target = self._work.get()
            try:
                self._run(job, target)
            finally:
                self._work.task_done()

    def _run(self, job: Job, target: Callable[[JobContext], Any]) -> None:
        try:
            with job._lock:
                if job.cancelled:
                    return
                job.status = "running"
                job.progress = {"message": "running", "percent": None}
                job.updated_at = _timestamp()
            try:
                result = target(JobContext(job))
            except Exception as exc:
                with job._lock:
                    if job.cancelled:
                        job.status = "cancelled"
                    else:
                        job.status = "failed"
                        job.error = redact(str(exc))
                    job.updated_at = _timestamp()
                return
            with job._lock:
                if job._cancellation_observed:
                    job.status = "cancelled"
                else:
                    job.status = "succeeded"
                    job.result = _redact_value(result)
                    job._cancel_event.clear()
                job.updated_at = _timestamp()
        finally:
            with self._lock:
                self._trim_terminal_jobs()
