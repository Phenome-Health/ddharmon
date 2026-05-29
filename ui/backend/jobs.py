"""In-memory job store for harmonization runs (mirrors biomapper-ui services/jobs.py).

A run is a long, mostly-CPU-bound pipeline (embed → BERTopic → sub-cluster → anchor
→ classify), so it executes in a daemon thread and reports progress by mutating its
:class:`Job` under a lock. The SSE endpoint polls ``to_dict()`` every 0.5s.

State is in-memory only — jobs are lost on restart (acceptable for the v1 single-user
GUI; swap in SQLite later if persistence is needed).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

# Non-terminal phases double as ``status`` values so the UI can show a phase label.
TERMINAL_STATES = {"complete", "error"}
_TTL_SECONDS = 3600


@dataclass
class Job:
    """A single harmonization run."""

    job_id: str
    display_name: str
    status: str = "pending"  # pending|loading|embedding|clustering|anchoring|classifying|complete|error
    phase: str = "pending"
    completed: int = 0
    total: int = 0
    error_message: str | None = None
    result: dict[str, Any] | None = None  # serialized summary + verdicts (camelCase)
    config: dict[str, Any] = field(default_factory=dict)
    decisions: dict[str, dict[str, str]] = field(default_factory=dict)  # subClusterId -> {decision, note}
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "jobId": self.job_id,
            "displayName": self.display_name,
            "status": self.status,
            "phase": self.phase,
            "completed": self.completed,
            "total": self.total,
            "errorMessage": self.error_message,
            "result": self.result,
            "config": self.config,
            "decisions": self.decisions,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }

    def summary_dict(self) -> dict[str, Any]:
        """Lightweight view for the jobs list (omits the heavy result payload)."""
        d = self.to_dict()
        d.pop("result", None)
        d["nVerdicts"] = len(self.result["verdicts"]) if self.result else 0
        return d


class JobStore:
    """Thread-safe in-memory job registry with TTL purging."""

    def __init__(self, ttl_seconds: int = _TTL_SECONDS) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def create(self, job_id: str, display_name: str, config: dict[str, Any]) -> Job:
        with self._lock:
            job = Job(job_id=job_id, display_name=display_name, config=config)
            self._jobs[job_id] = job
            return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def delete(self, job_id: str) -> bool:
        with self._lock:
            return self._jobs.pop(job_id, None) is not None

    def update(self, job_id: str, **fields: Any) -> None:
        """Set attributes on a job (status/phase/completed/total/error_message/result)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in fields.items():
                setattr(job, key, value)
            job.updated_at = time.time()

    def set_decision(self, job_id: str, sub_cluster_id: str, decision: str, note: str = "") -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            job.decisions[sub_cluster_id] = {"decision": decision, "note": note}
            job.updated_at = time.time()
            return True

    def purge_expired(self) -> None:
        cutoff = time.time() - self._ttl
        with self._lock:
            stale = [jid for jid, j in self._jobs.items() if j.updated_at < cutoff and j.status in TERMINAL_STATES]
            for jid in stale:
                self._jobs.pop(jid, None)


# Module-level singleton used by the app.
store = JobStore()
