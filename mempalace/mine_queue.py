"""Background mine jobs for the writable MCP hub.

Why the hub grows its own worker instead of handing mines to
``mempalace daemon``
--------------------------------------------------------------------------
The daemon (``mempalace/daemon.py``) already has a durable job queue, and the
obvious move is to forward hub mines to it. It is the wrong move here, for
three reasons that all come back to the single-writer rule in
``docs/write-routing-policy.md``:

1. The writable hub owns the palace writer lease for its whole lifetime
   (``_LEASE_SCOPE_LIFETIME``; the flock is released only at shutdown). A
   daemon running the mine is a *second* process trying to write the same
   file-backed palace — it would park every job behind the lock the hub is
   holding and never run one. That is the same dead end the earlier
   "make hook mines direct" attempt hit.
2. Chroma/HNSW consistency. The hub's collection handle is hot: opening a
   second ``PersistentClient`` on the same directory is exactly the
   stale-index hazard the lease exists to prevent. The mine must run inside
   the hub process, on the hub's own handle.
3. Operationally, a hub deployment is one supervised process. Requiring a
   second daemon to make Stop hooks work turns one moving part into two.

So this module is deliberately small and in-process: an ordered queue, one
worker thread, a dedupe index, and a JSONL journal. It does not try to be the
daemon's queue store. The trade-off it accepts is durability — a queue held in
memory is lost if the hub restarts. That is acceptable *because* the jobs are
hook-driven: the next Stop or PreCompact fires within minutes and re-submits
the same source. A queue that survived restarts would also have to survive the
watchdog killing a job mid-file, which is a much larger problem than the one
this fixes.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "JOURNAL_FILENAME",
    "MineJob",
    "MineQueue",
    "dedupe_key_for",
    "parse_mine_summary",
]

#: One JSON object per completed job, next to the palace it was mined into —
#: the operator-visible answer to "what did the hub actually mine?".
JOURNAL_FILENAME = "mine_jobs.jsonl"

_SUMMARY_FILES_RE = re.compile(r"Files processed:\s*(\d+)")
_SUMMARY_DRAWERS_RE = re.compile(r"Drawers filed:\s*(\d+)")


def dedupe_key_for(source: str, mode: str, wing: Optional[str]) -> str:
    """Identity of a mine request for dedupe purposes.

    Hooks fire repeatedly — Stop, SessionEnd and PreCompact can all aim at the
    same transcript within seconds — so a queue that stacked identical requests
    would mine the same 80 MB file three times. ``agent``/``limit``/``extract``
    are deliberately *not* part of the key: they do not change which source is
    being read, and letting them split the key would reopen the stacking.
    """
    return f"{os.path.abspath(os.path.expanduser(source))}|{mode}|{wing or ''}"


def parse_mine_summary(output: Optional[str]) -> dict:
    """Pull the file/drawer counts out of a miner's human-readable summary.

    Both miners print ``Files processed:`` and ``Drawers filed:`` in their
    end-of-mine block. Parsing them is what lets the journal answer "was that
    three-minute job actually doing anything?" without changing every miner's
    return type (they return ``None``). Missing values stay ``None`` rather
    than defaulting to 0 — an unparsed summary is not a mine that filed
    nothing.
    """
    if not output:
        return {"files": None, "drawers": None}
    files = _SUMMARY_FILES_RE.search(output)
    drawers = _SUMMARY_DRAWERS_RE.search(output)
    return {
        "files": int(files.group(1)) if files else None,
        "drawers": int(drawers.group(1)) if drawers else None,
    }


@dataclass
class MineJob:
    """One queued mine request."""

    id: str
    source: str
    mode: str
    wing: Optional[str]
    agent: str
    limit: int
    dry_run: bool
    extract: str
    dedupe_key: str
    enqueued_at: float
    state: str = "queued"  # queued -> running -> succeeded | failed
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[dict] = None
    done: threading.Event = field(default_factory=threading.Event)

    @property
    def duration(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else time.time()
        return end - self.started_at

    def wait(self, timeout: Optional[float]) -> bool:
        """Block until the job is terminal. False on timeout."""
        return self.done.wait(timeout)

    def snapshot(self, position: Optional[int] = None) -> dict:
        payload = {
            "job_id": self.id,
            "source": self.source,
            "mode": self.mode,
            "wing": self.wing,
            "state": self.state,
            "enqueued_at": self.enqueued_at,
        }
        if position is not None:
            payload["position"] = position
        if self.started_at is not None:
            payload["started_at"] = self.started_at
        if self.duration is not None:
            payload["duration_seconds"] = round(self.duration, 3)
        return payload


class MineQueue:
    """Ordered, deduplicated mine queue drained by one worker thread.

    Sequential by design: the worker takes the palace write lock per source
    file, and two workers would just contend for it while doubling the number
    of Chroma writers in flight.
    """

    def __init__(
        self,
        runner: Callable[[MineJob], dict],
        journal_path: Optional[str] = None,
        name: str = "mcp-mine-worker",
    ) -> None:
        self._runner = runner
        self._journal_path = journal_path
        self._name = name
        self._cond = threading.Condition()
        self._pending: list[MineJob] = []
        self._running: Optional[MineJob] = None
        self._by_key: dict[str, MineJob] = {}
        self._last_completed: Optional[dict] = None
        self._completed = 0
        self._failed = 0
        self._thread: Optional[threading.Thread] = None
        self._stopping = False

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the worker thread. Idempotent."""
        with self._cond:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stopping = False
            self._thread = threading.Thread(target=self._worker, name=self._name, daemon=True)
            self._thread.start()

    def stop(self, timeout: Optional[float] = 5.0) -> None:
        """Ask the worker to finish the current job and exit.

        Queued-but-unstarted jobs are abandoned: the hub is going away and the
        hooks that submitted them will fire again.
        """
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout)

    # ── submission ───────────────────────────────────────────────────────

    def submit(
        self,
        *,
        source: str,
        mode: str = "projects",
        wing: Optional[str] = None,
        agent: str = "mempalace",
        limit: int = 0,
        dry_run: bool = False,
        extract: str = "exchange",
    ) -> tuple[MineJob, bool]:
        """Enqueue a mine. Returns ``(job, created)``.

        ``created`` is False when an identical job is already queued or
        running — the caller gets that job's id back instead of stacking a
        duplicate.
        """
        key = dedupe_key_for(source, mode, wing)
        with self._cond:
            existing = self._by_key.get(key)
            if existing is not None:
                return existing, False
            job = MineJob(
                id=f"mine-{uuid.uuid4().hex[:12]}",
                source=os.path.abspath(os.path.expanduser(source)),
                mode=mode,
                wing=wing,
                agent=agent,
                limit=limit,
                dry_run=dry_run,
                extract=extract,
                dedupe_key=key,
                enqueued_at=time.time(),
            )
            self._by_key[key] = job
            self._pending.append(job)
            self._cond.notify_all()
            return job, True

    def position(self, job: MineJob) -> int:
        """0 when running (or terminal), else 1-based place in the queue."""
        with self._cond:
            return self._position_locked(job)

    def _position_locked(self, job: MineJob) -> int:
        for index, pending in enumerate(self._pending, 1):
            if pending is job:
                return index
        return 0

    def queued_payload(self, job: MineJob, created: bool) -> dict:
        """The structured async reply ``tool_mine`` returns to the caller."""
        with self._cond:
            position = self._position_locked(job)
            snapshot = job.snapshot(position=position)
        snapshot.update(
            {
                "success": True,
                "queued": True,
                "deduplicated": not created,
                "message": (
                    f"mine job {job.id} {'accepted' if created else 'already queued'} "
                    f"({job.state}, position {position}); "
                    "it runs in the hub's background worker"
                ),
            }
        )
        return snapshot

    # ── status ───────────────────────────────────────────────────────────

    def status(self) -> dict:
        with self._cond:
            running = self._running.snapshot() if self._running is not None else None
            queued = [job.snapshot(position=i) for i, job in enumerate(self._pending, 1)]
            return {
                "enabled": True,
                "running": running,
                "queued": len(queued),
                "queued_jobs": queued,
                "completed": self._completed,
                "failed": self._failed,
                "last_completed": self._last_completed,
                "worker_alive": bool(self._thread is not None and self._thread.is_alive()),
            }

    # ── worker ───────────────────────────────────────────────────────────

    def _next_job(self) -> Optional[MineJob]:
        with self._cond:
            while not self._pending and not self._stopping:
                self._cond.wait()
            if not self._pending:
                return None
            job = self._pending.pop(0)
            job.state = "running"
            job.started_at = time.time()
            self._running = job
            return job

    def _worker(self) -> None:
        while True:
            job = self._next_job()
            if job is None:
                return
            try:
                result = self._runner(job)
                if not isinstance(result, dict):
                    result = {"success": True, "output": str(result)}
            except BaseException as exc:  # noqa: BLE001 - a worker must not die
                # The runner already converts every ordinary miner failure into
                # a structured result; reaching here means something escaped it
                # (SystemExit from a library mine, MemoryError, …). Record it
                # and take the next job — a dead worker would silently stop
                # every future hook mine.
                logger.exception("mine worker: job %s raised", job.id)
                result = {
                    "success": False,
                    "error": f"mine failed: {exc}",
                    "error_class": type(exc).__name__,
                }
            self._finish(job, result)

    def _finish(self, job: MineJob, result: dict) -> None:
        with self._cond:
            job.result = result
            job.finished_at = time.time()
            job.state = "succeeded" if result.get("success") else "failed"
            self._running = None
            # Drop the dedupe slot only if it still points at this job.
            if self._by_key.get(job.dedupe_key) is job:
                del self._by_key[job.dedupe_key]
            if job.state == "succeeded":
                self._completed += 1
            else:
                self._failed += 1
            record = self._journal_record(job, result)
            self._last_completed = record
            self._cond.notify_all()
        job.done.set()
        self._append_journal(record)

    def _journal_record(self, job: MineJob, result: dict) -> dict:
        counts = parse_mine_summary(result.get("output"))
        record = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "event": "mine_job",
            "job_id": job.id,
            "source": job.source,
            "mode": job.mode,
            "wing": job.wing,
            "agent": job.agent,
            "state": job.state,
            "duration_seconds": round(job.duration or 0.0, 3),
            "files": counts["files"],
            "drawers": counts["drawers"],
            "pid": os.getpid(),
        }
        if not result.get("success"):
            record["error"] = str(result.get("error", ""))[:500]
            if result.get("error_class"):
                record["error_class"] = result["error_class"]
        return record

    def _append_journal(self, record: dict) -> None:
        """Best-effort: a journal failure must never fail a mine that ran."""
        if not self._journal_path:
            return
        try:
            with open(self._journal_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("mine journal append failed", exc_info=True)
