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

What the in-memory queue owes the operator instead of durability
--------------------------------------------------------------------------
Losing a queued job on restart is acceptable; losing it *silently* is not.
The CLI has already told the hook operator ``hub queued mine as mine-abc123``
by the time the job is dropped. So every state transition — accepted,
started, completed, interrupted, dropped — is appended to
:data:`JOURNAL_FILENAME`, and :func:`unfinished_jobs` reads the journal tail
back on startup so the hub can log (and ``/statusz`` and ``mempalace doctor``
can show) exactly which accepted mines the previous run never finished.
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
    "TERMINAL_STATES",
    "MineJob",
    "MineQueue",
    "dedupe_key_for",
    "parse_mine_summary",
    "unfinished_jobs",
]

#: One JSON object per job *state transition*, next to the palace it was mined
#: into — the operator-visible answer to "what did the hub actually mine, and
#: what did it accept but never finish?".
JOURNAL_FILENAME = "mine_jobs.jsonl"

#: Journal states after which a job will never be heard from again. Anything
#: else in the tail of a previous run's journal is an accepted mine that was
#: lost with the queue.
TERMINAL_STATES = frozenset({"succeeded", "failed", "interrupted", "dropped"})

#: How much of the journal tail :func:`unfinished_jobs` reads on startup.
_JOURNAL_TAIL_BYTES = 256 * 1024

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


def unfinished_jobs(journal_path: Optional[str], exclude_pid: Optional[int] = None) -> list[dict]:
    """Jobs the journal accepted but never closed out — one dict per job id.

    The queue lives in memory, so a hub that is SIGKILLed (or crashes, or is
    restarted mid-mine) loses whatever it had accepted. This is how that loss
    becomes visible: replay the journal tail, keep the last record per job id,
    and return the ones whose last state is not terminal. ``exclude_pid``
    drops the current process's own in-flight jobs, so a hub asking at startup
    sees only the *previous* run's casualties.

    Best-effort by construction: a missing, truncated or half-written journal
    yields fewer rows, never an exception — this is diagnostics, and it must
    not be able to stop a hub from starting.

    A record whose ``pid`` is still alive is never "lost" — its hub is simply
    still mid-mine, whether that's the process asking (already handled by
    ``exclude_pid``) or a different hub the caller has no other way to know
    about (a doctor run, say, invoked from a separate process while the hub
    that accepted the job is still running it). Reuses the same liveness
    check as the writer-lease reclamation path.
    """
    if not journal_path:
        return []
    from .server_registry import _pid_alive

    try:
        size = os.path.getsize(journal_path)
        with open(journal_path, "r", encoding="utf-8", errors="replace") as fh:
            if size > _JOURNAL_TAIL_BYTES:
                fh.seek(size - _JOURNAL_TAIL_BYTES)
                fh.readline()  # discard the partial line the seek landed in
            lines = fh.readlines()
    except OSError:
        return []
    latest: dict[str, dict] = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict) or record.get("event") != "mine_job":
            continue
        job_id = record.get("job_id")
        if not job_id:
            continue
        latest[job_id] = record
    return [
        record
        for record in latest.values()
        if record.get("state") not in TERMINAL_STATES
        and (exclude_pid is None or record.get("pid") != exclude_pid)
        and not _pid_alive(record.get("pid"))
    ]


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
    # queued -> running -> succeeded | failed | interrupted, or queued -> dropped
    state: str = "queued"
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[dict] = None
    done: threading.Event = field(default_factory=threading.Event)
    stop_requested: threading.Event = field(default_factory=threading.Event)

    def should_stop(self) -> bool:
        """True once the queue has asked this job to wind up.

        Threaded into the miners as ``file_lock.should_stop`` so
        :class:`~mempalace.mine_lock.FileLockCursor` stops advancing to the
        next source file. The job then finalizes normally and ends
        ``interrupted``: partial, but never a write torn in half, and never a
        write that outlives the writer lease.
        """
        return self.stop_requested.is_set()

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
        # Journal writes happen off the condition lock (see submit()/_finish()
        # docstrings), so the threads doing the actual file I/O can be
        # scheduled in a different order than their records were logically
        # produced in. ``seq`` is assigned — under whatever lock was already
        # held — at the moment a record is built, in true logical order; the
        # small ``_journal_cond``/``_journal_next_seq`` pair below then makes
        # each writer wait its turn so records still land on disk in that
        # same order, without any of them needing the (much hotter) queue
        # condition lock while they wait or write.
        self._seq_lock = threading.Lock()
        self._seq = 0
        self._journal_cond = threading.Condition()
        self._journal_next_seq = 1
        self._pending: list[MineJob] = []
        self._running: Optional[MineJob] = None
        self._by_key: dict[str, MineJob] = {}
        self._last_completed: Optional[dict] = None
        self._completed = 0
        self._failed = 0
        self._interrupted = 0
        self._dropped = 0
        # Accepted-but-unfinished jobs from a PREVIOUS run of this hub, read
        # off the journal once at construction. Reported in status() forever
        # after: the queue cannot recover them, so the least it can do is
        # keep saying they were lost.
        self.orphans: list[dict] = unfinished_jobs(journal_path, exclude_pid=os.getpid())
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

    def stop(self, timeout: Optional[float] = 5.0) -> dict:
        """Wind the worker down and report what that cost.

        Three things happen here, and the caller (the hub, on its way to
        handing back the writer lease) depends on all three:

        1. Queued-but-unstarted jobs are dropped *now*, out of the pending
           list, and journalled ``dropped``. Leaving them for the worker to
           notice was the old bug: ``_next_job`` only consulted ``_stopping``
           when the pending list was empty, so a "stopping" worker cheerfully
           drained the entire backlog after ``stop()`` returned.
        2. The in-flight job is asked to stop at its next file boundary, so
           its per-file cursor stops advancing instead of running the mine to
           the end of a 4 GB session directory.
        3. The join is bounded, and the returned dict says whether it held —
           ``worker_alive`` True means a job is *still* mid-file and the
           caller must not treat the lease as free.
        """
        with self._cond:
            self._stopping = True
            dropped = list(self._pending)
            self._pending.clear()
            now = time.time()
            self._dropped += len(dropped)
            for job in dropped:
                job.state = "dropped"
                job.finished_at = now
                if self._by_key.get(job.dedupe_key) is job:
                    del self._by_key[job.dedupe_key]
            running = self._running
            if running is not None:
                running.stop_requested.set()
            self._cond.notify_all()
            thread = self._thread
        for job in dropped:
            self._append_journal(
                self._journal_record(
                    job,
                    {"success": False, "error": "hub shut down before the job started"},
                )
            )
            job.done.set()
        if dropped:
            logger.warning(
                "mine queue: %d accepted mine job(s) dropped at shutdown and NOT mined: %s. "
                "They are journalled as 'dropped'; re-run them or wait for the next hook fire.",
                len(dropped),
                ", ".join(f"{job.id}({os.path.basename(job.source)})" for job in dropped),
            )
        if thread is not None:
            thread.join(timeout)
        alive = bool(thread is not None and thread.is_alive())
        with self._cond:
            still_running = self._running
        return {
            "dropped": [job.snapshot() for job in dropped],
            "interrupted": still_running.snapshot() if still_running is not None else None,
            "worker_alive": alive,
        }

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
            # The "accepted" record is the only evidence a job ever existed
            # if the hub dies before it runs, so it is built (and its ``seq``
            # assigned) right here, before the notify — guaranteeing it is
            # logically ordered ahead of any "started" record _next_job can
            # produce, since that requires this same lock. The actual file
            # write happens after the lock is released, below: submit()'s
            # caller should not pay for journal I/O while holding the queue
            # lock and blocking the worker's own state transitions.
            record = self._journal_record(job, {"success": True}, state="accepted")
            self._cond.notify_all()
        self._append_journal(record)
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
        # ``position`` is 0 for a running job as well as a queued-first one, so
        # spell the state out rather than printing "(running, position 0)" at
        # an operator reading a hook log.
        if job.state == "running":
            where = "already running"
        elif position:
            where = f"queued at position {position}"
        else:
            where = f"queued ({job.state})"
        snapshot.update(
            {
                "success": True,
                "queued": True,
                "deduplicated": not created,
                "message": (
                    f"mine job {job.id} {'accepted' if created else 'already submitted'}, "
                    f"{where}; it runs in the hub's background worker"
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
                "interrupted": self._interrupted,
                "dropped": self._dropped,
                "last_completed": self._last_completed,
                "worker_alive": bool(self._thread is not None and self._thread.is_alive()),
                # Jobs the PREVIOUS hub run accepted and never finished. Non-
                # empty here means an operator (or a hook) should resubmit.
                "lost_previous_run": len(self.orphans),
                "lost_previous_run_jobs": self.orphans,
            }

    # ── worker ───────────────────────────────────────────────────────────

    def _next_job(self) -> Optional[MineJob]:
        with self._cond:
            # Checked before AND after the wait, and ahead of the pending list
            # in both cases: a stopping worker must start nothing, not drain
            # the backlog it happens to be holding.
            if self._stopping:
                return None
            while not self._pending and not self._stopping:
                self._cond.wait()
            if self._stopping or not self._pending:
                return None
            job = self._pending.pop(0)
            job.state = "running"
            job.started_at = time.time()
            self._running = job
        self._append_journal(self._journal_record(job, {"success": True}, state="started"))
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
            if result.get("success"):
                # ``cut_short`` (set by the runner from the file-lock cursor's
                # own observation) says whether iteration actually broke off
                # at a file boundary — not just whether stop() was ever
                # requested, which can go true after the last file already
                # finished on its own. Fall back to should_stop() only when
                # the runner didn't report it, so older/stub runners keep
                # their previous behavior.
                cut_short = result.get("cut_short")
                if cut_short is None:
                    cut_short = job.should_stop()
                job.state = "interrupted" if cut_short else "succeeded"
            else:
                job.state = "failed"
            self._running = None
            # Drop the dedupe slot only if it still points at this job.
            if self._by_key.get(job.dedupe_key) is job:
                del self._by_key[job.dedupe_key]
            if job.state == "succeeded":
                self._completed += 1
            elif job.state == "interrupted":
                self._interrupted += 1
            else:
                self._failed += 1
            record = self._journal_record(job, result)
            self._last_completed = record
            self._cond.notify_all()
        job.done.set()
        self._append_journal(record)

    def _next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def _journal_record(self, job: MineJob, result: dict, state: Optional[str] = None) -> dict:
        counts = parse_mine_summary(result.get("output"))
        record = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "seq": self._next_seq(),
            "event": "mine_job",
            "job_id": job.id,
            "source": job.source,
            "mode": job.mode,
            "wing": job.wing,
            "agent": job.agent,
            "state": state or job.state,
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
        """Best-effort: a journal failure must never fail a mine that ran.

        Waits for records with a lower ``seq`` to be written first, so the
        journal's on-disk order matches the true logical order even though no
        caller holds the queue's condition lock while this runs. The wait is
        bounded — a lost or never-appended seq (there is none on any current
        call path, but this is diagnostics and must not be able to wedge a
        hub) degrades to writing out of order rather than blocking forever.
        """
        if not self._journal_path:
            return
        seq = record.get("seq")
        if seq is not None:
            with self._journal_cond:
                deadline = time.monotonic() + 5.0
                while self._journal_next_seq < seq:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not self._journal_cond.wait(remaining):
                        break
                self._write_journal_line(record)
                self._journal_next_seq = max(self._journal_next_seq, seq + 1)
                self._journal_cond.notify_all()
            return
        self._write_journal_line(record)

    def _write_journal_line(self, record: dict) -> None:
        try:
            with open(self._journal_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("mine journal append failed", exc_info=True)
