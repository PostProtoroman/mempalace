"""test_mine_queue.py — the hub's background mine worker.

Incident this covers (2026-09-02): Claude Code Stop/PreCompact hooks spawn
``mempalace mine <transcript> --mode convos``; the CLI forwards it to the live
hub; the hub ran ``tool_mine`` INLINE inside its exclusive write path. A 20-80 MB
transcript takes 3+ minutes, ``MEMPALACE_MCP_WRITE_STALL_EXIT_SECS=180`` fired,
the hub exited 75, KeepAlive restarted it, the mine was lost, and the next Stop
repeated it — four kills in fifteen minutes.

The fix has three moving parts, all asserted here:

* ``mempalace_mine`` on the writable hub ENQUEUES and returns a job id;
* one worker thread drains jobs sequentially, taking the hub's exclusive write
  lock PER SOURCE FILE so peers interleave;
* the stall watchdog measures those per-file holds, not the whole job.
"""

import json
import threading
import time
from unittest.mock import patch

import pytest

from mempalace import mcp_server
from mempalace.mine_lock import (
    FINALIZE_PHASE,
    OPEN_PHASE,
    FileLockCursor,
    GuardedFileLock,
)
from mempalace.mine_queue import MineQueue, dedupe_key_for, parse_mine_summary


# ── FileLockCursor ───────────────────────────────────────────────────────


class _RecordingLock:
    """A file_lock factory that records acquire/release order."""

    def __init__(self):
        self.events = []

    def __call__(self, label):
        events = self.events

        class _CM:
            def __enter__(self):
                events.append(("acquire", label))
                return self

            def __exit__(self, *exc):
                events.append(("release", label))
                return False

        return _CM()


def test_cursor_moves_the_lock_between_files():
    lock = _RecordingLock()
    cursor = FileLockCursor(lock)
    cursor.enter(OPEN_PHASE)
    seen = list(cursor.iter(["a", "b", "c"]))
    cursor.enter(FINALIZE_PHASE)
    cursor.release()

    assert seen == ["a", "b", "c"]
    assert lock.events == [
        ("acquire", OPEN_PHASE),
        ("release", OPEN_PHASE),
        ("acquire", "a"),
        ("release", "a"),
        ("acquire", "b"),
        ("release", "b"),
        ("acquire", "c"),
        ("release", "c"),
        ("acquire", FINALIZE_PHASE),
        ("release", FINALIZE_PHASE),
    ]
    assert cursor.active is False


def test_cursor_without_a_lock_is_a_no_op():
    """The CLI/library/stdio path must behave exactly as before."""
    cursor = FileLockCursor(None)
    cursor.enter(OPEN_PHASE)
    assert list(cursor.iter([1, 2])) == [1, 2]
    assert cursor.active is False
    cursor.release()


def test_cursor_release_is_idempotent():
    lock = _RecordingLock()
    cursor = FileLockCursor(lock)
    cursor.enter("x")
    cursor.release()
    cursor.release()
    assert lock.events == [("acquire", "x"), ("release", "x")]


# ── queue mechanics ──────────────────────────────────────────────────────


def _queue(runner, tmp_path=None):
    journal = str(tmp_path / "mine_jobs.jsonl") if tmp_path is not None else None
    q = MineQueue(runner, journal_path=journal)
    q.start()
    return q


def test_submit_returns_immediately_with_a_job_id():
    release = threading.Event()

    def runner(job):
        release.wait(5)
        return {"success": True, "output": "Files processed: 1\nDrawers filed: 2\n"}

    q = _queue(runner)
    try:
        started = time.monotonic()
        job, created = q.submit(source="/tmp/a", mode="convos")
        assert created is True
        assert job.id.startswith("mine-")
        # Enqueue is O(1) — it must not wait on the runner.
        assert time.monotonic() - started < 1.0
        payload = q.queued_payload(job, created)
        assert payload["success"] is True
        assert payload["queued"] is True
        assert payload["deduplicated"] is False
    finally:
        release.set()
        q.stop()


def test_duplicate_submission_returns_the_same_job():
    """Hooks fire repeatedly; stacking identical mines is the storm."""
    release = threading.Event()

    def runner(job):
        release.wait(5)
        return {"success": True, "output": ""}

    q = _queue(runner)
    try:
        first, created_first = q.submit(source="/tmp/a", mode="convos", wing="sessions")
        second, created_second = q.submit(source="/tmp/a", mode="convos", wing="sessions")
        # agent/limit/extract are not part of the key.
        third, created_third = q.submit(
            source="/tmp/a", mode="convos", wing="sessions", agent="other", limit=5
        )
        other, created_other = q.submit(source="/tmp/b", mode="convos", wing="sessions")

        assert created_first is True
        assert created_second is False and second is first
        assert created_third is False and third is first
        assert created_other is True and other is not first
        assert q.queued_payload(second, created_second)["deduplicated"] is True
    finally:
        release.set()
        q.stop()


def test_dedupe_key_ignores_agent_but_splits_on_mode_and_wing():
    base = dedupe_key_for("/tmp/a", "convos", "sessions")
    assert base == dedupe_key_for("/tmp/a", "convos", "sessions")
    assert base != dedupe_key_for("/tmp/a", "projects", "sessions")
    assert base != dedupe_key_for("/tmp/a", "convos", "other")


def test_worker_drains_jobs_sequentially():
    order = []
    overlap = []
    inside = threading.Lock()

    def runner(job):
        if not inside.acquire(blocking=False):
            overlap.append(job.id)
        else:
            time.sleep(0.05)
            inside.release()
        order.append(job.source)
        return {"success": True, "output": ""}

    q = _queue(runner)
    try:
        jobs = [q.submit(source=f"/tmp/{n}", mode="convos")[0] for n in "abcd"]
        for job in jobs:
            assert job.wait(10) is True
        assert order == ["/tmp/a", "/tmp/b", "/tmp/c", "/tmp/d"]
        assert overlap == []
    finally:
        q.stop()


def test_completed_job_is_journalled(tmp_path):
    def runner(job):
        return {
            "success": True,
            "output": "  Files processed: 7\n  Drawers filed: 42\n",
        }

    q = _queue(runner, tmp_path)
    try:
        job, _ = q.submit(source=str(tmp_path / "convos"), mode="convos", wing="sessions")
        assert job.wait(10)
    finally:
        q.stop()

    lines = (tmp_path / "mine_jobs.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "mine_job"
    assert record["job_id"] == job.id
    assert record["mode"] == "convos"
    assert record["wing"] == "sessions"
    assert record["state"] == "succeeded"
    assert record["files"] == 7
    assert record["drawers"] == 42
    assert record["duration_seconds"] >= 0


def test_failing_job_is_journalled_and_the_worker_survives(tmp_path):
    calls = []

    def runner(job):
        calls.append(job.source)
        if job.source.endswith("boom"):
            raise RuntimeError("chroma exploded")
        return {"success": True, "output": ""}

    q = _queue(runner, tmp_path)
    try:
        bad, _ = q.submit(source="/tmp/boom", mode="convos")
        assert bad.wait(10)
        assert bad.state == "failed"
        good, _ = q.submit(source="/tmp/ok", mode="convos")
        assert good.wait(10)
        assert good.state == "succeeded"
    finally:
        q.stop()

    records = [json.loads(line) for line in (tmp_path / "mine_jobs.jsonl").read_text().splitlines()]
    assert records[0]["state"] == "failed"
    assert "chroma exploded" in records[0]["error"]
    assert records[0]["error_class"] == "RuntimeError"
    assert records[1]["state"] == "succeeded"


def test_status_reports_running_queued_and_last_completed():
    gate = threading.Event()
    entered = threading.Event()

    def runner(job):
        entered.set()
        gate.wait(5)
        return {"success": True, "output": "Files processed: 1\nDrawers filed: 1\n"}

    q = _queue(runner)
    try:
        first, _ = q.submit(source="/tmp/a", mode="convos")
        assert entered.wait(5)
        second, _ = q.submit(source="/tmp/b", mode="convos")

        status = q.status()
        assert status["enabled"] is True
        assert status["running"]["job_id"] == first.id
        assert status["queued"] == 1
        assert status["queued_jobs"][0]["job_id"] == second.id
        assert status["queued_jobs"][0]["position"] == 1
        assert status["last_completed"] is None

        gate.set()
        assert second.wait(10)
        status = q.status()
        assert status["running"] is None
        assert status["queued"] == 0
        assert status["completed"] == 2
        assert status["last_completed"]["job_id"] == second.id
    finally:
        gate.set()
        q.stop()


def test_parse_mine_summary_leaves_unknown_counts_none():
    assert parse_mine_summary(None) == {"files": None, "drawers": None}
    assert parse_mine_summary("no numbers here") == {"files": None, "drawers": None}
    assert parse_mine_summary("Files processed: 3\nDrawers filed: 0\n") == {
        "files": 3,
        "drawers": 0,
    }


# ── tool_mine wiring ─────────────────────────────────────────────────────


@pytest.fixture
def hub_queue(monkeypatch, tmp_path):
    """Install a mine queue whose runner is controlled by the test."""
    palace = tmp_path / "palace"
    palace.mkdir()
    monkeypatch.setattr(
        mcp_server,
        "_config",
        type("C", (), {"palace_path": str(palace), "collection_name": "c", "backend": "chroma"})(),
    )
    holder = {}

    def install(runner):
        q = MineQueue(runner, journal_path=str(palace / "mine_jobs.jsonl"))
        q.start()
        monkeypatch.setattr(mcp_server, "_MINE_QUEUE", q)
        holder["q"] = q
        return q

    yield install, tmp_path
    q = holder.get("q")
    if q is not None:
        q.stop()


def test_tool_mine_enqueues_and_returns_a_job_id(hub_queue):
    install, tmp_path = hub_queue
    source = tmp_path / "convos"
    source.mkdir()
    release = threading.Event()
    install(lambda job: (release.wait(5), {"success": True, "output": ""})[1])

    try:
        result = mcp_server.tool_mine(source=str(source), mode="convos", wing="sessions")
        assert result["success"] is True
        assert result["queued"] is True
        assert result["job_id"].startswith("mine-")
        assert result["position"] in (0, 1)
        assert "output" not in result

        again = mcp_server.tool_mine(source=str(source), mode="convos", wing="sessions")
        assert again["job_id"] == result["job_id"]
        assert again["deduplicated"] is True
    finally:
        release.set()


def test_tool_mine_wait_true_returns_the_synchronous_contract(hub_queue):
    install, tmp_path = hub_queue
    source = tmp_path / "convos"
    source.mkdir()
    install(lambda job: {"success": True, "mode": job.mode, "dry_run": False, "output": "done"})

    result = mcp_server.tool_mine(source=str(source), mode="convos", wait=True, wait_timeout=10)
    assert result["success"] is True
    assert result["output"] == "done"
    assert result["mode"] == "convos"
    assert result["job_id"].startswith("mine-")
    assert "queued" not in result


def test_tool_mine_wait_true_times_out_without_losing_the_job(hub_queue):
    install, tmp_path = hub_queue
    source = tmp_path / "convos"
    source.mkdir()
    release = threading.Event()
    install(lambda job: (release.wait(10), {"success": True, "output": ""})[1])

    try:
        result = mcp_server.tool_mine(source=str(source), mode="convos", wait=True, wait_timeout=1)
        assert result["success"] is False
        assert result["error_class"] == "MineWaitTimeout"
        assert result["timed_out"] is True
        assert result["job_id"].startswith("mine-")
    finally:
        release.set()


def test_tool_mine_validation_errors_are_not_enqueued(hub_queue):
    install, tmp_path = hub_queue
    submitted = []
    install(lambda job: submitted.append(job.id) or {"success": True, "output": ""})

    assert mcp_server.tool_mine(source=str(tmp_path / "nope"))["success"] is False
    bad_mode = mcp_server.tool_mine(source=str(tmp_path), mode="nonsense")
    assert bad_mode["success"] is False
    assert "invalid mode" in bad_mode["error"]
    time.sleep(0.1)
    assert submitted == []


def test_status_and_statusz_expose_the_mine_queue(hub_queue, monkeypatch):
    install, tmp_path = hub_queue
    gate = threading.Event()
    entered = threading.Event()
    install(lambda job: (entered.set(), gate.wait(5), {"success": True, "output": ""})[2])
    source = tmp_path / "convos"
    source.mkdir()

    try:
        mcp_server.tool_mine(source=str(source), mode="convos")
        assert entered.wait(5)

        decorated = mcp_server._decorate_mcp_tool_result("mempalace_status", {"total_drawers": 0})
        assert decorated["mine_queue"]["enabled"] is True
        assert decorated["mine_queue"]["running"]["source"] == str(source)
    finally:
        gate.set()


def test_mine_queue_status_is_disabled_off_the_hub(monkeypatch):
    monkeypatch.setattr(mcp_server, "_MINE_QUEUE", None)
    assert mcp_server._mine_queue_status() == {"enabled": False}


def test_mine_async_kill_switch_keeps_the_inline_path(monkeypatch):
    monkeypatch.setenv("MEMPALACE_MCP_MINE_ASYNC", "0")
    monkeypatch.setattr(mcp_server, "_MINE_QUEUE", None)
    assert mcp_server._mine_async_disabled() is True
    assert mcp_server._start_mine_queue() is None
    assert mcp_server._MINE_QUEUE is None


# ── per-file lock scoping + watchdog ─────────────────────────────────────


def test_worker_releases_the_hub_write_lock_between_files(monkeypatch):
    """A peer's mutating tool must complete while a multi-file mine is mid-way.

    This is the property the whole change exists for: before it, an add_drawer
    from another agent waited for the entire transcript mine.
    """
    lock = mcp_server._RWLock()
    monkeypatch.setattr(mcp_server, "_HTTP_REQUEST_LOCK", lock)
    monkeypatch.setattr(mcp_server, "_MINE_QUEUE", object())  # arm the force=True path

    peer_done = threading.Event()
    first_file = threading.Event()
    finish = threading.Event()

    def mine_three_files():
        cursor = FileLockCursor(mcp_server._hub_mine_file_lock)
        for name in cursor.iter(["/p/one.jsonl", "/p/two.jsonl", "/p/three.jsonl"]):
            if name.endswith("one.jsonl"):
                first_file.set()
                # The peer only gets in if this hold is released with the file.
                peer_done.wait(5)
            time.sleep(0.01)
        cursor.release()
        finish.set()

    def peer_write():
        first_file.wait(5)
        with lock:
            peer_done.set()

    worker = threading.Thread(target=mine_three_files)
    peer = threading.Thread(target=peer_write)
    worker.start()
    peer.start()

    assert peer_done.wait(10) is True, "peer write never got the lock during the mine"
    assert finish.wait(10) is True
    worker.join(5)
    peer.join(5)


def test_watchdog_measures_per_file_holds_not_the_whole_job(monkeypatch):
    """A 3-file mine whose files each take longer than a tiny stall limit summed,
    but each shorter than it individually, must not trip the exit."""
    monkeypatch.setattr(mcp_server, "_HTTP_REQUEST_LOCK", mcp_server._RWLock())
    monkeypatch.setattr(mcp_server, "_MINE_QUEUE", object())
    monkeypatch.setattr(mcp_server, "_write_stall_inflight", None, raising=False)

    exit_secs = 0.30
    observed = []

    cursor = FileLockCursor(mcp_server._hub_mine_file_lock)
    for _name in cursor.iter(["a.jsonl", "b.jsonl", "c.jsonl"]):
        time.sleep(0.15)  # each file alone < exit_secs; the three summed > exit_secs
        inflight = mcp_server._write_stall_inflight
        assert inflight is not None, "the worker's per-file hold must be watched"
        elapsed = time.monotonic() - inflight["since"]
        observed.append(elapsed)
        assert mcp_server._write_stall_action(elapsed, 0.0, exit_secs, False) is None, (
            f"a progressing job tripped the watchdog after {elapsed:.2f}s"
        )
    cursor.release()

    assert sum(observed) > exit_secs, "the test did not actually exceed the limit in total"
    assert mcp_server._write_stall_inflight is None


def test_watchdog_still_trips_on_one_wedged_file(monkeypatch):
    """The safety property must survive: a single stuck file still exits."""
    monkeypatch.setattr(mcp_server, "_HTTP_REQUEST_LOCK", mcp_server._RWLock())
    monkeypatch.setattr(mcp_server, "_MINE_QUEUE", object())
    monkeypatch.setattr(mcp_server, "_write_stall_inflight", None, raising=False)

    with mcp_server._hub_mine_file_lock("/p/wedged.jsonl"):
        inflight = mcp_server._write_stall_inflight
        assert inflight["tool"] == "mempalace_mine[wedged.jsonl]"
        assert mcp_server._write_stall_action(200.0, 60.0, 180.0, False) == "exit"


def test_mine_tool_is_not_double_watched_while_the_queue_is_live(monkeypatch):
    """The dispatch-side watch on mempalace_mine only enqueues — watching it
    would report a queued-and-waiting call as a stalled chromadb write."""
    monkeypatch.setattr(mcp_server, "_write_stall_inflight", None, raising=False)

    monkeypatch.setattr(mcp_server, "_MINE_QUEUE", object())
    with mcp_server._write_stall_watch("mempalace_mine"):
        assert mcp_server._write_stall_inflight is None

    monkeypatch.setattr(mcp_server, "_MINE_QUEUE", None)
    with mcp_server._write_stall_watch("mempalace_mine"):
        assert mcp_server._write_stall_inflight["tool"] == "mempalace_mine"


def test_http_dispatch_routes_mine_lock_free_only_when_queued(monkeypatch):
    seen = {}

    def fake_handle(request):
        seen["locked"] = mcp_server._HTTP_REQUEST_LOCK._writer
        return {"ok": True}

    lock = mcp_server._RWLock()
    monkeypatch.setattr(mcp_server, "_HTTP_REQUEST_LOCK", lock)
    monkeypatch.setattr(mcp_server, "handle_request", fake_handle)
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "mempalace_mine", "arguments": {"source": "/tmp"}},
    }

    monkeypatch.setattr(mcp_server, "_MINE_QUEUE", object())
    mcp_server._http_dispatch(request)
    assert seen["locked"] is False

    monkeypatch.setattr(mcp_server, "_MINE_QUEUE", None)
    mcp_server._http_dispatch(request)
    assert seen["locked"] is True


# ── GuardedFileLock ──────────────────────────────────────────────────────


def test_guarded_lock_releases_a_hold_the_mine_leaked():
    """A miner that raises mid-file must not keep the hub's write lock.

    ``FileLockCursor.release()`` only covers the paths the miners guard:
    ``convo_miner`` runs its whole per-file loop outside its try/finally, and
    both it and ``miner`` open their collections outside it. On the hub that
    hold is the exclusive dispatch lock, so leaking one wedges every later
    write by every agent — worse than the restart storm this change fixes.
    """
    lock = _RecordingLock()
    with pytest.raises(RuntimeError):
        with GuardedFileLock(lock) as guarded:
            cursor = FileLockCursor(guarded)
            cursor.enter("a.jsonl")
            raise RuntimeError("corrupt index")  # cursor.release() never runs

    assert lock.events == [("acquire", "a.jsonl"), ("release", "a.jsonl")]


def test_guarded_lock_does_not_double_release_a_clean_mine():
    lock = _RecordingLock()
    with GuardedFileLock(lock) as guarded:
        cursor = FileLockCursor(guarded)
        cursor.enter("a.jsonl")
        cursor.enter("b.jsonl")
        cursor.release()

    assert lock.events == [
        ("acquire", "a.jsonl"),
        ("release", "a.jsonl"),
        ("acquire", "b.jsonl"),
        ("release", "b.jsonl"),
    ]
    assert guarded.release_dangling() is False


def test_guarded_lock_does_not_record_a_factory_that_failed_to_acquire():
    """An acquire that raises leaves nothing for release_dangling() to exit."""

    class _Boom:
        def __enter__(self):
            raise OSError("cannot lock")

        def __exit__(self, *exc):  # pragma: no cover - never entered
            raise AssertionError("exited a context that never entered")

    guarded = GuardedFileLock(lambda label: _Boom())
    with pytest.raises(OSError):
        with guarded("a.jsonl"):
            pass
    assert guarded.release_dangling() is False


def test_hub_mine_never_leaves_the_dispatch_lock_held(monkeypatch):
    """End to end: a mine that raises returns a structured error and the hub's
    exclusive write lock is free for the next caller."""
    lock = mcp_server._RWLock()
    monkeypatch.setattr(mcp_server, "_HTTP_REQUEST_LOCK", lock)

    def exploding_mine(**kwargs):
        cursor = FileLockCursor(kwargs["file_lock"])
        cursor.enter("big.jsonl")
        raise RuntimeError("chroma exploded mid-file")

    with patch("mempalace.miner.mine", side_effect=exploding_mine):
        result = mcp_server._mine_execute_on_hub("/tmp/src", mode="projects")

    assert result["success"] is False
    assert "chroma exploded" in result["error"]
    assert lock._writer is False
    # And the lock is genuinely re-acquirable, not just flagged free.
    with lock:
        pass
