"""Degraded MCP startup, durable logstream cursors, watchdog journal.

Lane: the hub must degrade, not die; watchers must arm themselves; cursors
need a home.

- Degraded startup: a held palace writer lease (the real flock mechanism the
  refusal uses) leaves the server serving logstream tools and reads, palace
  write tools return a structured retryable error naming the holder, and the
  server promotes itself to full mode once the lease is released.
- Durable cursors: per-agent get/set round-trip, atomicity, CLI, and
  `logstream watch --cursor-from-store`.
- Watchdog recovery journal: the write-stall exit path appends a structured
  jsonl line next to the palace.
"""

import json
import os
import threading
import time
from types import SimpleNamespace

import pytest

from mempalace import mcp_server
from mempalace.cli import cmd_logstream
from mempalace.logstream import LOGSTREAM_DB_FILENAME, Logstream
from mempalace.logstream_cursors import cursor_dir, cursor_get, cursor_set


# ── writer-lease degraded mode ─────────────────────────────────────────────


@pytest.fixture
def writer_lock_state(monkeypatch, tmp_path):
    """Isolate the module-level writer-lease state and the lock directory.

    HOME is redirected so mine_palace_lock's ~/.mempalace/locks never touches
    live state, per house rules.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(mcp_server, "_MCP_WRITER_LOCK_CM", None)
    monkeypatch.setattr(mcp_server, "_MCP_WRITER_READ_ONLY", False)
    monkeypatch.setattr(mcp_server, "_MCP_WRITER_LOCK_FAILED", False)
    monkeypatch.setattr(mcp_server, "_MCP_WRITER_LOCK_ERROR", "")
    monkeypatch.setattr(mcp_server, "_MCP_WRITER_LOCK_HOLDER", "")
    monkeypatch.setattr(mcp_server, "_MCP_WRITER_ATEXIT_REGISTERED", True)
    monkeypatch.setattr(mcp_server, "_discard_mcp_storage_handles", lambda: None)
    palace = tmp_path / "palace"
    palace.mkdir()
    monkeypatch.setattr(
        mcp_server, "_config", SimpleNamespace(palace_path=str(palace)), raising=False
    )
    yield str(palace)
    # Release anything a test left held so the flock does not leak.
    lock_cm = mcp_server._MCP_WRITER_LOCK_CM
    mcp_server._MCP_WRITER_LOCK_CM = None
    if lock_cm is not None:
        lock_cm.__exit__(None, None, None)


def _hold_palace_lock(monkeypatch, palace: str):
    """Hold the real mine_palace flock as if by a peer process.

    flock conflicts are per open-file-description, so a second independent
    open in this process contends exactly like a peer would — provided the
    in-process re-entrancy passthrough is disabled.
    """
    from mempalace import palace as palace_mod

    monkeypatch.setattr(palace_mod, "_held_by_this_process", lambda key: False)
    cm = palace_mod.mine_palace_lock(palace)
    cm.__enter__()
    return cm


class TestDegradedWriterLease:
    def test_held_lease_degrades_and_release_promotes(self, monkeypatch, writer_lock_state):
        """The lock-file mechanism the refusal uses: held → degraded, freed → full."""
        peer = _hold_palace_lock(monkeypatch, writer_lock_state)
        try:
            ok, reason = mcp_server._acquire_mcp_writer_lock()
            assert ok is False
            assert "already holds the palace lock" in reason
            mode = mcp_server._writer_mode()
            assert mode["mode"] == "degraded"
            assert "held by" in mode["holder"]
        finally:
            peer.__exit__(None, None, None)

        # Holder released the flock: the very next attempt promotes.
        ok, reason = mcp_server._acquire_mcp_writer_lock()
        assert ok is True and reason == ""
        assert mcp_server._writer_mode() == {"mode": "full"}

    def test_write_tool_refusal_is_structured_and_retryable(self, monkeypatch, writer_lock_state):
        peer = _hold_palace_lock(monkeypatch, writer_lock_state)
        try:
            refusal = mcp_server._mcp_peer_writer_refusal(7, "mempalace_add_drawer")
        finally:
            peer.__exit__(None, None, None)
        assert refusal is not None
        data = refusal["error"]["data"]
        assert data["retryable"] is True
        assert data["mode"] == "degraded"
        assert "held by" in data["holder"]
        assert data["tool"] == "mempalace_add_drawer"

    def test_logstream_tools_are_exempt_while_degraded(self, monkeypatch, writer_lock_state):
        peer = _hold_palace_lock(monkeypatch, writer_lock_state)
        try:
            for tool in ("mempalace_event_append", "mempalace_event_cursor_set"):
                assert mcp_server._mcp_peer_writer_refusal(1, tool) is None
            # Reads never consult the lease either.
            assert mcp_server._mcp_peer_writer_refusal(1, "mempalace_search") is None
        finally:
            peer.__exit__(None, None, None)

    def test_http_loop_starts_degraded_and_arms_retry(self, monkeypatch, writer_lock_state):
        events = []
        monkeypatch.setattr(mcp_server, "_READ_ONLY", False)
        monkeypatch.setattr(
            mcp_server,
            "_acquire_mcp_writer_lock",
            lambda: (False, "another writer owns the palace"),
        )
        monkeypatch.setattr(
            mcp_server,
            "_start_writer_lease_retry_thread",
            lambda: events.append("retry-thread"),
        )
        monkeypatch.setattr(mcp_server, "_refresh_vector_disabled_flag", lambda: None)
        monkeypatch.setattr(mcp_server, "_start_idle_exit_watchdog", lambda: None)
        monkeypatch.setattr(mcp_server, "_start_write_stall_watchdog", lambda: None)
        monkeypatch.setattr(mcp_server, "_serve_http", lambda host, port: events.append("serve"))

        mcp_server._run_http_loop()  # no SystemExit any more

        assert events == ["retry-thread", "serve"]

    def test_background_retry_promotes_when_lease_won(self, monkeypatch, writer_lock_state):
        promoted = threading.Event()
        attempts = []

        def fake_acquire():
            attempts.append(1)
            if len(attempts) < 2:
                return False, "still held"
            promoted.set()
            return True, ""

        monkeypatch.setattr(mcp_server, "_acquire_mcp_writer_lock", fake_acquire)
        monkeypatch.setattr(mcp_server, "_lease_retry_interval_secs", lambda: 0.01)

        mcp_server._start_writer_lease_retry_thread()

        assert promoted.wait(timeout=5.0), "retry thread never promoted"
        # Give the thread a beat to exit its loop, then confirm it stopped.
        time.sleep(0.05)
        final = len(attempts)
        time.sleep(0.05)
        assert len(attempts) == final, "retry thread kept running after promotion"

    def test_lease_retry_interval_is_bounded(self, monkeypatch):
        monkeypatch.setenv(mcp_server._LEASE_RETRY_SECS_ENV, "0.001")
        assert mcp_server._lease_retry_interval_secs() == mcp_server._LEASE_RETRY_SECS_MIN
        monkeypatch.setenv(mcp_server._LEASE_RETRY_SECS_ENV, "99999")
        assert mcp_server._lease_retry_interval_secs() == mcp_server._LEASE_RETRY_SECS_MAX
        monkeypatch.setenv(mcp_server._LEASE_RETRY_SECS_ENV, "nonsense")
        assert mcp_server._lease_retry_interval_secs() == mcp_server._LEASE_RETRY_SECS_DEFAULT

    def test_status_decoration_exposes_mode(self, monkeypatch, writer_lock_state):
        monkeypatch.setattr(mcp_server, "_MCP_WRITER_READ_ONLY", True)
        monkeypatch.setattr(mcp_server, "_MCP_WRITER_LOCK_HOLDER", "palace held by pid 4242")
        result = mcp_server._decorate_mcp_tool_result("mempalace_status", {})
        assert result["writer"]["mode"] == "degraded"
        assert result["writer"]["holder"] == "palace held by pid 4242"


# ── durable cursors ────────────────────────────────────────────────────────


class TestCursorStore:
    def test_round_trip(self, tmp_path):
        db = str(tmp_path / LOGSTREAM_DB_FILENAME)
        assert cursor_get(db, "mac-claude") is None
        record = cursor_set(db, "mac-claude", "evt_0000000042")
        assert record["event_id"] == "evt_0000000042"
        read = cursor_get(db, "mac-claude")
        assert read["event_id"] == "evt_0000000042"
        assert read["agent"] == "mac-claude"
        assert read["updated_at"]
        # Overwrite moves the bookmark.
        cursor_set(db, "mac-claude", "evt_0000000043")
        assert cursor_get(db, "mac-claude")["event_id"] == "evt_0000000043"
        # Per-agent isolation.
        assert cursor_get(db, "windows-codex") is None

    def test_atomic_write_preserves_previous_cursor_on_failure(self, tmp_path, monkeypatch):
        db = str(tmp_path / LOGSTREAM_DB_FILENAME)
        cursor_set(db, "mac-claude", "evt_1")

        real_replace = os.replace

        def failing_replace(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", failing_replace)
        with pytest.raises(OSError):
            cursor_set(db, "mac-claude", "evt_2")
        monkeypatch.setattr(os, "replace", real_replace)

        # Previous cursor intact, no torn state, temp file cleaned up.
        assert cursor_get(db, "mac-claude")["event_id"] == "evt_1"
        leftovers = [p for p in os.listdir(cursor_dir(db)) if p.endswith(".tmp")]
        assert leftovers == []

    def test_rejects_traversal_and_empty_ids(self, tmp_path):
        db = str(tmp_path / LOGSTREAM_DB_FILENAME)
        for bad in ("", "../evil", "a/b", ".", "..", None):
            with pytest.raises(ValueError):
                cursor_set(db, bad, "evt_1")
        with pytest.raises(ValueError):
            cursor_set(db, "ok-agent", "  ")

    def test_corrupt_file_reported_not_absent(self, tmp_path):
        db = str(tmp_path / LOGSTREAM_DB_FILENAME)
        cursor_set(db, "mac-claude", "evt_1")
        path = os.path.join(cursor_dir(db), "mac-claude.json")
        with open(path, "w") as f:
            f.write("{not json")
        record = cursor_get(db, "mac-claude")
        assert record["corrupt"] is True
        assert record["event_id"] is None


class TestCursorMcpTools:
    @pytest.fixture(autouse=True)
    def _palace(self, tmp_path, monkeypatch):
        palace = tmp_path / "palace"
        palace.mkdir()
        monkeypatch.setattr(
            mcp_server, "_config", SimpleNamespace(palace_path=str(palace)), raising=False
        )

    def test_get_set_round_trip(self):
        result = mcp_server.tool_event_cursor_get(agent="mac-claude")
        assert result == {"success": True, "agent": "mac-claude", "cursor": None}

        result = mcp_server.tool_event_cursor_set(agent="mac-claude", event_id="evt_7")
        assert result["success"] is True
        assert result["cursor"]["event_id"] == "evt_7"

        result = mcp_server.tool_event_cursor_get(agent="mac-claude")
        assert result["cursor"]["event_id"] == "evt_7"

    def test_invalid_inputs_are_structured_errors(self):
        assert mcp_server.tool_event_cursor_set(agent="../x", event_id="e")["success"] is False
        assert mcp_server.tool_event_cursor_set(agent="ok", event_id="")["success"] is False
        assert mcp_server.tool_event_cursor_get(agent="../x")["success"] is False

    def test_registered_as_lease_exempt_mutating_tool(self):
        assert "mempalace_event_cursor_get" in mcp_server.TOOLS
        assert "mempalace_event_cursor_set" in mcp_server.TOOLS
        assert "mempalace_event_cursor_set" in mcp_server._MUTATING_TOOLS
        assert "mempalace_event_cursor_set" in mcp_server._PEER_WRITER_EXEMPT_TOOLS
        assert "mempalace_event_cursor_get" not in mcp_server._MUTATING_TOOLS


# ── CLI: logstream cursor + watch --cursor-from-store ──────────────────────


def _cursor_args(palace, action, **overrides):
    fields = dict(
        palace=palace,
        logstream_action="cursor",
        cursor_action=action,
        agent="mac-claude",
        event_id=None,
        json=True,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _watch_args(palace, **overrides):
    fields = dict(
        palace=palace,
        logstream_action="watch",
        agent=None,
        stream=None,
        room=None,
        topic=None,
        type=None,
        status=None,
        to_agent=None,
        from_agent=None,
        exclude_from_agent=None,
        correlation_id=None,
        since_event_id=None,
        state_file=None,
        cursor_from_store=False,
        from_start=False,
        follow=False,
        idle_exit_ms=400,
        poll_timeout_ms=60,
        limit=50,
        json=True,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _append_event(palace, to_agent="mac-claude", body="hello"):
    ls = Logstream(db_path=os.path.join(palace, LOGSTREAM_DB_FILENAME))
    try:
        return ls.append_event(
            type="task.request",
            stream="project/mempalace",
            room="delegation",
            from_agent="windows-codex",
            to_agent=to_agent,
            body=body,
        )
    finally:
        ls.close()


class TestCursorCli:
    def test_set_then_get(self, tmp_path, capsys):
        palace = str(tmp_path)
        cmd_logstream(_cursor_args(palace, "set", event_id="evt_9"))
        out = json.loads(capsys.readouterr().out)
        assert out["event_id"] == "evt_9"

        cmd_logstream(_cursor_args(palace, "get"))
        out = json.loads(capsys.readouterr().out)
        assert out["cursor"]["event_id"] == "evt_9"

    def test_get_absent_is_null(self, tmp_path, capsys):
        cmd_logstream(_cursor_args(str(tmp_path), "get", agent="nobody"))
        out = json.loads(capsys.readouterr().out)
        assert out["cursor"] is None

    def test_set_requires_event_id(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc:
            cmd_logstream(_cursor_args(str(tmp_path), "set"))
        assert exc.value.code == 1
        assert "requires --event-id" in json.loads(capsys.readouterr().out)["error"]


class TestWatchCursorFromStore:
    def test_resumes_after_stored_cursor(self, tmp_path, capsys):
        palace = str(tmp_path)
        first = _append_event(palace, body="already processed")
        second = _append_event(palace, body="new mail")
        db = os.path.join(palace, LOGSTREAM_DB_FILENAME)
        cursor_set(db, "mac-claude", first["id"])

        cmd_logstream(_watch_args(palace, agent="mac-claude", cursor_from_store=True))
        payload = json.loads(capsys.readouterr().out.strip())
        ids = [e["id"] for e in payload["events"]]
        assert ids == [second["id"]]

    def test_requires_agent(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc:
            cmd_logstream(_watch_args(str(tmp_path), cursor_from_store=True))
        assert exc.value.code == 1
        assert "requires --agent" in json.loads(capsys.readouterr().out)["error"]

    def test_no_stored_cursor_still_starts_at_tip(self, tmp_path, capsys):
        palace = str(tmp_path)
        _append_event(palace, body="old news")
        with pytest.raises(SystemExit) as exc:
            cmd_logstream(
                _watch_args(palace, agent="mac-claude", cursor_from_store=True, idle_exit_ms=200)
            )
        assert exc.value.code == 2  # idle: the pre-existing event was not replayed
        payload = json.loads(capsys.readouterr().out.strip())
        assert payload["timed_out"] is True

    def test_explicit_since_event_id_outranks_store(self, tmp_path, capsys):
        palace = str(tmp_path)
        first = _append_event(palace, body="one")
        second = _append_event(palace, body="two")
        db = os.path.join(palace, LOGSTREAM_DB_FILENAME)
        # Store points past everything; explicit flag rewinds to after `first`.
        cursor_set(db, "mac-claude", second["id"])
        third = _append_event(palace, body="three")

        cmd_logstream(
            _watch_args(
                palace,
                agent="mac-claude",
                cursor_from_store=True,
                since_event_id=first["id"],
            )
        )
        payload = json.loads(capsys.readouterr().out.strip())
        ids = [e["id"] for e in payload["events"]]
        assert ids == [second["id"], third["id"]]


# ── watchdog recovery journal ──────────────────────────────────────────────


class TestWatchdogJournal:
    def test_appends_structured_line(self, tmp_path):
        palace = str(tmp_path)
        mcp_server._append_watchdog_journal("mempalace_add_drawer", 312.7, palace_path=palace)
        mcp_server._append_watchdog_journal("mempalace_mine", 601.2, palace_path=palace)

        path = os.path.join(palace, mcp_server._WATCHDOG_JOURNAL_FILENAME)
        lines = [json.loads(line) for line in open(path).read().splitlines()]
        assert len(lines) == 2
        first = lines[0]
        assert first["event"] == "write_stall_exit"
        assert first["tool"] == "mempalace_add_drawer"
        assert first["stall_seconds"] == 312.7
        assert first["pid"] == os.getpid()
        assert first["exit_code"] == mcp_server._WRITE_STALL_EXIT_CODE
        assert first["ts"].endswith("Z")
        assert lines[1]["tool"] == "mempalace_mine"

    def test_never_raises_without_palace(self, monkeypatch):
        monkeypatch.setattr(mcp_server, "_config", SimpleNamespace(palace_path=None), raising=False)
        mcp_server._append_watchdog_journal("mempalace_add_drawer", 10.0)  # no crash

    def test_never_raises_on_unwritable_path(self, tmp_path):
        mcp_server._append_watchdog_journal(
            "mempalace_add_drawer", 10.0, palace_path=str(tmp_path / "missing" / "dir")
        )  # no crash

    def test_exit_branch_journals_before_exit(self, monkeypatch, tmp_path):
        """The watchdog thread's exit action journals, then calls os._exit(75)."""
        calls = []
        monkeypatch.setattr(
            mcp_server, "_config", SimpleNamespace(palace_path=str(tmp_path)), raising=False
        )
        monkeypatch.setenv(mcp_server._WRITE_STALL_WARN_ENV, "0")
        monkeypatch.setenv(mcp_server._WRITE_STALL_EXIT_ENV, "0.2")

        exited = threading.Event()

        def fake_exit(code):
            calls.append(("exit", code))
            exited.set()
            # Park the daemon thread instead of raising: an exception here
            # would trip pytest's unhandled-thread-exception warning.
            while True:
                time.sleep(3600)

        monkeypatch.setattr(mcp_server.os, "_exit", fake_exit)

        with mcp_server._write_stall_lock:
            mcp_server._write_stall_inflight = {
                "tool": "mempalace_add_drawer",
                "since": time.monotonic() - 500.0,
                "warned": True,
            }
        try:
            mcp_server._start_write_stall_watchdog()
            assert exited.wait(timeout=5.0), "watchdog never fired"
        finally:
            with mcp_server._write_stall_lock:
                mcp_server._write_stall_inflight = None

        assert ("exit", mcp_server._WRITE_STALL_EXIT_CODE) in calls
        path = os.path.join(str(tmp_path), mcp_server._WATCHDOG_JOURNAL_FILENAME)
        record = json.loads(open(path).read().splitlines()[0])
        assert record["tool"] == "mempalace_add_drawer"
        assert record["stall_seconds"] >= 500.0
