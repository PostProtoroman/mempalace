"""The hub-restart lease gap: per-call writer leases and the hub grace window.

Failure this closes (live 2026-09-01 x2, 2026-09-02 x1): the HTTP hub holds
the palace writer flock for its whole lifetime. While it restarts, the flock
is free for a few seconds. A session's stdio MCP server that took a mutating
call in that window could not forward (hub gone), fell back to a direct write,
lazily acquired the flock — and then held it for its own process lifetime,
which is hours. The restarted hub came back degraded and stayed degraded, and
every hub-routed writer (codex capture) was blocked behind it.

Two independent defences, tested here:

1. Per-call lease scoping. A stdio/embedded server takes the flock for the
   duration of one mutating tool call and gives it straight back, so the worst
   case is one call's worth of contention rather than a whole session. The
   writable HTTP hub keeps its lifetime hold — it is the designated writer.
2. A hub restart grace window. A mutating call that finds the hub missing
   returns a structured retryable error for the first N seconds of the outage
   instead of writing directly; reads still fall back immediately.

Plus the property both rely on: flocks are released by the OS when a process
dies, so even a kill -9 mid-write cannot strand the palace.
"""

import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from types import SimpleNamespace

import pytest

from mempalace import mcp_server


MUTATING_TOOL = "mempalace_add_drawer"

# Captured before any fixture stubs it, so one test can exercise the real
# handle-closing that per-call release depends on.
_REAL_DISCARD_HANDLES = mcp_server._discard_mcp_storage_handles


@pytest.fixture
def lease_env(monkeypatch, tmp_path):
    """Isolated writer-lease state, lock directory and palace path.

    HOME is redirected so ``mine_palace_lock``'s ``~/.mempalace/locks`` never
    touches live state, per house rules.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("MEMPALACE_MCP_ALLOW_PEER_WRITER", raising=False)
    monkeypatch.delenv("MEMPALACE_HUB_GRACE_SECS", raising=False)
    monkeypatch.setattr(mcp_server, "_MCP_WRITER_LOCK_CM", None)
    monkeypatch.setattr(mcp_server, "_MCP_WRITER_READ_ONLY", False)
    monkeypatch.setattr(mcp_server, "_MCP_WRITER_LOCK_FAILED", False)
    monkeypatch.setattr(mcp_server, "_MCP_WRITER_LOCK_ERROR", "")
    monkeypatch.setattr(mcp_server, "_MCP_WRITER_LOCK_HOLDER", "")
    monkeypatch.setattr(mcp_server, "_MCP_WRITER_LOCK_HOLDER_PID", None)
    monkeypatch.setattr(mcp_server, "_MCP_WRITER_LEASE_DEPTH", 0)
    monkeypatch.setattr(mcp_server, "_MCP_WRITER_LEASE_SCOPE", mcp_server._LEASE_SCOPE_PER_CALL)
    monkeypatch.setattr(mcp_server, "_MCP_WRITER_ATEXIT_REGISTERED", True)
    monkeypatch.setattr(mcp_server, "_discard_mcp_storage_handles", lambda: None)
    palace = tmp_path / "palace"
    palace.mkdir()
    monkeypatch.setattr(
        mcp_server, "_config", SimpleNamespace(palace_path=str(palace)), raising=False
    )
    yield str(palace)
    lock_cm = mcp_server._MCP_WRITER_LOCK_CM
    mcp_server._MCP_WRITER_LOCK_CM = None
    if lock_cm is not None:
        lock_cm.__exit__(None, None, None)


def _install_fake_tool(monkeypatch, handler):
    """Replace one mutating tool's handler and neutralize unrelated preflight gates."""
    tools = dict(mcp_server.TOOLS)
    tools[MUTATING_TOOL] = {
        "description": "test double",
        "input_schema": {"type": "object", "properties": {}},
        "handler": handler,
    }
    monkeypatch.setattr(mcp_server, "TOOLS", tools)
    for gate in (
        "_mcp_read_only_refusal",
        "_mcp_sqlite_integrity_refusal",
        "_mcp_stale_library_refusal",
        "_mcp_diverged_index_refusal",
    ):
        monkeypatch.setattr(mcp_server, gate, lambda req_id, tool_name: None)


def _call(tool_name=MUTATING_TOOL, req_id=1):
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": {}},
    }


def _hold_palace_lock(monkeypatch, palace: str):
    """Hold the real mine_palace flock as a peer would.

    flock conflicts are per open-file-description, so a second independent
    open in this process contends exactly like a peer — provided the
    in-process re-entrancy passthrough is disabled.
    """
    from mempalace import palace as palace_mod

    monkeypatch.setattr(palace_mod, "_held_by_this_process", lambda key: False)
    cm = palace_mod.mine_palace_lock(palace)
    cm.__enter__()
    return cm


# ── 1. per-call lease scoping ──────────────────────────────────────────────


class TestPerCallLeaseScoping:
    def test_lease_is_held_only_during_the_mutating_call(self, monkeypatch, lease_env):
        """Not before, held during, released after — the whole point of the lane."""
        observed = {}

        def handler():
            observed["during"] = mcp_server._MCP_WRITER_LOCK_CM is not None
            return {"ok": True}

        _install_fake_tool(monkeypatch, handler)

        assert mcp_server._MCP_WRITER_LOCK_CM is None, "lease held before any call"
        response = mcp_server.handle_request(_call())

        assert "error" not in response, response
        assert observed["during"] is True, "handler ran without the writer lease"
        assert mcp_server._MCP_WRITER_LOCK_CM is None, (
            "stdio server kept the palace writer lease after the call — this is "
            "the hub-restart outage"
        )

    def test_lease_is_released_even_when_the_tool_raises(self, monkeypatch, lease_env):
        def handler():
            raise RuntimeError("boom")

        _install_fake_tool(monkeypatch, handler)
        response = mcp_server.handle_request(_call())

        assert "error" in response
        assert mcp_server._MCP_WRITER_LOCK_CM is None

    def test_read_tools_never_touch_the_lease(self, monkeypatch, lease_env):
        calls = []
        monkeypatch.setattr(
            mcp_server,
            "_acquire_mcp_writer_lock",
            lambda: (calls.append(1), (True, ""))[1],
        )
        assert mcp_server._mcp_peer_writer_refusal(1, "mempalace_search") is None
        assert calls == []

    def test_hub_lifetime_scope_keeps_the_lease(self, monkeypatch, lease_env):
        """The writable hub is the designated writer: it must not give the flock back."""
        monkeypatch.setattr(mcp_server, "_MCP_WRITER_LEASE_SCOPE", mcp_server._LEASE_SCOPE_LIFETIME)
        _install_fake_tool(monkeypatch, lambda: {"ok": True})

        mcp_server.handle_request(_call())

        assert mcp_server._MCP_WRITER_LOCK_CM is not None

    def test_nested_lease_scopes_release_once_at_the_outermost_exit(self, monkeypatch, lease_env):
        """A mine inside a tool (or a second worker thread) must not free the flock early."""
        assert mcp_server._acquire_mcp_writer_lock() == (True, "")
        with mcp_server._mcp_scoped_writer_lease(MUTATING_TOOL):
            with mcp_server._mcp_scoped_writer_lease(MUTATING_TOOL):
                assert mcp_server._MCP_WRITER_LOCK_CM is not None
            assert mcp_server._MCP_WRITER_LOCK_CM is not None, "inner scope released the lease"
        assert mcp_server._MCP_WRITER_LOCK_CM is None

    def test_reentrant_mine_inside_a_tool_still_works(self, monkeypatch, lease_env):
        """mine_palace_lock stays re-entrant under a held per-call lease."""
        from mempalace import palace as palace_mod

        entered = []

        def handler():
            with palace_mod.mine_palace_lock(lease_env):
                entered.append(True)
            return {"ok": True}

        _install_fake_tool(monkeypatch, handler)
        response = mcp_server.handle_request(_call())

        assert entered == [True], response
        assert mcp_server._MCP_WRITER_LOCK_CM is None

    def test_refusal_still_structured_when_a_peer_holds_the_lease(self, monkeypatch, lease_env):
        peer = _hold_palace_lock(monkeypatch, lease_env)
        try:
            _install_fake_tool(monkeypatch, lambda: pytest.fail("handler must not run"))
            response = mcp_server.handle_request(_call())
        finally:
            peer.__exit__(None, None, None)

        data = response["error"]["data"]
        assert data["retryable"] is True
        assert data["mode"] == "degraded"


class TestWriterModeTruthfulness:
    def test_per_call_server_reports_full_between_calls(self, monkeypatch, lease_env):
        """Holding nothing between calls is normal, not degraded."""
        _install_fake_tool(monkeypatch, lambda: {"ok": True})
        mcp_server.handle_request(_call())
        assert mcp_server._writer_mode() == {"mode": "full"}

    def test_departed_holder_does_not_leave_a_stale_degraded_flag(self, monkeypatch, lease_env):
        """A hub that restarted must not be reported as still owning the palace."""
        monkeypatch.setattr(mcp_server, "_MCP_WRITER_READ_ONLY", True)
        monkeypatch.setattr(mcp_server, "_MCP_WRITER_LOCK_HOLDER", "palace is held by PID 4242")
        monkeypatch.setattr(mcp_server, "_MCP_WRITER_LOCK_HOLDER_PID", 4242)
        from mempalace import server_registry

        monkeypatch.setattr(server_registry, "_pid_alive", lambda pid: False)

        assert mcp_server._writer_mode() == {"mode": "full"}

    def test_live_holder_still_reports_degraded(self, monkeypatch, lease_env):
        from mempalace import server_registry

        monkeypatch.setattr(mcp_server, "_MCP_WRITER_READ_ONLY", True)
        monkeypatch.setattr(mcp_server, "_MCP_WRITER_LOCK_HOLDER", "palace is held by PID 4242")
        monkeypatch.setattr(mcp_server, "_MCP_WRITER_LOCK_HOLDER_PID", 4242)
        monkeypatch.setattr(server_registry, "_pid_alive", lambda pid: True)

        mode = mcp_server._writer_mode()
        assert mode["mode"] == "degraded"
        assert mode["holder"] == "palace is held by PID 4242"

    def test_unknown_holder_pid_keeps_the_conservative_degraded_report(
        self, monkeypatch, lease_env
    ):
        monkeypatch.setattr(mcp_server, "_MCP_WRITER_READ_ONLY", True)
        monkeypatch.setattr(mcp_server, "_MCP_WRITER_LOCK_HOLDER", "another writer")
        monkeypatch.setattr(mcp_server, "_MCP_WRITER_LOCK_HOLDER_PID", None)
        assert mcp_server._writer_mode()["mode"] == "degraded"

    def test_refusal_records_the_holder_pid(self, monkeypatch, lease_env):
        peer = _hold_palace_lock(monkeypatch, lease_env)
        try:
            ok, _reason = mcp_server._acquire_mcp_writer_lock()
        finally:
            peer.__exit__(None, None, None)
        assert ok is False
        assert mcp_server._MCP_WRITER_LOCK_HOLDER_PID == os.getpid()


# ── 2. two-process race: the hub must win the flock within one retry ───────


# Readiness is signalled through a file, not stdout: importing mcp_server
# redirects stdout at import time (the MCP JSON-RPC guard), so a child's
# "ready" line would never reach the parent's pipe.
_CHILD_WRITE_LOOP = textwrap.dedent(
    """
    import os, sys, time
    from types import SimpleNamespace

    palace, progress, seconds, scope = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    deadline = time.time() + seconds

    from mempalace import mcp_server as m

    m._MCP_WRITER_LEASE_SCOPE = scope
    m._config = SimpleNamespace(palace_path=palace)
    m._discard_mcp_storage_handles = lambda: None
    m._MCP_WRITER_ATEXIT_REGISTERED = True
    for gate in ("_mcp_read_only_refusal", "_mcp_sqlite_integrity_refusal",
                 "_mcp_stale_library_refusal", "_mcp_diverged_index_refusal"):
        setattr(m, gate, lambda req_id, tool_name: None)

    tools = dict(m.TOOLS)
    tools["mempalace_add_drawer"] = {
        "description": "test double",
        "input_schema": {"type": "object", "properties": {}},
        "handler": lambda: (time.sleep(0.02), {"ok": True})[1],
    }
    m.TOOLS = tools

    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": "mempalace_add_drawer", "arguments": {}}}

    done = 0
    while time.time() < deadline:
        response = m.handle_request(request)
        assert "error" not in response, response
        done += 1
        with open(progress, "w") as fh:
            fh.write(str(done))
    """
)

_CHILD_HOLD_LOCK = textwrap.dedent(
    """
    import sys, time
    from mempalace.palace import mine_palace_lock

    with mine_palace_lock(sys.argv[1]):
        with open(sys.argv[2], "w") as fh:
            fh.write("held")
        time.sleep(120)
    """
)


def _child_env(home: str) -> dict:
    env = dict(os.environ)
    env["HOME"] = home
    env["PYTHONPATH"] = os.pathsep.join(
        [str(os.path.dirname(os.path.dirname(os.path.abspath(mcp_server.__file__))))]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env.pop("MEMPALACE_MCP_ALLOW_PEER_WRITER", None)
    return env


def _wait_for_progress(path, child, timeout=30.0, minimum=1):
    """Wait until the child has completed ``minimum`` iterations."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value = int(path.read_text().strip() or 0)
        except (OSError, ValueError):
            value = 0
        if value >= minimum:
            return value
        if child.poll() is not None:
            raise AssertionError(f"child exited early: {child.stderr.read()}")
        time.sleep(0.02)
    raise AssertionError(f"child never reached {minimum} iterations: {child.stderr.read()}")


@pytest.mark.skipif(os.name == "nt", reason="POSIX flock semantics")
class TestTwoProcessRace:
    def _run(self, tmp_path, monkeypatch, scope):
        """Contend for the palace flock against a live child server that is writing.

        Returns (acquired_within_2s, iterations_before, iterations_after).
        """
        home = tmp_path / "home"
        home.mkdir()
        palace = tmp_path / "palace"
        palace.mkdir()
        progress = tmp_path / "progress"

        child = subprocess.Popen(
            [sys.executable, "-c", _CHILD_WRITE_LOOP, str(palace), str(progress), "30", scope],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_child_env(str(home)),
        )
        try:
            # Contend with a server that is actively mutating, not an idle one.
            before = _wait_for_progress(progress, child, minimum=3)

            from mempalace import palace as palace_mod

            monkeypatch.setenv("HOME", str(home))
            deadline = time.monotonic() + 2.0
            acquired = False
            last_error = ""
            while time.monotonic() < deadline and not acquired:
                try:
                    with palace_mod.mine_palace_lock(str(palace)):
                        acquired = True
                except palace_mod.MineAlreadyRunning as exc:
                    last_error = str(exc)
                    time.sleep(0.02)
            assert child.poll() is None, f"child exited early: {child.stderr.read()}"
            after = _wait_for_progress(progress, child, minimum=before + 1)
            return acquired, before, after, last_error
        finally:
            child.kill()
            child.wait(timeout=10)

    def test_hub_wins_the_lease_while_a_session_server_keeps_writing(self, tmp_path, monkeypatch):
        """A restarting hub gets the flock within one retry interval, not never."""
        acquired, before, after, last_error = self._run(
            tmp_path, monkeypatch, mcp_server._LEASE_SCOPE_PER_CALL
        )
        assert acquired, f"hub never won the palace lease within 2s: {last_error}"
        assert after > before, "child stopped writing; the contention was not concurrent"

    def test_lifetime_scope_is_what_used_to_lock_the_hub_out(self, tmp_path, monkeypatch):
        """Falsifier for the test above: with the old lifetime hold, the hub loses.

        Same child, same 2s of retries, only the lease scope differs. If this
        ever starts passing the acquire loop has stopped proving anything.
        """
        acquired, _before, _after, last_error = self._run(
            tmp_path, monkeypatch, mcp_server._LEASE_SCOPE_LIFETIME
        )
        assert not acquired, "lifetime scope released the palace lease between calls"
        assert "is held by" in last_error


# ── 3. crash safety: the OS releases the flock ─────────────────────────────


@pytest.mark.skipif(os.name == "nt", reason="POSIX flock semantics")
def test_sigkill_mid_write_releases_the_flock_immediately(tmp_path, monkeypatch):
    """A stdio process killed mid-write cannot strand the palace.

    This is what makes per-call scoping safe to rely on: even the release path
    never running is survivable, because flock ownership dies with the file
    description.
    """
    home = tmp_path / "home"
    home.mkdir()
    palace = tmp_path / "palace"
    palace.mkdir()
    held = tmp_path / "held"

    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD_HOLD_LOCK, str(palace), str(held)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_child_env(str(home)),
    )
    try:
        deadline = time.monotonic() + 30.0
        while not held.exists() and time.monotonic() < deadline:
            assert child.poll() is None, child.stderr.read()
            time.sleep(0.02)
        assert held.exists(), child.stderr.read()

        from mempalace import palace as palace_mod

        monkeypatch.setenv("HOME", str(home))
        with pytest.raises(palace_mod.MineAlreadyRunning) as busy:
            with palace_mod.mine_palace_lock(str(palace)):
                pass
        assert busy.value.holder_pid == child.pid

        child.send_signal(signal.SIGKILL)
        child.wait(timeout=10)

        # No sleep, no retry loop: the kernel dropped the lock with the process.
        with palace_mod.mine_palace_lock(str(palace)):
            pass
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


# ── 4. hub restart grace window ────────────────────────────────────────────


@pytest.fixture
def hub_state(monkeypatch, lease_env):
    monkeypatch.setattr(mcp_server, "_hub_last_seen_monotonic", None)
    monkeypatch.setattr(mcp_server, "_hub_unavailable_since", None)
    monkeypatch.setattr(mcp_server, "_hub_proxy_target", lambda: None)
    return lease_env


class TestHubGraceWindow:
    def test_mutating_call_inside_the_window_is_retryable_not_direct(self, monkeypatch, hub_state):
        monkeypatch.setattr(mcp_server, "_hub_last_seen_monotonic", time.monotonic() - 1.0)
        monkeypatch.setattr(mcp_server, "_hub_unavailable_since", time.monotonic() - 1.0)
        monkeypatch.setattr(
            mcp_server,
            "handle_request",
            lambda request: pytest.fail("mutating call fell back to a direct write"),
        )

        response = mcp_server._dispatch_stdio_request(_call())

        assert response["error"]["code"] == -32001
        data = response["error"]["data"]
        assert data["retryable"] is True
        assert data["grace_window_secs"] == mcp_server._HUB_GRACE_SECS_DEFAULT
        assert "restarting" in response["error"]["message"]

    def test_mutating_call_after_the_window_resumes_direct(self, monkeypatch, hub_state):
        monkeypatch.setenv(mcp_server._HUB_GRACE_SECS_ENV, "5")
        monkeypatch.setattr(mcp_server, "_hub_last_seen_monotonic", time.monotonic() - 600.0)
        monkeypatch.setattr(mcp_server, "_hub_unavailable_since", time.monotonic() - 600.0)
        handled = []
        monkeypatch.setattr(
            mcp_server, "handle_request", lambda request: handled.append(request) or {"ok": True}
        )

        assert mcp_server._dispatch_stdio_request(_call()) == {"ok": True}
        assert len(handled) == 1

    def test_reads_always_fall_back_inside_the_window(self, monkeypatch, hub_state):
        monkeypatch.setattr(mcp_server, "_hub_last_seen_monotonic", time.monotonic())
        monkeypatch.setattr(mcp_server, "_hub_unavailable_since", time.monotonic())
        handled = []
        monkeypatch.setattr(
            mcp_server, "handle_request", lambda request: handled.append(request) or {"ok": True}
        )

        read = dict(_call(tool_name="mempalace_search"))
        assert mcp_server._dispatch_stdio_request(read) == {"ok": True}
        assert len(handled) == 1

    def test_never_seen_hub_writes_directly(self, monkeypatch, hub_state):
        """A session that never met a hub has no reason to stall its writes."""
        handled = []
        monkeypatch.setattr(
            mcp_server, "handle_request", lambda request: handled.append(request) or {"ok": True}
        )
        assert mcp_server._dispatch_stdio_request(_call()) == {"ok": True}
        assert len(handled) == 1
        assert mcp_server._hub_unavailable_since is None

    def test_disappearing_hub_starts_the_outage_clock(self, monkeypatch, lease_env):
        """Seeing a real serverinfo record and then losing it opens the window.

        This walks the actual registry lookup rather than the stubbed target,
        because "the record vanished" is precisely what a hub restart looks
        like from a session server.
        """
        from mempalace import server_registry

        monkeypatch.setattr(mcp_server, "_hub_last_seen_monotonic", None)
        monkeypatch.setattr(mcp_server, "_hub_unavailable_since", None)
        monkeypatch.setattr(mcp_server, "_hub_proxy_announced", True)

        live = {"pid": os.getpid() + 1, "host": "127.0.0.1", "port": 8765, "scheme": "http"}
        monkeypatch.setattr(server_registry, "read_live_serverinfo", lambda palace: live)
        monkeypatch.setattr(
            server_registry, "urlopen_with_server_tokens", lambda *a, **k: pytest.fail("no dial")
        )

        assert mcp_server._hub_proxy_target() is not None
        assert mcp_server._hub_last_seen_monotonic is not None
        assert mcp_server._hub_unavailable_since is None

        # Hub restarts: clear_serverinfo removed the record.
        monkeypatch.setattr(server_registry, "read_live_serverinfo", lambda palace: None)
        assert mcp_server._hub_proxy_target() is None
        assert mcp_server._hub_unavailable_since is not None
        assert mcp_server._hub_grace_active() is True

    def test_grace_window_env_and_config_knobs(self, monkeypatch, lease_env):
        monkeypatch.setenv(mcp_server._HUB_GRACE_SECS_ENV, "12.5")
        assert mcp_server._hub_grace_window_secs() == 12.5
        monkeypatch.setenv(mcp_server._HUB_GRACE_SECS_ENV, "-3")
        assert mcp_server._hub_grace_window_secs() == 0.0
        monkeypatch.setenv(mcp_server._HUB_GRACE_SECS_ENV, "999999")
        assert mcp_server._hub_grace_window_secs() == mcp_server._HUB_GRACE_SECS_MAX
        monkeypatch.setenv(mcp_server._HUB_GRACE_SECS_ENV, "nonsense")
        assert mcp_server._hub_grace_window_secs() == mcp_server._HUB_GRACE_SECS_DEFAULT

        monkeypatch.delenv(mcp_server._HUB_GRACE_SECS_ENV, raising=False)
        monkeypatch.setattr(
            mcp_server,
            "_config",
            SimpleNamespace(palace_path=lease_env, hub_grace_secs=7.0),
            raising=False,
        )
        assert mcp_server._hub_grace_window_secs() == 7.0

    def test_invalid_env_still_honours_configured_grace(self, monkeypatch, lease_env):
        """A typo in the env must not silently discard hub.grace_secs."""
        monkeypatch.setenv(mcp_server._HUB_GRACE_SECS_ENV, "nonsense")
        monkeypatch.setattr(
            mcp_server,
            "_config",
            SimpleNamespace(palace_path=lease_env, hub_grace_secs=7.0),
            raising=False,
        )
        assert mcp_server._hub_grace_window_secs() == 7.0

    def test_hub_restart_error_is_distinguishable_from_a_peer_writer_refusal(
        self, monkeypatch, hub_state
    ):
        """Both conditions ride -32001; ``condition`` tells them apart."""
        monkeypatch.setattr(mcp_server, "_hub_unavailable_since", time.monotonic())
        hub = mcp_server._hub_restarting_error(_call())
        peer = mcp_server._peer_writer_refusal_payload(1, MUTATING_TOOL, "peer holds it")
        assert hub["error"]["code"] == peer["error"]["code"] == -32001
        assert hub["error"]["data"]["condition"] == "hub_restarting"
        assert peer["error"]["data"]["condition"] == "peer_writer_active"

    def test_mutating_notification_is_written_rather_than_dropped(self, monkeypatch, hub_state):
        """A notification has nowhere to receive a retryable error.

        Holding one means the write vanishes with no diagnostic. One stranded
        lease is recoverable; a lost write is not.
        """
        monkeypatch.setattr(mcp_server, "_hub_last_seen_monotonic", time.monotonic())
        monkeypatch.setattr(mcp_server, "_hub_unavailable_since", time.monotonic())
        handled = []
        monkeypatch.setattr(
            mcp_server, "handle_request", lambda request: handled.append(request) or None
        )

        notification = _call()
        notification.pop("id")
        assert mcp_server._dispatch_stdio_request(notification) is None
        assert len(handled) == 1

    def test_zero_window_disables_the_hold(self, monkeypatch, hub_state):
        monkeypatch.setenv(mcp_server._HUB_GRACE_SECS_ENV, "0")
        monkeypatch.setattr(mcp_server, "_hub_last_seen_monotonic", time.monotonic())
        monkeypatch.setattr(mcp_server, "_hub_unavailable_since", time.monotonic())
        handled = []
        monkeypatch.setattr(
            mcp_server, "handle_request", lambda request: handled.append(request) or {"ok": True}
        )
        assert mcp_server._dispatch_stdio_request(_call()) == {"ok": True}
        assert len(handled) == 1


class TestLogstreamToolsSurviveTheGraceWindow:
    """The six logstream tools must not be stalled by a hub restart.

    They write only logstream.sqlite3 — an independent WAL database — take no
    palace lease, and were exempted from peer-writer refusal precisely so
    agent coordination survives a peer holding the palace. Holding them for
    the grace window would fail every coordination write for up to a minute on
    each hub restart, which is the escalation path this lane exists to close.
    """

    @pytest.mark.parametrize("tool", sorted(mcp_server._PEER_WRITER_EXEMPT_TOOLS))
    def test_exempt_tool_writes_through_a_hub_outage(self, monkeypatch, hub_state, tool):
        monkeypatch.setattr(mcp_server, "_hub_last_seen_monotonic", time.monotonic())
        monkeypatch.setattr(mcp_server, "_hub_unavailable_since", time.monotonic())
        handled = []
        monkeypatch.setattr(
            mcp_server, "handle_request", lambda request: handled.append(request) or {"ok": True}
        )

        assert mcp_server._hub_grace_active() is True
        assert mcp_server._dispatch_stdio_request(_call(tool_name=tool)) == {"ok": True}
        assert len(handled) == 1, f"{tool} was held by the grace window"

    def test_the_exempt_set_is_exactly_the_logstream_tools(self):
        """A palace-mutating tool must never join the pass-through by accident."""
        assert mcp_server._PEER_WRITER_EXEMPT_TOOLS == frozenset(
            {
                "mempalace_event_append",
                "mempalace_task_create",
                "mempalace_event_ack",
                "mempalace_event_cursor_set",
                "mempalace_artifact_put",
                "mempalace_patch_submit",
            }
        )
        assert mcp_server._PEER_WRITER_EXEMPT_TOOLS <= mcp_server._MUTATING_TOOLS

    def test_codex_capture_is_still_held(self, monkeypatch, hub_state):
        """It mutates the palace, so a direct write would strand the flock."""
        monkeypatch.setattr(mcp_server, "_hub_last_seen_monotonic", time.monotonic())
        monkeypatch.setattr(mcp_server, "_hub_unavailable_since", time.monotonic())
        monkeypatch.setattr(
            mcp_server,
            "handle_request",
            lambda request: pytest.fail("codex_capture fell back to a direct write"),
        )

        response = mcp_server._dispatch_stdio_request(_call(tool_name="mempalace_codex_capture"))
        assert response["error"]["code"] == -32001
        assert response["error"]["data"]["condition"] == "hub_restarting"

    def test_exempt_tools_are_still_never_replayed_after_a_mid_flight_failure(
        self, monkeypatch, lease_env
    ):
        """Pass-through applies to the *hub-missing* case only.

        A forward that died in flight may already have been executed by the
        hub, so replaying a logstream append locally would duplicate an
        append-only event. That guard keys off _request_is_mutating, which
        still covers all six.
        """
        import urllib.error

        monkeypatch.setattr(mcp_server, "_hub_last_seen_monotonic", time.monotonic())
        monkeypatch.setattr(mcp_server, "_hub_unavailable_since", None)
        monkeypatch.setattr(mcp_server, "_hub_proxy_target", lambda: ("http://h", {}))
        monkeypatch.setattr(
            mcp_server,
            "handle_request",
            lambda request: pytest.fail("a mid-flight logstream write was replayed locally"),
        )

        def boom(*args, **kwargs):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(mcp_server, "_forward_request_to_hub", boom)

        response = mcp_server._dispatch_stdio_request(_call(tool_name="mempalace_event_append"))
        assert response["error"]["code"] == -32000


class TestHubSightingBookkeeping:
    def test_live_record_bootstraps_but_does_not_clear_an_outage(self, monkeypatch, lease_env):
        monkeypatch.setattr(mcp_server, "_hub_last_seen_monotonic", None)
        monkeypatch.setattr(mcp_server, "_hub_unavailable_since", None)

        mcp_server._note_hub_record()
        assert mcp_server._hub_last_seen_monotonic is not None

        mcp_server._note_hub_missing()
        started = mcp_server._hub_unavailable_since
        assert started is not None

        # A hub whose pid is alive but whose port is dead must not reset its
        # own outage clock on every request, or the window never expires.
        mcp_server._note_hub_record()
        assert mcp_server._hub_unavailable_since == started

    def test_successful_forward_clears_the_outage(self, monkeypatch, lease_env):
        monkeypatch.setattr(mcp_server, "_hub_last_seen_monotonic", time.monotonic() - 10)
        monkeypatch.setattr(mcp_server, "_hub_unavailable_since", time.monotonic() - 10)
        mcp_server._note_hub_seen()
        assert mcp_server._hub_unavailable_since is None
        assert mcp_server._hub_grace_active() is False

    def test_forward_records_the_sighting(self, monkeypatch, lease_env):
        monkeypatch.setattr(mcp_server, "_hub_last_seen_monotonic", None)
        monkeypatch.setattr(mcp_server, "_hub_unavailable_since", None)
        monkeypatch.setattr(mcp_server, "_hub_proxy_target", lambda: ("http://h", {}))
        monkeypatch.setattr(
            mcp_server,
            "_forward_request_to_hub",
            lambda base_url, headers, request, palace: {"jsonrpc": "2.0", "id": 1, "result": {}},
        )

        mcp_server._dispatch_stdio_request(_call())

        assert mcp_server._hub_last_seen_monotonic is not None
        assert mcp_server._hub_unavailable_since is None

    def test_unreachable_hub_with_a_live_record_still_refuses_and_is_retryable(
        self, monkeypatch, lease_env
    ):
        """serverinfo present but the port is dead: -32000, never a direct write.

        This path never had a fallback to remove, so it keeps its existing
        error code; what it gains is an explicit ``retryable`` flag and an
        outage clock, so the caller and ``_hub_grace_active`` agree.
        """
        import urllib.error

        monkeypatch.setattr(mcp_server, "_hub_last_seen_monotonic", time.monotonic())
        monkeypatch.setattr(mcp_server, "_hub_unavailable_since", None)
        monkeypatch.setattr(mcp_server, "_hub_proxy_target", lambda: ("http://h", {}))

        def boom(*args, **kwargs):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(mcp_server, "_forward_request_to_hub", boom)
        monkeypatch.setattr(
            mcp_server,
            "handle_request",
            lambda request: pytest.fail("mutating call fell back to a direct write"),
        )

        response = mcp_server._dispatch_stdio_request(_call())
        assert response["error"]["code"] == -32000
        assert response["error"]["data"]["retryable"] is True
        assert mcp_server._hub_unavailable_since is not None


def test_forwarding_disabled_never_stalls_writes(monkeypatch, lease_env):
    """MEMPALACE_HUB_FORWARD=0 keeps its meaning: direct, no grace window."""
    monkeypatch.setenv("MEMPALACE_HUB_FORWARD", "0")
    monkeypatch.setattr(mcp_server, "_hub_last_seen_monotonic", time.monotonic())
    monkeypatch.setattr(mcp_server, "_hub_unavailable_since", time.monotonic())
    handled = []
    monkeypatch.setattr(
        mcp_server, "handle_request", lambda request: handled.append(request) or {"ok": True}
    )

    assert mcp_server._dispatch_stdio_request(_call()) == {"ok": True}
    assert len(handled) == 1


def test_statusz_writer_payload_names_the_lease_scope(monkeypatch, lease_env):
    monkeypatch.setattr(mcp_server, "_MCP_WRITER_LEASE_SCOPE", mcp_server._LEASE_SCOPE_PER_CALL)
    assert mcp_server._writer_lease_scope() == "per_call"
    monkeypatch.setattr(mcp_server, "_MCP_WRITER_LEASE_SCOPE", mcp_server._LEASE_SCOPE_LIFETIME)
    assert mcp_server._writer_lease_scope() == "lifetime"


def test_concurrent_calls_never_leave_a_handler_writing_without_the_lease(monkeypatch, lease_env):
    """Two real ``handle_request`` calls; no peer may take the palace mid-write.

    The regression pinned here: the flock is acquired in *preflight*, but the
    depth counter used to be incremented later, inside the scoped lease. Two
    concurrent mutating calls could interleave as "T2's preflight sees the
    lease already held by T1 -> T1's scope exits and releases it -> T2's
    handler runs with nothing held", and an outside process could then take
    the palace while T2 was mid-write.

    The predecessor of this test pre-acquired the lease and drove
    ``_mcp_scoped_writer_lease`` directly, which exercises the context manager
    in isolation and structurally cannot see that interleaving. This one goes
    through the real entry point and asserts the property that matters: while
    ANY handler body is executing, a peer's ``mine_palace_lock`` must fail.
    The gap *between* the two calls is not asserted — giving the flock back
    between calls is exactly what per-call scoping is for.
    """
    from mempalace import palace as palace_mod

    # Contend like a peer: without this, a second open in this process takes
    # the re-entrancy passthrough instead of conflicting on the flock.
    monkeypatch.setattr(palace_mod, "_held_by_this_process", lambda key: False)

    in_flight_guard = threading.Lock()
    state = {"in_flight": 0}
    thefts = []
    lease_seen = {}
    responses = {}
    t2_preflight_done = threading.Event()
    t1_finished = threading.Event()
    stop_stealing = threading.Event()

    def stealer():
        """Try to take the palace, but only while a handler is writing."""
        while not stop_stealing.is_set():
            with in_flight_guard:
                if state["in_flight"] > 0:
                    try:
                        cm = palace_mod.mine_palace_lock(lease_env)
                        cm.__enter__()
                        cm.__exit__(None, None, None)
                        thefts.append(time.monotonic())
                    except palace_mod.MineAlreadyRunning:
                        pass
            time.sleep(0.002)

    def handler():
        name = threading.current_thread().name
        with in_flight_guard:
            state["in_flight"] += 1
        try:
            lease_seen[name] = mcp_server._MCP_WRITER_LOCK_CM is not None
            if name == "T1":
                t2_preflight_done.wait(timeout=5)
            # Stay inside the write long enough for the stealer to get real
            # attempts in while this handler is on the clock.
            time.sleep(0.05)
        finally:
            with in_flight_guard:
                state["in_flight"] -= 1
        return {"ok": True}

    _install_fake_tool(monkeypatch, handler)

    real_preflight = mcp_server._mcp_tool_preflight_refusal

    def sequenced_preflight(req_id, tool_name):
        out = real_preflight(req_id, tool_name)
        if threading.current_thread().name == "T2":
            # T2 has passed the acquire; let T1 run and fully release before
            # T2 enters its scope. That is the interleaving under test.
            t2_preflight_done.set()
            t1_finished.wait(timeout=5)
        return out

    monkeypatch.setattr(mcp_server, "_mcp_tool_preflight_refusal", sequenced_preflight)

    thief = threading.Thread(target=stealer, name="peer", daemon=True)
    thief.start()

    def run_t1():
        try:
            responses["T1"] = mcp_server.handle_request(_call(req_id=1))
        finally:
            t1_finished.set()

    t2 = threading.Thread(
        target=lambda: responses.__setitem__("T2", mcp_server.handle_request(_call(req_id=2))),
        name="T2",
    )
    t1 = threading.Thread(target=run_t1, name="T1")
    t2.start()
    time.sleep(0.05)
    t1.start()
    t1.join(timeout=15)
    t2.join(timeout=15)
    stop_stealing.set()
    thief.join(timeout=5)

    assert not t1.is_alive() and not t2.is_alive()
    assert thefts == [], "a peer took the palace flock while a handler was writing"
    assert lease_seen == {"T1": True, "T2": True}, lease_seen
    for name in ("T1", "T2"):
        assert "result" in responses[name], responses[name]
    assert mcp_server._MCP_WRITER_LEASE_DEPTH == 0
    assert mcp_server._MCP_WRITER_LOCK_CM is None


def test_lease_lost_between_preflight_and_handler_refuses_instead_of_writing(
    monkeypatch, lease_env
):
    """The re-assert's own failure path: refuse, never write unprotected."""
    monkeypatch.setattr(mcp_server, "_mcp_read_only_refusal", lambda req_id, tool: None)
    monkeypatch.setattr(mcp_server, "_mcp_sqlite_integrity_refusal", lambda req_id, tool: None)
    monkeypatch.setattr(mcp_server, "_mcp_stale_library_refusal", lambda req_id, tool: None)
    monkeypatch.setattr(mcp_server, "_mcp_diverged_index_refusal", lambda req_id, tool: None)

    calls = []

    def acquire():
        calls.append(1)
        # Preflight wins; the scope's re-assert finds a peer holding it.
        if len(calls) == 1:
            return True, ""
        return False, "another mempalace writer already holds the palace lock"

    monkeypatch.setattr(mcp_server, "_acquire_mcp_writer_lock", acquire)
    _install_fake_tool(monkeypatch, lambda: pytest.fail("handler ran without the lease"))
    # _install_fake_tool re-stubs the other gates; keep our acquire double.
    monkeypatch.setattr(mcp_server, "_acquire_mcp_writer_lock", acquire)

    response = mcp_server.handle_request(_call())

    assert response["error"]["code"] == -32001
    assert response["error"]["data"]["condition"] == "peer_writer_active"
    assert response["error"]["data"]["retryable"] is True
    assert mcp_server._MCP_WRITER_LEASE_DEPTH == 0


def test_release_really_closes_storage_handles(monkeypatch, lease_env):
    """The un-stubbed release path: per-call scoping must drop writable handles.

    Every other test in this file stubs ``_discard_mcp_storage_handles`` for
    speed, so nothing exercised the behaviour the per-call lease depends on —
    a flock handed back while a Chroma PersistentClient stayed open would
    re-enter the stale-HNSW hazard the lease exists to prevent.
    """
    # Put the real implementation back over the fixture's stub.
    monkeypatch.setattr(mcp_server, "_discard_mcp_storage_handles", _REAL_DISCARD_HANDLES)
    monkeypatch.setattr(mcp_server, "_client_cache", object())
    monkeypatch.setattr(mcp_server, "_collection_cache", object())

    assert mcp_server._acquire_mcp_writer_lock() == (True, "")
    # Acquisition itself drops any query-only handle taken before promotion.
    assert mcp_server._client_cache is None
    mcp_server._client_cache = object()
    mcp_server._collection_cache = object()

    with mcp_server._mcp_scoped_writer_lease(MUTATING_TOOL):
        pass

    assert mcp_server._MCP_WRITER_LOCK_CM is None
    assert mcp_server._client_cache is None
    assert mcp_server._collection_cache is None


def test_failed_release_keeps_the_handle_rather_than_believing_it_is_free(monkeypatch, lease_env):
    """If ``__exit__`` raises, the OS may still hold the flock — say so."""

    class Boom:
        def __exit__(self, *exc):
            raise RuntimeError("close failed")

    boom = Boom()
    monkeypatch.setattr(mcp_server, "_MCP_WRITER_LOCK_CM", boom)

    with pytest.raises(RuntimeError):
        mcp_server._release_mcp_writer_lock()

    assert mcp_server._MCP_WRITER_LOCK_CM is boom
    # The scoped lease swallows it so a tool result is never replaced by a
    # release failure, and the handle survives for a later retry.
    monkeypatch.setattr(mcp_server, "_MCP_WRITER_LEASE_DEPTH", 0)
    monkeypatch.setattr(mcp_server, "_acquire_mcp_writer_lock", lambda: (True, ""))
    with mcp_server._mcp_scoped_writer_lease(MUTATING_TOOL):
        pass
    assert mcp_server._MCP_WRITER_LOCK_CM is boom
    mcp_server._MCP_WRITER_LOCK_CM = None


def test_json_serializable_hub_error(monkeypatch, lease_env):
    monkeypatch.setattr(mcp_server, "_hub_unavailable_since", time.monotonic())
    payload = mcp_server._hub_restarting_error(_call())
    json.dumps(payload)
