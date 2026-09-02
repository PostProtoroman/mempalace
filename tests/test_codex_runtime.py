"""Runtime blockers for Codex capture: hub-routed writes, service, status.

Covers the three things that stopped capture working on a real machine:

1. A hub owns the palace writer lease for its whole lifetime, so a follower
   that grabbed ``mine_palace_lock`` itself was refused exactly where a hub
   runs. Capture is handed to the hub instead — and never to both.
2. A foreground follower dies with the terminal. There is now an installable
   login service.
3. A hub restart briefly frees the flock. A follower that has watched a hub
   own this palace before now holds (writes nothing, takes no lock) for the
   grace window instead of racing the restarting hub for its own lease — see
   ``TestHubRestartGraceWindow`` below and docs/write-routing-policy.md.
"""

from __future__ import annotations

import json
import plistlib
import time

import pytest

from mempalace import codex_follower, codex_hub, codex_service
from mempalace.codex_follower import capture_pass, scan_once, status

from tests.test_codex_capture import _item, _meta, _write_rollout


# ── writer selection: hub vs direct, never both ──────────────────────


def test_hub_is_used_when_one_owns_the_palace(tmp_path, monkeypatch):
    calls = {"hub": 0, "lock": 0}

    def _fake_capture(palace_path, sessions_root, **kwargs):
        calls["hub"] += 1
        return {"writer": "hub", "drawers_added": 2, "drawers_already_present": 0}

    def _explode(*a, **k):  # the lock must never be taken when a hub answered
        calls["lock"] += 1
        raise AssertionError("follower took the palace lock while a hub was live")

    monkeypatch.setattr("mempalace.codex_hub.capture_via_hub", _fake_capture)
    monkeypatch.setattr("mempalace.palace.mine_palace_lock", _explode)

    stats = capture_pass(str(tmp_path / "palace"), str(tmp_path / "sessions"))
    assert stats["writer"] == "hub"
    assert calls == {"hub": 1, "lock": 0}


def test_direct_write_only_when_no_hub_owns_the_palace(tmp_path):
    palace = str(tmp_path / "palace")
    root = tmp_path / "sessions"
    _write_rollout(root, "d1", [_meta(), _item("user", "u1", "no hub here", 1)])

    stats = capture_pass(palace, str(root))
    assert stats["writer"] == "direct"
    assert stats["drawers_added"] == 1


def test_hub_failure_is_never_retried_locally(tmp_path, monkeypatch):
    """A hub that answered may have written. Retrying locally = two writers."""

    def _refuse(palace_path, sessions_root, **kwargs):
        raise codex_hub.HubRefused("hub returned HTTP 500 Internal Server Error")

    def _explode(*a, **k):
        raise AssertionError("fell back to a local write after the hub answered")

    monkeypatch.setattr("mempalace.codex_hub.capture_via_hub", _refuse)
    monkeypatch.setattr("mempalace.palace.mine_palace_lock", _explode)

    stats = capture_pass(str(tmp_path / "palace"), str(tmp_path / "sessions"))
    assert stats["writer"] == "hub"
    assert "500" in stats["error"]
    assert stats["drawers_added"] == 0


def test_lease_held_with_no_hub_is_reported_blocked_not_written(tmp_path, monkeypatch):
    from mempalace.palace import MineAlreadyRunning

    monkeypatch.setattr(
        "mempalace.codex_hub.capture_via_hub",
        lambda p, s, **kw: (_ for _ in ()).throw(codex_hub.HubUnavailable("none")),
    )

    def _held(*a, **k):
        raise MineAlreadyRunning("palace is held by PID 999")

    monkeypatch.setattr("mempalace.palace.mine_palace_lock", _held)

    stats = capture_pass(str(tmp_path / "palace"), str(tmp_path / "sessions"))
    assert stats["writer"] == "blocked"
    assert "999" in stats["error"]


def test_no_turn_is_dropped_when_a_pass_is_blocked(tmp_path, monkeypatch):
    """A blocked pass writes nothing, and the next unblocked pass gets it all."""
    from mempalace.palace import MineAlreadyRunning

    palace = str(tmp_path / "palace")
    root = tmp_path / "sessions"
    _write_rollout(root, "b1", [_meta(), _item("user", "u1", "must not be lost", 1)])

    monkeypatch.setattr(
        "mempalace.codex_hub.capture_via_hub",
        lambda p, s, **kw: (_ for _ in ()).throw(codex_hub.HubUnavailable("none")),
    )
    real_lock = __import__("mempalace.palace", fromlist=["mine_palace_lock"]).mine_palace_lock
    monkeypatch.setattr(
        "mempalace.palace.mine_palace_lock",
        lambda *a, **k: (_ for _ in ()).throw(MineAlreadyRunning("busy")),
    )
    assert capture_pass(palace, str(root))["writer"] == "blocked"

    monkeypatch.setattr("mempalace.palace.mine_palace_lock", real_lock)
    recovered = capture_pass(palace, str(root))
    assert recovered["writer"] == "direct"
    assert recovered["drawers_added"] == 1


# ── bounded hub calls ────────────────────────────────────────────────


def test_max_turns_bounds_a_pass_and_reports_pending(tmp_path):
    palace = str(tmp_path / "palace")
    root = tmp_path / "sessions"
    lines = [_meta()] + [_item("user", f"u{i}", f"turn {i}", i) for i in range(1, 9)]
    _write_rollout(root, "big", lines)

    first = scan_once(palace, str(root), max_turns=3)
    assert first["pending"] is True
    assert first["drawers_added"] == 3

    # Repeating drains the rest without duplicating the first three.
    total = first["drawers_added"]
    for _ in range(10):
        nxt = scan_once(palace, str(root), max_turns=3)
        total += nxt["drawers_added"]
        if not nxt["pending"]:
            break
    assert total == 8


def test_unbounded_scan_reports_not_pending(tmp_path):
    palace = str(tmp_path / "palace")
    root = tmp_path / "sessions"
    _write_rollout(root, "small", [_meta(), _item("user", "u1", "one", 1)])
    assert scan_once(palace, str(root))["pending"] is False


# ── status states ────────────────────────────────────────────────────


def _seed(tmp_path):
    palace = str(tmp_path / "palace")
    root = tmp_path / "sessions"
    _write_rollout(root, "s", [_meta(), _item("user", "u1", "hello", 1)])
    return palace, root


def test_status_is_active_when_running_and_clean(tmp_path, monkeypatch):
    import os

    palace, root = _seed(tmp_path)
    scan_once(palace, str(root))
    from mempalace.codex_follower import _write_heartbeat

    _write_heartbeat(
        palace, sessions_root=str(root), running=True, last_scan={"writer": "direct", "error": ""}
    )
    monkeypatch.setattr("mempalace.codex_follower._pid_alive", lambda pid: True)
    assert status(palace, str(root))["state"] == "ACTIVE"
    assert os.getpid()  # sanity


def test_status_is_degraded_when_follower_stopped(tmp_path, monkeypatch):
    palace, root = _seed(tmp_path)
    scan_once(palace, str(root))
    from mempalace.codex_follower import _write_heartbeat

    _write_heartbeat(
        palace, sessions_root=str(root), running=False, last_scan={"writer": "direct", "error": ""}
    )
    monkeypatch.setattr("mempalace.codex_follower._pid_alive", lambda pid: False)
    assert status(palace, str(root))["state"] == "DEGRADED"


def test_status_is_blocked_when_last_pass_could_not_write(tmp_path, monkeypatch):
    palace, root = _seed(tmp_path)
    from mempalace.codex_follower import _write_heartbeat

    _write_heartbeat(
        palace,
        sessions_root=str(root),
        running=True,
        last_scan={"writer": "blocked", "error": "palace writer lease held by another process"},
    )
    monkeypatch.setattr("mempalace.codex_follower._pid_alive", lambda pid: True)
    st = status(palace, str(root))
    assert st["state"] == "BLOCKED"
    assert "lease held" in st["last_error"]


def test_status_reports_hub_connectivity_and_lag(tmp_path, monkeypatch):
    palace, root = _seed(tmp_path)
    scan_once(palace, str(root))
    monkeypatch.setattr("mempalace.codex_hub.hub_target", lambda p: ("http://127.0.0.1:9999", {}))
    monkeypatch.setattr("mempalace.codex_hub.hub_is_healthy", lambda u, h: True)
    from mempalace.codex_follower import _write_heartbeat

    _write_heartbeat(
        palace, sessions_root=str(root), running=True, last_scan={"writer": "hub", "error": ""}
    )
    st = status(palace, str(root))
    assert st["hub_url"] == "http://127.0.0.1:9999"
    assert st["hub_reachable"] is True
    assert st["capture_lag_seconds"] is not None and st["capture_lag_seconds"] >= 0


def test_capture_lag_is_none_for_unparseable_timestamps(tmp_path):
    from mempalace.codex_follower import _lag_seconds

    assert _lag_seconds("", "2026-08-31T00:00:00+00:00") is None
    assert _lag_seconds("not-a-date", "2026-08-31T00:00:00+00:00") is None


# ── service lifecycle (isolated home only) ───────────────────────────


def test_plist_runs_the_follower_at_login_without_touching_agents_md(tmp_path):
    definition = codex_service.build_plist(
        str(tmp_path / "palace"), str(tmp_path / "sessions"), home=str(tmp_path / "home")
    )
    assert definition["Label"] == codex_service.LABEL
    assert definition["RunAtLoad"] is True
    assert definition["KeepAlive"] == {"SuccessfulExit": False}
    # The service must never rewrite the user's instructions file on login.
    assert "--no-recall" in definition["ProgramArguments"]
    assert "codex-capture" in definition["ProgramArguments"]


@pytest.mark.skipif(not codex_service.is_supported(), reason="launchd only")
def test_install_writes_plist_into_the_given_home_only(tmp_path):
    home = tmp_path / "home"
    palace = str(tmp_path / "palace")
    result = codex_service.install(palace, str(tmp_path / "sessions"), home=str(home), load=False)

    path = home / "Library" / "LaunchAgents" / f"{codex_service.LABEL}.plist"
    assert path.exists()
    assert result["plist"] == str(path)
    assert codex_service.is_installed(str(home))
    with open(path, "rb") as fh:
        assert plistlib.load(fh)["Label"] == codex_service.LABEL


@pytest.mark.skipif(not codex_service.is_supported(), reason="launchd only")
def test_uninstall_removes_only_managed_files(tmp_path):
    home = tmp_path / "home"
    palace_dir = tmp_path / "palace"
    root = tmp_path / "sessions"
    _write_rollout(root, "keep", [_meta(), _item("user", "u1", "keep this turn", 1)])
    scan_once(str(palace_dir), str(root))

    agents = home / "AGENTS.md"
    agents.parent.mkdir(parents=True, exist_ok=True)
    agents.write_text("# mine\n\nkeep my rules\n", encoding="utf-8")

    codex_service.install(str(palace_dir), str(root), home=str(home), load=False)
    result = codex_service.uninstall(str(palace_dir), home=str(home))

    assert not codex_service.is_installed(str(home))
    assert any("plist" in p for p in result["removed"])
    # User content and palace data survive.
    assert agents.read_text(encoding="utf-8") == "# mine\n\nkeep my rules\n"
    from mempalace.palace import get_collection

    stored = get_collection(str(palace_dir), create=False).get(
        where={"ingest_mode": "codex_session"}, include=["documents"]
    )
    assert any("keep this turn" in d for d in stored.get("documents") or [])


def test_uninstall_is_safe_when_nothing_is_installed(tmp_path):
    result = codex_service.uninstall(str(tmp_path / "palace"), home=str(tmp_path / "home"))
    assert result["removed"] == []


# ── failure escalation: a silent outage is impossible ────────────────


def _fail_stats(error="hub returned HTTP 500 Internal Server Error"):
    return {
        "writer": "hub",
        "error": error,
        "drawers_added": 0,
        "drawers_already_present": 0,
    }


def _ok_stats():
    return {"writer": "direct", "error": "", "drawers_added": 0, "drawers_already_present": 0}


def _outage_events(palace):
    import os

    from mempalace.logstream import LOGSTREAM_DB_FILENAME, Logstream

    ls = Logstream(db_path=os.path.join(palace, LOGSTREAM_DB_FILENAME))
    return ls.list_events(type="status", from_agent="codex-follower")


def _run_follow(palace, monkeypatch, results):
    """Run follow() against a scripted sequence of capture_pass results."""
    from mempalace.codex_follower import follow

    queue = list(results)
    monkeypatch.setattr("mempalace.codex_follower.capture_pass", lambda *a, **k: queue.pop(0))
    return follow(palace, "unused-root", interval=0, max_passes=len(results))


def test_consecutive_failures_escalate_to_error_and_one_event(tmp_path, monkeypatch, caplog):
    import logging

    palace = str(tmp_path / "palace")
    with caplog.at_level(logging.ERROR, logger="mempalace.codex_follower"):
        _run_follow(palace, monkeypatch, [_fail_stats()] * 6)

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors, "threshold crossing must log at ERROR, not lower"
    assert any("consecutive failed passes" in r.getMessage() for r in errors)


# ── hub restart grace window ─────────────────────────────────────────
#
# A hub (`mempalace serve`) holds the palace writer flock for its whole
# lifetime. During a restart the flock is briefly free; before this lane
# existed, the very next pass saw HubUnavailable and wrote directly, taking
# the flock the returning hub needed (live 2026-09-02: the hub sat in
# mode=degraded for 12s+ while the follower held the lease as a direct
# writer). See docs/write-routing-policy.md and mempalace/mcp_server.py's
# ``_hub_grace_active`` for the sibling lane on the MCP stdio side.


def _throw(exc):
    def _raiser(*_a, **_k):
        raise exc

    return _raiser


class TestHubRestartGraceWindow:
    def test_never_seen_hub_writes_directly_with_no_hold(self, tmp_path, monkeypatch):
        """A follower that has never captured through a hub for this palace
        keeps today's behaviour: no window, no hold, direct at once."""
        palace = str(tmp_path / "palace")
        root = tmp_path / "sessions"
        _write_rollout(root, "n1", [_meta(), _item("user", "u1", "cold start", 1)])

        monkeypatch.setattr(
            "mempalace.codex_hub.capture_via_hub",
            _throw(codex_hub.HubUnavailable("no hub")),
        )

        stats = capture_pass(palace, str(root))
        assert stats["writer"] == "direct"
        assert stats["drawers_added"] == 1
        assert not stats.get("held_for_hub_restart")

    def test_hold_inside_the_window_takes_no_lock_and_writes_nothing(self, tmp_path, monkeypatch):
        palace = str(tmp_path / "palace")
        root = tmp_path / "sessions"
        _write_rollout(root, "w1", [_meta(), _item("user", "u1", "must not be lost", 1)])
        key = codex_follower._palace_state_key(palace)
        now = time.monotonic()
        monkeypatch.setitem(codex_follower._hub_last_seen_monotonic, key, now)
        monkeypatch.setitem(codex_follower._hub_unavailable_since, key, now)

        monkeypatch.setattr(
            "mempalace.codex_hub.capture_via_hub",
            _throw(codex_hub.HubUnavailable("restarting")),
        )
        monkeypatch.setattr(
            "mempalace.palace.mine_palace_lock",
            _throw(AssertionError("held pass took the palace writer lease")),
        )

        stats = capture_pass(palace, str(root))
        assert stats["writer"] == "hold"
        assert stats["error"] == ""
        assert stats["held_for_hub_restart"] is True
        assert stats["drawers_added"] == 0
        # Nothing was written and the cursor never advanced, so the turn is
        # still there for the next pass to pick up.
        #
        # Escalation accounting for held passes is exercised end-to-end via
        # follow() below: test_held_pass_does_not_count_toward_escalation and
        # test_hold_does_not_reset_an_in_progress_failure_streak.

    def test_direct_resumes_once_the_window_elapses(self, tmp_path, monkeypatch):
        palace = str(tmp_path / "palace")
        root = tmp_path / "sessions"
        _write_rollout(root, "e1", [_meta(), _item("user", "u1", "after the window", 1)])
        key = codex_follower._palace_state_key(palace)
        long_ago = time.monotonic() - 120.0
        monkeypatch.setitem(codex_follower._hub_last_seen_monotonic, key, long_ago)
        monkeypatch.setitem(codex_follower._hub_unavailable_since, key, long_ago)

        monkeypatch.setattr(
            "mempalace.codex_hub.capture_via_hub",
            _throw(codex_hub.HubUnavailable("still gone")),
        )

        stats = capture_pass(palace, str(root))
        assert stats["writer"] == "direct"
        assert stats["drawers_added"] == 1

    def test_direct_failure_after_the_window_still_escalates(self, tmp_path, monkeypatch):
        """Item 4: a hold that outlives the window and then fails direct counts."""
        from mempalace.palace import MineAlreadyRunning

        palace = str(tmp_path / "palace")
        key = codex_follower._palace_state_key(palace)
        long_ago = time.monotonic() - 200.0
        monkeypatch.setitem(codex_follower._hub_last_seen_monotonic, key, long_ago)
        monkeypatch.setitem(codex_follower._hub_unavailable_since, key, long_ago)

        monkeypatch.setattr(
            "mempalace.codex_hub.capture_via_hub",
            _throw(codex_hub.HubUnavailable("still gone")),
        )
        monkeypatch.setattr(
            "mempalace.palace.mine_palace_lock",
            _throw(MineAlreadyRunning("palace is held by PID 4242")),
        )

        stats = capture_pass(palace, "unused-root")
        assert stats["writer"] == "blocked"
        assert stats["error"]

    def test_env_and_config_knobs_match_mcp_server(self, monkeypatch):
        monkeypatch.delenv("MEMPALACE_HUB_GRACE_SECS", raising=False)
        assert codex_follower._hub_grace_window_secs() == codex_follower._HUB_GRACE_SECS_DEFAULT
        monkeypatch.setenv("MEMPALACE_HUB_GRACE_SECS", "12.5")
        assert codex_follower._hub_grace_window_secs() == 12.5
        monkeypatch.setenv("MEMPALACE_HUB_GRACE_SECS", "999999")
        assert codex_follower._hub_grace_window_secs() == codex_follower._HUB_GRACE_SECS_MAX
        monkeypatch.setenv("MEMPALACE_HUB_GRACE_SECS", "not-a-number")
        assert codex_follower._hub_grace_window_secs() == codex_follower._HUB_GRACE_SECS_DEFAULT

    def test_zero_window_disables_the_hold(self, tmp_path, monkeypatch):
        palace = str(tmp_path / "palace")
        root = tmp_path / "sessions"
        _write_rollout(root, "z1", [_meta(), _item("user", "u1", "no window", 1)])
        key = codex_follower._palace_state_key(palace)
        now = time.monotonic()
        monkeypatch.setitem(codex_follower._hub_last_seen_monotonic, key, now)
        monkeypatch.setitem(codex_follower._hub_unavailable_since, key, now)
        monkeypatch.setenv("MEMPALACE_HUB_GRACE_SECS", "0")

        monkeypatch.setattr(
            "mempalace.codex_hub.capture_via_hub",
            _throw(codex_hub.HubUnavailable("gone")),
        )

        stats = capture_pass(palace, str(root))
        assert stats["writer"] == "direct"

    def test_status_reports_holding_for_hub_restart(self, tmp_path, monkeypatch):
        palace, root = _seed(tmp_path)
        scan_once(palace, str(root))
        from mempalace.codex_follower import _write_heartbeat

        _write_heartbeat(
            palace,
            sessions_root=str(root),
            running=True,
            last_scan={
                "writer": "hold",
                "error": "",
                "held_for_hub_restart": True,
                "grace_window_secs": 60.0,
                "waited_secs": 3.2,
            },
        )
        monkeypatch.setattr("mempalace.codex_follower._pid_alive", lambda pid: True)
        st = status(palace, str(root))
        assert st["state"] == "DEGRADED"
        assert st["holding_for_hub_restart"] is True
        assert st["detail"] == "holding for hub restart"
        assert st["unhealthy"] is False


class TestHubRestartRecovery:
    def test_hub_gone_hold_direct_then_hub_back(self, tmp_path, monkeypatch):
        """The full lifecycle: hub -> grace hold -> direct -> hub again.

        Recovery needs no special-case code: capture_pass always re-probes
        the hub first, so as soon as it answers again the very next pass goes
        back to writer=hub on its own.
        """
        import mempalace.palace as palace_mod

        palace = str(tmp_path / "palace")
        root = tmp_path / "sessions"
        _write_rollout(root, "r1", [_meta(), _item("user", "u1", "first turn", 1)])
        _write_rollout(root, "r2", [_meta(), _item("user", "u2", "written after recovery", 2)])
        real_lock = palace_mod.mine_palace_lock
        key = codex_follower._palace_state_key(palace)

        # Pass 1: hub answers and writes -- this process has now "seen" a hub.
        monkeypatch.setattr(
            "mempalace.codex_hub.capture_via_hub",
            lambda p, s, **kw: {
                "writer": "hub",
                "drawers_added": 1,
                "drawers_already_present": 0,
            },
        )
        stats1 = capture_pass(palace, str(root))
        assert stats1["writer"] == "hub"
        assert key in codex_follower._hub_last_seen_monotonic

        # Pass 2: hub vanishes -- inside the window, so hold; never touch the lock.
        monkeypatch.setattr(
            "mempalace.codex_hub.capture_via_hub",
            _throw(codex_hub.HubUnavailable("restarting")),
        )
        monkeypatch.setattr(
            "mempalace.palace.mine_palace_lock",
            _throw(AssertionError("held pass took the palace writer lease")),
        )
        stats2 = capture_pass(palace, str(root))
        assert stats2["writer"] == "hold"

        # Fast-forward past the window: now genuinely gone -- direct engages.
        monkeypatch.setitem(codex_follower._hub_unavailable_since, key, time.monotonic() - 120.0)
        monkeypatch.setattr("mempalace.palace.mine_palace_lock", real_lock)
        stats3 = capture_pass(palace, str(root))
        assert stats3["writer"] == "direct"
        # Pass 1 went through the (faked) hub and never actually wrote to the
        # palace, so both rollouts' turns are still unwritten here.
        assert stats3["drawers_added"] == 2

        # Pass 4: hub comes back on its own -- capture_pass tries it first.
        monkeypatch.setattr(
            "mempalace.codex_hub.capture_via_hub",
            lambda p, s, **kw: {
                "writer": "hub",
                "drawers_added": 0,
                "drawers_already_present": 2,
            },
        )
        stats4 = capture_pass(palace, str(root))
        assert stats4["writer"] == "hub"


class TestPerPassLeaseScoping:
    """Item 2: a direct pass's flock and storage handles live for one pass only."""

    def test_lock_file_is_free_between_direct_passes(self, tmp_path, monkeypatch):
        """A peer must be able to take the flock right after a direct pass ends."""
        import mempalace.palace as palace_mod

        palace = str(tmp_path / "palace")
        root = tmp_path / "sessions"
        _write_rollout(root, "l1", [_meta(), _item("user", "u1", "released after", 1)])

        monkeypatch.setattr(
            "mempalace.codex_hub.capture_via_hub",
            _throw(codex_hub.HubUnavailable("no hub")),
        )
        stats = capture_pass(palace, str(root))
        assert stats["writer"] == "direct"

        # A second, independent open-file-description contends exactly like a
        # peer process would -- but only if capture_pass actually gave the
        # flock back rather than holding it across the (simulated) sleep
        # between passes.
        monkeypatch.setattr(palace_mod, "_held_by_this_process", lambda key: False)
        peer = palace_mod.mine_palace_lock(palace)
        peer.__enter__()
        peer.__exit__(None, None, None)

    def test_storage_handle_is_discarded_after_a_direct_pass(self, tmp_path, monkeypatch):
        """The cached PersistentClient must not survive past the pass that opened it.

        A flock release with the backend's client still cached would leave a
        long-lived HNSW/FTS handle open across the release -- the same
        stale-index hazard mcp_server's ``_discard_mcp_storage_handles``
        exists to prevent (docs/write-routing-policy.md).
        """
        from mempalace.palace import get_backend_for_palace

        palace = str(tmp_path / "palace")
        root = tmp_path / "sessions"
        _write_rollout(root, "h1", [_meta(), _item("user", "u1", "handle closed", 1)])

        monkeypatch.setattr(
            "mempalace.codex_hub.capture_via_hub",
            _throw(codex_hub.HubUnavailable("no hub")),
        )
        stats = capture_pass(palace, str(root))
        assert stats["writer"] == "direct"
        assert stats["drawers_added"] == 1

        backend = get_backend_for_palace(palace)
        assert palace not in getattr(backend, "_clients", {palace: object()})

    def test_storage_handle_is_discarded_even_when_scan_raises(self, tmp_path, monkeypatch):
        palace = str(tmp_path / "palace")
        root = tmp_path / "sessions"
        _write_rollout(root, "x1", [_meta(), _item("user", "u1", "boom", 1)])

        monkeypatch.setattr(
            "mempalace.codex_hub.capture_via_hub",
            _throw(codex_hub.HubUnavailable("no hub")),
        )
        monkeypatch.setattr(
            "mempalace.codex_follower.scan_once",
            _throw(RuntimeError("boom")),
        )

        with pytest.raises(RuntimeError):
            capture_pass(palace, str(root))

        from mempalace.palace import get_backend_for_palace

        backend = get_backend_for_palace(palace)
        assert palace not in getattr(backend, "_clients", {palace: object()})


def test_held_pass_does_not_count_toward_escalation(tmp_path, monkeypatch):
    palace = str(tmp_path / "palace")
    hold_stats = {
        "writer": "hold",
        "error": "",
        "held_for_hub_restart": True,
        "drawers_added": 0,
        "drawers_already_present": 0,
    }
    _run_follow(palace, monkeypatch, [hold_stats] * (codex_follower.ESCALATION_THRESHOLD + 2))

    from mempalace.codex_follower import read_heartbeat

    hb = read_heartbeat(palace)
    assert hb["consecutive_failures"] == 0
    assert hb["outage_announced"] is False


def test_hold_does_not_reset_an_in_progress_failure_streak(tmp_path, monkeypatch):
    palace = str(tmp_path / "palace")
    hold_stats = {
        "writer": "hold",
        "error": "",
        "held_for_hub_restart": True,
        "drawers_added": 0,
        "drawers_already_present": 0,
    }
    results = [_fail_stats(), _fail_stats(), hold_stats, _fail_stats(), _fail_stats()]
    _run_follow(palace, monkeypatch, results)

    from mempalace.codex_follower import read_heartbeat

    hb = read_heartbeat(palace)
    assert hb["consecutive_failures"] == 4

    events = _outage_events(palace)
    assert len(events) == 1, "one status event per outage, not per pass"
    event = events[0]
    assert event["to_agent"] == "*"
    assert "mempalace_codex_capture" in event["body"]
    assert "HTTP 500" in event["body"]


def test_failures_below_threshold_do_not_announce(tmp_path, monkeypatch):
    palace = str(tmp_path / "palace")
    _run_follow(palace, monkeypatch, [_fail_stats()] * 3)
    assert _outage_events(palace) == []


def test_recovery_ends_the_outage_and_rearms_the_announcement(tmp_path, monkeypatch):
    palace = str(tmp_path / "palace")
    _run_follow(
        palace,
        monkeypatch,
        [_fail_stats()] * 4 + [_ok_stats()] + [_fail_stats("hub gone again")] * 4,
    )
    events = _outage_events(palace)
    assert len(events) == 2, "a second outage after recovery must announce again"
    assert "gone again" in events[1]["body"]


def test_status_surfaces_the_unhealthy_state(tmp_path, monkeypatch):
    palace = str(tmp_path / "palace")
    _run_follow(palace, monkeypatch, [_fail_stats()] * 4)
    st = status(palace, str(tmp_path / "sessions"))
    assert st["unhealthy"] is True
    assert st["consecutive_failures"] == 4
    assert st["state"] == "BLOCKED"


def test_healthy_follower_reports_zero_consecutive_failures(tmp_path, monkeypatch):
    palace = str(tmp_path / "palace")
    _run_follow(palace, monkeypatch, [_ok_stats()] * 2)
    st = status(palace, str(tmp_path / "sessions"))
    assert st["unhealthy"] is False
    assert st["consecutive_failures"] == 0


# ── capture tool stats: routing guard is visible to the caller ───────


def test_tool_reports_automation_and_subagent_counts(tmp_path, monkeypatch):
    from mempalace import mcp_server

    palace = str(tmp_path / "palace")
    root = tmp_path / "sessions"
    _write_rollout(root, "user", [_meta(), _item("user", "u1", "mine", 1)])
    _write_rollout(
        root,
        "auto",
        [_meta(originator="codex_exec", source="exec"), _item("user", "x1", "robot", 1)],
    )
    _write_rollout(
        root,
        "sub",
        [
            _meta(thread_id="t-sub", thread_source="subagent", parent_thread_id="p1"),
            _item("user", "s1", "delegated", 1, thread_id="t-sub"),
        ],
    )
    _write_rollout(root, "junk", ["not json", "also not json"])

    monkeypatch.setattr(mcp_server, "_config", type("C", (), {"palace_path": palace})())
    result = mcp_server.tool_codex_capture(sessions_root=str(root))
    assert result["success"] is True
    assert result["drawers_added"] == 2
    assert result["files_automation_skipped"] == 1
    assert result["subagent_turns_routed"] == 1
    assert result["files_unparseable"] == 1
    assert result["automation_turns_captured"] == 0

    opted = mcp_server.tool_codex_capture(sessions_root=str(root), include_automation=True)
    assert opted["success"] is True
    assert opted["automation_turns_captured"] == 1
    assert opted["files_automation_skipped"] == 0


# ── hub client parsing ───────────────────────────────────────────────


def test_hub_result_is_parsed_out_of_mcp_content_blocks(monkeypatch):
    payload = {
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({"success": True, "drawers_added": 4, "pending": False}),
                }
            ]
        }
    }

    class _Resp:
        def read(self):
            return json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp())
    parsed = codex_hub._call("http://x", {}, "/tmp/sessions", 10)
    assert parsed["drawers_added"] == 4
    assert parsed["pending"] is False


def test_hub_error_response_raises_refused(monkeypatch):
    payload = {"error": {"code": -32001, "message": "Peer MCP writer active"}}

    class _Resp:
        def read(self):
            return json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp())
    with pytest.raises(codex_hub.HubRefused, match="Peer MCP writer active"):
        codex_hub._call("http://x", {}, "/tmp/sessions", 10)


def test_read_only_hub_is_not_a_write_target(monkeypatch):
    monkeypatch.setattr(
        "mempalace.server_registry.read_live_serverinfo",
        lambda p: {"host": "127.0.0.1", "port": 1, "scheme": "http", "read_only": True},
    )
    assert codex_hub.hub_target("/tmp/palace") is None
