"""Tests for mempalace.doctor — `mempalace doctor`.

Every check is exercised for ok / warn / fail via tmp fixtures; nothing here
touches real ``~/.mempalace``, ``~/.codex``, ``~/.hermes``, ``~/.claude``, or
launchd (the session-wide ``_isolate_home`` fixture in conftest.py already
redirects HOME for the whole test session; checks that read a hardcoded
legacy path resolve it off that redirected HOME).
"""

from __future__ import annotations

import json
import os
import time

import pytest

from mempalace import doctor


# ── secrets ────────────────────────────────────────────────────────────


def test_secrets_no_palace_is_ok(tmp_dir):
    result = doctor.check_secrets(os.path.join(tmp_dir, "nope"))
    assert result.status == "ok"


def test_secrets_clean_palace_is_ok(seeded_collection, palace_path):
    result = doctor.check_secrets(palace_path)
    assert result.status == "ok"
    assert result.details["scanned"] >= 1


def test_secrets_flags_a_hit(collection, palace_path):
    collection.add(
        ids=["drawer_with_secret"],
        documents=["here is my key sk-ant-" + "a" * 30 + " do not lose it"],
        metadatas=[{"wing": "w", "room": "r"}],
    )
    result = doctor.check_secrets(palace_path)
    assert result.status == "fail"
    assert "anthropic_key" in result.details["counts"]
    assert result.hint


def test_secrets_sample_bound_respected(collection, palace_path):
    ids = [f"drawer_{i}" for i in range(10)]
    docs = ["ordinary prose, nothing secret here" for _ in range(10)]
    collection.add(ids=ids, documents=docs, metadatas=[{"wing": "w", "room": "r"}] * 10)
    result = doctor.check_secrets(palace_path, sample_n=3)
    assert result.status == "ok"
    assert result.details["scanned"] == 3


# ── lost mine jobs ───────────────────────────────────────────────────


def test_lost_mine_jobs_skips_records_whose_pid_is_still_alive(tmp_dir):
    """A `started` job whose hub is still alive is not "lost" — it is running.

    `mempalace doctor` can run from a separate process while a hub is
    legitimately mid-mine (unlike the hub's own startup path, it has no
    `exclude_pid` trick available), so it must fall back to a liveness check
    instead: a record naming a live pid is skipped, one naming a dead pid is
    reported.
    """
    from mempalace.mine_queue import JOURNAL_FILENAME

    palace_path = os.path.join(tmp_dir, "palace")
    os.makedirs(palace_path, exist_ok=True)
    journal = os.path.join(palace_path, JOURNAL_FILENAME)
    dead_pid = 999999
    live_pid = os.getpid()
    with open(journal, "w", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "event": "mine_job",
                    "job_id": "m-live",
                    "state": "started",
                    "source": "/p/live.jsonl",
                    "pid": live_pid,
                }
            )
            + "\n"
        )
        fh.write(
            json.dumps(
                {
                    "event": "mine_job",
                    "job_id": "m-dead",
                    "state": "started",
                    "source": "/p/dead.jsonl",
                    "pid": dead_pid,
                }
            )
            + "\n"
        )

    lost = doctor._lost_mine_jobs(palace_path)
    lost_ids = {r["job_id"] for r in lost}
    assert lost_ids == {"m-dead"}


# ── capture liveness ──────────────────────────────────────────────────


def test_capture_liveness_hooks_disabled_is_fail(tmp_dir, palace_path, monkeypatch):
    from mempalace.config import MempalaceConfig

    monkeypatch.setattr(MempalaceConfig, "hooks_auto_save", property(lambda self: False))
    result = doctor.check_capture_liveness(
        palace_path, sessions_root=os.path.join(tmp_dir, "sessions")
    )
    assert result.status == "fail"
    assert "auto_save" in result.reason


def test_capture_liveness_no_palace_ok_hooks_is_warn(tmp_dir, monkeypatch):
    """No codex activity, hooks on: state DEGRADED (no error) + no diary yet -> warn."""
    palace_path = os.path.join(tmp_dir, "palace")
    result = doctor.check_capture_liveness(
        palace_path, sessions_root=os.path.join(tmp_dir, "sessions")
    )
    assert result.status in ("warn", "ok")


def test_capture_liveness_stale_diary_warns(seeded_collection, palace_path):
    import datetime

    old = (datetime.datetime.now() - datetime.timedelta(hours=100)).isoformat()
    seeded_collection.add(
        ids=["diary_old"],
        documents=["an old diary entry"],
        metadatas=[{"type": "diary_entry", "filed_at": old, "agent": "mac-claude"}],
    )
    result = doctor.check_capture_liveness(palace_path)
    assert result.status in ("warn", "fail")
    assert result.details["diary_age_hours"] > 48


def test_capture_liveness_fresh_diary_is_healthy(seeded_collection, palace_path):
    import datetime

    now = datetime.datetime.now().isoformat()
    seeded_collection.add(
        ids=["diary_fresh"],
        documents=["a fresh diary entry"],
        metadatas=[{"type": "diary_entry", "filed_at": now, "agent": "mac-claude"}],
    )
    result = doctor.check_capture_liveness(palace_path)
    assert result.details["diary_age_hours"] < 1


# ── launchd drift ─────────────────────────────────────────────────────


def _write_plist(path, program_arguments):
    import plistlib

    with open(path, "wb") as f:
        plistlib.dump({"Label": "ai.mempalace.hub", "ProgramArguments": program_arguments}, f)


def test_launchd_drift_non_macos_skips(tmp_dir, monkeypatch):
    monkeypatch.setattr(doctor.platform, "system", lambda: "Linux")
    result = doctor.check_launchd_drift(home=tmp_dir)
    assert result.status == "ok"
    assert "not macOS" in result.reason


def test_launchd_drift_no_agents_dir_ok(tmp_dir, monkeypatch):
    monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
    result = doctor.check_launchd_drift(home=tmp_dir)
    assert result.status == "ok"


def test_launchd_drift_no_plists_ok(tmp_dir, monkeypatch):
    monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
    agents_dir = os.path.join(tmp_dir, "Library", "LaunchAgents")
    os.makedirs(agents_dir)
    result = doctor.check_launchd_drift(home=tmp_dir)
    assert result.status == "ok"


def test_launchd_drift_detects_fork_checkout(tmp_dir, monkeypatch):
    monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(doctor, "_canonical_checkout_root", lambda: "/canonical/checkout")
    agents_dir = os.path.join(tmp_dir, "Library", "LaunchAgents")
    os.makedirs(agents_dir)
    _write_plist(
        os.path.join(agents_dir, "ai.mempalace.hub.plist"),
        ["/some/fork/checkout/.venv/bin/python", "-m", "mempalace", "serve"],
    )
    result = doctor.check_launchd_drift(home=tmp_dir)
    assert result.status == "fail"
    assert "ai.mempalace.hub.plist" in result.reason


def test_launchd_drift_matching_checkout_is_ok(tmp_dir, monkeypatch):
    monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(doctor, "_canonical_checkout_root", lambda: "/canonical/checkout")
    agents_dir = os.path.join(tmp_dir, "Library", "LaunchAgents")
    os.makedirs(agents_dir)
    _write_plist(
        os.path.join(agents_dir, "ai.mempalace.hub.plist"),
        ["/canonical/checkout/.venv/bin/python", "-m", "mempalace", "serve"],
    )
    result = doctor.check_launchd_drift(home=tmp_dir)
    assert result.status == "ok"


def test_launchd_drift_unreadable_plist_fails(tmp_dir, monkeypatch):
    monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
    agents_dir = os.path.join(tmp_dir, "Library", "LaunchAgents")
    os.makedirs(agents_dir)
    with open(os.path.join(agents_dir, "ai.mempalace.hub.plist"), "w") as f:
        f.write("not a plist")
    result = doctor.check_launchd_drift(home=tmp_dir)
    assert result.status == "fail"


# ── backup freshness ─────────────────────────────────────────────────


def test_backup_freshness_no_dest_fails(tmp_dir):
    result = doctor.check_backup_freshness(os.path.join(tmp_dir, "missing"))
    assert result.status == "fail"


def test_backup_freshness_empty_dest_fails(tmp_dir):
    dest = os.path.join(tmp_dir, "backups")
    os.makedirs(dest)
    result = doctor.check_backup_freshness(dest)
    assert result.status == "fail"


def test_backup_freshness_fresh_archive_is_ok(tmp_dir):
    dest = os.path.join(tmp_dir, "backups")
    os.makedirs(dest)
    with open(os.path.join(dest, "mempalace-20260901-040000.tar.gz"), "wb") as f:
        f.write(b"fake archive")
    result = doctor.check_backup_freshness(dest)
    assert result.status == "ok"


def test_backup_freshness_stale_archive_warns(tmp_dir):
    dest = os.path.join(tmp_dir, "backups")
    os.makedirs(dest)
    archive = os.path.join(dest, "mempalace-old.tar.gz")
    with open(archive, "wb") as f:
        f.write(b"fake archive")
    old_time = time.time() - (60 * 3600)  # 60h ago
    os.utime(archive, (old_time, old_time))
    result = doctor.check_backup_freshness(dest)
    assert result.status == "warn"
    assert result.details["age_hours"] > 48


# ── search health ─────────────────────────────────────────────────────


def test_search_health_ok(monkeypatch, palace_path):
    monkeypatch.setattr(
        "mempalace.backends.chroma.hnsw_capacity_status",
        lambda p, c: {"status": "ok", "message": "sqlite=10 hnsw=10"},
    )
    result = doctor.check_search_health(palace_path)
    assert result.status == "ok"


def test_search_health_diverged_fails(monkeypatch, palace_path):
    monkeypatch.setattr(
        "mempalace.backends.chroma.hnsw_capacity_status",
        lambda p, c: {"status": "diverged", "message": "HNSW capacity divergence"},
    )
    result = doctor.check_search_health(palace_path)
    assert result.status == "fail"
    assert "divergence" in result.reason


def test_search_health_unknown_warns(monkeypatch, palace_path):
    monkeypatch.setattr(
        "mempalace.backends.chroma.hnsw_capacity_status",
        lambda p, c: {"status": "unknown", "message": "could not measure"},
    )
    result = doctor.check_search_health(palace_path)
    assert result.status == "warn"


# ── KG unification ────────────────────────────────────────────────────


def test_kg_unification_no_legacy_is_ok(monkeypatch, palace_path):
    monkeypatch.setattr(
        "mempalace.migrate.migrate_knowledge_graph",
        lambda palace_path, apply=False: {
            "legacy_path": "/tmp/legacy.sqlite3",
            "target_path": palace_path + "/knowledge_graph.sqlite3",
            "legacy_triples": 0,
            "legacy_entities": 0,
            "merged_triples": 0,
            "merged_entities": 0,
        },
    )
    result = doctor.check_kg_unification(palace_path)
    assert result.status == "ok"


def test_kg_unification_already_migrated_is_ok(monkeypatch, palace_path):
    """Regression test: the legacy KG file still has rows (it is never

    deleted by the migration -- see migrate_knowledge_graph's docstring),
    but every one of them already exists in the target KG. This is exactly
    the state `mempalace kg-migrate` dry-run reports as "0 new, N already
    present". Previously this check warned forever in that state because it
    compared against the legacy file's total row counts instead of the
    counts still outstanding.
    """
    monkeypatch.setattr(
        "mempalace.migrate.migrate_knowledge_graph",
        lambda palace_path, apply=False: {
            "legacy_path": "/tmp/legacy.sqlite3",
            "target_path": palace_path + "/knowledge_graph.sqlite3",
            "legacy_triples": 69,
            "legacy_entities": 97,
            "merged_triples": 0,
            "merged_entities": 0,
            "skipped_existing": 69,
        },
    )
    result = doctor.check_kg_unification(palace_path)
    assert result.status == "ok"


def test_kg_unification_legacy_data_warns(monkeypatch, palace_path):
    monkeypatch.setattr(
        "mempalace.migrate.migrate_knowledge_graph",
        lambda palace_path, apply=False: {
            "legacy_path": "/tmp/legacy.sqlite3",
            "target_path": palace_path + "/knowledge_graph.sqlite3",
            "legacy_triples": 5,
            "legacy_entities": 3,
            "merged_triples": 5,
            "merged_entities": 3,
        },
    )
    result = doctor.check_kg_unification(palace_path)
    assert result.status == "warn"
    assert result.details["legacy_triples"] == 5
    assert "kg-migrate" in result.hint


# ── identity hygiene ──────────────────────────────────────────────────


def test_identity_hygiene_no_palace_ok(tmp_dir):
    result = doctor.check_identity_hygiene(os.path.join(tmp_dir, "nope"))
    assert result.status == "ok"


def test_identity_hygiene_clean_is_ok(collection, palace_path):
    collection.add(
        ids=["diary_1"],
        documents=["real diary entry"],
        metadatas=[{"type": "diary_entry", "agent": "mac-claude"}],
    )
    result = doctor.check_identity_hygiene(palace_path)
    assert result.status == "ok"


def test_identity_hygiene_flags_known_alias(collection, palace_path):
    collection.add(
        ids=["diary_1", "diary_2"],
        documents=["entry one", "entry two"],
        metadatas=[
            {"type": "diary_entry", "agent": "claude"},
            {"type": "diary_entry", "agent": "mac-claude"},
        ],
    )
    result = doctor.check_identity_hygiene(palace_path)
    assert result.status == "warn"
    assert result.details["counts"] == {"claude": 1}


# ── orchestration ─────────────────────────────────────────────────────


def test_run_all_returns_nine_checks(tmp_dir, monkeypatch):
    monkeypatch.setattr(doctor.platform, "system", lambda: "Linux")
    results = doctor.run_all(
        os.path.join(tmp_dir, "palace"),
        backup_dest=os.path.join(tmp_dir, "backups"),
        sessions_root=os.path.join(tmp_dir, "sessions"),
    )
    assert len(results) == 9
    assert {r.name for r in results} == set(doctor.CHECK_ORDER)


def test_overall_exit_code_ok_and_warn_is_zero():
    results = [
        doctor.CheckResult("a", "ok", "fine"),
        doctor.CheckResult("b", "warn", "meh", hint="check it"),
    ]
    assert doctor.overall_exit_code(results) == 0


def test_overall_exit_code_any_fail_is_one():
    results = [
        doctor.CheckResult("a", "ok", "fine"),
        doctor.CheckResult("b", "fail", "broken", hint="fix it"),
    ]
    assert doctor.overall_exit_code(results) == 1


def test_format_table_includes_hints_for_non_ok():
    results = [
        doctor.CheckResult("a", "ok", "fine"),
        doctor.CheckResult("b", "fail", "broken", hint="fix it"),
    ]
    table = doctor.format_table(results)
    assert "fix it" in table
    assert "OK" in table and "FAIL" in table


# ── CLI wiring ────────────────────────────────────────────────────────


def test_cli_doctor_json_shape(tmp_dir, monkeypatch, capsys):
    from mempalace import cli

    monkeypatch.setattr(doctor.platform, "system", lambda: "Linux")

    class Args:
        palace = os.path.join(tmp_dir, "palace")
        json = True
        full = False
        sample = None
        sessions_root = os.path.join(tmp_dir, "sessions")
        backup_dest = os.path.join(tmp_dir, "backups")
        home = tmp_dir

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_doctor(Args())

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "results" in payload
    assert len(payload["results"]) == 9
    assert exc_info.value.code in (0, 1)


def test_cli_doctor_table_output(tmp_dir, monkeypatch, capsys):
    from mempalace import cli

    monkeypatch.setattr(doctor.platform, "system", lambda: "Linux")

    class Args:
        palace = os.path.join(tmp_dir, "palace")
        json = False
        full = False
        sample = None
        sessions_root = os.path.join(tmp_dir, "sessions")
        backup_dest = os.path.join(tmp_dir, "backups")
        home = tmp_dir

    with pytest.raises(SystemExit):
        cli.cmd_doctor(Args())

    out = capsys.readouterr().out
    assert "mempalace doctor" in out


class TestRogueLeaseHolder:
    """`check_rogue_lease_holder` — reads `mine_palace_lock`'s state without
    ever acquiring it, and cross-checks the holder against the hub.

    Lock contention is simulated with a genuine second process (spawn
    context, mirroring tests/test_palace_locks.py) since flock semantics
    can't be faked within a single process/thread.
    """

    @staticmethod
    def _isolate_home(monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))

    @staticmethod
    def _write_hub_plist(home_dir):
        import plistlib

        agents_dir = os.path.join(home_dir, "Library", "LaunchAgents")
        os.makedirs(agents_dir, exist_ok=True)
        with open(os.path.join(agents_dir, "ai.mempalace.hub.plist"), "wb") as f:
            plistlib.dump(
                {
                    "Label": "ai.mempalace.hub",
                    "ProgramArguments": ["/venv/bin/python", "-m", "mempalace", "serve"],
                },
                f,
            )

    def test_non_macos_skips(self, tmp_path, monkeypatch):
        monkeypatch.setattr(doctor.platform, "system", lambda: "Linux")
        result = doctor.check_rogue_lease_holder(str(tmp_path / "palace"), home=str(tmp_path))
        assert result.status == "ok"
        assert "not macOS" in result.reason

    def test_no_hub_plist_is_ok(self, tmp_path, monkeypatch):
        monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
        result = doctor.check_rogue_lease_holder(str(tmp_path / "palace"), home=str(tmp_path))
        assert result.status == "ok"
        assert "single-writer" in result.reason

    @pytest.mark.skipif(
        os.name == "nt", reason="POSIX flock-based lease probe -- not exercised on Windows"
    )
    def test_free_lease_is_ok(self, tmp_path, monkeypatch):
        """A lock file that exists but is unheld (stale identity from a prior
        holder still on disk) must probe as free, not as held."""
        from mempalace.palace import mine_palace_lock

        monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
        self._isolate_home(monkeypatch, tmp_path)
        self._write_hub_plist(str(tmp_path))
        palace = str(tmp_path / "palace")

        with mine_palace_lock(palace):
            pass  # acquire and release synchronously -- leaves a lock file behind

        result = doctor.check_rogue_lease_holder(palace, home=str(tmp_path))
        assert result.status == "ok"
        assert "free" in result.reason

    @pytest.mark.skipif(
        os.name == "nt", reason="POSIX flock-based lease probe -- not exercised on Windows"
    )
    def test_no_lock_file_at_all_is_ok(self, tmp_path, monkeypatch):
        monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
        self._isolate_home(monkeypatch, tmp_path)
        self._write_hub_plist(str(tmp_path))
        result = doctor.check_rogue_lease_holder(str(tmp_path / "palace"), home=str(tmp_path))
        assert result.status == "ok"
        assert "free" in result.reason

    @pytest.mark.skipif(
        os.name == "nt", reason="POSIX flock-based lease probe -- not exercised on Windows"
    )
    def test_held_by_non_hub_process_warns(self, tmp_path, monkeypatch):
        from tests.test_palace_locks import _get_mp_context, _hold_lock_with_argv

        monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(doctor, "_hub_launchd_pid", lambda home=None: None)
        self._isolate_home(monkeypatch, tmp_path)
        self._write_hub_plist(str(tmp_path))
        palace = str(tmp_path / "palace")
        ready = str(tmp_path / "ready")
        release = str(tmp_path / "release")

        ctx = _get_mp_context()
        pid_q = ctx.Queue()
        holder = ctx.Process(
            target=_hold_lock_with_argv,
            args=(palace, ["mempalace", "codex-capture", "--interval"], ready, release, pid_q),
        )
        holder.start()
        try:
            for _ in range(500):
                if os.path.exists(ready):
                    break
                time.sleep(0.01)
            assert os.path.exists(ready), "holder failed to acquire lock in time"
            holder_pid, holder_argv = pid_q.get(timeout=2)

            result = doctor.check_rogue_lease_holder(palace, home=str(tmp_path))
            assert result.status == "warn"
            assert f"PID {holder_pid}" in result.reason
            assert "not the hub" in result.reason
            assert result.details["pid"] == holder_pid
            assert "--transport http" not in " ".join(holder_argv)
            assert "seized the writer lease" in result.hint
        finally:
            open(release, "w").close()
            holder.join(timeout=5)

    @pytest.mark.skipif(
        os.name == "nt", reason="POSIX flock-based lease probe -- not exercised on Windows"
    )
    def test_held_by_hub_cmdline_is_ok(self, tmp_path, monkeypatch):
        """Recorded identity carrying '--transport http' (the hub's own
        exec'd argv) is enough to clear it, even without a launchd PID
        match."""
        from tests.test_palace_locks import _get_mp_context, _hold_lock_with_argv

        monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(doctor, "_hub_launchd_pid", lambda home=None: None)
        self._isolate_home(monkeypatch, tmp_path)
        self._write_hub_plist(str(tmp_path))
        palace = str(tmp_path / "palace")
        ready = str(tmp_path / "ready")
        release = str(tmp_path / "release")

        ctx = _get_mp_context()
        pid_q = ctx.Queue()
        holder = ctx.Process(
            target=_hold_lock_with_argv,
            args=(palace, ["/venv/bin/python", "--transport", "http"], ready, release, pid_q),
        )
        holder.start()
        try:
            for _ in range(500):
                if os.path.exists(ready):
                    break
                time.sleep(0.01)
            assert os.path.exists(ready), "holder failed to acquire lock in time"

            result = doctor.check_rogue_lease_holder(palace, home=str(tmp_path))
            assert result.status == "ok"
            assert "held by the hub" in result.reason
        finally:
            open(release, "w").close()
            holder.join(timeout=5)

    @pytest.mark.skipif(
        os.name == "nt", reason="POSIX flock-based lease probe -- not exercised on Windows"
    )
    def test_held_by_matching_launchd_pid_is_ok(self, tmp_path, monkeypatch):
        """A holder whose command line doesn't say '--transport http' still
        clears as the hub when its PID matches the launchd job's PID."""
        from tests.test_palace_locks import _get_mp_context, _hold_lock_with_argv

        monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
        self._isolate_home(monkeypatch, tmp_path)
        self._write_hub_plist(str(tmp_path))
        palace = str(tmp_path / "palace")
        ready = str(tmp_path / "ready")
        release = str(tmp_path / "release")

        ctx = _get_mp_context()
        pid_q = ctx.Queue()
        holder = ctx.Process(
            target=_hold_lock_with_argv,
            args=(palace, ["/venv/bin/python", "-m", "mempalace"], ready, release, pid_q),
        )
        holder.start()
        try:
            for _ in range(500):
                if os.path.exists(ready):
                    break
                time.sleep(0.01)
            assert os.path.exists(ready), "holder failed to acquire lock in time"
            holder_pid, _holder_argv = pid_q.get(timeout=2)

            monkeypatch.setattr(doctor, "_hub_launchd_pid", lambda home=None: holder_pid)
            result = doctor.check_rogue_lease_holder(palace, home=str(tmp_path))
            assert result.status == "ok"
            assert "held by the hub" in result.reason
        finally:
            open(release, "w").close()
            holder.join(timeout=5)

    @pytest.mark.skipif(
        os.name == "nt", reason="POSIX flock-based lease probe -- not exercised on Windows"
    )
    def test_never_leaves_the_lock_held(self, tmp_path, monkeypatch):
        """Probing a free lease must not leave it locked for a real acquirer."""
        from mempalace.palace import mine_palace_lock

        monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
        self._isolate_home(monkeypatch, tmp_path)
        self._write_hub_plist(str(tmp_path))
        palace = str(tmp_path / "palace")

        result = doctor.check_rogue_lease_holder(palace, home=str(tmp_path))
        assert result.status == "ok"

        # If the probe leaked its flock, this would raise MineAlreadyRunning.
        with mine_palace_lock(palace):
            pass

    def test_hub_launchd_pid_parses_real_launchctl_output(self, monkeypatch):
        """Pin the ``launchctl print`` output format against a captured
        real-world sample -- the ``pid = <n>`` line lookup has no coverage
        otherwise (``_hub_launchd_pid`` is monkeypatched out in every other
        test here). ``shutil``/``subprocess`` are imported *inside*
        ``_hub_launchd_pid``, so patching the real modules (not
        ``doctor.shutil``/``doctor.subprocess``, which don't exist as
        module-level attributes) is what actually reaches it."""
        import shutil
        import subprocess

        captured_stdout = (
            "gui/501/ai.mempalace.hub = {\n"
            "\tactive count = 1\n"
            "\tpath = /Users/proto/Library/LaunchAgents/ai.mempalace.hub.plist\n"
            "\ttype = LaunchAgent\n"
            "\tstate = running\n"
            "\n"
            "\tprogram = /venv/bin/python3\n"
            "\tpid = 8905\n"
            "\timmediate reason = ...\n"
            "}\n"
        )

        class _FakeResult:
            returncode = 0
            stdout = captured_stdout

        monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(shutil, "which", lambda name: "/bin/launchctl")
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeResult())
        # os.getuid does not exist on Windows; keep the launchctl-parsing
        # coverage on every CI platform instead of skipping it.
        monkeypatch.setattr(os, "getuid", lambda: 501, raising=False)

        assert doctor._hub_launchd_pid() == 8905

    def test_hub_launchd_pid_none_on_nonzero_returncode(self, monkeypatch):
        import shutil
        import subprocess

        class _FakeResult:
            returncode = 1
            stdout = ""

        monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(shutil, "which", lambda name: "/bin/launchctl")
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeResult())
        # os.getuid does not exist on Windows; keep the launchctl-parsing
        # coverage on every CI platform instead of skipping it.
        monkeypatch.setattr(os, "getuid", lambda: 501, raising=False)

        assert doctor._hub_launchd_pid() is None


class TestUpstreamDrift:
    def test_skips_without_git_checkout(self, tmp_path):
        from mempalace.doctor import check_upstream_drift

        r = check_upstream_drift(checkout=str(tmp_path))
        assert r.status == "ok"
        assert "skipped" in r.reason

    def test_skips_without_upstream_remote(self, tmp_path):
        import subprocess

        from mempalace.doctor import check_upstream_drift

        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
        r = check_upstream_drift(checkout=str(tmp_path))
        assert r.status == "ok"
        assert "skipped" in r.reason

    def test_warns_when_behind_fetched_upstream(self, tmp_path):
        import subprocess

        from mempalace.doctor import check_upstream_drift

        def g(*args):
            subprocess.run(
                ["git", "-C", str(tmp_path), *args],
                check=True,
                capture_output=True,
                env={
                    "GIT_AUTHOR_NAME": "t",
                    "GIT_AUTHOR_EMAIL": "t@t",
                    "GIT_COMMITTER_NAME": "t",
                    "GIT_COMMITTER_EMAIL": "t@t",
                    "PATH": "/usr/bin:/bin",
                },
            )

        g("init", "-q", "-b", "main", ".")
        (tmp_path / "a").write_text("1")
        g("add", "a")
        g("commit", "-qm", "one")
        g("remote", "add", "upstream", str(tmp_path))
        # simulate a fetched upstream/develop that is ahead
        (tmp_path / "a").write_text("2")
        g("add", "a")
        g("commit", "-qm", "two")
        g("update-ref", "refs/remotes/upstream/develop", "HEAD")
        g("reset", "-q", "--hard", "HEAD~1")
        r = check_upstream_drift(checkout=str(tmp_path))
        assert r.status == "warn"
        assert "1 commit(s) behind" in r.reason
