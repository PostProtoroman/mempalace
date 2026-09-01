"""Loss-proofing tests for the Hermes provider.

Covers the spool journal (crash replay, rotation, dropped-turn recovery),
ERROR-level escalation with rate limiting, palace-diary unification with
legacy migration, the single-client rule for prefetch/search, and the
packaging surface (entry point + hermes-install CLI).

Same import approach as ``test_hermes_integration.py``: the Hermes ABC is
stubbed and the provider module is loaded by file path.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import types
from pathlib import Path

import pytest


def _install_stub_memory_provider() -> None:
    if "agent.memory_provider" in sys.modules:
        return
    agent_mod = types.ModuleType("agent")
    mp_mod = types.ModuleType("agent.memory_provider")

    class MemoryProvider:
        pass

    mp_mod.MemoryProvider = MemoryProvider  # type: ignore[attr-defined]
    agent_mod.memory_provider = mp_mod  # type: ignore[attr-defined]
    sys.modules["agent"] = agent_mod
    sys.modules["agent.memory_provider"] = mp_mod


@pytest.fixture(scope="module")
def integration_module():
    _install_stub_memory_provider()
    path = (
        Path(__file__).resolve().parent.parent
        / "mempalace"
        / "integrations"
        / "hermes"
        / "__init__.py"
    )
    spec = importlib.util.spec_from_file_location("hermes_integration_hardening", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def make_provider(integration_module, palace_path, tmp_dir):
    """Factory for providers initialized against one shared temp palace."""
    created = []
    config_path = Path(tmp_dir) / "mempalace.json"
    config_path.write_text(json.dumps({"palace_path": palace_path}))

    def _make(initialize=True):
        provider = integration_module.MempalaceProvider()
        if initialize:
            provider.initialize("hardening-session", hermes_home=str(tmp_dir), platform="cli")
            provider._wake_up_done.wait(timeout=10)
        created.append(provider)
        return provider

    yield _make
    for p in created:
        p.shutdown()


@pytest.fixture(autouse=True)
def _isolate_palace_env(integration_module):
    """Undo the palace-env bridge initialize() publishes (process-global)."""
    import os

    original_env = os.environ.get("MEMPALACE_PALACE_PATH")
    original_sentinel = integration_module._ENV_PALACE_BRIDGED
    yield
    integration_module._ENV_PALACE_BRIDGED = original_sentinel
    if original_env is None:
        os.environ.pop("MEMPALACE_PALACE_PATH", None)
    else:
        os.environ["MEMPALACE_PALACE_PATH"] = original_env


def _drain(provider):
    provider._worker_queue.join()


def _stop_worker(provider):
    provider._worker_stop.set()
    if provider._worker_thread is not None:
        provider._worker_thread.join(timeout=10)


def _turn_metas(provider):
    with provider._collection_lock:
        col = provider._collection
        return col.get(where={"source": "hermes"}, include=["metadatas"]).get("metadatas") or []


# ---------------------------------------------------------------------------
# Spool journal
# ---------------------------------------------------------------------------


def test_sync_turn_journals_before_queueing(make_provider):
    provider = make_provider()
    provider.sync_turn("remember the tunnel", "noted")
    spool = provider._spool_path()
    assert spool.exists()
    lines = [json.loads(x) for x in spool.read_text().splitlines()]
    assert lines[0]["user"] == "remember the tunnel"
    assert lines[0]["turn_uid"]


def test_checkpoint_advances_after_flush(make_provider):
    provider = make_provider()
    provider.sync_turn("q1", "a1")
    _drain(provider)
    assert provider._spool_read_checkpoint() == provider._spool_path().stat().st_size
    assert provider._persistence_health()["spool_unflushed_bytes"] == 0


def test_spool_replay_after_simulated_crash(make_provider):
    # Crash sim: worker never persists (write path broken), process "dies"
    # with the spool journalled but the checkpoint at 0.
    p1 = make_provider()
    p1._file_turn = lambda payload: False  # persistence dead
    p1.sync_turn("what did we decide about the villa roof?", "alang-alang, per the BoQ")
    _drain(p1)
    assert p1._spool_read_checkpoint() == 0
    _stop_worker(p1)
    assert _turn_metas(p1) == []

    # Next run replays the unflushed line through the real filing path.
    p2 = make_provider()
    metas = _turn_metas(p2)
    assert len(metas) == 1
    assert metas[0]["turn_uid"]
    assert p2._spool_read_checkpoint() == p2._spool_path().stat().st_size

    # Replay is idempotent — a third initialize files nothing new.
    p3 = make_provider()
    assert len(_turn_metas(p3)) == 1


def test_spool_rotation_keeps_last_n(make_provider, monkeypatch, integration_module):
    provider = make_provider()
    monkeypatch.setattr(integration_module.MempalaceProvider, "SPOOL_MAX_BYTES", 64)
    provider.sync_turn("rotate me please, this line is long enough", "done")
    _drain(provider)
    _stop_worker(provider)

    p2 = make_provider()
    spool = p2._spool_path()
    assert Path(f"{spool}.1").exists()
    assert not spool.exists()
    assert p2._spool_read_checkpoint() == 0


def test_dropped_turn_is_spooled_and_replayed(make_provider):
    import queue as queue_mod

    provider = make_provider()
    _drain(provider)
    _stop_worker(provider)  # dead worker: queue can saturate
    provider._worker_queue = queue_mod.Queue(maxsize=1)
    provider._worker_stop.clear()

    provider.sync_turn("first fills the queue", "ok")
    provider.sync_turn("second is dropped", "ok")  # queue.Full path
    assert provider._dropped_turns == 1
    health = provider._persistence_health()
    assert health["total_failures"] >= 1
    # Spool carries a dropped marker line and the checkpoint is frozen
    # before the dropped turn.
    records = [json.loads(x) for x in provider._spool_path().read_text().splitlines()]
    assert any(r.get("dropped") for r in records)

    # A fresh initialize replays BOTH turns — nothing lost.
    p2 = make_provider()
    texts = [m for m in _turn_metas(p2)]
    assert len(texts) == 2


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------


def test_persistence_failures_log_error_rate_limited(make_provider, caplog):
    provider = make_provider(initialize=False)
    with caplog.at_level(logging.ERROR, logger="mempalace.hermes"):
        provider._persist_failure("disk on fire", RuntimeError("boom"))
        provider._persist_failure("disk on fire", RuntimeError("boom"))
        provider._persist_failure("disk on fire", RuntimeError("boom"))
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1  # rate-limited, not per-turn spam
    assert provider._persist_consecutive_failures == 3
    assert provider._persist_total_failures == 3
    provider._persist_success()
    assert provider._persist_consecutive_failures == 0
    assert provider._persist_total_failures == 3


def test_status_surfaces_persistence_health(make_provider):
    provider = make_provider()
    provider._persist_failure("stalled", RuntimeError("x"))
    status = provider._tool_status()
    health = status["persistence"]
    assert health["consecutive_failures"] == 1
    assert "stalled" in health["last_error"]
    assert "dropped_turns" in health and "spool_unflushed_bytes" in health


def test_file_turn_write_failure_escalates_not_debug(make_provider, caplog):
    provider = make_provider()

    class _Boom:
        def __getattr__(self, name):
            raise RuntimeError("chroma down")

    with provider._collection_lock:
        real_col = provider._collection
        provider._collection = _Boom()
    try:
        with caplog.at_level(logging.ERROR, logger="mempalace.hermes"):
            ok = provider._file_turn({"user": "u", "assistant": "a", "turn_uid": "t1"})
        assert ok is False
        assert any(r.levelno == logging.ERROR for r in caplog.records)
        assert provider._persist_consecutive_failures >= 1
    finally:
        with provider._collection_lock:
            provider._collection = real_col


# ---------------------------------------------------------------------------
# Diary unification
# ---------------------------------------------------------------------------


def test_diary_write_lands_in_palace_shape(make_provider):
    provider = make_provider()
    result = json.loads(
        provider.handle_tool_call("mempalace_diary_write", {"entry": "unified at last"})
    )
    assert result["status"] == "ok"
    with provider._collection_lock:
        got = provider._collection.get(where={"room": "diary"}, include=["documents", "metadatas"])
    assert got["documents"] == ["unified at last"]
    meta = got["metadatas"][0]
    assert meta["agent"] == "hermes"
    assert meta["hall"] == "hall_diary"
    assert meta["type"] == "diary_entry"
    assert meta["wing"] == "wing_hermes"
    # No private jsonl fork.
    assert not (Path(provider._palace_path).parent / "diary.jsonl").exists()


def test_legacy_diary_migrates_once_idempotently(make_provider):
    provider = make_provider()
    legacy = Path(provider._palace_path).parent / "diary.jsonl"
    legacy.write_text(
        json.dumps({"ts": "2026-08-01T10:00:00+00:00", "entry": "old entry one"})
        + "\n"
        + json.dumps({"ts": "2026-08-02T10:00:00+00:00", "entry": "old entry two"})
        + "\n"
    )
    read = json.loads(provider.handle_tool_call("mempalace_diary_read", {"n": 10}))
    contents = [e["entry"] for e in read["entries"]]
    assert contents == ["old entry one", "old entry two"]
    assert not legacy.exists()
    assert Path(f"{legacy}.migrated").exists()

    # Re-running the migration (fresh provider, .migrated renamed back)
    # upserts the same ids — no duplicates.
    provider._diary_migration_done = False
    Path(f"{legacy}.migrated").rename(legacy)
    read2 = json.loads(provider.handle_tool_call("mempalace_diary_read", {"n": 10}))
    assert len(read2["entries"]) == 2


def test_diary_roundtrip_via_palace(make_provider):
    provider = make_provider()
    write = json.loads(
        provider.handle_tool_call("mempalace_diary_write", {"entry": "closing the loop"})
    )
    assert write["status"] == "ok"
    read = json.loads(provider.handle_tool_call("mempalace_diary_read", {"n": 5}))
    assert read["entries"][-1]["entry"] == "closing the loop"
    assert read["entries"][-1]["content"] == "closing the loop"


# ---------------------------------------------------------------------------
# Single client
# ---------------------------------------------------------------------------


def test_prefetch_and_search_construct_no_new_client(make_provider, monkeypatch):
    provider = make_provider()
    provider.sync_turn("the villa uses alang-alang roofing", "filed")
    _drain(provider)

    import chromadb

    constructed = []
    real = chromadb.PersistentClient

    def _counting(*args, **kwargs):
        constructed.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(chromadb, "PersistentClient", _counting)

    provider.prefetch("roofing")
    provider._tool_search("roofing")
    assert constructed == []


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------


def test_hermes_entry_point_registered_in_pyproject():
    import tomllib

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    group = data["project"]["entry-points"]["hermes_agent.memory_providers"]
    assert group["mempalace"] == "mempalace.integrations.hermes:MempalaceProvider"


def test_hermes_install_dry_run_touches_nothing(tmp_path, capsys):
    from mempalace.cli import cmd_hermes_install

    args = types.SimpleNamespace(hermes_home=str(tmp_path / "hermes"), apply=False)
    rc = cmd_hermes_install(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry run" in out
    assert not (tmp_path / "hermes").exists()


def test_hermes_install_apply_copies_plugin(tmp_path, capsys):
    from mempalace.cli import cmd_hermes_install

    args = types.SimpleNamespace(hermes_home=str(tmp_path / "hermes"), apply=True)
    rc = cmd_hermes_install(args)
    assert rc == 0
    dest = tmp_path / "hermes" / "plugins" / "mempalace" / "__init__.py"
    assert dest.exists()
    # Hermes' loader text-scans for the ABC name — the copy must qualify.
    assert "MemoryProvider" in dest.read_text()
