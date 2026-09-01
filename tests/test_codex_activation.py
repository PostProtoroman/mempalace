"""Activation readiness for Codex capture.

Turning capture on must be a small, predictable act: it starts capturing new
turns, it does not import years of history behind the user's back, it files
turns where conversation mining would file them, it runs the MemPalace the
user installed rather than whatever is first on PATH, and it never edits the
user's instructions file unless asked.
"""

from __future__ import annotations

import plistlib
import sys

import pytest

from mempalace import codex_recall, codex_service
from mempalace.codex_follower import (
    capture_pass,
    ensure_activation,
    plan_backfill,
    read_activation,
    scan_once,
)
from mempalace.sources.codex import iter_turns, turn_to_drawer

from tests.test_codex_capture import _item, _meta, _write_rollout


def _docs(palace):
    from mempalace.palace import get_collection

    data = get_collection(palace, create=False).get(
        where={"ingest_mode": "codex_session"}, include=["documents"]
    )
    return data.get("documents") or []


# ── activation point ─────────────────────────────────────────────────


def test_activation_captures_new_turns_and_not_history(tmp_path):
    palace = str(tmp_path / "palace")
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "hist",
        [_meta(), _item("user", "h1", "OLD HISTORY", 1), _item("assistant", "h2", "OLD ANSWER", 2)],
    )

    ensure_activation(palace, str(root))
    assert scan_once(palace, str(root))["drawers_added"] == 0

    path.write_text(
        path.read_text(encoding="utf-8") + _item("user", "n1", "NEW TURN", 5) + "\n",
        encoding="utf-8",
    )
    assert scan_once(palace, str(root))["drawers_added"] == 1

    docs = _docs(palace)
    assert any("NEW TURN" in d for d in docs)
    assert not any("OLD HISTORY" in d for d in docs)


def test_history_is_never_imported_without_explicit_backfill(tmp_path):
    palace = str(tmp_path / "palace")
    root = tmp_path / "sessions"
    lines = [_meta()] + [_item("user", f"h{i}", f"history {i}", i) for i in range(1, 21)]
    _write_rollout(root, "big", lines)

    ensure_activation(palace, str(root))
    for _ in range(3):
        assert scan_once(palace, str(root))["drawers_added"] == 0
    assert _docs(palace) == []


def test_explicit_backfill_imports_history(tmp_path):
    palace = str(tmp_path / "palace")
    root = tmp_path / "sessions"
    _write_rollout(root, "h", [_meta(), _item("user", "h1", "OLD HISTORY", 1)])
    ensure_activation(palace, str(root))
    assert scan_once(palace, str(root))["drawers_added"] == 0

    assert scan_once(palace, str(root), backfill=True)["drawers_added"] == 1
    assert any("OLD HISTORY" in d for d in _docs(palace))


def test_backfill_ignores_the_forward_cursor(tmp_path):
    """Regression: the cursor is a forward optimisation and must not block a
    deliberate walk backwards into history."""
    palace = str(tmp_path / "palace")
    root = tmp_path / "sessions"
    path = _write_rollout(root, "c", [_meta(), _item("user", "old", "EARLY TURN", 1)])
    ensure_activation(palace, str(root))
    path.write_text(
        path.read_text(encoding="utf-8") + _item("user", "new", "LATER TURN", 9) + "\n",
        encoding="utf-8",
    )
    scan_once(palace, str(root))  # advances the cursor to ordinal 9
    assert any("LATER TURN" in d for d in _docs(palace))

    scan_once(palace, str(root), backfill=True)
    assert any("EARLY TURN" in d for d in _docs(palace))


def test_activation_point_is_recorded_once(tmp_path):
    palace = str(tmp_path / "palace")
    root = tmp_path / "sessions"
    _write_rollout(root, "a", [_meta(), _item("user", "u1", "hi", 3)])
    first = ensure_activation(palace, str(root))
    second = ensure_activation(palace, str(root))
    assert first["activated_at"] == second["activated_at"]
    assert read_activation(palace)["baseline"]["rollout-a"] == 3


def test_rollouts_created_after_activation_are_captured_in_full(tmp_path):
    palace = str(tmp_path / "palace")
    root = tmp_path / "sessions"
    _write_rollout(root, "before", [_meta(), _item("user", "b1", "PRE-EXISTING", 1)])
    ensure_activation(palace, str(root))

    _write_rollout(
        root, "after", [_meta(thread_id="t2"), _item("user", "a1", "BRAND NEW SESSION", 1)]
    )
    scan_once(palace, str(root))
    docs = _docs(palace)
    assert any("BRAND NEW SESSION" in d for d in docs)
    assert not any("PRE-EXISTING" in d for d in docs)


# ── backfill dry run ─────────────────────────────────────────────────


def test_backfill_dry_run_reports_without_writing(tmp_path):
    palace = str(tmp_path / "palace")
    root = tmp_path / "sessions"
    _write_rollout(
        root,
        "plan",
        [_meta(), _item("user", "u1", "one", 1), _item("assistant", "a1", "two", 2)],
    )
    _write_rollout(
        root, "auto", [_meta(originator="codex_exec", source="exec"), _item("user", "x", "y", 1)]
    )
    _write_rollout(
        root,
        "sub",
        [
            _meta(thread_id="t3", thread_source="subagent", parent_thread_id="p1"),
            _item("user", "s1", "sub turn", 1, thread_id="t3"),
        ],
    )

    plan = plan_backfill(palace, str(root))
    assert plan["files_scanned"] == 3
    assert plan["turns_found"] == 3
    assert plan["estimated_new_drawers"] == 3
    assert plan["automation_files_excluded"] == 1
    assert plan["subagent_turns_included"] == 1
    assert plan["destinations"]
    assert _docs(palace) == [] if (tmp_path / "palace").exists() else True


def test_dry_run_counts_already_present_separately(tmp_path):
    palace = str(tmp_path / "palace")
    root = tmp_path / "sessions"
    _write_rollout(root, "p", [_meta(), _item("user", "u1", "already here", 1)])
    scan_once(palace, str(root), backfill=True)

    plan = plan_backfill(palace, str(root))
    assert plan["already_present"] == 1
    assert plan["estimated_new_drawers"] == 0


# ── canonical routing ────────────────────────────────────────────────


def test_turns_are_routed_by_the_conversation_miner_resolvers(tmp_path):
    """No more unknown/unknown: wing and room come from convo_miner."""
    from mempalace.convo_miner import _resolve_wing, detect_convo_room

    codex_dir = tmp_path / ".codex" / "sessions"
    path = _write_rollout(
        codex_dir, "route", [_meta(), _item("user", "u1", "deploy the server", 1)]
    )
    turn = list(iter_turns(str(path)))[0]
    record = turn_to_drawer(turn, str(path))

    assert record.metadata["wing"] == _resolve_wing(path, None) == "wing_api"
    assert record.metadata["room"] == detect_convo_room(turn.text)
    assert record.metadata["wing"] != "unknown"
    assert record.metadata["room"] != "unknown"


def test_routing_metadata_is_preserved_end_to_end(tmp_path):
    from mempalace.sources.codex import SUBAGENT_WING

    palace = str(tmp_path / "palace")
    codex_dir = tmp_path / ".codex" / "sessions"
    _write_rollout(
        codex_dir,
        "meta",
        [
            _meta(thread_source="subagent", parent_thread_id="parent-9", cwd="/work/project"),
            _item("user", "u1", "hello there", 4),
        ],
    )
    scan_once(palace, str(codex_dir), backfill=True)

    from mempalace.palace import get_collection

    data = get_collection(palace, create=False).get(
        where={"ingest_mode": "codex_session"}, include=["metadatas"]
    )
    meta = (data.get("metadatas") or [])[0]
    # A subagent session never lands in the user's wing — it goes to the
    # dedicated subagent wing with its lineage intact.
    assert meta["wing"] == SUBAGENT_WING
    assert meta["room"] and meta["room"] != "unknown"
    assert meta["session_id"] and meta["thread_id"]
    assert meta["parent_thread_id"] == "parent-9"
    assert meta["thread_source"] == "subagent"
    assert meta["workspace"] == "/work/project"  # cwd carried from session_meta
    assert meta["authored_at"]
    assert meta["source_file"].endswith("rollout-meta.jsonl")


# ── exact executable ─────────────────────────────────────────────────


def test_plist_pins_an_absolute_interpreter_not_a_path_name(tmp_path):
    definition = codex_service.build_plist(
        str(tmp_path / "palace"), str(tmp_path / "sessions"), home=str(tmp_path / "home")
    )
    program = definition["ProgramArguments"]
    assert program[0].startswith("/"), "service must not rely on PATH lookup"
    assert program[0] == sys.executable
    assert program[1:3] == ["-m", "mempalace"]


def test_resolve_executable_refuses_an_interpreter_without_this_package(tmp_path):
    fake = tmp_path / "python-without-mempalace"
    fake.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    fake.chmod(0o755)
    with pytest.raises(codex_service.ServiceError, match="does not import this MemPalace"):
        codex_service.resolve_executable(str(fake))


def test_resolve_executable_accepts_the_running_interpreter(tmp_path):
    executable, env = codex_service.resolve_executable()
    assert executable == sys.executable
    assert "PYTHONPATH" not in env or env["PYTHONPATH"]


# ── recall stays explicit ────────────────────────────────────────────


def test_service_plist_never_installs_recall(tmp_path):
    definition = codex_service.build_plist(
        str(tmp_path / "palace"), str(tmp_path / "sessions"), home=str(tmp_path / "home")
    )
    assert "--no-recall" in definition["ProgramArguments"]


@pytest.mark.skipif(not codex_service.is_supported(), reason="launchd only")
def test_service_install_does_not_touch_agents_md(tmp_path):
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    agents = home / "AGENTS.md"
    agents.write_text("# mine\n\nuntouched\n", encoding="utf-8")
    before = agents.read_text(encoding="utf-8")

    codex_service.install(
        str(tmp_path / "palace"), str(tmp_path / "sessions"), home=str(home), load=False
    )
    assert agents.read_text(encoding="utf-8") == before


def test_capture_pass_does_not_touch_agents_md(tmp_path):
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    agents = home / "AGENTS.md"
    agents.write_text("# mine\n\nuntouched by capture\n", encoding="utf-8")
    before = agents.read_text(encoding="utf-8")

    root = tmp_path / "sessions"
    _write_rollout(root, "cap", [_meta(), _item("user", "u1", "hi", 1)])
    capture_pass(str(tmp_path / "palace"), str(root), backfill=True)

    assert agents.read_text(encoding="utf-8") == before
    assert not codex_recall.is_installed(str(agents))


def test_explicit_recall_install_preserves_user_content(tmp_path):
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# my rules\n\nkeep this line\n", encoding="utf-8")
    codex_recall.install(str(agents))
    body = agents.read_text(encoding="utf-8")
    assert "keep this line" in body
    assert codex_recall.is_installed(str(agents))


# ── restart idempotence with the new routing ─────────────────────────


def test_restart_after_activation_adds_nothing(tmp_path):
    palace = str(tmp_path / "palace")
    codex_dir = tmp_path / ".codex" / "sessions"
    path = _write_rollout(codex_dir, "idem", [_meta(), _item("user", "u1", "first", 1)])
    ensure_activation(palace, str(codex_dir))
    path.write_text(
        path.read_text(encoding="utf-8") + _item("assistant", "a1", "second", 7) + "\n",
        encoding="utf-8",
    )
    assert scan_once(palace, str(codex_dir))["drawers_added"] == 1
    for _ in range(3):
        assert scan_once(palace, str(codex_dir))["drawers_added"] == 0
    assert len(_docs(palace)) == 1


@pytest.mark.skipif(not codex_service.is_supported(), reason="launchd only")
def test_uninstall_preserves_palace_and_agents_md(tmp_path):
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    agents = home / "AGENTS.md"
    agents.write_text("# mine\n\nsurvives uninstall\n", encoding="utf-8")
    codex_recall.install(str(agents))
    before = agents.read_text(encoding="utf-8")

    palace = str(tmp_path / "palace")
    root = tmp_path / "sessions"
    _write_rollout(root, "keep", [_meta(), _item("user", "u1", "keep this turn", 1)])
    scan_once(palace, str(root), backfill=True)

    codex_service.install(palace, str(root), home=str(home), load=False)
    codex_service.uninstall(palace, home=str(home))

    assert not codex_service.is_installed(str(home))
    assert agents.read_text(encoding="utf-8") == before
    assert any("keep this turn" in d for d in _docs(palace))


@pytest.mark.skipif(not codex_service.is_supported(), reason="launchd only")
def test_installed_plist_round_trips(tmp_path):
    home = tmp_path / "home"
    codex_service.install(
        str(tmp_path / "palace"), str(tmp_path / "sessions"), home=str(home), load=False
    )
    with open(codex_service.plist_path(str(home)), "rb") as fh:
        definition = plistlib.load(fh)
    assert definition["RunAtLoad"] is True
    assert definition["ProgramArguments"][0] == sys.executable
