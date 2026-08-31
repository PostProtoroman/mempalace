"""Codex session capture: adapter, follower, and the end-to-end acceptance test.

The acceptance test at the bottom is the one that matters: it exercises the
whole slice — a fresh Codex session is captured automatically, a follower
restart plus rescan adds nothing, and a later session recalls the first
session's exact words.

Fixtures are synthetic but shaped like real Codex 0.151 output (verified
against the local corpus): ``UserMessage`` content blocks use ``"text"``
while ``AgentMessage`` uses ``"Text"``.
"""

from __future__ import annotations

import json

import pytest

from mempalace.codex_follower import scan_once, status
from mempalace.sources.codex import CodexSessionAdapter, iter_turns

THREAD = "01a00000-0000-7000-8000-000000000001"


def _meta(thread_id=THREAD, originator="Codex Desktop", source="vscode", **extra):
    payload = {
        "id": thread_id,
        "session_id": thread_id,
        "cwd": "/tmp/example",
        "originator": originator,
        "cli_version": "0.151.0-alpha.7.2",
        "source": source,
        "thread_source": "user",
        **extra,
    }
    return json.dumps(
        {
            "timestamp": "2026-08-31T11:47:00Z",
            "ordinal": 0,
            "type": "session_meta",
            "payload": payload,
        }
    )


def _item(role, item_id, text, ordinal, thread_id=THREAD, turn="t1"):
    item_type = "UserMessage" if role == "user" else "AgentMessage"
    # Real data: user blocks are "text", assistant blocks are "Text".
    block_type = "text" if role == "user" else "Text"
    return json.dumps(
        {
            "timestamp": "2026-08-31T11:47:05Z",
            "ordinal": ordinal,
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "thread_id": thread_id,
                "turn_id": turn,
                "item": {
                    "type": item_type,
                    "id": item_id,
                    "content": [{"type": block_type, "text": text}],
                },
            },
        }
    )


def _legacy(role, text, ordinal):
    return json.dumps(
        {
            "timestamp": "2026-07-01T10:00:00Z",
            "ordinal": ordinal,
            "type": "event_msg",
            "payload": {
                "type": f"{'user' if role == 'user' else 'agent'}_message",
                "message": text,
            },
        }
    )


def _write_rollout(root, name, lines):
    day = root / "2026" / "08" / "31"
    day.mkdir(parents=True, exist_ok=True)
    path = day / f"rollout-{name}.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ── adapter ──────────────────────────────────────────────────────────


def test_extracts_both_roles_with_real_block_casing(tmp_path):
    path = _write_rollout(
        tmp_path,
        "a",
        [
            _meta(),
            _item("user", "u1", "how do I ship this?", 1),
            _item("assistant", "a1", "run the follower", 2),
        ],
    )
    turns = list(iter_turns(str(path)))
    assert [t.role for t in turns] == ["user", "assistant"]
    assert turns[0].text == "how do I ship this?"
    assert turns[1].text == "run the follower"


def test_non_conversation_items_are_skipped(tmp_path):
    noise = json.dumps(
        {
            "type": "event_msg",
            "ordinal": 2,
            "payload": {
                "type": "item_completed",
                "thread_id": THREAD,
                "item": {
                    "type": "Reasoning",
                    "id": "r1",
                    "content": [{"type": "text", "text": "SECRET-THOUGHT"}],
                },
            },
        }
    )
    path = _write_rollout(
        tmp_path,
        "b",
        [_meta(), _item("user", "u1", "hi", 1), noise, _item("assistant", "a1", "yo", 3)],
    )
    texts = [t.text for t in iter_turns(str(path))]
    assert "SECRET-THOUGHT" not in texts
    assert texts == ["hi", "yo"]


def test_compacted_replacement_history_is_never_read(tmp_path):
    compacted = json.dumps(
        {
            "type": "compacted",
            "payload": {
                "replacement_history": [
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": "EARLIER-USER-LINE"}],
                    },
                    {
                        "role": "developer",
                        "content": [{"type": "input_text", "text": "SENTINEL-DEV-PROMPT"}],
                    },
                ]
            },
        }
    )
    path = _write_rollout(
        tmp_path,
        "c",
        [
            _meta(),
            _item("user", "u1", "EARLIER-USER-LINE", 1),
            compacted,
            _item("assistant", "a1", "ok", 3),
        ],
    )
    texts = [t.text for t in iter_turns(str(path))]
    assert texts.count("EARLIER-USER-LINE") == 1
    assert "SENTINEL-DEV-PROMPT" not in texts


@pytest.mark.parametrize(
    "originator,source",
    [("codex_exec", "exec"), ("codex_exec", "vscode"), ("Codex Desktop", "exec")],
)
def test_automation_sessions_are_excluded(tmp_path, originator, source):
    path = _write_rollout(
        tmp_path,
        f"auto-{originator}-{source}",
        [_meta(originator=originator, source=source), _item("user", "u1", "automated", 1)],
    )
    assert list(iter_turns(str(path))) == []


def test_subagent_lineage_is_preserved(tmp_path):
    path = _write_rollout(
        tmp_path,
        "sub",
        [
            _meta(thread_source="subagent", parent_thread_id="parent-123"),
            _item("user", "u1", "delegated", 1),
        ],
    )
    turn = list(iter_turns(str(path)))[0]
    assert turn.thread_source == "subagent"
    assert turn.parent_thread_id == "parent-123"


def test_resumed_thread_across_files_does_not_collide(tmp_path):
    """Regression: legacy identity must be file-scoped, not thread-scoped.

    A resumed session reuses its thread id in a new rollout with ordinals
    starting over. Deriving identity from (thread, ordinal) made two
    unrelated turns share a drawer id — measured 438 times on the real
    corpus — so the second silently overwrote the first.
    """
    a = _write_rollout(tmp_path, "resume-a", [_meta(), _legacy("user", "FIRST SESSION TEXT", 6)])
    b = _write_rollout(tmp_path, "resume-b", [_meta(), _legacy("user", "SECOND SESSION TEXT", 6)])
    ta = list(iter_turns(str(a)))[0]
    tb = list(iter_turns(str(b)))[0]
    assert ta.text != tb.text
    assert ta.drawer_id != tb.drawer_id


def test_drawer_ids_are_stable_across_reruns(tmp_path):
    path = _write_rollout(tmp_path, "stable", [_meta(), _item("user", "u1", "hello", 1)])
    assert [t.drawer_id for t in iter_turns(str(path))] == [
        t.drawer_id for t in iter_turns(str(path))
    ]


def test_adapter_declares_a_schema_covering_its_metadata(tmp_path):
    adapter = CodexSessionAdapter()
    schema = adapter.describe_schema()
    path = _write_rollout(tmp_path, "schema", [_meta(), _item("user", "u1", "hi", 1)])
    from mempalace.sources.codex import turn_to_drawer

    record = turn_to_drawer(list(iter_turns(str(path)))[0], str(path))
    for key in record.metadata:
        assert key in schema.fields, f"{key} written but not declared"
    for key, value in record.metadata.items():
        assert isinstance(value, (str, int, float, bool)), f"{key} is not a flat scalar"


# ── follower ─────────────────────────────────────────────────────────


def test_scan_is_idempotent_across_restarts(tmp_path):
    palace = str(tmp_path / "palace")
    root = tmp_path / "sessions"
    _write_rollout(
        root,
        "s1",
        [
            _meta(),
            _item("user", "u1", "first question", 1),
            _item("assistant", "a1", "first answer", 2),
        ],
    )

    first = scan_once(palace, str(root))
    assert first["drawers_added"] == 2

    # "Restart" = a brand new scan with no in-process state carried over.
    second = scan_once(palace, str(root))
    assert second["drawers_added"] == 0
    assert second["drawers_already_present"] + second["turns_skipped_by_cursor"] > 0


def test_appended_turns_are_picked_up_without_reduplicating(tmp_path):
    palace = str(tmp_path / "palace")
    root = tmp_path / "sessions"
    lines = [_meta(), _item("user", "u1", "q1", 1), _item("assistant", "a1", "a1", 2)]
    path = _write_rollout(root, "grow", lines)
    assert scan_once(palace, str(root))["drawers_added"] == 2

    path.write_text("\n".join(lines + [_item("user", "u2", "q2", 3)]) + "\n", encoding="utf-8")
    grown = scan_once(palace, str(root))
    assert grown["drawers_added"] == 1
    assert scan_once(palace, str(root))["drawers_added"] == 0


def test_status_reports_stored_codex_turns(tmp_path):
    palace = str(tmp_path / "palace")
    root = tmp_path / "sessions"
    _write_rollout(root, "st", [_meta(), _item("user", "u1", "hi", 1)])
    scan_once(palace, str(root))
    st = status(palace, str(root))
    assert st["codex_drawers"] == 1
    assert st["codex_threads"] == 1
    assert st["rollouts_on_disk"] == 1


# ── acceptance ───────────────────────────────────────────────────────


def test_acceptance_capture_restart_and_recall(tmp_path):
    """The full slice, end to end.

    1. A fresh Codex session exchanges two turns -> both land verbatim.
    2. The follower restarts and rescans -> zero new drawers.
    3. A second session starts -> the first session's exact words come back
       through search.
    """
    # search_memories is the programmatic entry point the MCP server itself
    # calls, so this exercises the same path Codex's recall will take.
    from mempalace.searcher import search_memories

    palace = str(tmp_path / "palace")
    root = tmp_path / "sessions"

    user_words = "the deploy key for the Baturiti villa dashboard rotates every 90 days"
    agent_words = "noted, I will rotate it on the first of each quarter"

    # 1. fresh session, two turns
    _write_rollout(
        root,
        "session-one",
        [_meta(), _item("user", "u1", user_words, 1), _item("assistant", "a1", agent_words, 2)],
    )
    first = scan_once(palace, str(root))
    assert first["drawers_added"] == 2, first

    from mempalace.palace import get_collection

    stored = get_collection(palace, create=True).get(
        where={"ingest_mode": "codex_session"}, include=["documents"]
    )
    docs = stored.get("documents") or []
    assert any(user_words in d for d in docs), "user turn not stored verbatim"
    assert any(agent_words in d for d in docs), "assistant turn not stored verbatim"

    # 2. restart the follower and rescan -> nothing new
    rescan = scan_once(palace, str(root))
    assert rescan["drawers_added"] == 0, rescan
    after = get_collection(palace, create=True).get(
        where={"ingest_mode": "codex_session"}, include=[]
    )
    assert len(after.get("ids") or []) == 2

    # 3. a second session starts; the first session's words are recallable
    _write_rollout(
        root,
        "session-two",
        [
            _meta(thread_id="01a00000-0000-7000-8000-000000000002"),
            _item(
                "user",
                "u9",
                "what did I say about the deploy key?",
                1,
                thread_id="01a00000-0000-7000-8000-000000000002",
            ),
        ],
    )
    scan_once(palace, str(root))

    hits = search_memories("deploy key Baturiti villa dashboard rotate", palace, n_results=10)
    blob = json.dumps(hits, default=str)
    assert user_words in blob, "first session's exact words were not recalled"
    # Verbatim, not a paraphrase: the stored text must match character for
    # character, which is the whole promise of the palace.
    texts = [h.get("text", "") for h in hits.get("results", [])]
    assert any(t.strip() == f"USER: {user_words}" for t in texts), texts


# ── recall installation ──────────────────────────────────────────────


def test_recall_block_installs_and_is_idempotent(tmp_path):
    from mempalace import codex_recall

    agents = tmp_path / "AGENTS.md"
    first = codex_recall.install(str(agents))
    assert first["changed"] is True
    assert codex_recall.is_installed(str(agents))

    second = codex_recall.install(str(agents))
    assert second["changed"] is False
    body = agents.read_text(encoding="utf-8")
    assert body.count(codex_recall.START) == 1, "re-install duplicated the block"


def test_recall_install_preserves_user_content(tmp_path):
    from mempalace import codex_recall

    agents = tmp_path / "AGENTS.md"
    agents.write_text("# My rules\n\nAlways use tabs.\n", encoding="utf-8")
    codex_recall.install(str(agents))
    body = agents.read_text(encoding="utf-8")
    assert "Always use tabs." in body
    assert codex_recall.START in body


def test_recall_block_refresh_replaces_in_place(tmp_path):
    from mempalace import codex_recall

    agents = tmp_path / "AGENTS.md"
    agents.write_text(
        f"keep me\n\n{codex_recall.START}\nstale text\n{codex_recall.END}\n\ntrailing\n",
        encoding="utf-8",
    )
    result = codex_recall.install(str(agents))
    body = agents.read_text(encoding="utf-8")
    assert result["replaced"] is True
    assert "stale text" not in body
    assert "keep me" in body and "trailing" in body
    assert body.count(codex_recall.START) == 1


def test_recall_block_tells_the_agent_to_search_first():
    from mempalace import codex_recall

    block = codex_recall.recall_block().lower()
    assert "mempalace_search" in block
    assert "verbatim" in block
