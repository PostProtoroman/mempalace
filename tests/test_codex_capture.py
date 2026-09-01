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


# ── routing guard: machine sessions never reach the user's wing ──────


@pytest.mark.parametrize("thread_source", ["subagent", "guardian_review"])
def test_machine_sessions_route_to_the_dedicated_wing(tmp_path, thread_source):
    from mempalace.palace import get_collection
    from mempalace.sources.codex import SUBAGENT_WING

    palace = str(tmp_path / "palace")
    root = tmp_path / "sessions"
    _write_rollout(
        root,
        "user",
        [_meta(), _item("user", "u1", "my own question", 1)],
    )
    _write_rollout(
        root,
        "machine",
        [
            _meta(thread_id="t-mach", thread_source=thread_source, parent_thread_id="parent-7"),
            _item("user", "m1", "delegated work", 1, thread_id="t-mach"),
        ],
    )

    stats = scan_once(palace, str(root))
    assert stats["drawers_added"] == 2
    assert stats["subagent_turns_routed"] == 1

    data = get_collection(palace, create=False).get(
        where={"ingest_mode": "codex_session"}, include=["metadatas", "documents"]
    )
    by_doc = {d: m for d, m in zip(data["documents"], data["metadatas"])}
    machine = next(m for d, m in by_doc.items() if "delegated work" in d)
    user = next(m for d, m in by_doc.items() if "my own question" in d)
    assert machine["wing"] == SUBAGENT_WING
    assert machine["parent_thread_id"] == "parent-7"
    assert user["wing"] != SUBAGENT_WING


def test_automation_is_skipped_by_default_and_counted(tmp_path):
    from mempalace.palace import get_collection

    palace = str(tmp_path / "palace")
    root = tmp_path / "sessions"
    _write_rollout(
        root,
        "auto",
        [_meta(originator="codex_exec", source="exec"), _item("user", "x1", "machine run", 1)],
    )

    stats = scan_once(palace, str(root))
    assert stats["drawers_added"] == 0
    assert stats["files_automation_skipped"] == 1
    assert stats["automation_turns_captured"] == 0
    data = get_collection(palace, create=True).get(include=[])
    assert (data.get("ids") or []) == []


def test_automation_opt_in_routes_to_the_automation_wing(tmp_path):
    from mempalace.palace import get_collection
    from mempalace.sources.codex import AUTOMATION_WING

    palace = str(tmp_path / "palace")
    root = tmp_path / "sessions"
    _write_rollout(
        root,
        "auto",
        [_meta(originator="codex_exec", source="exec"), _item("user", "x1", "machine run", 1)],
    )

    stats = scan_once(palace, str(root), include_automation=True)
    assert stats["drawers_added"] == 1
    assert stats["automation_turns_captured"] == 1
    assert stats["files_automation_skipped"] == 0

    meta = get_collection(palace, create=False).get(
        where={"ingest_mode": "codex_session"}, include=["metadatas"]
    )["metadatas"][0]
    assert meta["wing"] == AUTOMATION_WING
    assert meta["automation"] is True


# ── no garbage fallthrough ───────────────────────────────────────────


def test_unparseable_rollout_is_counted_warned_and_never_chunked(tmp_path, caplog):
    import logging

    from mempalace.palace import get_collection

    palace = str(tmp_path / "palace")
    root = tmp_path / "sessions"
    garbage = _write_rollout(
        root, "garbage", ["this is not json at all", "%%% neither is this line %%%"]
    )

    with caplog.at_level(logging.WARNING, logger="mempalace.codex_follower"):
        stats = scan_once(palace, str(root))

    assert stats["files_unparseable"] == 1
    assert stats["drawers_added"] == 0
    # Never degraded to raw-text chunking: the palace holds nothing at all.
    assert (get_collection(palace, create=True).get(include=[]).get("ids") or []) == []
    warned = [r for r in caplog.records if "unparseable rollout" in r.getMessage()]
    assert warned and str(garbage) in warned[0].getMessage()


def test_recognized_but_turnless_rollout_is_not_unparseable(tmp_path):
    palace = str(tmp_path / "palace")
    root = tmp_path / "sessions"
    _write_rollout(root, "metaonly", [_meta()])
    stats = scan_once(palace, str(root))
    assert stats["files_unparseable"] == 0


def test_empty_rollout_is_not_unparseable(tmp_path):
    palace = str(tmp_path / "palace")
    root = tmp_path / "sessions"
    day = root / "2026" / "08" / "31"
    day.mkdir(parents=True, exist_ok=True)
    (day / "rollout-empty.jsonl").write_text("", encoding="utf-8")
    stats = scan_once(palace, str(root))
    assert stats["files_unparseable"] == 0


# ── deterministic ids across resume and compaction ───────────────────


def test_resume_replay_with_native_ids_adds_nothing_new(tmp_path):
    """A resumed session re-emits earlier items (same thread id, same item
    ids) in a fresh rollout file. Deterministic (thread, item) drawer ids
    make the replay an upsert of existing rows — only the genuinely new
    turn lands."""
    from mempalace.palace import get_collection

    palace = str(tmp_path / "palace")
    root = tmp_path / "sessions"
    _write_rollout(
        root,
        "orig",
        [
            _meta(),
            _item("user", "u1", "original question", 1),
            _item("assistant", "a1", "answer", 2),
        ],
    )
    assert scan_once(palace, str(root))["drawers_added"] == 2

    _write_rollout(
        root,
        "resumed",
        [
            _meta(),
            _item("user", "u1", "original question", 1),
            _item("assistant", "a1", "answer", 2),
            _item("user", "u2", "follow-up after resume", 3),
        ],
    )
    stats = scan_once(palace, str(root))
    assert stats["drawers_added"] == 1, stats
    assert scan_once(palace, str(root))["drawers_added"] == 0

    ids = get_collection(palace, create=False).get(include=[]).get("ids") or []
    assert len(ids) == 3


def test_compaction_and_continuation_stay_idempotent(tmp_path):
    palace = str(tmp_path / "palace")
    root = tmp_path / "sessions"
    lines = [
        _meta(),
        _item("user", "u1", "before compaction", 1),
        _item("assistant", "a1", "ok", 2),
    ]
    path = _write_rollout(root, "compact", lines)
    assert scan_once(palace, str(root))["drawers_added"] == 2

    compacted = json.dumps(
        {
            "type": "compacted",
            "ordinal": 3,
            "payload": {
                "replacement_history": [
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": "before compaction"}],
                    }
                ]
            },
        }
    )
    path.write_text(
        "\n".join(lines + [compacted, _item("user", "u2", "after compaction", 4)]) + "\n",
        encoding="utf-8",
    )
    assert scan_once(palace, str(root))["drawers_added"] == 1
    assert scan_once(palace, str(root))["drawers_added"] == 0


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


def test_scan_once_redacts_secrets_before_write(tmp_path):
    """A rollout quoting a live key must reach the palace redacted — the
    follower's direct-upsert path is outside the adapter choke point and
    needs its own scrub (the 2026-09-01 audit found 77 keys stored verbatim
    by the previous deployment)."""
    root = tmp_path / "sessions"
    palace = str(tmp_path / "palace")
    secret = "sk-ant-" + "z9Y8" * 8
    _write_rollout(
        root,
        "leak",
        [
            _meta(),
            _item("user", "u1", f"my key is {secret} please use it", 1),
            _item("assistant", "a1", "never paste keys in chat", 2),
        ],
    )
    stats = scan_once(palace, str(root), backfill=True)
    assert stats["secrets_redacted"] == 1
    from mempalace.palace import get_collection

    col = get_collection(palace, create=True)
    docs = col.get(include=["documents"])["documents"]
    joined = "\n".join(docs)
    assert secret not in joined
    assert "[redacted:anthropic_key]" in joined
