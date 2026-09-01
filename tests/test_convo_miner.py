import json
import os
import tempfile
import shutil
import time
from pathlib import Path

import chromadb
import pytest

from mempalace.convo_miner import (
    _is_ai_tool_path,
    _register_file,
    _resolve_wing,
    mine_convos,
)
from mempalace.palace import (
    NORMALIZE_VERSION,
    MineAlreadyRunning,
    file_already_mined,
    prefetch_mined_set,
)


def test_convo_mining():
    tmpdir = tempfile.mkdtemp()
    with open(os.path.join(tmpdir, "chat.txt"), "w") as f:
        f.write(
            "> What is memory?\nMemory is persistence.\n\n> Why does it matter?\nIt enables continuity.\n\n> How do we build it?\nWith structured storage.\n"
        )

    palace_path = os.path.join(tmpdir, "palace")
    mine_convos(tmpdir, palace_path, wing="test_convos")

    client = chromadb.PersistentClient(path=palace_path)
    col = client.get_collection("mempalace_drawers")
    assert col.count() >= 2

    # Verify search works
    results = col.query(query_texts=["memory persistence"], n_results=1)
    assert len(results["documents"][0]) > 0

    shutil.rmtree(tmpdir, ignore_errors=True)


def test_mine_convos_end_to_end_redacts_secret_and_reports_summary(monkeypatch, capsys):
    """Full mine_convos() pipeline: a secret pasted mid-transcript must not
    reach the palace verbatim, and the summary must report the redaction
    rather than scrubbing silently."""
    monkeypatch.setenv("MEMPALACE_REDACTION_ENABLED", "true")

    fake_key = "sk-ant-api03-AbCdEfGh12345678901234567890ijklMNOPqrstUVWXyz9876"
    tmpdir = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmpdir, "chat.txt"), "w") as f:
            f.write(
                "> Can you check my key?\n"
                f"Sure, I see it: {fake_key} -- looks well-formed.\n\n"
                "> Thanks, what's next?\n"
                "Rotate it once you've confirmed it works.\n\n"
                "> Will do, anything else?\n"
                "That covers it for now.\n"
            )

        palace_path = os.path.join(tmpdir, "palace")
        mine_convos(tmpdir, palace_path, wing="test_convos_redact")

        out = capsys.readouterr().out
        assert "Secrets redacted:" in out
        assert "anthropic_key:" in out

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_drawers")
        stored = col.get(include=["documents"])["documents"]
        assert not any(fake_key in doc for doc in stored)
        assert any("[redacted:anthropic_key]" in doc for doc in stored)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_mine_convos_does_not_reprocess_short_files(capsys):
    """Short files are FILED verbatim (short content is content, not
    noise — 2026-09-01) and still skipped on re-run via the stored mtime."""
    tmpdir = tempfile.mkdtemp()
    try:
        # A very short file — used to be dropped below MIN_CHUNK_SIZE,
        # now filed as a real drawer.
        with open(os.path.join(tmpdir, "tiny.txt"), "w") as f:
            f.write("hi")

        palace_path = os.path.join(tmpdir, "palace")

        # First run -- file is processed and filed
        mine_convos(tmpdir, palace_path, wing="test")
        capsys.readouterr()  # drain output

        # Verify it was filed (resolve path -- macOS /var -> /private/var)
        resolved_file = str(Path(tmpdir).resolve() / "tiny.txt")
        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_drawers")
        assert file_already_mined(col, resolved_file)
        rows = col.get(where={"source_file": resolved_file}, include=["documents", "metadatas"])
        real = [
            doc
            for doc, meta in zip(rows["documents"], rows["metadatas"])
            if (meta or {}).get("ingest_mode") != "registry"
        ]
        assert real == ["hi"], "short content must be filed verbatim, not dropped"

        # Second run -- file should be skipped
        mine_convos(tmpdir, palace_path, wing="test")
        out2 = capsys.readouterr().out
        assert "Files skipped (already filed): 1" in out2
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_mine_convos_does_not_reprocess_empty_chunk_files(capsys):
    """Files that normalize but produce 0 exchange chunks get a sentinel."""
    tmpdir = tempfile.mkdtemp()
    try:
        # Content long enough to pass MIN_CHUNK_SIZE but with no exchange markers
        # (no "> " lines), so chunk_exchanges returns []
        with open(os.path.join(tmpdir, "no_exchanges.txt"), "w") as f:
            f.write("This is a plain paragraph without any exchange markers. " * 5)

        palace_path = os.path.join(tmpdir, "palace")

        mine_convos(tmpdir, palace_path, wing="test")
        mine_convos(tmpdir, palace_path, wing="test")
        out2 = capsys.readouterr().out
        assert "Files skipped (already filed): 1" in out2
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_mine_convos_allows_general_after_exchange(capsys):
    """A transcript mined as exchange can later be mined as general memories."""
    tmpdir = tempfile.mkdtemp()
    try:
        convo_path = Path(tmpdir) / "chat.txt"
        convo_path.write_text(
            "> What did we decide?\n"
            "We decided to use SQLite because it keeps the local setup simple.\n\n"
            "> What broke?\n"
            "The search failed because the old index was stale, and the fix was rebuild.\n"
        )
        palace_path = os.path.join(tmpdir, "palace")

        mine_convos(tmpdir, palace_path, wing="test", extract_mode="exchange")
        capsys.readouterr()
        mine_convos(tmpdir, palace_path, wing="test", extract_mode="general")
        out = capsys.readouterr().out

        assert "Files skipped (already filed): 0" in out

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_drawers")
        resolved = str(Path(tmpdir).resolve() / "chat.txt")
        rows = col.get(where={"source_file": resolved}, include=["metadatas"])
        modes = {meta.get("extract_mode") for meta in rows["metadatas"]}
        assert {"exchange", "general"} <= modes
        assert any(drawer_id.startswith("drawer_test_decision_") for drawer_id in rows["ids"])
        del col, client
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_mine_convos_rebuilds_stale_drawers_after_schema_bump(capsys):
    """When stored drawers have an older normalize_version, the next mine
    silently purges them and refiles — no manual erase required.

    This is what makes the strip_noise upgrade apply to existing corpora:
    users just run `mempalace mine` again and old noise-filled drawers get
    replaced with clean ones."""
    from mempalace.palace import NORMALIZE_VERSION

    tmpdir = tempfile.mkdtemp()
    try:
        convo_path = Path(tmpdir) / "chat.txt"
        convo_path.write_text(
            "> What is memory?\nMemory is persistence.\n\n"
            "> Why does it matter?\nIt enables continuity.\n\n"
            "> How do we build it?\nWith structured storage.\n"
        )
        palace_path = os.path.join(tmpdir, "palace")

        # First mine — stamps drawers with NORMALIZE_VERSION
        mine_convos(tmpdir, palace_path, wing="test")
        capsys.readouterr()

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_drawers")
        resolved = str(Path(tmpdir).resolve() / "chat.txt")
        first_pass = col.get(where={"source_file": resolved})
        first_ids = set(first_pass["ids"])
        assert first_ids, "first mine should produce drawers"
        for meta in first_pass["metadatas"]:
            assert meta.get("normalize_version") == NORMALIZE_VERSION

        # Simulate pre-v2 drawers: rewrite metadata to an older version,
        # and replace content with "noise" so we can see it get cleaned up.
        stale_metas = []
        for meta in first_pass["metadatas"]:
            stale = dict(meta)
            stale["normalize_version"] = 1
            stale_metas.append(stale)
        col.update(
            ids=list(first_pass["ids"]),
            documents=["STALE NOISE"] * len(first_pass["ids"]),
            metadatas=stale_metas,
        )
        # Add an extra orphan drawer that should also be purged.
        col.add(
            ids=["orphan_drawer"],
            documents=["OLD ORPHAN"],
            metadatas=[
                {
                    "wing": "test",
                    "room": "default",
                    "source_file": resolved,
                    "chunk_index": 999,
                    "normalize_version": 1,
                }
            ],
        )
        del col, client

        # Second mine — version gate should trigger rebuild
        mine_convos(tmpdir, palace_path, wing="test")
        out = capsys.readouterr().out
        assert "Files skipped (already filed): 0" in out, (
            "stale drawers should force a rebuild, not a skip"
        )

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_drawers")
        rebuilt = col.get(where={"source_file": resolved})
        # Orphan is gone
        assert "orphan_drawer" not in rebuilt["ids"]
        # No stale content survived
        assert all("STALE NOISE" not in d for d in rebuilt["documents"])
        assert all("OLD ORPHAN" not in d for d in rebuilt["documents"])
        # All rebuilt drawers carry the current version
        for meta in rebuilt["metadatas"]:
            assert meta.get("normalize_version") == NORMALIZE_VERSION
        del col, client
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _hold_palace_lock_in_child(palace_path, ready_flag, release_flag):
    """Acquire mine_palace_lock in a child process and hold until signalled.

    Cannot use threads because mine_palace_lock is intentionally re-entrant
    within a single thread (so ChromaCollection write methods can compose
    with miner.mine() without self-deadlock). The convos concurrency
    guarantee is across processes / threads, so the test has to mirror that.
    """
    import os as _os
    import time as _time

    from mempalace.palace import mine_palace_lock as _mpl

    with _mpl(palace_path):
        open(ready_flag, "w").close()
        for _ in range(500):
            if _os.path.exists(release_flag):
                return
            _time.sleep(0.01)


def test_mine_convos_refuses_concurrent_run_against_same_palace(tmp_path, monkeypatch):
    """A second `mine_convos` against a palace currently being mined must
    raise MineAlreadyRunning, not stack up as a waiter that drives parallel
    ChromaDB writes. Mirrors the guarantee already given by `miner.mine`
    (see test_palace_locks.py) for the convos code path.
    """
    import multiprocessing
    import time

    monkeypatch.setenv("HOME", str(tmp_path))
    convo_dir = tmp_path / "convos"
    convo_dir.mkdir()
    (convo_dir / "chat.txt").write_text("> q1\nshort answer.\n\n> q2\nanother short answer.\n")
    palace_path = str(tmp_path / "palace")
    ready_flag = str(tmp_path / "ready")
    release_flag = str(tmp_path / "release")

    ctx = multiprocessing.get_context("spawn")
    holder = ctx.Process(
        target=_hold_palace_lock_in_child,
        args=(palace_path, ready_flag, release_flag),
    )
    holder.start()
    try:
        # Wait for the child to actually hold the lock before we attempt
        # to acquire from this process.
        for _ in range(500):
            if os.path.exists(ready_flag):
                break
            time.sleep(0.01)
        assert os.path.exists(ready_flag), "child never acquired palace lock"

        with pytest.raises(MineAlreadyRunning):
            mine_convos(str(convo_dir), palace_path, wing="test")
    finally:
        open(release_flag, "w").close()
        holder.join(timeout=10)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5)


def test_mine_convos_dry_run_bypasses_palace_lock(tmp_path, monkeypatch):
    """Dry-run never writes to the palace, so it must coexist with a live
    mine instead of being blocked by the per-palace flock.
    """
    import multiprocessing
    import time

    monkeypatch.setenv("HOME", str(tmp_path))
    convo_dir = tmp_path / "convos"
    convo_dir.mkdir()
    (convo_dir / "chat.txt").write_text("> q1\nshort answer.\n\n> q2\nanother short answer.\n")
    palace_path = str(tmp_path / "palace")
    ready_flag = str(tmp_path / "ready_dry")
    release_flag = str(tmp_path / "release_dry")

    ctx = multiprocessing.get_context("spawn")
    holder = ctx.Process(
        target=_hold_palace_lock_in_child,
        args=(palace_path, ready_flag, release_flag),
    )
    holder.start()
    try:
        for _ in range(500):
            if os.path.exists(ready_flag):
                break
            time.sleep(0.01)
        assert os.path.exists(ready_flag), "child never acquired palace lock"

        # Must not raise — dry-run skips the lock entirely.
        mine_convos(str(convo_dir), palace_path, wing="test", dry_run=True)
    finally:
        open(release_flag, "w").close()
        holder.join(timeout=10)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5)


# ── _is_ai_tool_path / _resolve_wing — wing_api auto-routing ───────────
#
# When a user runs `mempalace mine --mode convos` against a directory
# inside a known AI-tool storage path (Claude Code's
# ~/.claude/projects/, OpenAI Codex's ~/.codex/, Google Gemini CLI's
# ~/.gemini/), the wing auto-defaults to "wing_api" rather than the
# directory basename. This keeps API-sourced conversations grouped
# under a single dedicated wing for visibility and privacy isolation.
#
# Explicit user-passed --wing always wins. Unrelated directories use
# the existing basename fallback unchanged.


def test_is_ai_tool_path_claude_projects_subdir(tmp_path):
    """A subdirectory inside ~/.claude/projects/ is an AI tool path."""
    target = tmp_path / ".claude" / "projects" / "-Users-test-myapp"
    target.mkdir(parents=True)
    assert _is_ai_tool_path(target) is True


def test_is_ai_tool_path_claude_projects_root(tmp_path):
    """The ~/.claude/projects/ directory itself is an AI tool path."""
    target = tmp_path / ".claude" / "projects"
    target.mkdir(parents=True)
    assert _is_ai_tool_path(target) is True


def test_is_ai_tool_path_codex_root(tmp_path):
    target = tmp_path / ".codex"
    target.mkdir()
    assert _is_ai_tool_path(target) is True


def test_is_ai_tool_path_codex_sessions(tmp_path):
    """Codex stores sessions under ~/.codex/sessions/YYYY/MM/DD/."""
    target = tmp_path / ".codex" / "sessions" / "2026" / "04" / "26"
    target.mkdir(parents=True)
    assert _is_ai_tool_path(target) is True


def test_is_ai_tool_path_gemini_root(tmp_path):
    target = tmp_path / ".gemini"
    target.mkdir()
    assert _is_ai_tool_path(target) is True


def test_is_ai_tool_path_gemini_chats(tmp_path):
    """Gemini stores sessions under ~/.gemini/tmp/<hash>/chats/."""
    target = tmp_path / ".gemini" / "tmp" / "abc123" / "chats"
    target.mkdir(parents=True)
    assert _is_ai_tool_path(target) is True


def test_is_ai_tool_path_dotclaude_without_projects_not_matched(tmp_path):
    """`.claude/` alone (without `/projects`) is the settings dir, not a
    conversation source — it MUST NOT auto-route to wing_api."""
    target = tmp_path / ".claude"
    target.mkdir()
    assert _is_ai_tool_path(target) is False


def test_is_ai_tool_path_unrelated_directory(tmp_path):
    target = tmp_path / "Documents" / "myproject"
    target.mkdir(parents=True)
    assert _is_ai_tool_path(target) is False


def test_is_ai_tool_path_substring_no_false_positive(tmp_path):
    """A directory NAMED like `.gemini-backup` or `.codex-archive` is NOT
    a real AI tool path. We use exact-segment match, not substring."""
    a = tmp_path / ".gemini-backup"
    a.mkdir()
    b = tmp_path / ".codex-archive"
    b.mkdir()
    assert _is_ai_tool_path(a) is False
    assert _is_ai_tool_path(b) is False


def test_resolve_wing_explicit_wins_over_auto_detection(tmp_path):
    """User-passed --wing always wins, even on an AI tool path."""
    target = tmp_path / ".claude" / "projects" / "-Users-x"
    target.mkdir(parents=True)
    assert _resolve_wing(target, wing="my_custom_wing") == "my_custom_wing"


def test_resolve_wing_claude_projects_auto_routes_to_wing_api(tmp_path):
    target = tmp_path / ".claude" / "projects" / "-Users-test-myapp"
    target.mkdir(parents=True)
    assert _resolve_wing(target, wing=None) == "wing_api"


def test_resolve_wing_codex_auto_routes_to_wing_api(tmp_path):
    target = tmp_path / ".codex" / "sessions" / "2026"
    target.mkdir(parents=True)
    assert _resolve_wing(target, wing=None) == "wing_api"


def test_resolve_wing_gemini_auto_routes_to_wing_api(tmp_path):
    target = tmp_path / ".gemini" / "tmp" / "abc" / "chats"
    target.mkdir(parents=True)
    assert _resolve_wing(target, wing=None) == "wing_api"


def test_resolve_wing_unrelated_dir_uses_basename_fallback(tmp_path):
    """Existing behavior preserved: arbitrary directories use the
    sanitized basename as the wing."""
    target = tmp_path / "MyProject Folder"
    target.mkdir()
    # Spaces become underscores, hyphens become underscores, lowercased.
    assert _resolve_wing(target, wing=None) == "myproject_folder"


def test_resolve_wing_empty_string_treated_as_no_wing(tmp_path):
    """An empty string for wing should behave like None — fall through to
    auto-detection / basename. Mirrors the original `if not wing:` guard."""
    target = tmp_path / ".gemini" / "tmp"
    target.mkdir(parents=True)
    assert _resolve_wing(target, wing="") == "wing_api"


def test_mine_convos_limit_skips_already_mined(capsys):
    """--limit N counts only new work, not already-mined skips (#1535)."""
    tmpdir = tempfile.mkdtemp()
    try:
        convo_text = (
            "> What is topic {i}?\n"
            "Topic {i} is about something important and interesting enough "
            "to produce at least one exchange chunk for the test.\n\n"
            "> Tell me more about topic {i}.\n"
            "Sure, topic {i} has many facets worth exploring in detail.\n"
        )
        for i in range(4):
            with open(os.path.join(tmpdir, f"chat_{i}.txt"), "w") as f:
                f.write(convo_text.format(i=i))

        palace_path = os.path.join(tmpdir, "palace")

        mine_convos(tmpdir, palace_path, wing="test")
        capsys.readouterr()

        for i in range(4, 7):
            with open(os.path.join(tmpdir, f"chat_{i}.txt"), "w") as f:
                f.write(convo_text.format(i=i))

        mine_convos(tmpdir, palace_path, wing="test", limit=2)
        out = capsys.readouterr().out

        assert "Files processed: 2" in out
        assert "Drawers filed:" in out
        for line in out.split("\n"):
            if "Drawers filed:" in line:
                filed = int(line.split(":")[1].strip())
                assert filed > 0, f"limit=2 should mine new files, got {filed}"
                break
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── mtime-aware re-mining ────────────────────────────────────────────
#
# Conversation transcripts are NOT immutable: a Claude Code session keeps
# appending to its own file while active, and /compact or /clear can
# rewrite one in place. These tests cover the fix -- convo mining used to
# treat "we've seen this source_file before" as sufficient to skip it
# forever (transcripts were assumed immutable), silently missing content
# appended after the first mine.


def test_mine_convos_reprocesses_when_file_grows(capsys):
    """A session file that grows after being mined must be picked up on
    the next mine, not skipped forever."""
    tmpdir = tempfile.mkdtemp()
    try:
        convo_path = Path(tmpdir) / "session.txt"
        convo_path.write_text(
            "> What is the plan?\nStart with the schema, then the API.\n\n"
            "> Any risks?\nMigration ordering is the main one.\n"
        )
        palace_path = os.path.join(tmpdir, "palace")

        mine_convos(tmpdir, palace_path, wing="test")
        capsys.readouterr()

        # Simulate the session being extended: real content added, mtime
        # bumped forward (avoids same-second mtime resolution flakiness).
        convo_path.write_text(
            "> What is the plan?\nStart with the schema, then the API.\n\n"
            "> Any risks?\nMigration ordering is the main one.\n\n"
            "> UNIQUE_GROWN_SESSION_MARKER, did we resolve it?\n"
            "Yes, resolved by locking the migration order explicitly.\n"
        )
        future = time.time() + 60
        os.utime(convo_path, (future, future))

        mine_convos(tmpdir, palace_path, wing="test")
        out = capsys.readouterr().out
        assert "Files skipped (already filed): 1" not in out

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_drawers")
        docs = col.get(include=["documents"])["documents"]
        assert any("UNIQUE_GROWN_SESSION_MARKER" in d for d in docs), (
            "grown session content was not picked up on re-mine"
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_mine_convos_unchanged_file_still_skipped(capsys):
    """A file whose content and mtime are unchanged must still be skipped
    -- the mtime check must not defeat the existing skip-on-unchanged
    optimization."""
    tmpdir = tempfile.mkdtemp()
    try:
        convo_path = Path(tmpdir) / "session.txt"
        convo_path.write_text(
            "> What is the plan?\nStart with the schema, then the API.\n\n"
            "> Any risks?\nMigration ordering is the main one.\n"
        )
        palace_path = os.path.join(tmpdir, "palace")

        mine_convos(tmpdir, palace_path, wing="test")
        capsys.readouterr()

        mine_convos(tmpdir, palace_path, wing="test")
        out = capsys.readouterr().out
        assert "Files skipped (already filed): 1" in out
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_mine_convos_grown_file_purges_stale_drawers_not_additive(capsys):
    """Re-mining a grown file must not leave duplicate/stale drawers behind
    -- purge-then-insert, not additive accumulation. Checks content
    directly (a drawer count comparison is fragile: ChromaDB collections
    can carry non-drawer bookkeeping rows unrelated to this behavior)."""
    tmpdir = tempfile.mkdtemp()
    try:
        convo_path = Path(tmpdir) / "session.txt"
        convo_path.write_text(
            "> What is the plan?\nUNIQUE_ORIGINAL_EXCHANGE_MARKER here.\n\n"
            "> Any risks?\nMigration ordering is the main one.\n"
        )
        palace_path = os.path.join(tmpdir, "palace")

        mine_convos(tmpdir, palace_path, wing="test")
        capsys.readouterr()

        convo_path.write_text(
            "> What is the plan?\nUNIQUE_ORIGINAL_EXCHANGE_MARKER here.\n\n"
            "> Any risks?\nMigration ordering is the main one.\n\n"
            "> One more exchange?\nUNIQUE_NEW_EXCHANGE_MARKER here.\n"
        )
        future = time.time() + 60
        os.utime(convo_path, (future, future))
        mine_convos(tmpdir, palace_path, wing="test")
        capsys.readouterr()

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_drawers")
        docs = col.get(include=["documents"])["documents"]

        original_hits = sum(1 for d in docs if "UNIQUE_ORIGINAL_EXCHANGE_MARKER" in d)
        new_hits = sum(1 for d in docs if "UNIQUE_NEW_EXCHANGE_MARKER" in d)
        assert original_hits == 1, (
            f"original exchange duplicated across re-mine: {original_hits} copies"
        )
        assert new_hits == 1, f"new exchange should appear exactly once, got {new_hits}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_mine_convos_skips_same_content_under_new_filename(capsys):
    """Re-exporting the same conversation from Claude/ChatGPT under a new
    filename (fresh export bundle, regenerated slug, etc.) must not create
    a duplicate set of drawers -- only the exact-new content should file."""
    tmpdir = tempfile.mkdtemp()
    try:
        transcript = (
            "> What is the plan?\nStart with the schema, then the API.\n\n"
            "> Any risks?\nMigration ordering is the main one.\n"
        )
        (Path(tmpdir) / "export_2026-01-01.txt").write_text(transcript)
        palace_path = os.path.join(tmpdir, "palace")
        mine_convos(tmpdir, palace_path, wing="test")

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_drawers")
        count_after_first = col.count()
        assert count_after_first >= 2

        # Simulate a later export: the same conversation lands under a new
        # filename, alongside one genuinely new conversation.
        (Path(tmpdir) / "export_2026-02-01.txt").write_text(transcript)
        (Path(tmpdir) / "export_2026-02-01_new.txt").write_text(
            "> What's next?\nUNIQUE_SECOND_EXPORT_MARKER covers the new work.\n"
        )
        mine_convos(tmpdir, palace_path, wing="test")
        out = capsys.readouterr().out
        assert "duplicate of export_2026-01-01.txt" in out

        col = client.get_collection("mempalace_drawers")
        docs = col.get(include=["documents"])["documents"]
        dup_hits = sum(1 for d in docs if "Migration ordering is the main one" in d)
        assert dup_hits == 1, f"duplicate transcript re-filed: {dup_hits} copies"
        assert any("UNIQUE_SECOND_EXPORT_MARKER" in d for d in docs)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _privacy_export_bundle(conversations):
    """Build a Claude.ai privacy-export-shaped JSON payload: an array of
    conversation objects, each with its own chat_messages list."""
    return [
        {
            "chat_messages": [
                {"sender": "human", "text": turn}
                if i % 2 == 0
                else {"sender": "assistant", "text": turn}
                for i, turn in enumerate(turns)
            ]
        }
        for turns in conversations
    ]


def test_mine_convos_skips_same_conversation_within_re_exported_bundle(capsys):
    """A Claude.ai privacy export bundles every conversation into one JSON
    file. Re-exporting that bundle under a new filename with one additional
    conversation must not re-file the conversations that didn't change --
    hashing the whole bundle would change the file-level hash the moment
    any conversation is added, hiding the ones that are still duplicates.
    """
    tmpdir = tempfile.mkdtemp()
    try:
        convo_a = ["What is the plan?", "CONVO_A_MARKER: start with the schema."]
        convo_b = ["Any risks?", "CONVO_B_MARKER: migration ordering is the main one."]
        convo_c = ["What's next?", "CONVO_C_MARKER: covers the new work."]

        bundle1 = _privacy_export_bundle([convo_a, convo_b])
        (Path(tmpdir) / "export_2026-01-01.json").write_text(json.dumps(bundle1))

        palace_path = os.path.join(tmpdir, "palace")
        mine_convos(tmpdir, palace_path, wing="test")

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_drawers")
        docs_after_first = col.get(include=["documents"])["documents"]
        assert any("CONVO_A_MARKER" in d for d in docs_after_first)
        assert any("CONVO_B_MARKER" in d for d in docs_after_first)

        # Re-export: same two conversations plus one genuinely new one, all
        # under a fresh filename (as a real re-export from Claude would do).
        bundle2 = _privacy_export_bundle([convo_a, convo_b, convo_c])
        (Path(tmpdir) / "export_2026-02-01.json").write_text(json.dumps(bundle2))

        mine_convos(tmpdir, palace_path, wing="test")

        col = client.get_collection("mempalace_drawers")
        docs = col.get(include=["documents"])["documents"]
        a_hits = sum(1 for d in docs if "CONVO_A_MARKER" in d)
        b_hits = sum(1 for d in docs if "CONVO_B_MARKER" in d)
        c_hits = sum(1 for d in docs if "CONVO_C_MARKER" in d)
        assert a_hits == 1, f"conversation A re-filed from the updated bundle: {a_hits} copies"
        assert b_hits == 1, f"conversation B re-filed from the updated bundle: {b_hits} copies"
        assert c_hits >= 1, "new conversation C was not filed at all"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_content_dedup_is_scoped_per_wing():
    """Mining the same transcript content into a second wing must file real
    drawers there, not just the registry sentinel -- the content-hash map
    is a dedup signal within a wing, not a cross-wing "already have this
    content anywhere" gate.
    """
    tmpdir = tempfile.mkdtemp()
    try:
        transcript = (
            "> What is the plan?\nStart with the schema, then the API.\n\n"
            "> Any risks?\nMigration ordering is the main one.\n"
        )
        dir_a = Path(tmpdir) / "wing_a_src"
        dir_b = Path(tmpdir) / "wing_b_src"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "session.txt").write_text(transcript)
        (dir_b / "session.txt").write_text(transcript)

        palace_path = os.path.join(tmpdir, "palace")
        mine_convos(str(dir_a), palace_path, wing="wing_a")
        mine_convos(str(dir_b), palace_path, wing="wing_b")

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_drawers")
        wing_b_docs = col.get(where={"wing": "wing_b"}, include=["documents", "metadatas"])
        real_drawers = [
            d
            for d, m in zip(wing_b_docs["documents"], wing_b_docs["metadatas"])
            if m.get("room") != "_registry"
        ]
        assert real_drawers, (
            "wing_b holds only the registry sentinel -- content dedup leaked across wings"
        )
        assert any("Migration ordering is the main one" in d for d in real_drawers)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_prefetch_mined_set_returns_stored_mtime():
    """prefetch_mined_set's dict carries each source_file's stored mtime,
    not just membership."""
    tmpdir = tempfile.mkdtemp()
    try:
        convo_path = Path(tmpdir) / "session.txt"
        convo_path.write_text(
            "> What is the plan?\nStart with the schema, then the API.\n\n"
            "> Any risks?\nMigration ordering is the main one.\n"
        )
        palace_path = os.path.join(tmpdir, "palace")
        mine_convos(tmpdir, palace_path, wing="test")

        resolved_file = str(convo_path.resolve())
        actual_mtime = os.path.getmtime(resolved_file)

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_drawers")
        mined = prefetch_mined_set(col, extract_mode="exchange")

        assert resolved_file in mined
        assert mined[resolved_file] is not None
        assert abs(mined[resolved_file] - actual_mtime) < 0.001
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_prefetch_mined_set_none_for_drawer_without_stored_mtime():
    """A drawer written before source_mtime existed (or with getmtime
    failure at write time) must surface as None, not be silently absent --
    None must be treated as stale by callers, not as 'unknown, assume ok'."""
    tmpdir = tempfile.mkdtemp()
    try:
        palace_path = os.path.join(tmpdir, "palace")
        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_or_create_collection("mempalace_drawers")
        col.upsert(
            ids=["drawer_legacy_1"],
            documents=["legacy content with no source_mtime field"],
            metadatas=[
                {
                    "wing": "test",
                    "room": "general",
                    "source_file": "/fake/legacy/file.txt",
                    "chunk_index": 0,
                    "extract_mode": "exchange",
                    "normalize_version": 999,  # force >= current version
                }
            ],
        )
        mined = prefetch_mined_set(col, extract_mode="exchange")
        assert "/fake/legacy/file.txt" in mined
        assert mined["/fake/legacy/file.txt"] is None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_prefetch_mined_set_omits_incomplete_chunk_total_group():
    """Mid-file partials with chunk_total must not bulk-skip the source (#2183)."""
    tmpdir = tempfile.mkdtemp()
    try:
        palace_path = os.path.join(tmpdir, "palace")
        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_or_create_collection("mempalace_drawers")
        mtime = 1_700_000_000.0
        source = "/fake/session.jsonl"
        # Only 2 of 3 expected chunks landed before a crash.
        col.upsert(
            ids=["d0", "d1"],
            documents=["chunk 0", "chunk 1"],
            metadatas=[
                {
                    "wing": "test",
                    "room": "general",
                    "source_file": source,
                    "chunk_index": 0,
                    "extract_mode": "exchange",
                    "normalize_version": NORMALIZE_VERSION,
                    "source_mtime": mtime,
                    "chunk_total": 3,
                },
                {
                    "wing": "test",
                    "room": "general",
                    "source_file": source,
                    "chunk_index": 1,
                    "extract_mode": "exchange",
                    "normalize_version": NORMALIZE_VERSION,
                    "source_mtime": mtime,
                    "chunk_total": 3,
                },
            ],
        )
        mined = prefetch_mined_set(col, extract_mode="exchange")
        assert source not in mined, (
            "prefetch_mined_set treated 2/3 chunks as fully filed — the bulk "
            "skip path would permanently strand the missing exchange (#2183)"
        )

        col.upsert(
            ids=["d2"],
            documents=["chunk 2"],
            metadatas=[
                {
                    "wing": "test",
                    "room": "general",
                    "source_file": source,
                    "chunk_index": 2,
                    "extract_mode": "exchange",
                    "normalize_version": NORMALIZE_VERSION,
                    "source_mtime": mtime,
                    "chunk_total": 3,
                }
            ],
        )
        mined = prefetch_mined_set(col, extract_mode="exchange")
        assert source in mined
        assert abs(mined[source] - mtime) < 0.001
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_mine_convos_reprocesses_legacy_drawer_without_stored_mtime(capsys):
    """A file mined before source_mtime was tracked (simulated: drawer
    written directly, no source_mtime field) must be re-mined on the next
    run, not skipped forever -- this is the one-time backfill behavior."""
    tmpdir = tempfile.mkdtemp()
    try:
        convo_path = Path(tmpdir) / "session.txt"
        convo_path.write_text(
            "> What is the plan?\nUNIQUE_LEGACY_BACKFILL_MARKER here.\n\n"
            "> Any risks?\nMigration ordering is the main one.\n"
        )
        resolved_file = str(convo_path.resolve())
        palace_path = os.path.join(tmpdir, "palace")

        # Simulate a pre-existing drawer from before source_mtime existed.
        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_or_create_collection("mempalace_drawers")
        from mempalace.palace import NORMALIZE_VERSION

        col.upsert(
            ids=["drawer_legacy_session_1"],
            documents=["stale legacy content, no mtime field"],
            metadatas=[
                {
                    "wing": "test",
                    "room": "general",
                    "source_file": resolved_file,
                    "chunk_index": 0,
                    "extract_mode": "exchange",
                    "normalize_version": NORMALIZE_VERSION,
                }
            ],
        )
        del col, client

        mine_convos(tmpdir, palace_path, wing="test")
        out = capsys.readouterr().out
        assert "Files skipped (already filed): 1" not in out

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_drawers")
        docs = col.get(include=["documents"])["documents"]
        assert any("UNIQUE_LEGACY_BACKFILL_MARKER" in d for d in docs)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_register_file_sentinel_includes_source_mtime():
    """The 0-chunk sentinel must stamp source_mtime too, so a file that
    later grows past the min-chunk-size floor is detected as changed
    instead of being skipped forever by the sentinel."""
    tmpdir = tempfile.mkdtemp()
    try:
        tiny_file = Path(tmpdir) / "tiny.txt"
        tiny_file.write_text("hi")
        palace_path = os.path.join(tmpdir, "palace")
        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_or_create_collection("mempalace_drawers")

        _register_file(col, str(tiny_file), "test", "mempalace", "exchange")

        mined = prefetch_mined_set(col, extract_mode="exchange")
        assert str(tiny_file) in mined
        assert mined[str(tiny_file)] is not None
        assert abs(mined[str(tiny_file)] - os.path.getmtime(tiny_file)) < 0.001
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# file_conversation_exchange — canonical single-exchange write path
# ---------------------------------------------------------------------------


class _RecordingCollection:
    """Captures upsert kwargs without a real ChromaDB behind it."""

    def __init__(self):
        self.upserts = []

    def upsert(self, *, ids, documents, metadatas):
        self.upserts.append({"ids": ids, "documents": documents, "metadatas": metadatas})


def _exchange_kwargs(**overrides):
    kwargs = {
        "wing": "wing_dev",
        "room": "conversations",
        "text": "User: hi\n\nAssistant: hello",
        "source_file": "hermes-session:s1",
        "agent": "hermes",
    }
    kwargs.update(overrides)
    return kwargs


def test_file_conversation_exchange_extra_metadata_cannot_clobber_canonical():
    """The docstring promises extras are append-only — colliding keys lose.

    PR #1915 review: ``metadata.update(extra_metadata)`` let a caller
    silently overwrite ``wing`` / ``filed_at`` / etc.
    """
    from mempalace.convo_miner import file_conversation_exchange

    col = _RecordingCollection()
    file_conversation_exchange(
        col,
        **_exchange_kwargs(),
        extra_metadata={"wing": "wing_evil", "filed_at": "1970-01-01", "source": "hermes"},
    )
    meta = col.upserts[0]["metadatas"][0]
    assert meta["wing"] == "wing_dev"
    assert meta["filed_at"] != "1970-01-01"
    # Non-colliding extras still land.
    assert meta["source"] == "hermes"


def test_file_conversation_exchange_invalid_wing_falls_back_to_wing_general():
    """A bad configured wing must not drop the turn — verbatim first.

    Same validation the MCP write tools apply (sanitize_name), but with a
    wing_general fallback instead of an error: live filing losing turns
    over a config typo would violate the 100%-recall promise.
    """
    from mempalace.convo_miner import file_conversation_exchange

    col = _RecordingCollection()
    file_conversation_exchange(col, **_exchange_kwargs(wing="../escape"))
    meta = col.upserts[0]["metadatas"][0]
    assert meta["wing"] == "wing_general"
    assert col.upserts[0]["documents"] == ["User: hi\n\nAssistant: hello"]


def test_file_conversation_exchange_invalid_room_falls_back_to_conversations():
    from mempalace.convo_miner import file_conversation_exchange

    col = _RecordingCollection()
    file_conversation_exchange(col, **_exchange_kwargs(room="a/b"))
    meta = col.upserts[0]["metadatas"][0]
    assert meta["room"] == "conversations"


# ---------------------------------------------------------------------------
# file_conversation_exchange — ingest-time secret redaction wiring
# ---------------------------------------------------------------------------

_FAKE_ANTHROPIC_KEY = "sk-ant-api03-AbCdEfGh12345678901234567890ijklMNOPqrstUVWXyz9876"


def test_file_conversation_exchange_redacts_secret_before_storing(monkeypatch, caplog):
    """The secret must not reach ``collection.upsert`` — not in the stored
    document, and (since redaction runs before ID derivation and entity
    extraction) nowhere else downstream either."""
    import logging

    import mempalace.convo_miner as convo_miner_mod
    from mempalace.convo_miner import file_conversation_exchange

    monkeypatch.setattr(convo_miner_mod, "_REDACTION_ENABLED_CACHE", True)
    col = _RecordingCollection()
    text = f"User: here's my key\n\nAssistant: {_FAKE_ANTHROPIC_KEY} noted, thanks"

    with caplog.at_level(logging.INFO, logger="mempalace_mcp"):
        drawer_id = file_conversation_exchange(col, **_exchange_kwargs(text=text))

    assert drawer_id is not None
    stored_doc = col.upserts[0]["documents"][0]
    assert _FAKE_ANTHROPIC_KEY not in stored_doc
    assert "[redacted:anthropic_key]" in stored_doc
    # Redaction must never be silent.
    assert any("redacted" in rec.message for rec in caplog.records)


def test_file_conversation_exchange_redact_disabled_stores_verbatim(monkeypatch):
    import mempalace.convo_miner as convo_miner_mod
    from mempalace.convo_miner import file_conversation_exchange

    monkeypatch.setattr(convo_miner_mod, "_REDACTION_ENABLED_CACHE", False)
    col = _RecordingCollection()
    text = f"User: here's my key\n\nAssistant: {_FAKE_ANTHROPIC_KEY} noted, thanks"

    file_conversation_exchange(col, **_exchange_kwargs(text=text))

    assert _FAKE_ANTHROPIC_KEY in col.upserts[0]["documents"][0]


def test_file_conversation_exchange_plain_exchange_unaffected(monkeypatch):
    """Negative case: ordinary conversation text survives byte-identical."""
    import mempalace.convo_miner as convo_miner_mod
    from mempalace.convo_miner import file_conversation_exchange

    monkeypatch.setattr(convo_miner_mod, "_REDACTION_ENABLED_CACHE", True)
    col = _RecordingCollection()
    text = "User: hi\n\nAssistant: hello"

    file_conversation_exchange(col, **_exchange_kwargs(text=text))

    assert col.upserts[0]["documents"] == [text]


def _write_dry_run_transcript(path: Path) -> None:
    path.write_text(
        "> What is the plan?\n"
        "Start with the schema, then the API.\n\n"
        "> Are there any risks?\n"
        "Migration ordering is the main one.\n\n"
        "> What comes next?\n"
        "Run focused tests before the full suite.\n",
        encoding="utf-8",
    )


def test_mine_convos_dry_run_skips_unchanged_mined_file(
    tmp_path,
    capsys,
    monkeypatch,
):
    monkeypatch.setenv("HOME", str(tmp_path))

    convo_dir = tmp_path / "convos"
    convo_dir.mkdir()
    transcript = convo_dir / "session.txt"
    _write_dry_run_transcript(transcript)
    palace_path = str(tmp_path / "palace")

    mine_convos(
        str(convo_dir),
        palace_path,
        wing="original",
    )
    capsys.readouterr()

    mine_convos(
        str(convo_dir),
        palace_path,
        wing="target",
        dry_run=True,
    )
    output = capsys.readouterr().out

    assert "[DRY RUN] session.txt" not in output
    assert "Files processed: 0" in output
    assert "Files skipped (already filed): 1" in output
    assert "Drawers filed: 0" in output


def test_mine_convos_dry_run_keeps_modified_file_as_work(
    tmp_path,
    capsys,
    monkeypatch,
):
    monkeypatch.setenv("HOME", str(tmp_path))

    convo_dir = tmp_path / "convos"
    convo_dir.mkdir()
    transcript = convo_dir / "session.txt"
    _write_dry_run_transcript(transcript)
    palace_path = str(tmp_path / "palace")

    mine_convos(
        str(convo_dir),
        palace_path,
        wing="original",
    )
    capsys.readouterr()

    transcript.write_text(
        transcript.read_text(encoding="utf-8")
        + "\n> Did the plan change?\n"
        + "Yes, add a migration rollback test.\n",
        encoding="utf-8",
    )

    future = time.time() + 60
    os.utime(transcript, (future, future))

    mine_convos(
        str(convo_dir),
        palace_path,
        wing="target",
        dry_run=True,
    )
    output = capsys.readouterr().out

    assert "[DRY RUN] session.txt" in output
    assert "Files processed: 1" in output
    assert "Files skipped (already filed): 0" in output


def test_mine_convos_dry_run_missing_palace_does_not_create_it(
    tmp_path,
    capsys,
    monkeypatch,
):
    monkeypatch.setenv("HOME", str(tmp_path))

    convo_dir = tmp_path / "convos"
    convo_dir.mkdir()
    transcript = convo_dir / "session.txt"
    _write_dry_run_transcript(transcript)
    palace_path = tmp_path / "palace"

    mine_convos(
        str(convo_dir),
        str(palace_path),
        wing="target",
        dry_run=True,
    )
    output = capsys.readouterr().out

    assert "[DRY RUN] session.txt" in output
    assert "Files processed: 1" in output
    assert "Files skipped (already filed): 0" in output
    assert not palace_path.exists()


def test_mine_convos_dry_run_single_file_does_not_scan_siblings(
    tmp_path,
    capsys,
):
    selected = tmp_path / "selected.txt"
    sibling = tmp_path / "sibling.txt"

    selected.write_text(
        "> Which transcript should be mined?\n"
        "SELECTED_ONLY_MARKER belongs to the active transcript.\n\n"
        "> Should sibling files be included?\n"
        "No. Only the selected transcript should be scanned.\n",
        encoding="utf-8",
    )
    sibling.write_text(
        "> Should this sibling be mined?\n"
        "SIBLING_SHOULD_NOT_BE_MINED by the single-file invocation.\n\n"
        "> Is that important?\n"
        "Yes. It keeps hook-triggered mining narrowly scoped.\n",
        encoding="utf-8",
    )

    palace_path = tmp_path / "palace"

    mine_convos(
        str(selected),
        str(palace_path),
        wing="sessions",
        dry_run=True,
    )
    output = capsys.readouterr().out

    assert "Files:   1" in output
    assert "[DRY RUN] selected.txt" in output
    assert "sibling.txt" not in output
    assert not palace_path.exists()


def test_mine_convos_quarantines_unparseable_structured_files(tmp_path, capsys, caplog):
    """A .json/.jsonl file that looks like a chat export but is declined by
    every parser must be QUARANTINED — skipped, counted, and warned about —
    not paragraph-chunked into garbage drawers (2026-09-01)."""
    import logging

    # Structured JSON no chat parser recognizes.
    (tmp_path / "mystery.json").write_text(
        json.dumps({"records": [{"foo": "bar" * 50}, {"baz": [1, 2, 3]}]}),
        encoding="utf-8",
    )
    palace_path = tmp_path / "palace"

    with caplog.at_level(logging.WARNING, logger="mempalace_mcp"):
        mine_convos(str(tmp_path), str(palace_path), wing="test")
    captured = capsys.readouterr()

    assert "Files quarantined (unparseable structured): 1" in captured.out
    assert "Drawers filed: 0" in captured.out
    assert "QUARANTINE: mystery.json" in captured.err
    assert any(
        "quarantined" in rec.getMessage() and "mystery.json" in rec.getMessage()
        for rec in caplog.records
    ), "expected a warning log naming the quarantined file"

    # No drawers (and no sentinel) were written for the quarantined file.
    client = chromadb.PersistentClient(path=str(palace_path))
    col = client.get_collection("mempalace_drawers")
    resolved = str((tmp_path / "mystery.json").resolve())
    rows = col.get(where={"source_file": resolved})
    assert rows["ids"] == [], "quarantined file must not produce drawers"


def test_mine_convos_plain_txt_with_json_content_keeps_paragraph_path(tmp_path, capsys):
    """Quarantine applies only to structured .json/.jsonl files. A plain
    .txt file whose content merely starts with '{' keeps the existing
    paragraph path."""
    (tmp_path / "notes.txt").write_text(
        "{ this is a note that merely opens with a brace }\n\n"
        "It continues with an ordinary paragraph about the project plan.\n",
        encoding="utf-8",
    )
    palace_path = tmp_path / "palace"

    mine_convos(str(tmp_path), str(palace_path), wing="test")
    out = capsys.readouterr().out

    assert "Files quarantined" not in out
    client = chromadb.PersistentClient(path=str(palace_path))
    col = client.get_collection("mempalace_drawers")
    resolved = str((tmp_path / "notes.txt").resolve())
    rows = col.get(where={"source_file": resolved}, include=["documents", "metadatas"])
    real = [
        doc
        for doc, meta in zip(rows["documents"], rows["metadatas"])
        if (meta or {}).get("ingest_mode") != "registry"
    ]
    assert real, ".txt content must still be filed via the paragraph path"


def test_mine_convos_stamps_chunker_version(tmp_path, capsys):
    """Re-mined drawers must be distinguishable from old-chunker drawers via
    the chunker_version metadata marker."""
    from mempalace.convo_miner import CONVO_CHUNKER_VERSION

    (tmp_path / "chat.txt").write_text(
        "> What is memory?\nMemory is persistence.\n\n"
        "> Why?\nContinuity.\n\n"
        "> How?\nStructured storage.\n",
        encoding="utf-8",
    )
    palace_path = tmp_path / "palace"
    mine_convos(str(tmp_path), str(palace_path), wing="test")
    capsys.readouterr()

    client = chromadb.PersistentClient(path=str(palace_path))
    col = client.get_collection("mempalace_drawers")
    resolved = str((tmp_path / "chat.txt").resolve())
    rows = col.get(where={"source_file": resolved}, include=["metadatas"])
    assert rows["metadatas"], "expected drawers for the transcript"
    assert all(
        meta.get("chunker_version") == CONVO_CHUNKER_VERSION for meta in rows["metadatas"]
    ), "every convo drawer must carry the chunker_version marker"
