"""
test_identity_aliases.py — Tests for the identity.aliases / identity.canonical
config keys (docs/rfcs/005-agent-identity-routing.md) and the diary-read
alias merge that consumes them.

Covers:
- MempalaceConfig.identity_canonical / identity_aliases / aliases_for_identity
- tool_diary_read merging alias-filed entries into a canonical-name read,
  chronologically, without redirecting writes
- _diary_agent_for_harness honoring identity.canonical over its hardcoded
  per-harness defaults
"""

import json
import os

import chromadb
import pytest

from mempalace.config import MempalaceConfig


def _write_config(cfg_dir, palace_path, identity=None):
    payload = {"palace_path": palace_path}
    if identity is not None:
        payload["identity"] = identity
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, "config.json"), "w") as f:
        json.dump(payload, f)
    return MempalaceConfig(config_dir=cfg_dir)


class TestConfigIdentityProperties:
    def test_defaults_are_empty(self, tmp_path):
        cfg = _write_config(str(tmp_path / "cfgdir"), str(tmp_path / "palace"))
        assert cfg.identity_canonical is None
        assert cfg.identity_aliases == {}
        assert cfg.aliases_for_identity("mac-claude") == []

    def test_canonical_from_file(self, tmp_path):
        cfg = _write_config(
            str(tmp_path / "cfgdir"),
            str(tmp_path / "palace"),
            identity={"canonical": "Mac-Claude"},
        )
        assert cfg.identity_canonical == "mac-claude"  # lowercased

    def test_canonical_env_overrides_file(self, tmp_path, monkeypatch):
        cfg = _write_config(
            str(tmp_path / "cfgdir"),
            str(tmp_path / "palace"),
            identity={"canonical": "file-name"},
        )
        monkeypatch.setenv("MEMPALACE_IDENTITY_CANONICAL", "Env-Name")
        assert cfg.identity_canonical == "env-name"

    def test_aliases_from_file_normalized_lowercase(self, tmp_path):
        cfg = _write_config(
            str(tmp_path / "cfgdir"),
            str(tmp_path / "palace"),
            identity={"aliases": {"Mac-Claude": ["Claude", "claude_code"]}},
        )
        assert cfg.identity_aliases == {"mac-claude": ["claude", "claude_code"]}
        assert cfg.aliases_for_identity("MAC-CLAUDE") == ["claude", "claude_code"]

    def test_malformed_aliases_entries_are_skipped(self, tmp_path):
        cfg = _write_config(
            str(tmp_path / "cfgdir"),
            str(tmp_path / "palace"),
            identity={
                "aliases": {
                    "good": ["a", "b"],
                    "bad-not-a-list": "oops",
                    "": ["ignored-empty-key"],
                }
            },
        )
        assert cfg.identity_aliases == {"good": ["a", "b"]}


@pytest.fixture
def palace_collection(tmp_path):
    palace_path = str(tmp_path / "palace")
    os.makedirs(palace_path)
    client = chromadb.PersistentClient(path=palace_path)
    col = client.get_or_create_collection("mempalace_drawers", metadata={"hnsw:space": "cosine"})
    yield palace_path, col
    client.close()


class TestDiaryReadAliasMerge:
    def test_read_merges_alias_entries_chronologically(self, monkeypatch, palace_collection):
        from mempalace import mcp_server
        from mempalace.mcp_server import tool_diary_read

        palace_path, col = palace_collection
        cfg = _write_config(
            str(palace_path) + "-cfg",
            palace_path,
            identity={"aliases": {"mac-claude": ["claude", "claude_code"]}},
        )
        monkeypatch.setattr(mcp_server, "_config", cfg)
        monkeypatch.setattr(mcp_server, "_get_collection", lambda *a, **kw: col)

        col.add(
            ids=["d1", "d2", "d3"],
            documents=[
                "oldest under claude",
                "middle under claude_code",
                "newest under mac-claude",
            ],
            metadatas=[
                {
                    "wing": "wing_x",
                    "room": "diary",
                    "agent": "claude",
                    "topic": "t",
                    "filed_at": "2026-01-01T00:00:00",
                    "date": "2026-01-01",
                },
                {
                    "wing": "wing_x",
                    "room": "diary",
                    "agent": "claude_code",
                    "topic": "t",
                    "filed_at": "2026-01-02T00:00:00",
                    "date": "2026-01-02",
                },
                {
                    "wing": "wing_x",
                    "room": "diary",
                    "agent": "mac-claude",
                    "topic": "t",
                    "filed_at": "2026-01-03T00:00:00",
                    "date": "2026-01-03",
                },
            ],
        )

        result = tool_diary_read(agent_name="mac-claude")
        assert result["total"] == 3
        contents = [e["content"] for e in result["entries"]]
        assert contents == [
            "newest under mac-claude",
            "middle under claude_code",
            "oldest under claude",
        ]

    def test_read_under_bare_alias_is_unaffected(self, monkeypatch, palace_collection):
        """Reading directly by an alias name (not the canonical) does not pull
        in the canonical or sibling-alias entries -- aliasing only widens a
        canonical-name read, per the config's canonical -> [aliases] shape."""
        from mempalace import mcp_server
        from mempalace.mcp_server import tool_diary_read

        palace_path, col = palace_collection
        cfg = _write_config(
            str(palace_path) + "-cfg",
            palace_path,
            identity={"aliases": {"mac-claude": ["claude"]}},
        )
        monkeypatch.setattr(mcp_server, "_config", cfg)
        monkeypatch.setattr(mcp_server, "_get_collection", lambda *a, **kw: col)

        col.add(
            ids=["d1", "d2"],
            documents=["under claude", "under mac-claude"],
            metadatas=[
                {
                    "wing": "wing_x",
                    "room": "diary",
                    "agent": "claude",
                    "topic": "t",
                    "filed_at": "2026-01-01T00:00:00",
                    "date": "2026-01-01",
                },
                {
                    "wing": "wing_x",
                    "room": "diary",
                    "agent": "mac-claude",
                    "topic": "t",
                    "filed_at": "2026-01-02T00:00:00",
                    "date": "2026-01-02",
                },
            ],
        )

        result = tool_diary_read(agent_name="claude")
        assert result["total"] == 1
        assert result["entries"][0]["content"] == "under claude"

    def test_write_is_not_redirected_to_canonical(self, monkeypatch, palace_collection):
        from mempalace import mcp_server
        from mempalace.mcp_server import tool_diary_write

        palace_path, col = palace_collection
        cfg = _write_config(
            str(palace_path) + "-cfg",
            palace_path,
            identity={"aliases": {"mac-claude": ["claude"]}},
        )
        monkeypatch.setattr(mcp_server, "_config", cfg)
        monkeypatch.setattr(mcp_server, "_get_collection", lambda *a, **kw: col)

        result = tool_diary_write(agent_name="claude", entry="hello")
        assert result["success"] is True
        assert result["agent"] == "claude"  # not redirected to "mac-claude"

        stored = col.get(where={"agent": "claude"}, include=["metadatas"])
        assert len(stored["ids"]) == 1


class TestDiaryAgentForHarnessConfigPrecedence:
    def test_configured_canonical_wins_over_hardcoded_default(self, tmp_path):
        from mempalace.hooks_cli import _diary_agent_for_harness

        cfg = _write_config(
            str(tmp_path / "cfgdir"),
            str(tmp_path / "palace"),
            identity={"canonical": "mac-claude"},
        )
        assert _diary_agent_for_harness("claude-code", config=cfg) == "mac-claude"
        assert _diary_agent_for_harness("codex", config=cfg) == "mac-claude"

    def test_falls_back_to_hardcoded_default_when_unconfigured(self, tmp_path):
        from mempalace.hooks_cli import _diary_agent_for_harness

        cfg = _write_config(str(tmp_path / "cfgdir"), str(tmp_path / "palace"))
        assert _diary_agent_for_harness("claude-code", config=cfg) == "claude"
        assert _diary_agent_for_harness("codex", config=cfg) == "codex"
