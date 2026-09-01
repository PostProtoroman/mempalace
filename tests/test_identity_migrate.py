"""
test_identity_migrate.py — Tests for `mempalace identity-migrate`
(mempalace.migrate.migrate_identity_aliases / plan_identity_renames).

Covers: dry-run-by-default, --apply writes, idempotency (second run finds
nothing), never touching non-diary drawers or document text, and the
no-config-configured no-op.
"""

import json
import os

import chromadb
import pytest

from mempalace.config import MempalaceConfig
from mempalace.migrate import migrate_identity_aliases, plan_identity_renames


def _write_config(cfg_dir, palace_path, identity=None):
    payload = {"palace_path": palace_path}
    if identity is not None:
        payload["identity"] = identity
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, "config.json"), "w") as f:
        json.dump(payload, f)
    return MempalaceConfig(config_dir=cfg_dir)


@pytest.fixture
def palace(tmp_path):
    palace_path = str(tmp_path / "palace")
    os.makedirs(palace_path)
    client = chromadb.PersistentClient(path=palace_path)
    col = client.get_or_create_collection("mempalace_drawers", metadata={"hnsw:space": "cosine"})
    yield palace_path, col
    client.close()


class TestPlanIdentityRenames:
    def test_only_diary_room_records_with_a_configured_alias_are_planned(self):
        alias_to_canonical = {"claude": "mac-claude", "claude_code": "mac-claude"}
        items = [
            ("d1", {"room": "diary", "agent": "claude"}),
            ("d2", {"room": "diary", "agent": "mac-claude"}),  # already canonical
            ("d3", {"room": "diary", "agent": "someone-else"}),  # not an alias
            ("d4", {"room": "not-diary", "agent": "claude"}),  # wrong room
            ("d5", {"room": "diary", "agent": "claude_code"}),
        ]
        summary, updates = plan_identity_renames(items, alias_to_canonical)
        assert summary == {("claude", "mac-claude"): 1, ("claude_code", "mac-claude"): 1}
        ids = {rec_id for rec_id, _ in updates}
        assert ids == {"d1", "d5"}
        for rec_id, meta in updates:
            assert meta["agent"] == "mac-claude"

    def test_only_agent_key_is_rewritten(self):
        alias_to_canonical = {"claude": "mac-claude"}
        items = [("d1", {"room": "diary", "agent": "claude", "topic": "t", "wing": "wing_x"})]
        _, updates = plan_identity_renames(items, alias_to_canonical)
        (rec_id, meta) = updates[0]
        assert meta["topic"] == "t"
        assert meta["wing"] == "wing_x"
        assert meta["agent"] == "mac-claude"


class TestMigrateIdentityAliases:
    def test_noop_when_no_aliases_configured(self, palace, capsys):
        palace_path, _col = palace
        cfg = _write_config(palace_path + "-cfg", palace_path)
        result = migrate_identity_aliases(palace_path, config=cfg)
        assert result is False
        assert "nothing to migrate" in capsys.readouterr().out

    def test_dry_run_is_the_default_and_writes_nothing(self, palace, capsys):
        palace_path, col = palace
        col.add(
            ids=["d1"],
            documents=["some checkpoint text"],
            metadatas=[
                {
                    "room": "diary",
                    "agent": "claude",
                    "wing": "wing_x",
                    "topic": "t",
                    "filed_at": "2026-01-01T00:00:00",
                    "date": "2026-01-01",
                }
            ],
        )
        cfg = _write_config(
            palace_path + "-cfg", palace_path, identity={"aliases": {"mac-claude": ["claude"]}}
        )

        result = migrate_identity_aliases(palace_path, config=cfg)
        assert result is True
        out = capsys.readouterr().out
        assert "DRY RUN" in out
        assert "'claude' -> 'mac-claude': 1" in out

        stored = col.get(ids=["d1"], include=["metadatas", "documents"])
        assert stored["metadatas"][0]["agent"] == "claude"  # untouched
        assert stored["documents"][0] == "some checkpoint text"  # text never touched

    def test_apply_rewrites_agent_metadata_only(self, palace):
        palace_path, col = palace
        col.add(
            ids=["d1"],
            documents=["some checkpoint text"],
            metadatas=[
                {
                    "room": "diary",
                    "agent": "claude",
                    "wing": "wing_x",
                    "topic": "t",
                    "filed_at": "2026-01-01T00:00:00",
                    "date": "2026-01-01",
                }
            ],
        )
        cfg = _write_config(
            palace_path + "-cfg", palace_path, identity={"aliases": {"mac-claude": ["claude"]}}
        )

        result = migrate_identity_aliases(palace_path, config=cfg, dry_run=False, apply=True)
        assert result is True

        stored = col.get(ids=["d1"], include=["metadatas", "documents"])
        assert stored["metadatas"][0]["agent"] == "mac-claude"
        assert stored["metadatas"][0]["wing"] == "wing_x"  # untouched
        assert stored["documents"][0] == "some checkpoint text"  # text never touched

    def test_apply_is_idempotent(self, palace):
        palace_path, col = palace
        col.add(
            ids=["d1"],
            documents=["some checkpoint text"],
            metadatas=[
                {
                    "room": "diary",
                    "agent": "claude",
                    "wing": "wing_x",
                    "topic": "t",
                    "filed_at": "2026-01-01T00:00:00",
                    "date": "2026-01-01",
                }
            ],
        )
        cfg = _write_config(
            palace_path + "-cfg", palace_path, identity={"aliases": {"mac-claude": ["claude"]}}
        )

        first = migrate_identity_aliases(palace_path, config=cfg, dry_run=False, apply=True)
        assert first is True

        second = migrate_identity_aliases(palace_path, config=cfg, dry_run=False, apply=True)
        assert second is False  # nothing left to migrate

        stored = col.get(ids=["d1"], include=["metadatas"])
        assert stored["metadatas"][0]["agent"] == "mac-claude"

    def test_non_diary_drawers_are_never_touched(self, palace):
        palace_path, col = palace
        col.add(
            ids=["d1"],
            documents=["a project note, not a diary entry"],
            metadatas=[{"room": "notes", "agent": "claude", "wing": "wing_x"}],
        )
        cfg = _write_config(
            palace_path + "-cfg", palace_path, identity={"aliases": {"mac-claude": ["claude"]}}
        )

        result = migrate_identity_aliases(palace_path, config=cfg, dry_run=False, apply=True)
        assert result is False

        stored = col.get(ids=["d1"], include=["metadatas"])
        assert stored["metadatas"][0]["agent"] == "claude"
