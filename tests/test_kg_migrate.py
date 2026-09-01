"""
test_kg_migrate.py — Tests for `mempalace kg-migrate` (mempalace.migrate.migrate_knowledge_graph).

Covers: dry-run-by-default, --apply writes, idempotency (second run merges
zero), never touching the legacy file, and empty/missing legacy KG handling.
"""

import sqlite3

from mempalace.knowledge_graph import KnowledgeGraph
from mempalace.migrate import migrate_knowledge_graph


def _seed_legacy_kg(path, triples):
    kg = KnowledgeGraph(db_path=str(path))
    for subject, predicate, obj, valid_from in triples:
        kg.add_triple(subject, predicate, obj, valid_from=valid_from)
    kg.close()


class TestMigrateKnowledgeGraphDryRun:
    def test_dry_run_is_the_default(self, tmp_path, monkeypatch):
        legacy = tmp_path / "legacy_kg.sqlite3"
        palace_dir = tmp_path / "palace"
        palace_dir.mkdir()
        _seed_legacy_kg(legacy, [("alice", "likes", "coffee", "2026-01-01")])
        monkeypatch.setattr("mempalace.knowledge_graph.LEGACY_KG_PATH", str(legacy))

        result = migrate_knowledge_graph(str(palace_dir))

        assert result["applied"] is False
        assert result["merged_triples"] == 1  # would-merge count, reported
        target = palace_dir / "knowledge_graph.sqlite3"
        # Dry run must write nothing: no target file, or an empty one.
        if target.exists():
            with sqlite3.connect(str(target)) as conn:
                count = conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
            assert count == 0

    def test_apply_writes_the_merge(self, tmp_path, monkeypatch):
        legacy = tmp_path / "legacy_kg.sqlite3"
        palace_dir = tmp_path / "palace"
        palace_dir.mkdir()
        _seed_legacy_kg(legacy, [("alice", "likes", "coffee", "2026-01-01")])
        monkeypatch.setattr("mempalace.knowledge_graph.LEGACY_KG_PATH", str(legacy))

        result = migrate_knowledge_graph(str(palace_dir), apply=True)

        assert result["applied"] is True
        assert result["merged_triples"] == 1

        target_kg = KnowledgeGraph(db_path=str(palace_dir / "knowledge_graph.sqlite3"))
        try:
            relations = target_kg.query_entity("alice")
        finally:
            target_kg.close()
        assert any(r.get("predicate") == "likes" and r.get("object") == "coffee" for r in relations)

    def test_apply_never_touches_legacy_file(self, tmp_path, monkeypatch):
        legacy = tmp_path / "legacy_kg.sqlite3"
        palace_dir = tmp_path / "palace"
        palace_dir.mkdir()
        _seed_legacy_kg(legacy, [("alice", "likes", "coffee", "2026-01-01")])
        monkeypatch.setattr("mempalace.knowledge_graph.LEGACY_KG_PATH", str(legacy))

        before = legacy.read_bytes()
        migrate_knowledge_graph(str(palace_dir), apply=True)
        after = legacy.read_bytes()

        assert before == after

    def test_idempotent_second_run_merges_zero(self, tmp_path, monkeypatch):
        legacy = tmp_path / "legacy_kg.sqlite3"
        palace_dir = tmp_path / "palace"
        palace_dir.mkdir()
        _seed_legacy_kg(
            legacy,
            [
                ("alice", "likes", "coffee", "2026-01-01"),
                ("bob", "works_on", "orion", "2026-02-01"),
            ],
        )
        monkeypatch.setattr("mempalace.knowledge_graph.LEGACY_KG_PATH", str(legacy))

        first = migrate_knowledge_graph(str(palace_dir), apply=True)
        assert first["merged_triples"] == 2

        second = migrate_knowledge_graph(str(palace_dir), apply=True)
        assert second["merged_triples"] == 0
        assert second["skipped_existing"] == 2

        # And the target only has the two triples once each, not duplicated.
        target_kg = KnowledgeGraph(db_path=str(palace_dir / "knowledge_graph.sqlite3"))
        try:
            with sqlite3.connect(str(palace_dir / "knowledge_graph.sqlite3")) as conn:
                count = conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
        finally:
            target_kg.close()
        assert count == 2

    def test_no_legacy_file_is_a_no_op(self, tmp_path, monkeypatch):
        palace_dir = tmp_path / "palace"
        palace_dir.mkdir()
        monkeypatch.setattr(
            "mempalace.knowledge_graph.LEGACY_KG_PATH", str(tmp_path / "does_not_exist.sqlite3")
        )

        result = migrate_knowledge_graph(str(palace_dir), apply=True)

        assert result["merged_triples"] == 0
        assert result["applied"] is False
        assert not (palace_dir / "knowledge_graph.sqlite3").exists()

    def test_legacy_and_target_same_file_is_a_no_op(self, tmp_path, monkeypatch):
        palace_dir = tmp_path / "palace"
        palace_dir.mkdir()
        target = palace_dir / "knowledge_graph.sqlite3"
        _seed_legacy_kg(target, [("alice", "likes", "coffee", "2026-01-01")])
        monkeypatch.setattr("mempalace.knowledge_graph.LEGACY_KG_PATH", str(target))

        result = migrate_knowledge_graph(str(palace_dir), apply=True)

        assert result["merged_triples"] == 0
        assert result["applied"] is False

    def test_preserves_existing_palace_facts(self, tmp_path, monkeypatch):
        """Migration must merge additively, never clobber facts already in
        the palace KG."""
        legacy = tmp_path / "legacy_kg.sqlite3"
        palace_dir = tmp_path / "palace"
        palace_dir.mkdir()
        _seed_legacy_kg(legacy, [("alice", "likes", "coffee", "2026-01-01")])
        _seed_legacy_kg(
            palace_dir / "knowledge_graph.sqlite3",
            [("bob", "works_on", "orion", "2026-02-01")],
        )
        monkeypatch.setattr("mempalace.knowledge_graph.LEGACY_KG_PATH", str(legacy))

        result = migrate_knowledge_graph(str(palace_dir), apply=True)
        assert result["merged_triples"] == 1

        target_kg = KnowledgeGraph(db_path=str(palace_dir / "knowledge_graph.sqlite3"))
        try:
            alice = target_kg.query_entity("alice")
            bob = target_kg.query_entity("bob")
        finally:
            target_kg.close()
        assert any(r.get("object") == "coffee" for r in alice)
        assert any(r.get("object") == "orion" for r in bob)
