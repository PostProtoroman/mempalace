"""Tests for `mempalace backup` / `mempalace restore` (mempalace.backups,
mempalace.cli.cmd_backup, mempalace.cli.cmd_restore).

The live palace (~/.mempalace, 918MB in production) has no backup command
today beyond an ad-hoc manual tar. These commands snapshot the palace home
into a portable, checksum-verified archive and restore it back. Every test
below builds its own scratch palace-home under ``tmp_path`` — none touch the
real ``~/.mempalace``.

Coverage:
- round trip: scratch palace-home (real sqlite db + plain files) -> backup ->
  restore -> byte-compare non-sqlite files, PRAGMA integrity_check the sqlite
- the sqlite backup-API snapshot is used (not a raw copy) even against a
  live WAL-mode connection with uncommitted-but-checkpointed writes
- the plain-copy fallback path when the backup API fails, and that it is
  recorded in the manifest + logged as a WARNING
- --keep retention pruning old archive+manifest pairs
- restore refuses: missing manifest, checksum mismatch, non-empty target,
  a target that exists and is not a directory
- restore's path-traversal guard on a hand-crafted malicious archive
- cmd_backup / cmd_restore CLI wiring: defaults, argument forwarding, and
  that library exceptions become a clean SystemExit(1) rather than a
  traceback
"""

import json
import os
import sqlite3
import tarfile
from argparse import Namespace

import pytest

from mempalace.backups import (
    DEFAULT_BACKUP_DEST,
    DEFAULT_BACKUP_KEEP,
    create_backup,
    restore_backup,
)
from mempalace.cli import _palace_home_dir, cmd_backup, cmd_restore


def _make_palace_home(base, *, wal=False):
    """Build a scratch palace-home: config/hallways/tunnels + palace/ + a
    top-level knowledge_graph.sqlite3, mirroring the real ~/.mempalace layout
    (mempalace/knowledge_graph.py: DEFAULT_KG_PATH is a sibling of palace/,
    not inside it).

    Returns the home Path plus the two sqlite paths for direct inspection.
    """
    home = base / "palace_home"
    home.mkdir()

    (home / "config.json").write_text(
        json.dumps({"palace_path": "unused-in-test"}), encoding="utf-8"
    )
    (home / "hallways.json").write_text(json.dumps({"hallways": []}), encoding="utf-8")
    (home / "tunnels.json").write_text(json.dumps({"tunnels": []}), encoding="utf-8")

    palace_dir = home / "palace"
    palace_dir.mkdir()
    (palace_dir / "notes.txt").write_text("verbatim drawer content\nline two\n", encoding="utf-8")
    nested = palace_dir / "wing_boss" / "room_2026-08-31"
    nested.mkdir(parents=True)
    (nested / "drawer.txt").write_text("a nested verbatim file", encoding="utf-8")

    kg_path = home / "knowledge_graph.sqlite3"
    conn = sqlite3.connect(str(kg_path))
    if wal:
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE facts (id INTEGER PRIMARY KEY, subject TEXT, predicate TEXT)")
    conn.executemany(
        "INSERT INTO facts (subject, predicate) VALUES (?, ?)",
        [("boss", "runs"), ("mempalace", "stores_verbatim")],
    )
    conn.commit()

    chroma_path = palace_dir / "chroma.sqlite3"
    conn2 = sqlite3.connect(str(chroma_path))
    conn2.execute("CREATE TABLE embeddings (id INTEGER PRIMARY KEY, vec TEXT)")
    conn2.execute("INSERT INTO embeddings (vec) VALUES ('[0.1, 0.2]')")
    conn2.commit()
    conn2.close()

    return home, kg_path, chroma_path, conn  # kg conn left open by callers that need it "live"


def _rel_files(root):
    """Every regular file under root, relative to root, sorted."""
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            full = os.path.join(dirpath, name)
            out.append(os.path.relpath(full, root).replace(os.sep, "/"))
    return sorted(out)


# ---------------------------------------------------------------------------
# create_backup / restore_backup round trip
# ---------------------------------------------------------------------------


def test_round_trip_byte_compares_files_and_integrity_checks_sqlite(tmp_path):
    home, kg_path, chroma_path, kg_conn = _make_palace_home(tmp_path)
    kg_conn.close()

    dest = tmp_path / "backups"
    result = create_backup(str(home), str(dest), keep=7, log=lambda _l: None)

    assert os.path.isfile(result["archive_path"])
    assert os.path.isfile(result["manifest_path"])
    assert result["archive_path"].endswith(".tar.gz")
    assert os.path.basename(result["archive_path"]).startswith("mempalace-")

    manifest = result["manifest"]
    assert manifest["archive_sha256"]
    assert manifest["file_count"] == len(_rel_files(home))
    # Both sqlite files went through the backup-API path, not the fallback.
    assert sorted(manifest["sqlite_snapshotted_via_backup_api"]) == [
        "knowledge_graph.sqlite3",
        "palace/chroma.sqlite3",
    ]
    assert manifest["sqlite_snapshotted_via_plain_copy_fallback"] == []

    target = tmp_path / "restored"
    restore_result = restore_backup(result["archive_path"], str(target), log=lambda _l: None)
    assert restore_result["target_dir"] == str(target)

    # Every file made it, restored tree matches the source tree exactly.
    assert _rel_files(target) == _rel_files(home)

    for relpath in (
        "config.json",
        "hallways.json",
        "tunnels.json",
        "palace/notes.txt",
        "palace/wing_boss/room_2026-08-31/drawer.txt",
    ):
        assert (target / relpath).read_bytes() == (home / relpath).read_bytes()

    for db_name in ("knowledge_graph.sqlite3", "palace/chroma.sqlite3"):
        restored_db = target / db_name
        conn = sqlite3.connect(str(restored_db))
        try:
            (integrity,) = conn.execute("PRAGMA integrity_check").fetchone()
            assert integrity == "ok"
        finally:
            conn.close()

    conn = sqlite3.connect(str(target / "knowledge_graph.sqlite3"))
    try:
        rows = conn.execute("SELECT subject, predicate FROM facts ORDER BY id").fetchall()
    finally:
        conn.close()
    assert rows == [("boss", "runs"), ("mempalace", "stores_verbatim")]


def test_round_trip_survives_live_wal_connection(tmp_path):
    """The sqlite file must snapshot correctly via the backup API even while
    a WAL-mode connection to it is still open (simulating a live MCP
    server/daemon writing to the palace during the backup)."""
    home, kg_path, _chroma, kg_conn = _make_palace_home(tmp_path, wal=True)
    try:
        # More committed writes after WAL mode was turned on, connection
        # stays open (not closed) through the backup call below.
        kg_conn.execute("INSERT INTO facts (subject, predicate) VALUES (?, ?)", ("bali", "villa"))
        kg_conn.commit()

        dest = tmp_path / "backups"
        result = create_backup(str(home), str(dest), keep=7, log=lambda _l: None)
        assert "knowledge_graph.sqlite3" in result["manifest"]["sqlite_snapshotted_via_backup_api"]

        target = tmp_path / "restored"
        restore_backup(result["archive_path"], str(target), log=lambda _l: None)

        conn = sqlite3.connect(str(target / "knowledge_graph.sqlite3"))
        try:
            (integrity,) = conn.execute("PRAGMA integrity_check").fetchone()
            assert integrity == "ok"
            count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        finally:
            conn.close()
        assert count == 3
    finally:
        kg_conn.close()


def test_sqlite_backup_api_failure_falls_back_to_plain_copy_and_warns(tmp_path, monkeypatch):
    home, kg_path, _chroma, kg_conn = _make_palace_home(tmp_path)
    kg_conn.close()

    import mempalace.backups as backups_mod

    def _boom(_src, _dst):
        raise sqlite3.OperationalError("simulated backup API failure")

    monkeypatch.setattr(backups_mod, "_snapshot_sqlite_via_backup_api", _boom)

    logs = []
    result = create_backup(str(home), str(tmp_path / "backups"), keep=7, log=logs.append)

    assert "knowledge_graph.sqlite3" in result["sqlite_fallback"]
    assert "palace/chroma.sqlite3" in result["sqlite_fallback"]
    assert result["sqlite_snapshotted"] == []
    assert any("WARNING" in line and "knowledge_graph.sqlite3" in line for line in logs)

    manifest = result["manifest"]
    assert sorted(manifest["sqlite_snapshotted_via_plain_copy_fallback"]) == [
        "knowledge_graph.sqlite3",
        "palace/chroma.sqlite3",
    ]

    # The fallback is a plain copy of a static (non-hot) file here, so it is
    # still a valid, restorable database.
    target = tmp_path / "restored"
    restore_backup(result["archive_path"], str(target), log=lambda _l: None)
    conn = sqlite3.connect(str(target / "knowledge_graph.sqlite3"))
    try:
        (integrity,) = conn.execute("PRAGMA integrity_check").fetchone()
    finally:
        conn.close()
    assert integrity == "ok"


def test_create_backup_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        create_backup(str(tmp_path / "does-not-exist"), str(tmp_path / "dest"))


# ---------------------------------------------------------------------------
# --keep retention
# ---------------------------------------------------------------------------


def test_keep_prunes_oldest_archive_and_manifest_pairs(tmp_path):
    home, _kg, _chroma, kg_conn = _make_palace_home(tmp_path)
    kg_conn.close()
    dest = tmp_path / "backups"
    dest.mkdir()

    # Seed 3 pre-existing (fake) archive+manifest pairs with distinct, spaced
    # mtimes so recency is unambiguous regardless of filesystem timestamp
    # resolution.
    seeded = []
    for i in range(3):
        stem = dest / f"mempalace-fake{i}"
        archive = stem.with_suffix(".tar.gz")
        archive.write_bytes(b"not a real archive")
        manifest = dest / f"mempalace-fake{i}.manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        ts = 1_700_000_000 + i * 3600
        os.utime(archive, (ts, ts))
        os.utime(manifest, (ts, ts))
        seeded.append((archive, manifest))

    # keep=2: this new backup plus the newest 1 of the 3 seeded ones survive;
    # the 2 oldest seeded pairs are pruned.
    result = create_backup(str(home), str(dest), keep=2, log=lambda _l: None)

    remaining_archives = sorted(dest.glob("mempalace-*.tar.gz"))
    remaining_manifests = sorted(dest.glob("mempalace-*.manifest.json"))
    assert len(remaining_archives) == 2
    assert len(remaining_manifests) == 2
    assert result["archive_path"] in [str(p) for p in remaining_archives]
    # The two oldest fakes are gone.
    assert not seeded[0][0].exists()
    assert not seeded[1][0].exists()
    assert not seeded[0][1].exists()
    assert not seeded[1][1].exists()
    # The newest fake survived.
    assert seeded[2][0].exists()
    assert seeded[2][1].exists()


def test_keep_zero_disables_pruning(tmp_path):
    home, _kg, _chroma, kg_conn = _make_palace_home(tmp_path)
    kg_conn.close()
    dest = tmp_path / "backups"

    for _ in range(3):
        create_backup(str(home), str(dest), keep=0, log=lambda _l: None)

    assert len(list(dest.glob("mempalace-*.tar.gz"))) == 3


def test_same_second_collision_gets_a_disambiguating_suffix_not_clobbered(tmp_path):
    """Two backups within the same wall-clock second (same %Y%m%d-%H%M%S
    timestamp) must not silently overwrite one another."""
    home, _kg, _chroma, kg_conn = _make_palace_home(tmp_path)
    kg_conn.close()
    dest = tmp_path / "backups"

    results = [create_backup(str(home), str(dest), keep=0, log=lambda _l: None) for _ in range(3)]

    archive_paths = {r["archive_path"] for r in results}
    manifest_paths = {r["manifest_path"] for r in results}
    assert len(archive_paths) == 3, "each call must get a distinct archive path"
    assert len(manifest_paths) == 3, "each call must get a distinct manifest path"
    for r in results:
        assert os.path.isfile(r["archive_path"])
        assert os.path.isfile(r["manifest_path"])


# ---------------------------------------------------------------------------
# restore refusals
# ---------------------------------------------------------------------------


def test_restore_refuses_missing_manifest(tmp_path):
    fake_archive = tmp_path / "mempalace-20260101-000000.tar.gz"
    with tarfile.open(fake_archive, "w:gz"):
        pass  # empty archive; the missing manifest is what matters here

    with pytest.raises(FileNotFoundError, match="Manifest not found"):
        restore_backup(str(fake_archive), str(tmp_path / "target"))


def test_restore_refuses_checksum_mismatch(tmp_path):
    home, _kg, _chroma, kg_conn = _make_palace_home(tmp_path)
    kg_conn.close()
    result = create_backup(str(home), str(tmp_path / "backups"), keep=7, log=lambda _l: None)

    # Corrupt the archive after the manifest recorded its original hash.
    with open(result["archive_path"], "ab") as f:
        f.write(b"\x00corruption")

    with pytest.raises(ValueError, match="checksum mismatch"):
        restore_backup(result["archive_path"], str(tmp_path / "target"))


def test_restore_refuses_nonempty_target(tmp_path):
    home, _kg, _chroma, kg_conn = _make_palace_home(tmp_path)
    kg_conn.close()
    result = create_backup(str(home), str(tmp_path / "backups"), keep=7, log=lambda _l: None)

    target = tmp_path / "target"
    target.mkdir()
    (target / "already-here.txt").write_text("do not overwrite me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        restore_backup(result["archive_path"], str(target))
    # Refusal must be a no-op: the pre-existing file is untouched.
    assert (target / "already-here.txt").read_text(encoding="utf-8") == "do not overwrite me"


def test_restore_refuses_target_that_is_a_file(tmp_path):
    home, _kg, _chroma, kg_conn = _make_palace_home(tmp_path)
    kg_conn.close()
    result = create_backup(str(home), str(tmp_path / "backups"), keep=7, log=lambda _l: None)

    target = tmp_path / "target-is-a-file"
    target.write_text("i am a file, not a directory", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not a directory"):
        restore_backup(result["archive_path"], str(target))


def test_restore_rejects_path_traversal_member(tmp_path):
    """A hand-crafted malicious archive whose member escapes target_dir must
    be refused even when its checksum matches its (also hand-crafted)
    manifest — the escape guard runs before any file is extracted."""
    evil_dir = tmp_path / "evil_source"
    evil_dir.mkdir()
    outside_marker = tmp_path / "evil_source" / "payload.txt"
    outside_marker.write_text("should never land outside target_dir", encoding="utf-8")

    archive_path = tmp_path / "mempalace-20260101-000000.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(str(outside_marker), arcname="../escaped.txt")

    import hashlib

    sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "mempalace-20260101-000000.manifest.json"
    manifest_path.write_text(json.dumps({"archive_sha256": sha, "file_count": 1}), encoding="utf-8")

    target = tmp_path / "target"
    with pytest.raises(ValueError, match="escapes the target directory"):
        restore_backup(str(archive_path), str(target))

    # Nothing was extracted outside target_dir, and target_dir itself should
    # not have picked up the escaping member either.
    assert not (tmp_path / "escaped.txt").exists()


# ---------------------------------------------------------------------------
# _palace_home_dir
# ---------------------------------------------------------------------------


def test_palace_home_dir_is_parent_of_palace_path(tmp_path):
    palace_path = tmp_path / "custom_home" / "palace"
    home = _palace_home_dir(Namespace(palace=str(palace_path)))
    assert home == str(palace_path.parent)


def test_palace_home_dir_normalizes_trailing_slash(tmp_path):
    palace_path = str(tmp_path / "custom_home" / "palace") + os.sep
    home = _palace_home_dir(Namespace(palace=palace_path))
    assert home == str(tmp_path / "custom_home")


# ---------------------------------------------------------------------------
# cmd_backup / cmd_restore CLI wiring
# ---------------------------------------------------------------------------


def test_cmd_backup_uses_defaults_and_reports_summary(monkeypatch, tmp_path, capsys):
    import mempalace.backups as backups_mod

    captured = {}

    def fake_create_backup(source_dir, dest_dir, *, keep, log=None):
        captured["source_dir"] = source_dir
        captured["dest_dir"] = dest_dir
        captured["keep"] = keep
        if log:
            log("  Backup: wrote fake archive")
        return {
            "archive_path": str(tmp_path / "mempalace-x.tar.gz"),
            "manifest_path": str(tmp_path / "mempalace-x.manifest.json"),
            "manifest": {"file_count": 3},
            "skipped": [],
        }

    monkeypatch.setattr(backups_mod, "create_backup", fake_create_backup)

    palace_home = tmp_path / "home"
    cmd_backup(Namespace(palace=str(palace_home / "palace"), dest=None, keep=None))

    assert captured["source_dir"] == str(palace_home)
    assert captured["dest_dir"] == DEFAULT_BACKUP_DEST
    assert captured["keep"] == DEFAULT_BACKUP_KEEP

    out = capsys.readouterr().out
    assert "mempalace-x.tar.gz" in out
    assert "Files:    3" in out


def test_cmd_backup_forwards_explicit_dest_and_keep(monkeypatch, tmp_path):
    import mempalace.backups as backups_mod

    captured = {}

    def fake_create_backup(source_dir, dest_dir, *, keep, log=None):
        captured["dest_dir"] = dest_dir
        captured["keep"] = keep
        return {
            "archive_path": "irrelevant",
            "manifest_path": "irrelevant",
            "manifest": {"file_count": 0},
            "skipped": [],
        }

    monkeypatch.setattr(backups_mod, "create_backup", fake_create_backup)

    explicit_dest = str(tmp_path / "my-backups")
    cmd_backup(Namespace(palace=str(tmp_path / "home" / "palace"), dest=explicit_dest, keep=3))

    assert captured["dest_dir"] == explicit_dest
    assert captured["keep"] == 3


def test_cmd_backup_reports_error_and_exits_nonzero(monkeypatch, tmp_path, capsys):
    import mempalace.backups as backups_mod

    def fake_create_backup(*_a, **_k):
        raise FileNotFoundError("Palace home not found or not a directory: /nope")

    monkeypatch.setattr(backups_mod, "create_backup", fake_create_backup)

    with pytest.raises(SystemExit) as exc_info:
        cmd_backup(Namespace(palace=str(tmp_path / "home" / "palace"), dest=None, keep=None))

    assert exc_info.value.code == 1
    assert "ERROR" in capsys.readouterr().err


def test_cmd_restore_forwards_args_and_reports_summary(monkeypatch, tmp_path, capsys):
    import mempalace.backups as backups_mod

    captured = {}

    def fake_restore_backup(archive_path, target_dir, *, log=None):
        captured["archive_path"] = archive_path
        captured["target_dir"] = target_dir
        return {
            "target_dir": target_dir,
            "manifest": {"created_at": "2026-08-31T04:00:00", "source_dir": "/home/.mempalace"},
            "archive_sha256": "deadbeef",
        }

    monkeypatch.setattr(backups_mod, "restore_backup", fake_restore_backup)

    archive = str(tmp_path / "mempalace-x.tar.gz")
    target = str(tmp_path / "restored")
    cmd_restore(Namespace(archive=archive, target=target))

    assert captured["archive_path"] == archive
    assert captured["target_dir"] == target
    out = capsys.readouterr().out
    assert "Restored to" in out
    assert "2026-08-31T04:00:00" in out


@pytest.mark.parametrize(
    "exc_type,message",
    [
        (FileNotFoundError, "archive not found"),
        (ValueError, "checksum mismatch"),
        (FileExistsError, "target is not empty"),
    ],
)
def test_cmd_restore_reports_error_and_exits_nonzero(
    monkeypatch, tmp_path, capsys, exc_type, message
):
    import mempalace.backups as backups_mod

    def fake_restore_backup(*_a, **_k):
        raise exc_type(message)

    monkeypatch.setattr(backups_mod, "restore_backup", fake_restore_backup)

    with pytest.raises(SystemExit) as exc_info:
        cmd_restore(Namespace(archive=str(tmp_path / "x.tar.gz"), target=str(tmp_path / "t")))

    assert exc_info.value.code == 1
    assert message in capsys.readouterr().err


# ---------------------------------------------------------------------------
# End-to-end through the CLI command functions (not just the library calls)
# ---------------------------------------------------------------------------


def test_end_to_end_backup_then_restore_via_cli_commands(tmp_path, capsys):
    home, _kg, _chroma, kg_conn = _make_palace_home(tmp_path)
    kg_conn.close()

    dest = tmp_path / "backups"
    cmd_backup(Namespace(palace=str(home / "palace"), dest=str(dest), keep=7))
    capsys.readouterr()  # drain backup's stdout

    (archive_path,) = list(dest.glob("mempalace-*.tar.gz"))
    target = tmp_path / "restored"
    cmd_restore(Namespace(archive=str(archive_path), target=str(target)))

    assert _rel_files(target) == _rel_files(home)
    conn = sqlite3.connect(str(target / "knowledge_graph.sqlite3"))
    try:
        (integrity,) = conn.execute("PRAGMA integrity_check").fetchone()
    finally:
        conn.close()
    assert integrity == "ok"
