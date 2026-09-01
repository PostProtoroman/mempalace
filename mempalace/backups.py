"""Writing and pruning palace backups.

``mempalace migrate`` and ``mempalace repair max-seq-id`` each write a fresh,
timestamped backup every time they run and historically never deleted the old
ones. On a machine that mines or repairs on a schedule those full-size copies
accumulate silently — a real palace was found with hundreds of gigabytes of
backups sitting beside only a few hundred megabytes of live data, nearly
filling the disk. This module prunes the backup set down to a bounded count
after each new backup is written.

The retention count comes from ``MempalaceConfig.max_backups`` (default 10).

``copy_palace_dir`` is the other half: the whole-directory copy that
``mempalace repair`` in its default mode and ``mempalace migrate`` take before
they overwrite a live palace. The other backup paths copy a single file and do
not use it.

``create_backup`` / ``restore_backup`` back ``mempalace backup`` / ``mempalace
restore`` (a separate feature from the above): a portable, timestamped archive
of the whole palace home — not just a same-machine safety copy an in-place
rebuild takes before overwriting itself. Retention there is a separate knob
(``--keep``, default ``DEFAULT_BACKUP_KEEP``) from ``max_backups``, because the
two accumulate at very different rates and for different reasons: migrate/
repair backups are an incidental byproduct of maintenance commands a user may
run often; ``mempalace backup`` is the deliberate, typically-scheduled disaster
recovery copy.
"""

import glob
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tarfile
import tempfile
from datetime import datetime

from .config import sqlite_read_uri
from .version import __version__

# Where ``mempalace backup`` writes archives when ``--dest`` is not given.
# Deliberately OUTSIDE the palace home (``~/.mempalace``) so a filesystem-level
# disaster that takes out the palace — a bad rebuild, a wiped home dir entry,
# a full disk that corrupts whatever inode was mid-write — does not also take
# out its own backups.
DEFAULT_BACKUP_DEST = os.path.expanduser("~/.mempalace-backups")

# How many ``mempalace backup`` archives ``--keep`` retains by default.
# Deliberately its own constant rather than ``MempalaceConfig.max_backups``:
# that setting already means something else (retention for the incidental
# ``migrate``/``repair`` safety copies) and accumulates on a different
# schedule than a deliberate, typically-daily disaster-recovery backup.
DEFAULT_BACKUP_KEEP = 7

_ARCHIVE_SUFFIX = ".tar.gz"
_MANIFEST_SUFFIX = ".manifest.json"


# Names for the file types a backup copy deliberately leaves behind. A socket
# or a named pipe makes ``shutil.copytree`` record an error for that entry and
# raise at the end. Device nodes do not, but dereferencing one copies the
# DEVICE instead of palace data, and a link to ``/dev/zero`` is copied without
# bound, so they are left out for the opposite reason: the copy would succeed
# at the wrong thing.
_UNCOPYABLE_FILE_TYPES = (
    (stat.S_ISSOCK, "socket"),
    (stat.S_ISFIFO, "named pipe"),
    (stat.S_ISCHR, "character device"),
    (stat.S_ISBLK, "block device"),
)

_UNNAMED_FILE_TYPE = "not a regular file or directory"


def _file_type_label(mode):
    """Name the file type in ``mode``, falling back to a generic phrase.

    The fallback covers a file type this table has no name for. Nothing on
    Linux reaches it, because ``stat`` there reports ``S_IFDOOR``,
    ``S_IFPORT`` and ``S_IFWHT`` as 0. macOS does define ``S_IFWHT``, a
    union-mount whiteout, which is precisely what the fallback exists for:
    an entry no copy can carry is skipped with an honest reason rather than
    read as copyable.
    """
    for is_type, label in _UNCOPYABLE_FILE_TYPES:
        if is_type(mode):
            return label
    return _UNNAMED_FILE_TYPE


def _uncopyable_reason(path, *, follow_symlinks):
    """Name why ``path`` cannot be copied into a backup, or return ``None``.

    This is an allowlist, matching how the rest of the package treats
    directory entries: only regular files and directories are copyable, and
    everything else is named and left out.

    Args:
        path: The directory entry to classify.
        follow_symlinks: Mirrors the copy's own link handling. ``True`` when
            the copy dereferences links, so the TARGET's file type decides;
            ``False`` when it recreates them as links, in which case any
            symlink is fine and its target is never read.

    Returns:
        A short human-readable reason, or ``None`` both when the entry is
        copyable and when its type could not be established.

    Those two ``None`` cases are deliberately the same answer: only a file
    type this could read names a reason. An entry whose type it could not
    read is left for the copy to attempt and, if it really is broken, to
    fail on.

    A failed ``stat`` therefore never becomes a reason to skip. It says the
    entry cannot be resolved right now, which is not the same as holding no
    data, and no errno separates the two: a symlink into a volume that is
    not mounted fails exactly like one whose target was deleted, and on
    Windows an unmapped drive letter and an unreachable network share both
    arrive as ``ENOENT`` as well.
    """
    try:
        st = os.lstat(path)
    except OSError:
        return None

    if stat.S_ISLNK(st.st_mode):
        if not follow_symlinks:
            return None
        try:
            st = os.stat(path)
        except OSError:
            return None

    if stat.S_ISREG(st.st_mode) or stat.S_ISDIR(st.st_mode):
        return None
    return _file_type_label(st.st_mode)


def _report_skipped(skipped, src, log):
    """Tell the operator what the copy left out.

    Every step that can fail is guarded on its own, because this also runs
    after the copy has failed: both callers pass ``print``, so one line the
    terminal will not take must cost neither the copy's own outcome nor the
    lines after it.
    """
    noun = "entry" if len(skipped) == 1 else "entries"
    lines = [f"  Backup: skipped {len(skipped)} {noun} that cannot be copied:"]
    for path, reason in skipped:
        try:
            name = os.path.relpath(path, src)
        except (OSError, ValueError):
            # On Windows ``relpath`` rejects paths on different drives; for a
            # relative ``src`` it also reaches ``os.getcwd()``, which can fail
            # on a deleted directory. Neither costs us the entry's name.
            name = path
        lines.append(f"    {name} ({reason})")

    for line in lines:
        try:
            log(line)
        except UnicodeError:
            # An stdout that cannot encode this entry's name. Dropping the
            # line would leave a header whose count disagrees with the list
            # under it, so the name is escaped and retried before it is
            # given up on.
            try:
                log(line.encode("ascii", "backslashreplace").decode("ascii"))
            except (OSError, UnicodeError):
                continue
        except OSError:
            # A write that the device refused. A short report is buffered
            # whole, so this arrives only once the report is long enough to
            # flush partway through, and then the header is already out.
            # Anything else is a caller passing something that is not a
            # working ``log``, which should surface rather than be swallowed.
            continue


def copy_palace_dir(src, dst, *, symlinks=False, log=None):
    """Copy a palace directory to ``dst``, skipping entries no copy can carry.

    ``shutil.copytree`` finishes the copy but raises ``shutil.Error`` at the
    end when a directory entry is not something it knows how to duplicate: a
    Unix domain socket, or a named pipe. The caller has to treat that as a
    failed backup, so the command died before the rebuild the backup was
    guarding (#2207). Such entries are runtime artifacts of whatever process
    created them and never hold palace data, so a backup without them is
    still a complete backup of the palace.

    Args:
        src: Palace directory to copy.
        dst: Destination path. Must not already exist.
        symlinks: Passed to ``shutil.copytree``. ``True`` recreates symlinks
            as symlinks, ``False`` copies what they point at.
        log: Optional callable (e.g. ``print``) for human-readable progress.

    Returns:
        The list of ``(path, reason)`` pairs that were skipped: sorted within
        each directory, in the order the copy visited the directories. Like
        ``prune_backups``, this both returns what it did and logs it.

    Every other copy failure still raises out of ``shutil.copytree``. A caller
    about to overwrite the live palace must still stop when its safety copy
    did not come out whole, so this narrows what the copy attempts rather
    than swallowing what it reports.

    ``shutil.copytree`` hands the callback names rather than the ``os.scandir``
    entries it already holds, so classifying costs one extra ``os.lstat`` per
    directory entry, and it classifies a whole directory before copying any of
    it. An entry whose type changes inside that window is handled by the
    earlier reading, which cuts both ways: one that became a socket still
    aborts the copy, and one that was a socket and became a regular file is
    skipped with its contents. Nothing in MemPalace replaces an entry that
    way, and the skipped name is printed either way.
    """
    skipped = []
    # ``shutil.copytree`` classifies the top directory before it creates the
    # destination, so a copy that never starts still produces a skip list.
    # Reporting one would tell the operator what a nonexistent backup is
    # missing, so both ways that happens are excluded below: a destination
    # already occupied (this snapshot), and a destination that could not be
    # created at all.
    dst_existed = os.path.lexists(dst)

    def _ignore(directory, names):
        ignored = set()
        found = []
        for name in names:
            path = os.path.join(directory, name)
            reason = _uncopyable_reason(path, follow_symlinks=not symlinks)
            if reason is not None:
                ignored.add(name)
                found.append((path, reason))
        skipped.extend(sorted(found))
        return ignored

    def _report():
        if skipped and log and not dst_existed and os.path.isdir(dst):
            _report_skipped(skipped, src, log)

    try:
        shutil.copytree(src, dst, symlinks=symlinks, ignore=_ignore)
    except BaseException:
        # Reported for a failed copy too, including an interrupted one: a
        # half-written backup is exactly when the operator needs to know what
        # was left out of it. But the copy's own failure is what the caller
        # has to diagnose, so a report that fails on top of it is dropped
        # rather than raised in its place.
        try:
            _report()
        except Exception:
            pass
        raise

    # Not suppressed here: with the backup complete and nothing destructive
    # done yet, a ``log`` that cannot be called at all is a caller's bug worth
    # hearing about. The terminal failures inside ``_report_skipped`` are
    # absorbed on both paths.
    _report()
    return skipped


def prune_backups(pattern, max_backups, *, log=None):
    """Delete the oldest backups matching ``pattern`` so at most ``max_backups`` remain.

    Args:
        pattern: A glob pattern matching the backup paths (files or
            directories). The caller is responsible for ``glob.escape``-ing
            any literal, non-wildcard portion that can contain glob
            metacharacters — palace paths sometimes do (e.g. a ``[``).
        max_backups: Number of most-recent backups to keep. ``None`` or any
            value ``<= 0`` disables pruning and returns immediately, so a
            backup set is never touched when the user has opted out.
        log: Optional callable (e.g. ``print``) for human-readable progress.

    Returns:
        The list of paths that were successfully removed.

    Recency is determined by filesystem mtime rather than by parsing the
    timestamp out of the name, so it stays correct even when two backup
    producers use different timestamp formats. Deletion failures are logged
    and skipped: pruning is best-effort cleanup and must never abort the
    migrate/repair operation that just completed successfully.
    """
    if max_backups is None or max_backups <= 0:
        return []

    scored = []
    for path in glob.glob(pattern):
        try:
            scored.append((os.path.getmtime(path), path))
        except OSError:
            # Vanished between glob and stat (concurrent prune / cleanup);
            # nothing for us to remove.
            continue

    if len(scored) <= max_backups:
        return []

    # Newest first; the path breaks mtime ties so ordering is deterministic.
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

    removed = []
    for _mtime, path in scored[max_backups:]:
        try:
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except OSError as exc:
            if log:
                log(f"  Backup prune: could not remove {path}: {exc}")
            continue
        removed.append(path)
        if log:
            log(f"  Backup prune: removed old backup {path}")

    return removed


# ---------------------------------------------------------------------------
# mempalace backup / mempalace restore
# ---------------------------------------------------------------------------


def _sha256_file(path, chunk_size=1024 * 1024):
    """Hash a file's contents without reading it all into memory at once."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_sqlite_via_backup_api(src_path, dst_path):
    """Write a transactionally-consistent copy of the SQLite db at ``src_path``.

    Uses SQLite's own online backup API (``sqlite3.Connection.backup``) rather
    than a byte-for-byte file copy. A raw copy of a database that a live
    process (the MCP server, a daemon, a concurrent CLI run) may be writing to
    can capture a torn page or a half-applied WAL frame — a file that looks
    fine (correct size, valid header) but fails ``PRAGMA integrity_check`` or
    reads back with missing rows. The backup API takes the same read lock a
    normal reader would and copies a single consistent snapshot, the same way
    the ``sqlite3`` CLI's ``.backup`` command does.

    ``dst_path`` must not already exist — a fresh, empty destination lets
    SQLite create it cleanly rather than have to make sense of whatever bytes
    were already there (see ``create_backup``, which always deletes the
    placeholder ``copy_palace_dir`` left behind before calling this).

    Raises ``sqlite3.Error`` on any failure (source is not a valid SQLite
    database, is locked in a way that blocks even a read snapshot, etc.) —
    callers are expected to fall back to a plain file copy and warn.
    """
    src_conn = sqlite3.connect(sqlite_read_uri(src_path), uri=True)
    try:
        dst_conn = sqlite3.connect(dst_path)
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
    finally:
        src_conn.close()


def _archive_stem(dest_dir, timestamp):
    return os.path.join(dest_dir, f"mempalace-{timestamp}")


def _manifest_path_for_archive(archive_path):
    """Return the ``.manifest.json`` path that sits beside ``archive_path``.

    Strips the known archive suffix rather than ``os.path.splitext`` so a
    two-part extension like ``.tar.gz`` loses both parts, not just ``.gz``.
    """
    if archive_path.endswith(_ARCHIVE_SUFFIX):
        stem = archive_path[: -len(_ARCHIVE_SUFFIX)]
    else:
        stem = os.path.splitext(archive_path)[0]
    return stem + _MANIFEST_SUFFIX


def create_backup(source_dir, dest_dir, *, keep=DEFAULT_BACKUP_KEEP, log=None):
    """Snapshot ``source_dir`` (the palace home) into a timestamped archive.

    Writes ``<dest_dir>/mempalace-<YYYYmmdd-HHMMSS>.tar.gz`` plus a sibling
    ``.manifest.json`` recording the archive's sha256, its file list (path +
    size), the mempalace version that wrote it, and which files needed the
    plain-copy fallback below. After a successful write, prunes older
    archives in ``dest_dir`` down to ``keep`` (see ``prune_backups``).

    The archive is built in three steps so that no file — SQLite or not — is
    ever read directly out of a live, possibly-still-being-written palace and
    straight into the tar:

    1. ``copy_palace_dir`` copies the whole tree into a private staging
       directory, skipping anything no copy can carry (sockets, pipes, device
       nodes — see its own docstring). This is deliberately allowed to
       raw-copy ``*.sqlite3`` files too, even though that copy is potentially
       torn: step 2 unconditionally discards and replaces it before anything
       reads the staged tree, so the torn bytes are never tarred. Doing it
       this way (raw-copy-then-replace) reuses ``copy_palace_dir``'s already
       extensively-tested symlink/skip/permission handling as-is, rather than
       teaching it to special-case SQLite files itself.
    2. Every ``*.sqlite3`` found in the staged tree is deleted and replaced
       with a fresh snapshot of the ORIGINAL (untouched, still-live) file via
       ``_snapshot_sqlite_via_backup_api``. If that fails for a given file
       (corrupt source, backup API unavailable, whatever) a WARNING is logged
       and a plain ``shutil.copy2`` of the original stands in instead — a
       possibly-torn snapshot is still better than no backup of that file at
       all, and the manifest records which files took this path.
    3. The staging directory is tarred (``tar.gz`` — see module docstring for
       why not zstd) and hashed; the staging directory is removed either way.

    Args:
        source_dir: The palace home to back up — the directory holding
            ``config.json``, ``hallways.json``, ``tunnels.json``, and the
            ``palace/`` data directory. Must already exist.
        dest_dir: Where to write the archive + manifest. Created if missing.
            Should live outside ``source_dir`` (see ``DEFAULT_BACKUP_DEST``);
            not enforced here since a caller may deliberately want otherwise
            (e.g. a test asserting the archive gets skipped by a later run
            of itself), but nesting it inside defeats the point of a backup.
        keep: Passed straight to ``prune_backups`` as ``max_backups`` after
            this backup is written; ``None`` or ``<= 0`` disables pruning.
        log: Optional callable (e.g. ``print``) for human-readable progress.
            Also forwarded to ``copy_palace_dir`` and ``prune_backups``.

    Returns:
        A dict: ``archive_path``, ``manifest_path``, ``manifest`` (the dict
        that was written), ``skipped`` (copy_palace_dir's skip list),
        ``sqlite_snapshotted`` and ``sqlite_fallback`` (relative paths, by
        which of the two step-2 paths each SQLite file took), and ``pruned``
        (paths ``prune_backups`` removed).

    Raises:
        FileNotFoundError: ``source_dir`` does not exist / is not a directory.
        Whatever ``copy_palace_dir`` or ``tarfile`` raise on an unrecoverable
        copy or archive failure (see ``copy_palace_dir``'s own docstring —
        only entries it can prove are uncopyable are skipped; everything else
        still aborts the backup loudly rather than silently write a partial
        one).
    """
    log = log or (lambda _line: None)
    source_dir = os.path.abspath(os.path.expanduser(str(source_dir)))
    dest_dir = os.path.abspath(os.path.expanduser(str(dest_dir)))

    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Palace home not found or not a directory: {source_dir}")

    os.makedirs(dest_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = _archive_stem(dest_dir, timestamp)
    archive_path = stem + _ARCHIVE_SUFFIX
    manifest_path = stem + _MANIFEST_SUFFIX
    # The timestamp has one-second resolution, so two calls inside the same
    # second (a test loop, a user re-running the command to double-check
    # something) would otherwise collide and the second call would silently
    # overwrite the first archive/manifest with `tarfile.open(..., "w:gz")` /
    # plain `open(..., "w")` — the one failure mode a backup command cannot
    # have. On a collision, append "-2", "-3", ... until both paths are free;
    # a normal daily-cron run never sees this branch.
    suffix = 1
    while os.path.exists(archive_path) or os.path.exists(manifest_path):
        suffix += 1
        stem = _archive_stem(dest_dir, f"{timestamp}-{suffix}")
        archive_path = stem + _ARCHIVE_SUFFIX
        manifest_path = stem + _MANIFEST_SUFFIX

    staging_parent = tempfile.mkdtemp(prefix="mempalace_backup_")
    try:
        # A path that does not exist yet: copy_palace_dir creates it (and
        # refuses to run if it already exists — see its own docstring).
        staging_dir = os.path.join(staging_parent, "snapshot")
        skipped = copy_palace_dir(source_dir, staging_dir, symlinks=False, log=log)

        sqlite_snapshotted = []
        sqlite_fallback = []
        for dirpath, _dirnames, filenames in os.walk(staging_dir):
            for name in filenames:
                if not name.endswith(".sqlite3"):
                    continue
                staged_path = os.path.join(dirpath, name)
                relpath = os.path.relpath(staged_path, staging_dir)
                original_path = os.path.join(source_dir, relpath)
                if not os.path.isfile(original_path):
                    # Vanished (or was never a plain file) between the copy
                    # above and this pass; leave whatever copy_palace_dir
                    # already staged for it rather than lose the entry.
                    continue
                try:
                    os.remove(staged_path)
                    _snapshot_sqlite_via_backup_api(original_path, staged_path)
                    sqlite_snapshotted.append(relpath)
                except (sqlite3.Error, OSError) as exc:
                    log(
                        f"  WARNING: sqlite backup API failed for {relpath} ({exc}); "
                        "falling back to a plain copy"
                    )
                    shutil.copy2(original_path, staged_path)
                    sqlite_fallback.append(relpath)

        files = []
        for dirpath, _dirnames, filenames in os.walk(staging_dir):
            for name in filenames:
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, staging_dir).replace(os.sep, "/")
                try:
                    size = os.path.getsize(full)
                except OSError:
                    size = None
                files.append({"path": rel, "size": size})
        files.sort(key=lambda f: f["path"])

        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(staging_dir, arcname=".")
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)

    archive_sha256 = _sha256_file(archive_path)
    manifest = {
        "manifest_version": 1,
        "mempalace_version": __version__,
        "created_at": datetime.now().astimezone().isoformat(),
        "source_dir": source_dir,
        "archive": os.path.basename(archive_path),
        "archive_format": "tar.gz",
        "archive_sha256": archive_sha256,
        "archive_size": os.path.getsize(archive_path),
        "file_count": len(files),
        "files": files,
        "sqlite_snapshotted_via_backup_api": sorted(sqlite_snapshotted),
        "sqlite_snapshotted_via_plain_copy_fallback": sorted(sqlite_fallback),
        "skipped": [{"path": p, "reason": reason} for p, reason in (skipped or [])],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
        f.write("\n")

    log(f"  Backup: wrote {archive_path} ({manifest['archive_size']} bytes)")
    if sqlite_fallback:
        log(
            f"  Backup: {len(sqlite_fallback)} sqlite file(s) used the plain-copy "
            "fallback — see the manifest for which"
        )

    pruned = []
    pruned += prune_backups(
        os.path.join(glob.escape(dest_dir), "mempalace-*" + _ARCHIVE_SUFFIX),
        keep,
        log=log,
    )
    pruned += prune_backups(
        os.path.join(glob.escape(dest_dir), "mempalace-*" + _MANIFEST_SUFFIX),
        keep,
        log=log,
    )

    return {
        "archive_path": archive_path,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "skipped": skipped,
        "sqlite_snapshotted": sqlite_snapshotted,
        "sqlite_fallback": sqlite_fallback,
        "pruned": pruned,
    }


def _safe_extract_all(tar, target_dir):
    """Extract every member of ``tar`` into ``target_dir``, refusing escapes.

    ``mempalace backup`` never writes a member that triggers this (symlinks
    are dereferenced during the copy — see ``copy_palace_dir(symlinks=False)``
    — so the archive holds regular files and directories only), but
    ``restore`` accepts an arbitrary archive path, so a member whose name or
    link target resolves outside ``target_dir`` (``../../etc/passwd``, an
    absolute path, a symlink pointing out) is rejected rather than followed.

    Prefers ``tarfile``'s own ``filter="data"`` hardening (3.12+) and falls
    back to the pre-3.12 API on the ``requires-python = ">=3.9"`` floor this
    package supports; the manual check above is the real guard either way.
    """
    target_dir = os.path.abspath(target_dir)
    for member in tar.getmembers():
        member_path = os.path.abspath(os.path.join(target_dir, member.name))
        if member_path != target_dir and not member_path.startswith(target_dir + os.sep):
            raise ValueError(
                f"Refusing to restore: archive member escapes the target directory: {member.name!r}"
            )
        if member.issym() or member.islnk():
            link_target = os.path.abspath(
                os.path.join(os.path.dirname(member_path), member.linkname)
            )
            if link_target != target_dir and not link_target.startswith(target_dir + os.sep):
                raise ValueError(
                    f"Refusing to restore: archive member {member.name!r} links "
                    f"outside the target directory ({member.linkname!r})"
                )

    try:
        tar.extractall(target_dir, filter="data")
    except TypeError:
        # Python < 3.12: extractall() has no `filter` parameter yet.
        tar.extractall(target_dir)


def restore_backup(archive_path, target_dir, *, log=None):
    """Restore a ``mempalace backup`` archive into ``target_dir``.

    Refuses to run against a ``target_dir`` that already exists and is
    non-empty — restoring is an unpack-from-scratch operation, not a merge,
    and unpacking on top of an existing palace (or anything else) risks
    silently mixing the two. Point it at a fresh or empty directory, then
    move it into place (e.g. over ``~/.mempalace``) once you have verified it.

    Verifies the archive's sha256 against the value recorded in its
    ``.manifest.json`` sibling BEFORE extracting anything, so a truncated
    download, a bit-rotted disk, or a tampered archive is caught up front
    rather than partially unpacked.

    Args:
        archive_path: Path to the ``.tar.gz`` archive ``mempalace backup``
            wrote. Its ``.manifest.json`` sibling must exist alongside it.
        target_dir: Directory to restore into. Created if it does not exist;
            rejected if it exists and is non-empty.
        log: Optional callable (e.g. ``print``) for human-readable progress.

    Returns:
        A dict: ``target_dir``, ``manifest`` (the parsed manifest), and
        ``archive_sha256`` (the verified hash).

    Raises:
        FileNotFoundError: the archive or its manifest does not exist.
        ValueError: the manifest has no recorded hash, or the archive's
            actual sha256 does not match it.
        FileExistsError: ``target_dir`` exists and is non-empty, or exists
            and is not a directory at all.
    """
    log = log or (lambda _line: None)
    archive_path = os.path.abspath(os.path.expanduser(str(archive_path)))
    target_dir = os.path.abspath(os.path.expanduser(str(target_dir)))

    if not os.path.isfile(archive_path):
        raise FileNotFoundError(f"Backup archive not found: {archive_path}")

    manifest_path = _manifest_path_for_archive(archive_path)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Manifest not found alongside archive (expected {manifest_path}); "
            "restore refuses to unpack an archive it cannot verify"
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    expected_sha256 = manifest.get("archive_sha256")
    if not expected_sha256:
        raise ValueError(f"Manifest {manifest_path} has no archive_sha256 to verify against")

    log(f"  Restore: verifying {os.path.basename(archive_path)} against its manifest...")
    actual_sha256 = _sha256_file(archive_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Archive checksum mismatch for {archive_path}: got {actual_sha256}, "
            f"manifest {manifest_path} expects {expected_sha256}. Refusing to "
            "restore a possibly truncated or corrupted archive."
        )

    if os.path.lexists(target_dir):
        if not os.path.isdir(target_dir):
            raise FileExistsError(f"Restore target exists and is not a directory: {target_dir}")
        if os.listdir(target_dir):
            raise FileExistsError(
                f"Restore target is not empty: {target_dir}. Restore into an empty "
                "or nonexistent directory, then move it into place yourself."
            )
    else:
        os.makedirs(target_dir, exist_ok=True)

    log(f"  Restore: checksum verified. Unpacking into {target_dir}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        _safe_extract_all(tar, target_dir)

    log(f"  Restore: unpacked {manifest.get('file_count', '?')} file(s) into {target_dir}")

    return {
        "target_dir": target_dir,
        "manifest": manifest,
        "archive_sha256": actual_sha256,
    }
