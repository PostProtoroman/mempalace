"""doctor.py — ``mempalace doctor``: one command that catches every silent
failure this stack has had.

Each check returns a :class:`CheckResult` (``ok`` / ``warn`` / ``fail`` +
one-line reason + a remediation hint) and never raises — a health check that
crashes is worse than useless, since it hides whatever it was checking for
behind a traceback instead of a status line. Every exception any check can
throw is caught at the check's own boundary and turned into a ``fail`` (or
``warn`` where the underlying condition is genuinely optional, e.g. macOS-only
launchd drift on other platforms) with the exception text as the reason.

Checks never touch live state. They only read: sampled drawer documents,
heartbeat files, launchd plists (read-only — never load/unload), backup
archive directory listings, HNSW/sqlite counts, legacy KG files opened
read-only via ``migrate_knowledge_graph(..., apply=False)``, a read-only
``launchctl print`` of the hub job, and a non-blocking probe of the palace
writer lease's flock (acquired and immediately released — never held).
"""

from __future__ import annotations

import os
import platform
import plistlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

DEFAULT_SAMPLE_DRAWERS = 5000
STALE_CAPTURE_WARN_HOURS = 48
STALE_BACKUP_WARN_HOURS = 48
STALE_DIARY_WARN_HOURS = 48

# Historical agent_name values filed before diary entries were routed under
# a real per-agent identity. No identity-lane config exists in this tree yet
# (searched for ``identity-migrate`` / ``identity_migrate`` — nothing found),
# so this is the hardcoded fallback the lane brief calls for. Replace with a
# read from the identity lane's config once that lane lands.
KNOWN_ALIAS_AGENT_NAMES = ("assistant", "claude", "checkpoint")


@dataclass
class CheckResult:
    name: str
    status: str  # "ok" | "warn" | "fail"
    reason: str
    hint: str = ""
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "hint": self.hint,
            "details": self.details,
        }


def _ok(name, reason, **details):
    return CheckResult(name, "ok", reason, details=details)


def _warn(name, reason, hint, **details):
    return CheckResult(name, "warn", reason, hint=hint, details=details)


def _fail(name, reason, hint, **details):
    return CheckResult(name, "fail", reason, hint=hint, details=details)


def _exc_result(name, exc, hint):
    return _fail(name, f"{type(exc).__name__}: {exc}", hint)


def _hours_since(iso_ts: str) -> "float | None":
    """Hours between now and an ISO-8601 timestamp, or None if unparseable."""
    if not iso_ts:
        return None
    try:
        ts = iso_ts
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return (now - dt).total_seconds() / 3600.0
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# 1. Secrets — sample-scan drawer documents for redact.py's patterns.
# ---------------------------------------------------------------------------


def check_secrets(
    palace_path: str, *, full: bool = False, sample_n: int = DEFAULT_SAMPLE_DRAWERS
) -> CheckResult:
    from .redact import redact_secrets

    from .backends.base import PalaceNotFoundError

    name = "secrets"
    try:
        from .palace import get_collection

        try:
            col = get_collection(palace_path, create=False, read_only=True)
        except PalaceNotFoundError:
            col = None
        if col is None:
            return _ok(name, "no palace found — nothing to scan", scanned=0)

        scanned = 0
        hits: dict[str, int] = {}
        hit_ids: list[str] = []
        offset = 0
        batch_size = 1000
        limit = None if full else sample_n
        while True:
            if limit is not None and scanned >= limit:
                break
            take = batch_size if limit is None else min(batch_size, limit - scanned)
            batch = col.get(limit=take, offset=offset, include=["documents"])
            ids = batch.get("ids") or []
            docs = batch.get("documents") or []
            if not ids:
                break
            for drawer_id, doc in zip(ids, docs):
                scanned += 1
                if not doc:
                    continue
                _redacted, counts = redact_secrets(doc)
                if counts:
                    for kind, n in counts.items():
                        hits[kind] = hits.get(kind, 0) + n
                    if len(hit_ids) < 20:
                        hit_ids.append(drawer_id)
            offset += len(ids)
            if len(ids) < take:
                break

        if hits:
            total = sum(hits.values())
            breakdown = ", ".join(f"{k}:{v}" for k, v in sorted(hits.items()))
            return _fail(
                name,
                f"{total} unredacted secret(s) found in {scanned} scanned drawer(s) ({breakdown})",
                "review the flagged drawers and redact/delete them by hand — "
                "ingest-time redaction only protects new writes; run with "
                "--full to scan every drawer instead of a sample",
                scanned=scanned,
                counts=hits,
                sample_drawer_ids=hit_ids,
                full_scan=full,
            )
        mode = "full palace" if full else f"first {sample_n} drawers"
        return _ok(
            name, f"no secrets found ({mode}, {scanned} scanned)", scanned=scanned, full_scan=full
        )
    except Exception as exc:  # pragma: no cover - defensive, see module docstring
        return _exc_result(name, exc, "check palace path/permissions and retry")


# ---------------------------------------------------------------------------
# 2. Capture liveness — codex follower heartbeat, hook config, diary age.
# ---------------------------------------------------------------------------


def check_capture_liveness(palace_path: str, sessions_root: "str | None" = None) -> CheckResult:
    from .codex_follower import DEFAULT_SESSIONS_ROOT
    from .codex_follower import status as codex_status
    from .config import MempalaceConfig

    name = "capture_liveness"
    try:
        root = sessions_root or DEFAULT_SESSIONS_ROOT
        st = codex_status(palace_path, root)

        cfg = MempalaceConfig()
        auto_save = cfg.hooks_auto_save
        mine_transcript = cfg.hooks_mine_transcript

        diary_age_hours = _latest_diary_age_hours(palace_path)
        lost_mines = _lost_mine_jobs(palace_path)

        problems = []
        if st["state"] == "BLOCKED":
            problems.append(f"codex capture BLOCKED ({st.get('last_error') or st.get('writer')})")
        if not auto_save:
            problems.append("hooks.auto_save is disabled — nothing is being captured")
        if problems:
            return _fail(
                name,
                "; ".join(problems),
                "run `mempalace codex-status` for details, or enable hooks.auto_save "
                "in ~/.mempalace/config.json",
                codex_state=st["state"],
                auto_save=auto_save,
                mine_transcript=mine_transcript,
                diary_age_hours=diary_age_hours,
            )

        warnings_ = []
        if st["state"] == "DEGRADED":
            warnings_.append(
                f"codex capture DEGRADED ({st.get('last_error') or 'follower not running'})"
            )
        if diary_age_hours is not None and diary_age_hours > STALE_DIARY_WARN_HOURS:
            warnings_.append(f"last diary checkpoint is {diary_age_hours:.0f}h old")
        elif diary_age_hours is None:
            warnings_.append("no diary checkpoint found yet")
        if lost_mines:
            # The hub's mine queue is in memory: a restart drops whatever it
            # had accepted, after the CLI reported the mine queued. The
            # journal is the only trace, so surface it here.
            warnings_.append(
                f"{len(lost_mines)} hub mine job(s) accepted but never finished "
                f"(e.g. {lost_mines[0].get('source')}) — a hub restart dropped them"
            )

        if warnings_:
            return _warn(
                name,
                "; ".join(warnings_),
                "run `mempalace codex-status`; file a diary checkpoint if the agent "
                "hasn't been active",
                codex_state=st["state"],
                auto_save=auto_save,
                mine_transcript=mine_transcript,
                diary_age_hours=diary_age_hours,
                lost_mine_jobs=len(lost_mines),
            )

        return _ok(
            name,
            f"codex capture {st['state']}, hooks.auto_save on, "
            f"last diary checkpoint {diary_age_hours:.0f}h ago",
            codex_state=st["state"],
            auto_save=auto_save,
            mine_transcript=mine_transcript,
            diary_age_hours=diary_age_hours,
            lost_mine_jobs=0,
        )
    except Exception as exc:  # pragma: no cover
        return _exc_result(name, exc, "check codex-status / config.json manually")


def _lost_mine_jobs(palace_path: str) -> list:
    """Hub mine jobs the journal accepted but never closed out.

    Best-effort and never fatal: `mempalace doctor` must still run on a palace
    that was never mined by a hub, where the journal does not exist.
    """
    import os

    from .mine_queue import JOURNAL_FILENAME, unfinished_jobs

    try:
        return unfinished_jobs(os.path.join(os.path.expanduser(palace_path), JOURNAL_FILENAME))
    except Exception:  # pragma: no cover - diagnostics must not fail the check
        return []


def _latest_diary_age_hours(palace_path: str) -> "float | None":
    from .backends.base import PalaceNotFoundError
    from .palace import get_collection

    try:
        try:
            col = get_collection(palace_path, create=False, read_only=True)
        except PalaceNotFoundError:
            col = None
        if col is None:
            return None
        latest = ""
        offset = 0
        batch_size = 1000
        while True:
            batch = col.get(
                where={"type": "diary_entry"},
                limit=batch_size,
                offset=offset,
                include=["metadatas"],
            )
            metas = batch.get("metadatas") or []
            if not metas:
                break
            for m in metas:
                ts = (m or {}).get("filed_at") or ""
                if ts > latest:
                    latest = ts
            offset += len(metas)
            if len(metas) < batch_size:
                break
        if not latest:
            return None
        return _hours_since(latest)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 3. Launchd drift (macOS only) — ProgramArguments must point inside the
#    canonical checkout this running package was imported from.
# ---------------------------------------------------------------------------


def _canonical_checkout_root() -> "str | None":
    """Best-effort: the checkout directory this running mempalace package
    lives in (parent of the ``mempalace/`` package dir), resolved through
    symlinks. Returns None if this doesn't look like an in-checkout install
    (e.g. installed into site-packages with no adjacent pyproject.toml)."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    checkout = os.path.dirname(pkg_dir)
    real = os.path.realpath(checkout)
    if os.path.exists(os.path.join(real, "pyproject.toml")):
        return real
    return None


def check_launchd_drift(home: "str | None" = None) -> CheckResult:
    name = "launchd_drift"
    if platform.system() != "Darwin":
        return _ok(name, "not macOS — skipped")

    try:
        agents_dir = os.path.join(home or os.path.expanduser("~"), "Library", "LaunchAgents")
        if not os.path.isdir(agents_dir):
            return _ok(name, "no ~/Library/LaunchAgents directory — nothing installed")

        expected_root = _canonical_checkout_root()
        plists = sorted(
            f
            for f in os.listdir(agents_dir)
            if f.startswith("ai.mempalace.") and f.endswith(".plist")
        )
        if not plists:
            return _ok(name, "no ai.mempalace.* launch agents installed")

        drifted = []
        checked = []
        for fname in plists:
            path = os.path.join(agents_dir, fname)
            try:
                with open(path, "rb") as f:
                    data = plistlib.load(f)
            except Exception as exc:
                drifted.append(f"{fname}: unreadable plist ({exc})")
                continue
            args = data.get("ProgramArguments") or []
            checked.append(fname)
            if expected_root is None:
                # Can't derive an expected checkout (not running from an
                # in-checkout install) — nothing to compare against.
                continue
            pointed_inside = any(
                os.path.realpath(os.path.expanduser(str(a))).startswith(expected_root + os.sep)
                or os.path.realpath(os.path.expanduser(str(a))) == expected_root
                for a in args
                if isinstance(a, str)
            )
            if not pointed_inside:
                drifted.append(f"{fname}: ProgramArguments do not point inside {expected_root}")

        if drifted:
            return _fail(
                name,
                "; ".join(drifted),
                "re-run the deploy/launchd install steps (see deploy/launchd/README.md) "
                "to regenerate the plist(s) from the canonical checkout",
                checked=checked,
            )
        if expected_root is None:
            return _warn(
                name,
                f"found {len(checked)} launch agent(s) but this mempalace isn't running "
                "from an in-checkout install — cannot verify their target",
                "verify manually with `launchctl print gui/$(id -u)/<label>`",
                checked=checked,
            )
        return _ok(
            name,
            f"{len(checked)} launch agent(s) all point inside {expected_root}",
            checked=checked,
        )
    except Exception as exc:  # pragma: no cover
        return _exc_result(name, exc, "inspect ~/Library/LaunchAgents manually")


# ---------------------------------------------------------------------------
# 4. Backup freshness.
# ---------------------------------------------------------------------------


def check_backup_freshness(dest_dir: "str | None" = None) -> CheckResult:
    from .backups import DEFAULT_BACKUP_DEST, _ARCHIVE_SUFFIX

    name = "backup_freshness"
    try:
        dest = os.path.expanduser(dest_dir) if dest_dir else DEFAULT_BACKUP_DEST
        if not os.path.isdir(dest):
            return _fail(
                name,
                f"no backup destination found at {dest}",
                "run `mempalace backup` (or install deploy/launchd/ai.mempalace.backup.plist)",
            )
        archives = [
            os.path.join(dest, f)
            for f in os.listdir(dest)
            if f.startswith("mempalace-") and f.endswith(_ARCHIVE_SUFFIX)
        ]
        if not archives:
            return _fail(
                name,
                f"no backup archives found in {dest}",
                "run `mempalace backup` (or install deploy/launchd/ai.mempalace.backup.plist)",
            )
        newest = max(archives, key=os.path.getmtime)
        age_hours = (time.time() - os.path.getmtime(newest)) / 3600.0
        if age_hours > STALE_BACKUP_WARN_HOURS:
            return _warn(
                name,
                f"newest backup is {age_hours:.0f}h old ({os.path.basename(newest)})",
                "run `mempalace backup` — or check the ai.mempalace.backup launchd agent is loaded",
                newest=newest,
                age_hours=age_hours,
            )
        return _ok(
            name,
            f"newest backup is {age_hours:.0f}h old ({os.path.basename(newest)})",
            newest=newest,
            age_hours=age_hours,
        )
    except Exception as exc:  # pragma: no cover
        return _exc_result(name, exc, "check the backup destination directory manually")


# ---------------------------------------------------------------------------
# 5. Search health — HNSW capacity/quarantine probe.
# ---------------------------------------------------------------------------


def check_search_health(
    palace_path: str, collection_name: str = "mempalace_drawers"
) -> CheckResult:
    name = "search_health"
    try:
        from .backends.chroma import hnsw_capacity_status

        info = hnsw_capacity_status(palace_path, collection_name)
        status = info.get("status")
        message = info.get("message") or ""
        if status == "diverged":
            return _fail(
                name,
                message or "HNSW capacity divergence detected",
                "run `mempalace repair` to rebuild the index, then re-check",
                **{k: v for k, v in info.items() if k != "message"},
            )
        if status == "unknown":
            return _warn(
                name,
                message or "search health could not be fully measured",
                "re-run once the palace is idle (no concurrent mine/search)",
                **{k: v for k, v in info.items() if k != "message"},
            )
        return _ok(
            name,
            message or "search index healthy",
            **{k: v for k, v in info.items() if k != "message"},
        )
    except Exception as exc:  # pragma: no cover
        return _exc_result(name, exc, "run `mempalace repair-status` for more detail")


# ---------------------------------------------------------------------------
# 6. KG unification — legacy sqlite vs palace KG.
# ---------------------------------------------------------------------------


def check_kg_unification(palace_path: str) -> CheckResult:
    import contextlib
    import io

    name = "kg_unification"
    try:
        from .migrate import migrate_knowledge_graph

        # migrate_knowledge_graph() prints a human-readable plan as a side
        # effect (existing `mempalace kg-migrate` CLI behavior) — swallow it
        # here so a probe running inside `doctor --json` doesn't leak stray
        # text before the JSON payload.
        with contextlib.redirect_stdout(io.StringIO()):
            result = migrate_knowledge_graph(palace_path=palace_path, apply=False)
        # NOTE: `legacy_triples`/`legacy_entities` are the TOTAL row counts in
        # the legacy sqlite file, not the counts still missing from the
        # palace KG -- migrate_knowledge_graph() already dedupes by row id
        # (see its `existing_triple_ids`/`existing_entity_ids` sets) and
        # reports the outstanding work in `merged_triples`/`merged_entities`.
        # Using the totals here made this check warn forever any time the
        # legacy file had *any* rows, even after a full `kg-migrate --apply`
        # (or, as observed, when `kg-migrate` dry-run already reports
        # "0 new, N already present"). Compare on the pending counts instead.
        pending_triples = result.get("merged_triples", 0)
        pending_entities = result.get("merged_entities", 0)
        if not pending_triples and not pending_entities:
            return _ok(name, "no legacy knowledge graph to migrate")
        return _warn(
            name,
            f"legacy KG at {result.get('legacy_path')} has {pending_triples} triple(s) / "
            f"{pending_entities} entitie(s) not yet in the palace KG",
            "run `mempalace kg-migrate --apply` to merge it in",
            legacy_triples=pending_triples,
            legacy_entities=pending_entities,
            legacy_path=result.get("legacy_path"),
            target_path=result.get("target_path"),
        )
    except Exception as exc:  # pragma: no cover
        return _exc_result(name, exc, "run `mempalace kg-migrate` manually to see the error")


# ---------------------------------------------------------------------------
# 7. Identity hygiene — diary entries filed under known-alias names.
# ---------------------------------------------------------------------------


def check_identity_hygiene(palace_path: str) -> CheckResult:
    from .backends.base import PalaceNotFoundError

    name = "identity_hygiene"
    try:
        from .palace import get_collection

        try:
            col = get_collection(palace_path, create=False, read_only=True)
        except PalaceNotFoundError:
            col = None
        if col is None:
            return _ok(name, "no palace found — nothing to check")

        alias_names = KNOWN_ALIAS_AGENT_NAMES
        counts: dict[str, int] = {}
        offset = 0
        batch_size = 1000
        while True:
            batch = col.get(
                where={"type": "diary_entry"},
                limit=batch_size,
                offset=offset,
                include=["metadatas"],
            )
            metas = batch.get("metadatas") or []
            if not metas:
                break
            for m in metas:
                agent = ((m or {}).get("agent") or "").lower()
                if agent in alias_names:
                    counts[agent] = counts.get(agent, 0) + 1
            offset += len(metas)
            if len(metas) < batch_size:
                break

        if counts:
            breakdown = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
            return _warn(
                name,
                f"diary entries filed under known-alias agent name(s): {breakdown}",
                "run `mempalace identity-migrate` once available, or re-file under the "
                "agent's real identity",
                counts=counts,
                checked_aliases=list(alias_names),
            )
        return _ok(
            name,
            "no diary entries under known-alias agent names",
            checked_aliases=list(alias_names),
        )
    except Exception as exc:  # pragma: no cover
        return _exc_result(name, exc, "inspect diary drawers' `agent` metadata manually")


# ---------------------------------------------------------------------------
# 8. Rogue lease holder (macOS only) — palace writer lease held by a
#    non-hub process while a hub launchd job is installed.
# ---------------------------------------------------------------------------


def _read_mine_lock_state(palace_path: str) -> dict:
    """Read `mine_palace_lock`'s current state for ``palace_path`` without
    ever acquiring or holding it.

    Tries a non-blocking exclusive flock on the same lock file
    `mine_palace_lock` uses: if it succeeds, the lease is free, and it is
    released immediately — this function never leaves the lock held. If it
    fails, the lease is held by someone else; the holder's identity
    (written by ``palace._write_lock_holder`` as ``"<pid> <argv0> <argv1>
    <argv2>"`` from byte 1 onward — byte 0 is the OS lock sentinel) is read
    back, along with the lock file's mtime as a best-effort proxy for how
    long it has been held (the holder truncates+writes+flushes that region
    right after acquiring, so mtime tracks the most recent acquire).

    POSIX only (``fcntl.flock``). The caller (``check_rogue_lease_holder``)
    already skips off-Darwin before reaching this function, but that guard
    is on ``platform.system()`` — mockable, and mocked by tests that force
    the Darwin path to run cross-platform. This function therefore carries
    its own, unmockable gate on ``os.name`` and never imports ``fcntl``
    (which does not exist on Windows) unless that gate has passed: on any
    non-POSIX platform it returns immediately with ``skipped`` set, and
    ``fcntl`` is imported lazily, after that check, only on the path that
    actually needs it.
    """
    from .palace import _LOCK_SENTINEL_BYTES, _format_lock_holder, mine_palace_lock_path

    lock_path = mine_palace_lock_path(palace_path)
    free = {
        "held": False,
        "pid": None,
        "cmdline": None,
        "age_hours": None,
        "holder_label": None,
        "lock_path": lock_path,
    }
    if os.name != "posix":
        return dict(free, skipped=True)
    if not os.path.exists(lock_path):
        return free

    import fcntl

    try:
        lf = open(lock_path, "rb")
    except FileNotFoundError:
        # Removed between the exists() check above and this open -- that's
        # still the "lease is free" case, not a failure.
        return free
    try:
        lf.seek(0)
        try:
            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lf, fcntl.LOCK_UN)
            return free
        except BlockingIOError:
            pass  # held by someone else -- read their recorded identity below

        lf.seek(_LOCK_SENTINEL_BYTES)
        content = lf.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        content = content.strip()

        pid = None
        cmdline = None
        if content:
            parts = content.split(maxsplit=1)
            if parts and parts[0].isdigit():
                pid = int(parts[0])
                cmdline = parts[1].strip() if len(parts) > 1 else ""

        age_hours = None
        try:
            age_hours = (time.time() - os.path.getmtime(lock_path)) / 3600.0
        except OSError:
            pass

        return {
            "held": True,
            "pid": pid,
            "cmdline": cmdline,
            "age_hours": age_hours,
            "holder_label": _format_lock_holder(content)
            if content
            else "another writer (identity not recorded)",
            "lock_path": lock_path,
        }
    finally:
        lf.close()


def _hub_launchd_pid(home: "str | None" = None) -> "int | None":
    """Best-effort PID of the running ``ai.mempalace.hub`` launchd job.

    A read-only ``launchctl print`` — never bootstrap/bootout/kickstart, in
    keeping with this module's "never touch live state" rule. Returns None
    on any failure (not macOS, no launchctl, job not loaded, unparseable
    output) rather than raising; the caller falls back to the
    ``--transport http`` command-line signal in that case.
    """
    import shutil
    import subprocess

    if platform.system() != "Darwin" or not shutil.which("launchctl"):
        return None
    try:
        domain = f"gui/{os.getuid()}"
        result = subprocess.run(
            ["launchctl", "print", f"{domain}/ai.mempalace.hub"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("pid = "):
                digits = line.split("=", 1)[1].strip()
                return int(digits) if digits.isdigit() else None
    except Exception:
        return None
    return None


def check_rogue_lease_holder(palace_path: str, home: "str | None" = None) -> CheckResult:
    """Warn when the palace writer lease (`mine_palace_lock`) is held by a
    process that is not the hub, while a hub launchd job is installed.

    Once ``ai.mempalace.hub`` is installed, the hub (``mempalace serve``,
    which execs into ``mempalace.mcp_server --transport http ...``) is meant
    to hold this lease for its entire lifetime — every other writer routes
    through it. If a hub restart races a session's own stdio MCP server for
    the non-blocking flock, that session's process can win and keep the
    lease past the restart, silently starving the hub of writes until that
    session's server exits. This check surfaces exactly that scenario —
    though not without touching the lock at all: it takes a non-blocking
    probe (acquire, immediately release) to tell "free" from "held", same as
    ``mine_palace_lock`` itself would. A `mempalace doctor` run that lands in
    the same microsecond as a hook-spawned ``mine``'s own non-blocking
    acquire can make that ``mine`` lose the race and drop its work
    (``MineAlreadyRunning``) — a window inherent to flock, not specific to
    this probe, but worth knowing before running doctor from a hot path.

    A holder is identified as the hub two ways: its recorded command line
    contains ``--transport http`` (the hub's own exec'd argv), or its PID
    matches the ``ai.mempalace.hub`` launchd job's PID. Either is enough,
    since the recorded command line is best-effort (only ``sys.argv[:3]``)
    and the launchd PID lookup can fail (job not loaded, no launchctl). Both
    signals trust the recorded PID's identity only while that PID is still
    alive: the lock body is written *after* the flock is acquired, so a
    holder caught in the gap between a new process's acquire and its own
    identity write still carries the *previous* holder's identity — if that
    previous holder has since exited, treating its stale record as "the hub"
    would clear an actual rogue.

    No hub plist installed -> single-writer deployments are normal -> OK.
    Lease free -> OK. Held by the hub -> OK. Held by anything else while a
    hub is installed -> WARN.
    """
    name = "rogue_lease_holder"
    if platform.system() != "Darwin":
        return _ok(name, "not macOS — skipped")

    try:
        agents_dir = os.path.join(home or os.path.expanduser("~"), "Library", "LaunchAgents")
        hub_plist = os.path.join(agents_dir, "ai.mempalace.hub.plist")
        if not os.path.isfile(hub_plist):
            return _ok(
                name, "no ai.mempalace.hub launch agent installed -- single-writer deployment"
            )

        state = _read_mine_lock_state(palace_path)
        if state.get("skipped"):
            return _ok(name, "flock unavailable on this platform -- skipped")
        if not state["held"]:
            return _ok(name, "palace writer lease is free")

        from .server_registry import _pid_alive

        pid = state["pid"]
        cmdline = state["cmdline"] or ""
        age_hours = state["age_hours"]
        holder_label = state["holder_label"]

        pid_alive = pid is not None and _pid_alive(pid)
        hub_pid = _hub_launchd_pid(home=home)
        is_hub = pid_alive and (
            "--transport http" in cmdline or (hub_pid is not None and pid == hub_pid)
        )

        if is_hub:
            return _ok(
                name,
                f"palace writer lease held by the hub ({holder_label})",
                pid=pid,
                cmdline=cmdline,
            )

        if pid is not None and not pid_alive:
            holder_label = (
                f"{holder_label} -- recorded PID {pid} is no longer running, identity is stale"
            )

        age_str = f"; held for {age_hours:.1f}h" if age_hours is not None else ""
        return _warn(
            name,
            f"palace writer lease is held by {holder_label}, not the hub{age_str}",
            "a session's stdio server seized the writer lease during a hub restart; "
            "it releases on exit -- or restart that session's MCP server",
            pid=pid,
            cmdline=cmdline,
            age_hours=age_hours,
            hub_pid=hub_pid,
        )
    except Exception as exc:  # pragma: no cover
        return _exc_result(name, exc, "inspect ~/.mempalace/locks manually")


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------

CHECK_ORDER = (
    "secrets",
    "capture_liveness",
    "launchd_drift",
    "backup_freshness",
    "search_health",
    "kg_unification",
    "identity_hygiene",
    "rogue_lease_holder",
    "upstream_drift",
)


def check_upstream_drift(checkout: "str | None" = None) -> CheckResult:
    """How far the deployed checkout has drifted from the upstream remote.

    Informational by design: behind-upstream is a WARN naming the update
    runbook, never a FAIL — updates here are supervised (live deployment,
    version-guarded services), and this check exists so drift is visible in
    the same place as every other health signal. Skips cleanly when there
    is no git checkout or no ``upstream`` remote.
    """
    import subprocess

    name = "upstream_drift"
    root = checkout or _canonical_checkout_root()
    if not root or not os.path.isdir(os.path.join(root, ".git")):
        return _ok(name, "no git checkout to compare (skipped)")
    try:
        remotes = subprocess.run(
            ["git", "-C", root, "remote"], capture_output=True, text=True, timeout=10
        ).stdout.split()
        if "upstream" not in remotes:
            return _ok(name, "no 'upstream' remote configured (skipped)")
        # Compare against the last *fetched* state — doctor must stay fast
        # and offline; refreshing the ref is the update runbook's job.
        behind = subprocess.run(
            ["git", "-C", root, "rev-list", "--count", "HEAD..upstream/develop"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if behind.returncode != 0:
            return _ok(name, "upstream/develop not fetched yet (skipped)")
        n = int(behind.stdout.strip() or "0")
        if n == 0:
            return _ok(name, "checkout is up to date with last-fetched upstream/develop")
        return _warn(
            name,
            f"checkout is {n} commit(s) behind last-fetched upstream/develop",
            "run the update runbook: git fetch upstream && merge the release tag, "
            "test, restart services",
            behind=n,
        )
    except Exception as exc:  # noqa: BLE001 - doctor checks never raise
        return _exc_result(name, exc, "run `git fetch upstream` by hand to inspect")


def run_all(
    palace_path: str,
    *,
    full_secrets_scan: bool = False,
    sample_n: int = DEFAULT_SAMPLE_DRAWERS,
    sessions_root: "str | None" = None,
    backup_dest: "str | None" = None,
    home: "str | None" = None,
) -> list[CheckResult]:
    return [
        check_secrets(palace_path, full=full_secrets_scan, sample_n=sample_n),
        check_capture_liveness(palace_path, sessions_root=sessions_root),
        check_launchd_drift(home=home),
        check_backup_freshness(backup_dest),
        check_search_health(palace_path),
        check_kg_unification(palace_path),
        check_identity_hygiene(palace_path),
        check_rogue_lease_holder(palace_path, home=home),
        check_upstream_drift(),
    ]


def overall_exit_code(results: "list[CheckResult]") -> int:
    return 1 if any(r.status == "fail" for r in results) else 0


def format_table(results: "list[CheckResult]") -> str:
    lines = []
    icon = {"ok": "OK", "warn": "WARN", "fail": "FAIL"}
    width = max(len(r.name) for r in results) if results else 0
    for r in results:
        lines.append(f"  [{icon[r.status]:>4}] {r.name:<{width}}  {r.reason}")
        if r.hint and r.status != "ok":
            lines.append(f"         {'':<{width}}  -> {r.hint}")
    return "\n".join(lines)
