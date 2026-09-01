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
archive directory listings, HNSW/sqlite counts, and legacy KG files opened
read-only via ``migrate_knowledge_graph(..., apply=False)``.
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
            )

        return _ok(
            name,
            f"codex capture {st['state']}, hooks.auto_save on, "
            f"last diary checkpoint {diary_age_hours:.0f}h ago",
            codex_state=st["state"],
            auto_save=auto_save,
            mine_transcript=mine_transcript,
            diary_age_hours=diary_age_hours,
        )
    except Exception as exc:  # pragma: no cover
        return _exc_result(name, exc, "check codex-status / config.json manually")


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
        legacy_triples = result.get("legacy_triples", 0)
        legacy_entities = result.get("legacy_entities", 0)
        if not legacy_triples and not legacy_entities:
            return _ok(name, "no legacy knowledge graph to migrate")
        return _warn(
            name,
            f"legacy KG at {result.get('legacy_path')} has {legacy_triples} triple(s) / "
            f"{legacy_entities} entitie(s) not yet in the palace KG",
            "run `mempalace kg-migrate --apply` to merge it in",
            legacy_triples=legacy_triples,
            legacy_entities=legacy_entities,
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
)


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
