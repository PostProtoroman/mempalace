"""Incremental follower for Codex CLI sessions.

Watches ``~/.codex/sessions/**/rollout-*.jsonl`` and files every completed
conversation turn into the palace as it appears, so Codex conversations are
captured without the user remembering to mine anything.

Cursor model — the palace *is* the cursor (RFC 002 goal 4, no sidecar):

* Every turn gets a deterministic drawer id derived from (thread id, item
  id), so writing the same turn twice is one upsert of one row.
* Before writing a batch the follower asks the collection which of those
  ids already exist. Rows that existed are counted as ``already_present``,
  never as ``added``. That is what makes "restart and rescan adds zero
  drawers" a measurable property rather than a hope.
* Per rollout, the highest ordinal already stored acts as a skip cursor so
  a long session is not re-parsed from the top on every pass. Correctness
  never depends on it: deleting the cursor only costs work, never
  duplicates.

Restart, resume, fork and compaction all reduce to the same case. Rollouts
are append-only, so an id minted for a turn keeps naming that turn; a
compaction appends a ``compacted`` record that this path never reads.

Writes go through one ``mine_palace_lock``, so Chroma writes stay
serialized and local.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .sources.codex import (
    ADAPTER_VERSION,
    classify_rollout,
    iter_rollouts,
    iter_turns,
    turn_to_drawer,
)

logger = logging.getLogger(__name__)

DEFAULT_SESSIONS_ROOT = os.path.join("~", ".codex", "sessions")
DEFAULT_INTERVAL_S = 15.0
BATCH_SIZE = 64

# After this many consecutive failed passes the follower escalates: an ERROR
# log line per failing pass, one logstream status event per outage, and an
# unhealthy flag in the heartbeat that `status` surfaces. The 2026-09-01
# outage failed silently every 15s for hours; this makes that impossible.
ESCALATION_THRESHOLD = 4

# Marker written next to the palace so `status` can prove a follower is live
# without guessing from the process table.
_STATE_DIRNAME = "codex_capture"


def state_dir(palace_path: str) -> Path:
    d = Path(palace_path).expanduser() / _STATE_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def heartbeat_path(palace_path: str) -> Path:
    return state_dir(palace_path) / "follower.json"


def _write_heartbeat(palace_path: str, **fields) -> None:
    payload = {"pid": os.getpid(), "updated_at": datetime.now(timezone.utc).isoformat(), **fields}
    try:
        heartbeat_path(palace_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:  # pragma: no cover - heartbeat is best effort
        logger.warning("codex follower: could not write heartbeat: %s", exc)


def read_heartbeat(palace_path: str) -> Optional[dict]:
    try:
        return json.loads(heartbeat_path(palace_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def activation_path(palace_path: str) -> Path:
    return state_dir(palace_path) / "activation.json"


def read_activation(palace_path: str) -> Optional[dict]:
    try:
        return json.loads(activation_path(palace_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def ensure_activation(palace_path: str, sessions_root: str) -> dict:
    """Record the activation point, once.

    Capture starts from *now*, not from the beginning of history. Installing
    the service on a machine with years of Codex sessions must not quietly
    import all of them — that is a large, surprising write the user never
    asked for, and it is the kind of thing that should be a decision rather
    than a side effect of turning capture on.

    The baseline is recorded as a per-rollout high-water ordinal rather than a
    wall-clock timestamp: rollout timestamps come from the file and clock skew
    or a resumed session would make a time cutoff wrong. Files that appear
    *after* activation have no baseline and are captured in full, which is
    exactly the "new sessions" case.
    """
    existing = read_activation(palace_path)
    if existing:
        return existing

    baseline: dict[str, int] = {}
    for rollout in iter_rollouts(sessions_root):
        highest = -1
        try:
            # Baseline automation sessions too, even though capture skips
            # them by default: opting in later must capture new automation
            # turns, not silently import that history in full.
            for turn in iter_turns(str(rollout), include_automation=True):
                if turn.ordinal > highest:
                    highest = turn.ordinal
        except OSError:
            continue
        if highest >= 0:
            baseline[rollout.stem] = highest

    record = {
        "activated_at": datetime.now(timezone.utc).isoformat(),
        "sessions_root": str(sessions_root),
        "baseline": baseline,
        "rollouts_at_activation": len(baseline),
    }
    try:
        activation_path(palace_path).write_text(json.dumps(record, indent=2), encoding="utf-8")
    except OSError as exc:  # pragma: no cover
        logger.warning("codex follower: could not record activation point: %s", exc)
    return record


def _baseline_for(activation: Optional[dict], rollout_stem: str) -> Optional[int]:
    if not activation:
        return None
    value = (activation.get("baseline") or {}).get(rollout_stem)
    return value if isinstance(value, int) else None


def plan_backfill(
    palace_path: str,
    sessions_root: str = DEFAULT_SESSIONS_ROOT,
    *,
    include_automation: bool = False,
) -> dict:
    """What a ``--backfill`` would import, without writing anything.

    ``include_automation`` must mirror the flag the real backfill will run
    with — otherwise the plan and the import diverge on automation files.
    """
    from .palace import get_collection

    try:
        collection = get_collection(palace_path, create=False)
    except Exception:
        collection = None

    files = 0
    turns = 0
    already = 0
    automations = 0
    subagent_files = 0
    subagent_turns = 0
    destinations: dict[str, int] = {}
    threads: set[str] = set()

    for rollout in iter_rollouts(sessions_root):
        files += 1
        info = classify_rollout(str(rollout))
        if info.get("automation") and not include_automation:
            automations += 1
            continue
        is_subagent = info.get("thread_source") == "subagent"
        if is_subagent:
            subagent_files += 1
        try:
            rollout_turns = list(iter_turns(str(rollout), include_automation=include_automation))
        except OSError:
            continue
        ids = [t.drawer_id for t in rollout_turns]
        present: set[str] = set()
        if collection is not None and ids:
            try:
                found = collection.get(ids=ids, include=[])
                present = set(found.get("ids") or [])
            except Exception:
                present = set()
        for turn in rollout_turns:
            turns += 1
            threads.add(turn.thread_id)
            if is_subagent:
                subagent_turns += 1
            if turn.drawer_id in present:
                already += 1
                continue
            record = turn_to_drawer(turn, str(rollout))
            key = f"{record.metadata.get('wing')}/{record.metadata.get('room')}"
            destinations[key] = destinations.get(key, 0) + 1

    return {
        "files_scanned": files,
        "turns_found": turns,
        "estimated_new_drawers": sum(destinations.values()),
        "already_present": already,
        "automation_files_excluded": automations,
        "subagent_files_included": subagent_files,
        "subagent_turns_included": subagent_turns,
        "threads_seen": len(threads),
        "destinations": dict(sorted(destinations.items(), key=lambda kv: -kv[1])),
    }


def _stored_max_ordinal(collection, source_file: str) -> Optional[int]:
    """Highest ordinal already stored for a rollout, or None.

    A backend error degrades to None — meaning "re-read the file" — which
    costs a rescan but can never duplicate, because the drawer ids are
    deterministic.
    """
    try:
        data = collection.get(where={"source_file": source_file}, include=["metadatas"])
    except Exception as exc:
        logger.warning("codex follower: cursor lookup failed for %s (%s)", source_file, exc)
        return None
    metas = data.get("metadatas") or []
    ordinals = [m.get("ordinal") for m in metas if m and isinstance(m.get("ordinal"), int)]
    return max(ordinals) if ordinals else None


def scan_once(
    palace_path: str,
    sessions_root: str = DEFAULT_SESSIONS_ROOT,
    *,
    collection=None,
    max_turns: Optional[int] = None,
    backfill: bool = False,
    include_automation: bool = False,
) -> dict:
    """One incremental pass over every rollout. Returns write statistics.

    ``backfill=False`` (the default) honours the activation point: turns that
    were already in a rollout when capture was switched on are left alone, so
    installing the service captures new turns rather than silently importing
    all of history. ``backfill=True`` ignores the baseline and imports
    everything.

    ``include_automation=False`` (the default) skips non-interactive ``codex
    exec`` sessions — machine output, not conversation. Opting in routes them
    to the dedicated automation wing, never the user's.

    ``max_turns`` bounds how many turns one call will write. It exists for the
    hub path: an unbounded first scan of a large corpus would hold the hub's
    write path for minutes, which is the starvation mode this transport
    already has history with. When the budget runs out the pass stops early
    and reports ``pending: True`` so the caller comes straight back.
    """
    from .palace import get_collection
    from .sources.codex import SUBAGENT_THREAD_SOURCES

    own_collection = collection is None
    if own_collection:
        collection = get_collection(palace_path, create=True)

    from .config import MempalaceConfig
    from .redact import redact_if_enabled

    redaction_enabled = MempalaceConfig().redaction_enabled
    redaction_counts: dict[str, int] = {}

    activation = None if backfill else read_activation(palace_path)
    added = already = skipped = 0
    before_activation = 0
    files_scanned = 0
    files_automation_skipped = 0
    files_unparseable = 0
    subagent_turns_routed = 0
    automation_turns_captured = 0
    written = 0
    pending = False
    threads: set[str] = set()

    batch_ids: list[str] = []
    batch_docs: list[str] = []
    batch_metas: list[dict] = []

    def _flush():
        nonlocal added, already
        if not batch_ids:
            return
        deduped: dict[str, tuple[str, dict]] = {}
        for drawer_id, doc, meta in zip(batch_ids, batch_docs, batch_metas):
            deduped[drawer_id] = (doc, meta)
        ids = list(deduped)
        docs = [redact_if_enabled(deduped[i][0], redaction_enabled, redaction_counts) for i in ids]
        metas = [deduped[i][1] for i in ids]
        try:
            existing = collection.get(ids=ids, include=[])
            present = set(existing.get("ids") or [])
        except Exception as exc:
            logger.warning("codex follower: existence pre-check failed (%s)", exc)
            present = set()
        new_count = sum(1 for i in ids if i not in present)
        collection.upsert(ids=ids, documents=docs, metadatas=metas)
        added += new_count
        already += len(ids) - new_count
        batch_ids.clear()
        batch_docs.clear()
        batch_metas.clear()

    for rollout in iter_rollouts(sessions_root):
        if pending:
            break
        files_scanned += 1
        source_file = str(rollout)
        # The cursor is a "don't re-read what we already passed" optimisation
        # for forward capture. A backfill is deliberately going *backwards*
        # into history, so honouring it there would skip exactly the turns the
        # user asked to import.
        cursor = None if backfill else _stored_max_ordinal(collection, source_file)
        baseline = _baseline_for(activation, rollout.stem)
        report: dict = {}
        try:
            turns = iter_turns(source_file, include_automation=include_automation, report=report)
            for turn in turns:
                if baseline is not None and turn.ordinal <= baseline:
                    # Predates activation; only --backfill imports these.
                    before_activation += 1
                    continue
                if cursor is not None and turn.ordinal < cursor:
                    skipped += 1
                    continue
                record = turn_to_drawer(turn, source_file)
                meta = dict(record.metadata)
                meta["source_file"] = source_file
                meta["adapter_name"] = "codex"
                meta["adapter_version"] = ADAPTER_VERSION
                meta["filed_at"] = datetime.now(timezone.utc).isoformat()
                threads.add(turn.thread_id)
                if turn.thread_source in SUBAGENT_THREAD_SOURCES:
                    subagent_turns_routed += 1
                elif turn.automation:
                    automation_turns_captured += 1
                batch_ids.append(turn.drawer_id)
                batch_docs.append(record.content)
                batch_metas.append(meta)
                written += 1
                if len(batch_ids) >= BATCH_SIZE:
                    _flush()
                if max_turns is not None and written >= max_turns:
                    # Budget spent. Stop cleanly rather than truncating a
                    # batch: the cursor and deterministic ids make the next
                    # call resume exactly here without duplicating.
                    pending = True
                    break
        except OSError as exc:
            logger.warning("codex follower: unreadable rollout %s (%s)", source_file, exc)
            continue
        if report.get("automation_skipped"):
            files_automation_skipped += 1
        elif (
            not pending
            and report.get("content_lines", 0) > 0
            and not report.get("recognized_entries", 0)
        ):
            # The file has content but the parser understood none of it.
            # This path never degrades to raw-text chunking — a previous
            # deployment filed 12 such files as raw-JSONL garbage drawers —
            # so the only correct moves are: say so, and count it.
            files_unparseable += 1
            logger.warning(
                "codex follower: unparseable rollout %s — %d non-empty line(s), "
                "no recognized entries; skipping (never ingested as raw text)",
                source_file,
                report.get("content_lines", 0),
            )

    _flush()

    if redaction_counts:
        logger.warning(
            "codex follower: redacted %d secret(s) before write: %s",
            sum(redaction_counts.values()),
            dict(sorted(redaction_counts.items())),
        )
    return {
        "files_scanned": files_scanned,
        "files_automation_skipped": files_automation_skipped,
        "files_unparseable": files_unparseable,
        "threads_seen": len(threads),
        "drawers_added": added,
        "drawers_already_present": already,
        "secrets_redacted": sum(redaction_counts.values()),
        "subagent_turns_routed": subagent_turns_routed,
        "automation_turns_captured": automation_turns_captured,
        "turns_skipped_by_cursor": skipped,
        "turns_before_activation": before_activation,
        "backfill": backfill,
        "include_automation": include_automation,
        "pending": pending,
    }


def capture_pass(
    palace_path: str,
    sessions_root: str = DEFAULT_SESSIONS_ROOT,
    *,
    backfill: bool = False,
    include_automation: bool = False,
) -> dict:
    """One capture pass, routed to whichever writer legitimately owns the palace.

    Exactly one of two paths runs, never both:

    * A live, writable hub owns the lease -> hand it the work over HTTP. This
      process never touches the palace lock in that case.
    * No hub -> take ``mine_palace_lock`` and write directly.

    A hub that answered and then failed is *not* retried locally: it may have
    applied the write, and a local retry would be a second writer. The failure
    is returned so status can show it, and the next pass tries again — no turn
    is dropped, because unwritten turns are still in the rollout and the
    cursor never advanced past them.
    """
    from .codex_hub import HubRefused, HubUnavailable, capture_via_hub

    try:
        stats = capture_via_hub(
            palace_path, sessions_root, backfill=backfill, include_automation=include_automation
        )
        stats.setdefault("writer", "hub")
        stats["error"] = ""
        return stats
    except HubUnavailable as exc:
        logger.debug("codex follower: no hub (%s); writing directly", exc)
    except HubRefused as exc:
        logger.error("codex follower: hub refused capture (%s)", exc)
        return {
            "writer": "hub",
            "error": str(exc),
            "drawers_added": 0,
            "drawers_already_present": 0,
            "turns_skipped_by_cursor": 0,
            "files_scanned": 0,
            "threads_seen": 0,
        }

    from .palace import MineAlreadyRunning, mine_palace_lock

    try:
        with mine_palace_lock(palace_path):
            stats = scan_once(
                palace_path,
                sessions_root,
                backfill=backfill,
                include_automation=include_automation,
            )
    except MineAlreadyRunning as exc:
        # Another writer holds the lease but published no hub endpoint we can
        # reach. Refusing is correct — two writers is the one thing we must
        # never do — and the turns stay in the rollout for the next pass.
        return {
            "writer": "blocked",
            "error": f"palace writer lease held by another process: {exc}",
            "drawers_added": 0,
            "drawers_already_present": 0,
            "turns_skipped_by_cursor": 0,
            "files_scanned": 0,
            "threads_seen": 0,
        }
    stats["writer"] = "direct"
    stats["error"] = ""
    return stats


def run_once(
    palace_path: str,
    sessions_root: str = DEFAULT_SESSIONS_ROOT,
    *,
    backfill: bool = False,
    include_automation: bool = False,
) -> dict:
    """One capture pass plus a heartbeat, for `codex-capture --once`."""
    stats = capture_pass(
        palace_path, sessions_root, backfill=backfill, include_automation=include_automation
    )
    _write_heartbeat(palace_path, sessions_root=str(sessions_root), last_scan=stats, running=False)
    return stats


def _announce_outage(palace_path: str, error: str, writer: str, consecutive: int) -> bool:
    """Append one logstream status event announcing a capture outage.

    Written straight into ``logstream.sqlite3`` rather than through the hub:
    the hub is exactly the component that is failing when this fires, and an
    alarm routed through the failing component is no alarm. Direct SQLite
    access is safe alongside a running hub — the logstream is WAL-mode and
    independent of the Chroma writer lease (the same pattern ``cli.py``'s
    ``_open_logstream`` documents).

    Returns True when the event was appended, so the caller announces once
    per outage and retries next pass if even this write failed.
    """
    try:
        from .logstream import LOGSTREAM_DB_FILENAME, Logstream

        ls = Logstream(db_path=os.path.join(os.path.expanduser(palace_path), LOGSTREAM_DB_FILENAME))
        ls.append_event(
            type="status",
            stream="service/codex-capture",
            room="ops",
            from_agent="codex-follower",
            to_agent="*",
            body=(
                f"codex capture outage: {consecutive} consecutive failed passes "
                f"(writer={writer or 'hub'}, tool=mempalace_codex_capture). "
                f"Last error: {error}"
            ),
        )
        return True
    except Exception as exc:
        logger.error("codex follower: could not announce outage on the logstream: %s", exc)
        return False


def follow(
    palace_path: str,
    sessions_root: str = DEFAULT_SESSIONS_ROOT,
    interval: float = DEFAULT_INTERVAL_S,
    *,
    max_passes: Optional[int] = None,
    include_automation: bool = False,
    escalation_threshold: int = ESCALATION_THRESHOLD,
) -> dict:
    """Scan on an interval until interrupted (or ``max_passes`` is reached).

    Failure escalation: ``escalation_threshold`` consecutive failed passes
    (default 4) flips the loop into a declared outage — every further failing
    pass logs at ERROR, one status event is appended to the logstream
    (``type=status``, ``to_agent=*``) per outage, and the heartbeat carries
    ``consecutive_failures`` so ``status`` shows unhealthy. A clean pass ends
    the outage and re-arms the announcement.
    """
    passes = 0
    consecutive_failures = 0
    outage_announced = False
    totals = {"drawers_added": 0, "drawers_already_present": 0, "passes": 0}
    try:
        while max_passes is None or passes < max_passes:
            stats = capture_pass(palace_path, sessions_root, include_automation=include_automation)
            passes += 1
            error = str(stats.get("error") or "")
            if error:
                consecutive_failures += 1
                if consecutive_failures >= escalation_threshold:
                    logger.error(
                        "codex follower: %d consecutive failed passes via %s: %s",
                        consecutive_failures,
                        stats.get("writer") or "hub",
                        error,
                    )
                    if not outage_announced:
                        outage_announced = _announce_outage(
                            palace_path, error, str(stats.get("writer") or ""), consecutive_failures
                        )
            else:
                if outage_announced:
                    logger.info(
                        "codex follower: capture recovered after %d failed pass(es)",
                        consecutive_failures,
                    )
                consecutive_failures = 0
                outage_announced = False
            totals["drawers_added"] += stats["drawers_added"]
            totals["drawers_already_present"] += stats["drawers_already_present"]
            totals["passes"] = passes
            _write_heartbeat(
                palace_path,
                sessions_root=str(sessions_root),
                last_scan=stats,
                passes=passes,
                interval=interval,
                running=True,
                consecutive_failures=consecutive_failures,
                outage_announced=outage_announced,
            )
            if max_passes is not None and passes >= max_passes:
                break
            time.sleep(interval)
    except KeyboardInterrupt:  # pragma: no cover - interactive stop
        pass
    finally:
        _write_heartbeat(
            palace_path,
            sessions_root=str(sessions_root),
            passes=passes,
            running=False,
            consecutive_failures=consecutive_failures,
            outage_announced=outage_announced,
        )
    return totals


def _lag_seconds(latest_turn_at: str, heartbeat_at: str) -> Optional[float]:
    """Seconds between the newest captured turn and the last successful pass.

    This is capture lag: how far behind the palace is, not how long ago the
    follower ran. Unparseable timestamps yield None rather than a wrong number.
    """
    from datetime import datetime as _dt

    def _parse(value: str):
        if not value:
            return None
        try:
            return _dt.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    turn = _parse(latest_turn_at)
    beat = _parse(heartbeat_at)
    if turn is None or beat is None:
        return None
    if turn.tzinfo is None:
        turn = turn.replace(tzinfo=timezone.utc)
    if beat.tzinfo is None:
        beat = beat.replace(tzinfo=timezone.utc)
    return max(0.0, (beat - turn).total_seconds())


def status(palace_path: str, sessions_root: str = DEFAULT_SESSIONS_ROOT) -> dict:
    """Capture health, as one of ACTIVE / DEGRADED / BLOCKED.

    * ``ACTIVE``   — the follower is running and its last pass wrote cleanly.
    * ``DEGRADED`` — capture is configured and the palace has turns, but the
      follower is not running, or its last pass reported an error it can
      recover from on the next attempt.
    * ``BLOCKED``  — the last pass could not write at all: another process
      owns the writer lease with no reachable hub, or the hub refused. Turns
      are not lost, but nothing is being captured until it clears. A follower
      that has hit :data:`ESCALATION_THRESHOLD` consecutive failed passes is
      also BLOCKED, with ``unhealthy: True`` and the failure count surfaced.
    """
    from .codex_hub import hub_is_healthy, hub_target

    hb = read_heartbeat(palace_path) or {}
    pid = hb.get("pid")
    running = bool(pid) and _pid_alive(int(pid)) and bool(hb.get("running"))
    last = hb.get("last_scan") or {}
    last_error = str(last.get("error") or "")
    writer = str(last.get("writer") or "")
    consecutive_failures = hb.get("consecutive_failures")
    if not isinstance(consecutive_failures, int):
        consecutive_failures = 0
    unhealthy = consecutive_failures >= ESCALATION_THRESHOLD

    target = hub_target(palace_path)
    hub_url = target[0] if target else ""
    hub_reachable = bool(target) and hub_is_healthy(*target)

    drawers = 0
    threads: set[str] = set()
    latest = ""
    palace_error = ""
    try:
        from .palace import get_collection

        collection = get_collection(palace_path, create=False)
        data = collection.get(where={"ingest_mode": "codex_session"}, include=["metadatas"])
        metas = data.get("metadatas") or []
        drawers = len(metas)
        for m in metas:
            if not m:
                continue
            if m.get("thread_id"):
                threads.add(m["thread_id"])
            ts = m.get("timestamp") or ""
            if ts > latest:
                latest = ts
    except Exception as exc:
        palace_error = f"{type(exc).__name__}: {exc}"
        logger.warning("codex status: palace read failed (%s)", exc)

    if writer == "blocked" or (writer == "hub" and last_error) or unhealthy:
        state = "BLOCKED"
    elif running and not last_error:
        state = "ACTIVE"
    else:
        state = "DEGRADED"

    return {
        "state": state,
        "unhealthy": unhealthy,
        "consecutive_failures": consecutive_failures,
        "outage_announced": bool(hb.get("outage_announced")),
        "follower_running": running,
        "follower_pid": pid,
        "heartbeat_at": hb.get("updated_at"),
        "writer": writer or ("hub" if hub_reachable else "direct"),
        "hub_url": hub_url,
        "hub_reachable": hub_reachable,
        "sessions_root": hb.get("sessions_root") or str(sessions_root),
        "rollouts_on_disk": len(iter_rollouts(sessions_root)),
        "codex_drawers": drawers,
        "codex_threads": len(threads),
        "latest_turn_at": latest,
        "capture_lag_seconds": _lag_seconds(latest, hb.get("updated_at") or ""),
        "last_error": last_error or palace_error,
        "last_scan": last,
    }
