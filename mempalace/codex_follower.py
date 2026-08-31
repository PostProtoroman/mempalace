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

from .sources.codex import ADAPTER_VERSION, iter_rollouts, iter_turns, turn_to_drawer

logger = logging.getLogger(__name__)

DEFAULT_SESSIONS_ROOT = os.path.join("~", ".codex", "sessions")
DEFAULT_INTERVAL_S = 15.0
BATCH_SIZE = 64

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
) -> dict:
    """One incremental pass over every rollout. Returns write statistics."""
    from .palace import get_collection

    own_collection = collection is None
    if own_collection:
        collection = get_collection(palace_path, create=True)

    added = already = skipped = 0
    files_scanned = 0
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
        docs = [deduped[i][0] for i in ids]
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
        files_scanned += 1
        source_file = str(rollout)
        cursor = _stored_max_ordinal(collection, source_file)
        try:
            turns = iter_turns(source_file)
            for turn in turns:
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
                batch_ids.append(turn.drawer_id)
                batch_docs.append(record.content)
                batch_metas.append(meta)
                if len(batch_ids) >= BATCH_SIZE:
                    _flush()
        except OSError as exc:
            logger.warning("codex follower: unreadable rollout %s (%s)", source_file, exc)
            continue

    _flush()

    return {
        "files_scanned": files_scanned,
        "threads_seen": len(threads),
        "drawers_added": added,
        "drawers_already_present": already,
        "turns_skipped_by_cursor": skipped,
    }


def run_once(palace_path: str, sessions_root: str = DEFAULT_SESSIONS_ROOT) -> dict:
    """One locked pass. Chroma writes stay serialized behind the palace lock."""
    from .palace import mine_palace_lock

    with mine_palace_lock(palace_path):
        stats = scan_once(palace_path, sessions_root)
    _write_heartbeat(palace_path, sessions_root=str(sessions_root), last_scan=stats, running=False)
    return stats


def follow(
    palace_path: str,
    sessions_root: str = DEFAULT_SESSIONS_ROOT,
    interval: float = DEFAULT_INTERVAL_S,
    *,
    max_passes: Optional[int] = None,
) -> dict:
    """Scan on an interval until interrupted (or ``max_passes`` is reached)."""
    from .palace import mine_palace_lock

    passes = 0
    totals = {"drawers_added": 0, "drawers_already_present": 0, "passes": 0}
    try:
        while max_passes is None or passes < max_passes:
            with mine_palace_lock(palace_path):
                stats = scan_once(palace_path, sessions_root)
            passes += 1
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
            )
            if max_passes is not None and passes >= max_passes:
                break
            time.sleep(interval)
    except KeyboardInterrupt:  # pragma: no cover - interactive stop
        pass
    finally:
        _write_heartbeat(
            palace_path, sessions_root=str(sessions_root), passes=passes, running=False
        )
    return totals


def status(palace_path: str, sessions_root: str = DEFAULT_SESSIONS_ROOT) -> dict:
    """Everything needed to prove capture is (or is not) active."""
    from .palace import get_collection

    hb = read_heartbeat(palace_path) or {}
    pid = hb.get("pid")
    live = bool(pid) and _pid_alive(int(pid)) and bool(hb.get("running"))

    drawers = 0
    threads: set[str] = set()
    latest = ""
    try:
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
        logger.warning("codex status: palace read failed (%s)", exc)

    return {
        "follower_running": live,
        "follower_pid": pid,
        "heartbeat_at": hb.get("updated_at"),
        "sessions_root": hb.get("sessions_root") or str(sessions_root),
        "rollouts_on_disk": len(iter_rollouts(sessions_root)),
        "codex_drawers": drawers,
        "codex_threads": len(threads),
        "latest_turn_at": latest,
        "last_scan": hb.get("last_scan"),
    }
