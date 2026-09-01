"""Durable per-agent logstream cursors (RFC 003).

The coordination protocol demands id-cursor resumption — "remember that id,
it is your cursor" — but until now shipped no place to keep the id: every
agent hand-carried its last-processed event id between turns, and most lost
it. This module gives the cursor a home NEXT TO the logstream database, so
it lives and dies (and replicates, and gets backed up) with the log it
indexes into.

Layout: ``<logstream dir>/logstream_cursors/<agent>.json``, one file per
agent. One-file-per-agent means two agents can checkpoint concurrently with
no read-modify-write race on a shared document; each write is a whole-file
atomic replace (tmp + fsync + os.replace), so a crash mid-write leaves the
previous cursor intact rather than a torn one. A torn cursor would be read
as corrupt and force a replay — safe but noisy; the atomic write avoids it.

Cursors are advisory bookmarks, not locks: reading one never blocks a peer,
and setting one only ever moves *your own* bookmark.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone

CURSOR_DIR_NAME = "logstream_cursors"

# Agent identities become filenames, so constrain them to a portable,
# non-traversing charset. This is deliberately stricter than the logstream's
# own agent fields ('*' broadcasts, for one, are not a cursor owner).
_AGENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def cursor_dir(db_path: str) -> str:
    """Directory holding per-agent cursor files, next to the logstream db."""
    return os.path.join(
        os.path.dirname(os.path.abspath(os.path.expanduser(db_path))), CURSOR_DIR_NAME
    )


def _validate_agent(agent: str) -> str:
    if not isinstance(agent, str) or not _AGENT_RE.match(agent or ""):
        raise ValueError(
            "invalid agent name for cursor storage: must be 1-128 chars of "
            "letters, digits, '.', '_' or '-', starting with a letter or digit "
            f"(got {agent!r})"
        )
    if agent in (".", ".."):  # unreachable via the regex, kept as a belt
        raise ValueError(f"invalid agent name for cursor storage: {agent!r}")
    return agent


def _cursor_path(db_path: str, agent: str) -> str:
    return os.path.join(cursor_dir(db_path), f"{_validate_agent(agent)}.json")


def cursor_get(db_path: str, agent: str) -> dict | None:
    """Return the stored cursor record for ``agent``, or None when absent.

    A corrupt or unreadable file is reported as a record with
    ``"corrupt": True`` instead of being silently treated as absent —
    absent means "first run, start at the tip is fine", corrupt means
    "state was lost, the caller should replay rather than skip".
    """
    path = _cursor_path(db_path, agent)
    try:
        with open(path, encoding="utf-8") as f:
            raw = f.read()
    except FileNotFoundError:
        return None
    except OSError:
        return {"agent": agent, "event_id": None, "corrupt": True}
    try:
        record = json.loads(raw)
        if not isinstance(record, dict) or not isinstance(record.get("event_id"), str):
            raise ValueError("cursor record malformed")
    except ValueError:
        return {"agent": agent, "event_id": None, "corrupt": True}
    record["agent"] = agent
    return record


def cursor_set(db_path: str, agent: str, event_id: str) -> dict:
    """Atomically persist ``agent``'s last-processed event id.

    Returns the record as written. The write is tmp + fsync + os.replace in
    the destination directory, so a reader never observes a partial file and
    a crash never destroys the previous cursor.
    """
    if not isinstance(event_id, str) or not event_id.strip():
        raise ValueError("event_id must be a non-empty string")
    event_id = event_id.strip()
    path = _cursor_path(db_path, agent)
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    record = {
        "agent": agent,
        "event_id": event_id,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    fd, tmp_path = tempfile.mkstemp(prefix=f".{agent}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return record
