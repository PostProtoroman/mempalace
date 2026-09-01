"""Route Codex capture writes through a live palace hub.

A hub (``mempalace serve``) holds the palace writer lease for its whole
lifetime. A follower that took ``mine_palace_lock`` itself would therefore be
refused on exactly the machines a hub is running on, and capture would stop
without anyone noticing. So when a writable hub owns the palace, capture is
handed to it over HTTP — the hub does the write with the lease it already
holds, and there is still exactly one writer.

The detection and forwarding shape deliberately mirrors ``cli.py``'s existing
hub-forwarded mine (``_forward_mine_to_hub``): read the registry, probe
``/healthz``, then POST a ``tools/call``. Nothing new is invented here.

Direct local writes remain only as the no-hub fallback. The choice is made
once per pass and is exclusive: if a hub answered, this process never touches
the palace lock, and if the hub fails mid-request the turn is reported as an
error rather than retried locally, because the hub may have already written
it and a local retry would be a second writer.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

HEALTH_TIMEOUT_S = 2.0
CALL_TIMEOUT_S = 120.0

# One hub call writes at most this many turns, so a first-time scan of a large
# corpus cannot park the hub's write path. The follower loops until the hub
# reports nothing pending.
TURNS_PER_CALL = 250
MAX_CALLS_PER_PASS = 200


class HubUnavailable(Exception):
    """No live, writable hub owns this palace."""


class HubRefused(Exception):
    """The hub answered and refused, or failed mid-request.

    Never retried locally: the hub may have applied the write already.
    """


def hub_target(palace_path: str) -> Optional[tuple[str, dict]]:
    """Return (base_url, headers) for a live writable hub, else None."""
    from . import server_registry

    try:
        info = server_registry.read_live_serverinfo(palace_path)
    except Exception as exc:
        logger.debug("codex hub: registry read failed (%s)", exc)
        return None
    if not info or info.get("read_only"):
        return None

    base_url = server_registry.client_base_url(info)
    headers = {"Content-Type": "application/json"}
    try:
        token = server_registry.load_server_token(palace_path)
    except Exception:
        token = ""
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return base_url, headers


def hub_is_healthy(base_url: str, headers: dict) -> bool:
    try:
        request = urllib.request.Request(f"{base_url}/healthz", headers=headers)
        with urllib.request.urlopen(request, timeout=HEALTH_TIMEOUT_S) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _call(
    base_url: str,
    headers: dict,
    sessions_root: str,
    max_turns: int,
    backfill: bool = False,
    include_automation: bool = False,
) -> dict:
    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "mempalace_codex_capture",
                "arguments": {
                    "sessions_root": sessions_root,
                    "max_turns": max_turns,
                    "backfill": backfill,
                    "include_automation": include_automation,
                },
            },
        }
    ).encode("utf-8")
    request = urllib.request.Request(f"{base_url}/mcp", data=body, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=CALL_TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise HubRefused(f"hub returned HTTP {exc.code} {exc.reason}") from exc
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
        # Mid-flight failure on a mutating call: the hub may have written.
        raise HubRefused(f"hub did not complete the capture call ({exc})") from exc

    if payload.get("error"):
        err = payload["error"]
        raise HubRefused(str(err.get("message") or err))

    # MCP tool results arrive as content blocks; the tool's dict is JSON in text.
    result = payload.get("result") or {}
    for block in result.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            try:
                parsed = json.loads(block.get("text") or "{}")
            except ValueError:
                continue
            if isinstance(parsed, dict):
                return parsed
    if isinstance(result, dict) and "drawers_added" in result:
        return result
    raise HubRefused("hub response carried no capture result")


def capture_via_hub(
    palace_path: str,
    sessions_root: str,
    *,
    backfill: bool = False,
    include_automation: bool = False,
) -> dict:
    """Run a full capture pass through the hub. Raises HubUnavailable/HubRefused."""
    target = hub_target(palace_path)
    if target is None:
        raise HubUnavailable("no live writable hub for this palace")
    base_url, headers = target
    if not hub_is_healthy(base_url, headers):
        raise HubUnavailable(f"hub at {base_url} did not answer /healthz")

    totals = {
        "drawers_added": 0,
        "drawers_already_present": 0,
        "turns_skipped_by_cursor": 0,
        "subagent_turns_routed": 0,
        "automation_turns_captured": 0,
        "files_scanned": 0,
        "files_automation_skipped": 0,
        "files_unparseable": 0,
        "threads_seen": 0,
        "hub_calls": 0,
        "writer": "hub",
        "hub_url": base_url,
    }
    for _ in range(MAX_CALLS_PER_PASS):
        stats = _call(
            base_url, headers, sessions_root, TURNS_PER_CALL, backfill, include_automation
        )
        if not stats.get("success", True):
            raise HubRefused(str(stats.get("error") or "hub capture failed"))
        totals["hub_calls"] += 1
        for key in (
            "drawers_added",
            "drawers_already_present",
            "turns_skipped_by_cursor",
            "subagent_turns_routed",
            "automation_turns_captured",
        ):
            totals[key] += int(stats.get(key) or 0)
        for key in ("files_scanned", "files_automation_skipped", "files_unparseable"):
            totals[key] = int(stats.get(key) or 0)
        totals["threads_seen"] = max(totals["threads_seen"], int(stats.get("threads_seen") or 0))
        if not stats.get("pending"):
            return totals
    # Budget exhausted without draining: not an error, just more next pass.
    totals["pending"] = True
    return totals
