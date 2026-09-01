"""Global ``~/.codex/hooks.json`` installer — the safety net for Codex capture.

The follower service (``deploy/launchd``) is the primary way Codex sessions
get captured: it tails ``~/.codex/sessions`` continuously. Codex CLI's
plugin manifest schema has no lifecycle-hooks field (see
``.codex-plugin/hooks.json`` and ``tests/test_codex_plugin_manifest.py``),
so a plugin cannot wire hooks on its own — but Codex's *global*
``~/.codex/hooks.json`` supports them directly. This module prints (and,
with consent, writes) that file so Stop/PreCompact fire a capture pass even
when the follower service is not running.

Never overwrites an existing ``hooks.json``: that file is user-owned and may
already carry the user's own hooks. If one exists, this prints a manual-merge
diff hint instead of touching it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_HOOKS_FILE = os.path.join("~", ".codex", "hooks.json")

_HOOK_NAMES = ("stop", "precompact")


def _hook_entry(hook_name: str) -> dict:
    return {
        "matcher": "*",
        "hooks": [
            {
                "type": "command",
                "command": f"mempalace hook run --hook {hook_name} --harness codex",
            }
        ],
    }


def hooks_config() -> dict:
    """The hooks.json content this installer wires: Stop + PreCompact."""
    return {
        "hooks": {
            "Stop": [_hook_entry("stop")],
            "PreCompact": [_hook_entry("precompact")],
        }
    }


def render_hooks_json() -> str:
    """Pretty-printed JSON text, newline-terminated, ready to write verbatim."""
    return json.dumps(hooks_config(), indent=2) + "\n"


def render_merge_hint(existing_text: str) -> str:
    """A human-readable hint for merging into an existing hooks.json by hand."""
    lines = [
        "An existing hooks.json was found — it was NOT modified.",
        "",
        'Add these entries under "hooks" (merge, don\'t replace, if you',
        "already have Stop or PreCompact hooks of your own):",
        "",
        render_hooks_json().rstrip("\n"),
        "",
        "Current file content:",
        "",
        existing_text.rstrip("\n"),
    ]
    return "\n".join(lines)


def install(hooks_file: str = DEFAULT_HOOKS_FILE, *, apply: bool = False) -> dict:
    """Print/plan the install. Writes only with ``apply=True`` and only if absent.

    Returns a dict describing what happened:
      - ``path``: resolved file path
      - ``existed``: whether a file was already there
      - ``written``: whether this call wrote the file
      - ``content``: the hooks.json text (for printing)
    """
    path = Path(hooks_file).expanduser()
    existed = path.exists()

    if existed:
        existing_text = path.read_text(encoding="utf-8")
        return {
            "path": str(path),
            "existed": True,
            "written": False,
            "content": render_hooks_json(),
            "merge_hint": render_merge_hint(existing_text),
        }

    content = render_hooks_json()
    if not apply:
        return {
            "path": str(path),
            "existed": False,
            "written": False,
            "content": content,
            "merge_hint": None,
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {
        "path": str(path),
        "existed": False,
        "written": True,
        "content": content,
        "merge_hint": None,
    }
