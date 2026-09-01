"""Install the MemPalace recall block into Codex's instructions file.

Codex reads ``~/.codex/AGENTS.md`` at session start, so that file is the
smallest place to make recall happen: it tells Codex to search the palace
before answering questions about prior work, people, or decisions. The
capture side fills the palace; this is the side that makes Codex look.

The block is delimited by HTML comment markers so re-installing replaces it
in place rather than appending a second copy, and removing it is a clean
cut. Everything the user wrote around it is preserved.
"""

from __future__ import annotations

import os
from pathlib import Path

START = "<!-- mempalace-codex-recall:start -->"
END = "<!-- mempalace-codex-recall:end -->"

DEFAULT_AGENTS_FILE = os.path.join("~", ".codex", "AGENTS.md")


def recall_block() -> str:
    """The instructions block, read from the packaged asset."""
    asset = Path(__file__).with_name("instructions") / "codex_recall.md"
    return asset.read_text(encoding="utf-8").strip()


def _strip_existing(text: str) -> str:
    if START not in text or END not in text:
        return text
    head, _, rest = text.partition(START)
    _, _, tail = rest.partition(END)
    return (head.rstrip() + "\n" + tail.lstrip()).strip()


def install(agents_file: str = DEFAULT_AGENTS_FILE) -> dict:
    """Write (or refresh) the recall block. Returns what changed."""
    path = Path(agents_file).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    had_block = START in existing
    preserved = _strip_existing(existing)

    block = recall_block()
    merged = f"{preserved}\n\n{block}\n" if preserved else f"{block}\n"

    if existing == merged:
        return {"path": str(path), "changed": False, "replaced": had_block}

    path.write_text(merged, encoding="utf-8")
    return {"path": str(path), "changed": True, "replaced": had_block}


def is_installed(agents_file: str = DEFAULT_AGENTS_FILE) -> bool:
    path = Path(agents_file).expanduser()
    try:
        return START in path.read_text(encoding="utf-8")
    except OSError:
        return False
