"""Docs-consistency tests for the Codex CLI ``~/.codex/hooks.json`` schema.

The real Codex CLI wraps hook config under a top-level ``"hooks"`` key with
nested matcher groups (``{"hooks": {"Stop": [{"matcher": ..., "hooks": [...]}]}}``)
— the same shape Claude Code uses. A flat ``{"Stop": [...]}`` map at the top
level is rejected. Verified against the real, working
``~/.codex/hooks.json`` on the machine this fix was made on.

These tests parse the fenced JSON code blocks that follow each doc's
"Install — Codex CLI" heading and assert they use the nested shape, so a
future edit that reintroduces the flat map fails CI instead of shipping.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

DOC_PATHS = [
    REPO_ROOT / "website" / "guide" / "hooks.md",
    REPO_ROOT / "hooks" / "README.md",
]


def _codex_json_blocks(text: str) -> list[dict]:
    """Return every fenced ```json block that appears after a Codex CLI
    install heading, up to the next top-level (##) heading."""
    blocks = []
    sections = re.split(r"\n(?=## )", text)
    for section in sections:
        if "Codex CLI" not in section.split("\n", 1)[0]:
            continue
        for match in re.finditer(r"```json\n(.*?)```", section, re.DOTALL):
            blocks.append(json.loads(match.group(1)))
    return blocks


@pytest.mark.parametrize("doc_path", DOC_PATHS, ids=lambda p: p.name)
def test_codex_hooks_json_examples_use_nested_schema(doc_path: Path):
    text = doc_path.read_text(encoding="utf-8")
    blocks = _codex_json_blocks(text)
    assert blocks, f"expected at least one ```json block under a Codex CLI heading in {doc_path}"

    for block in blocks:
        assert "hooks" in block, (
            f"{doc_path}: Codex hooks.json example is missing the top-level "
            f'"hooks" key — a flat event map is rejected by the real Codex CLI: {block}'
        )
        events = block["hooks"]
        assert isinstance(events, dict) and events, (
            f"{doc_path}: 'hooks' must be a non-empty object"
        )
        for event_name, groups in events.items():
            assert isinstance(groups, list) and groups, (
                f"{doc_path}: hooks.{event_name} must be a non-empty list of matcher groups"
            )
            for group in groups:
                assert "hooks" in group, (
                    f"{doc_path}: hooks.{event_name} entries must be matcher groups with "
                    f"their own nested 'hooks' array, not bare command objects: {group}"
                )


@pytest.mark.parametrize("doc_path", DOC_PATHS, ids=lambda p: p.name)
def test_codex_hooks_docs_state_global_scope(doc_path: Path):
    text = doc_path.read_text(encoding="utf-8")
    match = re.search(r"^## Install.*Codex CLI.*$", text, re.MULTILINE)
    assert match, f"{doc_path}: expected an '## Install — Codex CLI' heading"
    section = text[match.start() : match.start() + 2000]
    assert "global" in section.lower(), (
        f"{doc_path}: Codex CLI hooks section should state plainly that "
        "~/.codex/hooks.json is a global config, not a plugin manifest capability"
    )
