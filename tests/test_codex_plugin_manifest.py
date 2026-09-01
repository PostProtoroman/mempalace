"""Contract tests for the Codex plugin marketplace metadata."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MARKETPLACE_PATH = REPO_ROOT / ".agents" / "plugins" / "marketplace.json"
MANIFEST_PATH = REPO_ROOT / ".codex-plugin" / "plugin.json"
MCP_PATH = REPO_ROOT / ".mcp.json"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_marketplace_entry_uses_supported_codex_schema():
    marketplace = _read_json(MARKETPLACE_PATH)
    plugin = marketplace["plugins"][0]

    assert plugin["name"] == "mempalace"
    assert plugin["source"] == {"source": "local", "path": "./"}
    assert plugin["policy"] == {
        "installation": "AVAILABLE",
        "authentication": "ON_INSTALL",
    }


def test_plugin_manifest_references_supported_components():
    manifest = _read_json(MANIFEST_PATH)

    assert manifest["mcpServers"] == "./.mcp.json"
    assert "hooks" not in manifest


def test_plugin_manifest_relative_paths_resolve_inside_plugin_dir():
    """Codex resolves manifest-relative paths against the manifest's own
    directory (.codex-plugin/), not the repo root — the same directory
    plugin.json itself lives in. Every relative path the manifest declares
    must exist there, or Codex plugin loading fails silently at runtime.
    """
    manifest = _read_json(MANIFEST_PATH)
    plugin_dir = MANIFEST_PATH.parent

    skills_path = plugin_dir / manifest["skills"]
    assert skills_path.is_dir(), f"missing {skills_path}"

    mcp_path = plugin_dir / manifest["mcpServers"]
    assert mcp_path.is_file(), f"missing {mcp_path}"


def test_mcp_config_registers_mempalace_server():
    config = _read_json(MCP_PATH)

    assert config == {
        "mcpServers": {
            "mempalace": {
                "command": "mempalace-mcp",
            }
        }
    }
