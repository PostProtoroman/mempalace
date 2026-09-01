"""Tests for the global ~/.codex/hooks.json safety-net installer.

Uses only tmp_path-based files — never touches a real ~/.codex.
"""

from __future__ import annotations

import json
import subprocess
import sys

from mempalace import codex_hooks_install


def test_hooks_config_wires_stop_and_precompact_with_correct_nested_schema():
    config = codex_hooks_install.hooks_config()

    assert set(config["hooks"].keys()) == {"Stop", "PreCompact"}
    for name, hook_name in (("Stop", "stop"), ("PreCompact", "precompact")):
        entries = config["hooks"][name]
        assert entries == [
            {
                "matcher": "*",
                "hooks": [
                    {
                        "type": "command",
                        "command": f"mempalace hook run --hook {hook_name} --harness codex",
                    }
                ],
            }
        ]


def test_render_hooks_json_is_valid_json_matching_config():
    text = codex_hooks_install.render_hooks_json()
    assert json.loads(text) == codex_hooks_install.hooks_config()
    assert text.endswith("\n")


def test_install_dry_run_does_not_write(tmp_path):
    hooks_file = tmp_path / ".codex" / "hooks.json"

    result = codex_hooks_install.install(str(hooks_file), apply=False)

    assert result["existed"] is False
    assert result["written"] is False
    assert not hooks_file.exists()
    assert json.loads(result["content"]) == codex_hooks_install.hooks_config()


def test_install_apply_writes_when_absent(tmp_path):
    hooks_file = tmp_path / ".codex" / "hooks.json"

    result = codex_hooks_install.install(str(hooks_file), apply=True)

    assert result["written"] is True
    assert hooks_file.exists()
    assert json.loads(hooks_file.read_text(encoding="utf-8")) == codex_hooks_install.hooks_config()


def test_install_never_overwrites_existing_file(tmp_path):
    hooks_file = tmp_path / ".codex" / "hooks.json"
    hooks_file.parent.mkdir(parents=True)
    original = '{"hooks": {"UserPrompt": []}}'
    hooks_file.write_text(original, encoding="utf-8")

    result = codex_hooks_install.install(str(hooks_file), apply=True)

    assert result["existed"] is True
    assert result["written"] is False
    assert hooks_file.read_text(encoding="utf-8") == original
    assert (
        "not overwritten" in result["merge_hint"].lower() or "NOT modified" in result["merge_hint"]
    )
    assert "UserPrompt" in result["merge_hint"]
    assert "mempalace hook run --hook stop --harness codex" in result["merge_hint"]


def test_cli_dry_run_prints_content_and_leaves_no_file(tmp_path):
    hooks_file = tmp_path / "hooks.json"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mempalace.cli",
            "codex-hooks-install",
            "--hooks-file",
            str(hooks_file),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Would write" in proc.stdout
    assert '"Stop"' in proc.stdout
    assert '"PreCompact"' in proc.stdout
    assert not hooks_file.exists()


def test_cli_apply_writes_file(tmp_path):
    hooks_file = tmp_path / "hooks.json"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mempalace.cli",
            "codex-hooks-install",
            "--hooks-file",
            str(hooks_file),
            "--apply",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Wrote" in proc.stdout
    assert hooks_file.exists()
    assert json.loads(hooks_file.read_text(encoding="utf-8")) == codex_hooks_install.hooks_config()


def test_cli_apply_refuses_to_overwrite_existing(tmp_path):
    hooks_file = tmp_path / "hooks.json"
    hooks_file.write_text('{"hooks": {"Custom": []}}', encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mempalace.cli",
            "codex-hooks-install",
            "--hooks-file",
            str(hooks_file),
            "--apply",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "already exists" in proc.stdout
    assert "NOT overwritten" in proc.stdout
    assert hooks_file.read_text(encoding="utf-8") == '{"hooks": {"Custom": []}}'


def test_hook_names_are_valid_hook_run_choices():
    """The exact names this installer wires must be valid `hook run --hook` choices."""
    proc = subprocess.run(
        [sys.executable, "-m", "mempalace.cli", "hook", "run", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "stop" in proc.stdout
    assert "precompact" in proc.stdout
