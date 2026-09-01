# MemPalace - Codex CLI Plugin

Give your AI a persistent memory -- mine projects and conversations into a searchable palace backed by ChromaDB, with 44 MCP tools and guided skills.

## Prerequisites

- Python 3.9+
- Codex CLI installed and configured
- `uv tool install mempalace` (recommended) or `pip install mempalace`

## Installation

### Local Install

1. Copy or symlink the `.codex-plugin` directory into your project root:

```bash
cp -r .codex-plugin /path/to/your/project/.codex-plugin
```

2. Verify the plugin is detected:

```bash
codex --plugins
```

3. Initialize your palace:

```bash
codex /init
```

### Git Install

1. Clone the MemPalace repository:

```bash
git clone https://github.com/MemPalace/mempalace.git
cd mempalace
```

2. Install the Python package so the `mempalace-mcp` script lands on
   your PATH (the bundled `plugin.json` invokes it by bare name):

```bash
uv tool install --editable .   # or: pip install -e .
```

   Plain `uv sync` is **not** enough here — it installs the scripts into
   `.venv/bin/`, which Codex will not find unless you activate the venv
   before launching Codex.

3. The `.codex-plugin` directory is already in the repo root. Codex CLI will detect it automatically when you run Codex from inside the repository.

4. Initialize your palace:

```bash
codex /init
```

## Available Skills

| Skill | Description |
|-------|-------------|
| `/help` | Show available commands and usage tips |
| `/init` | Initialize a new memory palace |
| `/search` | Semantic search across all mined memories |
| `/mine` | Mine a project or conversation into your palace |
| `/status` | Show palace status, room counts, and health |

## Capturing conversations

Codex's plugin manifest supports skills and MCP servers, but not lifecycle
hooks — this plugin's own `hooks.json` (shipped in this directory) cannot be
referenced by any supported manifest field, so installing the plugin alone
does not auto-save turns. There are two real capture paths:

### 1. Follower service (primary)

A background service tails `~/.codex/sessions` continuously and files new
turns as they land, independent of whether Codex is even running. Install it
via the launchd unit in `deploy/launchd/` (see `deploy/launchd/README.md`),
or run it in the foreground / once:

```bash
mempalace codex-capture           # follow continuously
mempalace codex-capture --once    # one pass, then exit
mempalace codex-capture --backfill --dry-run   # preview importing existing history
```

Capture is incremental — re-running picks up new sessions without
duplicating what is already filed. Both the legacy
(`user_message`/`agent_message`) and current (`item_completed`, Codex
>= 0.149) transcript formats are supported. Check status with
`mempalace codex-status`.

### 2. Global hooks.json (safety net)

Codex's *global* `~/.codex/hooks.json` (distinct from this plugin's
manifest) does support lifecycle hooks. As a safety net for when the
follower service isn't running, wire Stop and PreCompact there:

```bash
mempalace codex-hooks-install          # print the exact file content + where it goes
mempalace codex-hooks-install --apply  # write it — only if no hooks.json exists yet
```

This never overwrites an existing `~/.codex/hooks.json` — if one is already
there, the command prints a merge hint instead so you can add the
MemPalace entries by hand alongside whatever you already have.

## Recall

The `recall` skill (`skills/recall/SKILL.md`) makes Codex search the palace
before answering about prior work, people, or decisions, instead of
guessing. It mirrors the policy installed into `~/.codex/AGENTS.md` by
`mempalace codex-recall install`.

## Support

- Repository: https://github.com/MemPalace/mempalace
- Issues: https://github.com/MemPalace/mempalace/issues
