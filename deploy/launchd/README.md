# launchd templates (macOS)

Per-user LaunchAgents for running MemPalace from one canonical checkout's
virtualenv. Two placeholders must be substituted before installing — launchd
expands neither `~` nor variables:

- `__CHECKOUT__` — absolute path to the canonical MemPalace checkout. The job
  runs `__CHECKOUT__/.venv/bin/python -m mempalace …`, so that venv must exist
  (`uv sync` in the checkout) and import this MemPalace.
- `__HOME__` — the user's home directory (used for `HOME` and the log paths).

| Template | Label | Runs |
|---|---|---|
| `ai.mempalace.hub.plist` | `ai.mempalace.hub` | `mempalace serve` — the palace hub, loopback-only on port 8765. Holds the palace writer lease; Codex capture routes writes through it. |
| `ai.mempalace.codex-capture.plist` | `ai.mempalace.codex-capture` | `mempalace codex-capture` — the Codex session follower, every 15s, never touching `~/.codex/AGENTS.md`. |

Load the hub before capture: a follower with no hub writes directly and would
then be refused once a hub takes the lease mid-flight.

## Install

```bash
CHECKOUT=/path/to/mempalace   # this checkout
mkdir -p ~/.mempalace/hub ~/.mempalace/codex_capture ~/Library/LaunchAgents

for name in ai.mempalace.hub ai.mempalace.codex-capture; do
  sed -e "s|__CHECKOUT__|$CHECKOUT|g" -e "s|__HOME__|$HOME|g" \
    "$CHECKOUT/deploy/launchd/$name.plist" > ~/Library/LaunchAgents/$name.plist
  launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/$name.plist
done
```

Verify:

```bash
launchctl print gui/$(id -u)/ai.mempalace.hub | head -5
launchctl print gui/$(id -u)/ai.mempalace.codex-capture | head -5
mempalace codex-status
```

## Uninstall

```bash
for name in ai.mempalace.codex-capture ai.mempalace.hub; do
  launchctl bootout gui/$(id -u)/$name 2>/dev/null
  rm -f ~/Library/LaunchAgents/$name.plist
done
```

Palace data, logstream, and `~/.codex/AGENTS.md` are never touched by install
or uninstall — only the plists and the service's own log files are managed.

Note: `mempalace codex-service install` writes an equivalent capture plist
programmatically (pinning the exact interpreter it is run from); these
templates exist for reviewable, repo-tracked deployments of both jobs from a
canonical checkout.
