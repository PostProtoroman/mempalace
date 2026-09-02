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

## Hub restart runbook

When restarting the hub (`launchctl bootout gui/$(id -u)/ai.mempalace.hub`
then `launchctl bootstrap gui/$(id -u)/ ~/Library/LaunchAgents/ai.mempalace.hub.plist`),
sessions do **not** need to be killed. The hub's writer lease is now per-call
scoped, so stdio and embedded servers can handle the restart gracefully without
holding the palace hostage. When the hub drops off the network, mutating calls
see a `-32001` error with `data.retryable: true` and a grace window that gives
the hub 60 seconds to come back (`MEMPALACE_HUB_GRACE_SECS`); reads fall back
to the local palace immediately. Restart the hub, and active sessions reconnect
on their next call — no reconfiguration, no manual intervention needed.

## Scheduled `mempalace backup`

`deploy/launchd/ai.mempalace.backup.plist` runs `mempalace backup` daily at
04:00 as a per-user launchd agent — the macOS equivalent of a cron job, but
one that also fires on wake if the machine was asleep through 04:00. It
writes to `~/.mempalace-backups` (outside the palace home, so a bad rebuild
of `~/.mempalace` can't take the backups down with it) and keeps the newest 7
archives by default; see `mempalace backup --help` / `mempalace restore
--help` for `--dest`, `--keep`, and how to restore an archive back.

### Install

```bash
# 1. Point the template at your checkout (the dir with pyproject.toml and
#    the .venv `uv sync --extra dev` created).
CHECKOUT="$(cd /path/to/mempalace && pwd)"
sed -i '' "s#__CHECKOUT__#${CHECKOUT}#g" deploy/launchd/ai.mempalace.backup.plist

# 2. Install into the per-user LaunchAgents dir and load it.
cp deploy/launchd/ai.mempalace.backup.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.mempalace.backup.plist
```

If `mempalace` was installed globally instead (`pipx install mempalace` /
`uv tool install mempalace`) rather than via an in-checkout `.venv`, edit the
`ProgramArguments` entry in the plist to that tool's absolute path (find it
with `which mempalace` in the shell that has it installed — launchd agents do
not inherit your shell's `PATH`) instead of running the `sed` substitution
above, and drop the `WorkingDirectory` key since there is no checkout to
anchor it to.

### Verify

```bash
# Run it once immediately, out of schedule, to confirm the paths are right:
launchctl kickstart -k gui/$(id -u)/ai.mempalace.backup

# Check it actually wrote something:
ls -la ~/.mempalace-backups
tail -20 "$CHECKOUT/.mempalace-backup.log"
```

A clean run's log ends with lines like:

```
  Archive:  /Users/you/.mempalace-backups/mempalace-20260901-040000.tar.gz
  Manifest: /Users/you/.mempalace-backups/mempalace-20260901-040000.manifest.json
  Files:    4218
```

### Uninstall

```bash
launchctl bootout gui/$(id -u)/ai.mempalace.backup
rm ~/Library/LaunchAgents/ai.mempalace.backup.plist
```

### Notes

- **Off-machine copy.** launchd only gets the archive onto the same disk as
  the palace it is backing up. For real disaster recovery (stolen laptop,
  failed drive), sync `~/.mempalace-backups` somewhere else too — the two
  most common options are cloud-drive sync (Dropbox/iCloud Drive/Google Drive
  pointed at that folder) or an `rclone`/`rsync` line appended as a second
  `ProgramArguments` step in a wrapper script. Neither is wired up by this
  template; it only covers the local snapshot.
- **Restore is manual, on purpose.** `mempalace restore ARCHIVE --target DIR`
  always unpacks into an empty directory and never overwrites `~/.mempalace`
  itself — move the restored tree into place yourself once you've checked it
  (e.g. `mempalace search` against `--palace DIR/palace` before you commit to
  the swap). There is no scheduled/automatic restore, and there shouldn't be.
- **Logs are not rotated.** `StandardOutPath`/`StandardErrorPath` both point
  at one file that only grows; launchd does not rotate it. Truncate it
  yourself occasionally, or point those two keys at `/usr/bin/logger`-style
  rotation if you'd rather not think about it.
