# launchd: scheduled `mempalace backup` (macOS)

`deploy/launchd/ai.mempalace.backup.plist` runs `mempalace backup` daily at
04:00 as a per-user launchd agent — the macOS equivalent of a cron job, but
one that also fires on wake if the machine was asleep through 04:00. It
writes to `~/.mempalace-backups` (outside the palace home, so a bad rebuild
of `~/.mempalace` can't take the backups down with it) and keeps the newest 7
archives by default; see `mempalace backup --help` / `mempalace restore
--help` for `--dest`, `--keep`, and how to restore an archive back.

## Install

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

## Verify

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

## Uninstall

```bash
launchctl bootout gui/$(id -u)/ai.mempalace.backup
rm ~/Library/LaunchAgents/ai.mempalace.backup.plist
```

## Notes

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
