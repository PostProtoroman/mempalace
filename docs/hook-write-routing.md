# Hook write routing

> [!IMPORTANT]
> This page documents unreleased hook write-routing behavior on `develop`.
> MemPalace v3.6.0 does not consume `write_routing.hooks`; see the
> [v3.6.0 write-routing policy](https://github.com/MemPalace/mempalace/blob/v3.6.0/docs/write-routing-policy.md#backward-compatibility)
> for behavior available in that release.

Hook-triggered writes use the shared write-routing policy introduced for the
Tier 3 rollout tracked in #1963.

## Scope

This applies to every routine write initiated by the Python hook layer:

- Stop-hook diary checkpoints;
- transcript/conversation ingest;
- project auto-ingest;
- SessionEnd final flushes;
- PreCompact transcript ingest;
- PreCompact project mining (submitted, not awaited — see below).

It does not change CLI write routing. CLI adoption is a separate follow-up.

## Policy behavior

### `direct`

Hooks use the existing direct in-process or subprocess paths.

The daemon is not probed.

### `prefer`

Hooks use the daemon when it is already healthy.

If the daemon is unavailable, hooks retain the historical direct fallback.

### `require`

Hooks use the daemon when it is already healthy.

If the daemon is unavailable:

- no in-process ChromaDB write runs;
- no direct `mempalace mine` subprocess is started;
- no daemon is cold-started from the hook;
- the hook log records the skipped operation;
- the hook returns a visible `systemMessage`;
- the Stop save marker is not advanced, allowing a later retry.

## Hooks submit mines; they do not wait for them

Every hook mine path — Stop/SessionEnd auto-ingest and PreCompact alike —
returns once the mine has been *submitted*: queued on the hub's background
worker, queued on the daemon, or started as a detached subprocess. None of
them waits for it to finish.

PreCompact used to be the exception: it ran `mempalace mine` with a 60-second
`subprocess.run` timeout and blocked. Against a live HTTP hub that request was
the one the hub ran inline inside its exclusive write path, where a 20-80 MB
transcript took minutes and the write-stall watchdog
(`MEMPALACE_MCP_WRITE_STALL_EXIT_SECS`) killed the hub as if the write were
wedged. The hub now enqueues the mine and answers immediately with a job id;
the worker takes the palace write lock one source file at a time.

Nothing is lost by not waiting: mining is idempotent (deterministic drawer ids
upsert to the same rows), so a mine still running when compaction happens keeps
filing the same content. The old behavior's 60-second kill was strictly worse —
it stopped the mine partway with no retry.

Follow a queued hub job in `mempalace_status.mine_queue` or the hub's
`/statusz`; every state transition — `accepted`, `started`, `succeeded`,
`failed`, `interrupted`, `dropped` — appends a line to `mine_jobs.jsonl` in the
palace directory.

### A hub restart drops queued mines

The hub's queue is in memory. If the hub restarts, whatever it had accepted
but not yet finished is gone — the CLI has already reported those mines as
queued, so this is a real loss, and it is made visible rather than fixed:

- the dropped jobs are journalled `dropped` (clean shutdown) and a WARNING
  names them;
- a job stopped mid-mine at shutdown ends at a file boundary and is journalled
  `interrupted` — partially mined, never torn;
- after a crash or SIGKILL the journal's last line for a job stays `accepted`
  or `started`. The next hub reads the journal tail at startup, logs a WARNING
  naming those jobs, and reports them as `mine_queue.lost_previous_run` in
  `/statusz`; `mempalace doctor` (`capture_liveness`) warns about them too.

Resubmit a lost mine, or wait for the next Stop/SessionEnd hook to re-fire —
mining is idempotent. A PreCompact-triggered `MEMPAL_DIR` mine may not re-fire
for a long time, so that one is worth resubmitting by hand.

## Why hooks do not start the daemon

Hooks operate under strict latency budgets. Starting a Python daemon and its
storage dependencies from a Stop or SessionEnd hook can exceed that budget.

A supervised installation using `require` must start the daemon earlier, for
example at login, plugin initialization, or session setup:

    mempalace daemon start

SessionStart performs a fast health probe in `require` mode and warns early if
the required daemon is unavailable.

## One decision per hook event

A Stop or SessionEnd event may perform several writes:

1. diary checkpoint;
2. transcript ingest;
3. project auto-ingest.

The route is resolved once and stored in a context-local value for the whole
write burst. This avoids repeated health probes and prevents different writes
from selecting inconsistent routes during the same event.

## Submission ambiguity

Once a daemon submission is attempted, an error never triggers direct
fallback. The daemon may have accepted the job before the client observed the
failure; retrying directly could duplicate content.

## Invalid configuration

An explicitly invalid routing policy fails closed: hook writes are blocked and
no direct ChromaDB fallback is attempted.

An unrelated configuration read/runtime failure preserves the historical
direct-save behavior so a final checkpoint is not lost because of an
independent configuration failure.

## Backward compatibility

The default policy remains `direct`.

Legacy settings remain supported through the shared policy resolver:

- `MEMPALACE_HOOKS_DAEMON=true` maps to `prefer`;
- `hooks.daemon: true` maps to `prefer`;
- false values map to `direct`.

## Configuration examples

Prefer the daemon but permit direct fallback:

    MEMPALACE_HOOK_WRITE_ROUTING=prefer

Require the daemon and prohibit direct writers:

    MEMPALACE_HOOK_WRITE_ROUTING=require

Configuration file:

    {
      "write_routing": {
        "hooks": "require"
      }
    }
