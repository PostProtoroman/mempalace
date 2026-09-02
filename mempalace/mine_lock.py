"""Per-file lock scoping for long mines.

A mine that holds one exclusive lock for its whole run is the shape of the
2026-09-02 hub restart storm: a 20-80 MB Claude Code transcript takes minutes
inside ``mempalace_mine``, the writable MCP hub ran it inline in its exclusive
dispatch lock, every other agent's palace write queued behind it, and the
write-stall watchdog (``MEMPALACE_MCP_WRITE_STALL_EXIT_SECS``) read the single
long hold as a wedged chromadb write and killed the process — four times in
fifteen minutes.

Both miners already do their real work one source file at a time:
``convo_miner`` purges and re-files per ``source_file`` under
``_file_chunks_locked``, and ``miner.process_file`` is per file. The source
file is therefore the natural lock granularity — no single hold spans more
than one file, and peers interleave between files.

``FileLockCursor`` hands the miners that granularity without restructuring
their loops. ``cursor.iter(files)`` releases the previous file's hold and
takes the next one as each file is yielded, so the loop body is unchanged;
``cursor.enter(label)`` covers the phases outside the loop (opening the
collection, the end-of-mine hallway/FTS work) and ``cursor.release()`` gives
the lock back.

With ``file_lock=None`` — every CLI, library and stdio caller — each method is
a no-op and the miners behave exactly as before.
"""

from __future__ import annotations

from typing import Callable, ContextManager, Iterable, Iterator, Optional

__all__ = ["FileLockCursor", "GuardedFileLock", "OPEN_PHASE", "FINALIZE_PHASE"]

# Labels for the two non-per-file phases, so an operator reading a stall
# warning can tell "stuck opening the collection" from "stuck on file N".
OPEN_PHASE = "<open>"
FINALIZE_PHASE = "<finalize>"

FileLockFactory = Callable[[str], ContextManager]


class FileLockCursor:
    """Hold at most one ``file_lock`` context at a time, moving it per file.

    Not thread-safe by design: one cursor belongs to one mine, and a mine is
    a single-threaded loop. The hub's worker thread owns its cursor for the
    life of one job.
    """

    def __init__(self, file_lock: Optional[FileLockFactory] = None) -> None:
        self._file_lock = file_lock
        self._cm: Optional[ContextManager] = None
        self.current: Optional[str] = None
        # A lock owner that can go away mid-mine (the hub, on shutdown) hangs
        # a ``should_stop`` predicate on its factory. Reading it off the
        # factory keeps three miners' signatures out of it.
        self._should_stop = getattr(file_lock, "should_stop", None)
        self.interrupted = False

    @property
    def active(self) -> bool:
        """True while a lock is held."""
        return self._cm is not None

    def enter(self, label: str) -> None:
        """Release whatever is held and take the lock for ``label``."""
        self.release()
        if self._file_lock is None:
            return
        cm = self._file_lock(label)
        cm.__enter__()
        # Assign only after __enter__ returns: a factory that raises must not
        # leave a half-entered context that release() would then exit.
        self._cm = cm
        self.current = label

    def iter(self, files: Iterable) -> Iterator:
        """Yield ``files``, moving the lock to each one before it is handed out.

        The hold is *not* released when the iterable is exhausted — the caller
        usually has end-of-mine work to do under the same lock and calls
        :meth:`enter` with :data:`FINALIZE_PHASE`, or :meth:`release`.

        If the lock owner's ``should_stop`` predicate goes true, iteration
        ends *here*, at a file boundary, with the hold released and
        :attr:`interrupted` set. The miner then runs its normal end-of-mine
        block over the files it did finish — a partial mine, not a torn one,
        and crucially not one still writing after the hub gave back the
        writer lease.
        """
        for item in files:
            if self._should_stop is not None and self._should_stop():
                self.interrupted = True
                self.release()
                return
            self.enter(str(item))
            yield item

    def release(self) -> None:
        """Give back the current hold, if any. Safe to call repeatedly."""
        cm, self._cm, self.current = self._cm, None, None
        if cm is not None:
            cm.__exit__(None, None, None)

    def __enter__(self) -> "FileLockCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False


class GuardedFileLock:
    """A ``file_lock`` factory that cannot leak a hold past the mine.

    ``FileLockCursor`` releases on the paths the miners guard, but not on all
    of them: ``_mine_convos_impl`` takes :data:`OPEN_PHASE` before its
    ``try``/``finally`` and runs its whole per-file loop outside it, and
    ``miner._mine_impl`` opens its collections outside its own ``try``. An
    exception there — a corrupt Chroma index, an unreadable transcript — would
    unwind past every ``release()`` with the hold still taken.

    Off the hub that leaks nothing anybody notices. On the hub the hold *is*
    the exclusive dispatch lock, so a leaked one wedges every subsequent write
    by every agent, permanently and silently: strictly worse than the restart
    storm this whole change exists to stop.

    Rather than restructure three miners' control flow around a guarantee they
    cannot see, the lock's owner enforces it. Wrap the factory, run the mine
    inside ``with GuardedFileLock(factory) as lock: ...``, and any hold still
    outstanding when the mine returns — however it returns — is released.
    """

    def __init__(self, factory: Optional[FileLockFactory]) -> None:
        self._factory = factory
        self._active: Optional[ContextManager] = None

    @property
    def should_stop(self):
        """Forward the wrapped factory's stop predicate, if it has one.

        ``FileLockCursor`` reads this off whatever object it was handed, and
        on the hub that object is the wrapper, not the factory underneath.
        """
        return getattr(self._factory, "should_stop", None)

    def __call__(self, label: str) -> ContextManager:
        return self._track(self._factory(label))

    def _track(self, cm: ContextManager) -> ContextManager:
        outer = self

        class _Tracked:
            def __enter__(self):
                result = cm.__enter__()
                # Recorded only after __enter__ succeeds, for the same reason
                # FileLockCursor assigns late: a factory that raises must not
                # leave something for release_dangling() to exit.
                outer._active = cm
                return result

            def __exit__(self, exc_type, exc, tb):
                if outer._active is cm:
                    outer._active = None
                return cm.__exit__(exc_type, exc, tb)

        return _Tracked()

    def release_dangling(self) -> bool:
        """Release a hold the mine left behind. True if there was one."""
        cm, self._active = self._active, None
        if cm is None:
            return False
        cm.__exit__(None, None, None)
        return True

    def __enter__(self) -> "GuardedFileLock":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release_dangling()
        return False
