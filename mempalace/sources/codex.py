"""Codex CLI session adapter (RFC 002 source adapter).

Turns OpenAI Codex rollout files (``~/.codex/sessions/YYYY/MM/DD/
rollout-*.jsonl``) into one drawer per completed conversation turn, so a
Codex session is persisted into the palace as it happens rather than only
when someone remembers to run ``mempalace mine``.

Two wire formats coexist in real corpora and both are supported:

- **legacy** — ``event_msg`` entries of type ``user_message`` /
  ``agent_message`` with the text in ``payload.message``.
- **current** (``item_completed``) — ``event_msg`` entries whose
  ``payload.item`` is a ``UserMessage`` / ``AgentMessage`` with the text in
  ``item.content[].text``, carrying native ``thread_id`` / ``turn_id`` /
  ``item.id`` identities.

Identity is what makes re-scanning free. The current format supplies real
per-item identities, which we use verbatim. Legacy rollouts carry no message
id, so a stable one is *derived* from (rollout file, line ordinal) — rollouts
are append-only, so a line's ordinal never changes and the derived id is
stable across restarts and rescans. It is deliberately scoped to the file
rather than the thread: a resumed session reuses its thread id in a new
rollout with ordinals starting over, so a thread-scoped derivation names two
unrelated turns (see :func:`_derived_item_id`).

Deliberately never ingested:

- ``compacted`` records. Their ``replacement_history`` re-embeds earlier
  conversation verbatim so the model can resume with a condensed context;
  filing it would duplicate every pre-compaction message once per
  compaction, and leak developer/system prompts in as if they were turns.
  The exclusion is structural: ``compacted`` is a top-level entry type, so
  it never reaches the ``event_msg`` branch and ``replacement_history`` is
  never read.
- Automation sessions (``originator == "codex_exec"`` / ``source ==
  "exec"``). These are non-interactive runs; they are machine output, not a
  conversation the user had.
- ``item_completed`` items of non-conversation types (``Reasoning``,
  ``CommandExecution``, ``ContextCompaction``, ``FileChange`` …).

Subagent sessions *are* ingested, with their lineage preserved:
``thread_source`` and ``parent_thread_id`` ride along as drawer metadata so
a subagent turn stays attributable to the thread that spawned it.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from .base import (
    AdapterSchema,
    BaseSourceAdapter,
    DrawerRecord,
    FieldSpec,
    RouteHint,
    SourceItemMetadata,
    SourceRef,
)

# item_completed payloads carry many item types; only these two are turns.
_ITEM_ROLES = {"UserMessage": "user", "AgentMessage": "assistant"}

# Legacy event_msg payload types that carry a conversation turn.
_LEGACY_ROLES = {"user_message": "user", "agent_message": "assistant"}

# session_meta values that mark a non-interactive automation run.
_AUTOMATION_ORIGINATORS = frozenset({"codex_exec"})
_AUTOMATION_SOURCES = frozenset({"exec"})

ADAPTER_VERSION = "1.0.0"


@dataclass(frozen=True)
class CodexTurn:
    """One completed conversation turn extracted from a rollout."""

    thread_id: str
    turn_id: str
    item_id: str
    role: str
    text: str
    timestamp: str
    ordinal: int
    wire_format: str  # "item_completed" | "legacy"
    thread_source: str
    parent_thread_id: str
    cwd: str
    originator: str

    @property
    def drawer_id(self) -> str:
        """Deterministic id: rescanning the same turn is an idempotent upsert.

        ``item_id`` is already unique per thread in the current format, and
        derived-but-stable for legacy, so (thread, item) is a safe key. Both
        halves are hashed rather than interpolated raw because thread ids and
        derived ids come from the file and must not be able to inject the
        separator and collide with a different pair.
        """
        digest = hashlib.sha256(f"{self.thread_id}\x00{self.item_id}".encode("utf-8")).hexdigest()
        return f"codex_{digest[:32]}"


def _is_automation(meta: dict) -> bool:
    """True when session_meta marks a non-interactive ``codex exec`` run."""
    originator = str(meta.get("originator") or "").strip().lower()
    source = (
        str(meta.get("source") or "").strip().lower() if isinstance(meta.get("source"), str) else ""
    )
    return originator in _AUTOMATION_ORIGINATORS or source in _AUTOMATION_SOURCES


def _item_text(item: dict) -> str:
    """Join the text blocks of a conversation item, verbatim.

    Block type casing differs by role in real data: ``UserMessage`` uses
    ``"text"`` while ``AgentMessage`` uses ``"Text"``. The comparison is
    therefore case-insensitive — a case-sensitive match silently drops every
    assistant turn while still looking like a successful ingest. Non-text
    blocks (``local_image``) are skipped.
    """
    blocks = item.get("content")
    if not isinstance(blocks, list):
        return ""
    parts = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if str(block.get("type", "")).lower() != "text":
            continue
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text)
    return "\n".join(parts)


def _derived_item_id(rollout_key: str, ordinal: int) -> str:
    """Stable identity for a legacy turn, which carries no message id.

    Scoped to the *rollout file*, not the thread. Resuming a session reuses
    its thread id in a brand new rollout file, and ordinals restart, so
    (thread, ordinal) names two unrelated turns — measured across the local
    corpus that collided 438 times, every one of them cross-file and 435 with
    genuinely different text, which would have silently overwritten real
    turns. Rollout filenames carry a timestamp and uuid and are unique, so
    (rollout, ordinal) is unique and still stable: rollouts are append-only,
    so a line's ordinal never changes.

    The file *stem* is used rather than the full path so relocating the
    sessions directory does not re-mint every id.
    """
    digest = hashlib.sha256(f"{rollout_key}\x00{ordinal}".encode("utf-8")).hexdigest()
    return f"legacy-{digest[:24]}"


def iter_turns(path: str) -> Iterator[CodexTurn]:
    """Yield every completed conversation turn in one rollout file.

    Malformed lines are skipped rather than aborting the file: a partially
    written trailing line is normal for a session that is still running, and
    the follower is expected to pick it up on the next pass.
    """
    p = Path(path)
    if not stat.S_ISREG(p.stat().st_mode):
        raise OSError(f"Refusing non-regular file: {path}")

    rollout_key = p.stem
    session_meta: dict = {}
    automation = False
    saw_meta = False

    with open(p, "r", encoding="utf-8", errors="replace") as fh:
        for line_no, raw in enumerate(fh):
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue

            entry_type = entry.get("type")

            if entry_type == "session_meta":
                payload = entry.get("payload")
                if isinstance(payload, dict):
                    # Resume appends session_meta again; first one wins for
                    # identity, but automation is sticky once seen.
                    if not saw_meta:
                        session_meta = payload
                        saw_meta = True
                    automation = automation or _is_automation(payload)
                continue

            if automation:
                # Non-interactive run: stop before extracting anything.
                return

            # Everything that is not event_msg is skipped, which is what keeps
            # `compacted` (and its replacement_history) structurally out.
            if entry_type != "event_msg":
                continue

            payload = entry.get("payload")
            if not isinstance(payload, dict):
                continue

            thread_id = str(
                payload.get("thread_id")
                or session_meta.get("session_id")
                or session_meta.get("id")
                or p.stem
            )
            timestamp = str(entry.get("timestamp") or "")
            ordinal = entry.get("ordinal")
            if not isinstance(ordinal, int):
                ordinal = line_no

            payload_type = payload.get("type", "")

            if payload_type == "item_completed":
                item = payload.get("item")
                if not isinstance(item, dict):
                    continue
                role = _ITEM_ROLES.get(item.get("type"))
                if role is None:
                    continue
                text = _item_text(item)
                if not text.strip():
                    continue
                item_id = str(item.get("id") or _derived_item_id(rollout_key, ordinal))
                yield CodexTurn(
                    thread_id=thread_id,
                    turn_id=str(payload.get("turn_id") or ""),
                    item_id=item_id,
                    role=role,
                    text=text,
                    timestamp=timestamp,
                    ordinal=ordinal,
                    wire_format="item_completed",
                    thread_source=str(session_meta.get("thread_source") or ""),
                    parent_thread_id=str(session_meta.get("parent_thread_id") or ""),
                    cwd=str(session_meta.get("cwd") or ""),
                    originator=str(session_meta.get("originator") or ""),
                )
                continue

            role = _LEGACY_ROLES.get(payload_type)
            if role is None:
                continue
            text = payload.get("message")
            if not isinstance(text, str) or not text.strip():
                continue
            yield CodexTurn(
                thread_id=thread_id,
                turn_id="",
                item_id=_derived_item_id(rollout_key, ordinal),
                role=role,
                text=text,
                timestamp=timestamp,
                ordinal=ordinal,
                wire_format="legacy",
                thread_source=str(session_meta.get("thread_source") or ""),
                parent_thread_id=str(session_meta.get("parent_thread_id") or ""),
                cwd=str(session_meta.get("cwd") or ""),
                originator=str(session_meta.get("originator") or ""),
            )


def turn_to_drawer(turn: CodexTurn, source_file: str) -> DrawerRecord:
    """Build the drawer for one turn.

    The document is the user's exact words with a role prefix, matching the
    shape ``sweeper.py`` already writes for Claude Code transcripts so both
    read back the same way.
    """
    return DrawerRecord(
        content=f"{turn.role.upper()}: {turn.text}",
        source_file=source_file,
        chunk_index=turn.ordinal,
        metadata={
            "session_id": turn.thread_id,
            "thread_id": turn.thread_id,
            "turn_id": turn.turn_id,
            "item_id": turn.item_id,
            "message_uuid": turn.item_id,
            "role": turn.role,
            "timestamp": turn.timestamp,
            "ordinal": turn.ordinal,
            "wire_format": turn.wire_format,
            "thread_source": turn.thread_source,
            "parent_thread_id": turn.parent_thread_id,
            "cwd": turn.cwd,
            "originator": turn.originator,
            "agent": "codex",
            "ingest_mode": "codex_session",
            "drawer_id": turn.drawer_id,
        },
        route_hint=RouteHint(wing="sessions"),
    )


def iter_rollouts(root: str) -> list[Path]:
    """Every rollout file under a Codex sessions root, oldest path first."""
    base = Path(root).expanduser()
    if not base.exists():
        return []
    if base.is_file():
        return [base]
    return sorted(base.rglob("rollout-*.jsonl"))


class CodexSessionAdapter(BaseSourceAdapter):
    """RFC 002 adapter over Codex CLI rollout files."""

    name = "codex"
    adapter_version = ADAPTER_VERSION
    capabilities = frozenset({"supports_incremental"})
    supported_modes = frozenset({"whole_record"})
    # Turn text is stored byte-for-byte; the only added bytes are the
    # "USER: " / "ASSISTANT: " role prefix, which is structure, not a
    # rewrite of what was said.
    declared_transformations = frozenset({"role_prefix"})
    default_privacy_class = "pii_potential"

    def describe_schema(self) -> AdapterSchema:
        f = FieldSpec
        return AdapterSchema(
            version=ADAPTER_VERSION,
            fields={
                "session_id": f(
                    type="string", required=True, description="Codex thread id", indexed=True
                ),
                "thread_id": f(
                    type="string", required=True, description="Codex thread id", indexed=True
                ),
                "turn_id": f(type="string", required=False, description="Codex turn id"),
                "item_id": f(
                    type="string", required=True, description="Codex item id (derived for legacy)"
                ),
                # Mirrors item_id under the name sweeper.py already uses for
                # Claude Code transcripts, so message-level tooling can key on
                # one field across both sources.
                "message_uuid": f(
                    type="string", required=True, description="Alias of item_id", indexed=True
                ),
                "role": f(
                    type="string", required=True, description="user | assistant", indexed=True
                ),
                "timestamp": f(
                    type="string", required=False, description="ISO-8601 turn timestamp"
                ),
                "ordinal": f(type="int", required=True, description="Position in the rollout"),
                "wire_format": f(
                    type="string", required=True, description="item_completed | legacy"
                ),
                "thread_source": f(
                    type="string", required=False, description="user | subagent | …"
                ),
                "parent_thread_id": f(
                    type="string",
                    required=False,
                    description="Spawning thread, for subagents",
                    indexed=True,
                ),
                "cwd": f(type="string", required=False, description="Session working directory"),
                "originator": f(type="string", required=False, description="Codex client identity"),
                "agent": f(
                    type="string", required=True, description="Always 'codex'", indexed=True
                ),
                "ingest_mode": f(
                    type="string", required=True, description="Always 'codex_session'"
                ),
                "drawer_id": f(type="string", required=True, description="Deterministic drawer id"),
            },
        )

    def ingest(self, *, source: SourceRef, palace) -> Iterator[object]:
        root = source.local_path or os.path.join(os.path.expanduser("~"), ".codex", "sessions")
        for rollout in iter_rollouts(root):
            try:
                st = rollout.stat()
            except OSError:
                continue
            yield SourceItemMetadata(
                source_file=str(rollout),
                version=f"{st.st_mtime_ns}:{st.st_size}",
                size_hint=st.st_size,
                route_hint=RouteHint(wing="sessions"),
            )
            if getattr(palace, "_skip_requested", False):
                palace._skip_requested = False
                continue
            try:
                turns = list(iter_turns(str(rollout)))
            except OSError:
                continue
            for turn in turns:
                yield turn_to_drawer(turn, str(rollout))

    def is_current(
        self,
        *,
        item: SourceItemMetadata,
        existing_metadata: Optional[dict],
    ) -> bool:
        """Skip a rollout whose recorded version token still matches.

        The token is (mtime_ns, size); a rollout that grew by one turn changes
        both. A stored token that is absent or different means re-read, which
        is safe because the drawer ids are deterministic.
        """
        if not existing_metadata:
            return False
        return existing_metadata.get("source_version") == item.version
