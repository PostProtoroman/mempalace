"""redact.py — conservative ingest-time secret redaction.

MemPalace's foundational promise is verbatim storage (see CLAUDE.md:
"Verbatim always"). Credentials are the one deliberate, narrow exception:
a live palace was found holding 77 Anthropic API keys, 6 AWS key IDs, and
55 JWTs mined straight out of transcripts and project files. Storing a
person's prose verbatim is the point; storing their `sk-ant-...` key
verbatim is a liability with no upside.

``redact_secrets`` is intentionally narrow. Every pattern requires a
literal, provider-specific prefix (``sk-ant-``, ``AKIA``, ``ghp_``, ...)
so it only ever fires on strings shaped like real credentials — never on
ordinary words, sentences, or code. When nothing matches, the input comes
back byte-identical (``redacted_text == text``); this is verified by a
negative test per pattern in ``tests/test_redact.py``. This module does
not attempt to catch every possible secret shape (arbitrary passwords,
generic "API_KEY=..." assignments, etc.) — that would risk mangling
prose and code, which the project's design principles forbid trading
away for a broader dragnet.
"""

from __future__ import annotations

import re

# PEM private key blocks span multiple lines, so they run first: matching
# (and replacing) the whole block up front means none of the single-line
# patterns below can later match a fragment of base64 living inside it.
_PEM_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)

# (kind, compiled pattern). Order matters only for the PEM block above —
# every other pattern has a disjoint literal prefix, so match order among
# them cannot change the result.
_PATTERNS: list[tuple[str, "re.Pattern[str]"]] = [
    ("private_key", _PEM_PRIVATE_KEY),
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}")),
    ("openai_key", re.compile(r"sk-proj-[A-Za-z0-9_-]{20,}")),
    ("aws_access_key_id", re.compile(r"AKIA[A-Z0-9]{16}")),
    # Fine-grained GitHub PATs (``github_pat_...``) before classic ones
    # (``ghp_...``) — the prefixes are disjoint so this is belt-and-braces,
    # not load-bearing.
    ("github_fine_grained_pat", re.compile(r"github_pat_[A-Za-z0-9_]{22,}")),
    ("github_pat", re.compile(r"ghp_[A-Za-z0-9]{36}")),
    ("slack_token", re.compile(r"xox[abpr]-[A-Za-z0-9-]{10,}")),
    ("google_api_key", re.compile(r"AIza[A-Za-z0-9_-]{35}")),
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
]


def redact_secrets(text: str) -> tuple[str, dict[str, int]]:
    """Replace conservatively-matched secret patterns in ``text``.

    Each match is replaced with ``[redacted:<kind>]``. Returns
    ``(redacted_text, counts)`` where ``counts`` maps kind -> number of
    matches redacted for that kind. A kind absent from ``counts`` means
    zero matches; ``counts`` is ``{}`` (and ``redacted_text == text``,
    byte-identical) when nothing in ``text`` matched any pattern.

    Falsy input (``None`` or ``""``) is returned unchanged with an empty
    counts dict rather than raising, since call sites in the miners
    already guard for empty content but do so at slightly different
    points in their pipelines.
    """
    if not text:
        return text, {}

    counts: dict[str, int] = {}
    redacted = text
    for kind, pattern in _PATTERNS:

        def _sub(match: "re.Match[str]", _kind: str = kind) -> str:
            counts[_kind] = counts.get(_kind, 0) + 1
            return f"[redacted:{_kind}]"

        redacted = pattern.sub(_sub, redacted)
    return redacted, counts


def redact_if_enabled(text: str, enabled: bool, counts: dict[str, int]) -> str:
    """Redact ``text`` via :func:`redact_secrets` when ``enabled``, folding
    the per-kind match counts into the caller-owned ``counts`` dict
    (mutated in place, added to rather than overwritten — safe to call once
    per file/chunk/exchange against one shared accumulator across a whole
    mine run). Returns ``text`` unchanged when ``enabled`` is False.

    Factored out of the miner call sites (rather than inlining the
    ``if enabled: redacted, file_counts = redact_secrets(text); for kind,
    n in file_counts.items(): ...`` shape at every write path) so each
    caller's own cyclomatic complexity doesn't grow with every additional
    redaction call site — see ``format_redaction_summary`` below for the
    matching rationale on the print side.
    """
    if not enabled:
        return text
    redacted, file_counts = redact_secrets(text)
    for kind, n in file_counts.items():
        counts[kind] = counts.get(kind, 0) + n
    return redacted


def print_redaction_summary(counts: dict[str, int]) -> None:
    """Print the mine-summary ``Secrets redacted: ...`` line, or nothing
    when ``counts`` is empty. The conditional and the print both live in
    here (not at the call site) as a single no-argument-shaped statement —
    ``print_redaction_summary(redaction_counts)`` — so wiring a redaction
    summary into an already-long mine loop function never adds a branch to
    ITS cyclomatic complexity.
    """
    if not counts:
        return
    total = sum(counts.values())
    breakdown = ", ".join(f"{kind}:{n}" for kind, n in sorted(counts.items()))
    print(f"  Secrets redacted: {total} ({breakdown})")
