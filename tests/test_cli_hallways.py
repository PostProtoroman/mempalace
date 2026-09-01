"""Tests for the `hallways` CLI command."""

from argparse import Namespace

import mempalace.hallways as hallways_mod
from mempalace.cli import cmd_hallways


def test_lists_sorted_by_count(monkeypatch, capsys):
    rows = [
        {
            "entity_a": "C",
            "entity_b": "D",
            "co_occurrence_count": 1,
            "wing": "w",
            "label": "C <-> D (x1)",
        },
        {
            "entity_a": "A",
            "entity_b": "B",
            "co_occurrence_count": 3,
            "wing": "w",
            "label": "A <-> B (x3)",
        },
    ]
    monkeypatch.setattr(hallways_mod, "list_hallways", lambda wing=None, config=None: list(rows))
    cmd_hallways(Namespace(wing=None, limit=50))
    out = capsys.readouterr().out
    assert "2 hallway(s)" in out
    assert "A <-> B (x3)" in out
    # Highest co-occurrence first.
    assert out.index("A <-> B") < out.index("C <-> D")


def test_respects_limit(monkeypatch, capsys):
    rows = [
        {"entity_a": f"E{i}", "entity_b": "X", "co_occurrence_count": i, "label": f"E{i} <-> X"}
        for i in range(5)
    ]
    monkeypatch.setattr(hallways_mod, "list_hallways", lambda wing=None, config=None: list(rows))
    cmd_hallways(Namespace(wing=None, limit=2))
    assert capsys.readouterr().out.count("<->") == 2


def test_negative_limit_shows_nothing_not_tail(monkeypatch, capsys):
    rows = [
        {"entity_a": f"E{i}", "entity_b": "X", "co_occurrence_count": i, "label": f"E{i} <-> X"}
        for i in range(5)
    ]
    monkeypatch.setattr(hallways_mod, "list_hallways", lambda wing=None, config=None: list(rows))
    cmd_hallways(Namespace(wing=None, limit=-2))
    # A negative limit must not slice from the end (which would print all-but-2).
    assert capsys.readouterr().out.count("<->") == 0


def test_empty_message(monkeypatch, capsys):
    monkeypatch.setattr(hallways_mod, "list_hallways", lambda wing=None, config=None: [])
    cmd_hallways(Namespace(wing="x", limit=50))
    assert "No hallways yet" in capsys.readouterr().out


def test_explicit_palace_scopes_hallway_listing(monkeypatch, tmp_path):
    calls = []

    def fake_list(wing=None, config=None):
        calls.append((wing, config.palace_path))
        return []

    selected = tmp_path / "selected" / "palace"
    monkeypatch.setattr(hallways_mod, "list_hallways", fake_list)

    cmd_hallways(Namespace(wing="wing_aya", limit=50, palace=str(selected)))

    assert calls == [("wing_aya", str(selected))]


def test_prune_dry_run_default(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        hallways_mod,
        "prune_hallways",
        lambda **kwargs: (
            calls.append(kwargs)
            or {
                "total": 5,
                "threshold": kwargs["threshold"],
                "wing": kwargs["wing"],
                "removed_count": 2,
                "kept_count": 3,
                "estimated_bytes_saved": 120,
                "dry_run": kwargs["dry_run"],
            }
        ),
    )
    cmd_hallways(Namespace(wing=None, limit=50, hallways_action="prune", threshold=3, apply=False))
    assert calls[0]["dry_run"] is True
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "Would remove: 2 / 5" in out


def test_prune_apply_disables_dry_run(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        hallways_mod,
        "prune_hallways",
        lambda **kwargs: (
            calls.append(kwargs)
            or {
                "total": 5,
                "threshold": kwargs["threshold"],
                "wing": kwargs["wing"],
                "removed_count": 2,
                "kept_count": 3,
                "estimated_bytes_saved": 120,
                "dry_run": kwargs["dry_run"],
            }
        ),
    )
    cmd_hallways(Namespace(wing=None, limit=50, hallways_action="prune", threshold=3, apply=True))
    assert calls[0]["dry_run"] is False
    out = capsys.readouterr().out
    assert "LIVE" in out
    assert "Removed: 2 / 5" in out
