"""Tests for the `dedup` CLI command -- dry-run must be the default."""

from argparse import Namespace

import mempalace.cli as cli_mod
import mempalace.dedup as dedup_mod


def test_dedup_defaults_to_dry_run(monkeypatch):
    calls = []
    monkeypatch.setattr(
        dedup_mod,
        "dedup_palace",
        lambda **kwargs: calls.append(kwargs),
    )
    cli_mod.cmd_dedup(Namespace(threshold=None, apply=False, stats=False, wing=None, source=None))
    assert len(calls) == 1
    assert calls[0]["dry_run"] is True


def test_dedup_apply_flag_disables_dry_run(monkeypatch):
    calls = []
    monkeypatch.setattr(
        dedup_mod,
        "dedup_palace",
        lambda **kwargs: calls.append(kwargs),
    )
    cli_mod.cmd_dedup(Namespace(threshold=None, apply=True, stats=False, wing=None, source=None))
    assert calls[0]["dry_run"] is False


def test_dedup_stats_only_does_not_call_dedup_palace(monkeypatch):
    palace_calls = []
    stats_calls = []
    monkeypatch.setattr(dedup_mod, "dedup_palace", lambda **kwargs: palace_calls.append(kwargs))
    monkeypatch.setattr(dedup_mod, "show_stats", lambda **kwargs: stats_calls.append(kwargs))
    cli_mod.cmd_dedup(Namespace(threshold=None, apply=False, stats=True, wing=None, source=None))
    assert palace_calls == []
    assert len(stats_calls) == 1


def test_dedup_default_threshold_used_when_none(monkeypatch):
    calls = []
    monkeypatch.setattr(dedup_mod, "dedup_palace", lambda **kwargs: calls.append(kwargs))
    cli_mod.cmd_dedup(Namespace(threshold=None, apply=False, stats=False, wing=None, source=None))
    assert calls[0]["threshold"] == dedup_mod.DEFAULT_THRESHOLD
