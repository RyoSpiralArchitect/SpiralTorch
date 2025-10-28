"""Unit tests for SearchLoop summary helpers exposed to Python."""

from __future__ import annotations

import pytest

pytest.importorskip("spiraltorch")

from spiraltorch.hpo import SearchLoop


SPACE = [
    {"name": "lr", "type": "float", "low": 0.0001, "high": 0.1},
]

STRATEGY = {"name": "bayesian", "seed": 123, "exploration": 0.2}


def test_best_trial_and_summary_minimize() -> None:
    loop = SearchLoop.create(SPACE, STRATEGY, maximize=False)
    trial_a = loop.suggest()
    loop.observe(trial_a["id"], 0.8)
    trial_b = loop.suggest()
    loop.observe(trial_b["id"], 0.2)

    best = loop.best_trial()
    assert best is not None
    assert best["id"] == trial_b["id"]
    assert pytest.approx(best["metric"]) == 0.2

    summary = loop.summary()
    assert summary["objective"] == "minimize"
    assert summary["completed_trials"] == 2
    assert summary["pending_trials"] == 0
    assert summary["best_trial"]["id"] == best["id"]


def test_best_trial_and_summary_maximize() -> None:
    loop = SearchLoop.create(SPACE, STRATEGY, maximize=True)
    t1 = loop.suggest()
    loop.observe(t1["id"], 0.3)
    t2 = loop.suggest()
    loop.observe(t2["id"], 0.9)

    best = loop.best_trial()
    assert best is not None
    assert best["id"] == t2["id"]

    summary = loop.summary()
    assert summary["objective"] == "maximize"
    assert summary["best_trial"]["id"] == t2["id"]


def test_best_trial_none_when_unobserved() -> None:
    loop = SearchLoop.create(SPACE, STRATEGY)
    loop.suggest()

    assert loop.best_trial() is None
    summary = loop.summary()
    assert summary["best_trial"] is None
