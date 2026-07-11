"""Unit tests for SearchLoop summary helpers exposed to Python."""

from __future__ import annotations

import json

import pytest

try:
    pytest.importorskip("spiraltorch")
except AttributeError as exc:  # pragma: no cover - environment-specific
    pytest.skip(
        f"spiraltorch import failed because torch is unavailable: {exc}",
        allow_module_level=True,
    )

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


@pytest.mark.parametrize(
    "space",
    [
        [{"name": "lr", "type": "float", "low": float("nan"), "high": 0.1}],
        [{"name": "layers", "type": "int", "low": 4, "high": 1}],
        [
            {"name": "duplicate", "type": "int", "low": 1, "high": 2},
            {"name": "duplicate", "type": "float", "low": 0.0, "high": 1.0},
        ],
    ],
)
def test_create_rejects_invalid_search_spaces(space: list[dict[str, object]]) -> None:
    with pytest.raises(ValueError, match="invalid parameter|defined more than once"):
        SearchLoop.create(space, STRATEGY)


def test_create_rejects_zero_resource_slots() -> None:
    with pytest.raises(ValueError, match="max_concurrent must be positive"):
        SearchLoop.create(SPACE, STRATEGY, {"max_concurrent": 0})


@pytest.mark.parametrize("metric", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_metric_does_not_consume_trial(metric: float) -> None:
    loop = SearchLoop.create(SPACE, STRATEGY)
    trial = loop.suggest()
    pending = loop.pending()

    with pytest.raises(ValueError, match="must be finite"):
        loop.observe(trial["id"], metric)

    assert loop.pending() == pending
    assert loop.completed() == []
    loop.observe(trial["id"], 0.5)
    assert loop.pending() == []


def test_from_checkpoint_rejects_inconsistent_scheduler_state() -> None:
    loop = SearchLoop.create(SPACE, STRATEGY)
    loop.suggest()
    checkpoint = json.loads(loop.checkpoint())
    checkpoint["scheduler"]["active_slots"] = 0

    with pytest.raises(ValueError, match="active slot count"):
        SearchLoop.from_checkpoint(SPACE, json.dumps(checkpoint))
