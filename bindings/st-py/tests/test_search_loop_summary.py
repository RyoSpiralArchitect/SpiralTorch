"""Unit tests for SearchLoop summary helpers exposed to Python."""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import textwrap
import weakref

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


def test_tracker_callbacks_can_reenter_search_loop_without_deadlock() -> None:
    script = textwrap.dedent(
        """
        import json
        from spiraltorch.hpo import SearchLoop

        space = [{"name": "lr", "type": "float", "low": 0.0001, "high": 0.1}]
        strategy = {"name": "bayesian", "seed": 123, "exploration": 0.2}

        class Tracker:
            def __init__(self):
                self.loop = None
                self.events = []

            def on_trial_start(self, trial):
                self.events.append("start")
                summary = self.loop.summary()
                pending = self.loop.pending()
                self.events.append(
                    ["summary", summary["pending_trials"], len(pending)]
                )
                self.loop.checkpoint()
                self.events.append("start_return")

            def on_trial_end(self, trial, metric):
                completed = self.loop.completed()
                best = self.loop.best_trial()
                self.events.append(
                    ["end", len(completed), best["id"], metric, trial["metric"]]
                )

            def on_checkpoint(self, checkpoint):
                state = json.loads(checkpoint)
                self.events.append(["checkpoint", len(state["pending"])])

        tracker = Tracker()
        loop = SearchLoop.create(space, strategy, tracker=tracker)
        tracker.loop = loop
        trial = loop.suggest()
        loop.observe(trial["id"], 0.25)
        print(json.dumps({"trial": trial["id"], "events": tracker.events}))
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload == {
        "trial": 0,
        "events": [
            "start",
            ["summary", 1, 1],
            "start_return",
            ["checkpoint", 1],
            ["end", 1, 0, 0.25, 0.25],
        ],
    }


def test_tracker_callback_errors_are_unraisable_and_state_remains_committed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RaisingTracker:
        def on_trial_start(self, _trial: dict[str, object]) -> None:
            raise RuntimeError("tracker boom")

    unraisable: list[object] = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)
    loop = SearchLoop.create(SPACE, STRATEGY, tracker=RaisingTracker())

    trial = loop.suggest()

    assert trial["id"] == 0
    assert loop.summary()["pending_trials"] == 1
    assert len(unraisable) == 1
    assert isinstance(unraisable[0].exc_value, RuntimeError)
    assert str(unraisable[0].exc_value) == "tracker boom"


def test_reentrant_tracker_event_storm_is_bounded() -> None:
    script = textwrap.dedent(
        """
        from spiraltorch.hpo import SearchLoop

        space = [{"name": "lr", "type": "float", "low": 0.0001, "high": 0.1}]
        strategy = {"name": "bayesian", "seed": 123, "exploration": 0.2}

        class Tracker:
            def __init__(self):
                self.loop = None
                self.checkpoints = 0

            def on_checkpoint(self, _checkpoint):
                self.checkpoints += 1
                self.loop.checkpoint()

        tracker = Tracker()
        loop = SearchLoop.create(space, strategy, tracker=tracker)
        tracker.loop = loop
        loop.checkpoint()
        print(tracker.checkpoints)
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    assert completed.stdout.strip().splitlines()[-1] == "1024"
    assert "tracker callback event budget exceeded" in completed.stderr


def test_tracker_loop_reference_cycle_is_collectible() -> None:
    class Tracker:
        pass

    tracker = Tracker()
    loop = SearchLoop.create(SPACE, STRATEGY, tracker=tracker)
    tracker.loop = loop
    tracker_ref = weakref.ref(tracker)

    del loop
    del tracker
    gc.collect()

    assert tracker_ref() is None
