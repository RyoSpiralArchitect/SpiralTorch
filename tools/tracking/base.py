"""Common interfaces for experiment tracking connectors."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

LOGGER = logging.getLogger("spiral.tracking")


@dataclass
class TrialEvent:
    """Lightweight container describing a trial for trackers."""

    id: int
    params: Dict[str, Any]
    metric: Optional[float] = None


class TrackingCallback:
    """Interface for logging trial lifecycle events."""

    def on_trial_start(self, trial: TrialEvent) -> None:  # pragma: no cover - optional override
        LOGGER.debug("trial %s started", trial.id)

    def on_trial_end(self, trial: TrialEvent) -> None:  # pragma: no cover - optional override
        LOGGER.debug("trial %s finished metric=%s", trial.id, trial.metric)

    def on_checkpoint(self, checkpoint_json: str) -> None:  # pragma: no cover - optional override
        LOGGER.debug("checkpoint updated (%s bytes)", len(checkpoint_json))


class CompositeTracker(TrackingCallback):
    """Fan-out tracker composing multiple callbacks."""

    def __init__(self, callbacks: Iterable[TrackingCallback]):
        self._callbacks: List[TrackingCallback] = list(callbacks)

    def on_trial_start(self, trial: TrialEvent) -> None:
        for cb in self._callbacks:
            cb.on_trial_start(trial)

    def on_trial_end(self, trial: TrialEvent) -> None:
        for cb in self._callbacks:
            cb.on_trial_end(trial)

    def on_checkpoint(self, checkpoint_json: str) -> None:
        for cb in self._callbacks:
            cb.on_checkpoint(checkpoint_json)


class ConsoleTracker(TrackingCallback):
    """Simple tracker that pretty prints progress to stdout."""

    def __init__(self, stream=None) -> None:
        self._stream = stream or LOGGER

    def on_trial_start(self, trial: TrialEvent) -> None:
        LOGGER.info("[trial %s] started", trial.id)

    def on_trial_end(self, trial: TrialEvent) -> None:
        LOGGER.info("[trial %s] metric=%s", trial.id, trial.metric)

    def on_checkpoint(self, checkpoint_json: str) -> None:
        LOGGER.debug("checkpoint snapshot: %s", checkpoint_json[:128])


def _coerce_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def build_tracker(name: str, **kwargs: Any) -> Optional[TrackingCallback]:
    """Instantiate a tracker by name."""

    if not name:
        return None
    name = name.lower()
    if name in {"console", "stdout"}:
        return ConsoleTracker()
    if name in {"mlflow", "ml"}:
        from .mlflow_connector import MLflowTracker

        coerced = {key: _coerce_value(str(value)) for key, value in kwargs.items()}
        return MLflowTracker(**coerced)
    if name in {"wandb", "weightsandbiases"}:
        from .wandb_connector import WandBTracker

        coerced = {key: _coerce_value(str(value)) for key, value in kwargs.items()}
        return WandBTracker(**coerced)
    LOGGER.warning("Unknown tracker '%s'", name)
    return None


__all__ = [
    "TrackingCallback",
    "CompositeTracker",
    "ConsoleTracker",
    "TrialEvent",
    "build_tracker",
]
