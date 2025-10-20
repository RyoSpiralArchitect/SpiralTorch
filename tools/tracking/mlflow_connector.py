"""MLflow connector for SpiralTorch search loops."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Optional

from .base import TrackingCallback, TrialEvent

LOGGER = logging.getLogger("spiral.tracking.mlflow")

try:  # pragma: no cover - optional dependency
    import mlflow
except Exception:  # pragma: no cover - optional dependency
    mlflow = None


@contextmanager
def _maybe_run(run_name: Optional[str], nested: bool = True):
    if mlflow is None:
        yield None
        return
    active = mlflow.active_run()
    if active is not None:
        yield active
        return
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        yield run


class MLflowTracker(TrackingCallback):
    """Logs trial events to MLflow if available."""

    def __init__(
        self,
        experiment: Optional[str] = None,
        run_name: Optional[str] = None,
        autolog: bool = False,
        log_params: bool = True,
    ) -> None:
        self._run_name = run_name or "spiral-search"
        self._enabled = mlflow is not None
        self._log_params = log_params
        if not self._enabled:
            LOGGER.warning("MLflow is not installed; MLflowTracker is disabled")
            return
        if experiment:
            mlflow.set_experiment(experiment)
        if autolog:
            try:
                mlflow.autolog(disable=True)  # disable global autologging to avoid interference
            except Exception:  # pragma: no cover - defensive
                LOGGER.debug("Failed to disable global mlflow autologging", exc_info=True)

    def on_trial_start(self, trial: TrialEvent) -> None:
        if not self._enabled:
            return
        with _maybe_run(self._run_name) as run:
            if run is None:
                return
            if self._log_params:
                params = {f"trial_{trial.id}_{k}": v for k, v in trial.params.items()}
                try:
                    mlflow.log_params(params)
                except Exception:  # pragma: no cover - mlflow failure
                    LOGGER.exception("Failed to log MLflow params for trial %s", trial.id)

    def on_trial_end(self, trial: TrialEvent) -> None:
        if not self._enabled:
            return
        with _maybe_run(self._run_name) as run:
            if run is None:
                return
            try:
                if trial.metric is not None:
                    mlflow.log_metric(f"trial/{trial.id}/metric", float(trial.metric))
            except Exception:  # pragma: no cover
                LOGGER.exception("Failed to log MLflow metric for trial %s", trial.id)

    def on_checkpoint(self, checkpoint_json: str) -> None:
        if not self._enabled:
            return
        with _maybe_run(self._run_name) as run:
            if run is None:
                return
            try:
                mlflow.log_text(checkpoint_json, "spiral_checkpoint.json")
            except Exception:  # pragma: no cover
                LOGGER.debug("Failed to log MLflow checkpoint", exc_info=True)


__all__ = ["MLflowTracker"]
