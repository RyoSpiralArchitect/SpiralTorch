"""Weights & Biases connector for SpiralTorch."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .base import TrackingCallback, TrialEvent

LOGGER = logging.getLogger("spiral.tracking.wandb")

try:  # pragma: no cover - optional dependency
    import wandb
except Exception:  # pragma: no cover - optional dependency
    wandb = None


class WandBTracker(TrackingCallback):
    """Logs trial metrics to Weights & Biases."""

    def __init__(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[str] = None,
    ) -> None:
        self._enabled = wandb is not None
        self._tags = [tag.strip() for tag in tags.split(",")] if tags else None
        if not self._enabled:
            LOGGER.warning("wandb is not installed; WandBTracker is disabled")
            self._run = None
            return
        self._run = wandb.init(
            project=project,
            entity=entity,
            name=run_name or "spiral-search",
            reinit=True,
        )
        if self._tags:
            self._run.tags = list(self._tags)

    def _log(self, data: Dict[str, Any]) -> None:
        if not self._enabled or self._run is None:
            return
        try:
            self._run.log(data)
        except Exception:  # pragma: no cover
            LOGGER.exception("Failed to log to wandb")

    def on_trial_start(self, trial: TrialEvent) -> None:
        if not self._enabled or self._run is None:
            return
        try:
            self._run.log({f"trial/{trial.id}/start": 1})
            self._run.config.update({f"trial_{trial.id}_{k}": v for k, v in trial.params.items()}, allow_val_change=True)
        except Exception:  # pragma: no cover
            LOGGER.debug("Failed to log wandb trial start", exc_info=True)

    def on_trial_end(self, trial: TrialEvent) -> None:
        if trial.metric is None:
            return
        self._log({f"trial/{trial.id}/metric": float(trial.metric)})

    def on_checkpoint(self, checkpoint_json: str) -> None:
        if not self._enabled or self._run is None:
            return
        try:
            self._run.summary["spiral_checkpoint"] = checkpoint_json
        except Exception:  # pragma: no cover
            LOGGER.debug("Failed to update wandb summary with checkpoint", exc_info=True)


__all__ = ["WandBTracker"]
