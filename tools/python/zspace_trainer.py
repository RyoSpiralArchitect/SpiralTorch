# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compatibility adapter for the Rust-backed public Z-space trainer.

The optimizer and regularizer semantics live in
``st-core::runtime::zspace_optimizer``. This module intentionally contains no
fallback implementation: constrained clients should install a SpiralTorch
wheel and use the same versioned Rust contract as Python and WASM.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from spiraltorch import ZMetrics
from spiraltorch import ZSpaceTrainer as _ZSpaceTrainer

__all__ = ["ZMetrics", "ZSpaceTrainer", "step_many"]


class ZSpaceTrainer(_ZSpaceTrainer):
    """Legacy import path for :class:`spiraltorch.ZSpaceTrainer`."""

    @property
    def z(self) -> list[float]:
        """Mutable legacy view; transitions are still evaluated in Rust."""

        return self._z  # type: ignore[attr-defined]

    @z.setter
    def z(self, values: Iterable[float]) -> None:
        with self._optimizer_lock:  # type: ignore[attr-defined]
            checkpoint = self.state_dict()
            checkpoint["z"] = list(values)
            self.load_state_dict(checkpoint)


def step_many(
    trainer: ZSpaceTrainer,
    samples: Iterable[Mapping[str, Any] | ZMetrics],
) -> list[float]:
    for metrics in samples:
        trainer.step(metrics)
    return trainer.state


if __name__ == "__main__":  # pragma: no cover - smoke example
    trainer = ZSpaceTrainer(z_dim=4)
    history = [
        {
            "speed": 0.8,
            "mem": 0.5,
            "stab": 0.6,
            "gradient": [0.1, -0.2, 0.3, -0.1],
        },
        {
            "speed": 0.6,
            "mem": 0.4,
            "stab": 0.7,
            "gradient": [0.05, -0.05, 0.1, 0.0],
        },
    ]
    print("Final Z state:", [round(value, 4) for value in step_many(trainer, history)])
