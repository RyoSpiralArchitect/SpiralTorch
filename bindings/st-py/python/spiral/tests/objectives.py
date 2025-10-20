"""Objectives used by integration tests."""

from __future__ import annotations

from typing import Dict


def quadratic_objective(params: Dict[str, float], *, offset: float = 0.0) -> float:
    """Smooth deterministic objective for exercising the search loop."""

    lr = float(params.get("lr", 0.0))
    layers = float(params.get("layers", 0.0))
    activation = params.get("activation", "relu")
    categorical_penalty = 0.0 if activation == "gelu" else 0.3
    return ((lr - 0.05) ** 2) + ((layers - 3.0) ** 2) * 0.05 + categorical_penalty + offset
