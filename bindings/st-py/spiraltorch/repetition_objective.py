"""Thin client for Rust-owned repetition objective coefficients."""

from __future__ import annotations

import sys
from typing import Any

__all__ = ["zspace_repetition_objective_control"]


def zspace_repetition_objective_control(
    *,
    config: dict[str, object],
    base_strength: float,
    completed_update_slots: int,
    active_position_count: int,
    eligible_target_count: int,
) -> dict[str, Any]:
    """Scale an active-position mean; counts come from a validated Rust plan.

    Eligibility is candidate-source-specific. Normalization is per microbatch,
    not a global token average across accumulated batches or distributed ranks.
    """
    package = sys.modules.get(__package__ or "spiraltorch")
    operation = getattr(
        getattr(package, "_rs", None), "_zspace_repetition_objective_control", None
    )
    if not callable(operation):
        raise RuntimeError(
            "repetition objective control requires the compiled Rust core"
        )
    result = operation(
        {
            "config": config,
            "base_strength": base_strength,
            "completed_update_slots": completed_update_slots,
            "active_position_count": active_position_count,
            "eligible_target_count": eligible_target_count,
        }
    )
    if not isinstance(result, dict):
        raise RuntimeError("Rust repetition objective returned a non-mapping")
    return result
