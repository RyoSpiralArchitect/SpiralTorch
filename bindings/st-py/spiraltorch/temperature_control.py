"""Thin Python client for Rust-owned stateful Z-space temperature control."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence
from typing import Any

ZSPACE_TEMPERATURE_CONTROL_CONTRACT_VERSION = (
    "spiraltorch.zspace_temperature_control.v1"
)
ZSPACE_TEMPERATURE_CONTROL_KIND = "spiraltorch.zspace_temperature_control"
ZSPACE_TEMPERATURE_CONTROL_SEMANTIC_OWNER = (
    "st-core::inference::temperature_control"
)
ZSPACE_TEMPERATURE_CONTROL_SEMANTIC_BACKEND = "rust"
ZSPACE_TEMPERATURE_CONTROL_BACKEND = "spiraltorch_temperature_control_core"

__all__ = [
    "ZSPACE_TEMPERATURE_CONTROL_BACKEND",
    "ZSPACE_TEMPERATURE_CONTROL_CONTRACT_VERSION",
    "ZSPACE_TEMPERATURE_CONTROL_KIND",
    "ZSPACE_TEMPERATURE_CONTROL_SEMANTIC_BACKEND",
    "ZSPACE_TEMPERATURE_CONTROL_SEMANTIC_OWNER",
    "zspace_temperature_control",
]


def _native_temperature_control(request: Mapping[str, object]) -> dict[str, Any]:
    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    apply_control = getattr(native, "_zspace_temperature_control", None)
    if not callable(apply_control):
        raise RuntimeError(
            "Z-space temperature control requires the compiled Rust semantic core; "
            "rebuild or reinstall SpiralTorch with _zspace_temperature_control"
        )
    contract = apply_control(dict(request))
    if not isinstance(contract, Mapping):
        raise RuntimeError(
            "native Z-space temperature control returned a non-mapping payload"
        )
    return dict(contract)


def _require_finite_number(contract: Mapping[str, Any], field: str) -> None:
    value = contract.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise RuntimeError(
            f"native Z-space temperature control returned invalid {field}"
        )


def _validate_contract(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("kind") != ZSPACE_TEMPERATURE_CONTROL_KIND
        or contract.get("contract_version")
        != ZSPACE_TEMPERATURE_CONTROL_CONTRACT_VERSION
        or contract.get("semantic_owner")
        != ZSPACE_TEMPERATURE_CONTROL_SEMANTIC_OWNER
        or contract.get("semantic_backend")
        != ZSPACE_TEMPERATURE_CONTROL_SEMANTIC_BACKEND
        or contract.get("backend") != ZSPACE_TEMPERATURE_CONTROL_BACKEND
    ):
        raise RuntimeError(
            "native Z-space temperature control returned an untrusted contract"
        )
    for field in ("config", "previous_state", "next_state", "effects"):
        if not isinstance(contract.get(field), Mapping):
            raise RuntimeError(
                f"native Z-space temperature control returned invalid {field}"
            )
    count = contract.get("probability_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise RuntimeError(
            "native Z-space temperature control returned invalid probability_count"
        )
    for field in (
        "input_probability_sum",
        "probability_sum_tolerance",
        "entropy",
        "entropy_error",
        "temperature_after_entropy",
        "temperature_after_z",
        "temperature_after_scale",
        "temperature",
    ):
        _require_finite_number(contract, field)
    for state_field in ("previous_state", "next_state"):
        state = contract[state_field]
        for field in (
            "temperature",
            "z_memory",
            "scale_memory",
            "gradient_pressure",
            "gradient_entropy_bias",
        ):
            _require_finite_number(state, field)


def zspace_temperature_control(
    probabilities: Sequence[float | int],
    *,
    config: Mapping[str, object],
    state: Mapping[str, object],
    feedback: Mapping[str, object] | None = None,
    gradient_heat: float | None = None,
) -> dict[str, Any]:
    """Advance the canonical Rust temperature controller by one atomic step."""

    request: dict[str, object] = {
        "probabilities": list(probabilities),
        "config": dict(config),
        "state": dict(state),
        "feedback": None if feedback is None else dict(feedback),
        "gradient_heat": gradient_heat,
    }
    contract = _native_temperature_control(request)
    _validate_contract(contract)
    if contract["probability_count"] != len(request["probabilities"]):
        raise RuntimeError(
            "native Z-space temperature control returned a misaligned probability_count"
        )
    return contract
