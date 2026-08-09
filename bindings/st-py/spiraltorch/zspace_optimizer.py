"""Thin Python client for Rust-owned Z-space meta-optimizer semantics."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from typing import Any

ZSPACE_META_OPTIMIZER_CONTRACT_VERSION = "spiraltorch.zspace_meta_optimizer.v2"
ZSPACE_META_OPTIMIZER_KIND = "spiraltorch.zspace_meta_optimizer"
ZSPACE_META_OPTIMIZER_SEMANTIC_OWNER = "st-core::runtime::zspace_optimizer"
ZSPACE_META_OPTIMIZER_SEMANTIC_BACKEND = "rust"
ZSPACE_META_OBJECTIVE_FORMULA = (
    "J_obs=sum_i(lambda_i*tanh(metric_i))+lambda_topos*tanh(topos_pressure)"
    "+lambda_frac_eff*R_alpha(z)"
)
ZSPACE_PARAMETER_CONTROL_CONTRACT_VERSION = "spiraltorch.zspace_parameter_control.v2"
ZSPACE_PARAMETER_CONTROL_KIND = "spiraltorch.zspace_parameter_control"
ZSPACE_PARAMETER_CONTROL_SEMANTIC_OWNER = "st-core::runtime::zspace_optimizer"
ZSPACE_PARAMETER_CONTROL_SEMANTIC_BACKEND = "rust"
ZSPACE_PARAMETER_CONTROL_MIN_LEARNING_RATE_SCALE = 0.1
ZSPACE_PARAMETER_CONTROL_MAX_LEARNING_RATE_SCALE = 1.25

__all__ = [
    "ZSPACE_META_OBJECTIVE_FORMULA",
    "ZSPACE_META_OPTIMIZER_CONTRACT_VERSION",
    "ZSPACE_META_OPTIMIZER_KIND",
    "ZSPACE_META_OPTIMIZER_SEMANTIC_BACKEND",
    "ZSPACE_META_OPTIMIZER_SEMANTIC_OWNER",
    "ZSPACE_PARAMETER_CONTROL_CONTRACT_VERSION",
    "ZSPACE_PARAMETER_CONTROL_KIND",
    "ZSPACE_PARAMETER_CONTROL_MAX_LEARNING_RATE_SCALE",
    "ZSPACE_PARAMETER_CONTROL_MIN_LEARNING_RATE_SCALE",
    "ZSPACE_PARAMETER_CONTROL_SEMANTIC_BACKEND",
    "ZSPACE_PARAMETER_CONTROL_SEMANTIC_OWNER",
    "zspace_meta_optimizer_init",
    "zspace_meta_optimizer_restore",
    "zspace_meta_optimizer_step",
    "zspace_parameter_control",
]


def _native_operation(name: str, payload: Mapping[str, object]) -> dict[str, Any]:
    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    operation = getattr(native, name, None)
    if not callable(operation):
        raise RuntimeError(
            "Z-space meta-optimization requires the compiled Rust semantic core; "
            f"rebuild or reinstall SpiralTorch with {name}"
        )
    contract = operation(dict(payload))
    if not isinstance(contract, Mapping):
        raise RuntimeError(f"native {name} returned a non-mapping payload")
    result = dict(contract)
    _validate_contract(result)
    return result


def _mapping(contract: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = contract.get(field)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"native Z-space meta-optimizer returned invalid {field}")
    return value


def _finite_vector(mapping: Mapping[str, Any], field: str) -> list[float]:
    value = mapping.get(field)
    if not isinstance(value, (list, tuple)):
        raise RuntimeError(f"native Z-space meta-optimizer returned invalid {field}")
    vector: list[float] = []
    for entry in value:
        if (
            isinstance(entry, bool)
            or not isinstance(entry, (int, float))
            or not math.isfinite(float(entry))
        ):
            raise RuntimeError(
                f"native Z-space meta-optimizer returned non-finite {field}"
            )
        vector.append(float(entry))
    return vector


def _validate_contract(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("kind") != ZSPACE_META_OPTIMIZER_KIND
        or contract.get("contract_version") != ZSPACE_META_OPTIMIZER_CONTRACT_VERSION
        or contract.get("semantic_owner") != ZSPACE_META_OPTIMIZER_SEMANTIC_OWNER
        or contract.get("semantic_backend") != ZSPACE_META_OPTIMIZER_SEMANTIC_BACKEND
    ):
        raise RuntimeError(
            "native Z-space meta-optimizer returned an untrusted contract"
        )

    config = _mapping(contract, "config")
    state = contract.get("state")
    if state is None:
        state = contract.get("state_after")
    if not isinstance(state, Mapping):
        raise RuntimeError("native Z-space meta-optimizer returned invalid state")
    dimension = config.get("dimension")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise RuntimeError("native Z-space meta-optimizer returned invalid dimension")
    for field in ("z", "first_moment", "second_moment"):
        if len(_finite_vector(state, field)) != dimension:
            raise RuntimeError(
                f"native Z-space meta-optimizer returned wrong-sized {field}"
            )
    step = state.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise RuntimeError("native Z-space meta-optimizer returned invalid step")

    if "state_after" in contract:
        if contract.get("transition_validated") is not True:
            raise RuntimeError(
                "native Z-space meta-optimizer returned an unvalidated transition"
            )
        if contract.get("objective_formula") != ZSPACE_META_OBJECTIVE_FORMULA:
            raise RuntimeError(
                "native Z-space meta-optimizer returned an unknown objective"
            )
        for field in (
            "objective",
            "fractional_regularizer",
            "topos_control",
            "gradient",
            "adam",
            "state_before",
        ):
            _mapping(contract, field)


def _validate_parameter_control(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("kind") != ZSPACE_PARAMETER_CONTROL_KIND
        or contract.get("contract_version") != ZSPACE_PARAMETER_CONTROL_CONTRACT_VERSION
        or contract.get("semantic_owner") != ZSPACE_PARAMETER_CONTROL_SEMANTIC_OWNER
        or contract.get("semantic_backend") != ZSPACE_PARAMETER_CONTROL_SEMANTIC_BACKEND
    ):
        raise RuntimeError(
            "native Z-space core returned an untrusted parameter control"
        )
    if (
        contract.get("source_contract_version")
        != ZSPACE_META_OPTIMIZER_CONTRACT_VERSION
        or contract.get("source_semantic_owner") != ZSPACE_META_OPTIMIZER_SEMANTIC_OWNER
    ):
        raise RuntimeError("native Z-space parameter control has an unknown source")

    source_step = contract.get("source_step")
    if (
        isinstance(source_step, bool)
        or not isinstance(source_step, int)
        or source_step <= 0
    ):
        raise RuntimeError(
            "native Z-space parameter control returned invalid source_step"
        )
    numeric: dict[str, float] = {}
    for field in (
        "absolute_learning_rate_scale",
        "source_learning_rate",
        "source_effective_learning_rate",
    ):
        value = contract.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise RuntimeError(
                f"native Z-space parameter control returned invalid {field}"
            )
        numeric[field] = float(value)
    scale = numeric["absolute_learning_rate_scale"]
    if not (
        ZSPACE_PARAMETER_CONTROL_MIN_LEARNING_RATE_SCALE
        <= scale
        <= ZSPACE_PARAMETER_CONTROL_MAX_LEARNING_RATE_SCALE
    ):
        raise RuntimeError("native Z-space parameter control returned an unsafe scale")
    expected = numeric["source_learning_rate"] * scale
    if not math.isclose(
        numeric["source_effective_learning_rate"],
        expected,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise RuntimeError(
            "native Z-space parameter control violated its rate invariant"
        )


def zspace_meta_optimizer_init(
    config: Mapping[str, object],
) -> dict[str, Any]:
    """Create a validated zero checkpoint in the canonical Rust core."""

    return _native_operation("_zspace_meta_optimizer_init", config)


def zspace_meta_optimizer_restore(
    *,
    config: Mapping[str, object],
    state: Mapping[str, object],
    strict: bool = True,
) -> dict[str, Any]:
    """Validate or dimension-coerce a checkpoint in the canonical Rust core."""

    return _native_operation(
        "_zspace_meta_optimizer_restore",
        {"config": dict(config), "state": dict(state), "strict": bool(strict)},
    )


def zspace_meta_optimizer_step(
    *,
    config: Mapping[str, object],
    state: Mapping[str, object],
    observation: Mapping[str, object],
) -> dict[str, Any]:
    """Evaluate one transactional state transition in the canonical Rust core."""

    return _native_operation(
        "_zspace_meta_optimizer_step",
        {
            "config": dict(config),
            "state": dict(state),
            "observation": dict(observation),
        },
    )


def zspace_parameter_control(
    report: Mapping[str, object],
) -> dict[str, Any]:
    """Validate a complete optimizer report and extract Rust-owned model control."""

    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    operation = getattr(native, "_zspace_parameter_control", None)
    if not callable(operation):
        raise RuntimeError(
            "Z-space parameter control requires the compiled Rust semantic core; "
            "rebuild or reinstall SpiralTorch with _zspace_parameter_control"
        )
    contract = operation(dict(report))
    if not isinstance(contract, Mapping):
        raise RuntimeError(
            "native _zspace_parameter_control returned a non-mapping payload"
        )
    result = dict(contract)
    _validate_parameter_control(result)
    return result
