"""Thin Python client for Rust-owned Z-space meta-optimizer semantics."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from typing import Any

ZSPACE_META_OPTIMIZER_CONTRACT_VERSION = "spiraltorch.zspace_meta_optimizer.v1"
ZSPACE_META_OPTIMIZER_KIND = "spiraltorch.zspace_meta_optimizer"
ZSPACE_META_OPTIMIZER_SEMANTIC_OWNER = "st-core::runtime::zspace_optimizer"
ZSPACE_META_OPTIMIZER_SEMANTIC_BACKEND = "rust"
ZSPACE_META_OBJECTIVE_FORMULA = (
    "J_obs=sum_i(lambda_i*tanh(metric_i))+lambda_topos*tanh(topos_pressure)"
    "+lambda_frac_eff*R_alpha(z)"
)

__all__ = [
    "ZSPACE_META_OBJECTIVE_FORMULA",
    "ZSPACE_META_OPTIMIZER_CONTRACT_VERSION",
    "ZSPACE_META_OPTIMIZER_KIND",
    "ZSPACE_META_OPTIMIZER_SEMANTIC_BACKEND",
    "ZSPACE_META_OPTIMIZER_SEMANTIC_OWNER",
    "zspace_meta_optimizer_init",
    "zspace_meta_optimizer_restore",
    "zspace_meta_optimizer_step",
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
