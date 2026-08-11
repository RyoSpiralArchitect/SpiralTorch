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
ZSPACE_PARAMETER_TRAJECTORY_CONTRACT_VERSION = (
    "spiraltorch.zspace_parameter_trajectory.v1"
)
ZSPACE_PARAMETER_TRAJECTORY_KIND = "spiraltorch.zspace_parameter_trajectory"
ZSPACE_PARAMETER_TRAJECTORY_SEMANTIC_OWNER = "st-core::runtime::zspace_optimizer"
ZSPACE_PARAMETER_TRAJECTORY_SEMANTIC_BACKEND = "rust"
ZSPACE_PARAMETER_TRAJECTORY_POLICY_CONTRACT_VERSION = (
    "spiraltorch.zspace_parameter_trajectory_policy.v1"
)
ZSPACE_PARAMETER_TRAJECTORY_POLICY_KIND = (
    "spiraltorch.zspace_parameter_trajectory_policy"
)
ZSPACE_PARAMETER_TRAJECTORY_POLICY_SEMANTIC_OWNER = (
    "st-core::runtime::zspace_optimizer"
)
ZSPACE_PARAMETER_TRAJECTORY_POLICY_SEMANTIC_BACKEND = "rust"
ZSPACE_PARAMETER_TRAJECTORY_POLICIES = ("dose_preserving_complement",)
ZSPACE_PARAMETER_TRAJECTORY_DOSE_PRESERVING_COMPLEMENT_RULE = (
    "s_i=1-a*(r_i-r_bar_w),r_bar_w=sum_i(w_i*r_i)/sum_i(w_i),"
    "a=max_safe_in_[0,1],sum_i(w_i*s_i)=sum_i(w_i)"
)
ZSPACE_OPTIMIZER_FEEDBACK_CONTRACT_VERSION = "spiraltorch.zspace_optimizer_feedback.v1"
ZSPACE_OPTIMIZER_FEEDBACK_KIND = "spiraltorch.zspace_optimizer_feedback"
ZSPACE_OPTIMIZER_FEEDBACK_SEMANTIC_OWNER = "st-core::runtime::zspace_optimizer_feedback"
ZSPACE_OPTIMIZER_FEEDBACK_SEMANTIC_BACKEND = "rust"
ZSPACE_OPTIMIZER_FEEDBACK_CONTROL_RULE = (
    "applied_scale=1+effective_gate*(proposed_scale-1)"
)

__all__ = [
    "ZSPACE_META_OBJECTIVE_FORMULA",
    "ZSPACE_META_OPTIMIZER_CONTRACT_VERSION",
    "ZSPACE_META_OPTIMIZER_KIND",
    "ZSPACE_META_OPTIMIZER_SEMANTIC_BACKEND",
    "ZSPACE_META_OPTIMIZER_SEMANTIC_OWNER",
    "ZSPACE_OPTIMIZER_FEEDBACK_CONTRACT_VERSION",
    "ZSPACE_OPTIMIZER_FEEDBACK_CONTROL_RULE",
    "ZSPACE_OPTIMIZER_FEEDBACK_KIND",
    "ZSPACE_OPTIMIZER_FEEDBACK_SEMANTIC_BACKEND",
    "ZSPACE_OPTIMIZER_FEEDBACK_SEMANTIC_OWNER",
    "ZSPACE_PARAMETER_CONTROL_CONTRACT_VERSION",
    "ZSPACE_PARAMETER_CONTROL_KIND",
    "ZSPACE_PARAMETER_CONTROL_MAX_LEARNING_RATE_SCALE",
    "ZSPACE_PARAMETER_CONTROL_MIN_LEARNING_RATE_SCALE",
    "ZSPACE_PARAMETER_CONTROL_SEMANTIC_BACKEND",
    "ZSPACE_PARAMETER_CONTROL_SEMANTIC_OWNER",
    "ZSPACE_PARAMETER_TRAJECTORY_CONTRACT_VERSION",
    "ZSPACE_PARAMETER_TRAJECTORY_KIND",
    "ZSPACE_PARAMETER_TRAJECTORY_DOSE_PRESERVING_COMPLEMENT_RULE",
    "ZSPACE_PARAMETER_TRAJECTORY_POLICIES",
    "ZSPACE_PARAMETER_TRAJECTORY_POLICY_CONTRACT_VERSION",
    "ZSPACE_PARAMETER_TRAJECTORY_POLICY_KIND",
    "ZSPACE_PARAMETER_TRAJECTORY_POLICY_SEMANTIC_BACKEND",
    "ZSPACE_PARAMETER_TRAJECTORY_POLICY_SEMANTIC_OWNER",
    "ZSPACE_PARAMETER_TRAJECTORY_SEMANTIC_BACKEND",
    "ZSPACE_PARAMETER_TRAJECTORY_SEMANTIC_OWNER",
    "validate_zspace_parameter_trajectory",
    "validate_zspace_parameter_trajectory_policy",
    "zspace_meta_optimizer_init",
    "zspace_meta_optimizer_restore",
    "zspace_meta_optimizer_step",
    "zspace_optimizer_feedback_control",
    "zspace_optimizer_feedback_init",
    "zspace_optimizer_feedback_observe",
    "zspace_optimizer_feedback_restore",
    "zspace_parameter_control",
    "zspace_parameter_trajectory",
    "zspace_parameter_trajectory_policy",
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


def _validate_parameter_trajectory(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("kind") != ZSPACE_PARAMETER_TRAJECTORY_KIND
        or contract.get("contract_version")
        != ZSPACE_PARAMETER_TRAJECTORY_CONTRACT_VERSION
        or contract.get("semantic_owner") != ZSPACE_PARAMETER_TRAJECTORY_SEMANTIC_OWNER
        or contract.get("semantic_backend")
        != ZSPACE_PARAMETER_TRAJECTORY_SEMANTIC_BACKEND
        or contract.get("trajectory_validated") is not True
    ):
        raise RuntimeError(
            "native Z-space core returned an untrusted parameter trajectory"
        )
    trajectory_id = contract.get("trajectory_id")
    if (
        not isinstance(trajectory_id, str)
        or not trajectory_id.startswith("sha256:")
        or len(trajectory_id) != 71
    ):
        raise RuntimeError(
            "native Z-space parameter trajectory returned invalid identity"
        )
    step_count = contract.get("step_count")
    group_count = contract.get("parameter_group_count")
    steps = contract.get("steps")
    request = contract.get("request")
    if (
        isinstance(step_count, bool)
        or not isinstance(step_count, int)
        or step_count <= 0
        or isinstance(group_count, bool)
        or not isinstance(group_count, int)
        or group_count <= 0
        or not isinstance(steps, list)
        or len(steps) != step_count
        or not isinstance(request, Mapping)
    ):
        raise RuntimeError(
            "native Z-space parameter trajectory returned invalid dimensions"
        )


def _validate_parameter_trajectory_policy(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("kind") != ZSPACE_PARAMETER_TRAJECTORY_POLICY_KIND
        or contract.get("contract_version")
        != ZSPACE_PARAMETER_TRAJECTORY_POLICY_CONTRACT_VERSION
        or contract.get("semantic_owner")
        != ZSPACE_PARAMETER_TRAJECTORY_POLICY_SEMANTIC_OWNER
        or contract.get("semantic_backend")
        != ZSPACE_PARAMETER_TRAJECTORY_POLICY_SEMANTIC_BACKEND
        or contract.get("policy_validated") is not True
    ):
        raise RuntimeError(
            "native Z-space core returned an untrusted parameter trajectory policy"
        )
    policy_id = contract.get("policy_id")
    source_id = contract.get("source_trajectory_id")
    if not all(
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        for value in (policy_id, source_id)
    ):
        raise RuntimeError(
            "native Z-space parameter trajectory policy returned invalid identity"
        )
    step_count = contract.get("step_count")
    steps = contract.get("steps")
    request = contract.get("request")
    if (
        isinstance(step_count, bool)
        or not isinstance(step_count, int)
        or step_count <= 0
        or not isinstance(steps, list)
        or len(steps) != step_count
        or not isinstance(request, Mapping)
        or contract.get("policy") not in ZSPACE_PARAMETER_TRAJECTORY_POLICIES
    ):
        raise RuntimeError(
            "native Z-space parameter trajectory policy returned invalid dimensions"
        )


def _validate_feedback_contract(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("kind") != ZSPACE_OPTIMIZER_FEEDBACK_KIND
        or contract.get("contract_version")
        != ZSPACE_OPTIMIZER_FEEDBACK_CONTRACT_VERSION
        or contract.get("semantic_owner") != ZSPACE_OPTIMIZER_FEEDBACK_SEMANTIC_OWNER
        or contract.get("semantic_backend")
        != ZSPACE_OPTIMIZER_FEEDBACK_SEMANTIC_BACKEND
    ):
        raise RuntimeError(
            "native Z-space core returned an untrusted optimizer feedback contract"
        )
    config = _mapping(contract, "config")
    state = contract.get("state", contract.get("state_after"))
    if not isinstance(state, Mapping):
        raise RuntimeError("native Z-space optimizer feedback returned invalid state")
    maximum_gate = config.get("maximum_gate")
    gate = state.get("gate")
    control_step = state.get("control_step")
    observation_count = state.get("observation_count")
    if (
        isinstance(maximum_gate, bool)
        or not isinstance(maximum_gate, (int, float))
        or not math.isfinite(float(maximum_gate))
        or not 0.0 < float(maximum_gate) <= 1.0
        or isinstance(gate, bool)
        or not isinstance(gate, (int, float))
        or not math.isfinite(float(gate))
        or not 0.0 <= float(gate) <= float(maximum_gate)
        or isinstance(control_step, bool)
        or not isinstance(control_step, int)
        or control_step < 0
        or isinstance(observation_count, bool)
        or not isinstance(observation_count, int)
        or observation_count < 0
    ):
        raise RuntimeError(
            "native Z-space optimizer feedback returned invalid bounded state"
        )
    if "state_after" not in contract:
        return
    if contract.get("transition_validated") is not True:
        raise RuntimeError(
            "native Z-space optimizer feedback returned an unvalidated transition"
        )
    if "applied_learning_rate_scale" not in contract:
        projection = contract.get("projection")
        if (
            not isinstance(projection, Mapping)
            or projection.get("semantic_backend") != "rust"
        ):
            raise RuntimeError(
                "native Z-space optimizer feedback returned an invalid projection"
            )
        return
    if contract.get("control_rule") != ZSPACE_OPTIMIZER_FEEDBACK_CONTROL_RULE:
        raise RuntimeError(
            "native Z-space optimizer feedback returned an unknown control rule"
        )
    numeric: dict[str, float] = {}
    for field in (
        "proposed_learning_rate_scale",
        "effective_feedback_gate",
        "applied_learning_rate_scale",
    ):
        value = contract.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise RuntimeError(
                f"native Z-space optimizer feedback returned invalid {field}"
            )
        numeric[field] = float(value)
    effective_gate = numeric["effective_feedback_gate"]
    if not 0.0 <= effective_gate <= float(maximum_gate):
        raise RuntimeError(
            "native Z-space optimizer feedback returned an unsafe effective gate"
        )
    expected = 1.0 + effective_gate * (numeric["proposed_learning_rate_scale"] - 1.0)
    if not math.isclose(
        numeric["applied_learning_rate_scale"],
        expected,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise RuntimeError(
            "native Z-space optimizer feedback violated its blend invariant"
        )


def _feedback_operation(
    name: str,
    payload: Mapping[str, object],
) -> dict[str, Any]:
    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    operation = getattr(native, name, None)
    if not callable(operation):
        raise RuntimeError(
            "Z-space optimizer feedback requires the compiled Rust semantic core; "
            f"rebuild or reinstall SpiralTorch with {name}"
        )
    contract = operation(dict(payload))
    if not isinstance(contract, Mapping):
        raise RuntimeError(f"native {name} returned a non-mapping payload")
    result = dict(contract)
    _validate_feedback_contract(result)
    return result


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


def zspace_optimizer_feedback_init(
    config: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Initialize a fail-closed feedback gate in the canonical Rust core."""

    return _feedback_operation(
        "_zspace_optimizer_feedback_init",
        {} if config is None else dict(config),
    )


def zspace_optimizer_feedback_restore(
    *,
    config: Mapping[str, object],
    state: Mapping[str, object],
) -> dict[str, Any]:
    """Validate a checkpointed feedback gate without changing its state."""

    return _feedback_operation(
        "_zspace_optimizer_feedback_restore",
        {"config": dict(config), "state": dict(state)},
    )


def zspace_optimizer_feedback_observe(
    *,
    config: Mapping[str, object],
    state: Mapping[str, object],
    observation: Mapping[str, object],
) -> dict[str, Any]:
    """Commit one completed trainer observation through the Rust loss guard."""

    return _feedback_operation(
        "_zspace_optimizer_feedback_observe",
        {
            "config": dict(config),
            "state": dict(state),
            "observation": dict(observation),
        },
    )


def zspace_optimizer_feedback_control(
    *,
    config: Mapping[str, object],
    state: Mapping[str, object],
    target_step: int,
    proposed_learning_rate_scale: float,
) -> dict[str, Any]:
    """Blend one proposed scale toward identity under Rust-owned feedback."""

    return _feedback_operation(
        "_zspace_optimizer_feedback_control",
        {
            "config": dict(config),
            "state": dict(state),
            "target_step": target_step,
            "proposed_learning_rate_scale": proposed_learning_rate_scale,
        },
    )


def _parameter_trajectory_operation(
    name: str,
    payload: Mapping[str, object],
) -> dict[str, Any]:
    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    operation = getattr(native, name, None)
    if not callable(operation):
        raise RuntimeError(
            "Z-space parameter trajectories require the compiled Rust semantic "
            f"core; rebuild or reinstall SpiralTorch with {name}"
        )
    contract = operation(dict(payload))
    if not isinstance(contract, Mapping):
        raise RuntimeError(f"native {name} returned a non-mapping payload")
    result = dict(contract)
    _validate_parameter_trajectory(result)
    return result


def zspace_parameter_trajectory(
    *,
    raw_learning_rate_scales: list[float] | tuple[float, ...],
    nominal_learning_rates: list[list[float]] | tuple[tuple[float, ...], ...],
) -> dict[str, Any]:
    """Factor schedule shape and integrated LR dose in the canonical Rust core."""

    return _parameter_trajectory_operation(
        "_zspace_parameter_trajectory",
        {
            "raw_learning_rate_scales": list(raw_learning_rate_scales),
            "nominal_learning_rates": [list(rates) for rates in nominal_learning_rates],
        },
    )


def validate_zspace_parameter_trajectory(
    report: Mapping[str, object],
) -> dict[str, Any]:
    """Recompute a serialized trajectory in Rust and reject changed fields."""

    return _parameter_trajectory_operation(
        "_zspace_parameter_trajectory_validate",
        report,
    )


def _parameter_trajectory_policy_operation(
    name: str,
    payload: Mapping[str, object],
) -> dict[str, Any]:
    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    operation = getattr(native, name, None)
    if not callable(operation):
        raise RuntimeError(
            "Z-space parameter trajectory policies require the compiled Rust "
            f"semantic core; rebuild or reinstall SpiralTorch with {name}"
        )
    contract = operation(dict(payload))
    if not isinstance(contract, Mapping):
        raise RuntimeError(f"native {name} returned a non-mapping payload")
    result = dict(contract)
    _validate_parameter_trajectory_policy(result)
    return result


def zspace_parameter_trajectory_policy(
    source_trajectory: Mapping[str, object],
    *,
    policy: str = "dose_preserving_complement",
) -> dict[str, Any]:
    """Apply a Rust-owned, dose-preserving policy to a validated trajectory."""

    return _parameter_trajectory_policy_operation(
        "_zspace_parameter_trajectory_policy",
        {
            "source_trajectory": dict(source_trajectory),
            "policy": str(policy).strip().lower(),
        },
    )


def validate_zspace_parameter_trajectory_policy(
    report: Mapping[str, object],
) -> dict[str, Any]:
    """Recompute a serialized trajectory policy in Rust and reject changes."""

    return _parameter_trajectory_policy_operation(
        "_zspace_parameter_trajectory_policy_validate",
        report,
    )
