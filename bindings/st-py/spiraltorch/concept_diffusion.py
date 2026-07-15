"""Thin Python client for Rust-owned Z-space concept diffusion."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence
from typing import Any

ZSPACE_CONCEPT_DIFFUSION_CONTRACT_VERSION = "spiraltorch.zspace_concept_diffusion.v1"
ZSPACE_CONCEPT_DIFFUSION_KIND = "spiraltorch.zspace_concept_diffusion"
ZSPACE_CONCEPT_DIFFUSION_SEMANTIC_OWNER = "st-core::inference::concept_diffusion"
ZSPACE_CONCEPT_DIFFUSION_SEMANTIC_BACKEND = "rust"
ZSPACE_CONCEPT_DIFFUSION_BACKEND = "spiraltorch_concept_diffusion_core"
ZSPACE_CONCEPT_DIFFUSION_EXECUTION_BACKEND = "f64_cpu"

__all__ = [
    "ZSPACE_CONCEPT_DIFFUSION_BACKEND",
    "ZSPACE_CONCEPT_DIFFUSION_CONTRACT_VERSION",
    "ZSPACE_CONCEPT_DIFFUSION_EXECUTION_BACKEND",
    "ZSPACE_CONCEPT_DIFFUSION_KIND",
    "ZSPACE_CONCEPT_DIFFUSION_SEMANTIC_BACKEND",
    "ZSPACE_CONCEPT_DIFFUSION_SEMANTIC_OWNER",
    "zspace_concept_diffusion",
]


def _native_concept_diffusion(request: Mapping[str, object]) -> dict[str, Any]:
    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    apply_diffusion = getattr(native, "_zspace_concept_diffusion", None)
    if not callable(apply_diffusion):
        raise RuntimeError(
            "Z-space concept diffusion requires the compiled Rust semantic core; "
            "rebuild or reinstall SpiralTorch with _zspace_concept_diffusion"
        )
    contract = apply_diffusion(dict(request))
    if not isinstance(contract, Mapping):
        raise RuntimeError(
            "native Z-space concept diffusion returned a non-mapping payload"
        )
    return dict(contract)


def _require_numeric_list(
    contract: Mapping[str, Any], field: str, expected: int
) -> None:
    values = contract.get(field)
    if not isinstance(values, list) or len(values) != expected:
        raise RuntimeError(f"native Z-space concept diffusion returned invalid {field}")
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise RuntimeError(
                f"native Z-space concept diffusion returned non-finite {field}"
            )


def _require_finite_number(contract: Mapping[str, Any], field: str) -> None:
    value = contract.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise RuntimeError(f"native Z-space concept diffusion returned invalid {field}")


def _validate_contract(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("kind") != ZSPACE_CONCEPT_DIFFUSION_KIND
        or contract.get("contract_version") != ZSPACE_CONCEPT_DIFFUSION_CONTRACT_VERSION
        or contract.get("semantic_owner") != ZSPACE_CONCEPT_DIFFUSION_SEMANTIC_OWNER
        or contract.get("semantic_backend") != ZSPACE_CONCEPT_DIFFUSION_SEMANTIC_BACKEND
        or contract.get("backend") != ZSPACE_CONCEPT_DIFFUSION_BACKEND
        or contract.get("execution_backend")
        != ZSPACE_CONCEPT_DIFFUSION_EXECUTION_BACKEND
    ):
        raise RuntimeError(
            "native Z-space concept diffusion returned an untrusted contract"
        )
    tags = contract.get("tags")
    if (
        not isinstance(tags, list)
        or not tags
        or any(not isinstance(tag, str) or not tag for tag in tags)
    ):
        raise RuntimeError("native Z-space concept diffusion returned invalid tags")
    count = len(tags)
    for field in (
        "previous_state",
        "state_after_observation",
        "state_after_bias",
        "next_state",
    ):
        _require_numeric_list(contract, field, count)
    for field in ("config", "effects"):
        if not isinstance(contract.get(field), Mapping):
            raise RuntimeError(
                f"native Z-space concept diffusion returned invalid {field}"
            )
    for field in (
        "input_probability_sum",
        "output_probability_sum",
        "probability_sum_tolerance",
    ):
        _require_finite_number(contract, field)
    tolerance = float(contract["probability_sum_tolerance"])
    if tolerance < 0.0:
        raise RuntimeError(
            "native Z-space concept diffusion returned invalid probability_sum_tolerance"
        )
    for field in (
        "previous_state",
        "state_after_observation",
        "state_after_bias",
        "next_state",
    ):
        values = contract[field]
        if any(float(value) < 0.0 for value in values):
            raise RuntimeError(
                f"native Z-space concept diffusion returned negative {field}"
            )
        if abs(math.fsum(float(value) for value in values) - 1.0) > tolerance:
            raise RuntimeError(
                f"native Z-space concept diffusion returned invalid mass for {field}"
            )


def zspace_concept_diffusion(
    tags: Sequence[str],
    state: Sequence[float | int],
    affinity: Sequence[Sequence[float | int]],
    *,
    diffusion_tensor: Sequence[Sequence[float | int]] | None = None,
    z_bias: Sequence[float | int] = (),
    observation: Mapping[str, object] | None = None,
    config: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Advance the canonical Rust graph heat flow by one atomic step."""

    request: dict[str, object] = {
        "tags": list(tags),
        "state": list(state),
        "affinity": [list(row) for row in affinity],
        "diffusion_tensor": (
            None
            if diffusion_tensor is None
            else [list(row) for row in diffusion_tensor]
        ),
        "z_bias": list(z_bias),
        "observation": None if observation is None else dict(observation),
        "config": {} if config is None else dict(config),
    }
    contract = _native_concept_diffusion(request)
    _validate_contract(contract)
    if contract["tags"] != request["tags"]:
        raise RuntimeError("native Z-space concept diffusion returned misaligned tags")
    return contract
