"""Thin Python client for Rust-owned imaginary-time Schrodinger evolution."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence
from typing import Any

ZSPACE_IMAGINARY_TIME_SCHRODINGER_CONTRACT_VERSION = (
    "spiraltorch.zspace_imaginary_time_schrodinger.v1"
)
ZSPACE_IMAGINARY_TIME_SCHRODINGER_KIND = "spiraltorch.zspace_imaginary_time_schrodinger"
ZSPACE_IMAGINARY_TIME_SCHRODINGER_SEMANTIC_OWNER = (
    "st-core::inference::imaginary_time_schrodinger"
)
ZSPACE_IMAGINARY_TIME_SCHRODINGER_SEMANTIC_BACKEND = "rust"
ZSPACE_IMAGINARY_TIME_SCHRODINGER_BACKEND = (
    "spiraltorch_imaginary_time_schrodinger_core"
)
ZSPACE_IMAGINARY_TIME_SCHRODINGER_EXECUTION_BACKEND = "f64_cpu"
ZSPACE_IMAGINARY_TIME_SCHRODINGER_ROUTE_BLOCKER = "f64_sparse_graph_state"

__all__ = [
    "ZSPACE_IMAGINARY_TIME_SCHRODINGER_BACKEND",
    "ZSPACE_IMAGINARY_TIME_SCHRODINGER_CONTRACT_VERSION",
    "ZSPACE_IMAGINARY_TIME_SCHRODINGER_EXECUTION_BACKEND",
    "ZSPACE_IMAGINARY_TIME_SCHRODINGER_KIND",
    "ZSPACE_IMAGINARY_TIME_SCHRODINGER_ROUTE_BLOCKER",
    "ZSPACE_IMAGINARY_TIME_SCHRODINGER_SEMANTIC_BACKEND",
    "ZSPACE_IMAGINARY_TIME_SCHRODINGER_SEMANTIC_OWNER",
    "zspace_imaginary_time_schrodinger",
]


def _native_schrodinger(request: Mapping[str, object]) -> dict[str, Any]:
    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    evolve = getattr(native, "_zspace_imaginary_time_schrodinger", None)
    if not callable(evolve):
        raise RuntimeError(
            "Z-space imaginary-time Schrodinger evolution requires the compiled "
            "Rust semantic core; rebuild or reinstall SpiralTorch with "
            "_zspace_imaginary_time_schrodinger"
        )
    contract = evolve(dict(request))
    if not isinstance(contract, Mapping):
        raise RuntimeError(
            "native Z-space imaginary-time Schrodinger returned a non-mapping payload"
        )
    return dict(contract)


def _numeric_list(contract: Mapping[str, Any], field: str, expected: int) -> list[Any]:
    values = contract.get(field)
    if not isinstance(values, list) or len(values) != expected:
        raise RuntimeError(
            f"native Z-space imaginary-time Schrodinger returned invalid {field}"
        )
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise RuntimeError(
                f"native Z-space imaginary-time Schrodinger returned non-finite {field}"
            )
    return values


def _finite_number(mapping: Mapping[str, Any], field: str) -> float:
    value = mapping.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise RuntimeError(
            f"native Z-space imaginary-time Schrodinger returned invalid {field}"
        )
    return float(value)


def _validate_contract(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("kind") != ZSPACE_IMAGINARY_TIME_SCHRODINGER_KIND
        or contract.get("contract_version")
        != ZSPACE_IMAGINARY_TIME_SCHRODINGER_CONTRACT_VERSION
        or contract.get("semantic_owner")
        != ZSPACE_IMAGINARY_TIME_SCHRODINGER_SEMANTIC_OWNER
        or contract.get("semantic_backend")
        != ZSPACE_IMAGINARY_TIME_SCHRODINGER_SEMANTIC_BACKEND
        or contract.get("backend") != ZSPACE_IMAGINARY_TIME_SCHRODINGER_BACKEND
        or contract.get("execution_backend")
        != ZSPACE_IMAGINARY_TIME_SCHRODINGER_EXECUTION_BACKEND
        or contract.get("route_blocker")
        != ZSPACE_IMAGINARY_TIME_SCHRODINGER_ROUTE_BLOCKER
    ):
        raise RuntimeError(
            "native Z-space imaginary-time Schrodinger returned an untrusted contract"
        )
    tags = contract.get("tags")
    if (
        not isinstance(tags, list)
        or not tags
        or any(not isinstance(tag, str) or not tag for tag in tags)
    ):
        raise RuntimeError(
            "native Z-space imaginary-time Schrodinger returned invalid tags"
        )
    count = len(tags)
    for field in ("config", "effects"):
        if not isinstance(contract.get(field), Mapping):
            raise RuntimeError(
                f"native Z-space imaginary-time Schrodinger returned invalid {field}"
            )
    effects = contract["effects"]
    fields = {
        field: _numeric_list(contract, field, count)
        for field in (
            "potential",
            "shifted_potential",
            "initial_amplitude",
            "final_amplitude",
            "probability",
            "log_amplitude_boost",
        )
    }
    if any(float(value) < 0.0 for value in fields["shifted_potential"]):
        raise RuntimeError(
            "native Z-space imaginary-time Schrodinger returned negative shifted_potential"
        )
    for field in ("initial_amplitude", "final_amplitude", "probability"):
        if any(float(value) < 0.0 for value in fields[field]):
            raise RuntimeError(
                f"native Z-space imaginary-time Schrodinger returned negative {field}"
            )
    probability_sum = _finite_number(contract, "probability_sum")
    probability_sum_tolerance = _finite_number(contract, "probability_sum_tolerance")
    if probability_sum_tolerance <= 0.0:
        raise RuntimeError(
            "native Z-space imaginary-time Schrodinger returned invalid "
            "probability_sum_tolerance"
        )
    if (
        abs(probability_sum - 1.0) > probability_sum_tolerance
        or abs(math.fsum(float(value) for value in fields["probability"]) - 1.0)
        > probability_sum_tolerance
    ):
        raise RuntimeError(
            "native Z-space imaginary-time Schrodinger returned invalid probability mass"
        )
    maximum_boost = max(float(value) for value in fields["log_amplitude_boost"])
    if maximum_boost != 0.0:
        raise RuntimeError(
            "native Z-space imaginary-time Schrodinger returned unnormalized log boost"
        )
    initial_energy = _finite_number(effects, "initial_rayleigh_energy")
    final_energy = _finite_number(effects, "final_rayleigh_energy")
    energy_tolerance = _finite_number(effects, "energy_tolerance")
    if energy_tolerance <= 0.0:
        raise RuntimeError(
            "native Z-space imaginary-time Schrodinger returned invalid energy_tolerance"
        )
    if final_energy > initial_energy + energy_tolerance:
        raise RuntimeError(
            "native Z-space imaginary-time Schrodinger violated energy monotonicity"
        )
    l2_norm_tolerance = _finite_number(effects, "l2_norm_tolerance")
    if l2_norm_tolerance <= 0.0:
        raise RuntimeError(
            "native Z-space imaginary-time Schrodinger returned invalid l2_norm_tolerance"
        )
    for field in ("initial_l2_norm", "final_l2_norm"):
        if abs(_finite_number(effects, field) - 1.0) > l2_norm_tolerance:
            raise RuntimeError(
                "native Z-space imaginary-time Schrodinger violated "
                f"{field} normalization"
            )
    edges = contract.get("edges")
    if not isinstance(edges, list):
        raise RuntimeError(
            "native Z-space imaginary-time Schrodinger returned invalid edges"
        )
    seen: set[tuple[int, int]] = set()
    for edge in edges:
        if not isinstance(edge, Mapping):
            raise RuntimeError(
                "native Z-space imaginary-time Schrodinger returned invalid edge"
            )
        left, right, weight = edge.get("left"), edge.get("right"), edge.get("weight")
        if (
            isinstance(left, bool)
            or not isinstance(left, int)
            or isinstance(right, bool)
            or not isinstance(right, int)
            or left < 0
            or left >= right
            or right >= count
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) <= 0.0
            or (left, right) in seen
        ):
            raise RuntimeError(
                "native Z-space imaginary-time Schrodinger returned invalid edge"
            )
        seen.add((left, right))


def zspace_imaginary_time_schrodinger(
    tags: Sequence[str],
    potential: Sequence[float | int],
    edges: Sequence[Mapping[str, object]],
    *,
    initial_amplitude: Sequence[float | int] = (),
    config: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Evolve a positive amplitude with the canonical Rust graph Hamiltonian."""

    request: dict[str, object] = {
        "tags": list(tags),
        "potential": list(potential),
        "edges": [dict(edge) for edge in edges],
        "initial_amplitude": list(initial_amplitude),
        "config": {} if config is None else dict(config),
    }
    contract = _native_schrodinger(request)
    _validate_contract(contract)
    if contract["tags"] != request["tags"]:
        raise RuntimeError(
            "native Z-space imaginary-time Schrodinger returned misaligned tags"
        )
    return contract
