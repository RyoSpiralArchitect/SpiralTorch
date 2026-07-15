"""Thin Python client for Rust-owned variational free-energy semantics."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from typing import Any

FREE_ENERGY_CONTRACT_VERSION = "spiraltorch.variational_free_energy.v1"
FREE_ENERGY_KIND = "spiraltorch.variational_free_energy"
FREE_ENERGY_SEMANTIC_OWNER = "st-core::heur::free_energy"
FREE_ENERGY_SEMANTIC_BACKEND = "rust"
FREE_ENERGY_FORMULA = (
    "F(q)=E_observed+(E_q[V]-E_prior[V])+temperature*KL(q||prior)"
)
FREE_ENERGY_ACCEPTANCE_RULE = (
    "P(accept)=1/(1+exp(F_candidate-F_neutral)),F_neutral=0"
)

__all__ = [
    "FREE_ENERGY_CONTRACT_VERSION",
    "FREE_ENERGY_ACCEPTANCE_RULE",
    "FREE_ENERGY_FORMULA",
    "FREE_ENERGY_KIND",
    "FREE_ENERGY_SEMANTIC_BACKEND",
    "FREE_ENERGY_SEMANTIC_OWNER",
    "zspace_free_energy",
]


def _native_free_energy(request: Mapping[str, object]) -> dict[str, Any]:
    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    evaluate = getattr(native, "_zspace_free_energy", None)
    if not callable(evaluate):
        raise RuntimeError(
            "variational free energy requires the compiled Rust semantic core; "
            "rebuild or reinstall SpiralTorch with _zspace_free_energy"
        )
    contract = evaluate(dict(request))
    if not isinstance(contract, Mapping):
        raise RuntimeError("native variational free energy returned a non-mapping payload")
    return dict(contract)


def _finite_number(mapping: Mapping[str, Any], field: str) -> float:
    value = mapping.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise RuntimeError(f"native variational free energy returned invalid {field}")
    return float(value)


def _mapping(contract: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = contract.get(field)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"native variational free energy returned invalid {field}")
    return value


def _validate_contract(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("kind") != FREE_ENERGY_KIND
        or contract.get("contract_version") != FREE_ENERGY_CONTRACT_VERSION
        or contract.get("semantic_owner") != FREE_ENERGY_SEMANTIC_OWNER
        or contract.get("semantic_backend") != FREE_ENERGY_SEMANTIC_BACKEND
        or contract.get("formula") != FREE_ENERGY_FORMULA
        or contract.get("acceptance_rule") != FREE_ENERGY_ACCEPTANCE_RULE
    ):
        raise RuntimeError(
            "native variational free energy returned an untrusted contract"
        )

    for field in ("config", "observation", "normalized", "distribution", "components"):
        _mapping(contract, field)
    free_energy = _finite_number(contract, "free_energy")
    utility = _finite_number(contract, "utility")
    acceptance_probability = _finite_number(contract, "acceptance_probability")
    component_residual = _finite_number(contract, "component_sum_residual")
    if not 0.0 <= acceptance_probability <= 1.0:
        raise RuntimeError(
            "native variational free energy returned invalid acceptance probability"
        )
    if not math.isclose(utility, -free_energy, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError("native variational free energy violated utility identity")
    if component_residual > 1e-10:
        raise RuntimeError("native variational free energy violated component identity")

    distribution = _mapping(contract, "distribution")
    probabilities = [
        _finite_number(distribution, "above"),
        _finite_number(distribution, "here"),
        _finite_number(distribution, "beneath"),
    ]
    if any(value < 0.0 or value > 1.0 for value in probabilities) or not math.isclose(
        sum(probabilities), 1.0, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise RuntimeError(
            "native variational free energy returned invalid band distribution"
        )
    for field in (
        "entropy",
        "normalized_entropy",
        "cross_entropy",
        "kl_divergence",
        "variational_identity_residual",
    ):
        _finite_number(distribution, field)
    if _finite_number(distribution, "variational_identity_residual") > 1e-10:
        raise RuntimeError("native variational free energy violated KL identity")


def zspace_free_energy(
    *,
    reference_loss: float = 0.0,
    candidate_loss: float = 0.0,
    step_time_ms: float = 0.0,
    memory_mb: float = 0.0,
    retry_rate: float = 0.0,
    observation_entropy: float = 0.0,
    external_penalty: float = 0.0,
    band: Mapping[str, float] | None = None,
    config: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Evaluate one free-energy request in the canonical Rust semantic core."""

    observation: dict[str, object] = {
        "reference_loss": reference_loss,
        "candidate_loss": candidate_loss,
        "step_time_ms": step_time_ms,
        "memory_mb": memory_mb,
        "retry_rate": retry_rate,
        "observation_entropy": observation_entropy,
        "external_penalty": external_penalty,
    }
    if band is not None:
        observation["band"] = dict(band)
    request: dict[str, object] = {
        "observation": observation,
        "config": {} if config is None else dict(config),
    }
    contract = _native_free_energy(request)
    _validate_contract(contract)
    return contract
