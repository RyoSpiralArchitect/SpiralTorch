"""Rust-owned projection of trainer observations into bounded telemetry."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from typing import Any

TRAINING_TELEMETRY_PROJECTION_CONTRACT_VERSION = (
    "spiraltorch.training_telemetry_projection.v1"
)
TRAINING_TELEMETRY_PROJECTION_KIND = "spiraltorch.training_telemetry_projection"
TRAINING_TELEMETRY_PROJECTION_SEMANTIC_OWNER = (
    "st-core::telemetry::training_projection"
)
TRAINING_TELEMETRY_PROJECTION_SEMANTIC_BACKEND = "rust"
TRAINING_TELEMETRY_PROJECTION_SIGNAL_SOURCE = "trainer_log_proxy"
TRAINING_TELEMETRY_PROJECTION_SIGNAL_SEMANTICS = "surrogate"

__all__ = [
    "TRAINING_TELEMETRY_PROJECTION_CONTRACT_VERSION",
    "TRAINING_TELEMETRY_PROJECTION_KIND",
    "TRAINING_TELEMETRY_PROJECTION_SEMANTIC_BACKEND",
    "TRAINING_TELEMETRY_PROJECTION_SEMANTIC_OWNER",
    "TRAINING_TELEMETRY_PROJECTION_SIGNAL_SEMANTICS",
    "TRAINING_TELEMETRY_PROJECTION_SIGNAL_SOURCE",
    "training_telemetry_projection",
]


def _native_projection(request: Mapping[str, object]) -> dict[str, Any]:
    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    project = getattr(native, "_training_telemetry_projection", None)
    if not callable(project):
        raise RuntimeError(
            "training telemetry projection requires the compiled Rust semantic core; "
            "rebuild or reinstall SpiralTorch with _training_telemetry_projection"
        )
    contract = project(dict(request))
    if not isinstance(contract, Mapping):
        raise RuntimeError("native training telemetry projection returned a non-mapping payload")
    return dict(contract)


def _validate_contract(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("kind") != TRAINING_TELEMETRY_PROJECTION_KIND
        or contract.get("contract_version")
        != TRAINING_TELEMETRY_PROJECTION_CONTRACT_VERSION
        or contract.get("semantic_owner")
        != TRAINING_TELEMETRY_PROJECTION_SEMANTIC_OWNER
        or contract.get("semantic_backend")
        != TRAINING_TELEMETRY_PROJECTION_SEMANTIC_BACKEND
        or contract.get("signal_source")
        != TRAINING_TELEMETRY_PROJECTION_SIGNAL_SOURCE
        or contract.get("signal_semantics")
        != TRAINING_TELEMETRY_PROJECTION_SIGNAL_SEMANTICS
    ):
        raise RuntimeError(
            "native training telemetry projection returned an untrusted contract"
        )
    for field in ("desire", "psi", "telemetry"):
        if not isinstance(contract.get(field), Mapping):
            raise RuntimeError(
                f"native training telemetry projection returned invalid {field}"
            )
    for key, value in contract["telemetry"].items():
        if (
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise RuntimeError(
                "native training telemetry projection returned non-finite telemetry"
            )


def training_telemetry_projection(
    *,
    step: float | int | None = None,
    max_steps: float | int | None = None,
    epoch: float | int | None = None,
    loss: float | int | None = None,
    previous_loss: float | int | None = None,
    grad_norm: float | int | None = None,
    learning_rate: float | int | None = None,
    desire_gain: float = 1.0,
    psi_gain: float = 1.0,
    learning_rate_scale: float = 10_000.0,
) -> dict[str, Any]:
    """Project trainer-log scalars through the canonical Rust surrogate contract."""

    observation = {
        key: value
        for key, value in {
            "step": step,
            "max_steps": max_steps,
            "epoch": epoch,
            "loss": loss,
            "previous_loss": previous_loss,
            "grad_norm": grad_norm,
            "learning_rate": learning_rate,
        }.items()
        if value is not None
    }
    contract = _native_projection(
        {
            "observation": observation,
            "config": {
                "desire_gain": desire_gain,
                "psi_gain": psi_gain,
                "learning_rate_scale": learning_rate_scale,
            },
        }
    )
    _validate_contract(contract)
    return contract
