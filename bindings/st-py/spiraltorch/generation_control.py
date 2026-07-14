"""Thin Python client for Rust-owned Z-space generation control."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence
from typing import Any

ZSPACE_GENERATION_CONTROL_CONTRACT_VERSION = (
    "spiraltorch.zspace_generation_control.v1"
)
ZSPACE_GENERATION_CONTROL_KIND = "spiraltorch.zspace_generation_control"
ZSPACE_GENERATION_CONTROL_SEMANTIC_OWNER = (
    "st-core::inference::generation_control"
)
ZSPACE_GENERATION_CONTROL_SEMANTIC_BACKEND = "rust"
ZSPACE_GENERATION_CONTROL_BACKEND = "spiraltorch_generation_control_core"

__all__ = [
    "ZSPACE_GENERATION_CONTROL_BACKEND",
    "ZSPACE_GENERATION_CONTROL_CONTRACT_VERSION",
    "ZSPACE_GENERATION_CONTROL_KIND",
    "ZSPACE_GENERATION_CONTROL_SEMANTIC_BACKEND",
    "ZSPACE_GENERATION_CONTROL_SEMANTIC_OWNER",
    "zspace_generation_control",
]


def _native_generation_control(request: Mapping[str, object]) -> dict[str, Any]:
    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    apply_control = getattr(native, "_zspace_generation_control", None)
    if not callable(apply_control):
        raise RuntimeError(
            "Z-space generation control requires the compiled Rust semantic core; "
            "rebuild or reinstall SpiralTorch with _zspace_generation_control"
        )
    contract = apply_control(dict(request))
    if not isinstance(contract, Mapping):
        raise RuntimeError(
            "native Z-space generation control returned a non-mapping payload"
        )
    return dict(contract)


def _validate_numeric_sequence(contract: Mapping[str, Any], field: str) -> None:
    values = contract.get(field)
    if not isinstance(values, list):
        raise RuntimeError(
            f"native Z-space generation control returned invalid {field}"
        )
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise RuntimeError(
                f"native Z-space generation control returned non-finite {field}"
            )


def _validate_contract(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("kind") != ZSPACE_GENERATION_CONTROL_KIND
        or contract.get("contract_version")
        != ZSPACE_GENERATION_CONTROL_CONTRACT_VERSION
        or contract.get("semantic_owner")
        != ZSPACE_GENERATION_CONTROL_SEMANTIC_OWNER
        or contract.get("semantic_backend")
        != ZSPACE_GENERATION_CONTROL_SEMANTIC_BACKEND
        or contract.get("backend") != ZSPACE_GENERATION_CONTROL_BACKEND
    ):
        raise RuntimeError(
            "native Z-space generation control returned an untrusted contract"
        )
    if not isinstance(contract.get("config"), Mapping):
        raise RuntimeError(
            "native Z-space generation control returned invalid resolved config"
        )
    for field in (
        "adjusted_logits",
        "repression_penalties",
        "ngram_repression_penalties",
        "probabilities",
        "log_probabilities",
    ):
        _validate_numeric_sequence(contract, field)
    count = contract.get("candidate_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise RuntimeError(
            "native Z-space generation control returned invalid candidate_count"
        )
    for field in (
        "adjusted_logits",
        "repression_penalties",
        "ngram_repression_penalties",
        "probabilities",
        "log_probabilities",
    ):
        if len(contract[field]) != count:
            raise RuntimeError(
                f"native Z-space generation control returned misaligned {field}"
            )


def zspace_generation_control(
    logits: Sequence[float | int],
    token_ids: Sequence[int],
    recent_tokens: Sequence[int] = (),
    *,
    curvature: float = -0.04,
    temperature: float = 1.0,
    entropy_target: float | None = None,
    entropy_tolerance: float = 1.0e-4,
    entropy_gain: float = 0.5,
    min_temperature: float | None = None,
    max_temperature: float | None = None,
    repression_window: int = 32,
    repression_strength: float = 1.0,
    last_token_repression: float = 0.5,
    ngram_size: int = 0,
    ngram_window: int = 0,
    ngram_repression_strength: float = 0.0,
    ngram_decay: float = 1.0,
) -> dict[str, Any]:
    """Apply the canonical Rust repression and adaptive-softmax contract."""

    config: dict[str, object] = {
        "curvature": curvature,
        "temperature": temperature,
        "entropy_target": entropy_target,
        "entropy_tolerance": entropy_tolerance,
        "entropy_gain": entropy_gain,
        "repression_window": repression_window,
        "repression_strength": repression_strength,
        "last_token_repression": last_token_repression,
        "ngram_size": ngram_size,
        "ngram_window": ngram_window,
        "ngram_repression_strength": ngram_repression_strength,
        "ngram_decay": ngram_decay,
    }
    if min_temperature is not None:
        config["min_temperature"] = min_temperature
    if max_temperature is not None:
        config["max_temperature"] = max_temperature
    contract = _native_generation_control(
        {
            "logits": list(logits),
            "token_ids": list(token_ids),
            "recent_tokens": list(recent_tokens),
            "config": config,
        }
    )
    _validate_contract(contract)
    return contract
