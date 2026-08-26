"""Thin Python client for Rust-owned token periodicity semantics."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from typing import Any, Optional


def _native_constant(name: str, fallback: object) -> object:
    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    return getattr(native, name, fallback)


# Source-only imports retain discoverable metadata. Executable operations fail
# closed unless the compiled Rust semantic core is available.
ZSPACE_PERIODICITY_CONTRACT_VERSION = str(
    _native_constant(
        "ZSPACE_PERIODICITY_CONTRACT_VERSION", "spiraltorch.zspace_periodicity.v1"
    )
)
ZSPACE_PERIODICITY_KIND = str(
    _native_constant("ZSPACE_PERIODICITY_KIND", "spiraltorch.zspace_periodicity")
)
ZSPACE_PERIODICITY_SEMANTIC_OWNER = str(
    _native_constant(
        "ZSPACE_PERIODICITY_SEMANTIC_OWNER",
        "st-core::runtime::zspace_periodicity",
    )
)
ZSPACE_PERIODICITY_SEMANTIC_BACKEND = str(
    _native_constant("ZSPACE_PERIODICITY_SEMANTIC_BACKEND", "rust")
)
ZSPACE_PERIODICITY_RULE = str(
    _native_constant(
        "ZSPACE_PERIODICITY_RULE",
        "longest trailing suffix formed by complete repetitions with "
        "period<=maximum_period and repetitions>=minimum_repetitions; maximize "
        "repeated_token_count, then suffix token_count, then prefer the smaller period",
    )
)
ZSPACE_PERIODICITY_ANALYSIS_ID_RULE = str(
    _native_constant(
        "ZSPACE_PERIODICITY_ANALYSIS_ID_RULE",
        "sha256 of UTF-8 JSON for [contract_version, canonical request] with no "
        "insignificant whitespace",
    )
)
ZSPACE_PERIODICITY_EVIDENCE_BOUNDARY = str(
    _native_constant(
        "ZSPACE_PERIODICITY_EVIDENCE_BOUNDARY",
        "periodicity is a structural token-ID suffix observation; it does not "
        "measure semantic quality, prove a generation loop will continue, or "
        "establish model or training superiority",
    )
)
ZSPACE_PERIODIC_SUFFIX_MAX_PERIOD = int(
    _native_constant("ZSPACE_PERIODIC_SUFFIX_MAX_PERIOD", 16)
)
ZSPACE_PERIODIC_SUFFIX_MIN_REPETITIONS = int(
    _native_constant("ZSPACE_PERIODIC_SUFFIX_MIN_REPETITIONS", 3)
)
ZSPACE_PERIODICITY_MAX_TOKENS = int(
    _native_constant("ZSPACE_PERIODICITY_MAX_TOKENS", 1_000_000)
)
ZSPACE_PERIODICITY_MAX_PERIOD = int(
    _native_constant("ZSPACE_PERIODICITY_MAX_PERIOD", 4_096)
)
ZSPACE_PERIODICITY_MAX_MINIMUM_REPETITIONS = int(
    _native_constant("ZSPACE_PERIODICITY_MAX_MINIMUM_REPETITIONS", 1_000_000)
)
ZSPACE_PERIODICITY_MAX_COMPARISON_WORK = int(
    _native_constant("ZSPACE_PERIODICITY_MAX_COMPARISON_WORK", 16_777_216)
)
ZSPACE_PERIODICITY_MAX_SAFE_INTEGER = int(
    _native_constant("ZSPACE_PERIODICITY_MAX_SAFE_INTEGER", 9_007_199_254_740_991)
)

__all__ = [
    "ZSPACE_PERIODICITY_ANALYSIS_ID_RULE",
    "ZSPACE_PERIODICITY_CONTRACT_VERSION",
    "ZSPACE_PERIODICITY_EVIDENCE_BOUNDARY",
    "ZSPACE_PERIODICITY_KIND",
    "ZSPACE_PERIODICITY_MAX_COMPARISON_WORK",
    "ZSPACE_PERIODICITY_MAX_MINIMUM_REPETITIONS",
    "ZSPACE_PERIODICITY_MAX_PERIOD",
    "ZSPACE_PERIODICITY_MAX_SAFE_INTEGER",
    "ZSPACE_PERIODICITY_MAX_TOKENS",
    "ZSPACE_PERIODICITY_RULE",
    "ZSPACE_PERIODICITY_SEMANTIC_BACKEND",
    "ZSPACE_PERIODICITY_SEMANTIC_OWNER",
    "ZSPACE_PERIODIC_SUFFIX_MAX_PERIOD",
    "ZSPACE_PERIODIC_SUFFIX_MIN_REPETITIONS",
    "validate_zspace_periodicity",
    "zspace_periodicity",
]


def _native_operation(name: str, payload: Mapping[str, object]) -> dict[str, Any]:
    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    operation = getattr(native, name, None)
    if not callable(operation):
        raise RuntimeError(
            "Z-space periodicity requires the compiled Rust semantic core; "
            f"rebuild or reinstall SpiralTorch with {name}"
        )
    contract = operation(dict(payload))
    if not isinstance(contract, Mapping):
        raise RuntimeError(f"native {name} returned a non-mapping payload")
    result = dict(contract)
    _validate_periodicity(result)
    return result


def _non_negative_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _ratio(value: object) -> bool:
    return value is None or (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and 0.0 <= float(value) <= 1.0
    )


def _validate_periodicity(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("kind") != ZSPACE_PERIODICITY_KIND
        or contract.get("contract_version") != ZSPACE_PERIODICITY_CONTRACT_VERSION
        or contract.get("semantic_owner") != ZSPACE_PERIODICITY_SEMANTIC_OWNER
        or contract.get("semantic_backend") != ZSPACE_PERIODICITY_SEMANTIC_BACKEND
        or contract.get("analysis_validated") is not True
        or contract.get("status") != "ready"
        or contract.get("rule") != ZSPACE_PERIODICITY_RULE
        or contract.get("analysis_id_rule") != ZSPACE_PERIODICITY_ANALYSIS_ID_RULE
        or contract.get("evidence_boundary") != ZSPACE_PERIODICITY_EVIDENCE_BOUNDARY
        or contract.get("analysis_scope")
        not in {"observed_sequence", "observed_sequence_with_appended_token"}
        or contract.get("efficacy_claim_ready") is not False
    ):
        raise RuntimeError("native Z-space core returned untrusted periodicity analysis")

    analysis_id = contract.get("analysis_id")
    request = contract.get("request")
    suffix = contract.get("periodic_suffix")
    periodic_loop_detected = contract.get("periodic_loop_detected")
    if (
        not isinstance(analysis_id, str)
        or not analysis_id.startswith("sha256:")
        or len(analysis_id) != 71
        or not isinstance(request, Mapping)
        or not isinstance(request.get("token_ids"), list)
        or not isinstance(request.get("config"), Mapping)
        or not isinstance(periodic_loop_detected, bool)
        or not _non_negative_int(contract.get("input_token_count"))
        or not _non_negative_int(contract.get("effective_token_count"))
        or not _non_negative_int(contract.get("candidate_period_count"))
        or not _non_negative_int(contract.get("comparison_work_upper_bound"))
        or not _ratio(contract.get("periodic_suffix_token_ratio"))
        or not _ratio(contract.get("periodic_suffix_repeated_token_ratio"))
    ):
        raise RuntimeError("native Z-space periodicity returned invalid dimensions")
    if periodic_loop_detected != isinstance(suffix, Mapping):
        raise RuntimeError("native Z-space periodicity returned an inconsistent suffix")


def zspace_periodicity(
    token_ids: Sequence[int],
    *,
    appended_token_id: Optional[int] = None,
    maximum_period: int = ZSPACE_PERIODIC_SUFFIX_MAX_PERIOD,
    minimum_repetitions: int = ZSPACE_PERIODIC_SUFFIX_MIN_REPETITIONS,
) -> dict[str, Any]:
    """Analyze a token suffix in the canonical bounded Rust kernel."""

    return _native_operation(
        "_zspace_periodicity",
        {
            "token_ids": list(token_ids),
            "appended_token_id": appended_token_id,
            "config": {
                "maximum_period": maximum_period,
                "minimum_repetitions": minimum_repetitions,
            },
        },
    )


def validate_zspace_periodicity(report: Mapping[str, object]) -> dict[str, Any]:
    """Recompute a serialized periodicity report in Rust and reject changes."""

    return _native_operation("_zspace_periodicity_validate", report)
