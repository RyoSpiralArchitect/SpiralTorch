"""Thin Python client for Rust-owned generation-quality evidence semantics."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any


def _native_constant(name: str, fallback: object) -> object:
    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    return getattr(native, name, fallback)


# Fallbacks keep source-only imports usable; every executable operation still
# requires the native core and validates these values against its report.
ZSPACE_GENERATION_EVIDENCE_CONTRACT_VERSION = str(
    _native_constant(
        "ZSPACE_GENERATION_EVIDENCE_CONTRACT_VERSION",
        "spiraltorch.zspace_generation_evidence.v1",
    )
)
ZSPACE_GENERATION_EVIDENCE_KIND = str(
    _native_constant(
        "ZSPACE_GENERATION_EVIDENCE_KIND",
        "spiraltorch.zspace_generation_evidence",
    )
)
ZSPACE_GENERATION_EVIDENCE_SEMANTIC_OWNER = str(
    _native_constant(
        "ZSPACE_GENERATION_EVIDENCE_SEMANTIC_OWNER",
        "st-core::runtime::zspace_generation_evidence",
    )
)
ZSPACE_GENERATION_EVIDENCE_SEMANTIC_BACKEND = str(
    _native_constant("ZSPACE_GENERATION_EVIDENCE_SEMANTIC_BACKEND", "rust")
)
ZSPACE_GENERATION_EVIDENCE_METRIC_RULE = str(
    _native_constant(
        "ZSPACE_GENERATION_EVIDENCE_METRIC_RULE",
        "continuation token IDs only; sample-local n-gram distinct/repeated-"
        "occurrence ratios for orders 1..4; adjacent equal-token ratio; longest "
        "trailing periodic suffix with period<=16 and >=3 repetitions",
    )
)
ZSPACE_GENERATION_EVIDENCE_LOOP_SCORE_RULE = str(
    _native_constant(
        "ZSPACE_GENERATION_EVIDENCE_LOOP_SCORE_RULE",
        "loop_score=trigram_repetition_ratio+consecutive_repetition_ratio+periodic_"
        "suffix_repeated_token_ratio; unavailable ratios contribute zero",
    )
)
ZSPACE_GENERATION_EVIDENCE_NGRAM_ORDERS = tuple(
    int(value)
    for value in _native_constant(
        "ZSPACE_GENERATION_EVIDENCE_NGRAM_ORDERS", (1, 2, 3, 4)
    )
)
ZSPACE_GENERATION_EVIDENCE_PERIODIC_SUFFIX_MAX_PERIOD = int(
    _native_constant("ZSPACE_GENERATION_EVIDENCE_PERIODIC_SUFFIX_MAX_PERIOD", 16)
)
ZSPACE_GENERATION_EVIDENCE_PERIODIC_SUFFIX_MIN_REPETITIONS = int(
    _native_constant("ZSPACE_GENERATION_EVIDENCE_PERIODIC_SUFFIX_MIN_REPETITIONS", 3)
)
ZSPACE_GENERATION_EVIDENCE_MAX_SAMPLES = int(
    _native_constant("ZSPACE_GENERATION_EVIDENCE_MAX_SAMPLES", 10_000)
)
ZSPACE_GENERATION_EVIDENCE_MAX_TOKENS_PER_SAMPLE = int(
    _native_constant("ZSPACE_GENERATION_EVIDENCE_MAX_TOKENS_PER_SAMPLE", 16_384)
)
ZSPACE_GENERATION_EVIDENCE_MAX_TOTAL_TOKENS = int(
    _native_constant("ZSPACE_GENERATION_EVIDENCE_MAX_TOTAL_TOKENS", 1_000_000)
)
ZSPACE_GENERATION_EVIDENCE_MAX_SAFE_INTEGER = int(
    _native_constant(
        "ZSPACE_GENERATION_EVIDENCE_MAX_SAFE_INTEGER", 9_007_199_254_740_991
    )
)

__all__ = [
    "ZSPACE_GENERATION_EVIDENCE_CONTRACT_VERSION",
    "ZSPACE_GENERATION_EVIDENCE_KIND",
    "ZSPACE_GENERATION_EVIDENCE_LOOP_SCORE_RULE",
    "ZSPACE_GENERATION_EVIDENCE_MAX_SAFE_INTEGER",
    "ZSPACE_GENERATION_EVIDENCE_MAX_SAMPLES",
    "ZSPACE_GENERATION_EVIDENCE_MAX_TOKENS_PER_SAMPLE",
    "ZSPACE_GENERATION_EVIDENCE_MAX_TOTAL_TOKENS",
    "ZSPACE_GENERATION_EVIDENCE_METRIC_RULE",
    "ZSPACE_GENERATION_EVIDENCE_NGRAM_ORDERS",
    "ZSPACE_GENERATION_EVIDENCE_PERIODIC_SUFFIX_MAX_PERIOD",
    "ZSPACE_GENERATION_EVIDENCE_PERIODIC_SUFFIX_MIN_REPETITIONS",
    "ZSPACE_GENERATION_EVIDENCE_SEMANTIC_BACKEND",
    "ZSPACE_GENERATION_EVIDENCE_SEMANTIC_OWNER",
    "validate_zspace_generation_evidence",
    "zspace_generation_evidence",
]


_MAX_NATIVE_ROOT_FIELDS = 32
_MAX_SAMPLE_FIELDS = 3


def _bounded_mapping_snapshot(
    value: dict[str, object],
    *,
    maximum: int,
    label: str,
) -> dict[str, object]:
    # Base descriptors inspect the concrete C type without consulting __class__.
    try:
        field_count = dict.__len__(value)
    except TypeError:
        raise TypeError(
            f"{label} must be a dict-backed mapping for bounded admission"
        ) from None
    if field_count > maximum:
        raise ValueError(f"{label} field count exceeds maximum {maximum}")
    return dict.copy(value)


def _native_operation(name: str, payload: dict[str, object]) -> dict[str, Any]:
    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    operation = getattr(native, name, None)
    if not callable(operation):
        raise RuntimeError(
            "Z-space generation evidence requires the compiled Rust semantic core; "
            f"rebuild or reinstall SpiralTorch with {name}"
        )
    contract = operation(
        _bounded_mapping_snapshot(
            payload,
            maximum=_MAX_NATIVE_ROOT_FIELDS,
            label=f"native {name} payload",
        )
    )
    if not isinstance(contract, Mapping):
        raise RuntimeError(f"native {name} returned a non-mapping payload")
    result = dict(contract)
    _validate_generation_evidence(result)
    return result


def _positive_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _non_negative_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _validate_generation_evidence(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("kind") != ZSPACE_GENERATION_EVIDENCE_KIND
        or contract.get("contract_version")
        != ZSPACE_GENERATION_EVIDENCE_CONTRACT_VERSION
        or contract.get("semantic_owner") != ZSPACE_GENERATION_EVIDENCE_SEMANTIC_OWNER
        or contract.get("semantic_backend")
        != ZSPACE_GENERATION_EVIDENCE_SEMANTIC_BACKEND
        or contract.get("evidence_validated") is not True
        or contract.get("status") != "ready"
        or contract.get("metric_rule") != ZSPACE_GENERATION_EVIDENCE_METRIC_RULE
        or contract.get("loop_score_rule") != ZSPACE_GENERATION_EVIDENCE_LOOP_SCORE_RULE
        or tuple(contract.get("ngram_orders") or ())
        != ZSPACE_GENERATION_EVIDENCE_NGRAM_ORDERS
        or contract.get("periodic_suffix_max_period")
        != ZSPACE_GENERATION_EVIDENCE_PERIODIC_SUFFIX_MAX_PERIOD
        or contract.get("periodic_suffix_min_repetitions")
        != ZSPACE_GENERATION_EVIDENCE_PERIODIC_SUFFIX_MIN_REPETITIONS
        or contract.get("evidence_scope") != "held_out_generation_token_observation"
        or contract.get("efficacy_claim_ready") is not False
    ):
        raise RuntimeError("native Z-space core returned untrusted generation evidence")
    evidence_id = contract.get("evidence_id")
    sample_count = contract.get("sample_count")
    request = contract.get("request")
    samples = contract.get("samples")
    aggregate = contract.get("aggregate")
    if (
        not isinstance(evidence_id, str)
        or not evidence_id.startswith("sha256:")
        or len(evidence_id) != 71
        or not _positive_int(sample_count)
        or not isinstance(request, Mapping)
        or not isinstance(samples, list)
        or len(samples) != sample_count
        or not isinstance(aggregate, Mapping)
        or aggregate.get("sample_count") != sample_count
    ):
        raise RuntimeError(
            "native Z-space generation evidence returned invalid dimensions"
        )
    if (
        not _non_negative_int(aggregate.get("total_token_count"))
        or not _non_negative_int(aggregate.get("periodic_loop_sample_count"))
        or not isinstance(aggregate.get("ngrams"), list)
        or [row.get("order") for row in aggregate["ngrams"] if isinstance(row, Mapping)]
        != list(ZSPACE_GENERATION_EVIDENCE_NGRAM_ORDERS)
    ):
        raise RuntimeError(
            "native Z-space generation evidence returned invalid aggregate metrics"
        )


def _bounded_samples(
    samples: list[dict[str, object]] | tuple[dict[str, object], ...],
) -> list[dict[str, object]]:
    # Keep admission hook-free while retaining list/tuple subclasses.
    try:
        sample_count = list.__len__(samples)
        sample_iterator = list.__iter__(samples)
    except TypeError:
        try:
            sample_count = tuple.__len__(samples)
            sample_iterator = tuple.__iter__(samples)
        except TypeError:
            raise TypeError(
                "generation evidence samples must be a list or tuple for bounded "
                "admission"
            ) from None
    if sample_count > ZSPACE_GENERATION_EVIDENCE_MAX_SAMPLES:
        raise ValueError(
            "generation evidence sample count exceeds maximum "
            f"{ZSPACE_GENERATION_EVIDENCE_MAX_SAMPLES}"
        )

    snapshot: list[dict[str, object]] = []
    for index, sample in enumerate(sample_iterator):
        snapshot.append(
            _bounded_mapping_snapshot(
                sample,
                maximum=_MAX_SAMPLE_FIELDS,
                label=f"generation evidence sample[{index}]",
            )
        )
    return snapshot


def zspace_generation_evidence(
    *,
    protocol_id: str,
    runtime_identity_id: str,
    model_artifact_id: str,
    prompt_set_id: str,
    decoding_config_id: str,
    samples: list[dict[str, object]] | tuple[dict[str, object], ...],
) -> dict[str, Any]:
    """Summarize held-out continuation token IDs in the canonical Rust core."""

    return _native_operation(
        "_zspace_generation_evidence",
        {
            "protocol_id": protocol_id,
            "runtime_identity_id": runtime_identity_id,
            "model_artifact_id": model_artifact_id,
            "prompt_set_id": prompt_set_id,
            "decoding_config_id": decoding_config_id,
            "samples": _bounded_samples(samples),
        },
    )


def validate_zspace_generation_evidence(
    report: dict[str, object],
) -> dict[str, Any]:
    """Recompute a serialized generation report in Rust and reject changes."""

    return _native_operation("_zspace_generation_evidence_validate", report)
