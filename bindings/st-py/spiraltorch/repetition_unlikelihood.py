"""Thin Python client for Rust-owned repetition-unlikelihood plans."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from typing import Any


def _native_constant(name: str, fallback: object) -> object:
    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    return getattr(native, name, fallback)


# Source-only imports retain introspection, but executable planning always
# requires the native core so clients cannot silently fork the semantics.
ZSPACE_REPETITION_UNLIKELIHOOD_CONTRACT_VERSION = str(
    _native_constant(
        "ZSPACE_REPETITION_UNLIKELIHOOD_CONTRACT_VERSION",
        "spiraltorch.zspace_repetition_unlikelihood.v3",
    )
)
ZSPACE_REPETITION_UNLIKELIHOOD_KIND = str(
    _native_constant(
        "ZSPACE_REPETITION_UNLIKELIHOOD_KIND",
        "spiraltorch.zspace_repetition_unlikelihood_plan",
    )
)
ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_OWNER = str(
    _native_constant(
        "ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_OWNER",
        "st-core::runtime::zspace_repetition_unlikelihood",
    )
)
ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_BACKEND = str(
    _native_constant("ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_BACKEND", "rust")
)
ZSPACE_REPETITION_UNLIKELIHOOD_DIFFERENTIATION_OWNER = str(
    _native_constant(
        "ZSPACE_REPETITION_UNLIKELIHOOD_DIFFERENTIATION_OWNER",
        "model-client-autograd",
    )
)
ZSPACE_REPETITION_UNLIKELIHOOD_PROPOSAL_OWNER = str(
    _native_constant(
        "ZSPACE_REPETITION_UNLIKELIHOOD_PROPOSAL_OWNER",
        "model-client-no-grad",
    )
)
ZSPACE_REPETITION_UNLIKELIHOOD_PROPOSAL_RULE = str(
    _native_constant("ZSPACE_REPETITION_UNLIKELIHOOD_PROPOSAL_RULE", "")
)
ZSPACE_REPETITION_UNLIKELIHOOD_CANDIDATE_RULE = str(
    _native_constant("ZSPACE_REPETITION_UNLIKELIHOOD_CANDIDATE_RULE", "")
)
ZSPACE_REPETITION_UNLIKELIHOOD_OBJECTIVE_RULE = str(
    _native_constant("ZSPACE_REPETITION_UNLIKELIHOOD_OBJECTIVE_RULE", "")
)
ZSPACE_REPETITION_UNLIKELIHOOD_PROBABILITY_EPSILON = float(
    _native_constant("ZSPACE_REPETITION_UNLIKELIHOOD_PROBABILITY_EPSILON", 1.0e-6)
)
ZSPACE_REPETITION_UNLIKELIHOOD_MAX_STRENGTH = float(
    _native_constant("ZSPACE_REPETITION_UNLIKELIHOOD_MAX_STRENGTH", 10.0)
)
ZSPACE_REPETITION_UNLIKELIHOOD_MIN_NGRAM_ORDER = int(
    _native_constant("ZSPACE_REPETITION_UNLIKELIHOOD_MIN_NGRAM_ORDER", 2)
)
ZSPACE_REPETITION_UNLIKELIHOOD_MAX_NGRAM_ORDER = int(
    _native_constant("ZSPACE_REPETITION_UNLIKELIHOOD_MAX_NGRAM_ORDER", 8)
)
ZSPACE_REPETITION_UNLIKELIHOOD_MAX_CONTEXT_WINDOW = int(
    _native_constant("ZSPACE_REPETITION_UNLIKELIHOOD_MAX_CONTEXT_WINDOW", 16_384)
)
ZSPACE_REPETITION_UNLIKELIHOOD_MAX_CANDIDATES_PER_POSITION = int(
    _native_constant("ZSPACE_REPETITION_UNLIKELIHOOD_MAX_CANDIDATES_PER_POSITION", 64)
)
ZSPACE_REPETITION_UNLIKELIHOOD_MAX_PROPOSAL_TOP_K = int(
    _native_constant("ZSPACE_REPETITION_UNLIKELIHOOD_MAX_PROPOSAL_TOP_K", 64)
)
ZSPACE_REPETITION_UNLIKELIHOOD_PERIODIC_SUFFIX_MAX_PERIOD = int(
    _native_constant("ZSPACE_REPETITION_UNLIKELIHOOD_PERIODIC_SUFFIX_MAX_PERIOD", 16)
)
ZSPACE_REPETITION_UNLIKELIHOOD_PERIODIC_SUFFIX_MIN_REPETITIONS = int(
    _native_constant(
        "ZSPACE_REPETITION_UNLIKELIHOOD_PERIODIC_SUFFIX_MIN_REPETITIONS", 3
    )
)
ZSPACE_REPETITION_UNLIKELIHOOD_MAX_SEQUENCES = int(
    _native_constant("ZSPACE_REPETITION_UNLIKELIHOOD_MAX_SEQUENCES", 4_096)
)
ZSPACE_REPETITION_UNLIKELIHOOD_MAX_TOKENS_PER_SEQUENCE = int(
    _native_constant("ZSPACE_REPETITION_UNLIKELIHOOD_MAX_TOKENS_PER_SEQUENCE", 16_384)
)
ZSPACE_REPETITION_UNLIKELIHOOD_MAX_TOTAL_TOKENS = int(
    _native_constant("ZSPACE_REPETITION_UNLIKELIHOOD_MAX_TOTAL_TOKENS", 1_000_000)
)
ZSPACE_REPETITION_UNLIKELIHOOD_MAX_WORK_UNITS = int(
    _native_constant("ZSPACE_REPETITION_UNLIKELIHOOD_MAX_WORK_UNITS", 64_000_000)
)
ZSPACE_REPETITION_UNLIKELIHOOD_MAX_MATERIALIZED_PLAN_BYTES = int(
    _native_constant(
        "ZSPACE_REPETITION_UNLIKELIHOOD_MAX_MATERIALIZED_PLAN_BYTES",
        32 * 1_024 * 1_024,
    )
)
ZSPACE_REPETITION_UNLIKELIHOOD_MAX_SAFE_INTEGER = int(
    _native_constant(
        "ZSPACE_REPETITION_UNLIKELIHOOD_MAX_SAFE_INTEGER", 9_007_199_254_740_991
    )
)
ZSPACE_REPETITION_UNLIKELIHOOD_WORK_UNIT_RULE = str(
    _native_constant("ZSPACE_REPETITION_UNLIKELIHOOD_WORK_UNIT_RULE", "")
)
ZSPACE_REPETITION_UNLIKELIHOOD_MATERIALIZED_PLAN_BYTE_RULE = str(
    _native_constant("ZSPACE_REPETITION_UNLIKELIHOOD_MATERIALIZED_PLAN_BYTE_RULE", "")
)

__all__ = [
    "ZSPACE_REPETITION_UNLIKELIHOOD_CANDIDATE_RULE",
    "ZSPACE_REPETITION_UNLIKELIHOOD_CONTRACT_VERSION",
    "ZSPACE_REPETITION_UNLIKELIHOOD_DIFFERENTIATION_OWNER",
    "ZSPACE_REPETITION_UNLIKELIHOOD_KIND",
    "ZSPACE_REPETITION_UNLIKELIHOOD_MATERIALIZED_PLAN_BYTE_RULE",
    "ZSPACE_REPETITION_UNLIKELIHOOD_MAX_CANDIDATES_PER_POSITION",
    "ZSPACE_REPETITION_UNLIKELIHOOD_MAX_CONTEXT_WINDOW",
    "ZSPACE_REPETITION_UNLIKELIHOOD_MAX_MATERIALIZED_PLAN_BYTES",
    "ZSPACE_REPETITION_UNLIKELIHOOD_MAX_NGRAM_ORDER",
    "ZSPACE_REPETITION_UNLIKELIHOOD_MAX_PROPOSAL_TOP_K",
    "ZSPACE_REPETITION_UNLIKELIHOOD_MAX_SAFE_INTEGER",
    "ZSPACE_REPETITION_UNLIKELIHOOD_MAX_SEQUENCES",
    "ZSPACE_REPETITION_UNLIKELIHOOD_MAX_STRENGTH",
    "ZSPACE_REPETITION_UNLIKELIHOOD_MAX_TOKENS_PER_SEQUENCE",
    "ZSPACE_REPETITION_UNLIKELIHOOD_MAX_TOTAL_TOKENS",
    "ZSPACE_REPETITION_UNLIKELIHOOD_MAX_WORK_UNITS",
    "ZSPACE_REPETITION_UNLIKELIHOOD_MIN_NGRAM_ORDER",
    "ZSPACE_REPETITION_UNLIKELIHOOD_OBJECTIVE_RULE",
    "ZSPACE_REPETITION_UNLIKELIHOOD_PERIODIC_SUFFIX_MAX_PERIOD",
    "ZSPACE_REPETITION_UNLIKELIHOOD_PERIODIC_SUFFIX_MIN_REPETITIONS",
    "ZSPACE_REPETITION_UNLIKELIHOOD_PROBABILITY_EPSILON",
    "ZSPACE_REPETITION_UNLIKELIHOOD_PROPOSAL_OWNER",
    "ZSPACE_REPETITION_UNLIKELIHOOD_PROPOSAL_RULE",
    "ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_BACKEND",
    "ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_OWNER",
    "ZSPACE_REPETITION_UNLIKELIHOOD_WORK_UNIT_RULE",
    "validate_zspace_repetition_unlikelihood_plan",
    "validate_zspace_repetition_unlikelihood_plan_trusted_legacy_replay",
    "zspace_repetition_unlikelihood_plan",
]


_MAX_NATIVE_ROOT_FIELDS = 64
_MAX_SEQUENCE_FIELDS = 4
_MAX_CANDIDATE_SOURCE_FIELDS = 2


def _bounded_mapping_snapshot(
    value: Mapping[str, object],
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


def _native_operation(name: str, payload: Mapping[str, object]) -> dict[str, Any]:
    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    operation = getattr(native, name, None)
    if not callable(operation):
        raise RuntimeError(
            "Z-space repetition unlikelihood requires the compiled Rust semantic "
            f"core; rebuild or reinstall SpiralTorch with {name}"
        )
    result = operation(
        _bounded_mapping_snapshot(
            payload,
            maximum=_MAX_NATIVE_ROOT_FIELDS,
            label=f"native {name} payload",
        )
    )
    if not isinstance(result, Mapping):
        raise RuntimeError(f"native {name} returned a non-mapping payload")
    plan = dict(result)
    _validate_plan(plan)
    return plan


def _non_negative_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _validate_plan(plan: Mapping[str, Any]) -> None:
    if (
        plan.get("contract_version") != ZSPACE_REPETITION_UNLIKELIHOOD_CONTRACT_VERSION
        or plan.get("kind") != ZSPACE_REPETITION_UNLIKELIHOOD_KIND
        or plan.get("semantic_owner") != ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_OWNER
        or plan.get("semantic_backend")
        != ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_BACKEND
        or plan.get("differentiation_owner")
        != ZSPACE_REPETITION_UNLIKELIHOOD_DIFFERENTIATION_OWNER
        or plan.get("proposal_owner") != ZSPACE_REPETITION_UNLIKELIHOOD_PROPOSAL_OWNER
        or plan.get("plan_validated") is not True
        or plan.get("status") != "ready"
        or plan.get("candidate_rule") != ZSPACE_REPETITION_UNLIKELIHOOD_CANDIDATE_RULE
        or plan.get("proposal_rule") != ZSPACE_REPETITION_UNLIKELIHOOD_PROPOSAL_RULE
        or plan.get("objective_rule") != ZSPACE_REPETITION_UNLIKELIHOOD_OBJECTIVE_RULE
        or plan.get("probability_epsilon")
        != ZSPACE_REPETITION_UNLIKELIHOOD_PROBABILITY_EPSILON
        or plan.get("periodic_suffix_max_period")
        != ZSPACE_REPETITION_UNLIKELIHOOD_PERIODIC_SUFFIX_MAX_PERIOD
        or plan.get("periodic_suffix_min_repetitions")
        != ZSPACE_REPETITION_UNLIKELIHOOD_PERIODIC_SUFFIX_MIN_REPETITIONS
        or plan.get("efficacy_claim_ready") is not False
    ):
        raise RuntimeError(
            "native Z-space core returned an untrusted repetition-unlikelihood plan"
        )
    plan_id = plan.get("plan_id")
    request = plan.get("request")
    positions = plan.get("positions")
    aggregate = plan.get("aggregate")
    if (
        not isinstance(plan_id, str)
        or not plan_id.startswith("sha256:")
        or len(plan_id) != 71
        or not isinstance(request, Mapping)
        or not isinstance(positions, list)
        or not isinstance(aggregate, Mapping)
        or aggregate.get("active_position_count") != len(positions)
        or not _non_negative_int(aggregate.get("candidate_count"))
    ):
        raise RuntimeError(
            "native Z-space core returned invalid repetition-unlikelihood dimensions"
        )
    candidate_count = 0
    for position in positions:
        if not isinstance(position, Mapping) or not isinstance(
            position.get("candidates"), list
        ):
            raise RuntimeError(
                "native Z-space core returned malformed repetition candidates"
            )
        candidate_count += len(position["candidates"])
    if aggregate.get("candidate_count") != candidate_count:
        raise RuntimeError(
            "native Z-space core returned inconsistent repetition candidate counts"
        )


def _bounded_sequences(
    sequences: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    # Keep admission hook-free while retaining list/tuple subclasses.
    try:
        sequence_count = list.__len__(sequences)
        sequence_iterator = list.__iter__(sequences)
    except TypeError:
        try:
            sequence_count = tuple.__len__(sequences)
            sequence_iterator = tuple.__iter__(sequences)
        except TypeError:
            raise TypeError(
                "repetition-unlikelihood sequences must be a list or tuple for "
                "bounded admission"
            ) from None
    if sequence_count > ZSPACE_REPETITION_UNLIKELIHOOD_MAX_SEQUENCES:
        raise ValueError(
            "repetition-unlikelihood sequence count exceeds maximum "
            f"{ZSPACE_REPETITION_UNLIKELIHOOD_MAX_SEQUENCES}"
        )

    snapshot: list[dict[str, object]] = []
    for index, sequence in enumerate(sequence_iterator):
        snapshot.append(
            _bounded_mapping_snapshot(
                sequence,
                maximum=_MAX_SEQUENCE_FIELDS,
                label=f"repetition-unlikelihood sequence[{index}]",
            )
        )
    return snapshot


def _candidate_source_payload(
    candidate_source: str | Mapping[str, object],
    *,
    ngram_order: int,
    proposal_top_k: int,
) -> dict[str, object]:
    # Avoid Mapping checks and subclass comparison hooks before Rust admission.
    try:
        return _bounded_mapping_snapshot(
            candidate_source,
            maximum=_MAX_CANDIDATE_SOURCE_FIELDS,
            label="repetition-unlikelihood candidate source",
        )
    except TypeError:
        pass
    try:
        source = str.__str__(candidate_source)
    except TypeError:
        source = None
    if source == "prior_continuation":
        return {"kind": source, "ngram_order": ngram_order}
    if source in {"model_topk_history", "model_topk_periodic"}:
        return {"kind": source, "proposal_top_k": proposal_top_k}
    raise ValueError(
        "candidate_source must be prior_continuation, model_topk_history, or "
        "model_topk_periodic"
    )


def zspace_repetition_unlikelihood_plan(
    *,
    sequences: Sequence[Mapping[str, object]],
    strength: float = 0.1,
    candidate_source: str | Mapping[str, object] = "prior_continuation",
    ngram_order: int = 3,
    proposal_top_k: int = 8,
    context_window: int = 128,
    max_candidates_per_position: int = 8,
) -> dict[str, Any]:
    """Plan bounded negative tokens in the canonical Rust semantic core."""

    return _native_operation(
        "_zspace_repetition_unlikelihood_plan",
        {
            "config": {
                "strength": strength,
                "candidate_source": _candidate_source_payload(
                    candidate_source,
                    ngram_order=ngram_order,
                    proposal_top_k=proposal_top_k,
                ),
                "context_window": context_window,
                "max_candidates_per_position": max_candidates_per_position,
            },
            "sequences": _bounded_sequences(sequences),
        },
    )


def validate_zspace_repetition_unlikelihood_plan(
    plan: Mapping[str, object],
) -> dict[str, Any]:
    """Recompute a plan in Rust within the shared work and materialization budgets."""

    return _native_operation("_zspace_repetition_unlikelihood_validate", plan)


def validate_zspace_repetition_unlikelihood_plan_trusted_legacy_replay(
    plan: Mapping[str, object],
) -> dict[str, Any]:
    """Replay a trusted historical v3 plan without the newer admission budgets.

    Never pass untrusted or remotely supplied input to this opt-in path. Use
    :func:`validate_zspace_repetition_unlikelihood_plan` for normal validation.
    """

    return _native_operation(
        "_zspace_repetition_unlikelihood_validate_trusted_legacy_replay",
        plan,
    )
