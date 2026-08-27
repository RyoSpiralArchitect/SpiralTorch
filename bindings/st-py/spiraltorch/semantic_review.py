"""Thin Python orchestration for Rust-owned blinded semantic review contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _native_constant(name: str, fallback: object) -> object:
    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    return getattr(native, name, fallback)


ZSPACE_SEMANTIC_REVIEW_PACKET_SCHEMA = str(
    _native_constant(
        "ZSPACE_SEMANTIC_REVIEW_PACKET_SCHEMA",
        "spiraltorch.hf_blinded_semantic_review_packet.v1",
    )
)
ZSPACE_SEMANTIC_REVIEW_PACKET_STATUS = str(
    _native_constant("ZSPACE_SEMANTIC_REVIEW_PACKET_STATUS", "ready_for_blinded_review")
)
ZSPACE_SEMANTIC_REVIEW_MAP_SCHEMA = str(
    _native_constant(
        "ZSPACE_SEMANTIC_REVIEW_MAP_SCHEMA",
        "spiraltorch.hf_blinded_semantic_review_map.v1",
    )
)
ZSPACE_SEMANTIC_REVIEW_MAP_STATUS = str(
    _native_constant("ZSPACE_SEMANTIC_REVIEW_MAP_STATUS", "sealed_pending_review")
)
ZSPACE_SEMANTIC_REVIEW_DRAFT_SCHEMA = str(
    _native_constant(
        "ZSPACE_SEMANTIC_REVIEW_DRAFT_SCHEMA",
        "spiraltorch.hf_blinded_semantic_review_draft.v1",
    )
)
ZSPACE_SEMANTIC_REVIEW_PACKET_CONTRACT_VERSION = str(
    _native_constant(
        "ZSPACE_SEMANTIC_REVIEW_PACKET_CONTRACT_VERSION",
        "spiraltorch.zspace_semantic_review_packet.v1",
    )
)
ZSPACE_SEMANTIC_REVIEW_DRAFT_CONTRACT_VERSION = str(
    _native_constant(
        "ZSPACE_SEMANTIC_REVIEW_DRAFT_CONTRACT_VERSION",
        "spiraltorch.zspace_semantic_review_draft.v1",
    )
)
ZSPACE_SEMANTIC_REVIEW_RESPONSE_CONTRACT_VERSION = str(
    _native_constant(
        "ZSPACE_SEMANTIC_REVIEW_RESPONSE_CONTRACT_VERSION",
        "spiraltorch.zspace_semantic_review_response.v1",
    )
)
ZSPACE_SEMANTIC_REVIEW_MAP_COMMITMENT_VERSION = str(
    _native_constant(
        "ZSPACE_SEMANTIC_REVIEW_MAP_COMMITMENT_VERSION",
        "spiraltorch.zspace_semantic_review_map_commitment.v1",
    )
)
ZSPACE_SEMANTIC_REVIEW_UNBLIND_CONTRACT_VERSION = str(
    _native_constant(
        "ZSPACE_SEMANTIC_REVIEW_UNBLIND_CONTRACT_VERSION",
        "spiraltorch.zspace_semantic_review_unblind.v1",
    )
)
ZSPACE_SEMANTIC_REVIEW_PACKET_KIND = str(
    _native_constant(
        "ZSPACE_SEMANTIC_REVIEW_PACKET_KIND",
        "spiraltorch.zspace_semantic_review_packet",
    )
)
ZSPACE_SEMANTIC_REVIEW_DRAFT_KIND = str(
    _native_constant(
        "ZSPACE_SEMANTIC_REVIEW_DRAFT_KIND",
        "spiraltorch.zspace_semantic_review_draft",
    )
)
ZSPACE_SEMANTIC_REVIEW_UNBLIND_KIND = str(
    _native_constant(
        "ZSPACE_SEMANTIC_REVIEW_UNBLIND_KIND",
        "spiraltorch.zspace_semantic_review_unblind",
    )
)
ZSPACE_SEMANTIC_REVIEW_SEMANTIC_OWNER = str(
    _native_constant(
        "ZSPACE_SEMANTIC_REVIEW_SEMANTIC_OWNER",
        "st-core::runtime::zspace_semantic_review",
    )
)
ZSPACE_SEMANTIC_REVIEW_SEMANTIC_BACKEND = str(
    _native_constant("ZSPACE_SEMANTIC_REVIEW_SEMANTIC_BACKEND", "rust")
)
ZSPACE_SEMANTIC_REVIEW_PACKET_ID_RULE = str(
    _native_constant(
        "ZSPACE_SEMANTIC_REVIEW_PACKET_ID_RULE",
        "sha256 of UTF-8 canonical JSON for the packet without packet_id; "
        "object keys sorted lexicographically, arrays preserved, no insignificant "
        "whitespace",
    )
)
ZSPACE_SEMANTIC_REVIEW_PACKET_TEXT_BYTE_RULE = str(
    _native_constant(
        "ZSPACE_SEMANTIC_REVIEW_PACKET_TEXT_BYTE_RULE",
        "sum of UTF-8 JSON-encoded string-content bytes across packet values "
        "and dynamic rubric keys; fixed field names, quotes, punctuation, and "
        "container overhead excluded",
    )
)
ZSPACE_SEMANTIC_REVIEW_MAP_ID_RULE = str(
    _native_constant(
        "ZSPACE_SEMANTIC_REVIEW_MAP_ID_RULE",
        "sha256 of UTF-8 canonical JSON for [map commitment version, map entries "
        "sorted by group_id]; object keys sorted lexicographically, no "
        "insignificant whitespace",
    )
)
ZSPACE_SEMANTIC_REVIEW_EVIDENCE_BOUNDARY = str(
    _native_constant(
        "ZSPACE_SEMANTIC_REVIEW_EVIDENCE_BOUNDARY",
        "the contract verifies packet and pre-review map-content commitments, "
        "structural blinding inputs, score bounds, coverage, and deterministic "
        "aggregation; it cannot prove that a reviewer remained blind or establish "
        "statistical or model superiority",
    )
)
ZSPACE_SEMANTIC_REVIEW_CANDIDATE_LABELS = tuple(
    str(value)
    for value in _native_constant(
        "ZSPACE_SEMANTIC_REVIEW_CANDIDATE_LABELS", ("A", "B", "C")
    )
)
ZSPACE_SEMANTIC_REVIEW_SCORE_DIMENSIONS = tuple(
    str(value)
    for value in _native_constant(
        "ZSPACE_SEMANTIC_REVIEW_SCORE_DIMENSIONS",
        ("fluency", "prompt_relevance", "local_coherence", "non_repetition"),
    )
)
ZSPACE_SEMANTIC_REVIEW_PREFERENCE_VALUES = tuple(
    str(value)
    for value in _native_constant(
        "ZSPACE_SEMANTIC_REVIEW_PREFERENCE_VALUES", ("A", "B", "C", "tie")
    )
)
ZSPACE_SEMANTIC_REVIEW_SCORE_MINIMUM = int(
    _native_constant("ZSPACE_SEMANTIC_REVIEW_SCORE_MINIMUM", 1)
)
ZSPACE_SEMANTIC_REVIEW_SCORE_MAXIMUM = int(
    _native_constant("ZSPACE_SEMANTIC_REVIEW_SCORE_MAXIMUM", 5)
)
ZSPACE_SEMANTIC_REVIEW_MAX_GROUPS = int(
    _native_constant("ZSPACE_SEMANTIC_REVIEW_MAX_GROUPS", 10_000)
)
ZSPACE_SEMANTIC_REVIEW_MAX_PROMPT_BYTES = int(
    _native_constant("ZSPACE_SEMANTIC_REVIEW_MAX_PROMPT_BYTES", 16_384)
)
ZSPACE_SEMANTIC_REVIEW_MAX_CONTINUATION_BYTES = int(
    _native_constant("ZSPACE_SEMANTIC_REVIEW_MAX_CONTINUATION_BYTES", 65_536)
)
ZSPACE_SEMANTIC_REVIEW_MAX_INSTRUCTIONS_BYTES = int(
    _native_constant("ZSPACE_SEMANTIC_REVIEW_MAX_INSTRUCTIONS_BYTES", 16_384)
)
ZSPACE_SEMANTIC_REVIEW_MAX_PACKET_TEXT_BYTES = int(
    _native_constant("ZSPACE_SEMANTIC_REVIEW_MAX_PACKET_TEXT_BYTES", 32 * 1_024 * 1_024)
)
ZSPACE_SEMANTIC_REVIEW_MAX_ARM_NAME_BYTES = int(
    _native_constant("ZSPACE_SEMANTIC_REVIEW_MAX_ARM_NAME_BYTES", 128)
)
ZSPACE_SEMANTIC_REVIEW_MAX_MAP_ENTRIES = int(
    _native_constant("ZSPACE_SEMANTIC_REVIEW_MAX_MAP_ENTRIES", 10_000)
)
ZSPACE_SEMANTIC_REVIEW_MAX_SAFE_INTEGER = int(
    _native_constant("ZSPACE_SEMANTIC_REVIEW_MAX_SAFE_INTEGER", 9_007_199_254_740_991)
)

__all__ = [
    "ZSPACE_SEMANTIC_REVIEW_CANDIDATE_LABELS",
    "ZSPACE_SEMANTIC_REVIEW_DRAFT_CONTRACT_VERSION",
    "ZSPACE_SEMANTIC_REVIEW_DRAFT_KIND",
    "ZSPACE_SEMANTIC_REVIEW_DRAFT_SCHEMA",
    "ZSPACE_SEMANTIC_REVIEW_EVIDENCE_BOUNDARY",
    "ZSPACE_SEMANTIC_REVIEW_MAP_COMMITMENT_VERSION",
    "ZSPACE_SEMANTIC_REVIEW_MAP_ID_RULE",
    "ZSPACE_SEMANTIC_REVIEW_MAP_SCHEMA",
    "ZSPACE_SEMANTIC_REVIEW_MAP_STATUS",
    "ZSPACE_SEMANTIC_REVIEW_MAX_ARM_NAME_BYTES",
    "ZSPACE_SEMANTIC_REVIEW_MAX_CONTINUATION_BYTES",
    "ZSPACE_SEMANTIC_REVIEW_MAX_GROUPS",
    "ZSPACE_SEMANTIC_REVIEW_MAX_INSTRUCTIONS_BYTES",
    "ZSPACE_SEMANTIC_REVIEW_MAX_MAP_ENTRIES",
    "ZSPACE_SEMANTIC_REVIEW_MAX_PACKET_TEXT_BYTES",
    "ZSPACE_SEMANTIC_REVIEW_MAX_PROMPT_BYTES",
    "ZSPACE_SEMANTIC_REVIEW_MAX_SAFE_INTEGER",
    "ZSPACE_SEMANTIC_REVIEW_PACKET_CONTRACT_VERSION",
    "ZSPACE_SEMANTIC_REVIEW_PACKET_ID_RULE",
    "ZSPACE_SEMANTIC_REVIEW_PACKET_KIND",
    "ZSPACE_SEMANTIC_REVIEW_PACKET_SCHEMA",
    "ZSPACE_SEMANTIC_REVIEW_PACKET_STATUS",
    "ZSPACE_SEMANTIC_REVIEW_PACKET_TEXT_BYTE_RULE",
    "ZSPACE_SEMANTIC_REVIEW_PREFERENCE_VALUES",
    "ZSPACE_SEMANTIC_REVIEW_RESPONSE_CONTRACT_VERSION",
    "ZSPACE_SEMANTIC_REVIEW_SCORE_DIMENSIONS",
    "ZSPACE_SEMANTIC_REVIEW_SCORE_MAXIMUM",
    "ZSPACE_SEMANTIC_REVIEW_SCORE_MINIMUM",
    "ZSPACE_SEMANTIC_REVIEW_SEMANTIC_BACKEND",
    "ZSPACE_SEMANTIC_REVIEW_SEMANTIC_OWNER",
    "ZSPACE_SEMANTIC_REVIEW_UNBLIND_CONTRACT_VERSION",
    "ZSPACE_SEMANTIC_REVIEW_UNBLIND_KIND",
    "new_zspace_semantic_review_draft",
    "seal_zspace_semantic_review_packet",
    "summarize_zspace_semantic_review_draft",
    "unblind_zspace_semantic_review",
    "validate_zspace_semantic_review_draft_receipt",
    "validate_zspace_semantic_review_draft_receipt_trusted_legacy_replay",
    "validate_zspace_semantic_review_packet",
    "validate_zspace_semantic_review_packet_trusted_legacy_replay",
    "validate_zspace_semantic_review_packet_receipt",
    "validate_zspace_semantic_review_packet_receipt_trusted_legacy_replay",
    "validate_zspace_semantic_review_unblind",
    "validate_zspace_semantic_review_unblind_trusted_legacy_replay",
    "zspace_semantic_review_map_id",
    "zspace_semantic_review_map_id_trusted_legacy_replay",
]


def _native_operation(name: str, payload: Mapping[str, object]) -> dict[str, Any]:
    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    operation = getattr(native, name, None)
    if not callable(operation):
        raise RuntimeError(
            "Z-space semantic review requires the compiled Rust semantic core; "
            f"rebuild or reinstall SpiralTorch with {name}"
        )
    result = operation(dict(payload))
    if not isinstance(result, Mapping):
        raise RuntimeError(f"native {name} returned a non-mapping payload")
    return dict(result)


def _native_identity_operation(name: str, payload: Mapping[str, object]) -> str:
    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    operation = getattr(native, name, None)
    if not callable(operation):
        raise RuntimeError(
            "Z-space semantic review requires the compiled Rust semantic core; "
            f"rebuild or reinstall SpiralTorch with {name}"
        )
    result = operation(dict(payload))
    if not _sha256_identity(result):
        raise RuntimeError(f"native {name} returned an invalid identity")
    return result


def _bounded_mapping_sequence(
    values: Sequence[Mapping[str, object]],
    *,
    maximum: int,
    label: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, value in enumerate(values):
        if index >= maximum:
            raise ValueError(f"{label} count exceeds maximum {maximum}")
        rows.append(dict(value))
    return rows


def _sha256_identity(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value[len("sha256:") :]
    return len(digest) == 64 and all(
        character in "0123456789abcdef" for character in digest
    )


def _validate_base(
    contract: Mapping[str, Any],
    *,
    kind: str,
    contract_version: str,
    validated_field: str,
) -> None:
    if (
        contract.get("kind") != kind
        or contract.get("contract_version") != contract_version
        or contract.get("semantic_owner") != ZSPACE_SEMANTIC_REVIEW_SEMANTIC_OWNER
        or contract.get("semantic_backend") != ZSPACE_SEMANTIC_REVIEW_SEMANTIC_BACKEND
        or contract.get(validated_field) is not True
        or contract.get("efficacy_claim_ready") is not False
        or contract.get("evidence_boundary") != ZSPACE_SEMANTIC_REVIEW_EVIDENCE_BOUNDARY
    ):
        raise RuntimeError(
            "native Z-space core returned untrusted semantic review evidence"
        )


def _validate_packet_receipt(contract: Mapping[str, Any]) -> None:
    _validate_base(
        contract,
        kind=ZSPACE_SEMANTIC_REVIEW_PACKET_KIND,
        contract_version=ZSPACE_SEMANTIC_REVIEW_PACKET_CONTRACT_VERSION,
        validated_field="packet_validated",
    )
    group_count = contract.get("group_count")
    candidate_count = contract.get("candidate_count")
    packet = contract.get("packet")
    if (
        contract.get("status") != "ready"
        or not _sha256_identity(contract.get("packet_id"))
        or not isinstance(group_count, int)
        or isinstance(group_count, bool)
        or group_count <= 0
        or candidate_count != group_count * len(ZSPACE_SEMANTIC_REVIEW_CANDIDATE_LABELS)
        or tuple(contract.get("candidate_labels") or ())
        != ZSPACE_SEMANTIC_REVIEW_CANDIDATE_LABELS
        or tuple(contract.get("score_dimensions") or ())
        != ZSPACE_SEMANTIC_REVIEW_SCORE_DIMENSIONS
        or contract.get("score_minimum") != ZSPACE_SEMANTIC_REVIEW_SCORE_MINIMUM
        or contract.get("score_maximum") != ZSPACE_SEMANTIC_REVIEW_SCORE_MAXIMUM
        or tuple(contract.get("preference_values") or ())
        != ZSPACE_SEMANTIC_REVIEW_PREFERENCE_VALUES
        or contract.get("packet_id_rule") != ZSPACE_SEMANTIC_REVIEW_PACKET_ID_RULE
        or contract.get("blinding_map_id_rule") != ZSPACE_SEMANTIC_REVIEW_MAP_ID_RULE
        or not _sha256_identity(contract.get("blinding_map_id"))
        or contract.get("human_review_complete") is not False
        or contract.get("unblind_ready") is not False
        or not isinstance(packet, Mapping)
        or packet.get("packet_id") != contract.get("packet_id")
        or packet.get("blinding_map_id") != contract.get("blinding_map_id")
    ):
        raise RuntimeError("native Z-space core returned an invalid packet receipt")


def _validate_draft_receipt(contract: Mapping[str, Any]) -> None:
    _validate_base(
        contract,
        kind=ZSPACE_SEMANTIC_REVIEW_DRAFT_KIND,
        contract_version=ZSPACE_SEMANTIC_REVIEW_DRAFT_CONTRACT_VERSION,
        validated_field="draft_validated",
    )
    complete = contract.get("human_review_complete") is True
    group_count = contract.get("group_count")
    completed = contract.get("completed_group_count")
    remaining = contract.get("remaining_group_count")
    missing = contract.get("missing_group_ids")
    request = contract.get("request")
    if (
        contract.get("status") not in {"in_progress", "ready_for_unblind"}
        or not _sha256_identity(contract.get("draft_id"))
        or not _sha256_identity(contract.get("packet_id"))
        or not _sha256_identity(contract.get("reviewer_id"))
        or not _sha256_identity(contract.get("review_session_id"))
        or not isinstance(group_count, int)
        or isinstance(group_count, bool)
        or group_count <= 0
        or not isinstance(completed, int)
        or isinstance(completed, bool)
        or not isinstance(remaining, int)
        or isinstance(remaining, bool)
        or completed + remaining != group_count
        or contract.get("scored_candidate_count")
        != completed * len(ZSPACE_SEMANTIC_REVIEW_CANDIDATE_LABELS)
        or not isinstance(missing, list)
        or len(missing) != remaining
        or not isinstance(request, Mapping)
    ):
        raise RuntimeError("native Z-space core returned an invalid draft receipt")
    response_id = contract.get("response_id")
    if complete:
        if (
            contract.get("status") != "ready_for_unblind"
            or contract.get("unblind_ready") is not True
            or not _sha256_identity(response_id)
            or remaining != 0
        ):
            raise RuntimeError(
                "native Z-space core returned an invalid complete response"
            )
    elif (
        contract.get("status") != "in_progress"
        or contract.get("unblind_ready") is not False
        or response_id is not None
    ):
        raise RuntimeError("native Z-space core sealed an incomplete semantic review")


def _validate_unblind_report(contract: Mapping[str, Any]) -> None:
    _validate_base(
        contract,
        kind=ZSPACE_SEMANTIC_REVIEW_UNBLIND_KIND,
        contract_version=ZSPACE_SEMANTIC_REVIEW_UNBLIND_CONTRACT_VERSION,
        validated_field="unblind_validated",
    )
    arm_count = contract.get("arm_count")
    arms = contract.get("arms")
    seeds = contract.get("seeds")
    if (
        contract.get("status") != "unblinded"
        or not _sha256_identity(contract.get("unblind_id"))
        or not _sha256_identity(contract.get("packet_id"))
        or not _sha256_identity(contract.get("response_id"))
        or not _sha256_identity(contract.get("blinding_map_id"))
        or contract.get("human_review_complete") is not True
        or contract.get("structural_blinding_inputs_validated") is not True
        or contract.get("reviewer_blinding_behavior_verified") is not False
        or not isinstance(arm_count, int)
        or isinstance(arm_count, bool)
        or arm_count <= 1
        or not isinstance(arms, list)
        or len(arms) != arm_count
        or not isinstance(seeds, list)
        or not seeds
        or not isinstance(contract.get("request"), Mapping)
    ):
        raise RuntimeError("native Z-space core returned an invalid unblind report")


def validate_zspace_semantic_review_packet(
    packet: Mapping[str, object],
) -> dict[str, Any]:
    """Validate a blinded packet and its canonical packet commitment in Rust."""

    result = _native_operation("_zspace_semantic_review_packet", packet)
    _validate_packet_receipt(result)
    return result


def validate_zspace_semantic_review_packet_trusted_legacy_replay(
    packet: Mapping[str, object],
) -> dict[str, Any]:
    """Replay a trusted historical v1 packet without newer admission budgets.

    Never pass untrusted or remotely supplied input to this opt-in path. Use
    :func:`validate_zspace_semantic_review_packet` for normal validation.
    """

    result = _native_operation(
        "_zspace_semantic_review_packet_trusted_legacy_replay", packet
    )
    _validate_packet_receipt(result)
    return result


def seal_zspace_semantic_review_packet(
    *,
    protocol_id: str,
    prompt_set_id: str,
    blinding_key_sha256: str,
    blinding_map_id: str,
    instructions: str,
    rubric: Mapping[str, str],
    groups: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    """Build, content-address, and validate a blinded packet entirely in Rust."""

    result = _native_operation(
        "_zspace_semantic_review_packet_seal",
        {
            "protocol_id": protocol_id,
            "prompt_set_id": prompt_set_id,
            "blinding_key_sha256": blinding_key_sha256,
            "blinding_map_id": blinding_map_id,
            "instructions": instructions,
            "rubric": dict(rubric),
            "groups": _bounded_mapping_sequence(
                groups,
                maximum=ZSPACE_SEMANTIC_REVIEW_MAX_GROUPS,
                label="semantic review group",
            ),
        },
    )
    _validate_packet_receipt(result)
    return result


def zspace_semantic_review_map_id(
    entries: Sequence[Mapping[str, object]],
) -> str:
    """Commit the exact pre-review candidate-to-arm assignments in Rust."""

    return _native_identity_operation(
        "_zspace_semantic_review_map_id",
        {
            "entries": _bounded_mapping_sequence(
                entries,
                maximum=ZSPACE_SEMANTIC_REVIEW_MAX_MAP_ENTRIES,
                label="semantic review map entry",
            )
        },
    )


def zspace_semantic_review_map_id_trusted_legacy_replay(
    entries: Sequence[Mapping[str, object]],
) -> str:
    """Replay a trusted historical v1 map commitment without newer budgets.

    Never pass untrusted or remotely supplied input to this opt-in path.
    """

    return _native_identity_operation(
        "_zspace_semantic_review_map_id_trusted_legacy_replay",
        {"entries": [dict(entry) for entry in entries]},
    )


def validate_zspace_semantic_review_packet_receipt(
    receipt: Mapping[str, object],
) -> dict[str, Any]:
    """Recompute a packet receipt in Rust and reject changes."""

    result = _native_operation("_zspace_semantic_review_packet_validate", receipt)
    _validate_packet_receipt(result)
    return result


def validate_zspace_semantic_review_packet_receipt_trusted_legacy_replay(
    receipt: Mapping[str, object],
) -> dict[str, Any]:
    """Replay a trusted historical v1 packet receipt without newer budgets.

    Never pass untrusted or remotely supplied input to this opt-in path.
    """

    result = _native_operation(
        "_zspace_semantic_review_packet_validate_trusted_legacy_replay",
        receipt,
    )
    _validate_packet_receipt(result)
    return result


def new_zspace_semantic_review_draft(
    *,
    packet_id: str,
    reviewer_id: str,
    review_session_id: str,
    responses: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Create an unvalidated draft payload for the Rust progress contract."""

    return {
        "schema": ZSPACE_SEMANTIC_REVIEW_DRAFT_SCHEMA,
        "packet_id": packet_id,
        "reviewer_id": reviewer_id,
        "review_session_id": review_session_id,
        "responses": _bounded_mapping_sequence(
            responses,
            maximum=ZSPACE_SEMANTIC_REVIEW_MAX_GROUPS,
            label="semantic review response",
        ),
    }


def summarize_zspace_semantic_review_draft(
    *,
    packet: Mapping[str, object],
    draft: Mapping[str, object],
) -> dict[str, Any]:
    """Validate and canonicalize partial or complete blinded review progress."""

    result = _native_operation(
        "_zspace_semantic_review_draft",
        {"packet": dict(packet), "draft": dict(draft)},
    )
    _validate_draft_receipt(result)
    return result


def validate_zspace_semantic_review_draft_receipt(
    receipt: Mapping[str, object],
) -> dict[str, Any]:
    """Recompute a draft receipt in Rust and reject changes."""

    result = _native_operation("_zspace_semantic_review_draft_validate", receipt)
    _validate_draft_receipt(result)
    return result


def validate_zspace_semantic_review_draft_receipt_trusted_legacy_replay(
    receipt: Mapping[str, object],
) -> dict[str, Any]:
    """Replay a trusted historical v1 draft receipt without newer budgets.

    Never pass untrusted or remotely supplied input to this opt-in path.
    """

    result = _native_operation(
        "_zspace_semantic_review_draft_validate_trusted_legacy_replay",
        receipt,
    )
    _validate_draft_receipt(result)
    return result


def unblind_zspace_semantic_review(
    *,
    packet: Mapping[str, object],
    draft: Mapping[str, object],
    blinding_map: Mapping[str, object],
) -> dict[str, Any]:
    """Unblind a complete response and aggregate arm/seed scores in Rust."""

    result = _native_operation(
        "_zspace_semantic_review_unblind",
        {
            "packet": dict(packet),
            "draft": dict(draft),
            "blinding_map": dict(blinding_map),
        },
    )
    _validate_unblind_report(result)
    return result


def validate_zspace_semantic_review_unblind(
    report: Mapping[str, object],
) -> dict[str, Any]:
    """Recompute an unblind report in Rust and reject changes."""

    result = _native_operation("_zspace_semantic_review_unblind_validate", report)
    _validate_unblind_report(result)
    return result


def validate_zspace_semantic_review_unblind_trusted_legacy_replay(
    report: Mapping[str, object],
) -> dict[str, Any]:
    """Replay a trusted historical v1 unblind report without newer budgets.

    Never pass untrusted or remotely supplied input to this opt-in path.
    """

    result = _native_operation(
        "_zspace_semantic_review_unblind_validate_trusted_legacy_replay",
        report,
    )
    _validate_unblind_report(result)
    return result


def _read_mapping(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return dict(value)


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _emit(value: Mapping[str, object], output: Path | None) -> None:
    if output is not None:
        _write_json_atomic(output, value)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _new_session_id() -> str:
    return "sha256:" + hashlib.sha256(secrets.token_bytes(32)).hexdigest()


def _bounded_score(prompt: str) -> int:
    while True:
        value = input(prompt).strip()
        try:
            score = int(value)
        except ValueError:
            score = 0
        if (
            ZSPACE_SEMANTIC_REVIEW_SCORE_MINIMUM
            <= score
            <= ZSPACE_SEMANTIC_REVIEW_SCORE_MAXIMUM
        ):
            return score
        print(
            f"Enter an integer from {ZSPACE_SEMANTIC_REVIEW_SCORE_MINIMUM} "
            f"through {ZSPACE_SEMANTIC_REVIEW_SCORE_MAXIMUM}."
        )


def _preference() -> str:
    while True:
        value = input("Preference [A/B/C/tie]: ").strip()
        if value in ZSPACE_SEMANTIC_REVIEW_PREFERENCE_VALUES:
            return value
        print("Enter A, B, C, or tie.")


def _terminal_text(value: str) -> str:
    output: list[str] = []
    for character in value:
        if character in {"\n", "\t"} or unicodedata.category(character) not in {
            "Cc",
            "Cf",
        }:
            output.append(character)
        else:
            codepoint = ord(character)
            output.append(
                f"\\u{codepoint:04x}" if codepoint <= 0xFFFF else f"\\U{codepoint:08x}"
            )
    return "".join(output)


def _review_group(group: Mapping[str, object]) -> dict[str, object]:
    group_id = group.get("group_id")
    prompt = group.get("prompt")
    candidates = group.get("candidates")
    if (
        not isinstance(group_id, str)
        or not isinstance(prompt, str)
        or not isinstance(candidates, list)
    ):
        raise RuntimeError(
            "validated packet group has an invalid Python representation"
        )
    print(f"\nGroup {group_id}\nPrompt: {_terminal_text(prompt)}")
    scores: list[dict[str, object]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise RuntimeError("validated packet candidate is not a mapping")
        label = candidate.get("candidate_label")
        continuation = candidate.get("continuation")
        if not isinstance(label, str) or not isinstance(continuation, str):
            raise RuntimeError("validated packet candidate has invalid fields")
        print(f"\n[{label}] {_terminal_text(continuation)}")
        row: dict[str, object] = {"candidate_label": label}
        for dimension in ZSPACE_SEMANTIC_REVIEW_SCORE_DIMENSIONS:
            row[dimension] = _bounded_score(f"{label} {dimension} [1-5]: ")
        scores.append(row)
    return {"group_id": group_id, "scores": scores, "preference": _preference()}


def _review_command(args: argparse.Namespace) -> int:
    packet = _read_mapping(args.packet, "packet")
    packet_receipt = validate_zspace_semantic_review_packet(packet)
    canonical_packet = dict(packet_receipt["packet"])
    if args.draft.exists():
        draft = _read_mapping(args.draft, "draft")
        if (
            args.reviewer_id is not None
            and draft.get("reviewer_id") != args.reviewer_id
        ):
            raise ValueError("--reviewer-id does not match the existing draft")
        if (
            args.review_session_id is not None
            and draft.get("review_session_id") != args.review_session_id
        ):
            raise ValueError("--review-session-id does not match the existing draft")
    else:
        if args.reviewer_id is None:
            raise ValueError("--reviewer-id is required when creating a new draft")
        draft = new_zspace_semantic_review_draft(
            packet_id=str(packet_receipt["packet_id"]),
            reviewer_id=args.reviewer_id,
            review_session_id=args.review_session_id or _new_session_id(),
        )
    receipt = summarize_zspace_semantic_review_draft(
        packet=canonical_packet, draft=draft
    )
    canonical_draft = dict(receipt["request"]["draft"])
    _write_json_atomic(args.draft, canonical_draft)
    groups = {
        str(group["group_id"]): group
        for group in canonical_packet["groups"]
        if isinstance(group, Mapping) and isinstance(group.get("group_id"), str)
    }
    completed_now = 0
    try:
        for group_id in receipt["missing_group_ids"]:
            if args.max_groups is not None and completed_now >= args.max_groups:
                break
            response = _review_group(groups[str(group_id)])
            responses = list(canonical_draft["responses"])
            responses.append(response)
            canonical_draft["responses"] = responses
            receipt = summarize_zspace_semantic_review_draft(
                packet=canonical_packet,
                draft=canonical_draft,
            )
            canonical_draft = dict(receipt["request"]["draft"])
            _write_json_atomic(args.draft, canonical_draft)
            completed_now += 1
            print(
                f"Saved {receipt['completed_group_count']}/{receipt['group_count']} "
                f"groups to {args.draft}"
            )
    except (EOFError, KeyboardInterrupt):
        print("\nReview paused after the last fully validated group.", file=sys.stderr)
    _emit(receipt, args.output)
    return 0 if receipt["human_review_complete"] else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spiral-hf-semantic-review",
        description="Rust-validated blinded semantic review for held-out HF generations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="validate a blinded packet")
    inspect_parser.add_argument("packet", type=Path)
    inspect_parser.add_argument("--output", type=Path)

    status_parser = subparsers.add_parser("status", help="validate draft progress")
    status_parser.add_argument("packet", type=Path)
    status_parser.add_argument("draft", type=Path)
    status_parser.add_argument("--output", type=Path)

    review_parser = subparsers.add_parser(
        "review", help="review or resume blinded groups"
    )
    review_parser.add_argument("packet", type=Path)
    review_parser.add_argument("--draft", type=Path, required=True)
    review_parser.add_argument("--reviewer-id")
    review_parser.add_argument("--review-session-id")
    review_parser.add_argument("--max-groups", type=int)
    review_parser.add_argument("--output", type=Path)

    unblind_parser = subparsers.add_parser(
        "unblind", help="unblind and aggregate a complete response"
    )
    unblind_parser.add_argument("packet", type=Path)
    unblind_parser.add_argument("draft", type=Path)
    unblind_parser.add_argument("blinding_map", type=Path)
    unblind_parser.add_argument("--output", type=Path)

    validate_parser = subparsers.add_parser(
        "validate-report", help="recompute and validate a saved receipt or report"
    )
    validate_parser.add_argument("report", type=Path)
    validate_parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the resumable semantic review command line client."""

    args = _parser().parse_args(argv)
    if args.command == "inspect":
        result = validate_zspace_semantic_review_packet(
            _read_mapping(args.packet, "packet")
        )
        _emit(result, args.output)
        return 0
    if args.command == "status":
        result = summarize_zspace_semantic_review_draft(
            packet=_read_mapping(args.packet, "packet"),
            draft=_read_mapping(args.draft, "draft"),
        )
        _emit(result, args.output)
        return 0 if result["human_review_complete"] else 2
    if args.command == "review":
        if args.max_groups is not None and args.max_groups <= 0:
            raise ValueError("--max-groups must be positive")
        return _review_command(args)
    if args.command == "unblind":
        packet = _read_mapping(args.packet, "packet")
        draft = _read_mapping(args.draft, "draft")
        draft_receipt = summarize_zspace_semantic_review_draft(
            packet=packet,
            draft=draft,
        )
        if draft_receipt["human_review_complete"] is not True:
            raise ValueError(
                "semantic review draft is incomplete; the blinding map was not read"
            )
        result = unblind_zspace_semantic_review(
            packet=packet,
            draft=draft,
            blinding_map=_read_mapping(args.blinding_map, "blinding map"),
        )
        _emit(result, args.output)
        return 0
    report = _read_mapping(args.report, "report")
    kind = report.get("kind")
    validators = {
        ZSPACE_SEMANTIC_REVIEW_PACKET_KIND: validate_zspace_semantic_review_packet_receipt,
        ZSPACE_SEMANTIC_REVIEW_DRAFT_KIND: validate_zspace_semantic_review_draft_receipt,
        ZSPACE_SEMANTIC_REVIEW_UNBLIND_KIND: validate_zspace_semantic_review_unblind,
    }
    validator = validators.get(kind)
    if validator is None:
        raise ValueError(f"unsupported semantic review report kind: {kind!r}")
    result = validator(report)
    _emit(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
