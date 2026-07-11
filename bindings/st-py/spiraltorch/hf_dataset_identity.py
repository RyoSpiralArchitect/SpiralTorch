"""Hugging Face dataset source and materialization contracts for fine-tuning."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "HF_DATASET_INPUT_IDENTITY_SCHEMA",
    "HF_DATASET_MATERIALIZATION_IDENTITY_SCHEMA",
    "HF_TOKENIZED_DATASET_IDENTITY_SCHEMA",
    "hf_dataset_input_identity_lines",
    "hf_dataset_input_identity_report",
    "hf_dataset_materialization_identity_lines",
    "hf_dataset_materialization_identity_report",
    "hf_tokenized_dataset_identity_lines",
    "hf_tokenized_dataset_identity_report",
]


HF_DATASET_INPUT_IDENTITY_SCHEMA = "spiraltorch.hf_dataset_input_identity.v1"
_HF_DATASET_INPUT_BUNDLE_SCHEMA = "spiraltorch.hf_dataset_input_bundle.v1"
HF_DATASET_MATERIALIZATION_IDENTITY_SCHEMA = (
    "spiraltorch.hf_dataset_materialization_identity.v1"
)
_HF_DATASET_MATERIALIZATION_BUNDLE_SCHEMA = (
    "spiraltorch.hf_dataset_materialization_bundle.v1"
)
HF_TOKENIZED_DATASET_IDENTITY_SCHEMA = (
    "spiraltorch.hf_tokenized_dataset_identity.v1"
)
_HF_TOKENIZED_DATASET_BUNDLE_SCHEMA = (
    "spiraltorch.hf_tokenized_dataset_bundle.v1"
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validated_identity_id(value: object | None) -> str | None:
    if value is None:
        return None
    identity_id = str(value).strip()
    digest = identity_id.removeprefix("sha256:")
    if (
        not identity_id.startswith("sha256:")
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(
            "expected_identity_id must be a lowercase sha256:<64 hex> identity id"
        )
    return identity_id


def _immutable_hub_revision(value: object | None) -> str | None:
    if value is None:
        return None
    revision = str(value).strip()
    if len(revision) != 40:
        return None
    if any(character not in "0123456789abcdef" for character in revision):
        return None
    return revision


def _dataset_info_sha(info: Any) -> str | None:
    value = info.get("sha") if isinstance(info, Mapping) else getattr(info, "sha", None)
    return _immutable_hub_revision(value)


def _dataset_info_id(info: Any, fallback: str) -> str:
    value = info.get("id") if isinstance(info, Mapping) else getattr(info, "id", None)
    resolved = str(value).strip() if value is not None else ""
    return resolved or fallback


def _resolve_dataset_revision(
    dataset_name: str,
    requested_revision: object | None,
    *,
    hub_api: Any | None,
) -> tuple[str, str | None, str, str | None]:
    pinned = _immutable_hub_revision(requested_revision)
    if pinned is not None:
        return dataset_name, pinned, "requested_commit", None
    try:
        if hub_api is None:
            from huggingface_hub import HfApi

            hub_api = HfApi()
        info = hub_api.dataset_info(
            dataset_name,
            revision=(
                None
                if requested_revision is None
                else str(requested_revision).strip() or None
            ),
        )
        resolved = _dataset_info_sha(info)
        if resolved is None:
            raise ValueError("dataset metadata did not expose a 40-hex commit SHA")
        return _dataset_info_id(info, dataset_name), resolved, "hub_dataset_info", None
    except Exception as exc:
        return dataset_name, None, "unresolved", f"{exc.__class__.__name__}: {exc}"


def hf_dataset_input_identity_report(
    *,
    dataset_name: object,
    dataset_config: object | None = None,
    requested_revision: object | None = None,
    train_split: object | None = "train",
    eval_split: object | None = None,
    text_column: object | None = "text",
    local_files: bool = False,
    expected_identity_id: str | None = None,
    phase: str = "preflight",
    hub_api: Any | None = None,
) -> dict[str, object]:
    """Resolve and fingerprint the immutable Hub dataset source selection.

    The identity covers the Hub repository commit and the logical config/split/
    text selection. It does not claim to hash rows fetched by external dataset
    builders; run cards expose that boundary explicitly.
    """

    expected_id = _validated_identity_id(expected_identity_id)
    resolved_phase = str(phase).strip()
    if not resolved_phase:
        raise ValueError("phase must not be empty")
    name = str(dataset_name).strip()
    if not name:
        raise ValueError("dataset_name must not be empty")
    config = None if dataset_config is None else str(dataset_config).strip() or None
    requested = (
        None if requested_revision is None else str(requested_revision).strip() or None
    )
    train = None if train_split is None else str(train_split).strip() or None
    evaluation = None if eval_split is None else str(eval_split).strip() or None
    column = None if text_column is None else str(text_column).strip() or None

    if local_files:
        errors = (
            ["expected remote dataset identity cannot verify a local corpus"]
            if expected_id is not None
            else []
        )
        return {
            "row_type": "hf_dataset_input_identity",
            "schema": HF_DATASET_INPUT_IDENTITY_SCHEMA,
            "status": "blocked" if errors else "not_applicable",
            "phase": resolved_phase,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "dataset_source": "local_files",
            "dataset_name": name,
            "effective_dataset_name": None,
            "dataset_config": config,
            "requested_revision": requested,
            "effective_revision": None,
            "revision_resolution_source": "not_applicable",
            "observed_identity_id": None,
            "expected_identity_id": expected_id,
            "expected_identity_verified": False if errors else None,
            "identity_verified": False,
            "path_independent": True,
            "coverage": "delegated_to_local_training_input_identity",
            "materialized_rows_verified": False,
            "identity_payload": None,
            "error_count": len(errors),
            "errors": errors,
        }

    (
        effective_dataset_name,
        effective_revision,
        resolution_source,
        resolution_error,
    ) = _resolve_dataset_revision(
        name,
        requested,
        hub_api=hub_api,
    )
    identity_payload = None
    observed_id = None
    if effective_revision is not None:
        identity_payload = {
            "schema": _HF_DATASET_INPUT_BUNDLE_SCHEMA,
            "dataset_name": effective_dataset_name,
            "dataset_config": config,
            "effective_revision": effective_revision,
            "train_split": train,
            "eval_split": evaluation,
            "text_column": column,
        }
        observed_id = f"sha256:{_sha256_bytes(_canonical_json_bytes(identity_payload))}"

    errors: list[str] = []
    if resolution_error is not None:
        errors.append(f"dataset revision resolution failed: {resolution_error}")
    if expected_id is not None and observed_id != expected_id:
        errors.append("dataset input identity does not match expected identity id")
    if expected_id is not None and errors:
        status = "blocked"
    elif errors:
        status = "evidence_incomplete"
    else:
        status = "ready"
    return {
        "row_type": "hf_dataset_input_identity",
        "schema": HF_DATASET_INPUT_IDENTITY_SCHEMA,
        "status": status,
        "phase": resolved_phase,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "dataset_source": "hf_dataset",
        "dataset_name": name,
        "effective_dataset_name": effective_dataset_name,
        "dataset_config": config,
        "requested_revision": requested,
        "effective_revision": effective_revision,
        "revision_resolution_source": resolution_source,
        "train_split": train,
        "eval_split": evaluation,
        "text_column": column,
        "observed_identity_id": observed_id,
        "expected_identity_id": expected_id,
        "expected_identity_verified": (
            None if expected_id is None else observed_id == expected_id and not errors
        ),
        "identity_verified": status == "ready",
        "path_independent": True,
        "coverage": "hub_repository_revision_and_logical_selection",
        "materialized_rows_verified": False,
        "identity_payload": identity_payload,
        "error_count": len(errors),
        "errors": errors,
    }


def hf_dataset_input_identity_lines(
    report: Mapping[str, object],
) -> list[str]:
    return [
        "hf_dataset_input_identity "
        f"status={report.get('status')} "
        f"phase={report.get('phase')} "
        f"source={report.get('dataset_source')} "
        f"dataset={report.get('dataset_name')} "
        f"effective_dataset={report.get('effective_dataset_name')} "
        f"config={report.get('dataset_config')} "
        f"requested_revision={report.get('requested_revision')} "
        f"effective_revision={report.get('effective_revision')} "
        f"resolution={report.get('revision_resolution_source')} "
        f"verified={report.get('identity_verified')} "
        f"observed={report.get('observed_identity_id')} "
        f"expected={report.get('expected_identity_id')} "
        f"errors={report.get('error_count')}"
    ]


def _materialized_split_identity(
    dataset: Any | None,
    *,
    role: str,
    text_column: str,
) -> dict[str, object]:
    if dataset is None:
        return {
            "role": role,
            "present": False,
            "row_count": 0,
            "utf8_bytes": 0,
            "empty_text_rows": 0,
            "content_sha256": None,
        }
    try:
        row_count = len(dataset)
    except (TypeError, AttributeError) as exc:
        raise TypeError(f"{role} dataset must expose a stable row count") from exc
    if row_count < 0:
        raise ValueError(f"{role} dataset reported a negative row count")

    digest = hashlib.sha256()
    utf8_bytes = 0
    empty_text_rows = 0
    for index in range(row_count):
        row = dataset[index]
        if not isinstance(row, Mapping):
            raise TypeError(f"{role} row {index} must be a mapping")
        if text_column not in row:
            raise KeyError(
                f"{role} row {index} does not contain text column {text_column!r}"
            )
        text = row[text_column]
        if not isinstance(text, str):
            raise TypeError(
                f"{role} row {index} text column {text_column!r} must be str"
            )
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
        utf8_bytes += len(encoded)
        empty_text_rows += int(not text)

    if len(dataset) != row_count:
        raise RuntimeError(f"{role} dataset changed while hashing materialized rows")
    return {
        "role": role,
        "present": True,
        "row_count": row_count,
        "utf8_bytes": utf8_bytes,
        "empty_text_rows": empty_text_rows,
        "content_sha256": digest.hexdigest(),
    }


def hf_dataset_materialization_identity_report(
    *,
    train_dataset: Any,
    eval_dataset: Any | None = None,
    text_column: object = "text",
    expected_identity_id: str | None = None,
    phase: str = "after_load",
) -> dict[str, object]:
    """Fingerprint the exact selected text rows presented for tokenization.

    The report hashes every selected train/eval text value in order without
    retaining corpus text. This closes the provenance gap left by dataset
    builders that fetch mutable content outside the pinned Hub repository.
    """

    expected_id = _validated_identity_id(expected_identity_id)
    resolved_phase = str(phase).strip()
    if not resolved_phase:
        raise ValueError("phase must not be empty")
    column = str(text_column).strip()
    if not column:
        raise ValueError("text_column must not be empty")

    errors: list[str] = []
    splits: list[dict[str, object]] = []
    for role, dataset in (("train", train_dataset), ("eval", eval_dataset)):
        try:
            splits.append(
                _materialized_split_identity(
                    dataset,
                    role=role,
                    text_column=column,
                )
            )
        except (KeyError, IndexError, OSError, RuntimeError, TypeError, ValueError) as exc:
            errors.append(f"{exc.__class__.__name__}: {exc}")
            splits.append(
                {
                    "role": role,
                    "present": dataset is not None,
                    "status": "blocked",
                }
            )

    identity_payload = None
    observed_id = None
    if not errors:
        identity_payload = {
            "schema": _HF_DATASET_MATERIALIZATION_BUNDLE_SCHEMA,
            "text_column": column,
            "splits": splits,
        }
        observed_id = f"sha256:{_sha256_bytes(_canonical_json_bytes(identity_payload))}"
    if expected_id is not None and observed_id != expected_id:
        errors.append("dataset materialization identity does not match expected id")

    status = "blocked" if errors else "ready"
    total_rows = sum(
        int(split.get("row_count") or 0)
        for split in splits
        if split.get("present") is True
    )
    total_utf8_bytes = sum(
        int(split.get("utf8_bytes") or 0)
        for split in splits
        if split.get("present") is True
    )
    return {
        "row_type": "hf_dataset_materialization_identity",
        "schema": HF_DATASET_MATERIALIZATION_IDENTITY_SCHEMA,
        "status": status,
        "phase": resolved_phase,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "text_column": column,
        "observed_identity_id": observed_id,
        "expected_identity_id": expected_id,
        "expected_identity_verified": (
            None if expected_id is None else observed_id == expected_id and not errors
        ),
        "identity_verified": status == "ready",
        "path_independent": True,
        "coverage": "exact_selected_text_rows_in_order",
        "materialized_rows_verified": status == "ready",
        "total_rows": total_rows,
        "total_utf8_bytes": total_utf8_bytes,
        "splits": splits,
        "identity_payload": identity_payload,
        "error_count": len(errors),
        "errors": errors,
    }


def hf_dataset_materialization_identity_lines(
    report: Mapping[str, object],
) -> list[str]:
    return [
        "hf_dataset_materialization_identity "
        f"status={report.get('status')} "
        f"phase={report.get('phase')} "
        f"column={report.get('text_column')} "
        f"rows={report.get('total_rows')} "
        f"utf8_bytes={report.get('total_utf8_bytes')} "
        f"verified={report.get('identity_verified')} "
        f"observed={report.get('observed_identity_id')} "
        f"expected={report.get('expected_identity_id')} "
        f"errors={report.get('error_count')}"
    ]


def _tokenized_value_payload(
    value: object,
    *,
    path: str,
) -> tuple[object, int]:
    if value is None:
        return ["none"], 1
    if isinstance(value, bool):
        return ["bool", value], 1
    if isinstance(value, numbers.Integral):
        return ["int", str(int(value))], 1
    if isinstance(value, numbers.Real):
        resolved = float(value)
        if not math.isfinite(resolved):
            raise ValueError(f"{path} contains a non-finite float")
        return ["float", resolved.hex()], 1
    if isinstance(value, str):
        return ["str", value], 1
    if isinstance(value, (bytes, bytearray, memoryview)):
        return ["bytes", bytes(value).hex()], 1
    if isinstance(value, Mapping):
        rows: list[list[object]] = []
        value_count = 0
        for key in sorted(value, key=lambda item: str(item)):
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key")
            payload, count = _tokenized_value_payload(
                value[key],
                path=f"{path}.{key}",
            )
            rows.append([key, payload])
            value_count += count
        return ["mapping", rows], value_count
    if isinstance(value, (list, tuple)):
        rows = []
        value_count = 0
        for index, item in enumerate(value):
            payload, count = _tokenized_value_payload(
                item,
                path=f"{path}[{index}]",
            )
            rows.append(payload)
            value_count += count
        return ["sequence", rows], value_count

    for method_name in ("tolist", "item"):
        method = getattr(value, method_name, None)
        if not callable(method):
            continue
        converted = method()
        if converted is value:
            continue
        return _tokenized_value_payload(converted, path=path)
    raise TypeError(
        f"{path} has unsupported tokenized value type "
        f"{value.__class__.__name__}"
    )


def _token_sequence_length(value: object, *, path: str) -> int:
    converted = value
    tolist = getattr(converted, "tolist", None)
    if not isinstance(converted, (list, tuple)) and callable(tolist):
        converted = tolist()
    if not isinstance(converted, (list, tuple)):
        raise TypeError(f"{path} must be a token sequence")
    for index, item in enumerate(converted):
        if isinstance(item, bool) or not isinstance(item, numbers.Integral):
            raise TypeError(f"{path}[{index}] must be an integer token id")
    return len(converted)


def _tokenized_split_identity(
    dataset: Any | None,
    *,
    role: str,
) -> dict[str, object]:
    if dataset is None:
        return {
            "role": role,
            "present": False,
            "row_count": 0,
            "columns": [],
            "content_sha256": None,
            "serialized_bytes": 0,
            "value_count": 0,
            "input_token_count": 0,
            "label_token_count": 0,
        }
    try:
        row_count = len(dataset)
    except (TypeError, AttributeError) as exc:
        raise TypeError(
            f"{role} tokenized dataset must expose a stable row count"
        ) from exc
    if row_count < 0:
        raise ValueError(f"{role} tokenized dataset reported a negative row count")

    digest = hashlib.sha256()
    columns: list[str] | None = None
    serialized_bytes = 0
    value_count = 0
    input_token_count = 0
    label_token_count = 0
    for index in range(row_count):
        row = dataset[index]
        if not isinstance(row, Mapping):
            raise TypeError(f"{role} tokenized row {index} must be a mapping")
        row_columns = sorted(str(key) for key in row)
        if any(not isinstance(key, str) for key in row):
            raise TypeError(f"{role} tokenized row {index} has a non-string column")
        if columns is None:
            columns = row_columns
            missing = [name for name in ("input_ids", "labels") if name not in columns]
            if missing:
                raise KeyError(
                    f"{role} tokenized dataset is missing required columns {missing!r}"
                )
        elif row_columns != columns:
            raise ValueError(
                f"{role} tokenized row {index} columns changed from "
                f"{columns!r} to {row_columns!r}"
            )

        row_payload, row_value_count = _tokenized_value_payload(
            row,
            path=f"{role}[{index}]",
        )
        encoded = _canonical_json_bytes(row_payload)
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
        serialized_bytes += len(encoded)
        value_count += row_value_count
        input_token_count += _token_sequence_length(
            row["input_ids"],
            path=f"{role}[{index}].input_ids",
        )
        label_token_count += _token_sequence_length(
            row["labels"],
            path=f"{role}[{index}].labels",
        )

    if len(dataset) != row_count:
        raise RuntimeError(f"{role} tokenized dataset changed while hashing rows")
    return {
        "role": role,
        "present": True,
        "row_count": row_count,
        "columns": columns or [],
        "content_sha256": digest.hexdigest(),
        "serialized_bytes": serialized_bytes,
        "value_count": value_count,
        "input_token_count": input_token_count,
        "label_token_count": label_token_count,
    }


def hf_tokenized_dataset_identity_report(
    *,
    train_dataset: Any,
    eval_dataset: Any | None = None,
    expected_identity_id: str | None = None,
    phase: str = "after_tokenization",
) -> dict[str, object]:
    """Fingerprint every tokenized block and column presented to Trainer."""

    expected_id = _validated_identity_id(expected_identity_id)
    resolved_phase = str(phase).strip()
    if not resolved_phase:
        raise ValueError("phase must not be empty")

    errors: list[str] = []
    splits: list[dict[str, object]] = []
    for role, dataset in (("train", train_dataset), ("eval", eval_dataset)):
        try:
            splits.append(_tokenized_split_identity(dataset, role=role))
        except (
            KeyError,
            IndexError,
            OSError,
            OverflowError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            errors.append(f"{exc.__class__.__name__}: {exc}")
            splits.append(
                {
                    "role": role,
                    "present": dataset is not None,
                    "status": "blocked",
                }
            )

    identity_payload = None
    observed_id = None
    if not errors:
        identity_payload = {
            "schema": _HF_TOKENIZED_DATASET_BUNDLE_SCHEMA,
            "splits": splits,
        }
        observed_id = f"sha256:{_sha256_bytes(_canonical_json_bytes(identity_payload))}"
    if expected_id is not None and observed_id != expected_id:
        errors.append("tokenized dataset identity does not match expected id")

    status = "blocked" if errors else "ready"
    present_splits = [split for split in splits if split.get("present") is True]
    return {
        "row_type": "hf_tokenized_dataset_identity",
        "schema": HF_TOKENIZED_DATASET_IDENTITY_SCHEMA,
        "status": status,
        "phase": resolved_phase,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "observed_identity_id": observed_id,
        "expected_identity_id": expected_id,
        "expected_identity_verified": (
            None if expected_id is None else observed_id == expected_id and not errors
        ),
        "identity_verified": status == "ready",
        "path_independent": True,
        "coverage": "exact_tokenized_rows_all_columns_in_order",
        "tokenized_rows_verified": status == "ready",
        "total_rows": sum(int(split.get("row_count") or 0) for split in present_splits),
        "total_serialized_bytes": sum(
            int(split.get("serialized_bytes") or 0) for split in present_splits
        ),
        "total_values": sum(
            int(split.get("value_count") or 0) for split in present_splits
        ),
        "total_input_tokens": sum(
            int(split.get("input_token_count") or 0) for split in present_splits
        ),
        "total_label_tokens": sum(
            int(split.get("label_token_count") or 0) for split in present_splits
        ),
        "splits": splits,
        "identity_payload": identity_payload,
        "error_count": len(errors),
        "errors": errors,
    }


def hf_tokenized_dataset_identity_lines(
    report: Mapping[str, object],
) -> list[str]:
    return [
        "hf_tokenized_dataset_identity "
        f"status={report.get('status')} "
        f"phase={report.get('phase')} "
        f"rows={report.get('total_rows')} "
        f"input_tokens={report.get('total_input_tokens')} "
        f"values={report.get('total_values')} "
        f"verified={report.get('identity_verified')} "
        f"observed={report.get('observed_identity_id')} "
        f"expected={report.get('expected_identity_id')} "
        f"errors={report.get('error_count')}"
    ]
