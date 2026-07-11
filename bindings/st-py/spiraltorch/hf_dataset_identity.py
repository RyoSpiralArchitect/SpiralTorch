"""Immutable Hugging Face dataset source contracts for fine-tuning."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "HF_DATASET_INPUT_IDENTITY_SCHEMA",
    "hf_dataset_input_identity_lines",
    "hf_dataset_input_identity_report",
]


HF_DATASET_INPUT_IDENTITY_SCHEMA = "spiraltorch.hf_dataset_input_identity.v1"
_HF_DATASET_INPUT_BUNDLE_SCHEMA = "spiraltorch.hf_dataset_input_bundle.v1"


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
