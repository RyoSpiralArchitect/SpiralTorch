"""Composite replay identity for one effective Hugging Face fine-tune run."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

__all__ = [
    "HF_FINETUNE_REPLAY_IDENTITY_SCHEMA",
    "hf_finetune_replay_identity_lines",
    "hf_finetune_replay_identity_report",
]


HF_FINETUNE_REPLAY_IDENTITY_SCHEMA = "spiraltorch.hf_finetune_replay_identity.v1"
_HF_FINETUNE_REPLAY_BUNDLE_SCHEMA = "spiraltorch.hf_finetune_replay_bundle.v1"

_REQUIRED_COMPONENTS = (
    "dataset_materialization",
    "tokenized_dataset",
    "model_runtime",
    "execution",
    "training_recipe",
)
_COMPONENT_ID_FIELDS = {
    "adapter_input": "observed_adapter_id",
    "training_input": "observed_input_id",
    "dataset_input": "observed_identity_id",
    "dataset_materialization": "observed_identity_id",
    "tokenized_dataset": "observed_identity_id",
    "model_runtime": "observed_identity_id",
    "execution": "observed_identity_id",
    "training_recipe": "observed_identity_id",
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _validated_identity_id(name: str, value: object | None) -> str | None:
    if value is None:
        return None
    identity_id = str(value).strip()
    digest = identity_id.removeprefix("sha256:")
    if (
        not identity_id.startswith("sha256:")
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{name} must be a lowercase sha256:<64 hex> identity id")
    return identity_id


def _component(
    role: str,
    report: Mapping[str, object] | None,
    *,
    required: bool,
    allow_not_applicable: bool = False,
) -> tuple[dict[str, object], list[str]]:
    if report is None:
        status = "missing" if required else "not_applicable"
        errors = [f"{role}: identity report is required"] if required else []
        return {
            "role": role,
            "status": status,
            "applicable": False,
            "schema": None,
            "identity_id": None,
        }, errors

    status = str(report.get("status") or "missing")
    schema = str(report.get("schema") or "").strip() or None
    if status == "not_applicable" and allow_not_applicable:
        return {
            "role": role,
            "status": status,
            "applicable": False,
            "schema": schema,
            "identity_id": None,
        }, []

    errors: list[str] = []
    if status != "ready":
        errors.append(f"{role}: identity status is {status}, expected ready")
    if schema is None:
        errors.append(f"{role}: identity schema is missing")
    identity_id = None
    raw_identity_id = report.get(_COMPONENT_ID_FIELDS[role])
    try:
        identity_id = _validated_identity_id(f"{role} identity id", raw_identity_id)
    except ValueError as exc:
        errors.append(str(exc))
    if identity_id is None:
        errors.append(f"{role}: observed identity id is missing")
    if report.get("identity_verified") is not True:
        errors.append(f"{role}: identity is not verified")

    component: dict[str, object] = {
        "role": role,
        "status": status,
        "applicable": True,
        "schema": schema,
        "identity_id": identity_id,
    }
    if role == "adapter_input":
        root_adapter_id = report.get("observed_root_adapter_id")
        try:
            root_adapter_id = _validated_identity_id(
                "adapter_input root adapter id",
                root_adapter_id,
            )
        except ValueError as exc:
            errors.append(str(exc))
            root_adapter_id = None
        lineage_depth = report.get("observed_lineage_depth")
        if lineage_depth is not None and (
            isinstance(lineage_depth, bool)
            or not isinstance(lineage_depth, int)
            or lineage_depth < 0
        ):
            errors.append("adapter_input lineage depth must be a non-negative integer")
            lineage_depth = None
        component["lineage"] = {
            "depth": lineage_depth,
            "root_adapter_id": root_adapter_id,
        }
    return component, errors


def hf_finetune_replay_identity_report(
    *,
    adapter_input_identity: Mapping[str, object] | None = None,
    adapter_input_required: bool = False,
    training_input_identity: Mapping[str, object] | None,
    dataset_input_identity: Mapping[str, object] | None,
    dataset_materialization_identity: Mapping[str, object] | None,
    tokenized_dataset_identity: Mapping[str, object] | None,
    model_runtime_identity: Mapping[str, object] | None,
    execution_identity: Mapping[str, object] | None,
    training_recipe_identity: Mapping[str, object] | None,
    expected_identity_id: str | None = None,
    phase: str = "before_trainer_init",
) -> dict[str, object]:
    """Bind every verified FT identity layer into one path-independent replay ID."""

    expected_id = _validated_identity_id("expected_identity_id", expected_identity_id)
    resolved_phase = str(phase).strip()
    if not resolved_phase:
        raise ValueError("phase must not be empty")

    component_inputs = (
        (
            "adapter_input",
            adapter_input_identity,
            bool(adapter_input_required),
            False,
        ),
        ("training_input", training_input_identity, False, True),
        ("dataset_input", dataset_input_identity, False, True),
        (
            "dataset_materialization",
            dataset_materialization_identity,
            True,
            False,
        ),
        ("tokenized_dataset", tokenized_dataset_identity, True, False),
        ("model_runtime", model_runtime_identity, True, False),
        ("execution", execution_identity, True, False),
        ("training_recipe", training_recipe_identity, True, False),
    )
    components: dict[str, dict[str, object]] = {}
    errors: list[str] = []
    for role, report, required, allow_not_applicable in component_inputs:
        component, component_errors = _component(
            role,
            report,
            required=required,
            allow_not_applicable=allow_not_applicable,
        )
        components[role] = component
        errors.extend(component_errors)

    if not any(
        components[role].get("status") == "ready"
        for role in ("training_input", "dataset_input")
    ):
        errors.append(
            "training_input or dataset_input must provide a ready source identity"
        )

    identity_payload = None
    observed_id = None
    if not errors:
        identity_payload = {
            "schema": _HF_FINETUNE_REPLAY_BUNDLE_SCHEMA,
            "components": components,
        }
        observed_id = (
            "sha256:"
            + hashlib.sha256(_canonical_json_bytes(identity_payload)).hexdigest()
        )
    if expected_id is not None and observed_id != expected_id:
        errors.append("fine-tune replay identity does not match expected identity id")

    status = "blocked" if errors else "ready"
    return {
        "row_type": "hf_finetune_replay_identity",
        "schema": HF_FINETUNE_REPLAY_IDENTITY_SCHEMA,
        "status": status,
        "phase": resolved_phase,
        "observed_identity_id": observed_id,
        "expected_identity_id": expected_id,
        "expected_identity_verified": (
            None if expected_id is None else observed_id == expected_id and not errors
        ),
        "identity_verified": status == "ready",
        "path_independent": True,
        "coverage": (
            "adapter_local_or_remote_data_materialized_and_tokenized_rows_model_"
            "runtime_execution_and_effective_training_recipe"
        ),
        "adapter_input_required": bool(adapter_input_required),
        "required_components": list(_REQUIRED_COMPONENTS),
        "component_count": len(components),
        "applicable_component_count": sum(
            component.get("applicable") is True for component in components.values()
        ),
        "ready_component_count": sum(
            component.get("status") == "ready" for component in components.values()
        ),
        "components": components,
        "identity_payload": identity_payload,
        "error_count": len(errors),
        "errors": errors,
    }


def hf_finetune_replay_identity_lines(
    report: Mapping[str, object],
) -> list[str]:
    """Render a compact composite replay identity line."""

    return [
        "hf_finetune_replay_identity "
        f"status={report.get('status')} "
        f"phase={report.get('phase')} "
        f"ready={report.get('ready_component_count')}/"
        f"{report.get('applicable_component_count')} "
        f"applicable={report.get('applicable_component_count')}/"
        f"{report.get('component_count')} "
        f"verified={report.get('identity_verified')} "
        f"observed={report.get('observed_identity_id')} "
        f"expected={report.get('expected_identity_id')} "
        f"errors={report.get('error_count')}"
    ]
