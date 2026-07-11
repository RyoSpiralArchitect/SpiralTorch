"""Effective Hugging Face fine-tuning recipe identity contracts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from enum import Enum

__all__ = [
    "HF_FINETUNE_TRAINING_RECIPE_IDENTITY_SCHEMA",
    "hf_finetune_training_recipe_identity_lines",
    "hf_finetune_training_recipe_identity_report",
]


HF_FINETUNE_TRAINING_RECIPE_IDENTITY_SCHEMA = (
    "spiraltorch.hf_finetune_training_recipe_identity.v1"
)
_HF_FINETUNE_TRAINING_RECIPE_BUNDLE_SCHEMA = (
    "spiraltorch.hf_finetune_training_recipe_bundle.v1"
)

# These fields describe weight updates, sample order, precision, or training
# control. Operational paths, logging destinations, and checkpoint retention are
# intentionally absent so a recipe can be replayed in a different directory.
_TRAINING_ARGUMENT_FIELDS = (
    "do_train",
    "do_eval",
    "num_train_epochs",
    "max_steps",
    "learning_rate",
    "weight_decay",
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
    "max_grad_norm",
    "optim",
    "optim_args",
    "optim_target_modules",
    "lr_scheduler_type",
    "lr_scheduler_kwargs",
    "warmup_ratio",
    "warmup_steps",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "auto_find_batch_size",
    "logging_strategy",
    "logging_steps",
    "logging_first_step",
    "seed",
    "data_seed",
    "sampler_seed",
    "full_determinism",
    "dataloader_drop_last",
    "dataloader_num_workers",
    "dataloader_pin_memory",
    "dataloader_persistent_workers",
    "dataloader_prefetch_factor",
    "group_by_length",
    "length_column_name",
    "remove_unused_columns",
    "label_names",
    "label_smoothing_factor",
    "gradient_checkpointing",
    "gradient_checkpointing_kwargs",
    "fp16",
    "bf16",
    "tf32",
    "fp16_full_eval",
    "bf16_full_eval",
    "half_precision_backend",
    "fp16_opt_level",
    "use_cpu",
    "no_cuda",
    "use_mps_device",
    "torch_compile",
    "torch_compile_backend",
    "torch_compile_mode",
    "use_liger_kernel",
    "liger_kernel_config",
    "neftune_noise_alpha",
    "ddp_find_unused_parameters",
    "ddp_bucket_cap_mb",
    "ddp_broadcast_buffers",
    "fsdp",
    "fsdp_config",
    "deepspeed",
    "accelerator_config",
    "parallelism_config",
    "average_tokens_across_devices",
    "ignore_data_skip",
    "restore_callback_states_from_checkpoint",
    "past_index",
    "eval_steps",
    "eval_delay",
    "eval_accumulation_steps",
    "eval_do_concat_batches",
    "eval_on_start",
    "batch_eval_metrics",
    "load_best_model_at_end",
    "metric_for_best_model",
    "greater_is_better",
    "world_size",
    "train_batch_size",
    "eval_batch_size",
    "parallel_mode",
)
_TRAINING_ARGUMENT_ALIASES = {
    "eval_strategy": ("eval_strategy", "evaluation_strategy"),
}
_MODEL_PREPARE_FIELDS = (
    "mode",
    "model_family",
    "adapter_attached",
    "adapter_attached_now",
    "adapter_preloaded",
    "adapter_origin",
    "active_adapter",
    "adapter_config_source",
    "adapter_config_applied",
)
_GRADIENT_CHECKPOINTING_FIELDS = (
    "requested",
    "enabled",
    "enable_input_require_grads",
    "use_cache_before",
    "use_cache_after",
)
_PARAMETER_REPORT_FIELDS = (
    "trainable_parameter_count",
    "frozen_parameter_count",
)
_PARAMETER_REPORT_ALIASES = {
    "parameter_count": ("parameter_count", "total_parameter_count"),
    "trainable_parameter_ratio": (
        "trainable_parameter_ratio",
        "trainable_fraction",
    ),
}
_ADAPTER_CONFIG_EXCLUDED_FIELDS = {
    "auto_mapping",
    "base_model_name_or_path",
    "revision",
    "row_type",
    "target_modules_source",
}
_FULL_FINETUNE_ADAPTER_CONFIG_FIELDS = {
    "enabled",
    "gradient_checkpointing",
    "mode",
    "model_family",
}
_MODEL_DTYPE_FIELDS = (
    "policy",
    "train_requested",
    "dtype_before",
    "dtype_after",
    "cast_status",
)
_RESUME_FIELDS = (
    "trainer_state_present",
    "optimizer_state_present",
    "scheduler_state_present",
    "rng_state_present",
    "adapter_weights_present",
    "full_model_weights_present",
    "exact_state_available",
    "global_step",
    "saved_max_steps",
    "requested_max_steps",
    "extension_requested",
    "scheduler_horizon_exhausted",
    "scheduler_extension_risk",
    "recommendation",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


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


def _canonical_value(value: object, *, path: str) -> object:
    if isinstance(value, Enum):
        return _canonical_value(value.value, path=path)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key in sorted(value, key=lambda item: str(item)):
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key")
            result[key] = _canonical_value(value[key], path=f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _canonical_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, (set, frozenset)):
        items = [_canonical_value(item, path=f"{path}[]") for item in value]
        return sorted(items, key=lambda item: _canonical_json_bytes(item))
    raise TypeError(f"{path} has unsupported value type {type(value).__name__}")


def _mapping(value: object | None, *, path: str) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return value


def _training_arguments_mapping(training_arguments: object) -> Mapping[str, object]:
    if isinstance(training_arguments, Mapping):
        return training_arguments
    to_dict = getattr(training_arguments, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if not isinstance(payload, Mapping):
            raise TypeError("training_arguments.to_dict() must return a mapping")
        values = dict(payload)
        for name in (
            "world_size",
            "train_batch_size",
            "eval_batch_size",
            "parallel_mode",
        ):
            if name not in values and hasattr(training_arguments, name):
                values[name] = getattr(training_arguments, name)
        return values
    values: dict[str, object] = {}
    names = [*_TRAINING_ARGUMENT_FIELDS]
    for aliases in _TRAINING_ARGUMENT_ALIASES.values():
        names.extend(aliases)
    for name in names:
        if hasattr(training_arguments, name):
            values[name] = getattr(training_arguments, name)
    if not values:
        raise TypeError(
            "training_arguments must be a mapping, expose to_dict(), or provide "
            "TrainingArguments attributes"
        )
    return values


def _selected_fields(
    source: Mapping[str, object],
    fields: Sequence[str],
    *,
    path: str,
) -> dict[str, object]:
    return {
        name: _canonical_value(source[name], path=f"{path}.{name}")
        for name in fields
        if name in source
    }


def _training_argument_payload(training_arguments: object) -> dict[str, object]:
    source = _training_arguments_mapping(training_arguments)
    payload = _selected_fields(
        source,
        _TRAINING_ARGUMENT_FIELDS,
        path="training_arguments",
    )
    for canonical_name, aliases in _TRAINING_ARGUMENT_ALIASES.items():
        for alias in aliases:
            if alias in source:
                payload[canonical_name] = _canonical_value(
                    source[alias],
                    path=f"training_arguments.{alias}",
                )
                break
    return dict(sorted(payload.items()))


def _model_prepare_payload(report: Mapping[str, object]) -> dict[str, object]:
    payload = _selected_fields(
        report,
        _MODEL_PREPARE_FIELDS,
        path="model_prepare_report",
    )
    config_source = report.get("adapter_config_source")
    if config_source == "loaded_artifact" and isinstance(
        report.get("runtime_adapter_config"), Mapping
    ):
        effective_config = report["runtime_adapter_config"]
    else:
        effective_config = report.get("adapter_config")
    if isinstance(effective_config, Mapping):
        config_fields = {
            str(key): value
            for key, value in effective_config.items()
            if str(key) not in _ADAPTER_CONFIG_EXCLUDED_FIELDS
        }
        if report.get("mode") == "full":
            config_fields = {
                key: value
                for key, value in config_fields.items()
                if key in _FULL_FINETUNE_ADAPTER_CONFIG_FIELDS
            }
        payload["effective_adapter_config"] = _canonical_value(
            config_fields,
            path="model_prepare_report.effective_adapter_config",
        )
    elif effective_config is not None:
        raise TypeError("model_prepare_report adapter config must be a mapping")

    target_report = report.get("target_report")
    if isinstance(target_report, Mapping) and "target_modules" in target_report:
        payload["effective_target_modules"] = _canonical_value(
            target_report["target_modules"],
            path="model_prepare_report.target_report.target_modules",
        )
    checkpointing = report.get("gradient_checkpointing")
    if isinstance(checkpointing, Mapping):
        payload["gradient_checkpointing"] = _selected_fields(
            checkpointing,
            _GRADIENT_CHECKPOINTING_FIELDS,
            path="model_prepare_report.gradient_checkpointing",
        )
    parameters = report.get("parameter_report_after")
    if isinstance(parameters, Mapping):
        parameter_payload = _selected_fields(
            parameters,
            _PARAMETER_REPORT_FIELDS,
            path="model_prepare_report.parameter_report_after",
        )
        for canonical_name, aliases in _PARAMETER_REPORT_ALIASES.items():
            for alias in aliases:
                if alias in parameters:
                    parameter_payload[canonical_name] = _canonical_value(
                        parameters[alias],
                        path=(f"model_prepare_report.parameter_report_after.{alias}"),
                    )
                    break
        payload["parameters"] = parameter_payload
    return payload


def hf_finetune_training_recipe_identity_report(
    training_arguments: object,
    *,
    model_prepare_report: Mapping[str, object],
    model_dtype_report: Mapping[str, object] | None = None,
    checkpoint_resume_report: Mapping[str, object] | None = None,
    trainer_contract: Mapping[str, object] | None = None,
    expected_identity_id: str | None = None,
    phase: str = "before_trainer_init",
) -> dict[str, object]:
    """Fingerprint the effective optimization recipe before Trainer is built.

    Model/data/runtime artifacts have separate identities. This contract covers
    the effective update schedule and trainable-model preparation while omitting
    paths and observability-only destinations, so moving a replay does not alter
    its recipe id.
    """

    expected_id = _validated_identity_id(expected_identity_id)
    resolved_phase = str(phase).strip()
    if not resolved_phase:
        raise ValueError("phase must not be empty")

    errors: list[str] = []
    identity_payload = None
    observed_id = None
    arguments_payload: dict[str, object] = {}
    try:
        arguments_payload = _training_argument_payload(training_arguments)
        if not arguments_payload:
            raise ValueError("no supported effective TrainingArguments fields found")
        model_payload = _model_prepare_payload(
            _mapping(model_prepare_report, path="model_prepare_report")
        )
        if model_payload.get("mode") not in {"full", "lora"}:
            raise ValueError(
                "model_prepare_report.mode must resolve to full or lora"
            )
        identity_payload = {
            "schema": _HF_FINETUNE_TRAINING_RECIPE_BUNDLE_SCHEMA,
            "training_arguments": arguments_payload,
            "model_preparation": model_payload,
            "model_dtype": _selected_fields(
                _mapping(model_dtype_report, path="model_dtype_report"),
                _MODEL_DTYPE_FIELDS,
                path="model_dtype_report",
            ),
            "checkpoint_resume": _selected_fields(
                _mapping(
                    checkpoint_resume_report,
                    path="checkpoint_resume_report",
                ),
                _RESUME_FIELDS,
                path="checkpoint_resume_report",
            ),
            "trainer_contract": _canonical_value(
                _mapping(trainer_contract, path="trainer_contract"),
                path="trainer_contract",
            ),
        }
        observed_id = (
            "sha256:"
            + hashlib.sha256(_canonical_json_bytes(identity_payload)).hexdigest()
        )
    except (TypeError, ValueError) as exc:
        errors.append(f"{exc.__class__.__name__}: {exc}")

    if expected_id is not None and observed_id != expected_id:
        errors.append("training recipe identity does not match expected id")
    status = "blocked" if errors else "ready"
    return {
        "row_type": "hf_finetune_training_recipe_identity",
        "schema": HF_FINETUNE_TRAINING_RECIPE_IDENTITY_SCHEMA,
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
            "effective_training_arguments_model_preparation_dtype_resume_and_"
            "trainer_control"
        ),
        "training_argument_count": len(arguments_payload),
        "training_argument_names": sorted(arguments_payload),
        "identity_payload": identity_payload,
        "error_count": len(errors),
        "errors": errors,
    }


def hf_finetune_training_recipe_identity_lines(
    report: Mapping[str, object],
) -> list[str]:
    """Render a compact effective training-recipe identity line."""

    return [
        "hf_finetune_training_recipe_identity "
        f"status={report.get('status')} "
        f"phase={report.get('phase')} "
        f"arguments={report.get('training_argument_count')} "
        f"verified={report.get('identity_verified')} "
        f"observed={report.get('observed_identity_id')} "
        f"expected={report.get('expected_identity_id')} "
        f"errors={report.get('error_count')}"
    ]
