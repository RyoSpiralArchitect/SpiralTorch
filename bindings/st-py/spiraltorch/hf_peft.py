"""Reusable PEFT/LoRA preparation for Hugging Face fine-tuning."""

from __future__ import annotations

import importlib
import inspect
import json
import math
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .hf_runtime_identity import hf_causal_lm_runtime_identity_report

__all__ = [
    "HF_CAUSAL_LM_ARTIFACT_KINDS",
    "HF_FINETUNE_LORA_TARGET_MODULES",
    "HF_FINETUNE_MODES",
    "HfCausalLmRuntimeIdentityError",
    "export_hf_merged_causal_lm",
    "hf_causal_lm_artifact_lines",
    "hf_causal_lm_artifact_report",
    "hf_finetune_adapter_config",
    "hf_finetune_lora_target_report",
    "hf_finetune_parameter_report",
    "hf_merged_causal_lm_export_lines",
    "load_hf_causal_lm_artifact",
    "prepare_hf_finetune_model",
    "summarize_hf_causal_lm_artifact",
]


HF_CAUSAL_LM_ARTIFACT_KINDS = ("auto", "full_model", "peft_adapter")
HF_FINETUNE_MODES = ("full", "lora")

_HF_ADAPTER_CONFIG_FILENAME = "adapter_config.json"
_HF_ADAPTER_WEIGHT_FILENAMES = (
    "adapter_model.safetensors",
    "adapter_model.safetensors.index.json",
    "adapter_model.bin",
)
_HF_FULL_MODEL_WEIGHT_FILENAMES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)
_HF_TOKENIZER_FILENAMES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
)

# PEFT matches these values against module-name suffixes. Keep the defaults
# conservative: attention projections are portable and avoid adapting every
# linear layer merely because a model family is new to SpiralTorch.
HF_FINETUNE_LORA_TARGET_MODULES: dict[str, tuple[str, ...]] = {
    "bloom": ("query_key_value", "dense"),
    "gpt2": ("c_attn", "c_proj"),
    "gpt_bigcode": ("c_attn", "c_proj"),
    "gpt_j": ("q_proj", "v_proj"),
    "gpt_neo": ("q_proj", "v_proj"),
    "gpt_neox": ("query_key_value", "dense"),
    "llama": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mistral": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mixtral": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "opt": ("q_proj", "k_proj", "v_proj", "out_proj"),
    "phi": ("q_proj", "k_proj", "v_proj", "dense"),
    "phi3": ("qkv_proj", "o_proj"),
    "qwen2": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "qwen3": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "smollm2": ("q_proj", "k_proj", "v_proj", "o_proj"),
}

_MODEL_FAMILY_ALIASES = {
    "gpt-neox": "gpt_neox",
    "gptneox": "gpt_neox",
    "gpt-j": "gpt_j",
    "gptj": "gpt_j",
    "gpt-neo": "gpt_neo",
    "gptneo": "gpt_neo",
    "qwen": "qwen2",
    "smollm": "smollm2",
}


class HfCausalLmRuntimeIdentityError(ValueError):
    """Raised before training when the loaded model basis changed identity."""

    def __init__(self, report: Mapping[str, object]) -> None:
        self.report = dict(report)
        errors = "; ".join(str(item) for item in report.get("errors", []))
        super().__init__(errors or "causal-LM runtime identity verification failed")


def _normalise_model_family(value: object) -> str | None:
    if value is None:
        return None
    family = str(value).strip().lower().replace(" ", "_")
    if not family:
        return None
    return _MODEL_FAMILY_ALIASES.get(family, family)


def _hub_commit_hash(value: object | None) -> str | None:
    if value is None:
        return None
    commit = str(value).strip().lower()
    if 7 <= len(commit) <= 64 and all(
        character in "0123456789abcdef" for character in commit
    ):
        return commit
    return None


def _string_values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Iterable[object] = value.split(",")
    elif isinstance(value, Iterable):
        values = value
    else:
        values = [value]
    return list(
        dict.fromkeys(str(item).strip() for item in values if str(item).strip())
    )


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if resolved <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return resolved


def _finite_number(value: object, *, label: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(resolved):
        raise ValueError(f"{label} must be finite")
    return resolved


def _normalise_mode(value: object) -> str:
    mode = str(value or "full").strip().lower().replace("_", "-")
    aliases = {
        "adapter": "lora",
        "full-finetune": "full",
        "full-fine-tune": "full",
        "peft": "lora",
    }
    mode = aliases.get(mode, mode)
    if mode not in HF_FINETUNE_MODES:
        raise ValueError(
            f"unknown fine-tune mode {value!r}; choices={','.join(HF_FINETUNE_MODES)}"
        )
    return mode


def _normalise_artifact_kind(value: object) -> str:
    kind = str(value or "auto").strip().lower().replace("-", "_")
    aliases = {
        "adapter": "peft_adapter",
        "full": "full_model",
        "lora": "peft_adapter",
        "model": "full_model",
        "peft": "peft_adapter",
    }
    kind = aliases.get(kind, kind)
    if kind not in HF_CAUSAL_LM_ARTIFACT_KINDS:
        choices = ",".join(HF_CAUSAL_LM_ARTIFACT_KINDS)
        raise ValueError(f"unknown HF artifact kind {value!r}; choices={choices}")
    return kind


def _artifact_local_directory(source: str) -> Path | None:
    path = Path(source).expanduser()
    return path if path.is_dir() else None


def _read_json_mapping(path: Path) -> tuple[dict[str, object] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"{exc.__class__.__name__}: {exc}"
    if not isinstance(payload, Mapping):
        return None, "adapter config must contain a JSON object"
    return dict(payload), None


def _existing_artifact_files(path: Path | None, names: Iterable[str]) -> list[str]:
    if path is None:
        return []
    return [name for name in names if (path / name).is_file()]


def hf_causal_lm_artifact_report(
    model_name_or_path: str | Path,
    *,
    artifact_kind: str = "auto",
    tokenizer_name_or_path: str | Path | None = None,
) -> dict[str, object]:
    """Describe a full Transformers model or a PEFT adapter artifact.

    Local adapter directories are detected without importing Transformers or
    PEFT. Remote adapter ids can be declared with ``artifact_kind="peft_adapter"``
    and are resolved lazily by :func:`load_hf_causal_lm_artifact`.
    """

    source = str(model_name_or_path).strip()
    if not source:
        raise ValueError("model_name_or_path must not be empty")
    requested_kind = _normalise_artifact_kind(artifact_kind)
    local_directory = _artifact_local_directory(source)
    adapter_config_path = (
        None
        if local_directory is None
        else local_directory / _HF_ADAPTER_CONFIG_FILENAME
    )
    adapter_config_present = bool(
        adapter_config_path is not None and adapter_config_path.is_file()
    )
    resolved_kind = (
        "peft_adapter"
        if requested_kind == "auto" and adapter_config_present
        else "full_model" if requested_kind == "auto" else requested_kind
    )
    adapter_config: dict[str, object] | None = None
    adapter_config_error: str | None = None
    if adapter_config_present and adapter_config_path is not None:
        adapter_config, adapter_config_error = _read_json_mapping(adapter_config_path)

    adapter_weights = _existing_artifact_files(
        local_directory,
        _HF_ADAPTER_WEIGHT_FILENAMES,
    )
    full_model_weights = _existing_artifact_files(
        local_directory,
        _HF_FULL_MODEL_WEIGHT_FILENAMES,
    )
    tokenizer_files = _existing_artifact_files(
        local_directory,
        _HF_TOKENIZER_FILENAMES,
    )
    base_model_name_or_path = source
    base_model_revision = None
    errors: list[str] = []
    runtime_resolution_required = False
    if resolved_kind == "peft_adapter":
        base_model_name_or_path = ""
        if local_directory is not None and not adapter_config_present:
            errors.append(
                f"local PEFT adapter is missing {_HF_ADAPTER_CONFIG_FILENAME}"
            )
        if adapter_config_error is not None:
            errors.append(f"invalid adapter config: {adapter_config_error}")
        if adapter_config is not None:
            base_model_name_or_path = str(
                adapter_config.get("base_model_name_or_path") or ""
            ).strip()
            base_model_revision = adapter_config.get("revision")
            if not base_model_name_or_path:
                errors.append("adapter config is missing base_model_name_or_path")
        elif local_directory is None:
            runtime_resolution_required = True
        if local_directory is not None and not adapter_weights:
            errors.append("local PEFT adapter has no adapter weight artifact")

    explicit_tokenizer = (
        None
        if tokenizer_name_or_path is None
        else str(tokenizer_name_or_path).strip() or None
    )
    if explicit_tokenizer is not None:
        tokenizer_source = explicit_tokenizer
        tokenizer_source_kind = "explicit"
    elif resolved_kind == "peft_adapter" and tokenizer_files:
        tokenizer_source = source
        tokenizer_source_kind = "adapter_artifact"
    elif base_model_name_or_path:
        tokenizer_source = base_model_name_or_path
        tokenizer_source_kind = "base_model"
    else:
        tokenizer_source = None
        tokenizer_source_kind = "runtime_resolution"

    return {
        "row_type": "hf_causal_lm_artifact_report",
        "status": "invalid" if errors else "ready",
        "requested_artifact_kind": requested_kind,
        "artifact_kind": resolved_kind,
        "artifact_source": source,
        "artifact_is_local": local_directory is not None,
        "artifact_local_path": (
            None if local_directory is None else str(local_directory)
        ),
        "adapter_config_present": adapter_config_present,
        "adapter_config_path": (
            None if adapter_config_path is None else str(adapter_config_path)
        ),
        "adapter_config": adapter_config,
        "adapter_config_error": adapter_config_error,
        "adapter_weight_files": adapter_weights,
        "adapter_weights_present": (
            None if local_directory is None else bool(adapter_weights)
        ),
        "full_model_weight_files": full_model_weights,
        "full_model_weights_present": (
            None if local_directory is None else bool(full_model_weights)
        ),
        "tokenizer_files": tokenizer_files,
        "tokenizer_files_present": (
            None if local_directory is None else bool(tokenizer_files)
        ),
        "base_model_name_or_path": base_model_name_or_path or None,
        "base_model_revision": base_model_revision,
        "tokenizer_source": tokenizer_source,
        "tokenizer_source_kind": tokenizer_source_kind,
        "requires_peft": resolved_kind == "peft_adapter",
        "runtime_resolution_required": runtime_resolution_required,
        "errors": errors,
    }


def hf_causal_lm_artifact_lines(
    report_or_source: Mapping[str, object] | str | Path,
    *,
    artifact_kind: str = "auto",
    tokenizer_name_or_path: str | Path | None = None,
) -> list[str]:
    """Render a compact artifact audit line."""

    report = (
        dict(report_or_source)
        if isinstance(report_or_source, Mapping)
        else hf_causal_lm_artifact_report(
            report_or_source,
            artifact_kind=artifact_kind,
            tokenizer_name_or_path=tokenizer_name_or_path,
        )
    )
    return [
        "hf_causal_lm_artifact "
        f"status={report.get('status')} "
        f"kind={report.get('artifact_kind')} "
        f"source={report.get('artifact_source')} "
        f"base={report.get('base_model_name_or_path')} "
        f"tokenizer={report.get('tokenizer_source')} "
        f"local={report.get('artifact_is_local')} "
        f"adapter_weights={report.get('adapter_weights_present')} "
        f"runtime_resolution={report.get('runtime_resolution_required')}"
    ]


def _model_family(model: Any, requested: object) -> str | None:
    explicit = _normalise_model_family(requested)
    if explicit is not None:
        return explicit
    config = getattr(model, "config", None)
    return _normalise_model_family(getattr(config, "model_type", None))


def _named_module_names(model: Any) -> list[str]:
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return []
    try:
        return [str(name) for name, _module in named_modules() if str(name)]
    except Exception:
        return []


def _module_name_matches(name: str, target: str) -> bool:
    return target == "all-linear" or name == target or name.endswith(f".{target}")


def hf_finetune_lora_target_report(
    model: Any,
    *,
    model_family: str | None = None,
    target_modules: object = None,
) -> dict[str, object]:
    """Resolve and validate LoRA module suffixes for one causal LM."""

    family = _model_family(model, model_family)
    requested = _string_values(target_modules)
    source = "explicit" if requested else "model_family_default"
    candidates = requested or list(
        HF_FINETUNE_LORA_TARGET_MODULES.get(family or "", ())
    )
    module_names = _named_module_names(model)

    if not candidates and module_names:
        source = "module_discovery"
        discovered: list[str] = []
        for defaults in HF_FINETUNE_LORA_TARGET_MODULES.values():
            for target in defaults:
                if target not in discovered and any(
                    _module_name_matches(name, target) for name in module_names
                ):
                    discovered.append(target)
        candidates = discovered

    if not candidates:
        raise ValueError(
            "could not resolve LoRA target modules; set model_family or provide "
            "explicit target_modules"
        )

    matched_names = [
        name
        for name in module_names
        if any(_module_name_matches(name, target) for target in candidates)
    ]
    if module_names and not matched_names:
        available_suffixes = sorted({name.rsplit(".", 1)[-1] for name in module_names})
        preview = ",".join(available_suffixes[:24]) or "none"
        raise ValueError(
            "none of the LoRA target modules matched the model: "
            f"targets={','.join(candidates)} available_suffixes={preview}"
        )

    unmatched_targets = [
        target
        for target in candidates
        if target != "all-linear"
        and module_names
        and not any(_module_name_matches(name, target) for name in module_names)
    ]
    if requested and unmatched_targets:
        raise ValueError(
            "explicit LoRA target modules did not all match the model: "
            f"unmatched={','.join(unmatched_targets)}"
        )

    verified_targets = [
        target
        for target in candidates
        if target == "all-linear"
        or not module_names
        or any(_module_name_matches(name, target) for name in module_names)
    ]
    return {
        "row_type": "hf_finetune_lora_target_report",
        "status": "ready",
        "model_family": family,
        "source": source,
        "requested_target_modules": requested,
        "target_modules": verified_targets,
        "target_module_count": len(verified_targets),
        "model_module_count": len(module_names),
        "matched_module_count": len(matched_names),
        "matched_module_names": matched_names[:64],
        "matched_module_names_truncated": len(matched_names) > 64,
        "unmatched_target_modules": unmatched_targets,
        "targets_verified": bool(module_names),
    }


def hf_finetune_adapter_config(
    *,
    mode: str = "full",
    model_family: str | None = None,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
    bias: str = "none",
    target_modules: object = None,
    modules_to_save: object = None,
    use_rslora: bool = False,
    gradient_checkpointing: bool = False,
) -> dict[str, object]:
    """Return a validated, JSON-safe full-FT or LoRA runtime contract."""

    resolved_mode = _normalise_mode(mode)
    resolved_rank = _positive_int(rank, label="rank")
    resolved_alpha = _finite_number(alpha, label="alpha")
    if resolved_alpha <= 0.0:
        raise ValueError("alpha must be positive")
    resolved_dropout = _finite_number(dropout, label="dropout")
    if resolved_dropout < 0.0 or resolved_dropout >= 1.0:
        raise ValueError("dropout must be in [0.0, 1.0)")
    resolved_bias = str(bias).strip().lower()
    if resolved_bias not in {"none", "all", "lora_only"}:
        raise ValueError("bias must be one of none,all,lora_only")
    family = _normalise_model_family(model_family)
    explicit_targets = _string_values(target_modules)
    default_targets = list(HF_FINETUNE_LORA_TARGET_MODULES.get(family or "", ()))
    return {
        "row_type": "hf_finetune_adapter_config",
        "mode": resolved_mode,
        "enabled": resolved_mode == "lora",
        "model_family": family,
        "rank": resolved_rank,
        "alpha": resolved_alpha,
        "dropout": resolved_dropout,
        "bias": resolved_bias,
        "target_modules": explicit_targets or default_targets,
        "target_modules_source": (
            "explicit"
            if explicit_targets
            else "model_family_default" if default_targets else "model_discovery"
        ),
        "modules_to_save": _string_values(modules_to_save),
        "use_rslora": bool(use_rslora),
        "gradient_checkpointing": bool(gradient_checkpointing),
    }


def _parameter_numel(parameter: Any) -> int:
    numel = getattr(parameter, "numel", None)
    if callable(numel):
        try:
            return max(0, int(numel()))
        except Exception:
            return 0
    return 0


def hf_finetune_parameter_report(model: Any) -> dict[str, object]:
    """Count total and trainable parameters without materialising tensors."""

    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        rows: list[tuple[str, Any]] = []
    else:
        try:
            rows = [(str(name), parameter) for name, parameter in named_parameters()]
        except Exception:
            rows = []
    total = sum(_parameter_numel(parameter) for _name, parameter in rows)
    trainable = sum(
        _parameter_numel(parameter)
        for _name, parameter in rows
        if bool(getattr(parameter, "requires_grad", False))
    )
    ratio = None if total <= 0 else trainable / total
    trainable_names = [
        name
        for name, parameter in rows
        if bool(getattr(parameter, "requires_grad", False))
    ]
    return {
        "row_type": "hf_finetune_parameter_report",
        "parameter_count": total,
        "trainable_parameter_count": trainable,
        "frozen_parameter_count": max(0, total - trainable),
        "trainable_parameter_ratio": ratio,
        "named_parameter_count": len(rows),
        "trainable_parameter_tensor_count": len(trainable_names),
        "trainable_parameter_names": trainable_names[:64],
        "trainable_parameter_names_truncated": len(trainable_names) > 64,
    }


def _loader_options(
    common: Mapping[str, object] | None,
    specific: Mapping[str, object] | None,
) -> dict[str, object]:
    options = dict(common or {})
    options.update(dict(specific or {}))
    return options


def _peft_loader_options(options: Mapping[str, object]) -> dict[str, object]:
    accepted = {
        "cache_dir",
        "force_download",
        "local_files_only",
        "proxies",
        "revision",
        "subfolder",
        "token",
    }
    return {key: value for key, value in options.items() if key in accepted}


def _runtime_adapter_config(peft_config: Any) -> dict[str, object]:
    to_dict = getattr(peft_config, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
        except Exception:
            payload = None
        if isinstance(payload, Mapping):
            return json.loads(json.dumps(dict(payload), default=str))
    payload = getattr(peft_config, "__dict__", None)
    if isinstance(payload, Mapping):
        return {
            str(key): value
            for key, value in payload.items()
            if isinstance(value, (bool, float, int, str, type(None)))
        }
    return {}


def _model_runtime_adapter_config(
    model: Any,
) -> tuple[str | None, dict[str, object] | None]:
    peft_configs = getattr(model, "peft_config", None)
    if isinstance(peft_configs, Mapping):
        if not peft_configs:
            return None, None
        active = getattr(model, "active_adapter", None)
        if callable(active):
            try:
                active = active()
            except Exception:
                active = None
        if isinstance(active, (list, tuple)):
            active = active[0] if active else None
        if active not in peft_configs:
            active = next(iter(peft_configs))
        config = peft_configs[active]
        return str(active), _runtime_adapter_config(config)
    if peft_configs is None:
        return None, None
    return None, _runtime_adapter_config(peft_configs)


def _merge_and_unload_adapter(model: Any, *, safe_merge: bool) -> Any:
    merge = getattr(model, "merge_and_unload", None)
    if not callable(merge):
        raise ValueError("loaded PEFT model does not support merge_and_unload")
    try:
        parameters = inspect.signature(merge).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "safe_merge" in parameters:
        return merge(safe_merge=bool(safe_merge))
    return merge()


def load_hf_causal_lm_artifact(
    model_name_or_path: str | Path,
    *,
    tokenizer_name_or_path: str | Path | None = None,
    artifact_kind: str = "auto",
    load_model: bool = True,
    load_tokenizer: bool = True,
    is_trainable: bool = False,
    merge_adapter: bool = False,
    safe_merge: bool = True,
    transformers_module: Any = None,
    peft_module: Any = None,
    loader_kwargs: Mapping[str, object] | None = None,
    config_kwargs: Mapping[str, object] | None = None,
    tokenizer_kwargs: Mapping[str, object] | None = None,
    model_kwargs: Mapping[str, object] | None = None,
    adapter_kwargs: Mapping[str, object] | None = None,
    expected_runtime_identity_id: str | None = None,
) -> tuple[Any, Any, Any, dict[str, object]]:
    """Load a full causal LM or reconstruct a PEFT adapter over its base model.

    The function returns ``(model, tokenizer, config, report)``. ``model`` is
    ``None`` when ``load_model`` is false. Optional libraries are imported only
    when the selected artifact requires them.
    """

    if merge_adapter and not load_model:
        raise ValueError("merge_adapter requires load_model=True")
    if merge_adapter and is_trainable:
        raise ValueError("a trainable adapter cannot be merged during load")
    artifact_report = hf_causal_lm_artifact_report(
        model_name_or_path,
        artifact_kind=artifact_kind,
        tokenizer_name_or_path=tokenizer_name_or_path,
    )
    if artifact_report["status"] != "ready":
        errors = "; ".join(str(item) for item in artifact_report["errors"])
        raise ValueError(f"invalid HF causal-LM artifact: {errors}")

    transformers = transformers_module or importlib.import_module("transformers")
    common_options = dict(loader_kwargs or {})
    source = str(artifact_report["artifact_source"])
    resolved_kind = str(artifact_report["artifact_kind"])
    base_source = artifact_report.get("base_model_name_or_path")
    peft = None
    peft_config = None
    runtime_adapter_config: dict[str, object] | None = None
    base_revision = artifact_report.get("base_model_revision")
    if resolved_kind == "peft_adapter" and not base_source:
        peft = peft_module or importlib.import_module("peft")
        peft_config_class = getattr(peft, "PeftConfig", None)
        from_pretrained = getattr(peft_config_class, "from_pretrained", None)
        if not callable(from_pretrained):
            raise ValueError("PEFT runtime does not expose PeftConfig.from_pretrained")
        peft_options = _peft_loader_options(
            _loader_options(common_options, adapter_kwargs)
        )
        peft_config = from_pretrained(source, **peft_options)
        base_source = str(
            getattr(peft_config, "base_model_name_or_path", "") or ""
        ).strip()
        if not base_source:
            raise ValueError("PEFT adapter did not resolve a base model")
        runtime_adapter_config = _runtime_adapter_config(peft_config)
        base_revision = getattr(peft_config, "revision", None)

    if not base_source:
        raise ValueError("HF causal-LM artifact did not resolve a model source")
    explicit_tokenizer = (
        None
        if tokenizer_name_or_path is None
        else str(tokenizer_name_or_path).strip() or None
    )
    if explicit_tokenizer is not None:
        tokenizer_source = explicit_tokenizer
        tokenizer_source_kind = "explicit"
    elif artifact_report.get("tokenizer_source_kind") == "adapter_artifact":
        tokenizer_source = source
        tokenizer_source_kind = "adapter_artifact"
    else:
        tokenizer_source = str(base_source)
        tokenizer_source_kind = "base_model"

    base_options = dict(common_options)
    if base_revision is not None:
        base_options.setdefault("revision", base_revision)
    resolved_config_options = _loader_options(base_options, config_kwargs)
    config = transformers.AutoConfig.from_pretrained(
        str(base_source),
        **resolved_config_options,
    )
    observed_base_commit = _hub_commit_hash(getattr(config, "_commit_hash", None))
    effective_base_commit = observed_base_commit or _hub_commit_hash(base_revision)
    base_source_is_local = Path(str(base_source)).expanduser().is_dir()
    commit_pin_applied = bool(
        not base_source_is_local
        and effective_base_commit is not None
    )
    pinned_base_options = dict(base_options)
    if commit_pin_applied:
        pinned_base_options["revision"] = effective_base_commit
    resolved_tokenizer_options = _loader_options(
        (
            pinned_base_options
            if tokenizer_source == str(base_source)
            else common_options
        ),
        tokenizer_kwargs,
    )
    resolved_model_options = _loader_options(pinned_base_options, model_kwargs)
    if commit_pin_applied:
        resolved_model_options["revision"] = effective_base_commit
        if tokenizer_source == str(base_source):
            resolved_tokenizer_options["revision"] = effective_base_commit
    tokenizer = (
        transformers.AutoTokenizer.from_pretrained(
            tokenizer_source,
            **resolved_tokenizer_options,
        )
        if load_tokenizer
        else None
    )
    runtime_identity_pre_model = hf_causal_lm_runtime_identity_report(
        base_model_source=str(base_source),
        base_model_revision=(
            effective_base_commit
            if commit_pin_applied
            else base_revision
        ),
        tokenizer_source=tokenizer_source,
        tokenizer_source_kind=tokenizer_source_kind,
        config=config,
        tokenizer=tokenizer,
        expected_identity_id=expected_runtime_identity_id,
        phase="pre_model_load",
    )
    if expected_runtime_identity_id is not None and runtime_identity_pre_model.get(
        "status"
    ) != "ready":
        raise HfCausalLmRuntimeIdentityError(runtime_identity_pre_model)

    model = None
    base_parameter_report = None
    loaded_parameter_report = None
    adapter_loaded = False
    adapter_merged = False
    active_adapter = None
    if load_model:
        resolved_model_options.setdefault("config", config)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            str(base_source),
            **resolved_model_options,
        )
        base_parameter_report = hf_finetune_parameter_report(model)
        if resolved_kind == "peft_adapter":
            peft = peft or peft_module or importlib.import_module("peft")
            peft_model_class = getattr(peft, "PeftModel", None)
            from_pretrained = getattr(peft_model_class, "from_pretrained", None)
            if not callable(from_pretrained):
                raise ValueError(
                    "PEFT runtime does not expose PeftModel.from_pretrained"
                )
            peft_options = _peft_loader_options(
                _loader_options(common_options, adapter_kwargs)
            )
            model = from_pretrained(
                model,
                source,
                is_trainable=bool(is_trainable),
                **peft_options,
            )
            adapter_loaded = True
            active_adapter, attached_runtime_config = _model_runtime_adapter_config(
                model
            )
            if attached_runtime_config is not None:
                runtime_adapter_config = attached_runtime_config
            if merge_adapter:
                model = _merge_and_unload_adapter(
                    model,
                    safe_merge=bool(safe_merge),
                )
                adapter_merged = True
        loaded_parameter_report = hf_finetune_parameter_report(model)

    runtime_identity_after_model = None
    if load_model:
        runtime_identity_after_model = hf_causal_lm_runtime_identity_report(
            base_model_source=str(base_source),
            base_model_revision=(
                effective_base_commit
                if commit_pin_applied
                else base_revision
            ),
            tokenizer_source=tokenizer_source,
            tokenizer_source_kind=tokenizer_source_kind,
            config=config,
            tokenizer=tokenizer,
            expected_identity_id=(
                expected_runtime_identity_id
                or runtime_identity_pre_model.get("observed_identity_id")
            ),
            phase="after_model_load",
        )
        if (
            runtime_identity_pre_model.get("status") == "ready"
            and runtime_identity_after_model.get("status") != "ready"
        ):
            raise HfCausalLmRuntimeIdentityError(runtime_identity_after_model)

    report = dict(artifact_report)
    report.update(
        {
            "row_type": "hf_causal_lm_load_report",
            "status": "loaded",
            "resolved_base_model_name_or_path": str(base_source),
            "resolved_base_model_revision": base_revision,
            "resolved_base_model_commit": effective_base_commit,
            "base_model_commit_pin_applied": commit_pin_applied,
            "base_model_effective_revision": (
                effective_base_commit
                if commit_pin_applied
                else base_revision
            ),
            "resolved_tokenizer_source": tokenizer_source,
            "resolved_tokenizer_source_kind": tokenizer_source_kind,
            "load_model_requested": bool(load_model),
            "model_loaded": model is not None,
            "model_class": None if model is None else model.__class__.__name__,
            "config_class": config.__class__.__name__,
            "load_tokenizer_requested": bool(load_tokenizer),
            "tokenizer_loaded": tokenizer is not None,
            "tokenizer_class": (
                None if tokenizer is None else tokenizer.__class__.__name__
            ),
            "adapter_loaded": adapter_loaded,
            "adapter_trainable": bool(is_trainable) if adapter_loaded else None,
            "active_adapter": active_adapter,
            "adapter_merge_requested": bool(merge_adapter),
            "adapter_merged": adapter_merged,
            "safe_merge": bool(safe_merge) if merge_adapter else None,
            "loaded_artifact_kind": (
                "merged_model" if adapter_merged else resolved_kind
            ),
            "peft_version": (
                None if peft is None else getattr(peft, "__version__", None)
            ),
            "runtime_adapter_config": runtime_adapter_config,
            "runtime_identity_pre_model": runtime_identity_pre_model,
            "runtime_identity_after_model": runtime_identity_after_model,
            "base_parameter_report": base_parameter_report,
            "loaded_parameter_report": loaded_parameter_report,
        }
    )
    return model, tokenizer, config, report


def summarize_hf_causal_lm_artifact(
    report: Mapping[str, object],
) -> dict[str, object]:
    """Flatten an artifact/load report for embedding in larger telemetry rows."""

    parameters = report.get("loaded_parameter_report")
    parameter_report = dict(parameters) if isinstance(parameters, Mapping) else {}
    runtime_identity_pre_model = report.get("runtime_identity_pre_model")
    runtime_identity_pre_model_payload = (
        dict(runtime_identity_pre_model)
        if isinstance(runtime_identity_pre_model, Mapping)
        else {}
    )
    runtime_identity_after_model = report.get("runtime_identity_after_model")
    runtime_identity_after_model_payload = (
        dict(runtime_identity_after_model)
        if isinstance(runtime_identity_after_model, Mapping)
        else {}
    )
    runtime_identity = (
        runtime_identity_after_model_payload or runtime_identity_pre_model_payload
    )
    return {
        "row_type": "hf_causal_lm_artifact_summary",
        "status": report.get("status"),
        "artifact_kind": report.get("artifact_kind"),
        "loaded_artifact_kind": report.get("loaded_artifact_kind"),
        "artifact_source": report.get("artifact_source"),
        "artifact_is_local": report.get("artifact_is_local"),
        "base_model_name_or_path": (
            report.get("resolved_base_model_name_or_path")
            or report.get("base_model_name_or_path")
        ),
        "base_model_revision": (
            report.get("resolved_base_model_revision")
            or report.get("base_model_revision")
        ),
        "base_model_commit": report.get("resolved_base_model_commit"),
        "base_model_commit_pin_applied": report.get(
            "base_model_commit_pin_applied"
        ),
        "base_model_effective_revision": report.get(
            "base_model_effective_revision"
        ),
        "tokenizer_source": (
            report.get("resolved_tokenizer_source") or report.get("tokenizer_source")
        ),
        "tokenizer_source_kind": (
            report.get("resolved_tokenizer_source_kind")
            or report.get("tokenizer_source_kind")
        ),
        "model_loaded": report.get("model_loaded"),
        "model_class": report.get("model_class"),
        "adapter_loaded": report.get("adapter_loaded"),
        "adapter_trainable": report.get("adapter_trainable"),
        "adapter_merged": report.get("adapter_merged"),
        "peft_version": report.get("peft_version"),
        "parameter_count": parameter_report.get("parameter_count"),
        "trainable_parameter_count": parameter_report.get("trainable_parameter_count"),
        "trainable_parameter_ratio": parameter_report.get("trainable_parameter_ratio"),
        "adapter_weight_files": report.get("adapter_weight_files"),
        "runtime_resolution_required": report.get("runtime_resolution_required"),
        "runtime_identity_status": runtime_identity.get("status"),
        "runtime_identity_verified": runtime_identity.get("identity_verified"),
        "runtime_identity_expected_id": runtime_identity.get(
            "expected_identity_id"
        ),
        "runtime_identity_observed_id": runtime_identity.get(
            "observed_identity_id"
        ),
        "runtime_identity_pre_model_status": runtime_identity_pre_model_payload.get(
            "status"
        ),
        "runtime_identity_after_model_status": (
            runtime_identity_after_model_payload.get("status")
        ),
    }


def export_hf_merged_causal_lm(
    adapter_name_or_path: str | Path,
    output_dir: str | Path,
    *,
    tokenizer_name_or_path: str | Path | None = None,
    safe_merge: bool = True,
    safe_serialization: bool = True,
    transformers_module: Any = None,
    peft_module: Any = None,
    loader_kwargs: Mapping[str, object] | None = None,
    config_kwargs: Mapping[str, object] | None = None,
    tokenizer_kwargs: Mapping[str, object] | None = None,
    model_kwargs: Mapping[str, object] | None = None,
    adapter_kwargs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Merge one PEFT adapter and atomically export a standalone full model."""

    source = str(adapter_name_or_path)
    output = Path(output_dir).expanduser()
    source_directory = _artifact_local_directory(source)
    if source_directory is not None and source_directory.resolve() == output.resolve():
        raise ValueError("merged output_dir must differ from the adapter directory")
    if output.exists() and not output.is_dir():
        raise ValueError(f"merged output_dir is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"merged output_dir must be absent or empty: {output}")

    model, tokenizer, _config, load_report = load_hf_causal_lm_artifact(
        adapter_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        artifact_kind="peft_adapter",
        load_model=True,
        merge_adapter=True,
        safe_merge=safe_merge,
        transformers_module=transformers_module,
        peft_module=peft_module,
        loader_kwargs=loader_kwargs,
        config_kwargs=config_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
        model_kwargs=model_kwargs,
        adapter_kwargs=adapter_kwargs,
    )
    save_model = getattr(model, "save_pretrained", None)
    save_tokenizer = getattr(tokenizer, "save_pretrained", None)
    if not callable(save_model):
        raise ValueError("merged model does not expose save_pretrained")
    if not callable(save_tokenizer):
        raise ValueError("tokenizer does not expose save_pretrained")

    output.parent.mkdir(parents=True, exist_ok=True)
    report_filename = "spiraltorch-hf-merged-export.json"
    with tempfile.TemporaryDirectory(
        dir=output.parent,
        prefix=f".{output.name}.spiraltorch-merge-",
    ) as temporary_root:
        staging = Path(temporary_root) / "artifact"
        staging.mkdir()
        save_model(staging, safe_serialization=bool(safe_serialization))
        save_tokenizer(staging)
        output_files = sorted(path.name for path in staging.iterdir() if path.is_file())
        report = {
            "row_type": "hf_merged_causal_lm_export_report",
            "status": "exported",
            "adapter_source": source,
            "output_dir": str(output),
            "report_path": str(output / report_filename),
            "safe_merge": bool(safe_merge),
            "safe_serialization": bool(safe_serialization),
            "load_report": load_report,
            "base_model_name_or_path": load_report.get(
                "resolved_base_model_name_or_path"
            ),
            "base_model_revision": load_report.get("resolved_base_model_revision"),
            "tokenizer_source": load_report.get("resolved_tokenizer_source"),
            "parameter_report": hf_finetune_parameter_report(model),
            "output_files": [*output_files, report_filename],
        }
        (staging / report_filename).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            output.rmdir()
        staging.replace(output)
    return report


def hf_merged_causal_lm_export_lines(
    report: Mapping[str, object],
) -> list[str]:
    """Render one compact merged-export audit line."""

    load_report = report.get("load_report")
    load_payload = dict(load_report) if isinstance(load_report, Mapping) else {}
    return [
        "hf_merged_causal_lm_export "
        f"status={report.get('status')} "
        f"adapter={report.get('adapter_source')} "
        f"base={report.get('base_model_name_or_path')} "
        f"output={report.get('output_dir')} "
        f"adapter_loaded={load_payload.get('adapter_loaded')} "
        f"adapter_merged={load_payload.get('adapter_merged')} "
        f"safe_merge={report.get('safe_merge')}"
    ]


def _gradient_checkpointing_report(model: Any, *, enabled: bool) -> dict[str, object]:
    report: dict[str, object] = {
        "requested": bool(enabled),
        "enabled": False,
        "enable_input_require_grads": False,
        "use_cache_before": None,
        "use_cache_after": None,
    }
    config = getattr(model, "config", None)
    if config is not None:
        report["use_cache_before"] = getattr(config, "use_cache", None)
    if enabled:
        enable = getattr(model, "gradient_checkpointing_enable", None)
        if not callable(enable):
            raise ValueError("model does not support gradient checkpointing")
        enable()
        report["enabled"] = True
        require_inputs = getattr(model, "enable_input_require_grads", None)
        if callable(require_inputs):
            require_inputs()
            report["enable_input_require_grads"] = True
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False
    if config is not None:
        report["use_cache_after"] = getattr(config, "use_cache", None)
    return report


def prepare_hf_finetune_model(
    model: Any,
    *,
    mode: str = "full",
    model_family: str | None = None,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
    bias: str = "none",
    target_modules: object = None,
    modules_to_save: object = None,
    use_rslora: bool = False,
    gradient_checkpointing: bool = False,
    preloaded_adapter: bool = False,
    peft_module: Any = None,
) -> tuple[Any, dict[str, object]]:
    """Prepare a Transformers causal LM for full FT or attach a PEFT LoRA adapter.

    ``peft`` is imported only for LoRA mode, so importing SpiralTorch does not
    make the optional HF stack an eager runtime dependency. Set
    ``preloaded_adapter`` when ``model`` is an already reconstructed trainable
    PEFT artifact; the existing adapter is then reused instead of wrapping the
    model a second time.
    """

    family = _model_family(model, model_family)
    config = hf_finetune_adapter_config(
        mode=mode,
        model_family=family,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        bias=bias,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        use_rslora=use_rslora,
        gradient_checkpointing=gradient_checkpointing,
    )
    before = hf_finetune_parameter_report(model)
    active_adapter, runtime_adapter_config = _model_runtime_adapter_config(model)
    if runtime_adapter_config is not None and config["mode"] == "full":
        raise ValueError(
            "full fine-tuning cannot reuse a PEFT-wrapped model; merge the adapter "
            "first or select LoRA mode"
        )
    if preloaded_adapter and config["mode"] != "lora":
        raise ValueError("preloaded_adapter requires LoRA fine-tuning mode")
    if preloaded_adapter and runtime_adapter_config is None:
        raise ValueError("preloaded_adapter requires a model with PEFT configuration")
    if runtime_adapter_config is not None and not preloaded_adapter:
        raise ValueError(
            "model already contains a PEFT adapter; set preloaded_adapter=True "
            "to continue it without attaching a second adapter"
        )
    checkpointing = _gradient_checkpointing_report(
        model,
        enabled=bool(config["gradient_checkpointing"]),
    )
    if config["mode"] == "full":
        return model, {
            "row_type": "hf_finetune_model_prepare_report",
            "status": "ready",
            "mode": "full",
            "adapter_attached": False,
            "adapter_attached_now": False,
            "adapter_preloaded": False,
            "adapter_origin": None,
            "active_adapter": None,
            "model_family": family,
            "adapter_config": config,
            "requested_adapter_config": config,
            "adapter_config_source": "request",
            "adapter_config_applied": True,
            "runtime_adapter_config": None,
            "target_report": None,
            "parameter_report_before": before,
            "parameter_report_after": hf_finetune_parameter_report(model),
            "gradient_checkpointing": checkpointing,
            "peft_version": None,
        }

    if preloaded_adapter:
        peft = peft_module or importlib.import_module("peft")
        after = hf_finetune_parameter_report(model)
        if after["trainable_parameter_count"] == 0:
            raise ValueError("preloaded PEFT adapter has no trainable parameters")
        return model, {
            "row_type": "hf_finetune_model_prepare_report",
            "status": "ready",
            "mode": "lora",
            "adapter_attached": True,
            "adapter_attached_now": False,
            "adapter_preloaded": True,
            "adapter_origin": "artifact",
            "active_adapter": active_adapter,
            "model_family": family,
            "adapter_config": config,
            "requested_adapter_config": config,
            "adapter_config_source": "loaded_artifact",
            "adapter_config_applied": False,
            "runtime_adapter_config": runtime_adapter_config,
            "target_report": None,
            "parameter_report_before": before,
            "parameter_report_after": after,
            "gradient_checkpointing": checkpointing,
            "peft_version": getattr(peft, "__version__", None),
        }

    target_report = hf_finetune_lora_target_report(
        model,
        model_family=family,
        target_modules=target_modules,
    )
    peft = peft_module or importlib.import_module("peft")
    task_type = getattr(getattr(peft, "TaskType", None), "CAUSAL_LM", "CAUSAL_LM")
    lora_kwargs: dict[str, object] = {
        "r": config["rank"],
        "lora_alpha": config["alpha"],
        "lora_dropout": config["dropout"],
        "bias": config["bias"],
        "task_type": task_type,
        "target_modules": target_report["target_modules"],
        "use_rslora": config["use_rslora"],
    }
    if config["modules_to_save"]:
        lora_kwargs["modules_to_save"] = config["modules_to_save"]
    if family in {"gpt2", "gpt_bigcode"}:
        lora_kwargs["fan_in_fan_out"] = True
    lora_config = peft.LoraConfig(**lora_kwargs)
    prepared = peft.get_peft_model(model, lora_config)
    after = hf_finetune_parameter_report(prepared)
    if after["trainable_parameter_count"] == 0:
        raise ValueError("LoRA preparation produced no trainable parameters")
    return prepared, {
        "row_type": "hf_finetune_model_prepare_report",
        "status": "ready",
        "mode": "lora",
        "adapter_attached": True,
        "adapter_attached_now": True,
        "adapter_preloaded": False,
        "adapter_origin": "new",
        "active_adapter": None,
        "model_family": family,
        "adapter_config": config,
        "requested_adapter_config": config,
        "adapter_config_source": "request",
        "adapter_config_applied": True,
        "runtime_adapter_config": None,
        "target_report": target_report,
        "parameter_report_before": before,
        "parameter_report_after": after,
        "gradient_checkpointing": checkpointing,
        "peft_version": getattr(peft, "__version__", None),
    }
