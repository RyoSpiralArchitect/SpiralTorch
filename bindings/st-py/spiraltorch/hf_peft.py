"""Reusable PEFT/LoRA preparation for Hugging Face fine-tuning."""

from __future__ import annotations

import importlib
import math
from collections.abc import Iterable
from typing import Any

__all__ = [
    "HF_FINETUNE_LORA_TARGET_MODULES",
    "HF_FINETUNE_MODES",
    "hf_finetune_adapter_config",
    "hf_finetune_lora_target_report",
    "hf_finetune_parameter_report",
    "prepare_hf_finetune_model",
]


HF_FINETUNE_MODES = ("full", "lora")

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


def _normalise_model_family(value: object) -> str | None:
    if value is None:
        return None
    family = str(value).strip().lower().replace(" ", "_")
    if not family:
        return None
    return _MODEL_FAMILY_ALIASES.get(family, family)


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
    peft_module: Any = None,
) -> tuple[Any, dict[str, object]]:
    """Prepare a Transformers causal LM for full FT or attach a PEFT LoRA adapter.

    ``peft`` is imported only for LoRA mode, so importing SpiralTorch does not
    make the optional HF stack an eager runtime dependency.
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
            "model_family": family,
            "adapter_config": config,
            "target_report": None,
            "parameter_report_before": before,
            "parameter_report_after": hf_finetune_parameter_report(model),
            "gradient_checkpointing": checkpointing,
            "peft_version": None,
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
        "model_family": family,
        "adapter_config": config,
        "target_report": target_report,
        "parameter_report_before": before,
        "parameter_report_after": after,
        "gradient_checkpointing": checkpointing,
        "peft_version": getattr(peft, "__version__", None),
    }
