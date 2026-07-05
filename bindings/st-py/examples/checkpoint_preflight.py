import argparse
import importlib
import json
from collections.abc import Mapping
from pathlib import Path

import spiraltorch as st
from spiraltorch.ecosystem import (
    bound_external_state_tensors,
    checkpoint_from_external_state,
    external_tensor_shape,
    slice_external_tensor,
    tensor_from_external,
)
from spiraltorch.nn import Linear, LoraLinear, ZSpaceProjector
from spiraltorch.runtime_imports import (
    TRANSFORMERS_TRACE_RUNTIME_IMPORT_PRESETS,
    runtime_import_names_from_args,
    runtime_import_probe_fields,
    runtime_import_requirement_failures,
)


VOCAB = 8
HIDDEN = 4
TARGET_CLASSES = 6
EXTERNAL_CLASSES = 4
CHECKPOINT_PROJECTION_STRENGTH = 0.5
CHECKPOINT_PROJECTION_CURVATURE = -0.5
CHECKPOINT_PROJECTION_FREQUENCY = 0.65
CHECKPOINT_SOURCE_GAIN = 1.0
CHECKPOINT_PROJECTION_PRESETS = {
    "healthy": {
        "checkpoint_projection_strength": 1.0,
        "checkpoint_projection_curvature": -0.04,
        "checkpoint_projection_frequency": CHECKPOINT_PROJECTION_FREQUENCY,
    }
}
HF_KEY_PRESETS = {
    "gpt2": {
        "embed_weight_key": "transformer.wte.weight",
        "embed_bias_key": "transformer.wte.bias",
        "lm_head_weight_key": "lm_head.weight",
        "lm_head_bias_key": "lm_head.bias",
    },
    "gpt2_bare": {
        "embed_weight_key": "wte.weight",
        "embed_bias_key": "wte.bias",
        "lm_head_weight_key": "lm_head.weight",
        "lm_head_bias_key": "lm_head.bias",
    },
    "gemma": {
        "embed_weight_key": "model.language_model.embed_tokens.weight",
        "embed_bias_key": "model.language_model.embed_tokens.bias",
        "lm_head_weight_key": "model.language_model.lm_head.weight",
        "lm_head_bias_key": "model.language_model.lm_head.bias",
    },
    "llama": {
        "embed_weight_key": "model.embed_tokens.weight",
        "embed_bias_key": "model.embed_tokens.bias",
        "lm_head_weight_key": "lm_head.weight",
        "lm_head_bias_key": "lm_head.bias",
    },
    "gpt_neox": {
        "embed_weight_key": "gpt_neox.embed_in.weight",
        "embed_bias_key": "gpt_neox.embed_in.bias",
        "lm_head_weight_key": "embed_out.weight",
        "lm_head_bias_key": "embed_out.bias",
    },
}
HF_UNUSED_KEYS = {
    "gpt2": "transformer.h.0.ln_1.weight",
    "gpt2_bare": "h.0.ln_1.weight",
    "gemma": "model.language_model.layers.0.input_layernorm.weight",
    "llama": "model.layers.0.input_layernorm.weight",
    "gpt_neox": "gpt_neox.layers.0.input_layernorm.weight",
}
AUTO_KEY_PRESET = "auto"
HF_KEY_PRESET_CHOICES = [AUTO_KEY_PRESET, *sorted(HF_KEY_PRESETS)]
SOURCE_CHOICES = (
    "toy",
    "hf-style",
    "hf-no-bias",
    "hf-llama",
    "hf-gpt-neox",
)


def _tensor_rows(tensor):
    data = tensor.data()
    return [
        data[row * tensor.cols : (row + 1) * tensor.cols]
        for row in range(tensor.rows)
    ]


def _tensor_shape(value, name):
    if _is_shape_tuple(value):
        return value
    return external_tensor_shape(value, name=name)


def _is_shape_tuple(value):
    return isinstance(value, tuple) and all(isinstance(dim, int) for dim in value)


def zero_row_tensor(cols):
    if cols is None or cols <= 0:
        raise ValueError(f"zero bias width must be positive, got {cols!r}")
    return st.Tensor(1, cols, [0.0] * cols)


def _transform_transposes(transform):
    return transform in {
        "transpose",
        "transpose_copy_overlap",
        "transpose_copy_overlap_zeros",
    }


def _infer_bias_cols(state, source_name):
    rows, cols = _tensor_shape(_state_value(state, source_name), source_name)
    if rows <= 0 or cols <= 0:
        raise ValueError(f"{source_name} has invalid shape {(rows, cols)!r}")
    return cols


def _infer_lm_head_bias_cols(state, source_name, *, transform):
    rows, cols = _tensor_shape(_state_value(state, source_name), source_name)
    if rows <= 0 or cols <= 0:
        raise ValueError(f"{source_name} has invalid shape {(rows, cols)!r}")
    return rows if _transform_transposes(transform) else cols


def _identity_transform(transform):
    return transform in {None, "", "identity"}


def _rule_value(target, transform):
    if _identity_transform(transform):
        return target
    return {
        "target": target,
        "transform": transform,
    }


def hf_lm_overlap_resize_kwargs():
    return {
        "embed_weight_transform": "copy_overlap_zeros",
        "embed_bias_transform": "copy_overlap_zeros",
        "lm_head_weight_transform": "transpose_copy_overlap_zeros",
        "lm_head_bias_transform": "copy_overlap_zeros",
    }


def hf_lm_tensor_bounds_for_module_shapes(
    module_shapes,
    *,
    key_preset="gpt2",
    lm_head_weight_transform="transpose",
    **overrides,
):
    vocab, hidden, target_classes = module_shapes
    key_kwargs = _hf_key_kwargs(key_preset, dict(overrides))
    bounds = {}

    def add_bound(key_name, shape):
        source_key = key_kwargs.get(key_name)
        if source_key is not None:
            bounds[source_key] = shape

    add_bound("embed_weight_key", (vocab, hidden))
    add_bound("embed_bias_key", (1, hidden))
    if _transform_transposes(lm_head_weight_transform):
        add_bound("lm_head_weight_key", (target_classes, hidden))
    else:
        add_bound("lm_head_weight_key", (hidden, target_classes))
    add_bound("lm_head_bias_key", (1, target_classes))
    return bounds


def add_checkpoint_projection_args(parser):
    parser.add_argument(
        "--checkpoint-projection",
        choices=["none", "zspace"],
        default="none",
        help="Optional projection policy to apply to embed/head checkpoint tensors before load.",
    )
    parser.add_argument(
        "--checkpoint-projection-preset",
        choices=sorted(CHECKPOINT_PROJECTION_PRESETS),
        default=None,
        help=(
            "Shortcut checkpoint projection preset. 'healthy' enables the "
            "current Z-space projection-health candidate."
        ),
    )
    parser.add_argument(
        "--checkpoint-projection-strength",
        type=float,
        default=CHECKPOINT_PROJECTION_STRENGTH,
        help="Z-space projection strength for --checkpoint-projection zspace.",
    )
    parser.add_argument(
        "--checkpoint-projection-curvature",
        type=float,
        default=CHECKPOINT_PROJECTION_CURVATURE,
        help="OpenTopos curvature for --checkpoint-projection zspace.",
    )
    parser.add_argument(
        "--checkpoint-projection-frequency",
        type=float,
        default=CHECKPOINT_PROJECTION_FREQUENCY,
        help="LanguageWaveEncoder frequency for --checkpoint-projection zspace.",
    )


def add_checkpoint_source_gain_args(parser):
    parser.add_argument(
        "--checkpoint-source-gain",
        type=float,
        default=CHECKPOINT_SOURCE_GAIN,
        help=(
            "Multiply mapped embed/head checkpoint tensors by this positive "
            "gain after optional projection and before module load."
        ),
    )


def add_transformers_audit_args(parser):
    parser.add_argument(
        "--transformers-audit",
        action="store_true",
        help=(
            "Optionally co-import local Hugging Face Transformers config/tokenizer "
            "metadata beside the checkpoint audit. Transformers remains an "
            "optional dependency."
        ),
    )
    parser.add_argument(
        "--transformers-model-path",
        type=Path,
        default=None,
        help=(
            "Local Transformers model/config/tokenizer directory. Defaults to "
            "--hf-state-dict when it is a directory, otherwise the state-dict parent."
        ),
    )
    parser.add_argument(
        "--transformers-revision",
        default=None,
        help="Optional Transformers revision forwarded to from_pretrained(...).",
    )
    parser.add_argument(
        "--allow-transformers-remote",
        action="store_true",
        help=(
            "Allow Transformers from_pretrained(...) to resolve remote files. "
            "By default audits are local_files_only to keep checkpoint preflight bounded."
        ),
    )
    parser.add_argument(
        "--transformers-trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code=True to Transformers loaders.",
    )
    parser.add_argument(
        "--skip-transformers-tokenizer",
        action="store_true",
        help="Only load AutoConfig metadata during --transformers-audit.",
    )
    parser.add_argument(
        "--transformers-load-model",
        action="store_true",
        help=(
            "Also instantiate AutoModelForCausalLM during --transformers-audit. "
            "This can be heavy and is off by default."
        ),
    )
    parser.add_argument(
        "--require-transformers-audit",
        action="store_true",
        help="Fail when the optional Transformers audit is requested but not clean.",
    )
    parser.add_argument(
        "--transformers-runtime-import",
        dest="runtime_imports",
        action="append",
        default=[],
        help=(
            "Additional Python module imported during --transformers-audit while "
            "SpiralTorch and Transformers audit code are loaded. May be repeated."
        ),
    )
    parser.add_argument(
        "--transformers-runtime-import-preset",
        dest="runtime_import_presets",
        action="append",
        choices=sorted(TRANSFORMERS_TRACE_RUNTIME_IMPORT_PRESETS),
        default=[],
        help=(
            "Named runtime import bundle to probe during --transformers-audit. "
            "'torch-transformers' probes Transformers plus torch; 'hf-runtime' "
            "also probes tokenizers. May be repeated."
        ),
    )
    parser.add_argument(
        "--transformers-runtime-contract-preset",
        "--runtime-contract-preset",
        dest="runtime_contract_presets",
        action="append",
        choices=sorted(TRANSFORMERS_TRACE_RUNTIME_IMPORT_PRESETS),
        default=[],
        help=(
            "Shortcut Transformers runtime contract preset: enables "
            "--transformers-audit, probes the preset modules, and requires them "
            "to import in the same audit process. May be repeated."
        ),
    )
    parser.add_argument(
        "--require-transformers-runtime-imports",
        dest="require_runtime_imports",
        action="store_true",
        help=(
            "Fail checkpoint preflight when any Transformers audit runtime import "
            "probe fails."
        ),
    )
    parser.add_argument(
        "--require-transformers-runtime-import",
        dest="required_runtime_imports",
        action="append",
        default=[],
        help=(
            "Require this module to import during --transformers-audit. May be "
            "repeated."
        ),
    )
    parser.add_argument(
        "--require-transformers-runtime-import-preset",
        dest="required_runtime_import_presets",
        action="append",
        choices=sorted(TRANSFORMERS_TRACE_RUNTIME_IMPORT_PRESETS),
        default=[],
        help=(
            "Require this named runtime import preset to be observed and satisfied "
            "during --transformers-audit. May be repeated."
        ),
    )


def transformers_runtime_import_requested(args):
    return any(
        [
            bool(getattr(args, "runtime_imports", []) or []),
            bool(getattr(args, "runtime_import_presets", []) or []),
            bool(getattr(args, "required_runtime_imports", []) or []),
            bool(getattr(args, "required_runtime_import_presets", []) or []),
            bool(getattr(args, "require_runtime_imports", False)),
        ]
    )


def append_unique(values, additions):
    return list(dict.fromkeys([*(values or []), *(additions or [])]))


def apply_transformers_runtime_contract_presets(args):
    presets = list(dict.fromkeys(getattr(args, "runtime_contract_presets", []) or []))
    if not presets:
        return args
    args.transformers_audit = True
    args.runtime_import_presets = append_unique(args.runtime_import_presets, presets)
    args.required_runtime_import_presets = append_unique(
        args.required_runtime_import_presets,
        presets,
    )
    args.require_runtime_imports = True
    return args


def checkpoint_projection_preset_values(args):
    preset = getattr(args, "checkpoint_projection_preset", None)
    if preset is None:
        return {}
    return CHECKPOINT_PROJECTION_PRESETS[preset]


def resolved_checkpoint_projection_policy(args):
    if getattr(args, "checkpoint_projection_preset", None) is not None:
        return "zspace"
    return getattr(args, "checkpoint_projection", "none")


def resolved_checkpoint_projection_value(args, name):
    defaults = {
        "checkpoint_projection_strength": CHECKPOINT_PROJECTION_STRENGTH,
        "checkpoint_projection_curvature": CHECKPOINT_PROJECTION_CURVATURE,
        "checkpoint_projection_frequency": CHECKPOINT_PROJECTION_FREQUENCY,
    }
    preset = checkpoint_projection_preset_values(args)
    return preset.get(name, getattr(args, name, defaults[name]))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preflight an external checkpoint key/shape transform handoff."
    )
    parser.add_argument(
        "--jsonl",
        default=None,
        help="Optional path for flat preflight report rows.",
    )
    parser.add_argument(
        "--compare-jsonl",
        default=None,
        help="Optional previous flat preflight report JSONL to compare.",
    )
    parser.add_argument(
        "--require-preflight-match",
        action="store_true",
        help="Fail when --compare-jsonl differs from the current preflight rows.",
    )
    parser.add_argument(
        "--source",
        choices=SOURCE_CHOICES,
        default="toy",
        help="Synthetic external checkpoint naming style to preflight.",
    )
    parser.add_argument(
        "--hf-state-dict",
        type=Path,
        default=None,
        help=(
            "Optional local HF/PyTorch state dict file or directory to preflight "
            "instead of --source. Supports .safetensors and torch .bin/.pt files."
        ),
    )
    parser.add_argument(
        "--key-preset",
        choices=HF_KEY_PRESET_CHOICES,
        default="gpt2",
        help=(
            "HF-style key preset used with --hf-state-dict. Use 'auto' to "
            "infer the layout from checkpoint shape metadata."
        ),
    )
    parser.add_argument(
        "--include-extra-key",
        dest="include_extra_keys",
        action="append",
        default=[],
        help="Additional external state-dict key to include in the preflight audit.",
    )
    parser.add_argument(
        "--no-synthesize-missing-biases",
        action="store_true",
        help="Require embed/head bias tensors to exist in --hf-state-dict.",
    )
    parser.add_argument(
        "--allow-overlap-resize",
        action="store_true",
        help=(
            "Explicitly adapt HF embed/head tensors to the requested module shape "
            "with overlap-copy and zero-fill transforms."
        ),
    )
    parser.add_argument(
        "--shape-only",
        action="store_true",
        help=(
            "For --hf-state-dict, audit key presence and tensor shapes without "
            "constructing SpiralTorch modules or materializing compatible tensors."
        ),
    )
    parser.add_argument(
        "--require-shape-materializable",
        action="store_true",
        help=(
            "With --shape-only, fail unless the requested module shape can be "
            "materialized exactly or via --allow-overlap-resize."
        ),
    )
    parser.add_argument(
        "--require-exact-shape-match",
        action="store_true",
        help="With --shape-only, fail unless checkpoint and requested shapes match exactly.",
    )
    parser.add_argument(
        "--require-detected-key-preset",
        choices=sorted(HF_KEY_PRESETS),
        default=None,
        help=(
            "With --shape-only, fail unless the detected/resolved key preset "
            "matches this value."
        ),
    )
    add_checkpoint_projection_args(parser)
    add_checkpoint_source_gain_args(parser)
    add_transformers_audit_args(parser)
    parser.add_argument(
        "--vocab",
        type=int,
        default=None,
        help="Override embed rows for --hf-state-dict; defaults to inferred shape.",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=None,
        help="Override hidden columns for --hf-state-dict; defaults to inferred shape.",
    )
    parser.add_argument(
        "--target-classes",
        type=int,
        default=None,
        help="Override head output classes for --hf-state-dict; defaults to inferred shape.",
    )
    args = parser.parse_args()
    apply_transformers_runtime_contract_presets(args)
    if args.require_preflight_match and args.compare_jsonl is None:
        parser.error("--require-preflight-match requires --compare-jsonl")
    if args.shape_only and args.hf_state_dict is None:
        parser.error("--shape-only requires --hf-state-dict")
    if args.key_preset == AUTO_KEY_PRESET and args.hf_state_dict is None:
        parser.error("--key-preset auto requires --hf-state-dict")
    if args.shape_only and args.compare_jsonl is not None:
        parser.error("--shape-only does not support --compare-jsonl")
    if args.require_transformers_audit and not args.transformers_audit:
        parser.error("--require-transformers-audit requires --transformers-audit")
    if args.transformers_audit and args.hf_state_dict is None and args.transformers_model_path is None:
        parser.error("--transformers-audit requires --hf-state-dict or --transformers-model-path")
    if transformers_runtime_import_requested(args) and not args.transformers_audit:
        parser.error("Transformers runtime import options require --transformers-audit")
    if args.require_runtime_imports and not runtime_import_names_from_args(
        args,
        preset_modules=TRANSFORMERS_TRACE_RUNTIME_IMPORT_PRESETS,
    ):
        parser.error(
            "--require-transformers-runtime-imports requires "
            "--transformers-runtime-import, --transformers-runtime-import-preset, "
            "or a direct --require-transformers-runtime-import/import-preset gate"
        )
    for name in [
        "require_shape_materializable",
        "require_exact_shape_match",
        "require_detected_key_preset",
    ]:
        if getattr(args, name) and not args.shape_only:
            parser.error(f"--{name.replace('_', '-')} requires --shape-only")
    for name in ["vocab", "hidden", "target_classes"]:
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.checkpoint_source_gain <= 0.0:
        parser.error("--checkpoint-source-gain must be positive")
    return args


def checkpoint_source_gain_value(args):
    return float(getattr(args, "checkpoint_source_gain", CHECKPOINT_SOURCE_GAIN))


def checkpoint_source_gain_fields(args):
    return {"checkpoint_source_gain": checkpoint_source_gain_value(args)}


def checkpoint_projection_fields(args):
    projection = resolved_checkpoint_projection_policy(args)
    if projection == "none":
        return {
            "checkpoint_projection": "none",
            "checkpoint_projection_strength": None,
            "checkpoint_projection_curvature": None,
            "checkpoint_projection_frequency": None,
        }
    return {
        "checkpoint_projection": projection,
        "checkpoint_projection_strength": float(
            resolved_checkpoint_projection_value(args, "checkpoint_projection_strength")
        ),
        "checkpoint_projection_curvature": float(
            resolved_checkpoint_projection_value(args, "checkpoint_projection_curvature")
        ),
        "checkpoint_projection_frequency": float(
            resolved_checkpoint_projection_value(args, "checkpoint_projection_frequency")
        ),
    }


def transformers_audit_requested(args):
    return bool(getattr(args, "transformers_audit", False))


def resolved_transformers_model_path(args):
    explicit = getattr(args, "transformers_model_path", None)
    if explicit is not None:
        return Path(explicit)
    hf_state_dict = getattr(args, "hf_state_dict", None)
    if hf_state_dict is None:
        return None
    path = Path(hf_state_dict)
    return path if path.is_dir() else path.parent


def _class_name(value):
    if value is None:
        return None
    return value.__class__.__name__


def _error_label(exc):
    return f"{exc.__class__.__name__}: {exc}"


def _string_list_label(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value) if value else None
    return str(value)


def _first_attr(value, names):
    for name in names:
        if hasattr(value, name):
            attr = getattr(value, name)
            if attr is not None:
                return attr
    return None


def _safe_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _tokenizer_vocab_size(tokenizer):
    length = None
    try:
        length = len(tokenizer)
    except TypeError:
        length = None
    return _safe_int(length if length is not None else getattr(tokenizer, "vocab_size", None))


def _model_parameter_count(model):
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return None
    total = 0
    for parameter in parameters():
        numel = getattr(parameter, "numel", None)
        if callable(numel):
            total += int(numel())
    return total


def _transformers_loader_kwargs(args):
    kwargs = {
        "local_files_only": not bool(getattr(args, "allow_transformers_remote", False)),
        "trust_remote_code": bool(getattr(args, "transformers_trust_remote_code", False)),
    }
    revision = getattr(args, "transformers_revision", None)
    if revision:
        kwargs["revision"] = revision
    return kwargs


def _base_transformers_audit_fields(args, path, module_shapes):
    vocab, hidden, target_classes = module_shapes
    return {
        "transformers_audit_requested": True,
        "transformers_audit_status": "not_run",
        "transformers_audit_error": None,
        "transformers_model_path": None if path is None else str(path),
        "transformers_local_files_only": not bool(
            getattr(args, "allow_transformers_remote", False)
        ),
        "transformers_trust_remote_code": bool(
            getattr(args, "transformers_trust_remote_code", False)
        ),
        "transformers_revision": getattr(args, "transformers_revision", None),
        "transformers_available": False,
        "transformers_version": None,
        "transformers_config_loaded": False,
        "transformers_config_class": None,
        "transformers_model_type": None,
        "transformers_architectures": None,
        "transformers_config_vocab_size": None,
        "transformers_config_hidden_size": None,
        "transformers_config_num_hidden_layers": None,
        "transformers_config_num_attention_heads": None,
        "transformers_config_max_position_embeddings": None,
        "transformers_config_vocab_matches_checkpoint": None,
        "transformers_config_hidden_matches_checkpoint": None,
        "transformers_config_lm_head_matches_checkpoint": None,
        "transformers_checkpoint_vocab": vocab,
        "transformers_checkpoint_hidden": hidden,
        "transformers_checkpoint_target_classes": target_classes,
        "transformers_tokenizer_requested": not bool(
            getattr(args, "skip_transformers_tokenizer", False)
        ),
        "transformers_tokenizer_loaded": False,
        "transformers_tokenizer_error": None,
        "transformers_tokenizer_class": None,
        "transformers_tokenizer_vocab_size": None,
        "transformers_tokenizer_model_max_length": None,
        "transformers_load_model_requested": bool(
            getattr(args, "transformers_load_model", False)
        ),
        "transformers_model_loaded": False,
        "transformers_model_error": None,
        "transformers_model_class": None,
        "transformers_model_parameter_count": None,
    }


def transformers_runtime_import_fields(args):
    if not transformers_runtime_import_requested(args):
        return {}
    return runtime_import_probe_fields(
        args,
        preset_modules=TRANSFORMERS_TRACE_RUNTIME_IMPORT_PRESETS,
    )


def transformers_runtime_audit_fields(args, module_shapes):
    if not transformers_audit_requested(args):
        return {}

    path = resolved_transformers_model_path(args)
    fields = _base_transformers_audit_fields(args, path, module_shapes)
    fields.update(transformers_runtime_import_fields(args))
    if path is None:
        fields.update(
            {
                "transformers_audit_status": "missing_path",
                "transformers_audit_error": "no Transformers model path resolved",
            }
        )
        return fields

    local_files_only = fields["transformers_local_files_only"]
    if local_files_only and not path.exists():
        fields.update(
            {
                "transformers_audit_status": "path_missing",
                "transformers_audit_error": f"{path} does not exist",
            }
        )
        return fields

    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        fields.update(
            {
                "transformers_audit_status": "missing_dependency",
                "transformers_audit_error": _error_label(exc),
            }
        )
        return fields

    fields["transformers_available"] = True
    fields["transformers_version"] = getattr(transformers, "__version__", None)
    loader_kwargs = _transformers_loader_kwargs(args)
    path_label = str(path)

    try:
        config = transformers.AutoConfig.from_pretrained(path_label, **loader_kwargs)
    except Exception as exc:  # pragma: no cover - exercised through tests with stubs.
        fields.update(
            {
                "transformers_audit_status": "config_error",
                "transformers_audit_error": _error_label(exc),
            }
        )
        return fields

    config_vocab = _safe_int(getattr(config, "vocab_size", None))
    config_hidden = _safe_int(
        _first_attr(config, ["hidden_size", "n_embd", "d_model"])
    )
    fields.update(
        {
            "transformers_config_loaded": True,
            "transformers_config_class": _class_name(config),
            "transformers_model_type": getattr(config, "model_type", None),
            "transformers_architectures": _string_list_label(
                getattr(config, "architectures", None)
            ),
            "transformers_config_vocab_size": config_vocab,
            "transformers_config_hidden_size": config_hidden,
            "transformers_config_num_hidden_layers": _safe_int(
                _first_attr(config, ["num_hidden_layers", "n_layer", "num_layers"])
            ),
            "transformers_config_num_attention_heads": _safe_int(
                _first_attr(config, ["num_attention_heads", "n_head"])
            ),
            "transformers_config_max_position_embeddings": _safe_int(
                _first_attr(config, ["max_position_embeddings", "n_positions"])
            ),
            "transformers_config_vocab_matches_checkpoint": (
                None if config_vocab is None else config_vocab == module_shapes[0]
            ),
            "transformers_config_hidden_matches_checkpoint": (
                None if config_hidden is None else config_hidden == module_shapes[1]
            ),
            "transformers_config_lm_head_matches_checkpoint": (
                None if config_vocab is None else config_vocab == module_shapes[2]
            ),
        }
    )

    status = "ok"
    if fields["transformers_tokenizer_requested"]:
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                path_label,
                **loader_kwargs,
            )
            fields.update(
                {
                    "transformers_tokenizer_loaded": True,
                    "transformers_tokenizer_class": _class_name(tokenizer),
                    "transformers_tokenizer_vocab_size": _tokenizer_vocab_size(tokenizer),
                    "transformers_tokenizer_model_max_length": _safe_int(
                        getattr(tokenizer, "model_max_length", None)
                    ),
                }
            )
        except Exception as exc:  # pragma: no cover - exercised through tests with stubs.
            status = "tokenizer_error"
            fields["transformers_tokenizer_error"] = _error_label(exc)

    if fields["transformers_load_model_requested"]:
        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                path_label,
                config=config,
                **loader_kwargs,
            )
            fields.update(
                {
                    "transformers_model_loaded": True,
                    "transformers_model_class": _class_name(model),
                    "transformers_model_parameter_count": _model_parameter_count(model),
                }
            )
        except Exception as exc:  # pragma: no cover - exercised through tests with stubs.
            status = "model_error" if status == "ok" else "partial_error"
            fields["transformers_model_error"] = _error_label(exc)

    fields["transformers_audit_status"] = status
    return fields


def print_transformers_audit(row):
    if "transformers_audit_status" not in row:
        return
    print(
        "transformers_audit "
        f"status={row['transformers_audit_status']} "
        f"path={row['transformers_model_path']} "
        f"available={row['transformers_available']} "
        f"config_loaded={row['transformers_config_loaded']} "
        f"model_type={row['transformers_model_type']} "
        f"config_vocab={row['transformers_config_vocab_size']} "
        f"config_hidden={row['transformers_config_hidden_size']} "
        "config_vocab_matches_checkpoint="
        f"{row['transformers_config_vocab_matches_checkpoint']} "
        "config_hidden_matches_checkpoint="
        f"{row['transformers_config_hidden_matches_checkpoint']} "
        f"tokenizer_loaded={row['transformers_tokenizer_loaded']} "
        f"tokenizer_vocab={row['transformers_tokenizer_vocab_size']} "
        f"model_loaded={row['transformers_model_loaded']} "
        f"model_parameters={row['transformers_model_parameter_count']} "
        f"error={row['transformers_audit_error']!r}"
    )
    print_transformers_runtime_imports(row)


def print_transformers_runtime_imports(row):
    if "runtime_imports_requested" not in row:
        return
    print(
        "transformers_runtime_imports "
        f"requested={row['runtime_imports_requested']} "
        f"imported={row['runtime_imports_imported']} "
        f"failed={row['runtime_imports_failed']} "
        f"all_ok={row['runtime_imports_all_ok']} "
        f"presets={row['runtime_import_presets']} "
        f"presets_satisfied={row['runtime_import_presets_satisfied']} "
        f"required_imports_passed={row['required_runtime_imports_passed']} "
        f"required_presets_passed={row['required_runtime_import_presets_passed']}"
    )


def check_transformers_runtime_import_gate(row, args):
    if not transformers_runtime_import_requested(args):
        return True
    failures = []
    if (
        getattr(args, "require_runtime_imports", False)
        and row.get("runtime_imports_all_ok") is not True
    ):
        failures.append("runtime_imports_failed:" + str(row.get("runtime_imports_failed")))
    failures.extend(runtime_import_requirement_failures(row))
    passed = not failures
    print(
        "transformers_runtime_import_gate "
        f"passed={passed} "
        f"failures={','.join(failures) if failures else 'none'}"
    )
    if not passed:
        raise RuntimeError(
            "Transformers runtime import gate failed: " + ", ".join(failures)
        )
    return True


def check_transformers_audit_gate(row, args):
    if not getattr(args, "require_transformers_audit", False):
        return True
    status = row.get("transformers_audit_status", "not_requested")
    passed = status == "ok"
    print(
        "transformers_audit_gate "
        f"passed={passed} "
        f"status={status}"
    )
    if not passed:
        detail = row.get("transformers_audit_error") or row.get(
            "transformers_tokenizer_error"
        ) or row.get("transformers_model_error")
        raise RuntimeError(
            f"Transformers audit gate failed: status={status} error={detail}"
        )
    return True


def projection_value_label(value):
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def checkpoint_projector(args):
    fields = checkpoint_projection_fields(args)
    if fields["checkpoint_projection"] != "zspace":
        return None
    topos = st.OpenTopos(
        fields["checkpoint_projection_curvature"],
        1e-5,
        10.0,
        256,
        16384,
    )
    encoder = st.LanguageWaveEncoder(
        topos.curvature(),
        fields["checkpoint_projection_frequency"],
    )
    return ZSpaceProjector(
        topos,
        encoder,
        strength=fields["checkpoint_projection_strength"],
    )


def rule_transform(rule):
    if isinstance(rule, dict):
        return rule.get("transform", "identity")
    return "identity"


def project_tensor_for_rule(tensor, rule, projector):
    if projector is None:
        return tensor
    if _transform_transposes(rule_transform(rule)):
        return projector.forward(tensor.transpose()).transpose()
    return projector.forward(tensor)


def project_checkpoint_tensors(checkpoint, rules, projector):
    if projector is None:
        return checkpoint
    projected = dict(checkpoint)
    for source_name, rule in rules.items():
        if source_name in projected:
            projected[source_name] = project_tensor_for_rule(
                projected[source_name],
                rule,
                projector,
            )
    return projected


def apply_checkpoint_projection(checkpoint, rules, args):
    return project_checkpoint_tensors(checkpoint, rules, checkpoint_projector(args))


def scale_checkpoint_tensor(tensor, gain):
    if gain == 1.0:
        return tensor
    return st.Tensor(
        tensor.rows,
        tensor.cols,
        [value * gain for value in tensor.data()],
    )


def scale_checkpoint_tensors(checkpoint, rules, gain):
    if gain == 1.0:
        return checkpoint
    scaled = dict(checkpoint)
    for source_name in rules:
        if source_name in scaled:
            scaled[source_name] = scale_checkpoint_tensor(scaled[source_name], gain)
    return scaled


def apply_checkpoint_source_gain(checkpoint, rules, args):
    return scale_checkpoint_tensors(
        checkpoint,
        rules,
        checkpoint_source_gain_value(args),
    )


def external_state():
    embed = Linear(VOCAB, HIDDEN, name="embed")
    head = Linear(HIDDEN, TARGET_CLASSES, name="head")
    embed_state = embed.state_dict()
    head_state = head.state_dict()
    raw_state = {
        "model.embed.weight": embed_state["embed::weight"],
        "model.embed.bias": embed_state["embed::bias"],
        "model.lm_head.weight": head_state["head::weight"].transpose(),
        "model.lm_head.bias": [0.1, -0.2, 0.3, -0.4],
        "model.unused.layernorm.weight": [[1.0, 1.0]],
    }
    return checkpoint_from_external_state(raw_state)


def key_rules():
    return {
        "model.embed.weight": "embed::weight",
        "model.embed.bias": "embed::bias",
        "model.lm_head.weight": {
            "target": "head::weight",
            "transform": "transpose",
        },
        "model.lm_head.bias": {
            "target": "head::bias",
            "transform": "copy_overlap_zeros",
        },
    }


def hf_lm_key_rules(
    *,
    embed_weight_key="transformer.wte.weight",
    embed_bias_key="transformer.wte.bias",
    lm_head_weight_key="lm_head.weight",
    lm_head_bias_key="lm_head.bias",
    embed_weight_target="embed::weight",
    embed_bias_target="embed::bias",
    lm_head_weight_target="head::weight",
    lm_head_bias_target="head::bias",
    embed_weight_transform="identity",
    embed_bias_transform="identity",
    lm_head_weight_transform="transpose",
    lm_head_bias_transform="copy_overlap_zeros",
):
    """Build a small HF/PyTorch-style LM checkpoint key map.

    PyTorch Linear weights are usually stored as `(out_features, in_features)`,
    so the LM head defaults to a transpose before loading into SpiralTorch's
    `(in_features, out_features)` linear layout. The head bias defaults to an
    overlap-copy transform to support smaller external vocab heads during early
    adapter smoke tests.
    """
    rules = {}
    if embed_weight_key is not None:
        rules[embed_weight_key] = _rule_value(embed_weight_target, embed_weight_transform)
    if embed_bias_key is not None:
        rules[embed_bias_key] = _rule_value(embed_bias_target, embed_bias_transform)
    if lm_head_weight_key is not None:
        rules[lm_head_weight_key] = {
            "target": lm_head_weight_target,
            "transform": lm_head_weight_transform,
        }
    if lm_head_bias_key is not None:
        rules[lm_head_bias_key] = {
            "target": lm_head_bias_target,
            "transform": lm_head_bias_transform,
        }
    return rules


def hf_lm_key_preset(name):
    if name == AUTO_KEY_PRESET:
        raise ValueError("auto key preset must be resolved from shape metadata first")
    try:
        return dict(HF_KEY_PRESETS[name])
    except KeyError as exc:
        supported = ", ".join(sorted(HF_KEY_PRESETS))
        raise ValueError(f"unsupported HF key preset: {name}; supported={supported}") from exc


def _state_value(state, key):
    try:
        return state[key]
    except KeyError as exc:
        raise KeyError(f"source state is missing required key: {key}") from exc


def hf_lm_checkpoint_from_spiraltorch_state(
    state,
    *,
    embed_weight_source="embed::weight",
    embed_bias_source="embed::bias",
    lm_head_weight_source="head::weight",
    lm_head_bias_source="head::bias",
    embed_bias_cols=None,
    lm_head_bias_cols=None,
    embed_weight_key="transformer.wte.weight",
    embed_bias_key="transformer.wte.bias",
    lm_head_weight_key="lm_head.weight",
    lm_head_bias_key="lm_head.bias",
    transpose_lm_head_weight=True,
):
    """Externalize a SpiralTorch LM-like state dict using HF/PyTorch key names."""
    raw_state = {}
    if embed_weight_key is not None:
        raw_state[embed_weight_key] = _state_value(state, embed_weight_source)
    if embed_bias_key is not None:
        if embed_bias_source is None:
            cols = (
                embed_bias_cols
                if embed_bias_cols is not None
                else _infer_bias_cols(state, embed_weight_source)
            )
            raw_state[embed_bias_key] = zero_row_tensor(cols)
        else:
            raw_state[embed_bias_key] = _state_value(state, embed_bias_source)
    if lm_head_weight_key is not None:
        weight = _state_value(state, lm_head_weight_source)
        raw_state[lm_head_weight_key] = (
            weight.transpose() if transpose_lm_head_weight else weight
        )
    if lm_head_bias_key is not None:
        if lm_head_bias_source is None:
            cols = (
                lm_head_bias_cols
                if lm_head_bias_cols is not None
                else _infer_bias_cols(state, lm_head_weight_source)
            )
            raw_state[lm_head_bias_key] = zero_row_tensor(cols)
        else:
            raw_state[lm_head_bias_key] = _state_value(state, lm_head_bias_source)
    return checkpoint_from_external_state(raw_state)


def hf_lm_handoff_from_spiraltorch_state(state, **kwargs):
    """Return `(checkpoint, key_rules)` for a HF/PyTorch-style LM handoff."""
    key_preset = kwargs.pop("key_preset", None)
    if key_preset is not None:
        for key, value in hf_lm_key_preset(key_preset).items():
            kwargs.setdefault(key, value)
    shared_keys = {
        "embed_weight_key",
        "embed_bias_key",
        "lm_head_weight_key",
        "lm_head_bias_key",
    }
    checkpoint_keys = shared_keys | {
        "embed_weight_source",
        "embed_bias_source",
        "lm_head_weight_source",
        "lm_head_bias_source",
        "embed_bias_cols",
        "lm_head_bias_cols",
        "transpose_lm_head_weight",
    }
    rule_keys = shared_keys | {
        "embed_weight_target",
        "embed_bias_target",
        "lm_head_weight_target",
        "lm_head_bias_target",
        "embed_weight_transform",
        "embed_bias_transform",
        "lm_head_weight_transform",
        "lm_head_bias_transform",
    }
    unknown = sorted(set(kwargs) - checkpoint_keys - rule_keys)
    if unknown:
        raise TypeError(f"unsupported HF LM handoff option(s): {', '.join(unknown)}")

    checkpoint_kwargs = {key: value for key, value in kwargs.items() if key in checkpoint_keys}
    rule_kwargs = {key: value for key, value in kwargs.items() if key in rule_keys}
    explicit_transform = rule_kwargs.get("lm_head_weight_transform")
    if "transpose_lm_head_weight" not in checkpoint_kwargs and explicit_transform is not None:
        checkpoint_kwargs["transpose_lm_head_weight"] = _transform_transposes(
            explicit_transform
        )
    if (
        not checkpoint_kwargs.get("transpose_lm_head_weight", True)
        and "lm_head_weight_transform" not in rule_kwargs
    ):
        rule_kwargs["lm_head_weight_transform"] = "identity"
    rule_transposes = _transform_transposes(
        rule_kwargs.get("lm_head_weight_transform", "transpose")
    )
    if checkpoint_kwargs.get("transpose_lm_head_weight", True) != rule_transposes:
        raise ValueError(
            "transpose_lm_head_weight must agree with lm_head_weight_transform "
            "for a reversible HF LM handoff"
        )

    checkpoint = hf_lm_checkpoint_from_spiraltorch_state(
        state,
        **checkpoint_kwargs,
    )
    rules = hf_lm_key_rules(**rule_kwargs)
    return checkpoint, rules


def _hf_key_kwargs(key_preset, overrides):
    key_kwargs = hf_lm_key_preset(key_preset) if key_preset is not None else {}
    for key in [
        "embed_weight_key",
        "embed_bias_key",
        "lm_head_weight_key",
        "lm_head_bias_key",
    ]:
        if key in overrides:
            key_kwargs[key] = overrides.pop(key)
    return key_kwargs


def hf_lm_handoff_from_external_state(
    state,
    *,
    key_preset="gpt2",
    synthesize_missing_biases=True,
    tie_missing_lm_head_weight=True,
    include_extra_keys=None,
    **kwargs,
):
    """Return `(checkpoint, key_rules)` from a HF/PyTorch-style external state."""
    key_kwargs = _hf_key_kwargs(key_preset, kwargs)
    rule_keys = {
        "embed_weight_target",
        "embed_bias_target",
        "lm_head_weight_target",
        "lm_head_bias_target",
        "embed_weight_transform",
        "embed_bias_transform",
        "lm_head_weight_transform",
        "lm_head_bias_transform",
    }
    unknown = sorted(set(kwargs) - rule_keys)
    if unknown:
        raise TypeError(f"unsupported external HF LM handoff option(s): {', '.join(unknown)}")
    rule_kwargs = {key: value for key, value in kwargs.items() if key in rule_keys}
    rules = hf_lm_key_rules(**key_kwargs, **rule_kwargs)

    lm_head_transform = rule_kwargs.get("lm_head_weight_transform", "transpose")
    raw_state = {}
    embed_weight_key = key_kwargs.get("embed_weight_key")
    lm_head_weight_key = key_kwargs.get("lm_head_weight_key")
    if embed_weight_key is not None:
        raw_state[embed_weight_key] = _state_value(state, embed_weight_key)
    if lm_head_weight_key is not None:
        if lm_head_weight_key in state:
            raw_state[lm_head_weight_key] = state[lm_head_weight_key]
        elif tie_missing_lm_head_weight and embed_weight_key in state:
            raw_state[lm_head_weight_key] = state[embed_weight_key]
        else:
            raw_state[lm_head_weight_key] = _state_value(state, lm_head_weight_key)

    embed_bias_key = key_kwargs.get("embed_bias_key")
    if embed_bias_key is not None:
        if embed_bias_key in state:
            raw_state[embed_bias_key] = state[embed_bias_key]
        elif synthesize_missing_biases:
            raw_state[embed_bias_key] = zero_row_tensor(
                _infer_bias_cols(state, key_kwargs["embed_weight_key"])
            )
        else:
            raw_state[embed_bias_key] = _state_value(state, embed_bias_key)

    lm_head_bias_key = key_kwargs.get("lm_head_bias_key")
    if lm_head_bias_key is not None:
        if lm_head_bias_key in state:
            raw_state[lm_head_bias_key] = state[lm_head_bias_key]
        elif synthesize_missing_biases:
            raw_state[lm_head_bias_key] = zero_row_tensor(
                _infer_lm_head_bias_cols(
                    raw_state,
                    key_kwargs["lm_head_weight_key"],
                    transform=lm_head_transform,
                )
            )
        else:
            raw_state[lm_head_bias_key] = _state_value(state, lm_head_bias_key)

    for key in include_extra_keys or []:
        raw_state[key] = _state_value(state, key)
    return checkpoint_from_external_state(raw_state), rules


def hf_lm_state_keys(key_preset="gpt2", *, include_extra_keys=None, **overrides):
    key_kwargs = _hf_key_kwargs(key_preset, dict(overrides))
    keys = {
        key
        for key in [
            key_kwargs.get("embed_weight_key"),
            key_kwargs.get("embed_bias_key"),
            key_kwargs.get("lm_head_weight_key"),
            key_kwargs.get("lm_head_bias_key"),
        ]
        if key is not None
    }
    keys.update(include_extra_keys or [])
    return keys


def all_hf_lm_state_keys(*, include_extra_keys=None):
    keys = set()
    for preset in sorted(HF_KEY_PRESETS):
        keys.update(hf_lm_state_keys(preset))
    keys.update(include_extra_keys or [])
    return keys


def hf_lm_shape_include_keys(key_preset, *, include_extra_keys=None):
    if key_preset == AUTO_KEY_PRESET:
        return all_hf_lm_state_keys(include_extra_keys=include_extra_keys)
    return hf_lm_state_keys(key_preset, include_extra_keys=include_extra_keys)


def detect_hf_lm_key_preset(shape_state):
    full_matches = []
    embed_only_matches = []
    for preset, key_kwargs in sorted(HF_KEY_PRESETS.items()):
        embed_present = key_kwargs["embed_weight_key"] in shape_state
        head_present = key_kwargs["lm_head_weight_key"] in shape_state
        if embed_present and head_present:
            full_matches.append(preset)
        elif embed_present:
            embed_only_matches.append(preset)
    matches = full_matches or embed_only_matches
    if not matches:
        supported = ", ".join(sorted(HF_KEY_PRESETS))
        available = ", ".join(sorted(shape_state)) or "none"
        raise ValueError(
            "could not auto-detect HF key preset; "
            f"supported={supported}; available_keys={available}"
        )
    if len(matches) > 1:
        raise ValueError(
            "ambiguous HF key preset auto-detect: "
            f"{', '.join(matches)}; pass --key-preset explicitly"
        )
    return matches[0]


def resolve_hf_lm_key_preset(key_preset, shape_state):
    if key_preset == AUTO_KEY_PRESET:
        return detect_hf_lm_key_preset(shape_state)
    return key_preset


def hf_lm_uses_tied_head_weight(state, *, key_preset="gpt2", **overrides):
    key_kwargs = _hf_key_kwargs(key_preset, dict(overrides))
    embed_key = key_kwargs["embed_weight_key"]
    head_key = key_kwargs["lm_head_weight_key"]
    return embed_key in state and head_key not in state


def _unwrap_torch_state_dict(payload):
    if isinstance(payload, Mapping):
        for nested_key in ["state_dict", "model"]:
            nested = payload.get(nested_key)
            if isinstance(nested, Mapping):
                return nested
        return payload
    raise TypeError(f"torch checkpoint payload is not a state dict: {type(payload)!r}")


def _filter_loaded_state_dict(state, include_keys, *, tensor_bounds=None):
    if include_keys is None:
        filtered = dict(state)
    else:
        include_keys = set(include_keys)
        filtered = {key: value for key, value in state.items() if key in include_keys}
    return bound_external_state_tensors(filtered, tensor_bounds)


def _safetensors_bounded_tensor(handle, name, bound):
    if bound is None:
        return handle.get_tensor(name)
    rows, cols = bound
    try:
        tensor_slice = handle.get_slice(name)
        shape = tuple(int(dim) for dim in tensor_slice.get_shape())
        if len(shape) == 1:
            return tensor_slice[: min(shape[0], cols)]
        if len(shape) == 2:
            return tensor_slice[: min(shape[0], rows), : min(shape[1], cols)]
        raise ValueError(f"{name} must be 1D or 2D to slice, got shape={shape}")
    except AttributeError:
        return slice_external_tensor(
            handle.get_tensor(name),
            rows=rows,
            cols=cols,
            name=name,
        )


def _load_torch_state_dict_file(path, *, include_keys=None, tensor_bounds=None):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "loading torch .bin/.pt state dicts requires PyTorch to be installed"
        ) from exc
    payload = torch.load(path, map_location="cpu")
    return _filter_loaded_state_dict(
        _unwrap_torch_state_dict(payload),
        include_keys,
        tensor_bounds=tensor_bounds,
    )


def _load_torch_state_dict_file_shapes(path, *, include_keys=None):
    state = _load_torch_state_dict_file(path, include_keys=include_keys)
    return {key: _tensor_shape(value, key) for key, value in state.items()}


def _load_safetensors_state_dict_file(path, *, include_keys=None, tensor_bounds=None):
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "loading .safetensors state dicts requires safetensors to be installed"
        ) from exc
    state = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        names = available if include_keys is None else available & set(include_keys)
        tensor_bounds = tensor_bounds or {}
        for name in sorted(names):
            state[name] = _safetensors_bounded_tensor(
                handle,
                name,
                tensor_bounds.get(name),
            )
    return state


def _safetensors_shape(handle, name):
    try:
        tensor_slice = handle.get_slice(name)
        return tuple(int(dim) for dim in tensor_slice.get_shape())
    except AttributeError:
        return _external_shape(handle.get_tensor(name), name)


def _load_safetensors_state_dict_file_shapes(path, *, include_keys=None):
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "loading .safetensors state dict shapes requires safetensors to be installed"
        ) from exc
    state = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        names = available if include_keys is None else available & set(include_keys)
        for name in sorted(names):
            state[name] = _safetensors_shape(handle, name)
    return state


def _load_hf_state_dict_file(path, *, include_keys=None, tensor_bounds=None):
    if path.suffix == ".safetensors":
        return _load_safetensors_state_dict_file(
            path,
            include_keys=include_keys,
            tensor_bounds=tensor_bounds,
        )
    return _load_torch_state_dict_file(
        path,
        include_keys=include_keys,
        tensor_bounds=tensor_bounds,
    )


def _load_hf_state_dict_file_shapes(path, *, include_keys=None):
    if path.suffix == ".safetensors":
        return _load_safetensors_state_dict_file_shapes(path, include_keys=include_keys)
    return _load_torch_state_dict_file_shapes(path, include_keys=include_keys)


def _hf_index_path(directory):
    for name in ["model.safetensors.index.json", "pytorch_model.bin.index.json"]:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def _load_hf_state_dict_from_index(
    directory,
    index_path,
    *,
    include_keys=None,
    tensor_bounds=None,
):
    with index_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise ValueError(f"{index_path} does not contain a HF weight_map")
    selected_keys = set(weight_map) if include_keys is None else set(include_keys)
    shards = sorted({weight_map[key] for key in selected_keys if key in weight_map})
    state = {}
    loaded_files = []
    for shard in shards:
        shard_path = directory / shard
        state.update(
            _load_hf_state_dict_file(
                shard_path,
                include_keys=selected_keys,
                tensor_bounds=tensor_bounds,
            )
        )
        loaded_files.append(str(shard_path))
    return state, loaded_files


def _load_hf_state_dict_shapes_from_index(directory, index_path, *, include_keys=None):
    with index_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise ValueError(f"{index_path} does not contain a HF weight_map")
    selected_keys = set(weight_map) if include_keys is None else set(include_keys)
    shards = sorted({weight_map[key] for key in selected_keys if key in weight_map})
    state = {}
    loaded_files = []
    for shard in shards:
        shard_path = directory / shard
        state.update(
            _load_hf_state_dict_file_shapes(shard_path, include_keys=selected_keys)
        )
        loaded_files.append(str(shard_path))
    return state, loaded_files


def load_hf_state_dict(path, *, include_keys=None, tensor_bounds=None):
    """Load a local HF/PyTorch state dict, optionally keeping only selected keys."""
    path = Path(path)
    if path.is_dir():
        index_path = _hf_index_path(path)
        if index_path is not None:
            return _load_hf_state_dict_from_index(
                path,
                index_path,
                include_keys=include_keys,
                tensor_bounds=tensor_bounds,
            )
        for name in ["model.safetensors", "pytorch_model.bin", "pytorch_model.pt"]:
            candidate = path / name
            if candidate.exists():
                state = _load_hf_state_dict_file(
                    candidate,
                    include_keys=include_keys,
                    tensor_bounds=tensor_bounds,
                )
                return state, [str(candidate)]
        raise FileNotFoundError(
            f"{path} has no supported HF state dict file or index"
        )
    state = _load_hf_state_dict_file(
        path,
        include_keys=include_keys,
        tensor_bounds=tensor_bounds,
    )
    return state, [str(path)]


def load_hf_state_dict_shapes(path, *, include_keys=None):
    """Load only key/shape metadata where possible for local HF state dicts."""
    path = Path(path)
    if path.is_dir():
        index_path = _hf_index_path(path)
        if index_path is not None:
            return _load_hf_state_dict_shapes_from_index(
                path,
                index_path,
                include_keys=include_keys,
            )
        for name in ["model.safetensors", "pytorch_model.bin", "pytorch_model.pt"]:
            candidate = path / name
            if candidate.exists():
                state = _load_hf_state_dict_file_shapes(
                    candidate,
                    include_keys=include_keys,
                )
                return state, [str(candidate)]
        raise FileNotFoundError(
            f"{path} has no supported HF state dict file or index"
        )
    state = _load_hf_state_dict_file_shapes(path, include_keys=include_keys)
    return state, [str(path)]


def infer_hf_lm_module_shapes(
    state,
    *,
    key_preset="gpt2",
    lm_head_weight_transform="transpose",
    tie_missing_lm_head_weight=True,
    **overrides,
):
    key_kwargs = _hf_key_kwargs(key_preset, dict(overrides))
    embed_key = key_kwargs["embed_weight_key"]
    head_key = key_kwargs["lm_head_weight_key"]
    vocab, hidden = _tensor_shape(_state_value(state, embed_key), embed_key)
    if head_key in state:
        head_rows, head_cols = _tensor_shape(_state_value(state, head_key), head_key)
    elif tie_missing_lm_head_weight:
        head_rows, head_cols = vocab, hidden
    else:
        head_rows, head_cols = _tensor_shape(_state_value(state, head_key), head_key)
    if _transform_transposes(lm_head_weight_transform):
        head_hidden, target_classes = head_cols, head_rows
    else:
        head_hidden, target_classes = head_rows, head_cols
    if hidden != head_hidden:
        raise ValueError(
            f"embed hidden={hidden} does not match adapted head hidden={head_hidden}"
        )
    return vocab, hidden, target_classes


def hf_style_external_state():
    return checkpoint_from_external_state(hf_style_external_state_rows())


def hf_style_external_state_rows():
    embed = Linear(VOCAB, HIDDEN, name="embed")
    head = Linear(HIDDEN, TARGET_CLASSES, name="head")
    embed_state = embed.state_dict()
    head_state = head.state_dict()
    return {
        "transformer.wte.weight": _tensor_rows(embed_state["embed::weight"]),
        "transformer.wte.bias": embed_state["embed::bias"].data(),
        "lm_head.weight": _tensor_rows(head_state["head::weight"].transpose()),
        "lm_head.bias": [0.1, -0.2, 0.3, -0.4],
        "transformer.h.0.ln_1.weight": [[1.0, 1.0]],
    }


def hf_no_bias_external_state():
    return hf_preset_external_state("gpt2", synthesize_bias=True)


def hf_preset_external_state(preset, *, synthesize_bias):
    embed = Linear(VOCAB, HIDDEN, name="embed")
    head = Linear(HIDDEN, TARGET_CLASSES, name="head")
    state = {}
    state.update(embed.state_dict())
    state.update(head.state_dict())
    key_kwargs = hf_lm_key_preset(preset)
    bias_kwargs = (
        {
            "embed_bias_source": None,
            "lm_head_bias_source": None,
        }
        if synthesize_bias
        else {}
    )
    checkpoint = hf_lm_checkpoint_from_spiraltorch_state(
        state,
        **key_kwargs,
        **bias_kwargs,
    )
    checkpoint.update(
        checkpoint_from_external_state(
            {
                HF_UNUSED_KEYS.get(preset, f"{preset}.unused.norm.weight"): [[1.0, 1.0]],
            }
        )
    )
    return checkpoint


def checkpoint_source(source):
    if source == "toy":
        return external_state(), key_rules()
    if source == "hf-style":
        return hf_style_external_state(), hf_lm_key_rules()
    if source == "hf-no-bias":
        return hf_no_bias_external_state(), hf_lm_key_rules()
    if source == "hf-llama":
        preset = hf_lm_key_preset("llama")
        return (
            hf_preset_external_state("llama", synthesize_bias=True),
            hf_lm_key_rules(**preset),
        )
    if source == "hf-gpt-neox":
        preset = hf_lm_key_preset("gpt_neox")
        return (
            hf_preset_external_state("gpt_neox", synthesize_bias=True),
            hf_lm_key_rules(**preset),
        )
    raise ValueError(f"unknown checkpoint source: {source}")


def shape_label(value):
    if value is None:
        return "none"
    return f"{value[0]}x{value[1]}"


def report_audit_signature(report):
    tokens = []
    for entry in report["entries"]:
        tokens.append(
            ":".join(
                [
                    str(entry["name"]),
                    str(entry["status"]),
                    str(entry["source_name"]),
                    str(entry["transform"]),
                    shape_label(entry["expected_shape"]),
                    shape_label(entry["source_shape"]),
                    shape_label(entry["original_source_shape"]),
                ]
            )
        )
    return "|".join(sorted(tokens))


def checkpoint_audit_fields(prefix, report, load=None):
    fields = {
        f"{prefix}_preflight_matched": report["matched"],
        f"{prefix}_preflight_extra": report["extra"],
        f"{prefix}_preflight_source_hash": report["source"]["hash"],
        f"{prefix}_preflight_matched_subset_hash": report["matched_subset"]["hash"],
        f"{prefix}_preflight_signature": report_audit_signature(report),
    }
    if load is not None:
        fields.update(
            {
                f"{prefix}_load_matched": load["matched"],
                f"{prefix}_load_source_hash": load["source"]["hash"],
                f"{prefix}_load_loaded_hash": load["loaded"]["hash"],
            }
        )
    return fields


def preflight_context_fields(args, source_label, loaded_files, module_shapes):
    vocab, hidden, target_classes = module_shapes
    fields = {
        "checkpoint_source": source_label,
        "checkpoint_loaded_files": len(loaded_files),
        "checkpoint_vocab": vocab,
        "checkpoint_hidden": hidden,
        "checkpoint_target_classes": target_classes,
        "checkpoint_overlap_resize": bool(getattr(args, "allow_overlap_resize", False)),
    }
    fields.update(checkpoint_projection_fields(args))
    fields.update(checkpoint_source_gain_fields(args))
    return fields


def shape_tuple_label(value):
    if value is None:
        return "none"
    return "x".join(str(dim) for dim in value)


def _csv_or_none(values):
    values = list(values)
    return ",".join(values) if values else "none"


def _resolved_shape_key_preset(args, shape_state):
    return resolve_hf_lm_key_preset(args.key_preset, shape_state)


def _required_hf_lm_keys(args, key_preset, shape_state=None):
    key_kwargs = hf_lm_key_preset(key_preset)
    required = [key_kwargs["embed_weight_key"]]
    tied_head = (
        shape_state is not None
        and hf_lm_uses_tied_head_weight(shape_state, key_preset=key_preset)
    )
    if not tied_head:
        required.append(key_kwargs["lm_head_weight_key"])
    if args.no_synthesize_missing_biases:
        required.extend(
            [
                key_kwargs["embed_bias_key"],
                key_kwargs["lm_head_bias_key"],
            ]
        )
    return [key for key in required if key is not None]


def hf_lm_shape_audit_row(args, source_label, loaded_files, shape_state):
    key_preset = _resolved_shape_key_preset(args, shape_state)
    key_kwargs = hf_lm_key_preset(key_preset)
    tied_head = hf_lm_uses_tied_head_weight(shape_state, key_preset=key_preset)
    inferred_shapes = infer_hf_lm_module_shapes(
        shape_state,
        key_preset=key_preset,
    )
    requested_shapes = resolved_module_shapes(args, inferred_shapes)
    required_keys = _required_hf_lm_keys(args, key_preset, shape_state)
    missing_required = [key for key in required_keys if key not in shape_state]
    extra_keys = list(args.include_extra_keys)
    missing_extra = [key for key in extra_keys if key not in shape_state]
    present_extra = [key for key in extra_keys if key in shape_state]
    projection_fields = checkpoint_projection_fields(args)
    source_gain_fields = checkpoint_source_gain_fields(args)
    exact_shape_match = inferred_shapes == requested_shapes
    can_materialize_requested = exact_shape_match or args.allow_overlap_resize
    row = {
        "row_type": "shape_audit",
        "checkpoint_source": source_label,
        "checkpoint_loaded_files": len(loaded_files),
        "checkpoint_loaded_file_paths": ",".join(loaded_files),
        "requested_key_preset": args.key_preset,
        "checkpoint_key_preset": key_preset,
        "checkpoint_vocab": inferred_shapes[0],
        "checkpoint_hidden": inferred_shapes[1],
        "checkpoint_target_classes": inferred_shapes[2],
        "requested_vocab": requested_shapes[0],
        "requested_hidden": requested_shapes[1],
        "requested_target_classes": requested_shapes[2],
        "exact_shape_match": exact_shape_match,
        "overlap_resize_required": not exact_shape_match,
        "overlap_resize_allowed": bool(args.allow_overlap_resize),
        "can_materialize_requested": can_materialize_requested,
        "required_keys": _csv_or_none(required_keys),
        "missing_required_keys": _csv_or_none(missing_required),
        "extra_audit_keys": _csv_or_none(extra_keys),
        "present_extra_keys": _csv_or_none(present_extra),
        "missing_extra_keys": _csv_or_none(missing_extra),
        "embed_weight_key": key_kwargs["embed_weight_key"],
        "embed_weight_shape": shape_tuple_label(
            shape_state.get(key_kwargs["embed_weight_key"])
        ),
        "embed_bias_key": key_kwargs["embed_bias_key"],
        "embed_bias_shape": shape_tuple_label(
            shape_state.get(key_kwargs["embed_bias_key"])
        ),
        "embed_bias_synthesized": key_kwargs["embed_bias_key"] not in shape_state
        and not args.no_synthesize_missing_biases,
        "lm_head_weight_key": key_kwargs["lm_head_weight_key"],
        "lm_head_weight_shape": shape_tuple_label(
            shape_state.get(
                key_kwargs["lm_head_weight_key"],
                shape_state.get(key_kwargs["embed_weight_key"]) if tied_head else None,
            )
        ),
        "lm_head_weight_synthesized_from_embed": tied_head,
        "lm_head_bias_key": key_kwargs["lm_head_bias_key"],
        "lm_head_bias_shape": shape_tuple_label(
            shape_state.get(key_kwargs["lm_head_bias_key"])
        ),
        "lm_head_bias_synthesized": key_kwargs["lm_head_bias_key"] not in shape_state
        and not args.no_synthesize_missing_biases,
        **projection_fields,
        **source_gain_fields,
    }
    row.update(transformers_runtime_audit_fields(args, inferred_shapes))
    return row


def print_shape_audit(row):
    print(
        "checkpoint_shape_audit "
        f"status=ok "
        f"source={row['checkpoint_source']} "
        f"loaded_files={row['checkpoint_loaded_files']} "
        f"requested_key_preset={row['requested_key_preset']} "
        f"key_preset={row['checkpoint_key_preset']} "
        f"checkpoint_shape={row['checkpoint_vocab']}x{row['checkpoint_hidden']}x{row['checkpoint_target_classes']} "
        f"requested_shape={row['requested_vocab']}x{row['requested_hidden']}x{row['requested_target_classes']} "
        f"exact_shape_match={row['exact_shape_match']} "
        f"overlap_resize_required={row['overlap_resize_required']} "
        f"overlap_resize_allowed={row['overlap_resize_allowed']} "
        f"can_materialize_requested={row['can_materialize_requested']} "
        f"missing_required_keys={row['missing_required_keys']} "
        f"missing_extra_keys={row['missing_extra_keys']} "
        f"lm_head_weight_synthesized_from_embed={row['lm_head_weight_synthesized_from_embed']} "
        f"checkpoint_projection={row['checkpoint_projection']} "
        f"checkpoint_projection_strength={projection_value_label(row['checkpoint_projection_strength'])} "
        f"checkpoint_projection_curvature={projection_value_label(row['checkpoint_projection_curvature'])} "
        f"checkpoint_projection_frequency={projection_value_label(row['checkpoint_projection_frequency'])} "
        f"checkpoint_source_gain={row['checkpoint_source_gain']:.6f}"
    )
    print_transformers_audit(row)


def shape_audit_gate_failures(row, args):
    failures = []
    if getattr(args, "require_shape_materializable", False) and not row[
        "can_materialize_requested"
    ]:
        failures.append("requested shape is not materializable without --allow-overlap-resize")
    if getattr(args, "require_exact_shape_match", False) and not row["exact_shape_match"]:
        failures.append("checkpoint/requested shapes do not match exactly")
    required_key_preset = getattr(args, "require_detected_key_preset", None)
    if required_key_preset is not None and row["checkpoint_key_preset"] != required_key_preset:
        failures.append(
            "detected key preset "
            f"{row['checkpoint_key_preset']} != required {required_key_preset}"
        )
    return failures


def shape_audit_gate_requested(args):
    return (
        getattr(args, "require_shape_materializable", False)
        or getattr(args, "require_exact_shape_match", False)
        or getattr(args, "require_detected_key_preset", None) is not None
    )


def check_shape_audit_gates(row, args):
    failures = shape_audit_gate_failures(row, args)
    print(
        "shape_audit_gate "
        f"passed={not failures} "
        f"failures={_csv_or_none(failures)}"
    )
    if failures:
        raise RuntimeError(
            "checkpoint shape audit gate failed: " + "; ".join(failures)
        )
    return True


def flatten_report(label, report, context_fields=None):
    report_row = {
        "row_type": "report",
        "label": label,
        "compatible": report["compatible"],
        "matched": report["matched"],
        "missing": report["missing"],
        "shape_mismatched": report["shape_mismatched"],
        "extra": report["extra"],
        "source_hash": report["source"]["hash"],
        "matched_subset_hash": report["matched_subset"]["hash"],
    }
    if context_fields is not None:
        report_row.update(context_fields)
    rows = [report_row]
    for entry in report["entries"]:
        rows.append(
            {
                "row_type": "entry",
                "label": label,
                "name": entry["name"],
                "status": entry["status"],
                "source_name": entry["source_name"],
                "transform": entry["transform"],
                "expected_shape": shape_label(entry["expected_shape"]),
                "source_shape": shape_label(entry["source_shape"]),
                "original_source_shape": shape_label(entry["original_source_shape"]),
            }
        )
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} invalid JSONL row: {exc}") from exc
    if not rows:
        raise ValueError(f"{path} did not contain any preflight rows")
    return rows


def preflight_row_key(row):
    row_type = row.get("row_type")
    label = row.get("label")
    if row_type == "report":
        return (row_type, label)
    if row_type == "entry":
        return (row_type, label, row.get("name"))
    raise ValueError(f"unsupported preflight row type: {row_type!r}")


def preflight_row_key_label(key):
    return "::".join("none" if part is None else str(part) for part in key)


def preflight_rows_by_key(rows, label):
    by_key = {}
    for row in rows:
        key = preflight_row_key(row)
        if key in by_key:
            raise ValueError(
                f"{label} has duplicate preflight row key: {preflight_row_key_label(key)}"
            )
        by_key[key] = row
    return by_key


def compare_preflight_rows(current_rows, baseline_rows):
    current = preflight_rows_by_key(current_rows, "current")
    baseline = preflight_rows_by_key(baseline_rows, "baseline")
    differences = []
    missing_field = "<missing>"
    for key in sorted(set(baseline) - set(current)):
        differences.append(
            {
                "kind": "missing",
                "key": key,
                "before": baseline[key],
                "after": None,
            }
        )
    for key in sorted(set(current) - set(baseline)):
        differences.append(
            {
                "kind": "extra",
                "key": key,
                "before": None,
                "after": current[key],
            }
        )
    for key in sorted(set(current) & set(baseline)):
        before = baseline[key]
        after = current[key]
        for field in sorted(set(before) | set(after)):
            before_has = field in before
            after_has = field in after
            before_value = before.get(field) if before_has else missing_field
            after_value = after.get(field) if after_has else missing_field
            if before_value != after_value:
                differences.append(
                    {
                        "kind": "changed",
                        "key": key,
                        "field": field,
                        "before": before_value,
                        "after": after_value,
                    }
                )
    return differences


def print_preflight_comparison(current_rows, baseline_rows, differences, baseline_path):
    print(
        "preflight_compare "
        f"baseline={baseline_path} "
        f"rows_before={len(baseline_rows)} "
        f"rows_after={len(current_rows)} "
        f"differences={len(differences)} "
        f"passed={not differences}"
    )
    for diff in differences:
        key = preflight_row_key_label(diff["key"])
        if diff["kind"] == "changed":
            print(
                "preflight_compare_diff "
                f"kind=changed key={key} field={diff['field']} "
                f"before={diff['before']!r} after={diff['after']!r}"
            )
        else:
            print(f"preflight_compare_diff kind={diff['kind']} key={key}")


def require_report(label, report):
    if not report["compatible"]:
        raise RuntimeError(
            f"{label} checkpoint preflight failed: "
            f"missing={report['missing']} shape_mismatched={report['shape_mismatched']}"
        )


def module_preflight_report(module, checkpoint, rules, *, lora_base=False):
    if lora_base:
        return module.base_state_dict_compatibility_with_key_map(checkpoint, rules)
    return module.state_dict_compatibility_with_key_map(checkpoint, rules)


def module_preflight_load(module, checkpoint, rules, *, lora_base=False):
    if lora_base:
        return module.load_base_from_state_dict_mapped(checkpoint, rules)
    return module.load_state_dict_subset_mapped_checked(checkpoint, rules)


def preflight_and_load(label, module, checkpoint, rules, *, lora_base=False, emit=True):
    report = module_preflight_report(
        module,
        checkpoint,
        rules,
        lora_base=lora_base,
    )
    if emit:
        print_report(label, report)
    require_report(label, report)
    load = module_preflight_load(
        module,
        checkpoint,
        rules,
        lora_base=lora_base,
    )
    if not load["matched"]:
        raise RuntimeError(f"{label} adapted checkpoint load mismatch: {load}")
    return report, load


def print_report(label, report):
    print(
        f"preflight_report label={label} "
        f"compatible={report['compatible']} "
        f"matched={report['matched']} "
        f"missing={report['missing']} "
        f"shape_mismatched={report['shape_mismatched']} "
        f"extra={report['extra']} "
        f"source_hash={report['source']['hash']} "
        f"matched_subset_hash={report['matched_subset']['hash']}"
    )
    for entry in report["entries"]:
        print(
            f"preflight_entry label={label} "
            f"name={entry['name']} "
            f"status={entry['status']} "
            f"source_name={entry['source_name']} "
            f"transform={entry['transform']} "
            f"expected_shape={shape_label(entry['expected_shape'])} "
            f"source_shape={shape_label(entry['source_shape'])} "
            f"original_source_shape={shape_label(entry['original_source_shape'])}"
        )


def resolved_checkpoint_source(args):
    if args.hf_state_dict is None:
        checkpoint, rules = checkpoint_source(args.source)
        return checkpoint, rules, args.source, [], (VOCAB, HIDDEN, TARGET_CLASSES)

    shape_include_keys = hf_lm_shape_include_keys(
        args.key_preset,
        include_extra_keys=args.include_extra_keys,
    )
    shape_state, _ = load_hf_state_dict_shapes(
        args.hf_state_dict,
        include_keys=shape_include_keys,
    )
    key_preset = resolve_hf_lm_key_preset(args.key_preset, shape_state)
    include_keys = hf_lm_state_keys(
        key_preset,
        include_extra_keys=args.include_extra_keys,
    )
    inferred_shapes = infer_hf_lm_module_shapes(
        shape_state,
        key_preset=key_preset,
    )
    target_shapes = resolved_module_shapes(args, inferred_shapes)
    resize_kwargs = hf_lm_overlap_resize_kwargs() if args.allow_overlap_resize else {}
    tensor_bounds = (
        hf_lm_tensor_bounds_for_module_shapes(
            target_shapes,
            key_preset=key_preset,
            lm_head_weight_transform=resize_kwargs.get(
                "lm_head_weight_transform",
                "transpose",
            ),
        )
        if args.allow_overlap_resize
        else None
    )
    external_state, loaded_files = load_hf_state_dict(
        args.hf_state_dict,
        include_keys=include_keys,
        tensor_bounds=tensor_bounds,
    )
    checkpoint, rules = hf_lm_handoff_from_external_state(
        external_state,
        key_preset=key_preset,
        synthesize_missing_biases=not args.no_synthesize_missing_biases,
        include_extra_keys=args.include_extra_keys,
        **resize_kwargs,
    )
    return (
        checkpoint,
        rules,
        f"hf-state-dict:{key_preset}",
        loaded_files,
        inferred_shapes,
    )


def resolved_module_shapes(args, inferred_shapes):
    vocab, hidden, target_classes = inferred_shapes
    return (
        args.vocab if args.vocab is not None else vocab,
        args.hidden if args.hidden is not None else hidden,
        args.target_classes if args.target_classes is not None else target_classes,
    )


def main():
    args = parse_args()
    if args.shape_only:
        include_keys = hf_lm_shape_include_keys(
            args.key_preset,
            include_extra_keys=args.include_extra_keys,
        )
        shape_state, loaded_files = load_hf_state_dict_shapes(
            args.hf_state_dict,
            include_keys=include_keys,
        )
        row = hf_lm_shape_audit_row(
            args,
            f"hf-state-dict:{args.key_preset}",
            loaded_files,
            shape_state,
        )
        print_shape_audit(row)
        if args.jsonl is not None:
            write_jsonl(args.jsonl, [row])
            print(f"shape_audit_jsonl={args.jsonl} rows=1")
        if shape_audit_gate_requested(args):
            check_shape_audit_gates(row, args)
        check_transformers_runtime_import_gate(row, args)
        check_transformers_audit_gate(row, args)
        return

    checkpoint, rules, source_label, loaded_files, inferred_shapes = (
        resolved_checkpoint_source(args)
    )
    vocab, hidden, target_classes = resolved_module_shapes(args, inferred_shapes)
    checkpoint = apply_checkpoint_projection(checkpoint, rules, args)
    checkpoint = apply_checkpoint_source_gain(checkpoint, rules, args)
    context_fields = preflight_context_fields(
        args,
        source_label,
        loaded_files,
        (vocab, hidden, target_classes),
    )
    context_fields.update(transformers_runtime_audit_fields(args, inferred_shapes))
    print_transformers_audit(context_fields)
    check_transformers_runtime_import_gate(context_fields, args)
    check_transformers_audit_gate(context_fields, args)

    embed = Linear(vocab, hidden, name="embed")
    embed_report, embed_load = preflight_and_load(
        "embed",
        embed,
        checkpoint,
        rules,
    )

    head = LoraLinear(hidden, target_classes, 2, alpha=8.0, name="head")
    head_report, head_load = preflight_and_load(
        "lora_head_base",
        head,
        checkpoint,
        rules,
        lora_base=True,
    )

    rows = flatten_report("embed", embed_report, context_fields)
    rows.extend(flatten_report("lora_head_base", head_report, context_fields))
    if args.jsonl is not None:
        write_jsonl(args.jsonl, rows)
        print(f"preflight_jsonl={args.jsonl} rows={len(rows)}")
    if args.compare_jsonl is not None:
        baseline_rows = load_jsonl(args.compare_jsonl)
        differences = compare_preflight_rows(rows, baseline_rows)
        print_preflight_comparison(rows, baseline_rows, differences, args.compare_jsonl)
        if differences and args.require_preflight_match:
            raise RuntimeError(
                "checkpoint preflight regression gate failed: "
                f"{len(differences)} row/field difference(s)"
            )
    print(
        f"checkpoint_preflight status=ok "
        f"source={source_label} "
        f"loaded_files={len(loaded_files)} "
        f"external_keys={len(checkpoint)} "
        f"vocab={vocab} "
        f"hidden={hidden} "
        f"target_classes={target_classes} "
        f"checkpoint_projection={context_fields['checkpoint_projection']} "
        f"checkpoint_projection_strength={projection_value_label(context_fields['checkpoint_projection_strength'])} "
        f"checkpoint_projection_curvature={projection_value_label(context_fields['checkpoint_projection_curvature'])} "
        f"checkpoint_projection_frequency={projection_value_label(context_fields['checkpoint_projection_frequency'])} "
        f"checkpoint_source_gain={context_fields['checkpoint_source_gain']:.6f} "
        f"embed_loaded={embed_load['matched']} "
        f"head_loaded={head_load['matched']}"
    )


if __name__ == "__main__":
    main()
