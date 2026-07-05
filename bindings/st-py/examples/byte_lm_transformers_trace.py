import argparse
import importlib
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from spiraltorch.runtime_imports import (
    TRANSFORMERS_TRACE_RUNTIME_IMPORT_PRESETS,
    module_file,
    module_name,
    module_version,
    runtime_import_names_from_args,
    runtime_import_probe_fields,
    runtime_import_presets_from_args,
    runtime_import_requirement_failures,
)

import spiraltorch as st
from spiraltorch.ecosystem import (
    external_tensor_last_token,
    external_tensor_metadata,
    external_tensor_shape,
    external_tensor_to_list,
)
from spiraltorch.nn import ZSpaceProjector


DEFAULT_PROMPT = "SpiralTorch routes meaning through Z-space."
ZSPACE_CURVATURE = -0.04
ZSPACE_FREQUENCY = 0.65
ZSPACE_STRENGTH = 1.0
RUNTIME_IMPORT_PRESETS = dict(TRANSFORMERS_TRACE_RUNTIME_IMPORT_PRESETS)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Trace local Transformers next-token logits into JSONL evidence, "
            "optionally attaching a bounded Z-space projection probe."
        )
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Local Transformers model/config/tokenizer directory or model id.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Prompt to trace. May be repeated. Defaults to a small SpiralTorch prompt.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional UTF-8 text file; non-empty lines are appended as prompts.",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Optional JSONL output path for manifest and per-prompt trace rows.",
    )
    parser.add_argument(
        "--compare-jsonl",
        type=Path,
        default=None,
        help="Optional previous Transformers trace JSONL to compare against.",
    )
    parser.add_argument(
        "--compare-output-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL output path for trace comparison rows.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of next-token candidates to keep in each prompt trace.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Transformers revision forwarded to from_pretrained(...).",
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Allow Transformers to resolve remote files. Defaults to local_files_only.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code=True to Transformers loaders.",
    )
    parser.add_argument(
        "--runtime-import",
        dest="runtime_imports",
        action="append",
        default=[],
        help=(
            "Additional Python module to import while SpiralTorch and "
            "Transformers are loaded. May be repeated; results are recorded "
            "in the trace manifest."
        ),
    )
    parser.add_argument(
        "--runtime-import-preset",
        dest="runtime_import_presets",
        action="append",
        choices=sorted(RUNTIME_IMPORT_PRESETS),
        default=[],
        help=(
            "Named runtime import bundle to probe while SpiralTorch and "
            "Transformers are loaded. 'torch-transformers' probes both "
            "modules; 'hf-runtime' also probes tokenizers; 'hf-finetune' and "
            "'hf-peft' add common FT dependencies. May be repeated."
        ),
    )
    parser.add_argument(
        "--runtime-contract-preset",
        dest="runtime_contract_presets",
        action="append",
        choices=sorted(RUNTIME_IMPORT_PRESETS),
        default=[],
        help=(
            "Shortcut runtime contract preset: probes the preset modules, "
            "requires them to import in the same SpiralTorch/Transformers trace "
            "process, and requires runtime metadata to match when comparing "
            "against --compare-jsonl. May be repeated."
        ),
    )
    parser.add_argument(
        "--require-runtime-imports",
        action="store_true",
        help="Fail if any --runtime-import probe cannot be imported.",
    )
    parser.add_argument(
        "--require-runtime-import",
        dest="required_runtime_imports",
        action="append",
        default=[],
        help=(
            "Require a named Python module to import in the same trace process. "
            "The module is probed even if it was not listed with --runtime-import."
        ),
    )
    parser.add_argument(
        "--require-runtime-import-preset",
        dest="required_runtime_import_presets",
        action="append",
        choices=sorted(RUNTIME_IMPORT_PRESETS),
        default=[],
        help=(
            "Require a named runtime import preset to be satisfied in the same "
            "trace process. The preset modules are probed even if the preset "
            "was not listed with --runtime-import-preset. FT-oriented presets "
            "include 'hf-finetune' and 'hf-peft'."
        ),
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Load config/tokenizer metadata but skip model inference.",
    )
    parser.add_argument(
        "--capture-hidden-states",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ask the model to return hidden states when tracing prompts.",
    )
    parser.add_argument(
        "--require-hidden-states",
        action="store_true",
        help="Fail if traced outputs do not include hidden states.",
    )
    parser.add_argument(
        "--zspace-project",
        action="store_true",
        help="Project the selected trace vector through SpiralTorch Z-space.",
    )
    parser.add_argument(
        "--zspace-source",
        choices=["hidden", "top_logits"],
        default="hidden",
        help="Trace vector to project when --zspace-project is enabled.",
    )
    parser.add_argument(
        "--zspace-curvature",
        type=float,
        default=ZSPACE_CURVATURE,
        help="OpenTopos curvature for --zspace-project.",
    )
    parser.add_argument(
        "--zspace-frequency",
        type=float,
        default=ZSPACE_FREQUENCY,
        help="LanguageWaveEncoder frequency for --zspace-project.",
    )
    parser.add_argument(
        "--zspace-strength",
        type=float,
        default=ZSPACE_STRENGTH,
        help="ZSpaceProjector strength for --zspace-project.",
    )
    parser.add_argument(
        "--require-zspace-projection",
        action="store_true",
        help="Fail if --zspace-project cannot produce projection metrics.",
    )
    parser.add_argument(
        "--require-trace-match",
        action="store_true",
        help="Fail when prompt scope or gated trace metrics differ from --compare-jsonl.",
    )
    parser.add_argument(
        "--require-runtime-metadata-match",
        action="store_true",
        help=(
            "Fail when trace manifest runtime metadata differs from "
            "--compare-jsonl. This catches config/tokenizer/model swaps before "
            "prompt logits are interpreted."
        ),
    )
    parser.add_argument(
        "--require-top-token-match",
        action="store_true",
        help="Fail when a prompt's top next-token id differs from --compare-jsonl.",
    )
    parser.add_argument(
        "--max-top-logit-regression",
        type=float,
        default=None,
        help="Maximum allowed drop in top-1 logit relative to --compare-jsonl.",
    )
    parser.add_argument(
        "--max-top-probability-regression",
        type=float,
        default=None,
        help="Maximum allowed drop in top-1 probability relative to --compare-jsonl.",
    )
    parser.add_argument(
        "--max-logit-l2-change",
        type=float,
        default=None,
        help="Maximum allowed absolute change in prompt logit L2 relative to --compare-jsonl.",
    )
    parser.add_argument(
        "--max-hidden-state-l2-change",
        type=float,
        default=None,
        help=(
            "Maximum allowed absolute change in final hidden-state L2 relative "
            "to --compare-jsonl."
        ),
    )
    parser.add_argument(
        "--require-zspace-status",
        default=None,
        help="Fail when prompt trace zspace_projection_status does not match this value.",
    )
    args = parser.parse_args()
    apply_runtime_contract_presets(args)
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.metadata_only and args.require_hidden_states:
        parser.error("--require-hidden-states is incompatible with --metadata-only")
    if args.require_zspace_projection and not args.zspace_project:
        parser.error("--require-zspace-projection requires --zspace-project")
    if args.require_runtime_imports and not runtime_import_names_from_args(
        args,
        preset_modules=RUNTIME_IMPORT_PRESETS,
    ):
        parser.error(
            "--require-runtime-imports requires --runtime-import "
            "or --runtime-import-preset"
        )
    if (
        args.compare_output_jsonl is not None
        and args.compare_jsonl is None
        and args.require_zspace_status is None
    ):
        parser.error(
            "--compare-output-jsonl requires --compare-jsonl or "
            "--require-zspace-status"
        )
    compare_gates = [
        args.require_trace_match,
        args.require_runtime_metadata_match,
        args.require_top_token_match,
        args.max_top_logit_regression is not None,
        args.max_top_probability_regression is not None,
        args.max_logit_l2_change is not None,
        args.max_hidden_state_l2_change is not None,
    ]
    if any(compare_gates) and args.compare_jsonl is None:
        parser.error("trace comparison gates require --compare-jsonl")
    for name in [
        "max_top_logit_regression",
        "max_top_probability_regression",
        "max_logit_l2_change",
        "max_hidden_state_l2_change",
    ]:
        value = getattr(args, name)
        if value is not None and value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    return args


def append_unique(values, additions):
    return list(dict.fromkeys([*(values or []), *(additions or [])]))


def apply_runtime_contract_presets(args):
    presets = list(dict.fromkeys(getattr(args, "runtime_contract_presets", []) or []))
    if not presets:
        return args
    args.runtime_import_presets = append_unique(args.runtime_import_presets, presets)
    args.required_runtime_import_presets = append_unique(
        args.required_runtime_import_presets,
        presets,
    )
    args.require_runtime_imports = True
    if getattr(args, "compare_jsonl", None) is not None:
        args.require_runtime_metadata_match = True
    return args


def loader_kwargs(args):
    kwargs = {
        "local_files_only": not args.allow_remote,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.revision:
        kwargs["revision"] = args.revision
    return kwargs


def safe_int(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def first_attr(value, names):
    for name in names:
        if hasattr(value, name):
            candidate = getattr(value, name)
            if candidate is not None:
                return candidate
    return None


def string_list_label(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value) or None
    return str(value)


def tokenizer_len(tokenizer):
    try:
        return safe_int(len(tokenizer))
    except TypeError:
        return None


def model_parameter_count(model):
    if model is None:
        return None
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return None
    total = 0
    try:
        for parameter in parameters():
            numel = getattr(parameter, "numel", None)
            count = safe_int(numel() if callable(numel) else None)
            if count is None:
                return None
            total += count
    except TypeError:
        return None
    return total


def load_prompts(args):
    prompts = list(args.prompt)
    if args.prompt_file is not None:
        with open(args.prompt_file, "r", encoding="utf-8") as handle:
            prompts.extend(line.strip() for line in handle if line.strip())
    return prompts or [DEFAULT_PROMPT]


def output_value(outputs, name):
    if isinstance(outputs, Mapping):
        return outputs.get(name)
    return getattr(outputs, name, None)


def logits_vector(outputs):
    logits = output_value(outputs, "logits")
    if logits is None:
        raise RuntimeError("Transformers output did not include logits")
    return external_tensor_to_list(
        external_tensor_last_token(logits, name="logits"),
        name="logits",
    )


def hidden_vector(outputs):
    hidden_states = output_value(outputs, "hidden_states")
    if not hidden_states:
        return None, None
    last_hidden = hidden_states[-1]
    selected = external_tensor_last_token(last_hidden, name="hidden_states[-1]")
    return (
        external_tensor_to_list(selected, name="hidden_states[-1]"),
        external_tensor_shape(last_hidden, name="hidden_states[-1]"),
    )


def stable_softmax(values):
    if not values:
        return []
    peak = max(values)
    exp_values = [math.exp(value - peak) for value in values]
    total = sum(exp_values)
    if total == 0.0:
        return [0.0 for _ in exp_values]
    return [value / total for value in exp_values]


def top_k_logits(values, k):
    probabilities = stable_softmax(values)
    ranked = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    top_indices = ranked[:k]
    return {
        "top_token_ids": top_indices,
        "top_logits": [values[index] for index in top_indices],
        "top_probabilities": [probabilities[index] for index in top_indices],
        "top_probability_sum": sum(probabilities[index] for index in top_indices),
    }


def decode_token(tokenizer, token_id):
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        return None
    try:
        return decode([int(token_id)])
    except TypeError:
        return decode(int(token_id))


def encoded_input_ids(encoded):
    return encoded.get("input_ids") if isinstance(encoded, Mapping) else getattr(
        encoded,
        "input_ids",
        None,
    )


def encoded_token_count(encoded):
    input_ids = encoded_input_ids(encoded)
    if input_ids is None:
        return None
    try:
        shape = external_tensor_shape(input_ids, name="input_ids")
    except (TypeError, ValueError):
        return None
    if len(shape) >= 2:
        return shape[-1]
    return shape[0]


def tensor_metadata_fields(prefix, value):
    fields = {
        f"{prefix}_tensor_available": value is not None,
        f"{prefix}_tensor_backend": None,
        f"{prefix}_tensor_type": None,
        f"{prefix}_tensor_module": None,
        f"{prefix}_tensor_shape": None,
        f"{prefix}_tensor_shape_rank": None,
        f"{prefix}_tensor_shape_error": None,
        f"{prefix}_tensor_dtype": None,
        f"{prefix}_tensor_device": None,
        f"{prefix}_tensor_device_kind": None,
        f"{prefix}_tensor_requires_grad": None,
    }
    if value is None:
        return fields

    metadata = external_tensor_metadata(value, name=prefix)
    fields.update(
        {
            f"{prefix}_tensor_backend": metadata.get("backend"),
            f"{prefix}_tensor_type": metadata.get("type"),
            f"{prefix}_tensor_module": metadata.get("module"),
            f"{prefix}_tensor_shape": metadata.get("shape_label"),
            f"{prefix}_tensor_shape_rank": metadata.get("shape_rank"),
            f"{prefix}_tensor_shape_error": metadata.get("shape_error"),
            f"{prefix}_tensor_dtype": metadata.get("dtype"),
            f"{prefix}_tensor_device": metadata.get("device"),
            f"{prefix}_tensor_device_kind": metadata.get("device_kind"),
            f"{prefix}_tensor_requires_grad": metadata.get("requires_grad"),
        }
    )
    return fields


def call_tokenizer(tokenizer, prompt):
    encoded = tokenizer(prompt, return_tensors="pt")
    if isinstance(encoded, Mapping):
        return dict(encoded)
    return encoded


def call_model(model, encoded, *, capture_hidden_states):
    if isinstance(encoded, Mapping):
        kwargs = dict(encoded)
        if capture_hidden_states:
            kwargs["output_hidden_states"] = True
        try:
            return model(**kwargs)
        except TypeError:
            kwargs.pop("output_hidden_states", None)
            return model(**kwargs)
    if capture_hidden_states:
        try:
            return model(encoded, output_hidden_states=True)
        except TypeError:
            pass
    return model(encoded)


def call_model_no_grad(model, encoded, *, capture_hidden_states):
    try:
        torch = importlib.import_module("torch")
        no_grad = getattr(torch, "no_grad", None)
    except ImportError:
        no_grad = None
    if not callable(no_grad):
        return call_model(
            model,
            encoded,
            capture_hidden_states=capture_hidden_states,
        )
    with no_grad():
        return call_model(
            model,
            encoded,
            capture_hidden_states=capture_hidden_states,
        )


def vector_l2(values):
    return math.sqrt(sum(value * value for value in values))


def vector_mean_abs(values):
    if not values:
        return None
    return sum(abs(value) for value in values) / len(values)


def zspace_projection_metrics(values, args):
    row = {
        "zspace_projection_requested": bool(args.zspace_project),
        "zspace_projection_source": args.zspace_source if args.zspace_project else None,
        "zspace_projection_status": "not_requested",
        "zspace_projection_error": None,
        "zspace_projection_curvature": args.zspace_curvature if args.zspace_project else None,
        "zspace_projection_frequency": args.zspace_frequency if args.zspace_project else None,
        "zspace_projection_strength": args.zspace_strength if args.zspace_project else None,
        "zspace_projection_dims": len(values) if args.zspace_project and values else 0,
        "zspace_projection_input_l2": None,
        "zspace_projection_output_l2": None,
        "zspace_projection_delta_l2": None,
        "zspace_projection_delta_input_l2_ratio": None,
        "zspace_projection_output_input_l2_ratio": None,
    }
    if not args.zspace_project:
        return row
    if not values:
        row.update(
            {
                "zspace_projection_status": "missing_source_vector",
                "zspace_projection_error": "selected trace vector is missing",
            }
        )
        return row
    try:
        tensor = st.Tensor(1, len(values), values)
        topos = st.OpenTopos(args.zspace_curvature)
        encoder = st.LanguageWaveEncoder(topos.curvature(), args.zspace_frequency)
        projector = ZSpaceProjector(topos, encoder, strength=args.zspace_strength)
        projected = projector.forward(tensor)
        projected_values = list(projected.data())
    except Exception as exc:  # pragma: no cover - depends on real SpiralTorch runtime.
        row.update(
            {
                "zspace_projection_status": "error",
                "zspace_projection_error": f"{exc.__class__.__name__}: {exc}",
            }
        )
        return row

    input_l2 = vector_l2(values)
    output_l2 = vector_l2(projected_values)
    delta_l2 = vector_l2(
        [
            after - before
            for before, after in zip(values, projected_values, strict=False)
        ]
    )
    row.update(
        {
            "zspace_projection_status": "ok",
            "zspace_projection_input_l2": input_l2,
            "zspace_projection_output_l2": output_l2,
            "zspace_projection_delta_l2": delta_l2,
            "zspace_projection_delta_input_l2_ratio": (
                None if input_l2 == 0.0 else delta_l2 / input_l2
            ),
            "zspace_projection_output_input_l2_ratio": (
                None if input_l2 == 0.0 else output_l2 / input_l2
            ),
        }
    )
    return row


def config_fields(config):
    return {
        "transformers_config_class": config.__class__.__name__,
        "transformers_model_type": getattr(config, "model_type", None),
        "transformers_architectures": string_list_label(
            getattr(config, "architectures", None)
        )
        or None,
        "transformers_config_vocab_size": safe_int(getattr(config, "vocab_size", None)),
        "transformers_config_hidden_size": safe_int(
            first_attr(config, ["hidden_size", "n_embd", "d_model"])
        ),
        "transformers_config_num_hidden_layers": safe_int(
            first_attr(config, ["num_hidden_layers", "n_layer", "num_layers"])
        ),
        "transformers_config_num_attention_heads": safe_int(
            first_attr(config, ["num_attention_heads", "n_head"])
        ),
        "transformers_config_max_position_embeddings": safe_int(
            first_attr(config, ["max_position_embeddings", "n_positions", "seq_length"])
        ),
    }


def tokenizer_fields(tokenizer):
    length = tokenizer_len(tokenizer)
    vocab_size = safe_int(getattr(tokenizer, "vocab_size", None))
    return {
        "tokenizer_class": tokenizer.__class__.__name__,
        "transformers_tokenizer_class": tokenizer.__class__.__name__,
        "transformers_tokenizer_vocab_size": (
            length if length is not None else vocab_size
        ),
        "transformers_tokenizer_len": length,
    }


def model_fields(model):
    return {
        "transformers_model_class": None if model is None else model.__class__.__name__,
        "transformers_model_parameter_count": model_parameter_count(model),
    }


def runtime_import_fields(args):
    return runtime_import_probe_fields(
        args,
        preset_modules=RUNTIME_IMPORT_PRESETS,
    )


def import_context_fields(transformers):
    transformers_imported = transformers is not None
    return {
        "spiraltorch_imported": True,
        "spiraltorch_version": module_version(st),
        "spiraltorch_module_name": module_name(st),
        "spiraltorch_module_file": module_file(st),
        "transformers_imported": transformers_imported,
        "transformers_module_name": module_name(transformers),
        "transformers_module_file": module_file(transformers),
        "transformers_spiraltorch_coimport_status": (
            "ok" if transformers_imported else "transformers_missing"
        ),
    }


def manifest_row(
    args,
    prompts,
    transformers,
    config,
    tokenizer,
    model_loaded,
    model=None,
):
    row = {
        "row_type": "transformers_trace_manifest",
        "model_path": str(args.model_path),
        "prompt_count": len(prompts),
        "top_k": args.top_k,
        "local_files_only": not args.allow_remote,
        "trust_remote_code": args.trust_remote_code,
        "revision": args.revision,
        "metadata_only": args.metadata_only,
        "capture_hidden_states": args.capture_hidden_states,
        "zspace_project": args.zspace_project,
        "zspace_source": args.zspace_source if args.zspace_project else None,
        "transformers_version": getattr(transformers, "__version__", None),
        "model_loaded": model_loaded,
    }
    row.update(import_context_fields(transformers))
    row.update(runtime_import_fields(args))
    row.update(config_fields(config))
    row.update(tokenizer_fields(tokenizer))
    row.update(model_fields(model if model_loaded else None))
    if getattr(args, "require_runtime_imports", False) and not row[
        "runtime_imports_all_ok"
    ]:
        raise RuntimeError(
            "runtime import probe failed: " + row["runtime_imports_failed"]
        )
    requirement_failures = runtime_import_requirement_failures(row)
    if requirement_failures:
        raise RuntimeError(
            "runtime import requirement failed: " + ", ".join(requirement_failures)
        )
    return row


def trace_prompt(args, tokenizer, model, prompt, index):
    encoded = call_tokenizer(tokenizer, prompt)
    outputs = call_model_no_grad(
        model,
        encoded,
        capture_hidden_states=args.capture_hidden_states,
    )
    input_ids = encoded_input_ids(encoded)
    logits_tensor = output_value(outputs, "logits")
    hidden_states = output_value(outputs, "hidden_states")
    hidden_tensor = hidden_states[-1] if hidden_states else None
    logits = logits_vector(outputs)
    top = top_k_logits(logits, args.top_k)
    hidden, hidden_shape = hidden_vector(outputs)
    if args.require_hidden_states and hidden is None:
        raise RuntimeError("Transformers output did not include hidden states")

    projection_source = hidden if args.zspace_source == "hidden" else top["top_logits"]
    projection = zspace_projection_metrics(projection_source, args)
    if args.require_zspace_projection and projection["zspace_projection_status"] != "ok":
        raise RuntimeError(
            "Z-space projection failed: "
            f"{projection['zspace_projection_status']} {projection['zspace_projection_error']}"
        )

    row = {
        "row_type": "transformers_prompt_trace",
        "prompt_index": index,
        "prompt": prompt,
        "input_token_count": encoded_token_count(encoded),
        "logit_vocab_size": len(logits),
        "logit_l2": vector_l2(logits),
        "logit_mean_abs": vector_mean_abs(logits),
        "hidden_state_available": hidden is not None,
        "hidden_state_shape": (
            None if hidden_shape is None else "x".join(str(dim) for dim in hidden_shape)
        ),
        "hidden_state_dims": 0 if hidden is None else len(hidden),
        "hidden_state_l2": None if hidden is None else vector_l2(hidden),
        "hidden_state_mean_abs": None if hidden is None else vector_mean_abs(hidden),
        "top_token_ids": ",".join(str(token_id) for token_id in top["top_token_ids"]),
        "top_token_texts": json.dumps(
            [decode_token(tokenizer, token_id) for token_id in top["top_token_ids"]],
            ensure_ascii=False,
        ),
        "top_logits": ",".join(f"{value:.9g}" for value in top["top_logits"]),
        "top_probabilities": ",".join(
            f"{value:.9g}" for value in top["top_probabilities"]
        ),
        "top_probability_sum": top["top_probability_sum"],
    }
    row.update(tensor_metadata_fields("input_ids", input_ids))
    row.update(tensor_metadata_fields("logits", logits_tensor))
    row.update(tensor_metadata_fields("hidden_state", hidden_tensor))
    row.update(projection)
    return row


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
        raise ValueError(f"{path} did not contain any trace rows")
    return rows


def prompt_trace_rows(rows):
    return [row for row in rows if row.get("row_type") == "transformers_prompt_trace"]


TRACE_RUNTIME_METADATA_FIELDS = [
    "model_path",
    "top_k",
    "local_files_only",
    "trust_remote_code",
    "revision",
    "metadata_only",
    "capture_hidden_states",
    "zspace_project",
    "zspace_source",
    "spiraltorch_imported",
    "spiraltorch_version",
    "spiraltorch_module_name",
    "transformers_imported",
    "transformers_module_name",
    "transformers_spiraltorch_coimport_status",
    "runtime_import_presets",
    "runtime_import_preset_modules",
    "runtime_import_presets_satisfied",
    "runtime_import_presets_failed",
    "runtime_import_preset_missing_modules",
    "required_runtime_imports",
    "required_runtime_imports_imported",
    "required_runtime_imports_missing",
    "required_runtime_imports_passed",
    "required_runtime_import_presets",
    "required_runtime_import_presets_observed",
    "required_runtime_import_presets_satisfied",
    "required_runtime_import_presets_missing",
    "required_runtime_import_presets_unsatisfied",
    "required_runtime_import_presets_passed",
    "runtime_imports_requested",
    "runtime_import_probe_count",
    "runtime_imports_imported",
    "runtime_imports_failed",
    "runtime_imports_all_ok",
    "runtime_import_coimport_status",
    "runtime_imports_coimported",
    "runtime_import_coimport_modules",
    "runtime_import_coimport_missing_modules",
    "runtime_import_versions",
    "runtime_import_install_hints",
    "runtime_import_failed_install_hints",
    "runtime_import_module_names",
    "transformers_version",
    "model_loaded",
    "transformers_config_class",
    "transformers_model_type",
    "transformers_architectures",
    "transformers_config_vocab_size",
    "transformers_config_hidden_size",
    "transformers_config_num_hidden_layers",
    "transformers_config_num_attention_heads",
    "transformers_config_max_position_embeddings",
    "transformers_tokenizer_class",
    "transformers_tokenizer_vocab_size",
    "transformers_tokenizer_len",
    "transformers_model_class",
    "transformers_model_parameter_count",
]


def trace_manifest_row(rows):
    manifests = [
        row for row in rows if row.get("row_type") == "transformers_trace_manifest"
    ]
    if len(manifests) > 1:
        raise ValueError("trace JSONL has multiple transformers_trace_manifest rows")
    return manifests[0] if manifests else None


def changed_runtime_metadata_fields(current, baseline):
    if current is None or baseline is None:
        return []
    return [
        field
        for field in TRACE_RUNTIME_METADATA_FIELDS
        if current.get(field) != baseline.get(field)
    ]


def runtime_metadata_failures(current, baseline, changed_fields, args):
    if not getattr(args, "require_runtime_metadata_match", False):
        return []
    if current is None or baseline is None:
        return ["runtime_metadata_missing"]
    if changed_fields:
        return ["runtime_metadata_changed"]
    return []


def prompt_rows_by_index(rows, label):
    by_index = {}
    for row in prompt_trace_rows(rows):
        index = row.get("prompt_index")
        if index in by_index:
            raise ValueError(f"{label} has duplicate prompt_index={index!r}")
        by_index[index] = row
    return by_index


def csv_values(value, cast):
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [cast(item) for item in value]
    return [cast(part) for part in str(value).split(",") if part != ""]


def csv_first(value, cast):
    values = csv_values(value, cast)
    return values[0] if values else None


def numeric(row, key):
    value = row.get(key)
    if value is None:
        return None
    return float(value)


def positive_drop(before, after):
    if before is None or after is None:
        return None
    return max(0.0, before - after)


def absolute_change(before, after):
    if before is None or after is None:
        return None
    return abs(after - before)


def exceeds(value, limit):
    return value is not None and limit is not None and value > limit


def failure_label(failures):
    return ",".join(failures) if failures else "none"


TRACE_TENSOR_RUNTIME_FIELDS = [
    "input_ids_tensor_backend",
    "input_ids_tensor_device",
    "input_ids_tensor_device_kind",
    "input_ids_tensor_dtype",
    "logits_tensor_backend",
    "logits_tensor_device",
    "logits_tensor_device_kind",
    "logits_tensor_dtype",
    "hidden_state_tensor_backend",
    "hidden_state_tensor_device",
    "hidden_state_tensor_device_kind",
    "hidden_state_tensor_dtype",
]


def changed_tensor_runtime_fields(current, baseline):
    changed = []
    for field in TRACE_TENSOR_RUNTIME_FIELDS:
        before = baseline.get(field)
        after = current.get(field)
        if before != after and (before is not None or after is not None):
            changed.append(field)
    return changed


def numeric_detail_values(rows, key):
    values = []
    for row in rows:
        value = row.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(number):
            values.append(number)
    return values


def max_detail_value(rows, key):
    values = numeric_detail_values(rows, key)
    return max(values) if values else None


def count_detail_flag(rows, key):
    return sum(1 for row in rows if row.get(key) is True)


def current_trace_gate_rows(rows, args):
    if args.require_zspace_status is None:
        return []
    prompt_rows = prompt_trace_rows(rows)
    details = []
    failures = 0
    for row in prompt_rows:
        status = row.get("zspace_projection_status")
        passed = status == args.require_zspace_status
        if not passed:
            failures += 1
        details.append(
            {
                "row_type": "transformers_trace_gate",
                "prompt_index": row.get("prompt_index"),
                "prompt": row.get("prompt"),
                "gate": "zspace_status",
                "expected": args.require_zspace_status,
                "actual": status,
                "passed": passed,
            }
        )
    return [
        {
            "row_type": "transformers_trace_gate_summary",
            "gate": "zspace_status",
            "prompt_rows": len(prompt_rows),
            "failures": failures,
            "passed": failures == 0,
        },
        *details,
    ]


def compare_prompt_rows(current, baseline, args):
    before_top_token = csv_first(baseline.get("top_token_ids"), int)
    after_top_token = csv_first(current.get("top_token_ids"), int)
    before_top_logit = csv_first(baseline.get("top_logits"), float)
    after_top_logit = csv_first(current.get("top_logits"), float)
    before_top_probability = csv_first(baseline.get("top_probabilities"), float)
    after_top_probability = csv_first(current.get("top_probabilities"), float)
    top_logit_regression = positive_drop(before_top_logit, after_top_logit)
    top_probability_regression = positive_drop(
        before_top_probability,
        after_top_probability,
    )
    logit_l2_change = absolute_change(
        numeric(baseline, "logit_l2"),
        numeric(current, "logit_l2"),
    )
    hidden_state_l2_change = absolute_change(
        numeric(baseline, "hidden_state_l2"),
        numeric(current, "hidden_state_l2"),
    )
    prompt_changed = current.get("prompt") != baseline.get("prompt")
    top_token_changed = before_top_token != after_top_token
    zspace_status_before = baseline.get("zspace_projection_status")
    zspace_status_after = current.get("zspace_projection_status")
    zspace_status_changed = (
        zspace_status_before != zspace_status_after
        and (zspace_status_before is not None or zspace_status_after is not None)
    )
    tensor_runtime_changes = changed_tensor_runtime_fields(current, baseline)
    failures = []
    if args.require_trace_match and prompt_changed:
        failures.append("prompt_changed")
    if args.require_top_token_match and top_token_changed:
        failures.append("top_token_changed")
    if exceeds(top_logit_regression, args.max_top_logit_regression):
        failures.append("top_logit_regression")
    if exceeds(top_probability_regression, args.max_top_probability_regression):
        failures.append("top_probability_regression")
    if exceeds(logit_l2_change, args.max_logit_l2_change):
        failures.append("logit_l2_change")
    if exceeds(hidden_state_l2_change, args.max_hidden_state_l2_change):
        failures.append("hidden_state_l2_change")
    return {
        "row_type": "transformers_trace_compare_prompt",
        "prompt_index": current.get("prompt_index"),
        "prompt_before": baseline.get("prompt"),
        "prompt_after": current.get("prompt"),
        "prompt_changed": prompt_changed,
        "top_token_before": before_top_token,
        "top_token_after": after_top_token,
        "top_token_changed": top_token_changed,
        "top_logit_before": before_top_logit,
        "top_logit_after": after_top_logit,
        "top_logit_regression": top_logit_regression,
        "top_probability_before": before_top_probability,
        "top_probability_after": after_top_probability,
        "top_probability_regression": top_probability_regression,
        "logit_l2_before": numeric(baseline, "logit_l2"),
        "logit_l2_after": numeric(current, "logit_l2"),
        "logit_l2_change": logit_l2_change,
        "hidden_state_l2_before": numeric(baseline, "hidden_state_l2"),
        "hidden_state_l2_after": numeric(current, "hidden_state_l2"),
        "hidden_state_l2_change": hidden_state_l2_change,
        "zspace_status_before": zspace_status_before,
        "zspace_status_after": zspace_status_after,
        "zspace_status_changed": zspace_status_changed,
        "tensor_runtime_changed": bool(tensor_runtime_changes),
        "tensor_runtime_changed_fields": failure_label(tensor_runtime_changes),
        "failures": failure_label(failures),
        "passed": not failures,
    }


def compare_trace_rows(current_rows, baseline_rows, args):
    current_manifest = trace_manifest_row(current_rows)
    baseline_manifest = trace_manifest_row(baseline_rows)
    changed_runtime_fields = changed_runtime_metadata_fields(
        current_manifest,
        baseline_manifest,
    )
    runtime_failures = runtime_metadata_failures(
        current_manifest,
        baseline_manifest,
        changed_runtime_fields,
        args,
    )
    current = prompt_rows_by_index(current_rows, "current")
    baseline = prompt_rows_by_index(baseline_rows, "baseline")
    details = []
    failures = len(runtime_failures)
    missing = sorted(set(baseline) - set(current))
    extra = sorted(set(current) - set(baseline))
    if args.require_trace_match:
        failures += len(missing) + len(extra)
    for index in missing:
        details.append(
            {
                "row_type": "transformers_trace_compare_prompt",
                "prompt_index": index,
                "status": "missing",
                "failures": "missing_prompt" if args.require_trace_match else "none",
                "passed": not args.require_trace_match,
            }
        )
    for index in extra:
        details.append(
            {
                "row_type": "transformers_trace_compare_prompt",
                "prompt_index": index,
                "status": "extra",
                "failures": "extra_prompt" if args.require_trace_match else "none",
                "passed": not args.require_trace_match,
            }
        )
    for index in sorted(set(current) & set(baseline)):
        detail = compare_prompt_rows(current[index], baseline[index], args)
        if not detail["passed"]:
            failures += 1
        details.append(detail)
    compared_details = [
        row
        for row in details
        if row.get("status") not in {"missing", "extra"}
    ]
    summary = {
        "row_type": "transformers_trace_compare_summary",
        "baseline_prompt_rows": len(baseline),
        "current_prompt_rows": len(current),
        "missing_prompt_rows": len(missing),
        "extra_prompt_rows": len(extra),
        "compared_prompt_rows": len(set(current) & set(baseline)),
        "require_trace_match": args.require_trace_match,
        "require_runtime_metadata_match": getattr(
            args,
            "require_runtime_metadata_match",
            False,
        ),
        "runtime_metadata_available": (
            current_manifest is not None and baseline_manifest is not None
        ),
        "runtime_metadata_changed_count": len(changed_runtime_fields),
        "runtime_metadata_changed_fields": failure_label(changed_runtime_fields),
        "runtime_metadata_failures": failure_label(runtime_failures),
        "require_top_token_match": args.require_top_token_match,
        "max_top_logit_regression": args.max_top_logit_regression,
        "max_top_probability_regression": args.max_top_probability_regression,
        "max_logit_l2_change": args.max_logit_l2_change,
        "max_hidden_state_l2_change": args.max_hidden_state_l2_change,
        "prompt_changed_rows": count_detail_flag(compared_details, "prompt_changed"),
        "top_token_changed_rows": count_detail_flag(
            compared_details,
            "top_token_changed",
        ),
        "zspace_status_changed_rows": count_detail_flag(
            compared_details,
            "zspace_status_changed",
        ),
        "tensor_runtime_changed_rows": count_detail_flag(
            compared_details,
            "tensor_runtime_changed",
        ),
        "observed_max_top_logit_regression": max_detail_value(
            compared_details,
            "top_logit_regression",
        ),
        "observed_max_top_probability_regression": max_detail_value(
            compared_details,
            "top_probability_regression",
        ),
        "observed_max_logit_l2_change": max_detail_value(
            compared_details,
            "logit_l2_change",
        ),
        "observed_max_hidden_state_l2_change": max_detail_value(
            compared_details,
            "hidden_state_l2_change",
        ),
        "failures": failures,
        "passed": failures == 0,
    }
    return [summary, *details]


def print_compare_summary(row):
    print(
        "transformers_trace_compare "
        f"baseline_prompt_rows={row['baseline_prompt_rows']} "
        f"current_prompt_rows={row['current_prompt_rows']} "
        f"missing_prompt_rows={row['missing_prompt_rows']} "
        f"extra_prompt_rows={row['extra_prompt_rows']} "
        f"runtime_metadata_changed_count={row['runtime_metadata_changed_count']} "
        f"runtime_metadata_failures={row['runtime_metadata_failures']} "
        f"prompt_changed_rows={row['prompt_changed_rows']} "
        f"top_token_changed_rows={row['top_token_changed_rows']} "
        f"zspace_status_changed_rows={row['zspace_status_changed_rows']} "
        f"tensor_runtime_changed_rows={row['tensor_runtime_changed_rows']} "
        "observed_max_top_logit_regression="
        f"{row['observed_max_top_logit_regression']} "
        "observed_max_top_probability_regression="
        f"{row['observed_max_top_probability_regression']} "
        f"failures={row['failures']} "
        f"passed={row['passed']}"
    )


def print_gate_summary(row):
    print(
        "transformers_trace_gate "
        f"gate={row['gate']} "
        f"prompt_rows={row['prompt_rows']} "
        f"failures={row['failures']} "
        f"passed={row['passed']}"
    )


def comparison_failed(rows):
    return any(row.get("passed") is False for row in rows)


def print_trace(row):
    print(
        "transformers_prompt_trace "
        f"prompt_index={row['prompt_index']} "
        f"tokens={row['input_token_count']} "
        f"vocab={row['logit_vocab_size']} "
        f"top_token_ids={row['top_token_ids']} "
        f"top_probability_sum={row['top_probability_sum']:.9g} "
        f"hidden_state_available={row['hidden_state_available']} "
        f"zspace_projection_status={row['zspace_projection_status']}"
    )


def main():
    args = parse_args()
    prompts = load_prompts(args)
    transformers = importlib.import_module("transformers")
    kwargs = loader_kwargs(args)
    config = transformers.AutoConfig.from_pretrained(str(args.model_path), **kwargs)
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(args.model_path), **kwargs)

    rows = []
    model = None
    if not args.metadata_only:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            str(args.model_path),
            config=config,
            **kwargs,
        )
        eval_fn = getattr(model, "eval", None)
        if callable(eval_fn):
            eval_fn()

    rows.append(
        manifest_row(
            args,
            prompts,
            transformers,
            config,
            tokenizer,
            model_loaded=model is not None,
            model=model,
        )
    )
    if model is not None:
        for index, prompt in enumerate(prompts):
            row = trace_prompt(args, tokenizer, model, prompt, index)
            rows.append(row)
            print_trace(row)

    if args.jsonl is not None:
        write_jsonl(args.jsonl, rows)
        print(f"transformers_trace_jsonl={args.jsonl} rows={len(rows)}")
    else:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False, sort_keys=True))

    comparison_rows = current_trace_gate_rows(rows, args)
    if args.compare_jsonl is not None:
        comparison_rows.extend(
            compare_trace_rows(rows, load_jsonl(args.compare_jsonl), args)
        )
    for row in comparison_rows:
        if row.get("row_type") == "transformers_trace_compare_summary":
            print_compare_summary(row)
        elif row.get("row_type") == "transformers_trace_gate_summary":
            print_gate_summary(row)
    if args.compare_output_jsonl is not None:
        write_jsonl(args.compare_output_jsonl, comparison_rows)
        print(
            f"transformers_trace_compare_jsonl={args.compare_output_jsonl} "
            f"rows={len(comparison_rows)}"
        )
    if comparison_failed(comparison_rows):
        raise RuntimeError("Transformers trace comparison gate failed")


if __name__ == "__main__":
    main()
