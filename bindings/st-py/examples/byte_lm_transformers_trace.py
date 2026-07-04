import argparse
import importlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

import spiraltorch as st
from spiraltorch.nn import ZSpaceProjector


DEFAULT_PROMPT = "SpiralTorch routes meaning through Z-space."
ZSPACE_CURVATURE = -0.04
ZSPACE_FREQUENCY = 0.65
ZSPACE_STRENGTH = 1.0


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
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.metadata_only and args.require_hidden_states:
        parser.error("--require-hidden-states is incompatible with --metadata-only")
    if args.require_zspace_projection and not args.zspace_project:
        parser.error("--require-zspace-projection requires --zspace-project")
    return args


def loader_kwargs(args):
    kwargs = {
        "local_files_only": not args.allow_remote,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.revision:
        kwargs["revision"] = args.revision
    return kwargs


def load_prompts(args):
    prompts = list(args.prompt)
    if args.prompt_file is not None:
        with open(args.prompt_file, "r", encoding="utf-8") as handle:
            prompts.extend(line.strip() for line in handle if line.strip())
    return prompts or [DEFAULT_PROMPT]


def tensor_shape(value):
    shape = getattr(value, "shape", None)
    if shape is None:
        size = getattr(value, "size", None)
        shape = size() if callable(size) else None
    if shape is not None:
        return tuple(int(dim) for dim in shape)
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            if value[0] and isinstance(value[0][0], (list, tuple)):
                return (len(value), len(value[0]), len(value[0][0]))
            return (len(value), len(value[0]))
        return (len(value),)
    return None


def materialize_tensor(value):
    tensor = value
    for method_name in ["detach", "cpu", "float"]:
        method = getattr(tensor, method_name, None)
        if callable(method):
            tensor = method()
    return tensor


def flatten_nested(value):
    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            flattened.extend(flatten_nested(item))
        return flattened
    return [value]


def tensor_to_list(value):
    materialized = materialize_tensor(value)
    tolist = getattr(materialized, "tolist", None)
    if callable(tolist):
        materialized = tolist()
    if isinstance(materialized, (list, tuple)):
        return [float(item) for item in flatten_nested(materialized)]
    return [float(materialized)]


def index_last_token(value):
    shape = tensor_shape(value)
    if shape is None:
        return value
    if len(shape) == 3:
        try:
            return value[0, shape[1] - 1, :]
        except (TypeError, IndexError):
            return value[0][shape[1] - 1]
    if len(shape) == 2:
        try:
            return value[shape[0] - 1, :]
        except (TypeError, IndexError):
            return value[shape[0] - 1]
    return value


def output_value(outputs, name):
    if isinstance(outputs, Mapping):
        return outputs.get(name)
    return getattr(outputs, name, None)


def logits_vector(outputs):
    logits = output_value(outputs, "logits")
    if logits is None:
        raise RuntimeError("Transformers output did not include logits")
    return tensor_to_list(index_last_token(logits))


def hidden_vector(outputs):
    hidden_states = output_value(outputs, "hidden_states")
    if not hidden_states:
        return None, None
    last_hidden = hidden_states[-1]
    selected = index_last_token(last_hidden)
    return tensor_to_list(selected), tensor_shape(last_hidden)


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


def encoded_token_count(encoded):
    input_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else getattr(
        encoded,
        "input_ids",
        None,
    )
    shape = tensor_shape(input_ids)
    if shape is None:
        return None
    if len(shape) >= 2:
        return shape[-1]
    return shape[0]


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
        "transformers_architectures": ",".join(
            str(item) for item in (getattr(config, "architectures", None) or [])
        )
        or None,
        "transformers_config_vocab_size": getattr(config, "vocab_size", None),
        "transformers_config_hidden_size": getattr(
            config,
            "hidden_size",
            getattr(config, "n_embd", None),
        ),
    }


def manifest_row(args, prompts, transformers, config, tokenizer, model_loaded):
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
        "tokenizer_class": tokenizer.__class__.__name__,
        "model_loaded": model_loaded,
    }
    row.update(config_fields(config))
    return row


def trace_prompt(args, tokenizer, model, prompt, index):
    encoded = call_tokenizer(tokenizer, prompt)
    outputs = call_model_no_grad(
        model,
        encoded,
        capture_hidden_states=args.capture_hidden_states,
    )
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
    row.update(projection)
    return row


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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


if __name__ == "__main__":
    main()
