#!/usr/bin/env python3
"""Sweep SpiralTorch Z-Space generation controls on a Hugging Face CausalLM."""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import math
import os
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from spiraltorch.hf_ft import (
    hf_finetune_generation_report,
    hf_finetune_model_profile_lines,
    resolve_hf_finetune_model_profile,
)
from spiraltorch.hf_generation import build_zspace_repression_logits_processor


DEFAULT_MODEL = "gpt2"
HF_OFFLINE_ENV_VARS = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
)


def _csv_items(value: str) -> list[str]:
    values = [item.strip() for item in str(value).split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("value list must not be empty")
    return values


def _float_values(value: str) -> list[float]:
    values = [float(item) for item in _csv_items(value)]
    if any(not math.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError("all values must be finite")
    return values


def _positive_float_values(value: str) -> list[float]:
    values = _float_values(value)
    if any(item <= 0.0 for item in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def _non_negative_float_values(value: str) -> list[float]:
    values = _float_values(value)
    if any(item < 0.0 for item in values):
        raise argparse.ArgumentTypeError("all values must be non-negative")
    return values


def _negative_float_values(value: str) -> list[float]:
    values = _float_values(value)
    if any(item >= 0.0 for item in values):
        raise argparse.ArgumentTypeError("all values must be negative")
    return values


def _positive_int_values(value: str) -> list[int]:
    values = [int(item) for item in _csv_items(value)]
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all values must be positive integers")
    return values


def _non_negative_int_values(value: str) -> list[int]:
    values = [int(item) for item in _csv_items(value)]
    if any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("all values must be non-negative integers")
    return values


def _unit_interval_float_values(value: str) -> list[float]:
    values = _float_values(value)
    if any(item < 0.0 or item > 1.0 for item in values):
        raise argparse.ArgumentTypeError("all values must be in [0.0, 1.0]")
    return values


def _optional_float_values(value: str) -> list[float | None]:
    result: list[float | None] = []
    for item in _csv_items(value):
        if item.lower() in {"none", "null", "off"}:
            result.append(None)
            continue
        parsed = float(item)
        if not math.isfinite(parsed):
            raise argparse.ArgumentTypeError("all numeric values must be finite")
        result.append(parsed)
    return result


def _label_number(value: float | int | None) -> str:
    if value is None:
        return "none"
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _argv_has_option(raw_argv: Sequence[str], *names: str) -> bool:
    prefixes = tuple(f"{name}=" for name in names)
    return any(arg in names or arg.startswith(prefixes) for arg in raw_argv)


def _mapping_or_empty(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _profile_float(value: object) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("profile generation value must be finite")
    return parsed


def _profile_int(value: object) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError("profile generation integer value must be non-negative")
    return parsed


def _apply_model_profile_defaults(
    args: argparse.Namespace,
    raw_argv: Sequence[str],
) -> None:
    args._hf_finetune_model_profile = None
    args._hf_finetune_model_profile_lines = []
    if args.model_configs is None and args.model_profile is None:
        return
    profile = resolve_hf_finetune_model_profile(
        args.model_configs,
        profile=args.model_profile,
    )
    args._hf_finetune_model_profile = profile
    args._hf_finetune_model_profile_lines = hf_finetune_model_profile_lines(profile)
    generation = _mapping_or_empty(profile.get("generation"))

    def set_if_missing(attr: str, value: object, *flags: str) -> None:
        if value is None or _argv_has_option(raw_argv, *flags):
            return
        setattr(args, attr, value)

    def set_scalar_if_missing(
        key: str,
        attr: str,
        *flags: str,
        caster=None,
    ) -> None:
        if key not in generation or _argv_has_option(raw_argv, *flags):
            return
        value = generation.get(key)
        if value is None:
            return
        setattr(args, attr, value if caster is None else caster(value))

    def set_grid_if_missing(
        key: str,
        attr: str,
        *flags: str,
        caster=None,
        allow_none: bool = False,
    ) -> None:
        if key not in generation or _argv_has_option(raw_argv, *flags):
            return
        value = generation.get(key)
        if value is None:
            if allow_none:
                setattr(args, attr, [None])
            return
        setattr(args, attr, [value if caster is None else caster(value)])

    set_if_missing("model_name", str(profile.get("model_name")), "--model-name")
    set_if_missing(
        "tokenizer_name",
        str(profile.get("tokenizer_name")),
        "--tokenizer-name",
    )
    set_scalar_if_missing(
        "max_new_tokens",
        "max_new_tokens",
        "--max-new-tokens",
        caster=_profile_int,
    )
    if "do_sample" in generation and not _argv_has_option(raw_argv, "--do-sample"):
        args.do_sample = bool(generation.get("do_sample"))
    set_scalar_if_missing(
        "temperature",
        "sample_temperature",
        "--sample-temperature",
        caster=_profile_float,
    )
    set_scalar_if_missing(
        "top_k",
        "sample_top_k",
        "--sample-top-k",
        caster=_profile_int,
    )
    set_grid_if_missing(
        "zspace_top_k",
        "zspace_top_k_values",
        "--zspace-top-k-values",
        caster=_profile_int,
    )
    set_grid_if_missing(
        "zspace_curvature",
        "zspace_curvature_values",
        "--zspace-curvature-values",
        caster=_profile_float,
    )
    set_grid_if_missing(
        "zspace_temperature",
        "zspace_temperature_values",
        "--zspace-temperature-values",
        caster=_profile_float,
    )
    set_grid_if_missing(
        "zspace_entropy_target",
        "zspace_entropy_target_values",
        "--zspace-entropy-target-values",
        caster=_profile_float,
        allow_none=True,
    )
    set_grid_if_missing(
        "zspace_entropy_gain",
        "zspace_entropy_gain_values",
        "--zspace-entropy-gain-values",
        caster=_profile_float,
    )
    set_scalar_if_missing(
        "zspace_entropy_tolerance",
        "zspace_entropy_tolerance",
        "--zspace-entropy-tolerance",
        caster=_profile_float,
    )
    set_scalar_if_missing(
        "zspace_min_temperature",
        "zspace_min_temperature",
        "--zspace-min-temperature",
        caster=_profile_float,
    )
    set_scalar_if_missing(
        "zspace_max_temperature",
        "zspace_max_temperature",
        "--zspace-max-temperature",
        caster=_profile_float,
    )
    set_grid_if_missing(
        "repression_window",
        "repression_window_values",
        "--repression-window-values",
        caster=_profile_int,
    )
    set_grid_if_missing(
        "repression_strength",
        "repression_strength_values",
        "--repression-strength-values",
        caster=_profile_float,
    )
    set_grid_if_missing(
        "last_token_repression",
        "last_token_repression_values",
        "--last-token-repression-values",
        caster=_profile_float,
    )
    set_grid_if_missing(
        "ngram_size",
        "ngram_size_values",
        "--ngram-size-values",
        caster=_profile_int,
    )
    set_grid_if_missing(
        "ngram_window",
        "ngram_window_values",
        "--ngram-window-values",
        caster=_profile_int,
    )
    set_grid_if_missing(
        "ngram_repression_strength",
        "ngram_repression_strength_values",
        "--ngram-repression-strength-values",
        caster=_profile_float,
    )
    set_grid_if_missing(
        "ngram_decay",
        "ngram_decay_values",
        "--ngram-decay-values",
        caster=_profile_float,
    )
    if "zspace_keep_non_top_k" in generation and not _argv_has_option(
        raw_argv,
        "--keep-non-top-k",
    ):
        args.keep_non_top_k = bool(generation.get("zspace_keep_non_top_k"))
    if "zspace_no_native" in generation and not _argv_has_option(
        raw_argv,
        "--zspace-no-native",
    ):
        args.zspace_no_native = bool(generation.get("zspace_no_native"))
    set_scalar_if_missing(
        "zspace_report_limit",
        "report_limit",
        "--report-limit",
        caster=_profile_int,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-configs", type=Path, default=None)
    parser.add_argument("--model-profile", default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--tokenizer-name",
        default=None,
        help=(
            "Optional tokenizer id/path. Defaults to --model-name; useful when "
            "--model-name is a fine-tuned checkpoint that does not carry "
            "tokenizer files."
        ),
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/hf-gpt2-zspace-generation-control-sweep.json"))
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--sample-top-k", type=int, default=0)
    parser.add_argument("--zspace-top-k-values", type=_positive_int_values, default=[64])
    parser.add_argument("--zspace-curvature-values", type=_negative_float_values, default=[-0.04])
    parser.add_argument("--zspace-temperature-values", type=_positive_float_values, default=[1.0])
    parser.add_argument("--zspace-entropy-target-values", type=_optional_float_values, default=[None, 3.0])
    parser.add_argument("--zspace-entropy-gain-values", type=_non_negative_float_values, default=[0.5])
    parser.add_argument("--zspace-entropy-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--zspace-min-temperature", type=float, default=0.7)
    parser.add_argument("--zspace-max-temperature", type=float, default=2.4)
    parser.add_argument("--repression-window-values", type=_positive_int_values, default=[16])
    parser.add_argument("--repression-strength-values", type=_non_negative_float_values, default=[0.0, 1.0])
    parser.add_argument("--last-token-repression-values", type=_non_negative_float_values, default=[0.0, 1.0])
    parser.add_argument("--ngram-size-values", type=_non_negative_int_values, default=[0])
    parser.add_argument("--ngram-window-values", type=_non_negative_int_values, default=[0])
    parser.add_argument("--ngram-repression-strength-values", type=_non_negative_float_values, default=[0.0])
    parser.add_argument("--ngram-decay-values", type=_unit_interval_float_values, default=[1.0])
    parser.add_argument("--keep-non-top-k", action="store_true")
    parser.add_argument("--zspace-no-native", action="store_true")
    parser.add_argument("--report-limit", type=int, default=64)
    args = parser.parse_args(argv)
    _apply_model_profile_defaults(args, raw_argv)
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.sample_temperature <= 0.0 or not math.isfinite(args.sample_temperature):
        parser.error("--sample-temperature must be finite and positive")
    if args.sample_top_k < 0:
        parser.error("--sample-top-k must be non-negative")
    if args.zspace_entropy_tolerance < 0.0 or not math.isfinite(
        args.zspace_entropy_tolerance
    ):
        parser.error("--zspace-entropy-tolerance must be finite and non-negative")
    if args.zspace_min_temperature <= 0.0 or not math.isfinite(
        args.zspace_min_temperature
    ):
        parser.error("--zspace-min-temperature must be finite and positive")
    if args.zspace_max_temperature <= 0.0 or not math.isfinite(
        args.zspace_max_temperature
    ):
        parser.error("--zspace-max-temperature must be finite and positive")
    if args.zspace_min_temperature > args.zspace_max_temperature:
        parser.error("--zspace-min-temperature must be <= --zspace-max-temperature")
    if args.report_limit < 0:
        parser.error("--report-limit must be non-negative")
    return args


@contextlib.contextmanager
def _hf_remote_access(allow_remote: bool):
    previous = {name: os.environ.get(name) for name in HF_OFFLINE_ENV_VARS}
    if allow_remote:
        for name in HF_OFFLINE_ENV_VARS:
            os.environ.pop(name, None)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _loader_kwargs(args: argparse.Namespace) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True
    return kwargs


def build_control_runs(args: argparse.Namespace) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    if not args.no_baseline:
        runs.append(
            {
                "name": "baseline-greedy" if not args.do_sample else "baseline-sample",
                "kind": "baseline",
                "config": {},
            }
        )
    grid = product(
        args.zspace_top_k_values,
        args.zspace_curvature_values,
        args.zspace_temperature_values,
        args.zspace_entropy_target_values,
        args.zspace_entropy_gain_values,
        args.repression_window_values,
        args.repression_strength_values,
        args.last_token_repression_values,
        args.ngram_size_values,
        args.ngram_window_values,
        args.ngram_repression_strength_values,
        args.ngram_decay_values,
    )
    for (
        top_k,
        curvature,
        temperature,
        entropy_target,
        entropy_gain,
        repression_window,
        repression_strength,
        last_token_repression,
        ngram_size,
        ngram_window,
        ngram_repression_strength,
        ngram_decay,
    ) in grid:
        name = (
            f"zt{_label_number(entropy_target)}"
            f"-rs{_label_number(repression_strength)}"
            f"-lr{_label_number(last_token_repression)}"
            f"-ng{ngram_size}"
            f"-nw{ngram_window}"
            f"-nr{_label_number(ngram_repression_strength)}"
            f"-k{top_k}"
        )
        runs.append(
            {
                "name": name,
                "kind": "zspace_repression_softmax",
                "config": {
                    "top_k": int(top_k),
                    "curvature": float(curvature),
                    "temperature": float(temperature),
                    "entropy_target": entropy_target,
                    "entropy_tolerance": float(args.zspace_entropy_tolerance),
                    "entropy_gain": float(entropy_gain),
                    "min_temperature": float(args.zspace_min_temperature),
                    "max_temperature": float(args.zspace_max_temperature),
                    "repression_window": int(repression_window),
                    "repression_strength": float(repression_strength),
                    "last_token_repression": float(last_token_repression),
                    "ngram_size": int(ngram_size),
                    "ngram_window": int(ngram_window),
                    "ngram_repression_strength": float(ngram_repression_strength),
                    "ngram_decay": float(ngram_decay),
                    "mask_non_top_k": not bool(args.keep_non_top_k),
                    "use_native_zspace": not bool(args.zspace_no_native),
                },
            }
        )
    return runs


def _model_device(model: Any) -> Any | None:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return None


def _move_to_device(value: Any, device: Any | None) -> Any:
    if device is None:
        return value
    mover = getattr(value, "to", None)
    if callable(mover):
        try:
            return mover(device)
        except (TypeError, RuntimeError, ValueError):
            return value
    if isinstance(value, Mapping):
        return {
            key: _move_to_device(item, device)
            for key, item in value.items()
        }
    return value


def _first_sequence(value: Any) -> Any:
    try:
        return value[0]
    except (TypeError, KeyError, IndexError):
        return value


def _last_dim(value: Any) -> int | None:
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[-1])
    try:
        return len(value)
    except TypeError:
        return None


def text_repetition_report(text: object, *, ngram_size: int = 3) -> dict[str, object]:
    words = str(text or "").split()
    if ngram_size <= 0:
        raise ValueError("ngram_size must be positive")
    ngrams = [
        tuple(words[index : index + ngram_size])
        for index in range(0, max(0, len(words) - ngram_size + 1))
    ]
    counts = Counter(ngrams)
    repeated_ngram_total = sum(count - 1 for count in counts.values() if count > 1)
    max_ngram_repetition = max(counts.values(), default=0)
    consecutive_repeated_tokens = sum(
        1 for before, after in zip(words, words[1:]) if before == after
    )
    unique_word_ratio = None if not words else len(set(words)) / len(words)
    loop_score = (
        float(repeated_ngram_total)
        + float(max(0, max_ngram_repetition - 1))
        + float(consecutive_repeated_tokens)
    )
    return {
        "word_count": len(words),
        "unique_word_ratio": unique_word_ratio,
        "ngram_size": ngram_size,
        "repeated_ngram_total": repeated_ngram_total,
        "max_ngram_repetition": max_ngram_repetition,
        "consecutive_repeated_tokens": consecutive_repeated_tokens,
        "loop_score": loop_score,
    }


def _processor_for_run(run: Mapping[str, object]) -> Any | None:
    if run.get("kind") == "baseline":
        return None
    config = run.get("config")
    if not isinstance(config, Mapping):
        return None
    return build_zspace_repression_logits_processor(**dict(config))


def _processor_list(processor: Any, transformers: Any) -> Any:
    processor_list_type = getattr(transformers, "LogitsProcessorList", None)
    if processor_list_type is not None:
        return processor_list_type([processor])
    return [processor]


@contextlib.contextmanager
def _prepare_special_tokens_batch_size_compat(model: Any):
    prepare = getattr(model, "_prepare_special_tokens", None)
    if not callable(prepare):
        yield False
        return
    try:
        signature = inspect.signature(prepare)
    except (TypeError, ValueError):
        yield False
        return
    parameters = signature.parameters
    accepts_batch_size = "batch_size" in parameters or any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in parameters.values()
    )
    if accepts_batch_size:
        yield False
        return

    sentinel = object()
    previous = getattr(model, "_prepare_special_tokens", sentinel)

    def _compat_prepare_special_tokens(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("batch_size", None)
        return prepare(*args, **kwargs)

    try:
        setattr(model, "_prepare_special_tokens", _compat_prepare_special_tokens)
    except Exception:
        yield False
        return
    try:
        yield True
    finally:
        try:
            if previous is sentinel:
                delattr(model, "_prepare_special_tokens")
            else:
                setattr(model, "_prepare_special_tokens", previous)
        except Exception:
            pass


def _generate_one(
    *,
    run: Mapping[str, object],
    transformers: Any,
    torch: Any,
    tokenizer: Any,
    model: Any,
    encoded: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, object]:
    processor = _processor_for_run(run)
    batch = _move_to_device(encoded, _model_device(model))
    generate_kwargs: dict[str, object] = {
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": bool(args.do_sample),
    }
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is not None:
        generate_kwargs["pad_token_id"] = pad_token_id
    elif eos_token_id is not None:
        generate_kwargs["pad_token_id"] = eos_token_id
    if eos_token_id is not None:
        generate_kwargs["eos_token_id"] = eos_token_id
    if args.do_sample:
        generate_kwargs["temperature"] = float(args.sample_temperature)
        if int(args.sample_top_k) > 0:
            generate_kwargs["top_k"] = int(args.sample_top_k)
    if processor is not None:
        generate_kwargs["logits_processor"] = _processor_list(processor, transformers)

    with torch.no_grad():
        with _prepare_special_tokens_batch_size_compat(model):
            output_ids = model.generate(**batch, **generate_kwargs)
    first_output = _first_sequence(output_ids)
    text = tokenizer.decode(first_output, skip_special_tokens=True)
    continuation = text[len(args.prompt) :] if text.startswith(args.prompt) else text
    input_token_count = _last_dim(encoded.get("input_ids"))
    output_token_count = _last_dim(first_output)
    control = None
    if processor is not None:
        control = processor.report(limit=int(args.report_limit))
    generation = hf_finetune_generation_report(
        stage=str(run.get("name") or "generation"),
        prompt=args.prompt,
        generated_text=text,
        generated_continuation_text=continuation,
        input_token_count=input_token_count,
        output_token_count=output_token_count,
        max_new_tokens=args.max_new_tokens,
        generation_method=(
            "model.generate"
            if processor is None
            else "model.generate+zspace_repression_softmax"
        ),
        generation_control=control,
    )
    return {
        "name": run.get("name"),
        "kind": run.get("kind"),
        "config": run.get("config"),
        "status": generation.get("status"),
        "generation": generation,
        "repetition": text_repetition_report(continuation),
    }


def _summary(runs: Sequence[Mapping[str, object]]) -> dict[str, object]:
    baseline = next((row for row in runs if row.get("kind") == "baseline"), None)
    baseline_hash = None
    if isinstance(baseline, Mapping):
        generation = baseline.get("generation")
        if isinstance(generation, Mapping):
            baseline_hash = generation.get("generated_continuation_sha256")
    completed = [row for row in runs if row.get("status") == "ok"]
    changed_from_baseline = 0
    for row in completed:
        generation = row.get("generation")
        if not isinstance(generation, Mapping) or row.get("kind") == "baseline":
            continue
        row_hash = generation.get("generated_continuation_sha256")
        if baseline_hash and row_hash and row_hash != baseline_hash:
            changed_from_baseline += 1

    def loop_score(row: Mapping[str, object]) -> float:
        repetition = row.get("repetition")
        if not isinstance(repetition, Mapping):
            return math.inf
        value = repetition.get("loop_score")
        return math.inf if value is None else float(value)

    best_loop = min(completed, key=loop_score, default=None)
    control_changed_counts = []
    control_call_counts = []
    control_reported_rows = []
    control_entropy_mins = []
    control_entropy_maxes = []
    control_temperature_mins = []
    control_temperature_maxes = []
    control_ngram_repressed_totals = []
    control_max_ngram_repressions = []
    for row in completed:
        generation = row.get("generation")
        if not isinstance(generation, Mapping):
            continue
        control = generation.get("generation_control")
        if isinstance(control, Mapping):
            value = control.get("top_token_changed_count")
            if isinstance(value, (int, float)):
                control_changed_counts.append(float(value))
            value = control.get("calls")
            if isinstance(value, (int, float)):
                control_call_counts.append(float(value))
            value = control.get("reported_rows")
            if isinstance(value, (int, float)):
                control_reported_rows.append(float(value))
            value = control.get("entropy_min")
            if isinstance(value, (int, float)):
                control_entropy_mins.append(float(value))
            value = control.get("entropy_max")
            if isinstance(value, (int, float)):
                control_entropy_maxes.append(float(value))
            value = control.get("temperature_min")
            if isinstance(value, (int, float)):
                control_temperature_mins.append(float(value))
            value = control.get("temperature_max")
            if isinstance(value, (int, float)):
                control_temperature_maxes.append(float(value))
            value = control.get("ngram_repressed_token_total")
            if isinstance(value, (int, float)):
                control_ngram_repressed_totals.append(float(value))
            value = control.get("max_ngram_repression")
            if isinstance(value, (int, float)):
                control_max_ngram_repressions.append(float(value))
    return {
        "row_type": "hf_gpt2_zspace_generation_control_sweep_summary",
        "completed_run_count": len(completed),
        "changed_from_baseline_count": changed_from_baseline,
        "best_loop_score_run": None if best_loop is None else best_loop.get("name"),
        "best_loop_score": None if best_loop is None else loop_score(best_loop),
        "max_top_token_changed_count": (
            max(control_changed_counts) if control_changed_counts else None
        ),
        "max_control_calls": max(control_call_counts) if control_call_counts else None,
        "max_control_reported_rows": (
            max(control_reported_rows) if control_reported_rows else None
        ),
        "control_entropy_min": (
            min(control_entropy_mins) if control_entropy_mins else None
        ),
        "control_entropy_max": (
            max(control_entropy_maxes) if control_entropy_maxes else None
        ),
        "control_temperature_min": (
            min(control_temperature_mins) if control_temperature_mins else None
        ),
        "control_temperature_max": (
            max(control_temperature_maxes) if control_temperature_maxes else None
        ),
        "max_control_ngram_repressed_token_total": (
            max(control_ngram_repressed_totals)
            if control_ngram_repressed_totals
            else None
        ),
        "max_control_ngram_repression": (
            max(control_max_ngram_repressions)
            if control_max_ngram_repressions
            else None
        ),
    }


def run_sweep(args: argparse.Namespace) -> dict[str, object]:
    runs = build_control_runs(args)
    report: dict[str, object] = {
        "row_type": "hf_gpt2_zspace_generation_control_sweep",
        "status": "planned" if args.dry_run else "running",
        "model_name": args.model_name,
        "tokenizer_name": args.tokenizer_name or args.model_name,
        "model_configs": (
            None if args.model_configs is None else str(args.model_configs)
        ),
        "model_profile": getattr(args, "_hf_finetune_model_profile", None),
        "model_profile_lines": list(
            getattr(args, "_hf_finetune_model_profile_lines", [])
        ),
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": bool(args.do_sample),
        "sample_temperature": args.sample_temperature,
        "sample_top_k": args.sample_top_k,
        "dry_run": bool(args.dry_run),
        "run_count": len(runs),
        "runs": runs,
    }
    if args.dry_run:
        report["summary"] = _summary([])
        return report

    import transformers  # type: ignore
    import torch  # type: ignore

    with _hf_remote_access(args.allow_remote):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.tokenizer_name or args.model_name,
            **_loader_kwargs(args),
        )
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = getattr(tokenizer, "eos_token", None)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_name,
            **_loader_kwargs(args),
        )
    if getattr(tokenizer, "pad_token_id", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    eval_model = getattr(model, "eval", None)
    if callable(eval_model):
        eval_model()
    encoded = tokenizer(args.prompt, return_tensors="pt")
    completed_runs = []
    for run in runs:
        try:
            completed_runs.append(
                _generate_one(
                    run=run,
                    transformers=transformers,
                    torch=torch,
                    tokenizer=tokenizer,
                    model=model,
                    encoded=encoded,
                    args=args,
                )
            )
        except Exception as exc:
            failed = dict(run)
            failed.update(
                {
                    "status": "error",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )
            completed_runs.append(failed)
    report["status"] = (
        "complete"
        if all(row.get("status") == "ok" for row in completed_runs)
        else "partial"
    )
    report["completed_run_count"] = sum(
        1 for row in completed_runs if row.get("status") == "ok"
    )
    report["failed_run_count"] = sum(
        1 for row in completed_runs if row.get("status") != "ok"
    )
    report["runs"] = completed_runs
    report["summary"] = _summary(completed_runs)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_sweep(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"generation_control_sweep {args.out}")
    return 0 if report.get("status") in {"planned", "complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
