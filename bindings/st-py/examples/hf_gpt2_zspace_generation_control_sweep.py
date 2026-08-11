#!/usr/bin/env python3
"""Sweep SpiralTorch Z-Space generation controls on a Hugging Face CausalLM."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import operator
import os
import sys
from collections.abc import Mapping, Sequence
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
from spiraltorch.hf_generation import (
    build_zspace_repression_logits_processor,
    hf_generation_batch_size_compat,
    zspace_generation_control_bridge_cli_args,
    zspace_generation_control_processor_kwargs,
    zspace_generation_control_sweep_cli_args,
)
from spiraltorch.hf_peft import (
    load_hf_causal_lm_artifact,
    summarize_hf_causal_lm_artifact,
)
from spiraltorch.hf_adapter import hf_adapter_fingerprint
from spiraltorch.generation_evidence import (
    ZSPACE_GENERATION_EVIDENCE_LOOP_SCORE_RULE,
    ZSPACE_GENERATION_EVIDENCE_METRIC_RULE,
    ZSPACE_GENERATION_EVIDENCE_SEMANTIC_OWNER,
    zspace_generation_evidence,
)


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
    runtime = _mapping_or_empty(profile.get("runtime"))

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
    if "allow_remote" in runtime and not _argv_has_option(raw_argv, "--allow-remote"):
        args.allow_remote = bool(runtime.get("allow_remote"))
    if "trust_remote_code" in runtime and not _argv_has_option(
        raw_argv,
        "--trust-remote-code",
    ):
        args.trust_remote_code = bool(runtime.get("trust_remote_code"))
    set_scalar_if_missing(
        "max_new_tokens",
        "max_new_tokens",
        "--max-new-tokens",
        caster=_profile_int,
    )
    if "do_sample" in generation and not _argv_has_option(
        raw_argv,
        "--do-sample",
        "--no-do-sample",
    ):
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


def _load_generation_prompt_rows(
    *,
    prompt: object,
    prompt_set: Path | None,
) -> list[dict[str, str]]:
    if prompt_set is None:
        text = str(prompt or "")
        if not text:
            raise ValueError("--prompt must not be empty")
        return [
            {
                "label": "prompt-0001",
                "text": text,
                "prompt_id": _sha256_id(
                    {
                        "schema": "spiraltorch.hf_generation_prompt.v1",
                        "text": text,
                    }
                ),
            }
        ]
    path = prompt_set.expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read --prompt-set {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("--prompt-set must contain a JSON object")
    if payload.get("schema") != "spiraltorch.hf_generation_prompt_set.v1":
        raise ValueError("--prompt-set uses an unsupported schema")
    raw_prompts = payload.get("prompts")
    if not isinstance(raw_prompts, list) or not raw_prompts:
        raise ValueError("--prompt-set prompts must be a non-empty list")
    rows: list[dict[str, str]] = []
    labels: set[str] = set()
    prompt_ids: set[str] = set()
    for index, raw in enumerate(raw_prompts):
        if not isinstance(raw, Mapping):
            raise ValueError(f"--prompt-set prompt {index} must be a mapping")
        label = str(raw.get("label") or f"prompt-{index + 1:04d}").strip()
        text = str(raw.get("text") or "")
        if not label or len(label) > 64:
            raise ValueError(f"--prompt-set prompt {index} has an invalid label")
        if not text:
            raise ValueError(f"--prompt-set prompt {index} has empty text")
        prompt_id = _sha256_id(
            {"schema": "spiraltorch.hf_generation_prompt.v1", "text": text}
        )
        if label in labels:
            raise ValueError(f"--prompt-set has duplicate label {label!r}")
        if prompt_id in prompt_ids:
            raise ValueError("--prompt-set contains duplicate prompt text")
        labels.add(label)
        prompt_ids.add(prompt_id)
        rows.append({"label": label, "text": text, "prompt_id": prompt_id})
    expected_set_id = _sha256_id(
        {
            "schema": "spiraltorch.hf_generation_prompt_set.v1",
            "prompt_ids": [row["prompt_id"] for row in rows],
        }
    )
    supplied_set_id = payload.get("prompt_set_id")
    if supplied_set_id is not None and supplied_set_id != expected_set_id:
        raise ValueError("--prompt-set prompt_set_id does not match its prompt texts")
    return rows


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
    parser.add_argument(
        "--model-artifact-kind",
        choices=("auto", "full-model", "peft-adapter"),
        default="auto",
        help=(
            "Interpret --model-name as a full model or PEFT adapter. auto "
            "detects local adapter_config.json artifacts."
        ),
    )
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-set", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/hf-gpt2-zspace-generation-control-sweep.json"),
    )
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--do-sample", dest="do_sample", action="store_true")
    sampling.add_argument("--no-do-sample", dest="do_sample", action="store_false")
    parser.set_defaults(do_sample=False)
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--sample-top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--zspace-top-k-values", type=_positive_int_values, default=[64]
    )
    parser.add_argument(
        "--zspace-curvature-values", type=_negative_float_values, default=[-0.04]
    )
    parser.add_argument(
        "--zspace-temperature-values", type=_positive_float_values, default=[1.0]
    )
    parser.add_argument(
        "--zspace-entropy-target-values",
        type=_optional_float_values,
        default=[None, 3.0],
    )
    parser.add_argument(
        "--zspace-entropy-gain-values", type=_non_negative_float_values, default=[0.5]
    )
    parser.add_argument("--zspace-entropy-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--zspace-min-temperature", type=float, default=0.7)
    parser.add_argument("--zspace-max-temperature", type=float, default=2.4)
    parser.add_argument(
        "--repression-window-values", type=_positive_int_values, default=[16]
    )
    parser.add_argument(
        "--repression-strength-values",
        type=_non_negative_float_values,
        default=[0.0, 1.0],
    )
    parser.add_argument(
        "--last-token-repression-values",
        type=_non_negative_float_values,
        default=[0.0, 1.0],
    )
    parser.add_argument(
        "--ngram-size-values", type=_non_negative_int_values, default=[0]
    )
    parser.add_argument(
        "--ngram-window-values", type=_non_negative_int_values, default=[0]
    )
    parser.add_argument(
        "--ngram-repression-strength-values",
        type=_non_negative_float_values,
        default=[0.0],
    )
    parser.add_argument(
        "--ngram-decay-values", type=_unit_interval_float_values, default=[1.0]
    )
    parser.add_argument("--keep-non-top-k", action="store_true")
    parser.add_argument("--zspace-no-native", action="store_true")
    parser.add_argument("--report-limit", type=int, default=64)
    args = parser.parse_args(argv)
    _apply_model_profile_defaults(args, raw_argv)
    if (args.prompt is None) == (args.prompt_set is None):
        parser.error("pass exactly one of --prompt or --prompt-set")
    if args.no_baseline and args.baseline_only:
        parser.error("--no-baseline and --baseline-only cannot be combined")
    try:
        args._generation_evidence_prompts = _load_generation_prompt_rows(
            prompt=args.prompt,
            prompt_set=args.prompt_set,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.sample_temperature <= 0.0 or not math.isfinite(args.sample_temperature):
        parser.error("--sample-temperature must be finite and positive")
    if args.sample_top_k < 0:
        parser.error("--sample-top-k must be non-negative")
    if args.seed < 0 or args.seed > 9_007_199_254_740_991:
        parser.error("--seed must be a non-negative cross-client safe integer")
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
    if args.baseline_only:
        return runs
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


def _generation_control_grid(args: argparse.Namespace) -> dict[str, object]:
    return {
        "top_k_values": list(args.zspace_top_k_values),
        "curvature_values": list(args.zspace_curvature_values),
        "temperature_values": list(args.zspace_temperature_values),
        "entropy_target_values": list(args.zspace_entropy_target_values),
        "entropy_gain_values": list(args.zspace_entropy_gain_values),
        "entropy_tolerance": args.zspace_entropy_tolerance,
        "min_temperature": args.zspace_min_temperature,
        "max_temperature": args.zspace_max_temperature,
        "repression_window_values": list(args.repression_window_values),
        "repression_strength_values": list(args.repression_strength_values),
        "last_token_repression_values": list(args.last_token_repression_values),
        "ngram_size_values": list(args.ngram_size_values),
        "ngram_window_values": list(args.ngram_window_values),
        "ngram_repression_strength_values": list(args.ngram_repression_strength_values),
        "ngram_decay_values": list(args.ngram_decay_values),
        "mask_non_top_k": not bool(args.keep_non_top_k),
        "use_native_zspace": not bool(args.zspace_no_native),
    }


def _first_zspace_config(
    runs: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    for run in runs:
        if run.get("kind") == "baseline":
            continue
        config = run.get("config")
        if isinstance(config, Mapping):
            return dict(config)
    return None


def _generation_control_plan(
    args: argparse.Namespace,
    runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    profile = getattr(args, "_hf_finetune_model_profile", None)
    profile_config = (
        zspace_generation_control_processor_kwargs(profile)
        if isinstance(profile, Mapping)
        else {}
    )
    resolved_config = _first_zspace_config(runs)
    return {
        "profile_recommended_config": profile_config,
        "resolved_config": resolved_config,
        "grid": _generation_control_grid(args),
        "sweep_cli_args": zspace_generation_control_sweep_cli_args(resolved_config),
        "bridge_cli_args": zspace_generation_control_bridge_cli_args(resolved_config),
    }


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
        return {key: _move_to_device(item, device) for key, item in value.items()}
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


def _sha256_id(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256_id(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(
        character in "0123456789abcdef" for character in digest
    )


def _local_full_model_artifact_id(directory: Path) -> str:
    resolved = directory.expanduser().resolve()
    weight_suffixes = (".safetensors", ".bin", ".pt", ".pth")
    weight_paths = sorted(
        path
        for path in resolved.rglob("*")
        if path.is_file() and path.name.endswith(weight_suffixes)
    )
    if not weight_paths:
        raise RuntimeError(
            f"local full-model generation evidence found no weight files in {resolved}"
        )
    config_path = resolved / "config.json"
    return _sha256_id(
        {
            "schema": "spiraltorch.hf_full_model_artifact_identity.v1",
            "config_sha256": (
                _sha256_file(config_path) if config_path.is_file() else None
            ),
            "weights": [
                {
                    "name": path.relative_to(resolved).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
                for path in weight_paths
            ],
        }
    )


def _model_artifact_id(
    args: argparse.Namespace,
    artifact_summary: Mapping[str, object],
    runtime_identity_id: str,
) -> str:
    source = Path(str(args.model_name)).expanduser()
    artifact_kind = artifact_summary.get("artifact_kind")
    if source.is_dir() and artifact_kind == "peft_adapter":
        adapter_id = hf_adapter_fingerprint(source).get("adapter_id")
        if _is_sha256_id(adapter_id):
            return str(adapter_id)
        raise RuntimeError(
            "local PEFT adapter fingerprint did not return an adapter_id"
        )
    if source.is_dir():
        return _local_full_model_artifact_id(source)
    return _sha256_id(
        {
            "schema": "spiraltorch.hf_resolved_model_artifact_identity.v1",
            "artifact_kind": artifact_kind,
            "base_model_name_or_path": artifact_summary.get("base_model_name_or_path"),
            "base_model_revision": artifact_summary.get("base_model_revision"),
            "base_model_commit": artifact_summary.get("base_model_commit"),
            "runtime_identity_id": runtime_identity_id,
        }
    )


def _sequence_token_ids(value: Any) -> list[int]:
    current = value
    for operation_name in ("detach", "cpu"):
        operation = getattr(current, operation_name, None)
        if callable(operation):
            current = operation()
    tolist = getattr(current, "tolist", None)
    if callable(tolist):
        current = tolist()
    if not isinstance(current, (list, tuple)):
        try:
            current = list(current)
        except TypeError as error:
            raise TypeError("generated token sequence is not iterable") from error
    if current and isinstance(current[0], (list, tuple)):
        current = current[0]
    token_ids: list[int] = []
    for token in current:
        if isinstance(token, bool):
            raise TypeError("generated token IDs must be integers")
        try:
            token_id = operator.index(token)
        except TypeError as error:
            raise TypeError("generated token IDs must be integers") from error
        if token_id < 0:
            raise ValueError("generated token IDs must be non-negative")
        token_ids.append(token_id)
    return token_ids


def _generation_evidence_protocol_id() -> str:
    return _sha256_id(
        {
            "schema": "spiraltorch.hf_generation_evidence_protocol.v1",
            "metric_rule": ZSPACE_GENERATION_EVIDENCE_METRIC_RULE,
            "loop_score_rule": ZSPACE_GENERATION_EVIDENCE_LOOP_SCORE_RULE,
            "sample_unit": "model_artifact_prompt_seed_control",
            "continuation_only": True,
        }
    )


def _decoding_config_id(
    args: argparse.Namespace,
    run: Mapping[str, object],
) -> str:
    return _sha256_id(
        {
            "schema": "spiraltorch.hf_generation_decoding_config.v1",
            "max_new_tokens": int(args.max_new_tokens),
            "do_sample": bool(args.do_sample),
            "sample_temperature": (
                float(args.sample_temperature) if args.do_sample else None
            ),
            "sample_top_k": int(args.sample_top_k) if args.do_sample else None,
            "seed": int(args.seed),
            "run_kind": run.get("kind"),
            "control_config": dict(run.get("config") or {}),
        }
    )


def _generation_evidence_plan(
    args: argparse.Namespace,
    runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    prompt_rows = list(args._generation_evidence_prompts)
    prompt_ids = [str(row["prompt_id"]) for row in prompt_rows]
    return {
        "schema": "spiraltorch.hf_generation_evidence_plan.v1",
        "semantic_owner": ZSPACE_GENERATION_EVIDENCE_SEMANTIC_OWNER,
        "protocol_id": _generation_evidence_protocol_id(),
        "prompt_count": len(prompt_rows),
        "prompts": prompt_rows,
        "prompt_set_id": _sha256_id(
            {
                "schema": "spiraltorch.hf_generation_prompt_set.v1",
                "prompt_ids": prompt_ids,
            }
        ),
        "seed": int(args.seed),
        "continuation_only": True,
        "decoding_config_ids": {
            str(run.get("name")): _decoding_config_id(args, run) for run in runs
        },
    }


def _compatibility_repetition_report(
    evidence: Mapping[str, object],
    *,
    ngram_size: int = 3,
) -> dict[str, object]:
    samples = evidence.get("samples")
    aggregate = evidence.get("aggregate")
    if (
        not isinstance(samples, list)
        or not samples
        or not isinstance(aggregate, Mapping)
    ):
        raise RuntimeError("generation evidence compatibility projection is malformed")
    ngrams = aggregate.get("ngrams")
    if not isinstance(ngrams, list):
        raise RuntimeError("generation evidence n-gram metrics are malformed")
    selected = next(
        (
            row
            for row in ngrams
            if isinstance(row, Mapping) and row.get("order") == ngram_size
        ),
        None,
    )
    if not isinstance(selected, Mapping):
        raise ValueError("ngram_size must be one of the Rust evidence orders 1,2,3,4")
    unigram = next(
        (row for row in ngrams if isinstance(row, Mapping) and row.get("order") == 1),
        {},
    )
    return {
        "metric_backend": "rust",
        "generation_evidence_id": evidence.get("evidence_id"),
        "word_count": None,
        "sample_count": aggregate.get("sample_count"),
        "token_count": aggregate.get("total_token_count"),
        "unique_word_ratio": None,
        "distinct_token_ratio": unigram.get("distinct_ratio"),
        "ngram_size": ngram_size,
        "repeated_ngram_total": selected.get("repeated_occurrence_count"),
        "max_ngram_repetition": selected.get("maximum_occurrence_count"),
        "consecutive_repeated_tokens": aggregate.get(
            "consecutive_repeated_token_count"
        ),
        "periodic_loop_detected": bool(aggregate.get("periodic_loop_sample_count")),
        "periodic_loop_sample_count": aggregate.get("periodic_loop_sample_count"),
        "periodic_loop_sample_ratio": aggregate.get("periodic_loop_sample_ratio"),
        "periodic_suffix_period": (
            samples[0].get("periodic_suffix_period")
            if len(samples) == 1 and isinstance(samples[0], Mapping)
            else None
        ),
        "periodic_suffix_repeated_token_count": aggregate.get(
            "periodic_suffix_repeated_token_count"
        ),
        "loop_score": aggregate.get("sample_mean_loop_score"),
        "maximum_loop_score": aggregate.get("maximum_loop_score"),
        "loop_score_rule": evidence.get("loop_score_rule"),
    }


def text_repetition_report(text: object, *, ngram_size: int = 3) -> dict[str, object]:
    words = str(text or "").split()
    vocabulary: dict[str, int] = {}
    token_ids = [vocabulary.setdefault(word, len(vocabulary)) for word in words]
    text_id = _sha256_id(
        {"schema": "spiraltorch.whitespace_token_probe.v1", "words": words}
    )
    evidence = zspace_generation_evidence(
        protocol_id=_generation_evidence_protocol_id(),
        runtime_identity_id=_sha256_id("whitespace-token-probe-runtime"),
        model_artifact_id=_sha256_id("whitespace-token-probe-model"),
        prompt_set_id=text_id,
        decoding_config_id=_sha256_id(
            {
                "schema": "spiraltorch.whitespace_token_probe.decode.v1",
                "ngram_size": ngram_size,
            }
        ),
        samples=[
            {
                "prompt_id": text_id,
                "seed": 0,
                "continuation_token_ids": token_ids,
            }
        ],
    )
    report = _compatibility_repetition_report(evidence, ngram_size=ngram_size)
    report["word_count"] = len(words)
    report["unique_word_ratio"] = report["distinct_token_ratio"]
    return report


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


_prepare_special_tokens_batch_size_compat = hf_generation_batch_size_compat


def _reset_generation_seed(torch: Any, seed: int) -> None:
    manual_seed = getattr(torch, "manual_seed", None)
    if callable(manual_seed):
        manual_seed(seed)
    cuda = getattr(torch, "cuda", None)
    manual_seed_all = getattr(cuda, "manual_seed_all", None)
    if callable(manual_seed_all):
        manual_seed_all(seed)


def _generate_one(
    *,
    run: Mapping[str, object],
    transformers: Any,
    torch: Any,
    tokenizer: Any,
    model: Any,
    encoded: Mapping[str, Any],
    args: argparse.Namespace,
    evidence_context: Mapping[str, str],
    prompt: str,
    prompt_label: str,
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

    _reset_generation_seed(torch, int(args.seed))
    with torch.no_grad():
        with _prepare_special_tokens_batch_size_compat(model):
            output_ids = model.generate(**batch, **generate_kwargs)
    first_output = _first_sequence(output_ids)
    text = tokenizer.decode(first_output, skip_special_tokens=True)
    continuation = text[len(prompt) :] if text.startswith(prompt) else text
    input_token_count = _last_dim(encoded.get("input_ids"))
    output_token_count = _last_dim(first_output)
    output_token_ids = _sequence_token_ids(first_output)
    if input_token_count is None or not 0 <= input_token_count <= len(output_token_ids):
        raise RuntimeError("unable to isolate generated continuation token IDs")
    continuation_token_ids = output_token_ids[input_token_count:]
    control = None
    if processor is not None:
        control = processor.report(limit=int(args.report_limit))
    generation = hf_finetune_generation_report(
        stage=str(run.get("name") or "generation"),
        prompt=prompt,
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
    generation_evidence = zspace_generation_evidence(
        protocol_id=evidence_context["protocol_id"],
        runtime_identity_id=evidence_context["runtime_identity_id"],
        model_artifact_id=evidence_context["model_artifact_id"],
        prompt_set_id=evidence_context["prompt_set_id"],
        decoding_config_id=_decoding_config_id(args, run),
        samples=[
            {
                "prompt_id": evidence_context["prompt_id"],
                "seed": int(args.seed),
                "continuation_token_ids": continuation_token_ids,
            }
        ],
    )
    return {
        "name": run.get("name"),
        "kind": run.get("kind"),
        "config": run.get("config"),
        "prompt_label": prompt_label,
        "prompt_id": evidence_context["prompt_id"],
        "status": generation.get("status"),
        "generation": generation,
        "generation_evidence": generation_evidence,
        "repetition": _compatibility_repetition_report(generation_evidence),
    }


def _combine_prompt_generations(
    *,
    run: Mapping[str, object],
    generations: Sequence[Mapping[str, object]],
    args: argparse.Namespace,
    evidence_context: Mapping[str, str],
) -> dict[str, object]:
    if not generations:
        raise RuntimeError("generation prompt suite produced no rows")
    evidence_samples: list[Mapping[str, object]] = []
    for row in generations:
        if row.get("status") != "ok":
            raise RuntimeError("generation prompt row did not complete successfully")
        evidence = row.get("generation_evidence")
        request = evidence.get("request") if isinstance(evidence, Mapping) else None
        samples = request.get("samples") if isinstance(request, Mapping) else None
        if not isinstance(samples, list) or len(samples) != 1:
            raise RuntimeError("generation prompt row has invalid Rust evidence")
        evidence_samples.append(samples[0])
    generation_evidence = zspace_generation_evidence(
        protocol_id=evidence_context["protocol_id"],
        runtime_identity_id=evidence_context["runtime_identity_id"],
        model_artifact_id=evidence_context["model_artifact_id"],
        prompt_set_id=evidence_context["prompt_set_id"],
        decoding_config_id=_decoding_config_id(args, run),
        samples=evidence_samples,
    )
    canonical_request = generation_evidence.get("request")
    canonical_samples = (
        canonical_request.get("samples")
        if isinstance(canonical_request, Mapping)
        else None
    )
    if not isinstance(canonical_samples, list):
        raise RuntimeError("combined Rust generation evidence has no canonical samples")
    first = generations[0]
    return {
        "name": run.get("name"),
        "kind": run.get("kind"),
        "config": run.get("config"),
        "status": "ok",
        "prompt_count": len(generations),
        "generation": first.get("generation"),
        "generations": [dict(row) for row in generations],
        "generated_continuation_set_id": _sha256_id(
            {
                "schema": "spiraltorch.hf_generated_continuation_set.v1",
                "prompt_set_id": evidence_context["prompt_set_id"],
                "samples": canonical_samples,
            }
        ),
        "generation_evidence": generation_evidence,
        "repetition": _compatibility_repetition_report(generation_evidence),
    }


def _summary(runs: Sequence[Mapping[str, object]]) -> dict[str, object]:
    baseline = next((row for row in runs if row.get("kind") == "baseline"), None)
    baseline_hash = None
    if isinstance(baseline, Mapping):
        baseline_hash = baseline.get("generated_continuation_set_id")
        if baseline_hash is None:
            generation = baseline.get("generation")
            if isinstance(generation, Mapping):
                baseline_hash = generation.get("generated_continuation_sha256")
    completed = [row for row in runs if row.get("status") == "ok"]
    changed_from_baseline = 0
    for row in completed:
        if row.get("kind") == "baseline":
            continue
        row_hash = row.get("generated_continuation_set_id")
        if row_hash is None:
            generation = row.get("generation")
            row_hash = (
                generation.get("generated_continuation_sha256")
                if isinstance(generation, Mapping)
                else None
            )
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
        prompt_rows = row.get("generations")
        candidate_rows = prompt_rows if isinstance(prompt_rows, list) else [row]
        for candidate in candidate_rows:
            if not isinstance(candidate, Mapping):
                continue
            generation = candidate.get("generation")
            if not isinstance(generation, Mapping):
                continue
            control = generation.get("generation_control")
            if not isinstance(control, Mapping):
                continue
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
    prompt_rows = list(args._generation_evidence_prompts)
    generation_control_plan = _generation_control_plan(args, runs)
    generation_evidence_plan = _generation_evidence_plan(args, runs)
    report: dict[str, object] = {
        "row_type": "hf_gpt2_zspace_generation_control_sweep",
        "status": "planned" if args.dry_run else "running",
        "model_name": args.model_name,
        "tokenizer_name": args.tokenizer_name or args.model_name,
        "model_artifact_kind_requested": args.model_artifact_kind,
        "model_artifact_report": None,
        "model_configs": (
            None if args.model_configs is None else str(args.model_configs)
        ),
        "model_profile": getattr(args, "_hf_finetune_model_profile", None),
        "model_profile_lines": list(
            getattr(args, "_hf_finetune_model_profile_lines", [])
        ),
        "allow_remote": bool(args.allow_remote),
        "trust_remote_code": bool(args.trust_remote_code),
        "prompt": prompt_rows[0]["text"] if len(prompt_rows) == 1 else None,
        "prompt_count": len(prompt_rows),
        "prompt_set": {
            "prompt_set_id": generation_evidence_plan["prompt_set_id"],
            "prompts": prompt_rows,
        },
        "max_new_tokens": args.max_new_tokens,
        "do_sample": bool(args.do_sample),
        "sample_temperature": args.sample_temperature,
        "sample_top_k": args.sample_top_k,
        "seed": args.seed,
        "dry_run": bool(args.dry_run),
        "run_count": len(runs),
        "generation_control_profile_config": generation_control_plan[
            "profile_recommended_config"
        ],
        "generation_control_resolved_config": generation_control_plan[
            "resolved_config"
        ],
        "generation_control_grid": generation_control_plan["grid"],
        "generation_control_sweep_cli_args": generation_control_plan["sweep_cli_args"],
        "generation_control_bridge_cli_args": generation_control_plan[
            "bridge_cli_args"
        ],
        "generation_evidence_plan": generation_evidence_plan,
        "runs": runs,
    }
    if args.dry_run:
        report["summary"] = _summary([])
        return report

    import transformers  # type: ignore
    import torch  # type: ignore

    with _hf_remote_access(args.allow_remote):
        model, tokenizer, _config, model_artifact_report = load_hf_causal_lm_artifact(
            args.model_name,
            tokenizer_name_or_path=args.tokenizer_name,
            artifact_kind=args.model_artifact_kind,
            transformers_module=transformers,
            loader_kwargs=_loader_kwargs(args),
        )
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = getattr(tokenizer, "eos_token", None)
    artifact_summary = summarize_hf_causal_lm_artifact(model_artifact_report)
    report["model_artifact_report"] = artifact_summary
    report["model_artifact_kind"] = model_artifact_report.get("artifact_kind")
    report["model_adapter_loaded"] = model_artifact_report.get("adapter_loaded")
    report["tokenizer_name"] = model_artifact_report.get("resolved_tokenizer_source")
    runtime_identity_id = artifact_summary.get("runtime_identity_observed_id")
    if not _is_sha256_id(runtime_identity_id):
        raise RuntimeError(
            "generation evidence requires a verified model/tokenizer runtime identity"
        )
    evidence_context = {
        "protocol_id": str(generation_evidence_plan["protocol_id"]),
        "runtime_identity_id": str(runtime_identity_id),
        "model_artifact_id": _model_artifact_id(
            args,
            artifact_summary,
            str(runtime_identity_id),
        ),
        "prompt_set_id": str(generation_evidence_plan["prompt_set_id"]),
    }
    report["generation_evidence_context"] = evidence_context
    if getattr(tokenizer, "pad_token_id", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    eval_model = getattr(model, "eval", None)
    if callable(eval_model):
        eval_model()
    completed_runs = []
    for run in runs:
        try:
            prompt_generations = []
            for prompt_row in prompt_rows:
                prompt = str(prompt_row["text"])
                encoded = tokenizer(prompt, return_tensors="pt")
                prompt_generations.append(
                    _generate_one(
                        run=run,
                        transformers=transformers,
                        torch=torch,
                        tokenizer=tokenizer,
                        model=model,
                        encoded=encoded,
                        args=args,
                        evidence_context={
                            **evidence_context,
                            "prompt_id": str(prompt_row["prompt_id"]),
                        },
                        prompt=prompt,
                        prompt_label=str(prompt_row["label"]),
                    )
                )
            completed_runs.append(
                _combine_prompt_generations(
                    run=run,
                    generations=prompt_generations,
                    args=args,
                    evidence_context=evidence_context,
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
