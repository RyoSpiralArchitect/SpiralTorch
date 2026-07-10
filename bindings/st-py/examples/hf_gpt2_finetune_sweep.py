#!/usr/bin/env python3
"""Run small local GPT-2 fine-tune sweeps and compare SpiralTorch run cards."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import spiraltorch as st


DEFAULT_BRIDGE = Path(__file__).resolve().with_name("hf_gpt2_finetune_bridge.py")


def _csv_ints(value: str, *, name: str) -> list[int]:
    values = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must contain at least one integer")
    return values


def _csv_floats(value: str, *, name: str) -> list[float]:
    values = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must contain at least one float")
    return values


def _positive_int_values(value: str) -> list[int]:
    values = _csv_ints(value, name="value list")
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def _int_values(value: str) -> list[int]:
    return _csv_ints(value, name="value list")


def _positive_float_values(value: str) -> list[float]:
    values = _csv_floats(value, name="value list")
    if any(item <= 0.0 for item in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser.add_argument("--bridge-script", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/hf-gpt2-ft-sweep"))
    parser.add_argument("--run-prefix", default="gpt2-ft")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--scale-up-command-path",
        type=Path,
        default=None,
        help=(
            "Write the distortion-adjusted scale-up replay command artifact here. "
            "Defaults to OUT_DIR/scale-up-command.json."
        ),
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Reuse existing successful per-run run cards and run only missing/failed rows.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun all rows even when --resume-existing would find a reusable card.",
    )
    parser.add_argument(
        "--model-configs",
        type=Path,
        default=None,
        help=(
            "Optional JSON config with Hugging Face model profiles. "
            "Omit to use SpiralTorch's built-in local smoke profiles."
        ),
    )
    parser.add_argument(
        "--model-profile",
        default=None,
        help=(
            "Model profile id to use as sweep defaults for model/tokenizer, "
            "block-size, sample limits, and generation settings."
        ),
    )
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument(
        "--model-artifact-kind",
        choices=("auto", "full-model", "peft-adapter"),
        default=None,
    )
    parser.add_argument(
        "--finetune-mode",
        choices=("full", "lora"),
        default=None,
    )
    parser.add_argument("--dataset-name", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--dataset-streaming", action="store_true")
    parser.add_argument("--streaming-shuffle-buffer-size", type=int, default=0)
    parser.add_argument("--streaming-validation-samples", type=int, default=0)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--train-file", action="append", type=Path, default=[])
    parser.add_argument("--validation-file", action="append", type=Path, default=[])
    parser.add_argument(
        "--dataset-format",
        choices=("text", "json", "csv"),
        default="text",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--corpus-scan", action="store_true")
    parser.add_argument("--corpus-scan-max-bytes-per-file", type=int, default=0)
    parser.add_argument("--corpus-scan-sample-lines", type=int, default=8)
    parser.add_argument("--eval-before-train", action="store_true")
    parser.add_argument("--no-eval-after-train", action="store_true")
    parser.add_argument(
        "--eval-after-train-policy",
        choices=("always", "never", "skip-if-final-step-eval"),
        default="always",
    )
    parser.add_argument("--adapter-promotion-gate", action="store_true")
    parser.add_argument(
        "--adapter-promotion-max-eval-loss-regression",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--adapter-promotion-require-generation-change",
        action="store_true",
    )
    parser.add_argument("--adapter-promotion-probe-prompt", default=None)
    parser.add_argument(
        "--adapter-promotion-probe-max-new-tokens",
        type=int,
        default=8,
    )
    parser.add_argument("--adapter-promotion-probe-device", default="auto")
    parser.add_argument(
        "--adapter-promotion-probe-timeout-seconds",
        type=float,
        default=900.0,
    )
    parser.add_argument("--no-trainer-trace", action="store_true")
    parser.add_argument("--trainer-telemetry", action="store_true")
    parser.add_argument("--trainer-telemetry-prefix", default="hf_ft")
    parser.add_argument("--trainer-desire-gain", type=float, default=1.0)
    parser.add_argument("--trainer-psi-gain", type=float, default=1.0)
    parser.add_argument(
        "--inference-distortion-sweep-report",
        type=Path,
        default=None,
        help=(
            "Forward a Z-Space inference-distortion sweep recommendation into "
            "each FT bridge run card and trainer trace."
        ),
    )
    parser.add_argument(
        "--inference-distortion-probe",
        type=Path,
        default=None,
        help=(
            "Forward one saved Z-Space inference-distortion probe directly into "
            "each FT bridge run card and trainer trace."
        ),
    )
    parser.add_argument("--generation-prompt", default=None)
    parser.add_argument("--generation-max-new-tokens", type=int, default=16)
    parser.add_argument("--generation-do-sample", action="store_true")
    parser.add_argument("--generation-temperature", type=float, default=1.0)
    parser.add_argument("--generation-top-k", type=int, default=0)
    parser.add_argument("--generation-zspace-softmax", action="store_true")
    parser.add_argument("--generation-from-inference-distortion", action="store_true")
    parser.add_argument("--generation-zspace-top-k", type=int, default=64)
    parser.add_argument("--generation-zspace-curvature", type=float, default=-0.04)
    parser.add_argument("--generation-zspace-temperature", type=float, default=1.0)
    parser.add_argument("--generation-zspace-entropy-target", type=float, default=None)
    parser.add_argument("--generation-zspace-entropy-gain", type=float, default=0.5)
    parser.add_argument(
        "--generation-zspace-entropy-tolerance",
        type=float,
        default=1.0e-4,
    )
    parser.add_argument("--generation-zspace-min-temperature", type=float, default=None)
    parser.add_argument("--generation-zspace-max-temperature", type=float, default=None)
    parser.add_argument("--generation-repression-window", type=int, default=32)
    parser.add_argument("--generation-repression-strength", type=float, default=1.0)
    parser.add_argument("--generation-last-token-repression", type=float, default=0.5)
    parser.add_argument("--generation-ngram-size", type=int, default=0)
    parser.add_argument("--generation-ngram-window", type=int, default=0)
    parser.add_argument("--generation-ngram-repression-strength", type=float, default=0.0)
    parser.add_argument("--generation-ngram-decay", type=float, default=1.0)
    parser.add_argument("--generation-zspace-report-limit", type=int, default=64)
    parser.add_argument("--generation-zspace-keep-non-top-k", action="store_true")
    parser.add_argument("--generation-zspace-no-native", action="store_true")
    parser.add_argument("--runtime-device-backend", action="append", default=[])
    parser.add_argument(
        "--require-runtime-device-ready-backend",
        action="append",
        default=[],
    )
    parser.add_argument("--require-wgpu-ready", action="store_true")
    parser.add_argument(
        "--no-require-hf-finetune",
        "--no-require-hf-gpt2-ft",
        dest="no_require_hf_gpt2_ft",
        action="store_true",
    )
    parser.add_argument("--zspace-probe", action="store_true")
    parser.add_argument("--zspace-probe-dim", type=int, default=64)
    parser.add_argument("--zspace-curvature", type=float, default=-0.04)
    parser.add_argument("--zspace-frequency", type=float, default=0.65)
    parser.add_argument("--zspace-strength", type=float, default=1.0)
    parser.add_argument("--block-size-values", type=_positive_int_values, default=[128])
    parser.add_argument(
        "--learning-rate-values",
        type=_positive_float_values,
        default=[5e-5],
    )
    parser.add_argument("--max-step-values", type=_int_values, default=[1])
    parser.add_argument("--seed-values", type=_int_values, default=[13])
    parser.add_argument("--max-train-samples", type=int, default=4096)
    parser.add_argument("--max-eval-samples", type=int, default=512)
    parser.add_argument("--max-eval-blocks", type=int, default=0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--min-free-disk-gb", type=float, default=0.0)
    parser.add_argument("--eval-accumulation-steps", type=int, default=0)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument(
        "--dataloader-pin-memory",
        choices=("auto", "true", "false"),
        default="auto",
    )
    args = parser.parse_args(argv)
    _apply_model_profile_defaults(args, parser=parser, raw_argv=raw_argv)
    if not args.bridge_script.is_file():
        parser.error(f"--bridge-script does not exist: {args.bridge_script}")
    if args.validation_file and not args.train_file:
        parser.error("--validation-file requires --train-file")
    if args.validation_file and args.validation_fraction > 0.0:
        parser.error(
            "--validation-file and --validation-fraction are mutually exclusive"
        )
    if args.validation_fraction < 0.0 or args.validation_fraction >= 1.0:
        parser.error("--validation-fraction must be in [0.0, 1.0)")
    if args.dataset_streaming and args.train_file:
        parser.error("--dataset-streaming is only supported for remote HF datasets")
    if args.dataset_streaming and (
        args.max_train_samples is None or args.max_train_samples <= 0
    ):
        parser.error("--dataset-streaming requires a positive --max-train-samples")
    if args.streaming_shuffle_buffer_size < 0:
        parser.error("--streaming-shuffle-buffer-size must be non-negative")
    if args.streaming_validation_samples < 0:
        parser.error("--streaming-validation-samples must be non-negative")
    if not math.isfinite(args.adapter_promotion_max_eval_loss_regression):
        parser.error("--adapter-promotion-max-eval-loss-regression must be finite")
    if args.adapter_promotion_gate:
        if args.finetune_mode != "lora":
            parser.error(
                "--adapter-promotion-gate requires --finetune-mode lora "
                "or a LoRA model profile"
            )
        if not args.eval_before_train:
            parser.error("--adapter-promotion-gate requires --eval-before-train")
        if args.no_eval_after_train or args.eval_after_train_policy == "never":
            parser.error("--adapter-promotion-gate requires after-train evaluation")
    if (
        args.adapter_promotion_require_generation_change
        and not args.adapter_promotion_gate
    ):
        parser.error(
            "--adapter-promotion-require-generation-change requires "
            "--adapter-promotion-gate"
        )
    if args.adapter_promotion_require_generation_change and not args.generation_prompt:
        parser.error(
            "--adapter-promotion-require-generation-change requires "
            "--generation-prompt"
        )
    if args.adapter_promotion_probe_max_new_tokens <= 0:
        parser.error("--adapter-promotion-probe-max-new-tokens must be positive")
    if not str(args.adapter_promotion_probe_device).strip():
        parser.error("--adapter-promotion-probe-device must not be empty")
    if (
        not math.isfinite(args.adapter_promotion_probe_timeout_seconds)
        or args.adapter_promotion_probe_timeout_seconds <= 0.0
    ):
        parser.error("--adapter-promotion-probe-timeout-seconds must be positive")
    if (
        args.adapter_promotion_gate
        and args.adapter_promotion_probe_prompt is not None
        and not str(args.adapter_promotion_probe_prompt)
    ):
        parser.error("--adapter-promotion-probe-prompt must not be empty")
    if args.generation_max_new_tokens <= 0:
        parser.error("--generation-max-new-tokens must be positive")
    if args.generation_temperature <= 0.0:
        parser.error("--generation-temperature must be positive")
    if args.generation_top_k < 0:
        parser.error("--generation-top-k must be non-negative")
    if args.generation_zspace_softmax:
        if args.generation_zspace_top_k <= 0:
            parser.error("--generation-zspace-top-k must be positive")
        if args.generation_zspace_curvature >= 0.0 or not math.isfinite(
            args.generation_zspace_curvature
        ):
            parser.error("--generation-zspace-curvature must be finite and negative")
        if args.generation_zspace_temperature <= 0.0 or not math.isfinite(
            args.generation_zspace_temperature
        ):
            parser.error("--generation-zspace-temperature must be finite and positive")
        if args.generation_zspace_entropy_target is not None and not math.isfinite(
            args.generation_zspace_entropy_target
        ):
            parser.error("--generation-zspace-entropy-target must be finite")
        if args.generation_zspace_entropy_gain < 0.0 or not math.isfinite(
            args.generation_zspace_entropy_gain
        ):
            parser.error("--generation-zspace-entropy-gain must be finite and non-negative")
        if args.generation_zspace_entropy_tolerance < 0.0 or not math.isfinite(
            args.generation_zspace_entropy_tolerance
        ):
            parser.error(
                "--generation-zspace-entropy-tolerance must be finite and non-negative"
            )
        if args.generation_zspace_min_temperature is not None and (
            args.generation_zspace_min_temperature <= 0.0
            or not math.isfinite(args.generation_zspace_min_temperature)
        ):
            parser.error("--generation-zspace-min-temperature must be finite and positive")
        if args.generation_zspace_max_temperature is not None and (
            args.generation_zspace_max_temperature <= 0.0
            or not math.isfinite(args.generation_zspace_max_temperature)
        ):
            parser.error("--generation-zspace-max-temperature must be finite and positive")
        if (
            args.generation_zspace_min_temperature is not None
            and args.generation_zspace_max_temperature is not None
            and args.generation_zspace_min_temperature
            > args.generation_zspace_max_temperature
        ):
            parser.error(
                "--generation-zspace-min-temperature must be <= "
                "--generation-zspace-max-temperature"
            )
        if args.generation_repression_window < 0:
            parser.error("--generation-repression-window must be non-negative")
        if args.generation_zspace_report_limit < 0:
            parser.error("--generation-zspace-report-limit must be non-negative")
        if args.generation_repression_strength < 0.0 or not math.isfinite(
            args.generation_repression_strength
        ):
            parser.error("--generation-repression-strength must be finite and non-negative")
        if args.generation_last_token_repression < 0.0 or not math.isfinite(
            args.generation_last_token_repression
        ):
            parser.error(
                "--generation-last-token-repression must be finite and non-negative"
            )
        if args.generation_ngram_size < 0:
            parser.error("--generation-ngram-size must be non-negative")
        if args.generation_ngram_window < 0:
            parser.error("--generation-ngram-window must be non-negative")
        if args.generation_ngram_repression_strength < 0.0 or not math.isfinite(
            args.generation_ngram_repression_strength
        ):
            parser.error(
                "--generation-ngram-repression-strength must be finite and non-negative"
            )
        if (
            args.generation_ngram_decay < 0.0
            or args.generation_ngram_decay > 1.0
            or not math.isfinite(args.generation_ngram_decay)
        ):
            parser.error("--generation-ngram-decay must be finite and in [0.0, 1.0]")
    if args.max_train_samples < 0 or args.max_eval_samples < 0:
        parser.error("--max-train-samples and --max-eval-samples must be non-negative")
    if args.max_eval_blocks < 0:
        parser.error("--max-eval-blocks must be non-negative")
    if args.save_total_limit is not None and args.save_total_limit <= 0:
        parser.error("--save-total-limit must be positive")
    if args.min_free_disk_gb < 0.0 or not math.isfinite(args.min_free_disk_gb):
        parser.error("--min-free-disk-gb must be finite and non-negative")
    if args.eval_accumulation_steps < 0:
        parser.error("--eval-accumulation-steps must be non-negative")
    if args.dataloader_num_workers < 0:
        parser.error("--dataloader-num-workers must be non-negative")
    if args.trainer_telemetry and args.no_trainer_trace:
        parser.error("--trainer-telemetry requires trainer tracing")
    if args.trainer_telemetry:
        if not str(args.trainer_telemetry_prefix).strip():
            parser.error("--trainer-telemetry-prefix must be non-empty")
        if args.trainer_desire_gain < 0.0 or not math.isfinite(args.trainer_desire_gain):
            parser.error("--trainer-desire-gain must be finite and non-negative")
        if args.trainer_psi_gain < 0.0 or not math.isfinite(args.trainer_psi_gain):
            parser.error("--trainer-psi-gain must be finite and non-negative")
    if (
        args.inference_distortion_sweep_report is not None
        and not args.inference_distortion_sweep_report.is_file()
    ):
        parser.error(
            "--inference-distortion-sweep-report does not exist: "
            f"{args.inference_distortion_sweep_report}"
        )
    if (
        args.inference_distortion_probe is not None
        and not args.inference_distortion_probe.is_file()
    ):
        parser.error(
            "--inference-distortion-probe does not exist: "
            f"{args.inference_distortion_probe}"
        )
    if (
        args.inference_distortion_sweep_report is not None
        and args.inference_distortion_probe is not None
    ):
        parser.error(
            "--inference-distortion-sweep-report and --inference-distortion-probe "
            "are mutually exclusive"
        )
    if (
        args.generation_from_inference_distortion
        and args.inference_distortion_sweep_report is None
        and args.inference_distortion_probe is None
    ):
        parser.error(
            "--generation-from-inference-distortion requires "
            "--inference-distortion-probe or --inference-distortion-sweep-report"
        )
    if args.generation_from_inference_distortion and not args.generation_prompt:
        parser.error("--generation-from-inference-distortion requires --generation-prompt")
    if args.corpus_scan and not args.train_file:
        parser.error("--corpus-scan requires --train-file")
    if args.corpus_scan_max_bytes_per_file < 0:
        parser.error("--corpus-scan-max-bytes-per-file must be non-negative")
    if args.corpus_scan_sample_lines < 0:
        parser.error("--corpus-scan-sample-lines must be non-negative")
    if args.zspace_probe_dim <= 0:
        parser.error("--zspace-probe-dim must be positive")
    for path in [*args.train_file, *args.validation_file]:
        if not path.is_file():
            parser.error(f"local corpus file does not exist: {path}")
    return args


def _argv_has_option(raw_argv: Sequence[str], *names: str) -> bool:
    prefixes = tuple(f"{name}=" for name in names)
    return any(arg in names or arg.startswith(prefixes) for arg in raw_argv)


def _profile_value(profile: Mapping[str, Any], section: str, key: str) -> Any:
    payload = profile.get(section)
    if isinstance(payload, Mapping):
        return payload.get(key)
    return None


def _profile_section(profile: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    payload = profile.get(section)
    return payload if isinstance(payload, Mapping) else {}


def _set_profile_default(
    args: argparse.Namespace,
    raw_argv: Sequence[str],
    attr: str,
    value: Any,
    *flags: str,
) -> None:
    if value is None or _argv_has_option(raw_argv, *flags):
        return
    setattr(args, attr, value)


def _set_profile_default_if_present(
    args: argparse.Namespace,
    raw_argv: Sequence[str],
    section: Mapping[str, Any],
    key: str,
    attr: str,
    *flags: str,
) -> None:
    if key not in section or _argv_has_option(raw_argv, *flags):
        return
    setattr(args, attr, section.get(key))


def _profile_path_list(value: Any) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        raw_values = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw_values = list(value)
    else:
        raw_values = [value]
    return [Path(str(item)) for item in raw_values if str(item)]


def _profile_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_values = value.split(",")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw_values = [str(item) for item in value]
    else:
        raw_values = [str(value)]
    return [item.strip() for item in raw_values if item.strip()]


def _apply_model_profile_defaults(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser,
    raw_argv: Sequence[str],
) -> None:
    args._hf_finetune_model_profile = None
    if args.model_configs is None and args.model_profile is None:
        return
    try:
        profile = st.resolve_hf_finetune_model_profile(
            args.model_configs,
            profile=args.model_profile,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(f"failed to resolve model profile: {exc}")
    args._hf_finetune_model_profile = profile
    _set_profile_default(
        args,
        raw_argv,
        "model_name",
        profile.get("model_name"),
        "--model-name",
    )
    _set_profile_default(
        args,
        raw_argv,
        "model_artifact_kind",
        profile.get("artifact_kind") or profile.get("model_artifact_kind"),
        "--model-artifact-kind",
    )
    if _argv_has_option(raw_argv, "--model-name") and not _argv_has_option(
        raw_argv,
        "--tokenizer-name",
    ):
        args.tokenizer_name = args.model_name
    else:
        _set_profile_default(
            args,
            raw_argv,
            "tokenizer_name",
            profile.get("tokenizer_name"),
            "--tokenizer-name",
        )
    training = _profile_section(profile, "training")
    adapter = _profile_section(profile, "adapter")
    _set_profile_default(
        args,
        raw_argv,
        "finetune_mode",
        training.get("finetune_mode") or adapter.get("type"),
        "--finetune-mode",
    )
    block_size = training.get("block_size") or profile.get("max_length")
    if block_size is not None:
        _set_profile_default(
            args,
            raw_argv,
            "block_size_values",
            [block_size],
            "--block-size-values",
        )
    for attr, key, flag in (
        ("max_train_samples", "max_train_samples", "--max-train-samples"),
        ("max_eval_samples", "max_eval_samples", "--max-eval-samples"),
        (
            "per_device_train_batch_size",
            "per_device_train_batch_size",
            "--per-device-train-batch-size",
        ),
        (
            "per_device_eval_batch_size",
            "per_device_eval_batch_size",
            "--per-device-eval-batch-size",
        ),
        (
            "gradient_accumulation_steps",
            "gradient_accumulation_steps",
            "--gradient-accumulation-steps",
        ),
        ("learning_rate_values", "learning_rate", "--learning-rate-values"),
        ("max_step_values", "max_steps", "--max-step-values"),
        ("logging_steps", "logging_steps", "--logging-steps"),
        ("save_steps", "save_steps", "--save-steps"),
        ("save_total_limit", "save_total_limit", "--save-total-limit"),
        ("eval_steps", "eval_steps", "--eval-steps"),
        (
            "eval_accumulation_steps",
            "eval_accumulation_steps",
            "--eval-accumulation-steps",
        ),
    ):
        value = _profile_value(profile, "training", key)
        if value is not None and attr in {"learning_rate_values", "max_step_values"}:
            value = [value]
        _set_profile_default(args, raw_argv, attr, value, flag)
    dataset = _profile_section(profile, "dataset")
    for attr, key, flag in (
        ("dataset_name", "name", "--dataset-name"),
        ("dataset_revision", "revision", "--dataset-revision"),
        ("train_split", "train_split", "--train-split"),
        ("eval_split", "eval_split", "--eval-split"),
        ("text_column", "text_column", "--text-column"),
        ("dataset_format", "format", "--dataset-format"),
        ("validation_fraction", "validation_fraction", "--validation-fraction"),
        (
            "streaming_shuffle_buffer_size",
            "streaming_shuffle_buffer_size",
            "--streaming-shuffle-buffer-size",
        ),
        (
            "streaming_validation_samples",
            "streaming_validation_samples",
            "--streaming-validation-samples",
        ),
    ):
        _set_profile_default_if_present(args, raw_argv, dataset, key, attr, flag)
    _set_profile_default_if_present(
        args,
        raw_argv,
        dataset,
        "config",
        "dataset_config",
        "--dataset-config",
    )
    _set_profile_default_if_present(
        args,
        raw_argv,
        dataset,
        "streaming",
        "dataset_streaming",
        "--dataset-streaming",
    )
    if "train_files" in dataset and not _argv_has_option(raw_argv, "--train-file"):
        args.train_file = _profile_path_list(dataset.get("train_files"))
    if "validation_files" in dataset and not _argv_has_option(
        raw_argv,
        "--validation-file",
    ):
        args.validation_file = _profile_path_list(dataset.get("validation_files"))
    for attr, key, flag in (
        ("generation_max_new_tokens", "max_new_tokens", "--generation-max-new-tokens"),
        ("generation_temperature", "temperature", "--generation-temperature"),
        ("generation_top_k", "top_k", "--generation-top-k"),
        ("generation_zspace_top_k", "zspace_top_k", "--generation-zspace-top-k"),
        (
            "generation_zspace_curvature",
            "zspace_curvature",
            "--generation-zspace-curvature",
        ),
        (
            "generation_zspace_temperature",
            "zspace_temperature",
            "--generation-zspace-temperature",
        ),
        (
            "generation_zspace_entropy_target",
            "zspace_entropy_target",
            "--generation-zspace-entropy-target",
        ),
        (
            "generation_zspace_entropy_gain",
            "zspace_entropy_gain",
            "--generation-zspace-entropy-gain",
        ),
        (
            "generation_zspace_entropy_tolerance",
            "zspace_entropy_tolerance",
            "--generation-zspace-entropy-tolerance",
        ),
        (
            "generation_zspace_min_temperature",
            "zspace_min_temperature",
            "--generation-zspace-min-temperature",
        ),
        (
            "generation_zspace_max_temperature",
            "zspace_max_temperature",
            "--generation-zspace-max-temperature",
        ),
        (
            "generation_repression_window",
            "repression_window",
            "--generation-repression-window",
        ),
        (
            "generation_repression_strength",
            "repression_strength",
            "--generation-repression-strength",
        ),
        (
            "generation_last_token_repression",
            "last_token_repression",
            "--generation-last-token-repression",
        ),
        ("generation_ngram_size", "ngram_size", "--generation-ngram-size"),
        ("generation_ngram_window", "ngram_window", "--generation-ngram-window"),
        (
            "generation_ngram_repression_strength",
            "ngram_repression_strength",
            "--generation-ngram-repression-strength",
        ),
        ("generation_ngram_decay", "ngram_decay", "--generation-ngram-decay"),
        (
            "generation_zspace_report_limit",
            "zspace_report_limit",
            "--generation-zspace-report-limit",
        ),
    ):
        _set_profile_default(
            args,
            raw_argv,
            attr,
            _profile_value(profile, "generation", key),
            flag,
        )
    for attr, key, flag in (
        ("generation_do_sample", "do_sample", "--generation-do-sample"),
        ("generation_zspace_softmax", "zspace_softmax", "--generation-zspace-softmax"),
        (
            "generation_zspace_keep_non_top_k",
            "zspace_keep_non_top_k",
            "--generation-zspace-keep-non-top-k",
        ),
        (
            "generation_zspace_no_native",
            "zspace_no_native",
            "--generation-zspace-no-native",
        ),
    ):
        value = _profile_value(profile, "generation", key)
        if value is True and not _argv_has_option(raw_argv, flag):
            setattr(args, attr, True)
    runtime = _profile_section(profile, "runtime")
    for attr, key, flag in (
        ("allow_remote", "allow_remote", "--allow-remote"),
        ("trust_remote_code", "trust_remote_code", "--trust-remote-code"),
        ("min_free_disk_gb", "min_free_disk_gb", "--min-free-disk-gb"),
        ("dataloader_pin_memory", "dataloader_pin_memory", "--dataloader-pin-memory"),
        (
            "dataloader_num_workers",
            "dataloader_num_workers",
            "--dataloader-num-workers",
        ),
        ("require_wgpu_ready", "require_wgpu_ready", "--require-wgpu-ready"),
        (
            "no_require_hf_gpt2_ft",
            "no_require_hf_finetune",
            "--no-require-hf-finetune",
        ),
        (
            "no_require_hf_gpt2_ft",
            "no_require_hf_gpt2_ft",
            "--no-require-hf-gpt2-ft",
        ),
    ):
        _set_profile_default_if_present(args, raw_argv, runtime, key, attr, flag)
    if "runtime_device_backends" in runtime and not _argv_has_option(
        raw_argv,
        "--runtime-device-backend",
    ):
        args.runtime_device_backend = _profile_string_list(
            runtime.get("runtime_device_backends")
        )
    if "required_runtime_device_ready_backends" in runtime and not _argv_has_option(
        raw_argv,
        "--require-runtime-device-ready-backend",
    ):
        args.require_runtime_device_ready_backend = _profile_string_list(
            runtime.get("required_runtime_device_ready_backends")
        )


def _append_repeated(command: list[str], flag: str, values: list[Path]) -> None:
    for value in values:
        command.extend([flag, str(value)])


def _run_name(
    args: argparse.Namespace,
    *,
    index: int,
    block_size: int,
    learning_rate: float,
    max_steps: int,
    seed: int,
) -> str:
    lr_label = f"{learning_rate:.0e}".replace("+", "").replace("-", "m")
    return (
        f"{args.run_prefix}-{index:03d}"
        f"-bs{block_size}-lr{lr_label}-steps{max_steps}-seed{seed}"
    )


def _bridge_command(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    run_card: Path,
    trainer_trace: Path,
    block_size: int,
    learning_rate: float,
    max_steps: int,
    seed: int,
) -> list[str]:
    command = [
        str(args.python),
        str(args.bridge_script),
        "--model-name",
        str(args.model_name),
        "--tokenizer-name",
        str(args.tokenizer_name or args.model_name),
        "--train",
        "--output-dir",
        str(run_dir),
        "--run-card",
        str(run_card),
        "--trainer-trace-jsonl",
        str(trainer_trace),
        "--text-column",
        str(args.text_column),
        "--block-size",
        str(block_size),
        "--learning-rate",
        str(learning_rate),
        "--max-steps",
        str(max_steps),
        "--seed",
        str(seed),
        "--max-train-samples",
        str(args.max_train_samples),
        "--max-eval-samples",
        str(args.max_eval_samples),
        "--max-eval-blocks",
        str(args.max_eval_blocks),
        "--per-device-train-batch-size",
        str(args.per_device_train_batch_size),
        "--per-device-eval-batch-size",
        str(args.per_device_eval_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--logging-steps",
        str(args.logging_steps),
        "--save-steps",
        str(args.save_steps),
        "--eval-steps",
        str(args.eval_steps),
        "--min-free-disk-gb",
        str(args.min_free_disk_gb),
        "--eval-accumulation-steps",
        str(args.eval_accumulation_steps),
        "--dataloader-num-workers",
        str(args.dataloader_num_workers),
        "--dataloader-pin-memory",
        str(args.dataloader_pin_memory),
    ]
    if args.model_artifact_kind is not None:
        command.extend(
            [
                "--model-artifact-kind",
                str(args.model_artifact_kind).replace("_", "-"),
            ]
        )
    if args.finetune_mode is not None:
        command.extend(["--finetune-mode", str(args.finetune_mode)])
    if args.save_total_limit is not None:
        command.extend(["--save-total-limit", str(args.save_total_limit)])
    if args.model_configs is not None:
        command.extend(["--model-configs", str(args.model_configs)])
    if args.model_profile is not None:
        command.extend(["--model-profile", str(args.model_profile)])
    if args.train_file:
        _append_repeated(command, "--train-file", args.train_file)
        _append_repeated(command, "--validation-file", args.validation_file)
        command.extend(["--dataset-format", str(args.dataset_format)])
        if args.validation_fraction > 0.0:
            command.extend(["--validation-fraction", str(args.validation_fraction)])
    else:
        command.extend(
            [
                "--dataset-name",
                str(args.dataset_name),
                "--train-split",
                str(args.train_split),
                "--eval-split",
                str(args.eval_split),
            ]
        )
        command.extend(
            [
                "--dataset-config",
                "" if args.dataset_config is None else str(args.dataset_config),
            ]
        )
        if args.dataset_revision is not None:
            command.extend(["--dataset-revision", str(args.dataset_revision)])
        if args.dataset_streaming:
            command.append("--dataset-streaming")
            command.extend(
                [
                    "--streaming-shuffle-buffer-size",
                    str(args.streaming_shuffle_buffer_size),
                    "--streaming-validation-samples",
                    str(args.streaming_validation_samples),
                ]
            )
    if args.allow_remote:
        command.append("--allow-remote")
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if args.corpus_scan:
        command.append("--corpus-scan")
        command.extend(
            [
                "--corpus-scan-max-bytes-per-file",
                str(args.corpus_scan_max_bytes_per_file),
                "--corpus-scan-sample-lines",
                str(args.corpus_scan_sample_lines),
            ]
        )
    if args.eval_before_train:
        command.append("--eval-before-train")
    command.extend(["--eval-after-train-policy", str(args.eval_after_train_policy)])
    if args.no_eval_after_train:
        command.append("--no-eval-after-train")
    if args.adapter_promotion_gate:
        command.append("--adapter-promotion-gate")
        command.extend(
            [
                "--adapter-promotion-max-eval-loss-regression",
                str(args.adapter_promotion_max_eval_loss_regression),
            ]
        )
        if args.adapter_promotion_require_generation_change:
            command.append("--adapter-promotion-require-generation-change")
        if args.adapter_promotion_probe_prompt is not None:
            command.extend(
                [
                    "--adapter-promotion-probe-prompt",
                    str(args.adapter_promotion_probe_prompt),
                ]
            )
        command.extend(
            [
                "--adapter-promotion-probe-max-new-tokens",
                str(args.adapter_promotion_probe_max_new_tokens),
                "--adapter-promotion-probe-device",
                str(args.adapter_promotion_probe_device),
                "--adapter-promotion-probe-timeout-seconds",
                str(args.adapter_promotion_probe_timeout_seconds),
            ]
        )
    if args.no_trainer_trace:
        command.append("--no-trainer-trace")
    if args.trainer_telemetry:
        command.append("--trainer-telemetry")
        command.extend(["--trainer-telemetry-prefix", str(args.trainer_telemetry_prefix)])
        command.extend(["--trainer-desire-gain", str(args.trainer_desire_gain)])
        command.extend(["--trainer-psi-gain", str(args.trainer_psi_gain)])
    if args.inference_distortion_sweep_report is not None:
        command.extend(
            [
                "--inference-distortion-sweep-report",
                str(args.inference_distortion_sweep_report),
            ]
        )
    if args.inference_distortion_probe is not None:
        command.extend(
            [
                "--inference-distortion-probe",
                str(args.inference_distortion_probe),
            ]
        )
    if args.generation_prompt:
        command.extend(["--generation-prompt", str(args.generation_prompt)])
        command.extend(
            ["--generation-max-new-tokens", str(args.generation_max_new_tokens)]
        )
        if args.generation_do_sample:
            command.append("--generation-do-sample")
        command.extend(["--generation-temperature", str(args.generation_temperature)])
        command.extend(["--generation-top-k", str(args.generation_top_k)])
        if args.generation_from_inference_distortion:
            command.append("--generation-from-inference-distortion")
        if args.generation_zspace_softmax:
            command.append("--generation-zspace-softmax")
            command.extend(["--generation-zspace-top-k", str(args.generation_zspace_top_k)])
            command.extend(
                ["--generation-zspace-curvature", str(args.generation_zspace_curvature)]
            )
            command.extend(
                [
                    "--generation-zspace-temperature",
                    str(args.generation_zspace_temperature),
                ]
            )
            if args.generation_zspace_entropy_target is not None:
                command.extend(
                    [
                        "--generation-zspace-entropy-target",
                        str(args.generation_zspace_entropy_target),
                    ]
                )
            command.extend(
                [
                    "--generation-zspace-entropy-gain",
                    str(args.generation_zspace_entropy_gain),
                ]
            )
            command.extend(
                [
                    "--generation-zspace-entropy-tolerance",
                    str(args.generation_zspace_entropy_tolerance),
                ]
            )
            if args.generation_zspace_min_temperature is not None:
                command.extend(
                    [
                        "--generation-zspace-min-temperature",
                        str(args.generation_zspace_min_temperature),
                    ]
                )
            if args.generation_zspace_max_temperature is not None:
                command.extend(
                    [
                        "--generation-zspace-max-temperature",
                        str(args.generation_zspace_max_temperature),
                    ]
                )
            command.extend(
                [
                    "--generation-repression-window",
                    str(args.generation_repression_window),
                ]
            )
            command.extend(
                [
                    "--generation-repression-strength",
                    str(args.generation_repression_strength),
                ]
            )
            command.extend(
                [
                    "--generation-last-token-repression",
                    str(args.generation_last_token_repression),
                ]
            )
            command.extend(["--generation-ngram-size", str(args.generation_ngram_size)])
            command.extend(
                ["--generation-ngram-window", str(args.generation_ngram_window)]
            )
            command.extend(
                [
                    "--generation-ngram-repression-strength",
                    str(args.generation_ngram_repression_strength),
                ]
            )
            command.extend(
                ["--generation-ngram-decay", str(args.generation_ngram_decay)]
            )
            command.extend(
                [
                    "--generation-zspace-report-limit",
                    str(args.generation_zspace_report_limit),
                ]
            )
            if args.generation_zspace_keep_non_top_k:
                command.append("--generation-zspace-keep-non-top-k")
            if args.generation_zspace_no_native:
                command.append("--generation-zspace-no-native")
    for backend in args.runtime_device_backend:
        command.extend(["--runtime-device-backend", str(backend)])
    for backend in args.require_runtime_device_ready_backend:
        command.extend(["--require-runtime-device-ready-backend", str(backend)])
    if args.require_wgpu_ready:
        command.append("--require-wgpu-ready")
    if args.no_require_hf_gpt2_ft:
        command.append("--no-require-hf-finetune")
    if args.zspace_probe:
        command.append("--zspace-probe")
        command.extend(["--zspace-probe-dim", str(args.zspace_probe_dim)])
        command.extend(["--zspace-curvature", str(args.zspace_curvature)])
        command.extend(["--zspace-frequency", str(args.zspace_frequency)])
        command.extend(["--zspace-strength", str(args.zspace_strength)])
    return command


def _resolved_model_profile(args: argparse.Namespace) -> dict[str, Any] | None:
    profile = getattr(args, "_hf_finetune_model_profile", None)
    return dict(profile) if isinstance(profile, Mapping) else None


def _resolved_model_profile_lines(args: argparse.Namespace) -> list[str]:
    profile = _resolved_model_profile(args)
    if profile is None:
        return []
    return st.hf_finetune_model_profile_lines(profile)


def build_sweep_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    runs = []
    grid = product(
        args.block_size_values,
        args.learning_rate_values,
        args.max_step_values,
        args.seed_values,
    )
    for index, (block_size, learning_rate, max_steps, seed) in enumerate(grid, 1):
        name = _run_name(
            args,
            index=index,
            block_size=block_size,
            learning_rate=learning_rate,
            max_steps=max_steps,
            seed=seed,
        )
        run_dir = args.out_dir / name
        run_card = run_dir / str(
            getattr(args, "run_card_filename", st.HF_GPT2_FT_RUN_CARD_FILENAME)
        )
        trainer_trace = run_dir / str(
            getattr(
                args,
                "trainer_trace_filename",
                st.HF_GPT2_FT_TRAINER_TRACE_FILENAME,
            )
        )
        command = _bridge_command(
            args,
            run_dir=run_dir,
            run_card=run_card,
            trainer_trace=trainer_trace,
            block_size=block_size,
            learning_rate=learning_rate,
            max_steps=max_steps,
            seed=seed,
        )
        runs.append(
            {
                "name": name,
                "index": index,
                "block_size": block_size,
                "learning_rate": learning_rate,
                "max_steps": max_steps,
                "seed": seed,
                "model_name": str(args.model_name),
                "tokenizer_name": str(args.tokenizer_name or args.model_name),
                "model_artifact_kind": args.model_artifact_kind,
                "finetune_mode": args.finetune_mode,
                "adapter_promotion_gate_requested": bool(
                    args.adapter_promotion_gate
                ),
                "adapter_promotion_max_eval_loss_regression": (
                    args.adapter_promotion_max_eval_loss_regression
                ),
                "adapter_promotion_require_generation_change": bool(
                    args.adapter_promotion_require_generation_change
                ),
                "adapter_promotion_require_artifact_probe": bool(
                    args.adapter_promotion_gate
                ),
                "adapter_promotion_probe_prompt": (
                    args.adapter_promotion_probe_prompt
                    or args.generation_prompt
                    or "SpiralTorch is"
                ),
                "adapter_promotion_probe_max_new_tokens": (
                    args.adapter_promotion_probe_max_new_tokens
                ),
                "adapter_promotion_probe_device": (
                    args.adapter_promotion_probe_device
                ),
                "adapter_promotion_probe_timeout_seconds": (
                    args.adapter_promotion_probe_timeout_seconds
                ),
                "dataset_name": str(args.dataset_name),
                "dataset_config": args.dataset_config,
                "dataset_revision": args.dataset_revision,
                "dataset_streaming": bool(args.dataset_streaming),
                "streaming_shuffle_buffer_size": args.streaming_shuffle_buffer_size,
                "streaming_validation_samples": args.streaming_validation_samples,
                "min_free_disk_gb": args.min_free_disk_gb,
                "model_configs": (
                    None if args.model_configs is None else str(args.model_configs)
                ),
                "model_profile_id": (
                    None
                    if _resolved_model_profile(args) is None
                    else _resolved_model_profile(args).get("profile_id")
                ),
                "run_dir": str(run_dir),
                "run_card": str(run_card),
                "trainer_trace_jsonl": str(trainer_trace),
                "command": command,
                "command_display": shlex.join(command),
                "returncode": None,
                "status": "planned",
                "reused": False,
            }
        )
    return runs


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _scale_up_command_path(args: argparse.Namespace) -> Path:
    return args.scale_up_command_path or (args.out_dir / "scale-up-command.json")


def _attach_scale_up_command_artifact(
    args: argparse.Namespace,
    report: dict[str, Any],
) -> dict[str, Any]:
    summary = report.get("summary")
    if not isinstance(summary, dict):
        summary = st.summarize_hf_gpt2_finetune_sweep_report(report)
        report["summary"] = summary
    scale_up_command = st.hf_gpt2_finetune_scale_up_command(summary)
    scale_up_path = _scale_up_command_path(args)
    scale_up_command["artifact_path"] = str(scale_up_path)
    report["scale_up_command_path"] = str(scale_up_path)
    report["scale_up_command_status"] = scale_up_command.get("status")
    report["scale_up_command_preview"] = scale_up_command.get("command_preview")
    report["scale_up_command"] = scale_up_command
    _write_json(scale_up_path, scale_up_command)
    report["summary"] = st.summarize_hf_gpt2_finetune_sweep_report(report)
    return scale_up_command


def _is_reusable_run_card(
    path: Path,
    *,
    require_adapter_promotion: bool = False,
) -> bool:
    if not path.is_file():
        return False
    try:
        card = st.load_hf_gpt2_finetune_run_card(path)
    except (OSError, ValueError):
        return False
    if card.get("row_type") != "hf_gpt2_finetune_run_card":
        return False
    if card.get("failure_stage") or card.get("failure_error"):
        return False
    if card.get("load_status") == "error":
        return False
    dataset_fit = card.get("dataset_fit_report")
    if isinstance(dataset_fit, dict) and dataset_fit.get("train_ready") is False:
        return False
    promotion = card.get("adapter_promotion")
    if require_adapter_promotion and (
        not isinstance(promotion, Mapping)
        or promotion.get("promotion_ready") is not True
    ):
        return False
    return True


def _adapter_promotion_evidence(path: Path) -> dict[str, object]:
    try:
        card = st.load_hf_gpt2_finetune_run_card(path)
    except (OSError, ValueError):
        return {}
    promotion = card.get("adapter_promotion")
    lineage = card.get("adapter_lineage")
    artifact_probe = card.get("adapter_artifact_probe")
    tokenizer_save = card.get("tokenizer_save_report")
    promotion_payload = promotion if isinstance(promotion, Mapping) else {}
    lineage_payload = lineage if isinstance(lineage, Mapping) else {}
    artifact_probe_payload = (
        artifact_probe if isinstance(artifact_probe, Mapping) else {}
    )
    artifact_probe_process = artifact_probe_payload.get("process_isolation")
    artifact_probe_process_payload = (
        artifact_probe_process
        if isinstance(artifact_probe_process, Mapping)
        else {}
    )
    tokenizer_save_payload = (
        tokenizer_save if isinstance(tokenizer_save, Mapping) else {}
    )
    tokenizer_files = tokenizer_save_payload.get("files")
    tokenizer_file_count = (
        len(tokenizer_files)
        if isinstance(tokenizer_files, Sequence)
        and not isinstance(tokenizer_files, (str, bytes))
        else 0
    )
    return {
        "adapter_lineage_status": lineage_payload.get("status"),
        "adapter_lineage_depth": lineage_payload.get("lineage_depth"),
        "adapter_promotion_status": promotion_payload.get("status"),
        "adapter_promotion_ready": promotion_payload.get("promotion_ready"),
        "adapter_promotion_recommendation": promotion_payload.get("recommendation"),
        "adapter_promotion_failed_checks": promotion_payload.get("failed_checks"),
        "adapter_promotion_missing_checks": promotion_payload.get("missing_checks"),
        "adapter_promotion_require_artifact_probe": promotion_payload.get(
            "require_artifact_probe"
        ),
        "adapter_artifact_probe_status": artifact_probe_payload.get("status"),
        "adapter_artifact_probe_report_path": artifact_probe_payload.get(
            "report_path"
        ),
        "adapter_artifact_probe_device": artifact_probe_payload.get("device"),
        "adapter_artifact_probe_new_token_count": artifact_probe_payload.get(
            "new_token_count"
        ),
        "adapter_artifact_probe_process_status": (
            artifact_probe_process_payload.get("status")
        ),
        "adapter_artifact_probe_process_fresh": (
            artifact_probe_process_payload.get("fresh_process")
        ),
        "adapter_artifact_probe_process_parent_pid": (
            artifact_probe_process_payload.get("parent_pid")
        ),
        "adapter_artifact_probe_process_pid": artifact_probe_process_payload.get(
            "pid"
        ),
        "adapter_artifact_probe_process_exit_code": (
            artifact_probe_process_payload.get("exit_code")
        ),
        "adapter_artifact_probe_process_timed_out": (
            artifact_probe_process_payload.get("timed_out")
        ),
        "adapter_artifact_probe_process_duration_seconds": (
            artifact_probe_process_payload.get("duration_seconds")
        ),
        "tokenizer_save_status": tokenizer_save_payload.get("status"),
        "tokenizer_save_file_count": tokenizer_file_count,
    }


def _inference_distortion_report_path(args: argparse.Namespace) -> str | None:
    return (
        None
        if args.inference_distortion_sweep_report is None
        else str(args.inference_distortion_sweep_report)
    )


def _inference_distortion_probe_path(args: argparse.Namespace) -> str | None:
    return (
        None
        if args.inference_distortion_probe is None
        else str(args.inference_distortion_probe)
    )


def _adapter_promotion_policy(args: argparse.Namespace) -> dict[str, object]:
    return {
        "model_artifact_kind": args.model_artifact_kind,
        "finetune_mode": args.finetune_mode,
        "adapter_promotion_gate_requested": bool(args.adapter_promotion_gate),
        "adapter_promotion_max_eval_loss_regression": (
            args.adapter_promotion_max_eval_loss_regression
        ),
        "adapter_promotion_require_generation_change": bool(
            args.adapter_promotion_require_generation_change
        ),
        "adapter_promotion_require_artifact_probe": bool(
            args.adapter_promotion_gate
        ),
        "adapter_promotion_probe_prompt": (
            args.adapter_promotion_probe_prompt
            or args.generation_prompt
            or "SpiralTorch is"
        ),
        "adapter_promotion_probe_max_new_tokens": (
            args.adapter_promotion_probe_max_new_tokens
        ),
        "adapter_promotion_probe_device": args.adapter_promotion_probe_device,
        "adapter_promotion_probe_timeout_seconds": (
            args.adapter_promotion_probe_timeout_seconds
        ),
    }


def _inference_distortion_source_path(args: argparse.Namespace) -> str | None:
    return _inference_distortion_report_path(args) or _inference_distortion_probe_path(args)


def _inference_distortion_handoff(args: argparse.Namespace) -> dict[str, Any] | None:
    source_path = _inference_distortion_source_path(args)
    if source_path is None:
        return None
    return st.hf_gpt2_finetune_inference_distortion_handoff_report(
        source_path,
    )


def _inference_distortion_handoff_lines(
    inference_handoff: dict[str, Any] | None,
) -> list[str]:
    if not isinstance(inference_handoff, dict):
        return []
    return st.hf_gpt2_finetune_inference_distortion_handoff_lines(
        inference_handoff,
    )


def _trainer_telemetry_auto_reason(
    args: argparse.Namespace,
    inference_handoff: dict[str, Any] | None,
) -> str | None:
    if args.trainer_telemetry or args.no_trainer_trace:
        return None
    if isinstance(inference_handoff, dict):
        return "inference_distortion_handoff"
    return None


def _trainer_telemetry_enabled(
    args: argparse.Namespace,
    inference_handoff: dict[str, Any] | None,
) -> bool:
    if args.no_trainer_trace:
        return False
    return bool(args.trainer_telemetry) or (
        _trainer_telemetry_auto_reason(args, inference_handoff) is not None
    )


def _generation_from_inference_distortion_plan(
    args: argparse.Namespace,
    inference_handoff: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not args.generation_from_inference_distortion:
        return None
    if not isinstance(inference_handoff, dict):
        return {"status": "missing_handoff"}
    processor_kwargs = inference_handoff.get("recommended_processor_kwargs")
    if not isinstance(processor_kwargs, dict) or not processor_kwargs:
        return {
            "status": "missing_processor_kwargs",
            "recommended_probe": inference_handoff.get("recommended_probe"),
            "source_kind": inference_handoff.get("source_kind"),
        }
    bridge_cli_args = st.zspace_generation_control_bridge_cli_args(
        processor_kwargs,
        include_enable_flag=True,
    )
    return {
        "status": "ok",
        "source_kind": inference_handoff.get("source_kind"),
        "recommended_probe": inference_handoff.get("recommended_probe"),
        "applied_arg_count": len(processor_kwargs),
        "processor_kwargs": dict(processor_kwargs),
        "bridge_cli_args": bridge_cli_args,
    }


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    runs = build_sweep_runs(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_profile = _resolved_model_profile(args)
    model_profile_lines = _resolved_model_profile_lines(args)
    inference_handoff = _inference_distortion_handoff(args)
    inference_handoff_lines = _inference_distortion_handoff_lines(inference_handoff)
    generation_inference_plan = _generation_from_inference_distortion_plan(
        args,
        inference_handoff,
    )
    trainer_telemetry_enabled = _trainer_telemetry_enabled(args, inference_handoff)
    trainer_telemetry_auto_reason = _trainer_telemetry_auto_reason(
        args,
        inference_handoff,
    )
    plan = {
        "row_type": "hf_gpt2_finetune_sweep_plan",
        "dry_run": bool(args.dry_run),
        "model_name": str(args.model_name),
        "tokenizer_name": str(args.tokenizer_name or args.model_name),
        "dataset_name": str(args.dataset_name),
        "dataset_config": args.dataset_config,
        "dataset_revision": args.dataset_revision,
        "dataset_streaming": bool(args.dataset_streaming),
        "streaming_shuffle_buffer_size": args.streaming_shuffle_buffer_size,
        "streaming_validation_samples": args.streaming_validation_samples,
        "min_free_disk_gb": args.min_free_disk_gb,
        **_adapter_promotion_policy(args),
        "model_configs": None if args.model_configs is None else str(args.model_configs),
        "model_profile": model_profile,
        "model_profile_lines": model_profile_lines,
        "inference_distortion_sweep_report": _inference_distortion_report_path(args),
        "inference_distortion_probe": _inference_distortion_probe_path(args),
        "inference_distortion_handoff": inference_handoff,
        "inference_distortion_handoff_lines": inference_handoff_lines,
        "trainer_telemetry_requested": bool(args.trainer_telemetry),
        "trainer_telemetry_enabled": trainer_telemetry_enabled,
        "trainer_telemetry_auto_reason": trainer_telemetry_auto_reason,
        "generation_from_inference_distortion": bool(
            args.generation_from_inference_distortion
        ),
        "generation_from_inference_distortion_plan": generation_inference_plan,
        "run_count": len(runs),
        "runs": runs,
    }
    _write_json(args.out_dir / "sweep-plan.json", plan)
    if args.dry_run:
        report = {
            "row_type": "hf_gpt2_finetune_sweep_report",
            "dry_run": True,
            "run_count": len(runs),
            "attempted_run_count": 0,
            "completed_run_count": 0,
            "promotion_evaluated_run_count": 0,
            "failed_run_count": 0,
            "reused_run_count": 0,
            "skipped_run_count": len(runs),
            "plan_path": str(args.out_dir / "sweep-plan.json"),
            "report_path": str(args.out_dir / "sweep-report.json"),
            "model_name": str(args.model_name),
            "tokenizer_name": str(args.tokenizer_name or args.model_name),
            "dataset_name": str(args.dataset_name),
            "dataset_config": args.dataset_config,
            "dataset_revision": args.dataset_revision,
            "dataset_streaming": bool(args.dataset_streaming),
            "streaming_shuffle_buffer_size": args.streaming_shuffle_buffer_size,
            "streaming_validation_samples": args.streaming_validation_samples,
            "min_free_disk_gb": args.min_free_disk_gb,
            **_adapter_promotion_policy(args),
            "model_configs": (
                None if args.model_configs is None else str(args.model_configs)
            ),
            "model_profile": model_profile,
            "model_profile_lines": model_profile_lines,
            "inference_distortion_sweep_report": _inference_distortion_report_path(args),
            "inference_distortion_probe": _inference_distortion_probe_path(args),
            "inference_distortion_handoff": inference_handoff,
            "inference_distortion_handoff_lines": inference_handoff_lines,
            "trainer_telemetry_requested": bool(args.trainer_telemetry),
            "trainer_telemetry_enabled": trainer_telemetry_enabled,
            "trainer_telemetry_auto_reason": trainer_telemetry_auto_reason,
            "generation_from_inference_distortion": bool(
                args.generation_from_inference_distortion
            ),
            "generation_from_inference_distortion_plan": generation_inference_plan,
            "comparison": None,
            "runs": runs,
        }
        report["summary"] = st.summarize_hf_gpt2_finetune_sweep_report(report)
        _attach_scale_up_command_artifact(args, report)
        _write_json(args.out_dir / "sweep-report.json", report)
        return report

    completed_cards = []
    completed_labels = []
    comparison_cards = []
    comparison_labels = []
    for run in runs:
        run_card = Path(str(run["run_card"]))
        if (
            args.resume_existing
            and not args.force
            and _is_reusable_run_card(
                run_card,
                require_adapter_promotion=bool(args.adapter_promotion_gate),
            )
        ):
            print(f"sweep_reuse {run['name']}")
            run["returncode"] = 0
            run["status"] = "reused"
            run["reused"] = True
            run.update(_adapter_promotion_evidence(run_card))
            completed_cards.append(str(run_card))
            completed_labels.append(str(run["name"]))
            comparison_cards.append(str(run_card))
            comparison_labels.append(str(run["name"]))
            continue
        Path(str(run["run_dir"])).mkdir(parents=True, exist_ok=True)
        print(f"sweep_run {run['name']}")
        result = subprocess.run(run["command"], check=False)
        run["returncode"] = int(result.returncode)
        evidence = _adapter_promotion_evidence(run_card) if run_card.is_file() else {}
        run.update(evidence)
        if result.returncode == 0 and run_card.is_file():
            run["status"] = "completed"
            completed_cards.append(str(run_card))
            completed_labels.append(str(run["name"]))
            comparison_cards.append(str(run_card))
            comparison_labels.append(str(run["name"]))
        else:
            run["status"] = "failed"
            run["reused"] = False
            if args.adapter_promotion_gate and evidence.get(
                "adapter_promotion_status"
            ) in {"blocked", "needs_evidence"}:
                comparison_cards.append(str(run_card))
                comparison_labels.append(str(run["name"]))
            if args.fail_fast:
                break

    comparison = (
        st.compare_hf_gpt2_finetune_run_cards(
            comparison_cards,
            run_labels=comparison_labels,
        )
        if comparison_cards
        else None
    )
    attempted_run_count = sum(
        1
        for run in runs
        if run.get("returncode") is not None and run.get("status") != "reused"
    )
    reused_run_count = sum(1 for run in runs if run.get("status") == "reused")
    failed_run_count = sum(
        1
        for run in runs
        if run.get("returncode") is not None
        and run.get("status") != "reused"
        and int(run["returncode"]) != 0
    )
    report = {
        "row_type": "hf_gpt2_finetune_sweep_report",
        "dry_run": False,
        "run_count": len(runs),
        "attempted_run_count": attempted_run_count,
        "completed_run_count": len(completed_cards),
        "promotion_evaluated_run_count": sum(
            1 for run in runs if run.get("adapter_promotion_status") is not None
        ),
        "failed_run_count": failed_run_count,
        "reused_run_count": reused_run_count,
        "skipped_run_count": sum(1 for run in runs if run.get("status") == "planned"),
        "plan_path": str(args.out_dir / "sweep-plan.json"),
        "report_path": str(args.out_dir / "sweep-report.json"),
        "model_name": str(args.model_name),
        "tokenizer_name": str(args.tokenizer_name or args.model_name),
        "dataset_name": str(args.dataset_name),
        "dataset_config": args.dataset_config,
        "dataset_revision": args.dataset_revision,
        "dataset_streaming": bool(args.dataset_streaming),
        "streaming_shuffle_buffer_size": args.streaming_shuffle_buffer_size,
        "streaming_validation_samples": args.streaming_validation_samples,
        "min_free_disk_gb": args.min_free_disk_gb,
        **_adapter_promotion_policy(args),
        "model_configs": None if args.model_configs is None else str(args.model_configs),
        "model_profile": model_profile,
        "model_profile_lines": model_profile_lines,
        "inference_distortion_sweep_report": _inference_distortion_report_path(args),
        "inference_distortion_probe": _inference_distortion_probe_path(args),
        "inference_distortion_handoff": inference_handoff,
        "inference_distortion_handoff_lines": inference_handoff_lines,
        "trainer_telemetry_requested": bool(args.trainer_telemetry),
        "trainer_telemetry_enabled": trainer_telemetry_enabled,
        "trainer_telemetry_auto_reason": trainer_telemetry_auto_reason,
        "generation_from_inference_distortion": bool(
            args.generation_from_inference_distortion
        ),
        "generation_from_inference_distortion_plan": generation_inference_plan,
        "comparison": comparison,
        "runs": runs,
    }
    report["summary"] = st.summarize_hf_gpt2_finetune_sweep_report(report)
    _attach_scale_up_command_artifact(args, report)
    _write_json(args.out_dir / "sweep-report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_sweep(args)
    report_path = args.out_dir / "sweep-report.json"
    print(f"sweep_report {report_path}")
    scale_up_command = report.get("scale_up_command")
    if isinstance(scale_up_command, dict):
        print(
            "scale_up_command "
            f"{report.get('scale_up_command_path')} "
            f"status={scale_up_command.get('status')}"
        )
        command_display = scale_up_command.get("command_display")
        if command_display:
            print(f"scale_up_replay {command_display}")
    failed = int(report.get("failed_run_count") or 0)
    summary = report.get("summary")
    if (
        args.adapter_promotion_gate
        and not args.dry_run
        and isinstance(summary, Mapping)
        and summary.get("status") == "no_promotion_ready_runs"
    ):
        return 1
    return 1 if failed and args.fail_fast else 0


if __name__ == "__main__":
    raise SystemExit(main())
