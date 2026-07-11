#!/usr/bin/env python3
"""Local GPT-2 fine-tuning bridge with SpiralTorch runtime/Z-Space preflight."""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib
import inspect
import json
import math
import os
import random
import shlex
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import spiraltorch as st
from spiraltorch.hf_adapter import (
    HF_ADAPTER_INPUT_IDENTITY_SCHEMA,
    hf_adapter_input_identity_lines,
    hf_adapter_input_identity_report,
    hf_adapter_lineage_lines,
    hf_adapter_promotion_lines,
    write_hf_adapter_lineage,
    write_hf_adapter_promotion,
)
from spiraltorch.hf_execution_identity import (
    HF_FINETUNE_EXECUTION_IDENTITY_SCHEMA,
    hf_finetune_execution_identity_lines,
    hf_finetune_execution_identity_report,
)
from spiraltorch.hf_dataset_identity import (
    HF_DATASET_INPUT_IDENTITY_SCHEMA,
    HF_DATASET_MATERIALIZATION_IDENTITY_SCHEMA,
    HF_TOKENIZED_DATASET_IDENTITY_SCHEMA,
    hf_dataset_input_identity_lines,
    hf_dataset_input_identity_report,
    hf_dataset_materialization_identity_lines,
    hf_dataset_materialization_identity_report,
    hf_tokenized_dataset_identity_lines,
    hf_tokenized_dataset_identity_report,
)
from spiraltorch.hf_ft import (
    HF_GPT2_FT_DEFAULT_DEVICE_BACKENDS,
    hf_gpt2_finetune_checkpoint_resume_lines,
    hf_gpt2_finetune_checkpoint_resume_report,
    hf_gpt2_finetune_corpus_file_report,
    hf_gpt2_finetune_corpus_scan_report,
    hf_gpt2_finetune_dataset_fit_report,
    hf_gpt2_finetune_disk_headroom_plan,
    hf_gpt2_finetune_eval_report,
    hf_gpt2_finetune_generation_report,
    hf_gpt2_finetune_geometry_guard_horizon_report,
    hf_gpt2_finetune_inference_distortion_handoff_report,
    hf_gpt2_finetune_inference_distortion_handoff_lines,
    hf_gpt2_finetune_preflight_report,
    hf_gpt2_finetune_summary_lines,
    hf_gpt2_finetune_trainer_trace_callback,
    hf_gpt2_finetune_zspace_probe,
    hf_finetune_model_profile_lines,
    resolve_hf_finetune_model_profile,
    summarize_hf_gpt2_finetune_trainer_trace,
    write_hf_gpt2_finetune_run_card,
)
from spiraltorch.hf_generation import (
    build_zspace_repression_logits_processor,
    hf_causal_lm_artifact_probe_lines,
    hf_causal_lm_artifact_subprocess_probe_report,
    hf_generation_batch_size_compat,
)
from spiraltorch.hf_input_identity import (
    HF_FINETUNE_INPUT_IDENTITY_SCHEMA,
    hf_finetune_input_identity_lines,
    hf_finetune_input_identity_report,
)
from spiraltorch.hf_replay_identity import (
    HF_FINETUNE_REPLAY_IDENTITY_SCHEMA,
    hf_finetune_replay_identity_lines,
    hf_finetune_replay_identity_report,
)
from spiraltorch.hf_training_identity import (
    HF_FINETUNE_TRAINING_RECIPE_IDENTITY_SCHEMA,
    hf_finetune_training_recipe_identity_lines,
    hf_finetune_training_recipe_identity_report,
)
from spiraltorch.hf_peft import (
    HfCausalLmRuntimeIdentityError,
    hf_causal_lm_artifact_lines,
    hf_causal_lm_artifact_report,
    hf_finetune_adapter_config,
    load_hf_causal_lm_artifact,
    prepare_hf_finetune_model,
    summarize_hf_causal_lm_artifact,
)
from spiraltorch.hf_runtime_identity import hf_causal_lm_runtime_identity_lines


DEFAULT_MODEL = "gpt2"
DEFAULT_DATASET = "wikitext"
DEFAULT_DATASET_CONFIG = "wikitext-2-raw-v1"
HF_OFFLINE_ENV_VARS = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
)
HF_ADAPTER_ARTIFACT_PROBE_FILENAME = "spiraltorch-hf-artifact-probe.json"


def _valid_adapter_id(value: object | None) -> bool:
    if value is None:
        return True
    adapter_id = str(value).strip()
    digest = adapter_id.removeprefix("sha256:")
    return bool(
        adapter_id.startswith("sha256:")
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )


def _trainer_geometry_guard_active(args: argparse.Namespace) -> bool:
    return bool(
        args.trainer_min_desire_stability_guard is not None
        or args.trainer_max_psi_total_guard is not None
    )


def _trainer_geometry_guard_horizon_report(
    args: argparse.Namespace,
    *,
    initial_step: int = 0,
) -> dict[str, object]:
    return hf_gpt2_finetune_geometry_guard_horizon_report(
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        initial_step=initial_step,
        min_desire_stability_guard=(
            args.trainer_min_desire_stability_guard
        ),
        max_psi_total_guard=args.trainer_max_psi_total_guard,
        minimum_observations=args.trainer_geometry_guard_min_events,
        patience=args.trainer_geometry_guard_patience,
    )


def _checkpoint_resume_initial_step(
    report: Mapping[str, object] | None,
) -> int:
    if not isinstance(report, Mapping):
        return 0
    value = report.get("global_step")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _apply_checkpoint_resume_geometry_guard_horizon(
    args: argparse.Namespace,
    report: Mapping[str, object] | None,
) -> dict[str, object]:
    initial_step = _checkpoint_resume_initial_step(report)
    horizon = _trainer_geometry_guard_horizon_report(
        args,
        initial_step=initial_step,
    )
    args._trainer_geometry_guard_initial_step = initial_step
    args._trainer_geometry_guard_horizon = horizon
    return horizon


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
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
            "Model profile id from --model-configs. The profile supplies model, "
            "tokenizer, block-size, and generation defaults unless explicit CLI "
            "flags override them."
        ),
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--model-artifact-kind",
        choices=("auto", "full-model", "peft-adapter"),
        default="auto",
        help=(
            "Interpret --model-name as a full model or PEFT adapter. auto "
            "detects local adapter_config.json artifacts; remote adapters must "
            "use peft-adapter."
        ),
    )
    parser.add_argument(
        "--expected-parent-adapter-id",
        default=None,
        help=(
            "Fail before model loading unless a local PEFT continuation input "
            "matches this sha256 adapter identity."
        ),
    )
    parser.add_argument(
        "--expected-parent-lineage-depth",
        type=int,
        default=None,
        help="Expected lineage depth recorded by the continuation input manifest.",
    )
    parser.add_argument(
        "--expected-root-adapter-id",
        default=None,
        help="Expected sha256 root identity for the continuation input lineage.",
    )
    parser.add_argument(
        "--expected-training-input-id",
        default=None,
        help=(
            "Fail before model loading unless local model-config, corpus, "
            "distortion, and checkpoint inputs match this sha256 bundle identity."
        ),
    )
    parser.add_argument(
        "--expected-dataset-input-id",
        default=None,
        help=(
            "Fail before model loading unless the remote Hugging Face dataset "
            "repository commit and logical config/split selection match this "
            "sha256 identity."
        ),
    )
    parser.add_argument(
        "--expected-dataset-materialization-id",
        default=None,
        help=(
            "Fail after dataset selection and before tokenization unless every "
            "selected train/eval text row matches this sha256 identity."
        ),
    )
    parser.add_argument(
        "--expected-tokenized-dataset-id",
        default=None,
        help=(
            "Fail after tokenization and before model preparation unless every "
            "training/eval block and column matches this sha256 identity."
        ),
    )
    parser.add_argument(
        "--expected-runtime-input-id",
        default=None,
        help=(
            "Fail before model weight loading unless the resolved base-model "
            "commit/content and tokenizer semantics match this sha256 identity."
        ),
    )
    parser.add_argument(
        "--expected-execution-input-id",
        default=None,
        help=(
            "Fail before model weight loading unless the SpiralTorch build, "
            "HF package contents, Python/platform, device capabilities, and "
            "compute-affecting environment match this sha256 identity."
        ),
    )
    parser.add_argument(
        "--expected-training-recipe-id",
        default=None,
        help=(
            "Fail after effective TrainingArguments and model preparation resolve "
            "but before Trainer construction unless the optimization recipe "
            "matches this sha256 identity."
        ),
    )
    parser.add_argument(
        "--expected-finetune-replay-id",
        default=None,
        help=(
            "Fail before Trainer construction unless the complete model, data, "
            "runtime, execution, and effective recipe identity bundle matches "
            "this sha256 identity."
        ),
    )
    parser.add_argument(
        "--tokenizer-name",
        default=None,
        help=(
            "Optional tokenizer id/path. Defaults to the selected model profile's "
            "tokenizer_name, or --model-name when no profile tokenizer is active."
        ),
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument(
        "--dataset-revision",
        default=None,
        help=(
            "Optional Hugging Face dataset revision/branch/commit. Use this to "
            "pin a large remote corpus while keeping the run reproducible."
        ),
    )
    parser.add_argument(
        "--dataset-streaming",
        action="store_true",
        help=(
            "Load remote HF datasets with streaming=True, then materialize only "
            "--max-train-samples/--max-eval-samples rows before tokenization."
        ),
    )
    parser.add_argument(
        "--streaming-shuffle-buffer-size",
        type=int,
        default=0,
        help=(
            "Shuffle streamed train rows with this buffer before sampling. "
            "Use 0 to keep provider order."
        ),
    )
    parser.add_argument(
        "--streaming-validation-samples",
        type=int,
        default=0,
        help=(
            "When streaming and no eval split is loaded, take this many rows "
            "from the train stream as validation before train sampling."
        ),
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--text-column", default="text")
    parser.add_argument(
        "--train-file",
        action="append",
        type=Path,
        default=[],
        help=(
            "Local corpus file for training. May be repeated. When present, "
            "--dataset-name/--dataset-config are bypassed."
        ),
    )
    parser.add_argument(
        "--validation-file",
        action="append",
        type=Path,
        default=[],
        help="Local corpus file for validation/eval. May be repeated.",
    )
    parser.add_argument(
        "--dataset-format",
        choices=("text", "json", "csv"),
        default="text",
        help="datasets.load_dataset builder used for --train-file inputs.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.0,
        help=(
            "If using --train-file without --validation-file, split this "
            "fraction from train as validation."
        ),
    )
    parser.add_argument(
        "--corpus-scan",
        action="store_true",
        help=(
            "Stream local corpus files before loading HF datasets and record "
            "line/byte/sample stats in the run card."
        ),
    )
    parser.add_argument(
        "--corpus-scan-max-bytes-per-file",
        type=int,
        default=0,
        help=(
            "Bound --corpus-scan per file. The default 0 scans each local "
            "file fully."
        ),
    )
    parser.add_argument(
        "--corpus-scan-sample-lines",
        type=int,
        default=8,
        help="Number of nonempty preview lines to retain per scanned local file.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/hf-gpt2-ft"))
    parser.add_argument("--run-card", type=Path, default=None)
    parser.add_argument("--trainer-trace-jsonl", type=Path, default=None)
    parser.add_argument("--trainer-trace-run-id", default=None)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help=(
            "Resume Trainer.train() from an existing checkpoint directory, "
            "including optimizer/scheduler state when present."
        ),
    )
    parser.add_argument(
        "--adapter-promotion-gate",
        action="store_true",
        help=(
            "After LoRA training, require lineage integrity, changed adapter "
            "weights, successful before/after eval, and bounded eval regression."
        ),
    )
    parser.add_argument(
        "--adapter-promotion-max-eval-loss-regression",
        type=float,
        default=0.0,
        help=(
            "Largest allowed eval_after - eval_before value for adapter "
            "promotion. Negative values require improvement."
        ),
    )
    parser.add_argument(
        "--adapter-promotion-require-generation-change",
        action="store_true",
        help="Also require before/after generation hashes to differ for promotion.",
    )
    parser.add_argument(
        "--adapter-promotion-probe-prompt",
        default=None,
        help=(
            "Prompt for the post-save fresh artifact reload probe. Defaults to "
            "--generation-prompt or 'SpiralTorch is'."
        ),
    )
    parser.add_argument(
        "--adapter-promotion-probe-max-new-tokens",
        type=int,
        default=8,
        help="Bounded token budget for the post-save artifact generation probe.",
    )
    parser.add_argument(
        "--adapter-promotion-probe-device",
        default="auto",
        help="Device for the fresh artifact probe: auto, cpu, mps, cuda, or cuda:N.",
    )
    parser.add_argument(
        "--adapter-promotion-probe-timeout-seconds",
        type=float,
        default=900.0,
        help="Wall-clock timeout for the isolated artifact-probe worker.",
    )
    parser.add_argument("--no-trainer-trace", action="store_true")
    parser.add_argument("--trainer-telemetry", action="store_true")
    parser.add_argument("--trainer-telemetry-prefix", default="hf_ft")
    parser.add_argument("--trainer-desire-gain", type=float, default=1.0)
    parser.add_argument("--trainer-psi-gain", type=float, default=1.0)
    parser.add_argument(
        "--trainer-min-desire-stability-guard",
        type=float,
        default=None,
        help="Gracefully stop when guarded running-mean desire stability is lower.",
    )
    parser.add_argument(
        "--trainer-max-psi-total-guard",
        type=float,
        default=None,
        help="Gracefully stop when guarded maximum trainer psi is higher.",
    )
    parser.add_argument(
        "--trainer-geometry-guard-min-events",
        type=int,
        default=3,
        help="Minimum metric observations per active geometry guard.",
    )
    parser.add_argument(
        "--trainer-geometry-guard-patience",
        type=int,
        default=2,
        help="Consecutive breached telemetry frames required before stopping.",
    )
    parser.add_argument(
        "--no-trainer-loss-guard",
        action="store_true",
        help=(
            "Disable the trainer trace callback guard that stops on non-finite "
            "or explosively large losses."
        ),
    )
    parser.add_argument(
        "--trainer-loss-guard-threshold",
        type=float,
        default=1.0e6,
        help="Stop training when a logged train loss exceeds this absolute value.",
    )
    parser.add_argument(
        "--inference-distortion-sweep-report",
        type=Path,
        default=None,
        help=(
            "Attach a Z-Space inference-distortion sweep recommendation to the "
            "FT run card and trainer trace handoff."
        ),
    )
    parser.add_argument(
        "--inference-distortion-probe",
        type=Path,
        default=None,
        help=(
            "Attach one saved Z-Space inference-distortion probe directly; it "
            "is promoted to the same FT handoff shape as a sweep report."
        ),
    )
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--train", action="store_true", help="Actually run Trainer.train().")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Load model/tokenizer/dataset metadata but do not run Trainer.train().",
    )
    parser.add_argument(
        "--tokenize-only",
        action="store_true",
        help=(
            "Load and tokenize the selected dataset, verify exact Trainer-input "
            "identity, then stop before model preparation or Trainer construction."
        ),
    )
    parser.add_argument(
        "--training-recipe-only",
        action="store_true",
        help=(
            "Resolve the exact training recipe and stop before Trainer "
            "construction. The run-card launch command is rewritten to --train "
            "with the observed recipe identity pinned."
        ),
    )
    parser.add_argument("--max-train-samples", type=int, default=4096)
    parser.add_argument("--max-eval-samples", type=int, default=512)
    parser.add_argument(
        "--max-eval-blocks",
        type=int,
        default=0,
        help=(
            "Cap tokenized eval blocks after grouping. The default 0 keeps all "
            "blocks produced by --max-eval-samples."
        ),
    )
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument(
        "--eval-before-train",
        action="store_true",
        help=(
            "Run Trainer.evaluate() before Trainer.train() when an eval split is "
            "available, and record loss/perplexity in the run card."
        ),
    )
    parser.add_argument(
        "--no-eval-after-train",
        action="store_true",
        help="Skip the final Trainer.evaluate() run-card pass after training.",
    )
    parser.add_argument(
        "--eval-after-train-policy",
        choices=("always", "never", "skip-if-final-step-eval"),
        default="always",
        help=(
            "Control the post-train run-card evaluate pass. "
            "'skip-if-final-step-eval' avoids a duplicate full eval when "
            "max_steps lands exactly on an eval_steps boundary."
        ),
    )
    parser.add_argument(
        "--generation-prompt",
        default=None,
        help=(
            "Optional prompt to generate before and after training. Samples are "
            "written to the run card for qualitative FT comparison."
        ),
    )
    parser.add_argument("--generation-max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--generation-do-sample",
        action="store_true",
        help="Use sampling for generation samples instead of deterministic decode.",
    )
    parser.add_argument(
        "--generation-temperature",
        type=float,
        default=1.0,
        help="Sampling temperature used only with --generation-do-sample.",
    )
    parser.add_argument(
        "--generation-top-k",
        type=int,
        default=0,
        help="Optional top-k sampling cutoff used only with --generation-do-sample.",
    )
    parser.add_argument(
        "--generation-zspace-softmax",
        action="store_true",
        help=(
            "Apply a SpiralTorch ZSpaceSoftmax logits processor with optional "
            "repetition repression before generation token selection."
        ),
    )
    parser.add_argument(
        "--generation-from-inference-distortion",
        action="store_true",
        help=(
            "Populate generation Z-Space/repression settings from the attached "
            "inference-distortion probe or sweep handoff."
        ),
    )
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
    parser.add_argument(
        "--generation-zspace-report-limit",
        type=int,
        default=64,
        help=(
            "Maximum number of recent Z-Space generation-control calls to keep "
            "in each run-card generation_control.rows payload. Use 0 for "
            "aggregate-only telemetry."
        ),
    )
    parser.add_argument(
        "--generation-zspace-keep-non-top-k",
        action="store_true",
        help="Leave logits outside the Z-Space top-k set unchanged.",
    )
    parser.add_argument(
        "--generation-zspace-no-native",
        action="store_true",
        help="Use the pure-Python Z-Space softmax fallback even when native NN is available.",
    )
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--model-train-dtype",
        choices=("auto", "native", "float32"),
        default="auto",
        help=(
            "Model dtype policy before Trainer.train(). auto casts fp16/bf16 "
            "loaded checkpoints back to float32 for stable continued FT."
        ),
    )
    parser.add_argument(
        "--finetune-mode",
        choices=("full", "lora"),
        default="full",
        help="Train every parameter or attach a PEFT LoRA adapter before Trainer.train().",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Trade compute for lower activation memory and disable model use_cache.",
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-bias",
        choices=("none", "all", "lora_only"),
        default="none",
    )
    parser.add_argument(
        "--lora-target-module",
        action="append",
        default=[],
        help=(
            "PEFT target module suffix. Repeat to override model-family defaults; "
            "for example q_proj,k_proj,v_proj,o_proj."
        ),
    )
    parser.add_argument(
        "--lora-module-to-save",
        action="append",
        default=[],
        help="Additional full module to keep trainable and save beside the adapter.",
    )
    parser.add_argument(
        "--lora-use-rslora",
        action="store_true",
        help="Use rank-stabilized LoRA scaling when supported by PEFT.",
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help=(
            "Maximum number of Trainer checkpoints to keep. Lower this for "
            "long local FT runs on disk-constrained machines."
        ),
    )
    parser.add_argument(
        "--min-free-disk-gb",
        type=float,
        default=0.0,
        help=(
            "Abort before model/dataset loading unless the output filesystem "
            "has at least this many GiB free. The default 0 only records "
            "disk telemetry."
        ),
    )
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--eval-accumulation-steps", type=int, default=0)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument(
        "--dataloader-pin-memory",
        choices=("auto", "true", "false"),
        default="auto",
        help=(
            "TrainingArguments dataloader_pin_memory policy. auto enables it "
            "for CUDA and disables it for CPU/MPS to avoid unsupported MPS "
            "pin-memory warnings."
        ),
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--runtime-device-backend",
        action="append",
        default=[],
        help=(
            "SpiralTorch backend to report. Defaults to wgpu,cpu. This audits "
            "SpiralTorch runtime readiness; PyTorch still owns HF model kernels."
        ),
    )
    parser.add_argument(
        "--require-runtime-device-ready-backend",
        action="append",
        default=[],
        help="Fail preflight unless this SpiralTorch backend is runtime-ready.",
    )
    parser.add_argument(
        "--require-wgpu-ready",
        action="store_true",
        help="Shortcut for --require-runtime-device-ready-backend wgpu.",
    )
    parser.add_argument(
        "--no-require-hf-finetune",
        "--no-require-hf-gpt2-ft",
        dest="no_require_hf_gpt2_ft",
        action="store_true",
        help="Report missing full HF fine-tune imports without failing preflight.",
    )
    parser.add_argument("--zspace-probe", action="store_true")
    parser.add_argument("--zspace-probe-dim", type=int, default=64)
    parser.add_argument("--zspace-curvature", type=float, default=-0.04)
    parser.add_argument("--zspace-frequency", type=float, default=0.65)
    parser.add_argument("--zspace-strength", type=float, default=1.0)
    args = parser.parse_args(argv)
    _apply_model_profile_defaults(args, parser=parser, raw_argv=raw_argv)
    if args.model_artifact_kind == "peft-adapter" and args.finetune_mode != "lora":
        parser.error("--model-artifact-kind peft-adapter requires --finetune-mode lora")
    if not _valid_adapter_id(args.expected_parent_adapter_id):
        parser.error("--expected-parent-adapter-id must be sha256:<64 lowercase hex>")
    if not _valid_adapter_id(args.expected_root_adapter_id):
        parser.error("--expected-root-adapter-id must be sha256:<64 lowercase hex>")
    if not _valid_adapter_id(args.expected_training_input_id):
        parser.error("--expected-training-input-id must be sha256:<64 lowercase hex>")
    if not _valid_adapter_id(args.expected_dataset_input_id):
        parser.error("--expected-dataset-input-id must be sha256:<64 lowercase hex>")
    if not _valid_adapter_id(args.expected_dataset_materialization_id):
        parser.error(
            "--expected-dataset-materialization-id must be "
            "sha256:<64 lowercase hex>"
        )
    if not _valid_adapter_id(args.expected_tokenized_dataset_id):
        parser.error(
            "--expected-tokenized-dataset-id must be sha256:<64 lowercase hex>"
        )
    if not _valid_adapter_id(args.expected_runtime_input_id):
        parser.error("--expected-runtime-input-id must be sha256:<64 lowercase hex>")
    if not _valid_adapter_id(args.expected_execution_input_id):
        parser.error(
            "--expected-execution-input-id must be sha256:<64 lowercase hex>"
        )
    if not _valid_adapter_id(args.expected_training_recipe_id):
        parser.error(
            "--expected-training-recipe-id must be sha256:<64 lowercase hex>"
        )
    if not _valid_adapter_id(args.expected_finetune_replay_id):
        parser.error(
            "--expected-finetune-replay-id must be sha256:<64 lowercase hex>"
        )
    if (
        args.expected_parent_lineage_depth is not None
        and args.expected_parent_lineage_depth < 0
    ):
        parser.error("--expected-parent-lineage-depth must be non-negative")
    if (
        args.expected_parent_adapter_id is None
        and (
            args.expected_parent_lineage_depth is not None
            or args.expected_root_adapter_id is not None
        )
    ):
        parser.error(
            "--expected-parent-lineage-depth/--expected-root-adapter-id require "
            "--expected-parent-adapter-id"
        )
    if args.expected_parent_adapter_id is not None and args.finetune_mode != "lora":
        parser.error("--expected-parent-adapter-id requires --finetune-mode lora")
    if not math.isfinite(args.adapter_promotion_max_eval_loss_regression):
        parser.error("--adapter-promotion-max-eval-loss-regression must be finite")
    if args.adapter_promotion_gate:
        if not args.train or args.finetune_mode != "lora":
            parser.error("--adapter-promotion-gate requires --train --finetune-mode lora")
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
    if args.max_train_samples is not None and args.max_train_samples < 0:
        parser.error("--max-train-samples must be non-negative")
    if args.max_eval_samples is not None and args.max_eval_samples < 0:
        parser.error("--max-eval-samples must be non-negative")
    if args.max_eval_blocks is not None and args.max_eval_blocks < 0:
        parser.error("--max-eval-blocks must be non-negative")
    if args.streaming_shuffle_buffer_size < 0:
        parser.error("--streaming-shuffle-buffer-size must be non-negative")
    if args.streaming_validation_samples < 0:
        parser.error("--streaming-validation-samples must be non-negative")
    if args.dataset_streaming and args.train_file:
        parser.error("--dataset-streaming is only supported for remote HF datasets")
    if args.dataset_streaming and (
        args.max_train_samples is None or args.max_train_samples <= 0
    ):
        parser.error("--dataset-streaming requires a positive --max-train-samples")
    if args.block_size <= 0:
        parser.error("--block-size must be positive")
    if args.generation_max_new_tokens <= 0:
        parser.error("--generation-max-new-tokens must be positive")
    if args.save_total_limit <= 0:
        parser.error("--save-total-limit must be positive")
    if args.min_free_disk_gb < 0.0 or not math.isfinite(args.min_free_disk_gb):
        parser.error("--min-free-disk-gb must be finite and non-negative")
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
    if args.per_device_train_batch_size <= 0:
        parser.error("--per-device-train-batch-size must be positive")
    if args.per_device_eval_batch_size <= 0:
        parser.error("--per-device-eval-batch-size must be positive")
    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient-accumulation-steps must be positive")
    if args.eval_accumulation_steps < 0:
        parser.error("--eval-accumulation-steps must be non-negative")
    if args.dataloader_num_workers < 0:
        parser.error("--dataloader-num-workers must be non-negative")
    geometry_guard_active = _trainer_geometry_guard_active(args)
    if (args.trainer_telemetry or geometry_guard_active) and args.no_trainer_trace:
        parser.error("trainer telemetry and geometry guards require trainer tracing")
    if args.trainer_telemetry or geometry_guard_active:
        if not str(args.trainer_telemetry_prefix).strip():
            parser.error("--trainer-telemetry-prefix must be non-empty")
        if args.trainer_desire_gain < 0.0 or not math.isfinite(args.trainer_desire_gain):
            parser.error("--trainer-desire-gain must be finite and non-negative")
        if args.trainer_psi_gain < 0.0 or not math.isfinite(args.trainer_psi_gain):
            parser.error("--trainer-psi-gain must be finite and non-negative")
    for name, value in (
        (
            "--trainer-min-desire-stability-guard",
            args.trainer_min_desire_stability_guard,
        ),
        ("--trainer-max-psi-total-guard", args.trainer_max_psi_total_guard),
    ):
        if value is not None and (
            not math.isfinite(value) or not 0.0 <= value <= 1.0
        ):
            parser.error(f"{name} must be finite and in [0.0, 1.0]")
    if args.trainer_geometry_guard_min_events <= 0:
        parser.error("--trainer-geometry-guard-min-events must be positive")
    if args.trainer_geometry_guard_patience <= 0:
        parser.error("--trainer-geometry-guard-patience must be positive")
    geometry_guard_horizon = _trainer_geometry_guard_horizon_report(args)
    args._trainer_geometry_guard_horizon = geometry_guard_horizon
    if geometry_guard_active and geometry_guard_horizon.get("ready") is not True:
        parser.error(
            "trainer geometry guard is not observable: "
            f"{geometry_guard_horizon.get('status')} "
            "(set explicit --max-steps and reduce --logging-steps; "
            f"minimum max_steps={geometry_guard_horizon.get('minimum_max_steps')}, "
            "recommended logging_steps="
            f"{geometry_guard_horizon.get('recommended_logging_steps')})"
        )
    if args.trainer_loss_guard_threshold < 0.0 or not math.isfinite(
        args.trainer_loss_guard_threshold
    ):
        parser.error("--trainer-loss-guard-threshold must be finite and non-negative")
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
    run_modes = (
        args.train,
        args.metadata_only,
        args.tokenize_only,
        args.training_recipe_only,
    )
    if sum(bool(value) for value in run_modes) > 1:
        parser.error(
            "--train, --metadata-only, --tokenize-only, and "
            "--training-recipe-only are mutually exclusive"
        )
    if (
        args.expected_dataset_materialization_id is not None
        and not args.train
        and not args.metadata_only
        and not args.tokenize_only
        and not args.training_recipe_only
    ):
        parser.error(
            "--expected-dataset-materialization-id requires --train or "
            "--metadata-only or --tokenize-only"
        )
    if (
        args.expected_tokenized_dataset_id is not None
        and not args.train
        and not args.tokenize_only
        and not args.training_recipe_only
    ):
        parser.error(
            "--expected-tokenized-dataset-id requires --train, --tokenize-only, "
            "or --training-recipe-only"
        )
    if (
        args.expected_training_recipe_id is not None
        and not args.train
        and not args.training_recipe_only
    ):
        parser.error(
            "--expected-training-recipe-id requires --train or "
            "--training-recipe-only"
        )
    if (
        args.expected_finetune_replay_id is not None
        and not args.train
        and not args.training_recipe_only
    ):
        parser.error(
            "--expected-finetune-replay-id requires --train or "
            "--training-recipe-only"
        )
    if args.validation_file and not args.train_file:
        parser.error("--validation-file requires --train-file")
    if args.validation_file and args.validation_fraction > 0.0:
        parser.error(
            "--validation-file and --validation-fraction are mutually exclusive"
        )
    if args.validation_fraction < 0.0 or args.validation_fraction >= 1.0:
        parser.error("--validation-fraction must be in [0.0, 1.0)")
    if args.corpus_scan and not args.train_file:
        parser.error("--corpus-scan requires --train-file")
    if args.corpus_scan_max_bytes_per_file < 0:
        parser.error("--corpus-scan-max-bytes-per-file must be non-negative")
    if args.corpus_scan_sample_lines < 0:
        parser.error("--corpus-scan-sample-lines must be non-negative")
    for path in [*args.train_file, *args.validation_file]:
        if not path.is_file():
            parser.error(f"local corpus file does not exist: {path}")
    if (
        args.resume_from_checkpoint is not None
        and not args.resume_from_checkpoint.is_dir()
    ):
        parser.error(
            "--resume-from-checkpoint does not exist or is not a directory: "
            f"{args.resume_from_checkpoint}"
        )
    try:
        args._hf_finetune_adapter_config = _adapter_config_from_args(args)
    except ValueError as exc:
        parser.error(f"invalid fine-tune adapter configuration: {exc}")
    args._hf_finetune_launch_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        *raw_argv,
    ]
    args._hf_finetune_launch_command_source = "hf_gpt2_finetune_bridge"
    args._hf_finetune_launch_cwd = str(Path.cwd().resolve())
    return args


def _argv_has_option(raw_argv: Sequence[str], *names: str) -> bool:
    prefixes = tuple(f"{name}=" for name in names)
    return any(arg in names or arg.startswith(prefixes) for arg in raw_argv)


def _replace_or_append_command_option(
    command: Sequence[str],
    flag: str,
    value: object,
) -> list[str]:
    output: list[str] = []
    index = 0
    while index < len(command):
        item = str(command[index])
        if item == flag:
            index += 2
            continue
        if item.startswith(f"{flag}="):
            index += 1
            continue
        output.append(item)
        index += 1
    output.extend([flag, str(value)])
    return output


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
    args._hf_model_name_explicit = _argv_has_option(raw_argv, "--model-name")
    args._hf_tokenizer_name_explicit = _argv_has_option(
        raw_argv,
        "--tokenizer-name",
    )
    if args.model_configs is None and args.model_profile is None:
        return
    try:
        profile = resolve_hf_finetune_model_profile(
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
        args.tokenizer_name = None
    else:
        _set_profile_default(
            args,
            raw_argv,
            "tokenizer_name",
            profile.get("tokenizer_name"),
            "--tokenizer-name",
        )
    _set_profile_default(
        args,
        raw_argv,
        "block_size",
        _profile_value(profile, "training", "block_size") or profile.get("max_length"),
        "--block-size",
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
        ("learning_rate", "learning_rate", "--learning-rate"),
        ("num_train_epochs", "num_train_epochs", "--num-train-epochs"),
        ("max_steps", "max_steps", "--max-steps"),
        ("logging_steps", "logging_steps", "--logging-steps"),
        ("save_steps", "save_steps", "--save-steps"),
        ("eval_steps", "eval_steps", "--eval-steps"),
        (
            "eval_accumulation_steps",
            "eval_accumulation_steps",
            "--eval-accumulation-steps",
        ),
        ("save_total_limit", "save_total_limit", "--save-total-limit"),
    ):
        _set_profile_default(
            args,
            raw_argv,
            attr,
            _profile_value(profile, "training", key),
            flag,
        )
    training = _profile_section(profile, "training")
    _set_profile_default_if_present(
        args,
        raw_argv,
        training,
        "finetune_mode",
        "finetune_mode",
        "--finetune-mode",
    )
    if training.get("gradient_checkpointing") is True and not _argv_has_option(
        raw_argv,
        "--gradient-checkpointing",
    ):
        args.gradient_checkpointing = True
    adapter = _profile_section(profile, "adapter")
    for attr, key, flag in (
        ("lora_rank", "rank", "--lora-rank"),
        ("lora_alpha", "alpha", "--lora-alpha"),
        ("lora_dropout", "dropout", "--lora-dropout"),
        ("lora_bias", "bias", "--lora-bias"),
    ):
        _set_profile_default_if_present(args, raw_argv, adapter, key, attr, flag)
    if adapter.get("use_rslora") is True and not _argv_has_option(
        raw_argv,
        "--lora-use-rslora",
    ):
        args.lora_use_rslora = True
    if "target_modules" in adapter and not _argv_has_option(
        raw_argv,
        "--lora-target-module",
    ):
        args.lora_target_module = _profile_string_list(
            adapter.get("target_modules")
        )
    if "modules_to_save" in adapter and not _argv_has_option(
        raw_argv,
        "--lora-module-to-save",
    ):
        args.lora_module_to_save = _profile_string_list(
            adapter.get("modules_to_save")
        )
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
        ("model_train_dtype", "model_train_dtype", "--model-train-dtype"),
        ("dataloader_pin_memory", "dataloader_pin_memory", "--dataloader-pin-memory"),
        (
            "dataloader_num_workers",
            "dataloader_num_workers",
            "--dataloader-num-workers",
        ),
        ("min_free_disk_gb", "min_free_disk_gb", "--min-free-disk-gb"),
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


def _module(name: str) -> Any:
    return importlib.import_module(name)


def _select_rows(dataset: Any, limit: int | None) -> Any:
    if limit is None or limit <= 0:
        return dataset
    count = min(int(limit), len(dataset))
    return dataset.select(range(count))


def _limit_tokenized_eval_dataset(
    eval_dataset: Any | None,
    args: argparse.Namespace,
) -> tuple[Any | None, int | None, int | None]:
    if eval_dataset is None:
        return None, None, None
    before = len(eval_dataset)
    limited = _select_rows(eval_dataset, getattr(args, "max_eval_blocks", 0))
    return limited, before, len(limited)


def _load_dataset_split(
    datasets: Any,
    args: argparse.Namespace,
    split: str | None,
) -> Any | None:
    if not split:
        return None
    kwargs = {"split": split}
    if args.dataset_revision:
        kwargs["revision"] = args.dataset_revision
    if args.dataset_streaming and not _has_local_corpus(args):
        kwargs["streaming"] = True
    with _streaming_dataset_cpu_default_device(args):
        if args.dataset_config:
            return datasets.load_dataset(
                args.dataset_name,
                args.dataset_config,
                **kwargs,
            )
        return datasets.load_dataset(args.dataset_name, **kwargs)


def _has_local_corpus(args: argparse.Namespace) -> bool:
    return bool(args.train_file)


def _training_semantics_requested(args: argparse.Namespace) -> bool:
    return bool(args.train or args.training_recipe_only)


def _requires_remote_dataset_identity(args: argparse.Namespace) -> bool:
    return bool(
        not _has_local_corpus(args)
        and (
            _training_semantics_requested(args)
            or args.metadata_only
            or args.tokenize_only
            or args.expected_dataset_input_id is not None
        )
    )


def _requires_dataset_materialization_identity(args: argparse.Namespace) -> bool:
    return bool(
        _training_semantics_requested(args)
        or args.metadata_only
        or args.tokenize_only
        or args.expected_dataset_materialization_id is not None
    )


def _requires_tokenized_dataset_identity(args: argparse.Namespace) -> bool:
    return bool(
        _training_semantics_requested(args)
        or args.tokenize_only
        or args.expected_tokenized_dataset_id is not None
    )


def _requires_training_recipe_identity(args: argparse.Namespace) -> bool:
    return bool(
        _training_semantics_requested(args)
        or args.expected_training_recipe_id is not None
    )


def _requires_finetune_replay_identity(args: argparse.Namespace) -> bool:
    return bool(
        _training_semantics_requested(args)
        or args.expected_finetune_replay_id is not None
    )


def _uses_streaming_dataset(args: argparse.Namespace) -> bool:
    return bool(args.dataset_streaming and not _has_local_corpus(args))


@contextlib.contextmanager
def _streaming_dataset_cpu_default_device(args: argparse.Namespace):
    if not _uses_streaming_dataset(args):
        yield
        return
    try:
        torch = importlib.import_module("torch")
        get_default_device = getattr(torch, "get_default_device", None)
        set_default_device = getattr(torch, "set_default_device", None)
        if not callable(get_default_device) or not callable(set_default_device):
            yield
            return
        previous = get_default_device()
        set_default_device("cpu")
    except Exception:
        yield
        return
    try:
        yield
    finally:
        try:
            set_default_device(previous)
        except Exception:
            pass


def _local_data_files(args: argparse.Namespace) -> dict[str, list[str]]:
    data_files = {"train": [str(path) for path in args.train_file]}
    if args.validation_file:
        data_files["validation"] = [str(path) for path in args.validation_file]
    return data_files


def _maybe_shuffle_streaming_dataset(dataset: Any, args: argparse.Namespace) -> Any:
    buffer_size = int(getattr(args, "streaming_shuffle_buffer_size", 0) or 0)
    if buffer_size <= 0:
        return dataset
    shuffle = getattr(dataset, "shuffle", None)
    if not callable(shuffle):
        return dataset
    return shuffle(buffer_size=buffer_size, seed=int(args.seed))


def _streaming_take(dataset: Any, count: int) -> Any:
    take = getattr(dataset, "take", None)
    if callable(take):
        return take(int(count))
    rows = []
    for index, row in enumerate(dataset):
        if index >= int(count):
            break
        rows.append(row)
    return rows


def _streaming_skip(dataset: Any, count: int) -> Any:
    skip = getattr(dataset, "skip", None)
    if callable(skip):
        return skip(int(count))

    def skipped_rows():
        for index, row in enumerate(dataset):
            if index >= int(count):
                yield row

    return skipped_rows()


def _materialize_streaming_dataset(
    datasets: Any,
    dataset: Any | None,
    *,
    limit: int | None,
) -> Any | None:
    if dataset is None:
        return None
    if limit is None or int(limit) <= 0:
        return None
    rows = []
    for index, row in enumerate(dataset):
        if index >= int(limit):
            break
        rows.append(dict(row) if isinstance(row, Mapping) else row)
    return datasets.Dataset.from_list(rows)


def _corpus_file_report(args: argparse.Namespace) -> dict[str, object] | None:
    if not _has_local_corpus(args):
        return None
    return hf_gpt2_finetune_corpus_file_report(
        train_files=args.train_file,
        validation_files=args.validation_file,
        dataset_format=args.dataset_format,
        text_column=args.text_column,
    )


def _corpus_scan_report(args: argparse.Namespace) -> dict[str, object] | None:
    if not _has_local_corpus(args) or not args.corpus_scan:
        return None
    max_bytes = (
        None
        if args.corpus_scan_max_bytes_per_file <= 0
        else int(args.corpus_scan_max_bytes_per_file)
    )
    return hf_gpt2_finetune_corpus_scan_report(
        train_files=args.train_file,
        validation_files=args.validation_file,
        dataset_format=args.dataset_format,
        text_column=args.text_column,
        sample_line_limit=args.corpus_scan_sample_lines,
        max_bytes_per_file=max_bytes,
    )


def _attach_local_corpus_reports(
    card: dict[str, Any],
    args: argparse.Namespace,
    *,
    corpus_file_report: Mapping[str, object] | None,
    corpus_scan_report: Mapping[str, object] | None,
) -> dict[str, Any]:
    if not _has_local_corpus(args):
        return card
    card.update(
        {
            "dataset_source": "local_files",
            "dataset_format": args.dataset_format,
            "corpus_file_report": corpus_file_report,
            "corpus_scan_report": corpus_scan_report,
            "validation_fraction": args.validation_fraction,
        }
    )
    return card


def _load_raw_datasets(
    datasets: Any,
    args: argparse.Namespace,
) -> tuple[Any, Any | None, dict[str, object] | None]:
    if not _has_local_corpus(args):
        raw_train = _load_dataset_split(datasets, args, args.train_split)
        raw_eval = None
        try:
            raw_eval = _load_dataset_split(datasets, args, args.eval_split)
        except Exception:
            if (
                not _uses_streaming_dataset(args)
                or int(args.streaming_validation_samples) <= 0
            ):
                raise
        if _uses_streaming_dataset(args):
            with _streaming_dataset_cpu_default_device(args):
                raw_train = _maybe_shuffle_streaming_dataset(raw_train, args)
                if raw_eval is None and int(args.streaming_validation_samples) > 0:
                    raw_eval = _streaming_take(
                        raw_train,
                        int(args.streaming_validation_samples),
                    )
                    raw_train = _streaming_skip(
                        raw_train,
                        int(args.streaming_validation_samples),
                    )
                raw_train = _materialize_streaming_dataset(
                    datasets,
                    raw_train,
                    limit=args.max_train_samples,
                )
                raw_eval = _materialize_streaming_dataset(
                    datasets,
                    raw_eval,
                    limit=args.max_eval_samples,
                )
        return (
            raw_train,
            raw_eval,
            None,
        )

    corpus_report = _corpus_file_report(args)
    loaded = datasets.load_dataset(
        args.dataset_format,
        data_files=_local_data_files(args),
    )
    raw_train = loaded["train"]
    raw_eval = loaded["validation"] if "validation" in loaded else None
    if raw_eval is None and args.validation_fraction > 0.0:
        split = raw_train.train_test_split(
            test_size=float(args.validation_fraction),
            seed=int(args.seed),
        )
        raw_train = split["train"]
        raw_eval = split["test"]
    return raw_train, raw_eval, corpus_report


def _loader_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "local_files_only": not args.allow_remote,
        "trust_remote_code": args.trust_remote_code,
    }


def _hf_remote_access_report(args: argparse.Namespace) -> dict[str, object]:
    return {
        "allow_remote": bool(args.allow_remote),
        "offline_env": {name: os.environ.get(name) for name in HF_OFFLINE_ENV_VARS},
        "offline_env_overridden": bool(args.allow_remote)
        and any(os.environ.get(name) for name in HF_OFFLINE_ENV_VARS),
    }


def _trainer_telemetry_auto_reason(
    args: argparse.Namespace,
    inference_distortion_handoff: Mapping[str, object] | None,
) -> str | None:
    if bool(getattr(args, "trainer_telemetry", False)):
        return None
    if bool(getattr(args, "no_trainer_trace", False)):
        return None
    if (
        getattr(args, "trainer_min_desire_stability_guard", None) is not None
        or getattr(args, "trainer_max_psi_total_guard", None) is not None
    ):
        return "geometry_guard"
    if isinstance(inference_distortion_handoff, Mapping):
        return "inference_distortion_handoff"
    return None


def _trainer_telemetry_enabled(
    args: argparse.Namespace,
    inference_distortion_handoff: Mapping[str, object] | None,
) -> bool:
    if bool(getattr(args, "no_trainer_trace", False)):
        return False
    return bool(getattr(args, "trainer_telemetry", False)) or (
        _trainer_telemetry_auto_reason(args, inference_distortion_handoff) is not None
    )


@contextlib.contextmanager
def _hf_remote_access(args: argparse.Namespace):
    if not args.allow_remote:
        yield
        return

    old_env = {name: os.environ.get(name) for name in HF_OFFLINE_ENV_VARS}
    patched_attrs = []
    try:
        for name in HF_OFFLINE_ENV_VARS:
            os.environ[name] = "0"
        for module_name, attr_names in (
            ("huggingface_hub.constants", ("HF_HUB_OFFLINE",)),
            ("transformers.utils.hub", ("HF_HUB_OFFLINE",)),
            ("datasets.config", ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")),
        ):
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            for attr_name in attr_names:
                if hasattr(module, attr_name):
                    patched_attrs.append((module, attr_name, getattr(module, attr_name)))
                    setattr(module, attr_name, False)
        yield
    finally:
        for module, attr_name, old_value in reversed(patched_attrs):
            setattr(module, attr_name, old_value)
        for name, old_value in old_env.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value


def _tokenizer_vocab_size(tokenizer: Any) -> int | None:
    try:
        return int(len(tokenizer))
    except TypeError:
        vocab_size = getattr(tokenizer, "vocab_size", None)
        return None if vocab_size is None else int(vocab_size)


def _text_rows(dataset: Any, column: str, limit: int = 8) -> list[str]:
    rows = []
    for index in range(min(limit, len(dataset))):
        value = dataset[index].get(column)
        if isinstance(value, str) and value.strip():
            rows.append(value.strip())
    return rows


def _tokenize_dataset(dataset: Any, tokenizer: Any, args: argparse.Namespace) -> Any:
    if args.text_column not in getattr(dataset, "column_names", []):
        raise KeyError(
            f"dataset split does not contain text column {args.text_column!r}; "
            f"columns={getattr(dataset, 'column_names', [])!r}"
        )

    def tokenize(batch: Mapping[str, list[Any]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch[args.text_column])

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=list(getattr(dataset, "column_names", [])),
        desc="Tokenizing text",
    )

    def group_texts(
        examples: Mapping[str, list[list[int]]],
    ) -> dict[str, list[list[int]]]:
        concatenated = {
            key: sum((list(row) for row in rows), [])
            for key, rows in examples.items()
            if rows
        }
        if "input_ids" not in concatenated:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        total_length = (
            len(concatenated["input_ids"]) // args.block_size
        ) * args.block_size
        result = {
            key: [
                values[index : index + args.block_size]
                for index in range(0, total_length, args.block_size)
            ]
            for key, values in concatenated.items()
        }
        result["labels"] = list(result["input_ids"])
        return result

    return tokenized.map(group_texts, batched=True, desc="Grouping token blocks")


def _strategy_argument_name(training_arguments_cls: type) -> str:
    params = inspect.signature(training_arguments_cls.__init__).parameters
    if "eval_strategy" in params:
        return "eval_strategy"
    return "evaluation_strategy"


def _training_argument_parameter_names(training_arguments_cls: type) -> set[str] | None:
    params = inspect.signature(training_arguments_cls.__init__).parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return None
    return set(params)


def _filter_training_arguments_kwargs(
    training_arguments_cls: type,
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    names = _training_argument_parameter_names(training_arguments_cls)
    if names is None:
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in names}


def _dropped_training_arguments_kwargs(
    training_arguments_cls: type,
    kwargs: Mapping[str, Any],
) -> list[str]:
    names = _training_argument_parameter_names(training_arguments_cls)
    if names is None:
        return []
    return sorted(key for key in kwargs if key not in names)


def _raw_training_arguments_kwargs(
    args: argparse.Namespace,
    *,
    has_eval: bool,
    cls: type,
) -> dict[str, Any]:
    strategy_key = _strategy_argument_name(cls)
    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "overwrite_output_dir": True,
        "do_train": _training_semantics_requested(args),
        "do_eval": bool(has_eval),
        "num_train_epochs": float(args.num_train_epochs),
        "learning_rate": float(args.learning_rate),
        "per_device_train_batch_size": int(args.per_device_train_batch_size),
        "per_device_eval_batch_size": int(args.per_device_eval_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "logging_steps": int(args.logging_steps),
        "save_steps": int(args.save_steps),
        "save_total_limit": int(args.save_total_limit),
        "report_to": ["none"],
        "seed": int(args.seed),
        "save_strategy": "steps" if _training_semantics_requested(args) else "no",
        strategy_key: (
            "steps" if has_eval and _training_semantics_requested(args) else "no"
        ),
    }
    resolved_pin_memory = getattr(args, "_resolved_dataloader_pin_memory", None)
    if resolved_pin_memory is not None:
        kwargs["dataloader_pin_memory"] = bool(resolved_pin_memory)
    if getattr(args, "dataloader_num_workers", None) is not None:
        kwargs["dataloader_num_workers"] = int(args.dataloader_num_workers)
    if has_eval and getattr(args, "eval_accumulation_steps", 0) > 0:
        kwargs["eval_accumulation_steps"] = int(args.eval_accumulation_steps)
    if bool(getattr(args, "gradient_checkpointing", False)):
        kwargs["gradient_checkpointing"] = True
    if has_eval:
        kwargs["eval_steps"] = int(args.eval_steps)
    if args.max_steps is not None and args.max_steps > 0:
        kwargs["max_steps"] = int(args.max_steps)
    return kwargs


def _training_arguments_kwargs(
    args: argparse.Namespace,
    *,
    has_eval: bool,
    cls: type,
) -> dict[str, Any]:
    return _filter_training_arguments_kwargs(
        cls,
        _raw_training_arguments_kwargs(args, has_eval=has_eval, cls=cls),
    )


def _torch_cuda_available(torch: Any) -> bool:
    cuda = getattr(torch, "cuda", None)
    available = getattr(cuda, "is_available", None)
    if not callable(available):
        return False
    try:
        return bool(available())
    except Exception:
        return False


def _resolve_dataloader_pin_memory(torch: Any, args: argparse.Namespace) -> bool:
    mode = str(getattr(args, "dataloader_pin_memory", "auto"))
    if mode == "true":
        return True
    if mode == "false":
        return False
    return _torch_cuda_available(torch)


def _final_step_eval_likely(args: argparse.Namespace, *, has_eval: bool) -> bool:
    if not has_eval or not bool(getattr(args, "train", False)):
        return False
    max_steps = int(getattr(args, "max_steps", -1) or -1)
    eval_steps = int(getattr(args, "eval_steps", 0) or 0)
    if max_steps <= 0 or eval_steps <= 0:
        return False
    return max_steps % eval_steps == 0


def _eval_after_train_skipped_reason(
    args: argparse.Namespace,
    *,
    has_eval: bool,
) -> str | None:
    if bool(getattr(args, "no_eval_after_train", False)):
        return "no_eval_after_train_requested"
    policy = str(getattr(args, "eval_after_train_policy", "always"))
    if policy == "never":
        return "eval_after_train_policy_never"
    if policy == "skip-if-final-step-eval" and _final_step_eval_likely(
        args,
        has_eval=has_eval,
    ):
        return "final_step_eval_already_requested"
    return None


def _trainer_train_kwargs(args: argparse.Namespace) -> dict[str, object]:
    if args.resume_from_checkpoint is None:
        return {}
    return {"resume_from_checkpoint": str(args.resume_from_checkpoint)}


def _set_seed(torch: Any, transformers: Any, seed: int) -> None:
    random.seed(seed)
    set_seed = getattr(transformers, "set_seed", None)
    if callable(set_seed):
        set_seed(seed)
    manual_seed = getattr(torch, "manual_seed", None)
    if callable(manual_seed):
        manual_seed(seed)


def _model_device(model: Any) -> Any | None:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return None
    try:
        first_param = next(iter(parameters()))
    except StopIteration:
        return None
    return getattr(first_param, "device", None)


def _adapter_promotion_probe_prompt(args: argparse.Namespace) -> str:
    return str(
        args.adapter_promotion_probe_prompt
        or args.generation_prompt
        or "SpiralTorch is"
    )


def _save_finetune_tokenizer(
    tokenizer: Any,
    output_dir: Path,
) -> dict[str, object]:
    save_pretrained = getattr(tokenizer, "save_pretrained", None)
    if not callable(save_pretrained):
        return {
            "status": "unavailable",
            "output_dir": str(output_dir),
            "files": [],
            "error": "tokenizer does not expose save_pretrained",
        }
    try:
        written = save_pretrained(str(output_dir))
    except Exception as exc:
        return {
            "status": "error",
            "output_dir": str(output_dir),
            "files": [],
            "error": f"{exc.__class__.__name__}: {exc}",
        }
    files = (
        [str(value) for value in written]
        if isinstance(written, Sequence) and not isinstance(written, (str, bytes))
        else []
    )
    return {
        "status": "ready",
        "output_dir": str(output_dir),
        "files": files,
        "error": None,
    }


def _finetune_tokenizer_source(
    args: argparse.Namespace,
    tokenizer_save_report: Mapping[str, object],
) -> str | None:
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    )
    if tokenizer_save_report.get("status") == "ready" and any(
        (args.output_dir / name).is_file() for name in tokenizer_files
    ):
        return None
    return _tokenizer_name(args)


def _release_accelerator_cache(torch: Any) -> dict[str, object]:
    synchronized: list[str] = []
    released: list[str] = []
    errors: list[str] = []
    runtimes: list[tuple[str, Any]] = []
    for backend in ("cuda", "mps"):
        runtime = getattr(torch, backend, None)
        available = getattr(runtime, "is_available", None)
        if callable(available):
            try:
                if not bool(available()):
                    continue
            except Exception as exc:
                errors.append(
                    f"{backend}.is_available: {exc.__class__.__name__}: {exc}"
                )
                continue
        runtimes.append((backend, runtime))
    for backend, runtime in runtimes:
        synchronize = getattr(runtime, "synchronize", None)
        if not callable(synchronize):
            continue
        try:
            synchronize()
        except Exception as exc:
            errors.append(f"{backend}.synchronize: {exc.__class__.__name__}: {exc}")
        else:
            synchronized.append(backend)
    collected = gc.collect()
    for backend, runtime in runtimes:
        empty_cache = getattr(runtime, "empty_cache", None)
        if not callable(empty_cache):
            continue
        try:
            empty_cache()
        except Exception as exc:
            errors.append(f"{backend}: {exc.__class__.__name__}: {exc}")
        else:
            released.append(backend)
    return {
        "row_type": "hf_finetune_artifact_probe_runtime_release",
        "gc_collected": collected,
        "accelerators_synchronized": synchronized,
        "accelerator_caches_released": released,
        "errors": errors,
    }


def _run_adapter_artifact_probe(
    args: argparse.Namespace,
    *,
    tokenizer_source: str | None,
) -> dict[str, object]:
    probe_path = args.output_dir / HF_ADAPTER_ARTIFACT_PROBE_FILENAME
    prompt = _adapter_promotion_probe_prompt(args)
    try:
        report = hf_causal_lm_artifact_subprocess_probe_report(
            args.output_dir,
            tokenizer_name_or_path=tokenizer_source,
            artifact_kind="peft_adapter",
            prompt=prompt,
            max_new_tokens=args.adapter_promotion_probe_max_new_tokens,
            do_sample=False,
            device=args.adapter_promotion_probe_device,
            local_files_only=True,
            trust_remote_code=args.trust_remote_code,
            timeout_seconds=args.adapter_promotion_probe_timeout_seconds,
            report_path=probe_path,
        )
    except Exception as exc:
        report = {
            "row_type": "hf_causal_lm_artifact_probe",
            "status": "error",
            "artifact": {
                "artifact_kind": "peft_adapter",
                "artifact_source": str(args.output_dir.resolve()),
                "adapter_loaded": False,
            },
            "prompt": prompt,
            "device": args.adapter_promotion_probe_device,
            "new_token_count": None,
            "generated_text_changed": None,
            "local_files_only": True,
            "process_isolation": {
                "schema": "spiraltorch.hf_artifact_probe_process.v1",
                "status": "error",
                "fresh_process": False,
                "exit_code": None,
                "timed_out": False,
            },
            "error": f"{exc.__class__.__name__}: {exc}",
        }
    report["report_path"] = str(probe_path.resolve())
    report["tokenizer_source_requested"] = tokenizer_source
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    probe_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _model_first_parameter_dtype(model: Any) -> str | None:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return None
    try:
        first_param = next(iter(parameters()))
    except StopIteration:
        return None
    dtype = getattr(first_param, "dtype", None)
    return None if dtype is None else str(dtype)


def _prepare_model_train_dtype(model: Any, args: argparse.Namespace) -> dict[str, object]:
    mode = str(getattr(args, "model_train_dtype", "auto"))
    before = _model_first_parameter_dtype(model)
    should_cast = _training_semantics_requested(args) and (
        mode == "float32" or (mode == "auto" and before in {"torch.float16", "torch.bfloat16"})
    )
    cast_status = "not_requested"
    if should_cast:
        cast_model = getattr(model, "float", None)
        if callable(cast_model):
            cast_model()
            cast_status = "cast_float32"
        else:
            cast_status = "missing_float_method"
    return {
        "row_type": "hf_gpt2_finetune_model_dtype_report",
        "policy": mode,
        "train_requested": _training_semantics_requested(args),
        "dtype_before": before,
        "dtype_after": _model_first_parameter_dtype(model),
        "cast_status": cast_status,
    }


def _move_batch_to_device(batch: Mapping[str, Any], device: Any | None) -> dict[str, Any]:
    if device is None:
        return dict(batch)
    moved = {}
    for key, value in batch.items():
        move = getattr(value, "to", None)
        moved[key] = move(device) if callable(move) else value
    return moved


_prepare_special_tokens_batch_size_compat = hf_generation_batch_size_compat


def _last_dim(value: Any) -> int | None:
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[-1])
    try:
        return len(value)
    except TypeError:
        return None


def _first_sequence(value: Any) -> Any:
    try:
        return value[0]
    except (TypeError, KeyError, IndexError):
        return value


_GENERATION_PROCESSOR_ARG_MAP = {
    "top_k": "generation_zspace_top_k",
    "curvature": "generation_zspace_curvature",
    "temperature": "generation_zspace_temperature",
    "entropy_target": "generation_zspace_entropy_target",
    "entropy_tolerance": "generation_zspace_entropy_tolerance",
    "entropy_gain": "generation_zspace_entropy_gain",
    "min_temperature": "generation_zspace_min_temperature",
    "max_temperature": "generation_zspace_max_temperature",
    "repression_window": "generation_repression_window",
    "repression_strength": "generation_repression_strength",
    "last_token_repression": "generation_last_token_repression",
    "ngram_size": "generation_ngram_size",
    "ngram_window": "generation_ngram_window",
    "ngram_repression_strength": "generation_ngram_repression_strength",
    "ngram_decay": "generation_ngram_decay",
}


def _apply_inference_distortion_generation_defaults(
    args: argparse.Namespace,
    inference_distortion_handoff: Mapping[str, object] | None,
) -> Mapping[str, object] | None:
    if not getattr(args, "generation_from_inference_distortion", False):
        return None

    def _record(report: Mapping[str, object]) -> Mapping[str, object]:
        payload = dict(report)
        setattr(args, "_generation_from_inference_distortion_applied", payload)
        return payload

    if not isinstance(inference_distortion_handoff, Mapping):
        return _record({"status": "missing_handoff"})
    processor_kwargs = inference_distortion_handoff.get("recommended_processor_kwargs")
    if not isinstance(processor_kwargs, Mapping) or not processor_kwargs:
        return _record(
            {
                "status": "missing_processor_kwargs",
                "source_kind": inference_distortion_handoff.get("source_kind"),
                "recommended_probe": inference_distortion_handoff.get(
                    "recommended_probe"
                ),
            }
        )

    args.generation_zspace_softmax = True
    applied: dict[str, object] = {}
    for source_key, attr in _GENERATION_PROCESSOR_ARG_MAP.items():
        if source_key not in processor_kwargs:
            continue
        value = processor_kwargs[source_key]
        setattr(args, attr, value)
        applied[attr] = value
    if "mask_non_top_k" in processor_kwargs:
        keep_non_top_k = not bool(processor_kwargs["mask_non_top_k"])
        args.generation_zspace_keep_non_top_k = keep_non_top_k
        applied["generation_zspace_keep_non_top_k"] = keep_non_top_k
    if "use_native_zspace" in processor_kwargs:
        no_native = not bool(processor_kwargs["use_native_zspace"])
        args.generation_zspace_no_native = no_native
        applied["generation_zspace_no_native"] = no_native

    report = {
        "status": "ok",
        "source_kind": inference_distortion_handoff.get("source_kind"),
        "recommended_probe": inference_distortion_handoff.get("recommended_probe"),
        "applied_arg_count": len(applied),
        "applied_args": applied,
        "processor_kwargs": dict(processor_kwargs),
    }
    return _record(report)


def _generation_logits_processor(args: argparse.Namespace) -> Any | None:
    if not getattr(args, "generation_zspace_softmax", False):
        return None
    return build_zspace_repression_logits_processor(
        top_k=getattr(args, "generation_zspace_top_k", 64),
        curvature=getattr(args, "generation_zspace_curvature", -0.04),
        temperature=getattr(args, "generation_zspace_temperature", 1.0),
        entropy_target=getattr(args, "generation_zspace_entropy_target", None),
        entropy_tolerance=getattr(args, "generation_zspace_entropy_tolerance", 1.0e-4),
        entropy_gain=getattr(args, "generation_zspace_entropy_gain", 0.5),
        min_temperature=getattr(args, "generation_zspace_min_temperature", None),
        max_temperature=getattr(args, "generation_zspace_max_temperature", None),
        repression_window=getattr(args, "generation_repression_window", 32),
        repression_strength=getattr(args, "generation_repression_strength", 1.0),
        last_token_repression=getattr(args, "generation_last_token_repression", 0.5),
        ngram_size=getattr(args, "generation_ngram_size", 0),
        ngram_repression_strength=getattr(
            args,
            "generation_ngram_repression_strength",
            0.0,
        ),
        ngram_window=getattr(args, "generation_ngram_window", 0),
        ngram_decay=getattr(args, "generation_ngram_decay", 1.0),
        mask_non_top_k=not getattr(args, "generation_zspace_keep_non_top_k", False),
        use_native_zspace=not getattr(args, "generation_zspace_no_native", False),
    )


def _generation_logits_processor_list(processor: Any) -> Any:
    try:
        transformers = importlib.import_module("transformers")
        processor_list_type = getattr(transformers, "LogitsProcessorList", None)
        if processor_list_type is not None:
            return processor_list_type([processor])
    except Exception:
        pass
    return [processor]


def _generation_processor_report_for_args(
    processor: Any | None,
    args: argparse.Namespace,
) -> Mapping[str, object] | None:
    if processor is None:
        return None
    report = getattr(processor, "report", None)
    if not callable(report):
        return None
    limit = getattr(args, "generation_zspace_report_limit", 64)
    payload = report(limit=int(limit))
    return dict(payload) if isinstance(payload, Mapping) else None


def _next_token_from_logits(
    torch: Any,
    logits: Any,
    args: argparse.Namespace,
    input_ids: Any = None,
    logits_processor: Any | None = None,
) -> Any:
    last_logits = logits[:, -1, :]
    if logits_processor is not None:
        last_logits = logits_processor(input_ids, last_logits)
    if not args.generation_do_sample:
        return torch.argmax(last_logits, dim=-1, keepdim=True)
    scaled = last_logits / float(args.generation_temperature)
    if int(args.generation_top_k) > 0:
        vocab_size = int(getattr(scaled, "shape", [0, 0])[-1])
        k = min(int(args.generation_top_k), vocab_size)
        values, indices = torch.topk(scaled, k=k, dim=-1)
        probs = torch.softmax(values, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        return indices.gather(-1, sampled)
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _manual_forward_generate(
    torch: Any,
    tokenizer: Any,
    model: Any,
    batch: Mapping[str, Any],
    args: argparse.Namespace,
    logits_processor: Any | None = None,
) -> Any:
    input_ids = batch.get("input_ids")
    if input_ids is None:
        raise ValueError("tokenizer output did not include input_ids")
    generated = input_ids
    attention_mask = batch.get("attention_mask")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    for _ in range(int(args.generation_max_new_tokens)):
        call_kwargs = {"input_ids": generated}
        if attention_mask is not None:
            call_kwargs["attention_mask"] = attention_mask
        outputs = model(**call_kwargs)
        logits = getattr(outputs, "logits", None)
        if logits is None and isinstance(outputs, Mapping):
            logits = outputs.get("logits")
        if logits is None:
            raise ValueError("model forward output did not include logits")
        next_token = _next_token_from_logits(
            torch,
            logits,
            args,
            input_ids=generated,
            logits_processor=logits_processor,
        )
        generated = torch.cat([generated, next_token], dim=-1)
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token)],
                dim=-1,
            )
        if eos_token_id is not None:
            try:
                if bool((next_token == eos_token_id).all().item()):
                    break
            except (AttributeError, TypeError, ValueError):
                pass
    return generated


def _generation_sample(
    torch: Any,
    tokenizer: Any,
    model: Any,
    args: argparse.Namespace,
    *,
    stage: str,
) -> dict[str, object] | None:
    if not args.generation_prompt:
        return None
    prompt = str(args.generation_prompt)
    generation_method = "model.generate"
    logits_processor = None
    fallback_error = None
    try:
        logits_processor = _generation_logits_processor(args)
        if logits_processor is not None:
            generation_method = "model.generate+zspace_repression_softmax"
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
        input_tokens = _last_dim(input_ids)
        model_device = _model_device(model)
        batch = (
            _move_batch_to_device(encoded, model_device)
            if isinstance(encoded, Mapping)
            else encoded
        )
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": int(args.generation_max_new_tokens),
            "do_sample": bool(args.generation_do_sample),
        }
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is not None:
            generate_kwargs["pad_token_id"] = pad_token_id
        elif eos_token_id is not None:
            generate_kwargs["pad_token_id"] = eos_token_id
        if eos_token_id is not None:
            generate_kwargs["eos_token_id"] = eos_token_id
        if args.generation_do_sample:
            generate_kwargs["temperature"] = float(args.generation_temperature)
            if int(args.generation_top_k) > 0:
                generate_kwargs["top_k"] = int(args.generation_top_k)
        if logits_processor is not None:
            generate_kwargs["logits_processor"] = _generation_logits_processor_list(
                logits_processor
            )

        was_training = bool(getattr(model, "training", False))
        eval_model = getattr(model, "eval", None)
        train_model = getattr(model, "train", None)
        if callable(eval_model):
            eval_model()
        try:
            with torch.no_grad():
                try:
                    with _prepare_special_tokens_batch_size_compat(model):
                        if isinstance(batch, Mapping):
                            output_ids = model.generate(**batch, **generate_kwargs)
                        else:
                            output_ids = model.generate(batch, **generate_kwargs)
                except Exception as generate_exc:
                    if not isinstance(batch, Mapping):
                        raise
                    fallback_error = f"{generate_exc.__class__.__name__}: {generate_exc}"
                    generation_method = (
                        "manual_forward_fallback+zspace_repression_softmax"
                        if logits_processor is not None
                        else "manual_forward_fallback"
                    )
                    reset_report = getattr(logits_processor, "reset_report", None)
                    if callable(reset_report):
                        reset_report()
                    output_ids = _manual_forward_generate(
                        torch,
                        tokenizer,
                        model,
                        batch,
                        args,
                        logits_processor=logits_processor,
                    )
        finally:
            if was_training and callable(train_model):
                train_model()

        first_output = _first_sequence(output_ids)
        output_tokens = _last_dim(first_output)
        text = tokenizer.decode(first_output, skip_special_tokens=True)
        continuation = text[len(prompt) :] if text.startswith(prompt) else text
        return hf_gpt2_finetune_generation_report(
            stage=stage,
            prompt=prompt,
            generated_text=text,
            generated_continuation_text=continuation,
            input_token_count=input_tokens,
            output_token_count=output_tokens,
            max_new_tokens=args.generation_max_new_tokens,
            generation_method=generation_method,
            fallback_error=fallback_error,
            generation_control=_generation_processor_report_for_args(
                logits_processor,
                args,
            ),
        )
    except Exception as exc:
        return hf_gpt2_finetune_generation_report(
            stage=stage,
            prompt=prompt,
            max_new_tokens=args.generation_max_new_tokens,
            generation_method=generation_method,
            fallback_error=fallback_error,
            generation_control=_generation_processor_report_for_args(
                logits_processor,
                args,
            ),
            error=f"{exc.__class__.__name__}: {exc}",
        )


def _trainer_eval_report(
    trainer: Any,
    *,
    stage: str,
    eval_dataset_available: bool,
) -> dict[str, object]:
    if not eval_dataset_available:
        return hf_gpt2_finetune_eval_report(
            stage=stage,
            skipped_reason="eval_dataset_unavailable",
        )
    try:
        return hf_gpt2_finetune_eval_report(
            stage=stage,
            metrics=dict(trainer.evaluate() or {}),
        )
    except Exception as exc:
        return hf_gpt2_finetune_eval_report(
            stage=stage,
            error=f"{exc.__class__.__name__}: {exc}",
        )


def _resolved_model_profile(args: argparse.Namespace) -> Mapping[str, Any] | None:
    profile = getattr(args, "_hf_finetune_model_profile", None)
    return profile if isinstance(profile, Mapping) else None


def _model_family(args: argparse.Namespace) -> str | None:
    profile = _resolved_model_profile(args)
    if profile is None:
        return None
    family = profile.get("model_family")
    return None if family is None else str(family)


def _adapter_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    return hf_finetune_adapter_config(
        mode=getattr(args, "finetune_mode", "full"),
        model_family=_model_family(args),
        rank=getattr(args, "lora_rank", 16),
        alpha=getattr(args, "lora_alpha", 32.0),
        dropout=getattr(args, "lora_dropout", 0.05),
        bias=getattr(args, "lora_bias", "none"),
        target_modules=getattr(args, "lora_target_module", None),
        modules_to_save=getattr(args, "lora_module_to_save", None),
        use_rslora=bool(getattr(args, "lora_use_rslora", False)),
        gradient_checkpointing=bool(
            getattr(args, "gradient_checkpointing", False)
        ),
    )


def _tokenizer_name(args: argparse.Namespace) -> str:
    tokenizer_name = getattr(args, "tokenizer_name", None)
    if isinstance(tokenizer_name, str) and tokenizer_name.strip():
        return tokenizer_name
    if bool(getattr(args, "_hf_model_name_explicit", False)):
        return str(args.model_name)
    profile = _resolved_model_profile(args)
    if profile is not None:
        profile_tokenizer = profile.get("tokenizer_name")
        if isinstance(profile_tokenizer, str) and profile_tokenizer.strip():
            return profile_tokenizer
    return str(args.model_name)


def _tokenizer_override(args: argparse.Namespace) -> str | None:
    tokenizer_name = getattr(args, "tokenizer_name", None)
    if isinstance(tokenizer_name, str) and tokenizer_name.strip():
        profile = _resolved_model_profile(args)
        if (
            not bool(getattr(args, "_hf_tokenizer_name_explicit", False))
            and isinstance(profile, Mapping)
            and profile.get("artifact_kind") == "peft_adapter"
            and tokenizer_name == str(args.model_name)
        ):
            return None
        return tokenizer_name
    return None


def _model_artifact_report(args: argparse.Namespace) -> dict[str, object]:
    return hf_causal_lm_artifact_report(
        args.model_name,
        artifact_kind=args.model_artifact_kind,
        tokenizer_name_or_path=_tokenizer_override(args),
    )


def _adapter_input_identity_report(
    args: argparse.Namespace,
    artifact_report: Mapping[str, object],
    *,
    phase: str,
) -> dict[str, object] | None:
    expected_id = getattr(args, "expected_parent_adapter_id", None)
    expected_depth = getattr(args, "expected_parent_lineage_depth", None)
    expected_root_id = getattr(args, "expected_root_adapter_id", None)
    expected = expected_id is not None
    if artifact_report.get("artifact_kind") != "peft_adapter":
        if not expected:
            return None
        return {
            "row_type": "hf_adapter_input_identity",
            "schema": HF_ADAPTER_INPUT_IDENTITY_SCHEMA,
            "status": "blocked",
            "phase": phase,
            "expected_adapter_id": expected_id,
            "identity_verified": False,
            "error_count": 1,
            "errors": ["expected parent identity requires a PEFT adapter input"],
        }
    adapter_path = artifact_report.get("artifact_local_path")
    if adapter_path is None:
        if not expected:
            return None
        return {
            "row_type": "hf_adapter_input_identity",
            "schema": HF_ADAPTER_INPUT_IDENTITY_SCHEMA,
            "status": "blocked",
            "phase": phase,
            "expected_adapter_id": expected_id,
            "identity_verified": False,
            "error_count": 1,
            "errors": ["expected parent identity requires a local adapter input"],
        }
    try:
        return hf_adapter_input_identity_report(
            str(adapter_path),
            expected_adapter_id=expected_id,
            expected_lineage_depth=expected_depth,
            expected_root_adapter_id=expected_root_id,
            require_lineage=expected,
            phase=phase,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "row_type": "hf_adapter_input_identity",
            "schema": HF_ADAPTER_INPUT_IDENTITY_SCHEMA,
            "status": "blocked",
            "phase": phase,
            "expected_adapter_id": expected_id,
            "identity_verified": False,
            "error_count": 1,
            "errors": [f"{exc.__class__.__name__}: {exc}"],
        }


def _training_input_identity_report(
    args: argparse.Namespace,
    *,
    phase: str,
    expected_input_id: str | None = None,
) -> dict[str, object]:
    expected_id = (
        expected_input_id
        if expected_input_id is not None
        else getattr(args, "expected_training_input_id", None)
    )
    try:
        return hf_finetune_input_identity_report(
            model_configs=args.model_configs,
            train_files=args.train_file,
            validation_files=args.validation_file,
            inference_distortion_sweep_report=(
                args.inference_distortion_sweep_report
            ),
            inference_distortion_probe=args.inference_distortion_probe,
            resume_from_checkpoint=args.resume_from_checkpoint,
            expected_input_id=expected_id,
            phase=phase,
        )
    except (OSError, TypeError, ValueError) as exc:
        return {
            "row_type": "hf_finetune_input_identity",
            "schema": HF_FINETUNE_INPUT_IDENTITY_SCHEMA,
            "status": "blocked",
            "phase": phase,
            "expected_input_id": expected_id,
            "identity_verified": False,
            "error_count": 1,
            "errors": [f"{exc.__class__.__name__}: {exc}"],
        }


def _dataset_input_identity_report(
    args: argparse.Namespace,
    *,
    phase: str,
    expected_identity_id: str | None = None,
) -> dict[str, object]:
    expected_id = (
        expected_identity_id
        if expected_identity_id is not None
        else getattr(args, "expected_dataset_input_id", None)
    )
    try:
        return hf_dataset_input_identity_report(
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            requested_revision=args.dataset_revision,
            train_split=args.train_split,
            eval_split=args.eval_split,
            text_column=args.text_column,
            local_files=_has_local_corpus(args),
            expected_identity_id=expected_id,
            phase=phase,
        )
    except (OSError, TypeError, ValueError) as exc:
        return {
            "row_type": "hf_dataset_input_identity",
            "schema": HF_DATASET_INPUT_IDENTITY_SCHEMA,
            "status": "blocked" if expected_id is not None else "evidence_incomplete",
            "phase": phase,
            "expected_identity_id": expected_id,
            "observed_identity_id": None,
            "identity_verified": False,
            "error_count": 1,
            "errors": [f"{exc.__class__.__name__}: {exc}"],
        }


def _dataset_materialization_identity_report(
    args: argparse.Namespace,
    train_dataset: Any,
    eval_dataset: Any | None,
    *,
    expected_identity_id: str | None = None,
) -> dict[str, object]:
    expected_id = (
        expected_identity_id
        if expected_identity_id is not None
        else getattr(args, "expected_dataset_materialization_id", None)
    )
    try:
        return hf_dataset_materialization_identity_report(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            text_column=args.text_column,
            expected_identity_id=expected_id,
            phase="after_selection",
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "row_type": "hf_dataset_materialization_identity",
            "schema": HF_DATASET_MATERIALIZATION_IDENTITY_SCHEMA,
            "status": "blocked",
            "phase": "after_selection",
            "expected_identity_id": expected_id,
            "observed_identity_id": None,
            "identity_verified": False,
            "materialized_rows_verified": False,
            "error_count": 1,
            "errors": [f"{exc.__class__.__name__}: {exc}"],
        }


def _tokenized_dataset_identity_report(
    args: argparse.Namespace,
    train_dataset: Any,
    eval_dataset: Any | None,
    *,
    expected_identity_id: str | None = None,
) -> dict[str, object]:
    expected_id = (
        expected_identity_id
        if expected_identity_id is not None
        else getattr(args, "expected_tokenized_dataset_id", None)
    )
    try:
        return hf_tokenized_dataset_identity_report(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            expected_identity_id=expected_id,
            phase="after_tokenization",
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "row_type": "hf_tokenized_dataset_identity",
            "schema": HF_TOKENIZED_DATASET_IDENTITY_SCHEMA,
            "status": "blocked",
            "phase": "after_tokenization",
            "expected_identity_id": expected_id,
            "observed_identity_id": None,
            "identity_verified": False,
            "tokenized_rows_verified": False,
            "error_count": 1,
            "errors": [f"{exc.__class__.__name__}: {exc}"],
        }


def _training_recipe_trainer_contract(args: argparse.Namespace) -> dict[str, object]:
    trace_enabled = not bool(args.no_trainer_trace)
    return {
        "trainer": "transformers.Trainer",
        "data_collator": {
            "class": "transformers.DataCollatorForLanguageModeling",
            "mlm": False,
        },
        "trainer_train": {
            "resume_from_checkpoint": args.resume_from_checkpoint is not None,
        },
        "evaluation_control": {
            "eval_before_train": bool(args.eval_before_train),
            "eval_after_train_policy": args.eval_after_train_policy,
            "no_eval_after_train": bool(args.no_eval_after_train),
        },
        "trace_callback": {
            "enabled": trace_enabled,
            "stop_on_nonfinite_loss": (
                trace_enabled and not bool(args.no_trainer_loss_guard)
            ),
            "loss_guard_threshold": (
                args.trainer_loss_guard_threshold if trace_enabled else None
            ),
        },
    }


def _training_recipe_identity_report(
    args: argparse.Namespace,
    training_arguments: object,
    model_prepare_report: Mapping[str, object],
    model_dtype_report: Mapping[str, object] | None,
    *,
    expected_identity_id: str | None = None,
) -> dict[str, object]:
    expected_id = (
        expected_identity_id
        if expected_identity_id is not None
        else getattr(args, "expected_training_recipe_id", None)
    )
    try:
        return hf_finetune_training_recipe_identity_report(
            training_arguments,
            model_prepare_report=model_prepare_report,
            model_dtype_report=model_dtype_report,
            checkpoint_resume_report=getattr(
                args,
                "_hf_checkpoint_resume_report",
                None,
            ),
            trainer_contract=_training_recipe_trainer_contract(args),
            expected_identity_id=expected_id,
            phase="before_trainer_init",
        )
    except (TypeError, ValueError) as exc:
        return {
            "row_type": "hf_finetune_training_recipe_identity",
            "schema": HF_FINETUNE_TRAINING_RECIPE_IDENTITY_SCHEMA,
            "status": "blocked",
            "phase": "before_trainer_init",
            "expected_identity_id": expected_id,
            "observed_identity_id": None,
            "identity_verified": False,
            "path_independent": True,
            "error_count": 1,
            "errors": [f"{exc.__class__.__name__}: {exc}"],
        }


def _finetune_replay_identity_report(
    args: argparse.Namespace,
    *,
    adapter_input_identity: Mapping[str, object] | None,
    adapter_input_required: bool,
    training_input_identity: Mapping[str, object] | None,
    dataset_input_identity: Mapping[str, object] | None,
    dataset_materialization_identity: Mapping[str, object] | None,
    tokenized_dataset_identity: Mapping[str, object] | None,
    model_runtime_identity: Mapping[str, object] | None,
    execution_identity: Mapping[str, object] | None,
    training_recipe_identity: Mapping[str, object] | None,
    expected_identity_id: str | None = None,
) -> dict[str, object]:
    expected_id = (
        expected_identity_id
        if expected_identity_id is not None
        else getattr(args, "expected_finetune_replay_id", None)
    )
    try:
        return hf_finetune_replay_identity_report(
            adapter_input_identity=adapter_input_identity,
            adapter_input_required=adapter_input_required,
            training_input_identity=training_input_identity,
            dataset_input_identity=dataset_input_identity,
            dataset_materialization_identity=dataset_materialization_identity,
            tokenized_dataset_identity=tokenized_dataset_identity,
            model_runtime_identity=model_runtime_identity,
            execution_identity=execution_identity,
            training_recipe_identity=training_recipe_identity,
            expected_identity_id=expected_id,
            phase="before_trainer_init",
        )
    except (TypeError, ValueError) as exc:
        return {
            "row_type": "hf_finetune_replay_identity",
            "schema": HF_FINETUNE_REPLAY_IDENTITY_SCHEMA,
            "status": "blocked",
            "phase": "before_trainer_init",
            "expected_identity_id": expected_id,
            "observed_identity_id": None,
            "identity_verified": False,
            "path_independent": True,
            "error_count": 1,
            "errors": [f"{exc.__class__.__name__}: {exc}"],
        }


def _canonicalize_dataset_input_launch_command(
    args: argparse.Namespace,
    report: Mapping[str, object],
) -> None:
    if report.get("status") != "ready":
        return
    effective_revision = report.get("effective_revision")
    effective_dataset_name = report.get("effective_dataset_name")
    observed_id = report.get("observed_identity_id")
    if (
        effective_dataset_name is None
        or effective_revision is None
        or observed_id is None
    ):
        return
    args.dataset_name = str(effective_dataset_name)
    args.dataset_revision = str(effective_revision)
    command = list(getattr(args, "_hf_finetune_launch_command", []) or [])
    if not command:
        return
    command = _replace_or_append_command_option(
        command,
        "--dataset-name",
        effective_dataset_name,
    )
    command = _replace_or_append_command_option(
        command,
        "--dataset-revision",
        effective_revision,
    )
    if not _argv_has_option(command, "--expected-dataset-input-id"):
        command.extend(["--expected-dataset-input-id", str(observed_id)])
    args._hf_finetune_launch_command = command


def _canonicalize_dataset_materialization_launch_command(
    args: argparse.Namespace,
    report: Mapping[str, object],
) -> None:
    command = list(getattr(args, "_hf_finetune_launch_command", []) or [])
    observed_id = report.get("observed_identity_id")
    if (
        command
        and report.get("status") == "ready"
        and observed_id is not None
        and not _argv_has_option(command, "--expected-dataset-materialization-id")
    ):
        command.extend(
            ["--expected-dataset-materialization-id", str(observed_id)]
        )
        args._hf_finetune_launch_command = command


def _canonicalize_tokenized_dataset_launch_command(
    args: argparse.Namespace,
    report: Mapping[str, object],
) -> None:
    command = list(getattr(args, "_hf_finetune_launch_command", []) or [])
    observed_id = report.get("observed_identity_id")
    if (
        command
        and report.get("status") == "ready"
        and observed_id is not None
        and not _argv_has_option(command, "--expected-tokenized-dataset-id")
    ):
        command.extend(["--expected-tokenized-dataset-id", str(observed_id)])
    if getattr(args, "tokenize_only", False):
        command = [item for item in command if item != "--tokenize-only"]
        if "--train" not in command:
            command.append("--train")
    args._hf_finetune_launch_command = command


def _canonicalize_training_recipe_launch_command(
    args: argparse.Namespace,
    report: Mapping[str, object],
) -> None:
    command = list(getattr(args, "_hf_finetune_launch_command", []) or [])
    observed_id = report.get("observed_identity_id")
    if not command or report.get("status") != "ready" or observed_id is None:
        return
    if not _argv_has_option(command, "--expected-training-recipe-id"):
        command.extend(["--expected-training-recipe-id", str(observed_id)])
    if getattr(args, "training_recipe_only", False):
        command = [item for item in command if item != "--training-recipe-only"]
        if "--train" not in command:
            command.append("--train")
    args._hf_finetune_launch_command = command


def _canonicalize_finetune_replay_launch_command(
    args: argparse.Namespace,
    report: Mapping[str, object],
) -> None:
    command = list(getattr(args, "_hf_finetune_launch_command", []) or [])
    observed_id = report.get("observed_identity_id")
    if not command or report.get("status") != "ready" or observed_id is None:
        return
    if not _argv_has_option(command, "--expected-finetune-replay-id"):
        command.extend(["--expected-finetune-replay-id", str(observed_id)])
    args._hf_finetune_launch_command = command


def _canonicalize_training_input_launch_command(
    args: argparse.Namespace,
    report: Mapping[str, object],
) -> None:
    command = list(getattr(args, "_hf_finetune_launch_command", []) or [])
    if not command:
        return
    path_options: list[tuple[str, Sequence[object]]] = [
        (
            "--model-configs",
            () if args.model_configs is None else (args.model_configs,),
        ),
        ("--train-file", tuple(args.train_file)),
        ("--validation-file", tuple(args.validation_file)),
        (
            "--inference-distortion-sweep-report",
            ()
            if args.inference_distortion_sweep_report is None
            else (args.inference_distortion_sweep_report,),
        ),
        (
            "--inference-distortion-probe",
            ()
            if args.inference_distortion_probe is None
            else (args.inference_distortion_probe,),
        ),
        (
            "--resume-from-checkpoint",
            ()
            if args.resume_from_checkpoint is None
            else (args.resume_from_checkpoint,),
        ),
    ]
    for flag, values in path_options:
        if not values or _argv_has_option(command, flag):
            continue
        for value in values:
            command.extend([flag, str(value)])
    observed_id = report.get("observed_input_id")
    if (
        report.get("status") == "ready"
        and observed_id is not None
        and not _argv_has_option(command, "--expected-training-input-id")
    ):
        command.extend(["--expected-training-input-id", str(observed_id)])
    args._hf_finetune_launch_command = command


def _canonicalize_runtime_input_launch_command(
    args: argparse.Namespace,
    report: Mapping[str, object],
) -> None:
    command = list(getattr(args, "_hf_finetune_launch_command", []) or [])
    observed_id = report.get("observed_identity_id")
    if (
        command
        and report.get("status") == "ready"
        and observed_id is not None
        and not _argv_has_option(command, "--expected-runtime-input-id")
    ):
        command.extend(["--expected-runtime-input-id", str(observed_id)])
        args._hf_finetune_launch_command = command


def _execution_input_identity_report(
    args: argparse.Namespace,
    runtime_preflight: Mapping[str, object],
    torch: Any,
    *,
    phase: str,
    expected_identity_id: str | None = None,
) -> dict[str, object]:
    expected_id = (
        expected_identity_id
        if expected_identity_id is not None
        else getattr(args, "expected_execution_input_id", None)
    )
    try:
        return hf_finetune_execution_identity_report(
            runtime_preflight,
            torch_module=torch,
            expected_identity_id=expected_id,
            phase=phase,
        )
    except (OSError, TypeError, ValueError) as exc:
        return {
            "row_type": "hf_finetune_execution_identity",
            "schema": HF_FINETUNE_EXECUTION_IDENTITY_SCHEMA,
            "status": "blocked",
            "phase": phase,
            "expected_identity_id": expected_id,
            "observed_identity_id": None,
            "identity_verified": False,
            "error_count": 1,
            "errors": [f"{exc.__class__.__name__}: {exc}"],
        }


def _canonicalize_execution_input_launch_command(
    args: argparse.Namespace,
    report: Mapping[str, object],
) -> None:
    command = list(getattr(args, "_hf_finetune_launch_command", []) or [])
    observed_id = report.get("observed_identity_id")
    if (
        command
        and report.get("status") == "ready"
        and observed_id is not None
        and not _argv_has_option(command, "--expected-execution-input-id")
    ):
        command.extend(["--expected-execution-input-id", str(observed_id)])
        args._hf_finetune_launch_command = command


def _refresh_card_launch_command(
    card: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    command = list(getattr(args, "_hf_finetune_launch_command", []) or [])
    card["launch_command"] = command or None
    card["launch_command_display"] = shlex.join(command) if command else None
    nested_preflight = card.get("preflight")
    if isinstance(nested_preflight, dict) and nested_preflight is not card:
        nested_preflight["launch_command"] = command or None
        nested_preflight["launch_command_display"] = (
            shlex.join(command) if command else None
        )


def _finetune_start_report(
    args: argparse.Namespace,
    artifact_report: Mapping[str, object],
) -> dict[str, object]:
    resume_report = dict(
        getattr(args, "_hf_checkpoint_resume_report", {}) or {}
    )
    input_identity = dict(
        getattr(args, "_hf_adapter_input_identity_report", {}) or {}
    )
    training_input_identity = dict(
        getattr(args, "_hf_training_input_identity_report", {}) or {}
    )
    dataset_input_identity = dict(
        getattr(args, "_hf_dataset_input_identity_report", {}) or {}
    )
    dataset_materialization_identity = dict(
        getattr(args, "_hf_dataset_materialization_identity_report", {}) or {}
    )
    tokenized_dataset_identity = dict(
        getattr(args, "_hf_tokenized_dataset_identity_report", {}) or {}
    )
    runtime_input_identity = dict(
        getattr(args, "_hf_runtime_input_identity_report", {}) or {}
    )
    execution_input_identity = dict(
        getattr(args, "_hf_execution_input_identity_report", {}) or {}
    )
    training_recipe_identity = dict(
        getattr(args, "_hf_training_recipe_identity_report", {}) or {}
    )
    finetune_replay_identity = dict(
        getattr(args, "_hf_finetune_replay_identity_report", {}) or {}
    )
    adapter_preloaded = artifact_report.get("artifact_kind") == "peft_adapter"
    trainer_resume = args.resume_from_checkpoint is not None
    if adapter_preloaded and trainer_resume:
        mode = "adapter_trainer_checkpoint_resume"
    elif trainer_resume:
        mode = "trainer_checkpoint_resume"
    elif adapter_preloaded:
        mode = "adapter_warm_start"
    elif args.finetune_mode == "lora":
        mode = "new_adapter"
    else:
        mode = "full_model"
    return {
        "row_type": "hf_finetune_start_report",
        "status": "ready",
        "mode": mode,
        "model_artifact_kind": artifact_report.get("artifact_kind"),
        "model_source": artifact_report.get("artifact_source"),
        "base_model_name_or_path": artifact_report.get(
            "base_model_name_or_path"
        ),
        "adapter_preloaded": adapter_preloaded,
        "adapter_weights_source": (
            artifact_report.get("artifact_source") if adapter_preloaded else None
        ),
        "adapter_input_identity": input_identity or None,
        "expected_parent_adapter_id": getattr(
            args,
            "expected_parent_adapter_id",
            None,
        ),
        "parent_adapter_identity_verified": (
            None
            if not input_identity
            else input_identity.get("identity_verified") is True
        ),
        "training_input_identity": training_input_identity or None,
        "training_input_identity_verified": (
            None
            if not training_input_identity
            or training_input_identity.get("status") == "not_applicable"
            else training_input_identity.get("identity_verified") is True
        ),
        "dataset_input_identity": dataset_input_identity or None,
        "dataset_input_identity_verified": (
            None
            if not dataset_input_identity
            or dataset_input_identity.get("status") == "not_applicable"
            else dataset_input_identity.get("identity_verified") is True
        ),
        "dataset_materialization_identity": (
            dataset_materialization_identity or None
        ),
        "dataset_materialization_identity_verified": (
            None
            if not dataset_materialization_identity
            else dataset_materialization_identity.get("identity_verified") is True
        ),
        "tokenized_dataset_identity": tokenized_dataset_identity or None,
        "tokenized_dataset_identity_verified": (
            None
            if not tokenized_dataset_identity
            else tokenized_dataset_identity.get("identity_verified") is True
        ),
        "runtime_input_identity": runtime_input_identity or None,
        "runtime_input_identity_verified": (
            None
            if not runtime_input_identity
            else runtime_input_identity.get("identity_verified") is True
        ),
        "execution_input_identity": execution_input_identity or None,
        "execution_input_identity_verified": (
            None
            if not execution_input_identity
            else execution_input_identity.get("identity_verified") is True
        ),
        "training_recipe_identity": training_recipe_identity or None,
        "training_recipe_identity_verified": (
            None
            if not training_recipe_identity
            else training_recipe_identity.get("identity_verified") is True
        ),
        "finetune_replay_identity": finetune_replay_identity or None,
        "finetune_replay_identity_verified": (
            None
            if not finetune_replay_identity
            else finetune_replay_identity.get("identity_verified") is True
        ),
        "weights_only_warm_start": adapter_preloaded and not trainer_resume,
        "trainer_checkpoint_resume": trainer_resume,
        "trainer_checkpoint_source": (
            None
            if args.resume_from_checkpoint is None
            else str(args.resume_from_checkpoint)
        ),
        "optimizer_scheduler_state_requested": trainer_resume,
        "exact_state_available": resume_report.get("exact_state_available"),
        "scheduler_horizon_exhausted": resume_report.get(
            "scheduler_horizon_exhausted"
        ),
        "scheduler_extension_risk": resume_report.get(
            "scheduler_extension_risk"
        ),
        "resume_recommendation": resume_report.get("recommendation"),
    }


def _runtime_backends(args: argparse.Namespace) -> list[str]:
    return args.runtime_device_backend or list(HF_GPT2_FT_DEFAULT_DEVICE_BACKENDS)


def _required_ready_backends(args: argparse.Namespace) -> list[str]:
    backends = list(args.require_runtime_device_ready_backend or [])
    if args.require_wgpu_ready and "wgpu" not in backends:
        backends.append("wgpu")
    return backends


def _preflight_dataset_name(args: argparse.Namespace) -> str:
    return "local-files" if _has_local_corpus(args) else args.dataset_name


def _preflight_dataset_config(args: argparse.Namespace) -> str | None:
    return args.dataset_format if _has_local_corpus(args) else args.dataset_config


def _disk_usage_anchor(path: Path) -> Path:
    candidate = path if path.exists() else path.parent
    while not candidate.exists() and candidate.parent != candidate:
        candidate = candidate.parent
    return candidate


def _disk_report(path: Path, *, min_free_gb: float = 0.0) -> dict[str, Any]:
    anchor = _disk_usage_anchor(path)
    report: dict[str, Any] = {
        "row_type": "hf_gpt2_ft_disk_report",
        "path": str(path),
        "anchor": str(anchor),
        "min_free_gb": float(min_free_gb),
    }
    try:
        usage = shutil.disk_usage(anchor)
    except OSError as exc:
        report.update(
            {
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
                "meets_min_free": None,
            }
        )
        return report

    gib = 1024.0**3
    free_gb = usage.free / gib
    meets_min_free = min_free_gb <= 0.0 or free_gb >= min_free_gb
    report.update(
        {
            "status": "ok" if meets_min_free else "blocked",
            "total_bytes": int(usage.total),
            "used_bytes": int(usage.used),
            "free_bytes": int(usage.free),
            "total_gb": usage.total / gib,
            "used_gb": usage.used / gib,
            "free_gb": free_gb,
            "meets_min_free": bool(meets_min_free),
        }
    )
    return report


def _base_run_card(
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
    *,
    corpus_file_report: Mapping[str, object] | None,
    corpus_scan_report: Mapping[str, object] | None,
    inference_distortion_handoff: Mapping[str, object] | None,
    transformers: Any = None,
    torch: Any = None,
    datasets: Any = None,
) -> dict[str, Any]:
    artifact_report = dict(
        getattr(args, "_hf_causal_lm_artifact_report", {}) or {}
    )
    launch_command = list(
        getattr(args, "_hf_finetune_launch_command", []) or []
    )
    return {
        "row_type": "hf_gpt2_finetune_run_card",
        "launch_command": launch_command or None,
        "launch_command_display": shlex.join(launch_command)
        if launch_command
        else None,
        "launch_command_source": getattr(
            args,
            "_hf_finetune_launch_command_source",
            None,
        ),
        "launch_cwd": getattr(args, "_hf_finetune_launch_cwd", None),
        "preflight": preflight,
        "disk_report": dict(preflight.get("disk_report") or {}),
        "disk_headroom_plan": dict(preflight.get("disk_headroom_plan") or {}),
        "checkpoint_resume_report": preflight.get("checkpoint_resume_report"),
        "spiraltorch_version": getattr(st, "__version__", None),
        "transformers_version": getattr(transformers, "__version__", None),
        "torch_version": getattr(torch, "__version__", None),
        "datasets_version": getattr(datasets, "__version__", None),
        "model_name": args.model_name,
        "tokenizer_name": _tokenizer_name(args),
        "model_artifact_kind_requested": args.model_artifact_kind,
        "model_configs": None if args.model_configs is None else str(args.model_configs),
        "model_profile_id": (
            None
            if _resolved_model_profile(args) is None
            else _resolved_model_profile(args).get("profile_id")
        ),
        "model_profile": (
            None
            if _resolved_model_profile(args) is None
            else dict(_resolved_model_profile(args))
        ),
        "model_train_dtype": args.model_train_dtype,
        "model_dtype_report": None,
        "finetune_mode": args.finetune_mode,
        "adapter_config": dict(args._hf_finetune_adapter_config),
        "model_prepare_report": None,
        "model_artifact_kind": artifact_report.get("artifact_kind"),
        "model_artifact_report": summarize_hf_causal_lm_artifact(
            artifact_report
        ),
        "model_artifact_load_report": None,
        "finetune_start_report": _finetune_start_report(args, artifact_report),
        "adapter_input_identity": (
            dict(getattr(args, "_hf_adapter_input_identity_report", {}) or {})
            or None
        ),
        "adapter_input_identity_after_load": None,
        "training_input_identity": (
            dict(getattr(args, "_hf_training_input_identity_report", {}) or {})
            or None
        ),
        "training_input_identity_after_load": None,
        "dataset_input_identity": (
            dict(getattr(args, "_hf_dataset_input_identity_report", {}) or {})
            or None
        ),
        "dataset_input_identity_after_load": None,
        "dataset_input_identity_contract": {
            "status": (
                "not_applicable"
                if _has_local_corpus(args)
                else "enforced"
                if args.expected_dataset_input_id is not None
                else "adopted"
            ),
            "expected_identity_id": args.expected_dataset_input_id,
            "observed_identity_id": dict(
                getattr(args, "_hf_dataset_input_identity_report", {}) or {}
            ).get("observed_identity_id"),
            "effective_dataset_name": dict(
                getattr(args, "_hf_dataset_input_identity_report", {}) or {}
            ).get("effective_dataset_name"),
            "fail_fast": _requires_remote_dataset_identity(args),
        },
        "dataset_materialization_identity": None,
        "dataset_materialization_identity_contract": {
            "status": (
                "pending"
                if _requires_dataset_materialization_identity(args)
                else "not_requested"
            ),
            "expected_identity_id": args.expected_dataset_materialization_id,
            "observed_identity_id": None,
            "identity_verified": False,
            "fail_fast": _requires_dataset_materialization_identity(args),
            "verification_phase": "after_selection",
        },
        "tokenized_dataset_identity": None,
        "tokenized_dataset_identity_contract": {
            "status": (
                "pending"
                if _requires_tokenized_dataset_identity(args)
                else "not_requested"
            ),
            "expected_identity_id": args.expected_tokenized_dataset_id,
            "observed_identity_id": None,
            "identity_verified": False,
            "fail_fast": _requires_tokenized_dataset_identity(args),
            "verification_phase": "after_tokenization",
        },
        "training_recipe_identity": None,
        "training_recipe_identity_contract": {
            "status": (
                "pending"
                if _requires_training_recipe_identity(args)
                else "not_requested"
            ),
            "expected_identity_id": args.expected_training_recipe_id,
            "observed_identity_id": None,
            "identity_verified": False,
            "fail_fast": _requires_training_recipe_identity(args),
            "verification_phase": "before_trainer_init",
            "scale_up_inheritance": "recompute_after_intentional_recipe_change",
        },
        "finetune_replay_identity": None,
        "finetune_replay_identity_contract": {
            "status": (
                "pending"
                if _requires_finetune_replay_identity(args)
                else "not_requested"
            ),
            "expected_identity_id": args.expected_finetune_replay_id,
            "observed_identity_id": None,
            "identity_verified": False,
            "fail_fast": _requires_finetune_replay_identity(args),
            "verification_phase": "before_trainer_init",
            "scale_up_inheritance": "recompute_after_intentional_run_change",
        },
        "model_runtime_identity_pre_model": None,
        "model_runtime_identity_after_model": None,
        "model_runtime_identity_contract": {
            "status": (
                "enforced"
                if args.expected_runtime_input_id is not None
                else "observe"
            ),
            "expected_identity_id": args.expected_runtime_input_id,
            "fail_fast": args.expected_runtime_input_id is not None,
        },
        "finetune_execution_identity_pre_model": (
            dict(
                getattr(args, "_hf_execution_input_identity_report", {}) or {}
            )
            or None
        ),
        "finetune_execution_identity_after_model": None,
        "finetune_execution_identity_contract": {
            "status": (
                "enforced"
                if args.expected_execution_input_id is not None
                else "observe"
            ),
            "expected_identity_id": args.expected_execution_input_id,
            "fail_fast": True,
        },
        "adapter_lineage": None,
        "adapter_promotion": None,
        "tokenizer_save_report": None,
        "adapter_artifact_probe": None,
        "adapter_artifact_probe_runtime_release": None,
        "adapter_promotion_gate_requested": bool(args.adapter_promotion_gate),
        "adapter_promotion_require_artifact_probe": bool(
            args.adapter_promotion_gate
        ),
        "adapter_promotion_max_eval_loss_regression": (
            args.adapter_promotion_max_eval_loss_regression
        ),
        "adapter_promotion_require_generation_change": bool(
            args.adapter_promotion_require_generation_change
        ),
        "adapter_promotion_probe_prompt": _adapter_promotion_probe_prompt(args),
        "adapter_promotion_probe_max_new_tokens": (
            args.adapter_promotion_probe_max_new_tokens
        ),
        "adapter_promotion_probe_device": args.adapter_promotion_probe_device,
        "adapter_promotion_probe_timeout_seconds": (
            args.adapter_promotion_probe_timeout_seconds
        ),
        "adapter_saved": None,
        "resume_from_checkpoint": (
            None
            if args.resume_from_checkpoint is None
            else str(args.resume_from_checkpoint)
        ),
        "dataset_name": _preflight_dataset_name(args),
        "dataset_config": _preflight_dataset_config(args),
        "dataset_revision": args.dataset_revision,
        "dataset_streaming": bool(args.dataset_streaming),
        "streaming_shuffle_buffer_size": args.streaming_shuffle_buffer_size,
        "streaming_validation_samples": args.streaming_validation_samples,
        "dataset_source": "local_files" if _has_local_corpus(args) else "hf_dataset",
        "dataset_format": args.dataset_format if _has_local_corpus(args) else None,
        "corpus_file_report": corpus_file_report,
        "corpus_scan_report": corpus_scan_report,
        "validation_fraction": (
            args.validation_fraction if _has_local_corpus(args) else None
        ),
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "text_column": args.text_column,
        "block_size": args.block_size,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "max_eval_blocks": args.max_eval_blocks,
        "eval_before_train_requested": bool(args.eval_before_train),
        "eval_after_train_policy": args.eval_after_train_policy,
        "eval_after_train_requested": (
            not bool(args.no_eval_after_train)
            and args.eval_after_train_policy != "never"
        ),
        "eval_before_train": None,
        "eval_after_train": None,
        "dataloader_pin_memory": args.dataloader_pin_memory,
        "resolved_dataloader_pin_memory": getattr(
            args,
            "_resolved_dataloader_pin_memory",
            None,
        ),
        "dataloader_num_workers": args.dataloader_num_workers,
        "eval_accumulation_steps": args.eval_accumulation_steps,
        "generation_prompt": args.generation_prompt,
        "generation_max_new_tokens": args.generation_max_new_tokens,
        "generation_do_sample": bool(args.generation_do_sample),
        "generation_temperature": args.generation_temperature,
        "generation_top_k": args.generation_top_k,
        "generation_from_inference_distortion": bool(
            getattr(args, "generation_from_inference_distortion", False)
        ),
        "generation_from_inference_distortion_applied": getattr(
            args,
            "_generation_from_inference_distortion_applied",
            None,
        ),
        "generation_before_train": None,
        "generation_after_train": None,
        "zspace_probe": None,
        "train_requested": bool(args.train),
        "metadata_only": bool(args.metadata_only),
        "tokenize_only": bool(args.tokenize_only),
        "training_recipe_only": bool(args.training_recipe_only),
        "training_semantics_requested": _training_semantics_requested(args),
        "trainer_trace_jsonl": (
            None
            if args.metadata_only or args.tokenize_only or args.training_recipe_only
            else str(_trainer_trace_path(args))
        ),
        "trainer_telemetry_requested": bool(args.trainer_telemetry),
        "trainer_telemetry_enabled": _trainer_telemetry_enabled(
            args,
            inference_distortion_handoff,
        ),
        "trainer_telemetry_auto_reason": _trainer_telemetry_auto_reason(
            args,
            inference_distortion_handoff,
        ),
        "trainer_telemetry_prefix": args.trainer_telemetry_prefix,
        "trainer_desire_gain": args.trainer_desire_gain,
        "trainer_psi_gain": args.trainer_psi_gain,
        "trainer_min_desire_stability_guard": (
            args.trainer_min_desire_stability_guard
        ),
        "trainer_max_psi_total_guard": args.trainer_max_psi_total_guard,
        "trainer_geometry_guard_min_events": (
            args.trainer_geometry_guard_min_events
        ),
        "trainer_geometry_guard_patience": args.trainer_geometry_guard_patience,
        "trainer_geometry_guard_initial_step": getattr(
            args,
            "_trainer_geometry_guard_initial_step",
            0,
        ),
        "trainer_geometry_guard_active": _trainer_geometry_guard_active(args),
        "trainer_geometry_guard_horizon": dict(
            getattr(args, "_trainer_geometry_guard_horizon", {}) or {}
        ),
        "trainer_loss_guard_enabled": not bool(args.no_trainer_loss_guard),
        "trainer_loss_guard_threshold": args.trainer_loss_guard_threshold,
        "inference_distortion_sweep_report": (
            None
            if args.inference_distortion_sweep_report is None
            else str(args.inference_distortion_sweep_report)
        ),
        "inference_distortion_probe": (
            None
            if args.inference_distortion_probe is None
            else str(args.inference_distortion_probe)
        ),
        "inference_distortion_handoff": (
            dict(inference_distortion_handoff)
            if isinstance(inference_distortion_handoff, Mapping)
            else None
        ),
        "inference_distortion_handoff_lines": list(
            preflight.get("inference_distortion_handoff_lines") or []
        ),
        "load_status": "pending",
        "failure_stage": None,
        "failure_error": None,
        "dataset_fit_report": None,
    }


def _run_card_path(args: argparse.Namespace) -> Path:
    return args.run_card or (args.output_dir / st.HF_GPT2_FT_RUN_CARD_FILENAME)


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _adapter_output_collision(
    args: argparse.Namespace,
    model_artifact_report: Mapping[str, object],
) -> bool:
    artifact_local_path = model_artifact_report.get("artifact_local_path")
    if (
        model_artifact_report.get("artifact_kind") != "peft_adapter"
        or artifact_local_path is None
    ):
        return False
    artifact_path = Path(str(artifact_local_path))
    return _path_is_within(args.output_dir, artifact_path) or _path_is_within(
        artifact_path,
        args.output_dir,
    )


def _write_card(card: Mapping[str, Any], args: argparse.Namespace) -> None:
    path = _run_card_path(args)
    write_hf_gpt2_finetune_run_card(card, path)
    print(f"run_card {path}")


def _trainer_trace_path(args: argparse.Namespace) -> Path | None:
    if args.no_trainer_trace:
        return None
    return args.trainer_trace_jsonl or (
        args.output_dir / st.HF_GPT2_FT_TRAINER_TRACE_FILENAME
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    remote_access_report = _hf_remote_access_report(args)
    with _hf_remote_access(args):
        return _main_with_runtime_access(args, remote_access_report)


def _main_with_runtime_access(
    args: argparse.Namespace,
    remote_access_report: Mapping[str, object],
) -> int:
    model_artifact_report = _model_artifact_report(args)
    args._hf_causal_lm_artifact_report = model_artifact_report
    adapter_input_identity = _adapter_input_identity_report(
        args,
        model_artifact_report,
        phase="preflight",
    )
    args._hf_adapter_input_identity_report = adapter_input_identity
    training_input_identity = _training_input_identity_report(
        args,
        phase="preflight",
    )
    args._hf_training_input_identity_report = training_input_identity
    _canonicalize_training_input_launch_command(args, training_input_identity)
    dataset_input_identity = _dataset_input_identity_report(
        args,
        phase="preflight",
    )
    args._hf_dataset_input_identity_report = dataset_input_identity
    _canonicalize_dataset_input_launch_command(args, dataset_input_identity)
    checkpoint_resume_report = (
        None
        if args.resume_from_checkpoint is None
        else hf_gpt2_finetune_checkpoint_resume_report(
            args.resume_from_checkpoint,
            requested_max_steps=(args.max_steps if args.max_steps > 0 else None),
        )
    )
    args._hf_checkpoint_resume_report = checkpoint_resume_report
    _apply_checkpoint_resume_geometry_guard_horizon(
        args,
        checkpoint_resume_report,
    )
    corpus_file_report = _corpus_file_report(args)
    corpus_scan_report = _corpus_scan_report(args)
    inference_distortion_source = (
        args.inference_distortion_sweep_report or args.inference_distortion_probe
    )
    inference_distortion_handoff = (
        None
        if inference_distortion_source is None
        else hf_gpt2_finetune_inference_distortion_handoff_report(
            inference_distortion_source,
        )
    )
    generation_inference_distortion = _apply_inference_distortion_generation_defaults(
        args,
        inference_distortion_handoff,
    )
    preflight = hf_gpt2_finetune_preflight_report(
        model_name=args.model_name,
        dataset_name=_preflight_dataset_name(args),
        dataset_config=_preflight_dataset_config(args),
        dataset_revision=args.dataset_revision,
        dataset_streaming=args.dataset_streaming,
        streaming_shuffle_buffer_size=args.streaming_shuffle_buffer_size,
        streaming_validation_samples=args.streaming_validation_samples,
        train_split=args.train_split,
        eval_split=args.eval_split,
        text_column=args.text_column,
        runtime_device_backends=_runtime_backends(args),
        required_runtime_device_ready_backends=_required_ready_backends(args),
        require_hf_gpt2_ft=not args.no_require_hf_gpt2_ft,
        finetune_mode=args.finetune_mode,
    )
    preflight["hf_remote_access"] = dict(remote_access_report)
    preflight["model_configs"] = (
        None if args.model_configs is None else str(args.model_configs)
    )
    preflight["model_profile"] = (
        None
        if _resolved_model_profile(args) is None
        else dict(_resolved_model_profile(args))
    )
    preflight["tokenizer_name"] = _tokenizer_name(args)
    preflight["model_artifact_kind_requested"] = args.model_artifact_kind
    preflight["model_artifact_report"] = dict(model_artifact_report)
    preflight["model_artifact_summary"] = summarize_hf_causal_lm_artifact(
        model_artifact_report
    )
    preflight["adapter_input_identity"] = (
        None
        if adapter_input_identity is None
        else dict(adapter_input_identity)
    )
    preflight["training_input_identity"] = dict(training_input_identity)
    preflight["dataset_input_identity"] = dict(dataset_input_identity)
    preflight["dataset_input_identity_contract"] = {
        "status": (
            "not_applicable"
            if dataset_input_identity.get("status") == "not_applicable"
            else "enforced"
            if args.expected_dataset_input_id is not None
            else "adopted"
            if dataset_input_identity.get("status") == "ready"
            else "evidence_incomplete"
        ),
        "expected_identity_id": args.expected_dataset_input_id,
        "observed_identity_id": dataset_input_identity.get(
            "observed_identity_id"
        ),
        "identity_verified": dataset_input_identity.get("identity_verified"),
        "effective_revision": dataset_input_identity.get("effective_revision"),
        "effective_dataset_name": dataset_input_identity.get(
            "effective_dataset_name"
        ),
        "fail_fast": _requires_remote_dataset_identity(args),
        "verification_phase": "dataset_load",
    }
    preflight["dataset_materialization_identity"] = None
    preflight["dataset_materialization_identity_contract"] = {
        "status": (
            "pending"
            if _requires_dataset_materialization_identity(args)
            else "not_requested"
        ),
        "expected_identity_id": args.expected_dataset_materialization_id,
        "observed_identity_id": None,
        "identity_verified": False,
        "fail_fast": _requires_dataset_materialization_identity(args),
        "verification_phase": "after_selection",
    }
    preflight["tokenized_dataset_identity"] = None
    preflight["tokenized_dataset_identity_contract"] = {
        "status": (
            "pending"
            if _requires_tokenized_dataset_identity(args)
            else "not_requested"
        ),
        "expected_identity_id": args.expected_tokenized_dataset_id,
        "observed_identity_id": None,
        "identity_verified": False,
        "fail_fast": _requires_tokenized_dataset_identity(args),
        "verification_phase": "after_tokenization",
    }
    preflight["training_recipe_identity"] = None
    preflight["training_recipe_identity_contract"] = {
        "status": (
            "pending"
            if _requires_training_recipe_identity(args)
            else "not_requested"
        ),
        "expected_identity_id": args.expected_training_recipe_id,
        "observed_identity_id": None,
        "identity_verified": False,
        "fail_fast": _requires_training_recipe_identity(args),
        "verification_phase": "before_trainer_init",
        "scale_up_inheritance": "recompute_after_intentional_recipe_change",
    }
    preflight["finetune_replay_identity"] = None
    preflight["finetune_replay_identity_contract"] = {
        "status": (
            "pending"
            if _requires_finetune_replay_identity(args)
            else "not_requested"
        ),
        "expected_identity_id": args.expected_finetune_replay_id,
        "observed_identity_id": None,
        "identity_verified": False,
        "fail_fast": _requires_finetune_replay_identity(args),
        "verification_phase": "before_trainer_init",
        "scale_up_inheritance": "recompute_after_intentional_run_change",
    }
    preflight["model_runtime_identity_contract"] = {
        "status": (
            "enforced" if args.expected_runtime_input_id is not None else "observe"
        ),
        "expected_identity_id": args.expected_runtime_input_id,
        "fail_fast": args.expected_runtime_input_id is not None,
        "verification_phase": "model_load",
    }
    model_artifact_compatible = not (
        model_artifact_report.get("artifact_kind") == "peft_adapter"
        and args.finetune_mode != "lora"
    )
    adapter_output_collision = _adapter_output_collision(
        args,
        model_artifact_report,
    )
    preflight["model_artifact_compatible"] = model_artifact_compatible
    preflight["adapter_output_collision"] = adapter_output_collision
    preflight["finetune_start_report"] = _finetune_start_report(
        args,
        model_artifact_report,
    )
    preflight["checkpoint_resume_report"] = checkpoint_resume_report
    preflight["resume_from_checkpoint"] = (
        None
        if args.resume_from_checkpoint is None
        else str(args.resume_from_checkpoint)
    )
    preflight["inference_distortion_sweep_report"] = (
        None
        if args.inference_distortion_sweep_report is None
        else str(args.inference_distortion_sweep_report)
    )
    preflight["inference_distortion_probe"] = (
        None
        if args.inference_distortion_probe is None
        else str(args.inference_distortion_probe)
    )
    preflight["inference_distortion_handoff"] = (
        None
        if inference_distortion_handoff is None
        else dict(inference_distortion_handoff)
    )
    preflight["inference_distortion_handoff_lines"] = (
        []
        if inference_distortion_handoff is None
        else hf_gpt2_finetune_inference_distortion_handoff_lines(
            inference_distortion_handoff
        )
    )
    preflight["generation_from_inference_distortion_applied"] = (
        None
        if generation_inference_distortion is None
        else dict(generation_inference_distortion)
    )
    preflight["trainer_telemetry_requested"] = bool(args.trainer_telemetry)
    preflight["trainer_telemetry_enabled"] = _trainer_telemetry_enabled(
        args,
        inference_distortion_handoff,
    )
    preflight["trainer_telemetry_auto_reason"] = _trainer_telemetry_auto_reason(
        args,
        inference_distortion_handoff,
    )
    preflight["trainer_loss_guard_enabled"] = not bool(args.no_trainer_loss_guard)
    preflight["trainer_loss_guard_threshold"] = args.trainer_loss_guard_threshold
    preflight["model_train_dtype"] = args.model_train_dtype
    preflight["finetune_mode"] = args.finetune_mode
    preflight["adapter_config"] = dict(args._hf_finetune_adapter_config)
    launch_command = list(
        getattr(args, "_hf_finetune_launch_command", []) or []
    )
    preflight["launch_command"] = launch_command or None
    preflight["launch_command_display"] = (
        shlex.join(launch_command) if launch_command else None
    )
    preflight["launch_command_source"] = getattr(
        args,
        "_hf_finetune_launch_command_source",
        None,
    )
    preflight["launch_cwd"] = getattr(args, "_hf_finetune_launch_cwd", None)
    preflight["dataset_revision"] = args.dataset_revision
    preflight["dataset_streaming"] = bool(args.dataset_streaming)
    preflight["streaming_shuffle_buffer_size"] = args.streaming_shuffle_buffer_size
    preflight["streaming_validation_samples"] = args.streaming_validation_samples
    preflight["disk_report"] = _disk_report(
        args.output_dir,
        min_free_gb=args.min_free_disk_gb,
    )
    preflight["disk_headroom_plan"] = hf_gpt2_finetune_disk_headroom_plan(
        args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        save_total_limit=args.save_total_limit,
    )
    _attach_local_corpus_reports(
        preflight,
        args,
        corpus_file_report=corpus_file_report,
        corpus_scan_report=corpus_scan_report,
    )
    profile = _resolved_model_profile(args)
    if profile is not None:
        for line in hf_finetune_model_profile_lines(profile):
            print(line)
    for line in hf_causal_lm_artifact_lines(model_artifact_report):
        print(line)
    if adapter_input_identity is not None:
        for line in hf_adapter_input_identity_lines(adapter_input_identity):
            print(line)
    for line in hf_finetune_input_identity_lines(training_input_identity):
        print(line)
    for line in hf_dataset_input_identity_lines(dataset_input_identity):
        print(line)
    if checkpoint_resume_report is not None:
        for line in hf_gpt2_finetune_checkpoint_resume_lines(
            checkpoint_resume_report
        ):
            print(line)
    for line in hf_gpt2_finetune_summary_lines(preflight):
        print(line)
    if (
        model_artifact_report.get("status") != "ready"
        or not model_artifact_compatible
        or adapter_output_collision
        or (
            adapter_input_identity is not None
            and adapter_input_identity.get("status") != "ready"
        )
        or training_input_identity.get("status") == "blocked"
        or (
            _requires_remote_dataset_identity(args)
            and dataset_input_identity.get("status") != "ready"
        )
        or dataset_input_identity.get("status") == "blocked"
    ):
        if not model_artifact_compatible:
            preflight["model_artifact_error"] = (
                "PEFT adapter artifacts require --finetune-mode lora"
            )
        if adapter_output_collision:
            preflight["model_artifact_error"] = (
                "adapter input and output directories must not overlap"
            )
            artifact_path = Path(str(model_artifact_report["artifact_local_path"]))
            if _path_is_within(_run_card_path(args), artifact_path):
                print(
                    "run_card_skipped adapter input and run-card output overlap"
                )
                return 1
        if (
            adapter_input_identity is not None
            and adapter_input_identity.get("status") != "ready"
        ):
            preflight["model_artifact_error"] = (
                "adapter continuation input identity verification failed"
            )
        if training_input_identity.get("status") == "blocked":
            preflight["training_input_identity_error"] = (
                "fine-tune local input identity verification failed"
            )
        if dataset_input_identity.get("status") not in {"ready", "not_applicable"}:
            preflight["dataset_input_identity_error"] = (
                "remote dataset commit identity verification failed"
            )
        _write_card(preflight, args)
        return 1
    if (
        checkpoint_resume_report is not None
        and checkpoint_resume_report.get("status") == "invalid"
    ):
        _write_card(preflight, args)
        return 1
    geometry_guard_horizon = getattr(
        args,
        "_trainer_geometry_guard_horizon",
        {},
    )
    if (
        _trainer_geometry_guard_active(args)
        and geometry_guard_horizon.get("ready") is not True
    ):
        preflight.update(
            {
                "failure_stage": "trainer_geometry_guard_horizon",
                "failure_error": (
                    "trainer geometry guard cannot arm in the remaining "
                    "checkpoint-resume segment: "
                    f"{geometry_guard_horizon.get('status')}"
                ),
            }
        )
        _write_card(preflight, args)
        return 1
    if not preflight["runtime_import_preflight_passed"]:
        _write_card(preflight, args)
        return 1
    if preflight["disk_report"].get("status") != "ok":
        _write_card(preflight, args)
        return 1
    if (
        not _training_semantics_requested(args)
        and not args.metadata_only
        and not args.tokenize_only
    ):
        _write_card(preflight, args)
        return 0

    transformers = _module("transformers")
    torch = _module("torch")
    datasets = _module("datasets")
    args._resolved_dataloader_pin_memory = _resolve_dataloader_pin_memory(torch, args)
    _set_seed(torch, transformers, args.seed)

    execution_identity_pre_model = _execution_input_identity_report(
        args,
        preflight,
        torch,
        phase="pre_model_load",
    )
    args._hf_execution_input_identity_report = execution_identity_pre_model
    _canonicalize_execution_input_launch_command(
        args,
        execution_identity_pre_model,
    )
    preflight["finetune_execution_identity_pre_model"] = dict(
        execution_identity_pre_model
    )
    preflight["finetune_execution_identity_contract"] = {
        "status": (
            "enforced"
            if args.expected_execution_input_id is not None
            else "adopted"
            if execution_identity_pre_model.get("status") == "ready"
            else "evidence_incomplete"
        ),
        "expected_identity_id": args.expected_execution_input_id,
        "observed_identity_id": execution_identity_pre_model.get(
            "observed_identity_id"
        ),
        "identity_verified": execution_identity_pre_model.get(
            "identity_verified"
        ),
        "fail_fast": True,
        "verification_phase": "pre_model_load",
    }
    _refresh_card_launch_command(preflight, args)
    for line in hf_finetune_execution_identity_lines(
        execution_identity_pre_model
    ):
        print(line)
    if execution_identity_pre_model.get("status") != "ready":
        preflight["load_status"] = "error"
        preflight["failure_stage"] = "finetune_execution_identity_pre_model"
        preflight["failure_error"] = (
            "fine-tune execution environment identity verification failed"
        )
        _write_card(preflight, args)
        return 1

    card = _base_run_card(
        args,
        preflight,
        corpus_file_report=corpus_file_report,
        corpus_scan_report=corpus_scan_report,
        inference_distortion_handoff=inference_distortion_handoff,
        transformers=transformers,
        torch=torch,
        datasets=datasets,
    )
    try:
        model, tokenizer, _model_config, model_artifact_load_report = (
            load_hf_causal_lm_artifact(
                args.model_name,
                tokenizer_name_or_path=_tokenizer_override(args),
                artifact_kind=args.model_artifact_kind,
                is_trainable=(
                    model_artifact_report.get("artifact_kind") == "peft_adapter"
                ),
                transformers_module=transformers,
                loader_kwargs=_loader_kwargs(args),
                expected_runtime_identity_id=args.expected_runtime_input_id,
            )
        )
        card["model_artifact_load_report"] = summarize_hf_causal_lm_artifact(
            model_artifact_load_report
        )
        card["model_artifact_kind"] = model_artifact_load_report.get(
            "artifact_kind"
        )
        runtime_identity_pre_model = model_artifact_load_report.get(
            "runtime_identity_pre_model"
        )
        runtime_identity_after_model = model_artifact_load_report.get(
            "runtime_identity_after_model"
        )
        card["model_runtime_identity_pre_model"] = (
            dict(runtime_identity_pre_model)
            if isinstance(runtime_identity_pre_model, Mapping)
            else None
        )
        card["model_runtime_identity_after_model"] = (
            dict(runtime_identity_after_model)
            if isinstance(runtime_identity_after_model, Mapping)
            else None
        )
        strongest_runtime_identity = (
            card["model_runtime_identity_after_model"]
            or card["model_runtime_identity_pre_model"]
            or {}
        )
        args._hf_runtime_input_identity_report = strongest_runtime_identity
        _canonicalize_runtime_input_launch_command(
            args,
            strongest_runtime_identity,
        )
        _refresh_card_launch_command(card, args)
        card["model_runtime_identity_contract"] = {
            "status": (
                "enforced"
                if args.expected_runtime_input_id is not None
                else "adopted"
                if strongest_runtime_identity.get("status") == "ready"
                else "evidence_incomplete"
            ),
            "expected_identity_id": args.expected_runtime_input_id,
            "observed_identity_id": strongest_runtime_identity.get(
                "observed_identity_id"
            ),
            "identity_verified": strongest_runtime_identity.get(
                "identity_verified"
            ),
            "fail_fast": args.expected_runtime_input_id is not None,
        }
        card["finetune_start_report"] = _finetune_start_report(
            args,
            model_artifact_load_report,
        )
        adapter_input_identity_after_load = _adapter_input_identity_report(
            args,
            model_artifact_load_report,
            phase="after_load",
        )
        card["adapter_input_identity_after_load"] = (
            None
            if adapter_input_identity_after_load is None
            else dict(adapter_input_identity_after_load)
        )
        training_input_identity_after_load = _training_input_identity_report(
            args,
            phase="after_load",
            expected_input_id=(
                None
                if training_input_identity.get("status") == "not_applicable"
                else training_input_identity.get("observed_input_id")
            ),
        )
        card["training_input_identity_after_load"] = dict(
            training_input_identity_after_load
        )
        execution_identity_after_model = _execution_input_identity_report(
            args,
            preflight,
            torch,
            phase="after_model_load",
            expected_identity_id=(
                args.expected_execution_input_id
                or execution_identity_pre_model.get("observed_identity_id")
            ),
        )
        card["finetune_execution_identity_after_model"] = dict(
            execution_identity_after_model
        )
        args._hf_execution_input_identity_report = execution_identity_after_model
        _canonicalize_execution_input_launch_command(
            args,
            execution_identity_after_model,
        )
        _refresh_card_launch_command(card, args)
        card["finetune_execution_identity_contract"] = {
            "status": (
                "enforced"
                if args.expected_execution_input_id is not None
                else "adopted"
                if execution_identity_after_model.get("status") == "ready"
                else "evidence_incomplete"
            ),
            "expected_identity_id": args.expected_execution_input_id,
            "observed_identity_id": execution_identity_after_model.get(
                "observed_identity_id"
            ),
            "identity_verified": execution_identity_after_model.get(
                "identity_verified"
            ),
            "fail_fast": True,
        }
        card["finetune_start_report"] = _finetune_start_report(
            args,
            model_artifact_load_report,
        )
        card["tokenizer_name"] = model_artifact_load_report.get(
            "resolved_tokenizer_source"
        )
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = getattr(tokenizer, "eos_token", None)
        card["model_dtype_report"] = _prepare_model_train_dtype(model, args)
        if getattr(tokenizer, "pad_token_id", None) is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
    except HfCausalLmRuntimeIdentityError as exc:
        identity = dict(exc.report)
        phase = str(identity.get("phase") or "model_load")
        card.update(
            {
                "load_status": "error",
                "failure_stage": f"model_runtime_identity_{phase}",
                "failure_error": f"{exc.__class__.__name__}: {exc}",
                "model_runtime_identity_pre_model": (
                    identity if phase == "pre_model_load" else None
                ),
                "model_runtime_identity_after_model": (
                    identity if phase == "after_model_load" else None
                ),
            }
        )
        _write_card(card, args)
        return 1
    except Exception as exc:
        card.update(
            {
                "load_status": "error",
                "failure_stage": "model_tokenizer_load",
                "failure_error": f"{exc.__class__.__name__}: {exc}",
            }
        )
        _write_card(card, args)
        return 1
    if (
        adapter_input_identity_after_load is not None
        and adapter_input_identity_after_load.get("status") != "ready"
    ):
        card.update(
            {
                "load_status": "error",
                "failure_stage": "adapter_input_identity_after_load",
                "failure_error": (
                    "adapter continuation input identity changed during model load"
                ),
            }
        )
        _write_card(card, args)
        return 1
    if training_input_identity_after_load.get("status") == "blocked":
        card.update(
            {
                "load_status": "error",
                "failure_stage": "training_input_identity_after_load",
                "failure_error": (
                    "fine-tune local inputs changed during model loading"
                ),
            }
        )
        _write_card(card, args)
        return 1
    if execution_identity_after_model.get("status") != "ready":
        card.update(
            {
                "load_status": "error",
                "failure_stage": "finetune_execution_identity_after_model",
                "failure_error": (
                    "fine-tune execution environment changed during model loading"
                ),
            }
        )
        _write_card(card, args)
        return 1
    for runtime_identity in (
        card.get("model_runtime_identity_pre_model"),
        card.get("model_runtime_identity_after_model"),
    ):
        if isinstance(runtime_identity, Mapping):
            for line in hf_causal_lm_runtime_identity_lines(runtime_identity):
                print(line)
    for execution_identity in (
        card.get("finetune_execution_identity_pre_model"),
        card.get("finetune_execution_identity_after_model"),
    ):
        if isinstance(execution_identity, Mapping):
            for line in hf_finetune_execution_identity_lines(execution_identity):
                print(line)
    card["load_status"] = "ok"
    card["generation_before_train"] = _generation_sample(
        torch,
        tokenizer,
        model,
        args,
        stage="before_train",
    )

    try:
        raw_train, raw_eval, loaded_corpus_report = _load_raw_datasets(
            datasets,
            args,
        )
    except Exception as exc:
        card.update(
            {
                "load_status": "error",
                "failure_stage": "dataset_load",
                "failure_error": f"{exc.__class__.__name__}: {exc}",
            }
        )
        _write_card(card, args)
        return 1
    dataset_input_identity_after_load = _dataset_input_identity_report(
        args,
        phase="after_load",
        expected_identity_id=(
            None
            if dataset_input_identity.get("status") == "not_applicable"
            else dataset_input_identity.get("observed_identity_id")
        ),
    )
    card["dataset_input_identity_after_load"] = dict(
        dataset_input_identity_after_load
    )
    card["dataset_input_identity_contract"] = {
        "status": (
            "not_applicable"
            if dataset_input_identity_after_load.get("status") == "not_applicable"
            else "enforced"
            if args.expected_dataset_input_id is not None
            else "adopted"
            if dataset_input_identity_after_load.get("status") == "ready"
            else "evidence_incomplete"
        ),
        "expected_identity_id": (
            args.expected_dataset_input_id
            or dataset_input_identity.get("observed_identity_id")
        ),
        "observed_identity_id": dataset_input_identity_after_load.get(
            "observed_identity_id"
        ),
        "identity_verified": dataset_input_identity_after_load.get(
            "identity_verified"
        ),
        "effective_revision": dataset_input_identity_after_load.get(
            "effective_revision"
        ),
        "effective_dataset_name": dataset_input_identity_after_load.get(
            "effective_dataset_name"
        ),
        "fail_fast": _requires_remote_dataset_identity(args),
        "verification_phase": "after_load",
    }
    if dataset_input_identity_after_load.get("status") not in {
        "ready",
        "not_applicable",
    }:
        card.update(
            {
                "load_status": "error",
                "failure_stage": "dataset_input_identity_after_load",
                "failure_error": (
                    "remote dataset identity changed during dataset loading"
                ),
            }
        )
        _write_card(card, args)
        return 1
    for line in hf_dataset_input_identity_lines(dataset_input_identity_after_load):
        print(line)
    if loaded_corpus_report is not None:
        corpus_file_report = loaded_corpus_report
        card["corpus_file_report"] = corpus_file_report
    raw_train = _select_rows(raw_train, args.max_train_samples)
    raw_eval = (
        None if raw_eval is None else _select_rows(raw_eval, args.max_eval_samples)
    )
    dataset_materialization_identity = _dataset_materialization_identity_report(
        args,
        raw_train,
        raw_eval,
    )
    args._hf_dataset_materialization_identity_report = (
        dataset_materialization_identity
    )
    _canonicalize_dataset_materialization_launch_command(
        args,
        dataset_materialization_identity,
    )
    _refresh_card_launch_command(card, args)
    card["dataset_materialization_identity"] = dict(
        dataset_materialization_identity
    )
    card["dataset_materialization_identity_contract"] = {
        "status": (
            "blocked"
            if dataset_materialization_identity.get("status") != "ready"
            else "enforced"
            if args.expected_dataset_materialization_id is not None
            else "adopted"
        ),
        "expected_identity_id": (
            args.expected_dataset_materialization_id
            or dataset_materialization_identity.get("observed_identity_id")
        ),
        "observed_identity_id": dataset_materialization_identity.get(
            "observed_identity_id"
        ),
        "identity_verified": dataset_materialization_identity.get(
            "identity_verified"
        ),
        "total_rows": dataset_materialization_identity.get("total_rows"),
        "total_utf8_bytes": dataset_materialization_identity.get(
            "total_utf8_bytes"
        ),
        "fail_fast": True,
        "verification_phase": "after_selection",
    }
    nested_preflight = card.get("preflight")
    if isinstance(nested_preflight, dict):
        nested_preflight["dataset_materialization_identity"] = dict(
            dataset_materialization_identity
        )
        nested_preflight["dataset_materialization_identity_contract"] = dict(
            card["dataset_materialization_identity_contract"]
        )
    card["finetune_start_report"] = _finetune_start_report(
        args,
        model_artifact_load_report,
    )
    for line in hf_dataset_materialization_identity_lines(
        dataset_materialization_identity
    ):
        print(line)
    if dataset_materialization_identity.get("status") != "ready":
        card.update(
            {
                "load_status": "error",
                "failure_stage": "dataset_materialization_identity_after_selection",
                "failure_error": (
                    "selected dataset rows do not match the required "
                    "materialization identity"
                ),
            }
        )
        _write_card(card, args)
        return 1
    preview_texts = _text_rows(raw_train, args.text_column)

    zspace_probe = None
    preview_token_ids: list[int | float] = []
    if args.zspace_probe and preview_texts:
        encoded = tokenizer(preview_texts[0])
        preview_token_ids = list(encoded.get("input_ids", []))
        zspace_probe = hf_gpt2_finetune_zspace_probe(
            preview_token_ids,
            dim=args.zspace_probe_dim,
            vocab_size=_tokenizer_vocab_size(tokenizer),
            curvature=args.zspace_curvature,
            frequency=args.zspace_frequency,
            strength=args.zspace_strength,
        )

    card.update(
        {
            "train_split": args.train_split,
            "eval_split": args.eval_split,
            "text_column": args.text_column,
            "raw_train_rows": len(raw_train),
            "raw_eval_rows": None if raw_eval is None else len(raw_eval),
            "block_size": args.block_size,
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "zspace_probe": zspace_probe,
        }
    )
    if args.metadata_only:
        _write_card(card, args)
        return 0

    try:
        train_dataset = _tokenize_dataset(raw_train, tokenizer, args)
        eval_dataset = (
            None if raw_eval is None else _tokenize_dataset(raw_eval, tokenizer, args)
        )
        (
            eval_dataset,
            tokenized_eval_rows_before_limit,
            tokenized_eval_rows,
        ) = _limit_tokenized_eval_dataset(eval_dataset, args)
    except Exception as exc:
        card.update(
            {
                "failure_stage": "dataset_tokenize",
                "failure_error": f"{exc.__class__.__name__}: {exc}",
            }
        )
        _write_card(card, args)
        return 1
    tokenized_train_rows = len(train_dataset)
    dataset_fit_report = hf_gpt2_finetune_dataset_fit_report(
        raw_train_rows=len(raw_train),
        raw_eval_rows=None if raw_eval is None else len(raw_eval),
        tokenized_train_rows=tokenized_train_rows,
        tokenized_eval_rows=tokenized_eval_rows,
        block_size=args.block_size,
    )
    if dataset_fit_report["eval_dropped_empty"] is True:
        eval_dataset = None
    card.update(
        {
            "tokenized_train_rows": tokenized_train_rows,
            "tokenized_eval_rows_before_block_limit": (
                tokenized_eval_rows_before_limit
            ),
            "tokenized_eval_rows": tokenized_eval_rows,
            "dataset_fit_report": dataset_fit_report,
        }
    )
    tokenized_dataset_identity = _tokenized_dataset_identity_report(
        args,
        train_dataset,
        eval_dataset,
    )
    args._hf_tokenized_dataset_identity_report = tokenized_dataset_identity
    _canonicalize_tokenized_dataset_launch_command(
        args,
        tokenized_dataset_identity,
    )
    _refresh_card_launch_command(card, args)
    card["tokenized_dataset_identity"] = dict(tokenized_dataset_identity)
    card["tokenized_dataset_identity_contract"] = {
        "status": (
            "blocked"
            if tokenized_dataset_identity.get("status") != "ready"
            else "enforced"
            if args.expected_tokenized_dataset_id is not None
            else "adopted"
        ),
        "expected_identity_id": (
            args.expected_tokenized_dataset_id
            or tokenized_dataset_identity.get("observed_identity_id")
        ),
        "observed_identity_id": tokenized_dataset_identity.get(
            "observed_identity_id"
        ),
        "identity_verified": tokenized_dataset_identity.get("identity_verified"),
        "total_rows": tokenized_dataset_identity.get("total_rows"),
        "total_input_tokens": tokenized_dataset_identity.get("total_input_tokens"),
        "fail_fast": True,
        "verification_phase": "after_tokenization",
    }
    nested_preflight = card.get("preflight")
    if isinstance(nested_preflight, dict):
        nested_preflight["tokenized_dataset_identity"] = dict(
            tokenized_dataset_identity
        )
        nested_preflight["tokenized_dataset_identity_contract"] = dict(
            card["tokenized_dataset_identity_contract"]
        )
    card["finetune_start_report"] = _finetune_start_report(
        args,
        model_artifact_load_report,
    )
    for line in hf_tokenized_dataset_identity_lines(tokenized_dataset_identity):
        print(line)
    if tokenized_dataset_identity.get("status") != "ready":
        card.update(
            {
                "failure_stage": "tokenized_dataset_identity_after_tokenization",
                "failure_error": (
                    "tokenized training blocks do not match the required identity"
                ),
            }
        )
        _write_card(card, args)
        return 1
    if dataset_fit_report["train_ready"] is not True:
        card.update(
            {
                "failure_stage": "dataset_fit",
                "failure_error": (
                    "tokenized train split produced too few blocks: "
                    f"{dataset_fit_report['warnings']}"
                ),
            }
        )
        _write_card(card, args)
        return 1
    if args.tokenize_only:
        _write_card(card, args)
        return 0
    try:
        model, model_prepare_report = prepare_hf_finetune_model(
            model,
            mode=args.finetune_mode,
            model_family=_model_family(args),
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            bias=args.lora_bias,
            target_modules=args.lora_target_module,
            modules_to_save=args.lora_module_to_save,
            use_rslora=args.lora_use_rslora,
            gradient_checkpointing=args.gradient_checkpointing,
            preloaded_adapter=(
                model_artifact_report.get("artifact_kind") == "peft_adapter"
            ),
        )
        card["model_prepare_report"] = model_prepare_report
    except Exception as exc:
        card.update(
            {
                "failure_stage": "model_adapter_prepare",
                "failure_error": f"{exc.__class__.__name__}: {exc}",
            }
        )
        _write_card(card, args)
        return 1
    collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    training_args_cls = transformers.TrainingArguments
    raw_training_kwargs = _raw_training_arguments_kwargs(
        args,
        has_eval=eval_dataset is not None,
        cls=training_args_cls,
    )
    training_kwargs = _filter_training_arguments_kwargs(
        training_args_cls,
        raw_training_kwargs,
    )
    card["training_arguments_kwargs"] = sorted(training_kwargs)
    card["training_arguments_dropped_kwargs"] = _dropped_training_arguments_kwargs(
        training_args_cls,
        raw_training_kwargs,
    )
    try:
        training_args = training_args_cls(**training_kwargs)
    except Exception as exc:
        card.update(
            {
                "failure_stage": "training_arguments_init",
                "failure_error": f"{exc.__class__.__name__}: {exc}",
            }
        )
        _write_card(card, args)
        return 1
    model_dtype_report = card.get("model_dtype_report")
    training_recipe_identity = _training_recipe_identity_report(
        args,
        training_args,
        model_prepare_report,
        model_dtype_report if isinstance(model_dtype_report, Mapping) else None,
    )
    args._hf_training_recipe_identity_report = training_recipe_identity
    _canonicalize_training_recipe_launch_command(
        args,
        training_recipe_identity,
    )
    _refresh_card_launch_command(card, args)
    card["training_recipe_identity"] = dict(training_recipe_identity)
    card["training_recipe_identity_contract"] = {
        "status": (
            "blocked"
            if training_recipe_identity.get("status") != "ready"
            else "enforced"
            if args.expected_training_recipe_id is not None
            else "adopted"
        ),
        "expected_identity_id": (
            args.expected_training_recipe_id
            or training_recipe_identity.get("observed_identity_id")
        ),
        "observed_identity_id": training_recipe_identity.get(
            "observed_identity_id"
        ),
        "identity_verified": training_recipe_identity.get("identity_verified"),
        "path_independent": training_recipe_identity.get("path_independent"),
        "training_argument_count": training_recipe_identity.get(
            "training_argument_count"
        ),
        "fail_fast": True,
        "verification_phase": "before_trainer_init",
        "scale_up_inheritance": "recompute_after_intentional_recipe_change",
    }
    nested_preflight = card.get("preflight")
    if isinstance(nested_preflight, dict):
        nested_preflight["training_recipe_identity"] = dict(
            training_recipe_identity
        )
        nested_preflight["training_recipe_identity_contract"] = dict(
            card["training_recipe_identity_contract"]
        )
    card["finetune_start_report"] = _finetune_start_report(
        args,
        model_artifact_load_report,
    )
    for line in hf_finetune_training_recipe_identity_lines(
        training_recipe_identity
    ):
        print(line)
    if training_recipe_identity.get("status") != "ready":
        card.update(
            {
                "failure_stage": "training_recipe_identity_before_trainer_init",
                "failure_error": (
                    "effective training recipe does not match the required "
                    "identity"
                ),
            }
        )
        _write_card(card, args)
        return 1
    finetune_replay_identity = _finetune_replay_identity_report(
        args,
        adapter_input_identity=adapter_input_identity_after_load,
        adapter_input_required=(
            model_artifact_load_report.get("artifact_kind") == "peft_adapter"
        ),
        training_input_identity=training_input_identity_after_load,
        dataset_input_identity=dataset_input_identity_after_load,
        dataset_materialization_identity=dataset_materialization_identity,
        tokenized_dataset_identity=tokenized_dataset_identity,
        model_runtime_identity=strongest_runtime_identity,
        execution_identity=execution_identity_after_model,
        training_recipe_identity=training_recipe_identity,
    )
    args._hf_finetune_replay_identity_report = finetune_replay_identity
    _canonicalize_finetune_replay_launch_command(
        args,
        finetune_replay_identity,
    )
    _refresh_card_launch_command(card, args)
    card["finetune_replay_identity"] = dict(finetune_replay_identity)
    card["finetune_replay_identity_contract"] = {
        "status": (
            "blocked"
            if finetune_replay_identity.get("status") != "ready"
            else "enforced"
            if args.expected_finetune_replay_id is not None
            else "adopted"
        ),
        "expected_identity_id": (
            args.expected_finetune_replay_id
            or finetune_replay_identity.get("observed_identity_id")
        ),
        "observed_identity_id": finetune_replay_identity.get(
            "observed_identity_id"
        ),
        "identity_verified": finetune_replay_identity.get("identity_verified"),
        "path_independent": finetune_replay_identity.get("path_independent"),
        "component_count": finetune_replay_identity.get("component_count"),
        "applicable_component_count": finetune_replay_identity.get(
            "applicable_component_count"
        ),
        "ready_component_count": finetune_replay_identity.get(
            "ready_component_count"
        ),
        "fail_fast": True,
        "verification_phase": "before_trainer_init",
        "scale_up_inheritance": "recompute_after_intentional_run_change",
    }
    if isinstance(nested_preflight, dict):
        nested_preflight["finetune_replay_identity"] = dict(
            finetune_replay_identity
        )
        nested_preflight["finetune_replay_identity_contract"] = dict(
            card["finetune_replay_identity_contract"]
        )
    card["finetune_start_report"] = _finetune_start_report(
        args,
        model_artifact_load_report,
    )
    for line in hf_finetune_replay_identity_lines(finetune_replay_identity):
        print(line)
    if finetune_replay_identity.get("status") != "ready":
        card.update(
            {
                "failure_stage": "finetune_replay_identity_before_trainer_init",
                "failure_error": (
                    "complete fine-tune replay identity does not match the "
                    "required run contract"
                ),
            }
        )
        _write_card(card, args)
        return 1
    if args.training_recipe_only:
        _write_card(card, args)
        return 0
    trace_path = _trainer_trace_path(args)
    callbacks = []
    if trace_path is not None:
        callbacks.append(
            hf_gpt2_finetune_trainer_trace_callback(
                trace_path,
                run_id=args.trainer_trace_run_id or args.output_dir.name,
                zspace_probe_tokens=preview_token_ids if args.zspace_probe else None,
                zspace_probe_kwargs={
                    "dim": args.zspace_probe_dim,
                    "vocab_size": _tokenizer_vocab_size(tokenizer),
                    "curvature": args.zspace_curvature,
                    "frequency": args.zspace_frequency,
                    "strength": args.zspace_strength,
                },
                training_telemetry=_trainer_telemetry_enabled(
                    args,
                    card.get("inference_distortion_handoff"),
                ),
                telemetry_prefix=args.trainer_telemetry_prefix,
                desire_gain=args.trainer_desire_gain,
                psi_gain=args.trainer_psi_gain,
                stop_on_nonfinite_loss=not bool(args.no_trainer_loss_guard),
                loss_guard_threshold=args.trainer_loss_guard_threshold,
                min_desire_stability_guard=(
                    args.trainer_min_desire_stability_guard
                ),
                max_psi_total_guard=args.trainer_max_psi_total_guard,
                geometry_guard_min_events=(
                    args.trainer_geometry_guard_min_events
                ),
                geometry_guard_patience=args.trainer_geometry_guard_patience,
                geometry_guard_initial_step=getattr(
                    args,
                    "_trainer_geometry_guard_initial_step",
                    0,
                ),
                inference_distortion_handoff=card.get("inference_distortion_handoff"),
            )
        )
    try:
        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            callbacks=callbacks,
        )
        if args.eval_before_train:
            card["eval_before_train"] = _trainer_eval_report(
                trainer,
                stage="before_train",
                eval_dataset_available=eval_dataset is not None,
            )
        train_result = trainer.train(**_trainer_train_kwargs(args))
        trainer.save_model(str(args.output_dir))
    except Exception as exc:
        card.update(
            {
                "failure_stage": "trainer_train",
                "failure_error": f"{exc.__class__.__name__}: {exc}",
            }
        )
        _write_card(card, args)
        return 1
    eval_after_skip_reason = _eval_after_train_skipped_reason(
        args,
        has_eval=eval_dataset is not None,
    )
    if eval_after_skip_reason is None:
        card["eval_after_train"] = _trainer_eval_report(
            trainer,
            stage="after_train",
            eval_dataset_available=eval_dataset is not None,
        )
    else:
        card["eval_after_train"] = hf_gpt2_finetune_eval_report(
            stage="after_train",
            skipped_reason=eval_after_skip_reason,
        )
    card["generation_after_train"] = _generation_sample(
        torch,
        tokenizer,
        getattr(trainer, "model", model),
        args,
        stage="after_train",
    )
    trainer_trace_summary = (
        None
        if trace_path is None or not trace_path.exists()
        else summarize_hf_gpt2_finetune_trainer_trace(trace_path)
    )
    card.update(
        {
            "trainer_metrics": dict(getattr(train_result, "metrics", {}) or {}),
            "trainer_trace_jsonl": None if trace_path is None else str(trace_path),
            "trainer_trace_summary": trainer_trace_summary,
            "model_saved": True,
            "adapter_saved": (
                (args.output_dir / "adapter_config.json").is_file()
                if args.finetune_mode == "lora"
                else None
            ),
        }
    )
    tokenizer_save_report = _save_finetune_tokenizer(tokenizer, args.output_dir)
    card["tokenizer_save_report"] = tokenizer_save_report
    if card["adapter_saved"] is True:
        if args.adapter_promotion_gate:
            del train_result
            del trainer
            del model
            card["adapter_artifact_probe_runtime_release"] = (
                _release_accelerator_cache(torch)
            )
            artifact_probe = _run_adapter_artifact_probe(
                args,
                tokenizer_source=_finetune_tokenizer_source(
                    args,
                    tokenizer_save_report,
                ),
            )
            card["adapter_artifact_probe"] = artifact_probe
            for line in hf_causal_lm_artifact_probe_lines(artifact_probe):
                print(line)
        parent_adapter = (
            args.model_name
            if model_artifact_report.get("artifact_kind") == "peft_adapter"
            and model_artifact_report.get("artifact_is_local") is True
            else None
        )
        try:
            lineage = write_hf_adapter_lineage(
                args.output_dir,
                parent_adapter=parent_adapter,
                run_card=card,
                run_card_path=_run_card_path(args),
            )
            card["adapter_lineage"] = lineage
            for line in hf_adapter_lineage_lines(lineage):
                print(line)
            if args.adapter_promotion_gate:
                promotion = write_hf_adapter_promotion(
                    args.output_dir,
                    card,
                    parent_adapter=parent_adapter,
                    max_eval_loss_regression=(
                        args.adapter_promotion_max_eval_loss_regression
                    ),
                    require_generation_changed=(
                        args.adapter_promotion_require_generation_change
                    ),
                    require_artifact_probe=True,
                )
                card["adapter_promotion"] = {
                    key: promotion.get(key)
                    for key in (
                        "row_type",
                        "schema",
                        "status",
                        "promotion_ready",
                        "recommendation",
                        "candidate_adapter_id",
                        "parent_adapter_id",
                        "lineage_depth",
                        "eval_before_loss",
                        "eval_after_loss",
                        "eval_loss_regression",
                        "max_eval_loss_regression",
                        "require_artifact_probe",
                        "artifact_probe_status",
                        "artifact_probe_report_path",
                        "artifact_probe_device",
                        "artifact_probe_tokenizer_source_kind",
                        "artifact_probe_new_token_count",
                        "artifact_probe_candidate_matches",
                        "artifact_probe_local_files_only",
                        "artifact_probe_do_sample",
                        "artifact_probe_process_status",
                        "artifact_probe_process_fresh",
                        "artifact_probe_process_parent_pid",
                        "artifact_probe_process_pid",
                        "artifact_probe_process_exit_code",
                        "artifact_probe_process_timed_out",
                        "artifact_probe_process_duration_seconds",
                        "failed_checks",
                        "missing_checks",
                        "report_path",
                    )
                }
                for line in hf_adapter_promotion_lines(promotion):
                    print(line)
        except Exception as exc:
            card.update(
                {
                    "failure_stage": "adapter_lineage_promotion",
                    "failure_error": f"{exc.__class__.__name__}: {exc}",
                }
            )
            _write_card(card, args)
            return 1
    _write_card(card, args)
    if args.adapter_promotion_gate and not bool(
        (card.get("adapter_promotion") or {}).get("promotion_ready")
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
