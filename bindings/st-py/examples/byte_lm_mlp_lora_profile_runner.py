import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

from sparse_finetune_compare import attach_requested_wgpu_component_backend_summary


PROFILE_NAMES = [
    "strong_effect",
    "selective_gap",
    "selective_ratio",
]
PROMOTION_METRICS = [
    "target_loss_delta_mean",
    "target_retention_gap_mean",
    "target_retention_ratio",
]
DEFAULT_PROMOTION_METRIC = "target_retention_ratio"
DEFAULT_OUTPUT_DIR = Path("/tmp/spiraltorch-profile-runs")
DEFAULT_OUTPUT_PREFIX = "spiraltorch-byte-lm-mlp-lora-profile"


def parse_optional_int_value(raw, *, flag_name, parser=None):
    if raw is None:
        return None
    token = str(raw).strip()
    if token.lower() in {"none", "null", "off"}:
        return None
    try:
        value = int(token)
    except ValueError:
        if parser is not None:
            parser.error(f"{flag_name} must be a positive integer or none")
        raise
    if value <= 0:
        if parser is not None:
            parser.error(f"{flag_name} must be positive or none")
        raise ValueError(f"{flag_name} must be positive or none")
    return value


def ft_control_override_from_args(args, *, parser=None):
    override = {}
    if args.override_ft_epochs is not None:
        override["ft_epochs"] = args.override_ft_epochs
    if args.override_target_min_loss_delta is not None:
        override["target_min_loss_delta_policy"] = args.override_target_min_loss_delta
    if args.override_patience is not None:
        override["early_stopping_patience"] = parse_optional_int_value(
            args.override_patience,
            flag_name="--override-patience",
            parser=parser,
        )
    if args.override_min_delta is not None:
        override["early_stopping_min_delta"] = args.override_min_delta
    if args.override_lr_decay_patience is not None:
        override["lr_decay_patience"] = parse_optional_int_value(
            args.override_lr_decay_patience,
            flag_name="--override-lr-decay-patience",
            parser=parser,
        )
    if args.override_lr_decay_factor is not None:
        override["lr_decay_factor"] = args.override_lr_decay_factor
    if args.override_lr_decay_min_delta is not None:
        override["lr_decay_min_delta"] = args.override_lr_decay_min_delta
    return override or None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Materialize byte_lm_mlp_lora_sweep.py commands from "
            "checkpoint_source_profile JSONL rows."
        )
    )
    parser.add_argument(
        "--profile-jsonl",
        type=Path,
        default=None,
        help=(
            "checkpoint_source_profile JSONL emitted by "
            "byte_lm_mlp_lora_source_compare.py. Required unless "
            "--current-run-summary-jsonl is used for standalone compare."
        ),
    )
    parser.add_argument(
        "--source-path",
        dest="source_paths",
        action="append",
        default=[],
        help=(
            "Map a selected source label to a local checkpoint path, as "
            "source_label=/path/to/checkpoint. May be repeated."
        ),
    )
    parser.add_argument(
        "--profile",
        dest="profiles",
        action="append",
        choices=PROFILE_NAMES,
        default=[],
        help="Run only this profile lane. May be repeated; defaults to all rows.",
    )
    parser.add_argument(
        "--promotion-input-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint_source_profile_promotion JSONL used to "
            "materialize only promoted profile lanes from --profile-jsonl."
        ),
    )
    parser.add_argument(
        "--promotion-selection-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional JSONL path for a checkpoint_source_profile_promotion_selection "
            "summary when --promotion-input-jsonl is used."
        ),
    )
    parser.add_argument(
        "--include-non-ready-promotions",
        action="store_true",
        help="When --promotion-input-jsonl is used, include rows that are not promotion-ready.",
    )
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        default=[],
        help="Override profile case labels with this case. May be repeated.",
    )
    parser.add_argument(
        "--case-jsonl",
        dest="case_jsonls",
        action="append",
        type=Path,
        default=[],
        help=(
            "Forward an external byte_lm_mlp_lora_sweep.py case JSONL into "
            "each materialized profile command. May be repeated."
        ),
    )
    parser.add_argument(
        "--lora-config-jsonl",
        dest="lora_config_jsonls",
        action="append",
        type=Path,
        default=[],
        help=(
            "Forward an external byte_lm_mlp_lora_sweep.py LoRA config JSONL "
            "into each materialized profile command. May be repeated."
        ),
    )
    parser.add_argument(
        "--config",
        dest="configs",
        action="append",
        default=[],
        help="Override the profile base LoRA config. May be repeated.",
    )
    parser.add_argument(
        "--override-ft-epochs",
        type=int,
        default=None,
        help=(
            "Override the materialized sweep FT epoch count while keeping "
            "profile command metadata and training_policy_key in sync."
        ),
    )
    parser.add_argument(
        "--override-target-min-loss-delta",
        type=float,
        default=None,
        help="Override the materialized sweep target-min-loss-delta policy.",
    )
    parser.add_argument(
        "--override-patience",
        default=None,
        help=(
            "Override materialized sweep early-stopping patience. Use 'none' "
            "to disable patience."
        ),
    )
    parser.add_argument(
        "--override-min-delta",
        type=float,
        default=None,
        help="Override materialized sweep early-stopping min-delta.",
    )
    parser.add_argument(
        "--override-lr-decay-patience",
        default=None,
        help=(
            "Override materialized sweep LR-decay patience. Use 'none' to "
            "disable plateau LR decay."
        ),
    )
    parser.add_argument(
        "--override-lr-decay-factor",
        type=float,
        default=None,
        help="Override materialized sweep LR-decay factor.",
    )
    parser.add_argument(
        "--override-lr-decay-min-delta",
        type=float,
        default=None,
        help="Override materialized sweep LR-decay min-delta.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated per-profile sweep JSONL outputs.",
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Filename prefix for generated per-profile sweep JSONL outputs.",
    )
    parser.add_argument(
        "--commands-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL path for materialized command rows.",
    )
    parser.add_argument(
        "--run-summary-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional JSONL path for checkpoint_source_profile_run rows merged "
            "from generated aggregate outputs. The aggregate files must exist, "
            "either from --run or a previous run with the same output paths."
        ),
    )
    parser.add_argument(
        "--run-events-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional JSONL path for checkpoint_source_profile_command_event "
            "rows emitted while --run executes materialized commands. The file "
            "is updated after each command so partial failures remain inspectable."
        ),
    )
    parser.add_argument(
        "--compare-run-summary-jsonl",
        type=Path,
        default=None,
        help="Optional previous checkpoint_source_profile_run JSONL baseline to compare.",
    )
    parser.add_argument(
        "--current-run-summary-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional current checkpoint_source_profile_run JSONL for standalone "
            "comparison against --compare-run-summary-jsonl."
        ),
    )
    parser.add_argument(
        "--promotion-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional JSONL path for checkpoint_source_profile_promotion rows "
            "ranked from the current run summary. Works with generated summaries "
            "or --current-run-summary-jsonl."
        ),
    )
    parser.add_argument(
        "--promotion-metric",
        choices=PROMOTION_METRICS,
        default=DEFAULT_PROMOTION_METRIC,
        help=(
            "Run-summary metric used to rank checkpoint_source_profile_promotion "
            f"rows. Defaults to {DEFAULT_PROMOTION_METRIC}."
        ),
    )
    parser.add_argument(
        "--promotion-ready-top-k",
        type=int,
        default=1,
        help=(
            "Mark the top K ranked checkpoint_source_profile_promotion rows as "
            "promotion-ready. Tied best rows are always ready. Defaults to 1."
        ),
    )
    parser.add_argument(
        "--promotion-ready-within",
        type=float,
        default=None,
        help=(
            "Also mark rows promotion-ready when their --promotion-metric value "
            "is within this absolute gap from the best value."
        ),
    )
    parser.add_argument(
        "--promotion-ready-min-target-retention-ratio",
        type=float,
        default=None,
        help=(
            "Only mark promotion rows ready when target_retention_ratio clears "
            "this floor. Non-ready rows remain in the promotion JSONL."
        ),
    )
    parser.add_argument(
        "--promotion-ready-min-accepted-rate",
        type=float,
        default=None,
        help=(
            "Only mark promotion rows ready when accepted_rate clears this 0..1 "
            "floor. Non-ready rows remain in the promotion JSONL."
        ),
    )
    parser.add_argument(
        "--promotion-ready-min-movement-ok-rate",
        type=float,
        default=None,
        help=(
            "Only mark promotion rows ready when movement_ok_rate clears this "
            "0..1 floor. Non-ready rows remain in the promotion JSONL."
        ),
    )
    parser.add_argument(
        "--promotion-ready-min-retention-accuracy-margin",
        type=float,
        default=None,
        help=(
            "Only mark promotion rows ready when retention_accuracy_margin_min "
            "clears this floor. Non-ready rows remain in the promotion JSONL."
        ),
    )
    parser.add_argument(
        "--promotion-ready-min-retention-perplexity-margin",
        type=float,
        default=None,
        help=(
            "Only mark promotion rows ready when retention_perplexity_margin_min "
            "clears this floor. Non-ready rows remain in the promotion JSONL."
        ),
    )
    parser.add_argument(
        "--promotion-ready-min-epoch-wgpu-hit-rate",
        type=float,
        default=None,
        help=(
            "Only mark promotion rows ready when "
            "epoch_tensor_backend_requested_wgpu_hit_rate_mean clears this "
            "0..1 floor. Non-ready rows remain in the promotion JSONL."
        ),
    )
    parser.add_argument(
        "--promotion-ready-max-epoch-wgpu-runtime-fallback-rate",
        type=float,
        default=None,
        help=(
            "Only mark promotion rows ready when "
            "epoch_tensor_backend_requested_wgpu_runtime_fallback_rate_mean is "
            "at or below this 0..1 ceiling."
        ),
    )
    parser.add_argument(
        "--promotion-ready-max-epoch-wgpu-component-fallback-rate",
        type=float,
        default=None,
        help=(
            "Only mark promotion rows ready when "
            "epoch_tensor_backend_requested_wgpu_component_fallback_rate_mean "
            "is at or below this 0..1 ceiling."
        ),
    )
    parser.add_argument(
        "--promotion-ready-max-input-promotion-metric-regression",
        type=float,
        default=None,
        help=(
            "Only mark promotion rows ready when input_promotion_metric_regression "
            "is at or below this ceiling. Non-ready rows remain in the promotion JSONL."
        ),
    )
    parser.add_argument(
        "--promotion-ready-require-guard-counts-available",
        action="store_true",
        help=(
            "Only mark promotion rows ready when all aggregated guard epoch counts "
            "came from exact backend diagnostics."
        ),
    )
    parser.add_argument(
        "--promotion-ready-min-guard-acceptance-rate-mean",
        type=float,
        default=None,
        help=(
            "Only mark promotion rows ready when guard_acceptance_rate_mean "
            "clears this 0..1 floor."
        ),
    )
    parser.add_argument(
        "--promotion-ready-max-guard-retention-rejected-epochs-mean",
        type=float,
        default=None,
        help=(
            "Only mark promotion rows ready when guard_retention_rejected_epochs_mean "
            "is at or below this ceiling."
        ),
    )
    parser.add_argument(
        "--promotion-ready-max-guard-retention-rejected-rate-mean",
        type=float,
        default=None,
        help=(
            "Only mark promotion rows ready when guard_retention_rejected_rate_mean "
            "is at or below this 0..1 ceiling."
        ),
    )
    parser.add_argument(
        "--promotion-ready-max-guard-target-stale-rate-mean",
        type=float,
        default=None,
        help=(
            "Only mark promotion rows ready when guard_target_stale_rate_mean "
            "is at or below this 0..1 ceiling."
        ),
    )
    parser.add_argument(
        "--promotion-ready-max-guard-target-stale-epochs-mean",
        type=float,
        default=None,
        help=(
            "Only mark promotion rows ready when guard_target_stale_epochs_mean "
            "is at or below this ceiling."
        ),
    )
    parser.add_argument(
        "--min-promotion-ready-count",
        type=int,
        default=None,
        help=(
            "Fail after writing --promotion-jsonl or before materializing "
            "--promotion-input-jsonl when fewer than this many promotion rows "
            "are ready."
        ),
    )
    parser.add_argument(
        "--min-promotion-ready-rate",
        type=float,
        default=None,
        help=(
            "Fail after writing --promotion-jsonl or before materializing "
            "--promotion-input-jsonl when the ready row fraction is below this "
            "0..1 floor."
        ),
    )
    parser.add_argument(
        "--min-promotion-ready-guard-policy-count",
        type=int,
        default=None,
        help=(
            "Fail after writing --promotion-jsonl or before materializing "
            "--promotion-input-jsonl when fewer than this many ready promotion "
            "rows carry guard-readiness policy."
        ),
    )
    parser.add_argument(
        "--require-promotion-ready-guard-policy",
        action="store_true",
        help="Fail when any ready promotion row lacks guard-readiness policy.",
    )
    parser.add_argument(
        "--max-run-target-loss-regression",
        type=float,
        default=None,
        help="Fail when target_loss_delta_mean regresses from the run-summary baseline.",
    )
    parser.add_argument(
        "--max-run-retention-loss-regression",
        type=float,
        default=None,
        help="Fail when retention_loss_delta_mean regresses from the run-summary baseline.",
    )
    parser.add_argument(
        "--max-run-target-retention-gap-regression",
        type=float,
        default=None,
        help="Fail when target_retention_gap_mean regresses from the run-summary baseline.",
    )
    parser.add_argument(
        "--max-run-target-retention-ratio-regression",
        type=float,
        default=None,
        help="Fail when target_retention_ratio regresses from the run-summary baseline.",
    )
    parser.add_argument(
        "--min-run-target-retention-ratio",
        type=float,
        default=None,
        help="Fail when current-run target_retention_ratio is below this floor.",
    )
    parser.add_argument(
        "--max-run-accepted-rate-regression",
        type=float,
        default=None,
        help="Fail when accepted_rate drops from the run-summary baseline.",
    )
    parser.add_argument(
        "--min-run-accepted-rate",
        type=float,
        default=None,
        help="Fail when current-run accepted_rate is below this 0..1 floor.",
    )
    parser.add_argument(
        "--max-run-movement-ok-rate-regression",
        type=float,
        default=None,
        help="Fail when movement_ok_rate drops from the run-summary baseline.",
    )
    parser.add_argument(
        "--max-run-guard-acceptance-rate-regression",
        type=float,
        default=None,
        help=(
            "Fail when guard_acceptance_rate_mean drops from the run-summary "
            "baseline by more than this amount."
        ),
    )
    parser.add_argument(
        "--max-run-guard-retention-rejected-rate-regression",
        type=float,
        default=None,
        help=(
            "Fail when guard_retention_rejected_rate_mean rises from the "
            "run-summary baseline by more than this amount."
        ),
    )
    parser.add_argument(
        "--max-run-guard-target-stale-rate-regression",
        type=float,
        default=None,
        help=(
            "Fail when guard_target_stale_rate_mean rises from the "
            "run-summary baseline by more than this amount."
        ),
    )
    parser.add_argument(
        "--max-run-epoch-wgpu-hit-rate-regression",
        type=float,
        default=None,
        help=(
            "Fail when epoch_tensor_backend_requested_wgpu_hit_rate_mean drops "
            "from the run-summary baseline by more than this amount."
        ),
    )
    parser.add_argument(
        "--max-run-epoch-wgpu-runtime-fallback-rate-regression",
        type=float,
        default=None,
        help=(
            "Fail when epoch_tensor_backend_requested_wgpu_runtime_fallback_rate_mean "
            "rises from the run-summary baseline by more than this amount."
        ),
    )
    parser.add_argument(
        "--max-run-epoch-wgpu-component-fallback-rate-regression",
        type=float,
        default=None,
        help=(
            "Fail when epoch_tensor_backend_requested_wgpu_component_fallback_rate_mean "
            "rises from the run-summary baseline by more than this amount."
        ),
    )
    parser.add_argument(
        "--min-run-movement-ok-rate",
        type=float,
        default=None,
        help="Fail when current-run movement_ok_rate is below this 0..1 floor.",
    )
    parser.add_argument(
        "--min-run-retention-accuracy-margin",
        type=float,
        default=None,
        help="Fail when retention_accuracy_margin_min is below this current-run floor.",
    )
    parser.add_argument(
        "--min-run-retention-perplexity-margin",
        type=float,
        default=None,
        help="Fail when retention_perplexity_margin_min is below this current-run floor.",
    )
    parser.add_argument(
        "--max-run-input-promotion-metric-regression",
        type=float,
        default=None,
        help=(
            "Fail when a promoted run's current input_promotion_metric value "
            "falls below input_promotion_value by more than this amount."
        ),
    )
    parser.add_argument(
        "--min-run-epoch-wgpu-hit-rate",
        type=float,
        default=None,
        help=(
            "Fail when current-run epoch_tensor_backend_requested_wgpu_hit_rate_mean "
            "is below this 0..1 floor."
        ),
    )
    parser.add_argument(
        "--max-run-epoch-wgpu-runtime-fallback-rate",
        type=float,
        default=None,
        help=(
            "Fail when current-run "
            "epoch_tensor_backend_requested_wgpu_runtime_fallback_rate_mean "
            "exceeds this 0..1 ceiling."
        ),
    )
    parser.add_argument(
        "--max-run-epoch-wgpu-component-fallback-rate",
        type=float,
        default=None,
        help=(
            "Fail when current-run "
            "epoch_tensor_backend_requested_wgpu_component_fallback_rate_mean "
            "exceeds this 0..1 ceiling."
        ),
    )
    parser.add_argument(
        "--require-run-guard-counts-available",
        action="store_true",
        help="Fail when current-run guard epoch counts are not all exact backend diagnostics.",
    )
    parser.add_argument(
        "--min-run-guard-acceptance-rate-mean",
        type=float,
        default=None,
        help="Fail when current-run guard_acceptance_rate_mean is below this 0..1 floor.",
    )
    parser.add_argument(
        "--max-run-guard-retention-rejected-epochs-mean",
        type=float,
        default=None,
        help="Fail when current-run guard_retention_rejected_epochs_mean exceeds this ceiling.",
    )
    parser.add_argument(
        "--max-run-guard-target-stale-epochs-mean",
        type=float,
        default=None,
        help="Fail when current-run guard_target_stale_epochs_mean exceeds this ceiling.",
    )
    parser.add_argument(
        "--max-run-guard-retention-rejected-rate-mean",
        type=float,
        default=None,
        help="Fail when current-run guard_retention_rejected_rate_mean exceeds this 0..1 ceiling.",
    )
    parser.add_argument(
        "--max-run-guard-target-stale-rate-mean",
        type=float,
        default=None,
        help="Fail when current-run guard_target_stale_rate_mean exceeds this 0..1 ceiling.",
    )
    parser.add_argument(
        "--require-run-source-match",
        action="store_true",
        help="Fail when a profile lane changes selected_source versus the baseline.",
    )
    parser.add_argument(
        "--require-run-config-match",
        action="store_true",
        help="Fail when a profile lane changes config versus the baseline.",
    )
    parser.add_argument(
        "--require-run-case-scope-match",
        action="store_true",
        help="Fail when a profile lane changes cases or case_labels versus the baseline.",
    )
    parser.add_argument(
        "--require-run-training-policy-match",
        action="store_true",
        help=(
            "Fail when a profile lane changes adapter decay, clipping, accumulation, "
            "FT-control policy, or training_policy_key versus the baseline."
        ),
    )
    parser.add_argument(
        "--require-run-input-promotion-match",
        action="store_true",
        help=(
            "Fail when a promoted profile lane changes input_promotion_* provenance "
            "versus the baseline."
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run byte_lm_mlp_lora_sweep.py.",
    )
    parser.add_argument(
        "--sweep-script",
        type=Path,
        default=Path(__file__).with_name("byte_lm_mlp_lora_sweep.py"),
        help="Path to byte_lm_mlp_lora_sweep.py.",
    )
    parser.add_argument(
        "--extra-arg",
        dest="extra_args",
        action="append",
        default=[],
        help="Extra argument appended to every sweep command. May be repeated.",
    )
    parser.add_argument(
        "--no-aggregate-gates",
        action="store_true",
        help="Do not add multi-case aggregate coverage gates to generated commands.",
    )
    parser.add_argument(
        "--min-aggregate-epoch-wgpu-hit-rate",
        type=float,
        default=None,
        help=(
            "Forward a current aggregate "
            "epoch_tensor_backend_requested_wgpu_hit_rate_mean floor to every "
            "materialized sweep command."
        ),
    )
    parser.add_argument(
        "--max-aggregate-epoch-wgpu-runtime-fallback-rate",
        type=float,
        default=None,
        help=(
            "Forward a current aggregate "
            "epoch_tensor_backend_requested_wgpu_runtime_fallback_rate_mean "
            "ceiling to every materialized sweep command."
        ),
    )
    parser.add_argument(
        "--max-aggregate-epoch-wgpu-component-fallback-rate",
        type=float,
        default=None,
        help=(
            "Forward a current aggregate "
            "epoch_tensor_backend_requested_wgpu_component_fallback_rate_mean "
            "ceiling to every materialized sweep command."
        ),
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually run generated commands. Without this, commands are only printed/written.",
    )
    args = parser.parse_args()
    if len(set(args.profiles)) != len(args.profiles):
        parser.error("--profile values must be unique")
    if args.output_prefix == "":
        parser.error("--output-prefix must be non-empty")
    if args.override_ft_epochs is not None and args.override_ft_epochs <= 0:
        parser.error("--override-ft-epochs must be positive")
    for name in [
        "override_target_min_loss_delta",
        "override_min_delta",
        "override_lr_decay_min_delta",
    ]:
        value = getattr(args, name)
        if value is not None and value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.override_lr_decay_factor is not None and not (
        0.0 < args.override_lr_decay_factor < 1.0
    ):
        parser.error("--override-lr-decay-factor must be in (0, 1)")
    args.ft_control_override = ft_control_override_from_args(args, parser=parser)
    if args.promotion_ready_top_k < 1:
        parser.error("--promotion-ready-top-k must be at least 1")
    if args.promotion_ready_within is not None and args.promotion_ready_within < 0.0:
        parser.error("--promotion-ready-within must be non-negative")
    for name in [
        "promotion_ready_min_target_retention_ratio",
        "promotion_ready_min_accepted_rate",
        "promotion_ready_min_movement_ok_rate",
        "promotion_ready_min_retention_accuracy_margin",
        "promotion_ready_min_retention_perplexity_margin",
        "promotion_ready_min_epoch_wgpu_hit_rate",
        "promotion_ready_max_epoch_wgpu_runtime_fallback_rate",
        "promotion_ready_max_epoch_wgpu_component_fallback_rate",
        "promotion_ready_max_input_promotion_metric_regression",
        "promotion_ready_min_guard_acceptance_rate_mean",
        "promotion_ready_max_guard_retention_rejected_epochs_mean",
        "promotion_ready_max_guard_target_stale_epochs_mean",
        "promotion_ready_max_guard_retention_rejected_rate_mean",
        "promotion_ready_max_guard_target_stale_rate_mean",
    ]:
        value = getattr(args, name)
        if value is not None and value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
        if (
            value is not None
            and ("_rate_" in name or name.endswith("_rate"))
            and value > 1.0
        ):
            parser.error(f"--{name.replace('_', '-')} must be at most 1.0")
    if args.min_promotion_ready_count is not None and args.min_promotion_ready_count < 0:
        parser.error("--min-promotion-ready-count must be non-negative")
    if (
        args.min_promotion_ready_guard_policy_count is not None
        and args.min_promotion_ready_guard_policy_count < 0
    ):
        parser.error("--min-promotion-ready-guard-policy-count must be non-negative")
    if args.min_promotion_ready_rate is not None:
        if args.min_promotion_ready_rate < 0.0:
            parser.error("--min-promotion-ready-rate must be non-negative")
        if args.min_promotion_ready_rate > 1.0:
            parser.error("--min-promotion-ready-rate must be at most 1.0")
    if (
        (
            args.min_promotion_ready_count is not None
            or args.min_promotion_ready_rate is not None
            or args.min_promotion_ready_guard_policy_count is not None
            or args.require_promotion_ready_guard_policy
        )
        and args.promotion_jsonl is None
        and args.promotion_input_jsonl is None
    ):
        parser.error("promotion ready gates require --promotion-jsonl or --promotion-input-jsonl")
    if args.profile_jsonl is None and args.current_run_summary_jsonl is None:
        parser.error("--profile-jsonl is required unless --current-run-summary-jsonl is provided")
    if args.current_run_summary_jsonl is not None:
        if args.compare_run_summary_jsonl is None:
            parser.error("--current-run-summary-jsonl requires --compare-run-summary-jsonl")
        if (
            args.run
            or args.run_summary_jsonl is not None
            or args.run_events_jsonl is not None
            or args.commands_jsonl is not None
            or args.promotion_input_jsonl is not None
            or args.promotion_selection_jsonl is not None
            or args.case_jsonls
            or args.lora_config_jsonls
            or args.ft_control_override is not None
        ):
            parser.error(
                "--current-run-summary-jsonl cannot be combined with --run, "
                "--run-summary-jsonl, --run-events-jsonl, --commands-jsonl, "
                "--promotion-input-jsonl, --promotion-selection-jsonl, or "
                "--case-jsonl/--lora-config-jsonl/--override-ft-*"
            )
    if args.promotion_selection_jsonl is not None and args.promotion_input_jsonl is None:
        parser.error("--promotion-selection-jsonl requires --promotion-input-jsonl")
    if args.run_events_jsonl is not None and not args.run:
        parser.error("--run-events-jsonl requires --run")
    for name in [
        "max_run_target_loss_regression",
        "max_run_retention_loss_regression",
        "max_run_target_retention_gap_regression",
        "max_run_target_retention_ratio_regression",
        "min_run_target_retention_ratio",
        "max_run_accepted_rate_regression",
        "min_run_accepted_rate",
        "max_run_movement_ok_rate_regression",
        "max_run_guard_acceptance_rate_regression",
        "max_run_guard_retention_rejected_rate_regression",
        "max_run_guard_target_stale_rate_regression",
        "max_run_epoch_wgpu_hit_rate_regression",
        "max_run_epoch_wgpu_runtime_fallback_rate_regression",
        "max_run_epoch_wgpu_component_fallback_rate_regression",
        "min_run_movement_ok_rate",
        "min_run_retention_accuracy_margin",
        "min_run_retention_perplexity_margin",
        "max_run_input_promotion_metric_regression",
        "min_run_epoch_wgpu_hit_rate",
        "max_run_epoch_wgpu_runtime_fallback_rate",
        "max_run_epoch_wgpu_component_fallback_rate",
        "min_run_guard_acceptance_rate_mean",
        "max_run_guard_retention_rejected_epochs_mean",
        "max_run_guard_target_stale_epochs_mean",
        "max_run_guard_retention_rejected_rate_mean",
        "max_run_guard_target_stale_rate_mean",
        "min_aggregate_epoch_wgpu_hit_rate",
        "max_aggregate_epoch_wgpu_runtime_fallback_rate",
        "max_aggregate_epoch_wgpu_component_fallback_rate",
    ]:
        value = getattr(args, name)
        if value is not None and value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
        if (
            value is not None
            and ("_rate_" in name or name.endswith("_rate"))
            and value > 1.0
        ):
            parser.error(f"--{name.replace('_', '-')} must be at most 1.0")
    aggregate_epoch_wgpu_gate_requested = (
        args.min_aggregate_epoch_wgpu_hit_rate is not None
        or args.max_aggregate_epoch_wgpu_runtime_fallback_rate is not None
        or args.max_aggregate_epoch_wgpu_component_fallback_rate is not None
    )
    if args.no_aggregate_gates and aggregate_epoch_wgpu_gate_requested:
        parser.error(
            "aggregate epoch WGPU gates cannot be combined with --no-aggregate-gates"
        )
    run_gate_requested = (
        args.max_run_target_loss_regression is not None
        or args.max_run_retention_loss_regression is not None
        or args.max_run_target_retention_gap_regression is not None
        or args.max_run_target_retention_ratio_regression is not None
        or args.max_run_accepted_rate_regression is not None
        or args.max_run_movement_ok_rate_regression is not None
        or args.max_run_guard_acceptance_rate_regression is not None
        or args.max_run_guard_retention_rejected_rate_regression is not None
        or args.max_run_guard_target_stale_rate_regression is not None
        or args.max_run_epoch_wgpu_hit_rate_regression is not None
        or args.max_run_epoch_wgpu_runtime_fallback_rate_regression is not None
        or args.max_run_epoch_wgpu_component_fallback_rate_regression is not None
        or args.require_run_source_match
        or args.require_run_config_match
        or args.require_run_case_scope_match
        or args.require_run_training_policy_match
        or args.require_run_input_promotion_match
    )
    if run_gate_requested and args.compare_run_summary_jsonl is None:
        parser.error("run-summary regression gates require --compare-run-summary-jsonl")
    current_run_gate_requested = (
        args.min_run_target_retention_ratio is not None
        or args.min_run_accepted_rate is not None
        or args.min_run_movement_ok_rate is not None
        or args.min_run_retention_accuracy_margin is not None
        or args.min_run_retention_perplexity_margin is not None
        or args.max_run_input_promotion_metric_regression is not None
        or args.min_run_epoch_wgpu_hit_rate is not None
        or args.max_run_epoch_wgpu_runtime_fallback_rate is not None
        or args.max_run_epoch_wgpu_component_fallback_rate is not None
        or args.require_run_guard_counts_available
        or args.min_run_guard_acceptance_rate_mean is not None
        or args.max_run_guard_retention_rejected_epochs_mean is not None
        or args.max_run_guard_target_stale_epochs_mean is not None
        or args.max_run_guard_retention_rejected_rate_mean is not None
        or args.max_run_guard_target_stale_rate_mean is not None
    )
    if (
        current_run_gate_requested
        and args.current_run_summary_jsonl is None
        and args.run_summary_jsonl is None
        and args.compare_run_summary_jsonl is None
        and args.promotion_jsonl is None
    ):
        parser.error(
            "current-run summary gates require --run-summary-jsonl or "
            "--compare-run-summary-jsonl or --promotion-jsonl"
        )
    parse_source_paths(args.source_paths, parser=parser)
    return args


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def is_numeric_value(value):
    return not isinstance(value, bool) and isinstance(value, (int, float))


def numeric_value(row, key):
    value = row.get(key)
    if not is_numeric_value(value):
        raise ValueError(f"{row.get('source_profile')} row missing numeric {key}")
    return float(value)


def optional_numeric_value(row, key):
    value = row.get(key)
    if value is None:
        return None
    if not is_numeric_value(value):
        raise ValueError(f"{row.get('source_profile')} row missing numeric {key}")
    return float(value)


def target_retention_selectivity(row):
    target = numeric_value(row, "target_loss_delta_mean")
    retention = numeric_value(row, "retention_loss_delta_mean")
    gap = row.get("target_retention_gap_mean")
    if gap is None:
        gap = target - retention
    elif not is_numeric_value(gap):
        raise ValueError(
            f"{row.get('source_profile')} row missing numeric target_retention_gap_mean"
        )
    else:
        gap = float(gap)

    ratio = row.get("target_retention_ratio")
    if ratio is None:
        ratio = target / retention if retention > 0.0 else None
    elif not is_numeric_value(ratio):
        raise ValueError(
            f"{row.get('source_profile')} row missing numeric target_retention_ratio"
        )
    else:
        ratio = float(ratio)
    return gap, ratio


def normalize_profile_run_selectivity(row):
    normalized = dict(row)
    gap, ratio = target_retention_selectivity(normalized)
    normalized["target_retention_gap_mean"] = gap
    normalized["target_retention_ratio"] = ratio
    return normalized


def normalize_promotion_selectivity(row):
    normalized = dict(row)
    if (
        "target_loss_delta_mean" in normalized
        or "retention_loss_delta_mean" in normalized
    ):
        gap, ratio = target_retention_selectivity(normalized)
        normalized["target_retention_gap_mean"] = gap
        normalized["target_retention_ratio"] = ratio
    metric = normalized.get("promotion_metric")
    if (
        normalized.get("promotion_value") is None
        and metric in PROMOTION_METRICS
        and is_numeric_value(normalized.get(metric))
    ):
        normalized["promotion_value"] = float(normalized[metric])
    return normalized


def optional_numeric_label(value):
    if value is None:
        return "none"
    if is_numeric_value(value):
        return f"{float(value):.9f}"
    return str(value)


def parse_source_paths(entries, *, parser=None):
    mapping = {}
    for entry in entries:
        if "=" not in entry:
            message = f"--source-path must be source_label=/path, got {entry!r}"
            if parser is not None:
                parser.error(message)
            raise ValueError(message)
        label, path = entry.split("=", 1)
        if not label or not path:
            message = f"--source-path must include non-empty label and path: {entry!r}"
            if parser is not None:
                parser.error(message)
            raise ValueError(message)
        if label in mapping:
            message = f"duplicate --source-path label: {label}"
            if parser is not None:
                parser.error(message)
            raise ValueError(message)
        mapping[label] = Path(path)
    return mapping


def profile_rows(rows, profile_names=None):
    requested = set(profile_names or [])
    seen = set()
    profiles = []
    for row in rows:
        if row.get("row_type") != "checkpoint_source_profile":
            continue
        profile = row.get("source_profile")
        if not isinstance(profile, str) or not profile:
            raise ValueError("checkpoint_source_profile row missing source_profile")
        if requested and profile not in requested:
            continue
        if profile in seen:
            raise ValueError(f"duplicate checkpoint source profile: {profile}")
        seen.add(profile)
        profiles.append(row)
    missing = sorted(requested - seen)
    if missing:
        raise ValueError(f"profile JSONL missing requested profile(s): {','.join(missing)}")
    if not profiles:
        raise ValueError("profile JSONL contains no selected checkpoint_source_profile rows")
    return profiles


def promotion_run_key(row):
    run_key = row.get("run_key")
    if isinstance(run_key, str) and run_key:
        return run_key
    profile = row.get("source_profile")
    if not isinstance(profile, str) or not profile:
        raise ValueError("checkpoint_source_profile_promotion row missing source_profile")
    config = row.get("config") or row.get("selected_config") or row.get("run_config_key")
    if isinstance(config, str) and config:
        return f"{profile}::{config}"
    return profile


def promotion_non_ready_reason(row):
    failures = row.get("promotion_ready_floor_failures")
    if isinstance(failures, list) and failures:
        labels = [str(failure) for failure in failures]
        return "floor_failures=" + ",".join(labels)
    if row.get("promotion_ready_floor_passed") is False:
        return "floor_failed"
    return f"promotion_ready={row.get('promotion_ready')}"


GUARD_PROMOTION_FAILURE_PREFIXES = (
    "guard_epoch_counts_available_all",
    "guard_acceptance_rate_mean",
    "guard_retention_rejected_epochs_mean",
    "guard_target_stale_epochs_mean",
    "guard_retention_rejected_rate_mean",
    "guard_target_stale_rate_mean",
)


def promotion_guard_policy_requested(row):
    return (
        row.get("promotion_ready_require_guard_counts_available") is True
        or row.get("promotion_ready_min_guard_acceptance_rate_mean") is not None
        or row.get("promotion_ready_max_guard_retention_rejected_epochs_mean") is not None
        or row.get("promotion_ready_max_guard_target_stale_epochs_mean") is not None
        or row.get("promotion_ready_max_guard_retention_rejected_rate_mean") is not None
        or row.get("promotion_ready_max_guard_target_stale_rate_mean") is not None
    )


def promotion_guard_failure_reasons(row):
    failures = row.get("promotion_ready_floor_failures")
    if not isinstance(failures, list):
        return []
    reasons = []
    for failure in failures:
        label = str(failure)
        if any(label.startswith(prefix) for prefix in GUARD_PROMOTION_FAILURE_PREFIXES):
            reasons.append(label)
    return reasons


def no_ready_promotions_message(rows):
    details = []
    for row in rows[:5]:
        details.append(
            f"{promotion_run_key(row)}({promotion_non_ready_reason(row)})"
        )
    suffix = ""
    if len(rows) > len(details):
        suffix = f"; +{len(rows) - len(details)} more"
    return (
        "promotion JSONL contains no ready checkpoint_source_profile_promotion rows; "
        "non_ready=" + "; ".join(details) + suffix
    )


def selected_promotion_rows(rows, profile_names=None):
    requested = set(profile_names or [])
    selected = []
    for row in rows:
        if row.get("row_type") != "checkpoint_source_profile_promotion":
            continue
        profile = row.get("source_profile")
        if not isinstance(profile, str) or not profile:
            raise ValueError("checkpoint_source_profile_promotion row missing source_profile")
        if requested and profile not in requested:
            continue
        selected.append(normalize_promotion_selectivity(row))
    return selected


def promotion_selection_summary(rows, profile_names=None, *, ready_only=True):
    selected = selected_promotion_rows(rows, profile_names)
    ready = [row for row in selected if row.get("promotion_ready") is True]
    non_ready = [row for row in selected if row.get("promotion_ready") is not True]
    materialized = ready if ready_only else selected
    materialized_count = len(materialized)
    non_ready_details = [
        f"{promotion_run_key(row)}({promotion_non_ready_reason(row)})"
        for row in non_ready[:5]
    ]
    guard_failures = [
        (row, promotion_guard_failure_reasons(row))
        for row in non_ready
        if promotion_guard_failure_reasons(row)
    ]
    guard_failure_details = [
        f"{promotion_run_key(row)}({','.join(reasons)})"
        for row, reasons in guard_failures[:5]
    ]
    return {
        "row_type": "checkpoint_source_profile_promotion_selection",
        "selected_promotions": len(selected),
        "ready_promotions": len(ready),
        "non_ready_promotions": len(non_ready),
        "materialized_promotions": materialized_count,
        "guard_policy_promotions": sum(
            1 for row in selected if promotion_guard_policy_requested(row)
        ),
        "ready_guard_policy_promotions": sum(
            1 for row in ready if promotion_guard_policy_requested(row)
        ),
        "materialized_guard_policy_promotions": sum(
            1 for row in materialized if promotion_guard_policy_requested(row)
        ),
        "non_ready_guard_failure_promotions": len(guard_failures),
        "promotion_ready_only": ready_only,
        "requested_profiles": ",".join(profile_names or []),
        "non_ready_details": non_ready_details,
        "non_ready_details_truncated": len(non_ready) > len(non_ready_details),
        "non_ready_guard_failure_details": guard_failure_details,
        "non_ready_guard_failure_details_truncated": (
            len(guard_failures) > len(guard_failure_details)
        ),
    }


def print_promotion_selection_summary(row):
    non_ready_label = "none"
    details = row.get("non_ready_details") or []
    if details:
        non_ready_label = ";".join(str(detail) for detail in details)
        if row.get("non_ready_details_truncated"):
            non_ready_label += ";..."
    guard_failure_label = "none"
    guard_details = row.get("non_ready_guard_failure_details") or []
    if guard_details:
        guard_failure_label = ";".join(str(detail) for detail in guard_details)
        if row.get("non_ready_guard_failure_details_truncated"):
            guard_failure_label += ";..."
    print(
        f"profile_promotion_selection selected={row['selected_promotions']} "
        f"ready={row['ready_promotions']} "
        f"non_ready={row['non_ready_promotions']} "
        f"materialized={row['materialized_promotions']} "
        f"guard_policy={row.get('guard_policy_promotions', 0)} "
        f"ready_guard_policy={row.get('ready_guard_policy_promotions', 0)} "
        f"materialized_guard_policy={row.get('materialized_guard_policy_promotions', 0)} "
        f"non_ready_guard_failures={row.get('non_ready_guard_failure_promotions', 0)} "
        f"ready_only={row['promotion_ready_only']} "
        f"requested_profiles={row.get('requested_profiles') or 'all'} "
        f"non_ready_details={non_ready_label} "
        f"non_ready_guard_failure_details={guard_failure_label}"
    )


def promotion_rows(rows, profile_names=None, *, ready_only=True):
    requested = set(profile_names or [])
    seen = set()
    promotions = []
    selected = selected_promotion_rows(rows, profile_names)
    for row in selected:
        if ready_only and row.get("promotion_ready") is not True:
            continue
        run_key = promotion_run_key(row)
        if run_key in seen:
            raise ValueError(f"duplicate checkpoint source promotion run key: {run_key}")
        seen.add(run_key)
        promotions.append(row)
    if not promotions:
        if ready_only and selected:
            raise ValueError(no_ready_promotions_message(selected))
        raise ValueError("promotion JSONL contains no selected checkpoint_source_profile_promotion rows")
    return sorted(
        promotions,
        key=lambda row: (
            int(row.get("promotion_rank"))
            if isinstance(row.get("promotion_rank"), int)
            and not isinstance(row.get("promotion_rank"), bool)
            else 0,
            promotion_run_key(row),
        ),
    )


def case_count(row):
    value = row.get("cases")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def validate_promotion_case_scope(profile_row, promotion):
    profile = profile_row.get("source_profile")
    profile_labels = profile_case_labels(profile_row)
    promotion_labels = profile_case_labels(promotion)
    if promotion_labels and promotion_labels != profile_labels:
        raise ValueError(
            f"promotion case scope mismatch for {profile}: "
            f"profile={','.join(profile_labels) or 'none'} "
            f"promotion={','.join(promotion_labels)}"
        )
    profile_cases = case_count(profile_row)
    promotion_cases = case_count(promotion)
    if (
        profile_cases is not None
        and promotion_cases is not None
        and profile_cases != promotion_cases
    ):
        raise ValueError(
            f"promotion case count mismatch for {profile}: "
            f"profile={profile_cases} promotion={promotion_cases}"
        )


def promotion_source_label(promotion):
    label = promotion.get("selected_source") or promotion.get("checkpoint_source_label")
    if not isinstance(label, str) or not label:
        profile = promotion.get("source_profile")
        raise ValueError(f"promotion selected_source missing for {profile}")
    return label


def promotion_config_key(promotion):
    config = (
        promotion.get("config")
        or promotion.get("selected_config")
        or promotion.get("run_config_key")
    )
    if not isinstance(config, str) or not config:
        profile = promotion.get("source_profile")
        raise ValueError(f"promotion config missing for {profile}")
    return config


def parse_training_policy_key(policy_key):
    if not isinstance(policy_key, str) or "=" not in policy_key:
        return None
    parsed = {}
    for part in policy_key.split("|"):
        if "=" not in part:
            return None
        key, value = part.split("=", 1)
        if not key:
            return None
        parsed[key] = value
    return parsed


def training_policies_match_except_ft_control(profile_policy, promoted_policy):
    profile_values = parse_training_policy_key(profile_policy)
    promoted_values = parse_training_policy_key(promoted_policy)
    if profile_values is None or promoted_values is None:
        return False
    for field in TRAINING_POLICY_FIELDS:
        if field in FT_CONTROL_POLICY_FIELDS:
            continue
        if profile_values.get(field) != promoted_values.get(field):
            return False
    return True


def validate_promotion_training_policy(profile_row, promotion):
    profile = profile_row.get("source_profile")
    promoted_policy = promotion.get("training_policy_key")
    if not isinstance(promoted_policy, str) or not promoted_policy:
        raise ValueError(f"promotion training_policy_key missing for {profile}")
    profile_policy = profile_row.get("training_policy_key")
    if not isinstance(profile_policy, str) or not profile_policy:
        raise ValueError(f"profile training_policy_key missing for promotion {profile}")
    accepted_profile_policies = {profile_policy}
    normalized_policy = training_policy_key(normalized_training_policy_row(profile_row))
    if normalized_policy:
        accepted_profile_policies.add(normalized_policy)
    if promoted_policy not in accepted_profile_policies and not any(
        training_policies_match_except_ft_control(profile_policy, promoted_policy)
        for profile_policy in accepted_profile_policies
    ):
        raise ValueError(
            f"promotion training_policy_key mismatch for {profile}: "
            f"profile={profile_policy} promotion={promoted_policy}"
        )


def promoted_profile_rows(profile_jsonl_rows, promotion_jsonl_rows, profile_names=None, *, ready_only=True):
    base_profiles = {
        row["source_profile"]: row
        for row in profile_rows(profile_jsonl_rows, profile_names)
    }
    promoted = []
    for promotion in promotion_rows(
        promotion_jsonl_rows,
        profile_names,
        ready_only=ready_only,
    ):
        profile = promotion["source_profile"]
        if profile not in base_profiles:
            raise ValueError(f"promotion JSONL selected profile missing from profile JSONL: {profile}")
        row = dict(base_profiles[profile])
        promoted_source = promotion_source_label(promotion)
        source = selected_source_label(row)
        if promoted_source != source:
            raise ValueError(
                f"promotion source mismatch for {profile}: profile={source} promotion={promoted_source}"
            )
        row["selected_source"] = promoted_source
        validate_promotion_case_scope(row, promotion)
        selected_config = promotion_config_key(promotion)
        row["selected_config"] = selected_config
        row["config"] = selected_config
        row["base_config"] = selected_config.split("::", 1)[0]
        validate_promotion_training_policy(row, promotion)
        for key in TRAINING_POLICY_FIELDS:
            if key in promotion:
                row[key] = promotion.get(key)
        row.update(
            {
                "promotion_run_key": promotion_run_key(promotion),
                "promotion_rank": promotion.get("promotion_rank"),
                "promotion_metric": promotion.get("promotion_metric"),
                "promotion_value": promotion.get("promotion_value"),
                "promotion_ready": promotion.get("promotion_ready"),
                "promotion_ready_top_k": promotion.get("promotion_ready_top_k"),
                "promotion_ready_within": promotion.get("promotion_ready_within"),
                "promotion_ready_floor_passed": promotion.get(
                    "promotion_ready_floor_passed"
                ),
                "promotion_ready_floor_failures": promotion.get(
                    "promotion_ready_floor_failures"
                ),
                "promotion_ready_min_target_retention_ratio": promotion.get(
                    "promotion_ready_min_target_retention_ratio"
                ),
                "promotion_ready_min_accepted_rate": promotion.get(
                    "promotion_ready_min_accepted_rate"
                ),
                "promotion_ready_min_movement_ok_rate": promotion.get(
                    "promotion_ready_min_movement_ok_rate"
                ),
                "promotion_ready_min_retention_accuracy_margin": promotion.get(
                    "promotion_ready_min_retention_accuracy_margin"
                ),
                "promotion_ready_min_retention_perplexity_margin": promotion.get(
                    "promotion_ready_min_retention_perplexity_margin"
                ),
                "promotion_ready_min_epoch_wgpu_hit_rate": promotion.get(
                    "promotion_ready_min_epoch_wgpu_hit_rate"
                ),
                "promotion_ready_max_epoch_wgpu_runtime_fallback_rate": promotion.get(
                    "promotion_ready_max_epoch_wgpu_runtime_fallback_rate"
                ),
                "promotion_ready_max_epoch_wgpu_component_fallback_rate": promotion.get(
                    "promotion_ready_max_epoch_wgpu_component_fallback_rate"
                ),
                "promotion_ready_max_input_promotion_metric_regression": promotion.get(
                    "promotion_ready_max_input_promotion_metric_regression"
                ),
                "promotion_ready_require_guard_counts_available": promotion.get(
                    "promotion_ready_require_guard_counts_available"
                ),
                "promotion_ready_min_guard_acceptance_rate_mean": promotion.get(
                    "promotion_ready_min_guard_acceptance_rate_mean"
                ),
                "promotion_ready_max_guard_retention_rejected_epochs_mean": promotion.get(
                    "promotion_ready_max_guard_retention_rejected_epochs_mean"
                ),
                "promotion_ready_max_guard_target_stale_epochs_mean": promotion.get(
                    "promotion_ready_max_guard_target_stale_epochs_mean"
                ),
                "promotion_ready_max_guard_retention_rejected_rate_mean": promotion.get(
                    "promotion_ready_max_guard_retention_rejected_rate_mean"
                ),
                "promotion_ready_max_guard_target_stale_rate_mean": promotion.get(
                    "promotion_ready_max_guard_target_stale_rate_mean"
                ),
                "winner_metric": promotion.get("promotion_metric", row.get("winner_metric")),
                "winner_value": promotion.get("promotion_value", row.get("winner_value")),
            }
        )
        promoted.append(row)
    return promoted


def selected_source_label(row):
    label = row.get("selected_source") or row.get("checkpoint_source_label")
    if not isinstance(label, str) or not label:
        raise ValueError(f"profile {row.get('source_profile')} missing selected source")
    return label


def profile_case_labels(row):
    labels = row.get("case_labels")
    if not isinstance(labels, str):
        return []
    return [label for label in labels.split(",") if label]


def selected_cases(row, override_cases=None):
    cases = list(override_cases or [])
    if cases:
        return cases
    cases = profile_case_labels(row)
    if not cases:
        raise ValueError(f"profile {row.get('source_profile')} has no case labels")
    return cases


def selected_configs(row, override_configs=None):
    configs = list(override_configs or [])
    if configs:
        return configs
    base_config = row.get("base_config")
    if isinstance(base_config, str) and base_config:
        return [base_config]
    selected_config = row.get("selected_config") or row.get("config")
    if isinstance(selected_config, str) and selected_config:
        return [selected_config.split("::", 1)[0]]
    raise ValueError(f"profile {row.get('source_profile')} has no config")


def safe_slug(value):
    chars = []
    for char in str(value).lower():
        if char.isalnum():
            chars.append(char)
        elif chars and chars[-1] != "-":
            chars.append("-")
    slug = "".join(chars).strip("-")
    return slug or "profile"


def profile_policy_key(row, configs):
    selected_config = row.get("selected_config") or row.get("config")
    if not configs:
        if isinstance(selected_config, str) and selected_config:
            return selected_config
        return None
    suffix = None
    if isinstance(selected_config, str) and "::" in selected_config:
        suffix = selected_config.split("::", 1)[1]
    config_key = "::".join(configs)
    if suffix:
        return f"{config_key}::{suffix}"
    if config_key:
        return config_key
    if isinstance(selected_config, str) and selected_config:
        return selected_config
    return None


def label_number(value):
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def optional_int_label(prefix, value):
    return f"{prefix}none" if value is None else f"{prefix}{int(value)}"


def fmt_policy_value(value):
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.9f}"
    return str(value)


def training_policy_key(row):
    return "|".join(
        f"{field}={fmt_policy_value(row.get(field))}" for field in TRAINING_POLICY_FIELDS
    )


def normalized_ft_control_variant(row):
    if not any(
        key in row
        for key in [
            "ft_epochs",
            "target_min_loss_delta_policy",
            "early_stopping_patience",
            "early_stopping_min_delta",
            "lr_decay_patience",
            "lr_decay_factor",
            "lr_decay_min_delta",
        ]
    ):
        return row.get("ft_control_variant")
    labels = []
    ft_epochs = row.get("ft_epochs")
    if is_numeric_value(ft_epochs):
        labels.append(f"ep{int(ft_epochs)}")
    target_min_loss_delta = row.get("target_min_loss_delta_policy")
    if is_numeric_value(target_min_loss_delta):
        labels.append(f"tmin{label_number(target_min_loss_delta)}")
    if "early_stopping_patience" in row:
        labels.append(optional_int_label("pat", row.get("early_stopping_patience")))
    min_delta = row.get("early_stopping_min_delta")
    if is_numeric_value(min_delta):
        labels.append(f"md{label_number(min_delta)}")
    if "lr_decay_patience" in row:
        labels.append(optional_int_label("ldp", row.get("lr_decay_patience")))
    lr_decay_factor = row.get("lr_decay_factor")
    if is_numeric_value(lr_decay_factor):
        labels.append(f"ldf{label_number(lr_decay_factor)}")
    lr_decay_min_delta = row.get("lr_decay_min_delta")
    if is_numeric_value(lr_decay_min_delta):
        labels.append(f"ldmd{label_number(lr_decay_min_delta)}")
    if labels:
        return "::".join(labels)
    return row.get("ft_control_variant")


def normalized_training_policy_row(row):
    policy = dict(row)
    ft_control = normalized_ft_control_variant(policy)
    if ft_control is not None:
        policy["ft_control_variant"] = ft_control
    if policy.get("early_stopping_min_delta") is None and ft_control is not None:
        policy["early_stopping_min_delta"] = 0.0
    if policy.get("lr_decay_factor") is None and ft_control is not None:
        policy["lr_decay_factor"] = 0.5
    if policy.get("lr_decay_min_delta") is None and ft_control is not None:
        policy["lr_decay_min_delta"] = 0.0
    return policy


def canonical_training_policy_key(row):
    return training_policy_key(normalized_training_policy_row(row))


def generated_training_policy_keys(row):
    return {
        training_policy_key(row),
        canonical_training_policy_key(row),
    }


def policy_value(aggregate, command_row, key):
    value = aggregate.get(key)
    if value is not None:
        return value
    return command_row.get(key)


def normalized_profile_policy_key(row, configs):
    key = profile_policy_key(row, configs)
    old_ft_control = row.get("ft_control_variant")
    new_ft_control = normalized_ft_control_variant(row)
    if (
        isinstance(key, str)
        and isinstance(old_ft_control, str)
        and isinstance(new_ft_control, str)
        and old_ft_control
        and old_ft_control != new_ft_control
    ):
        if new_ft_control in key:
            return key
        if old_ft_control not in key:
            return key
        return key.replace(old_ft_control, new_ft_control, 1)
    return key


def profile_output_paths(row, output_dir, output_prefix, configs=None):
    profile = safe_slug(row.get("source_profile"))
    source = safe_slug(selected_source_label(row))
    policy_key = normalized_profile_policy_key(row, list(configs or []))
    if isinstance(policy_key, str) and policy_key:
        policy_slug = safe_slug(policy_key)
        stem = "-".join([output_prefix, profile, source, policy_slug])
        return output_dir / f"{stem}.jsonl", output_dir / f"{stem}-aggregate.jsonl"

    gain = row.get("checkpoint_source_gain", 1.0)
    if isinstance(gain, bool):
        gain_label = str(gain)
    elif isinstance(gain, (int, float)):
        gain_label = f"{float(gain):g}"
    else:
        gain_label = str(gain)
    gain_slug = safe_slug(f"g{gain_label}")
    parts = [output_prefix, profile, source, gain_slug]
    weight_decay = row.get("adapter_weight_decay")
    if row.get("adapter_weight_decay_variant") is not None:
        if isinstance(weight_decay, bool):
            decay_label = str(weight_decay)
        elif isinstance(weight_decay, (int, float)):
            decay_label = f"{float(weight_decay):g}"
        else:
            decay_label = str(weight_decay)
        parts.append(safe_slug(f"wd{decay_label}"))
    stem = "-".join(parts)
    return output_dir / f"{stem}.jsonl", output_dir / f"{stem}-aggregate.jsonl"


def profile_flag_fragment(row):
    fragment = row.get("checkpoint_source_flag_fragment")
    if isinstance(fragment, list) and all(isinstance(value, str) for value in fragment):
        return list(fragment)
    raise ValueError(
        f"profile {row.get('source_profile')} missing checkpoint_source_flag_fragment"
    )


def replace_or_append_ft_control(value, old_ft_controls, new_ft_control):
    if not isinstance(value, str) or not value or not new_ft_control:
        return value
    if value == new_ft_control:
        return value
    candidates = sorted(
        {
            old_ft_control
            for old_ft_control in old_ft_controls
            if isinstance(old_ft_control, str) and old_ft_control
        },
        key=len,
        reverse=True,
    )
    for old_ft_control in candidates:
        if old_ft_control in value:
            return value.replace(old_ft_control, new_ft_control, 1)
    if new_ft_control in value:
        return value
    return f"{value}::{new_ft_control}"


def apply_ft_control_override(row, override):
    if not override:
        return dict(row)
    effective = dict(row)
    old_ft_controls = [
        effective.get("ft_control_variant"),
        normalized_ft_control_variant(effective),
    ]
    effective.update(override)
    effective = normalized_training_policy_row(effective)
    new_ft_control = effective.get("ft_control_variant")
    for key in ["config", "selected_config", "run_config_key"]:
        if key in effective:
            effective[key] = replace_or_append_ft_control(
                effective.get(key),
                old_ft_controls,
                new_ft_control,
            )
    return effective


def ft_control_flag_fragment(row):
    if row.get("ft_control_variant") is None:
        return []
    flags = []
    ft_epochs = row.get("ft_epochs")
    if isinstance(ft_epochs, int) and not isinstance(ft_epochs, bool):
        flags.extend(["--ft-epochs-list", str(ft_epochs)])
    target_min_loss_delta = row.get("target_min_loss_delta_policy")
    if is_numeric_value(target_min_loss_delta):
        flags.extend(["--target-min-loss-deltas", f"{float(target_min_loss_delta):g}"])
    patience = row.get("early_stopping_patience")
    if patience is None:
        flags.extend(["--patiences", "none"])
    elif isinstance(patience, int) and not isinstance(patience, bool):
        flags.extend(["--patiences", str(patience)])
    min_delta = row.get("early_stopping_min_delta")
    if is_numeric_value(min_delta):
        flags.extend(["--min-deltas", f"{float(min_delta):g}"])
    lr_decay_patience = row.get("lr_decay_patience")
    if lr_decay_patience is None:
        flags.extend(["--lr-decay-patiences", "none"])
    elif isinstance(lr_decay_patience, int) and not isinstance(lr_decay_patience, bool):
        flags.extend(["--lr-decay-patiences", str(lr_decay_patience)])
    lr_decay_factor = row.get("lr_decay_factor")
    if is_numeric_value(lr_decay_factor):
        flags.extend(["--lr-decay-factors", f"{float(lr_decay_factor):g}"])
    lr_decay_min_delta = row.get("lr_decay_min_delta")
    if is_numeric_value(lr_decay_min_delta):
        flags.extend(["--lr-decay-min-deltas", f"{float(lr_decay_min_delta):g}"])
    return flags


def build_profile_command(
    row,
    *,
    source_paths,
    cases=None,
    configs=None,
    output_dir=DEFAULT_OUTPUT_DIR,
    output_prefix=DEFAULT_OUTPUT_PREFIX,
    python_executable=sys.executable,
    sweep_script=Path(__file__).with_name("byte_lm_mlp_lora_sweep.py"),
    case_jsonls=None,
    lora_config_jsonls=None,
    extra_args=None,
    aggregate_gates=True,
    min_aggregate_epoch_wgpu_hit_rate=None,
    max_aggregate_epoch_wgpu_runtime_fallback_rate=None,
    max_aggregate_epoch_wgpu_component_fallback_rate=None,
    ft_control_override=None,
):
    effective_row = apply_ft_control_override(row, ft_control_override)
    source = selected_source_label(effective_row)
    if source not in source_paths:
        raise ValueError(
            f"profile {effective_row.get('source_profile')} missing source path for {source}"
        )
    cases = selected_cases(effective_row, cases)
    configs = selected_configs(effective_row, configs)
    run_config_key = normalized_profile_policy_key(effective_row, configs)
    policy_row = normalized_training_policy_row(effective_row)
    selected_config = (
        run_config_key or effective_row.get("selected_config") or effective_row.get("config")
    )
    jsonl_path, aggregate_path = profile_output_paths(
        effective_row,
        output_dir,
        output_prefix,
        configs,
    )
    command = [
        str(python_executable),
        str(sweep_script),
        "--hf-state-dict",
        str(source_paths[source]),
    ]
    command.extend(profile_flag_fragment(row))
    if ft_control_override or effective_row.get("promotion_run_key") is not None:
        command.extend(ft_control_flag_fragment(policy_row))
    for case_jsonl in case_jsonls or []:
        command.extend(["--case-jsonl", str(case_jsonl)])
    for lora_config_jsonl in lora_config_jsonls or []:
        command.extend(["--lora-config-jsonl", str(lora_config_jsonl)])
    for case in cases:
        command.extend(["--case", case])
    for config in configs:
        command.extend(["--config", config])
    if aggregate_gates:
        command.extend(["--min-aggregate-cases", str(len(cases))])
        for case in cases:
            command.extend(["--require-aggregate-case", case])
        command.extend(
            [
                "--min-aggregate-accepted-rate",
                "1.0",
                "--min-aggregate-movement-ok-rate",
                "1.0",
            ]
        )
        if min_aggregate_epoch_wgpu_hit_rate is not None:
            command.extend(
                [
                    "--min-aggregate-epoch-wgpu-hit-rate",
                    f"{float(min_aggregate_epoch_wgpu_hit_rate):g}",
                ]
            )
        if max_aggregate_epoch_wgpu_runtime_fallback_rate is not None:
            command.extend(
                [
                    "--max-aggregate-epoch-wgpu-runtime-fallback-rate",
                    f"{float(max_aggregate_epoch_wgpu_runtime_fallback_rate):g}",
                ]
            )
        if max_aggregate_epoch_wgpu_component_fallback_rate is not None:
            command.extend(
                [
                    "--max-aggregate-epoch-wgpu-component-fallback-rate",
                    f"{float(max_aggregate_epoch_wgpu_component_fallback_rate):g}",
                ]
            )
    command.extend(["--jsonl", str(jsonl_path), "--aggregate-jsonl", str(aggregate_path)])
    command.extend(extra_args or [])
    return {
        "row_type": "checkpoint_source_profile_command",
        "source_profile": effective_row.get("source_profile"),
        "selected_source": source,
        "selected_config": selected_config,
        "run_config_key": run_config_key,
        "checkpoint_source_gain": effective_row.get("checkpoint_source_gain", 1.0),
        "adapter_weight_decay_variant": policy_row.get("adapter_weight_decay_variant"),
        "adapter_weight_decay": policy_row.get("adapter_weight_decay"),
        "max_grad_norm_variant": policy_row.get("max_grad_norm_variant"),
        "max_grad_norm": policy_row.get("max_grad_norm"),
        "gradient_accumulation_steps_variant": policy_row.get(
            "gradient_accumulation_steps_variant"
        ),
        "gradient_accumulation_steps": policy_row.get("gradient_accumulation_steps"),
        "training_policy_key": training_policy_key(policy_row),
        "ft_control_variant": policy_row.get("ft_control_variant"),
        "ft_epochs": policy_row.get("ft_epochs"),
        "target_min_loss_delta_policy": policy_row.get("target_min_loss_delta_policy"),
        "early_stopping_patience": policy_row.get("early_stopping_patience"),
        "early_stopping_min_delta": policy_row.get("early_stopping_min_delta"),
        "lr_decay_patience": policy_row.get("lr_decay_patience"),
        "lr_decay_factor": policy_row.get("lr_decay_factor"),
        "lr_decay_min_delta": policy_row.get("lr_decay_min_delta"),
        "winner_metric": effective_row.get("winner_metric"),
        "winner_value": effective_row.get("winner_value"),
        "promotion_run_key": effective_row.get("promotion_run_key"),
        "promotion_rank": effective_row.get("promotion_rank"),
        "promotion_metric": effective_row.get("promotion_metric"),
        "promotion_value": effective_row.get("promotion_value"),
        "promotion_ready": effective_row.get("promotion_ready"),
        "promotion_ready_top_k": effective_row.get("promotion_ready_top_k"),
        "promotion_ready_within": effective_row.get("promotion_ready_within"),
        "promotion_ready_floor_passed": effective_row.get("promotion_ready_floor_passed"),
        "promotion_ready_floor_failures": effective_row.get("promotion_ready_floor_failures"),
        "promotion_ready_min_target_retention_ratio": effective_row.get(
            "promotion_ready_min_target_retention_ratio"
        ),
        "promotion_ready_min_accepted_rate": effective_row.get(
            "promotion_ready_min_accepted_rate"
        ),
        "promotion_ready_min_movement_ok_rate": effective_row.get(
            "promotion_ready_min_movement_ok_rate"
        ),
        "promotion_ready_min_retention_accuracy_margin": effective_row.get(
            "promotion_ready_min_retention_accuracy_margin"
        ),
        "promotion_ready_min_retention_perplexity_margin": effective_row.get(
            "promotion_ready_min_retention_perplexity_margin"
        ),
        "promotion_ready_min_epoch_wgpu_hit_rate": effective_row.get(
            "promotion_ready_min_epoch_wgpu_hit_rate"
        ),
        "promotion_ready_max_epoch_wgpu_runtime_fallback_rate": effective_row.get(
            "promotion_ready_max_epoch_wgpu_runtime_fallback_rate"
        ),
        "promotion_ready_max_epoch_wgpu_component_fallback_rate": effective_row.get(
            "promotion_ready_max_epoch_wgpu_component_fallback_rate"
        ),
        "promotion_ready_max_input_promotion_metric_regression": effective_row.get(
            "promotion_ready_max_input_promotion_metric_regression"
        ),
        "promotion_ready_require_guard_counts_available": effective_row.get(
            "promotion_ready_require_guard_counts_available"
        ),
        "promotion_ready_min_guard_acceptance_rate_mean": effective_row.get(
            "promotion_ready_min_guard_acceptance_rate_mean"
        ),
        "promotion_ready_max_guard_retention_rejected_epochs_mean": effective_row.get(
            "promotion_ready_max_guard_retention_rejected_epochs_mean"
        ),
        "promotion_ready_max_guard_target_stale_epochs_mean": effective_row.get(
            "promotion_ready_max_guard_target_stale_epochs_mean"
        ),
        "promotion_ready_max_guard_retention_rejected_rate_mean": effective_row.get(
            "promotion_ready_max_guard_retention_rejected_rate_mean"
        ),
        "promotion_ready_max_guard_target_stale_rate_mean": effective_row.get(
            "promotion_ready_max_guard_target_stale_rate_mean"
        ),
        "cases": ",".join(cases),
        "case_jsonls": ",".join(str(path) for path in case_jsonls or []),
        "lora_config_jsonls": ",".join(
            str(path) for path in lora_config_jsonls or []
        ),
        "configs": ",".join(configs),
        "jsonl": str(jsonl_path),
        "aggregate_jsonl": str(aggregate_path),
        "command": command,
        "shell": shlex.join(command),
    }


def command_rows_for_profiles(
    rows,
    *,
    source_paths,
    profiles=None,
    cases=None,
    configs=None,
    output_dir=DEFAULT_OUTPUT_DIR,
    output_prefix=DEFAULT_OUTPUT_PREFIX,
    python_executable=sys.executable,
    sweep_script=Path(__file__).with_name("byte_lm_mlp_lora_sweep.py"),
    case_jsonls=None,
    lora_config_jsonls=None,
    extra_args=None,
    aggregate_gates=True,
    min_aggregate_epoch_wgpu_hit_rate=None,
    max_aggregate_epoch_wgpu_runtime_fallback_rate=None,
    max_aggregate_epoch_wgpu_component_fallback_rate=None,
    promotion_rows=None,
    promotion_ready_only=True,
    ft_control_override=None,
):
    selected = (
        promoted_profile_rows(
            rows,
            promotion_rows,
            profiles,
            ready_only=promotion_ready_only,
        )
        if promotion_rows is not None
        else profile_rows(rows, profiles)
    )
    return [
        build_profile_command(
            row,
            source_paths=source_paths,
            cases=cases,
            configs=configs,
            output_dir=output_dir,
            output_prefix=output_prefix,
            python_executable=python_executable,
            sweep_script=sweep_script,
            case_jsonls=case_jsonls,
            lora_config_jsonls=lora_config_jsonls,
            extra_args=extra_args,
            aggregate_gates=aggregate_gates,
            min_aggregate_epoch_wgpu_hit_rate=min_aggregate_epoch_wgpu_hit_rate,
            max_aggregate_epoch_wgpu_runtime_fallback_rate=(
                max_aggregate_epoch_wgpu_runtime_fallback_rate
            ),
            max_aggregate_epoch_wgpu_component_fallback_rate=(
                max_aggregate_epoch_wgpu_component_fallback_rate
            ),
            ft_control_override=ft_control_override,
        )
        for row in selected
    ]


def print_command_rows(rows):
    for row in rows:
        promotion_label = ""
        if row.get("promotion_rank") is not None:
            promotion_label = (
                f"promotion_rank={row.get('promotion_rank')} "
                f"promotion_metric={row.get('promotion_metric')} "
                f"promotion_value={optional_numeric_label(row.get('promotion_value'))} "
                f"promotion_ready_top_k={row.get('promotion_ready_top_k')} "
                f"promotion_ready_within={optional_numeric_label(row.get('promotion_ready_within'))} "
                f"promotion_ready_floor_passed={row.get('promotion_ready_floor_passed')} "
                f"promotion_ready_min_epoch_wgpu_hit_rate={optional_numeric_label(row.get('promotion_ready_min_epoch_wgpu_hit_rate'))} "
                f"promotion_ready_max_epoch_wgpu_runtime_fallback_rate={optional_numeric_label(row.get('promotion_ready_max_epoch_wgpu_runtime_fallback_rate'))} "
                f"promotion_ready_max_epoch_wgpu_component_fallback_rate={optional_numeric_label(row.get('promotion_ready_max_epoch_wgpu_component_fallback_rate'))} "
                f"promotion_ready_require_guard_counts_available={row.get('promotion_ready_require_guard_counts_available')} "
                f"promotion_ready_min_guard_acceptance_rate_mean={optional_numeric_label(row.get('promotion_ready_min_guard_acceptance_rate_mean'))} "
                f"promotion_ready_max_guard_retention_rejected_epochs_mean={optional_numeric_label(row.get('promotion_ready_max_guard_retention_rejected_epochs_mean'))} "
                f"promotion_ready_max_guard_target_stale_epochs_mean={optional_numeric_label(row.get('promotion_ready_max_guard_target_stale_epochs_mean'))} "
                f"promotion_ready_max_guard_retention_rejected_rate_mean={optional_numeric_label(row.get('promotion_ready_max_guard_retention_rejected_rate_mean'))} "
                f"promotion_ready_max_guard_target_stale_rate_mean={optional_numeric_label(row.get('promotion_ready_max_guard_target_stale_rate_mean'))} "
            )
        print(
            f"profile_command profile={row['source_profile']} "
            f"source={row['selected_source']} "
            f"checkpoint_source_gain={row['checkpoint_source_gain']} "
            f"adapter_weight_decay={row.get('adapter_weight_decay')} "
            f"max_grad_norm={row.get('max_grad_norm')} "
            f"gradient_accumulation_steps={row.get('gradient_accumulation_steps')} "
            f"training_policy_key={row.get('training_policy_key')} "
            f"ft_epochs={row.get('ft_epochs')} "
            f"target_min_loss_delta={row.get('target_min_loss_delta_policy')} "
            f"patience={row.get('early_stopping_patience')} "
            f"lr_decay_patience={row.get('lr_decay_patience')} "
            f"{promotion_label}"
            f"jsonl={row['jsonl']} "
            f"aggregate_jsonl={row['aggregate_jsonl']} "
            f"shell={row['shell']}"
        )


def command_event_row(
    command_row,
    *,
    index,
    status,
    returncode=None,
    elapsed_seconds=None,
    error=None,
):
    row = {
        "row_type": "checkpoint_source_profile_command_event",
        "source_profile": command_row.get("source_profile"),
        "selected_source": command_row.get("selected_source"),
        "selected_config": command_row.get("selected_config"),
        "run_config_key": command_row.get("run_config_key"),
        "command_index": index,
        "status": status,
        "returncode": returncode,
        "elapsed_seconds": elapsed_seconds,
        "profile_command_jsonl": command_row.get("jsonl"),
        "profile_command_aggregate_jsonl": command_row.get("aggregate_jsonl"),
        "profile_command_shell": command_row.get("shell"),
    }
    for key in [
        "promotion_run_key",
        "promotion_rank",
        "promotion_metric",
        "promotion_value",
        "promotion_ready",
        "promotion_ready_top_k",
        "promotion_ready_within",
        "promotion_ready_floor_passed",
        "promotion_ready_floor_failures",
        "promotion_ready_min_epoch_wgpu_hit_rate",
        "promotion_ready_max_epoch_wgpu_runtime_fallback_rate",
        "promotion_ready_max_epoch_wgpu_component_fallback_rate",
        "promotion_ready_require_guard_counts_available",
        "promotion_ready_min_guard_acceptance_rate_mean",
        "promotion_ready_max_guard_retention_rejected_epochs_mean",
        "promotion_ready_max_guard_target_stale_epochs_mean",
        "promotion_ready_max_guard_retention_rejected_rate_mean",
        "promotion_ready_max_guard_target_stale_rate_mean",
    ]:
        if command_row.get(key) is not None:
            row[key] = command_row.get(key)
    if error is not None:
        row["error"] = str(error)
    return row


def print_command_event(row):
    print(
        f"profile_command_event profile={row.get('source_profile')} "
        f"status={row.get('status')} "
        f"returncode={row.get('returncode')} "
        f"elapsed_seconds={optional_numeric_label(row.get('elapsed_seconds'))} "
        f"aggregate_jsonl={row.get('profile_command_aggregate_jsonl')}"
    )


def run_command_rows(rows, *, events_jsonl=None):
    events = []
    for index, row in enumerate(rows, 1):
        started = time.monotonic()
        try:
            completed = subprocess.run(row["command"], check=False)
            elapsed = time.monotonic() - started
            status = "passed" if completed.returncode == 0 else "failed"
            event = command_event_row(
                row,
                index=index,
                status=status,
                returncode=completed.returncode,
                elapsed_seconds=elapsed,
            )
            events.append(event)
            print_command_event(event)
            if events_jsonl is not None:
                write_jsonl(events_jsonl, events)
                print(f"profile_run_events_jsonl={events_jsonl} rows={len(events)}")
            if completed.returncode != 0:
                raise subprocess.CalledProcessError(completed.returncode, row["command"])
        except Exception as exc:
            if not isinstance(exc, subprocess.CalledProcessError):
                elapsed = time.monotonic() - started
                event = command_event_row(
                    row,
                    index=index,
                    status="spawn_error",
                    elapsed_seconds=elapsed,
                    error=exc,
                )
                events.append(event)
                print_command_event(event)
                if events_jsonl is not None:
                    write_jsonl(events_jsonl, events)
                    print(f"profile_run_events_jsonl={events_jsonl} rows={len(events)}")
            raise
    return events


INPUT_PROMOTION_FIELDS = [
    ("promotion_run_key", "input_promotion_run_key"),
    ("promotion_rank", "input_promotion_rank"),
    ("promotion_metric", "input_promotion_metric"),
    ("promotion_value", "input_promotion_value"),
    ("promotion_ready", "input_promotion_ready"),
    ("promotion_ready_top_k", "input_promotion_ready_top_k"),
    ("promotion_ready_within", "input_promotion_ready_within"),
    ("promotion_ready_floor_passed", "input_promotion_ready_floor_passed"),
    ("promotion_ready_floor_failures", "input_promotion_ready_floor_failures"),
    (
        "promotion_ready_min_target_retention_ratio",
        "input_promotion_ready_min_target_retention_ratio",
    ),
    ("promotion_ready_min_accepted_rate", "input_promotion_ready_min_accepted_rate"),
    (
        "promotion_ready_min_movement_ok_rate",
        "input_promotion_ready_min_movement_ok_rate",
    ),
    (
        "promotion_ready_min_retention_accuracy_margin",
        "input_promotion_ready_min_retention_accuracy_margin",
    ),
    (
        "promotion_ready_min_retention_perplexity_margin",
        "input_promotion_ready_min_retention_perplexity_margin",
    ),
    (
        "promotion_ready_min_epoch_wgpu_hit_rate",
        "input_promotion_ready_min_epoch_wgpu_hit_rate",
    ),
    (
        "promotion_ready_max_epoch_wgpu_runtime_fallback_rate",
        "input_promotion_ready_max_epoch_wgpu_runtime_fallback_rate",
    ),
    (
        "promotion_ready_max_epoch_wgpu_component_fallback_rate",
        "input_promotion_ready_max_epoch_wgpu_component_fallback_rate",
    ),
    (
        "promotion_ready_max_input_promotion_metric_regression",
        "input_promotion_ready_max_input_promotion_metric_regression",
    ),
    (
        "promotion_ready_require_guard_counts_available",
        "input_promotion_ready_require_guard_counts_available",
    ),
    (
        "promotion_ready_min_guard_acceptance_rate_mean",
        "input_promotion_ready_min_guard_acceptance_rate_mean",
    ),
    (
        "promotion_ready_max_guard_retention_rejected_epochs_mean",
        "input_promotion_ready_max_guard_retention_rejected_epochs_mean",
    ),
    (
        "promotion_ready_max_guard_target_stale_epochs_mean",
        "input_promotion_ready_max_guard_target_stale_epochs_mean",
    ),
    (
        "promotion_ready_max_guard_retention_rejected_rate_mean",
        "input_promotion_ready_max_guard_retention_rejected_rate_mean",
    ),
    (
        "promotion_ready_max_guard_target_stale_rate_mean",
        "input_promotion_ready_max_guard_target_stale_rate_mean",
    ),
]


def input_promotion_fields(command_row):
    return {
        summary_key: command_row.get(command_key)
        for command_key, summary_key in INPUT_PROMOTION_FIELDS
        if command_row.get(command_key) is not None
    }


def input_promotion_metric_fields(row):
    if row.get("input_promotion_metric") is None:
        return {}
    metric, input_value, current_value, regression = input_promotion_metric_regression(row)
    return {
        "input_promotion_metric_current": current_value,
        "input_promotion_metric_delta": (
            None if current_value is None else current_value - input_value
        ),
        "input_promotion_metric_regression": regression,
    }


def resolved_training_policy_key(command_row, aggregate, aggregate_path):
    aggregate_key = aggregate.get("training_policy_key")
    command_key = command_row.get("training_policy_key")
    if aggregate_key is not None and command_key is not None and aggregate_key != command_key:
        aggregate_canonical = canonical_training_policy_key(aggregate)
        command_canonical = canonical_training_policy_key(command_row)
        if (
            aggregate_key in generated_training_policy_keys(aggregate)
            and command_key in generated_training_policy_keys(command_row)
            and aggregate_canonical == command_canonical
        ):
            return aggregate_canonical
        raise ValueError(
            f"{aggregate_path} training_policy_key mismatch for profile "
            f"{command_row.get('source_profile')}: aggregate={aggregate_key} "
            f"command={command_key}"
        )
    if aggregate_key is not None:
        if aggregate_key in generated_training_policy_keys(aggregate):
            return canonical_training_policy_key(aggregate)
        return aggregate_key
    if command_key is not None:
        if command_key in generated_training_policy_keys(command_row):
            return canonical_training_policy_key(command_row)
        return command_key
    return None


def profile_run_summary_rows(command_rows):
    summaries = []
    for command_row in command_rows:
        aggregate_path = Path(command_row["aggregate_jsonl"])
        if not aggregate_path.exists():
            raise FileNotFoundError(
                f"profile {command_row['source_profile']} aggregate output missing: {aggregate_path}"
            )
        aggregate_rows = load_jsonl(aggregate_path)
        if not aggregate_rows:
            raise ValueError(f"profile aggregate output is empty: {aggregate_path}")
        for aggregate in aggregate_rows:
            if aggregate.get("row_type") != "config_aggregate":
                raise ValueError(f"{aggregate_path} expected row_type='config_aggregate'")
            gap, ratio = target_retention_selectivity(aggregate)
            summary = dict(aggregate)
            attach_requested_wgpu_component_backend_summary(summary)
            summary.update(
                {
                    "row_type": "checkpoint_source_profile_run",
                    "aggregate_row_type": aggregate.get("row_type"),
                    "source_profile": command_row["source_profile"],
                    "selected_source": command_row["selected_source"],
                    "selected_config": command_row.get("selected_config"),
                    "run_config_key": command_row.get("run_config_key"),
                    "adapter_weight_decay_variant": aggregate.get(
                        "adapter_weight_decay_variant",
                        command_row.get("adapter_weight_decay_variant"),
                    ),
                    "adapter_weight_decay": aggregate.get(
                        "adapter_weight_decay",
                        command_row.get("adapter_weight_decay"),
                    ),
                    "max_grad_norm_variant": aggregate.get(
                        "max_grad_norm_variant",
                        command_row.get("max_grad_norm_variant"),
                    ),
                    "max_grad_norm": aggregate.get(
                        "max_grad_norm",
                        command_row.get("max_grad_norm"),
                    ),
                    "gradient_accumulation_steps_variant": aggregate.get(
                        "gradient_accumulation_steps_variant",
                        command_row.get("gradient_accumulation_steps_variant"),
                    ),
                    "gradient_accumulation_steps": aggregate.get(
                        "gradient_accumulation_steps",
                        command_row.get("gradient_accumulation_steps"),
                    ),
                    "training_policy_key": resolved_training_policy_key(
                        command_row,
                        aggregate,
                        aggregate_path,
                    ),
                    "ft_control_variant": policy_value(
                        aggregate,
                        command_row,
                        "ft_control_variant",
                    ),
                    "ft_epochs": policy_value(aggregate, command_row, "ft_epochs"),
                    "target_min_loss_delta_policy": policy_value(
                        aggregate,
                        command_row,
                        "target_min_loss_delta_policy",
                    ),
                    "early_stopping_patience": policy_value(
                        aggregate,
                        command_row,
                        "early_stopping_patience",
                    ),
                    "early_stopping_min_delta": policy_value(
                        aggregate,
                        command_row,
                        "early_stopping_min_delta",
                    ),
                    "lr_decay_patience": policy_value(
                        aggregate,
                        command_row,
                        "lr_decay_patience",
                    ),
                    "lr_decay_factor": policy_value(
                        aggregate,
                        command_row,
                        "lr_decay_factor",
                    ),
                    "lr_decay_min_delta": policy_value(
                        aggregate,
                        command_row,
                        "lr_decay_min_delta",
                    ),
                    "profile_winner_metric": command_row.get("winner_metric"),
                    "profile_winner_value": command_row.get("winner_value"),
                    "profile_command_jsonl": command_row["jsonl"],
                    "profile_command_aggregate_jsonl": command_row["aggregate_jsonl"],
                    "profile_command_shell": command_row["shell"],
                    "target_retention_gap_mean": gap,
                    "target_retention_ratio": ratio,
                }
            )
            summary.update(input_promotion_fields(command_row))
            summary.update(input_promotion_metric_fields(summary))
            summaries.append(summary)
    return summaries


def print_run_summary_rows(rows):
    for row in rows:
        ratio = row.get("target_retention_ratio")
        ratio_label = "none" if ratio is None else f"{float(ratio):.9f}"
        input_promotion_label = ""
        if row.get("input_promotion_rank") is not None:
            input_promotion_label = (
                f"input_promotion_rank={row.get('input_promotion_rank')} "
                f"input_promotion_metric={row.get('input_promotion_metric')} "
                f"input_promotion_value={optional_numeric_label(row.get('input_promotion_value'))} "
                f"input_promotion_metric_current={optional_numeric_label(row.get('input_promotion_metric_current'))} "
                f"input_promotion_metric_regression={optional_numeric_label(row.get('input_promotion_metric_regression'))} "
            )
        guard_label = ""
        if row.get("guard_epoch_counts_available_cases") is not None:
            guard_label = (
                f"guard_epoch_counts_available_cases={row.get('guard_epoch_counts_available_cases')} "
                f"guard_epoch_counts_available_all={row.get('guard_epoch_counts_available_all')} "
                f"guard_accepted_epochs_mean={optional_numeric_label(row.get('guard_accepted_epochs_mean'))} "
                f"guard_retention_rejected_epochs_mean={optional_numeric_label(row.get('guard_retention_rejected_epochs_mean'))} "
                f"guard_target_stale_epochs_mean={optional_numeric_label(row.get('guard_target_stale_epochs_mean'))} "
                f"guard_acceptance_rate_mean={optional_numeric_label(row.get('guard_acceptance_rate_mean'))} "
                f"guard_retention_rejected_rate_mean={optional_numeric_label(row.get('guard_retention_rejected_rate_mean'))} "
                f"guard_target_stale_rate_mean={optional_numeric_label(row.get('guard_target_stale_rate_mean'))} "
            )
        wgpu_component_label = ""
        if (
            row.get("tensor_backend_requested_wgpu_component_fallback_top")
            is not None
        ):
            wgpu_component_label = (
                "wgpu_component_fallback_top="
                f"{row.get('tensor_backend_requested_wgpu_component_fallback_top')} "
                "wgpu_component_hit_top="
                f"{row.get('tensor_backend_requested_wgpu_component_hit_top')} "
            )
        wgpu_epoch_label = ""
        if row.get("epoch_tensor_backend_requested_wgpu_hit_rate_mean") is not None:
            wgpu_epoch_label = (
                "wgpu_epoch_hit_rate_mean="
                f"{optional_numeric_label(row.get('epoch_tensor_backend_requested_wgpu_hit_rate_mean'))} "
                "wgpu_epoch_runtime_fallback_rate_mean="
                f"{optional_numeric_label(row.get('epoch_tensor_backend_requested_wgpu_runtime_fallback_rate_mean'))} "
                "wgpu_epoch_component_fallback_rate_mean="
                f"{optional_numeric_label(row.get('epoch_tensor_backend_requested_wgpu_component_fallback_rate_mean'))} "
            )
        print(
            f"profile_run profile={row['source_profile']} "
            f"source={row['selected_source']} "
            f"config={row.get('config')} "
            f"checkpoint_source_gain={row.get('checkpoint_source_gain', 1.0)} "
            f"adapter_weight_decay={row.get('adapter_weight_decay')} "
            f"max_grad_norm={row.get('max_grad_norm')} "
            f"gradient_accumulation_steps={row.get('gradient_accumulation_steps')} "
            f"cases={row.get('cases')} "
            f"accepted_rate={float(row.get('accepted_rate', 0.0)):.9f} "
            f"movement_ok_rate={float(row.get('movement_ok_rate', 0.0)):.9f} "
            f"target_loss_delta_mean={float(row['target_loss_delta_mean']):.9f} "
            f"retention_loss_delta_mean={float(row['retention_loss_delta_mean']):.9f} "
            f"target_retention_gap_mean={float(row['target_retention_gap_mean']):.9f} "
            f"{guard_label}"
            f"{wgpu_component_label}"
            f"{wgpu_epoch_label}"
            f"{input_promotion_label}"
            f"target_retention_ratio={ratio_label}"
        )


def run_summary_key(row):
    profile = row.get("source_profile")
    if not isinstance(profile, str) or not profile:
        raise ValueError("checkpoint_source_profile_run row missing source_profile")
    config = row.get("config") or row.get("selected_config")
    if isinstance(config, str) and config:
        return f"{profile}::{config}"
    run_config = row.get("run_config_key")
    if isinstance(run_config, str) and run_config:
        return f"{profile}::{run_config}"
    return profile


def run_summary_rows_by_profile(rows, label):
    by_profile = {}
    for row in rows:
        if row.get("row_type") != "checkpoint_source_profile_run":
            raise ValueError(f"{label} expected row_type='checkpoint_source_profile_run'")
        profile = run_summary_key(row)
        if profile in by_profile:
            raise ValueError(f"{label} contains duplicate profile run key: {profile}")
        by_profile[profile] = normalize_profile_run_selectivity(row)
    if not by_profile:
        raise ValueError(f"{label} contains no checkpoint_source_profile_run rows")
    return by_profile


PROMOTION_COPY_FIELDS = [
    "source_profile",
    "selected_source",
    "checkpoint_source_label",
    "checkpoint_source_gain",
    "config",
    "selected_config",
    "run_config_key",
    "training_policy_key",
    "cases",
    "case_labels",
    "accepted_rate",
    "movement_ok_rate",
    "guard_epoch_counts_available_cases",
    "guard_epoch_counts_available_all",
    "guard_accepted_epochs_total",
    "guard_accepted_epochs_mean",
    "guard_accepted_epochs_max",
    "guard_retention_rejected_epochs_total",
    "guard_retention_rejected_epochs_mean",
    "guard_retention_rejected_epochs_max",
    "guard_target_stale_epochs_total",
    "guard_target_stale_epochs_mean",
    "guard_target_stale_epochs_max",
    "guard_acceptance_rate_mean",
    "guard_acceptance_rate_min",
    "guard_retention_rejected_rate_mean",
    "guard_retention_rejected_rate_max",
    "guard_target_stale_rate_mean",
    "guard_target_stale_rate_max",
    "epoch_tensor_backend_requested_wgpu_hits_total",
    "epoch_tensor_backend_requested_wgpu_runtime_fallbacks_total",
    "epoch_tensor_backend_requested_wgpu_total_sum",
    "epoch_tensor_backend_requested_wgpu_hit_rate_mean",
    "epoch_tensor_backend_requested_wgpu_runtime_fallback_rate_mean",
    "epoch_tensor_backend_requested_wgpu_component_hits_total",
    "epoch_tensor_backend_requested_wgpu_component_fallbacks_total",
    "epoch_tensor_backend_requested_wgpu_component_total_sum",
    "epoch_tensor_backend_requested_wgpu_component_hit_rate_mean",
    "epoch_tensor_backend_requested_wgpu_component_fallback_rate_mean",
    "target_loss_delta_mean",
    "retention_loss_delta_mean",
    "target_retention_gap_mean",
    "target_retention_ratio",
    "retention_accuracy_margin_min",
    "retention_perplexity_margin_min",
    "adapter_weight_decay_variant",
    "adapter_weight_decay",
    "max_grad_norm_variant",
    "max_grad_norm",
    "gradient_accumulation_steps_variant",
    "gradient_accumulation_steps",
    "ft_control_variant",
    "ft_epochs",
    "target_min_loss_delta_policy",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "lr_decay_patience",
    "lr_decay_factor",
    "lr_decay_min_delta",
    "profile_winner_metric",
    "profile_winner_value",
    "profile_command_jsonl",
    "profile_command_aggregate_jsonl",
    "profile_command_shell",
    "input_promotion_run_key",
    "input_promotion_rank",
    "input_promotion_metric",
    "input_promotion_value",
    "input_promotion_metric_current",
    "input_promotion_metric_delta",
    "input_promotion_metric_regression",
    "input_promotion_ready",
    "input_promotion_ready_top_k",
    "input_promotion_ready_within",
    "input_promotion_ready_floor_passed",
    "input_promotion_ready_floor_failures",
    "input_promotion_ready_min_target_retention_ratio",
    "input_promotion_ready_min_accepted_rate",
    "input_promotion_ready_min_movement_ok_rate",
    "input_promotion_ready_min_retention_accuracy_margin",
    "input_promotion_ready_min_retention_perplexity_margin",
    "input_promotion_ready_min_epoch_wgpu_hit_rate",
    "input_promotion_ready_max_epoch_wgpu_runtime_fallback_rate",
    "input_promotion_ready_max_epoch_wgpu_component_fallback_rate",
    "input_promotion_ready_max_input_promotion_metric_regression",
    "input_promotion_ready_require_guard_counts_available",
    "input_promotion_ready_min_guard_acceptance_rate_mean",
    "input_promotion_ready_max_guard_retention_rejected_epochs_mean",
    "input_promotion_ready_max_guard_target_stale_epochs_mean",
    "input_promotion_ready_max_guard_retention_rejected_rate_mean",
    "input_promotion_ready_max_guard_target_stale_rate_mean",
]


def is_promotion_ready(rank, value, best_value, *, ready_top_k=1, ready_within=None):
    if value == best_value:
        return True
    if rank <= ready_top_k:
        return True
    return ready_within is not None and best_value - value <= ready_within


PROMOTION_READY_FLOORS = [
    (
        "target_retention_ratio",
        "promotion_ready_min_target_retention_ratio",
        "target_retention_ratio",
    ),
    ("accepted_rate", "promotion_ready_min_accepted_rate", "accepted_rate"),
    (
        "movement_ok_rate",
        "promotion_ready_min_movement_ok_rate",
        "movement_ok_rate",
    ),
    (
        "retention_accuracy_margin_min",
        "promotion_ready_min_retention_accuracy_margin",
        "retention_accuracy_margin_min",
    ),
    (
        "retention_perplexity_margin_min",
        "promotion_ready_min_retention_perplexity_margin",
        "retention_perplexity_margin_min",
    ),
    (
        "epoch_tensor_backend_requested_wgpu_hit_rate_mean",
        "promotion_ready_min_epoch_wgpu_hit_rate",
        "epoch_wgpu_hit_rate_mean",
    ),
    (
        "guard_acceptance_rate_mean",
        "promotion_ready_min_guard_acceptance_rate_mean",
        "guard_acceptance_rate_mean",
    ),
]


PROMOTION_READY_CEILINGS = [
    (
        "input_promotion_metric_regression",
        "promotion_ready_max_input_promotion_metric_regression",
        "input_promotion_metric_regression",
    ),
    (
        "epoch_tensor_backend_requested_wgpu_runtime_fallback_rate_mean",
        "promotion_ready_max_epoch_wgpu_runtime_fallback_rate",
        "epoch_wgpu_runtime_fallback_rate_mean",
    ),
    (
        "epoch_tensor_backend_requested_wgpu_component_fallback_rate_mean",
        "promotion_ready_max_epoch_wgpu_component_fallback_rate",
        "epoch_wgpu_component_fallback_rate_mean",
    ),
    (
        "guard_retention_rejected_epochs_mean",
        "promotion_ready_max_guard_retention_rejected_epochs_mean",
        "guard_retention_rejected_epochs_mean",
    ),
    (
        "guard_target_stale_epochs_mean",
        "promotion_ready_max_guard_target_stale_epochs_mean",
        "guard_target_stale_epochs_mean",
    ),
    (
        "guard_retention_rejected_rate_mean",
        "promotion_ready_max_guard_retention_rejected_rate_mean",
        "guard_retention_rejected_rate_mean",
    ),
    (
        "guard_target_stale_rate_mean",
        "promotion_ready_max_guard_target_stale_rate_mean",
        "guard_target_stale_rate_mean",
    ),
]


def promotion_ready_floor_settings(
    *,
    min_target_retention_ratio=None,
    min_accepted_rate=None,
    min_movement_ok_rate=None,
    min_retention_accuracy_margin=None,
    min_retention_perplexity_margin=None,
    min_epoch_wgpu_hit_rate=None,
    min_guard_acceptance_rate_mean=None,
):
    return {
        "promotion_ready_min_target_retention_ratio": min_target_retention_ratio,
        "promotion_ready_min_accepted_rate": min_accepted_rate,
        "promotion_ready_min_movement_ok_rate": min_movement_ok_rate,
        "promotion_ready_min_retention_accuracy_margin": min_retention_accuracy_margin,
        "promotion_ready_min_retention_perplexity_margin": min_retention_perplexity_margin,
        "promotion_ready_min_epoch_wgpu_hit_rate": min_epoch_wgpu_hit_rate,
        "promotion_ready_min_guard_acceptance_rate_mean": min_guard_acceptance_rate_mean,
    }


def promotion_ready_ceiling_settings(
    *,
    max_input_promotion_metric_regression=None,
    max_epoch_wgpu_runtime_fallback_rate=None,
    max_epoch_wgpu_component_fallback_rate=None,
    max_guard_retention_rejected_epochs_mean=None,
    max_guard_target_stale_epochs_mean=None,
    max_guard_retention_rejected_rate_mean=None,
    max_guard_target_stale_rate_mean=None,
):
    return {
        "promotion_ready_max_input_promotion_metric_regression": (
            max_input_promotion_metric_regression
        ),
        "promotion_ready_max_epoch_wgpu_runtime_fallback_rate": (
            max_epoch_wgpu_runtime_fallback_rate
        ),
        "promotion_ready_max_epoch_wgpu_component_fallback_rate": (
            max_epoch_wgpu_component_fallback_rate
        ),
        "promotion_ready_max_guard_retention_rejected_epochs_mean": (
            max_guard_retention_rejected_epochs_mean
        ),
        "promotion_ready_max_guard_target_stale_epochs_mean": (
            max_guard_target_stale_epochs_mean
        ),
        "promotion_ready_max_guard_retention_rejected_rate_mean": (
            max_guard_retention_rejected_rate_mean
        ),
        "promotion_ready_max_guard_target_stale_rate_mean": (
            max_guard_target_stale_rate_mean
        ),
    }


def promotion_ready_floor_failures(row, floor_settings, ceiling_settings=None):
    failures = []
    for row_key, field_key, label in PROMOTION_READY_FLOORS:
        floor = floor_settings.get(field_key)
        if floor is None:
            continue
        value = row.get(row_key)
        if not is_numeric_value(value):
            failures.append(f"{label}=unavailable")
        elif float(value) < float(floor):
            failures.append(f"{label}<{float(floor):.9f}")
    for row_key, field_key, label in PROMOTION_READY_CEILINGS:
        ceiling = (ceiling_settings or {}).get(field_key)
        if ceiling is None:
            continue
        value = row.get(row_key)
        if not is_numeric_value(value):
            failures.append(f"{label}=unavailable")
        elif float(value) > float(ceiling):
            failures.append(f"{label}>{float(ceiling):.9f}")
    return failures


def profile_run_promotion_rows(
    rows,
    *,
    promotion_metric=DEFAULT_PROMOTION_METRIC,
    ready_top_k=1,
    ready_within=None,
    ready_min_target_retention_ratio=None,
    ready_min_accepted_rate=None,
    ready_min_movement_ok_rate=None,
    ready_min_retention_accuracy_margin=None,
    ready_min_retention_perplexity_margin=None,
    ready_min_epoch_wgpu_hit_rate=None,
    ready_min_guard_acceptance_rate_mean=None,
    ready_max_input_promotion_metric_regression=None,
    ready_max_epoch_wgpu_runtime_fallback_rate=None,
    ready_max_epoch_wgpu_component_fallback_rate=None,
    ready_require_guard_counts_available=False,
    ready_max_guard_retention_rejected_epochs_mean=None,
    ready_max_guard_target_stale_epochs_mean=None,
    ready_max_guard_retention_rejected_rate_mean=None,
    ready_max_guard_target_stale_rate_mean=None,
):
    if promotion_metric not in PROMOTION_METRICS:
        raise ValueError(f"unknown promotion metric: {promotion_metric}")
    if ready_top_k < 1:
        raise ValueError("ready_top_k must be at least 1")
    if ready_within is not None and ready_within < 0.0:
        raise ValueError("ready_within must be non-negative")
    floor_settings = promotion_ready_floor_settings(
        min_target_retention_ratio=ready_min_target_retention_ratio,
        min_accepted_rate=ready_min_accepted_rate,
        min_movement_ok_rate=ready_min_movement_ok_rate,
        min_retention_accuracy_margin=ready_min_retention_accuracy_margin,
        min_retention_perplexity_margin=ready_min_retention_perplexity_margin,
        min_epoch_wgpu_hit_rate=ready_min_epoch_wgpu_hit_rate,
        min_guard_acceptance_rate_mean=ready_min_guard_acceptance_rate_mean,
    )
    ceiling_settings = promotion_ready_ceiling_settings(
        max_input_promotion_metric_regression=(
            ready_max_input_promotion_metric_regression
        ),
        max_epoch_wgpu_runtime_fallback_rate=(
            ready_max_epoch_wgpu_runtime_fallback_rate
        ),
        max_epoch_wgpu_component_fallback_rate=(
            ready_max_epoch_wgpu_component_fallback_rate
        ),
        max_guard_retention_rejected_epochs_mean=(
            ready_max_guard_retention_rejected_epochs_mean
        ),
        max_guard_target_stale_epochs_mean=ready_max_guard_target_stale_epochs_mean,
        max_guard_retention_rejected_rate_mean=(
            ready_max_guard_retention_rejected_rate_mean
        ),
        max_guard_target_stale_rate_mean=ready_max_guard_target_stale_rate_mean,
    )
    for key, floor in floor_settings.items():
        if floor is not None and floor < 0.0:
            raise ValueError(f"{key} must be non-negative")
        if floor is not None and key.endswith("_rate") and floor > 1.0:
            raise ValueError(f"{key} must be at most 1.0")
    for key, ceiling in ceiling_settings.items():
        if ceiling is not None and ceiling < 0.0:
            raise ValueError(f"{key} must be non-negative")
        if (
            ceiling is not None
            and ("_rate_" in key or key.endswith("_rate"))
            and ceiling > 1.0
        ):
            raise ValueError(f"{key} must be at most 1.0")
    floors_requested = any(value is not None for value in floor_settings.values())
    ceilings_requested = any(value is not None for value in ceiling_settings.values())
    guard_counts_requested = bool(ready_require_guard_counts_available)
    current = run_summary_rows_by_profile(rows, "current")
    scored = [
        (run_key, row, optional_numeric_value(row, promotion_metric))
        for run_key, row in current.items()
    ]
    ranked = sorted(
        scored,
        key=lambda item: (
            item[2] is None,
            -item[2] if item[2] is not None else 0.0,
            item[0],
        ),
    )
    best_value = next((value for _run_key, _row, value in ranked if value is not None), None)
    promotions = []
    for rank, (run_key, row, value) in enumerate(ranked, 1):
        rank_ready = (
            best_value is not None
            and value is not None
            and is_promotion_ready(
                rank,
                value,
                best_value,
                ready_top_k=ready_top_k,
                ready_within=ready_within,
            )
        )
        floor_failures = promotion_ready_floor_failures(
            row,
            floor_settings,
            ceiling_settings,
        )
        if (
            ready_require_guard_counts_available
            and row.get("guard_epoch_counts_available_all") is not True
        ):
            floor_failures.append("guard_epoch_counts_available_all=false")
        if value is None:
            floor_failures.insert(0, f"{promotion_metric}=unavailable")
        promotion = {
            "row_type": "checkpoint_source_profile_promotion",
            "promotion_rank": rank,
            "promotion_metric": promotion_metric,
            "promotion_value": value,
            "promotion_ready": rank_ready and not floor_failures,
            "promotion_ready_top_k": ready_top_k,
            "promotion_ready_within": ready_within,
            "run_key": run_key,
        }
        if floors_requested or ceilings_requested or guard_counts_requested or value is None:
            promotion["promotion_ready_floor_passed"] = not floor_failures
            promotion["promotion_ready_floor_failures"] = floor_failures
            if ready_require_guard_counts_available:
                promotion["promotion_ready_require_guard_counts_available"] = True
            for key, floor in floor_settings.items():
                if floor is not None:
                    promotion[key] = floor
            for key, ceiling in ceiling_settings.items():
                if ceiling is not None:
                    promotion[key] = ceiling
        for key in PROMOTION_COPY_FIELDS:
            if key in row:
                promotion[key] = row.get(key)
        promotions.append(promotion)
    return promotions


def print_promotion_rows(rows):
    for row in rows:
        floor_label = ""
        if row.get("promotion_ready_floor_passed") is not None:
            failures = row.get("promotion_ready_floor_failures") or []
            floor_label = (
                f"ready_floor_passed={row.get('promotion_ready_floor_passed')} "
                f"ready_floor_failures={','.join(failures) or 'none'} "
            )
        print(
            f"profile_promotion rank={row['promotion_rank']} "
            f"ready={row['promotion_ready']} "
            f"profile={row.get('source_profile')} "
            f"run_key={row.get('run_key')} "
            f"metric={row['promotion_metric']} "
            f"value={optional_numeric_label(row.get('promotion_value'))} "
            f"ready_top_k={row.get('promotion_ready_top_k')} "
            f"ready_within={optional_numeric_label(row.get('promotion_ready_within'))} "
            f"{floor_label}"
            f"source={row.get('selected_source')} "
            f"config={row.get('config')} "
            f"cases={row.get('cases')} "
            f"accepted_rate={optional_numeric_label(row.get('accepted_rate'))} "
            f"movement_ok_rate={optional_numeric_label(row.get('movement_ok_rate'))} "
            f"target_retention_ratio={optional_numeric_label(row.get('target_retention_ratio'))}"
        )


def check_promotion_ready_gates(
    rows,
    *,
    min_ready_count=None,
    min_ready_rate=None,
    min_ready_guard_policy_count=None,
    require_ready_guard_policy=False,
):
    ready_rows = [row for row in rows if row.get("promotion_ready") is True]
    ready_count = len(ready_rows)
    ready_guard_policy_count = sum(
        1 for row in ready_rows if promotion_guard_policy_requested(row)
    )
    total = len(rows)
    ready_rate = ready_count / total if total else 0.0
    ready_guard_policy_rate = (
        ready_guard_policy_count / ready_count if ready_count else 0.0
    )
    passed = (
        (min_ready_count is None or ready_count >= min_ready_count)
        and (min_ready_rate is None or ready_rate >= min_ready_rate)
        and (
            min_ready_guard_policy_count is None
            or ready_guard_policy_count >= min_ready_guard_policy_count
        )
        and (
            not require_ready_guard_policy
            or ready_guard_policy_count == ready_count
        )
    )
    print(
        f"profile_promotion_gate rows={total} "
        f"ready_count={ready_count} "
        f"ready_rate={ready_rate:.9f} "
        f"ready_guard_policy_count={ready_guard_policy_count} "
        f"ready_guard_policy_rate={ready_guard_policy_rate:.9f} "
        f"min_ready_count={optional_numeric_label(min_ready_count)} "
        f"min_ready_rate={optional_numeric_label(min_ready_rate)} "
        f"min_ready_guard_policy_count={optional_numeric_label(min_ready_guard_policy_count)} "
        f"require_ready_guard_policy={require_ready_guard_policy} "
        f"passed={passed}"
    )
    failures = []
    if min_ready_count is not None and ready_count < min_ready_count:
        failures.append(f"ready_count {ready_count} below floor {min_ready_count}")
    if min_ready_rate is not None and ready_rate < min_ready_rate:
        failures.append(f"ready_rate {ready_rate:.9f} below floor {min_ready_rate:.9f}")
    if (
        min_ready_guard_policy_count is not None
        and ready_guard_policy_count < min_ready_guard_policy_count
    ):
        failures.append(
            f"ready_guard_policy_count {ready_guard_policy_count} below floor "
            f"{min_ready_guard_policy_count}"
        )
    if require_ready_guard_policy and ready_guard_policy_count != ready_count:
        failures.append(
            f"ready_guard_policy_count {ready_guard_policy_count} below ready_count "
            f"{ready_count}"
        )
    if failures:
        raise RuntimeError("profile promotion ready gate failed: " + "; ".join(failures))
    return ready_count


def max_regression(before, now, key):
    return max(0.0, numeric_value(before, key) - numeric_value(now, key))


def max_increase_regression(before, now, key):
    return max(0.0, numeric_value(now, key) - numeric_value(before, key))


def optional_max_regression(before, now, key):
    before_value = optional_numeric_value(before, key)
    now_value = optional_numeric_value(now, key)
    if before_value is None or now_value is None:
        return None
    return max(0.0, before_value - now_value)


def input_promotion_metric_regression(row):
    metric = row.get("input_promotion_metric")
    if not isinstance(metric, str) or not metric:
        raise ValueError("input_promotion_metric unavailable")
    if metric not in PROMOTION_METRICS:
        raise ValueError(f"unsupported input_promotion_metric {metric!r}")
    input_value = numeric_value(row, "input_promotion_value")
    current_value = optional_numeric_value(row, metric)
    regression = None if current_value is None else max(0.0, input_value - current_value)
    return metric, input_value, current_value, regression


def check_profile_run_gates(
    rows,
    *,
    min_target_retention_ratio=None,
    min_accepted_rate=None,
    min_movement_ok_rate=None,
    min_retention_accuracy_margin=None,
    min_retention_perplexity_margin=None,
    max_input_promotion_metric_regression=None,
    min_epoch_wgpu_hit_rate=None,
    max_epoch_wgpu_runtime_fallback_rate=None,
    max_epoch_wgpu_component_fallback_rate=None,
    require_guard_counts_available=False,
    min_guard_acceptance_rate_mean=None,
    max_guard_retention_rejected_epochs_mean=None,
    max_guard_target_stale_epochs_mean=None,
    max_guard_retention_rejected_rate_mean=None,
    max_guard_target_stale_rate_mean=None,
):
    current = run_summary_rows_by_profile(rows, "current")
    failures = []
    for profile in sorted(current):
        row = current[profile]
        display_profile = row.get("source_profile")
        ratio = row.get("target_retention_ratio")
        ratio_passed = True
        if min_target_retention_ratio is not None:
            if ratio is None:
                ratio_passed = False
                failures.append(f"{profile}: target_retention_ratio unavailable")
            elif not is_numeric_value(ratio):
                raise ValueError(f"{profile} row missing numeric target_retention_ratio")
            else:
                ratio_passed = float(ratio) >= min_target_retention_ratio
                if not ratio_passed:
                    failures.append(
                        f"{profile}: target_retention_ratio {float(ratio):.9f} below floor"
                    )
        accepted_rate = None
        accepted_passed = True
        if min_accepted_rate is not None:
            accepted_rate = numeric_value(row, "accepted_rate")
            accepted_passed = accepted_rate >= min_accepted_rate
            if not accepted_passed:
                failures.append(f"{profile}: accepted_rate {accepted_rate:.9f} below floor")
        movement_ok_rate = None
        movement_passed = True
        if min_movement_ok_rate is not None:
            movement_ok_rate = numeric_value(row, "movement_ok_rate")
            movement_passed = movement_ok_rate >= min_movement_ok_rate
            if not movement_passed:
                failures.append(
                    f"{profile}: movement_ok_rate {movement_ok_rate:.9f} below floor"
                )
        accuracy_margin = None
        accuracy_passed = True
        if min_retention_accuracy_margin is not None:
            accuracy_margin = numeric_value(row, "retention_accuracy_margin_min")
            accuracy_passed = accuracy_margin >= min_retention_accuracy_margin
            if not accuracy_passed:
                failures.append(
                    f"{profile}: retention_accuracy_margin_min {accuracy_margin:.9f} below floor"
                )
        perplexity_margin = None
        perplexity_passed = True
        if min_retention_perplexity_margin is not None:
            perplexity_margin = numeric_value(row, "retention_perplexity_margin_min")
            perplexity_passed = perplexity_margin >= min_retention_perplexity_margin
            if not perplexity_passed:
                failures.append(
                    f"{profile}: retention_perplexity_margin_min {perplexity_margin:.9f} below floor"
                )
        input_promotion_metric = None
        input_promotion_value = None
        input_promotion_current = None
        input_promotion_regression = None
        input_promotion_passed = True
        if max_input_promotion_metric_regression is not None:
            try:
                (
                    input_promotion_metric,
                    input_promotion_value,
                    input_promotion_current,
                    input_promotion_regression,
                ) = input_promotion_metric_regression(row)
            except ValueError as exc:
                input_promotion_passed = False
                failures.append(f"{profile}: {exc}")
            else:
                if input_promotion_regression is None:
                    input_promotion_passed = False
                    failures.append(f"{profile}: input_promotion_metric_regression unavailable")
                elif input_promotion_regression > max_input_promotion_metric_regression:
                    input_promotion_passed = False
                    failures.append(
                        f"{profile}: input_promotion_metric_regression "
                        f"{input_promotion_regression:.9f} above ceiling"
                    )
        epoch_wgpu_hit_rate = None
        epoch_wgpu_hit_rate_passed = True
        if min_epoch_wgpu_hit_rate is not None:
            epoch_wgpu_hit_rate = numeric_value(
                row,
                "epoch_tensor_backend_requested_wgpu_hit_rate_mean",
            )
            epoch_wgpu_hit_rate_passed = epoch_wgpu_hit_rate >= min_epoch_wgpu_hit_rate
            if not epoch_wgpu_hit_rate_passed:
                failures.append(
                    f"{profile}: epoch_wgpu_hit_rate_mean "
                    f"{epoch_wgpu_hit_rate:.9f} below floor"
                )
        epoch_wgpu_runtime_fallback_rate = None
        epoch_wgpu_runtime_fallback_rate_passed = True
        if max_epoch_wgpu_runtime_fallback_rate is not None:
            epoch_wgpu_runtime_fallback_rate = numeric_value(
                row,
                "epoch_tensor_backend_requested_wgpu_runtime_fallback_rate_mean",
            )
            epoch_wgpu_runtime_fallback_rate_passed = (
                epoch_wgpu_runtime_fallback_rate
                <= max_epoch_wgpu_runtime_fallback_rate
            )
            if not epoch_wgpu_runtime_fallback_rate_passed:
                failures.append(
                    f"{profile}: epoch_wgpu_runtime_fallback_rate_mean "
                    f"{epoch_wgpu_runtime_fallback_rate:.9f} above ceiling"
                )
        epoch_wgpu_component_fallback_rate = None
        epoch_wgpu_component_fallback_rate_passed = True
        if max_epoch_wgpu_component_fallback_rate is not None:
            epoch_wgpu_component_fallback_rate = numeric_value(
                row,
                "epoch_tensor_backend_requested_wgpu_component_fallback_rate_mean",
            )
            epoch_wgpu_component_fallback_rate_passed = (
                epoch_wgpu_component_fallback_rate
                <= max_epoch_wgpu_component_fallback_rate
            )
            if not epoch_wgpu_component_fallback_rate_passed:
                failures.append(
                    f"{profile}: epoch_wgpu_component_fallback_rate_mean "
                    f"{epoch_wgpu_component_fallback_rate:.9f} above ceiling"
                )
        guard_counts_available = row.get("guard_epoch_counts_available_all")
        guard_counts_passed = True
        if require_guard_counts_available:
            guard_counts_passed = guard_counts_available is True
            if not guard_counts_passed:
                failures.append(f"{profile}: guard_epoch_counts_available_all is not true")
        guard_acceptance_rate = None
        guard_acceptance_rate_passed = True
        if min_guard_acceptance_rate_mean is not None:
            guard_acceptance_rate = numeric_value(row, "guard_acceptance_rate_mean")
            guard_acceptance_rate_passed = (
                guard_acceptance_rate >= min_guard_acceptance_rate_mean
            )
            if not guard_acceptance_rate_passed:
                failures.append(
                    f"{profile}: guard_acceptance_rate_mean "
                    f"{guard_acceptance_rate:.9f} below floor"
                )
        guard_retention_rejected = None
        guard_retention_rejected_passed = True
        if max_guard_retention_rejected_epochs_mean is not None:
            guard_retention_rejected = numeric_value(
                row,
                "guard_retention_rejected_epochs_mean",
            )
            guard_retention_rejected_passed = (
                guard_retention_rejected <= max_guard_retention_rejected_epochs_mean
            )
            if not guard_retention_rejected_passed:
                failures.append(
                    f"{profile}: guard_retention_rejected_epochs_mean "
                    f"{guard_retention_rejected:.9f} above ceiling"
                )
        guard_target_stale = None
        guard_target_stale_passed = True
        if max_guard_target_stale_epochs_mean is not None:
            guard_target_stale = numeric_value(row, "guard_target_stale_epochs_mean")
            guard_target_stale_passed = (
                guard_target_stale <= max_guard_target_stale_epochs_mean
            )
            if not guard_target_stale_passed:
                failures.append(
                    f"{profile}: guard_target_stale_epochs_mean "
                    f"{guard_target_stale:.9f} above ceiling"
                )
        guard_retention_rejected_rate = None
        guard_retention_rejected_rate_passed = True
        if max_guard_retention_rejected_rate_mean is not None:
            guard_retention_rejected_rate = numeric_value(
                row,
                "guard_retention_rejected_rate_mean",
            )
            guard_retention_rejected_rate_passed = (
                guard_retention_rejected_rate <= max_guard_retention_rejected_rate_mean
            )
            if not guard_retention_rejected_rate_passed:
                failures.append(
                    f"{profile}: guard_retention_rejected_rate_mean "
                    f"{guard_retention_rejected_rate:.9f} above ceiling"
                )
        guard_target_stale_rate = None
        guard_target_stale_rate_passed = True
        if max_guard_target_stale_rate_mean is not None:
            guard_target_stale_rate = numeric_value(row, "guard_target_stale_rate_mean")
            guard_target_stale_rate_passed = (
                guard_target_stale_rate <= max_guard_target_stale_rate_mean
            )
            if not guard_target_stale_rate_passed:
                failures.append(
                    f"{profile}: guard_target_stale_rate_mean "
                    f"{guard_target_stale_rate:.9f} above ceiling"
                )
        passed = (
            ratio_passed
            and accepted_passed
            and movement_passed
            and accuracy_passed
            and perplexity_passed
            and input_promotion_passed
            and epoch_wgpu_hit_rate_passed
            and epoch_wgpu_runtime_fallback_rate_passed
            and epoch_wgpu_component_fallback_rate_passed
            and guard_counts_passed
            and guard_acceptance_rate_passed
            and guard_retention_rejected_passed
            and guard_target_stale_passed
            and guard_retention_rejected_rate_passed
            and guard_target_stale_rate_passed
        )
        print(
            f"profile_run_gate profile={display_profile} "
            f"run_key={profile} "
            f"target_retention_ratio={optional_numeric_label(ratio)} "
            f"min_target_retention_ratio={optional_numeric_label(min_target_retention_ratio)} "
            f"accepted_rate={optional_numeric_label(accepted_rate)} "
            f"min_accepted_rate={optional_numeric_label(min_accepted_rate)} "
            f"movement_ok_rate={optional_numeric_label(movement_ok_rate)} "
            f"min_movement_ok_rate={optional_numeric_label(min_movement_ok_rate)} "
            f"retention_accuracy_margin_min={optional_numeric_label(accuracy_margin)} "
            f"min_retention_accuracy_margin={optional_numeric_label(min_retention_accuracy_margin)} "
            f"retention_perplexity_margin_min={optional_numeric_label(perplexity_margin)} "
            f"min_retention_perplexity_margin={optional_numeric_label(min_retention_perplexity_margin)} "
            f"input_promotion_metric={input_promotion_metric or 'none'} "
            f"input_promotion_value={optional_numeric_label(input_promotion_value)} "
            f"input_promotion_current={optional_numeric_label(input_promotion_current)} "
            f"input_promotion_metric_regression={optional_numeric_label(input_promotion_regression)} "
            f"max_input_promotion_metric_regression={optional_numeric_label(max_input_promotion_metric_regression)} "
            f"epoch_wgpu_hit_rate_mean={optional_numeric_label(epoch_wgpu_hit_rate)} "
            f"min_epoch_wgpu_hit_rate={optional_numeric_label(min_epoch_wgpu_hit_rate)} "
            f"epoch_wgpu_runtime_fallback_rate_mean={optional_numeric_label(epoch_wgpu_runtime_fallback_rate)} "
            f"max_epoch_wgpu_runtime_fallback_rate={optional_numeric_label(max_epoch_wgpu_runtime_fallback_rate)} "
            f"epoch_wgpu_component_fallback_rate_mean={optional_numeric_label(epoch_wgpu_component_fallback_rate)} "
            f"max_epoch_wgpu_component_fallback_rate={optional_numeric_label(max_epoch_wgpu_component_fallback_rate)} "
            f"guard_epoch_counts_available_all={guard_counts_available} "
            f"require_guard_counts_available={require_guard_counts_available} "
            f"guard_acceptance_rate_mean={optional_numeric_label(guard_acceptance_rate)} "
            f"min_guard_acceptance_rate_mean={optional_numeric_label(min_guard_acceptance_rate_mean)} "
            f"guard_retention_rejected_epochs_mean={optional_numeric_label(guard_retention_rejected)} "
            f"max_guard_retention_rejected_epochs_mean={optional_numeric_label(max_guard_retention_rejected_epochs_mean)} "
            f"guard_target_stale_epochs_mean={optional_numeric_label(guard_target_stale)} "
            f"max_guard_target_stale_epochs_mean={optional_numeric_label(max_guard_target_stale_epochs_mean)} "
            f"guard_retention_rejected_rate_mean={optional_numeric_label(guard_retention_rejected_rate)} "
            f"max_guard_retention_rejected_rate_mean={optional_numeric_label(max_guard_retention_rejected_rate_mean)} "
            f"guard_target_stale_rate_mean={optional_numeric_label(guard_target_stale_rate)} "
            f"max_guard_target_stale_rate_mean={optional_numeric_label(max_guard_target_stale_rate_mean)} "
            f"passed={passed}"
        )
    if failures:
        raise RuntimeError("profile run summary gate failed: " + "; ".join(failures))
    return len(current)


TRAINING_POLICY_FIELDS = [
    "adapter_weight_decay_variant",
    "adapter_weight_decay",
    "max_grad_norm_variant",
    "max_grad_norm",
    "gradient_accumulation_steps_variant",
    "gradient_accumulation_steps",
    "ft_control_variant",
    "ft_epochs",
    "target_min_loss_delta_policy",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "lr_decay_patience",
    "lr_decay_factor",
    "lr_decay_min_delta",
]


FT_CONTROL_POLICY_FIELDS = {
    "ft_control_variant",
    "ft_epochs",
    "target_min_loss_delta_policy",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "lr_decay_patience",
    "lr_decay_factor",
    "lr_decay_min_delta",
}


TRAINING_POLICY_COMPARE_FIELDS = TRAINING_POLICY_FIELDS + ["training_policy_key"]


def changed_training_policy_fields(now, before):
    return [
        key
        for key in TRAINING_POLICY_COMPARE_FIELDS
        if now.get(key) != before.get(key)
    ]


INPUT_PROMOTION_COMPARE_FIELDS = [
    "input_promotion_run_key",
    "input_promotion_rank",
    "input_promotion_metric",
    "input_promotion_value",
    "input_promotion_ready",
    "input_promotion_ready_top_k",
    "input_promotion_ready_within",
    "input_promotion_ready_floor_passed",
    "input_promotion_ready_floor_failures",
    "input_promotion_ready_min_target_retention_ratio",
    "input_promotion_ready_min_accepted_rate",
    "input_promotion_ready_min_movement_ok_rate",
    "input_promotion_ready_min_retention_accuracy_margin",
    "input_promotion_ready_min_retention_perplexity_margin",
    "input_promotion_ready_min_epoch_wgpu_hit_rate",
    "input_promotion_ready_max_epoch_wgpu_runtime_fallback_rate",
    "input_promotion_ready_max_epoch_wgpu_component_fallback_rate",
    "input_promotion_ready_max_input_promotion_metric_regression",
    "input_promotion_ready_require_guard_counts_available",
    "input_promotion_ready_min_guard_acceptance_rate_mean",
    "input_promotion_ready_max_guard_retention_rejected_epochs_mean",
    "input_promotion_ready_max_guard_target_stale_epochs_mean",
    "input_promotion_ready_max_guard_retention_rejected_rate_mean",
    "input_promotion_ready_max_guard_target_stale_rate_mean",
]


def input_promotion_compare_value(row, key):
    if key == "input_promotion_ready_top_k":
        value = row.get(key)
        if value is None and row.get("input_promotion_rank") is not None:
            return 1
        return value
    return row.get(key)


def changed_input_promotion_fields(now, before):
    return [
        key
        for key in INPUT_PROMOTION_COMPARE_FIELDS
        if input_promotion_compare_value(now, key)
        != input_promotion_compare_value(before, key)
    ]


def compare_profile_run_summaries(
    current_rows,
    baseline_rows,
    *,
    max_target_loss_regression=None,
    max_retention_loss_regression=None,
    max_target_retention_gap_regression=None,
    max_target_retention_ratio_regression=None,
    min_target_retention_ratio=None,
    max_accepted_rate_regression=None,
    min_accepted_rate=None,
    max_movement_ok_rate_regression=None,
    max_guard_acceptance_rate_regression=None,
    max_guard_retention_rejected_rate_regression=None,
    max_guard_target_stale_rate_regression=None,
    max_epoch_wgpu_hit_rate_regression=None,
    max_epoch_wgpu_runtime_fallback_rate_regression=None,
    max_epoch_wgpu_component_fallback_rate_regression=None,
    min_movement_ok_rate=None,
    min_retention_accuracy_margin=None,
    min_retention_perplexity_margin=None,
    min_epoch_wgpu_hit_rate=None,
    max_epoch_wgpu_runtime_fallback_rate=None,
    max_epoch_wgpu_component_fallback_rate=None,
    require_source_match=False,
    require_config_match=False,
    require_case_scope_match=False,
    require_training_policy_match=False,
    require_input_promotion_match=False,
):
    current = run_summary_rows_by_profile(current_rows, "current")
    baseline = run_summary_rows_by_profile(baseline_rows, "baseline")
    missing = sorted(set(baseline) - set(current))
    if missing:
        raise RuntimeError(
            "baseline profile runs missing from current compare: " + ",".join(missing)
        )

    failures = []
    for profile in sorted(set(current) & set(baseline)):
        now = current[profile]
        before = baseline[profile]
        display_profile = now.get("source_profile")
        target_regression = max_regression(before, now, "target_loss_delta_mean")
        retention_regression = max_regression(before, now, "retention_loss_delta_mean")
        gap_regression = max_regression(before, now, "target_retention_gap_mean")
        ratio_regression = optional_max_regression(before, now, "target_retention_ratio")
        before_ratio = optional_numeric_value(before, "target_retention_ratio")
        ratio = optional_numeric_value(now, "target_retention_ratio")
        accepted_rate_regression = max_regression(before, now, "accepted_rate")
        accepted_rate = numeric_value(now, "accepted_rate")
        movement_ok_rate_regression = max_regression(before, now, "movement_ok_rate")
        movement_ok_rate = numeric_value(now, "movement_ok_rate")
        guard_acceptance_rate_regression = None
        if max_guard_acceptance_rate_regression is not None:
            guard_acceptance_rate_regression = max_regression(
                before,
                now,
                "guard_acceptance_rate_mean",
            )
        guard_retention_rejected_rate_regression = None
        if max_guard_retention_rejected_rate_regression is not None:
            guard_retention_rejected_rate_regression = max_increase_regression(
                before,
                now,
                "guard_retention_rejected_rate_mean",
            )
        guard_target_stale_rate_regression = None
        if max_guard_target_stale_rate_regression is not None:
            guard_target_stale_rate_regression = max_increase_regression(
                before,
                now,
                "guard_target_stale_rate_mean",
            )
        epoch_wgpu_hit_rate_regression = None
        if max_epoch_wgpu_hit_rate_regression is not None:
            epoch_wgpu_hit_rate_regression = max_regression(
                before,
                now,
                "epoch_tensor_backend_requested_wgpu_hit_rate_mean",
            )
        epoch_wgpu_runtime_fallback_rate_regression = None
        if max_epoch_wgpu_runtime_fallback_rate_regression is not None:
            epoch_wgpu_runtime_fallback_rate_regression = max_increase_regression(
                before,
                now,
                "epoch_tensor_backend_requested_wgpu_runtime_fallback_rate_mean",
            )
        epoch_wgpu_component_fallback_rate_regression = None
        if max_epoch_wgpu_component_fallback_rate_regression is not None:
            epoch_wgpu_component_fallback_rate_regression = max_increase_regression(
                before,
                now,
                "epoch_tensor_backend_requested_wgpu_component_fallback_rate_mean",
            )
        if min_epoch_wgpu_hit_rate is not None:
            epoch_wgpu_hit_rate = numeric_value(
                now,
                "epoch_tensor_backend_requested_wgpu_hit_rate_mean",
            )
        else:
            epoch_wgpu_hit_rate = optional_numeric_value(
                now,
                "epoch_tensor_backend_requested_wgpu_hit_rate_mean",
            )
        if max_epoch_wgpu_runtime_fallback_rate is not None:
            epoch_wgpu_runtime_fallback_rate = numeric_value(
                now,
                "epoch_tensor_backend_requested_wgpu_runtime_fallback_rate_mean",
            )
        else:
            epoch_wgpu_runtime_fallback_rate = optional_numeric_value(
                now,
                "epoch_tensor_backend_requested_wgpu_runtime_fallback_rate_mean",
            )
        if max_epoch_wgpu_component_fallback_rate is not None:
            epoch_wgpu_component_fallback_rate = numeric_value(
                now,
                "epoch_tensor_backend_requested_wgpu_component_fallback_rate_mean",
            )
        else:
            epoch_wgpu_component_fallback_rate = optional_numeric_value(
                now,
                "epoch_tensor_backend_requested_wgpu_component_fallback_rate_mean",
            )
        source_changed = now.get("selected_source") != before.get("selected_source")
        config_changed = now.get("config") != before.get("config")
        case_scope_changed = (
            now.get("cases") != before.get("cases")
            or now.get("case_labels") != before.get("case_labels")
        )
        training_policy_changes = changed_training_policy_fields(now, before)
        input_promotion_changes = changed_input_promotion_fields(now, before)
        input_promotion_ready_top_k_before = input_promotion_compare_value(
            before,
            "input_promotion_ready_top_k",
        )
        input_promotion_ready_top_k_after = input_promotion_compare_value(
            now,
            "input_promotion_ready_top_k",
        )
        accuracy_margin = numeric_value(now, "retention_accuracy_margin_min")
        perplexity_margin = numeric_value(now, "retention_perplexity_margin_min")
        passed = (
            (max_target_loss_regression is None or target_regression <= max_target_loss_regression)
            and (
                max_retention_loss_regression is None
                or retention_regression <= max_retention_loss_regression
            )
            and (
                max_target_retention_gap_regression is None
                or gap_regression <= max_target_retention_gap_regression
            )
            and (
                max_target_retention_ratio_regression is None
                or (
                    ratio_regression is not None
                    and ratio_regression <= max_target_retention_ratio_regression
                )
            )
            and (
                min_target_retention_ratio is None
                or (ratio is not None and ratio >= min_target_retention_ratio)
            )
            and (
                max_accepted_rate_regression is None
                or accepted_rate_regression <= max_accepted_rate_regression
            )
            and (min_accepted_rate is None or accepted_rate >= min_accepted_rate)
            and (
                max_movement_ok_rate_regression is None
                or movement_ok_rate_regression <= max_movement_ok_rate_regression
            )
            and (
                max_guard_acceptance_rate_regression is None
                or guard_acceptance_rate_regression <= max_guard_acceptance_rate_regression
            )
            and (
                max_guard_retention_rejected_rate_regression is None
                or guard_retention_rejected_rate_regression
                <= max_guard_retention_rejected_rate_regression
            )
            and (
                max_guard_target_stale_rate_regression is None
                or guard_target_stale_rate_regression
                <= max_guard_target_stale_rate_regression
            )
            and (
                max_epoch_wgpu_hit_rate_regression is None
                or epoch_wgpu_hit_rate_regression
                <= max_epoch_wgpu_hit_rate_regression
            )
            and (
                max_epoch_wgpu_runtime_fallback_rate_regression is None
                or epoch_wgpu_runtime_fallback_rate_regression
                <= max_epoch_wgpu_runtime_fallback_rate_regression
            )
            and (
                max_epoch_wgpu_component_fallback_rate_regression is None
                or epoch_wgpu_component_fallback_rate_regression
                <= max_epoch_wgpu_component_fallback_rate_regression
            )
            and (min_movement_ok_rate is None or movement_ok_rate >= min_movement_ok_rate)
            and (
                min_retention_accuracy_margin is None
                or accuracy_margin >= min_retention_accuracy_margin
            )
            and (
                min_retention_perplexity_margin is None
                or perplexity_margin >= min_retention_perplexity_margin
            )
            and (
                min_epoch_wgpu_hit_rate is None
                or epoch_wgpu_hit_rate >= min_epoch_wgpu_hit_rate
            )
            and (
                max_epoch_wgpu_runtime_fallback_rate is None
                or epoch_wgpu_runtime_fallback_rate
                <= max_epoch_wgpu_runtime_fallback_rate
            )
            and (
                max_epoch_wgpu_component_fallback_rate is None
                or epoch_wgpu_component_fallback_rate
                <= max_epoch_wgpu_component_fallback_rate
            )
            and (not require_source_match or not source_changed)
            and (not require_config_match or not config_changed)
            and (not require_case_scope_match or not case_scope_changed)
            and (not require_training_policy_match or not training_policy_changes)
            and (not require_input_promotion_match or not input_promotion_changes)
        )
        print(
            f"profile_run_compare profile={display_profile} "
            f"run_key={profile} "
            f"source_before={before.get('selected_source')} "
            f"source_after={now.get('selected_source')} "
            f"cases_before={before.get('cases')} "
            f"cases_after={now.get('cases')} "
            f"case_labels_before={before.get('case_labels')} "
            f"case_labels_after={now.get('case_labels')} "
            f"target_loss_delta_before={numeric_value(before, 'target_loss_delta_mean'):.9f} "
            f"target_loss_delta_after={numeric_value(now, 'target_loss_delta_mean'):.9f} "
            f"retention_loss_delta_before={numeric_value(before, 'retention_loss_delta_mean'):.9f} "
            f"retention_loss_delta_after={numeric_value(now, 'retention_loss_delta_mean'):.9f} "
            f"target_retention_gap_before={numeric_value(before, 'target_retention_gap_mean'):.9f} "
            f"target_retention_gap_after={numeric_value(now, 'target_retention_gap_mean'):.9f} "
            f"target_retention_ratio_before={optional_numeric_label(before_ratio)} "
            f"target_retention_ratio_after={optional_numeric_label(ratio)} "
            f"min_target_retention_ratio={min_target_retention_ratio} "
            f"accepted_rate_regression={accepted_rate_regression:.9f} "
            f"accepted_rate={accepted_rate:.9f} "
            f"min_accepted_rate={min_accepted_rate} "
            f"movement_ok_rate_regression={movement_ok_rate_regression:.9f} "
            f"movement_ok_rate={movement_ok_rate:.9f} "
            f"min_movement_ok_rate={min_movement_ok_rate} "
            f"guard_acceptance_rate_regression={optional_numeric_label(guard_acceptance_rate_regression)} "
            f"max_guard_acceptance_rate_regression={optional_numeric_label(max_guard_acceptance_rate_regression)} "
            f"guard_retention_rejected_rate_regression={optional_numeric_label(guard_retention_rejected_rate_regression)} "
            f"max_guard_retention_rejected_rate_regression={optional_numeric_label(max_guard_retention_rejected_rate_regression)} "
            f"guard_target_stale_rate_regression={optional_numeric_label(guard_target_stale_rate_regression)} "
            f"max_guard_target_stale_rate_regression={optional_numeric_label(max_guard_target_stale_rate_regression)} "
            f"epoch_wgpu_hit_rate_regression={optional_numeric_label(epoch_wgpu_hit_rate_regression)} "
            f"max_epoch_wgpu_hit_rate_regression={optional_numeric_label(max_epoch_wgpu_hit_rate_regression)} "
            f"epoch_wgpu_runtime_fallback_rate_regression={optional_numeric_label(epoch_wgpu_runtime_fallback_rate_regression)} "
            f"max_epoch_wgpu_runtime_fallback_rate_regression={optional_numeric_label(max_epoch_wgpu_runtime_fallback_rate_regression)} "
            f"epoch_wgpu_component_fallback_rate_regression={optional_numeric_label(epoch_wgpu_component_fallback_rate_regression)} "
            f"max_epoch_wgpu_component_fallback_rate_regression={optional_numeric_label(max_epoch_wgpu_component_fallback_rate_regression)} "
            f"epoch_wgpu_hit_rate_mean={optional_numeric_label(epoch_wgpu_hit_rate)} "
            f"min_epoch_wgpu_hit_rate={optional_numeric_label(min_epoch_wgpu_hit_rate)} "
            f"epoch_wgpu_runtime_fallback_rate_mean={optional_numeric_label(epoch_wgpu_runtime_fallback_rate)} "
            f"max_epoch_wgpu_runtime_fallback_rate={optional_numeric_label(max_epoch_wgpu_runtime_fallback_rate)} "
            f"epoch_wgpu_component_fallback_rate_mean={optional_numeric_label(epoch_wgpu_component_fallback_rate)} "
            f"max_epoch_wgpu_component_fallback_rate={optional_numeric_label(max_epoch_wgpu_component_fallback_rate)} "
            f"retention_accuracy_margin_min={accuracy_margin:.9f} "
            f"retention_perplexity_margin_min={perplexity_margin:.9f} "
            f"source_changed={source_changed} "
            f"config_changed={config_changed} "
            f"case_scope_changed={case_scope_changed} "
            f"training_policy_changed={bool(training_policy_changes)} "
            f"training_policy_key_before={before.get('training_policy_key')} "
            f"training_policy_key_after={now.get('training_policy_key')} "
            f"input_promotion_changed={bool(input_promotion_changes)} "
            f"input_promotion_rank_before={before.get('input_promotion_rank')} "
            f"input_promotion_rank_after={now.get('input_promotion_rank')} "
            f"input_promotion_metric_before={before.get('input_promotion_metric')} "
            f"input_promotion_metric_after={now.get('input_promotion_metric')} "
            f"input_promotion_ready_top_k_before={input_promotion_ready_top_k_before} "
            f"input_promotion_ready_top_k_after={input_promotion_ready_top_k_after} "
            f"input_promotion_ready_within_before={before.get('input_promotion_ready_within')} "
            f"input_promotion_ready_within_after={now.get('input_promotion_ready_within')} "
            f"input_promotion_ready_floor_passed_before={before.get('input_promotion_ready_floor_passed')} "
            f"input_promotion_ready_floor_passed_after={now.get('input_promotion_ready_floor_passed')} "
            f"passed={passed}"
        )
        if max_target_loss_regression is not None and target_regression > max_target_loss_regression:
            failures.append(f"{profile}: target_loss_delta_mean regressed by {target_regression:.9f}")
        if (
            max_retention_loss_regression is not None
            and retention_regression > max_retention_loss_regression
        ):
            failures.append(
                f"{profile}: retention_loss_delta_mean regressed by {retention_regression:.9f}"
            )
        if (
            max_target_retention_gap_regression is not None
            and gap_regression > max_target_retention_gap_regression
        ):
            failures.append(
                f"{profile}: target_retention_gap_mean regressed by {gap_regression:.9f}"
            )
        if (
            max_target_retention_ratio_regression is not None
        ):
            if ratio_regression is None:
                failures.append(f"{profile}: target_retention_ratio regression is unavailable")
            elif ratio_regression > max_target_retention_ratio_regression:
                failures.append(
                    f"{profile}: target_retention_ratio regressed by {ratio_regression:.9f}"
                )
        if min_target_retention_ratio is not None:
            if ratio is None:
                failures.append(f"{profile}: target_retention_ratio is unavailable")
            elif ratio < min_target_retention_ratio:
                failures.append(f"{profile}: target_retention_ratio {ratio:.9f} below floor")
        if (
            max_accepted_rate_regression is not None
            and accepted_rate_regression > max_accepted_rate_regression
        ):
            failures.append(f"{profile}: accepted_rate regressed by {accepted_rate_regression:.9f}")
        if min_accepted_rate is not None and accepted_rate < min_accepted_rate:
            failures.append(f"{profile}: accepted_rate {accepted_rate:.9f} below floor")
        if (
            max_movement_ok_rate_regression is not None
            and movement_ok_rate_regression > max_movement_ok_rate_regression
        ):
            failures.append(
                f"{profile}: movement_ok_rate regressed by {movement_ok_rate_regression:.9f}"
            )
        if (
            max_guard_acceptance_rate_regression is not None
            and guard_acceptance_rate_regression > max_guard_acceptance_rate_regression
        ):
            failures.append(
                f"{profile}: guard_acceptance_rate_mean regressed by "
                f"{guard_acceptance_rate_regression:.9f}"
            )
        if (
            max_guard_retention_rejected_rate_regression is not None
            and guard_retention_rejected_rate_regression
            > max_guard_retention_rejected_rate_regression
        ):
            failures.append(
                f"{profile}: guard_retention_rejected_rate_mean regressed by "
                f"{guard_retention_rejected_rate_regression:.9f}"
            )
        if (
            max_guard_target_stale_rate_regression is not None
            and guard_target_stale_rate_regression > max_guard_target_stale_rate_regression
        ):
            failures.append(
                f"{profile}: guard_target_stale_rate_mean regressed by "
                f"{guard_target_stale_rate_regression:.9f}"
            )
        if (
            max_epoch_wgpu_hit_rate_regression is not None
            and epoch_wgpu_hit_rate_regression > max_epoch_wgpu_hit_rate_regression
        ):
            failures.append(
                f"{profile}: epoch_wgpu_hit_rate_mean regressed by "
                f"{epoch_wgpu_hit_rate_regression:.9f}"
            )
        if (
            max_epoch_wgpu_runtime_fallback_rate_regression is not None
            and epoch_wgpu_runtime_fallback_rate_regression
            > max_epoch_wgpu_runtime_fallback_rate_regression
        ):
            failures.append(
                f"{profile}: epoch_wgpu_runtime_fallback_rate_mean regressed by "
                f"{epoch_wgpu_runtime_fallback_rate_regression:.9f}"
            )
        if (
            max_epoch_wgpu_component_fallback_rate_regression is not None
            and epoch_wgpu_component_fallback_rate_regression
            > max_epoch_wgpu_component_fallback_rate_regression
        ):
            failures.append(
                f"{profile}: epoch_wgpu_component_fallback_rate_mean regressed by "
                f"{epoch_wgpu_component_fallback_rate_regression:.9f}"
            )
        if min_movement_ok_rate is not None and movement_ok_rate < min_movement_ok_rate:
            failures.append(f"{profile}: movement_ok_rate {movement_ok_rate:.9f} below floor")
        if min_retention_accuracy_margin is not None and accuracy_margin < min_retention_accuracy_margin:
            failures.append(
                f"{profile}: retention_accuracy_margin_min {accuracy_margin:.9f} below floor"
            )
        if (
            min_retention_perplexity_margin is not None
            and perplexity_margin < min_retention_perplexity_margin
        ):
            failures.append(
                f"{profile}: retention_perplexity_margin_min {perplexity_margin:.9f} below floor"
            )
        if (
            min_epoch_wgpu_hit_rate is not None
            and epoch_wgpu_hit_rate < min_epoch_wgpu_hit_rate
        ):
            failures.append(
                f"{profile}: epoch_wgpu_hit_rate_mean "
                f"{epoch_wgpu_hit_rate:.9f} below floor"
            )
        if (
            max_epoch_wgpu_runtime_fallback_rate is not None
            and epoch_wgpu_runtime_fallback_rate > max_epoch_wgpu_runtime_fallback_rate
        ):
            failures.append(
                f"{profile}: epoch_wgpu_runtime_fallback_rate_mean "
                f"{epoch_wgpu_runtime_fallback_rate:.9f} above ceiling"
            )
        if (
            max_epoch_wgpu_component_fallback_rate is not None
            and epoch_wgpu_component_fallback_rate
            > max_epoch_wgpu_component_fallback_rate
        ):
            failures.append(
                f"{profile}: epoch_wgpu_component_fallback_rate_mean "
                f"{epoch_wgpu_component_fallback_rate:.9f} above ceiling"
            )
        if require_source_match and source_changed:
            failures.append(
                f"{profile}: selected_source changed from {before.get('selected_source')} to {now.get('selected_source')}"
            )
        if require_config_match and config_changed:
            failures.append(f"{profile}: config changed from {before.get('config')} to {now.get('config')}")
        if require_case_scope_match and case_scope_changed:
            failures.append(
                f"{profile}: case scope changed from "
                f"{before.get('cases')}:{before.get('case_labels')} to "
                f"{now.get('cases')}:{now.get('case_labels')}"
            )
        if require_training_policy_match and training_policy_changes:
            changed = ",".join(training_policy_changes)
            failures.append(f"{profile}: training policy changed fields={changed}")
        if require_input_promotion_match and input_promotion_changes:
            changed = ",".join(input_promotion_changes)
            failures.append(f"{profile}: input promotion changed fields={changed}")

    extra = sorted(set(current) - set(baseline))
    for profile in extra:
        row = current[profile]
        print(
            f"profile_run_compare profile={row.get('source_profile')} "
            f"run_key={profile} baseline_missing=true"
        )
    if failures:
        raise RuntimeError("profile run summary regression gate failed: " + "; ".join(failures))
    return len(set(current) & set(baseline))


def main():
    args = parse_args()
    if args.current_run_summary_jsonl is not None:
        current_rows = load_jsonl(args.current_run_summary_jsonl)
        compared = compare_profile_run_summaries(
            current_rows,
            load_jsonl(args.compare_run_summary_jsonl),
            max_target_loss_regression=args.max_run_target_loss_regression,
            max_retention_loss_regression=args.max_run_retention_loss_regression,
            max_target_retention_gap_regression=args.max_run_target_retention_gap_regression,
            max_target_retention_ratio_regression=args.max_run_target_retention_ratio_regression,
            min_target_retention_ratio=args.min_run_target_retention_ratio,
            max_accepted_rate_regression=args.max_run_accepted_rate_regression,
            min_accepted_rate=args.min_run_accepted_rate,
            max_movement_ok_rate_regression=args.max_run_movement_ok_rate_regression,
            max_guard_acceptance_rate_regression=(
                args.max_run_guard_acceptance_rate_regression
            ),
            max_guard_retention_rejected_rate_regression=(
                args.max_run_guard_retention_rejected_rate_regression
            ),
            max_guard_target_stale_rate_regression=(
                args.max_run_guard_target_stale_rate_regression
            ),
            max_epoch_wgpu_hit_rate_regression=(
                args.max_run_epoch_wgpu_hit_rate_regression
            ),
            max_epoch_wgpu_runtime_fallback_rate_regression=(
                args.max_run_epoch_wgpu_runtime_fallback_rate_regression
            ),
            max_epoch_wgpu_component_fallback_rate_regression=(
                args.max_run_epoch_wgpu_component_fallback_rate_regression
            ),
            min_movement_ok_rate=args.min_run_movement_ok_rate,
            min_retention_accuracy_margin=args.min_run_retention_accuracy_margin,
            min_retention_perplexity_margin=args.min_run_retention_perplexity_margin,
            min_epoch_wgpu_hit_rate=args.min_run_epoch_wgpu_hit_rate,
            max_epoch_wgpu_runtime_fallback_rate=(
                args.max_run_epoch_wgpu_runtime_fallback_rate
            ),
            max_epoch_wgpu_component_fallback_rate=(
                args.max_run_epoch_wgpu_component_fallback_rate
            ),
            require_source_match=args.require_run_source_match,
            require_config_match=args.require_run_config_match,
            require_case_scope_match=args.require_run_case_scope_match,
            require_training_policy_match=args.require_run_training_policy_match,
            require_input_promotion_match=args.require_run_input_promotion_match,
        )
        print(f"profile_run_compare_rows={compared}")
        if (
            args.max_run_input_promotion_metric_regression is not None
            or args.require_run_guard_counts_available
            or args.min_run_guard_acceptance_rate_mean is not None
            or args.max_run_guard_retention_rejected_epochs_mean is not None
            or args.max_run_guard_target_stale_epochs_mean is not None
            or args.max_run_guard_retention_rejected_rate_mean is not None
            or args.max_run_guard_target_stale_rate_mean is not None
        ):
            check_profile_run_gates(
                current_rows,
                max_input_promotion_metric_regression=(
                    args.max_run_input_promotion_metric_regression
                ),
                require_guard_counts_available=args.require_run_guard_counts_available,
                min_guard_acceptance_rate_mean=args.min_run_guard_acceptance_rate_mean,
                max_guard_retention_rejected_epochs_mean=(
                    args.max_run_guard_retention_rejected_epochs_mean
                ),
                max_guard_target_stale_epochs_mean=(
                    args.max_run_guard_target_stale_epochs_mean
                ),
                max_guard_retention_rejected_rate_mean=(
                    args.max_run_guard_retention_rejected_rate_mean
                ),
                max_guard_target_stale_rate_mean=args.max_run_guard_target_stale_rate_mean,
            )
        if args.promotion_jsonl is not None:
            promotions = profile_run_promotion_rows(
                current_rows,
                promotion_metric=args.promotion_metric,
                ready_top_k=args.promotion_ready_top_k,
                ready_within=args.promotion_ready_within,
                ready_min_target_retention_ratio=(
                    args.promotion_ready_min_target_retention_ratio
                ),
                ready_min_accepted_rate=args.promotion_ready_min_accepted_rate,
                ready_min_movement_ok_rate=args.promotion_ready_min_movement_ok_rate,
                ready_min_retention_accuracy_margin=(
                    args.promotion_ready_min_retention_accuracy_margin
                ),
                ready_min_retention_perplexity_margin=(
                    args.promotion_ready_min_retention_perplexity_margin
                ),
                ready_min_epoch_wgpu_hit_rate=(
                    args.promotion_ready_min_epoch_wgpu_hit_rate
                ),
                ready_min_guard_acceptance_rate_mean=(
                    args.promotion_ready_min_guard_acceptance_rate_mean
                ),
                ready_max_input_promotion_metric_regression=(
                    args.promotion_ready_max_input_promotion_metric_regression
                ),
                ready_max_epoch_wgpu_runtime_fallback_rate=(
                    args.promotion_ready_max_epoch_wgpu_runtime_fallback_rate
                ),
                ready_max_epoch_wgpu_component_fallback_rate=(
                    args.promotion_ready_max_epoch_wgpu_component_fallback_rate
                ),
                ready_require_guard_counts_available=(
                    args.promotion_ready_require_guard_counts_available
                ),
                ready_max_guard_retention_rejected_epochs_mean=(
                    args.promotion_ready_max_guard_retention_rejected_epochs_mean
                ),
                ready_max_guard_target_stale_epochs_mean=(
                    args.promotion_ready_max_guard_target_stale_epochs_mean
                ),
                ready_max_guard_retention_rejected_rate_mean=(
                    args.promotion_ready_max_guard_retention_rejected_rate_mean
                ),
                ready_max_guard_target_stale_rate_mean=(
                    args.promotion_ready_max_guard_target_stale_rate_mean
                ),
            )
            print_promotion_rows(promotions)
            write_jsonl(args.promotion_jsonl, promotions)
            print(f"profile_promotion_jsonl={args.promotion_jsonl} rows={len(promotions)}")
            if (
                args.min_promotion_ready_count is not None
                or args.min_promotion_ready_rate is not None
                or args.min_promotion_ready_guard_policy_count is not None
                or args.require_promotion_ready_guard_policy
            ):
                check_promotion_ready_gates(
                    promotions,
                    min_ready_count=args.min_promotion_ready_count,
                    min_ready_rate=args.min_promotion_ready_rate,
                    min_ready_guard_policy_count=(
                        args.min_promotion_ready_guard_policy_count
                    ),
                    require_ready_guard_policy=(
                        args.require_promotion_ready_guard_policy
                    ),
                )
        return
    source_paths = parse_source_paths(args.source_paths)
    promotion_jsonl_rows = (
        load_jsonl(args.promotion_input_jsonl)
        if args.promotion_input_jsonl is not None
        else None
    )
    if promotion_jsonl_rows is not None:
        selection = promotion_selection_summary(
            promotion_jsonl_rows,
            args.profiles,
            ready_only=not args.include_non_ready_promotions,
        )
        print_promotion_selection_summary(selection)
        if args.promotion_selection_jsonl is not None:
            write_jsonl(args.promotion_selection_jsonl, [selection])
            print(
                f"profile_promotion_selection_jsonl={args.promotion_selection_jsonl} rows=1"
            )
        if (
            args.min_promotion_ready_count is not None
            or args.min_promotion_ready_rate is not None
            or args.min_promotion_ready_guard_policy_count is not None
            or args.require_promotion_ready_guard_policy
        ):
            check_promotion_ready_gates(
                selected_promotion_rows(promotion_jsonl_rows, args.profiles),
                min_ready_count=args.min_promotion_ready_count,
                min_ready_rate=args.min_promotion_ready_rate,
                min_ready_guard_policy_count=args.min_promotion_ready_guard_policy_count,
                require_ready_guard_policy=args.require_promotion_ready_guard_policy,
            )
    rows = command_rows_for_profiles(
        load_jsonl(args.profile_jsonl),
        source_paths=source_paths,
        profiles=args.profiles,
        cases=args.cases,
        configs=args.configs,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        python_executable=args.python,
        sweep_script=args.sweep_script,
        case_jsonls=args.case_jsonls,
        lora_config_jsonls=args.lora_config_jsonls,
        extra_args=args.extra_args,
        aggregate_gates=not args.no_aggregate_gates,
        min_aggregate_epoch_wgpu_hit_rate=args.min_aggregate_epoch_wgpu_hit_rate,
        max_aggregate_epoch_wgpu_runtime_fallback_rate=(
            args.max_aggregate_epoch_wgpu_runtime_fallback_rate
        ),
        max_aggregate_epoch_wgpu_component_fallback_rate=(
            args.max_aggregate_epoch_wgpu_component_fallback_rate
        ),
        promotion_rows=promotion_jsonl_rows,
        promotion_ready_only=not args.include_non_ready_promotions,
        ft_control_override=args.ft_control_override,
    )
    print_command_rows(rows)
    if args.commands_jsonl is not None:
        write_jsonl(args.commands_jsonl, rows)
        print(f"profile_commands_jsonl={args.commands_jsonl} rows={len(rows)}")
    if args.run:
        run_command_rows(rows, events_jsonl=args.run_events_jsonl)
    needs_summaries = (
        args.run_summary_jsonl is not None
        or args.compare_run_summary_jsonl is not None
        or args.promotion_jsonl is not None
    )
    if needs_summaries:
        summaries = profile_run_summary_rows(rows)
        print_run_summary_rows(summaries)
    else:
        summaries = None
    if args.run_summary_jsonl is not None:
        write_jsonl(args.run_summary_jsonl, summaries)
        print(f"profile_run_summary_jsonl={args.run_summary_jsonl} rows={len(summaries)}")
    if args.compare_run_summary_jsonl is not None:
        compared = compare_profile_run_summaries(
            summaries,
            load_jsonl(args.compare_run_summary_jsonl),
            max_target_loss_regression=args.max_run_target_loss_regression,
            max_retention_loss_regression=args.max_run_retention_loss_regression,
            max_target_retention_gap_regression=args.max_run_target_retention_gap_regression,
            max_target_retention_ratio_regression=args.max_run_target_retention_ratio_regression,
            min_target_retention_ratio=args.min_run_target_retention_ratio,
            max_accepted_rate_regression=args.max_run_accepted_rate_regression,
            min_accepted_rate=args.min_run_accepted_rate,
            max_movement_ok_rate_regression=args.max_run_movement_ok_rate_regression,
            max_guard_acceptance_rate_regression=(
                args.max_run_guard_acceptance_rate_regression
            ),
            max_guard_retention_rejected_rate_regression=(
                args.max_run_guard_retention_rejected_rate_regression
            ),
            max_guard_target_stale_rate_regression=(
                args.max_run_guard_target_stale_rate_regression
            ),
            max_epoch_wgpu_hit_rate_regression=(
                args.max_run_epoch_wgpu_hit_rate_regression
            ),
            max_epoch_wgpu_runtime_fallback_rate_regression=(
                args.max_run_epoch_wgpu_runtime_fallback_rate_regression
            ),
            max_epoch_wgpu_component_fallback_rate_regression=(
                args.max_run_epoch_wgpu_component_fallback_rate_regression
            ),
            min_movement_ok_rate=args.min_run_movement_ok_rate,
            min_retention_accuracy_margin=args.min_run_retention_accuracy_margin,
            min_retention_perplexity_margin=args.min_run_retention_perplexity_margin,
            min_epoch_wgpu_hit_rate=args.min_run_epoch_wgpu_hit_rate,
            max_epoch_wgpu_runtime_fallback_rate=(
                args.max_run_epoch_wgpu_runtime_fallback_rate
            ),
            max_epoch_wgpu_component_fallback_rate=(
                args.max_run_epoch_wgpu_component_fallback_rate
            ),
            require_source_match=args.require_run_source_match,
            require_config_match=args.require_run_config_match,
            require_case_scope_match=args.require_run_case_scope_match,
            require_training_policy_match=args.require_run_training_policy_match,
            require_input_promotion_match=args.require_run_input_promotion_match,
        )
        print(f"profile_run_compare_rows={compared}")
        if (
            args.max_run_input_promotion_metric_regression is not None
            or args.require_run_guard_counts_available
            or args.min_run_guard_acceptance_rate_mean is not None
            or args.max_run_guard_retention_rejected_epochs_mean is not None
            or args.max_run_guard_target_stale_epochs_mean is not None
            or args.max_run_guard_retention_rejected_rate_mean is not None
            or args.max_run_guard_target_stale_rate_mean is not None
        ):
            check_profile_run_gates(
                summaries,
                max_input_promotion_metric_regression=(
                    args.max_run_input_promotion_metric_regression
                ),
                require_guard_counts_available=args.require_run_guard_counts_available,
                min_guard_acceptance_rate_mean=args.min_run_guard_acceptance_rate_mean,
                max_guard_retention_rejected_epochs_mean=(
                    args.max_run_guard_retention_rejected_epochs_mean
                ),
                max_guard_target_stale_epochs_mean=(
                    args.max_run_guard_target_stale_epochs_mean
                ),
                max_guard_retention_rejected_rate_mean=(
                    args.max_run_guard_retention_rejected_rate_mean
                ),
                max_guard_target_stale_rate_mean=args.max_run_guard_target_stale_rate_mean,
            )
    elif (
        args.min_run_target_retention_ratio is not None
        or args.min_run_accepted_rate is not None
        or args.min_run_movement_ok_rate is not None
        or args.min_run_retention_accuracy_margin is not None
        or args.min_run_retention_perplexity_margin is not None
        or args.max_run_input_promotion_metric_regression is not None
        or args.min_run_epoch_wgpu_hit_rate is not None
        or args.max_run_epoch_wgpu_runtime_fallback_rate is not None
        or args.max_run_epoch_wgpu_component_fallback_rate is not None
        or args.require_run_guard_counts_available
        or args.min_run_guard_acceptance_rate_mean is not None
        or args.max_run_guard_retention_rejected_epochs_mean is not None
        or args.max_run_guard_target_stale_epochs_mean is not None
        or args.max_run_guard_retention_rejected_rate_mean is not None
        or args.max_run_guard_target_stale_rate_mean is not None
    ):
        check_profile_run_gates(
            summaries,
            min_target_retention_ratio=args.min_run_target_retention_ratio,
            min_accepted_rate=args.min_run_accepted_rate,
            min_movement_ok_rate=args.min_run_movement_ok_rate,
            min_retention_accuracy_margin=args.min_run_retention_accuracy_margin,
            min_retention_perplexity_margin=args.min_run_retention_perplexity_margin,
            max_input_promotion_metric_regression=(
                args.max_run_input_promotion_metric_regression
            ),
            min_epoch_wgpu_hit_rate=args.min_run_epoch_wgpu_hit_rate,
            max_epoch_wgpu_runtime_fallback_rate=(
                args.max_run_epoch_wgpu_runtime_fallback_rate
            ),
            max_epoch_wgpu_component_fallback_rate=(
                args.max_run_epoch_wgpu_component_fallback_rate
            ),
            require_guard_counts_available=args.require_run_guard_counts_available,
            min_guard_acceptance_rate_mean=args.min_run_guard_acceptance_rate_mean,
            max_guard_retention_rejected_epochs_mean=(
                args.max_run_guard_retention_rejected_epochs_mean
            ),
            max_guard_target_stale_epochs_mean=args.max_run_guard_target_stale_epochs_mean,
            max_guard_retention_rejected_rate_mean=(
                args.max_run_guard_retention_rejected_rate_mean
            ),
            max_guard_target_stale_rate_mean=args.max_run_guard_target_stale_rate_mean,
        )
    if args.promotion_jsonl is not None:
        promotions = profile_run_promotion_rows(
            summaries,
            promotion_metric=args.promotion_metric,
            ready_top_k=args.promotion_ready_top_k,
            ready_within=args.promotion_ready_within,
            ready_min_target_retention_ratio=(
                args.promotion_ready_min_target_retention_ratio
            ),
            ready_min_accepted_rate=args.promotion_ready_min_accepted_rate,
            ready_min_movement_ok_rate=args.promotion_ready_min_movement_ok_rate,
            ready_min_retention_accuracy_margin=(
                args.promotion_ready_min_retention_accuracy_margin
            ),
            ready_min_retention_perplexity_margin=(
                args.promotion_ready_min_retention_perplexity_margin
            ),
            ready_min_epoch_wgpu_hit_rate=(
                args.promotion_ready_min_epoch_wgpu_hit_rate
            ),
            ready_min_guard_acceptance_rate_mean=(
                args.promotion_ready_min_guard_acceptance_rate_mean
            ),
            ready_max_input_promotion_metric_regression=(
                args.promotion_ready_max_input_promotion_metric_regression
            ),
            ready_max_epoch_wgpu_runtime_fallback_rate=(
                args.promotion_ready_max_epoch_wgpu_runtime_fallback_rate
            ),
            ready_max_epoch_wgpu_component_fallback_rate=(
                args.promotion_ready_max_epoch_wgpu_component_fallback_rate
            ),
            ready_require_guard_counts_available=(
                args.promotion_ready_require_guard_counts_available
            ),
            ready_max_guard_retention_rejected_epochs_mean=(
                args.promotion_ready_max_guard_retention_rejected_epochs_mean
            ),
            ready_max_guard_target_stale_epochs_mean=(
                args.promotion_ready_max_guard_target_stale_epochs_mean
            ),
            ready_max_guard_retention_rejected_rate_mean=(
                args.promotion_ready_max_guard_retention_rejected_rate_mean
            ),
            ready_max_guard_target_stale_rate_mean=(
                args.promotion_ready_max_guard_target_stale_rate_mean
            ),
        )
        print_promotion_rows(promotions)
        write_jsonl(args.promotion_jsonl, promotions)
        print(f"profile_promotion_jsonl={args.promotion_jsonl} rows={len(promotions)}")
        if (
            args.min_promotion_ready_count is not None
            or args.min_promotion_ready_rate is not None
            or args.min_promotion_ready_guard_policy_count is not None
            or args.require_promotion_ready_guard_policy
        ):
            check_promotion_ready_gates(
                promotions,
                min_ready_count=args.min_promotion_ready_count,
                min_ready_rate=args.min_promotion_ready_rate,
                min_ready_guard_policy_count=args.min_promotion_ready_guard_policy_count,
                require_ready_guard_policy=args.require_promotion_ready_guard_policy,
            )


if __name__ == "__main__":
    main()
