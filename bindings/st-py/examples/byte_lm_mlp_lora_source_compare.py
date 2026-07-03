import argparse
from pathlib import Path

from byte_lm_mlp_lora_sweep import (
    aggregate_case_labels,
    is_numeric_value,
    load_jsonl,
    numeric_value,
    summary_bool_value,
    validate_aggregate_row,
    write_jsonl,
)


WINNER_METRICS = [
    "target_loss_delta_mean",
    "target_retention_gap_mean",
    "target_retention_ratio",
]

PROFILE_METRICS = [
    ("strong_effect", "target_loss_delta_mean"),
    ("selective_gap", "target_retention_gap_mean"),
    ("selective_ratio", "target_retention_ratio"),
]
PROFILE_METRIC_BY_NAME = dict(PROFILE_METRICS)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare real-HF checkpoint sources from byte MLP LoRA sweep "
            "aggregate JSONL rows."
        )
    )
    parser.add_argument(
        "--aggregate-jsonl",
        dest="aggregate_jsonls",
        action="append",
        type=Path,
        required=True,
        help=(
            "A byte_lm_mlp_lora_sweep.py --aggregate-jsonl file. May be "
            "repeated for multiple checkpoint sources."
        ),
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Optional output path for ranked checkpoint_source_candidate rows.",
    )
    parser.add_argument(
        "--profile-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional output path for checkpoint_source_profile rows that select "
            "guarded source/gain lanes by reusable profile metrics."
        ),
    )
    parser.add_argument(
        "--profile",
        dest="profiles",
        action="append",
        choices=[name for name, _metric in PROFILE_METRICS],
        default=[],
        help=(
            "Source/gain profile lane to emit. May be repeated. Defaults to all "
            "profile lanes when --profile-jsonl is used."
        ),
    )
    parser.add_argument(
        "--winner-metric",
        choices=WINNER_METRICS,
        default="target_loss_delta_mean",
        help="Metric used for the primary source winner ranking.",
    )
    parser.add_argument(
        "--min-sources",
        type=int,
        default=None,
        help="Fail unless at least this many distinct source labels are present.",
    )
    parser.add_argument(
        "--require-source",
        dest="required_sources",
        action="append",
        default=[],
        help="Fail unless this checkpoint_source_label is present. May be repeated.",
    )
    parser.add_argument(
        "--min-cases",
        type=int,
        default=None,
        help="Fail unless every candidate includes at least this many cases.",
    )
    parser.add_argument(
        "--require-case",
        dest="required_cases",
        action="append",
        default=[],
        help="Fail unless every candidate includes this case label. May be repeated.",
    )
    parser.add_argument(
        "--require-accepted-all",
        action="store_true",
        help="Fail unless every candidate accepted every case.",
    )
    parser.add_argument(
        "--require-movement-ok-all",
        action="store_true",
        help="Fail unless every candidate observed trainable movement in every case.",
    )
    parser.add_argument(
        "--require-training-policy-scope-match",
        action="store_true",
        help=(
            "Fail unless all source candidates share the same adapter decay, "
            "clipping, accumulation, and FT-control policy."
        ),
    )
    parser.add_argument(
        "--min-accepted-rate",
        type=float,
        default=None,
        help="Fail when any candidate accepted_rate is below this 0..1 floor.",
    )
    parser.add_argument(
        "--min-movement-ok-rate",
        type=float,
        default=None,
        help="Fail when any candidate movement_ok_rate is below this 0..1 floor.",
    )
    parser.add_argument(
        "--min-target-loss-delta-mean",
        type=float,
        default=None,
        help="Fail when any candidate target_loss_delta_mean is below this floor.",
    )
    parser.add_argument(
        "--min-retention-loss-delta-mean",
        type=float,
        default=None,
        help="Fail when any candidate retention_loss_delta_mean is below this floor.",
    )
    parser.add_argument(
        "--min-target-retention-gap-mean",
        type=float,
        default=None,
        help="Fail when target_loss_delta_mean - retention_loss_delta_mean is below this floor.",
    )
    parser.add_argument(
        "--min-target-retention-ratio",
        type=float,
        default=None,
        help="Fail when target_loss_delta_mean / retention_loss_delta_mean is below this floor.",
    )
    parser.add_argument(
        "--min-profile-target-retention-ratio",
        type=float,
        default=None,
        help=(
            "Fail when a selected checkpoint_source_profile row has "
            "target_retention_ratio below this floor. This gates winners "
            "without rejecting weaker comparison candidates."
        ),
    )
    parser.add_argument(
        "--min-retention-accuracy-margin",
        type=float,
        default=None,
        help="Fail when retention_accuracy_margin_min is below this non-negative floor.",
    )
    parser.add_argument(
        "--min-retention-perplexity-margin",
        type=float,
        default=None,
        help="Fail when retention_perplexity_margin_min is absent or below this non-negative floor.",
    )
    args = parser.parse_args()
    if args.min_sources is not None and args.min_sources <= 0:
        parser.error("--min-sources must be positive")
    if args.min_cases is not None and args.min_cases <= 0:
        parser.error("--min-cases must be positive")
    for name in [
        "min_accepted_rate",
        "min_movement_ok_rate",
        "min_target_retention_ratio",
        "min_profile_target_retention_ratio",
        "min_retention_accuracy_margin",
        "min_retention_perplexity_margin",
    ]:
        value = getattr(args, name)
        if value is not None and value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
        if value is not None and name.endswith("_rate") and value > 1.0:
            parser.error(f"--{name.replace('_', '-')} must be at most 1.0")
    if len(set(args.required_sources)) != len(args.required_sources):
        parser.error("--require-source values must be unique")
    if len(set(args.required_cases)) != len(args.required_cases):
        parser.error("--require-case values must be unique")
    if len(set(args.profiles)) != len(args.profiles):
        parser.error("--profile values must be unique")
    return args


def fmt_optional(value):
    if value is None:
        return "none"
    if is_numeric_value(value):
        return f"{float(value):.9f}"
    return str(value)


def checkpoint_source_label(row):
    label = row.get("checkpoint_source_label")
    if isinstance(label, str) and label:
        return label
    preset = row.get("checkpoint_key_preset")
    if isinstance(preset, str) and preset:
        return preset
    raise ValueError(f"aggregate row {row.get('config')} missing checkpoint source label")


def candidate_key(row):
    config = row.get("config")
    if not isinstance(config, str) or not config:
        raise ValueError("source candidate row missing config")
    return f"{checkpoint_source_label(row)}::{config}"


def safe_ratio(numerator, denominator):
    if denominator <= 0.0:
        return None
    return numerator / denominator


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


def training_policy_signature(row):
    return tuple(row.get(field) for field in TRAINING_POLICY_FIELDS)


def training_policy_key(row):
    return "|".join(
        f"{field}={fmt_optional(row.get(field))}" for field in TRAINING_POLICY_FIELDS
    )


def training_policy_description(row):
    return training_policy_key(row).replace("|", ",")


def attach_training_policy_key(row):
    row["training_policy_key"] = training_policy_key(row)
    return row


def training_policy_scope_failures(candidates):
    by_signature = {}
    for row in candidates:
        by_signature.setdefault(training_policy_signature(row), []).append(row)
    if len(by_signature) <= 1:
        return []
    return [
        "training policy scope mismatch: "
        + "; ".join(
            (
                f"{'|'.join(row['source_candidate_key'] for row in rows)}"
                f"[{training_policy_description(rows[0])}]"
            )
            for rows in by_signature.values()
        )
    ]


def source_candidate_from_aggregate(row, source_path):
    validate_aggregate_row(row, str(source_path))
    if row.get("row_type") != "config_aggregate":
        raise ValueError(f"{source_path} expected row_type='config_aggregate'")
    target = numeric_value(row, "target_loss_delta_mean")
    retention = numeric_value(row, "retention_loss_delta_mean")
    candidate = {
        "row_type": "checkpoint_source_candidate",
        "source_candidate_key": candidate_key(row),
        "source_aggregate_path": str(source_path),
        "checkpoint_source_gain": row.get("checkpoint_source_gain", 1.0),
        "target_retention_gap_mean": target - retention,
        "target_retention_ratio": safe_ratio(target, retention),
    }
    for key in [
        "config",
        "base_config",
        "checkpoint_projection_variant",
        "checkpoint_source_gain_variant",
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
        "ft_early_stopped_cases",
        "ft_early_stopped_any",
        "ft_lr_decay_steps_mean",
        "ft_lr_decay_steps_max",
        "ft_final_hyper_learning_rate_min",
        "ft_final_fallback_learning_rate_min",
        "cases",
        "case_labels",
        "accepted_cases",
        "rejected_cases",
        "accepted_rate",
        "accepted_all",
        "movement_ok_cases",
        "movement_not_ok_cases",
        "movement_ok_rate",
        "movement_ok_all",
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
        "target_loss_delta_mean",
        "retention_loss_delta_mean",
        "retention_accuracy_delta_mean",
        "target_loss_margin_mean",
        "target_loss_margin_min",
        "retention_loss_margin_mean",
        "retention_loss_margin_min",
        "retention_accuracy_margin_mean",
        "retention_accuracy_margin_min",
        "retention_perplexity_margin_mean",
        "retention_perplexity_margin_min",
        "checkpoint_key_preset",
        "checkpoint_source_origin",
        "checkpoint_source_label",
        "checkpoint_loaded_files",
        "checkpoint_vocab",
        "checkpoint_hidden",
        "checkpoint_target_classes",
        "checkpoint_overlap_resize",
        "checkpoint_projection",
        "checkpoint_projection_strength",
        "checkpoint_projection_curvature",
        "checkpoint_projection_frequency",
        "checkpoint_source_gain",
    ]:
        if key in row:
            candidate[key] = row[key]
    attach_training_policy_key(candidate)
    return candidate


def source_candidates_from_rows(rows, source_path):
    return [source_candidate_from_aggregate(row, source_path) for row in rows]


def load_source_candidates(paths):
    candidates = []
    seen = set()
    for path in paths:
        for candidate in source_candidates_from_rows(load_jsonl(path), path):
            key = candidate["source_candidate_key"]
            if key in seen:
                raise ValueError(f"duplicate source/config candidate: {key}")
            seen.add(key)
            candidates.append(candidate)
    if not candidates:
        raise ValueError("no checkpoint source candidates loaded")
    return candidates


def rank_value(row, metric):
    value = row.get(metric)
    if value is None:
        return float("-inf")
    if not is_numeric_value(value):
        raise ValueError(f"{row.get('source_candidate_key')} missing numeric {metric}")
    return float(value)


def ranked_candidates(candidates, metric):
    return sorted(
        candidates,
        key=lambda row: (
            rank_value(row, metric),
            rank_value(row, "target_loss_delta_mean"),
            rank_value(row, "retention_loss_delta_mean"),
            row["source_candidate_key"],
        ),
        reverse=True,
    )


def source_winner(candidates, metric):
    ranked = ranked_candidates(candidates, metric)
    if not ranked:
        raise RuntimeError("source compare has no candidates")
    winner = ranked[0]
    return winner, rank_value(winner, metric)


def selected_profile_specs(profile_names):
    names = list(profile_names or [name for name, _metric in PROFILE_METRICS])
    return [(name, PROFILE_METRIC_BY_NAME[name]) for name in names]


def checkpoint_source_flag_fragment(row):
    flags = []
    label = row.get("checkpoint_source_label")
    if isinstance(label, str) and label:
        flags.extend(["--checkpoint-source-label", label])
    key_preset = row.get("checkpoint_key_preset")
    if isinstance(key_preset, str) and key_preset:
        flags.extend(["--key-preset", key_preset])
    if row.get("checkpoint_overlap_resize"):
        flags.append("--allow-overlap-resize")
    projection = row.get("checkpoint_projection")
    if isinstance(projection, str) and projection:
        flags.extend(["--checkpoint-projection", projection])
    for field, flag in [
        ("checkpoint_projection_strength", "--checkpoint-projection-strength"),
        ("checkpoint_projection_curvature", "--checkpoint-projection-curvature"),
        ("checkpoint_projection_frequency", "--checkpoint-projection-frequency"),
    ]:
        value = row.get(field)
        if is_numeric_value(value):
            flags.extend([flag, f"{float(value):g}"])
    gain = row.get("checkpoint_source_gain", 1.0)
    if is_numeric_value(gain):
        flags.extend(["--checkpoint-source-gain", f"{float(gain):g}"])
    weight_decay = row.get("adapter_weight_decay")
    if row.get("adapter_weight_decay_variant") is not None and is_numeric_value(weight_decay):
        flags.extend(["--adapter-weight-decays", f"{float(weight_decay):g}"])
    max_grad_norm = row.get("max_grad_norm")
    if row.get("max_grad_norm_variant") is not None and is_numeric_value(max_grad_norm):
        flags.extend(["--max-grad-norms", f"{float(max_grad_norm):g}"])
    accumulation_steps = row.get("gradient_accumulation_steps")
    if (
        row.get("gradient_accumulation_steps_variant") is not None
        and isinstance(accumulation_steps, int)
        and not isinstance(accumulation_steps, bool)
    ):
        flags.extend(["--gradient-accumulation-steps-list", str(accumulation_steps)])
    if row.get("ft_control_variant") is not None:
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


def source_profile_rows(candidates, profile_names=None):
    rows = []
    for profile, metric in selected_profile_specs(profile_names):
        ranked = ranked_candidates(candidates, metric)
        if not ranked:
            raise RuntimeError("source compare has no candidates")
        winner = ranked[0]
        row = dict(winner)
        row.update(
            {
                "row_type": "checkpoint_source_profile",
                "source_profile": profile,
                "winner_metric": metric,
                "winner_value": rank_value(winner, metric),
                "source_rank": 1,
                "selected_source": checkpoint_source_label(winner),
                "selected_config": winner.get("config"),
                "checkpoint_source_flag_fragment": checkpoint_source_flag_fragment(winner),
            }
        )
        rows.append(row)
    return rows


def check_source_coverage(
    candidates,
    *,
    min_sources=None,
    required_sources=None,
    min_cases=None,
    required_cases=None,
    require_accepted_all=False,
    require_movement_ok_all=False,
    require_training_policy_scope_match=False,
    min_accepted_rate=None,
    min_movement_ok_rate=None,
    min_target_loss_delta_mean=None,
    min_retention_loss_delta_mean=None,
    min_target_retention_gap_mean=None,
    min_target_retention_ratio=None,
    min_retention_accuracy_margin=None,
    min_retention_perplexity_margin=None,
):
    required_sources = list(required_sources or [])
    required_cases = list(required_cases or [])
    sources = sorted({checkpoint_source_label(row) for row in candidates})
    failures = []
    if min_sources is not None and len(sources) < min_sources:
        failures.append(f"sources {len(sources)} below floor {min_sources}")
    missing_sources = [source for source in required_sources if source not in sources]
    if missing_sources:
        failures.append(f"missing sources {','.join(missing_sources)}")
    if require_training_policy_scope_match:
        failures.extend(training_policy_scope_failures(candidates))

    for row in candidates:
        key = row["source_candidate_key"]
        case_labels = aggregate_case_labels(row)
        cases = numeric_value(row, "cases")
        accepted_rate = numeric_value(row, "accepted_rate")
        movement_ok_rate = numeric_value(row, "movement_ok_rate")
        target = numeric_value(row, "target_loss_delta_mean")
        retention = numeric_value(row, "retention_loss_delta_mean")
        gap = numeric_value(row, "target_retention_gap_mean")
        ratio = row.get("target_retention_ratio")
        retention_accuracy_margin = (
            numeric_value(row, "retention_accuracy_margin_min")
            if min_retention_accuracy_margin is not None
            else row.get("retention_accuracy_margin_min")
        )
        retention_perplexity_margin = (
            numeric_value(row, "retention_perplexity_margin_min")
            if min_retention_perplexity_margin is not None
            else row.get("retention_perplexity_margin_min")
        )
        missing_cases = [case for case in required_cases if case not in case_labels]
        print(
            f"source_coverage candidate={key} "
            f"cases={int(cases)} "
            f"case_labels={','.join(case_labels) or 'none'} "
            f"accepted_rate={accepted_rate:.9f} "
            f"accepted_all={summary_bool_value(row, 'accepted_all', False)} "
            f"movement_ok_rate={movement_ok_rate:.9f} "
            f"movement_ok_all={summary_bool_value(row, 'movement_ok_all', False)} "
            f"checkpoint_source_gain={fmt_optional(row.get('checkpoint_source_gain', 1.0))} "
            f"adapter_weight_decay={fmt_optional(row.get('adapter_weight_decay'))} "
            f"max_grad_norm={fmt_optional(row.get('max_grad_norm'))} "
            f"gradient_accumulation_steps={fmt_optional(row.get('gradient_accumulation_steps'))} "
            f"ft_epochs={fmt_optional(row.get('ft_epochs'))} "
            f"target_min_loss_delta={fmt_optional(row.get('target_min_loss_delta_policy'))} "
            f"patience={fmt_optional(row.get('early_stopping_patience'))} "
            f"lr_decay_patience={fmt_optional(row.get('lr_decay_patience'))} "
            f"ft_lr_decay_steps_max={fmt_optional(row.get('ft_lr_decay_steps_max'))} "
            f"target_loss_delta_mean={target:.9f} "
            f"retention_loss_delta_mean={retention:.9f} "
            f"target_retention_gap_mean={gap:.9f} "
            f"target_retention_ratio={fmt_optional(ratio)} "
            f"retention_accuracy_margin_min={fmt_optional(retention_accuracy_margin)} "
            f"retention_perplexity_margin_min={fmt_optional(retention_perplexity_margin)}"
        )
        if min_cases is not None and cases < min_cases:
            failures.append(f"{key}: cases {int(cases)} below floor {min_cases}")
        if missing_cases:
            failures.append(f"{key}: missing cases {','.join(missing_cases)}")
        if require_accepted_all and not summary_bool_value(row, "accepted_all", False):
            failures.append(f"{key}: not all cases accepted")
        if require_movement_ok_all and not summary_bool_value(row, "movement_ok_all", False):
            failures.append(f"{key}: not all cases observed movement")
        if min_accepted_rate is not None and accepted_rate < min_accepted_rate:
            failures.append(
                f"{key}: accepted_rate {accepted_rate:.9f} below floor {min_accepted_rate:.9f}"
            )
        if min_movement_ok_rate is not None and movement_ok_rate < min_movement_ok_rate:
            failures.append(
                f"{key}: movement_ok_rate {movement_ok_rate:.9f} below floor {min_movement_ok_rate:.9f}"
            )
        if min_target_loss_delta_mean is not None and target < min_target_loss_delta_mean:
            failures.append(
                f"{key}: target_loss_delta_mean {target:.9f} below floor {min_target_loss_delta_mean:.9f}"
            )
        if (
            min_retention_loss_delta_mean is not None
            and retention < min_retention_loss_delta_mean
        ):
            failures.append(
                f"{key}: retention_loss_delta_mean {retention:.9f} below floor {min_retention_loss_delta_mean:.9f}"
            )
        if min_target_retention_gap_mean is not None and gap < min_target_retention_gap_mean:
            failures.append(
                f"{key}: target_retention_gap_mean {gap:.9f} below floor {min_target_retention_gap_mean:.9f}"
            )
        if min_target_retention_ratio is not None:
            if ratio is None:
                failures.append(f"{key}: target_retention_ratio unavailable")
            elif ratio < min_target_retention_ratio:
                failures.append(
                    f"{key}: target_retention_ratio {ratio:.9f} below floor {min_target_retention_ratio:.9f}"
                )
        if (
            min_retention_accuracy_margin is not None
            and retention_accuracy_margin < min_retention_accuracy_margin
        ):
            failures.append(
                f"{key}: retention_accuracy_margin_min {retention_accuracy_margin:.9f} below floor {min_retention_accuracy_margin:.9f}"
            )
        if (
            min_retention_perplexity_margin is not None
            and retention_perplexity_margin < min_retention_perplexity_margin
        ):
            failures.append(
                f"{key}: retention_perplexity_margin_min {retention_perplexity_margin:.9f} below floor {min_retention_perplexity_margin:.9f}"
            )

    if failures:
        raise RuntimeError("checkpoint source coverage gate failed: " + "; ".join(failures))
    return len(candidates)


def print_ranking(candidates, metric):
    ranked = ranked_candidates(candidates, metric)
    for rank, row in enumerate(ranked, 1):
        row["source_rank"] = rank
        print(
            f"source_candidate rank={rank} "
            f"metric={metric} "
            f"source={checkpoint_source_label(row)} "
            f"key_preset={row.get('checkpoint_key_preset')} "
            f"config={row.get('config')} "
            f"checkpoint_source_gain={fmt_optional(row.get('checkpoint_source_gain', 1.0))} "
            f"adapter_weight_decay={fmt_optional(row.get('adapter_weight_decay'))} "
            f"max_grad_norm={fmt_optional(row.get('max_grad_norm'))} "
            f"gradient_accumulation_steps={fmt_optional(row.get('gradient_accumulation_steps'))} "
            f"ft_epochs={fmt_optional(row.get('ft_epochs'))} "
            f"target_min_loss_delta={fmt_optional(row.get('target_min_loss_delta_policy'))} "
            f"patience={fmt_optional(row.get('early_stopping_patience'))} "
            f"lr_decay_patience={fmt_optional(row.get('lr_decay_patience'))} "
            f"cases={row.get('cases')} "
            f"case_labels={row.get('case_labels')} "
            f"accepted_rate={numeric_value(row, 'accepted_rate'):.9f} "
            f"movement_ok_rate={numeric_value(row, 'movement_ok_rate'):.9f} "
            f"target_loss_delta_mean={numeric_value(row, 'target_loss_delta_mean'):.9f} "
            f"retention_loss_delta_mean={numeric_value(row, 'retention_loss_delta_mean'):.9f} "
            f"target_retention_gap_mean={numeric_value(row, 'target_retention_gap_mean'):.9f} "
            f"target_retention_ratio={fmt_optional(row.get('target_retention_ratio'))} "
            f"retention_accuracy_margin_min={fmt_optional(row.get('retention_accuracy_margin_min'))} "
            f"retention_perplexity_margin_min={fmt_optional(row.get('retention_perplexity_margin_min'))}"
        )
    winner, value = source_winner(candidates, metric)
    print(
        f"source_winner metric={metric} "
        f"source={checkpoint_source_label(winner)} "
        f"config={winner.get('config')} "
        f"value={value:.9f}"
    )
    if metric != "target_retention_gap_mean":
        gap_winner, gap_value = source_winner(candidates, "target_retention_gap_mean")
        print(
            "source_selectivity_winner "
            "metric=target_retention_gap_mean "
            f"source={checkpoint_source_label(gap_winner)} "
            f"config={gap_winner.get('config')} "
            f"value={gap_value:.9f}"
        )
    return ranked


def print_source_profiles(rows):
    for row in rows:
        print(
            f"source_profile profile={row['source_profile']} "
            f"metric={row['winner_metric']} "
            f"source={row['selected_source']} "
            f"config={row.get('selected_config')} "
            f"checkpoint_source_gain={fmt_optional(row.get('checkpoint_source_gain', 1.0))} "
            f"adapter_weight_decay={fmt_optional(row.get('adapter_weight_decay'))} "
            f"max_grad_norm={fmt_optional(row.get('max_grad_norm'))} "
            f"gradient_accumulation_steps={fmt_optional(row.get('gradient_accumulation_steps'))} "
            f"ft_epochs={fmt_optional(row.get('ft_epochs'))} "
            f"target_min_loss_delta={fmt_optional(row.get('target_min_loss_delta_policy'))} "
            f"patience={fmt_optional(row.get('early_stopping_patience'))} "
            f"lr_decay_patience={fmt_optional(row.get('lr_decay_patience'))} "
            f"winner_value={row['winner_value']:.9f} "
            f"guard_epoch_counts_available_all={row.get('guard_epoch_counts_available_all')} "
            f"guard_accepted_epochs_mean={fmt_optional(row.get('guard_accepted_epochs_mean'))} "
            f"guard_retention_rejected_epochs_mean={fmt_optional(row.get('guard_retention_rejected_epochs_mean'))} "
            f"guard_target_stale_epochs_mean={fmt_optional(row.get('guard_target_stale_epochs_mean'))} "
            f"target_loss_delta_mean={numeric_value(row, 'target_loss_delta_mean'):.9f} "
            f"retention_loss_delta_mean={numeric_value(row, 'retention_loss_delta_mean'):.9f} "
            f"target_retention_gap_mean={numeric_value(row, 'target_retention_gap_mean'):.9f} "
            f"target_retention_ratio={fmt_optional(row.get('target_retention_ratio'))} "
            f"retention_accuracy_margin_min={fmt_optional(row.get('retention_accuracy_margin_min'))} "
            f"retention_perplexity_margin_min={fmt_optional(row.get('retention_perplexity_margin_min'))}"
        )


def check_source_profile_gates(rows, *, min_profile_target_retention_ratio=None):
    failures = []
    for row in rows:
        profile = row.get("source_profile")
        ratio = row.get("target_retention_ratio")
        if min_profile_target_retention_ratio is None:
            continue
        if ratio is None:
            failures.append(f"{profile}: target_retention_ratio unavailable")
            print(
                f"source_profile_gate profile={profile} "
                "target_retention_ratio=none "
                f"min_target_retention_ratio={min_profile_target_retention_ratio:.9f} "
                "passed=False"
            )
            continue
        if not is_numeric_value(ratio):
            raise ValueError(f"profile {profile} missing numeric target_retention_ratio")
        passed = float(ratio) >= min_profile_target_retention_ratio
        print(
            f"source_profile_gate profile={profile} "
            f"target_retention_ratio={float(ratio):.9f} "
            f"min_target_retention_ratio={min_profile_target_retention_ratio:.9f} "
            f"passed={passed}"
        )
        if not passed:
            failures.append(
                f"{profile}: target_retention_ratio {float(ratio):.9f} below floor "
                f"{min_profile_target_retention_ratio:.9f}"
            )
    if failures:
        raise RuntimeError("checkpoint source profile gate failed: " + "; ".join(failures))
    return len(rows)


def main():
    args = parse_args()
    candidates = load_source_candidates(args.aggregate_jsonls)
    check_source_coverage(
        candidates,
        min_sources=args.min_sources,
        required_sources=args.required_sources,
        min_cases=args.min_cases,
        required_cases=args.required_cases,
        require_accepted_all=args.require_accepted_all,
        require_movement_ok_all=args.require_movement_ok_all,
        require_training_policy_scope_match=args.require_training_policy_scope_match,
        min_accepted_rate=args.min_accepted_rate,
        min_movement_ok_rate=args.min_movement_ok_rate,
        min_target_loss_delta_mean=args.min_target_loss_delta_mean,
        min_retention_loss_delta_mean=args.min_retention_loss_delta_mean,
        min_target_retention_gap_mean=args.min_target_retention_gap_mean,
        min_target_retention_ratio=args.min_target_retention_ratio,
        min_retention_accuracy_margin=args.min_retention_accuracy_margin,
        min_retention_perplexity_margin=args.min_retention_perplexity_margin,
    )
    ranked = print_ranking(candidates, args.winner_metric)
    profiles = source_profile_rows(candidates, args.profiles)
    check_source_profile_gates(
        profiles,
        min_profile_target_retention_ratio=args.min_profile_target_retention_ratio,
    )
    if args.profile_jsonl is not None or args.profiles:
        print_source_profiles(profiles)
    if args.jsonl is not None:
        write_jsonl(args.jsonl, ranked)
        print(f"source_compare_jsonl={args.jsonl} rows={len(ranked)}")
    if args.profile_jsonl is not None:
        write_jsonl(args.profile_jsonl, profiles)
        print(f"source_profile_jsonl={args.profile_jsonl} rows={len(profiles)}")


if __name__ == "__main__":
    main()
