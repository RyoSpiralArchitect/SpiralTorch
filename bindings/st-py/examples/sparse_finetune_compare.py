import json

from spiraltorch.nn import compare_sparse_finetune_summaries


def add_summary_compare_args(parser, *, subject):
    parser.add_argument(
        "--max-target-loss-regression",
        type=float,
        default=None,
        help=(
            "Fail when target_loss_delta regresses from --compare-jsonl by more "
            "than this non-negative tolerance."
        ),
    )
    parser.add_argument(
        "--max-retention-loss-regression",
        type=float,
        default=None,
        help=(
            "Fail when retention_loss_delta regresses from --compare-jsonl by more "
            "than this non-negative tolerance."
        ),
    )
    parser.add_argument(
        "--max-target-retention-gap-regression",
        type=float,
        default=None,
        help=(
            "Fail when target_retention_gap regresses from --compare-jsonl by "
            "more than this non-negative tolerance."
        ),
    )
    parser.add_argument(
        "--max-target-retention-ratio-regression",
        type=float,
        default=None,
        help=(
            "Fail when target_retention_ratio regresses from --compare-jsonl by "
            "more than this non-negative tolerance."
        ),
    )
    parser.add_argument(
        "--min-target-loss-margin",
        type=float,
        default=None,
        help=(
            "Fail when current target_loss_margin is below this non-negative "
            "floor."
        ),
    )
    parser.add_argument(
        "--min-target-retention-ratio",
        type=float,
        default=None,
        help=(
            "Fail when current target_retention_ratio is absent or below this "
            "non-negative floor."
        ),
    )
    parser.add_argument(
        "--min-retention-loss-margin",
        type=float,
        default=None,
        help=(
            "Fail when current retention_loss_margin is below this non-negative "
            "floor."
        ),
    )
    parser.add_argument(
        "--min-retention-accuracy-margin",
        type=float,
        default=None,
        help=(
            "Fail when current retention_accuracy_margin is below this "
            "non-negative floor."
        ),
    )
    parser.add_argument(
        "--min-retention-perplexity-margin",
        type=float,
        default=None,
        help=(
            "Fail when current retention_perplexity_margin is absent or below "
            "this non-negative floor."
        ),
    )
    parser.add_argument(
        "--require-status-match",
        action="store_true",
        help=f"Fail when a compared {subject} changes SparseFineTuneReport status.",
    )
    parser.add_argument(
        "--require-accepted-match",
        action="store_true",
        help=f"Fail when a compared {subject} changes sparse FT guard acceptance.",
    )
    parser.add_argument(
        "--require-guard-match",
        action="store_true",
        help="Fail when compared sparse retention guard settings differ.",
    )
    parser.add_argument(
        "--require-movement-tolerance-match",
        action="store_true",
        help="Fail when compared parameter movement audit tolerances differ.",
    )
    parser.add_argument(
        "--require-resume-match",
        action="store_true",
        help="Fail when compared FT-ready resume fingerprints differ.",
    )
    parser.add_argument(
        "--require-checkpoint-match",
        action="store_true",
        help=(
            "Fail when compared checkpoint preflight/load audit fields differ. "
            "Rows without checkpoint audit fields are unaffected."
        ),
    )


def summary_compare_gate_requested(args):
    return (
        args.max_target_loss_regression is not None
        or args.max_retention_loss_regression is not None
        or args.max_target_retention_gap_regression is not None
        or args.max_target_retention_ratio_regression is not None
        or args.min_target_loss_margin is not None
        or args.min_target_retention_ratio is not None
        or args.min_retention_loss_margin is not None
        or args.min_retention_accuracy_margin is not None
        or args.min_retention_perplexity_margin is not None
        or args.require_status_match
        or args.require_accepted_match
        or args.require_guard_match
        or args.require_movement_tolerance_match
        or args.require_resume_match
        or getattr(args, "require_checkpoint_match", False)
    )


def validate_summary_compare_args(parser, args):
    if args.max_target_loss_regression is not None and args.max_target_loss_regression < 0.0:
        parser.error("--max-target-loss-regression must be non-negative")
    if (
        args.max_retention_loss_regression is not None
        and args.max_retention_loss_regression < 0.0
    ):
        parser.error("--max-retention-loss-regression must be non-negative")
    if (
        args.max_target_retention_gap_regression is not None
        and args.max_target_retention_gap_regression < 0.0
    ):
        parser.error("--max-target-retention-gap-regression must be non-negative")
    if (
        args.max_target_retention_ratio_regression is not None
        and args.max_target_retention_ratio_regression < 0.0
    ):
        parser.error("--max-target-retention-ratio-regression must be non-negative")
    for name in [
        "min_target_loss_margin",
        "min_target_retention_ratio",
        "min_retention_loss_margin",
        "min_retention_accuracy_margin",
        "min_retention_perplexity_margin",
    ]:
        value = getattr(args, name)
        if value is not None and value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    return summary_compare_gate_requested(args)


def compare_summaries(
    current,
    baseline,
    *,
    max_target_loss_regression,
    max_retention_loss_regression,
    min_target_loss_margin,
    min_retention_loss_margin,
    min_retention_accuracy_margin,
    min_retention_perplexity_margin,
    require_status_match,
    require_accepted_match,
    require_guard_match,
    require_movement_tolerance_match,
    require_resume_match,
    max_target_retention_gap_regression=None,
    max_target_retention_ratio_regression=None,
    min_target_retention_ratio=None,
):
    kwargs = {
        "max_target_loss_regression": max_target_loss_regression,
        "max_retention_loss_regression": max_retention_loss_regression,
        "max_target_retention_gap_regression": max_target_retention_gap_regression,
        "max_target_retention_ratio_regression": max_target_retention_ratio_regression,
        "min_target_loss_margin": min_target_loss_margin,
        "min_target_retention_ratio": min_target_retention_ratio,
        "min_retention_loss_margin": min_retention_loss_margin,
        "min_retention_accuracy_margin": min_retention_accuracy_margin,
        "min_retention_perplexity_margin": min_retention_perplexity_margin,
        "require_status_match": require_status_match,
        "require_accepted_match": require_accepted_match,
        "require_guard_match": require_guard_match,
        "require_movement_tolerance_match": require_movement_tolerance_match,
        "require_resume_match": require_resume_match,
    }
    try:
        comparison = compare_sparse_finetune_summaries(current, baseline, **kwargs)
    except TypeError as exc:
        local_gate_keys = {
            "require_accepted_match",
            "min_target_loss_margin",
            "min_retention_loss_margin",
            "min_retention_accuracy_margin",
            "min_retention_perplexity_margin",
            "max_target_retention_gap_regression",
            "max_target_retention_ratio_regression",
            "min_target_retention_ratio",
        }
        if not any(key in str(exc) for key in local_gate_keys):
            raise
        # Older local wheels predate some Rust-side gates; keep the CLI gates usable.
        for key in local_gate_keys:
            kwargs.pop(key, None)
        comparison = compare_sparse_finetune_summaries(current, baseline, **kwargs)
    if require_accepted_match and comparison["accepted_changed"]:
        comparison = dict(comparison)
        comparison["passed"] = False
    comparison = dict(comparison)
    for key, default in [
        ("target_loss_regression", 0.0),
        ("retention_loss_regression", 0.0),
        ("target_retention_gap_regression", 0.0),
        ("target_retention_ratio_regression", None),
        ("status_changed", False),
        ("accepted_changed", False),
        ("guard_changed", False),
        ("movement_tolerance_changed", False),
        ("resume_changed", False),
    ]:
        comparison.setdefault(key, default)
    current_margins = summary_guard_margins(current)
    baseline_margins = summary_guard_margins(baseline)
    for key, value in current_margins.items():
        comparison.setdefault(f"current_{key}", value)
    for key, value in baseline_margins.items():
        comparison.setdefault(f"baseline_{key}", value)
    current_selectivity = summary_target_retention_selectivity(current)
    baseline_selectivity = summary_target_retention_selectivity(baseline)
    for key, value in current_selectivity.items():
        comparison[f"current_{key}"] = value
    for key, value in baseline_selectivity.items():
        comparison[f"baseline_{key}"] = value
    gap_change = current_selectivity["target_retention_gap"] - baseline_selectivity[
        "target_retention_gap"
    ]
    comparison["target_retention_gap_change"] = gap_change
    comparison["target_retention_gap_regression"] = max(0.0, -gap_change)
    current_ratio = current_selectivity["target_retention_ratio"]
    baseline_ratio = baseline_selectivity["target_retention_ratio"]
    if current_ratio is not None and baseline_ratio is not None:
        ratio_change = current_ratio - baseline_ratio
        comparison["target_retention_ratio_change"] = ratio_change
        comparison["target_retention_ratio_regression"] = max(0.0, -ratio_change)
    else:
        comparison["target_retention_ratio_change"] = None
        comparison["target_retention_ratio_regression"] = None
    if (
        max_target_retention_gap_regression is not None
        and comparison["target_retention_gap_regression"]
        > max_target_retention_gap_regression
    ):
        comparison["passed"] = False
    if max_target_retention_ratio_regression is not None:
        ratio_regression = comparison["target_retention_ratio_regression"]
        if ratio_regression is None or ratio_regression > max_target_retention_ratio_regression:
            comparison["passed"] = False
    if min_target_retention_ratio is None:
        comparison["target_retention_ratio_shortfall"] = (
            None if current_ratio is None else 0.0
        )
    elif current_ratio is None:
        comparison["target_retention_ratio_shortfall"] = None
        comparison["passed"] = False
    else:
        shortfall = max(0.0, min_target_retention_ratio - current_ratio)
        comparison["target_retention_ratio_shortfall"] = shortfall
        if current_ratio < min_target_retention_ratio:
            comparison["passed"] = False
    for key, floor in [
        ("target_loss_margin", min_target_loss_margin),
        ("retention_loss_margin", min_retention_loss_margin),
        ("retention_accuracy_margin", min_retention_accuracy_margin),
        ("retention_perplexity_margin", min_retention_perplexity_margin),
    ]:
        shortfall_key = f"{key}_shortfall"
        current_value = current_margins[key]
        if floor is None:
            comparison.setdefault(shortfall_key, None if current_value is None else 0.0)
            continue
        if current_value is None:
            comparison[shortfall_key] = None
            comparison["passed"] = False
            continue
        shortfall = max(0.0, floor - current_value)
        comparison.setdefault(shortfall_key, shortfall)
        if current_value < floor:
            comparison["passed"] = False
    return comparison


def is_numeric_value(value):
    return not isinstance(value, bool) and isinstance(value, (int, float))


def summary_numeric_value(summary, key, default=0.0):
    value = summary.get(key, default)
    if not is_numeric_value(value):
        raise ValueError(f"summary {key} is not numeric")
    return float(value)


def optional_summary_numeric_value(summary, key):
    value = summary.get(key)
    if value is None:
        return None
    if not is_numeric_value(value):
        raise ValueError(f"summary {key} is not numeric")
    return float(value)


_MISSING = object()


def summary_bool_value(summary, key, default=_MISSING):
    if key in summary:
        value = summary[key]
    elif default is not _MISSING:
        value = default
    else:
        raise ValueError(f"summary {key} is missing")
    if not isinstance(value, bool):
        raise ValueError(f"summary {key} is not boolean")
    return value


def summary_guard_margins(summary):
    target_loss_delta = summary_numeric_value(summary, "target_loss_delta")
    target_min_loss_delta = summary_numeric_value(summary, "target_min_loss_delta")
    best_retention_loss_increase = summary_numeric_value(
        summary, "best_retention_loss_increase"
    )
    best_retention_accuracy_drop = summary_numeric_value(
        summary, "best_retention_accuracy_drop"
    )
    retention_max_loss_increase = summary_numeric_value(
        summary, "retention_max_loss_increase"
    )
    retention_max_accuracy_drop = summary_numeric_value(
        summary, "retention_max_accuracy_drop"
    )
    retention_perplexity_margin = optional_summary_numeric_value(
        summary, "retention_perplexity_margin"
    )
    if retention_perplexity_margin is None:
        retention_max_perplexity_increase = summary.get(
            "retention_max_perplexity_increase"
        )
        if retention_max_perplexity_increase is not None:
            retention_perplexity_margin = summary_numeric_value(
                summary, "retention_max_perplexity_increase"
            ) - summary_numeric_value(
                summary, "best_retention_perplexity_increase"
            )
    return {
        "target_loss_margin": (
            optional_summary_numeric_value(summary, "target_loss_margin")
            if "target_loss_margin" in summary
            else target_loss_delta - target_min_loss_delta
        ),
        "retention_loss_margin": (
            optional_summary_numeric_value(summary, "retention_loss_margin")
            if "retention_loss_margin" in summary
            else retention_max_loss_increase - best_retention_loss_increase
        ),
        "retention_accuracy_margin": (
            optional_summary_numeric_value(summary, "retention_accuracy_margin")
            if "retention_accuracy_margin" in summary
            else retention_max_accuracy_drop - best_retention_accuracy_drop
        ),
        "retention_perplexity_margin": retention_perplexity_margin,
    }


def summary_target_retention_selectivity(summary):
    target_loss_delta = summary_numeric_value(summary, "target_loss_delta")
    retention_loss_delta = summary_numeric_value(summary, "retention_loss_delta")
    gap = (
        optional_summary_numeric_value(summary, "target_retention_gap")
        if "target_retention_gap" in summary
        else target_loss_delta - retention_loss_delta
    )
    ratio = (
        optional_summary_numeric_value(summary, "target_retention_ratio")
        if "target_retention_ratio" in summary
        else (
            target_loss_delta / retention_loss_delta
            if retention_loss_delta > 0.0
            else None
        )
    )
    return {
        "target_retention_gap": gap,
        "target_retention_ratio": ratio,
    }


def attach_summary_guard_margins(row):
    row.update(summary_guard_margins(row))
    return row


def summary_guard_counts(summary, captured=None):
    def captured_value(key):
        if captured is None or not hasattr(captured, key):
            return None
        return getattr(captured, key)

    def summary_or_captured(key):
        value = summary.get(key)
        return value if value is not None else captured_value(key)

    accepted_epochs = summary_or_captured("guard_accepted_epochs")
    retention_rejected_epochs = summary_or_captured("guard_retention_rejected_epochs")
    target_stale_epochs = summary_or_captured("guard_target_stale_epochs")
    counts_available = (
        accepted_epochs is not None
        and retention_rejected_epochs is not None
        and target_stale_epochs is not None
    )
    guarded_best_epoch = summary.get("guarded_best_epoch")
    if guarded_best_epoch is None:
        guarded_best_epoch = captured_value("guarded_best_epoch")
    accepted = summary.get("accepted")
    if not isinstance(accepted, bool):
        accepted = guarded_best_epoch is not None
    if accepted_epochs is None:
        accepted_epochs = 1 if accepted or guarded_best_epoch is not None else 0
    if retention_rejected_epochs is None:
        retention_rejected_epochs = 0
    if target_stale_epochs is None:
        target_stale_epochs = 0
    guard_epochs_run = summary_or_captured("guard_epochs_run")
    if guard_epochs_run is None:
        guard_epochs_run = summary.get("epochs_run")
    if guard_epochs_run is None:
        guard_epochs_run = (
            int(accepted_epochs)
            + int(retention_rejected_epochs)
            + int(target_stale_epochs)
        )

    def summary_rate_or_count_ratio(rate_key, count):
        value = summary.get(rate_key)
        if value is not None:
            return optional_summary_numeric_value(summary, rate_key)
        if not guard_epochs_run:
            return 0.0
        return float(count) / float(guard_epochs_run)

    return {
        "guard_epochs_run": int(guard_epochs_run),
        "guard_accepted_epochs": int(accepted_epochs),
        "guard_retention_rejected_epochs": int(retention_rejected_epochs),
        "guard_target_stale_epochs": int(target_stale_epochs),
        "guard_acceptance_rate": summary_rate_or_count_ratio(
            "guard_acceptance_rate",
            accepted_epochs,
        ),
        "guard_retention_rejected_rate": summary_rate_or_count_ratio(
            "guard_retention_rejected_rate",
            retention_rejected_epochs,
        ),
        "guard_target_stale_rate": summary_rate_or_count_ratio(
            "guard_target_stale_rate",
            target_stale_epochs,
        ),
        "guard_epoch_counts_available": counts_available,
    }


def attach_summary_guard_counts(row, captured=None):
    row.update(summary_guard_counts(row, captured))
    return row


CHECKPOINT_AUDIT_PREFIXES = ["base", "embed", "head"]
CHECKPOINT_AUDIT_SUFFIXES = [
    "preflight_matched",
    "preflight_extra",
    "preflight_source_hash",
    "preflight_matched_subset_hash",
    "preflight_signature",
    "load_matched",
    "load_source_hash",
    "load_loaded_hash",
]
CHECKPOINT_AUDIT_FIELDS = [
    f"{prefix}_{suffix}"
    for prefix in CHECKPOINT_AUDIT_PREFIXES
    for suffix in CHECKPOINT_AUDIT_SUFFIXES
] + [
    "checkpoint_key_preset",
    "checkpoint_source_origin",
    "checkpoint_loaded_files",
    "checkpoint_vocab",
    "checkpoint_hidden",
    "checkpoint_target_classes",
    "checkpoint_overlap_resize",
    "checkpoint_projection",
    "checkpoint_projection_strength",
    "checkpoint_projection_curvature",
    "checkpoint_projection_frequency",
]
MISSING_AUDIT_FIELD = "<missing>"


def checkpoint_audit_differences(current, baseline):
    differences = []
    for field in CHECKPOINT_AUDIT_FIELDS:
        current_has = field in current
        baseline_has = field in baseline
        if not current_has and not baseline_has:
            continue
        if current_has != baseline_has:
            differences.append(
                (
                    field,
                    baseline.get(field) if baseline_has else MISSING_AUDIT_FIELD,
                    current.get(field) if current_has else MISSING_AUDIT_FIELD,
                )
            )
            continue
        current_value = current.get(field)
        baseline_value = baseline.get(field)
        if current_value != baseline_value:
            differences.append((field, baseline_value, current_value))
    return differences


def checkpoint_audit_failures(label, current, baseline):
    return [
        f"{label}: checkpoint audit {field} changed from {before!r} to {after!r}"
        for field, before, after in checkpoint_audit_differences(current, baseline)
    ]


def write_summary_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def load_summary_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} invalid JSONL row: {exc}") from exc
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} did not contain any summary rows")
    return rows


def load_single_summary_jsonl(path):
    rows = load_summary_jsonl(path)
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one summary row, found {len(rows)}")
    return rows[0]


def compare_single_summary(current, baseline, args, *, label=None, failure_prefix="summary"):
    comparison = compare_summaries(
        current,
        baseline,
        max_target_loss_regression=args.max_target_loss_regression,
        max_retention_loss_regression=args.max_retention_loss_regression,
        max_target_retention_gap_regression=getattr(
            args,
            "max_target_retention_gap_regression",
            None,
        ),
        max_target_retention_ratio_regression=getattr(
            args,
            "max_target_retention_ratio_regression",
            None,
        ),
        min_target_loss_margin=args.min_target_loss_margin,
        min_target_retention_ratio=getattr(args, "min_target_retention_ratio", None),
        min_retention_loss_margin=args.min_retention_loss_margin,
        min_retention_accuracy_margin=args.min_retention_accuracy_margin,
        min_retention_perplexity_margin=args.min_retention_perplexity_margin,
        require_status_match=args.require_status_match,
        require_accepted_match=args.require_accepted_match,
        require_guard_match=args.require_guard_match,
        require_movement_tolerance_match=args.require_movement_tolerance_match,
        require_resume_match=args.require_resume_match,
    )
    compare_label = label or current.get("config") or current.get("example") or "summary"
    failures = summary_compare_failures(
        compare_label,
        comparison,
        max_target_loss_regression=args.max_target_loss_regression,
        max_retention_loss_regression=args.max_retention_loss_regression,
        max_target_retention_gap_regression=getattr(
            args,
            "max_target_retention_gap_regression",
            None,
        ),
        max_target_retention_ratio_regression=getattr(
            args,
            "max_target_retention_ratio_regression",
            None,
        ),
        min_target_loss_margin=args.min_target_loss_margin,
        min_target_retention_ratio=getattr(args, "min_target_retention_ratio", None),
        min_retention_loss_margin=args.min_retention_loss_margin,
        min_retention_accuracy_margin=args.min_retention_accuracy_margin,
        min_retention_perplexity_margin=args.min_retention_perplexity_margin,
        require_status_match=args.require_status_match,
        require_accepted_match=args.require_accepted_match,
        require_guard_match=args.require_guard_match,
        require_movement_tolerance_match=args.require_movement_tolerance_match,
        require_resume_match=args.require_resume_match,
    )
    checkpoint_changed = bool(checkpoint_audit_differences(current, baseline))
    if getattr(args, "require_checkpoint_match", False):
        failures.extend(checkpoint_audit_failures(compare_label, current, baseline))
    print(
        "summary_compare "
        f"config={compare_label} "
        f"target_loss_regression={comparison['target_loss_regression']:.9f} "
        f"retention_loss_regression={comparison['retention_loss_regression']:.9f} "
        f"target_retention_gap_regression={comparison['target_retention_gap_regression']:.9f} "
        f"target_retention_ratio_regression={comparison['target_retention_ratio_regression']} "
        f"status_changed={comparison['status_changed']} "
        f"accepted_changed={comparison['accepted_changed']} "
        f"checkpoint_changed={checkpoint_changed} "
        f"passed={comparison['passed']}"
    )
    if failures or not comparison["passed"]:
        details = "; ".join(failures) if failures else "comparison returned passed=false"
        raise RuntimeError(f"{failure_prefix} summary regression gate failed: {details}")
    return comparison


def summary_compare_failures(
    label,
    comparison,
    *,
    max_target_loss_regression=None,
    max_retention_loss_regression=None,
    max_target_retention_gap_regression=None,
    max_target_retention_ratio_regression=None,
    min_target_loss_margin=None,
    min_target_retention_ratio=None,
    min_retention_loss_margin=None,
    min_retention_accuracy_margin=None,
    min_retention_perplexity_margin=None,
    require_status_match=False,
    require_accepted_match=False,
    require_guard_match=False,
    require_movement_tolerance_match=False,
    require_resume_match=False,
):
    failures = []
    if comparison["status_changed"] and require_status_match:
        failures.append(
            f"{label}: status changed from {comparison['baseline_status']} "
            f"to {comparison['current_status']}"
        )
    if comparison["accepted_changed"] and require_accepted_match:
        failures.append(f"{label}: sparse FT guard acceptance changed")
    if comparison["guard_changed"] and require_guard_match:
        failures.append(f"{label}: sparse retention guard settings changed")
    if comparison["movement_tolerance_changed"] and require_movement_tolerance_match:
        failures.append(f"{label}: movement audit tolerance changed")
    if comparison["resume_changed"] and require_resume_match:
        failures.append(f"{label}: FT-ready resume fingerprint changed")
    if (
        max_target_loss_regression is not None
        and comparison["target_loss_regression"] > max_target_loss_regression
    ):
        failures.append(
            f"{label}: target_loss_delta regressed by "
            f"{comparison['target_loss_regression']:.9f}"
        )
    if (
        max_retention_loss_regression is not None
        and comparison["retention_loss_regression"] > max_retention_loss_regression
    ):
        failures.append(
            f"{label}: retention_loss_delta regressed by "
            f"{comparison['retention_loss_regression']:.9f}"
        )
    if (
        max_target_retention_gap_regression is not None
        and comparison["target_retention_gap_regression"]
        > max_target_retention_gap_regression
    ):
        failures.append(
            f"{label}: target_retention_gap regressed by "
            f"{comparison['target_retention_gap_regression']:.9f}"
        )
    if max_target_retention_ratio_regression is not None:
        ratio_regression = comparison["target_retention_ratio_regression"]
        if ratio_regression is None:
            failures.append(f"{label}: target_retention_ratio regression is unavailable")
        elif ratio_regression > max_target_retention_ratio_regression:
            failures.append(
                f"{label}: target_retention_ratio regressed by "
                f"{ratio_regression:.9f}"
            )
    if min_target_retention_ratio is not None:
        current_ratio = comparison.get("current_target_retention_ratio")
        if current_ratio is None:
            failures.append(f"{label}: target_retention_ratio is unavailable")
        elif current_ratio < min_target_retention_ratio:
            failures.append(
                f"{label}: target_retention_ratio {current_ratio:.9f} "
                f"below floor {min_target_retention_ratio:.9f}"
            )
    for key, floor, description in [
        ("target_loss_margin", min_target_loss_margin, "target loss"),
        ("retention_loss_margin", min_retention_loss_margin, "retention loss"),
        (
            "retention_accuracy_margin",
            min_retention_accuracy_margin,
            "retention accuracy",
        ),
        (
            "retention_perplexity_margin",
            min_retention_perplexity_margin,
            "retention perplexity",
        ),
    ]:
        if floor is None:
            continue
        current_value = comparison.get(f"current_{key}")
        if current_value is None:
            failures.append(f"{label}: {description} margin is unavailable")
        elif current_value < floor:
            failures.append(
                f"{label}: {description} margin {current_value:.9f} "
                f"below floor {floor:.9f}"
            )
    return failures
