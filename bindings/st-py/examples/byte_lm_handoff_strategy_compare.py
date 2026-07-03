import argparse
from pathlib import Path

import spiraltorch as st
from spiraltorch.nn import (
    Linear,
    LoraLinear,
    Relu,
    Sequential,
    SoftmaxCrossEntropy,
    ZSpaceProjector,
)

from byte_lm_mlp_lora_adapter import (
    ACCUMULATION_STEPS,
    BATCH_WINDOWS,
    CONTEXT,
    FT_EPOCHS,
    FT_MOVEMENT_TOLERANCE,
    HIDDEN,
    LORA_ALPHA,
    LORA_RANK,
    VOCAB,
    evaluate,
    loader,
    metric,
    train_source,
)
from checkpoint_preflight import (
    HF_KEY_PRESETS,
    HF_UNUSED_KEYS,
    checkpoint_audit_fields,
    hf_lm_handoff_from_external_state,
    hf_lm_key_preset,
    hf_lm_overlap_resize_kwargs,
    infer_hf_lm_module_shapes,
    preflight_and_load,
)
from sparse_finetune_compare import (
    add_summary_compare_args,
    attach_summary_guard_counts,
    attach_summary_guard_margins,
    checkpoint_audit_differences,
    checkpoint_audit_failures,
    compare_summaries,
    load_summary_jsonl,
    summary_bool_value,
    summary_compare_failures,
    summary_guard_margins,
    validate_summary_compare_args,
    write_summary_jsonl,
)


ZSPACE_STRENGTH = 0.5
ZSPACE_CURVATURE = -1.0
ZSPACE_FREQUENCY = 0.65
ZSPACE_PRESETS = {
    "healthy": {
        "zspace_strength": 1.0,
        "zspace_curvature": -0.04,
        "zspace_frequency": ZSPACE_FREQUENCY,
    }
}
DEFAULT_CASE_LABEL = "adapter_ja"

CASE_SPECS = [
    {
        "label": DEFAULT_CASE_LABEL,
        "source_text": "spiraltorch adapters inherit byte memories; adapters inherit byte memories",
        "target_text": "螺旋adapterは小さくFTできるbyte",
    },
    {
        "label": "route_cats",
        "source_text": "graphs route tensors; routes keep tensor memory; routes graph byte",
        "target_text": "猫byte螺旋byte route猫byte",
    },
    {
        "label": "geometry_tokens",
        "source_text": "hyperbolic geometry keeps local token memories aligned",
        "target_text": "zspace geometry adapter token route",
    },
]

STRATEGY_SPECS = [
    {
        "label": "hf_exact",
        "allow_overlap_resize": False,
        "extra_vocab_rows": 0,
        "extra_hidden_cols": 0,
        "extra_head_rows": 0,
        "projection": "none",
    },
    {
        "label": "hf_overlap_resize",
        "allow_overlap_resize": True,
        "extra_vocab_rows": 4,
        "extra_hidden_cols": 1,
        "extra_head_rows": 3,
        "projection": "none",
    },
    {
        "label": "hf_zspace_projected",
        "allow_overlap_resize": False,
        "extra_vocab_rows": 0,
        "extra_hidden_cols": 0,
        "extra_head_rows": 0,
        "projection": "zspace",
        "zspace_strength": ZSPACE_STRENGTH,
        "zspace_curvature": ZSPACE_CURVATURE,
        "zspace_frequency": ZSPACE_FREQUENCY,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare exact, overlap-resized, and Z-space-projected HF handoff strategies."
    )
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        choices=[case["label"] for case in CASE_SPECS],
        help="Run only this target corpus case. May be repeated.",
    )
    parser.add_argument(
        "--strategy",
        dest="strategies",
        action="append",
        choices=[spec["label"] for spec in STRATEGY_SPECS],
        help=(
            "Run only this handoff strategy. May be repeated; hf_exact is added "
            "automatically when comparing another strategy."
        ),
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Optional path for flat SparseFineTuneReport summary rows.",
    )
    parser.add_argument(
        "--aggregate-jsonl",
        type=Path,
        default=None,
        help="Optional path for strategy-level aggregate metric rows.",
    )
    parser.add_argument(
        "--compare-jsonl",
        type=Path,
        default=None,
        help="Optional previous strategy summary JSONL to compare against this run.",
    )
    parser.add_argument(
        "--compare-aggregate-jsonl",
        type=Path,
        default=None,
        help="Optional previous strategy aggregate JSONL to compare against this run.",
    )
    parser.add_argument(
        "--key-preset",
        choices=sorted(HF_KEY_PRESETS),
        default="llama",
        help="HF-style checkpoint key preset for the synthetic external handoff.",
    )
    parser.add_argument(
        "--zspace-strength",
        type=float,
        default=ZSPACE_STRENGTH,
        help="Projection blend strength for hf_zspace_projected.",
    )
    parser.add_argument(
        "--zspace-preset",
        choices=sorted(ZSPACE_PRESETS),
        default=None,
        help=(
            "Shortcut Z-space projection preset. 'healthy' uses the current "
            "projection-health candidate surfaced by byte_lm_zspace_compare."
        ),
    )
    parser.add_argument(
        "--zspace-strengths",
        default=None,
        help=(
            "Comma-separated projection strengths for a Z-space grid sweep. "
            "When set, each grid point becomes a distinct strategy row."
        ),
    )
    parser.add_argument(
        "--zspace-curvature",
        type=float,
        default=ZSPACE_CURVATURE,
        help="OpenTopos curvature for hf_zspace_projected.",
    )
    parser.add_argument(
        "--zspace-curvatures",
        default=None,
        help=(
            "Comma-separated OpenTopos curvatures for a Z-space grid sweep. "
            "Defaults to --zspace-curvature when omitted."
        ),
    )
    parser.add_argument(
        "--zspace-frequency",
        type=float,
        default=ZSPACE_FREQUENCY,
        help="LanguageWaveEncoder frequency for hf_zspace_projected.",
    )
    parser.add_argument(
        "--zspace-frequencies",
        default=None,
        help=(
            "Comma-separated LanguageWaveEncoder frequencies for a Z-space "
            "grid sweep. Defaults to --zspace-frequency when omitted."
        ),
    )
    parser.add_argument(
        "--require-winner-match",
        action="store_true",
        help="Fail when the winning handoff strategy changes.",
    )
    parser.add_argument(
        "--max-aggregate-target-loss-regression",
        type=float,
        default=None,
        help=(
            "Fail when aggregate target_loss_delta_mean regresses from "
            "--compare-aggregate-jsonl by more than this non-negative tolerance."
        ),
    )
    parser.add_argument(
        "--max-aggregate-retention-loss-regression",
        type=float,
        default=None,
        help=(
            "Fail when aggregate retention_loss_delta_mean regresses from "
            "--compare-aggregate-jsonl by more than this non-negative tolerance."
        ),
    )
    parser.add_argument(
        "--max-aggregate-accepted-rate-regression",
        type=float,
        default=None,
        help=(
            "Fail when aggregate accepted_rate drops from --compare-aggregate-jsonl "
            "by more than this non-negative tolerance."
        ),
    )
    parser.add_argument(
        "--max-aggregate-movement-ok-rate-regression",
        type=float,
        default=None,
        help=(
            "Fail when aggregate movement_ok_rate drops from --compare-aggregate-jsonl "
            "by more than this non-negative tolerance."
        ),
    )
    parser.add_argument(
        "--min-aggregate-target-loss-margin",
        type=float,
        default=None,
        help="Fail when aggregate target_loss_margin_min is below this non-negative floor.",
    )
    parser.add_argument(
        "--min-aggregate-retention-loss-margin",
        type=float,
        default=None,
        help="Fail when aggregate retention_loss_margin_min is below this non-negative floor.",
    )
    parser.add_argument(
        "--require-aggregate-winner-match",
        action="store_true",
        help="Fail when the case-averaged winning handoff strategy changes.",
    )
    parser.add_argument(
        "--require-aggregate-accepted-all",
        action="store_true",
        help="Fail when any current handoff strategy aggregate has a non-accepted case.",
    )
    parser.add_argument(
        "--min-aggregate-cases",
        type=int,
        default=None,
        help="Fail when a current handoff strategy aggregate includes fewer than this many cases.",
    )
    parser.add_argument(
        "--require-aggregate-case",
        dest="require_aggregate_cases",
        action="append",
        choices=[case["label"] for case in CASE_SPECS],
        default=[],
        help="Fail when a current handoff strategy aggregate is missing this case label. May be repeated.",
    )
    parser.add_argument(
        "--min-aggregate-accepted-rate",
        type=float,
        default=None,
        help="Fail when current aggregate accepted_rate is below this 0..1 floor.",
    )
    parser.add_argument(
        "--min-aggregate-movement-ok-rate",
        type=float,
        default=None,
        help="Fail when current aggregate movement_ok_rate is below this 0..1 floor.",
    )
    add_summary_compare_args(parser, subject="handoff strategy")
    args = parser.parse_args()
    gate_requested = validate_summary_compare_args(parser, args) or args.require_winner_match
    if args.compare_jsonl is None and gate_requested:
        parser.error("regression gate options require --compare-jsonl")
    aggregate_gate_requested = (
        args.max_aggregate_target_loss_regression is not None
        or args.max_aggregate_retention_loss_regression is not None
        or args.max_aggregate_accepted_rate_regression is not None
        or args.max_aggregate_movement_ok_rate_regression is not None
        or args.min_aggregate_target_loss_margin is not None
        or args.min_aggregate_retention_loss_margin is not None
        or args.require_aggregate_winner_match
    )
    if args.compare_aggregate_jsonl is None and aggregate_gate_requested:
        parser.error("aggregate regression gate options require --compare-aggregate-jsonl")
    for name in [
        "max_aggregate_target_loss_regression",
        "max_aggregate_retention_loss_regression",
        "max_aggregate_accepted_rate_regression",
        "max_aggregate_movement_ok_rate_regression",
        "min_aggregate_target_loss_margin",
        "min_aggregate_retention_loss_margin",
        "min_aggregate_accepted_rate",
        "min_aggregate_movement_ok_rate",
    ]:
        value = getattr(args, name)
        if value is not None and value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
        if value is not None and "_rate" in name and value > 1.0:
            parser.error(f"--{name.replace('_', '-')} must be at most 1.0")
    if args.min_aggregate_cases is not None and args.min_aggregate_cases <= 0:
        parser.error("--min-aggregate-cases must be positive")
    if len(set(args.require_aggregate_cases)) != len(args.require_aggregate_cases):
        parser.error("--require-aggregate-case values must be unique")
    try:
        for attr in ["zspace_strengths", "zspace_curvatures", "zspace_frequencies"]:
            values = getattr(args, attr)
            if values is not None and not parse_float_list(values):
                parser.error(f"--{attr.replace('_', '-')} must contain at least one number")
    except ValueError as exc:
        parser.error(f"invalid Z-space grid value: {exc}")
    return args


def selected_strategies(labels):
    if labels is None:
        return STRATEGY_SPECS
    selected = set(labels)
    if selected != {"hf_exact"}:
        selected.add("hf_exact")
    return [spec for spec in STRATEGY_SPECS if spec["label"] in selected]


def selected_cases(labels):
    if labels is None:
        return [CASE_SPECS[0]]
    selected = set(labels)
    return [case for case in CASE_SPECS if case["label"] in selected]


def parse_float_list(raw):
    if raw is None:
        return []
    values = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    return values


def zspace_preset_values(args):
    preset = getattr(args, "zspace_preset", None)
    if preset is None:
        return {}
    return ZSPACE_PRESETS[preset]


def resolved_zspace_value(args, name, default):
    preset = zspace_preset_values(args)
    return preset.get(name, getattr(args, name, default))


def zspace_grid_values(args):
    return {
        "zspace_strength": parse_float_list(getattr(args, "zspace_strengths", None))
        or [resolved_zspace_value(args, "zspace_strength", ZSPACE_STRENGTH)],
        "zspace_curvature": parse_float_list(getattr(args, "zspace_curvatures", None))
        or [resolved_zspace_value(args, "zspace_curvature", ZSPACE_CURVATURE)],
        "zspace_frequency": parse_float_list(getattr(args, "zspace_frequencies", None))
        or [resolved_zspace_value(args, "zspace_frequency", ZSPACE_FREQUENCY)],
    }


def uses_zspace_grid(args):
    return any(
        getattr(args, attr, None) is not None
        for attr in ["zspace_strengths", "zspace_curvatures", "zspace_frequencies"]
    ) or getattr(args, "zspace_preset", None) is not None


def label_number(value):
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def zspace_variant_label(strength, curvature, frequency):
    return (
        f"hf_zspace_s{label_number(strength)}"
        f"_c{label_number(curvature)}"
        f"_f{label_number(frequency)}"
    )


def zspace_grid_specs(strategy, args):
    grid = zspace_grid_values(args)
    specs = []
    for strength in grid["zspace_strength"]:
        for curvature in grid["zspace_curvature"]:
            for frequency in grid["zspace_frequency"]:
                configured = dict(strategy)
                configured.update(
                    {
                        "label": zspace_variant_label(strength, curvature, frequency),
                        "strategy_family": strategy["label"],
                        "zspace_strength": strength,
                        "zspace_curvature": curvature,
                        "zspace_frequency": frequency,
                    }
                )
                specs.append(configured)
    return specs


def strategy_with_projection_args(strategy, args):
    configured = dict(strategy)
    if configured["projection"] == "zspace":
        configured.update(
            {
                "zspace_strength": resolved_zspace_value(
                    args, "zspace_strength", ZSPACE_STRENGTH
                ),
                "zspace_curvature": resolved_zspace_value(
                    args, "zspace_curvature", ZSPACE_CURVATURE
                ),
                "zspace_frequency": resolved_zspace_value(
                    args, "zspace_frequency", ZSPACE_FREQUENCY
                ),
            }
        )
    return configured


def configured_strategies(args):
    strategies = []
    for strategy in selected_strategies(args.strategies):
        if strategy["projection"] == "zspace" and uses_zspace_grid(args):
            strategies.extend(zspace_grid_specs(strategy, args))
        else:
            strategies.append(strategy_with_projection_args(strategy, args))
    return strategies


def new_runtime():
    session = st.SpiralSession(
        device="wgpu",
        curvature=-1.0,
        hyper_learning_rate=0.5,
        fallback_learning_rate=0.1,
    )
    trainer = session.trainer()
    trainer.set_max_grad_norm(2.0)
    trainer.set_gradient_accumulation_steps(ACCUMULATION_STEPS)
    schedule = trainer.roundtable(rows=CONTEXT * BATCH_WINDOWS, cols=VOCAB)
    return session, trainer, schedule


def projection_audit_fields(strategy):
    if strategy["projection"] != "zspace":
        return {
            "checkpoint_projection": strategy["projection"],
            "checkpoint_projection_strength": None,
            "checkpoint_projection_curvature": None,
            "checkpoint_projection_frequency": None,
        }
    return {
        "checkpoint_projection": strategy["projection"],
        "checkpoint_projection_strength": float(
            strategy.get("zspace_strength", ZSPACE_STRENGTH)
        ),
        "checkpoint_projection_curvature": float(
            strategy.get("zspace_curvature", ZSPACE_CURVATURE)
        ),
        "checkpoint_projection_frequency": float(
            strategy.get("zspace_frequency", ZSPACE_FREQUENCY)
        ),
    }


def projection_value_label(value):
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def zspace_projector(strategy):
    fields = projection_audit_fields(strategy)
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


def expand_tensor(tensor, rows, cols):
    source_data = tensor.data()
    return st.Tensor(
        rows,
        cols,
        [
            source_data[row * tensor.cols + col]
            if row < tensor.rows and col < tensor.cols
            else 0.0
            for row in range(rows)
            for col in range(cols)
        ],
    )


def source_tensor_for_strategy(source_state, name, strategy, projector):
    tensor = source_state[name]
    if strategy["projection"] == "zspace":
        return projector.forward(tensor)
    return tensor


def external_state_for_strategy(source_state, strategy, *, key_preset):
    key_kwargs = hf_lm_key_preset(key_preset)
    projector = zspace_projector(strategy) if strategy["projection"] == "zspace" else None
    embed_weight = source_tensor_for_strategy(
        source_state,
        "embed::weight",
        strategy,
        projector,
    )
    embed_bias = source_tensor_for_strategy(
        source_state,
        "embed::bias",
        strategy,
        projector,
    )
    head_weight = source_tensor_for_strategy(
        source_state,
        "head::weight",
        strategy,
        projector,
    )
    head_bias = source_tensor_for_strategy(
        source_state,
        "head::bias",
        strategy,
        projector,
    )
    raw_state = {
        key_kwargs["embed_weight_key"]: expand_tensor(
            embed_weight,
            VOCAB + strategy["extra_vocab_rows"],
            HIDDEN + strategy["extra_hidden_cols"],
        ),
        key_kwargs["embed_bias_key"]: expand_tensor(
            embed_bias,
            1,
            HIDDEN + strategy["extra_hidden_cols"],
        ),
        key_kwargs["lm_head_weight_key"]: expand_tensor(
            head_weight.transpose(),
            VOCAB + strategy["extra_head_rows"],
            HIDDEN + strategy["extra_hidden_cols"],
        ),
        key_kwargs["lm_head_bias_key"]: expand_tensor(
            head_bias,
            1,
            VOCAB + strategy["extra_head_rows"],
        ),
        HF_UNUSED_KEYS.get(key_preset, f"{key_preset}.unused.norm.weight"): st.Tensor(
            1,
            2,
            [1.0, 1.0],
        ),
    }
    include_extra_keys = [
        HF_UNUSED_KEYS.get(key_preset, f"{key_preset}.unused.norm.weight"),
    ]
    resize_kwargs = (
        hf_lm_overlap_resize_kwargs()
        if strategy["allow_overlap_resize"]
        else {}
    )
    checkpoint, rules = hf_lm_handoff_from_external_state(
        raw_state,
        key_preset=key_preset,
        include_extra_keys=include_extra_keys,
        **resize_kwargs,
    )
    shapes = infer_hf_lm_module_shapes(raw_state, key_preset=key_preset)
    return checkpoint, rules, include_extra_keys, shapes


def build_lora_mlp(source_state, source_key_map, *, label_prefix, emit_preflight=True):
    embed = Linear(VOCAB, HIDDEN, name="embed")
    embed_report, embed_load = preflight_and_load(
        f"{label_prefix}_embed",
        embed,
        source_state,
        source_key_map,
        emit=emit_preflight,
    )
    embed.set_trainable(False)

    head = LoraLinear(HIDDEN, VOCAB, LORA_RANK, alpha=LORA_ALPHA, name="head")
    head_report, head_load = preflight_and_load(
        f"{label_prefix}_head_base",
        head,
        source_state,
        source_key_map,
        lora_base=True,
        emit=emit_preflight,
    )
    return Sequential([embed, Relu(), head]), embed_load, head_load, embed_report, head_report


def run_strategy(
    strategy,
    case,
    source_state,
    source_delta,
    source_stats,
    source_samples,
    target_samples,
    *,
    key_preset,
):
    checkpoint, rules, include_extra_keys, checkpoint_shapes = external_state_for_strategy(
        source_state,
        strategy,
        key_preset=key_preset,
    )
    session, trainer, schedule = new_runtime()
    loss = SoftmaxCrossEntropy()
    label_prefix = f"{case['label']}_{strategy['label']}"
    target, embed_load, head_load, embed_report, head_report = build_lora_mlp(
        checkpoint,
        rules,
        label_prefix=label_prefix,
    )
    if not embed_load["matched"] or not head_load["matched"]:
        raise RuntimeError(
            f"{strategy['label']} base load mismatch: embed={embed_load} head={head_load}"
        )

    session.prepare_module(target)
    frozen_base = target.set_parameters_trainable_by_suffix("::weight", False)
    frozen_base += target.set_parameters_trainable_by_suffix("::bias", False)
    boosted_adapter = target.set_parameters_learning_rate_scale_by_contains("lora_", 4.0)

    ft_ready_state = target.state_dict()
    ft_ready_resume = trainer.resume_fingerprint(target)
    resumed, _, _, _, _ = build_lora_mlp(
        checkpoint,
        rules,
        label_prefix=f"{label_prefix}_resume",
        emit_preflight=False,
    )
    session.prepare_module(resumed)
    resumed.set_parameters_trainable_by_suffix("::weight", False)
    resumed.set_parameters_trainable_by_suffix("::bias", False)
    resumed.set_parameters_learning_rate_scale_by_contains("lora_", 4.0)
    resume_load = resumed.load_state_dict_checked(ft_ready_state)
    resume_check = trainer.resume_fingerprint(resumed)
    if not resume_load["matched"] or resume_check["hash"] != ft_ready_resume["hash"]:
        raise RuntimeError(
            f"{strategy['label']} FT-ready resume fingerprint mismatch: "
            f"load={resume_load} expected={ft_ready_resume} actual={resume_check}"
        )

    ft_report = session.train_epochs_restore_best_sparse_with_finetune_report(
        trainer,
        target,
        loss,
        loader(target_samples, seed=11),
        loader(target_samples, seed=17),
        loader(source_samples, seed=13),
        schedule,
        epochs=FT_EPOCHS,
        movement_tolerance=FT_MOVEMENT_TOLERANCE,
        max_loss_increase=10.0,
        max_accuracy_drop=1.0,
        max_perplexity_increase=100.0,
        target_min_loss_delta=0.0,
    )
    if ft_report.captured.guarded_best_epoch is None:
        raise RuntimeError(
            f"{strategy['label']} retention guard rejected every fine-tune epoch"
        )
    if ft_report.target_delta["loss_delta"] <= 0.0:
        raise RuntimeError(
            f"{strategy['label']} fine-tune loss did not improve: {ft_report.target_delta}"
        )
    if not ft_report.movement_ok:
        raise RuntimeError(f"{strategy['label']} unexpected movement: {ft_report.movement}")

    summary = ft_report.summary()
    movement = ft_report.movement
    margins = summary_guard_margins(summary)
    projection_fields = projection_audit_fields(strategy)
    print(
        f"case={case['label']} "
        f"handoff_strategy={strategy['label']} "
        f"checkpoint_key_preset={key_preset} "
        f"checkpoint_overlap_resize={strategy['allow_overlap_resize']} "
        f"checkpoint_projection={projection_fields['checkpoint_projection']} "
        f"checkpoint_projection_strength={projection_value_label(projection_fields['checkpoint_projection_strength'])} "
        f"checkpoint_projection_curvature={projection_value_label(projection_fields['checkpoint_projection_curvature'])} "
        f"checkpoint_projection_frequency={projection_value_label(projection_fields['checkpoint_projection_frequency'])} "
        f"checkpoint_vocab={checkpoint_shapes[0]} "
        f"checkpoint_hidden={checkpoint_shapes[1]} "
        f"checkpoint_target_classes={checkpoint_shapes[2]} "
        f"extra_audit_keys={len(include_extra_keys)} "
        f"ft_loss_delta={metric(ft_report.target_delta, 'loss_delta'):.6f} "
        f"retention_loss_delta={metric(ft_report.retention_delta, 'loss_delta'):.6f} "
        f"target_loss_margin={margins['target_loss_margin']:.6f} "
        f"retention_loss_margin={margins['retention_loss_margin']:.6f} "
        f"movement_status={movement['status']} "
        f"resume_hash={summary['resume_hash']}"
    )
    row = dict(summary)
    attach_summary_guard_margins(row)
    attach_summary_guard_counts(row, ft_report.captured)
    row.update(
        {
            "example": "byte_lm_handoff_strategy_compare",
            "case": case["label"],
            "config": strategy["label"],
            "strategy": strategy["label"],
            "strategy_family": strategy.get("strategy_family", strategy["label"]),
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "hidden": HIDDEN,
            "source_rows": len(source_samples) * CONTEXT,
            "target_rows": len(target_samples) * CONTEXT,
            "source_batches": source_stats.batches,
            "source_optimizer_steps": source_stats.optimizer_steps,
            "source_loss_delta": metric(source_delta, "loss_delta"),
            "source_accuracy_delta": metric(source_delta, "accuracy_delta"),
            "source_perplexity_delta": metric(source_delta, "perplexity_delta"),
            "frozen_base_params": frozen_base,
            "boosted_adapter_params": boosted_adapter,
            "checkpoint_key_preset": key_preset,
            "checkpoint_source_origin": "synthetic_hf_strategy",
            "checkpoint_loaded_files": 0,
            "checkpoint_vocab": checkpoint_shapes[0],
            "checkpoint_hidden": checkpoint_shapes[1],
            "checkpoint_target_classes": checkpoint_shapes[2],
            "checkpoint_overlap_resize": strategy["allow_overlap_resize"],
        }
    )
    row.update(projection_fields)
    row.update(checkpoint_audit_fields("embed", embed_report, embed_load))
    row.update(checkpoint_audit_fields("head", head_report, head_load))
    return row


def row_key(row):
    case = row.get("case") or DEFAULT_CASE_LABEL
    strategy = row.get("strategy") or row.get("config")
    if not isinstance(case, str) or not case:
        raise ValueError(f"row has invalid case: {case!r}")
    if not isinstance(strategy, str) or not strategy:
        raise ValueError(f"row has invalid strategy: {strategy!r}")
    return f"{case}::{strategy}"


def rows_by_case_strategy(rows, label):
    by_key = {}
    for row in rows:
        key = row_key(row)
        if key in by_key:
            raise ValueError(f"{label} contains duplicate case/strategy: {key}")
        by_key[key] = row
    return by_key


def is_numeric_value(value):
    return not isinstance(value, bool) and isinstance(value, (int, float))


def numeric_value(row, key):
    value = row.get(key)
    if not is_numeric_value(value):
        raise ValueError(f"{row.get('strategy')} row missing numeric {key}")
    return float(value)


def optional_numeric_value(row, key):
    value = row.get(key)
    if not is_numeric_value(value):
        return float("nan")
    return float(value)


def winner_score(row):
    return (
        numeric_value(row, "target_loss_delta"),
        numeric_value(row, "retention_loss_delta"),
        numeric_value(row, "retention_accuracy_delta"),
    )


def strategy_winner(rows):
    candidates = [
        row
        for row in rows
        if summary_bool_value(row, "accepted", False)
        and summary_bool_value(row, "movement_ok", False)
        and numeric_value(row, "target_loss_delta") > 0.0
    ]
    if not candidates:
        raise RuntimeError("handoff strategy compare has no accepted improving rows")
    best = max(winner_score(row) for row in candidates)
    winners = sorted(
        row_key(row)
        for row in candidates
        if winner_score(row) == best
    )
    return "+".join(winners), best


def compare_strategy_rows(current_rows, baseline_rows, args):
    current = rows_by_case_strategy(current_rows, "current")
    baseline = rows_by_case_strategy(baseline_rows, "baseline")
    missing_current = sorted(set(baseline) - set(current))
    if missing_current:
        raise RuntimeError(
            "baseline case/strategies missing from current compare: "
            + ",".join(missing_current)
        )
    common = sorted(set(current) & set(baseline))
    if not common:
        raise RuntimeError("no overlapping case/strategies between current compare and baseline")

    failures = []
    current_winner, current_score = strategy_winner(list(current.values()))
    baseline_winner, baseline_score = strategy_winner(list(baseline.values()))
    winner_changed = current_winner != baseline_winner
    print(
        f"strategy_winner_compare before={baseline_winner} "
        f"after={current_winner} "
        f"target_loss_delta_before={baseline_score[0]:.9f} "
        f"target_loss_delta_after={current_score[0]:.9f} "
        f"retention_loss_delta_before={baseline_score[1]:.9f} "
        f"retention_loss_delta_after={current_score[1]:.9f} "
        f"winner_changed={winner_changed} "
        f"passed={not winner_changed or not args.require_winner_match}"
    )
    if winner_changed and args.require_winner_match:
        failures.append(f"winner changed from {baseline_winner} to {current_winner}")

    for strategy in common:
        now = current[strategy]
        before = baseline[strategy]
        checkpoint_changed = bool(checkpoint_audit_differences(now, before))
        comparison = compare_summaries(
            now,
            before,
            max_target_loss_regression=args.max_target_loss_regression,
            max_retention_loss_regression=args.max_retention_loss_regression,
            min_target_loss_margin=args.min_target_loss_margin,
            min_retention_loss_margin=args.min_retention_loss_margin,
            min_retention_accuracy_margin=args.min_retention_accuracy_margin,
            min_retention_perplexity_margin=args.min_retention_perplexity_margin,
            require_status_match=args.require_status_match,
            require_accepted_match=args.require_accepted_match,
            require_guard_match=args.require_guard_match,
            require_movement_tolerance_match=args.require_movement_tolerance_match,
            require_resume_match=args.require_resume_match,
        )
        print(
            f"summary_compare case_strategy={strategy} "
            f"target_loss_delta_change={comparison['target_loss_delta_change']:.9f} "
            f"retention_loss_delta_change={comparison['retention_loss_delta_change']:.9f} "
            f"target_loss_regression={comparison['target_loss_regression']:.9f} "
            f"retention_loss_regression={comparison['retention_loss_regression']:.9f} "
            f"status_changed={comparison['status_changed']} "
            f"accepted_changed={comparison['accepted_changed']} "
            f"resume_changed={comparison['resume_changed']} "
            f"checkpoint_changed={checkpoint_changed} "
            f"passed={comparison['passed']}"
        )
        failures.extend(
            summary_compare_failures(
                strategy,
                comparison,
                max_target_loss_regression=args.max_target_loss_regression,
                max_retention_loss_regression=args.max_retention_loss_regression,
                min_target_loss_margin=args.min_target_loss_margin,
                min_retention_loss_margin=args.min_retention_loss_margin,
                min_retention_accuracy_margin=args.min_retention_accuracy_margin,
                min_retention_perplexity_margin=args.min_retention_perplexity_margin,
                require_status_match=args.require_status_match,
                require_accepted_match=args.require_accepted_match,
                require_guard_match=args.require_guard_match,
                require_movement_tolerance_match=args.require_movement_tolerance_match,
                require_resume_match=args.require_resume_match,
            )
        )
        if args.require_checkpoint_match:
            failures.extend(checkpoint_audit_failures(strategy, now, before))
        if not comparison["passed"]:
            failures.append(f"{strategy}: comparison returned passed=false")

    if failures:
        raise RuntimeError("handoff strategy regression gate failed: " + "; ".join(failures))
    return len(common)


def mean(values):
    values = list(values)
    if not values:
        raise ValueError("cannot average an empty value list")
    return sum(values) / len(values)


def consistent_value(rows, key):
    values = [row.get(key) for row in rows]
    first = values[0]
    if any(value != first for value in values[1:]):
        strategy = rows[0].get("strategy")
        raise ValueError(f"strategy {strategy} has inconsistent aggregate field {key}")
    return first


def strategy_rows(rows):
    by_strategy = {}
    for row in rows:
        strategy = row.get("strategy") or row.get("config")
        if not isinstance(strategy, str) or not strategy:
            raise ValueError(f"row has invalid strategy: {strategy!r}")
        by_strategy.setdefault(strategy, []).append(row)
    return by_strategy


def aggregate_strategy_rows(rows):
    aggregates = []
    for strategy, grouped in strategy_rows(rows).items():
        case_labels = []
        for row in grouped:
            case = row.get("case") or DEFAULT_CASE_LABEL
            if not isinstance(case, str) or not case:
                raise ValueError(f"strategy {strategy} has invalid aggregate case: {case!r}")
            if case in case_labels:
                raise ValueError(f"strategy {strategy} contains duplicate case: {case}")
            case_labels.append(case)
        case_count = len(case_labels)
        accepted_cases = sum(
            1 for row in grouped if summary_bool_value(row, "accepted", False)
        )
        movement_ok_cases = sum(
            1 for row in grouped if summary_bool_value(row, "movement_ok", False)
        )
        aggregate = {
            "row_type": "strategy_aggregate",
            "strategy": strategy,
            "strategy_family": consistent_value(grouped, "strategy_family"),
            "cases": case_count,
            "case_labels": ",".join(case_labels),
            "accepted_cases": accepted_cases,
            "rejected_cases": case_count - accepted_cases,
            "accepted_rate": accepted_cases / case_count,
            "accepted_all": accepted_cases == case_count,
            "movement_ok_cases": movement_ok_cases,
            "movement_not_ok_cases": case_count - movement_ok_cases,
            "movement_ok_rate": movement_ok_cases / case_count,
            "movement_ok_all": movement_ok_cases == case_count,
            "target_loss_delta_mean": mean(
                numeric_value(row, "target_loss_delta") for row in grouped
            ),
            "retention_loss_delta_mean": mean(
                numeric_value(row, "retention_loss_delta") for row in grouped
            ),
            "retention_accuracy_delta_mean": mean(
                numeric_value(row, "retention_accuracy_delta") for row in grouped
            ),
            "target_loss_margin_mean": mean(
                numeric_value(row, "target_loss_margin") for row in grouped
            ),
            "target_loss_margin_min": min(
                numeric_value(row, "target_loss_margin") for row in grouped
            ),
            "retention_loss_margin_mean": mean(
                numeric_value(row, "retention_loss_margin") for row in grouped
            ),
            "retention_loss_margin_min": min(
                numeric_value(row, "retention_loss_margin") for row in grouped
            ),
            "checkpoint_key_preset": consistent_value(grouped, "checkpoint_key_preset"),
            "checkpoint_source_origin": consistent_value(grouped, "checkpoint_source_origin"),
            "checkpoint_vocab": consistent_value(grouped, "checkpoint_vocab"),
            "checkpoint_hidden": consistent_value(grouped, "checkpoint_hidden"),
            "checkpoint_target_classes": consistent_value(
                grouped, "checkpoint_target_classes"
            ),
            "checkpoint_overlap_resize": consistent_value(
                grouped, "checkpoint_overlap_resize"
            ),
            "checkpoint_projection": consistent_value(grouped, "checkpoint_projection"),
            "checkpoint_projection_strength": consistent_value(
                grouped, "checkpoint_projection_strength"
            ),
            "checkpoint_projection_curvature": consistent_value(
                grouped, "checkpoint_projection_curvature"
            ),
            "checkpoint_projection_frequency": consistent_value(
                grouped, "checkpoint_projection_frequency"
            ),
        }
        aggregates.append(aggregate)
    return aggregates


def aggregate_score(row):
    return (
        numeric_value(row, "target_loss_delta_mean"),
        numeric_value(row, "retention_loss_delta_mean"),
        numeric_value(row, "retention_accuracy_delta_mean"),
    )


def aggregate_winner(rows):
    candidates = [
        row
        for row in rows
        if summary_bool_value(row, "accepted_all", False)
        and summary_bool_value(row, "movement_ok_all", False)
        and numeric_value(row, "target_loss_delta_mean") > 0.0
    ]
    if not candidates:
        raise RuntimeError("handoff aggregate compare has no accepted improving rows")
    best = max(aggregate_score(row) for row in candidates)
    winners = sorted(
        row["strategy"] for row in candidates if aggregate_score(row) == best
    )
    return "+".join(winners), best


def optional_aggregate_winner(rows):
    try:
        return aggregate_winner(rows)
    except RuntimeError:
        return None, None


def optional_int_value(row, key):
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def required_aggregate_int_value(row, key):
    value = optional_int_value(row, key)
    if value is None:
        raise ValueError(
            f"aggregate row {row.get('strategy')} missing integer {key}"
        )
    return value


def required_aggregate_bool_value(row, key):
    value = row.get(key)
    if not isinstance(value, bool):
        raise ValueError(
            f"aggregate row {row.get('strategy')} missing boolean {key}"
        )
    return value


def rate_consistency_failure(row, count_key, rate_key, cases):
    if count_key not in row or rate_key not in row:
        return None
    count = optional_int_value(row, count_key)
    if count is None:
        return f"{count_key} is not an integer"
    rate = row.get(rate_key)
    if not is_numeric_value(rate):
        return f"{rate_key} is not numeric"
    if rate < 0.0 or rate > 1.0:
        return f"{rate_key} {float(rate):.9f} outside 0..1"
    expected = count / cases
    if abs(float(rate) - expected) > 1e-9:
        return f"{rate_key} {float(rate):.9f} != {count_key}/{cases} {expected:.9f}"
    return None


def aggregate_case_labels(row):
    labels = row.get("case_labels")
    if not isinstance(labels, str):
        return []
    return [label for label in labels.split(",") if label]


def aggregate_row_consistency_failures(row, label):
    strategy = row.get("strategy")
    prefix = f"{label} aggregate {strategy}"
    failures = []
    cases = optional_int_value(row, "cases")
    if cases is None or cases <= 0:
        failures.append(f"{prefix}: cases must be a positive integer")
        return failures

    raw_case_labels = row.get("case_labels")
    case_labels = aggregate_case_labels(row)
    if not isinstance(raw_case_labels, str) or not case_labels:
        failures.append(f"{prefix}: case_labels must be a non-empty comma list")
    else:
        if len(case_labels) != cases:
            failures.append(
                f"{prefix}: case_labels count {len(case_labels)} != cases {cases}"
            )
        if len(set(case_labels)) != len(case_labels):
            failures.append(f"{prefix}: case_labels contains duplicates")

    for key in ["accepted_all", "movement_ok_all"]:
        if key in row and not isinstance(row.get(key), bool):
            failures.append(f"{prefix}: {key} must be boolean")
    for count_key in [
        "accepted_cases",
        "rejected_cases",
        "movement_ok_cases",
        "movement_not_ok_cases",
    ]:
        if count_key in row:
            count = optional_int_value(row, count_key)
            if count is None or count < 0:
                failures.append(f"{prefix}: {count_key} must be a non-negative integer")

    for left_key, right_key in [
        ("accepted_cases", "rejected_cases"),
        ("movement_ok_cases", "movement_not_ok_cases"),
    ]:
        left = optional_int_value(row, left_key)
        right = optional_int_value(row, right_key)
        if left is not None and right is not None and left + right != cases:
            failures.append(
                f"{prefix}: {left_key}+{right_key} {left + right} != cases {cases}"
            )

    for count_key, rate_key in [
        ("accepted_cases", "accepted_rate"),
        ("movement_ok_cases", "movement_ok_rate"),
    ]:
        failure = rate_consistency_failure(row, count_key, rate_key, cases)
        if failure is not None:
            failures.append(f"{prefix}: {failure}")

    for count_key, all_key in [
        ("accepted_cases", "accepted_all"),
        ("movement_ok_cases", "movement_ok_all"),
    ]:
        count = optional_int_value(row, count_key)
        all_value = row.get(all_key)
        if count is not None and all_key in row:
            if isinstance(all_value, bool) and all_value != (count == cases):
                failures.append(
                    f"{prefix}: {all_key} {all_value} inconsistent with {count_key}/{cases}"
                )
    return failures


def validate_aggregate_row(row, label):
    failures = aggregate_row_consistency_failures(row, label)
    if failures:
        raise ValueError("; ".join(failures))
    return row


def aggregate_rows_by_strategy(rows, label):
    by_strategy = {}
    for row in rows:
        if row.get("row_type") != "strategy_aggregate":
            raise ValueError(f"{label} expected row_type='strategy_aggregate'")
        strategy = row.get("strategy")
        if not isinstance(strategy, str) or not strategy:
            raise ValueError(f"{label} aggregate row has invalid strategy: {strategy!r}")
        if strategy in by_strategy:
            raise ValueError(f"{label} contains duplicate aggregate strategy: {strategy}")
        validate_aggregate_row(row, label)
        by_strategy[strategy] = row
    return by_strategy


def check_aggregate_coverage(
    rows,
    *,
    require_accepted_all=False,
    min_cases=None,
    required_cases=None,
    min_accepted_rate=None,
    min_movement_ok_rate=None,
):
    required_cases = list(required_cases or [])
    failures = []
    for row in rows:
        validate_aggregate_row(row, "current")
        strategy = row.get("strategy")
        accepted_cases = required_aggregate_int_value(row, "accepted_cases")
        cases = required_aggregate_int_value(row, "cases")
        case_labels = aggregate_case_labels(row)
        accepted_rate = numeric_value(row, "accepted_rate")
        movement_ok_cases = required_aggregate_int_value(row, "movement_ok_cases")
        movement_ok_rate = numeric_value(row, "movement_ok_rate")
        accepted_all = required_aggregate_bool_value(row, "accepted_all")
        movement_ok_all = required_aggregate_bool_value(row, "movement_ok_all")
        missing_cases = [case for case in required_cases if case not in case_labels]
        print(
            f"aggregate_acceptance strategy={strategy} "
            f"accepted_cases={accepted_cases} "
            f"cases={cases} "
            f"case_labels={','.join(case_labels) or 'none'} "
            f"accepted_rate={accepted_rate:.9f} "
            f"accepted_all={accepted_all} "
            f"movement_ok_cases={movement_ok_cases} "
            f"movement_ok_rate={movement_ok_rate:.9f} "
            f"movement_ok_all={movement_ok_all}"
        )
        if min_cases is not None and cases < min_cases:
            failures.append(
                f"{strategy}: aggregate cases {cases} below floor {min_cases}"
            )
        if missing_cases:
            failures.append(
                f"{strategy}: missing aggregate cases {','.join(missing_cases)}"
            )
        if require_accepted_all and not accepted_all:
            failures.append(
                f"{strategy}: accepted {accepted_cases}/{cases} aggregate cases"
            )
        if min_accepted_rate is not None and accepted_rate < min_accepted_rate:
            failures.append(
                f"{strategy}: accepted_rate {accepted_rate:.9f} below floor {min_accepted_rate:.9f}"
            )
        if min_movement_ok_rate is not None and movement_ok_rate < min_movement_ok_rate:
            failures.append(
                f"{strategy}: movement_ok_rate {movement_ok_rate:.9f} below floor {min_movement_ok_rate:.9f}"
            )
    if failures:
        raise RuntimeError("handoff aggregate coverage gate failed: " + "; ".join(failures))
    return len(rows)


def compare_aggregate_rows(current_rows, baseline_rows, args):
    current = aggregate_rows_by_strategy(current_rows, "current")
    baseline = aggregate_rows_by_strategy(baseline_rows, "baseline")
    missing_current = sorted(set(baseline) - set(current))
    if missing_current:
        raise RuntimeError(
            "baseline aggregate strategies missing from current compare: "
            + ",".join(missing_current)
        )
    common = sorted(set(current) & set(baseline))
    if not common:
        raise RuntimeError("no overlapping aggregate strategies between current and baseline")

    failures = []
    current_winner, current_score = optional_aggregate_winner(list(current.values()))
    baseline_winner, baseline_score = optional_aggregate_winner(list(baseline.values()))
    winner_changed = current_winner != baseline_winner
    current_target_score = current_score[0] if current_score is not None else float("nan")
    current_retention_score = (
        current_score[1] if current_score is not None else float("nan")
    )
    baseline_target_score = (
        baseline_score[0] if baseline_score is not None else float("nan")
    )
    baseline_retention_score = (
        baseline_score[1] if baseline_score is not None else float("nan")
    )
    winner_compare_passed = (
        not args.require_aggregate_winner_match
        or (
            current_winner is not None
            and baseline_winner is not None
            and not winner_changed
        )
    )
    print(
        f"aggregate_winner_compare before={baseline_winner or 'none'} "
        f"after={current_winner or 'none'} "
        f"target_loss_delta_before={baseline_target_score:.9f} "
        f"target_loss_delta_after={current_target_score:.9f} "
        f"retention_loss_delta_before={baseline_retention_score:.9f} "
        f"retention_loss_delta_after={current_retention_score:.9f} "
        f"winner_changed={winner_changed} "
        f"passed={winner_compare_passed}"
    )
    if args.require_aggregate_winner_match:
        if current_winner is None or baseline_winner is None:
            failures.append("aggregate winner unavailable")
        elif winner_changed:
            failures.append(f"aggregate winner changed from {baseline_winner} to {current_winner}")

    for strategy in common:
        now = current[strategy]
        before = baseline[strategy]
        scope_changed = (
            now.get("case_labels") != before.get("case_labels")
            or now.get("cases") != before.get("cases")
        )
        target_regression = max(
            0.0,
            numeric_value(before, "target_loss_delta_mean")
            - numeric_value(now, "target_loss_delta_mean"),
        )
        retention_regression = max(
            0.0,
            numeric_value(before, "retention_loss_delta_mean")
            - numeric_value(now, "retention_loss_delta_mean"),
        )
        accepted_rate_before = optional_numeric_value(before, "accepted_rate")
        accepted_rate_after = optional_numeric_value(now, "accepted_rate")
        movement_ok_rate_before = optional_numeric_value(before, "movement_ok_rate")
        movement_ok_rate_after = optional_numeric_value(now, "movement_ok_rate")
        accepted_rate_regression = (
            max(
                0.0,
                numeric_value(before, "accepted_rate")
                - numeric_value(now, "accepted_rate"),
            )
            if args.max_aggregate_accepted_rate_regression is not None
            else 0.0
        )
        movement_ok_rate_regression = (
            max(
                0.0,
                numeric_value(before, "movement_ok_rate")
                - numeric_value(now, "movement_ok_rate"),
            )
            if args.max_aggregate_movement_ok_rate_regression is not None
            else 0.0
        )
        target_margin_min = numeric_value(now, "target_loss_margin_min")
        retention_margin_min = numeric_value(now, "retention_loss_margin_min")
        checkpoint_changed = bool(checkpoint_audit_differences(now, before))
        target_regression_ok = (
            args.max_aggregate_target_loss_regression is None
            or target_regression <= args.max_aggregate_target_loss_regression
        )
        retention_regression_ok = (
            args.max_aggregate_retention_loss_regression is None
            or retention_regression <= args.max_aggregate_retention_loss_regression
        )
        accepted_rate_ok = (
            args.max_aggregate_accepted_rate_regression is None
            or accepted_rate_regression <= args.max_aggregate_accepted_rate_regression
        )
        movement_ok_rate_ok = (
            args.max_aggregate_movement_ok_rate_regression is None
            or movement_ok_rate_regression
            <= args.max_aggregate_movement_ok_rate_regression
        )
        target_margin_ok = (
            args.min_aggregate_target_loss_margin is None
            or target_margin_min >= args.min_aggregate_target_loss_margin
        )
        retention_margin_ok = (
            args.min_aggregate_retention_loss_margin is None
            or retention_margin_min >= args.min_aggregate_retention_loss_margin
        )
        checkpoint_ok = not args.require_checkpoint_match or not checkpoint_changed
        row_passed = (
            not scope_changed
            and target_regression_ok
            and retention_regression_ok
            and accepted_rate_ok
            and movement_ok_rate_ok
            and target_margin_ok
            and retention_margin_ok
            and checkpoint_ok
        )
        print(
            f"aggregate_compare strategy={strategy} "
            f"target_loss_delta_mean_before={numeric_value(before, 'target_loss_delta_mean'):.9f} "
            f"target_loss_delta_mean_after={numeric_value(now, 'target_loss_delta_mean'):.9f} "
            f"retention_loss_delta_mean_before={numeric_value(before, 'retention_loss_delta_mean'):.9f} "
            f"retention_loss_delta_mean_after={numeric_value(now, 'retention_loss_delta_mean'):.9f} "
            f"target_loss_regression={target_regression:.9f} "
            f"retention_loss_regression={retention_regression:.9f} "
            f"accepted_rate_before={accepted_rate_before:.9f} "
            f"accepted_rate_after={accepted_rate_after:.9f} "
            f"accepted_rate_regression={accepted_rate_regression:.9f} "
            f"movement_ok_rate_before={movement_ok_rate_before:.9f} "
            f"movement_ok_rate_after={movement_ok_rate_after:.9f} "
            f"movement_ok_rate_regression={movement_ok_rate_regression:.9f} "
            f"target_loss_margin_min={target_margin_min:.9f} "
            f"retention_loss_margin_min={retention_margin_min:.9f} "
            f"checkpoint_changed={checkpoint_changed} "
            f"scope_changed={scope_changed} "
            f"passed={row_passed}"
        )
        if scope_changed:
            failures.append(f"{strategy}: aggregate case scope changed")
        if (
            args.max_aggregate_target_loss_regression is not None
            and target_regression > args.max_aggregate_target_loss_regression
        ):
            failures.append(
                f"{strategy}: aggregate target_loss_delta_mean regressed by "
                f"{target_regression:.9f}"
            )
        if (
            args.max_aggregate_retention_loss_regression is not None
            and retention_regression > args.max_aggregate_retention_loss_regression
        ):
            failures.append(
                f"{strategy}: aggregate retention_loss_delta_mean regressed by "
                f"{retention_regression:.9f}"
            )
        if (
            args.max_aggregate_accepted_rate_regression is not None
            and accepted_rate_regression > args.max_aggregate_accepted_rate_regression
        ):
            failures.append(
                f"{strategy}: aggregate accepted_rate regressed by {accepted_rate_regression:.9f}"
            )
        if (
            args.max_aggregate_movement_ok_rate_regression is not None
            and movement_ok_rate_regression
            > args.max_aggregate_movement_ok_rate_regression
        ):
            failures.append(
                f"{strategy}: aggregate movement_ok_rate regressed by {movement_ok_rate_regression:.9f}"
            )
        if (
            args.min_aggregate_target_loss_margin is not None
            and target_margin_min < args.min_aggregate_target_loss_margin
        ):
            failures.append(
                f"{strategy}: aggregate target_loss_margin_min {target_margin_min:.9f} "
                f"below floor {args.min_aggregate_target_loss_margin:.9f}"
            )
        if (
            args.min_aggregate_retention_loss_margin is not None
            and retention_margin_min < args.min_aggregate_retention_loss_margin
        ):
            failures.append(
                f"{strategy}: aggregate retention_loss_margin_min {retention_margin_min:.9f} "
                f"below floor {args.min_aggregate_retention_loss_margin:.9f}"
            )
        if args.require_checkpoint_match:
            failures.extend(checkpoint_audit_failures(strategy, now, before))

    if failures:
        raise RuntimeError("handoff aggregate regression gate failed: " + "; ".join(failures))
    return len(common)


def main():
    args = parse_args()
    rows = []
    cases = selected_cases(args.cases)
    strategies = configured_strategies(args)
    print(
        f"handoff_strategy_sweep key_preset={args.key_preset} "
        f"cases={len(cases)} strategies={len(strategies)}"
    )
    for case in cases:
        source_samples = st.dataset.byte_lm_windows(case["source_text"], CONTEXT)
        target_samples = st.dataset.byte_lm_windows(case["target_text"], CONTEXT)

        source_session, source_trainer, source_schedule = new_runtime()
        source_model, source_stats, source_delta = train_source(
            source_session,
            source_trainer,
            source_schedule,
            SoftmaxCrossEntropy(),
            source_samples,
        )
        source_state = source_model.state_dict()
        print(
            f"handoff_strategy_case case={case['label']} "
            f"source_windows={len(source_samples)} target_windows={len(target_samples)} "
            f"source_loss_delta={metric(source_delta, 'loss_delta'):.6f}"
        )

        for strategy in strategies:
            rows.append(
                run_strategy(
                    strategy,
                    case,
                    source_state,
                    source_delta,
                    source_stats,
                    source_samples,
                    target_samples,
                    key_preset=args.key_preset,
                )
            )

    winner, score = strategy_winner(rows)
    print(
        f"handoff_strategy_winner strategy={winner} "
        f"target_loss_delta={score[0]:.6f} "
        f"retention_loss_delta={score[1]:.6f} "
        f"retention_accuracy_delta={score[2]:.6f}"
    )
    aggregate_rows = aggregate_strategy_rows(rows)
    aggregate_winner_label, aggregate_winner_score = optional_aggregate_winner(aggregate_rows)
    for row in aggregate_rows:
        print(
            f"handoff_strategy_aggregate strategy={row['strategy']} "
            f"cases={row['cases']} "
            f"case_labels={row['case_labels']} "
            f"target_loss_delta_mean={row['target_loss_delta_mean']:.6f} "
            f"retention_loss_delta_mean={row['retention_loss_delta_mean']:.6f} "
            f"target_loss_margin_min={row['target_loss_margin_min']:.6f} "
            f"retention_loss_margin_min={row['retention_loss_margin_min']:.6f} "
            f"accepted_cases={row['accepted_cases']} "
            f"accepted_rate={row['accepted_rate']:.6f} "
            f"accepted_all={row['accepted_all']} "
            f"movement_ok_cases={row['movement_ok_cases']} "
            f"movement_ok_rate={row['movement_ok_rate']:.6f} "
            f"movement_ok_all={row['movement_ok_all']}"
        )
    if aggregate_winner_label is None:
        print("handoff_strategy_aggregate_winner strategy=none winner_available=false")
    else:
        print(
            f"handoff_strategy_aggregate_winner strategy={aggregate_winner_label} "
            f"target_loss_delta_mean={aggregate_winner_score[0]:.6f} "
            f"retention_loss_delta_mean={aggregate_winner_score[1]:.6f} "
            f"retention_accuracy_delta_mean={aggregate_winner_score[2]:.6f}"
        )
    if args.jsonl is not None:
        write_summary_jsonl(args.jsonl, rows)
        print(f"summary_jsonl={args.jsonl} rows={len(rows)}")
    if args.aggregate_jsonl is not None:
        write_summary_jsonl(args.aggregate_jsonl, aggregate_rows)
        print(f"aggregate_jsonl={args.aggregate_jsonl} rows={len(aggregate_rows)}")
    aggregate_coverage_gate = (
        args.require_aggregate_accepted_all
        or args.min_aggregate_cases is not None
        or bool(args.require_aggregate_cases)
        or args.min_aggregate_accepted_rate is not None
        or args.min_aggregate_movement_ok_rate is not None
    )
    if aggregate_coverage_gate:
        checked = check_aggregate_coverage(
            aggregate_rows,
            require_accepted_all=args.require_aggregate_accepted_all,
            min_cases=args.min_aggregate_cases,
            required_cases=args.require_aggregate_cases,
            min_accepted_rate=args.min_aggregate_accepted_rate,
            min_movement_ok_rate=args.min_aggregate_movement_ok_rate,
        )
        print(f"aggregate_coverage_rows={checked}")
    if args.compare_jsonl is not None:
        compared = compare_strategy_rows(rows, load_summary_jsonl(args.compare_jsonl), args)
        print(f"summary_compare_rows={compared} baseline={args.compare_jsonl}")
    if args.compare_aggregate_jsonl is not None:
        compared = compare_aggregate_rows(
            aggregate_rows,
            load_summary_jsonl(args.compare_aggregate_jsonl),
            args,
        )
        print(f"aggregate_compare_rows={compared} baseline={args.compare_aggregate_jsonl}")


if __name__ == "__main__":
    main()
