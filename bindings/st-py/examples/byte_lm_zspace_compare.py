import argparse
import json
from pathlib import Path

import spiraltorch as st
from spiraltorch.nn import Linear, Sequential, SoftmaxCrossEntropy, ZSpaceProjector
from spiraltorch.nn import sparse_classification_delta

from sparse_finetune_compare import (
    add_summary_compare_args,
    attach_summary_guard_counts,
    attach_summary_guard_margins,
    checkpoint_audit_differences,
    checkpoint_audit_failures,
    compare_summaries,
    summary_bool_value,
    summary_guard_margins,
    summary_compare_failures,
    validate_summary_compare_args,
)


VOCAB = st.dataset.BYTE_LM_VOCAB
HIDDEN = 24
CONTEXT = 8
BATCH_WINDOWS = 4
ACCUMULATION_STEPS = 2
PARAM_SCALE = 0.003
FT_EPOCHS = 3
FT_TARGET_MIN_LOSS_DELTA = 1e-4
FT_MOVEMENT_TOLERANCE = 1e-6
ZSPACE_FREQUENCY = 0.65
PROJECTION_PROBE_SAMPLES_PER_SPLIT = 1
PROJECTION_VARIANCE_COLLAPSE_RATIO = 0.25
PROJECTION_NORM_EXPANSION_RATIO = 3.0
EPSILON = 1e-12


ROUTE_SPECS = [
    {"label": "baseline", "strength": None, "curvature": -1.0},
    {"label": "zspace_s025", "strength": 0.25, "curvature": -1.0},
    {"label": "zspace_s050", "strength": 0.50, "curvature": -1.0},
    {"label": "zspace_s035_c025", "strength": 0.35, "curvature": -0.25},
    {"label": "zspace_post_s050_c025", "strength": 0.50, "curvature": -0.25},
    {"label": "zspace_s075_c025", "strength": 0.75, "curvature": -0.25},
    {"label": "zspace_s100_c025", "strength": 1.00, "curvature": -0.25},
    {"label": "zspace_s050_c010", "strength": 0.50, "curvature": -0.10},
    {"label": "zspace_s075_c010", "strength": 0.75, "curvature": -0.10},
    {"label": "zspace_s100_c010", "strength": 1.00, "curvature": -0.10},
    {"label": "zspace_s100_c0075", "strength": 1.00, "curvature": -0.075},
    {"label": "zspace_s100_c005", "strength": 1.00, "curvature": -0.05},
    {"label": "zspace_s100_c004", "strength": 1.00, "curvature": -0.04},
    {"label": "zspace_s100_c0035", "strength": 1.00, "curvature": -0.035},
    {"label": "zspace_s100_c0025", "strength": 1.00, "curvature": -0.025},
    {"label": "zspace_s090_c0025", "strength": 0.90, "curvature": -0.025},
    {"label": "zspace_s100_c001", "strength": 1.00, "curvature": -0.01},
    {"label": "zspace_s100_c0005", "strength": 1.00, "curvature": -0.005},
    {"label": "zspace_s075_c005", "strength": 0.75, "curvature": -0.05},
    {"label": "zspace_s075_c050", "strength": 0.75, "curvature": -0.50},
    {"label": "zspace_s050_c050", "strength": 0.50, "curvature": -0.50},
]
ROUTE_SPEC_BY_LABEL = {spec["label"]: spec for spec in ROUTE_SPECS}
DEFAULT_ROUTE_LABELS = [
    "baseline",
    "zspace_s025",
    "zspace_s050",
    "zspace_post_s050_c025",
]
FINE_ROUTE_LABELS = [
    "baseline",
    "zspace_s035_c025",
    "zspace_post_s050_c025",
    "zspace_s075_c025",
    "zspace_s050_c050",
]
RIDGE_ROUTE_LABELS = [
    "baseline",
    "zspace_post_s050_c025",
    "zspace_s075_c025",
    "zspace_s100_c025",
    "zspace_s075_c010",
    "zspace_s075_c050",
]
CREST_ROUTE_LABELS = [
    "baseline",
    "zspace_s050_c010",
    "zspace_s075_c010",
    "zspace_s100_c010",
    "zspace_s075_c005",
    "zspace_s075_c025",
]
SUMMIT_ROUTE_LABELS = [
    "baseline",
    "zspace_s075_c005",
    "zspace_s100_c010",
    "zspace_s100_c005",
    "zspace_s100_c0025",
    "zspace_s100_c025",
]
HORIZON_ROUTE_LABELS = [
    "baseline",
    "zspace_s100_c010",
    "zspace_s100_c005",
    "zspace_s100_c0025",
    "zspace_s100_c001",
    "zspace_s100_c0005",
]
HEALTH_ROUTE_LABELS = [
    "baseline",
    "zspace_s100_c010",
    "zspace_s100_c0075",
    "zspace_s100_c005",
    "zspace_s100_c004",
    "zspace_s100_c0035",
    "zspace_s090_c0025",
]

CASE_SPECS = [
    {
        "label": "byte_patterns_to_jp",
        "source_docs": [
            "spiraltorch learns byte patterns",
            "byte routes remember source structure",
        ],
        "target_docs": [
            "螺旋byteは猫byteを忘れない",
            "zspace byte route",
        ],
    },
    {
        "label": "routes_to_cats",
        "source_docs": [
            "graphs route tensors; routes graph byte",
            "routes keep tensor memory",
        ],
        "target_docs": [
            "猫byte螺旋byte",
            "route猫byte",
        ],
    },
    {
        "label": "geometry_tokens",
        "source_docs": [
            "hyperbolic geometry keeps local token memories aligned",
            "zspace routes curvature through byte language",
        ],
        "target_docs": [
            "zspace geometry adapter token route",
            "curvature byte route",
        ],
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare tiny tokenizerless byte-LM baseline and Z-space routes."
    )
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        choices=[case["label"] for case in CASE_SPECS],
        help="Run only this corpus case. May be repeated.",
    )
    parser.add_argument(
        "--route",
        dest="routes",
        action="append",
        choices=[spec["label"] for spec in ROUTE_SPECS],
        help=(
            "Run only this route. May be repeated; baseline is added automatically "
            "when comparing a Z-space route."
        ),
    )
    parser.add_argument(
        "--route-preset",
        choices=[
            "default",
            "fine",
            "ridge",
            "crest",
            "summit",
            "horizon",
            "health",
            "all",
        ],
        default="default",
        help=(
            "Route set to run when --route is not provided. 'fine' explores "
            "strength/curvature neighbors around zspace_post_s050_c025; "
            "'ridge' explores strength/curvature neighbors around zspace_s075_c025; "
            "'crest' explores the shallower-curvature ridge around zspace_s075_c010; "
            "'summit' brackets the strength=1.0 shallow-curvature winners; "
            "'horizon' checks whether the near-zero curvature edge keeps improving; "
            "'health' refines the strongest projection-healthy shallow route."
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
        help="Optional path for route-level aggregate metric rows.",
    )
    parser.add_argument(
        "--compare-aggregate-jsonl",
        type=Path,
        default=None,
        help="Optional previous route-level aggregate JSONL to compare against this run.",
    )
    parser.add_argument(
        "--compare-jsonl",
        type=Path,
        default=None,
        help="Optional previous route summary JSONL to compare against this run.",
    )
    parser.add_argument(
        "--max-aggregate-source-loss-regression",
        type=float,
        default=None,
        help=(
            "Fail when aggregate source_loss_delta_mean regresses from "
            "--compare-aggregate-jsonl by more than this non-negative tolerance."
        ),
    )
    parser.add_argument(
        "--max-aggregate-ft-loss-regression",
        type=float,
        default=None,
        help=(
            "Fail when aggregate ft_loss_delta_mean regresses from "
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
        help=(
            "Fail when a current route's aggregate target_loss_margin_min is "
            "below this non-negative floor."
        ),
    )
    parser.add_argument(
        "--min-aggregate-retention-loss-margin",
        type=float,
        default=None,
        help=(
            "Fail when a current route's aggregate retention_loss_margin_min "
            "is below this non-negative floor."
        ),
    )
    parser.add_argument(
        "--min-aggregate-retention-accuracy-margin",
        type=float,
        default=None,
        help=(
            "Fail when a current route's aggregate retention_accuracy_margin_min "
            "is below this non-negative floor."
        ),
    )
    parser.add_argument(
        "--min-aggregate-retention-perplexity-margin",
        type=float,
        default=None,
        help=(
            "Fail when a current route's aggregate retention_perplexity_margin_min "
            "is absent or below this non-negative floor."
        ),
    )
    parser.add_argument(
        "--require-aggregate-winner-match",
        action="store_true",
        help="Fail when aggregate source/FT/retention winner routes change.",
    )
    parser.add_argument(
        "--require-aggregate-accepted-all",
        action="store_true",
        help="Fail when any current route aggregate has a non-accepted case.",
    )
    parser.add_argument(
        "--min-aggregate-cases",
        type=int,
        default=None,
        help="Fail when a current route aggregate includes fewer than this many cases.",
    )
    parser.add_argument(
        "--require-aggregate-case",
        dest="require_aggregate_cases",
        action="append",
        choices=[case["label"] for case in CASE_SPECS],
        default=[],
        help="Fail when a current route aggregate is missing this case label. May be repeated.",
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
    parser.add_argument(
        "--allow-zspace-nonadvantage",
        action="store_true",
        help=(
            "Do not fail when the best Z-space aggregate route does not beat "
            "baseline; still print the advantage diagnostics."
        ),
    )
    add_summary_compare_args(parser, subject="route")
    args = parser.parse_args()
    if (
        args.max_aggregate_source_loss_regression is not None
        and args.max_aggregate_source_loss_regression < 0.0
    ):
        parser.error("--max-aggregate-source-loss-regression must be non-negative")
    if (
        args.max_aggregate_ft_loss_regression is not None
        and args.max_aggregate_ft_loss_regression < 0.0
    ):
        parser.error("--max-aggregate-ft-loss-regression must be non-negative")
    if (
        args.max_aggregate_retention_loss_regression is not None
        and args.max_aggregate_retention_loss_regression < 0.0
    ):
        parser.error("--max-aggregate-retention-loss-regression must be non-negative")
    for name in [
        "max_aggregate_accepted_rate_regression",
        "max_aggregate_movement_ok_rate_regression",
        "min_aggregate_accepted_rate",
        "min_aggregate_movement_ok_rate",
    ]:
        value = getattr(args, name)
        if value is not None and value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
        if value is not None and value > 1.0:
            parser.error(f"--{name.replace('_', '-')} must be at most 1.0")
    if args.min_aggregate_cases is not None and args.min_aggregate_cases <= 0:
        parser.error("--min-aggregate-cases must be positive")
    if len(set(args.require_aggregate_cases)) != len(args.require_aggregate_cases):
        parser.error("--require-aggregate-case values must be unique")
    for name in [
        "min_aggregate_target_loss_margin",
        "min_aggregate_retention_loss_margin",
        "min_aggregate_retention_accuracy_margin",
        "min_aggregate_retention_perplexity_margin",
    ]:
        value = getattr(args, name)
        if value is not None and value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    gate_requested = validate_summary_compare_args(parser, args)
    if args.compare_jsonl is None and gate_requested:
        parser.error("regression gate options require --compare-jsonl")
    aggregate_gate_requested = (
        args.max_aggregate_source_loss_regression is not None
        or args.max_aggregate_ft_loss_regression is not None
        or args.max_aggregate_retention_loss_regression is not None
        or args.max_aggregate_accepted_rate_regression is not None
        or args.max_aggregate_movement_ok_rate_regression is not None
        or args.min_aggregate_target_loss_margin is not None
        or args.min_aggregate_retention_loss_margin is not None
        or args.min_aggregate_retention_accuracy_margin is not None
        or args.min_aggregate_retention_perplexity_margin is not None
        or args.require_aggregate_winner_match
    )
    if args.compare_aggregate_jsonl is None and aggregate_gate_requested:
        parser.error("aggregate regression gate options require --compare-aggregate-jsonl")
    if args.routes is not None and set(args.routes) == {"baseline"}:
        parser.error("--route must include at least one Z-space route")
    return args


def loader(samples, seed):
    return st.dataset.from_vec(samples).shuffle(seed).batched(BATCH_WINDOWS)


def evaluate(session, trainer, model, loss, samples, seed):
    return session.evaluate_sparse_classification_epoch(
        trainer,
        model,
        loss,
        loader(samples, seed),
    )


def route_label(spec):
    return spec["label"]


def route_specs_for_labels(labels):
    return [ROUTE_SPEC_BY_LABEL[label] for label in labels]


def selected_cases(labels):
    if labels is None:
        return CASE_SPECS
    selected = set(labels)
    return [case for case in CASE_SPECS if case["label"] in selected]


def selected_routes(labels, preset="default"):
    if labels is None:
        if preset == "default":
            return route_specs_for_labels(DEFAULT_ROUTE_LABELS)
        elif preset == "fine":
            return route_specs_for_labels(FINE_ROUTE_LABELS)
        elif preset == "ridge":
            return route_specs_for_labels(RIDGE_ROUTE_LABELS)
        elif preset == "crest":
            return route_specs_for_labels(CREST_ROUTE_LABELS)
        elif preset == "summit":
            return route_specs_for_labels(SUMMIT_ROUTE_LABELS)
        elif preset == "horizon":
            return route_specs_for_labels(HORIZON_ROUTE_LABELS)
        elif preset == "health":
            return route_specs_for_labels(HEALTH_ROUTE_LABELS)
        elif preset == "all":
            return ROUTE_SPECS
        else:
            raise ValueError(f"unknown route preset: {preset!r}")
    selected = list(dict.fromkeys(labels))
    if "baseline" not in selected:
        selected.insert(0, "baseline")
    return route_specs_for_labels(selected)


def make_model(spec):
    layers = [Linear(VOCAB, HIDDEN, name="embed")]
    if spec["strength"] is not None:
        topos = st.OpenTopos(spec["curvature"], 1e-5, 10.0, 256, 16384)
        encoder = st.LanguageWaveEncoder(topos.curvature(), ZSPACE_FREQUENCY)
        layers.append(ZSpaceProjector(topos, encoder, strength=spec["strength"]))
    layers.append(Linear(HIDDEN, VOCAB, name="head"))
    model = Sequential(layers)
    scale_state_dict(model, PARAM_SCALE)
    return model


def scale_state_dict(model, factor):
    scaled = {}
    for name, tensor in model.state_dict().items():
        scaled[name] = st.Tensor(
            tensor.rows,
            tensor.cols,
            [value * factor for value in tensor.data()],
        )
    model.load_state_dict(scaled)


def column_variance_mean(tensor):
    rows, cols = tensor.shape()
    if rows <= 1 or cols == 0:
        return 0.0
    data = tensor.data()
    total = 0.0
    for col in range(cols):
        values = [data[row * cols + col] for row in range(rows)]
        mean = sum(values) / rows
        total += sum((value - mean) ** 2 for value in values) / rows
    return total / cols


def projection_probe_inputs(source_samples, target_samples):
    return (
        list(source_samples[:PROJECTION_PROBE_SAMPLES_PER_SPLIT])
        + list(target_samples[:PROJECTION_PROBE_SAMPLES_PER_SPLIT])
    )


def projection_diagnostics(model, spec, samples):
    if spec["strength"] is None:
        return {
            "projection_probe_samples": 0,
            "projection_probe_rows": 0,
            "projection_input_l2": None,
            "projection_output_l2": None,
            "projection_delta_l2": None,
            "projection_delta_input_l2_ratio": None,
            "projection_output_input_l2_ratio": None,
            "projection_input_col_variance_mean": None,
            "projection_output_col_variance_mean": None,
            "projection_output_input_col_variance_ratio": None,
        }
    probes = list(samples)
    if not probes:
        return {
            "projection_probe_samples": 0,
            "projection_probe_rows": 0,
            "projection_input_l2": 0.0,
            "projection_output_l2": 0.0,
            "projection_delta_l2": 0.0,
            "projection_delta_input_l2_ratio": None,
            "projection_output_input_l2_ratio": None,
            "projection_input_col_variance_mean": None,
            "projection_output_col_variance_mean": None,
            "projection_output_input_col_variance_ratio": None,
        }
    state = model.state_dict()
    topos = st.OpenTopos(spec["curvature"], 1e-5, 10.0, 256, 16384)
    encoder = st.LanguageWaveEncoder(topos.curvature(), ZSPACE_FREQUENCY)
    projector = ZSpaceProjector(topos, encoder, strength=spec["strength"])
    input_l2_sq = 0.0
    output_l2_sq = 0.0
    delta_l2_sq = 0.0
    input_variance_weighted = 0.0
    output_variance_weighted = 0.0
    rows_total = 0
    for input_tensor, _target_tensor in probes:
        hidden = input_tensor.matmul(state["embed::weight"])
        hidden.add_row_inplace(state["embed::bias"].data())
        projected = projector.forward(hidden)
        delta = projected.sub(hidden)
        rows = hidden.rows
        rows_total += rows
        input_l2_sq += hidden.squared_l2_norm()
        output_l2_sq += projected.squared_l2_norm()
        delta_l2_sq += delta.squared_l2_norm()
        input_variance_weighted += column_variance_mean(hidden) * rows
        output_variance_weighted += column_variance_mean(projected) * rows
    input_l2 = input_l2_sq**0.5
    output_l2 = output_l2_sq**0.5
    delta_l2 = delta_l2_sq**0.5
    input_variance = (
        input_variance_weighted / rows_total if rows_total > 0 else None
    )
    output_variance = (
        output_variance_weighted / rows_total if rows_total > 0 else None
    )
    return {
        "projection_probe_samples": len(probes),
        "projection_probe_rows": rows_total,
        "projection_input_l2": input_l2,
        "projection_output_l2": output_l2,
        "projection_delta_l2": delta_l2,
        "projection_delta_input_l2_ratio": (
            delta_l2 / input_l2 if input_l2 > EPSILON else None
        ),
        "projection_output_input_l2_ratio": (
            output_l2 / input_l2 if input_l2 > EPSILON else None
        ),
        "projection_input_col_variance_mean": input_variance,
        "projection_output_col_variance_mean": output_variance,
        "projection_output_input_col_variance_ratio": (
            output_variance / input_variance
            if input_variance is not None and input_variance > EPSILON
            else None
        ),
    }


def require_loss_delta(label, delta):
    if delta["loss_delta"] <= 0.0:
        raise RuntimeError(f"{label} loss did not improve: {delta}")


def run_route(case, spec, source_samples, target_samples):
    case_label = case["label"]
    route = route_label(spec)
    session = st.SpiralSession(
        device="wgpu",
        curvature=-1.0,
        hyper_learning_rate=0.08,
        fallback_learning_rate=0.02,
    )
    trainer = session.trainer()
    trainer.set_max_grad_norm(0.75)
    trainer.set_gradient_accumulation_steps(ACCUMULATION_STEPS)
    schedule = trainer.roundtable(rows=CONTEXT * BATCH_WINDOWS, cols=VOCAB)
    loss = SoftmaxCrossEntropy()

    source = make_model(spec)
    session.prepare_module(source)
    source_before = evaluate(session, trainer, source, loss, source_samples, seed=7)
    session.train_epoch(
        trainer,
        source,
        loss,
        loader(source_samples, seed=7),
        schedule,
    )
    source_after = evaluate(session, trainer, source, loss, source_samples, seed=7)
    source_delta = sparse_classification_delta(source_before, source_after)
    require_loss_delta(f"{case_label} {route} source", source_delta)
    projection = projection_diagnostics(
        source,
        spec,
        projection_probe_inputs(source_samples, target_samples),
    )

    target = make_model(spec)
    session.prepare_module(target)
    load = target.load_state_dict_checked(source.state_dict())
    if not load["matched"]:
        raise RuntimeError(f"{case_label} {route} checkpoint fingerprint mismatch: {load}")

    frozen_embed = target.set_parameters_trainable_by_prefix("embed::", False)
    boosted_head = target.set_parameters_learning_rate_scale_by_prefix("head::", 1.1)
    ft_train_samples = st.dataset.interleave_replay_samples(
        target_samples,
        source_samples,
        target_per_replay=1,
    )
    ft_train_rows = st.dataset.byte_lm_sample_stats(ft_train_samples)["active_rows"]
    ft_report = session.train_epochs_restore_best_sparse_with_finetune_report(
        trainer,
        target,
        loss,
        loader(ft_train_samples, seed=11),
        loader(target_samples, seed=17),
        loader(source_samples, seed=13),
        schedule,
        epochs=FT_EPOCHS,
        movement_tolerance=FT_MOVEMENT_TOLERANCE,
        max_loss_increase=0.5,
        max_accuracy_drop=0.15,
        max_perplexity_increase=1.0,
        target_min_loss_delta=FT_TARGET_MIN_LOSS_DELTA,
    )
    ft_capture = ft_report.captured
    if ft_capture.guarded_best_epoch is None:
        raise RuntimeError(f"{case_label} {route} retention guard rejected every fine-tune epoch")
    ft_delta = ft_report.target_delta
    retention_delta = ft_report.retention_delta
    ft_summary = ft_report.summary()
    require_loss_delta(f"{case_label} {route} fine_tune", ft_delta)

    movement = ft_report.movement
    if not ft_report.movement_ok:
        raise RuntimeError(f"{case_label} {route} unexpected parameter movement: {movement}")

    return {
        "case": case_label,
        "route": route,
        "zspace_strength": spec["strength"],
        "zspace_curvature": spec["curvature"],
        "zspace_frequency": ZSPACE_FREQUENCY if spec["strength"] is not None else None,
        "source": source_delta,
        "ft": ft_delta,
        "retention": retention_delta,
        "ft_capture": ft_capture,
        "ft_report": ft_report,
        "ft_summary": ft_summary,
        "ft_train_samples": len(ft_train_samples),
        "ft_train_rows": ft_train_rows,
        "frozen_embed": frozen_embed,
        "boosted_head": boosted_head,
        "movement": movement,
        "projection": projection,
    }


def metric(delta, name):
    return delta[name]


def print_report(report):
    margins = summary_guard_margins(report["ft_summary"])
    print(
        f"case={report['case']} "
        f"route={report['route']} "
        f"zspace_strength={report['zspace_strength'] if report['zspace_strength'] is not None else 'none'} "
        f"zspace_curvature={report['zspace_curvature']:.3f} "
        f"source_loss_delta={metric(report['source'], 'loss_delta'):.6f} "
        f"source_accuracy_delta={metric(report['source'], 'accuracy_delta'):.6f} "
        f"source_perplexity_delta={metric(report['source'], 'perplexity_delta'):.6f} "
        f"ft_loss_delta={metric(report['ft'], 'loss_delta'):.6f} "
        f"ft_accuracy_delta={metric(report['ft'], 'accuracy_delta'):.6f} "
        f"ft_perplexity_delta={metric(report['ft'], 'perplexity_delta'):.6f} "
        f"retention_loss_delta={metric(report['retention'], 'loss_delta'):.6f} "
        f"retention_accuracy_delta={metric(report['retention'], 'accuracy_delta'):.6f} "
        f"retention_perplexity_delta={metric(report['retention'], 'perplexity_delta'):.6f} "
        f"ft_train_samples={report['ft_train_samples']} "
        f"ft_train_rows={report['ft_train_rows']} "
        f"guarded_best_epoch={report['ft_capture'].guarded_best_epoch} "
        f"retention_loss_ceiling={report['ft_capture'].max_allowed_retention_loss:.6f} "
        f"retention_accuracy_floor={report['ft_capture'].min_allowed_retention_accuracy:.6f} "
        f"target_min_loss_delta={report['ft_capture'].retention_guard['target_min_loss_delta']:.6f} "
        f"target_loss_margin={margins['target_loss_margin']:.6f} "
        f"retention_loss_margin={margins['retention_loss_margin']:.6f} "
        f"retention_accuracy_margin={margins['retention_accuracy_margin']:.6f} "
        f"movement_tolerance={report['ft_summary']['movement_tolerance']:.6f} "
        f"projection_delta_input_l2_ratio={fmt_optional(report['projection']['projection_delta_input_l2_ratio'])} "
        f"projection_output_input_l2_ratio={fmt_optional(report['projection']['projection_output_input_l2_ratio'])} "
        f"projection_output_input_col_variance_ratio={fmt_optional(report['projection']['projection_output_input_col_variance_ratio'])} "
        f"resume_hash={report['ft_summary']['resume_hash']} "
        f"resume_trainer_hash={report['ft_summary']['resume_trainer_hash']} "
        f"resume_parameter_training_hash={report['ft_summary']['resume_parameter_training_hash']} "
        f"best_retention_loss_increase={report['ft_capture'].best_retention_loss_increase:.6f} "
        f"best_retention_accuracy_drop={report['ft_capture'].best_retention_accuracy_drop:.6f} "
        f"best_retention_perplexity_increase={report['ft_capture'].best_retention_perplexity_increase:.6f} "
        f"report_status={report['ft_report'].status} "
        f"frozen_embed_params={report['frozen_embed']} "
        f"boosted_head_params={report['boosted_head']} "
        f"movement_status={report['movement']['status']}"
    )


def summary_row(report, source_docs, target_docs, source_rows, target_rows):
    row = dict(report["ft_summary"])
    attach_summary_guard_margins(row)
    attach_summary_guard_counts(row, report["ft_report"].captured)
    row.update(
        {
            "example": "byte_lm_zspace_compare",
            "case": report["case"],
            "route": report["route"],
            "zspace_strength": report["zspace_strength"],
            "zspace_curvature": report["zspace_curvature"],
            "zspace_frequency": report["zspace_frequency"],
            "source_docs": source_docs,
            "target_docs": target_docs,
            "source_rows": source_rows,
            "target_rows": target_rows,
            "ft_train_samples": report["ft_train_samples"],
            "ft_train_rows": report["ft_train_rows"],
            "source_loss_delta": metric(report["source"], "loss_delta"),
            "source_accuracy_delta": metric(report["source"], "accuracy_delta"),
            "source_perplexity_delta": metric(report["source"], "perplexity_delta"),
            "frozen_embed_params": report["frozen_embed"],
            "boosted_head_params": report["boosted_head"],
        }
    )
    row.update(report["projection"])
    return row


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def load_jsonl(path):
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
            if "case" not in row:
                raise ValueError(f"{path}:{line_no} missing 'case'")
            if "route" not in row:
                raise ValueError(f"{path}:{line_no} missing 'route'")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} did not contain any summary rows")
    return rows


def load_aggregate_jsonl(path):
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
            if row.get("row_type") != "route_aggregate":
                raise ValueError(f"{path}:{line_no} expected row_type='route_aggregate'")
            if "route" not in row:
                raise ValueError(f"{path}:{line_no} missing 'route'")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} did not contain any aggregate rows")
    return rows


def row_key(row):
    case = row.get("case")
    route = row.get("route")
    if not isinstance(case, str) or not case:
        raise ValueError(f"row has invalid case: {case!r}")
    if not isinstance(route, str) or not route:
        raise ValueError(f"row has invalid route: {route!r}")
    return f"{case}::{route}"


def rows_by_case_route(rows, label):
    by_key = {}
    for row in rows:
        key = row_key(row)
        if key in by_key:
            raise ValueError(f"{label} contains duplicate case/route: {key}")
        by_key[key] = row
    return by_key


def aggregate_rows_by_route(rows, label):
    by_route = {}
    for row in rows:
        route = row.get("route")
        if not isinstance(route, str) or not route:
            raise ValueError(f"{label} aggregate row has invalid route: {route!r}")
        if route in by_route:
            raise ValueError(f"{label} contains duplicate aggregate route: {route}")
        validate_aggregate_row(row, label)
        by_route[route] = row
    return by_route


def optional_int_value(row, key):
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def required_aggregate_int_value(row, key):
    value = optional_int_value(row, key)
    if value is None:
        raise ValueError(f"aggregate row {row.get('route')} missing integer {key}")
    return value


def required_aggregate_bool_value(row, key):
    value = row.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"aggregate row {row.get('route')} missing boolean {key}")
    return value


def is_numeric_value(value):
    return not isinstance(value, bool) and isinstance(value, (int, float))


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
    route = row.get("route")
    prefix = f"{label} aggregate {route}"
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

    routes = optional_int_value(row, "routes")
    if routes is None or routes <= 0:
        failures.append(f"{prefix}: routes must be a positive integer")

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
            if not isinstance(all_value, bool):
                failures.append(f"{prefix}: {all_key} must be boolean")
            elif all_value != (count == cases):
                failures.append(
                    f"{prefix}: {all_key} {all_value} inconsistent with {count_key}/{cases}"
                )
    return failures


def validate_aggregate_row(row, label):
    failures = aggregate_row_consistency_failures(row, label)
    if failures:
        raise ValueError("; ".join(failures))
    return row


def numeric_value(row, key):
    value = row.get(key)
    if not is_numeric_value(value):
        raise ValueError(f"aggregate row {row.get('route')} missing numeric {key}")
    return float(value)


def optional_numeric_value(row, key):
    value = row.get(key)
    if value is None:
        return None
    if not is_numeric_value(value):
        raise ValueError(f"aggregate row {row.get('route')} has non-numeric {key}")
    return float(value)


def aggregate_winner(rows, metric_name):
    if not rows:
        return "none"
    best = max(numeric_value(row, metric_name) for row in rows)
    winners = [
        row["route"]
        for row in rows
        if abs(numeric_value(row, metric_name) - best) <= 1e-9
    ]
    return "+".join(sorted(winners))


def aggregate_winners(rows):
    return {
        "source": aggregate_winner(rows, "source_loss_delta_mean"),
        "ft": aggregate_winner(rows, "ft_loss_delta_mean"),
        "retention": aggregate_winner(rows, "retention_loss_delta_mean"),
    }


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
        route = row.get("route")
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
            f"aggregate_acceptance route={route} "
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
                f"{route}: aggregate cases {cases} below floor {min_cases}"
            )
        if missing_cases:
            failures.append(
                f"{route}: missing aggregate cases {','.join(missing_cases)}"
            )
        if require_accepted_all and not accepted_all:
            failures.append(f"{route}: accepted {accepted_cases}/{cases} aggregate cases")
        if min_accepted_rate is not None and accepted_rate < min_accepted_rate:
            failures.append(
                f"{route}: accepted_rate {accepted_rate:.9f} below floor {min_accepted_rate:.9f}"
            )
        if min_movement_ok_rate is not None and movement_ok_rate < min_movement_ok_rate:
            failures.append(
                f"{route}: movement_ok_rate {movement_ok_rate:.9f} below floor {min_movement_ok_rate:.9f}"
            )
    if failures:
        raise RuntimeError("Z-space aggregate coverage gate failed: " + "; ".join(failures))
    return len(rows)


def compare_summary_rows(
    current_rows,
    baseline_rows,
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
    require_checkpoint_match,
    allow_missing_current=False,
):
    current = rows_by_case_route(current_rows, "current")
    baseline = rows_by_case_route(baseline_rows, "baseline")
    missing_current = sorted(set(baseline) - set(current))
    if missing_current:
        for key in missing_current:
            row = baseline[key]
            print(f"summary_compare case={row['case']} route={row['route']} current_missing=true")
        if not allow_missing_current:
            raise RuntimeError(
                "baseline case/routes missing from current compare: " + ",".join(missing_current)
            )

    common = sorted(set(current) & set(baseline))
    if not common:
        raise RuntimeError("no overlapping case/routes between current compare and baseline")

    failures = []
    for key in common:
        now = current[key]
        before = baseline[key]
        checkpoint_changed = bool(checkpoint_audit_differences(now, before))
        comparison = compare_summaries(
            now,
            before,
            max_target_loss_regression=max_target_loss_regression,
            max_retention_loss_regression=max_retention_loss_regression,
            min_target_loss_margin=min_target_loss_margin,
            min_retention_loss_margin=min_retention_loss_margin,
            min_retention_accuracy_margin=min_retention_accuracy_margin,
            min_retention_perplexity_margin=min_retention_perplexity_margin,
            require_status_match=require_status_match,
            require_accepted_match=require_accepted_match,
            require_guard_match=require_guard_match,
            require_movement_tolerance_match=require_movement_tolerance_match,
            require_resume_match=require_resume_match,
        )
        case = now["case"]
        route = now["route"]
        print(
            f"summary_compare case={case} "
            f"route={route} "
            f"target_loss_delta_change={comparison['target_loss_delta_change']:.9f} "
            f"retention_loss_delta_change={comparison['retention_loss_delta_change']:.9f} "
            f"target_loss_regression={comparison['target_loss_regression']:.9f} "
            f"retention_loss_regression={comparison['retention_loss_regression']:.9f} "
            f"target_loss_margin={comparison['current_target_loss_margin']:.9f} "
            f"retention_loss_margin={comparison['current_retention_loss_margin']:.9f} "
            f"retention_accuracy_margin={comparison['current_retention_accuracy_margin']:.9f} "
            f"status_before={comparison['baseline_status']} "
            f"status_after={comparison['current_status']} "
            f"accepted_before={comparison['baseline_accepted']} "
            f"accepted_after={comparison['current_accepted']} "
            f"accepted_changed={comparison['accepted_changed']} "
            f"guard_changed={comparison['guard_changed']} "
            f"movement_tolerance_before={comparison['baseline_movement_tolerance']:.9f} "
            f"movement_tolerance_after={comparison['current_movement_tolerance']:.9f} "
            f"movement_tolerance_changed={comparison['movement_tolerance_changed']} "
            f"resume_before={comparison['baseline_resume_hash']} "
            f"resume_after={comparison['current_resume_hash']} "
            f"resume_changed={comparison['resume_changed']} "
            f"checkpoint_changed={checkpoint_changed} "
            f"passed={comparison['passed']}"
        )
        failures.extend(
            summary_compare_failures(
                key,
                comparison,
                max_target_loss_regression=max_target_loss_regression,
                max_retention_loss_regression=max_retention_loss_regression,
                min_target_loss_margin=min_target_loss_margin,
                min_retention_loss_margin=min_retention_loss_margin,
                min_retention_accuracy_margin=min_retention_accuracy_margin,
                min_retention_perplexity_margin=min_retention_perplexity_margin,
                require_status_match=require_status_match,
                require_accepted_match=require_accepted_match,
                require_guard_match=require_guard_match,
                require_movement_tolerance_match=require_movement_tolerance_match,
                require_resume_match=require_resume_match,
            )
        )
        if require_checkpoint_match:
            failures.extend(checkpoint_audit_failures(key, now, before))

    extra_current = sorted(set(current) - set(baseline))
    for key in extra_current:
        row = current[key]
        print(f"summary_compare case={row['case']} route={row['route']} baseline_missing=true")
    if failures:
        raise RuntimeError("Z-space compare regression gate failed: " + "; ".join(failures))
    return len(common)


def compare_aggregate_rows(
    current_rows,
    baseline_rows,
    max_source_loss_regression,
    max_ft_loss_regression,
    max_retention_loss_regression,
    min_target_loss_margin,
    min_retention_loss_margin,
    min_retention_accuracy_margin,
    min_retention_perplexity_margin,
    require_winner_match,
    allow_missing_current=False,
    max_aggregate_accepted_rate_regression=None,
    max_aggregate_movement_ok_rate_regression=None,
):
    current = aggregate_rows_by_route(current_rows, "current")
    baseline = aggregate_rows_by_route(baseline_rows, "baseline")
    missing_current = sorted(set(baseline) - set(current))
    if missing_current:
        for route in missing_current:
            print(f"aggregate_compare route={route} current_missing=true")
        if not allow_missing_current:
            raise RuntimeError(
                "baseline aggregate routes missing from current compare: "
                + ",".join(missing_current)
            )

    common = sorted(set(current) & set(baseline))
    if not common:
        raise RuntimeError("no overlapping aggregate routes between current and baseline")

    failures = []
    current_winners = aggregate_winners(list(current.values()))
    baseline_winners = aggregate_winners(list(baseline.values()))
    route_set_changed = set(current) != set(baseline)
    winner_changed = current_winners != baseline_winners
    print(
        "aggregate_winner_compare "
        f"source_before={baseline_winners['source']} "
        f"source_after={current_winners['source']} "
        f"ft_before={baseline_winners['ft']} "
        f"ft_after={current_winners['ft']} "
        f"retention_before={baseline_winners['retention']} "
        f"retention_after={current_winners['retention']} "
        f"route_set_changed={route_set_changed} "
        f"winner_changed={winner_changed} "
        f"passed={not winner_changed or not require_winner_match}"
    )
    if winner_changed and require_winner_match:
        failures.append(
            "aggregate winners changed from "
            f"source={baseline_winners['source']},ft={baseline_winners['ft']},"
            f"retention={baseline_winners['retention']} to "
            f"source={current_winners['source']},ft={current_winners['ft']},"
            f"retention={current_winners['retention']}"
        )

    for route in common:
        now = current[route]
        before = baseline[route]
        source_change = (
            numeric_value(now, "source_loss_delta_mean")
            - numeric_value(before, "source_loss_delta_mean")
        )
        ft_change = (
            numeric_value(now, "ft_loss_delta_mean")
            - numeric_value(before, "ft_loss_delta_mean")
        )
        retention_change = (
            numeric_value(now, "retention_loss_delta_mean")
            - numeric_value(before, "retention_loss_delta_mean")
        )
        source_regression = max(0.0, -source_change)
        ft_regression = max(0.0, -ft_change)
        retention_regression = max(0.0, -retention_change)
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
            if max_aggregate_accepted_rate_regression is not None
            else 0.0
        )
        movement_ok_rate_regression = (
            max(
                0.0,
                numeric_value(before, "movement_ok_rate")
                - numeric_value(now, "movement_ok_rate"),
            )
            if max_aggregate_movement_ok_rate_regression is not None
            else 0.0
        )
        scope_changed = (
            now.get("case_labels") != before.get("case_labels")
            or now.get("cases") != before.get("cases")
            or now.get("routes") != before.get("routes")
        )
        source_advantage_change = (
            numeric_value(now, "source_loss_delta_advantage")
            - numeric_value(before, "source_loss_delta_advantage")
        )
        ft_advantage_change = (
            numeric_value(now, "ft_loss_delta_advantage")
            - numeric_value(before, "ft_loss_delta_advantage")
        )
        retention_advantage_change = (
            numeric_value(now, "retention_loss_delta_advantage")
            - numeric_value(before, "retention_loss_delta_advantage")
        )
        target_loss_margin_min = optional_numeric_value(now, "target_loss_margin_min")
        retention_loss_margin_min = optional_numeric_value(now, "retention_loss_margin_min")
        retention_accuracy_margin_min = optional_numeric_value(
            now, "retention_accuracy_margin_min"
        )
        retention_perplexity_margin_min = optional_numeric_value(
            now, "retention_perplexity_margin_min"
        )
        target_margin_ok = (
            min_target_loss_margin is None
            or (
                target_loss_margin_min is not None
                and target_loss_margin_min >= min_target_loss_margin
            )
        )
        retention_loss_margin_ok = (
            min_retention_loss_margin is None
            or (
                retention_loss_margin_min is not None
                and retention_loss_margin_min >= min_retention_loss_margin
            )
        )
        retention_accuracy_margin_ok = (
            min_retention_accuracy_margin is None
            or (
                retention_accuracy_margin_min is not None
                and retention_accuracy_margin_min >= min_retention_accuracy_margin
            )
        )
        retention_perplexity_margin_ok = (
            min_retention_perplexity_margin is None
            or (
                retention_perplexity_margin_min is not None
                and retention_perplexity_margin_min >= min_retention_perplexity_margin
            )
        )
        accepted_rate_ok = (
            max_aggregate_accepted_rate_regression is None
            or accepted_rate_regression <= max_aggregate_accepted_rate_regression
        )
        movement_ok_rate_ok = (
            max_aggregate_movement_ok_rate_regression is None
            or movement_ok_rate_regression <= max_aggregate_movement_ok_rate_regression
        )
        passed = (
            (
                max_source_loss_regression is None
                or source_regression <= max_source_loss_regression
            )
            and (
                max_ft_loss_regression is None
                or ft_regression <= max_ft_loss_regression
            )
            and (
                max_retention_loss_regression is None
                or retention_regression <= max_retention_loss_regression
            )
            and target_margin_ok
            and retention_loss_margin_ok
            and retention_accuracy_margin_ok
            and retention_perplexity_margin_ok
            and accepted_rate_ok
            and movement_ok_rate_ok
            and not scope_changed
        )
        print(
            f"aggregate_compare route={route} "
            f"source_loss_delta_mean_change={source_change:.9f} "
            f"ft_loss_delta_mean_change={ft_change:.9f} "
            f"retention_loss_delta_mean_change={retention_change:.9f} "
            f"source_loss_regression={source_regression:.9f} "
            f"ft_loss_regression={ft_regression:.9f} "
            f"retention_loss_regression={retention_regression:.9f} "
            f"accepted_rate_before={fmt_optional(accepted_rate_before)} "
            f"accepted_rate_after={fmt_optional(accepted_rate_after)} "
            f"accepted_rate_regression={accepted_rate_regression:.9f} "
            f"movement_ok_rate_before={fmt_optional(movement_ok_rate_before)} "
            f"movement_ok_rate_after={fmt_optional(movement_ok_rate_after)} "
            f"movement_ok_rate_regression={movement_ok_rate_regression:.9f} "
            f"source_advantage_change={source_advantage_change:.9f} "
            f"ft_advantage_change={ft_advantage_change:.9f} "
            f"retention_advantage_change={retention_advantage_change:.9f} "
            f"target_loss_margin_min={fmt_optional(target_loss_margin_min)} "
            f"retention_loss_margin_min={fmt_optional(retention_loss_margin_min)} "
            f"retention_accuracy_margin_min={fmt_optional(retention_accuracy_margin_min)} "
            f"retention_perplexity_margin_min={fmt_optional(retention_perplexity_margin_min)} "
            f"cases_before={before.get('cases')} "
            f"cases_after={now.get('cases')} "
            f"scope_changed={scope_changed} "
            f"passed={passed}"
        )
        if scope_changed:
            failures.append(f"{route}: aggregate case scope changed")
        if (
            max_source_loss_regression is not None
            and source_regression > max_source_loss_regression
        ):
            failures.append(
                f"{route}: source_loss_delta_mean regressed by {source_regression:.9f}"
            )
        if (
            max_ft_loss_regression is not None
            and ft_regression > max_ft_loss_regression
        ):
            failures.append(f"{route}: ft_loss_delta_mean regressed by {ft_regression:.9f}")
        if (
            max_retention_loss_regression is not None
            and retention_regression > max_retention_loss_regression
        ):
            failures.append(
                f"{route}: retention_loss_delta_mean regressed by {retention_regression:.9f}"
            )
        if (
            max_aggregate_accepted_rate_regression is not None
            and accepted_rate_regression > max_aggregate_accepted_rate_regression
        ):
            failures.append(
                f"{route}: aggregate accepted_rate regressed by {accepted_rate_regression:.9f}"
            )
        if (
            max_aggregate_movement_ok_rate_regression is not None
            and movement_ok_rate_regression > max_aggregate_movement_ok_rate_regression
        ):
            failures.append(
                f"{route}: aggregate movement_ok_rate regressed by {movement_ok_rate_regression:.9f}"
            )
        for value, floor, label in [
            (target_loss_margin_min, min_target_loss_margin, "target_loss_margin_min"),
            (
                retention_loss_margin_min,
                min_retention_loss_margin,
                "retention_loss_margin_min",
            ),
            (
                retention_accuracy_margin_min,
                min_retention_accuracy_margin,
                "retention_accuracy_margin_min",
            ),
            (
                retention_perplexity_margin_min,
                min_retention_perplexity_margin,
                "retention_perplexity_margin_min",
            ),
        ]:
            if floor is None:
                continue
            if value is None:
                failures.append(f"{route}: aggregate {label} is unavailable")
            elif value < floor:
                failures.append(
                    f"{route}: aggregate {label} {value:.9f} below floor {floor:.9f}"
                )

    extra_current = sorted(set(current) - set(baseline))
    for route in extra_current:
        print(f"aggregate_compare route={route} baseline_missing=true")
    if failures:
        raise RuntimeError(
            "Z-space aggregate regression gate failed: " + "; ".join(failures)
        )
    return len(common)


def aggregate_reports(reports, route_specs):
    aggregates = []
    for spec in route_specs:
        route = route_label(spec)
        selected = [report for report in reports if report["route"] == route]
        if not selected:
            raise RuntimeError(f"Z-space compare missing reports for route {route}")
        denom = len(selected)
        accepted_cases = sum(
            1
            for report in selected
            if summary_bool_value(report["ft_summary"], "accepted", True)
        )
        movement_ok_cases = sum(
            1
            for report in selected
            if summary_bool_value(report["ft_summary"], "movement_ok", True)
        )

        def mean(delta_name, metric_name):
            return sum(metric(report[delta_name], metric_name) for report in selected) / denom

        def margin_values(name):
            values = []
            for report in selected:
                value = summary_guard_margins(report["ft_summary"])[name]
                if value is not None:
                    values.append(value)
            return values

        def margin_mean(name):
            values = margin_values(name)
            return None if not values else sum(values) / len(values)

        def margin_min(name):
            values = margin_values(name)
            return None if not values else min(values)

        def projection_values(name):
            values = []
            for report in selected:
                value = report["projection"].get(name)
                if value is not None:
                    values.append(value)
            return values

        def projection_mean(name):
            values = projection_values(name)
            return None if not values else sum(values) / len(values)

        projection_delta_ratio = projection_mean("projection_delta_input_l2_ratio")
        projection_output_ratio = projection_mean("projection_output_input_l2_ratio")
        projection_variance_ratio = projection_mean(
            "projection_output_input_col_variance_ratio"
        )
        projection_variance_collapse_risk = (
            projection_variance_ratio is not None
            and projection_variance_ratio < PROJECTION_VARIANCE_COLLAPSE_RATIO
        )
        projection_norm_expansion_risk = (
            projection_output_ratio is not None
            and projection_output_ratio > PROJECTION_NORM_EXPANSION_RATIO
        )
        projection_healthy = (
            spec["strength"] is not None
            and projection_delta_ratio is not None
            and projection_output_ratio is not None
            and projection_variance_ratio is not None
            and not projection_variance_collapse_risk
            and not projection_norm_expansion_risk
        )

        aggregates.append(
            {
                "route": route,
                "zspace_strength": spec["strength"],
                "zspace_curvature": spec["curvature"],
                "zspace_frequency": ZSPACE_FREQUENCY if spec["strength"] is not None else None,
                "cases": denom,
                "case_labels": ",".join(report["case"] for report in selected),
                "accepted_cases": accepted_cases,
                "rejected_cases": denom - accepted_cases,
                "accepted_rate": accepted_cases / denom,
                "accepted_all": accepted_cases == denom,
                "movement_ok_cases": movement_ok_cases,
                "movement_not_ok_cases": denom - movement_ok_cases,
                "movement_ok_rate": movement_ok_cases / denom,
                "movement_ok_all": movement_ok_cases == denom,
                "projection_probe_samples": sum(
                    report["projection"]["projection_probe_samples"]
                    for report in selected
                ),
                "projection_probe_rows": sum(
                    report["projection"]["projection_probe_rows"]
                    for report in selected
                ),
                "projection_delta_input_l2_ratio_mean": projection_delta_ratio,
                "projection_output_input_l2_ratio_mean": projection_output_ratio,
                "projection_output_input_col_variance_ratio_mean": projection_variance_ratio,
                "projection_variance_collapse_risk": projection_variance_collapse_risk,
                "projection_norm_expansion_risk": projection_norm_expansion_risk,
                "projection_healthy": projection_healthy,
                "source_loss_delta": mean("source", "loss_delta"),
                "source_accuracy_delta": mean("source", "accuracy_delta"),
                "source_perplexity_delta": mean("source", "perplexity_delta"),
                "ft_loss_delta": mean("ft", "loss_delta"),
                "ft_accuracy_delta": mean("ft", "accuracy_delta"),
                "ft_perplexity_delta": mean("ft", "perplexity_delta"),
                "retention_loss_delta": mean("retention", "loss_delta"),
                "retention_accuracy_delta": mean("retention", "accuracy_delta"),
                "retention_perplexity_delta": mean("retention", "perplexity_delta"),
                "target_loss_margin_mean": margin_mean("target_loss_margin"),
                "target_loss_margin_min": margin_min("target_loss_margin"),
                "retention_loss_margin_mean": margin_mean("retention_loss_margin"),
                "retention_loss_margin_min": margin_min("retention_loss_margin"),
                "retention_accuracy_margin_mean": margin_mean("retention_accuracy_margin"),
                "retention_accuracy_margin_min": margin_min("retention_accuracy_margin"),
                "retention_perplexity_margin_mean": margin_mean(
                    "retention_perplexity_margin"
                ),
                "retention_perplexity_margin_min": margin_min(
                    "retention_perplexity_margin"
                ),
            }
        )
    return aggregates


def aggregate_row(aggregate, baseline, case_specs, route_specs):
    route_count = len(route_specs)
    source_advantage = aggregate["source_loss_delta"] - baseline["source_loss_delta"]
    ft_advantage = aggregate["ft_loss_delta"] - baseline["ft_loss_delta"]
    retention_advantage = (
        aggregate["retention_loss_delta"] - baseline["retention_loss_delta"]
    )
    row = {
        "example": "byte_lm_zspace_compare",
        "row_type": "route_aggregate",
        "route": aggregate["route"],
        "zspace_strength": aggregate["zspace_strength"],
        "zspace_curvature": aggregate["zspace_curvature"],
        "zspace_frequency": aggregate["zspace_frequency"],
        "cases": aggregate["cases"],
        "case_labels": aggregate["case_labels"],
        "routes": route_count,
        "accepted_cases": aggregate["accepted_cases"],
        "rejected_cases": aggregate["rejected_cases"],
        "accepted_rate": aggregate["accepted_rate"],
        "accepted_all": aggregate["accepted_all"],
        "movement_ok_cases": aggregate["movement_ok_cases"],
        "movement_not_ok_cases": aggregate["movement_not_ok_cases"],
        "movement_ok_rate": aggregate["movement_ok_rate"],
        "movement_ok_all": aggregate["movement_ok_all"],
        "projection_probe_samples": aggregate["projection_probe_samples"],
        "projection_probe_rows": aggregate["projection_probe_rows"],
        "projection_delta_input_l2_ratio_mean": aggregate[
            "projection_delta_input_l2_ratio_mean"
        ],
        "projection_output_input_l2_ratio_mean": aggregate[
            "projection_output_input_l2_ratio_mean"
        ],
        "projection_output_input_col_variance_ratio_mean": aggregate[
            "projection_output_input_col_variance_ratio_mean"
        ],
        "projection_variance_collapse_risk": aggregate[
            "projection_variance_collapse_risk"
        ],
        "projection_norm_expansion_risk": aggregate["projection_norm_expansion_risk"],
        "projection_healthy": aggregate["projection_healthy"],
        "source_loss_delta_mean": aggregate["source_loss_delta"],
        "source_accuracy_delta_mean": aggregate["source_accuracy_delta"],
        "source_perplexity_delta_mean": aggregate["source_perplexity_delta"],
        "ft_loss_delta_mean": aggregate["ft_loss_delta"],
        "ft_accuracy_delta_mean": aggregate["ft_accuracy_delta"],
        "ft_perplexity_delta_mean": aggregate["ft_perplexity_delta"],
        "retention_loss_delta_mean": aggregate["retention_loss_delta"],
        "retention_accuracy_delta_mean": aggregate["retention_accuracy_delta"],
        "retention_perplexity_delta_mean": aggregate["retention_perplexity_delta"],
        "target_loss_margin_mean": aggregate["target_loss_margin_mean"],
        "target_loss_margin_min": aggregate["target_loss_margin_min"],
        "retention_loss_margin_mean": aggregate["retention_loss_margin_mean"],
        "retention_loss_margin_min": aggregate["retention_loss_margin_min"],
        "retention_accuracy_margin_mean": aggregate["retention_accuracy_margin_mean"],
        "retention_accuracy_margin_min": aggregate["retention_accuracy_margin_min"],
        "retention_perplexity_margin_mean": aggregate[
            "retention_perplexity_margin_mean"
        ],
        "retention_perplexity_margin_min": aggregate["retention_perplexity_margin_min"],
    }
    row.update(
        {
            "source_loss_delta_advantage": source_advantage,
            "ft_loss_delta_advantage": ft_advantage,
            "retention_loss_delta_advantage": retention_advantage,
            "loss_delta_advantage_sum": (
                source_advantage + ft_advantage + retention_advantage
            ),
            "source_accuracy_delta_advantage": (
                aggregate["source_accuracy_delta"] - baseline["source_accuracy_delta"]
            ),
            "ft_accuracy_delta_advantage": (
                aggregate["ft_accuracy_delta"] - baseline["ft_accuracy_delta"]
            ),
            "retention_accuracy_delta_advantage": (
                aggregate["retention_accuracy_delta"]
                - baseline["retention_accuracy_delta"]
            ),
            "source_perplexity_delta_advantage": (
                aggregate["source_perplexity_delta"]
                - baseline["source_perplexity_delta"]
            ),
            "ft_perplexity_delta_advantage": (
                aggregate["ft_perplexity_delta"] - baseline["ft_perplexity_delta"]
            ),
            "retention_perplexity_delta_advantage": (
                aggregate["retention_perplexity_delta"]
                - baseline["retention_perplexity_delta"]
            ),
        }
    )
    return row


def ranked_route_rows(rows):
    return sorted(
        rows,
        key=lambda row: (
            numeric_value(row, "loss_delta_advantage_sum"),
            numeric_value(row, "ft_loss_delta_advantage"),
            numeric_value(row, "source_loss_delta_advantage"),
            row["route"],
        ),
        reverse=True,
    )


def healthy_ranked_route_rows(rows):
    return ranked_route_rows(
        [
            row
            for row in rows
            if row["zspace_strength"] is not None
            and row.get("projection_healthy") is True
            and numeric_value(row, "loss_delta_advantage_sum") > 0.0
            and row.get("accepted_all") is True
            and row.get("movement_ok_all") is True
        ]
    )


def route_edge_diagnostics(rows):
    ranked_zspace = [
        row for row in ranked_route_rows(rows) if row["zspace_strength"] is not None
    ]
    if not ranked_zspace:
        return None
    best = ranked_zspace[0]
    strengths = [float(row["zspace_strength"]) for row in ranked_zspace]
    abs_curvatures = [abs(float(row["zspace_curvature"])) for row in ranked_zspace]
    max_strength = max(strengths)
    min_abs_curvature = min(abs_curvatures)
    max_abs_curvature = max(abs_curvatures)
    best_abs_curvature = abs(float(best["zspace_curvature"]))
    variance_ratio = best.get("projection_output_input_col_variance_ratio_mean")
    output_ratio = best.get("projection_output_input_l2_ratio_mean")
    return {
        "route": best["route"],
        "zspace_strength": best["zspace_strength"],
        "zspace_curvature": best["zspace_curvature"],
        "loss_delta_advantage_sum": best["loss_delta_advantage_sum"],
        "projection_delta_input_l2_ratio_mean": best.get(
            "projection_delta_input_l2_ratio_mean"
        ),
        "projection_output_input_l2_ratio_mean": best.get(
            "projection_output_input_l2_ratio_mean"
        ),
        "projection_output_input_col_variance_ratio_mean": best.get(
            "projection_output_input_col_variance_ratio_mean"
        ),
        "max_strength": max_strength,
        "min_abs_curvature": min_abs_curvature,
        "max_abs_curvature": max_abs_curvature,
        "strength_edge": abs(float(best["zspace_strength"]) - max_strength) <= 1e-9,
        "near_zero_curvature_edge": abs(best_abs_curvature - min_abs_curvature) <= 1e-9,
        "steep_curvature_edge": abs(best_abs_curvature - max_abs_curvature) <= 1e-9,
        "projection_variance_collapse_risk": (
            variance_ratio is not None
            and variance_ratio < PROJECTION_VARIANCE_COLLAPSE_RATIO
        ),
        "projection_norm_expansion_risk": (
            output_ratio is not None and output_ratio > PROJECTION_NORM_EXPANSION_RATIO
        ),
    }


def best_zspace_aggregate(aggregates, metric_name):
    candidates = [
        row
        for row in aggregates
        if row["zspace_strength"] is not None and row.get(metric_name) is not None
    ]
    if not candidates:
        raise RuntimeError(f"Z-space compare did not aggregate any {metric_name} routes")
    return max(candidates, key=lambda row: row[metric_name])


def require_advantage(label, route, advantage, *, allow_nonadvantage=False):
    passed = advantage > 0.0
    print(
        f"zspace_advantage_check label={label} "
        f"route={route} "
        f"advantage={advantage:.6f} "
        f"passed={passed} "
        f"required={not allow_nonadvantage}"
    )
    if not passed and not allow_nonadvantage:
        raise RuntimeError(
            f"best Z-space {label} route did not beat baseline: "
            f"route={route} advantage={advantage:.6f}"
        )
    return passed


def fmt_optional(value):
    return "none" if value is None else f"{value:.6f}"


def main():
    args = parse_args()
    case_specs = selected_cases(args.cases)
    route_specs = selected_routes(args.routes, args.route_preset)
    print(
        f"compare=python_byte_lm_zspace "
        f"context={CONTEXT} hidden={HIDDEN} cases={len(case_specs)} routes={len(route_specs)}"
    )
    reports = []
    rows = []
    for case in case_specs:
        source_samples = st.dataset.byte_lm_corpus_windows(case["source_docs"], CONTEXT)
        target_samples = st.dataset.byte_lm_corpus_windows(case["target_docs"], CONTEXT)
        source_rows = st.dataset.byte_lm_sample_stats(source_samples)["active_rows"]
        target_rows = st.dataset.byte_lm_sample_stats(target_samples)["active_rows"]
        print(
            f"corpus_case={case['label']} "
            f"source_docs={len(case['source_docs'])} target_docs={len(case['target_docs'])} "
            f"source_windows={len(source_samples)} target_windows={len(target_samples)} "
            f"source_rows={source_rows} target_rows={target_rows}"
        )
        for spec in route_specs:
            report = run_route(case, spec, source_samples, target_samples)
            reports.append(report)
            rows.append(
                summary_row(
                    report,
                    len(case["source_docs"]),
                    len(case["target_docs"]),
                    source_rows,
                    target_rows,
                )
            )
    for report in reports:
        print_report(report)

    if args.jsonl is not None:
        write_jsonl(args.jsonl, rows)
        print(f"summary_jsonl={args.jsonl} rows={len(rows)}")
    if args.compare_jsonl is not None:
        baseline_rows = load_jsonl(args.compare_jsonl)
        compared = compare_summary_rows(
            rows,
            baseline_rows,
            args.max_target_loss_regression,
            args.max_retention_loss_regression,
            args.min_target_loss_margin,
            args.min_retention_loss_margin,
            args.min_retention_accuracy_margin,
            args.min_retention_perplexity_margin,
            args.require_status_match,
            args.require_accepted_match,
            args.require_guard_match,
            args.require_movement_tolerance_match,
            args.require_resume_match,
            args.require_checkpoint_match,
            allow_missing_current=args.cases is not None or args.routes is not None,
        )
        print(f"summary_compare_rows={compared} baseline={args.compare_jsonl}")

    aggregates = aggregate_reports(reports, route_specs)
    baseline = next(row for row in aggregates if row["route"] == "baseline")
    for aggregate in aggregates:
        print(
            f"route_metric_summary route={aggregate['route']} "
            f"cases={aggregate['cases']} "
            f"case_labels={aggregate['case_labels']} "
            f"source_loss_delta_mean={aggregate['source_loss_delta']:.6f} "
            f"source_accuracy_delta_mean={aggregate['source_accuracy_delta']:.6f} "
            f"source_perplexity_delta_mean={aggregate['source_perplexity_delta']:.6f} "
            f"ft_loss_delta_mean={aggregate['ft_loss_delta']:.6f} "
            f"ft_accuracy_delta_mean={aggregate['ft_accuracy_delta']:.6f} "
            f"ft_perplexity_delta_mean={aggregate['ft_perplexity_delta']:.6f} "
            f"retention_loss_delta_mean={aggregate['retention_loss_delta']:.6f} "
            f"retention_accuracy_delta_mean={aggregate['retention_accuracy_delta']:.6f} "
            f"retention_perplexity_delta_mean={aggregate['retention_perplexity_delta']:.6f} "
            f"target_loss_margin_mean={fmt_optional(aggregate['target_loss_margin_mean'])} "
            f"target_loss_margin_min={fmt_optional(aggregate['target_loss_margin_min'])} "
            f"retention_loss_margin_mean={fmt_optional(aggregate['retention_loss_margin_mean'])} "
            f"retention_loss_margin_min={fmt_optional(aggregate['retention_loss_margin_min'])} "
            f"retention_accuracy_margin_mean={fmt_optional(aggregate['retention_accuracy_margin_mean'])} "
            f"retention_accuracy_margin_min={fmt_optional(aggregate['retention_accuracy_margin_min'])} "
            f"accepted_cases={aggregate['accepted_cases']} "
            f"accepted_rate={aggregate['accepted_rate']:.6f} "
            f"accepted_all={aggregate['accepted_all']} "
            f"movement_ok_cases={aggregate['movement_ok_cases']} "
            f"movement_ok_rate={aggregate['movement_ok_rate']:.6f} "
            f"movement_ok_all={aggregate['movement_ok_all']} "
            f"projection_delta_input_l2_ratio_mean={fmt_optional(aggregate['projection_delta_input_l2_ratio_mean'])} "
            f"projection_output_input_l2_ratio_mean={fmt_optional(aggregate['projection_output_input_l2_ratio_mean'])} "
            f"projection_output_input_col_variance_ratio_mean={fmt_optional(aggregate['projection_output_input_col_variance_ratio_mean'])} "
            f"projection_healthy={aggregate['projection_healthy']} "
            f"projection_variance_collapse_risk={aggregate['projection_variance_collapse_risk']} "
            f"projection_norm_expansion_risk={aggregate['projection_norm_expansion_risk']}"
        )
    if args.aggregate_jsonl is not None:
        aggregate_rows = [
            aggregate_row(aggregate, baseline, case_specs, route_specs)
            for aggregate in aggregates
        ]
        write_jsonl(args.aggregate_jsonl, aggregate_rows)
        print(f"aggregate_jsonl={args.aggregate_jsonl} rows={len(aggregate_rows)}")
    else:
        aggregate_rows = [
            aggregate_row(aggregate, baseline, case_specs, route_specs)
            for aggregate in aggregates
        ]
    for rank, row in enumerate(ranked_route_rows(aggregate_rows), 1):
        print(
            f"route_rank_summary rank={rank} "
            f"route={row['route']} "
            f"loss_delta_advantage_sum={row['loss_delta_advantage_sum']:.9f} "
            f"source_loss_delta_advantage={row['source_loss_delta_advantage']:.9f} "
            f"ft_loss_delta_advantage={row['ft_loss_delta_advantage']:.9f} "
            f"retention_loss_delta_advantage={row['retention_loss_delta_advantage']:.9f} "
            f"accepted_rate={row['accepted_rate']:.9f} "
            f"movement_ok_rate={row['movement_ok_rate']:.9f} "
            f"target_loss_margin_min={fmt_optional(row['target_loss_margin_min'])} "
            f"retention_loss_margin_min={fmt_optional(row['retention_loss_margin_min'])}"
        )
    healthy_rows = healthy_ranked_route_rows(aggregate_rows)
    if healthy_rows:
        for rank, row in enumerate(healthy_rows, 1):
            print(
                f"route_health_rank_summary rank={rank} "
                f"route={row['route']} "
                f"loss_delta_advantage_sum={row['loss_delta_advantage_sum']:.9f} "
                f"projection_healthy={row['projection_healthy']} "
                f"projection_delta_input_l2_ratio_mean={fmt_optional(row['projection_delta_input_l2_ratio_mean'])} "
                f"projection_output_input_l2_ratio_mean={fmt_optional(row['projection_output_input_l2_ratio_mean'])} "
                f"projection_output_input_col_variance_ratio_mean={fmt_optional(row['projection_output_input_col_variance_ratio_mean'])} "
                f"projection_variance_collapse_risk={row['projection_variance_collapse_risk']} "
                f"projection_norm_expansion_risk={row['projection_norm_expansion_risk']} "
                f"accepted_rate={row['accepted_rate']:.9f} "
                f"movement_ok_rate={row['movement_ok_rate']:.9f}"
            )
    else:
        print("route_health_rank_summary rank=none healthy_routes=0")
    edge = route_edge_diagnostics(aggregate_rows)
    if edge is not None:
        print(
            f"route_edge_check route={edge['route']} "
            f"zspace_strength={edge['zspace_strength']:.6f} "
            f"zspace_curvature={edge['zspace_curvature']:.6f} "
            f"loss_delta_advantage_sum={edge['loss_delta_advantage_sum']:.9f} "
            f"projection_delta_input_l2_ratio_mean={fmt_optional(edge['projection_delta_input_l2_ratio_mean'])} "
            f"projection_output_input_l2_ratio_mean={fmt_optional(edge['projection_output_input_l2_ratio_mean'])} "
            f"projection_output_input_col_variance_ratio_mean={fmt_optional(edge['projection_output_input_col_variance_ratio_mean'])} "
            f"projection_variance_collapse_risk={edge['projection_variance_collapse_risk']} "
            f"projection_norm_expansion_risk={edge['projection_norm_expansion_risk']} "
            f"max_strength={edge['max_strength']:.6f} "
            f"min_abs_curvature={edge['min_abs_curvature']:.6f} "
            f"max_abs_curvature={edge['max_abs_curvature']:.6f} "
            f"strength_edge={edge['strength_edge']} "
            f"near_zero_curvature_edge={edge['near_zero_curvature_edge']} "
            f"steep_curvature_edge={edge['steep_curvature_edge']}"
        )
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
    if args.compare_aggregate_jsonl is not None:
        baseline_aggregate_rows = load_aggregate_jsonl(args.compare_aggregate_jsonl)
        compared = compare_aggregate_rows(
            aggregate_rows,
            baseline_aggregate_rows,
            args.max_aggregate_source_loss_regression,
            args.max_aggregate_ft_loss_regression,
            args.max_aggregate_retention_loss_regression,
            args.min_aggregate_target_loss_margin,
            args.min_aggregate_retention_loss_margin,
            args.min_aggregate_retention_accuracy_margin,
            args.min_aggregate_retention_perplexity_margin,
            args.require_aggregate_winner_match,
            allow_missing_current=args.cases is not None or args.routes is not None,
            max_aggregate_accepted_rate_regression=args.max_aggregate_accepted_rate_regression,
            max_aggregate_movement_ok_rate_regression=args.max_aggregate_movement_ok_rate_regression,
        )
        print(
            f"aggregate_compare_rows={compared} "
            f"baseline={args.compare_aggregate_jsonl}"
        )

    best_source = best_zspace_aggregate(aggregates, "source_loss_delta")
    best_ft = best_zspace_aggregate(aggregates, "ft_loss_delta")
    best_retention = best_zspace_aggregate(aggregates, "retention_loss_delta")
    best_target_margin = best_zspace_aggregate(aggregates, "target_loss_margin_min")
    best_retention_loss_margin = best_zspace_aggregate(
        aggregates, "retention_loss_margin_min"
    )
    best_retention_accuracy_margin = best_zspace_aggregate(
        aggregates, "retention_accuracy_margin_min"
    )
    source_advantage = (
        best_source["source_loss_delta"]
        - baseline["source_loss_delta"]
    )
    ft_advantage = best_ft["ft_loss_delta"] - baseline["ft_loss_delta"]
    retention_advantage = (
        best_retention["retention_loss_delta"]
        - baseline["retention_loss_delta"]
    )
    require_advantage(
        "source",
        best_source["route"],
        source_advantage,
        allow_nonadvantage=args.allow_zspace_nonadvantage,
    )
    require_advantage(
        "fine-tune",
        best_ft["route"],
        ft_advantage,
        allow_nonadvantage=args.allow_zspace_nonadvantage,
    )
    print(
        "zspace_winners "
        f"source_route={best_source['route']} "
        f"ft_route={best_ft['route']} "
        f"retention_route={best_retention['route']}"
    )
    print(
        "zspace_margin_winners "
        f"target_loss_margin_route={best_target_margin['route']} "
        f"retention_loss_margin_route={best_retention_loss_margin['route']} "
        f"retention_accuracy_margin_route={best_retention_accuracy_margin['route']}"
    )
    print(
        "zspace_advantage "
        f"source_loss_delta={source_advantage:.6f} "
        f"ft_loss_delta={ft_advantage:.6f} "
        f"retention_loss_delta={retention_advantage:.6f}"
    )


if __name__ == "__main__":
    main()
