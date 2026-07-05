import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import spiraltorch as st
from spiraltorch.nn import (
    Linear,
    LoraLinear,
    Relu,
    Sequential,
    SoftmaxCrossEntropy,
    sparse_classification_delta,
)

from checkpoint_preflight import (
    AUTO_KEY_PRESET,
    HF_KEY_PRESET_CHOICES,
    add_checkpoint_projection_args,
    add_checkpoint_source_gain_args,
    apply_checkpoint_projection,
    apply_checkpoint_source_gain,
    checkpoint_audit_fields,
    checkpoint_projection_fields,
    checkpoint_source_gain_fields,
    checkpoint_source_gain_value,
    hf_lm_handoff_from_external_state,
    hf_lm_handoff_from_spiraltorch_state,
    hf_lm_shape_include_keys,
    hf_lm_state_keys,
    hf_lm_overlap_resize_kwargs,
    hf_lm_tensor_bounds_for_module_shapes,
    infer_hf_lm_module_shapes,
    load_hf_state_dict,
    load_hf_state_dict_shapes,
    preflight_and_load,
    projection_value_label,
    resolve_hf_lm_key_preset,
    resolved_checkpoint_projection_value,
)
from sparse_finetune_compare import (
    add_summary_compare_args,
    attach_epoch_tensor_backend_aggregate_fields,
    attach_epoch_tensor_backend_fields,
    attach_requested_wgpu_component_backend_summary,
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
CONTEXT = 4
BATCH_WINDOWS = 4
ACCUMULATION_STEPS = 2
SOURCE_EPOCHS = 4
FT_EPOCHS = 10
PARAM_SCALE = 0.003
FT_MOVEMENT_TOLERANCE = 1e-6
DEFAULT_ADAPTER_WEIGHT_DECAY = 0.0
DEFAULT_MAX_GRAD_NORM = 2.0
DEFAULT_GRADIENT_ACCUMULATION_STEPS = ACCUMULATION_STEPS
DEFAULT_TARGET_MIN_LOSS_DELTA = 0.0
DEFAULT_MIN_DELTA = 0.0
DEFAULT_LR_DECAY_FACTOR = 0.5
DEFAULT_LR_DECAY_MIN_DELTA = 0.0
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

SWEEP_CONFIGS = [
    {"label": "r6_a32_lr3", "rank": 6, "alpha": 32.0, "adapter_lr_scale": 3.0},
    {"label": "r12_a64_lr4", "rank": 12, "alpha": 64.0, "adapter_lr_scale": 4.0},
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep tiny byte-MLP LoRA head rank/alpha/LR settings."
    )
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        help=(
            "Run only this target corpus case. May be repeated. Labels can come "
            "from built-in cases or --case-jsonl rows."
        ),
    )
    parser.add_argument(
        "--case-jsonl",
        dest="case_jsonls",
        action="append",
        type=Path,
        default=[],
        help=(
            "Additional byte-LM case JSONL. Each row needs label plus "
            "source_text/source_docs and target_text/target_docs."
        ),
    )
    parser.add_argument(
        "--config",
        dest="configs",
        action="append",
        help=(
            "Run only this LoRA sweep config. May be repeated. Labels can "
            "come from built-in configs or --lora-config-jsonl rows."
        ),
    )
    parser.add_argument(
        "--lora-config-jsonl",
        dest="lora_config_jsonls",
        action="append",
        type=Path,
        default=[],
        help=(
            "Additional LoRA config JSONL. Each byte_lm_lora_config row needs "
            "label, rank, alpha, and adapter_lr_scale."
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
        help="Optional path for case-averaged LoRA config aggregate rows.",
    )
    parser.add_argument(
        "--compare-jsonl",
        type=Path,
        default=None,
        help="Optional previous summary JSONL to compare against this run.",
    )
    parser.add_argument(
        "--compare-aggregate-jsonl",
        type=Path,
        default=None,
        help="Optional previous aggregate JSONL to compare against this run.",
    )
    parser.add_argument(
        "--key-preset",
        choices=HF_KEY_PRESET_CHOICES,
        default="gpt2",
        help=(
            "HF-style checkpoint key preset for the embed/head handoff. Use "
            "'auto' with --hf-state-dict to infer from checkpoint shape metadata."
        ),
    )
    parser.add_argument(
        "--hf-state-dict",
        type=Path,
        default=None,
        help=(
            "Optional local HF/PyTorch state dict file or directory to sweep as "
            "the frozen embed/head source instead of training a dense source."
        ),
    )
    parser.add_argument(
        "--checkpoint-source-label",
        default=None,
        help=(
            "Optional short label for checkpoint source rows. Defaults to the "
            "resolved key preset for --hf-state-dict."
        ),
    )
    parser.add_argument(
        "--include-extra-key",
        dest="include_extra_keys",
        action="append",
        default=[],
        help="Additional external state-dict key to include in checkpoint audit rows.",
    )
    parser.add_argument(
        "--no-synthesize-missing-biases",
        action="store_true",
        help="Require embed/head bias tensors to exist in --hf-state-dict.",
    )
    parser.add_argument(
        "--allow-overlap-resize",
        action="store_true",
        help=(
            "Explicitly adapt HF embed/head tensors to the byte-smoke shape "
            "with overlap-copy and zero-fill transforms."
        ),
    )
    add_checkpoint_projection_args(parser)
    add_checkpoint_source_gain_args(parser)
    parser.add_argument(
        "--checkpoint-projection-strengths",
        default=None,
        help=(
            "Comma-separated Z-space projection strengths for a checkpoint "
            "projection grid. Defaults to --checkpoint-projection-strength."
        ),
    )
    parser.add_argument(
        "--checkpoint-projection-curvatures",
        default=None,
        help=(
            "Comma-separated Z-space curvatures for a checkpoint projection "
            "grid. Defaults to --checkpoint-projection-curvature."
        ),
    )
    parser.add_argument(
        "--checkpoint-projection-frequencies",
        default=None,
        help=(
            "Comma-separated language-wave frequencies for a checkpoint "
            "projection grid. Defaults to --checkpoint-projection-frequency."
        ),
    )
    parser.add_argument(
        "--checkpoint-source-gains",
        default=None,
        help=(
            "Comma-separated positive checkpoint source gains for a source "
            "amplitude grid. Defaults to --checkpoint-source-gain."
        ),
    )
    parser.add_argument(
        "--adapter-weight-decays",
        default=None,
        help=(
            "Comma-separated non-negative decoupled weight decay values for "
            "trainable LoRA adapter parameters. Defaults to 0.0."
        ),
    )
    parser.add_argument(
        "--max-grad-norms",
        default=None,
        help=(
            "Comma-separated positive global gradient clipping thresholds for "
            "LoRA FT policy lanes. Defaults to 2.0."
        ),
    )
    parser.add_argument(
        "--gradient-accumulation-steps-list",
        default=None,
        help=(
            "Comma-separated positive accumulation-step counts for LoRA FT "
            "policy lanes. Defaults to 2."
        ),
    )
    parser.add_argument(
        "--ft-epochs-list",
        default=None,
        help="Comma-separated positive sparse FT epoch counts. Defaults to 10.",
    )
    parser.add_argument(
        "--ft-control-jsonl",
        dest="ft_control_jsonls",
        action="append",
        type=Path,
        default=[],
        help=(
            "Additional sparse FT-control lanes as JSONL rows. Each row needs "
            "ft_epochs and may set label, target_min_loss_delta_policy, "
            "early_stopping_patience, early_stopping_min_delta, "
            "lr_decay_patience, lr_decay_factor, and lr_decay_min_delta. "
            "Using this disables the FT-control grid options."
        ),
    )
    parser.add_argument(
        "--target-min-loss-deltas",
        default=None,
        help=(
            "Comma-separated non-negative target validation loss improvements "
            "required for FT acceptance. Defaults to 0.0."
        ),
    )
    parser.add_argument(
        "--patiences",
        default=None,
        help=(
            "Comma-separated positive early-stopping patience values, or 'none'. "
            "Defaults to none."
        ),
    )
    parser.add_argument(
        "--min-deltas",
        default=None,
        help=(
            "Comma-separated non-negative validation-loss drops required to reset "
            "early-stopping patience. Defaults to 0.0."
        ),
    )
    parser.add_argument(
        "--lr-decay-patiences",
        default=None,
        help=(
            "Comma-separated positive plateau LR-decay patience values, or 'none'. "
            "Defaults to none."
        ),
    )
    parser.add_argument(
        "--lr-decay-factors",
        default=None,
        help="Comma-separated LR plateau factors in (0, 1). Defaults to 0.5.",
    )
    parser.add_argument(
        "--lr-decay-min-deltas",
        default=None,
        help=(
            "Comma-separated non-negative validation-loss drops required to reset "
            "LR plateau patience. Defaults to 0.0."
        ),
    )
    add_summary_compare_args(parser, subject="LoRA config")
    parser.add_argument(
        "--require-winner-match",
        action="store_true",
        help="Fail when the winning LoRA sweep config changes.",
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
        "--min-aggregate-retention-accuracy-margin",
        type=float,
        default=None,
        help=(
            "Fail when aggregate retention_accuracy_margin_min is below this "
            "non-negative floor."
        ),
    )
    parser.add_argument(
        "--min-aggregate-retention-perplexity-margin",
        type=float,
        default=None,
        help=(
            "Fail when aggregate retention_perplexity_margin_min is absent or "
            "below this non-negative floor."
        ),
    )
    parser.add_argument(
        "--require-aggregate-winner-match",
        action="store_true",
        help="Fail when the case-averaged winning LoRA config changes.",
    )
    parser.add_argument(
        "--require-aggregate-accepted-all",
        action="store_true",
        help="Fail when any current aggregate config has a non-accepted case.",
    )
    parser.add_argument(
        "--min-aggregate-cases",
        type=int,
        default=None,
        help="Fail when a current aggregate config includes fewer than this many cases.",
    )
    parser.add_argument(
        "--require-aggregate-case",
        dest="require_aggregate_cases",
        action="append",
        default=[],
        help=(
            "Fail when a current aggregate config is missing this case label. "
            "May be repeated; labels can come from --case-jsonl."
        ),
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
    args = parser.parse_args()
    if args.checkpoint_projection_preset is not None:
        args.checkpoint_projection = "zspace"
    projection_grid_requested = uses_checkpoint_projection_grid(args)
    if args.checkpoint_projection != "zspace" and projection_grid_requested:
        parser.error("checkpoint projection grid options require --checkpoint-projection zspace")
    for attr in [
        "checkpoint_projection_strengths",
        "checkpoint_projection_curvatures",
        "checkpoint_projection_frequencies",
        "checkpoint_source_gains",
        "adapter_weight_decays",
        "max_grad_norms",
        "target_min_loss_deltas",
        "min_deltas",
        "lr_decay_factors",
        "lr_decay_min_deltas",
    ]:
        values = getattr(args, attr, None)
        if values is not None and not parse_float_list(values):
            parser.error(f"--{attr.replace('_', '-')} must include at least one float")
        if values is not None:
            parsed = parse_float_list(values)
            if attr == "adapter_weight_decays":
                if any(value < 0.0 for value in parsed):
                    parser.error(f"--{attr.replace('_', '-')} values must be non-negative")
            elif attr == "max_grad_norms":
                if any(value <= 0.0 for value in parsed):
                    parser.error(f"--{attr.replace('_', '-')} values must be positive")
            elif attr == "lr_decay_factors":
                if any(value <= 0.0 or value >= 1.0 for value in parsed):
                    parser.error(f"--{attr.replace('_', '-')} values must be in (0, 1)")
            elif attr in {"target_min_loss_deltas", "min_deltas", "lr_decay_min_deltas"}:
                if any(value < 0.0 for value in parsed):
                    parser.error(f"--{attr.replace('_', '-')} values must be non-negative")
            elif attr == "checkpoint_projection_curvatures":
                if any(value >= 0.0 for value in parsed):
                    parser.error(f"--{attr.replace('_', '-')} values must be negative")
            elif any(value <= 0.0 for value in parsed):
                parser.error(f"--{attr.replace('_', '-')} values must be positive")
    if args.gradient_accumulation_steps_list is not None:
        accumulation_steps = parse_int_list(args.gradient_accumulation_steps_list)
        if not accumulation_steps:
            parser.error("--gradient-accumulation-steps-list must include at least one integer")
        if any(value <= 0 for value in accumulation_steps):
            parser.error("--gradient-accumulation-steps-list values must be positive")
    if args.ft_epochs_list is not None:
        ft_epochs = parse_int_list(args.ft_epochs_list)
        if not ft_epochs:
            parser.error("--ft-epochs-list must include at least one integer")
        if any(value <= 0 for value in ft_epochs):
            parser.error("--ft-epochs-list values must be positive")
    if args.ft_control_jsonls:
        grid_attrs = [
            "ft_epochs_list",
            "target_min_loss_deltas",
            "patiences",
            "min_deltas",
            "lr_decay_patiences",
            "lr_decay_factors",
            "lr_decay_min_deltas",
        ]
        conflicts = [attr for attr in grid_attrs if getattr(args, attr, None) is not None]
        if conflicts:
            parser.error(
                "--ft-control-jsonl cannot be combined with "
                + ", ".join(f"--{attr.replace('_', '-')}" for attr in conflicts)
            )
    for attr in ["patiences", "lr_decay_patiences"]:
        values = getattr(args, attr, None)
        if values is not None:
            parsed = parse_optional_int_list(values)
            if not parsed:
                parser.error(f"--{attr.replace('_', '-')} must include at least one value")
            if any(value is not None and value <= 0 for value in parsed):
                parser.error(f"--{attr.replace('_', '-')} values must be positive or none")
    gate_requested = validate_summary_compare_args(parser, args) or args.require_winner_match
    if args.compare_jsonl is None and gate_requested:
        parser.error("regression gate options require --compare-jsonl")
    if args.key_preset == AUTO_KEY_PRESET and args.hf_state_dict is None:
        parser.error("--key-preset auto requires --hf-state-dict")
    if args.checkpoint_source_gain <= 0.0:
        parser.error("--checkpoint-source-gain must be positive")
    aggregate_gate_requested = (
        args.max_aggregate_target_loss_regression is not None
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
    for name in [
        "max_aggregate_target_loss_regression",
        "max_aggregate_retention_loss_regression",
        "max_aggregate_accepted_rate_regression",
        "max_aggregate_movement_ok_rate_regression",
        "min_aggregate_target_loss_margin",
        "min_aggregate_retention_loss_margin",
        "min_aggregate_retention_accuracy_margin",
        "min_aggregate_retention_perplexity_margin",
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
    return args


def read_case_text(row, field, *, path, line_no):
    value = row.get(field)
    if isinstance(value, str) and value:
        return value
    docs_field = f"{field.rsplit('_', 1)[0]}_docs"
    docs = row.get(docs_field)
    if isinstance(docs, list) and docs and all(isinstance(doc, str) and doc for doc in docs):
        return "\n".join(docs)
    raise ValueError(
        f"{path}:{line_no}: case row missing non-empty {field} or {docs_field}"
    )


def load_case_jsonl(path):
    cases = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: case row must be a JSON object")
            row_type = row.get("row_type")
            if row_type is not None and row_type != "byte_lm_case":
                continue
            label = row.get("label")
            if not isinstance(label, str) or not label:
                raise ValueError(f"{path}:{line_no}: case row missing non-empty label")
            if "," in label:
                raise ValueError(f"{path}:{line_no}: case label must not contain ','")
            cases.append(
                {
                    "label": label,
                    "source_text": read_case_text(
                        row,
                        "source_text",
                        path=path,
                        line_no=line_no,
                    ),
                    "target_text": read_case_text(
                        row,
                        "target_text",
                        path=path,
                        line_no=line_no,
                    ),
                }
            )
    if not cases:
        raise ValueError(f"{path}: no byte_lm_case rows found")
    return cases


def case_specs_with_external(case_jsonls):
    cases = list(CASE_SPECS)
    for path in case_jsonls or []:
        cases.extend(load_case_jsonl(path))
    seen = set()
    duplicates = []
    for case in cases:
        label = case["label"]
        if label in seen:
            duplicates.append(label)
        seen.add(label)
    if duplicates:
        raise ValueError(
            "duplicate byte-LM case label(s): " + ",".join(sorted(set(duplicates)))
        )
    return cases


def selected_cases(labels, case_specs=None):
    case_specs = list(case_specs or CASE_SPECS)
    if labels is None:
        return [case_specs[0]]
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate --case label(s): " + ",".join(labels))
    by_label = {case["label"]: case for case in case_specs}
    missing = [label for label in labels if label not in by_label]
    if missing:
        raise ValueError(
            "unknown byte-LM case label(s): "
            + ",".join(missing)
            + "; available="
            + ",".join(by_label)
        )
    return [by_label[label] for label in labels]


def positive_int_field(row, key, *, path, line_no):
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}:{line_no}: {key} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{path}:{line_no}: {key} must be positive")
    return value


def lora_config_from_row(row, *, path, line_no):
    label = row.get("label")
    if not isinstance(label, str) or not label:
        raise ValueError(f"{path}:{line_no}: LoRA config row missing non-empty label")
    if "," in label:
        raise ValueError(f"{path}:{line_no}: LoRA config label must not contain ','")
    if "::" in label:
        raise ValueError(f"{path}:{line_no}: LoRA config label must not contain '::'")
    return {
        "label": label,
        "rank": positive_int_field(row, "rank", path=path, line_no=line_no),
        "alpha": numeric_field(row, "alpha", path=path, line_no=line_no, positive=True),
        "adapter_lr_scale": numeric_field(
            row,
            "adapter_lr_scale",
            path=path,
            line_no=line_no,
            positive=True,
        ),
    }


def load_lora_config_jsonl(path):
    configs = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"{path}:{line_no}: LoRA config row must be a JSON object"
                )
            row_type = row.get("row_type")
            if row_type is not None and row_type != "byte_lm_lora_config":
                continue
            configs.append(lora_config_from_row(row, path=path, line_no=line_no))
    if not configs:
        raise ValueError(f"{path}: no byte_lm_lora_config rows found")
    return configs


def lora_configs_with_external(lora_config_jsonls):
    configs = [dict(config) for config in SWEEP_CONFIGS]
    for path in lora_config_jsonls or []:
        configs.extend(load_lora_config_jsonl(path))
    seen = set()
    duplicates = []
    for config in configs:
        label = config["label"]
        if label in seen:
            duplicates.append(label)
        seen.add(label)
    if duplicates:
        raise ValueError(
            "duplicate LoRA config label(s): " + ",".join(sorted(set(duplicates)))
        )
    return configs


def selected_configs(labels, config_specs=None):
    config_specs = list(config_specs or SWEEP_CONFIGS)
    if labels is None:
        return config_specs
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate --config label(s): " + ",".join(labels))
    by_label = {config["label"]: config for config in config_specs}
    missing = [label for label in labels if label not in by_label]
    if missing:
        raise ValueError(
            "unknown LoRA config label(s): "
            + ",".join(missing)
            + "; available="
            + ",".join(by_label)
        )
    return [by_label[label] for label in labels]


def parse_float_list(raw):
    if raw is None:
        return []
    values = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    return values


def parse_int_list(raw):
    if raw is None:
        return []
    values = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    return values


def parse_optional_int_list(raw):
    if raw is None:
        return []
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() in {"none", "null", "off"}:
            values.append(None)
        else:
            values.append(int(token))
    return values


def uses_checkpoint_projection_grid(args):
    return any(
        getattr(args, attr, None) is not None
        for attr in [
            "checkpoint_projection_strengths",
            "checkpoint_projection_curvatures",
            "checkpoint_projection_frequencies",
        ]
    ) or getattr(args, "checkpoint_projection_preset", None) is not None


def uses_checkpoint_source_gain_grid(args):
    return getattr(args, "checkpoint_source_gains", None) is not None


def uses_adapter_weight_decay_grid(args):
    return getattr(args, "adapter_weight_decays", None) is not None


def uses_training_policy_grid(args):
    return (
        getattr(args, "max_grad_norms", None) is not None
        or getattr(args, "gradient_accumulation_steps_list", None) is not None
    )


def uses_ft_control_grid(args):
    return any(
        getattr(args, attr, None) is not None
        for attr in [
            "ft_epochs_list",
            "target_min_loss_deltas",
            "patiences",
            "min_deltas",
            "lr_decay_patiences",
            "lr_decay_factors",
            "lr_decay_min_deltas",
        ]
    )


def label_number(value):
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def checkpoint_projection_variant_label(strength, curvature, frequency):
    return (
        f"zspace_s{label_number(strength)}"
        f"_c{label_number(curvature)}"
        f"_f{label_number(frequency)}"
    )


def checkpoint_source_gain_variant_label(gain):
    return f"gain_g{label_number(gain)}"


def adapter_weight_decay_variant_label(weight_decay):
    return f"wd{label_number(weight_decay)}"


def max_grad_norm_variant_label(max_grad_norm):
    return f"gn{label_number(max_grad_norm)}"


def gradient_accumulation_variant_label(steps):
    return f"accum{int(steps)}"


def optional_int_label(prefix, value):
    return f"{prefix}none" if value is None else f"{prefix}{int(value)}"


def ft_control_variant_label(fields):
    labels = [f"ep{int(fields['ft_epochs'])}"]
    labels.append(f"tmin{label_number(fields['target_min_loss_delta_policy'])}")
    labels.append(optional_int_label("pat", fields["early_stopping_patience"]))
    labels.append(f"md{label_number(fields['early_stopping_min_delta'])}")
    labels.append(optional_int_label("ldp", fields["lr_decay_patience"]))
    labels.append(f"ldf{label_number(fields['lr_decay_factor'])}")
    labels.append(f"ldmd{label_number(fields['lr_decay_min_delta'])}")
    return "::".join(labels)


def optional_int_field(row, key, *, path, line_no):
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}:{line_no}: {key} must be an integer or null")
    if value <= 0:
        raise ValueError(f"{path}:{line_no}: {key} must be positive or null")
    return value


def numeric_field(
    row, key, default=None, *, path, line_no, positive=False, non_negative=False
):
    value = row.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}:{line_no}: {key} must be numeric")
    value = float(value)
    if positive and value <= 0.0:
        raise ValueError(f"{path}:{line_no}: {key} must be positive")
    if non_negative and value < 0.0:
        raise ValueError(f"{path}:{line_no}: {key} must be non-negative")
    return value


def ft_control_fields_from_row(row, *, path, line_no):
    ft_epochs = row.get("ft_epochs")
    if isinstance(ft_epochs, bool) or not isinstance(ft_epochs, int):
        raise ValueError(f"{path}:{line_no}: ft_epochs must be a positive integer")
    if ft_epochs <= 0:
        raise ValueError(f"{path}:{line_no}: ft_epochs must be positive")
    lr_decay_factor = numeric_field(
        row,
        "lr_decay_factor",
        DEFAULT_LR_DECAY_FACTOR,
        path=path,
        line_no=line_no,
        positive=True,
    )
    if lr_decay_factor >= 1.0:
        raise ValueError(f"{path}:{line_no}: lr_decay_factor must be in (0, 1)")
    return {
        "ft_epochs": ft_epochs,
        "target_min_loss_delta_policy": numeric_field(
            row,
            "target_min_loss_delta_policy",
            row.get("target_min_loss_delta", DEFAULT_TARGET_MIN_LOSS_DELTA),
            path=path,
            line_no=line_no,
            non_negative=True,
        ),
        "early_stopping_patience": optional_int_field(
            row,
            "early_stopping_patience",
            path=path,
            line_no=line_no,
        ),
        "early_stopping_min_delta": numeric_field(
            row,
            "early_stopping_min_delta",
            row.get("min_delta", DEFAULT_MIN_DELTA),
            path=path,
            line_no=line_no,
            non_negative=True,
        ),
        "lr_decay_patience": optional_int_field(
            row,
            "lr_decay_patience",
            path=path,
            line_no=line_no,
        ),
        "lr_decay_factor": lr_decay_factor,
        "lr_decay_min_delta": numeric_field(
            row,
            "lr_decay_min_delta",
            DEFAULT_LR_DECAY_MIN_DELTA,
            path=path,
            line_no=line_no,
            non_negative=True,
        ),
    }


def load_ft_control_jsonl(path):
    variants = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: FT-control row must be a JSON object")
            row_type = row.get("row_type")
            if row_type is not None and row_type != "byte_lm_ft_control":
                continue
            fields = ft_control_fields_from_row(row, path=path, line_no=line_no)
            label = row.get("label")
            if label is not None and not isinstance(label, str):
                raise ValueError(f"{path}:{line_no}: label must be a string")
            variants.append(
                {
                    "label": label or ft_control_variant_label(fields),
                    "fields": fields,
                }
            )
    if not variants:
        raise ValueError(f"{path}: no byte_lm_ft_control rows found")
    return variants


def ft_control_variants_from_jsonls(paths):
    variants = []
    for path in paths or []:
        variants.extend(load_ft_control_jsonl(path))
    seen = set()
    duplicates = []
    for variant in variants:
        label = variant["label"]
        if label in seen:
            duplicates.append(label)
        seen.add(label)
    if duplicates:
        raise ValueError(
            "duplicate FT-control lane label(s): " + ",".join(sorted(set(duplicates)))
        )
    return variants


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


def attach_training_policy_key(row):
    row["training_policy_key"] = training_policy_key(row)
    return row


def checkpoint_projection_variants(args):
    if checkpoint_projection_fields(args)["checkpoint_projection"] != "zspace":
        return [
            {
                "label": None,
                "args": args,
                "fields": checkpoint_projection_fields(args),
            }
        ]
    strengths = (
        parse_float_list(getattr(args, "checkpoint_projection_strengths", None))
        or [resolved_checkpoint_projection_value(args, "checkpoint_projection_strength")]
    )
    curvatures = (
        parse_float_list(getattr(args, "checkpoint_projection_curvatures", None))
        or [resolved_checkpoint_projection_value(args, "checkpoint_projection_curvature")]
    )
    frequencies = (
        parse_float_list(getattr(args, "checkpoint_projection_frequencies", None))
        or [resolved_checkpoint_projection_value(args, "checkpoint_projection_frequency")]
    )
    variants = []
    label_variants = uses_checkpoint_projection_grid(args) or (
        len(strengths) * len(curvatures) * len(frequencies) > 1
    )
    for strength in strengths:
        for curvature in curvatures:
            for frequency in frequencies:
                variant_args = SimpleNamespace(
                    checkpoint_projection="zspace",
                    checkpoint_projection_strength=strength,
                    checkpoint_projection_curvature=curvature,
                    checkpoint_projection_frequency=frequency,
                )
                variants.append(
                    {
                        "label": (
                            checkpoint_projection_variant_label(
                                strength,
                                curvature,
                                frequency,
                            )
                            if label_variants
                            else None
                        ),
                        "args": variant_args,
                        "fields": checkpoint_projection_fields(variant_args),
                    }
                )
    return variants


def checkpoint_source_gain_variants(args):
    gains = parse_float_list(getattr(args, "checkpoint_source_gains", None)) or [
        checkpoint_source_gain_value(args)
    ]
    label_variants = uses_checkpoint_source_gain_grid(args) or len(gains) > 1
    variants = []
    for gain in gains:
        variant_args = SimpleNamespace(checkpoint_source_gain=gain)
        variants.append(
            {
                "label": checkpoint_source_gain_variant_label(gain)
                if label_variants
                else None,
                "args": variant_args,
                "fields": checkpoint_source_gain_fields(variant_args),
            }
        )
    return variants


def adapter_weight_decay_variants(args):
    values = parse_float_list(getattr(args, "adapter_weight_decays", None)) or [
        DEFAULT_ADAPTER_WEIGHT_DECAY
    ]
    label_variants = uses_adapter_weight_decay_grid(args) or len(values) > 1
    variants = []
    for weight_decay in values:
        variants.append(
            {
                "label": adapter_weight_decay_variant_label(weight_decay)
                if label_variants
                else None,
                "fields": {"adapter_weight_decay": weight_decay},
            }
        )
    return variants


def training_policy_variants(args):
    max_grad_norms = parse_float_list(getattr(args, "max_grad_norms", None)) or [
        DEFAULT_MAX_GRAD_NORM
    ]
    accumulation_steps = parse_int_list(
        getattr(args, "gradient_accumulation_steps_list", None)
    ) or [DEFAULT_GRADIENT_ACCUMULATION_STEPS]
    label_variants = (
        uses_training_policy_grid(args)
        or len(max_grad_norms) * len(accumulation_steps) > 1
    )
    variants = []
    for max_grad_norm in max_grad_norms:
        for steps in accumulation_steps:
            max_grad_norm_label = (
                max_grad_norm_variant_label(max_grad_norm) if label_variants else None
            )
            accumulation_label = (
                gradient_accumulation_variant_label(steps) if label_variants else None
            )
            labels = [
                label
                for label in [max_grad_norm_label, accumulation_label]
                if label is not None
            ]
            variants.append(
                {
                    "label": "::".join(labels) if labels else None,
                    "fields": {
                        "max_grad_norm": max_grad_norm,
                        "gradient_accumulation_steps": steps,
                    },
                    "max_grad_norm_label": max_grad_norm_label,
                    "gradient_accumulation_steps_label": accumulation_label,
                }
            )
    return variants


def ft_control_variants(args):
    jsonl_paths = getattr(args, "ft_control_jsonls", None)
    if jsonl_paths:
        return ft_control_variants_from_jsonls(jsonl_paths)
    ft_epochs_values = parse_int_list(getattr(args, "ft_epochs_list", None)) or [
        FT_EPOCHS
    ]
    target_min_loss_delta_values = parse_float_list(
        getattr(args, "target_min_loss_deltas", None)
    ) or [DEFAULT_TARGET_MIN_LOSS_DELTA]
    patience_values = parse_optional_int_list(getattr(args, "patiences", None)) or [None]
    min_delta_values = parse_float_list(getattr(args, "min_deltas", None)) or [
        DEFAULT_MIN_DELTA
    ]
    lr_decay_patience_values = parse_optional_int_list(
        getattr(args, "lr_decay_patiences", None)
    ) or [None]
    lr_decay_factor_values = parse_float_list(
        getattr(args, "lr_decay_factors", None)
    ) or [DEFAULT_LR_DECAY_FACTOR]
    lr_decay_min_delta_values = parse_float_list(
        getattr(args, "lr_decay_min_deltas", None)
    ) or [DEFAULT_LR_DECAY_MIN_DELTA]
    label_ft_epochs = getattr(args, "ft_epochs_list", None) is not None or len(
        ft_epochs_values
    ) > 1
    label_target_min = getattr(args, "target_min_loss_deltas", None) is not None or len(
        target_min_loss_delta_values
    ) > 1
    label_patience = getattr(args, "patiences", None) is not None or len(
        patience_values
    ) > 1
    label_min_delta = getattr(args, "min_deltas", None) is not None or len(
        min_delta_values
    ) > 1
    label_lr_decay_patience = getattr(args, "lr_decay_patiences", None) is not None or len(
        lr_decay_patience_values
    ) > 1
    label_lr_decay_factor = getattr(args, "lr_decay_factors", None) is not None or len(
        lr_decay_factor_values
    ) > 1
    label_lr_decay_min_delta = getattr(args, "lr_decay_min_deltas", None) is not None or len(
        lr_decay_min_delta_values
    ) > 1
    variants = []
    for ft_epochs in ft_epochs_values:
        for target_min_loss_delta in target_min_loss_delta_values:
            for patience in patience_values:
                for min_delta in min_delta_values:
                    for lr_decay_patience in lr_decay_patience_values:
                        for lr_decay_factor in lr_decay_factor_values:
                            for lr_decay_min_delta in lr_decay_min_delta_values:
                                labels = []
                                if label_ft_epochs or ft_epochs != FT_EPOCHS:
                                    labels.append(f"ep{ft_epochs}")
                                if (
                                    label_target_min
                                    or target_min_loss_delta != DEFAULT_TARGET_MIN_LOSS_DELTA
                                ):
                                    labels.append(
                                        f"tmin{label_number(target_min_loss_delta)}"
                                    )
                                if label_patience or patience is not None:
                                    labels.append(optional_int_label("pat", patience))
                                if label_min_delta or min_delta != DEFAULT_MIN_DELTA:
                                    labels.append(f"md{label_number(min_delta)}")
                                if label_lr_decay_patience or lr_decay_patience is not None:
                                    labels.append(optional_int_label("ldp", lr_decay_patience))
                                if label_lr_decay_factor or lr_decay_factor != DEFAULT_LR_DECAY_FACTOR:
                                    labels.append(f"ldf{label_number(lr_decay_factor)}")
                                if (
                                    label_lr_decay_min_delta
                                    or lr_decay_min_delta != DEFAULT_LR_DECAY_MIN_DELTA
                                ):
                                    labels.append(
                                        f"ldmd{label_number(lr_decay_min_delta)}"
                                    )
                                variants.append(
                                    {
                                        "label": "::".join(labels) if labels else None,
                                        "fields": {
                                            "ft_epochs": ft_epochs,
                                            "target_min_loss_delta_policy": target_min_loss_delta,
                                            "early_stopping_patience": patience,
                                            "early_stopping_min_delta": min_delta,
                                            "lr_decay_patience": lr_decay_patience,
                                            "lr_decay_factor": lr_decay_factor,
                                            "lr_decay_min_delta": lr_decay_min_delta,
                                        },
                                    }
                                )
    return variants


def config_for_checkpoint_variant(
    config,
    projection_variant,
    source_gain_variant,
    adapter_weight_decay_variant=None,
    training_policy_variant=None,
    ft_control_variant=None,
):
    labels = [
        label
        for label in [
            projection_variant["label"],
            source_gain_variant["label"],
            (
                adapter_weight_decay_variant["label"]
                if adapter_weight_decay_variant is not None
                else None
            ),
            (
                training_policy_variant["label"]
                if training_policy_variant is not None
                else None
            ),
            (
                ft_control_variant["label"]
                if ft_control_variant is not None
                else None
            ),
        ]
        if label is not None
    ]
    projected = dict(config)
    if adapter_weight_decay_variant is not None:
        projected.update(adapter_weight_decay_variant["fields"])
        projected["adapter_weight_decay_label"] = adapter_weight_decay_variant["label"]
    else:
        projected.setdefault("adapter_weight_decay", DEFAULT_ADAPTER_WEIGHT_DECAY)
        projected.setdefault("adapter_weight_decay_label", None)
    if training_policy_variant is not None:
        projected.update(training_policy_variant["fields"])
        projected["max_grad_norm_label"] = training_policy_variant[
            "max_grad_norm_label"
        ]
        projected["gradient_accumulation_steps_label"] = training_policy_variant[
            "gradient_accumulation_steps_label"
        ]
    else:
        projected.setdefault("max_grad_norm", DEFAULT_MAX_GRAD_NORM)
        projected.setdefault(
            "gradient_accumulation_steps",
            DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        )
        projected.setdefault("max_grad_norm_label", None)
        projected.setdefault("gradient_accumulation_steps_label", None)
    if ft_control_variant is not None:
        projected.update(ft_control_variant["fields"])
        projected["ft_control_label"] = ft_control_variant["label"]
    else:
        projected.setdefault("ft_epochs", FT_EPOCHS)
        projected.setdefault("target_min_loss_delta_policy", DEFAULT_TARGET_MIN_LOSS_DELTA)
        projected.setdefault("early_stopping_patience", None)
        projected.setdefault("early_stopping_min_delta", DEFAULT_MIN_DELTA)
        projected.setdefault("lr_decay_patience", None)
        projected.setdefault("lr_decay_factor", DEFAULT_LR_DECAY_FACTOR)
        projected.setdefault("lr_decay_min_delta", DEFAULT_LR_DECAY_MIN_DELTA)
        projected.setdefault("ft_control_label", None)
    if not labels:
        return projected
    projected["base_label"] = config["label"]
    projected["projection_label"] = projection_variant["label"]
    projected["source_gain_label"] = source_gain_variant["label"]
    projected["adapter_weight_decay_label"] = (
        adapter_weight_decay_variant["label"]
        if adapter_weight_decay_variant is not None
        else None
    )
    projected["max_grad_norm_label"] = (
        training_policy_variant["max_grad_norm_label"]
        if training_policy_variant is not None
        else None
    )
    projected["gradient_accumulation_steps_label"] = (
        training_policy_variant["gradient_accumulation_steps_label"]
        if training_policy_variant is not None
        else None
    )
    projected["ft_control_label"] = (
        ft_control_variant["label"] if ft_control_variant is not None else None
    )
    projected["label"] = "::".join([config["label"], *labels])
    return projected


def config_for_projection_variant(config, variant):
    gain_variant = {
        "label": None,
        "fields": {"checkpoint_source_gain": 1.0},
    }
    return config_for_checkpoint_variant(config, variant, gain_variant)


def loader(samples, seed):
    return st.dataset.from_vec(samples).shuffle(seed).batched(BATCH_WINDOWS)


def evaluate(session, trainer, model, loss, samples, seed):
    return session.evaluate_sparse_classification_epoch(
        trainer,
        model,
        loss,
        loader(samples, seed),
    )


def scale_state_dict(model, factor):
    scaled = {}
    for name, tensor in model.state_dict().items():
        scaled[name] = st.Tensor(
            tensor.rows,
            tensor.cols,
            [value * factor for value in tensor.data()],
        )
    model.load_state_dict(scaled)


def build_dense_mlp():
    model = Sequential(
        [
            Linear(VOCAB, HIDDEN, name="embed"),
            Relu(),
            Linear(HIDDEN, VOCAB, name="head"),
        ]
    )
    scale_state_dict(model, PARAM_SCALE)
    return model


def externalize_mlp_state(source_state, *, key_preset):
    return hf_lm_handoff_from_spiraltorch_state(source_state, key_preset=key_preset)


def externalize_mlp_state_from_hf_state_dict(args):
    shape_include_keys = hf_lm_shape_include_keys(
        args.key_preset,
        include_extra_keys=args.include_extra_keys,
    )
    shape_state, _ = load_hf_state_dict_shapes(
        args.hf_state_dict,
        include_keys=shape_include_keys,
    )
    key_preset = resolve_hf_lm_key_preset(args.key_preset, shape_state)
    include_keys = hf_lm_state_keys(
        key_preset,
        include_extra_keys=args.include_extra_keys,
    )
    shapes = infer_hf_lm_module_shapes(shape_state, key_preset=key_preset)
    expected = (VOCAB, HIDDEN, VOCAB)
    if shapes != expected and not args.allow_overlap_resize:
        raise RuntimeError(
            "byte MLP LoRA sweep HF state-dict source requires "
            f"(vocab, hidden, target_classes)={expected}, got {shapes}. "
            "Run checkpoint_preflight.py first for arbitrary real LLM shapes, "
            "or pass --allow-overlap-resize to explicitly overlap-copy into "
            "the byte-smoke adapter shape."
        )
    resize_kwargs = hf_lm_overlap_resize_kwargs() if args.allow_overlap_resize else {}
    tensor_bounds = (
        hf_lm_tensor_bounds_for_module_shapes(
            expected,
            key_preset=key_preset,
            lm_head_weight_transform=resize_kwargs.get(
                "lm_head_weight_transform",
                "transpose",
            ),
        )
        if args.allow_overlap_resize
        else None
    )
    raw_state, loaded_files = load_hf_state_dict(
        args.hf_state_dict,
        include_keys=include_keys,
        tensor_bounds=tensor_bounds,
    )
    checkpoint, rules = hf_lm_handoff_from_external_state(
        raw_state,
        key_preset=key_preset,
        synthesize_missing_biases=not args.no_synthesize_missing_biases,
        include_extra_keys=args.include_extra_keys,
        **resize_kwargs,
    )
    args.key_preset = key_preset
    return checkpoint, rules, loaded_files, shapes


def build_lora_mlp(
    source_state,
    source_key_map,
    rank,
    alpha,
    *,
    label_prefix,
    emit_preflight=True,
):
    embed = Linear(VOCAB, HIDDEN, name="embed")
    embed_report, embed_load = preflight_and_load(
        f"{label_prefix}_embed",
        embed,
        source_state,
        source_key_map,
        emit=emit_preflight,
    )
    embed.set_trainable(False)

    head = LoraLinear(HIDDEN, VOCAB, rank, alpha=alpha, name="head")
    head_report, head_load = preflight_and_load(
        f"{label_prefix}_head_base",
        head,
        source_state,
        source_key_map,
        lora_base=True,
        emit=emit_preflight,
    )
    return Sequential([embed, Relu(), head]), embed_load, head_load, embed_report, head_report


def new_runtime(
    max_grad_norm=DEFAULT_MAX_GRAD_NORM,
    gradient_accumulation_steps=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
):
    session = st.SpiralSession(
        device="wgpu",
        curvature=-1.0,
        hyper_learning_rate=0.5,
        fallback_learning_rate=0.1,
    )
    trainer = session.trainer()
    trainer.set_max_grad_norm(max_grad_norm)
    trainer.set_gradient_accumulation_steps(gradient_accumulation_steps)
    schedule = trainer.roundtable(rows=CONTEXT * BATCH_WINDOWS, cols=VOCAB)
    return session, trainer, schedule


def train_source(session, trainer, schedule, loss, source_samples):
    source = build_dense_mlp()
    session.prepare_module(source)
    before = evaluate(session, trainer, source, loss, source_samples, seed=7)
    last_stats = None
    for _ in range(SOURCE_EPOCHS):
        last_stats = session.train_epoch(
            trainer,
            source,
            loss,
            loader(source_samples, seed=7),
            schedule,
        )
    after = evaluate(session, trainer, source, loss, source_samples, seed=7)
    delta = sparse_classification_delta(before, after)
    if delta["loss_delta"] <= 0.0:
        raise RuntimeError(f"dense MLP source did not improve: {delta}")
    return source.state_dict(), last_stats, delta


def zero_source_delta():
    return {
        "loss_delta": 0.0,
        "accuracy_delta": 0.0,
        "perplexity_delta": 0.0,
    }


def fine_tune_config(
    session,
    trainer,
    schedule,
    loss,
    config,
    source_state,
    source_key_map,
    source_samples,
    target_samples,
    key_preset,
    projection_fields,
    source_gain_fields,
):
    max_grad_norm = config.get("max_grad_norm", DEFAULT_MAX_GRAD_NORM)
    gradient_accumulation_steps = config.get(
        "gradient_accumulation_steps",
        DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    )
    ft_epochs = config.get("ft_epochs", FT_EPOCHS)
    target_min_loss_delta = config.get(
        "target_min_loss_delta_policy",
        DEFAULT_TARGET_MIN_LOSS_DELTA,
    )
    patience = config.get("early_stopping_patience")
    min_delta = config.get("early_stopping_min_delta", DEFAULT_MIN_DELTA)
    lr_decay_patience = config.get("lr_decay_patience")
    lr_decay_factor = config.get("lr_decay_factor", DEFAULT_LR_DECAY_FACTOR)
    lr_decay_min_delta = config.get(
        "lr_decay_min_delta",
        DEFAULT_LR_DECAY_MIN_DELTA,
    )
    trainer.set_max_grad_norm(max_grad_norm)
    trainer.set_gradient_accumulation_steps(gradient_accumulation_steps)

    target, embed_load, head_load, embed_report, head_report = build_lora_mlp(
        source_state,
        source_key_map,
        config["rank"],
        config["alpha"],
        label_prefix=config["label"],
    )
    if not embed_load["matched"] or not head_load["matched"]:
        raise RuntimeError(
            f"{config['label']} base load mismatch: embed={embed_load} head={head_load}"
        )

    session.prepare_module(target)
    frozen_base = target.set_parameters_trainable_by_suffix("::weight", False)
    frozen_base += target.set_parameters_trainable_by_suffix("::bias", False)
    adapter_weight_decay = config.get(
        "adapter_weight_decay",
        DEFAULT_ADAPTER_WEIGHT_DECAY,
    )
    boosted_adapter = target.set_parameters_learning_rate_scale_by_contains(
        "lora_",
        config["adapter_lr_scale"],
    )
    decayed_adapter = target.set_parameters_weight_decay_by_contains(
        "lora_",
        adapter_weight_decay,
    )

    ft_report = session.train_epochs_restore_best_sparse_with_finetune_report(
        trainer,
        target,
        loss,
        loader(target_samples, seed=11),
        loader(target_samples, seed=17),
        loader(source_samples, seed=13),
        schedule,
        epochs=ft_epochs,
        movement_tolerance=FT_MOVEMENT_TOLERANCE,
        max_loss_increase=10.0,
        max_accuracy_drop=1.0,
        max_perplexity_increase=100.0,
        target_min_loss_delta=target_min_loss_delta,
        patience=patience,
        min_delta=min_delta,
        lr_decay_patience=lr_decay_patience,
        lr_decay_factor=lr_decay_factor,
        lr_decay_min_delta=lr_decay_min_delta,
    )

    summary = ft_report.summary()
    return {
        "config": config,
        "summary": summary,
        "captured": ft_report.captured,
        "ft": ft_report.target_delta,
        "retention": ft_report.retention_delta,
        "movement": ft_report.movement,
        "frozen_base": frozen_base,
        "boosted_adapter": boosted_adapter,
        "decayed_adapter": decayed_adapter,
        "embed_load": embed_load,
        "head_load": head_load,
        "embed_preflight": embed_report,
        "head_preflight": head_report,
        "key_preset": key_preset,
        "projection_fields": projection_fields,
        "source_gain_fields": source_gain_fields,
    }


def metric(delta, name):
    return delta[name]


def print_report(report):
    config = report["config"]
    margins = summary_guard_margins(report["summary"])
    guard_counts = attach_summary_guard_counts(
        dict(report["summary"]),
        report["captured"],
    )
    print(
        f"lora_config={config['label']} "
        f"rank={config['rank']} "
        f"alpha={config['alpha']:.3f} "
        f"adapter_lr_scale={config['adapter_lr_scale']:.3f} "
        f"adapter_weight_decay={config.get('adapter_weight_decay', DEFAULT_ADAPTER_WEIGHT_DECAY):.6f} "
        f"max_grad_norm={config.get('max_grad_norm', DEFAULT_MAX_GRAD_NORM):.6f} "
        f"gradient_accumulation_steps={config.get('gradient_accumulation_steps', DEFAULT_GRADIENT_ACCUMULATION_STEPS)} "
        f"ft_epochs={config.get('ft_epochs', FT_EPOCHS)} "
        f"target_min_loss_delta={config.get('target_min_loss_delta_policy', DEFAULT_TARGET_MIN_LOSS_DELTA):.6f} "
        f"patience={config.get('early_stopping_patience') if config.get('early_stopping_patience') is not None else 'none'} "
        f"min_delta={config.get('early_stopping_min_delta', DEFAULT_MIN_DELTA):.6f} "
        f"lr_decay_patience={config.get('lr_decay_patience') if config.get('lr_decay_patience') is not None else 'none'} "
        f"lr_decay_factor={config.get('lr_decay_factor', DEFAULT_LR_DECAY_FACTOR):.6f} "
        f"lr_decay_min_delta={config.get('lr_decay_min_delta', DEFAULT_LR_DECAY_MIN_DELTA):.6f} "
        f"checkpoint_key_preset={report['key_preset']} "
        f"checkpoint_projection={report['projection_fields']['checkpoint_projection']} "
        f"checkpoint_projection_strength={projection_value_label(report['projection_fields']['checkpoint_projection_strength'])} "
        f"checkpoint_projection_curvature={projection_value_label(report['projection_fields']['checkpoint_projection_curvature'])} "
        f"checkpoint_projection_frequency={projection_value_label(report['projection_fields']['checkpoint_projection_frequency'])} "
        f"checkpoint_source_gain={report['source_gain_fields']['checkpoint_source_gain']:.6f} "
        f"accepted={report['summary']['accepted']} "
        f"status={report['summary']['status']} "
        f"guarded_best_epoch={report['captured'].guarded_best_epoch} "
        f"guard_accepted_epochs={guard_counts['guard_accepted_epochs']} "
        f"guard_retention_rejected_epochs={guard_counts['guard_retention_rejected_epochs']} "
        f"guard_target_stale_epochs={guard_counts['guard_target_stale_epochs']} "
        f"guard_epoch_counts_available={guard_counts['guard_epoch_counts_available']} "
        f"ft_loss_delta={metric(report['ft'], 'loss_delta'):.6f} "
        f"ft_accuracy_delta={metric(report['ft'], 'accuracy_delta'):.6f} "
        f"ft_perplexity_delta={metric(report['ft'], 'perplexity_delta'):.6f} "
        f"retention_loss_delta={metric(report['retention'], 'loss_delta'):.6f} "
        f"retention_accuracy_delta={metric(report['retention'], 'accuracy_delta'):.6f} "
        f"retention_perplexity_delta={metric(report['retention'], 'perplexity_delta'):.6f} "
        f"target_loss_margin={margins['target_loss_margin']:.6f} "
        f"retention_loss_margin={margins['retention_loss_margin']:.6f} "
        f"retention_accuracy_margin={margins['retention_accuracy_margin']:.6f} "
        f"movement_status={report['movement']['status']} "
        f"frozen_stable={report['movement']['frozen_stable']} "
        f"trainable_moved={report['movement']['trainable_movement_observed']} "
        f"frozen_base_params={report['frozen_base']} "
        f"boosted_adapter_params={report['boosted_adapter']} "
        f"decayed_adapter_params={report['decayed_adapter']} "
        f"resume_hash={report['summary']['resume_hash']}"
    )


def resolved_checkpoint_source_label(args, source_origin):
    if args.checkpoint_source_label:
        return args.checkpoint_source_label
    if source_origin == "hf_state_dict":
        return args.key_preset
    return source_origin


def summary_row(
    report,
    source_delta,
    source_rows,
    target_rows,
    *,
    case_label,
    source_origin,
    source_label,
    loaded_files,
    checkpoint_shapes,
    overlap_resize,
):
    config = report["config"]
    row = dict(report["summary"])
    attach_summary_guard_margins(row)
    attach_summary_guard_counts(row, report["captured"])
    attach_epoch_tensor_backend_fields(row, report["summary"], report["captured"])
    attach_requested_wgpu_component_backend_summary(row)
    row.update(
        {
            "example": "byte_lm_mlp_lora_sweep",
            "case": case_label,
            "config": config["label"],
            "base_config": config.get("base_label", config["label"]),
            "checkpoint_projection_variant": config.get("projection_label"),
            "checkpoint_source_gain_variant": config.get("source_gain_label"),
            "adapter_weight_decay_variant": config.get("adapter_weight_decay_label"),
            "max_grad_norm_variant": config.get("max_grad_norm_label"),
            "gradient_accumulation_steps_variant": config.get(
                "gradient_accumulation_steps_label"
            ),
            "ft_control_variant": config.get("ft_control_label"),
            "rank": config["rank"],
            "alpha": config["alpha"],
            "adapter_lr_scale": config["adapter_lr_scale"],
            "adapter_weight_decay": config.get(
                "adapter_weight_decay",
                DEFAULT_ADAPTER_WEIGHT_DECAY,
            ),
            "max_grad_norm": config.get("max_grad_norm", DEFAULT_MAX_GRAD_NORM),
            "gradient_accumulation_steps": config.get(
                "gradient_accumulation_steps",
                DEFAULT_GRADIENT_ACCUMULATION_STEPS,
            ),
            "ft_epochs": config.get("ft_epochs", FT_EPOCHS),
            "target_min_loss_delta_policy": config.get(
                "target_min_loss_delta_policy",
                DEFAULT_TARGET_MIN_LOSS_DELTA,
            ),
            "early_stopping_patience": config.get("early_stopping_patience"),
            "early_stopping_min_delta": config.get(
                "early_stopping_min_delta",
                DEFAULT_MIN_DELTA,
            ),
            "lr_decay_patience": config.get("lr_decay_patience"),
            "lr_decay_factor": config.get("lr_decay_factor", DEFAULT_LR_DECAY_FACTOR),
            "lr_decay_min_delta": config.get(
                "lr_decay_min_delta",
                DEFAULT_LR_DECAY_MIN_DELTA,
            ),
            "ft_early_stopped": report["captured"].early_stopped,
            "ft_stop_epoch": report["captured"].stop_epoch,
            "ft_lr_decay_steps": report["captured"].lr_decay_steps,
            "ft_final_hyper_learning_rate": report[
                "captured"
            ].final_hyper_learning_rate,
            "ft_final_fallback_learning_rate": report[
                "captured"
            ].final_fallback_learning_rate,
            "source_rows": source_rows,
            "target_rows": target_rows,
            "source_loss_delta": metric(source_delta, "loss_delta"),
            "source_accuracy_delta": metric(source_delta, "accuracy_delta"),
            "source_perplexity_delta": metric(source_delta, "perplexity_delta"),
            "frozen_base_params": report["frozen_base"],
            "boosted_adapter_params": report["boosted_adapter"],
            "decayed_adapter_params": report["decayed_adapter"],
            "checkpoint_key_preset": report["key_preset"],
            "checkpoint_source_origin": source_origin,
            "checkpoint_source_label": source_label,
            "checkpoint_loaded_files": len(loaded_files),
            "checkpoint_vocab": checkpoint_shapes[0],
            "checkpoint_hidden": checkpoint_shapes[1],
            "checkpoint_target_classes": checkpoint_shapes[2],
            "checkpoint_overlap_resize": overlap_resize,
        }
    )
    row.update(report["projection_fields"])
    row.update(report["source_gain_fields"])
    row.update(
        checkpoint_audit_fields(
            "embed",
            report["embed_preflight"],
            report["embed_load"],
        )
    )
    row.update(
        checkpoint_audit_fields(
            "head",
            report["head_preflight"],
            report["head_load"],
        )
    )
    attach_training_policy_key(row)
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
            if "config" not in row:
                raise ValueError(f"{path}:{line_no} missing 'config'")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} did not contain any summary rows")
    return rows


def rows_by_config(rows, label):
    by_config = {}
    for row in rows:
        config = row_compare_key(row)
        if not isinstance(config, str) or not config:
            raise ValueError(f"{label} row has invalid config: {config!r}")
        if config in by_config:
            raise ValueError(f"{label} contains duplicate config: {config}")
        by_config[config] = row
    return by_config


def row_compare_key(row):
    config = row.get("config")
    case = row.get("case")
    if case and case != DEFAULT_CASE_LABEL:
        return f"{case}::{config}"
    return config


def is_numeric_value(value):
    return not isinstance(value, bool) and isinstance(value, (int, float))


def numeric_value(row, key):
    value = row.get(key)
    if not is_numeric_value(value):
        raise ValueError(f"{row.get('config')} row missing numeric {key}")
    return float(value)


def optional_numeric_value(row, key):
    value = row.get(key)
    if not is_numeric_value(value):
        return float("nan")
    return float(value)


def optional_int_value(row, key):
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def required_aggregate_int_value(row, key):
    value = optional_int_value(row, key)
    if value is None:
        raise ValueError(f"aggregate row {row.get('config')} missing integer {key}")
    return value


def required_aggregate_bool_value(row, key):
    value = row.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"aggregate row {row.get('config')} missing boolean {key}")
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


def winner_score(row):
    return (
        numeric_value(row, "target_loss_delta"),
        numeric_value(row, "retention_loss_delta"),
        numeric_value(row, "retention_accuracy_delta"),
    )


def sweep_winner(rows):
    candidates = [
        row
        for row in rows
        if summary_bool_value(row, "accepted", False)
        and summary_bool_value(row, "movement_ok", False)
        and numeric_value(row, "target_loss_delta") > 0.0
    ]
    if not candidates:
        raise RuntimeError("LoRA sweep compare has no accepted improving rows")
    best = max(winner_score(row) for row in candidates)
    winners = sorted(
        row_compare_key(row)
        for row in candidates
        if winner_score(row) == best
    )
    return "+".join(winners), best


def optional_sweep_winner(rows):
    try:
        return sweep_winner(rows)
    except RuntimeError:
        return None, None


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
    require_winner_match,
    allow_missing_current=False,
):
    current = rows_by_config(current_rows, "current")
    baseline = rows_by_config(baseline_rows, "baseline")
    missing_current = sorted(set(baseline) - set(current))
    if missing_current:
        for config in missing_current:
            print(f"summary_compare config={config} current_missing=true")
        if not allow_missing_current:
            raise RuntimeError(
                "baseline configs missing from current sweep: " + ",".join(missing_current)
            )

    common = sorted(set(current) & set(baseline))
    if not common:
        raise RuntimeError("no overlapping configs between current sweep and baseline")

    failures = []
    current_winner, current_score = optional_sweep_winner(list(current.values()))
    baseline_winner, baseline_score = optional_sweep_winner(list(baseline.values()))
    winner_changed = current_winner != baseline_winner
    current_target_score = current_score[0] if current_score is not None else float("nan")
    current_retention_score = (
        current_score[1] if current_score is not None else float("nan")
    )
    current_retention_accuracy_score = (
        current_score[2] if current_score is not None else float("nan")
    )
    baseline_target_score = (
        baseline_score[0] if baseline_score is not None else float("nan")
    )
    baseline_retention_score = (
        baseline_score[1] if baseline_score is not None else float("nan")
    )
    baseline_retention_accuracy_score = (
        baseline_score[2] if baseline_score is not None else float("nan")
    )
    winner_compare_passed = (
        not require_winner_match
        or (
            current_winner is not None
            and baseline_winner is not None
            and not winner_changed
        )
    )
    print(
        f"winner_compare before={baseline_winner or 'none'} "
        f"after={current_winner or 'none'} "
        f"target_loss_delta_before={baseline_target_score:.9f} "
        f"target_loss_delta_after={current_target_score:.9f} "
        f"retention_loss_delta_before={baseline_retention_score:.9f} "
        f"retention_loss_delta_after={current_retention_score:.9f} "
        f"retention_accuracy_delta_before={baseline_retention_accuracy_score:.9f} "
        f"retention_accuracy_delta_after={current_retention_accuracy_score:.9f} "
        f"winner_changed={winner_changed} "
        f"passed={winner_compare_passed}"
    )
    if require_winner_match:
        if current_winner is None or baseline_winner is None:
            failures.append("winner unavailable")
        elif winner_changed:
            failures.append(
                f"winner changed from {baseline_winner} to {current_winner}"
            )

    for config in common:
        now = current[config]
        before = baseline[config]
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
        print(
            f"summary_compare config={config} "
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
                config,
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
            failures.extend(checkpoint_audit_failures(config, now, before))

    extra_current = sorted(set(current) - set(baseline))
    for config in extra_current:
        print(f"summary_compare config={config} baseline_missing=true")
    if failures:
        raise RuntimeError("LoRA sweep regression gate failed: " + "; ".join(failures))
    return len(common)


def mean(values):
    values = list(values)
    if not values:
        raise ValueError("cannot average an empty value list")
    return sum(values) / len(values)


def optional_numeric_or_zero(row, key):
    value = optional_numeric_value(row, key)
    return 0.0 if value != value else value


GUARD_COUNT_KEYS = (
    "guard_accepted_epochs",
    "guard_retention_rejected_epochs",
    "guard_target_stale_epochs",
)


def guard_epochs_run_or_inferred(row):
    value = optional_numeric_value(row, "guard_epochs_run")
    if value == value:
        return value
    total = sum(optional_numeric_or_zero(row, key) for key in GUARD_COUNT_KEYS)
    if total > 0.0:
        return total
    return optional_numeric_or_zero(row, "epochs_run")


def guard_rate_or_count_ratio(row, rate_key, count_key):
    value = optional_numeric_value(row, rate_key)
    if value == value:
        return value
    epochs = guard_epochs_run_or_inferred(row)
    if epochs <= 0.0:
        return 0.0
    return optional_numeric_or_zero(row, count_key) / epochs


def consistent_value(rows, key):
    values = [row.get(key) for row in rows]
    first = values[0]
    if any(value != first for value in values[1:]):
        config = rows[0].get("config")
        raise ValueError(f"config {config} has inconsistent aggregate field {key}")
    return first


def consistent_training_policy_key(rows):
    keys = [training_policy_key(row) for row in rows]
    for row, key in zip(rows, keys):
        existing = row.get("training_policy_key")
        if existing is not None and existing != key:
            config = rows[0].get("config")
            raise ValueError(f"config {config} has stale aggregate field training_policy_key")
    first = keys[0]
    if any(key != first for key in keys[1:]):
        config = rows[0].get("config")
        raise ValueError(f"config {config} has inconsistent aggregate field training_policy_key")
    return first


def config_rows(rows):
    by_config = {}
    for row in rows:
        config = row.get("config")
        if not isinstance(config, str) or not config:
            raise ValueError(f"row has invalid config: {config!r}")
        by_config.setdefault(config, []).append(row)
    return by_config


def aggregate_config_rows(rows):
    aggregates = []
    for config, grouped in config_rows(rows).items():
        case_labels = []
        for row in grouped:
            case = row.get("case") or DEFAULT_CASE_LABEL
            if not isinstance(case, str) or not case:
                raise ValueError(f"config {config} has invalid aggregate case: {case!r}")
            if case in case_labels:
                raise ValueError(f"config {config} contains duplicate case: {case}")
            case_labels.append(case)
        case_count = len(case_labels)
        accepted_cases = sum(
            1 for row in grouped if summary_bool_value(row, "accepted", False)
        )
        movement_ok_cases = sum(
            1 for row in grouped if summary_bool_value(row, "movement_ok", False)
        )
        guard_epoch_counts_available_cases = sum(
            1
            for row in grouped
            if summary_bool_value(row, "guard_epoch_counts_available", False)
        )
        guard_accepted_epochs = [
            optional_numeric_or_zero(row, "guard_accepted_epochs") for row in grouped
        ]
        guard_retention_rejected_epochs = [
            optional_numeric_or_zero(row, "guard_retention_rejected_epochs")
            for row in grouped
        ]
        guard_target_stale_epochs = [
            optional_numeric_or_zero(row, "guard_target_stale_epochs") for row in grouped
        ]
        guard_acceptance_rates = [
            guard_rate_or_count_ratio(
                row,
                "guard_acceptance_rate",
                "guard_accepted_epochs",
            )
            for row in grouped
        ]
        guard_retention_rejected_rates = [
            guard_rate_or_count_ratio(
                row,
                "guard_retention_rejected_rate",
                "guard_retention_rejected_epochs",
            )
            for row in grouped
        ]
        guard_target_stale_rates = [
            guard_rate_or_count_ratio(
                row,
                "guard_target_stale_rate",
                "guard_target_stale_epochs",
            )
            for row in grouped
        ]
        aggregate = {
            "row_type": "config_aggregate",
            "config": config,
            "base_config": consistent_value(grouped, "base_config"),
            "checkpoint_projection_variant": consistent_value(
                grouped,
                "checkpoint_projection_variant",
            ),
            "checkpoint_source_gain_variant": consistent_value(
                grouped,
                "checkpoint_source_gain_variant",
            ),
            "adapter_weight_decay_variant": consistent_value(
                grouped,
                "adapter_weight_decay_variant",
            ),
            "adapter_weight_decay": consistent_value(grouped, "adapter_weight_decay"),
            "max_grad_norm_variant": consistent_value(
                grouped,
                "max_grad_norm_variant",
            ),
            "max_grad_norm": consistent_value(grouped, "max_grad_norm"),
            "gradient_accumulation_steps_variant": consistent_value(
                grouped,
                "gradient_accumulation_steps_variant",
            ),
            "gradient_accumulation_steps": consistent_value(
                grouped,
                "gradient_accumulation_steps",
            ),
            "ft_control_variant": consistent_value(grouped, "ft_control_variant"),
            "ft_epochs": consistent_value(grouped, "ft_epochs"),
            "target_min_loss_delta_policy": consistent_value(
                grouped,
                "target_min_loss_delta_policy",
            ),
            "early_stopping_patience": consistent_value(
                grouped,
                "early_stopping_patience",
            ),
            "early_stopping_min_delta": consistent_value(
                grouped,
                "early_stopping_min_delta",
            ),
            "lr_decay_patience": consistent_value(grouped, "lr_decay_patience"),
            "lr_decay_factor": consistent_value(grouped, "lr_decay_factor"),
            "lr_decay_min_delta": consistent_value(grouped, "lr_decay_min_delta"),
            "training_policy_key": consistent_training_policy_key(grouped),
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
            "guard_epoch_counts_available_cases": guard_epoch_counts_available_cases,
            "guard_epoch_counts_available_all": (
                guard_epoch_counts_available_cases == case_count
            ),
            "guard_accepted_epochs_total": sum(guard_accepted_epochs),
            "guard_accepted_epochs_mean": mean(guard_accepted_epochs),
            "guard_accepted_epochs_max": max(guard_accepted_epochs),
            "guard_retention_rejected_epochs_total": sum(
                guard_retention_rejected_epochs
            ),
            "guard_retention_rejected_epochs_mean": mean(
                guard_retention_rejected_epochs
            ),
            "guard_retention_rejected_epochs_max": max(guard_retention_rejected_epochs),
            "guard_target_stale_epochs_total": sum(guard_target_stale_epochs),
            "guard_target_stale_epochs_mean": mean(guard_target_stale_epochs),
            "guard_target_stale_epochs_max": max(guard_target_stale_epochs),
            "guard_acceptance_rate_mean": mean(guard_acceptance_rates),
            "guard_acceptance_rate_min": min(guard_acceptance_rates),
            "guard_retention_rejected_rate_mean": mean(guard_retention_rejected_rates),
            "guard_retention_rejected_rate_max": max(guard_retention_rejected_rates),
            "guard_target_stale_rate_mean": mean(guard_target_stale_rates),
            "guard_target_stale_rate_max": max(guard_target_stale_rates),
            "ft_early_stopped_cases": sum(
                1 for row in grouped if summary_bool_value(row, "ft_early_stopped", False)
            ),
            "ft_early_stopped_any": any(
                summary_bool_value(row, "ft_early_stopped", False) for row in grouped
            ),
            "ft_lr_decay_steps_mean": mean(
                numeric_value(row, "ft_lr_decay_steps") for row in grouped
            ),
            "ft_lr_decay_steps_max": max(
                numeric_value(row, "ft_lr_decay_steps") for row in grouped
            ),
            "ft_final_hyper_learning_rate_min": min(
                numeric_value(row, "ft_final_hyper_learning_rate") for row in grouped
            ),
            "ft_final_fallback_learning_rate_min": min(
                numeric_value(row, "ft_final_fallback_learning_rate") for row in grouped
            ),
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
            "retention_accuracy_margin_mean": mean(
                numeric_value(row, "retention_accuracy_margin") for row in grouped
            ),
            "retention_accuracy_margin_min": min(
                numeric_value(row, "retention_accuracy_margin") for row in grouped
            ),
            "retention_perplexity_margin_mean": mean(
                numeric_value(row, "retention_perplexity_margin") for row in grouped
            ),
            "retention_perplexity_margin_min": min(
                numeric_value(row, "retention_perplexity_margin") for row in grouped
            ),
            "checkpoint_key_preset": consistent_value(grouped, "checkpoint_key_preset"),
            "checkpoint_source_origin": consistent_value(
                grouped,
                "checkpoint_source_origin",
            ),
            "checkpoint_source_label": consistent_value(
                grouped,
                "checkpoint_source_label",
            ),
            "checkpoint_loaded_files": consistent_value(
                grouped,
                "checkpoint_loaded_files",
            ),
            "checkpoint_vocab": consistent_value(grouped, "checkpoint_vocab"),
            "checkpoint_hidden": consistent_value(grouped, "checkpoint_hidden"),
            "checkpoint_target_classes": consistent_value(
                grouped,
                "checkpoint_target_classes",
            ),
            "checkpoint_overlap_resize": consistent_value(
                grouped,
                "checkpoint_overlap_resize",
            ),
            "checkpoint_projection": consistent_value(grouped, "checkpoint_projection"),
            "checkpoint_projection_strength": consistent_value(
                grouped,
                "checkpoint_projection_strength",
            ),
            "checkpoint_projection_curvature": consistent_value(
                grouped,
                "checkpoint_projection_curvature",
            ),
            "checkpoint_projection_frequency": consistent_value(
                grouped,
                "checkpoint_projection_frequency",
            ),
            "checkpoint_source_gain": consistent_value(
                grouped,
                "checkpoint_source_gain",
            ),
        }
        attach_requested_wgpu_component_backend_summary(
            aggregate,
            source_rows=grouped,
        )
        attach_epoch_tensor_backend_aggregate_fields(aggregate, grouped)
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
        raise RuntimeError("LoRA sweep aggregate compare has no accepted improving rows")
    best = max(aggregate_score(row) for row in candidates)
    winners = sorted(row["config"] for row in candidates if aggregate_score(row) == best)
    return "+".join(winners), best


def optional_aggregate_winner(rows):
    try:
        return aggregate_winner(rows)
    except RuntimeError:
        return None, None


def aggregate_rows_by_config(rows, label):
    by_config = {}
    for row in rows:
        if row.get("row_type") != "config_aggregate":
            raise ValueError(f"{label} expected row_type='config_aggregate'")
        config = row.get("config")
        if not isinstance(config, str) or not config:
            raise ValueError(f"{label} aggregate row has invalid config: {config!r}")
        if config in by_config:
            raise ValueError(f"{label} contains duplicate aggregate config: {config}")
        validate_aggregate_row(row, label)
        by_config[config] = row
    return by_config


def aggregate_case_labels(row):
    labels = row.get("case_labels")
    if not isinstance(labels, str):
        return []
    return [label for label in labels.split(",") if label]


def aggregate_row_consistency_failures(row, label):
    config = row.get("config")
    prefix = f"{label} aggregate {config}"
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

    for count_key in [
        "accepted_cases",
        "rejected_cases",
        "movement_ok_cases",
        "movement_not_ok_cases",
        "guard_epoch_counts_available_cases",
    ]:
        if count_key in row:
            count = optional_int_value(row, count_key)
            if count is None or count < 0:
                failures.append(f"{prefix}: {count_key} must be a non-negative integer")
            elif count > cases:
                failures.append(f"{prefix}: {count_key} {count} exceeds cases {cases}")

    count_pairs = [
        ("accepted_cases", "rejected_cases"),
        ("movement_ok_cases", "movement_not_ok_cases"),
    ]
    for left_key, right_key in count_pairs:
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
        ("guard_epoch_counts_available_cases", "guard_epoch_counts_available_all"),
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
    for metric_key in [
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
    ]:
        if metric_key in row:
            value = row.get(metric_key)
            if not is_numeric_value(value) or float(value) < 0.0:
                failures.append(f"{prefix}: {metric_key} must be non-negative numeric")
            elif (
                metric_key.endswith("_rate_mean")
                or metric_key.endswith("_rate_min")
                or metric_key.endswith("_rate_max")
            ):
                if float(value) > 1.0:
                    failures.append(f"{prefix}: {metric_key} must be at most 1.0")
    return failures


def validate_aggregate_row(row, label):
    failures = aggregate_row_consistency_failures(row, label)
    if failures:
        raise ValueError("; ".join(failures))
    return row


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
        config = row.get("config")
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
            f"aggregate_acceptance config={config} "
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
                f"{config}: aggregate cases {cases} below floor {min_cases}"
            )
        if missing_cases:
            failures.append(
                f"{config}: missing aggregate cases {','.join(missing_cases)}"
            )
        if require_accepted_all and not accepted_all:
            failures.append(
                f"{config}: accepted {accepted_cases}/{cases} aggregate cases"
            )
        if min_accepted_rate is not None and accepted_rate < min_accepted_rate:
            failures.append(
                f"{config}: accepted_rate {accepted_rate:.9f} below floor {min_accepted_rate:.9f}"
            )
        if min_movement_ok_rate is not None and movement_ok_rate < min_movement_ok_rate:
            failures.append(
                f"{config}: movement_ok_rate {movement_ok_rate:.9f} below floor {min_movement_ok_rate:.9f}"
            )
    if failures:
        raise RuntimeError("LoRA sweep aggregate coverage gate failed: " + "; ".join(failures))
    return len(rows)


def check_aggregate_accepted_all(rows):
    return check_aggregate_coverage(rows, require_accepted_all=True)


def compare_aggregate_rows(current_rows, baseline_rows, args):
    current = aggregate_rows_by_config(current_rows, "current")
    baseline = aggregate_rows_by_config(baseline_rows, "baseline")
    missing_current = sorted(set(baseline) - set(current))
    if missing_current:
        raise RuntimeError(
            "baseline aggregate configs missing from current compare: "
            + ",".join(missing_current)
        )
    common = sorted(set(current) & set(baseline))
    if not common:
        raise RuntimeError("no overlapping aggregate configs between current and baseline")

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
            failures.append(
                f"aggregate winner changed from {baseline_winner} to {current_winner}"
            )

    for config in common:
        now = current[config]
        before = baseline[config]
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
        target_margin = numeric_value(now, "target_loss_margin_min")
        retention_margin = numeric_value(now, "retention_loss_margin_min")
        min_retention_accuracy_margin = getattr(
            args,
            "min_aggregate_retention_accuracy_margin",
            None,
        )
        min_retention_perplexity_margin = getattr(
            args,
            "min_aggregate_retention_perplexity_margin",
            None,
        )
        retention_accuracy_margin = (
            numeric_value(now, "retention_accuracy_margin_min")
            if min_retention_accuracy_margin is not None
            else optional_numeric_value(now, "retention_accuracy_margin_min")
        )
        retention_perplexity_margin = (
            numeric_value(now, "retention_perplexity_margin_min")
            if min_retention_perplexity_margin is not None
            else optional_numeric_value(now, "retention_perplexity_margin_min")
        )
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
            or target_margin >= args.min_aggregate_target_loss_margin
        )
        retention_margin_ok = (
            args.min_aggregate_retention_loss_margin is None
            or retention_margin >= args.min_aggregate_retention_loss_margin
        )
        retention_accuracy_margin_ok = (
            min_retention_accuracy_margin is None
            or retention_accuracy_margin >= min_retention_accuracy_margin
        )
        retention_perplexity_margin_ok = (
            min_retention_perplexity_margin is None
            or retention_perplexity_margin >= min_retention_perplexity_margin
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
            and retention_accuracy_margin_ok
            and retention_perplexity_margin_ok
            and checkpoint_ok
        )
        print(
            f"aggregate_compare config={config} "
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
            f"target_loss_margin_min={target_margin:.9f} "
            f"retention_loss_margin_min={retention_margin:.9f} "
            f"retention_accuracy_margin_min={retention_accuracy_margin:.9f} "
            f"retention_perplexity_margin_min={retention_perplexity_margin:.9f} "
            f"checkpoint_changed={checkpoint_changed} "
            f"scope_changed={scope_changed} "
            f"passed={row_passed}"
        )
        if scope_changed:
            failures.append(f"{config}: aggregate case scope changed")
        if (
            args.max_aggregate_target_loss_regression is not None
            and target_regression > args.max_aggregate_target_loss_regression
        ):
            failures.append(
                f"{config}: aggregate target_loss_delta_mean regressed by {target_regression:.9f}"
            )
        if (
            args.max_aggregate_retention_loss_regression is not None
            and retention_regression > args.max_aggregate_retention_loss_regression
        ):
            failures.append(
                f"{config}: aggregate retention_loss_delta_mean regressed by {retention_regression:.9f}"
            )
        if (
            args.max_aggregate_accepted_rate_regression is not None
            and accepted_rate_regression > args.max_aggregate_accepted_rate_regression
        ):
            failures.append(
                f"{config}: aggregate accepted_rate regressed by {accepted_rate_regression:.9f}"
            )
        if (
            args.max_aggregate_movement_ok_rate_regression is not None
            and movement_ok_rate_regression
            > args.max_aggregate_movement_ok_rate_regression
        ):
            failures.append(
                f"{config}: aggregate movement_ok_rate regressed by {movement_ok_rate_regression:.9f}"
            )
        if (
            args.min_aggregate_target_loss_margin is not None
            and target_margin < args.min_aggregate_target_loss_margin
        ):
            failures.append(
                f"{config}: aggregate target_loss_margin_min {target_margin:.9f} below floor"
            )
        if (
            args.min_aggregate_retention_loss_margin is not None
            and retention_margin < args.min_aggregate_retention_loss_margin
        ):
            failures.append(
                f"{config}: aggregate retention_loss_margin_min {retention_margin:.9f} below floor"
            )
        if (
            min_retention_accuracy_margin is not None
            and retention_accuracy_margin < min_retention_accuracy_margin
        ):
            failures.append(
                f"{config}: aggregate retention_accuracy_margin_min {retention_accuracy_margin:.9f} below floor"
            )
        if (
            min_retention_perplexity_margin is not None
            and retention_perplexity_margin < min_retention_perplexity_margin
        ):
            failures.append(
                f"{config}: aggregate retention_perplexity_margin_min {retention_perplexity_margin:.9f} below floor"
            )
        if args.require_checkpoint_match:
            failures.extend(checkpoint_audit_failures(config, now, before))

    if failures:
        raise RuntimeError("LoRA sweep aggregate regression gate failed: " + "; ".join(failures))
    return len(common)


def run_case(args, case, config_specs):
    case_label = case["label"]
    source_text = case["source_text"]
    target_text = case["target_text"]
    source_samples = st.dataset.byte_lm_windows(source_text, CONTEXT)
    target_samples = st.dataset.byte_lm_windows(target_text, CONTEXT)
    source_rows = st.dataset.byte_lm_sample_stats(source_samples)["active_rows"]
    target_rows = st.dataset.byte_lm_sample_stats(target_samples)["active_rows"]

    session, trainer, schedule = new_runtime()
    loss = SoftmaxCrossEntropy()
    if args.hf_state_dict is None:
        source_state, source_stats, source_delta = train_source(
            session,
            trainer,
            schedule,
            loss,
            source_samples,
        )
        external_source_state, source_key_map = externalize_mlp_state(
            source_state,
            key_preset=args.key_preset,
        )
        source_origin = "trained_dense"
        loaded_files = []
        checkpoint_shapes = (VOCAB, HIDDEN, VOCAB)
        overlap_resize = False
    else:
        external_source_state, source_key_map, loaded_files, checkpoint_shapes = (
            externalize_mlp_state_from_hf_state_dict(args)
        )
        source_stats = SimpleNamespace(batches=0, optimizer_steps=0)
        source_delta = zero_source_delta()
        source_origin = "hf_state_dict"
        overlap_resize = args.allow_overlap_resize
    checkpoint_source_label = resolved_checkpoint_source_label(args, source_origin)
    projection_variants = checkpoint_projection_variants(args)
    source_gain_variants = checkpoint_source_gain_variants(args)
    adapter_weight_decay_variants_ = adapter_weight_decay_variants(args)
    training_policy_variants_ = training_policy_variants(args)
    ft_control_variants_ = ft_control_variants(args)
    reports = []
    for projection_variant in projection_variants:
        projected_source_state = apply_checkpoint_projection(
            external_source_state,
            source_key_map,
            projection_variant["args"],
        )
        for source_gain_variant in source_gain_variants:
            gained_source_state = apply_checkpoint_source_gain(
                projected_source_state,
                source_key_map,
                source_gain_variant["args"],
            )
            for config in config_specs:
                for adapter_weight_decay_variant in adapter_weight_decay_variants_:
                    for training_policy_variant in training_policy_variants_:
                        for ft_control_variant in ft_control_variants_:
                            reports.append(
                                fine_tune_config(
                                    session,
                                    trainer,
                                    schedule,
                                    loss,
                                    config_for_checkpoint_variant(
                                        config,
                                        projection_variant,
                                        source_gain_variant,
                                        adapter_weight_decay_variant,
                                        training_policy_variant,
                                        ft_control_variant,
                                    ),
                                    gained_source_state,
                                    source_key_map,
                                    source_samples,
                                    target_samples,
                                    args.key_preset,
                                    projection_variant["fields"],
                                    source_gain_variant["fields"],
                                )
                            )

    accepted = [
        report
        for report in reports
        if report["summary"]["accepted"]
        and report["summary"]["movement_ok"]
        and metric(report["ft"], "loss_delta") > 0.0
    ]
    if not accepted:
        print(
            f"sweep_case_no_accepted case={case_label} "
            f"configs={len(reports)}"
        )

    header_projection_fields = checkpoint_projection_fields(args)
    print(
        f"sweep=python_byte_lm_mlp_lora "
        f"case={case_label} "
        f"vocab={VOCAB} hidden={HIDDEN} context={CONTEXT} "
        f"source_windows={len(source_samples)} target_windows={len(target_samples)} "
        f"source_rows={source_rows} target_rows={target_rows} "
        f"source_batches={source_stats.batches} "
        f"source_optimizer_steps={source_stats.optimizer_steps} "
        f"source_accumulation_steps={DEFAULT_GRADIENT_ACCUMULATION_STEPS} "
        f"checkpoint_key_preset={args.key_preset} "
        f"checkpoint_source_origin={source_origin} "
        f"checkpoint_source_label={checkpoint_source_label} "
        f"checkpoint_loaded_files={len(loaded_files)} "
        f"checkpoint_overlap_resize={overlap_resize} "
        f"checkpoint_projection={header_projection_fields['checkpoint_projection']} "
        f"checkpoint_projection_strength={projection_value_label(header_projection_fields['checkpoint_projection_strength'])} "
        f"checkpoint_projection_curvature={projection_value_label(header_projection_fields['checkpoint_projection_curvature'])} "
        f"checkpoint_projection_frequency={projection_value_label(header_projection_fields['checkpoint_projection_frequency'])} "
        f"checkpoint_source_gain={checkpoint_source_gain_value(args):.6f} "
        f"checkpoint_projection_grid={uses_checkpoint_projection_grid(args)} "
        f"checkpoint_source_gain_grid={uses_checkpoint_source_gain_grid(args)} "
        f"adapter_weight_decay_grid={uses_adapter_weight_decay_grid(args)} "
        f"training_policy_grid={uses_training_policy_grid(args)} "
        f"ft_control_grid={uses_ft_control_grid(args)} "
        f"projection_variants={len(projection_variants)} "
        f"source_gain_variants={len(source_gain_variants)} "
        f"adapter_weight_decay_variants={len(adapter_weight_decay_variants_)} "
        f"training_policy_variants={len(training_policy_variants_)} "
        f"ft_control_variants={len(ft_control_variants_)} "
        f"configs={len(reports)}"
    )
    print(
        "source_delta "
        f"loss_delta={metric(source_delta, 'loss_delta'):.6f} "
        f"accuracy_delta={metric(source_delta, 'accuracy_delta'):.6f} "
        f"perplexity_delta={metric(source_delta, 'perplexity_delta'):.6f}"
    )
    for report in reports:
        print_report(report)

    rows = [
        summary_row(
            report,
            source_delta,
            source_rows,
            target_rows,
            case_label=case_label,
            source_origin=source_origin,
            source_label=checkpoint_source_label,
            loaded_files=loaded_files,
            checkpoint_shapes=checkpoint_shapes,
            overlap_resize=overlap_resize,
        )
        for report in reports
    ]
    return rows


def main():
    args = parse_args()
    case_specs = case_specs_with_external(args.case_jsonls)
    lora_config_specs = lora_configs_with_external(args.lora_config_jsonls)
    config_specs = selected_configs(args.configs, lora_config_specs)
    cases = selected_cases(args.cases, case_specs)
    rows = []
    for case in cases:
        rows.extend(run_case(args, case, config_specs))

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
            args.require_winner_match,
            allow_missing_current=args.configs is not None or args.cases is not None,
        )
        print(f"summary_compare_rows={compared} baseline={args.compare_jsonl}")
    aggregate_rows = aggregate_config_rows(rows)
    if args.aggregate_jsonl is not None:
        write_jsonl(args.aggregate_jsonl, aggregate_rows)
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
    if args.compare_aggregate_jsonl is not None:
        baseline_aggregates = load_jsonl(args.compare_aggregate_jsonl)
        compared = compare_aggregate_rows(aggregate_rows, baseline_aggregates, args)
        print(
            f"aggregate_compare_rows={compared} "
            f"baseline={args.compare_aggregate_jsonl}"
        )
    winner_key, winner_score_value = optional_sweep_winner(rows)
    if winner_key is None:
        print("lora_winner config=none winner_available=false")
    else:
        print(
            f"lora_winner config={winner_key} "
            f"ft_loss_delta={winner_score_value[0]:.6f} "
            f"retention_loss_delta={winner_score_value[1]:.6f} "
            f"retention_accuracy_delta={winner_score_value[2]:.6f} "
            f"winner_available=true"
        )


if __name__ == "__main__":
    main()
