import argparse
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
    project_checkpoint_tensors,
    projection_value_label,
    resolve_hf_lm_key_preset,
)
from sparse_finetune_compare import (
    add_summary_compare_args,
    attach_summary_guard_counts,
    attach_summary_guard_margins,
    compare_single_summary,
    load_single_summary_jsonl,
    validate_summary_compare_args,
    write_summary_jsonl,
)


VOCAB = st.dataset.BYTE_LM_VOCAB
HIDDEN = 24
CONTEXT = 4
BATCH_WINDOWS = 4
ACCUMULATION_STEPS = 2
SOURCE_EPOCHS = 4
FT_EPOCHS = 10
LORA_RANK = 12
LORA_ALPHA = 64.0
DEFAULT_ADAPTER_WEIGHT_DECAY = 0.0
DEFAULT_MAX_GRAD_NORM = 2.0
DEFAULT_GRADIENT_ACCUMULATION_STEPS = ACCUMULATION_STEPS
DEFAULT_TARGET_MIN_LOSS_DELTA = 0.0
DEFAULT_MIN_DELTA = 0.0
DEFAULT_LR_DECAY_FACTOR = 0.5
DEFAULT_LR_DECAY_MIN_DELTA = 0.0
PARAM_SCALE = 0.003
FT_MOVEMENT_TOLERANCE = 1e-6


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a tiny tokenizerless byte-MLP LoRA adapter FT smoke."
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Optional path for one flat SparseFineTuneReport summary row.",
    )
    parser.add_argument(
        "--compare-jsonl",
        type=Path,
        default=None,
        help="Optional previous summary JSONL row to compare against this run.",
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
            "Optional local HF/PyTorch state dict file or directory to use as "
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
        "--adapter-weight-decay",
        type=float,
        default=DEFAULT_ADAPTER_WEIGHT_DECAY,
        help=(
            "Decoupled weight decay for trainable LoRA adapter parameters. "
            "Defaults to 0.0 so existing smoke baselines stay unchanged."
        ),
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=DEFAULT_MAX_GRAD_NORM,
        help="Positive global gradient clipping threshold used for the FT trainer.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        help="Positive mini-batches accumulated before each FT optimizer step.",
    )
    parser.add_argument(
        "--ft-epochs",
        type=int,
        default=FT_EPOCHS,
        help="Positive number of sparse retention-guarded FT epochs.",
    )
    parser.add_argument(
        "--target-min-loss-delta",
        type=float,
        default=DEFAULT_TARGET_MIN_LOSS_DELTA,
        help="Non-negative target validation loss improvement required for FT acceptance.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Optional positive early-stopping patience for FT validation loss.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=DEFAULT_MIN_DELTA,
        help="Non-negative validation-loss drop required to reset early-stopping patience.",
    )
    parser.add_argument(
        "--lr-decay-patience",
        type=int,
        default=None,
        help="Optional positive plateau patience before decaying trainer learning rates.",
    )
    parser.add_argument(
        "--lr-decay-factor",
        type=float,
        default=DEFAULT_LR_DECAY_FACTOR,
        help="Learning-rate multiplier used after plateau; must be in (0, 1).",
    )
    parser.add_argument(
        "--lr-decay-min-delta",
        type=float,
        default=DEFAULT_LR_DECAY_MIN_DELTA,
        help="Non-negative validation-loss drop required to reset LR plateau patience.",
    )
    add_summary_compare_args(parser, subject="MLP LoRA adapter run")
    args = parser.parse_args()
    gate_requested = validate_summary_compare_args(parser, args)
    if args.compare_jsonl is None and gate_requested:
        parser.error("regression gate options require --compare-jsonl")
    if args.key_preset == AUTO_KEY_PRESET and args.hf_state_dict is None:
        parser.error("--key-preset auto requires --hf-state-dict")
    if args.checkpoint_source_gain <= 0.0:
        parser.error("--checkpoint-source-gain must be positive")
    if args.adapter_weight_decay < 0.0:
        parser.error("--adapter-weight-decay must be non-negative")
    if args.max_grad_norm <= 0.0:
        parser.error("--max-grad-norm must be positive")
    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient-accumulation-steps must be positive")
    if args.ft_epochs <= 0:
        parser.error("--ft-epochs must be positive")
    for name in ["target_min_loss_delta", "min_delta", "lr_decay_min_delta"]:
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.patience is not None and args.patience <= 0:
        parser.error("--patience must be positive")
    if args.lr_decay_patience is not None and args.lr_decay_patience <= 0:
        parser.error("--lr-decay-patience must be positive")
    if args.lr_decay_factor <= 0.0 or args.lr_decay_factor >= 1.0:
        parser.error("--lr-decay-factor must be in (0, 1)")
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


def scale_state_dict(model, factor):
    scaled = {}
    for name, tensor in model.state_dict().items():
        scaled[name] = st.Tensor(
            tensor.rows,
            tensor.cols,
            [value * factor for value in tensor.data()],
        )
    model.load_state_dict(scaled)


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
            "byte MLP LoRA HF state-dict smoke requires "
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


def build_lora_mlp(source_state, source_key_map, *, emit_preflight=True):
    embed = Linear(VOCAB, HIDDEN, name="embed")
    embed_report, embed_load = preflight_and_load(
        "byte_mlp_lora_embed",
        embed,
        source_state,
        source_key_map,
        emit=emit_preflight,
    )
    embed.set_trainable(False)

    head = LoraLinear(
        HIDDEN,
        VOCAB,
        LORA_RANK,
        alpha=LORA_ALPHA,
        name="head",
    )
    head_report, head_load = preflight_and_load(
        "byte_mlp_lora_head_base",
        head,
        source_state,
        source_key_map,
        lora_base=True,
        emit=emit_preflight,
    )

    return Sequential([embed, Relu(), head]), embed_load, head_load, embed_report, head_report


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
    return source, last_stats, delta


def zero_source_delta():
    return {
        "loss_delta": 0.0,
        "accuracy_delta": 0.0,
        "perplexity_delta": 0.0,
    }


def require_loss_delta(label, delta):
    if delta["loss_delta"] <= 0.0:
        raise RuntimeError(f"{label} loss did not improve: {delta}")


def print_delta(label, delta):
    print(
        f"{label}_delta "
        f"loss_delta={delta['loss_delta']:.6f} "
        f"accuracy_delta={delta['accuracy_delta']:.6f} "
        f"perplexity_delta={delta['perplexity_delta']:.6f}"
    )


def metric(delta, name):
    return delta[name]


def label_number(value):
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def adapter_weight_decay_variant_label(weight_decay):
    if weight_decay == DEFAULT_ADAPTER_WEIGHT_DECAY:
        return None
    return f"wd{label_number(weight_decay)}"


def max_grad_norm_variant_label(max_grad_norm):
    if max_grad_norm == DEFAULT_MAX_GRAD_NORM:
        return None
    return f"gn{label_number(max_grad_norm)}"


def gradient_accumulation_variant_label(steps):
    if steps == DEFAULT_GRADIENT_ACCUMULATION_STEPS:
        return None
    return f"accum{int(steps)}"


def ft_control_variant_label(
    ft_epochs=FT_EPOCHS,
    target_min_loss_delta=DEFAULT_TARGET_MIN_LOSS_DELTA,
    patience=None,
    min_delta=DEFAULT_MIN_DELTA,
    lr_decay_patience=None,
    lr_decay_factor=DEFAULT_LR_DECAY_FACTOR,
    lr_decay_min_delta=DEFAULT_LR_DECAY_MIN_DELTA,
):
    suffixes = []
    if ft_epochs != FT_EPOCHS:
        suffixes.append(f"ep{ft_epochs}")
    if target_min_loss_delta != DEFAULT_TARGET_MIN_LOSS_DELTA:
        suffixes.append(f"tmin{label_number(target_min_loss_delta)}")
    if patience is not None:
        suffixes.append(f"pat{patience}")
    if min_delta != DEFAULT_MIN_DELTA:
        suffixes.append(f"md{label_number(min_delta)}")
    if lr_decay_patience is not None:
        suffixes.append(f"ldp{lr_decay_patience}")
    if lr_decay_factor != DEFAULT_LR_DECAY_FACTOR:
        suffixes.append(f"ldf{label_number(lr_decay_factor)}")
    if lr_decay_min_delta != DEFAULT_LR_DECAY_MIN_DELTA:
        suffixes.append(f"ldmd{label_number(lr_decay_min_delta)}")
    if not suffixes:
        return None
    return "::".join(suffixes)


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


def adapter_config_label(
    adapter_weight_decay,
    max_grad_norm=DEFAULT_MAX_GRAD_NORM,
    gradient_accumulation_steps=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    ft_epochs=FT_EPOCHS,
    target_min_loss_delta=DEFAULT_TARGET_MIN_LOSS_DELTA,
    patience=None,
    min_delta=DEFAULT_MIN_DELTA,
    lr_decay_patience=None,
    lr_decay_factor=DEFAULT_LR_DECAY_FACTOR,
    lr_decay_min_delta=DEFAULT_LR_DECAY_MIN_DELTA,
):
    label = f"r{LORA_RANK}_a{int(LORA_ALPHA)}"
    suffixes = []
    for policy_label in [
        adapter_weight_decay_variant_label(adapter_weight_decay),
        max_grad_norm_variant_label(max_grad_norm),
        gradient_accumulation_variant_label(gradient_accumulation_steps),
    ]:
        if policy_label is not None:
            suffixes.append(policy_label)
    ft_control_label = ft_control_variant_label(
        ft_epochs,
        target_min_loss_delta,
        patience,
        min_delta,
        lr_decay_patience,
        lr_decay_factor,
        lr_decay_min_delta,
    )
    if ft_control_label is not None:
        suffixes.append(ft_control_label)
    if not suffixes:
        return label
    return "::".join([label, *suffixes])


def resolved_checkpoint_source_label(args, source_origin):
    if args.checkpoint_source_label:
        return args.checkpoint_source_label
    if source_origin == "hf_state_dict":
        return args.key_preset
    return source_origin


def summary_row(
    ft_report,
    source_delta,
    source_rows,
    target_rows,
    source_stats,
    frozen_base,
    boosted_adapter,
    decayed_adapter,
    adapter_weight_decay,
    embed_report,
    head_report,
    embed_load,
    head_load,
    key_preset,
    source_origin,
    source_label,
    loaded_files,
    checkpoint_shapes,
    overlap_resize,
    projection_fields,
    source_gain_fields,
    max_grad_norm,
    gradient_accumulation_steps,
    ft_epochs,
    target_min_loss_delta,
    patience,
    min_delta,
    lr_decay_patience,
    lr_decay_factor,
    lr_decay_min_delta,
):
    row = dict(ft_report.summary())
    attach_summary_guard_margins(row)
    attach_summary_guard_counts(row, ft_report.captured)
    row.update(
        {
            "example": "byte_lm_mlp_lora_adapter",
            "config": adapter_config_label(
                adapter_weight_decay,
                max_grad_norm,
                gradient_accumulation_steps,
                ft_epochs,
                target_min_loss_delta,
                patience,
                min_delta,
                lr_decay_patience,
                lr_decay_factor,
                lr_decay_min_delta,
            ),
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "adapter_weight_decay_variant": adapter_weight_decay_variant_label(
                adapter_weight_decay
            ),
            "adapter_weight_decay": adapter_weight_decay,
            "max_grad_norm_variant": max_grad_norm_variant_label(max_grad_norm),
            "max_grad_norm": max_grad_norm,
            "gradient_accumulation_steps_variant": gradient_accumulation_variant_label(
                gradient_accumulation_steps
            ),
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "ft_control_variant": ft_control_variant_label(
                ft_epochs,
                target_min_loss_delta,
                patience,
                min_delta,
                lr_decay_patience,
                lr_decay_factor,
                lr_decay_min_delta,
            ),
            "ft_epochs": ft_epochs,
            "target_min_loss_delta_policy": target_min_loss_delta,
            "early_stopping_patience": patience,
            "early_stopping_min_delta": min_delta,
            "lr_decay_patience": lr_decay_patience,
            "lr_decay_factor": lr_decay_factor,
            "lr_decay_min_delta": lr_decay_min_delta,
            "ft_early_stopped": ft_report.captured.early_stopped,
            "ft_stop_epoch": ft_report.captured.stop_epoch,
            "ft_lr_decay_steps": ft_report.captured.lr_decay_steps,
            "ft_final_hyper_learning_rate": ft_report.captured.final_hyper_learning_rate,
            "ft_final_fallback_learning_rate": ft_report.captured.final_fallback_learning_rate,
            "hidden": HIDDEN,
            "source_rows": source_rows,
            "target_rows": target_rows,
            "source_batches": source_stats.batches,
            "source_optimizer_steps": source_stats.optimizer_steps,
            "source_loss_delta": metric(source_delta, "loss_delta"),
            "source_accuracy_delta": metric(source_delta, "accuracy_delta"),
            "source_perplexity_delta": metric(source_delta, "perplexity_delta"),
            "frozen_base_params": frozen_base,
            "boosted_adapter_params": boosted_adapter,
            "decayed_adapter_params": decayed_adapter,
            "checkpoint_key_preset": key_preset,
            "checkpoint_source_origin": source_origin,
            "checkpoint_source_label": source_label,
            "checkpoint_loaded_files": len(loaded_files),
            "checkpoint_vocab": checkpoint_shapes[0],
            "checkpoint_hidden": checkpoint_shapes[1],
            "checkpoint_target_classes": checkpoint_shapes[2],
            "checkpoint_overlap_resize": overlap_resize,
        }
    )
    row.update(projection_fields)
    row.update(source_gain_fields)
    row.update(checkpoint_audit_fields("embed", embed_report, embed_load))
    row.update(checkpoint_audit_fields("head", head_report, head_load))
    attach_training_policy_key(row)
    return row


def main():
    args = parse_args()
    source_text = (
        "spiraltorch adapters inherit byte memories; adapters inherit byte memories"
    )
    target_text = "螺旋adapterは小さくFTできるbyte"
    source_samples = st.dataset.byte_lm_windows(source_text, CONTEXT)
    target_samples = st.dataset.byte_lm_windows(target_text, CONTEXT)

    session = st.SpiralSession(
        device="wgpu",
        curvature=-1.0,
        hyper_learning_rate=0.5,
        fallback_learning_rate=0.1,
    )
    trainer = session.trainer()
    trainer.set_max_grad_norm(args.max_grad_norm)
    trainer.set_gradient_accumulation_steps(args.gradient_accumulation_steps)
    schedule = trainer.roundtable(rows=CONTEXT * BATCH_WINDOWS, cols=VOCAB)
    loss = SoftmaxCrossEntropy()
    projection_fields = checkpoint_projection_fields(args)
    source_gain_fields = checkpoint_source_gain_fields(args)

    if args.hf_state_dict is None:
        source, source_stats, source_delta = train_source(
            session,
            trainer,
            schedule,
            loss,
            source_samples,
        )
        source_state = source.state_dict()
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

    external_source_state = apply_checkpoint_projection(
        external_source_state,
        source_key_map,
        args,
    )
    external_source_state = apply_checkpoint_source_gain(
        external_source_state,
        source_key_map,
        args,
    )
    target, embed_load, head_load, embed_report, head_report = build_lora_mlp(
        external_source_state,
        source_key_map,
    )
    if not embed_load["matched"] or not head_load["matched"]:
        raise RuntimeError(
            f"MLP LoRA base load mismatch: embed={embed_load} head={head_load}"
        )

    session.prepare_module(target)
    frozen_base = target.set_parameters_trainable_by_suffix("::weight", False)
    frozen_base += target.set_parameters_trainable_by_suffix("::bias", False)
    boosted_adapter = target.set_parameters_learning_rate_scale_by_contains("lora_", 4.0)
    decayed_adapter = target.set_parameters_weight_decay_by_contains(
        "lora_",
        args.adapter_weight_decay,
    )

    ft_ready_state = target.state_dict()
    ft_ready_resume = trainer.resume_fingerprint(target)
    resumed, _, _, _, _ = build_lora_mlp(
        external_source_state,
        source_key_map,
        emit_preflight=False,
    )
    session.prepare_module(resumed)
    resumed.set_parameters_trainable_by_suffix("::weight", False)
    resumed.set_parameters_trainable_by_suffix("::bias", False)
    resumed.set_parameters_learning_rate_scale_by_contains("lora_", 4.0)
    resumed.set_parameters_weight_decay_by_contains("lora_", args.adapter_weight_decay)
    resume_load = resumed.load_state_dict_checked(ft_ready_state)
    resume_check = trainer.resume_fingerprint(resumed)
    if not resume_load["matched"] or resume_check["hash"] != ft_ready_resume["hash"]:
        raise RuntimeError(
            "MLP LoRA FT-ready resume fingerprint mismatch: "
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
        epochs=args.ft_epochs,
        movement_tolerance=FT_MOVEMENT_TOLERANCE,
        max_loss_increase=10.0,
        max_accuracy_drop=1.0,
        max_perplexity_increase=100.0,
        target_min_loss_delta=args.target_min_loss_delta,
        patience=args.patience,
        min_delta=args.min_delta,
        lr_decay_patience=args.lr_decay_patience,
        lr_decay_factor=args.lr_decay_factor,
        lr_decay_min_delta=args.lr_decay_min_delta,
    )
    if ft_report.captured.guarded_best_epoch is None:
        raise RuntimeError("MLP LoRA retention guard rejected every fine-tune epoch")

    ft_delta = ft_report.target_delta
    retention_delta = ft_report.retention_delta
    require_loss_delta("MLP LoRA fine-tune", ft_delta)
    if not ft_report.movement_ok:
        raise RuntimeError(f"unexpected MLP LoRA movement: {ft_report.movement}")

    summary = ft_report.summary()
    guard_counts = attach_summary_guard_counts(dict(summary), ft_report.captured)
    movement = ft_report.movement
    checkpoint_source_label = resolved_checkpoint_source_label(args, source_origin)
    print(
        f"vocab_mode=byte vocab={VOCAB} hidden={HIDDEN} context={CONTEXT} "
        f"adapter_rank={LORA_RANK} adapter_alpha={LORA_ALPHA:.3f} "
        f"adapter_weight_decay={args.adapter_weight_decay:.6f} "
        f"max_grad_norm={args.max_grad_norm:.6f} "
        f"gradient_accumulation_steps={args.gradient_accumulation_steps} "
        f"ft_epochs={args.ft_epochs} "
        f"target_min_loss_delta={args.target_min_loss_delta:.6f} "
        f"patience={args.patience if args.patience is not None else 'none'} "
        f"min_delta={args.min_delta:.6f} "
        f"lr_decay_patience={args.lr_decay_patience if args.lr_decay_patience is not None else 'none'} "
        f"lr_decay_factor={args.lr_decay_factor:.6f} "
        f"lr_decay_min_delta={args.lr_decay_min_delta:.6f} "
        f"checkpoint_key_preset={args.key_preset} "
        f"checkpoint_source_origin={source_origin} "
        f"checkpoint_source_label={checkpoint_source_label} "
        f"checkpoint_loaded_files={len(loaded_files)} "
        f"checkpoint_overlap_resize={overlap_resize} "
        f"checkpoint_projection={projection_fields['checkpoint_projection']} "
        f"checkpoint_projection_strength={projection_value_label(projection_fields['checkpoint_projection_strength'])} "
        f"checkpoint_projection_curvature={projection_value_label(projection_fields['checkpoint_projection_curvature'])} "
        f"checkpoint_projection_frequency={projection_value_label(projection_fields['checkpoint_projection_frequency'])} "
        f"checkpoint_source_gain={source_gain_fields['checkpoint_source_gain']:.6f} "
        f"source_windows={len(source_samples)} target_windows={len(target_samples)}"
    )
    print(
        f"source_batches={source_stats.batches} "
        f"source_optimizer_steps={source_stats.optimizer_steps} "
        f"target_batches={ft_report.captured.train_summary.batches} "
        f"target_optimizer_steps={ft_report.captured.train_summary.optimizer_steps} "
        f"accumulation_steps={args.gradient_accumulation_steps} "
        f"ft_early_stopped={ft_report.captured.early_stopped} "
        f"ft_stop_epoch={ft_report.captured.stop_epoch} "
        f"ft_lr_decay_steps={ft_report.captured.lr_decay_steps} "
        f"ft_final_hyper_lr={ft_report.captured.final_hyper_learning_rate:.6f} "
        f"ft_final_fallback_lr={ft_report.captured.final_fallback_learning_rate:.6f} "
        f"frozen_base_params={frozen_base} "
        f"boosted_adapter_params={boosted_adapter} "
        f"decayed_adapter_params={decayed_adapter}"
    )
    print_delta("source", source_delta)
    print_delta("fine_tune", ft_delta)
    print_delta("retention", retention_delta)
    print(
        "checkpoint "
        f"embed_preflight_matched={embed_report['matched']} "
        f"embed_preflight_extra={embed_report['extra']} "
        f"embed_hash={embed_load['source']['hash']} "
        f"embed_loaded_hash={embed_load['loaded']['hash']} "
        f"embed_load_matched={embed_load['matched']} "
        f"head_preflight_matched={head_report['matched']} "
        f"head_preflight_extra={head_report['extra']} "
        f"head_hash={head_load['source']['hash']} "
        f"head_loaded_hash={head_load['loaded']['hash']} "
        f"head_load_matched={head_load['matched']}"
    )
    print(
        "resume_fingerprint "
        f"hash={summary['resume_hash']} "
        f"trainer_hash={summary['resume_trainer_hash']} "
        f"parameter_training_hash={summary['resume_parameter_training_hash']} "
        f"matched={resume_check['hash'] == ft_ready_resume['hash']}"
    )
    print(
        f"report_status={ft_report.status} "
        f"guard_accepted_epochs={guard_counts['guard_accepted_epochs']} "
        f"guard_retention_rejected_epochs={guard_counts['guard_retention_rejected_epochs']} "
        f"guard_target_stale_epochs={guard_counts['guard_target_stale_epochs']} "
        f"guard_epoch_counts_available={guard_counts['guard_epoch_counts_available']} "
        f"movement_status={movement['status']} "
        f"frozen_stable={movement['frozen_stable']} "
        f"trainable_moved={movement['trainable_movement_observed']} "
        f"trainable_changed={movement['trainable_changed']} "
        f"frozen_changed={movement['frozen_changed']}"
    )
    row = summary_row(
        ft_report,
        source_delta,
        source_rows=len(source_samples) * CONTEXT,
        target_rows=len(target_samples) * CONTEXT,
        source_stats=source_stats,
        frozen_base=frozen_base,
        boosted_adapter=boosted_adapter,
        decayed_adapter=decayed_adapter,
        adapter_weight_decay=args.adapter_weight_decay,
        embed_report=embed_report,
        head_report=head_report,
        embed_load=embed_load,
        head_load=head_load,
        key_preset=args.key_preset,
        source_origin=source_origin,
        source_label=checkpoint_source_label,
        loaded_files=loaded_files,
        checkpoint_shapes=checkpoint_shapes,
        overlap_resize=overlap_resize,
        projection_fields=projection_fields,
        source_gain_fields=source_gain_fields,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ft_epochs=args.ft_epochs,
        target_min_loss_delta=args.target_min_loss_delta,
        patience=args.patience,
        min_delta=args.min_delta,
        lr_decay_patience=args.lr_decay_patience,
        lr_decay_factor=args.lr_decay_factor,
        lr_decay_min_delta=args.lr_decay_min_delta,
    )
    if args.jsonl is not None:
        write_summary_jsonl(args.jsonl, [row])
        print(f"summary_jsonl={args.jsonl} rows=1")
    if args.compare_jsonl is not None:
        compare_single_summary(
            row,
            load_single_summary_jsonl(args.compare_jsonl),
            args,
            failure_prefix="MLP LoRA",
        )


if __name__ == "__main__":
    main()
