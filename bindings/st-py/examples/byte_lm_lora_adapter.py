import argparse
from pathlib import Path

import spiraltorch as st
from spiraltorch.nn import (
    Linear,
    LoraLinear,
    SoftmaxCrossEntropy,
    sparse_classification_delta,
)

from checkpoint_preflight import (
    HF_KEY_PRESETS,
    checkpoint_audit_fields,
    hf_lm_handoff_from_spiraltorch_state,
    preflight_and_load,
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
CONTEXT = 1
BATCH_WINDOWS = 2
ACCUMULATION_STEPS = 2
SOURCE_EPOCHS = 2
FT_EPOCHS = 6
LORA_RANK = 4
LORA_ALPHA = 16.0
FT_MOVEMENT_TOLERANCE = 1e-6


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a tiny tokenizerless byte-LM LoRA adapter FT smoke."
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
        choices=sorted(HF_KEY_PRESETS),
        default="gpt2",
        help="HF-style checkpoint key preset for the dense-head handoff.",
    )
    add_summary_compare_args(parser, subject="LoRA adapter run")
    args = parser.parse_args()
    gate_requested = validate_summary_compare_args(parser, args)
    if args.compare_jsonl is None and gate_requested:
        parser.error("regression gate options require --compare-jsonl")
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


def train_dense_source(session, trainer, schedule, loss, source_samples):
    source = Linear(VOCAB, VOCAB, name="py_lora_byte_lm")
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
        raise RuntimeError(f"dense source did not improve: {delta}")
    return source, last_stats, delta


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


def summary_row(
    ft_report,
    source_delta,
    source_rows,
    target_rows,
    source_stats,
    base_report,
    base_load,
    key_preset,
):
    row = dict(ft_report.summary())
    attach_summary_guard_margins(row)
    attach_summary_guard_counts(row, ft_report.captured)
    row.update(
        {
            "example": "byte_lm_lora_adapter",
            "config": f"r{LORA_RANK}_a{int(LORA_ALPHA)}",
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "source_rows": source_rows,
            "target_rows": target_rows,
            "source_batches": source_stats.batches,
            "source_optimizer_steps": source_stats.optimizer_steps,
            "source_loss_delta": metric(source_delta, "loss_delta"),
            "source_accuracy_delta": metric(source_delta, "accuracy_delta"),
            "source_perplexity_delta": metric(source_delta, "perplexity_delta"),
            "checkpoint_key_preset": key_preset,
        }
    )
    row.update(checkpoint_audit_fields("base", base_report, base_load))
    return row


def externalize_dense_head_state(source_state, *, key_preset):
    return hf_lm_handoff_from_spiraltorch_state(
        source_state,
        key_preset=key_preset,
        embed_weight_key=None,
        embed_bias_key=None,
        lm_head_weight_source="py_lora_byte_lm::weight",
        lm_head_bias_source="py_lora_byte_lm::bias",
        lm_head_weight_target="py_lora_byte_lm::weight",
        lm_head_bias_target="py_lora_byte_lm::bias",
        lm_head_bias_transform="identity",
    )


def main():
    args = parse_args()
    source_text = "adapter source source source"
    target_text = "螺旋adapter螺旋"
    source_samples = st.dataset.byte_lm_windows(source_text, CONTEXT)
    target_samples = st.dataset.byte_lm_windows(target_text, CONTEXT)

    session = st.SpiralSession(
        device="wgpu",
        curvature=-1.0,
        hyper_learning_rate=0.2,
        fallback_learning_rate=0.05,
    )
    trainer = session.trainer()
    trainer.set_max_grad_norm(1.0)
    trainer.set_gradient_accumulation_steps(ACCUMULATION_STEPS)
    schedule = trainer.roundtable(rows=CONTEXT * BATCH_WINDOWS, cols=VOCAB)
    loss = SoftmaxCrossEntropy()

    source, source_stats, source_delta = train_dense_source(
        session,
        trainer,
        schedule,
        loss,
        source_samples,
    )
    source_state = source.state_dict()
    external_source_state, source_key_map = externalize_dense_head_state(
        source_state,
        key_preset=args.key_preset,
    )

    lora = LoraLinear(
        VOCAB,
        VOCAB,
        LORA_RANK,
        alpha=LORA_ALPHA,
        name="py_lora_byte_lm",
    )
    base_report, base_load = preflight_and_load(
        "byte_lm_lora_base",
        lora,
        external_source_state,
        source_key_map,
        lora_base=True,
    )

    target = lora
    session.prepare_module(target)
    ft_ready_state = target.state_dict()
    ft_ready_resume = trainer.resume_fingerprint(target)

    resumed_lora = LoraLinear(
        VOCAB,
        VOCAB,
        LORA_RANK,
        alpha=LORA_ALPHA,
        name="py_lora_byte_lm",
    )
    resumed = resumed_lora
    session.prepare_module(resumed)
    resume_load = resumed.load_state_dict_checked(ft_ready_state)
    resume_check = trainer.resume_fingerprint(resumed)
    if not resume_load["matched"] or resume_check["hash"] != ft_ready_resume["hash"]:
        raise RuntimeError(
            "LoRA FT-ready resume fingerprint mismatch: "
            f"load={resume_load} expected={ft_ready_resume} actual={resume_check}"
        )

    ft_train_samples = st.dataset.interleave_replay_samples(
        target_samples,
        source_samples,
        target_per_replay=1,
    )
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
        max_loss_increase=10.0,
        max_accuracy_drop=1.0,
        max_perplexity_increase=100.0,
        target_min_loss_delta=0.0,
    )
    if ft_report.captured.guarded_best_epoch is None:
        raise RuntimeError("LoRA retention guard rejected every fine-tune epoch")

    ft_delta = ft_report.target_delta
    retention_delta = ft_report.retention_delta
    require_loss_delta("LoRA fine-tune", ft_delta)
    if not ft_report.movement_ok:
        raise RuntimeError(f"unexpected LoRA movement: {ft_report.movement}")

    summary = ft_report.summary()
    movement = ft_report.movement
    print(
        f"vocab_mode=byte vocab={VOCAB} context={CONTEXT} "
        f"adapter_rank={LORA_RANK} adapter_alpha={LORA_ALPHA:.3f} "
        f"checkpoint_key_preset={args.key_preset} "
        f"source_windows={len(source_samples)} target_windows={len(target_samples)}"
    )
    print(
        f"source_batches={source_stats.batches} "
        f"source_optimizer_steps={source_stats.optimizer_steps} "
        f"target_batches={ft_report.captured.train_summary.batches} "
        f"target_optimizer_steps={ft_report.captured.train_summary.optimizer_steps} "
        f"accumulation_steps={ACCUMULATION_STEPS}"
    )
    print_delta("source", source_delta)
    print_delta("fine_tune", ft_delta)
    print_delta("retention", retention_delta)
    print(
        "checkpoint "
        f"base_preflight_matched={base_report['matched']} "
        f"base_preflight_extra={base_report['extra']} "
        f"base_source_hash={base_load['source']['hash']} "
        f"base_loaded_hash={base_load['loaded']['hash']} "
        f"base_load_matched={base_load['matched']}"
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
        base_report=base_report,
        base_load=base_load,
        key_preset=args.key_preset,
    )
    if args.jsonl is not None:
        write_summary_jsonl(args.jsonl, [row])
        print(f"summary_jsonl={args.jsonl} rows=1")
    if args.compare_jsonl is not None:
        compare_single_summary(
            row,
            load_single_summary_jsonl(args.compare_jsonl),
            args,
            failure_prefix="LoRA adapter",
        )


if __name__ == "__main__":
    main()
