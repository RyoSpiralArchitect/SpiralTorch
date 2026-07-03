import spiraltorch as st
from spiraltorch.nn import Linear, SoftmaxCrossEntropy, sparse_classification_delta


VOCAB = st.dataset.BYTE_LM_VOCAB
CONTEXT = 8
BATCH_WINDOWS = 4
ACCUMULATION_STEPS = 2
FT_EPOCHS = 3
FT_TARGET_MIN_LOSS_DELTA = 1e-4
FT_MOVEMENT_TOLERANCE = 1e-6


def loader(samples, seed):
    return st.dataset.from_vec(samples).shuffle(seed).batched(BATCH_WINDOWS)


def evaluate(session, trainer, model, loss, samples, seed):
    return session.evaluate_sparse_classification_epoch(
        trainer,
        model,
        loss,
        loader(samples, seed),
    )


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


def main():
    source_docs = [
        "spiraltorch learns from source bytes",
        "byte windows keep tokenizerless FT honest",
        "",
    ]
    target_docs = [
        "螺旋はbyteで壊れない",
        "猫byteはsourceを少し覚える",
    ]
    source_samples = st.dataset.byte_lm_corpus_windows(source_docs, CONTEXT)
    target_samples = st.dataset.byte_lm_corpus_windows(target_docs, CONTEXT)
    source_rows = st.dataset.byte_lm_sample_stats(source_samples)["active_rows"]
    target_rows = st.dataset.byte_lm_sample_stats(target_samples)["active_rows"]

    session = st.SpiralSession(
        device="wgpu",
        curvature=-1.0,
        hyper_learning_rate=0.05,
        fallback_learning_rate=0.01,
    )
    trainer = session.trainer()
    trainer.set_max_grad_norm(0.25)
    trainer.set_gradient_accumulation_steps(ACCUMULATION_STEPS)
    schedule = trainer.roundtable(rows=CONTEXT * BATCH_WINDOWS, cols=VOCAB)
    loss = SoftmaxCrossEntropy()

    source = Linear(VOCAB, VOCAB, name="py_byte_lm")
    session.prepare_module(source)
    source_before = evaluate(session, trainer, source, loss, source_samples, seed=7)
    source_stats = session.train_epoch(
        trainer,
        source,
        loss,
        loader(source_samples, seed=7),
        schedule,
    )
    source_after = evaluate(session, trainer, source, loss, source_samples, seed=7)
    source_delta = sparse_classification_delta(source_before, source_after)
    require_loss_delta("source", source_delta)

    target = Linear(VOCAB, VOCAB, name="py_byte_lm")
    session.prepare_module(target)
    load = target.load_state_dict_checked(source.state_dict())
    if not load["matched"]:
        raise RuntimeError(f"checkpoint fingerprint mismatch: {load}")

    frozen_weights = target.set_parameters_trainable_by_suffix("::weight", False)
    boosted_bias = target.set_parameters_learning_rate_scale_by_suffix("::bias", 1.1)
    ft_ready_state = target.state_dict()
    ft_ready_resume = trainer.resume_fingerprint(target)
    resumed = Linear(VOCAB, VOCAB, name="py_byte_lm")
    session.prepare_module(resumed)
    resume_load = resumed.load_state_dict_checked(ft_ready_state)
    resumed.set_parameters_trainable_by_suffix("::weight", False)
    resumed.set_parameters_learning_rate_scale_by_suffix("::bias", 1.1)
    resume_check = trainer.resume_fingerprint(resumed)
    if not resume_load["matched"] or resume_check["hash"] != ft_ready_resume["hash"]:
        raise RuntimeError(
            "FT-ready resume fingerprint mismatch: "
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
        max_loss_increase=0.5,
        max_accuracy_drop=0.15,
        max_perplexity_increase=1.0,
        target_min_loss_delta=FT_TARGET_MIN_LOSS_DELTA,
    )
    ft_capture = ft_report.captured
    if ft_capture.guarded_best_epoch is None:
        raise RuntimeError("retention guard rejected every fine-tune epoch")
    ft_delta = ft_report.target_delta
    retention_delta = ft_report.retention_delta
    ft_summary = ft_report.summary()
    require_loss_delta("fine_tune", ft_delta)

    movement = ft_report.movement
    if not ft_report.movement_ok:
        raise RuntimeError(f"unexpected parameter movement: {movement}")

    print(
        f"vocab_mode=byte vocab={VOCAB} context={CONTEXT} "
        f"source_docs={len(source_docs)} target_docs={len(target_docs)} "
        f"source_windows={len(source_samples)} target_windows={len(target_samples)} "
        f"source_rows={source_rows} target_rows={target_rows}"
    )
    print(
        "resume_fingerprint "
        f"hash={ft_ready_resume['hash']} "
        f"trainer_hash={ft_ready_resume['trainer']['hash']} "
        f"parameter_training_hash={ft_ready_resume['parameter_training']['hash']} "
        f"matched={resume_check['hash'] == ft_ready_resume['hash']}"
    )
    print(
        f"source_batches={source_stats.batches} source_optimizer_steps={source_stats.optimizer_steps} "
        f"target_batches={ft_capture.train_summary.batches} "
        f"target_optimizer_steps={ft_capture.train_summary.optimizer_steps} "
        f"accumulation_steps={ACCUMULATION_STEPS} frozen_weight_params={frozen_weights} "
        f"boosted_bias_params={boosted_bias}"
    )
    print_delta("source", source_delta)
    print_delta("fine_tune", ft_delta)
    print_delta("retention", retention_delta)
    print(
        "retention_guard "
        f"guarded_best_epoch={ft_capture.guarded_best_epoch} "
        f"max_allowed_loss={ft_capture.max_allowed_retention_loss:.6f} "
        f"min_allowed_accuracy={ft_capture.min_allowed_retention_accuracy:.6f} "
        f"max_allowed_perplexity={ft_capture.max_allowed_retention_perplexity:.6f} "
        f"target_min_loss_delta={ft_capture.retention_guard['target_min_loss_delta']:.6f} "
        f"movement_tolerance={ft_summary['movement_tolerance']:.6f} "
        f"best_retention_loss_increase={ft_capture.best_retention_loss_increase:.6f} "
        f"best_retention_accuracy_drop={ft_capture.best_retention_accuracy_drop:.6f} "
        f"best_retention_perplexity_increase={ft_capture.best_retention_perplexity_increase:.6f}"
    )
    print(
        f"report_status={ft_report.status} "
        f"movement_status={movement['status']} "
        f"frozen_stable={movement['frozen_stable']} "
        f"trainable_moved={movement['trainable_movement_observed']} "
        f"summary_status={ft_summary['status']}"
    )


if __name__ == "__main__":
    main()
