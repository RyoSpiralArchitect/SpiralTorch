import argparse
import json
from pathlib import Path

import spiraltorch as st
from spiraltorch.nn import (
    Linear,
    SoftmaxCrossEntropy,
    sparse_classification_delta,
)

from sparse_finetune_compare import (
    add_summary_compare_args,
    attach_summary_guard_counts,
    attach_summary_guard_margins,
    checkpoint_audit_differences,
    checkpoint_audit_failures,
    compare_summaries,
    summary_bool_value,
    summary_guard_margins,
    summary_numeric_value,
    summary_compare_failures,
    validate_summary_compare_args,
)


VOCAB = st.dataset.BYTE_LM_VOCAB
CONTEXT = 4
BATCH_WINDOWS = 2
ACCUMULATION_STEPS = 2
FT_EPOCHS = 1
FT_TARGET_MIN_LOSS_DELTA = 1e-4
FT_MOVEMENT_TOLERANCE = 1e-6
REPLAY_RATIOS = [None, 1, 3]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a tiny tokenizerless byte-LM replay-ratio FT sweep."
    )
    parser.add_argument(
        "--ratio",
        dest="ratios",
        action="append",
        choices=[ratio_label(ratio) for ratio in REPLAY_RATIOS],
        help="Run only this replay ratio. May be repeated.",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Optional path for flat SparseFineTuneReport summary rows.",
    )
    parser.add_argument(
        "--compare-jsonl",
        type=Path,
        default=None,
        help="Optional previous summary JSONL to compare against this run.",
    )
    add_summary_compare_args(parser, subject="ratio")
    parser.add_argument(
        "--require-winner-match",
        action="store_true",
        help="Fail when the winning replay ratio changes versus --compare-jsonl.",
    )
    args = parser.parse_args()
    if args.ratios is not None and all(
        ratio == ratio_label(None) for ratio in args.ratios
    ):
        parser.error("--ratio must include at least one replay ratio")
    gate_requested = validate_summary_compare_args(parser, args) or args.require_winner_match
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


def new_runtime():
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
    return session, trainer, schedule


def train_source(session, trainer, schedule, source_samples):
    loss = SoftmaxCrossEntropy()
    source = Linear(VOCAB, VOCAB, name="py_byte_lm_sweep")
    session.prepare_module(source)

    before = evaluate(session, trainer, source, loss, source_samples, seed=7)
    stats = session.train_epoch(
        trainer,
        source,
        loss,
        loader(source_samples, seed=7),
        schedule,
    )
    after = evaluate(session, trainer, source, loss, source_samples, seed=7)
    delta = sparse_classification_delta(before, after)
    if delta["loss_delta"] <= 0.0:
        raise RuntimeError(f"source pretrain did not improve: {delta}")
    return source.state_dict(), stats, delta


def ratio_label(target_per_replay):
    if target_per_replay is None:
        return "target_only"
    return f"target_per_replay_{target_per_replay}"


def selected_replay_ratios(labels):
    if labels is None:
        return list(REPLAY_RATIOS)

    by_label = {ratio_label(ratio): ratio for ratio in REPLAY_RATIOS}
    selected = []
    for label in labels:
        ratio = by_label[label]
        if ratio not in selected:
            selected.append(ratio)
    return selected


def fine_tune_with_ratio(
    session,
    trainer,
    schedule,
    target_per_replay,
    source_state,
    source_samples,
    target_samples,
):
    loss = SoftmaxCrossEntropy()
    target = Linear(VOCAB, VOCAB, name="py_byte_lm_sweep")
    session.prepare_module(target)
    load = target.load_state_dict_checked(source_state)
    if not load["matched"]:
        raise RuntimeError(f"checkpoint fingerprint mismatch: {load}")

    frozen_weights = target.set_parameters_trainable_by_suffix("::weight", False)
    boosted_bias = target.set_parameters_learning_rate_scale_by_suffix("::bias", 1.1)

    if target_per_replay is None:
        train_samples = target_samples
    else:
        train_samples = st.dataset.interleave_replay_samples(
            target_samples,
            source_samples,
            target_per_replay=target_per_replay,
        )
    train_rows = st.dataset.byte_lm_sample_stats(train_samples)["active_rows"]

    report = session.train_epochs_restore_best_sparse_with_finetune_report(
        trainer,
        target,
        loss,
        loader(train_samples, seed=11),
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

    captured = report.captured
    summary = report.summary()
    ft_delta = report.target_delta
    retention_delta = report.retention_delta
    movement = report.movement
    accepted = summary["accepted"]
    if accepted and (not summary["target_loss_improved"] or not summary["movement_ok"]):
        raise RuntimeError(
            f"{ratio_label(target_per_replay)} accepted an invalid FT state: "
            f"status={summary['status']} ft_delta={ft_delta} movement={movement}"
        )

    return {
        "label": ratio_label(target_per_replay),
        "target_per_replay": target_per_replay,
        "train_samples": len(train_samples),
        "train_rows": train_rows,
        "accepted": accepted,
        "captured": captured,
        "ft": ft_delta,
        "retention": retention_delta,
        "movement": movement,
        "summary": summary,
        "status": summary["status"],
        "frozen_weights": frozen_weights,
        "boosted_bias": boosted_bias,
    }


def metric(delta, name):
    return delta[name]


def print_report(report):
    ratio = "none" if report["target_per_replay"] is None else report["target_per_replay"]
    margins = summary_guard_margins(report["summary"])
    print(
        f"ratio={report['label']} "
        f"target_per_replay={ratio} "
        f"accepted={report['accepted']} "
        f"guarded_best_epoch={report['captured'].guarded_best_epoch} "
        f"train_samples={report['train_samples']} "
        f"train_rows={report['train_rows']} "
        f"ft_loss_delta={metric(report['ft'], 'loss_delta'):.6f} "
        f"ft_accuracy_delta={metric(report['ft'], 'accuracy_delta'):.6f} "
        f"ft_perplexity_delta={metric(report['ft'], 'perplexity_delta'):.6f} "
        f"retention_loss_delta={metric(report['retention'], 'loss_delta'):.6f} "
        f"retention_accuracy_delta={metric(report['retention'], 'accuracy_delta'):.6f} "
        f"retention_perplexity_delta={metric(report['retention'], 'perplexity_delta'):.6f} "
        f"best_retention_loss_increase={report['captured'].best_retention_loss_increase:.6f} "
        f"best_retention_accuracy_drop={report['captured'].best_retention_accuracy_drop:.6f} "
        f"best_retention_perplexity_increase={report['captured'].best_retention_perplexity_increase:.6f} "
        f"target_min_loss_delta={report['captured'].retention_guard['target_min_loss_delta']:.6f} "
        f"target_loss_margin={margins['target_loss_margin']:.6f} "
        f"retention_loss_margin={margins['retention_loss_margin']:.6f} "
        f"retention_accuracy_margin={margins['retention_accuracy_margin']:.6f} "
        f"movement_tolerance={report['summary']['movement_tolerance']:.6f} "
        f"resume_hash={report['summary']['resume_hash']} "
        f"report_status={report['status']} "
        f"summary_optimizer_steps={report['summary']['optimizer_steps']} "
        f"movement_status={report['movement']['status']} "
        f"frozen_weight_params={report['frozen_weights']} "
        f"boosted_bias_params={report['boosted_bias']}"
    )


def summary_row(report, source_delta, source_rows, target_rows):
    row = dict(report["summary"])
    attach_summary_guard_margins(row)
    attach_summary_guard_counts(row, report["captured"])
    row.update(
        {
            "example": "byte_lm_replay_sweep",
            "ratio": report["label"],
            "target_per_replay": report["target_per_replay"],
            "train_samples": report["train_samples"],
            "train_rows_checked": report["train_rows"],
            "source_rows": source_rows,
            "target_rows": target_rows,
            "target_min_loss_delta": report["summary"]["target_min_loss_delta"],
            "movement_tolerance": report["summary"]["movement_tolerance"],
            "source_loss_delta": metric(source_delta, "loss_delta"),
            "source_accuracy_delta": metric(source_delta, "accuracy_delta"),
            "source_perplexity_delta": metric(source_delta, "perplexity_delta"),
            "frozen_weight_params": report["frozen_weights"],
            "boosted_bias_params": report["boosted_bias"],
        }
    )
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
            if "ratio" not in row:
                raise ValueError(f"{path}:{line_no} missing 'ratio'")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} did not contain any summary rows")
    return rows


def rows_by_ratio(rows, label):
    by_ratio = {}
    for row in rows:
        ratio = row.get("ratio")
        if not isinstance(ratio, str) or not ratio:
            raise ValueError(f"{label} row has invalid ratio: {ratio!r}")
        if ratio in by_ratio:
            raise ValueError(f"{label} contains duplicate ratio: {ratio}")
        by_ratio[ratio] = row
    return by_ratio


def replay_winner(rows):
    candidates = [
        row
        for row in rows
        if row.get("target_per_replay") is not None
        and summary_bool_value(row, "accepted", False)
        and summary_numeric_value(row, "target_loss_delta") > 0.0
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            summary_numeric_value(row, "target_loss_delta"),
            summary_numeric_value(row, "retention_loss_delta"),
            summary_numeric_value(row, "retention_accuracy_delta"),
        ),
    )


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
    allow_missing_current,
):
    current = rows_by_ratio(current_rows, "current")
    baseline = rows_by_ratio(baseline_rows, "baseline")
    missing_current = sorted(set(baseline) - set(current))
    if missing_current:
        for ratio in missing_current:
            print(f"summary_compare ratio={ratio} current_missing=true")
        if not allow_missing_current:
            raise RuntimeError(
                "baseline ratios missing from current sweep: " + ",".join(missing_current)
            )

    common = sorted(set(current) & set(baseline))
    if not common:
        raise RuntimeError("no overlapping ratios between current sweep and baseline")

    failures = []
    for ratio in common:
        now = current[ratio]
        before = baseline[ratio]
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
            f"summary_compare ratio={ratio} "
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
                ratio,
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
            failures.extend(checkpoint_audit_failures(ratio, now, before))

    baseline_winner = replay_winner(baseline_rows)
    current_winner = replay_winner(current_rows)
    if baseline_winner is not None and current_winner is not None:
        winner_changed = baseline_winner["ratio"] != current_winner["ratio"]
        print(
            f"winner_compare before={baseline_winner['ratio']} "
            f"after={current_winner['ratio']} "
            f"target_loss_delta_before={summary_numeric_value(baseline_winner, 'target_loss_delta'):.9f} "
            f"target_loss_delta_after={summary_numeric_value(current_winner, 'target_loss_delta'):.9f} "
            f"retention_loss_delta_before={summary_numeric_value(baseline_winner, 'retention_loss_delta'):.9f} "
            f"retention_loss_delta_after={summary_numeric_value(current_winner, 'retention_loss_delta'):.9f} "
            f"winner_changed={winner_changed} "
            f"passed={not (winner_changed and require_winner_match)}"
        )
        if winner_changed and require_winner_match:
            failures.append(
                f"winner changed from {baseline_winner['ratio']} "
                f"to {current_winner['ratio']}"
            )
    elif require_winner_match:
        failures.append("winner could not be selected for current or baseline rows")

    extra_current = sorted(set(current) - set(baseline))
    for ratio in extra_current:
        print(f"summary_compare ratio={ratio} baseline_missing=true")
    if failures:
        raise RuntimeError("replay sweep regression gate failed: " + "; ".join(failures))
    return len(common)


def main():
    args = parse_args()
    selected_ratios = selected_replay_ratios(args.ratios)
    source_docs = [
        "source byte anchor",
        "retain old bytes",
    ]
    target_docs = [
        "螺旋byte",
        "猫byte",
    ]
    source_samples = st.dataset.byte_lm_corpus_windows(source_docs, CONTEXT)
    target_samples = st.dataset.byte_lm_corpus_windows(target_docs, CONTEXT)
    source_rows = st.dataset.byte_lm_sample_stats(source_samples)["active_rows"]
    target_rows = st.dataset.byte_lm_sample_stats(target_samples)["active_rows"]

    session, trainer, schedule = new_runtime()
    source_state, source_stats, source_delta = train_source(
        session,
        trainer,
        schedule,
        source_samples,
    )
    reports = [
        fine_tune_with_ratio(
            session,
            trainer,
            schedule,
            ratio,
            source_state,
            source_samples,
            target_samples,
        )
        for ratio in selected_ratios
    ]

    accepted_replay = [
        report
        for report in reports
        if report["target_per_replay"] is not None
        and report["accepted"]
        and report["ft"]["loss_delta"] > 0.0
    ]
    if not accepted_replay:
        raise RuntimeError("no replay ratio produced an accepted FT improvement")

    winner = max(
        accepted_replay,
        key=lambda report: (
            metric(report["ft"], "loss_delta"),
            metric(report["retention"], "loss_delta"),
            metric(report["retention"], "accuracy_delta"),
        ),
    )

    print(
        f"sweep=python_byte_lm_replay "
        f"vocab={VOCAB} context={CONTEXT} "
        f"source_docs={len(source_docs)} target_docs={len(target_docs)} "
        f"source_windows={len(source_samples)} target_windows={len(target_samples)} "
        f"source_rows={source_rows} target_rows={target_rows} "
        f"source_batches={source_stats.batches} "
        f"source_optimizer_steps={source_stats.optimizer_steps} "
        f"accumulation_steps={ACCUMULATION_STEPS} "
        f"ratios={len(selected_ratios)}"
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
        summary_row(report, source_delta, source_rows, target_rows)
        for report in reports
    ]
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
            args.ratios is not None,
        )
        print(f"summary_compare_rows={compared} baseline={args.compare_jsonl}")
    print(
        f"replay_winner ratio={winner['label']} "
        f"ft_loss_delta={metric(winner['ft'], 'loss_delta'):.6f} "
        f"retention_loss_delta={metric(winner['retention'], 'loss_delta'):.6f} "
        f"retention_accuracy_delta={metric(winner['retention'], 'accuracy_delta'):.6f}"
    )


if __name__ == "__main__":
    main()
