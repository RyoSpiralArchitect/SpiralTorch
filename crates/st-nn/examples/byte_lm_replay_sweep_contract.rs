// ============================================================================
//  SpiralReality Proprietary
//  Copyright (c) 2025 SpiralReality. All Rights Reserved.
//
//  NOTICE: This file contains confidential and proprietary information of
//  SpiralReality. ANY USE, COPYING, MODIFICATION, DISTRIBUTION, DISPLAY,
//  OR DISCLOSURE OF THIS FILE, IN WHOLE OR IN PART, IS STRICTLY PROHIBITED
//  WITHOUT THE PRIOR WRITTEN CONSENT OF SPIRALREALITY.
//
//  NO LICENSE IS GRANTED OR IMPLIED BY THIS FILE. THIS SOFTWARE IS PROVIDED
//  "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
//  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
//  PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL SPIRALREALITY OR ITS
//  SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
//  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
//  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// ============================================================================

//! Tokenizerless byte-LM replay-ratio sweep contract.
//!
//! The Python facade has a replay sweep that compares target-only fine-tuning
//! with deterministic source replay ratios. This Rust-native contract keeps the
//! same invariant close to the backend: every ratio starts from the same checked
//! source state, frozen weights must stay stable, accepted replay ratios must
//! move trainable bias parameters, and the best replay ratio must beat
//! target-only on both target and source-retention loss deltas under
//! `SparseFineTuneReportSummary::compare_to`.

use st_core::backend::device_caps::DeviceCaps;
use st_nn::{
    byte_lm_corpus_windows, byte_lm_sample_stats, dataset_from_vec, interleave_replay_samples,
    EpochStats, Linear, Module, ModuleTrainer, RoundtableConfig, SoftmaxCrossEntropy,
    SparseClassificationDelta, SparseClassificationMetrics, SparseFineTuneRegressionLimits,
    SparseFineTuneReport, SparseFineTuneReportSummary, SparseRetentionGuardConfig, Tensor,
    BYTE_LM_VOCAB,
};
use st_tensor::pure::{PureResult, TensorError};

const VOCAB: usize = BYTE_LM_VOCAB;
const CONTEXT: usize = 4;
const BATCH_WINDOWS: usize = 2;
const ACCUMULATION_STEPS: usize = 2;
const FT_EPOCHS: usize = 1;
const FT_TARGET_MIN_LOSS_DELTA: f32 = 1e-4;
const FT_MOVEMENT_TOLERANCE: f32 = 1e-6;
const REPLAY_RATIOS: [Option<usize>; 3] = [None, Some(1), Some(3)];

#[derive(Debug, Clone)]
struct ReplayReport {
    label: &'static str,
    target_per_replay: Option<usize>,
    train_samples: usize,
    train_rows: usize,
    summary: SparseFineTuneReportSummary,
    target: SparseClassificationDelta,
    retention: SparseClassificationDelta,
}

fn zero_parameters<M: Module>(model: &mut M) -> PureResult<()> {
    model.visit_parameters_mut(&mut |param| {
        for value in param.value_mut().data_mut() {
            *value = 0.0;
        }
        Ok(())
    })
}

fn replay_label(target_per_replay: Option<usize>) -> &'static str {
    match target_per_replay {
        None => "target_only",
        Some(1) => "target_per_replay_1",
        Some(3) => "target_per_replay_3",
        Some(_) => "target_per_replay_custom",
    }
}

fn new_trainer() -> PureResult<ModuleTrainer> {
    let caps = DeviceCaps::wgpu(32, true, 256);
    let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
    trainer.set_max_grad_norm(Some(0.25))?;
    trainer.set_gradient_accumulation_steps(ACCUMULATION_STEPS)?;
    Ok(trainer)
}

fn train_once(
    trainer: &mut ModuleTrainer,
    model: &mut Linear,
    samples: &[(Tensor, Tensor)],
) -> PureResult<EpochStats> {
    let token_rows = (CONTEXT * BATCH_WINDOWS) as u32;
    let schedule = trainer.roundtable(token_rows, VOCAB as u32, RoundtableConfig::default());
    let loader = dataset_from_vec(samples.to_vec())
        .shuffle(7)
        .batched(BATCH_WINDOWS);
    let mut loss = SoftmaxCrossEntropy::new();
    trainer.train_epoch(model, &mut loss, loader, &schedule)
}

fn evaluate_metrics(
    trainer: &ModuleTrainer,
    model: &Linear,
    samples: &[(Tensor, Tensor)],
    seed: u64,
) -> PureResult<SparseClassificationMetrics> {
    let loss = SoftmaxCrossEntropy::new();
    let metrics = trainer.evaluate_sparse_classification_epoch(
        model,
        &loss,
        dataset_from_vec(samples.to_vec())
            .shuffle(seed)
            .batched(BATCH_WINDOWS),
    )?;
    if metrics.active_rows == 0 {
        return Err(contract_error("byte LM replay evaluation received no rows"));
    }
    Ok(metrics)
}

fn train_source(
    trainer: &mut ModuleTrainer,
    source_samples: &[(Tensor, Tensor)],
) -> PureResult<(Linear, EpochStats, SparseClassificationDelta)> {
    let mut source = Linear::new("byte_lm_replay", VOCAB, VOCAB)?;
    zero_parameters(&mut source)?;
    trainer.prepare(&mut source)?;

    let before = evaluate_metrics(trainer, &source, source_samples, 7)?;
    let stats = train_once(trainer, &mut source, source_samples)?;
    let after = evaluate_metrics(trainer, &source, source_samples, 7)?;
    let delta = before.delta_to(after);
    if delta.loss_delta <= 0.0 {
        return Err(contract_error(format!(
            "source pretrain did not improve: loss_delta={:.6}",
            delta.loss_delta
        )));
    }
    Ok((source, stats, delta))
}

fn train_with_finetune_report(
    trainer: &mut ModuleTrainer,
    model: &mut Linear,
    train_samples: Vec<(Tensor, Tensor)>,
    validation_samples: &[(Tensor, Tensor)],
    retention_samples: &[(Tensor, Tensor)],
) -> PureResult<SparseFineTuneReport> {
    let token_rows = (CONTEXT * BATCH_WINDOWS) as u32;
    let schedule = trainer.roundtable(token_rows, VOCAB as u32, RoundtableConfig::default());
    let train_loader = dataset_from_vec(train_samples)
        .shuffle(11)
        .batched(BATCH_WINDOWS);
    let validation_loader = dataset_from_vec(validation_samples.to_vec())
        .shuffle(17)
        .batched(BATCH_WINDOWS);
    let retention_loader = dataset_from_vec(retention_samples.to_vec())
        .shuffle(13)
        .batched(BATCH_WINDOWS);
    let mut loss = SoftmaxCrossEntropy::new();
    trainer.train_epochs_restore_best_sparse_with_finetune_report(
        model,
        &mut loss,
        train_loader,
        validation_loader,
        retention_loader,
        &schedule,
        FT_EPOCHS,
        SparseRetentionGuardConfig::new(0.5, 0.15)?
            .with_max_perplexity_increase(1.0)?
            .with_target_min_loss_delta(FT_TARGET_MIN_LOSS_DELTA)?,
        FT_MOVEMENT_TOLERANCE,
    )
}

fn fine_tune_with_ratio(
    trainer: &mut ModuleTrainer,
    target_per_replay: Option<usize>,
    source_state: &std::collections::HashMap<String, Tensor>,
    source_samples: &[(Tensor, Tensor)],
    target_samples: &[(Tensor, Tensor)],
) -> PureResult<ReplayReport> {
    let mut target = Linear::new("byte_lm_replay", VOCAB, VOCAB)?;
    zero_parameters(&mut target)?;
    trainer.prepare(&mut target)?;
    let load = target.load_state_dict_checked(source_state)?;
    if !load.matched {
        return Err(contract_error("replay source checkpoint mismatch"));
    }

    let frozen_weights = target.set_parameters_trainable_by_suffix("::weight", false)?;
    let boosted_bias = target.set_parameters_learning_rate_scale_by_suffix("::bias", 1.1)?;
    if frozen_weights == 0 || boosted_bias == 0 {
        return Err(contract_error(
            "replay trainability controls matched no parameters",
        ));
    }

    let train_samples = match target_per_replay {
        None => target_samples.to_vec(),
        Some(ratio) => interleave_replay_samples(target_samples, source_samples, ratio)?,
    };
    let train_rows = byte_lm_sample_stats(&train_samples, None).active_rows;
    let train_len = train_samples.len();
    let report = train_with_finetune_report(
        trainer,
        &mut target,
        train_samples,
        target_samples,
        source_samples,
    )?;
    let summary = report.summary();
    if summary.accepted && (!summary.target_loss_improved || !summary.movement_ok) {
        return Err(contract_error(format!(
            "{} accepted an invalid FT state: status={}",
            replay_label(target_per_replay),
            summary.status
        )));
    }
    if !report.movement.frozen_stable() {
        return Err(contract_error(format!(
            "{} moved frozen weights",
            replay_label(target_per_replay)
        )));
    }

    Ok(ReplayReport {
        label: replay_label(target_per_replay),
        target_per_replay,
        train_samples: train_len,
        train_rows,
        summary,
        target: report.target_delta,
        retention: report.retention_delta,
    })
}

fn replay_winner(reports: &[ReplayReport]) -> Option<&ReplayReport> {
    reports
        .iter()
        .filter(|report| {
            report.target_per_replay.is_some()
                && report.summary.accepted
                && report.summary.target_loss_delta > 0.0
        })
        .max_by(|left, right| {
            replay_score(left)
                .partial_cmp(&replay_score(right))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

fn replay_score(report: &ReplayReport) -> (f32, f32, f32) {
    (
        report.summary.target_loss_delta,
        report.summary.retention_loss_delta,
        report.summary.retention_accuracy_delta,
    )
}

fn contract_error(message: impl Into<String>) -> TensorError {
    TensorError::IoError {
        message: message.into(),
    }
}

fn main() -> PureResult<()> {
    let source_docs = ["source byte anchor", "retain old bytes"];
    let target_docs = ["螺旋byte", "猫byte"];
    let source_samples = byte_lm_corpus_windows(&source_docs, CONTEXT)?;
    let target_samples = byte_lm_corpus_windows(&target_docs, CONTEXT)?;
    let source_rows = byte_lm_sample_stats(&source_samples, None).active_rows;
    let target_rows = byte_lm_sample_stats(&target_samples, None).active_rows;

    let mut trainer = new_trainer()?;
    let (source, source_stats, source_delta) = train_source(&mut trainer, &source_samples)?;
    let source_state = source.state_dict()?;

    let mut reports = Vec::new();
    for ratio in REPLAY_RATIOS {
        reports.push(fine_tune_with_ratio(
            &mut trainer,
            ratio,
            &source_state,
            &source_samples,
            &target_samples,
        )?);
    }

    let target_only = reports
        .iter()
        .find(|report| report.target_per_replay.is_none())
        .ok_or_else(|| contract_error("missing target-only replay baseline"))?;
    let winner = replay_winner(&reports)
        .ok_or_else(|| contract_error("no replay ratio produced an accepted FT improvement"))?;
    let compare_limits = SparseFineTuneRegressionLimits::new()
        .with_max_target_loss_regression(0.0)?
        .with_max_retention_loss_regression(0.0)?
        .with_guard_match_required(true)
        .with_movement_tolerance_match_required(true)
        .with_resume_match_required(true);
    let winner_compare = winner
        .summary
        .compare_to(&target_only.summary, compare_limits)?;
    if !winner_compare.passed {
        return Err(contract_error(format!(
            "replay winner did not beat target-only without regression: winner={} target_regression={:.6} retention_regression={:.6} accepted_changed={}",
            winner.label,
            winner_compare.target_loss_regression,
            winner_compare.retention_loss_regression,
            winner_compare.accepted_changed
        )));
    }

    println!(
        "sweep=rust_byte_lm_replay vocab={VOCAB} context={CONTEXT} source_docs={} target_docs={} source_windows={} target_windows={} source_rows={} target_rows={} source_batches={} source_optimizer_steps={} accumulation_steps={ACCUMULATION_STEPS} ratios={}",
        source_docs.len(),
        target_docs.len(),
        source_samples.len(),
        target_samples.len(),
        source_rows,
        target_rows,
        source_stats.batches,
        source_stats.optimizer_steps,
        reports.len()
    );
    println!(
        "source_delta loss_delta={:.6} accuracy_delta={:.6} perplexity_delta={:.6}",
        source_delta.loss_delta, source_delta.accuracy_delta, source_delta.perplexity_delta
    );
    for report in &reports {
        let ratio = report
            .target_per_replay
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string());
        let target_retention_ratio = report
            .summary
            .target_retention_ratio
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "none".to_string());
        println!(
            "ratio={} target_per_replay={} accepted={} guarded_best_epoch={:?} train_samples={} train_rows={} ft_loss_delta={:.6} ft_accuracy_delta={:.6} ft_perplexity_delta={:.6} retention_loss_delta={:.6} retention_accuracy_delta={:.6} retention_perplexity_delta={:.6} target_retention_gap={:.6} target_retention_ratio={} target_min_loss_delta={:.6} target_loss_margin={:.6} retention_loss_margin={:.6} retention_accuracy_margin={:.6} movement_tolerance={:.6} resume_hash={} report_status={} summary_optimizer_steps={} movement_status={}",
            report.label,
            ratio,
            report.summary.accepted,
            report.summary.guarded_best_epoch,
            report.train_samples,
            report.train_rows,
            report.target.loss_delta,
            report.target.accuracy_delta,
            report.target.perplexity_delta,
            report.retention.loss_delta,
            report.retention.accuracy_delta,
            report.retention.perplexity_delta,
            report.summary.target_retention_gap,
            target_retention_ratio,
            report.summary.target_min_loss_delta,
            report.summary.target_loss_margin,
            report.summary.retention_loss_margin,
            report.summary.retention_accuracy_margin,
            report.summary.movement_tolerance,
            report.summary.resume_hash,
            report.summary.status,
            report.summary.optimizer_steps,
            report.summary.movement_status
        );
    }
    println!(
        "replay_summary_compare winner={} baseline=target_only target_loss_delta_change={:.6} retention_loss_delta_change={:.6} target_loss_regression={:.6} retention_loss_regression={:.6} status_changed={} accepted_changed={} guard_changed={} movement_tolerance_changed={} resume_changed={} passed={}",
        winner.label,
        winner_compare.target_loss_delta_change,
        winner_compare.retention_loss_delta_change,
        winner_compare.target_loss_regression,
        winner_compare.retention_loss_regression,
        winner_compare.status_changed,
        winner_compare.accepted_changed,
        winner_compare.guard_changed,
        winner_compare.movement_tolerance_changed,
        winner_compare.resume_changed,
        winner_compare.passed
    );
    println!(
        "replay_winner ratio={} ft_loss_delta={:.6} retention_loss_delta={:.6} retention_accuracy_delta={:.6} target_retention_gap={:.6} target_retention_ratio={}",
        winner.label,
        winner.summary.target_loss_delta,
        winner.summary.retention_loss_delta,
        winner.summary.retention_accuracy_delta,
        winner.summary.target_retention_gap,
        winner
            .summary
            .target_retention_ratio
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "none".to_string())
    );

    Ok(())
}
