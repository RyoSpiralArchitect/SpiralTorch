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

//! Tokenizerless next-byte fine-tuning contract smoke.
//!
//! This is deliberately small and CPU/WGPU-planner agnostic: a 256-way byte
//! predictor consumes one-hot byte inputs and sparse next-byte class targets,
//! then fine-tunes on UTF-8 Japanese bytes with the weight matrix frozen. The
//! contract asserts that byte data has no unknown-token path, checkpoint
//! fingerprints match, frozen weights stay stable, and the trainable bias still
//! moves.

use st_core::backend::device_caps::DeviceCaps;
use st_nn::{
    byte_lm_windows, dataset_from_vec, interleave_replay_samples, load_json_checked, save_json,
    EpochStats, Linear, Module, ModuleTrainer, RoundtableConfig, SoftmaxCrossEntropy,
    SparseClassificationMetrics, SparseFineTuneReport, SparseRetentionGuardConfig, Tensor,
    BYTE_LM_VOCAB,
};
use st_tensor::pure::{PureResult, TensorError};

const VOCAB: usize = BYTE_LM_VOCAB;
const CONTEXT: usize = 8;
const BATCH_WINDOWS: usize = 4;
const ACCUMULATION_STEPS: usize = 2;
const FT_EPOCHS: usize = 3;
const RETENTION_MAX_LOSS_INCREASE: f32 = 0.5;
const RETENTION_MAX_ACCURACY_DROP: f32 = 0.15;
const RETENTION_MAX_PERPLEXITY_INCREASE: f32 = 1.0;
const FT_TARGET_MIN_LOSS_DELTA: f32 = 1e-4;
const FT_MOVEMENT_TOLERANCE: f32 = 1e-6;

fn zero_parameters<M: Module>(model: &mut M) -> PureResult<()> {
    model.visit_parameters_mut(&mut |param| {
        for value in param.value_mut().data_mut() {
            *value = 0.0;
        }
        Ok(())
    })
}

fn train_once(
    trainer: &mut ModuleTrainer,
    model: &mut Linear,
    samples: Vec<(Tensor, Tensor)>,
) -> PureResult<EpochStats> {
    let token_rows = (CONTEXT * BATCH_WINDOWS) as u32;
    let schedule = trainer.roundtable(token_rows, VOCAB as u32, RoundtableConfig::default());
    let loader = dataset_from_vec(samples).batched(BATCH_WINDOWS);
    let mut loss = SoftmaxCrossEntropy::new();
    trainer.train_epoch(model, &mut loss, loader, &schedule)
}

fn evaluate_metrics(
    trainer: &ModuleTrainer,
    model: &Linear,
    samples: &[(Tensor, Tensor)],
) -> PureResult<SparseClassificationMetrics> {
    let loss = SoftmaxCrossEntropy::new();
    let metrics = trainer.evaluate_sparse_classification_epoch(
        model,
        &loss,
        dataset_from_vec(samples.to_vec()).batched(BATCH_WINDOWS),
    )?;
    if metrics.active_rows == 0 {
        return Err(contract_error("byte LM evaluation received no rows"));
    }
    Ok(metrics)
}

fn train_with_finetune_report(
    trainer: &mut ModuleTrainer,
    model: &mut Linear,
    train_samples: Vec<(Tensor, Tensor)>,
    validation_samples: Vec<(Tensor, Tensor)>,
    retention_samples: Vec<(Tensor, Tensor)>,
) -> PureResult<SparseFineTuneReport> {
    let token_rows = (CONTEXT * BATCH_WINDOWS) as u32;
    let schedule = trainer.roundtable(token_rows, VOCAB as u32, RoundtableConfig::default());
    let train_loader = dataset_from_vec(train_samples).batched(BATCH_WINDOWS);
    let validation_loader = dataset_from_vec(validation_samples).batched(BATCH_WINDOWS);
    let retention_loader = dataset_from_vec(retention_samples).batched(BATCH_WINDOWS);
    let mut loss = SoftmaxCrossEntropy::new();
    trainer.train_epochs_restore_best_sparse_with_finetune_report(
        model,
        &mut loss,
        train_loader,
        validation_loader,
        retention_loader,
        &schedule,
        FT_EPOCHS,
        SparseRetentionGuardConfig::new(RETENTION_MAX_LOSS_INCREASE, RETENTION_MAX_ACCURACY_DROP)?
            .with_max_perplexity_increase(RETENTION_MAX_PERPLEXITY_INCREASE)?
            .with_target_min_loss_delta(FT_TARGET_MIN_LOSS_DELTA)?,
        FT_MOVEMENT_TOLERANCE,
    )
}

fn contract_error(message: impl Into<String>) -> TensorError {
    TensorError::IoError {
        message: message.into(),
    }
}

fn main() -> PureResult<()> {
    let source_text = "spiraltorch learns from source bytes";
    let reload_text = "螺旋はbyteで壊れない";
    let source_samples = byte_lm_windows(source_text, CONTEXT)?;
    let reload_samples = byte_lm_windows(reload_text, CONTEXT)?;
    let source_windows = source_samples.len();
    let reload_windows = reload_samples.len();

    let caps = DeviceCaps::wgpu(32, true, 256);
    let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
    trainer.set_max_grad_norm(Some(0.25))?;
    trainer.set_gradient_accumulation_steps(ACCUMULATION_STEPS)?;

    let mut source = Linear::new("byte_lm", VOCAB, VOCAB)?;
    zero_parameters(&mut source)?;
    trainer.prepare(&mut source)?;
    let source_before = evaluate_metrics(&trainer, &source, &source_samples)?;
    let source_stats = train_once(&mut trainer, &mut source, source_samples.clone())?;
    let source_after = evaluate_metrics(&trainer, &source, &source_samples)?;
    let source_delta = source_before.delta_to(source_after);

    let checkpoint = std::env::temp_dir().join(format!(
        "spiraltorch_byte_lm_finetune_contract_{}.json",
        std::process::id()
    ));
    save_json(&source, &checkpoint)?;

    let mut target = Linear::new("byte_lm", VOCAB, VOCAB)?;
    zero_parameters(&mut target)?;
    trainer.prepare(&mut target)?;
    let load = load_json_checked(&mut target, &checkpoint)?;
    let _ = std::fs::remove_file(&checkpoint);
    if !load.matched {
        return Err(contract_error("byte LM checkpoint fingerprint mismatch"));
    }

    let frozen_weight_params = target.set_parameters_trainable_by_suffix("::weight", false)?;
    let boosted_bias_params = target.set_parameters_learning_rate_scale_by_suffix("::bias", 1.1)?;
    let ft_ready_state = target.state_dict()?;
    let ft_ready_resume = trainer.resume_fingerprint(&target)?;
    let mut resumed = Linear::new("byte_lm", VOCAB, VOCAB)?;
    trainer.prepare(&mut resumed)?;
    let resume_load = resumed.load_state_dict_checked(&ft_ready_state)?;
    resumed.set_parameters_trainable_by_suffix("::weight", false)?;
    resumed.set_parameters_learning_rate_scale_by_suffix("::bias", 1.1)?;
    let resume_audit = trainer.audit_resume_fingerprint(&resumed, &ft_ready_resume)?;
    if !resume_load.matched || !resume_audit.matched {
        return Err(contract_error(
            "byte LM FT-ready resume fingerprint mismatch",
        ));
    }
    let ft_train_samples = interleave_replay_samples(&reload_samples, &source_samples, 1)?;
    let ft_report = train_with_finetune_report(
        &mut trainer,
        &mut target,
        ft_train_samples,
        reload_samples.clone(),
        source_samples.clone(),
    )?;
    let ft_capture = &ft_report.captured;
    if ft_capture.guarded_best_epoch.is_none() {
        return Err(contract_error(
            "byte LM retention guard rejected every fine-tune epoch",
        ));
    }
    let ft_delta = ft_report.target_delta;
    let retention_delta = ft_report.retention_delta;
    let ft_summary = ft_report.summary();
    let movement = &ft_report.movement;
    if !ft_report.movement.frozen_stable() {
        return Err(contract_error(
            "byte LM frozen matrix moved during fine-tune",
        ));
    }
    if !ft_report.movement.trainable_movement_observed() {
        return Err(contract_error("byte LM trainable bias did not move"));
    }

    println!("vocab_mode=byte vocab={VOCAB} context={CONTEXT} unknown_fraction=0.0");
    println!(
        "source_bytes={} reload_bytes={} source_windows={} reload_windows={}",
        source_text.len(),
        reload_text.len(),
        source_windows,
        reload_windows
    );
    println!(
        "source_batches={} source_optimizer_steps={} reload_batches={} reload_optimizer_steps={} source_rows={} reload_rows={} accumulation_steps={ACCUMULATION_STEPS} frozen_weight_params={} boosted_bias_params={}",
        source_stats.batches,
        source_stats.optimizer_steps,
        ft_capture.train_summary.batches,
        ft_capture.train_summary.optimizer_steps,
        source_stats.rows,
        ft_capture.train_summary.rows,
        frozen_weight_params,
        boosted_bias_params
    );
    println!(
        "source_loss_before={:.6} source_loss_after={:.6} source_accuracy_before={:.6} source_accuracy_after={:.6} source_perplexity_before={:.6} source_perplexity_after={:.6} source_eval_rows={}",
        source_before.mean_loss,
        source_after.mean_loss,
        source_before.accuracy,
        source_after.accuracy,
        source_before.perplexity,
        source_after.perplexity,
        source_after.active_rows
    );
    println!(
        "source_metric_delta loss_delta={:.6} accuracy_delta={:.6} perplexity_delta={:.6}",
        source_delta.loss_delta, source_delta.accuracy_delta, source_delta.perplexity_delta
    );
    println!(
        "ft_loss_before={:.6} ft_loss_after={:.6} ft_accuracy_before={:.6} ft_accuracy_after={:.6} ft_perplexity_before={:.6} ft_perplexity_after={:.6} ft_eval_rows={}",
        ft_delta.before.mean_loss,
        ft_delta.after.mean_loss,
        ft_delta.before.accuracy,
        ft_delta.after.accuracy,
        ft_delta.before.perplexity,
        ft_delta.after.perplexity,
        ft_delta.after.active_rows
    );
    println!(
        "ft_metric_delta loss_delta={:.6} accuracy_delta={:.6} perplexity_delta={:.6}",
        ft_delta.loss_delta, ft_delta.accuracy_delta, ft_delta.perplexity_delta
    );
    println!(
        "retention_loss_before={:.6} retention_loss_after={:.6} retention_accuracy_before={:.6} retention_accuracy_after={:.6} retention_perplexity_before={:.6} retention_perplexity_after={:.6} retention_eval_rows={}",
        retention_delta.before.mean_loss,
        retention_delta.after.mean_loss,
        retention_delta.before.accuracy,
        retention_delta.after.accuracy,
        retention_delta.before.perplexity,
        retention_delta.after.perplexity,
        retention_delta.after.active_rows
    );
    println!(
        "retention_metric_delta loss_delta={:.6} accuracy_delta={:.6} perplexity_delta={:.6}",
        retention_delta.loss_delta,
        retention_delta.accuracy_delta,
        retention_delta.perplexity_delta
    );
    println!(
        "ft_selectivity target_retention_gap={:.6} target_retention_ratio={}",
        ft_summary.target_retention_gap,
        ft_summary
            .target_retention_ratio
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "none".to_string())
    );
    println!(
        "retention_guard max_loss_increase={:.6} max_accuracy_drop={:.6} max_perplexity_increase={:.6} target_min_loss_delta={:.6} max_allowed_loss={:.6} min_allowed_accuracy={:.6} max_allowed_perplexity={:.6} baseline_loss={:.6} baseline_accuracy={:.6} baseline_perplexity={:.6} guarded_best_epoch={:?} guard_accepted_epochs={} guard_retention_rejected_epochs={} guard_target_stale_epochs={} best_validation_loss={:.6} best_validation_accuracy={:.6} best_validation_perplexity={:.6} best_retention_loss={:.6} best_retention_accuracy={:.6} best_retention_perplexity={:.6} best_retention_loss_increase={:.6} best_retention_accuracy_drop={:.6} best_retention_perplexity_increase={:.6}",
        ft_capture.retention_guard.max_loss_increase,
        ft_capture.retention_guard.max_accuracy_drop,
        ft_capture.retention_guard.max_perplexity_increase.unwrap_or(0.0),
        ft_capture.retention_guard.target_min_loss_delta,
        ft_capture.max_allowed_retention_loss,
        ft_capture.min_allowed_retention_accuracy,
        ft_capture.max_allowed_retention_perplexity.unwrap_or(0.0),
        ft_capture.retention_baseline.mean_loss,
        ft_capture.retention_baseline.accuracy,
        ft_capture.retention_baseline.perplexity,
        ft_capture.guarded_best_epoch,
        ft_capture.guard_accepted_epochs,
        ft_capture.guard_retention_rejected_epochs,
        ft_capture.guard_target_stale_epochs,
        ft_capture.best_validation_metrics.mean_loss,
        ft_capture.best_validation_metrics.accuracy,
        ft_capture.best_validation_metrics.perplexity,
        ft_capture.best_retention_metrics.mean_loss,
        ft_capture.best_retention_metrics.accuracy,
        ft_capture.best_retention_metrics.perplexity,
        ft_capture.best_retention_loss_increase,
        ft_capture.best_retention_accuracy_drop,
        ft_capture.best_retention_perplexity_increase
    );
    println!("source_epoch_loss={:.6}", source_stats.average_loss_per_row);
    println!(
        "reload_epoch_loss={:.6}",
        ft_capture.train_summary.final_loss_per_row
    );
    println!("source_hash={}", load.source.hash);
    println!("loaded_hash={}", load.loaded.hash);
    println!("load_matched={}", load.matched);
    println!(
        "resume_hash={} trainer_hash={} parameter_training_hash={} resume_matched={}",
        ft_ready_resume.hash,
        ft_ready_resume.trainer.hash,
        ft_ready_resume.parameter_training.hash,
        resume_audit.matched
    );
    println!(
        "report_status={} movement_status={} movement_tolerance={:.6} frozen_stable={} trainable_moved={}",
        ft_report.status(),
        movement.status(),
        ft_report.movement_tolerance,
        movement.frozen_stable(),
        movement.trainable_movement_observed()
    );

    Ok(())
}
