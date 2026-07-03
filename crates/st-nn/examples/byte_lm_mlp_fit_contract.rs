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

//! Multi-epoch tokenizerless byte-LM fitting contract.
//!
//! This keeps the model intentionally small: a byte one-hot input flows through
//! a Rust-native `Linear -> Relu -> Linear` head. The contract proves that
//! windowed sparse next-byte targets can reduce row-weighted CE loss across
//! epochs, reload from a checked checkpoint, then perform head-only fine-tuning
//! while the frozen byte embedding remains stable.

use st_core::backend::device_caps::DeviceCaps;
use st_nn::{
    byte_lm_windows, dataset_from_vec, load_json_checked, save_json, validate_byte_lm_samples,
    EarlyStoppingConfig, EpochValidationBestState, Linear, LrPlateauConfig, Module, ModuleTrainer,
    Relu, RoundtableConfig, Sequential, SoftmaxCrossEntropy, SparseClassificationMetrics,
    SparseFineTuneReport, SparseRetentionGuardConfig, Tensor, ValidationTrainingControls,
    BYTE_LM_VOCAB,
};
use st_tensor::pure::{PureResult, TensorError};

const VOCAB: usize = BYTE_LM_VOCAB;
const HIDDEN: usize = 32;
const CONTEXT: usize = 8;
const BATCH_WINDOWS: usize = 4;
const ACCUMULATION_STEPS: usize = 2;
const SOURCE_EPOCHS: usize = 6;
const FT_EPOCHS: usize = 8;
const EARLY_STOPPING_PATIENCE: usize = 2;
const EARLY_STOPPING_MIN_DELTA: f32 = 0.0;
const LR_PLATEAU_PATIENCE: usize = 2;
const LR_PLATEAU_FACTOR: f32 = 0.8;
const LR_PLATEAU_MIN_DELTA: f32 = 0.0;
const FT_RETENTION_MAX_LOSS_INCREASE: f32 = 0.75;
const FT_RETENTION_MAX_ACCURACY_DROP: f32 = 0.5;
const FT_RETENTION_MAX_PERPLEXITY_INCREASE: f32 = 2.0;
const FT_TARGET_MIN_LOSS_DELTA: f32 = 1e-4;
const FT_MOVEMENT_TOLERANCE: f32 = 1e-6;

fn train_validation_split(
    samples: &[(Tensor, Tensor)],
) -> PureResult<(Vec<(Tensor, Tensor)>, Vec<(Tensor, Tensor)>)> {
    if samples.len() < 2 {
        return Err(contract_error(
            "byte MLP split requires at least two window samples",
        ));
    }
    let mut train = Vec::new();
    let mut validation = Vec::new();
    for (idx, sample) in samples.iter().cloned().enumerate() {
        if idx % 4 == 0 {
            validation.push(sample);
        } else {
            train.push(sample);
        }
    }
    if train.is_empty() || validation.is_empty() {
        return Err(contract_error("byte MLP split produced an empty partition"));
    }
    Ok((train, validation))
}

fn build_model() -> PureResult<Sequential> {
    let mut model = Sequential::new();
    model.push(Linear::new("embed", VOCAB, HIDDEN)?);
    model.push(Relu::new());
    model.push(Linear::new("head", HIDDEN, VOCAB)?);
    scale_parameters(&mut model, 0.001)?;
    Ok(model)
}

fn scale_parameters<M: Module>(model: &mut M, factor: f32) -> PureResult<()> {
    model.visit_parameters_mut(&mut |param| {
        for value in param.value_mut().data_mut() {
            *value *= factor;
        }
        Ok(())
    })
}

fn train_best_epochs<M: Module>(
    trainer: &mut ModuleTrainer,
    model: &mut M,
    train_samples: &[(Tensor, Tensor)],
    validation_samples: &[(Tensor, Tensor)],
    epochs: usize,
) -> PureResult<EpochValidationBestState> {
    let token_rows = (CONTEXT * BATCH_WINDOWS) as u32;
    let schedule = trainer.roundtable(token_rows, VOCAB as u32, RoundtableConfig::default());
    let train_loader = dataset_from_vec(train_samples.to_vec()).batched(BATCH_WINDOWS);
    let validation_loader = dataset_from_vec(validation_samples.to_vec()).batched(BATCH_WINDOWS);
    let mut loss = SoftmaxCrossEntropy::new();
    let controls = ValidationTrainingControls::default()
        .with_early_stopping(EarlyStoppingConfig::new(
            EARLY_STOPPING_PATIENCE,
            EARLY_STOPPING_MIN_DELTA,
        )?)
        .with_lr_plateau(LrPlateauConfig::new(
            LR_PLATEAU_PATIENCE,
            LR_PLATEAU_FACTOR,
            LR_PLATEAU_MIN_DELTA,
        )?);
    trainer.train_epochs_restore_best_on_validation_with_controls(
        model,
        &mut loss,
        train_loader,
        validation_loader,
        &schedule,
        epochs,
        controls,
    )
}

fn train_best_epochs_with_finetune_report<M: Module>(
    trainer: &mut ModuleTrainer,
    model: &mut M,
    train_samples: &[(Tensor, Tensor)],
    validation_samples: &[(Tensor, Tensor)],
    retention_samples: &[(Tensor, Tensor)],
    epochs: usize,
) -> PureResult<SparseFineTuneReport> {
    let token_rows = (CONTEXT * BATCH_WINDOWS) as u32;
    let schedule = trainer.roundtable(token_rows, VOCAB as u32, RoundtableConfig::default());
    let train_loader = dataset_from_vec(train_samples.to_vec()).batched(BATCH_WINDOWS);
    let validation_loader = dataset_from_vec(validation_samples.to_vec()).batched(BATCH_WINDOWS);
    let retention_loader = dataset_from_vec(retention_samples.to_vec()).batched(BATCH_WINDOWS);
    let mut loss = SoftmaxCrossEntropy::new();
    let controls = ValidationTrainingControls::default()
        .with_early_stopping(EarlyStoppingConfig::new(
            EARLY_STOPPING_PATIENCE,
            EARLY_STOPPING_MIN_DELTA,
        )?)
        .with_lr_plateau(LrPlateauConfig::new(
            LR_PLATEAU_PATIENCE,
            LR_PLATEAU_FACTOR,
            LR_PLATEAU_MIN_DELTA,
        )?);
    let guard = SparseRetentionGuardConfig::new(
        FT_RETENTION_MAX_LOSS_INCREASE,
        FT_RETENTION_MAX_ACCURACY_DROP,
    )?
    .with_max_perplexity_increase(FT_RETENTION_MAX_PERPLEXITY_INCREASE)?
    .with_target_min_loss_delta(FT_TARGET_MIN_LOSS_DELTA)?;
    trainer.train_epochs_restore_best_sparse_with_finetune_report_and_controls(
        model,
        &mut loss,
        train_loader,
        validation_loader,
        retention_loader,
        &schedule,
        epochs,
        guard,
        FT_MOVEMENT_TOLERANCE,
        controls,
    )
}

fn evaluate_metrics<M: Module>(
    trainer: &ModuleTrainer,
    model: &M,
    samples: &[(Tensor, Tensor)],
) -> PureResult<SparseClassificationMetrics> {
    let loss = SoftmaxCrossEntropy::new();
    let metrics = trainer.evaluate_sparse_classification_epoch(
        model,
        &loss,
        dataset_from_vec(samples.to_vec()).batched(BATCH_WINDOWS),
    )?;
    if metrics.active_rows == 0 {
        return Err(contract_error("byte MLP evaluation received no rows"));
    }
    Ok(metrics)
}

fn require_loss_drop(label: &str, before: f32, after: f32) -> PureResult<()> {
    if after < before && before.is_finite() && after.is_finite() {
        Ok(())
    } else {
        Err(contract_error(format!(
            "{label} loss did not improve: before={before:.6} after={after:.6}"
        )))
    }
}

fn contract_error(message: impl Into<String>) -> TensorError {
    TensorError::IoError {
        message: message.into(),
    }
}

fn main() -> PureResult<()> {
    let source_text =
        "spiraltorch learns byte patterns; spiraltorch learns byte patterns; byte byte byte";
    let reload_text = "螺旋byte螺旋byte螺旋byteは壊れないbyte";
    let source_samples = byte_lm_windows(source_text, CONTEXT)?;
    let reload_samples = byte_lm_windows(reload_text, CONTEXT)?;
    let source_sample_stats = validate_byte_lm_samples(&source_samples, None)?;
    let reload_sample_stats = validate_byte_lm_samples(&reload_samples, None)?;
    let (source_train, source_validation) = train_validation_split(&source_samples)?;
    let (reload_train, _reload_validation) = train_validation_split(&reload_samples)?;
    let source_windows = source_samples.len();
    let reload_windows = reload_samples.len();

    let caps = DeviceCaps::wgpu(32, true, 256);
    let mut trainer = ModuleTrainer::new(caps, -1.0, 0.08, 0.02);
    trainer.set_max_grad_norm(Some(0.75))?;
    trainer.set_gradient_accumulation_steps(ACCUMULATION_STEPS)?;

    let mut source = build_model()?;
    trainer.prepare(&mut source)?;
    let source_before = evaluate_metrics(&trainer, &source, &source_samples)?;
    let source_best = train_best_epochs(
        &mut trainer,
        &mut source,
        &source_train,
        &source_validation,
        SOURCE_EPOCHS,
    )?;
    let source_after = evaluate_metrics(&trainer, &source, &source_samples)?;
    let source_delta = source_before.delta_to(source_after);
    require_loss_drop("source", source_before.mean_loss, source_after.mean_loss)?;

    let checkpoint = std::env::temp_dir().join(format!(
        "spiraltorch_byte_mlp_fit_contract_{}.json",
        std::process::id()
    ));
    save_json(&source, &checkpoint)?;

    let mut target = build_model()?;
    trainer.prepare(&mut target)?;
    let load = load_json_checked(&mut target, &checkpoint)?;
    let _ = std::fs::remove_file(&checkpoint);
    if !load.matched {
        return Err(contract_error("byte MLP checkpoint fingerprint mismatch"));
    }

    let frozen_embed = target.set_parameters_trainable_by_prefix("embed::", false)?;
    let boosted_head = target.set_parameters_learning_rate_scale_by_prefix("head::", 1.1)?;
    let ft_report = train_best_epochs_with_finetune_report(
        &mut trainer,
        &mut target,
        &reload_train,
        &reload_samples,
        &source_samples,
        FT_EPOCHS,
    )?;
    let ft_best = &ft_report.captured;
    let ft_delta = ft_report.target_delta;
    let retention_delta = ft_report.retention_delta;
    let ft_summary = ft_report.summary();
    require_loss_drop(
        "head-only fine-tune",
        ft_delta.before.mean_loss,
        ft_delta.after.mean_loss,
    )?;

    let movement = &ft_report.movement;
    if !movement.frozen_stable() {
        return Err(contract_error(
            "byte MLP frozen embedding moved during fine-tune",
        ));
    }
    if !movement.trainable_movement_observed() {
        return Err(contract_error("byte MLP trainable head did not move"));
    }

    let source_summary = &source_best.validation_summary;
    let ft_validation_last_loss = ft_best
        .validation_history
        .last()
        .map(|metrics| metrics.mean_loss)
        .unwrap_or(ft_best.validation_baseline.mean_loss);
    let ft_validation_best_improvement =
        ft_best.validation_baseline.mean_loss - ft_best.best_validation_metrics.mean_loss;

    println!("vocab_mode=byte vocab={VOCAB} hidden={HIDDEN} context={CONTEXT}");
    println!(
        "source_bytes={} reload_bytes={} source_windows={} reload_windows={}",
        source_text.len(),
        reload_text.len(),
        source_windows,
        reload_windows
    );
    println!(
        "byte_mlp_sample_stats source_samples={} source_rows={} source_active_rows={} reload_samples={} reload_rows={} reload_active_rows={}",
        source_sample_stats.samples,
        source_sample_stats.total_rows,
        source_sample_stats.active_rows,
        reload_sample_stats.samples,
        reload_sample_stats.total_rows,
        reload_sample_stats.active_rows
    );
    println!(
        "source_epochs={SOURCE_EPOCHS} ft_epochs={FT_EPOCHS} accumulation_steps={ACCUMULATION_STEPS} early_stopping_patience={EARLY_STOPPING_PATIENCE} early_stopping_min_delta={EARLY_STOPPING_MIN_DELTA:.6} lr_plateau_patience={LR_PLATEAU_PATIENCE} lr_plateau_factor={LR_PLATEAU_FACTOR:.3} lr_plateau_min_delta={LR_PLATEAU_MIN_DELTA:.6} ft_target_min_loss_delta={FT_TARGET_MIN_LOSS_DELTA:.6} ft_movement_tolerance={FT_MOVEMENT_TOLERANCE:.6} frozen_embed_params={} boosted_head_params={} source_train_windows={} source_validation_windows={} ft_train_windows={} ft_validation_windows={} source_batches={} source_optimizer_steps={} ft_batches={} ft_optimizer_steps={} source_epochs_run={} ft_epochs_run={} source_early_stopped={} ft_early_stopped={} source_lr_decay_steps={} ft_lr_decay_steps={} source_final_hyper_lr={:.6} ft_final_hyper_lr={:.6}",
        frozen_embed,
        boosted_head,
        source_train.len(),
        source_validation.len(),
        reload_train.len(),
        reload_samples.len(),
        source_best.train_summary.batches,
        source_best.train_summary.optimizer_steps,
        ft_best.train_summary.batches,
        ft_best.train_summary.optimizer_steps,
        source_best.train_summary.epochs,
        ft_best.train_summary.epochs,
        source_best.early_stopped,
        ft_best.early_stopped,
        source_best.lr_decay_steps,
        ft_best.lr_decay_steps,
        source_best.final_hyper_learning_rate,
        ft_best.final_hyper_learning_rate,
    );
    println!(
        "source_loss_before={:.6} source_accuracy_before={:.6} source_perplexity_before={:.6} source_eval_rows={}",
        source_before.mean_loss,
        source_before.accuracy,
        source_before.perplexity,
        source_before.active_rows
    );
    println!(
        "source_validation_loss_last_epoch={:.6}",
        source_summary.final_loss_per_row
    );
    println!(
        "source_validation_best_epoch={:?} source_validation_best_loss={:.6} source_validation_best_improvement={:.6}",
        source_summary.best_epoch,
        source_summary.best_loss_per_row,
        source_summary.best_improvement
    );
    println!(
        "source_loss_after={:.6} source_accuracy_after={:.6} source_perplexity_after={:.6}",
        source_after.mean_loss, source_after.accuracy, source_after.perplexity
    );
    println!(
        "source_metric_delta loss_delta={:.6} accuracy_delta={:.6} perplexity_delta={:.6}",
        source_delta.loss_delta, source_delta.accuracy_delta, source_delta.perplexity_delta
    );
    println!(
        "source_best_hash={} source_final_hash={} source_best_differs_from_final={}",
        source_best.best_fingerprint.hash,
        source_best.final_fingerprint.hash,
        source_best.best_differs_from_final
    );
    println!(
        "ft_loss_before={:.6} ft_accuracy_before={:.6} ft_perplexity_before={:.6} ft_eval_rows={}",
        ft_delta.before.mean_loss,
        ft_delta.before.accuracy,
        ft_delta.before.perplexity,
        ft_delta.before.active_rows
    );
    println!(
        "ft_validation_loss_last_epoch={:.6}",
        ft_validation_last_loss
    );
    println!(
        "ft_validation_best_epoch={:?} guard_accepted_epochs={} guard_retention_rejected_epochs={} guard_target_stale_epochs={} ft_validation_best_loss={:.6} ft_validation_best_improvement={:.6}",
        ft_best.guarded_best_epoch,
        ft_best.guard_accepted_epochs,
        ft_best.guard_retention_rejected_epochs,
        ft_best.guard_target_stale_epochs,
        ft_best.best_validation_metrics.mean_loss,
        ft_validation_best_improvement
    );
    println!(
        "ft_loss_after={:.6} ft_accuracy_after={:.6} ft_perplexity_after={:.6}",
        ft_delta.after.mean_loss, ft_delta.after.accuracy, ft_delta.after.perplexity
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
        "retention_guard target_min_loss_delta={:.6}",
        ft_best.retention_guard.target_min_loss_delta
    );
    println!(
        "ft_best_hash={} ft_final_hash={} ft_best_differs_from_final={}",
        ft_best.best_fingerprint.hash,
        ft_best.final_fingerprint.hash,
        ft_best.best_differs_from_final
    );
    println!("source_hash={}", load.source.hash);
    println!("loaded_hash={}", load.loaded.hash);
    println!("load_matched={}", load.matched);
    println!(
        "resume_hash={} trainer_hash={} parameter_training_hash={}",
        ft_summary.resume_hash,
        ft_summary.resume_trainer_hash,
        ft_summary.resume_parameter_training_hash
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
