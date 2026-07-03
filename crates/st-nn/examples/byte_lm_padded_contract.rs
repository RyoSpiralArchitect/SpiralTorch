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

//! Tokenizerless padded byte-LM contract.
//!
//! Variable-length byte spans are padded to a fixed row count and use a sparse
//! `ignore_index` target for padded rows. The contract proves DataLoader
//! batching, CE loss, trainer row accounting, checkpoint reloads, and head-only
//! fine-tuning all ignore padded rows instead of letting them dilute validation
//! or training diagnostics.

use st_core::backend::device_caps::DeviceCaps;
use st_nn::{
    byte_lm_sample_stats, dataset_from_vec, load_json_checked, padded_byte_lm_samples, save_json,
    EpochStats, Linear, Module, ModuleTrainer, RoundtableConfig, SoftmaxCrossEntropy,
    SparseClassificationMetrics, SparseFineTuneReport, SparseRetentionGuardConfig, Tensor,
    BYTE_LM_VOCAB,
};
use st_tensor::pure::{PureResult, TensorError};

const VOCAB: usize = BYTE_LM_VOCAB;
const PAD_ROWS: usize = 8;
const BATCH_SAMPLES: usize = 2;
const ACCUMULATION_STEPS: usize = 2;
const IGNORE_INDEX: i32 = -1;
const LABEL_SMOOTHING: f32 = 0.01;
const FT_RETENTION_MAX_LOSS_INCREASE: f32 = 0.5;
const FT_RETENTION_MAX_ACCURACY_DROP: f32 = 0.25;
const FT_RETENTION_MAX_PERPLEXITY_INCREASE: f32 = 1.0;
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
    let schedule = trainer.roundtable(
        (PAD_ROWS * BATCH_SAMPLES) as u32,
        VOCAB as u32,
        RoundtableConfig::default(),
    );
    let loader = dataset_from_vec(samples).batched(BATCH_SAMPLES);
    let mut loss =
        SoftmaxCrossEntropy::with_ignore_index_and_label_smoothing(IGNORE_INDEX, LABEL_SMOOTHING)?;
    trainer.train_epoch(model, &mut loss, loader, &schedule)
}

fn evaluate_metrics(
    trainer: &ModuleTrainer,
    model: &Linear,
    samples: &[(Tensor, Tensor)],
) -> PureResult<SparseClassificationMetrics> {
    let loss =
        SoftmaxCrossEntropy::with_ignore_index_and_label_smoothing(IGNORE_INDEX, LABEL_SMOOTHING)?;
    let metrics = trainer.evaluate_sparse_classification_epoch(
        model,
        &loss,
        dataset_from_vec(samples.to_vec()).batched(BATCH_SAMPLES),
    )?;
    if metrics.active_rows == 0 {
        return Err(contract_error(
            "padded byte LM evaluation had no active rows",
        ));
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
    let schedule = trainer.roundtable(
        (PAD_ROWS * BATCH_SAMPLES) as u32,
        VOCAB as u32,
        RoundtableConfig::default(),
    );
    let train_loader = dataset_from_vec(train_samples).batched(BATCH_SAMPLES);
    let validation_loader = dataset_from_vec(validation_samples).batched(BATCH_SAMPLES);
    let retention_loader = dataset_from_vec(retention_samples).batched(BATCH_SAMPLES);
    let mut loss =
        SoftmaxCrossEntropy::with_ignore_index_and_label_smoothing(IGNORE_INDEX, LABEL_SMOOTHING)?;
    let guard = SparseRetentionGuardConfig::new(
        FT_RETENTION_MAX_LOSS_INCREASE,
        FT_RETENTION_MAX_ACCURACY_DROP,
    )?
    .with_max_perplexity_increase(FT_RETENTION_MAX_PERPLEXITY_INCREASE)?
    .with_target_min_loss_delta(FT_TARGET_MIN_LOSS_DELTA)?;
    trainer.train_epochs_restore_best_sparse_with_finetune_report(
        model,
        &mut loss,
        train_loader,
        validation_loader,
        retention_loader,
        &schedule,
        1,
        guard,
        FT_MOVEMENT_TOLERANCE,
    )
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
    let source_texts = ["spiral", "torch", "byte", ""];
    let reload_texts = ["螺旋byte", "猫byte", "z", ""];
    let source_samples = padded_byte_lm_samples(&source_texts, PAD_ROWS, IGNORE_INDEX)?;
    let reload_samples = padded_byte_lm_samples(&reload_texts, PAD_ROWS, IGNORE_INDEX)?;
    let source_sample_stats = byte_lm_sample_stats(&source_samples, Some(IGNORE_INDEX));
    let reload_sample_stats = byte_lm_sample_stats(&reload_samples, Some(IGNORE_INDEX));
    let source_active_rows = source_sample_stats.active_rows;
    let reload_active_rows = reload_sample_stats.active_rows;
    let source_total_rows = source_sample_stats.total_rows;
    let reload_total_rows = reload_sample_stats.total_rows;
    if source_active_rows == 0 || reload_active_rows == 0 {
        return Err(contract_error("padded byte LM requires active rows"));
    }

    let caps = DeviceCaps::wgpu(32, true, 256);
    let mut trainer = ModuleTrainer::new(caps, -1.0, 0.06, 0.015);
    trainer.set_max_grad_norm(Some(0.5))?;
    trainer.set_gradient_accumulation_steps(ACCUMULATION_STEPS)?;

    let mut source = Linear::new("padded_byte_lm", VOCAB, VOCAB)?;
    zero_parameters(&mut source)?;
    trainer.prepare(&mut source)?;
    let source_before = evaluate_metrics(&trainer, &source, &source_samples)?;
    let source_stats = train_once(&mut trainer, &mut source, source_samples.clone())?;
    let source_after = evaluate_metrics(&trainer, &source, &source_samples)?;
    let source_delta = source_before.delta_to(source_after);
    require_loss_drop(
        "padded source",
        source_before.mean_loss,
        source_after.mean_loss,
    )?;
    if source_stats.rows != source_active_rows {
        return Err(contract_error(format!(
            "source stats counted {} rows but expected {source_active_rows} active rows",
            source_stats.rows
        )));
    }

    let checkpoint = std::env::temp_dir().join(format!(
        "spiraltorch_byte_lm_padded_contract_{}.json",
        std::process::id()
    ));
    save_json(&source, &checkpoint)?;

    let mut target = Linear::new("padded_byte_lm", VOCAB, VOCAB)?;
    zero_parameters(&mut target)?;
    trainer.prepare(&mut target)?;
    let load = load_json_checked(&mut target, &checkpoint)?;
    let _ = std::fs::remove_file(&checkpoint);
    if !load.matched {
        return Err(contract_error(
            "padded byte LM checkpoint fingerprint mismatch",
        ));
    }

    let frozen_weight_params = target.set_parameters_trainable_by_suffix("::weight", false)?;
    let boosted_bias_params = target.set_parameters_learning_rate_scale_by_suffix("::bias", 1.1)?;
    let ft_report = train_with_finetune_report(
        &mut trainer,
        &mut target,
        reload_samples.clone(),
        reload_samples.clone(),
        source_samples.clone(),
    )?;
    let reload_stats = &ft_report.captured.train_summary;
    let ft_delta = ft_report.target_delta;
    let retention_delta = ft_report.retention_delta;
    let ft_summary = ft_report.summary();
    require_loss_drop(
        "padded head-only fine-tune",
        ft_delta.before.mean_loss,
        ft_delta.after.mean_loss,
    )?;
    if reload_stats.rows != reload_active_rows {
        return Err(contract_error(format!(
            "reload stats counted {} rows but expected {reload_active_rows} active rows",
            reload_stats.rows
        )));
    }

    let movement = &ft_report.movement;
    if !movement.frozen_stable() {
        return Err(contract_error(
            "padded byte LM frozen matrix moved during fine-tune",
        ));
    }
    if !movement.trainable_movement_observed() {
        return Err(contract_error("padded byte LM trainable bias did not move"));
    }

    println!(
        "vocab_mode=byte_padded vocab={VOCAB} pad_rows={PAD_ROWS} ignore_index={IGNORE_INDEX} label_smoothing={LABEL_SMOOTHING:.4} ft_target_min_loss_delta={FT_TARGET_MIN_LOSS_DELTA:.6} ft_movement_tolerance={FT_MOVEMENT_TOLERANCE:.6}"
    );
    println!(
        "source_samples={} reload_samples={} source_total_rows={} source_active_rows={} reload_total_rows={} reload_active_rows={}",
        source_samples.len(),
        reload_samples.len(),
        source_total_rows,
        source_active_rows,
        reload_total_rows,
        reload_active_rows
    );
    println!(
        "source_batches={} source_optimizer_steps={} reload_batches={} reload_optimizer_steps={} source_stats_rows={} reload_stats_rows={} accumulation_steps={ACCUMULATION_STEPS} frozen_weight_params={} boosted_bias_params={} guard_accepted_epochs={} guard_retention_rejected_epochs={} guard_target_stale_epochs={}",
        source_stats.batches,
        source_stats.optimizer_steps,
        reload_stats.batches,
        reload_stats.optimizer_steps,
        source_stats.rows,
        reload_stats.rows,
        frozen_weight_params,
        boosted_bias_params,
        ft_report.captured.guard_accepted_epochs,
        ft_report.captured.guard_retention_rejected_epochs,
        ft_report.captured.guard_target_stale_epochs
    );
    println!(
        "source_loss_before={:.6} source_loss_after={:.6} source_accuracy_before={:.6} source_accuracy_after={:.6} source_perplexity_before={:.6} source_perplexity_after={:.6}",
        source_before.mean_loss,
        source_after.mean_loss,
        source_before.accuracy,
        source_after.accuracy,
        source_before.perplexity,
        source_after.perplexity
    );
    println!(
        "source_metric_delta loss_delta={:.6} accuracy_delta={:.6} perplexity_delta={:.6}",
        source_delta.loss_delta, source_delta.accuracy_delta, source_delta.perplexity_delta
    );
    println!(
        "ft_loss_before={:.6} ft_loss_after={:.6} ft_accuracy_before={:.6} ft_accuracy_after={:.6} ft_perplexity_before={:.6} ft_perplexity_after={:.6}",
        ft_delta.before.mean_loss,
        ft_delta.after.mean_loss,
        ft_delta.before.accuracy,
        ft_delta.after.accuracy,
        ft_delta.before.perplexity,
        ft_delta.after.perplexity
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
        "retention_guard target_min_loss_delta={:.6}",
        ft_report.captured.retention_guard.target_min_loss_delta
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
