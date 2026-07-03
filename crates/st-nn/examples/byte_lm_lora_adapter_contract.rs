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
//  SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
//  IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
//  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// ============================================================================

//! Tokenizerless byte-LM low-rank adapter fine-tuning contract.
//!
//! The contract starts from a trained dense `Linear` checkpoint, loads only its
//! base weights into `LoraLinear`, then fine-tunes the low-rank adapter while
//! the dense base remains frozen. This is the smallest non-scratch FT path for
//! validating adapter-style training before lifting the same pattern into larger
//! language-model stacks.

use st_core::backend::device_caps::DeviceCaps;
use st_nn::{
    byte_lm_windows, dataset_from_vec, interleave_replay_samples, load_json_checked, save_json,
    validate_byte_lm_samples, EpochStats, Linear, LoraLinear, Module, ModuleTrainer,
    RoundtableConfig, SoftmaxCrossEntropy, SparseClassificationMetrics, SparseFineTuneReport,
    SparseRetentionGuardConfig, StateCompatibilityReport, StateKeyMapRule, StateTensorTransform,
    Tensor, BYTE_LM_VOCAB,
};
use st_tensor::pure::{PureResult, TensorError};
use std::collections::HashMap;

const VOCAB: usize = BYTE_LM_VOCAB;
const CONTEXT: usize = 1;
const BATCH_WINDOWS: usize = 2;
const ACCUMULATION_STEPS: usize = 2;
const SOURCE_EPOCHS: usize = 2;
const FT_EPOCHS: usize = 6;
const LORA_RANK: usize = 4;
const LORA_ALPHA: f32 = 16.0;
const RETENTION_MAX_LOSS_INCREASE: f32 = 10.0;
const RETENTION_MAX_ACCURACY_DROP: f32 = 1.0;
const RETENTION_MAX_PERPLEXITY_INCREASE: f32 = 100.0;
const FT_TARGET_MIN_LOSS_DELTA: f32 = 0.0;
const FT_MOVEMENT_TOLERANCE: f32 = 1e-6;

fn zero_parameters<M: Module>(model: &mut M) -> PureResult<()> {
    model.visit_parameters_mut(&mut |param| {
        for value in param.value_mut().data_mut() {
            *value = 0.0;
        }
        Ok(())
    })
}

fn train_epochs<M: Module>(
    trainer: &mut ModuleTrainer,
    model: &mut M,
    samples: Vec<(Tensor, Tensor)>,
    epochs: usize,
) -> PureResult<EpochStats> {
    let token_rows = (CONTEXT * BATCH_WINDOWS) as u32;
    let schedule = trainer.roundtable(token_rows, VOCAB as u32, RoundtableConfig::default());
    let mut loss = SoftmaxCrossEntropy::new();
    let mut last = None;
    for _ in 0..epochs {
        let loader = dataset_from_vec(samples.clone()).batched(BATCH_WINDOWS);
        last = Some(trainer.train_epoch(model, &mut loss, loader, &schedule)?);
    }
    last.ok_or_else(|| contract_error("adapter contract received zero training epochs"))
}

fn train_with_finetune_report<M: Module>(
    trainer: &mut ModuleTrainer,
    model: &mut M,
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
        return Err(contract_error(
            "adapter byte LM evaluation received no rows",
        ));
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

fn externalize_dense_head_state(
    dense_state: &HashMap<String, Tensor>,
) -> (HashMap<String, Tensor>, HashMap<String, StateKeyMapRule>) {
    let external_state = HashMap::from([
        (
            "external.lm_head.weight".to_string(),
            dense_state["byte_lm::weight"].transpose(),
        ),
        (
            "external.lm_head.bias".to_string(),
            dense_state["byte_lm::bias"].clone(),
        ),
    ]);
    let key_rules = HashMap::from([
        (
            "external.lm_head.weight".to_string(),
            StateKeyMapRule::with_transform("byte_lm::weight", StateTensorTransform::Transpose),
        ),
        (
            "external.lm_head.bias".to_string(),
            StateKeyMapRule::new("byte_lm::bias"),
        ),
    ]);
    (external_state, key_rules)
}

fn print_preflight_report(label: &str, report: &StateCompatibilityReport) {
    println!(
        "preflight_report label={label} compatible={} matched={} missing={} shape_mismatched={} extra={} source_hash={} matched_subset_hash={}",
        report.compatible,
        report.matched,
        report.missing,
        report.shape_mismatched,
        report.extra,
        report.source.hash,
        report.matched_subset.hash
    );
    for entry in &report.entries {
        println!(
            "preflight_entry label={label} name={} status={} source_name={} transform={} expected_shape={:?} source_shape={:?} original_source_shape={:?}",
            entry.name,
            entry.status.as_str(),
            entry.source_name.as_deref().unwrap_or("none"),
            entry.transform.as_str(),
            entry.expected_shape,
            entry.source_shape,
            entry.original_source_shape
        );
    }
}

fn main() -> PureResult<()> {
    let source_text = "adapter source source source";
    let reload_text = "螺旋adapter螺旋";
    let source_samples = byte_lm_windows(source_text, CONTEXT)?;
    let reload_samples = byte_lm_windows(reload_text, CONTEXT)?;
    let source_sample_stats = validate_byte_lm_samples(&source_samples, None)?;
    let reload_sample_stats = validate_byte_lm_samples(&reload_samples, None)?;
    let source_windows = source_samples.len();
    let reload_windows = reload_samples.len();

    let caps = DeviceCaps::wgpu(32, true, 256);
    let mut trainer = ModuleTrainer::new(caps, -1.0, 0.2, 0.05);
    trainer.set_max_grad_norm(Some(1.0))?;
    trainer.set_gradient_accumulation_steps(ACCUMULATION_STEPS)?;

    let mut source = Linear::new("byte_lm", VOCAB, VOCAB)?;
    zero_parameters(&mut source)?;
    trainer.prepare(&mut source)?;
    let source_before = evaluate_metrics(&trainer, &source, &source_samples)?;
    let source_stats = train_epochs(
        &mut trainer,
        &mut source,
        source_samples.clone(),
        SOURCE_EPOCHS,
    )?;
    let source_after = evaluate_metrics(&trainer, &source, &source_samples)?;
    let source_delta = source_before.delta_to(source_after);
    require_loss_drop(
        "source dense pretrain",
        source_before.mean_loss,
        source_after.mean_loss,
    )?;

    let checkpoint = std::env::temp_dir().join(format!(
        "spiraltorch_byte_lm_lora_adapter_contract_{}.json",
        std::process::id()
    ));
    save_json(&source, &checkpoint)?;

    let mut restored_source = Linear::new("byte_lm", VOCAB, VOCAB)?;
    zero_parameters(&mut restored_source)?;
    trainer.prepare(&mut restored_source)?;
    let dense_load = load_json_checked(&mut restored_source, &checkpoint)?;
    let _ = std::fs::remove_file(&checkpoint);
    if !dense_load.matched {
        return Err(contract_error(
            "adapter source dense checkpoint fingerprint mismatch",
        ));
    }

    let dense_state = restored_source.state_dict()?;
    let (external_dense_state, dense_key_rules) = externalize_dense_head_state(&dense_state);
    let mut target = LoraLinear::new("byte_lm", VOCAB, VOCAB, LORA_RANK, LORA_ALPHA)?;
    trainer.prepare(&mut target)?;
    let base_compatibility = target
        .base_state_dict_compatibility_with_key_rules(&external_dense_state, &dense_key_rules)?;
    print_preflight_report("byte_lm_lora_base", &base_compatibility);
    if !base_compatibility.compatible {
        return Err(contract_error(format!(
            "adapter base checkpoint incompatible: missing={} shape_mismatched={}",
            base_compatibility.missing, base_compatibility.shape_mismatched
        )));
    }
    let base_load =
        target.load_base_from_state_dict_adapted(&external_dense_state, &dense_key_rules)?;
    if !base_load.matched {
        return Err(contract_error("adapter base load fingerprint mismatch"));
    }

    let ft_ready_state = target.state_dict()?;
    let ft_ready_resume = trainer.resume_fingerprint(&target)?;
    let mut resumed = LoraLinear::new("byte_lm", VOCAB, VOCAB, LORA_RANK, LORA_ALPHA)?;
    trainer.prepare(&mut resumed)?;
    let resume_load = resumed.load_state_dict_checked(&ft_ready_state)?;
    let resume_audit = trainer.audit_resume_fingerprint(&resumed, &ft_ready_resume)?;
    if !resume_load.matched || !resume_audit.matched {
        return Err(contract_error(
            "adapter FT-ready resume fingerprint mismatch",
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
    let ft_summary = ft_report.summary();
    let ft_capture = &ft_report.captured;
    if ft_capture.guarded_best_epoch.is_none() {
        return Err(contract_error(
            "adapter retention guard rejected every fine-tune epoch",
        ));
    }
    let ft_delta = ft_report.target_delta;
    let retention_delta = ft_report.retention_delta;
    require_loss_drop(
        "adapter fine-tune",
        ft_delta.before.mean_loss,
        ft_delta.after.mean_loss,
    )?;

    let movement = &ft_report.movement;
    if !movement.frozen_stable() {
        return Err(contract_error("adapter frozen base moved during fine-tune"));
    }
    if !movement.trainable_movement_observed() {
        return Err(contract_error("adapter low-rank parameters did not move"));
    }

    println!(
        "vocab_mode=byte vocab={VOCAB} context={CONTEXT} adapter_rank={LORA_RANK} adapter_alpha={LORA_ALPHA:.3} unknown_fraction=0.0"
    );
    println!(
        "source_bytes={} reload_bytes={} source_windows={} reload_windows={}",
        source_text.len(),
        reload_text.len(),
        source_windows,
        reload_windows
    );
    println!(
        "source_epochs={SOURCE_EPOCHS} ft_epochs={FT_EPOCHS} accumulation_steps={ACCUMULATION_STEPS} source_batches={} source_optimizer_steps={} ft_batches={} ft_optimizer_steps={} guard_accepted_epochs={} guard_retention_rejected_epochs={} guard_target_stale_epochs={}",
        source_stats.batches,
        source_stats.optimizer_steps,
        ft_capture.train_summary.batches,
        ft_capture.train_summary.optimizer_steps,
        ft_capture.guard_accepted_epochs,
        ft_capture.guard_retention_rejected_epochs,
        ft_capture.guard_target_stale_epochs
    );
    println!(
        "byte_lm_lora_sample_stats source_samples={} source_rows={} source_active_rows={} reload_samples={} reload_rows={} reload_active_rows={}",
        source_sample_stats.samples,
        source_sample_stats.total_rows,
        source_sample_stats.active_rows,
        reload_sample_stats.samples,
        reload_sample_stats.total_rows,
        reload_sample_stats.active_rows
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
        "source_hash={} loaded_hash={} dense_load_matched={} base_source_hash={} base_loaded_hash={} base_load_matched={}",
        dense_load.source.hash,
        dense_load.loaded.hash,
        dense_load.matched,
        base_load.source.hash,
        base_load.loaded.hash,
        base_load.matched
    );
    println!(
        "resume_hash={} trainer_hash={} parameter_training_hash={} resume_matched={}",
        ft_summary.resume_hash,
        ft_summary.resume_trainer_hash,
        ft_summary.resume_parameter_training_hash,
        resume_audit.matched
    );
    println!(
        "report_status={} movement_status={} movement_tolerance={:.6} frozen_stable={} trainable_moved={} trainable_changed={} frozen_changed={}",
        ft_report.status(),
        movement.status(),
        ft_report.movement_tolerance,
        movement.frozen_stable(),
        movement.trainable_movement_observed(),
        movement.trainable_changed,
        movement.frozen_changed
    );

    Ok(())
}
