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

//! Tokenizerless byte-MLP LoRA head fine-tuning contract.
//!
//! This lifts the adapter smoke from a single dense byte head into a tiny
//! `Linear -> Relu -> LoraLinear` stack. A trained dense MLP checkpoint seeds
//! the frozen embedding and the LoRA head base; only the low-rank head delta is
//! allowed to move during retention-guarded target fine-tuning.

use st_core::backend::device_caps::DeviceCaps;
use st_nn::{
    byte_lm_windows, dataset_from_vec, load_json_checked, save_json, validate_byte_lm_samples,
    EarlyStoppingConfig, EpochValidationBestState, Linear, LoraLinear, LrPlateauConfig, Module,
    ModuleTrainer, Relu, RoundtableConfig, Sequential, SoftmaxCrossEntropy,
    SparseClassificationMetrics, SparseFineTuneReport, SparseRetentionGuardConfig,
    StateCompatibilityReport, StateKeyMapRule, StateLoadReport, StateTensorTransform, Tensor,
    ValidationTrainingControls, BYTE_LM_VOCAB,
};
use st_tensor::pure::{PureResult, TensorError};
use std::collections::HashMap;

const VOCAB: usize = BYTE_LM_VOCAB;
const HIDDEN: usize = 24;
const CONTEXT: usize = 4;
const BATCH_WINDOWS: usize = 4;
const ACCUMULATION_STEPS: usize = 2;
const SOURCE_EPOCHS: usize = 4;
const FT_EPOCHS: usize = 10;
const EARLY_STOPPING_PATIENCE: usize = 2;
const EARLY_STOPPING_MIN_DELTA: f32 = 0.0;
const LR_PLATEAU_PATIENCE: usize = 2;
const LR_PLATEAU_FACTOR: f32 = 0.8;
const LR_PLATEAU_MIN_DELTA: f32 = 0.0;
const LORA_RANK: usize = 12;
const LORA_ALPHA: f32 = 64.0;
const FT_RETENTION_MAX_LOSS_INCREASE: f32 = 10.0;
const FT_RETENTION_MAX_ACCURACY_DROP: f32 = 1.0;
const FT_RETENTION_MAX_PERPLEXITY_INCREASE: f32 = 100.0;
const FT_TARGET_MIN_LOSS_DELTA: f32 = 0.0;
const FT_MOVEMENT_TOLERANCE: f32 = 1e-6;

fn train_validation_split(
    samples: &[(Tensor, Tensor)],
) -> PureResult<(Vec<(Tensor, Tensor)>, Vec<(Tensor, Tensor)>)> {
    if samples.len() < 2 {
        return Err(contract_error(
            "byte MLP LoRA split requires at least two window samples",
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
        return Err(contract_error(
            "byte MLP LoRA split produced an empty partition",
        ));
    }
    Ok((train, validation))
}

fn scale_parameters<M: Module>(model: &mut M, factor: f32) -> PureResult<()> {
    model.visit_parameters_mut(&mut |param| {
        for value in param.value_mut().data_mut() {
            *value *= factor;
        }
        Ok(())
    })
}

fn build_dense_model() -> PureResult<Sequential> {
    let mut model = Sequential::new();
    model.push(Linear::new("embed", VOCAB, HIDDEN)?);
    model.push(Relu::new());
    model.push(Linear::new("head", HIDDEN, VOCAB)?);
    scale_parameters(&mut model, 0.003)?;
    Ok(model)
}

fn build_lora_head_model(
    source_state: &HashMap<String, Tensor>,
    source_key_rules: &HashMap<String, StateKeyMapRule>,
    emit_preflight: bool,
) -> PureResult<(Sequential, StateLoadReport, StateLoadReport)> {
    let mut embed = Linear::new("embed", VOCAB, HIDDEN)?;
    let embed_compatibility =
        embed.state_dict_compatibility_with_key_rules(source_state, source_key_rules)?;
    if emit_preflight {
        print_preflight_report("byte_mlp_lora_embed", &embed_compatibility);
    }
    if !embed_compatibility.compatible {
        return Err(contract_error(format!(
            "byte MLP LoRA embed checkpoint incompatible: missing={} shape_mismatched={}",
            embed_compatibility.missing, embed_compatibility.shape_mismatched
        )));
    }
    let embed_load =
        embed.load_state_dict_subset_adapted_checked(source_state, source_key_rules)?;
    embed.set_trainable(false)?;

    let mut head = LoraLinear::new("head", HIDDEN, VOCAB, LORA_RANK, LORA_ALPHA)?;
    let head_compatibility =
        head.base_state_dict_compatibility_with_key_rules(source_state, source_key_rules)?;
    if emit_preflight {
        print_preflight_report("byte_mlp_lora_head_base", &head_compatibility);
    }
    if !head_compatibility.compatible {
        return Err(contract_error(format!(
            "byte MLP LoRA head checkpoint incompatible: missing={} shape_mismatched={}",
            head_compatibility.missing, head_compatibility.shape_mismatched
        )));
    }
    let head_load = head.load_base_from_state_dict_adapted(source_state, source_key_rules)?;

    let mut model = Sequential::new();
    model.push(embed);
    model.push(Relu::new());
    model.push(head);
    Ok((model, embed_load, head_load))
}

fn externalize_mlp_state(
    source_state: &HashMap<String, Tensor>,
) -> (HashMap<String, Tensor>, HashMap<String, StateKeyMapRule>) {
    let external_state = HashMap::from([
        (
            "external.embed.weight".to_string(),
            source_state["embed::weight"].clone(),
        ),
        (
            "external.embed.bias".to_string(),
            source_state["embed::bias"].clone(),
        ),
        (
            "external.lm_head.weight".to_string(),
            source_state["head::weight"].transpose(),
        ),
        (
            "external.lm_head.bias".to_string(),
            source_state["head::bias"].clone(),
        ),
    ]);
    let key_rules = HashMap::from([
        (
            "external.embed.weight".to_string(),
            StateKeyMapRule::new("embed::weight"),
        ),
        (
            "external.embed.bias".to_string(),
            StateKeyMapRule::new("embed::bias"),
        ),
        (
            "external.lm_head.weight".to_string(),
            StateKeyMapRule::with_transform("head::weight", StateTensorTransform::Transpose),
        ),
        (
            "external.lm_head.bias".to_string(),
            StateKeyMapRule::new("head::bias"),
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
        return Err(contract_error("byte MLP LoRA evaluation received no rows"));
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
    let source_text = "spiraltorch adapters inherit byte memories; adapters inherit byte memories";
    let reload_text = "螺旋adapterは小さくFTできるbyte";
    let source_samples = byte_lm_windows(source_text, CONTEXT)?;
    let reload_samples = byte_lm_windows(reload_text, CONTEXT)?;
    let source_sample_stats = validate_byte_lm_samples(&source_samples, None)?;
    let reload_sample_stats = validate_byte_lm_samples(&reload_samples, None)?;
    let (source_train, source_validation) = train_validation_split(&source_samples)?;
    let (reload_train, _reload_validation) = train_validation_split(&reload_samples)?;

    let caps = DeviceCaps::wgpu(32, true, 256);
    let mut trainer = ModuleTrainer::new(caps, -1.0, 0.5, 0.1);
    trainer.set_max_grad_norm(Some(2.0))?;
    trainer.set_gradient_accumulation_steps(ACCUMULATION_STEPS)?;

    let mut source = build_dense_model()?;
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
    require_loss_drop(
        "dense MLP source",
        source_before.mean_loss,
        source_after.mean_loss,
    )?;

    let checkpoint = std::env::temp_dir().join(format!(
        "spiraltorch_byte_mlp_lora_adapter_contract_{}.json",
        std::process::id()
    ));
    save_json(&source, &checkpoint)?;
    let mut reloaded_source = build_dense_model()?;
    trainer.prepare(&mut reloaded_source)?;
    let dense_load = load_json_checked(&mut reloaded_source, &checkpoint)?;
    let _ = std::fs::remove_file(&checkpoint);
    if !dense_load.matched {
        return Err(contract_error(
            "byte MLP LoRA dense checkpoint fingerprint mismatch",
        ));
    }
    let source_state = reloaded_source.state_dict()?;
    let (external_source_state, source_key_rules) = externalize_mlp_state(&source_state);

    let (mut target, embed_load, head_load) =
        build_lora_head_model(&external_source_state, &source_key_rules, true)?;
    if !embed_load.matched || !head_load.matched {
        return Err(contract_error(
            "byte MLP LoRA base load fingerprint mismatch",
        ));
    }
    trainer.prepare(&mut target)?;
    let frozen_base = target.set_parameters_trainable_by_suffix("::weight", false)?
        + target.set_parameters_trainable_by_suffix("::bias", false)?;
    let boosted_adapter = target.set_parameters_learning_rate_scale_by_contains("lora_", 4.0)?;

    let ft_ready_state = target.state_dict()?;
    let ft_ready_resume = trainer.resume_fingerprint(&target)?;
    let (mut resumed, _, _) =
        build_lora_head_model(&external_source_state, &source_key_rules, false)?;
    trainer.prepare(&mut resumed)?;
    resumed.set_parameters_trainable_by_suffix("::weight", false)?;
    resumed.set_parameters_trainable_by_suffix("::bias", false)?;
    resumed.set_parameters_learning_rate_scale_by_contains("lora_", 4.0)?;
    let resume_load = resumed.load_state_dict_checked(&ft_ready_state)?;
    let resume_audit = trainer.audit_resume_fingerprint(&resumed, &ft_ready_resume)?;
    if !resume_load.matched || !resume_audit.matched {
        return Err(contract_error(
            "byte MLP LoRA FT-ready resume fingerprint mismatch",
        ));
    }

    let ft_report = train_best_epochs_with_finetune_report(
        &mut trainer,
        &mut target,
        &reload_train,
        &reload_samples,
        &source_samples,
        FT_EPOCHS,
    )?;
    let ft_best = &ft_report.captured;
    if ft_best.guarded_best_epoch.is_none() {
        return Err(contract_error(
            "byte MLP LoRA retention guard rejected every fine-tune epoch",
        ));
    }
    let ft_delta = ft_report.target_delta;
    let retention_delta = ft_report.retention_delta;
    let ft_summary = ft_report.summary();
    require_loss_drop(
        "MLP LoRA adapter fine-tune",
        ft_delta.before.mean_loss,
        ft_delta.after.mean_loss,
    )?;

    let movement = &ft_report.movement;
    if !movement.frozen_stable() {
        return Err(contract_error("byte MLP LoRA frozen base moved"));
    }
    if !movement.trainable_movement_observed() {
        return Err(contract_error(
            "byte MLP LoRA adapter parameters did not move",
        ));
    }

    println!(
        "vocab_mode=byte vocab={VOCAB} hidden={HIDDEN} context={CONTEXT} adapter_rank={LORA_RANK} adapter_alpha={LORA_ALPHA:.3}"
    );
    println!(
        "source_bytes={} reload_bytes={} source_windows={} reload_windows={}",
        source_text.len(),
        reload_text.len(),
        source_samples.len(),
        reload_samples.len()
    );
    println!(
        "byte_mlp_lora_sample_stats source_samples={} source_rows={} source_active_rows={} reload_samples={} reload_rows={} reload_active_rows={}",
        source_sample_stats.samples,
        source_sample_stats.total_rows,
        source_sample_stats.active_rows,
        reload_sample_stats.samples,
        reload_sample_stats.total_rows,
        reload_sample_stats.active_rows
    );
    println!(
        "source_epochs={SOURCE_EPOCHS} ft_epochs={FT_EPOCHS} accumulation_steps={ACCUMULATION_STEPS} frozen_base_params={} boosted_adapter_params={} source_batches={} source_optimizer_steps={} ft_batches={} ft_optimizer_steps={} source_epochs_run={} ft_epochs_run={} source_early_stopped={} ft_early_stopped={} source_lr_decay_steps={} ft_lr_decay_steps={} guard_accepted_epochs={} guard_retention_rejected_epochs={} guard_target_stale_epochs={}",
        frozen_base,
        boosted_adapter,
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
        ft_best.guard_accepted_epochs,
        ft_best.guard_retention_rejected_epochs,
        ft_best.guard_target_stale_epochs
    );
    println!(
        "source_metric_delta loss_delta={:.6} accuracy_delta={:.6} perplexity_delta={:.6}",
        source_delta.loss_delta, source_delta.accuracy_delta, source_delta.perplexity_delta
    );
    println!(
        "ft_metric_delta loss_delta={:.6} accuracy_delta={:.6} perplexity_delta={:.6}",
        ft_delta.loss_delta, ft_delta.accuracy_delta, ft_delta.perplexity_delta
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
        "dense_hash={} dense_loaded_hash={} dense_load_matched={} embed_hash={} embed_loaded_hash={} embed_load_matched={} head_hash={} head_loaded_hash={} head_load_matched={}",
        dense_load.source.hash,
        dense_load.loaded.hash,
        dense_load.matched,
        embed_load.source.hash,
        embed_load.loaded.hash,
        embed_load.matched,
        head_load.source.hash,
        head_load.loaded.hash,
        head_load.matched
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
