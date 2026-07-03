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

//! Honest tokenizerless byte-LM comparison for Z-space routing.
//!
//! The contract runs the same small byte MLP through a plain Euclidean baseline
//! and a small sweep of stateless `ZSpaceProjector` strengths/placements/curvatures.
//! It proves each route can source-train, reload checked checkpoints, perform
//! head-only fine-tuning, and requires at least one Z-space route to beat the
//! baseline on aggregate source and FT row-weighted loss deltas across multiple
//! source/target corpus pairs.

use st_core::backend::device_caps::DeviceCaps;
use st_nn::{
    byte_lm_windows, dataset_from_vec, interleave_replay_samples, load_json_checked, save_json,
    EarlyStoppingConfig, EpochValidationBestState, LanguageWaveEncoder, Linear, LrPlateauConfig,
    Module, ModuleTrainer, OpenCartesianTopos, Relu, RoundtableConfig, Sequential,
    SoftmaxCrossEntropy, SparseClassificationDelta, SparseClassificationMetrics,
    SparseFineTuneRegressionLimits, SparseFineTuneReport, SparseFineTuneReportSummary,
    SparseRetentionGuardConfig, Tensor, ValidationTrainingControls, ZSpaceProjector, BYTE_LM_VOCAB,
};
use st_tensor::pure::{PureResult, TensorError};

const VOCAB: usize = BYTE_LM_VOCAB;
const HIDDEN: usize = 24;
const CONTEXT: usize = 8;
const BATCH_WINDOWS: usize = 4;
const ACCUMULATION_STEPS: usize = 2;
const SOURCE_EPOCHS: usize = 4;
const FT_EPOCHS: usize = 5;
const EARLY_STOPPING_PATIENCE: usize = 2;
const EARLY_STOPPING_MIN_DELTA: f32 = 0.0;
const LR_PLATEAU_PATIENCE: usize = 2;
const LR_PLATEAU_FACTOR: f32 = 0.8;
const LR_PLATEAU_MIN_DELTA: f32 = 0.0;
const FT_RETENTION_MAX_LOSS_INCREASE: f32 = 0.5;
const FT_RETENTION_MAX_ACCURACY_DROP: f32 = 0.25;
const FT_RETENTION_MAX_PERPLEXITY_INCREASE: f32 = 1.0;
const FT_TARGET_MIN_LOSS_DELTA: f32 = 1e-4;
const FT_MOVEMENT_TOLERANCE: f32 = 1e-6;
const PARAM_SCALE: f32 = 0.003;
const ZSPACE_CURVATURE: f32 = -1.0;
const BASELINE_ADVANTAGE_EPS: f32 = 1e-5;
const SMOKE_ENV: &str = "SPIRALTORCH_BYTE_LM_ZSPACE_SMOKE";

#[derive(Debug, Clone, Copy)]
enum ProjectorPlacement {
    PreRelu,
    PostRelu,
}

impl ProjectorPlacement {
    fn label(self) -> &'static str {
        match self {
            ProjectorPlacement::PreRelu => "pre_relu",
            ProjectorPlacement::PostRelu => "post_relu",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct RouteSpec {
    label: &'static str,
    zspace_strength: Option<f32>,
    placement: Option<ProjectorPlacement>,
    curvature: f32,
    param_scale: f32,
}

impl RouteSpec {
    fn baseline() -> Self {
        Self {
            label: "baseline",
            zspace_strength: None,
            placement: None,
            curvature: ZSPACE_CURVATURE,
            param_scale: PARAM_SCALE,
        }
    }

    fn zspace(label: &'static str, strength: f32) -> Self {
        Self::zspace_at(
            label,
            strength,
            ProjectorPlacement::PostRelu,
            ZSPACE_CURVATURE,
        )
    }

    fn zspace_at(
        label: &'static str,
        strength: f32,
        placement: ProjectorPlacement,
        curvature: f32,
    ) -> Self {
        Self {
            label,
            zspace_strength: Some(strength),
            placement: Some(placement),
            curvature,
            param_scale: PARAM_SCALE,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CorpusCase {
    label: &'static str,
    source_text: &'static str,
    reload_text: &'static str,
}

#[derive(Debug, Clone)]
struct RouteReport {
    case_label: String,
    label: String,
    zspace_strength: Option<f32>,
    placement: Option<ProjectorPlacement>,
    curvature: f32,
    param_scale: f32,
    source_windows: usize,
    reload_windows: usize,
    source: SparseClassificationDelta,
    ft: SparseClassificationDelta,
    retention: SparseClassificationDelta,
    source_best_epoch: Option<usize>,
    ft_guarded_best_epoch: Option<usize>,
    ft_retention_loss_ceiling: f32,
    ft_retention_accuracy_floor: f32,
    ft_retention_perplexity_ceiling: Option<f32>,
    ft_best_retention_loss_increase: f32,
    ft_best_retention_accuracy_drop: f32,
    ft_best_retention_perplexity_increase: f32,
    source_batches: usize,
    ft_batches: usize,
    source_optimizer_steps: usize,
    ft_optimizer_steps: usize,
    source_epochs_run: usize,
    ft_epochs_run: usize,
    source_early_stopped: bool,
    ft_early_stopped: bool,
    source_lr_decay_steps: usize,
    ft_lr_decay_steps: usize,
    source_final_hyper_lr: f32,
    ft_final_hyper_lr: f32,
    frozen_embed_params: usize,
    boosted_head_params: usize,
    source_hash: String,
    loaded_hash: String,
    source_best_hash: String,
    source_final_hash: String,
    source_best_differs_from_final: bool,
    ft_best_hash: String,
    ft_final_hash: String,
    ft_best_differs_from_final: bool,
    ft_summary: SparseFineTuneReportSummary,
    ft_report_status: &'static str,
    movement_status: &'static str,
}

impl RouteReport {
    fn source_delta(&self) -> f32 {
        self.source.loss_delta
    }

    fn ft_delta(&self) -> f32 {
        self.ft.loss_delta
    }

    fn source_accuracy_delta(&self) -> f32 {
        self.source.accuracy_delta
    }

    fn ft_accuracy_delta(&self) -> f32 {
        self.ft.accuracy_delta
    }

    fn source_perplexity_delta(&self) -> f32 {
        self.source.perplexity_delta
    }

    fn ft_perplexity_delta(&self) -> f32 {
        self.ft.perplexity_delta
    }

    fn retention_delta(&self) -> f32 {
        self.retention.loss_delta
    }

    fn retention_accuracy_delta(&self) -> f32 {
        self.retention.accuracy_delta
    }

    fn retention_perplexity_delta(&self) -> f32 {
        self.retention.perplexity_delta
    }

    fn target_loss_margin(&self) -> f32 {
        self.ft_summary.target_loss_margin
    }

    fn retention_loss_margin(&self) -> f32 {
        self.ft_summary.retention_loss_margin
    }

    fn retention_accuracy_margin(&self) -> f32 {
        self.ft_summary.retention_accuracy_margin
    }
}

impl RouteAggregate {
    fn source_delta(&self) -> f32 {
        self.source_loss_delta
    }

    fn ft_delta(&self) -> f32 {
        self.ft_loss_delta
    }

    fn retention_delta(&self) -> f32 {
        self.retention_loss_delta
    }

    fn source_accuracy_delta(&self) -> f32 {
        self.source_accuracy_delta
    }

    fn ft_accuracy_delta(&self) -> f32 {
        self.ft_accuracy_delta
    }

    fn retention_accuracy_delta(&self) -> f32 {
        self.retention_accuracy_delta
    }

    fn source_perplexity_delta(&self) -> f32 {
        self.source_perplexity_delta
    }

    fn ft_perplexity_delta(&self) -> f32 {
        self.ft_perplexity_delta
    }

    fn retention_perplexity_delta(&self) -> f32 {
        self.retention_perplexity_delta
    }

    fn target_loss_margin_min(&self) -> f32 {
        self.target_loss_margin_min
    }

    fn retention_loss_margin_min(&self) -> f32 {
        self.retention_loss_margin_min
    }

    fn retention_accuracy_margin_min(&self) -> f32 {
        self.retention_accuracy_margin_min
    }
}

#[derive(Debug, Clone)]
struct RouteAggregate {
    label: String,
    is_zspace: bool,
    cases: usize,
    case_labels: String,
    accepted_cases: usize,
    accepted_rate: f32,
    accepted_all: bool,
    movement_ok_cases: usize,
    movement_ok_rate: f32,
    movement_ok_all: bool,
    source_loss_delta: f32,
    source_accuracy_delta: f32,
    source_perplexity_delta: f32,
    ft_loss_delta: f32,
    ft_accuracy_delta: f32,
    ft_perplexity_delta: f32,
    retention_loss_delta: f32,
    retention_accuracy_delta: f32,
    retention_perplexity_delta: f32,
    target_loss_margin_mean: f32,
    target_loss_margin_min: f32,
    retention_loss_margin_mean: f32,
    retention_loss_margin_min: f32,
    retention_accuracy_margin_mean: f32,
    retention_accuracy_margin_min: f32,
}

fn train_validation_split(
    samples: &[(Tensor, Tensor)],
) -> PureResult<(Vec<(Tensor, Tensor)>, Vec<(Tensor, Tensor)>)> {
    if samples.len() < 2 {
        return Err(contract_error(
            "byte compare split requires at least two window samples",
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
            "byte compare split produced an empty partition",
        ));
    }
    Ok((train, validation))
}

fn build_model(spec: RouteSpec) -> PureResult<Sequential> {
    let mut model = Sequential::new();
    model.push(Linear::new("embed", VOCAB, HIDDEN)?);
    if matches!(spec.placement, Some(ProjectorPlacement::PreRelu)) {
        model.push(zspace_projector(
            spec.curvature,
            spec.zspace_strength.unwrap_or(1.0),
        )?);
    }
    model.push(Relu::new());
    if matches!(spec.placement, Some(ProjectorPlacement::PostRelu)) {
        model.push(zspace_projector(
            spec.curvature,
            spec.zspace_strength.unwrap_or(1.0),
        )?);
    }
    model.push(Linear::new("head", HIDDEN, VOCAB)?);
    scale_parameters(&mut model, spec.param_scale)?;
    Ok(model)
}

fn zspace_projector(curvature: f32, strength: f32) -> PureResult<ZSpaceProjector> {
    let topos = OpenCartesianTopos::new(curvature, 1e-5, 10.0, 256, 16384)?;
    let encoder = LanguageWaveEncoder::new(topos.curvature(), 0.65)?;
    ZSpaceProjector::with_strength(topos, encoder, strength)
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

fn train_best_epochs_with_sparse_retention<M: Module>(
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
        return Err(contract_error("byte compare evaluation received no rows"));
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

fn run_route(
    case: CorpusCase,
    spec: RouteSpec,
    source_samples: &[(Tensor, Tensor)],
    reload_samples: &[(Tensor, Tensor)],
) -> PureResult<RouteReport> {
    let caps = DeviceCaps::wgpu(32, true, 256);
    let mut trainer = ModuleTrainer::new(caps, -1.0, 0.08, 0.02);
    trainer.set_max_grad_norm(Some(0.75))?;
    trainer.set_gradient_accumulation_steps(ACCUMULATION_STEPS)?;
    let (source_train, source_validation) = train_validation_split(source_samples)?;
    let (reload_train, reload_validation) = train_validation_split(reload_samples)?;

    let mut source = build_model(spec)?;
    trainer.prepare(&mut source)?;
    let source_before = evaluate_metrics(&trainer, &source, source_samples)?;
    let source_best = train_best_epochs(
        &mut trainer,
        &mut source,
        &source_train,
        &source_validation,
        SOURCE_EPOCHS,
    )?;
    let source_after = evaluate_metrics(&trainer, &source, source_samples)?;
    require_loss_drop(
        &format!("{} source", spec.label),
        source_before.mean_loss,
        source_after.mean_loss,
    )?;

    let checkpoint = std::env::temp_dir().join(format!(
        "spiraltorch_byte_compare_{}_{}_{}.json",
        case.label,
        spec.label,
        std::process::id()
    ));
    save_json(&source, &checkpoint)?;

    let mut target = build_model(spec)?;
    trainer.prepare(&mut target)?;
    let load = load_json_checked(&mut target, &checkpoint)?;
    let _ = std::fs::remove_file(&checkpoint);
    if !load.matched {
        return Err(contract_error(format!(
            "{} checkpoint fingerprint mismatch",
            spec.label
        )));
    }

    let frozen_embed_params = target.set_parameters_trainable_by_prefix("embed::", false)?;
    let boosted_head_params = target.set_parameters_learning_rate_scale_by_prefix("head::", 1.1)?;
    let ft_train = interleave_replay_samples(&reload_train, &source_train, 1)?;
    let ft_report = train_best_epochs_with_sparse_retention(
        &mut trainer,
        &mut target,
        &ft_train,
        &reload_validation,
        source_samples,
        FT_EPOCHS,
    )?;
    let ft_summary = ft_report.summary();
    let ft_best = &ft_report.captured;
    require_loss_drop(
        &format!("{} head-only fine-tune", spec.label),
        ft_report.target_delta.before.mean_loss,
        ft_report.target_delta.after.mean_loss,
    )?;

    let movement = &ft_report.movement;
    if !movement.frozen_stable() {
        return Err(contract_error(format!(
            "{} frozen embedding moved during fine-tune",
            spec.label
        )));
    }
    if !movement.trainable_movement_observed() {
        return Err(contract_error(format!(
            "{} trainable head did not move",
            spec.label
        )));
    }

    Ok(RouteReport {
        case_label: case.label.to_string(),
        label: spec.label.to_string(),
        zspace_strength: spec.zspace_strength,
        placement: spec.placement,
        curvature: spec.curvature,
        param_scale: spec.param_scale,
        source_windows: source_samples.len(),
        reload_windows: reload_samples.len(),
        source: source_before.delta_to(source_after),
        ft: ft_report.target_delta,
        retention: ft_report.retention_delta,
        source_best_epoch: source_best.validation_summary.best_epoch,
        ft_guarded_best_epoch: ft_best.guarded_best_epoch,
        ft_retention_loss_ceiling: ft_best.max_allowed_retention_loss,
        ft_retention_accuracy_floor: ft_best.min_allowed_retention_accuracy,
        ft_retention_perplexity_ceiling: ft_best.max_allowed_retention_perplexity,
        ft_best_retention_loss_increase: ft_best.best_retention_loss_increase,
        ft_best_retention_accuracy_drop: ft_best.best_retention_accuracy_drop,
        ft_best_retention_perplexity_increase: ft_best.best_retention_perplexity_increase,
        source_batches: source_best.train_summary.batches,
        ft_batches: ft_best.train_summary.batches,
        source_optimizer_steps: source_best.train_summary.optimizer_steps,
        ft_optimizer_steps: ft_best.train_summary.optimizer_steps,
        source_epochs_run: source_best.train_summary.epochs,
        ft_epochs_run: ft_best.train_summary.epochs,
        source_early_stopped: source_best.early_stopped,
        ft_early_stopped: ft_best.early_stopped,
        source_lr_decay_steps: source_best.lr_decay_steps,
        ft_lr_decay_steps: ft_best.lr_decay_steps,
        source_final_hyper_lr: source_best.final_hyper_learning_rate,
        ft_final_hyper_lr: ft_best.final_hyper_learning_rate,
        frozen_embed_params,
        boosted_head_params,
        source_hash: load.source.hash,
        loaded_hash: load.loaded.hash,
        source_best_hash: source_best.best_fingerprint.hash,
        source_final_hash: source_best.final_fingerprint.hash,
        source_best_differs_from_final: source_best.best_differs_from_final,
        ft_best_hash: ft_best.best_fingerprint.hash.clone(),
        ft_final_hash: ft_best.final_fingerprint.hash.clone(),
        ft_best_differs_from_final: ft_best.best_differs_from_final,
        ft_summary,
        ft_report_status: ft_report.status(),
        movement_status: movement.status(),
    })
}

fn aggregate_reports(
    reports: &[RouteReport],
    specs: &[RouteSpec],
) -> PureResult<Vec<RouteAggregate>> {
    let mut aggregates = Vec::with_capacity(specs.len());
    for spec in specs {
        let selected: Vec<&RouteReport> = reports
            .iter()
            .filter(|report| report.label == spec.label)
            .collect();
        if selected.is_empty() {
            return Err(contract_error(format!(
                "zspace compare missing reports for route {}",
                spec.label
            )));
        }
        let denom = selected.len() as f32;
        let mean = |metric: fn(&RouteReport) -> f32| {
            selected.iter().map(|report| metric(report)).sum::<f32>() / denom
        };
        let min = |metric: fn(&RouteReport) -> f32| {
            selected
                .iter()
                .map(|report| metric(report))
                .fold(f32::INFINITY, f32::min)
        };
        let case_labels = selected
            .iter()
            .map(|report| report.case_label.as_str())
            .collect::<Vec<_>>()
            .join("|");
        let accepted_cases = selected
            .iter()
            .filter(|report| report.ft_summary.accepted)
            .count();
        let movement_ok_cases = selected
            .iter()
            .filter(|report| report.ft_summary.movement_ok)
            .count();

        aggregates.push(RouteAggregate {
            label: spec.label.to_string(),
            is_zspace: spec.zspace_strength.is_some(),
            cases: selected.len(),
            case_labels,
            accepted_cases,
            accepted_rate: accepted_cases as f32 / denom,
            accepted_all: accepted_cases == selected.len(),
            movement_ok_cases,
            movement_ok_rate: movement_ok_cases as f32 / denom,
            movement_ok_all: movement_ok_cases == selected.len(),
            source_loss_delta: mean(RouteReport::source_delta),
            source_accuracy_delta: mean(RouteReport::source_accuracy_delta),
            source_perplexity_delta: mean(RouteReport::source_perplexity_delta),
            ft_loss_delta: mean(RouteReport::ft_delta),
            ft_accuracy_delta: mean(RouteReport::ft_accuracy_delta),
            ft_perplexity_delta: mean(RouteReport::ft_perplexity_delta),
            retention_loss_delta: mean(RouteReport::retention_delta),
            retention_accuracy_delta: mean(RouteReport::retention_accuracy_delta),
            retention_perplexity_delta: mean(RouteReport::retention_perplexity_delta),
            target_loss_margin_mean: mean(RouteReport::target_loss_margin),
            target_loss_margin_min: min(RouteReport::target_loss_margin),
            retention_loss_margin_mean: mean(RouteReport::retention_loss_margin),
            retention_loss_margin_min: min(RouteReport::retention_loss_margin),
            retention_accuracy_margin_mean: mean(RouteReport::retention_accuracy_margin),
            retention_accuracy_margin_min: min(RouteReport::retention_accuracy_margin),
        });
    }
    Ok(aggregates)
}

fn winner_by_aggregate_metric<F>(aggregates: &[RouteAggregate], metric: F) -> String
where
    F: Fn(&RouteAggregate) -> f32 + Copy,
{
    if aggregates.is_empty() {
        return "none".to_string();
    }
    let best = aggregates
        .iter()
        .map(metric)
        .fold(f32::NEG_INFINITY, f32::max);
    let winners: Vec<&str> = aggregates
        .iter()
        .filter(|aggregate| {
            let value = metric(aggregate);
            (value - best).abs() <= 1e-6
        })
        .map(|aggregate| aggregate.label.as_str())
        .collect();
    if winners.len() == aggregates.len() {
        "tie".to_string()
    } else {
        winners.join("+")
    }
}

fn winner_by_aggregate_delta(aggregates: &[RouteAggregate], ft: bool) -> String {
    winner_by_aggregate_metric(aggregates, |aggregate| {
        if ft {
            aggregate.ft_delta()
        } else {
            aggregate.source_delta()
        }
    })
}

fn winner_by_aggregate_accuracy_delta(aggregates: &[RouteAggregate], ft: bool) -> String {
    winner_by_aggregate_metric(aggregates, |aggregate| {
        if ft {
            aggregate.ft_accuracy_delta()
        } else {
            aggregate.source_accuracy_delta()
        }
    })
}

fn winner_by_aggregate_perplexity_delta(aggregates: &[RouteAggregate], ft: bool) -> String {
    winner_by_aggregate_metric(aggregates, |aggregate| {
        if ft {
            aggregate.ft_perplexity_delta()
        } else {
            aggregate.source_perplexity_delta()
        }
    })
}

fn best_zspace_aggregate_delta(aggregates: &[RouteAggregate], ft: bool) -> Option<&RouteAggregate> {
    aggregates
        .iter()
        .filter(|aggregate| aggregate.is_zspace)
        .max_by(|left, right| {
            let lhs = if ft {
                left.ft_delta()
            } else {
                left.source_delta()
            };
            let rhs = if ft {
                right.ft_delta()
            } else {
                right.source_delta()
            };
            lhs.partial_cmp(&rhs).unwrap_or(std::cmp::Ordering::Equal)
        })
}

fn require_zspace_aggregate_advantage(
    aggregates: &[RouteAggregate],
) -> PureResult<(&RouteAggregate, &RouteAggregate)> {
    let baseline = aggregates
        .iter()
        .find(|aggregate| aggregate.label == "baseline")
        .ok_or_else(|| contract_error("zspace compare missing aggregate baseline route"))?;
    let best_source = best_zspace_aggregate_delta(aggregates, false)
        .ok_or_else(|| contract_error("zspace compare missing aggregate source zspace route"))?;
    let best_ft = best_zspace_aggregate_delta(aggregates, true)
        .ok_or_else(|| contract_error("zspace compare missing aggregate FT zspace route"))?;

    let source_advantage = best_source.source_delta() - baseline.source_delta();
    let ft_advantage = best_ft.ft_delta() - baseline.ft_delta();
    if source_advantage <= BASELINE_ADVANTAGE_EPS {
        return Err(contract_error(format!(
            "best aggregate zspace source delta did not beat baseline: baseline={:.6} best={} delta={:.6} advantage={:.6}",
            baseline.source_delta(),
            best_source.label,
            best_source.source_delta(),
            source_advantage
        )));
    }
    if ft_advantage <= BASELINE_ADVANTAGE_EPS {
        return Err(contract_error(format!(
            "best aggregate zspace FT delta did not beat baseline: baseline={:.6} best={} delta={:.6} advantage={:.6}",
            baseline.ft_delta(),
            best_ft.label,
            best_ft.ft_delta(),
            ft_advantage
        )));
    }
    Ok((best_source, best_ft))
}

fn contract_error(message: impl Into<String>) -> TensorError {
    TensorError::IoError {
        message: message.into(),
    }
}

fn smoke_enabled() -> bool {
    std::env::var(SMOKE_ENV)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn main() -> PureResult<()> {
    let smoke = smoke_enabled();
    let all_cases = [
        CorpusCase {
            label: "byte_patterns_to_jp",
            source_text:
                "spiraltorch learns byte patterns; spiraltorch learns byte patterns; byte byte byte",
            reload_text: "螺旋byte螺旋byte螺旋byteは壊れないbyte",
        },
        CorpusCase {
            label: "routes_to_cats",
            source_text: "graphs route tensors; routes graph byte",
            reload_text: "猫byte螺旋byte",
        },
    ];
    let all_specs = [
        RouteSpec::baseline(),
        RouteSpec::zspace("zspace_s025", 0.25),
        RouteSpec::zspace("zspace_s050", 0.50),
        RouteSpec::zspace("zspace_s100", 1.00),
        RouteSpec::zspace_at(
            "zspace_pre_s050",
            0.50,
            ProjectorPlacement::PreRelu,
            ZSPACE_CURVATURE,
        ),
        RouteSpec::zspace_at(
            "zspace_post_s050_c025",
            0.50,
            ProjectorPlacement::PostRelu,
            -0.25,
        ),
    ];
    let cases: Vec<CorpusCase> = if smoke {
        all_cases.to_vec()
    } else {
        all_cases.to_vec()
    };
    let specs: Vec<RouteSpec> = if smoke {
        vec![RouteSpec::baseline(), all_specs[5]]
    } else {
        all_specs.to_vec()
    };
    let mut reports = Vec::with_capacity(cases.len() * specs.len());
    for case in &cases {
        let source_samples = byte_lm_windows(case.source_text, CONTEXT)?;
        let reload_samples = byte_lm_windows(case.reload_text, CONTEXT)?;
        for spec in specs.iter().copied() {
            reports.push(run_route(*case, spec, &source_samples, &reload_samples)?);
        }
    }
    let aggregates = aggregate_reports(&reports, &specs)?;
    let (best_source_zspace, best_ft_zspace) = if smoke {
        (
            best_zspace_aggregate_delta(&aggregates, false).ok_or_else(|| {
                contract_error("zspace smoke missing aggregate source zspace route")
            })?,
            best_zspace_aggregate_delta(&aggregates, true)
                .ok_or_else(|| contract_error("zspace smoke missing aggregate FT zspace route"))?,
        )
    } else {
        require_zspace_aggregate_advantage(&aggregates)?
    };

    println!(
        "compare=byte_lm_zspace_sweep smoke={} vocab={VOCAB} hidden={HIDDEN} context={CONTEXT} cases={} routes={}",
        smoke,
        cases.len(),
        specs.len()
    );
    println!(
        "source_epochs={SOURCE_EPOCHS} ft_epochs={FT_EPOCHS} accumulation_steps={ACCUMULATION_STEPS} early_stopping_patience={EARLY_STOPPING_PATIENCE} early_stopping_min_delta={EARLY_STOPPING_MIN_DELTA:.6} lr_plateau_patience={LR_PLATEAU_PATIENCE} lr_plateau_factor={LR_PLATEAU_FACTOR:.3} lr_plateau_min_delta={LR_PLATEAU_MIN_DELTA:.6} ft_retention_max_loss_increase={FT_RETENTION_MAX_LOSS_INCREASE:.6} ft_retention_max_accuracy_drop={FT_RETENTION_MAX_ACCURACY_DROP:.6} ft_retention_max_perplexity_increase={FT_RETENTION_MAX_PERPLEXITY_INCREASE:.6} ft_target_min_loss_delta={FT_TARGET_MIN_LOSS_DELTA:.6} ft_movement_tolerance={FT_MOVEMENT_TOLERANCE:.6}"
    );
    println!("corpus_case_header=case,source_windows,reload_windows,source_bytes,reload_bytes");
    for case in &cases {
        let report = reports
            .iter()
            .find(|report| report.case_label == case.label)
            .ok_or_else(|| contract_error(format!("missing corpus report for {}", case.label)))?;
        println!(
            "corpus_case={},{},{},{},{}",
            case.label,
            report.source_windows,
            report.reload_windows,
            case.source_text.len(),
            case.reload_text.len()
        );
    }
    let summary_compare_limits = SparseFineTuneRegressionLimits::new()
        .with_max_target_loss_regression(0.0)?
        .with_max_retention_loss_regression(0.0)?
        .with_status_match_required(true)
        .with_accepted_match_required(true)
        .with_guard_match_required(true)
        .with_movement_tolerance_match_required(true);
    println!(
        "ft_summary_compare_header=case,route,baseline,target_loss_delta_change,retention_loss_delta_change,target_loss_regression,retention_loss_regression,status_changed,accepted_changed,guard_changed,movement_tolerance_changed,resume_changed,passed"
    );
    for report in &reports {
        println!(
            "case={} route={} zspace_strength={} placement={} curvature={:.3} param_scale={:.4} source_windows={} reload_windows={} source_before={:.6} source_after={:.6} source_delta={:.6} source_accuracy_before={:.6} source_accuracy_after={:.6} source_perplexity_before={:.6} source_perplexity_after={:.6} source_eval_rows={} source_validation_best_epoch={:?} ft_before={:.6} ft_after={:.6} ft_delta={:.6} ft_accuracy_before={:.6} ft_accuracy_after={:.6} ft_perplexity_before={:.6} ft_perplexity_after={:.6} ft_eval_rows={} ft_guarded_best_epoch={:?} ft_report_status={} movement_status={} ft_movement_tolerance={:.6} ft_resume_hash={} ft_resume_trainer_hash={} ft_resume_parameter_training_hash={} retention_before={:.6} retention_after={:.6} retention_delta={:.6} retention_accuracy_before={:.6} retention_accuracy_after={:.6} retention_perplexity_before={:.6} retention_perplexity_after={:.6} retention_eval_rows={} ft_target_min_loss_delta={:.6} ft_target_loss_margin={:.6} ft_retention_loss_margin={:.6} ft_retention_accuracy_margin={:.6} ft_retention_loss_ceiling={:.6} ft_retention_accuracy_floor={:.6} ft_retention_perplexity_ceiling={:.6} ft_best_retention_loss_increase={:.6} ft_best_retention_accuracy_drop={:.6} ft_best_retention_perplexity_increase={:.6} frozen_embed_params={} boosted_head_params={} source_epochs_run={} source_early_stopped={} ft_epochs_run={} ft_early_stopped={} source_lr_decay_steps={} ft_lr_decay_steps={} source_final_hyper_lr={:.6} ft_final_hyper_lr={:.6} source_batches={} source_optimizer_steps={} ft_batches={} ft_optimizer_steps={}",
            report.case_label,
            report.label,
            report
                .zspace_strength
                .map(|strength| format!("{strength:.2}"))
                .unwrap_or_else(|| "none".to_string()),
            report
                .placement
                .map(ProjectorPlacement::label)
                .unwrap_or("none"),
            report.curvature,
            report.param_scale,
            report.source_windows,
            report.reload_windows,
            report.source.before.mean_loss,
            report.source.after.mean_loss,
            report.source_delta(),
            report.source.before.accuracy,
            report.source.after.accuracy,
            report.source.before.perplexity,
            report.source.after.perplexity,
            report.source.after.active_rows,
            report.source_best_epoch,
            report.ft.before.mean_loss,
            report.ft.after.mean_loss,
            report.ft_delta(),
            report.ft.before.accuracy,
            report.ft.after.accuracy,
            report.ft.before.perplexity,
            report.ft.after.perplexity,
            report.ft.after.active_rows,
            report.ft_guarded_best_epoch,
            report.ft_report_status,
            report.movement_status,
            report.ft_summary.movement_tolerance,
            report.ft_summary.resume_hash,
            report.ft_summary.resume_trainer_hash,
            report.ft_summary.resume_parameter_training_hash,
            report.retention.before.mean_loss,
            report.retention.after.mean_loss,
            report.retention_delta(),
            report.retention.before.accuracy,
            report.retention.after.accuracy,
            report.retention.before.perplexity,
            report.retention.after.perplexity,
            report.retention.after.active_rows,
            report.ft_summary.target_min_loss_delta,
            report.ft_summary.target_loss_margin,
            report.ft_summary.retention_loss_margin,
            report.ft_summary.retention_accuracy_margin,
            report.ft_retention_loss_ceiling,
            report.ft_retention_accuracy_floor,
            report.ft_retention_perplexity_ceiling.unwrap_or(0.0),
            report.ft_best_retention_loss_increase,
            report.ft_best_retention_accuracy_drop,
            report.ft_best_retention_perplexity_increase,
            report.frozen_embed_params,
            report.boosted_head_params,
            report.source_epochs_run,
            report.source_early_stopped,
            report.ft_epochs_run,
            report.ft_early_stopped,
            report.source_lr_decay_steps,
            report.ft_lr_decay_steps,
            report.source_final_hyper_lr,
            report.ft_final_hyper_lr,
            report.source_batches,
            report.source_optimizer_steps,
            report.ft_batches,
            report.ft_optimizer_steps
        );
        println!(
            "case={} route={} source_hash={} loaded_hash={} load_matched={} source_best_hash={} source_final_hash={} source_best_differs_from_final={} ft_best_hash={} ft_final_hash={} ft_best_differs_from_final={}",
            report.case_label,
            report.label,
            report.source_hash,
            report.loaded_hash,
            report.source_hash == report.loaded_hash,
            report.source_best_hash,
            report.source_final_hash,
            report.source_best_differs_from_final,
            report.ft_best_hash,
            report.ft_final_hash,
            report.ft_best_differs_from_final
        );
        if report.label != "baseline" {
            let baseline = reports
                .iter()
                .find(|candidate| {
                    candidate.case_label == report.case_label && candidate.label == "baseline"
                })
                .ok_or_else(|| {
                    contract_error(format!(
                        "missing baseline report for case {}",
                        report.case_label
                    ))
                })?;
            let comparison = report
                .ft_summary
                .compare_to(&baseline.ft_summary, summary_compare_limits)?;
            println!(
                "ft_summary_compare={},{},baseline,{:.6},{:.6},{:.6},{:.6},{},{},{},{},{},{}",
                report.case_label,
                report.label,
                comparison.target_loss_delta_change,
                comparison.retention_loss_delta_change,
                comparison.target_loss_regression,
                comparison.retention_loss_regression,
                comparison.status_changed,
                comparison.accepted_changed,
                comparison.guard_changed,
                comparison.movement_tolerance_changed,
                comparison.resume_changed,
                comparison.passed
            );
        }
    }
    println!(
        "route_metric_summary_header=route,cases,case_labels,accepted_cases,accepted_rate,accepted_all,movement_ok_cases,movement_ok_rate,movement_ok_all,source_loss_delta_mean,source_accuracy_delta_mean,source_perplexity_delta_mean,ft_loss_delta_mean,ft_accuracy_delta_mean,ft_perplexity_delta_mean,retention_loss_delta_mean,retention_accuracy_delta_mean,retention_perplexity_delta_mean,target_loss_margin_mean,target_loss_margin_min,retention_loss_margin_mean,retention_loss_margin_min,retention_accuracy_margin_mean,retention_accuracy_margin_min"
    );
    for aggregate in &aggregates {
        println!(
            "route_metric_summary={},{},{},{},{:.6},{},{},{:.6},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            aggregate.label,
            aggregate.cases,
            aggregate.case_labels,
            aggregate.accepted_cases,
            aggregate.accepted_rate,
            aggregate.accepted_all,
            aggregate.movement_ok_cases,
            aggregate.movement_ok_rate,
            aggregate.movement_ok_all,
            aggregate.source_delta(),
            aggregate.source_accuracy_delta(),
            aggregate.source_perplexity_delta(),
            aggregate.ft_delta(),
            aggregate.ft_accuracy_delta(),
            aggregate.ft_perplexity_delta(),
            aggregate.retention_delta(),
            aggregate.retention_accuracy_delta(),
            aggregate.retention_perplexity_delta(),
            aggregate.target_loss_margin_mean,
            aggregate.target_loss_margin_min,
            aggregate.retention_loss_margin_mean,
            aggregate.retention_loss_margin_min,
            aggregate.retention_accuracy_margin_mean,
            aggregate.retention_accuracy_margin_min
        );
    }
    println!(
        "source_delta_winner={} ft_delta_winner={} retention_delta_winner={}",
        winner_by_aggregate_delta(&aggregates, false),
        winner_by_aggregate_delta(&aggregates, true),
        winner_by_aggregate_metric(&aggregates, RouteAggregate::retention_delta)
    );
    println!(
        "metric_winners source_loss_delta={} ft_loss_delta={} retention_loss_delta={} source_accuracy_delta={} ft_accuracy_delta={} retention_accuracy_delta={} source_perplexity_delta={} ft_perplexity_delta={} retention_perplexity_delta={}",
        winner_by_aggregate_delta(&aggregates, false),
        winner_by_aggregate_delta(&aggregates, true),
        winner_by_aggregate_metric(&aggregates, RouteAggregate::retention_delta),
        winner_by_aggregate_accuracy_delta(&aggregates, false),
        winner_by_aggregate_accuracy_delta(&aggregates, true),
        winner_by_aggregate_metric(&aggregates, RouteAggregate::retention_accuracy_delta),
        winner_by_aggregate_perplexity_delta(&aggregates, false),
        winner_by_aggregate_perplexity_delta(&aggregates, true),
        winner_by_aggregate_metric(&aggregates, RouteAggregate::retention_perplexity_delta)
    );
    println!(
        "margin_winners target_loss_margin_min={} retention_loss_margin_min={} retention_accuracy_margin_min={}",
        winner_by_aggregate_metric(&aggregates, RouteAggregate::target_loss_margin_min),
        winner_by_aggregate_metric(&aggregates, RouteAggregate::retention_loss_margin_min),
        winner_by_aggregate_metric(&aggregates, RouteAggregate::retention_accuracy_margin_min)
    );
    println!(
        "zspace_aggregate_advantage_source={} advantage={:.6} zspace_aggregate_advantage_ft={} advantage={:.6}",
        best_source_zspace.label,
        best_source_zspace.source_delta()
            - aggregates
                .iter()
                .find(|aggregate| aggregate.label == "baseline")
                .map(RouteAggregate::source_delta)
                .unwrap_or(0.0),
        best_ft_zspace.label,
        best_ft_zspace.ft_delta()
            - aggregates
                .iter()
                .find(|aggregate| aggregate.label == "baseline")
                .map(RouteAggregate::ft_delta)
                .unwrap_or(0.0)
    );

    Ok(())
}
