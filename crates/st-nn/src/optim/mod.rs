// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::Module;
use crate::PureResult;
use st_core::ops::zspace_round::SpectralFeatureSample;
use st_core::runtime::zspace_optimizer::{
    zspace_parameter_control_from_report, ZSpaceMetaOptimizerStepReport, ZSpaceParameterControl,
};
use st_tensor::{emit_tensor_op, emit_tensor_op_meta, TensorError};
use std::f32::consts::PI;

/// Trait implemented by adapters that emit local learning-rate multipliers based on
/// spectral statistics extracted from Z-space gradients.
pub trait LocalLearningRateAdapter {
    /// Returns the preferred sheet hint used when analysing spectral energy.
    fn sheet_hint(&self) -> usize {
        1
    }

    /// Computes a multiplicative factor applied to the current accumulator buffers.
    fn scale_factor(&mut self, parameter: &str, features: &SpectralFeatureSample) -> f32;

    /// Notifies the adapter that the global learning rate has been scaled.
    fn on_global_scale(&mut self, _factor: f32) {}
}

/// Learning-rate adapter driven by spectral characteristics extracted from Z-space
/// gradients. The adapter keeps an exponential moving average of recent features so
/// the emitted multipliers remain smooth.
#[derive(Debug, Clone)]
pub struct SpectralLrAdapter {
    sheet_hint: usize,
    smoothing: f32,
    curvature_target: f32,
    curvature_gain: f32,
    spin_gain: f32,
    energy_gain: f32,
    sheet_gain: f32,
    min_scale: f32,
    max_scale: f32,
    avg_curvature: f32,
    avg_spin: f32,
    avg_energy: f32,
}

/// Serializable summary of [`SpectralLrAdapter`] state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectralLrAdapterState {
    pub sheet_hint: usize,
    pub smoothing: f32,
    pub curvature_target: f32,
    pub curvature_gain: f32,
    pub spin_gain: f32,
    pub energy_gain: f32,
    pub sheet_gain: f32,
    pub min_scale: f32,
    pub max_scale: f32,
    pub avg_curvature: f32,
    pub avg_spin: f32,
    pub avg_energy: f32,
}

impl Default for SpectralLrAdapter {
    fn default() -> Self {
        Self {
            sheet_hint: 8,
            smoothing: 0.2,
            curvature_target: 0.0,
            curvature_gain: 0.3,
            spin_gain: 0.2,
            energy_gain: 0.15,
            sheet_gain: 0.1,
            min_scale: 0.25,
            max_scale: 4.0,
            avg_curvature: 0.0,
            avg_spin: 0.0,
            avg_energy: 0.0,
        }
    }
}

impl SpectralLrAdapter {
    /// Builds a new adapter using default gains.
    pub fn new() -> Self {
        Self::default()
    }

    /// Overrides the number of sheets used when binning spectral energy.
    pub fn with_sheet_hint(mut self, hint: usize) -> Self {
        self.sheet_hint = hint.max(1);
        self
    }

    /// Synchronises the curvature target used when computing the curvature term.
    pub fn set_curvature_target(&mut self, curvature: f32) {
        if curvature.is_finite() {
            self.curvature_target = curvature;
        }
    }

    /// Clears the running averages maintained by the adapter.
    pub fn reset(&mut self) {
        self.avg_curvature = 0.0;
        self.avg_spin = 0.0;
        self.avg_energy = 0.0;
    }

    /// Returns a copyable state summary suitable for traces and checkpoints.
    pub fn state(&self) -> SpectralLrAdapterState {
        SpectralLrAdapterState {
            sheet_hint: self.sheet_hint,
            smoothing: self.smoothing,
            curvature_target: self.curvature_target,
            curvature_gain: self.curvature_gain,
            spin_gain: self.spin_gain,
            energy_gain: self.energy_gain,
            sheet_gain: self.sheet_gain,
            min_scale: self.min_scale,
            max_scale: self.max_scale,
            avg_curvature: self.avg_curvature,
            avg_spin: self.avg_spin,
            avg_energy: self.avg_energy,
        }
    }

    fn smooth(current: f32, observed: f32, alpha: f32) -> f32 {
        if alpha <= 0.0 {
            observed
        } else {
            current + alpha * (observed - current)
        }
    }
}

fn finite_meta_f32(value: f32) -> serde_json::Value {
    serde_json::Value::from(if value.is_finite() { value as f64 } else { 0.0 })
}

fn optimizer_mode_label(mode: OptimizerMode) -> &'static str {
    match mode {
        OptimizerMode::Euclidean => "euclidean",
        OptimizerMode::Realgrad { .. } => "realgrad",
        OptimizerMode::Hypergrad { .. } => "hypergrad",
    }
}

fn checked_scaled_learning_rate(rate: f32, factor: f32) -> PureResult<f32> {
    let next = rate * factor;
    if next <= 0.0 || !next.is_finite() {
        return Err(TensorError::NonPositiveLearningRate { rate: next });
    }
    Ok(next)
}

/// Audit receipt for applying one Rust-validated meta-optimizer control to
/// model-parameter learning rates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZSpaceParameterControlReceipt {
    pub control_contract_version: &'static str,
    pub control_kind: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub source_meta_step: u64,
    pub previous_source_meta_step: Option<u64>,
    pub previous_absolute_learning_rate_scale: f32,
    pub absolute_learning_rate_scale: f32,
    pub applied_learning_rate_ratio: f32,
    pub source_learning_rate: f64,
    pub source_effective_learning_rate: f64,
    pub changed: bool,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ZSpaceParameterControlPlan {
    receipt: ZSpaceParameterControlReceipt,
}

impl ZSpaceParameterControlPlan {
    pub(crate) fn receipt(self) -> ZSpaceParameterControlReceipt {
        self.receipt
    }
}

pub(crate) fn plan_zspace_parameter_control(
    current_scale: f32,
    current_step: Option<u64>,
    control: &ZSpaceParameterControl,
) -> PureResult<ZSpaceParameterControlPlan> {
    if current_scale <= 0.0 || !current_scale.is_finite() {
        return Err(TensorError::NonPositiveLearningRate {
            rate: current_scale,
        });
    }
    let absolute_scale = control.absolute_learning_rate_scale() as f32;
    if absolute_scale <= 0.0 || !absolute_scale.is_finite() {
        return Err(TensorError::NonPositiveLearningRate {
            rate: absolute_scale,
        });
    }
    let source_step = control.source_step();
    if let Some(previous_step) = current_step {
        if source_step < previous_step {
            return Err(TensorError::Generic(format!(
                "stale Z-space parameter control step {source_step}; latest applied step is {previous_step}"
            )));
        }
        if source_step == previous_step && absolute_scale != current_scale {
            return Err(TensorError::Generic(format!(
                "conflicting Z-space parameter control replay at step {source_step}"
            )));
        }
    }
    let applied_ratio = if current_step == Some(source_step) {
        1.0
    } else {
        checked_scaled_learning_rate(1.0 / current_scale, absolute_scale)?
    };
    Ok(ZSpaceParameterControlPlan {
        receipt: ZSpaceParameterControlReceipt {
            control_contract_version: control.contract_version(),
            control_kind: control.kind(),
            semantic_owner: control.semantic_owner(),
            semantic_backend: control.semantic_backend(),
            source_meta_step: source_step,
            previous_source_meta_step: current_step,
            previous_absolute_learning_rate_scale: current_scale,
            absolute_learning_rate_scale: absolute_scale,
            applied_learning_rate_ratio: applied_ratio,
            source_learning_rate: control.source_learning_rate(),
            source_effective_learning_rate: control.source_effective_learning_rate(),
            changed: applied_ratio != 1.0,
        },
    })
}

pub(crate) fn scale_module_learning_rates<M: Module + ?Sized>(
    module: &mut M,
    factor: f32,
) -> PureResult<()> {
    if factor <= 0.0 || !factor.is_finite() {
        return Err(TensorError::NonPositiveLearningRate { rate: factor });
    }
    module.visit_parameters(&mut |parameter| parameter.validate_learning_rate_scale(factor))?;
    module.visit_parameters_mut(&mut |parameter| parameter.try_scale_learning_rate(factor))
}

pub(crate) fn parameter_control_error(error: impl std::fmt::Display) -> TensorError {
    TensorError::BackendFailure {
        backend: "zspace_parameter_control",
        message: error.to_string(),
    }
}

pub(crate) fn emit_parameter_control_receipt(
    scope: &'static str,
    receipt: ZSpaceParameterControlReceipt,
) {
    emit_tensor_op("zspace_parameter_control_apply", &[1, 4], &[1, 1]);
    emit_tensor_op_meta("zspace_parameter_control_apply", || {
        serde_json::json!({
            "backend": "optimizer_control_cpu",
            "requested_backend": "host",
            "kind": "st_nn_zspace_parameter_control_apply",
            "control_contract_version": receipt.control_contract_version,
            "control_kind": receipt.control_kind,
            "semantic_owner": receipt.semantic_owner,
            "semantic_backend": receipt.semantic_backend,
            "scope": scope,
            "source_meta_step": receipt.source_meta_step,
            "previous_source_meta_step": receipt.previous_source_meta_step,
            "previous_absolute_learning_rate_scale": finite_meta_f32(
                receipt.previous_absolute_learning_rate_scale,
            ),
            "absolute_learning_rate_scale": finite_meta_f32(
                receipt.absolute_learning_rate_scale,
            ),
            "applied_learning_rate_ratio": finite_meta_f32(
                receipt.applied_learning_rate_ratio,
            ),
            "source_learning_rate": receipt.source_learning_rate,
            "source_effective_learning_rate": receipt.source_effective_learning_rate,
            "changed": receipt.changed,
        })
    });
}

impl LocalLearningRateAdapter for SpectralLrAdapter {
    fn sheet_hint(&self) -> usize {
        self.sheet_hint
    }

    fn scale_factor(&mut self, _parameter: &str, features: &SpectralFeatureSample) -> f32 {
        let features_finite = features.curvature.is_finite()
            && features.spin.is_finite()
            && features.energy.is_finite()
            && features.sheet_confidence.is_finite();
        let config_finite = self.smoothing.is_finite()
            && self.curvature_target.is_finite()
            && self.curvature_gain.is_finite()
            && self.spin_gain.is_finite()
            && self.energy_gain.is_finite()
            && self.sheet_gain.is_finite()
            && self.min_scale.is_finite()
            && self.max_scale.is_finite()
            && self.min_scale <= self.max_scale
            && self.min_scale > 0.0;
        let state_finite = self.avg_curvature.is_finite()
            && self.avg_spin.is_finite()
            && self.avg_energy.is_finite();
        let candidate_avg_curvature = if features_finite && config_finite && state_finite {
            Self::smooth(self.avg_curvature, features.curvature, self.smoothing)
        } else {
            self.avg_curvature
        };
        let candidate_avg_spin = if features_finite && config_finite && state_finite {
            Self::smooth(self.avg_spin, features.spin, self.smoothing)
        } else {
            self.avg_spin
        };
        let candidate_avg_energy = if features_finite && config_finite && state_finite {
            Self::smooth(self.avg_energy, features.energy, self.smoothing)
        } else {
            self.avg_energy
        };

        let curvature_delta = if features_finite && config_finite {
            features.curvature - self.curvature_target
        } else {
            f32::NAN
        };
        let curvature_term = 1.0 + self.curvature_gain * curvature_delta;
        let spin_term = 1.0 + self.spin_gain * features.spin;
        let expected_conf = 1.0 / self.sheet_hint.max(1) as f32;
        let sheet_term = 1.0 + self.sheet_gain * (features.sheet_confidence - expected_conf);
        let energy_term = 1.0 + self.energy_gain * (features.energy - candidate_avg_energy);
        let raw_scale = curvature_term * spin_term * sheet_term * energy_term;
        let state_committed = features_finite
            && config_finite
            && state_finite
            && candidate_avg_curvature.is_finite()
            && candidate_avg_spin.is_finite()
            && candidate_avg_energy.is_finite()
            && curvature_term.is_finite()
            && spin_term.is_finite()
            && sheet_term.is_finite()
            && energy_term.is_finite()
            && raw_scale.is_finite()
            && raw_scale > 0.0;
        let clamped = if state_committed {
            self.avg_curvature = candidate_avg_curvature;
            self.avg_spin = candidate_avg_spin;
            self.avg_energy = candidate_avg_energy;
            raw_scale.clamp(self.min_scale, self.max_scale)
        } else {
            1.0
        };
        emit_tensor_op("spectral_lr_scale", &[1, 5], &[1, 1]);
        emit_tensor_op_meta("spectral_lr_scale", || {
            serde_json::json!({
                "backend": "optimizer_control_cpu",
                "requested_backend": "host",
                "kind": "st_nn_spectral_lr_scale",
                "parameter": _parameter,
                "sheet_hint": self.sheet_hint,
                "sheet_index": features.sheet_index,
                "sheet_confidence": finite_meta_f32(features.sheet_confidence),
                "expected_sheet_confidence": finite_meta_f32(expected_conf),
                "curvature": finite_meta_f32(features.curvature),
                "curvature_target": finite_meta_f32(self.curvature_target),
                "curvature_delta": finite_meta_f32(curvature_delta),
                "spin": finite_meta_f32(features.spin),
                "energy": finite_meta_f32(features.energy),
                "avg_curvature": finite_meta_f32(self.avg_curvature),
                "avg_spin": finite_meta_f32(self.avg_spin),
                "avg_energy": finite_meta_f32(self.avg_energy),
                "features_finite": features_finite,
                "config_finite": config_finite,
                "state_finite": state_finite,
                "state_committed": state_committed,
                "curvature_term": finite_meta_f32(curvature_term),
                "spin_term": finite_meta_f32(spin_term),
                "sheet_term": finite_meta_f32(sheet_term),
                "energy_term": finite_meta_f32(energy_term),
                "raw_scale": finite_meta_f32(raw_scale),
                "scale": finite_meta_f32(clamped),
                "min_scale": finite_meta_f32(self.min_scale),
                "max_scale": finite_meta_f32(self.max_scale),
            })
        });
        clamped
    }

    fn on_global_scale(&mut self, factor: f32) {
        if factor.is_finite() && factor > 0.0 {
            self.reset();
        }
    }
}

/// Modes supported by [`ZSpaceParameterOptimizer`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizerMode {
    /// Vanilla Euclidean accumulation using the fallback learning rate.
    Euclidean,
    /// Attach realgrad tapes with the supplied learning rate.
    Realgrad { learning_rate: f32 },
    /// Attach hypergrad tapes with a curvature-aware learning rate.
    Hypergrad { curvature: f32, learning_rate: f32 },
}

impl OptimizerMode {
    /// Returns an Euclidean optimiser mode.
    pub fn euclidean() -> Self {
        Self::Euclidean
    }

    /// Builds a realgrad mode while validating the learning rate.
    pub fn realgrad(learning_rate: f32) -> PureResult<Self> {
        if learning_rate <= 0.0 || !learning_rate.is_finite() {
            return Err(TensorError::NonPositiveLearningRate {
                rate: learning_rate,
            });
        }
        Ok(Self::Realgrad { learning_rate })
    }

    /// Builds a hypergrad mode while validating the learning rate.
    pub fn hypergrad(curvature: f32, learning_rate: f32) -> PureResult<Self> {
        if learning_rate <= 0.0 || !learning_rate.is_finite() {
            return Err(TensorError::NonPositiveLearningRate {
                rate: learning_rate,
            });
        }
        if !curvature.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "hypergrad_curvature",
                value: curvature,
            });
        }
        Ok(Self::Hypergrad {
            curvature,
            learning_rate,
        })
    }
}

/// High-level optimizer that applies accumulated gradients to model parameters.
///
/// Latent Z-state semantics live in `st-core::runtime::zspace_optimizer`; this
/// type consumes only the validated parameter-control projection of that
/// report.
pub struct ZSpaceParameterOptimizer {
    fallback_lr: f32,
    mode: OptimizerMode,
    adapter: Option<Box<dyn LocalLearningRateAdapter + Send>>,
    meta_learning_rate_scale: f32,
    meta_optimizer_step: Option<u64>,
}

/// Copyable optimiser state used by traces, tests, and future checkpoints.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZSpaceParameterOptimizerState {
    pub fallback_learning_rate: f32,
    pub mode: OptimizerMode,
    pub adapter_installed: bool,
    pub meta_learning_rate_scale: f32,
    pub meta_optimizer_step: Option<u64>,
}

impl core::fmt::Debug for ZSpaceParameterOptimizer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ZSpaceParameterOptimizer")
            .field("fallback_lr", &self.fallback_lr)
            .field("mode", &self.mode)
            .field("has_adapter", &self.adapter.is_some())
            .field("meta_learning_rate_scale", &self.meta_learning_rate_scale)
            .field("meta_optimizer_step", &self.meta_optimizer_step)
            .finish()
    }
}

impl ZSpaceParameterOptimizer {
    /// Creates a new optimiser using Euclidean accumulation by default.
    pub fn new(fallback_lr: f32) -> PureResult<Self> {
        if fallback_lr <= 0.0 || !fallback_lr.is_finite() {
            return Err(TensorError::NonPositiveLearningRate { rate: fallback_lr });
        }
        Ok(Self {
            fallback_lr,
            mode: OptimizerMode::Euclidean,
            adapter: None,
            meta_learning_rate_scale: 1.0,
            meta_optimizer_step: None,
        })
    }

    /// Overrides the optimisation mode without touching accumulator state.
    pub fn set_mode(&mut self, mode: OptimizerMode) {
        self.mode = mode;
    }

    /// Returns the current optimisation mode.
    pub fn mode(&self) -> OptimizerMode {
        self.mode
    }

    /// Returns the fallback learning rate used when no tapes are present.
    pub fn fallback_learning_rate(&self) -> f32 {
        self.fallback_lr
    }

    /// Returns the latest absolute meta-optimizer scale applied to parameter
    /// learning rates.
    pub fn meta_learning_rate_scale(&self) -> f32 {
        self.meta_learning_rate_scale
    }

    /// Returns the latest accepted meta-optimizer step, if any.
    pub fn meta_optimizer_step(&self) -> Option<u64> {
        self.meta_optimizer_step
    }

    /// Returns a copyable snapshot of the optimiser configuration.
    pub fn state(&self) -> ZSpaceParameterOptimizerState {
        ZSpaceParameterOptimizerState {
            fallback_learning_rate: self.fallback_lr,
            mode: self.mode,
            adapter_installed: self.adapter.is_some(),
            meta_learning_rate_scale: self.meta_learning_rate_scale,
            meta_optimizer_step: self.meta_optimizer_step,
        }
    }

    /// Overrides the fallback learning rate.
    pub fn set_fallback_learning_rate(&mut self, learning_rate: f32) -> PureResult<()> {
        if learning_rate <= 0.0 || !learning_rate.is_finite() {
            return Err(TensorError::NonPositiveLearningRate {
                rate: learning_rate,
            });
        }
        self.fallback_lr = learning_rate;
        Ok(())
    }

    /// Installs a spectral adapter used to modulate local learning rates.
    pub fn set_adapter<A>(&mut self, adapter: A)
    where
        A: LocalLearningRateAdapter + Send + 'static,
    {
        self.adapter = Some(Box::new(adapter));
    }

    /// Replaces the current adapter with an arbitrary boxed instance.
    pub fn set_adapter_box(&mut self, adapter: Option<Box<dyn LocalLearningRateAdapter + Send>>) {
        self.adapter = adapter;
    }

    /// Returns a shared reference to the installed adapter, if any.
    pub fn adapter(&self) -> Option<&dyn LocalLearningRateAdapter> {
        self.adapter
            .as_ref()
            .map(|adapter| adapter.as_ref() as &dyn LocalLearningRateAdapter)
    }

    /// Returns a mutable reference to the installed adapter, if any.
    pub fn adapter_mut(&mut self) -> Option<&mut dyn LocalLearningRateAdapter> {
        self.adapter
            .as_mut()
            .map(|adapter| adapter.as_mut() as &mut dyn LocalLearningRateAdapter)
    }

    /// Attaches the appropriate gradient tapes to the provided module.
    pub fn prepare_module<M: Module>(&self, module: &mut M) -> PureResult<()> {
        match self.mode {
            OptimizerMode::Euclidean => Ok(()),
            OptimizerMode::Realgrad { learning_rate } => module.attach_realgrad(learning_rate),
            OptimizerMode::Hypergrad {
                curvature,
                learning_rate,
            } => module.attach_hypergrad(curvature, learning_rate),
        }
    }

    /// Applies an optimisation step, routing gradients through the optional adapter.
    pub fn step<M: Module>(&mut self, module: &mut M) -> PureResult<()> {
        match self.adapter.as_deref_mut() {
            Some(adapter) => module.apply_step_with_adapter(self.fallback_lr, Some(adapter)),
            None => module.apply_step_with_adapter(self.fallback_lr, None),
        }
    }

    /// Clears accumulated gradients and hypergrad buffers.
    pub fn zero_grad<M: Module>(&self, module: &mut M) -> PureResult<()> {
        module.zero_accumulators()
    }

    /// Scales the effective learning rate across all attached tapes.
    pub fn scale_learning_rate<M: Module + ?Sized>(
        &mut self,
        module: &mut M,
        factor: f32,
    ) -> PureResult<()> {
        if factor <= 0.0 || !factor.is_finite() {
            return Err(TensorError::NonPositiveLearningRate { rate: factor });
        }
        let before_lr = self.fallback_lr;
        let before_mode = self.mode;
        let adapter_installed = self.adapter.is_some();
        let next_fallback_lr = checked_scaled_learning_rate(self.fallback_lr, factor)?;
        let next_mode_lr = match self.mode {
            OptimizerMode::Euclidean => None,
            OptimizerMode::Realgrad { learning_rate }
            | OptimizerMode::Hypergrad { learning_rate, .. } => {
                Some(checked_scaled_learning_rate(learning_rate, factor)?)
            }
        };
        scale_module_learning_rates(module, factor)?;
        self.fallback_lr = next_fallback_lr;
        if let Some(adapter) = self.adapter.as_deref_mut() {
            adapter.on_global_scale(factor);
        }
        match &mut self.mode {
            OptimizerMode::Euclidean => {}
            OptimizerMode::Realgrad { learning_rate } => {
                *learning_rate = next_mode_lr.expect("realgrad learning rate");
            }
            OptimizerMode::Hypergrad { learning_rate, .. } => {
                *learning_rate = next_mode_lr.expect("hypergrad learning rate");
            }
        }
        emit_tensor_op("zspace_optimizer_lr_scale", &[1, 1], &[1, 1]);
        emit_tensor_op_meta("zspace_optimizer_lr_scale", || {
            serde_json::json!({
                "backend": "optimizer_control_cpu",
                "requested_backend": "host",
                "kind": "st_nn_zspace_optimizer_lr_scale",
                "mode_before": optimizer_mode_label(before_mode),
                "mode_after": optimizer_mode_label(self.mode),
                "adapter_installed": adapter_installed,
                "factor": finite_meta_f32(factor),
                "fallback_lr_before": finite_meta_f32(before_lr),
                "fallback_lr_after": finite_meta_f32(self.fallback_lr),
            })
        });
        Ok(())
    }

    /// Applies the parameter-safe projection of a typed Rust meta-optimizer
    /// report. Replaying the same report is idempotent; stale or conflicting
    /// reports fail before mutating learning rates.
    pub fn apply_zspace_meta_optimizer_report<M: Module + ?Sized>(
        &mut self,
        module: &mut M,
        report: &ZSpaceMetaOptimizerStepReport,
    ) -> PureResult<ZSpaceParameterControlReceipt> {
        let control =
            zspace_parameter_control_from_report(report).map_err(parameter_control_error)?;
        self.apply_zspace_parameter_control(module, &control)
    }

    /// Applies a control previously validated by the Rust semantic core.
    pub fn apply_zspace_parameter_control<M: Module + ?Sized>(
        &mut self,
        module: &mut M,
        control: &ZSpaceParameterControl,
    ) -> PureResult<ZSpaceParameterControlReceipt> {
        let plan = plan_zspace_parameter_control(
            self.meta_learning_rate_scale,
            self.meta_optimizer_step,
            control,
        )?;
        let receipt = plan.receipt();
        if receipt.changed {
            self.scale_learning_rate(module, receipt.applied_learning_rate_ratio)?;
        }
        self.meta_learning_rate_scale = receipt.absolute_learning_rate_scale;
        self.meta_optimizer_step = Some(receipt.source_meta_step);
        emit_parameter_control_receipt("zspace_parameter_optimizer", receipt);
        Ok(receipt)
    }
}

/// Backward-compatible alias for the pre-contract optimizer name.
pub type ZSpaceOptimizer = ZSpaceParameterOptimizer;

/// Backward-compatible alias for the pre-contract optimizer state name.
pub type ZSpaceOptimizerState = ZSpaceParameterOptimizerState;

/// Trait implemented by learning-rate schedules that cooperate with
/// [`ZSpaceParameterOptimizer`].
pub trait LrScheduler {
    /// Advances the schedule returning the new learning rate.
    fn step(&mut self) -> f32;
    /// Returns the most recent learning rate produced by [`Self::step`].
    fn current_lr(&self) -> f32;
    /// Resets the scheduler to its initial state.
    fn reset(&mut self);
}

/// Cosine decay scheduler with linear warmup inspired by PyTorch's implementation.
#[derive(Debug, Clone)]
pub struct WarmupCosineScheduler {
    base_lr: f32,
    min_lr: f32,
    warmup_steps: u32,
    total_steps: u32,
    step: u32,
    last_lr: f32,
}

impl WarmupCosineScheduler {
    /// Creates a new warmup + cosine scheduler.
    pub fn new(base_lr: f32, min_lr: f32, warmup_steps: u32, total_steps: u32) -> PureResult<Self> {
        if base_lr <= 0.0 || !base_lr.is_finite() {
            return Err(TensorError::NonPositiveLearningRate { rate: base_lr });
        }
        if min_lr < 0.0 || !min_lr.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "scheduler_min_lr",
                value: min_lr,
            });
        }
        if total_steps == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: total_steps as usize,
                cols: warmup_steps as usize,
            });
        }
        if warmup_steps > total_steps {
            return Err(TensorError::InvalidDimensions {
                rows: warmup_steps as usize,
                cols: total_steps as usize,
            });
        }
        let initial_lr = if warmup_steps > 0 {
            min_lr.min(base_lr)
        } else {
            base_lr
        };
        Ok(Self {
            base_lr,
            min_lr,
            warmup_steps,
            total_steps,
            step: 0,
            last_lr: initial_lr,
        })
    }

    /// Applies the current learning rate to the optimiser and attached module.
    pub fn step_optimizer<M: Module>(
        &mut self,
        optimizer: &mut ZSpaceParameterOptimizer,
        module: &mut M,
    ) -> PureResult<f32> {
        let scheduled_lr = self.step();
        let lr = checked_scaled_learning_rate(scheduled_lr, optimizer.meta_learning_rate_scale())?;
        let current = optimizer.fallback_learning_rate();
        if current > 0.0 {
            let factor = lr / current;
            optimizer.scale_learning_rate(module, factor)?;
        } else {
            optimizer.set_fallback_learning_rate(lr)?;
        }
        Ok(lr)
    }
}

impl LrScheduler for WarmupCosineScheduler {
    fn step(&mut self) -> f32 {
        self.step = self.step.saturating_add(1);
        let phase = if self.warmup_steps > 0 && self.step <= self.warmup_steps {
            "warmup"
        } else {
            "cosine"
        };
        let lr = if self.warmup_steps > 0 && self.step <= self.warmup_steps {
            let progress = self.step as f32 / self.warmup_steps as f32;
            self.min_lr + (self.base_lr - self.min_lr) * progress
        } else {
            let decay_steps = (self.total_steps - self.warmup_steps).max(1);
            let elapsed = (self.step - self.warmup_steps).min(decay_steps);
            let progress = elapsed as f32 / decay_steps as f32;
            let cosine = 0.5 * (1.0 + (PI * progress).cos());
            self.min_lr + (self.base_lr - self.min_lr) * cosine
        };
        self.last_lr = lr.max(self.min_lr);
        emit_tensor_op("warmup_cosine_lr_step", &[1, 4], &[1, 1]);
        emit_tensor_op_meta("warmup_cosine_lr_step", || {
            serde_json::json!({
                "backend": "optimizer_control_cpu",
                "requested_backend": "host",
                "kind": "st_nn_warmup_cosine_lr_step",
                "phase": phase,
                "step": self.step,
                "base_lr": finite_meta_f32(self.base_lr),
                "min_lr": finite_meta_f32(self.min_lr),
                "last_lr": finite_meta_f32(self.last_lr),
                "warmup_steps": self.warmup_steps,
                "total_steps": self.total_steps,
            })
        });
        self.last_lr
    }

    fn current_lr(&self) -> f32 {
        self.last_lr
    }

    fn reset(&mut self) {
        self.step = 0;
        self.last_lr = if self.warmup_steps > 0 {
            self.min_lr.min(self.base_lr)
        } else {
            self.base_lr
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::linear::Linear;
    use crate::Tensor;
    use st_core::runtime::zspace_optimizer::{
        transition_zspace_meta_optimizer, ZSpaceMetaObservation, ZSpaceMetaOptimizerConfig,
        ZSpaceMetaOptimizerState, ZSpaceMetaOptimizerStepRequest,
    };
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};

    fn observer_lock() -> std::sync::MutexGuard<'static, ()> {
        crate::test_global_state_lock()
    }

    fn meta_report(
        state: ZSpaceMetaOptimizerState,
        learning_rate_scale: f64,
    ) -> ZSpaceMetaOptimizerStepReport {
        let config = ZSpaceMetaOptimizerConfig {
            dimension: 2,
            topos_control_gain: 1.0,
            ..ZSpaceMetaOptimizerConfig::default()
        };
        transition_zspace_meta_optimizer(ZSpaceMetaOptimizerStepRequest {
            config,
            state,
            observation: ZSpaceMetaObservation {
                gradient: vec![0.1, -0.2],
                telemetry: BTreeMap::from([(
                    "topos.training_hints.learning_rate_scale".to_owned(),
                    learning_rate_scale,
                )]),
                ..ZSpaceMetaObservation::default()
            },
        })
        .expect("meta report")
    }

    #[test]
    fn spectral_adapter_increases_scale_on_high_curvature() {
        let mut adapter = SpectralLrAdapter::default();
        adapter.set_curvature_target(0.2);
        let features = SpectralFeatureSample {
            sheet_index: 1,
            sheet_confidence: 0.6,
            curvature: 0.8,
            spin: 0.4,
            energy: 0.5,
        };
        let factor = adapter.scale_factor("weight", &features);
        assert!(factor > 1.0, "expected factor > 1, got {factor}");
    }

    #[test]
    fn spectral_adapter_emits_scale_backend_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let mut adapter = SpectralLrAdapter::default().with_sheet_hint(4);
        adapter.set_curvature_target(0.2);
        let features = SpectralFeatureSample {
            sheet_index: 2,
            sheet_confidence: 0.75,
            curvature: 0.8,
            spin: 0.4,
            energy: 0.5,
        };
        let factor = adapter.scale_factor("weight", &features);
        st_tensor::set_thread_meta_observer(previous);

        assert!(factor > 1.0);
        let events = events.lock().unwrap();
        let meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "spectral_lr_scale"
                    && data["kind"] == "st_nn_spectral_lr_scale"
                    && data["parameter"] == "weight"
            })
            .expect("spectral lr scale metadata event");
        assert_eq!(meta.1["backend"], "optimizer_control_cpu");
        assert_eq!(meta.1["requested_backend"], "host");
        assert_eq!(meta.1["sheet_hint"], 4);
        assert_eq!(meta.1["sheet_index"], 2);
        assert!(meta.1["scale"].as_f64().unwrap() > 1.0);
        assert!(meta.1["curvature_term"].as_f64().unwrap() > 1.0);
    }

    #[test]
    fn spectral_adapter_resets_on_global_scale() {
        let mut adapter = SpectralLrAdapter::default();
        adapter.set_curvature_target(0.3);
        let features = SpectralFeatureSample {
            sheet_index: 0,
            sheet_confidence: 0.4,
            curvature: 0.7,
            spin: -0.2,
            energy: 0.6,
        };
        let _ = adapter.scale_factor("weight", &features);
        adapter.on_global_scale(0.5);
        assert_eq!(adapter.avg_curvature, 0.0);
        assert_eq!(adapter.avg_spin, 0.0);
        assert_eq!(adapter.avg_energy, 0.0);
    }

    #[test]
    fn spectral_adapter_rejects_non_finite_features_without_poisoning_state() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let mut adapter = SpectralLrAdapter::default().with_sheet_hint(4);
        adapter.set_curvature_target(0.25);
        adapter.set_curvature_target(f32::NAN);
        let warm_features = SpectralFeatureSample {
            sheet_index: 1,
            sheet_confidence: 0.5,
            curvature: 0.45,
            spin: 0.2,
            energy: 0.4,
        };
        let _ = adapter.scale_factor("weight", &warm_features);
        let before = adapter.state();
        let bad_features = SpectralFeatureSample {
            sheet_index: 2,
            sheet_confidence: 0.75,
            curvature: f32::NAN,
            spin: 0.4,
            energy: f32::INFINITY,
        };

        let factor = adapter.scale_factor("weight", &bad_features);
        st_tensor::set_thread_meta_observer(previous);

        assert_eq!(factor, 1.0);
        assert_eq!(adapter.state(), before);
        let events = events.lock().unwrap();
        let meta = events
            .iter()
            .rev()
            .find(|(op_name, data)| {
                *op_name == "spectral_lr_scale"
                    && data["kind"] == "st_nn_spectral_lr_scale"
                    && data["parameter"] == "weight"
            })
            .expect("spectral lr scale metadata event");
        assert_eq!(meta.1["features_finite"], false);
        assert_eq!(meta.1["state_committed"], false);
        assert_eq!(meta.1["scale"], 1.0);
    }

    #[test]
    fn spectral_adapter_state_tracks_running_features() {
        let mut adapter = SpectralLrAdapter::default().with_sheet_hint(4);
        adapter.set_curvature_target(0.25);
        let features = SpectralFeatureSample {
            sheet_index: 2,
            sheet_confidence: 0.75,
            curvature: 0.5,
            spin: 0.25,
            energy: 0.4,
        };
        let _ = adapter.scale_factor("weight", &features);

        let state = adapter.state();
        assert_eq!(state.sheet_hint, 4);
        assert_eq!(state.curvature_target, 0.25);
        assert!(state.avg_curvature > 0.0);
        assert!(state.avg_spin > 0.0);
        assert!(state.avg_energy > 0.0);
    }

    #[test]
    fn zspace_optimizer_updates_parameters() {
        let mut layer = Linear::new("opt", 3, 2).unwrap();
        let mut optimizer = ZSpaceParameterOptimizer::new(0.05).unwrap();
        optimizer.prepare_module(&mut layer).unwrap();
        let input = Tensor::from_vec(1, 3, vec![0.5, -1.0, 2.0]).unwrap();
        let target = Tensor::zeros(1, 2).unwrap();
        let output = layer.forward(&input).unwrap();
        let diff = output.sub(&target).unwrap();
        let grad = diff.scale(1.0).unwrap();
        let _ = layer.backward(&input, &grad).unwrap();
        let before = layer.weight().value().clone();
        optimizer.step(&mut layer).unwrap();
        optimizer.zero_grad(&mut layer).unwrap();
        assert_ne!(before, *layer.weight().value());
        let initial_lr = optimizer.fallback_learning_rate();
        optimizer.scale_learning_rate(&mut layer, 0.5).unwrap();
        assert!((optimizer.fallback_learning_rate() - initial_lr * 0.5).abs() < 1e-6);
    }

    #[test]
    fn zspace_optimizer_rejects_overflow_lr_scale_without_mutating_state() {
        let mut layer = Linear::new("opt_overflow", 2, 2).unwrap();
        let mut optimizer = ZSpaceParameterOptimizer::new(f32::MAX).unwrap();
        optimizer.set_mode(OptimizerMode::realgrad(f32::MAX).unwrap());
        let before = optimizer.state();

        let err = optimizer.scale_learning_rate(&mut layer, 2.0).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonPositiveLearningRate { rate } if !rate.is_finite()
        ));
        assert_eq!(optimizer.state(), before);
    }

    #[test]
    fn zspace_optimizer_state_tracks_mode_and_adapter() {
        let mut optimizer = ZSpaceParameterOptimizer::new(0.05).unwrap();
        assert_eq!(
            optimizer.state(),
            ZSpaceParameterOptimizerState {
                fallback_learning_rate: 0.05,
                mode: OptimizerMode::Euclidean,
                adapter_installed: false,
                meta_learning_rate_scale: 1.0,
                meta_optimizer_step: None,
            }
        );

        optimizer.set_mode(OptimizerMode::realgrad(0.02).unwrap());
        optimizer.set_adapter(SpectralLrAdapter::default());
        let state = optimizer.state();
        assert_eq!(state.fallback_learning_rate, 0.05);
        assert_eq!(
            state.mode,
            OptimizerMode::Realgrad {
                learning_rate: 0.02
            }
        );
        assert!(state.adapter_installed);
    }

    #[test]
    fn parameter_optimizer_applies_absolute_meta_scale_idempotently() {
        let mut layer = Linear::new("meta_control", 2, 2).unwrap();
        let mut optimizer = ZSpaceParameterOptimizer::new(0.1).unwrap();
        optimizer.set_mode(OptimizerMode::realgrad(0.05).unwrap());
        optimizer.prepare_module(&mut layer).unwrap();
        let first = meta_report(ZSpaceMetaOptimizerState::zeros(2), 0.5);

        let receipt = optimizer
            .apply_zspace_meta_optimizer_report(&mut layer, &first)
            .unwrap();
        assert!(receipt.changed);
        assert_eq!(receipt.applied_learning_rate_ratio, 0.5);
        assert_eq!(optimizer.fallback_learning_rate(), 0.05);
        assert_eq!(optimizer.state().meta_learning_rate_scale, 0.5);
        assert_eq!(optimizer.state().meta_optimizer_step, Some(1));

        let replay = optimizer
            .apply_zspace_meta_optimizer_report(&mut layer, &first)
            .unwrap();
        assert!(!replay.changed);
        assert_eq!(replay.applied_learning_rate_ratio, 1.0);
        assert_eq!(optimizer.fallback_learning_rate(), 0.05);

        let second = meta_report(first.state_after.clone(), 0.8);
        let second_receipt = optimizer
            .apply_zspace_meta_optimizer_report(&mut layer, &second)
            .unwrap();
        assert_eq!(second_receipt.applied_learning_rate_ratio, 1.6);
        assert!((optimizer.fallback_learning_rate() - 0.08).abs() < 1.0e-7);

        let before_stale = optimizer.state();
        let error = optimizer
            .apply_zspace_meta_optimizer_report(&mut layer, &first)
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("stale Z-space parameter control"));
        assert_eq!(optimizer.state(), before_stale);
    }

    #[test]
    fn parameter_optimizer_rejects_unvalidated_meta_report_without_mutation() {
        let mut layer = Linear::new("meta_unvalidated", 2, 2).unwrap();
        let mut optimizer = ZSpaceParameterOptimizer::new(0.1).unwrap();
        let mut report = meta_report(ZSpaceMetaOptimizerState::zeros(2), 0.5);
        report.transition_validated = false;
        let before = optimizer.state();

        let error = optimizer
            .apply_zspace_meta_optimizer_report(&mut layer, &report)
            .unwrap_err();

        assert!(error.to_string().contains("did not validate"));
        assert_eq!(optimizer.state(), before);
    }

    #[test]
    fn parameter_optimizer_preflights_attached_tape_learning_rates() {
        let mut layer = Linear::new("tape_overflow", 2, 2).unwrap();
        layer.attach_realgrad(f32::MAX).unwrap();
        let mut optimizer = ZSpaceParameterOptimizer::new(0.1).unwrap();
        let before = optimizer.state();

        let error = optimizer.scale_learning_rate(&mut layer, 2.0).unwrap_err();

        assert!(matches!(
            error,
            TensorError::NonPositiveLearningRate { rate } if !rate.is_finite()
        ));
        assert_eq!(optimizer.state(), before);
        layer
            .visit_parameters(&mut |parameter| {
                assert_eq!(parameter.realgrad().unwrap().learning_rate(), f32::MAX);
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn warmup_scheduler_decays_learning_rate() {
        let mut layer = Linear::new("sched", 2, 2).unwrap();
        let mut optimizer = ZSpaceParameterOptimizer::new(0.1).unwrap();
        optimizer.prepare_module(&mut layer).unwrap();
        let mut scheduler = WarmupCosineScheduler::new(0.1, 0.01, 2, 6).unwrap();
        let lr1 = scheduler
            .step_optimizer(&mut optimizer, &mut layer)
            .unwrap();
        let lr2 = scheduler
            .step_optimizer(&mut optimizer, &mut layer)
            .unwrap();
        let lr3 = scheduler
            .step_optimizer(&mut optimizer, &mut layer)
            .unwrap();
        assert!(lr1 >= 0.01);
        assert!(lr2 >= lr3);
        scheduler.reset();
        assert!((scheduler.current_lr() - 0.01).abs() < 1e-6);
    }

    #[test]
    fn warmup_scheduler_preserves_active_meta_scale() {
        let mut layer = Linear::new("sched_meta_overlay", 2, 2).unwrap();
        let mut optimizer = ZSpaceParameterOptimizer::new(0.1).unwrap();
        let report = meta_report(ZSpaceMetaOptimizerState::zeros(2), 0.5);
        optimizer
            .apply_zspace_meta_optimizer_report(&mut layer, &report)
            .unwrap();
        let mut scheduler = WarmupCosineScheduler::new(0.1, 0.01, 2, 6).unwrap();

        let effective_lr = scheduler
            .step_optimizer(&mut optimizer, &mut layer)
            .unwrap();

        assert!((scheduler.current_lr() - 0.055).abs() < 1.0e-6);
        assert!((effective_lr - 0.0275).abs() < 1.0e-6);
        assert!((optimizer.fallback_learning_rate() - effective_lr).abs() < 1.0e-6);
        assert_eq!(optimizer.meta_learning_rate_scale(), 0.5);
    }

    #[test]
    fn scheduler_and_optimizer_emit_lr_backend_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let mut layer = Linear::new("sched_meta", 2, 2).unwrap();
        let mut optimizer = ZSpaceParameterOptimizer::new(0.1).unwrap();
        optimizer.set_mode(OptimizerMode::realgrad(0.05).unwrap());
        optimizer.prepare_module(&mut layer).unwrap();
        let mut scheduler = WarmupCosineScheduler::new(0.1, 0.01, 2, 6).unwrap();
        let lr = scheduler
            .step_optimizer(&mut optimizer, &mut layer)
            .unwrap();
        st_tensor::set_thread_meta_observer(previous);

        assert!(lr > 0.01);
        let events = events.lock().unwrap();
        let scheduler_meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "warmup_cosine_lr_step"
                    && data["kind"] == "st_nn_warmup_cosine_lr_step"
                    && data["step"] == 1
            })
            .expect("warmup cosine metadata event");
        assert_eq!(scheduler_meta.1["phase"], "warmup");
        assert_eq!(scheduler_meta.1["backend"], "optimizer_control_cpu");
        assert_eq!(scheduler_meta.1["requested_backend"], "host");
        assert_eq!(scheduler_meta.1["warmup_steps"], 2);

        let optimizer_meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "zspace_optimizer_lr_scale"
                    && data["kind"] == "st_nn_zspace_optimizer_lr_scale"
                    && data["mode_after"] == "realgrad"
            })
            .expect("optimizer lr scale metadata event");
        assert_eq!(optimizer_meta.1["backend"], "optimizer_control_cpu");
        assert_eq!(optimizer_meta.1["requested_backend"], "host");
        assert!(optimizer_meta.1["factor"].as_f64().unwrap() > 0.0);
        assert!(optimizer_meta.1["fallback_lr_after"].as_f64().unwrap() > 0.0);
    }
}
