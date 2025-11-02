// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::Module;
use crate::PureResult;
use st_core::ops::zspace_round::SpectralFeatureSample;
use st_tensor::TensorError;
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
        self.curvature_target = curvature;
    }

    /// Clears the running averages maintained by the adapter.
    pub fn reset(&mut self) {
        self.avg_curvature = 0.0;
        self.avg_spin = 0.0;
        self.avg_energy = 0.0;
    }

    fn smooth(current: f32, observed: f32, alpha: f32) -> f32 {
        if alpha <= 0.0 {
            observed
        } else {
            current + alpha * (observed - current)
        }
    }
}

impl LocalLearningRateAdapter for SpectralLrAdapter {
    fn sheet_hint(&self) -> usize {
        self.sheet_hint
    }

    fn scale_factor(&mut self, _parameter: &str, features: &SpectralFeatureSample) -> f32 {
        self.avg_curvature = Self::smooth(self.avg_curvature, features.curvature, self.smoothing);
        self.avg_spin = Self::smooth(self.avg_spin, features.spin, self.smoothing);
        self.avg_energy = Self::smooth(self.avg_energy, features.energy, self.smoothing);

        let curvature_delta = features.curvature - self.curvature_target;
        let curvature_term = 1.0 + self.curvature_gain * curvature_delta;
        let spin_term = 1.0 + self.spin_gain * features.spin;
        let expected_conf = 1.0 / self.sheet_hint.max(1) as f32;
        let sheet_term = 1.0 + self.sheet_gain * (features.sheet_confidence - expected_conf);
        let energy_term = 1.0 + self.energy_gain * (features.energy - self.avg_energy);

        let mut scale = curvature_term * spin_term * sheet_term * energy_term;
        if !scale.is_finite() || scale <= 0.0 {
            scale = 1.0;
        }
        scale.clamp(self.min_scale, self.max_scale)
    }

    fn on_global_scale(&mut self, factor: f32) {
        if factor.is_finite() && factor > 0.0 {
            self.reset();
        }
    }
}

/// Modes supported by [`ZSpaceOptimizer`].
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

/// High-level optimiser that orchestrates Z-space aware updates.
pub struct ZSpaceOptimizer {
    fallback_lr: f32,
    mode: OptimizerMode,
    adapter: Option<Box<dyn LocalLearningRateAdapter + Send>>,
}

impl core::fmt::Debug for ZSpaceOptimizer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ZSpaceOptimizer")
            .field("fallback_lr", &self.fallback_lr)
            .field("mode", &self.mode)
            .field("has_adapter", &self.adapter.is_some())
            .finish()
    }
}

impl ZSpaceOptimizer {
    /// Creates a new optimiser using Euclidean accumulation by default.
    pub fn new(fallback_lr: f32) -> PureResult<Self> {
        if fallback_lr <= 0.0 || !fallback_lr.is_finite() {
            return Err(TensorError::NonPositiveLearningRate { rate: fallback_lr });
        }
        Ok(Self {
            fallback_lr,
            mode: OptimizerMode::Euclidean,
            adapter: None,
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
    pub fn scale_learning_rate<M: Module>(
        &mut self,
        module: &mut M,
        factor: f32,
    ) -> PureResult<()> {
        if factor <= 0.0 || !factor.is_finite() {
            return Err(TensorError::NonPositiveLearningRate { rate: factor });
        }
        self.fallback_lr *= factor;
        module.visit_parameters_mut(&mut |param| {
            param.scale_learning_rate(factor);
            Ok(())
        })?;
        if let Some(adapter) = self.adapter.as_deref_mut() {
            adapter.on_global_scale(factor);
        }
        match &mut self.mode {
            OptimizerMode::Euclidean => {}
            OptimizerMode::Realgrad { learning_rate } => {
                *learning_rate *= factor;
            }
            OptimizerMode::Hypergrad { learning_rate, .. } => {
                *learning_rate *= factor;
            }
        }
        Ok(())
    }
}

/// Trait implemented by learning-rate schedules that cooperate with [`ZSpaceOptimizer`].
pub trait LrScheduler {
    /// Advances the schedule returning the new learning rate.
    fn step(&mut self) -> f32;
    /// Returns the most recent learning rate produced by [`step`].
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
        optimizer: &mut ZSpaceOptimizer,
        module: &mut M,
    ) -> PureResult<f32> {
        let lr = self.step();
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
    fn zspace_optimizer_updates_parameters() {
        let mut layer = Linear::new("opt", 3, 2).unwrap();
        let mut optimizer = ZSpaceOptimizer::new(0.05).unwrap();
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
    fn warmup_scheduler_decays_learning_rate() {
        let mut layer = Linear::new("sched", 2, 2).unwrap();
        let mut optimizer = ZSpaceOptimizer::new(0.1).unwrap();
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
}
