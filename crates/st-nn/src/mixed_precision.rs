// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::Module;
use crate::optim::ZSpaceOptimizer;
use crate::{PureResult, Tensor};
use st_tensor::TensorError;
use std::sync::atomic::{AtomicBool, Ordering};

static AUTOCAST_ENABLED: AtomicBool = AtomicBool::new(false);

/// Returns whether autocast has been enabled for the current thread.
pub fn autocast_enabled() -> bool {
    AUTOCAST_ENABLED.load(Ordering::SeqCst)
}

/// Guard that toggles autocast for the duration of its lifetime.
#[derive(Debug)]
pub struct AutocastGuard {
    previous: bool,
}

impl AutocastGuard {
    /// Creates a new guard setting the autocast flag to `enabled`.
    pub fn new(enabled: bool) -> Self {
        let previous = AUTOCAST_ENABLED.swap(enabled, Ordering::SeqCst);
        Self { previous }
    }

    /// Convenience helper that enables autocast for the guard lifetime.
    pub fn enable() -> Self {
        Self::new(true)
    }
}

impl Drop for AutocastGuard {
    fn drop(&mut self) {
        AUTOCAST_ENABLED.store(self.previous, Ordering::SeqCst);
    }
}

/// Dynamic loss scaler inspired by PyTorch's AMP utilities.
#[derive(Debug, Clone)]
pub struct GradScaler {
    scale: f32,
    growth_factor: f32,
    backoff_factor: f32,
    growth_interval: u32,
    growth_tracker: u32,
    min_scale: f32,
    max_scale: f32,
}

impl GradScaler {
    /// Builds a new scaler with explicit growth/backoff configuration.
    pub fn new(
        scale: f32,
        growth_factor: f32,
        backoff_factor: f32,
        growth_interval: u32,
    ) -> PureResult<Self> {
        if scale <= 0.0 || !scale.is_finite() {
            return Err(TensorError::NonPositiveLearningRate { rate: scale });
        }
        if growth_factor <= 1.0 || !growth_factor.is_finite() {
            return Err(TensorError::InvalidValue {
                label: "grad_scaler_growth_factor",
            });
        }
        if backoff_factor <= 0.0 || backoff_factor >= 1.0 || !backoff_factor.is_finite() {
            return Err(TensorError::InvalidValue {
                label: "grad_scaler_backoff_factor",
            });
        }
        if growth_interval == 0 {
            return Err(TensorError::InvalidValue {
                label: "grad_scaler_growth_interval",
            });
        }
        Ok(Self {
            scale,
            growth_factor,
            backoff_factor,
            growth_interval,
            growth_tracker: 0,
            min_scale: 1.0,
            max_scale: 65536.0,
        })
    }

    /// Configures the minimum/maximum scaling range enforced by [`update_scale`].
    pub fn with_limits(mut self, min_scale: f32, max_scale: f32) -> Self {
        if min_scale > 0.0 && min_scale.is_finite() {
            self.min_scale = min_scale;
        }
        if max_scale >= self.min_scale && max_scale.is_finite() {
            self.max_scale = max_scale;
        }
        self
    }

    /// Returns the current scaling factor.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Multiplies a scalar loss by the current scale.
    pub fn scale_loss(&self, loss: f32) -> f32 {
        loss * self.scale
    }

    /// Scales every element of the provided tensor by the current scale.
    pub fn scale_tensor(&self, tensor: &Tensor) -> PureResult<Tensor> {
        tensor.scale(self.scale)
    }

    /// Applies the inverse scale to all parameter accumulators.
    pub fn unscale_module<M: Module>(&self, module: &mut M) -> PureResult<()> {
        if self.scale == 1.0 {
            return Ok(());
        }
        let inverse = 1.0 / self.scale;
        module.visit_parameters_mut(&mut |param| {
            param.scale_accumulators(inverse);
            Ok(())
        })
    }

    /// Checks whether any accumulator contains `NaN` or `Inf` values.
    pub fn has_overflow<M: Module>(&self, module: &M) -> PureResult<bool> {
        let mut overflow = false;
        module.visit_parameters(&mut |param| {
            let norm = param.accumulators_norm_sq();
            if !norm.is_finite() {
                overflow = true;
            }
            Ok(())
        })?;
        Ok(overflow)
    }

    fn backoff(&mut self) {
        self.scale = (self.scale * self.backoff_factor).clamp(self.min_scale, self.max_scale);
        self.growth_tracker = 0;
    }

    fn grow(&mut self) {
        self.scale = (self.scale * self.growth_factor).clamp(self.min_scale, self.max_scale);
        self.growth_tracker = 0;
    }

    /// Updates the internal scale based on whether an overflow was detected.
    pub fn update(&mut self, found_inf: bool) {
        if found_inf {
            self.backoff();
        } else {
            self.growth_tracker = self.growth_tracker.saturating_add(1);
            if self.growth_tracker >= self.growth_interval {
                self.grow();
            }
        }
    }

    /// Unscales gradients, checks for overflow, and steps the optimiser if safe.
    ///
    /// Returns `true` when the optimiser step was executed, `false` otherwise.
    pub fn step<M: Module>(
        &mut self,
        optimizer: &mut ZSpaceOptimizer,
        module: &mut M,
    ) -> PureResult<bool> {
        self.unscale_module(module)?;
        let overflow = self.has_overflow(module)?;
        if overflow {
            self.update(true);
            module.zero_accumulators()?;
            return Ok(false);
        }
        optimizer.step(module)?;
        self.update(false);
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::linear::Linear;
    use crate::optim::OptimizerMode;
    use crate::Tensor;

    #[test]
    fn autocast_guard_restores_previous_state() {
        AUTOCAST_ENABLED.store(false, Ordering::SeqCst);
        {
            let _guard = AutocastGuard::enable();
            assert!(autocast_enabled());
        }
        assert!(!autocast_enabled());
    }

    #[test]
    fn grad_scaler_scales_and_steps_optimizer() {
        let mut layer = Linear::new("amp", 2, 2).unwrap();
        let mut optimizer = ZSpaceOptimizer::new(0.05).unwrap();
        optimizer.set_mode(OptimizerMode::realgrad(0.05).unwrap());
        optimizer.prepare_module(&mut layer).unwrap();
        let mut scaler = GradScaler::new(2.0, 2.0, 0.5, 2)
            .unwrap()
            .with_limits(1.0, 16.0);
        let input = Tensor::from_vec(1, 2, vec![0.3, -0.7]).unwrap();
        let target = Tensor::zeros(1, 2).unwrap();
        let output = layer.forward(&input).unwrap();
        let loss = output.sub(&target).unwrap();
        let scaled = scaler.scale_tensor(&loss).unwrap();
        let _ = layer.backward(&input, &scaled).unwrap();
        let stepped = scaler.step(&mut optimizer, &mut layer).unwrap();
        assert!(stepped);
        assert!(scaler.scale() >= 2.0);
    }
}
