// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use st_core::telemetry::hub::SoftlogicZFeedback;

#[derive(Clone, Debug)]
pub struct TemperatureController {
    value: f32,
    target_entropy: f32,
    eta: f32,
    min: f32,
    max: f32,
    grad_norm_avg: f32,
    grad_sparsity_avg: f32,
    grad_alpha: f32,
    grad_norm_weight: f32,
    grad_sparsity_weight: f32,
}

impl TemperatureController {
    pub fn new(value: f32, target_entropy: f32, eta: f32, min: f32, max: f32) -> Self {
        let mut controller = Self {
            value: value.max(min).min(max),
            target_entropy: target_entropy.max(1e-4),
            eta: eta.max(0.0),
            min: min.max(1e-3),
            max: max.max(min.max(1e-3)),
            grad_norm_avg: 0.0,
            grad_sparsity_avg: 0.5,
            grad_alpha: 0.25,
            grad_norm_weight: 0.05,
            grad_sparsity_weight: 0.04,
        };
        if controller.value < controller.min {
            controller.value = controller.min;
        }
        controller
    }

    pub fn with_feedback(mut self, kappa: f32, relax: f32) -> Self {
        self.z_kappa = kappa.max(0.0);
        self.z_relax = relax.clamp(0.0, 1.0);
        self
    }

    pub fn value(&self) -> f32 {
        self.value
    }

    pub fn update(&mut self, distribution: &[f32]) -> f32 {
        self.update_with_gradient(distribution, 1.0)
    }

    pub fn observe_grad(&mut self, norm: f32, sparsity: f32) {
        let norm = norm.max(0.0);
        let sparsity = sparsity.clamp(0.0, 1.0);
        let alpha = self.grad_alpha.clamp(0.0, 1.0);
        self.grad_norm_avg = (1.0 - alpha) * self.grad_norm_avg + alpha * norm;
        self.grad_sparsity_avg = (1.0 - alpha) * self.grad_sparsity_avg + alpha * sparsity;
    }

    pub fn update_with_gradient(&mut self, distribution: &[f32], gradient_gain: f32) -> f32 {
        let entropy = entropy(distribution);
        let delta = entropy - self.target_entropy;
        let grad_term = gradient_gain
            * (self.grad_norm_avg * self.grad_norm_weight
                - (self.grad_sparsity_avg - 0.5) * self.grad_sparsity_weight);
        self.value = (self.value + self.eta * delta + grad_term).clamp(self.min, self.max);
        self.value
    }
}

pub fn entropy(distribution: &[f32]) -> f32 {
    let mut h = 0.0f32;
    for &p in distribution {
        if p > 0.0 {
            h -= p * p.max(1e-9).ln();
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn controller_tracks_gradient_pressure() {
        let mut controller = TemperatureController::new(1.0, 0.8, 0.4, 0.4, 2.0);
        let baseline = controller.update(&[0.6, 0.4]);
        controller.observe_grad(32.0, 0.15);
        let warmed = controller.update_with_gradient(&[0.6, 0.4], 1.5);
        assert!(warmed >= baseline);
        controller.observe_grad(0.0, 0.95);
        let cooled = controller.update_with_gradient(&[0.6, 0.4], 1.5);
        assert!(cooled >= warmed);
    }
}
