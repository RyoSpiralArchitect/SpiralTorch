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
    z_kappa: f32,
    z_relax: f32,
    z_memory: f32,
    grad_pressure: f32,
    grad_entropy: f32,
}

impl TemperatureController {
    pub fn new(value: f32, target_entropy: f32, eta: f32, min: f32, max: f32) -> Self {
        let mut controller = Self {
            value: value.max(min).min(max),
            target_entropy: target_entropy.max(1e-4),
            eta: eta.max(0.0),
            min: min.max(1e-3),
            max: max.max(min.max(1e-3)),
            z_kappa: 0.35,
            z_relax: 0.2,
            z_memory: 0.0,
            grad_pressure: 0.0,
            grad_entropy: 0.0,
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

    pub fn update(&mut self, distribution: &[f32], z_feedback: Option<&SoftlogicZFeedback>) -> f32 {
        let entropy = entropy(distribution);
        let delta = entropy - self.target_entropy;
        self.value = (self.value + self.eta * delta).clamp(self.min, self.max);

        if let Some(feedback) = z_feedback {
            let total_band =
                feedback.band_energy.0 + feedback.band_energy.1 + feedback.band_energy.2;
            let snr = (total_band + feedback.psi_total.abs()).max(1e-3);
            let drift_norm = (feedback.drift / snr).abs().min(4.0);
            let magnitude = feedback.z_signal.abs().min(4.0);
            let explore = (drift_norm - magnitude).max(0.0);
            let settle = (magnitude - drift_norm).max(0.0);
            self.z_memory =
                (1.0 - self.z_relax) * self.z_memory + self.z_relax * (explore - settle);
            let bias = (-self.z_kappa * magnitude).exp();
            let roam = 1.0 + self.z_kappa * (explore + self.z_memory.max(0.0));
            let anchor = 1.0 / (1.0 + self.z_kappa * (settle + (-self.z_memory).max(0.0)));
            self.value = (self.value * bias * roam * anchor).clamp(self.min, self.max);
        }

        self.value
    }

    pub fn observe_grad(&mut self, pressure: f32, entropy_bias: f32) {
        self.grad_pressure = pressure.max(0.0);
        self.grad_entropy = entropy_bias.clamp(0.0, 1.0);
    }

    pub fn update_with_gradient(&mut self, distribution: &[f32], heat: f32) -> f32 {
        let baseline = self.update(distribution, None);
        if self.grad_pressure > 0.0 {
            let effective_heat = heat.max(0.0);
            let adjustment = self.grad_pressure * self.grad_entropy * effective_heat;
            self.value = (baseline + self.eta * adjustment).clamp(self.min, self.max);
            self.grad_pressure *= 0.5;
            self.grad_entropy *= 0.5;
        }
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
        let baseline = controller.update(&[0.6, 0.4], None);
        controller.observe_grad(32.0, 0.15);
        let warmed = controller.update_with_gradient(&[0.6, 0.4], 1.5);
        assert!(warmed >= baseline);
        controller.observe_grad(0.0, 0.95);
        let cooled = controller.update_with_gradient(&[0.6, 0.4], 1.5);
        assert!(cooled <= warmed);
        assert!(cooled >= baseline);
    }
}
