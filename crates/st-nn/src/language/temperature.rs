// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#[derive(Clone, Debug)]
pub struct TemperatureController {
    value: f32,
    target_entropy: f32,
    eta: f32,
    min: f32,
    max: f32,
}

impl TemperatureController {
    pub fn new(value: f32, target_entropy: f32, eta: f32, min: f32, max: f32) -> Self {
        let mut controller = Self {
            value: value.max(min).min(max),
            target_entropy: target_entropy.max(1e-4),
            eta: eta.max(0.0),
            min: min.max(1e-3),
            max: max.max(min.max(1e-3)),
        };
        if controller.value < controller.min {
            controller.value = controller.min;
        }
        controller
    }

    pub fn value(&self) -> f32 {
        self.value
    }

    pub fn update(&mut self, distribution: &[f32]) -> f32 {
        let entropy = entropy(distribution);
        let delta = entropy - self.target_entropy;
        self.value = (self.value + self.eta * delta).clamp(self.min, self.max);
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
