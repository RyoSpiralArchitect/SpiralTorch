// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use st_core::ops::zspace_round::SpectralFeatureSample;

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
