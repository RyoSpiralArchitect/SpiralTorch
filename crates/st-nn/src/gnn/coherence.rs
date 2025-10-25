// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::fmt;

use st_core::telemetry::hub::{subscribe_psi_metrics, PsiMetricsFrame, PsiMetricsSubscription};
use st_core::telemetry::psi::PsiComponent;

/// Harmonises neighbourhood blending weights using live ψ telemetry, keeping graph flows coherent.
pub struct PsiCoherenceAdaptor {
    subscription: Option<PsiMetricsSubscription>,
    clip: f32,
    min_scale: f32,
    max_scale: f32,
    smoothing: f32,
    ema_scale: f32,
    epsilon: f32,
    hop_bias: f32,
}

impl PsiCoherenceAdaptor {
    /// Builds a new adaptor that subscribes to the global ψ metrics stream when available.
    pub fn new() -> Self {
        Self::default()
    }

    /// Constructs an adaptor using a custom subscription (or `None` for fallbacks).
    pub fn with_subscription(subscription: Option<PsiMetricsSubscription>) -> Self {
        Self {
            subscription,
            clip: 2.5,
            min_scale: 0.65,
            max_scale: 1.85,
            smoothing: 0.2,
            ema_scale: 1.0,
            epsilon: 1.0e-6,
            hop_bias: 0.08,
        }
    }

    /// Blends ψ telemetry into the provided neighbourhood weights, biasing deeper hops when the
    /// coherence budget allows and falling back to the base weights when telemetry is unavailable.
    pub fn cohere_weights(&mut self, base: Vec<f32>) -> Vec<f32> {
        if base.is_empty() {
            return base;
        }
        let mut adjusted = base.clone();
        let mut fallback = false;
        if let Some(subscription) = self.subscription.as_ref() {
            let frame = subscription.snapshot();
            if let Some(scale) = self.compute_target_coherence(&frame) {
                let clamped = scale.clamp(self.min_scale, self.max_scale);
                self.ema_scale = self.smoothing * clamped + (1.0 - self.smoothing) * self.ema_scale;
                let scale = self.ema_scale;
                for weight in &mut adjusted {
                    *weight = (*weight * scale).clamp(-self.clip, self.clip);
                }
                if !self.rebalance_inplace(&mut adjusted) {
                    fallback = true;
                }
            } else {
                self.ema_scale = self.smoothing * 1.0 + (1.0 - self.smoothing) * self.ema_scale;
                if !self.rebalance_inplace(&mut adjusted) {
                    fallback = true;
                }
            }
        } else {
            self.ema_scale = self.smoothing * 1.0 + (1.0 - self.smoothing) * self.ema_scale;
            if !self.rebalance_inplace(&mut adjusted) {
                fallback = true;
            }
        }

        if fallback {
            base
        } else {
            adjusted
        }
    }

    fn compute_target_coherence(&self, frame: &PsiMetricsFrame) -> Option<f32> {
        let reading = frame.reading.as_ref()?;
        let breakdown = &reading.breakdown;
        let drift = breakdown
            .get(&PsiComponent::ACT_DRIFT)
            .copied()
            .unwrap_or_default();
        let band_energy = breakdown
            .get(&PsiComponent::BAND_ENERGY)
            .copied()
            .unwrap_or_default();
        let curvature = breakdown
            .get(&PsiComponent::POSITIVE_CURVATURE)
            .copied()
            .unwrap_or_default();
        let grad_norm = breakdown
            .get(&PsiComponent::GRAD_NORM)
            .copied()
            .unwrap_or_default();
        let update_ratio = breakdown
            .get(&PsiComponent::UPDATE_RATIO)
            .copied()
            .unwrap_or_default();
        let stability = frame
            .spiral
            .as_ref()
            .map(|advisory| advisory.stability_score())
            .or_else(|| frame.tuning.as_ref().map(|t| t.stability_score))
            .unwrap_or(1.0);

        let energy_term = (1.0 + band_energy.abs()).sqrt();
        let curvature_term = 1.0 + curvature.clamp(0.0, 1.75);
        let drift_damping = 1.0 / (1.0 + drift.abs());
        let update_damping = 1.0 / (1.0 + update_ratio.abs() + grad_norm.abs());
        let stability_term = (0.35 + 0.65 * stability).clamp(0.35, 1.6);
        let total_term = (1.0 + reading.total.abs()).clamp(0.65, 2.15);

        let coherence =
            energy_term * curvature_term * stability_term * drift_damping * update_damping
                / total_term;
        if coherence.is_finite() {
            Some(coherence)
        } else {
            None
        }
    }

    fn rebalance_inplace(&self, weights: &mut [f32]) -> bool {
        let mut biased_mass = 0.0f32;
        for (idx, weight) in weights.iter_mut().enumerate() {
            let bias = 1.0 + self.hop_bias * idx as f32;
            *weight = (*weight * bias).clamp(-self.clip, self.clip);
            biased_mass += weight.abs();
        }
        if biased_mass <= self.epsilon {
            return false;
        }
        let inv_mass = 1.0 / biased_mass;
        for weight in weights.iter_mut() {
            *weight *= inv_mass;
        }
        true
    }
}

impl Default for PsiCoherenceAdaptor {
    fn default() -> Self {
        Self::with_subscription(Some(subscribe_psi_metrics()))
    }
}

impl fmt::Debug for PsiCoherenceAdaptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PsiCoherenceAdaptor")
            .field("clip", &self.clip)
            .field("min_scale", &self.min_scale)
            .field("max_scale", &self.max_scale)
            .field("smoothing", &self.smoothing)
            .field("ema_scale", &self.ema_scale)
            .field("hop_bias", &self.hop_bias)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebalances_without_subscription() {
        let mut adaptor = PsiCoherenceAdaptor::with_subscription(None);
        let base = vec![0.0, 2.0, 1.0];
        let coherent = adaptor.cohere_weights(base.clone());
        let mass: f32 = coherent.iter().map(|w| w.abs()).sum();
        assert!((mass - 1.0).abs() < 1.0e-5);
        let zeros = vec![0.0, 0.0];
        assert_eq!(adaptor.cohere_weights(zeros.clone()), zeros);
    }
}
