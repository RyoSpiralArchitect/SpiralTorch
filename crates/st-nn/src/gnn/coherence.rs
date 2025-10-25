// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::collections::VecDeque;
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
    history_limit: usize,
    scale_history: VecDeque<f32>,
    diagnostics: PsiCoherenceDiagnostics,
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
            history_limit: 12,
            scale_history: VecDeque::with_capacity(12),
            diagnostics: PsiCoherenceDiagnostics::default(),
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
        let mut diagnostics = PsiCoherenceDiagnostics {
            base_mass: base.iter().map(|w| w.abs()).sum(),
            ..PsiCoherenceDiagnostics::default()
        };
        if let Some(subscription) = self.subscription.as_ref() {
            let frame = subscription.snapshot();
            diagnostics.events_applied = frame.events.len();
            let event_modulation = self.event_modulation(&frame.events);
            diagnostics.event_modulation = event_modulation;
            diagnostics.telemetry_active = frame.reading.is_some();
            if let Some(scale) = self.compute_target_coherence(&frame) {
                let clamped = scale.clamp(self.min_scale, self.max_scale) * event_modulation;
                self.ema_scale = self.smoothing * clamped + (1.0 - self.smoothing) * self.ema_scale;
                let scale = self.ema_scale;
                diagnostics.applied_scale = scale;
                for weight in &mut adjusted {
                    *weight = (*weight * scale).clamp(-self.clip, self.clip);
                }
                if !self.rebalance_inplace(&mut adjusted) {
                    fallback = true;
                }
            } else {
                self.ema_scale = self.smoothing * 1.0 + (1.0 - self.smoothing) * self.ema_scale;
                diagnostics.applied_scale = self.ema_scale;
                if !self.rebalance_inplace(&mut adjusted) {
                    fallback = true;
                }
            }
        } else {
            self.ema_scale = self.smoothing * 1.0 + (1.0 - self.smoothing) * self.ema_scale;
            diagnostics.applied_scale = self.ema_scale;
            diagnostics.telemetry_active = false;
            if !self.rebalance_inplace(&mut adjusted) {
                fallback = true;
            }
        }

        if fallback {
            diagnostics.fallback = true;
            diagnostics.ema_scale = self.ema_scale;
            diagnostics.adjusted_mass = diagnostics.base_mass;
            self.diagnostics = diagnostics;
            base
        } else {
            diagnostics.adjusted_mass = adjusted.iter().map(|w| w.abs()).sum();
            diagnostics.ema_scale = self.ema_scale;
            self.record_scale(diagnostics.applied_scale);
            self.apply_history_stats(&mut diagnostics);
            self.diagnostics = diagnostics;
            adjusted
        }
    }

    /// Returns the most recent diagnostics snapshot emitted by [`Self::cohere_weights`].
    pub fn last_diagnostics(&self) -> &PsiCoherenceDiagnostics {
        &self.diagnostics
    }

    /// Clears the exponential smoothing accumulator and the rolling history window.
    pub fn reset(&mut self) {
        self.ema_scale = 1.0;
        self.scale_history.clear();
        self.diagnostics = PsiCoherenceDiagnostics::default();
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

    fn record_scale(&mut self, scale: f32) {
        if !scale.is_finite() {
            return;
        }
        if self.scale_history.len() == self.history_limit {
            self.scale_history.pop_front();
        }
        self.scale_history.push_back(scale);
    }

    fn apply_history_stats(&self, diagnostics: &mut PsiCoherenceDiagnostics) {
        if self.scale_history.is_empty() {
            return;
        }
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f32;
        for value in &self.scale_history {
            min = min.min(*value);
            max = max.max(*value);
            sum += *value;
        }
        diagnostics.history_min = Some(min);
        diagnostics.history_max = Some(max);
        let len = self.scale_history.len() as f32;
        let mean = sum / len;
        let mut variance = 0.0f32;
        for value in &self.scale_history {
            let delta = *value - mean;
            variance += delta * delta;
        }
        variance /= len;
        diagnostics.history_std = Some(variance.sqrt());
    }

    fn event_modulation(&self, events: &[st_core::telemetry::psi::PsiEvent]) -> f32 {
        use st_core::telemetry::psi::PsiEvent;
        if events.is_empty() {
            return 1.0;
        }
        let mut modulation = 1.0f32;
        for event in events {
            match event {
                PsiEvent::ThresholdCross { component, up, .. } => {
                    let delta = match *component {
                        PsiComponent::ACT_DRIFT | PsiComponent::UPDATE_RATIO => {
                            if *up {
                                -0.12
                            } else {
                                0.06
                            }
                        }
                        PsiComponent::GRAD_NORM => {
                            if *up {
                                -0.08
                            } else {
                                0.04
                            }
                        }
                        PsiComponent::BAND_ENERGY | PsiComponent::ATTN_ENTROPY => {
                            if *up {
                                0.1
                            } else {
                                -0.05
                            }
                        }
                        PsiComponent::LOSS => {
                            if *up {
                                -0.05
                            } else {
                                0.02
                            }
                        }
                        PsiComponent::POSITIVE_CURVATURE => {
                            if *up {
                                0.07
                            } else {
                                -0.03
                            }
                        }
                        _ => 0.0,
                    };
                    modulation *= 1.0 + delta;
                }
            }
        }
        modulation.clamp(0.6, 1.6)
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
            .field("history_limit", &self.history_limit)
            .field("history_len", &self.scale_history.len())
            .finish()
    }
}

/// Snapshot of the last coherence pass, exposing modulation metadata for downstream debugging.
#[derive(Debug, Clone)]
pub struct PsiCoherenceDiagnostics {
    pub applied_scale: f32,
    pub ema_scale: f32,
    pub event_modulation: f32,
    pub events_applied: usize,
    pub fallback: bool,
    pub telemetry_active: bool,
    pub base_mass: f32,
    pub adjusted_mass: f32,
    pub history_min: Option<f32>,
    pub history_max: Option<f32>,
    pub history_std: Option<f32>,
}

impl Default for PsiCoherenceDiagnostics {
    fn default() -> Self {
        Self {
            applied_scale: 0.0,
            ema_scale: 0.0,
            event_modulation: 1.0,
            events_applied: 0,
            fallback: false,
            telemetry_active: false,
            base_mass: 0.0,
            adjusted_mass: 0.0,
            history_min: None,
            history_max: None,
            history_std: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use st_core::telemetry::hub::{
        clear_last_psi, clear_last_psi_events, set_last_psi, set_last_psi_events,
    };
    use st_core::telemetry::psi::{PsiEvent, PsiReading};
    use std::collections::HashMap;

    #[test]
    fn rebalances_without_subscription() {
        let mut adaptor = PsiCoherenceAdaptor::with_subscription(None);
        let base = vec![0.0, 2.0, 1.0];
        let coherent = adaptor.cohere_weights(base.clone());
        let mass: f32 = coherent.iter().map(|w| w.abs()).sum();
        assert!((mass - 1.0).abs() < 1.0e-5);
        let zeros = vec![0.0, 0.0];
        assert_eq!(adaptor.cohere_weights(zeros.clone()), zeros);
        let diagnostics = adaptor.last_diagnostics();
        assert!(diagnostics.fallback);
        assert!(!diagnostics.telemetry_active);
        assert!(diagnostics.history_std.is_none());
    }

    #[test]
    fn diagnostics_track_modulation_from_events() {
        clear_last_psi();
        clear_last_psi_events();
        let subscription = subscribe_psi_metrics();
        let mut adaptor = PsiCoherenceAdaptor::with_subscription(Some(subscription));
        let mut breakdown = HashMap::new();
        breakdown.insert(PsiComponent::ACT_DRIFT, 0.35);
        breakdown.insert(PsiComponent::BAND_ENERGY, 0.8);
        breakdown.insert(PsiComponent::POSITIVE_CURVATURE, 0.45);
        breakdown.insert(PsiComponent::GRAD_NORM, 0.2);
        breakdown.insert(PsiComponent::UPDATE_RATIO, 0.15);
        let reading = PsiReading {
            total: 0.95,
            breakdown,
            step: 42,
        };
        set_last_psi(&reading);
        let events = vec![
            PsiEvent::ThresholdCross {
                component: PsiComponent::ACT_DRIFT,
                value: 0.35,
                threshold: 0.25,
                up: true,
                step: 42,
            },
            PsiEvent::ThresholdCross {
                component: PsiComponent::BAND_ENERGY,
                value: 0.8,
                threshold: 0.5,
                up: true,
                step: 42,
            },
        ];
        set_last_psi_events(&events);
        let base = vec![0.4, 0.6, 0.2];
        let adjusted = adaptor.cohere_weights(base.clone());
        let diagnostics = adaptor.last_diagnostics().clone();
        assert!(!diagnostics.fallback);
        assert_eq!(diagnostics.events_applied, 2);
        assert!(diagnostics.event_modulation > 0.6 && diagnostics.event_modulation < 1.6);
        assert!(diagnostics.telemetry_active);
        assert!(diagnostics.history_std.unwrap() >= 0.0);
        let adjusted_mass: f32 = adjusted.iter().map(|w| w.abs()).sum();
        assert!((adjusted_mass - 1.0).abs() < 1.0e-5);
        clear_last_psi_events();
        clear_last_psi();
    }
}
