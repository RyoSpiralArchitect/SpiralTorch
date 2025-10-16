// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::collections::VecDeque;

use st_core::telemetry::chrono::{ChronoHarmonics, ChronoLoopSignal, ChronoSummary};
use st_core::telemetry::hub::LoopbackEnvelope;
use st_core::theory::observability::{
    ObservabilityAssessment, ObservabilityConfig, ObservationalCoalgebra, SlotSymmetry,
};
use st_core::util::math::{LeechProjector, LEECH_PACKING_DENSITY};
use st_tensor::{DifferentialResonance, Tensor};

/// Configuration describing how geometric observability is converted into
/// feedback for the learning loop.
#[derive(Clone, Debug)]
pub struct GeometryFeedbackConfig {
    /// Observability blueprint used to build the coalgebra.
    pub observability: ObservabilityConfig,
    /// Absolute value threshold used to consider a tensor activation "visible".
    pub activation_threshold: f32,
    /// Number of efficiency measurements folded into the moving average.
    pub smoothing_window: usize,
    /// Minimum multiplicative factor applied to the learning rate.
    pub min_learning_rate_scale: f32,
    /// Maximum multiplicative factor applied to the learning rate.
    pub max_learning_rate_scale: f32,
    /// Target rank of the Z-space slice we are probing (defaults to the Leech shell).
    pub z_space_rank: usize,
    /// Weight applied to the Leech lattice density correction when scaling η.
    pub leech_density_weight: f64,
    /// Number of terms used in the Ramanujan π estimator.
    pub ramanujan_iterations: usize,
    /// Softening factor controlling how aggressively η is pushed into [0, 1].
    pub softening_beta: f32,
}

impl GeometryFeedbackConfig {
    /// Provides a conservative default configuration tuned for the policy
    /// gradient helper.
    pub fn default_policy() -> Self {
        Self {
            observability: ObservabilityConfig::new(1, 5, SlotSymmetry::Symmetric),
            activation_threshold: 1e-3,
            smoothing_window: 4,
            min_learning_rate_scale: 0.5,
            max_learning_rate_scale: 2.5,
            z_space_rank: 24,
            leech_density_weight: 0.35,
            ramanujan_iterations: 3,
            softening_beta: 0.75,
        }
    }
}

/// Smoothed geometry report emitted after processing a resonance snapshot.
#[derive(Clone, Debug)]
pub struct GeometryFeedbackSignal {
    /// Detailed comparison between observed and theoretical structure.
    pub assessment: ObservabilityAssessment,
    /// Moving-average of the efficiency ratios used for scaling.
    pub averaged_efficiency: f64,
    /// Smoothed estimate of the effective rank being activated in Z-space.
    pub smoothed_rank: f64,
    /// Smoothed Leech packing pressure contributing to the scale.
    pub smoothed_pressure: f64,
    /// Multiplicative factor suggested for the learning rate.
    pub learning_rate_scale: f32,
    /// Rolling-average of the emitted scale to help detect drift.
    pub rolling_scale: f32,
}

/// Rolling diagnostics describing how the feedback loop is adapting itself.
#[derive(Clone, Debug, Default)]
pub struct GeometryTelemetry {
    /// Smoothed estimate of the effective rank under observation.
    pub rolling_rank: f64,
    /// Smoothed Leech packing pressure after weighting.
    pub rolling_pressure: f64,
    /// Rolling-average of the emitted learning-rate scale.
    pub rolling_scale: f32,
    /// Current minimum learning-rate multiplier after self-tuning.
    pub min_scale: f32,
    /// Current maximum learning-rate multiplier after self-tuning.
    pub max_scale: f32,
    /// Active Leech density weight following self-rewrites.
    pub leech_density_weight: f64,
    /// Loopback gain distilled from the latest chrono loop signal.
    pub loop_gain: f32,
    /// Collapse pressure inferred from collapse drive pulses or decay drift.
    pub collapse_bias: f32,
    /// Timestamp of the most recent loopback modulation.
    pub loop_timestamp: f32,
    /// Active softening factor after loopback rewrites.
    pub softening_beta: f32,
    /// Aggregated participation weight from loopback broadcasts.
    pub loop_support: f32,
    /// Latest SpiralK script snippet attached to the loopback envelope.
    pub loop_script: Option<String>,
}

/// Coalgebra-powered feedback loop that measures resonance geometry and emits
/// learning rate adjustments.
#[derive(Clone, Debug)]
pub struct GeometryFeedback {
    coalgebra: ObservationalCoalgebra,
    threshold: f32,
    history: VecDeque<f64>,
    window: usize,
    min_scale: f32,
    max_scale: f32,
    z_rank: usize,
    leech_weight: f64,
    leech_projector: LeechProjector,
    ramanujan_pi: f64,
    softening_beta: f32,
    base_softening_beta: f32,
    rank_history: VecDeque<f64>,
    pressure_history: VecDeque<f64>,
    scale_history: VecDeque<f32>,
    telemetry: GeometryTelemetry,
    last_signal: Option<GeometryFeedbackSignal>,
    loop_gain: f32,
    collapse_bias: f32,
    loop_timestamp: f32,
    loop_fresh: bool,
    loop_support: f32,
    loop_script: Option<String>,
}

impl GeometryFeedback {
    /// Builds a new feedback controller from the provided configuration.
    pub fn new(config: GeometryFeedbackConfig) -> Self {
        let window = config.smoothing_window.max(1);
        let (min_scale, max_scale) =
            if config.min_learning_rate_scale <= config.max_learning_rate_scale {
                (
                    config.min_learning_rate_scale,
                    config.max_learning_rate_scale,
                )
            } else {
                (
                    config.max_learning_rate_scale,
                    config.min_learning_rate_scale,
                )
            };
        let mut min_scale = min_scale.max(f32::EPSILON);
        let max_scale = max_scale.max(f32::EPSILON);
        let mut clamped_max = max_scale.clamp(2.0, 3.0);
        if min_scale >= clamped_max {
            clamped_max = (min_scale + 2.0).clamp(2.0, 3.0);
        }
        min_scale = min_scale.min(clamped_max - f32::EPSILON).max(f32::EPSILON);
        let z_rank = config.z_space_rank.max(1);
        let leech_weight = config.leech_density_weight.max(0.0);
        let ramanujan_pi = Self::ramanujan_pi(config.ramanujan_iterations.max(1));
        let softening_beta = config.softening_beta.max(0.0);
        let leech_projector = LeechProjector::new(z_rank, leech_weight);

        Self {
            coalgebra: ObservationalCoalgebra::new(config.observability),
            threshold: config.activation_threshold.abs().max(f32::EPSILON),
            history: VecDeque::with_capacity(window),
            window,
            min_scale,
            max_scale: clamped_max,
            z_rank,
            leech_weight,
            leech_projector,
            ramanujan_pi,
            softening_beta,
            base_softening_beta: softening_beta,
            rank_history: VecDeque::with_capacity(window),
            pressure_history: VecDeque::with_capacity(window),
            scale_history: VecDeque::with_capacity(window),
            telemetry: GeometryTelemetry::default(),
            last_signal: None,
            loop_gain: 0.0,
            collapse_bias: 0.0,
            loop_timestamp: 0.0,
            loop_fresh: false,
            loop_support: 0.0,
            loop_script: None,
        }
    }

    /// Returns the most recent signal emitted by the controller.
    pub fn last_signal(&self) -> Option<&GeometryFeedbackSignal> {
        self.last_signal.as_ref()
    }

    /// Returns rolling diagnostics summarising the most recent updates.
    pub fn telemetry(&self) -> &GeometryTelemetry {
        &self.telemetry
    }

    /// Integrates a chrono loop signal so temporal harmonics can steer the
    /// self-rewrites before the next resonance measurement.
    pub fn integrate_loop_signal(&mut self, signal: &ChronoLoopSignal) {
        let summary = &signal.summary;
        let drift = summary.mean_abs_drift.abs();
        let energy_std = summary.energy_std.max(0.0);
        let energy_band = (summary.max_energy - summary.min_energy).max(0.0);
        let mut gain = drift + energy_std.sqrt() + energy_band.powf(0.25);
        if let Some(harmonics) = &signal.harmonics {
            if let Some(peak) = &harmonics.dominant_drift {
                gain += peak.magnitude;
            }
            if let Some(peak) = &harmonics.dominant_energy {
                gain += peak.magnitude * 0.75;
            }
        }
        self.loop_gain = gain.clamp(0.0, 12.0);
        let decay_bias = (-summary.mean_decay).max(0.0).clamp(0.0, 6.0);
        self.collapse_bias = self.collapse_bias.max(decay_bias);
        self.loop_timestamp = summary.latest_timestamp;
        self.loop_support = summary.frames as f32;
        if let Some(script) = Self::signal_script(signal) {
            self.loop_script = Some(script);
        }
        let normalized = (self.loop_gain / 12.0).clamp(0.0, 1.0);
        let target_beta = 0.6 + normalized * 1.4;
        self.softening_beta = (self.softening_beta * 0.7 + target_beta * 0.3).clamp(0.3, 4.0);
        self.loop_fresh = true;
    }

    /// Integrates aggregated loopback envelopes emitted by other SpiralTorch nodes.
    pub fn absorb_loopback(&mut self, envelopes: &[LoopbackEnvelope]) {
        if envelopes.is_empty() {
            return;
        }
        let mut total_support = 0.0f32;
        let mut duration_acc = 0.0f32;
        let mut frames_total: usize = 0;
        let mut drift_acc = 0.0f32;
        let mut abs_drift_acc = 0.0f32;
        let mut drift_std_acc = 0.0f32;
        let mut energy_acc = 0.0f32;
        let mut energy_std_acc = 0.0f32;
        let mut decay_acc = 0.0f32;
        let mut min_energy = f32::INFINITY;
        let mut max_energy = f32::NEG_INFINITY;
        let mut latest_ts = self.loop_timestamp;
        let mut collapse_acc = 0.0f32;
        let mut collapse_weight = 0.0f32;
        let mut z_acc = 0.0f32;
        let mut z_weight = 0.0f32;
        let mut best_harmonics: Option<ChronoHarmonics> = None;
        let mut best_score = f32::NEG_INFINITY;
        let mut fallback_script: Option<String> = None;

        for envelope in envelopes {
            let support = envelope.support.max(f32::EPSILON);
            let summary = &envelope.loop_signal.summary;
            total_support += support;
            duration_acc += summary.duration * support;
            frames_total = frames_total.saturating_add(summary.frames);
            drift_acc += summary.mean_drift * support;
            abs_drift_acc += summary.mean_abs_drift * support;
            drift_std_acc += summary.drift_std.powi(2) * support;
            energy_acc += summary.mean_energy * support;
            energy_std_acc += summary.energy_std.powi(2) * support;
            decay_acc += summary.mean_decay * support;
            min_energy = min_energy.min(summary.min_energy);
            max_energy = max_energy.max(summary.max_energy);
            latest_ts = latest_ts.max(summary.latest_timestamp);
            if let Some(total) = envelope.collapse_total {
                let positive = total.abs().sqrt().clamp(0.0, 6.0);
                collapse_acc += positive * support;
                collapse_weight += support;
            }
            if let Some(z) = envelope.z_signal {
                z_acc += z * support;
                z_weight += support;
            }
            if let Some(harmonics) = &envelope.loop_signal.harmonics {
                let drift_peak = harmonics
                    .dominant_drift
                    .as_ref()
                    .map(|peak| peak.magnitude)
                    .unwrap_or(0.0);
                let energy_peak = harmonics
                    .dominant_energy
                    .as_ref()
                    .map(|peak| peak.magnitude)
                    .unwrap_or(0.0);
                let score = (drift_peak + energy_peak) * support;
                if score > best_score {
                    best_score = score;
                    best_harmonics = Some(harmonics.clone());
                }
            }
            if fallback_script.is_none() {
                if let Some(script) = Self::signal_script(&envelope.loop_signal) {
                    fallback_script = Some(script);
                } else if let Some(script) = envelope.script_hint.clone() {
                    fallback_script = Some(script);
                }
            }
        }

        if total_support <= f32::EPSILON {
            return;
        }

        let inv = 1.0 / total_support;
        let summary = ChronoSummary {
            frames: frames_total.max(1),
            duration: (duration_acc * inv).max(f32::EPSILON),
            latest_timestamp: latest_ts,
            mean_drift: drift_acc * inv,
            mean_abs_drift: abs_drift_acc * inv,
            drift_std: (drift_std_acc * inv).max(0.0).sqrt(),
            mean_energy: energy_acc * inv,
            energy_std: (energy_std_acc * inv).max(0.0).sqrt(),
            mean_decay: decay_acc * inv,
            min_energy: if min_energy.is_finite() {
                min_energy
            } else {
                0.0
            },
            max_energy: if max_energy.is_finite() {
                max_energy
            } else {
                0.0
            },
        };

        let aggregated = ChronoLoopSignal::new(summary, best_harmonics);
        self.integrate_loop_signal(&aggregated);
        self.loop_support = total_support;
        if let Some(script) = fallback_script {
            if Self::signal_script(&aggregated).is_none() {
                self.loop_script = Some(script);
            }
        }
        if collapse_weight > 0.0 {
            let mean_collapse = (collapse_acc / collapse_weight).clamp(0.0, 6.0);
            self.collapse_bias = self.collapse_bias.max(mean_collapse);
            self.loop_fresh = true;
        }
        if z_weight > 0.0 {
            let mean_z = (z_acc / z_weight).clamp(-2.0, 2.0);
            let target = self.base_softening_beta + mean_z * 0.4;
            self.softening_beta = (self.softening_beta * 0.85 + target * 0.15).clamp(0.3, 4.0);
        }
        if total_support > 1.0 {
            let gain_scale = (total_support / envelopes.len() as f32).clamp(0.5, 6.0);
            self.loop_gain = (self.loop_gain * gain_scale).clamp(0.0, 12.0);
        }
    }

    /// Injects collapse drive pressure so the Leech weighting can respond to
    /// structural emergencies even when no loop signal is available.
    pub fn inject_collapse_bias(&mut self, intensity: f32) {
        if !intensity.is_finite() {
            return;
        }
        let bias = intensity.abs().sqrt().clamp(0.0, 6.0);
        self.collapse_bias = self.collapse_bias.max(bias);
        self.loop_fresh = true;
    }

    /// Measures the provided resonance snapshot and emits a smoothed feedback
    /// signal. The controller keeps internal state so repeated calls fold into
    /// the moving average.
    pub fn process_resonance(
        &mut self,
        resonance: &DifferentialResonance,
    ) -> GeometryFeedbackSignal {
        let observed = self.observed_counts(resonance);
        let assessment = self.coalgebra.assess(&observed);
        let mean_efficiency = if assessment.efficiency.is_empty() {
            1.0
        } else {
            assessment.efficiency.iter().sum::<f64>() / assessment.efficiency.len() as f64
        };
        self.history.push_back(mean_efficiency);
        while self.history.len() > self.window {
            self.history.pop_front();
        }
        let averaged = self.history.iter().copied().sum::<f64>() / self.history.len() as f64;
        let geodesic = self.geodesic_projection(resonance);
        let densified =
            self.leech_weight * LEECH_PACKING_DENSITY * geodesic * (self.z_rank as f64).sqrt();
        let normalized = ((averaged + densified) / self.ramanujan_pi).clamp(0.0, 1.0);
        let softened = self.soft_project(normalized as f32);
        let mut scale = self.min_scale + (self.max_scale - self.min_scale) * softened;

        let rank_estimate = self.rank_estimate(&observed, &assessment);
        self.rank_history.push_back(rank_estimate);
        while self.rank_history.len() > self.window {
            self.rank_history.pop_front();
        }
        let smoothed_rank =
            self.rank_history.iter().copied().sum::<f64>() / self.rank_history.len() as f64;

        self.pressure_history.push_back(densified);
        while self.pressure_history.len() > self.window {
            self.pressure_history.pop_front();
        }
        let smoothed_pressure =
            self.pressure_history.iter().copied().sum::<f64>() / self.pressure_history.len() as f64;

        self.scale_history.push_back(scale);
        while self.scale_history.len() > self.window {
            self.scale_history.pop_front();
        }
        let mut rolling_scale =
            self.scale_history.iter().copied().sum::<f32>() / self.scale_history.len() as f32;

        self.auto_tune(smoothed_rank, smoothed_pressure, rolling_scale);

        scale = scale.clamp(self.min_scale, self.max_scale);
        if let Some(last) = self.scale_history.back_mut() {
            *last = scale;
        }
        rolling_scale =
            self.scale_history.iter().copied().sum::<f32>() / self.scale_history.len() as f32;

        self.telemetry = GeometryTelemetry {
            rolling_rank: smoothed_rank,
            rolling_pressure: smoothed_pressure,
            rolling_scale,
            min_scale: self.min_scale,
            max_scale: self.max_scale,
            leech_density_weight: self.leech_weight,
            loop_gain: self.loop_gain,
            collapse_bias: self.collapse_bias,
            loop_timestamp: self.loop_timestamp,
            softening_beta: self.softening_beta,
            loop_support: self.loop_support,
            loop_script: self.loop_script.clone(),
        };

        let signal = GeometryFeedbackSignal {
            assessment,
            averaged_efficiency: averaged,
            smoothed_rank,
            smoothed_pressure,
            learning_rate_scale: scale.max(self.min_scale),
            rolling_scale,
        };
        self.last_signal = Some(signal.clone());
        signal
    }

    fn observed_counts(&mut self, resonance: &DifferentialResonance) -> Vec<u128> {
        vec![
            Self::count_active(&resonance.homotopy_flow, self.threshold),
            Self::count_active(&resonance.functor_linearisation, self.threshold),
            Self::count_active(&resonance.recursive_objective, self.threshold),
            Self::count_active(&resonance.infinity_projection, self.threshold),
            Self::count_active(&resonance.infinity_energy, self.threshold),
        ]
    }

    fn count_active(tensor: &Tensor, threshold: f32) -> u128 {
        tensor
            .data()
            .iter()
            .filter(|value| value.abs() >= threshold)
            .count() as u128
    }

    fn geodesic_projection(&self, resonance: &DifferentialResonance) -> f64 {
        let norms = [
            Self::l2_norm(&resonance.homotopy_flow),
            Self::l2_norm(&resonance.functor_linearisation),
            Self::l2_norm(&resonance.recursive_objective),
            Self::l2_norm(&resonance.infinity_projection),
            Self::l2_norm(&resonance.infinity_energy),
        ];
        norms.iter().sum::<f64>() / norms.len() as f64
    }

    fn l2_norm(tensor: &Tensor) -> f64 {
        tensor
            .data()
            .iter()
            .map(|value| (*value as f64).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn soft_project(&self, value: f32) -> f32 {
        let beta = (1.0 + self.softening_beta).max(f32::EPSILON);
        value.clamp(0.0, 1.0).powf(1.0 / beta)
    }

    fn rank_estimate(&self, observed: &[u128], assessment: &ObservabilityAssessment) -> f64 {
        let observed_total: f64 = observed.iter().map(|value| *value as f64).sum();
        let expected_total: f64 = assessment
            .expected
            .iter()
            .map(|value| *value as f64)
            .sum::<f64>()
            .max(f64::MIN_POSITIVE);
        let ratio = (observed_total / expected_total).clamp(0.0, 1.0);
        ratio * self.z_rank as f64
    }

    fn auto_tune(&mut self, rank: f64, pressure: f64, rolling_scale: f32) {
        let recommended = self.max_scale.clamp(2.0, 3.0);
        if (self.max_scale - recommended).abs() > f32::EPSILON {
            self.max_scale = recommended.max(self.min_scale + f32::EPSILON);
        }

        let max_pressure = CORE_LEECH_PACKING_DENSITY * (self.z_rank as f64).sqrt();
        if max_pressure > 0.0 {
            let pressure_ratio = (pressure / max_pressure).clamp(0.0, 4.0);
            if pressure_ratio > 1.2 {
                self.leech_weight = (self.leech_weight * 0.9).max(0.0);
            } else if pressure_ratio < 0.6 {
                self.leech_weight = (self.leech_weight * 1.1).max(0.0).min(16.0);
            }
        }

        let target_rank = self.z_rank as f64 * 0.85;
        if rank < target_rank {
            self.min_scale = (self.min_scale * 1.05).min(self.max_scale - f32::EPSILON);
        } else if rank > self.z_rank as f64 * 0.97 {
            self.min_scale = (self.min_scale * 0.95).max(f32::EPSILON);
        }

        if rolling_scale > self.max_scale * 0.92 {
            self.max_scale = (self.max_scale - 0.05).max(2.0);
        } else if rolling_scale < (self.min_scale + self.max_scale) * 0.25 {
            self.max_scale = (self.max_scale + 0.05).min(3.0);
        }

        if self.loop_fresh {
            let normalized = (self.loop_gain / 8.0).clamp(0.0, 1.5);
            let tighten = normalized.min(1.0);
            let relax = (1.0 - tighten).max(0.0);
            self.max_scale = (self.max_scale - 0.08 * tighten).max(2.0);
            if tighten > 0.3 {
                self.min_scale = (self.min_scale * (1.0 - 0.06 * tighten)).max(f32::EPSILON);
            } else {
                self.min_scale = (self.min_scale * (1.0 + 0.05 * relax))
                    .min(self.max_scale - f32::EPSILON)
                    .max(f32::EPSILON);
            }
            let collapse_push = (self.collapse_bias / 6.0).clamp(0.0, 1.0) as f64;
            let pressure_gain = 1.0 + collapse_push * 0.4 + (tighten as f64) * 0.3;
            self.leech_weight = (self.leech_weight * pressure_gain).clamp(0.0, 32.0);
            self.loop_fresh = false;
        } else {
            self.loop_gain *= 0.94;
            self.collapse_bias *= 0.9;
            self.softening_beta =
                (self.softening_beta * 0.95 + self.base_softening_beta * 0.05).clamp(0.3, 4.0);
            self.leech_weight = (self.leech_weight * 0.995).max(0.0);
        }

        self.max_scale = self.max_scale.clamp(2.0, 3.0);
        if self.min_scale >= self.max_scale {
            self.min_scale = (self.max_scale * 0.5).max(f32::EPSILON);
        }

        self.leech_projector = LeechProjector::new(self.z_rank, self.leech_weight);
    }

    fn ramanujan_pi(iterations: usize) -> f64 {
        let mut sum = 0.0;
        let mut factor = 1.0;
        let base = 396_f64.powi(4);
        for k in 0..iterations {
            sum += factor * (1103.0 + 26390.0 * k as f64);
            let k1 = k + 1;
            let numerator =
                (4 * k1 - 3) as f64 * (4 * k1 - 2) as f64 * (4 * k1 - 1) as f64 * (4 * k1) as f64;
            let denominator = (k1 as f64).powi(4) * base;
            factor *= numerator / denominator;
        }
        let prefactor = (2.0 * 2.0_f64.sqrt()) / 9801.0;
        (prefactor * sum).recip()
    }
}

const LEECH_PACKING_DENSITY: f64 = 0.001_929_574_309_403_922_5;

#[cfg(test)]
mod tests {
    use super::*;
    use st_core::telemetry::chrono::{
        ChronoHarmonics, ChronoLoopSignal, ChronoPeak, ChronoSummary,
    };
    use st_core::theory::observability::SlotSymmetry;

    fn tensor_from(values: &[f32]) -> Tensor {
        Tensor::from_vec(1, values.len(), values.to_vec()).unwrap()
    }

    fn resonance_from(values: &[f32]) -> DifferentialResonance {
        let tensor = tensor_from(values);
        DifferentialResonance {
            homotopy_flow: tensor.clone(),
            functor_linearisation: tensor.clone(),
            recursive_objective: tensor.clone(),
            infinity_projection: tensor.clone(),
            infinity_energy: tensor,
        }
    }

    #[test]
    fn geometry_feedback_smoothes_efficiency() {
        let config = GeometryFeedbackConfig {
            observability: ObservabilityConfig::new(1, 5, SlotSymmetry::Symmetric),
            activation_threshold: 0.05,
            smoothing_window: 2,
            min_learning_rate_scale: 0.5,
            max_learning_rate_scale: 1.0,
            z_space_rank: 24,
            leech_density_weight: 0.4,
            ramanujan_iterations: 3,
            softening_beta: 0.5,
        };
        let mut feedback = GeometryFeedback::new(config.clone());
        let resonance_high = resonance_from(&[0.5, 0.4, -0.3, 0.2, 0.1]);
        let resonance_low = resonance_from(&[0.01, 0.02, -0.01, 0.0, 0.03]);

        let first = feedback.process_resonance(&resonance_high);
        assert!(first.learning_rate_scale > config.min_learning_rate_scale);
        assert!(first.smoothed_pressure > 0.0);
        assert!(first.smoothed_rank <= config.z_space_rank as f64);

        let second = feedback.process_resonance(&resonance_low);
        assert!(second.learning_rate_scale <= first.learning_rate_scale);
        assert!(second.rolling_scale >= config.min_learning_rate_scale);
        assert!(feedback.last_signal().is_some());
        let telemetry = feedback.telemetry().clone();
        assert!(telemetry.rolling_scale >= config.min_learning_rate_scale);
        assert!(telemetry.max_scale <= 3.0);
    }

    #[test]
    fn leech_density_enriches_scaling() {
        let resonance = resonance_from(&[0.2, -0.4, 0.6, -0.8, 1.0]);
        let base_config = GeometryFeedbackConfig {
            observability: ObservabilityConfig::new(1, 5, SlotSymmetry::Symmetric),
            activation_threshold: 0.05,
            smoothing_window: 2,
            min_learning_rate_scale: 0.5,
            max_learning_rate_scale: 1.5,
            z_space_rank: 4,
            leech_density_weight: 0.0,
            ramanujan_iterations: 3,
            softening_beta: 0.5,
        };
        let mut base = GeometryFeedback::new(base_config.clone());
        let mut enriched_config = base_config;
        enriched_config.z_space_rank = 24;
        enriched_config.leech_density_weight = 2.0;
        let mut enriched = GeometryFeedback::new(enriched_config);

        let base_scale = base.process_resonance(&resonance).learning_rate_scale;
        let enriched_scale = enriched.process_resonance(&resonance).learning_rate_scale;
        assert!(enriched_scale >= base_scale);
    }

    #[test]
    fn telemetry_auto_tunes_clamp_and_pressure() {
        let config = GeometryFeedbackConfig {
            observability: ObservabilityConfig::new(1, 5, SlotSymmetry::Symmetric),
            activation_threshold: 0.01,
            smoothing_window: 3,
            min_learning_rate_scale: 0.4,
            max_learning_rate_scale: 8.0,
            z_space_rank: 24,
            leech_density_weight: 1.5,
            ramanujan_iterations: 3,
            softening_beta: 0.6,
        };
        let mut feedback = GeometryFeedback::new(config);
        let resonance = resonance_from(&[1.2, 1.0, -0.9, 0.8, -0.7]);

        for _ in 0..5 {
            feedback.process_resonance(&resonance);
        }

        let telemetry = feedback.telemetry().clone();
        assert!(telemetry.max_scale <= 3.0);
        assert!(telemetry.max_scale >= 2.0);
        assert!(telemetry.leech_density_weight >= 0.0);
        assert!(telemetry.rolling_pressure > 0.0);
    }

    #[test]
    fn loop_signal_modulates_feedback() {
        let mut feedback = GeometryFeedback::new(GeometryFeedbackConfig::default_policy());
        let summary = ChronoSummary {
            frames: 4,
            duration: 1.2,
            latest_timestamp: 1.2,
            mean_drift: -0.2,
            mean_abs_drift: 1.4,
            drift_std: 0.3,
            mean_energy: 3.0,
            energy_std: 0.8,
            mean_decay: -0.45,
            min_energy: 2.4,
            max_energy: 3.6,
        };
        let harmonics = ChronoHarmonics {
            frames: 4,
            duration: 1.2,
            sample_rate: 3.3,
            nyquist: 1.65,
            drift_power: vec![0.0; 4],
            energy_power: vec![0.0; 4],
            dominant_drift: Some(ChronoPeak {
                frequency: 0.5,
                magnitude: 0.8,
                phase: 0.0,
            }),
            dominant_energy: Some(ChronoPeak {
                frequency: 0.8,
                magnitude: 0.6,
                phase: 0.25,
            }),
        };
        let signal = ChronoLoopSignal::new(summary, Some(harmonics));
        feedback.integrate_loop_signal(&signal);
        feedback.inject_collapse_bias(9.0);

        let resonance = resonance_from(&[0.6, -0.4, 0.2, -0.1, 0.3]);
        let _ = feedback.process_resonance(&resonance);
        let telemetry = feedback.telemetry();
        assert!(telemetry.loop_gain > 0.0);
        assert!(telemetry.collapse_bias > 0.0);
        assert!(telemetry.softening_beta > 0.6);
        assert!(telemetry.leech_density_weight > 0.35);
        assert!(telemetry.loop_timestamp >= 1.2 - f32::EPSILON);
        assert!(telemetry.loop_support >= 4.0);
    }

    #[test]
    fn loopback_envelopes_merge_signals() {
        let mut feedback = GeometryFeedback::new(GeometryFeedbackConfig::default_policy());
        let summary_a = ChronoSummary {
            frames: 3,
            duration: 0.9,
            latest_timestamp: 0.9,
            mean_drift: -0.3,
            mean_abs_drift: 0.7,
            drift_std: 0.2,
            mean_energy: 2.5,
            energy_std: 0.4,
            mean_decay: -0.15,
            min_energy: 2.0,
            max_energy: 3.1,
        };
        let summary_b = ChronoSummary {
            frames: 4,
            duration: 1.4,
            latest_timestamp: 1.4,
            mean_drift: 0.1,
            mean_abs_drift: 0.5,
            drift_std: 0.25,
            mean_energy: 1.8,
            energy_std: 0.3,
            mean_decay: -0.05,
            min_energy: 1.4,
            max_energy: 2.2,
        };
        let signal_a = ChronoLoopSignal::new(summary_a, None);
        let signal_b = ChronoLoopSignal::new(summary_b, None);
        let envelope_a = LoopbackEnvelope::new(signal_a)
            .with_support(1.2)
            .with_collapse_total(Some(1.6))
            .with_z_signal(Some(0.3))
            .with_script_hint(Some("spiralk:above".to_string()));
        let envelope_b = LoopbackEnvelope::new(signal_b)
            .with_support(2.0)
            .with_z_signal(Some(-0.2));

        feedback.absorb_loopback(&[envelope_a, envelope_b]);
        let resonance = resonance_from(&[0.4, -0.3, 0.2, -0.1, 0.5]);
        let _ = feedback.process_resonance(&resonance);
        let telemetry = feedback.telemetry().clone();
        assert!(telemetry.loop_gain > 0.0);
        assert!(telemetry.collapse_bias > 0.0);
        assert!(telemetry.loop_support > 1.0);
        assert!(telemetry.softening_beta > 0.6 - f32::EPSILON);
        assert!(telemetry.loop_script.is_some());
    }
}

impl GeometryFeedback {
    #[cfg(feature = "kdsl")]
    fn signal_script(signal: &ChronoLoopSignal) -> Option<String> {
        signal.spiralk_script.clone()
    }

    #[cfg(not(feature = "kdsl"))]
    fn signal_script(_signal: &ChronoLoopSignal) -> Option<String> {
        None
    }
}
