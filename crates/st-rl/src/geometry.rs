// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::collections::VecDeque;

use st_core::theory::observability::{
    ObservabilityAssessment, ObservabilityConfig, ObservationalCoalgebra, SlotSymmetry,
};
use st_core::util::math::{ramanujan_pi, LeechProjector};
use st_tensor::pure::{DifferentialResonance, Tensor};

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
            max_learning_rate_scale: 1.5,
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
    /// Multiplicative factor suggested for the learning rate.
    pub learning_rate_scale: f32,
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
    leech_projector: LeechProjector,
    ramanujan_pi: f64,
    softening_beta: f32,
    last_signal: Option<GeometryFeedbackSignal>,
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
        Self {
            coalgebra: ObservationalCoalgebra::new(config.observability),
            threshold: config.activation_threshold.abs().max(f32::EPSILON),
            history: VecDeque::with_capacity(window),
            window,
            min_scale: min_scale.max(f32::EPSILON),
            max_scale: max_scale.max(f32::EPSILON),
            leech_projector: LeechProjector::new(
                config.z_space_rank.max(1),
                config.leech_density_weight,
            ),
            ramanujan_pi: ramanujan_pi(config.ramanujan_iterations.max(1)),
            softening_beta: config.softening_beta.max(0.0),
            last_signal: None,
        }
    }

    /// Returns the most recent signal emitted by the controller.
    pub fn last_signal(&self) -> Option<&GeometryFeedbackSignal> {
        self.last_signal.as_ref()
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
        let densified = self.leech_projector.enrich(geodesic);
        let normalized = ((averaged + densified) / self.ramanujan_pi).clamp(0.0, 1.0);
        let softened = self.soft_project(normalized as f32);
        let scale = self.min_scale + (self.max_scale - self.min_scale) * softened;
        let signal = GeometryFeedbackSignal {
            assessment,
            averaged_efficiency: averaged,
            learning_rate_scale: scale.max(self.min_scale),
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
}

#[cfg(test)]
mod tests {
    use super::*;
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

        let second = feedback.process_resonance(&resonance_low);
        assert!(second.learning_rate_scale <= first.learning_rate_scale);
        assert!(feedback.last_signal().is_some());
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
}
