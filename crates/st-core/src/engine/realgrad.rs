// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::ops::realgrad::{
    RealGradConfig, RealGradKernel, RealGradProjection, RealGradTuning, SchwartzSequence,
    TemperedRealGradProjection,
};
use crate::telemetry::hub::{set_last_realgrad, RealGradPulse};

/// High level driver that reuses the cached RealGrad kernel, applies adaptive tuning,
/// and reports telemetry back into the global hub so runtimes can react to Z-space
/// energy changes.
#[derive(Debug)]
pub struct RealGradEngine {
    kernel: RealGradKernel,
    tolerance: f32,
    last_tuning: Option<RealGradTuning>,
}

impl RealGradEngine {
    /// Creates a new engine using the provided configuration.
    pub fn new(config: RealGradConfig) -> Self {
        let kernel = RealGradKernel::new(config);
        let tolerance = (kernel.config().residual_threshold * 0.5).max(0.0);
        Self {
            kernel,
            tolerance,
            last_tuning: None,
        }
    }

    /// Returns the current RealGrad configuration.
    pub fn config(&self) -> RealGradConfig {
        self.kernel.config()
    }

    /// Returns the tolerance used for tempered sequence convergence.
    pub fn tolerance(&self) -> f32 {
        self.tolerance
    }

    /// Updates the tolerance used for tempered projections.
    pub fn set_tolerance(&mut self, tolerance: f32) {
        self.tolerance = tolerance.max(0.0);
    }

    /// Returns the cached Ramanujan π estimate used by the engine.
    pub fn ramanujan_pi(&self) -> f32 {
        self.kernel.ramanujan_pi()
    }

    /// Projects a single Euclidean gradient and records the telemetry pulse.
    pub fn project(&mut self, values: &[f32]) -> RealGradProjection {
        let projection = self.kernel.project(values);
        self.emit_pulse(&projection, None);
        projection
    }

    /// Projects a Schwartz sequence, recording the tempered telemetry pulse.
    pub fn project_tempered(&mut self, sequence: &SchwartzSequence) -> TemperedRealGradProjection {
        let tempered = self.kernel.project_tempered(sequence, self.tolerance);
        self.emit_pulse(&tempered.projection, Some(&tempered));
        tempered
    }

    /// Calibrates the engine using the provided projection and returns the tuning diagnostics.
    pub fn calibrate(&mut self, projection: &RealGradProjection) -> RealGradTuning {
        let (config, tuning) = self.kernel.config().calibrate(projection);
        self.kernel = RealGradKernel::new(config);
        self.tolerance = (self.kernel.config().residual_threshold * 0.5).max(0.0);
        self.last_tuning = Some(tuning);
        tuning
    }

    /// Returns the diagnostics produced by the latest calibration, if any.
    pub fn last_tuning(&self) -> Option<RealGradTuning> {
        self.last_tuning
    }

    fn emit_pulse(
        &self,
        projection: &RealGradProjection,
        tempered: Option<&TemperedRealGradProjection>,
    ) {
        let monad_energy = projection.residual_energy();
        let z_energy = projection.z_energy();
        let total = (monad_energy + z_energy).max(f32::EPSILON);
        let residual_ratio = monad_energy / total;
        let lebesgue_ratio = if projection.lebesgue_measure > 0.0 {
            monad_energy / projection.lebesgue_measure
        } else {
            0.0
        };
        let (iterations, convergence_error, dominated, converged) = if let Some(tempered) = tempered
        {
            (
                tempered.iterations as u32,
                tempered.convergence_error,
                tempered.dominated,
                tempered.converged(self.tolerance),
            )
        } else {
            (1, 0.0, true, true)
        };

        let pulse = RealGradPulse {
            lebesgue_measure: projection.lebesgue_measure,
            monad_energy,
            z_energy,
            residual_ratio,
            lebesgue_ratio,
            ramanujan_pi: projection.ramanujan_pi,
            tolerance: self.tolerance,
            convergence_error,
            iterations,
            dominated,
            converged,
        };
        set_last_realgrad(&pulse);
    }
}

/// Convenience wrapper that projects and calibrates in a single step, returning the
/// resulting projection and the tuning diagnostics.
pub fn project_and_calibrate(
    engine: &mut RealGradEngine,
    values: &[f32],
) -> (RealGradProjection, RealGradTuning) {
    let projection = engine.project(values);
    let tuning = engine.calibrate(&projection);
    (projection, tuning)
}

/// Applies the tempered projection path and calibrates the engine from the resulting
/// projection, returning both the tempered output and the tuning diagnostics.
pub fn project_tempered_and_calibrate(
    engine: &mut RealGradEngine,
    sequence: &SchwartzSequence,
) -> (TemperedRealGradProjection, RealGradTuning) {
    let tempered = engine.project_tempered(sequence);
    let tuning = engine.calibrate(&tempered.projection);
    (tempered, tuning)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_projects_and_calibrates() {
        crate::telemetry::hub::clear_last_realgrad_for_test();
        let mut engine = RealGradEngine::new(RealGradConfig::default());
        let values = vec![0.5f32, -0.25, 0.75, -0.5];
        let projection = engine.project(&values);
        assert_eq!(projection.realgrad.len(), values.len());
        let tuning = engine.calibrate(&projection);
        assert!(tuning.adjustment_factor.is_finite());
        assert!(engine.last_tuning().is_some());
        assert!(engine.tolerance() >= 0.0);
        let pulse = crate::telemetry::hub::get_last_realgrad().expect("pulse recorded");
        assert!(pulse.lebesgue_measure > 0.0);
        assert!(pulse.iterations >= 1);
    }

    #[test]
    fn engine_handles_tempered_sequences() {
        crate::telemetry::hub::clear_last_realgrad_for_test();
        let mut members = Vec::new();
        for scale in [1.0f32, 2.0] {
            members.push(vec![(-scale).exp(), 0.5 * (-scale).exp()]);
        }
        let sequence = SchwartzSequence::new(members, vec![1.0, 1.0]);
        let mut engine = RealGradEngine::new(RealGradConfig::default());
        engine.set_tolerance(1.0e-3);
        let tempered = engine.project_tempered(&sequence);
        assert!(tempered.iterations >= 1);
        assert!(tempered.convergence_error >= 0.0);
        let pulse = crate::telemetry::hub::get_last_realgrad().expect("tempered pulse");
        assert_eq!(pulse.iterations as usize, tempered.iterations);
        assert_eq!(pulse.dominated, tempered.dominated);
    }
}
