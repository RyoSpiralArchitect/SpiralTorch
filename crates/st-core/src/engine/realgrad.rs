// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::ops::realgrad::{
    RealGradConfig, RealGradKernel, RealGradProjection, RealGradTuning, SchwartzSequence,
    TemperedRealGradProjection, TransparentGradientOpticsConfig,
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
    ema_alpha: f32,
    rolling_gradient_norm: Option<f32>,
    rolling_residual_ratio: Option<f32>,
    last_pulse: Option<RealGradPulse>,
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
            ema_alpha: 0.2,
            rolling_gradient_norm: None,
            rolling_residual_ratio: None,
            last_pulse: None,
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

    /// Returns the smoothing factor used for the rolling telemetry metrics.
    pub fn ema_alpha(&self) -> f32 {
        self.ema_alpha
    }

    /// Updates the smoothing factor used for the rolling telemetry metrics.
    pub fn set_ema_alpha(&mut self, alpha: f32) {
        self.ema_alpha = alpha.clamp(0.0, 1.0);
    }

    /// Returns the configured transparent gradient optics, if enabled.
    pub fn optics(&self) -> Option<TransparentGradientOpticsConfig> {
        self.kernel.optics()
    }

    /// Updates the transparent gradient optics configuration used by the engine.
    pub fn set_optics(&mut self, optics: Option<TransparentGradientOpticsConfig>) {
        self.kernel.set_optics(optics);
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

    /// Projects a Euclidean gradient and returns both the projection and the emitted telemetry.
    pub fn project_with_pulse(&mut self, values: &[f32]) -> (RealGradProjection, RealGradPulse) {
        let projection = self.project(values);
        let pulse = self
            .last_pulse
            .expect("telemetry pulse emitted after projection");
        (projection, pulse)
    }

    /// Projects a tempered sequence and returns both the projection and the emitted telemetry.
    pub fn project_tempered_with_pulse(
        &mut self,
        sequence: &SchwartzSequence,
    ) -> (TemperedRealGradProjection, RealGradPulse) {
        let tempered = self.project_tempered(sequence);
        let pulse = self
            .last_pulse
            .expect("telemetry pulse emitted after tempered projection");
        (tempered, pulse)
    }

    /// Returns the most recent telemetry pulse emitted by the engine, if any.
    pub fn last_pulse(&self) -> Option<RealGradPulse> {
        self.last_pulse
    }

    /// Returns the rolling gradient norm tracked by the engine.
    pub fn rolling_gradient_norm(&self) -> Option<f32> {
        self.rolling_gradient_norm
    }

    /// Returns the rolling residual ratio tracked by the engine.
    pub fn rolling_residual_ratio(&self) -> Option<f32> {
        self.rolling_residual_ratio
    }

    /// Clears the accumulated rolling statistics and stored telemetry pulse.
    pub fn clear_history(&mut self) {
        self.rolling_gradient_norm = None;
        self.rolling_residual_ratio = None;
        self.last_pulse = None;
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
        &mut self,
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
        let gradient_summary = projection.gradient_summary();
        let transparency = projection.transparency_summary();
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

        let mut pulse = RealGradPulse {
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
            gradient_norm: gradient_summary.norm,
            gradient_sparsity: gradient_summary.sparsity,
            rolling_gradient_norm: 0.0,
            rolling_residual_ratio: 0.0,
            transparency,
        };

        let rolling_gradient = match self.rolling_gradient_norm {
            Some(prev) if self.ema_alpha > 0.0 => {
                prev + self.ema_alpha * (pulse.gradient_norm - prev)
            }
            Some(_) => pulse.gradient_norm,
            None => pulse.gradient_norm,
        };
        self.rolling_gradient_norm = Some(rolling_gradient);

        let rolling_residual = match self.rolling_residual_ratio {
            Some(prev) if self.ema_alpha > 0.0 => {
                prev + self.ema_alpha * (pulse.residual_ratio - prev)
            }
            Some(_) => pulse.residual_ratio,
            None => pulse.residual_ratio,
        };
        self.rolling_residual_ratio = Some(rolling_residual);

        pulse.rolling_gradient_norm = rolling_gradient;
        pulse.rolling_residual_ratio = rolling_residual;

        self.last_pulse = Some(pulse);
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
    use std::sync::{Mutex, OnceLock};

    fn telemetry_guard() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .expect("lock telemetry guard")
    }

    fn assert_close(label: &str, lhs: f32, rhs: f32) {
        let diff = (lhs - rhs).abs();
        let scale = lhs.abs().max(rhs.abs()).max(1.0);
        assert!(
            diff <= 1.0e-5 * scale,
            "{label} differed beyond tolerance: left={lhs}, right={rhs}, diff={diff}"
        );
    }

    #[test]
    fn engine_projects_and_calibrates() {
        let _guard = telemetry_guard();
        crate::telemetry::hub::clear_last_realgrad_for_test();
        let mut engine = RealGradEngine::new(RealGradConfig::default());
        let values = vec![0.5f32, -0.25, 0.75, -0.5];
        let projection = engine.project(&values);
        assert_eq!(projection.realgrad.len(), values.len());
        let local_pulse = engine.last_pulse().expect("pulse stored in engine");
        let tuning = engine.calibrate(&projection);
        assert!(tuning.adjustment_factor.is_finite());
        assert!(engine.last_tuning().is_some());
        assert!(engine.tolerance() >= 0.0);
        let pulse = crate::telemetry::hub::get_last_realgrad().expect("pulse recorded");
        assert!(pulse.lebesgue_measure > 0.0);
        assert!(pulse.iterations >= 1);
        assert!(pulse.gradient_norm >= 0.0);
        assert!(pulse.gradient_sparsity >= 0.0 && pulse.gradient_sparsity <= 1.0);
        assert!(pulse.rolling_gradient_norm >= 0.0);
        assert!(pulse.rolling_residual_ratio >= 0.0);
        assert_close(
            "gradient_norm",
            pulse.gradient_norm,
            local_pulse.gradient_norm,
        );
        assert_close(
            "rolling_gradient_norm",
            pulse.rolling_gradient_norm,
            local_pulse.rolling_gradient_norm,
        );
        assert_close(
            "engine rolling_gradient_norm",
            engine.rolling_gradient_norm().unwrap(),
            pulse.rolling_gradient_norm,
        );
        assert_close(
            "engine rolling_residual_ratio",
            engine.rolling_residual_ratio().unwrap(),
            pulse.rolling_residual_ratio,
        );
    }

    #[test]
    fn engine_handles_tempered_sequences() {
        let _guard = telemetry_guard();
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
        assert!(pulse.gradient_norm >= 0.0);
        assert!(pulse.gradient_sparsity >= 0.0 && pulse.gradient_sparsity <= 1.0);
        assert!(pulse.rolling_gradient_norm >= 0.0);
        assert!(pulse.rolling_residual_ratio >= 0.0);
    }

    #[test]
    fn engine_project_with_pulse_roundtrips() {
        let _guard = telemetry_guard();
        crate::telemetry::hub::clear_last_realgrad_for_test();
        let mut engine = RealGradEngine::new(RealGradConfig::default());
        let values = vec![1.0f32, 0.5, -0.25, -0.75];
        let (projection, pulse) = engine.project_with_pulse(&values);
        assert_eq!(projection.realgrad.len(), values.len());
        let cached = engine.last_pulse().expect("pulse cached");
        assert_eq!(pulse, cached);
        assert_eq!(
            pulse.rolling_gradient_norm,
            engine.rolling_gradient_norm().unwrap()
        );
        assert_eq!(
            pulse.rolling_residual_ratio,
            engine.rolling_residual_ratio().unwrap()
        );
    }

    #[test]
    fn engine_allows_history_reset() {
        let _guard = telemetry_guard();
        crate::telemetry::hub::clear_last_realgrad_for_test();
        let mut engine = RealGradEngine::new(RealGradConfig::default());
        let values = vec![0.1f32, 0.2, 0.3, 0.4];
        engine.project(&values);
        assert!(engine.last_pulse().is_some());
        assert!(engine.rolling_gradient_norm().is_some());
        engine.clear_history();
        assert!(engine.last_pulse().is_none());
        assert!(engine.rolling_gradient_norm().is_none());
        assert!(engine.rolling_residual_ratio().is_none());
    }

    #[test]
    fn engine_allows_custom_ema_alpha() {
        let mut engine = RealGradEngine::new(RealGradConfig::default());
        assert!((engine.ema_alpha() - 0.2).abs() < f32::EPSILON);
        engine.set_ema_alpha(0.75);
        assert!((engine.ema_alpha() - 0.75).abs() < f32::EPSILON);
        engine.set_ema_alpha(-2.0);
        assert_eq!(engine.ema_alpha(), 0.0);
        engine.set_ema_alpha(1.5);
        assert_eq!(engine.ema_alpha(), 1.0);
    }

    #[test]
    fn engine_emits_transparency_summary_with_optics() {
        let _guard = telemetry_guard();
        crate::telemetry::hub::clear_last_realgrad_for_test();
        let optics = TransparentGradientOpticsConfig {
            refractive_index: 1.5,
            transparency: 0.8,
            absorption: 0.05,
            diffusion: 0.25,
            phase_shift: 0.1,
        };
        let mut engine = RealGradEngine::new(RealGradConfig::default().with_optics(optics));
        let values = vec![0.75f32, -0.3, 0.45, -0.6, 0.15];
        let projection = engine.project(&values);
        let trace_summary = projection
            .transparency_summary()
            .expect("projection transparency summary");
        assert!((trace_summary.transparency_gain - optics.transparency).abs() < 1.0e-6);
        let pulse = engine.last_pulse().expect("pulse emitted");
        let pulse_summary = pulse.transparency.expect("pulse transparency summary");
        assert_eq!(pulse_summary, trace_summary);
    }
}
