// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::ops::realgrad::{
    RealGradConfig, RealGradKernel, RealGradProjection, RealGradTuning, SchwartzSequence,
    SpectrumNorm, TemperedRealGradProjection, TransparentGradientOpticsConfig,
};
use crate::telemetry::hub::{set_last_realgrad, RealGradPulse};
use st_tensor::{emit_tensor_op, emit_tensor_op_meta};

fn finite_meta_f32(value: f32) -> serde_json::Value {
    serde_json::Value::from(if value.is_finite() { value as f64 } else { 0.0 })
}

fn spectrum_norm_label(norm: SpectrumNorm) -> &'static str {
    match norm {
        SpectrumNorm::Unitary => "unitary",
        SpectrumNorm::Forward => "forward",
        SpectrumNorm::Backward => "backward",
        SpectrumNorm::EnergyPreserving => "energy_preserving",
        SpectrumNorm::LebesgueL1 => "lebesgue_l1",
        SpectrumNorm::Whitened => "whitened",
    }
}

fn pulse_regime_label(pulse: &RealGradPulse) -> &'static str {
    if !pulse.converged || !pulse.dominated {
        "tempered_unstable"
    } else if pulse.residual_ratio >= 0.5 {
        "residual_dominant"
    } else if pulse.gradient_sparsity >= 0.75 {
        "sparse_gradient"
    } else if pulse.transparency.is_some() {
        "transparent_optics"
    } else {
        "balanced"
    }
}

fn emit_realgrad_projection_meta(
    config: RealGradConfig,
    tolerance: f32,
    ema_alpha: f32,
    projection: &RealGradProjection,
    tempered: Option<&TemperedRealGradProjection>,
    pulse: &RealGradPulse,
) {
    let output_cols = 8usize + usize::from(pulse.transparency.is_some());
    emit_tensor_op(
        "realgrad_projection_pulse",
        &[projection.realgrad.len().max(1)],
        &[1, output_cols],
    );
    emit_tensor_op_meta("realgrad_projection_pulse", || {
        let mut payload = serde_json::Map::new();
        payload.insert("backend".into(), "cpu".into());
        payload.insert("requested_backend".into(), "auto".into());
        payload.insert("kind".into(), "st_core_realgrad_projection_pulse".into());
        payload.insert("regime".into(), pulse_regime_label(pulse).into());
        payload.insert("sample_len".into(), projection.realgrad.len().into());
        payload.insert("z_space_len".into(), projection.z_space.len().into());
        payload.insert("spectrum_len".into(), projection.spectrum.len().into());
        payload.insert("residual_count".into(), projection.monad_biome.len().into());
        payload.insert(
            "tempered".into(),
            tempered
                .map(|projection| projection.iterations > 0)
                .unwrap_or(false)
                .into(),
        );
        payload.insert(
            "tempered_iterations".into(),
            tempered
                .map(|projection| projection.iterations)
                .unwrap_or(0)
                .into(),
        );
        payload.insert(
            "tempered_dominated".into(),
            tempered
                .map(|projection| projection.dominated)
                .unwrap_or(true)
                .into(),
        );
        payload.insert("converged".into(), pulse.converged.into());
        payload.insert("dominated".into(), pulse.dominated.into());
        payload.insert("iterations".into(), pulse.iterations.into());
        payload.insert("tolerance".into(), finite_meta_f32(tolerance));
        payload.insert("ema_alpha".into(), finite_meta_f32(ema_alpha));
        payload.insert(
            "lebesgue_measure".into(),
            finite_meta_f32(pulse.lebesgue_measure),
        );
        payload.insert("monad_energy".into(), finite_meta_f32(pulse.monad_energy));
        payload.insert("z_energy".into(), finite_meta_f32(pulse.z_energy));
        payload.insert(
            "residual_ratio".into(),
            finite_meta_f32(pulse.residual_ratio),
        );
        payload.insert(
            "lebesgue_ratio".into(),
            finite_meta_f32(pulse.lebesgue_ratio),
        );
        payload.insert("ramanujan_pi".into(), finite_meta_f32(pulse.ramanujan_pi));
        payload.insert(
            "convergence_error".into(),
            finite_meta_f32(pulse.convergence_error),
        );
        payload.insert("gradient_norm".into(), finite_meta_f32(pulse.gradient_norm));
        payload.insert(
            "gradient_sparsity".into(),
            finite_meta_f32(pulse.gradient_sparsity),
        );
        payload.insert(
            "rolling_gradient_norm".into(),
            finite_meta_f32(pulse.rolling_gradient_norm),
        );
        payload.insert(
            "rolling_residual_ratio".into(),
            finite_meta_f32(pulse.rolling_residual_ratio),
        );
        payload.insert(
            "config_ramanujan_iterations".into(),
            config.ramanujan_iterations.into(),
        );
        payload.insert("config_z_rank".into(), config.z_rank.into());
        payload.insert("config_z_weight".into(), config.z_weight.into());
        payload.insert(
            "config_residual_threshold".into(),
            finite_meta_f32(config.residual_threshold),
        );
        payload.insert(
            "config_spectrum_norm".into(),
            spectrum_norm_label(config.spectrum_norm).into(),
        );
        payload.insert("optics_enabled".into(), pulse.transparency.is_some().into());
        if let Some(summary) = pulse.transparency {
            payload.insert(
                "optics_transparency_gain".into(),
                finite_meta_f32(summary.transparency_gain),
            );
            payload.insert(
                "optics_mean_attenuation".into(),
                finite_meta_f32(summary.mean_attenuation),
            );
            payload.insert(
                "optics_max_attenuation".into(),
                finite_meta_f32(summary.max_attenuation),
            );
            payload.insert(
                "optics_mean_refraction".into(),
                finite_meta_f32(summary.mean_refraction),
            );
            payload.insert(
                "optics_diffusion_energy".into(),
                finite_meta_f32(summary.diffusion_energy),
            );
            payload.insert(
                "optics_phase_variation".into(),
                finite_meta_f32(summary.phase_variation),
            );
            payload.insert(
                "optics_jacobian_norm".into(),
                finite_meta_f32(summary.jacobian_norm),
            );
        }
        serde_json::Value::Object(payload)
    });
}

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
        emit_realgrad_projection_meta(
            self.kernel.config(),
            self.tolerance,
            self.ema_alpha,
            projection,
            tempered,
            &pulse,
        );
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
    use std::sync::{Arc, Mutex, OnceLock};

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
    fn engine_projection_emits_backend_meta() {
        let _observer_lock = crate::telemetry::tensor_observer_lock();
        let _guard = telemetry_guard();
        crate::telemetry::hub::clear_last_realgrad_for_test();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let optics = TransparentGradientOpticsConfig {
            refractive_index: 1.4,
            transparency: 0.85,
            absorption: 0.02,
            diffusion: 0.2,
            phase_shift: 0.05,
        };
        let config = RealGradConfig {
            residual_threshold: 10.0,
            ..RealGradConfig::default()
        }
        .with_optics(optics);
        let mut engine = RealGradEngine::new(config);
        let projection = engine.project(&[0.75, -0.3, 0.45, -0.6, 0.15]);
        st_tensor::set_tensor_op_meta_observer(previous);

        assert!(projection.optics_trace().is_some());
        let events = events.lock().unwrap();
        let meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "realgrad_projection_pulse"
                    && data["kind"] == "st_core_realgrad_projection_pulse"
                    && data["sample_len"] == 5
            })
            .expect("realgrad projection metadata event");
        assert_eq!(meta.1["backend"], "cpu");
        assert_eq!(meta.1["regime"], "transparent_optics");
        assert_eq!(meta.1["optics_enabled"], true);
        assert_eq!(meta.1["residual_count"], 0);
        assert!(meta.1["gradient_norm"].as_f64().unwrap() > 0.0);
        assert!(meta.1["optics_jacobian_norm"].as_f64().unwrap() > 0.0);
        assert_eq!(meta.1["config_residual_threshold"], 10.0);
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
