// SPDX-License-Identifier: AGPL-3.0-or-later

//! Canonical stateful Z-space temperature-control semantics.
//!
//! This module owns validation and the deterministic transition from a
//! probability distribution plus optional Z-space and gradient feedback into
//! the next controller state. Bindings may transport state, but must not
//! reconstruct these formulas.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const ZSPACE_TEMPERATURE_CONTROL_CONTRACT_VERSION: &str =
    "spiraltorch.zspace_temperature_control.v1";
pub const ZSPACE_TEMPERATURE_CONTROL_KIND: &str = "spiraltorch.zspace_temperature_control";
pub const ZSPACE_TEMPERATURE_CONTROL_SEMANTIC_OWNER: &str =
    "st-core::inference::temperature_control";
pub const ZSPACE_TEMPERATURE_CONTROL_SEMANTIC_BACKEND: &str = "rust";
pub const ZSPACE_TEMPERATURE_CONTROL_BACKEND: &str = "spiraltorch_temperature_control_core";

const BASE_PROBABILITY_SUM_TOLERANCE: f64 = 1.0e-6;
const F32_SUM_TOLERANCE_MULTIPLIER: f64 = 4.0;
const MIN_FEEDBACK_SNR: f64 = 1.0e-3;
const MAX_NORMALIZED_FEEDBACK: f64 = 4.0;
const MIN_SCALE_FACTOR: f64 = 0.5;
const MAX_SCALE_FACTOR: f64 = 2.0;

#[derive(Debug, Error, PartialEq)]
pub enum TemperatureControlError {
    #[error("temperature control distribution must not be empty")]
    EmptyDistribution,
    #[error("temperature control field '{field}' must be finite, got {value}")]
    NonFinite { field: &'static str, value: f64 },
    #[error("temperature control field '{field}' must be positive, got {value}")]
    NonPositive { field: &'static str, value: f64 },
    #[error("temperature control field '{field}' must be non-negative, got {value}")]
    Negative { field: &'static str, value: f64 },
    #[error("temperature control field '{field}' must be in [{min}, {max}], got {value}")]
    OutOfRange {
        field: &'static str,
        value: f64,
        min: f64,
        max: f64,
    },
    #[error(
        "temperature control bounds require min_temperature <= max_temperature, got {min_temperature} > {max_temperature}"
    )]
    InvalidTemperatureBounds {
        min_temperature: f64,
        max_temperature: f64,
    },
    #[error("temperature control probability mass must be within {tolerance} of 1, got {sum}")]
    InvalidProbabilityMass { sum: f64, tolerance: f64 },
    #[error("derived temperature control field '{field}' must be finite, got {value}")]
    NonFiniteDerived { field: &'static str, value: f64 },
}

fn require_finite(field: &'static str, value: f64) -> Result<f64, TemperatureControlError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(TemperatureControlError::NonFinite { field, value })
    }
}

fn require_positive(field: &'static str, value: f64) -> Result<f64, TemperatureControlError> {
    require_finite(field, value)?;
    if value > 0.0 {
        Ok(value)
    } else {
        Err(TemperatureControlError::NonPositive { field, value })
    }
}

fn require_non_negative(field: &'static str, value: f64) -> Result<f64, TemperatureControlError> {
    require_finite(field, value)?;
    if value >= 0.0 {
        Ok(value)
    } else {
        Err(TemperatureControlError::Negative { field, value })
    }
}

fn require_range(
    field: &'static str,
    value: f64,
    min: f64,
    max: f64,
) -> Result<f64, TemperatureControlError> {
    require_finite(field, value)?;
    if (min..=max).contains(&value) {
        Ok(value)
    } else {
        Err(TemperatureControlError::OutOfRange {
            field,
            value,
            min,
            max,
        })
    }
}

fn require_derived_finite(field: &'static str, value: f64) -> Result<f64, TemperatureControlError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(TemperatureControlError::NonFiniteDerived { field, value })
    }
}

fn checked_add(field: &'static str, left: f64, right: f64) -> Result<f64, TemperatureControlError> {
    require_derived_finite(field, left + right)
}

fn checked_mul(field: &'static str, left: f64, right: f64) -> Result<f64, TemperatureControlError> {
    require_derived_finite(field, left * right)
}

/// Probability-mass tolerance for f32-originated distributions transported as f64.
pub fn temperature_probability_sum_tolerance(value_count: usize) -> f64 {
    BASE_PROBABILITY_SUM_TOLERANCE
        + (value_count as f64).sqrt() * f64::from(f32::EPSILON) * F32_SUM_TOLERANCE_MULTIPLIER
}

fn distribution_entropy_and_mass(
    probabilities: &[f64],
) -> Result<(f64, f64, f64), TemperatureControlError> {
    if probabilities.is_empty() {
        return Err(TemperatureControlError::EmptyDistribution);
    }

    let mut sum = 0.0;
    for &probability in probabilities {
        require_range("probability", probability, 0.0, 1.0)?;
        sum = checked_add("probability_sum", sum, probability)?;
    }
    let tolerance = temperature_probability_sum_tolerance(probabilities.len());
    if (sum - 1.0).abs() > tolerance {
        return Err(TemperatureControlError::InvalidProbabilityMass { sum, tolerance });
    }
    require_positive("probability_sum", sum)?;

    let mut entropy = 0.0;
    for &probability in probabilities {
        let normalized = require_derived_finite("normalized_probability", probability / sum)?;
        if normalized > 0.0 {
            let term = require_derived_finite("entropy_term", -normalized * normalized.ln())?;
            entropy = checked_add("entropy", entropy, term)?;
        }
    }
    Ok((entropy, sum, tolerance))
}

/// Computes entropy using the same validation and normalization as the state transition.
pub fn temperature_distribution_entropy(
    probabilities: &[f64],
) -> Result<f64, TemperatureControlError> {
    distribution_entropy_and_mass(probabilities).map(|(entropy, _, _)| entropy)
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct TemperatureControlConfig {
    target_entropy: f64,
    eta: f64,
    min_temperature: f64,
    max_temperature: f64,
    z_kappa: f64,
    z_relax: f64,
    scale_gain: f64,
    gradient_decay: f64,
}

impl Default for TemperatureControlConfig {
    fn default() -> Self {
        Self {
            target_entropy: 0.8,
            eta: 0.4,
            min_temperature: 0.4,
            max_temperature: 2.0,
            z_kappa: 0.35,
            z_relax: 0.2,
            scale_gain: 0.25,
            gradient_decay: 0.5,
        }
    }
}

impl TemperatureControlConfig {
    pub fn new(
        target_entropy: f64,
        eta: f64,
        min_temperature: f64,
        max_temperature: f64,
    ) -> Result<Self, TemperatureControlError> {
        let config = Self {
            target_entropy,
            eta,
            min_temperature,
            max_temperature,
            ..Self::default()
        };
        config.validate()?;
        Ok(config)
    }

    pub fn with_feedback(
        mut self,
        z_kappa: f64,
        z_relax: f64,
    ) -> Result<Self, TemperatureControlError> {
        self.z_kappa = z_kappa;
        self.z_relax = z_relax;
        self.validate()?;
        Ok(self)
    }

    pub fn with_scale_gain(mut self, scale_gain: f64) -> Result<Self, TemperatureControlError> {
        self.scale_gain = scale_gain;
        self.validate()?;
        Ok(self)
    }

    pub fn with_gradient_decay(
        mut self,
        gradient_decay: f64,
    ) -> Result<Self, TemperatureControlError> {
        self.gradient_decay = gradient_decay;
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), TemperatureControlError> {
        require_non_negative("target_entropy", self.target_entropy)?;
        require_non_negative("eta", self.eta)?;
        require_positive("min_temperature", self.min_temperature)?;
        require_positive("max_temperature", self.max_temperature)?;
        if self.min_temperature > self.max_temperature {
            return Err(TemperatureControlError::InvalidTemperatureBounds {
                min_temperature: self.min_temperature,
                max_temperature: self.max_temperature,
            });
        }
        require_non_negative("z_kappa", self.z_kappa)?;
        require_range("z_relax", self.z_relax, 0.0, 1.0)?;
        require_range("scale_gain", self.scale_gain, 0.0, 1.0)?;
        require_range("gradient_decay", self.gradient_decay, 0.0, 1.0)?;
        Ok(())
    }

    pub fn target_entropy(&self) -> f64 {
        self.target_entropy
    }

    pub fn eta(&self) -> f64 {
        self.eta
    }

    pub fn min_temperature(&self) -> f64 {
        self.min_temperature
    }

    pub fn max_temperature(&self) -> f64 {
        self.max_temperature
    }

    pub fn z_kappa(&self) -> f64 {
        self.z_kappa
    }

    pub fn z_relax(&self) -> f64 {
        self.z_relax
    }

    pub fn scale_gain(&self) -> f64 {
        self.scale_gain
    }

    pub fn gradient_decay(&self) -> f64 {
        self.gradient_decay
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct TemperatureControlState {
    temperature: f64,
    z_memory: f64,
    scale_memory: f64,
    gradient_pressure: f64,
    gradient_entropy_bias: f64,
}

impl Default for TemperatureControlState {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            z_memory: 0.0,
            scale_memory: 0.0,
            gradient_pressure: 0.0,
            gradient_entropy_bias: 0.0,
        }
    }
}

impl TemperatureControlState {
    pub fn new(temperature: f64) -> Result<Self, TemperatureControlError> {
        require_positive("temperature", temperature)?;
        Ok(Self {
            temperature,
            ..Self::default()
        })
    }

    pub fn with_gradient_observation(
        mut self,
        pressure: f64,
        entropy_bias: f64,
    ) -> Result<Self, TemperatureControlError> {
        self.gradient_pressure = require_non_negative("gradient_pressure", pressure)?;
        self.gradient_entropy_bias =
            require_range("gradient_entropy_bias", entropy_bias, 0.0, 1.0)?;
        Ok(self)
    }

    pub fn validate_for(
        &self,
        config: &TemperatureControlConfig,
    ) -> Result<(), TemperatureControlError> {
        require_positive("temperature", self.temperature)?;
        require_range(
            "temperature",
            self.temperature,
            config.min_temperature,
            config.max_temperature,
        )?;
        require_range("z_memory", self.z_memory, -4.0, 4.0)?;
        require_range("scale_memory", self.scale_memory, -1.0, 1.0)?;
        require_non_negative("gradient_pressure", self.gradient_pressure)?;
        require_range(
            "gradient_entropy_bias",
            self.gradient_entropy_bias,
            0.0,
            1.0,
        )?;
        Ok(())
    }

    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    pub fn z_memory(&self) -> f64 {
        self.z_memory
    }

    pub fn scale_memory(&self) -> f64 {
        self.scale_memory
    }

    pub fn gradient_pressure(&self) -> f64 {
        self.gradient_pressure
    }

    pub fn gradient_entropy_bias(&self) -> f64 {
        self.gradient_entropy_bias
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct TemperatureControlFeedback {
    pub psi_total: f64,
    pub band_energy: [f64; 3],
    pub drift: f64,
    pub z_signal: f64,
    pub scale_log_radius: Option<f64>,
}

impl TemperatureControlFeedback {
    fn validate(&self) -> Result<(), TemperatureControlError> {
        require_finite("psi_total", self.psi_total)?;
        for energy in self.band_energy {
            require_non_negative("band_energy", energy)?;
        }
        require_finite("drift", self.drift)?;
        require_finite("z_signal", self.z_signal)?;
        if let Some(log_radius) = self.scale_log_radius {
            require_finite("scale_log_radius", log_radius)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct TemperatureControlRequest {
    pub probabilities: Vec<f64>,
    pub config: TemperatureControlConfig,
    pub state: TemperatureControlState,
    pub feedback: Option<TemperatureControlFeedback>,
    pub gradient_heat: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct TemperatureControlEffects {
    pub entropy_adjustment: f64,
    pub z_feedback_applied: bool,
    pub drift_normalized: f64,
    pub z_magnitude: f64,
    pub explore: f64,
    pub settle: f64,
    pub z_bias: f64,
    pub roam_factor: f64,
    pub anchor_factor: f64,
    pub z_factor: f64,
    pub scale_feedback_applied: bool,
    pub scale_bias: f64,
    pub scale_factor: f64,
    pub gradient_heat_provided: bool,
    pub gradient_feedback_applied: bool,
    pub gradient_heat: f64,
    pub gradient_drive: f64,
    pub gradient_adjustment: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TemperatureControlPayload {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub backend: &'static str,
    pub config: TemperatureControlConfig,
    pub previous_state: TemperatureControlState,
    pub next_state: TemperatureControlState,
    pub probability_count: usize,
    pub input_probability_sum: f64,
    pub probability_sum_tolerance: f64,
    pub entropy: f64,
    /// Signed control error: target entropy minus observed entropy.
    pub entropy_error: f64,
    pub temperature_after_entropy: f64,
    pub temperature_after_z: f64,
    pub temperature_after_scale: f64,
    pub temperature: f64,
    pub effects: TemperatureControlEffects,
}

/// Applies one complete temperature-control transition without mutating caller state.
pub fn apply_temperature_control(
    request: TemperatureControlRequest,
) -> Result<TemperatureControlPayload, TemperatureControlError> {
    let TemperatureControlRequest {
        probabilities,
        config,
        state,
        feedback,
        gradient_heat,
    } = request;
    config.validate()?;
    state.validate_for(&config)?;
    if let Some(feedback) = feedback {
        feedback.validate()?;
    }
    if let Some(heat) = gradient_heat {
        require_non_negative("gradient_heat", heat)?;
    }

    let (entropy, probability_sum, probability_sum_tolerance) =
        distribution_entropy_and_mass(&probabilities)?;
    let entropy_error = require_derived_finite("entropy_error", config.target_entropy - entropy)?;
    let entropy_adjustment = checked_mul("entropy_adjustment", config.eta, entropy_error)?;
    let temperature_after_entropy = checked_add(
        "temperature_after_entropy",
        state.temperature,
        entropy_adjustment,
    )?
    .clamp(config.min_temperature, config.max_temperature);

    let mut next_z_memory = state.z_memory;
    let mut next_scale_memory = state.scale_memory;
    let mut drift_normalized = 0.0;
    let mut z_magnitude = 0.0;
    let mut explore = 0.0;
    let mut settle = 0.0;
    let mut z_bias = 1.0;
    let mut roam_factor = 1.0;
    let mut anchor_factor = 1.0;
    let mut z_factor = 1.0;
    let mut scale_bias = 0.0;
    let mut scale_factor = 1.0;
    let mut temperature_after_z = temperature_after_entropy;
    let mut temperature_after_scale = temperature_after_entropy;
    let mut scale_feedback_applied = false;

    if let Some(feedback) = feedback {
        let first_bands = checked_add(
            "band_energy_sum",
            feedback.band_energy[0],
            feedback.band_energy[1],
        )?;
        let total_band = checked_add("band_energy_sum", first_bands, feedback.band_energy[2])?;
        let signal_to_noise = checked_add(
            "feedback_signal_to_noise",
            total_band,
            feedback.psi_total.abs(),
        )?
        .max(MIN_FEEDBACK_SNR);
        drift_normalized = require_derived_finite(
            "drift_normalized",
            (feedback.drift / signal_to_noise)
                .abs()
                .min(MAX_NORMALIZED_FEEDBACK),
        )?;
        z_magnitude = feedback.z_signal.abs().min(MAX_NORMALIZED_FEEDBACK);
        explore = (drift_normalized - z_magnitude).max(0.0);
        settle = (z_magnitude - drift_normalized).max(0.0);

        let retained_z_memory =
            checked_mul("retained_z_memory", 1.0 - config.z_relax, state.z_memory)?;
        let observed_z_memory = checked_mul("observed_z_memory", config.z_relax, explore - settle)?;
        next_z_memory = checked_add("next_z_memory", retained_z_memory, observed_z_memory)?;
        require_range("next_z_memory", next_z_memory, -4.0, 4.0)?;

        let z_bias_exponent = checked_mul("z_bias_exponent", -config.z_kappa, z_magnitude)?;
        z_bias = require_derived_finite("z_bias", z_bias_exponent.exp())?;
        roam_factor = checked_add(
            "roam_factor",
            1.0,
            checked_mul(
                "roam_drive",
                config.z_kappa,
                explore + next_z_memory.max(0.0),
            )?,
        )?;
        let anchor_denominator = checked_add(
            "anchor_denominator",
            1.0,
            checked_mul(
                "anchor_drive",
                config.z_kappa,
                settle + (-next_z_memory).max(0.0),
            )?,
        )?;
        anchor_factor = require_derived_finite("anchor_factor", 1.0 / anchor_denominator)?;
        z_factor = checked_mul(
            "z_factor",
            checked_mul("z_bias_roam_factor", z_bias, roam_factor)?,
            anchor_factor,
        )?;
        temperature_after_z =
            checked_mul("temperature_after_z", temperature_after_entropy, z_factor)?
                .clamp(config.min_temperature, config.max_temperature);
        temperature_after_scale = temperature_after_z;

        if config.scale_gain > 0.0 {
            if let Some(log_radius) = feedback.scale_log_radius {
                scale_feedback_applied = true;
                scale_bias = require_derived_finite("scale_bias", (-log_radius).tanh())?;
                let retained_scale_memory = checked_mul(
                    "retained_scale_memory",
                    1.0 - config.z_relax,
                    state.scale_memory,
                )?;
                let observed_scale_memory =
                    checked_mul("observed_scale_memory", config.z_relax, scale_bias)?;
                next_scale_memory = checked_add(
                    "next_scale_memory",
                    retained_scale_memory,
                    observed_scale_memory,
                )?;
                require_range("next_scale_memory", next_scale_memory, -1.0, 1.0)?;
                scale_factor = checked_add(
                    "scale_factor",
                    1.0,
                    checked_mul("scale_memory_gain", config.scale_gain, next_scale_memory)?,
                )?
                .clamp(MIN_SCALE_FACTOR, MAX_SCALE_FACTOR);
                temperature_after_scale =
                    checked_mul("temperature_after_scale", temperature_after_z, scale_factor)?
                        .clamp(config.min_temperature, config.max_temperature);
            }
        }
    }

    let mut temperature = temperature_after_scale;
    let mut next_gradient_pressure = state.gradient_pressure;
    let mut next_gradient_entropy_bias = state.gradient_entropy_bias;
    let mut gradient_drive = 0.0;
    let mut gradient_adjustment = 0.0;
    let gradient_feedback_applied = gradient_heat.is_some() && state.gradient_pressure > 0.0;
    if let Some(heat) = gradient_heat {
        if state.gradient_pressure > 0.0 {
            gradient_drive = checked_mul(
                "gradient_drive",
                checked_mul(
                    "gradient_pressure_entropy_bias",
                    state.gradient_pressure,
                    state.gradient_entropy_bias,
                )?,
                heat,
            )?;
            gradient_adjustment = checked_mul("gradient_adjustment", config.eta, gradient_drive)?;
            temperature = checked_add(
                "temperature_after_gradient",
                temperature_after_scale,
                gradient_adjustment,
            )?
            .clamp(config.min_temperature, config.max_temperature);
            next_gradient_pressure = checked_mul(
                "next_gradient_pressure",
                state.gradient_pressure,
                config.gradient_decay,
            )?;
            next_gradient_entropy_bias = checked_mul(
                "next_gradient_entropy_bias",
                state.gradient_entropy_bias,
                config.gradient_decay,
            )?;
        }
    }

    let next_state = TemperatureControlState {
        temperature,
        z_memory: next_z_memory,
        scale_memory: next_scale_memory,
        gradient_pressure: next_gradient_pressure,
        gradient_entropy_bias: next_gradient_entropy_bias,
    };
    next_state.validate_for(&config)?;

    Ok(TemperatureControlPayload {
        kind: ZSPACE_TEMPERATURE_CONTROL_KIND,
        contract_version: ZSPACE_TEMPERATURE_CONTROL_CONTRACT_VERSION,
        semantic_owner: ZSPACE_TEMPERATURE_CONTROL_SEMANTIC_OWNER,
        semantic_backend: ZSPACE_TEMPERATURE_CONTROL_SEMANTIC_BACKEND,
        backend: ZSPACE_TEMPERATURE_CONTROL_BACKEND,
        config,
        previous_state: state,
        next_state,
        probability_count: probabilities.len(),
        input_probability_sum: probability_sum,
        probability_sum_tolerance,
        entropy,
        entropy_error,
        temperature_after_entropy,
        temperature_after_z,
        temperature_after_scale,
        temperature,
        effects: TemperatureControlEffects {
            entropy_adjustment,
            z_feedback_applied: feedback.is_some(),
            drift_normalized,
            z_magnitude,
            explore,
            settle,
            z_bias,
            roam_factor,
            anchor_factor,
            z_factor,
            scale_feedback_applied,
            scale_bias,
            scale_factor,
            gradient_heat_provided: gradient_heat.is_some(),
            gradient_feedback_applied,
            gradient_heat: gradient_heat.unwrap_or(0.0),
            gradient_drive,
            gradient_adjustment,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> TemperatureControlRequest {
        TemperatureControlRequest {
            probabilities: vec![0.6, 0.4],
            config: TemperatureControlConfig::new(0.8, 0.2, 0.3, 2.0)
                .expect("valid config")
                .with_feedback(0.4, 0.2)
                .expect("valid feedback config")
                .with_scale_gain(0.6)
                .expect("valid scale config"),
            state: TemperatureControlState::new(1.0).expect("valid state"),
            feedback: None,
            gradient_heat: None,
        }
    }

    #[test]
    fn transition_is_deterministic_and_rust_owned() {
        let mut input = request();
        input.feedback = Some(TemperatureControlFeedback {
            psi_total: 0.1,
            band_energy: [0.2, 0.1, 0.05],
            drift: 0.4,
            z_signal: 0.1,
            scale_log_radius: Some(0.25f64.ln()),
        });

        let first = apply_temperature_control(input.clone()).expect("valid transition");
        let second = apply_temperature_control(input).expect("valid transition");

        assert_eq!(first, second);
        assert_eq!(first.kind, ZSPACE_TEMPERATURE_CONTROL_KIND);
        assert_eq!(
            first.semantic_owner,
            "st-core::inference::temperature_control"
        );
        assert!(first.effects.z_feedback_applied);
        assert!(first.effects.scale_feedback_applied);
        assert!(first.temperature > first.temperature_after_z);
    }

    #[test]
    fn entropy_feedback_is_negative_and_converges_toward_the_target() {
        fn binary_distribution(temperature: f64) -> Vec<f64> {
            let high = (2.0 / temperature).exp();
            let total = high + 1.0;
            vec![high / total, 1.0 / total]
        }

        let config =
            TemperatureControlConfig::new(0.5, 0.35, 0.1, 4.0).expect("valid convergence config");
        let mut state = TemperatureControlState::new(0.3).expect("valid initial state");
        let initial_probabilities = binary_distribution(state.temperature());
        let initial_entropy = temperature_distribution_entropy(&initial_probabilities)
            .expect("valid initial distribution");
        let initial_error = (config.target_entropy() - initial_entropy).abs();

        for _ in 0..24 {
            let probabilities = binary_distribution(state.temperature());
            let output = apply_temperature_control(TemperatureControlRequest {
                probabilities,
                config,
                state,
                feedback: None,
                gradient_heat: None,
            })
            .expect("valid convergence transition");
            state = output.next_state;
        }

        let final_probabilities = binary_distribution(state.temperature());
        let final_entropy = temperature_distribution_entropy(&final_probabilities)
            .expect("valid final distribution");
        let final_error = (config.target_entropy() - final_entropy).abs();

        assert!(state.temperature() > 0.3);
        assert!(final_error < initial_error * 0.1);
    }

    #[test]
    fn entropy_feedback_cools_an_overdispersed_distribution() {
        let mut input = request();
        input.config =
            TemperatureControlConfig::new(0.2, 0.4, 0.3, 2.0).expect("valid cooling config");
        input.probabilities = vec![0.5, 0.5];

        let output = apply_temperature_control(input).expect("valid cooling transition");

        assert!(output.entropy_error < 0.0);
        assert!(output.temperature_after_entropy < output.previous_state.temperature());
    }

    #[test]
    fn gradient_feedback_is_consumed_with_configured_decay() {
        let mut input = request();
        input.state = input
            .state
            .with_gradient_observation(32.0, 0.15)
            .expect("valid gradient observation");
        input.gradient_heat = Some(1.5);

        let output = apply_temperature_control(input).expect("valid transition");

        assert!(output.effects.gradient_feedback_applied);
        assert!(output.effects.gradient_adjustment > 0.0);
        assert_eq!(output.next_state.gradient_pressure(), 16.0);
        assert_eq!(output.next_state.gradient_entropy_bias(), 0.075);
    }

    #[test]
    fn invalid_probability_mass_fails_closed() {
        let mut input = request();
        input.probabilities = vec![0.8, 0.8];

        let error = apply_temperature_control(input).expect_err("mass must be rejected");

        assert!(matches!(
            error,
            TemperatureControlError::InvalidProbabilityMass { .. }
        ));
    }

    #[test]
    fn invalid_feedback_fails_before_a_state_can_be_committed() {
        let mut input = request();
        input.feedback = Some(TemperatureControlFeedback {
            drift: f64::NAN,
            ..TemperatureControlFeedback::default()
        });
        let previous = input.state;

        let error = apply_temperature_control(input).expect_err("NaN must be rejected");

        assert!(matches!(
            error,
            TemperatureControlError::NonFinite {
                field: "drift",
                value,
            } if value.is_nan()
        ));
        assert_eq!(previous.temperature(), 1.0);
    }

    #[test]
    fn derived_feedback_overflow_is_not_hidden_by_exponential_saturation() {
        let mut input = request();
        input.config = input
            .config
            .with_feedback(f64::MAX, 0.2)
            .expect("finite feedback config");
        input.feedback = Some(TemperatureControlFeedback {
            z_signal: 4.0,
            ..TemperatureControlFeedback::default()
        });

        let error = apply_temperature_control(input).expect_err("overflow must fail closed");

        assert!(matches!(
            error,
            TemperatureControlError::NonFiniteDerived {
                field: "z_bias_exponent",
                ..
            }
        ));
    }

    #[test]
    fn f32_originated_probability_mass_is_normalized_explicitly() {
        let probabilities = [f64::from(0.6f32), f64::from(0.4f32)];

        let entropy = temperature_distribution_entropy(&probabilities)
            .expect("f32 probability mass is within the documented tolerance");

        assert!((entropy - 0.673_011_667).abs() < 1.0e-8);
    }

    #[test]
    fn config_rejects_inverted_temperature_bounds() {
        let error = TemperatureControlConfig::new(0.8, 0.2, 2.0, 0.3)
            .expect_err("inverted bounds must fail");

        assert_eq!(
            error,
            TemperatureControlError::InvalidTemperatureBounds {
                min_temperature: 2.0,
                max_temperature: 0.3,
            }
        );
    }

    #[test]
    fn request_deserialization_rejects_unknown_fields() {
        let error = serde_json::from_str::<TemperatureControlRequest>(
            r#"{"probabilities":[1.0],"temperatur":1.0}"#,
        )
        .expect_err("contract drift must fail closed");

        assert!(error.to_string().contains("unknown field"));
    }
}
