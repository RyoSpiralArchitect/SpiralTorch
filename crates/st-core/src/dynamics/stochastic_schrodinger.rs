// SPDX-License-Identifier: AGPL-3.0-or-later

//! Real-time stochastic Schrodinger evolution for real-valued tensor clients.
//!
//! Each tensor row is embedded as the real quadrature of a complex wavefunction.
//! One step applies a Strang split between a diagonal learned Hamiltonian and
//! disjoint Hermitian pair hopping. Gaussian samples drive diagonal Stratonovich
//! phase diffusion, while uniform no-jump attenuation models open-system loss.
//! The squared phase-noise scale is the off-diagonal ensemble dephasing rate;
//! loss and dephasing therefore remain distinct. The returned real quadrature
//! remains tensor-compatible, while the imaginary quadrature and audit keep the
//! underlying complex evolution explicit.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const STOCHASTIC_SCHRODINGER_CONTRACT_VERSION: &str =
    "spiraltorch.stochastic_real_time_schrodinger.v1";
pub const STOCHASTIC_SCHRODINGER_KIND: &str = "spiraltorch.stochastic_real_time_schrodinger";
pub const STOCHASTIC_SCHRODINGER_SEMANTIC_OWNER: &str = "st-core::dynamics::stochastic_schrodinger";
pub const STOCHASTIC_SCHRODINGER_SEMANTIC_BACKEND: &str = "rust";
pub const STOCHASTIC_SCHRODINGER_INTEGRATOR: &str = "strang_diagonal_pair_unitary";
pub const STOCHASTIC_SCHRODINGER_CALCULUS: &str = "stratonovich";
pub const STOCHASTIC_SCHRODINGER_OUTPUT_OBSERVABLE: &str = "real_quadrature";
pub const STOCHASTIC_SCHRODINGER_HAMILTONIAN: &str =
    "learned_diagonal_plus_disjoint_adjacent_hermitian_pair_hopping";
pub const STOCHASTIC_SCHRODINGER_NOISE_MODEL: &str = "independent_gaussian_phase_diffusion";
pub const STOCHASTIC_SCHRODINGER_OPEN_SYSTEM_MODE: &str = "uniform_no_jump_loss";

const NORM_ERROR_FACTOR: f64 = 128.0 * f32::EPSILON as f64;
const FORMULA_ERROR_FACTOR: f64 = 256.0 * f32::EPSILON as f64;

#[derive(Clone, Debug, Error, PartialEq)]
pub enum StochasticSchrodingerError {
    #[error("stochastic Schrodinger feature count must be positive")]
    EmptyFeatures,
    #[error("stochastic Schrodinger tensor shape ({rows} x {cols}) exceeds usize range")]
    ShapeOverflow { rows: usize, cols: usize },
    #[error("stochastic Schrodinger field '{field}' has length {actual}, expected {expected}")]
    LengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("stochastic Schrodinger field '{field}' must be finite, got {value}")]
    NonFinite { field: &'static str, value: f32 },
    #[error("stochastic Schrodinger field '{field}' must be non-negative, got {value}")]
    Negative { field: &'static str, value: f32 },
    #[error("derived stochastic Schrodinger field '{field}' must be finite, got {value}")]
    NonFiniteDerived { field: &'static str, value: f32 },
    #[error("stochastic Schrodinger row {row} norm error {error} exceeds tolerance {tolerance}")]
    NormInvariant {
        row: usize,
        error: f64,
        tolerance: f64,
    },
    #[error("stochastic Schrodinger phase {index} error {error} exceeds tolerance {tolerance}")]
    PhaseInvariant {
        index: usize,
        error: f64,
        tolerance: f64,
    },
    #[error(
        "stochastic Schrodinger {quadrature} quadrature {index} error {error} exceeds tolerance {tolerance}"
    )]
    EvolutionInvariant {
        quadrature: &'static str,
        index: usize,
        error: f64,
        tolerance: f64,
    },
    #[error(
        "stochastic Schrodinger backward field '{field}' index {index} error {error} exceeds tolerance {tolerance}"
    )]
    BackwardInvariant {
        field: &'static str,
        index: usize,
        error: f64,
        tolerance: f64,
    },
}

fn require_finite(field: &'static str, value: f32) -> Result<f32, StochasticSchrodingerError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(StochasticSchrodingerError::NonFinite { field, value })
    }
}

fn require_non_negative(
    field: &'static str,
    value: f32,
) -> Result<f32, StochasticSchrodingerError> {
    require_finite(field, value)?;
    if value >= 0.0 {
        Ok(value)
    } else {
        Err(StochasticSchrodingerError::Negative { field, value })
    }
}

fn require_derived_finite(
    field: &'static str,
    value: f32,
) -> Result<f32, StochasticSchrodingerError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(StochasticSchrodingerError::NonFiniteDerived { field, value })
    }
}

fn checked_volume(rows: usize, cols: usize) -> Result<usize, StochasticSchrodingerError> {
    if cols == 0 {
        return Err(StochasticSchrodingerError::EmptyFeatures);
    }
    rows.checked_mul(cols)
        .ok_or(StochasticSchrodingerError::ShapeOverflow { rows, cols })
}

fn validate_length(
    field: &'static str,
    values: &[f32],
    expected: usize,
) -> Result<(), StochasticSchrodingerError> {
    if values.len() != expected {
        return Err(StochasticSchrodingerError::LengthMismatch {
            field,
            expected,
            actual: values.len(),
        });
    }
    for &value in values {
        require_finite(field, value)?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct StochasticSchrodingerConfig {
    time_step: f32,
    hopping_rate: f32,
    #[serde(alias = "decoherence_rate")]
    loss_rate: f32,
    noise_scale: f32,
}

impl Default for StochasticSchrodingerConfig {
    fn default() -> Self {
        Self {
            time_step: 0.1,
            hopping_rate: 1.0,
            loss_rate: 0.0,
            noise_scale: 0.0,
        }
    }
}

impl StochasticSchrodingerConfig {
    pub fn new(loss_rate: f32, noise_scale: f32) -> Result<Self, StochasticSchrodingerError> {
        let config = Self {
            loss_rate,
            noise_scale,
            ..Self::default()
        };
        config.validate()?;
        Ok(config)
    }

    pub fn with_time_step(mut self, time_step: f32) -> Result<Self, StochasticSchrodingerError> {
        self.time_step = time_step;
        self.validate()?;
        Ok(self)
    }

    pub fn with_hopping_rate(
        mut self,
        hopping_rate: f32,
    ) -> Result<Self, StochasticSchrodingerError> {
        self.hopping_rate = hopping_rate;
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), StochasticSchrodingerError> {
        require_non_negative("time_step", self.time_step)?;
        require_finite("hopping_rate", self.hopping_rate)?;
        require_non_negative("loss_rate", self.loss_rate)?;
        require_non_negative("noise_scale", self.noise_scale)?;
        require_derived_finite("hopping_angle", self.hopping_rate * self.time_step)?;
        require_derived_finite("noise_step_scale", self.noise_scale * self.time_step.sqrt())?;
        require_derived_finite(
            "ensemble_dephasing_rate",
            self.noise_scale * self.noise_scale,
        )?;
        require_derived_finite("damping_exponent", -0.5 * self.loss_rate * self.time_step)?;
        Ok(())
    }

    pub fn time_step(&self) -> f32 {
        self.time_step
    }

    pub fn hopping_rate(&self) -> f32 {
        self.hopping_rate
    }

    pub fn loss_rate(&self) -> f32 {
        self.loss_rate
    }

    /// Compatibility alias for the old name. Uniform no-jump attenuation is
    /// loss; ensemble decoherence is induced by independent phase diffusion.
    pub fn decoherence_rate(&self) -> f32 {
        self.loss_rate()
    }

    pub fn noise_scale(&self) -> f32 {
        self.noise_scale
    }

    pub fn hopping_angle(&self) -> f32 {
        self.hopping_rate * self.time_step
    }

    pub fn noise_step_scale(&self) -> f32 {
        self.noise_scale * self.time_step.sqrt()
    }

    pub fn damping_factor(&self) -> f32 {
        (-0.5 * self.loss_rate * self.time_step).exp()
    }

    pub fn ensemble_dephasing_rate(&self) -> f32 {
        self.noise_scale * self.noise_scale
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct StochasticSchrodingerAudit {
    pub rows: usize,
    pub features: usize,
    pub pair_count: usize,
    pub damping_factor: f32,
    pub ensemble_dephasing_rate: f32,
    pub expected_norm_ratio: f64,
    pub initial_norm_squared: f64,
    pub final_norm_squared: f64,
    pub expected_final_norm_squared: f64,
    pub max_row_norm_error: f64,
    pub max_row_norm_tolerance_ratio: f64,
    pub aggregate_norm_error: f64,
    pub aggregate_norm_tolerance: f64,
    pub max_phase_error: f64,
    pub max_output_error: f64,
    pub max_formula_tolerance_ratio: f64,
    pub phase_rms: f64,
    pub standard_normal_rms: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StochasticSchrodingerStep {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub integrator: &'static str,
    pub stochastic_calculus: &'static str,
    pub hamiltonian: &'static str,
    pub noise_model: &'static str,
    pub open_system_mode: &'static str,
    pub output_observable: &'static str,
    pub config: StochasticSchrodingerConfig,
    pub output_real: Vec<f32>,
    pub output_imaginary: Vec<f32>,
    pub phase: Vec<f32>,
    pub audit: StochasticSchrodingerAudit,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StochasticSchrodingerBackward {
    pub grad_input: Vec<f32>,
    pub grad_potential: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct StochasticSchrodingerBackwardAudit {
    pub rows: usize,
    pub features: usize,
    pub max_grad_input_error: f64,
    pub max_grad_potential_error: f64,
    pub max_formula_tolerance_ratio: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct StochasticSchrodingerAuditRequest<'a> {
    pub input: &'a [f32],
    pub potential: &'a [f32],
    pub output_real: &'a [f32],
    pub output_imaginary: &'a [f32],
    pub phase: &'a [f32],
    pub standard_normal: &'a [f32],
    pub rows: usize,
    pub features: usize,
    pub config: StochasticSchrodingerConfig,
}

#[derive(Clone, Copy, Debug)]
pub struct StochasticSchrodingerBackwardAuditRequest<'a> {
    pub input: &'a [f32],
    pub phase: &'a [f32],
    pub grad_output: &'a [f32],
    pub grad_input: &'a [f32],
    pub grad_potential: &'a [f32],
    pub rows: usize,
    pub features: usize,
    pub config: StochasticSchrodingerConfig,
}

/// Validates the wavefunction state and Hamiltonian before sampling noise.
pub fn validate_stochastic_schrodinger_state(
    input: &[f32],
    potential: &[f32],
    rows: usize,
    features: usize,
    config: StochasticSchrodingerConfig,
) -> Result<(), StochasticSchrodingerError> {
    config.validate()?;
    let volume = checked_volume(rows, features)?;
    validate_length("input", input, volume)?;
    validate_length("potential", potential, features)?;
    for &value in potential {
        require_derived_finite("phase", value * config.time_step())?;
    }
    Ok(())
}

pub fn validate_stochastic_schrodinger_inputs(
    input: &[f32],
    potential: &[f32],
    standard_normal: &[f32],
    rows: usize,
    features: usize,
    config: StochasticSchrodingerConfig,
) -> Result<(), StochasticSchrodingerError> {
    validate_stochastic_schrodinger_state(input, potential, rows, features, config)?;
    let volume = checked_volume(rows, features)?;
    validate_length("standard_normal", standard_normal, volume)?;
    for (index, &sample) in standard_normal.iter().enumerate() {
        let col = index % features;
        let phase = potential[col] * config.time_step() + sample * config.noise_step_scale();
        require_derived_finite("phase", phase)?;
    }
    Ok(())
}

fn norm_tolerance(features: usize, expected_norm_squared: f64) -> f64 {
    NORM_ERROR_FACTOR * (1.0 + (features as f64).sqrt()) * (1.0 + expected_norm_squared)
}

fn formula_tolerance(expected: f32) -> f64 {
    FORMULA_ERROR_FACTOR * (1.0 + f64::from(expected).abs())
}

fn adjacent_partner(col: usize, features: usize) -> Option<usize> {
    if col.is_multiple_of(2) {
        (col + 1 < features).then_some(col + 1)
    } else {
        Some(col - 1)
    }
}

fn stable_pair_phase_mean(left: f32, right: f32) -> f32 {
    0.5 * left + 0.5 * right
}

#[allow(clippy::too_many_arguments)]
fn evolve_real_input_quadratures(
    input: &[f32],
    phase: &[f32],
    row_start: usize,
    col: usize,
    features: usize,
    hopping_cos: f32,
    hopping_sin: f32,
    damping: f32,
) -> (f32, f32) {
    let index = row_start + col;
    let own_phase = phase[index];
    if let Some(partner_col) = adjacent_partner(col, features) {
        let partner = row_start + partner_col;
        let mean_phase = stable_pair_phase_mean(own_phase, phase[partner]);
        (
            damping
                * (hopping_cos * input[index] * own_phase.cos()
                    - hopping_sin * input[partner] * mean_phase.sin()),
            damping
                * (-hopping_cos * input[index] * own_phase.sin()
                    - hopping_sin * input[partner] * mean_phase.cos()),
        )
    } else {
        (
            damping * input[index] * own_phase.cos(),
            -damping * input[index] * own_phase.sin(),
        )
    }
}

pub fn audit_stochastic_schrodinger_step(
    request: StochasticSchrodingerAuditRequest<'_>,
) -> Result<StochasticSchrodingerAudit, StochasticSchrodingerError> {
    let StochasticSchrodingerAuditRequest {
        input,
        potential,
        output_real,
        output_imaginary,
        phase,
        standard_normal,
        rows,
        features,
        config,
    } = request;
    validate_stochastic_schrodinger_inputs(
        input,
        potential,
        standard_normal,
        rows,
        features,
        config,
    )?;
    let volume = checked_volume(rows, features)?;
    validate_length("output_real", output_real, volume)?;
    validate_length("output_imaginary", output_imaginary, volume)?;
    validate_length("phase", phase, volume)?;

    let damping_factor = config.damping_factor();
    require_derived_finite("damping_factor", damping_factor)?;
    let hopping_cos = config.hopping_angle().cos();
    let hopping_sin = config.hopping_angle().sin();
    require_derived_finite("hopping_cos", hopping_cos)?;
    require_derived_finite("hopping_sin", hopping_sin)?;
    let expected_norm_ratio = f64::from(damping_factor).powi(2);
    let mut expected_phase = vec![0.0; volume];
    let mut max_phase_error = 0.0f64;
    let mut max_output_error = 0.0f64;
    let mut max_formula_tolerance_ratio = 0.0f64;
    for index in 0..volume {
        let col = index % features;
        let expected = require_derived_finite(
            "expected_phase",
            potential[col] * config.time_step()
                + standard_normal[index] * config.noise_step_scale(),
        )?;
        expected_phase[index] = expected;
        let error = f64::from((phase[index] - expected).abs());
        let tolerance = formula_tolerance(expected);
        if error > tolerance {
            return Err(StochasticSchrodingerError::PhaseInvariant {
                index,
                error,
                tolerance,
            });
        }
        max_phase_error = max_phase_error.max(error);
        max_formula_tolerance_ratio = max_formula_tolerance_ratio.max(error / tolerance);
    }
    for row in 0..rows {
        let row_start = row * features;
        for col in 0..features {
            let index = row_start + col;
            let (expected_real, expected_imaginary) = evolve_real_input_quadratures(
                input,
                &expected_phase,
                row_start,
                col,
                features,
                hopping_cos,
                hopping_sin,
                damping_factor,
            );
            for (quadrature, actual, expected) in [
                ("real", output_real[index], expected_real),
                ("imaginary", output_imaginary[index], expected_imaginary),
            ] {
                let error = f64::from((actual - expected).abs());
                let tolerance = formula_tolerance(expected);
                if error > tolerance {
                    return Err(StochasticSchrodingerError::EvolutionInvariant {
                        quadrature,
                        index,
                        error,
                        tolerance,
                    });
                }
                max_output_error = max_output_error.max(error);
                max_formula_tolerance_ratio = max_formula_tolerance_ratio.max(error / tolerance);
            }
        }
    }

    let mut initial_norm_squared = 0.0f64;
    let mut final_norm_squared = 0.0f64;
    let mut expected_final_norm_squared = 0.0f64;
    let mut max_row_norm_error = 0.0f64;
    let mut max_row_norm_tolerance_ratio = 0.0f64;
    let mut aggregate_norm_tolerance = 0.0f64;
    for row in 0..rows {
        let start = row * features;
        let end = start + features;
        let initial = input[start..end]
            .iter()
            .map(|&value| f64::from(value).powi(2))
            .sum::<f64>();
        let final_norm = output_real[start..end]
            .iter()
            .zip(&output_imaginary[start..end])
            .map(|(&real, &imaginary)| f64::from(real).powi(2) + f64::from(imaginary).powi(2))
            .sum::<f64>();
        let expected = initial * expected_norm_ratio;
        let error = (final_norm - expected).abs();
        let tolerance = norm_tolerance(features, expected);
        if error > tolerance {
            return Err(StochasticSchrodingerError::NormInvariant {
                row,
                error,
                tolerance,
            });
        }
        initial_norm_squared += initial;
        final_norm_squared += final_norm;
        expected_final_norm_squared += expected;
        max_row_norm_error = max_row_norm_error.max(error);
        max_row_norm_tolerance_ratio = max_row_norm_tolerance_ratio.max(error / tolerance);
        aggregate_norm_tolerance += tolerance;
    }

    let aggregate_norm_error = (final_norm_squared - expected_final_norm_squared).abs();
    let phase_rms = if phase.is_empty() {
        0.0
    } else {
        (phase
            .iter()
            .map(|&value| f64::from(value).powi(2))
            .sum::<f64>()
            / phase.len() as f64)
            .sqrt()
    };
    let standard_normal_rms = if standard_normal.is_empty() {
        0.0
    } else {
        (standard_normal
            .iter()
            .map(|&value| f64::from(value).powi(2))
            .sum::<f64>()
            / standard_normal.len() as f64)
            .sqrt()
    };
    Ok(StochasticSchrodingerAudit {
        rows,
        features,
        pair_count: features / 2,
        damping_factor,
        ensemble_dephasing_rate: config.ensemble_dephasing_rate(),
        expected_norm_ratio,
        initial_norm_squared,
        final_norm_squared,
        expected_final_norm_squared,
        max_row_norm_error,
        max_row_norm_tolerance_ratio,
        aggregate_norm_error,
        aggregate_norm_tolerance,
        max_phase_error,
        max_output_error,
        max_formula_tolerance_ratio,
        phase_rms,
        standard_normal_rms,
    })
}

pub fn apply_stochastic_schrodinger_step(
    input: &[f32],
    potential: &[f32],
    standard_normal: &[f32],
    rows: usize,
    features: usize,
    config: StochasticSchrodingerConfig,
) -> Result<StochasticSchrodingerStep, StochasticSchrodingerError> {
    validate_stochastic_schrodinger_inputs(
        input,
        potential,
        standard_normal,
        rows,
        features,
        config,
    )?;
    let volume = checked_volume(rows, features)?;
    let mut output_real = vec![0.0; volume];
    let mut output_imaginary = vec![0.0; volume];
    let mut phase = vec![0.0; volume];
    let hopping_angle = config.hopping_angle();
    let hopping_cos = hopping_angle.cos();
    let hopping_sin = hopping_angle.sin();
    let damping = config.damping_factor();
    require_derived_finite("hopping_cos", hopping_cos)?;
    require_derived_finite("hopping_sin", hopping_sin)?;
    require_derived_finite("damping_factor", damping)?;

    for index in 0..volume {
        let col = index % features;
        phase[index] = require_derived_finite(
            "phase",
            potential[col] * config.time_step()
                + standard_normal[index] * config.noise_step_scale(),
        )?;
    }
    for row in 0..rows {
        let row_start = row * features;
        for col in 0..features {
            let index = row_start + col;
            let (real, imaginary) = evolve_real_input_quadratures(
                input,
                &phase,
                row_start,
                col,
                features,
                hopping_cos,
                hopping_sin,
                damping,
            );
            output_real[index] = require_derived_finite("output_real", real)?;
            output_imaginary[index] = require_derived_finite("output_imaginary", imaginary)?;
        }
    }
    let audit = audit_stochastic_schrodinger_step(StochasticSchrodingerAuditRequest {
        input,
        potential,
        output_real: &output_real,
        output_imaginary: &output_imaginary,
        phase: &phase,
        standard_normal,
        rows,
        features,
        config,
    })?;
    Ok(StochasticSchrodingerStep {
        kind: STOCHASTIC_SCHRODINGER_KIND,
        contract_version: STOCHASTIC_SCHRODINGER_CONTRACT_VERSION,
        semantic_owner: STOCHASTIC_SCHRODINGER_SEMANTIC_OWNER,
        semantic_backend: STOCHASTIC_SCHRODINGER_SEMANTIC_BACKEND,
        integrator: STOCHASTIC_SCHRODINGER_INTEGRATOR,
        stochastic_calculus: STOCHASTIC_SCHRODINGER_CALCULUS,
        hamiltonian: STOCHASTIC_SCHRODINGER_HAMILTONIAN,
        noise_model: STOCHASTIC_SCHRODINGER_NOISE_MODEL,
        open_system_mode: STOCHASTIC_SCHRODINGER_OPEN_SYSTEM_MODE,
        output_observable: STOCHASTIC_SCHRODINGER_OUTPUT_OBSERVABLE,
        config,
        output_real,
        output_imaginary,
        phase,
        audit,
    })
}

pub fn backward_stochastic_schrodinger_step(
    input: &[f32],
    phase: &[f32],
    grad_output: &[f32],
    rows: usize,
    features: usize,
    config: StochasticSchrodingerConfig,
) -> Result<StochasticSchrodingerBackward, StochasticSchrodingerError> {
    config.validate()?;
    let volume = checked_volume(rows, features)?;
    validate_length("input", input, volume)?;
    validate_length("phase", phase, volume)?;
    validate_length("grad_output", grad_output, volume)?;
    let hopping_angle = config.hopping_angle();
    let hopping_cos = hopping_angle.cos();
    let hopping_sin = hopping_angle.sin();
    let damping = config.damping_factor();
    let mut grad_input = vec![0.0; volume];
    let mut grad_potential = vec![0.0; features];

    for row in 0..rows {
        let row_start = row * features;
        for (col, grad_potential_value) in grad_potential.iter_mut().enumerate() {
            let index = row_start + col;
            let own_phase = phase[index];
            let (grad_amplitude, grad_phase) =
                if let Some(partner_col) = adjacent_partner(col, features) {
                    let partner = row_start + partner_col;
                    let mean_phase = stable_pair_phase_mean(own_phase, phase[partner]);
                    let own_grad = grad_output[index];
                    let partner_grad = grad_output[partner];
                    let grad_amplitude = damping
                        * (own_grad * hopping_cos * own_phase.cos()
                            - partner_grad * hopping_sin * mean_phase.sin());
                    let grad_phase = damping
                        * (own_grad
                            * (-hopping_cos * input[index] * own_phase.sin()
                                - 0.5 * hopping_sin * input[partner] * mean_phase.cos())
                            - 0.5 * partner_grad * hopping_sin * input[index] * mean_phase.cos());
                    (grad_amplitude, grad_phase)
                } else {
                    (
                        damping * grad_output[index] * own_phase.cos(),
                        -damping * grad_output[index] * input[index] * own_phase.sin(),
                    )
                };
            grad_input[index] = require_derived_finite("grad_input", grad_amplitude)?;
            *grad_potential_value = require_derived_finite(
                "grad_potential",
                *grad_potential_value + grad_phase * config.time_step(),
            )?;
        }
    }
    Ok(StochasticSchrodingerBackward {
        grad_input,
        grad_potential,
    })
}

/// Verifies externally executed gradients against the canonical Rust backward.
pub fn audit_stochastic_schrodinger_backward(
    request: StochasticSchrodingerBackwardAuditRequest<'_>,
) -> Result<StochasticSchrodingerBackwardAudit, StochasticSchrodingerError> {
    let StochasticSchrodingerBackwardAuditRequest {
        input,
        phase,
        grad_output,
        grad_input,
        grad_potential,
        rows,
        features,
        config,
    } = request;
    let expected =
        backward_stochastic_schrodinger_step(input, phase, grad_output, rows, features, config)?;
    let volume = checked_volume(rows, features)?;
    validate_length("grad_input", grad_input, volume)?;
    validate_length("grad_potential", grad_potential, features)?;

    let mut max_grad_input_error = 0.0f64;
    let mut max_grad_potential_error = 0.0f64;
    let mut max_formula_tolerance_ratio = 0.0f64;
    for (field, actual, expected_values, max_error) in [
        (
            "grad_input",
            grad_input,
            expected.grad_input.as_slice(),
            &mut max_grad_input_error,
        ),
        (
            "grad_potential",
            grad_potential,
            expected.grad_potential.as_slice(),
            &mut max_grad_potential_error,
        ),
    ] {
        for (index, (&actual, &expected)) in actual.iter().zip(expected_values).enumerate() {
            let error = f64::from((actual - expected).abs());
            let tolerance = formula_tolerance(expected);
            if error > tolerance {
                return Err(StochasticSchrodingerError::BackwardInvariant {
                    field,
                    index,
                    error,
                    tolerance,
                });
            }
            *max_error = (*max_error).max(error);
            max_formula_tolerance_ratio = max_formula_tolerance_ratio.max(error / tolerance);
        }
    }
    Ok(StochasticSchrodingerBackwardAudit {
        rows,
        features,
        max_grad_input_error,
        max_grad_potential_error,
        max_formula_tolerance_ratio,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> StochasticSchrodingerConfig {
        StochasticSchrodingerConfig::new(0.4, 0.2)
            .expect("valid config")
            .with_time_step(0.15)
            .expect("valid time step")
            .with_hopping_rate(0.7)
            .expect("valid hopping")
    }

    #[test]
    fn zero_time_is_exact_real_quadrature_identity() {
        let config = config().with_time_step(0.0).expect("zero time");
        let input = vec![0.3, -0.7, 1.2];
        let step = apply_stochastic_schrodinger_step(
            &input,
            &[0.8, -0.2, 0.5],
            &[1.0, -2.0, 0.3],
            1,
            3,
            config,
        )
        .expect("identity step");

        assert_eq!(step.output_real, input);
        assert_eq!(step.output_imaginary, vec![-0.0, 0.0, -0.0]);
        assert_eq!(
            step.contract_version,
            STOCHASTIC_SCHRODINGER_CONTRACT_VERSION
        );
        assert_eq!(step.hamiltonian, STOCHASTIC_SCHRODINGER_HAMILTONIAN);
        assert_eq!(step.noise_model, STOCHASTIC_SCHRODINGER_NOISE_MODEL);
        assert_eq!(
            step.open_system_mode,
            STOCHASTIC_SCHRODINGER_OPEN_SYSTEM_MODE
        );
        assert_eq!(step.audit.expected_norm_ratio, 1.0);
        assert_eq!(step.audit.max_row_norm_error, 0.0);
    }

    #[test]
    fn unitary_pair_hopping_preserves_norm_without_damping() {
        let config = StochasticSchrodingerConfig::new(0.0, 0.3)
            .expect("valid config")
            .with_time_step(0.2)
            .expect("valid time")
            .with_hopping_rate(1.1)
            .expect("valid hopping");
        let step = apply_stochastic_schrodinger_step(
            &[0.2, -0.5, 0.8, 0.1],
            &[0.4, -0.3, 0.2, 0.7],
            &[0.1, -0.2, 0.3, -0.4],
            1,
            4,
            config,
        )
        .expect("unitary step");

        assert!((step.audit.initial_norm_squared - step.audit.final_norm_squared).abs() < 1.0e-6);
        assert!(step
            .output_imaginary
            .iter()
            .any(|value| value.abs() > 1.0e-4));
    }

    #[test]
    fn no_jump_loss_contracts_norm_by_the_exact_ratio() {
        let step = apply_stochastic_schrodinger_step(
            &[0.2, -0.5, 0.8, 0.1],
            &[0.4, -0.3, 0.2, 0.7],
            &[0.0; 4],
            1,
            4,
            config(),
        )
        .expect("damped step");

        let expected = (-config().loss_rate() * config().time_step()).exp();
        assert!((step.audit.expected_norm_ratio - f64::from(expected)).abs() < 1.0e-6);
        assert!(step.audit.aggregate_norm_error <= step.audit.aggregate_norm_tolerance);
        assert_eq!(
            step.audit.ensemble_dephasing_rate,
            config().noise_scale().powi(2)
        );
    }

    #[test]
    fn analytic_backward_matches_central_difference() {
        let config = config();
        let input = vec![0.2, -0.5, 0.8, -0.1, 0.4, -0.7];
        let potential = vec![0.4, -0.3, 0.2];
        let noise = vec![0.1, -0.2, 0.3, -0.4, 0.2, 0.5];
        let grad = vec![0.7, -0.4, 0.2, -0.3, 0.6, -0.1];
        let step = apply_stochastic_schrodinger_step(&input, &potential, &noise, 2, 3, config)
            .expect("forward");
        let backward =
            backward_stochastic_schrodinger_step(&input, &step.phase, &grad, 2, 3, config)
                .expect("backward");
        let loss = |input: &[f32], potential: &[f32]| -> f32 {
            apply_stochastic_schrodinger_step(input, potential, &noise, 2, 3, config)
                .expect("perturbed forward")
                .output_real
                .iter()
                .zip(&grad)
                .map(|(value, grad)| value * grad)
                .sum()
        };
        let epsilon = 1.0e-3;
        for index in 0..input.len() {
            let mut plus = input.clone();
            let mut minus = input.clone();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numeric = (loss(&plus, &potential) - loss(&minus, &potential)) / (2.0 * epsilon);
            assert!((numeric - backward.grad_input[index]).abs() < 2.0e-4);
        }
        for index in 0..potential.len() {
            let mut plus = potential.clone();
            let mut minus = potential.clone();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numeric = (loss(&input, &plus) - loss(&input, &minus)) / (2.0 * epsilon);
            assert!((numeric - backward.grad_potential[index]).abs() < 2.0e-4);
        }
    }

    #[test]
    fn backward_audit_rejects_gradient_drift() {
        let input = [0.2, -0.5, 0.8];
        let potential = [0.4, -0.3, 0.2];
        let noise = [0.1, -0.2, 0.3];
        let grad_output = [0.7, -0.4, 0.2];
        let step = apply_stochastic_schrodinger_step(&input, &potential, &noise, 1, 3, config())
            .expect("forward");
        let backward =
            backward_stochastic_schrodinger_step(&input, &step.phase, &grad_output, 1, 3, config())
                .expect("backward");
        let request = |grad_input: &[f32], grad_potential: &[f32]| {
            audit_stochastic_schrodinger_backward(StochasticSchrodingerBackwardAuditRequest {
                input: &input,
                phase: &step.phase,
                grad_output: &grad_output,
                grad_input,
                grad_potential,
                rows: 1,
                features: 3,
                config: config(),
            })
        };
        let audit = request(&backward.grad_input, &backward.grad_potential)
            .expect("canonical backward audit");
        assert_eq!(audit.max_formula_tolerance_ratio, 0.0);

        let mut drifted = backward.grad_potential.clone();
        drifted[1] += 0.01;
        assert!(matches!(
            request(&backward.grad_input, &drifted),
            Err(StochasticSchrodingerError::BackwardInvariant {
                field: "grad_potential",
                index: 1,
                ..
            })
        ));
    }

    #[test]
    fn malformed_or_non_finite_inputs_fail_closed() {
        assert!(matches!(
            apply_stochastic_schrodinger_step(&[], &[], &[], 0, 0, config()),
            Err(StochasticSchrodingerError::EmptyFeatures)
        ));
        assert!(matches!(
            apply_stochastic_schrodinger_step(&[0.0, 1.0], &[0.0], &[0.0, 0.0], 1, 2, config()),
            Err(StochasticSchrodingerError::LengthMismatch {
                field: "potential",
                ..
            })
        ));
        assert!(matches!(
            apply_stochastic_schrodinger_step(
                &[0.0, f32::NAN],
                &[0.0, 0.0],
                &[0.0, 0.0],
                1,
                2,
                config()
            ),
            Err(StochasticSchrodingerError::NonFinite { field: "input", .. })
        ));
        let extreme = StochasticSchrodingerConfig::new(0.0, 0.0)
            .expect("finite config")
            .with_time_step(f32::MAX)
            .expect("finite derived scalars")
            .with_hopping_rate(0.0)
            .expect("zero hopping");
        assert!(matches!(
            apply_stochastic_schrodinger_step(&[1.0], &[f32::MAX], &[0.0], 1, 1, extreme),
            Err(StochasticSchrodingerError::NonFiniteDerived { field: "phase", .. })
        ));
    }

    #[test]
    fn audit_rejects_phase_or_evolution_drift() {
        let input = [0.2, -0.5, 0.8];
        let potential = [0.4, -0.3, 0.2];
        let noise = [0.1, -0.2, 0.3];
        let step = apply_stochastic_schrodinger_step(&input, &potential, &noise, 1, 3, config())
            .expect("forward");

        let mut phase = step.phase.clone();
        phase[0] += 0.01;
        assert!(matches!(
            audit_stochastic_schrodinger_step(StochasticSchrodingerAuditRequest {
                input: &input,
                potential: &potential,
                output_real: &step.output_real,
                output_imaginary: &step.output_imaginary,
                phase: &phase,
                standard_normal: &noise,
                rows: 1,
                features: 3,
                config: config(),
            }),
            Err(StochasticSchrodingerError::PhaseInvariant { index: 0, .. })
        ));

        let mut output_real = step.output_real.clone();
        output_real[1] += 0.01;
        assert!(matches!(
            audit_stochastic_schrodinger_step(StochasticSchrodingerAuditRequest {
                input: &input,
                potential: &potential,
                output_real: &output_real,
                output_imaginary: &step.output_imaginary,
                phase: &step.phase,
                standard_normal: &noise,
                rows: 1,
                features: 3,
                config: config(),
            }),
            Err(StochasticSchrodingerError::EvolutionInvariant {
                quadrature: "real",
                index: 1,
                ..
            })
        ));
    }

    #[test]
    fn legacy_decoherence_config_name_deserializes_as_loss() {
        let config: StochasticSchrodingerConfig = serde_json::from_value(serde_json::json!({
            "decoherence_rate": 0.4,
            "noise_scale": 0.2
        }))
        .expect("legacy config");
        assert_eq!(config.loss_rate(), 0.4);
        let serialized = serde_json::to_value(config).expect("serialize config");
        assert!(
            (serialized["loss_rate"].as_f64().expect("numeric loss rate") - 0.4).abs() < 1.0e-6
        );
        assert!(serialized.get("decoherence_rate").is_none());
    }
}
