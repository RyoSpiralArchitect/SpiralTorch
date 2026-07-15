// SPDX-License-Identifier: AGPL-3.0-or-later

//! Canonical open-topos guarded resonance dynamics.
//!
//! Every tensor value is a driven resonator stalk. Given the learned drive
//! `d = input * gate`, one transition unrolls the contraction
//!
//! `r_(n+1) = j_topos(d + coupling * r_n),  r_0 = 0`.
//!
//! The open-cartesian rewrite `j_topos` bounds the response, while
//! `0 <= coupling < 1` gives a unique fixed point and a finite amplification
//! bound. Rust owns the recurrence, stability gate, exact unrolled derivative,
//! and semantic audits. Execution backends only evaluate this contract.

use serde::{Deserialize, Serialize};
use st_tensor::topos::OpenCartesianTopos;
use thiserror::Error;

pub const TOPOS_RESONATOR_CONTRACT_VERSION: &str = "spiraltorch.topos_resonator.v1";
pub const TOPOS_RESONATOR_KIND: &str = "spiraltorch.topos_resonator";
pub const TOPOS_RESONATOR_SEMANTIC_OWNER: &str = "st-core::dynamics::topos_resonator";
pub const TOPOS_RESONATOR_SEMANTIC_BACKEND: &str = "rust";
pub const TOPOS_RESONATOR_EQUATION: &str = "r[n+1]=j_topos(input*gate+coupling*r[n]);r[0]=0";
pub const TOPOS_RESONATOR_REWRITE: &str = "open_cartesian_porous_rewrite";
pub const TOPOS_RESONATOR_SCHEME: &str = "finite_picard_iteration";
pub const TOPOS_RESONATOR_STATE: &str = "elementwise_resonance_stalk";
pub const TOPOS_RESONATOR_STABILITY: &str = "strict_contraction_and_open_topos_envelope";
pub const TOPOS_RESONATOR_BACKWARD: &str = "analytic_unrolled_drive_sensitivity";
pub const TOPOS_RESONATOR_MAX_ITERATIONS: usize = 4096;

const FORMULA_ERROR_FACTOR: f64 = 512.0 * f32::EPSILON as f64;

#[derive(Clone, Debug, Error, PartialEq)]
pub enum ToposResonatorError {
    #[error("Topos resonator feature count must be positive")]
    EmptyFeatures,
    #[error("Topos resonator tensor shape ({rows} x {features}) exceeds usize range")]
    ShapeOverflow { rows: usize, features: usize },
    #[error("Topos resonator field '{field}' has length {actual}, expected {expected}")]
    LengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("Topos resonator field '{field}' must be finite, got {value}")]
    NonFinite { field: &'static str, value: f32 },
    #[error("Topos resonator coupling must be in [0, 1), got {coupling}")]
    InvalidCoupling { coupling: f32 },
    #[error("Topos resonator iterations must be in 1..={limit}, got {iterations}")]
    InvalidIterations { iterations: usize, limit: usize },
    #[error("Topos resonator iterations {iterations} reach the loop-free topos depth {max_depth}")]
    ToposDepthExceeded { iterations: usize, max_depth: usize },
    #[error("Topos resonator volume {volume} exceeds the topos limit {max_volume}")]
    ToposVolumeExceeded { volume: usize, max_volume: usize },
    #[error("derived Topos resonator field '{field}' must be finite, got {value}")]
    NonFiniteDerived { field: &'static str, value: f32 },
    #[error(
        "Topos resonator field '{field}' index {index} error {error} exceeds tolerance {tolerance}"
    )]
    EvolutionInvariant {
        field: &'static str,
        index: usize,
        error: f64,
        tolerance: f64,
    },
    #[error(
        "Topos resonator backward field '{field}' index {index} error {error} exceeds tolerance {tolerance}"
    )]
    BackwardInvariant {
        field: &'static str,
        index: usize,
        error: f64,
        tolerance: f64,
    },
    #[error(
        "Topos resonator stability field '{field}' value {observed} exceeds bound {bound} by more than tolerance {tolerance}"
    )]
    StabilityInvariant {
        field: &'static str,
        observed: f64,
        bound: f64,
        tolerance: f64,
    },
}

fn require_finite(field: &'static str, value: f32) -> Result<f32, ToposResonatorError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ToposResonatorError::NonFinite { field, value })
    }
}

fn require_derived_finite(field: &'static str, value: f32) -> Result<f32, ToposResonatorError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ToposResonatorError::NonFiniteDerived { field, value })
    }
}

fn checked_volume(rows: usize, features: usize) -> Result<usize, ToposResonatorError> {
    if features == 0 {
        return Err(ToposResonatorError::EmptyFeatures);
    }
    rows.checked_mul(features)
        .ok_or(ToposResonatorError::ShapeOverflow { rows, features })
}

fn validate_length(
    field: &'static str,
    values: &[f32],
    expected: usize,
) -> Result<(), ToposResonatorError> {
    if values.len() != expected {
        return Err(ToposResonatorError::LengthMismatch {
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
pub struct ToposResonatorConfig {
    coupling: f32,
    iterations: usize,
}

impl Default for ToposResonatorConfig {
    fn default() -> Self {
        Self {
            coupling: 0.25,
            iterations: 4,
        }
    }
}

impl ToposResonatorConfig {
    pub fn new(coupling: f32, iterations: usize) -> Result<Self, ToposResonatorError> {
        let config = Self {
            coupling,
            iterations,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn with_coupling(mut self, coupling: f32) -> Result<Self, ToposResonatorError> {
        self.coupling = coupling;
        self.validate()?;
        Ok(self)
    }

    pub fn with_iterations(mut self, iterations: usize) -> Result<Self, ToposResonatorError> {
        self.iterations = iterations;
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), ToposResonatorError> {
        if !self.coupling.is_finite() || !(0.0..1.0).contains(&self.coupling) {
            return Err(ToposResonatorError::InvalidCoupling {
                coupling: self.coupling,
            });
        }
        if self.iterations == 0 || self.iterations > TOPOS_RESONATOR_MAX_ITERATIONS {
            return Err(ToposResonatorError::InvalidIterations {
                iterations: self.iterations,
                limit: TOPOS_RESONATOR_MAX_ITERATIONS,
            });
        }
        require_derived_finite("amplification_bound", self.amplification_bound())?;
        require_derived_finite(
            "finite_amplification_bound",
            self.finite_amplification_bound(),
        )?;
        Ok(())
    }

    pub fn coupling(&self) -> f32 {
        self.coupling
    }

    pub fn iterations(&self) -> usize {
        self.iterations
    }

    pub fn contraction_bound(&self) -> f32 {
        self.coupling
    }

    pub fn amplification_bound(&self) -> f32 {
        1.0 / (1.0 - self.coupling)
    }

    /// Tight gain bound for the configured finite Picard unroll.
    pub fn finite_amplification_bound(&self) -> f32 {
        let coupling = self.coupling as f64;
        let mut term = 1.0f64;
        let mut sum = 0.0f64;
        for _ in 0..self.iterations {
            sum += term;
            term *= coupling;
        }
        sum as f32
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ToposResonatorRequest<'a> {
    pub input: &'a [f32],
    pub gate: &'a [f32],
    pub rows: usize,
    pub features: usize,
    pub config: ToposResonatorConfig,
    pub topos: &'a OpenCartesianTopos,
}

#[derive(Clone, Copy, Debug)]
pub struct ToposResonatorBackwardRequest<'a> {
    pub request: ToposResonatorRequest<'a>,
    pub grad_output: &'a [f32],
}

#[derive(Clone, Copy, Debug)]
pub struct ToposResonatorAuditRequest<'a> {
    pub request: ToposResonatorRequest<'a>,
    pub output: &'a [f32],
}

#[derive(Clone, Copy, Debug)]
pub struct ToposResonatorBackwardAuditRequest<'a> {
    pub request: ToposResonatorBackwardRequest<'a>,
    pub grad_input: &'a [f32],
    pub grad_gate: &'a [f32],
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ToposResonatorAudit {
    pub rows: usize,
    pub features: usize,
    pub iterations: usize,
    pub coupling: f32,
    pub contraction_bound: f32,
    pub amplification_bound: f32,
    pub finite_amplification_bound: f32,
    pub curvature: f32,
    pub saturation: f32,
    pub porosity: f32,
    pub tolerance: f32,
    pub input_rms: f64,
    pub drive_rms: f64,
    pub output_rms: f64,
    pub observed_amplification: f64,
    pub amplification_margin: f64,
    pub last_update_linf: f32,
    pub fixed_point_residual_linf: f32,
    pub convergence_threshold: f64,
    pub converged: bool,
    pub closure_adjustment_linf: f32,
    pub rewritten_values: usize,
    pub max_output_error: f64,
    pub max_formula_tolerance_ratio: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ToposResonatorStep {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub equation: &'static str,
    pub rewrite: &'static str,
    pub scheme: &'static str,
    pub state: &'static str,
    pub stability: &'static str,
    pub config: ToposResonatorConfig,
    pub output: Vec<f32>,
    pub audit: ToposResonatorAudit,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToposResonatorBackward {
    pub grad_input: Vec<f32>,
    pub grad_gate: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ToposResonatorBackwardAudit {
    pub rows: usize,
    pub features: usize,
    pub iterations: usize,
    pub max_abs_drive_sensitivity: f32,
    pub drive_sensitivity_bound: f32,
    pub drive_sensitivity_margin: f64,
    pub grad_input_rms: f64,
    pub grad_gate_rms: f64,
    pub max_grad_input_error: f64,
    pub max_grad_gate_error: f64,
    pub max_formula_tolerance_ratio: f64,
}

fn validate_request(request: ToposResonatorRequest<'_>) -> Result<usize, ToposResonatorError> {
    request.config.validate()?;
    let volume = checked_volume(request.rows, request.features)?;
    validate_length("input", request.input, volume)?;
    validate_length("gate", request.gate, volume)?;
    if volume > request.topos.max_volume() {
        return Err(ToposResonatorError::ToposVolumeExceeded {
            volume,
            max_volume: request.topos.max_volume(),
        });
    }
    if request.config.iterations() >= request.topos.max_depth() {
        return Err(ToposResonatorError::ToposDepthExceeded {
            iterations: request.config.iterations(),
            max_depth: request.topos.max_depth(),
        });
    }
    Ok(volume)
}

fn root_mean_square(values: &[f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sum_squares = values
        .iter()
        .map(|&value| {
            let value = value as f64;
            value * value
        })
        .sum::<f64>();
    (sum_squares / values.len() as f64).sqrt()
}

#[derive(Debug)]
struct EvolvedResonance {
    drive: Vec<f32>,
    output: Vec<f32>,
    last_update_linf: f32,
    fixed_point_residual_linf: f32,
    closure_adjustment_linf: f32,
    rewritten_values: usize,
}

fn evolve_resonance(
    request: ToposResonatorRequest<'_>,
) -> Result<EvolvedResonance, ToposResonatorError> {
    let volume = validate_request(request)?;
    let mut drive = Vec::with_capacity(volume);
    for (&input, &gate) in request.input.iter().zip(request.gate) {
        drive.push(require_derived_finite("drive", input * gate)?);
    }
    let coupling = request.config.coupling();
    let mut state = vec![0.0f32; volume];
    let mut next = vec![0.0f32; volume];
    let mut last_update_linf = 0.0f32;
    let mut closure_adjustment_linf = 0.0f32;
    let mut rewritten_values = 0usize;
    for _ in 0..request.config.iterations() {
        last_update_linf = 0.0;
        for index in 0..volume {
            let raw =
                require_derived_finite("resonance_drive", drive[index] + coupling * state[index])?;
            let (rewritten, _) = request.topos.saturate_with_slope(raw);
            require_derived_finite("resonance_rewrite", rewritten)?;
            let adjustment = (rewritten - raw).abs();
            closure_adjustment_linf = closure_adjustment_linf.max(adjustment);
            if rewritten != raw {
                rewritten_values = rewritten_values.saturating_add(1);
            }
            last_update_linf = last_update_linf.max((rewritten - state[index]).abs());
            next[index] = rewritten;
        }
        std::mem::swap(&mut state, &mut next);
    }
    let mut fixed_point_residual_linf = 0.0f32;
    for index in 0..volume {
        let raw =
            require_derived_finite("fixed_point_drive", drive[index] + coupling * state[index])?;
        let target = request.topos.saturate(raw);
        fixed_point_residual_linf = fixed_point_residual_linf.max((target - state[index]).abs());
    }
    Ok(EvolvedResonance {
        drive,
        output: state,
        last_update_linf,
        fixed_point_residual_linf,
        closure_adjustment_linf,
        rewritten_values,
    })
}

fn formula_tolerance(expected: f32) -> f64 {
    FORMULA_ERROR_FACTOR * (1.0 + expected.abs() as f64)
}

fn validate_stability_bound(
    field: &'static str,
    observed: f64,
    bound: f32,
) -> Result<f64, ToposResonatorError> {
    let bound = bound as f64;
    let tolerance = FORMULA_ERROR_FACTOR * (1.0 + bound.abs());
    if observed > bound + tolerance {
        return Err(ToposResonatorError::StabilityInvariant {
            field,
            observed,
            bound,
            tolerance,
        });
    }
    Ok(bound - observed)
}

fn compare_field(
    field: &'static str,
    expected: &[f32],
    observed: &[f32],
    backward: bool,
) -> Result<(f64, f64), ToposResonatorError> {
    let mut max_error = 0.0f64;
    let mut max_ratio = 0.0f64;
    for (index, (&expected, &observed)) in expected.iter().zip(observed).enumerate() {
        let error = (expected as f64 - observed as f64).abs();
        let tolerance = formula_tolerance(expected);
        if error > tolerance {
            return if backward {
                Err(ToposResonatorError::BackwardInvariant {
                    field,
                    index,
                    error,
                    tolerance,
                })
            } else {
                Err(ToposResonatorError::EvolutionInvariant {
                    field,
                    index,
                    error,
                    tolerance,
                })
            };
        }
        max_error = max_error.max(error);
        max_ratio = max_ratio.max(error / tolerance.max(f64::MIN_POSITIVE));
    }
    Ok((max_error, max_ratio))
}

fn audit_from_evolved(
    request: ToposResonatorRequest<'_>,
    evolved: &EvolvedResonance,
    max_output_error: f64,
    max_formula_tolerance_ratio: f64,
) -> Result<ToposResonatorAudit, ToposResonatorError> {
    let input_rms = root_mean_square(request.input);
    let drive_rms = root_mean_square(&evolved.drive);
    let output_rms = root_mean_square(&evolved.output);
    let observed_amplification = if drive_rms > f64::EPSILON {
        output_rms / drive_rms
    } else {
        0.0
    };
    let finite_amplification_bound = request.config.finite_amplification_bound();
    let amplification_margin = validate_stability_bound(
        "observed_amplification",
        observed_amplification,
        finite_amplification_bound,
    )?;
    let max_abs_output = evolved
        .output
        .iter()
        .fold(0.0f32, |maximum, value| maximum.max(value.abs()));
    let convergence_threshold = request.topos.tolerance() as f64 * (1.0 + max_abs_output as f64);
    Ok(ToposResonatorAudit {
        rows: request.rows,
        features: request.features,
        iterations: request.config.iterations(),
        coupling: request.config.coupling(),
        contraction_bound: request.config.contraction_bound(),
        amplification_bound: request.config.amplification_bound(),
        finite_amplification_bound,
        curvature: request.topos.curvature(),
        saturation: request.topos.saturation(),
        porosity: request.topos.porosity(),
        tolerance: request.topos.tolerance(),
        input_rms,
        drive_rms,
        output_rms,
        observed_amplification,
        amplification_margin,
        last_update_linf: evolved.last_update_linf,
        fixed_point_residual_linf: evolved.fixed_point_residual_linf,
        convergence_threshold,
        converged: evolved.fixed_point_residual_linf as f64 <= convergence_threshold,
        closure_adjustment_linf: evolved.closure_adjustment_linf,
        rewritten_values: evolved.rewritten_values,
        max_output_error,
        max_formula_tolerance_ratio,
    })
}

pub fn validate_topos_resonator_state(
    request: ToposResonatorRequest<'_>,
) -> Result<(), ToposResonatorError> {
    validate_request(request)?;
    for (&input, &gate) in request.input.iter().zip(request.gate) {
        require_derived_finite("drive", input * gate)?;
    }
    Ok(())
}

pub fn apply_topos_resonator(
    request: ToposResonatorRequest<'_>,
) -> Result<ToposResonatorStep, ToposResonatorError> {
    let evolved = evolve_resonance(request)?;
    let audit = audit_from_evolved(request, &evolved, 0.0, 0.0)?;
    Ok(ToposResonatorStep {
        kind: TOPOS_RESONATOR_KIND,
        contract_version: TOPOS_RESONATOR_CONTRACT_VERSION,
        semantic_owner: TOPOS_RESONATOR_SEMANTIC_OWNER,
        semantic_backend: TOPOS_RESONATOR_SEMANTIC_BACKEND,
        equation: TOPOS_RESONATOR_EQUATION,
        rewrite: TOPOS_RESONATOR_REWRITE,
        scheme: TOPOS_RESONATOR_SCHEME,
        state: TOPOS_RESONATOR_STATE,
        stability: TOPOS_RESONATOR_STABILITY,
        config: request.config,
        output: evolved.output,
        audit,
    })
}

pub fn audit_topos_resonator(
    request: ToposResonatorAuditRequest<'_>,
) -> Result<ToposResonatorAudit, ToposResonatorError> {
    let volume = validate_request(request.request)?;
    validate_length("output", request.output, volume)?;
    let evolved = evolve_resonance(request.request)?;
    let (max_error, max_ratio) = compare_field("output", &evolved.output, request.output, false)?;
    audit_from_evolved(request.request, &evolved, max_error, max_ratio)
}

fn backward_resonance(
    request: ToposResonatorBackwardRequest<'_>,
) -> Result<(ToposResonatorBackward, f32), ToposResonatorError> {
    let volume = validate_request(request.request)?;
    validate_length("grad_output", request.grad_output, volume)?;
    let coupling = request.request.config.coupling();
    let iterations = request.request.config.iterations();
    let mut grad_input = Vec::with_capacity(volume);
    let mut grad_gate = Vec::with_capacity(volume);
    let mut max_abs_drive_sensitivity = 0.0f32;
    for index in 0..volume {
        let drive = require_derived_finite(
            "drive",
            request.request.input[index] * request.request.gate[index],
        )?;
        let mut state = 0.0f32;
        let mut drive_sensitivity = 0.0f32;
        for _ in 0..iterations {
            let raw = require_derived_finite("resonance_drive", drive + coupling * state)?;
            let (next_state, slope) = request.request.topos.saturate_with_slope(raw);
            drive_sensitivity = require_derived_finite(
                "drive_sensitivity",
                slope * (1.0 + coupling * drive_sensitivity),
            )?;
            state = next_state;
        }
        max_abs_drive_sensitivity = max_abs_drive_sensitivity.max(drive_sensitivity.abs());
        let grad_drive =
            require_derived_finite("grad_drive", request.grad_output[index] * drive_sensitivity)?;
        grad_input.push(require_derived_finite(
            "grad_input",
            grad_drive * request.request.gate[index],
        )?);
        grad_gate.push(require_derived_finite(
            "grad_gate",
            grad_drive * request.request.input[index],
        )?);
    }
    Ok((
        ToposResonatorBackward {
            grad_input,
            grad_gate,
        },
        max_abs_drive_sensitivity,
    ))
}

pub fn backward_topos_resonator(
    request: ToposResonatorBackwardRequest<'_>,
) -> Result<ToposResonatorBackward, ToposResonatorError> {
    backward_resonance(request).map(|(backward, _)| backward)
}

pub fn audit_topos_resonator_backward(
    request: ToposResonatorBackwardAuditRequest<'_>,
) -> Result<ToposResonatorBackwardAudit, ToposResonatorError> {
    let volume = validate_request(request.request.request)?;
    validate_length("grad_input", request.grad_input, volume)?;
    validate_length("grad_gate", request.grad_gate, volume)?;
    let (expected, max_abs_drive_sensitivity) = backward_resonance(request.request)?;
    let (max_grad_input_error, input_ratio) =
        compare_field("grad_input", &expected.grad_input, request.grad_input, true)?;
    let (max_grad_gate_error, gate_ratio) =
        compare_field("grad_gate", &expected.grad_gate, request.grad_gate, true)?;
    let drive_sensitivity_bound = request.request.request.config.finite_amplification_bound();
    let drive_sensitivity_margin = validate_stability_bound(
        "max_abs_drive_sensitivity",
        max_abs_drive_sensitivity as f64,
        drive_sensitivity_bound,
    )?;
    Ok(ToposResonatorBackwardAudit {
        rows: request.request.request.rows,
        features: request.request.request.features,
        iterations: request.request.request.config.iterations(),
        max_abs_drive_sensitivity,
        drive_sensitivity_bound,
        drive_sensitivity_margin,
        grad_input_rms: root_mean_square(request.grad_input),
        grad_gate_rms: root_mean_square(request.grad_gate),
        max_grad_input_error,
        max_grad_gate_error,
        max_formula_tolerance_ratio: input_ratio.max(gate_ratio),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn topos(saturation: f32, porosity: f32) -> OpenCartesianTopos {
        OpenCartesianTopos::new(-1.0, 1e-6, saturation, 32, 128)
            .unwrap()
            .with_porosity(porosity)
            .unwrap()
    }

    fn request<'a>(
        input: &'a [f32],
        gate: &'a [f32],
        config: ToposResonatorConfig,
        topos: &'a OpenCartesianTopos,
    ) -> ToposResonatorRequest<'a> {
        ToposResonatorRequest {
            input,
            gate,
            rows: 1,
            features: input.len(),
            config,
            topos,
        }
    }

    #[test]
    fn zero_coupling_is_the_canonical_topos_rewrite() {
        let topos = topos(1.0, 0.0);
        let config = ToposResonatorConfig::new(0.0, 4).unwrap();
        let step =
            apply_topos_resonator(request(&[0.5, 2.0], &[1.0, 1.0], config, &topos)).unwrap();
        assert_eq!(step.output, vec![0.5, 1.0]);
        assert_eq!(step.audit.rewritten_values, 4);
        assert_eq!(step.semantic_owner, TOPOS_RESONATOR_SEMANTIC_OWNER);
    }

    #[test]
    fn unsaturated_resonance_matches_the_geometric_response() {
        let topos = topos(100.0, 0.2);
        let config = ToposResonatorConfig::new(0.5, 4).unwrap();
        let step = apply_topos_resonator(request(&[2.0], &[0.5], config, &topos)).unwrap();
        assert!((step.output[0] - 1.875).abs() < 1e-6);
        assert!((step.audit.observed_amplification - 1.875).abs() < 1e-6);
        assert_eq!(step.audit.finite_amplification_bound, 1.875);
        assert!(step.audit.amplification_margin.abs() < 1e-6);
        assert!(step.audit.fixed_point_residual_linf > 0.0);
        assert!(step.audit.fixed_point_residual_linf < step.audit.last_update_linf);
    }

    #[test]
    fn open_topos_bounds_positive_feedback() {
        let topos = topos(1.0, 0.0);
        let config = ToposResonatorConfig::new(0.9, 16).unwrap();
        let step = apply_topos_resonator(request(&[100.0], &[100.0], config, &topos)).unwrap();
        assert_eq!(step.output, vec![1.0]);
        assert!(step.audit.closure_adjustment_linf > 1_000.0);
        assert!(step.audit.rewritten_values > 0);
        assert!(step.audit.converged);
    }

    #[test]
    fn backward_matches_finite_difference_away_from_rewrite_kinks() {
        let topos = topos(100.0, 0.2);
        let config = ToposResonatorConfig::new(0.35, 5).unwrap();
        let input = [0.4, -0.25];
        let gate = [1.2, 0.8];
        let grad_output = [0.7, -0.3];
        let core_request = request(&input, &gate, config, &topos);
        let backward = backward_topos_resonator(ToposResonatorBackwardRequest {
            request: core_request,
            grad_output: &grad_output,
        })
        .unwrap();
        let epsilon = 1e-3f32;
        for index in 0..input.len() {
            let mut plus = input;
            let mut minus = input;
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let plus_output = apply_topos_resonator(request(&plus, &gate, config, &topos))
                .unwrap()
                .output;
            let minus_output = apply_topos_resonator(request(&minus, &gate, config, &topos))
                .unwrap()
                .output;
            let plus_loss = plus_output
                .iter()
                .zip(grad_output)
                .map(|(&value, grad)| value * grad)
                .sum::<f32>();
            let minus_loss = minus_output
                .iter()
                .zip(grad_output)
                .map(|(&value, grad)| value * grad)
                .sum::<f32>();
            let numeric = (plus_loss - minus_loss) / (2.0 * epsilon);
            assert!((backward.grad_input[index] - numeric).abs() < 2e-4);
        }
        for index in 0..gate.len() {
            let mut plus = gate;
            let mut minus = gate;
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let plus_output = apply_topos_resonator(request(&input, &plus, config, &topos))
                .unwrap()
                .output;
            let minus_output = apply_topos_resonator(request(&input, &minus, config, &topos))
                .unwrap()
                .output;
            let plus_loss = plus_output
                .iter()
                .zip(grad_output)
                .map(|(&value, grad)| value * grad)
                .sum::<f32>();
            let minus_loss = minus_output
                .iter()
                .zip(grad_output)
                .map(|(&value, grad)| value * grad)
                .sum::<f32>();
            let numeric = (plus_loss - minus_loss) / (2.0 * epsilon);
            assert!((backward.grad_gate[index] - numeric).abs() < 2e-4);
        }
    }

    #[test]
    fn invalid_stability_and_topos_limits_fail_before_execution() {
        assert!(matches!(
            ToposResonatorConfig::new(1.0, 4),
            Err(ToposResonatorError::InvalidCoupling { .. })
        ));
        assert!(matches!(
            ToposResonatorConfig::new(0.2, 0),
            Err(ToposResonatorError::InvalidIterations { .. })
        ));
        let shallow = OpenCartesianTopos::new(-1.0, 1e-6, 10.0, 4, 8).unwrap();
        let config = ToposResonatorConfig::new(0.2, 4).unwrap();
        assert!(matches!(
            apply_topos_resonator(request(&[1.0], &[1.0], config, &shallow)),
            Err(ToposResonatorError::ToposDepthExceeded { .. })
        ));
    }

    #[test]
    fn semantic_audits_reject_forward_and_backward_drift() {
        let topos = topos(10.0, 0.2);
        let config = ToposResonatorConfig::default();
        let core_request = request(&[0.5, -0.25], &[1.0, 0.75], config, &topos);
        let step = apply_topos_resonator(core_request).unwrap();
        let mut drifted = step.output.clone();
        drifted[0] += 0.01;
        assert!(matches!(
            audit_topos_resonator(ToposResonatorAuditRequest {
                request: core_request,
                output: &drifted,
            }),
            Err(ToposResonatorError::EvolutionInvariant { .. })
        ));

        let backward_request = ToposResonatorBackwardRequest {
            request: core_request,
            grad_output: &[0.3, -0.2],
        };
        let backward = backward_topos_resonator(backward_request).unwrap();
        let mut drifted_gradient = backward.grad_gate.clone();
        drifted_gradient[1] += 0.01;
        assert!(matches!(
            audit_topos_resonator_backward(ToposResonatorBackwardAuditRequest {
                request: backward_request,
                grad_input: &backward.grad_input,
                grad_gate: &drifted_gradient,
            }),
            Err(ToposResonatorError::BackwardInvariant { .. })
        ));
    }

    #[test]
    fn empty_batch_remains_an_audited_transition() {
        let topos = topos(10.0, 0.2);
        let request = ToposResonatorRequest {
            input: &[],
            gate: &[],
            rows: 0,
            features: 2,
            config: ToposResonatorConfig::default(),
            topos: &topos,
        };
        let step = apply_topos_resonator(request).unwrap();
        assert!(step.output.is_empty());
        assert_eq!(step.audit.rows, 0);
        assert!(step.audit.converged);
    }
}
