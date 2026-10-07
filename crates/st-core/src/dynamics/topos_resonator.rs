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

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ToposResonatorBackward {
    pub grad_input: Vec<f32>,
    pub grad_gate: Vec<f32>,
}

/// Immutable CPU operator sharing the same transition and VJP as the NN layer.
/// Clients transport tensor rows; no client-side recurrence is required.
#[derive(Clone, Debug)]
pub struct ToposResonatorOperator {
    config: ToposResonatorConfig,
    topos: OpenCartesianTopos,
}

/// Owned finite-unroll tape. The caller retains optimizer and broadcast policy.
/// Four f32 vectors are retained; VJPs do not rerun the resonance or read live inputs.
#[derive(Clone, Debug)]
pub struct ToposResonatorLearningBatch {
    input: Vec<f32>,
    gate: Vec<f32>,
    drive_sensitivity: Vec<f32>,
    step: ToposResonatorStep,
}

impl ToposResonatorLearningBatch {
    pub fn input(&self) -> &[f32] {
        &self.input
    }

    pub fn gate(&self) -> &[f32] {
        &self.gate
    }

    pub fn step(&self) -> &ToposResonatorStep {
        &self.step
    }

    pub fn output(&self) -> &[f32] {
        &self.step.output
    }

    pub fn vjp(&self, grad_output: &[f32]) -> Result<ToposResonatorBackward, ToposResonatorError> {
        validate_length("grad_output", grad_output, self.input.len())?;
        let mut grad_input = Vec::with_capacity(self.input.len());
        let mut grad_gate = Vec::with_capacity(self.input.len());
        for (index, &upstream) in grad_output.iter().enumerate() {
            // Keep (upstream * sensitivity) * parameter in the legacy order.
            // Caching sensitivity * parameter would change f32 rounding.
            let grad_drive =
                require_derived_finite("grad_drive", upstream * self.drive_sensitivity[index])?;
            grad_input.push(require_derived_finite(
                "grad_input",
                grad_drive * self.gate[index],
            )?);
            grad_gate.push(require_derived_finite(
                "grad_gate",
                grad_drive * self.input[index],
            )?);
        }
        Ok(ToposResonatorBackward {
            grad_input,
            grad_gate,
        })
    }

    /// Audit a pullback of this immutable, already-audited transition without
    /// replaying its Picard iterations. External executor results still use
    /// `audit_topos_resonator_backward` for independent formula comparison.
    pub fn vjp_audited(
        &self,
        grad_output: &[f32],
    ) -> Result<(ToposResonatorBackward, ToposResonatorBackwardAudit), ToposResonatorError> {
        let backward = self.vjp(grad_output)?;
        let max_abs_drive_sensitivity = self
            .drive_sensitivity
            .iter()
            .fold(0.0f32, |maximum, value| maximum.max(value.abs()));
        let audit = backward_audit_from_gradients(
            self.step.audit.rows,
            self.step.audit.features,
            self.step.config,
            max_abs_drive_sensitivity,
            &backward.grad_input,
            &backward.grad_gate,
        )?;
        Ok((backward, audit))
    }
}

impl ToposResonatorOperator {
    pub fn new(
        config: ToposResonatorConfig,
        topos: OpenCartesianTopos,
    ) -> Result<Self, ToposResonatorError> {
        let operator = Self { config, topos };
        validate_topos_resonator_state(operator.request(&[], &[], 0, 1))?;
        Ok(operator)
    }

    pub fn config(&self) -> ToposResonatorConfig {
        self.config
    }

    pub fn topos(&self) -> &OpenCartesianTopos {
        &self.topos
    }

    fn request<'a>(
        &'a self,
        input: &'a [f32],
        gate: &'a [f32],
        rows: usize,
        features: usize,
    ) -> ToposResonatorRequest<'a> {
        ToposResonatorRequest {
            input,
            gate,
            rows,
            features,
            config: self.config,
            topos: &self.topos,
        }
    }

    pub fn forward(
        &self,
        input: &[f32],
        gate: &[f32],
        rows: usize,
        features: usize,
    ) -> Result<ToposResonatorStep, ToposResonatorError> {
        apply_topos_resonator(self.request(input, gate, rows, features))
    }

    /// Capture the same audited forward and exact finite-unroll sensitivity.
    /// Inputs and gate are per-element; no reduction or averaging is introduced.
    pub fn capture(
        &self,
        input: &[f32],
        gate: &[f32],
        rows: usize,
        features: usize,
    ) -> Result<ToposResonatorLearningBatch, ToposResonatorError> {
        let (step, drive_sensitivity) = self.capture_step(input, gate, rows, features)?;
        Ok(ToposResonatorLearningBatch {
            input: input.to_vec(),
            gate: gate.to_vec(),
            drive_sensitivity,
            step,
        })
    }

    /// Retains both Rust-owned input allocations without cloning them.
    /// The audited transition is identical to `capture`; buffers are consumed
    /// even on failure. Foreign clients must still establish Rust ownership.
    pub fn capture_owned(
        &self,
        input: Vec<f32>,
        gate: Vec<f32>,
        rows: usize,
        features: usize,
    ) -> Result<ToposResonatorLearningBatch, ToposResonatorError> {
        let (step, drive_sensitivity) = self.capture_step(&input, &gate, rows, features)?;
        Ok(ToposResonatorLearningBatch {
            input,
            gate,
            drive_sensitivity,
            step,
        })
    }

    fn capture_step(
        &self,
        input: &[f32],
        gate: &[f32],
        rows: usize,
        features: usize,
    ) -> Result<(ToposResonatorStep, Vec<f32>), ToposResonatorError> {
        let request = self.request(input, gate, rows, features);
        let mut evolved = evolve_resonance::<true>(request)?;
        let drive_sensitivity = std::mem::take(&mut evolved.drive_sensitivity);
        let step = finish_resonance(request, evolved)?;
        Ok((step, drive_sensitivity))
    }

    pub fn backward(
        &self,
        input: &[f32],
        gate: &[f32],
        grad_output: &[f32],
        rows: usize,
        features: usize,
    ) -> Result<ToposResonatorBackward, ToposResonatorError> {
        backward_topos_resonator(ToposResonatorBackwardRequest {
            request: self.request(input, gate, rows, features),
            grad_output,
        })
    }
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
    drive_sensitivity: Vec<f32>,
    last_update_linf: f32,
    fixed_point_residual_linf: f32,
    closure_adjustment_linf: f32,
    rewritten_values: usize,
}

fn evolve_resonance<const CAPTURE_SENSITIVITY: bool>(
    request: ToposResonatorRequest<'_>,
) -> Result<EvolvedResonance, ToposResonatorError> {
    let volume = validate_request(request)?;
    let mut drive = Vec::with_capacity(volume);
    for (&input, &gate) in request.input.iter().zip(request.gate) {
        drive.push(require_derived_finite("drive", input * gate)?);
    }
    let coupling = request.config.coupling();
    let mut state = vec![0.0f32; volume];
    let mut sensitivity = if CAPTURE_SENSITIVITY {
        vec![0.0f32; volume]
    } else {
        Vec::new()
    };
    let mut last_update_linf = 0.0f32;
    let mut closure_adjustment_linf = 0.0f32;
    let mut rewritten_values = 0usize;
    for iteration in 0..request.config.iterations() {
        for index in 0..volume {
            let raw =
                require_derived_finite("resonance_drive", drive[index] + coupling * state[index])?;
            let (rewritten, slope) = request.topos.saturate_with_slope(raw);
            require_derived_finite("resonance_rewrite", rewritten)?;
            if CAPTURE_SENSITIVITY {
                sensitivity[index] = require_derived_finite(
                    "drive_sensitivity",
                    slope * (1.0 + coupling * sensitivity[index]),
                )?;
            }
            let adjustment = (rewritten - raw).abs();
            closure_adjustment_linf = closure_adjustment_linf.max(adjustment);
            if rewritten != raw {
                rewritten_values = rewritten_values.saturating_add(1);
            }
            if iteration + 1 == request.config.iterations() {
                last_update_linf = last_update_linf.max((rewritten - state[index]).abs());
            }
            // Stalks are independent; retain iteration-major guard order without a second buffer.
            state[index] = rewritten;
        }
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
        drive_sensitivity: sensitivity,
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
    finish_resonance(request, evolve_resonance::<false>(request)?)
}

fn finish_resonance(
    request: ToposResonatorRequest<'_>,
    evolved: EvolvedResonance,
) -> Result<ToposResonatorStep, ToposResonatorError> {
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
    let evolved = evolve_resonance::<false>(request.request)?;
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
    let mut audit = backward_audit_from_gradients(
        request.request.request.rows,
        request.request.request.features,
        request.request.request.config,
        max_abs_drive_sensitivity,
        request.grad_input,
        request.grad_gate,
    )?;
    audit.max_grad_input_error = max_grad_input_error;
    audit.max_grad_gate_error = max_grad_gate_error;
    audit.max_formula_tolerance_ratio = input_ratio.max(gate_ratio);
    Ok(audit)
}

fn backward_audit_from_gradients(
    rows: usize,
    features: usize,
    config: ToposResonatorConfig,
    max_abs_drive_sensitivity: f32,
    grad_input: &[f32],
    grad_gate: &[f32],
) -> Result<ToposResonatorBackwardAudit, ToposResonatorError> {
    let drive_sensitivity_bound = config.finite_amplification_bound();
    let drive_sensitivity_margin = validate_stability_bound(
        "max_abs_drive_sensitivity",
        max_abs_drive_sensitivity as f64,
        drive_sensitivity_bound,
    )?;
    Ok(ToposResonatorBackwardAudit {
        rows,
        features,
        iterations: config.iterations(),
        max_abs_drive_sensitivity,
        drive_sensitivity_bound,
        drive_sensitivity_margin,
        grad_input_rms: root_mean_square(grad_input),
        grad_gate_rms: root_mean_square(grad_gate),
        max_grad_input_error: 0.0,
        max_grad_gate_error: 0.0,
        max_formula_tolerance_ratio: 0.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captured_audited_vjp_matches_recomputed_audit_and_gradient_bits() {
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for rows in [0, 1, 7] {
            for (coupling, iterations) in [(0.0, 1), (0.25, 4), (0.9, 16)] {
                for porosity in [0.0, 0.2, 1.0] {
                    let features = 17;
                    let input: Vec<_> = (0..rows * features)
                        .map(|i| (i % 13) as f32 / 3.0 - 2.0)
                        .collect();
                    let gate: Vec<_> = (0..input.len())
                        .map(|i| (i % 7) as f32 / 2.0 - 1.0)
                        .collect();
                    let topos = topos(1.0, porosity);
                    let config = ToposResonatorConfig::new(coupling, iterations).unwrap();
                    let operator = ToposResonatorOperator::new(config, topos.clone()).unwrap();
                    let batch = operator.capture(&input, &gate, rows, features).unwrap();
                    assert_eq!(bits(batch.input()), bits(&input));
                    assert_eq!(bits(batch.gate()), bits(&gate));
                    for sign in [-1.0, 0.0, 1.0] {
                        let dy: Vec<_> = (0..input.len())
                            .map(|i| sign * ((i % 11) as f32 / 7.0 - 0.5))
                            .collect();
                        let request = ToposResonatorBackwardRequest {
                            request: operator.request(&input, &gate, rows, features),
                            grad_output: &dy,
                        };
                        let expected = backward_topos_resonator(request).unwrap();
                        let expected_audit =
                            audit_topos_resonator_backward(ToposResonatorBackwardAuditRequest {
                                request,
                                grad_input: &expected.grad_input,
                                grad_gate: &expected.grad_gate,
                            })
                            .unwrap();
                        let (actual, audit) = batch.vjp_audited(&dy).unwrap();
                        assert_eq!(bits(&actual.grad_input), bits(&expected.grad_input));
                        assert_eq!(bits(&actual.grad_gate), bits(&expected.grad_gate));
                        assert_eq!(
                            serde_json::to_string(&audit).unwrap(),
                            serde_json::to_string(&expected_audit).unwrap()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn captured_audited_vjp_preserves_errors_and_recovers_without_live_inputs() {
        let batch = ToposResonatorOperator::new(ToposResonatorConfig::default(), topos(1.0, 0.2))
            .unwrap()
            .capture_owned(vec![1.0], vec![0.0], 1, 1)
            .unwrap();
        let expected = batch.vjp_audited(&[0.5]).unwrap();
        for dy in [vec![], vec![f32::NAN], vec![f32::INFINITY], vec![f32::MAX]] {
            assert_eq!(
                format!("{:?}", batch.vjp_audited(&dy).unwrap_err()),
                format!("{:?}", batch.vjp(&dy).unwrap_err())
            );
            assert_eq!(batch.vjp_audited(&[0.5]).unwrap(), expected);
        }
    }

    #[test]
    fn owned_capture_retains_allocations_and_matches_borrowed_audits_and_pullbacks() {
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for rows in [0, 1, 3] {
            for (coupling, iterations) in [(0.0, 1), (0.25, 4), (0.75, 16)] {
                for porosity in [0.0, 0.2] {
                    let features = 7;
                    let volume = rows * features;
                    let dy = vec![0.3; volume];
                    let (borrowed, owned, expected) = {
                        let operator = ToposResonatorOperator::new(
                            ToposResonatorConfig::new(coupling, iterations).unwrap(),
                            topos(1.0, porosity),
                        )
                        .unwrap();
                        let mut input = Vec::with_capacity(volume + 13);
                        let mut gate = Vec::with_capacity(volume + 17);
                        let pattern = [0.0, -0.0, f32::from_bits(1), -0.3, 0.5, 1.3, -2.0];
                        input.extend((0..volume).map(|i| pattern[i % pattern.len()]));
                        gate.extend((0..volume).map(|i| (i % 3) as f32 - 1.0));
                        let expected = operator
                            .backward(&input, &gate, &dy, rows, features)
                            .unwrap();
                        let borrowed = operator.capture(&input, &gate, rows, features).unwrap();
                        let allocations = (
                            input.as_ptr(),
                            input.capacity(),
                            gate.as_ptr(),
                            gate.capacity(),
                        );
                        let owned = operator.capture_owned(input, gate, rows, features).unwrap();
                        assert_eq!(
                            (
                                owned.input.as_ptr(),
                                owned.input.capacity(),
                                owned.gate.as_ptr(),
                                owned.gate.capacity()
                            ),
                            allocations
                        );
                        (borrowed, owned, expected)
                    };
                    assert_eq!(bits(owned.output()), bits(borrowed.output()));
                    assert_eq!(
                        serde_json::to_string(owned.step()).unwrap(),
                        serde_json::to_string(borrowed.step()).unwrap()
                    );
                    for _ in 0..2 {
                        let actual = owned.vjp(&dy).unwrap();
                        assert_eq!(bits(&actual.grad_input), bits(&expected.grad_input));
                        assert_eq!(bits(&actual.grad_gate), bits(&expected.grad_gate));
                    }
                    let cloned = owned.clone();
                    drop(owned);
                    assert_eq!(bits(cloned.output()), bits(borrowed.output()));
                    assert_eq!(
                        bits(&cloned.vjp(&dy).unwrap().grad_gate),
                        bits(&expected.grad_gate)
                    );
                }
            }
        }
    }

    #[test]
    fn owned_capture_preserves_guard_errors_without_corrupting_the_operator() {
        let operator =
            ToposResonatorOperator::new(ToposResonatorConfig::default(), topos(1.0, 0.2)).unwrap();
        let good = operator.capture_owned(vec![1.0], vec![0.2], 1, 1).unwrap();
        for (input, gate, rows, features) in [
            (vec![], vec![], 0, 0),
            (vec![], vec![], usize::MAX, 2),
            (vec![1.0], vec![], 1, 1),
            (vec![1.0; 129], vec![1.0; 129], 1, 129),
            (vec![f32::NAN], vec![1.0], 1, 1),
            (vec![1.0], vec![f32::INFINITY], 1, 1),
            (vec![f32::MAX], vec![2.0], 1, 1),
        ] {
            let expected = operator.capture(&input, &gate, rows, features).unwrap_err();
            let actual = operator
                .capture_owned(input, gate, rows, features)
                .unwrap_err();
            assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
            let next = operator.capture_owned(vec![1.0], vec![0.2], 1, 1).unwrap();
            assert_eq!(next.step(), good.step());
            assert_eq!(next.vjp(&[0.3]).unwrap(), good.vjp(&[0.3]).unwrap());
        }
    }

    // Frozen iteration-major, double-buffered traversal from before the in-place change.
    fn legacy_evolution<const CAPTURE: bool>(
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
        let mut sensitivity = if CAPTURE {
            vec![0.0f32; volume]
        } else {
            Vec::new()
        };
        let mut last_update_linf = 0.0f32;
        let mut closure_adjustment_linf = 0.0f32;
        let mut rewritten_values = 0usize;
        for _ in 0..request.config.iterations() {
            last_update_linf = 0.0;
            for index in 0..volume {
                let raw = require_derived_finite(
                    "resonance_drive",
                    drive[index] + coupling * state[index],
                )?;
                let (rewritten, slope) = request.topos.saturate_with_slope(raw);
                require_derived_finite("resonance_rewrite", rewritten)?;
                if CAPTURE {
                    sensitivity[index] = require_derived_finite(
                        "drive_sensitivity",
                        slope * (1.0 + coupling * sensitivity[index]),
                    )?;
                }
                closure_adjustment_linf = closure_adjustment_linf.max((rewritten - raw).abs());
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
            let raw = require_derived_finite(
                "fixed_point_drive",
                drive[index] + coupling * state[index],
            )?;
            let target = request.topos.saturate(raw);
            fixed_point_residual_linf =
                fixed_point_residual_linf.max((target - state[index]).abs());
        }
        Ok(EvolvedResonance {
            drive,
            output: state,
            drive_sensitivity: sensitivity,
            last_update_linf,
            fixed_point_residual_linf,
            closure_adjustment_linf,
            rewritten_values,
        })
    }

    fn assert_legacy_evolution<const CAPTURE: bool>(
        request: ToposResonatorRequest<'_>,
        expect_success: bool,
    ) {
        let expected = legacy_evolution::<CAPTURE>(request);
        assert_eq!(expected.is_ok(), expect_success);
        let actual = evolve_resonance::<CAPTURE>(request);
        match (actual, expected) {
            (Ok(actual), Ok(expected)) => {
                let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&actual.drive), bits(&expected.drive));
                assert_eq!(bits(&actual.output), bits(&expected.output));
                assert_eq!(
                    bits(&actual.drive_sensitivity),
                    bits(&expected.drive_sensitivity)
                );
                assert_eq!(
                    actual.last_update_linf.to_bits(),
                    expected.last_update_linf.to_bits()
                );
                assert_eq!(
                    actual.fixed_point_residual_linf.to_bits(),
                    expected.fixed_point_residual_linf.to_bits()
                );
                assert_eq!(
                    actual.closure_adjustment_linf.to_bits(),
                    expected.closure_adjustment_linf.to_bits()
                );
                assert_eq!(actual.rewritten_values, expected.rewritten_values);
                let actual = finish_resonance(request, actual).unwrap();
                let expected = finish_resonance(request, expected).unwrap();
                assert_eq!(
                    serde_json::to_string(&actual).unwrap(),
                    serde_json::to_string(&expected).unwrap()
                );
            }
            (actual, expected) => assert_eq!(format!("{actual:?}"), format!("{expected:?}")),
        }
    }

    #[test]
    fn in_place_evolution_preserves_legacy_bits_and_audits() {
        let inputs = [
            0.0,
            -0.0,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::from_bits(1.0f32.to_bits() - 1),
            1.0,
            f32::from_bits(1.0f32.to_bits() + 1),
            -1.0,
            2.0,
            -40.0,
        ];
        let gates = [0.0, -0.0, 1.0, -1.0, 0.5, 2.0, -0.75];
        for volume in [0, 1, 17, 64, 65, 128, 257] {
            let input: Vec<_> = (0..volume).map(|i| inputs[i % inputs.len()]).collect();
            let gate: Vec<_> = (0..volume).map(|i| gates[i % gates.len()]).collect();
            for (coupling, iterations) in [(0.0, 1), (0.25, 2), (0.25, 4), (0.75, 16), (0.99, 63)] {
                for porosity in [
                    0.0,
                    f32::EPSILON,
                    f32::from_bits(f32::EPSILON.to_bits() + 1),
                    0.2,
                    1.0,
                ] {
                    for saturation in [1e-20, 1.0, 1e20] {
                        let topos = OpenCartesianTopos::new(-1.0, 1e-6, saturation, 64, 257)
                            .unwrap()
                            .with_porosity(porosity)
                            .unwrap();
                        let request = ToposResonatorRequest {
                            input: &input,
                            gate: &gate,
                            rows: usize::from(volume != 0),
                            features: volume.max(1),
                            config: ToposResonatorConfig::new(coupling, iterations).unwrap(),
                            topos: &topos,
                        };
                        assert_legacy_evolution::<false>(request, true);
                        assert_legacy_evolution::<true>(request, true);
                    }
                }
            }
        }
        let topos = OpenCartesianTopos::new(-1.0, 1e-6, 1.0, 4097, 10).unwrap();
        let request = request(
            &inputs,
            &[1.0; 10],
            ToposResonatorConfig::new(f32::from_bits(1.0f32.to_bits() - 1), 4096).unwrap(),
            &topos,
        );
        assert_legacy_evolution::<false>(request, true);
        assert_legacy_evolution::<true>(request, true);
    }

    #[test]
    fn in_place_evolution_preserves_legacy_error_order() {
        let topos = OpenCartesianTopos::new(-1.0, 1e-6, f32::MAX, 32, 128).unwrap();
        for (input, gate) in [
            (vec![f32::NAN], vec![f32::INFINITY]),
            (vec![1.0], vec![f32::NEG_INFINITY]),
            (vec![f32::MAX, -f32::MAX], vec![2.0, 2.0]),
            // Later elements fail earlier iterations; point-major traversal would reorder errors.
            (vec![f32::MAX * 0.45, -f32::MAX], vec![1.0, 1.0]),
            (vec![-f32::MAX * 0.45, f32::MAX], vec![1.0, 1.0]),
        ] {
            for iterations in [1, 2, 3, 16] {
                let request = request(
                    &input,
                    &gate,
                    ToposResonatorConfig::new(0.9, iterations).unwrap(),
                    &topos,
                );
                assert_legacy_evolution::<false>(request, false);
                assert_legacy_evolution::<true>(request, false);
            }
        }
    }

    #[test]
    fn captured_pullbacks_preserve_audit_bits_ownership_and_invalid_upstream_guards() {
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for rows in [0, 1, 7] {
            for (coupling, iterations) in [(0.0, 1), (0.25, 4), (0.9, 16)] {
                for porosity in [0.0, 0.2] {
                    let features = 17;
                    let mut input: Vec<_> = (0..rows * features)
                        .map(|i| (i % 13) as f32 / 3.0 - 2.0)
                        .collect();
                    let mut gate: Vec<_> = (0..input.len())
                        .map(|i| (i % 7) as f32 / 2.0 - 1.0)
                        .collect();
                    let dy: Vec<_> = (0..input.len())
                        .map(|i| (i % 11) as f32 / 7.0 - 0.5)
                        .collect();
                    let (forward, expected, snapshot) = {
                        let operator = ToposResonatorOperator::new(
                            ToposResonatorConfig::new(coupling, iterations).unwrap(),
                            topos(1.0, porosity),
                        )
                        .unwrap();
                        (
                            operator.forward(&input, &gate, rows, features).unwrap(),
                            operator
                                .backward(&input, &gate, &dy, rows, features)
                                .unwrap(),
                            operator.capture(&input, &gate, rows, features).unwrap(),
                        )
                    };
                    assert_eq!(bits(snapshot.output()), bits(&forward.output));
                    assert_eq!(
                        serde_json::to_string(snapshot.step()).unwrap(),
                        serde_json::to_string(&forward).unwrap()
                    );
                    input.fill(f32::NAN);
                    gate.fill(f32::NAN);
                    for _ in 0..2 {
                        let actual = snapshot.vjp(&dy).unwrap();
                        assert_eq!(bits(&actual.grad_input), bits(&expected.grad_input));
                        assert_eq!(bits(&actual.grad_gate), bits(&expected.grad_gate));
                    }
                    assert!(snapshot.vjp(&vec![0.0; dy.len() + 1]).is_err());
                    if !dy.is_empty() {
                        assert!(snapshot.vjp(&vec![f32::NAN; dy.len()]).is_err());
                    }
                }
            }
        }
    }

    #[test]
    fn capture_retains_shape_budget_finite_and_gradient_overflow_guards() {
        let operator =
            ToposResonatorOperator::new(ToposResonatorConfig::default(), topos(1.0, 0.2)).unwrap();
        assert!(operator.capture(&[], &[], 0, 0).is_err());
        assert!(operator.capture(&[], &[], usize::MAX, 2).is_err());
        assert!(operator.capture(&[1.0], &[1.0], 1, 2).is_err());
        assert!(operator.capture(&[1.0; 129], &[1.0; 129], 1, 129).is_err());
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(operator.capture(&[invalid], &[1.0], 1, 1).is_err());
            assert!(operator.capture(&[1.0], &[invalid], 1, 1).is_err());
        }
        assert!(operator.capture(&[f32::MAX], &[2.0], 1, 1).is_err());
        let batch = operator.capture(&[1.0], &[0.0], 1, 1).unwrap();
        assert!(operator
            .backward(&[1.0], &[0.0], &[f32::MAX], 1, 1)
            .is_err());
        assert!(batch.vjp(&[f32::MAX]).is_err());
        assert!(batch.vjp(&[1.0]).is_ok());
    }

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
    fn immutable_operator_preserves_core_forward_backward_and_validation() {
        let topos = topos(1.0, 0.2);
        let config = ToposResonatorConfig::new(0.35, 6).unwrap();
        let operator = ToposResonatorOperator::new(config, topos.clone()).unwrap();
        let input = [0.2, -0.4, 4.0, -3.0];
        let gate = [0.3, -0.2, 0.7, 0.5];
        let grad_output = [0.4, -0.7, -0.3, 0.8];
        let request = ToposResonatorRequest {
            input: &input,
            gate: &gate,
            rows: 2,
            features: 2,
            config,
            topos: &topos,
        };
        assert_eq!(
            operator.forward(&input, &gate, 2, 2).unwrap(),
            apply_topos_resonator(request).unwrap()
        );
        assert_eq!(
            operator
                .backward(&input, &gate, &grad_output, 2, 2)
                .unwrap(),
            backward_topos_resonator(ToposResonatorBackwardRequest {
                request,
                grad_output: &grad_output
            })
            .unwrap()
        );
        assert!(operator.forward(&input, &gate, 2, 3).is_err());
        assert!(operator
            .backward(&input, &gate, &[f32::NAN; 4], 2, 2)
            .is_err());
        assert!(operator.forward(&[], &[], 0, 2).unwrap().output.is_empty());
        assert!(operator
            .backward(&[], &[], &[], 0, 2)
            .unwrap()
            .grad_gate
            .is_empty());
        let shallow = OpenCartesianTopos::new(-1.0, 1e-6, 1.0, 6, 128).unwrap();
        assert!(ToposResonatorOperator::new(config, shallow).is_err());
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
