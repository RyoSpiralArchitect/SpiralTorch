// SPDX-License-Identifier: AGPL-3.0-or-later

//! Canonical damped Klein-Gordon evolution on a periodic feature lattice.
//!
//! Every tensor row is an independent one-dimensional field trajectory. The
//! phase-space state is explicit: `field` is phi and `momentum` is d(phi)/dt.
//! A step uses exact half damping around a velocity-Verlet update of
//!
//! `d2(phi)/dt2 + damping * d(phi)/dt - c2 * laplacian(phi)
//!      + mass_squared * phi + self_coupling * phi^3 = source`.
//!
//! The learned `source` is a frozen scalar condensate during one step. Rust
//! owns the equation, stability gate, discrete adjoint, and audits. Execution
//! backends only evaluate this versioned contract.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const KLEIN_GORDON_CONTRACT_VERSION: &str = "spiraltorch.klein_gordon_phase_space.v1";
pub const KLEIN_GORDON_KIND: &str = "spiraltorch.klein_gordon_phase_space";
pub const KLEIN_GORDON_SEMANTIC_OWNER: &str = "st-core::dynamics::klein_gordon";
pub const KLEIN_GORDON_SEMANTIC_BACKEND: &str = "rust";
pub const KLEIN_GORDON_EQUATION: &str =
    "d2_phi_dt2+damping*d_phi_dt-c2*laplacian(phi)+mass_squared*phi+self_coupling*phi3=source";
pub const KLEIN_GORDON_INTEGRATOR: &str = "strang_exact_damping_velocity_verlet";
pub const KLEIN_GORDON_BOUNDARY: &str = "periodic_feature_lattice";
pub const KLEIN_GORDON_STATE: &str = "canonical_field_momentum";
pub const KLEIN_GORDON_SOURCE_MODEL: &str = "learned_static_scalar_condensate";
pub const KLEIN_GORDON_BACKWARD: &str = "analytic_discrete_adjoint";

/// Velocity-Verlet is linearly stable below phase 2. The smaller contract
/// limit leaves room for nonlinear curvature and backend roundoff.
pub const KLEIN_GORDON_MAX_PHASE_STEP: f32 = 1.9;

const FORMULA_ERROR_FACTOR: f64 = 512.0 * f32::EPSILON as f64;

#[derive(Clone, Debug, Error, PartialEq)]
pub enum KleinGordonError {
    #[error("Klein-Gordon feature count must be positive")]
    EmptyFeatures,
    #[error("Klein-Gordon tensor shape ({rows} x {features}) exceeds usize range")]
    ShapeOverflow { rows: usize, features: usize },
    #[error("Klein-Gordon field '{field}' has length {actual}, expected {expected}")]
    LengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("Klein-Gordon field '{field}' must be finite, got {value}")]
    NonFinite { field: &'static str, value: f32 },
    #[error("Klein-Gordon field '{field}' must be non-negative, got {value}")]
    Negative { field: &'static str, value: f32 },
    #[error("Klein-Gordon field '{field}' must be positive, got {value}")]
    NonPositive { field: &'static str, value: f32 },
    #[error("derived Klein-Gordon field '{field}' must be finite, got {value}")]
    NonFiniteDerived { field: &'static str, value: f32 },
    #[error("Klein-Gordon phase step {phase_step} exceeds stability limit {limit} ({stage})")]
    StabilityViolation {
        stage: &'static str,
        phase_step: f32,
        limit: f32,
    },
    #[error(
        "Klein-Gordon field '{field}' index {index} error {error} exceeds tolerance {tolerance}"
    )]
    EvolutionInvariant {
        field: &'static str,
        index: usize,
        error: f64,
        tolerance: f64,
    },
    #[error(
        "Klein-Gordon backward field '{field}' index {index} error {error} exceeds tolerance {tolerance}"
    )]
    BackwardInvariant {
        field: &'static str,
        index: usize,
        error: f64,
        tolerance: f64,
    },
}

fn require_finite(field: &'static str, value: f32) -> Result<f32, KleinGordonError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(KleinGordonError::NonFinite { field, value })
    }
}

fn require_non_negative(field: &'static str, value: f32) -> Result<f32, KleinGordonError> {
    require_finite(field, value)?;
    if value >= 0.0 {
        Ok(value)
    } else {
        Err(KleinGordonError::Negative { field, value })
    }
}

fn require_positive(field: &'static str, value: f32) -> Result<f32, KleinGordonError> {
    require_finite(field, value)?;
    if value > 0.0 {
        Ok(value)
    } else {
        Err(KleinGordonError::NonPositive { field, value })
    }
}

fn require_derived_finite(field: &'static str, value: f32) -> Result<f32, KleinGordonError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(KleinGordonError::NonFiniteDerived { field, value })
    }
}

fn checked_volume(rows: usize, features: usize) -> Result<usize, KleinGordonError> {
    if features == 0 {
        return Err(KleinGordonError::EmptyFeatures);
    }
    rows.checked_mul(features)
        .ok_or(KleinGordonError::ShapeOverflow { rows, features })
}

fn validate_length(
    field: &'static str,
    values: &[f32],
    expected: usize,
) -> Result<(), KleinGordonError> {
    if values.len() != expected {
        return Err(KleinGordonError::LengthMismatch {
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
pub struct KleinGordonConfig {
    time_step: f32,
    damping: f32,
    wave_speed: f32,
    lattice_spacing: f32,
    self_coupling: f32,
}

impl Default for KleinGordonConfig {
    fn default() -> Self {
        Self {
            time_step: 0.1,
            damping: 0.0,
            wave_speed: 1.0,
            lattice_spacing: 1.0,
            self_coupling: 0.0,
        }
    }
}

impl KleinGordonConfig {
    pub fn new(time_step: f32, damping: f32) -> Result<Self, KleinGordonError> {
        let config = Self {
            time_step,
            damping,
            ..Self::default()
        };
        config.validate()?;
        Ok(config)
    }

    pub fn with_wave_speed(mut self, wave_speed: f32) -> Result<Self, KleinGordonError> {
        self.wave_speed = wave_speed;
        self.validate()?;
        Ok(self)
    }

    pub fn with_lattice_spacing(mut self, lattice_spacing: f32) -> Result<Self, KleinGordonError> {
        self.lattice_spacing = lattice_spacing;
        self.validate()?;
        Ok(self)
    }

    pub fn with_self_coupling(mut self, self_coupling: f32) -> Result<Self, KleinGordonError> {
        self.self_coupling = self_coupling;
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), KleinGordonError> {
        require_positive("time_step", self.time_step)?;
        require_non_negative("damping", self.damping)?;
        require_non_negative("wave_speed", self.wave_speed)?;
        require_positive("lattice_spacing", self.lattice_spacing)?;
        require_non_negative("self_coupling", self.self_coupling)?;
        require_derived_finite("damping_exponent", -0.5 * self.damping * self.time_step)?;
        require_derived_finite("damping_half_factor", self.damping_half_factor())?;
        require_derived_finite("laplacian_scale", self.laplacian_scale())?;
        Ok(())
    }

    pub fn time_step(&self) -> f32 {
        self.time_step
    }

    pub fn damping(&self) -> f32 {
        self.damping
    }

    pub fn wave_speed(&self) -> f32 {
        self.wave_speed
    }

    pub fn lattice_spacing(&self) -> f32 {
        self.lattice_spacing
    }

    pub fn self_coupling(&self) -> f32 {
        self.self_coupling
    }

    pub fn damping_half_factor(&self) -> f32 {
        (-0.5 * self.damping * self.time_step).exp()
    }

    pub fn laplacian_scale(&self) -> f32 {
        let ratio = self.wave_speed / self.lattice_spacing;
        ratio * ratio
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct KleinGordonAudit {
    pub rows: usize,
    pub features: usize,
    pub damping_half_factor: f32,
    pub input_max_frequency_squared: f32,
    pub output_max_frequency_squared: f32,
    pub max_phase_step: f32,
    pub stability_limit: f32,
    pub stability_margin: f32,
    pub input_energy: f64,
    pub output_energy: f64,
    pub energy_change: f64,
    pub relative_energy_change: f64,
    pub max_field_error: f64,
    pub max_momentum_error: f64,
    pub max_formula_tolerance_ratio: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct KleinGordonStep {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub equation: &'static str,
    pub integrator: &'static str,
    pub boundary: &'static str,
    pub state: &'static str,
    pub source_model: &'static str,
    pub config: KleinGordonConfig,
    pub field: Vec<f32>,
    pub momentum: Vec<f32>,
    pub audit: KleinGordonAudit,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KleinGordonBackward {
    pub grad_field: Vec<f32>,
    pub grad_momentum: Vec<f32>,
    pub grad_mass_squared: Vec<f32>,
    pub grad_source: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct KleinGordonBackwardAudit {
    pub rows: usize,
    pub features: usize,
    pub max_grad_field_error: f64,
    pub max_grad_momentum_error: f64,
    pub max_grad_mass_squared_error: f64,
    pub max_grad_source_error: f64,
    pub max_formula_tolerance_ratio: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct KleinGordonAuditRequest<'a> {
    pub field: &'a [f32],
    pub momentum: &'a [f32],
    pub mass_squared: &'a [f32],
    pub source: &'a [f32],
    pub output_field: &'a [f32],
    pub output_momentum: &'a [f32],
    pub rows: usize,
    pub features: usize,
    pub config: KleinGordonConfig,
}

#[derive(Clone, Copy, Debug)]
pub struct KleinGordonBackwardRequest<'a> {
    pub field: &'a [f32],
    pub momentum: &'a [f32],
    pub mass_squared: &'a [f32],
    pub source: &'a [f32],
    pub grad_output_field: &'a [f32],
    pub grad_output_momentum: &'a [f32],
    pub rows: usize,
    pub features: usize,
    pub config: KleinGordonConfig,
}

#[derive(Clone, Copy, Debug)]
pub struct KleinGordonBackwardAuditRequest<'a> {
    pub request: KleinGordonBackwardRequest<'a>,
    pub grad_field: &'a [f32],
    pub grad_momentum: &'a [f32],
    pub grad_mass_squared: &'a [f32],
    pub grad_source: &'a [f32],
}

pub fn validate_klein_gordon_state(
    field: &[f32],
    momentum: &[f32],
    mass_squared: &[f32],
    source: &[f32],
    rows: usize,
    features: usize,
    config: KleinGordonConfig,
) -> Result<(), KleinGordonError> {
    config.validate()?;
    let volume = checked_volume(rows, features)?;
    validate_length("field", field, volume)?;
    validate_length("momentum", momentum, volume)?;
    validate_length("mass_squared", mass_squared, features)?;
    validate_length("source", source, features)?;
    let snapshot = stability_snapshot(field, mass_squared, features, config)?;
    require_stable("input", snapshot)?;
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct StabilitySnapshot {
    max_frequency_squared: f32,
    phase_step: f32,
}

fn stability_snapshot(
    field: &[f32],
    mass_squared: &[f32],
    features: usize,
    config: KleinGordonConfig,
) -> Result<StabilitySnapshot, KleinGordonError> {
    let mut max_local_curvature = 0.0f32;
    for (index, &value) in field.iter().enumerate() {
        let col = index % features;
        let nonlinear = 3.0 * config.self_coupling() * value * value;
        let curvature =
            require_derived_finite("local_curvature", (mass_squared[col] + nonlinear).abs())?;
        max_local_curvature = max_local_curvature.max(curvature);
    }
    if field.is_empty() {
        for &mass in mass_squared {
            max_local_curvature = max_local_curvature.max(mass.abs());
        }
    }
    let lattice_bound = if features > 1 {
        4.0 * config.laplacian_scale()
    } else {
        0.0
    };
    let max_frequency_squared =
        require_derived_finite("max_frequency_squared", lattice_bound + max_local_curvature)?;
    let phase_step = require_derived_finite(
        "phase_step",
        config.time_step() * max_frequency_squared.sqrt(),
    )?;
    Ok(StabilitySnapshot {
        max_frequency_squared,
        phase_step,
    })
}

fn require_stable(
    stage: &'static str,
    snapshot: StabilitySnapshot,
) -> Result<(), KleinGordonError> {
    if snapshot.phase_step <= KLEIN_GORDON_MAX_PHASE_STEP {
        Ok(())
    } else {
        Err(KleinGordonError::StabilityViolation {
            stage,
            phase_step: snapshot.phase_step,
            limit: KLEIN_GORDON_MAX_PHASE_STEP,
        })
    }
}

fn periodic_neighbors(index: usize, features: usize) -> (usize, usize) {
    let col = index % features;
    let row_start = index - col;
    let previous = row_start + if col == 0 { features - 1 } else { col - 1 };
    let next = row_start + if col + 1 == features { 0 } else { col + 1 };
    (previous, next)
}

fn force_into(
    field: &[f32],
    mass_squared: &[f32],
    source: &[f32],
    features: usize,
    config: KleinGordonConfig,
    output: &mut [f32],
) -> Result<(), KleinGordonError> {
    let laplacian_scale = config.laplacian_scale();
    let self_coupling = config.self_coupling();
    for (index, out) in output.iter_mut().enumerate() {
        let col = index % features;
        let value = field[index];
        let laplacian = if features == 1 {
            0.0
        } else {
            let (previous, next) = periodic_neighbors(index, features);
            field[previous] - 2.0 * value + field[next]
        };
        *out = require_derived_finite(
            "force",
            laplacian_scale * laplacian
                - mass_squared[col] * value
                - self_coupling * value * value * value
                + source[col],
        )?;
    }
    Ok(())
}

#[derive(Debug)]
struct EvolvedState {
    field: Vec<f32>,
    momentum: Vec<f32>,
    input_stability: StabilitySnapshot,
    output_stability: StabilitySnapshot,
}

fn evolve_klein_gordon(
    field: &[f32],
    momentum: &[f32],
    mass_squared: &[f32],
    source: &[f32],
    features: usize,
    config: KleinGordonConfig,
) -> Result<EvolvedState, KleinGordonError> {
    let input_stability = stability_snapshot(field, mass_squared, features, config)?;
    require_stable("input", input_stability)?;
    let volume = field.len();
    let time_step = config.time_step();
    let half_step = 0.5 * time_step;
    let damping = config.damping_half_factor();

    let mut force_initial = vec![0.0; volume];
    force_into(
        field,
        mass_squared,
        source,
        features,
        config,
        &mut force_initial,
    )?;
    let mut momentum_half = vec![0.0; volume];
    let mut output_field = vec![0.0; volume];
    for index in 0..volume {
        momentum_half[index] = require_derived_finite(
            "momentum_half",
            damping * momentum[index] + half_step * force_initial[index],
        )?;
        output_field[index] = require_derived_finite(
            "output_field",
            field[index] + time_step * momentum_half[index],
        )?;
    }

    let output_stability = stability_snapshot(&output_field, mass_squared, features, config)?;
    require_stable("output", output_stability)?;
    let mut force_final = vec![0.0; volume];
    force_into(
        &output_field,
        mass_squared,
        source,
        features,
        config,
        &mut force_final,
    )?;
    let mut output_momentum = vec![0.0; volume];
    for index in 0..volume {
        output_momentum[index] = require_derived_finite(
            "output_momentum",
            damping * (momentum_half[index] + half_step * force_final[index]),
        )?;
    }

    Ok(EvolvedState {
        field: output_field,
        momentum: output_momentum,
        input_stability,
        output_stability,
    })
}

fn total_energy(
    field: &[f32],
    momentum: &[f32],
    mass_squared: &[f32],
    source: &[f32],
    features: usize,
    config: KleinGordonConfig,
) -> f64 {
    let laplacian_scale = config.laplacian_scale() as f64;
    let self_coupling = config.self_coupling() as f64;
    let mut energy = 0.0f64;
    for index in 0..field.len() {
        let col = index % features;
        let value = field[index] as f64;
        let velocity = momentum[index] as f64;
        energy += 0.5 * velocity * velocity;
        energy += 0.5 * mass_squared[col] as f64 * value * value;
        energy += 0.25 * self_coupling * value.powi(4);
        energy -= source[col] as f64 * value;
        if features > 1 {
            let row_start = index - col;
            let next = row_start + if col + 1 == features { 0 } else { col + 1 };
            let delta = field[next] as f64 - value;
            energy += 0.5 * laplacian_scale * delta * delta;
        }
    }
    energy
}

fn formula_tolerance(expected: f32) -> f64 {
    FORMULA_ERROR_FACTOR * (1.0 + expected.abs() as f64)
}

fn compare_field(
    field: &'static str,
    expected: &[f32],
    observed: &[f32],
    backward: bool,
) -> Result<(f64, f64), KleinGordonError> {
    let mut max_error = 0.0f64;
    let mut max_ratio = 0.0f64;
    for (index, (&expected, &observed)) in expected.iter().zip(observed).enumerate() {
        let error = (expected as f64 - observed as f64).abs();
        let tolerance = formula_tolerance(expected);
        if error > tolerance {
            return if backward {
                Err(KleinGordonError::BackwardInvariant {
                    field,
                    index,
                    error,
                    tolerance,
                })
            } else {
                Err(KleinGordonError::EvolutionInvariant {
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

pub fn audit_klein_gordon_step(
    request: KleinGordonAuditRequest<'_>,
) -> Result<KleinGordonAudit, KleinGordonError> {
    validate_klein_gordon_state(
        request.field,
        request.momentum,
        request.mass_squared,
        request.source,
        request.rows,
        request.features,
        request.config,
    )?;
    let volume = checked_volume(request.rows, request.features)?;
    validate_length("output_field", request.output_field, volume)?;
    validate_length("output_momentum", request.output_momentum, volume)?;
    let expected = evolve_klein_gordon(
        request.field,
        request.momentum,
        request.mass_squared,
        request.source,
        request.features,
        request.config,
    )?;
    let (max_field_error, field_ratio) =
        compare_field("output_field", &expected.field, request.output_field, false)?;
    let (max_momentum_error, momentum_ratio) = compare_field(
        "output_momentum",
        &expected.momentum,
        request.output_momentum,
        false,
    )?;
    let input_energy = total_energy(
        request.field,
        request.momentum,
        request.mass_squared,
        request.source,
        request.features,
        request.config,
    );
    let output_energy = total_energy(
        request.output_field,
        request.output_momentum,
        request.mass_squared,
        request.source,
        request.features,
        request.config,
    );
    let energy_change = output_energy - input_energy;
    let energy_scale = input_energy.abs().max(output_energy.abs()).max(1.0e-12);
    Ok(KleinGordonAudit {
        rows: request.rows,
        features: request.features,
        damping_half_factor: request.config.damping_half_factor(),
        input_max_frequency_squared: expected.input_stability.max_frequency_squared,
        output_max_frequency_squared: expected.output_stability.max_frequency_squared,
        max_phase_step: expected
            .input_stability
            .phase_step
            .max(expected.output_stability.phase_step),
        stability_limit: KLEIN_GORDON_MAX_PHASE_STEP,
        stability_margin: KLEIN_GORDON_MAX_PHASE_STEP
            - expected
                .input_stability
                .phase_step
                .max(expected.output_stability.phase_step),
        input_energy,
        output_energy,
        energy_change,
        relative_energy_change: energy_change / energy_scale,
        max_field_error,
        max_momentum_error,
        max_formula_tolerance_ratio: field_ratio.max(momentum_ratio),
    })
}

pub fn apply_klein_gordon_step(
    field: &[f32],
    momentum: &[f32],
    mass_squared: &[f32],
    source: &[f32],
    rows: usize,
    features: usize,
    config: KleinGordonConfig,
) -> Result<KleinGordonStep, KleinGordonError> {
    validate_klein_gordon_state(
        field,
        momentum,
        mass_squared,
        source,
        rows,
        features,
        config,
    )?;
    let evolved = evolve_klein_gordon(field, momentum, mass_squared, source, features, config)?;
    let audit = audit_klein_gordon_step(KleinGordonAuditRequest {
        field,
        momentum,
        mass_squared,
        source,
        output_field: &evolved.field,
        output_momentum: &evolved.momentum,
        rows,
        features,
        config,
    })?;
    Ok(KleinGordonStep {
        kind: KLEIN_GORDON_KIND,
        contract_version: KLEIN_GORDON_CONTRACT_VERSION,
        semantic_owner: KLEIN_GORDON_SEMANTIC_OWNER,
        semantic_backend: KLEIN_GORDON_SEMANTIC_BACKEND,
        equation: KLEIN_GORDON_EQUATION,
        integrator: KLEIN_GORDON_INTEGRATOR,
        boundary: KLEIN_GORDON_BOUNDARY,
        state: KLEIN_GORDON_STATE,
        source_model: KLEIN_GORDON_SOURCE_MODEL,
        config,
        field: evolved.field,
        momentum: evolved.momentum,
        audit,
    })
}

fn add_force_jacobian_transpose(
    state: &[f32],
    mass_squared: &[f32],
    adjoint: &[f32],
    features: usize,
    config: KleinGordonConfig,
    output: &mut [f32],
) {
    let laplacian_scale = config.laplacian_scale();
    let self_coupling = config.self_coupling();
    for index in 0..state.len() {
        let col = index % features;
        let laplacian_adjoint = if features == 1 {
            0.0
        } else {
            let (previous, next) = periodic_neighbors(index, features);
            adjoint[previous] - 2.0 * adjoint[index] + adjoint[next]
        };
        output[index] += laplacian_scale * laplacian_adjoint
            - (mass_squared[col] + 3.0 * self_coupling * state[index] * state[index])
                * adjoint[index];
    }
}

pub fn backward_klein_gordon_step(
    request: KleinGordonBackwardRequest<'_>,
) -> Result<KleinGordonBackward, KleinGordonError> {
    validate_klein_gordon_state(
        request.field,
        request.momentum,
        request.mass_squared,
        request.source,
        request.rows,
        request.features,
        request.config,
    )?;
    let volume = checked_volume(request.rows, request.features)?;
    validate_length("grad_output_field", request.grad_output_field, volume)?;
    validate_length("grad_output_momentum", request.grad_output_momentum, volume)?;
    let evolved = evolve_klein_gordon(
        request.field,
        request.momentum,
        request.mass_squared,
        request.source,
        request.features,
        request.config,
    )?;
    let time_step = request.config.time_step();
    let half_step = 0.5 * time_step;
    let damping = request.config.damping_half_factor();

    let mut adjoint_force_final = vec![0.0; volume];
    for (out, &grad) in adjoint_force_final
        .iter_mut()
        .zip(request.grad_output_momentum)
    {
        *out = half_step * damping * grad;
    }
    let mut adjoint_field_final = request.grad_output_field.to_vec();
    add_force_jacobian_transpose(
        &evolved.field,
        request.mass_squared,
        &adjoint_force_final,
        request.features,
        request.config,
        &mut adjoint_field_final,
    );
    let mut adjoint_momentum_half = vec![0.0; volume];
    for index in 0..volume {
        adjoint_momentum_half[index] =
            damping * request.grad_output_momentum[index] + time_step * adjoint_field_final[index];
    }
    let mut adjoint_force_initial = vec![0.0; volume];
    for (out, &adjoint) in adjoint_force_initial.iter_mut().zip(&adjoint_momentum_half) {
        *out = half_step * adjoint;
    }
    let mut grad_field = adjoint_field_final;
    add_force_jacobian_transpose(
        request.field,
        request.mass_squared,
        &adjoint_force_initial,
        request.features,
        request.config,
        &mut grad_field,
    );
    let grad_momentum: Vec<f32> = adjoint_momentum_half
        .iter()
        .map(|&adjoint| damping * adjoint)
        .collect();
    let mut grad_mass_squared = vec![0.0; request.features];
    let mut grad_source = vec![0.0; request.features];
    for index in 0..volume {
        let col = index % request.features;
        grad_mass_squared[col] += -evolved.field[index] * adjoint_force_final[index]
            - request.field[index] * adjoint_force_initial[index];
        grad_source[col] += adjoint_force_final[index] + adjoint_force_initial[index];
    }
    for (field, values) in [
        ("grad_field", grad_field.as_slice()),
        ("grad_momentum", grad_momentum.as_slice()),
        ("grad_mass_squared", grad_mass_squared.as_slice()),
        ("grad_source", grad_source.as_slice()),
    ] {
        for &value in values {
            require_derived_finite(field, value)?;
        }
    }
    Ok(KleinGordonBackward {
        grad_field,
        grad_momentum,
        grad_mass_squared,
        grad_source,
    })
}

pub fn audit_klein_gordon_backward(
    request: KleinGordonBackwardAuditRequest<'_>,
) -> Result<KleinGordonBackwardAudit, KleinGordonError> {
    let expected = backward_klein_gordon_step(request.request)?;
    let volume = checked_volume(request.request.rows, request.request.features)?;
    validate_length("grad_field", request.grad_field, volume)?;
    validate_length("grad_momentum", request.grad_momentum, volume)?;
    validate_length(
        "grad_mass_squared",
        request.grad_mass_squared,
        request.request.features,
    )?;
    validate_length("grad_source", request.grad_source, request.request.features)?;
    let (max_grad_field_error, field_ratio) =
        compare_field("grad_field", &expected.grad_field, request.grad_field, true)?;
    let (max_grad_momentum_error, momentum_ratio) = compare_field(
        "grad_momentum",
        &expected.grad_momentum,
        request.grad_momentum,
        true,
    )?;
    let (max_grad_mass_squared_error, mass_ratio) = compare_field(
        "grad_mass_squared",
        &expected.grad_mass_squared,
        request.grad_mass_squared,
        true,
    )?;
    let (max_grad_source_error, source_ratio) = compare_field(
        "grad_source",
        &expected.grad_source,
        request.grad_source,
        true,
    )?;
    Ok(KleinGordonBackwardAudit {
        rows: request.request.rows,
        features: request.request.features,
        max_grad_field_error,
        max_grad_momentum_error,
        max_grad_mass_squared_error,
        max_grad_source_error,
        max_formula_tolerance_ratio: field_ratio
            .max(momentum_ratio)
            .max(mass_ratio)
            .max(source_ratio),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(left: f32, right: f32, tolerance: f32) {
        let error = (left - right).abs();
        assert!(
            error <= tolerance,
            "left={left} right={right} error={error} tolerance={tolerance}"
        );
    }

    fn loss(step: &KleinGordonStep, grad_field: &[f32], grad_momentum: &[f32]) -> f32 {
        step.field
            .iter()
            .zip(grad_field)
            .map(|(value, grad)| value * grad)
            .chain(
                step.momentum
                    .iter()
                    .zip(grad_momentum)
                    .map(|(value, grad)| value * grad),
            )
            .sum()
    }

    #[test]
    fn equilibrium_is_fixed_and_metadata_is_explicit() {
        let config = KleinGordonConfig::new(0.1, 0.0).unwrap();
        let mass = [0.5, 0.5, 0.5];
        let source = [0.1, 0.1, 0.1];
        let field = [0.2, 0.2, 0.2];
        let momentum = [0.0; 3];
        let step =
            apply_klein_gordon_step(&field, &momentum, &mass, &source, 1, 3, config).unwrap();
        assert_eq!(step.contract_version, KLEIN_GORDON_CONTRACT_VERSION);
        assert_eq!(step.semantic_owner, KLEIN_GORDON_SEMANTIC_OWNER);
        assert_eq!(step.integrator, KLEIN_GORDON_INTEGRATOR);
        assert_eq!(step.boundary, KLEIN_GORDON_BOUNDARY);
        for value in step.field {
            assert_close(value, 0.2, 1.0e-7);
        }
        for value in step.momentum {
            assert_close(value, 0.0, 1.0e-7);
        }
    }

    #[test]
    fn periodic_lattice_transports_local_displacement() {
        let config = KleinGordonConfig::new(0.1, 0.0).unwrap();
        let step = apply_klein_gordon_step(
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0; 4],
            &[0.0; 4],
            &[0.0; 4],
            1,
            4,
            config,
        )
        .unwrap();
        assert!(step.field[0] < 1.0);
        assert!(step.field[1] > 0.0);
        assert!(step.field[3] > 0.0);
        assert_close(step.field[1], step.field[3], 1.0e-7);
    }

    #[test]
    fn undamped_nonlinear_energy_remains_bounded_over_long_trajectory() {
        let config = KleinGordonConfig::new(0.02, 0.0)
            .unwrap()
            .with_wave_speed(0.7)
            .unwrap()
            .with_self_coupling(0.08)
            .unwrap();
        let mass = [0.2; 5];
        let source = [0.0; 5];
        let mut field = vec![0.4, -0.2, 0.1, 0.3, -0.15];
        let mut momentum = vec![0.0, 0.15, -0.1, 0.05, -0.08];
        let initial = total_energy(&field, &momentum, &mass, &source, 5, config);
        let mut max_relative_error = 0.0f64;
        for _ in 0..4_000 {
            let step =
                apply_klein_gordon_step(&field, &momentum, &mass, &source, 1, 5, config).unwrap();
            field = step.field;
            momentum = step.momentum;
            let energy = step.audit.output_energy;
            max_relative_error =
                max_relative_error.max((energy - initial).abs() / initial.abs().max(1.0e-12));
        }
        assert!(
            max_relative_error < 5.0e-4,
            "symplectic energy envelope drifted by {max_relative_error}"
        );
    }

    #[test]
    fn damping_reduces_energy_over_trajectory() {
        let config = KleinGordonConfig::new(0.03, 0.4).unwrap();
        let mass = [0.4; 4];
        let source = [0.0; 4];
        let mut field = vec![0.4, -0.3, 0.2, -0.1];
        let mut momentum = vec![0.3, -0.2, 0.1, 0.05];
        let initial = total_energy(&field, &momentum, &mass, &source, 4, config);
        let mut final_energy = initial;
        for _ in 0..1_000 {
            let step =
                apply_klein_gordon_step(&field, &momentum, &mass, &source, 1, 4, config).unwrap();
            final_energy = step.audit.output_energy;
            field = step.field;
            momentum = step.momentum;
        }
        assert!(final_energy < initial * 0.05);
    }

    #[test]
    fn analytic_discrete_adjoint_matches_central_difference() {
        let config = KleinGordonConfig::new(0.07, 0.03)
            .unwrap()
            .with_wave_speed(0.8)
            .unwrap()
            .with_lattice_spacing(1.2)
            .unwrap()
            .with_self_coupling(0.04)
            .unwrap();
        let field = vec![0.25, -0.32, 0.17];
        let momentum = vec![0.08, -0.05, 0.11];
        let mass = vec![0.2, -0.1, 0.3];
        let source = vec![0.03, -0.02, 0.01];
        let grad_field = vec![0.4, -0.25, 0.18];
        let grad_momentum = vec![-0.12, 0.21, 0.07];
        let request = KleinGordonBackwardRequest {
            field: &field,
            momentum: &momentum,
            mass_squared: &mass,
            source: &source,
            grad_output_field: &grad_field,
            grad_output_momentum: &grad_momentum,
            rows: 1,
            features: 3,
            config,
        };
        let backward = backward_klein_gordon_step(request).unwrap();
        let epsilon = 5.0e-4;
        for (kind, values, expected) in [
            ("field", field.as_slice(), backward.grad_field.as_slice()),
            (
                "momentum",
                momentum.as_slice(),
                backward.grad_momentum.as_slice(),
            ),
            (
                "mass",
                mass.as_slice(),
                backward.grad_mass_squared.as_slice(),
            ),
            ("source", source.as_slice(), backward.grad_source.as_slice()),
        ] {
            for index in 0..values.len() {
                let mut plus_field = field.clone();
                let mut minus_field = field.clone();
                let mut plus_momentum = momentum.clone();
                let mut minus_momentum = momentum.clone();
                let mut plus_mass = mass.clone();
                let mut minus_mass = mass.clone();
                let mut plus_source = source.clone();
                let mut minus_source = source.clone();
                match kind {
                    "field" => {
                        plus_field[index] += epsilon;
                        minus_field[index] -= epsilon;
                    }
                    "momentum" => {
                        plus_momentum[index] += epsilon;
                        minus_momentum[index] -= epsilon;
                    }
                    "mass" => {
                        plus_mass[index] += epsilon;
                        minus_mass[index] -= epsilon;
                    }
                    "source" => {
                        plus_source[index] += epsilon;
                        minus_source[index] -= epsilon;
                    }
                    _ => unreachable!(),
                }
                let plus = apply_klein_gordon_step(
                    &plus_field,
                    &plus_momentum,
                    &plus_mass,
                    &plus_source,
                    1,
                    3,
                    config,
                )
                .unwrap();
                let minus = apply_klein_gordon_step(
                    &minus_field,
                    &minus_momentum,
                    &minus_mass,
                    &minus_source,
                    1,
                    3,
                    config,
                )
                .unwrap();
                let numerical = (loss(&plus, &grad_field, &grad_momentum)
                    - loss(&minus, &grad_field, &grad_momentum))
                    / (2.0 * epsilon);
                assert_close(expected[index], numerical, 4.0e-4);
            }
        }
    }

    #[test]
    fn audits_reject_forward_and_backward_drift() {
        let config = KleinGordonConfig::new(0.1, 0.05).unwrap();
        let field = [0.2, -0.1, 0.3];
        let momentum = [0.05, 0.02, -0.04];
        let mass = [0.2, 0.25, 0.3];
        let source = [0.01, -0.02, 0.03];
        let step =
            apply_klein_gordon_step(&field, &momentum, &mass, &source, 1, 3, config).unwrap();
        let mut bad_field = step.field.clone();
        bad_field[1] += 0.01;
        let error = audit_klein_gordon_step(KleinGordonAuditRequest {
            field: &field,
            momentum: &momentum,
            mass_squared: &mass,
            source: &source,
            output_field: &bad_field,
            output_momentum: &step.momentum,
            rows: 1,
            features: 3,
            config,
        })
        .unwrap_err();
        assert!(matches!(error, KleinGordonError::EvolutionInvariant { .. }));

        let backward_request = KleinGordonBackwardRequest {
            field: &field,
            momentum: &momentum,
            mass_squared: &mass,
            source: &source,
            grad_output_field: &[0.2, -0.3, 0.1],
            grad_output_momentum: &[0.0; 3],
            rows: 1,
            features: 3,
            config,
        };
        let backward = backward_klein_gordon_step(backward_request).unwrap();
        let mut bad_gradient = backward.grad_field.clone();
        bad_gradient[0] += 0.02;
        let error = audit_klein_gordon_backward(KleinGordonBackwardAuditRequest {
            request: backward_request,
            grad_field: &bad_gradient,
            grad_momentum: &backward.grad_momentum,
            grad_mass_squared: &backward.grad_mass_squared,
            grad_source: &backward.grad_source,
        })
        .unwrap_err();
        assert!(matches!(error, KleinGordonError::BackwardInvariant { .. }));
    }

    #[test]
    fn stability_and_non_finite_inputs_fail_closed() {
        let unstable = KleinGordonConfig::new(2.0, 0.0).unwrap();
        let error =
            apply_klein_gordon_step(&[1.0, 0.0], &[0.0; 2], &[1.0; 2], &[0.0; 2], 1, 2, unstable)
                .unwrap_err();
        assert!(matches!(error, KleinGordonError::StabilityViolation { .. }));

        let error = apply_klein_gordon_step(
            &[f32::NAN],
            &[0.0],
            &[0.1],
            &[0.0],
            1,
            1,
            KleinGordonConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(error, KleinGordonError::NonFinite { .. }));
    }

    #[test]
    fn empty_batch_preserves_parameter_shape() {
        let step = apply_klein_gordon_step(
            &[],
            &[],
            &[0.1, 0.2],
            &[0.0, 0.0],
            0,
            2,
            KleinGordonConfig::default(),
        )
        .unwrap();
        assert!(step.field.is_empty());
        assert!(step.momentum.is_empty());
        assert_eq!(step.audit.rows, 0);
        assert!(step.audit.input_energy.abs() < f64::EPSILON);
    }
}
