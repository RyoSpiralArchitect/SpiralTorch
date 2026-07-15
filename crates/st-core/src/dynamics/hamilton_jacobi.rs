// SPDX-License-Identifier: AGPL-3.0-or-later

//! Canonical Hamilton-Jacobi evolution on a periodic feature lattice.
//!
//! Every tensor row is an independent scalar action field `S(x)`. A step solves
//!
//! `dS/dt + |grad(S)|^2 / (2 * mass) + potential = viscosity * laplacian(S)`
//!
//! with a global Lax-Friedrichs numerical Hamiltonian. The configured
//! characteristic-speed limit supplies the numerical viscosity and is checked
//! against the action field before every transition. Rust owns the equation,
//! monotone CFL gate, discrete adjoint, and audits. Execution backends only
//! evaluate this versioned contract.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const HAMILTON_JACOBI_CONTRACT_VERSION: &str = "spiraltorch.hamilton_jacobi_action.v1";
pub const HAMILTON_JACOBI_KIND: &str = "spiraltorch.hamilton_jacobi_action";
pub const HAMILTON_JACOBI_SEMANTIC_OWNER: &str = "st-core::dynamics::hamilton_jacobi";
pub const HAMILTON_JACOBI_SEMANTIC_BACKEND: &str = "rust";
pub const HAMILTON_JACOBI_EQUATION: &str =
    "dS_dt+grad(S)^2/(2*mass)+potential=viscosity*laplacian(S)";
pub const HAMILTON_JACOBI_HAMILTONIAN: &str =
    "quadratic_kinetic_convex_in_momentum_plus_learned_static_potential";
pub const HAMILTON_JACOBI_SCHEME: &str = "global_lax_friedrichs_forward_euler";
pub const HAMILTON_JACOBI_BOUNDARY: &str = "periodic_feature_lattice";
pub const HAMILTON_JACOBI_STATE: &str = "scalar_action_field";
pub const HAMILTON_JACOBI_STABILITY: &str = "characteristic_bound_and_monotone_cfl";
pub const HAMILTON_JACOBI_BACKWARD: &str = "analytic_discrete_adjoint";

/// A small margin below the sharp monotone CFL limit leaves room for backend
/// roundoff and repeated integration.
pub const HAMILTON_JACOBI_MAX_MONOTONE_CFL: f32 = 0.95;

const FORMULA_ERROR_FACTOR: f64 = 512.0 * f32::EPSILON as f64;
const CHARACTERISTIC_ERROR_FACTOR: f32 = 64.0 * f32::EPSILON;

#[derive(Clone, Debug, Error, PartialEq)]
pub enum HamiltonJacobiError {
    #[error("Hamilton-Jacobi feature count must be positive")]
    EmptyFeatures,
    #[error("Hamilton-Jacobi tensor shape ({rows} x {features}) exceeds usize range")]
    ShapeOverflow { rows: usize, features: usize },
    #[error("Hamilton-Jacobi field '{field}' has length {actual}, expected {expected}")]
    LengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("Hamilton-Jacobi field '{field}' must be finite, got {value}")]
    NonFinite { field: &'static str, value: f32 },
    #[error("Hamilton-Jacobi field '{field}' must be non-negative, got {value}")]
    Negative { field: &'static str, value: f32 },
    #[error("Hamilton-Jacobi field '{field}' must be positive, got {value}")]
    NonPositive { field: &'static str, value: f32 },
    #[error("derived Hamilton-Jacobi field '{field}' must be finite, got {value}")]
    NonFiniteDerived { field: &'static str, value: f32 },
    #[error(
        "Hamilton-Jacobi characteristic speed {observed} exceeds configured limit {limit} ({stage})"
    )]
    CharacteristicBound {
        stage: &'static str,
        observed: f32,
        limit: f32,
    },
    #[error("Hamilton-Jacobi monotone CFL {observed} exceeds limit {limit}")]
    MonotoneCfl { observed: f32, limit: f32 },
    #[error(
        "Hamilton-Jacobi field '{field}' index {index} error {error} exceeds tolerance {tolerance}"
    )]
    EvolutionInvariant {
        field: &'static str,
        index: usize,
        error: f64,
        tolerance: f64,
    },
    #[error(
        "Hamilton-Jacobi backward field '{field}' index {index} error {error} exceeds tolerance {tolerance}"
    )]
    BackwardInvariant {
        field: &'static str,
        index: usize,
        error: f64,
        tolerance: f64,
    },
}

fn require_finite(field: &'static str, value: f32) -> Result<f32, HamiltonJacobiError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(HamiltonJacobiError::NonFinite { field, value })
    }
}

fn require_non_negative(field: &'static str, value: f32) -> Result<f32, HamiltonJacobiError> {
    require_finite(field, value)?;
    if value >= 0.0 {
        Ok(value)
    } else {
        Err(HamiltonJacobiError::Negative { field, value })
    }
}

fn require_positive(field: &'static str, value: f32) -> Result<f32, HamiltonJacobiError> {
    require_finite(field, value)?;
    if value > 0.0 {
        Ok(value)
    } else {
        Err(HamiltonJacobiError::NonPositive { field, value })
    }
}

fn require_derived_finite(field: &'static str, value: f32) -> Result<f32, HamiltonJacobiError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(HamiltonJacobiError::NonFiniteDerived { field, value })
    }
}

fn checked_volume(rows: usize, features: usize) -> Result<usize, HamiltonJacobiError> {
    if features == 0 {
        return Err(HamiltonJacobiError::EmptyFeatures);
    }
    rows.checked_mul(features)
        .ok_or(HamiltonJacobiError::ShapeOverflow { rows, features })
}

fn validate_length(
    field: &'static str,
    values: &[f32],
    expected: usize,
) -> Result<(), HamiltonJacobiError> {
    if values.len() != expected {
        return Err(HamiltonJacobiError::LengthMismatch {
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
pub struct HamiltonJacobiConfig {
    time_step: f32,
    mass: f32,
    lattice_spacing: f32,
    characteristic_speed_limit: f32,
    viscosity: f32,
}

impl Default for HamiltonJacobiConfig {
    fn default() -> Self {
        Self {
            time_step: 0.1,
            mass: 1.0,
            lattice_spacing: 1.0,
            characteristic_speed_limit: 2.0,
            viscosity: 0.0,
        }
    }
}

impl HamiltonJacobiConfig {
    pub fn new(time_step: f32) -> Result<Self, HamiltonJacobiError> {
        let config = Self {
            time_step,
            ..Self::default()
        };
        config.validate()?;
        Ok(config)
    }

    pub fn with_time_step(mut self, time_step: f32) -> Result<Self, HamiltonJacobiError> {
        self.time_step = time_step;
        self.validate()?;
        Ok(self)
    }

    pub fn with_mass(mut self, mass: f32) -> Result<Self, HamiltonJacobiError> {
        self.mass = mass;
        self.validate()?;
        Ok(self)
    }

    pub fn with_lattice_spacing(
        mut self,
        lattice_spacing: f32,
    ) -> Result<Self, HamiltonJacobiError> {
        self.lattice_spacing = lattice_spacing;
        self.validate()?;
        Ok(self)
    }

    pub fn with_characteristic_speed_limit(
        mut self,
        characteristic_speed_limit: f32,
    ) -> Result<Self, HamiltonJacobiError> {
        self.characteristic_speed_limit = characteristic_speed_limit;
        self.validate()?;
        Ok(self)
    }

    pub fn with_viscosity(mut self, viscosity: f32) -> Result<Self, HamiltonJacobiError> {
        self.viscosity = viscosity;
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), HamiltonJacobiError> {
        require_positive("time_step", self.time_step)?;
        require_positive("mass", self.mass)?;
        require_positive("lattice_spacing", self.lattice_spacing)?;
        require_non_negative(
            "characteristic_speed_limit",
            self.characteristic_speed_limit,
        )?;
        require_non_negative("viscosity", self.viscosity)?;
        for (field, value) in [
            ("inverse_mass", self.inverse_mass()),
            ("central_gradient_scale", self.central_gradient_scale()),
            ("laplacian_scale", self.laplacian_scale()),
            ("numerical_viscosity", self.numerical_viscosity()),
            ("effective_viscosity", self.effective_viscosity()),
            (
                "diffusion_step_coefficient",
                self.diffusion_step_coefficient(),
            ),
            ("monotone_cfl", self.monotone_cfl()),
        ] {
            require_derived_finite(field, value)?;
        }
        if self.monotone_cfl() > HAMILTON_JACOBI_MAX_MONOTONE_CFL {
            return Err(HamiltonJacobiError::MonotoneCfl {
                observed: self.monotone_cfl(),
                limit: HAMILTON_JACOBI_MAX_MONOTONE_CFL,
            });
        }
        Ok(())
    }

    pub fn time_step(&self) -> f32 {
        self.time_step
    }

    pub fn mass(&self) -> f32 {
        self.mass
    }

    pub fn inverse_mass(&self) -> f32 {
        1.0 / self.mass
    }

    pub fn lattice_spacing(&self) -> f32 {
        self.lattice_spacing
    }

    pub fn characteristic_speed_limit(&self) -> f32 {
        self.characteristic_speed_limit
    }

    pub fn viscosity(&self) -> f32 {
        self.viscosity
    }

    pub fn central_gradient_scale(&self) -> f32 {
        0.5 / self.lattice_spacing
    }

    pub fn laplacian_scale(&self) -> f32 {
        1.0 / (self.lattice_spacing * self.lattice_spacing)
    }

    pub fn kinetic_prefactor(&self) -> f32 {
        0.5 * self.inverse_mass()
    }

    pub fn numerical_viscosity(&self) -> f32 {
        0.5 * self.characteristic_speed_limit * self.lattice_spacing
    }

    pub fn effective_viscosity(&self) -> f32 {
        self.numerical_viscosity() + self.viscosity
    }

    pub fn diffusion_step_coefficient(&self) -> f32 {
        self.time_step * self.effective_viscosity() * self.laplacian_scale()
    }

    pub fn monotone_cfl(&self) -> f32 {
        2.0 * self.diffusion_step_coefficient()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct HamiltonJacobiAudit {
    pub rows: usize,
    pub features: usize,
    pub input_characteristic_speed: f32,
    pub output_characteristic_speed: f32,
    pub max_characteristic_speed: f32,
    pub characteristic_speed_limit: f32,
    pub characteristic_margin: f32,
    pub monotone_cfl: f32,
    pub cfl_limit: f32,
    pub cfl_margin: f32,
    pub numerical_viscosity: f32,
    pub physical_viscosity: f32,
    pub effective_viscosity: f32,
    pub max_abs_gradient: f32,
    pub max_abs_hamiltonian: f32,
    pub max_abs_residual: f32,
    pub input_total_variation: f64,
    pub output_total_variation: f64,
    pub input_action_mean: f64,
    pub output_action_mean: f64,
    pub max_action_error: f64,
    pub max_formula_tolerance_ratio: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct HamiltonJacobiStep {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub equation: &'static str,
    pub hamiltonian_model: &'static str,
    pub scheme: &'static str,
    pub boundary: &'static str,
    pub state: &'static str,
    pub stability: &'static str,
    pub config: HamiltonJacobiConfig,
    pub action: Vec<f32>,
    pub characteristic_velocity: Vec<f32>,
    pub hamiltonian: Vec<f32>,
    pub audit: HamiltonJacobiAudit,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HamiltonJacobiBackward {
    pub grad_action: Vec<f32>,
    pub grad_potential: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct HamiltonJacobiBackwardAudit {
    pub rows: usize,
    pub features: usize,
    pub max_grad_action_error: f64,
    pub max_grad_potential_error: f64,
    pub max_formula_tolerance_ratio: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct HamiltonJacobiAuditRequest<'a> {
    pub action: &'a [f32],
    pub potential: &'a [f32],
    pub output_action: &'a [f32],
    pub rows: usize,
    pub features: usize,
    pub config: HamiltonJacobiConfig,
}

#[derive(Clone, Copy, Debug)]
pub struct HamiltonJacobiBackwardRequest<'a> {
    pub action: &'a [f32],
    pub potential: &'a [f32],
    pub grad_output_action: &'a [f32],
    pub rows: usize,
    pub features: usize,
    pub config: HamiltonJacobiConfig,
}

#[derive(Clone, Copy, Debug)]
pub struct HamiltonJacobiBackwardAuditRequest<'a> {
    pub request: HamiltonJacobiBackwardRequest<'a>,
    pub grad_action: &'a [f32],
    pub grad_potential: &'a [f32],
}

#[derive(Clone, Copy, Debug)]
struct StateDiagnostics {
    observed_characteristic_speed: f32,
    max_abs_gradient: f32,
    total_variation: f64,
    action_mean: f64,
}

fn periodic_neighbors(index: usize, features: usize) -> (usize, usize) {
    let col = index % features;
    let row_start = index - col;
    let previous = row_start + if col == 0 { features - 1 } else { col - 1 };
    let next = row_start + if col + 1 == features { 0 } else { col + 1 };
    (previous, next)
}

fn state_diagnostics(
    action: &[f32],
    features: usize,
    config: HamiltonJacobiConfig,
) -> Result<StateDiagnostics, HamiltonJacobiError> {
    let mut observed_characteristic_speed = 0.0f32;
    let mut max_abs_gradient = 0.0f32;
    let mut total_variation = 0.0f64;
    let mut sum = 0.0f64;
    let inverse_spacing = 1.0 / config.lattice_spacing();
    let inverse_mass = config.inverse_mass();
    for (index, &value) in action.iter().enumerate() {
        sum += value as f64;
        if features > 1 {
            let (previous, next) = periodic_neighbors(index, features);
            let backward_slope = (value - action[previous]) * inverse_spacing;
            let forward_slope = (action[next] - value) * inverse_spacing;
            let central_gradient = 0.5 * (backward_slope + forward_slope);
            max_abs_gradient = max_abs_gradient.max(central_gradient.abs());
            observed_characteristic_speed = observed_characteristic_speed
                .max(backward_slope.abs().max(forward_slope.abs()) * inverse_mass);
            total_variation += (action[next] - value).abs() as f64;
        }
    }
    for (field, value) in [
        (
            "observed_characteristic_speed",
            observed_characteristic_speed,
        ),
        ("max_abs_gradient", max_abs_gradient),
    ] {
        require_derived_finite(field, value)?;
    }
    let action_mean = if action.is_empty() {
        0.0
    } else {
        sum / action.len() as f64
    };
    Ok(StateDiagnostics {
        observed_characteristic_speed,
        max_abs_gradient,
        total_variation,
        action_mean,
    })
}

fn require_characteristic_bound(
    stage: &'static str,
    diagnostics: StateDiagnostics,
    config: HamiltonJacobiConfig,
) -> Result<(), HamiltonJacobiError> {
    let limit = config.characteristic_speed_limit();
    let tolerance = CHARACTERISTIC_ERROR_FACTOR * (1.0 + limit.abs());
    if diagnostics.observed_characteristic_speed <= limit + tolerance {
        Ok(())
    } else {
        Err(HamiltonJacobiError::CharacteristicBound {
            stage,
            observed: diagnostics.observed_characteristic_speed,
            limit,
        })
    }
}

pub fn validate_hamilton_jacobi_state(
    action: &[f32],
    potential: &[f32],
    rows: usize,
    features: usize,
    config: HamiltonJacobiConfig,
) -> Result<(), HamiltonJacobiError> {
    config.validate()?;
    let volume = checked_volume(rows, features)?;
    validate_length("action", action, volume)?;
    validate_length("potential", potential, features)?;
    let diagnostics = state_diagnostics(action, features, config)?;
    require_characteristic_bound("input", diagnostics, config)
}

#[derive(Debug)]
struct EvolvedAction {
    action: Vec<f32>,
    characteristic_velocity: Vec<f32>,
    hamiltonian: Vec<f32>,
    residual: Vec<f32>,
    input_diagnostics: StateDiagnostics,
    output_diagnostics: StateDiagnostics,
}

fn evolve_hamilton_jacobi(
    action: &[f32],
    potential: &[f32],
    features: usize,
    config: HamiltonJacobiConfig,
) -> Result<EvolvedAction, HamiltonJacobiError> {
    let input_diagnostics = state_diagnostics(action, features, config)?;
    require_characteristic_bound("input", input_diagnostics, config)?;
    let mut output = vec![0.0; action.len()];
    let mut characteristic_velocity = vec![0.0; action.len()];
    let mut hamiltonian = vec![0.0; action.len()];
    let mut residual = vec![0.0; action.len()];
    let central_gradient_scale = config.central_gradient_scale();
    let laplacian_scale = config.laplacian_scale();
    let inverse_mass = config.inverse_mass();
    let kinetic_prefactor = config.kinetic_prefactor();
    let effective_viscosity = config.effective_viscosity();
    let time_step = config.time_step();
    for index in 0..action.len() {
        let col = index % features;
        let value = action[index];
        let (gradient, laplacian) = if features == 1 {
            (0.0, 0.0)
        } else {
            let (previous, next) = periodic_neighbors(index, features);
            (
                (action[next] - action[previous]) * central_gradient_scale,
                (action[previous] - 2.0 * value + action[next]) * laplacian_scale,
            )
        };
        characteristic_velocity[index] =
            require_derived_finite("characteristic_velocity", inverse_mass * gradient)?;
        hamiltonian[index] = require_derived_finite(
            "hamiltonian",
            kinetic_prefactor * gradient * gradient + potential[col],
        )?;
        residual[index] = require_derived_finite(
            "residual",
            hamiltonian[index] - effective_viscosity * laplacian,
        )?;
        output[index] =
            require_derived_finite("output_action", value - time_step * residual[index])?;
    }
    let output_diagnostics = state_diagnostics(&output, features, config)?;
    require_characteristic_bound("output", output_diagnostics, config)?;
    Ok(EvolvedAction {
        action: output,
        characteristic_velocity,
        hamiltonian,
        residual,
        input_diagnostics,
        output_diagnostics,
    })
}

fn formula_tolerance(expected: f32) -> f64 {
    FORMULA_ERROR_FACTOR * (1.0 + expected.abs() as f64)
}

fn compare_field(
    field: &'static str,
    expected: &[f32],
    observed: &[f32],
    backward: bool,
) -> Result<(f64, f64), HamiltonJacobiError> {
    let mut max_error = 0.0f64;
    let mut max_ratio = 0.0f64;
    for (index, (&expected, &observed)) in expected.iter().zip(observed).enumerate() {
        let error = (expected as f64 - observed as f64).abs();
        let tolerance = formula_tolerance(expected);
        if error > tolerance {
            return if backward {
                Err(HamiltonJacobiError::BackwardInvariant {
                    field,
                    index,
                    error,
                    tolerance,
                })
            } else {
                Err(HamiltonJacobiError::EvolutionInvariant {
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
    evolved: &EvolvedAction,
    rows: usize,
    features: usize,
    config: HamiltonJacobiConfig,
    max_action_error: f64,
    max_formula_tolerance_ratio: f64,
) -> HamiltonJacobiAudit {
    HamiltonJacobiAudit {
        rows,
        features,
        input_characteristic_speed: evolved.input_diagnostics.observed_characteristic_speed,
        output_characteristic_speed: evolved.output_diagnostics.observed_characteristic_speed,
        max_characteristic_speed: evolved
            .input_diagnostics
            .observed_characteristic_speed
            .max(evolved.output_diagnostics.observed_characteristic_speed),
        characteristic_speed_limit: config.characteristic_speed_limit(),
        characteristic_margin: config.characteristic_speed_limit()
            - evolved
                .input_diagnostics
                .observed_characteristic_speed
                .max(evolved.output_diagnostics.observed_characteristic_speed),
        monotone_cfl: config.monotone_cfl(),
        cfl_limit: HAMILTON_JACOBI_MAX_MONOTONE_CFL,
        cfl_margin: HAMILTON_JACOBI_MAX_MONOTONE_CFL - config.monotone_cfl(),
        numerical_viscosity: config.numerical_viscosity(),
        physical_viscosity: config.viscosity(),
        effective_viscosity: config.effective_viscosity(),
        max_abs_gradient: evolved.input_diagnostics.max_abs_gradient,
        max_abs_hamiltonian: evolved
            .hamiltonian
            .iter()
            .fold(0.0f32, |maximum, value| maximum.max(value.abs())),
        max_abs_residual: evolved
            .residual
            .iter()
            .fold(0.0f32, |maximum, value| maximum.max(value.abs())),
        input_total_variation: evolved.input_diagnostics.total_variation,
        output_total_variation: evolved.output_diagnostics.total_variation,
        input_action_mean: evolved.input_diagnostics.action_mean,
        output_action_mean: evolved.output_diagnostics.action_mean,
        max_action_error,
        max_formula_tolerance_ratio,
    }
}

pub fn audit_hamilton_jacobi_step(
    request: HamiltonJacobiAuditRequest<'_>,
) -> Result<HamiltonJacobiAudit, HamiltonJacobiError> {
    validate_hamilton_jacobi_state(
        request.action,
        request.potential,
        request.rows,
        request.features,
        request.config,
    )?;
    let volume = checked_volume(request.rows, request.features)?;
    validate_length("output_action", request.output_action, volume)?;
    let expected = evolve_hamilton_jacobi(
        request.action,
        request.potential,
        request.features,
        request.config,
    )?;
    let (max_action_error, ratio) = compare_field(
        "output_action",
        &expected.action,
        request.output_action,
        false,
    )?;
    Ok(audit_from_evolved(
        &expected,
        request.rows,
        request.features,
        request.config,
        max_action_error,
        ratio,
    ))
}

pub fn apply_hamilton_jacobi_step(
    action: &[f32],
    potential: &[f32],
    rows: usize,
    features: usize,
    config: HamiltonJacobiConfig,
) -> Result<HamiltonJacobiStep, HamiltonJacobiError> {
    validate_hamilton_jacobi_state(action, potential, rows, features, config)?;
    let evolved = evolve_hamilton_jacobi(action, potential, features, config)?;
    let audit = audit_from_evolved(&evolved, rows, features, config, 0.0, 0.0);
    Ok(HamiltonJacobiStep {
        kind: HAMILTON_JACOBI_KIND,
        contract_version: HAMILTON_JACOBI_CONTRACT_VERSION,
        semantic_owner: HAMILTON_JACOBI_SEMANTIC_OWNER,
        semantic_backend: HAMILTON_JACOBI_SEMANTIC_BACKEND,
        equation: HAMILTON_JACOBI_EQUATION,
        hamiltonian_model: HAMILTON_JACOBI_HAMILTONIAN,
        scheme: HAMILTON_JACOBI_SCHEME,
        boundary: HAMILTON_JACOBI_BOUNDARY,
        state: HAMILTON_JACOBI_STATE,
        stability: HAMILTON_JACOBI_STABILITY,
        config,
        action: evolved.action,
        characteristic_velocity: evolved.characteristic_velocity,
        hamiltonian: evolved.hamiltonian,
        audit,
    })
}

pub fn backward_hamilton_jacobi_step(
    request: HamiltonJacobiBackwardRequest<'_>,
) -> Result<HamiltonJacobiBackward, HamiltonJacobiError> {
    validate_hamilton_jacobi_state(
        request.action,
        request.potential,
        request.rows,
        request.features,
        request.config,
    )?;
    let volume = checked_volume(request.rows, request.features)?;
    validate_length("grad_output_action", request.grad_output_action, volume)?;
    let mut grad_action = vec![0.0; volume];
    let mut grad_potential = vec![0.0; request.features];
    let time_step = request.config.time_step();
    let gradient_scale = request.config.central_gradient_scale();
    let inverse_mass = request.config.inverse_mass();
    let diffusion = request.config.diffusion_step_coefficient();
    for index in 0..volume {
        let col = index % request.features;
        let upstream = request.grad_output_action[index];
        grad_action[index] += upstream;
        grad_potential[col] += -time_step * upstream;
        if request.features > 1 {
            let (previous, next) = periodic_neighbors(index, request.features);
            let gradient = (request.action[next] - request.action[previous]) * gradient_scale;
            let kinetic_adjoint = time_step * inverse_mass * gradient * gradient_scale * upstream;
            grad_action[previous] += kinetic_adjoint + diffusion * upstream;
            grad_action[index] += -2.0 * diffusion * upstream;
            grad_action[next] += -kinetic_adjoint + diffusion * upstream;
        }
    }
    for (field, values) in [
        ("grad_action", grad_action.as_slice()),
        ("grad_potential", grad_potential.as_slice()),
    ] {
        for &value in values {
            require_derived_finite(field, value)?;
        }
    }
    Ok(HamiltonJacobiBackward {
        grad_action,
        grad_potential,
    })
}

pub fn audit_hamilton_jacobi_backward(
    request: HamiltonJacobiBackwardAuditRequest<'_>,
) -> Result<HamiltonJacobiBackwardAudit, HamiltonJacobiError> {
    let expected = backward_hamilton_jacobi_step(request.request)?;
    let volume = checked_volume(request.request.rows, request.request.features)?;
    validate_length("grad_action", request.grad_action, volume)?;
    validate_length(
        "grad_potential",
        request.grad_potential,
        request.request.features,
    )?;
    let (max_grad_action_error, action_ratio) = compare_field(
        "grad_action",
        &expected.grad_action,
        request.grad_action,
        true,
    )?;
    let (max_grad_potential_error, potential_ratio) = compare_field(
        "grad_potential",
        &expected.grad_potential,
        request.grad_potential,
        true,
    )?;
    Ok(HamiltonJacobiBackwardAudit {
        rows: request.request.rows,
        features: request.request.features,
        max_grad_action_error,
        max_grad_potential_error,
        max_formula_tolerance_ratio: action_ratio.max(potential_ratio),
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

    fn loss(step: &HamiltonJacobiStep, gradient: &[f32]) -> f32 {
        step.action
            .iter()
            .zip(gradient)
            .map(|(value, gradient)| value * gradient)
            .sum()
    }

    #[test]
    fn constant_action_obeys_static_potential_and_metadata() {
        let config = HamiltonJacobiConfig::new(0.1).unwrap();
        let step =
            apply_hamilton_jacobi_step(&[0.4, 0.4, 0.4], &[0.0, 0.2, -0.1], 1, 3, config).unwrap();
        assert_eq!(step.contract_version, HAMILTON_JACOBI_CONTRACT_VERSION);
        assert_eq!(step.semantic_owner, HAMILTON_JACOBI_SEMANTIC_OWNER);
        assert_eq!(step.scheme, HAMILTON_JACOBI_SCHEME);
        assert_eq!(step.boundary, HAMILTON_JACOBI_BOUNDARY);
        assert_close(step.action[0], 0.4, 1.0e-7);
        assert_close(step.action[1], 0.38, 1.0e-7);
        assert_close(step.action[2], 0.41, 1.0e-7);
        assert!(step
            .characteristic_velocity
            .iter()
            .all(|value| *value == 0.0));
    }

    #[test]
    fn monotone_scheme_preserves_componentwise_order() {
        let config = HamiltonJacobiConfig::new(0.08)
            .unwrap()
            .with_characteristic_speed_limit(2.5)
            .unwrap();
        let lower = [-0.3, 0.1, 0.35, -0.05, 0.2];
        let upper = [-0.28, 0.14, 0.38, 0.01, 0.24];
        let potential = [0.02, -0.03, 0.01, 0.04, -0.02];
        let lower_step = apply_hamilton_jacobi_step(&lower, &potential, 1, 5, config).unwrap();
        let upper_step = apply_hamilton_jacobi_step(&upper, &potential, 1, 5, config).unwrap();
        for (&lower, &upper) in lower_step.action.iter().zip(&upper_step.action) {
            assert!(lower <= upper + 1.0e-7, "lower={lower} upper={upper}");
        }
    }

    #[test]
    fn characteristic_and_cfl_guards_fail_closed() {
        let cfl = HamiltonJacobiConfig::new(0.5).unwrap_err();
        assert!(matches!(cfl, HamiltonJacobiError::MonotoneCfl { .. }));

        let config = HamiltonJacobiConfig::new(0.1)
            .unwrap()
            .with_characteristic_speed_limit(0.5)
            .unwrap();
        let error =
            apply_hamilton_jacobi_step(&[0.0, 1.0, -1.0], &[0.0; 3], 1, 3, config).unwrap_err();
        assert!(matches!(
            error,
            HamiltonJacobiError::CharacteristicBound { .. }
        ));

        let output_error =
            apply_hamilton_jacobi_step(&[0.0, 0.0, 0.0], &[0.0, 10.0, 0.0], 1, 3, config)
                .unwrap_err();
        assert!(matches!(
            output_error,
            HamiltonJacobiError::CharacteristicBound {
                stage: "output",
                ..
            }
        ));
    }

    #[test]
    fn coupled_cfl_parameters_can_reach_a_valid_non_default_regime() {
        let config = HamiltonJacobiConfig::default()
            .with_characteristic_speed_limit(0.5)
            .unwrap()
            .with_time_step(0.5)
            .unwrap();
        assert_close(config.time_step(), 0.5, f32::EPSILON);
        assert_close(config.characteristic_speed_limit(), 0.5, f32::EPSILON);
        assert!(config.monotone_cfl() <= HAMILTON_JACOBI_MAX_MONOTONE_CFL);
    }

    #[test]
    fn analytic_discrete_adjoint_matches_central_difference() {
        let config = HamiltonJacobiConfig::new(0.06)
            .unwrap()
            .with_mass(1.4)
            .unwrap()
            .with_lattice_spacing(0.8)
            .unwrap()
            .with_characteristic_speed_limit(1.8)
            .unwrap()
            .with_viscosity(0.03)
            .unwrap();
        let action = vec![0.18, -0.22, 0.31, 0.05];
        let potential = vec![0.03, -0.02, 0.04, 0.01];
        let grad_output = vec![0.4, -0.25, 0.18, 0.09];
        let backward = backward_hamilton_jacobi_step(HamiltonJacobiBackwardRequest {
            action: &action,
            potential: &potential,
            grad_output_action: &grad_output,
            rows: 1,
            features: 4,
            config,
        })
        .unwrap();
        let epsilon = 5.0e-4;
        for (kind, values, expected) in [
            ("action", action.as_slice(), backward.grad_action.as_slice()),
            (
                "potential",
                potential.as_slice(),
                backward.grad_potential.as_slice(),
            ),
        ] {
            for index in 0..values.len() {
                let mut plus_action = action.clone();
                let mut minus_action = action.clone();
                let mut plus_potential = potential.clone();
                let mut minus_potential = potential.clone();
                if kind == "action" {
                    plus_action[index] += epsilon;
                    minus_action[index] -= epsilon;
                } else {
                    plus_potential[index] += epsilon;
                    minus_potential[index] -= epsilon;
                }
                let plus = apply_hamilton_jacobi_step(&plus_action, &plus_potential, 1, 4, config)
                    .unwrap();
                let minus =
                    apply_hamilton_jacobi_step(&minus_action, &minus_potential, 1, 4, config)
                        .unwrap();
                let numerical =
                    (loss(&plus, &grad_output) - loss(&minus, &grad_output)) / (2.0 * epsilon);
                assert_close(numerical, expected[index], 2.0e-4);
            }
        }
    }

    #[test]
    fn long_characteristic_trajectory_remains_finite_and_maximum_stable() {
        let config = HamiltonJacobiConfig::new(0.02)
            .unwrap()
            .with_characteristic_speed_limit(2.0)
            .unwrap();
        let potential = [0.0; 7];
        let mut action = vec![0.4, -0.2, 0.1, 0.35, -0.15, 0.05, 0.22];
        let initial_max = action.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for _ in 0..1_000 {
            let step = apply_hamilton_jacobi_step(&action, &potential, 1, 7, config).unwrap();
            assert!(step.action.iter().all(|value| value.is_finite()));
            action = step.action;
        }
        let final_max = action.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(final_max <= initial_max + 1.0e-6);
    }

    #[test]
    fn physical_viscosity_reduces_cusp_variation() {
        let base = HamiltonJacobiConfig::new(0.04)
            .unwrap()
            .with_characteristic_speed_limit(2.0)
            .unwrap();
        let viscous = base.with_viscosity(0.2).unwrap();
        let action = [0.0, 0.0, 1.0, 0.0, 0.0];
        let potential = [0.0; 5];
        let inviscid_step = apply_hamilton_jacobi_step(&action, &potential, 1, 5, base).unwrap();
        let viscous_step = apply_hamilton_jacobi_step(&action, &potential, 1, 5, viscous).unwrap();
        assert!(
            viscous_step.audit.output_total_variation < inviscid_step.audit.output_total_variation
        );
    }

    #[test]
    fn external_forward_and_backward_audits_reject_drift() {
        let config = HamiltonJacobiConfig::new(0.1).unwrap();
        let action = [0.1, -0.2, 0.3];
        let potential = [0.02, 0.01, -0.03];
        let step = apply_hamilton_jacobi_step(&action, &potential, 1, 3, config).unwrap();
        let mut drifted_action = step.action.clone();
        drifted_action[1] += 1.0e-2;
        let forward_error = audit_hamilton_jacobi_step(HamiltonJacobiAuditRequest {
            action: &action,
            potential: &potential,
            output_action: &drifted_action,
            rows: 1,
            features: 3,
            config,
        })
        .unwrap_err();
        assert!(matches!(
            forward_error,
            HamiltonJacobiError::EvolutionInvariant { .. }
        ));

        let grad_output = [0.2, -0.1, 0.4];
        let request = HamiltonJacobiBackwardRequest {
            action: &action,
            potential: &potential,
            grad_output_action: &grad_output,
            rows: 1,
            features: 3,
            config,
        };
        let backward = backward_hamilton_jacobi_step(request).unwrap();
        let mut drifted_gradient = backward.grad_action.clone();
        drifted_gradient[0] -= 1.0e-2;
        let backward_error = audit_hamilton_jacobi_backward(HamiltonJacobiBackwardAuditRequest {
            request,
            grad_action: &drifted_gradient,
            grad_potential: &backward.grad_potential,
        })
        .unwrap_err();
        assert!(matches!(
            backward_error,
            HamiltonJacobiError::BackwardInvariant { .. }
        ));
    }

    #[test]
    fn empty_batch_is_a_valid_audited_transition() {
        let config = HamiltonJacobiConfig::new(0.1).unwrap();
        let step = apply_hamilton_jacobi_step(&[], &[0.1, -0.2], 0, 2, config).unwrap();
        assert!(step.action.is_empty());
        assert_eq!(step.audit.rows, 0);
        assert_eq!(step.audit.input_total_variation, 0.0);
        let backward = backward_hamilton_jacobi_step(HamiltonJacobiBackwardRequest {
            action: &[],
            potential: &[0.1, -0.2],
            grad_output_action: &[],
            rows: 0,
            features: 2,
            config,
        })
        .unwrap();
        assert!(backward.grad_action.is_empty());
        assert_eq!(backward.grad_potential, vec![0.0, 0.0]);
    }
}
