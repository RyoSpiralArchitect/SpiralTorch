// SPDX-License-Identifier: AGPL-3.0-or-later

//! Canonical positive-amplitude imaginary-time Schrodinger evolution.
//!
//! The Hamiltonian is the sum of an undirected weighted graph Laplacian and a
//! diagonal scalar potential. Evolution uses a positivity-preserving Euler
//! approximation to `exp(-tau H)` in the log domain. Language bindings may
//! transport requests and payloads, but must not reconstruct this evolution.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use thiserror::Error;

pub const ZSPACE_IMAGINARY_TIME_SCHRODINGER_CONTRACT_VERSION: &str =
    "spiraltorch.zspace_imaginary_time_schrodinger.v1";
pub const ZSPACE_IMAGINARY_TIME_SCHRODINGER_KIND: &str =
    "spiraltorch.zspace_imaginary_time_schrodinger";
pub const ZSPACE_IMAGINARY_TIME_SCHRODINGER_SEMANTIC_OWNER: &str =
    "st-core::inference::imaginary_time_schrodinger";
pub const ZSPACE_IMAGINARY_TIME_SCHRODINGER_SEMANTIC_BACKEND: &str = "rust";
pub const ZSPACE_IMAGINARY_TIME_SCHRODINGER_BACKEND: &str =
    "spiraltorch_imaginary_time_schrodinger_core";
pub const ZSPACE_IMAGINARY_TIME_SCHRODINGER_EXECUTION_BACKEND: &str = "f64_cpu";
pub const ZSPACE_IMAGINARY_TIME_SCHRODINGER_ROUTE_BLOCKER: &str = "f64_sparse_graph_state";

const INVARIANT_TOLERANCE: f64 = 1.0e-10;
const MAX_CFL_LIMIT: f64 = 1.0 - f64::EPSILON;

type WeightedAdjacency = Vec<Vec<(usize, f64)>>;

#[derive(Debug, Error, PartialEq)]
pub enum ImaginaryTimeSchrodingerError {
    #[error("imaginary-time Schrodinger potential must not be empty")]
    EmptyState,
    #[error("imaginary-time Schrodinger field '{field}' has length {actual}, expected {expected}")]
    DimensionMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("imaginary-time Schrodinger tag at index {index} must not be empty")]
    EmptyTag { index: usize },
    #[error("imaginary-time Schrodinger tag '{tag}' is duplicated")]
    DuplicateTag { tag: String },
    #[error("imaginary-time Schrodinger field '{field}' must be finite, got {value}")]
    NonFinite { field: &'static str, value: f64 },
    #[error("imaginary-time Schrodinger field '{field}' must be non-negative, got {value}")]
    Negative { field: &'static str, value: f64 },
    #[error("imaginary-time Schrodinger field '{field}' must be positive, got {value}")]
    NonPositive { field: &'static str, value: f64 },
    #[error("imaginary-time Schrodinger field '{field}' must be in [{min}, {max}], got {value}")]
    OutOfRange {
        field: &'static str,
        value: f64,
        min: f64,
        max: f64,
    },
    #[error("imaginary-time Schrodinger max_substeps must be positive")]
    EmptyStepBudget,
    #[error(
        "imaginary-time Schrodinger edge {edge} endpoint {endpoint} is outside dimension {dimension}"
    )]
    EdgeOutOfRange {
        edge: usize,
        endpoint: usize,
        dimension: usize,
    },
    #[error("imaginary-time Schrodinger edge {edge} must connect distinct nodes")]
    SelfEdge { edge: usize },
    #[error(
        "imaginary-time Schrodinger edge {edge} must be canonical with left < right, got ({left}, {right})"
    )]
    NonCanonicalEdge {
        edge: usize,
        left: usize,
        right: usize,
    },
    #[error("imaginary-time Schrodinger edge ({left}, {right}) is duplicated")]
    DuplicateEdge { left: usize, right: usize },
    #[error(
        "imaginary-time Schrodinger needs at least {required_at_least} substeps, exceeding configured maximum {maximum}"
    )]
    StepBudgetExceeded {
        required_at_least: f64,
        maximum: usize,
    },
    #[error("derived imaginary-time Schrodinger field '{field}' must be finite, got {value}")]
    NonFiniteDerived { field: &'static str, value: f64 },
    #[error(
        "imaginary-time Schrodinger Rayleigh energy increased: before={before}, after={after}"
    )]
    EnergyIncreased { before: f64, after: f64 },
    #[error(
        "imaginary-time Schrodinger probability mass must be within {tolerance} of 1, got {sum}"
    )]
    InvalidProbabilityMass { sum: f64, tolerance: f64 },
    #[error(
        "imaginary-time Schrodinger amplitude L2 norm for '{field}' must be within {tolerance} of 1, got {norm}"
    )]
    InvalidL2Norm {
        field: &'static str,
        norm: f64,
        tolerance: f64,
    },
}

fn require_finite(field: &'static str, value: f64) -> Result<f64, ImaginaryTimeSchrodingerError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ImaginaryTimeSchrodingerError::NonFinite { field, value })
    }
}

fn require_non_negative(
    field: &'static str,
    value: f64,
) -> Result<f64, ImaginaryTimeSchrodingerError> {
    require_finite(field, value)?;
    if value >= 0.0 {
        Ok(value)
    } else {
        Err(ImaginaryTimeSchrodingerError::Negative { field, value })
    }
}

fn require_positive(field: &'static str, value: f64) -> Result<f64, ImaginaryTimeSchrodingerError> {
    require_finite(field, value)?;
    if value > 0.0 {
        Ok(value)
    } else {
        Err(ImaginaryTimeSchrodingerError::NonPositive { field, value })
    }
}

fn require_derived_finite(
    field: &'static str,
    value: f64,
) -> Result<f64, ImaginaryTimeSchrodingerError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ImaginaryTimeSchrodingerError::NonFiniteDerived { field, value })
    }
}

fn checked_add(
    field: &'static str,
    left: f64,
    right: f64,
) -> Result<f64, ImaginaryTimeSchrodingerError> {
    require_derived_finite(field, left + right)
}

fn checked_mul(
    field: &'static str,
    left: f64,
    right: f64,
) -> Result<f64, ImaginaryTimeSchrodingerError> {
    require_derived_finite(field, left * right)
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ImaginaryTimeSchrodingerConfig {
    imaginary_time: f64,
    cfl_limit: f64,
    max_substeps: usize,
}

impl Default for ImaginaryTimeSchrodingerConfig {
    fn default() -> Self {
        Self {
            imaginary_time: 1.0,
            cfl_limit: 0.9,
            max_substeps: 16_384,
        }
    }
}

impl ImaginaryTimeSchrodingerConfig {
    pub fn new(imaginary_time: f64) -> Result<Self, ImaginaryTimeSchrodingerError> {
        let config = Self {
            imaginary_time,
            ..Self::default()
        };
        config.validate()?;
        Ok(config)
    }

    pub fn with_cfl_limit(mut self, cfl_limit: f64) -> Result<Self, ImaginaryTimeSchrodingerError> {
        self.cfl_limit = cfl_limit;
        self.validate()?;
        Ok(self)
    }

    pub fn with_max_substeps(
        mut self,
        max_substeps: usize,
    ) -> Result<Self, ImaginaryTimeSchrodingerError> {
        self.max_substeps = max_substeps;
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), ImaginaryTimeSchrodingerError> {
        require_non_negative("imaginary_time", self.imaginary_time)?;
        require_finite("cfl_limit", self.cfl_limit)?;
        if !(0.0..=MAX_CFL_LIMIT).contains(&self.cfl_limit) || self.cfl_limit == 0.0 {
            return Err(ImaginaryTimeSchrodingerError::OutOfRange {
                field: "cfl_limit",
                value: self.cfl_limit,
                min: f64::MIN_POSITIVE,
                max: MAX_CFL_LIMIT,
            });
        }
        if self.max_substeps == 0 {
            return Err(ImaginaryTimeSchrodingerError::EmptyStepBudget);
        }
        Ok(())
    }

    pub fn imaginary_time(&self) -> f64 {
        self.imaginary_time
    }

    pub fn cfl_limit(&self) -> f64 {
        self.cfl_limit
    }

    pub fn max_substeps(&self) -> usize {
        self.max_substeps
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImaginaryTimeSchrodingerEdge {
    pub left: usize,
    pub right: usize,
    pub weight: f64,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ImaginaryTimeSchrodingerRequest {
    pub tags: Vec<String>,
    pub potential: Vec<f64>,
    pub edges: Vec<ImaginaryTimeSchrodingerEdge>,
    pub initial_amplitude: Vec<f64>,
    pub config: ImaginaryTimeSchrodingerConfig,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ImaginaryTimeSchrodingerEffects {
    pub evolution_applied: bool,
    pub edge_count: usize,
    pub substeps: usize,
    pub substep_imaginary_time: f64,
    pub potential_shift: f64,
    pub max_degree: f64,
    pub spectral_upper_bound: f64,
    pub initial_rayleigh_energy: f64,
    pub final_rayleigh_energy: f64,
    pub rayleigh_energy_drop: f64,
    pub energy_tolerance: f64,
    pub initial_l2_norm: f64,
    pub final_l2_norm: f64,
    pub l2_norm_tolerance: f64,
    pub initial_residual_l2: f64,
    pub final_residual_l2: f64,
    pub accumulated_log_norm: f64,
    pub probability_entropy: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ImaginaryTimeSchrodingerPayload {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub backend: &'static str,
    pub execution_backend: &'static str,
    pub route_blocker: &'static str,
    pub config: ImaginaryTimeSchrodingerConfig,
    pub tags: Vec<String>,
    pub potential: Vec<f64>,
    pub shifted_potential: Vec<f64>,
    pub edges: Vec<ImaginaryTimeSchrodingerEdge>,
    pub initial_amplitude: Vec<f64>,
    pub final_amplitude: Vec<f64>,
    pub probability: Vec<f64>,
    pub log_amplitude_boost: Vec<f64>,
    pub probability_sum: f64,
    pub probability_sum_tolerance: f64,
    pub effects: ImaginaryTimeSchrodingerEffects,
}

fn validate_tags(tags: &[String], dimension: usize) -> Result<(), ImaginaryTimeSchrodingerError> {
    if tags.len() != dimension {
        return Err(ImaginaryTimeSchrodingerError::DimensionMismatch {
            field: "tags",
            expected: dimension,
            actual: tags.len(),
        });
    }
    let mut seen = HashSet::with_capacity(tags.len());
    for (index, tag) in tags.iter().enumerate() {
        if tag.trim().is_empty() {
            return Err(ImaginaryTimeSchrodingerError::EmptyTag { index });
        }
        if !seen.insert(tag.as_str()) {
            return Err(ImaginaryTimeSchrodingerError::DuplicateTag { tag: tag.clone() });
        }
    }
    Ok(())
}

fn validate_edges(
    edges: &[ImaginaryTimeSchrodingerEdge],
    dimension: usize,
) -> Result<(WeightedAdjacency, Vec<f64>), ImaginaryTimeSchrodingerError> {
    let mut adjacency = vec![Vec::new(); dimension];
    let mut degree = vec![0.0; dimension];
    let mut seen = HashSet::with_capacity(edges.len());
    for (edge_index, edge) in edges.iter().enumerate() {
        if edge.left >= dimension {
            return Err(ImaginaryTimeSchrodingerError::EdgeOutOfRange {
                edge: edge_index,
                endpoint: edge.left,
                dimension,
            });
        }
        if edge.right >= dimension {
            return Err(ImaginaryTimeSchrodingerError::EdgeOutOfRange {
                edge: edge_index,
                endpoint: edge.right,
                dimension,
            });
        }
        if edge.left == edge.right {
            return Err(ImaginaryTimeSchrodingerError::SelfEdge { edge: edge_index });
        }
        if edge.left > edge.right {
            return Err(ImaginaryTimeSchrodingerError::NonCanonicalEdge {
                edge: edge_index,
                left: edge.left,
                right: edge.right,
            });
        }
        require_positive("edge_weight", edge.weight)?;
        if !seen.insert((edge.left, edge.right)) {
            return Err(ImaginaryTimeSchrodingerError::DuplicateEdge {
                left: edge.left,
                right: edge.right,
            });
        }
        degree[edge.left] = checked_add("graph_degree", degree[edge.left], edge.weight)?;
        degree[edge.right] = checked_add("graph_degree", degree[edge.right], edge.weight)?;
        adjacency[edge.left].push((edge.right, edge.weight));
        adjacency[edge.right].push((edge.left, edge.weight));
    }
    Ok((adjacency, degree))
}

fn log_sum_exp(
    values: impl IntoIterator<Item = f64>,
) -> Result<f64, ImaginaryTimeSchrodingerError> {
    let values = values.into_iter().collect::<Vec<_>>();
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    require_derived_finite("log_sum_exp_maximum", maximum)?;
    let mut sum = 0.0;
    for value in values {
        sum = checked_add("log_sum_exp_sum", sum, (value - maximum).exp())?;
    }
    require_positive("log_sum_exp_sum", sum)?;
    require_derived_finite("log_sum_exp", maximum + sum.ln())
}

fn normalize_log_l2(log_amplitude: &mut [f64]) -> Result<f64, ImaginaryTimeSchrodingerError> {
    let log_norm = 0.5 * log_sum_exp(log_amplitude.iter().map(|value| 2.0 * value))?;
    for value in log_amplitude {
        *value = require_derived_finite("normalized_log_amplitude", *value - log_norm)?;
    }
    Ok(log_norm)
}

fn amplitudes_from_log(log_amplitude: &[f64]) -> Vec<f64> {
    log_amplitude.iter().map(|value| value.exp()).collect()
}

fn audited_l2_norm(
    field: &'static str,
    amplitude: &[f64],
    tolerance: f64,
) -> Result<f64, ImaginaryTimeSchrodingerError> {
    let squared = amplitude.iter().try_fold(0.0, |sum, &value| {
        checked_add(
            "amplitude_l2_squared",
            sum,
            checked_mul("amplitude_l2_term", value, value)?,
        )
    })?;
    let norm = require_derived_finite("amplitude_l2_norm", squared.sqrt())?;
    if (norm - 1.0).abs() > tolerance {
        return Err(ImaginaryTimeSchrodingerError::InvalidL2Norm {
            field,
            norm,
            tolerance,
        });
    }
    Ok(norm)
}

fn hamiltonian_apply(
    amplitude: &[f64],
    shifted_potential: &[f64],
    adjacency: &[Vec<(usize, f64)>],
) -> Result<Vec<f64>, ImaginaryTimeSchrodingerError> {
    let mut output = vec![0.0; amplitude.len()];
    for index in 0..amplitude.len() {
        let mut value = checked_mul(
            "hamiltonian_potential",
            shifted_potential[index],
            amplitude[index],
        )?;
        for &(neighbor, weight) in &adjacency[index] {
            let difference = require_derived_finite(
                "hamiltonian_difference",
                amplitude[index] - amplitude[neighbor],
            )?;
            value = checked_add(
                "hamiltonian_value",
                value,
                checked_mul("hamiltonian_edge", weight, difference)?,
            )?;
        }
        output[index] = value;
    }
    Ok(output)
}

fn rayleigh_energy(
    amplitude: &[f64],
    shifted_potential: &[f64],
    adjacency: &[Vec<(usize, f64)>],
) -> Result<f64, ImaginaryTimeSchrodingerError> {
    let applied = hamiltonian_apply(amplitude, shifted_potential, adjacency)?;
    amplitude
        .iter()
        .zip(applied)
        .try_fold(0.0, |sum, (&left, right)| {
            checked_add(
                "rayleigh_energy",
                sum,
                checked_mul("rayleigh_term", left, right)?,
            )
        })
}

fn residual_l2(
    amplitude: &[f64],
    shifted_potential: &[f64],
    adjacency: &[Vec<(usize, f64)>],
    energy: f64,
) -> Result<f64, ImaginaryTimeSchrodingerError> {
    let applied = hamiltonian_apply(amplitude, shifted_potential, adjacency)?;
    let squared = amplitude
        .iter()
        .zip(applied)
        .try_fold(0.0, |sum, (&amplitude, applied)| {
            let residual =
                require_derived_finite("hamiltonian_residual", applied - energy * amplitude)?;
            checked_add(
                "hamiltonian_residual_squared",
                sum,
                checked_mul("hamiltonian_residual_term", residual, residual)?,
            )
        })?;
    require_derived_finite("hamiltonian_residual_l2", squared.sqrt())
}

fn probability_from_log(
    log_amplitude: &[f64],
) -> Result<(Vec<f64>, f64, f64), ImaginaryTimeSchrodingerError> {
    let mut probability = log_amplitude
        .iter()
        .map(|value| (2.0 * value).exp())
        .collect::<Vec<_>>();
    let sum = probability.iter().try_fold(0.0, |sum, &value| {
        checked_add("probability_sum", sum, value)
    })?;
    require_positive("probability_sum", sum)?;
    for value in &mut probability {
        *value = require_derived_finite("normalized_probability", *value / sum)?;
    }
    let normalized_sum = probability.iter().try_fold(0.0, |sum, &value| {
        checked_add("normalized_probability_sum", sum, value)
    })?;
    let tolerance = INVARIANT_TOLERANCE * (1.0 + probability.len() as f64);
    if (normalized_sum - 1.0).abs() > tolerance {
        return Err(ImaginaryTimeSchrodingerError::InvalidProbabilityMass {
            sum: normalized_sum,
            tolerance,
        });
    }
    Ok((probability, normalized_sum, tolerance))
}

fn entropy(probability: &[f64]) -> Result<f64, ImaginaryTimeSchrodingerError> {
    probability.iter().try_fold(0.0, |sum, &value| {
        if value == 0.0 {
            Ok(sum)
        } else {
            checked_add("probability_entropy", sum, -value * value.ln())
        }
    })
}

/// Evolves a positive amplitude under a graph Hamiltonian in imaginary time.
pub fn apply_imaginary_time_schrodinger(
    request: ImaginaryTimeSchrodingerRequest,
) -> Result<ImaginaryTimeSchrodingerPayload, ImaginaryTimeSchrodingerError> {
    let ImaginaryTimeSchrodingerRequest {
        tags,
        potential,
        edges,
        initial_amplitude,
        config,
    } = request;
    config.validate()?;
    if potential.is_empty() {
        return Err(ImaginaryTimeSchrodingerError::EmptyState);
    }
    let dimension = potential.len();
    validate_tags(&tags, dimension)?;
    for &value in &potential {
        require_finite("potential", value)?;
    }
    let (adjacency, degree) = validate_edges(&edges, dimension)?;

    let potential_shift = potential.iter().copied().fold(f64::INFINITY, f64::min);
    require_derived_finite("potential_shift", potential_shift)?;
    let shifted_potential = potential
        .iter()
        .map(|value| require_derived_finite("shifted_potential", value - potential_shift))
        .collect::<Result<Vec<_>, _>>()?;

    let initial_amplitude = if initial_amplitude.is_empty() {
        vec![1.0; dimension]
    } else {
        if initial_amplitude.len() != dimension {
            return Err(ImaginaryTimeSchrodingerError::DimensionMismatch {
                field: "initial_amplitude",
                expected: dimension,
                actual: initial_amplitude.len(),
            });
        }
        initial_amplitude
    };
    for &value in &initial_amplitude {
        require_positive("initial_amplitude", value)?;
    }
    let mut initial_log_amplitude = initial_amplitude
        .iter()
        .map(|value| value.ln())
        .collect::<Vec<_>>();
    normalize_log_l2(&mut initial_log_amplitude)?;
    let normalized_initial_amplitude = amplitudes_from_log(&initial_log_amplitude);
    let l2_norm_tolerance = INVARIANT_TOLERANCE * (1.0 + (dimension as f64).sqrt());
    let initial_l2_norm = audited_l2_norm(
        "initial_amplitude",
        &normalized_initial_amplitude,
        l2_norm_tolerance,
    )?;

    let max_degree = degree.iter().copied().fold(0.0, f64::max);
    let spectral_upper_bound = degree.iter().zip(&shifted_potential).try_fold(
        0.0f64,
        |bound, (&degree, &potential)| {
            Ok(bound.max(checked_add(
                "spectral_upper_bound",
                checked_mul("spectral_degree_bound", 2.0, degree)?,
                potential,
            )?))
        },
    )?;
    let required_substeps = if config.imaginary_time == 0.0 || spectral_upper_bound == 0.0 {
        0.0
    } else {
        require_derived_finite(
            "required_substeps",
            config.imaginary_time * spectral_upper_bound / config.cfl_limit,
        )?
        .ceil()
        .max(1.0)
    };
    if required_substeps > config.max_substeps as f64 {
        return Err(ImaginaryTimeSchrodingerError::StepBudgetExceeded {
            required_at_least: required_substeps,
            maximum: config.max_substeps,
        });
    }
    let substeps = required_substeps as usize;
    let substep_imaginary_time = if substeps == 0 {
        0.0
    } else {
        require_derived_finite(
            "substep_imaginary_time",
            config.imaginary_time / substeps as f64,
        )?
    };

    let initial_rayleigh_energy = rayleigh_energy(
        &normalized_initial_amplitude,
        &shifted_potential,
        &adjacency,
    )?;
    let initial_residual_l2 = residual_l2(
        &normalized_initial_amplitude,
        &shifted_potential,
        &adjacency,
        initial_rayleigh_energy,
    )?;

    let mut log_amplitude = initial_log_amplitude;
    let mut accumulated_log_norm = 0.0;
    for _ in 0..substeps {
        let mut next = Vec::with_capacity(dimension);
        for index in 0..dimension {
            let exit_rate = checked_add("exit_rate", degree[index], shifted_potential[index])?;
            let self_coefficient = require_derived_finite(
                "self_coefficient",
                1.0 - substep_imaginary_time * exit_rate,
            )?;
            require_positive("self_coefficient", self_coefficient)?;
            let mut terms = Vec::with_capacity(adjacency[index].len() + 1);
            terms.push(self_coefficient.ln() + log_amplitude[index]);
            for &(neighbor, weight) in &adjacency[index] {
                let coefficient = checked_mul(
                    "edge_transition_coefficient",
                    substep_imaginary_time,
                    weight,
                )?;
                if coefficient > 0.0 {
                    terms.push(coefficient.ln() + log_amplitude[neighbor]);
                }
            }
            next.push(log_sum_exp(terms)?);
        }
        accumulated_log_norm = checked_add(
            "accumulated_log_norm",
            accumulated_log_norm,
            normalize_log_l2(&mut next)?,
        )?;
        log_amplitude = next;
    }

    let final_amplitude = amplitudes_from_log(&log_amplitude);
    let final_l2_norm = audited_l2_norm("final_amplitude", &final_amplitude, l2_norm_tolerance)?;
    let final_rayleigh_energy = rayleigh_energy(&final_amplitude, &shifted_potential, &adjacency)?;
    let energy_tolerance = INVARIANT_TOLERANCE * (1.0 + initial_rayleigh_energy.abs());
    if final_rayleigh_energy > initial_rayleigh_energy + energy_tolerance {
        return Err(ImaginaryTimeSchrodingerError::EnergyIncreased {
            before: initial_rayleigh_energy,
            after: final_rayleigh_energy,
        });
    }
    let final_residual_l2 = residual_l2(
        &final_amplitude,
        &shifted_potential,
        &adjacency,
        final_rayleigh_energy,
    )?;
    let maximum_log_amplitude = log_amplitude
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let log_amplitude_boost = log_amplitude
        .iter()
        .map(|value| require_derived_finite("log_amplitude_boost", value - maximum_log_amplitude))
        .collect::<Result<Vec<_>, _>>()?;
    let (probability, probability_sum, probability_sum_tolerance) =
        probability_from_log(&log_amplitude)?;
    let probability_entropy = entropy(&probability)?;
    let raw_energy_drop = require_derived_finite(
        "rayleigh_energy_drop",
        initial_rayleigh_energy - final_rayleigh_energy,
    )?;

    Ok(ImaginaryTimeSchrodingerPayload {
        kind: ZSPACE_IMAGINARY_TIME_SCHRODINGER_KIND,
        contract_version: ZSPACE_IMAGINARY_TIME_SCHRODINGER_CONTRACT_VERSION,
        semantic_owner: ZSPACE_IMAGINARY_TIME_SCHRODINGER_SEMANTIC_OWNER,
        semantic_backend: ZSPACE_IMAGINARY_TIME_SCHRODINGER_SEMANTIC_BACKEND,
        backend: ZSPACE_IMAGINARY_TIME_SCHRODINGER_BACKEND,
        execution_backend: ZSPACE_IMAGINARY_TIME_SCHRODINGER_EXECUTION_BACKEND,
        route_blocker: ZSPACE_IMAGINARY_TIME_SCHRODINGER_ROUTE_BLOCKER,
        config,
        tags,
        potential,
        shifted_potential,
        edges,
        initial_amplitude: normalized_initial_amplitude,
        final_amplitude,
        probability,
        log_amplitude_boost,
        probability_sum,
        probability_sum_tolerance,
        effects: ImaginaryTimeSchrodingerEffects {
            evolution_applied: substeps > 0,
            edge_count: adjacency.iter().map(Vec::len).sum::<usize>() / 2,
            substeps,
            substep_imaginary_time,
            potential_shift,
            max_degree,
            spectral_upper_bound,
            initial_rayleigh_energy,
            final_rayleigh_energy,
            rayleigh_energy_drop: raw_energy_drop.max(0.0),
            energy_tolerance,
            initial_l2_norm,
            final_l2_norm,
            l2_norm_tolerance,
            initial_residual_l2,
            final_residual_l2,
            accumulated_log_norm,
            probability_entropy,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> ImaginaryTimeSchrodingerRequest {
        ImaginaryTimeSchrodingerRequest {
            tags: vec!["left".to_owned(), "right".to_owned()],
            potential: vec![0.0, 2.0],
            edges: vec![ImaginaryTimeSchrodingerEdge {
                left: 0,
                right: 1,
                weight: 1.0,
            }],
            initial_amplitude: Vec::new(),
            config: ImaginaryTimeSchrodingerConfig::new(1.0).expect("valid config"),
        }
    }

    #[test]
    fn lower_potential_gains_probability_and_energy_descends() {
        let payload = apply_imaginary_time_schrodinger(request()).expect("valid evolution");

        assert!(payload.probability[0] > payload.probability[1]);
        assert_eq!(payload.log_amplitude_boost[0], 0.0);
        assert!(payload.log_amplitude_boost[1] < 0.0);
        assert!(payload.effects.final_rayleigh_energy < payload.effects.initial_rayleigh_energy);
        assert!(payload.effects.rayleigh_energy_drop > 0.0);
        assert!((payload.probability_sum - 1.0).abs() < 1.0e-12);
        assert!(payload.probability_sum_tolerance > 0.0);
        assert!(payload.effects.energy_tolerance > 0.0);
        assert!(payload.effects.l2_norm_tolerance > 0.0);
        assert!(payload.effects.substeps >= 1);
    }

    #[test]
    fn constant_potential_is_an_exact_gauge() {
        let baseline = apply_imaginary_time_schrodinger(request()).expect("baseline");
        let mut shifted = request();
        for value in &mut shifted.potential {
            *value += 137.0;
        }
        let shifted = apply_imaginary_time_schrodinger(shifted).expect("shifted gauge");

        for (left, right) in baseline.probability.iter().zip(&shifted.probability) {
            assert!((left - right).abs() < 1.0e-12);
        }
        assert_eq!(shifted.effects.potential_shift, 137.0);
    }

    #[test]
    fn zero_time_is_identity_after_l2_normalization() {
        let mut request = request();
        request.initial_amplitude = vec![3.0, 4.0];
        request.config = ImaginaryTimeSchrodingerConfig::new(0.0).expect("zero time");
        let payload = apply_imaginary_time_schrodinger(request).expect("identity evolution");

        assert!((payload.initial_amplitude[0] - 0.6).abs() < 1.0e-12);
        assert!((payload.initial_amplitude[1] - 0.8).abs() < 1.0e-12);
        assert_eq!(payload.final_amplitude, payload.initial_amplitude);
        assert!((payload.probability[0] - 0.36).abs() < 1.0e-12);
        assert!((payload.probability[1] - 0.64).abs() < 1.0e-12);
        assert_eq!(payload.effects.substeps, 0);
        assert!(!payload.effects.evolution_applied);
    }

    #[test]
    fn disconnected_deep_potential_remains_finite_in_log_domain() {
        let request = ImaginaryTimeSchrodingerRequest {
            tags: vec!["ground".to_owned(), "suppressed".to_owned()],
            potential: vec![0.0, 1_000.0],
            edges: Vec::new(),
            initial_amplitude: Vec::new(),
            config: ImaginaryTimeSchrodingerConfig::new(10.0)
                .expect("valid time")
                .with_max_substeps(16_384)
                .expect("valid budget"),
        };
        let payload = apply_imaginary_time_schrodinger(request).expect("stable log evolution");

        assert!(payload.log_amplitude_boost[1].is_finite());
        assert!(payload.log_amplitude_boost[1] < -1_000.0);
        assert_eq!(payload.probability[0], 1.0);
        assert_eq!(payload.probability[1], 0.0);
    }

    #[test]
    fn step_budget_fails_before_evolution() {
        let request = ImaginaryTimeSchrodingerRequest {
            config: ImaginaryTimeSchrodingerConfig::new(100.0)
                .expect("valid time")
                .with_max_substeps(1)
                .expect("valid budget"),
            ..request()
        };
        assert!(matches!(
            apply_imaginary_time_schrodinger(request),
            Err(ImaginaryTimeSchrodingerError::StepBudgetExceeded { .. })
        ));
    }

    #[test]
    fn malformed_graph_fails_closed() {
        let mut malformed = request();
        malformed.edges[0].left = 1;
        malformed.edges[0].right = 0;
        assert!(matches!(
            apply_imaginary_time_schrodinger(malformed),
            Err(ImaginaryTimeSchrodingerError::NonCanonicalEdge { .. })
        ));

        let mut duplicate = request();
        duplicate.edges.push(duplicate.edges[0]);
        assert!(matches!(
            apply_imaginary_time_schrodinger(duplicate),
            Err(ImaginaryTimeSchrodingerError::DuplicateEdge { .. })
        ));

        let mut zero_weight = request();
        zero_weight.edges[0].weight = 0.0;
        assert!(matches!(
            apply_imaginary_time_schrodinger(zero_weight),
            Err(ImaginaryTimeSchrodingerError::NonPositive {
                field: "edge_weight",
                ..
            })
        ));
    }

    #[test]
    fn non_positive_initial_amplitude_is_rejected() {
        let request = ImaginaryTimeSchrodingerRequest {
            initial_amplitude: vec![1.0, 0.0],
            ..request()
        };
        assert!(matches!(
            apply_imaginary_time_schrodinger(request),
            Err(ImaginaryTimeSchrodingerError::NonPositive {
                field: "initial_amplitude",
                ..
            })
        ));
    }

    #[test]
    fn serialized_requests_fail_closed_on_unknown_fields() {
        let error = serde_json::from_value::<ImaginaryTimeSchrodingerRequest>(serde_json::json!({
            "tags": ["only"],
            "potential": [0.0],
            "hamiltonian": []
        }))
        .expect_err("unknown request field must fail closed");
        assert!(error.to_string().contains("unknown field"));

        let error = serde_json::from_value::<ImaginaryTimeSchrodingerRequest>(serde_json::json!({
            "tags": ["only"],
            "potential": [0.0],
            "config": {"imaginary_time": 1.0, "integrator": "euler"}
        }))
        .expect_err("unknown config field must fail closed");
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn varied_graphs_preserve_normalization_and_descend_in_energy() {
        fn uniform(seed: &mut u64) -> f64 {
            *seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((*seed >> 11) as f64) / ((1_u64 << 53) as f64)
        }

        let mut seed = 0x5c48_524f_4449_4e47;
        for case in 0..32 {
            let dimension = 2 + case % 7;
            let mut edges = Vec::new();
            for left in 0..dimension {
                for right in (left + 1)..dimension {
                    if right == left + 1 || uniform(&mut seed) > 0.65 {
                        edges.push(ImaginaryTimeSchrodingerEdge {
                            left,
                            right,
                            weight: 0.05 + 0.95 * uniform(&mut seed),
                        });
                    }
                }
            }
            let potential = (0..dimension)
                .map(|_| 4.0 * uniform(&mut seed) - 2.0)
                .collect::<Vec<_>>();
            let initial_amplitude = (0..dimension)
                .map(|_| 0.1 + 1.9 * uniform(&mut seed))
                .collect::<Vec<_>>();
            let config = ImaginaryTimeSchrodingerConfig::new(0.05 + 1.95 * uniform(&mut seed))
                .expect("valid randomized time");
            let payload = apply_imaginary_time_schrodinger(ImaginaryTimeSchrodingerRequest {
                tags: (0..dimension)
                    .map(|index| format!("node:{index}"))
                    .collect(),
                potential,
                edges,
                initial_amplitude,
                config,
            })
            .expect("stable randomized evolution");

            assert!(
                payload.effects.final_rayleigh_energy
                    <= payload.effects.initial_rayleigh_energy + payload.effects.energy_tolerance
            );
            assert!(
                (payload.effects.initial_l2_norm - 1.0).abs() <= payload.effects.l2_norm_tolerance
            );
            assert!(
                (payload.effects.final_l2_norm - 1.0).abs() <= payload.effects.l2_norm_tolerance
            );
            assert!((payload.probability_sum - 1.0).abs() <= payload.probability_sum_tolerance);
        }
    }
}
