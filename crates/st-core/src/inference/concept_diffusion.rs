// SPDX-License-Identifier: AGPL-3.0-or-later

//! Canonical probability-simplex diffusion and Fisher-Rao semantics.
//!
//! This module owns the deterministic transition from a labelled probability
//! state, an affinity graph, optional anisotropic conductivity, observations,
//! and Z-space bias into the next state. Language bindings may transport the
//! matrices and state, but must not reconstruct these formulas.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use thiserror::Error;

pub const ZSPACE_CONCEPT_DIFFUSION_CONTRACT_VERSION: &str =
    "spiraltorch.zspace_concept_diffusion.v1";
pub const ZSPACE_CONCEPT_DIFFUSION_KIND: &str = "spiraltorch.zspace_concept_diffusion";
pub const ZSPACE_CONCEPT_DIFFUSION_SEMANTIC_OWNER: &str = "st-core::inference::concept_diffusion";
pub const ZSPACE_CONCEPT_DIFFUSION_SEMANTIC_BACKEND: &str = "rust";
pub const ZSPACE_CONCEPT_DIFFUSION_BACKEND: &str = "spiraltorch_concept_diffusion_core";
pub const ZSPACE_CONCEPT_DIFFUSION_EXECUTION_BACKEND: &str = "f64_cpu";

const BASE_PROBABILITY_SUM_TOLERANCE: f64 = 1.0e-6;
const F32_SUM_TOLERANCE_MULTIPLIER: f64 = 4.0;
const INTERNAL_INVARIANT_TOLERANCE: f64 = 1.0e-10;
pub const FISHER_RAO_KL_FLOOR: f64 = 1.0e-12;

#[derive(Debug, Error, PartialEq)]
pub enum ConceptDiffusionError {
    #[error("concept diffusion state must not be empty")]
    EmptyState,
    #[error("concept diffusion field '{field}' has length {actual}, expected {expected}")]
    DimensionMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("concept diffusion tag at index {index} must not be empty")]
    EmptyTag { index: usize },
    #[error("concept diffusion tag '{tag}' is duplicated")]
    DuplicateTag { tag: String },
    #[error("concept diffusion field '{field}' must be finite, got {value}")]
    NonFinite { field: &'static str, value: f64 },
    #[error("concept diffusion field '{field}' must be non-negative, got {value}")]
    Negative { field: &'static str, value: f64 },
    #[error("concept diffusion field '{field}' must be positive, got {value}")]
    NonPositive { field: &'static str, value: f64 },
    #[error("concept diffusion field '{field}' must be in [{min}, {max}], got {value}")]
    OutOfRange {
        field: &'static str,
        value: f64,
        min: f64,
        max: f64,
    },
    #[error(
        "concept diffusion probability mass for '{field}' must be within {tolerance} of 1, got {sum}"
    )]
    InvalidProbabilityMass {
        field: &'static str,
        sum: f64,
        tolerance: f64,
    },
    #[error(
        "concept diffusion matrix '{field}' is asymmetric at ({row}, {col}): {left} vs {right} (tolerance {tolerance})"
    )]
    AsymmetricMatrix {
        field: &'static str,
        row: usize,
        col: usize,
        left: f64,
        right: f64,
        tolerance: f64,
    },
    #[error(
        "concept diffusion needs at least {required_at_least} CFL substeps, exceeding configured maximum {maximum}"
    )]
    StepBudgetExceeded {
        required_at_least: f64,
        maximum: usize,
    },
    #[error("derived concept diffusion field '{field}' must be finite, got {value}")]
    NonFiniteDerived { field: &'static str, value: f64 },
    #[error("concept diffusion entropy decreased under heat flow: before={before}, after={after}")]
    EntropyDecreased { before: f64, after: f64 },
    #[error(
        "concept diffusion Dirichlet energy increased under heat flow: before={before}, after={after}"
    )]
    DirichletEnergyIncreased { before: f64, after: f64 },
}

fn require_finite(field: &'static str, value: f64) -> Result<f64, ConceptDiffusionError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ConceptDiffusionError::NonFinite { field, value })
    }
}

fn require_non_negative(field: &'static str, value: f64) -> Result<f64, ConceptDiffusionError> {
    require_finite(field, value)?;
    if value >= 0.0 {
        Ok(value)
    } else {
        Err(ConceptDiffusionError::Negative { field, value })
    }
}

fn require_positive(field: &'static str, value: f64) -> Result<f64, ConceptDiffusionError> {
    require_finite(field, value)?;
    if value > 0.0 {
        Ok(value)
    } else {
        Err(ConceptDiffusionError::NonPositive { field, value })
    }
}

fn require_range(
    field: &'static str,
    value: f64,
    min: f64,
    max: f64,
) -> Result<f64, ConceptDiffusionError> {
    require_finite(field, value)?;
    if (min..=max).contains(&value) {
        Ok(value)
    } else {
        Err(ConceptDiffusionError::OutOfRange {
            field,
            value,
            min,
            max,
        })
    }
}

fn require_derived_finite(field: &'static str, value: f64) -> Result<f64, ConceptDiffusionError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ConceptDiffusionError::NonFiniteDerived { field, value })
    }
}

fn checked_add(field: &'static str, left: f64, right: f64) -> Result<f64, ConceptDiffusionError> {
    require_derived_finite(field, left + right)
}

fn checked_mul(field: &'static str, left: f64, right: f64) -> Result<f64, ConceptDiffusionError> {
    require_derived_finite(field, left * right)
}

pub fn concept_probability_sum_tolerance(value_count: usize) -> f64 {
    BASE_PROBABILITY_SUM_TOLERANCE
        + (value_count as f64).sqrt() * f64::from(f32::EPSILON) * F32_SUM_TOLERANCE_MULTIPLIER
}

struct NormalizedDistribution {
    values: Vec<f64>,
    input_sum: f64,
    tolerance: f64,
}

fn validate_distribution(
    field: &'static str,
    values: &[f64],
    expected: Option<usize>,
) -> Result<NormalizedDistribution, ConceptDiffusionError> {
    if values.is_empty() {
        return Err(ConceptDiffusionError::EmptyState);
    }
    if let Some(expected) = expected {
        if values.len() != expected {
            return Err(ConceptDiffusionError::DimensionMismatch {
                field,
                expected,
                actual: values.len(),
            });
        }
    }
    let mut sum = 0.0;
    for &value in values {
        require_non_negative(field, value)?;
        sum = checked_add("probability_sum", sum, value)?;
    }
    let tolerance = concept_probability_sum_tolerance(values.len());
    if (sum - 1.0).abs() > tolerance {
        return Err(ConceptDiffusionError::InvalidProbabilityMass {
            field,
            sum,
            tolerance,
        });
    }
    require_positive("probability_sum", sum)?;
    let normalized = values
        .iter()
        .copied()
        .map(|value| require_derived_finite("normalized_probability", value / sum))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(NormalizedDistribution {
        values: normalized,
        input_sum: sum,
        tolerance,
    })
}

fn normalize_internal(values: &mut [f64]) -> Result<f64, ConceptDiffusionError> {
    let mut sum = 0.0;
    for value in values.iter_mut() {
        require_derived_finite("next_probability", *value)?;
        if *value < -INTERNAL_INVARIANT_TOLERANCE {
            return Err(ConceptDiffusionError::Negative {
                field: "next_probability",
                value: *value,
            });
        }
        if *value < 0.0 {
            *value = 0.0;
        }
        sum = checked_add("next_probability_sum", sum, *value)?;
    }
    require_positive("next_probability_sum", sum)?;
    for value in values {
        *value = require_derived_finite("normalized_next_probability", *value / sum)?;
    }
    Ok(sum)
}

fn distribution_entropy(values: &[f64]) -> Result<f64, ConceptDiffusionError> {
    let mut entropy = 0.0;
    for &value in values {
        if value > 0.0 {
            let term = require_derived_finite("entropy_term", -value * value.ln())?;
            entropy = checked_add("entropy", entropy, term)?;
        }
    }
    Ok(entropy)
}

fn validate_tags(tags: &[String], expected: usize) -> Result<(), ConceptDiffusionError> {
    if tags.len() != expected {
        return Err(ConceptDiffusionError::DimensionMismatch {
            field: "tags",
            expected,
            actual: tags.len(),
        });
    }
    let mut seen = HashSet::with_capacity(tags.len());
    for (index, tag) in tags.iter().enumerate() {
        if tag.trim().is_empty() {
            return Err(ConceptDiffusionError::EmptyTag { index });
        }
        if !seen.insert(tag.as_str()) {
            return Err(ConceptDiffusionError::DuplicateTag { tag: tag.clone() });
        }
    }
    Ok(())
}

fn validate_symmetric_nonnegative_matrix(
    field: &'static str,
    matrix: &[Vec<f64>],
    dimension: usize,
    symmetry_tolerance: f64,
) -> Result<(), ConceptDiffusionError> {
    if matrix.len() != dimension {
        return Err(ConceptDiffusionError::DimensionMismatch {
            field,
            expected: dimension,
            actual: matrix.len(),
        });
    }
    for row in matrix {
        if row.len() != dimension {
            return Err(ConceptDiffusionError::DimensionMismatch {
                field,
                expected: dimension,
                actual: row.len(),
            });
        }
        for &value in row {
            require_non_negative(field, value)?;
        }
    }
    for (row, row_values) in matrix.iter().enumerate() {
        for (col, col_values) in matrix.iter().enumerate().skip(row + 1) {
            let left = row_values[col];
            let right = col_values[row];
            let scale = left.abs().max(right.abs()).max(1.0);
            let tolerance = checked_mul("matrix_symmetry_tolerance", symmetry_tolerance, scale)?;
            if (left - right).abs() > tolerance {
                return Err(ConceptDiffusionError::AsymmetricMatrix {
                    field,
                    row,
                    col,
                    left,
                    right,
                    tolerance,
                });
            }
        }
    }
    Ok(())
}

/// Validates the discrete symmetric conductivity tensor used by the graph heat flow.
pub fn validate_concept_diffusion_conductivity(
    matrix: &[Vec<f64>],
    dimension: usize,
    symmetry_tolerance: f64,
) -> Result<(), ConceptDiffusionError> {
    require_non_negative("symmetry_tolerance", symmetry_tolerance)?;
    validate_symmetric_nonnegative_matrix("diffusion_tensor", matrix, dimension, symmetry_tolerance)
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConceptDiffusionConfig {
    timestep: f64,
    cfl_limit: f64,
    max_substeps: usize,
    symmetry_tolerance: f64,
}

impl Default for ConceptDiffusionConfig {
    fn default() -> Self {
        Self {
            timestep: 1.0e-2,
            cfl_limit: 0.9,
            max_substeps: 4096,
            symmetry_tolerance: 1.0e-10,
        }
    }
}

impl ConceptDiffusionConfig {
    pub fn new(timestep: f64) -> Result<Self, ConceptDiffusionError> {
        let config = Self {
            timestep,
            ..Self::default()
        };
        config.validate()?;
        Ok(config)
    }

    pub fn with_timestep(mut self, timestep: f64) -> Result<Self, ConceptDiffusionError> {
        self.timestep = timestep;
        self.validate()?;
        Ok(self)
    }

    pub fn with_cfl_limit(mut self, cfl_limit: f64) -> Result<Self, ConceptDiffusionError> {
        self.cfl_limit = cfl_limit;
        self.validate()?;
        Ok(self)
    }

    pub fn with_max_substeps(mut self, max_substeps: usize) -> Result<Self, ConceptDiffusionError> {
        self.max_substeps = max_substeps;
        self.validate()?;
        Ok(self)
    }

    pub fn with_symmetry_tolerance(
        mut self,
        symmetry_tolerance: f64,
    ) -> Result<Self, ConceptDiffusionError> {
        self.symmetry_tolerance = symmetry_tolerance;
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), ConceptDiffusionError> {
        require_non_negative("timestep", self.timestep)?;
        require_range("cfl_limit", self.cfl_limit, f64::MIN_POSITIVE, 1.0)?;
        if self.max_substeps == 0 {
            return Err(ConceptDiffusionError::NonPositive {
                field: "max_substeps",
                value: 0.0,
            });
        }
        require_non_negative("symmetry_tolerance", self.symmetry_tolerance)?;
        Ok(())
    }

    pub fn timestep(&self) -> f64 {
        self.timestep
    }

    pub fn cfl_limit(&self) -> f64 {
        self.cfl_limit
    }

    pub fn max_substeps(&self) -> usize {
        self.max_substeps
    }

    pub fn symmetry_tolerance(&self) -> f64 {
        self.symmetry_tolerance
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ConceptDiffusionObservation {
    pub probabilities: Vec<f64>,
    pub weight: f64,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ConceptDiffusionRequest {
    pub tags: Vec<String>,
    pub state: Vec<f64>,
    pub affinity: Vec<Vec<f64>>,
    pub diffusion_tensor: Option<Vec<Vec<f64>>>,
    pub z_bias: Vec<f64>,
    pub observation: Option<ConceptDiffusionObservation>,
    pub config: ConceptDiffusionConfig,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ConceptDiffusionEffects {
    pub observation_provided: bool,
    pub observation_applied: bool,
    pub observation_weight: f64,
    pub z_bias_applied: bool,
    pub diffusion_applied: bool,
    pub edge_count: usize,
    pub substeps: usize,
    pub substep_timestep: f64,
    pub max_degree: f64,
    pub entropy_before: f64,
    pub entropy_after_observation: f64,
    pub entropy_after_bias: f64,
    pub entropy_after_diffusion: f64,
    pub dirichlet_energy_before: f64,
    pub dirichlet_energy_after: f64,
    pub total_variation: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ConceptDiffusionPayload {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub backend: &'static str,
    pub execution_backend: &'static str,
    pub config: ConceptDiffusionConfig,
    pub tags: Vec<String>,
    pub previous_state: Vec<f64>,
    pub state_after_observation: Vec<f64>,
    pub state_after_bias: Vec<f64>,
    pub next_state: Vec<f64>,
    pub input_probability_sum: f64,
    pub output_probability_sum: f64,
    pub probability_sum_tolerance: f64,
    pub conductivity_source: &'static str,
    pub effects: ConceptDiffusionEffects,
}

/// Fisher-Rao comparison for two categorical probability distributions.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct FisherRaoComparison {
    /// Directed KL values use `kl_floor` at zero-probability denominators.
    pub forward_kl: f64,
    pub reverse_kl: f64,
    pub symmetric_kl: f64,
    pub kl_floor: f64,
    pub bhattacharyya_coefficient: f64,
    pub fisher_rao_distance: f64,
    /// The categorical simplex is a sphere of radius two under the square-root map.
    pub sectional_curvature: Option<f64>,
}

pub fn compare_fisher_rao(
    left: &[f64],
    right: &[f64],
) -> Result<FisherRaoComparison, ConceptDiffusionError> {
    let left = validate_distribution("left_probability", left, None)?;
    let right = validate_distribution("right_probability", right, Some(left.values.len()))?;
    let mut forward_kl = 0.0;
    let mut backward_kl = 0.0;
    let mut coefficient = 0.0;
    for (&left_value, &right_value) in left.values.iter().zip(&right.values) {
        if left_value > 0.0 {
            let denominator = right_value.max(FISHER_RAO_KL_FLOOR);
            forward_kl = checked_add(
                "forward_kl",
                forward_kl,
                require_derived_finite(
                    "forward_kl_term",
                    left_value * (left_value / denominator).ln(),
                )?,
            )?;
        }
        if right_value > 0.0 {
            let denominator = left_value.max(FISHER_RAO_KL_FLOOR);
            backward_kl = checked_add(
                "backward_kl",
                backward_kl,
                require_derived_finite(
                    "backward_kl_term",
                    right_value * (right_value / denominator).ln(),
                )?,
            )?;
        }
        coefficient = checked_add(
            "bhattacharyya_coefficient",
            coefficient,
            require_derived_finite("bhattacharyya_term", (left_value * right_value).sqrt())?,
        )?;
    }
    coefficient = coefficient.clamp(0.0, 1.0);
    let fisher_rao_distance =
        require_derived_finite("fisher_rao_distance", 2.0 * coefficient.acos())?;
    Ok(FisherRaoComparison {
        forward_kl,
        reverse_kl: backward_kl,
        symmetric_kl: require_derived_finite("symmetric_kl", 0.5 * (forward_kl + backward_kl))?,
        kl_floor: FISHER_RAO_KL_FLOOR,
        bhattacharyya_coefficient: coefficient,
        fisher_rao_distance,
        sectional_curvature: (left.values.len() >= 3).then_some(0.25),
    })
}

/// Returns the exact ambient Fisher-information diagonal for an interior point.
/// Boundary points are singular and therefore fail instead of being clamped.
pub fn fisher_information_diagonal(
    probabilities: &[f64],
) -> Result<Vec<f64>, ConceptDiffusionError> {
    let probabilities = validate_distribution("fisher_probability", probabilities, None)?;
    probabilities
        .values
        .into_iter()
        .map(|value| {
            require_positive("fisher_probability", value)?;
            require_derived_finite("fisher_diagonal", 1.0 / value)
        })
        .collect()
}

/// Blends a validated probability observation without mutating the input state.
pub fn blend_concept_diffusion_observation(
    state: &[f64],
    observation: &ConceptDiffusionObservation,
) -> Result<Vec<f64>, ConceptDiffusionError> {
    let state = validate_distribution("state", state, None)?;
    let observation_values = validate_distribution(
        "observation_probability",
        &observation.probabilities,
        Some(state.values.len()),
    )?;
    let weight = require_range("observation_weight", observation.weight, 0.0, 1.0)?;
    let retained = 1.0 - weight;
    let mut blended = state
        .values
        .iter()
        .zip(&observation_values.values)
        .map(|(&current, &observed)| {
            checked_add(
                "blended_probability",
                checked_mul("retained_probability", retained, current)?,
                checked_mul("observed_probability", weight, observed)?,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    normalize_internal(&mut blended)?;
    Ok(blended)
}

fn apply_z_bias(
    state: &[f64],
    z_bias: &[f64],
    timestep: f64,
) -> Result<(Vec<f64>, bool), ConceptDiffusionError> {
    let applied = timestep > 0.0 && z_bias.iter().any(|value| *value != 0.0);
    if !applied {
        return Ok((state.to_vec(), false));
    }
    let scaled = z_bias
        .iter()
        .copied()
        .map(|value| checked_mul("z_bias_exponent", timestep, value))
        .collect::<Result<Vec<_>, _>>()?;
    let max_supported = state
        .iter()
        .zip(&scaled)
        .filter_map(|(&probability, &value)| (probability > 0.0).then_some(value))
        .fold(f64::NEG_INFINITY, f64::max);
    require_derived_finite("z_bias_max_exponent", max_supported)?;
    let mut tilted = state
        .iter()
        .zip(&scaled)
        .map(|(&probability, &value)| {
            checked_mul(
                "z_bias_weighted_probability",
                probability,
                require_derived_finite("z_bias_weight", (value - max_supported).exp())?,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    normalize_internal(&mut tilted)?;
    Ok((tilted, true))
}

struct ConductanceGraph {
    matrix: Vec<Vec<f64>>,
    degree: Vec<f64>,
    edge_count: usize,
    max_degree: f64,
}

fn build_conductance(
    affinity: &[Vec<f64>],
    diffusion_tensor: &[Vec<f64>],
) -> Result<ConductanceGraph, ConceptDiffusionError> {
    let dimension = affinity.len();
    let mut conductance = vec![vec![0.0; dimension]; dimension];
    let mut degree = vec![0.0; dimension];
    let mut edge_count = 0;
    for row in 0..dimension {
        for col in row + 1..dimension {
            let value = checked_mul(
                "edge_conductance",
                affinity[row][col],
                diffusion_tensor[row][col],
            )?;
            conductance[row][col] = value;
            conductance[col][row] = value;
            degree[row] = checked_add("graph_degree", degree[row], value)?;
            degree[col] = checked_add("graph_degree", degree[col], value)?;
            if value > 0.0 {
                edge_count += 1;
            }
        }
    }
    let max_degree = degree.iter().copied().fold(0.0, f64::max);
    Ok(ConductanceGraph {
        matrix: conductance,
        degree,
        edge_count,
        max_degree,
    })
}

fn dirichlet_energy(state: &[f64], conductance: &[Vec<f64>]) -> Result<f64, ConceptDiffusionError> {
    let mut energy = 0.0;
    for row in 0..state.len() {
        for col in row + 1..state.len() {
            let delta = require_derived_finite("dirichlet_delta", state[row] - state[col])?;
            let term = checked_mul(
                "dirichlet_term",
                conductance[row][col],
                checked_mul("dirichlet_delta_squared", delta, delta)?,
            )?;
            energy = checked_add("dirichlet_energy", energy, term)?;
        }
    }
    Ok(energy)
}

struct HeatFlowResult {
    state: Vec<f64>,
    edge_count: usize,
    substeps: usize,
    substep_timestep: f64,
    max_degree: f64,
    energy_before: f64,
    energy_after: f64,
}

fn apply_heat_flow(
    state: &[f64],
    affinity: &[Vec<f64>],
    diffusion_tensor: &[Vec<f64>],
    config: ConceptDiffusionConfig,
) -> Result<HeatFlowResult, ConceptDiffusionError> {
    let ConductanceGraph {
        matrix: conductance,
        degree,
        edge_count,
        max_degree,
    } = build_conductance(affinity, diffusion_tensor)?;
    let energy_before = dirichlet_energy(state, &conductance)?;
    if config.timestep == 0.0 || edge_count == 0 || max_degree == 0.0 {
        return Ok(HeatFlowResult {
            state: state.to_vec(),
            edge_count,
            substeps: 0,
            substep_timestep: 0.0,
            max_degree,
            energy_before,
            energy_after: energy_before,
        });
    }

    let required = require_derived_finite(
        "required_substeps",
        config.timestep * max_degree / config.cfl_limit,
    )?
    .ceil()
    .max(1.0);
    if required > config.max_substeps as f64 {
        return Err(ConceptDiffusionError::StepBudgetExceeded {
            required_at_least: required,
            maximum: config.max_substeps,
        });
    }
    let substeps = required as usize;
    let substep_timestep =
        require_derived_finite("substep_timestep", config.timestep / substeps as f64)?;
    let mut current = state.to_vec();
    for _ in 0..substeps {
        let mut next = vec![0.0; current.len()];
        for row in 0..current.len() {
            let retained = require_derived_finite(
                "heat_retained_weight",
                1.0 - substep_timestep * degree[row],
            )?;
            if retained < -INTERNAL_INVARIANT_TOLERANCE {
                return Err(ConceptDiffusionError::Negative {
                    field: "heat_retained_weight",
                    value: retained,
                });
            }
            let mut value =
                checked_mul("heat_retained_probability", retained.max(0.0), current[row])?;
            for col in 0..current.len() {
                if row == col || conductance[row][col] == 0.0 {
                    continue;
                }
                value = checked_add(
                    "heat_probability",
                    value,
                    checked_mul(
                        "heat_incoming_probability",
                        substep_timestep,
                        checked_mul(
                            "heat_weighted_neighbor",
                            conductance[row][col],
                            current[col],
                        )?,
                    )?,
                )?;
            }
            next[row] = value;
        }
        normalize_internal(&mut next)?;
        current = next;
    }
    let energy_after = dirichlet_energy(&current, &conductance)?;
    Ok(HeatFlowResult {
        state: current,
        edge_count,
        substeps,
        substep_timestep,
        max_degree,
        energy_before,
        energy_after,
    })
}

fn total_variation(left: &[f64], right: &[f64]) -> Result<f64, ConceptDiffusionError> {
    let l1 = left.iter().zip(right).try_fold(0.0, |sum, (&a, &b)| {
        checked_add("total_variation_l1", sum, (a - b).abs())
    })?;
    require_derived_finite("total_variation", 0.5 * l1)
}

/// Applies one atomic observation, Z-bias, and graph heat-flow transition.
pub fn apply_concept_diffusion(
    request: ConceptDiffusionRequest,
) -> Result<ConceptDiffusionPayload, ConceptDiffusionError> {
    let ConceptDiffusionRequest {
        tags,
        state,
        affinity,
        diffusion_tensor,
        z_bias,
        observation,
        config,
    } = request;
    config.validate()?;
    let state = validate_distribution("state", &state, None)?;
    let dimension = state.values.len();
    validate_tags(&tags, dimension)?;
    validate_symmetric_nonnegative_matrix(
        "affinity",
        &affinity,
        dimension,
        config.symmetry_tolerance,
    )?;
    let (diffusion_tensor, conductivity_source) = match diffusion_tensor {
        Some(matrix) => {
            validate_concept_diffusion_conductivity(&matrix, dimension, config.symmetry_tolerance)?;
            (matrix, "request")
        }
        None => (vec![vec![1.0; dimension]; dimension], "uniform"),
    };
    let z_bias = if z_bias.is_empty() {
        vec![0.0; dimension]
    } else {
        if z_bias.len() != dimension {
            return Err(ConceptDiffusionError::DimensionMismatch {
                field: "z_bias",
                expected: dimension,
                actual: z_bias.len(),
            });
        }
        for &value in &z_bias {
            require_finite("z_bias", value)?;
        }
        z_bias
    };

    let previous_state = state.values;
    let entropy_before = distribution_entropy(&previous_state)?;
    let observation_provided = observation.is_some();
    let observation_weight = observation.as_ref().map_or(0.0, |value| value.weight);
    let state_after_observation = match observation.as_ref() {
        Some(observation) => blend_concept_diffusion_observation(&previous_state, observation)?,
        None => previous_state.clone(),
    };
    let observation_applied = observation_provided && observation_weight > 0.0;
    let entropy_after_observation = distribution_entropy(&state_after_observation)?;
    let (state_after_bias, z_bias_applied) =
        apply_z_bias(&state_after_observation, &z_bias, config.timestep)?;
    let entropy_after_bias = distribution_entropy(&state_after_bias)?;
    let heat = apply_heat_flow(&state_after_bias, &affinity, &diffusion_tensor, config)?;
    let entropy_after_diffusion = distribution_entropy(&heat.state)?;
    let entropy_tolerance = INTERNAL_INVARIANT_TOLERANCE * (1.0 + entropy_after_bias.abs());
    if entropy_after_diffusion + entropy_tolerance < entropy_after_bias {
        return Err(ConceptDiffusionError::EntropyDecreased {
            before: entropy_after_bias,
            after: entropy_after_diffusion,
        });
    }
    let energy_tolerance = INTERNAL_INVARIANT_TOLERANCE * (1.0 + heat.energy_before.abs());
    if heat.energy_after > heat.energy_before + energy_tolerance {
        return Err(ConceptDiffusionError::DirichletEnergyIncreased {
            before: heat.energy_before,
            after: heat.energy_after,
        });
    }
    let output_probability_sum = heat.state.iter().try_fold(0.0, |sum, &value| {
        checked_add("output_probability_sum", sum, value)
    })?;
    let variation = total_variation(&previous_state, &heat.state)?;

    Ok(ConceptDiffusionPayload {
        kind: ZSPACE_CONCEPT_DIFFUSION_KIND,
        contract_version: ZSPACE_CONCEPT_DIFFUSION_CONTRACT_VERSION,
        semantic_owner: ZSPACE_CONCEPT_DIFFUSION_SEMANTIC_OWNER,
        semantic_backend: ZSPACE_CONCEPT_DIFFUSION_SEMANTIC_BACKEND,
        backend: ZSPACE_CONCEPT_DIFFUSION_BACKEND,
        execution_backend: ZSPACE_CONCEPT_DIFFUSION_EXECUTION_BACKEND,
        config,
        tags,
        previous_state,
        state_after_observation,
        state_after_bias,
        next_state: heat.state,
        input_probability_sum: state.input_sum,
        output_probability_sum,
        probability_sum_tolerance: state.tolerance,
        conductivity_source,
        effects: ConceptDiffusionEffects {
            observation_provided,
            observation_applied,
            observation_weight,
            z_bias_applied,
            diffusion_applied: heat.substeps > 0,
            edge_count: heat.edge_count,
            substeps: heat.substeps,
            substep_timestep: heat.substep_timestep,
            max_degree: heat.max_degree,
            entropy_before,
            entropy_after_observation,
            entropy_after_bias,
            entropy_after_diffusion,
            dirichlet_energy_before: heat.energy_before,
            dirichlet_energy_after: heat.energy_after,
            total_variation: variation,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> ConceptDiffusionRequest {
        ConceptDiffusionRequest {
            tags: vec!["left".to_owned(), "right".to_owned()],
            state: vec![1.0, 0.0],
            affinity: vec![vec![0.0, 1.0], vec![1.0, 0.0]],
            diffusion_tensor: None,
            z_bias: Vec::new(),
            observation: None,
            config: ConceptDiffusionConfig::new(0.25).expect("valid config"),
        }
    }

    #[test]
    fn heat_flow_moves_a_peak_toward_uniformity() {
        let output = apply_concept_diffusion(request()).expect("valid heat flow");

        assert_eq!(output.next_state, vec![0.75, 0.25]);
        assert!(output.effects.entropy_after_diffusion > output.effects.entropy_after_bias);
        assert!(output.effects.dirichlet_energy_after < output.effects.dirichlet_energy_before);
        assert_eq!(output.effects.substeps, 1);
        assert_eq!(output.effects.edge_count, 1);
        assert_eq!(
            output.semantic_owner,
            ZSPACE_CONCEPT_DIFFUSION_SEMANTIC_OWNER
        );
    }

    #[test]
    fn diffusion_never_uses_the_old_anti_diffusion_sign() {
        let mut input = request();
        input.state = vec![0.75, 0.25];

        let output = apply_concept_diffusion(input).expect("valid heat flow");

        assert!(output.next_state[0] < 0.75);
        assert!(output.next_state[1] > 0.25);
    }

    #[test]
    fn positive_z_bias_tilts_mass_toward_the_selected_tag() {
        let mut input = request();
        input.state = vec![0.5, 0.5];
        input.affinity = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        input.z_bias = vec![0.0, 2.0];
        input.config = ConceptDiffusionConfig::new(0.5).expect("valid config");

        let output = apply_concept_diffusion(input).expect("valid bias tilt");

        assert!(output.next_state[1] > 0.5);
        assert!(output.effects.z_bias_applied);
        assert!(!output.effects.diffusion_applied);
    }

    #[test]
    fn observation_blend_is_explicit_and_mass_preserving() {
        let blended = blend_concept_diffusion_observation(
            &[0.8, 0.2],
            &ConceptDiffusionObservation {
                probabilities: vec![0.2, 0.8],
                weight: 0.25,
            },
        )
        .expect("valid observation");

        assert!((blended[0] - 0.65).abs() < 1.0e-12);
        assert!((blended[1] - 0.35).abs() < 1.0e-12);
        assert!((blended.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn cfl_substeps_keep_large_valid_steps_on_the_simplex() {
        let mut input = request();
        input.config = ConceptDiffusionConfig::new(2.0).expect("valid config");

        let output = apply_concept_diffusion(input).expect("CFL substeps stabilize the step");

        assert_eq!(output.effects.substeps, 3);
        assert!(output.next_state.iter().all(|value| *value >= 0.0));
        assert!((output.next_state.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn anisotropic_conductivity_does_not_leak_through_closed_edges() {
        let output = apply_concept_diffusion(ConceptDiffusionRequest {
            tags: vec!["source".to_owned(), "open".to_owned(), "closed".to_owned()],
            state: vec![1.0, 0.0, 0.0],
            affinity: vec![
                vec![0.0, 1.0, 1.0],
                vec![1.0, 0.0, 1.0],
                vec![1.0, 1.0, 0.0],
            ],
            diffusion_tensor: Some(vec![
                vec![1.0, 1.0, 0.0],
                vec![1.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ]),
            z_bias: Vec::new(),
            observation: None,
            config: ConceptDiffusionConfig::new(0.25).expect("valid config"),
        })
        .expect("valid anisotropic heat flow");

        assert_eq!(output.next_state, vec![0.75, 0.25, 0.0]);
        assert_eq!(output.conductivity_source, "request");
        assert_eq!(output.effects.edge_count, 1);
    }

    #[test]
    fn configured_substep_budget_fails_closed() {
        let mut input = request();
        input.config = ConceptDiffusionConfig::new(10.0)
            .expect("valid config")
            .with_max_substeps(2)
            .expect("valid budget");

        let error = apply_concept_diffusion(input).expect_err("budget must be enforced");

        assert!(matches!(
            error,
            ConceptDiffusionError::StepBudgetExceeded { maximum: 2, .. }
        ));
    }

    #[test]
    fn asymmetric_affinity_and_negative_conductivity_are_rejected() {
        let mut asymmetric = request();
        asymmetric.affinity[1][0] = 0.5;
        assert!(matches!(
            apply_concept_diffusion(asymmetric),
            Err(ConceptDiffusionError::AsymmetricMatrix {
                field: "affinity",
                ..
            })
        ));

        let mut negative = request();
        negative.diffusion_tensor = Some(vec![vec![1.0, -0.1], vec![-0.1, 1.0]]);
        assert!(matches!(
            apply_concept_diffusion(negative),
            Err(ConceptDiffusionError::Negative {
                field: "diffusion_tensor",
                ..
            })
        ));
    }

    #[test]
    fn invalid_mass_and_duplicate_labels_are_rejected() {
        let mut invalid_mass = request();
        invalid_mass.state = vec![0.8, 0.8];
        assert!(matches!(
            apply_concept_diffusion(invalid_mass),
            Err(ConceptDiffusionError::InvalidProbabilityMass { field: "state", .. })
        ));

        let mut duplicate = request();
        duplicate.tags = vec!["same".to_owned(), "same".to_owned()];
        assert!(matches!(
            apply_concept_diffusion(duplicate),
            Err(ConceptDiffusionError::DuplicateTag { .. })
        ));
    }

    #[test]
    fn fisher_rao_geometry_matches_the_square_root_embedding() {
        let identical =
            compare_fisher_rao(&[0.5, 0.25, 0.25], &[0.5, 0.25, 0.25]).expect("valid comparison");
        assert!(identical.forward_kl.abs() < 1.0e-12);
        assert!(identical.reverse_kl.abs() < 1.0e-12);
        assert_eq!(identical.kl_floor, FISHER_RAO_KL_FLOOR);
        assert!(identical.fisher_rao_distance.abs() < 1.0e-12);
        assert!((identical.bhattacharyya_coefficient - 1.0).abs() < 1.0e-12);
        assert_eq!(identical.sectional_curvature, Some(0.25));

        let orthogonal = compare_fisher_rao(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0])
            .expect("valid boundary comparison");
        assert!(orthogonal.forward_kl > 0.0);
        assert_eq!(orthogonal.forward_kl, orthogonal.reverse_kl);
        assert!((orthogonal.fisher_rao_distance - std::f64::consts::PI).abs() < 1.0e-12);
        assert_eq!(orthogonal.bhattacharyya_coefficient, 0.0);
    }

    #[test]
    fn fisher_information_is_exact_on_the_interior_and_singular_on_the_boundary() {
        let diagonal = fisher_information_diagonal(&[0.25, 0.75]).expect("interior metric");
        assert_eq!(diagonal, vec![4.0, 4.0 / 3.0]);

        assert!(matches!(
            fisher_information_diagonal(&[1.0, 0.0]),
            Err(ConceptDiffusionError::NonPositive {
                field: "fisher_probability",
                ..
            })
        ));
    }

    #[test]
    fn request_deserialization_rejects_contract_drift() {
        let error = serde_json::from_str::<ConceptDiffusionRequest>(
            r#"{"tags":["left"],"state":[1.0],"affinitty":[[0.0]]}"#,
        )
        .expect_err("unknown fields must fail closed");

        assert!(error.to_string().contains("unknown field"));
    }
}
