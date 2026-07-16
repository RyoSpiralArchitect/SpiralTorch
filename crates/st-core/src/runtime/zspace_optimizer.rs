// SPDX-License-Identifier: AGPL-3.0-or-later

//! Canonical Z-space meta-optimizer semantics.
//!
//! This module owns the observed-resource objective, periodic fractional
//! Sobolev regularizer, Topos control resolution, gradient projection, and
//! Adam state transition used by Z-space clients. Python and WASM bindings
//! transport requests and reports; they must not reconstruct these rules.

use rustfft::{num_complex::Complex, FftPlanner};
use serde::{Deserialize, Serialize};
use st_tensor::{
    topos_optimizer_gradient_bias_amplitude, topos_optimizer_gradient_bias_basis,
    topos_optimizer_gradient_clip_threshold, TOPOS_OPTIMIZER_GRADIENT_BIAS_NORMALIZATION,
    TOPOS_OPTIMIZER_GRADIENT_BIAS_RULE, TOPOS_OPTIMIZER_GRADIENT_CLIP_NORMALIZATION,
    TOPOS_OPTIMIZER_GRADIENT_CLIP_RULE,
};
use std::collections::BTreeMap;
use thiserror::Error;

pub const ZSPACE_META_OPTIMIZER_CONTRACT_VERSION: &str = "spiraltorch.zspace_meta_optimizer.v2";
pub const ZSPACE_META_OPTIMIZER_KIND: &str = "spiraltorch.zspace_meta_optimizer";
pub const ZSPACE_META_OPTIMIZER_SEMANTIC_OWNER: &str = "st-core::runtime::zspace_optimizer";
pub const ZSPACE_META_OPTIMIZER_SEMANTIC_BACKEND: &str = "rust";
pub const ZSPACE_PARAMETER_CONTROL_CONTRACT_VERSION: &str =
    "spiraltorch.zspace_parameter_control.v2";
pub const ZSPACE_PARAMETER_CONTROL_KIND: &str = "spiraltorch.zspace_parameter_control";
pub const ZSPACE_PARAMETER_CONTROL_SEMANTIC_OWNER: &str = "st-core::runtime::zspace_optimizer";
pub const ZSPACE_PARAMETER_CONTROL_SEMANTIC_BACKEND: &str = "rust";
pub const ZSPACE_PARAMETER_CONTROL_MIN_LEARNING_RATE_SCALE: f64 = 0.1;
pub const ZSPACE_PARAMETER_CONTROL_MAX_LEARNING_RATE_SCALE: f64 = 1.25;
/// Allocation and transform guard for state carried by untrusted clients.
pub const ZSPACE_META_OPTIMIZER_MAX_DIMENSION: usize = 4_096;
/// Largest step represented exactly by Python, JSON, and JavaScript clients.
pub const ZSPACE_META_OPTIMIZER_MAX_SAFE_STEP: u64 = 9_007_199_254_740_991;
pub const ZSPACE_META_OBJECTIVE_FORMULA: &str =
    "J_obs=sum_i(lambda_i*tanh(metric_i))+lambda_topos*tanh(topos_pressure)+lambda_frac_eff*R_alpha(z)";
pub const ZSPACE_FRACTIONAL_REGULARIZER_FORMULA: &str =
    "R_alpha(z)=mean_k((k/k_max)^(2*alpha)*|DFT(z)_k|^2),k=0..floor(n/2)";
pub const ZSPACE_ADAM_UPDATE_RULE: &str = "z_next=z-lr_eff*m_hat/(sqrt(v_hat)+epsilon)";
pub const ZSPACE_GRADIENT_RULE: &str =
    "g=clip_rms(bias_rms(project(tanh(g_observed))+lambda_frac_eff*normalized(dR_alpha/dz)))";

const DERIVED_TOLERANCE: f64 = 128.0 * f64::EPSILON;

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GradientProjection {
    /// Repeat short gradients and truncate long gradients to the Z dimension.
    #[default]
    TileOrTruncate,
    /// Require the observed gradient to have exactly the Z dimension.
    Exact,
}

/// Weights applied to the observed objective and fractional regularizer.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ZSpaceMetaOptimizerWeights {
    pub speed: f64,
    pub memory: f64,
    pub stability: f64,
    pub fractional: f64,
    #[serde(alias = "drs")]
    pub drift_response: f64,
}

impl Default for ZSpaceMetaOptimizerWeights {
    fn default() -> Self {
        Self {
            speed: 0.5,
            memory: 0.3,
            stability: 0.2,
            fractional: 0.1,
            drift_response: 0.0,
        }
    }
}

/// Complete semantic configuration for a Z-space meta-optimizer.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ZSpaceMetaOptimizerConfig {
    pub dimension: usize,
    #[serde(alias = "alpha")]
    pub fractional_order: f64,
    pub weights: ZSpaceMetaOptimizerWeights,
    #[serde(alias = "lr")]
    pub learning_rate: f64,
    #[serde(alias = "beta1")]
    pub first_moment_decay: f64,
    #[serde(alias = "beta2")]
    pub second_moment_decay: f64,
    #[serde(alias = "eps")]
    pub epsilon: f64,
    pub topos_control_gain: f64,
    pub gradient_projection: GradientProjection,
}

impl Default for ZSpaceMetaOptimizerConfig {
    fn default() -> Self {
        Self {
            dimension: 4,
            fractional_order: 0.35,
            weights: ZSpaceMetaOptimizerWeights::default(),
            learning_rate: 1.0e-2,
            first_moment_decay: 0.9,
            second_moment_decay: 0.999,
            epsilon: 1.0e-8,
            topos_control_gain: 0.0,
            gradient_projection: GradientProjection::TileOrTruncate,
        }
    }
}

impl ZSpaceMetaOptimizerConfig {
    pub fn validate(&self) -> Result<(), ZSpaceMetaOptimizerError> {
        if self.dimension == 0 {
            return Err(ZSpaceMetaOptimizerError::InvalidDimension {
                dimension: self.dimension,
            });
        }
        if self.dimension > ZSPACE_META_OPTIMIZER_MAX_DIMENSION {
            return Err(ZSpaceMetaOptimizerError::DimensionLimit {
                dimension: self.dimension,
                maximum: ZSPACE_META_OPTIMIZER_MAX_DIMENSION,
            });
        }
        require_positive("config.fractional_order", self.fractional_order)?;
        require_non_negative("config.weights.speed", self.weights.speed)?;
        require_non_negative("config.weights.memory", self.weights.memory)?;
        require_non_negative("config.weights.stability", self.weights.stability)?;
        require_non_negative("config.weights.fractional", self.weights.fractional)?;
        require_non_negative("config.weights.drift_response", self.weights.drift_response)?;
        require_positive("config.learning_rate", self.learning_rate)?;
        require_decay("config.first_moment_decay", self.first_moment_decay)?;
        require_decay("config.second_moment_decay", self.second_moment_decay)?;
        require_positive("config.epsilon", self.epsilon)?;
        require_non_negative("config.topos_control_gain", self.topos_control_gain)?;
        Ok(())
    }
}

/// Adam state and Z coordinates. The second moment is always non-negative.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ZSpaceMetaOptimizerState {
    pub z: Vec<f64>,
    #[serde(alias = "moment")]
    pub first_moment: Vec<f64>,
    #[serde(alias = "velocity")]
    pub second_moment: Vec<f64>,
    pub step: u64,
}

impl ZSpaceMetaOptimizerState {
    pub fn zeros(dimension: usize) -> Self {
        Self {
            z: vec![0.0; dimension],
            first_moment: vec![0.0; dimension],
            second_moment: vec![0.0; dimension],
            step: 0,
        }
    }

    fn validate_values(&self) -> Result<(), ZSpaceMetaOptimizerError> {
        require_finite_vector("state.z", &self.z)?;
        require_finite_vector("state.first_moment", &self.first_moment)?;
        require_finite_vector("state.second_moment", &self.second_moment)?;
        for (index, value) in self.second_moment.iter().copied().enumerate() {
            if value < 0.0 {
                return Err(ZSpaceMetaOptimizerError::NegativeSecondMoment { index, value });
            }
        }
        if self.step > ZSPACE_META_OPTIMIZER_MAX_SAFE_STEP {
            return Err(ZSpaceMetaOptimizerError::StepLimit {
                step: self.step,
                maximum: ZSPACE_META_OPTIMIZER_MAX_SAFE_STEP,
            });
        }
        Ok(())
    }

    fn validate(&self, dimension: usize) -> Result<(), ZSpaceMetaOptimizerError> {
        require_vector_length("state.z", &self.z, dimension)?;
        require_vector_length("state.first_moment", &self.first_moment, dimension)?;
        require_vector_length("state.second_moment", &self.second_moment, dimension)?;
        self.validate_values()
    }
}

/// One observed resource/gradient sample.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ZSpaceMetaObservation {
    pub speed: f64,
    pub memory: f64,
    pub stability: f64,
    #[serde(alias = "drs")]
    pub drift_response: f64,
    pub gradient: Vec<f64>,
    /// Flattened telemetry. Topos controls use the `topos.*` namespace.
    pub telemetry: BTreeMap<String, f64>,
}

impl ZSpaceMetaObservation {
    fn validate(&self) -> Result<(), ZSpaceMetaOptimizerError> {
        require_finite("observation.speed", self.speed)?;
        require_finite("observation.memory", self.memory)?;
        require_finite("observation.stability", self.stability)?;
        require_finite("observation.drift_response", self.drift_response)?;
        require_finite_vector("observation.gradient", &self.gradient)?;
        for (key, value) in &self.telemetry {
            require_finite(&format!("observation.telemetry.{key}"), *value)?;
        }
        Ok(())
    }
}

/// Stateless transition request shared by native clients.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceMetaOptimizerStepRequest {
    pub config: ZSpaceMetaOptimizerConfig,
    pub state: ZSpaceMetaOptimizerState,
    pub observation: ZSpaceMetaObservation,
}

/// Restore request used to validate or dimension-coerce checkpoints in Rust.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceMetaOptimizerRestoreRequest {
    pub config: ZSpaceMetaOptimizerConfig,
    pub state: ZSpaceMetaOptimizerState,
    #[serde(default = "default_true")]
    pub strict: bool,
}

fn default_true() -> bool {
    true
}

/// Validated checkpoint returned by initialization and restore operations.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceMetaOptimizerCheckpoint {
    pub contract_version: &'static str,
    pub kind: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub config: ZSpaceMetaOptimizerConfig,
    pub state: ZSpaceMetaOptimizerState,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceObjectiveReport {
    pub normalized_speed: f64,
    pub normalized_memory: f64,
    pub normalized_stability: f64,
    pub normalized_drift_response: f64,
    pub speed_term: f64,
    pub memory_term: f64,
    pub stability_term: f64,
    pub drift_response_term: f64,
    pub topos_term: f64,
    pub fractional_term: f64,
    pub observed_resource_penalty: f64,
    pub objective_before: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FractionalRegularizerReport {
    pub formula: &'static str,
    pub order: f64,
    pub signal_length: usize,
    pub spectral_bins: usize,
    pub energy: f64,
    pub raw_gradient: Vec<f64>,
    pub gradient_normalization_scale: f64,
    pub normalized_gradient: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ToposOptimizerControlReport {
    pub present: bool,
    pub active: bool,
    pub closure_pressure: f64,
    pub depth_pressure: f64,
    pub volume_pressure: f64,
    pub guard_strength: f64,
    pub step_damping: f64,
    pub sampling_focus: f64,
    pub openness: f64,
    pub exploration_hint: f64,
    pub pressure: f64,
    pub learning_rate_hint: f64,
    pub learning_rate_scale: f64,
    pub effective_learning_rate: f64,
    pub clip_hint: f64,
    pub clip_scale: f64,
    pub gradient_clip_rule: &'static str,
    pub gradient_clip_normalization: &'static str,
    pub biased_gradient_rms: f64,
    pub gradient_clip_threshold: Option<f64>,
    pub regularization_hint: f64,
    pub regularization_scale: f64,
    pub effective_fractional_weight: f64,
    pub gradient_bias_rule: &'static str,
    pub gradient_bias_normalization: &'static str,
    pub gradient_bias_scale: f64,
    pub gradient_bias_basis: Vec<f64>,
    pub raw_gradient_rms: f64,
    pub gradient_bias_amplitude: f64,
    pub gradient_bias: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceGradientReport {
    pub rule: &'static str,
    pub source_dimension: usize,
    pub target_dimension: usize,
    pub projection: GradientProjection,
    pub observed: Vec<f64>,
    pub projected_normalized: Vec<f64>,
    pub before_clip: Vec<f64>,
    pub applied: Vec<f64>,
    pub clipped_values: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AdamTransitionReport {
    pub rule: &'static str,
    pub step: u64,
    pub first_moment_bias_correction: f64,
    pub second_moment_bias_correction: f64,
    pub effective_learning_rate: f64,
    /// Additive parameter delta (`z_next - z_before`).
    pub parameter_delta: Vec<f64>,
}

/// Full audit of one validated transition; stateful clients commit `state_after`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceMetaOptimizerStepReport {
    pub contract_version: &'static str,
    pub kind: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub objective_formula: &'static str,
    pub transition_validated: bool,
    pub config: ZSpaceMetaOptimizerConfig,
    pub observation: ZSpaceMetaObservation,
    pub objective: ZSpaceObjectiveReport,
    pub fractional_regularizer: FractionalRegularizerReport,
    pub topos_control: ToposOptimizerControlReport,
    pub gradient: ZSpaceGradientReport,
    pub adam: AdamTransitionReport,
    pub state_before: ZSpaceMetaOptimizerState,
    pub state_after: ZSpaceMetaOptimizerState,
}

/// Validated control-plane instruction emitted from one meta-optimizer report.
///
/// This is intentionally narrower than [`ZSpaceMetaOptimizerStepReport`]. The
/// latent gradient, clipping, regularization, and Topos bias remain owned by
/// the latent transition; only its bounded absolute learning-rate scale crosses
/// into model-parameter optimization.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ZSpaceParameterControl {
    contract_version: &'static str,
    kind: &'static str,
    semantic_owner: &'static str,
    semantic_backend: &'static str,
    source_contract_version: &'static str,
    source_semantic_owner: &'static str,
    source_step: u64,
    absolute_learning_rate_scale: f64,
    source_learning_rate: f64,
    source_effective_learning_rate: f64,
}

impl ZSpaceParameterControl {
    pub fn contract_version(&self) -> &'static str {
        self.contract_version
    }

    pub fn kind(&self) -> &'static str {
        self.kind
    }

    pub fn semantic_owner(&self) -> &'static str {
        self.semantic_owner
    }

    pub fn semantic_backend(&self) -> &'static str {
        self.semantic_backend
    }

    pub fn source_contract_version(&self) -> &'static str {
        self.source_contract_version
    }

    pub fn source_semantic_owner(&self) -> &'static str {
        self.source_semantic_owner
    }

    pub fn source_step(&self) -> u64 {
        self.source_step
    }

    pub fn absolute_learning_rate_scale(&self) -> f64 {
        self.absolute_learning_rate_scale
    }

    pub fn source_learning_rate(&self) -> f64 {
        self.source_learning_rate
    }

    pub fn source_effective_learning_rate(&self) -> f64 {
        self.source_effective_learning_rate
    }
}

#[derive(Debug, Deserialize)]
struct ParameterControlReportProof {
    contract_version: String,
    kind: String,
    semantic_owner: String,
    semantic_backend: String,
    transition_validated: bool,
    config: ZSpaceMetaOptimizerConfig,
    observation: ZSpaceMetaObservation,
    topos_control: ParameterControlToposProof,
    adam: ParameterControlAdamProof,
    state_before: ParameterControlStateProof,
    state_after: ParameterControlStateProof,
}

#[derive(Debug, Deserialize)]
struct ParameterControlToposProof {
    learning_rate_scale: f64,
    effective_learning_rate: f64,
}

#[derive(Debug, Deserialize)]
struct ParameterControlAdamProof {
    step: u64,
    effective_learning_rate: f64,
}

#[derive(Debug, Deserialize)]
struct ParameterControlStateProof {
    step: u64,
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum ZSpaceMetaOptimizerError {
    #[error("Z-space optimizer dimension must be positive, received {dimension}")]
    InvalidDimension { dimension: usize },
    #[error("Z-space optimizer dimension {dimension} exceeds maximum {maximum}")]
    DimensionLimit { dimension: usize, maximum: usize },
    #[error("{field} must be finite, received {value}")]
    NonFinite { field: String, value: f64 },
    #[error("{field} must be non-negative, received {value}")]
    Negative { field: String, value: f64 },
    #[error("{field} must be positive, received {value}")]
    NonPositive { field: String, value: f64 },
    #[error("{field} must satisfy 0 <= decay < 1, received {value}")]
    InvalidDecay { field: String, value: f64 },
    #[error("{field} must contain {expected} values, received {actual}")]
    StateLength {
        field: String,
        expected: usize,
        actual: usize,
    },
    #[error("observed gradient must contain {expected} values, received {actual}")]
    GradientLength { expected: usize, actual: usize },
    #[error("state.second_moment[{index}] must be non-negative, received {value}")]
    NegativeSecondMoment { index: usize, value: f64 },
    #[error("optimizer step {step} exceeds cross-client maximum {maximum}")]
    StepLimit { step: u64, maximum: u64 },
    #[error("derived {field} is not finite")]
    DerivedNonFinite { field: String },
    #[error("derived {field} must be positive, received {value}")]
    DerivedNonPositive { field: String, value: f64 },
    #[error(
        "derived invariant {field} exceeded tolerance: residual={residual}, tolerance={tolerance}"
    )]
    InvariantViolation {
        field: String,
        residual: f64,
        tolerance: f64,
    },
    #[error("malformed Z-space meta-optimizer report: {message}")]
    MalformedReport { message: String },
    #[error("untrusted {field}: expected {expected}, received {actual}")]
    ContractMismatch {
        field: &'static str,
        expected: &'static str,
        actual: String,
    },
    #[error("Z-space meta-optimizer report did not validate its transition")]
    UnvalidatedTransition,
    #[error(
        "inconsistent Z-space meta-optimizer steps: before={before}, after={after}, adam={adam}"
    )]
    InconsistentReportStep { before: u64, after: u64, adam: u64 },
    #[error(
        "parameter learning-rate scale must remain in [{minimum}, {maximum}], received {value}"
    )]
    ParameterControlScaleOutOfRange {
        value: f64,
        minimum: f64,
        maximum: f64,
    },
}

/// Stateful Rust API for direct use without a language binding.
#[derive(Clone, Debug)]
pub struct ZSpaceMetaOptimizer {
    config: ZSpaceMetaOptimizerConfig,
    state: ZSpaceMetaOptimizerState,
    last_report: Option<ZSpaceMetaOptimizerStepReport>,
}

impl ZSpaceMetaOptimizer {
    pub fn new(config: ZSpaceMetaOptimizerConfig) -> Result<Self, ZSpaceMetaOptimizerError> {
        let checkpoint = initialize_zspace_meta_optimizer(config)?;
        Ok(Self {
            config: checkpoint.config,
            state: checkpoint.state,
            last_report: None,
        })
    }

    pub fn from_checkpoint(
        config: ZSpaceMetaOptimizerConfig,
        state: ZSpaceMetaOptimizerState,
    ) -> Result<Self, ZSpaceMetaOptimizerError> {
        let checkpoint = restore_zspace_meta_optimizer(ZSpaceMetaOptimizerRestoreRequest {
            config,
            state,
            strict: true,
        })?;
        Ok(Self {
            config: checkpoint.config,
            state: checkpoint.state,
            last_report: None,
        })
    }

    pub fn config(&self) -> &ZSpaceMetaOptimizerConfig {
        &self.config
    }

    pub fn state(&self) -> &ZSpaceMetaOptimizerState {
        &self.state
    }

    pub fn last_report(&self) -> Option<&ZSpaceMetaOptimizerStepReport> {
        self.last_report.as_ref()
    }

    pub fn checkpoint(&self) -> ZSpaceMetaOptimizerCheckpoint {
        checkpoint(self.config.clone(), self.state.clone())
    }

    pub fn reset(&mut self) {
        self.state = ZSpaceMetaOptimizerState::zeros(self.config.dimension);
        self.last_report = None;
    }

    /// Computes and validates the complete transition before mutating state.
    pub fn try_step(
        &mut self,
        observation: ZSpaceMetaObservation,
    ) -> Result<&ZSpaceMetaOptimizerStepReport, ZSpaceMetaOptimizerError> {
        let report = transition_zspace_meta_optimizer(ZSpaceMetaOptimizerStepRequest {
            config: self.config.clone(),
            state: self.state.clone(),
            observation,
        })?;
        self.state = report.state_after.clone();
        Ok(self.last_report.insert(report))
    }
}

pub fn initialize_zspace_meta_optimizer(
    config: ZSpaceMetaOptimizerConfig,
) -> Result<ZSpaceMetaOptimizerCheckpoint, ZSpaceMetaOptimizerError> {
    config.validate()?;
    Ok(checkpoint(
        config.clone(),
        ZSpaceMetaOptimizerState::zeros(config.dimension),
    ))
}

pub fn restore_zspace_meta_optimizer(
    request: ZSpaceMetaOptimizerRestoreRequest,
) -> Result<ZSpaceMetaOptimizerCheckpoint, ZSpaceMetaOptimizerError> {
    request.config.validate()?;
    // Non-strict restore may resize vectors, but it must never hide invalid
    // values in the discarded tail.
    request.state.validate_values()?;
    let dimension = request.config.dimension;
    let state = ZSpaceMetaOptimizerState {
        z: restore_vector("state.z", request.state.z, dimension, request.strict)?,
        first_moment: restore_vector(
            "state.first_moment",
            request.state.first_moment,
            dimension,
            request.strict,
        )?,
        second_moment: restore_vector(
            "state.second_moment",
            request.state.second_moment,
            dimension,
            request.strict,
        )?,
        step: request.state.step,
    };
    state.validate(dimension)?;
    Ok(checkpoint(request.config, state))
}

pub fn transition_zspace_meta_optimizer(
    request: ZSpaceMetaOptimizerStepRequest,
) -> Result<ZSpaceMetaOptimizerStepReport, ZSpaceMetaOptimizerError> {
    request.config.validate()?;
    request.state.validate(request.config.dimension)?;
    request.observation.validate()?;

    let state_before = request.state.clone();
    let fractional = fractional_regularizer(&state_before.z, request.config.fractional_order)?;
    let mut topos = resolve_topos_control(
        &request.config,
        &request.observation.telemetry,
        request.config.weights.fractional,
    )?;
    let projected = project_observed_gradient(
        &request.observation.gradient,
        request.config.dimension,
        request.config.gradient_projection,
    )?;

    let mut regularized_gradient = Vec::with_capacity(request.config.dimension);
    for (index, projected_gradient) in projected.iter().copied().enumerate() {
        let regularized = checked_add(
            &format!("gradient.regularized[{index}]"),
            projected_gradient,
            checked_mul(
                &format!("gradient.fractional[{index}]"),
                topos.effective_fractional_weight,
                fractional.normalized_gradient[index],
            )?,
        )?;
        regularized_gradient.push(regularized);
    }
    let raw_gradient_rms = gradient_rms("gradient.raw_rms", &regularized_gradient)?;
    let gradient_bias_amplitude =
        topos_optimizer_gradient_bias_amplitude(raw_gradient_rms, topos.gradient_bias_scale);
    require_derived_finite("gradient.topos_bias_amplitude", gradient_bias_amplitude)?;
    let gradient_bias = topos
        .gradient_bias_basis
        .iter()
        .cycle()
        .take(request.config.dimension)
        .enumerate()
        .map(|(index, basis)| {
            checked_mul(
                &format!("gradient.topos_bias[{index}]"),
                gradient_bias_amplitude,
                *basis,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut before_clip = Vec::with_capacity(request.config.dimension);
    for (index, (regularized, bias)) in regularized_gradient
        .iter()
        .copied()
        .zip(gradient_bias.iter().copied())
        .enumerate()
    {
        before_clip.push(checked_add(
            &format!("gradient.topos_biased[{index}]"),
            regularized,
            bias,
        )?);
    }
    let biased_gradient_rms = gradient_rms("gradient.biased_rms", &before_clip)?;
    let gradient_clip_threshold =
        topos_optimizer_gradient_clip_threshold(biased_gradient_rms, topos.clip_scale);
    if let Some(threshold) = gradient_clip_threshold {
        require_derived_finite("gradient.clip_threshold", threshold)?;
    }
    topos.raw_gradient_rms = raw_gradient_rms;
    topos.gradient_bias_amplitude = gradient_bias_amplitude;
    topos.gradient_bias = gradient_bias;
    topos.biased_gradient_rms = biased_gradient_rms;
    topos.gradient_clip_threshold = gradient_clip_threshold;

    let mut applied = before_clip.clone();
    let mut clipped_values = 0;
    if let Some(threshold) = gradient_clip_threshold {
        for value in &mut applied {
            let clipped = value.clamp(-threshold, threshold);
            if clipped != *value {
                clipped_values += 1;
            }
            *value = clipped;
        }
    }
    require_finite_vector("gradient.applied", &applied)?;

    let next_step =
        state_before
            .step
            .checked_add(1)
            .ok_or(ZSpaceMetaOptimizerError::StepLimit {
                step: u64::MAX,
                maximum: ZSPACE_META_OPTIMIZER_MAX_SAFE_STEP,
            })?;
    if next_step > ZSPACE_META_OPTIMIZER_MAX_SAFE_STEP {
        return Err(ZSpaceMetaOptimizerError::StepLimit {
            step: next_step,
            maximum: ZSPACE_META_OPTIMIZER_MAX_SAFE_STEP,
        });
    }
    let first_bias = bias_correction(request.config.first_moment_decay, next_step);
    let second_bias = bias_correction(request.config.second_moment_decay, next_step);
    require_derived_positive("adam.first_moment_bias_correction", first_bias)?;
    require_derived_positive("adam.second_moment_bias_correction", second_bias)?;

    let mut next_first = Vec::with_capacity(request.config.dimension);
    let mut next_second = Vec::with_capacity(request.config.dimension);
    let mut next_z = Vec::with_capacity(request.config.dimension);
    let mut parameter_delta = Vec::with_capacity(request.config.dimension);
    for (index, gradient) in applied.iter().copied().enumerate() {
        let first = checked_add(
            &format!("adam.first_moment[{index}]"),
            checked_mul(
                &format!("adam.first_decay[{index}]"),
                request.config.first_moment_decay,
                state_before.first_moment[index],
            )?,
            checked_mul(
                &format!("adam.first_gradient[{index}]"),
                1.0 - request.config.first_moment_decay,
                gradient,
            )?,
        )?;
        let gradient_squared = checked_mul(
            &format!("adam.gradient_squared[{index}]"),
            gradient,
            gradient,
        )?;
        let second = checked_add(
            &format!("adam.second_moment[{index}]"),
            checked_mul(
                &format!("adam.second_decay[{index}]"),
                request.config.second_moment_decay,
                state_before.second_moment[index],
            )?,
            checked_mul(
                &format!("adam.second_gradient[{index}]"),
                1.0 - request.config.second_moment_decay,
                gradient_squared,
            )?,
        )?;
        if second < 0.0 {
            return Err(ZSpaceMetaOptimizerError::NegativeSecondMoment {
                index,
                value: second,
            });
        }
        let first_hat = checked_div(&format!("adam.first_hat[{index}]"), first, first_bias)?;
        let second_hat = checked_div(&format!("adam.second_hat[{index}]"), second, second_bias)?;
        if second_hat < 0.0 {
            return Err(ZSpaceMetaOptimizerError::NegativeSecondMoment {
                index,
                value: second_hat,
            });
        }
        let denominator = checked_add(
            &format!("adam.denominator[{index}]"),
            second_hat.sqrt(),
            request.config.epsilon,
        )?;
        require_derived_positive(&format!("adam.denominator[{index}]"), denominator)?;
        let delta = -checked_div(
            &format!("adam.parameter_delta[{index}]"),
            checked_mul(
                &format!("adam.scaled_first_hat[{index}]"),
                topos.effective_learning_rate,
                first_hat,
            )?,
            denominator,
        )?;
        let z = checked_add(
            &format!("state_after.z[{index}]"),
            state_before.z[index],
            delta,
        )?;
        next_first.push(first);
        next_second.push(second);
        parameter_delta.push(delta);
        next_z.push(z);
    }

    let state_after = ZSpaceMetaOptimizerState {
        z: next_z,
        first_moment: next_first,
        second_moment: next_second,
        step: next_step,
    };
    state_after.validate(request.config.dimension)?;

    let objective = objective_report(&request.config, &request.observation, &fractional, &topos)?;
    verify_parameter_delta(&state_before.z, &state_after.z, &parameter_delta)?;
    let effective_learning_rate = topos.effective_learning_rate;

    Ok(ZSpaceMetaOptimizerStepReport {
        contract_version: ZSPACE_META_OPTIMIZER_CONTRACT_VERSION,
        kind: ZSPACE_META_OPTIMIZER_KIND,
        semantic_owner: ZSPACE_META_OPTIMIZER_SEMANTIC_OWNER,
        semantic_backend: ZSPACE_META_OPTIMIZER_SEMANTIC_BACKEND,
        objective_formula: ZSPACE_META_OBJECTIVE_FORMULA,
        transition_validated: true,
        config: request.config.clone(),
        observation: request.observation.clone(),
        objective,
        fractional_regularizer: fractional,
        topos_control: topos,
        gradient: ZSpaceGradientReport {
            rule: ZSPACE_GRADIENT_RULE,
            source_dimension: request.observation.gradient.len(),
            target_dimension: request.config.dimension,
            projection: request.config.gradient_projection,
            observed: request.observation.gradient.clone(),
            projected_normalized: projected,
            before_clip,
            applied,
            clipped_values,
        },
        adam: AdamTransitionReport {
            rule: ZSPACE_ADAM_UPDATE_RULE,
            step: next_step,
            first_moment_bias_correction: first_bias,
            second_moment_bias_correction: second_bias,
            effective_learning_rate,
            parameter_delta,
        },
        state_before,
        state_after,
    })
}

/// Extracts the model-parameter control that is semantically safe to share
/// from a typed meta-optimizer report.
pub fn zspace_parameter_control_from_report(
    report: &ZSpaceMetaOptimizerStepReport,
) -> Result<ZSpaceParameterControl, ZSpaceMetaOptimizerError> {
    validate_parameter_control_fields(
        report.contract_version,
        report.kind,
        report.semantic_owner,
        report.semantic_backend,
        report.transition_validated,
        &report.config,
        &report.observation,
        report.topos_control.learning_rate_scale,
        report.topos_control.effective_learning_rate,
        report.adam.effective_learning_rate,
        report.state_before.step,
        report.state_after.step,
        report.adam.step,
    )
}

/// Validates a serialized meta-optimizer report before extracting parameter
/// control. Language bindings should pass the complete report here instead of
/// rebuilding the control from selected fields.
pub fn zspace_parameter_control_from_value(
    report: serde_json::Value,
) -> Result<ZSpaceParameterControl, ZSpaceMetaOptimizerError> {
    let proof: ParameterControlReportProof = serde_json::from_value(report).map_err(|error| {
        ZSpaceMetaOptimizerError::MalformedReport {
            message: error.to_string(),
        }
    })?;
    validate_parameter_control_fields(
        proof.contract_version.as_str(),
        proof.kind.as_str(),
        proof.semantic_owner.as_str(),
        proof.semantic_backend.as_str(),
        proof.transition_validated,
        &proof.config,
        &proof.observation,
        proof.topos_control.learning_rate_scale,
        proof.topos_control.effective_learning_rate,
        proof.adam.effective_learning_rate,
        proof.state_before.step,
        proof.state_after.step,
        proof.adam.step,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_parameter_control_fields(
    contract_version: &str,
    kind: &str,
    semantic_owner: &str,
    semantic_backend: &str,
    transition_validated: bool,
    config: &ZSpaceMetaOptimizerConfig,
    observation: &ZSpaceMetaObservation,
    absolute_learning_rate_scale: f64,
    topos_effective_learning_rate: f64,
    adam_effective_learning_rate: f64,
    state_before_step: u64,
    state_after_step: u64,
    adam_step: u64,
) -> Result<ZSpaceParameterControl, ZSpaceMetaOptimizerError> {
    require_contract_field(
        "report.contract_version",
        contract_version,
        ZSPACE_META_OPTIMIZER_CONTRACT_VERSION,
    )?;
    require_contract_field("report.kind", kind, ZSPACE_META_OPTIMIZER_KIND)?;
    require_contract_field(
        "report.semantic_owner",
        semantic_owner,
        ZSPACE_META_OPTIMIZER_SEMANTIC_OWNER,
    )?;
    require_contract_field(
        "report.semantic_backend",
        semantic_backend,
        ZSPACE_META_OPTIMIZER_SEMANTIC_BACKEND,
    )?;
    if !transition_validated {
        return Err(ZSpaceMetaOptimizerError::UnvalidatedTransition);
    }

    config.validate()?;
    observation.validate()?;
    let canonical_topos =
        resolve_topos_control(config, &observation.telemetry, config.weights.fractional)?;
    verify_scalar_invariant(
        "topos_control.learning_rate_scale",
        absolute_learning_rate_scale,
        canonical_topos.learning_rate_scale,
    )?;
    let source_learning_rate = config.learning_rate;
    require_positive("report.config.learning_rate", source_learning_rate)?;
    require_positive(
        "report.topos_control.learning_rate_scale",
        absolute_learning_rate_scale,
    )?;
    if !(ZSPACE_PARAMETER_CONTROL_MIN_LEARNING_RATE_SCALE
        ..=ZSPACE_PARAMETER_CONTROL_MAX_LEARNING_RATE_SCALE)
        .contains(&absolute_learning_rate_scale)
    {
        return Err(ZSpaceMetaOptimizerError::ParameterControlScaleOutOfRange {
            value: absolute_learning_rate_scale,
            minimum: ZSPACE_PARAMETER_CONTROL_MIN_LEARNING_RATE_SCALE,
            maximum: ZSPACE_PARAMETER_CONTROL_MAX_LEARNING_RATE_SCALE,
        });
    }

    let expected_step = state_before_step.checked_add(1).ok_or(
        ZSpaceMetaOptimizerError::InconsistentReportStep {
            before: state_before_step,
            after: state_after_step,
            adam: adam_step,
        },
    )?;
    if expected_step != state_after_step || state_after_step != adam_step {
        return Err(ZSpaceMetaOptimizerError::InconsistentReportStep {
            before: state_before_step,
            after: state_after_step,
            adam: adam_step,
        });
    }
    if state_after_step > ZSPACE_META_OPTIMIZER_MAX_SAFE_STEP {
        return Err(ZSpaceMetaOptimizerError::StepLimit {
            step: state_after_step,
            maximum: ZSPACE_META_OPTIMIZER_MAX_SAFE_STEP,
        });
    }

    let expected_effective_learning_rate = checked_mul(
        "parameter_control.effective_learning_rate",
        source_learning_rate,
        absolute_learning_rate_scale,
    )?;
    verify_scalar_invariant(
        "topos_control.effective_learning_rate",
        topos_effective_learning_rate,
        expected_effective_learning_rate,
    )?;
    verify_scalar_invariant(
        "adam.effective_learning_rate",
        adam_effective_learning_rate,
        expected_effective_learning_rate,
    )?;

    Ok(ZSpaceParameterControl {
        contract_version: ZSPACE_PARAMETER_CONTROL_CONTRACT_VERSION,
        kind: ZSPACE_PARAMETER_CONTROL_KIND,
        semantic_owner: ZSPACE_PARAMETER_CONTROL_SEMANTIC_OWNER,
        semantic_backend: ZSPACE_PARAMETER_CONTROL_SEMANTIC_BACKEND,
        source_contract_version: ZSPACE_META_OPTIMIZER_CONTRACT_VERSION,
        source_semantic_owner: ZSPACE_META_OPTIMIZER_SEMANTIC_OWNER,
        source_step: state_after_step,
        absolute_learning_rate_scale,
        source_learning_rate,
        source_effective_learning_rate: expected_effective_learning_rate,
    })
}

fn checkpoint(
    config: ZSpaceMetaOptimizerConfig,
    state: ZSpaceMetaOptimizerState,
) -> ZSpaceMetaOptimizerCheckpoint {
    ZSpaceMetaOptimizerCheckpoint {
        contract_version: ZSPACE_META_OPTIMIZER_CONTRACT_VERSION,
        kind: ZSPACE_META_OPTIMIZER_KIND,
        semantic_owner: ZSPACE_META_OPTIMIZER_SEMANTIC_OWNER,
        semantic_backend: ZSPACE_META_OPTIMIZER_SEMANTIC_BACKEND,
        config,
        state,
    }
}

fn restore_vector(
    field: &str,
    mut values: Vec<f64>,
    dimension: usize,
    strict: bool,
) -> Result<Vec<f64>, ZSpaceMetaOptimizerError> {
    if strict && values.len() != dimension {
        return Err(ZSpaceMetaOptimizerError::StateLength {
            field: field.to_owned(),
            expected: dimension,
            actual: values.len(),
        });
    }
    values.resize(dimension, 0.0);
    values.truncate(dimension);
    require_finite_vector(field, &values)?;
    Ok(values)
}

fn fractional_regularizer(
    z: &[f64],
    order: f64,
) -> Result<FractionalRegularizerReport, ZSpaceMetaOptimizerError> {
    require_positive("config.fractional_order", order)?;
    require_finite_vector("state.z", z)?;
    let signal_length = z.len();
    let spectral_bins = signal_length / 2 + 1;
    if signal_length <= 1 || spectral_bins <= 1 {
        return Ok(FractionalRegularizerReport {
            formula: ZSPACE_FRACTIONAL_REGULARIZER_FORMULA,
            order,
            signal_length,
            spectral_bins,
            energy: 0.0,
            raw_gradient: vec![0.0; signal_length],
            gradient_normalization_scale: 1.0,
            normalized_gradient: vec![0.0; signal_length],
        });
    }

    let mut spectrum = z
        .iter()
        .copied()
        .map(|value| Complex::new(value, 0.0))
        .collect::<Vec<_>>();
    let mut planner = FftPlanner::<f64>::new();
    let transform = planner.plan_fft_forward(signal_length);
    transform.process(&mut spectrum);
    for (frequency, coefficient) in spectrum.iter().take(spectral_bins).enumerate() {
        require_derived_finite(
            &format!("fractional.spectrum[{frequency}].real"),
            coefficient.re,
        )?;
        require_derived_finite(
            &format!("fractional.spectrum[{frequency}].imaginary"),
            coefficient.im,
        )?;
    }

    let denominator = spectral_bins as f64;
    let max_frequency = (spectral_bins - 1) as f64;
    let mut energy = 0.0;
    // FFT(weight * conj(spectrum)) evaluates the analytic one-sided
    // derivative at every signal coordinate in O(n log n).
    let mut gradient_spectrum = vec![Complex::new(0.0, 0.0); signal_length];
    for (frequency, coefficient) in spectrum.iter().copied().take(spectral_bins).enumerate() {
        let normalized_frequency = frequency as f64 / max_frequency;
        let weight = normalized_frequency.powf(2.0 * order);
        require_derived_finite("fractional.weight", weight)?;
        let magnitude_squared = checked_add(
            "fractional.magnitude_squared",
            checked_mul("fractional.real_squared", coefficient.re, coefficient.re)?,
            checked_mul(
                "fractional.imaginary_squared",
                coefficient.im,
                coefficient.im,
            )?,
        )?;
        energy = checked_add(
            "fractional.energy",
            energy,
            checked_mul("fractional.weighted_energy", weight, magnitude_squared)?,
        )?;
        gradient_spectrum[frequency] = Complex::new(
            checked_mul("fractional.gradient_spectrum.real", weight, coefficient.re)?,
            checked_mul(
                "fractional.gradient_spectrum.imaginary",
                -weight,
                coefficient.im,
            )?,
        );
    }
    energy = checked_div("fractional.energy_mean", energy, denominator)?;
    transform.process(&mut gradient_spectrum);
    let gradient_scale = 2.0 / denominator;
    let raw_gradient = gradient_spectrum
        .iter()
        .enumerate()
        .map(|(index, value)| {
            checked_mul(
                &format!("fractional.raw_gradient[{index}]"),
                gradient_scale,
                value.re,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let normalization_scale = raw_gradient
        .iter()
        .copied()
        .map(f64::abs)
        .fold(1.0, f64::max);
    require_derived_positive(
        "fractional.gradient_normalization_scale",
        normalization_scale,
    )?;
    let normalized_gradient = raw_gradient
        .iter()
        .enumerate()
        .map(|(index, value)| {
            checked_div(
                &format!("fractional.normalized_gradient[{index}]"),
                *value,
                normalization_scale,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(FractionalRegularizerReport {
        formula: ZSPACE_FRACTIONAL_REGULARIZER_FORMULA,
        order,
        signal_length,
        spectral_bins,
        energy,
        raw_gradient,
        gradient_normalization_scale: normalization_scale,
        normalized_gradient,
    })
}

fn resolve_topos_control(
    config: &ZSpaceMetaOptimizerConfig,
    telemetry: &BTreeMap<String, f64>,
    fractional_weight: f64,
) -> Result<ToposOptimizerControlReport, ZSpaceMetaOptimizerError> {
    let present = [
        "closure_pressure",
        "depth_pressure",
        "volume_pressure",
        "guard_strength",
        "training_hints.step_damping",
        "step_damping",
        "sampling_focus",
        "openness",
        "exploration_hint",
        "training_hints.learning_rate_scale",
        "learning_rate_scale",
        "training_hints.clip_scale",
        "clip_scale",
        "training_hints.regularization_scale",
        "regularization_scale",
        "training_hints.gradient_bias_scale",
        "gradient_bias_scale",
    ]
    .into_iter()
    .any(|key| topos_value(telemetry, key).is_some());
    let gain = config.topos_control_gain;
    let closure = topos_unit_value(telemetry, "closure_pressure").unwrap_or(0.0);
    let depth = topos_unit_value(telemetry, "depth_pressure").unwrap_or(0.0);
    let volume = topos_unit_value(telemetry, "volume_pressure").unwrap_or(0.0);
    let guard = topos_unit_value(telemetry, "guard_strength").unwrap_or(closure);
    let damping = topos_value(telemetry, "training_hints.step_damping")
        .or_else(|| topos_value(telemetry, "step_damping"))
        .map(|value| value.clamp(0.0, 1.0))
        .unwrap_or(checked_mul("topos.default_step_damping", closure, guard)?);
    let focus = topos_unit_value(telemetry, "sampling_focus").unwrap_or(guard);
    let openness =
        topos_unit_value(telemetry, "openness").unwrap_or((1.0 - closure).clamp(0.0, 1.0));
    let exploration = topos_unit_value(telemetry, "exploration_hint").unwrap_or(0.0);
    let raw_learning_rate_hint = topos_value(telemetry, "training_hints.learning_rate_scale")
        .or_else(|| topos_value(telemetry, "learning_rate_scale"))
        .unwrap_or(1.0);
    let raw_clip_hint = topos_value(telemetry, "training_hints.clip_scale")
        .or_else(|| topos_value(telemetry, "clip_scale"))
        .unwrap_or(1.0);
    let raw_regularization_hint = topos_value(telemetry, "training_hints.regularization_scale")
        .or_else(|| topos_value(telemetry, "regularization_scale"))
        .unwrap_or(1.0);
    let raw_bias_hint = topos_value(telemetry, "training_hints.gradient_bias_scale")
        .or_else(|| topos_value(telemetry, "gradient_bias_scale"))
        .unwrap_or(0.05);

    let pressure = if present {
        closure
            .max(depth)
            .max(volume)
            .max(guard)
            .max(damping)
            .max(focus)
    } else {
        0.0
    };
    let active = present && gain > 0.0;
    let learning_rate_hint = raw_learning_rate_hint.clamp(0.1, 1.25);
    let learning_rate_scale = if active {
        blend_topos_scale(
            "topos.learning_rate_scale",
            gain,
            learning_rate_hint,
            0.1,
            1.25,
        )?
    } else {
        1.0
    };
    let effective_learning_rate = checked_mul(
        "topos.effective_learning_rate",
        config.learning_rate,
        learning_rate_scale,
    )?;
    require_derived_positive("topos.effective_learning_rate", effective_learning_rate)?;

    let clip_hint = raw_clip_hint.clamp(0.25, 1.0);
    let clip_scale = if active {
        blend_topos_scale("topos.clip_scale", gain, clip_hint, 0.25, 1.0)?
    } else {
        1.0
    };
    let regularization_hint = raw_regularization_hint.clamp(0.25, 4.0);
    let regularization_scale = if active {
        blend_topos_scale(
            "topos.regularization_scale",
            gain,
            regularization_hint,
            0.25,
            4.0,
        )?
    } else {
        1.0
    };
    let effective_fractional_weight = checked_mul(
        "topos.effective_fractional_weight",
        fractional_weight,
        regularization_scale,
    )?;
    let gradient_bias_scale = if active {
        checked_mul(
            "topos.gradient_bias_scale",
            raw_bias_hint.clamp(0.0, 0.35),
            gain,
        )?
    } else {
        0.0
    };
    let basis = topos_optimizer_gradient_bias_basis(
        closure,
        volume,
        depth,
        guard,
        damping,
        focus,
        learning_rate_hint,
        regularization_scale,
        openness,
        exploration,
    );

    Ok(ToposOptimizerControlReport {
        present,
        active,
        closure_pressure: closure,
        depth_pressure: depth,
        volume_pressure: volume,
        guard_strength: guard,
        step_damping: damping,
        sampling_focus: focus,
        openness,
        exploration_hint: exploration,
        pressure,
        learning_rate_hint,
        learning_rate_scale,
        effective_learning_rate,
        clip_hint,
        clip_scale,
        gradient_clip_rule: TOPOS_OPTIMIZER_GRADIENT_CLIP_RULE,
        gradient_clip_normalization: TOPOS_OPTIMIZER_GRADIENT_CLIP_NORMALIZATION,
        biased_gradient_rms: 0.0,
        gradient_clip_threshold: None,
        regularization_hint,
        regularization_scale,
        effective_fractional_weight,
        gradient_bias_rule: TOPOS_OPTIMIZER_GRADIENT_BIAS_RULE,
        gradient_bias_normalization: TOPOS_OPTIMIZER_GRADIENT_BIAS_NORMALIZATION,
        gradient_bias_scale,
        gradient_bias_basis: basis.to_vec(),
        raw_gradient_rms: 0.0,
        gradient_bias_amplitude: 0.0,
        gradient_bias: vec![0.0; config.dimension],
    })
}

fn topos_value(telemetry: &BTreeMap<String, f64>, key: &str) -> Option<f64> {
    telemetry.get(&format!("topos.{key}")).copied()
}

fn topos_unit_value(telemetry: &BTreeMap<String, f64>, key: &str) -> Option<f64> {
    topos_value(telemetry, key).map(|value| value.clamp(0.0, 1.0))
}

fn blend_topos_scale(
    field: &str,
    gain: f64,
    hint: f64,
    minimum: f64,
    maximum: f64,
) -> Result<f64, ZSpaceMetaOptimizerError> {
    Ok(checked_add(field, 1.0, checked_mul(field, gain, hint - 1.0)?)?.clamp(minimum, maximum))
}

fn project_observed_gradient(
    gradient: &[f64],
    dimension: usize,
    projection: GradientProjection,
) -> Result<Vec<f64>, ZSpaceMetaOptimizerError> {
    if projection == GradientProjection::Exact && gradient.len() != dimension {
        return Err(ZSpaceMetaOptimizerError::GradientLength {
            expected: dimension,
            actual: gradient.len(),
        });
    }
    if gradient.is_empty() {
        return Ok(vec![0.0; dimension]);
    }
    Ok((0..dimension)
        .map(|index| gradient[index % gradient.len()].tanh())
        .collect())
}

fn gradient_rms(field: &str, gradient: &[f64]) -> Result<f64, ZSpaceMetaOptimizerError> {
    if gradient.is_empty() {
        return Ok(0.0);
    }
    let sum_squares =
        gradient
            .iter()
            .copied()
            .enumerate()
            .try_fold(0.0, |sum, (index, value)| {
                checked_add(
                    field,
                    sum,
                    checked_mul(&format!("{field}.square[{index}]"), value, value)?,
                )
            })?;
    let mean_square = checked_div(field, sum_squares, gradient.len() as f64)?;
    let rms = mean_square.sqrt();
    require_derived_finite(field, rms)?;
    Ok(rms)
}

fn objective_report(
    config: &ZSpaceMetaOptimizerConfig,
    observation: &ZSpaceMetaObservation,
    fractional: &FractionalRegularizerReport,
    topos: &ToposOptimizerControlReport,
) -> Result<ZSpaceObjectiveReport, ZSpaceMetaOptimizerError> {
    let normalized_speed = observation.speed.tanh();
    let normalized_memory = observation.memory.tanh();
    let normalized_stability = observation.stability.tanh();
    let normalized_drift_response = observation.drift_response.tanh();
    let speed_term = checked_mul("objective.speed", config.weights.speed, normalized_speed)?;
    let memory_term = checked_mul("objective.memory", config.weights.memory, normalized_memory)?;
    let stability_term = checked_mul(
        "objective.stability",
        config.weights.stability,
        normalized_stability,
    )?;
    let drift_response_term = checked_mul(
        "objective.drift_response",
        config.weights.drift_response,
        normalized_drift_response,
    )?;
    let topos_term = checked_mul(
        "objective.topos",
        config.topos_control_gain,
        topos.pressure.tanh(),
    )?;
    let fractional_term = checked_mul(
        "objective.fractional",
        topos.effective_fractional_weight,
        fractional.energy,
    )?;
    let observed_resource_penalty = checked_sum(
        "objective.observed_resource_penalty",
        [
            speed_term,
            memory_term,
            stability_term,
            drift_response_term,
            topos_term,
        ],
    )?;
    let objective_before = checked_add(
        "objective.before",
        observed_resource_penalty,
        fractional_term,
    )?;
    Ok(ZSpaceObjectiveReport {
        normalized_speed,
        normalized_memory,
        normalized_stability,
        normalized_drift_response,
        speed_term,
        memory_term,
        stability_term,
        drift_response_term,
        topos_term,
        fractional_term,
        observed_resource_penalty,
        objective_before,
    })
}

fn bias_correction(decay: f64, step: u64) -> f64 {
    if decay == 0.0 {
        1.0
    } else {
        -(decay.ln() * step as f64).exp_m1()
    }
}

fn verify_parameter_delta(
    before: &[f64],
    after: &[f64],
    delta: &[f64],
) -> Result<(), ZSpaceMetaOptimizerError> {
    for index in 0..before.len() {
        let derived = after[index] - before[index];
        let scale = before[index]
            .abs()
            .max(after[index].abs())
            .max(delta[index].abs())
            .max(1.0);
        let residual = (derived - delta[index]).abs();
        let tolerance = DERIVED_TOLERANCE * scale;
        if residual > tolerance {
            return Err(ZSpaceMetaOptimizerError::InvariantViolation {
                field: format!("adam.parameter_delta_residual[{index}]"),
                residual,
                tolerance,
            });
        }
    }
    Ok(())
}

fn verify_scalar_invariant(
    field: &str,
    actual: f64,
    expected: f64,
) -> Result<(), ZSpaceMetaOptimizerError> {
    require_derived_finite(field, actual)?;
    require_derived_finite(&format!("{field}.expected"), expected)?;
    let scale = actual.abs().max(expected.abs()).max(1.0);
    let residual = (actual - expected).abs();
    let tolerance = DERIVED_TOLERANCE * scale;
    if residual > tolerance {
        return Err(ZSpaceMetaOptimizerError::InvariantViolation {
            field: field.to_owned(),
            residual,
            tolerance,
        });
    }
    Ok(())
}

fn require_contract_field(
    field: &'static str,
    actual: &str,
    expected: &'static str,
) -> Result<(), ZSpaceMetaOptimizerError> {
    if actual != expected {
        return Err(ZSpaceMetaOptimizerError::ContractMismatch {
            field,
            expected,
            actual: actual.to_owned(),
        });
    }
    Ok(())
}

fn require_vector_length(
    field: &str,
    values: &[f64],
    expected: usize,
) -> Result<(), ZSpaceMetaOptimizerError> {
    if values.len() != expected {
        return Err(ZSpaceMetaOptimizerError::StateLength {
            field: field.to_owned(),
            expected,
            actual: values.len(),
        });
    }
    Ok(())
}

fn require_finite_vector(field: &str, values: &[f64]) -> Result<(), ZSpaceMetaOptimizerError> {
    for (index, value) in values.iter().copied().enumerate() {
        require_finite(&format!("{field}[{index}]"), value)?;
    }
    Ok(())
}

fn require_finite(field: &str, value: f64) -> Result<(), ZSpaceMetaOptimizerError> {
    if !value.is_finite() {
        return Err(ZSpaceMetaOptimizerError::NonFinite {
            field: field.to_owned(),
            value,
        });
    }
    Ok(())
}

fn require_non_negative(field: &str, value: f64) -> Result<(), ZSpaceMetaOptimizerError> {
    require_finite(field, value)?;
    if value < 0.0 {
        return Err(ZSpaceMetaOptimizerError::Negative {
            field: field.to_owned(),
            value,
        });
    }
    Ok(())
}

fn require_positive(field: &str, value: f64) -> Result<(), ZSpaceMetaOptimizerError> {
    require_finite(field, value)?;
    if value <= 0.0 {
        return Err(ZSpaceMetaOptimizerError::NonPositive {
            field: field.to_owned(),
            value,
        });
    }
    Ok(())
}

fn require_decay(field: &str, value: f64) -> Result<(), ZSpaceMetaOptimizerError> {
    require_finite(field, value)?;
    if !(0.0..1.0).contains(&value) {
        return Err(ZSpaceMetaOptimizerError::InvalidDecay {
            field: field.to_owned(),
            value,
        });
    }
    Ok(())
}

fn require_derived_finite(field: &str, value: f64) -> Result<(), ZSpaceMetaOptimizerError> {
    if !value.is_finite() {
        return Err(ZSpaceMetaOptimizerError::DerivedNonFinite {
            field: field.to_owned(),
        });
    }
    Ok(())
}

fn require_derived_positive(field: &str, value: f64) -> Result<(), ZSpaceMetaOptimizerError> {
    require_derived_finite(field, value)?;
    if value <= 0.0 {
        return Err(ZSpaceMetaOptimizerError::DerivedNonPositive {
            field: field.to_owned(),
            value,
        });
    }
    Ok(())
}

fn checked_add(field: &str, left: f64, right: f64) -> Result<f64, ZSpaceMetaOptimizerError> {
    let value = left + right;
    require_derived_finite(field, value)?;
    Ok(value)
}

fn checked_mul(field: &str, left: f64, right: f64) -> Result<f64, ZSpaceMetaOptimizerError> {
    let value = left * right;
    require_derived_finite(field, value)?;
    Ok(value)
}

fn checked_div(
    field: &str,
    numerator: f64,
    denominator: f64,
) -> Result<f64, ZSpaceMetaOptimizerError> {
    require_derived_positive(&format!("{field}.denominator"), denominator)?;
    let value = numerator / denominator;
    require_derived_finite(field, value)?;
    Ok(value)
}

fn checked_sum<const N: usize>(
    field: &str,
    values: [f64; N],
) -> Result<f64, ZSpaceMetaOptimizerError> {
    values
        .into_iter()
        .try_fold(0.0, |total, value| checked_add(field, total, value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn default_request() -> ZSpaceMetaOptimizerStepRequest {
        let config = ZSpaceMetaOptimizerConfig::default();
        ZSpaceMetaOptimizerStepRequest {
            state: ZSpaceMetaOptimizerState::zeros(config.dimension),
            observation: ZSpaceMetaObservation {
                speed: 0.8,
                memory: 0.5,
                stability: 0.6,
                gradient: vec![0.1, -0.2, 0.3, -0.1],
                ..ZSpaceMetaObservation::default()
            },
            config,
        }
    }

    fn regularizer_energy(z: &[f64], order: f64) -> f64 {
        fractional_regularizer(z, order)
            .expect("regularizer")
            .energy
    }

    #[test]
    fn initializes_a_versioned_zero_checkpoint() {
        let checkpoint = initialize_zspace_meta_optimizer(ZSpaceMetaOptimizerConfig::default())
            .expect("valid defaults");
        assert_eq!(
            checkpoint.contract_version,
            ZSPACE_META_OPTIMIZER_CONTRACT_VERSION
        );
        assert_eq!(checkpoint.semantic_backend, "rust");
        assert_eq!(checkpoint.state, ZSpaceMetaOptimizerState::zeros(4));
    }

    #[test]
    fn rejects_dimensions_above_the_client_allocation_guard() {
        let config = ZSpaceMetaOptimizerConfig {
            dimension: ZSPACE_META_OPTIMIZER_MAX_DIMENSION + 1,
            ..ZSpaceMetaOptimizerConfig::default()
        };
        assert_eq!(
            initialize_zspace_meta_optimizer(config),
            Err(ZSpaceMetaOptimizerError::DimensionLimit {
                dimension: ZSPACE_META_OPTIMIZER_MAX_DIMENSION + 1,
                maximum: ZSPACE_META_OPTIMIZER_MAX_DIMENSION,
            })
        );
    }

    #[test]
    fn analytic_fractional_gradient_matches_central_difference() {
        for order in [0.2, 0.35, 0.9, 1.4] {
            let z = vec![0.4, -0.7, 0.2, 1.1, -0.3, 0.8];
            let report = fractional_regularizer(&z, order).expect("regularizer");
            let step = 1.0e-6;
            for index in 0..z.len() {
                let mut plus = z.clone();
                let mut minus = z.clone();
                plus[index] += step;
                minus[index] -= step;
                let numerical = (regularizer_energy(&plus, order)
                    - regularizer_energy(&minus, order))
                    / (2.0 * step);
                assert_relative_eq!(
                    report.raw_gradient[index],
                    numerical,
                    epsilon = 5.0e-8,
                    max_relative = 5.0e-7
                );
            }
        }
    }

    #[test]
    fn fractional_energy_is_quadratic_and_gradient_obeys_euler_identity() {
        let z = vec![0.4, -0.7, 0.2, 1.1, -0.3];
        let scaled = z.iter().map(|value| 3.0 * value).collect::<Vec<_>>();
        let report = fractional_regularizer(&z, 0.65).expect("regularizer");
        let scaled_report = fractional_regularizer(&scaled, 0.65).expect("regularizer");
        assert_relative_eq!(scaled_report.energy, 9.0 * report.energy, epsilon = 1.0e-11);
        let directional = z
            .iter()
            .zip(&report.raw_gradient)
            .map(|(value, gradient)| value * gradient)
            .sum::<f64>();
        assert_relative_eq!(directional, 2.0 * report.energy, epsilon = 1.0e-11);
    }

    #[test]
    fn first_step_matches_legacy_adam_when_fractional_state_is_zero() {
        let request = default_request();
        let report = transition_zspace_meta_optimizer(request.clone()).expect("valid step");
        let expected_objective = 0.5 * 0.8_f64.tanh() + 0.3 * 0.5_f64.tanh() + 0.2 * 0.6_f64.tanh();
        assert_relative_eq!(
            report.objective.objective_before,
            expected_objective,
            epsilon = 1.0e-14
        );
        assert_eq!(report.fractional_regularizer.energy, 0.0);
        assert_eq!(report.state_after.step, 1);
        for ((z, observed), expected_sign) in report
            .state_after
            .z
            .iter()
            .zip(&report.gradient.projected_normalized)
            .zip([-1.0, 1.0, -1.0, 1.0])
        {
            assert_eq!(z.signum(), expected_sign);
            assert!(observed.abs() > 0.0);
            assert!(z.abs() <= request.config.learning_rate);
        }
    }

    #[test]
    fn topos_hints_scale_actual_learning_rate_and_regularizer() {
        let mut request = default_request();
        request.config.topos_control_gain = 0.5;
        request.observation.telemetry = BTreeMap::from([
            ("topos.closure_pressure".to_owned(), 0.75),
            ("topos.training_hints.learning_rate_scale".to_owned(), 0.5),
            ("topos.training_hints.regularization_scale".to_owned(), 2.0),
            ("topos.training_hints.clip_scale".to_owned(), 0.8),
        ]);
        request.state.z = vec![0.2, -0.1, 0.4, -0.3];
        let report = transition_zspace_meta_optimizer(request.clone()).expect("valid step");
        assert!(report.topos_control.active);
        assert_relative_eq!(report.topos_control.learning_rate_scale, 0.75);
        assert_relative_eq!(
            report.topos_control.effective_learning_rate,
            request.config.learning_rate * 0.75
        );
        assert_relative_eq!(report.topos_control.regularization_scale, 1.5);
        assert_relative_eq!(
            report.topos_control.effective_fractional_weight,
            request.config.weights.fractional * 1.5
        );
        assert_relative_eq!(
            report.objective.fractional_term,
            report.fractional_regularizer.energy * report.topos_control.effective_fractional_weight
        );
    }

    #[test]
    fn parameter_control_is_identical_for_typed_and_serialized_reports() {
        let mut request = default_request();
        request.config.topos_control_gain = 0.5;
        request.observation.telemetry =
            BTreeMap::from([("topos.training_hints.learning_rate_scale".to_owned(), 0.5)]);
        let report = transition_zspace_meta_optimizer(request).expect("valid step");

        let typed = zspace_parameter_control_from_report(&report).expect("typed control");
        let serialized = zspace_parameter_control_from_value(
            serde_json::to_value(&report).expect("serialized report"),
        )
        .expect("serialized control");

        assert_eq!(typed, serialized);
        assert_eq!(
            typed.contract_version(),
            ZSPACE_PARAMETER_CONTROL_CONTRACT_VERSION
        );
        assert_eq!(typed.kind(), ZSPACE_PARAMETER_CONTROL_KIND);
        assert_eq!(
            typed.semantic_backend(),
            ZSPACE_PARAMETER_CONTROL_SEMANTIC_BACKEND
        );
        assert_eq!(typed.source_step(), 1);
        assert_relative_eq!(typed.absolute_learning_rate_scale(), 0.75);
        assert_relative_eq!(
            typed.source_effective_learning_rate(),
            typed.source_learning_rate() * typed.absolute_learning_rate_scale()
        );
        let encoded = serde_json::to_value(typed).expect("serialized control");
        assert_eq!(
            encoded["contract_version"],
            ZSPACE_PARAMETER_CONTROL_CONTRACT_VERSION
        );
        assert_eq!(encoded["kind"], ZSPACE_PARAMETER_CONTROL_KIND);
        assert_eq!(
            encoded["semantic_owner"],
            ZSPACE_PARAMETER_CONTROL_SEMANTIC_OWNER
        );
    }

    #[test]
    fn parameter_control_rejects_tampered_report_invariants() {
        let report = transition_zspace_meta_optimizer(default_request()).expect("valid step");
        let mut encoded = serde_json::to_value(&report).expect("serialized report");
        encoded["topos_control"]["effective_learning_rate"] = serde_json::json!(0.5);

        assert!(matches!(
            zspace_parameter_control_from_value(encoded),
            Err(ZSpaceMetaOptimizerError::InvariantViolation { .. })
        ));

        let mut coordinated_tamper = serde_json::to_value(&report).expect("serialized report");
        coordinated_tamper["topos_control"]["learning_rate_scale"] = serde_json::json!(0.5);
        coordinated_tamper["topos_control"]["effective_learning_rate"] = serde_json::json!(0.005);
        coordinated_tamper["adam"]["effective_learning_rate"] = serde_json::json!(0.005);
        assert!(matches!(
            zspace_parameter_control_from_value(coordinated_tamper),
            Err(ZSpaceMetaOptimizerError::InvariantViolation { ref field, .. })
                if field == "topos_control.learning_rate_scale"
        ));

        let mut unvalidated = serde_json::to_value(&report).expect("serialized report");
        unvalidated["transition_validated"] = serde_json::json!(false);
        assert_eq!(
            zspace_parameter_control_from_value(unvalidated),
            Err(ZSpaceMetaOptimizerError::UnvalidatedTransition)
        );
    }

    #[test]
    fn topos_controls_are_bounded_and_default_clip_is_a_no_op() {
        let mut request = default_request();
        request.config.weights.fractional = 0.0;
        request.config.topos_control_gain = 1.0;
        request.observation.gradient = vec![100.0; request.config.dimension];
        request.observation.telemetry = BTreeMap::from([
            ("topos.closure_pressure".to_owned(), -2.0),
            ("topos.depth_pressure".to_owned(), 3.0),
            ("topos.volume_pressure".to_owned(), 2.0),
            ("topos.guard_strength".to_owned(), -4.0),
            ("topos.openness".to_owned(), 8.0),
            ("topos.exploration_hint".to_owned(), -1.0),
        ]);

        let report = transition_zspace_meta_optimizer(request).expect("bounded controls");

        assert_eq!(report.topos_control.closure_pressure, 0.0);
        assert_eq!(report.topos_control.depth_pressure, 1.0);
        assert_eq!(report.topos_control.volume_pressure, 1.0);
        assert_eq!(report.topos_control.guard_strength, 0.0);
        assert_eq!(report.topos_control.openness, 1.0);
        assert_eq!(report.topos_control.exploration_hint, 0.0);
        assert_eq!(report.topos_control.clip_scale, 1.0);
        assert_eq!(report.topos_control.gradient_clip_threshold, None);
        assert_eq!(report.gradient.clipped_values, 0);
        assert!(report.gradient.applied.iter().any(|value| *value > 1.0));
    }

    #[test]
    fn topos_clip_uses_biased_gradient_rms_instead_of_an_absolute_threshold() {
        let mut request = default_request();
        request.config.dimension = 16;
        request.config.weights.fractional = 0.0;
        request.config.topos_control_gain = 1.0;
        request.state = ZSpaceMetaOptimizerState::zeros(request.config.dimension);
        request.observation.gradient = vec![0.0; request.config.dimension];
        request.observation.gradient[0] = 100.0;
        request.observation.telemetry = BTreeMap::from([
            ("topos.training_hints.clip_scale".to_owned(), 0.25),
            ("topos.training_hints.gradient_bias_scale".to_owned(), 0.0),
        ]);

        let report = transition_zspace_meta_optimizer(request).expect("RMS-clipped transition");
        let expected_rms = 1.0 / 4.0;
        let expected_threshold = expected_rms / 0.75;
        assert_eq!(
            report.topos_control.gradient_clip_rule,
            TOPOS_OPTIMIZER_GRADIENT_CLIP_RULE
        );
        assert_eq!(
            report.topos_control.gradient_clip_normalization,
            TOPOS_OPTIMIZER_GRADIENT_CLIP_NORMALIZATION
        );
        assert_relative_eq!(report.topos_control.biased_gradient_rms, expected_rms);
        assert_relative_eq!(
            report
                .topos_control
                .gradient_clip_threshold
                .expect("active clip"),
            expected_threshold
        );
        assert_eq!(report.gradient.clipped_values, 1);
        assert_relative_eq!(report.gradient.before_clip[0], 1.0);
        assert_relative_eq!(report.gradient.applied[0], expected_threshold);
        assert!(expected_threshold > report.topos_control.clip_scale);
    }

    #[test]
    fn unrelated_topos_telemetry_does_not_activate_optimizer_controls() {
        let mut request = default_request();
        request.config.topos_control_gain = 1.0;
        request.observation.telemetry = BTreeMap::from([("topos.report_only".to_owned(), 1.0)]);

        let report = transition_zspace_meta_optimizer(request).expect("unrelated telemetry");

        assert!(!report.topos_control.present);
        assert!(!report.topos_control.active);
        assert_eq!(report.topos_control.pressure, 0.0);
        assert!(report
            .topos_control
            .gradient_bias
            .iter()
            .all(|value| *value == 0.0));
    }

    #[test]
    fn exact_gradient_projection_rejects_dimension_mismatch() {
        let mut request = default_request();
        request.config.gradient_projection = GradientProjection::Exact;
        request.observation.gradient = vec![0.1, 0.2];
        assert_eq!(
            transition_zspace_meta_optimizer(request),
            Err(ZSpaceMetaOptimizerError::GradientLength {
                expected: 4,
                actual: 2
            })
        );

        let mut empty = default_request();
        empty.config.gradient_projection = GradientProjection::Exact;
        empty.observation.gradient.clear();
        assert_eq!(
            transition_zspace_meta_optimizer(empty),
            Err(ZSpaceMetaOptimizerError::GradientLength {
                expected: 4,
                actual: 0
            })
        );
    }

    #[test]
    fn tile_projection_repeats_short_observed_gradient() {
        let mut request = default_request();
        request.observation.gradient = vec![0.1, -0.2];
        request.config.weights.fractional = 0.0;
        let report = transition_zspace_meta_optimizer(request).expect("valid step");
        assert_eq!(
            report.gradient.projected_normalized,
            vec![
                0.1_f64.tanh(),
                (-0.2_f64).tanh(),
                0.1_f64.tanh(),
                (-0.2_f64).tanh()
            ]
        );
    }

    #[test]
    fn rejected_stateful_step_preserves_state_and_last_report() {
        let mut optimizer =
            ZSpaceMetaOptimizer::new(ZSpaceMetaOptimizerConfig::default()).expect("optimizer");
        optimizer
            .try_step(default_request().observation)
            .expect("first step");
        let state_before = optimizer.state().clone();
        let report_before = optimizer.last_report().cloned();
        let invalid = ZSpaceMetaObservation {
            speed: f64::NAN,
            ..ZSpaceMetaObservation::default()
        };
        assert!(optimizer.try_step(invalid).is_err());
        assert_eq!(optimizer.state(), &state_before);
        assert_eq!(optimizer.last_report(), report_before.as_ref());
    }

    #[test]
    fn step_counter_stays_exact_across_python_and_wasm_clients() {
        let mut request = default_request();
        request.state.step = ZSPACE_META_OPTIMIZER_MAX_SAFE_STEP;
        assert_eq!(
            transition_zspace_meta_optimizer(request),
            Err(ZSpaceMetaOptimizerError::StepLimit {
                step: ZSPACE_META_OPTIMIZER_MAX_SAFE_STEP + 1,
                maximum: ZSPACE_META_OPTIMIZER_MAX_SAFE_STEP,
            })
        );
    }

    #[test]
    fn derived_overflow_rejects_the_entire_transition() {
        let mut request = default_request();
        request.config.weights.speed = f64::MAX;
        request.config.weights.memory = f64::MAX;
        request.observation.speed = 1.0;
        request.observation.memory = 1.0;
        assert!(matches!(
            transition_zspace_meta_optimizer(request),
            Err(ZSpaceMetaOptimizerError::DerivedNonFinite { .. })
        ));
    }

    #[test]
    fn non_strict_restore_pads_and_truncates_in_rust() {
        let config = ZSpaceMetaOptimizerConfig {
            dimension: 3,
            ..ZSpaceMetaOptimizerConfig::default()
        };
        let checkpoint = restore_zspace_meta_optimizer(ZSpaceMetaOptimizerRestoreRequest {
            config,
            state: ZSpaceMetaOptimizerState {
                z: vec![1.0],
                first_moment: vec![2.0, 3.0, 4.0, 5.0],
                second_moment: vec![],
                step: 7,
            },
            strict: false,
        })
        .expect("coerced checkpoint");
        assert_eq!(checkpoint.state.z, vec![1.0, 0.0, 0.0]);
        assert_eq!(checkpoint.state.first_moment, vec![2.0, 3.0, 4.0]);
        assert_eq!(checkpoint.state.second_moment, vec![0.0, 0.0, 0.0]);
        assert_eq!(checkpoint.state.step, 7);
    }

    #[test]
    fn non_strict_restore_rejects_invalid_values_in_discarded_tails() {
        let config = ZSpaceMetaOptimizerConfig {
            dimension: 2,
            ..ZSpaceMetaOptimizerConfig::default()
        };
        let invalid = restore_zspace_meta_optimizer(ZSpaceMetaOptimizerRestoreRequest {
            config,
            state: ZSpaceMetaOptimizerState {
                z: vec![0.0, 0.0, f64::NAN],
                first_moment: vec![0.0, 0.0],
                second_moment: vec![0.0, 0.0, -1.0],
                step: 0,
            },
            strict: false,
        });
        assert!(matches!(
            invalid,
            Err(ZSpaceMetaOptimizerError::NonFinite { .. })
        ));
    }

    #[test]
    fn strict_restore_rejects_wrong_dimensions_and_negative_second_moment() {
        let config = ZSpaceMetaOptimizerConfig::default();
        let wrong = restore_zspace_meta_optimizer(ZSpaceMetaOptimizerRestoreRequest {
            config: config.clone(),
            state: ZSpaceMetaOptimizerState::default(),
            strict: true,
        });
        assert!(matches!(
            wrong,
            Err(ZSpaceMetaOptimizerError::StateLength { .. })
        ));

        let mut state = ZSpaceMetaOptimizerState::zeros(config.dimension);
        state.second_moment[2] = -0.1;
        assert_eq!(
            restore_zspace_meta_optimizer(ZSpaceMetaOptimizerRestoreRequest {
                config,
                state,
                strict: true,
            }),
            Err(ZSpaceMetaOptimizerError::NegativeSecondMoment {
                index: 2,
                value: -0.1
            })
        );
    }

    #[test]
    fn serde_accepts_legacy_hyperparameter_and_state_names() {
        let request: ZSpaceMetaOptimizerStepRequest = serde_json::from_value(serde_json::json!({
            "config": {
                "dimension": 2,
                "alpha": 0.4,
                "lr": 0.02,
                "beta1": 0.8,
                "beta2": 0.9,
                "eps": 1e-7
            },
            "state": {
                "z": [0.0, 0.0],
                "moment": [0.0, 0.0],
                "velocity": [0.0, 0.0],
                "step": 0
            },
            "observation": {"gradient": [0.2, -0.1]}
        }))
        .expect("legacy aliases");
        assert_eq!(request.config.fractional_order, 0.4);
        assert_eq!(request.config.learning_rate, 0.02);
        assert_eq!(request.state.first_moment, vec![0.0, 0.0]);
        assert!(transition_zspace_meta_optimizer(request).is_ok());
    }
}
