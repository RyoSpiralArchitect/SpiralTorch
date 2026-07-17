//! Canonical validation contract for trainer optimizer controls.
//!
//! Trainer hosts may choose how to orchestrate training, but Rust owns the
//! admissible curvature, learning-rate, and gradient-guard semantics.

use serde::{Deserialize, Serialize};
use st_tensor::{AmegaHypergrad, AmegaHypergradCheckpoint, AmegaRealgrad, AmegaRealgradCheckpoint};
use thiserror::Error;

use crate::inference::zspace_coherence::ZSpaceCoherenceLabel;
use crate::telemetry::zspace_region::ZSpaceRegionDescriptor;

pub const TRAINER_OPTIMIZER_CONFIG_CONTRACT_VERSION: &str =
    "spiraltorch.trainer_optimizer_config.v1";
pub const TRAINER_OPTIMIZER_CONFIG_KIND: &str = "spiraltorch.trainer_optimizer_config";
pub const TRAINER_OPTIMIZER_CONFIG_SEMANTIC_OWNER: &str = "st-core::runtime::trainer_optimizer";
pub const TRAINER_OPTIMIZER_CONFIG_SEMANTIC_BACKEND: &str = "rust";
pub const TRAINER_OPTIMIZER_CONFIG_VALIDATION_RULE: &str =
    "curvature<0;hyper_learning_rate>0;fallback_learning_rate>0;real_learning_rate=null|>0;grad_clip_max_norm=null|>0;all_values_finite;invalid_updates_rejected";
pub const TRAINER_CURVATURE_PRESSURE_MAX: f32 = 1.0e19;
pub const TRAINER_OPTIMIZER_CHECKPOINT_CONTRACT_VERSION: &str =
    "spiraltorch.trainer_optimizer_checkpoint.v1";
pub const TRAINER_OPTIMIZER_CHECKPOINT_KIND: &str = "spiraltorch.trainer_optimizer_checkpoint";
pub const TRAINER_OPTIMIZER_CHECKPOINT_RESUME_SCOPE: &str =
    "trainer_optimizer_and_builtin_update_policy;parameter_values_external_and_fingerprint_guarded;external_runtime_components_reported";

#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum TrainerOptimizerConfigError {
    #[error("curvature must be negative and finite, got {value}")]
    InvalidCurvature { value: f32 },
    #[error("{field} must be positive and finite, got {value}")]
    InvalidPositiveFinite { field: &'static str, value: f32 },
}

impl TrainerOptimizerConfigError {
    /// Preserves established tensor error categories at `PureResult` boundaries.
    pub fn into_tensor_error(self) -> st_tensor::TensorError {
        match self {
            Self::InvalidCurvature { value } => {
                st_tensor::TensorError::NonHyperbolicCurvature { curvature: value }
            }
            Self::InvalidPositiveFinite {
                field: "hyper_learning_rate" | "fallback_learning_rate" | "real_learning_rate",
                value,
            } => st_tensor::TensorError::NonPositiveLearningRate { rate: value },
            Self::InvalidPositiveFinite { field, value } => st_tensor::TensorError::Generic(
                format!("{field} must be positive and finite, got {value}"),
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainerOptimizerConfig {
    pub curvature: f32,
    pub hyper_learning_rate: f32,
    pub fallback_learning_rate: f32,
    pub real_learning_rate: Option<f32>,
    pub grad_clip_max_norm: Option<f32>,
}

impl TrainerOptimizerConfig {
    pub fn try_new(
        curvature: f32,
        hyper_learning_rate: f32,
        fallback_learning_rate: f32,
    ) -> Result<Self, TrainerOptimizerConfigError> {
        let config = Self {
            curvature,
            hyper_learning_rate,
            fallback_learning_rate,
            real_learning_rate: None,
            grad_clip_max_norm: None,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), TrainerOptimizerConfigError> {
        if !self.curvature.is_finite() || self.curvature >= 0.0 {
            return Err(TrainerOptimizerConfigError::InvalidCurvature {
                value: self.curvature,
            });
        }
        validate_positive_finite("hyper_learning_rate", self.hyper_learning_rate)?;
        validate_positive_finite("fallback_learning_rate", self.fallback_learning_rate)?;
        if let Some(value) = self.real_learning_rate {
            validate_positive_finite("real_learning_rate", value)?;
        }
        if let Some(value) = self.grad_clip_max_norm {
            validate_positive_finite("grad_clip_max_norm", value)?;
        }
        Ok(())
    }

    pub fn with_real_learning_rate(
        mut self,
        value: Option<f32>,
    ) -> Result<Self, TrainerOptimizerConfigError> {
        self.real_learning_rate = value;
        self.validate()?;
        Ok(self)
    }

    pub fn with_grad_clip_max_norm(
        mut self,
        value: Option<f32>,
    ) -> Result<Self, TrainerOptimizerConfigError> {
        self.grad_clip_max_norm = value;
        self.validate()?;
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct TrainerOptimizerConfigContract {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub validation_rule: &'static str,
    pub realgrad_enabled: bool,
    pub gradient_clip_enabled: bool,
    pub config: TrainerOptimizerConfig,
}

pub fn evaluate_trainer_optimizer_config(
    config: TrainerOptimizerConfig,
) -> Result<TrainerOptimizerConfigContract, TrainerOptimizerConfigError> {
    config.validate()?;
    Ok(TrainerOptimizerConfigContract {
        kind: TRAINER_OPTIMIZER_CONFIG_KIND,
        contract_version: TRAINER_OPTIMIZER_CONFIG_CONTRACT_VERSION,
        semantic_owner: TRAINER_OPTIMIZER_CONFIG_SEMANTIC_OWNER,
        semantic_backend: TRAINER_OPTIMIZER_CONFIG_SEMANTIC_BACKEND,
        validation_rule: TRAINER_OPTIMIZER_CONFIG_VALIDATION_RULE,
        realgrad_enabled: config.real_learning_rate.is_some(),
        gradient_clip_enabled: config.grad_clip_max_norm.is_some(),
        config,
    })
}

/// Serializable state of the built-in local spectral learning-rate adapter.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpectralLrAdapterState {
    pub sheet_hint: usize,
    pub smoothing: f32,
    pub curvature_target: f32,
    pub curvature_gain: f32,
    pub spin_gain: f32,
    pub energy_gain: f32,
    pub sheet_gain: f32,
    pub min_scale: f32,
    pub max_scale: f32,
    pub avg_curvature: f32,
    pub avg_spin: f32,
    pub avg_energy: f32,
}

impl Default for SpectralLrAdapterState {
    fn default() -> Self {
        Self {
            sheet_hint: 8,
            smoothing: 0.2,
            curvature_target: 0.0,
            curvature_gain: 0.3,
            spin_gain: 0.2,
            energy_gain: 0.15,
            sheet_gain: 0.1,
            min_scale: 0.25,
            max_scale: 4.0,
            avg_curvature: 0.0,
            avg_spin: 0.0,
            avg_energy: 0.0,
        }
    }
}

/// Complete state of the trainer curvature scheduler.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainerCurvatureSchedulerState {
    pub min_curvature: f32,
    pub max_curvature: f32,
    pub target_pressure: f32,
    pub tolerance: f32,
    pub step: f32,
    pub proportional_gain: f32,
    pub smoothing: f32,
    pub current: f32,
    pub ema_pressure: Option<f32>,
    pub ema_pressure2: Option<f32>,
    pub stability_threshold: f32,
    pub stability_boost: f32,
    pub stable_steps: u32,
    pub dither_strength: f32,
    pub dither_period: u32,
    pub dither_sign: f32,
}

/// Rust-owned SoftLogic configuration transported inside trainer checkpoints.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainerSoftLogicConfigState {
    pub inertia: f32,
    pub inertia_min: f32,
    pub inertia_drift_k: f32,
    pub inertia_z_k: f32,
    pub drift_gain: f32,
    pub psi_gain: f32,
    pub loss_gain: f32,
    pub floor: f32,
    pub scale_gain: f32,
    pub region_gain: f32,
    pub region_factor_gain: f32,
    pub energy_equalize_gain: f32,
    pub mean_normalize_gain: f32,
    pub energy_equalize_auto: f32,
    pub mean_normalize_auto: f32,
}

impl Default for TrainerSoftLogicConfigState {
    fn default() -> Self {
        Self {
            inertia: 0.65,
            inertia_min: 0.15,
            inertia_drift_k: 0.6,
            inertia_z_k: 0.2,
            drift_gain: 0.25,
            psi_gain: 0.5,
            loss_gain: 0.35,
            floor: 0.25,
            scale_gain: 0.2,
            region_gain: 0.15,
            region_factor_gain: 0.35,
            energy_equalize_gain: 0.0,
            mean_normalize_gain: 0.0,
            energy_equalize_auto: 0.0,
            mean_normalize_auto: 0.0,
        }
    }
}

/// Minimal feedback state that affects future SoftLogic decisions.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainerSoftLogicFeedbackState {
    pub z_signal: f32,
    pub scale_log_radius: Option<f32>,
}

/// Stateful portion of the built-in SoftLogic update policy.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainerSoftLogicState {
    pub config: TrainerSoftLogicConfigState,
    pub last_weights: [f32; 3],
    pub last_z: f32,
    pub last_feedback: Option<TrainerSoftLogicFeedbackState>,
    pub last_inertia: f32,
    pub last_region: Option<ZSpaceRegionDescriptor>,
    pub last_region_factor: f32,
    pub last_region_scale: [f32; 3],
    pub equalize_state: f32,
    pub mean_normalize_state: f32,
    pub pending_events: Vec<String>,
    pub equalize_guard_on: bool,
    pub equalize_clamp_on: bool,
    pub normalize_on: bool,
}

impl Default for TrainerSoftLogicState {
    fn default() -> Self {
        Self {
            config: TrainerSoftLogicConfigState::default(),
            last_weights: [1.0; 3],
            last_z: 0.0,
            last_feedback: None,
            last_inertia: 0.65,
            last_region: None,
            last_region_factor: 1.0,
            last_region_scale: [1.0; 3],
            equalize_state: 1.0,
            mean_normalize_state: 1.0,
            pending_events: Vec::new(),
            equalize_guard_on: false,
            equalize_clamp_on: false,
            normalize_on: false,
        }
    }
}

/// Complete mutable state of the coherence-driven spectral policy.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainerSpectralPolicyState {
    pub smoothing: f32,
    pub event_smoothing: f32,
    pub turnover_smoothing: f32,
    pub coherence_gain: f32,
    pub curvature_gain: f32,
    pub sheet_gain: f32,
    pub spin_gain: f32,
    pub radius_gain: f32,
    pub energy_gain: f32,
    pub phase_gain: f32,
    pub stuck_phase_gain: f32,
    pub max_phase_gain: f32,
    pub stuck_turnover_threshold: f32,
    pub dominant_turnover: f32,
    pub last_dominant: Option<usize>,
    pub last_label: Option<ZSpaceCoherenceLabel>,
    pub min_lr_scale: f32,
    pub max_lr_scale: f32,
    pub max_lr_step: f32,
    pub min_band_scale: f32,
    pub max_band_scale: f32,
    pub band_state: [f32; 3],
    pub lr_state: f32,
    pub applied_lr_scale: f32,
    pub local_lr_state: [f32; 3],
}

/// Stateful edge detector used by the built-in `TrainerPhase` event stream.
///
/// The thresholds are included because they may originate from environment
/// overrides. Omitting them would make a restored trainer emit different phase
/// transitions even when its optimizer update remained numerically identical.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainerPhaseTrackerState {
    pub turnover_spike_threshold: f32,
    pub loss_ema_alpha: f32,
    pub loss_spike_ratio: f32,
    pub drift_spike_threshold: f32,
    pub last_label: Option<ZSpaceCoherenceLabel>,
    pub last_turnover: Option<f32>,
    pub last_band: Option<u8>,
    pub last_drift_abs: Option<f32>,
    pub loss_ema: Option<f32>,
    pub loss_spiking: bool,
}

impl Default for TrainerPhaseTrackerState {
    fn default() -> Self {
        Self {
            turnover_spike_threshold: 0.35,
            loss_ema_alpha: 0.12,
            loss_spike_ratio: 0.25,
            drift_spike_threshold: 0.25,
            last_label: None,
            last_turnover: None,
            last_band: None,
            last_drift_abs: None,
            loss_ema: None,
            loss_spiking: false,
        }
    }
}

/// Cumulative backend evidence retained across completed training epochs.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainerBackendCounters {
    pub epochs_recorded: u64,
    pub ops_total: u64,
    pub fallbacks: u64,
    pub backend_wgpu: u64,
    pub backend_cuda: u64,
    pub backend_hip: u64,
    pub backend_cpu: u64,
    pub backend_other: u64,
    pub requested_wgpu_hits: u64,
    pub requested_wgpu_runtime_fallbacks: u64,
    pub requested_wgpu_component_hits: u64,
    pub requested_wgpu_component_fallbacks: u64,
}

impl TrainerBackendCounters {
    pub fn saturating_accumulate(&mut self, other: Self) {
        self.epochs_recorded = self.epochs_recorded.saturating_add(other.epochs_recorded);
        self.ops_total = self.ops_total.saturating_add(other.ops_total);
        self.fallbacks = self.fallbacks.saturating_add(other.fallbacks);
        self.backend_wgpu = self.backend_wgpu.saturating_add(other.backend_wgpu);
        self.backend_cuda = self.backend_cuda.saturating_add(other.backend_cuda);
        self.backend_hip = self.backend_hip.saturating_add(other.backend_hip);
        self.backend_cpu = self.backend_cpu.saturating_add(other.backend_cpu);
        self.backend_other = self.backend_other.saturating_add(other.backend_other);
        self.requested_wgpu_hits = self
            .requested_wgpu_hits
            .saturating_add(other.requested_wgpu_hits);
        self.requested_wgpu_runtime_fallbacks = self
            .requested_wgpu_runtime_fallbacks
            .saturating_add(other.requested_wgpu_runtime_fallbacks);
        self.requested_wgpu_component_hits = self
            .requested_wgpu_component_hits
            .saturating_add(other.requested_wgpu_component_hits);
        self.requested_wgpu_component_fallbacks = self
            .requested_wgpu_component_fallbacks
            .saturating_add(other.requested_wgpu_component_fallbacks);
    }
}

/// Stable execution identity that must match before state is restored.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainerExecutionTopology {
    pub backend: String,
    pub subgroup: bool,
    pub lane_width: u32,
    pub max_workgroup: u32,
    pub shared_mem_per_workgroup: Option<u32>,
    pub accelerator_fallback: String,
    pub tensor_util_wgpu_min_values: usize,
    pub training_device_enabled: bool,
    pub training_rank: usize,
    pub training_world_size: usize,
    pub external_state_required: Vec<String>,
}

/// Per-parameter optimizer state. Parameter values remain in the module state
/// dictionary and are guarded by a deterministic fingerprint.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainerParameterOptimizerState {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub value_fingerprint: String,
    pub euclidean_gradient: Option<Vec<f32>>,
    pub hypergrad: Option<AmegaHypergradCheckpoint>,
    pub realgrad: Option<AmegaRealgradCheckpoint>,
}

/// Mutable trainer state whose next update is fully Rust-owned.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainerOptimizerRuntimeState {
    pub epoch: u64,
    pub meta_learning_rate_scale: f32,
    pub meta_optimizer_step: Option<u64>,
    pub spectral_adapter: SpectralLrAdapterState,
    pub curvature_scheduler: Option<TrainerCurvatureSchedulerState>,
    pub spectral_policy: Option<TrainerSpectralPolicyState>,
    pub phase_tracker: TrainerPhaseTrackerState,
    pub softlogic: TrainerSoftLogicState,
    pub backend_counters: TrainerBackendCounters,
    pub last_accumulator_sync_buffers: u64,
    pub last_accumulator_sync_values: u64,
}

/// Versioned, deserializable optimizer checkpoint shared by Rust clients and
/// exposed unchanged through Python and WASM.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainerOptimizerCheckpoint {
    pub kind: String,
    pub contract_version: String,
    pub semantic_owner: String,
    pub semantic_backend: String,
    pub resume_scope: String,
    pub config: TrainerOptimizerConfig,
    pub topology: TrainerExecutionTopology,
    pub state: TrainerOptimizerRuntimeState,
    pub parameters: Vec<TrainerParameterOptimizerState>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TrainerOptimizerCheckpointValidation {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub resume_scope: &'static str,
    pub parameter_count: usize,
    pub hypergrad_parameters: usize,
    pub realgrad_parameters: usize,
    pub euclidean_parameters: usize,
    pub captures_inflight_accumulators: bool,
    pub parameter_values_external: bool,
    pub deterministic_resume_ready: bool,
    pub external_state_required: Vec<String>,
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum TrainerOptimizerCheckpointError {
    #[error(transparent)]
    Config(#[from] TrainerOptimizerConfigError),
    #[error("checkpoint {field} must be {expected}, got {actual}")]
    InvalidMetadata {
        field: &'static str,
        expected: &'static str,
        actual: String,
    },
    #[error("invalid checkpoint state at {field}: {message}")]
    InvalidState { field: String, message: String },
    #[error("duplicate checkpoint parameter {name}")]
    DuplicateParameter { name: String },
}

pub fn build_trainer_optimizer_checkpoint(
    config: TrainerOptimizerConfig,
    topology: TrainerExecutionTopology,
    state: TrainerOptimizerRuntimeState,
    mut parameters: Vec<TrainerParameterOptimizerState>,
) -> Result<TrainerOptimizerCheckpoint, TrainerOptimizerCheckpointError> {
    parameters.sort_by(|left, right| left.name.cmp(&right.name));
    let checkpoint = TrainerOptimizerCheckpoint {
        kind: TRAINER_OPTIMIZER_CHECKPOINT_KIND.to_owned(),
        contract_version: TRAINER_OPTIMIZER_CHECKPOINT_CONTRACT_VERSION.to_owned(),
        semantic_owner: TRAINER_OPTIMIZER_CONFIG_SEMANTIC_OWNER.to_owned(),
        semantic_backend: TRAINER_OPTIMIZER_CONFIG_SEMANTIC_BACKEND.to_owned(),
        resume_scope: TRAINER_OPTIMIZER_CHECKPOINT_RESUME_SCOPE.to_owned(),
        config,
        topology,
        state,
        parameters,
    };
    evaluate_trainer_optimizer_checkpoint(&checkpoint)?;
    Ok(checkpoint)
}

pub fn evaluate_trainer_optimizer_checkpoint(
    checkpoint: &TrainerOptimizerCheckpoint,
) -> Result<TrainerOptimizerCheckpointValidation, TrainerOptimizerCheckpointError> {
    validate_checkpoint_metadata("kind", &checkpoint.kind, TRAINER_OPTIMIZER_CHECKPOINT_KIND)?;
    validate_checkpoint_metadata(
        "contract_version",
        &checkpoint.contract_version,
        TRAINER_OPTIMIZER_CHECKPOINT_CONTRACT_VERSION,
    )?;
    validate_checkpoint_metadata(
        "semantic_owner",
        &checkpoint.semantic_owner,
        TRAINER_OPTIMIZER_CONFIG_SEMANTIC_OWNER,
    )?;
    validate_checkpoint_metadata(
        "semantic_backend",
        &checkpoint.semantic_backend,
        TRAINER_OPTIMIZER_CONFIG_SEMANTIC_BACKEND,
    )?;
    validate_checkpoint_metadata(
        "resume_scope",
        &checkpoint.resume_scope,
        TRAINER_OPTIMIZER_CHECKPOINT_RESUME_SCOPE,
    )?;
    checkpoint.config.validate()?;
    checkpoint.topology.validate()?;
    checkpoint.state.validate(&checkpoint.config)?;

    let mut names = std::collections::HashSet::with_capacity(checkpoint.parameters.len());
    let mut hypergrad_parameters = 0usize;
    let mut realgrad_parameters = 0usize;
    let mut euclidean_parameters = 0usize;
    let mut captures_inflight_accumulators = false;
    for parameter in &checkpoint.parameters {
        if !names.insert(parameter.name.as_str()) {
            return Err(TrainerOptimizerCheckpointError::DuplicateParameter {
                name: parameter.name.clone(),
            });
        }
        parameter.validate(&checkpoint.config)?;
        hypergrad_parameters += usize::from(parameter.hypergrad.is_some());
        realgrad_parameters += usize::from(parameter.realgrad.is_some());
        euclidean_parameters += usize::from(parameter.euclidean_gradient.is_some());
        captures_inflight_accumulators |= parameter.has_inflight_accumulator();
    }

    Ok(TrainerOptimizerCheckpointValidation {
        kind: TRAINER_OPTIMIZER_CHECKPOINT_KIND,
        contract_version: TRAINER_OPTIMIZER_CHECKPOINT_CONTRACT_VERSION,
        semantic_owner: TRAINER_OPTIMIZER_CONFIG_SEMANTIC_OWNER,
        semantic_backend: TRAINER_OPTIMIZER_CONFIG_SEMANTIC_BACKEND,
        resume_scope: TRAINER_OPTIMIZER_CHECKPOINT_RESUME_SCOPE,
        parameter_count: checkpoint.parameters.len(),
        hypergrad_parameters,
        realgrad_parameters,
        euclidean_parameters,
        captures_inflight_accumulators,
        parameter_values_external: true,
        deterministic_resume_ready: checkpoint.topology.external_state_required.is_empty(),
        external_state_required: checkpoint.topology.external_state_required.clone(),
    })
}

impl SpectralLrAdapterState {
    pub fn validate(&self) -> Result<(), TrainerOptimizerCheckpointError> {
        if self.sheet_hint == 0 {
            return Err(checkpoint_state_error(
                "state.spectral_adapter.sheet_hint",
                "must be positive",
            ));
        }
        validate_range(
            "state.spectral_adapter.smoothing",
            self.smoothing,
            f32::EPSILON,
            1.0,
        )?;
        validate_finite(
            "state.spectral_adapter.curvature_target",
            self.curvature_target,
        )?;
        for (field, value) in [
            ("curvature_gain", self.curvature_gain),
            ("spin_gain", self.spin_gain),
            ("energy_gain", self.energy_gain),
            ("sheet_gain", self.sheet_gain),
        ] {
            validate_non_negative(&format!("state.spectral_adapter.{field}"), value)?;
        }
        validate_positive("state.spectral_adapter.min_scale", self.min_scale)?;
        validate_positive("state.spectral_adapter.max_scale", self.max_scale)?;
        if self.max_scale < self.min_scale {
            return Err(checkpoint_state_error(
                "state.spectral_adapter.max_scale",
                "must be greater than or equal to min_scale",
            ));
        }
        for (field, value) in [
            ("avg_curvature", self.avg_curvature),
            ("avg_spin", self.avg_spin),
            ("avg_energy", self.avg_energy),
        ] {
            validate_finite(&format!("state.spectral_adapter.{field}"), value)?;
        }
        Ok(())
    }
}

impl TrainerCurvatureSchedulerState {
    pub fn validate(&self) -> Result<(), TrainerOptimizerCheckpointError> {
        validate_negative(
            "state.curvature_scheduler.min_curvature",
            self.min_curvature,
        )?;
        validate_negative(
            "state.curvature_scheduler.max_curvature",
            self.max_curvature,
        )?;
        if self.min_curvature > self.max_curvature {
            return Err(checkpoint_state_error(
                "state.curvature_scheduler",
                "min_curvature must be less than or equal to max_curvature",
            ));
        }
        validate_range(
            "state.curvature_scheduler.target_pressure",
            self.target_pressure,
            0.0,
            TRAINER_CURVATURE_PRESSURE_MAX,
        )?;
        validate_non_negative("state.curvature_scheduler.tolerance", self.tolerance)?;
        validate_positive("state.curvature_scheduler.step", self.step)?;
        validate_non_negative(
            "state.curvature_scheduler.proportional_gain",
            self.proportional_gain,
        )?;
        validate_range(
            "state.curvature_scheduler.smoothing",
            self.smoothing,
            0.01,
            1.0,
        )?;
        validate_negative("state.curvature_scheduler.current", self.current)?;
        if self.current < self.min_curvature || self.current > self.max_curvature {
            return Err(checkpoint_state_error(
                "state.curvature_scheduler.current",
                "must stay within curvature bounds",
            ));
        }
        if self.ema_pressure.is_some() != self.ema_pressure2.is_some() {
            return Err(checkpoint_state_error(
                "state.curvature_scheduler.ema_pressure",
                "first and second pressure moments must be present together",
            ));
        }
        if let Some(value) = self.ema_pressure {
            validate_non_negative("state.curvature_scheduler.ema_pressure", value)?;
        }
        if let Some(value) = self.ema_pressure2 {
            validate_non_negative("state.curvature_scheduler.ema_pressure2", value)?;
        }
        validate_positive(
            "state.curvature_scheduler.stability_threshold",
            self.stability_threshold,
        )?;
        validate_non_negative(
            "state.curvature_scheduler.stability_boost",
            self.stability_boost,
        )?;
        validate_range(
            "state.curvature_scheduler.dither_strength",
            self.dither_strength,
            0.0,
            1.0,
        )?;
        if self.dither_period == 0 {
            return Err(checkpoint_state_error(
                "state.curvature_scheduler.dither_period",
                "must be positive",
            ));
        }
        if self.dither_sign != -1.0 && self.dither_sign != 1.0 {
            return Err(checkpoint_state_error(
                "state.curvature_scheduler.dither_sign",
                "must be -1 or 1",
            ));
        }
        Ok(())
    }
}

impl TrainerSoftLogicConfigState {
    pub fn validate(&self) -> Result<(), TrainerOptimizerCheckpointError> {
        for (field, value, min, max) in [
            ("inertia", self.inertia, 0.0, 0.95),
            ("inertia_min", self.inertia_min, 0.0, 0.95),
            ("inertia_drift_k", self.inertia_drift_k, 0.0, 4.0),
            ("inertia_z_k", self.inertia_z_k, 0.0, 2.0),
            ("drift_gain", self.drift_gain, 0.0, 1.0),
            ("psi_gain", self.psi_gain, 0.0, 2.0),
            ("loss_gain", self.loss_gain, 0.0, 1.5),
            ("floor", self.floor, 0.05, 1.0),
            ("scale_gain", self.scale_gain, 0.0, 1.5),
            ("region_gain", self.region_gain, 0.0, 1.5),
            ("region_factor_gain", self.region_factor_gain, 0.0, 2.0),
            ("energy_equalize_gain", self.energy_equalize_gain, 0.0, 1.0),
            ("mean_normalize_gain", self.mean_normalize_gain, 0.0, 1.0),
            ("energy_equalize_auto", self.energy_equalize_auto, 0.0, 1.0),
            ("mean_normalize_auto", self.mean_normalize_auto, 0.0, 1.0),
        ] {
            validate_range(&format!("state.softlogic.config.{field}"), value, min, max)?;
        }
        if self.inertia_min > self.inertia {
            return Err(checkpoint_state_error(
                "state.softlogic.config.inertia_min",
                "must be less than or equal to inertia",
            ));
        }
        Ok(())
    }
}

impl TrainerSoftLogicState {
    pub fn validate(&self) -> Result<(), TrainerOptimizerCheckpointError> {
        self.config.validate()?;
        for (index, value) in self.last_weights.iter().copied().enumerate() {
            validate_positive(&format!("state.softlogic.last_weights[{index}]"), value)?;
        }
        validate_range("state.softlogic.last_z", self.last_z, -1.0, 1.0)?;
        if let Some(feedback) = self.last_feedback {
            validate_range(
                "state.softlogic.last_feedback.z_signal",
                feedback.z_signal,
                -1.0,
                1.0,
            )?;
            if let Some(log_radius) = feedback.scale_log_radius {
                validate_finite("state.softlogic.last_feedback.scale_log_radius", log_radius)?;
                let physical = log_radius.exp();
                if !physical.is_finite() || physical <= 0.0 {
                    return Err(checkpoint_state_error(
                        "state.softlogic.last_feedback.scale_log_radius",
                        "must reconstruct a finite positive radius",
                    ));
                }
            }
        }
        validate_range("state.softlogic.last_inertia", self.last_inertia, 0.0, 0.95)?;
        if let Some(region) = self.last_region {
            validate_range(
                "state.softlogic.last_region.spin_alignment",
                region.spin_alignment,
                -1.0,
                1.0,
            )?;
            validate_range(
                "state.softlogic.last_region.normalized_radius",
                region.normalized_radius,
                0.0,
                1.0,
            )?;
            validate_non_negative(
                "state.softlogic.last_region.curvature_radius",
                region.curvature_radius,
            )?;
            validate_non_negative(
                "state.softlogic.last_region.geodesic_radius",
                region.geodesic_radius,
            )?;
        }
        validate_non_negative(
            "state.softlogic.last_region_factor",
            self.last_region_factor,
        )?;
        for (index, value) in self.last_region_scale.iter().copied().enumerate() {
            validate_positive(
                &format!("state.softlogic.last_region_scale[{index}]"),
                value,
            )?;
        }
        validate_range(
            "state.softlogic.equalize_state",
            self.equalize_state,
            0.0,
            1.0,
        )?;
        validate_range(
            "state.softlogic.mean_normalize_state",
            self.mean_normalize_state,
            0.0,
            1.0,
        )?;
        if self
            .pending_events
            .iter()
            .any(|event| event.trim().is_empty())
        {
            return Err(checkpoint_state_error(
                "state.softlogic.pending_events",
                "event labels must not be empty",
            ));
        }
        Ok(())
    }
}

impl TrainerSpectralPolicyState {
    pub fn validate(&self) -> Result<(), TrainerOptimizerCheckpointError> {
        for (field, value) in [
            ("smoothing", self.smoothing),
            ("event_smoothing", self.event_smoothing),
            ("turnover_smoothing", self.turnover_smoothing),
        ] {
            validate_range(
                &format!("state.spectral_policy.{field}"),
                value,
                1.0e-3,
                1.0,
            )?;
        }
        for (field, value) in [
            ("coherence_gain", self.coherence_gain),
            ("curvature_gain", self.curvature_gain),
            ("sheet_gain", self.sheet_gain),
            ("spin_gain", self.spin_gain),
            ("radius_gain", self.radius_gain),
            ("energy_gain", self.energy_gain),
            ("phase_gain", self.phase_gain),
            ("stuck_phase_gain", self.stuck_phase_gain),
            ("max_phase_gain", self.max_phase_gain),
            ("stuck_turnover_threshold", self.stuck_turnover_threshold),
        ] {
            validate_non_negative(&format!("state.spectral_policy.{field}"), value)?;
        }
        validate_range(
            "state.spectral_policy.dominant_turnover",
            self.dominant_turnover,
            0.0,
            1.0,
        )?;
        validate_positive("state.spectral_policy.min_lr_scale", self.min_lr_scale)?;
        validate_positive("state.spectral_policy.max_lr_scale", self.max_lr_scale)?;
        if self.max_lr_scale <= self.min_lr_scale {
            return Err(checkpoint_state_error(
                "state.spectral_policy.max_lr_scale",
                "must be greater than min_lr_scale",
            ));
        }
        validate_range(
            "state.spectral_policy.max_lr_step",
            self.max_lr_step,
            1.0,
            f32::MAX,
        )?;
        validate_positive("state.spectral_policy.min_band_scale", self.min_band_scale)?;
        validate_positive("state.spectral_policy.max_band_scale", self.max_band_scale)?;
        if self.max_band_scale <= self.min_band_scale {
            return Err(checkpoint_state_error(
                "state.spectral_policy.max_band_scale",
                "must be greater than min_band_scale",
            ));
        }
        for (field, values) in [
            ("band_state", self.band_state),
            ("local_lr_state", self.local_lr_state),
        ] {
            for (index, value) in values.into_iter().enumerate() {
                validate_range(
                    &format!("state.spectral_policy.{field}[{index}]"),
                    value,
                    self.min_band_scale,
                    self.max_band_scale,
                )?;
            }
        }
        validate_range(
            "state.spectral_policy.lr_state",
            self.lr_state,
            self.min_lr_scale,
            self.max_lr_scale,
        )?;
        validate_range(
            "state.spectral_policy.applied_lr_scale",
            self.applied_lr_scale,
            self.min_lr_scale,
            self.max_lr_scale,
        )?;
        Ok(())
    }
}

impl TrainerPhaseTrackerState {
    pub fn validate(&self) -> Result<(), TrainerOptimizerCheckpointError> {
        validate_range(
            "state.phase_tracker.turnover_spike_threshold",
            self.turnover_spike_threshold,
            0.0,
            1.0,
        )?;
        validate_range(
            "state.phase_tracker.loss_ema_alpha",
            self.loss_ema_alpha,
            f32::EPSILON,
            1.0,
        )?;
        validate_range(
            "state.phase_tracker.loss_spike_ratio",
            self.loss_spike_ratio,
            0.0,
            10.0,
        )?;
        validate_non_negative(
            "state.phase_tracker.drift_spike_threshold",
            self.drift_spike_threshold,
        )?;
        if let Some(value) = self.last_turnover {
            validate_range("state.phase_tracker.last_turnover", value, 0.0, 1.0)?;
        }
        if self.last_band.is_some_and(|band| band > 2) {
            return Err(checkpoint_state_error(
                "state.phase_tracker.last_band",
                "must be one of the Above, Here, or Beneath band indices",
            ));
        }
        if let Some(value) = self.last_drift_abs {
            validate_non_negative("state.phase_tracker.last_drift_abs", value)?;
        }
        if let Some(value) = self.loss_ema {
            validate_non_negative("state.phase_tracker.loss_ema", value)?;
        }
        if self.loss_spiking && self.loss_ema.is_none() {
            return Err(checkpoint_state_error(
                "state.phase_tracker.loss_spiking",
                "requires a loss EMA",
            ));
        }
        Ok(())
    }
}

impl TrainerExecutionTopology {
    pub fn validate(&self) -> Result<(), TrainerOptimizerCheckpointError> {
        if !matches!(
            self.backend.as_str(),
            "cpu" | "wgpu" | "mps" | "cuda" | "hip"
        ) {
            return Err(checkpoint_state_error(
                "topology.backend",
                "must use a known Rust backend label",
            ));
        }
        if self.lane_width == 0 || self.max_workgroup < self.lane_width {
            return Err(checkpoint_state_error(
                "topology.max_workgroup",
                "must be at least one native lane",
            ));
        }
        if !matches!(self.accelerator_fallback.as_str(), "allow" | "forbid") {
            return Err(checkpoint_state_error(
                "topology.accelerator_fallback",
                "must be allow or forbid",
            ));
        }
        if self.training_world_size == 0 || self.training_rank >= self.training_world_size {
            return Err(checkpoint_state_error(
                "topology.training_world_size",
                "must be positive and greater than training_rank",
            ));
        }
        if !self.training_device_enabled
            && (self.training_rank != 0 || self.training_world_size != 1)
        {
            return Err(checkpoint_state_error(
                "topology.training_device_enabled",
                "disabled synchronization must use rank 0 and world size 1",
            ));
        }
        let mut previous: Option<&str> = None;
        for component in &self.external_state_required {
            let component = component.trim();
            if component.is_empty() {
                return Err(checkpoint_state_error(
                    "topology.external_state_required",
                    "component labels must not be empty",
                ));
            }
            if previous.is_some_and(|value| value >= component) {
                return Err(checkpoint_state_error(
                    "topology.external_state_required",
                    "component labels must be sorted and unique",
                ));
            }
            previous = Some(component);
        }
        Ok(())
    }
}

impl TrainerParameterOptimizerState {
    pub fn validate(
        &self,
        config: &TrainerOptimizerConfig,
    ) -> Result<(), TrainerOptimizerCheckpointError> {
        if self.name.trim().is_empty() {
            return Err(checkpoint_state_error(
                "parameters.name",
                "must not be empty",
            ));
        }
        let expected = self
            .rows
            .checked_mul(self.cols)
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                checkpoint_state_error(
                    &format!("parameters.{}.shape", self.name),
                    "must be non-zero and must not overflow",
                )
            })?;
        if self.value_fingerprint.len() != 16
            || !self
                .value_fingerprint
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(checkpoint_state_error(
                &format!("parameters.{}.value_fingerprint", self.name),
                "must be a 16-digit hexadecimal fingerprint",
            ));
        }
        if self.euclidean_gradient.is_some()
            && (self.hypergrad.is_some() || self.realgrad.is_some())
        {
            return Err(checkpoint_state_error(
                &format!("parameters.{}.euclidean_gradient", self.name),
                "cannot coexist with Amega tapes",
            ));
        }
        if let Some(gradient) = &self.euclidean_gradient {
            validate_parameter_vector(&self.name, "euclidean_gradient", gradient, expected)?;
        }
        if let Some(hypergrad) = &self.hypergrad {
            if (hypergrad.rows, hypergrad.cols) != (self.rows, self.cols) {
                return Err(checkpoint_state_error(
                    &format!("parameters.{}.hypergrad.shape", self.name),
                    "must match the parameter shape",
                ));
            }
            AmegaHypergrad::from_checkpoint(hypergrad.clone()).map_err(|error| {
                checkpoint_state_error(
                    &format!("parameters.{}.hypergrad", self.name),
                    &error.to_string(),
                )
            })?;
            if hypergrad.curvature.to_bits() != config.curvature.to_bits()
                || hypergrad.learning_rate.to_bits() != config.hyper_learning_rate.to_bits()
            {
                return Err(checkpoint_state_error(
                    &format!("parameters.{}.hypergrad", self.name),
                    "curvature and learning rate must match trainer config",
                ));
            }
        }
        if let Some(realgrad) = &self.realgrad {
            if (realgrad.rows, realgrad.cols) != (self.rows, self.cols) {
                return Err(checkpoint_state_error(
                    &format!("parameters.{}.realgrad.shape", self.name),
                    "must match the parameter shape",
                ));
            }
            AmegaRealgrad::from_checkpoint(realgrad.clone()).map_err(|error| {
                checkpoint_state_error(
                    &format!("parameters.{}.realgrad", self.name),
                    &error.to_string(),
                )
            })?;
            if config.real_learning_rate.map(f32::to_bits) != Some(realgrad.learning_rate.to_bits())
            {
                return Err(checkpoint_state_error(
                    &format!("parameters.{}.realgrad", self.name),
                    "learning rate must match enabled trainer realgrad config",
                ));
            }
        }
        Ok(())
    }

    fn has_inflight_accumulator(&self) -> bool {
        self.euclidean_gradient
            .as_deref()
            .is_some_and(has_nonzero_value)
            || self
                .hypergrad
                .as_ref()
                .is_some_and(|state| has_nonzero_value(&state.gradient))
            || self
                .realgrad
                .as_ref()
                .is_some_and(|state| has_nonzero_value(&state.gradient))
    }
}

impl TrainerOptimizerRuntimeState {
    pub fn validate(
        &self,
        config: &TrainerOptimizerConfig,
    ) -> Result<(), TrainerOptimizerCheckpointError> {
        validate_positive(
            "state.meta_learning_rate_scale",
            self.meta_learning_rate_scale,
        )?;
        self.spectral_adapter.validate()?;
        if self.spectral_adapter.curvature_target.to_bits() != config.curvature.to_bits() {
            return Err(checkpoint_state_error(
                "state.spectral_adapter.curvature_target",
                "must match trainer curvature",
            ));
        }
        if let Some(scheduler) = self.curvature_scheduler {
            scheduler.validate()?;
            if scheduler.current.to_bits() != config.curvature.to_bits() {
                return Err(checkpoint_state_error(
                    "state.curvature_scheduler.current",
                    "must match trainer curvature",
                ));
            }
        }
        if let Some(policy) = self.spectral_policy {
            policy.validate()?;
        }
        self.phase_tracker.validate()?;
        self.softlogic.validate()?;
        if self.backend_counters.epochs_recorded > self.epoch {
            return Err(checkpoint_state_error(
                "state.backend_counters.epochs_recorded",
                "cannot exceed trainer epoch",
            ));
        }
        if self.last_accumulator_sync_buffers == 0 && self.last_accumulator_sync_values != 0 {
            return Err(checkpoint_state_error(
                "state.last_accumulator_sync_values",
                "must be zero when no buffers were synchronized",
            ));
        }
        Ok(())
    }
}

fn validate_checkpoint_metadata(
    field: &'static str,
    actual: &str,
    expected: &'static str,
) -> Result<(), TrainerOptimizerCheckpointError> {
    if actual == expected {
        Ok(())
    } else {
        Err(TrainerOptimizerCheckpointError::InvalidMetadata {
            field,
            expected,
            actual: actual.to_owned(),
        })
    }
}

fn checkpoint_state_error(field: &str, message: &str) -> TrainerOptimizerCheckpointError {
    TrainerOptimizerCheckpointError::InvalidState {
        field: field.to_owned(),
        message: message.to_owned(),
    }
}

fn validate_finite(field: &str, value: f32) -> Result<(), TrainerOptimizerCheckpointError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(checkpoint_state_error(field, "must be finite"))
    }
}

fn validate_positive(field: &str, value: f32) -> Result<(), TrainerOptimizerCheckpointError> {
    validate_finite(field, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(checkpoint_state_error(field, "must be positive"))
    }
}

fn validate_negative(field: &str, value: f32) -> Result<(), TrainerOptimizerCheckpointError> {
    validate_finite(field, value)?;
    if value < 0.0 {
        Ok(())
    } else {
        Err(checkpoint_state_error(field, "must be negative"))
    }
}

fn validate_non_negative(field: &str, value: f32) -> Result<(), TrainerOptimizerCheckpointError> {
    validate_finite(field, value)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(checkpoint_state_error(field, "must be non-negative"))
    }
}

fn validate_range(
    field: &str,
    value: f32,
    min: f32,
    max: f32,
) -> Result<(), TrainerOptimizerCheckpointError> {
    validate_finite(field, value)?;
    if (min..=max).contains(&value) {
        Ok(())
    } else {
        Err(checkpoint_state_error(
            field,
            &format!("must be in [{min}, {max}]"),
        ))
    }
}

fn validate_parameter_vector(
    parameter: &str,
    field: &str,
    values: &[f32],
    expected: usize,
) -> Result<(), TrainerOptimizerCheckpointError> {
    if values.len() != expected {
        return Err(checkpoint_state_error(
            &format!("parameters.{parameter}.{field}"),
            &format!("expected {expected} values, got {}", values.len()),
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(checkpoint_state_error(
            &format!("parameters.{parameter}.{field}"),
            "all values must be finite",
        ));
    }
    Ok(())
}

fn has_nonzero_value(values: &[f32]) -> bool {
    values.iter().any(|value| *value != 0.0)
}

fn validate_positive_finite(
    field: &'static str,
    value: f32,
) -> Result<(), TrainerOptimizerConfigError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(TrainerOptimizerConfigError::InvalidPositiveFinite { field, value });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn valid_softlogic_state() -> TrainerSoftLogicState {
        TrainerSoftLogicState {
            config: TrainerSoftLogicConfigState {
                inertia: 0.65,
                inertia_min: 0.15,
                inertia_drift_k: 0.6,
                inertia_z_k: 0.2,
                drift_gain: 0.25,
                psi_gain: 0.5,
                loss_gain: 0.35,
                floor: 0.25,
                scale_gain: 0.2,
                region_gain: 0.15,
                region_factor_gain: 0.35,
                energy_equalize_gain: 0.0,
                mean_normalize_gain: 0.0,
                energy_equalize_auto: 0.0,
                mean_normalize_auto: 0.0,
            },
            last_weights: [1.0; 3],
            last_z: 0.0,
            last_feedback: None,
            last_inertia: 0.65,
            last_region: None,
            last_region_factor: 1.0,
            last_region_scale: [1.0; 3],
            equalize_state: 1.0,
            mean_normalize_state: 1.0,
            pending_events: Vec::new(),
            equalize_guard_on: false,
            equalize_clamp_on: false,
            normalize_on: false,
        }
    }

    fn valid_checkpoint() -> TrainerOptimizerCheckpoint {
        let config = TrainerOptimizerConfig::try_new(-1.0, 0.02, 0.01).unwrap();
        let hypergrad = AmegaHypergrad::new(-1.0, 0.02, 1, 2).unwrap().checkpoint();
        build_trainer_optimizer_checkpoint(
            config,
            TrainerExecutionTopology {
                backend: "cpu".to_owned(),
                subgroup: true,
                lane_width: 8,
                max_workgroup: 8,
                shared_mem_per_workgroup: None,
                accelerator_fallback: "allow".to_owned(),
                tensor_util_wgpu_min_values: 1024,
                training_device_enabled: false,
                training_rank: 0,
                training_world_size: 1,
                external_state_required: Vec::new(),
            },
            TrainerOptimizerRuntimeState {
                epoch: 1,
                meta_learning_rate_scale: 1.0,
                meta_optimizer_step: None,
                spectral_adapter: SpectralLrAdapterState {
                    sheet_hint: 8,
                    smoothing: 0.2,
                    curvature_target: -1.0,
                    curvature_gain: 0.3,
                    spin_gain: 0.2,
                    energy_gain: 0.15,
                    sheet_gain: 0.1,
                    min_scale: 0.25,
                    max_scale: 4.0,
                    avg_curvature: 0.0,
                    avg_spin: 0.0,
                    avg_energy: 0.0,
                },
                curvature_scheduler: None,
                spectral_policy: None,
                phase_tracker: TrainerPhaseTrackerState::default(),
                softlogic: valid_softlogic_state(),
                backend_counters: TrainerBackendCounters {
                    epochs_recorded: 1,
                    ops_total: 4,
                    backend_cpu: 4,
                    ..TrainerBackendCounters::default()
                },
                last_accumulator_sync_buffers: 0,
                last_accumulator_sync_values: 0,
            },
            vec![TrainerParameterOptimizerState {
                name: "weight".to_owned(),
                rows: 1,
                cols: 2,
                value_fingerprint: "0123456789abcdef".to_owned(),
                euclidean_gradient: None,
                hypergrad: Some(hypergrad),
                realgrad: None,
            }],
        )
        .unwrap()
    }

    #[test]
    fn valid_config_emits_a_versioned_rust_contract() {
        let config = TrainerOptimizerConfig::try_new(-1.0, 0.02, 0.01)
            .unwrap()
            .with_real_learning_rate(Some(0.005))
            .unwrap()
            .with_grad_clip_max_norm(Some(1.5))
            .unwrap();

        let contract = evaluate_trainer_optimizer_config(config).unwrap();

        assert_eq!(contract.kind, TRAINER_OPTIMIZER_CONFIG_KIND);
        assert_eq!(
            contract.contract_version,
            TRAINER_OPTIMIZER_CONFIG_CONTRACT_VERSION
        );
        assert_eq!(
            contract.semantic_owner,
            TRAINER_OPTIMIZER_CONFIG_SEMANTIC_OWNER
        );
        assert_eq!(
            contract.semantic_backend,
            TRAINER_OPTIMIZER_CONFIG_SEMANTIC_BACKEND
        );
        assert!(contract.realgrad_enabled);
        assert!(contract.gradient_clip_enabled);
        assert_eq!(contract.config, config);
    }

    #[test]
    fn curvature_and_learning_rates_fail_closed() {
        assert!(matches!(
            TrainerOptimizerConfig::try_new(0.0, 0.02, 0.01),
            Err(TrainerOptimizerConfigError::InvalidCurvature { .. })
        ));
        assert!(matches!(
            TrainerOptimizerConfig::try_new(f32::NAN, 0.02, 0.01),
            Err(TrainerOptimizerConfigError::InvalidCurvature { .. })
        ));
        assert!(matches!(
            TrainerOptimizerConfig::try_new(-1.0, 0.0, 0.01),
            Err(TrainerOptimizerConfigError::InvalidPositiveFinite {
                field: "hyper_learning_rate",
                ..
            })
        ));
        assert!(matches!(
            TrainerOptimizerConfig::try_new(-1.0, 0.02, f32::INFINITY),
            Err(TrainerOptimizerConfigError::InvalidPositiveFinite {
                field: "fallback_learning_rate",
                ..
            })
        ));
    }

    #[test]
    fn optional_control_updates_reject_noise_without_changing_the_source() {
        let config = TrainerOptimizerConfig::try_new(-1.0, 0.02, 0.01)
            .unwrap()
            .with_real_learning_rate(Some(0.005))
            .unwrap()
            .with_grad_clip_max_norm(Some(1.5))
            .unwrap();

        assert!(config.with_real_learning_rate(Some(f32::NAN)).is_err());
        assert!(config.with_grad_clip_max_norm(Some(0.0)).is_err());
        assert_eq!(config.real_learning_rate, Some(0.005));
        assert_eq!(config.grad_clip_max_norm, Some(1.5));

        let invalid_base = TrainerOptimizerConfig {
            curvature: 0.5,
            ..config
        };
        assert!(matches!(
            invalid_base.with_real_learning_rate(None),
            Err(TrainerOptimizerConfigError::InvalidCurvature { .. })
        ));
    }

    #[test]
    fn serde_ingress_rejects_unknown_fields() {
        let error = serde_json::from_value::<TrainerOptimizerConfig>(json!({
            "curvature": -1.0,
            "hyper_learning_rate": 0.02,
            "fallback_learning_rate": 0.01,
            "real_learning_rate": null,
            "grad_clip_max_norm": null,
            "commander": "python"
        }))
        .unwrap_err();

        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn checkpoint_roundtrips_as_a_versioned_rust_contract() {
        let checkpoint = valid_checkpoint();
        let encoded = serde_json::to_string(&checkpoint).unwrap();
        let decoded: TrainerOptimizerCheckpoint = serde_json::from_str(&encoded).unwrap();
        let report = evaluate_trainer_optimizer_checkpoint(&decoded).unwrap();

        assert_eq!(decoded, checkpoint);
        assert_eq!(report.kind, TRAINER_OPTIMIZER_CHECKPOINT_KIND);
        assert_eq!(
            report.contract_version,
            TRAINER_OPTIMIZER_CHECKPOINT_CONTRACT_VERSION
        );
        assert_eq!(report.semantic_backend, "rust");
        assert_eq!(report.parameter_count, 1);
        assert_eq!(report.hypergrad_parameters, 1);
        assert!(report.deterministic_resume_ready);
        assert!(report.parameter_values_external);
    }

    #[test]
    fn checkpoint_tampering_and_duplicate_parameters_fail_closed() {
        let mut checkpoint = valid_checkpoint();
        checkpoint.contract_version = "python.v9".to_owned();
        assert!(matches!(
            evaluate_trainer_optimizer_checkpoint(&checkpoint),
            Err(TrainerOptimizerCheckpointError::InvalidMetadata {
                field: "contract_version",
                ..
            })
        ));

        let mut checkpoint = valid_checkpoint();
        checkpoint.state.spectral_adapter.avg_energy = f32::NAN;
        assert!(matches!(
            evaluate_trainer_optimizer_checkpoint(&checkpoint),
            Err(TrainerOptimizerCheckpointError::InvalidState { .. })
        ));

        let mut checkpoint = valid_checkpoint();
        checkpoint.state.phase_tracker.loss_ema = Some(f32::NAN);
        assert!(matches!(
            evaluate_trainer_optimizer_checkpoint(&checkpoint),
            Err(TrainerOptimizerCheckpointError::InvalidState { .. })
        ));

        let mut checkpoint = valid_checkpoint();
        checkpoint.parameters.push(checkpoint.parameters[0].clone());
        assert!(matches!(
            evaluate_trainer_optimizer_checkpoint(&checkpoint),
            Err(TrainerOptimizerCheckpointError::DuplicateParameter { .. })
        ));
    }

    #[test]
    fn checkpoint_reports_external_runtime_state_and_rejects_unknown_fields() {
        let mut checkpoint = valid_checkpoint();
        checkpoint.topology.external_state_required = vec!["desire_bridge".to_owned()];
        let report = evaluate_trainer_optimizer_checkpoint(&checkpoint).unwrap();
        assert!(!report.deterministic_resume_ready);
        assert_eq!(report.external_state_required, ["desire_bridge"]);

        let mut payload = serde_json::to_value(checkpoint).unwrap();
        payload
            .as_object_mut()
            .unwrap()
            .insert("commander".to_owned(), json!("wasm"));
        let error = serde_json::from_value::<TrainerOptimizerCheckpoint>(payload).unwrap_err();
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn optimizer_errors_preserve_existing_tensor_error_categories() {
        let curvature =
            TrainerOptimizerConfigError::InvalidCurvature { value: 0.5 }.into_tensor_error();
        assert!(matches!(
            curvature,
            st_tensor::TensorError::NonHyperbolicCurvature { curvature: 0.5 }
        ));

        let rate = TrainerOptimizerConfigError::InvalidPositiveFinite {
            field: "hyper_learning_rate",
            value: 0.0,
        }
        .into_tensor_error();
        assert!(matches!(
            rate,
            st_tensor::TensorError::NonPositiveLearningRate { rate: 0.0 }
        ));

        let clip = TrainerOptimizerConfigError::InvalidPositiveFinite {
            field: "grad_clip_max_norm",
            value: 0.0,
        }
        .into_tensor_error();
        assert!(
            matches!(clip, st_tensor::TensorError::Generic(message) if message.contains("grad_clip_max_norm"))
        );
    }
}
