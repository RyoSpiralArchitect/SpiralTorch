// SPDX-License-Identifier: AGPL-3.0-or-later

//! Rust-owned feedback gate for external Z-space optimizer controls.
//!
//! Clients transport trainer observations and proposed learning-rate scales.
//! Rust owns loss projection, state transitions, staleness, and the bounded
//! blend back toward the identity update.

use crate::runtime::zspace_optimizer::{
    ZSPACE_META_OPTIMIZER_MAX_SAFE_STEP, ZSPACE_PARAMETER_CONTROL_MAX_LEARNING_RATE_SCALE,
    ZSPACE_PARAMETER_CONTROL_MIN_LEARNING_RATE_SCALE,
};
use crate::telemetry::training_projection::{
    project_training_telemetry, TrainingTelemetryObservation, TrainingTelemetryProjectionConfig,
    TrainingTelemetryProjectionPayload, TrainingTelemetryProjectionRequest,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const ZSPACE_OPTIMIZER_FEEDBACK_CONTRACT_VERSION: &str =
    "spiraltorch.zspace_optimizer_feedback.v1";
pub const ZSPACE_OPTIMIZER_FEEDBACK_KIND: &str = "spiraltorch.zspace_optimizer_feedback";
pub const ZSPACE_OPTIMIZER_FEEDBACK_SEMANTIC_OWNER: &str =
    "st-core::runtime::zspace_optimizer_feedback";
pub const ZSPACE_OPTIMIZER_FEEDBACK_SEMANTIC_BACKEND: &str = "rust";
pub const ZSPACE_OPTIMIZER_FEEDBACK_CONTROL_RULE: &str =
    "applied_scale=1+effective_gate*(proposed_scale-1)";

const DERIVED_TOLERANCE: f64 = 128.0 * f64::EPSILON;

/// Loss-guard policy for one external optimizer-control stream.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ZSpaceOptimizerFeedbackConfig {
    /// EMA update weight for the absolute loss level.
    pub loss_ema_alpha: f64,
    /// EMA update weight for relative loss deltas.
    pub relative_delta_ema_alpha: f64,
    /// Denominator floor used when the previous loss is near zero.
    pub loss_floor: f64,
    /// Positive relative-delta EMA that starts attenuating the gate.
    pub regression_threshold: f64,
    /// Positive relative-delta EMA that immediately closes the gate.
    pub halt_threshold: f64,
    /// Negative relative-delta EMA magnitude that permits recovery.
    pub recovery_threshold: f64,
    /// Additive gate reduction after a non-halting regression.
    pub attenuation_rate: f64,
    /// Additive gate increase after supported improvement.
    pub recovery_rate: f64,
    /// Consecutive regressions that close the gate even below halt threshold.
    pub halt_regression_streak: u64,
    /// Consecutive improvements required to leave a halted state.
    pub resume_improvement_streak: u64,
    /// Observations collected under identity control before opening the gate.
    pub warmup_observations: u64,
    /// Additional optimizer updates allowed after the freshest observation.
    pub max_stale_updates: u64,
    /// Upper bound on how much of the proposed deviation may cross the gate.
    pub maximum_gate: f64,
}

impl Default for ZSpaceOptimizerFeedbackConfig {
    fn default() -> Self {
        Self {
            loss_ema_alpha: 0.2,
            relative_delta_ema_alpha: 0.5,
            loss_floor: 1.0e-8,
            regression_threshold: 0.01,
            halt_threshold: 0.05,
            recovery_threshold: 0.0025,
            attenuation_rate: 0.25,
            recovery_rate: 0.125,
            halt_regression_streak: 2,
            resume_improvement_streak: 2,
            warmup_observations: 2,
            max_stale_updates: 0,
            maximum_gate: 1.0,
        }
    }
}

impl ZSpaceOptimizerFeedbackConfig {
    pub fn validate(&self) -> Result<(), ZSpaceOptimizerFeedbackError> {
        require_unit_interval_open_zero("config.loss_ema_alpha", self.loss_ema_alpha)?;
        require_unit_interval_open_zero(
            "config.relative_delta_ema_alpha",
            self.relative_delta_ema_alpha,
        )?;
        require_positive("config.loss_floor", self.loss_floor)?;
        require_non_negative("config.regression_threshold", self.regression_threshold)?;
        require_non_negative("config.halt_threshold", self.halt_threshold)?;
        if self.halt_threshold < self.regression_threshold {
            return Err(ZSpaceOptimizerFeedbackError::InvalidConfig {
                field: "config.halt_threshold",
                message: "must be greater than or equal to regression_threshold",
            });
        }
        require_non_negative("config.recovery_threshold", self.recovery_threshold)?;
        require_unit_interval_open_zero("config.attenuation_rate", self.attenuation_rate)?;
        require_unit_interval_open_zero("config.recovery_rate", self.recovery_rate)?;
        require_positive_count("config.halt_regression_streak", self.halt_regression_streak)?;
        require_positive_count(
            "config.resume_improvement_streak",
            self.resume_improvement_streak,
        )?;
        require_safe_count("config.warmup_observations", self.warmup_observations)?;
        require_safe_count("config.max_stale_updates", self.max_stale_updates)?;
        require_unit_interval_open_zero("config.maximum_gate", self.maximum_gate)?;
        Ok(())
    }
}

/// Checkpointable state shared by native, Python, and WASM clients.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ZSpaceOptimizerFeedbackState {
    pub control_step: u64,
    pub observation_count: u64,
    pub last_observation_step: Option<u64>,
    pub last_loss: Option<f64>,
    pub loss_ema: Option<f64>,
    pub relative_loss_delta_ema: Option<f64>,
    pub gate: f64,
    pub regression_streak: u64,
    pub improvement_streak: u64,
    pub halted: bool,
}

impl Default for ZSpaceOptimizerFeedbackState {
    fn default() -> Self {
        Self {
            control_step: 0,
            observation_count: 0,
            last_observation_step: None,
            last_loss: None,
            loss_ema: None,
            relative_loss_delta_ema: None,
            gate: 0.0,
            regression_streak: 0,
            improvement_streak: 0,
            halted: false,
        }
    }
}

impl ZSpaceOptimizerFeedbackState {
    fn validate(
        &self,
        config: &ZSpaceOptimizerFeedbackConfig,
    ) -> Result<(), ZSpaceOptimizerFeedbackError> {
        require_safe_count("state.control_step", self.control_step)?;
        require_safe_count("state.observation_count", self.observation_count)?;
        require_safe_count("state.regression_streak", self.regression_streak)?;
        require_safe_count("state.improvement_streak", self.improvement_streak)?;
        require_range("state.gate", self.gate, 0.0, config.maximum_gate)?;
        for (field, value) in [
            ("state.last_loss", self.last_loss),
            ("state.loss_ema", self.loss_ema),
            (
                "state.relative_loss_delta_ema",
                self.relative_loss_delta_ema,
            ),
        ] {
            if let Some(value) = value {
                require_finite(field, value)?;
            }
        }
        if self.observation_count == 0 {
            if self.last_observation_step.is_some()
                || self.last_loss.is_some()
                || self.loss_ema.is_some()
                || self.relative_loss_delta_ema.is_some()
                || self.gate != 0.0
                || self.regression_streak != 0
                || self.improvement_streak != 0
                || self.halted
            {
                return Err(ZSpaceOptimizerFeedbackError::InvalidState {
                    field: "state.observation_count",
                    message: "zero observations require an empty fail-closed state",
                });
            }
        } else {
            let Some(last_observation_step) = self.last_observation_step else {
                return Err(ZSpaceOptimizerFeedbackError::InvalidState {
                    field: "state.last_observation_step",
                    message: "is required after the first observation",
                });
            };
            if last_observation_step > self.control_step {
                return Err(ZSpaceOptimizerFeedbackError::InvalidState {
                    field: "state.last_observation_step",
                    message: "cannot be ahead of the completed control step",
                });
            }
            if self.last_loss.is_none() || self.loss_ema.is_none() {
                return Err(ZSpaceOptimizerFeedbackError::InvalidState {
                    field: "state.last_loss",
                    message: "loss and loss EMA are required after observation",
                });
            }
            if self.observation_count == 1 && self.relative_loss_delta_ema.is_some() {
                return Err(ZSpaceOptimizerFeedbackError::InvalidState {
                    field: "state.relative_loss_delta_ema",
                    message: "the first observation has no preceding loss delta",
                });
            }
            if self.observation_count > 1 && self.relative_loss_delta_ema.is_none() {
                return Err(ZSpaceOptimizerFeedbackError::InvalidState {
                    field: "state.relative_loss_delta_ema",
                    message: "is required after the second observation",
                });
            }
            if self.relative_loss_delta_ema.is_none()
                && (self.gate != 0.0
                    || self.regression_streak != 0
                    || self.improvement_streak != 0
                    || self.halted)
            {
                return Err(ZSpaceOptimizerFeedbackError::InvalidState {
                    field: "state.gate",
                    message: "the first observation must remain fail closed",
                });
            }
            if self.observation_count <= config.warmup_observations
                && (self.gate != 0.0
                    || self.regression_streak != 0
                    || self.improvement_streak != 0
                    || self.halted)
            {
                return Err(ZSpaceOptimizerFeedbackError::InvalidState {
                    field: "state.gate",
                    message: "warmup observations must remain fail closed",
                });
            }
            let maximum_observations = self.control_step.checked_add(1).ok_or(
                ZSpaceOptimizerFeedbackError::StepLimit {
                    step: self.control_step,
                    maximum: ZSPACE_META_OPTIMIZER_MAX_SAFE_STEP,
                },
            )?;
            if self.observation_count > maximum_observations {
                return Err(ZSpaceOptimizerFeedbackError::InvalidState {
                    field: "state.observation_count",
                    message: "exceeds the number of observable control steps",
                });
            }
        }
        if self.regression_streak > 0 && self.improvement_streak > 0 {
            return Err(ZSpaceOptimizerFeedbackError::InvalidState {
                field: "state.regression_streak",
                message: "regression and improvement streaks cannot coexist",
            });
        }
        if self.halted && self.gate != 0.0 {
            return Err(ZSpaceOptimizerFeedbackError::InvalidState {
                field: "state.gate",
                message: "a halted gate must apply identity control",
            });
        }
        if self.observation_count > config.warmup_observations {
            if !self.halted
                && (self.regression_streak >= config.halt_regression_streak
                    || self
                        .relative_loss_delta_ema
                        .is_some_and(|delta| delta >= config.halt_threshold))
            {
                return Err(ZSpaceOptimizerFeedbackError::InvalidState {
                    field: "state.halted",
                    message: "halt evidence requires a halted gate",
                });
            }
            if self.halted && self.improvement_streak >= config.resume_improvement_streak {
                return Err(ZSpaceOptimizerFeedbackError::InvalidState {
                    field: "state.improvement_streak",
                    message: "resume evidence requires an open halted state",
                });
            }
        }
        Ok(())
    }
}

/// One completed trainer observation. Previous loss is always taken from state.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceOptimizerFeedbackObservation {
    pub step: u64,
    #[serde(default)]
    pub max_steps: Option<u64>,
    #[serde(default)]
    pub epoch: Option<f64>,
    pub loss: f64,
    #[serde(default)]
    pub grad_norm: Option<f64>,
    #[serde(default)]
    pub learning_rate: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceOptimizerFeedbackRestoreRequest {
    pub config: ZSpaceOptimizerFeedbackConfig,
    pub state: ZSpaceOptimizerFeedbackState,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceOptimizerFeedbackObserveRequest {
    pub config: ZSpaceOptimizerFeedbackConfig,
    pub state: ZSpaceOptimizerFeedbackState,
    pub observation: ZSpaceOptimizerFeedbackObservation,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceOptimizerFeedbackControlRequest {
    pub config: ZSpaceOptimizerFeedbackConfig,
    pub state: ZSpaceOptimizerFeedbackState,
    pub target_step: u64,
    pub proposed_learning_rate_scale: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ZSpaceOptimizerFeedbackObservationAction {
    Initialize,
    Warmup,
    Hold,
    Recover,
    Attenuate,
    Halt,
    HoldHalted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ZSpaceOptimizerFeedbackControlDisposition {
    NoFeedback,
    Warmup,
    Halted,
    Stale,
    IdentityGate,
    Active,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceOptimizerFeedbackCheckpoint {
    pub contract_version: &'static str,
    pub kind: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub config: ZSpaceOptimizerFeedbackConfig,
    pub state: ZSpaceOptimizerFeedbackState,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceOptimizerFeedbackObservationReport {
    pub contract_version: &'static str,
    pub kind: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub transition_validated: bool,
    pub config: ZSpaceOptimizerFeedbackConfig,
    pub observation: ZSpaceOptimizerFeedbackObservation,
    pub projection: TrainingTelemetryProjectionPayload,
    pub relative_loss_delta: Option<f64>,
    pub relative_loss_delta_ema: Option<f64>,
    pub action: ZSpaceOptimizerFeedbackObservationAction,
    pub gate_before: f64,
    pub gate_after: f64,
    pub state_before: ZSpaceOptimizerFeedbackState,
    pub state_after: ZSpaceOptimizerFeedbackState,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceOptimizerFeedbackControlReport {
    pub contract_version: &'static str,
    pub kind: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub control_rule: &'static str,
    pub transition_validated: bool,
    pub config: ZSpaceOptimizerFeedbackConfig,
    pub target_step: u64,
    pub proposed_learning_rate_scale: f64,
    pub proposed_deviation_from_identity: f64,
    pub feedback_observation_age_updates: Option<u64>,
    pub feedback_gate: f64,
    pub effective_feedback_gate: f64,
    pub applied_learning_rate_scale: f64,
    pub applied_deviation_from_identity: f64,
    pub identity_applied: bool,
    pub disposition: ZSpaceOptimizerFeedbackControlDisposition,
    pub state_before: ZSpaceOptimizerFeedbackState,
    pub state_after: ZSpaceOptimizerFeedbackState,
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum ZSpaceOptimizerFeedbackError {
    #[error("{field} must be finite, received {value}")]
    NonFinite { field: &'static str, value: f64 },
    #[error("{field} must be non-negative, received {value}")]
    Negative { field: &'static str, value: f64 },
    #[error("{field} must be positive, received {value}")]
    NonPositive { field: &'static str, value: f64 },
    #[error("{field} must be in [{minimum}, {maximum}], received {value}")]
    OutOfRange {
        field: &'static str,
        value: f64,
        minimum: f64,
        maximum: f64,
    },
    #[error("invalid Z-space optimizer feedback config {field}: {message}")]
    InvalidConfig {
        field: &'static str,
        message: &'static str,
    },
    #[error("invalid Z-space optimizer feedback state {field}: {message}")]
    InvalidState {
        field: &'static str,
        message: &'static str,
    },
    #[error("optimizer feedback step {step} exceeds cross-client maximum {maximum}")]
    StepLimit { step: u64, maximum: u64 },
    #[error("feedback observation step {observed} must equal completed control step {expected}")]
    ObservationStepMismatch { observed: u64, expected: u64 },
    #[error("feedback observation step {observed} is not newer than {previous}")]
    DuplicateObservation { observed: u64, previous: u64 },
    #[error("feedback control target {observed} must equal next step {expected}")]
    ControlStepMismatch { observed: u64, expected: u64 },
    #[error("training telemetry projection failed: {message}")]
    Projection { message: String },
    #[error("derived feedback field {field} is not finite")]
    DerivedNonFinite { field: &'static str },
    #[error(
        "feedback control invariant {field} exceeded tolerance: residual={residual}, tolerance={tolerance}"
    )]
    InvariantViolation {
        field: &'static str,
        residual: f64,
        tolerance: f64,
    },
}

pub fn initialize_zspace_optimizer_feedback(
    config: ZSpaceOptimizerFeedbackConfig,
) -> Result<ZSpaceOptimizerFeedbackCheckpoint, ZSpaceOptimizerFeedbackError> {
    config.validate()?;
    Ok(checkpoint(config, ZSpaceOptimizerFeedbackState::default()))
}

pub fn restore_zspace_optimizer_feedback(
    request: ZSpaceOptimizerFeedbackRestoreRequest,
) -> Result<ZSpaceOptimizerFeedbackCheckpoint, ZSpaceOptimizerFeedbackError> {
    request.config.validate()?;
    request.state.validate(&request.config)?;
    Ok(checkpoint(request.config, request.state))
}

pub fn observe_zspace_optimizer_feedback(
    request: ZSpaceOptimizerFeedbackObserveRequest,
) -> Result<ZSpaceOptimizerFeedbackObservationReport, ZSpaceOptimizerFeedbackError> {
    let config = request.config;
    let state_before = request.state;
    let observation = request.observation;
    config.validate()?;
    state_before.validate(&config)?;
    validate_observation(&observation)?;
    if observation.step != state_before.control_step {
        return Err(ZSpaceOptimizerFeedbackError::ObservationStepMismatch {
            observed: observation.step,
            expected: state_before.control_step,
        });
    }
    if let Some(previous) = state_before.last_observation_step {
        if observation.step <= previous {
            return Err(ZSpaceOptimizerFeedbackError::DuplicateObservation {
                observed: observation.step,
                previous,
            });
        }
    }

    let projection = project_training_telemetry(TrainingTelemetryProjectionRequest {
        observation: TrainingTelemetryObservation {
            step: Some(observation.step as f64),
            max_steps: observation.max_steps.map(|value| value as f64),
            epoch: observation.epoch,
            loss: Some(observation.loss),
            previous_loss: state_before.last_loss,
            grad_norm: observation.grad_norm,
            learning_rate: observation.learning_rate,
        },
        config: TrainingTelemetryProjectionConfig::default(),
    })
    .map_err(|error| ZSpaceOptimizerFeedbackError::Projection {
        message: error.to_string(),
    })?;

    let relative_loss_delta = match (projection.loss_delta, state_before.last_loss) {
        (Some(delta), Some(previous_loss)) => {
            let denominator = previous_loss.abs().max(config.loss_floor);
            Some(checked_derived("relative_loss_delta", delta / denominator)?)
        }
        _ => None,
    };
    let relative_loss_delta_ema = match relative_loss_delta {
        Some(value) => Some(ema_update(
            state_before.relative_loss_delta_ema,
            value,
            config.relative_delta_ema_alpha,
            "relative_loss_delta_ema",
        )?),
        None => None,
    };
    let loss_ema = ema_update(
        state_before.loss_ema,
        observation.loss,
        config.loss_ema_alpha,
        "loss_ema",
    )?;

    let mut state_after = state_before.clone();
    state_after.observation_count =
        checked_increment("state.observation_count", state_before.observation_count)?;
    state_after.last_observation_step = Some(observation.step);
    state_after.last_loss = Some(observation.loss);
    state_after.loss_ema = Some(loss_ema);
    state_after.relative_loss_delta_ema = relative_loss_delta_ema;
    let gate_before = state_before.gate;

    let action = match relative_loss_delta_ema {
        None => {
            state_after.gate = 0.0;
            state_after.regression_streak = 0;
            state_after.improvement_streak = 0;
            ZSpaceOptimizerFeedbackObservationAction::Initialize
        }
        Some(_) if state_after.observation_count <= config.warmup_observations => {
            state_after.gate = 0.0;
            state_after.regression_streak = 0;
            state_after.improvement_streak = 0;
            state_after.halted = false;
            ZSpaceOptimizerFeedbackObservationAction::Warmup
        }
        Some(delta_ema) => update_gate_from_delta(&config, &mut state_after, delta_ema)?,
    };
    state_after.validate(&config)?;

    Ok(ZSpaceOptimizerFeedbackObservationReport {
        contract_version: ZSPACE_OPTIMIZER_FEEDBACK_CONTRACT_VERSION,
        kind: ZSPACE_OPTIMIZER_FEEDBACK_KIND,
        semantic_owner: ZSPACE_OPTIMIZER_FEEDBACK_SEMANTIC_OWNER,
        semantic_backend: ZSPACE_OPTIMIZER_FEEDBACK_SEMANTIC_BACKEND,
        transition_validated: true,
        config,
        observation,
        projection,
        relative_loss_delta,
        relative_loss_delta_ema,
        action,
        gate_before,
        gate_after: state_after.gate,
        state_before,
        state_after,
    })
}

pub fn control_zspace_optimizer_feedback(
    request: ZSpaceOptimizerFeedbackControlRequest,
) -> Result<ZSpaceOptimizerFeedbackControlReport, ZSpaceOptimizerFeedbackError> {
    let config = request.config;
    let state_before = request.state;
    config.validate()?;
    state_before.validate(&config)?;
    require_range(
        "request.proposed_learning_rate_scale",
        request.proposed_learning_rate_scale,
        ZSPACE_PARAMETER_CONTROL_MIN_LEARNING_RATE_SCALE,
        ZSPACE_PARAMETER_CONTROL_MAX_LEARNING_RATE_SCALE,
    )?;
    let expected_target = state_before.control_step.checked_add(1).ok_or(
        ZSpaceOptimizerFeedbackError::StepLimit {
            step: state_before.control_step,
            maximum: ZSPACE_META_OPTIMIZER_MAX_SAFE_STEP,
        },
    )?;
    if expected_target > ZSPACE_META_OPTIMIZER_MAX_SAFE_STEP {
        return Err(ZSpaceOptimizerFeedbackError::StepLimit {
            step: expected_target,
            maximum: ZSPACE_META_OPTIMIZER_MAX_SAFE_STEP,
        });
    }
    if request.target_step != expected_target {
        return Err(ZSpaceOptimizerFeedbackError::ControlStepMismatch {
            observed: request.target_step,
            expected: expected_target,
        });
    }

    let observation_age = state_before
        .last_observation_step
        .map(|step| state_before.control_step - step);
    let disposition = if state_before.observation_count == 0 {
        ZSpaceOptimizerFeedbackControlDisposition::NoFeedback
    } else if state_before.observation_count <= config.warmup_observations {
        ZSpaceOptimizerFeedbackControlDisposition::Warmup
    } else if state_before.halted {
        ZSpaceOptimizerFeedbackControlDisposition::Halted
    } else if observation_age.is_some_and(|age| age > config.max_stale_updates) {
        ZSpaceOptimizerFeedbackControlDisposition::Stale
    } else if state_before.gate == 0.0 {
        ZSpaceOptimizerFeedbackControlDisposition::IdentityGate
    } else {
        ZSpaceOptimizerFeedbackControlDisposition::Active
    };
    let effective_gate = if disposition == ZSpaceOptimizerFeedbackControlDisposition::Active {
        state_before.gate
    } else {
        0.0
    };
    let proposed_deviation = checked_derived(
        "proposed_deviation_from_identity",
        request.proposed_learning_rate_scale - 1.0,
    )?;
    let applied_deviation = checked_derived(
        "applied_deviation_from_identity",
        effective_gate * proposed_deviation,
    )?;
    let applied_scale = checked_derived("applied_learning_rate_scale", 1.0 + applied_deviation)?;
    require_range(
        "applied_learning_rate_scale",
        applied_scale,
        ZSPACE_PARAMETER_CONTROL_MIN_LEARNING_RATE_SCALE,
        ZSPACE_PARAMETER_CONTROL_MAX_LEARNING_RATE_SCALE,
    )?;
    let expected_applied = 1.0 + effective_gate * proposed_deviation;
    let residual = (applied_scale - expected_applied).abs();
    let tolerance = DERIVED_TOLERANCE * expected_applied.abs().max(1.0);
    if residual > tolerance {
        return Err(ZSpaceOptimizerFeedbackError::InvariantViolation {
            field: "control_rule",
            residual,
            tolerance,
        });
    }

    let mut state_after = state_before.clone();
    state_after.control_step = request.target_step;
    state_after.validate(&config)?;

    Ok(ZSpaceOptimizerFeedbackControlReport {
        contract_version: ZSPACE_OPTIMIZER_FEEDBACK_CONTRACT_VERSION,
        kind: ZSPACE_OPTIMIZER_FEEDBACK_KIND,
        semantic_owner: ZSPACE_OPTIMIZER_FEEDBACK_SEMANTIC_OWNER,
        semantic_backend: ZSPACE_OPTIMIZER_FEEDBACK_SEMANTIC_BACKEND,
        control_rule: ZSPACE_OPTIMIZER_FEEDBACK_CONTROL_RULE,
        transition_validated: true,
        config,
        target_step: request.target_step,
        proposed_learning_rate_scale: request.proposed_learning_rate_scale,
        proposed_deviation_from_identity: proposed_deviation,
        feedback_observation_age_updates: observation_age,
        feedback_gate: state_before.gate,
        effective_feedback_gate: effective_gate,
        applied_learning_rate_scale: applied_scale,
        applied_deviation_from_identity: applied_deviation,
        identity_applied: applied_deviation == 0.0,
        disposition,
        state_before,
        state_after,
    })
}

fn checkpoint(
    config: ZSpaceOptimizerFeedbackConfig,
    state: ZSpaceOptimizerFeedbackState,
) -> ZSpaceOptimizerFeedbackCheckpoint {
    ZSpaceOptimizerFeedbackCheckpoint {
        contract_version: ZSPACE_OPTIMIZER_FEEDBACK_CONTRACT_VERSION,
        kind: ZSPACE_OPTIMIZER_FEEDBACK_KIND,
        semantic_owner: ZSPACE_OPTIMIZER_FEEDBACK_SEMANTIC_OWNER,
        semantic_backend: ZSPACE_OPTIMIZER_FEEDBACK_SEMANTIC_BACKEND,
        config,
        state,
    }
}

fn update_gate_from_delta(
    config: &ZSpaceOptimizerFeedbackConfig,
    state: &mut ZSpaceOptimizerFeedbackState,
    delta_ema: f64,
) -> Result<ZSpaceOptimizerFeedbackObservationAction, ZSpaceOptimizerFeedbackError> {
    if delta_ema >= config.halt_threshold {
        state.regression_streak =
            checked_increment("state.regression_streak", state.regression_streak)?;
        state.improvement_streak = 0;
        state.gate = 0.0;
        state.halted = true;
        return Ok(ZSpaceOptimizerFeedbackObservationAction::Halt);
    }
    if delta_ema > config.regression_threshold {
        state.regression_streak =
            checked_increment("state.regression_streak", state.regression_streak)?;
        state.improvement_streak = 0;
        if state.regression_streak >= config.halt_regression_streak {
            state.gate = 0.0;
            state.halted = true;
            return Ok(ZSpaceOptimizerFeedbackObservationAction::Halt);
        }
        if state.halted {
            state.gate = 0.0;
            return Ok(ZSpaceOptimizerFeedbackObservationAction::HoldHalted);
        }
        state.gate = (state.gate - config.attenuation_rate).max(0.0);
        return Ok(ZSpaceOptimizerFeedbackObservationAction::Attenuate);
    }
    if delta_ema < -config.recovery_threshold {
        state.improvement_streak =
            checked_increment("state.improvement_streak", state.improvement_streak)?;
        state.regression_streak = 0;
        if state.halted && state.improvement_streak < config.resume_improvement_streak {
            state.gate = 0.0;
            return Ok(ZSpaceOptimizerFeedbackObservationAction::HoldHalted);
        }
        state.halted = false;
        state.gate = (state.gate + config.recovery_rate).min(config.maximum_gate);
        return Ok(ZSpaceOptimizerFeedbackObservationAction::Recover);
    }
    state.regression_streak = 0;
    state.improvement_streak = 0;
    if state.halted {
        state.gate = 0.0;
        Ok(ZSpaceOptimizerFeedbackObservationAction::HoldHalted)
    } else {
        Ok(ZSpaceOptimizerFeedbackObservationAction::Hold)
    }
}

fn validate_observation(
    observation: &ZSpaceOptimizerFeedbackObservation,
) -> Result<(), ZSpaceOptimizerFeedbackError> {
    require_safe_count("observation.step", observation.step)?;
    if let Some(max_steps) = observation.max_steps {
        if max_steps == 0 {
            return Err(ZSpaceOptimizerFeedbackError::InvalidState {
                field: "observation.max_steps",
                message: "must be positive",
            });
        }
        require_safe_count("observation.max_steps", max_steps)?;
        if observation.step > max_steps {
            return Err(ZSpaceOptimizerFeedbackError::InvalidState {
                field: "observation.step",
                message: "cannot exceed max_steps",
            });
        }
    }
    require_finite("observation.loss", observation.loss)?;
    if let Some(epoch) = observation.epoch {
        require_non_negative("observation.epoch", epoch)?;
    }
    if let Some(grad_norm) = observation.grad_norm {
        require_non_negative("observation.grad_norm", grad_norm)?;
    }
    if let Some(learning_rate) = observation.learning_rate {
        require_non_negative("observation.learning_rate", learning_rate)?;
    }
    Ok(())
}

fn ema_update(
    previous: Option<f64>,
    value: f64,
    alpha: f64,
    field: &'static str,
) -> Result<f64, ZSpaceOptimizerFeedbackError> {
    let updated = match previous {
        Some(previous) => previous + alpha * (value - previous),
        None => value,
    };
    checked_derived(field, updated)
}

fn checked_increment(field: &'static str, value: u64) -> Result<u64, ZSpaceOptimizerFeedbackError> {
    let next = value
        .checked_add(1)
        .ok_or(ZSpaceOptimizerFeedbackError::StepLimit {
            step: value,
            maximum: ZSPACE_META_OPTIMIZER_MAX_SAFE_STEP,
        })?;
    require_safe_count(field, next)?;
    Ok(next)
}

fn checked_derived(field: &'static str, value: f64) -> Result<f64, ZSpaceOptimizerFeedbackError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ZSpaceOptimizerFeedbackError::DerivedNonFinite { field })
    }
}

fn require_finite(field: &'static str, value: f64) -> Result<(), ZSpaceOptimizerFeedbackError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ZSpaceOptimizerFeedbackError::NonFinite { field, value })
    }
}

fn require_non_negative(
    field: &'static str,
    value: f64,
) -> Result<(), ZSpaceOptimizerFeedbackError> {
    require_finite(field, value)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(ZSpaceOptimizerFeedbackError::Negative { field, value })
    }
}

fn require_positive(field: &'static str, value: f64) -> Result<(), ZSpaceOptimizerFeedbackError> {
    require_finite(field, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(ZSpaceOptimizerFeedbackError::NonPositive { field, value })
    }
}

fn require_range(
    field: &'static str,
    value: f64,
    minimum: f64,
    maximum: f64,
) -> Result<(), ZSpaceOptimizerFeedbackError> {
    require_finite(field, value)?;
    if (minimum..=maximum).contains(&value) {
        Ok(())
    } else {
        Err(ZSpaceOptimizerFeedbackError::OutOfRange {
            field,
            value,
            minimum,
            maximum,
        })
    }
}

fn require_unit_interval_open_zero(
    field: &'static str,
    value: f64,
) -> Result<(), ZSpaceOptimizerFeedbackError> {
    require_finite(field, value)?;
    if value > 0.0 && value <= 1.0 {
        Ok(())
    } else {
        Err(ZSpaceOptimizerFeedbackError::OutOfRange {
            field,
            value,
            minimum: f64::EPSILON,
            maximum: 1.0,
        })
    }
}

fn require_positive_count(
    field: &'static str,
    value: u64,
) -> Result<(), ZSpaceOptimizerFeedbackError> {
    if value == 0 {
        return Err(ZSpaceOptimizerFeedbackError::InvalidConfig {
            field,
            message: "must be positive",
        });
    }
    require_safe_count(field, value)
}

fn require_safe_count(
    _field: &'static str,
    value: u64,
) -> Result<(), ZSpaceOptimizerFeedbackError> {
    if value <= ZSPACE_META_OPTIMIZER_MAX_SAFE_STEP {
        Ok(())
    } else {
        Err(ZSpaceOptimizerFeedbackError::StepLimit {
            step: value,
            maximum: ZSPACE_META_OPTIMIZER_MAX_SAFE_STEP,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn control(
        config: &ZSpaceOptimizerFeedbackConfig,
        state: ZSpaceOptimizerFeedbackState,
        target_step: u64,
        proposed: f64,
    ) -> ZSpaceOptimizerFeedbackControlReport {
        control_zspace_optimizer_feedback(ZSpaceOptimizerFeedbackControlRequest {
            config: config.clone(),
            state,
            target_step,
            proposed_learning_rate_scale: proposed,
        })
        .expect("control")
    }

    fn observe(
        config: &ZSpaceOptimizerFeedbackConfig,
        state: ZSpaceOptimizerFeedbackState,
        loss: f64,
    ) -> ZSpaceOptimizerFeedbackObservationReport {
        observe_zspace_optimizer_feedback(ZSpaceOptimizerFeedbackObserveRequest {
            config: config.clone(),
            observation: ZSpaceOptimizerFeedbackObservation {
                step: state.control_step,
                max_steps: Some(32),
                epoch: None,
                loss,
                grad_norm: Some(1.0),
                learning_rate: Some(1.0e-4),
            },
            state,
        })
        .expect("observation")
    }

    #[test]
    fn feedback_starts_at_identity_and_only_blends_the_proposed_deviation() {
        let config = ZSpaceOptimizerFeedbackConfig::default();
        let checkpoint = initialize_zspace_optimizer_feedback(config.clone()).unwrap();
        let report = control(&config, checkpoint.state, 1, 0.5);

        assert_eq!(
            report.disposition,
            ZSpaceOptimizerFeedbackControlDisposition::NoFeedback
        );
        assert_eq!(report.effective_feedback_gate, 0.0);
        assert_eq!(report.applied_learning_rate_scale, 1.0);
        assert!(report.identity_applied);

        let mut state = report.state_after;
        state.control_step = 2;
        state.observation_count = 3;
        state.last_observation_step = Some(2);
        state.last_loss = Some(1.0);
        state.loss_ema = Some(1.0);
        state.relative_loss_delta_ema = Some(-0.1);
        state.gate = 0.25;
        let report = control(&config, state, 3, 0.5);

        assert_eq!(
            report.disposition,
            ZSpaceOptimizerFeedbackControlDisposition::Active
        );
        assert_relative_eq!(report.applied_learning_rate_scale, 0.875);
        assert_relative_eq!(report.applied_deviation_from_identity, -0.125);
    }

    #[test]
    fn improvements_open_the_gate_and_regressions_halt_it() {
        let config = ZSpaceOptimizerFeedbackConfig {
            warmup_observations: 1,
            relative_delta_ema_alpha: 1.0,
            recovery_rate: 0.25,
            attenuation_rate: 0.25,
            ..ZSpaceOptimizerFeedbackConfig::default()
        };
        let mut state = initialize_zspace_optimizer_feedback(config.clone())
            .unwrap()
            .state;

        state = control(&config, state, 1, 0.8).state_after;
        let first = observe(&config, state, 2.0);
        assert_eq!(
            first.action,
            ZSpaceOptimizerFeedbackObservationAction::Initialize
        );
        state = first.state_after;

        state = control(&config, state, 2, 0.8).state_after;
        let improving = observe(&config, state, 1.8);
        assert_eq!(
            improving.action,
            ZSpaceOptimizerFeedbackObservationAction::Recover
        );
        assert_relative_eq!(improving.gate_after, 0.25);
        state = improving.state_after;

        let active = control(&config, state, 3, 0.8);
        assert_eq!(
            active.disposition,
            ZSpaceOptimizerFeedbackControlDisposition::Active
        );
        assert_relative_eq!(active.applied_learning_rate_scale, 0.95);
        let regression = observe(&config, active.state_after, 1.9);
        assert_eq!(
            regression.action,
            ZSpaceOptimizerFeedbackObservationAction::Halt
        );
        assert!(regression.state_after.halted);
        assert_eq!(regression.state_after.gate, 0.0);

        let halted = control(&config, regression.state_after, 4, 0.8);
        assert_eq!(
            halted.disposition,
            ZSpaceOptimizerFeedbackControlDisposition::Halted
        );
        assert_eq!(halted.applied_learning_rate_scale, 1.0);
    }

    #[test]
    fn a_delta_equal_to_equal_regression_and_halt_thresholds_halts() {
        let config = ZSpaceOptimizerFeedbackConfig {
            warmup_observations: 0,
            relative_delta_ema_alpha: 1.0,
            regression_threshold: 0.125,
            halt_threshold: 0.125,
            recovery_rate: 0.5,
            ..ZSpaceOptimizerFeedbackConfig::default()
        };
        let mut state = ZSpaceOptimizerFeedbackState::default();

        state = control(&config, state, 1, 0.8).state_after;
        state = observe(&config, state, 1.0).state_after;
        state = control(&config, state, 2, 0.8).state_after;
        state = observe(&config, state, 0.5).state_after;
        assert_eq!(state.gate, 0.5);
        state = control(&config, state, 3, 0.8).state_after;

        let report = observe(&config, state, 0.5625);

        assert_eq!(report.relative_loss_delta_ema, Some(0.125));
        assert_eq!(
            report.action,
            ZSpaceOptimizerFeedbackObservationAction::Halt
        );
        assert!(report.state_after.halted);
        assert_eq!(report.state_after.gate, 0.0);
    }

    #[test]
    fn stale_feedback_returns_to_identity_without_erasing_learned_gate() {
        let config = ZSpaceOptimizerFeedbackConfig {
            warmup_observations: 0,
            max_stale_updates: 0,
            ..ZSpaceOptimizerFeedbackConfig::default()
        };
        let mut state = ZSpaceOptimizerFeedbackState {
            control_step: 1,
            observation_count: 2,
            last_observation_step: Some(1),
            last_loss: Some(1.0),
            loss_ema: Some(1.0),
            relative_loss_delta_ema: Some(-0.1),
            gate: 0.5,
            ..ZSpaceOptimizerFeedbackState::default()
        };
        let fresh = control(&config, state, 2, 1.2);
        assert_eq!(
            fresh.disposition,
            ZSpaceOptimizerFeedbackControlDisposition::Active
        );
        assert_relative_eq!(fresh.applied_learning_rate_scale, 1.1);
        state = fresh.state_after;

        let stale = control(&config, state, 3, 1.2);
        assert_eq!(
            stale.disposition,
            ZSpaceOptimizerFeedbackControlDisposition::Stale
        );
        assert_eq!(stale.applied_learning_rate_scale, 1.0);
        assert_eq!(stale.state_after.gate, 0.5);
    }

    #[test]
    fn checkpoint_restore_replays_the_same_observation_and_control() {
        let config = ZSpaceOptimizerFeedbackConfig {
            warmup_observations: 0,
            ..ZSpaceOptimizerFeedbackConfig::default()
        };
        let state = control(&config, ZSpaceOptimizerFeedbackState::default(), 1, 0.75).state_after;
        let observed = observe(&config, state, 2.0);
        let restored = restore_zspace_optimizer_feedback(ZSpaceOptimizerFeedbackRestoreRequest {
            config: config.clone(),
            state: observed.state_after.clone(),
        })
        .unwrap();

        let direct = control(&config, observed.state_after, 2, 0.75);
        let replayed = control(&config, restored.state, 2, 0.75);
        assert_eq!(direct, replayed);
    }

    #[test]
    fn invalid_steps_scales_and_halted_state_fail_closed() {
        let config = ZSpaceOptimizerFeedbackConfig::default();
        let state = ZSpaceOptimizerFeedbackState::default();
        assert!(matches!(
            control_zspace_optimizer_feedback(ZSpaceOptimizerFeedbackControlRequest {
                config: config.clone(),
                state: state.clone(),
                target_step: 2,
                proposed_learning_rate_scale: 1.0,
            }),
            Err(ZSpaceOptimizerFeedbackError::ControlStepMismatch { .. })
        ));
        assert!(matches!(
            control_zspace_optimizer_feedback(ZSpaceOptimizerFeedbackControlRequest {
                config: config.clone(),
                state: state.clone(),
                target_step: 1,
                proposed_learning_rate_scale: 2.0,
            }),
            Err(ZSpaceOptimizerFeedbackError::OutOfRange { .. })
        ));

        let invalid_warmup = ZSpaceOptimizerFeedbackState {
            observation_count: 1,
            last_observation_step: Some(0),
            last_loss: Some(1.0),
            loss_ema: Some(1.0),
            gate: 0.5,
            ..ZSpaceOptimizerFeedbackState::default()
        };
        assert!(matches!(
            restore_zspace_optimizer_feedback(ZSpaceOptimizerFeedbackRestoreRequest {
                config: config.clone(),
                state: invalid_warmup,
            }),
            Err(ZSpaceOptimizerFeedbackError::InvalidState { .. })
        ));

        let invalid = ZSpaceOptimizerFeedbackState {
            observation_count: 1,
            last_observation_step: Some(0),
            last_loss: Some(1.0),
            loss_ema: Some(1.0),
            gate: 0.5,
            halted: true,
            ..ZSpaceOptimizerFeedbackState::default()
        };
        assert!(matches!(
            restore_zspace_optimizer_feedback(ZSpaceOptimizerFeedbackRestoreRequest {
                config,
                state: invalid,
            }),
            Err(ZSpaceOptimizerFeedbackError::InvalidState { .. })
        ));
    }

    #[test]
    fn rust_state_supplies_previous_loss_to_the_projection() {
        let config = ZSpaceOptimizerFeedbackConfig {
            warmup_observations: 0,
            relative_delta_ema_alpha: 1.0,
            ..ZSpaceOptimizerFeedbackConfig::default()
        };
        let state = control(&config, ZSpaceOptimizerFeedbackState::default(), 1, 1.0).state_after;
        let first = observe(&config, state, 4.0);
        let state = control(&config, first.state_after, 2, 1.0).state_after;
        let second = observe(&config, state, 3.0);

        assert_eq!(second.projection.previous_loss, Some(4.0));
        assert_eq!(second.projection.loss_delta, Some(-1.0));
        assert_relative_eq!(second.relative_loss_delta.unwrap(), -0.25);
    }

    #[test]
    fn fixed_loss_replay_recovers_attenuates_and_halts_toward_identity() {
        // The losses are exogenous: this verifies guard response, not training efficacy.
        let config = ZSpaceOptimizerFeedbackConfig {
            warmup_observations: 1,
            relative_delta_ema_alpha: 1.0,
            recovery_rate: 0.5,
            attenuation_rate: 0.25,
            halt_regression_streak: 2,
            ..ZSpaceOptimizerFeedbackConfig::default()
        };
        let mut state = ZSpaceOptimizerFeedbackState::default();
        let mut applied = Vec::new();

        let first = control(&config, state, 1, 0.8);
        applied.push(first.applied_learning_rate_scale);
        state = observe(&config, first.state_after, 2.0).state_after;

        let warmup = control(&config, state, 2, 0.8);
        applied.push(warmup.applied_learning_rate_scale);
        state = observe(&config, warmup.state_after, 1.8).state_after;

        let recovered = control(&config, state, 3, 0.8);
        applied.push(recovered.applied_learning_rate_scale);
        let attenuated = observe(&config, recovered.state_after, 1.83);
        assert_eq!(
            attenuated.action,
            ZSpaceOptimizerFeedbackObservationAction::Attenuate
        );

        let reduced = control(&config, attenuated.state_after, 4, 0.8);
        applied.push(reduced.applied_learning_rate_scale);
        let halted = observe(&config, reduced.state_after, 1.87);
        assert_eq!(
            halted.action,
            ZSpaceOptimizerFeedbackObservationAction::Halt
        );

        let identity = control(&config, halted.state_after, 5, 0.8);
        applied.push(identity.applied_learning_rate_scale);

        assert_eq!(
            identity.disposition,
            ZSpaceOptimizerFeedbackControlDisposition::Halted
        );
        assert_eq!(applied, vec![1.0, 1.0, 0.9, 0.95, 1.0]);
    }
}
