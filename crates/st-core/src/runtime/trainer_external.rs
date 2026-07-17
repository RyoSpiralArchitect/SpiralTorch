//! Versioned state contract for trainer components outside the optimizer core.
//!
//! Rust owns validation and component accounting. Native orchestrators may
//! restore concrete resources; browser clients only preflight the same payload.

use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use thiserror::Error;

use crate::distributed::AccumulatorSynchronizerCheckpoint;
use crate::inference::gnn_roundtable::{
    derive_gnn_roundtable_influence, GnnRoundtableInfluencePayload, GnnRoundtableSignalObservation,
    GNN_ROUNDTABLE_MAX_HISTORY_SIGNALS,
};
use crate::inference::zspace_coherence::{
    classify_zspace_coherence, derive_zspace_coherence_control, is_zspace_coherence_swap_invariant,
    validate_zspace_coherence_distribution_observation, ZSpaceCoherenceClassificationPayload,
    ZSpaceCoherenceClassificationPolicy, ZSpaceCoherenceClassificationRequest,
    ZSpaceCoherenceControlPayload, ZSpaceCoherenceDistributionWitness,
};

pub const TRAINER_EXTERNAL_CHECKPOINT_KIND: &str = "spiraltorch.trainer_external_state_checkpoint";
pub const TRAINER_EXTERNAL_CHECKPOINT_CONTRACT_VERSION: &str =
    "spiraltorch.trainer_external_state_checkpoint.v5";
pub const TRAINER_EXTERNAL_CHECKPOINT_SEMANTIC_OWNER: &str = "st-core::runtime::trainer_external";
pub const TRAINER_EXTERNAL_CHECKPOINT_SEMANTIC_BACKEND: &str = "rust";
pub const TRAINER_EXTERNAL_MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

pub const DESIRE_TRAINER_COMPONENT: &str = "desire_bridge";
pub const DESIRE_ROUNDTABLE_COMPONENT: &str = "desire_roundtable_bridge";
pub const COHERENCE_BRIDGE_COMPONENT: &str = "coherence_bridge";
pub const GNN_ROUNDTABLE_BRIDGE_COMPONENT: &str = "gnn_roundtable_bridge";
pub const PSI_METER_COMPONENT: &str = "psi_meter";
pub const ACCUMULATOR_SYNCHRONIZER_COMPONENT: &str = "accumulator_synchronizer";

pub const PSI_METER_CHECKPOINT_KIND: &str = "spiraltorch.psi_meter_checkpoint";
pub const PSI_METER_CHECKPOINT_CONTRACT_VERSION: &str = "spiraltorch.psi_meter_checkpoint.v1";
pub const PSI_METER_CHECKPOINT_SEMANTIC_OWNER: &str = "st-core::telemetry::psi";
pub const PSI_METER_CHECKPOINT_SEMANTIC_BACKEND: &str = "rust";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainerTimestampCheckpoint {
    pub unix_seconds: u64,
    pub subsec_nanos: u32,
}

impl TrainerTimestampCheckpoint {
    /// Builds an exact portable timestamp from its duration since the Unix epoch.
    pub fn try_from_unix_duration(
        field: &str,
        duration: Duration,
    ) -> Result<Self, TrainerExternalCheckpointError> {
        let checkpoint = Self {
            unix_seconds: duration.as_secs(),
            subsec_nanos: duration.subsec_nanos(),
        };
        checkpoint.validate(field)?;
        Ok(checkpoint)
    }

    pub fn try_from_system_time(
        field: &str,
        timestamp: SystemTime,
    ) -> Result<Self, TrainerExternalCheckpointError> {
        let duration = timestamp
            .duration_since(UNIX_EPOCH)
            .map_err(|_| state_error(field, "must not precede the Unix epoch"))?;
        Self::try_from_unix_duration(field, duration)
    }

    /// Returns the exact portable duration represented by this timestamp.
    pub fn try_to_unix_duration(
        self,
        field: &str,
    ) -> Result<Duration, TrainerExternalCheckpointError> {
        self.validate(field)?;
        Ok(Duration::new(self.unix_seconds, self.subsec_nanos))
    }

    pub fn try_to_system_time(
        self,
        field: &str,
    ) -> Result<SystemTime, TrainerExternalCheckpointError> {
        let duration = self.try_to_unix_duration(field)?;
        UNIX_EPOCH
            .checked_add(duration)
            .ok_or_else(|| state_error(field, "overflows SystemTime"))
    }

    /// Validates the cross-runtime integer and nanosecond bounds.
    pub fn validate(&self, field: &str) -> Result<(), TrainerExternalCheckpointError> {
        if self.unix_seconds > TRAINER_EXTERNAL_MAX_SAFE_INTEGER {
            return Err(state_error(
                &format!("{field}.unix_seconds"),
                "must not exceed JavaScript's exact integer limit",
            ));
        }
        if self.subsec_nanos >= 1_000_000_000 {
            return Err(state_error(
                &format!("{field}.subsec_nanos"),
                "must be less than one second",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DesireTrainerPhaseCheckpoint {
    Observation,
    Injection,
    Integration,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DesireTrainerWeightsCheckpoint {
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
    pub lambda: f32,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DesireTrainerTriggerCheckpoint {
    pub report_tokens: Vec<u64>,
    pub report_scores: Vec<f32>,
    pub mean_penalty: f32,
    pub mean_entropy: f32,
    pub temperature: f32,
    pub samples: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DesireTrainerEventCheckpoint {
    pub timestamp: TrainerTimestampCheckpoint,
    pub phase: DesireTrainerPhaseCheckpoint,
    pub temperature: f32,
    pub entropy: f32,
    pub hypergrad_penalty: f32,
    pub weights: DesireTrainerWeightsCheckpoint,
    pub trigger: Option<DesireTrainerTriggerCheckpoint>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DesireTrainerQueueCheckpoint {
    /// Events remain in the exact FIFO order consumed by the next trainer step.
    pub events: Vec<DesireTrainerEventCheckpoint>,
}

impl DesireTrainerQueueCheckpoint {
    pub fn validate(&self) -> Result<(), TrainerExternalCheckpointError> {
        for (event_index, event) in self.events.iter().enumerate() {
            let prefix = format!("desire_trainer.events[{event_index}]");
            event.timestamp.validate(&format!("{prefix}.timestamp"))?;
            validate_positive(&format!("{prefix}.temperature"), event.temperature)?;
            validate_non_negative(&format!("{prefix}.entropy"), event.entropy)?;
            validate_non_negative(
                &format!("{prefix}.hypergrad_penalty"),
                event.hypergrad_penalty,
            )?;
            for (label, value) in [
                ("alpha", event.weights.alpha),
                ("beta", event.weights.beta),
                ("gamma", event.weights.gamma),
                ("lambda", event.weights.lambda),
            ] {
                validate_non_negative(&format!("{prefix}.weights.{label}"), value)?;
            }
            if let Some(trigger) = &event.trigger {
                validate_desire_trainer_trigger(&prefix, trigger)?;
            }
        }
        Ok(())
    }
}

fn validate_desire_trainer_trigger(
    event_prefix: &str,
    trigger: &DesireTrainerTriggerCheckpoint,
) -> Result<(), TrainerExternalCheckpointError> {
    let prefix = format!("{event_prefix}.trigger");
    if trigger.report_tokens.len() != trigger.report_scores.len() {
        return Err(state_error(
            &format!("{prefix}.report_tokens"),
            "must have the same length as report_scores",
        ));
    }
    for (index, token) in trigger.report_tokens.iter().enumerate() {
        if *token > TRAINER_EXTERNAL_MAX_SAFE_INTEGER {
            return Err(state_error(
                &format!("{prefix}.report_tokens[{index}]"),
                "must not exceed JavaScript's exact integer limit",
            ));
        }
        if trigger.report_tokens[..index].contains(token) {
            return Err(state_error(
                &format!("{prefix}.report_tokens[{index}]"),
                "must be unique within the trigger report",
            ));
        }
    }
    for (index, score) in trigger.report_scores.iter().copied().enumerate() {
        validate_non_negative(&format!("{prefix}.report_scores[{index}]"), score)?;
    }
    validate_non_negative(&format!("{prefix}.mean_penalty"), trigger.mean_penalty)?;
    validate_non_negative(&format!("{prefix}.mean_entropy"), trigger.mean_entropy)?;
    validate_positive(&format!("{prefix}.temperature"), trigger.temperature)?;
    if trigger.samples == 0 {
        return Err(state_error(
            &format!("{prefix}.samples"),
            "must be positive",
        ));
    }
    if trigger.samples > TRAINER_EXTERNAL_MAX_SAFE_INTEGER {
        return Err(state_error(
            &format!("{prefix}.samples"),
            "must not exceed JavaScript's exact integer limit",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PsiMeterComponent {
    ActDrift,
    AttnEntropy,
    BandEnergy,
    GradNorm,
    Loss,
    PositiveCurvature,
    UpdateRatio,
}

impl PsiMeterComponent {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ActDrift => "act_drift",
            Self::AttnEntropy => "attn_entropy",
            Self::BandEnergy => "band_energy",
            Self::GradNorm => "grad_norm",
            Self::Loss => "loss",
            Self::PositiveCurvature => "positive_curvature",
            Self::UpdateRatio => "update_ratio",
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PsiMeterComponentValue {
    pub component: PsiMeterComponent,
    pub value: f32,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PsiMeterCheckpoint {
    pub kind: String,
    pub contract_version: String,
    pub semantic_owner: String,
    pub semantic_backend: String,
    pub enabled: bool,
    pub components: Vec<PsiMeterComponent>,
    pub weights: Vec<PsiMeterComponentValue>,
    pub ema_alpha: f32,
    pub sample_rate: u32,
    pub thresholds: Vec<PsiMeterComponentValue>,
    pub ema: Vec<PsiMeterComponentValue>,
    pub step: u64,
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum PsiMeterCheckpointError {
    #[error("psi meter checkpoint {field} must be {expected}, got {actual}")]
    InvalidMetadata {
        field: &'static str,
        expected: &'static str,
        actual: String,
    },
    #[error("invalid psi meter checkpoint state at {field}: {message}")]
    InvalidState { field: String, message: String },
}

impl PsiMeterCheckpointError {
    pub(crate) fn invalid_state(field: &str, message: &str) -> Self {
        Self::InvalidState {
            field: field.to_owned(),
            message: message.to_owned(),
        }
    }
}

impl PsiMeterCheckpoint {
    pub fn validate(&self) -> Result<(), PsiMeterCheckpointError> {
        validate_psi_metadata("kind", &self.kind, PSI_METER_CHECKPOINT_KIND)?;
        validate_psi_metadata(
            "contract_version",
            &self.contract_version,
            PSI_METER_CHECKPOINT_CONTRACT_VERSION,
        )?;
        validate_psi_metadata(
            "semantic_owner",
            &self.semantic_owner,
            PSI_METER_CHECKPOINT_SEMANTIC_OWNER,
        )?;
        validate_psi_metadata(
            "semantic_backend",
            &self.semantic_backend,
            PSI_METER_CHECKPOINT_SEMANTIC_BACKEND,
        )?;
        validate_psi_components("components", &self.components, false)?;
        if !self.ema_alpha.is_finite() || !(0.0..1.0).contains(&self.ema_alpha) {
            return Err(PsiMeterCheckpointError::invalid_state(
                "ema_alpha",
                "must be finite and in (0, 1)",
            ));
        }
        if self.sample_rate == 0 {
            return Err(PsiMeterCheckpointError::invalid_state(
                "sample_rate",
                "must be positive",
            ));
        }
        if self.step > TRAINER_EXTERNAL_MAX_SAFE_INTEGER {
            return Err(PsiMeterCheckpointError::invalid_state(
                "step",
                "must not exceed JavaScript's exact integer limit",
            ));
        }
        validate_psi_component_values("weights", &self.weights, true, None)?;
        validate_psi_component_values("thresholds", &self.thresholds, false, None)?;
        validate_psi_component_values("ema", &self.ema, false, Some(&self.components))?;
        Ok(())
    }
}

fn validate_psi_metadata(
    field: &'static str,
    actual: &str,
    expected: &'static str,
) -> Result<(), PsiMeterCheckpointError> {
    if actual == expected {
        Ok(())
    } else {
        Err(PsiMeterCheckpointError::InvalidMetadata {
            field,
            expected,
            actual: actual.to_owned(),
        })
    }
}

fn validate_psi_components(
    field: &str,
    components: &[PsiMeterComponent],
    allow_empty: bool,
) -> Result<(), PsiMeterCheckpointError> {
    if !allow_empty && components.is_empty() {
        return Err(PsiMeterCheckpointError::invalid_state(
            field,
            "must contain at least one component",
        ));
    }
    let mut previous = None;
    for component in components {
        if previous.is_some_and(|value| value >= *component) {
            return Err(PsiMeterCheckpointError::invalid_state(
                field,
                "components must be sorted and unique",
            ));
        }
        previous = Some(*component);
    }
    Ok(())
}

fn validate_psi_component_values(
    field: &str,
    entries: &[PsiMeterComponentValue],
    non_negative: bool,
    allowed: Option<&[PsiMeterComponent]>,
) -> Result<(), PsiMeterCheckpointError> {
    let components = entries
        .iter()
        .map(|entry| entry.component)
        .collect::<Vec<_>>();
    validate_psi_components(field, &components, true)?;
    for entry in entries {
        if !entry.value.is_finite() {
            return Err(PsiMeterCheckpointError::invalid_state(
                &format!("{field}.{}", entry.component.as_str()),
                "must be finite",
            ));
        }
        if non_negative && entry.value < 0.0 {
            return Err(PsiMeterCheckpointError::invalid_state(
                &format!("{field}.{}", entry.component.as_str()),
                "must be non-negative",
            ));
        }
        if allowed.is_some_and(|components| components.binary_search(&entry.component).is_err()) {
            return Err(PsiMeterCheckpointError::invalid_state(
                &format!("{field}.{}", entry.component.as_str()),
                "must belong to the active component set",
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DesireRoundtableImpulseCheckpoint {
    pub multipliers: [f32; 3],
    pub drift: f32,
    pub timestamp: TrainerTimestampCheckpoint,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DesireRoundtablePendingSummaryCheckpoint {
    pub steps: u64,
    pub triggers: u64,
    pub sum_entropy: f32,
    pub sum_temperature: f32,
    pub sum_alpha: f32,
    pub sum_beta: f32,
    pub sum_gamma: f32,
    pub sum_lambda: f32,
    pub sum_above: f32,
    pub sum_here: f32,
    pub sum_beneath: f32,
    pub sum_drift: f32,
    pub last_timestamp: TrainerTimestampCheckpoint,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DesireRoundtableCheckpoint {
    pub blend: f32,
    pub drift_gain: f32,
    pub latest: Option<DesireRoundtableImpulseCheckpoint>,
    pub pending_summary: Option<DesireRoundtablePendingSummaryCheckpoint>,
}

impl DesireRoundtableCheckpoint {
    pub fn validate(&self) -> Result<(), TrainerExternalCheckpointError> {
        validate_range("desire_roundtable.blend", self.blend, 0.0, 1.0)?;
        validate_range("desire_roundtable.drift_gain", self.drift_gain, 0.0, 1.2)?;
        if let Some(latest) = self.latest {
            for (index, value) in latest.multipliers.into_iter().enumerate() {
                validate_range(
                    &format!("desire_roundtable.latest.multipliers[{index}]"),
                    value,
                    0.35,
                    1.65,
                )?;
            }
            validate_range("desire_roundtable.latest.drift", latest.drift, -1.0, 1.0)?;
            latest
                .timestamp
                .validate("desire_roundtable.latest.timestamp")?;
        }
        if let Some(pending) = self.pending_summary {
            if pending.steps == 0 {
                return Err(state_error(
                    "desire_roundtable.pending_summary.steps",
                    "must be positive when a pending summary is present",
                ));
            }
            if pending.steps > TRAINER_EXTERNAL_MAX_SAFE_INTEGER {
                return Err(state_error(
                    "desire_roundtable.pending_summary.steps",
                    "must not exceed JavaScript's exact integer limit",
                ));
            }
            if pending.triggers > pending.steps {
                return Err(state_error(
                    "desire_roundtable.pending_summary.triggers",
                    "must not exceed steps",
                ));
            }
            pending
                .last_timestamp
                .validate("desire_roundtable.pending_summary.last_timestamp")?;
            for (label, value) in [
                ("sum_entropy", pending.sum_entropy),
                ("sum_temperature", pending.sum_temperature),
                ("sum_alpha", pending.sum_alpha),
                ("sum_beta", pending.sum_beta),
                ("sum_gamma", pending.sum_gamma),
                ("sum_lambda", pending.sum_lambda),
                ("sum_above", pending.sum_above),
                ("sum_here", pending.sum_here),
                ("sum_beneath", pending.sum_beneath),
            ] {
                validate_non_negative(
                    &format!("desire_roundtable.pending_summary.{label}"),
                    value,
                )?;
            }
            if !pending.sum_drift.is_finite() {
                return Err(state_error(
                    "desire_roundtable.pending_summary.sum_drift",
                    "must be finite",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainerCoherenceSignalCheckpoint {
    pub dominant_channel: Option<u64>,
    pub preserved_channels: u64,
    pub z_bias: f32,
    pub distribution_witness: ZSpaceCoherenceDistributionWitness,
    pub energy_ratio: f64,
    pub raw_mean_coherence: f64,
    pub classification_policy: ZSpaceCoherenceClassificationPolicy,
    pub repaired_non_finite_weights: u64,
    pub repaired_negative_weights: u64,
    pub pre_discard_repaired_non_finite: u64,
    pub pre_discard_repaired_negative: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainerCoherenceSignalProjection {
    pub control: ZSpaceCoherenceControlPayload,
    pub classification: ZSpaceCoherenceClassificationPayload,
}

impl TrainerCoherenceSignalCheckpoint {
    pub fn canonical_projection(
        &self,
    ) -> Result<TrainerCoherenceSignalProjection, TrainerExternalCheckpointError> {
        let preserved_channels = usize::try_from(self.preserved_channels).map_err(|_| {
            state_error(
                "coherence_bridge.signal.preserved_channels",
                "does not fit this platform",
            )
        })?;
        let dominant_channel = self
            .dominant_channel
            .map(|dominant| {
                usize::try_from(dominant).map_err(|_| {
                    state_error(
                        "coherence_bridge.signal.dominant_channel",
                        "does not fit this platform",
                    )
                })
            })
            .transpose()?;
        let distribution = validate_zspace_coherence_distribution_observation(
            &self.distribution_witness,
            preserved_channels,
            dominant_channel,
        )
        .map_err(|error| TrainerExternalCheckpointError::Coherence(error.to_string()))?;
        let channels = u64::try_from(distribution.channels).map_err(|_| {
            state_error(
                "coherence_bridge.signal.distribution_witness.normalized_weights",
                "channel count does not fit the portable checkpoint domain",
            )
        })?;
        if !self.z_bias.is_finite() {
            return Err(state_error(
                "coherence_bridge.signal.z_bias",
                "must be finite",
            ));
        }
        for (field, value) in [
            ("preserved_channels", self.preserved_channels),
            (
                "repaired_non_finite_weights",
                self.repaired_non_finite_weights,
            ),
            ("repaired_negative_weights", self.repaired_negative_weights),
            (
                "pre_discard_repaired_non_finite",
                self.pre_discard_repaired_non_finite,
            ),
            (
                "pre_discard_repaired_negative",
                self.pre_discard_repaired_negative,
            ),
        ] {
            validate_safe_integer(&format!("coherence_bridge.signal.{field}"), value)?;
        }
        for (field, non_finite, negative) in [
            (
                "repairs_total",
                self.repaired_non_finite_weights,
                self.repaired_negative_weights,
            ),
            (
                "pre_discard_repairs_total",
                self.pre_discard_repaired_non_finite,
                self.pre_discard_repaired_negative,
            ),
        ] {
            let repairs = non_finite.checked_add(negative).ok_or_else(|| {
                state_error(
                    &format!("coherence_bridge.signal.{field}"),
                    "overflows the portable checkpoint domain",
                )
            })?;
            if repairs > channels {
                return Err(state_error(
                    &format!("coherence_bridge.signal.{field}"),
                    "must not exceed the distribution channel count",
                ));
            }
        }

        let control = derive_zspace_coherence_control(
            distribution,
            self.energy_ratio,
            self.raw_mean_coherence,
        )
        .map_err(|error| TrainerExternalCheckpointError::Coherence(error.to_string()))?;
        let classification = classify_zspace_coherence(ZSpaceCoherenceClassificationRequest {
            energy_ratio: self.energy_ratio,
            swap_invariant: is_zspace_coherence_swap_invariant(
                &self.distribution_witness.normalized_weights,
            ),
            policy: self.classification_policy,
        })
        .map_err(|error| TrainerExternalCheckpointError::Coherence(error.to_string()))?;
        Ok(TrainerCoherenceSignalProjection {
            control,
            classification,
        })
    }

    pub fn validate(&self) -> Result<(), TrainerExternalCheckpointError> {
        self.canonical_projection().map(|_| ())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainerCoherenceBridgeCheckpoint {
    pub subscribed: bool,
    pub pending: Option<TrainerCoherenceSignalCheckpoint>,
    pub latest: Option<TrainerCoherenceSignalCheckpoint>,
}

impl TrainerCoherenceBridgeCheckpoint {
    pub fn validate(&self) -> Result<(), TrainerExternalCheckpointError> {
        if !self.subscribed && self.latest.is_some() {
            return Err(state_error(
                "coherence_bridge.latest",
                "requires an active ZSpaceTrace subscription",
            ));
        }
        if !self.subscribed && self.pending.is_none() {
            return Err(state_error(
                "coherence_bridge",
                "must contain a pending signal or an active subscription",
            ));
        }
        if let Some(signal) = &self.pending {
            signal.validate()?;
        }
        if let Some(signal) = &self.latest {
            signal.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainerGnnRoundtableSignalCheckpoint {
    pub observation: GnnRoundtableSignalObservation,
    pub issued_at: TrainerTimestampCheckpoint,
}

impl TrainerGnnRoundtableSignalCheckpoint {
    pub fn canonical_influence(
        &self,
    ) -> Result<GnnRoundtableInfluencePayload, TrainerExternalCheckpointError> {
        derive_gnn_roundtable_influence(self.observation)
            .map_err(|error| TrainerExternalCheckpointError::GnnRoundtable(error.to_string()))
    }

    fn validate(&self, field: &str) -> Result<(), TrainerExternalCheckpointError> {
        self.issued_at.validate(&format!("{field}.issued_at"))?;
        self.canonical_influence().map(|_| ())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainerGnnRoundtableBridgeCheckpoint {
    pub history_limit: u64,
    pub history: Vec<TrainerGnnRoundtableSignalCheckpoint>,
    pub latest: Option<TrainerGnnRoundtableSignalCheckpoint>,
    pub trainer_last: Option<TrainerGnnRoundtableSignalCheckpoint>,
}

impl TrainerGnnRoundtableBridgeCheckpoint {
    pub fn validate(&self) -> Result<(), TrainerExternalCheckpointError> {
        validate_safe_integer("gnn_roundtable_bridge.history_limit", self.history_limit)?;
        if self.history_limit == 0
            || self.history_limit
                > u64::try_from(GNN_ROUNDTABLE_MAX_HISTORY_SIGNALS).unwrap_or(u64::MAX)
        {
            return Err(state_error(
                "gnn_roundtable_bridge.history_limit",
                "must be within the canonical bridge history range",
            ));
        }
        let history_limit = usize::try_from(self.history_limit).map_err(|_| {
            state_error(
                "gnn_roundtable_bridge.history_limit",
                "does not fit this platform",
            )
        })?;
        if self.history.len() > history_limit {
            return Err(state_error(
                "gnn_roundtable_bridge.history",
                "must not exceed history_limit",
            ));
        }
        for (index, signal) in self.history.iter().enumerate() {
            signal.validate(&format!("gnn_roundtable_bridge.history[{index}]"))?;
        }
        if let Some(signal) = &self.latest {
            signal.validate("gnn_roundtable_bridge.latest")?;
        }
        if let Some(signal) = &self.trainer_last {
            signal.validate("gnn_roundtable_bridge.trainer_last")?;
        }
        if let Some(last_history) = self.history.last() {
            if self.latest.as_ref() != Some(last_history) {
                return Err(state_error(
                    "gnn_roundtable_bridge.latest",
                    "must equal the newest retained history signal",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainerExternalStateCheckpoint {
    pub kind: String,
    pub contract_version: String,
    pub semantic_owner: String,
    pub semantic_backend: String,
    pub required_components: Vec<String>,
    pub desire_trainer: Option<DesireTrainerQueueCheckpoint>,
    pub desire_roundtable: Option<DesireRoundtableCheckpoint>,
    pub coherence_bridge: Option<TrainerCoherenceBridgeCheckpoint>,
    pub gnn_roundtable_bridge: Option<TrainerGnnRoundtableBridgeCheckpoint>,
    pub psi_meter: Option<PsiMeterCheckpoint>,
    pub accumulator_synchronizer: Option<AccumulatorSynchronizerCheckpoint>,
    pub unresolved_components: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TrainerExternalCheckpointValidation {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub required_components: Vec<String>,
    pub captured_components: Vec<String>,
    pub unresolved_components: Vec<String>,
    pub reattach_required_components: Vec<String>,
    pub reattached_components: Vec<String>,
    pub payload_complete: bool,
    pub deterministic_resume_ready: bool,
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum TrainerExternalCheckpointError {
    #[error("external checkpoint {field} must be {expected}, got {actual}")]
    InvalidMetadata {
        field: &'static str,
        expected: &'static str,
        actual: String,
    },
    #[error("invalid external checkpoint state at {field}: {message}")]
    InvalidState { field: String, message: String },
    #[error(transparent)]
    Psi(#[from] PsiMeterCheckpointError),
    #[error("invalid coherence bridge checkpoint: {0}")]
    Coherence(String),
    #[error("invalid GNN roundtable bridge checkpoint: {0}")]
    GnnRoundtable(String),
    #[error("invalid accumulator synchronizer checkpoint: {0}")]
    Accumulator(String),
}

#[derive(Clone, Debug, Default)]
pub struct TrainerExternalCheckpointComponents {
    desire_trainer: Option<DesireTrainerQueueCheckpoint>,
    desire_roundtable: Option<DesireRoundtableCheckpoint>,
    coherence_bridge: Option<TrainerCoherenceBridgeCheckpoint>,
    gnn_roundtable_bridge: Option<TrainerGnnRoundtableBridgeCheckpoint>,
    psi_meter: Option<PsiMeterCheckpoint>,
    accumulator_synchronizer: Option<AccumulatorSynchronizerCheckpoint>,
}

impl TrainerExternalCheckpointComponents {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_desire_trainer(mut self, state: Option<DesireTrainerQueueCheckpoint>) -> Self {
        self.desire_trainer = state;
        self
    }

    pub fn with_desire_roundtable(mut self, state: Option<DesireRoundtableCheckpoint>) -> Self {
        self.desire_roundtable = state;
        self
    }

    pub fn with_coherence_bridge(
        mut self,
        state: Option<TrainerCoherenceBridgeCheckpoint>,
    ) -> Self {
        self.coherence_bridge = state;
        self
    }

    pub fn with_gnn_roundtable_bridge(
        mut self,
        state: Option<TrainerGnnRoundtableBridgeCheckpoint>,
    ) -> Self {
        self.gnn_roundtable_bridge = state;
        self
    }

    pub fn with_psi_meter(mut self, state: Option<PsiMeterCheckpoint>) -> Self {
        self.psi_meter = state;
        self
    }

    pub fn with_accumulator_synchronizer(
        mut self,
        state: Option<AccumulatorSynchronizerCheckpoint>,
    ) -> Self {
        self.accumulator_synchronizer = state;
        self
    }
}

pub fn build_trainer_external_state_checkpoint(
    required_components: Vec<String>,
    desire_roundtable: Option<DesireRoundtableCheckpoint>,
    psi_meter: Option<PsiMeterCheckpoint>,
    accumulator_synchronizer: Option<AccumulatorSynchronizerCheckpoint>,
) -> Result<TrainerExternalStateCheckpoint, TrainerExternalCheckpointError> {
    build_trainer_external_state_checkpoint_from_components(
        required_components,
        TrainerExternalCheckpointComponents::new()
            .with_desire_roundtable(desire_roundtable)
            .with_psi_meter(psi_meter)
            .with_accumulator_synchronizer(accumulator_synchronizer),
    )
}

pub fn build_trainer_external_state_checkpoint_with_desire_trainer(
    required_components: Vec<String>,
    desire_trainer: Option<DesireTrainerQueueCheckpoint>,
    desire_roundtable: Option<DesireRoundtableCheckpoint>,
    psi_meter: Option<PsiMeterCheckpoint>,
    accumulator_synchronizer: Option<AccumulatorSynchronizerCheckpoint>,
) -> Result<TrainerExternalStateCheckpoint, TrainerExternalCheckpointError> {
    build_trainer_external_state_checkpoint_from_components(
        required_components,
        TrainerExternalCheckpointComponents::new()
            .with_desire_trainer(desire_trainer)
            .with_desire_roundtable(desire_roundtable)
            .with_psi_meter(psi_meter)
            .with_accumulator_synchronizer(accumulator_synchronizer),
    )
}

pub fn build_trainer_external_state_checkpoint_from_components(
    mut required_components: Vec<String>,
    components: TrainerExternalCheckpointComponents,
) -> Result<TrainerExternalStateCheckpoint, TrainerExternalCheckpointError> {
    let TrainerExternalCheckpointComponents {
        desire_trainer,
        desire_roundtable,
        coherence_bridge,
        gnn_roundtable_bridge,
        psi_meter,
        accumulator_synchronizer,
    } = components;
    required_components.sort_unstable();
    required_components.dedup();
    let captured = captured_components(
        desire_trainer.as_ref(),
        desire_roundtable.as_ref(),
        coherence_bridge.as_ref(),
        gnn_roundtable_bridge.as_ref(),
        psi_meter.as_ref(),
        accumulator_synchronizer.as_ref(),
    );
    let unresolved_components = required_components
        .iter()
        .filter(|component| captured.binary_search(component).is_err())
        .cloned()
        .collect();
    let checkpoint = TrainerExternalStateCheckpoint {
        kind: TRAINER_EXTERNAL_CHECKPOINT_KIND.to_owned(),
        contract_version: TRAINER_EXTERNAL_CHECKPOINT_CONTRACT_VERSION.to_owned(),
        semantic_owner: TRAINER_EXTERNAL_CHECKPOINT_SEMANTIC_OWNER.to_owned(),
        semantic_backend: TRAINER_EXTERNAL_CHECKPOINT_SEMANTIC_BACKEND.to_owned(),
        required_components,
        desire_trainer,
        desire_roundtable,
        coherence_bridge,
        gnn_roundtable_bridge,
        psi_meter,
        accumulator_synchronizer,
        unresolved_components,
    };
    evaluate_trainer_external_state_checkpoint(&checkpoint)?;
    Ok(checkpoint)
}

pub fn evaluate_trainer_external_state_checkpoint(
    checkpoint: &TrainerExternalStateCheckpoint,
) -> Result<TrainerExternalCheckpointValidation, TrainerExternalCheckpointError> {
    evaluate_trainer_external_state_checkpoint_with_reattached(checkpoint, &[])
}

/// Evaluates a native restore after concrete resources have been reattached.
pub fn evaluate_trainer_external_state_restore(
    checkpoint: &TrainerExternalStateCheckpoint,
    reattached_components: &[String],
) -> Result<TrainerExternalCheckpointValidation, TrainerExternalCheckpointError> {
    evaluate_trainer_external_state_checkpoint_with_reattached(checkpoint, reattached_components)
}

fn evaluate_trainer_external_state_checkpoint_with_reattached(
    checkpoint: &TrainerExternalStateCheckpoint,
    reattached_components: &[String],
) -> Result<TrainerExternalCheckpointValidation, TrainerExternalCheckpointError> {
    validate_metadata("kind", &checkpoint.kind, TRAINER_EXTERNAL_CHECKPOINT_KIND)?;
    validate_metadata(
        "contract_version",
        &checkpoint.contract_version,
        TRAINER_EXTERNAL_CHECKPOINT_CONTRACT_VERSION,
    )?;
    validate_metadata(
        "semantic_owner",
        &checkpoint.semantic_owner,
        TRAINER_EXTERNAL_CHECKPOINT_SEMANTIC_OWNER,
    )?;
    validate_metadata(
        "semantic_backend",
        &checkpoint.semantic_backend,
        TRAINER_EXTERNAL_CHECKPOINT_SEMANTIC_BACKEND,
    )?;
    validate_sorted_unique("required_components", &checkpoint.required_components)?;
    validate_sorted_unique("unresolved_components", &checkpoint.unresolved_components)?;
    if let Some(state) = &checkpoint.desire_trainer {
        state.validate()?;
    }
    if let Some(state) = checkpoint.desire_roundtable {
        state.validate()?;
    }
    if let Some(state) = &checkpoint.coherence_bridge {
        state.validate()?;
    }
    if let Some(state) = &checkpoint.gnn_roundtable_bridge {
        state.validate()?;
    }
    if let Some(state) = &checkpoint.psi_meter {
        state.validate()?;
    }
    if let Some(state) = &checkpoint.accumulator_synchronizer {
        state
            .validate()
            .map_err(|error| TrainerExternalCheckpointError::Accumulator(error.to_string()))?;
    }

    let captured = captured_components(
        checkpoint.desire_trainer.as_ref(),
        checkpoint.desire_roundtable.as_ref(),
        checkpoint.coherence_bridge.as_ref(),
        checkpoint.gnn_roundtable_bridge.as_ref(),
        checkpoint.psi_meter.as_ref(),
        checkpoint.accumulator_synchronizer.as_ref(),
    );
    for component in &captured {
        if checkpoint
            .required_components
            .binary_search(component)
            .is_err()
        {
            return Err(state_error(
                "required_components",
                &format!("captured component {component} is not required"),
            ));
        }
    }
    let expected_unresolved = checkpoint
        .required_components
        .iter()
        .filter(|component| captured.binary_search(component).is_err())
        .cloned()
        .collect::<Vec<_>>();
    if checkpoint.unresolved_components != expected_unresolved {
        return Err(state_error(
            "unresolved_components",
            "must exactly list every required component not captured by this contract",
        ));
    }

    let mut reattach_required_components = Vec::new();
    if checkpoint.accumulator_synchronizer.is_some() {
        reattach_required_components.push(ACCUMULATOR_SYNCHRONIZER_COMPONENT.to_owned());
    }
    validate_sorted_unique("reattached_components", reattached_components)?;
    if reattached_components.iter().any(|component| {
        reattach_required_components
            .binary_search(component)
            .is_err()
    }) {
        return Err(state_error(
            "reattached_components",
            "must only contain resources required by this checkpoint",
        ));
    }
    let payload_complete = checkpoint.unresolved_components.is_empty();
    let deterministic_resume_ready =
        payload_complete && reattached_components == reattach_required_components;
    Ok(TrainerExternalCheckpointValidation {
        kind: TRAINER_EXTERNAL_CHECKPOINT_KIND,
        contract_version: TRAINER_EXTERNAL_CHECKPOINT_CONTRACT_VERSION,
        semantic_owner: TRAINER_EXTERNAL_CHECKPOINT_SEMANTIC_OWNER,
        semantic_backend: TRAINER_EXTERNAL_CHECKPOINT_SEMANTIC_BACKEND,
        required_components: checkpoint.required_components.clone(),
        captured_components: captured,
        unresolved_components: checkpoint.unresolved_components.clone(),
        reattach_required_components,
        reattached_components: reattached_components.to_vec(),
        payload_complete,
        deterministic_resume_ready,
    })
}

fn captured_components(
    desire_trainer: Option<&DesireTrainerQueueCheckpoint>,
    desire_roundtable: Option<&DesireRoundtableCheckpoint>,
    coherence_bridge: Option<&TrainerCoherenceBridgeCheckpoint>,
    gnn_roundtable_bridge: Option<&TrainerGnnRoundtableBridgeCheckpoint>,
    psi_meter: Option<&PsiMeterCheckpoint>,
    accumulator_synchronizer: Option<&AccumulatorSynchronizerCheckpoint>,
) -> Vec<String> {
    let mut captured = Vec::with_capacity(6);
    if desire_trainer.is_some() {
        captured.push(DESIRE_TRAINER_COMPONENT.to_owned());
    }
    if desire_roundtable.is_some() {
        captured.push(DESIRE_ROUNDTABLE_COMPONENT.to_owned());
    }
    if coherence_bridge.is_some() {
        captured.push(COHERENCE_BRIDGE_COMPONENT.to_owned());
    }
    if gnn_roundtable_bridge.is_some() {
        captured.push(GNN_ROUNDTABLE_BRIDGE_COMPONENT.to_owned());
    }
    if psi_meter.is_some() {
        captured.push(PSI_METER_COMPONENT.to_owned());
    }
    if accumulator_synchronizer.is_some_and(|checkpoint| !checkpoint.requires_opaque_reattach()) {
        captured.push(ACCUMULATOR_SYNCHRONIZER_COMPONENT.to_owned());
    }
    captured.sort_unstable();
    captured
}

fn validate_metadata(
    field: &'static str,
    actual: &str,
    expected: &'static str,
) -> Result<(), TrainerExternalCheckpointError> {
    if actual == expected {
        Ok(())
    } else {
        Err(TrainerExternalCheckpointError::InvalidMetadata {
            field,
            expected,
            actual: actual.to_owned(),
        })
    }
}

fn validate_sorted_unique(
    field: &str,
    values: &[String],
) -> Result<(), TrainerExternalCheckpointError> {
    let mut previous: Option<&str> = None;
    for value in values {
        if value.trim().is_empty() {
            return Err(state_error(field, "component labels must not be empty"));
        }
        if previous.is_some_and(|previous| previous >= value.as_str()) {
            return Err(state_error(
                field,
                "component labels must be sorted and unique",
            ));
        }
        previous = Some(value);
    }
    Ok(())
}

fn validate_safe_integer(field: &str, value: u64) -> Result<(), TrainerExternalCheckpointError> {
    if value <= TRAINER_EXTERNAL_MAX_SAFE_INTEGER {
        Ok(())
    } else {
        Err(state_error(
            field,
            "must not exceed JavaScript's exact integer limit",
        ))
    }
}

fn validate_range(
    field: &str,
    value: f32,
    min: f32,
    max: f32,
) -> Result<(), TrainerExternalCheckpointError> {
    if value.is_finite() && (min..=max).contains(&value) {
        Ok(())
    } else {
        Err(state_error(
            field,
            &format!("must be finite and in [{min}, {max}]"),
        ))
    }
}

fn validate_non_negative(field: &str, value: f32) -> Result<(), TrainerExternalCheckpointError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(state_error(field, "must be finite and non-negative"))
    }
}

fn validate_positive(field: &str, value: f32) -> Result<(), TrainerExternalCheckpointError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(state_error(field, "must be finite and positive"))
    }
}

fn state_error(field: &str, message: &str) -> TrainerExternalCheckpointError {
    TrainerExternalCheckpointError::InvalidState {
        field: field.to_owned(),
        message: message.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::{
        ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_CONTRACT_VERSION,
        ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_KIND,
        ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_SEMANTIC_BACKEND,
        ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_SEMANTIC_OWNER,
        ACCUMULATOR_SYNCHRONIZER_OPAQUE_PROVIDER,
    };

    fn psi_checkpoint() -> PsiMeterCheckpoint {
        PsiMeterCheckpoint {
            kind: PSI_METER_CHECKPOINT_KIND.to_owned(),
            contract_version: PSI_METER_CHECKPOINT_CONTRACT_VERSION.to_owned(),
            semantic_owner: PSI_METER_CHECKPOINT_SEMANTIC_OWNER.to_owned(),
            semantic_backend: PSI_METER_CHECKPOINT_SEMANTIC_BACKEND.to_owned(),
            enabled: true,
            components: vec![PsiMeterComponent::Loss],
            weights: vec![PsiMeterComponentValue {
                component: PsiMeterComponent::Loss,
                value: 1.0,
            }],
            ema_alpha: 0.2,
            sample_rate: 1,
            thresholds: Vec::new(),
            ema: Vec::new(),
            step: 0,
        }
    }

    fn accumulator_checkpoint() -> AccumulatorSynchronizerCheckpoint {
        AccumulatorSynchronizerCheckpoint {
            kind: ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_KIND.to_owned(),
            contract_version: ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_CONTRACT_VERSION.to_owned(),
            semantic_owner: ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_SEMANTIC_OWNER.to_owned(),
            semantic_backend: ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_SEMANTIC_BACKEND.to_owned(),
            provider: ACCUMULATOR_SYNCHRONIZER_OPAQUE_PROVIDER.to_owned(),
            rank: 0,
            world_size: 1,
            state: None,
        }
    }

    fn desire_trainer_checkpoint() -> DesireTrainerQueueCheckpoint {
        DesireTrainerQueueCheckpoint {
            events: vec![DesireTrainerEventCheckpoint {
                timestamp: TrainerTimestampCheckpoint {
                    unix_seconds: 17,
                    subsec_nanos: 23,
                },
                phase: DesireTrainerPhaseCheckpoint::Injection,
                temperature: 0.8,
                entropy: 1.2,
                hypergrad_penalty: 0.3,
                weights: DesireTrainerWeightsCheckpoint {
                    alpha: 0.7,
                    beta: 0.2,
                    gamma: 0.5,
                    lambda: 0.1,
                },
                trigger: Some(DesireTrainerTriggerCheckpoint {
                    report_tokens: vec![7, 11],
                    report_scores: vec![0.8, 0.4],
                    mean_penalty: 0.3,
                    mean_entropy: 1.2,
                    temperature: 0.8,
                    samples: 12,
                }),
            }],
        }
    }

    fn coherence_signal_checkpoint() -> TrainerCoherenceSignalCheckpoint {
        TrainerCoherenceSignalCheckpoint {
            dominant_channel: Some(0),
            preserved_channels: 3,
            z_bias: 0.2,
            distribution_witness:
                crate::inference::zspace_coherence::build_zspace_coherence_distribution_witness(&[
                    0.6_f64, 0.3, 0.1,
                ])
                .unwrap(),
            energy_ratio: 0.75,
            raw_mean_coherence: 0.4,
            classification_policy: ZSpaceCoherenceClassificationPolicy::default(),
            repaired_non_finite_weights: 0,
            repaired_negative_weights: 0,
            pre_discard_repaired_non_finite: 0,
            pre_discard_repaired_negative: 0,
        }
    }

    fn gnn_roundtable_signal_checkpoint(
        unix_seconds: u64,
        subsec_nanos: u32,
    ) -> TrainerGnnRoundtableSignalCheckpoint {
        TrainerGnnRoundtableSignalCheckpoint {
            observation: GnnRoundtableSignalObservation {
                band_energy: crate::inference::gnn_roundtable::GnnRoundtableBandEnergyObservation {
                    above: 1.4,
                    here: 0.45,
                    beneath: 0.2,
                    drift: 0.35,
                },
                band_sizes: crate::inference::gnn_roundtable::GnnRoundtableBandSizes {
                    above: 4,
                    here: 2,
                    beneath: 2,
                },
                spectral: crate::inference::gnn_roundtable::GnnRoundtableSpectralObservation {
                    sheet_index: 1,
                    sheet_confidence: 0.9,
                    curvature: 0.6,
                    spin: 0.85,
                    energy: 0.72,
                },
            },
            issued_at: TrainerTimestampCheckpoint {
                unix_seconds,
                subsec_nanos,
            },
        }
    }

    #[test]
    fn external_checkpoint_accounts_for_captured_and_unresolved_components() {
        let checkpoint = build_trainer_external_state_checkpoint_with_desire_trainer(
            vec![
                PSI_METER_COMPONENT.to_owned(),
                "blackcat_runtime".to_owned(),
                DESIRE_TRAINER_COMPONENT.to_owned(),
                DESIRE_ROUNDTABLE_COMPONENT.to_owned(),
                ACCUMULATOR_SYNCHRONIZER_COMPONENT.to_owned(),
            ],
            Some(desire_trainer_checkpoint()),
            Some(DesireRoundtableCheckpoint {
                blend: 0.4,
                drift_gain: 0.3,
                latest: Some(DesireRoundtableImpulseCheckpoint {
                    multipliers: [1.1, 0.9, 1.0],
                    drift: 0.2,
                    timestamp: TrainerTimestampCheckpoint {
                        unix_seconds: 7,
                        subsec_nanos: 0,
                    },
                }),
                pending_summary: None,
            }),
            Some(psi_checkpoint()),
            Some(accumulator_checkpoint()),
        )
        .unwrap();
        let report = evaluate_trainer_external_state_checkpoint(&checkpoint).unwrap();

        assert_eq!(
            report.unresolved_components,
            [ACCUMULATOR_SYNCHRONIZER_COMPONENT, "blackcat_runtime"]
        );
        assert_eq!(
            report.reattach_required_components,
            [ACCUMULATOR_SYNCHRONIZER_COMPONENT]
        );
        assert!(!report.payload_complete);
        assert!(!report.deterministic_resume_ready);
    }

    #[test]
    fn external_checkpoint_rejects_dishonest_component_accounting() {
        let mut checkpoint = build_trainer_external_state_checkpoint(
            vec![PSI_METER_COMPONENT.to_owned(), "autopilot".to_owned()],
            None,
            Some(psi_checkpoint()),
            None,
        )
        .unwrap();
        checkpoint.unresolved_components.clear();
        assert!(matches!(
            evaluate_trainer_external_state_checkpoint(&checkpoint),
            Err(TrainerExternalCheckpointError::InvalidState { .. })
        ));
    }

    #[test]
    fn external_checkpoint_requires_verified_native_resource_reattachment() {
        let mut accumulator = accumulator_checkpoint();
        accumulator.provider = "st-core.tests.accumulator.v1".to_owned();
        accumulator.state = Some(serde_json::json!({ "group": "alpha" }));
        let checkpoint = build_trainer_external_state_checkpoint(
            vec![ACCUMULATOR_SYNCHRONIZER_COMPONENT.to_owned()],
            None,
            None,
            Some(accumulator),
        )
        .unwrap();

        let preflight = evaluate_trainer_external_state_checkpoint(&checkpoint).unwrap();
        assert!(preflight.payload_complete);
        assert!(!preflight.deterministic_resume_ready);
        assert_eq!(
            preflight.reattach_required_components,
            [ACCUMULATOR_SYNCHRONIZER_COMPONENT]
        );
        assert!(preflight.reattached_components.is_empty());

        let restored = evaluate_trainer_external_state_restore(
            &checkpoint,
            &[ACCUMULATOR_SYNCHRONIZER_COMPONENT.to_owned()],
        )
        .unwrap();
        assert!(restored.deterministic_resume_ready);
        assert_eq!(
            restored.reattached_components,
            [ACCUMULATOR_SYNCHRONIZER_COMPONENT]
        );
    }

    #[test]
    fn external_checkpoint_serde_rejects_unknown_fields() {
        let checkpoint = build_trainer_external_state_checkpoint(
            vec![PSI_METER_COMPONENT.to_owned()],
            None,
            Some(psi_checkpoint()),
            None,
        )
        .unwrap();
        let mut payload = serde_json::to_value(checkpoint).unwrap();
        payload
            .as_object_mut()
            .unwrap()
            .insert("commander".to_owned(), serde_json::json!("wasm"));
        let error = serde_json::from_value::<TrainerExternalStateCheckpoint>(payload).unwrap_err();
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn external_checkpoint_rejects_cross_runtime_integer_overflow() {
        let mut psi = psi_checkpoint();
        psi.step = TRAINER_EXTERNAL_MAX_SAFE_INTEGER + 1;
        assert!(matches!(
            build_trainer_external_state_checkpoint(
                vec![PSI_METER_COMPONENT.to_owned()],
                None,
                Some(psi),
                None,
            ),
            Err(TrainerExternalCheckpointError::Psi(
                PsiMeterCheckpointError::InvalidState { .. }
            ))
        ));

        let result = build_trainer_external_state_checkpoint(
            vec![DESIRE_ROUNDTABLE_COMPONENT.to_owned()],
            Some(DesireRoundtableCheckpoint {
                blend: 0.4,
                drift_gain: 0.3,
                latest: Some(DesireRoundtableImpulseCheckpoint {
                    multipliers: [1.0; 3],
                    drift: 0.0,
                    timestamp: TrainerTimestampCheckpoint {
                        unix_seconds: TRAINER_EXTERNAL_MAX_SAFE_INTEGER + 1,
                        subsec_nanos: 0,
                    },
                }),
                pending_summary: None,
            }),
            None,
            None,
        );
        assert!(matches!(
            result,
            Err(TrainerExternalCheckpointError::InvalidState { .. })
        ));
    }

    #[test]
    fn desire_trainer_queue_is_captured_in_fifo_order() {
        let mut queue = desire_trainer_checkpoint();
        let mut second = queue.events[0].clone();
        second.timestamp = TrainerTimestampCheckpoint {
            unix_seconds: 23,
            subsec_nanos: 42,
        };
        second.phase = DesireTrainerPhaseCheckpoint::Integration;
        second.trigger = None;
        queue.events.push(second);

        let checkpoint = build_trainer_external_state_checkpoint_with_desire_trainer(
            vec![DESIRE_TRAINER_COMPONENT.to_owned()],
            Some(queue.clone()),
            None,
            None,
            None,
        )
        .unwrap();
        let encoded = serde_json::to_string(&checkpoint).unwrap();
        let decoded: TrainerExternalStateCheckpoint = serde_json::from_str(&encoded).unwrap();
        let report = evaluate_trainer_external_state_checkpoint(&decoded).unwrap();

        assert_eq!(decoded.desire_trainer, Some(queue));
        assert_eq!(report.captured_components, [DESIRE_TRAINER_COMPONENT]);
        assert!(report.payload_complete);
        assert!(report.deterministic_resume_ready);
    }

    #[test]
    fn desire_trainer_queue_rejects_malformed_trigger_without_reconstruction() {
        let mut queue = desire_trainer_checkpoint();
        queue.events[0]
            .trigger
            .as_mut()
            .unwrap()
            .report_scores
            .pop();

        let error = build_trainer_external_state_checkpoint_with_desire_trainer(
            vec![DESIRE_TRAINER_COMPONENT.to_owned()],
            Some(queue),
            None,
            None,
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("same length"));
    }

    #[test]
    fn desire_trainer_queue_rejects_non_finite_and_unsafe_values() {
        let mut non_finite = desire_trainer_checkpoint();
        non_finite.events[0].weights.gamma = f32::NAN;
        assert!(build_trainer_external_state_checkpoint_with_desire_trainer(
            vec![DESIRE_TRAINER_COMPONENT.to_owned()],
            Some(non_finite),
            None,
            None,
            None,
        )
        .is_err());

        let mut unsafe_integer = desire_trainer_checkpoint();
        unsafe_integer.events[0]
            .trigger
            .as_mut()
            .unwrap()
            .report_tokens[0] = TRAINER_EXTERNAL_MAX_SAFE_INTEGER + 1;
        assert!(build_trainer_external_state_checkpoint_with_desire_trainer(
            vec![DESIRE_TRAINER_COMPONENT.to_owned()],
            Some(unsafe_integer),
            None,
            None,
            None,
        )
        .is_err());

        let mut invalid_timestamp = desire_trainer_checkpoint();
        invalid_timestamp.events[0].timestamp.subsec_nanos = 1_000_000_000;
        assert!(build_trainer_external_state_checkpoint_with_desire_trainer(
            vec![DESIRE_TRAINER_COMPONENT.to_owned()],
            Some(invalid_timestamp),
            None,
            None,
            None,
        )
        .is_err());
    }

    #[test]
    fn coherence_signal_reconstructs_only_through_canonical_rust_contracts() {
        let signal = coherence_signal_checkpoint();
        let projection = signal.canonical_projection().unwrap();

        assert_eq!(projection.control.channels, 3);
        assert_eq!(projection.control.raw_mean_coherence, 0.4);
        assert_eq!(projection.control.energy_ratio, 0.75);
        assert_eq!(
            projection.classification.label.as_str(),
            "cascade_imbalance"
        );
        assert!(!projection.classification.swap_invariant);

        let checkpoint = build_trainer_external_state_checkpoint_from_components(
            vec![COHERENCE_BRIDGE_COMPONENT.to_owned()],
            TrainerExternalCheckpointComponents::new().with_coherence_bridge(Some(
                TrainerCoherenceBridgeCheckpoint {
                    subscribed: true,
                    pending: Some(signal),
                    latest: Some(coherence_signal_checkpoint()),
                },
            )),
        )
        .unwrap();
        let report = evaluate_trainer_external_state_checkpoint(&checkpoint).unwrap();
        assert_eq!(report.captured_components, [COHERENCE_BRIDGE_COMPONENT]);
        assert!(report.deterministic_resume_ready);
    }

    #[test]
    fn coherence_checkpoint_rejects_tampered_evidence_and_topology() {
        let mut invalid_simplex = coherence_signal_checkpoint();
        invalid_simplex.distribution_witness.normalized_weights[0] = 0.9;
        assert!(invalid_simplex.validate().is_err());

        let mut invalid_dominant = coherence_signal_checkpoint();
        invalid_dominant.dominant_channel = Some(3);
        assert!(invalid_dominant.validate().is_err());

        let mut inconsistent_dominant = coherence_signal_checkpoint();
        inconsistent_dominant.dominant_channel = Some(1);
        assert!(inconsistent_dominant.validate().is_err());

        let mut inconsistent_support = coherence_signal_checkpoint();
        inconsistent_support.preserved_channels = 2;
        assert!(inconsistent_support.validate().is_err());

        let invalid_topology = TrainerCoherenceBridgeCheckpoint {
            subscribed: false,
            pending: None,
            latest: Some(coherence_signal_checkpoint()),
        };
        assert!(invalid_topology.validate().is_err());

        let mut unsafe_counter = coherence_signal_checkpoint();
        unsafe_counter.repaired_negative_weights = TRAINER_EXTERNAL_MAX_SAFE_INTEGER + 1;
        assert!(unsafe_counter.validate().is_err());

        let mut impossible_repairs = coherence_signal_checkpoint();
        impossible_repairs.repaired_non_finite_weights = 2;
        impossible_repairs.repaired_negative_weights = 2;
        assert!(impossible_repairs.validate().is_err());
    }

    #[test]
    fn timestamp_checkpoint_roundtrips_nanoseconds_through_the_shared_contract() {
        let duration = Duration::new(17, 42);
        let checkpoint =
            TrainerTimestampCheckpoint::try_from_unix_duration("timestamp", duration).unwrap();
        assert_eq!(checkpoint.unix_seconds, 17);
        assert_eq!(checkpoint.subsec_nanos, 42);
        assert_eq!(
            checkpoint.try_to_unix_duration("timestamp").unwrap(),
            duration
        );

        // SystemTime is only an OS boundary view and may have coarser resolution.
        let platform_timestamp = UNIX_EPOCH + duration;
        let platform_checkpoint = TrainerTimestampCheckpoint::try_from_system_time(
            "platform_timestamp",
            platform_timestamp,
        )
        .unwrap();
        assert_eq!(
            platform_checkpoint
                .try_to_system_time("platform_timestamp")
                .unwrap(),
            platform_timestamp
        );
        assert!(TrainerTimestampCheckpoint::try_from_system_time(
            "timestamp",
            UNIX_EPOCH.checked_sub(Duration::from_secs(1)).unwrap(),
        )
        .is_err());
    }

    #[test]
    fn gnn_roundtable_checkpoint_rederives_influence_and_preserves_fifo() {
        let first = gnn_roundtable_signal_checkpoint(17, 42);
        let second = gnn_roundtable_signal_checkpoint(18, 7);
        let bridge = TrainerGnnRoundtableBridgeCheckpoint {
            history_limit: 4,
            history: vec![first.clone(), second.clone()],
            latest: Some(second.clone()),
            trainer_last: Some(first),
        };
        let checkpoint = build_trainer_external_state_checkpoint_from_components(
            vec![GNN_ROUNDTABLE_BRIDGE_COMPONENT.to_owned()],
            TrainerExternalCheckpointComponents::new().with_gnn_roundtable_bridge(Some(bridge)),
        )
        .unwrap();
        let encoded = serde_json::to_string(&checkpoint).unwrap();
        let decoded: TrainerExternalStateCheckpoint = serde_json::from_str(&encoded).unwrap();
        let report = evaluate_trainer_external_state_checkpoint(&decoded).unwrap();
        let influence = decoded
            .gnn_roundtable_bridge
            .as_ref()
            .unwrap()
            .latest
            .as_ref()
            .unwrap()
            .canonical_influence()
            .unwrap();

        assert_eq!(checkpoint, decoded);
        assert_eq!(
            report.captured_components,
            [GNN_ROUNDTABLE_BRIDGE_COMPONENT]
        );
        assert!(report.deterministic_resume_ready);
        assert!(influence.multipliers[0] > influence.multipliers[2]);
    }

    #[test]
    fn gnn_roundtable_checkpoint_rejects_signal_and_history_tampering() {
        let first = gnn_roundtable_signal_checkpoint(17, 42);
        let second = gnn_roundtable_signal_checkpoint(18, 7);
        let valid = TrainerGnnRoundtableBridgeCheckpoint {
            history_limit: 2,
            history: vec![first.clone(), second.clone()],
            latest: Some(second),
            trainer_last: Some(first.clone()),
        };
        valid.validate().unwrap();

        let mut negative_energy = valid.clone();
        negative_energy.history[0].observation.band_energy.above = -0.1;
        assert!(matches!(
            negative_energy.validate(),
            Err(TrainerExternalCheckpointError::GnnRoundtable(_))
        ));

        let mut zero_band = valid.clone();
        zero_band.history[0].observation.band_sizes.here = 0;
        assert!(zero_band.validate().is_err());

        let mut invalid_timestamp = valid.clone();
        invalid_timestamp.history[0].issued_at.subsec_nanos = 1_000_000_000;
        assert!(invalid_timestamp.validate().is_err());

        let mut over_limit = valid.clone();
        over_limit.history_limit = 1;
        assert!(over_limit.validate().is_err());

        let mut inconsistent_latest = valid.clone();
        inconsistent_latest.latest = Some(first);
        assert!(inconsistent_latest.validate().is_err());

        let mut excessive_limit = valid;
        excessive_limit.history_limit =
            u64::try_from(GNN_ROUNDTABLE_MAX_HISTORY_SIGNALS).unwrap() + 1;
        assert!(excessive_limit.validate().is_err());
    }
}
