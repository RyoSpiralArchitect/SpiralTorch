//! Versioned state contract for trainer components outside the optimizer core.
//!
//! Rust owns validation and component accounting. Native orchestrators may
//! restore concrete resources; browser clients only preflight the same payload.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::distributed::AccumulatorSynchronizerCheckpoint;

pub const TRAINER_EXTERNAL_CHECKPOINT_KIND: &str = "spiraltorch.trainer_external_state_checkpoint";
pub const TRAINER_EXTERNAL_CHECKPOINT_CONTRACT_VERSION: &str =
    "spiraltorch.trainer_external_state_checkpoint.v1";
pub const TRAINER_EXTERNAL_CHECKPOINT_SEMANTIC_OWNER: &str = "st-core::runtime::trainer_external";
pub const TRAINER_EXTERNAL_CHECKPOINT_SEMANTIC_BACKEND: &str = "rust";
pub const TRAINER_EXTERNAL_MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

pub const DESIRE_ROUNDTABLE_COMPONENT: &str = "desire_roundtable_bridge";
pub const PSI_METER_COMPONENT: &str = "psi_meter";
pub const ACCUMULATOR_SYNCHRONIZER_COMPONENT: &str = "accumulator_synchronizer";

pub const PSI_METER_CHECKPOINT_KIND: &str = "spiraltorch.psi_meter_checkpoint";
pub const PSI_METER_CHECKPOINT_CONTRACT_VERSION: &str = "spiraltorch.psi_meter_checkpoint.v1";
pub const PSI_METER_CHECKPOINT_SEMANTIC_OWNER: &str = "st-core::telemetry::psi";
pub const PSI_METER_CHECKPOINT_SEMANTIC_BACKEND: &str = "rust";

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
    pub timestamp_unix_millis: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DesireRoundtableCheckpoint {
    pub blend: f32,
    pub drift_gain: f32,
    pub latest: Option<DesireRoundtableImpulseCheckpoint>,
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
            if latest.timestamp_unix_millis > TRAINER_EXTERNAL_MAX_SAFE_INTEGER {
                return Err(state_error(
                    "desire_roundtable.latest.timestamp_unix_millis",
                    "must not exceed JavaScript's exact integer limit",
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
    pub desire_roundtable: Option<DesireRoundtableCheckpoint>,
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
    #[error("invalid accumulator synchronizer checkpoint: {0}")]
    Accumulator(String),
}

pub fn build_trainer_external_state_checkpoint(
    mut required_components: Vec<String>,
    desire_roundtable: Option<DesireRoundtableCheckpoint>,
    psi_meter: Option<PsiMeterCheckpoint>,
    accumulator_synchronizer: Option<AccumulatorSynchronizerCheckpoint>,
) -> Result<TrainerExternalStateCheckpoint, TrainerExternalCheckpointError> {
    required_components.sort_unstable();
    required_components.dedup();
    let captured = captured_components(
        desire_roundtable.as_ref(),
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
        desire_roundtable,
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
    if let Some(state) = checkpoint.desire_roundtable {
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
        checkpoint.desire_roundtable.as_ref(),
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
    desire_roundtable: Option<&DesireRoundtableCheckpoint>,
    psi_meter: Option<&PsiMeterCheckpoint>,
    accumulator_synchronizer: Option<&AccumulatorSynchronizerCheckpoint>,
) -> Vec<String> {
    let mut captured = Vec::with_capacity(3);
    if desire_roundtable.is_some() {
        captured.push(DESIRE_ROUNDTABLE_COMPONENT.to_owned());
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

    #[test]
    fn external_checkpoint_accounts_for_captured_and_unresolved_components() {
        let checkpoint = build_trainer_external_state_checkpoint(
            vec![
                PSI_METER_COMPONENT.to_owned(),
                "blackcat_runtime".to_owned(),
                DESIRE_ROUNDTABLE_COMPONENT.to_owned(),
                ACCUMULATOR_SYNCHRONIZER_COMPONENT.to_owned(),
            ],
            Some(DesireRoundtableCheckpoint {
                blend: 0.4,
                drift_gain: 0.3,
                latest: Some(DesireRoundtableImpulseCheckpoint {
                    multipliers: [1.1, 0.9, 1.0],
                    drift: 0.2,
                    timestamp_unix_millis: 7,
                }),
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
                    timestamp_unix_millis: TRAINER_EXTERNAL_MAX_SAFE_INTEGER + 1,
                }),
            }),
            None,
            None,
        );
        assert!(matches!(
            result,
            Err(TrainerExternalCheckpointError::InvalidState { .. })
        ));
    }
}
