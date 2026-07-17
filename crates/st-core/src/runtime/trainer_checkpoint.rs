//! Integrity-bound checkpoint contract for a `ModuleTrainer` runtime.
//!
//! The optimizer and external-runtime payloads remain independently versioned,
//! but this envelope binds them into one Rust-owned restore decision. Parameter
//! values stay in the module state dictionary and concrete resources such as a
//! distributed accumulator synchronizer must be reattached before commit.

use super::trainer_external::{
    evaluate_trainer_external_state_checkpoint, evaluate_trainer_external_state_restore,
    TrainerExternalCheckpointError, TrainerExternalCheckpointValidation,
    TrainerExternalStateCheckpoint,
};
use super::trainer_optimizer::{
    evaluate_trainer_optimizer_checkpoint, TrainerOptimizerCheckpoint,
    TrainerOptimizerCheckpointError, TrainerOptimizerCheckpointValidation,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use thiserror::Error;

pub const TRAINER_RUNTIME_CHECKPOINT_BUNDLE_KIND: &str =
    "spiraltorch.trainer_runtime_checkpoint_bundle";
pub const TRAINER_RUNTIME_CHECKPOINT_BUNDLE_CONTRACT_VERSION: &str =
    "spiraltorch.trainer_runtime_checkpoint_bundle.v2";
pub const TRAINER_RUNTIME_CHECKPOINT_BUNDLE_SEMANTIC_OWNER: &str =
    "st-core::runtime::trainer_checkpoint";
pub const TRAINER_RUNTIME_CHECKPOINT_BUNDLE_SEMANTIC_BACKEND: &str = "rust";
pub const TRAINER_RUNTIME_CHECKPOINT_BUNDLE_RESUME_SCOPE: &str =
    "module_trainer_optimizer_and_supported_external_runtime_state;parameter_values_external_and_fingerprint_guarded;concrete_resources_reattached_before_commit";

/// Versioned envelope binding optimizer and external-runtime state together.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainerRuntimeCheckpointBundle {
    pub kind: String,
    pub contract_version: String,
    pub semantic_owner: String,
    pub semantic_backend: String,
    pub resume_scope: String,
    pub optimizer_sha256: String,
    pub external_sha256: String,
    pub optimizer: TrainerOptimizerCheckpoint,
    pub external: TrainerExternalStateCheckpoint,
}

/// Composed readiness receipt returned by both preflight and native restore.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TrainerRuntimeCheckpointValidation {
    pub kind: &'static str,
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub resume_scope: &'static str,
    pub optimizer_sha256: String,
    pub external_sha256: String,
    pub parameter_count: usize,
    pub captures_inflight_accumulators: bool,
    pub parameter_values_external: bool,
    pub required_components: Vec<String>,
    pub captured_components: Vec<String>,
    pub unresolved_components: Vec<String>,
    pub reattach_required_components: Vec<String>,
    pub reattached_components: Vec<String>,
    pub payload_complete: bool,
    pub deterministic_resume_ready: bool,
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum TrainerRuntimeCheckpointError {
    #[error(transparent)]
    Optimizer(#[from] TrainerOptimizerCheckpointError),
    #[error(transparent)]
    External(#[from] TrainerExternalCheckpointError),
    #[error("runtime checkpoint {field} must be {expected}, got {actual}")]
    InvalidMetadata {
        field: &'static str,
        expected: &'static str,
        actual: String,
    },
    #[error("invalid runtime checkpoint state at {field}: {message}")]
    InvalidState { field: String, message: String },
    #[error("runtime checkpoint {field} integrity mismatch: expected {expected}, got {actual}")]
    IntegrityMismatch {
        field: &'static str,
        expected: String,
        actual: String,
    },
    #[error("runtime checkpoint encoding failed: {0}")]
    Encoding(String),
}

/// Builds one envelope from payloads captured at the same trainer boundary.
pub fn build_trainer_runtime_checkpoint_bundle(
    optimizer: TrainerOptimizerCheckpoint,
    external: TrainerExternalStateCheckpoint,
) -> Result<TrainerRuntimeCheckpointBundle, TrainerRuntimeCheckpointError> {
    let optimizer_validation = evaluate_trainer_optimizer_checkpoint(&optimizer)?;
    let external_validation = evaluate_trainer_external_state_checkpoint(&external)?;
    validate_component_alignment(&optimizer_validation, &external_validation)?;

    let optimizer_sha256 = payload_sha256(&optimizer)?;
    let external_sha256 = payload_sha256(&external)?;
    let bundle = TrainerRuntimeCheckpointBundle {
        kind: TRAINER_RUNTIME_CHECKPOINT_BUNDLE_KIND.to_owned(),
        contract_version: TRAINER_RUNTIME_CHECKPOINT_BUNDLE_CONTRACT_VERSION.to_owned(),
        semantic_owner: TRAINER_RUNTIME_CHECKPOINT_BUNDLE_SEMANTIC_OWNER.to_owned(),
        semantic_backend: TRAINER_RUNTIME_CHECKPOINT_BUNDLE_SEMANTIC_BACKEND.to_owned(),
        resume_scope: TRAINER_RUNTIME_CHECKPOINT_BUNDLE_RESUME_SCOPE.to_owned(),
        optimizer_sha256,
        external_sha256,
        optimizer,
        external,
    };
    evaluate_trainer_runtime_checkpoint_bundle(&bundle)?;
    Ok(bundle)
}

/// Validates a portable payload without claiming native resource reattachment.
pub fn evaluate_trainer_runtime_checkpoint_bundle(
    bundle: &TrainerRuntimeCheckpointBundle,
) -> Result<TrainerRuntimeCheckpointValidation, TrainerRuntimeCheckpointError> {
    evaluate_trainer_runtime_checkpoint_bundle_with_reattached(bundle, &[])
}

/// Validates a native restore after concrete resources have been reattached.
pub fn evaluate_trainer_runtime_checkpoint_restore(
    bundle: &TrainerRuntimeCheckpointBundle,
    reattached_components: &[String],
) -> Result<TrainerRuntimeCheckpointValidation, TrainerRuntimeCheckpointError> {
    evaluate_trainer_runtime_checkpoint_bundle_with_reattached(bundle, reattached_components)
}

fn evaluate_trainer_runtime_checkpoint_bundle_with_reattached(
    bundle: &TrainerRuntimeCheckpointBundle,
    reattached_components: &[String],
) -> Result<TrainerRuntimeCheckpointValidation, TrainerRuntimeCheckpointError> {
    validate_metadata("kind", &bundle.kind, TRAINER_RUNTIME_CHECKPOINT_BUNDLE_KIND)?;
    validate_metadata(
        "contract_version",
        &bundle.contract_version,
        TRAINER_RUNTIME_CHECKPOINT_BUNDLE_CONTRACT_VERSION,
    )?;
    validate_metadata(
        "semantic_owner",
        &bundle.semantic_owner,
        TRAINER_RUNTIME_CHECKPOINT_BUNDLE_SEMANTIC_OWNER,
    )?;
    validate_metadata(
        "semantic_backend",
        &bundle.semantic_backend,
        TRAINER_RUNTIME_CHECKPOINT_BUNDLE_SEMANTIC_BACKEND,
    )?;
    validate_metadata(
        "resume_scope",
        &bundle.resume_scope,
        TRAINER_RUNTIME_CHECKPOINT_BUNDLE_RESUME_SCOPE,
    )?;
    validate_digest("optimizer_sha256", &bundle.optimizer_sha256)?;
    validate_digest("external_sha256", &bundle.external_sha256)?;
    validate_integrity(
        "optimizer_sha256",
        &bundle.optimizer_sha256,
        payload_sha256(&bundle.optimizer)?,
    )?;
    validate_integrity(
        "external_sha256",
        &bundle.external_sha256,
        payload_sha256(&bundle.external)?,
    )?;

    let optimizer = evaluate_trainer_optimizer_checkpoint(&bundle.optimizer)?;
    let external = if reattached_components.is_empty() {
        evaluate_trainer_external_state_checkpoint(&bundle.external)?
    } else {
        evaluate_trainer_external_state_restore(&bundle.external, reattached_components)?
    };
    validate_component_alignment(&optimizer, &external)?;

    Ok(TrainerRuntimeCheckpointValidation {
        kind: TRAINER_RUNTIME_CHECKPOINT_BUNDLE_KIND,
        contract_version: TRAINER_RUNTIME_CHECKPOINT_BUNDLE_CONTRACT_VERSION,
        semantic_owner: TRAINER_RUNTIME_CHECKPOINT_BUNDLE_SEMANTIC_OWNER,
        semantic_backend: TRAINER_RUNTIME_CHECKPOINT_BUNDLE_SEMANTIC_BACKEND,
        resume_scope: TRAINER_RUNTIME_CHECKPOINT_BUNDLE_RESUME_SCOPE,
        optimizer_sha256: bundle.optimizer_sha256.clone(),
        external_sha256: bundle.external_sha256.clone(),
        parameter_count: optimizer.parameter_count,
        captures_inflight_accumulators: optimizer.captures_inflight_accumulators,
        parameter_values_external: optimizer.parameter_values_external,
        required_components: external.required_components,
        captured_components: external.captured_components,
        unresolved_components: external.unresolved_components,
        reattach_required_components: external.reattach_required_components,
        reattached_components: external.reattached_components,
        payload_complete: external.payload_complete,
        deterministic_resume_ready: external.deterministic_resume_ready,
    })
}

fn validate_component_alignment(
    optimizer: &TrainerOptimizerCheckpointValidation,
    external: &TrainerExternalCheckpointValidation,
) -> Result<(), TrainerRuntimeCheckpointError> {
    if optimizer.external_state_required != external.required_components {
        return Err(state_error(
            "external.required_components",
            "must exactly match optimizer.topology.external_state_required",
        ));
    }
    Ok(())
}

fn payload_sha256<T: Serialize>(payload: &T) -> Result<String, TrainerRuntimeCheckpointError> {
    let encoded = serde_json::to_vec(payload)
        .map_err(|error| TrainerRuntimeCheckpointError::Encoding(error.to_string()))?;
    let digest = Sha256::digest(encoded);
    let mut hexadecimal = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut hexadecimal, "{byte:02x}").expect("writing to a String cannot fail");
    }
    Ok(hexadecimal)
}

fn validate_digest(field: &str, digest: &str) -> Result<(), TrainerRuntimeCheckpointError> {
    if digest.len() == 64
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(state_error(field, "must be a 64-digit lowercase SHA-256"))
    }
}

fn validate_integrity(
    field: &'static str,
    expected: &str,
    actual: String,
) -> Result<(), TrainerRuntimeCheckpointError> {
    if expected == actual {
        Ok(())
    } else {
        Err(TrainerRuntimeCheckpointError::IntegrityMismatch {
            field,
            expected: expected.to_owned(),
            actual,
        })
    }
}

fn validate_metadata(
    field: &'static str,
    actual: &str,
    expected: &'static str,
) -> Result<(), TrainerRuntimeCheckpointError> {
    if actual == expected {
        Ok(())
    } else {
        Err(TrainerRuntimeCheckpointError::InvalidMetadata {
            field,
            expected,
            actual: actual.to_owned(),
        })
    }
}

fn state_error(field: &str, message: &str) -> TrainerRuntimeCheckpointError {
    TrainerRuntimeCheckpointError::InvalidState {
        field: field.to_owned(),
        message: message.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::{
        AccumulatorSynchronizerCheckpoint, ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_CONTRACT_VERSION,
        ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_KIND,
        ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_SEMANTIC_BACKEND,
        ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_SEMANTIC_OWNER,
    };
    use crate::runtime::trainer_external::{
        build_trainer_external_state_checkpoint, ACCUMULATOR_SYNCHRONIZER_COMPONENT,
    };
    use crate::runtime::trainer_optimizer::{
        build_trainer_optimizer_checkpoint, SpectralLrAdapterState, TrainerBackendCounters,
        TrainerExecutionTopology, TrainerOptimizerConfig, TrainerOptimizerRuntimeState,
        TrainerPhaseTrackerState, TrainerSoftLogicConfigState, TrainerSoftLogicState,
    };
    use serde_json::json;

    fn softlogic_state() -> TrainerSoftLogicState {
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

    fn optimizer(required: Vec<String>) -> TrainerOptimizerCheckpoint {
        build_trainer_optimizer_checkpoint(
            TrainerOptimizerConfig::try_new(-1.0, 0.02, 0.01).unwrap(),
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
                external_state_required: required,
            },
            TrainerOptimizerRuntimeState {
                epoch: 3,
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
                softlogic: softlogic_state(),
                backend_counters: TrainerBackendCounters::default(),
                last_accumulator_sync_buffers: 0,
                last_accumulator_sync_values: 0,
            },
            Vec::new(),
        )
        .unwrap()
    }

    fn external(required: Vec<String>) -> TrainerExternalStateCheckpoint {
        build_trainer_external_state_checkpoint(required, None, None, None).unwrap()
    }

    #[test]
    fn bundle_binds_complete_child_payloads() {
        let bundle =
            build_trainer_runtime_checkpoint_bundle(optimizer(Vec::new()), external(Vec::new()))
                .unwrap();
        let encoded = serde_json::to_string(&bundle).unwrap();
        let decoded: TrainerRuntimeCheckpointBundle = serde_json::from_str(&encoded).unwrap();
        let report = evaluate_trainer_runtime_checkpoint_bundle(&decoded).unwrap();

        assert_eq!(bundle, decoded);
        assert_eq!(report.optimizer_sha256.len(), 64);
        assert_eq!(report.external_sha256.len(), 64);
        assert!(report.payload_complete);
        assert!(report.deterministic_resume_ready);
        assert!(report.parameter_values_external);
    }

    #[test]
    fn bundle_rejects_child_tampering_even_when_child_state_is_valid() {
        let mut bundle =
            build_trainer_runtime_checkpoint_bundle(optimizer(Vec::new()), external(Vec::new()))
                .unwrap();
        bundle.optimizer.state.epoch += 1;

        assert!(matches!(
            evaluate_trainer_runtime_checkpoint_bundle(&bundle),
            Err(TrainerRuntimeCheckpointError::IntegrityMismatch {
                field: "optimizer_sha256",
                ..
            })
        ));
    }

    #[test]
    fn bundle_rejects_mismatched_component_sets() {
        assert!(matches!(
            build_trainer_runtime_checkpoint_bundle(
                optimizer(vec!["blackcat_runtime".to_owned()]),
                external(Vec::new()),
            ),
            Err(TrainerRuntimeCheckpointError::InvalidState { .. })
        ));
    }

    #[test]
    fn bundle_keeps_unresolved_runtime_state_fail_closed() {
        let required = vec!["blackcat_runtime".to_owned()];
        let bundle = build_trainer_runtime_checkpoint_bundle(
            optimizer(required.clone()),
            external(required),
        )
        .unwrap();
        let report = evaluate_trainer_runtime_checkpoint_bundle(&bundle).unwrap();

        assert_eq!(report.unresolved_components, ["blackcat_runtime"]);
        assert!(!report.payload_complete);
        assert!(!report.deterministic_resume_ready);
    }

    #[test]
    fn bundle_requires_native_resource_reattachment() {
        let required = vec![ACCUMULATOR_SYNCHRONIZER_COMPONENT.to_owned()];
        let accumulator = AccumulatorSynchronizerCheckpoint {
            kind: ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_KIND.to_owned(),
            contract_version: ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_CONTRACT_VERSION.to_owned(),
            semantic_owner: ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_SEMANTIC_OWNER.to_owned(),
            semantic_backend: ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_SEMANTIC_BACKEND.to_owned(),
            provider: "test-accumulator".to_owned(),
            rank: 0,
            world_size: 1,
            state: Some(json!({"mode": "mean"})),
        };
        let external = build_trainer_external_state_checkpoint(
            required.clone(),
            None,
            None,
            Some(accumulator),
        )
        .unwrap();
        let bundle =
            build_trainer_runtime_checkpoint_bundle(optimizer(required), external).unwrap();

        let portable = evaluate_trainer_runtime_checkpoint_bundle(&bundle).unwrap();
        assert!(portable.payload_complete);
        assert!(!portable.deterministic_resume_ready);

        let native = evaluate_trainer_runtime_checkpoint_restore(
            &bundle,
            &[ACCUMULATOR_SYNCHRONIZER_COMPONENT.to_owned()],
        )
        .unwrap();
        assert!(native.deterministic_resume_ready);
    }

    #[test]
    fn bundle_serde_rejects_unknown_fields() {
        let bundle =
            build_trainer_runtime_checkpoint_bundle(optimizer(Vec::new()), external(Vec::new()))
                .unwrap();
        let mut payload = serde_json::to_value(bundle).unwrap();
        payload
            .as_object_mut()
            .unwrap()
            .insert("python_ready".to_owned(), json!(true));

        assert!(serde_json::from_value::<TrainerRuntimeCheckpointBundle>(payload).is_err());
    }
}
