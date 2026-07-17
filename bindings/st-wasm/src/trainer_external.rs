use serde_json::Value;
use st_core::runtime::trainer_external::{
    evaluate_trainer_external_state_checkpoint, TrainerExternalCheckpointError,
    TrainerExternalStateCheckpoint,
};

#[cfg(target_arch = "wasm32")]
use serde::Serialize;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::js_error;

fn checkpoint_from_value(value: Value) -> Result<TrainerExternalStateCheckpoint, String> {
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn checkpoint_from_json(checkpoint_json: &str) -> Result<TrainerExternalStateCheckpoint, String> {
    let value = serde_json::from_str(checkpoint_json).map_err(|error| error.to_string())?;
    checkpoint_from_value(value)
}

fn add_execution_client(payload: &mut Value) {
    payload
        .as_object_mut()
        .expect("trainer external checkpoint validation is an object")
        .insert(
            "execution_client".to_owned(),
            Value::String("wasm".to_owned()),
        );
}

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

/// Preflight Rust-owned external trainer state without restoring native resources.
pub fn trainer_external_state_checkpoint_value(
    checkpoint: &TrainerExternalStateCheckpoint,
) -> Result<Value, TrainerExternalCheckpointError> {
    let mut payload = serde_json::to_value(evaluate_trainer_external_state_checkpoint(checkpoint)?)
        .expect("trainer external checkpoint validation is serializable");
    add_execution_client(&mut payload);
    Ok(payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = trainerExternalStateCheckpointJson)]
pub fn trainer_external_state_checkpoint_json(checkpoint_json: &str) -> Result<String, JsValue> {
    let checkpoint = checkpoint_from_json(checkpoint_json).map_err(js_error)?;
    let payload = trainer_external_state_checkpoint_value(&checkpoint).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = trainerExternalStateCheckpointObject)]
pub fn trainer_external_state_checkpoint_object(checkpoint: &JsValue) -> Result<JsValue, JsValue> {
    let checkpoint =
        serde_wasm_bindgen::from_value::<Value>(checkpoint.clone()).map_err(js_error)?;
    let checkpoint = checkpoint_from_value(checkpoint).map_err(js_error)?;
    let payload = trainer_external_state_checkpoint_value(&checkpoint).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use st_core::distributed::{
        AccumulatorSynchronizerCheckpoint, ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_CONTRACT_VERSION,
        ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_KIND,
        ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_SEMANTIC_BACKEND,
        ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_SEMANTIC_OWNER,
    };
    use st_core::inference::gnn_roundtable::{
        GnnRoundtableBandEnergyObservation, GnnRoundtableBandSizes, GnnRoundtableSignalObservation,
        GnnRoundtableSpectralObservation,
    };
    use st_core::inference::zspace_coherence::{
        build_zspace_coherence_distribution_witness, ZSpaceCoherenceClassificationPolicy,
    };
    use st_core::runtime::trainer_external::{
        build_trainer_external_state_checkpoint_from_components, DesireRoundtableCheckpoint,
        DesireTrainerEventCheckpoint, DesireTrainerPhaseCheckpoint, DesireTrainerQueueCheckpoint,
        DesireTrainerWeightsCheckpoint, TrainerCoherenceBridgeCheckpoint,
        TrainerCoherenceSignalCheckpoint, TrainerExternalCheckpointComponents,
        TrainerGnnRoundtableBridgeCheckpoint, TrainerGnnRoundtableSignalCheckpoint,
        TrainerTimestampCheckpoint, ACCUMULATOR_SYNCHRONIZER_COMPONENT, COHERENCE_BRIDGE_COMPONENT,
        DESIRE_ROUNDTABLE_COMPONENT, DESIRE_TRAINER_COMPONENT, GNN_ROUNDTABLE_BRIDGE_COMPONENT,
    };

    fn without_client(mut payload: Value) -> Value {
        payload
            .as_object_mut()
            .expect("trainer external checkpoint validation object")
            .remove("execution_client");
        payload
    }

    fn valid_checkpoint() -> TrainerExternalStateCheckpoint {
        let coherence = TrainerCoherenceSignalCheckpoint {
            dominant_channel: Some(0),
            preserved_channels: 3,
            z_bias: 0.2,
            distribution_witness: build_zspace_coherence_distribution_witness(&[0.6_f64, 0.3, 0.1])
                .unwrap(),
            energy_ratio: 0.75,
            raw_mean_coherence: 0.4,
            classification_policy: ZSpaceCoherenceClassificationPolicy::default(),
            repaired_non_finite_weights: 0,
            repaired_negative_weights: 0,
            pre_discard_repaired_non_finite: 0,
            pre_discard_repaired_negative: 0,
        };
        let gnn_signal = TrainerGnnRoundtableSignalCheckpoint {
            observation: GnnRoundtableSignalObservation {
                band_energy: GnnRoundtableBandEnergyObservation {
                    above: 1.4,
                    here: 0.45,
                    beneath: 0.2,
                    drift: 0.35,
                },
                band_sizes: GnnRoundtableBandSizes {
                    above: 4,
                    here: 2,
                    beneath: 2,
                },
                spectral: GnnRoundtableSpectralObservation {
                    sheet_index: 1,
                    sheet_confidence: 0.9,
                    curvature: 0.6,
                    spin: 0.85,
                    energy: 0.72,
                },
            },
            issued_at: TrainerTimestampCheckpoint {
                unix_seconds: 19,
                subsec_nanos: 11,
            },
        };
        build_trainer_external_state_checkpoint_from_components(
            vec![
                ACCUMULATOR_SYNCHRONIZER_COMPONENT.to_owned(),
                COHERENCE_BRIDGE_COMPONENT.to_owned(),
                DESIRE_ROUNDTABLE_COMPONENT.to_owned(),
                DESIRE_TRAINER_COMPONENT.to_owned(),
                GNN_ROUNDTABLE_BRIDGE_COMPONENT.to_owned(),
            ],
            TrainerExternalCheckpointComponents::new()
                .with_desire_trainer(Some(DesireTrainerQueueCheckpoint {
                    events: vec![DesireTrainerEventCheckpoint {
                        timestamp: TrainerTimestampCheckpoint {
                            unix_seconds: 17,
                            subsec_nanos: 42,
                        },
                        phase: DesireTrainerPhaseCheckpoint::Observation,
                        temperature: 0.9,
                        entropy: 0.7,
                        hypergrad_penalty: 0.1,
                        weights: DesireTrainerWeightsCheckpoint {
                            alpha: 0.4,
                            beta: 0.3,
                            gamma: 0.2,
                            lambda: 0.1,
                        },
                        trigger: None,
                    }],
                }))
                .with_desire_roundtable(Some(DesireRoundtableCheckpoint {
                    blend: 0.4,
                    drift_gain: 0.3,
                    latest: None,
                    pending_summary: None,
                }))
                .with_coherence_bridge(Some(TrainerCoherenceBridgeCheckpoint {
                    subscribed: true,
                    pending: Some(coherence.clone()),
                    latest: Some(coherence),
                }))
                .with_gnn_roundtable_bridge(Some(TrainerGnnRoundtableBridgeCheckpoint {
                    history_limit: 4,
                    history: vec![gnn_signal.clone()],
                    latest: Some(gnn_signal.clone()),
                    trainer_last: Some(gnn_signal),
                }))
                .with_accumulator_synchronizer(Some(AccumulatorSynchronizerCheckpoint {
                    kind: ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_KIND.to_owned(),
                    contract_version: ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_CONTRACT_VERSION
                        .to_owned(),
                    semantic_owner: ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_SEMANTIC_OWNER.to_owned(),
                    semantic_backend: ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_SEMANTIC_BACKEND
                        .to_owned(),
                    provider: "spiraltorch-wasm.tests.accumulator.v1".to_owned(),
                    rank: 0,
                    world_size: 1,
                    state: Some(json!({ "group": "alpha" })),
                })),
        )
        .unwrap()
    }

    #[test]
    fn wasm_external_checkpoint_preflight_matches_the_rust_validator() {
        let checkpoint =
            checkpoint_from_json(&serde_json::to_string(&valid_checkpoint()).unwrap()).unwrap();
        let rust =
            serde_json::to_value(evaluate_trainer_external_state_checkpoint(&checkpoint).unwrap())
                .unwrap();
        let wasm = without_client(trainer_external_state_checkpoint_value(&checkpoint).unwrap());

        assert_eq!(wasm, rust);
        assert_eq!(wasm["semantic_backend"], "rust");
        assert_eq!(wasm["payload_complete"], true);
        assert_eq!(wasm["deterministic_resume_ready"], false);
        assert_eq!(checkpoint.desire_trainer.unwrap().events.len(), 1);
        assert!(checkpoint.coherence_bridge.unwrap().subscribed);
        assert_eq!(checkpoint.gnn_roundtable_bridge.unwrap().history.len(), 1);
        assert_eq!(
            wasm["reattach_required_components"],
            json!([ACCUMULATOR_SYNCHRONIZER_COMPONENT])
        );
        assert_eq!(wasm["reattached_components"], json!([]));
    }

    #[test]
    fn wasm_external_checkpoint_ingress_and_semantics_fail_closed() {
        let checkpoint = valid_checkpoint();
        let mut unknown = serde_json::to_value(&checkpoint).unwrap();
        unknown
            .as_object_mut()
            .unwrap()
            .insert("commander".to_owned(), json!("browser"));
        assert!(checkpoint_from_value(unknown)
            .unwrap_err()
            .contains("unknown field"));

        let mut tampered = checkpoint.clone();
        tampered.semantic_backend = "wasm".to_owned();
        assert!(matches!(
            trainer_external_state_checkpoint_value(&tampered),
            Err(TrainerExternalCheckpointError::InvalidMetadata {
                field: "semantic_backend",
                ..
            })
        ));

        let mut dishonest = checkpoint;
        dishonest.unresolved_components = vec![DESIRE_ROUNDTABLE_COMPONENT.to_owned()];
        assert!(matches!(
            trainer_external_state_checkpoint_value(&dishonest),
            Err(TrainerExternalCheckpointError::InvalidState { .. })
        ));

        let mut invalid_coherence = valid_checkpoint();
        invalid_coherence
            .coherence_bridge
            .as_mut()
            .unwrap()
            .pending
            .as_mut()
            .unwrap()
            .dominant_channel = Some(1);
        assert!(matches!(
            trainer_external_state_checkpoint_value(&invalid_coherence),
            Err(TrainerExternalCheckpointError::Coherence(_))
        ));

        let mut invalid_gnn = valid_checkpoint();
        invalid_gnn.gnn_roundtable_bridge.as_mut().unwrap().history[0]
            .observation
            .spectral
            .sheet_confidence = 1.5;
        assert!(matches!(
            trainer_external_state_checkpoint_value(&invalid_gnn),
            Err(TrainerExternalCheckpointError::GnnRoundtable(_))
        ));
    }
}
