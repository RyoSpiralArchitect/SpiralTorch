use serde_json::Value;
use st_core::runtime::trainer_checkpoint::{
    evaluate_trainer_runtime_checkpoint_bundle, TrainerRuntimeCheckpointBundle,
    TrainerRuntimeCheckpointError,
};

#[cfg(target_arch = "wasm32")]
use serde::Serialize;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::js_error;

fn bundle_from_value(value: Value) -> Result<TrainerRuntimeCheckpointBundle, String> {
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn bundle_from_json(bundle_json: &str) -> Result<TrainerRuntimeCheckpointBundle, String> {
    let value = serde_json::from_str(bundle_json).map_err(|error| error.to_string())?;
    bundle_from_value(value)
}

fn add_execution_client(payload: &mut Value) {
    payload
        .as_object_mut()
        .expect("trainer runtime checkpoint validation is an object")
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

/// Preflights an integrity-bound trainer bundle without executing `st-nn`.
pub fn trainer_runtime_checkpoint_bundle_value(
    bundle: &TrainerRuntimeCheckpointBundle,
) -> Result<Value, TrainerRuntimeCheckpointError> {
    let mut payload = serde_json::to_value(evaluate_trainer_runtime_checkpoint_bundle(bundle)?)
        .expect("trainer runtime checkpoint validation is serializable");
    add_execution_client(&mut payload);
    Ok(payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = trainerRuntimeCheckpointBundleJson)]
pub fn trainer_runtime_checkpoint_bundle_json(bundle_json: &str) -> Result<String, JsValue> {
    let bundle = bundle_from_json(bundle_json).map_err(js_error)?;
    let payload = trainer_runtime_checkpoint_bundle_value(&bundle).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = trainerRuntimeCheckpointBundleObject)]
pub fn trainer_runtime_checkpoint_bundle_object(bundle: &JsValue) -> Result<JsValue, JsValue> {
    let bundle = serde_wasm_bindgen::from_value::<Value>(bundle.clone()).map_err(js_error)?;
    let bundle = bundle_from_value(bundle).map_err(js_error)?;
    let payload = trainer_runtime_checkpoint_bundle_value(&bundle).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use st_core::runtime::trainer_checkpoint::build_trainer_runtime_checkpoint_bundle;
    use st_core::runtime::trainer_external::build_trainer_external_state_checkpoint;
    use st_core::runtime::trainer_optimizer::{
        build_trainer_optimizer_checkpoint, SpectralLrAdapterState, TrainerBackendCounters,
        TrainerExecutionTopology, TrainerOptimizerConfig, TrainerOptimizerRuntimeState,
        TrainerPhaseTrackerState, TrainerSoftLogicState,
    };

    fn valid_bundle() -> TrainerRuntimeCheckpointBundle {
        let mut spectral_adapter = SpectralLrAdapterState::default();
        spectral_adapter.curvature_target = -1.0;
        let optimizer = build_trainer_optimizer_checkpoint(
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
                external_state_required: Vec::new(),
            },
            TrainerOptimizerRuntimeState {
                epoch: 0,
                meta_learning_rate_scale: 1.0,
                meta_optimizer_step: None,
                spectral_adapter,
                curvature_scheduler: None,
                spectral_policy: None,
                phase_tracker: TrainerPhaseTrackerState::default(),
                softlogic: TrainerSoftLogicState::default(),
                backend_counters: TrainerBackendCounters::default(),
                last_accumulator_sync_buffers: 0,
                last_accumulator_sync_values: 0,
            },
            Vec::new(),
        )
        .unwrap();
        let external =
            build_trainer_external_state_checkpoint(Vec::new(), None, None, None).unwrap();
        build_trainer_runtime_checkpoint_bundle(optimizer, external).unwrap()
    }

    fn without_client(mut payload: Value) -> Value {
        payload
            .as_object_mut()
            .expect("trainer runtime checkpoint validation object")
            .remove("execution_client");
        payload
    }

    #[test]
    fn wasm_runtime_bundle_preflight_matches_the_rust_validator() {
        let bundle = bundle_from_json(&serde_json::to_string(&valid_bundle()).unwrap()).unwrap();
        let rust =
            serde_json::to_value(evaluate_trainer_runtime_checkpoint_bundle(&bundle).unwrap())
                .unwrap();
        let wasm = without_client(trainer_runtime_checkpoint_bundle_value(&bundle).unwrap());

        assert_eq!(wasm, rust);
        assert_eq!(wasm["semantic_backend"], "rust");
        assert_eq!(wasm["deterministic_resume_ready"], true);
    }

    #[test]
    fn wasm_runtime_bundle_ingress_and_integrity_fail_closed() {
        let bundle = valid_bundle();
        let mut unknown = serde_json::to_value(&bundle).unwrap();
        unknown
            .as_object_mut()
            .unwrap()
            .insert("commander".to_owned(), json!("browser"));
        assert!(bundle_from_value(unknown)
            .unwrap_err()
            .contains("unknown field"));

        let mut tampered = bundle;
        tampered.optimizer.state.epoch = 4;
        assert!(matches!(
            trainer_runtime_checkpoint_bundle_value(&tampered),
            Err(TrainerRuntimeCheckpointError::IntegrityMismatch {
                field: "optimizer_sha256",
                ..
            })
        ));
    }
}
