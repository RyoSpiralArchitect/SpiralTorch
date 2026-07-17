use serde_json::Value;
use st_core::runtime::trainer_optimizer::{
    evaluate_trainer_optimizer_checkpoint, evaluate_trainer_optimizer_config,
    TrainerOptimizerCheckpoint, TrainerOptimizerCheckpointError, TrainerOptimizerConfig,
    TrainerOptimizerConfigError,
};

#[cfg(target_arch = "wasm32")]
use serde::Serialize;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::js_error;

fn config_from_value(value: Value) -> Result<TrainerOptimizerConfig, String> {
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn config_from_json(config_json: &str) -> Result<TrainerOptimizerConfig, String> {
    let value = serde_json::from_str(config_json).map_err(|error| error.to_string())?;
    config_from_value(value)
}

fn checkpoint_from_value(value: Value) -> Result<TrainerOptimizerCheckpoint, String> {
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn checkpoint_from_json(checkpoint_json: &str) -> Result<TrainerOptimizerCheckpoint, String> {
    let value = serde_json::from_str(checkpoint_json).map_err(|error| error.to_string())?;
    checkpoint_from_value(value)
}

fn add_execution_client(payload: &mut Value) {
    payload
        .as_object_mut()
        .expect("trainer optimizer config payload is an object")
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

/// Evaluate browser-provided trainer controls through the shared Rust contract.
pub fn trainer_optimizer_config_value(
    config: TrainerOptimizerConfig,
) -> Result<Value, TrainerOptimizerConfigError> {
    let mut payload = serde_json::to_value(evaluate_trainer_optimizer_config(config)?)
        .expect("trainer optimizer config payload is serializable");
    add_execution_client(&mut payload);
    Ok(payload)
}

/// Validate a transported trainer checkpoint without pretending WASM runs `st-nn`.
pub fn trainer_optimizer_checkpoint_value(
    checkpoint: &TrainerOptimizerCheckpoint,
) -> Result<Value, TrainerOptimizerCheckpointError> {
    let mut payload = serde_json::to_value(evaluate_trainer_optimizer_checkpoint(checkpoint)?)
        .expect("trainer optimizer checkpoint validation is serializable");
    add_execution_client(&mut payload);
    Ok(payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = trainerOptimizerConfigJson)]
pub fn trainer_optimizer_config_json(config_json: &str) -> Result<String, JsValue> {
    let config = config_from_json(config_json).map_err(js_error)?;
    let payload = trainer_optimizer_config_value(config).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = trainerOptimizerConfigObject)]
pub fn trainer_optimizer_config_object(config: &JsValue) -> Result<JsValue, JsValue> {
    let config = serde_wasm_bindgen::from_value::<Value>(config.clone()).map_err(js_error)?;
    let config = config_from_value(config).map_err(js_error)?;
    let payload = trainer_optimizer_config_value(config).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = trainerOptimizerCheckpointJson)]
pub fn trainer_optimizer_checkpoint_json(checkpoint_json: &str) -> Result<String, JsValue> {
    let checkpoint = checkpoint_from_json(checkpoint_json).map_err(js_error)?;
    let payload = trainer_optimizer_checkpoint_value(&checkpoint).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = trainerOptimizerCheckpointObject)]
pub fn trainer_optimizer_checkpoint_object(checkpoint: &JsValue) -> Result<JsValue, JsValue> {
    let checkpoint =
        serde_wasm_bindgen::from_value::<Value>(checkpoint.clone()).map_err(js_error)?;
    let checkpoint = checkpoint_from_value(checkpoint).map_err(js_error)?;
    let payload = trainer_optimizer_checkpoint_value(&checkpoint).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use st_core::runtime::trainer_optimizer::{
        build_trainer_optimizer_checkpoint, SpectralLrAdapterState, TrainerBackendCounters,
        TrainerExecutionTopology, TrainerOptimizerRuntimeState, TrainerParameterOptimizerState,
        TrainerPhaseTrackerState, TrainerSoftLogicState, TRAINER_OPTIMIZER_MAX_SAFE_INTEGER,
    };
    use st_tensor::AmegaHypergrad;

    fn without_client(mut payload: Value) -> Value {
        payload
            .as_object_mut()
            .expect("trainer optimizer config payload object")
            .remove("execution_client");
        payload
    }

    fn valid_checkpoint() -> TrainerOptimizerCheckpoint {
        let config = TrainerOptimizerConfig::try_new(-1.0, 0.02, 0.01).unwrap();
        let mut spectral_adapter = SpectralLrAdapterState::default();
        spectral_adapter.curvature_target = -1.0;
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
            vec![TrainerParameterOptimizerState {
                name: "weight".to_owned(),
                rows: 1,
                cols: 2,
                value_fingerprint: "0123456789abcdef".to_owned(),
                euclidean_gradient: None,
                hypergrad: Some(AmegaHypergrad::new(-1.0, 0.02, 1, 2).unwrap().checkpoint()),
                realgrad: None,
            }],
        )
        .unwrap()
    }

    #[test]
    fn wasm_trainer_optimizer_config_matches_the_rust_contract() {
        let config = config_from_json(
            r#"{
                "curvature":-1.0,
                "hyper_learning_rate":0.02,
                "fallback_learning_rate":0.01,
                "real_learning_rate":0.005,
                "grad_clip_max_norm":1.5
            }"#,
        )
        .expect("valid trainer optimizer config");
        let rust = serde_json::to_value(
            evaluate_trainer_optimizer_config(config).expect("valid Rust contract"),
        )
        .expect("serializable Rust contract");
        let wasm = without_client(
            trainer_optimizer_config_value(config).expect("valid WASM contract transport"),
        );

        assert_eq!(wasm, rust);
        assert_eq!(wasm["semantic_backend"], "rust");
        assert_eq!(wasm["realgrad_enabled"], true);
        assert_eq!(wasm["gradient_clip_enabled"], true);
    }

    #[test]
    fn wasm_ingress_and_rust_validation_fail_closed() {
        let unknown = config_from_json(
            r#"{
                "curvature":-1.0,
                "hyper_learning_rate":0.02,
                "fallback_learning_rate":0.01,
                "commander":"browser"
            }"#,
        )
        .expect_err("unknown config fields must fail closed");
        assert!(unknown.contains("unknown field"));

        let invalid = config_from_value(json!({
            "curvature": 1.0,
            "hyper_learning_rate": 0.02,
            "fallback_learning_rate": 0.01
        }))
        .expect("semantic errors are evaluated by the shared contract");
        assert!(matches!(
            trainer_optimizer_config_value(invalid),
            Err(TrainerOptimizerConfigError::InvalidCurvature { .. })
        ));
    }

    #[test]
    fn wasm_checkpoint_preflight_matches_the_rust_validator() {
        let checkpoint = checkpoint_from_json(&serde_json::to_string(&valid_checkpoint()).unwrap())
            .expect("checkpoint JSON ingress");
        let rust = serde_json::to_value(
            evaluate_trainer_optimizer_checkpoint(&checkpoint).expect("valid Rust checkpoint"),
        )
        .unwrap();
        let wasm = without_client(
            trainer_optimizer_checkpoint_value(&checkpoint)
                .expect("valid WASM checkpoint transport"),
        );

        assert_eq!(wasm, rust);
        assert_eq!(wasm["semantic_backend"], "rust");
        assert_eq!(wasm["deterministic_resume_ready"], true);
        assert_eq!(wasm["parameter_count"], 1);
    }

    #[test]
    fn wasm_checkpoint_ingress_rejects_schema_and_semantic_tampering() {
        let checkpoint = valid_checkpoint();
        let mut unknown = serde_json::to_value(&checkpoint).unwrap();
        unknown
            .as_object_mut()
            .unwrap()
            .insert("commander".to_owned(), json!("browser"));
        assert!(checkpoint_from_value(unknown)
            .unwrap_err()
            .contains("unknown field"));

        let mut tampered = checkpoint;
        tampered.semantic_backend = "wasm".to_owned();
        assert!(matches!(
            trainer_optimizer_checkpoint_value(&tampered),
            Err(TrainerOptimizerCheckpointError::InvalidMetadata {
                field: "semantic_backend",
                ..
            })
        ));

        let mut imprecise = valid_checkpoint();
        imprecise.state.meta_optimizer_step = Some(TRAINER_OPTIMIZER_MAX_SAFE_INTEGER + 1);
        let error = trainer_optimizer_checkpoint_value(&imprecise)
            .expect_err("WASM must reject integers that JavaScript cannot preserve");
        assert!(matches!(
            error,
            TrainerOptimizerCheckpointError::InvalidState { ref field, .. }
                if field == "state.meta_optimizer_step"
        ));
    }
}
