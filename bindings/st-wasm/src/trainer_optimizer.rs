use serde_json::Value;
use st_core::runtime::trainer_optimizer::{
    evaluate_trainer_optimizer_config, TrainerOptimizerConfig, TrainerOptimizerConfigError,
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn without_client(mut payload: Value) -> Value {
        payload
            .as_object_mut()
            .expect("trainer optimizer config payload object")
            .remove("execution_client");
        payload
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
}
