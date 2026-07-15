use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;
use st_core::runtime::zspace_optimizer::{
    initialize_zspace_meta_optimizer, restore_zspace_meta_optimizer,
    transition_zspace_meta_optimizer, ZSpaceMetaOptimizerConfig, ZSpaceMetaOptimizerRestoreRequest,
    ZSpaceMetaOptimizerStepRequest,
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::js_error;

fn request_from_value<T: DeserializeOwned>(value: Value, label: &str) -> Result<T, String> {
    if !value.is_object() {
        return Err(format!("{label} must be an object"));
    }
    serde_json::from_value(value).map_err(|error| format!("invalid {label}: {error}"))
}

fn request_from_json<T: DeserializeOwned>(request_json: &str, label: &str) -> Result<T, String> {
    let value: Value = serde_json::from_str(request_json)
        .map_err(|error| format!("invalid {label} JSON: {error}"))?;
    request_from_value(value, label)
}

fn response_value<T: Serialize>(response: &T) -> Result<Value, String> {
    let mut value = serde_json::to_value(response)
        .map_err(|error| format!("Z-space meta-optimizer response encoding failed: {error}"))?;
    value
        .as_object_mut()
        .expect("Z-space meta-optimizer response is an object")
        .insert(
            "execution_client".to_owned(),
            Value::String("wasm".to_owned()),
        );
    Ok(value)
}

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

pub fn zspace_meta_optimizer_init_value(
    config: ZSpaceMetaOptimizerConfig,
) -> Result<Value, String> {
    let checkpoint = initialize_zspace_meta_optimizer(config).map_err(|error| error.to_string())?;
    response_value(&checkpoint)
}

pub fn zspace_meta_optimizer_restore_value(
    request: ZSpaceMetaOptimizerRestoreRequest,
) -> Result<Value, String> {
    let checkpoint = restore_zspace_meta_optimizer(request).map_err(|error| error.to_string())?;
    response_value(&checkpoint)
}

pub fn zspace_meta_optimizer_step_value(
    request: ZSpaceMetaOptimizerStepRequest,
) -> Result<Value, String> {
    let report = transition_zspace_meta_optimizer(request).map_err(|error| error.to_string())?;
    response_value(&report)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceMetaOptimizerInitJson)]
pub fn zspace_meta_optimizer_init_json(config_json: &str) -> Result<String, JsValue> {
    let config =
        request_from_json(config_json, "Z-space meta-optimizer config").map_err(js_error)?;
    let payload = zspace_meta_optimizer_init_value(config).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceMetaOptimizerInitObject)]
pub fn zspace_meta_optimizer_init_object(config: &JsValue) -> Result<JsValue, JsValue> {
    let value = serde_wasm_bindgen::from_value::<Value>(config.clone()).map_err(js_error)?;
    let config = request_from_value(value, "Z-space meta-optimizer config").map_err(js_error)?;
    let payload = zspace_meta_optimizer_init_value(config).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceMetaOptimizerRestoreJson)]
pub fn zspace_meta_optimizer_restore_json(request_json: &str) -> Result<String, JsValue> {
    let request = request_from_json(request_json, "Z-space meta-optimizer restore request")
        .map_err(js_error)?;
    let payload = zspace_meta_optimizer_restore_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceMetaOptimizerRestoreObject)]
pub fn zspace_meta_optimizer_restore_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let value = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request =
        request_from_value(value, "Z-space meta-optimizer restore request").map_err(js_error)?;
    let payload = zspace_meta_optimizer_restore_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceMetaOptimizerStepJson)]
pub fn zspace_meta_optimizer_step_json(request_json: &str) -> Result<String, JsValue> {
    let request =
        request_from_json(request_json, "Z-space meta-optimizer step request").map_err(js_error)?;
    let payload = zspace_meta_optimizer_step_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceMetaOptimizerStepObject)]
pub fn zspace_meta_optimizer_step_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let value = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request =
        request_from_value(value, "Z-space meta-optimizer step request").map_err(js_error)?;
    let payload = zspace_meta_optimizer_step_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use st_core::runtime::zspace_optimizer::{
        restore_zspace_meta_optimizer, transition_zspace_meta_optimizer,
        ZSPACE_META_OPTIMIZER_MAX_SAFE_STEP,
    };

    #[test]
    fn wasm_step_matches_the_rust_contract_exactly() {
        let request: ZSpaceMetaOptimizerStepRequest = serde_json::from_value(json!({
            "config": {"dimension": 4},
            "state": {
                "z": [0.2, -0.1, 0.4, -0.3],
                "first_moment": [0.0, 0.0, 0.0, 0.0],
                "second_moment": [0.0, 0.0, 0.0, 0.0],
                "step": 0
            },
            "observation": {
                "speed": 0.8,
                "memory": 0.5,
                "stability": 0.6,
                "gradient": [0.1, -0.2, 0.3, -0.1],
                "telemetry": {
                    "topos.closure_pressure": 0.75,
                    "topos.training_hints.learning_rate_scale": 0.5
                }
            }
        }))
        .expect("valid request");
        let mut expected = serde_json::to_value(
            transition_zspace_meta_optimizer(request.clone()).expect("Rust transition"),
        )
        .expect("serializable report");
        expected.as_object_mut().expect("object").insert(
            "execution_client".to_owned(),
            Value::String("wasm".to_owned()),
        );

        let actual = zspace_meta_optimizer_step_value(request).expect("WASM transition");

        assert_eq!(actual, expected);
        assert_eq!(actual["semantic_backend"], "rust");
        assert_eq!(actual["execution_client"], "wasm");
        assert_eq!(actual["transition_validated"], true);
    }

    #[test]
    fn wasm_restore_uses_rust_dimension_coercion() {
        let request: ZSpaceMetaOptimizerRestoreRequest = serde_json::from_value(json!({
            "config": {"dimension": 3},
            "state": {
                "z": [1.0],
                "first_moment": [2.0, 3.0, 4.0, 5.0],
                "second_moment": [],
                "step": 7
            },
            "strict": false
        }))
        .expect("valid request");
        let expected = restore_zspace_meta_optimizer(request.clone()).expect("Rust restore");
        let actual = zspace_meta_optimizer_restore_value(request).expect("WASM restore");

        assert_eq!(actual["state"]["z"], json!([1.0, 0.0, 0.0]));
        assert_eq!(actual["state"]["first_moment"], json!([2.0, 3.0, 4.0]));
        assert_eq!(actual["state"]["second_moment"], json!([0.0, 0.0, 0.0]));
        assert_eq!(
            actual["state"],
            serde_json::to_value(expected.state).unwrap()
        );
    }

    #[test]
    fn wasm_rejects_invalid_rust_state() {
        let request: ZSpaceMetaOptimizerRestoreRequest = serde_json::from_value(json!({
            "config": {"dimension": 2},
            "state": {
                "z": [0.0, 0.0],
                "first_moment": [0.0, 0.0],
                "second_moment": [0.0, -0.1],
                "step": 0
            },
            "strict": true
        }))
        .expect("syntactically valid request");
        let error = zspace_meta_optimizer_restore_value(request)
            .expect_err("negative second moment must fail closed");
        assert!(error.contains("second_moment[1]"));
    }

    #[test]
    fn wasm_requires_object_requests() {
        let error = request_from_value::<ZSpaceMetaOptimizerConfig>(
            json!([]),
            "Z-space meta-optimizer config",
        )
        .expect_err("array is not a config object");
        assert_eq!(error, "Z-space meta-optimizer config must be an object");
    }

    #[test]
    fn wasm_step_counter_stops_at_the_shared_exact_integer_limit() {
        let request: ZSpaceMetaOptimizerStepRequest = serde_json::from_value(json!({
            "config": {"dimension": 2},
            "state": {
                "z": [0.0, 0.0],
                "first_moment": [0.0, 0.0],
                "second_moment": [0.0, 0.0],
                "step": ZSPACE_META_OPTIMIZER_MAX_SAFE_STEP
            },
            "observation": {"gradient": [0.1, -0.2]}
        }))
        .expect("exact cross-client step");

        let error = zspace_meta_optimizer_step_value(request)
            .expect_err("next step must exceed the shared exact integer limit");

        assert!(error.contains("cross-client maximum"));
    }
}
