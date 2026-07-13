use serde_json::Value;
use st_core::telemetry::training_projection::{
    project_training_telemetry, TrainingTelemetryProjectionError,
    TrainingTelemetryProjectionRequest,
};

#[cfg(target_arch = "wasm32")]
use serde::Serialize;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::js_error;

fn request_from_value(value: Value) -> Result<TrainingTelemetryProjectionRequest, String> {
    let request = value
        .as_object()
        .ok_or_else(|| "training telemetry projection request must be an object".to_owned())?;
    for field in ["observation", "config"] {
        if request.get(field).is_some_and(|value| !value.is_object()) {
            return Err(format!(
                "training telemetry projection '{field}' must be an object"
            ));
        }
    }
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn request_from_json(input_json: &str) -> Result<TrainingTelemetryProjectionRequest, String> {
    let value = serde_json::from_str(input_json).map_err(|error| error.to_string())?;
    request_from_value(value)
}

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

/// Project training telemetry through the shared Rust contract for WASM clients.
pub fn training_telemetry_projection_value(
    request: TrainingTelemetryProjectionRequest,
) -> Result<Value, TrainingTelemetryProjectionError> {
    let mut payload = serde_json::to_value(project_training_telemetry(request)?)
        .expect("training telemetry projection payload is serializable");
    payload
        .as_object_mut()
        .expect("training telemetry projection payload is an object")
        .insert(
            "execution_client".to_owned(),
            Value::String("wasm".to_owned()),
        );
    Ok(payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = trainingTelemetryProjectionJson)]
pub fn training_telemetry_projection_json(request_json: &str) -> Result<String, JsValue> {
    let request = request_from_json(request_json).map_err(js_error)?;
    let payload = training_telemetry_projection_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = trainingTelemetryProjectionObject)]
pub fn training_telemetry_projection_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let request = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request = request_from_value(request).map_err(js_error)?;
    let payload = training_telemetry_projection_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn wasm_projection_matches_rust_except_client_metadata() {
        let request: TrainingTelemetryProjectionRequest = serde_json::from_value(json!({
            "observation": {
                "step": 4.0,
                "max_steps": 10.0,
                "loss": 2.0,
                "previous_loss": 2.5,
                "grad_norm": 4.0,
                "learning_rate": 0.00005
            },
            "config": {"desire_gain": 1.2, "psi_gain": 0.8}
        }))
        .expect("valid request");
        let expected = serde_json::to_value(
            project_training_telemetry(request.clone()).expect("valid Rust projection"),
        )
        .expect("serializable Rust projection");
        let mut actual =
            training_telemetry_projection_value(request).expect("valid WASM projection");
        actual
            .as_object_mut()
            .expect("WASM projection object")
            .remove("execution_client");

        assert_eq!(actual, expected);
    }

    #[test]
    fn wasm_ingress_fails_closed_on_contract_drift() {
        let unknown = request_from_json(r#"{"observation":{"loss":2.0,"lozz":1.0}}"#)
            .expect_err("unknown request fields must fail closed");
        assert!(unknown.contains("unknown field"));

        let wrong_shape =
            request_from_json(r#"{"observation":[]}"#).expect_err("observation must be an object");
        assert_eq!(
            wrong_shape,
            "training telemetry projection 'observation' must be an object"
        );

        let wrong_request = request_from_json(r#"[]"#).expect_err("request must be an object");
        assert_eq!(
            wrong_request,
            "training telemetry projection request must be an object"
        );
    }

    #[test]
    fn wasm_projection_preserves_rust_validation() {
        let request = request_from_json(r#"{"observation":{"grad_norm":-1.0}}"#)
            .expect("syntactically valid request");
        let error = training_telemetry_projection_value(request)
            .expect_err("negative gradient norm must fail closed");
        assert_eq!(
            error,
            TrainingTelemetryProjectionError::NegativeObservation { field: "grad_norm" }
        );
    }
}
