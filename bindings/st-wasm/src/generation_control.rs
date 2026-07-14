use serde_json::Value;
use st_core::inference::generation_control::{
    apply_zspace_generation_control, ZSpaceGenerationControlError, ZSpaceGenerationControlRequest,
};

#[cfg(target_arch = "wasm32")]
use serde::Serialize;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::js_error;

fn request_from_value(value: Value) -> Result<ZSpaceGenerationControlRequest, String> {
    let request = value
        .as_object()
        .ok_or_else(|| "Z-space generation control request must be an object".to_owned())?;
    for field in ["logits", "token_ids", "recent_tokens"] {
        if request.get(field).is_some_and(|value| !value.is_array()) {
            return Err(format!(
                "Z-space generation control '{field}' must be an array"
            ));
        }
    }
    if request
        .get("config")
        .is_some_and(|value| !value.is_object())
    {
        return Err("Z-space generation control 'config' must be an object".to_owned());
    }
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn request_from_json(input_json: &str) -> Result<ZSpaceGenerationControlRequest, String> {
    let value = serde_json::from_str(input_json).map_err(|error| error.to_string())?;
    request_from_value(value)
}

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

pub fn zspace_generation_control_value(
    request: ZSpaceGenerationControlRequest,
) -> Result<Value, ZSpaceGenerationControlError> {
    Ok(
        serde_json::to_value(apply_zspace_generation_control(request)?)
            .expect("Z-space generation control payload is serializable"),
    )
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceGenerationControlJson)]
pub fn zspace_generation_control_json(request_json: &str) -> Result<String, JsValue> {
    let request = request_from_json(request_json).map_err(js_error)?;
    let payload = zspace_generation_control_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceGenerationControlObject)]
pub fn zspace_generation_control_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let request = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request = request_from_value(request).map_err(js_error)?;
    let payload = zspace_generation_control_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use st_core::inference::generation_control::apply_zspace_generation_control;

    #[test]
    fn wasm_generation_control_matches_rust_exactly() {
        let request: ZSpaceGenerationControlRequest = serde_json::from_value(json!({
            "logits": [4.0, 3.5, 1.0],
            "token_ids": [0, 1, 2],
            "recent_tokens": [0, 0, 0],
            "config": {
                "curvature": -1.0,
                "entropy_target": 1.0,
                "min_temperature": 0.5,
                "max_temperature": 2.0,
                "repression_window": 4,
                "repression_strength": 2.0,
                "last_token_repression": 1.0
            }
        }))
        .expect("valid request");
        let expected = serde_json::to_value(
            apply_zspace_generation_control(request.clone()).expect("valid Rust control"),
        )
        .expect("serializable Rust control");
        let actual =
            zspace_generation_control_value(request).expect("valid WASM generation control");

        assert_eq!(actual, expected);
    }

    #[test]
    fn wasm_generation_control_ingress_fails_closed_on_contract_drift() {
        let unknown = request_from_json(r#"{"config":{"represson_strength":1.0}}"#)
            .expect_err("unknown request fields must fail closed");
        assert!(unknown.contains("unknown field"));

        let wrong_shape =
            request_from_json(r#"{"logits":{}}"#).expect_err("candidate logits must be an array");
        assert_eq!(
            wrong_shape,
            "Z-space generation control 'logits' must be an array"
        );

        let wrong_request = request_from_json(r#"[]"#).expect_err("request must be an object");
        assert_eq!(
            wrong_request,
            "Z-space generation control request must be an object"
        );
    }

    #[test]
    fn wasm_generation_control_preserves_rust_validation() {
        let request = request_from_json(r#"{"config":{"repression_strength":-1.0}}"#)
            .expect("syntactically valid request");
        let error = zspace_generation_control_value(request)
            .expect_err("negative repression must fail closed");
        assert_eq!(
            error,
            ZSpaceGenerationControlError::Negative {
                field: "repression_strength"
            }
        );
    }
}
