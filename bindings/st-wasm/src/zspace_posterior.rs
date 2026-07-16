use serde_json::Value;
use st_core::inference::zspace_posterior::{
    decode_zspace_posterior, project_zspace_posterior, ZSpacePosteriorDecodeRequest,
    ZSpacePosteriorError, ZSpacePosteriorProjectionRequest,
};

#[cfg(target_arch = "wasm32")]
use serde::Serialize;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::js_error;

fn add_execution_client(payload: &mut Value) {
    let payload = payload
        .as_object_mut()
        .expect("Z-space posterior payload is an object");
    payload.insert(
        "execution_client".to_owned(),
        Value::String("wasm".to_owned()),
    );
    for field in ["prior", "telemetry"] {
        if let Some(nested) = payload.get_mut(field).and_then(Value::as_object_mut) {
            nested.insert(
                "execution_client".to_owned(),
                Value::String("wasm".to_owned()),
            );
        }
    }
}

fn request_object<'a>(
    value: &'a Value,
    context: &str,
) -> Result<&'a serde_json::Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{context} request must be an object"))
}

fn decode_request_from_value(value: Value) -> Result<ZSpacePosteriorDecodeRequest, String> {
    let request = request_object(&value, "Z-space posterior decode")?;
    if request
        .get("z_state")
        .is_some_and(|value| !value.is_array())
    {
        return Err("Z-space posterior decode 'z_state' must be an array".to_owned());
    }
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn decode_request_from_json(input_json: &str) -> Result<ZSpacePosteriorDecodeRequest, String> {
    let value = serde_json::from_str(input_json).map_err(|error| error.to_string())?;
    decode_request_from_value(value)
}

fn projection_request_from_value(value: Value) -> Result<ZSpacePosteriorProjectionRequest, String> {
    let request = request_object(&value, "Z-space posterior projection")?;
    if request
        .get("z_state")
        .is_some_and(|value| !value.is_array())
    {
        return Err("Z-space posterior projection 'z_state' must be an array".to_owned());
    }
    if request
        .get("partial")
        .is_some_and(|value| !value.is_object())
    {
        return Err("Z-space posterior projection 'partial' must be an object".to_owned());
    }
    if request
        .get("telemetry")
        .is_some_and(|value| !value.is_array())
    {
        return Err("Z-space posterior projection 'telemetry' must be an array".to_owned());
    }
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn projection_request_from_json(
    input_json: &str,
) -> Result<ZSpacePosteriorProjectionRequest, String> {
    let value = serde_json::from_str(input_json).map_err(|error| error.to_string())?;
    projection_request_from_value(value)
}

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

/// Decode a latent state through the shared Rust contract and label WASM as the client.
pub fn zspace_posterior_decode_value(
    request: ZSpacePosteriorDecodeRequest,
) -> Result<Value, ZSpacePosteriorError> {
    let mut payload = serde_json::to_value(decode_zspace_posterior(request)?)
        .expect("Z-space posterior decode payload is serializable");
    add_execution_client(&mut payload);
    Ok(payload)
}

/// Project partial observations through the shared Rust contract and label WASM as the client.
pub fn zspace_posterior_project_value(
    request: ZSpacePosteriorProjectionRequest,
) -> Result<Value, ZSpacePosteriorError> {
    let mut payload = serde_json::to_value(project_zspace_posterior(request)?)
        .expect("Z-space posterior projection payload is serializable");
    add_execution_client(&mut payload);
    Ok(payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspacePosteriorDecodeJson)]
pub fn zspace_posterior_decode_json(request_json: &str) -> Result<String, JsValue> {
    let request = decode_request_from_json(request_json).map_err(js_error)?;
    let payload = zspace_posterior_decode_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspacePosteriorDecodeObject)]
pub fn zspace_posterior_decode_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let request = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request = decode_request_from_value(request).map_err(js_error)?;
    let payload = zspace_posterior_decode_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspacePosteriorProjectJson)]
pub fn zspace_posterior_project_json(request_json: &str) -> Result<String, JsValue> {
    let request = projection_request_from_json(request_json).map_err(js_error)?;
    let payload = zspace_posterior_project_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspacePosteriorProjectObject)]
pub fn zspace_posterior_project_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let request = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request = projection_request_from_value(request).map_err(js_error)?;
    let payload = zspace_posterior_project_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn remove_client_metadata(payload: &mut Value) {
        let payload = payload.as_object_mut().expect("posterior payload object");
        payload.remove("execution_client");
        for field in ["prior", "telemetry"] {
            if let Some(nested) = payload.get_mut(field).and_then(Value::as_object_mut) {
                nested.remove("execution_client");
            }
        }
    }

    #[test]
    fn wasm_posterior_decode_matches_rust_except_client_metadata() {
        let request = ZSpacePosteriorDecodeRequest {
            z_state: vec![0.12, -0.03, 0.48, -0.2],
            alpha: 0.35,
        };
        let expected = serde_json::to_value(
            decode_zspace_posterior(request.clone()).expect("valid Rust posterior decode"),
        )
        .expect("serializable Rust posterior decode");
        let mut actual =
            zspace_posterior_decode_value(request).expect("valid WASM posterior decode");
        remove_client_metadata(&mut actual);

        assert_eq!(actual, expected);
    }

    #[test]
    fn wasm_posterior_projection_matches_rust_except_client_metadata() {
        let request: ZSpacePosteriorProjectionRequest = serde_json::from_value(json!({
            "z_state": [0.12, -0.03, 0.48, -0.2],
            "alpha": 0.35,
            "partial": {"speed": 0.3, "mem": -0.2, "gradient": [2.0, -1.0]},
            "gradient_basis": "test.posterior.control.v1",
            "smoothing": 0.35,
            "telemetry": [{"psi": {"energy": 2.0, "focus": 0.4}}]
        }))
        .expect("valid posterior projection request");
        let expected = serde_json::to_value(
            project_zspace_posterior(request.clone()).expect("valid Rust posterior projection"),
        )
        .expect("serializable Rust posterior projection");
        let mut actual =
            zspace_posterior_project_value(request).expect("valid WASM posterior projection");
        remove_client_metadata(&mut actual);

        assert_eq!(actual, expected);
    }

    #[test]
    fn wasm_posterior_ingress_fails_closed_on_contract_drift() {
        let unknown = decode_request_from_json(r#"{"z_state":[0.1],"fractional_order":0.4}"#)
            .expect_err("unknown request fields must fail closed");
        assert!(unknown.contains("unknown field"));

        let wrong_state = decode_request_from_json(r#"{"z_state":{}}"#)
            .expect_err("posterior state must be an array");
        assert_eq!(
            wrong_state,
            "Z-space posterior decode 'z_state' must be an array"
        );

        let wrong_telemetry =
            projection_request_from_json(r#"{"z_state":[0.1],"telemetry":{"psi":1.0}}"#)
                .expect_err("posterior telemetry must be an array");
        assert_eq!(
            wrong_telemetry,
            "Z-space posterior projection 'telemetry' must be an array"
        );
    }

    #[test]
    fn wasm_posterior_preserves_rust_validation() {
        let request = projection_request_from_json(r#"{"z_state":[0.1],"smoothing":1.1}"#)
            .expect("syntactically valid request");
        let error = zspace_posterior_project_value(request)
            .expect_err("out-of-range smoothing must fail closed");
        assert_eq!(error, ZSpacePosteriorError::InvalidSmoothing { value: 1.1 });

        let request =
            projection_request_from_json(r#"{"z_state":[0.1],"partial":{"gradient":[0.2]}}"#)
                .expect("syntactically valid untagged control gradient");
        let error = zspace_posterior_project_value(request)
            .expect_err("untagged external controls must fail closed");
        assert_eq!(error, ZSpacePosteriorError::MissingControlGradientBasis);
    }
}
