use serde_json::Value;
use st_core::inference::zspace_coherence::{
    project_zspace_coherence, ZSpaceCoherenceProjectionError, ZSpaceCoherenceProjectionRequest,
};

#[cfg(target_arch = "wasm32")]
use serde::Serialize;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::js_error;

fn add_execution_client(payload: &mut Value) {
    payload
        .as_object_mut()
        .expect("Z-space coherence projection payload is an object")
        .insert(
            "execution_client".to_owned(),
            Value::String("wasm".to_owned()),
        );
}

fn request_from_value(value: Value) -> Result<ZSpaceCoherenceProjectionRequest, String> {
    let request = value
        .as_object()
        .ok_or_else(|| "Z-space coherence projection request must be an object".to_owned())?;
    if request
        .get("diagnostics")
        .is_some_and(|value| !value.is_object())
    {
        return Err("Z-space coherence projection 'diagnostics' must be an object".to_owned());
    }
    if request
        .get("coherence")
        .is_some_and(|value| !value.is_array())
    {
        return Err("Z-space coherence projection 'coherence' must be an array".to_owned());
    }
    if request
        .get("contour")
        .is_some_and(|value| !value.is_object() && !value.is_null())
    {
        return Err("Z-space coherence projection 'contour' must be an object".to_owned());
    }
    if request
        .get("config")
        .is_some_and(|value| !value.is_object())
    {
        return Err("Z-space coherence projection 'config' must be an object".to_owned());
    }
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn request_from_json(input_json: &str) -> Result<ZSpaceCoherenceProjectionRequest, String> {
    let value = serde_json::from_str(input_json).map_err(|error| error.to_string())?;
    request_from_value(value)
}

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

/// Project coherence diagnostics through the shared Rust contract for WASM clients.
pub fn zspace_coherence_project_value(
    request: ZSpaceCoherenceProjectionRequest,
) -> Result<Value, ZSpaceCoherenceProjectionError> {
    let mut payload = serde_json::to_value(project_zspace_coherence(request)?)
        .expect("Z-space coherence projection payload is serializable");
    add_execution_client(&mut payload);
    Ok(payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceCoherenceProjectJson)]
pub fn zspace_coherence_project_json(request_json: &str) -> Result<String, JsValue> {
    let request = request_from_json(request_json).map_err(js_error)?;
    let payload = zspace_coherence_project_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceCoherenceProjectObject)]
pub fn zspace_coherence_project_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let request = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request = request_from_value(request).map_err(js_error)?;
    let payload = zspace_coherence_project_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request() -> ZSpaceCoherenceProjectionRequest {
        serde_json::from_value(json!({
            "diagnostics": {
                "mean_coherence": 0.3333333333333333,
                "coherence_entropy": 1.0296530140645737,
                "energy_ratio": 0.7,
                "z_bias": -0.12,
                "fractional_order": 0.4,
                "normalized_weights": [0.5, 0.3, 0.2],
                "preserved_channels": 3,
                "discarded_channels": 0,
                "dominant_channel": 0
            },
            "coherence": [0.6, 0.3, 0.1]
        }))
        .expect("valid coherence request")
    }

    #[test]
    fn wasm_projection_matches_rust_except_client_metadata() {
        let request = request();
        let expected = serde_json::to_value(
            project_zspace_coherence(request.clone()).expect("valid Rust projection"),
        )
        .expect("serializable Rust projection");
        let mut actual = zspace_coherence_project_value(request).expect("valid WASM projection");
        actual
            .as_object_mut()
            .expect("projection object")
            .remove("execution_client");

        assert_eq!(actual, expected);
    }

    #[test]
    fn wasm_ingress_is_strict_and_accepts_trace_entropy_alias() {
        let parsed = request_from_json(
            r#"{"diagnostics":{"mean_coherence":0.2,"entropy":0.3,"energy_ratio":0.6,"z_bias":0.1,"fractional_order":0.5}}"#,
        )
        .expect("trace alias");
        assert_eq!(parsed.diagnostics.coherence_entropy, 0.3);

        assert!(request_from_json("[]")
            .expect_err("array request must fail")
            .contains("must be an object"));
        assert!(request_from_json(
            r#"{"diagnostics":{"mean_coherence":0.2,"coherence_entropy":0.3,"energy_ratio":0.6,"z_bias":0.1,"fractional_order":0.5},"unknown":true}"#,
        )
        .expect_err("unknown field must fail")
        .contains("unknown field"));
        assert!(request_from_json(
            r#"{"diagnostics":{"mean_coherence":0.2,"coherence_entropy":0.3,"energy_ratio":0.6,"z_bias":0.1,"fractional_order":0.5,"preserved_channels":1},"config":null}"#,
        )
        .expect_err("null config must fail")
        .contains("config' must be an object"));
    }
}
