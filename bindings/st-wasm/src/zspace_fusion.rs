use serde_json::Value;
use st_core::telemetry::zspace_fusion::{
    fuse_zspace_partials, fuse_zspace_telemetry, project_zspace_metric_gradient, ZSpaceFusionError,
    ZSpaceMetricGradientProjectionRequest, ZSpacePartialFusionRequest,
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
        .expect("Z-space fusion payload is an object");
    payload.insert(
        "execution_client".to_owned(),
        Value::String("wasm".to_owned()),
    );
    if let Some(telemetry) = payload.get_mut("telemetry").and_then(Value::as_object_mut) {
        telemetry.insert(
            "execution_client".to_owned(),
            Value::String("wasm".to_owned()),
        );
    }
}

fn telemetry_inputs_from_value(value: Value) -> Result<Vec<Value>, String> {
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn telemetry_inputs_from_json(input_json: &str) -> Result<Vec<Value>, String> {
    let value = serde_json::from_str(input_json).map_err(|error| error.to_string())?;
    telemetry_inputs_from_value(value)
}

fn partial_fusion_request_from_value(value: Value) -> Result<ZSpacePartialFusionRequest, String> {
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn partial_fusion_request_from_json(
    request_json: &str,
) -> Result<ZSpacePartialFusionRequest, String> {
    let value = serde_json::from_str(request_json).map_err(|error| error.to_string())?;
    partial_fusion_request_from_value(value)
}

fn metric_gradient_request_from_value(
    value: Value,
) -> Result<ZSpaceMetricGradientProjectionRequest, String> {
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn metric_gradient_request_from_json(
    request_json: &str,
) -> Result<ZSpaceMetricGradientProjectionRequest, String> {
    let value = serde_json::from_str(request_json).map_err(|error| error.to_string())?;
    metric_gradient_request_from_value(value)
}

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

/// Fuse telemetry through the shared Rust contract and label WASM as the client.
pub fn zspace_telemetry_fusion_value(payloads: &[Value]) -> Result<Value, ZSpaceFusionError> {
    let mut payload = serde_json::to_value(fuse_zspace_telemetry(payloads)?)
        .expect("Z-space telemetry fusion payload is serializable");
    add_execution_client(&mut payload);
    Ok(payload)
}

/// Fuse Z-space partials through the shared Rust contract and label WASM as the client.
pub fn zspace_partial_fusion_value(
    request: ZSpacePartialFusionRequest,
) -> Result<Value, ZSpaceFusionError> {
    let mut payload = serde_json::to_value(fuse_zspace_partials(request)?)
        .expect("Z-space partial fusion payload is serializable");
    add_execution_client(&mut payload);
    Ok(payload)
}

/// Project canonical metrics through the shared Rust basis and label WASM as the client.
pub fn zspace_metric_gradient_projection_value(
    request: ZSpaceMetricGradientProjectionRequest,
) -> Result<Value, ZSpaceFusionError> {
    let mut payload = serde_json::to_value(project_zspace_metric_gradient(request)?)
        .expect("Z-space metric-gradient payload is serializable");
    add_execution_client(&mut payload);
    Ok(payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceTelemetryFusionJson)]
pub fn zspace_telemetry_fusion_json(payloads_json: &str) -> Result<String, JsValue> {
    let payloads = telemetry_inputs_from_json(payloads_json).map_err(js_error)?;
    let payload = zspace_telemetry_fusion_value(&payloads).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceTelemetryFusionObject)]
pub fn zspace_telemetry_fusion_object(payloads: &JsValue) -> Result<JsValue, JsValue> {
    let payloads = serde_wasm_bindgen::from_value::<Value>(payloads.clone()).map_err(js_error)?;
    let payloads = telemetry_inputs_from_value(payloads).map_err(js_error)?;
    let payload = zspace_telemetry_fusion_value(&payloads).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspacePartialFusionJson)]
pub fn zspace_partial_fusion_json(request_json: &str) -> Result<String, JsValue> {
    let request = partial_fusion_request_from_json(request_json).map_err(js_error)?;
    let payload = zspace_partial_fusion_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspacePartialFusionObject)]
pub fn zspace_partial_fusion_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let request = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request = partial_fusion_request_from_value(request).map_err(js_error)?;
    let payload = zspace_partial_fusion_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceMetricGradientProjectionJson)]
pub fn zspace_metric_gradient_projection_json(request_json: &str) -> Result<String, JsValue> {
    let request = metric_gradient_request_from_json(request_json).map_err(js_error)?;
    let payload = zspace_metric_gradient_projection_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceMetricGradientProjectionObject)]
pub fn zspace_metric_gradient_projection_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let request = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request = metric_gradient_request_from_value(request).map_err(js_error)?;
    let payload = zspace_metric_gradient_projection_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn wasm_telemetry_fusion_matches_rust_except_client_metadata() {
        let inputs = vec![
            json!({"psi": {"energy": 2.0, "ready": true}}),
            json!({"psi.energy": "4.0", "ignored": [1, 2]}),
        ];
        let expected = serde_json::to_value(
            fuse_zspace_telemetry(&inputs).expect("valid Rust telemetry fusion"),
        )
        .expect("serializable Rust telemetry fusion");
        let mut actual =
            zspace_telemetry_fusion_value(&inputs).expect("valid WASM telemetry fusion");
        actual
            .as_object_mut()
            .expect("WASM telemetry payload object")
            .remove("execution_client");

        assert_eq!(actual, expected);
    }

    #[test]
    fn wasm_partial_fusion_matches_rust_except_client_metadata() {
        let request: ZSpacePartialFusionRequest = serde_json::from_value(json!({
            "partials": [
                {
                    "metrics": {"velocity": 1.0, "grad": [1.0, 2.0]},
                    "weight": 1.0,
                    "origin": "wasm-a",
                    "gradient_basis": "test.basis.v1",
                    "telemetry": {"psi": {"energy": 0.25}}
                },
                {
                    "metrics": {"speed": 5.0, "gradient": [5.0]},
                    "weight": 3.0,
                    "origin": "wasm-b",
                    "gradient_basis": "test.basis.v1"
                }
            ],
            "strategy": "mean",
            "gradient_alignment": "pad_zero",
            "telemetry": [{"browser": {"webgpu_ready": true}}]
        }))
        .expect("valid partial request");
        let expected = serde_json::to_value(
            fuse_zspace_partials(request.clone()).expect("valid Rust partial fusion"),
        )
        .expect("serializable Rust partial fusion");
        let mut actual = zspace_partial_fusion_value(request).expect("valid WASM partial fusion");
        let actual = actual.as_object_mut().expect("WASM partial payload object");
        actual.remove("execution_client");
        actual
            .get_mut("telemetry")
            .and_then(Value::as_object_mut)
            .expect("nested telemetry payload object")
            .remove("execution_client");

        assert_eq!(Value::Object(actual.clone()), expected);
    }

    #[test]
    fn wasm_metric_gradient_projection_matches_rust_except_client_metadata() {
        let request = metric_gradient_request_from_value(json!({
            "metrics": {
                "speed": 0.1,
                "memory": 0.2,
                "stability": 0.3,
                "frac": 0.4,
                "drs": -0.5
            },
            "dimension": 7
        }))
        .expect("valid metric-gradient request");
        let expected = serde_json::to_value(
            project_zspace_metric_gradient(request.clone()).expect("valid Rust projection"),
        )
        .expect("serializable Rust projection");
        let mut actual = zspace_metric_gradient_projection_value(request)
            .expect("valid WASM metric-gradient projection");
        actual
            .as_object_mut()
            .expect("WASM metric-gradient payload object")
            .remove("execution_client");

        assert_eq!(actual, expected);
    }

    #[test]
    fn wasm_partial_fusion_transports_post_fusion_metric_projection() {
        let request = partial_fusion_request_from_value(json!({
            "partials": [
                {
                    "metrics": {
                        "speed": 1.0,
                        "memory": 2.0,
                        "stability": 3.0,
                        "frac": 4.0,
                        "drs": 5.0,
                        "gradient": [99.0, 98.0]
                    },
                    "gradient_basis": "browser.features.v1"
                },
                {
                    "metrics": {
                        "speed": 3.0,
                        "memory": 4.0,
                        "stability": 5.0,
                        "frac": 6.0,
                        "drs": 7.0,
                        "gradient": [-9.0]
                    },
                    "gradient_basis": "native.features.v1"
                }
            ],
            "metric_gradient_dimension": 4
        }))
        .expect("valid post-fusion projection request");
        let expected =
            serde_json::to_value(fuse_zspace_partials(request.clone()).expect("valid Rust fusion"))
                .expect("serializable Rust fusion");
        let mut actual = zspace_partial_fusion_value(request).expect("valid WASM fusion");
        let actual = actual
            .as_object_mut()
            .expect("WASM partial-fusion payload object");
        actual.remove("execution_client");
        actual
            .get_mut("telemetry")
            .and_then(Value::as_object_mut)
            .expect("nested telemetry payload object")
            .remove("execution_client");

        assert_eq!(Value::Object(actual.clone()), expected);
        assert_eq!(actual["gradient_source"], "canonical_metrics");
        assert_eq!(actual["gradient_replaced_source_count"], 2);
    }

    #[test]
    fn wasm_partial_ingress_fails_closed_on_contract_drift() {
        let unknown = partial_fusion_request_from_json(r#"{"partials":[],"stratgey":"mean"}"#)
            .expect_err("unknown request fields must fail closed");
        assert!(unknown.contains("unknown field"));

        let invalid_strategy =
            partial_fusion_request_from_json(r#"{"partials":[],"strategy":"product"}"#)
                .expect_err("unknown strategies must fail closed");
        assert!(invalid_strategy.contains("unknown variant"));

        let invalid_alignment =
            partial_fusion_request_from_json(r#"{"partials":[],"gradient_alignment":"truncate"}"#)
                .expect_err("unknown gradient alignment must fail closed");
        assert!(invalid_alignment.contains("unknown variant"));

        let invalid_projection =
            metric_gradient_request_from_json(r#"{"metrics":{},"dimension":4,"dim":4}"#)
                .expect_err("unknown metric-gradient fields must fail closed");
        assert!(invalid_projection.contains("unknown field"));
    }

    #[test]
    fn wasm_partial_fusion_inherits_strict_gradient_alignment() {
        let request = partial_fusion_request_from_json(
            r#"{
                "partials": [
                    {"metrics": {"gradient": [1.0, 2.0]}},
                    {"metrics": {"gradient": [3.0]}}
                ]
            }"#,
        )
        .expect("defaulted request is valid");
        let error = zspace_partial_fusion_value(request)
            .expect_err("WASM must inherit Rust's strict default");
        assert!(matches!(
            error,
            ZSpaceFusionError::GradientDimensionMismatch { index: 1, .. }
        ));
    }

    #[test]
    fn wasm_telemetry_ingress_requires_an_array_of_objects() {
        let not_an_array = telemetry_inputs_from_json(r#"{"psi":1.0}"#)
            .expect_err("telemetry collection must be an array");
        assert!(not_an_array.contains("sequence"));

        let not_an_object = telemetry_inputs_from_json(r#"[1.0]"#)
            .expect("shape validation belongs to the Rust fusion contract");
        let error = zspace_telemetry_fusion_value(&not_an_object)
            .expect_err("telemetry entries must be objects");
        assert!(matches!(
            error,
            ZSpaceFusionError::TelemetryNotObject { index: 0 }
        ));
    }
}
