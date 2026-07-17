use serde_json::Value;
use st_core::backend::runtime_route::{
    evaluate_runtime_device_route, RuntimeDeviceRouteError, RuntimeDeviceRoutePayload,
    RuntimeDeviceRouteRequest,
};

#[cfg(target_arch = "wasm32")]
use serde::Serialize;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::js_error;

fn request_from_value(value: Value) -> Result<RuntimeDeviceRouteRequest, String> {
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn request_from_json(request_json: &str) -> Result<RuntimeDeviceRouteRequest, String> {
    let value = serde_json::from_str(request_json).map_err(|error| error.to_string())?;
    request_from_value(value)
}

fn payload_from_value(value: Value) -> Result<RuntimeDeviceRoutePayload, String> {
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn payload_from_json(payload_json: &str) -> Result<RuntimeDeviceRoutePayload, String> {
    let value = serde_json::from_str(payload_json).map_err(|error| error.to_string())?;
    payload_from_value(value)
}

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

/// Evaluate browser-observed device rows through the shared Rust contract.
pub fn runtime_device_route_value(
    request: RuntimeDeviceRouteRequest,
) -> Result<Value, RuntimeDeviceRouteError> {
    let payload = evaluate_runtime_device_route(request)?.with_execution_client("wasm")?;
    Ok(serde_json::to_value(payload).expect("runtime-device route payload is serializable"))
}

/// Validate a persisted route contract through the same Rust semantic owner.
pub fn validate_runtime_device_route_value(
    payload: RuntimeDeviceRoutePayload,
    request: Option<RuntimeDeviceRouteRequest>,
) -> Result<Value, RuntimeDeviceRouteError> {
    if let Some(request) = request {
        payload.validate_against(request)?;
    } else {
        payload.validate()?;
    }
    Ok(serde_json::to_value(payload).expect("runtime-device route payload is serializable"))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = runtimeDeviceRouteJson)]
pub fn runtime_device_route_json(request_json: &str) -> Result<String, JsValue> {
    let request = request_from_json(request_json).map_err(js_error)?;
    let payload = runtime_device_route_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = runtimeDeviceRouteObject)]
pub fn runtime_device_route_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let request = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request = request_from_value(request).map_err(js_error)?;
    let payload = runtime_device_route_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = runtimeDeviceRouteValidateJson)]
pub fn runtime_device_route_validate_json(payload_json: &str) -> Result<String, JsValue> {
    let payload = payload_from_json(payload_json).map_err(js_error)?;
    let payload = validate_runtime_device_route_value(payload, None).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = runtimeDeviceRouteValidateObject)]
pub fn runtime_device_route_validate_object(payload: &JsValue) -> Result<JsValue, JsValue> {
    let payload = serde_wasm_bindgen::from_value::<Value>(payload.clone()).map_err(js_error)?;
    let payload = payload_from_value(payload).map_err(js_error)?;
    let payload = validate_runtime_device_route_value(payload, None).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = runtimeDeviceRouteValidateAgainstJson)]
pub fn runtime_device_route_validate_against_json(
    payload_json: &str,
    request_json: &str,
) -> Result<String, JsValue> {
    let payload = payload_from_json(payload_json).map_err(js_error)?;
    let request = request_from_json(request_json).map_err(js_error)?;
    let payload = validate_runtime_device_route_value(payload, Some(request)).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = runtimeDeviceRouteValidateAgainstObject)]
pub fn runtime_device_route_validate_against_object(
    payload: &JsValue,
    request: &JsValue,
) -> Result<JsValue, JsValue> {
    let payload = serde_wasm_bindgen::from_value::<Value>(payload.clone()).map_err(js_error)?;
    let payload = payload_from_value(payload).map_err(js_error)?;
    let request = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request = request_from_value(request).map_err(js_error)?;
    let payload = validate_runtime_device_route_value(payload, Some(request)).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn without_client(mut payload: Value) -> Value {
        payload
            .as_object_mut()
            .expect("runtime-device route payload object")
            .remove("execution_client");
        payload
    }

    #[test]
    fn wasm_runtime_device_route_matches_the_rust_contract() {
        let request: RuntimeDeviceRouteRequest = serde_json::from_value(json!({
            "reports": [
                {
                    "requested_backend": "mps",
                    "effective_backend": "wgpu",
                    "runtime_ready": true,
                    "requested_backend_runtime_ready": false,
                    "effective_backend_runtime_ready": true,
                    "runtime_status": "kernel_wired",
                    "requested_backend_runtime_status": "placeholder",
                    "effective_backend_runtime_status": "kernel_wired",
                    "error": "native MPS kernels are not wired"
                }
            ],
            "requested_backends": ["mps"],
            "required_available_backends": ["mps"],
            "required_ready_backends": ["mps"]
        }))
        .expect("valid runtime-device route request");
        let rust = serde_json::to_value(
            evaluate_runtime_device_route(request.clone()).expect("valid Rust route"),
        )
        .expect("serializable Rust route");
        let wasm_transport =
            runtime_device_route_value(request).expect("valid WASM route transport");
        assert_eq!(wasm_transport["execution_client"], "wasm");
        let wasm = without_client(wasm_transport);

        assert_eq!(wasm, rust);
        assert_eq!(
            wasm["contract_version"],
            "spiraltorch.runtime_device_route.v4"
        );
        assert_eq!(wasm["committed"], true);
        assert_eq!(wasm["request_sha256"].as_str().unwrap().len(), 64);
        assert_eq!(wasm["output_sha256"].as_str().unwrap().len(), 64);
        assert_eq!(wasm["evidence"][0]["requested_backend"], "mps");
        assert_eq!(wasm["routes"][0]["route"], "surrogate");
        assert_eq!(wasm["routes"][0]["native_readiness"], "not_ready");
        assert_eq!(wasm["routes"][0]["native_ready"], false);
        assert_eq!(wasm["routes"][0]["route_readiness"], "ready");
        assert_eq!(wasm["routes"][0]["route_ready"], true);
        assert_eq!(wasm["runtime_readiness"], "ready");
        assert_eq!(wasm["runtime_ready"], true);
        assert_eq!(wasm["runtime_ready_basis"], "required_ready_backends");
        assert_eq!(wasm["runtime_missing_ready_backends"], json!([]));
        assert_eq!(wasm["passed"], true);
    }

    #[test]
    fn wasm_transport_preserves_unknown_readiness() {
        let request = request_from_json(
            r#"{
                "reports":[{"requested_backend":"cpu","runtime_status":"cpu"}],
                "requested_backends":["cpu"],
                "required_ready_backends":["cpu"]
            }"#,
        )
        .expect("valid unknown-readiness request");
        let payload = runtime_device_route_value(request).expect("valid WASM route transport");

        assert_eq!(payload["routes"][0]["native_readiness"], "unknown");
        assert_eq!(payload["routes"][0]["native_ready"], Value::Null);
        assert_eq!(payload["routes"][0]["route_readiness"], "unknown");
        assert_eq!(payload["routes"][0]["route_ready"], false);
        assert_eq!(payload["routes"][0]["route_status"], "unknown");
        assert_eq!(payload["route_not_ready_backends"], json!([]));
        assert_eq!(payload["route_readiness_unknown_backends"], json!(["cpu"]));
        assert_eq!(payload["required_ready_backends_unknown"], json!(["cpu"]));
        assert_eq!(payload["required_ready_backends_passed"], false);
        assert_eq!(payload["runtime_readiness"], "unknown");
        assert_eq!(payload["runtime_ready"], false);
        assert_eq!(payload["runtime_unknown_ready_backends"], json!(["cpu"]));
        assert_eq!(
            payload["failures"],
            json!(["runtime_device_readiness_unknown:cpu"])
        );
    }

    #[test]
    fn wasm_ingress_and_rust_semantics_fail_closed() {
        let unknown = request_from_json(r#"{"reports":[],"commander":"python"}"#)
            .expect_err("unknown request fields must fail closed");
        assert!(unknown.contains("unknown field"));

        let drifted = request_from_json(
            r#"{
                "reports":[{
                    "requested_backend":"wgpu",
                    "runtime_ready":true,
                    "effective_backend_runtime_ready":false
                }]
            }"#,
        )
        .expect("readiness drift is a semantic validation error");
        let error = runtime_device_route_value(drifted)
            .expect_err("shared Rust evaluator must reject readiness drift");
        assert!(matches!(
            error,
            RuntimeDeviceRouteError::ConflictingRouteReadiness { index: 0, .. }
        ));
    }

    #[test]
    fn wasm_can_validate_and_replay_persisted_route_contracts() {
        let request = request_from_json(
            r#"{
                "reports":[{
                    "requested_backend":"wgpu",
                    "runtime_ready":true,
                    "runtime_status":"kernel_wired"
                }],
                "requested_backends":["wgpu"],
                "required_ready_backends":["wgpu"]
            }"#,
        )
        .expect("valid route request");
        let payload = runtime_device_route_value(request.clone()).expect("valid route payload");
        let typed = payload_from_value(payload.clone()).expect("typed route payload");
        let validated = validate_runtime_device_route_value(typed, Some(request))
            .expect("persisted payload replays");
        assert_eq!(validated, payload);

        let mut tampered = payload;
        tampered["routes"][0]["route_ready"] = false.into();
        let typed = payload_from_value(tampered).expect("tampered shape still decodes");
        let error = validate_runtime_device_route_value(typed, None)
            .expect_err("Rust validation must reject route tampering");
        assert!(matches!(
            error,
            RuntimeDeviceRouteError::InvalidPayload {
                field: "payload",
                ..
            }
        ));
    }
}
