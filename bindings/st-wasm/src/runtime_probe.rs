use serde_json::Value;
use st_core::backend::runtime_probe::{
    evaluate_runtime_device_probe, observe_runtime_device_probe, RuntimeDeviceProbeError,
    RuntimeDeviceProbeObservationRequest, RuntimeDeviceProbePayload, RuntimeDeviceProbeRequest,
};

#[cfg(target_arch = "wasm32")]
use serde::Serialize;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::js_error;

fn request_from_value(value: Value) -> Result<RuntimeDeviceProbeRequest, String> {
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn request_from_json(request_json: &str) -> Result<RuntimeDeviceProbeRequest, String> {
    let value = serde_json::from_str(request_json).map_err(|error| error.to_string())?;
    request_from_value(value)
}

fn observation_request_from_value(
    value: Value,
) -> Result<RuntimeDeviceProbeObservationRequest, String> {
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn observation_request_from_json(
    request_json: &str,
) -> Result<RuntimeDeviceProbeObservationRequest, String> {
    let value = serde_json::from_str(request_json).map_err(|error| error.to_string())?;
    observation_request_from_value(value)
}

fn payload_from_value(value: Value) -> Result<RuntimeDeviceProbePayload, String> {
    let canonical = value
        .as_object()
        .and_then(|object| object.get("contract"))
        .cloned()
        .unwrap_or(value);
    serde_json::from_value(canonical).map_err(|error| error.to_string())
}

fn payload_from_json(payload_json: &str) -> Result<RuntimeDeviceProbePayload, String> {
    let value = serde_json::from_str(payload_json).map_err(|error| error.to_string())?;
    payload_from_value(value)
}

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

/// Evaluate a browser request through the shared Rust probe contract.
pub fn runtime_device_probe_value(
    request: RuntimeDeviceProbeRequest,
) -> Result<Value, RuntimeDeviceProbeError> {
    let payload = evaluate_runtime_device_probe(request)?.with_execution_client("wasm")?;
    Ok(payload.to_transport_value())
}

/// Observe and resolve a browser device request through the shared Rust owner.
pub fn observe_runtime_device_probe_value(
    request: RuntimeDeviceProbeObservationRequest,
) -> Result<Value, RuntimeDeviceProbeError> {
    let payload = observe_runtime_device_probe(request)?.with_execution_client("wasm")?;
    Ok(payload.to_transport_value())
}

/// Validate a persisted probe contract without re-probing mutable hardware.
pub fn validate_runtime_device_probe_value(
    payload: RuntimeDeviceProbePayload,
    request: Option<RuntimeDeviceProbeRequest>,
) -> Result<Value, RuntimeDeviceProbeError> {
    if let Some(request) = request {
        payload.validate_against(request)?;
    } else {
        payload.validate()?;
    }
    Ok(serde_json::to_value(payload).expect("runtime-device probe payload is serializable"))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = runtimeDeviceProbeJson)]
pub fn runtime_device_probe_json(request_json: &str) -> Result<String, JsValue> {
    let request = request_from_json(request_json).map_err(js_error)?;
    let payload = runtime_device_probe_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = runtimeDeviceProbeObject)]
pub fn runtime_device_probe_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let request = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request = request_from_value(request).map_err(js_error)?;
    let payload = runtime_device_probe_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = runtimeDeviceProbeObserveJson)]
pub fn runtime_device_probe_observe_json(request_json: &str) -> Result<String, JsValue> {
    let request = observation_request_from_json(request_json).map_err(js_error)?;
    let payload = observe_runtime_device_probe_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = runtimeDeviceProbeObserveObject)]
pub fn runtime_device_probe_observe_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let request = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request = observation_request_from_value(request).map_err(js_error)?;
    let payload = observe_runtime_device_probe_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = runtimeDeviceProbeValidateJson)]
pub fn runtime_device_probe_validate_json(payload_json: &str) -> Result<String, JsValue> {
    let payload = payload_from_json(payload_json).map_err(js_error)?;
    let payload = validate_runtime_device_probe_value(payload, None).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = runtimeDeviceProbeValidateObject)]
pub fn runtime_device_probe_validate_object(payload: &JsValue) -> Result<JsValue, JsValue> {
    let payload = serde_wasm_bindgen::from_value::<Value>(payload.clone()).map_err(js_error)?;
    let payload = payload_from_value(payload).map_err(js_error)?;
    let payload = validate_runtime_device_probe_value(payload, None).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = runtimeDeviceProbeValidateAgainstJson)]
pub fn runtime_device_probe_validate_against_json(
    payload_json: &str,
    request_json: &str,
) -> Result<String, JsValue> {
    let payload = payload_from_json(payload_json).map_err(js_error)?;
    let request = request_from_json(request_json).map_err(js_error)?;
    let payload = validate_runtime_device_probe_value(payload, Some(request)).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = runtimeDeviceProbeValidateAgainstObject)]
pub fn runtime_device_probe_validate_against_object(
    payload: &JsValue,
    request: &JsValue,
) -> Result<JsValue, JsValue> {
    let payload = serde_wasm_bindgen::from_value::<Value>(payload.clone()).map_err(js_error)?;
    let payload = payload_from_value(payload).map_err(js_error)?;
    let request = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request = request_from_value(request).map_err(js_error)?;
    let payload = validate_runtime_device_probe_value(payload, Some(request)).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cpu_request() -> RuntimeDeviceProbeRequest {
        request_from_value(json!({
            "requested_backend": "cpu",
            "caps": {
                "backend": "cpu",
                "subgroup": false,
                "lane_width": 1,
                "max_workgroup": 128,
                "shared_mem_per_workgroup": null
            },
            "requested_workgroup": 64,
            "cols": 4096
        }))
        .expect("valid CPU probe request")
    }

    #[test]
    fn wasm_probe_transport_matches_the_rust_contract() {
        let request = cpu_request();
        let rust = serde_json::to_value(
            evaluate_runtime_device_probe(request.clone()).expect("valid Rust probe"),
        )
        .expect("serializable Rust probe");
        let wasm = runtime_device_probe_value(request).expect("valid WASM probe transport");
        assert_eq!(wasm["execution_client"], "wasm");
        assert_eq!(wasm["kind"], "spiraltorch.runtime_device_probe");
        assert_eq!(wasm["runtime_ready"], true);
        assert_eq!(wasm["route_evidence"]["requested_backend"], "cpu");

        let mut canonical = wasm["contract"].clone();
        canonical
            .as_object_mut()
            .expect("canonical probe object")
            .remove("execution_client");
        assert_eq!(canonical, rust);
    }

    #[test]
    fn wasm_observer_delegates_backend_resolution_to_rust() {
        let request = observation_request_from_value(json!({
            "requested_backend": "cpu",
            "caps_overrides": {"max_workgroup": 64},
            "requested_workgroup": 63,
            "cols": 1024
        }))
        .expect("valid unresolved observation request");

        let wasm = observe_runtime_device_probe_value(request)
            .expect("Rust-owned WASM observation succeeds");

        assert_eq!(wasm["execution_client"], "wasm");
        assert_eq!(wasm["requested_backend"], "cpu");
        assert_eq!(wasm["effective_backend"], "cpu");
        assert_eq!(wasm["request"]["caps"]["backend"], "cpu");
        assert_eq!(wasm["request"]["caps"]["max_workgroup"], 64);
        assert_eq!(wasm["requested_runtime"], wasm["effective_runtime"]);
    }

    #[test]
    fn wasm_mps_observer_commits_the_rust_selected_caps() {
        let request = observation_request_from_value(json!({
            "requested_backend": "mps",
            "caps_overrides": {"max_workgroup": 64},
            "requested_workgroup": 63,
            "cols": 1024
        }))
        .expect("valid unresolved MPS observation request");

        let wasm = observe_runtime_device_probe_value(request)
            .expect("Rust-owned MPS observation succeeds");

        assert_eq!(wasm["requested_backend"], "mps");
        assert!(matches!(
            wasm["effective_backend"].as_str(),
            Some("wgpu" | "cpu")
        ));
        assert_eq!(wasm["request"]["caps"]["max_workgroup"], 64);
        assert_eq!(wasm["planner_caps"], wasm["request"]["caps"]);
        assert_eq!(wasm["planner_surrogate_backend"], wasm["effective_backend"]);
    }

    #[test]
    fn wasm_can_validate_and_replay_persisted_probe_contracts() {
        let request = cpu_request();
        let transport = runtime_device_probe_value(request.clone()).expect("valid probe payload");
        let typed = payload_from_value(transport.clone()).expect("typed probe payload");
        let validated = validate_runtime_device_probe_value(typed, Some(request))
            .expect("persisted probe replays");
        assert_eq!(validated, transport["contract"]);

        let mut tampered = transport["contract"].clone();
        tampered["aligned_workgroup"] = 1.into();
        let typed = payload_from_value(tampered).expect("tampered shape still decodes");
        let error = validate_runtime_device_probe_value(typed, None)
            .expect_err("Rust validation must reject probe tampering");
        assert!(matches!(
            error,
            RuntimeDeviceProbeError::InvalidPayload {
                field: "derived_metrics",
                ..
            }
        ));
    }

    #[test]
    fn wasm_probe_ingress_rejects_unknown_fields() {
        let error = request_from_json(
            r#"{
                "requested_backend":"cpu",
                "caps":{
                    "backend":"cpu",
                    "subgroup":false,
                    "lane_width":1,
                    "max_workgroup":128,
                    "shared_mem_per_workgroup":null
                },
                "commander":"browser"
            }"#,
        )
        .expect_err("unknown request fields must fail closed");
        assert!(error.contains("unknown field"));

        let error = observation_request_from_json(
            r#"{"requested_backend":"cpu","effective_backend":"wgpu"}"#,
        )
        .expect_err("browser clients must not select the effective backend");
        assert!(error.contains("unknown field"));
    }
}
