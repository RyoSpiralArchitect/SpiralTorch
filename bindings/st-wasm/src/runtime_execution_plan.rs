use serde_json::Value;
use st_core::backend::execution_plan::{
    evaluate_runtime_execution_plan, RuntimeExecutionPlanError, RuntimeExecutionPlanPayload,
    RuntimeExecutionPlanRequest,
};

#[cfg(target_arch = "wasm32")]
use serde::Serialize;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::js_error;

fn request_from_value(mut value: Value) -> Result<RuntimeExecutionPlanRequest, String> {
    if let Some(runtime_probe) = value
        .as_object_mut()
        .and_then(|request| request.get_mut("runtime_probe"))
    {
        if let Some(contract) = runtime_probe
            .as_object()
            .and_then(|transport| transport.get("contract"))
            .cloned()
        {
            if !contract.is_object() {
                return Err("runtime probe contract must be an object".to_owned());
            }
            *runtime_probe = contract;
        }
    }
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn request_from_json(request_json: &str) -> Result<RuntimeExecutionPlanRequest, String> {
    let value = serde_json::from_str(request_json).map_err(|error| error.to_string())?;
    request_from_value(value)
}

fn payload_from_value(value: Value) -> Result<RuntimeExecutionPlanPayload, String> {
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn payload_from_json(payload_json: &str) -> Result<RuntimeExecutionPlanPayload, String> {
    let value = serde_json::from_str(payload_json).map_err(|error| error.to_string())?;
    payload_from_value(value)
}

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

/// Evaluate a browser-supplied request through the shared Rust execution-plan contract.
pub fn runtime_execution_plan_value(
    request: RuntimeExecutionPlanRequest,
) -> Result<Value, RuntimeExecutionPlanError> {
    let payload = evaluate_runtime_execution_plan(request)?.with_execution_client("wasm")?;
    Ok(serde_json::to_value(payload).expect("runtime execution-plan payload is serializable"))
}

/// Validate a persisted execution-plan contract through the same Rust semantic owner.
pub fn validate_runtime_execution_plan_value(
    payload: RuntimeExecutionPlanPayload,
    request: Option<RuntimeExecutionPlanRequest>,
) -> Result<Value, RuntimeExecutionPlanError> {
    if let Some(request) = request {
        payload.validate_against(request)?;
    } else {
        payload.validate()?;
    }
    Ok(serde_json::to_value(payload).expect("runtime execution-plan payload is serializable"))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = runtimeExecutionPlanJson)]
pub fn runtime_execution_plan_json(request_json: &str) -> Result<String, JsValue> {
    let request = request_from_json(request_json).map_err(js_error)?;
    let payload = runtime_execution_plan_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = runtimeExecutionPlanObject)]
pub fn runtime_execution_plan_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let request = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request = request_from_value(request).map_err(js_error)?;
    let payload = runtime_execution_plan_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = runtimeExecutionPlanValidateJson)]
pub fn runtime_execution_plan_validate_json(payload_json: &str) -> Result<String, JsValue> {
    let payload = payload_from_json(payload_json).map_err(js_error)?;
    let payload = validate_runtime_execution_plan_value(payload, None).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = runtimeExecutionPlanValidateObject)]
pub fn runtime_execution_plan_validate_object(payload: &JsValue) -> Result<JsValue, JsValue> {
    let payload = serde_wasm_bindgen::from_value::<Value>(payload.clone()).map_err(js_error)?;
    let payload = payload_from_value(payload).map_err(js_error)?;
    let payload = validate_runtime_execution_plan_value(payload, None).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = runtimeExecutionPlanValidateAgainstJson)]
pub fn runtime_execution_plan_validate_against_json(
    payload_json: &str,
    request_json: &str,
) -> Result<String, JsValue> {
    let payload = payload_from_json(payload_json).map_err(js_error)?;
    let request = request_from_json(request_json).map_err(js_error)?;
    let payload =
        validate_runtime_execution_plan_value(payload, Some(request)).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = runtimeExecutionPlanValidateAgainstObject)]
pub fn runtime_execution_plan_validate_against_object(
    payload: &JsValue,
    request: &JsValue,
) -> Result<JsValue, JsValue> {
    let payload = serde_wasm_bindgen::from_value::<Value>(payload.clone()).map_err(js_error)?;
    let payload = payload_from_value(payload).map_err(js_error)?;
    let request = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request = request_from_value(request).map_err(js_error)?;
    let payload =
        validate_runtime_execution_plan_value(payload, Some(request)).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use st_core::backend::device_caps::{BackendKind, DeviceCaps};
    use st_core::backend::execution_plan::{
        AcceleratorFallback, ExecutionConfig, RuntimeExecutionComponent,
    };
    use st_core::backend::runtime_probe::{
        evaluate_runtime_device_probe, RuntimeDeviceProbeRequest,
    };

    fn cpu_probe_request() -> RuntimeDeviceProbeRequest {
        RuntimeDeviceProbeRequest {
            requested_backend: BackendKind::Cpu,
            caps: DeviceCaps::cpu(),
            mps_probe: None,
            requested_workgroup: None,
            cols: None,
            tile_hint: None,
            compaction_hint: None,
        }
    }

    fn cpu_request() -> RuntimeExecutionPlanRequest {
        let probe = evaluate_runtime_device_probe(cpu_probe_request()).expect("CPU probe");
        RuntimeExecutionPlanRequest {
            runtime_probe: probe,
            execution_config: ExecutionConfig::new(AcceleratorFallback::Allow, 1024),
            tensor_util_values: Some(2048),
            required_native_components: vec![RuntimeExecutionComponent::DenseMatmul],
        }
    }

    fn without_client(mut payload: Value) -> Value {
        payload
            .as_object_mut()
            .expect("runtime execution-plan payload object")
            .remove("execution_client");
        payload
    }

    #[test]
    fn wasm_execution_plan_matches_the_rust_contract() {
        let request = cpu_request();
        let rust = serde_json::to_value(
            evaluate_runtime_execution_plan(request.clone()).expect("Rust execution plan"),
        )
        .expect("serializable Rust plan");
        let wasm_transport =
            runtime_execution_plan_value(request).expect("WASM execution-plan transport");

        assert_eq!(wasm_transport["execution_client"], "wasm");
        assert_eq!(without_client(wasm_transport.clone()), rust);
        assert_eq!(
            wasm_transport["contract_version"],
            "spiraltorch.runtime_execution_plan.v1"
        );
        assert_eq!(wasm_transport["execution_allowed"], true);
        assert_eq!(wasm_transport["all_components_native"], true);
        assert_eq!(wasm_transport["policy"]["dense_matmul"], "faer");
        assert_eq!(
            wasm_transport["runtime_route"]["execution_client"],
            Value::Null
        );
        assert_eq!(wasm_transport["committed"], true);
        assert_eq!(wasm_transport["request_sha256"].as_str().unwrap().len(), 64);
        assert_eq!(wasm_transport["output_sha256"].as_str().unwrap().len(), 64);
    }

    #[test]
    fn wasm_execution_plan_accepts_the_runtime_probe_transport() {
        let probe_transport = crate::runtime_probe::runtime_device_probe_value(cpu_probe_request())
            .expect("WASM runtime probe transport");
        let mut request_value =
            serde_json::to_value(cpu_request()).expect("serializable execution-plan request");
        request_value["runtime_probe"] = probe_transport;

        let from_object = request_from_value(request_value.clone()).expect("object request");
        let request_json = serde_json::to_string(&request_value).expect("request JSON");
        let from_json = request_from_json(&request_json).expect("JSON request");
        assert_eq!(from_object, from_json);

        let payload =
            runtime_execution_plan_value(from_object).expect("plan accepts probe transport");
        assert_eq!(payload["execution_allowed"], true);
        assert_eq!(
            payload["request"]["runtime_probe"]["execution_client"],
            Value::Null
        );
        assert_eq!(payload["runtime_route"]["execution_client"], Value::Null);

        let mut malformed = request_value;
        malformed["runtime_probe"]["contract"] = "not-an-object".into();
        let error = request_from_value(malformed).expect_err("invalid contract must fail closed");
        assert!(error.contains("runtime probe contract must be an object"));
    }

    #[test]
    fn wasm_execution_plan_ingress_and_replay_fail_closed() {
        let request = cpu_request();
        let request_json = serde_json::to_string(&request).expect("request JSON");
        let parsed = request_from_json(&request_json).expect("typed request");
        let payload = runtime_execution_plan_value(parsed).expect("WASM execution plan");
        let typed = payload_from_value(payload.clone()).expect("typed payload");
        let validated = validate_runtime_execution_plan_value(typed, Some(request))
            .expect("persisted payload replays");
        assert_eq!(validated, payload);

        let unknown = request_from_json(
            &serde_json::to_string(&json!({
                "runtime_probe": payload["request"]["runtime_probe"],
                "execution_config": {
                    "accelerator_fallback": "allow",
                    "tensor_util_wgpu_min_values": 1024
                },
                "commander": "python"
            }))
            .expect("unknown-field JSON"),
        )
        .expect_err("unknown request fields must fail closed");
        assert!(unknown.contains("unknown field"));

        let mut tampered = payload;
        tampered["policy"]["softmax"] = "auto".into();
        let typed = payload_from_value(tampered).expect("tampered shape decodes");
        let error = validate_runtime_execution_plan_value(typed, None)
            .expect_err("Rust replay rejects tampering");
        assert!(matches!(
            error,
            RuntimeExecutionPlanError::InvalidPayload {
                field: "payload",
                ..
            }
        ));
    }
}
