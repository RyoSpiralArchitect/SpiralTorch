use serde_json::Value;
use st_core::backend::execution_plan::{
    validate_tensor_execution_receipt_against_runtime_plan, RuntimeExecutionPlanPayload,
    RuntimeExecutionReceiptValidationError,
};
use st_tensor::{TensorExecutionContractError, TensorExecutionReceipt};

#[cfg(target_arch = "wasm32")]
use serde::Serialize;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::js_error;

fn receipt_from_value(value: Value) -> Result<TensorExecutionReceipt, String> {
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn receipt_from_json(receipt_json: &str) -> Result<TensorExecutionReceipt, String> {
    let value = serde_json::from_str(receipt_json).map_err(|error| error.to_string())?;
    receipt_from_value(value)
}

fn runtime_plan_from_value(value: Value) -> Result<RuntimeExecutionPlanPayload, String> {
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn runtime_plan_from_json(runtime_plan_json: &str) -> Result<RuntimeExecutionPlanPayload, String> {
    let value = serde_json::from_str(runtime_plan_json).map_err(|error| error.to_string())?;
    runtime_plan_from_value(value)
}

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

/// Validate a transported receipt through the shared Rust semantic owner.
pub fn validate_tensor_execution_receipt_value(
    receipt: TensorExecutionReceipt,
) -> Result<Value, TensorExecutionContractError> {
    receipt.validate()?;
    Ok(serde_json::to_value(receipt).expect("tensor execution receipt is serializable"))
}

/// Prove through Rust that one committed runtime plan authorized a receipt.
pub fn validate_tensor_execution_receipt_against_runtime_plan_value(
    receipt: TensorExecutionReceipt,
    runtime_plan: RuntimeExecutionPlanPayload,
) -> Result<Value, RuntimeExecutionReceiptValidationError> {
    validate_tensor_execution_receipt_against_runtime_plan(&runtime_plan, &receipt)?;
    Ok(serde_json::to_value(receipt).expect("tensor execution receipt is serializable"))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = tensorExecutionReceiptValidateJson)]
pub fn tensor_execution_receipt_validate_json(receipt_json: &str) -> Result<String, JsValue> {
    let receipt = receipt_from_json(receipt_json).map_err(js_error)?;
    let receipt = validate_tensor_execution_receipt_value(receipt).map_err(js_error)?;
    serde_json::to_string(&receipt).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = tensorExecutionReceiptValidateObject)]
pub fn tensor_execution_receipt_validate_object(receipt: &JsValue) -> Result<JsValue, JsValue> {
    let receipt = serde_wasm_bindgen::from_value::<Value>(receipt.clone()).map_err(js_error)?;
    let receipt = receipt_from_value(receipt).map_err(js_error)?;
    let receipt = validate_tensor_execution_receipt_value(receipt).map_err(js_error)?;
    to_json_compatible_js(&receipt)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = tensorExecutionReceiptValidateAgainstRuntimePlanJson)]
pub fn tensor_execution_receipt_validate_against_runtime_plan_json(
    receipt_json: &str,
    runtime_plan_json: &str,
) -> Result<String, JsValue> {
    let receipt = receipt_from_json(receipt_json).map_err(js_error)?;
    let runtime_plan = runtime_plan_from_json(runtime_plan_json).map_err(js_error)?;
    let receipt =
        validate_tensor_execution_receipt_against_runtime_plan_value(receipt, runtime_plan)
            .map_err(js_error)?;
    serde_json::to_string(&receipt).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = tensorExecutionReceiptValidateAgainstRuntimePlanObject)]
pub fn tensor_execution_receipt_validate_against_runtime_plan_object(
    receipt: &JsValue,
    runtime_plan: &JsValue,
) -> Result<JsValue, JsValue> {
    let receipt = serde_wasm_bindgen::from_value::<Value>(receipt.clone()).map_err(js_error)?;
    let receipt = receipt_from_value(receipt).map_err(js_error)?;
    let runtime_plan =
        serde_wasm_bindgen::from_value::<Value>(runtime_plan.clone()).map_err(js_error)?;
    let runtime_plan = runtime_plan_from_value(runtime_plan).map_err(js_error)?;
    let receipt =
        validate_tensor_execution_receipt_against_runtime_plan_value(receipt, runtime_plan)
            .map_err(js_error)?;
    to_json_compatible_js(&receipt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use st_core::backend::device_caps::{BackendKind, DeviceCaps};
    use st_core::backend::execution_plan::{
        evaluate_runtime_execution_plan, observe_runtime_execution_plan_capabilities,
        AcceleratorFallback, ExecutionConfig, RuntimeComponentResolution, RuntimeComponentWorkload,
        RuntimeExecutionPlanRequest,
    };
    use st_core::backend::runtime_probe::{
        evaluate_runtime_device_probe, RuntimeDeviceProbeRequest,
    };

    fn direct_cpu_receipt() -> TensorExecutionReceipt {
        serde_json::from_value(json!({
            "kind": "spiraltorch.tensor_execution_receipt",
            "contract_version": "spiraltorch.tensor_execution_receipt.v1",
            "semantic_owner": "st-tensor::execution",
            "component": "softmax",
            "operation": "row_softmax",
            "workload": {"component": "softmax", "rows": 2, "cols": 3},
            "requested_backend": "cpu",
            "selected_backend": "cpu",
            "executed_backend": "cpu",
            "kernel_backend": "cpu",
            "route_status": "direct"
        }))
        .expect("typed receipt")
    }

    fn cpu_softmax_plan(threshold: usize) -> RuntimeExecutionPlanPayload {
        let runtime_probe = evaluate_runtime_device_probe(RuntimeDeviceProbeRequest {
            requested_backend: BackendKind::Cpu,
            caps: DeviceCaps::cpu(),
            mps_probe: None,
            requested_workgroup: None,
            cols: None,
            tile_hint: None,
            compaction_hint: None,
        })
        .expect("CPU runtime probe");
        let request = observe_runtime_execution_plan_capabilities(RuntimeExecutionPlanRequest {
            runtime_probe,
            execution_config: ExecutionConfig::new(AcceleratorFallback::Allow, threshold),
            component_resolution: RuntimeComponentResolution::Deferred,
            component_workloads: vec![RuntimeComponentWorkload::Softmax { rows: 2, cols: 3 }],
            component_capability_observation: None,
            tensor_util_values: None,
            required_native_components: Vec::new(),
        })
        .expect("CPU softmax capability observation");
        evaluate_runtime_execution_plan(request).expect("CPU softmax execution plan")
    }

    #[test]
    fn wasm_transport_preserves_the_rust_receipt_contract() {
        let receipt = direct_cpu_receipt();
        let rust = serde_json::to_value(&receipt).unwrap();
        let transported = validate_tensor_execution_receipt_value(receipt).unwrap();

        assert_eq!(transported, rust);
        assert_eq!(
            transported["contract_version"],
            "spiraltorch.tensor_execution_receipt.v1"
        );
        assert_eq!(transported["semantic_owner"], "st-tensor::execution");
    }

    #[test]
    fn wasm_transport_rejects_route_reconstruction() {
        let mut receipt = direct_cpu_receipt();
        receipt.executed_backend = Some(st_tensor::TensorExecutionBackend::Wgpu);

        assert!(validate_tensor_execution_receipt_value(receipt).is_err());
    }

    #[test]
    fn wasm_transport_rejects_an_unimplemented_component_backend_pair() {
        let mut receipt = direct_cpu_receipt();
        receipt.requested_backend = st_tensor::TensorExecutionBackend::Faer;
        receipt.selected_backend = st_tensor::TensorExecutionBackend::Faer;
        receipt.executed_backend = Some(st_tensor::TensorExecutionBackend::Faer);

        assert!(validate_tensor_execution_receipt_value(receipt).is_err());
    }

    #[test]
    fn wasm_transport_rejects_a_reconstructed_kernel_backend() {
        let mut receipt = direct_cpu_receipt();
        receipt.kernel_backend = Some(st_tensor::TensorExecutionKernelBackend::WgpuDense);

        assert!(validate_tensor_execution_receipt_value(receipt).is_err());
    }

    #[test]
    fn wasm_transport_delegates_plan_authorization_to_rust() {
        let plan = cpu_softmax_plan(1024);
        let mut receipt = direct_cpu_receipt();
        receipt.runtime_execution_plan_output_sha256 = Some(plan.output_sha256.clone());
        let rust = serde_json::to_value(&receipt).unwrap();

        let transported = validate_tensor_execution_receipt_against_runtime_plan_value(
            receipt.clone(),
            plan.clone(),
        )
        .unwrap();
        assert_eq!(transported, rust);

        let foreign_plan = cpu_softmax_plan(2048);
        assert_ne!(foreign_plan.output_sha256, plan.output_sha256);
        assert!(
            validate_tensor_execution_receipt_against_runtime_plan_value(
                receipt.clone(),
                foreign_plan,
            )
            .is_err()
        );

        receipt.workload = st_tensor::TensorExecutionWorkload::Softmax { rows: 2, cols: 5 };
        receipt.validate().unwrap();
        assert!(
            validate_tensor_execution_receipt_against_runtime_plan_value(receipt, plan).is_err()
        );
    }
}
