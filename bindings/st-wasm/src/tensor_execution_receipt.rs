use serde_json::Value;
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
            "route_status": "direct"
        }))
        .expect("typed receipt")
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
}
