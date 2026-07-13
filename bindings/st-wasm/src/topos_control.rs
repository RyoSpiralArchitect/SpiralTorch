use serde_json::Value;
use st_tensor::{TensorError, ToposControlSignal, ToposControlSignalInput};

const TOPOS_CONTROL_SIGNAL_INPUT_KEYS: &[&str] = &[
    "curvature",
    "tolerance",
    "saturation",
    "porosity",
    "max_depth",
    "max_volume",
    "observed_depth",
    "visited_volume",
];

fn topos_control_signal_input_from_value(value: Value) -> Result<ToposControlSignalInput, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "Topos control signal input must be an object".to_owned())?;
    if let Some(key) = object
        .keys()
        .find(|key| !TOPOS_CONTROL_SIGNAL_INPUT_KEYS.contains(&key.as_str()))
    {
        return Err(format!("unsupported Topos control signal key: {key}"));
    }
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn topos_control_signal_input_from_json(
    input_json: &str,
) -> Result<ToposControlSignalInput, String> {
    let value = serde_json::from_str(input_json).map_err(|error| error.to_string())?;
    topos_control_signal_input_from_value(value)
}

#[cfg(test)]
use st_tensor::TOPOS_CONTROL_SIGNAL_CONTRACT_VERSION;

#[cfg(target_arch = "wasm32")]
use serde::Serialize;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::js_error;

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

/// Build the shared Topos control-signal payload without introducing WASM semantics.
pub fn topos_control_signal_value(input: ToposControlSignalInput) -> Result<Value, TensorError> {
    let signal = ToposControlSignal::from_input(input)?;
    let mut payload =
        serde_json::to_value(signal.payload()).expect("Topos control payload is serializable");
    let payload_map = payload
        .as_object_mut()
        .expect("Topos control payload is an object");
    payload_map.insert(
        "execution_client".to_owned(),
        Value::String("wasm".to_owned()),
    );
    if let Some(route) = payload_map
        .get_mut("runtime_route")
        .and_then(Value::as_object_mut)
    {
        route.insert(
            "execution_client".to_owned(),
            Value::String("wasm".to_owned()),
        );
    }
    Ok(payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = toposControlSignalJson)]
pub fn topos_control_signal_json(input_json: &str) -> Result<String, JsValue> {
    let input = topos_control_signal_input_from_json(input_json).map_err(js_error)?;
    let payload = topos_control_signal_value(input).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = toposControlSignalObject)]
pub fn topos_control_signal_object(input: &JsValue) -> Result<JsValue, JsValue> {
    let input = serde_wasm_bindgen::from_value::<Value>(input.clone()).map_err(js_error)?;
    let input = topos_control_signal_input_from_value(input).map_err(js_error)?;
    let payload = topos_control_signal_value(input).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wasm_control_signal_uses_st_tensor_contract() {
        let payload = topos_control_signal_value(ToposControlSignalInput {
            max_depth: 10,
            max_volume: 100,
            observed_depth: 4,
            visited_volume: 25,
            ..ToposControlSignalInput::default()
        })
        .expect("valid control signal");

        assert_eq!(payload["kind"], "spiraltorch.topos_control_signal");
        assert_eq!(
            payload["contract_version"],
            TOPOS_CONTROL_SIGNAL_CONTRACT_VERSION
        );
        assert_eq!(payload["semantic_backend"], "rust");
        assert_eq!(payload["execution_client"], "wasm");
        let closure_pressure = payload["closure_pressure"]
            .as_f64()
            .expect("closure pressure is numeric");
        assert!((closure_pressure - 0.4).abs() < 1e-6);
        assert_eq!(payload["runtime_route"]["semantic_backend"], "rust");
    }

    #[test]
    fn wasm_control_signal_matches_rust_payload_except_client_metadata() {
        let input = ToposControlSignalInput {
            porosity: 0.35,
            max_depth: 12,
            max_volume: 80,
            observed_depth: 7,
            visited_volume: 31,
            ..ToposControlSignalInput::default()
        };
        let signal = ToposControlSignal::from_input(input).expect("valid control signal");
        let expected = serde_json::to_value(signal.payload()).expect("serializable Rust payload");
        let mut actual = topos_control_signal_value(input).expect("valid WASM payload");
        let actual_map = actual.as_object_mut().expect("WASM payload object");
        actual_map.remove("execution_client");
        actual_map
            .get_mut("runtime_route")
            .and_then(Value::as_object_mut)
            .expect("runtime route object")
            .remove("execution_client");

        assert_eq!(actual, expected);
    }

    #[test]
    fn wasm_control_signal_rejects_invalid_topology() {
        let result = topos_control_signal_value(ToposControlSignalInput {
            curvature: f32::NAN,
            ..ToposControlSignalInput::default()
        });
        assert!(matches!(result, Err(TensorError::NonFiniteValue { .. })));
    }

    #[test]
    fn wasm_control_signal_ingress_rejects_unknown_keys() {
        let error = topos_control_signal_input_from_json(r#"{"porosity":0.25,"porostiy":0.75}"#)
            .expect_err("unknown Topos control key must fail closed");

        assert!(error.contains("unsupported Topos control signal key: porostiy"));
    }
}
