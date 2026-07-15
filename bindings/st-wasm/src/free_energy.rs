use serde_json::Value;
use st_core::heur::free_energy::{evaluate_free_energy, FreeEnergyError, FreeEnergyRequest};

#[cfg(target_arch = "wasm32")]
use serde::Serialize;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::js_error;

fn add_execution_client(payload: &mut Value) {
    payload
        .as_object_mut()
        .expect("variational free-energy payload is an object")
        .insert(
            "execution_client".to_owned(),
            Value::String("wasm".to_owned()),
        );
}

fn request_from_value(value: Value) -> Result<FreeEnergyRequest, String> {
    if !value.is_object() {
        return Err("variational free-energy request must be an object".to_string());
    }
    serde_json::from_value(value)
        .map_err(|error| format!("invalid variational free-energy request: {error}"))
}

fn request_from_json(request_json: &str) -> Result<FreeEnergyRequest, String> {
    let value: Value = serde_json::from_str(request_json)
        .map_err(|error| format!("invalid variational free-energy JSON: {error}"))?;
    request_from_value(value)
}

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

pub fn zspace_free_energy_value(request: FreeEnergyRequest) -> Result<Value, FreeEnergyError> {
    let mut payload = serde_json::to_value(evaluate_free_energy(request)?)
        .expect("variational free-energy payload is serializable");
    add_execution_client(&mut payload);
    Ok(payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceFreeEnergyJson)]
pub fn zspace_free_energy_json(request_json: &str) -> Result<String, JsValue> {
    let request = request_from_json(request_json).map_err(js_error)?;
    let payload = zspace_free_energy_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceFreeEnergyObject)]
pub fn zspace_free_energy_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let request = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request = request_from_value(request).map_err(js_error)?;
    let payload = zspace_free_energy_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn wasm_free_energy_matches_rust_exactly() {
        let request: FreeEnergyRequest = serde_json::from_value(json!({
            "observation": {
                "reference_loss": 0.8,
                "candidate_loss": 0.5,
                "step_time_ms": 12.0,
                "memory_mb": 256.0,
                "retry_rate": 0.05,
                "observation_entropy": 0.4,
                "external_penalty": 0.1,
                "band": {"above": 0.6, "here": 0.3, "beneath": 0.1}
            }
        }))
        .expect("valid request");
        let mut expected =
            serde_json::to_value(evaluate_free_energy(request).expect("valid Rust evaluation"))
                .expect("serializable Rust report");
        add_execution_client(&mut expected);
        let actual = zspace_free_energy_value(request).expect("valid WASM evaluation");

        assert_eq!(actual, expected);
        assert_eq!(actual["semantic_backend"], "rust");
        assert_eq!(actual["execution_client"], "wasm");
        assert_eq!(actual["distribution"]["dominant_band"], "above");
    }

    #[test]
    fn wasm_free_energy_preserves_rust_validation() {
        let request = request_from_json(
            r#"{"observation":{"band":{"above":-1.0,"here":1.0,"beneath":1.0}}}"#,
        )
        .expect("syntactically valid request");
        let error =
            zspace_free_energy_value(request).expect_err("negative band energy must fail closed");
        assert!(matches!(error, FreeEnergyError::Negative { .. }));

        let wrong_request = request_from_json(r#"[]"#).expect_err("request must be an object");
        assert_eq!(
            wrong_request,
            "variational free-energy request must be an object"
        );
    }

    #[test]
    fn wasm_zero_band_mass_uses_the_rust_configured_prior() {
        let request = request_from_value(json!({
            "config": {
                "prior": {"above": 0.5, "here": 0.3, "beneath": 0.2},
                "band_potentials": {"above": -0.2, "here": 0.0, "beneath": 0.4}
            }
        }))
        .expect("valid non-uniform prior");
        let payload = zspace_free_energy_value(request).expect("valid Rust evaluation");

        assert_eq!(payload["distribution"]["status"], "prior_zero_mass");
        assert_eq!(payload["distribution"]["above"], 0.5);
        assert_eq!(payload["distribution"]["here"], 0.3);
        assert_eq!(payload["distribution"]["beneath"], 0.2);
        assert_eq!(payload["distribution"]["kl_divergence"], 0.0);
        assert_eq!(payload["components"]["band_potential"], 0.0);
    }
}
