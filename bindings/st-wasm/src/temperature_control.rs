use serde_json::Value;
use st_core::inference::temperature_control::{
    apply_temperature_control, TemperatureControlError, TemperatureControlRequest,
};

#[cfg(target_arch = "wasm32")]
use serde::Serialize;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::js_error;

fn request_from_value(value: Value) -> Result<TemperatureControlRequest, String> {
    let request = value
        .as_object()
        .ok_or_else(|| "Z-space temperature control request must be an object".to_owned())?;
    if request
        .get("probabilities")
        .is_some_and(|value| !value.is_array())
    {
        return Err("Z-space temperature control 'probabilities' must be an array".to_owned());
    }
    for field in ["config", "state"] {
        if request.get(field).is_some_and(|value| !value.is_object()) {
            return Err(format!(
                "Z-space temperature control '{field}' must be an object"
            ));
        }
    }
    if request
        .get("feedback")
        .is_some_and(|value| !value.is_object() && !value.is_null())
    {
        return Err("Z-space temperature control 'feedback' must be an object".to_owned());
    }
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn request_from_json(input_json: &str) -> Result<TemperatureControlRequest, String> {
    let value = serde_json::from_str(input_json).map_err(|error| error.to_string())?;
    request_from_value(value)
}

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

pub fn zspace_temperature_control_value(
    request: TemperatureControlRequest,
) -> Result<Value, TemperatureControlError> {
    Ok(serde_json::to_value(apply_temperature_control(request)?)
        .expect("Z-space temperature control payload is serializable"))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceTemperatureControlJson)]
pub fn zspace_temperature_control_json(request_json: &str) -> Result<String, JsValue> {
    let request = request_from_json(request_json).map_err(js_error)?;
    let payload = zspace_temperature_control_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceTemperatureControlObject)]
pub fn zspace_temperature_control_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let request = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request = request_from_value(request).map_err(js_error)?;
    let payload = zspace_temperature_control_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use st_core::inference::temperature_control::apply_temperature_control;

    #[test]
    fn wasm_temperature_control_matches_rust_exactly() {
        let request: TemperatureControlRequest = serde_json::from_value(json!({
            "probabilities": [0.6, 0.4],
            "config": {
                "target_entropy": 0.8,
                "eta": 0.2,
                "min_temperature": 0.3,
                "max_temperature": 2.0,
                "z_kappa": 0.4,
                "z_relax": 0.2,
                "scale_gain": 0.6,
                "gradient_decay": 0.5
            },
            "state": {
                "temperature": 1.0,
                "z_memory": 0.0,
                "scale_memory": 0.0,
                "gradient_pressure": 12.0,
                "gradient_entropy_bias": 0.2
            },
            "feedback": {
                "psi_total": 0.1,
                "band_energy": [0.2, 0.1, 0.05],
                "drift": 0.4,
                "z_signal": 0.1,
                "scale_log_radius": -1.3862943611198906
            },
            "gradient_heat": 0.5
        }))
        .expect("valid request");
        let expected = serde_json::to_value(
            apply_temperature_control(request.clone()).expect("valid Rust transition"),
        )
        .expect("serializable Rust transition");
        let actual =
            zspace_temperature_control_value(request).expect("valid WASM temperature transition");

        assert_eq!(actual, expected);
        assert!(actual["entropy_error"].as_f64().expect("numeric error") > 0.0);
        assert!(
            actual["temperature_after_entropy"]
                .as_f64()
                .expect("numeric intermediate temperature")
                > actual["previous_state"]["temperature"]
                    .as_f64()
                    .expect("numeric previous temperature")
        );
    }

    #[test]
    fn wasm_temperature_ingress_fails_closed_on_contract_drift() {
        let unknown = request_from_json(r#"{"probabilities":[1.0],"temperatur":1.0}"#)
            .expect_err("unknown request fields must fail closed");
        assert!(unknown.contains("unknown field"));

        let wrong_shape = request_from_json(r#"{"probabilities":{}}"#)
            .expect_err("probabilities must be an array");
        assert_eq!(
            wrong_shape,
            "Z-space temperature control 'probabilities' must be an array"
        );

        let null_state =
            request_from_json(r#"{"state":null}"#).expect_err("state must be a concrete object");
        assert_eq!(
            null_state,
            "Z-space temperature control 'state' must be an object"
        );

        let wrong_request = request_from_json(r#"[]"#).expect_err("request must be an object");
        assert_eq!(
            wrong_request,
            "Z-space temperature control request must be an object"
        );
    }

    #[test]
    fn wasm_temperature_control_preserves_rust_validation() {
        let request = request_from_json(r#"{"probabilities":[0.8,0.8]}"#)
            .expect("syntactically valid request");
        let error = zspace_temperature_control_value(request)
            .expect_err("invalid probability mass must fail closed");

        assert!(matches!(
            error,
            TemperatureControlError::InvalidProbabilityMass { .. }
        ));
    }
}
