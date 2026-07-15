use serde_json::Value;
use st_core::inference::imaginary_time_schrodinger::{
    apply_imaginary_time_schrodinger, ImaginaryTimeSchrodingerError,
    ImaginaryTimeSchrodingerRequest,
};

#[cfg(target_arch = "wasm32")]
use serde::Serialize;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::js_error;

fn request_from_value(value: Value) -> Result<ImaginaryTimeSchrodingerRequest, String> {
    let request = value
        .as_object()
        .ok_or_else(|| "Z-space imaginary-time Schrodinger request must be an object".to_owned())?;
    for field in ["tags", "potential", "edges", "initial_amplitude"] {
        if request.get(field).is_some_and(|value| !value.is_array()) {
            return Err(format!(
                "Z-space imaginary-time Schrodinger '{field}' must be an array"
            ));
        }
    }
    if request
        .get("config")
        .is_some_and(|value| !value.is_object())
    {
        return Err("Z-space imaginary-time Schrodinger 'config' must be an object".to_owned());
    }
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn request_from_json(input_json: &str) -> Result<ImaginaryTimeSchrodingerRequest, String> {
    let value = serde_json::from_str(input_json).map_err(|error| error.to_string())?;
    request_from_value(value)
}

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

pub fn zspace_imaginary_time_schrodinger_value(
    request: ImaginaryTimeSchrodingerRequest,
) -> Result<Value, ImaginaryTimeSchrodingerError> {
    Ok(
        serde_json::to_value(apply_imaginary_time_schrodinger(request)?)
            .expect("Z-space imaginary-time Schrodinger payload is serializable"),
    )
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceImaginaryTimeSchrodingerJson)]
pub fn zspace_imaginary_time_schrodinger_json(request_json: &str) -> Result<String, JsValue> {
    let request = request_from_json(request_json).map_err(js_error)?;
    let payload = zspace_imaginary_time_schrodinger_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceImaginaryTimeSchrodingerObject)]
pub fn zspace_imaginary_time_schrodinger_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let request = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request = request_from_value(request).map_err(js_error)?;
    let payload = zspace_imaginary_time_schrodinger_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn wasm_schrodinger_matches_rust_exactly() {
        let request: ImaginaryTimeSchrodingerRequest = serde_json::from_value(json!({
            "tags": ["left", "right"],
            "potential": [0.0, 2.0],
            "edges": [{"left": 0, "right": 1, "weight": 1.0}],
            "config": {"imaginary_time": 1.0}
        }))
        .expect("valid request");
        let expected = serde_json::to_value(
            apply_imaginary_time_schrodinger(request.clone()).expect("valid Rust evolution"),
        )
        .expect("serializable Rust evolution");
        let actual = zspace_imaginary_time_schrodinger_value(request)
            .expect("valid WASM Schrodinger evolution");

        assert_eq!(actual, expected);
        assert!(
            actual["probability"][0].as_f64().unwrap() > actual["probability"][1].as_f64().unwrap()
        );
        assert!(
            actual["effects"]["final_rayleigh_energy"].as_f64().unwrap()
                < actual["effects"]["initial_rayleigh_energy"]
                    .as_f64()
                    .unwrap()
        );
    }

    #[test]
    fn wasm_schrodinger_ingress_fails_closed_on_contract_drift() {
        let unknown = request_from_json(r#"{"tags":["only"],"potential":[0.0],"hamiltonian":[]}"#)
            .expect_err("unknown request fields must fail closed");
        assert!(unknown.contains("unknown field"));

        let wrong_shape =
            request_from_json(r#"{"potential":{}}"#).expect_err("potential must be an array");
        assert_eq!(
            wrong_shape,
            "Z-space imaginary-time Schrodinger 'potential' must be an array"
        );

        let null_config =
            request_from_json(r#"{"config":null}"#).expect_err("config must be a concrete object");
        assert_eq!(
            null_config,
            "Z-space imaginary-time Schrodinger 'config' must be an object"
        );
    }

    #[test]
    fn wasm_schrodinger_preserves_rust_validation() {
        let request = request_from_json(
            r#"{"tags":["left","right"],"potential":[0.0,1.0],"edges":[{"left":1,"right":0,"weight":1.0}]}"#,
        )
        .expect("syntactically valid request");
        let error = zspace_imaginary_time_schrodinger_value(request)
            .expect_err("non-canonical edge must fail closed");

        assert!(matches!(
            error,
            ImaginaryTimeSchrodingerError::NonCanonicalEdge { .. }
        ));
    }
}
