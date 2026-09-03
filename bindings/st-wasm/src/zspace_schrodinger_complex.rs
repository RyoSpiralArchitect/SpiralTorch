// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::Value;
use st_core::runtime::zspace_stochastic_schrodinger::{
    run_zspace_stochastic_schrodinger_complex_step,
    validate_zspace_stochastic_schrodinger_complex_value, ZSpaceSchrodingerComplexRequest,
    ZSpaceStochasticSchrodingerProtocolError, ZSPACE_SCHRODINGER_COMPLEX_MAX_INGRESS_BYTES,
    ZSPACE_SCHRODINGER_COMPLEX_MAX_INGRESS_NODES, ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_DEPTH,
};

fn complex_json_value(text: &str) -> Result<Value, String> {
    crate::utils::bounded_json_value_from_str(
        text,
        ZSPACE_SCHRODINGER_COMPLEX_MAX_INGRESS_BYTES,
        ZSPACE_SCHRODINGER_COMPLEX_MAX_INGRESS_NODES,
        ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_DEPTH,
        "complex Schrodinger JSON",
    )
}

fn encode<T: serde::Serialize>(
    value: T,
) -> Result<Value, ZSpaceStochasticSchrodingerProtocolError> {
    serde_json::to_value(value).map_err(|e| ZSpaceStochasticSchrodingerProtocolError::Encoding {
        message: e.to_string(),
    })
}

pub fn zspace_stochastic_schrodinger_complex_step_value(
    request: ZSpaceSchrodingerComplexRequest,
) -> Result<Value, ZSpaceStochasticSchrodingerProtocolError> {
    encode(run_zspace_stochastic_schrodinger_complex_step(request)?)
}

pub fn validate_zspace_stochastic_schrodinger_complex_receipt_value(
    receipt: Value,
) -> Result<Value, ZSpaceStochasticSchrodingerProtocolError> {
    encode(validate_zspace_stochastic_schrodinger_complex_value(
        receipt,
    )?)
}

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceStochasticSchrodingerComplexStepJson)]
pub fn zspace_stochastic_schrodinger_complex_step_json(
    request_json: &js_sys::JsString,
) -> Result<String, JsValue> {
    use crate::utils::{bounded_json_string_from_js, js_error};
    let text = bounded_json_string_from_js(
        request_json,
        ZSPACE_SCHRODINGER_COMPLEX_MAX_INGRESS_BYTES,
        "complex Schrodinger request",
    )?;
    let value = complex_json_value(&text).map_err(js_error)?;
    let request = serde_json::from_value(value).map_err(js_error)?;
    let receipt = zspace_stochastic_schrodinger_complex_step_value(request).map_err(js_error)?;
    serde_json::to_string(&receipt).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = validateZspaceStochasticSchrodingerComplexJson)]
pub fn validate_zspace_stochastic_schrodinger_complex_json(
    receipt_json: &js_sys::JsString,
) -> Result<String, JsValue> {
    use crate::utils::{bounded_json_string_from_js, js_error};
    let text = bounded_json_string_from_js(
        receipt_json,
        ZSPACE_SCHRODINGER_COMPLEX_MAX_INGRESS_BYTES,
        "complex Schrodinger receipt",
    )?;
    let receipt = complex_json_value(&text).map_err(js_error)?;
    let receipt =
        validate_zspace_stochastic_schrodinger_complex_receipt_value(receipt).map_err(js_error)?;
    serde_json::to_string(&receipt).map_err(js_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn maximum_complex_receipt_fits_the_advertised_ingress() {
        use st_core::dynamics::stochastic_schrodinger::StochasticSchrodingerConfig;
        use st_core::runtime::zspace_stochastic_schrodinger::{
            ZSpaceSchrodingerComplexCotangent, ZSpaceStochasticSchrodingerForwardRequest,
            ZSPACE_STOCHASTIC_SCHRODINGER_MAX_FEATURES, ZSPACE_STOCHASTIC_SCHRODINGER_MAX_VALUES,
        };
        let n = ZSPACE_STOCHASTIC_SCHRODINGER_MAX_VALUES;
        let features = ZSPACE_STOCHASTIC_SCHRODINGER_MAX_FEATURES;
        let receipt =
            run_zspace_stochastic_schrodinger_complex_step(ZSpaceSchrodingerComplexRequest {
                forward_request: ZSpaceStochasticSchrodingerForwardRequest {
                    input: vec![f32::MAX; n],
                    potential: vec![f32::MAX; features],
                    standard_normal: vec![f32::MAX; n],
                    rows: n / features,
                    features,
                    config: StochasticSchrodingerConfig::default()
                        .with_time_step(0.0)
                        .unwrap(),
                },
                input_imaginary: vec![f32::MAX; n],
                cotangent: Some(ZSpaceSchrodingerComplexCotangent {
                    real: vec![f32::MAX; n],
                    imaginary: vec![f32::MAX; n],
                }),
            })
            .unwrap();
        let text = serde_json::to_string(&receipt).unwrap();
        assert!(text.len() as u64 <= ZSPACE_SCHRODINGER_COMPLEX_MAX_INGRESS_BYTES);
        // Exercise the real byte/node/depth admission path, not just a hand-count.
        let parsed = complex_json_value(&text).unwrap();
        assert_eq!(
            parsed["gradient"]["grad_input_imaginary"]
                .as_array()
                .unwrap()
                .len(),
            n
        );
    }

    #[test]
    fn complex_wasm_surface_matches_rust_and_rejects_drift() {
        let value = complex_json_value(r#"{"forward_request":{"input":[0.5,0.2],"potential":[0.1,-0.3],"standard_normal":[0.4,0.2],"rows":1,"features":2},"input_imaginary":[0.2,-0.4],"cotangent":{"real":[1.0,0.0],"imaginary":[0.0,1.0]}}"#).unwrap();
        let request: ZSpaceSchrodingerComplexRequest = serde_json::from_value(value).unwrap();
        let expected =
            encode(run_zspace_stochastic_schrodinger_complex_step(request.clone()).unwrap())
                .unwrap();
        let actual = zspace_stochastic_schrodinger_complex_step_value(request).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(
            validate_zspace_stochastic_schrodinger_complex_receipt_value(actual.clone()).unwrap(),
            actual
        );
        let mut altered = actual;
        altered["step"]["output_imaginary"][0] = json!(9.0);
        assert!(validate_zspace_stochastic_schrodinger_complex_receipt_value(altered).is_err());
        assert!(complex_json_value(&format!("{}0{}", "[".repeat(12), "]".repeat(12))).is_err());
        assert!(complex_json_value(r#"{"x":NaN}"#).is_err());
    }
}
