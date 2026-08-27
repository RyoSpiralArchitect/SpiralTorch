// SPDX-License-Identifier: AGPL-3.0-or-later

use serde::de::DeserializeOwned;
use serde_json::Value;
use st_core::runtime::zspace_stochastic_schrodinger::{
    run_zspace_stochastic_schrodinger_forward, run_zspace_stochastic_schrodinger_vjp,
    validate_zspace_stochastic_schrodinger_forward_value,
    validate_zspace_stochastic_schrodinger_vjp_value, ZSpaceStochasticSchrodingerForwardRequest,
    ZSpaceStochasticSchrodingerProtocolError, ZSpaceStochasticSchrodingerVjpRequest,
    ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_BYTES,
    ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_DEPTH,
    ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_NODES,
};

use crate::utils::bounded_json_value_from_str;

#[cfg(target_arch = "wasm32")]
use js_sys::JsString;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::{bounded_json_string_from_js, js_error};

fn bounded_json_value(input_json: &str, context: &str) -> Result<Value, String> {
    bounded_json_value_with_limit(
        input_json,
        context,
        ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_BYTES,
    )
}

fn bounded_json_value_with_limit(
    input_json: &str,
    context: &str,
    maximum_bytes: u64,
) -> Result<Value, String> {
    bounded_json_value_from_str(
        input_json,
        maximum_bytes,
        ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_NODES,
        ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_DEPTH,
        context,
    )
}

fn typed_request_from_json<T: DeserializeOwned>(
    input_json: &str,
    context: &str,
) -> Result<T, String> {
    let value = bounded_json_value(input_json, context)?;
    if !value.is_object() {
        return Err(format!("{context} must be an object"));
    }
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn receipt_from_json(input_json: &str, context: &str) -> Result<Value, String> {
    let value = bounded_json_value(input_json, context)?;
    if value.is_object() {
        Ok(value)
    } else {
        Err(format!("{context} must be an object"))
    }
}

pub fn zspace_stochastic_schrodinger_forward_value(
    request: ZSpaceStochasticSchrodingerForwardRequest,
) -> Result<Value, ZSpaceStochasticSchrodingerProtocolError> {
    serde_json::to_value(run_zspace_stochastic_schrodinger_forward(request)?).map_err(|error| {
        ZSpaceStochasticSchrodingerProtocolError::Encoding {
            message: error.to_string(),
        }
    })
}

pub fn validate_zspace_stochastic_schrodinger_forward_receipt_value(
    receipt: Value,
) -> Result<Value, ZSpaceStochasticSchrodingerProtocolError> {
    serde_json::to_value(validate_zspace_stochastic_schrodinger_forward_value(
        receipt,
    )?)
    .map_err(|error| ZSpaceStochasticSchrodingerProtocolError::Encoding {
        message: error.to_string(),
    })
}

pub fn zspace_stochastic_schrodinger_vjp_value(
    request: ZSpaceStochasticSchrodingerVjpRequest,
) -> Result<Value, ZSpaceStochasticSchrodingerProtocolError> {
    serde_json::to_value(run_zspace_stochastic_schrodinger_vjp(request)?).map_err(|error| {
        ZSpaceStochasticSchrodingerProtocolError::Encoding {
            message: error.to_string(),
        }
    })
}

pub fn validate_zspace_stochastic_schrodinger_vjp_receipt_value(
    receipt: Value,
) -> Result<Value, ZSpaceStochasticSchrodingerProtocolError> {
    serde_json::to_value(validate_zspace_stochastic_schrodinger_vjp_value(receipt)?).map_err(
        |error| ZSpaceStochasticSchrodingerProtocolError::Encoding {
            message: error.to_string(),
        },
    )
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceStochasticSchrodingerForwardJson)]
pub fn zspace_stochastic_schrodinger_forward_json(
    request_json: &JsString,
) -> Result<String, JsValue> {
    let request_json = bounded_json_string_from_js(
        request_json,
        ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_BYTES,
        "Z-space stochastic Schrodinger forward request JSON",
    )?;
    let request = typed_request_from_json(
        &request_json,
        "Z-space stochastic Schrodinger forward request",
    )
    .map_err(js_error)?;
    let receipt = zspace_stochastic_schrodinger_forward_value(request).map_err(js_error)?;
    serde_json::to_string(&receipt).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = validateZspaceStochasticSchrodingerForwardJson)]
pub fn validate_zspace_stochastic_schrodinger_forward_json(
    receipt_json: &JsString,
) -> Result<String, JsValue> {
    let receipt_json = bounded_json_string_from_js(
        receipt_json,
        ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_BYTES,
        "Z-space stochastic Schrodinger forward receipt JSON",
    )?;
    let receipt = receipt_from_json(
        &receipt_json,
        "Z-space stochastic Schrodinger forward receipt",
    )
    .map_err(js_error)?;
    let receipt =
        validate_zspace_stochastic_schrodinger_forward_receipt_value(receipt).map_err(js_error)?;
    serde_json::to_string(&receipt).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceStochasticSchrodingerVjpJson)]
pub fn zspace_stochastic_schrodinger_vjp_json(request_json: &JsString) -> Result<String, JsValue> {
    let request_json = bounded_json_string_from_js(
        request_json,
        ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_BYTES,
        "Z-space stochastic Schrodinger VJP request JSON",
    )?;
    let request =
        typed_request_from_json(&request_json, "Z-space stochastic Schrodinger VJP request")
            .map_err(js_error)?;
    let receipt = zspace_stochastic_schrodinger_vjp_value(request).map_err(js_error)?;
    serde_json::to_string(&receipt).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = validateZspaceStochasticSchrodingerVjpJson)]
pub fn validate_zspace_stochastic_schrodinger_vjp_json(
    receipt_json: &JsString,
) -> Result<String, JsValue> {
    let receipt_json = bounded_json_string_from_js(
        receipt_json,
        ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_BYTES,
        "Z-space stochastic Schrodinger VJP receipt JSON",
    )?;
    let receipt = receipt_from_json(&receipt_json, "Z-space stochastic Schrodinger VJP receipt")
        .map_err(js_error)?;
    let receipt =
        validate_zspace_stochastic_schrodinger_vjp_receipt_value(receipt).map_err(js_error)?;
    serde_json::to_string(&receipt).map_err(js_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn forward_request() -> ZSpaceStochasticSchrodingerForwardRequest {
        serde_json::from_value(json!({
            "input": [1.0, 0.25, -0.5, 0.75, 0.1, -0.2],
            "potential": [0.2, -0.1, 0.05],
            "standard_normal": [0.1, -0.3, 0.2, 0.0, 0.4, -0.2],
            "rows": 2,
            "features": 3,
            "config": {
                "time_step": 0.08,
                "hopping_rate": 0.35,
                "loss_rate": 0.02,
                "noise_scale": 0.15
            }
        }))
        .expect("valid request")
    }

    #[test]
    fn wasm_forward_and_vjp_match_rust_exactly() {
        let forward_request = forward_request();
        let expected_forward = serde_json::to_value(
            run_zspace_stochastic_schrodinger_forward(forward_request.clone()).expect("forward"),
        )
        .expect("serializable forward");
        let actual_forward = zspace_stochastic_schrodinger_forward_value(forward_request.clone())
            .expect("WASM forward");
        assert_eq!(actual_forward, expected_forward);
        assert_eq!(
            validate_zspace_stochastic_schrodinger_forward_receipt_value(actual_forward.clone())
                .expect("forward replay"),
            actual_forward
        );

        let vjp_request = ZSpaceStochasticSchrodingerVjpRequest {
            forward_request,
            grad_output_real: vec![0.2, -0.4, 0.1, 0.3, 0.0, -0.2],
        };
        let expected_vjp = serde_json::to_value(
            run_zspace_stochastic_schrodinger_vjp(vjp_request.clone()).expect("VJP"),
        )
        .expect("serializable VJP");
        let actual_vjp = zspace_stochastic_schrodinger_vjp_value(vjp_request).expect("WASM VJP");
        assert_eq!(actual_vjp, expected_vjp);
        assert_eq!(
            validate_zspace_stochastic_schrodinger_vjp_receipt_value(actual_vjp.clone())
                .expect("VJP replay"),
            actual_vjp
        );
    }

    #[test]
    fn wasm_validators_reject_tampering() {
        let mut forward =
            zspace_stochastic_schrodinger_forward_value(forward_request()).expect("forward");
        forward["step"]["phase"][0] = json!(999.0);
        assert!(matches!(
            validate_zspace_stochastic_schrodinger_forward_receipt_value(forward),
            Err(ZSpaceStochasticSchrodingerProtocolError::MalformedReceipt { .. })
        ));

        let request = ZSpaceStochasticSchrodingerVjpRequest {
            forward_request: forward_request(),
            grad_output_real: vec![1.0; 6],
        };
        let mut vjp = zspace_stochastic_schrodinger_vjp_value(request).expect("VJP");
        vjp["result"]["grad_input"][0] = json!(999.0);
        assert!(matches!(
            validate_zspace_stochastic_schrodinger_vjp_receipt_value(vjp),
            Err(ZSpaceStochasticSchrodingerProtocolError::MalformedReceipt { .. })
        ));
    }

    #[test]
    fn wasm_json_ingress_is_bounded_and_fail_closed() {
        assert!(
            typed_request_from_json::<ZSpaceStochasticSchrodingerForwardRequest>(
                "[]",
                "forward request",
            )
            .expect_err("request must be an object")
            .contains("must be an object")
        );
        assert!(typed_request_from_json::<ZSpaceStochasticSchrodingerForwardRequest>(
            r#"{"input":[1.0],"potential":[0.0],"standard_normal":[0.0],"rows":1,"features":1,"phase":[0.0]}"#,
            "forward request",
        )
        .expect_err("unknown phase must fail")
        .contains("unknown field"));
        assert!(
            bounded_json_value(r#"{"rows":1,"rows":2}"#, "forward request",)
                .expect_err("duplicate keys fail closed")
                .contains("duplicate JSON object key")
        );
        assert!(
            bounded_json_value_with_limit("{\"input\":[]}", "forward request", 4)
                .expect_err("oversized JSON fails before parsing")
                .contains("exceeds WASM ingress budget")
        );
    }

    #[test]
    fn wasm_types_declare_json_only_dynamics_surface() {
        let declarations = include_str!("../types/spiraltorch-wasm.d.ts");

        for symbol in [
            "ZSpaceStochasticSchrodingerCanonicalForwardRequest",
            "ZSpaceStochasticSchrodingerCanonicalVjpRequest",
            "ZSpaceStochasticSchrodingerForwardRequest",
            "ZSpaceStochasticSchrodingerForwardReceipt",
            "ZSpaceStochasticSchrodingerVjpRequest",
            "ZSpaceStochasticSchrodingerVjpReceipt",
            "zspaceStochasticSchrodingerForwardJson",
            "validateZspaceStochasticSchrodingerForwardJson",
            "zspaceStochasticSchrodingerVjpJson",
            "validateZspaceStochasticSchrodingerVjpJson",
        ] {
            assert!(declarations.contains(symbol), "missing {symbol}");
        }
        assert!(!declarations.contains("zspaceStochasticSchrodingerForwardObject"));
        assert!(!declarations.contains("zspaceStochasticSchrodingerVjpObject"));
    }
}
