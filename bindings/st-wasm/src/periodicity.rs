use serde_json::Value;
use st_core::runtime::zspace_periodicity::{
    analyze_zspace_periodicity, validate_zspace_periodicity_value, ZSpacePeriodicityError,
    ZSpacePeriodicityRequest, ZSPACE_PERIODICITY_MAX_INGRESS_BYTES,
    ZSPACE_PERIODICITY_MAX_INGRESS_DEPTH, ZSPACE_PERIODICITY_MAX_INGRESS_NODES,
};

use crate::utils::bounded_json_value_from_str;

#[cfg(target_arch = "wasm32")]
use js_sys::JsString;
#[cfg(target_arch = "wasm32")]
use serde::Serialize;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::{bounded_json_string_from_js, js_error, snapshot_json_compatible_js_value};

fn bounded_json_value(input_json: &str, context: &str) -> Result<Value, String> {
    bounded_json_value_with_limit(input_json, context, ZSPACE_PERIODICITY_MAX_INGRESS_BYTES)
}

fn bounded_json_value_with_limit(
    input_json: &str,
    context: &str,
    maximum_bytes: u64,
) -> Result<Value, String> {
    bounded_json_value_from_str(
        input_json,
        maximum_bytes,
        ZSPACE_PERIODICITY_MAX_INGRESS_NODES,
        ZSPACE_PERIODICITY_MAX_INGRESS_DEPTH,
        context,
    )
}

#[cfg(target_arch = "wasm32")]
fn snapshot_periodicity_js_value(value: &JsValue, context: &str) -> Result<JsValue, JsValue> {
    snapshot_json_compatible_js_value(
        value,
        ZSPACE_PERIODICITY_MAX_INGRESS_BYTES,
        ZSPACE_PERIODICITY_MAX_INGRESS_NODES,
        ZSPACE_PERIODICITY_MAX_INGRESS_DEPTH,
        context,
    )
}

fn request_from_value(value: Value) -> Result<ZSpacePeriodicityRequest, String> {
    let request = value
        .as_object()
        .ok_or_else(|| "Z-space periodicity request must be an object".to_owned())?;
    if request
        .get("token_ids")
        .is_some_and(|value| !value.is_array())
    {
        return Err("Z-space periodicity 'token_ids' must be an array".to_owned());
    }
    if request
        .get("config")
        .is_some_and(|value| !value.is_object())
    {
        return Err("Z-space periodicity 'config' must be an object".to_owned());
    }
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn request_from_json(input_json: &str) -> Result<ZSpacePeriodicityRequest, String> {
    let value = bounded_json_value(input_json, "Z-space periodicity request")?;
    request_from_value(value)
}

fn report_from_value(value: Value) -> Result<Value, String> {
    if !value.is_object() {
        return Err("Z-space periodicity report must be an object".to_owned());
    }
    Ok(value)
}

fn report_from_json(input_json: &str) -> Result<Value, String> {
    let value = bounded_json_value(input_json, "Z-space periodicity report")?;
    report_from_value(value)
}

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

pub fn zspace_periodicity_value(
    request: ZSpacePeriodicityRequest,
) -> Result<Value, ZSpacePeriodicityError> {
    Ok(serde_json::to_value(analyze_zspace_periodicity(request)?)
        .expect("Z-space periodicity report is serializable"))
}

pub fn validate_zspace_periodicity_report_value(
    report: Value,
) -> Result<Value, ZSpacePeriodicityError> {
    Ok(
        serde_json::to_value(validate_zspace_periodicity_value(report)?)
            .expect("Z-space periodicity report is serializable"),
    )
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspacePeriodicityJson)]
pub fn zspace_periodicity_json(request_json: &JsString) -> Result<String, JsValue> {
    let request_json = bounded_json_string_from_js(
        request_json,
        ZSPACE_PERIODICITY_MAX_INGRESS_BYTES,
        "Z-space periodicity request JSON",
    )?;
    let request = request_from_json(&request_json).map_err(js_error)?;
    let report = zspace_periodicity_value(request).map_err(js_error)?;
    serde_json::to_string(&report).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspacePeriodicityObject)]
pub fn zspace_periodicity_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let request = snapshot_periodicity_js_value(request, "Z-space periodicity request")?;
    let request = serde_wasm_bindgen::from_value::<Value>(request).map_err(js_error)?;
    let request = request_from_value(request).map_err(js_error)?;
    let report = zspace_periodicity_value(request).map_err(js_error)?;
    to_json_compatible_js(&report)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = validateZspacePeriodicityJson)]
pub fn validate_zspace_periodicity_json(report_json: &JsString) -> Result<String, JsValue> {
    let report_json = bounded_json_string_from_js(
        report_json,
        ZSPACE_PERIODICITY_MAX_INGRESS_BYTES,
        "Z-space periodicity report JSON",
    )?;
    let report = report_from_json(&report_json).map_err(js_error)?;
    let report = validate_zspace_periodicity_report_value(report).map_err(js_error)?;
    serde_json::to_string(&report).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = validateZspacePeriodicityObject)]
pub fn validate_zspace_periodicity_object(report: &JsValue) -> Result<JsValue, JsValue> {
    let report = snapshot_periodicity_js_value(report, "Z-space periodicity report")?;
    let report = serde_wasm_bindgen::from_value::<Value>(report).map_err(js_error)?;
    let report = report_from_value(report).map_err(js_error)?;
    let report = validate_zspace_periodicity_report_value(report).map_err(js_error)?;
    to_json_compatible_js(&report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use st_core::runtime::zspace_periodicity::analyze_zspace_periodicity;

    #[test]
    fn wasm_periodicity_matches_rust_exactly() {
        let request: ZSpacePeriodicityRequest = serde_json::from_value(json!({
            "token_ids": [9, 1, 2, 1, 2, 1],
            "appended_token_id": 2
        }))
        .expect("valid request");
        let expected = serde_json::to_value(
            analyze_zspace_periodicity(request.clone()).expect("valid Rust analysis"),
        )
        .expect("serializable Rust analysis");
        let actual = zspace_periodicity_value(request).expect("valid WASM analysis");

        assert_eq!(actual, expected);
    }

    #[test]
    fn wasm_periodicity_validator_rejects_tampering() {
        let request = request_from_json(r#"{"token_ids":[1,2,1,2,1,2]}"#).expect("valid request");
        let report = zspace_periodicity_value(request).expect("valid analysis");
        assert_eq!(
            validate_zspace_periodicity_report_value(report.clone()).expect("canonical report"),
            report
        );

        let mut tampered = report;
        tampered["periodic_loop_detected"] = Value::Bool(false);
        let error = validate_zspace_periodicity_report_value(tampered)
            .expect_err("tampered report must fail");
        assert!(matches!(
            error,
            ZSpacePeriodicityError::MalformedReport { .. }
        ));
    }

    #[test]
    fn wasm_periodicity_validator_accepts_javascript_json_number_spelling() {
        let request = request_from_json(r#"{"token_ids":[1,1,1]}"#).expect("valid request");
        let canonical = zspace_periodicity_value(request).expect("valid analysis");
        let mut stored = canonical.clone();
        stored["periodic_suffix_token_ratio"] = json!(1);
        let stored_json = serde_json::to_string(&stored).expect("stored browser JSON");
        let parsed = report_from_json(&stored_json).expect("reloaded browser JSON");

        assert_eq!(
            validate_zspace_periodicity_report_value(parsed).expect("numeric JSON round-trip"),
            canonical
        );
    }

    #[test]
    fn wasm_periodicity_ingress_fails_closed_on_contract_drift() {
        let unknown = request_from_json(r#"{"token_ids":[],"max_period":4}"#)
            .expect_err("unknown request fields must fail closed");
        assert!(unknown.contains("unknown field"));

        let wrong_tokens =
            request_from_json(r#"{"token_ids":{}}"#).expect_err("token IDs must be an array");
        assert_eq!(
            wrong_tokens,
            "Z-space periodicity 'token_ids' must be an array"
        );

        let wrong_config = request_from_json(r#"{"token_ids":[],"config":[]}"#)
            .expect_err("configuration must be an object");
        assert_eq!(
            wrong_config,
            "Z-space periodicity 'config' must be an object"
        );

        let wrong_request = request_from_json(r#"[]"#).expect_err("request must be an object");
        assert_eq!(
            wrong_request,
            "Z-space periodicity request must be an object"
        );

        let wrong_report = report_from_json(r#"[]"#).expect_err("report must be an object");
        assert_eq!(wrong_report, "Z-space periodicity report must be an object");

        let duplicate = request_from_json(r#"{"token_ids":[],"token_ids":[1]}"#)
            .expect_err("duplicate keys must fail closed");
        assert!(duplicate.contains("duplicate JSON object key"));
    }

    #[test]
    fn wasm_periodicity_json_is_bounded_before_deserialization() {
        let oversized =
            bounded_json_value_with_limit(r#"{"token_ids":[]}"#, "test periodicity request", 4)
                .expect_err("byte budget must run before deserialization");
        assert!(oversized.contains("exceeds WASM ingress budget of 4 bytes"));

        let node_error = bounded_json_value_from_str(
            r#"{"token_ids":[1]}"#,
            ZSPACE_PERIODICITY_MAX_INGRESS_BYTES,
            2,
            ZSPACE_PERIODICITY_MAX_INGRESS_DEPTH,
            "test periodicity request",
        )
        .expect_err("node budget must run during materialization");
        assert!(node_error.contains("2 JSON nodes"));

        let depth_error = bounded_json_value_from_str(
            r#"{"token_ids":[1]}"#,
            ZSPACE_PERIODICITY_MAX_INGRESS_BYTES,
            ZSPACE_PERIODICITY_MAX_INGRESS_NODES,
            1,
            "test periodicity request",
        )
        .expect_err("depth budget must run during materialization");
        assert!(depth_error.contains("depth limit 1"));
    }
}
