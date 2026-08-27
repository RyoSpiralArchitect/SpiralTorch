// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::Value;
use st_core::runtime::zspace_generation_evidence::{
    summarize_zspace_generation_evidence, validate_zspace_generation_evidence_value,
    ZSpaceGenerationEvidenceError, ZSpaceGenerationEvidenceRequest,
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

const WASM_GENERATION_EVIDENCE_MAX_INGRESS_BYTES: u64 = 96 * 1_024 * 1_024;
const WASM_GENERATION_EVIDENCE_MAX_INGRESS_NODES: u64 = 4_000_000;
const WASM_GENERATION_EVIDENCE_MAX_INGRESS_DEPTH: u32 = 32;

fn bounded_json_value(input_json: &str, context: &str) -> Result<Value, String> {
    bounded_json_value_with_limit(
        input_json,
        context,
        WASM_GENERATION_EVIDENCE_MAX_INGRESS_BYTES,
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
        WASM_GENERATION_EVIDENCE_MAX_INGRESS_NODES,
        WASM_GENERATION_EVIDENCE_MAX_INGRESS_DEPTH,
        context,
    )
}

#[cfg(target_arch = "wasm32")]
fn snapshot_generation_js_value(value: &JsValue, context: &str) -> Result<JsValue, JsValue> {
    snapshot_json_compatible_js_value(
        value,
        WASM_GENERATION_EVIDENCE_MAX_INGRESS_BYTES,
        WASM_GENERATION_EVIDENCE_MAX_INGRESS_NODES,
        WASM_GENERATION_EVIDENCE_MAX_INGRESS_DEPTH,
        context,
    )
}

fn request_from_value(value: Value) -> Result<ZSpaceGenerationEvidenceRequest, String> {
    if !value.is_object() {
        return Err("Z-space generation evidence request must be an object".to_owned());
    }
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn request_from_json(input_json: &str) -> Result<ZSpaceGenerationEvidenceRequest, String> {
    let value = bounded_json_value(input_json, "Z-space generation evidence request")?;
    request_from_value(value)
}

fn report_from_value(value: Value) -> Result<Value, String> {
    if !value.is_object() {
        return Err("Z-space generation evidence report must be an object".to_owned());
    }
    Ok(value)
}

fn report_from_json(input_json: &str) -> Result<Value, String> {
    let value = bounded_json_value(input_json, "Z-space generation evidence report")?;
    report_from_value(value)
}

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

pub fn zspace_generation_evidence_value(
    request: ZSpaceGenerationEvidenceRequest,
) -> Result<Value, ZSpaceGenerationEvidenceError> {
    Ok(
        serde_json::to_value(summarize_zspace_generation_evidence(request)?)
            .expect("Z-space generation evidence report is serializable"),
    )
}

pub fn validate_zspace_generation_evidence_report_value(
    report: Value,
) -> Result<Value, ZSpaceGenerationEvidenceError> {
    Ok(
        serde_json::to_value(validate_zspace_generation_evidence_value(report)?)
            .expect("Z-space generation evidence report is serializable"),
    )
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceGenerationEvidenceJson)]
pub fn zspace_generation_evidence_json(request_json: &JsString) -> Result<String, JsValue> {
    let request_json = bounded_json_string_from_js(
        request_json,
        WASM_GENERATION_EVIDENCE_MAX_INGRESS_BYTES,
        "Z-space generation evidence request JSON",
    )?;
    let request = request_from_json(&request_json).map_err(js_error)?;
    let report = zspace_generation_evidence_value(request).map_err(js_error)?;
    serde_json::to_string(&report).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceGenerationEvidenceObject)]
pub fn zspace_generation_evidence_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let request = snapshot_generation_js_value(request, "Z-space generation evidence request")?;
    let request = serde_wasm_bindgen::from_value::<Value>(request).map_err(js_error)?;
    let request = request_from_value(request).map_err(js_error)?;
    let report = zspace_generation_evidence_value(request).map_err(js_error)?;
    to_json_compatible_js(&report)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = validateZspaceGenerationEvidenceJson)]
pub fn validate_zspace_generation_evidence_json(report_json: &JsString) -> Result<String, JsValue> {
    let report_json = bounded_json_string_from_js(
        report_json,
        WASM_GENERATION_EVIDENCE_MAX_INGRESS_BYTES,
        "Z-space generation evidence report JSON",
    )?;
    let report = report_from_json(&report_json).map_err(js_error)?;
    let report = validate_zspace_generation_evidence_report_value(report).map_err(js_error)?;
    serde_json::to_string(&report).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = validateZspaceGenerationEvidenceObject)]
pub fn validate_zspace_generation_evidence_object(report: &JsValue) -> Result<JsValue, JsValue> {
    let report = snapshot_generation_js_value(report, "Z-space generation evidence report")?;
    let report = serde_wasm_bindgen::from_value::<Value>(report).map_err(js_error)?;
    let report = report_from_value(report).map_err(js_error)?;
    let report = validate_zspace_generation_evidence_report_value(report).map_err(js_error)?;
    to_json_compatible_js(&report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use st_core::runtime::zspace_generation_evidence::summarize_zspace_generation_evidence;

    fn request() -> ZSpaceGenerationEvidenceRequest {
        serde_json::from_value(json!({
            "protocol_id": format!("sha256:{}", "a".repeat(64)),
            "runtime_identity_id": format!("sha256:{}", "b".repeat(64)),
            "model_artifact_id": format!("sha256:{}", "c".repeat(64)),
            "prompt_set_id": format!("sha256:{}", "d".repeat(64)),
            "decoding_config_id": format!("sha256:{}", "e".repeat(64)),
            "samples": [{
                "prompt_id": format!("sha256:{}", "f".repeat(64)),
                "seed": 17,
                "continuation_token_ids": [9, 1, 2, 1, 2, 1, 2]
            }]
        }))
        .expect("valid request")
    }

    #[test]
    fn wasm_generation_evidence_matches_rust_exactly() {
        let request = request();
        let expected = serde_json::to_value(
            summarize_zspace_generation_evidence(request.clone()).expect("Rust evidence"),
        )
        .expect("serializable Rust evidence");
        let actual = zspace_generation_evidence_value(request).expect("WASM evidence");

        assert_eq!(actual, expected);
        assert_eq!(actual["semantic_backend"], "rust");
        assert_eq!(actual["aggregate"]["periodic_loop_sample_count"], 1);
    }

    #[test]
    fn wasm_generation_evidence_validator_rejects_tampering() {
        let report = zspace_generation_evidence_value(request()).expect("canonical evidence");
        assert_eq!(
            validate_zspace_generation_evidence_report_value(report.clone())
                .expect("canonical report"),
            report
        );

        let mut tampered = report;
        tampered["aggregate"]["sample_mean_loop_score"] = json!(0.0);
        let error = validate_zspace_generation_evidence_report_value(tampered)
            .expect_err("tampered evidence must fail");
        assert!(matches!(
            error,
            ZSpaceGenerationEvidenceError::MalformedReport { .. }
        ));
    }

    #[test]
    fn wasm_generation_evidence_validator_accepts_browser_number_spelling() {
        let canonical = zspace_generation_evidence_value(request()).expect("canonical evidence");
        let mut stored = canonical.clone();
        stored["samples"][0]["consecutive_repetition_ratio"] = json!(0);
        stored["aggregate"]["periodic_loop_sample_ratio"] = json!(1);

        assert_eq!(
            validate_zspace_generation_evidence_report_value(stored)
                .expect("browser JSON numeric round trip"),
            canonical
        );
    }

    #[test]
    fn wasm_generation_evidence_validator_accepts_serialized_periodic_loop_score() {
        let mut request = request();
        request.samples[0].continuation_token_ids = vec![1, 2, 1, 2, 1, 2];
        let canonical = zspace_generation_evidence_value(request).expect("canonical evidence");
        let stored = report_from_json(
            &serde_json::to_string(&canonical).expect("serializable generation evidence"),
        )
        .expect("bounded JSON report");

        assert_eq!(
            validate_zspace_generation_evidence_report_value(stored)
                .expect("serialized periodic loop report"),
            canonical
        );
    }

    #[test]
    fn wasm_generation_evidence_ingress_fails_closed() {
        let unknown = request_from_json(
            r#"{"protocol_id":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","runtime_identity_id":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","model_artifact_id":"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","prompt_set_id":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","decoding_config_id":"sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","samples":[],"browser_metric":"local"}"#,
        )
        .expect_err("unknown request fields must fail closed");
        assert!(unknown.contains("unknown field"));

        assert_eq!(
            request_from_json("[]").expect_err("request must be an object"),
            "Z-space generation evidence request must be an object"
        );
        assert_eq!(
            report_from_json("[]").expect_err("report must be an object"),
            "Z-space generation evidence report must be an object"
        );
        assert!(bounded_json_value(
            r#"{"protocol_id":"a","protocol_id":"b"}"#,
            "generation evidence",
        )
        .expect_err("duplicate keys fail closed")
        .contains("duplicate JSON object key"));
        assert!(
            bounded_json_value_with_limit(r#"{"samples":[]}"#, "generation evidence", 4,)
                .expect_err("oversized JSON fails before parsing")
                .contains("exceeds WASM ingress budget")
        );
    }

    #[test]
    fn wasm_generation_evidence_types_declare_the_complete_surface() {
        let declarations = include_str!("../types/spiraltorch-wasm.d.ts");

        for symbol in [
            "ZSpaceGenerationEvidenceRequest",
            "ZSpaceGenerationEvidenceReport",
            "zspaceGenerationEvidenceJson",
            "zspaceGenerationEvidenceObject",
            "validateZspaceGenerationEvidenceJson",
            "validateZspaceGenerationEvidenceObject",
        ] {
            assert!(declarations.contains(symbol), "missing {symbol}");
        }
        assert!(declarations.contains("st-core::runtime::zspace_generation_evidence"));
    }
}
