use serde_json::Value;
use st_core::runtime::zspace_generation_evidence::{
    summarize_zspace_generation_evidence, validate_zspace_generation_evidence_value,
    ZSpaceGenerationEvidenceError, ZSpaceGenerationEvidenceRequest,
};

#[cfg(target_arch = "wasm32")]
use serde::Serialize;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::js_error;

fn request_from_value(value: Value) -> Result<ZSpaceGenerationEvidenceRequest, String> {
    if !value.is_object() {
        return Err("Z-space generation evidence request must be an object".to_owned());
    }
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn request_from_json(input_json: &str) -> Result<ZSpaceGenerationEvidenceRequest, String> {
    let value = serde_json::from_str(input_json).map_err(|error| error.to_string())?;
    request_from_value(value)
}

fn report_from_value(value: Value) -> Result<Value, String> {
    if !value.is_object() {
        return Err("Z-space generation evidence report must be an object".to_owned());
    }
    Ok(value)
}

fn report_from_json(input_json: &str) -> Result<Value, String> {
    let value = serde_json::from_str(input_json).map_err(|error| error.to_string())?;
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
pub fn zspace_generation_evidence_json(request_json: &str) -> Result<String, JsValue> {
    let request = request_from_json(request_json).map_err(js_error)?;
    let report = zspace_generation_evidence_value(request).map_err(js_error)?;
    serde_json::to_string(&report).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceGenerationEvidenceObject)]
pub fn zspace_generation_evidence_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let request = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request = request_from_value(request).map_err(js_error)?;
    let report = zspace_generation_evidence_value(request).map_err(js_error)?;
    to_json_compatible_js(&report)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = validateZspaceGenerationEvidenceJson)]
pub fn validate_zspace_generation_evidence_json(report_json: &str) -> Result<String, JsValue> {
    let report = report_from_json(report_json).map_err(js_error)?;
    let report = validate_zspace_generation_evidence_report_value(report).map_err(js_error)?;
    serde_json::to_string(&report).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = validateZspaceGenerationEvidenceObject)]
pub fn validate_zspace_generation_evidence_object(report: &JsValue) -> Result<JsValue, JsValue> {
    let report = serde_wasm_bindgen::from_value::<Value>(report.clone()).map_err(js_error)?;
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
