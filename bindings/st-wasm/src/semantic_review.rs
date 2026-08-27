// SPDX-License-Identifier: AGPL-3.0-or-later

use serde::de::DeserializeOwned;
use serde_json::Value;
use st_core::runtime::zspace_semantic_review::{
    seal_zspace_semantic_review_packet, summarize_zspace_semantic_review_draft,
    unblind_zspace_semantic_review, validate_zspace_semantic_review_draft_receipt_value,
    validate_zspace_semantic_review_packet, validate_zspace_semantic_review_packet_receipt_value,
    validate_zspace_semantic_review_unblind_value, zspace_semantic_review_map_id,
    ZSpaceSemanticReviewDraftRequest, ZSpaceSemanticReviewError, ZSpaceSemanticReviewMapIdRequest,
    ZSpaceSemanticReviewPacket, ZSpaceSemanticReviewPacketRequest,
    ZSpaceSemanticReviewUnblindRequest,
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

const WASM_SEMANTIC_REVIEW_MAX_INGRESS_BYTES: u64 = 64 * 1_024 * 1_024;
const WASM_SEMANTIC_REVIEW_MAX_INGRESS_NODES: u64 = 1_000_000;
const WASM_SEMANTIC_REVIEW_MAX_INGRESS_DEPTH: u32 = 32;

fn typed_from_value<T: DeserializeOwned>(value: Value, context: &str) -> Result<T, String> {
    if !value.is_object() {
        return Err(format!("{context} must be an object"));
    }
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn typed_from_json<T: DeserializeOwned>(input_json: &str, context: &str) -> Result<T, String> {
    let value = bounded_json_value_from_str(
        input_json,
        WASM_SEMANTIC_REVIEW_MAX_INGRESS_BYTES,
        WASM_SEMANTIC_REVIEW_MAX_INGRESS_NODES,
        WASM_SEMANTIC_REVIEW_MAX_INGRESS_DEPTH,
        context,
    )?;
    typed_from_value(value, context)
}

fn report_from_value(value: Value, context: &str) -> Result<Value, String> {
    if value.is_object() {
        Ok(value)
    } else {
        Err(format!("{context} must be an object"))
    }
}

fn report_from_json(input_json: &str, context: &str) -> Result<Value, String> {
    let value = bounded_json_value_from_str(
        input_json,
        WASM_SEMANTIC_REVIEW_MAX_INGRESS_BYTES,
        WASM_SEMANTIC_REVIEW_MAX_INGRESS_NODES,
        WASM_SEMANTIC_REVIEW_MAX_INGRESS_DEPTH,
        context,
    )?;
    report_from_value(value, context)
}

fn encoded<T: serde::Serialize>(value: T) -> Value {
    serde_json::to_value(value).expect("Z-space semantic-review artifacts are serializable")
}

pub fn seal_zspace_semantic_review_packet_value(
    request: ZSpaceSemanticReviewPacketRequest,
) -> Result<Value, ZSpaceSemanticReviewError> {
    Ok(encoded(seal_zspace_semantic_review_packet(request)?))
}

pub fn zspace_semantic_review_map_id_value(
    request: ZSpaceSemanticReviewMapIdRequest,
) -> Result<String, ZSpaceSemanticReviewError> {
    zspace_semantic_review_map_id(request.entries)
}

pub fn validate_zspace_semantic_review_packet_value(
    packet: ZSpaceSemanticReviewPacket,
) -> Result<Value, ZSpaceSemanticReviewError> {
    Ok(encoded(validate_zspace_semantic_review_packet(packet)?))
}

pub fn validate_zspace_semantic_review_packet_receipt_report_value(
    receipt: Value,
) -> Result<Value, ZSpaceSemanticReviewError> {
    Ok(encoded(
        validate_zspace_semantic_review_packet_receipt_value(receipt)?,
    ))
}

pub fn summarize_zspace_semantic_review_draft_value(
    request: ZSpaceSemanticReviewDraftRequest,
) -> Result<Value, ZSpaceSemanticReviewError> {
    Ok(encoded(summarize_zspace_semantic_review_draft(
        request.packet,
        request.draft,
    )?))
}

pub fn validate_zspace_semantic_review_draft_receipt_report_value(
    receipt: Value,
) -> Result<Value, ZSpaceSemanticReviewError> {
    Ok(encoded(
        validate_zspace_semantic_review_draft_receipt_value(receipt)?,
    ))
}

pub fn unblind_zspace_semantic_review_value(
    request: ZSpaceSemanticReviewUnblindRequest,
) -> Result<Value, ZSpaceSemanticReviewError> {
    Ok(encoded(unblind_zspace_semantic_review(
        request.packet,
        request.draft,
        request.blinding_map,
    )?))
}

pub fn validate_zspace_semantic_review_unblind_report_value(
    report: Value,
) -> Result<Value, ZSpaceSemanticReviewError> {
    Ok(encoded(validate_zspace_semantic_review_unblind_value(
        report,
    )?))
}

#[cfg(target_arch = "wasm32")]
fn snapshot_semantic_review_js_value(value: &JsValue, context: &str) -> Result<JsValue, JsValue> {
    snapshot_json_compatible_js_value(
        value,
        WASM_SEMANTIC_REVIEW_MAX_INGRESS_BYTES,
        WASM_SEMANTIC_REVIEW_MAX_INGRESS_NODES,
        WASM_SEMANTIC_REVIEW_MAX_INGRESS_DEPTH,
        context,
    )
}

#[cfg(target_arch = "wasm32")]
fn string_from_js(value: &JsString, context: &str) -> Result<String, JsValue> {
    bounded_json_string_from_js(value, WASM_SEMANTIC_REVIEW_MAX_INGRESS_BYTES, context)
}

#[cfg(target_arch = "wasm32")]
fn value_from_js(value: &JsValue, context: &str) -> Result<Value, JsValue> {
    let snapshot = snapshot_semantic_review_js_value(value, context)?;
    serde_wasm_bindgen::from_value::<Value>(snapshot).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
fn value_to_json(value: &Value) -> Result<String, JsValue> {
    serde_json::to_string(value).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = sealZspaceSemanticReviewPacketJson)]
pub fn seal_zspace_semantic_review_packet_json(request_json: &JsString) -> Result<String, JsValue> {
    let input = string_from_js(request_json, "Z-space semantic-review packet request JSON")?;
    let request =
        typed_from_json(&input, "Z-space semantic-review packet request").map_err(js_error)?;
    value_to_json(&seal_zspace_semantic_review_packet_value(request).map_err(js_error)?)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = sealZspaceSemanticReviewPacketObject)]
pub fn seal_zspace_semantic_review_packet_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let request = typed_from_value(
        value_from_js(request, "Z-space semantic-review packet request")?,
        "Z-space semantic-review packet request",
    )
    .map_err(js_error)?;
    to_json_compatible_js(&seal_zspace_semantic_review_packet_value(request).map_err(js_error)?)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceSemanticReviewMapIdJson)]
pub fn zspace_semantic_review_map_id_json(request_json: &JsString) -> Result<String, JsValue> {
    let input = string_from_js(request_json, "Z-space semantic-review map request JSON")?;
    let request =
        typed_from_json(&input, "Z-space semantic-review map request").map_err(js_error)?;
    zspace_semantic_review_map_id_value(request).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceSemanticReviewMapIdObject)]
pub fn zspace_semantic_review_map_id_object(request: &JsValue) -> Result<String, JsValue> {
    let request = typed_from_value(
        value_from_js(request, "Z-space semantic-review map request")?,
        "Z-space semantic-review map request",
    )
    .map_err(js_error)?;
    zspace_semantic_review_map_id_value(request).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = validateZspaceSemanticReviewPacketJson)]
pub fn validate_zspace_semantic_review_packet_json(
    packet_json: &JsString,
) -> Result<String, JsValue> {
    let input = string_from_js(packet_json, "Z-space semantic-review packet JSON")?;
    let packet = typed_from_json(&input, "Z-space semantic-review packet").map_err(js_error)?;
    value_to_json(&validate_zspace_semantic_review_packet_value(packet).map_err(js_error)?)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = validateZspaceSemanticReviewPacketObject)]
pub fn validate_zspace_semantic_review_packet_object(packet: &JsValue) -> Result<JsValue, JsValue> {
    let packet = typed_from_value(
        value_from_js(packet, "Z-space semantic-review packet")?,
        "Z-space semantic-review packet",
    )
    .map_err(js_error)?;
    to_json_compatible_js(&validate_zspace_semantic_review_packet_value(packet).map_err(js_error)?)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = validateZspaceSemanticReviewPacketReceiptJson)]
pub fn validate_zspace_semantic_review_packet_receipt_json(
    receipt_json: &JsString,
) -> Result<String, JsValue> {
    let input = string_from_js(receipt_json, "Z-space semantic-review packet receipt JSON")?;
    let receipt =
        report_from_json(&input, "Z-space semantic-review packet receipt").map_err(js_error)?;
    value_to_json(
        &validate_zspace_semantic_review_packet_receipt_report_value(receipt).map_err(js_error)?,
    )
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = validateZspaceSemanticReviewPacketReceiptObject)]
pub fn validate_zspace_semantic_review_packet_receipt_object(
    receipt: &JsValue,
) -> Result<JsValue, JsValue> {
    let receipt = report_from_value(
        value_from_js(receipt, "Z-space semantic-review packet receipt")?,
        "Z-space semantic-review packet receipt",
    )
    .map_err(js_error)?;
    to_json_compatible_js(
        &validate_zspace_semantic_review_packet_receipt_report_value(receipt).map_err(js_error)?,
    )
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = summarizeZspaceSemanticReviewDraftJson)]
pub fn summarize_zspace_semantic_review_draft_json(
    request_json: &JsString,
) -> Result<String, JsValue> {
    let input = string_from_js(request_json, "Z-space semantic-review draft request JSON")?;
    let request =
        typed_from_json(&input, "Z-space semantic-review draft request").map_err(js_error)?;
    value_to_json(&summarize_zspace_semantic_review_draft_value(request).map_err(js_error)?)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = summarizeZspaceSemanticReviewDraftObject)]
pub fn summarize_zspace_semantic_review_draft_object(
    request: &JsValue,
) -> Result<JsValue, JsValue> {
    let request = typed_from_value(
        value_from_js(request, "Z-space semantic-review draft request")?,
        "Z-space semantic-review draft request",
    )
    .map_err(js_error)?;
    to_json_compatible_js(&summarize_zspace_semantic_review_draft_value(request).map_err(js_error)?)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = validateZspaceSemanticReviewDraftReceiptJson)]
pub fn validate_zspace_semantic_review_draft_receipt_json(
    receipt_json: &JsString,
) -> Result<String, JsValue> {
    let input = string_from_js(receipt_json, "Z-space semantic-review draft receipt JSON")?;
    let receipt =
        report_from_json(&input, "Z-space semantic-review draft receipt").map_err(js_error)?;
    value_to_json(
        &validate_zspace_semantic_review_draft_receipt_report_value(receipt).map_err(js_error)?,
    )
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = validateZspaceSemanticReviewDraftReceiptObject)]
pub fn validate_zspace_semantic_review_draft_receipt_object(
    receipt: &JsValue,
) -> Result<JsValue, JsValue> {
    let receipt = report_from_value(
        value_from_js(receipt, "Z-space semantic-review draft receipt")?,
        "Z-space semantic-review draft receipt",
    )
    .map_err(js_error)?;
    to_json_compatible_js(
        &validate_zspace_semantic_review_draft_receipt_report_value(receipt).map_err(js_error)?,
    )
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = unblindZspaceSemanticReviewJson)]
pub fn unblind_zspace_semantic_review_json(request_json: &JsString) -> Result<String, JsValue> {
    let input = string_from_js(request_json, "Z-space semantic-review unblind request JSON")?;
    let request =
        typed_from_json(&input, "Z-space semantic-review unblind request").map_err(js_error)?;
    value_to_json(&unblind_zspace_semantic_review_value(request).map_err(js_error)?)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = unblindZspaceSemanticReviewObject)]
pub fn unblind_zspace_semantic_review_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let request = typed_from_value(
        value_from_js(request, "Z-space semantic-review unblind request")?,
        "Z-space semantic-review unblind request",
    )
    .map_err(js_error)?;
    to_json_compatible_js(&unblind_zspace_semantic_review_value(request).map_err(js_error)?)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = validateZspaceSemanticReviewUnblindJson)]
pub fn validate_zspace_semantic_review_unblind_json(
    report_json: &JsString,
) -> Result<String, JsValue> {
    let input = string_from_js(report_json, "Z-space semantic-review unblind report JSON")?;
    let report =
        report_from_json(&input, "Z-space semantic-review unblind report").map_err(js_error)?;
    value_to_json(&validate_zspace_semantic_review_unblind_report_value(report).map_err(js_error)?)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = validateZspaceSemanticReviewUnblindObject)]
pub fn validate_zspace_semantic_review_unblind_object(
    report: &JsValue,
) -> Result<JsValue, JsValue> {
    let report = report_from_value(
        value_from_js(report, "Z-space semantic-review unblind report")?,
        "Z-space semantic-review unblind report",
    )
    .map_err(js_error)?;
    to_json_compatible_js(
        &validate_zspace_semantic_review_unblind_report_value(report).map_err(js_error)?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn identity(character: char) -> String {
        format!("sha256:{}", character.to_string().repeat(64))
    }

    fn map_request() -> ZSpaceSemanticReviewMapIdRequest {
        typed_from_value(
            json!({
                "entries": [
                    {
                        "group_id": identity('2'),
                        "seed": 17,
                        "prompt_id": identity('4'),
                        "candidate_to_arm": {"A": "periodic", "B": "baseline", "C": "history"}
                    },
                    {
                        "group_id": identity('1'),
                        "seed": 13,
                        "prompt_id": identity('3'),
                        "candidate_to_arm": {"A": "baseline", "B": "history", "C": "periodic"}
                    }
                ]
            }),
            "map request",
        )
        .expect("valid map request")
    }

    fn packet_request() -> ZSpaceSemanticReviewPacketRequest {
        let map_id = zspace_semantic_review_map_id_value(map_request()).expect("map id");
        typed_from_value(
            json!({
                "protocol_id": identity('a'),
                "prompt_set_id": identity('b'),
                "blinding_key_sha256": "c".repeat(64),
                "blinding_map_id": map_id,
                "instructions": "Score while blind.",
                "rubric": {
                    "fluency": "integer 1 through 5",
                    "prompt_relevance": "integer 1 through 5",
                    "local_coherence": "integer 1 through 5",
                    "non_repetition": "integer 1 through 5",
                    "preference": "A, B, C, or tie"
                },
                "groups": [
                    {
                        "group_id": identity('1'),
                        "prompt": "A prompt",
                        "candidates": [
                            {"candidate_label": "A", "continuation": "one"},
                            {"candidate_label": "B", "continuation": "two"},
                            {"candidate_label": "C", "continuation": "three"}
                        ]
                    },
                    {
                        "group_id": identity('2'),
                        "prompt": "Another prompt",
                        "candidates": [
                            {"candidate_label": "A", "continuation": "four"},
                            {"candidate_label": "B", "continuation": "five"},
                            {"candidate_label": "C", "continuation": "six"}
                        ]
                    }
                ]
            }),
            "packet request",
        )
        .expect("valid packet request")
    }

    fn complete_draft_request(packet: &Value) -> ZSpaceSemanticReviewDraftRequest {
        typed_from_value(
            json!({
                "packet": packet,
                "draft": {
                    "schema": "spiraltorch.hf_blinded_semantic_review_draft.v1",
                    "packet_id": packet["packet_id"],
                    "reviewer_id": identity('d'),
                    "review_session_id": identity('e'),
                    "responses": [
                        {
                            "group_id": identity('1'),
                            "scores": [
                                {"candidate_label": "C", "fluency": 1, "prompt_relevance": 1, "local_coherence": 1, "non_repetition": 1},
                                {"candidate_label": "A", "fluency": 5, "prompt_relevance": 5, "local_coherence": 5, "non_repetition": 5},
                                {"candidate_label": "B", "fluency": 3, "prompt_relevance": 3, "local_coherence": 3, "non_repetition": 3}
                            ],
                            "preference": "A"
                        },
                        {
                            "group_id": identity('2'),
                            "scores": [
                                {"candidate_label": "A", "fluency": 4, "prompt_relevance": 4, "local_coherence": 4, "non_repetition": 4},
                                {"candidate_label": "B", "fluency": 2, "prompt_relevance": 2, "local_coherence": 2, "non_repetition": 2},
                                {"candidate_label": "C", "fluency": 3, "prompt_relevance": 3, "local_coherence": 3, "non_repetition": 3}
                            ],
                            "preference": "tie"
                        }
                    ]
                }
            }),
            "draft request",
        )
        .expect("valid draft request")
    }

    #[test]
    fn wasm_semantic_review_lifecycle_matches_rust_core() {
        let request = packet_request();
        let expected = encoded(
            seal_zspace_semantic_review_packet(request.clone()).expect("Rust packet receipt"),
        );
        let receipt =
            seal_zspace_semantic_review_packet_value(request).expect("WASM packet receipt");
        assert_eq!(receipt, expected);
        assert_eq!(receipt["semantic_backend"], "rust");

        let packet = receipt["packet"].clone();
        assert_eq!(
            validate_zspace_semantic_review_packet_receipt_report_value(receipt.clone())
                .expect("packet receipt replay"),
            receipt
        );
        let draft_request = complete_draft_request(&packet);
        let draft_receipt = summarize_zspace_semantic_review_draft_value(draft_request.clone())
            .expect("draft receipt");
        assert_eq!(draft_receipt["status"], "ready_for_unblind");

        let unblind_request = typed_from_value(
            json!({
                "packet": packet,
                "draft": draft_request.draft,
                "blinding_map": {
                    "schema": "spiraltorch.hf_blinded_semantic_review_map.v1",
                    "status": "sealed_pending_review",
                    "protocol_id": identity('a'),
                    "packet_id": draft_receipt["packet_id"],
                    "blinding_key_sha256": "c".repeat(64),
                    "entries": map_request().entries
                }
            }),
            "unblind request",
        )
        .expect("valid unblind request");
        let report = unblind_zspace_semantic_review_value(unblind_request).expect("unblind report");
        assert_eq!(report["status"], "unblinded");
        assert_eq!(report["arm_count"], 3);
        assert_eq!(
            validate_zspace_semantic_review_unblind_report_value(report.clone())
                .expect("unblind replay"),
            report
        );
    }

    #[test]
    fn wasm_semantic_review_replay_accepts_browser_number_spelling() {
        let receipt = seal_zspace_semantic_review_packet_value(packet_request()).expect("packet");
        let request = complete_draft_request(&receipt["packet"]);
        let canonical = summarize_zspace_semantic_review_draft_value(request).expect("draft");
        let mut browser = canonical.clone();
        browser["completion_ratio"] = json!(1);
        assert_eq!(
            validate_zspace_semantic_review_draft_receipt_report_value(browser)
                .expect("browser numeric spelling"),
            canonical
        );
    }

    #[test]
    fn wasm_semantic_review_ingress_fails_closed() {
        assert_eq!(
            typed_from_json::<ZSpaceSemanticReviewMapIdRequest>("[]", "map")
                .expect_err("map must be an object"),
            "map must be an object"
        );
        let unknown =
            serde_json::to_string(&json!({"entries": [], "client_only": true})).expect("JSON");
        assert!(
            typed_from_json::<ZSpaceSemanticReviewMapIdRequest>(&unknown, "map")
                .expect_err("unknown fields fail")
                .contains("unknown field")
        );
        assert!(matches!(
            zspace_semantic_review_map_id_value(ZSpaceSemanticReviewMapIdRequest {
                entries: Vec::new()
            }),
            Err(ZSpaceSemanticReviewError::EmptyMap)
        ));
    }

    #[test]
    fn wasm_semantic_review_types_declare_the_complete_surface() {
        let declarations = include_str!("../types/spiraltorch-wasm.d.ts");
        for symbol in [
            "ZSpaceSemanticReviewPacketRequest",
            "ZSpaceSemanticReviewPacketReceipt",
            "ZSpaceSemanticReviewDraftReceipt",
            "ZSpaceSemanticReviewUnblindReport",
            "sealZspaceSemanticReviewPacketJson",
            "zspaceSemanticReviewMapIdObject",
            "validateZspaceSemanticReviewPacketReceiptObject",
            "summarizeZspaceSemanticReviewDraftObject",
            "validateZspaceSemanticReviewDraftReceiptObject",
            "unblindZspaceSemanticReviewObject",
            "validateZspaceSemanticReviewUnblindObject",
        ] {
            assert!(declarations.contains(symbol), "missing {symbol}");
        }
        assert!(declarations.contains("st-core::runtime::zspace_semantic_review"));
        assert!(declarations.contains("packet_validated: true;\n        status: \"ready\";"));
    }
}
