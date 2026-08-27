// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::Value;
use st_core::runtime::zspace_repetition_unlikelihood::{
    plan_zspace_repetition_unlikelihood, validate_zspace_repetition_unlikelihood_value,
    ZSpaceRepetitionUnlikelihoodError, ZSpaceRepetitionUnlikelihoodRequest,
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

const WASM_REPETITION_UNLIKELIHOOD_MAX_INGRESS_BYTES: u64 = 64 * 1_024 * 1_024;
const WASM_REPETITION_UNLIKELIHOOD_MAX_INGRESS_NODES: u64 = 8_000_000;
const WASM_REPETITION_UNLIKELIHOOD_MAX_INGRESS_DEPTH: u32 = 32;

fn bounded_json_value(input_json: &str, context: &str) -> Result<Value, String> {
    bounded_json_value_with_limit(
        input_json,
        context,
        WASM_REPETITION_UNLIKELIHOOD_MAX_INGRESS_BYTES,
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
        WASM_REPETITION_UNLIKELIHOOD_MAX_INGRESS_NODES,
        WASM_REPETITION_UNLIKELIHOOD_MAX_INGRESS_DEPTH,
        context,
    )
}

#[cfg(target_arch = "wasm32")]
fn snapshot_repetition_js_value(value: &JsValue, context: &str) -> Result<JsValue, JsValue> {
    snapshot_json_compatible_js_value(
        value,
        WASM_REPETITION_UNLIKELIHOOD_MAX_INGRESS_BYTES,
        WASM_REPETITION_UNLIKELIHOOD_MAX_INGRESS_NODES,
        WASM_REPETITION_UNLIKELIHOOD_MAX_INGRESS_DEPTH,
        context,
    )
}

fn request_from_value(value: Value) -> Result<ZSpaceRepetitionUnlikelihoodRequest, String> {
    if !value.is_object() {
        return Err("Z-space repetition-unlikelihood request must be an object".to_owned());
    }
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn request_from_json(input_json: &str) -> Result<ZSpaceRepetitionUnlikelihoodRequest, String> {
    let value = bounded_json_value(input_json, "Z-space repetition-unlikelihood request")?;
    request_from_value(value)
}

fn plan_from_value(value: Value) -> Result<Value, String> {
    if !value.is_object() {
        return Err("Z-space repetition-unlikelihood plan must be an object".to_owned());
    }
    Ok(value)
}

fn plan_from_json(input_json: &str) -> Result<Value, String> {
    let value = bounded_json_value(input_json, "Z-space repetition-unlikelihood plan")?;
    plan_from_value(value)
}

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

pub fn zspace_repetition_unlikelihood_plan_value(
    request: ZSpaceRepetitionUnlikelihoodRequest,
) -> Result<Value, ZSpaceRepetitionUnlikelihoodError> {
    Ok(
        serde_json::to_value(plan_zspace_repetition_unlikelihood(request)?)
            .expect("Z-space repetition-unlikelihood plan is serializable"),
    )
}

pub fn validate_zspace_repetition_unlikelihood_plan_value(
    plan: Value,
) -> Result<Value, ZSpaceRepetitionUnlikelihoodError> {
    Ok(
        serde_json::to_value(validate_zspace_repetition_unlikelihood_value(plan)?)
            .expect("Z-space repetition-unlikelihood plan is serializable"),
    )
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceRepetitionUnlikelihoodPlanJson)]
pub fn zspace_repetition_unlikelihood_plan_json(
    request_json: &JsString,
) -> Result<String, JsValue> {
    let request_json = bounded_json_string_from_js(
        request_json,
        WASM_REPETITION_UNLIKELIHOOD_MAX_INGRESS_BYTES,
        "Z-space repetition-unlikelihood request JSON",
    )?;
    let request = request_from_json(&request_json).map_err(js_error)?;
    let plan = zspace_repetition_unlikelihood_plan_value(request).map_err(js_error)?;
    serde_json::to_string(&plan).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = zspaceRepetitionUnlikelihoodPlanObject)]
pub fn zspace_repetition_unlikelihood_plan_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let request = snapshot_repetition_js_value(request, "Z-space repetition-unlikelihood request")?;
    let request = serde_wasm_bindgen::from_value::<Value>(request).map_err(js_error)?;
    let request = request_from_value(request).map_err(js_error)?;
    let plan = zspace_repetition_unlikelihood_plan_value(request).map_err(js_error)?;
    to_json_compatible_js(&plan)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = validateZspaceRepetitionUnlikelihoodPlanJson)]
pub fn validate_zspace_repetition_unlikelihood_plan_json(
    plan_json: &JsString,
) -> Result<String, JsValue> {
    let plan_json = bounded_json_string_from_js(
        plan_json,
        WASM_REPETITION_UNLIKELIHOOD_MAX_INGRESS_BYTES,
        "Z-space repetition-unlikelihood plan JSON",
    )?;
    let plan = plan_from_json(&plan_json).map_err(js_error)?;
    let plan = validate_zspace_repetition_unlikelihood_plan_value(plan).map_err(js_error)?;
    serde_json::to_string(&plan).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = validateZspaceRepetitionUnlikelihoodPlanObject)]
pub fn validate_zspace_repetition_unlikelihood_plan_object(
    plan: &JsValue,
) -> Result<JsValue, JsValue> {
    let plan = snapshot_repetition_js_value(plan, "Z-space repetition-unlikelihood plan")?;
    let plan = serde_wasm_bindgen::from_value::<Value>(plan).map_err(js_error)?;
    let plan = plan_from_value(plan).map_err(js_error)?;
    let plan = validate_zspace_repetition_unlikelihood_plan_value(plan).map_err(js_error)?;
    to_json_compatible_js(&plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use st_core::runtime::zspace_repetition_unlikelihood::{
        plan_zspace_repetition_unlikelihood, ZSpaceRepetitionUnlikelihoodCandidateSource,
        ZSpaceRepetitionUnlikelihoodConfig, ZSpaceRepetitionUnlikelihoodSequence,
        ZSPACE_REPETITION_UNLIKELIHOOD_MAX_MATERIALIZED_PLAN_BYTES,
        ZSPACE_REPETITION_UNLIKELIHOOD_MAX_WORK_UNITS,
    };

    fn request() -> ZSpaceRepetitionUnlikelihoodRequest {
        serde_json::from_value(json!({
            "config": {
                "strength": 1.0,
                "candidate_source": {"kind": "prior_continuation", "ngram_order": 3},
                "context_window": 16,
                "max_candidates_per_position": 8
            },
            "sequences": [{
                "token_ids": [1, 2, 3, 1, 2, 4],
                "token_mask": [true, true, true, true, true, true],
                "label_mask": [true, true, true, true, true, true]
            }]
        }))
        .expect("valid request")
    }

    fn over_budget_request() -> ZSpaceRepetitionUnlikelihoodRequest {
        let token_count = 6_000usize;
        ZSpaceRepetitionUnlikelihoodRequest {
            config: ZSpaceRepetitionUnlikelihoodConfig {
                strength: 0.1,
                candidate_source: ZSpaceRepetitionUnlikelihoodCandidateSource::PriorContinuation {
                    ngram_order: 3,
                },
                context_window: token_count,
                max_candidates_per_position: 8,
            },
            sequences: vec![ZSpaceRepetitionUnlikelihoodSequence {
                token_ids: (0..token_count as u64).collect(),
                token_mask: vec![true; token_count],
                label_mask: vec![true; token_count],
                proposal_token_ids: None,
            }],
        }
    }

    fn materialization_heavy_request() -> ZSpaceRepetitionUnlikelihoodRequest {
        let token_count = 100_000usize;
        let mut remaining = token_count;
        let mut sequences = Vec::new();
        while remaining > 0 {
            let sequence_token_count = remaining.min(16_384);
            let token_ids = (0..sequence_token_count)
                .map(|index| (index % 2) as u64)
                .collect::<Vec<_>>();
            let mut proposal_rows = vec![Vec::new(); sequence_token_count];
            for target_index in 1..sequence_token_count {
                proposal_rows[target_index] = vec![token_ids[target_index - 1]];
            }
            sequences.push(ZSpaceRepetitionUnlikelihoodSequence {
                token_ids,
                token_mask: vec![true; sequence_token_count],
                label_mask: vec![true; sequence_token_count],
                proposal_token_ids: Some(proposal_rows),
            });
            remaining -= sequence_token_count;
        }
        ZSpaceRepetitionUnlikelihoodRequest {
            config: ZSpaceRepetitionUnlikelihoodConfig {
                strength: 0.1,
                candidate_source: ZSpaceRepetitionUnlikelihoodCandidateSource::ModelTopkHistory {
                    proposal_top_k: 1,
                },
                context_window: 1,
                max_candidates_per_position: 1,
            },
            sequences,
        }
    }

    #[test]
    fn wasm_repetition_plan_matches_rust_exactly() {
        let request = request();
        let expected = serde_json::to_value(
            plan_zspace_repetition_unlikelihood(request.clone()).expect("Rust plan"),
        )
        .expect("serializable Rust plan");
        let actual = zspace_repetition_unlikelihood_plan_value(request).expect("WASM plan");

        assert_eq!(actual, expected);
        assert_eq!(
            actual["contract_version"],
            "spiraltorch.zspace_repetition_unlikelihood.v3"
        );
        assert_eq!(actual["semantic_backend"], "rust");
        assert_eq!(actual["aggregate"]["candidate_count"], 1);
    }

    #[test]
    fn wasm_repetition_validator_accepts_browser_numbers_and_rejects_tampering() {
        let canonical = zspace_repetition_unlikelihood_plan_value(request()).expect("plan");
        let mut stored = canonical.clone();
        stored["request"]["config"]["strength"] = json!(1);
        stored["aggregate"]["mean_candidates_per_active_position"] = json!(1);
        assert_eq!(
            validate_zspace_repetition_unlikelihood_plan_value(stored)
                .expect("browser JSON numeric round trip"),
            canonical
        );

        let mut tampered = canonical;
        tampered["positions"][0]["candidates"][0]["token_id"] = json!(7);
        let error = validate_zspace_repetition_unlikelihood_plan_value(tampered)
            .expect_err("tampered plan must fail");
        assert!(matches!(
            error,
            ZSpaceRepetitionUnlikelihoodError::MalformedPlan { .. }
        ));
    }

    #[test]
    fn wasm_repetition_ingress_fails_closed() {
        let unknown = request_from_json(
            r#"{"config":{"strength":0.1,"candidate_source":{"kind":"prior_continuation","ngram_order":3},"context_window":16,"max_candidates_per_position":8},"sequences":[{"token_ids":[1,2,3],"token_mask":[true,true,true],"label_mask":[true,true,true]}],"browser_rule":"local"}"#,
        )
        .expect_err("unknown request fields must fail closed");
        assert!(unknown.contains("unknown field"));
        assert_eq!(
            request_from_json("[]").expect_err("request must be an object"),
            "Z-space repetition-unlikelihood request must be an object"
        );
        assert_eq!(
            plan_from_json("[]").expect_err("plan must be an object"),
            "Z-space repetition-unlikelihood plan must be an object"
        );
    }

    #[test]
    fn wasm_repetition_json_ingress_is_bounded_before_parsing() {
        let error = bounded_json_value("{", "test plan")
            .expect_err("malformed JSON should fail after the production size preflight");
        assert!(error.contains("EOF"));

        let oversized_error = bounded_json_value_with_limit("{}", "test plan", 1)
            .expect_err("byte limit must run before JSON parsing");
        assert_eq!(
            oversized_error,
            "test plan exceeds WASM ingress budget of 1 bytes"
        );
    }

    #[test]
    fn wasm_repetition_path_enforces_the_rust_work_budget() {
        let error = zspace_repetition_unlikelihood_plan_value(over_budget_request())
            .expect_err("Rust work budget must protect the WASM path");

        assert!(matches!(
            error,
            ZSpaceRepetitionUnlikelihoodError::WorkBudgetExceeded {
                maximum_work_units: ZSPACE_REPETITION_UNLIKELIHOOD_MAX_WORK_UNITS,
                ..
            }
        ));
    }

    #[test]
    fn wasm_repetition_validator_never_bypasses_the_rust_work_budget() {
        let forged = json!({
            "request": serde_json::to_value(over_budget_request()).expect("request value")
        });
        let error = validate_zspace_repetition_unlikelihood_plan_value(forged)
            .expect_err("untrusted browser replay must be bounded");

        assert!(matches!(
            error,
            ZSpaceRepetitionUnlikelihoodError::WorkBudgetExceeded {
                maximum_work_units: ZSPACE_REPETITION_UNLIKELIHOOD_MAX_WORK_UNITS,
                ..
            }
        ));
    }

    #[test]
    fn wasm_repetition_paths_reject_materialization_heavy_requests() {
        let request = materialization_heavy_request();
        let projected_positions = 100_000 - request.sequences.len() as u64;
        let error = zspace_repetition_unlikelihood_plan_value(request.clone())
            .expect_err("browser planning must be materialization-bounded");
        let ZSpaceRepetitionUnlikelihoodError::MaterializedPlanBudgetExceeded {
            maximum_bytes,
            estimated_positions,
            estimated_candidates,
            ..
        } = error
        else {
            panic!("unexpected error: {error}");
        };
        assert_eq!(
            maximum_bytes,
            ZSPACE_REPETITION_UNLIKELIHOOD_MAX_MATERIALIZED_PLAN_BYTES
        );
        assert_eq!(estimated_positions, projected_positions);
        assert_eq!(estimated_candidates, projected_positions);

        let forged = json!({
            "request": serde_json::to_value(request).expect("request value")
        });
        assert!(matches!(
            validate_zspace_repetition_unlikelihood_plan_value(forged),
            Err(ZSpaceRepetitionUnlikelihoodError::MaterializedPlanBudgetExceeded { .. })
        ));
    }

    #[test]
    fn wasm_repetition_types_declare_the_complete_surface() {
        let declarations = include_str!("../types/spiraltorch-wasm.d.ts");

        for symbol in [
            "ZSpaceRepetitionUnlikelihoodRequest",
            "ZSpaceRepetitionUnlikelihoodPlan",
            "zspaceRepetitionUnlikelihoodPlanJson",
            "zspaceRepetitionUnlikelihoodPlanObject",
            "validateZspaceRepetitionUnlikelihoodPlanJson",
            "validateZspaceRepetitionUnlikelihoodPlanObject",
        ] {
            assert!(declarations.contains(symbol), "missing {symbol}");
        }
        assert!(declarations.contains("st-core::runtime::zspace_repetition_unlikelihood"));
    }
}
