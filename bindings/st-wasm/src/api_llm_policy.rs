use serde_json::Value;
use st_core::runtime::api_llm_route_policy::{
    evaluate_api_llm_route_policy, ApiLlmRoutePolicyError, ApiLlmRoutePolicyEvaluationRequest,
};

#[cfg(target_arch = "wasm32")]
use serde::Serialize;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::js_error;

fn add_execution_client(payload: &mut Value) {
    payload
        .as_object_mut()
        .expect("API LLM route-policy payload is an object")
        .insert(
            "execution_client".to_owned(),
            Value::String("wasm".to_owned()),
        );
}

fn evaluation_request_from_value(
    value: Value,
) -> Result<ApiLlmRoutePolicyEvaluationRequest, String> {
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn evaluation_request_from_json(
    request_json: &str,
) -> Result<ApiLlmRoutePolicyEvaluationRequest, String> {
    let value = serde_json::from_str(request_json).map_err(|error| error.to_string())?;
    evaluation_request_from_value(value)
}

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

/// Evaluate API LLM trace routes through the shared Rust contract.
pub fn api_llm_route_policy_evaluate_value(
    request: ApiLlmRoutePolicyEvaluationRequest,
) -> Result<Value, ApiLlmRoutePolicyError> {
    let mut payload = serde_json::to_value(evaluate_api_llm_route_policy(request)?)
        .expect("API LLM route-policy evaluation payload is serializable");
    add_execution_client(&mut payload);
    Ok(payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = apiLlmRoutePolicyEvaluateJson)]
pub fn api_llm_route_policy_evaluate_json(request_json: &str) -> Result<String, JsValue> {
    let request = evaluation_request_from_json(request_json).map_err(js_error)?;
    let payload = api_llm_route_policy_evaluate_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = apiLlmRoutePolicyEvaluateObject)]
pub fn api_llm_route_policy_evaluate_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let value: Value = serde_wasm_bindgen::from_value(request.clone()).map_err(js_error)?;
    let request = evaluation_request_from_value(value).map_err(js_error)?;
    let payload = api_llm_route_policy_evaluate_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn without_client(mut payload: Value) -> Value {
        payload
            .as_object_mut()
            .expect("API LLM route-policy payload object")
            .remove("execution_client");
        payload
    }

    #[test]
    fn wasm_evaluation_matches_the_rust_contract() {
        let request: ApiLlmRoutePolicyEvaluationRequest = serde_json::from_value(json!({
            "rows": [
                {
                    "label": "fast",
                    "count": 4,
                    "confidence_mean": 0.7,
                    "latency_ms_mean": 80.0,
                    "total_tokens": 32.0,
                    "completion_rate": 1.0,
                    "incomplete_rate": 0.0,
                    "empty_text_rate": 0.0,
                    "refusal_rate": 0.0
                },
                {
                    "label": "grounded",
                    "count": 4,
                    "confidence_mean": 0.8,
                    "text_quality_score": 0.9,
                    "latency_ms_mean": 200.0,
                    "total_tokens": 64.0,
                    "completion_rate": 1.0,
                    "incomplete_rate": 0.0,
                    "empty_text_rate": 0.0,
                    "refusal_rate": 0.0
                }
            ],
            "near_best_tolerance": 0.1
        }))
        .expect("valid evaluation request");
        let expected = serde_json::to_value(
            evaluate_api_llm_route_policy(request.clone()).expect("valid Rust evaluation"),
        )
        .expect("serializable Rust evaluation");
        let actual = without_client(
            api_llm_route_policy_evaluate_value(request).expect("valid WASM evaluation"),
        );

        assert_eq!(actual, expected);
    }

    #[test]
    fn wasm_ingress_fails_closed_on_contract_drift() {
        let unknown_field =
            evaluation_request_from_json(r#"{"rows":[],"route_score_formula":"python"}"#)
                .expect_err("unknown request fields must fail closed");
        assert!(unknown_field.contains("unknown field"));

        let oversized =
            evaluation_request_from_json(r#"{"rows":[{"label":"wide","count":4294967296}]}"#)
                .expect_err("wire integers must be target independent");
        assert!(oversized.contains("u32"));

        let drifted = evaluation_request_from_json(
            r#"{
                "rows":[{
                    "label":"guarded",
                    "count":1,
                    "confidence_mean":0.8,
                    "route_score":1.0,
                    "selection_scores":{"balanced":1.0}
                }]
            }"#,
        )
        .expect("client projections remain decodable transport fields");
        let payload = api_llm_route_policy_evaluate_value(drifted)
            .expect("Rust recomputes client projections");
        assert_ne!(payload["rows"][0]["route_score"], 1.0);
        assert_ne!(payload["rows"][0]["selection_scores"]["balanced"], 1.0);
    }
}
