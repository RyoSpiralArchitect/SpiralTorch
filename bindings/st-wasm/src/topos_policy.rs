use serde_json::Value;
use st_core::runtime::topos_route_policy::{
    build_topos_route_rewards, evaluate_topos_route_policy, resolve_topos_route_policy,
    ToposRoutePolicyError, ToposRoutePolicyEvaluationRequest, ToposRoutePolicyResolveRequest,
    ToposRouteRewardsRequest,
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
        .expect("Topos route-policy payload is an object")
        .insert(
            "execution_client".to_owned(),
            Value::String("wasm".to_owned()),
        );
}

fn evaluation_request_from_value(
    value: Value,
) -> Result<ToposRoutePolicyEvaluationRequest, String> {
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn evaluation_request_from_json(
    request_json: &str,
) -> Result<ToposRoutePolicyEvaluationRequest, String> {
    let value = serde_json::from_str(request_json).map_err(|error| error.to_string())?;
    evaluation_request_from_value(value)
}

fn rewards_request_from_value(value: Value) -> Result<ToposRouteRewardsRequest, String> {
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn rewards_request_from_json(request_json: &str) -> Result<ToposRouteRewardsRequest, String> {
    let value = serde_json::from_str(request_json).map_err(|error| error.to_string())?;
    rewards_request_from_value(value)
}

fn resolve_request_from_value(value: Value) -> Result<ToposRoutePolicyResolveRequest, String> {
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn resolve_request_from_json(request_json: &str) -> Result<ToposRoutePolicyResolveRequest, String> {
    let value = serde_json::from_str(request_json).map_err(|error| error.to_string())?;
    resolve_request_from_value(value)
}

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

/// Evaluate route-policy profiles through the shared Rust contract.
pub fn topos_route_policy_evaluate_value(
    request: ToposRoutePolicyEvaluationRequest,
) -> Result<Value, ToposRoutePolicyError> {
    let mut payload = serde_json::to_value(evaluate_topos_route_policy(request)?)
        .expect("Topos route-policy evaluation payload is serializable");
    add_execution_client(&mut payload);
    Ok(payload)
}

/// Project route rewards through the shared Rust contract.
pub fn topos_route_policy_rewards_value(
    request: ToposRouteRewardsRequest,
) -> Result<Value, ToposRoutePolicyError> {
    let mut payload = serde_json::to_value(build_topos_route_rewards(request)?)
        .expect("Topos route-reward payload is serializable");
    add_execution_client(&mut payload);
    Ok(payload)
}

/// Resolve a selected route through the shared Rust contract.
pub fn topos_route_policy_resolve_value(
    request: ToposRoutePolicyResolveRequest,
) -> Result<Value, ToposRoutePolicyError> {
    let mut payload = serde_json::to_value(resolve_topos_route_policy(request)?)
        .expect("Topos route-policy resolution payload is serializable");
    add_execution_client(&mut payload);
    Ok(payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = toposRoutePolicyEvaluateJson)]
pub fn topos_route_policy_evaluate_json(request_json: &str) -> Result<String, JsValue> {
    let request = evaluation_request_from_json(request_json).map_err(js_error)?;
    let payload = topos_route_policy_evaluate_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = toposRoutePolicyEvaluateObject)]
pub fn topos_route_policy_evaluate_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let request = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request = evaluation_request_from_value(request).map_err(js_error)?;
    let payload = topos_route_policy_evaluate_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = toposRoutePolicyRewardsJson)]
pub fn topos_route_policy_rewards_json(request_json: &str) -> Result<String, JsValue> {
    let request = rewards_request_from_json(request_json).map_err(js_error)?;
    let payload = topos_route_policy_rewards_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = toposRoutePolicyRewardsObject)]
pub fn topos_route_policy_rewards_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let request = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request = rewards_request_from_value(request).map_err(js_error)?;
    let payload = topos_route_policy_rewards_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = toposRoutePolicyResolveJson)]
pub fn topos_route_policy_resolve_json(request_json: &str) -> Result<String, JsValue> {
    let request = resolve_request_from_json(request_json).map_err(js_error)?;
    let payload = topos_route_policy_resolve_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = toposRoutePolicyResolveObject)]
pub fn topos_route_policy_resolve_object(request: &JsValue) -> Result<JsValue, JsValue> {
    let request = serde_wasm_bindgen::from_value::<Value>(request.clone()).map_err(js_error)?;
    let request = resolve_request_from_value(request).map_err(js_error)?;
    let payload = topos_route_policy_resolve_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn without_client(mut payload: Value) -> Value {
        payload
            .as_object_mut()
            .expect("Topos route-policy payload object")
            .remove("execution_client");
        payload
    }

    #[test]
    fn wasm_evaluation_matches_the_rust_contract() {
        let request: ToposRoutePolicyEvaluationRequest = serde_json::from_value(json!({
            "rows": [
                {
                    "label": "open",
                    "count": 2,
                    "trace_route_score": 0.4,
                    "response_text_quality_score": 0.7
                },
                {
                    "label": "guarded",
                    "count": 2,
                    "trace_route_score": 0.8,
                    "response_text_quality_score": 0.9
                }
            ]
        }))
        .expect("valid evaluation request");
        let expected = serde_json::to_value(
            evaluate_topos_route_policy(request.clone()).expect("valid Rust evaluation"),
        )
        .expect("serializable Rust evaluation");
        let actual = without_client(
            topos_route_policy_evaluate_value(request).expect("valid WASM evaluation"),
        );

        assert_eq!(actual, expected);
    }

    #[test]
    fn wasm_rewards_and_resolution_match_the_rust_contract() {
        let rewards_request: ToposRouteRewardsRequest = serde_json::from_value(json!({
            "profile": "grounded",
            "rows": [
                {
                    "label": "open",
                    "selection_scores": {"grounded": 0.2}
                },
                {
                    "label": "guarded",
                    "selection_scores": {"grounded": 0.8}
                }
            ]
        }))
        .expect("valid rewards request");
        let rust_rewards =
            build_topos_route_rewards(rewards_request.clone()).expect("valid Rust rewards");
        let wasm_rewards = without_client(
            topos_route_policy_rewards_value(rewards_request).expect("valid WASM rewards"),
        );
        assert_eq!(
            wasm_rewards,
            serde_json::to_value(&rust_rewards).expect("serializable Rust rewards")
        );

        let resolve_request = ToposRoutePolicyResolveRequest {
            rewards: rust_rewards.rewards,
            selected_label: Some("guarded".to_owned()),
            selected_index: 0,
        };
        let expected = serde_json::to_value(
            resolve_topos_route_policy(resolve_request.clone()).expect("valid Rust resolution"),
        )
        .expect("serializable Rust resolution");
        let actual = without_client(
            topos_route_policy_resolve_value(resolve_request).expect("valid WASM resolution"),
        );
        assert_eq!(actual, expected);
        assert_eq!(actual["resolution"], "label");
        assert_eq!(actual["selected_label"], "guarded");
    }

    #[test]
    fn wasm_ingress_fails_closed_on_contract_drift() {
        let unknown_field =
            evaluation_request_from_json(r#"{"rows":[],"selection_profile":"balanced"}"#)
                .expect_err("unknown request fields must fail closed");
        assert!(unknown_field.contains("unknown field"));

        let unknown_profile = rewards_request_from_json(r#"{"rows":[],"profile":"commander"}"#)
            .expect_err("unknown profiles must fail closed");
        assert!(unknown_profile.contains("unknown route-policy profile"));

        let normalized = rewards_request_from_json(r#"{"rows":[],"profile":" Grounded "}"#)
            .expect("profile normalization belongs to the Rust ingress");
        assert_eq!(normalized.profile.as_str(), "grounded");

        let oversized =
            evaluation_request_from_json(r#"{"rows":[{"label":"wide","count":4294967296}]}"#)
                .expect_err("wire integers must not depend on the compilation target");
        assert!(oversized.contains("u32"));

        let drifted = resolve_request_from_json(
            r#"{
                "rewards":[{"index":1,"label":"guarded","reward":0.8}],
                "selected_index":0
            }"#,
        )
        .expect("identity drift is a semantic validation error");
        let error = topos_route_policy_resolve_value(drifted)
            .expect_err("the shared Rust resolver must reject identity drift");
        assert!(matches!(
            error,
            ToposRoutePolicyError::RewardIndexMismatch {
                position: 0,
                actual: 1,
                expected: 0
            }
        ));
    }
}
