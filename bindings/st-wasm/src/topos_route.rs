use serde_json::{json, Value};
use st_tensor::{
    ToposRuntimeProfile, ToposRuntimeProfileInput, ToposRuntimeRoute,
    TOPOS_RUNTIME_ROUTE_CONTRACT_VERSION, TOPOS_RUNTIME_ROUTE_SEMANTIC_OWNER,
};

#[cfg(target_arch = "wasm32")]
use serde::Serialize;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::js_error;

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

/// Build the shared Topos runtime-route payload without introducing WASM-specific semantics.
pub fn topos_runtime_route_value(input: ToposRuntimeProfileInput) -> Value {
    topos_runtime_route_from_route_value(ToposRuntimeProfile::from_input(input).route())
}

pub(crate) fn topos_runtime_route_from_route_value(route: ToposRuntimeRoute) -> Value {
    let profile = route.profile();
    let scores = route.scores();
    json!({
        "kind": "spiraltorch.topos_runtime_route",
        "contract_version": TOPOS_RUNTIME_ROUTE_CONTRACT_VERSION,
        "semantic_owner": TOPOS_RUNTIME_ROUTE_SEMANTIC_OWNER,
        "semantic_backend": "rust",
        "execution_client": "wasm",
        "mode": route.mode_label(),
        "mode_id": route.mode_id(),
        "score": route.score(),
        "score_key": route.score_key(),
        "learning_action": route.learning_action(),
        "inference_action": route.inference_action(),
        "scores": {
            "training": scores.training_score(),
            "inference": scores.inference_score(),
            "guard": scores.guard_score(),
            "exploration": scores.exploration_score(),
            "context": scores.context_score(),
            "vector": scores.vector(),
        },
        "runtime_profile": {
            "training_gain": profile.training_gain(),
            "inference_gain": profile.inference_gain(),
            "closure_risk": profile.closure_risk(),
            "exploration_budget": profile.exploration_budget(),
            "control_energy": profile.control_energy(),
            "training_rate_scale": profile.training_rate_scale(),
            "training_gradient_bias_scale": profile.training_gradient_bias_scale(),
            "inference_temperature": profile.inference_temperature(),
            "inference_top_p": profile.inference_top_p(),
            "inference_context_weight": profile.inference_context_weight(),
            "learning_inference_balance": profile.learning_inference_balance(),
            "vector": profile.vector(),
        },
    })
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = toposRuntimeRouteJson)]
pub fn topos_runtime_route_json(profile_json: &str) -> Result<String, JsValue> {
    let input = serde_json::from_str::<ToposRuntimeProfileInput>(profile_json).map_err(js_error)?;
    serde_json::to_string(&topos_runtime_route_value(input)).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = toposRuntimeRouteObject)]
pub fn topos_runtime_route_object(profile: &JsValue) -> Result<JsValue, JsValue> {
    let input = serde_wasm_bindgen::from_value::<ToposRuntimeProfileInput>(profile.clone())
        .map_err(js_error)?;
    to_json_compatible_js(&topos_runtime_route_value(input))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wasm_route_payload_uses_st_tensor_contract() {
        let payload = topos_runtime_route_value(ToposRuntimeProfileInput::default());

        assert_eq!(payload["kind"], "spiraltorch.topos_runtime_route");
        assert_eq!(
            payload["contract_version"],
            TOPOS_RUNTIME_ROUTE_CONTRACT_VERSION
        );
        assert_eq!(
            payload["semantic_owner"],
            TOPOS_RUNTIME_ROUTE_SEMANTIC_OWNER
        );
        assert_eq!(payload["semantic_backend"], "rust");
        assert_eq!(payload["execution_client"], "wasm");
        assert_eq!(payload["mode"], "contextual");
        assert_eq!(payload["score_key"], "context");
        assert_eq!(payload["scores"]["context"], payload["score"]);
    }

    #[test]
    fn wasm_route_payload_reports_normalized_profile() {
        let payload = topos_runtime_route_value(ToposRuntimeProfileInput {
            closure_risk: -2.0,
            exploration_budget: 4.0,
            inference_top_p: 0.0,
            inference_context_weight: 8.0,
            ..ToposRuntimeProfileInput::default()
        });

        assert_eq!(payload["runtime_profile"]["closure_risk"], 0.0);
        assert_eq!(payload["runtime_profile"]["exploration_budget"], 1.0);
        let top_p = payload["runtime_profile"]["inference_top_p"]
            .as_f64()
            .expect("top-p is numeric");
        let context_weight = payload["runtime_profile"]["inference_context_weight"]
            .as_f64()
            .expect("context weight is numeric");
        assert!((top_p - 0.05).abs() < 1e-6);
        assert!((context_weight - 1.25).abs() < 1e-6);
    }

    #[test]
    fn wasm_profile_json_accepts_partial_and_roundtrip_payloads() {
        let input = serde_json::from_str::<ToposRuntimeProfileInput>(
            r#"{"closure_risk":0.7,"vector":[0.0,0.7,0.0,1.0,1.0,1.0]}"#,
        )
        .expect("partial runtime profile");
        let payload = topos_runtime_route_value(input);

        let closure_risk = payload["runtime_profile"]["closure_risk"]
            .as_f64()
            .expect("closure risk is numeric");
        assert!((closure_risk - 0.7).abs() < 1e-6);
        assert_eq!(payload["runtime_profile"]["training_gain"], 1.0);
        assert_eq!(payload["runtime_profile"]["inference_top_p"], 1.0);
    }
}
