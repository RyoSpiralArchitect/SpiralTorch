use serde_json::Value;
use st_tensor::{ToposRuntimeProfile, ToposRuntimeProfileInput, ToposRuntimeRoute};

#[cfg(test)]
use st_tensor::{TOPOS_RUNTIME_ROUTE_CONTRACT_VERSION, TOPOS_RUNTIME_ROUTE_SEMANTIC_OWNER};

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
    let mut payload =
        serde_json::to_value(route.payload()).expect("Topos runtime route payload is serializable");
    let object = payload
        .as_object_mut()
        .expect("Topos runtime route payload is an object");
    object.insert(
        "execution_client".to_owned(),
        Value::String("wasm".to_owned()),
    );
    payload
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
