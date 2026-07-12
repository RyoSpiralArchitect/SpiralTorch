use serde_json::{json, Value};
use st_tensor::{
    TensorError, ToposControlSignal, ToposControlSignalInput,
    TOPOS_CONTROL_SIGNAL_CONTRACT_VERSION, TOPOS_CONTROL_SIGNAL_SEMANTIC_OWNER,
};

use crate::topos_route::topos_runtime_route_from_route_value;

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

/// Build the shared Topos control-signal payload without introducing WASM semantics.
pub fn topos_control_signal_value(input: ToposControlSignalInput) -> Result<Value, TensorError> {
    let signal = ToposControlSignal::from_input(input)?;
    let training_hints = signal.training_hints();
    let training_plan = signal.training_plan(1.0);
    let inference_hints = signal.inference_hints();
    let inference_plan = signal.inference_plan(1.0, 1.0, 1.0, 0.0, 0.0);
    let runtime_profile = signal.runtime_profile(1.0, 1.0, 1.0, 1.0, 0.0, 0.0);
    let runtime_route = signal.runtime_route(1.0, 1.0, 1.0, 1.0, 0.0, 0.0);
    let training_hints = json!({
        "learning_rate_scale": training_hints.learning_rate_scale(),
        "regularization_scale": training_hints.regularization_scale(),
        "step_damping": training_hints.step_damping(),
        "gradient_bias_scale": training_hints.gradient_bias_scale(),
        "clip_scale": training_hints.clip_scale(),
        "momentum_damping": training_hints.momentum_damping(),
        "vector": training_hints.vector(),
    });
    let training_plan = json!({
        "gain": training_plan.gain(),
        "learning_rate_scale": training_plan.learning_rate_scale(),
        "regularization_scale": training_plan.regularization_scale(),
        "step_damping": training_plan.step_damping(),
        "gradient_bias_scale": training_plan.gradient_bias_scale(),
        "clip_scale": training_plan.clip_scale(),
        "momentum_damping": training_plan.momentum_damping(),
        "raw_rate_scale": training_plan.raw_rate_scale(),
        "rate_scale": training_plan.rate_scale(),
        "effective_gradient_bias_scale": training_plan.effective_gradient_bias_scale(),
        "effective_momentum_damping": training_plan.effective_momentum_damping(),
        "vector": training_plan.vector(),
    });
    let inference_hints = json!({
        "temperature_scale": inference_hints.temperature_scale(),
        "top_p_scale": inference_hints.top_p_scale(),
        "sampling_focus": inference_hints.sampling_focus(),
        "frequency_penalty_bias": inference_hints.frequency_penalty_bias(),
        "presence_penalty_bias": inference_hints.presence_penalty_bias(),
        "context_weight": inference_hints.context_weight(),
        "vector": inference_hints.vector(),
    });
    let inference_plan = json!({
        "gain": inference_plan.gain(),
        "temperature": inference_plan.temperature(),
        "top_p": inference_plan.top_p(),
        "frequency_penalty": inference_plan.frequency_penalty(),
        "presence_penalty": inference_plan.presence_penalty(),
        "context_weight": inference_plan.context_weight(),
        "temperature_scale": inference_plan.temperature_scale(),
        "top_p_scale": inference_plan.top_p_scale(),
        "sampling_focus": inference_plan.sampling_focus(),
        "vector": inference_plan.vector(),
    });
    let runtime_profile = json!({
        "training_gain": runtime_profile.training_gain(),
        "inference_gain": runtime_profile.inference_gain(),
        "closure_risk": runtime_profile.closure_risk(),
        "exploration_budget": runtime_profile.exploration_budget(),
        "control_energy": runtime_profile.control_energy(),
        "training_rate_scale": runtime_profile.training_rate_scale(),
        "training_gradient_bias_scale": runtime_profile.training_gradient_bias_scale(),
        "inference_temperature": runtime_profile.inference_temperature(),
        "inference_top_p": runtime_profile.inference_top_p(),
        "inference_context_weight": runtime_profile.inference_context_weight(),
        "learning_inference_balance": runtime_profile.learning_inference_balance(),
        "vector": runtime_profile.vector(),
    });
    let mut payload = json!({
        "kind": "spiraltorch.topos_control_signal",
        "contract_version": TOPOS_CONTROL_SIGNAL_CONTRACT_VERSION,
        "semantic_owner": TOPOS_CONTROL_SIGNAL_SEMANTIC_OWNER,
        "semantic_backend": "rust",
        "execution_client": "wasm",
        "curvature": signal.curvature(),
        "tolerance": signal.tolerance(),
        "saturation": signal.saturation(),
        "porosity": signal.porosity(),
        "max_depth": signal.max_depth(),
        "max_volume": signal.max_volume(),
        "observed_depth": signal.observed_depth(),
        "visited_volume": signal.visited_volume(),
        "remaining_volume": signal.remaining_volume(),
    });
    let derived = json!({
        "depth_pressure": signal.depth_pressure(),
        "volume_pressure": signal.volume_pressure(),
        "closure_pressure": signal.closure_pressure(),
        "openness": signal.openness(),
        "guard_strength": signal.guard_strength(),
        "stability_hint": signal.stability_hint(),
        "exploration_hint": signal.exploration_hint(),
        "learning_rate_scale": signal.learning_rate_scale(),
        "temperature_scale": signal.temperature_scale(),
        "regularization_scale": signal.regularization_scale(),
        "step_damping": signal.step_damping(),
        "sampling_focus": signal.sampling_focus(),
        "runtime_hints": signal.runtime_hints(),
        "gradient": signal.gradient(),
    });
    let clients = json!({
        "training_hints": training_hints,
        "training_plan": training_plan,
        "inference_hints": inference_hints,
        "inference_plan": inference_plan,
        "runtime_profile": runtime_profile,
        "runtime_route": topos_runtime_route_from_route_value(runtime_route),
    });
    let Value::Object(ref mut payload_map) = payload else {
        unreachable!("json object literal must serialize as an object")
    };
    for section in [derived, clients] {
        let Value::Object(section) = section else {
            unreachable!("json object literal must serialize as an object")
        };
        payload_map.extend(section);
    }
    Ok(payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = toposControlSignalJson)]
pub fn topos_control_signal_json(input_json: &str) -> Result<String, JsValue> {
    let input = serde_json::from_str::<ToposControlSignalInput>(input_json).map_err(js_error)?;
    let payload = topos_control_signal_value(input).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = toposControlSignalObject)]
pub fn topos_control_signal_object(input: &JsValue) -> Result<JsValue, JsValue> {
    let input = serde_wasm_bindgen::from_value::<ToposControlSignalInput>(input.clone())
        .map_err(js_error)?;
    let payload = topos_control_signal_value(input).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wasm_control_signal_uses_st_tensor_contract() {
        let payload = topos_control_signal_value(ToposControlSignalInput {
            max_depth: 10,
            max_volume: 100,
            observed_depth: 4,
            visited_volume: 25,
            ..ToposControlSignalInput::default()
        })
        .expect("valid control signal");

        assert_eq!(payload["kind"], "spiraltorch.topos_control_signal");
        assert_eq!(
            payload["contract_version"],
            TOPOS_CONTROL_SIGNAL_CONTRACT_VERSION
        );
        assert_eq!(payload["semantic_backend"], "rust");
        assert_eq!(payload["execution_client"], "wasm");
        let closure_pressure = payload["closure_pressure"]
            .as_f64()
            .expect("closure pressure is numeric");
        assert!((closure_pressure - 0.4).abs() < 1e-6);
        assert_eq!(payload["runtime_route"]["semantic_backend"], "rust");
    }

    #[test]
    fn wasm_control_signal_rejects_invalid_topology() {
        let result = topos_control_signal_value(ToposControlSignalInput {
            curvature: f32::NAN,
            ..ToposControlSignalInput::default()
        });
        assert!(matches!(result, Err(TensorError::NonFiniteValue { .. })));
    }
}
