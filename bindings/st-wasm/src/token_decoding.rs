//! Bounded typed-array transport; ranking, probability and repression belong to st-core.
use crate::utils::{js_error, js_is_typed_array, js_u32};
use js_sys::{Float32Array, Uint32Array};
use st_core::inference::token_decoding::{
    greedy_token_id_f32, prepare_token_candidates_f32, TokenDecodingConfig, TokenSelection,
};
use wasm_bindgen::prelude::*;

/// Score raw logits, without decoding, sampling, or loading a language model.
#[wasm_bindgen(js_name = scoreTokenLogitsJsonF32)]
pub fn score_token_logits_json(
    #[wasm_bindgen(unchecked_param_type = "Float32Array")] logits: &JsValue,
    #[wasm_bindgen(unchecked_param_type = "number")] target_id: &JsValue,
) -> Result<String, JsValue> {
    use st_tensor::classification::TargetClassScore;
    let target = js_u32(target_id, "target_id")? as usize;
    if !js_is_typed_array(logits, "Float32Array")? {
        return Err(js_error("logits must be a Float32Array"));
    }
    // Read internal slots, not shadowable properties. Fix the view length before copying,
    // including for subarrays, other realms and growable backing buffers.
    let float_proto = js_sys::Object::get_prototype_of(&Float32Array::new_with_length(0));
    let typed_proto = js_sys::Object::get_prototype_of(float_proto.as_ref());
    let slot = |key: &str| -> Result<JsValue, JsValue> {
        let desc = js_sys::Object::get_own_property_descriptor(&typed_proto, &key.into());
        js_sys::Reflect::get(&desc, &"get".into())?
            .dyn_into::<js_sys::Function>()?
            .call0(logits)
    };
    let count = js_u32(&slot("length")?, "vocabulary")?;
    if count as usize > st_core::inference::token_decoding::MAX_VOCABULARY {
        return Err(js_error("logits exceed vocabulary budget"));
    }
    TargetClassScore::validate_domain(count as usize, target).map_err(js_error)?;
    let offset = js_u32(&slot("byteOffset")?, "byteOffset")?;
    let view = Float32Array::new_with_byte_offset_and_length(&slot("buffer")?, offset, count);
    let score = TargetClassScore::from_logits(&view.to_vec(), target).map_err(js_error)?;
    serde_json::to_string(&score).map_err(js_error)
}

fn inputs(
    logits: &JsValue,
    history: &JsValue,
    json: &str,
) -> Result<(Vec<f32>, Vec<usize>, TokenDecodingConfig), JsValue> {
    let logits = logits
        .dyn_ref::<Float32Array>()
        .ok_or_else(|| js_error("logits must be a Float32Array"))?;
    let history = history
        .dyn_ref::<Uint32Array>()
        .ok_or_else(|| js_error("history must be a Uint32Array"))?;
    if json.len() > 65_536 {
        return Err(js_error("token decoding config exceeds byte budget"));
    }
    let config: TokenDecodingConfig = serde_json::from_str(json).map_err(js_error)?;
    if config.selection != TokenSelection::Greedy {
        return Err(js_error("expected greedy selection"));
    }
    config
        .validate(logits.length() as usize, history.length() as usize)
        .map_err(js_error)?;
    Ok((
        logits.to_vec(),
        history.to_vec().into_iter().map(|v| v as usize).collect(),
        config,
    ))
}

#[wasm_bindgen(js_name = greedyTokenIdF32)]
pub fn greedy_token_id(
    #[wasm_bindgen(unchecked_param_type = "Float32Array")] logits: &JsValue,
    #[wasm_bindgen(unchecked_param_type = "Uint32Array")] history: &JsValue,
    config_json: &str,
) -> Result<u32, JsValue> {
    let (logits, history, cfg) = inputs(logits, history, config_json)?;
    greedy_token_id_f32(&logits, &history, &cfg)
        .map(|id| id as u32)
        .map_err(js_error)
}

#[wasm_bindgen(js_name = greedyTokenDecisionJsonF32)]
pub fn greedy_token_decision_json(
    #[wasm_bindgen(unchecked_param_type = "Float32Array")] logits: &JsValue,
    #[wasm_bindgen(unchecked_param_type = "Uint32Array")] history: &JsValue,
    config_json: &str,
) -> Result<String, JsValue> {
    let (logits, history, cfg) = inputs(logits, history, config_json)?;
    let decision = prepare_token_candidates_f32(&logits, &history, &cfg)
        .map_err(js_error)?
        .greedy();
    serde_json::to_string(&decision).map_err(js_error)
}
