//! Bounded typed-array transport; ranking, probability and repression belong to st-core.
use crate::utils::js_error;
use js_sys::{Float32Array, Uint32Array};
use st_core::inference::token_decoding::{
    greedy_token_id_f32, prepare_token_candidates_f32, TokenDecodingConfig, TokenSelection,
};
use wasm_bindgen::prelude::*;

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
    logits: &JsValue,
    history: &JsValue,
    config_json: &str,
) -> Result<u32, JsValue> {
    let (logits, history, cfg) = inputs(logits, history, config_json)?;
    greedy_token_id_f32(&logits, &history, &cfg)
        .map(|id| id as u32)
        .map_err(js_error)
}

#[wasm_bindgen(js_name = greedyTokenDecisionJsonF32)]
pub fn greedy_token_decision_json(
    logits: &JsValue,
    history: &JsValue,
    config_json: &str,
) -> Result<String, JsValue> {
    let (logits, history, cfg) = inputs(logits, history, config_json)?;
    let decision = prepare_token_candidates_f32(&logits, &history, &cfg)
        .map_err(js_error)?
        .greedy();
    serde_json::to_string(&decision).map_err(js_error)
}
