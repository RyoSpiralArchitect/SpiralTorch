use crate::utils::{bounded_json_string_from_js, js_error};
use js_sys::JsString;
use serde::Deserialize;
use st_nn::language_supervision::CausalTargetAlignment;
use wasm_bindgen::prelude::*;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Encoding {
    source_length: u32,
    token_ids: Vec<u32>,
    token_offsets: Vec<[u32; 2]>,
    target_spans: Vec<[u32; 2]>,
}

/// Tokenization stays client-owned. Output contains ordinary numeric IDs,
/// unshifted labels and [token start, token end, source start, source end] runs.
#[wasm_bindgen(js_name = alignCausalTargetsJson)]
pub fn align_causal_targets_json(input: &JsString) -> Result<String, JsValue> {
    let raw = bounded_json_string_from_js(input, 8 * 1024 * 1024, "causal target alignment")?;
    let encoding: Encoding = serde_json::from_str(&raw).map_err(js_error)?;
    let plan = CausalTargetAlignment::new(
        encoding.source_length,
        encoding.token_ids,
        &encoding.token_offsets,
        &encoding.target_spans,
    )
    .map_err(js_error)?;
    serde_json::to_string(&plan).map_err(js_error)
}
