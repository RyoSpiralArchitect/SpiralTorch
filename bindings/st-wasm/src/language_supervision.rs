use crate::utils::{bounded_json_string_from_js, js_error};
use js_sys::JsString;
use serde::Deserialize;
use st_nn::language_supervision::{CausalCompletion, CausalTargetAlignment};
use wasm_bindgen::prelude::*;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Encoding {
    source_length: u32,
    token_ids: Vec<u32>,
    token_offsets: Vec<[u32; 2]>,
    target_spans: Vec<[u32; 2]>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CompletionSource {
    length: u32,
    offsets: Vec<[u32; 2]>,
    answer_span: [u32; 2],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CompletionInput {
    prompt_ids: Vec<u32>,
    prompt_answer_ids: Vec<u32>,
    training_ids: Vec<u32>,
    #[serde(default, deserialize_with = "present_source")]
    source: Option<CompletionSource>,
}

fn present_source<'de, D: serde::Deserializer<'de>>(
    de: D,
) -> Result<Option<CompletionSource>, D::Error> {
    CompletionSource::deserialize(de).map(Some)
}

/// Compact training IDs/labels and the two verified token boundaries.
#[wasm_bindgen(js_name = prepareCausalCompletionJson)]
pub fn prepare_causal_completion_json(input: &JsString) -> Result<String, JsValue> {
    let raw = bounded_json_string_from_js(input, 8 * 1024 * 1024, "causal completion")?;
    let v: CompletionInput = serde_json::from_str(&raw).map_err(js_error)?;
    CausalCompletion::admit_counts(
        v.prompt_ids.len(),
        v.prompt_answer_ids.len(),
        v.training_ids.len(),
    )
    .map_err(js_error)?;
    let prepared = match v.source {
        Some(s) => {
            let a =
                CausalTargetAlignment::new(s.length, v.training_ids, &s.offsets, &[s.answer_span])
                    .map_err(js_error)?;
            CausalCompletion::from_alignment(&a, &v.prompt_ids, &v.prompt_answer_ids)
        }
        None => CausalCompletion::from_tokenizations(
            &v.prompt_ids,
            &v.prompt_answer_ids,
            v.training_ids,
        ),
    }
    .map_err(js_error)?;
    serde_json::to_string(&prepared).map_err(js_error)
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
