//! Bounded host-memory access to the st-nn decoder. JS never proposes or selects weights.
use crate::geometry_artifact::GeometryAdapterArtifact;
use crate::utils::{bounded_json_string_from_js, js_error, js_u32};
use js_sys::{JsString, Uint32Array, Uint8Array};
use serde::Deserialize;
use st_nn::causal_lm::{
    CausalLmDeclaredSetStep, CausalLmLimits, DeclaredRivalRow, NativeCausalLm as Decoder,
    NativeCausalLmContext as CoreContext, MAX_DECLARED_RIVAL_IDS, MAX_LINE_MENU_CANDIDATES,
    MAX_LINE_MENU_TARGETS,
};
use st_nn::weights::{SafeTensorArchive, WeightLoadLimits};
use wasm_bindgen::prelude::*;

const FILE_BYTES: usize = 64 << 20;
const PARAMETERS: usize = 8 << 20;
const WINDOW_IDS: usize = 65_536;
const CACHE_VALUES: usize = 1 << 20;

#[wasm_bindgen(inline_js = r#"
const proto = Object.getPrototypeOf(Uint8Array.prototype);
const getter = (p, k) => Function.prototype.call.bind(Object.getOwnPropertyDescriptor(p, k).get);
const brand = getter(proto, Symbol.toStringTag), length = getter(proto, 'length');
const buffer = getter(proto, 'buffer'), ordinary = getter(ArrayBuffer.prototype, 'byteLength');
const Bytes = Uint8Array, Words = Uint32Array, copy = Function.prototype.call.bind(Bytes.prototype.set);
export function causalLmArchiveSnapshot(value, maximum) {
    if (brand(value) !== 'Uint8Array') throw new TypeError('weights must be a Uint8Array');
    ordinary(buffer(value)); // A concurrently writable SharedArrayBuffer is not a snapshot.
    const n = length(value);
    if (n < 8 || n > maximum) throw new RangeError('safetensors archive byte budget');
    const result = new Bytes(n);
    copy(result, value);
    return result;
}
export function causalLmIdCount(value, maximum) {
    if (brand(value) !== 'Uint32Array') throw new TypeError('token IDs must be a Uint32Array');
    ordinary(buffer(value));
    const n = length(value);
    if (n === 0 || n > maximum) throw new RangeError('invalid cached token count');
    return n;
}
export function causalLmIdSnapshot(value, admitted) {
    if (length(value) !== admitted) throw new RangeError('token count changed after admission');
    const result = new Words(admitted);
    copy(result, value);
    return result;
}
"#)]
extern "C" {
    #[wasm_bindgen(catch, js_name = causalLmArchiveSnapshot)]
    fn archive_snapshot(value: &JsValue, maximum: usize) -> Result<Uint8Array, JsValue>;
    #[wasm_bindgen(catch, js_name = causalLmIdCount)]
    fn id_count(value: &Uint32Array, maximum: usize) -> Result<usize, JsValue>;
    #[wasm_bindgen(catch, js_name = causalLmIdSnapshot)]
    fn id_snapshot(value: &Uint32Array, admitted: usize) -> Result<Uint32Array, JsValue>;
}

fn admitted_ids(ids: &Uint32Array, admitted: usize) -> Result<Vec<usize>, JsValue> {
    Ok(id_snapshot(ids, admitted)?
        .to_vec()
        .into_iter()
        .map(|id| id as usize)
        .collect())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BatchWire {
    shape: [usize; 2],
    ids: Vec<usize>,
    labels: Option<Vec<i64>>,
    #[serde(default)]
    contrasts: Vec<(usize, usize, Vec<usize>)>,
}

/// Unshifted [batch,time] IDs, optional labels (-100 ignores) and sorted [position,target,rivals].
#[wasm_bindgen]
pub struct CausalLmDeclaredBatch {
    shape: [usize; 2],
    ids: Vec<usize>,
    labels: Option<Vec<i64>>,
    rows: Vec<DeclaredRivalRow>,
}

#[wasm_bindgen]
impl CausalLmDeclaredBatch {
    #[wasm_bindgen(js_name = fromJson)]
    pub fn from_json(value: &JsString) -> Result<Self, JsValue> {
        let text = bounded_json_string_from_js(value, 1 << 20, "causal batch")?;
        let v: BatchWire = serde_json::from_str(&text).map_err(js_error)?;
        let [b, t] = v.shape;
        let count = b.checked_mul(t).filter(|&n| n <= WINDOW_IDS);
        if b == 0
            || t < 2
            || count != Some(v.ids.len())
            || v.labels.as_ref().is_some_and(|x| Some(x.len()) != count)
        {
            return Err(js_error("invalid causal batch shape or window length"));
        }
        if v.contrasts.len() > v.ids.len()
            || v.contrasts.iter().map(|r| r.2.len()).sum::<usize>() > MAX_DECLARED_RIVAL_IDS
        {
            return Err(js_error("causal batch exceeds declared rival budget"));
        }
        Ok(Self {
            shape: v.shape,
            ids: v.ids,
            labels: v.labels,
            rows: v
                .contrasts
                .into_iter()
                .map(|(position, target, rivals)| DeclaredRivalRow {
                    position,
                    target,
                    rivals,
                })
                .collect(),
        })
    }
}

#[wasm_bindgen]
pub struct NativeCausalLm {
    model: Decoder,
}

#[wasm_bindgen]
impl NativeCausalLm {
    /// One in-memory, bounded GPT-2-format safetensors archive. No network, GPU or model code.
    #[wasm_bindgen(js_name = fromGpt2)]
    pub fn from_gpt2(config: &JsString, weights: &JsValue) -> Result<Self, JsValue> {
        let config = bounded_json_string_from_js(config, 131_072, "GPT-2 config")?;
        let bytes = archive_snapshot(weights, FILE_BYTES)?.to_vec();
        let archive = SafeTensorArchive::from_owned_bytes(
            bytes,
            WeightLoadLimits {
                max_file_bytes: FILE_BYTES,
                max_header_bytes: 1 << 20,
                max_tensors: 4096,
                max_tensor_elements: PARAMETERS,
                ..Default::default()
            },
        )
        .map_err(js_error)?;
        let limits = CausalLmLimits {
            max_parameter_elements: PARAMETERS,
            max_activation_elements: 1 << 20,
            max_attention_pairs: 1 << 20,
            max_logit_elements: 1 << 20,
            max_training_checkpoint_elements: 1 << 20,
        };
        Decoder::from_gpt2_json(config.as_bytes(), &archive, limits)
            .map(|model| Self { model })
            .map_err(js_error)
    }

    #[wasm_bindgen(getter, js_name = baseModelIdentity)]
    pub fn base_model_identity(&self) -> String {
        self.model.base_model_identity().to_owned()
    }

    #[wasm_bindgen(js_name = withGeometry)]
    pub fn with_geometry(&self, artifact: &GeometryAdapterArtifact) -> Result<Self, JsValue> {
        self.model
            .with_geometry(artifact.inner.clone())
            .map(|model| Self { model })
            .map_err(js_error)
    }

    #[wasm_bindgen(js_name = geometryArtifact)]
    pub fn geometry_artifact(&self) -> Option<GeometryAdapterArtifact> {
        self.model
            .geometry_artifact()
            .cloned()
            .map(|inner| GeometryAdapterArtifact { inner })
    }

    /// Final-position vocabulary logits for one nonempty prompt, without appending a target.
    #[wasm_bindgen(js_name = nextLogits)]
    pub fn next_logits(&self, prompt: &JsString) -> Result<Vec<f32>, JsValue> {
        let prompt = bounded_json_string_from_js(prompt, 1 << 20, "causal prompt")?;
        let ids: Vec<usize> = serde_json::from_str(&prompt).map_err(js_error)?;
        if ids.len() > WINDOW_IDS {
            return Err(js_error("causal prompt exceeds token budget"));
        }
        self.model
            .next_token_logits(&ids)
            .map(|row| row.data().to_vec())
            .map_err(js_error)
    }

    /// Own the model snapshot and its native K/V state, including causal geometry history.
    pub fn prefill(
        &self,
        prompt: &Uint32Array,
        #[wasm_bindgen(unchecked_param_type = "number")] max_cache_values: &JsValue,
    ) -> Result<NativeCausalLmContext, JsValue> {
        let budget = js_u32(max_cache_values, "cache value limit")? as usize;
        if !(1..=CACHE_VALUES).contains(&budget) {
            return Err(js_error("invalid WASM cache value limit"));
        }
        let count = self
            .model
            .context_prefill_count(id_count(prompt, WINDOW_IDS)?, budget)
            .map_err(js_error)?;
        CoreContext::new(self.model.clone(), admitted_ids(prompt, count)?, budget)
            .map(|inner| NativeCausalLmContext { inner })
            .map_err(js_error)
    }

    #[wasm_bindgen(js_name = declaredSetNllJson)]
    pub fn declared_set_nll_json(&self, b: &CausalLmDeclaredBatch) -> Result<String, JsValue> {
        let scores = self
            .model
            .declared_set_nll(&b.ids, b.shape, b.labels.as_deref(), &b.rows)
            .map_err(js_error)?;
        serde_json::to_string(&scores).map_err(js_error)
    }

    #[wasm_bindgen(js_name = stepDeclaredSetGeometryAtDistance)]
    pub fn step_declared_set_geometry_at_distance(
        &self,
        b: &CausalLmDeclaredBatch,
        distance: &JsValue,
    ) -> Result<CausalLmDeclaredStep, JsValue> {
        let distance = distance
            .as_f64()
            .ok_or_else(|| js_error("distance must be a number"))?;
        let (model, report) = self
            .model
            .step_declared_set_geometry_at_distance(
                &b.ids,
                b.shape,
                b.labels.as_deref(),
                &b.rows,
                distance,
            )
            .map_err(js_error)?;
        Ok(CausalLmDeclaredStep { model, report })
    }
}

#[wasm_bindgen]
pub struct NativeCausalLmContext {
    inner: CoreContext,
}

#[wasm_bindgen]
impl NativeCausalLmContext {
    /// Score a sorted menu of packed content IDs and lengths, plus one separate stop per line.
    #[wasm_bindgen(js_name = scoreLinesJson)]
    pub fn score_lines_json(
        &self,
        content: &Uint32Array,
        lengths: &Uint32Array,
        #[wasm_bindgen(unchecked_param_type = "number")] stop_id: &JsValue,
    ) -> Result<String, JsValue> {
        let stop = js_u32(stop_id, "stop ID")? as usize;
        let lens = admitted_ids(lengths, id_count(lengths, MAX_LINE_MENU_CANDIDATES)?)?;
        self.inner
            .admit_line_lengths(&lens, stop)
            .map_err(js_error)?;
        let n = id_count(content, MAX_LINE_MENU_TARGETS)?;
        if n != lens.iter().sum::<usize>() {
            return Err(JsValue::from_str(
                "packed content does not match candidate lengths",
            ));
        }
        let flat = admitted_ids(content, n)?;
        let mut remaining = flat.as_slice();
        let lines: Vec<_> = lens
            .into_iter()
            .map(|len| {
                let (line, tail) = remaining.split_at(len);
                remaining = tail;
                line.to_vec()
            })
            .collect();
        let r = self.inner.score_lines(&lines, stop).map_err(js_error)?;
        serde_json::to_string(&r).map_err(js_error)
    }

    /// Fork the prefix. No JS KV, hidden re-prefill or intermediate vocabulary readouts.
    #[wasm_bindgen(js_name = withSuffix)]
    pub fn with_suffix(&self, suffix: &Uint32Array) -> Result<Self, JsValue> {
        let count = self
            .inner
            .suffix_input_count(id_count(suffix, WINDOW_IDS)?)
            .map_err(js_error)?;
        self.inner
            .with_suffix(&admitted_ids(suffix, count)?)
            .map(|inner| Self { inner })
            .map_err(js_error)
    }

    #[wasm_bindgen(js_name = nextLogits)]
    pub fn next_logits(&self) -> Vec<f32> {
        self.inner.logit_values().to_vec()
    }

    #[wasm_bindgen(getter, js_name = cacheValues)]
    pub fn cache_values(&self) -> usize {
        self.inner.cache_values()
    }

    #[wasm_bindgen(getter, js_name = tokenCount)]
    pub fn token_count(&self) -> usize {
        self.inner.token_ids().len()
    }

    #[wasm_bindgen(getter, js_name = geometryArtifactId)]
    pub fn geometry_artifact_id(&self) -> Option<String> {
        self.inner.geometry_artifact_id().map(str::to_owned)
    }
}

#[wasm_bindgen]
pub struct CausalLmDeclaredStep {
    model: Decoder,
    report: CausalLmDeclaredSetStep,
}

#[wasm_bindgen]
impl CausalLmDeclaredStep {
    #[wasm_bindgen(js_name = reportJson)]
    pub fn report_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.report).map_err(js_error)
    }

    /// Consumes the step, not the origin. On a finite rejection this returns the unchanged model.
    #[wasm_bindgen(js_name = intoModel)]
    pub fn into_model(self) -> NativeCausalLm {
        NativeCausalLm { model: self.model }
    }
}
