//! Packaged language-study clients. Model, resume and admission semantics live in st-nn.
use crate::utils::js_error;
use js_sys::{JsString, Uint8Array};
use st_nn::resident::{
    ByteCorpusInferencePolicy, ByteCorpusModelArtifact, ByteCorpusServingGate, ByteCorpusStudy,
    ByteCorpusStudyCheckpoint, BYTE_CORPUS_CHECKPOINT_SELECTION_MAX_CANDIDATES,
    BYTE_CORPUS_STUDY_MAX_BYTES,
};
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[cfg(feature = "webgpu")]
mod gpu;
#[cfg(feature = "webgpu")]
pub use gpu::{WasmByteCandidate, WasmByteInference, WasmByteServing};

#[wasm_bindgen(inline_js = r#"
const byteLmCharCodeAt = Function.prototype.call.bind(String.prototype.charCodeAt);
const byteLmEncoder = new TextEncoder();
const byteLmEncode = Function.prototype.call.bind(TextEncoder.prototype.encode);
const ByteLmTypeError = TypeError;
const ByteLmRangeError = RangeError;
export function byteLmStringSnapshot(value, maximum) {
    if (typeof value !== 'string' || value.length > maximum) {
        throw new ByteLmTypeError('expected a bounded primitive string');
    }
    // A lossy UTF-16 -> UTF-8 replacement would change a pinned identity.
    let bytes = 0;
    for (let i = 0; i < value.length; i++) {
        const c = byteLmCharCodeAt(value, i);
        if (c >= 0xd800 && c <= 0xdbff) {
            const next = byteLmCharCodeAt(value, ++i);
            if (!(next >= 0xdc00 && next <= 0xdfff)) {
                throw new ByteLmTypeError('unpaired UTF-16 surrogate');
            }
            bytes += 4;
        } else if (c >= 0xdc00 && c <= 0xdfff) {
            throw new ByteLmTypeError('unpaired UTF-16 surrogate');
        } else {
            bytes += c < 0x80 ? 1 : (c < 0x800 ? 2 : 3);
        }
        if (bytes > maximum) {
            throw new ByteLmRangeError('byte language string exceeds WASM ingress budget');
        }
    }
    // Avoid generated string-copy code invoking a mutable charCodeAt prototype
    // while a Rust owner is borrowed. The result is a fresh, private byte array.
    return byteLmEncode(byteLmEncoder, value);
}
"#)]
extern "C" {
    #[wasm_bindgen(catch, js_name = byteLmStringSnapshot)]
    fn string_snapshot(value: &JsString, maximum: f64) -> Result<Uint8Array, JsValue>;
}

fn text(value: &JsString, limit: usize, context: &str) -> Result<String, JsValue> {
    let bytes = string_snapshot(value, limit as f64)?.to_vec();
    String::from_utf8(bytes).map_err(|error| js_error(format!("{context}: {error}")))
}

fn json(value: &JsString) -> Result<String, JsValue> {
    text(value, BYTE_CORPUS_STUDY_MAX_BYTES, "byte language JSON")
}

fn policy(value: &JsString) -> Result<ByteCorpusInferencePolicy, JsValue> {
    serde_json::from_value(serde_json::Value::String(text(value, 128, "policy")?)).map_err(js_error)
}

#[wasm_bindgen(js_name = ByteCorpusStudy)]
pub struct WasmByteStudy {
    inner: Rc<ByteCorpusStudy>,
}

#[wasm_bindgen(js_class = ByteCorpusStudy)]
impl WasmByteStudy {
    #[wasm_bindgen(js_name = fromJson)]
    pub fn from_json(input: JsString) -> Result<WasmByteStudy, JsValue> {
        Ok(Self {
            inner: Rc::new(ByteCorpusStudy::from_json(json(&input)?.as_bytes()).map_err(js_error)?),
        })
    }

    #[wasm_bindgen(getter, js_name = requestSha256)]
    pub fn request_sha256(&self) -> String {
        self.inner.request_sha256().to_owned()
    }

    #[wasm_bindgen(getter, js_name = totalUpdates)]
    pub fn total_updates(&self) -> usize {
        self.inner.total_updates()
    }

    #[wasm_bindgen(getter, js_name = designSchema)]
    pub fn design_schema(&self) -> String {
        self.inner.design_schema().to_owned()
    }

    #[wasm_bindgen(getter, js_name = optimizerRecipeJson)]
    pub fn optimizer_recipe_json(&self) -> Result<String, JsValue> {
        self.inner.optimizer_recipe_json().map_err(js_error)
    }

    #[wasm_bindgen(js_name = checkpointSizeBound)]
    pub fn checkpoint_size_bound(&self) -> Result<usize, JsValue> {
        self.inner.checkpoint_size_bound().map_err(js_error)
    }

    #[wasm_bindgen(js_name = checkpointFromJson)]
    pub fn checkpoint_from_json(&self, input: JsString) -> Result<WasmByteCheckpoint, JsValue> {
        Ok(WasmByteCheckpoint {
            inner: Rc::new(
                self.inner
                    .checkpoint_from_json(json(&input)?.as_bytes())
                    .map_err(js_error)?,
            ),
        })
    }

    #[wasm_bindgen(js_name = caseArtifact)]
    pub fn case_artifact(
        &self,
        checkpoint: &WasmByteCheckpoint,
        case_name: JsString,
        inference_policy: JsString,
    ) -> Result<WasmByteArtifact, JsValue> {
        Ok(WasmByteArtifact {
            inner: Rc::new(
                self.inner
                    .case_artifact(
                        &checkpoint.inner,
                        &text(&case_name, BYTE_CORPUS_STUDY_MAX_BYTES, "case name")?,
                        policy(&inference_policy)?,
                    )
                    .map_err(js_error)?,
            ),
        })
    }

    /// Own the request and accept saved checkpoint JSON incrementally.
    #[wasm_bindgen(js_name = checkpointCandidates)]
    pub fn checkpoint_candidates(&self) -> WasmByteCheckpointCandidates {
        WasmByteCheckpointCandidates {
            study: self.inner.clone(),
            inner: Vec::new(),
            canonical_bytes: 0,
        }
    }

    #[wasm_bindgen(js_name = validateArtifact)]
    pub fn validate_artifact(
        &self,
        checkpoint: &WasmByteCheckpoint,
        artifact: &WasmByteArtifact,
    ) -> Result<(), JsValue> {
        self.inner
            .validate_artifact(&checkpoint.inner, &artifact.inner)
            .map_err(js_error)
    }
}

#[wasm_bindgen(js_name = ByteCorpusStudyCheckpoint)]
pub struct WasmByteCheckpoint {
    inner: Rc<ByteCorpusStudyCheckpoint>,
}

#[wasm_bindgen(js_class = ByteCorpusStudyCheckpoint)]
impl WasmByteCheckpoint {
    #[wasm_bindgen(getter, js_name = completedUpdates)]
    pub fn completed_updates(&self) -> usize {
        self.inner.completed_updates()
    }

    #[wasm_bindgen(getter, js_name = requestSha256)]
    pub fn request_sha256(&self) -> String {
        self.inner.request_sha256().to_owned()
    }

    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> Result<String, JsValue> {
        self.inner.to_json().map_err(js_error)
    }
}

/// Bounded imports, without multiple borrowed handles in one WASM ABI call.
/// Invalid/disposed objects cannot strand another owner's generated borrow.
#[wasm_bindgen(js_name = ByteCorpusCheckpointCandidates)]
pub struct WasmByteCheckpointCandidates {
    study: Rc<ByteCorpusStudy>,
    inner: Vec<Rc<ByteCorpusStudyCheckpoint>>,
    canonical_bytes: usize,
}

#[wasm_bindgen(js_class = ByteCorpusCheckpointCandidates)]
impl WasmByteCheckpointCandidates {
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.inner.len()
    }

    #[wasm_bindgen(getter)]
    pub fn capacity(&self) -> usize {
        BYTE_CORPUS_CHECKPOINT_SELECTION_MAX_CANDIDATES
    }

    /// Import opaque saved bytes without a JS parse/serialize roundtrip.
    #[wasm_bindgen(js_name = appendJson)]
    pub fn append_json(&mut self, input: JsString) -> Result<(), JsValue> {
        if self.inner.len() >= BYTE_CORPUS_CHECKPOINT_SELECTION_MAX_CANDIDATES {
            return Err(js_error("checkpoint candidates are full (maximum 32)"));
        }
        let checkpoint = self
            .study
            .checkpoint_from_json(json(&input)?.as_bytes())
            .map_err(js_error)?;
        let canonical_bytes = self
            .canonical_bytes
            .checked_add(checkpoint.to_json().map_err(js_error)?.len())
            .filter(|&n| n <= BYTE_CORPUS_STUDY_MAX_BYTES)
            .ok_or_else(|| js_error("selection checkpoints exceed 64 MiB combined"))?;
        self.inner.push(Rc::new(checkpoint));
        self.canonical_bytes = canonical_bytes;
        Ok(())
    }

    /// Rust owns chronological/history validation and development-loss selection.
    /// No adapter acquisition, training or serving promotion occurs.
    #[wasm_bindgen(js_name = selectCheckpoint)]
    pub fn select_checkpoint(
        &self,
        case_name: JsString,
        inference_policy: JsString,
    ) -> Result<WasmByteCheckpointSelection, JsValue> {
        let name = text(&case_name, BYTE_CORPUS_STUDY_MAX_BYTES, "case name")?;
        let policy = policy(&inference_policy)?;
        let refs: Vec<_> = self.inner.iter().map(Rc::as_ref).collect();
        let selected = self
            .study
            .select_checkpoint(&refs, &name, policy)
            .map_err(js_error)?;
        Ok(WasmByteCheckpointSelection {
            artifact: Rc::new(selected.artifact),
            report: serde_json::to_string(&selected.report).map_err(js_error)?,
        })
    }
}

/// An owned result. Retrieved artifacts remain valid after this handle is freed.
#[wasm_bindgen(js_name = ByteCorpusCheckpointSelection)]
pub struct WasmByteCheckpointSelection {
    artifact: Rc<ByteCorpusModelArtifact>,
    report: String,
}

#[wasm_bindgen(js_class = ByteCorpusCheckpointSelection)]
impl WasmByteCheckpointSelection {
    #[wasm_bindgen(getter)]
    pub fn artifact(&self) -> WasmByteArtifact {
        WasmByteArtifact {
            inner: self.artifact.clone(),
        }
    }

    #[wasm_bindgen(getter, js_name = reportJson)]
    pub fn report_json(&self) -> String {
        self.report.clone()
    }
}

#[wasm_bindgen(js_name = ByteCorpusStudySegment)]
pub struct WasmByteSegment {
    checkpoint: Rc<ByteCorpusStudyCheckpoint>,
    report: String,
}

#[wasm_bindgen(js_class = ByteCorpusStudySegment)]
impl WasmByteSegment {
    #[wasm_bindgen(getter)]
    pub fn checkpoint(&self) -> WasmByteCheckpoint {
        WasmByteCheckpoint {
            inner: self.checkpoint.clone(),
        }
    }

    #[wasm_bindgen(getter, js_name = reportJson)]
    pub fn report_json(&self) -> String {
        self.report.clone()
    }
}

#[wasm_bindgen(js_name = ByteCorpusModelArtifact)]
pub struct WasmByteArtifact {
    inner: Rc<ByteCorpusModelArtifact>,
}

#[wasm_bindgen(js_class = ByteCorpusModelArtifact)]
impl WasmByteArtifact {
    /// Obtain the expected identity independently of the untrusted payload.
    #[wasm_bindgen(js_name = fromJson)]
    pub fn from_json(input: JsString, expected_id: JsString) -> Result<WasmByteArtifact, JsValue> {
        let expected = text(&expected_id, 71, "artifact identity")?;
        Ok(Self {
            inner: Rc::new(
                ByteCorpusModelArtifact::from_json(json(&input)?.as_bytes(), &expected)
                    .map_err(js_error)?,
            ),
        })
    }

    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> String {
        self.inner.to_json().to_owned()
    }

    #[wasm_bindgen(getter, js_name = artifactId)]
    pub fn artifact_id(&self) -> String {
        self.inner.artifact_id().to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn policy(&self) -> String {
        self.inner.policy().as_str().to_owned()
    }

    #[wasm_bindgen(getter, js_name = caseName)]
    pub fn case_name(&self) -> String {
        self.inner.case_name().to_owned()
    }

    #[wasm_bindgen(getter, js_name = requestSha256)]
    pub fn request_sha256(&self) -> String {
        self.inner.request_sha256().to_owned()
    }

    #[wasm_bindgen(getter, js_name = studyCheckpointSha256)]
    pub fn study_checkpoint_sha256(&self) -> String {
        self.inner.study_checkpoint_sha256().to_owned()
    }

    #[wasm_bindgen(getter, js_name = attemptedRevision)]
    pub fn attempted_revision(&self) -> String {
        self.inner.checkpoint().attempted_revision().to_string()
    }
}

#[wasm_bindgen(js_name = ByteCorpusServingGate)]
pub struct WasmByteServingGate {
    inner: Rc<ByteCorpusServingGate>,
}

#[wasm_bindgen(js_class = ByteCorpusServingGate)]
impl WasmByteServingGate {
    #[wasm_bindgen(js_name = fromJson)]
    pub fn from_json(input: JsString) -> Result<WasmByteServingGate, JsValue> {
        Ok(Self {
            inner: Rc::new(
                ByteCorpusServingGate::from_json(json(&input)?.as_bytes()).map_err(js_error)?,
            ),
        })
    }

    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> String {
        self.inner.to_json().to_owned()
    }

    #[wasm_bindgen(getter, js_name = requestSha256)]
    pub fn request_sha256(&self) -> String {
        self.inner.request_sha256().to_owned()
    }

    #[wasm_bindgen(getter, js_name = schemaVersion)]
    pub fn schema_version(&self) -> u8 {
        self.inner.schema_version()
    }

    #[wasm_bindgen(getter, js_name = cohortNames)]
    pub fn cohort_names(&self) -> Vec<String> {
        self.inner
            .cohort_names()
            .into_iter()
            .map(str::to_owned)
            .collect()
    }

    #[wasm_bindgen(getter, js_name = batchCount)]
    pub fn batch_count(&self) -> usize {
        self.inner.batch_count()
    }

    #[wasm_bindgen(getter, js_name = batchShape)]
    pub fn batch_shape(&self) -> Vec<usize> {
        self.inner.batch_shape().to_vec()
    }

    #[wasm_bindgen(js_name = validateArtifact)]
    pub fn validate_artifact(&self, artifact: &WasmByteArtifact) -> Result<(), JsValue> {
        self.inner
            .validate_artifact(&artifact.inner)
            .map_err(js_error)
    }
}
