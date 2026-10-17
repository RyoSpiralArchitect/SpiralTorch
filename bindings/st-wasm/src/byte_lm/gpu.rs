//! Device work uses the package runtime and owned futures, never an async JS borrow.
use super::*;
use crate::utils::{js_is_typed_array, js_u32};
use crate::wgpu_resident::ensure_runtime;
use js_sys::{Number, Promise, Uint8Array};
use st_nn::resident::{
    ByteLmBatch, ResidentByteCandidate, ResidentByteInference, ResidentByteServing,
};
use std::cell::{Ref, RefCell};
use std::ops::{Deref, DerefMut};
use wasm_bindgen_futures::future_to_promise;

#[wasm_bindgen(inline_js = r#"
const typed = Object.getPrototypeOf(Uint8Array.prototype);
const lengthOf = Object.getOwnPropertyDescriptor(typed, 'length').get;
const bufferOf = Object.getOwnPropertyDescriptor(typed, 'buffer').get;
const ordinaryBufferLength = Object.getOwnPropertyDescriptor(ArrayBuffer.prototype, 'byteLength').get;
const copyBytes = Uint8Array.prototype.set;
export function byteLmSnapshotBytes(value, expected) {
    if (lengthOf.call(value) !== expected) {
        throw new RangeError('bytes differ from the exact model window shape');
    }
    // Reject shared storage: a concurrent writer would defeat a stable ingress snapshot.
    ordinaryBufferLength.call(bufferOf.call(value));
    const result = new Uint8Array(expected);
    copyBytes.call(result, value);
    return result;
}
"#)]
extern "C" {
    #[wasm_bindgen(catch, js_name = byteLmSnapshotBytes)]
    fn snapshot_bytes(value: &JsValue, expected: usize) -> Result<Uint8Array, JsValue>;
}

fn bytes(value: &JsValue, expected: usize) -> Result<Vec<u8>, JsValue> {
    if !js_is_typed_array(value, "Uint8Array")? {
        return Err(js_error(
            "bytes must be a Uint8Array, not an array or a coercible object",
        ));
    }
    Ok(snapshot_bytes(value, expected)?.to_vec())
}

fn generation(value: &JsString) -> Result<u64, JsValue> {
    let value = text(value, 20, "generation")?;
    let parsed: u64 = value.parse().map_err(js_error)?;
    if parsed.to_string() != value {
        return Err(js_error(
            "generation must be a canonical decimal u64 string",
        ));
    }
    Ok(parsed)
}

fn host_batch(value: &JsValue, shape: [usize; 2]) -> Result<ByteLmBatch, JsValue> {
    let row = shape[1] + 1;
    let data = bytes(value, shape[0] * row)?;
    ByteLmBatch::from_windows(&data.chunks_exact(row).collect::<Vec<_>>()).map_err(js_error)
}

fn prompts(value: &JsValue, shape: [usize; 2]) -> Result<Vec<Vec<u8>>, JsValue> {
    Ok(bytes(value, shape[0] * shape[1])?
        .chunks_exact(shape[1])
        .map(<[u8]>::to_vec)
        .collect())
}

/// A lease is acquired synchronously, so overlapping operations fail before
/// dispatch. Drop restores the owner after success or rejection. Freeing the
/// exported handle does not cancel a pending operation or destroy its owner.
struct Owner<T>(Rc<RefCell<Option<T>>>);
struct Lease<T> {
    slot: Rc<RefCell<Option<T>>>,
    value: Option<T>,
}

impl<T> Owner<T> {
    fn new(value: T) -> Self {
        Self(Rc::new(RefCell::new(Some(value))))
    }
    fn read(&self) -> Result<Ref<'_, T>, JsValue> {
        Ref::filter_map(self.0.borrow(), Option::as_ref)
            .map_err(|_| js_error("byte owner operation in progress"))
    }
    fn lease(&self) -> Result<Lease<T>, JsValue> {
        let value = self
            .0
            .borrow_mut()
            .take()
            .ok_or_else(|| js_error("byte owner operation in progress"))?;
        Ok(Lease {
            slot: self.0.clone(),
            value: Some(value),
        })
    }
}
impl<T> Deref for Lease<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.value.as_ref().expect("live lease")
    }
}
impl<T> DerefMut for Lease<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.value.as_mut().expect("live lease")
    }
}
impl<T> Drop for Lease<T> {
    fn drop(&mut self) {
        *self.slot.borrow_mut() = self.value.take();
    }
}

fn advance(
    study: Rc<ByteCorpusStudy>,
    checkpoint: Option<Rc<ByteCorpusStudyCheckpoint>>,
    stop: Number,
) -> Result<Promise, JsValue> {
    let stop = js_u32(stop.as_ref(), "stop_after")? as usize;
    study
        .validate_segment(checkpoint.as_deref(), stop)
        .map_err(js_error)?;
    Ok(future_to_promise(async move {
        let segment = study
            .advance(ensure_runtime().await?, checkpoint.as_deref(), stop)
            .await
            .map_err(js_error)?;
        Ok(WasmByteSegment {
            checkpoint: Rc::new(segment.checkpoint),
            report: segment.report.to_string(),
        }
        .into())
    }))
}

#[wasm_bindgen(js_class = ByteCorpusStudy)]
impl WasmByteStudy {
    /// Start a fresh study, stopping at an absolute accepted-update boundary.
    #[wasm_bindgen(js_name = advanceWebGpu, unchecked_return_type = "Promise<ByteCorpusStudySegment>")]
    pub fn advance_webgpu(&self, stop_after: Number) -> Result<Promise, JsValue> {
        advance(self.inner.clone(), None, stop_after)
    }

    #[wasm_bindgen(js_name = resumeWebGpu, unchecked_return_type = "Promise<ByteCorpusStudySegment>")]
    pub fn resume_webgpu(
        &self,
        checkpoint: &WasmByteCheckpoint,
        stop_after: Number,
    ) -> Result<Promise, JsValue> {
        advance(
            self.inner.clone(),
            Some(checkpoint.inner.clone()),
            stop_after,
        )
    }
}

#[wasm_bindgen(js_class = ByteCorpusModelArtifact)]
impl WasmByteArtifact {
    #[wasm_bindgen(js_name = restoreWebGpu, unchecked_return_type = "Promise<ResidentByteInference>")]
    pub fn restore_webgpu(&self) -> Promise {
        let artifact = self.inner.clone();
        future_to_promise(async move {
            let inner = artifact
                .restore_wgpu(ensure_runtime().await?)
                .map_err(js_error)?;
            let shape = inner.input_layout().shape();
            Ok(WasmByteInference {
                shape: [shape[0], shape[1]],
                inner: Owner::new(inner),
            }
            .into())
        })
    }
}

#[wasm_bindgen(js_name = ResidentByteInference)]
pub struct WasmByteInference {
    inner: Owner<ResidentByteInference>,
    shape: [usize; 2],
}

#[wasm_bindgen(js_class = ResidentByteInference)]
impl WasmByteInference {
    #[wasm_bindgen(getter, js_name = artifactId)]
    pub fn artifact_id(&self) -> Result<String, JsValue> {
        Ok(self.inner.read()?.artifact_id().to_owned())
    }

    #[wasm_bindgen(getter, js_name = windowShape)]
    pub fn window_shape(&self) -> Vec<usize> {
        self.shape.to_vec()
    }

    /// Fixed B*(T+1) windows; logits and mean next-byte CE are explicit readbacks.
    #[wasm_bindgen(unchecked_return_type = "Promise<string>")]
    pub fn forward(&self, windows: JsValue) -> Result<Promise, JsValue> {
        let mut owner = self.inner.lease()?;
        let host = host_batch(&windows, self.shape)?;
        let batch = owner.prepare_batch(&host).map_err(js_error)?;
        let tagged = owner.forward(&batch).map_err(js_error)?;
        Ok(future_to_promise(async move {
            read_forward(owner.tensor_device(), tagged, None)
                .await
                .map(JsValue::from)
        }))
    }

    #[wasm_bindgen(js_name = generateGreedy, unchecked_return_type = "Promise<string>")]
    pub fn generate_greedy(
        &self,
        input: JsValue,
        max_new_bytes: Number,
    ) -> Result<Promise, JsValue> {
        let mut owner = self.inner.lease()?;
        let count = js_u32(max_new_bytes.as_ref(), "max_new_bytes")? as usize;
        owner
            .validate_generation(std::iter::repeat_n(self.shape[1], self.shape[0]), count)
            .map_err(js_error)?;
        let rows = prompts(&input, self.shape)?;
        Ok(future_to_promise(async move {
            owner
                .generate_greedy(&rows, count)
                .await
                .and_then(|v| v.to_json())
                .map(JsValue::from)
                .map_err(js_error)
        }))
    }
}

async fn read_forward(
    device: &st_backend_wgpu::resident_tensor::TensorDevice,
    tagged: st_nn::resident::ResidentByteInferenceForward,
    serving_generation: Option<String>,
) -> Result<String, JsValue> {
    let forward = tagged.forward();
    let loss = forward
        .next_byte_loss(
            st_tensor::CrossEntropyConfig::default()
                .kernel_spec()
                .map_err(js_error)?,
        )
        .map_err(js_error)?;
    let values = device
        .snapshot_many(&[forward.prediction(), loss.value()])
        .map_err(js_error)?
        .read_async()
        .await
        .map_err(js_error)?;
    Ok(serde_json::json!({
        "artifact_id": tagged.artifact_id(), "revision": forward.parameter_revision().to_string(),
        "generation": serving_generation, "logits": values[0], "loss": values[1][0],
    })
    .to_string())
}

#[wasm_bindgen(js_class = ByteCorpusServingGate)]
impl WasmByteServingGate {
    #[wasm_bindgen(js_name = bootstrapWebGpu, unchecked_return_type = "Promise<ResidentByteServing>")]
    pub fn bootstrap_webgpu(&self, artifact: &WasmByteArtifact) -> Result<Promise, JsValue> {
        self.inner
            .validate_artifact(&artifact.inner)
            .map_err(js_error)?;
        let gate = self.inner.clone();
        let artifact = artifact.inner.clone();
        Ok(future_to_promise(async move {
            let inner = ResidentByteServing::bootstrap(
                ensure_runtime().await?,
                (*artifact).clone(),
                (*gate).clone(),
            )
            .await
            .map_err(js_error)?;
            Ok(WasmByteServing {
                shape: gate.batch_shape(),
                inner: Owner::new(inner),
            }
            .into())
        }))
    }
}

#[wasm_bindgen(js_name = ResidentByteCandidate)]
pub struct WasmByteCandidate {
    inner: Option<ResidentByteCandidate>,
    accepted: bool,
    artifact_id: String,
    report: String,
}

#[wasm_bindgen(js_class = ResidentByteCandidate)]
impl WasmByteCandidate {
    #[wasm_bindgen(getter)]
    pub fn accepted(&self) -> bool {
        self.accepted
    }
    #[wasm_bindgen(getter)]
    pub fn consumed(&self) -> bool {
        self.inner.is_none()
    }
    #[wasm_bindgen(getter, js_name = artifactId)]
    pub fn artifact_id(&self) -> String {
        self.artifact_id.clone()
    }
    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> String {
        self.report.clone()
    }
}

#[wasm_bindgen(js_name = ResidentByteServing)]
pub struct WasmByteServing {
    inner: Owner<ResidentByteServing>,
    shape: [usize; 2],
}

#[wasm_bindgen(js_class = ResidentByteServing)]
impl WasmByteServing {
    #[wasm_bindgen(getter, js_name = artifactId)]
    pub fn artifact_id(&self) -> Result<String, JsValue> {
        Ok(self.inner.read()?.artifact_id().to_owned())
    }
    #[wasm_bindgen(getter)]
    pub fn generation(&self) -> Result<String, JsValue> {
        Ok(self.inner.read()?.generation().to_string())
    }
    #[wasm_bindgen(getter, js_name = gateSha256)]
    pub fn gate_sha256(&self) -> Result<String, JsValue> {
        Ok(self.inner.read()?.gate_sha256().to_owned())
    }
    #[wasm_bindgen(getter, js_name = previousArtifactId)]
    pub fn previous_artifact_id(&self) -> Result<Option<String>, JsValue> {
        Ok(self.inner.read()?.previous_artifact_id().map(str::to_owned))
    }
    #[wasm_bindgen(getter, js_name = windowShape)]
    pub fn window_shape(&self) -> Vec<usize> {
        self.shape.to_vec()
    }
    #[wasm_bindgen(js_name = bootstrapJson)]
    pub fn bootstrap_json(&self) -> Result<String, JsValue> {
        self.inner.read()?.bootstrap_json().map_err(js_error)
    }

    #[wasm_bindgen(js_name = prepareCandidate, unchecked_return_type = "Promise<ResidentByteCandidate>")]
    pub fn prepare_candidate(&self, artifact: &WasmByteArtifact) -> Result<Promise, JsValue> {
        let owner = self.inner.lease()?;
        let artifact = artifact.inner.clone();
        Ok(future_to_promise(async move {
            let inner = owner
                .prepare_candidate((*artifact).clone())
                .await
                .map_err(js_error)?;
            Ok(WasmByteCandidate {
                accepted: inner.accepted(),
                artifact_id: inner.artifact_id().to_owned(),
                report: inner.to_json().map_err(js_error)?,
                inner: Some(inner),
            }
            .into())
        }))
    }

    pub fn activate(&self, candidate: &mut WasmByteCandidate) -> Result<String, JsValue> {
        let mut owner = self.inner.lease()?;
        let candidate = candidate
            .inner
            .take()
            .ok_or_else(|| js_error("candidate permit already consumed"))?;
        Ok(owner.activate(candidate).map_err(js_error)?.to_string())
    }

    pub fn rollback(&self, expected_generation: JsString) -> Result<String, JsValue> {
        let expected = generation(&expected_generation)?;
        Ok(self
            .inner
            .lease()?
            .rollback(expected)
            .map_err(js_error)?
            .to_string())
    }

    #[wasm_bindgen(unchecked_return_type = "Promise<string>")]
    pub fn forward(&self, windows: JsValue) -> Result<Promise, JsValue> {
        let mut owner = self.inner.lease()?;
        let host = host_batch(&windows, self.shape)?;
        let batch = owner.prepare_batch(&host).map_err(js_error)?;
        let tagged = owner.forward(&batch).map_err(js_error)?;
        let generation = owner.generation().to_string();
        Ok(future_to_promise(async move {
            read_forward(owner.tensor_device(), tagged, Some(generation))
                .await
                .map(JsValue::from)
        }))
    }

    #[wasm_bindgen(js_name = generateGreedy, unchecked_return_type = "Promise<string>")]
    pub fn generate_greedy(
        &self,
        input: JsValue,
        max_new_bytes: Number,
        expected_generation: JsString,
    ) -> Result<Promise, JsValue> {
        let mut owner = self.inner.lease()?;
        let count = js_u32(max_new_bytes.as_ref(), "max_new_bytes")? as usize;
        let expected = generation(&expected_generation)?;
        owner
            .validate_generation(
                std::iter::repeat_n(self.shape[1], self.shape[0]),
                count,
                expected,
            )
            .map_err(js_error)?;
        let rows = prompts(&input, self.shape)?;
        Ok(future_to_promise(async move {
            owner
                .generate_greedy(&rows, count, expected)
                .await
                .and_then(|v| v.to_json())
                .map(JsValue::from)
                .map_err(js_error)
        }))
    }
}
