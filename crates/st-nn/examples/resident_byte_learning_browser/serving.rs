//! Thin standalone WASM client; admission and owner selection remain in st-nn.
use st_nn::resident::{
    ByteCorpusModelArtifact, ByteCorpusServingGate, ByteLmBatch, ResidentByteCandidate,
    ResidentByteServing,
};
use wasm_bindgen::prelude::*;

fn error(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

#[wasm_bindgen(js_name = ByteCorpusServingGate)]
pub struct WasmServingGate {
    inner: ByteCorpusServingGate,
}

#[wasm_bindgen(js_class = ByteCorpusServingGate)]
impl WasmServingGate {
    pub fn from_json(input: &str) -> Result<WasmServingGate, JsValue> {
        Ok(Self {
            inner: ByteCorpusServingGate::from_json(input.as_bytes()).map_err(error)?,
        })
    }

    pub fn to_json(&self) -> String {
        self.inner.to_json().to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn request_sha256(&self) -> String {
        self.inner.request_sha256().to_owned()
    }

    pub async fn bootstrap_wgpu(
        &self,
        artifact: &str,
        expected_id: &str,
    ) -> Result<WasmServing, JsValue> {
        let artifact =
            ByteCorpusModelArtifact::from_json(artifact.as_bytes(), expected_id).map_err(error)?;
        self.inner.validate_artifact(&artifact).map_err(error)?;
        let runtime =
            st_backend_wgpu::runtime::WgpuRuntime::request_headless("byte.serving.browser")
                .await
                .map_err(error)?;
        Ok(WasmServing {
            inner: ResidentByteServing::bootstrap(runtime, artifact, self.inner.clone())
                .await
                .map_err(error)?,
            shape: self.inner.batch_shape(),
        })
    }
}

#[wasm_bindgen(js_name = ResidentByteCandidate)]
pub struct WasmCandidate {
    inner: Option<ResidentByteCandidate>,
    accepted: bool,
    artifact_id: String,
    report: String,
}

#[wasm_bindgen(js_class = ResidentByteCandidate)]
impl WasmCandidate {
    #[wasm_bindgen(getter)]
    pub fn accepted(&self) -> bool {
        self.accepted
    }
    #[wasm_bindgen(getter)]
    pub fn consumed(&self) -> bool {
        self.inner.is_none()
    }
    #[wasm_bindgen(getter)]
    pub fn artifact_id(&self) -> String {
        self.artifact_id.clone()
    }
    pub fn to_json(&self) -> String {
        self.report.clone()
    }
}

#[wasm_bindgen(js_name = ResidentByteServing)]
pub struct WasmServing {
    inner: ResidentByteServing,
    shape: [usize; 2],
}

#[wasm_bindgen(js_class = ResidentByteServing)]
impl WasmServing {
    #[wasm_bindgen(getter)]
    pub fn artifact_id(&self) -> String {
        self.inner.artifact_id().to_owned()
    }
    /// Decimal string, so JS never rounds a generation counter.
    #[wasm_bindgen(getter)]
    pub fn generation(&self) -> String {
        self.inner.generation().to_string()
    }
    #[wasm_bindgen(getter)]
    pub fn gate_sha256(&self) -> String {
        self.inner.gate_sha256().to_owned()
    }
    #[wasm_bindgen(getter)]
    pub fn previous_artifact_id(&self) -> Option<String> {
        self.inner.previous_artifact_id().map(str::to_owned)
    }
    pub fn bootstrap_json(&self) -> Result<String, JsValue> {
        self.inner.bootstrap_json().map_err(error)
    }
    pub async fn prepare_candidate(
        &self,
        artifact: &str,
        expected_id: &str,
    ) -> Result<WasmCandidate, JsValue> {
        let artifact =
            ByteCorpusModelArtifact::from_json(artifact.as_bytes(), expected_id).map_err(error)?;
        let inner = self
            .inner
            .prepare_candidate(artifact)
            .await
            .map_err(error)?;
        Ok(WasmCandidate {
            accepted: inner.accepted(),
            artifact_id: inner.artifact_id().to_owned(),
            report: inner.to_json().map_err(error)?,
            inner: Some(inner),
        })
    }
    pub fn activate(&mut self, candidate: &mut WasmCandidate) -> Result<String, JsValue> {
        let candidate = candidate
            .inner
            .take()
            .ok_or_else(|| error("candidate permit already consumed"))?;
        Ok(self.inner.activate(candidate).map_err(error)?.to_string())
    }
    pub fn rollback(&mut self, expected_generation: &str) -> Result<String, JsValue> {
        let generation: u64 = expected_generation.parse().map_err(error)?;
        if generation.to_string() != expected_generation {
            return Err(error("generation must be a canonical decimal u64"));
        }
        Ok(self.inner.rollback(generation).map_err(error)?.to_string())
    }
    /// B rows of T+1 bytes, with the final shifted byte supplying a loss label.
    /// This maps a completed forward, not a streaming decoder or a KV cache.
    pub async fn forward(&mut self, windows: Vec<u8>) -> Result<String, JsValue> {
        use st_kernel_contracts::classification::{ClassReduction, CrossEntropySpec};
        let row = self.shape[1] + 1;
        if windows.len() != self.shape[0] * row {
            return Err(error("windows differ from serving shape"));
        }
        let host = ByteLmBatch::from_windows(&windows.chunks_exact(row).collect::<Vec<_>>())
            .map_err(error)?;
        let batch = self.inner.prepare_batch(&host).map_err(error)?;
        let tagged = self.inner.forward(&batch).map_err(error)?;
        let forward = tagged.forward();
        let loss = forward
            .next_byte_loss(CrossEntropySpec::new(ClassReduction::Mean, -100, 0.).map_err(error)?)
            .map_err(error)?;
        let values = self
            .inner
            .tensor_device()
            .snapshot_many(&[forward.prediction(), loss.value()])
            .map_err(error)?
            .read_async()
            .await
            .map_err(error)?;
        Ok(serde_json::json!({"artifact_id":tagged.artifact_id(),
            "generation":self.generation(), "revision":forward.parameter_revision(),
            "logits":values[0], "loss":values[1][0]})
        .to_string())
    }
}
