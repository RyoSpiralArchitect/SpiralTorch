//! Rust owns the dataset stream, update settlement and restart semantics.
use crate::utils::{js_error, js_u32};
use crate::vision_transforms::WasmVisionTransformPipeline;
use crate::wgpu_tensor::{values, WasmWgpuTensor, WasmWgpuTensorDevice};
use js_sys::{Array, Promise};
use st_tensor::Tensor;
use st_vision::resident_trainer::{
    ResidentVisionStepOutcome, ResidentVisionSubmission, ResidentVisionTrainer,
    ResidentVisionTrainerConfig, VisionTrainingCheckpoint, VisionTrainingCheckpointSnapshot,
};
use st_vision::{
    find_dataset_descriptor, DatasetSample, ImageTensor, TensorVisionDataset, VisionDataset,
};
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

/// In-memory classification samples. Creating a trainer freezes a copy; later
/// additions to this collection cannot alter an active owner's input stream.
#[wasm_bindgen(js_name = TensorVisionDataset)]
pub struct WasmTensorVisionDataset {
    inner: TensorVisionDataset,
}

#[wasm_bindgen(js_class = TensorVisionDataset)]
impl WasmTensorVisionDataset {
    #[wasm_bindgen(constructor)]
    pub fn new(descriptor: &str) -> Result<Self, JsValue> {
        let descriptor = find_dataset_descriptor(descriptor)
            .ok_or_else(|| js_error("unknown vision dataset"))?;
        Ok(Self {
            inner: TensorVisionDataset::new(descriptor.clone()),
        })
    }

    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.inner.len()
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "Binding exposes an explicit CHW sample and its label"
    )]
    pub fn push(
        &mut self,
        #[wasm_bindgen(unchecked_param_type = "number")] channels: JsValue,
        #[wasm_bindgen(unchecked_param_type = "number")] height: JsValue,
        #[wasm_bindgen(unchecked_param_type = "number")] width: JsValue,
        #[wasm_bindgen(unchecked_param_type = "Float32Array")] data: JsValue,
        #[wasm_bindgen(unchecked_param_type = "number")] class_id: JsValue,
        label: Option<String>,
    ) -> Result<(), JsValue> {
        let c = js_u32(&channels, "channels")? as usize;
        let h = js_u32(&height, "height")? as usize;
        let w = js_u32(&width, "width")? as usize;
        let class = js_u32(&class_id, "class_id")?;
        if f64::from(class as f32) != f64::from(class) {
            return Err(js_error("class_id is not exactly representable as f32"));
        }
        let image = ImageTensor::new(c, h, w, values(data)?.to_vec()).map_err(js_error)?;
        let mut sample = DatasetSample::new(image)
            .with_target(Tensor::from_vec(1, 1, vec![class as f32]).map_err(js_error)?);
        sample.label = label;
        self.inner.push_sample(sample).map_err(js_error)
    }
}

#[wasm_bindgen(js_name = ResidentVisionTrainer)]
pub struct WasmResidentVisionTrainer {
    inner: ResidentVisionTrainer<TensorVisionDataset>,
}

impl WasmResidentVisionTrainer {
    fn build(
        device: &WasmWgpuTensorDevice,
        dataset: &WasmTensorVisionDataset,
        dataset_sha256: &str,
        config_json: &str,
        pipeline: Option<&WasmVisionTransformPipeline>,
    ) -> Result<Self, JsValue> {
        let config = ResidentVisionTrainerConfig::from_json(config_json).map_err(js_error)?;
        Ok(Self {
            inner: ResidentVisionTrainer::from_dataset(
                &config,
                device.inner.clone(),
                Arc::new(dataset.inner.clone()),
                pipeline.map(|p| p.inner.clone()),
                dataset_sha256,
            )
            .map_err(js_error)?,
        })
    }

    fn restore(
        device: &WasmWgpuTensorDevice,
        dataset: &WasmTensorVisionDataset,
        dataset_sha256: &str,
        payload: &str,
        pipeline: Option<&WasmVisionTransformPipeline>,
    ) -> Result<Self, JsValue> {
        let checkpoint = VisionTrainingCheckpoint::from_json(payload).map_err(js_error)?;
        Ok(Self {
            inner: ResidentVisionTrainer::from_dataset_checkpoint(
                device.inner.clone(),
                Arc::new(dataset.inner.clone()),
                pipeline.map(|p| p.inner.clone()),
                dataset_sha256,
                &checkpoint,
            )
            .map_err(js_error)?,
        })
    }
}

#[wasm_bindgen(js_class = ResidentVisionTrainer)]
impl WasmResidentVisionTrainer {
    #[wasm_bindgen(js_name = defaultConfigJson)]
    pub fn default_config_json() -> Result<String, JsValue> {
        ResidentVisionTrainerConfig::default()
            .to_json()
            .map_err(js_error)
    }

    pub fn create(
        device: &WasmWgpuTensorDevice,
        dataset: &WasmTensorVisionDataset,
        dataset_sha256: &str,
        config_json: &str,
    ) -> Result<Self, JsValue> {
        Self::build(device, dataset, dataset_sha256, config_json, None)
    }

    #[wasm_bindgen(js_name = createWithPipeline)]
    pub fn create_with_pipeline(
        device: &WasmWgpuTensorDevice,
        dataset: &WasmTensorVisionDataset,
        dataset_sha256: &str,
        config_json: &str,
        pipeline: &WasmVisionTransformPipeline,
    ) -> Result<Self, JsValue> {
        Self::build(device, dataset, dataset_sha256, config_json, Some(pipeline))
    }

    #[wasm_bindgen(js_name = fromCheckpointJson)]
    pub fn from_checkpoint_json(
        device: &WasmWgpuTensorDevice,
        dataset: &WasmTensorVisionDataset,
        dataset_sha256: &str,
        payload: &str,
    ) -> Result<Self, JsValue> {
        Self::restore(device, dataset, dataset_sha256, payload, None)
    }

    #[wasm_bindgen(js_name = fromCheckpointJsonWithPipeline)]
    pub fn from_checkpoint_json_with_pipeline(
        device: &WasmWgpuTensorDevice,
        dataset: &WasmTensorVisionDataset,
        dataset_sha256: &str,
        payload: &str,
        pipeline: &WasmVisionTransformPipeline,
    ) -> Result<Self, JsValue> {
        Self::restore(device, dataset, dataset_sha256, payload, Some(pipeline))
    }

    #[wasm_bindgen(js_name = restoreCheckpointJson)]
    pub fn restore_checkpoint_json(&mut self, payload: &str) -> Result<(), JsValue> {
        let checkpoint = VisionTrainingCheckpoint::from_json(payload).map_err(js_error)?;
        self.inner.restore_checkpoint(&checkpoint).map_err(js_error)
    }

    #[wasm_bindgen(js_name = stateJson)]
    pub fn state_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(self.inner.state()).map_err(js_error)
    }

    #[wasm_bindgen(getter, js_name = hasPendingUpdate)]
    pub fn has_pending_update(&self) -> bool {
        self.inner.has_pending_update()
    }

    #[wasm_bindgen(js_name = submitNext)]
    pub fn submit_next(&mut self) -> Result<WasmResidentVisionSubmission, JsValue> {
        Ok(WasmResidentVisionSubmission {
            inner: self.inner.submit_next().map_err(js_error)?,
        })
    }

    /// The mutable WASM borrow spans this Promise. Concurrent reuse/free fails
    /// rather than detaching the update from its Rust owner.
    pub async fn settle(&mut self) -> Result<WasmResidentVisionStepOutcome, JsValue> {
        Ok(WasmResidentVisionStepOutcome {
            inner: self.inner.settle_async().await.map_err(js_error)?,
        })
    }

    #[wasm_bindgen(js_name = checkpointSnapshot)]
    pub fn checkpoint_snapshot(&self) -> Result<WasmVisionTrainingCheckpointSnapshot, JsValue> {
        Ok(WasmVisionTrainingCheckpointSnapshot {
            inner: Some(self.inner.checkpoint_snapshot().map_err(js_error)?),
        })
    }
}

#[wasm_bindgen(js_name = ResidentVisionSubmission)]
pub struct WasmResidentVisionSubmission {
    inner: ResidentVisionSubmission,
}

#[wasm_bindgen(js_class = ResidentVisionSubmission)]
impl WasmResidentVisionSubmission {
    #[wasm_bindgen(getter, js_name = attemptedRevision)]
    pub fn attempted_revision(&self) -> u64 {
        self.inner.attempted_revision
    }
    #[wasm_bindgen(getter)]
    pub fn epoch(&self) -> u64 {
        self.inner.epoch
    }
    #[wasm_bindgen(getter, js_name = learningRate)]
    pub fn learning_rate(&self) -> f32 {
        self.inner.learning_rate
    }
    #[wasm_bindgen(unchecked_return_type = "(string | null)[]")]
    pub fn labels(&self) -> Array {
        self.inner
            .labels
            .iter()
            .map(|v| v.as_deref().map_or(JsValue::NULL, JsValue::from_str))
            .collect()
    }
    pub fn images(&self) -> WasmWgpuTensor {
        WasmWgpuTensor {
            inner: self.inner.images.clone(),
        }
    }
    #[wasm_bindgen(js_name = lossTensor)]
    pub fn loss_tensor(&self) -> WasmWgpuTensor {
        WasmWgpuTensor {
            inner: self.inner.loss.clone(),
        }
    }
}

#[wasm_bindgen(js_name = ResidentVisionStepOutcome)]
pub struct WasmResidentVisionStepOutcome {
    inner: ResidentVisionStepOutcome,
}

#[wasm_bindgen(js_class = ResidentVisionStepOutcome)]
impl WasmResidentVisionStepOutcome {
    #[wasm_bindgen(getter, js_name = attemptedRevision)]
    pub fn attempted_revision(&self) -> u64 {
        self.inner.attempted_revision
    }
    #[wasm_bindgen(getter)]
    pub fn accepted(&self) -> bool {
        self.inner.accepted
    }
}

#[wasm_bindgen(js_name = VisionTrainingCheckpointSnapshot)]
pub struct WasmVisionTrainingCheckpointSnapshot {
    inner: Option<VisionTrainingCheckpointSnapshot>,
}

#[wasm_bindgen(js_class = VisionTrainingCheckpointSnapshot)]
impl WasmVisionTrainingCheckpointSnapshot {
    #[wasm_bindgen(js_name = readJson, unchecked_return_type = "Promise<string>")]
    pub fn read_json(&mut self) -> Result<Promise, JsValue> {
        let inner = self
            .inner
            .take()
            .ok_or_else(|| js_error("snapshot has already been consumed"))?;
        Ok(future_to_promise(async move {
            Ok(JsValue::from_str(
                &inner
                    .read_async()
                    .await
                    .map_err(js_error)?
                    .to_json()
                    .map_err(js_error)?,
            ))
        }))
    }
}
