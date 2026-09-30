//! Thin clients of the Rust classifier; readback and checkpoint mapping stay explicit.
use crate::utils::{js_error, js_u32};
use crate::wgpu_tensor::{WasmWgpuTensor, WasmWgpuTensorDevice};
use js_sys::{Array, BigInt, Promise};
use st_backend_wgpu::resident_training::parameters::ResidentParameterUpdate;
use st_vision::models::convnext::{
    ConvNeXtClassifier, ConvNeXtClassifierCheckpoint, ConvNeXtClassifierCheckpointSnapshot,
    ConvNeXtConfig, ResidentConvNeXtClassifier, ResidentConvNeXtForward, ResidentConvNeXtGradients,
};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

#[wasm_bindgen(js_name = ResidentConvNeXtClassifier)]
pub struct WasmResidentClassifier {
    inner: ResidentConvNeXtClassifier,
}

#[wasm_bindgen(js_class = ResidentConvNeXtClassifier)]
impl WasmResidentClassifier {
    #[wasm_bindgen(js_name = defaultConfigJson)]
    pub fn default_config_json() -> Result<String, JsValue> {
        serde_json::to_string(&ConvNeXtConfig::default()).map_err(js_error)
    }
    pub fn create(
        device: &WasmWgpuTensorDevice,
        config_json: &str,
        #[wasm_bindgen(unchecked_param_type = "number")] num_classes: JsValue,
        #[wasm_bindgen(unchecked_param_type = "number")] batch_size: JsValue,
        seed: BigInt,
    ) -> Result<Self, JsValue> {
        let classes = js_u32(&num_classes, "num_classes")? as usize;
        let batch = js_u32(&batch_size, "batch_size")? as usize;
        let seed = seed
            .to_string(10)
            .map_err(|_| js_error("invalid seed"))?
            .as_string()
            .ok_or_else(|| js_error("invalid seed"))?
            .parse::<u64>()
            .map_err(js_error)?;
        let config: ConvNeXtConfig = serde_json::from_str(config_json).map_err(js_error)?;
        Ok(Self {
            inner: ConvNeXtClassifier::new(config, classes, seed)
                .map_err(js_error)?
                .compile_resident_training(device.inner.clone(), batch)
                .map_err(js_error)?,
        })
    }
    #[wasm_bindgen(js_name = fromCheckpointJson)]
    pub fn from_checkpoint_json(
        device: &WasmWgpuTensorDevice,
        payload: &str,
    ) -> Result<Self, JsValue> {
        Ok(Self {
            inner: ConvNeXtClassifierCheckpoint::from_json(payload)
                .map_err(js_error)?
                .restore_resident(device.inner.clone())
                .map_err(js_error)?,
        })
    }
    #[wasm_bindgen(getter, js_name = inputShape)]
    pub fn input_shape(&self) -> Vec<u32> {
        self.inner.input_shape().iter().map(|&v| v as u32).collect()
    }
    #[wasm_bindgen(getter, js_name = outputShape)]
    pub fn output_shape(&self) -> Vec<u32> {
        self.inner
            .output_shape()
            .iter()
            .map(|&v| v as u32)
            .collect()
    }
    #[wasm_bindgen(getter, js_name = attemptedUpdates)]
    pub fn attempted_updates(&self) -> u64 {
        self.inner.parameter_snapshot().revision()
    }
    #[wasm_bindgen(js_name = parameterNames)]
    pub fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names().to_vec()
    }
    #[wasm_bindgen(js_name = parameterTensors, unchecked_return_type = "WgpuTensor[]")]
    pub fn parameter_tensors(&self) -> Array {
        self.inner
            .parameter_snapshot()
            .values()
            .iter()
            .cloned()
            .map(|inner| JsValue::from(WasmWgpuTensor { inner }))
            .collect()
    }
    #[wasm_bindgen(js_name = tensorDevice)]
    pub fn tensor_device(&self) -> WasmWgpuTensorDevice {
        WasmWgpuTensorDevice {
            inner: self.inner.tensor_device().clone(),
        }
    }
    pub fn forward(&mut self, input: &WasmWgpuTensor) -> Result<WasmConvNeXtForward, JsValue> {
        Ok(WasmConvNeXtForward {
            inner: self.inner.forward(&input.inner).map_err(js_error)?,
        })
    }
    pub fn backward(
        &mut self,
        forward: &WasmConvNeXtForward,
        cotangent: &WasmWgpuTensor,
    ) -> Result<WasmConvNeXtGradients, JsValue> {
        Ok(WasmConvNeXtGradients {
            inner: self
                .inner
                .backward(&forward.inner, &cotangent.inner)
                .map_err(js_error)?,
        })
    }
    pub fn sgd(
        &mut self,
        gradients: &WasmConvNeXtGradients,
        rate: f32,
    ) -> Result<WasmConvNeXtUpdate, JsValue> {
        Ok(WasmConvNeXtUpdate {
            inner: self.inner.sgd(&gradients.inner, rate).map_err(js_error)?,
        })
    }
    #[wasm_bindgen(js_name = checkpointSnapshot)]
    pub fn checkpoint_snapshot(&self) -> Result<WasmConvNeXtCheckpointSnapshot, JsValue> {
        Ok(WasmConvNeXtCheckpointSnapshot {
            inner: Some(self.inner.checkpoint_snapshot().map_err(js_error)?),
        })
    }
}

#[wasm_bindgen(js_name = ConvNeXtForward)]
pub struct WasmConvNeXtForward {
    inner: ResidentConvNeXtForward,
}
#[wasm_bindgen(js_class = ConvNeXtForward)]
impl WasmConvNeXtForward {
    #[wasm_bindgen(getter, js_name = parameterRevision)]
    pub fn parameter_revision(&self) -> u64 {
        self.inner.parameter_revision()
    }
    #[wasm_bindgen(js_name = predictionTensor)]
    pub fn prediction_tensor(&self) -> WasmWgpuTensor {
        WasmWgpuTensor {
            inner: self.inner.prediction().clone(),
        }
    }
}

#[wasm_bindgen(js_name = ConvNeXtGradients)]
pub struct WasmConvNeXtGradients {
    inner: ResidentConvNeXtGradients,
}
#[wasm_bindgen(js_class = ConvNeXtGradients)]
impl WasmConvNeXtGradients {
    #[wasm_bindgen(js_name = inputGradientTensor)]
    pub fn input_gradient_tensor(&self) -> WasmWgpuTensor {
        WasmWgpuTensor {
            inner: self.inner.input_gradient().clone(),
        }
    }
    #[wasm_bindgen(js_name = parameterGradientTensors, unchecked_return_type = "WgpuTensor[]")]
    pub fn parameter_gradient_tensors(&self) -> Array {
        self.inner
            .parameter_gradients()
            .iter()
            .cloned()
            .map(|inner| JsValue::from(WasmWgpuTensor { inner }))
            .collect()
    }
}

#[wasm_bindgen(js_name = ConvNeXtUpdate)]
pub struct WasmConvNeXtUpdate {
    inner: ResidentParameterUpdate,
}
#[wasm_bindgen(js_class = ConvNeXtUpdate)]
impl WasmConvNeXtUpdate {
    #[wasm_bindgen(getter, js_name = attemptedRevision)]
    pub fn attempted_revision(&self) -> u64 {
        self.inner.revision()
    }
    /// Explicit copy/map of the frozen flags; submission alone does not prove acceptance.
    #[wasm_bindgen(unchecked_return_type = "Promise<bigint>")]
    pub fn read(&self) -> Result<Promise, JsValue> {
        let snapshot = self.inner.snapshot().map_err(js_error)?;
        Ok(future_to_promise(async move {
            Ok(BigInt::from(snapshot.read_async().await.map_err(js_error)?).into())
        }))
    }
}

#[wasm_bindgen(js_name = ConvNeXtCheckpointSnapshot)]
pub struct WasmConvNeXtCheckpointSnapshot {
    inner: Option<ConvNeXtClassifierCheckpointSnapshot>,
}
#[wasm_bindgen(js_class = ConvNeXtCheckpointSnapshot)]
impl WasmConvNeXtCheckpointSnapshot {
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
