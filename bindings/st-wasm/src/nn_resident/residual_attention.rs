//! Browser handles over the same residual block and atomic update as native.
use super::*;
#[cfg(feature = "webgpu")]
use crate::wgpu_tensor::{WasmAttentionBiases, WasmWgpuTensor, WasmWgpuTensorDevice};
use st_nn::resident::ResidualAttentionPlan;
#[cfg(feature = "webgpu")]
use st_nn::resident::{
    ResidentResidualAttentionForward, ResidentResidualAttentionTraining,
    ResidentResidualAttentionVjp,
};

#[wasm_bindgen(js_name = ResidualAttentionPlan)]
pub struct WasmResidualAttentionPlan {
    inner: ResidualAttentionPlan,
}

#[wasm_bindgen(js_class = ResidualAttentionPlan)]
impl WasmResidualAttentionPlan {
    #[wasm_bindgen(js_name = fromPlans)]
    pub fn from_plans(
        pre: &WasmInferencePlan,
        attention: &WasmAttentionPlan,
        feed_forward: &WasmInferencePlan,
    ) -> Result<WasmResidualAttentionPlan, JsValue> {
        Ok(Self {
            inner: ResidualAttentionPlan::from_plans(
                &pre.inner,
                &attention.inner,
                &feed_forward.inner,
            )
            .map_err(js_error)?,
        })
    }
    #[wasm_bindgen(getter, js_name = inputShape)]
    pub fn input_shape(&self) -> Vec<u32> {
        self.inner
            .input_layout()
            .shape()
            .iter()
            .map(|&v| v as u32)
            .collect()
    }
    #[wasm_bindgen(getter, js_name = outputShape)]
    pub fn output_shape(&self) -> Vec<u32> {
        self.inner
            .output_layout()
            .shape()
            .iter()
            .map(|&v| v as u32)
            .collect()
    }
    #[wasm_bindgen(js_name = compileTrainingWebGpu, unchecked_return_type = "Promise<ResidentResidualAttentionTraining>")]
    pub fn compile_training_webgpu(
        &self,
        tile_mnk: Option<Array>,
        kernel: Option<JsString>,
        accumulation: Option<JsString>,
    ) -> Result<Promise, JsValue> {
        #[cfg(feature = "webgpu")]
        {
            let (tile, kernel, accumulation) = gpu_options(tile_mnk, kernel, accumulation)?;
            // The JS plan may be freed before asynchronous device acquisition finishes.
            let plan = self.inner.clone();
            Ok(future_to_promise(async move {
                let runtime = crate::wgpu_resident::ensure_runtime().await?;
                let inner = plan
                    .compile_training_wgpu_with_options(runtime, tile, kernel, accumulation)
                    .map_err(js_error)?;
                Ok(WasmResidualAttentionTraining { inner }.into())
            }))
        }
        #[cfg(not(feature = "webgpu"))]
        {
            let _ = (tile_mnk, kernel, accumulation);
            Err(js_error(
                "resident residual attention training requires the webgpu build feature",
            ))
        }
    }
}

#[wasm_bindgen(js_name = ResidentResidualAttentionTraining)]
pub struct WasmResidualAttentionTraining {
    #[cfg(feature = "webgpu")]
    inner: ResidentResidualAttentionTraining,
}

#[cfg(feature = "webgpu")]
#[wasm_bindgen(js_class = ResidentResidualAttentionTraining)]
impl WasmResidualAttentionTraining {
    #[wasm_bindgen(getter, js_name = inputShape)]
    pub fn input_shape(&self) -> Vec<u32> {
        self.inner
            .input_layout()
            .shape()
            .iter()
            .map(|&v| v as u32)
            .collect()
    }
    #[wasm_bindgen(getter, js_name = outputShape)]
    pub fn output_shape(&self) -> Vec<u32> {
        self.inner
            .output_layout()
            .shape()
            .iter()
            .map(|&v| v as u32)
            .collect()
    }
    #[wasm_bindgen(getter, js_name = attemptedUpdates)]
    pub fn attempted_updates(&self) -> u64 {
        self.inner.parameter_snapshot().revision()
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
    pub fn forward(
        &mut self,
        input: &WasmWgpuTensor,
        biases: &WasmAttentionBiases,
    ) -> Result<WasmResidualAttentionForward, JsValue> {
        Ok(WasmResidualAttentionForward {
            inner: self
                .inner
                .forward(
                    &input.inner,
                    biases.z_bias.as_ref(),
                    biases.pair_bias.as_ref(),
                )
                .map_err(js_error)?,
        })
    }
    pub fn backward(
        &mut self,
        forward: &WasmResidualAttentionForward,
        cotangent: &WasmWgpuTensor,
    ) -> Result<WasmResidualAttentionVjp, JsValue> {
        Ok(WasmResidualAttentionVjp {
            inner: self
                .inner
                .backward(&forward.inner, &cotangent.inner)
                .map_err(js_error)?,
        })
    }
    pub fn sgd(
        &mut self,
        gradients: &WasmResidualAttentionVjp,
        rate: f32,
    ) -> Result<WasmResidentParameterUpdate, JsValue> {
        Ok(WasmResidentParameterUpdate {
            inner: self.inner.sgd(&gradients.inner, rate).map_err(js_error)?,
        })
    }
}

#[wasm_bindgen(js_name = ResidualAttentionForward)]
pub struct WasmResidualAttentionForward {
    #[cfg(feature = "webgpu")]
    inner: ResidentResidualAttentionForward,
}
#[cfg(feature = "webgpu")]
#[wasm_bindgen(js_class = ResidualAttentionForward)]
impl WasmResidualAttentionForward {
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

#[wasm_bindgen(js_name = ResidualAttentionGradients)]
pub struct WasmResidualAttentionVjp {
    #[cfg(feature = "webgpu")]
    inner: ResidentResidualAttentionVjp,
}
#[cfg(feature = "webgpu")]
#[wasm_bindgen(js_class = ResidualAttentionGradients)]
impl WasmResidualAttentionVjp {
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
    #[wasm_bindgen(js_name = zBiasGradientTensor)]
    pub fn z_bias_gradient_tensor(&self) -> Option<WasmWgpuTensor> {
        self.inner
            .z_bias_gradient()
            .cloned()
            .map(|inner| WasmWgpuTensor { inner })
    }
    #[wasm_bindgen(js_name = pairBiasGradientTensor)]
    pub fn pair_bias_gradient_tensor(&self) -> Option<WasmWgpuTensor> {
        self.inner
            .pair_bias_gradient()
            .cloned()
            .map(|inner| WasmWgpuTensor { inner })
    }
}
