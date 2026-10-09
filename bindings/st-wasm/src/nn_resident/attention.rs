//! Browser handles over Rust-owned projection composition, VJP and shared SGD.
use super::*;
#[cfg(feature = "webgpu")]
use crate::wgpu_tensor::{WasmAttentionBiases, WasmWgpuTensor, WasmWgpuTensorDevice};
use st_nn::resident::{AttentionInferencePlan, AttentionMask};
#[cfg(feature = "webgpu")]
use st_nn::resident::{ResidentAttentionForward, ResidentAttentionTraining, ResidentAttentionVjp};

#[wasm_bindgen(js_name = AttentionInferencePlan)]
pub struct WasmAttentionPlan {
    inner: AttentionInferencePlan,
}

#[wasm_bindgen(js_class = AttentionInferencePlan)]
impl WasmAttentionPlan {
    #[wasm_bindgen(js_name = fromProjectionPlans)]
    pub fn from_projection_plans(
        query: &WasmInferencePlan,
        key: &WasmInferencePlan,
        value: &WasmInferencePlan,
        output: &WasmInferencePlan,
        heads: Number,
        causal_offset: Option<Number>,
    ) -> Result<WasmAttentionPlan, JsValue> {
        let heads = js_u32(heads.as_ref(), "heads")? as usize;
        let mask = causal_offset
            .map(|v| js_u32(v.as_ref(), "causal_offset").map(|n| n as usize))
            .transpose()?
            .map_or(AttentionMask::None, |query_offset| AttentionMask::Causal {
                query_offset,
            });
        Ok(Self {
            inner: AttentionInferencePlan::from_projection_plans(
                heads,
                mask,
                [&query.inner, &key.inner, &value.inner, &output.inner],
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
    #[wasm_bindgen(js_name = compileTrainingWebGpu, unchecked_return_type = "Promise<ResidentAttentionTraining>")]
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
                Ok(WasmAttentionTraining { inner }.into())
            }))
        }
        #[cfg(not(feature = "webgpu"))]
        {
            let _ = (tile_mnk, kernel, accumulation);
            Err(js_error(
                "resident attention training requires the webgpu build feature",
            ))
        }
    }
}

#[wasm_bindgen(js_name = ResidentAttentionTraining)]
pub struct WasmAttentionTraining {
    #[cfg(feature = "webgpu")]
    inner: ResidentAttentionTraining,
}

#[cfg(feature = "webgpu")]
#[wasm_bindgen(js_class = ResidentAttentionTraining)]
impl WasmAttentionTraining {
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
    ) -> Result<WasmAttentionForward, JsValue> {
        Ok(WasmAttentionForward {
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
        forward: &WasmAttentionForward,
        cotangent: &WasmWgpuTensor,
    ) -> Result<WasmAttentionVjp, JsValue> {
        Ok(WasmAttentionVjp {
            inner: self
                .inner
                .backward(&forward.inner, &cotangent.inner)
                .map_err(js_error)?,
        })
    }
    pub fn sgd(
        &mut self,
        gradients: &WasmAttentionVjp,
        rate: f32,
    ) -> Result<WasmResidentParameterUpdate, JsValue> {
        Ok(WasmResidentParameterUpdate {
            inner: self.inner.sgd(&gradients.inner, rate).map_err(js_error)?,
        })
    }
}

#[wasm_bindgen(js_name = AttentionForward)]
pub struct WasmAttentionForward {
    #[cfg(feature = "webgpu")]
    inner: ResidentAttentionForward,
}
#[cfg(feature = "webgpu")]
#[wasm_bindgen(js_class = AttentionForward)]
impl WasmAttentionForward {
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

#[wasm_bindgen(js_name = AttentionGradients)]
pub struct WasmAttentionVjp {
    #[cfg(feature = "webgpu")]
    inner: ResidentAttentionVjp,
}
#[cfg(feature = "webgpu")]
#[wasm_bindgen(js_class = AttentionGradients)]
impl WasmAttentionVjp {
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
