//! No client-side differentiation: expose Rust-owned forward tokens and VJPs.
use super::*;
#[cfg(feature = "webgpu")]
use crate::wgpu_tensor::{WasmWgpuTensor, WasmWgpuTensorDevice};
#[cfg(feature = "webgpu")]
use st_backend_wgpu::resident_training::graph::{
    GraphForward, GraphGradients, ResidentGraphAutograd,
};

#[cfg(feature = "webgpu")]
pub(super) fn compile(
    plan: &InferencePlan,
    tile: Option<Array>,
    kernel: Option<JsString>,
    accumulation: Option<JsString>,
) -> Result<Promise, JsValue> {
    let (tile, kernel, accumulation) = gpu_options(tile, kernel, accumulation)?;
    let plan = plan.clone();
    Ok(future_to_promise(async move {
        let runtime = crate::wgpu_resident::ensure_runtime().await?;
        let inner = plan
            .compile_graph_autograd_wgpu_with_options(runtime, tile, kernel, accumulation)
            .map_err(js_error)?;
        Ok(WasmResidentGraphAutograd { inner }.into())
    }))
}

#[wasm_bindgen(js_name = ResidentGraphAutograd)]
pub struct WasmResidentGraphAutograd {
    #[cfg(feature = "webgpu")]
    inner: ResidentGraphAutograd,
}
#[cfg(feature = "webgpu")]
#[wasm_bindgen(js_class = ResidentGraphAutograd)]
impl WasmResidentGraphAutograd {
    #[wasm_bindgen(getter, js_name = inputShape)]
    pub fn input_shape(&self) -> Vec<u32> {
        self.inner
            .input_layout()
            .shape()
            .iter()
            .map(|&n| n as u32)
            .collect()
    }
    #[wasm_bindgen(getter, js_name = outputShape)]
    pub fn output_shape(&self) -> Vec<u32> {
        self.inner
            .output_layout()
            .shape()
            .iter()
            .map(|&n| n as u32)
            .collect()
    }
    #[wasm_bindgen(getter, js_name = stageCount)]
    pub fn stage_count(&self) -> usize {
        self.inner.stage_count()
    }
    #[wasm_bindgen(getter, js_name = parameterCount)]
    pub fn parameter_count(&self) -> usize {
        self.inner.parameter_count()
    }
    #[wasm_bindgen(getter, js_name = inputGeneration)]
    pub fn input_generation(&self) -> u64 {
        self.inner.input_generation()
    }
    #[wasm_bindgen(getter, js_name = submittedForwards)]
    pub fn submitted_forwards(&self) -> u64 {
        self.inner.submitted_forwards()
    }
    #[wasm_bindgen(getter, js_name = submittedBackwards)]
    pub fn submitted_backwards(&self) -> u64 {
        self.inner.submitted_backwards()
    }
    #[wasm_bindgen(js_name = tensorDevice)]
    pub fn tensor_device(&self) -> WasmWgpuTensorDevice {
        WasmWgpuTensorDevice {
            inner: self.inner.tensor_device().clone(),
        }
    }
    #[wasm_bindgen(js_name = adapterInfo, unchecked_return_type = "{ name: string; backend: string; device_type: string }")]
    pub fn adapter_info(&self) -> Result<JsValue, JsValue> {
        let info = self.inner.adapter_info();
        crate::utils::json_to_js_value(&serde_json::json!({"name":info.name,"backend":format!("{:?}",info.backend),"device_type":format!("{:?}",info.device_type)}).to_string())
    }
    pub fn upload(
        &mut self,
        #[wasm_bindgen(unchecked_param_type = "Float32Array")] data: JsValue,
    ) -> Result<(), JsValue> {
        self.inner
            .upload(&crate::wgpu_tensor::values(data)?.to_vec())
            .map_err(js_error)
    }
    #[wasm_bindgen(js_name = setInputTensor)]
    pub fn set_input_tensor(&mut self, input: &WasmWgpuTensor) -> Result<(), JsValue> {
        self.inner.set_input_tensor(&input.inner).map_err(js_error)
    }
    pub fn forward(&mut self) -> Result<WasmGraphForward, JsValue> {
        Ok(WasmGraphForward {
            inner: self.inner.forward().map_err(js_error)?,
        })
    }
    pub fn backward(
        &mut self,
        forward: &WasmGraphForward,
        cotangent: &WasmWgpuTensor,
    ) -> Result<WasmGraphGradients, JsValue> {
        Ok(WasmGraphGradients {
            inner: self
                .inner
                .backward(&forward.inner, &cotangent.inner)
                .map_err(js_error)?,
        })
    }
}

#[wasm_bindgen(js_name = GraphForward)]
pub struct WasmGraphForward {
    #[cfg(feature = "webgpu")]
    inner: GraphForward,
}
#[cfg(feature = "webgpu")]
#[wasm_bindgen(js_class = GraphForward)]
impl WasmGraphForward {
    #[wasm_bindgen(getter, js_name = inputGeneration)]
    pub fn input_generation(&self) -> u64 {
        self.inner.input_generation()
    }
    #[wasm_bindgen(getter, js_name = submittedForward)]
    pub fn submitted_forward(&self) -> u64 {
        self.inner.submitted_forward()
    }
    #[wasm_bindgen(js_name = predictionTensor)]
    pub fn prediction_tensor(&self) -> WasmWgpuTensor {
        WasmWgpuTensor {
            inner: self.inner.prediction().clone(),
        }
    }
}

#[wasm_bindgen(js_name = GraphGradients)]
pub struct WasmGraphGradients {
    #[cfg(feature = "webgpu")]
    inner: GraphGradients,
}
#[cfg(feature = "webgpu")]
#[wasm_bindgen(js_class = GraphGradients)]
impl WasmGraphGradients {
    #[wasm_bindgen(getter, js_name = inputGeneration)]
    pub fn input_generation(&self) -> u64 {
        self.inner.input_generation()
    }
    #[wasm_bindgen(getter, js_name = submittedForward)]
    pub fn submitted_forward(&self) -> u64 {
        self.inner.submitted_forward()
    }
    #[wasm_bindgen(getter, js_name = submittedBackward)]
    pub fn submitted_backward(&self) -> u64 {
        self.inner.submitted_backward()
    }
    #[wasm_bindgen(getter, js_name = parameterCount)]
    pub fn parameter_count(&self) -> usize {
        self.inner.parameter_gradients().len()
    }
    #[wasm_bindgen(js_name = inputGradientTensor)]
    pub fn input_gradient_tensor(&self) -> WasmWgpuTensor {
        WasmWgpuTensor {
            inner: self.inner.input_gradient().clone(),
        }
    }
    #[wasm_bindgen(js_name = parameterGradientTensor)]
    pub fn parameter_gradient_tensor(
        &self,
        #[wasm_bindgen(unchecked_param_type = "number")] index: JsValue,
    ) -> Result<WasmWgpuTensor, JsValue> {
        let n = index
            .as_f64()
            .filter(|n| {
                n.is_finite()
                    && *n >= 0.
                    && n.fract() == 0.
                    && *n < self.inner.parameter_gradients().len() as f64
            })
            .ok_or_else(|| js_error("parameter gradient index out of range"))?;
        Ok(WasmWgpuTensor {
            inner: self.inner.parameter_gradients()[n as usize].clone(),
        })
    }
}
