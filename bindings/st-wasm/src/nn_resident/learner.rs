#[cfg(feature = "webgpu")]
use super::autograd::{WasmGraphForward, WasmGraphGradients};
#[cfg(feature = "webgpu")]
use super::graph::WasmGraphTrainingParametersSnapshot;
use super::*;
#[cfg(feature = "webgpu")]
use crate::wgpu_tensor::{WasmWgpuTensor, WasmWgpuTensorDevice};
#[cfg(feature = "webgpu")]
use st_backend_wgpu::resident_training::graph as backend;

#[cfg(feature = "webgpu")]
pub(super) fn compile(
    plan: &InferencePlan,
    policy: st_nn::resident::GraphGradientPolicy,
    tile: Option<Array>,
    kernel: Option<JsString>,
    accumulation: Option<JsString>,
) -> Result<Promise, JsValue> {
    let (tile, kernel, accumulation) = gpu_options(tile, kernel, accumulation)?;
    let plan = plan.clone();
    Ok(future_to_promise(async move {
        let runtime = crate::wgpu_resident::ensure_runtime().await?;
        let inner = plan
            .compile_graph_learner_wgpu_with_options(runtime, policy, tile, kernel, accumulation)
            .map_err(js_error)?;
        Ok(WasmResidentGraphLearner { inner }.into())
    }))
}
#[wasm_bindgen(js_name=GraphGradientBatch)]
#[derive(Default)]
pub struct WasmGraphGradientBatch {
    #[cfg(feature = "webgpu")]
    inner: backend::GraphGradientBatch,
}
#[cfg(feature = "webgpu")]
#[wasm_bindgen(js_class=GraphGradientBatch)]
impl WasmGraphGradientBatch {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: backend::GraphGradientBatch::new(),
        }
    }
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.inner.len()
    }
    pub fn add(&mut self, gradients: &WasmGraphGradients, weight: f32) -> Result<(), JsValue> {
        self.inner.add(&gradients.inner, weight).map_err(js_error)
    }
}

#[wasm_bindgen(js_name=ResidentGraphLearner)]
pub struct WasmResidentGraphLearner {
    #[cfg(feature = "webgpu")]
    inner: backend::ResidentGraphLearner,
}
#[cfg(feature = "webgpu")]
#[wasm_bindgen(js_class=ResidentGraphLearner)]
impl WasmResidentGraphLearner {
    #[wasm_bindgen(getter,js_name=inputShape)]
    pub fn input_shape(&self) -> Vec<u32> {
        self.inner
            .input_layout()
            .shape()
            .iter()
            .map(|&v| v as u32)
            .collect()
    }
    #[wasm_bindgen(getter,js_name=outputShape)]
    pub fn output_shape(&self) -> Vec<u32> {
        self.inner
            .output_layout()
            .shape()
            .iter()
            .map(|&v| v as u32)
            .collect()
    }
    #[wasm_bindgen(getter,js_name=stageCount)]
    pub fn stage_count(&self) -> usize {
        self.inner.stage_count()
    }
    #[wasm_bindgen(getter,js_name=parameterCount)]
    pub fn parameter_count(&self) -> usize {
        self.inner.parameter_count()
    }
    #[wasm_bindgen(getter,js_name=gradientPolicy)]
    pub fn gradient_policy(&self) -> String {
        self.inner.gradient_policy().as_str().to_owned()
    }
    #[wasm_bindgen(getter,js_name=inputGeneration)]
    pub fn input_generation(&self) -> u64 {
        self.inner.input_generation()
    }
    #[wasm_bindgen(getter,js_name=submittedForwards)]
    pub fn submitted_forwards(&self) -> u64 {
        self.inner.submitted_forwards()
    }
    #[wasm_bindgen(getter,js_name=submittedBackwards)]
    pub fn submitted_backwards(&self) -> u64 {
        self.inner.submitted_backwards()
    }
    #[wasm_bindgen(getter,js_name=submittedUpdates)]
    pub fn submitted_updates(&self) -> u64 {
        self.inner.submitted_updates()
    }
    #[wasm_bindgen(js_name=tensorDevice)]
    pub fn tensor_device(&self) -> WasmWgpuTensorDevice {
        WasmWgpuTensorDevice {
            inner: self.inner.tensor_device().clone(),
        }
    }
    #[wasm_bindgen(js_name=adapterInfo,unchecked_return_type="{ name: string; backend: string; device_type: string }")]
    pub fn adapter_info(&self) -> Result<JsValue, JsValue> {
        let info = self.inner.adapter_info();
        crate::utils::json_to_js_value(&serde_json::json!({"name":info.name,"backend":format!("{:?}",info.backend),"device_type":format!("{:?}",info.device_type)}).to_string())
    }
    pub fn upload(
        &mut self,
        #[wasm_bindgen(unchecked_param_type = "Float32Array")] input: JsValue,
    ) -> Result<(), JsValue> {
        self.inner
            .upload(&crate::wgpu_tensor::values(input)?.to_vec())
            .map_err(js_error)
    }
    #[wasm_bindgen(js_name=setInputTensor)]
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
    pub fn sgd(&mut self, gradients: &WasmGraphGradients, rate: f32) -> Result<u64, JsValue> {
        self.inner.sgd(&gradients.inner, rate).map_err(js_error)
    }
    #[wasm_bindgen(js_name=sgdWeighted)]
    pub fn sgd_weighted(
        &mut self,
        batch: &WasmGraphGradientBatch,
        rate: f32,
    ) -> Result<u64, JsValue> {
        self.inner.sgd_batch(&batch.inner, rate).map_err(js_error)
    }
    #[wasm_bindgen(js_name=gradientAccumulator)]
    pub fn gradient_accumulator(&self) -> Result<WasmGraphGradientAccumulator, JsValue> {
        Ok(WasmGraphGradientAccumulator {
            inner: self.inner.gradient_accumulator().map_err(js_error)?,
        })
    }
    #[wasm_bindgen(js_name=zeroAccumulator)]
    pub fn zero_accumulator(
        &self,
        accumulator: &mut WasmGraphGradientAccumulator,
    ) -> Result<(), JsValue> {
        self.inner
            .zero_accumulator(&mut accumulator.inner)
            .map_err(js_error)
    }
    pub fn accumulate(
        &mut self,
        accumulator: &mut WasmGraphGradientAccumulator,
        gradients: &WasmGraphGradients,
        weight: f32,
    ) -> Result<u64, JsValue> {
        self.inner
            .accumulate(&mut accumulator.inner, &gradients.inner, weight)
            .map_err(js_error)
    }
    #[wasm_bindgen(js_name=sgdAccumulated)]
    pub fn sgd_accumulated(
        &mut self,
        accumulator: &WasmGraphGradientAccumulator,
        rate: f32,
    ) -> Result<u64, JsValue> {
        self.inner
            .sgd_accumulated(&accumulator.inner, rate)
            .map_err(js_error)
    }
    #[wasm_bindgen(js_name=parameterSnapshot)]
    pub fn parameter_snapshot(&self) -> Result<WasmGraphTrainingParametersSnapshot, JsValue> {
        Ok(WasmGraphTrainingParametersSnapshot {
            inner: Some(self.inner.parameter_snapshot().map_err(js_error)?),
        })
    }
    #[wasm_bindgen(js_name=updateSnapshot)]
    pub fn update_snapshot(&self) -> Result<WasmGraphUpdateSnapshot, JsValue> {
        let inner = self.inner.update_snapshot().map_err(js_error)?;
        Ok(WasmGraphUpdateSnapshot {
            update: inner.submitted_update(),
            generation: inner.input_generation(),
            forward: inner.submitted_forward(),
            inner: Some(inner),
        })
    }
}

#[wasm_bindgen(js_name=GraphGradientAccumulator)]
pub struct WasmGraphGradientAccumulator {
    #[cfg(feature = "webgpu")]
    inner: backend::GraphGradientAccumulator,
}
#[cfg(feature = "webgpu")]
#[wasm_bindgen(js_class=GraphGradientAccumulator)]
impl WasmGraphGradientAccumulator {
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> u64 {
        self.inner.len()
    }
    #[wasm_bindgen(getter, js_name=parameterGeneration)]
    pub fn parameter_generation(&self) -> u64 {
        self.inner.parameter_generation()
    }
    #[wasm_bindgen(js_name=parameterGradientTensors, unchecked_return_type="WgpuTensor[]")]
    pub fn parameter_gradient_tensors(&self) -> Result<Array, JsValue> {
        Ok(self
            .inner
            .parameter_gradients()
            .map_err(js_error)?
            .into_iter()
            .map(|inner| JsValue::from(WasmWgpuTensor { inner }))
            .collect())
    }
}

#[wasm_bindgen(js_name=GraphUpdateSnapshot)]
pub struct WasmGraphUpdateSnapshot {
    #[cfg(feature = "webgpu")]
    inner: Option<backend::GraphUpdateReadback>,
    #[cfg(feature = "webgpu")]
    update: u64,
    #[cfg(feature = "webgpu")]
    generation: u64,
    #[cfg(feature = "webgpu")]
    forward: u64,
}
#[cfg(feature = "webgpu")]
#[wasm_bindgen(js_class=GraphUpdateSnapshot)]
impl WasmGraphUpdateSnapshot {
    #[wasm_bindgen(getter,js_name=submittedUpdate)]
    pub fn submitted_update(&self) -> u64 {
        self.update
    }
    #[wasm_bindgen(getter,js_name=inputGeneration)]
    pub fn input_generation(&self) -> u64 {
        self.generation
    }
    #[wasm_bindgen(getter,js_name=submittedForward)]
    pub fn submitted_forward(&self) -> u64 {
        self.forward
    }
    #[wasm_bindgen(unchecked_return_type = "Promise<bigint>")]
    pub fn read(&mut self) -> Result<Promise, JsValue> {
        let inner = self
            .inner
            .take()
            .ok_or_else(|| js_error("snapshot has already been consumed"))?;
        Ok(future_to_promise(async move {
            Ok(js_sys::BigInt::from(inner.read_async().await.map_err(js_error)?).into())
        }))
    }
}
