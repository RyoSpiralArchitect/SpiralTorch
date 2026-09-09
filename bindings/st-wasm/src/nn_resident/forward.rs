use super::*;
#[cfg(feature = "webgpu")]
use crate::wgpu_tensor::{WasmWgpuTensor, WasmWgpuTensorDevice};
#[cfg(feature = "webgpu")]
use st_backend_wgpu::resident_graph::{GraphReadback, ResidentGraph};

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
            .compile_graph_wgpu_with_options(runtime, tile, kernel, accumulation)
            .map_err(js_error)?;
        Ok(WasmResidentGraphInference { inner }.into())
    }))
}

#[wasm_bindgen(js_name = ResidentGraphInference)]
pub struct WasmResidentGraphInference {
    #[cfg(feature = "webgpu")]
    inner: ResidentGraph,
}
#[cfg(feature = "webgpu")]
#[wasm_bindgen(js_class = ResidentGraphInference)]
impl WasmResidentGraphInference {
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
    #[wasm_bindgen(getter)]
    pub fn generation(&self) -> u64 {
        self.inner.generation()
    }
    #[wasm_bindgen(getter, js_name = submittedDispatches)]
    pub fn submitted_dispatches(&self) -> u64 {
        self.inner.submitted_dispatches()
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
        crate::utils::json_to_js_value(
            &serde_json::json!({"name":info.name,"backend":format!("{:?}",info.backend),
            "device_type":format!("{:?}",info.device_type)})
            .to_string(),
        )
    }
    pub fn upload(
        &mut self,
        #[wasm_bindgen(unchecked_param_type = "Float32Array")] data: JsValue,
    ) -> Result<(), JsValue> {
        let data = crate::wgpu_tensor::values(data)?;
        self.inner.upload(&data.to_vec()).map_err(js_error)
    }
    #[wasm_bindgen(js_name = setInputTensor)]
    pub fn set_input_tensor(&mut self, input: &WasmWgpuTensor) -> Result<(), JsValue> {
        self.inner.set_input_tensor(&input.inner).map_err(js_error)
    }
    pub fn dispatch(&mut self) -> Result<u64, JsValue> {
        self.inner.dispatch().map_err(js_error)
    }
    #[wasm_bindgen(js_name = outputTensor)]
    pub fn output_tensor(&self) -> Result<WasmWgpuTensor, JsValue> {
        Ok(WasmWgpuTensor {
            inner: self.inner.output_tensor().map_err(js_error)?,
        })
    }
    pub fn snapshot(&self) -> Result<WasmGraphInferenceSnapshot, JsValue> {
        let inner = self.inner.snapshot().map_err(js_error)?;
        Ok(WasmGraphInferenceSnapshot {
            shape: inner.layout().shape().iter().map(|&v| v as u32).collect(),
            generation: inner.generation(),
            dispatch: inner.dispatch(),
            inner: Some(inner),
        })
    }
}

#[wasm_bindgen(js_name = GraphInferenceSnapshot)]
pub struct WasmGraphInferenceSnapshot {
    #[cfg(feature = "webgpu")]
    inner: Option<GraphReadback>,
    #[cfg(feature = "webgpu")]
    shape: Vec<u32>,
    #[cfg(feature = "webgpu")]
    generation: u64,
    #[cfg(feature = "webgpu")]
    dispatch: u64,
}
#[cfg(feature = "webgpu")]
#[wasm_bindgen(js_class = GraphInferenceSnapshot)]
impl WasmGraphInferenceSnapshot {
    #[wasm_bindgen(getter)]
    pub fn shape(&self) -> Vec<u32> {
        self.shape.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn generation(&self) -> u64 {
        self.generation
    }
    #[wasm_bindgen(getter, js_name = submittedDispatch)]
    pub fn submitted_dispatch(&self) -> u64 {
        self.dispatch
    }
    #[wasm_bindgen(js_name = readValues, unchecked_return_type = "Promise<Float32Array>")]
    pub fn read_values(&mut self) -> Result<Promise, JsValue> {
        let inner = self
            .inner
            .take()
            .ok_or_else(|| js_error("snapshot has already been consumed"))?;
        Ok(future_to_promise(async move {
            let data = inner.read_async().await.map_err(js_error)?;
            Ok(Float32Array::from(data.as_slice()).into())
        }))
    }
}
