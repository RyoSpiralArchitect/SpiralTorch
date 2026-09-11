//! Browser handles over the same portable Rust NN plan used by Python.

use crate::utils::{js_error, js_u32};
use js_sys::{Array, JsString, Number, Promise};
use st_nn::resident::{InferencePlan, DEFAULT_MAX_PLAN_JSON_BYTES};
use wasm_bindgen::prelude::*;

#[cfg(feature = "webgpu")]
use js_sys::Float32Array;
#[cfg(feature = "webgpu")]
use st_backend_wgpu::{
    resident_dense::{DenseReadback, ResidentDense},
    resident_matmul::{MatmulAccumulation, MatmulKernel, MatmulTile},
};
#[cfg(feature = "webgpu")]
use wasm_bindgen_futures::future_to_promise;

mod autograd;
mod forward;
pub use autograd::{WasmGraphForward, WasmGraphGradients, WasmResidentGraphAutograd};
mod graph;
mod learner;
pub use forward::{WasmGraphInferenceSnapshot, WasmResidentGraphInference};
pub use graph::{
    WasmGraphTrainingParametersSnapshot, WasmGraphTrainingSnapshot, WasmGraphTrainingState,
    WasmResidentGraphTraining,
};
pub use learner::{WasmGraphGradientBatch, WasmGraphUpdateSnapshot, WasmResidentGraphLearner};
mod training;
pub use training::{
    WasmResidentTraining, WasmTrainingLossSnapshot, WasmTrainingParametersSnapshot,
    WasmTrainingSnapshot, WasmTrainingState,
};

#[cfg(feature = "webgpu")]
fn gpu_options(
    tile_mnk: Option<Array>,
    kernel: Option<JsString>,
    accumulation: Option<JsString>,
) -> Result<(MatmulTile, MatmulKernel, MatmulAccumulation), JsValue> {
    let tile = if let Some(values) = tile_mnk {
        if values.length() != 3 {
            return Err(js_error("tile_mnk must have three dimensions"));
        }
        MatmulTile::new(
            js_u32(&values.get(0), "tile_m")?,
            js_u32(&values.get(1), "tile_n")?,
            js_u32(&values.get(2), "tile_k")?,
        )
        .map_err(js_error)?
    } else {
        MatmulTile::default()
    };
    let kernel = kernel
        .map(|v| {
            v.as_string()
                .ok_or_else(|| js_error("kernel must be a string"))
        })
        .transpose()?
        .map(|v| v.parse::<MatmulKernel>().map_err(js_error))
        .transpose()?
        .unwrap_or(MatmulKernel::Scalar);
    let accumulation = accumulation
        .map(|v| {
            v.as_string()
                .ok_or_else(|| js_error("accumulation must be a string"))
        })
        .transpose()?
        .map(|v| v.parse::<MatmulAccumulation>().map_err(js_error))
        .transpose()?
        .unwrap_or_default();
    Ok((tile, kernel, accumulation))
}

#[wasm_bindgen(js_name = InferencePlan)]
pub struct WasmInferencePlan {
    inner: InferencePlan,
}

#[wasm_bindgen(js_class = InferencePlan)]
impl WasmInferencePlan {
    #[wasm_bindgen(js_name = fromJson)]
    pub fn from_json(
        payload: JsString,
        max_bytes: Option<Number>,
    ) -> Result<WasmInferencePlan, JsValue> {
        let max_bytes = max_bytes
            .as_ref()
            .map(|value| js_u32(value.as_ref(), "max_bytes"))
            .transpose()?
            .map(|n| n as usize)
            .unwrap_or(DEFAULT_MAX_PLAN_JSON_BYTES);
        let value: &JsValue = payload.as_ref();
        if !value.is_string() {
            return Err(js_error("plan must be a JSON string"));
        }
        // UTF-16 length is a lower bound on UTF-8 bytes; bound the FFI copy too.
        if payload.length() as usize > max_bytes {
            return Err(js_error("plan string already exceeds the JSON byte budget"));
        }
        let payload = payload.as_string().unwrap();
        let inner = InferencePlan::from_json_with_limit(&payload, max_bytes).map_err(js_error)?;
        Ok(Self { inner })
    }

    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> Result<String, JsValue> {
        self.inner.to_json().map_err(js_error)
    }

    /// Return a new Rust-fused plan; parameter IDs stay fixed, stage IDs may change.
    #[wasm_bindgen(js_name = fusePointwise)]
    pub fn fuse_pointwise(&self) -> Result<WasmInferencePlan, JsValue> {
        Ok(Self {
            inner: self.inner.fuse_pointwise().map_err(js_error)?,
        })
    }

    #[wasm_bindgen(getter, js_name = inputShape)]
    pub fn input_shape(&self) -> Vec<u32> {
        self.inner
            .input_layout()
            .shape()
            .iter()
            .map(|&d| d as u32)
            .collect()
    }
    #[wasm_bindgen(getter, js_name = outputShape)]
    pub fn output_shape(&self) -> Vec<u32> {
        self.inner
            .output_layout()
            .shape()
            .iter()
            .map(|&d| d as u32)
            .collect()
    }
    #[wasm_bindgen(getter, js_name = stageCount)]
    pub fn stage_count(&self) -> usize {
        self.inner.stage_count()
    }
    #[wasm_bindgen(getter, js_name = sourceOperationCount)]
    pub fn source_operation_count(&self) -> usize {
        self.inner.source_operation_count()
    }

    #[wasm_bindgen(getter, js_name = isDense)]
    pub fn is_dense(&self) -> bool {
        self.inner.is_dense()
    }

    #[cfg(feature = "webgpu")]
    #[wasm_bindgen(js_name = compileWebGpu, unchecked_return_type = "Promise<ResidentInference>")]
    pub fn compile_webgpu(
        &self,
        tile_mnk: Option<Array>,
        kernel: Option<JsString>,
        accumulation: Option<JsString>,
    ) -> Result<Promise, JsValue> {
        require_dense(&self.inner)?;
        let (tile, kernel, accumulation) = gpu_options(tile_mnk, kernel, accumulation)?;
        // Clone before returning the promise: freeing the JS plan is safe while it compiles.
        let plan = self.inner.clone();
        Ok(future_to_promise(async move {
            let runtime = crate::wgpu_resident::ensure_runtime().await?;
            let inner = plan
                .compile_wgpu_with_options(runtime, tile, kernel, accumulation)
                .map_err(js_error)?;
            Ok(WasmResidentInference { inner }.into())
        }))
    }

    #[cfg(not(feature = "webgpu"))]
    #[wasm_bindgen(js_name = compileWebGpu, unchecked_return_type = "Promise<ResidentInference>")]
    pub fn compile_webgpu(&self) -> Result<Promise, JsValue> {
        Err(js_error(
            "resident inference requires the webgpu build feature",
        ))
    }

    #[wasm_bindgen(js_name = compileTrainingWebGpu, unchecked_return_type = "Promise<ResidentTraining>")]
    pub fn compile_training_webgpu(
        &self,
        tile_mnk: Option<Array>,
        kernel: Option<JsString>,
        accumulation: Option<JsString>,
    ) -> Result<Promise, JsValue> {
        #[cfg(feature = "webgpu")]
        {
            training::compile(&self.inner, tile_mnk, kernel, accumulation)
        }
        #[cfg(not(feature = "webgpu"))]
        {
            let _ = (tile_mnk, kernel, accumulation);
            Err(js_error(
                "resident training requires the webgpu build feature",
            ))
        }
    }

    #[wasm_bindgen(js_name = compileGraphWebGpu, unchecked_return_type = "Promise<ResidentGraphInference>")]
    pub fn compile_graph_webgpu(
        &self,
        tile_mnk: Option<Array>,
        kernel: Option<JsString>,
        accumulation: Option<JsString>,
    ) -> Result<Promise, JsValue> {
        #[cfg(feature = "webgpu")]
        {
            forward::compile(&self.inner, tile_mnk, kernel, accumulation)
        }
        #[cfg(not(feature = "webgpu"))]
        {
            let _ = (tile_mnk, kernel, accumulation);
            Err(js_error(
                "resident graph inference requires the webgpu build feature",
            ))
        }
    }

    #[wasm_bindgen(js_name = compileGraphAutogradWebGpu, unchecked_return_type = "Promise<ResidentGraphAutograd>")]
    pub fn compile_graph_autograd_webgpu(
        &self,
        tile_mnk: Option<Array>,
        kernel: Option<JsString>,
        accumulation: Option<JsString>,
    ) -> Result<Promise, JsValue> {
        #[cfg(feature = "webgpu")]
        {
            autograd::compile(&self.inner, tile_mnk, kernel, accumulation)
        }
        #[cfg(not(feature = "webgpu"))]
        {
            let _ = (tile_mnk, kernel, accumulation);
            Err(js_error(
                "resident graph autograd requires the webgpu build feature",
            ))
        }
    }

    #[wasm_bindgen(js_name = compileGraphTrainingWebGpu, unchecked_return_type = "Promise<ResidentGraphTraining>")]
    pub fn compile_graph_training_webgpu(
        &self,
        gradient_policy: JsString,
        tile_mnk: Option<Array>,
        kernel: Option<JsString>,
        accumulation: Option<JsString>,
    ) -> Result<Promise, JsValue> {
        let policy = gradient_policy
            .as_string()
            .ok_or_else(|| js_error("gradient_policy must be a string"))?
            .parse::<st_nn::resident::GraphGradientPolicy>()
            .map_err(js_error)?;
        #[cfg(feature = "webgpu")]
        {
            graph::compile(&self.inner, policy, tile_mnk, kernel, accumulation)
        }
        #[cfg(not(feature = "webgpu"))]
        {
            let _ = (policy, tile_mnk, kernel, accumulation);
            Err(js_error(
                "resident graph training requires the webgpu build feature",
            ))
        }
    }

    #[wasm_bindgen(js_name=compileGraphLearnerWebGpu,unchecked_return_type="Promise<ResidentGraphLearner>")]
    pub fn compile_graph_learner_webgpu(
        &self,
        gradient_policy: JsString,
        tile_mnk: Option<Array>,
        kernel: Option<JsString>,
        accumulation: Option<JsString>,
    ) -> Result<Promise, JsValue> {
        let policy = gradient_policy
            .as_string()
            .ok_or_else(|| js_error("gradient_policy must be a string"))?
            .parse::<st_nn::resident::GraphGradientPolicy>()
            .map_err(js_error)?;
        #[cfg(feature = "webgpu")]
        {
            learner::compile(&self.inner, policy, tile_mnk, kernel, accumulation)
        }
        #[cfg(not(feature = "webgpu"))]
        {
            let _ = (policy, tile_mnk, kernel, accumulation);
            Err(js_error(
                "resident graph learning requires the webgpu build feature",
            ))
        }
    }
}

#[cfg(feature = "webgpu")]
fn require_dense(plan: &InferencePlan) -> Result<(), JsValue> {
    if !plan.is_dense() {
        return Err(js_error(st_nn::resident::InferenceError::RequiresGraph));
    }
    Ok(())
}

#[wasm_bindgen(js_name = ResidentInference)]
pub struct WasmResidentInference {
    #[cfg(feature = "webgpu")]
    inner: ResidentDense,
}

#[cfg(feature = "webgpu")]
#[wasm_bindgen(js_class = ResidentInference)]
impl WasmResidentInference {
    #[wasm_bindgen(js_name = setInputTensor)]
    pub fn set_input_tensor(
        &mut self,
        input: &crate::wgpu_tensor::WasmWgpuTensor,
    ) -> Result<(), JsValue> {
        self.inner.set_input_tensor(&input.inner).map_err(js_error)
    }
    #[wasm_bindgen(js_name = tensorSnapshot)]
    pub fn tensor_snapshot(
        &self,
        device: &crate::wgpu_tensor::WasmWgpuTensorDevice,
    ) -> Result<crate::wgpu_tensor::WasmWgpuTensor, JsValue> {
        Ok(crate::wgpu_tensor::WasmWgpuTensor {
            inner: self
                .inner
                .tensor_snapshot(&device.inner)
                .map_err(js_error)?,
        })
    }
    #[wasm_bindgen(getter, js_name = inputShape)]
    pub fn input_shape(&self) -> Vec<u32> {
        self.inner
            .input_layout()
            .shape()
            .iter()
            .map(|&d| d as u32)
            .collect()
    }
    #[wasm_bindgen(getter, js_name = outputShape)]
    pub fn output_shape(&self) -> Vec<u32> {
        self.inner
            .output_layout()
            .shape()
            .iter()
            .map(|&d| d as u32)
            .collect()
    }
    #[wasm_bindgen(getter, js_name = stageCount)]
    pub fn stage_count(&self) -> usize {
        self.inner.stage_count()
    }
    #[wasm_bindgen(getter)]
    pub fn generation(&self) -> u64 {
        self.inner.generation()
    }

    #[wasm_bindgen(js_name = adapterInfo, unchecked_return_type = "{ name: string; backend: string; device_type: string }")]
    pub fn adapter_info(&self) -> Result<JsValue, JsValue> {
        let info = self.inner.adapter_info();
        crate::utils::json_to_js_value(
            &serde_json::json!({
                "name": info.name, "backend": format!("{:?}", info.backend),
                "device_type": format!("{:?}", info.device_type),
            })
            .to_string(),
        )
    }

    pub fn upload(
        &mut self,
        #[wasm_bindgen(unchecked_param_type = "Float32Array")] values: JsValue,
    ) -> Result<(), JsValue> {
        if !crate::utils::js_is_typed_array(&values, "Float32Array")? {
            return Err(js_error("input must be a Float32Array"));
        }
        let values: Float32Array = values.unchecked_into();
        self.inner.upload(&values.to_vec()).map_err(js_error)
    }

    pub fn dispatch(&mut self) -> Result<u64, JsValue> {
        self.inner.dispatch().map_err(js_error)
    }

    pub fn snapshot(&self) -> Result<WasmInferenceSnapshot, JsValue> {
        let snapshot = self.inner.snapshot().map_err(js_error)?;
        Ok(WasmInferenceSnapshot {
            shape: snapshot
                .layout()
                .shape()
                .iter()
                .map(|&d| d as u32)
                .collect(),
            generation: snapshot.generation(),
            inner: Some(snapshot),
        })
    }
}

#[wasm_bindgen(js_name = InferenceSnapshot)]
pub struct WasmInferenceSnapshot {
    #[cfg(feature = "webgpu")]
    inner: Option<DenseReadback>,
    #[cfg(feature = "webgpu")]
    shape: Vec<u32>,
    #[cfg(feature = "webgpu")]
    generation: u64,
}

#[cfg(feature = "webgpu")]
#[wasm_bindgen(js_class = InferenceSnapshot)]
impl WasmInferenceSnapshot {
    #[wasm_bindgen(getter)]
    pub fn shape(&self) -> Vec<u32> {
        self.shape.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    #[wasm_bindgen(js_name = readValues, unchecked_return_type = "Promise<Float32Array>")]
    pub fn read_values(&mut self) -> Result<Promise, JsValue> {
        let snapshot = self
            .inner
            .take()
            .ok_or_else(|| js_error("snapshot has already been consumed"))?;
        Ok(future_to_promise(async move {
            let values = snapshot.read_async().await.map_err(js_error)?;
            Ok(Float32Array::from(values.as_slice()).into())
        }))
    }
}
