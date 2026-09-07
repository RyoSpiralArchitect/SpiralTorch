//! Browser handles over the same portable Rust NN plan used by Python.

use crate::utils::{js_error, js_u32};
use js_sys::{JsString, Number, Promise};
use st_nn::resident::{InferencePlan, DEFAULT_MAX_PLAN_JSON_BYTES};
use wasm_bindgen::prelude::*;

#[cfg(feature = "webgpu")]
use js_sys::{Array, Float32Array};
#[cfg(feature = "webgpu")]
use st_backend_wgpu::{
    resident_dense::{DenseReadback, ResidentDense},
    resident_matmul::{MatmulAccumulation, MatmulKernel, MatmulTile},
};
#[cfg(feature = "webgpu")]
use wasm_bindgen_futures::future_to_promise;

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
        Ok(Self {
            inner: InferencePlan::from_json_with_limit(&payload, max_bytes).map_err(js_error)?,
        })
    }

    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> Result<String, JsValue> {
        self.inner.to_json().map_err(js_error)
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

    #[cfg(feature = "webgpu")]
    #[wasm_bindgen(js_name = compileWebGpu, unchecked_return_type = "Promise<ResidentInference>")]
    pub fn compile_webgpu(
        &self,
        tile_mnk: Option<Array>,
        kernel: Option<JsString>,
        accumulation: Option<JsString>,
    ) -> Result<Promise, JsValue> {
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
            .map(|value| {
                value
                    .as_string()
                    .ok_or_else(|| js_error("kernel must be a string"))
            })
            .transpose()?
            .map(|value| value.parse::<MatmulKernel>().map_err(js_error))
            .transpose()?
            .unwrap_or(MatmulKernel::Scalar);
        let accumulation = accumulation
            .map(|value| {
                value
                    .as_string()
                    .ok_or_else(|| js_error("accumulation must be a string"))
            })
            .transpose()?
            .map(|value| value.parse::<MatmulAccumulation>().map_err(js_error))
            .transpose()?
            .unwrap_or_default();
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
}

#[wasm_bindgen(js_name = ResidentInference)]
pub struct WasmResidentInference {
    #[cfg(feature = "webgpu")]
    inner: ResidentDense,
}

#[cfg(feature = "webgpu")]
#[wasm_bindgen(js_class = ResidentInference)]
impl WasmResidentInference {
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
