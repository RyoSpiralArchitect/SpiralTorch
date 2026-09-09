//! Browser ownership handles over the same Rust N-D storage as Python.
use crate::utils::{js_error, js_u32};
use js_sys::{Array, Float32Array, Number, Promise};
use st_backend_wgpu::resident_tensor::{ResidentTensor, TensorDevice, TensorReadback};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

fn dimensions(value: &JsValue) -> Result<Vec<usize>, JsValue> {
    if !Array::is_array(value) {
        return Err(js_error("shape/axes must be an array of integers"));
    }
    let values: &Array = value.unchecked_ref();
    values
        .iter()
        .map(|v| js_u32(&v, "dimension/index").map(|n| n as usize))
        .collect()
}
pub(crate) fn values(value: JsValue) -> Result<Float32Array, JsValue> {
    if !crate::utils::js_is_typed_array(&value, "Float32Array")? {
        return Err(js_error("values must be a Float32Array"));
    }
    Ok(value.unchecked_into())
}

#[wasm_bindgen(js_name = WgpuTensorDevice)]
pub struct WasmWgpuTensorDevice {
    pub(crate) inner: TensorDevice,
}
#[wasm_bindgen(js_class = WgpuTensorDevice)]
impl WasmWgpuTensorDevice {
    pub async fn create() -> Result<WasmWgpuTensorDevice, JsValue> {
        let runtime = crate::wgpu_resident::ensure_runtime().await?;
        Ok(Self {
            inner: TensorDevice::new(runtime).map_err(js_error)?,
        })
    }
    pub fn upload(
        &self,
        #[wasm_bindgen(unchecked_param_type = "number[]")] shape: JsValue,
        #[wasm_bindgen(unchecked_param_type = "Float32Array")] data: JsValue,
    ) -> Result<WasmWgpuTensor, JsValue> {
        let shape = dimensions(&shape)?;
        let data = values(data)?;
        Ok(WasmWgpuTensor {
            inner: self
                .inner
                .upload(&shape, &data.to_vec())
                .map_err(js_error)?,
        })
    }
    #[wasm_bindgen(js_name = adapterInfo, unchecked_return_type = "{ name: string; backend: string; device_type: string }")]
    pub fn adapter_info(&self) -> Result<JsValue, JsValue> {
        let info = self.inner.runtime().adapter_info();
        crate::utils::json_to_js_value(
            &serde_json::json!({"name":info.name,
            "backend":format!("{:?}",info.backend),"device_type":format!("{:?}",info.device_type)})
            .to_string(),
        )
    }
}

#[wasm_bindgen(js_name = WgpuTensor)]
pub struct WasmWgpuTensor {
    pub(crate) inner: ResidentTensor,
}
#[wasm_bindgen(js_class = WgpuTensor)]
impl WasmWgpuTensor {
    #[wasm_bindgen(getter)]
    pub fn shape(&self) -> Vec<u32> {
        self.inner
            .layout()
            .shape()
            .iter()
            .map(|&v| v as u32)
            .collect()
    }
    #[wasm_bindgen(getter)]
    pub fn strides(&self) -> Vec<u32> {
        self.inner
            .layout()
            .strides()
            .iter()
            .map(|&v| v as u32)
            .collect()
    }
    #[wasm_bindgen(getter)]
    pub fn offset(&self) -> usize {
        self.inner.layout().offset()
    }
    #[wasm_bindgen(getter)]
    pub fn numel(&self) -> usize {
        self.inner.layout().len()
    }
    #[wasm_bindgen(getter, js_name = isContiguous)]
    pub fn is_contiguous(&self) -> bool {
        self.inner.layout().is_contiguous()
    }
    pub fn device(&self) -> WasmWgpuTensorDevice {
        WasmWgpuTensorDevice {
            inner: self.inner.device().clone(),
        }
    }
    #[wasm_bindgen(js_name = sharesStorageWith)]
    pub fn shares_storage_with(&self, other: &Self) -> bool {
        self.inner.shares_storage_with(&other.inner)
    }
    pub fn reshape(
        &self,
        #[wasm_bindgen(unchecked_param_type = "number[]")] shape: JsValue,
    ) -> Result<WasmWgpuTensor, JsValue> {
        Ok(Self {
            inner: self.inner.reshape(&dimensions(&shape)?).map_err(js_error)?,
        })
    }
    pub fn permute(
        &self,
        #[wasm_bindgen(unchecked_param_type = "number[]")] axes: JsValue,
    ) -> Result<WasmWgpuTensor, JsValue> {
        Ok(Self {
            inner: self.inner.permute(&dimensions(&axes)?).map_err(js_error)?,
        })
    }
    #[wasm_bindgen(js_name = broadcastTo)]
    pub fn broadcast_to(
        &self,
        #[wasm_bindgen(unchecked_param_type = "number[]")] shape: JsValue,
    ) -> Result<WasmWgpuTensor, JsValue> {
        Ok(Self {
            inner: self
                .inner
                .broadcast_to(&dimensions(&shape)?)
                .map_err(js_error)?,
        })
    }
    pub fn narrow(
        &self,
        axis: Number,
        start: Number,
        length: Number,
    ) -> Result<WasmWgpuTensor, JsValue> {
        Ok(Self {
            inner: self
                .inner
                .narrow(
                    js_u32(axis.as_ref(), "axis")? as usize,
                    js_u32(start.as_ref(), "start")? as usize,
                    js_u32(length.as_ref(), "length")? as usize,
                )
                .map_err(js_error)?,
        })
    }
    pub fn contiguous(&self) -> Result<WasmWgpuTensor, JsValue> {
        Ok(Self {
            inner: self.inner.contiguous().map_err(js_error)?,
        })
    }
    pub fn add(&self, rhs: &Self) -> Result<WasmWgpuTensor, JsValue> {
        Ok(Self {
            inner: self.inner.add(&rhs.inner).map_err(js_error)?,
        })
    }
    pub fn mul(&self, rhs: &Self) -> Result<WasmWgpuTensor, JsValue> {
        Ok(Self {
            inner: self.inner.mul(&rhs.inner).map_err(js_error)?,
        })
    }
    pub fn relu(&self) -> Result<WasmWgpuTensor, JsValue> {
        Ok(Self {
            inner: self.inner.relu().map_err(js_error)?,
        })
    }
    pub fn gelu(&self) -> Result<WasmWgpuTensor, JsValue> {
        Ok(Self {
            inner: self.inner.gelu().map_err(js_error)?,
        })
    }
    pub fn snapshot(&self) -> Result<WasmWgpuTensorSnapshot, JsValue> {
        let inner = self.inner.snapshot().map_err(js_error)?;
        Ok(WasmWgpuTensorSnapshot {
            shape: inner.layout().shape().iter().map(|&v| v as u32).collect(),
            inner: Some(inner),
        })
    }
}

#[wasm_bindgen(js_name = WgpuTensorSnapshot)]
pub struct WasmWgpuTensorSnapshot {
    inner: Option<TensorReadback>,
    shape: Vec<u32>,
}
#[wasm_bindgen(js_class = WgpuTensorSnapshot)]
impl WasmWgpuTensorSnapshot {
    #[wasm_bindgen(getter)]
    pub fn shape(&self) -> Vec<u32> {
        self.shape.clone()
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
