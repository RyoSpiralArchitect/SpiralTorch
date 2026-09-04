use js_sys::{Float32Array, Number, Promise};
use st_backend_wgpu::{
    resident_matmul::{MatmulShape, ResidentMatmul},
    runtime,
};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

fn error(err: impl std::fmt::Display) -> JsValue {
    js_sys::Error::new(&err.to_string()).into()
}

fn dimension(value: &JsValue) -> Result<usize, JsValue> {
    let value = crate::utils::js_u32(value, "dimension or repetition")?;
    if value == 0 {
        return Err(error("dimension or repetition must be positive"));
    }
    Ok(value as usize)
}

fn float32_array(value: JsValue) -> Result<Float32Array, JsValue> {
    if !crate::utils::js_is_typed_array(&value, "Float32Array")? {
        return Err(error("operand must be a Float32Array"));
    }
    Ok(value.unchecked_into())
}

/// An explicit WebGPU workspace. It never falls back to WASM CPU tensor math.
#[wasm_bindgen(js_name = WgpuMatmul)]
pub struct WasmWgpuMatmul {
    inner: ResidentMatmul,
}

#[wasm_bindgen(js_class = WgpuMatmul)]
impl WasmWgpuMatmul {
    #[wasm_bindgen(js_name = create)]
    pub async fn create(
        rows: Number,
        inner: Number,
        cols: Number,
    ) -> Result<WasmWgpuMatmul, JsValue> {
        let shape = MatmulShape::new(
            dimension(rows.as_ref())?,
            dimension(inner.as_ref())?,
            dimension(cols.as_ref())?,
        )
        .map_err(error)?;
        let runtime = match runtime::default_runtime() {
            Some(runtime) => runtime,
            None => {
                let candidate = runtime::WgpuRuntime::request_headless("wasm.resident.matmul")
                    .await
                    .map_err(error)?;
                // Concurrent create() promises share whichever runtime installed first.
                let _ = runtime::install_default_runtime(candidate);
                runtime::default_runtime()
                    .ok_or_else(|| error("WebGPU runtime installation failed"))?
            }
        };
        Ok(Self {
            inner: ResidentMatmul::new(runtime, shape).map_err(error)?,
        })
    }

    #[wasm_bindgen(js_name = shape)]
    pub fn shape(&self) -> Vec<u32> {
        let (m, k, n) = self.inner.shape().dimensions();
        vec![m as u32, k as u32, n as u32]
    }

    #[wasm_bindgen(getter, js_name = generation)]
    pub fn generation(&self) -> u64 {
        self.inner.generation()
    }

    #[wasm_bindgen(getter, js_name = outputIsCurrent)]
    pub fn output_is_current(&self) -> bool {
        self.inner.output_is_current()
    }

    #[wasm_bindgen(js_name = adapterInfo)]
    pub fn adapter_info(&self) -> Result<JsValue, JsValue> {
        let info = self.inner.adapter_info();
        serde_wasm_bindgen::to_value(&serde_json::json!({
            "name": info.name, "vendor": info.vendor, "device": info.device,
            "backend": format!("{:?}", info.backend), "device_type": format!("{:?}", info.device_type),
            "driver": info.driver, "driver_info": info.driver_info,
        })).map_err(error)
    }

    #[wasm_bindgen(js_name = upload)]
    pub fn upload(&mut self, lhs: Float32Array, rhs: Float32Array) -> Result<(), JsValue> {
        let lhs = float32_array(lhs.into())?;
        let rhs = float32_array(rhs.into())?;
        let (m, k, n) = self.inner.shape().dimensions();
        if lhs.length() as usize != m * k || rhs.length() as usize != k * n {
            return Err(error("operand lengths must match the workspace"));
        }
        self.inner
            .upload(&lhs.to_vec(), &rhs.to_vec())
            .map_err(error)
    }

    #[wasm_bindgen(js_name = uploadRhs)]
    pub fn upload_rhs(&mut self, rhs: Float32Array) -> Result<(), JsValue> {
        let rhs = float32_array(rhs.into())?;
        let (_, k, n) = self.inner.shape().dimensions();
        if rhs.length() as usize != k * n {
            return Err(error("RHS length must match the workspace"));
        }
        self.inner.upload_rhs(&rhs.to_vec()).map_err(error)
    }

    #[wasm_bindgen(js_name = setLhsFrom)]
    pub fn set_lhs_from(&mut self, source: &WasmWgpuMatmul) -> Result<(), JsValue> {
        self.inner.set_lhs_from(&source.inner).map_err(error)
    }

    /// Repetitions are validated before any JS-to-u32 narrowing.
    #[wasm_bindgen(js_name = dispatch)]
    pub fn dispatch(&mut self, repetitions: Option<Number>) -> Result<u64, JsValue> {
        let repetitions = match repetitions {
            None => 1,
            Some(value) => dimension(value.as_ref())? as u32,
        };
        self.inner.dispatch(repetitions).map_err(error)
    }

    #[wasm_bindgen(js_name = synchronize, unchecked_return_type = "Promise<void>")]
    pub fn synchronize(&self) -> Result<Promise, JsValue> {
        let completion = self.inner.synchronize_async().map_err(error)?;
        Ok(future_to_promise(async move {
            completion.await.map_err(error)?;
            Ok(JsValue::UNDEFINED)
        }))
    }

    /// Snapshot synchronously, then await mapping without holding a JS object borrow.
    #[wasm_bindgen(js_name = readback, unchecked_return_type = "Promise<Float32Array>")]
    pub fn readback(&self) -> Result<Promise, JsValue> {
        let snapshot = self.inner.snapshot().map_err(error)?;
        Ok(future_to_promise(async move {
            let values = snapshot.read_async().await.map_err(error)?;
            Ok(Float32Array::from(values.as_slice()).into())
        }))
    }
}
