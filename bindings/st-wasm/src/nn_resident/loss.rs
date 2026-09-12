//! The same Rust Loss used by native Modules, with explicit resident outputs.
use super::*;
use crate::wgpu_tensor::WasmWgpuTensor;
use st_nn::{Loss, MeanSquaredError};

#[wasm_bindgen(js_name = ResidentLoss)]
pub struct WasmResidentLoss {
    inner: st_backend_wgpu::resident_tensor::loss::ResidentLoss,
}

#[wasm_bindgen(js_class = ResidentLoss)]
impl WasmResidentLoss {
    #[wasm_bindgen(js_name = lossTensor)]
    pub fn loss_tensor(&self) -> WasmWgpuTensor {
        WasmWgpuTensor {
            inner: self.inner.value().clone(),
        }
    }
    #[wasm_bindgen(js_name = predictionGradientTensor)]
    pub fn prediction_gradient_tensor(&self) -> WasmWgpuTensor {
        WasmWgpuTensor {
            inner: self.inner.prediction_gradient().clone(),
        }
    }
}

#[wasm_bindgen(js_name = MeanSquaredError)]
pub struct WasmMeanSquaredError {
    inner: MeanSquaredError,
}

#[wasm_bindgen(js_class = MeanSquaredError)]
impl WasmMeanSquaredError {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: MeanSquaredError::new(),
        }
    }

    #[wasm_bindgen(js_name = evaluateResident)]
    pub fn evaluate_resident(
        &mut self,
        prediction: &WasmWgpuTensor,
        target: &WasmWgpuTensor,
    ) -> Result<WasmResidentLoss, JsValue> {
        Ok(WasmResidentLoss {
            inner: self
                .inner
                .evaluate_resident(&prediction.inner, &target.inner)
                .map_err(js_error)?,
        })
    }
}

impl Default for WasmMeanSquaredError {
    fn default() -> Self {
        Self::new()
    }
}
