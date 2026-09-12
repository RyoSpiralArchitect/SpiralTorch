//! The same Rust Loss used by native Modules, with explicit resident outputs.
use super::*;
use crate::wgpu_tensor::WasmWgpuTensor;
use st_nn::{CrossEntropyWithLogits, Loss, MeanSquaredError};
use st_tensor::CrossEntropyConfig;

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

#[wasm_bindgen(js_name = CrossEntropyWithLogits)]
pub struct WasmCrossEntropyWithLogits {
    inner: CrossEntropyWithLogits,
}

#[wasm_bindgen(js_class = CrossEntropyWithLogits)]
impl WasmCrossEntropyWithLogits {
    #[wasm_bindgen(constructor)]
    pub fn new(
        reduction: Option<String>,
        ignore_index: Option<js_sys::BigInt>,
        label_smoothing: Option<f64>,
    ) -> Result<Self, JsValue> {
        let ignore_index = match ignore_index {
            None => -100,
            Some(value) => {
                let raw: &JsValue = value.as_ref();
                if !raw.is_bigint() || value != js_sys::BigInt::as_int_n(64., &value) {
                    return Err(js_error(
                        "ignore_index must be a bigint representable as i64",
                    ));
                }
                value
                    .to_string(10)
                    .map_err(|_| js_error("invalid ignore_index"))?
                    .as_string()
                    .ok_or_else(|| js_error("invalid ignore_index"))?
                    .parse::<i64>()
                    .map_err(js_error)?
            }
        };
        let config = CrossEntropyConfig {
            reduction: reduction
                .as_deref()
                .unwrap_or("mean")
                .parse()
                .map_err(js_error)?,
            ignore_index,
            label_smoothing: label_smoothing.unwrap_or(0.),
        };
        Ok(Self {
            inner: CrossEntropyWithLogits::new(config).map_err(js_error)?,
        })
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
