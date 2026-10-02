use crate::utils::{js_error, js_u32};
use js_sys::Number;
use st_nn::{WaveGateKernel as CoreKernel, WaveGateLearningBatch as CoreBatch, WaveGateVjp};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WaveGateKernel {
    inner: CoreKernel,
}

#[wasm_bindgen]
pub struct WaveGateLearningBatch {
    inner: CoreBatch,
}

#[wasm_bindgen]
pub struct WaveGatePullback {
    inner: WaveGateVjp,
}

fn number(value: &Number, label: &str) -> Result<f32, JsValue> {
    let raw: &JsValue = value.as_ref();
    raw.as_f64()
        .map(|value| value as f32)
        .ok_or_else(|| js_error(format!("{label} must be a number")))
}

#[wasm_bindgen]
impl WaveGateKernel {
    #[wasm_bindgen(constructor)]
    pub fn new(
        curvature: Number,
        saturation: Number,
        porosity: Number,
        max_values: Number,
    ) -> Result<Self, JsValue> {
        Ok(Self {
            inner: CoreKernel::new(
                number(&curvature, "curvature")?,
                number(&saturation, "saturation")?,
                number(&porosity, "porosity")?,
                js_u32(max_values.as_ref(), "max_values")? as usize,
            )
            .map_err(js_error)?,
        })
    }

    pub fn forward(
        &self,
        input: &[f32],
        gate: &[f32],
        bias: &[f32],
        rows: Number,
        features: Number,
    ) -> Result<WaveGateLearningBatch, JsValue> {
        self.inner
            .forward(
                input,
                gate,
                bias,
                js_u32(rows.as_ref(), "rows")? as usize,
                js_u32(features.as_ref(), "features")? as usize,
            )
            .map(|inner| WaveGateLearningBatch { inner })
            .map_err(js_error)
    }
}

#[wasm_bindgen]
impl WaveGateLearningBatch {
    #[wasm_bindgen(getter)]
    pub fn output(&self) -> Vec<f32> {
        self.inner.output().data().to_vec()
    }

    pub fn conditioning_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.inner.conditioning()).map_err(js_error)
    }

    pub fn vjp(&self, upstream: &[f32]) -> Result<WaveGatePullback, JsValue> {
        self.inner
            .vjp(upstream)
            .map(|inner| WaveGatePullback { inner })
            .map_err(js_error)
    }
}

#[wasm_bindgen]
impl WaveGatePullback {
    #[wasm_bindgen(getter)]
    pub fn grad_input(&self) -> Vec<f32> {
        self.inner.grad_input.data().to_vec()
    }
    #[wasm_bindgen(getter)]
    pub fn grad_gate(&self) -> Vec<f32> {
        self.inner.grad_gate.data().to_vec()
    }
    #[wasm_bindgen(getter)]
    pub fn grad_bias(&self) -> Vec<f32> {
        self.inner.grad_bias.data().to_vec()
    }
}
