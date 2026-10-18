use crate::utils::{js_error, js_u32};
use js_sys::Float32Array;
use st_nn::{
    ChannelAffineKernel as CoreKernel, ChannelAffineLearningBatch as CoreBatch, ChannelAffineVjp,
};
use wasm_bindgen::prelude::*;

fn values(input: &JsValue, budget: usize, expected: usize) -> Result<Vec<f32>, JsValue> {
    let array = input
        .dyn_ref::<Float32Array>()
        .ok_or_else(|| js_error("expected a Float32Array"))?;
    let count = array.length() as usize;
    if count > budget || count != expected {
        return Err(js_error(
            "float32 array exceeds budget or differs from shape",
        ));
    }
    Ok(array.to_vec())
}

#[wasm_bindgen]
pub struct ChannelAffineKernel {
    inner: CoreKernel,
}

#[wasm_bindgen]
pub struct ChannelAffineLearningBatch {
    inner: CoreBatch,
}

#[wasm_bindgen]
pub struct ChannelAffinePullback {
    inner: ChannelAffineVjp,
}

#[wasm_bindgen]
impl ChannelAffineKernel {
    #[wasm_bindgen(constructor)]
    pub fn new(max_values: JsValue) -> Result<Self, JsValue> {
        Ok(Self {
            inner: CoreKernel::new(js_u32(&max_values, "max_values")? as usize)
                .map_err(js_error)?,
        })
    }

    pub fn forward(
        &self,
        input: JsValue,
        gate: JsValue,
        bias: JsValue,
        rows: JsValue,
        features: JsValue,
    ) -> Result<ChannelAffineLearningBatch, JsValue> {
        let rows = js_u32(&rows, "rows")? as usize;
        let features = js_u32(&features, "features")? as usize;
        let count = rows
            .checked_mul(features)
            .ok_or_else(|| js_error("shape overflow"))?;
        let input = values(&input, self.inner.max_values(), count)?;
        let gate = values(&gate, self.inner.max_values(), features)?;
        let bias = values(&bias, self.inner.max_values(), features)?;
        self.inner
            .forward(&input, &gate, &bias, rows, features)
            .map(|inner| ChannelAffineLearningBatch { inner })
            .map_err(js_error)
    }
}

#[wasm_bindgen]
impl ChannelAffineLearningBatch {
    #[wasm_bindgen(getter)]
    pub fn output(&self) -> Vec<f32> {
        self.inner.output().data().to_vec()
    }

    pub fn vjp(&self, upstream: JsValue) -> Result<ChannelAffinePullback, JsValue> {
        let count = self.inner.output().data().len();
        let upstream = values(&upstream, count, count)?;
        self.inner
            .vjp(&upstream)
            .map(|inner| ChannelAffinePullback { inner })
            .map_err(js_error)
    }
}

#[wasm_bindgen]
impl ChannelAffinePullback {
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
