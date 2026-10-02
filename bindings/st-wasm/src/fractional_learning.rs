use crate::utils::{js_error, js_u32};
use js_sys::Number;
use st_frac::learning::{
    FractionalGlGradients as CoreGradients, FractionalGlKernel as CoreKernel,
    FractionalGlLearningBatch as CoreBatch,
};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct FractionalGlKernel {
    inner: CoreKernel,
}

#[wasm_bindgen]
pub struct FractionalGlLearningBatch {
    inner: CoreBatch,
}

#[wasm_bindgen]
pub struct FractionalGlGradients {
    inner: CoreGradients,
}

fn number(value: &Number, label: &str) -> Result<f32, JsValue> {
    let raw: &JsValue = value.as_ref();
    raw.as_f64()
        .map(|v| v as f32)
        .ok_or_else(|| js_error(format!("{label} must be a number")))
}

#[wasm_bindgen]
impl FractionalGlKernel {
    #[wasm_bindgen(constructor)]
    pub fn new(
        kernel_len: Number,
        step: Number,
        max_values: Number,
        max_products: Number,
    ) -> Result<Self, JsValue> {
        CoreKernel::new(
            js_u32(kernel_len.as_ref(), "kernel_len")? as usize,
            number(&step, "step")?,
            js_u32(max_values.as_ref(), "max_values")? as usize,
            js_u32(max_products.as_ref(), "max_products")? as usize,
        )
        .map(|inner| Self { inner })
        .map_err(js_error)
    }

    pub fn forward(
        &self,
        input: &[f32],
        shape: &[u32],
        axis: Number,
        alpha: Number,
    ) -> Result<FractionalGlLearningBatch, JsValue> {
        self.inner
            .forward(
                input,
                &shape.iter().map(|&v| v as usize).collect::<Vec<_>>(),
                js_u32(axis.as_ref(), "axis")? as usize,
                number(&alpha, "alpha")?,
            )
            .map(|inner| FractionalGlLearningBatch { inner })
            .map_err(js_error)
    }

    pub fn forward_history(
        &self,
        input: &[f32],
        shape: &[u32],
        axis: Number,
        alpha: Number,
    ) -> Result<FractionalGlLearningBatch, JsValue> {
        self.inner
            .forward_history(
                input,
                &shape.iter().map(|&v| v as usize).collect::<Vec<_>>(),
                js_u32(axis.as_ref(), "axis")? as usize,
                number(&alpha, "alpha")?,
            )
            .map(|inner| FractionalGlLearningBatch { inner })
            .map_err(js_error)
    }
}

#[wasm_bindgen]
impl FractionalGlLearningBatch {
    #[wasm_bindgen(getter)]
    pub fn output(&self) -> Vec<f32> {
        self.inner.output().iter().copied().collect()
    }

    pub fn vjp(&self, upstream: &[f32]) -> Result<FractionalGlGradients, JsValue> {
        self.inner
            .vjp(upstream)
            .map(|inner| FractionalGlGradients { inner })
            .map_err(js_error)
    }

    pub fn jvp(&self, input_tangent: &[f32], alpha_tangent: Number) -> Result<Vec<f32>, JsValue> {
        self.inner
            .jvp(input_tangent, number(&alpha_tangent, "alpha_tangent")?)
            .map_err(js_error)
    }
}

#[wasm_bindgen]
impl FractionalGlGradients {
    #[wasm_bindgen(getter)]
    pub fn input(&self) -> Vec<f32> {
        self.inner.input.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn alpha(&self) -> f32 {
        self.inner.alpha
    }
}
