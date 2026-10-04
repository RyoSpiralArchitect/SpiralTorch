use crate::utils::{js_error, js_u32};
use js_sys::Number;
use st_frac::learning::{
    FractionalGlGainGradients as CoreGainGradients, FractionalGlGainLearningBatch as CoreGainBatch,
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

#[wasm_bindgen]
pub struct FractionalGlGainLearningBatch {
    inner: CoreGainBatch,
}

#[wasm_bindgen]
pub struct FractionalGlGainGradients {
    inner: CoreGainGradients,
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

    pub fn gain_from_log_gain(log_gain: Number) -> Result<f32, JsValue> {
        CoreKernel::gain_from_log_gain(number(&log_gain, "log_gain")?).map_err(js_error)
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

    pub fn forward_history_l2(
        &self,
        input: &[f32],
        shape: &[u32],
        axis: Number,
        alpha: Number,
        gain: Number,
    ) -> Result<FractionalGlLearningBatch, JsValue> {
        self.inner
            .forward_history_l2(
                input,
                &shape.iter().map(|&v| v as usize).collect::<Vec<_>>(),
                js_u32(axis.as_ref(), "axis")? as usize,
                number(&alpha, "alpha")?,
                number(&gain, "gain")?,
            )
            .map(|inner| FractionalGlLearningBatch { inner })
            .map_err(js_error)
    }

    pub fn forward_history_log_gain(
        &self,
        input: &[f32],
        shape: &[u32],
        axis: Number,
        alpha: Number,
        log_gain: Number,
    ) -> Result<FractionalGlGainLearningBatch, JsValue> {
        self.inner
            .forward_history_log_gain(
                input,
                &shape.iter().map(|&v| v as usize).collect::<Vec<_>>(),
                js_u32(axis.as_ref(), "axis")? as usize,
                number(&alpha, "alpha")?,
                number(&log_gain, "log_gain")?,
            )
            .map(|inner| FractionalGlGainLearningBatch { inner })
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

    pub fn vjp_input(&self, upstream: &[f32]) -> Result<Vec<f32>, JsValue> {
        self.inner.vjp_input(upstream).map_err(js_error)
    }

    pub fn vjp_alpha(&self, upstream: &[f32]) -> Result<f32, JsValue> {
        self.inner.vjp_alpha(upstream).map_err(js_error)
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

#[wasm_bindgen]
impl FractionalGlGainLearningBatch {
    #[wasm_bindgen(getter)]
    pub fn output(&self) -> Vec<f32> {
        self.inner.output().iter().copied().collect()
    }
    #[wasm_bindgen(getter)]
    pub fn gain(&self) -> f32 {
        self.inner.gain()
    }
    pub fn vjp(&self, upstream: &[f32]) -> Result<FractionalGlGainGradients, JsValue> {
        self.inner
            .vjp(upstream)
            .map(|inner| FractionalGlGainGradients { inner })
            .map_err(js_error)
    }
    pub fn vjp_input(&self, upstream: &[f32]) -> Result<Vec<f32>, JsValue> {
        self.inner.vjp_input(upstream).map_err(js_error)
    }
    pub fn vjp_alpha(&self, upstream: &[f32]) -> Result<f32, JsValue> {
        self.inner.vjp_alpha(upstream).map_err(js_error)
    }
    pub fn vjp_log_gain(&self, upstream: &[f32]) -> Result<f32, JsValue> {
        self.inner.vjp_log_gain(upstream).map_err(js_error)
    }
    /// [alpha, log_gain], without an unrequested input-gradient allocation.
    pub fn vjp_parameters(&self, upstream: &[f32]) -> Result<Vec<f32>, JsValue> {
        self.inner
            .vjp_parameters(upstream)
            .map(|(a, g)| vec![a, g])
            .map_err(js_error)
    }
    pub fn jvp(
        &self,
        input_tangent: &[f32],
        alpha_tangent: Number,
        log_gain_tangent: Number,
    ) -> Result<Vec<f32>, JsValue> {
        self.inner
            .jvp(
                input_tangent,
                number(&alpha_tangent, "alpha_tangent")?,
                number(&log_gain_tangent, "log_gain_tangent")?,
            )
            .map_err(js_error)
    }
}

#[wasm_bindgen]
impl FractionalGlGainGradients {
    #[wasm_bindgen(getter)]
    pub fn input(&self) -> Vec<f32> {
        self.inner.input.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn alpha(&self) -> f32 {
        self.inner.alpha
    }
    #[wasm_bindgen(getter)]
    pub fn log_gain(&self) -> f32 {
        self.inner.log_gain
    }
}
