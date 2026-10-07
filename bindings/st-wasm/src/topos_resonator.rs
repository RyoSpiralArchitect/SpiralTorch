//! Browser transport for the same finite-unroll operator used by Rust and Python.

#[cfg(target_arch = "wasm32")]
use st_core::dynamics::topos_resonator::{
    ToposResonatorBackward, ToposResonatorConfig, ToposResonatorLearningBatch as CoreLearningBatch,
    ToposResonatorOperator,
};
#[cfg(target_arch = "wasm32")]
use st_tensor::topos::OpenCartesianTopos;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::{js_error, js_u32};
#[cfg(target_arch = "wasm32")]
use js_sys::Number;

#[cfg(target_arch = "wasm32")]
fn scalar(value: &Number, label: &str) -> Result<f32, JsValue> {
    let raw: &JsValue = value.as_ref();
    raw.as_f64()
        .map(|value| value as f32)
        .ok_or_else(|| js_error(format!("{label} must be a number")))
}

/// Explicit scalar f32 WASM execution, not a WebGPU resident operator.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct ToposResonatorKernel {
    operator: ToposResonatorOperator,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct ToposResonatorLearningBatch {
    inner: CoreLearningBatch,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct ToposResonatorPullback {
    inner: ToposResonatorBackward,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl ToposResonatorKernel {
    #[wasm_bindgen(constructor)]
    pub fn new(
        coupling: Number,
        iterations: Number,
        saturation: Number,
        porosity: Number,
        max_values: Number,
    ) -> Result<ToposResonatorKernel, JsValue> {
        let coupling = scalar(&coupling, "coupling")?;
        let iterations = js_u32(iterations.as_ref(), "iterations")? as usize;
        let saturation = scalar(&saturation, "saturation")?;
        let porosity = scalar(&porosity, "porosity")?;
        let max_values = js_u32(max_values.as_ref(), "max_values")? as usize;
        let config = ToposResonatorConfig::new(coupling, iterations).map_err(js_error)?;
        let topos = OpenCartesianTopos::new(-1.0, 1e-6, saturation, iterations + 1, max_values)
            .map_err(js_error)?
            .with_porosity(porosity)
            .map_err(js_error)?;
        Ok(Self {
            operator: ToposResonatorOperator::new(config, topos).map_err(js_error)?,
        })
    }

    #[wasm_bindgen(getter, js_name = executionBackend)]
    pub fn execution_backend(&self) -> String {
        "rust_f32_wasm".to_owned()
    }

    pub fn forward(
        &self,
        input: &[f32],
        gate: &[f32],
        rows: Number,
        features: Number,
    ) -> Result<Vec<f32>, JsValue> {
        let rows = js_u32(rows.as_ref(), "rows")? as usize;
        let features = js_u32(features.as_ref(), "features")? as usize;
        self.operator
            .forward(input, gate, rows, features)
            .map(|step| step.output)
            .map_err(js_error)
    }

    /// Returns per-element `grad_input` and `grad_gate`; callers reduce broadcasts.
    pub fn backward(
        &self,
        input: &[f32],
        gate: &[f32],
        grad_output: &[f32],
        rows: Number,
        features: Number,
    ) -> Result<JsValue, JsValue> {
        let rows = js_u32(rows.as_ref(), "rows")? as usize;
        let features = js_u32(features.as_ref(), "features")? as usize;
        let gradients = self
            .operator
            .backward(input, gate, grad_output, rows, features)
            .map_err(js_error)?;
        serde_wasm_bindgen::to_value(&gradients).map_err(js_error)
    }

    pub fn capture(
        &self,
        input: Vec<f32>,
        gate: Vec<f32>,
        rows: Number,
        features: Number,
    ) -> Result<ToposResonatorLearningBatch, JsValue> {
        let rows = js_u32(rows.as_ref(), "rows")? as usize;
        let features = js_u32(features.as_ref(), "features")? as usize;
        self.operator
            .capture_owned(input, gate, rows, features)
            .map(|inner| ToposResonatorLearningBatch { inner })
            .map_err(js_error)
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl ToposResonatorLearningBatch {
    #[wasm_bindgen(getter)]
    pub fn output(&self) -> Vec<f32> {
        self.inner.output().to_vec()
    }

    pub fn audit_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.inner.step().audit).map_err(js_error)
    }

    pub fn vjp(&self, upstream: &[f32]) -> Result<ToposResonatorPullback, JsValue> {
        self.inner
            .vjp(upstream)
            .map(|inner| ToposResonatorPullback { inner })
            .map_err(js_error)
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl ToposResonatorPullback {
    #[wasm_bindgen(getter)]
    pub fn grad_input(&self) -> Vec<f32> {
        self.inner.grad_input.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn grad_gate(&self) -> Vec<f32> {
        self.inner.grad_gate.clone()
    }
}
