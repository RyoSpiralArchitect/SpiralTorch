use js_sys::{Float32Array, Uint32Array};
use st_frac::mellin::{MellinEvalPlan, MellinLogGrid};
use st_frac::mellin_types::{ComplexScalar, Scalar};
use wasm_bindgen::prelude::*;

use crate::utils::js_error;

fn float32array_to_vec(buffer: &Float32Array) -> Vec<f32> {
    let len = buffer.length() as usize;
    let mut host = vec![0.0f32; len];
    buffer.copy_to(&mut host);
    host
}

fn interleaved_to_complex(buffer: &Float32Array) -> Result<Vec<ComplexScalar>, JsValue> {
    let host = float32array_to_vec(buffer);
    if host.len() % 2 != 0 {
        return Err(js_error("expected interleaved complex Float32Array (re, im, ...)"));
    }
    Ok(host
        .chunks_exact(2)
        .map(|pair| ComplexScalar::new(pair[0], pair[1]))
        .collect())
}

fn complex_to_interleaved(values: &[ComplexScalar]) -> Vec<f32> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for value in values {
        out.push(value.re);
        out.push(value.im);
    }
    out
}

fn scalar_array(values: &[Scalar]) -> Float32Array {
    Float32Array::from(values)
}

fn complex_array(values: &[ComplexScalar]) -> Float32Array {
    let host = complex_to_interleaved(values);
    Float32Array::from(host.as_slice())
}

#[wasm_bindgen]
pub struct WasmMellinEvalPlan {
    plan: MellinEvalPlan,
}

#[wasm_bindgen]
impl WasmMellinEvalPlan {
    #[wasm_bindgen(js_name = many)]
    pub fn many(log_start: Scalar, log_step: Scalar, s_values: &Float32Array) -> Result<Self, JsValue> {
        let s_values = interleaved_to_complex(s_values)?;
        let plan = MellinEvalPlan::many(log_start, log_step, &s_values).map_err(js_error)?;
        Ok(Self { plan })
    }

    #[wasm_bindgen(js_name = verticalLine)]
    pub fn vertical_line(
        log_start: Scalar,
        log_step: Scalar,
        real: Scalar,
        imag_values: &Float32Array,
    ) -> Result<Self, JsValue> {
        let imag_values = float32array_to_vec(imag_values);
        let plan = MellinEvalPlan::vertical_line(log_start, log_step, real, &imag_values).map_err(js_error)?;
        Ok(Self { plan })
    }

    #[wasm_bindgen(js_name = mesh)]
    pub fn mesh(
        log_start: Scalar,
        log_step: Scalar,
        real_values: &Float32Array,
        imag_values: &Float32Array,
    ) -> Result<Self, JsValue> {
        let real_values = float32array_to_vec(real_values);
        let imag_values = float32array_to_vec(imag_values);
        let plan = MellinEvalPlan::mesh(log_start, log_step, &real_values, &imag_values).map_err(js_error)?;
        Ok(Self { plan })
    }

    #[wasm_bindgen(getter, js_name = logStart)]
    pub fn log_start(&self) -> Scalar {
        self.plan.log_start()
    }

    #[wasm_bindgen(getter, js_name = logStep)]
    pub fn log_step(&self) -> Scalar {
        self.plan.log_step()
    }

    pub fn len(&self) -> usize {
        self.plan.len()
    }

    pub fn shape(&self) -> Uint32Array {
        let (rows, cols) = self.plan.shape();
        Uint32Array::from(&[rows as u32, cols as u32][..])
    }
}

#[wasm_bindgen]
pub struct WasmMellinLogGrid {
    grid: MellinLogGrid,
}

#[wasm_bindgen]
impl WasmMellinLogGrid {
    #[wasm_bindgen(constructor)]
    pub fn new(log_start: Scalar, log_step: Scalar, samples: &Float32Array) -> Result<Self, JsValue> {
        let samples = interleaved_to_complex(samples)?;
        let grid = MellinLogGrid::new(log_start, log_step, samples).map_err(js_error)?;
        Ok(Self { grid })
    }

    pub fn len(&self) -> usize {
        self.grid.len()
    }

    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.grid.is_empty()
    }

    #[wasm_bindgen(getter, js_name = logStart)]
    pub fn log_start(&self) -> Scalar {
        self.grid.log_start()
    }

    #[wasm_bindgen(getter, js_name = logStep)]
    pub fn log_step(&self) -> Scalar {
        self.grid.log_step()
    }

    pub fn samples(&self) -> Float32Array {
        complex_array(self.grid.samples())
    }

    pub fn weights(&self) -> Float32Array {
        scalar_array(self.grid.weights())
    }

    pub fn support(&self) -> Float32Array {
        let (start, end) = self.grid.support();
        scalar_array(&[start, end])
    }

    #[wasm_bindgen(js_name = weightedSeries)]
    pub fn weighted_series(&self) -> Result<Float32Array, JsValue> {
        let series = self.grid.weighted_series().map_err(js_error)?;
        Ok(complex_array(&series))
    }

    #[wasm_bindgen(js_name = planMany)]
    pub fn plan_many(&self, s_values: &Float32Array) -> Result<WasmMellinEvalPlan, JsValue> {
        let s_values = interleaved_to_complex(s_values)?;
        let plan = MellinEvalPlan::many(self.grid.log_start(), self.grid.log_step(), &s_values).map_err(js_error)?;
        Ok(WasmMellinEvalPlan { plan })
    }

    #[wasm_bindgen(js_name = planVerticalLine)]
    pub fn plan_vertical_line(
        &self,
        real: Scalar,
        imag_values: &Float32Array,
    ) -> Result<WasmMellinEvalPlan, JsValue> {
        let imag_values = float32array_to_vec(imag_values);
        let plan = self.grid.plan_vertical_line(real, &imag_values).map_err(js_error)?;
        Ok(WasmMellinEvalPlan { plan })
    }

    #[wasm_bindgen(js_name = planMesh)]
    pub fn plan_mesh(
        &self,
        real_values: &Float32Array,
        imag_values: &Float32Array,
    ) -> Result<WasmMellinEvalPlan, JsValue> {
        let real_values = float32array_to_vec(real_values);
        let imag_values = float32array_to_vec(imag_values);
        let plan = self.grid.plan_mesh(&real_values, &imag_values).map_err(js_error)?;
        Ok(WasmMellinEvalPlan { plan })
    }

    #[wasm_bindgen(js_name = evaluatePlan)]
    pub fn evaluate_plan(&self, plan: &WasmMellinEvalPlan) -> Result<Float32Array, JsValue> {
        let values = self.grid.evaluate_plan(&plan.plan).map_err(js_error)?;
        Ok(complex_array(&values))
    }

    #[wasm_bindgen(js_name = evaluatePlanMagnitude)]
    pub fn evaluate_plan_magnitude(&self, plan: &WasmMellinEvalPlan) -> Result<Float32Array, JsValue> {
        let values = self.grid.evaluate_plan_magnitude(&plan.plan).map_err(js_error)?;
        Ok(scalar_array(&values))
    }

    #[wasm_bindgen(js_name = evaluatePlanLogMagnitude)]
    pub fn evaluate_plan_log_magnitude(
        &self,
        plan: &WasmMellinEvalPlan,
        epsilon: Scalar,
    ) -> Result<Float32Array, JsValue> {
        let values = self
            .grid
            .evaluate_plan_log_magnitude(&plan.plan, epsilon)
            .map_err(js_error)?;
        Ok(scalar_array(&values))
    }

    #[wasm_bindgen(js_name = trainStepMatchGridPlan)]
    pub fn train_step_match_grid_plan(
        &mut self,
        plan: &WasmMellinEvalPlan,
        target: &WasmMellinLogGrid,
        lr: Scalar,
    ) -> Result<Scalar, JsValue> {
        self.grid
            .train_step_l2_plan_match_grid(&plan.plan, &target.grid, lr)
            .map_err(js_error)
    }

    pub fn evaluate(&self, s: &Float32Array) -> Result<Float32Array, JsValue> {
        let host = float32array_to_vec(s);
        if host.len() != 2 {
            return Err(js_error("expected complex point [re, im]"));
        }
        let s = ComplexScalar::new(host[0], host[1]);
        let value = self.grid.evaluate(s).map_err(js_error)?;
        Ok(scalar_array(&[value.re, value.im]))
    }

    #[wasm_bindgen(js_name = evaluateMany)]
    pub fn evaluate_many(&self, s_values: &Float32Array) -> Result<Float32Array, JsValue> {
        let s_values = interleaved_to_complex(s_values)?;
        let values = self.grid.evaluate_many(&s_values).map_err(js_error)?;
        Ok(complex_array(&values))
    }

    #[wasm_bindgen(js_name = evaluateVerticalLine)]
    pub fn evaluate_vertical_line(
        &self,
        real: Scalar,
        imag_values: &Float32Array,
    ) -> Result<Float32Array, JsValue> {
        let imag_values = float32array_to_vec(imag_values);
        let values = self
            .grid
            .evaluate_vertical_line(real, &imag_values)
            .map_err(js_error)?;
        Ok(complex_array(&values))
    }

    #[wasm_bindgen(js_name = evaluateMesh)]
    pub fn evaluate_mesh(
        &self,
        real_values: &Float32Array,
        imag_values: &Float32Array,
    ) -> Result<Float32Array, JsValue> {
        let real_values = float32array_to_vec(real_values);
        let imag_values = float32array_to_vec(imag_values);
        let values = self
            .grid
            .evaluate_mesh(&real_values, &imag_values)
            .map_err(js_error)?;
        Ok(complex_array(&values))
    }

    #[wasm_bindgen(js_name = evaluateMeshMagnitude)]
    pub fn evaluate_mesh_magnitude(
        &self,
        real_values: &Float32Array,
        imag_values: &Float32Array,
    ) -> Result<Float32Array, JsValue> {
        let real_values = float32array_to_vec(real_values);
        let imag_values = float32array_to_vec(imag_values);
        let values = self
            .grid
            .evaluate_mesh_magnitude(&real_values, &imag_values)
            .map_err(js_error)?;
        Ok(scalar_array(&values))
    }

    #[wasm_bindgen(js_name = evaluateMeshLogMagnitude)]
    pub fn evaluate_mesh_log_magnitude(
        &self,
        real_values: &Float32Array,
        imag_values: &Float32Array,
        epsilon: Scalar,
    ) -> Result<Float32Array, JsValue> {
        let real_values = float32array_to_vec(real_values);
        let imag_values = float32array_to_vec(imag_values);
        let values = self
            .grid
            .evaluate_mesh_log_magnitude(&real_values, &imag_values, epsilon)
            .map_err(js_error)?;
        Ok(scalar_array(&values))
    }

    #[wasm_bindgen(js_name = hilbertInnerProduct)]
    pub fn hilbert_inner_product(&self, other: &WasmMellinLogGrid) -> Result<Float32Array, JsValue> {
        let ip = self
            .grid
            .hilbert_inner_product(&other.grid)
            .map_err(js_error)?;
        Ok(scalar_array(&[ip.re, ip.im]))
    }

    #[wasm_bindgen(js_name = hilbertNorm)]
    pub fn hilbert_norm(&self) -> Result<Scalar, JsValue> {
        self.grid.hilbert_norm().map_err(js_error)
    }
}

#[wasm_bindgen(js_name = mellin_exp_decay_samples)]
pub fn mellin_exp_decay_samples(
    log_start: Scalar,
    log_step: Scalar,
    len: usize,
) -> Result<Float32Array, JsValue> {
    let samples = st_frac::mellin::sample_log_uniform(log_start, log_step, len, |x| {
        ComplexScalar::new((-x).exp(), 0.0)
    })
    .map_err(js_error)?;
    Ok(complex_array(&samples))
}

#[wasm_bindgen(js_name = mellin_exp_decay_samples_scaled)]
pub fn mellin_exp_decay_samples_scaled(
    log_start: Scalar,
    log_step: Scalar,
    len: usize,
    rate: Scalar,
) -> Result<Float32Array, JsValue> {
    let samples = st_frac::mellin::sample_log_uniform_exp_decay_scaled(log_start, log_step, len, rate)
        .map_err(js_error)?;
    Ok(complex_array(&samples))
}
