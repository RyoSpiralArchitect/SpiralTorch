use serde_json::{json, Value};
use st_frac::fractal_field::FractalFieldGenerator;
use st_frac::mellin_types::{ComplexScalar, Scalar};

#[cfg(target_arch = "wasm32")]
use js_sys::Float32Array;
#[cfg(target_arch = "wasm32")]
use serde::Serialize;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::js_error;

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

fn normalised(value: f64) -> f64 {
    let value = value.abs();
    value / (1.0 + value)
}

fn finite_f64(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn support_value(log_start: Scalar, log_step: Scalar, len: usize) -> Value {
    let end = if len == 0 {
        log_start
    } else {
        log_start + log_step * (len.saturating_sub(1) as Scalar)
    };
    json!({
        "log_start": log_start,
        "log_step": log_step,
        "len": len,
        "support": [log_start, end],
    })
}

fn sample_abs(sample: ComplexScalar) -> f64 {
    f64::from(sample.norm())
}

fn sample_phase(sample: ComplexScalar) -> f64 {
    f64::from(sample.im.atan2(sample.re))
}

fn preview_value(
    field: &[ComplexScalar],
    log_start: Scalar,
    log_step: Scalar,
    preview_len: usize,
) -> Value {
    json!(field
        .iter()
        .take(preview_len)
        .enumerate()
        .map(|(index, sample)| {
            json!({
                "index": index,
                "log": log_start + log_step * index as Scalar,
                "re": sample.re,
                "im": sample.im,
                "abs": sample_abs(*sample),
                "phase": sample_phase(*sample),
            })
        })
        .collect::<Vec<_>>())
}

pub fn fractal_field_probe_from_generator_value(
    generator: &FractalFieldGenerator,
    log_start: Scalar,
    log_step: Scalar,
    len: usize,
    preview_len: usize,
) -> Result<Value, String> {
    let field = generator
        .branching_field(log_start, log_step, len)
        .map_err(|err| err.to_string())?;
    let mut energy = 0.0f64;
    let mut mean_abs = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut mean_real = 0.0f64;
    let mut mean_imag = 0.0f64;
    let mut total_variation = 0.0f64;

    for sample in &field {
        let abs = sample_abs(*sample);
        energy += abs * abs;
        mean_abs += abs;
        max_abs = max_abs.max(abs);
        mean_real += f64::from(sample.re);
        mean_imag += f64::from(sample.im);
    }
    for pair in field.windows(2) {
        total_variation += sample_abs(pair[1] - pair[0]);
    }

    let count = field.len() as f64;
    if count > 0.0 {
        energy /= count;
        mean_abs /= count;
        mean_real /= count;
        mean_imag /= count;
    }
    if count > 1.0 {
        total_variation /= count - 1.0;
    }

    let phase_drift = match (field.first(), field.last()) {
        (Some(first), Some(last)) if field.len() > 1 => sample_phase(*last) - sample_phase(*first),
        _ => 0.0,
    };

    Ok(json!({
        "kind": "spiraltorch.wasm_fractal_field_probe",
        "source_crate": "st-frac::fractal_field",
        "mode": "branching_field",
        "generator": {
            "octaves": generator.octaves(),
            "lacunarity": generator.lacunarity(),
            "gain": generator.gain(),
            "iterations": generator.iterations(),
        },
        "log_lattice": support_value(log_start, log_step, field.len()),
        "sample_count": field.len(),
        "preview_count": preview_len.min(field.len()),
        "energy": finite_f64(energy),
        "mean_abs": finite_f64(mean_abs),
        "max_abs": finite_f64(max_abs),
        "mean_real": finite_f64(mean_real),
        "mean_imag": finite_f64(mean_imag),
        "phase_drift": finite_f64(phase_drift),
        "total_variation": finite_f64(total_variation),
        "coherence_score": 1.0 - normalised(total_variation),
        "samples": preview_value(&field, log_start, log_step, preview_len),
    }))
}

pub fn fractal_field_probe_value(
    octaves: u32,
    lacunarity: f32,
    gain: f32,
    iterations: u32,
    log_start: Scalar,
    log_step: Scalar,
    len: usize,
    preview_len: usize,
) -> Result<Value, String> {
    let generator = FractalFieldGenerator::new(octaves, lacunarity, gain, iterations)
        .map_err(|err| err.to_string())?;
    fractal_field_probe_from_generator_value(&generator, log_start, log_step, len, preview_len)
}

#[cfg(target_arch = "wasm32")]
fn field_to_array(field: &[ComplexScalar]) -> Float32Array {
    let mut out = Vec::with_capacity(field.len() * 2);
    for sample in field {
        out.push(sample.re);
        out.push(sample.im);
    }
    Float32Array::from(out.as_slice())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmFractalFieldGenerator {
    inner: FractalFieldGenerator,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmFractalFieldGenerator {
    #[wasm_bindgen(constructor)]
    pub fn new(octaves: u32, lacunarity: f32, gain: f32, iterations: u32) -> Result<Self, JsValue> {
        let inner =
            FractalFieldGenerator::new(octaves, lacunarity, gain, iterations).map_err(js_error)?;
        Ok(Self { inner })
    }

    #[wasm_bindgen(getter)]
    pub fn octaves(&self) -> u32 {
        self.inner.octaves()
    }

    #[wasm_bindgen(getter)]
    pub fn lacunarity(&self) -> f32 {
        self.inner.lacunarity()
    }

    #[wasm_bindgen(getter)]
    pub fn gain(&self) -> f32 {
        self.inner.gain()
    }

    #[wasm_bindgen(getter)]
    pub fn iterations(&self) -> u32 {
        self.inner.iterations()
    }

    #[wasm_bindgen(js_name = branchingField)]
    pub fn branching_field(
        &self,
        log_start: f32,
        log_step: f32,
        len: usize,
    ) -> Result<Float32Array, JsValue> {
        let field = self
            .inner
            .branching_field(log_start, log_step, len)
            .map_err(js_error)?;
        Ok(field_to_array(&field))
    }

    #[wasm_bindgen(js_name = probeObject)]
    pub fn probe_object(
        &self,
        log_start: f32,
        log_step: f32,
        len: usize,
        preview_len: usize,
    ) -> Result<JsValue, JsValue> {
        let value = fractal_field_probe_from_generator_value(
            &self.inner,
            log_start,
            log_step,
            len,
            preview_len,
        )
        .map_err(js_error)?;
        to_json_compatible_js(&value)
    }

    #[wasm_bindgen(js_name = probeJson)]
    pub fn probe_json(
        &self,
        log_start: f32,
        log_step: f32,
        len: usize,
        preview_len: usize,
    ) -> Result<String, JsValue> {
        fractal_field_probe_from_generator_value(&self.inner, log_start, log_step, len, preview_len)
            .and_then(|value| serde_json::to_string(&value).map_err(|err| err.to_string()))
            .map_err(js_error)
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = fractalFieldProbeObject)]
pub fn fractal_field_probe_object(
    octaves: u32,
    lacunarity: f32,
    gain: f32,
    iterations: u32,
    log_start: f32,
    log_step: f32,
    len: usize,
    preview_len: usize,
) -> Result<JsValue, JsValue> {
    let value = fractal_field_probe_value(
        octaves,
        lacunarity,
        gain,
        iterations,
        log_start,
        log_step,
        len,
        preview_len,
    )
    .map_err(js_error)?;
    to_json_compatible_js(&value)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = fractalFieldProbeJson)]
pub fn fractal_field_probe_json(
    octaves: u32,
    lacunarity: f32,
    gain: f32,
    iterations: u32,
    log_start: f32,
    log_step: f32,
    len: usize,
    preview_len: usize,
) -> Result<String, JsValue> {
    fractal_field_probe_value(
        octaves,
        lacunarity,
        gain,
        iterations,
        log_start,
        log_step,
        len,
        preview_len,
    )
    .and_then(|value| serde_json::to_string(&value).map_err(|err| err.to_string()))
    .map_err(js_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractal_probe_reports_field_statistics() {
        let probe = fractal_field_probe_value(3, 2.0, 0.5, 16, -2.0, 0.25, 12, 4)
            .expect("fractal field probe");

        assert_eq!(probe["kind"], "spiraltorch.wasm_fractal_field_probe");
        assert_eq!(probe["source_crate"], "st-frac::fractal_field");
        assert_eq!(probe["sample_count"], 12);
        assert_eq!(probe["preview_count"], 4);
        assert!(probe["energy"].as_f64().unwrap() > 0.0);
        assert!(probe["max_abs"].as_f64().unwrap() > 0.0);
        assert_eq!(probe["samples"].as_array().unwrap().len(), 4);
    }
}
