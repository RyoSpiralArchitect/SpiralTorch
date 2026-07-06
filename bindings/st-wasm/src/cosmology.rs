use serde_json::{json, Value};
use st_frac::cosmology::{LogZSeries, SeriesOptions, WeightNormalisation, WindowFunction};
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

fn parse_window(window: &str) -> Result<WindowFunction, String> {
    match window.to_ascii_lowercase().as_str() {
        "rect" | "rectangular" | "none" => Ok(WindowFunction::Rectangular),
        "hann" => Ok(WindowFunction::Hann),
        other => Err(format!(
            "unknown window '{other}', expected 'rectangular' or 'hann'"
        )),
    }
}

fn parse_normalisation(normalisation: &str) -> Result<WeightNormalisation, String> {
    match normalisation.to_ascii_lowercase().as_str() {
        "none" => Ok(WeightNormalisation::None),
        "l1" => Ok(WeightNormalisation::L1),
        "l2" => Ok(WeightNormalisation::L2),
        other => Err(format!(
            "unknown normalisation '{other}', expected 'none', 'l1', or 'l2'"
        )),
    }
}

fn window_name(window: WindowFunction) -> &'static str {
    match window {
        WindowFunction::Rectangular => "rectangular",
        WindowFunction::Hann => "hann",
    }
}

fn normalisation_name(normalisation: WeightNormalisation) -> &'static str {
    match normalisation {
        WeightNormalisation::None => "none",
        WeightNormalisation::L1 => "l1",
        WeightNormalisation::L2 => "l2",
    }
}

fn finite_f64(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn normalised(value: f64) -> f64 {
    let value = value.abs();
    value / (1.0 + value)
}

fn complex_abs(value: ComplexScalar) -> f64 {
    f64::from(value.norm())
}

fn complex_phase(value: ComplexScalar) -> f64 {
    f64::from(value.im.atan2(value.re))
}

fn scalar_stats_value(values: &[Scalar]) -> Value {
    if values.is_empty() {
        return json!({
            "count": 0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "energy": 0.0,
        });
    }

    let mut mean = 0.0f64;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut energy = 0.0f64;
    for value in values {
        let value = f64::from(*value);
        mean += value;
        min = min.min(value);
        max = max.max(value);
        energy += value * value;
    }
    let count = values.len() as f64;
    json!({
        "count": values.len(),
        "mean": finite_f64(mean / count),
        "min": finite_f64(min),
        "max": finite_f64(max),
        "energy": finite_f64(energy / count),
    })
}

fn projection_stats_value(values: &[ComplexScalar], preview_len: usize) -> Value {
    let mut mean_abs = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut energy = 0.0f64;
    for value in values {
        let abs = complex_abs(*value);
        mean_abs += abs;
        max_abs = max_abs.max(abs);
        energy += abs * abs;
    }
    let count = values.len() as f64;
    if count > 0.0 {
        mean_abs /= count;
        energy /= count;
    }
    let phase_drift = match (values.first(), values.last()) {
        (Some(first), Some(last)) if values.len() > 1 => {
            complex_phase(*last) - complex_phase(*first)
        }
        _ => 0.0,
    };
    let preview = values
        .iter()
        .take(preview_len)
        .enumerate()
        .map(|(index, value)| {
            json!({
                "index": index,
                "re": value.re,
                "im": value.im,
                "abs": complex_abs(*value),
                "phase": complex_phase(*value),
            })
        })
        .collect::<Vec<_>>();

    json!({
        "count": values.len(),
        "mean_abs": finite_f64(mean_abs),
        "max_abs": finite_f64(max_abs),
        "energy": finite_f64(energy),
        "phase_drift": finite_f64(phase_drift),
        "stability_score": 1.0 - normalised(max_abs - mean_abs),
        "preview_count": preview_len.min(values.len()),
        "preview": preview,
    })
}

fn lattice_value(series: &LogZSeries) -> Value {
    let end = if series.is_empty() {
        series.log_start()
    } else {
        series.log_start() + series.log_step() * (series.len().saturating_sub(1) as Scalar)
    };
    json!({
        "log_start": series.log_start(),
        "log_step": series.log_step(),
        "len": series.len(),
        "support": [series.log_start(), end],
    })
}

pub fn log_z_series_probe_from_series_value(
    series: &LogZSeries,
    z_values: &[ComplexScalar],
    preview_len: usize,
) -> Result<Value, String> {
    let projection = series
        .evaluate_many_z(z_values)
        .map_err(|err| err.to_string())?;
    let options = series.options();
    Ok(json!({
        "kind": "spiraltorch.wasm_log_z_series_probe",
        "source_crate": "st-frac::cosmology",
        "mode": "log_z_series",
        "log_lattice": lattice_value(series),
        "options": {
            "window": window_name(options.window),
            "normalisation": normalisation_name(options.normalisation),
        },
        "sample_count": series.len(),
        "sample_stats": scalar_stats_value(series.samples()),
        "weight_stats": scalar_stats_value(series.weights()),
        "z_count": z_values.len(),
        "projection": projection_stats_value(&projection, preview_len),
    }))
}

pub fn log_z_series_probe_value(
    log_start: Scalar,
    log_step: Scalar,
    samples: &[Scalar],
    window: &str,
    normalisation: &str,
    z_values: &[ComplexScalar],
    preview_len: usize,
) -> Result<Value, String> {
    let window = parse_window(window)?;
    let normalisation = parse_normalisation(normalisation)?;
    let series = LogZSeries::from_samples_with_options(
        log_start,
        log_step,
        samples.to_vec(),
        SeriesOptions {
            window,
            normalisation,
        },
    )
    .map_err(|err| err.to_string())?;
    log_z_series_probe_from_series_value(&series, z_values, preview_len)
}

#[cfg(target_arch = "wasm32")]
fn float32array_to_vec(buffer: &Float32Array) -> Vec<f32> {
    let len = buffer.length() as usize;
    let mut host = vec![0.0f32; len];
    buffer.copy_to(&mut host);
    host
}

#[cfg(target_arch = "wasm32")]
fn interleaved_to_complex(buffer: &Float32Array) -> Result<Vec<ComplexScalar>, JsValue> {
    let host = float32array_to_vec(buffer);
    if host.len() % 2 != 0 {
        return Err(js_error(
            "expected interleaved complex Float32Array (re, im, ...)",
        ));
    }
    Ok(host
        .chunks_exact(2)
        .map(|pair| ComplexScalar::new(pair[0], pair[1]))
        .collect())
}

#[cfg(target_arch = "wasm32")]
fn complex_point(buffer: &Float32Array) -> Result<ComplexScalar, JsValue> {
    let host = float32array_to_vec(buffer);
    if host.len() != 2 {
        return Err(js_error("expected complex point [re, im]"));
    }
    Ok(ComplexScalar::new(host[0], host[1]))
}

#[cfg(target_arch = "wasm32")]
fn complex_array(values: &[ComplexScalar]) -> Float32Array {
    let mut out = Vec::with_capacity(values.len() * 2);
    for value in values {
        out.push(value.re);
        out.push(value.im);
    }
    Float32Array::from(out.as_slice())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmLogZSeries {
    inner: LogZSeries,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmLogZSeries {
    #[wasm_bindgen(constructor)]
    pub fn new(
        log_start: f32,
        log_step: f32,
        samples: &Float32Array,
        window: &str,
        normalisation: &str,
    ) -> Result<Self, JsValue> {
        let window = parse_window(window).map_err(js_error)?;
        let normalisation = parse_normalisation(normalisation).map_err(js_error)?;
        let inner = LogZSeries::from_samples_with_options(
            log_start,
            log_step,
            float32array_to_vec(samples),
            SeriesOptions {
                window,
                normalisation,
            },
        )
        .map_err(js_error)?;
        Ok(Self { inner })
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[wasm_bindgen(getter, js_name = logStart)]
    pub fn log_start(&self) -> f32 {
        self.inner.log_start()
    }

    #[wasm_bindgen(getter, js_name = logStep)]
    pub fn log_step(&self) -> f32 {
        self.inner.log_step()
    }

    #[wasm_bindgen(getter)]
    pub fn window(&self) -> String {
        window_name(self.inner.options().window).to_string()
    }

    #[wasm_bindgen(getter)]
    pub fn normalisation(&self) -> String {
        normalisation_name(self.inner.options().normalisation).to_string()
    }

    pub fn samples(&self) -> Float32Array {
        Float32Array::from(self.inner.samples())
    }

    pub fn weights(&self) -> Float32Array {
        Float32Array::from(self.inner.weights())
    }

    #[wasm_bindgen(js_name = evaluateZ)]
    pub fn evaluate_z(&self, z: &Float32Array) -> Result<Float32Array, JsValue> {
        let z = complex_point(z)?;
        let value = self.inner.evaluate_z(z).map_err(js_error)?;
        Ok(complex_array(&[value]))
    }

    #[wasm_bindgen(js_name = evaluateManyZ)]
    pub fn evaluate_many_z(&self, z_values: &Float32Array) -> Result<Float32Array, JsValue> {
        let z_values = interleaved_to_complex(z_values)?;
        let values = self.inner.evaluate_many_z(&z_values).map_err(js_error)?;
        Ok(complex_array(&values))
    }

    #[wasm_bindgen(js_name = probeObject)]
    pub fn probe_object(
        &self,
        z_values: &Float32Array,
        preview_len: usize,
    ) -> Result<JsValue, JsValue> {
        let z_values = interleaved_to_complex(z_values)?;
        let value = log_z_series_probe_from_series_value(&self.inner, &z_values, preview_len)
            .map_err(js_error)?;
        to_json_compatible_js(&value)
    }

    #[wasm_bindgen(js_name = probeJson)]
    pub fn probe_json(
        &self,
        z_values: &Float32Array,
        preview_len: usize,
    ) -> Result<String, JsValue> {
        let z_values = interleaved_to_complex(z_values)?;
        log_z_series_probe_from_series_value(&self.inner, &z_values, preview_len)
            .and_then(|value| serde_json::to_string(&value).map_err(|err| err.to_string()))
            .map_err(js_error)
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = logZSeriesProbeObject)]
pub fn log_z_series_probe_object(
    log_start: f32,
    log_step: f32,
    samples: &Float32Array,
    window: &str,
    normalisation: &str,
    z_values: &Float32Array,
    preview_len: usize,
) -> Result<JsValue, JsValue> {
    let samples = float32array_to_vec(samples);
    let z_values = interleaved_to_complex(z_values)?;
    let value = log_z_series_probe_value(
        log_start,
        log_step,
        &samples,
        window,
        normalisation,
        &z_values,
        preview_len,
    )
    .map_err(js_error)?;
    to_json_compatible_js(&value)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = logZSeriesProbeJson)]
pub fn log_z_series_probe_json(
    log_start: f32,
    log_step: f32,
    samples: &Float32Array,
    window: &str,
    normalisation: &str,
    z_values: &Float32Array,
    preview_len: usize,
) -> Result<String, JsValue> {
    let samples = float32array_to_vec(samples);
    let z_values = interleaved_to_complex(z_values)?;
    log_z_series_probe_value(
        log_start,
        log_step,
        &samples,
        window,
        normalisation,
        &z_values,
        preview_len,
    )
    .and_then(|value| serde_json::to_string(&value).map_err(|err| err.to_string()))
    .map_err(js_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_z_probe_reports_projection_stats() {
        let probe = log_z_series_probe_value(
            0.0,
            0.25,
            &[1.0, 2.0, 3.0, 4.0],
            "hann",
            "l1",
            &[ComplexScalar::new(0.5, 0.0), ComplexScalar::new(0.2, 0.3)],
            2,
        )
        .expect("log-z series probe");

        assert_eq!(probe["kind"], "spiraltorch.wasm_log_z_series_probe");
        assert_eq!(probe["source_crate"], "st-frac::cosmology");
        assert_eq!(probe["sample_count"], 4);
        assert_eq!(probe["options"]["window"], "hann");
        assert_eq!(probe["options"]["normalisation"], "l1");
        assert_eq!(probe["projection"]["count"], 2);
        assert!(probe["projection"]["energy"].as_f64().unwrap() > 0.0);
        assert_eq!(probe["projection"]["preview"].as_array().unwrap().len(), 2);
    }
}
