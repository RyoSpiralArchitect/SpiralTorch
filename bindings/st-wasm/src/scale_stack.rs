use ndarray::{Array2, ArrayD, IxDyn};
use serde_json::{json, Value};
use st_frac::scale_stack::{InterfaceMode, ScaleSample, ScaleStack, SemanticMetric};

#[cfg(target_arch = "wasm32")]
use js_sys::{Float32Array, Uint32Array};
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

fn parse_metric(metric: &str) -> Result<SemanticMetric, String> {
    match metric.to_ascii_lowercase().as_str() {
        "euclidean" => Ok(SemanticMetric::Euclidean),
        "cosine" => Ok(SemanticMetric::Cosine),
        other => Err(format!(
            "unknown semantic metric '{other}', expected 'euclidean' or 'cosine'"
        )),
    }
}

fn mode_label(mode: &InterfaceMode) -> &'static str {
    match mode {
        InterfaceMode::Scalar => "scalar",
        InterfaceMode::Semantic { metric, .. } => match metric {
            SemanticMetric::Euclidean => "semantic::euclidean",
            SemanticMetric::Cosine => "semantic::cosine",
        },
    }
}

fn samples_value(samples: &[ScaleSample]) -> Value {
    json!(samples
        .iter()
        .map(|sample| json!({
            "scale": sample.scale,
            "gate_mean": sample.gate_mean,
        }))
        .collect::<Vec<_>>())
}

fn persistence_value(stack: &ScaleStack) -> Value {
    json!(stack
        .persistence_measure()
        .into_iter()
        .map(|bin| json!({
            "scale_low": bin.scale_low,
            "scale_high": bin.scale_high,
            "mass": bin.mass,
        }))
        .collect::<Vec<_>>())
}

pub fn scale_stack_probe_value(
    stack: &ScaleStack,
    ambient_dim: f64,
    dimension_window: usize,
    levels: &[f64],
) -> Value {
    let coherence_profile = levels
        .iter()
        .map(|level| {
            json!({
                "level": level,
                "scale": stack.coherence_break_scale(*level),
            })
        })
        .collect::<Vec<_>>();

    json!({
        "kind": "spiraltorch.wasm_scale_stack_probe",
        "source_crate": "st-frac::scale_stack",
        "mode": mode_label(stack.mode()),
        "threshold": stack.threshold(),
        "sample_count": stack.samples().len(),
        "samples": samples_value(stack.samples()),
        "persistence": persistence_value(stack),
        "interface_density": stack.interface_density(),
        "moment_0": stack.moment(0),
        "moment_1": stack.moment(1),
        "moment_2": stack.moment(2),
        "boundary_dimension": stack.estimate_boundary_dimension(ambient_dim, dimension_window),
        "coherence_profile": coherence_profile,
    })
}

pub fn scalar_scale_stack_probe_value(
    field: &[f32],
    shape: &[usize],
    scales: &[f64],
    threshold: f32,
    ambient_dim: f64,
    dimension_window: usize,
    levels: &[f64],
) -> Result<Value, String> {
    let array = ArrayD::from_shape_vec(IxDyn(shape), field.to_vec())
        .map_err(|_| "field shape does not match provided dimensions".to_string())?;
    let stack = ScaleStack::from_scalar_field(array.view(), scales, threshold)
        .map_err(|err| err.to_string())?;
    Ok(scale_stack_probe_value(
        &stack,
        ambient_dim,
        dimension_window,
        levels,
    ))
}

pub fn semantic_scale_stack_probe_value(
    embeddings: &[f32],
    rows: usize,
    dims: usize,
    scales: &[f64],
    threshold: f32,
    metric: &str,
    ambient_dim: f64,
    dimension_window: usize,
    levels: &[f64],
) -> Result<Value, String> {
    let metric = parse_metric(metric)?;
    let array = Array2::from_shape_vec((rows, dims), embeddings.to_vec())
        .map_err(|_| "embedding length does not match rows * dims".to_string())?;
    let stack =
        ScaleStack::from_semantic_field(array.view().into_dyn(), scales, threshold, 1, metric)
            .map_err(|err| err.to_string())?;
    Ok(scale_stack_probe_value(
        &stack,
        ambient_dim,
        dimension_window,
        levels,
    ))
}

#[cfg(target_arch = "wasm32")]
fn float32array_to_vec(buffer: &Float32Array) -> Vec<f32> {
    let len = buffer.length() as usize;
    let mut host = vec![0.0f32; len];
    buffer.copy_to(&mut host);
    host
}

#[cfg(target_arch = "wasm32")]
fn float32array_to_f64_vec(buffer: &Float32Array) -> Vec<f64> {
    float32array_to_vec(buffer)
        .into_iter()
        .map(f64::from)
        .collect()
}

#[cfg(target_arch = "wasm32")]
fn uint32array_to_usize_vec(buffer: &Uint32Array) -> Vec<usize> {
    let len = buffer.length() as usize;
    let mut host = vec![0u32; len];
    buffer.copy_to(&mut host);
    host.into_iter().map(|value| value as usize).collect()
}

#[cfg(target_arch = "wasm32")]
fn samples_array(stack: &ScaleStack) -> Float32Array {
    let mut out = Vec::with_capacity(stack.samples().len() * 2);
    for sample in stack.samples() {
        out.push(sample.scale as f32);
        out.push(sample.gate_mean as f32);
    }
    Float32Array::from(out.as_slice())
}

#[cfg(target_arch = "wasm32")]
fn persistence_array(stack: &ScaleStack) -> Float32Array {
    let bins = stack.persistence_measure();
    let mut out = Vec::with_capacity(bins.len() * 3);
    for bin in bins {
        out.push(bin.scale_low as f32);
        out.push(bin.scale_high as f32);
        out.push(bin.mass as f32);
    }
    Float32Array::from(out.as_slice())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmScaleStack {
    inner: ScaleStack,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmScaleStack {
    #[wasm_bindgen(js_name = scalar)]
    pub fn scalar(
        field: &Float32Array,
        shape: &Uint32Array,
        scales: &Float32Array,
        threshold: f32,
    ) -> Result<Self, JsValue> {
        let field = float32array_to_vec(field);
        let shape = uint32array_to_usize_vec(shape);
        let scales = float32array_to_f64_vec(scales);
        let array = ArrayD::from_shape_vec(IxDyn(&shape), field)
            .map_err(|_| js_error("field shape does not match provided dimensions"))?;
        let inner =
            ScaleStack::from_scalar_field(array.view(), &scales, threshold).map_err(js_error)?;
        Ok(Self { inner })
    }

    #[wasm_bindgen(js_name = semantic)]
    pub fn semantic(
        embeddings: &Float32Array,
        rows: usize,
        dims: usize,
        scales: &Float32Array,
        threshold: f32,
        metric: &str,
    ) -> Result<Self, JsValue> {
        let embeddings = float32array_to_vec(embeddings);
        let scales = float32array_to_f64_vec(scales);
        let metric = parse_metric(metric).map_err(js_error)?;
        let array = Array2::from_shape_vec((rows, dims), embeddings)
            .map_err(|_| js_error("embedding length does not match rows * dims"))?;
        let inner =
            ScaleStack::from_semantic_field(array.view().into_dyn(), &scales, threshold, 1, metric)
                .map_err(js_error)?;
        Ok(Self { inner })
    }

    #[wasm_bindgen(getter)]
    pub fn threshold(&self) -> f32 {
        self.inner.threshold()
    }

    #[wasm_bindgen(getter)]
    pub fn mode(&self) -> String {
        mode_label(self.inner.mode()).to_string()
    }

    #[wasm_bindgen(getter, js_name = sampleCount)]
    pub fn sample_count(&self) -> usize {
        self.inner.samples().len()
    }

    pub fn samples(&self) -> Float32Array {
        samples_array(&self.inner)
    }

    pub fn persistence(&self) -> Float32Array {
        persistence_array(&self.inner)
    }

    #[wasm_bindgen(js_name = interfaceDensity)]
    pub fn interface_density(&self) -> Option<f64> {
        self.inner.interface_density()
    }

    pub fn moment(&self, order: u32) -> f64 {
        self.inner.moment(order)
    }

    #[wasm_bindgen(js_name = boundaryDimension)]
    pub fn boundary_dimension(&self, ambient_dim: f64, window: usize) -> Option<f64> {
        self.inner.estimate_boundary_dimension(ambient_dim, window)
    }

    #[wasm_bindgen(js_name = coherenceBreakScale)]
    pub fn coherence_break_scale(&self, level: f64) -> Option<f64> {
        self.inner.coherence_break_scale(level)
    }

    #[wasm_bindgen(js_name = coherenceProfile)]
    pub fn coherence_profile(&self, levels: &Float32Array) -> Float32Array {
        let levels = float32array_to_f64_vec(levels);
        let profile = self
            .inner
            .coherence_profile(&levels)
            .into_iter()
            .map(|value| value.unwrap_or(f64::NAN) as f32)
            .collect::<Vec<_>>();
        Float32Array::from(profile.as_slice())
    }

    #[wasm_bindgen(js_name = toObject)]
    pub fn to_object(
        &self,
        ambient_dim: f64,
        dimension_window: usize,
        levels: &Float32Array,
    ) -> Result<JsValue, JsValue> {
        let levels = float32array_to_f64_vec(levels);
        to_json_compatible_js(&scale_stack_probe_value(
            &self.inner,
            ambient_dim,
            dimension_window,
            &levels,
        ))
    }

    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(
        &self,
        ambient_dim: f64,
        dimension_window: usize,
        levels: &Float32Array,
    ) -> Result<String, JsValue> {
        let levels = float32array_to_f64_vec(levels);
        serde_json::to_string(&scale_stack_probe_value(
            &self.inner,
            ambient_dim,
            dimension_window,
            &levels,
        ))
        .map_err(js_error)
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = scalarScaleStackProbeObject)]
pub fn scalar_scale_stack_probe_object(
    field: &Float32Array,
    shape: &Uint32Array,
    scales: &Float32Array,
    threshold: f32,
    ambient_dim: f64,
    dimension_window: usize,
    levels: &Float32Array,
) -> Result<JsValue, JsValue> {
    let field = float32array_to_vec(field);
    let shape = uint32array_to_usize_vec(shape);
    let scales = float32array_to_f64_vec(scales);
    let levels = float32array_to_f64_vec(levels);
    let value = scalar_scale_stack_probe_value(
        &field,
        &shape,
        &scales,
        threshold,
        ambient_dim,
        dimension_window,
        &levels,
    )
    .map_err(js_error)?;
    to_json_compatible_js(&value)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = scalarScaleStackProbeJson)]
pub fn scalar_scale_stack_probe_json(
    field: &Float32Array,
    shape: &Uint32Array,
    scales: &Float32Array,
    threshold: f32,
    ambient_dim: f64,
    dimension_window: usize,
    levels: &Float32Array,
) -> Result<String, JsValue> {
    let field = float32array_to_vec(field);
    let shape = uint32array_to_usize_vec(shape);
    let scales = float32array_to_f64_vec(scales);
    let levels = float32array_to_f64_vec(levels);
    scalar_scale_stack_probe_value(
        &field,
        &shape,
        &scales,
        threshold,
        ambient_dim,
        dimension_window,
        &levels,
    )
    .and_then(|value| serde_json::to_string(&value).map_err(|err| err.to_string()))
    .map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = semanticScaleStackProbeObject)]
pub fn semantic_scale_stack_probe_object(
    embeddings: &Float32Array,
    rows: usize,
    dims: usize,
    scales: &Float32Array,
    threshold: f32,
    metric: &str,
    ambient_dim: f64,
    dimension_window: usize,
    levels: &Float32Array,
) -> Result<JsValue, JsValue> {
    let embeddings = float32array_to_vec(embeddings);
    let scales = float32array_to_f64_vec(scales);
    let levels = float32array_to_f64_vec(levels);
    let value = semantic_scale_stack_probe_value(
        &embeddings,
        rows,
        dims,
        &scales,
        threshold,
        metric,
        ambient_dim,
        dimension_window,
        &levels,
    )
    .map_err(js_error)?;
    to_json_compatible_js(&value)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = semanticScaleStackProbeJson)]
pub fn semantic_scale_stack_probe_json(
    embeddings: &Float32Array,
    rows: usize,
    dims: usize,
    scales: &Float32Array,
    threshold: f32,
    metric: &str,
    ambient_dim: f64,
    dimension_window: usize,
    levels: &Float32Array,
) -> Result<String, JsValue> {
    let embeddings = float32array_to_vec(embeddings);
    let scales = float32array_to_f64_vec(scales);
    let levels = float32array_to_f64_vec(levels);
    semantic_scale_stack_probe_value(
        &embeddings,
        rows,
        dims,
        &scales,
        threshold,
        metric,
        ambient_dim,
        dimension_window,
        &levels,
    )
    .and_then(|value| serde_json::to_string(&value).map_err(|err| err.to_string()))
    .map_err(js_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_probe_reports_persistence() {
        let probe = scalar_scale_stack_probe_value(
            &[
                0.0, 0.0, 1.0, 1.0, //
                0.0, 0.0, 1.0, 1.0, //
                0.0, 0.0, 1.0, 1.0, //
                0.0, 0.0, 1.0, 1.0,
            ],
            &[4, 4],
            &[1.0, 2.0, 3.0],
            0.01,
            2.0,
            3,
            &[0.25, 0.5],
        )
        .expect("scalar scale-stack probe");

        assert_eq!(probe["kind"], "spiraltorch.wasm_scale_stack_probe");
        assert_eq!(probe["source_crate"], "st-frac::scale_stack");
        assert_eq!(probe["mode"], "scalar");
        assert_eq!(probe["sample_count"], 3);
        assert!(probe["moment_0"].as_f64().unwrap() > 0.0);
        assert!(probe["persistence"].as_array().unwrap().len() > 0);
    }

    #[test]
    fn semantic_probe_uses_metric_label() {
        let probe = semantic_scale_stack_probe_value(
            &[
                1.0, 0.0, //
                0.9, 0.1, //
                0.0, 1.0,
            ],
            3,
            2,
            &[1.0, 2.0],
            0.2,
            "cosine",
            1.0,
            2,
            &[0.2],
        )
        .expect("semantic scale-stack probe");

        assert_eq!(probe["mode"], "semantic::cosine");
        assert_eq!(probe["sample_count"], 2);
        assert_eq!(
            probe["coherence_profile"].as_array().unwrap()[0]["level"],
            0.2
        );
    }
}
