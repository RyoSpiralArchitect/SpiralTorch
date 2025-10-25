use std::collections::HashMap;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use pyo3::{wrap_pyfunction, Bound, PyRefMut};

use crate::{psi_synchro::PyZPulse, tensor::tensor_err_to_py, tensor::PyTensor};
use st_core::theory::zpulse::ZScale;
use st_logic::contextual_observation::{
    Arrangement, LagrangianGate, LagrangianGateConfig, OrientationGauge, PureAtom,
};
use st_vision::contextual::{VisionContextualGate, VisionContextualPulse};

const MIN_SMOOTHING: f32 = 0.0;
const MAX_SMOOTHING: f32 = 0.999;

#[derive(Clone, Copy)]
enum TapeTarget {
    Hypergrad,
    Realgrad,
}

fn parse_gauge(value: &str) -> PyResult<OrientationGauge> {
    match value.to_ascii_lowercase().as_str() {
        "preserve" | "keep" | "stable" => Ok(OrientationGauge::Preserve),
        "swap" | "flip" | "invert" => Ok(OrientationGauge::Swap),
        other => Err(PyValueError::new_err(format!(
            "unknown gauge '{other}' (expected 'preserve' or 'swap')"
        ))),
    }
}

fn parse_atoms(placements: Vec<i64>) -> PyResult<Vec<PureAtom>> {
    placements
        .into_iter()
        .map(|value| match value {
            0 => Ok(PureAtom::A),
            1 => Ok(PureAtom::B),
            other => Err(PyValueError::new_err(format!(
                "placements must be 0 or 1, got {other}"
            ))),
        })
        .collect()
}

fn build_arrangement(
    placements: Vec<PureAtom>,
    edges: Option<Vec<(usize, usize)>>,
) -> PyResult<Arrangement> {
    if let Some(ref edge_list) = edges {
        for &(u, v) in edge_list {
            if u >= placements.len() || v >= placements.len() {
                return Err(PyValueError::new_err(
                    "edge indices must be within the placement range",
                ));
            }
            if u == v {
                return Err(PyValueError::new_err(
                    "edge endpoints must reference distinct indices",
                ));
            }
        }
    }
    Ok(match edges {
        Some(edge_list) => Arrangement::new(placements, edge_list),
        None => Arrangement::from_line(placements),
    })
}

#[pyclass(module = "spiraltorch.vision", name = "VisionPulseFrame")]
pub(crate) struct PyVisionPulseFrame {
    summary: String,
    highlights: Vec<String>,
    label: Option<String>,
    lexical_weight: f32,
    signature: Option<(usize, usize, isize)>,
    support: usize,
    dimensions: Option<(usize, usize)>,
    pulse: PyZPulse,
}

impl PyVisionPulseFrame {
    fn from_pulse(pulse: VisionContextualPulse) -> Self {
        let VisionContextualPulse {
            narrative,
            projection,
            pulse,
            dimensions,
        } = pulse;
        let label = projection.label.map(|label| label.as_str().to_string());
        let lexical_weight = projection.lexical_weight();
        let signature = projection.signature.as_ref().map(|signature| {
            (
                signature.boundary_edges,
                signature.absolute_population_imbalance,
                signature.cluster_imbalance,
            )
        });
        let support = projection.support;
        Self {
            summary: narrative.summary,
            highlights: narrative.highlights,
            label,
            lexical_weight,
            signature,
            support,
            dimensions,
            pulse: PyZPulse::from_pulse(pulse),
        }
    }
}

#[pymethods]
impl PyVisionPulseFrame {
    #[getter]
    pub fn summary(&self) -> &str {
        &self.summary
    }

    #[getter]
    pub fn highlights(&self) -> Vec<String> {
        self.highlights.clone()
    }

    #[getter]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    #[getter]
    pub fn lexical_weight(&self) -> f32 {
        self.lexical_weight
    }

    #[getter]
    pub fn signature(&self) -> Option<(usize, usize, i64)> {
        self.signature
            .map(|(boundary, population, cluster)| (boundary, population, cluster as i64))
    }

    #[getter]
    pub fn support(&self) -> usize {
        self.support
    }

    #[getter]
    pub fn dimensions(&self) -> Option<(usize, usize)> {
        self.dimensions
    }

    #[getter]
    pub fn pulse(&self) -> PyZPulse {
        self.pulse.clone()
    }
}

#[pyclass(
    module = "spiraltorch.vision",
    name = "ContextualVisionGate",
    unsendable
)]
pub(crate) struct PyVisionContextualGate {
    gate: VisionContextualGate,
    default_gauge: OrientationGauge,
    default_pivot: f32,
}

#[pymethods]
impl PyVisionContextualGate {
    #[new]
    #[pyo3(signature = (
        *,
        pivot=0.0,
        gauge="preserve",
        tempo_normaliser=None,
        energy_gain=1.0,
        drift_gain=1.0,
        bias_gain=1.0,
        support_gain=1.0,
        scale=None,
        quality_floor=0.0,
        stderr_gain=1.0
    ))]
    pub fn new(
        pivot: f32,
        gauge: &str,
        tempo_normaliser: Option<f32>,
        energy_gain: f32,
        drift_gain: f32,
        bias_gain: f32,
        support_gain: f32,
        scale: Option<(f32, f32)>,
        quality_floor: f32,
        stderr_gain: f32,
    ) -> PyResult<Self> {
        let default_gauge = parse_gauge(gauge)?;
        let mut config = LagrangianGateConfig::default()
            .tempo_normaliser(tempo_normaliser.unwrap_or(1.0))
            .energy_gain(energy_gain)
            .drift_gain(drift_gain)
            .bias_gain(bias_gain)
            .support_gain(support_gain)
            .quality_floor(quality_floor)
            .stderr_gain(stderr_gain);
        if let Some((physical, log)) = scale {
            let scale = ZScale::from_components(physical, log).ok_or_else(|| {
                PyValueError::new_err("scale must have positive finite radius components")
            })?;
            config = config.scale(Some(scale));
        }
        Ok(Self {
            gate: VisionContextualGate::new(LagrangianGate::new(config)),
            default_gauge,
            default_pivot: pivot,
        })
    }

    #[pyo3(signature = (tensor, *, gauge=None, ts=0, pivot=None))]
    pub fn project_tensor(
        &self,
        tensor: &PyTensor,
        gauge: Option<&str>,
        ts: u64,
        pivot: Option<f32>,
    ) -> PyResult<PyVisionPulseFrame> {
        let gauge = match gauge {
            Some(value) => parse_gauge(value)?,
            None => self.default_gauge,
        };
        let pivot = pivot.unwrap_or(self.default_pivot);
        let pulse = self
            .gate
            .gate_from_tensor(&tensor.inner, pivot, gauge, ts)
            .map_err(tensor_err_to_py)?;
        Ok(PyVisionPulseFrame::from_pulse(pulse))
    }

    #[pyo3(signature = (placements, edges=None, *, gauge=None, ts=0, dimensions=None))]
    pub fn project_arrangement(
        &self,
        placements: Vec<i64>,
        edges: Option<Vec<(usize, usize)>>,
        gauge: Option<&str>,
        ts: u64,
        dimensions: Option<(usize, usize)>,
    ) -> PyResult<PyVisionPulseFrame> {
        let atoms = parse_atoms(placements)?;
        let arrangement = build_arrangement(atoms, edges)?;
        let gauge = match gauge {
            Some(value) => parse_gauge(value)?,
            None => self.default_gauge,
        };
        let pulse = self
            .gate
            .gate_from_arrangement_with_dims(&arrangement, gauge, ts, dimensions)
            .map_err(tensor_err_to_py)?;
        Ok(PyVisionPulseFrame::from_pulse(pulse))
    }

    #[getter]
    pub fn gauge(&self) -> &'static str {
        match self.default_gauge {
            OrientationGauge::Preserve => "preserve",
            OrientationGauge::Swap => "swap",
        }
    }

    #[getter]
    pub fn pivot(&self) -> f32 {
        self.default_pivot
    }
}

#[pyclass(module = "spiraltorch", name = "CanvasSnapshot")]
pub(crate) struct PyCanvasSnapshot {
    canvas: Vec<Vec<f32>>,
    hypergrad: Vec<Vec<f32>>,
    realgrad: Vec<Vec<f32>>,
    summary: HashMap<String, HashMap<String, f32>>,
    patch: Option<Vec<Vec<f32>>>,
}

impl PyCanvasSnapshot {
    fn new(
        canvas: Vec<Vec<f32>>,
        hypergrad: Vec<Vec<f32>>,
        realgrad: Vec<Vec<f32>>,
        summary: HashMap<String, HashMap<String, f32>>,
        patch: Option<Vec<Vec<f32>>>,
    ) -> Self {
        Self {
            canvas,
            hypergrad,
            realgrad,
            summary,
            patch,
        }
    }
}

#[pymethods]
impl PyCanvasSnapshot {
    #[getter]
    fn canvas(&self) -> Vec<Vec<f32>> {
        self.canvas.clone()
    }

    #[getter]
    fn hypergrad(&self) -> Vec<Vec<f32>> {
        self.hypergrad.clone()
    }

    #[getter]
    fn realgrad(&self) -> Vec<Vec<f32>> {
        self.realgrad.clone()
    }

    #[getter]
    fn summary(&self) -> HashMap<String, HashMap<String, f32>> {
        self.summary.clone()
    }

    #[getter]
    fn patch(&self) -> Option<Vec<Vec<f32>>> {
        self.patch.clone()
    }

    #[setter]
    fn set_patch(&mut self, patch: Option<Vec<Vec<f32>>>) {
        self.patch = patch;
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "CanvasSnapshot(canvas_shape=({rows}, {cols}))",
            rows = self.canvas.len(),
            cols = self.canvas.first().map(|row| row.len()).unwrap_or(0)
        ))
    }
}

#[pyclass(module = "spiraltorch", name = "CanvasTransformer", unsendable)]
pub(crate) struct PyCanvasTransformer {
    width: usize,
    height: usize,
    smoothing: f32,
    canvas: Vec<f32>,
    hypergrad: Vec<f32>,
    realgrad: Vec<f32>,
}

#[pymethods]
impl PyCanvasTransformer {
    #[new]
    #[pyo3(signature = (width, height, *, smoothing=0.85))]
    fn new(width: usize, height: usize, smoothing: f32) -> PyResult<Self> {
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err("width and height must be positive"));
        }
        let clamped = smoothing.clamp(MIN_SMOOTHING, MAX_SMOOTHING);
        let size = width * height;
        Ok(Self {
            width,
            height,
            smoothing: clamped,
            canvas: vec![0.0; size],
            hypergrad: vec![0.0; size],
            realgrad: vec![0.0; size],
        })
    }

    #[getter]
    fn smoothing(&self) -> f32 {
        self.smoothing
    }

    fn refresh(
        &mut self,
        py: Python<'_>,
        projection: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<Vec<f32>>> {
        self.refresh_from_any(py, projection)
    }

    fn accumulate_hypergrad(
        &mut self,
        py: Python<'_>,
        gradient: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.accumulate_from_any(py, gradient, TapeTarget::Hypergrad)
    }

    fn accumulate_realgrad(&mut self, py: Python<'_>, gradient: &Bound<'_, PyAny>) -> PyResult<()> {
        self.accumulate_from_any(py, gradient, TapeTarget::Realgrad)
    }

    fn reset(&mut self) {
        for store in [&mut self.canvas, &mut self.hypergrad, &mut self.realgrad] {
            store.fill(0.0);
        }
    }

    fn gradient_summary(&self) -> HashMap<String, HashMap<String, f32>> {
        self.gradient_summary_map()
    }

    #[pyo3(signature = (vision, weight=1.0))]
    fn emit_zspace_patch(
        &mut self,
        py: Python<'_>,
        vision: &Bound<'_, PyAny>,
        weight: f32,
    ) -> PyResult<Vec<Vec<f32>>> {
        let projection = vision.call_method0("project").map_err(|err| {
            PyValueError::new_err(format!(
                "failed to obtain projection from vision pipeline: {err}"
            ))
        })?;
        let mut patch = self.refresh_from_any(py, &projection)?;
        if (weight - 1.0).abs() > f32::EPSILON {
            let scale = weight;
            for row in &mut patch {
                for value in row {
                    *value *= scale;
                }
            }
        }
        Ok(patch)
    }

    fn canvas(&self) -> Vec<Vec<f32>> {
        matrix_rows_from_slice(&self.canvas, self.height, self.width)
    }

    fn hypergrad(&self) -> Vec<Vec<f32>> {
        matrix_rows_from_slice(&self.hypergrad, self.height, self.width)
    }

    fn realgrad(&self) -> Vec<Vec<f32>> {
        matrix_rows_from_slice(&self.realgrad, self.height, self.width)
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("width", self.width)?;
        dict.set_item("height", self.height)?;
        dict.set_item("smoothing", self.smoothing)?;
        dict.set_item("canvas", self.canvas())?;
        dict.set_item("hypergrad", self.hypergrad())?;
        dict.set_item("realgrad", self.realgrad())?;
        Ok(dict.into())
    }

    #[pyo3(signature = (state, *, strict=true))]
    fn load_state_dict(
        &mut self,
        py: Python<'_>,
        state: &Bound<'_, PyAny>,
        strict: bool,
    ) -> PyResult<()> {
        let mapping = state
            .downcast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("state must be a mapping with string keys"))?;

        let mut width = self.width;
        let mut height = self.height;

        if let Some(value) = mapping.get_item("width")? {
            width = value.extract()?;
        }
        if let Some(value) = mapping.get_item("height")? {
            height = value.extract()?;
        }
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err(
                "state width and height must be positive",
            ));
        }
        if strict && (width != self.width || height != self.height) {
            return Err(PyValueError::new_err(
                "state dimensions do not match the canvas transformer",
            ));
        }

        if width != self.width || height != self.height {
            self.resize(width, height);
        }

        if let Some(value) = mapping.get_item("smoothing")? {
            self.smoothing = value.extract::<f32>()?.clamp(MIN_SMOOTHING, MAX_SMOOTHING);
        }

        if let Some(matrix) = mapping.get_item("canvas")? {
            let values = coerce_matrix(py, &matrix, self.height, self.width)?;
            self.canvas.copy_from_slice(&values);
        }

        if let Some(matrix) = mapping.get_item("hypergrad")? {
            let values = coerce_matrix(py, &matrix, self.height, self.width)?;
            self.hypergrad.copy_from_slice(&values);
        }

        if let Some(matrix) = mapping.get_item("realgrad")? {
            let values = coerce_matrix(py, &matrix, self.height, self.width)?;
            self.realgrad.copy_from_slice(&values);
        }

        Ok(())
    }

    fn snapshot(&self) -> PyCanvasSnapshot {
        self.snapshot_with_patch(None)
    }
}

impl PyCanvasTransformer {
    fn resize(&mut self, width: usize, height: usize) {
        let size = width * height;
        self.width = width;
        self.height = height;
        self.canvas.resize(size, 0.0);
        self.hypergrad.resize(size, 0.0);
        self.realgrad.resize(size, 0.0);
        self.canvas.fill(0.0);
        self.hypergrad.fill(0.0);
        self.realgrad.fill(0.0);
    }

    fn refresh_from_any(
        &mut self,
        py: Python<'_>,
        projection: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let matrix = coerce_matrix(py, projection, self.height, self.width)?;
        self.apply_smoothing(&matrix);
        let patch = matrix_rows_from_vec(matrix, self.height, self.width);
        Ok(patch)
    }

    fn apply_smoothing(&mut self, matrix: &[f32]) {
        let alpha = 1.0 - self.smoothing;
        for (dst, &src) in self.canvas.iter_mut().zip(matrix.iter()) {
            *dst = self.smoothing * *dst + alpha * src;
        }
    }

    fn accumulate_from_any(
        &mut self,
        py: Python<'_>,
        gradient: &Bound<'_, PyAny>,
        target: TapeTarget,
    ) -> PyResult<()> {
        let values = coerce_matrix(py, gradient, self.height, self.width)?;
        let store = match target {
            TapeTarget::Hypergrad => &mut self.hypergrad,
            TapeTarget::Realgrad => &mut self.realgrad,
        };
        for (dst, src) in store.iter_mut().zip(values.into_iter()) {
            *dst += src;
        }
        Ok(())
    }

    fn gradient_summary_map(&self) -> HashMap<String, HashMap<String, f32>> {
        let mut summary = HashMap::new();
        summary.insert("hypergrad".to_string(), matrix_summary(&self.hypergrad));
        summary.insert("realgrad".to_string(), matrix_summary(&self.realgrad));
        summary
    }

    fn snapshot_with_patch(&self, patch: Option<Vec<Vec<f32>>>) -> PyCanvasSnapshot {
        PyCanvasSnapshot::new(
            self.canvas(),
            self.hypergrad(),
            self.realgrad(),
            self.gradient_summary_map(),
            patch,
        )
    }
}

fn coerce_matrix(
    py: Python<'_>,
    any: &Bound<'_, PyAny>,
    height: usize,
    width: usize,
) -> PyResult<Vec<f32>> {
    if let Ok(tensor) = any.extract::<Py<PyTensor>>() {
        let tensor_ref = tensor.borrow(py);
        let (rows, cols) = tensor_ref.inner.shape();
        if rows != height || cols != width {
            return Err(PyValueError::new_err(format!(
                "expected tensor shape ({height}, {width}), received ({rows}, {cols})"
            )));
        }
        return Ok(tensor_ref.inner.data().to_vec());
    }

    let rows: Vec<Vec<f32>> = any.extract().map_err(|_| {
        PyTypeError::new_err("expected a Tensor or sequence of sequences of floats")
    })?;
    if rows.len() != height {
        return Err(PyValueError::new_err(format!(
            "expected {height} rows, received {}",
            rows.len()
        )));
    }
    let mut out = Vec::with_capacity(height * width);
    for (idx, row) in rows.iter().enumerate() {
        if row.len() != width {
            return Err(PyValueError::new_err(format!(
                "row {idx} expected {width} columns, received {}",
                row.len()
            )));
        }
        out.extend(row.iter().copied());
    }
    Ok(out)
}

fn matrix_rows_from_slice(data: &[f32], height: usize, width: usize) -> Vec<Vec<f32>> {
    data.chunks(width)
        .take(height)
        .map(|row| row.to_vec())
        .collect()
}

fn matrix_rows_from_vec(data: Vec<f32>, height: usize, width: usize) -> Vec<Vec<f32>> {
    data.chunks(width)
        .take(height)
        .map(|row| row.to_vec())
        .collect()
}

fn matrix_summary(data: &[f32]) -> HashMap<String, f32> {
    let mut summary = HashMap::new();
    if data.is_empty() {
        summary.insert("l1".to_string(), 0.0);
        summary.insert("l2".to_string(), 0.0);
        summary.insert("linf".to_string(), 0.0);
        summary.insert("mean".to_string(), 0.0);
        return summary;
    }

    let mut l1 = 0.0f32;
    let mut l2 = 0.0f32;
    let mut linf = 0.0f32;
    let mut sum = 0.0f32;
    for &value in data {
        let abs = value.abs();
        l1 += abs;
        l2 += value * value;
        linf = linf.max(abs);
        sum += value;
    }
    summary.insert("l1".to_string(), l1);
    summary.insert("l2".to_string(), l2.sqrt());
    summary.insert("linf".to_string(), linf);
    summary.insert("mean".to_string(), sum / data.len() as f32);
    summary
}

#[pyfunction]
#[pyo3(signature = (vision, canvas, *, hypergrad=None, realgrad=None, weight=1.0, include_patch=false))]
fn apply_vision_update(
    py: Python<'_>,
    vision: &Bound<'_, PyAny>,
    canvas: &Bound<'_, PyAny>,
    hypergrad: Option<&Bound<'_, PyAny>>,
    realgrad: Option<&Bound<'_, PyAny>>,
    weight: f32,
    include_patch: bool,
) -> PyResult<PyCanvasSnapshot> {
    let mut canvas_ref: PyRefMut<PyCanvasTransformer> = canvas.extract()?;
    let patch = canvas_ref.emit_zspace_patch(py, vision, weight)?;
    if let Some(hyper) = hypergrad {
        canvas_ref.accumulate_from_any(py, hyper, TapeTarget::Hypergrad)?;
    }
    if let Some(real) = realgrad {
        canvas_ref.accumulate_from_any(py, real, TapeTarget::Realgrad)?;
    }
    let patch_opt = if include_patch { Some(patch) } else { None };
    Ok(canvas_ref.snapshot_with_patch(patch_opt))
}

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "spiraltorch.vision")?;
    module.add(
        "__doc__",
        "Vision canvas transformers, contextual gates, and helpers",
    )?;
    module.add_class::<PyCanvasTransformer>()?;
    module.add_class::<PyCanvasSnapshot>()?;
    module.add_class::<PyVisionContextualGate>()?;
    module.add_class::<PyVisionPulseFrame>()?;
    let apply_update = wrap_pyfunction!(apply_vision_update, module.clone())?;
    module.add_function(apply_update)?;
    module.add(
        "__all__",
        vec![
            "CanvasTransformer",
            "CanvasSnapshot",
            "apply_vision_update",
            "ContextualVisionGate",
            "VisionPulseFrame",
        ],
    )?;
    let module_obj = module.to_object(py);
    parent.add_submodule(&module)?;
    parent.add("vision", module_obj.clone_ref(py))?;
    parent.add("CanvasTransformer", module.getattr("CanvasTransformer")?)?;
    parent.add("CanvasSnapshot", module.getattr("CanvasSnapshot")?)?;
    parent.add(
        "apply_vision_update",
        module.getattr("apply_vision_update")?,
    )?;
    parent.add(
        "ContextualVisionGate",
        module.getattr("ContextualVisionGate")?,
    )?;
    parent.add("VisionPulseFrame", module.getattr("VisionPulseFrame")?)?;
    Ok(())
}
