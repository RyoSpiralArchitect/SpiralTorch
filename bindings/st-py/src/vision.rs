use std::collections::HashMap;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};
use pyo3::{wrap_pyfunction, Bound, PyRefMut};

use crate::telemetry::PyAtlasFrame;
use crate::tensor::{tensor_err_to_py, PyTensor};
use crate::theory::PyZRelativityModel;
use st_tensor::wasm_canvas::{
    CanvasPalette, FractalCanvas as PureFractalCanvas, InfiniteZSpacePatch as PureInfiniteZSpacePatch,
};
use st_tensor::{Tensor, TensorError};

const MIN_SMOOTHING: f32 = 0.0;
const MAX_SMOOTHING: f32 = 0.999;

fn push_slice_stats(metrics: &mut HashMap<String, f64>, base: &str, data: &[f32]) {
    let total = data.len() as f64;
    let mut finite_count = 0usize;
    let mut sum = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;

    for &value in data {
        if !value.is_finite() {
            continue;
        }
        finite_count += 1;
        let v = value as f64;
        sum += v;
        sum_abs += v.abs();
        sum_sq += v * v;
        if value < min {
            min = value;
        }
        if value > max {
            max = value;
        }
    }

    metrics.insert(
        format!("{base}.finite_fraction"),
        if total > 0.0 {
            finite_count as f64 / total
        } else {
            0.0
        },
    );

    if finite_count == 0 {
        return;
    }

    let denom = finite_count as f64;
    metrics.insert(format!("{base}.mean"), sum / denom);
    metrics.insert(format!("{base}.abs_mean"), sum_abs / denom);
    metrics.insert(format!("{base}.min"), min as f64);
    metrics.insert(format!("{base}.max"), max as f64);
    metrics.insert(format!("{base}.rms"), (sum_sq / denom).sqrt());
    metrics.insert(format!("{base}.l2"), sum_sq.sqrt());
}

fn tensor_to_rows(tensor: &Tensor) -> Vec<Vec<f32>> {
    let (rows, cols) = tensor.shape();
    let data = tensor.data();
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let start = row * cols;
        out.push(data[start..start + cols].to_vec());
    }
    out
}

#[derive(Clone, Copy)]
enum TapeTarget {
    Hypergrad,
    Realgrad,
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

#[pyclass(module = "spiraltorch.vision", name = "InfiniteZSpacePatch")]
#[derive(Clone)]
pub(crate) struct PyInfiniteZSpacePatch {
    inner: PureInfiniteZSpacePatch,
}

impl From<PureInfiniteZSpacePatch> for PyInfiniteZSpacePatch {
    fn from(inner: PureInfiniteZSpacePatch) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyInfiniteZSpacePatch {
    #[getter]
    pub fn dimension(&self) -> f32 {
        self.inner.dimension()
    }

    #[getter]
    pub fn zoom(&self) -> f32 {
        self.inner.zoom()
    }

    #[getter]
    pub fn support(&self) -> (f32, f32) {
        self.inner.support()
    }

    #[getter]
    pub fn mellin_weights(&self) -> Vec<f32> {
        self.inner.mellin_weights().to_vec()
    }

    #[getter]
    pub fn density(&self) -> Vec<f32> {
        self.inner.density().to_vec()
    }

    pub fn eta_bar(&self) -> f32 {
        self.inner.eta_bar()
    }
}

#[pyclass(module = "spiraltorch.vision", name = "FractalCanvas")]
#[derive(Clone)]
pub(crate) struct PyFractalCanvas {
    inner: PureFractalCanvas,
    dimension: f32,
}

#[pymethods]
impl PyFractalCanvas {
    #[new]
    #[pyo3(signature = (dimension=2.0, capacity=32, width=64, height=64))]
    pub fn new(dimension: f32, capacity: usize, width: usize, height: usize) -> PyResult<Self> {
        let canvas = PureFractalCanvas::new(capacity.max(4), width.max(1), height.max(1))
            .map_err(tensor_err_to_py)?;
        Ok(Self {
            inner: canvas,
            dimension,
        })
    }

    #[getter]
    pub fn dimension(&self) -> f32 {
        self.dimension
    }

    pub fn set_dimension(&mut self, dimension: f32) {
        self.dimension = dimension;
    }

    #[pyo3(signature = (dimension=None, zoom=None, steps=None))]
    pub fn emit_zspace_patch(
        &self,
        dimension: Option<f32>,
        zoom: Option<f32>,
        steps: Option<usize>,
    ) -> PyResult<PyInfiniteZSpacePatch> {
        let dim = dimension.unwrap_or(self.dimension);
        let zoom_value = zoom.unwrap_or(f32::INFINITY);
        let step_value = steps.unwrap_or(96);
        let patch = self
            .inner
            .emit_zspace_patch(dim, zoom_value, step_value)
            .map_err(tensor_err_to_py)?;
        Ok(PyInfiniteZSpacePatch::from(patch))
    }

    #[pyo3(signature = (dimension=None))]
    pub fn emit_zspace_infinite(&self, dimension: Option<f32>) -> PyResult<PyInfiniteZSpacePatch> {
        let dim = dimension.unwrap_or(self.dimension);
        let patch = self
            .inner
            .emit_zspace_infinite(dim)
            .map_err(tensor_err_to_py)?;
        Ok(PyInfiniteZSpacePatch::from(patch))
    }

    #[pyo3(signature = (zoom=None, steps=None, dimension=None))]
    pub fn emit_infinite_z(
        &self,
        zoom: Option<f32>,
        steps: Option<usize>,
        dimension: Option<f32>,
    ) -> PyResult<PyInfiniteZSpacePatch> {
        self.emit_zspace_patch(dimension, zoom, steps)
    }
}

#[pyclass(module = "spiraltorch.canvas", name = "CanvasProjector", unsendable)]
pub(crate) struct PyCanvasProjector {
    inner: PureFractalCanvas,
}

#[pymethods]
impl PyCanvasProjector {
    #[new]
    #[pyo3(signature = (width=64, height=64, *, capacity=32, palette="blue-magenta"))]
    fn new(width: usize, height: usize, capacity: usize, palette: &str) -> PyResult<Self> {
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err("width and height must be positive"));
        }
        if capacity == 0 {
            return Err(PyValueError::new_err("capacity must be positive"));
        }
        let mut canvas = PureFractalCanvas::new(capacity, width, height).map_err(tensor_err_to_py)?;
        let palette = CanvasPalette::parse(palette)
            .ok_or_else(|| PyValueError::new_err(format!("unknown palette '{palette}'")))?;
        canvas.projector_mut().set_palette(palette);
        Ok(Self { inner: canvas })
    }

    #[getter]
    fn width(&self) -> usize {
        self.inner.width()
    }

    #[getter]
    fn height(&self) -> usize {
        self.inner.height()
    }

    fn queue_len(&self) -> usize {
        self.inner.scheduler().len()
    }

    fn total_weight(&self) -> f32 {
        self.inner.scheduler().total_weight()
    }

    fn palette(&self) -> &'static str {
        self.inner.projector().palette().canonical_name()
    }

    fn set_palette(&mut self, name: &str) -> PyResult<()> {
        let palette = CanvasPalette::parse(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown palette '{name}'")))?;
        self.inner.projector_mut().set_palette(palette);
        Ok(())
    }

    fn reset_normalizer(&mut self) {
        self.inner.projector_mut().normalizer_mut().reset();
    }

    #[pyo3(signature = (relation, *, coherence=1.0, tension=1.0, depth=0))]
    fn push_patch(
        &self,
        py: Python<'_>,
        relation: &Bound<'_, PyAny>,
        coherence: f32,
        tension: f32,
        depth: u32,
    ) -> PyResult<()> {
        let height = self.inner.height();
        let width = self.inner.width();
        let flat = coerce_matrix(py, relation, height, width)?;
        let tensor = Tensor::from_vec(height, width, flat).map_err(tensor_err_to_py)?;
        self.inner
            .push_patch(tensor, coherence, tension, depth)
            .map_err(tensor_err_to_py)
    }

    /// Emits a Z-space patch derived from the current colour-energy field.
    ///
    /// This is useful for feedback loops: call `emit_zspace_patch(...)` and
    /// feed the returned `relation` back into `push_patch(...)`.
    #[pyo3(signature = (*, coherence=1.0, tension=1.0, depth=0))]
    fn emit_zspace_patch(
        &mut self,
        py: Python<'_>,
        coherence: f32,
        tension: f32,
        depth: u32,
    ) -> PyResult<Py<PyDict>> {
        let patch = self
            .inner
            .projector_mut()
            .emit_zspace_patch(coherence, tension, depth)
            .map_err(tensor_err_to_py)?;
        let relation = Py::new(py, PyTensor::from_tensor(patch.relation().clone()))?;
        let dict = PyDict::new_bound(py);
        dict.set_item("relation", relation)?;
        dict.set_item("coherence", patch.coherence())?;
        dict.set_item("tension", patch.tension())?;
        dict.set_item("depth", patch.depth())?;
        dict.set_item("weight", patch.weight())?;
        Ok(dict.into())
    }

    /// Emits an AR/WebGPU-friendly trail packet derived from the current vector field.
    #[pyo3(signature = (curvature=1.0))]
    fn emit_wasm_trail(&mut self, py: Python<'_>, curvature: f32) -> PyResult<Py<PyDict>> {
        let trail = self
            .inner
            .projector_mut()
            .emit_wasm_trail(curvature)
            .map_err(tensor_err_to_py)?;
        let samples = trail.samples().len();
        let flat = trail.as_f32_slice();
        let tensor = Tensor::from_vec(samples, 7, flat).map_err(tensor_err_to_py)?;
        let samples_tensor = Py::new(py, PyTensor::from_tensor(tensor))?;

        let dict = PyDict::new_bound(py);
        dict.set_item("curvature", trail.curvature())?;
        dict.set_item("width", trail.width())?;
        dict.set_item("height", trail.height())?;
        dict.set_item("samples", samples_tensor)?;
        Ok(dict.into())
    }

    /// Emits an `AtlasFrame` (telemetry packet) describing the current canvas state.
    #[pyo3(signature = (*, prefix="canvas", refresh=true, timestamp=None))]
    fn emit_atlas_frame(
        &mut self,
        prefix: &str,
        refresh: bool,
        timestamp: Option<f64>,
    ) -> PyResult<PyAtlasFrame> {
        let prefix = prefix.trim();
        if prefix.is_empty() {
            return Err(PyValueError::new_err("prefix must be non-empty"));
        }

        if refresh {
            match self.inner.refresh_tensor() {
                Ok(_) | Err(TensorError::EmptyInput(_)) => {}
                Err(err) => return Err(tensor_err_to_py(err)),
            }
        }

        let projector = self.inner.projector();
        let tensor = projector.tensor();
        let (rows, cols) = tensor.shape();
        let mut metrics: HashMap<String, f64> = HashMap::new();
        metrics.insert(format!("{prefix}.queue_len"), self.inner.scheduler().len() as f64);
        metrics.insert(
            format!("{prefix}.total_weight"),
            self.inner.scheduler().total_weight() as f64,
        );
        metrics.insert(format!("{prefix}.shape.rows"), rows as f64);
        metrics.insert(format!("{prefix}.shape.cols"), cols as f64);

        push_slice_stats(&mut metrics, &format!("{prefix}.energy"), tensor.data());

        let palette = projector.palette();
        let (palette_id, palette_blue_magenta, palette_turbo, palette_grayscale) = match palette {
            CanvasPalette::BlueMagenta => (0.0, 1.0, 0.0, 0.0),
            CanvasPalette::Turbo => (1.0, 0.0, 1.0, 0.0),
            CanvasPalette::Grayscale => (2.0, 0.0, 0.0, 1.0),
        };
        metrics.insert(format!("{prefix}.palette.id"), palette_id);
        metrics.insert(format!("{prefix}.palette.blue_magenta"), palette_blue_magenta);
        metrics.insert(format!("{prefix}.palette.turbo"), palette_turbo);
        metrics.insert(format!("{prefix}.palette.grayscale"), palette_grayscale);

        let normalizer = projector.normalizer();
        let normalizer_state = normalizer.state();
        metrics.insert(format!("{prefix}.normalizer.alpha"), normalizer.alpha() as f64);
        metrics.insert(
            format!("{prefix}.normalizer.epsilon"),
            normalizer.epsilon() as f64,
        );
        metrics.insert(
            format!("{prefix}.normalizer.has_state"),
            if normalizer_state.is_some() { 1.0 } else { 0.0 },
        );
        if let Some((min, max)) = normalizer_state {
            metrics.insert(format!("{prefix}.normalizer.min"), min as f64);
            metrics.insert(format!("{prefix}.normalizer.max"), max as f64);
            metrics.insert(format!("{prefix}.normalizer.span"), (max - min) as f64);
            metrics.insert(format!("{prefix}.normalizer.center"), ((min + max) * 0.5) as f64);
        }

        let field = projector.vector_field();
        let vectors = field.vectors();
        metrics.insert(format!("{prefix}.trail.samples"), vectors.len() as f64);

        if !vectors.is_empty() {
            let curvature = 1.0f32;
            let curvature_scale = curvature.tanh();
            metrics.insert(format!("{prefix}.trail.curvature"), curvature as f64);
            metrics.insert(
                format!("{prefix}.trail.curvature_scale"),
                curvature_scale as f64,
            );

            let mut energy_total = vectors.len() as f64;
            let mut energy_finite = 0usize;
            let mut energy_sum = 0.0f64;
            let mut energy_sum_abs = 0.0f64;
            let mut energy_sum_sq = 0.0f64;
            let mut energy_min = f32::INFINITY;
            let mut energy_max = f32::NEG_INFINITY;

            let mut z_finite = 0usize;
            let mut z_sum = 0.0f64;
            let mut z_sum_abs = 0.0f64;
            let mut z_sum_sq = 0.0f64;
            let mut z_min = f32::INFINITY;
            let mut z_max = f32::NEG_INFINITY;

            let mut chroma_total = 0usize;
            let mut chroma_finite = 0usize;
            let mut chroma_sum_abs = 0.0f64;
            let mut chroma_sum_sq = 0.0f64;
            let mut chroma_min = f32::INFINITY;
            let mut chroma_max = f32::NEG_INFINITY;

            for vector in vectors {
                let energy = vector[0];
                if energy.is_finite() {
                    energy_finite += 1;
                    let v = energy as f64;
                    energy_sum += v;
                    energy_sum_abs += v.abs();
                    energy_sum_sq += v * v;
                    if energy < energy_min {
                        energy_min = energy;
                    }
                    if energy > energy_max {
                        energy_max = energy;
                    }

                    let z = curvature_scale * energy;
                    if z.is_finite() {
                        z_finite += 1;
                        let z64 = z as f64;
                        z_sum += z64;
                        z_sum_abs += z64.abs();
                        z_sum_sq += z64 * z64;
                        if z < z_min {
                            z_min = z;
                        }
                        if z > z_max {
                            z_max = z;
                        }
                    }
                }

                for channel in vector.iter().skip(1) {
                    chroma_total += 1;
                    let c = *channel;
                    if !c.is_finite() {
                        continue;
                    }
                    chroma_finite += 1;
                    let c64 = c as f64;
                    chroma_sum_abs += c64.abs();
                    chroma_sum_sq += c64 * c64;
                    if c < chroma_min {
                        chroma_min = c;
                    }
                    if c > chroma_max {
                        chroma_max = c;
                    }
                }
            }

            if energy_total <= 0.0 {
                energy_total = 0.0;
            }
            metrics.insert(
                format!("{prefix}.trail.energy.finite_fraction"),
                if energy_total > 0.0 {
                    energy_finite as f64 / energy_total
                } else {
                    0.0
                },
            );
            if energy_finite > 0 {
                let denom = energy_finite as f64;
                metrics.insert(format!("{prefix}.trail.energy.mean"), energy_sum / denom);
                metrics.insert(format!("{prefix}.trail.energy.abs_mean"), energy_sum_abs / denom);
                metrics.insert(format!("{prefix}.trail.energy.min"), energy_min as f64);
                metrics.insert(format!("{prefix}.trail.energy.max"), energy_max as f64);
                metrics.insert(format!("{prefix}.trail.energy.rms"), (energy_sum_sq / denom).sqrt());
                metrics.insert(format!("{prefix}.trail.energy.l2"), energy_sum_sq.sqrt());
            }

            metrics.insert(
                format!("{prefix}.trail.z.finite_fraction"),
                if energy_total > 0.0 {
                    z_finite as f64 / energy_total
                } else {
                    0.0
                },
            );
            if z_finite > 0 {
                let denom = z_finite as f64;
                metrics.insert(format!("{prefix}.trail.z.mean"), z_sum / denom);
                metrics.insert(format!("{prefix}.trail.z.abs_mean"), z_sum_abs / denom);
                metrics.insert(format!("{prefix}.trail.z.min"), z_min as f64);
                metrics.insert(format!("{prefix}.trail.z.max"), z_max as f64);
                metrics.insert(format!("{prefix}.trail.z.rms"), (z_sum_sq / denom).sqrt());
                metrics.insert(format!("{prefix}.trail.z.l2"), z_sum_sq.sqrt());
            }

            metrics.insert(
                format!("{prefix}.trail.chroma.finite_fraction"),
                if chroma_total > 0 {
                    chroma_finite as f64 / chroma_total as f64
                } else {
                    0.0
                },
            );
            if chroma_finite > 0 {
                let denom = chroma_finite as f64;
                metrics.insert(format!("{prefix}.trail.chroma.abs_mean"), chroma_sum_abs / denom);
                metrics.insert(format!("{prefix}.trail.chroma.min"), chroma_min as f64);
                metrics.insert(format!("{prefix}.trail.chroma.max"), chroma_max as f64);
                metrics.insert(format!("{prefix}.trail.chroma.rms"), (chroma_sum_sq / denom).sqrt());
                metrics.insert(format!("{prefix}.trail.chroma.l2"), chroma_sum_sq.sqrt());
            }
        }

        if let Ok(fft_db) = field.fft_rows_power_db_tensor(false) {
            let (fft_rows, fft_cols) = fft_db.shape();
            metrics.insert(format!("{prefix}.fft_db.shape.rows"), fft_rows as f64);
            metrics.insert(format!("{prefix}.fft_db.shape.cols"), fft_cols as f64);
            push_slice_stats(&mut metrics, &format!("{prefix}.fft_db"), fft_db.data());

            let mut counts = [0usize; 4];
            let mut sums = [0.0f64; 4];
            let mut mins = [f32::INFINITY; 4];
            let mut maxs = [f32::NEG_INFINITY; 4];

            for (idx, &value) in fft_db.data().iter().enumerate() {
                if !value.is_finite() {
                    continue;
                }
                let ch = idx % 4;
                counts[ch] += 1;
                sums[ch] += value as f64;
                if value < mins[ch] {
                    mins[ch] = value;
                }
                if value > maxs[ch] {
                    maxs[ch] = value;
                }
            }

            let labels = ["energy", "chroma_r", "chroma_g", "chroma_b"];
            for (idx, label) in labels.iter().enumerate() {
                let count = counts[idx];
                if count == 0 {
                    continue;
                }
                let denom = count as f64;
                metrics.insert(format!("{prefix}.fft_db.{label}.mean"), sums[idx] / denom);
                metrics.insert(format!("{prefix}.fft_db.{label}.min"), mins[idx] as f64);
                metrics.insert(format!("{prefix}.fft_db.{label}.max"), maxs[idx] as f64);
            }
        }

        PyAtlasFrame::from_metrics(metrics, timestamp)
    }

    fn clear_queue(&self) {
        while self.inner.scheduler().pop().is_some() {}
    }

    /// Returns the current RGBA surface (without forcing a refresh).
    fn rgba<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, self.inner.projector().surface().as_rgba())
    }

    /// Refreshes the projector and returns the RGBA surface as bytes.
    fn refresh_rgba<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        match self.inner.refresh() {
            Ok(rgba) => Ok(PyBytes::new_bound(py, rgba)),
            Err(TensorError::EmptyInput(_)) => Ok(self.rgba(py)),
            Err(err) => Err(tensor_err_to_py(err)),
        }
    }

    /// Refreshes the projector and returns the latest relation tensor.
    fn refresh_tensor(&mut self, py: Python<'_>) -> PyResult<PyTensor> {
        let _ = py;
        match self.inner.refresh_tensor() {
            Ok(tensor) => Ok(PyTensor::from_tensor(tensor.clone())),
            Err(TensorError::EmptyInput(_)) => Ok(PyTensor::from_tensor(self.inner.projector().tensor().clone())),
            Err(err) => Err(tensor_err_to_py(err)),
        }
    }

    /// Returns the last relation tensor (without forcing a refresh).
    fn tensor(&self, py: Python<'_>) -> PyResult<PyTensor> {
        let _ = py;
        Ok(PyTensor::from_tensor(self.inner.projector().tensor().clone()))
    }

    /// Refreshes and returns the row-wise complex FFT spectrum as a tensor with shape `(height, width * 8)`.
    #[pyo3(signature = (*, inverse=false))]
    fn refresh_vector_fft_tensor(&mut self, inverse: bool) -> PyResult<PyTensor> {
        let tensor = self
            .inner
            .projector_mut()
            .refresh_vector_fft_tensor(inverse)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(tensor))
    }

    /// Refreshes and returns the row-wise FFT log-power (dB) tensor with shape `(height, width * 4)`.
    #[pyo3(signature = (*, inverse=false))]
    fn refresh_vector_fft_power_db_tensor(&mut self, inverse: bool) -> PyResult<PyTensor> {
        let tensor = self
            .inner
            .projector_mut()
            .refresh_vector_fft_power_db_tensor(inverse)
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(tensor))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "CanvasProjector(width={width}, height={height}, palette={palette})",
            width = self.inner.width(),
            height = self.inner.height(),
            palette = self.palette(),
        ))
    }
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

#[pyfunction]
fn canvas_available_palettes() -> Vec<&'static str> {
    CanvasPalette::ALL
        .iter()
        .map(|palette| palette.canonical_name())
        .collect()
}

#[pyfunction]
fn canvas_canonical_palette(name: &str) -> PyResult<&'static str> {
    CanvasPalette::parse(name)
        .map(|palette| palette.canonical_name())
        .ok_or_else(|| PyValueError::new_err(format!("unknown palette '{name}'")))
}

#[pyfunction]
#[pyo3(signature = (model, field="block"))]
pub fn zrelativity_heatmap(model: &PyZRelativityModel, field: &str) -> PyResult<Vec<Vec<f32>>> {
    let bundle = model.inner.tensor_bundle().map_err(tensor_err_to_py)?;
    let lower = field.to_ascii_lowercase();
    let tensor = match lower.as_str() {
        "block" => &bundle.block_metric,
        "effective" => &bundle.effective_metric,
        "gauge" => &bundle.gauge_field,
        "moduli" => &bundle.scalar_moduli,
        "field_equation" => &bundle.field_equation,
        "warp" => bundle
            .warp
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("warp factor not present on model"))?,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown ZRelativity field '{other}' (expected block/effective/gauge/moduli/field_equation/warp)"
            )))
        }
    };
    Ok(tensor_to_rows(tensor))
}

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    parent.add_class::<PyCanvasTransformer>()?;
    parent.add_class::<PyCanvasSnapshot>()?;
    parent.add_class::<PyCanvasProjector>()?;
    parent.add_class::<PyFractalCanvas>()?;
    parent.add_class::<PyInfiniteZSpacePatch>()?;
    parent.add_function(wrap_pyfunction!(apply_vision_update, parent)?)?;
    parent.add_function(wrap_pyfunction!(canvas_available_palettes, parent)?)?;
    parent.add_function(wrap_pyfunction!(canvas_canonical_palette, parent)?)?;
    parent.add_function(wrap_pyfunction!(zrelativity_heatmap, parent)?)?;
    parent.add("__doc__", "Canvas transformer utilities")?;
    let _ = py;
    Ok(())
}
