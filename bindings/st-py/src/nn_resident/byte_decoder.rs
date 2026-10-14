//! Python owns handles only; geometry, causality, updates and resume live in Rust.
use super::residual_attention::PyResidualAttentionPlan;
use super::*;
use pyo3::types::{PyBytes, PyDict};
use st_nn::resident::{
    ByteDecoderCheckpoint, ByteDecoderGeometryPlan, ByteDecoderPairMetric,
    ByteDecoderParameterLayout, ByteDecoderPlan, ByteLmBatch,
};

mod training;
use training::*;
mod artifact;
mod serving;
pub(super) use artifact::PyByteCorpusModelArtifact;

fn metric(value: &str) -> PyResult<ByteDecoderPairMetric> {
    serde_json::from_value(serde_json::Value::String(value.to_owned()))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

fn metric_name(value: ByteDecoderPairMetric) -> String {
    // The enum's serde names are the Rust checkpoint protocol, not binding aliases.
    serde_json::to_value(value)
        .unwrap()
        .as_str()
        .unwrap()
        .to_owned()
}

fn parameter_layout(py: Python<'_>, layout: &ByteDecoderParameterLayout) -> PyResult<Py<PyAny>> {
    let range = |r: std::ops::Range<usize>| (r.start, r.end);
    let result = PyDict::new(py);
    result.set_item("count", layout.len())?;
    result.set_item("token_embedding", layout.token_embedding())?;
    result.set_item("position_embedding", layout.position_embedding())?;
    result.set_item(
        "blocks",
        layout
            .blocks()
            .iter()
            .cloned()
            .map(range)
            .collect::<Vec<_>>(),
    )?;
    result.set_item("head", range(layout.head()))?;
    if let Some(g) = layout.geometry() {
        let geometry = PyDict::new(py);
        geometry.set_item("all", range(g.all()))?;
        geometry.set_item("projection", range(g.projection()))?;
        geometry.set_item("raw_decay", g.raw_decay())?;
        geometry.set_item("raw_phase", g.raw_phase())?;
        geometry.set_item("raw_gains", range(g.raw_gains()))?;
        geometry.set_item("mixture_logits", g.mixture_logits().map(range))?;
        geometry.set_item(
            "query_projections",
            g.query_projections()
                .iter()
                .cloned()
                .map(range)
                .collect::<Vec<_>>(),
        )?;
        result.set_item("geometry", geometry)?;
    } else {
        result.set_item("geometry", py.None())?;
    }
    Ok(result.into_any().unbind())
}

#[pyclass(name = "ByteLmBatch", module = "spiraltorch.nn", frozen)]
pub(super) struct PyByteLmBatch {
    inner: ByteLmBatch,
}

#[pymethods]
impl PyByteLmBatch {
    /// Equal-length T+1 byte windows. No text encoding, padding or concatenation.
    #[staticmethod]
    fn from_windows(windows: Vec<Bound<'_, PyBytes>>) -> PyResult<Self> {
        let windows: Vec<_> = windows.iter().map(|w| w.as_bytes()).collect();
        Ok(Self {
            inner: ByteLmBatch::from_windows(&windows).map_err(plan_error)?,
        })
    }

    /// Selections use (document index, byte offset), never character offsets.
    #[staticmethod]
    fn from_documents(
        documents: Vec<Bound<'_, PyBytes>>,
        selections: &Bound<'_, PyAny>,
        steps: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let integer = |v: &Bound<'_, PyAny>| -> PyResult<usize> {
            if v.is_instance_of::<PyBool>() {
                return Err(PyTypeError::new_err(
                    "byte indices must be integers, not bool",
                ));
            }
            v.extract()
        };
        let steps = integer(steps)?;
        let selections = selections
            .try_iter()?
            .map(|row| {
                let row = row?;
                let parts = row.try_iter()?.collect::<PyResult<Vec<_>>>()?;
                if parts.len() != 2 {
                    return Err(PyValueError::new_err(
                        "selection needs document index and byte offset",
                    ));
                }
                Ok((integer(&parts[0])?, integer(&parts[1])?))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let documents: Vec<_> = documents.iter().map(|d| d.as_bytes()).collect();
        Ok(Self {
            inner: ByteLmBatch::from_documents(&documents, &selections, steps)
                .map_err(plan_error)?,
        })
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        let [batch, steps] = self.inner.shape();
        (batch, steps)
    }
    #[getter]
    fn input_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.inner.input_bytes())
    }
    #[getter]
    fn target_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.inner.target_bytes())
    }
}

#[pyclass(name = "ByteDecoderGeometryPlan", module = "spiraltorch.nn", frozen)]
pub(super) struct PyByteDecoderGeometryPlan {
    inner: ByteDecoderGeometryPlan,
}

#[pymethods]
impl PyByteDecoderGeometryPlan {
    #[new]
    fn new(
        projection: &PyInferencePlan,
        raw_decay: Vec<f32>,
        raw_phase: Vec<f32>,
        raw_gains: Vec<Vec<f32>>,
        curvature: f32,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: ByteDecoderGeometryPlan::new(
                &projection.inner,
                &raw_decay,
                &raw_phase,
                &raw_gains,
                curvature,
            )
            .map_err(plan_error)?,
        })
    }

    /// Return a new plan; existing compiled models and plans are unchanged.
    fn with_pair_metric(&self, name: &str) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_pair_metric(metric(name)?),
        })
    }
    fn with_head_pair_metrics(&self, metrics: Vec<Vec<String>>) -> PyResult<Self> {
        let metrics = metrics
            .iter()
            .map(|row| row.iter().map(|s| metric(s)).collect())
            .collect::<PyResult<Vec<Vec<_>>>>()?;
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_head_pair_metrics(&metrics)
                .map_err(plan_error)?,
        })
    }
    fn with_mixture(
        &self,
        metrics: Vec<String>,
        raw_logits: Vec<Vec<f32>>,
        scales: Vec<Vec<f32>>,
    ) -> PyResult<Self> {
        let metrics = metrics
            .iter()
            .map(|s| metric(s))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_mixture(&metrics, &raw_logits, &scales)
                .map_err(plan_error)?,
        })
    }
    /// Router inputs are causal geometry coordinates, not the attention Q tensor.
    fn with_query_projections(
        &self,
        projections: Vec<PyRef<'_, PyInferencePlan>>,
    ) -> PyResult<Self> {
        let projections = projections
            .iter()
            .map(|p| p.inner.clone())
            .collect::<Vec<_>>();
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_query_projections(&projections)
                .map_err(plan_error)?,
        })
    }
    #[getter]
    fn pair_metric(&self) -> Option<String> {
        self.inner.pair_metric().map(metric_name)
    }
    #[getter]
    fn head_pair_metrics(&self) -> Option<Vec<Vec<String>>> {
        self.inner.head_pair_metrics().map(|rows| {
            rows.iter()
                .map(|r| r.iter().copied().map(metric_name).collect())
                .collect()
        })
    }
    #[getter]
    fn mixture_metrics(&self) -> Option<Vec<String>> {
        self.inner
            .mixture()
            .map(|m| m.metrics().iter().copied().map(metric_name).collect())
    }
    #[getter]
    fn contextual(&self) -> bool {
        self.inner
            .mixture()
            .is_some_and(|m| !m.query_projections().is_empty())
    }
    #[getter]
    fn curvature(&self) -> f32 {
        self.inner.curvature()
    }
}

#[pyclass(name = "ByteDecoderPlan", module = "spiraltorch.nn", frozen)]
pub(super) struct PyByteDecoderPlan {
    inner: ByteDecoderPlan,
}

#[pymethods]
impl PyByteDecoderPlan {
    #[staticmethod]
    fn from_plans(
        token_table: &PyTensor,
        position_table: &PyTensor,
        blocks: Vec<PyRef<'_, PyResidualAttentionPlan>>,
        head: &PyInferencePlan,
    ) -> PyResult<Self> {
        let blocks = blocks.iter().map(|b| b.inner.clone()).collect::<Vec<_>>();
        Ok(Self {
            inner: ByteDecoderPlan::from_plans(
                &token_table.inner,
                &position_table.inner,
                &blocks,
                &head.inner,
            )
            .map_err(plan_error)?,
        })
    }
    fn with_causal_geometry(&self, geometry: &PyByteDecoderGeometryPlan) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_causal_geometry(geometry.inner.clone())
                .map_err(plan_error)?,
        })
    }
    fn with_initial_geometry_raw_gains(&self, raw_gains: Vec<Vec<f32>>) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_initial_geometry_raw_gains(&raw_gains)
                .map_err(plan_error)?,
        })
    }
    #[getter]
    fn causal_geometry(&self) -> Option<PyByteDecoderGeometryPlan> {
        self.inner
            .causal_geometry()
            .cloned()
            .map(|inner| PyByteDecoderGeometryPlan { inner })
    }
    #[getter]
    fn input_shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.input_layout().shape().iter().copied())
    }
    #[getter]
    fn output_shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.output_layout().shape().iter().copied())
    }
    #[getter]
    fn block_count(&self) -> usize {
        self.inner.block_count()
    }
    #[getter]
    fn position_capacity(&self) -> usize {
        self.inner.position_capacity()
    }
    #[getter]
    fn embedding_width(&self) -> usize {
        self.inner.embedding_width()
    }
    fn parameter_layout(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        parameter_layout(py, self.inner.parameter_layout())
    }
    fn initial_checkpoint(&self) -> PyByteDecoderCheckpoint {
        PyByteDecoderCheckpoint {
            inner: self.inner.initial_checkpoint(),
        }
    }
    #[pyo3(signature = (*, tile_mnk=None, kernel="scalar", accumulation="sequential"))]
    fn compile_training_wgpu(
        &self,
        py: Python<'_>,
        tile_mnk: Option<&Bound<'_, PyAny>>,
        kernel: &str,
        accumulation: &str,
    ) -> PyResult<PyResidentByteDecoder> {
        compile(
            py,
            self.inner.initial_checkpoint(),
            tile_mnk,
            kernel,
            accumulation,
        )
    }
}

#[pyclass(name = "ByteDecoderCheckpoint", module = "spiraltorch.nn", frozen)]
pub(super) struct PyByteDecoderCheckpoint {
    inner: ByteDecoderCheckpoint,
}

impl From<ByteDecoderCheckpoint> for PyByteDecoderCheckpoint {
    fn from(inner: ByteDecoderCheckpoint) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyByteDecoderCheckpoint {
    #[staticmethod]
    #[pyo3(signature = (payload, *, max_bytes=DEFAULT_MAX_PLAN_JSON_BYTES))]
    fn from_json(payload: &str, max_bytes: usize) -> PyResult<Self> {
        Ok(Self {
            inner: ByteDecoderCheckpoint::from_json_with_limit(payload, max_bytes)
                .map_err(plan_error)?,
        })
    }
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(plan_error)
    }
    #[getter]
    fn attempted_revision(&self) -> u64 {
        self.inner.attempted_revision()
    }
    /// Frozen topology and values; compiling the plan starts a new revision zero.
    fn plan(&self) -> PyByteDecoderPlan {
        PyByteDecoderPlan {
            inner: self.inner.plan().clone(),
        }
    }
    /// Restore values and attempted revision under a fresh parameter owner.
    #[pyo3(signature = (*, tile_mnk=None, kernel="scalar", accumulation="sequential"))]
    fn restore_wgpu(
        &self,
        py: Python<'_>,
        tile_mnk: Option<&Bound<'_, PyAny>>,
        kernel: &str,
        accumulation: &str,
    ) -> PyResult<PyResidentByteDecoder> {
        compile(py, self.inner.clone(), tile_mnk, kernel, accumulation)
    }
}

fn compile(
    py: Python<'_>,
    checkpoint: ByteDecoderCheckpoint,
    tile_mnk: Option<&Bound<'_, PyAny>>,
    kernel: &str,
    accumulation: &str,
) -> PyResult<PyResidentByteDecoder> {
    #[cfg(feature = "wgpu")]
    {
        let (tile, kernel, accumulation) = gpu_options(tile_mnk, kernel, accumulation)?;
        py.detach(move || {
            let (runtime, _) =
                st_backend_wgpu::runtime::ensure_default_runtime_blocking("python.nn.byte_decoder")
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(PyResidentByteDecoder {
                inner: checkpoint
                    .restore_wgpu_with_options(runtime, tile, kernel, accumulation)
                    .map_err(plan_error)?,
            })
        })
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let _ = (py, checkpoint, tile_mnk, kernel, accumulation);
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "resident byte training requires the wgpu feature",
        ))
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyByteLmBatch>()?;
    module.add_class::<PyByteDecoderGeometryPlan>()?;
    module.add_class::<PyByteDecoderPlan>()?;
    module.add_class::<PyByteDecoderCheckpoint>()?;
    artifact::register(module)?;
    serving::register(module)?;
    training::register(module)
}
