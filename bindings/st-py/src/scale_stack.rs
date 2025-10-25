use ndarray::{Array2, ArrayD, IxDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Bound;
use st_frac::scale_stack::{
    InterfaceMode, ScaleSample, ScaleStack, ScaleStackError, SemanticMetric,
};

fn parse_metric(metric: &str) -> PyResult<SemanticMetric> {
    match metric.to_ascii_lowercase().as_str() {
        "euclidean" => Ok(SemanticMetric::Euclidean),
        "cosine" => Ok(SemanticMetric::Cosine),
        other => Err(PyValueError::new_err(format!(
            "unknown semantic metric '{other}', expected 'euclidean' or 'cosine'"
        ))),
    }
}

fn map_error(err: ScaleStackError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[pyclass(name = "ScaleStack")]
pub struct PyScaleStack {
    inner: ScaleStack,
}

impl PyScaleStack {
    fn from_inner(inner: ScaleStack) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyScaleStack {
    #[getter]
    fn threshold(&self) -> f32 {
        self.inner.threshold()
    }

    #[getter]
    fn mode(&self) -> String {
        match self.inner.mode() {
            InterfaceMode::Scalar => "scalar".to_string(),
            InterfaceMode::Semantic { metric, .. } => match metric {
                SemanticMetric::Euclidean => "semantic::euclidean".to_string(),
                SemanticMetric::Cosine => "semantic::cosine".to_string(),
            },
        }
    }

    #[getter]
    fn samples(&self) -> Vec<(f64, f64)> {
        self.inner
            .samples()
            .iter()
            .map(|ScaleSample { scale, gate_mean }| (*scale, *gate_mean))
            .collect()
    }

    fn persistence(&self) -> Vec<(f64, f64, f64)> {
        self.inner
            .persistence_measure()
            .into_iter()
            .map(|bin| (bin.scale_low, bin.scale_high, bin.mass))
            .collect()
    }

    fn interface_density(&self) -> Option<f64> {
        self.inner.interface_density()
    }

    fn moment(&self, order: u32) -> f64 {
        self.inner.moment(order)
    }

    fn boundary_dimension(&self, ambient_dim: f64, window: usize) -> Option<f64> {
        self.inner.estimate_boundary_dimension(ambient_dim, window)
    }

    fn coherence_break_scale(&self, level: f64) -> Option<f64> {
        self.inner.coherence_break_scale(level)
    }

    fn coherence_profile(&self, levels: Vec<f64>) -> Vec<Option<f64>> {
        self.inner.coherence_profile(&levels)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "ScaleStack(mode={}, samples={})",
            self.mode(),
            self.inner.samples().len()
        ))
    }
}

#[pyfunction]
#[pyo3(signature = (field, shape, scales, threshold))]
fn scalar_scale_stack(
    field: Vec<f32>,
    shape: Vec<usize>,
    scales: Vec<f64>,
    threshold: f32,
) -> PyResult<PyScaleStack> {
    let array = ArrayD::from_shape_vec(IxDyn(&shape), field)
        .map_err(|_| PyValueError::new_err("field shape does not match provided dimensions"))?;
    let stack =
        ScaleStack::from_scalar_field(array.view(), &scales, threshold).map_err(map_error)?;
    Ok(PyScaleStack::from_inner(stack))
}

#[pyfunction]
#[pyo3(signature = (embeddings, scales, threshold, metric="euclidean"))]
fn semantic_scale_stack(
    embeddings: Vec<Vec<f32>>,
    scales: Vec<f64>,
    threshold: f32,
    metric: &str,
) -> PyResult<PyScaleStack> {
    let metric = parse_metric(metric)?;
    let feature_dim = if let Some(sample) = embeddings.first() {
        sample.len()
    } else {
        0
    };
    if embeddings.iter().any(|sample| sample.len() != feature_dim) {
        return Err(PyValueError::new_err(
            "embeddings must share the same feature dimension",
        ));
    }
    let mut flat = Vec::with_capacity(embeddings.len() * feature_dim.max(1));
    for sample in &embeddings {
        flat.extend_from_slice(sample);
    }
    let array = Array2::from_shape_vec((embeddings.len(), feature_dim), flat)
        .map_err(|_| PyValueError::new_err("unable to reshape embeddings"))?;
    let stack =
        ScaleStack::from_semantic_field(array.view().into_dyn(), &scales, threshold, 1, metric)
            .map_err(map_error)?;
    Ok(PyScaleStack::from_inner(stack))
}

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "scale_stack")?;
    module.add_class::<PyScaleStack>()?;
    module.add_function(wrap_pyfunction!(scalar_scale_stack, &module)?)?;
    module.add_function(wrap_pyfunction!(semantic_scale_stack, &module)?)?;
    module.add(
        "__all__",
        vec!["ScaleStack", "scalar_scale_stack", "semantic_scale_stack"],
    )?;
    module.add("__doc__", "Scale persistence helpers")?;
    parent.add_submodule(&module)?;
    parent.add_class::<PyScaleStack>()?;
    parent.add_function(wrap_pyfunction!(scalar_scale_stack, parent)?)?;
    parent.add_function(wrap_pyfunction!(semantic_scale_stack, parent)?)?;
    Ok(())
}
