use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use st_frac::learning::{FractionalGlKernel, FractionalGlLearningBatch};

fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pyclass(name = "FractionalGlKernel", module = "spiraltorch", frozen)]
pub struct PyFractionalGlKernel {
    inner: FractionalGlKernel,
}

#[pyclass(name = "FractionalGlLearningBatch", module = "spiraltorch", frozen)]
pub struct PyFractionalGlLearningBatch {
    inner: FractionalGlLearningBatch,
}

#[pymethods]
impl PyFractionalGlKernel {
    #[new]
    #[pyo3(signature = (*, kernel_len=32, step=1.0, max_values=1_048_576, max_products=16_777_216))]
    fn new(kernel_len: usize, step: f32, max_values: usize, max_products: usize) -> PyResult<Self> {
        FractionalGlKernel::new(kernel_len, step, max_values, max_products)
            .map(|inner| Self { inner })
            .map_err(value_error)
    }

    #[getter]
    fn execution_backend(&self) -> &'static str {
        "rust_f32_cpu"
    }

    fn configuration_json(&self) -> String {
        serde_json::json!({
            "kernel_len": self.inner.kernel_len(), "step": self.inner.step(),
            "max_values": self.inner.max_values(), "max_products": self.inner.max_products(),
        })
        .to_string()
    }

    fn forward(
        &self,
        py: Python<'_>,
        input: Vec<f32>,
        shape: Vec<usize>,
        axis: usize,
        alpha: f32,
    ) -> PyResult<PyFractionalGlLearningBatch> {
        py.detach(|| self.inner.forward(&input, &shape, axis, alpha))
            .map(|inner| PyFractionalGlLearningBatch { inner })
            .map_err(value_error)
    }
}

#[pymethods]
impl PyFractionalGlLearningBatch {
    #[getter]
    fn output(&self) -> Vec<f32> {
        self.inner.output().iter().copied().collect()
    }

    fn vjp(&self, py: Python<'_>, upstream: Vec<f32>) -> PyResult<(Vec<f32>, f32)> {
        py.detach(|| self.inner.vjp(&upstream))
            .map(|gradient| (gradient.input, gradient.alpha))
            .map_err(value_error)
    }

    fn jvp(
        &self,
        py: Python<'_>,
        input_tangent: Vec<f32>,
        alpha_tangent: f32,
    ) -> PyResult<Vec<f32>> {
        py.detach(|| self.inner.jvp(&input_tangent, alpha_tangent))
            .map_err(value_error)
    }
}

pub(crate) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_class::<PyFractionalGlKernel>()?;
    parent.add_class::<PyFractionalGlLearningBatch>()
}
