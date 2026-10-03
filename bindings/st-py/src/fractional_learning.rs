use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyByteArray;
use st_frac::learning::{FractionalGlKernel, FractionalGlLearningBatch};

use crate::f32_buffer::{read_f32, write_f32};

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

    fn forward_history(
        &self,
        py: Python<'_>,
        input: Vec<f32>,
        shape: Vec<usize>,
        axis: usize,
        alpha: f32,
    ) -> PyResult<PyFractionalGlLearningBatch> {
        py.detach(|| self.inner.forward_history(&input, &shape, axis, alpha))
            .map(|inner| PyFractionalGlLearningBatch { inner })
            .map_err(value_error)
    }

    /// C-contiguous native-f32 buffer, copied before Rust releases the GIL.
    fn forward_buffer(
        &self,
        py: Python<'_>,
        input: &Bound<'_, PyAny>,
        shape: Vec<usize>,
        axis: usize,
        alpha: f32,
    ) -> PyResult<PyFractionalGlLearningBatch> {
        let input = read_f32(input, self.inner.max_values(), None)?;
        self.forward(py, input, shape, axis, alpha)
    }

    fn forward_history_buffer(
        &self,
        py: Python<'_>,
        input: &Bound<'_, PyAny>,
        shape: Vec<usize>,
        axis: usize,
        alpha: f32,
    ) -> PyResult<PyFractionalGlLearningBatch> {
        let input = read_f32(input, self.inner.max_values(), None)?;
        self.forward_history(py, input, shape, axis, alpha)
    }
}

#[pymethods]
impl PyFractionalGlLearningBatch {
    #[getter]
    fn output(&self) -> Vec<f32> {
        self.inner.output().iter().copied().collect()
    }

    /// Fresh writable native-f32 bytes, independent of this immutable snapshot.
    fn output_buffer<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyByteArray>> {
        write_f32(py, self.inner.output().iter().copied())
    }

    fn vjp(&self, py: Python<'_>, upstream: Vec<f32>) -> PyResult<(Vec<f32>, f32)> {
        py.detach(|| self.inner.vjp(&upstream))
            .map(|gradient| (gradient.input, gradient.alpha))
            .map_err(value_error)
    }

    fn vjp_input(&self, py: Python<'_>, upstream: Vec<f32>) -> PyResult<Vec<f32>> {
        py.detach(|| self.inner.vjp_input(&upstream))
            .map_err(value_error)
    }

    fn vjp_alpha(&self, py: Python<'_>, upstream: Vec<f32>) -> PyResult<f32> {
        py.detach(|| self.inner.vjp_alpha(&upstream))
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

    fn vjp_buffer<'py>(
        &self,
        py: Python<'py>,
        upstream: &Bound<'_, PyAny>,
    ) -> PyResult<(Bound<'py, PyByteArray>, f32)> {
        let len = self.inner.output().len();
        let upstream = read_f32(upstream, len, Some(len))?;
        let gradient = py
            .detach(|| self.inner.vjp(&upstream))
            .map_err(value_error)?;
        Ok((write_f32(py, gradient.input.into_iter())?, gradient.alpha))
    }

    fn vjp_input_buffer<'py>(
        &self,
        py: Python<'py>,
        upstream: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyByteArray>> {
        let len = self.inner.output().len();
        let upstream = read_f32(upstream, len, Some(len))?;
        let gradient = py
            .detach(|| self.inner.vjp_input(&upstream))
            .map_err(value_error)?;
        write_f32(py, gradient.into_iter())
    }

    fn vjp_alpha_buffer(&self, py: Python<'_>, upstream: &Bound<'_, PyAny>) -> PyResult<f32> {
        let len = self.inner.output().len();
        let upstream = read_f32(upstream, len, Some(len))?;
        py.detach(|| self.inner.vjp_alpha(&upstream))
            .map_err(value_error)
    }

    fn jvp_buffer<'py>(
        &self,
        py: Python<'py>,
        input_tangent: &Bound<'_, PyAny>,
        alpha_tangent: f32,
    ) -> PyResult<Bound<'py, PyByteArray>> {
        let len = self.inner.output().len();
        let tangent = read_f32(input_tangent, len, Some(len))?;
        let output = py
            .detach(|| self.inner.jvp(&tangent, alpha_tangent))
            .map_err(value_error)?;
        write_f32(py, output.into_iter())
    }
}

pub(crate) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_class::<PyFractionalGlKernel>()?;
    parent.add_class::<PyFractionalGlLearningBatch>()
}
