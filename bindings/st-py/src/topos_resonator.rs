use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyByteArray;
use st_core::dynamics::topos_resonator::{
    ToposResonatorConfig, ToposResonatorLearningBatch, ToposResonatorOperator,
};
use st_tensor::topos::OpenCartesianTopos;

use crate::f32_buffer::{read_f32, write_f32};

fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// Explicit f32 CPU execution; GPU tensors are transported by the client.
#[pyclass(name = "ToposResonatorKernel", module = "spiraltorch", frozen)]
pub struct PyToposResonatorKernel {
    pub(crate) operator: ToposResonatorOperator,
}

#[pyclass(name = "ToposResonatorLearningBatch", module = "spiraltorch", frozen)]
pub struct PyToposResonatorLearningBatch {
    inner: ToposResonatorLearningBatch,
}

#[pymethods]
impl PyToposResonatorKernel {
    #[new]
    #[pyo3(signature = (*, coupling=0.25, iterations=4, saturation=1.0, porosity=0.0, max_values=1_048_576))]
    fn new(
        coupling: f32,
        iterations: usize,
        saturation: f32,
        porosity: f32,
        max_values: usize,
    ) -> PyResult<Self> {
        let config = ToposResonatorConfig::new(coupling, iterations).map_err(value_error)?;
        let topos = OpenCartesianTopos::new(-1.0, 1e-6, saturation, iterations + 1, max_values)
            .map_err(value_error)?
            .with_porosity(porosity)
            .map_err(value_error)?;
        Ok(Self {
            operator: ToposResonatorOperator::new(config, topos).map_err(value_error)?,
        })
    }

    #[getter]
    fn execution_backend(&self) -> &'static str {
        "rust_f32_cpu"
    }

    #[getter]
    fn max_values(&self) -> usize {
        self.operator.topos().max_volume()
    }

    fn configuration_json(&self) -> String {
        let config = self.operator.config();
        let topos = self.operator.topos();
        serde_json::json!({
            "coupling": config.coupling(), "iterations": config.iterations(),
            "saturation": topos.saturation(), "porosity": topos.porosity(),
            "max_values": topos.max_volume(),
        })
        .to_string()
    }

    fn forward(
        &self,
        py: Python<'_>,
        input: Vec<f32>,
        gate: Vec<f32>,
        rows: usize,
        features: usize,
    ) -> PyResult<Vec<f32>> {
        py.detach(|| self.operator.forward(&input, &gate, rows, features))
            .map(|step| step.output)
            .map_err(value_error)
    }

    fn backward(
        &self,
        py: Python<'_>,
        input: Vec<f32>,
        gate: Vec<f32>,
        grad_output: Vec<f32>,
        rows: usize,
        features: usize,
    ) -> PyResult<(Vec<f32>, Vec<f32>)> {
        py.detach(|| {
            self.operator
                .backward(&input, &gate, &grad_output, rows, features)
        })
        .map(|result| (result.grad_input, result.grad_gate))
        .map_err(value_error)
    }

    fn forward_buffer<'py>(
        &self,
        py: Python<'py>,
        input: &Bound<'_, PyAny>,
        gate: &Bound<'_, PyAny>,
        rows: usize,
        features: usize,
    ) -> PyResult<Bound<'py, PyByteArray>> {
        let input = read_f32(input, self.max_values(), None)?;
        let gate = read_f32(gate, self.max_values(), Some(input.len()))?;
        let values = self.forward(py, input, gate, rows, features)?;
        write_f32(py, values.into_iter())
    }

    fn backward_buffer<'py>(
        &self,
        py: Python<'py>,
        input: &Bound<'_, PyAny>,
        gate: &Bound<'_, PyAny>,
        grad_output: &Bound<'_, PyAny>,
        rows: usize,
        features: usize,
    ) -> PyResult<(Bound<'py, PyByteArray>, Bound<'py, PyByteArray>)> {
        let input = read_f32(input, self.max_values(), None)?;
        let gate = read_f32(gate, self.max_values(), Some(input.len()))?;
        let grad_output = read_f32(grad_output, self.max_values(), Some(input.len()))?;
        let (dx, dg) = self.backward(py, input, gate, grad_output, rows, features)?;
        Ok((
            write_f32(py, dx.into_iter())?,
            write_f32(py, dg.into_iter())?,
        ))
    }

    fn capture(
        &self,
        py: Python<'_>,
        input: Vec<f32>,
        gate: Vec<f32>,
        rows: usize,
        features: usize,
    ) -> PyResult<PyToposResonatorLearningBatch> {
        py.detach(|| self.operator.capture_owned(input, gate, rows, features))
            .map(|inner| PyToposResonatorLearningBatch { inner })
            .map_err(value_error)
    }

    fn capture_buffer(
        &self,
        py: Python<'_>,
        input: &Bound<'_, PyAny>,
        gate: &Bound<'_, PyAny>,
        rows: usize,
        features: usize,
    ) -> PyResult<PyToposResonatorLearningBatch> {
        let input = read_f32(input, self.max_values(), None)?;
        let gate = read_f32(gate, self.max_values(), Some(input.len()))?;
        self.capture(py, input, gate, rows, features)
    }
}

#[pymethods]
impl PyToposResonatorLearningBatch {
    #[getter]
    fn output(&self) -> Vec<f32> {
        self.inner.output().to_vec()
    }

    fn output_buffer<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyByteArray>> {
        write_f32(py, self.inner.output().iter().copied())
    }

    fn audit_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| serde_json::to_string(&self.inner.step().audit))
            .map_err(value_error)
    }

    fn vjp(&self, py: Python<'_>, upstream: Vec<f32>) -> PyResult<(Vec<f32>, Vec<f32>)> {
        py.detach(|| self.inner.vjp(&upstream))
            .map(|g| (g.grad_input, g.grad_gate))
            .map_err(value_error)
    }

    fn vjp_buffer<'py>(
        &self,
        py: Python<'py>,
        upstream: &Bound<'_, PyAny>,
    ) -> PyResult<(Bound<'py, PyByteArray>, Bound<'py, PyByteArray>)> {
        let count = self.inner.output().len();
        let upstream = read_f32(upstream, count, Some(count))?;
        let (dx, dg) = self.vjp(py, upstream)?;
        Ok((
            write_f32(py, dx.into_iter())?,
            write_f32(py, dg.into_iter())?,
        ))
    }
}

pub(crate) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_class::<PyToposResonatorKernel>()?;
    parent.add_class::<PyToposResonatorLearningBatch>()
}
