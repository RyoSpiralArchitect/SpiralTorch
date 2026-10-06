use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyByteArray;
use st_nn::{WaveGateKernel, WaveGateLearningBatch};

use crate::f32_buffer::{read_f32, write_f32};

type BufferVjp<'py> = (
    Bound<'py, PyByteArray>,
    Bound<'py, PyByteArray>,
    Bound<'py, PyByteArray>,
);

fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pyclass(name = "WaveGateKernel", module = "spiraltorch", frozen)]
pub struct PyWaveGateKernel {
    inner: WaveGateKernel,
}

#[pyclass(name = "WaveGateLearningBatch", module = "spiraltorch", frozen)]
pub struct PyWaveGateLearningBatch {
    inner: WaveGateLearningBatch,
}

#[pymethods]
impl PyWaveGateKernel {
    #[new]
    #[pyo3(signature = (*, curvature=-1.0, saturation=1.0, porosity=0.05, max_values=1_048_576))]
    fn new(curvature: f32, saturation: f32, porosity: f32, max_values: usize) -> PyResult<Self> {
        Ok(Self {
            inner: WaveGateKernel::new(curvature, saturation, porosity, max_values)
                .map_err(value_error)?,
        })
    }

    #[getter]
    fn execution_backend(&self) -> &'static str {
        "rust_f32_cpu"
    }

    #[getter]
    fn max_values(&self) -> usize {
        self.inner.topos().max_volume()
    }

    fn configuration_json(&self) -> String {
        let topos = self.inner.topos();
        serde_json::json!({
            "curvature": topos.curvature(), "saturation": topos.saturation(),
            "porosity": topos.porosity(), "max_values": topos.max_volume(),
        })
        .to_string()
    }

    fn forward(
        &self,
        py: Python<'_>,
        input: Vec<f32>,
        gate: Vec<f32>,
        bias: Vec<f32>,
        rows: usize,
        features: usize,
    ) -> PyResult<PyWaveGateLearningBatch> {
        py.detach(|| self.inner.forward(&input, &gate, &bias, rows, features))
            .map(|inner| PyWaveGateLearningBatch { inner })
            .map_err(value_error)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_with_log_radius(
        &self,
        py: Python<'_>,
        input: Vec<f32>,
        gate: Vec<f32>,
        bias: Vec<f32>,
        rows: usize,
        features: usize,
        log_radius: f32,
    ) -> PyResult<PyWaveGateLearningBatch> {
        py.detach(|| {
            self.inner
                .forward_with_log_radius(&input, &gate, &bias, rows, features, log_radius)
        })
        .map(|inner| PyWaveGateLearningBatch { inner })
        .map_err(value_error)
    }

    /// Owned native-f32 copy; no foreign writable storage survives this call.
    fn forward_buffer(
        &self,
        py: Python<'_>,
        input: &Bound<'_, PyAny>,
        gate: &Bound<'_, PyAny>,
        bias: &Bound<'_, PyAny>,
        rows: usize,
        features: usize,
    ) -> PyResult<PyWaveGateLearningBatch> {
        let input = read_f32(input, self.max_values(), None)?;
        let gate = read_f32(gate, self.max_values(), Some(features))?;
        let bias = read_f32(bias, self.max_values(), Some(features))?;
        self.forward(py, input, gate, bias, rows, features)
    }

    #[allow(clippy::too_many_arguments)] // Same controls as the sequence transport.
    fn forward_with_log_radius_buffer(
        &self,
        py: Python<'_>,
        input: &Bound<'_, PyAny>,
        gate: &Bound<'_, PyAny>,
        bias: &Bound<'_, PyAny>,
        rows: usize,
        features: usize,
        log_radius: f32,
    ) -> PyResult<PyWaveGateLearningBatch> {
        let input = read_f32(input, self.max_values(), None)?;
        let gate = read_f32(gate, self.max_values(), Some(features))?;
        let bias = read_f32(bias, self.max_values(), Some(features))?;
        self.forward_with_log_radius(py, input, gate, bias, rows, features, log_radius)
    }
}

#[pymethods]
impl PyWaveGateLearningBatch {
    #[getter]
    fn output(&self) -> Vec<f32> {
        self.inner.output().data().to_vec()
    }

    fn output_buffer<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyByteArray>> {
        write_f32(py, self.inner.output().data().iter().copied())
    }

    fn conditioning_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| serde_json::to_string(&self.inner.conditioning()))
            .map_err(value_error)
    }

    fn vjp(&self, py: Python<'_>, upstream: Vec<f32>) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        py.detach(|| self.inner.vjp(&upstream))
            .map(|vjp| {
                (
                    vjp.grad_input.data().to_vec(),
                    vjp.grad_gate.data().to_vec(),
                    vjp.grad_bias.data().to_vec(),
                )
            })
            .map_err(value_error)
    }

    fn vjp_with_log_radius(
        &self,
        py: Python<'_>,
        upstream: Vec<f32>,
    ) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>, f32)> {
        py.detach(|| self.inner.vjp_with_log_radius(&upstream))
            .map(|(vjp, radius)| {
                (
                    vjp.grad_input.data().to_vec(),
                    vjp.grad_gate.data().to_vec(),
                    vjp.grad_bias.data().to_vec(),
                    radius,
                )
            })
            .map_err(value_error)
    }

    fn vjp_buffer<'py>(
        &self,
        py: Python<'py>,
        upstream: &Bound<'_, PyAny>,
    ) -> PyResult<BufferVjp<'py>> {
        let count = self.inner.output().data().len();
        let upstream = read_f32(upstream, count, Some(count))?;
        let vjp = py
            .detach(|| self.inner.vjp(&upstream))
            .map_err(value_error)?;
        Ok((
            write_f32(py, vjp.grad_input.data().iter().copied())?,
            write_f32(py, vjp.grad_gate.data().iter().copied())?,
            write_f32(py, vjp.grad_bias.data().iter().copied())?,
        ))
    }

    #[allow(clippy::type_complexity)] // Flat tuple matches the existing scalar-radius VJP.
    fn vjp_with_log_radius_buffer<'py>(
        &self,
        py: Python<'py>,
        upstream: &Bound<'_, PyAny>,
    ) -> PyResult<(
        Bound<'py, PyByteArray>,
        Bound<'py, PyByteArray>,
        Bound<'py, PyByteArray>,
        f32,
    )> {
        let count = self.inner.output().data().len();
        let upstream = read_f32(upstream, count, Some(count))?;
        let (vjp, radius) = py
            .detach(|| self.inner.vjp_with_log_radius(&upstream))
            .map_err(value_error)?;
        Ok((
            write_f32(py, vjp.grad_input.data().iter().copied())?,
            write_f32(py, vjp.grad_gate.data().iter().copied())?,
            write_f32(py, vjp.grad_bias.data().iter().copied())?,
            radius,
        ))
    }
}

pub(crate) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_class::<PyWaveGateKernel>()?;
    parent.add_class::<PyWaveGateLearningBatch>()
}
