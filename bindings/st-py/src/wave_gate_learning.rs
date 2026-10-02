use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use st_nn::{WaveGateKernel, WaveGateLearningBatch};

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
}

#[pymethods]
impl PyWaveGateLearningBatch {
    #[getter]
    fn output(&self) -> Vec<f32> {
        self.inner.output().data().to_vec()
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
}

pub(crate) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_class::<PyWaveGateKernel>()?;
    parent.add_class::<PyWaveGateLearningBatch>()
}
