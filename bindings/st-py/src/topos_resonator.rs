use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use st_core::dynamics::topos_resonator::{ToposResonatorConfig, ToposResonatorOperator};
use st_tensor::topos::OpenCartesianTopos;

fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// Explicit f32 CPU execution; GPU tensors are transported by the client.
#[pyclass(name = "ToposResonatorKernel", module = "spiraltorch", frozen)]
pub struct PyToposResonatorKernel {
    operator: ToposResonatorOperator,
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
}

pub(crate) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_class::<PyToposResonatorKernel>()
}
