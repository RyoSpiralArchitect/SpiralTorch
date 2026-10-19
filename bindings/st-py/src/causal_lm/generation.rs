//! Bounded transport only: sampling, EOS, control and random continuation belong to Rust.
use super::{error, tokens, PyNativeCausalLm};
use pyo3::prelude::*;
use st_nn::causal_lm::{CausalLmGenerationConfig, CausalLmGenerationStep, NativeCausalLmGenerator};

#[pyclass(name = "NativeCausalLmGenerator", module = "spiraltorch")]
pub struct PyNativeCausalLmGenerator {
    inner: NativeCausalLmGenerator,
}

#[pyclass(name = "NativeCausalLmGenerationStep", module = "spiraltorch", frozen)]
pub struct PyNativeCausalLmGenerationStep {
    inner: CausalLmGenerationStep,
}

#[pymethods]
impl PyNativeCausalLmGenerator {
    #[new]
    fn new(
        py: Python<'_>,
        model: &PyNativeCausalLm,
        prompt_ids: &Bound<'_, PyAny>,
        config_json: &str,
    ) -> PyResult<Self> {
        let config = CausalLmGenerationConfig::from_json(config_json.as_bytes()).map_err(error)?;
        let length = prompt_ids.len()?;
        config.validate(&model.inner, length).map_err(error)?;
        let prompt = tokens(prompt_ids, length)?;
        py.detach(|| NativeCausalLmGenerator::new(model.inner.clone(), prompt, config))
            .map(|inner| Self { inner })
            .map_err(error)
    }
    #[staticmethod]
    fn from_json(
        py: Python<'_>,
        model: &PyNativeCausalLm,
        payload: &str,
        expected_id: &str,
    ) -> PyResult<Self> {
        py.detach(|| {
            NativeCausalLmGenerator::from_json(model.inner.clone(), payload.as_bytes(), expected_id)
        })
        .map(|inner| Self { inner })
        .map_err(error)
    }
    fn step(&mut self, py: Python<'_>) -> PyResult<Option<PyNativeCausalLmGenerationStep>> {
        py.detach(|| self.inner.step())
            .map(|report| report.map(|inner| PyNativeCausalLmGenerationStep { inner }))
            .map_err(error)
    }
    #[getter]
    fn finished(&self) -> bool {
        self.inner.finished()
    }
    #[getter]
    fn finish_reason(&self) -> Option<&'static str> {
        self.inner.finish_reason()
    }
    #[getter]
    fn token_ids(&self) -> Vec<usize> {
        self.inner.token_ids().to_vec()
    }
    #[getter]
    fn generated_ids(&self) -> Vec<usize> {
        self.inner.generated_ids().to_vec()
    }
    #[getter]
    fn checkpoint_id(&self) -> PyResult<String> {
        self.inner.checkpoint_id().map_err(error)
    }
    fn config_json(&self) -> PyResult<String> {
        serde_json::to_string(self.inner.config()).map_err(error)
    }
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(error)
    }
}

#[pymethods]
impl PyNativeCausalLmGenerationStep {
    #[getter]
    fn step(&self) -> usize {
        self.inner.step
    }
    #[getter]
    fn token_id(&self) -> usize {
        self.inner.decision.token_id
    }
    #[getter]
    fn probability(&self) -> f64 {
        self.inner.decision.probability
    }
    #[getter]
    fn entropy(&self) -> f64 {
        self.inner.decision.entropy
    }
    #[getter]
    fn finish_reason(&self) -> Option<&'static str> {
        self.inner.finish_reason
    }
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(error)
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyNativeCausalLmGenerator>()?;
    module.add_class::<PyNativeCausalLmGenerationStep>()?;
    Ok(())
}
