//! Bounded transport only: sampling, EOS, control and random continuation belong to Rust.
use super::{checkpoint_error, dimension, error, session_input, tokens, PyNativeCausalLm};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString, PyTuple};
use st_nn::causal_lm::{CausalLmGenerationConfig, CausalLmGenerationStep, NativeCausalLmGenerator};
use std::path::PathBuf;

#[pyclass(name = "NativeCausalLmGenerator", module = "spiraltorch")]
pub struct PyNativeCausalLmGenerator {
    inner: NativeCausalLmGenerator,
}

impl PyNativeCausalLmGenerator {
    pub(super) fn from_native(inner: NativeCausalLmGenerator) -> Self {
        Self { inner }
    }
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
        config_json: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let bytes = session_input::payload_with_limit(
            config_json,
            CausalLmGenerationConfig::MAX_JSON_BYTES,
        )?;
        let config = CausalLmGenerationConfig::from_json(bytes.as_bytes()).map_err(error)?;
        let length = prompt_ids.len()?;
        config.validate(&model.inner, length).map_err(error)?;
        let prompt = tokens(prompt_ids, length)?;
        py.detach(|| NativeCausalLmGenerator::new(model.inner.clone(), prompt, config))
            .map(|inner| Self { inner })
            .map_err(error)
    }
    #[staticmethod]
    fn validate_prompt_length(
        model: &PyNativeCausalLm,
        prompt_length: &Bound<'_, PyAny>,
        config_json: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let length = dimension(prompt_length)?;
        let bytes = session_input::payload_with_limit(
            config_json,
            CausalLmGenerationConfig::MAX_JSON_BYTES,
        )?;
        let config = CausalLmGenerationConfig::from_json(bytes.as_bytes()).map_err(error)?;
        config.validate(&model.inner, length).map_err(error)
    }
    #[staticmethod]
    fn from_json(
        py: Python<'_>,
        model: &PyNativeCausalLm,
        payload: &Bound<'_, PyAny>,
        expected_id: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let pin = session_input::pin(expected_id)?;
        let bytes = session_input::payload_with_limit(
            payload,
            NativeCausalLmGenerator::MAX_CHECKPOINT_BYTES,
        )?;
        let bytes = bytes.as_bytes();
        py.detach(|| NativeCausalLmGenerator::from_json(model.inner.clone(), bytes, &pin))
            .map(|inner| Self { inner })
            .map_err(error)
    }
    fn step(&mut self, py: Python<'_>) -> PyResult<Option<PyNativeCausalLmGenerationStep>> {
        py.detach(|| self.inner.step())
            .map(|report| report.map(|inner| PyNativeCausalLmGenerationStep { inner }))
            .map_err(error)
    }
    fn step_token_id(&mut self, py: Python<'_>) -> PyResult<Option<usize>> {
        py.detach(|| self.inner.step_token_id()).map_err(error)
    }
    fn advance_ids(&mut self, py: Python<'_>, count: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
        let count = dimension(count)?;
        py.detach(|| self.inner.advance_ids(count)).map_err(error)
    }
    #[getter]
    fn projection_id(&self) -> Option<&str> {
        self.inner.projection_id()
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
    fn checkpoint_with_id(&self, py: Python<'_>) -> PyResult<(String, String)> {
        py.detach(|| self.inner.checkpoint_with_id()).map_err(error)
    }
    fn save_checkpoint_new<'py>(
        &self,
        py: Python<'py>,
        path: PathBuf,
    ) -> PyResult<Bound<'py, PyString>> {
        let staged = py
            .detach(|| self.inner.prepare_checkpoint_save(path))
            .map_err(checkpoint_error)?;
        let pin = PyString::new(py, staged.checkpoint_id());
        py.detach(|| staged.publish()).map_err(checkpoint_error)?;
        Ok(pin)
    }
    fn advance_to_checkpoint_new<'py>(
        &mut self,
        py: Python<'py>,
        stop_after: &Bound<'_, PyAny>,
        path: PathBuf,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let stop = dimension(stop_after)?;
        let staged = py
            .detach(|| self.inner.prepare_checkpoint_interval(stop, path))
            .map_err(checkpoint_error)?;
        let steps = staged
            .reports()
            .iter()
            .cloned()
            .map(|inner| Py::new(py, PyNativeCausalLmGenerationStep { inner }))
            .collect::<PyResult<Vec<_>>>()?;
        let steps = PyList::new(py, steps)?.into_any();
        let pin = PyString::new(py, staged.checkpoint_id()).into_any();
        let result = PyTuple::new(py, [steps, pin])?;
        py.detach(|| staged.publish()).map_err(checkpoint_error)?;
        Ok(result)
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
