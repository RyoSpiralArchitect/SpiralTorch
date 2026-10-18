//! Thin owner transport; no key/value mutation or geometry semantics in Python.
use super::{dimension, error, limit, tokens, PyNativeCausalLm, PyNdTensor};
use pyo3::prelude::*;
use st_nn::causal_lm::NativeCausalLmContext;

#[pyclass(name = "NativeCausalLmContext", module = "spiraltorch")]
pub struct PyNativeCausalLmContext {
    inner: NativeCausalLmContext,
}

#[pymethods]
impl PyNativeCausalLmContext {
    #[getter]
    fn readout_accumulator(&self) -> &'static str {
        self.inner.readout_accumulator().as_str()
    }

    #[new]
    #[pyo3(signature = (model, prompt_ids, *, max_cache_values=None))]
    fn new(
        py: Python<'_>,
        model: &PyNativeCausalLm,
        prompt_ids: &Bound<'_, PyAny>,
        max_cache_values: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let budget = limit(
            max_cache_values,
            NativeCausalLmContext::DEFAULT_MAX_CACHE_VALUES,
        )?;
        let count = prompt_ids.len()?;
        model
            .inner
            .context_prefill_count(count, budget)
            .map_err(error)?;
        let ids = tokens(prompt_ids, count)?;
        py.detach(|| NativeCausalLmContext::new(model.inner.clone(), ids, budget))
            .map(|inner| Self { inner })
            .map_err(error)
    }
    fn append(&mut self, py: Python<'_>, token_id: &Bound<'_, PyAny>) -> PyResult<PyNdTensor> {
        let id = dimension(token_id)?;
        py.detach(|| self.inner.append(id))
            .map(|inner| PyNdTensor { inner })
            .map_err(error)
    }
    fn logits(&self, py: Python<'_>) -> PyResult<PyNdTensor> {
        py.detach(|| self.inner.logits())
            .map(|inner| PyNdTensor { inner })
            .map_err(error)
    }
    #[getter]
    fn token_ids(&self) -> Vec<usize> {
        self.inner.token_ids().to_vec()
    }
    #[getter]
    fn cache_values(&self) -> usize {
        self.inner.cache_values()
    }
    #[getter]
    fn max_cache_values(&self) -> usize {
        self.inner.max_cache_values()
    }
    #[getter]
    fn base_model_identity(&self) -> &str {
        self.inner.base_model_identity()
    }
    #[getter]
    fn geometry_artifact_id(&self) -> Option<&str> {
        self.inner.geometry_artifact_id()
    }
    #[getter]
    fn execution_backend(&self) -> &'static str {
        "rust_f32_cpu_prepacked_kv"
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyNativeCausalLmContext>()?;
    Ok(())
}
