use super::{dimension, error, PyNativeCausalLm};
use pyo3::prelude::*;
use st_nn::causal_lm::NativeCausalLmTrainingPrefixCache;

/// In-memory only. The budget counts retained f32 values; statistics contain no token IDs or activations.
#[pyclass(name = "NativeCausalLmTrainingPrefixCache", module = "spiraltorch")]
pub(super) struct PyTrainingPrefixCache {
    pub(super) inner: NativeCausalLmTrainingPrefixCache,
}

#[pymethods]
impl PyTrainingPrefixCache {
    #[classattr]
    const MAX_VALUES: usize = NativeCausalLmTrainingPrefixCache::MAX_VALUES;
    #[classattr]
    const MAX_ENTRIES: usize = NativeCausalLmTrainingPrefixCache::MAX_ENTRIES;

    #[new]
    #[pyo3(signature=(model, *, max_values, max_entries))]
    fn new(
        model: &PyNativeCausalLm,
        max_values: &Bound<'_, PyAny>,
        max_entries: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let inner = NativeCausalLmTrainingPrefixCache::new(
            &model.inner,
            dimension(max_values)?,
            dimension(max_entries)?,
        )
        .map_err(error)?;
        Ok(Self { inner })
    }

    fn stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::json::json_to_py(
            py,
            &serde_json::to_value(self.inner.stats()).map_err(error)?,
        )
    }

    fn clear(&mut self, py: Python<'_>) {
        py.detach(|| self.inner.clear());
    }
}
