//! Generic acceptance handle over the existing Rust parameter transaction.
use super::*;
#[cfg(feature = "wgpu")]
use st_backend_wgpu::resident_training::parameters::ResidentParameterUpdate;

#[pyclass(name = "ResidentParameterUpdate", module = "spiraltorch.nn", frozen)]
pub(super) struct PyResidentParameterUpdate {
    #[cfg(feature = "wgpu")]
    pub(super) inner: ResidentParameterUpdate,
}

#[cfg(feature = "wgpu")]
#[pymethods]
impl PyResidentParameterUpdate {
    #[getter]
    fn attempted_revision(&self) -> u64 {
        self.inner.revision()
    }
    /// Explicit flag readback. A submitted update is not proof of acceptance.
    fn read(&self, py: Python<'_>) -> PyResult<u64> {
        py.detach(|| self.inner.snapshot().and_then(|snapshot| snapshot.read()))
            .map_err(training::training_error)
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyResidentParameterUpdate>()
}
