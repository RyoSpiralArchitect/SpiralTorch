use super::*;
#[cfg(feature = "wgpu")]
use std::sync::Mutex;

/// Diagnostic result with independently consumable, one-shot GPU readback.
#[pyclass(
    name = "ByteCorpusBoundaryObservation",
    module = "spiraltorch.nn",
    frozen
)]
pub(super) struct PyByteCorpusBoundaryObservation {
    run_outcome_json: String,
    #[cfg(feature = "wgpu")]
    boundary: Mutex<Option<st_nn::resident::ByteDecoderCheckpointHandles>>,
}

impl PyByteCorpusBoundaryObservation {
    #[cfg(feature = "wgpu")]
    pub(super) fn from_observation(
        observation: st_nn::resident::ByteCorpusBoundaryObservation,
    ) -> PyResult<Self> {
        Ok(Self {
            run_outcome_json: serde_json::to_string(&observation.run_outcome)
                .map_err(value_error)?,
            boundary: Mutex::new(observation.boundary),
        })
    }
}

#[pymethods]
impl PyByteCorpusBoundaryObservation {
    /// Immutable original outcome, still available after a diagnostic read fails.
    #[getter]
    fn run_outcome_json(&self) -> &str {
        &self.run_outcome_json
    }

    /// Consume once, even on failure. Save run_outcome_json before calling this.
    fn read_boundary(&self, py: Python<'_>) -> PyResult<PyByteDecoderCheckpoint> {
        #[cfg(feature = "wgpu")]
        {
            py.detach(|| {
                let handles = self
                    .boundary
                    .lock()
                    .map_err(|_| PyRuntimeError::new_err("boundary lock poisoned"))?
                    .take()
                    .ok_or_else(|| {
                        PyRuntimeError::new_err("boundary unavailable or already consumed")
                    })?;
                handles
                    .snapshot()
                    .and_then(|capture| capture.read())
                    .map(PyByteDecoderCheckpoint::from)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })
        }
        #[cfg(not(feature = "wgpu"))]
        {
            let _ = py;
            requires_wgpu()
        }
    }
}
