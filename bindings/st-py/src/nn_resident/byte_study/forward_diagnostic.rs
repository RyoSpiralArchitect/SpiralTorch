use super::*;
#[cfg(feature = "wgpu")]
use std::sync::Mutex;

/// Immutable observed outcome and a separately consumable graph diagnostic.
#[pyclass(
    name = "ByteCorpusForwardDiagnosticObservation",
    module = "spiraltorch.nn",
    frozen
)]
pub(super) struct PyByteCorpusForwardDiagnosticObservation {
    run_outcome_json: String,
    report_json: Option<String>,
    #[cfg(feature = "wgpu")]
    diagnostic: Mutex<Option<st_nn::resident::ByteDecoderGraphDiagnosticReadback>>,
}

impl PyByteCorpusForwardDiagnosticObservation {
    #[cfg(feature = "wgpu")]
    pub(super) fn from_observation(
        observation: st_nn::resident::ByteCorpusForwardDiagnosticObservation,
    ) -> PyResult<Self> {
        Ok(Self {
            report_json: observation
                .run_outcome
                .get("report")
                .map(serde_json::to_string)
                .transpose()
                .map_err(value_error)?,
            run_outcome_json: serde_json::to_string(&observation.run_outcome)
                .map_err(value_error)?,
            diagnostic: Mutex::new(observation.diagnostic),
        })
    }
}

#[pymethods]
impl PyByteCorpusForwardDiagnosticObservation {
    /// Original instrumented run outcome, unchanged by diagnostic read errors.
    #[getter]
    fn run_outcome_json(&self) -> &str {
        &self.run_outcome_json
    }

    /// Completed report serialized by Rust, or None for a failed observed run.
    #[getter]
    fn report_json(&self) -> Option<&str> {
        self.report_json.as_deref()
    }

    /// Consume once, including on map or encoding failure. Save the outcome first.
    /// Rust emits raw u32 bits, with a 256 MiB encoded limit, not a peak-memory cap.
    fn read_diagnostic_json(&self, py: Python<'_>) -> PyResult<String> {
        #[cfg(feature = "wgpu")]
        {
            py.detach(|| {
                let readback = {
                    self.diagnostic
                        .lock()
                        .map_err(|_| PyRuntimeError::new_err("diagnostic lock poisoned"))?
                        .take()
                        .ok_or_else(|| {
                            PyRuntimeError::new_err("diagnostic unavailable or already consumed")
                        })?
                };
                readback
                    .read()
                    .and_then(|diagnostic| diagnostic.to_json())
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
