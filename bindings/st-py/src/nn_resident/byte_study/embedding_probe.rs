use super::*;
use st_nn::resident::ByteEmbeddingProbe;

pub(super) fn select<'a>(
    study: &'a ByteCorpusStudy,
    py: Python<'_>,
    case: &str,
    update: &Bound<'_, PyAny>,
    token: &Bound<'_, PyAny>,
    feature: &Bound<'_, PyAny>,
) -> PyResult<ByteEmbeddingProbe<'a>> {
    let update = diagnostic_index(update)?;
    let token = u8::try_from(diagnostic_index(token)?)
        .map_err(|_| PyValueError::new_err("token must be in 0..=255"))?;
    let feature = diagnostic_index(feature)?;
    py.detach(|| {
        study
            .embedding_update_probe(case, update, token, feature)
            .map_err(value_error)
    })
}

#[pyclass(name = "ByteEmbeddingObservation", module = "spiraltorch.nn", frozen)]
pub(super) struct PyByteEmbeddingObservation {
    #[pyo3(get)]
    run_outcome_json: String,
    #[cfg(feature = "wgpu")]
    views: std::sync::Mutex<Option<st_nn::resident::ByteEmbeddingViews>>,
}

#[cfg(feature = "wgpu")]
impl PyByteEmbeddingObservation {
    pub(super) fn from_observation(
        result: st_nn::resident::ByteEmbeddingObservation,
    ) -> PyResult<Self> {
        Ok(Self {
            run_outcome_json: serde_json::to_string(&result.run_outcome).map_err(value_error)?,
            views: std::sync::Mutex::new(result.views),
        })
    }
}

#[pymethods]
impl PyByteEmbeddingObservation {
    /// Copy and map the four guarded scalar views once, even on read failure.
    fn read_bits_json(&self, py: Python<'_>) -> PyResult<String> {
        #[cfg(feature = "wgpu")]
        {
            py.detach(|| {
                let views = self
                    .views
                    .lock()
                    .map_err(|_| PyRuntimeError::new_err("embedding probe lock poisoned"))?
                    .take()
                    .ok_or_else(|| {
                        PyRuntimeError::new_err("embedding probe unavailable or already consumed")
                    })?;
                let bits = views
                    .read_bits()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                serde_json::to_string(&bits).map_err(value_error)
            })
        }
        #[cfg(not(feature = "wgpu"))]
        {
            let _ = py;
            requires_wgpu()
        }
    }
}
