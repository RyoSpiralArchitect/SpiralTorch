//! Python owns handles; evaluation, admission and owner swaps stay in Rust.
use super::*;
use st_nn::resident::ByteCorpusServingGate;
#[cfg(feature = "wgpu")]
use {
    crate::wgpu_tensor::PyWgpuTensorDevice,
    st_nn::resident::{ResidentByteCandidate, ResidentByteServing},
};

fn serving_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pyclass(name = "ByteCorpusServingGate", module = "spiraltorch.nn", frozen)]
struct PyByteCorpusServingGate {
    inner: ByteCorpusServingGate,
}

#[pymethods]
impl PyByteCorpusServingGate {
    #[staticmethod]
    fn from_json(py: Python<'_>, payload: &Bound<'_, PyAny>) -> PyResult<Self> {
        let input = super::super::byte_study::json_input(payload)?;
        py.detach(|| {
            ByteCorpusServingGate::from_json(&input)
                .map(|inner| Self { inner })
                .map_err(serving_error)
        })
    }
    fn to_json(&self) -> &str {
        self.inner.to_json()
    }
    #[getter]
    fn request_sha256(&self) -> &str {
        self.inner.request_sha256()
    }
    #[getter]
    fn batch_count(&self) -> usize {
        self.inner.batch_count()
    }
    #[getter]
    fn schema_version(&self) -> u8 {
        self.inner.schema_version()
    }
    #[getter]
    fn cohort_names(&self) -> Vec<&str> {
        self.inner.cohort_names()
    }
    #[getter]
    fn batch_shape(&self) -> (usize, usize) {
        let [batch, steps] = self.inner.batch_shape();
        (batch, steps)
    }
    fn validate_artifact(
        &self,
        py: Python<'_>,
        artifact: &PyByteCorpusModelArtifact,
    ) -> PyResult<()> {
        py.detach(|| {
            self.inner
                .validate_artifact(artifact.as_inner())
                .map_err(serving_error)
        })
    }
    /// Measure absolute gates before making the first owner active.
    fn bootstrap_wgpu(
        &self,
        py: Python<'_>,
        artifact: &PyByteCorpusModelArtifact,
    ) -> PyResult<PyResidentByteServing> {
        self.inner
            .validate_artifact(artifact.as_inner())
            .map_err(serving_error)?;
        #[cfg(feature = "wgpu")]
        {
            py.detach(|| {
                let (runtime, _) = st_backend_wgpu::runtime::ensure_default_runtime_blocking(
                    "python.nn.byte_serving",
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                pollster::block_on(ResidentByteServing::bootstrap(
                    runtime,
                    artifact.as_inner().clone(),
                    self.inner.clone(),
                ))
                .map(|inner| PyResidentByteServing { inner })
                .map_err(serving_error)
            })
        }
        #[cfg(not(feature = "wgpu"))]
        {
            let _ = py;
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "serving requires the wgpu feature",
            ))
        }
    }
}

#[pyclass(name = "ResidentByteCandidate", module = "spiraltorch.nn")]
struct PyResidentByteCandidate {
    #[cfg(feature = "wgpu")]
    inner: Option<ResidentByteCandidate>,
    #[cfg(feature = "wgpu")]
    accepted: bool,
    #[cfg(feature = "wgpu")]
    artifact_id: String,
    #[cfg(feature = "wgpu")]
    report: String,
}

#[cfg(feature = "wgpu")]
#[pymethods]
impl PyResidentByteCandidate {
    #[getter]
    fn accepted(&self) -> bool {
        self.accepted
    }
    #[getter]
    fn artifact_id(&self) -> &str {
        &self.artifact_id
    }
    #[getter]
    fn consumed(&self) -> bool {
        self.inner.is_none()
    }
    /// An immutable observation, not an importable activation permit.
    fn to_json(&self) -> &str {
        &self.report
    }
}

#[pyclass(name = "ResidentByteServing", module = "spiraltorch.nn")]
struct PyResidentByteServing {
    #[cfg(feature = "wgpu")]
    inner: ResidentByteServing,
}

#[cfg(feature = "wgpu")]
#[pymethods]
impl PyResidentByteServing {
    #[getter]
    fn artifact_id(&self) -> &str {
        self.inner.artifact_id()
    }
    #[getter]
    fn generation(&self) -> u64 {
        self.inner.generation()
    }
    #[getter]
    fn gate_sha256(&self) -> &str {
        self.inner.gate_sha256()
    }
    #[getter]
    fn previous_artifact_id(&self) -> Option<&str> {
        self.inner.previous_artifact_id()
    }
    fn bootstrap_json(&self) -> PyResult<String> {
        self.inner.bootstrap_json().map_err(serving_error)
    }
    fn tensor_device(&self) -> PyWgpuTensorDevice {
        PyWgpuTensorDevice {
            inner: self.inner.tensor_device().clone(),
        }
    }
    fn prepare_batch(
        &self,
        py: Python<'_>,
        batch: &PyByteLmBatch,
    ) -> PyResult<PyResidentByteBatch> {
        py.detach(|| {
            self.inner
                .prepare_batch(&batch.inner)
                .map(PyResidentByteBatch::from)
                .map_err(serving_error)
        })
    }
    fn forward(
        &mut self,
        py: Python<'_>,
        batch: &PyResidentByteBatch,
    ) -> PyResult<PyByteDecoderForward> {
        py.detach(|| {
            self.inner
                .forward(batch.as_inner())
                .map(PyByteDecoderForward::from)
                .map_err(serving_error)
        })
    }
    fn prepare_candidate(
        &self,
        py: Python<'_>,
        artifact: &PyByteCorpusModelArtifact,
    ) -> PyResult<PyResidentByteCandidate> {
        py.detach(|| {
            let inner =
                pollster::block_on(self.inner.prepare_candidate(artifact.as_inner().clone()))
                    .map_err(serving_error)?;
            Ok(PyResidentByteCandidate {
                accepted: inner.accepted(),
                artifact_id: inner.artifact_id().to_owned(),
                report: inner.to_json().map_err(serving_error)?,
                inner: Some(inner),
            })
        })
    }
    /// One activation attempt consumes the permit, including a rejected attempt.
    fn activate(
        &mut self,
        py: Python<'_>,
        candidate: &mut PyResidentByteCandidate,
    ) -> PyResult<String> {
        let candidate = candidate
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("candidate permit already consumed"))?;
        py.detach(|| {
            let event = self.inner.activate(candidate).map_err(serving_error)?;
            serde_json::to_string(&event).map_err(serving_error)
        })
    }
    fn rollback(
        &mut self,
        py: Python<'_>,
        expected_generation: &Bound<'_, PyAny>,
    ) -> PyResult<String> {
        if expected_generation.is_instance_of::<pyo3::types::PyBool>() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "generation must be an integer, not bool",
            ));
        }
        let generation = expected_generation.extract::<u64>()?;
        py.detach(|| {
            let event = self.inner.rollback(generation).map_err(serving_error)?;
            serde_json::to_string(&event).map_err(serving_error)
        })
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyByteCorpusServingGate>()?;
    module.add_class::<PyResidentByteCandidate>()?;
    module.add_class::<PyResidentByteServing>()?;
    Ok(())
}
