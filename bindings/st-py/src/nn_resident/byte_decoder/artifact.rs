//! Hash-pinned, policy-constrained inference; all semantics remain in Rust.
use super::*;
#[cfg(feature = "wgpu")]
use crate::wgpu_tensor::PyWgpuTensorDevice;
use st_nn::resident::ByteCorpusModelArtifact;
#[cfg(feature = "wgpu")]
use st_nn::resident::ResidentByteInference;

#[pyclass(name = "ByteCorpusModelArtifact", module = "spiraltorch.nn", frozen)]
pub(in crate::nn_resident) struct PyByteCorpusModelArtifact {
    inner: ByteCorpusModelArtifact,
}

impl From<ByteCorpusModelArtifact> for PyByteCorpusModelArtifact {
    fn from(inner: ByteCorpusModelArtifact) -> Self {
        Self { inner }
    }
}

impl PyByteCorpusModelArtifact {
    pub(in crate::nn_resident) fn as_inner(&self) -> &ByteCorpusModelArtifact {
        &self.inner
    }
}

#[pymethods]
impl PyByteCorpusModelArtifact {
    #[staticmethod]
    #[pyo3(signature = (payload, *, expected_id))]
    fn from_json(py: Python<'_>, payload: &Bound<'_, PyAny>, expected_id: &str) -> PyResult<Self> {
        let input = super::super::byte_study::json_input(payload)?;
        py.detach(|| {
            ByteCorpusModelArtifact::from_json(&input, expected_id)
                .map(Self::from)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    fn to_json(&self) -> &str {
        self.inner.to_json()
    }
    #[getter]
    fn artifact_id(&self) -> &str {
        self.inner.artifact_id()
    }
    #[getter]
    fn policy(&self) -> &str {
        self.inner.policy().as_str()
    }
    #[getter]
    fn case_name(&self) -> &str {
        self.inner.case_name()
    }
    #[getter]
    fn request_sha256(&self) -> &str {
        self.inner.request_sha256()
    }
    #[getter]
    fn study_checkpoint_sha256(&self) -> &str {
        self.inner.study_checkpoint_sha256()
    }
    fn checkpoint(&self, py: Python<'_>) -> PyByteDecoderCheckpoint {
        py.detach(|| self.inner.checkpoint().clone().into())
    }
    fn restore_wgpu(&self, py: Python<'_>) -> PyResult<PyResidentByteInference> {
        #[cfg(feature = "wgpu")]
        {
            py.detach(|| {
                let (runtime, _) = st_backend_wgpu::runtime::ensure_default_runtime_blocking(
                    "python.nn.byte_inference",
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                Ok(PyResidentByteInference {
                    inner: self.inner.restore_wgpu(runtime).map_err(plan_error)?,
                })
            })
        }
        #[cfg(not(feature = "wgpu"))]
        {
            let _ = py;
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "artifact inference requires the wgpu feature",
            ))
        }
    }
}

#[pyclass(name = "ResidentByteInference", module = "spiraltorch.nn")]
struct PyResidentByteInference {
    #[cfg(feature = "wgpu")]
    inner: ResidentByteInference,
}

#[cfg(feature = "wgpu")]
#[pymethods]
impl PyResidentByteInference {
    #[getter]
    fn artifact_id(&self) -> &str {
        self.inner.artifact_id()
    }
    #[getter]
    fn attempted_updates(&self) -> u64 {
        self.inner.attempted_revision()
    }
    #[getter]
    fn input_shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.input_layout().shape().iter().copied())
    }
    #[getter]
    fn output_shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.output_layout().shape().iter().copied())
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
                .map_err(plan_error)
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
                .map_err(plan_error)
        })
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyByteCorpusModelArtifact>()?;
    module.add_class::<PyResidentByteInference>()?;
    Ok(())
}
