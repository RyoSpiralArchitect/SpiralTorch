//! Thin clients of the same bounded study runner used by native and WASM.
use super::byte_decoder::{PyByteCorpusModelArtifact, PyByteDecoderCheckpoint};
use super::*;
use pyo3::types::{PyBytes, PyString};
#[cfg(feature = "wgpu")]
use st_nn::resident::ByteCorpusStudySegment;
use st_nn::resident::{ByteCorpusStudy, ByteCorpusStudyCheckpoint, BYTE_CORPUS_STUDY_MAX_BYTES};
use std::sync::Arc;

mod preparation;

fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

pub(super) fn json_input(value: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    use std::borrow::Cow;
    let bytes = if let Ok(value) = value.cast::<PyBytes>() {
        Cow::Borrowed(value.as_bytes())
    } else if let Ok(value) = value.cast::<PyString>() {
        if value.len()? > BYTE_CORPUS_STUDY_MAX_BYTES {
            return Err(PyValueError::new_err("study JSON exceeds 64 MiB"));
        }
        // abi3-py38 cannot borrow every Python string. Preserve strict UTF-8,
        // including an error for lone surrogates, without lossy conversion.
        match value.to_cow()? {
            Cow::Borrowed(text) => Cow::Borrowed(text.as_bytes()),
            Cow::Owned(text) => Cow::Owned(text.into_bytes()),
        }
    } else {
        return Err(PyTypeError::new_err(
            "expected exact JSON str or bytes, not a decoded object",
        ));
    };
    if bytes.len() > BYTE_CORPUS_STUDY_MAX_BYTES {
        return Err(PyValueError::new_err("study JSON exceeds 64 MiB"));
    }
    Ok(bytes.into_owned())
}

fn index(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    if value.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "study indices must be integers, not bool",
        ));
    }
    value.extract()
}

#[cfg(not(feature = "wgpu"))]
fn requires_wgpu<T>() -> PyResult<T> {
    Err(pyo3::exceptions::PyNotImplementedError::new_err(
        "study execution and measured preparation require the wgpu feature",
    ))
}

#[cfg(feature = "wgpu")]
fn on_gpu<T: Send>(
    py: Python<'_>,
    run: impl FnOnce(st_backend_wgpu::runtime::WgpuRuntime) -> PyResult<T> + Send,
) -> PyResult<T> {
    py.detach(move || {
        let (runtime, _) = st_backend_wgpu::runtime::ensure_default_runtime_blocking(
            "python.nn.byte_corpus_study",
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        run(runtime)
    })
}

#[pyclass(name = "ByteCorpusStudy", module = "spiraltorch.nn", frozen)]
pub(super) struct PyByteCorpusStudy {
    inner: Arc<ByteCorpusStudy>,
}

#[pymethods]
impl PyByteCorpusStudy {
    /// Preserve exact request bytes; no implicit JSON reconstruction or device.
    #[staticmethod]
    fn from_json(py: Python<'_>, payload: &Bound<'_, PyAny>) -> PyResult<Self> {
        let input = json_input(payload)?;
        py.detach(move || {
            Ok(Self {
                inner: Arc::new(ByteCorpusStudy::from_json(&input).map_err(value_error)?),
            })
        })
    }

    #[getter]
    fn total_updates(&self) -> usize {
        self.inner.total_updates()
    }

    #[getter]
    fn request_sha256(&self) -> &str {
        self.inner.request_sha256()
    }

    fn checkpoint_size_bound(&self, py: Python<'_>) -> PyResult<usize> {
        py.detach(|| self.inner.checkpoint_size_bound().map_err(value_error))
    }

    fn checkpoint_from_json(
        &self,
        py: Python<'_>,
        payload: &Bound<'_, PyAny>,
    ) -> PyResult<PyByteCorpusStudyCheckpoint> {
        let input = json_input(payload)?;
        py.detach(|| {
            Ok(PyByteCorpusStudyCheckpoint {
                inner: Arc::new(
                    self.inner
                        .checkpoint_from_json(&input)
                        .map_err(value_error)?,
                ),
            })
        })
    }

    /// CPU-only, all-case validation followed by an exact-name model handoff.
    /// Retain the study request/checkpoint for provenance and frozen-rate resume.
    /// Extraction does not promote the model or attest its training history.
    fn case_checkpoint(
        &self,
        py: Python<'_>,
        checkpoint: &PyByteCorpusStudyCheckpoint,
        case_name: &str,
    ) -> PyResult<PyByteDecoderCheckpoint> {
        py.detach(|| {
            self.inner
                .case_checkpoint(&checkpoint.inner, case_name)
                .map(PyByteDecoderCheckpoint::from)
                .map_err(value_error)
        })
    }

    #[pyo3(signature = (checkpoint, case_name, *, policy))]
    fn case_artifact(
        &self,
        py: Python<'_>,
        checkpoint: &PyByteCorpusStudyCheckpoint,
        case_name: &str,
        policy: &str,
    ) -> PyResult<PyByteCorpusModelArtifact> {
        let policy = serde_json::from_value(serde_json::Value::String(policy.to_owned()))
            .map_err(value_error)?;
        py.detach(|| {
            self.inner
                .case_artifact(&checkpoint.inner, case_name, policy)
                .map(PyByteCorpusModelArtifact::from)
                .map_err(value_error)
        })
    }

    fn validate_artifact(
        &self,
        py: Python<'_>,
        checkpoint: &PyByteCorpusStudyCheckpoint,
        artifact: &PyByteCorpusModelArtifact,
    ) -> PyResult<()> {
        py.detach(|| {
            self.inner
                .validate_artifact(&checkpoint.inner, artifact.as_inner())
                .map_err(value_error)
        })
    }

    /// CPU-only request, cursor, history and checkpoint ownership validation.
    #[pyo3(signature = (stop_after=None, *, resume=None))]
    fn validate_segment(
        &self,
        py: Python<'_>,
        stop_after: Option<&Bound<'_, PyAny>>,
        resume: Option<&PyByteCorpusStudyCheckpoint>,
    ) -> PyResult<()> {
        let stop = stop_after
            .map(index)
            .transpose()?
            .unwrap_or(self.inner.total_updates());
        py.detach(|| {
            self.inner
                .validate_segment(resume.map(|c| c.inner.as_ref()), stop)
                .map_err(value_error)
        })
    }

    /// Uninterrupted execution without checkpoint capture; returns Rust JSON.
    fn run_wgpu(&self, py: Python<'_>) -> PyResult<String> {
        #[cfg(feature = "wgpu")]
        {
            on_gpu(py, |runtime| {
                let report = pollster::block_on(self.inner.run(runtime))
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                serde_json::to_string(&report).map_err(value_error)
            })
        }
        #[cfg(not(feature = "wgpu"))]
        {
            let _ = py;
            requires_wgpu()
        }
    }

    /// Advance every case atomically to one absolute accepted-update boundary.
    #[pyo3(signature = (stop_after=None, *, resume=None))]
    fn advance_wgpu(
        &self,
        py: Python<'_>,
        stop_after: Option<&Bound<'_, PyAny>>,
        resume: Option<&PyByteCorpusStudyCheckpoint>,
    ) -> PyResult<PyByteCorpusStudySegment> {
        let stop = stop_after
            .map(index)
            .transpose()?
            .unwrap_or(self.inner.total_updates());
        let checkpoint = resume.map(|c| c.inner.as_ref());
        // A bad request/checkpoint must fail before acquiring a runtime.
        py.detach(|| {
            self.inner
                .validate_segment(checkpoint, stop)
                .map_err(value_error)
        })?;
        #[cfg(feature = "wgpu")]
        {
            on_gpu(py, |runtime| {
                let segment = pollster::block_on(self.inner.advance(runtime, checkpoint, stop))
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                PyByteCorpusStudySegment::from_segment(segment)
            })
        }
        #[cfg(not(feature = "wgpu"))]
        {
            requires_wgpu()
        }
    }
}

#[derive(Clone)]
#[pyclass(
    name = "ByteCorpusStudyCheckpoint",
    module = "spiraltorch.nn",
    frozen,
    skip_from_py_object
)]
pub(super) struct PyByteCorpusStudyCheckpoint {
    inner: Arc<ByteCorpusStudyCheckpoint>,
}

#[pymethods]
impl PyByteCorpusStudyCheckpoint {
    #[getter]
    fn completed_updates(&self) -> usize {
        self.inner.completed_updates()
    }

    #[getter]
    fn request_sha256(&self) -> &str {
        self.inner.request_sha256()
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.inner.to_json().map_err(value_error))
    }
}

#[pyclass(name = "ByteCorpusStudySegment", module = "spiraltorch.nn", frozen)]
pub(super) struct PyByteCorpusStudySegment {
    report_json: String,
    checkpoint: PyByteCorpusStudyCheckpoint,
}

impl PyByteCorpusStudySegment {
    #[cfg(feature = "wgpu")]
    fn from_segment(segment: ByteCorpusStudySegment) -> PyResult<Self> {
        Ok(Self {
            report_json: serde_json::to_string(&segment.report).map_err(value_error)?,
            checkpoint: PyByteCorpusStudyCheckpoint {
                inner: Arc::new(segment.checkpoint),
            },
        })
    }
}

#[pymethods]
impl PyByteCorpusStudySegment {
    #[getter]
    fn report_json(&self) -> &str {
        &self.report_json
    }

    #[getter]
    fn checkpoint(&self) -> PyByteCorpusStudyCheckpoint {
        self.checkpoint.clone()
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyByteCorpusStudy>()?;
    module.add_class::<PyByteCorpusStudyCheckpoint>()?;
    module.add_class::<PyByteCorpusStudySegment>()?;
    module.add("BYTE_CORPUS_STUDY_MAX_BYTES", BYTE_CORPUS_STUDY_MAX_BYTES)?;
    preparation::register(module)
}
