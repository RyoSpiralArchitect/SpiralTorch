use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBool;
use st_nn::geometry_artifact::GeometryAdapterArtifact;
use st_nn::language_windows::{CausalDocumentWindows, MAX_DOCUMENTS};

fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn window_integer(value: &Bound<'_, PyAny>) -> PyResult<u32> {
    let boolean_dtype = value
        .getattr("dtype")
        .and_then(|dtype| dtype.str()?.extract::<String>())
        .is_ok_and(|dtype| dtype == "bool" || dtype == "torch.bool");
    if value.is_instance_of::<PyBool>() || boolean_dtype {
        return Err(PyTypeError::new_err("window integers cannot be booleans"));
    }
    value.extract()
}

#[pyclass(name = "GeometryAdapterArtifact", module = "spiraltorch", frozen)]
pub struct PyGeometryAdapterArtifact {
    inner: GeometryAdapterArtifact,
}

#[pymethods]
impl PyGeometryAdapterArtifact {
    /// Construct a newly validated artifact, not a trusted external load.
    #[staticmethod]
    fn create_json(payload: &str) -> PyResult<Self> {
        GeometryAdapterArtifact::create_json(payload.as_bytes())
            .map(|inner| Self { inner })
            .map_err(value_error)
    }

    #[staticmethod]
    fn from_json(
        payload: &str,
        expected_id: &str,
        expected_base_model_identity: &str,
    ) -> PyResult<Self> {
        GeometryAdapterArtifact::from_json(
            payload.as_bytes(),
            expected_id,
            expected_base_model_identity,
        )
        .map(|inner| Self { inner })
        .map_err(value_error)
    }

    fn to_json(&self) -> &str {
        self.inner.to_json()
    }

    fn state_json(&self) -> PyResult<String> {
        self.inner.state_json().map_err(value_error)
    }

    #[getter]
    fn artifact_id(&self) -> &str {
        self.inner.artifact_id()
    }

    #[getter]
    fn layout_id(&self) -> PyResult<String> {
        self.inner.layout_id().map_err(value_error)
    }

    #[getter]
    fn base_model_identity(&self) -> &str {
        &self.inner.record().base_model_identity
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyGeometryAdapterArtifact>()?;
    module.add_class::<PyCausalDocumentWindows>()
}

#[pyclass(name = "CausalDocumentWindows", module = "spiraltorch", frozen)]
pub struct PyCausalDocumentWindows {
    inner: CausalDocumentWindows,
}

#[pymethods]
impl PyCausalDocumentWindows {
    #[classattr]
    const MAX_DOCUMENTS: usize = MAX_DOCUMENTS;

    #[new]
    fn new(
        document_lengths: &Bound<'_, PyAny>,
        context_length: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let context_length = window_integer(context_length)?;
        let count = document_lengths.len()?;
        if count > MAX_DOCUMENTS {
            return Err(value_error("document count exceeds budget"));
        }
        let mut lengths = Vec::new();
        lengths.try_reserve_exact(count).map_err(value_error)?;
        // Index only the checked count; do not trust an unbounded __iter__.
        for index in 0..count {
            lengths.push(window_integer(&document_lengths.get_item(index)?)?);
        }
        CausalDocumentWindows::from_lengths(lengths, context_length)
            .map(|inner| Self { inner })
            .map_err(value_error)
    }

    #[staticmethod]
    fn create_json(payload: &str) -> PyResult<Self> {
        CausalDocumentWindows::create_json(payload.as_bytes())
            .map(|inner| Self { inner })
            .map_err(value_error)
    }

    fn __len__(&self) -> usize {
        self.inner.windows().len()
    }

    fn window(&self, index: &Bound<'_, PyAny>) -> PyResult<(u32, u32, u32)> {
        let index = window_integer(index)? as usize;
        self.inner
            .windows()
            .get(index)
            .map(|w| (w.document, w.start, w.length))
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("window index out of range"))
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(value_error)
    }

    #[getter]
    fn plan_id(&self) -> PyResult<String> {
        self.inner.plan_id().map_err(value_error)
    }
}
