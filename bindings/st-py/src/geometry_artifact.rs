use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use st_nn::geometry_artifact::GeometryAdapterArtifact;

fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
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
    module.add_class::<PyGeometryAdapterArtifact>()
}
