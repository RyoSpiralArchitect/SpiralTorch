//! Transport only: all optimizer validation, arithmetic and ownership are Rust.
use super::{error, rate, PyGeometryAdapterArtifact};
use pyo3::prelude::*;
use st_nn::geometry_runtime::{AdamWConfig, AdamWStep, GlobalNormClip, NativeGeometryAdamW};

fn scalar(value: Option<&Bound<'_, PyAny>>, default: f32) -> PyResult<f32> {
    value.map(rate).transpose().map(|v| v.unwrap_or(default))
}

pub(crate) fn adamw_step(
    learning_rate: &Bound<'_, PyAny>,
    weight_decay: Option<&Bound<'_, PyAny>>,
) -> PyResult<AdamWStep> {
    AdamWStep::new(rate(learning_rate)?, scalar(weight_decay, 0.)?).map_err(error)
}

#[pyclass(name = "NativeGeometryAdamW", module = "spiraltorch", frozen)]
pub struct PyNativeGeometryAdamW {
    pub(crate) inner: NativeGeometryAdamW,
}

#[pymethods]
impl PyNativeGeometryAdamW {
    #[new]
    #[pyo3(signature = (artifact, *, beta1=None, beta2=None, epsilon=None, max_gradient_norm=None))]
    fn new(
        py: Python<'_>,
        artifact: &PyGeometryAdapterArtifact,
        beta1: Option<&Bound<'_, PyAny>>,
        beta2: Option<&Bound<'_, PyAny>>,
        epsilon: Option<&Bound<'_, PyAny>>,
        max_gradient_norm: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let defaults = AdamWConfig::default();
        let config = AdamWConfig::new(
            scalar(beta1, defaults.beta1())?,
            scalar(beta2, defaults.beta2())?,
            scalar(epsilon, defaults.epsilon())?,
        )
        .map_err(error)?;
        let clip = max_gradient_norm
            .map(rate)
            .transpose()?
            .map(GlobalNormClip::new)
            .transpose()
            .map_err(error)?;
        py.detach(|| NativeGeometryAdamW::new(&artifact.inner, config, clip))
            .map(|inner| Self { inner })
            .map_err(error)
    }

    #[staticmethod]
    fn from_json(
        py: Python<'_>,
        payload: &str,
        expected_id: &str,
        artifact: &PyGeometryAdapterArtifact,
    ) -> PyResult<Self> {
        py.detach(|| {
            NativeGeometryAdamW::from_json(payload.as_bytes(), expected_id, &artifact.inner)
        })
        .map(|inner| Self { inner })
        .map_err(error)
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.inner.to_json()).map_err(error)
    }

    #[getter]
    fn checkpoint_id(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.inner.checkpoint_id()).map_err(error)
    }

    #[getter]
    fn geometry_artifact_id(&self) -> &str {
        self.inner.geometry_artifact_id()
    }

    #[getter]
    fn revision(&self) -> u64 {
        self.inner.revision()
    }

    #[getter]
    fn max_gradient_norm(&self) -> Option<f32> {
        self.inner.max_gradient_norm()
    }
}
