//! Thin attention bindings. Shapes, masking and derivatives belong to Rust.
use super::*;
#[cfg(feature = "wgpu")]
use st_backend_wgpu::resident_tensor::attention::{AttentionMask, ResidentAttentionGradients};

#[cfg(feature = "wgpu")]
pub(super) fn mask(offset: Option<&Bound<'_, PyAny>>) -> PyResult<AttentionMask> {
    offset.map(index).transpose().map(|value| {
        value.map_or(AttentionMask::None, |query_offset| AttentionMask::Causal {
            query_offset,
        })
    })
}

#[pyclass(name = "WgpuAttentionGradients", module = "spiraltorch.wgpu", frozen)]
pub(crate) struct PyAttentionGradients {
    #[cfg(feature = "wgpu")]
    pub(super) inner: ResidentAttentionGradients,
}

#[cfg(feature = "wgpu")]
#[pymethods]
impl PyAttentionGradients {
    #[getter]
    fn query(&self) -> PyWgpuTensor {
        PyWgpuTensor {
            inner: self.inner.query.clone(),
        }
    }
    #[getter]
    fn key(&self) -> PyWgpuTensor {
        PyWgpuTensor {
            inner: self.inner.key.clone(),
        }
    }
    #[getter]
    fn value(&self) -> PyWgpuTensor {
        PyWgpuTensor {
            inner: self.inner.value.clone(),
        }
    }
    #[getter]
    fn z_bias(&self) -> Option<PyWgpuTensor> {
        self.inner
            .z_bias
            .clone()
            .map(|inner| PyWgpuTensor { inner })
    }
    #[getter]
    fn pair_bias(&self) -> Option<PyWgpuTensor> {
        self.inner
            .pair_bias
            .clone()
            .map(|inner| PyWgpuTensor { inner })
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyAttentionGradients>()
}
