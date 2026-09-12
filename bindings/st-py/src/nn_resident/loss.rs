//! Thin transport of the ordinary Rust Loss's resident value/cotangent pair.
use super::*;
#[cfg(feature = "wgpu")]
use crate::wgpu_tensor::PyWgpuTensor;

#[pyclass(name = "ResidentLoss", module = "spiraltorch.nn")]
pub(crate) struct PyResidentLoss {
    #[cfg(feature = "wgpu")]
    inner: st_backend_wgpu::resident_tensor::loss::ResidentLoss,
}

#[cfg(feature = "wgpu")]
#[pymethods]
impl PyResidentLoss {
    fn loss_tensor(&self) -> PyWgpuTensor {
        PyWgpuTensor {
            inner: self.inner.value().clone(),
        }
    }
    fn prediction_gradient_tensor(&self) -> PyWgpuTensor {
        PyWgpuTensor {
            inner: self.inner.prediction_gradient().clone(),
        }
    }
}

pub(crate) fn evaluate_loss(
    loss: &mut dyn st_nn::Loss,
    prediction: &Bound<'_, PyAny>,
    target: &Bound<'_, PyAny>,
) -> PyResult<PyResidentLoss> {
    #[cfg(feature = "wgpu")]
    {
        let message =
            || PyTypeError::new_err("expected two WgpuTensor inputs; no implicit device transfer");
        let prediction = prediction
            .extract::<PyRef<'_, PyWgpuTensor>>()
            .map_err(|_| message())?;
        let target = target
            .extract::<PyRef<'_, PyWgpuTensor>>()
            .map_err(|_| message())?;
        Ok(PyResidentLoss {
            inner: loss
                .evaluate_resident(&prediction.inner, &target.inner)
                .map_err(plan_error)?,
        })
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let _ = (loss, prediction, target);
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "resident losses require the wgpu feature",
        ))
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyResidentLoss>()?;
    Ok(())
}
