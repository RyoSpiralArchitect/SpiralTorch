//! Ownership handles only: model, loss, differentiation and update rules stay in Rust.
use super::*;
use crate::wgpu_tensor::{PyWgpuTensor, PyWgpuTensorDevice};
use st_backend_wgpu::resident_training::parameters::ResidentParameterUpdate;
use st_vision::models::convnext::{
    ConvNeXtClassifier, ConvNeXtClassifierCheckpoint, ConvNeXtClassifierCheckpointSnapshot,
    ConvNeXtConfig, ResidentConvNeXtClassifier, ResidentConvNeXtForward, ResidentConvNeXtGradients,
};

fn error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pyclass(name = "ResidentConvNeXtClassifier", module = "spiraltorch.vision")]
pub(super) struct PyResidentClassifier {
    inner: ResidentConvNeXtClassifier,
}

#[pymethods]
impl PyResidentClassifier {
    #[staticmethod]
    fn default_config_json() -> PyResult<String> {
        serde_json::to_string(&ConvNeXtConfig::default()).map_err(error)
    }

    #[staticmethod]
    #[pyo3(signature = (device, config_json, num_classes, batch_size, seed=0))]
    fn create(
        py: Python<'_>,
        device: &PyWgpuTensorDevice,
        config_json: &str,
        num_classes: usize,
        batch_size: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let config: ConvNeXtConfig = serde_json::from_str(config_json).map_err(error)?;
        let inner = py.detach(|| {
            ConvNeXtClassifier::new(config, num_classes, seed)
                .map_err(error)?
                .compile_resident_training(device.inner.clone(), batch_size)
                .map_err(error)
        })?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn from_checkpoint_json(
        py: Python<'_>,
        device: &PyWgpuTensorDevice,
        payload: &str,
    ) -> PyResult<Self> {
        let inner = py.detach(|| {
            ConvNeXtClassifierCheckpoint::from_json(payload)
                .map_err(error)?
                .restore_resident(device.inner.clone())
                .map_err(error)
        })?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn host_from_checkpoint_json(py: Python<'_>, payload: &str) -> PyResult<PyVisionModel> {
        let inner = py.detach(|| {
            ConvNeXtClassifierCheckpoint::from_json(payload)
                .map_err(error)?
                .to_host()
                .map_err(error)?
                .into_vision_model()
                .map_err(error)
        })?;
        Ok(PyVisionModel::from_inner(inner))
    }

    #[getter]
    fn input_shape(&self) -> Vec<usize> {
        self.inner.input_shape().to_vec()
    }
    #[getter]
    fn output_shape(&self) -> Vec<usize> {
        self.inner.output_shape().to_vec()
    }
    #[getter]
    fn attempted_updates(&self) -> u64 {
        self.inner.parameter_snapshot().revision()
    }
    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names().to_vec()
    }
    fn parameter_tensors(&self) -> Vec<PyWgpuTensor> {
        self.inner
            .parameter_snapshot()
            .values()
            .iter()
            .cloned()
            .map(|inner| PyWgpuTensor { inner })
            .collect()
    }
    fn tensor_device(&self) -> PyWgpuTensorDevice {
        PyWgpuTensorDevice {
            inner: self.inner.tensor_device().clone(),
        }
    }
    fn forward(&mut self, py: Python<'_>, input: &PyWgpuTensor) -> PyResult<PyConvNeXtForward> {
        Ok(PyConvNeXtForward {
            inner: py
                .detach(|| self.inner.forward(&input.inner))
                .map_err(error)?,
        })
    }
    fn backward(
        &mut self,
        py: Python<'_>,
        forward: &PyConvNeXtForward,
        cotangent: &PyWgpuTensor,
    ) -> PyResult<PyConvNeXtGradients> {
        Ok(PyConvNeXtGradients {
            inner: py
                .detach(|| self.inner.backward(&forward.inner, &cotangent.inner))
                .map_err(error)?,
        })
    }
    fn sgd(
        &mut self,
        py: Python<'_>,
        gradients: &PyConvNeXtGradients,
        rate: f32,
    ) -> PyResult<PyConvNeXtUpdate> {
        Ok(PyConvNeXtUpdate {
            inner: py
                .detach(|| self.inner.sgd(&gradients.inner, rate))
                .map_err(error)?,
        })
    }
    fn checkpoint_snapshot(&self, py: Python<'_>) -> PyResult<PyConvNeXtCheckpointSnapshot> {
        Ok(PyConvNeXtCheckpointSnapshot {
            inner: Some(
                py.detach(|| self.inner.checkpoint_snapshot())
                    .map_err(error)?,
            ),
        })
    }
}

#[pyclass(name = "ConvNeXtForward", module = "spiraltorch.vision")]
pub(super) struct PyConvNeXtForward {
    inner: ResidentConvNeXtForward,
}
#[pymethods]
impl PyConvNeXtForward {
    #[getter]
    fn parameter_revision(&self) -> u64 {
        self.inner.parameter_revision()
    }
    fn prediction_tensor(&self) -> PyWgpuTensor {
        PyWgpuTensor {
            inner: self.inner.prediction().clone(),
        }
    }
}

#[pyclass(name = "ConvNeXtGradients", module = "spiraltorch.vision")]
pub(super) struct PyConvNeXtGradients {
    inner: ResidentConvNeXtGradients,
}
#[pymethods]
impl PyConvNeXtGradients {
    fn input_gradient_tensor(&self) -> PyWgpuTensor {
        PyWgpuTensor {
            inner: self.inner.input_gradient().clone(),
        }
    }
    fn parameter_gradient_tensors(&self) -> Vec<PyWgpuTensor> {
        self.inner
            .parameter_gradients()
            .iter()
            .cloned()
            .map(|inner| PyWgpuTensor { inner })
            .collect()
    }
}

#[pyclass(name = "ConvNeXtUpdate", module = "spiraltorch.vision")]
pub(super) struct PyConvNeXtUpdate {
    inner: ResidentParameterUpdate,
}
#[pymethods]
impl PyConvNeXtUpdate {
    #[getter]
    fn attempted_revision(&self) -> u64 {
        self.inner.revision()
    }
    /// Explicit readback of this update's frozen flags, not a submission receipt.
    fn read(&self, py: Python<'_>) -> PyResult<u64> {
        py.detach(|| self.inner.snapshot().and_then(|snapshot| snapshot.read()))
            .map_err(error)
    }
}

#[pyclass(name = "ConvNeXtCheckpointSnapshot", module = "spiraltorch.vision")]
pub(super) struct PyConvNeXtCheckpointSnapshot {
    inner: Option<ConvNeXtClassifierCheckpointSnapshot>,
}
#[pymethods]
impl PyConvNeXtCheckpointSnapshot {
    fn read_json(&mut self, py: Python<'_>) -> PyResult<String> {
        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("snapshot has already been consumed"))?;
        py.detach(|| inner.read().map_err(error)?.to_json().map_err(error))
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyResidentClassifier>()?;
    module.add_class::<PyConvNeXtForward>()?;
    module.add_class::<PyConvNeXtGradients>()?;
    module.add_class::<PyConvNeXtUpdate>()?;
    module.add_class::<PyConvNeXtCheckpointSnapshot>()?;
    Ok(())
}
