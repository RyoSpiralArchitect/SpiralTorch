//! Thin handles for the Rust model/input/schedule owner.
use super::*;
use crate::wgpu_tensor::{PyWgpuTensor, PyWgpuTensorDevice};
use st_vision::resident_trainer::{
    ResidentVisionStepOutcome, ResidentVisionSubmission, ResidentVisionTrainer,
    ResidentVisionTrainerConfig, VisionTrainingCheckpoint, VisionTrainingCheckpointSnapshot,
};

fn error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pyclass(name = "ResidentVisionTrainer", module = "spiraltorch.vision")]
pub(super) struct PyResidentVisionTrainer {
    inner: ResidentVisionTrainer<TensorVisionDataset>,
}

#[pymethods]
impl PyResidentVisionTrainer {
    #[staticmethod]
    fn default_config_json() -> PyResult<String> {
        ResidentVisionTrainerConfig::default()
            .to_json()
            .map_err(error)
    }

    #[staticmethod]
    #[pyo3(signature = (device, dataset, dataset_sha256, config_json, pipeline=None))]
    fn create(
        py: Python<'_>,
        device: &PyWgpuTensorDevice,
        dataset: &PyTensorVisionDataset,
        dataset_sha256: &str,
        config_json: &str,
        pipeline: Option<&PyTransformPipeline>,
    ) -> PyResult<Self> {
        let config = ResidentVisionTrainerConfig::from_json(config_json).map_err(error)?;
        let dataset = Arc::new(dataset.inner.clone());
        let pipeline = pipeline.map(|p| p.inner.clone());
        Ok(Self {
            inner: py
                .detach(|| {
                    ResidentVisionTrainer::from_dataset(
                        &config,
                        device.inner.clone(),
                        dataset,
                        pipeline,
                        dataset_sha256,
                    )
                })
                .map_err(error)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (device, dataset, dataset_sha256, payload, pipeline=None))]
    fn from_checkpoint_json(
        py: Python<'_>,
        device: &PyWgpuTensorDevice,
        dataset: &PyTensorVisionDataset,
        dataset_sha256: &str,
        payload: &str,
        pipeline: Option<&PyTransformPipeline>,
    ) -> PyResult<Self> {
        let checkpoint = VisionTrainingCheckpoint::from_json(payload).map_err(error)?;
        let dataset = Arc::new(dataset.inner.clone());
        let pipeline = pipeline.map(|p| p.inner.clone());
        Ok(Self {
            inner: py
                .detach(|| {
                    ResidentVisionTrainer::from_dataset_checkpoint(
                        device.inner.clone(),
                        dataset,
                        pipeline,
                        dataset_sha256,
                        &checkpoint,
                    )
                })
                .map_err(error)?,
        })
    }

    fn restore_checkpoint_json(&mut self, py: Python<'_>, payload: &str) -> PyResult<()> {
        py.detach(|| {
            let checkpoint = VisionTrainingCheckpoint::from_json(payload).map_err(error)?;
            self.inner.restore_checkpoint(&checkpoint).map_err(error)
        })
    }

    fn state_json(&self) -> PyResult<String> {
        serde_json::to_string(self.inner.state()).map_err(error)
    }

    fn apply_zspace_meta_optimizer_report_json(
        &mut self,
        py: Python<'_>,
        report: &str,
    ) -> PyResult<String> {
        py.detach(|| {
            let receipt = self
                .inner
                .apply_zspace_meta_optimizer_report_json(report)
                .map_err(error)?;
            serde_json::to_string(&receipt).map_err(error)
        })
    }

    #[getter]
    fn has_pending_update(&self) -> bool {
        self.inner.has_pending_update()
    }

    fn submit_next(&mut self, py: Python<'_>) -> PyResult<PyResidentVisionSubmission> {
        Ok(PyResidentVisionSubmission {
            inner: py.detach(|| self.inner.submit_next()).map_err(error)?,
        })
    }

    fn settle(&mut self, py: Python<'_>) -> PyResult<PyResidentVisionStepOutcome> {
        Ok(PyResidentVisionStepOutcome {
            inner: py.detach(|| self.inner.settle()).map_err(error)?,
        })
    }

    fn checkpoint_snapshot(&self, py: Python<'_>) -> PyResult<PyVisionTrainingCheckpointSnapshot> {
        Ok(PyVisionTrainingCheckpointSnapshot {
            inner: Some(
                py.detach(|| self.inner.checkpoint_snapshot())
                    .map_err(error)?,
            ),
        })
    }
}

#[pyclass(name = "ResidentVisionSubmission", module = "spiraltorch.vision")]
pub(super) struct PyResidentVisionSubmission {
    inner: ResidentVisionSubmission,
}

#[pymethods]
impl PyResidentVisionSubmission {
    #[getter]
    fn attempted_revision(&self) -> u64 {
        self.inner.attempted_revision
    }
    #[getter]
    fn epoch(&self) -> u64 {
        self.inner.epoch
    }
    #[getter]
    fn learning_rate(&self) -> f32 {
        self.inner.learning_rate
    }
    fn labels(&self) -> Vec<Option<String>> {
        self.inner.labels.clone()
    }
    fn images(&self) -> PyWgpuTensor {
        PyWgpuTensor {
            inner: self.inner.images.clone(),
        }
    }
    fn loss_tensor(&self) -> PyWgpuTensor {
        PyWgpuTensor {
            inner: self.inner.loss.clone(),
        }
    }
}

#[pyclass(name = "ResidentVisionStepOutcome", module = "spiraltorch.vision")]
pub(super) struct PyResidentVisionStepOutcome {
    inner: ResidentVisionStepOutcome,
}

#[pymethods]
impl PyResidentVisionStepOutcome {
    #[getter]
    fn attempted_revision(&self) -> u64 {
        self.inner.attempted_revision
    }
    #[getter]
    fn accepted(&self) -> bool {
        self.inner.accepted
    }
}

#[pyclass(
    name = "VisionTrainingCheckpointSnapshot",
    module = "spiraltorch.vision"
)]
pub(super) struct PyVisionTrainingCheckpointSnapshot {
    inner: Option<VisionTrainingCheckpointSnapshot>,
}

#[pymethods]
impl PyVisionTrainingCheckpointSnapshot {
    fn read_json(&mut self, py: Python<'_>) -> PyResult<String> {
        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("snapshot has already been consumed"))?;
        py.detach(|| inner.read().map_err(error)?.to_json().map_err(error))
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyResidentVisionTrainer>()?;
    module.add_class::<PyResidentVisionSubmission>()?;
    module.add_class::<PyResidentVisionStepOutcome>()?;
    module.add_class::<PyVisionTrainingCheckpointSnapshot>()?;
    Ok(())
}
