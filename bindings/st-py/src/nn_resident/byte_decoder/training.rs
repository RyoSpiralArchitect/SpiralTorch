//! Owning resident handles. Transfers and acceptance reads are explicit.
#[cfg(feature = "wgpu")]
use super::super::parameters::PyResidentParameterUpdate;
use super::*;
#[cfg(feature = "wgpu")]
use crate::wgpu_tensor::{PyWgpuTensor, PyWgpuTensorDevice};
#[cfg(feature = "wgpu")]
use st_nn::resident::{
    ByteDecoderBias, ByteDecoderCheckpointReadback, ByteDecoderQueryPullback, ResidentByteBatch,
    ResidentByteDecoder, ResidentByteDecoderForward, ResidentByteDecoderVjp,
};

#[pyclass(name = "ResidentByteBatch", module = "spiraltorch.nn", frozen)]
pub(super) struct PyResidentByteBatch {
    #[cfg(feature = "wgpu")]
    inner: ResidentByteBatch,
}

#[cfg(feature = "wgpu")]
#[pymethods]
impl PyResidentByteBatch {
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.shape().iter().copied())
    }
    fn targets_tensor(&self) -> PyWgpuTensor {
        PyWgpuTensor {
            inner: self.inner.targets().clone(),
        }
    }
}

#[pyclass(name = "ResidentByteDecoder", module = "spiraltorch.nn")]
pub(super) struct PyResidentByteDecoder {
    #[cfg(feature = "wgpu")]
    pub(super) inner: ResidentByteDecoder,
}

#[cfg(feature = "wgpu")]
#[pymethods]
impl PyResidentByteDecoder {
    #[getter]
    fn input_shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.input_layout().shape().iter().copied())
    }
    #[getter]
    fn output_shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.output_layout().shape().iter().copied())
    }
    #[getter]
    fn attempted_updates(&self) -> u64 {
        self.inner.parameter_snapshot().revision()
    }
    fn parameter_layout(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        parameter_layout(py, self.inner.parameter_layout())
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
    /// Explicit byte-index and target upload; reuse this batch across steps.
    fn prepare_batch(
        &self,
        py: Python<'_>,
        batch: &PyByteLmBatch,
    ) -> PyResult<PyResidentByteBatch> {
        Ok(PyResidentByteBatch {
            inner: py
                .detach(|| self.inner.prepare_batch(&batch.inner))
                .map_err(plan_error)?,
        })
    }
    /// Caller-owned external biases must themselves be causal.
    #[pyo3(signature = (batch, *, z_biases=None, pair_biases=None))]
    fn forward(
        &mut self,
        py: Python<'_>,
        batch: &PyResidentByteBatch,
        z_biases: Option<Vec<Option<PyRef<'_, PyWgpuTensor>>>>,
        pair_biases: Option<Vec<Option<PyRef<'_, PyWgpuTensor>>>>,
    ) -> PyResult<PyByteDecoderForward> {
        let count = self.inner.parameter_layout().blocks().len();
        let z = z_biases
            .map(|v| {
                v.into_iter()
                    .map(|t| t.map(|t| t.inner.clone()))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| vec![None; count]);
        let pairs = pair_biases
            .map(|v| {
                v.into_iter()
                    .map(|t| t.map(|t| t.inner.clone()))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| vec![None; count]);
        if z.len() != count || pairs.len() != count {
            return Err(PyValueError::new_err(
                "external bias lists must cover every block",
            ));
        }
        let biases = z
            .iter()
            .zip(&pairs)
            .map(|(z, p)| ByteDecoderBias {
                z_bias: z.as_ref(),
                pair_bias: p.as_ref(),
            })
            .collect::<Vec<_>>();
        Ok(PyByteDecoderForward {
            inner: py
                .detach(|| {
                    self.inner
                        .forward_with_external_biases(&batch.inner, &biases)
                })
                .map_err(plan_error)?,
        })
    }
    /// StopCoordinates is an explicit ablation, not a freeze of router parameters.
    #[pyo3(signature = (forward, cotangent, *, query_pullback="full"))]
    fn backward(
        &mut self,
        py: Python<'_>,
        forward: &PyByteDecoderForward,
        cotangent: &PyWgpuTensor,
        query_pullback: &str,
    ) -> PyResult<PyByteDecoderGradients> {
        let policy = match query_pullback {
            "full" => ByteDecoderQueryPullback::Full,
            "stop_coordinates" => ByteDecoderQueryPullback::StopCoordinates,
            _ => {
                return Err(PyValueError::new_err(
                    "query_pullback must be full or stop_coordinates",
                ))
            }
        };
        Ok(PyByteDecoderGradients {
            inner: py
                .detach(|| {
                    self.inner.backward_with_query_pullback(
                        &forward.inner,
                        &cotangent.inner,
                        policy,
                    )
                })
                .map_err(plan_error)?,
        })
    }
    fn sgd(
        &mut self,
        py: Python<'_>,
        gradients: &PyByteDecoderGradients,
        rate: f32,
    ) -> PyResult<PyResidentParameterUpdate> {
        Ok(PyResidentParameterUpdate {
            inner: py
                .detach(|| self.inner.sgd(&gradients.inner, rate))
                .map_err(plan_error)?,
        })
    }
    /// One rate per tensor in parameter_layout order. Zero freezes values, not VJPs.
    fn sgd_with_rates(
        &mut self,
        py: Python<'_>,
        gradients: &PyByteDecoderGradients,
        rates: Vec<f32>,
    ) -> PyResult<PyResidentParameterUpdate> {
        Ok(PyResidentParameterUpdate {
            inner: py
                .detach(|| self.inner.sgd_with_rates(&gradients.inner, &rates))
                .map_err(plan_error)?,
        })
    }
    fn checkpoint_snapshot(&self, py: Python<'_>) -> PyResult<PyByteDecoderCheckpointSnapshot> {
        Ok(PyByteDecoderCheckpointSnapshot {
            inner: Some(
                py.detach(|| self.inner.checkpoint_snapshot())
                    .map_err(plan_error)?,
            ),
        })
    }
}

#[pyclass(name = "ByteDecoderForward", module = "spiraltorch.nn", frozen)]
pub(super) struct PyByteDecoderForward {
    #[cfg(feature = "wgpu")]
    inner: ResidentByteDecoderForward,
}

#[cfg(feature = "wgpu")]
#[pymethods]
impl PyByteDecoderForward {
    #[getter]
    fn parameter_revision(&self) -> u64 {
        self.inner.parameter_revision()
    }
    fn prediction_tensor(&self) -> PyWgpuTensor {
        PyWgpuTensor {
            inner: self.inner.prediction().clone(),
        }
    }
    fn geometry_pair_bias_tensor(&self, block: usize) -> Option<PyWgpuTensor> {
        self.inner
            .geometry_pair_bias(block)
            .cloned()
            .map(|inner| PyWgpuTensor { inner })
    }
    fn geometry_mixture_weights_tensor(&self, block: usize) -> Option<PyWgpuTensor> {
        self.inner
            .geometry_mixture_weights(block)
            .cloned()
            .map(|inner| PyWgpuTensor { inner })
    }
    #[pyo3(signature = (*, reduction="mean", label_smoothing=0.0))]
    fn next_byte_loss(
        &self,
        py: Python<'_>,
        reduction: &str,
        label_smoothing: f64,
    ) -> PyResult<PyResidentLoss> {
        let spec = crate::tensor::parse_cross_entropy_config(reduction, -100, label_smoothing)?
            .kernel_spec()
            .map_err(crate::tensor::tensor_err_to_py)?;
        Ok(PyResidentLoss {
            inner: py
                .detach(|| self.inner.next_byte_loss(spec))
                .map_err(plan_error)?,
        })
    }
}

#[pyclass(name = "ByteDecoderGradients", module = "spiraltorch.nn", frozen)]
pub(super) struct PyByteDecoderGradients {
    #[cfg(feature = "wgpu")]
    inner: ResidentByteDecoderVjp,
}
#[cfg(feature = "wgpu")]
#[pymethods]
impl PyByteDecoderGradients {
    #[getter]
    fn query_pullback(&self) -> &'static str {
        match self.inner.query_pullback() {
            ByteDecoderQueryPullback::Full => "full",
            ByteDecoderQueryPullback::StopCoordinates => "stop_coordinates",
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
    fn embedding_output_gradient_tensor(&self) -> PyWgpuTensor {
        PyWgpuTensor {
            inner: self.inner.embedding_output_gradient().clone(),
        }
    }
    fn z_bias_gradient_tensor(&self, block: usize) -> PyResult<Option<PyWgpuTensor>> {
        let bias =
            self.inner.bias_gradients().get(block).ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err("block index out of range")
            })?;
        Ok(bias.z_bias().cloned().map(|inner| PyWgpuTensor { inner }))
    }
    fn pair_bias_gradient_tensor(&self, block: usize) -> PyResult<Option<PyWgpuTensor>> {
        let bias =
            self.inner.bias_gradients().get(block).ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err("block index out of range")
            })?;
        Ok(bias
            .pair_bias()
            .cloned()
            .map(|inner| PyWgpuTensor { inner }))
    }
}

#[pyclass(name = "ByteDecoderCheckpointSnapshot", module = "spiraltorch.nn")]
pub(super) struct PyByteDecoderCheckpointSnapshot {
    #[cfg(feature = "wgpu")]
    inner: Option<ByteDecoderCheckpointReadback>,
}
#[cfg(feature = "wgpu")]
#[pymethods]
impl PyByteDecoderCheckpointSnapshot {
    /// Consume one immutable capture, including device validation, explicitly.
    fn read(&mut self, py: Python<'_>) -> PyResult<PyByteDecoderCheckpoint> {
        let capture = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("snapshot has already been consumed"))?;
        Ok(PyByteDecoderCheckpoint {
            inner: py.detach(|| capture.read()).map_err(plan_error)?,
        })
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyResidentByteBatch>()?;
    module.add_class::<PyResidentByteDecoder>()?;
    module.add_class::<PyByteDecoderForward>()?;
    module.add_class::<PyByteDecoderGradients>()?;
    module.add_class::<PyByteDecoderCheckpointSnapshot>()?;
    Ok(())
}
