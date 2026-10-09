//! Thin owning handles over the Rust residual attention training contract.
use super::attention::PyAttentionPlan;
#[cfg(feature = "wgpu")]
use super::parameters::PyResidentParameterUpdate;
use super::*;
#[cfg(feature = "wgpu")]
use crate::wgpu_tensor::{PyWgpuTensor, PyWgpuTensorDevice};
use st_nn::resident::ResidualAttentionPlan;
#[cfg(feature = "wgpu")]
use st_nn::resident::{
    ResidentResidualAttentionForward, ResidentResidualAttentionTraining,
    ResidentResidualAttentionVjp,
};

#[pyclass(name = "ResidualAttentionPlan", module = "spiraltorch.nn", frozen)]
pub(super) struct PyResidualAttentionPlan {
    inner: ResidualAttentionPlan,
}

#[pymethods]
impl PyResidualAttentionPlan {
    #[staticmethod]
    fn from_plans(
        py: Python<'_>,
        pre: &PyInferencePlan,
        attention: &PyAttentionPlan,
        feed_forward: &PyInferencePlan,
    ) -> PyResult<Self> {
        let inner = py
            .detach(|| {
                ResidualAttentionPlan::from_plans(&pre.inner, &attention.inner, &feed_forward.inner)
            })
            .map_err(plan_error)?;
        Ok(Self { inner })
    }
    #[getter]
    fn input_shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.input_layout().shape().iter().copied())
    }
    #[getter]
    fn output_shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.output_layout().shape().iter().copied())
    }
    #[pyo3(signature = (*, tile_mnk=None, kernel="scalar", accumulation="sequential"))]
    fn compile_training_wgpu(
        &self,
        py: Python<'_>,
        tile_mnk: Option<&Bound<'_, PyAny>>,
        kernel: &str,
        accumulation: &str,
    ) -> PyResult<PyResidualAttentionTraining> {
        #[cfg(feature = "wgpu")]
        {
            let (tile, kernel, accumulation) = gpu_options(tile_mnk, kernel, accumulation)?;
            let plan = self.inner.clone();
            py.detach(move || {
                let (runtime, _) = st_backend_wgpu::runtime::ensure_default_runtime_blocking(
                    "python.nn.residual_attention_training",
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                Ok(PyResidualAttentionTraining {
                    inner: plan
                        .compile_training_wgpu_with_options(runtime, tile, kernel, accumulation)
                        .map_err(plan_error)?,
                })
            })
        }
        #[cfg(not(feature = "wgpu"))]
        {
            let _ = (py, tile_mnk, kernel, accumulation);
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "resident residual attention training requires the 'wgpu' feature",
            ))
        }
    }
}

#[pyclass(name = "ResidentResidualAttentionTraining", module = "spiraltorch.nn")]
pub(super) struct PyResidualAttentionTraining {
    #[cfg(feature = "wgpu")]
    inner: ResidentResidualAttentionTraining,
}
#[cfg(feature = "wgpu")]
#[pymethods]
impl PyResidualAttentionTraining {
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
    #[pyo3(signature = (input, *, z_bias=None, pair_bias=None))]
    fn forward(
        &mut self,
        py: Python<'_>,
        input: &PyWgpuTensor,
        z_bias: Option<&PyWgpuTensor>,
        pair_bias: Option<&PyWgpuTensor>,
    ) -> PyResult<PyResidualAttentionForward> {
        Ok(PyResidualAttentionForward {
            inner: py
                .detach(|| {
                    self.inner.forward(
                        &input.inner,
                        z_bias.map(|b| &b.inner),
                        pair_bias.map(|b| &b.inner),
                    )
                })
                .map_err(plan_error)?,
        })
    }
    fn backward(
        &mut self,
        py: Python<'_>,
        forward: &PyResidualAttentionForward,
        cotangent: &PyWgpuTensor,
    ) -> PyResult<PyResidualAttentionGradients> {
        Ok(PyResidualAttentionGradients {
            inner: py
                .detach(|| self.inner.backward(&forward.inner, &cotangent.inner))
                .map_err(plan_error)?,
        })
    }
    fn sgd(
        &mut self,
        py: Python<'_>,
        gradients: &PyResidualAttentionGradients,
        rate: f32,
    ) -> PyResult<PyResidentParameterUpdate> {
        Ok(PyResidentParameterUpdate {
            inner: py
                .detach(|| self.inner.sgd(&gradients.inner, rate))
                .map_err(plan_error)?,
        })
    }
}

#[pyclass(name = "ResidualAttentionForward", module = "spiraltorch.nn", frozen)]
pub(super) struct PyResidualAttentionForward {
    #[cfg(feature = "wgpu")]
    inner: ResidentResidualAttentionForward,
}
#[cfg(feature = "wgpu")]
#[pymethods]
impl PyResidualAttentionForward {
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

#[pyclass(name = "ResidualAttentionGradients", module = "spiraltorch.nn", frozen)]
pub(super) struct PyResidualAttentionGradients {
    #[cfg(feature = "wgpu")]
    inner: ResidentResidualAttentionVjp,
}
#[cfg(feature = "wgpu")]
#[pymethods]
impl PyResidualAttentionGradients {
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
    fn z_bias_gradient_tensor(&self) -> Option<PyWgpuTensor> {
        self.inner
            .z_bias_gradient()
            .cloned()
            .map(|inner| PyWgpuTensor { inner })
    }
    fn pair_bias_gradient_tensor(&self) -> Option<PyWgpuTensor> {
        self.inner
            .pair_bias_gradient()
            .cloned()
            .map(|inner| PyWgpuTensor { inner })
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyResidualAttentionPlan>()?;
    module.add_class::<PyResidualAttentionTraining>()?;
    module.add_class::<PyResidualAttentionForward>()?;
    module.add_class::<PyResidualAttentionGradients>()?;
    Ok(())
}
