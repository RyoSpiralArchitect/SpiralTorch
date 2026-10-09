//! Plan conversion and owning handles; all attention/training semantics stay in Rust.
#[cfg(feature = "wgpu")]
use super::parameters::PyResidentParameterUpdate;
use super::*;
#[cfg(feature = "wgpu")]
use crate::wgpu_tensor::{PyWgpuTensor, PyWgpuTensorDevice};
use st_nn::resident::{AttentionInferencePlan, AttentionMask};
#[cfg(feature = "wgpu")]
use st_nn::resident::{ResidentAttentionForward, ResidentAttentionTraining, ResidentAttentionVjp};

fn integer(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    if value.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "heads/offset must be an integer, not bool",
        ));
    }
    value.extract()
}

#[pyclass(name = "AttentionInferencePlan", module = "spiraltorch.nn", frozen)]
pub(super) struct PyAttentionPlan {
    inner: AttentionInferencePlan,
}

#[pymethods]
impl PyAttentionPlan {
    #[staticmethod]
    #[pyo3(signature = (query, key, value, output, *, heads, causal_offset=None))]
    #[allow(clippy::too_many_arguments)]
    fn from_projection_plans(
        py: Python<'_>,
        query: &PyInferencePlan,
        key: &PyInferencePlan,
        value: &PyInferencePlan,
        output: &PyInferencePlan,
        heads: &Bound<'_, PyAny>,
        causal_offset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let heads = integer(heads)?;
        let mask = causal_offset
            .map(integer)
            .transpose()?
            .map_or(AttentionMask::None, |query_offset| AttentionMask::Causal {
                query_offset,
            });
        let inner = py
            .detach(|| {
                AttentionInferencePlan::from_projection_plans(
                    heads,
                    mask,
                    [&query.inner, &key.inner, &value.inner, &output.inner],
                )
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
    ) -> PyResult<PyAttentionTraining> {
        #[cfg(feature = "wgpu")]
        {
            let (tile, kernel, accumulation) = gpu_options(tile_mnk, kernel, accumulation)?;
            let plan = self.inner.clone();
            py.detach(move || {
                let (runtime, _) = st_backend_wgpu::runtime::ensure_default_runtime_blocking(
                    "python.nn.attention_training",
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                Ok(PyAttentionTraining {
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
                "resident attention training requires the 'wgpu' feature",
            ))
        }
    }
}

#[pyclass(name = "ResidentAttentionTraining", module = "spiraltorch.nn")]
pub(super) struct PyAttentionTraining {
    #[cfg(feature = "wgpu")]
    inner: ResidentAttentionTraining,
}
#[cfg(feature = "wgpu")]
#[pymethods]
impl PyAttentionTraining {
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
    ) -> PyResult<PyAttentionForward> {
        Ok(PyAttentionForward {
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
        forward: &PyAttentionForward,
        cotangent: &PyWgpuTensor,
    ) -> PyResult<PyAttentionGradients> {
        Ok(PyAttentionGradients {
            inner: py
                .detach(|| self.inner.backward(&forward.inner, &cotangent.inner))
                .map_err(plan_error)?,
        })
    }
    fn sgd(
        &mut self,
        py: Python<'_>,
        gradients: &PyAttentionGradients,
        rate: f32,
    ) -> PyResult<PyResidentParameterUpdate> {
        Ok(PyResidentParameterUpdate {
            inner: py
                .detach(|| self.inner.sgd(&gradients.inner, rate))
                .map_err(plan_error)?,
        })
    }
}

#[pyclass(name = "AttentionForward", module = "spiraltorch.nn", frozen)]
pub(super) struct PyAttentionForward {
    #[cfg(feature = "wgpu")]
    inner: ResidentAttentionForward,
}
#[cfg(feature = "wgpu")]
#[pymethods]
impl PyAttentionForward {
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

#[pyclass(name = "AttentionGradients", module = "spiraltorch.nn", frozen)]
pub(super) struct PyAttentionGradients {
    #[cfg(feature = "wgpu")]
    inner: ResidentAttentionVjp,
}
#[cfg(feature = "wgpu")]
#[pymethods]
impl PyAttentionGradients {
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
    module.add_class::<PyAttentionPlan>()?;
    module.add_class::<PyAttentionTraining>()?;
    module.add_class::<PyAttentionForward>()?;
    module.add_class::<PyAttentionGradients>()?;
    Ok(())
}
