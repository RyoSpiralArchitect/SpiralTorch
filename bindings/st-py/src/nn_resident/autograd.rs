//! Thin Python ownership for the Rust graph tape and exact resident VJPs.
#[cfg(feature = "wgpu")]
use super::training::training_error;
use super::*;
#[cfg(feature = "wgpu")]
use crate::wgpu_tensor::{PyWgpuTensor, PyWgpuTensorDevice};
#[cfg(feature = "wgpu")]
use st_backend_wgpu::{
    resident_training::graph::{GraphForward, GraphGradients, ResidentGraphAutograd},
    runtime,
};

pub(super) fn compile(
    plan: &InferencePlan,
    py: Python<'_>,
    tile: Option<&Bound<'_, PyAny>>,
    kernel: &str,
    accumulation: &str,
) -> PyResult<PyResidentGraphAutograd> {
    #[cfg(feature = "wgpu")]
    {
        let (tile, kernel, accumulation) = gpu_options(tile, kernel, accumulation)?;
        let plan = plan.clone();
        py.detach(move || {
            let (runtime, _) = runtime::ensure_default_runtime_blocking("python.nn.graph_autograd")
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(PyResidentGraphAutograd {
                inner: plan
                    .compile_graph_autograd_wgpu_with_options(runtime, tile, kernel, accumulation)
                    .map_err(plan_error)?,
            })
        })
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let _ = (plan, py, tile, kernel, accumulation);
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "resident graph autograd requires a wheel built with the 'wgpu' feature",
        ))
    }
}

#[pyclass(name = "ResidentGraphAutograd", module = "spiraltorch.nn")]
pub(super) struct PyResidentGraphAutograd {
    #[cfg(feature = "wgpu")]
    inner: ResidentGraphAutograd,
}
#[cfg(feature = "wgpu")]
#[pymethods]
impl PyResidentGraphAutograd {
    #[getter]
    fn input_shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.input_layout().shape().iter().copied())
    }
    #[getter]
    fn output_shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.output_layout().shape().iter().copied())
    }
    #[getter]
    fn stage_count(&self) -> usize {
        self.inner.stage_count()
    }
    #[getter]
    fn parameter_count(&self) -> usize {
        self.inner.parameter_count()
    }
    #[getter]
    fn input_generation(&self) -> u64 {
        self.inner.input_generation()
    }
    #[getter]
    fn submitted_forwards(&self) -> u64 {
        self.inner.submitted_forwards()
    }
    #[getter]
    fn submitted_backwards(&self) -> u64 {
        self.inner.submitted_backwards()
    }
    fn tensor_device(&self) -> PyWgpuTensorDevice {
        PyWgpuTensorDevice {
            inner: self.inner.tensor_device().clone(),
        }
    }
    fn adapter_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let info = self.inner.adapter_info();
        let result = PyDict::new(py);
        result.set_item("name", &info.name)?;
        result.set_item("backend", format!("{:?}", info.backend))?;
        result.set_item("device_type", format!("{:?}", info.device_type))?;
        Ok(result)
    }
    fn upload_values(&mut self, py: Python<'_>, values: Vec<f32>) -> PyResult<()> {
        py.detach(|| self.inner.upload(&values))
            .map_err(training_error)
    }
    fn set_input_tensor(&mut self, py: Python<'_>, input: &PyWgpuTensor) -> PyResult<()> {
        py.detach(|| self.inner.set_input_tensor(&input.inner))
            .map_err(training_error)
    }
    fn forward(&mut self, py: Python<'_>) -> PyResult<PyGraphForward> {
        Ok(PyGraphForward {
            inner: py.detach(|| self.inner.forward()).map_err(training_error)?,
        })
    }
    fn backward(
        &mut self,
        py: Python<'_>,
        forward: &PyGraphForward,
        cotangent: &PyWgpuTensor,
    ) -> PyResult<PyGraphGradients> {
        Ok(PyGraphGradients {
            inner: py
                .detach(|| self.inner.backward(&forward.inner, &cotangent.inner))
                .map_err(training_error)?,
        })
    }
}

#[pyclass(name = "GraphForward", module = "spiraltorch.nn")]
pub(super) struct PyGraphForward {
    #[cfg(feature = "wgpu")]
    pub(super) inner: GraphForward,
}
#[cfg(feature = "wgpu")]
#[pymethods]
impl PyGraphForward {
    #[getter]
    fn input_generation(&self) -> u64 {
        self.inner.input_generation()
    }
    #[getter]
    fn submitted_forward(&self) -> u64 {
        self.inner.submitted_forward()
    }
    fn prediction_tensor(&self) -> PyWgpuTensor {
        PyWgpuTensor {
            inner: self.inner.prediction().clone(),
        }
    }
}

#[pyclass(name = "GraphGradients", module = "spiraltorch.nn")]
pub(super) struct PyGraphGradients {
    #[cfg(feature = "wgpu")]
    pub(super) inner: GraphGradients,
}
#[cfg(feature = "wgpu")]
#[pymethods]
impl PyGraphGradients {
    #[getter]
    fn input_generation(&self) -> u64 {
        self.inner.input_generation()
    }
    #[getter]
    fn submitted_forward(&self) -> u64 {
        self.inner.submitted_forward()
    }
    #[getter]
    fn submitted_backward(&self) -> u64 {
        self.inner.submitted_backward()
    }
    #[getter]
    fn parameter_count(&self) -> usize {
        self.inner.parameter_gradients().len()
    }
    fn input_gradient_tensor(&self) -> PyWgpuTensor {
        PyWgpuTensor {
            inner: self.inner.input_gradient().clone(),
        }
    }
    fn parameter_gradient_tensor(&self, index: usize) -> PyResult<PyWgpuTensor> {
        Ok(PyWgpuTensor {
            inner: self
                .inner
                .parameter_gradients()
                .get(index)
                .ok_or_else(|| {
                    pyo3::exceptions::PyIndexError::new_err("parameter gradient index out of range")
                })?
                .clone(),
        })
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyResidentGraphAutograd>()?;
    module.add_class::<PyGraphForward>()?;
    module.add_class::<PyGraphGradients>()?;
    Ok(())
}
