//! Forward-only graph handle; all lowering, guards and GPU ownership are Rust-owned.
use super::*;
#[cfg(feature = "wgpu")]
use crate::wgpu_tensor::{PyWgpuTensor, PyWgpuTensorDevice};
#[cfg(feature = "wgpu")]
use st_backend_wgpu::{
    resident_graph::{GraphInferenceError, GraphReadback, ResidentGraph},
    runtime,
};

#[cfg(feature = "wgpu")]
pub(super) fn error(err: GraphInferenceError) -> PyErr {
    match err {
        GraphInferenceError::Tensor(e) => crate::wgpu_tensor::error(e),
        GraphInferenceError::Kernel(e) => gpu_error(e),
        GraphInferenceError::Runtime(_) => PyRuntimeError::new_err(err.to_string()),
        _ => PyValueError::new_err(err.to_string()),
    }
}

pub(super) fn compile(
    plan: &InferencePlan,
    py: Python<'_>,
    tile: Option<&Bound<'_, PyAny>>,
    kernel: &str,
    accumulation: &str,
) -> PyResult<PyResidentGraphInference> {
    #[cfg(feature = "wgpu")]
    {
        let (tile, kernel, accumulation) = gpu_options(tile, kernel, accumulation)?;
        let plan = plan.clone();
        py.detach(move || {
            let (runtime, _) = runtime::ensure_default_runtime_blocking("python.nn.graph_forward")
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(PyResidentGraphInference {
                inner: plan
                    .compile_graph_wgpu_with_options(runtime, tile, kernel, accumulation)
                    .map_err(plan_error)?,
            })
        })
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let _ = (plan, py, tile, kernel, accumulation);
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "resident graph inference requires a wheel built with the 'wgpu' feature",
        ))
    }
}

#[pyclass(name = "ResidentGraphInference", module = "spiraltorch.nn")]
pub(super) struct PyResidentGraphInference {
    #[cfg(feature = "wgpu")]
    inner: ResidentGraph,
}
#[cfg(feature = "wgpu")]
#[pymethods]
impl PyResidentGraphInference {
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
    fn generation(&self) -> u64 {
        self.inner.generation()
    }
    #[getter]
    fn submitted_dispatches(&self) -> u64 {
        self.inner.submitted_dispatches()
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
        py.detach(|| self.inner.upload(&values)).map_err(error)
    }
    fn set_input_tensor(&mut self, py: Python<'_>, input: &PyWgpuTensor) -> PyResult<()> {
        py.detach(|| self.inner.set_input_tensor(&input.inner))
            .map_err(error)
    }
    fn dispatch(&mut self, py: Python<'_>) -> PyResult<u64> {
        py.detach(|| self.inner.dispatch()).map_err(error)
    }
    fn forward_tensor(&mut self, py: Python<'_>, input: &PyWgpuTensor) -> PyResult<PyWgpuTensor> {
        Ok(PyWgpuTensor {
            inner: py
                .detach(|| self.inner.forward_tensor(&input.inner))
                .map_err(error)?,
        })
    }
    fn output_tensor(&self, py: Python<'_>) -> PyResult<PyWgpuTensor> {
        Ok(PyWgpuTensor {
            inner: py.detach(|| self.inner.output_tensor()).map_err(error)?,
        })
    }
    fn snapshot(&self, py: Python<'_>) -> PyResult<PyGraphInferenceSnapshot> {
        let inner = py.detach(|| self.inner.snapshot()).map_err(error)?;
        Ok(PyGraphInferenceSnapshot {
            shape: inner.layout().shape().to_vec(),
            generation: inner.generation(),
            dispatch: inner.dispatch(),
            inner: Some(inner),
        })
    }
}

#[pyclass(name = "GraphInferenceSnapshot", module = "spiraltorch.nn")]
pub(super) struct PyGraphInferenceSnapshot {
    #[cfg(feature = "wgpu")]
    inner: Option<GraphReadback>,
    #[cfg(feature = "wgpu")]
    shape: Vec<usize>,
    #[cfg(feature = "wgpu")]
    generation: u64,
    #[cfg(feature = "wgpu")]
    dispatch: u64,
}
#[cfg(feature = "wgpu")]
#[pymethods]
impl PyGraphInferenceSnapshot {
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.shape.iter().copied())
    }
    #[getter]
    fn generation(&self) -> u64 {
        self.generation
    }
    #[getter]
    fn submitted_dispatch(&self) -> u64 {
        self.dispatch
    }
    fn read_values(&mut self, py: Python<'_>) -> PyResult<Vec<f32>> {
        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("snapshot has already been consumed"))?;
        py.detach(|| inner.read()).map_err(error)
    }
}
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyResidentGraphInference>()?;
    module.add_class::<PyGraphInferenceSnapshot>()?;
    Ok(())
}
