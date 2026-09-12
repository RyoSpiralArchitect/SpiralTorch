//! Thin custom-objective learner and owning gradient batch bindings.
#[cfg(feature = "wgpu")]
use super::autograd::{PyGraphForward, PyGraphGradients};
#[cfg(feature = "wgpu")]
use super::graph::PyGraphTrainingParametersSnapshot;
#[cfg(feature = "wgpu")]
use super::training::training_error;
use super::*;
#[cfg(feature = "wgpu")]
use crate::wgpu_tensor::{PyWgpuTensor, PyWgpuTensorDevice};
#[cfg(feature = "wgpu")]
use st_backend_wgpu::{resident_training::graph as backend, runtime};

pub(super) fn compile(
    plan: &InferencePlan,
    py: Python<'_>,
    policy: &str,
    tile: Option<&Bound<'_, PyAny>>,
    kernel: &str,
    accumulation: &str,
) -> PyResult<PyResidentGraphLearner> {
    let policy = policy
        .parse::<st_nn::resident::GraphGradientPolicy>()
        .map_err(PyValueError::new_err)?;
    #[cfg(feature = "wgpu")]
    {
        let (tile, kernel, accumulation) = gpu_options(tile, kernel, accumulation)?;
        let plan = plan.clone();
        py.detach(move || {
            let (runtime, _) = runtime::ensure_default_runtime_blocking("python.nn.graph_learner")
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(PyResidentGraphLearner {
                inner: plan
                    .compile_graph_learner_wgpu_with_options(
                        runtime,
                        policy,
                        tile,
                        kernel,
                        accumulation,
                    )
                    .map_err(plan_error)?,
            })
        })
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let _ = (plan, py, policy, tile, kernel, accumulation);
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "resident graph learning requires a wheel built with the 'wgpu' feature",
        ))
    }
}

#[pyclass(name = "GraphGradientBatch", module = "spiraltorch.nn")]
pub(super) struct PyGraphGradientBatch {
    #[cfg(feature = "wgpu")]
    inner: backend::GraphGradientBatch,
}
#[cfg(feature = "wgpu")]
#[pymethods]
impl PyGraphGradientBatch {
    #[new]
    fn new() -> Self {
        Self {
            inner: backend::GraphGradientBatch::new(),
        }
    }
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    fn add(&mut self, gradients: &PyGraphGradients, weight: f32) -> PyResult<()> {
        self.inner
            .add(&gradients.inner, weight)
            .map_err(training_error)
    }
}

#[pyclass(name = "ResidentGraphLearner", module = "spiraltorch.nn")]
pub(super) struct PyResidentGraphLearner {
    #[cfg(feature = "wgpu")]
    inner: backend::ResidentGraphLearner,
}
#[cfg(feature = "wgpu")]
#[pymethods]
impl PyResidentGraphLearner {
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
    fn gradient_policy(&self) -> &'static str {
        self.inner.gradient_policy().as_str()
    }
    #[getter]
    fn grad_clip_max_norm(&self) -> Option<f32> {
        self.inner.grad_clip_max_norm()
    }
    fn set_grad_clip_max_norm(&mut self, py: Python<'_>, max_norm: f32) -> PyResult<()> {
        py.detach(|| self.inner.set_grad_clip_max_norm(max_norm))
            .map_err(training_error)
    }
    fn clear_grad_clip(&mut self) {
        self.inner.clear_grad_clip();
    }
    #[getter]
    fn momentum_damping(&self) -> Option<f32> {
        self.inner.momentum_damping()
    }
    fn set_momentum_damping(&mut self, py: Python<'_>, damping: f32) -> PyResult<()> {
        py.detach(|| self.inner.set_momentum_damping(damping))
            .map_err(training_error)
    }
    fn clear_momentum(&mut self) {
        self.inner.clear_momentum();
    }
    fn reset_momentum(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.reset_momentum())
            .map_err(training_error)
    }
    fn momentum_tensors(&self, py: Python<'_>) -> PyResult<Vec<PyWgpuTensor>> {
        Ok(py
            .detach(|| self.inner.momentum_tensors())
            .map_err(training_error)?
            .into_iter()
            .map(|inner| PyWgpuTensor { inner })
            .collect())
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
    #[getter]
    fn submitted_updates(&self) -> u64 {
        self.inner.submitted_updates()
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
    fn upload_values(&mut self, py: Python<'_>, input: Vec<f32>) -> PyResult<()> {
        py.detach(|| self.inner.upload(&input))
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
    fn sgd(&mut self, py: Python<'_>, gradients: &PyGraphGradients, rate: f32) -> PyResult<u64> {
        py.detach(|| self.inner.sgd(&gradients.inner, rate))
            .map_err(training_error)
    }
    fn sgd_weighted(
        &mut self,
        py: Python<'_>,
        batch: &PyGraphGradientBatch,
        rate: f32,
    ) -> PyResult<u64> {
        py.detach(|| self.inner.sgd_batch(&batch.inner, rate))
            .map_err(training_error)
    }
    fn gradient_accumulator(&self, py: Python<'_>) -> PyResult<PyGraphGradientAccumulator> {
        Ok(PyGraphGradientAccumulator {
            inner: py
                .detach(|| self.inner.gradient_accumulator())
                .map_err(training_error)?,
        })
    }
    fn zero_accumulator(
        &self,
        py: Python<'_>,
        accumulator: &mut PyGraphGradientAccumulator,
    ) -> PyResult<()> {
        py.detach(|| self.inner.zero_accumulator(&mut accumulator.inner))
            .map_err(training_error)
    }
    fn accumulate(
        &mut self,
        py: Python<'_>,
        accumulator: &mut PyGraphGradientAccumulator,
        gradients: &PyGraphGradients,
        weight: f32,
    ) -> PyResult<u64> {
        py.detach(|| {
            self.inner
                .accumulate(&mut accumulator.inner, &gradients.inner, weight)
        })
        .map_err(training_error)
    }
    fn sgd_accumulated(
        &mut self,
        py: Python<'_>,
        accumulator: &PyGraphGradientAccumulator,
        rate: f32,
    ) -> PyResult<u64> {
        py.detach(|| self.inner.sgd_accumulated(&accumulator.inner, rate))
            .map_err(training_error)
    }
    fn parameter_snapshot(&self, py: Python<'_>) -> PyResult<PyGraphTrainingParametersSnapshot> {
        Ok(PyGraphTrainingParametersSnapshot {
            inner: Some(
                py.detach(|| self.inner.parameter_snapshot())
                    .map_err(training_error)?,
            ),
        })
    }
    fn update_snapshot(&self, py: Python<'_>) -> PyResult<PyGraphUpdateSnapshot> {
        let inner = py
            .detach(|| self.inner.update_snapshot())
            .map_err(training_error)?;
        Ok(PyGraphUpdateSnapshot {
            update: inner.submitted_update(),
            generation: inner.input_generation(),
            forward: inner.submitted_forward(),
            inner: Some(inner),
        })
    }
}

#[pyclass(name = "GraphGradientAccumulator", module = "spiraltorch.nn")]
pub(super) struct PyGraphGradientAccumulator {
    #[cfg(feature = "wgpu")]
    inner: backend::GraphGradientAccumulator,
}
#[cfg(feature = "wgpu")]
#[pymethods]
impl PyGraphGradientAccumulator {
    fn __len__(&self) -> PyResult<usize> {
        isize::try_from(self.inner.len())
            .map(|n| n as usize)
            .map_err(|_| {
                pyo3::exceptions::PyOverflowError::new_err(
                    "accumulator length exceeds Python sequence limits",
                )
            })
    }
    #[getter]
    fn parameter_generation(&self) -> u64 {
        self.inner.parameter_generation()
    }
    fn parameter_gradient_tensors(&self, py: Python<'_>) -> PyResult<Vec<PyWgpuTensor>> {
        Ok(py
            .detach(|| self.inner.parameter_gradients())
            .map_err(training_error)?
            .into_iter()
            .map(|inner| PyWgpuTensor { inner })
            .collect())
    }
}

#[pyclass(name = "GraphUpdateSnapshot", module = "spiraltorch.nn")]
pub(super) struct PyGraphUpdateSnapshot {
    #[cfg(feature = "wgpu")]
    inner: Option<backend::GraphUpdateReadback>,
    #[cfg(feature = "wgpu")]
    update: u64,
    #[cfg(feature = "wgpu")]
    generation: u64,
    #[cfg(feature = "wgpu")]
    forward: u64,
}
#[cfg(feature = "wgpu")]
#[pymethods]
impl PyGraphUpdateSnapshot {
    #[getter]
    fn submitted_update(&self) -> u64 {
        self.update
    }
    #[getter]
    fn input_generation(&self) -> u64 {
        self.generation
    }
    #[getter]
    fn submitted_forward(&self) -> u64 {
        self.forward
    }
    fn read(&mut self, py: Python<'_>) -> PyResult<u64> {
        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("snapshot has already been consumed"))?;
        py.detach(|| inner.read()).map_err(training_error)
    }
}
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyResidentGraphLearner>()?;
    module.add_class::<PyGraphGradientBatch>()?;
    module.add_class::<PyGraphGradientAccumulator>()?;
    module.add_class::<PyGraphUpdateSnapshot>()?;
    Ok(())
}
