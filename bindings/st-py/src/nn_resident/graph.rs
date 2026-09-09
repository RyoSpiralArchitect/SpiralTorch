//! Thin owning handles over Rust's general resident graph training.
#[cfg(feature = "wgpu")]
use super::training::{training_error, PyTrainingLossSnapshot};
use super::*;
#[cfg(feature = "wgpu")]
use pyo3::exceptions::PyIndexError;
#[cfg(feature = "wgpu")]
use st_backend_wgpu::{resident_training::graph as backend, runtime};

pub(super) fn compile(
    plan: &InferencePlan,
    py: Python<'_>,
    policy: &str,
    tile: Option<&Bound<'_, PyAny>>,
    kernel: &str,
    accumulation: &str,
) -> PyResult<PyResidentGraphTraining> {
    let policy = policy
        .parse::<st_nn::resident::GraphGradientPolicy>()
        .map_err(PyValueError::new_err)?;
    #[cfg(feature = "wgpu")]
    {
        let (tile, kernel, accumulation) = gpu_options(tile, kernel, accumulation)?;
        let plan = plan.clone();
        py.detach(move || {
            let (runtime, _) = runtime::ensure_default_runtime_blocking("python.nn.graph_training")
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let inner = plan
                .compile_graph_training_wgpu_with_options(
                    runtime,
                    policy,
                    tile,
                    kernel,
                    accumulation,
                )
                .map_err(plan_error)?;
            Ok(PyResidentGraphTraining { inner })
        })
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let _ = (plan, py, policy, tile, kernel, accumulation);
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "resident graph training requires a wheel built with the 'wgpu' feature",
        ))
    }
}

#[pyclass(name = "ResidentGraphTraining", module = "spiraltorch.nn")]
pub(super) struct PyResidentGraphTraining {
    #[cfg(feature = "wgpu")]
    inner: backend::ResidentGraphTraining,
}

#[cfg(feature = "wgpu")]
#[pymethods]
impl PyResidentGraphTraining {
    fn upload_batch_tensors(
        &mut self,
        py: Python<'_>,
        input: &crate::wgpu_tensor::PyWgpuTensor,
        target: &crate::wgpu_tensor::PyWgpuTensor,
    ) -> PyResult<()> {
        py.detach(|| self.inner.upload_batch_tensors(&input.inner, &target.inner))
            .map_err(training_error)
    }
    fn prediction_tensor(&self, py: Python<'_>) -> PyResult<crate::wgpu_tensor::PyWgpuTensor> {
        Ok(crate::wgpu_tensor::PyWgpuTensor {
            inner: py
                .detach(|| self.inner.prediction_tensor())
                .map_err(training_error)?,
        })
    }
    fn input_gradient_tensor(&self, py: Python<'_>) -> PyResult<crate::wgpu_tensor::PyWgpuTensor> {
        Ok(crate::wgpu_tensor::PyWgpuTensor {
            inner: py
                .detach(|| self.inner.input_gradient_tensor())
                .map_err(training_error)?,
        })
    }
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
    fn submitted_steps(&self) -> u64 {
        self.inner.submitted_steps()
    }
    #[getter]
    fn batch_generation(&self) -> u64 {
        self.inner.batch_generation()
    }

    fn adapter_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let info = self.inner.adapter_info();
        let result = PyDict::new(py);
        result.set_item("name", &info.name)?;
        result.set_item("backend", format!("{:?}", info.backend))?;
        result.set_item("device_type", format!("{:?}", info.device_type))?;
        Ok(result)
    }
    fn upload_batch_values(
        &mut self,
        py: Python<'_>,
        input: Vec<f32>,
        target: Vec<f32>,
    ) -> PyResult<()> {
        py.detach(|| self.inner.upload_batch(&input, &target))
            .map_err(training_error)
    }
    fn upload_batch(
        &mut self,
        py: Python<'_>,
        input: &PyTensor,
        target: &PyTensor,
    ) -> PyResult<()> {
        for (tensor, layout) in [
            (&input.inner, self.inner.input_layout()),
            (&target.inner, self.inner.output_layout()),
        ] {
            let cols = *layout.shape().last().unwrap();
            if tensor.shape() != (layout.len() / cols, cols) {
                return Err(PyValueError::new_err(
                    "Tensor shape must match the compiled leading-axes matrix",
                ));
            }
        }
        let input = input
            .inner
            .to_layout(st_tensor::Layout::RowMajor)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let target = target
            .inner
            .to_layout(st_tensor::Layout::RowMajor)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        py.detach(|| self.inner.upload_batch(input.data(), target.data()))
            .map_err(training_error)
    }
    /// Enqueued attempt, not proof of acceptance. Read an owning loss/state snapshot.
    fn step(&mut self, py: Python<'_>, learning_rate: &Bound<'_, PyAny>) -> PyResult<u64> {
        if learning_rate.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(
                "learning_rate must be a number, not bool",
            ));
        }
        let learning_rate = learning_rate.extract::<f32>()?;
        py.detach(|| self.inner.step(learning_rate))
            .map_err(training_error)
    }
    fn loss_snapshot(&self, py: Python<'_>) -> PyResult<PyTrainingLossSnapshot> {
        let inner = py
            .detach(|| self.inner.loss_snapshot())
            .map_err(training_error)?;
        Ok(PyTrainingLossSnapshot {
            step: inner.submitted_step(),
            generation: inner.batch_generation(),
            inner: Some(inner),
        })
    }
    fn state_snapshot(&self, py: Python<'_>) -> PyResult<PyGraphTrainingSnapshot> {
        let inner = py
            .detach(|| self.inner.state_snapshot())
            .map_err(training_error)?;
        Ok(PyGraphTrainingSnapshot {
            step: inner.submitted_step(),
            generation: inner.batch_generation(),
            input_shape: inner.input_layout().shape().to_vec(),
            output_shape: inner.output_layout().shape().to_vec(),
            policy: inner.gradient_policy().as_str(),
            inner: Some(inner),
        })
    }
    fn parameter_snapshot(&self, py: Python<'_>) -> PyResult<PyGraphTrainingParametersSnapshot> {
        Ok(PyGraphTrainingParametersSnapshot {
            inner: Some(
                py.detach(|| self.inner.parameter_snapshot())
                    .map_err(training_error)?,
            ),
        })
    }
}

#[pyclass(name = "GraphTrainingSnapshot", module = "spiraltorch.nn")]
struct PyGraphTrainingSnapshot {
    #[cfg(feature = "wgpu")]
    inner: Option<backend::GraphStateReadback>,
    #[cfg(feature = "wgpu")]
    step: u64,
    #[cfg(feature = "wgpu")]
    generation: u64,
    #[cfg(feature = "wgpu")]
    input_shape: Vec<usize>,
    #[cfg(feature = "wgpu")]
    output_shape: Vec<usize>,
    #[cfg(feature = "wgpu")]
    policy: &'static str,
}

#[cfg(feature = "wgpu")]
#[pymethods]
impl PyGraphTrainingSnapshot {
    #[getter]
    fn submitted_step(&self) -> u64 {
        self.step
    }
    #[getter]
    fn batch_generation(&self) -> u64 {
        self.generation
    }
    #[getter]
    fn gradient_policy(&self) -> &'static str {
        self.policy
    }
    #[getter]
    fn input_shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.input_shape.iter().copied())
    }
    #[getter]
    fn output_shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.output_shape.iter().copied())
    }
    fn read_state(&mut self, py: Python<'_>) -> PyResult<PyGraphTrainingState> {
        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("snapshot has already been consumed"))?;
        Ok(PyGraphTrainingState {
            inner: py.detach(|| inner.read()).map_err(training_error)?,
        })
    }
}

#[pyclass(name = "GraphTrainingParametersSnapshot", module = "spiraltorch.nn")]
struct PyGraphTrainingParametersSnapshot {
    #[cfg(feature = "wgpu")]
    inner: Option<backend::GraphParameterReadback>,
}

#[cfg(feature = "wgpu")]
#[pymethods]
impl PyGraphTrainingParametersSnapshot {
    fn read_plan(&mut self, py: Python<'_>) -> PyResult<PyInferencePlan> {
        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("snapshot has already been consumed"))?;
        py.detach(|| {
            let graph = inner.read().map_err(training_error)?;
            Ok(PyInferencePlan {
                inner: InferencePlan::from_graph_definition(graph).map_err(plan_error)?,
            })
        })
    }
}

#[pyclass(name = "GraphTrainingState", module = "spiraltorch.nn", frozen)]
struct PyGraphTrainingState {
    #[cfg(feature = "wgpu")]
    inner: backend::GraphState,
}

#[cfg(feature = "wgpu")]
impl PyGraphTrainingState {
    fn parameter_id(&self, id: &Bound<'_, PyAny>) -> PyResult<usize> {
        if id.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(
                "parameter must be an integer, not bool",
            ));
        }
        let index = id.extract::<usize>()?;
        if index >= self.inner.graph.parameters().len() {
            return Err(PyIndexError::new_err("parameter out of range"));
        }
        Ok(index)
    }
}

#[cfg(feature = "wgpu")]
#[pymethods]
impl PyGraphTrainingState {
    #[getter]
    fn loss(&self) -> f32 {
        self.inner.loss
    }
    #[getter]
    fn submitted_step(&self) -> u64 {
        self.inner.submitted_step
    }
    #[getter]
    fn batch_generation(&self) -> u64 {
        self.inner.batch_generation
    }
    #[getter]
    fn gradient_policy(&self) -> &'static str {
        self.inner.gradient_policy.as_str()
    }
    #[getter]
    fn stage_count(&self) -> usize {
        self.inner.graph.stages().len()
    }
    #[getter]
    fn parameter_count(&self) -> usize {
        self.inner.graph.parameters().len()
    }
    #[getter]
    fn input_shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.graph.input_layout().shape().iter().copied())
    }
    #[getter]
    fn output_shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.graph.output_layout().shape().iter().copied())
    }
    fn prediction_values(&self) -> Vec<f32> {
        self.inner.prediction.clone()
    }
    fn input_gradient_values(&self) -> Vec<f32> {
        self.inner.input_gradient.clone()
    }
    fn parameter_role(&self, parameter: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        Ok(self.inner.graph.parameters()[self.parameter_id(parameter)?]
            .role
            .as_str())
    }
    fn parameter_shape<'py>(
        &self,
        py: Python<'py>,
        parameter: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            self.inner.graph.parameters()[self.parameter_id(parameter)?]
                .shape
                .iter()
                .copied(),
        )
    }
    fn parameter_values(&self, parameter: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
        Ok(self.inner.graph.parameters()[self.parameter_id(parameter)?]
            .values
            .clone())
    }
    fn parameter_gradient_values(&self, parameter: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
        Ok(self.inner.raw_gradients[self.parameter_id(parameter)?].clone())
    }
    fn effective_gradient_values(&self, parameter: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
        Ok(self.inner.effective_gradients[self.parameter_id(parameter)?].clone())
    }
    /// Weight-only portable v2 plan. Runtime counters, batch and policy are not serialized.
    fn to_plan(&self, py: Python<'_>) -> PyResult<PyInferencePlan> {
        py.detach(|| {
            Ok(PyInferencePlan {
                inner: InferencePlan::from_graph_definition(self.inner.graph.clone())
                    .map_err(plan_error)?,
            })
        })
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyResidentGraphTraining>()?;
    module.add_class::<PyGraphTrainingSnapshot>()?;
    module.add_class::<PyGraphTrainingParametersSnapshot>()?;
    module.add_class::<PyGraphTrainingState>()?;
    Ok(())
}
