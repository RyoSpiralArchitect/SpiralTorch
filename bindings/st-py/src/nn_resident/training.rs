//! Owning Python handles, not Python loss/optimizer semantics.
use super::*;
#[cfg(feature = "wgpu")]
use pyo3::exceptions::PyIndexError;
#[cfg(feature = "wgpu")]
use st_backend_wgpu::{resident_training as backend, runtime};

pub(super) fn compile(
    plan: &InferencePlan,
    py: Python<'_>,
    tile: Option<&Bound<'_, PyAny>>,
    kernel: &str,
    accumulation: &str,
) -> PyResult<PyResidentTraining> {
    #[cfg(feature = "wgpu")]
    {
        require_dense(plan)?;
        let (tile, kernel, accumulation) = gpu_options(tile, kernel, accumulation)?;
        let plan = plan.clone();
        py.detach(move || {
            let (runtime, _) = runtime::ensure_default_runtime_blocking("python.nn.training")
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let inner = plan
                .compile_training_wgpu_with_options(runtime, tile, kernel, accumulation)
                .map_err(plan_error)?;
            Ok(PyResidentTraining { inner, plan })
        })
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let _ = (plan, py, tile, kernel, accumulation);
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "resident training requires a wheel built with the 'wgpu' feature",
        ))
    }
}

#[cfg(feature = "wgpu")]
pub(super) fn training_error(error: backend::TrainingError) -> PyErr {
    use st_backend_wgpu::resident_matmul::MatmulError;
    match error {
        backend::TrainingError::Dense(error) => gpu_error(error),
        backend::TrainingError::Runtime(_)
        | backend::TrainingError::Matmul(MatmulError::Runtime(_)) => {
            PyRuntimeError::new_err(error.to_string())
        }
        backend::TrainingError::Rejected { stage, flags } => {
            let exception = PyValueError::new_err(error.to_string());
            let decorated: PyResult<()> = Python::attach(|py| {
                let value = exception.value(py);
                value.setattr("stage", stage)?;
                value.setattr("flags", flags)?;
                value.setattr("code", "training_step_rejected")
            });
            decorated.err().unwrap_or(exception)
        }
        _ => PyValueError::new_err(error.to_string()),
    }
}

#[pyclass(name = "ResidentTraining", module = "spiraltorch.nn")]
pub(super) struct PyResidentTraining {
    #[cfg(feature = "wgpu")]
    inner: backend::ResidentDenseTraining,
    #[cfg(feature = "wgpu")]
    plan: InferencePlan,
}

#[cfg(feature = "wgpu")]
#[pymethods]
impl PyResidentTraining {
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

    /// Enqueue only. Reading its snapshot proves numerical acceptance.
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

    fn state_snapshot(&self, py: Python<'_>) -> PyResult<PyTrainingSnapshot> {
        let inner = py
            .detach(|| self.inner.state_snapshot())
            .map_err(training_error)?;
        Ok(PyTrainingSnapshot {
            step: inner.submitted_step(),
            generation: inner.batch_generation(),
            input_shape: inner.input_layout().shape().to_vec(),
            output_shape: inner.output_layout().shape().to_vec(),
            inner: Some(inner),
            plan: self.plan.clone(),
        })
    }

    fn parameter_snapshot(&self, py: Python<'_>) -> PyResult<PyTrainingParametersSnapshot> {
        Ok(PyTrainingParametersSnapshot {
            inner: Some(
                py.detach(|| self.inner.parameter_snapshot())
                    .map_err(training_error)?,
            ),
            plan: self.plan.clone(),
        })
    }
}

#[pyclass(name = "TrainingLossSnapshot", module = "spiraltorch.nn")]
pub(super) struct PyTrainingLossSnapshot {
    #[cfg(feature = "wgpu")]
    pub(super) inner: Option<backend::StepReadback>,
    #[cfg(feature = "wgpu")]
    pub(super) step: u64,
    #[cfg(feature = "wgpu")]
    pub(super) generation: u64,
}

#[cfg(feature = "wgpu")]
#[pymethods]
impl PyTrainingLossSnapshot {
    #[getter]
    fn submitted_step(&self) -> u64 {
        self.step
    }
    #[getter]
    fn batch_generation(&self) -> u64 {
        self.generation
    }

    fn read(&mut self, py: Python<'_>) -> PyResult<f32> {
        let snapshot = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("snapshot has already been consumed"))?;
        py.detach(|| snapshot.read()).map_err(training_error)
    }
}

#[pyclass(name = "TrainingSnapshot", module = "spiraltorch.nn")]
struct PyTrainingSnapshot {
    #[cfg(feature = "wgpu")]
    inner: Option<backend::TrainingStateReadback>,
    #[cfg(feature = "wgpu")]
    plan: InferencePlan,
    #[cfg(feature = "wgpu")]
    step: u64,
    #[cfg(feature = "wgpu")]
    generation: u64,
    #[cfg(feature = "wgpu")]
    input_shape: Vec<usize>,
    #[cfg(feature = "wgpu")]
    output_shape: Vec<usize>,
}

#[cfg(feature = "wgpu")]
#[pymethods]
impl PyTrainingSnapshot {
    #[getter]
    fn submitted_step(&self) -> u64 {
        self.step
    }
    #[getter]
    fn batch_generation(&self) -> u64 {
        self.generation
    }
    #[getter]
    fn input_shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.input_shape.iter().copied())
    }
    #[getter]
    fn output_shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.output_shape.iter().copied())
    }
    fn read_state(&mut self, py: Python<'_>) -> PyResult<PyTrainingState> {
        let snapshot = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("snapshot has already been consumed"))?;
        let inner = py.detach(|| snapshot.read()).map_err(training_error)?;
        Ok(PyTrainingState {
            inner,
            plan: self.plan.clone(),
        })
    }
}

#[pyclass(name = "TrainingParametersSnapshot", module = "spiraltorch.nn")]
struct PyTrainingParametersSnapshot {
    #[cfg(feature = "wgpu")]
    inner: Option<backend::ParameterReadback>,
    #[cfg(feature = "wgpu")]
    plan: InferencePlan,
}

#[cfg(feature = "wgpu")]
#[pymethods]
impl PyTrainingParametersSnapshot {
    fn read_plan(&mut self, py: Python<'_>) -> PyResult<PyInferencePlan> {
        let snapshot = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("snapshot has already been consumed"))?;
        let plan = &self.plan;
        let inner = py.detach(|| {
            let layers = snapshot.read().map_err(training_error)?;
            plan.with_dense_parameters(layers).map_err(plan_error)
        })?;
        Ok(PyInferencePlan { inner })
    }
}

#[pyclass(name = "TrainingState", module = "spiraltorch.nn", frozen)]
struct PyTrainingState {
    #[cfg(feature = "wgpu")]
    inner: backend::TrainingState,
    #[cfg(feature = "wgpu")]
    plan: InferencePlan,
}

#[cfg(feature = "wgpu")]
impl PyTrainingState {
    fn stage_index(stage: &Bound<'_, PyAny>) -> PyResult<usize> {
        if stage.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err("stage must be an integer, not bool"));
        }
        stage.extract::<usize>()
    }

    fn parameter_tensor(&self, index: usize, gradient: bool, bias: bool) -> PyResult<PyTensor> {
        let layer = self
            .inner
            .parameters
            .get(index)
            .ok_or_else(|| PyIndexError::new_err("stage out of range"))?;
        let values = if gradient {
            let g = &self.inner.parameter_gradients[index];
            if bias {
                &g.bias
            } else {
                &g.weights
            }
        } else if bias {
            &layer.bias
        } else {
            &layer.weights
        };
        Ok(PyTensor::from_tensor(
            st_tensor::Tensor::from_vec(
                if bias { 1 } else { layer.inner },
                layer.cols,
                values.clone(),
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        ))
    }
}

#[cfg(feature = "wgpu")]
#[pymethods]
impl PyTrainingState {
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
    fn stage_count(&self) -> usize {
        self.inner.parameters.len()
    }
    #[getter]
    fn input_shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.input_layout.shape().iter().copied())
    }
    #[getter]
    fn output_shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.output_layout.shape().iter().copied())
    }
    fn prediction_values(&self) -> Vec<f32> {
        self.inner.prediction.clone()
    }
    fn input_gradient_values(&self) -> Vec<f32> {
        self.inner.input_gradient.clone()
    }
    fn weight(&self, stage: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        self.parameter_tensor(Self::stage_index(stage)?, false, false)
    }
    fn bias(&self, stage: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        self.parameter_tensor(Self::stage_index(stage)?, false, true)
    }
    fn weight_gradient(&self, stage: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        self.parameter_tensor(Self::stage_index(stage)?, true, false)
    }
    fn bias_gradient(&self, stage: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        self.parameter_tensor(Self::stage_index(stage)?, true, true)
    }
    fn to_plan(&self, py: Python<'_>) -> PyResult<PyInferencePlan> {
        let inner = py
            .detach(|| {
                self.plan
                    .with_dense_parameters(self.inner.parameters.clone())
            })
            .map_err(plan_error)?;
        Ok(PyInferencePlan { inner })
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyResidentTraining>()?;
    module.add_class::<PyTrainingLossSnapshot>()?;
    module.add_class::<PyTrainingSnapshot>()?;
    module.add_class::<PyTrainingParametersSnapshot>()?;
    module.add_class::<PyTrainingState>()?;
    Ok(())
}
