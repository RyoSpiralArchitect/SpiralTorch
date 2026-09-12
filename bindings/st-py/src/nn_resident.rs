//! Python transports existing Rust NN plans and owning GPU snapshots.

use crate::tensor::PyTensor;
#[cfg(feature = "wgpu")]
use pyo3::{exceptions::PyRuntimeError, types::PyDict};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyBool, PyTuple},
};
use st_nn::resident::{InferenceError, InferencePlan, DEFAULT_MAX_PLAN_JSON_BYTES};
use st_tensor::NdLayout;

mod autograd;
mod forward;
mod graph;
mod learner;
mod loss;
pub(crate) use loss::{evaluate_loss, PyResidentLoss};
mod training;

/// Input type selects an explicit host or resident route. Never upload or read
/// back implicitly, and never reinterpret an unsupported module as CPU work.
pub(crate) fn forward_argument(
    module: &dyn st_nn::Module,
    input: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let py = input.py();
    if let Ok(input) = input.extract::<PyRef<'_, PyTensor>>() {
        let output = module
            .forward(&input.inner)
            .map_err(crate::tensor::tensor_err_to_py)?;
        return Ok(Py::new(py, PyTensor::from_tensor(output))?.into_any());
    }
    #[cfg(feature = "wgpu")]
    if let Ok(input) = input.extract::<PyRef<'_, crate::wgpu_tensor::PyWgpuTensor>>() {
        let inner = module.forward_resident(&input.inner).map_err(plan_error)?;
        return Ok(Py::new(py, crate::wgpu_tensor::PyWgpuTensor { inner })?.into_any());
    }
    Err(PyTypeError::new_err(
        "expected Tensor or WgpuTensor; no implicit device transfer",
    ))
}

/// Explicit terminal capture; never accepts a host tensor or uploads implicitly.
pub(crate) fn snapshot_argument(
    module: &dyn st_nn::Module,
    input: &Bound<'_, PyAny>,
) -> PyResult<crate::wgpu_tensor::PyWgpuTensorSnapshot> {
    #[cfg(feature = "wgpu")]
    {
        let input = input
            .extract::<PyRef<'_, crate::wgpu_tensor::PyWgpuTensor>>()
            .map_err(|_| {
                PyTypeError::new_err("expected WgpuTensor; no implicit device transfer")
            })?;
        let inner = module
            .forward_resident_snapshot(&input.inner)
            .map_err(plan_error)?;
        Ok(crate::wgpu_tensor::PyWgpuTensorSnapshot::from_readback(
            inner,
        ))
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let _ = (module, input);
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "requires the wgpu feature",
        ))
    }
}

pub(crate) fn cache_info(module: &dyn st_nn::Module, py: Python<'_>) -> PyResult<Py<PyAny>> {
    #[cfg(feature = "wgpu")]
    {
        let stats = module
            .resident_forward_stats()
            .ok_or_else(|| PyValueError::new_err("module has no resident cache"))?;
        let info = PyDict::new(py);
        info.set_item("compilations", stats.compilations)?;
        info.set_item("cache_hits", stats.cache_hits)?;
        info.set_item("submitted_forwards", stats.submitted_forwards)?;
        Ok(info.into_any().unbind())
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let _ = (module, py);
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "requires the wgpu feature",
        ))
    }
}

pub(crate) fn clear_cache(module: &dyn st_nn::Module) -> PyResult<()> {
    #[cfg(feature = "wgpu")]
    {
        module.clear_resident_forward_cache();
        Ok(())
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let _ = module;
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "requires the wgpu feature",
        ))
    }
}

fn plan_error(error: InferenceError) -> PyErr {
    #[cfg(feature = "wgpu")]
    match error {
        InferenceError::Gpu(error) => return gpu_error(error),
        InferenceError::Training(error) => return training::training_error(error),
        InferenceError::GraphGpu(error) => return forward::error(error),
        InferenceError::ResidentTensor(error) => return crate::wgpu_tensor::error(error),
        _ => {}
    }
    PyValueError::new_err(error.to_string())
}

#[cfg(feature = "wgpu")]
fn gpu_options(
    tile_mnk: Option<&Bound<'_, PyAny>>,
    kernel: &str,
    accumulation: &str,
) -> PyResult<(
    st_backend_wgpu::resident_matmul::MatmulTile,
    st_backend_wgpu::resident_matmul::MatmulKernel,
    st_backend_wgpu::resident_matmul::MatmulAccumulation,
)> {
    use st_backend_wgpu::resident_matmul::MatmulTile;
    let tile = if let Some(tile) = tile_mnk {
        let dimensions: Vec<u32> = tile
            .try_iter()?
            .map(|item| {
                let item = item?;
                if item.is_instance_of::<PyBool>() {
                    return Err(PyTypeError::new_err(
                        "tile dimensions must be integers, not bool",
                    ));
                }
                item.extract::<u32>()
            })
            .collect::<PyResult<_>>()?;
        if dimensions.len() != 3 {
            return Err(PyValueError::new_err("tile_mnk must have three dimensions"));
        }
        MatmulTile::new(dimensions[0], dimensions[1], dimensions[2])
            .map_err(|err| PyValueError::new_err(err.to_string()))?
    } else {
        MatmulTile::default()
    };
    Ok((
        tile,
        kernel.parse().map_err(PyValueError::new_err)?,
        accumulation.parse().map_err(PyValueError::new_err)?,
    ))
}

fn input_layout(value: &Bound<'_, PyAny>) -> PyResult<NdLayout> {
    let shape: Vec<usize> = value
        .try_iter()?
        .map(|item| {
            let item = item?;
            if item.is_instance_of::<PyBool>() {
                return Err(PyTypeError::new_err(
                    "input dimensions must be integers, not bool",
                ));
            }
            item.extract::<usize>()
        })
        .collect::<PyResult<_>>()?;
    NdLayout::contiguous(&shape).map_err(|err| PyValueError::new_err(err.to_string()))
}

pub(crate) fn plan_for(
    module: &impl st_nn::module::Module,
    shape: &Bound<'_, PyAny>,
) -> PyResult<PyInferencePlan> {
    Ok(PyInferencePlan {
        inner: InferencePlan::from_module(module, input_layout(shape)?).map_err(plan_error)?,
    })
}

#[cfg(feature = "wgpu")]
fn require_dense(plan: &InferencePlan) -> PyResult<()> {
    if !plan.is_dense() {
        return Err(plan_error(InferenceError::RequiresGraph));
    }
    Ok(())
}

#[pyclass(name = "InferencePlan", module = "spiraltorch.nn")]
pub(crate) struct PyInferencePlan {
    inner: InferencePlan,
}

#[pymethods]
impl PyInferencePlan {
    /// Explicitly hand learned weights back to the baseline-matching source model.
    #[pyo3(signature = (module, updated, *, optimizer_state="reject"))]
    fn apply_parameters_to(
        &self,
        module: &Bound<'_, pyo3::types::PyAny>,
        updated: &PyInferencePlan,
        optimizer_state: &str,
    ) -> PyResult<usize> {
        let policy = optimizer_state.parse().map_err(plan_error)?;
        crate::nn::with_module_mut(module, |model| {
            Ok(self
                .inner
                .apply_parameters_to(model, &updated.inner, policy))
        })?
        .map_err(plan_error)
    }

    #[staticmethod]
    #[pyo3(signature = (payload, *, max_bytes=DEFAULT_MAX_PLAN_JSON_BYTES))]
    fn from_json(payload: &str, max_bytes: usize) -> PyResult<Self> {
        Ok(Self {
            inner: InferencePlan::from_json_with_limit(payload, max_bytes).map_err(plan_error)?,
        })
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(plan_error)
    }

    /// Return a new Rust-fused plan; parameter IDs stay fixed, stage IDs may change.
    fn fuse_pointwise(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.fuse_pointwise().map_err(plan_error)?,
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
    fn source_operation_count(&self) -> usize {
        self.inner.source_operation_count()
    }

    #[getter]
    fn is_dense(&self) -> bool {
        self.inner.is_dense()
    }

    /// Compile this fixed parameter snapshot; source model updates are not followed.
    #[pyo3(signature = (*, tile_mnk=None, kernel="scalar", accumulation="sequential"))]
    fn compile_wgpu(
        &self,
        py: Python<'_>,
        tile_mnk: Option<&Bound<'_, PyAny>>,
        kernel: &str,
        accumulation: &str,
    ) -> PyResult<PyResidentInference> {
        #[cfg(feature = "wgpu")]
        {
            use st_backend_wgpu::runtime;
            require_dense(&self.inner)?;
            let (tile, kernel, accumulation) = gpu_options(tile_mnk, kernel, accumulation)?;
            let plan = self.inner.clone();
            py.detach(move || {
                let (runtime, _) = runtime::ensure_default_runtime_blocking("python.nn.resident")
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                Ok(PyResidentInference {
                    inner: plan
                        .compile_wgpu_with_options(runtime, tile, kernel, accumulation)
                        .map_err(plan_error)?,
                })
            })
        }
        #[cfg(not(feature = "wgpu"))]
        {
            let _ = (py, tile_mnk, kernel, accumulation);
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "resident inference requires a wheel built with the 'wgpu' feature",
            ))
        }
    }

    #[pyo3(signature = (*, tile_mnk=None, kernel="scalar", accumulation="sequential"))]
    fn compile_training_wgpu(
        &self,
        py: Python<'_>,
        tile_mnk: Option<&Bound<'_, PyAny>>,
        kernel: &str,
        accumulation: &str,
    ) -> PyResult<training::PyResidentTraining> {
        training::compile(&self.inner, py, tile_mnk, kernel, accumulation)
    }

    #[pyo3(signature = (*, tile_mnk=None, kernel="scalar", accumulation="sequential"))]
    fn compile_graph_autograd_wgpu(
        &self,
        py: Python<'_>,
        tile_mnk: Option<&Bound<'_, PyAny>>,
        kernel: &str,
        accumulation: &str,
    ) -> PyResult<autograd::PyResidentGraphAutograd> {
        autograd::compile(&self.inner, py, tile_mnk, kernel, accumulation)
    }

    #[pyo3(signature = (*, tile_mnk=None, kernel="scalar", accumulation="sequential"))]
    fn compile_graph_wgpu(
        &self,
        py: Python<'_>,
        tile_mnk: Option<&Bound<'_, PyAny>>,
        kernel: &str,
        accumulation: &str,
    ) -> PyResult<forward::PyResidentGraphInference> {
        forward::compile(&self.inner, py, tile_mnk, kernel, accumulation)
    }

    #[pyo3(signature = (*, gradient_policy, tile_mnk=None, kernel="scalar", accumulation="sequential"))]
    fn compile_graph_learner_wgpu(
        &self,
        py: Python<'_>,
        gradient_policy: &str,
        tile_mnk: Option<&Bound<'_, PyAny>>,
        kernel: &str,
        accumulation: &str,
    ) -> PyResult<learner::PyResidentGraphLearner> {
        learner::compile(
            &self.inner,
            py,
            gradient_policy,
            tile_mnk,
            kernel,
            accumulation,
        )
    }

    #[pyo3(signature = (*, gradient_policy, tile_mnk=None, kernel="scalar", accumulation="sequential"))]
    fn compile_graph_training_wgpu(
        &self,
        py: Python<'_>,
        gradient_policy: &str,
        tile_mnk: Option<&Bound<'_, PyAny>>,
        kernel: &str,
        accumulation: &str,
    ) -> PyResult<graph::PyResidentGraphTraining> {
        graph::compile(
            &self.inner,
            py,
            gradient_policy,
            tile_mnk,
            kernel,
            accumulation,
        )
    }
}

#[pyclass(name = "ResidentInference", module = "spiraltorch.nn")]
pub(crate) struct PyResidentInference {
    #[cfg(feature = "wgpu")]
    inner: st_backend_wgpu::resident_dense::ResidentDense,
}

#[cfg(feature = "wgpu")]
fn gpu_error(error: st_backend_wgpu::resident_dense::DenseError) -> PyErr {
    use st_backend_wgpu::{resident_dense::DenseError, resident_matmul::MatmulError};
    match error {
        DenseError::Runtime(_) | DenseError::Matmul(MatmulError::Runtime(_)) => {
            PyRuntimeError::new_err(error.to_string())
        }
        _ => PyValueError::new_err(error.to_string()),
    }
}

#[cfg(feature = "wgpu")]
#[pymethods]
impl PyResidentInference {
    fn set_input_tensor(
        &mut self,
        py: Python<'_>,
        input: &crate::wgpu_tensor::PyWgpuTensor,
    ) -> PyResult<()> {
        py.detach(|| self.inner.set_input_tensor(&input.inner))
            .map_err(gpu_error)
    }
    fn tensor_snapshot(
        &self,
        py: Python<'_>,
        device: &crate::wgpu_tensor::PyWgpuTensorDevice,
    ) -> PyResult<crate::wgpu_tensor::PyWgpuTensor> {
        Ok(crate::wgpu_tensor::PyWgpuTensor {
            inner: py
                .detach(|| self.inner.tensor_snapshot(&device.inner))
                .map_err(gpu_error)?,
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
    fn generation(&self) -> u64 {
        self.inner.generation()
    }
    #[getter]
    fn stage_count(&self) -> usize {
        self.inner.stage_count()
    }

    fn adapter_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let info = self.inner.adapter_info();
        let dict = PyDict::new(py);
        dict.set_item("name", &info.name)?;
        dict.set_item("backend", format!("{:?}", info.backend))?;
        dict.set_item("device_type", format!("{:?}", info.device_type))?;
        Ok(dict)
    }

    fn upload_values(&mut self, py: Python<'_>, values: Vec<f32>) -> PyResult<()> {
        py.detach(|| self.inner.upload(&values)).map_err(gpu_error)
    }

    /// Tensor input must be the exact flattened leading-axes matrix, in logical row-major order.
    fn upload(&mut self, py: Python<'_>, input: &PyTensor) -> PyResult<()> {
        let layout = self.inner.input_layout();
        let cols = *layout.shape().last().unwrap();
        if input.inner.shape() != (layout.len() / cols, cols) {
            return Err(PyValueError::new_err(
                "Tensor shape must match the compiled leading-axes matrix",
            ));
        }
        let input = input
            .inner
            .to_layout(st_tensor::Layout::RowMajor)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        py.detach(|| self.inner.upload(input.data()))
            .map_err(gpu_error)
    }

    fn dispatch(&mut self, py: Python<'_>) -> PyResult<u64> {
        py.detach(|| self.inner.dispatch()).map_err(gpu_error)
    }

    fn snapshot(&self, py: Python<'_>) -> PyResult<PyInferenceSnapshot> {
        let snapshot = py.detach(|| self.inner.snapshot()).map_err(gpu_error)?;
        Ok(PyInferenceSnapshot {
            shape: snapshot.layout().shape().to_vec(),
            generation: snapshot.generation(),
            inner: Some(snapshot),
        })
    }

    /// Ordinary 2D Tensor convenience; N-D clients keep shape on the owned snapshot.
    fn forward(&mut self, py: Python<'_>, input: &PyTensor) -> PyResult<PyTensor> {
        if self.inner.input_layout().rank() != 2 {
            return Err(PyValueError::new_err("N-D plans use upload/dispatch/snapshot/read_values; forward returns a 2D Tensor only"));
        }
        self.upload(py, input)?;
        self.dispatch(py)?;
        self.snapshot(py)?.read_tensor(py)
    }

    fn __call__(&mut self, py: Python<'_>, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(py, input)
    }
}

#[pyclass(name = "InferenceSnapshot", module = "spiraltorch.nn")]
pub(crate) struct PyInferenceSnapshot {
    #[cfg(feature = "wgpu")]
    inner: Option<st_backend_wgpu::resident_dense::DenseReadback>,
    #[cfg(feature = "wgpu")]
    shape: Vec<usize>,
    #[cfg(feature = "wgpu")]
    generation: u64,
}

#[cfg(feature = "wgpu")]
#[pymethods]
impl PyInferenceSnapshot {
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.shape.iter().copied())
    }
    #[getter]
    fn generation(&self) -> u64 {
        self.generation
    }

    /// Consume exactly this snapshot, including its deferred device validation.
    fn read_values(&mut self, py: Python<'_>) -> PyResult<Vec<f32>> {
        let snapshot = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("snapshot has already been consumed"))?;
        py.detach(|| snapshot.read()).map_err(gpu_error)
    }

    fn read_tensor(&mut self, py: Python<'_>) -> PyResult<PyTensor> {
        if self.shape.len() != 2 {
            return Err(PyValueError::new_err(
                "N-D snapshots use read_values with the original shape; no implicit 2D reshape",
            ));
        }
        let values = self.read_values(py)?;
        Ok(PyTensor::from_tensor(
            st_tensor::Tensor::from_vec(self.shape[0], self.shape[1], values)
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?,
        ))
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyInferencePlan>()?;
    module.add_class::<PyResidentInference>()?;
    module.add_class::<PyInferenceSnapshot>()?;
    training::register(module)?;
    graph::register(module)?;
    forward::register(module)?;
    autograd::register(module)?;
    learner::register(module)?;
    loss::register(module)?;
    Ok(())
}
