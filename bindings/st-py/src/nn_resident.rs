//! Python transports existing Rust NN plans and owning GPU snapshots.

#[cfg(feature = "wgpu")]
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

mod training;

fn plan_error(error: InferenceError) -> PyErr {
    #[cfg(feature = "wgpu")]
    match error {
        InferenceError::Gpu(error) => return gpu_error(error),
        InferenceError::Training(error) => return training::training_error(error),
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
        inner: dense_plan(
            InferencePlan::from_module(module, input_layout(shape)?).map_err(plan_error)?,
        )?,
    })
}

// This facade has only dense executors until the general graph wrapper lands.
fn dense_plan(plan: InferencePlan) -> PyResult<InferencePlan> {
    if !plan.is_dense() {
        return Err(plan_error(InferenceError::RequiresGraph));
    }
    Ok(plan)
}

#[pyclass(name = "InferencePlan", module = "spiraltorch.nn")]
pub(crate) struct PyInferencePlan {
    inner: InferencePlan,
}

#[pymethods]
impl PyInferencePlan {
    #[staticmethod]
    #[pyo3(signature = (payload, *, max_bytes=DEFAULT_MAX_PLAN_JSON_BYTES))]
    fn from_json(payload: &str, max_bytes: usize) -> PyResult<Self> {
        Ok(Self {
            inner: dense_plan(
                InferencePlan::from_json_with_limit(payload, max_bytes).map_err(plan_error)?,
            )?,
        })
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(plan_error)
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
    Ok(())
}
