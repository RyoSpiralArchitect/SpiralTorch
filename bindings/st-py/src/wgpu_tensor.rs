//! Owning Python handles over Rust's immutable N-D GPU storage.
use pyo3::prelude::*;
#[cfg(feature = "wgpu")]
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    types::{PyBool, PyDict, PyTuple},
};
#[cfg(feature = "wgpu")]
use st_backend_wgpu::{
    resident_tensor::{ResidentTensor, TensorDevice, TensorError, TensorReadback},
    runtime,
};

#[cfg(feature = "wgpu")]
pub(crate) fn error(err: TensorError) -> PyErr {
    match err {
        TensorError::Runtime(_) => PyRuntimeError::new_err(err.to_string()),
        _ => PyValueError::new_err(err.to_string()),
    }
}

#[cfg(feature = "wgpu")]
fn index(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    if value.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "dimension/index must be an integer, not bool",
        ));
    }
    value.extract()
}
#[cfg(feature = "wgpu")]
fn dimensions(value: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    value.try_iter()?.map(|item| index(&item?)).collect()
}

#[pyclass(name = "WgpuTensorDevice", module = "spiraltorch.wgpu")]
pub(crate) struct PyWgpuTensorDevice {
    #[cfg(feature = "wgpu")]
    pub(crate) inner: TensorDevice,
}

#[pymethods]
impl PyWgpuTensorDevice {
    #[staticmethod]
    fn create(py: Python<'_>) -> PyResult<Self> {
        #[cfg(feature = "wgpu")]
        {
            py.detach(|| {
                let (runtime, _) =
                    runtime::ensure_default_runtime_blocking("python.resident.tensor")
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                Ok(Self {
                    inner: TensorDevice::new(runtime).map_err(error)?,
                })
            })
        }
        #[cfg(not(feature = "wgpu"))]
        {
            let _ = py;
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "WgpuTensorDevice requires a wheel built with the 'wgpu' feature",
            ))
        }
    }

    #[cfg(feature = "wgpu")]
    fn upload(
        &self,
        py: Python<'_>,
        shape: &Bound<'_, PyAny>,
        values: Vec<f32>,
    ) -> PyResult<PyWgpuTensor> {
        let shape = dimensions(shape)?;
        Ok(PyWgpuTensor {
            inner: py
                .detach(|| self.inner.upload(&shape, &values))
                .map_err(error)?,
        })
    }

    #[cfg(feature = "wgpu")]
    fn adapter_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let info = self.inner.runtime().adapter_info();
        let result = PyDict::new(py);
        result.set_item("name", &info.name)?;
        result.set_item("backend", format!("{:?}", info.backend))?;
        result.set_item("device_type", format!("{:?}", info.device_type))?;
        Ok(result)
    }
}

#[pyclass(name = "WgpuTensor", module = "spiraltorch.wgpu", frozen)]
pub(crate) struct PyWgpuTensor {
    #[cfg(feature = "wgpu")]
    pub(crate) inner: ResidentTensor,
}

#[cfg(feature = "wgpu")]
#[pymethods]
impl PyWgpuTensor {
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.layout().shape().iter().copied())
    }
    #[getter]
    fn strides<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.layout().strides().iter().copied())
    }
    #[getter]
    fn offset(&self) -> usize {
        self.inner.layout().offset()
    }
    #[getter]
    fn numel(&self) -> usize {
        self.inner.layout().len()
    }
    #[getter]
    fn is_contiguous(&self) -> bool {
        self.inner.layout().is_contiguous()
    }
    fn device(&self) -> PyWgpuTensorDevice {
        PyWgpuTensorDevice {
            inner: self.inner.device().clone(),
        }
    }
    fn shares_storage_with(&self, other: &Self) -> bool {
        self.inner.shares_storage_with(&other.inner)
    }
    fn reshape(&self, shape: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.reshape(&dimensions(shape)?).map_err(error)?,
        })
    }
    fn permute(&self, axes: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.permute(&dimensions(axes)?).map_err(error)?,
        })
    }
    fn broadcast_to(&self, shape: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .broadcast_to(&dimensions(shape)?)
                .map_err(error)?,
        })
    }
    fn narrow(
        &self,
        axis: &Bound<'_, PyAny>,
        start: &Bound<'_, PyAny>,
        length: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .narrow(index(axis)?, index(start)?, index(length)?)
                .map_err(error)?,
        })
    }
    fn contiguous(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self {
            inner: py.detach(|| self.inner.contiguous()).map_err(error)?,
        })
    }
    fn add(&self, py: Python<'_>, rhs: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: py.detach(|| self.inner.add(&rhs.inner)).map_err(error)?,
        })
    }
    fn mul(&self, py: Python<'_>, rhs: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: py.detach(|| self.inner.mul(&rhs.inner)).map_err(error)?,
        })
    }
    fn relu(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self {
            inner: py.detach(|| self.inner.relu()).map_err(error)?,
        })
    }
    fn gelu(&self, py: Python<'_>) -> PyResult<Self> {
        Ok(Self {
            inner: py.detach(|| self.inner.gelu()).map_err(error)?,
        })
    }
    fn snapshot(&self, py: Python<'_>) -> PyResult<PyWgpuTensorSnapshot> {
        let inner = py.detach(|| self.inner.snapshot()).map_err(error)?;
        Ok(PyWgpuTensorSnapshot {
            shape: inner.layout().shape().to_vec(),
            inner: Some(inner),
        })
    }
}

#[pyclass(name = "WgpuTensorSnapshot", module = "spiraltorch.wgpu")]
pub(crate) struct PyWgpuTensorSnapshot {
    #[cfg(feature = "wgpu")]
    inner: Option<TensorReadback>,
    #[cfg(feature = "wgpu")]
    shape: Vec<usize>,
}
#[cfg(feature = "wgpu")]
#[pymethods]
impl PyWgpuTensorSnapshot {
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.shape.iter().copied())
    }
    fn read_values(&mut self, py: Python<'_>) -> PyResult<Vec<f32>> {
        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("snapshot has already been consumed"))?;
        py.detach(|| inner.read()).map_err(error)
    }
}

pub(crate) fn register(parent: &Bound<'_, PyModule>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    for target in [parent, module] {
        target.add_class::<PyWgpuTensorDevice>()?;
        target.add_class::<PyWgpuTensor>()?;
        target.add_class::<PyWgpuTensorSnapshot>()?;
    }
    Ok(())
}
