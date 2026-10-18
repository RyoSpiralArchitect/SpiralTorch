//! Thin bindings for owned Rust checkpoint ingress and N-D host snapshots.

use crate::channel_affine::reject_bool;
use crate::f32_buffer::write_f32;
use crate::tensor::PyTensor;
use pyo3::exceptions::{PyKeyError, PyMemoryError, PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict};
use st_nn::weights::{SafeTensorArchive, WeightLoadError, WeightLoadLimits};
use st_tensor::NdTensor;
use std::collections::BTreeMap;
use std::path::PathBuf;

fn error(value: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(value.to_string())
}

fn weight_error(value: WeightLoadError) -> PyErr {
    match value {
        WeightLoadError::Missing(name) => PyKeyError::new_err(name),
        WeightLoadError::Allocation(message) => PyMemoryError::new_err(message),
        WeightLoadError::Io(err) => PyOSError::new_err(err.to_string()),
        other => error(other),
    }
}

fn integer(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    reject_bool(value)?;
    value.extract()
}

fn limits(
    file: Option<&Bound<'_, PyAny>>,
    header: Option<&Bound<'_, PyAny>>,
    tensors: Option<&Bound<'_, PyAny>>,
    rank: Option<&Bound<'_, PyAny>>,
    elements: Option<&Bound<'_, PyAny>>,
) -> PyResult<WeightLoadLimits> {
    let default = WeightLoadLimits::default();
    let result = WeightLoadLimits {
        max_file_bytes: file
            .map(integer)
            .transpose()?
            .unwrap_or(default.max_file_bytes),
        max_header_bytes: header
            .map(integer)
            .transpose()?
            .unwrap_or(default.max_header_bytes),
        max_tensors: tensors
            .map(integer)
            .transpose()?
            .unwrap_or(default.max_tensors),
        max_rank: rank.map(integer).transpose()?.unwrap_or(default.max_rank),
        max_tensor_elements: elements
            .map(integer)
            .transpose()?
            .unwrap_or(default.max_tensor_elements),
    };
    result.validate().map_err(weight_error)?;
    Ok(result)
}

fn dimensions(value: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let rank = value.len()?;
    if rank > 128 {
        return Err(error("N-D view rank exceeds 128"));
    }
    (0..rank).map(|i| integer(&value.get_item(i)?)).collect()
}

#[pyclass(name = "SafeTensorArchive", module = "spiraltorch", frozen)]
pub struct PySafeTensorArchive {
    pub(crate) inner: SafeTensorArchive,
}

#[pyclass(name = "NdTensor", module = "spiraltorch", frozen)]
pub struct PyNdTensor {
    pub(crate) inner: NdTensor,
}

#[pymethods]
impl PySafeTensorArchive {
    #[new]
    #[pyo3(signature = (path, *, max_file_bytes=None, max_header_bytes=None, max_tensors=None, max_rank=None, max_tensor_elements=None))]
    fn new(
        py: Python<'_>,
        path: PathBuf,
        max_file_bytes: Option<&Bound<'_, PyAny>>,
        max_header_bytes: Option<&Bound<'_, PyAny>>,
        max_tensors: Option<&Bound<'_, PyAny>>,
        max_rank: Option<&Bound<'_, PyAny>>,
        max_tensor_elements: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let limits = limits(
            max_file_bytes,
            max_header_bytes,
            max_tensors,
            max_rank,
            max_tensor_elements,
        )?;
        py.detach(|| SafeTensorArchive::open(path, limits))
            .map(|inner| Self { inner })
            .map_err(weight_error)
    }

    #[staticmethod]
    #[pyo3(signature = (payload, *, max_file_bytes=None, max_header_bytes=None, max_tensors=None, max_rank=None, max_tensor_elements=None))]
    fn from_bytes(
        py: Python<'_>,
        payload: &Bound<'_, PyBytes>,
        max_file_bytes: Option<&Bound<'_, PyAny>>,
        max_header_bytes: Option<&Bound<'_, PyAny>>,
        max_tensors: Option<&Bound<'_, PyAny>>,
        max_rank: Option<&Bound<'_, PyAny>>,
        max_tensor_elements: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let limits = limits(
            max_file_bytes,
            max_header_bytes,
            max_tensors,
            max_rank,
            max_tensor_elements,
        )?;
        let bytes = payload.as_bytes();
        py.detach(|| SafeTensorArchive::from_bytes(bytes, limits))
            .map(|inner| Self { inner })
            .map_err(weight_error)
    }

    fn names(&self) -> Vec<String> {
        self.inner.names()
    }

    #[getter]
    fn content_id(&self, py: Python<'_>) -> String {
        py.detach(|| self.inner.content_id())
    }

    fn metadata(&self) -> BTreeMap<String, String> {
        self.inner.metadata()
    }

    fn info<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyDict>> {
        let info = self.inner.info(name).map_err(weight_error)?;
        let output = PyDict::new(py);
        output.set_item("dtype", info.dtype)?;
        output.set_item("shape", info.shape)?;
        output.set_item("elements", info.elements)?;
        output.set_item("byte_len", info.byte_len)?;
        Ok(output)
    }

    fn tensor(&self, py: Python<'_>, name: &str) -> PyResult<PyNdTensor> {
        py.detach(|| self.inner.tensor(name))
            .map(|inner| PyNdTensor { inner })
            .map_err(weight_error)
    }
}

#[pymethods]
impl PyNdTensor {
    #[staticmethod]
    fn from_tensor(py: Python<'_>, tensor: &PyTensor) -> PyResult<Self> {
        py.detach(|| NdTensor::from_tensor(&tensor.inner))
            .map(|inner| Self { inner })
            .map_err(error)
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn reshape(&self, shape: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.inner
            .reshape(&dimensions(shape)?)
            .map(|inner| Self { inner })
            .map_err(error)
    }

    fn permute(&self, axes: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.inner
            .permute(&dimensions(axes)?)
            .map(|inner| Self { inner })
            .map_err(error)
    }

    fn values(&self, py: Python<'_>) -> PyResult<Vec<f32>> {
        py.detach(|| self.inner.read_values()).map_err(error)
    }

    fn values_buffer<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyByteArray>> {
        let values = self.values(py)?;
        write_f32(py, values.into_iter())
    }

    fn to_tensor(&self, py: Python<'_>) -> PyResult<PyTensor> {
        py.detach(|| self.inner.to_tensor())
            .map(PyTensor::from_tensor)
            .map_err(error)
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySafeTensorArchive>()?;
    module.add_class::<PyNdTensor>()?;
    Ok(())
}
