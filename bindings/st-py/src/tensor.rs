use std::ffi::{c_void, CStr};

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule};
use pyo3::wrap_pyfunction;
use pyo3::Bound;
use st_tensor::dlpack::{drop_exported_state, DLManagedTensor, DLPACK_CAPSULE_NAME};
use st_tensor::{Tensor, TensorError};

const USED_DLPACK_CAPSULE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"used_dltensor\0") };

#[pyclass(module = "spiraltorch")]
#[derive(Clone)]
pub(crate) struct PyTensor {
    pub(crate) inner: Tensor,
}

impl PyTensor {
    pub(crate) fn from_tensor(tensor: Tensor) -> Self {
        Self { inner: tensor }
    }

    #[cfg(test)]
    pub(crate) fn inner(&self) -> &Tensor {
        &self.inner
    }
}

#[pymethods]
impl PyTensor {
    #[new]
    #[pyo3(signature = (rows, cols, data=None))]
    fn new(rows: usize, cols: usize, data: Option<Vec<f32>>) -> PyResult<Self> {
        let tensor = match data {
            Some(buffer) => Tensor::from_vec(rows, cols, buffer),
            None => Tensor::zeros(rows, cols),
        }
        .map_err(tensor_err_to_py)?;
        Ok(Self { inner: tensor })
    }

    #[staticmethod]
    pub fn zeros(rows: usize, cols: usize) -> PyResult<Self> {
        let tensor = Tensor::zeros(rows, cols).map_err(tensor_err_to_py)?;
        Ok(Self { inner: tensor })
    }

    #[staticmethod]
    pub fn from_dlpack(py: Python<'_>, capsule: PyObject) -> PyResult<Self> {
        let capsule = capsule.bind(py);
        from_dlpack_impl(py, &capsule)
    }

    pub fn to_dlpack(&self, py: Python<'_>) -> PyResult<PyObject> {
        to_dlpack_impl(py, &self.inner)
    }

    #[pyo3(signature = (*, stream=None))]
    pub fn __dlpack__(&self, py: Python<'_>, stream: Option<PyObject>) -> PyResult<PyObject> {
        let _ = stream;
        self.to_dlpack(py)
    }

    pub fn __dlpack_device__(&self) -> (i32, i32) {
        (1, 0)
    }

    #[getter]
    fn rows(&self) -> usize {
        self.inner.shape().0
    }

    #[getter]
    fn cols(&self) -> usize {
        self.inner.shape().1
    }

    pub fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    pub fn tolist(&self) -> Vec<Vec<f32>> {
        let (rows, cols) = self.inner.shape();
        let data = self.inner.data();
        let mut out = Vec::with_capacity(rows);
        for r in 0..rows {
            let start = r * cols;
            out.push(data[start..start + cols].to_vec());
        }
        out
    }
}

#[pyfunction]
#[pyo3(name = "from_dlpack")]
fn tensor_from_dlpack(py: Python<'_>, capsule: PyObject) -> PyResult<PyTensor> {
    PyTensor::from_dlpack(py, capsule)
}

#[pyfunction]
#[pyo3(name = "to_dlpack")]
fn tensor_to_dlpack(py: Python<'_>, tensor: &PyTensor) -> PyResult<PyObject> {
    tensor.to_dlpack(py)
}

pub(crate) fn register(py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_function(wrap_pyfunction!(tensor_from_dlpack, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_to_dlpack, m)?)?;
    m.add("__doc__", "Tensor helpers and DLPack interop.")?;
    let _ = py;
    Ok(())
}

pub(crate) fn tensor_err_to_py(err: TensorError) -> PyErr {
    match err {
        TensorError::InvalidDimensions { .. }
        | TensorError::DataLength { .. }
        | TensorError::EmptyInput(_)
        | TensorError::DlpackError { .. } => PyValueError::new_err(err.to_string()),
        _ => PyRuntimeError::new_err(err.to_string()),
    }
}

fn to_dlpack_impl(py: Python<'_>, tensor: &Tensor) -> PyResult<PyObject> {
    let managed = tensor.to_dlpack().map_err(tensor_err_to_py)?;
    unsafe {
        extern "C" fn drop_capsule(ptr: *mut ffi::PyObject) {
            unsafe {
                let tensor_ptr = ffi::PyCapsule_GetPointer(ptr, DLPACK_CAPSULE_NAME.as_ptr());
                if tensor_ptr.is_null() {
                    ffi::PyErr_Clear();
                    return;
                }
                drop_exported_state(tensor_ptr as *mut DLManagedTensor);
            }
        }

        let capsule = ffi::PyCapsule_New(
            managed as *mut c_void,
            DLPACK_CAPSULE_NAME.as_ptr(),
            Some(drop_capsule),
        );
        if capsule.is_null() {
            drop_exported_state(managed);
            return Err(PyErr::fetch(py));
        }
        Ok(PyObject::from_owned_ptr(py, capsule))
    }
}

fn from_dlpack_impl(py: Python<'_>, capsule: &Bound<PyAny>) -> PyResult<PyTensor> {
    let owned_capsule = ensure_dlpack_capsule(py, capsule)?;
    let capsule_ref = owned_capsule.bind(py);

    unsafe {
        let managed_ptr =
            ffi::PyCapsule_GetPointer(capsule_ref.as_ptr(), DLPACK_CAPSULE_NAME.as_ptr());
        if managed_ptr.is_null() {
            return Err(PyErr::fetch(py));
        }

        if ffi::PyCapsule_SetName(capsule_ref.as_ptr(), USED_DLPACK_CAPSULE_NAME.as_ptr()) != 0 {
            return Err(PyErr::fetch(py));
        }
        if ffi::PyCapsule_SetDestructor(capsule_ref.as_ptr(), None) != 0 {
            return Err(PyErr::fetch(py));
        }

        let tensor =
            Tensor::from_dlpack(managed_ptr as *mut DLManagedTensor).map_err(tensor_err_to_py)?;
        Ok(PyTensor::from_tensor(tensor))
    }
}

fn ensure_dlpack_capsule(py: Python<'_>, candidate: &Bound<PyAny>) -> PyResult<Py<PyAny>> {
    unsafe {
        if ffi::PyCapsule_CheckExact(candidate.as_ptr()) == 1 {
            return Ok(candidate.clone().unbind());
        }
    }

    if let Ok(method) = candidate.getattr("__dlpack__") {
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("stream", py.None())?;
        let capsule = method.call((), Some(&kwargs))?;
        unsafe {
            if ffi::PyCapsule_CheckExact(capsule.as_ptr()) == 1 {
                return Ok(capsule.into());
            }
        }
        return Err(PyTypeError::new_err(
            "__dlpack__ did not return a DLPack capsule",
        ));
    }

    Err(PyTypeError::new_err(
        "expected a DLPack capsule or object implementing __dlpack__",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dlpack_roundtrip_shares_buffer() {
        Python::with_gil(|py| {
            let tensor = PyTensor::new(2, 3, Some(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])).unwrap();
            let capsule = tensor.to_dlpack(py).unwrap();
            let restored = PyTensor::from_dlpack(py, capsule).unwrap();
            assert_eq!(
                tensor.inner().data().as_ptr(),
                restored.inner().data().as_ptr()
            );
        });
    }

    #[test]
    fn dlpack_device_reports_cpu() {
        Python::with_gil(|py| {
            let tensor = PyTensor::zeros(1, 1).unwrap();
            let device = tensor.__dlpack_device__();
            assert_eq!(device, (1, 0));
            let _ = py;
        });
    }
}
