
use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::types::{PyAny, PyTuple};
use numpy::{PyReadonlyArrayDyn, PyArrayDyn, PyArray0};
use st_core::{Tensor, DType};

fn map_err(e: st_core::SpiralError) -> PyErr {
    match e {
        st_core::SpiralError::Shape(msg) => PyValueError::new_err(msg),
        st_core::SpiralError::DType(msg) => PyTypeError::new_err(msg),
        st_core::SpiralError::Device(msg) => PyRuntimeError::new_err(msg),
    }
}

fn any_to_tensor<'py>(py: Python<'py>, any: &PyAny) -> PyResult<st_core::Tensor> {
    if let Ok(tref) = any.extract::<PyRef<PyTensor>>() { return Ok(tref.inner.clone()); }
    if let Ok(arr) = any.extract::<PyReadonlyArrayDyn<f32>>() { let a = arr.as_array().to_owned().into_dyn(); return Ok(Tensor::from_array(a)); }
    if let Ok(arr) = any.extract::<PyReadonlyArrayDyn<f64>>() { let a64 = arr.as_array().to_owned().into_dyn(); let a = a64.mapv(|v| v as f32); return Ok(Tensor::from_array(a)); }
    if let Ok(f) = any.extract::<f32>() { let s = PyArray0::<f32>::from_owned_array(py, ndarray::arr0(f)); return Ok(Tensor::from_array(s.to_owned_array().into_dyn())); }
    if let Ok(i) = any.extract::<i64>() { let s = PyArray0::<f32>::from_owned_array(py, ndarray::arr0(i as f32)); return Ok(Tensor::from_array(s.to_owned_array().into_dyn())); }
    Err(PyTypeError::new_err("Unsupported operand: expected PyTensor, ndarray, or number"))
}

#[pyclass(module = "spiraltorch_rs")]
pub struct PyTensor { pub(crate) inner: Tensor }

#[pymethods]
impl PyTensor {
    #[staticmethod]
    pub fn from_f32(py: Python<'_>, arr: PyReadonlyArrayDyn<f32>, requires_grad: bool) -> PyResult<Self> {
        let a = arr.as_array().to_owned().into_dyn();
        let t = Tensor::from_array(a).requires_grad(requires_grad);
        Ok(PyTensor { inner: t })
    }
    #[staticmethod]
    pub fn from_i32(arr: PyReadonlyArrayDyn<i32>) -> PyResult<Self> { Ok(PyTensor { inner: Tensor::from_i32(arr.as_array().to_owned().into_dyn()) }) }
    #[staticmethod]
    pub fn from_bool(arr: PyReadonlyArrayDyn<bool>) -> PyResult<Self> { Ok(PyTensor { inner: Tensor::from_bool(arr.as_array().to_owned().into_dyn()) }) }
    pub fn dtype(&self) -> String { match self.inner.dtype() { DType::F32 => "f32".to_string(), DType::I32 => "i32".to_string(), DType::Bool => "bool".to_string(), } }
    pub fn shape(&self) -> Vec<usize> { self.inner.shape() }
    pub fn ndim(&self) -> usize { self.inner.ndim() }
    pub fn numpy<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        match self.inner.dtype() {
            DType::F32 => Ok(PyArrayDyn::<f32>::from_owned_array(py, self.inner.data()).into_py(py)),
            DType::I32 => Ok(PyArrayDyn::<i32>::from_owned_array(py, self.inner.data_i32()).into_py(py)),
            DType::Bool => Ok(PyArrayDyn::<bool>::from_owned_array(py, self.inner.data_bool()).into_py(py)),
        }
    }
    pub fn grad<'py>(&self, py: Python<'py>) -> PyResult<Option<PyObject>> {
        match self.inner.grad() { Some(g) => Ok(Some(PyArrayDyn::<f32>::from_owned_array(py, g).into_py(py))), None => Ok(None) }
    }
    pub fn backward(&self) -> PyResult<()> { self.inner.backward().map_err(map_err) }
    pub fn backward_with_grad(&self, grad: PyReadonlyArrayDyn<f32>) -> PyResult<()> { self.inner.backward_with_grad(&grad.as_array().to_owned().into_dyn()).map_err(map_err) }
    #[classattr] const __array_priority__: f64 = 1000.0;
    fn __repr__(&self) -> PyResult<String> { Ok(format!("<PyTensor dtype={} shape={:?}>", self.dtype(), self.shape())) }
}

// Index reduce & segments
#[pyfunction]
pub fn index_reduce(base: &PyTensor, dim: isize, index1d: PyReadonlyArrayDyn<i64>, src: &PyTensor, reduce: Option<String>, include_self: Option<bool>) -> PyResult<PyTensor> {
    let idx = Tensor::from_i32(index1d.as_array().to_owned().into_dyn().mapv(|v| v as i32));
    let red = reduce.unwrap_or_else(|| "mean".to_string());
    let inc = include_self.unwrap_or(true);
    Ok(PyTensor { inner: st_core::ops::index_reduce(&base.inner, dim, &idx, &src.inner, &red, inc).map_err(map_err)? })
}

#[pyfunction] pub fn segment_sum(data: &PyTensor, idx: PyReadonlyArrayDyn<i64>, k: usize) -> PyResult<PyTensor> {
    let id = Tensor::from_i32(idx.as_array().to_owned().into_dyn().mapv(|v| v as i32));
    Ok(PyTensor { inner: st_core::ops::segment_sum(&data.inner, &id, k).map_err(map_err)? })
}
#[pyfunction] pub fn segment_mean(data: &PyTensor, idx: PyReadonlyArrayDyn<i64>, k: usize) -> PyResult<PyTensor> {
    let id = Tensor::from_i32(idx.as_array().to_owned().into_dyn().mapv(|v| v as i32));
    Ok(PyTensor { inner: st_core::ops::segment_mean(&data.inner, &id, k).map_err(map_err)? })
}
#[pyfunction] pub fn segment_max(data: &PyTensor, idx: PyReadonlyArrayDyn<i64>, k: usize) -> PyResult<PyTensor> {
    let id = Tensor::from_i32(idx.as_array().to_owned().into_dyn().mapv(|v| v as i32));
    Ok(PyTensor { inner: st_core::ops::segment_max(&data.inner, &id, k).map_err(map_err)? })
}
#[pyfunction] pub fn segment_min(data: &PyTensor, idx: PyReadonlyArrayDyn<i64>, k: usize) -> PyResult<PyTensor> {
    let id = Tensor::from_i32(idx.as_array().to_owned().into_dyn().mapv(|v| v as i32));
    Ok(PyTensor { inner: st_core::ops::segment_min(&data.inner, &id, k).map_err(map_err)? })
}

// Ragged & coalesce
#[pyfunction] pub fn coalesce_indices(idx: PyReadonlyArrayDyn<i64>) -> PyResult<(PyObject, PyObject, usize)> {
    Python::with_gil(|py| {
        let id = Tensor::from_i32(idx.as_array().to_owned().into_dyn().mapv(|v| v as i32));
        let (u, r, k) = st_core::ops::coalesce_indices(&id).map_err(map_err)?;
        let u_py = PyArrayDyn::<i32>::from_owned_array(py, u.data_i32()).into_py(py);
        let r_py = PyArrayDyn::<i32>::from_owned_array(py, r.data_i32()).into_py(py);
        Ok((u_py, r_py, k))
    })
}
#[pyfunction] pub fn ragged_segment_sum(data: &PyTensor, row_splits: PyReadonlyArrayDyn<i64>) -> PyResult<PyTensor> {
    let rs = Tensor::from_i32(row_splits.as_array().to_owned().into_dyn().mapv(|v| v as i32));
    Ok(PyTensor { inner: st_core::ops::ragged_segment_sum(&data.inner, &rs).map_err(map_err)? })
}
#[pyfunction] pub fn ragged_segment_mean(data: &PyTensor, row_splits: PyReadonlyArrayDyn<i64>) -> PyResult<PyTensor> {
    let rs = Tensor::from_i32(row_splits.as_array().to_owned().into_dyn().mapv(|v| v as i32));
    Ok(PyTensor { inner: st_core::ops::ragged_segment_mean(&data.inner, &rs).map_err(map_err)? })
}
#[pyfunction] pub fn ragged_segment_max(data: &PyTensor, row_splits: PyReadonlyArrayDyn<i64>) -> PyResult<PyTensor> {
    let rs = Tensor::from_i32(row_splits.as_array().to_owned().into_dyn().mapv(|v| v as i32));
    Ok(PyTensor { inner: st_core::ops::ragged_segment_max(&data.inner, &rs).map_err(map_err)? })
}
#[pyfunction] pub fn ragged_segment_min(data: &PyTensor, row_splits: PyReadonlyArrayDyn<i64>) -> PyResult<PyTensor> {
    let rs = Tensor::from_i32(row_splits.as_array().to_owned().into_dyn().mapv(|v| v as i32));
    Ok(PyTensor { inner: st_core::ops::ragged_segment_min(&data.inner, &rs).map_err(map_err)? })
}

// einsum & logprod
#[pyfunction]
pub fn einsum(py: Python<'_>, spec: &str, tensors: &PyTuple, optimize: Option<bool>) -> PyResult<PyTensor> {
    let mut vec: Vec<Tensor> = Vec::new();
    for any in tensors { vec.push(any_to_tensor(py, any)?); }
    let opt = optimize.unwrap_or(false);
    Ok(PyTensor { inner: st_core::ops::einsum_opt(spec, &vec, opt).map_err(map_err)? })
}

#[pyfunction] 
pub fn logprod(x: &PyTensor, dim: isize, keepdim: Option<bool>, eps: Option<f32>, nan_policy: Option<String>, inf_policy: Option<String>) 
    -> PyResult<(PyTensor, PyTensor)> 
{
    let kd = keepdim.unwrap_or(false);
    let e = eps.unwrap_or(1e-12);
    let np = nan_policy.unwrap_or_else(|| "propagate".to_string());
    let ip = inf_policy.unwrap_or_else(|| "propagate".to_string());
    let (l, s) = st_core::ops::logprod(&x.inner, dim, kd, e, &np, &ip).map_err(map_err)?;
    Ok((PyTensor{inner:l}, PyTensor{inner:s}))
}

#[pyfunction] pub fn sum(x: &PyTensor) -> PyResult<PyTensor> { Ok(PyTensor { inner: st_core::ops::sum(&x.inner).map_err(map_err)? }) }

#[pymodule]
fn spiraltorch_rs(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_function(wrap_pyfunction!(index_reduce, m)?)?;
    m.add_function(wrap_pyfunction!(segment_sum, m)?)?;
    m.add_function(wrap_pyfunction!(segment_mean, m)?)?;
    m.add_function(wrap_pyfunction!(segment_max, m)?)?;
    m.add_function(wrap_pyfunction!(segment_min, m)?)?;
    m.add_function(wrap_pyfunction!(coalesce_indices, m)?)?;
    m.add_function(wrap_pyfunction!(ragged_segment_sum, m)?)?;
    m.add_function(wrap_pyfunction!(ragged_segment_mean, m)?)?;
    m.add_function(wrap_pyfunction!(ragged_segment_max, m)?)?;
    m.add_function(wrap_pyfunction!(ragged_segment_min, m)?)?;
    m.add_function(wrap_pyfunction!(einsum, m)?)?;
    m.add_function(wrap_pyfunction!(logprod, m)?)?;
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    Ok(())
}
