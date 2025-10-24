use std::ffi::{c_void, CStr};

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyModule};
use pyo3::wrap_pyfunction;
use pyo3::{Bound, PyRef, PyRefMut};
use st_tensor::dlpack::{drop_exported_state, DLManagedTensor, DLPACK_CAPSULE_NAME};
use st_tensor::{MatmulBackend, Tensor, TensorError};

fn parse_backend(label: Option<&str>) -> MatmulBackend {
    match label.unwrap_or("auto") {
        "auto" => MatmulBackend::Auto,
        "faer" => MatmulBackend::CpuFaer,
        "naive" => MatmulBackend::CpuNaive,
        #[cfg(feature = "wgpu")]
        "wgpu" => MatmulBackend::GpuWgpu,
        other => {
            tracing::warn!("[spiraltorch] unknown backend '{other}', falling back to 'auto'");
            MatmulBackend::Auto
        }
    }
}

enum BorrowedF32 {
    Tensor(Tensor),
    Owned(Vec<f32>),
}

impl BorrowedF32 {
    fn as_slice(&self) -> &[f32] {
        match self {
            BorrowedF32::Tensor(tensor) => tensor.data(),
            BorrowedF32::Owned(values) => values.as_slice(),
        }
    }
}

#[derive(Copy, Clone)]
struct TensorOutPtr(usize);

unsafe impl Send for TensorOutPtr {}

impl TensorOutPtr {
    unsafe fn as_mut_ptr(self) -> *mut Tensor {
        self.0 as *mut Tensor
    }

    unsafe fn clone_tensor(self) -> Tensor {
        (&*(self.0 as *mut Tensor)).clone()
    }
}

fn try_extract_tensor_from_dlpack(py: Python<'_>, any: &Bound<PyAny>) -> PyResult<Option<Tensor>> {
    match ensure_dlpack_capsule(py, any) {
        Ok(capsule) => {
            let capsule = capsule.bind(py);
            let tensor = from_dlpack_impl(py, &capsule)?;
            Ok(Some(tensor.inner))
        }
        Err(err) => {
            if err.is_instance_of::<PyTypeError>(py) {
                Ok(None)
            } else {
                Err(err)
            }
        }
    }
}

fn borrow_f32_argument(any: &Bound<PyAny>) -> PyResult<BorrowedF32> {
    let py = any.py();
    if let Ok(tensor) = any.extract::<PyRef<PyTensor>>() {
        return Ok(BorrowedF32::Tensor(tensor.inner.clone()));
    }

    if let Some(tensor) = try_extract_tensor_from_dlpack(py, any)? {
        return Ok(BorrowedF32::Tensor(tensor));
    }

    let values: Vec<f32> = any.extract()?;
    Ok(BorrowedF32::Owned(values))
}

const USED_DLPACK_CAPSULE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"used_dltensor\0") };

#[pyclass(module = "spiraltorch", name = "Tensor")]
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

    /// Sample from N(mean, std^2) to create a (rows x cols) tensor.
    #[staticmethod]
    #[pyo3(signature = (rows, cols, mean=0.0, std=1.0, seed=None))]
    pub fn randn(
        rows: usize,
        cols: usize,
        mean: f32,
        std: f32,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let tensor =
            Tensor::random_normal(rows, cols, mean, std, seed).map_err(tensor_err_to_py)?;
        Ok(Self { inner: tensor })
    }

    /// Sample from Uniform[min, max) to create a (rows x cols) tensor.
    #[staticmethod]
    #[pyo3(signature = (rows, cols, min=0.0, max=1.0, seed=None))]
    pub fn rand(rows: usize, cols: usize, min: f32, max: f32, seed: Option<u64>) -> PyResult<Self> {
        let tensor =
            Tensor::random_uniform(rows, cols, min, max, seed).map_err(tensor_err_to_py)?;
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
        if let Some(stream_obj) = stream {
            let stream = stream_obj.bind(py);
            if !stream.is_none() {
                let value: isize = stream
                    .extract()
                    .map_err(|_| PyValueError::new_err("CPU tensors expect stream=None or 0"))?;
                if value != 0 {
                    return Err(PyValueError::new_err("CPU tensors expect stream=None or 0"));
                }
            }
        }
        self.to_dlpack(py)
    }

    pub fn __dlpack_device__(&self) -> (i32, i32) {
        let device = self.inner.dlpack_device();
        (device.device_type, device.device_id)
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

    /// Matrix multiply: self @ other
    #[pyo3(signature = (other, *, backend=None, out=None))]
    pub fn matmul(
        &self,
        other: &PyTensor,
        backend: Option<&str>,
        out: Option<&Bound<PyAny>>,
        py: Python<'_>,
    ) -> PyResult<PyTensor> {
        let backend = parse_backend(backend);
        if let Some(cell) = out {
            let mut dst = cell.extract::<PyRefMut<PyTensor>>()?;
            let dst_ptr = TensorOutPtr((&mut dst.inner as *mut Tensor) as usize);
            drop(dst);

            // SAFETY: `dst_ptr` points to the destination tensor owned by the Python
            // object. We drop the `PyRefMut` before releasing the GIL, ensuring there
            // are no outstanding Rust borrows when the computation runs.
            let dst_ptr_closure = dst_ptr;
            py.allow_threads(move || unsafe {
                self.inner.matmul_into_with_backend(
                    &other.inner,
                    &mut *dst_ptr_closure.as_mut_ptr(),
                    backend,
                )
            })
            .map_err(tensor_err_to_py)?;

            // SAFETY: `dst_ptr` still references the tensor stored in the Python
            // object. Cloning after the computation completes is sound and gives us
            // a handle to return to Python while leaving the buffer in place.
            let tensor = unsafe { dst_ptr.clone_tensor() };
            return Ok(PyTensor { inner: tensor });
        }

        let tensor = py
            .allow_threads(|| self.inner.matmul_with_backend(&other.inner, backend))
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor { inner: tensor })
    }

    /// Matrix multiply with bias and ReLU fusion.
    #[pyo3(signature = (other, bias, *, backend=None, out=None))]
    pub fn matmul_bias_relu(
        &self,
        other: &PyTensor,
        bias: &Bound<PyAny>,
        backend: Option<&str>,
        out: Option<&Bound<PyAny>>,
        py: Python<'_>,
    ) -> PyResult<PyTensor> {
        let backend = parse_backend(backend);
        let bias = borrow_f32_argument(bias)?;

        if let Some(cell) = out {
            let mut dst = cell.extract::<PyRefMut<PyTensor>>()?;
            let dst_ptr = TensorOutPtr((&mut dst.inner as *mut Tensor) as usize);
            drop(dst);

            let slice = bias.as_slice();
            // SAFETY: see the explanation in `matmul`; we drop the borrow before
            // releasing the GIL and only touch the tensor through `dst_ptr`.
            let dst_ptr_closure = dst_ptr;
            py.allow_threads(move || unsafe {
                self.inner.matmul_bias_relu_into_with_backend(
                    &other.inner,
                    slice,
                    &mut *dst_ptr_closure.as_mut_ptr(),
                    backend,
                )
            })
            .map_err(tensor_err_to_py)?;

            // SAFETY: `dst_ptr` remains valid for the duration of the call.
            let tensor = unsafe { dst_ptr.clone_tensor() };
            return Ok(PyTensor { inner: tensor });
        }

        let slice = bias.as_slice();
        let tensor = py
            .allow_threads(|| {
                self.inner
                    .matmul_bias_relu_with_backend(&other.inner, slice, backend)
            })
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor { inner: tensor })
    }

    /// Matrix multiply fused with bias, residual addition, and ReLU activation.
    #[pyo3(signature = (other, bias, residual, *, backend=None, out=None))]
    pub fn matmul_bias_add_relu(
        &self,
        other: &PyTensor,
        bias: &Bound<PyAny>,
        residual: &PyTensor,
        backend: Option<&str>,
        out: Option<&Bound<PyAny>>,
        py: Python<'_>,
    ) -> PyResult<PyTensor> {
        let backend = parse_backend(backend);
        let bias = borrow_f32_argument(bias)?;

        if let Some(cell) = out {
            let mut dst = cell.extract::<PyRefMut<PyTensor>>()?;
            let dst_ptr = TensorOutPtr((&mut dst.inner as *mut Tensor) as usize);
            drop(dst);

            let slice = bias.as_slice();
            // SAFETY: identical reasoning to `matmul` — the raw pointer targets the
            // tensor inside `out` and no Rust borrow lives across the GIL release.
            let dst_ptr_closure = dst_ptr;
            py.allow_threads(move || unsafe {
                self.inner.matmul_bias_add_relu_into_with_backend(
                    &other.inner,
                    slice,
                    &residual.inner,
                    &mut *dst_ptr_closure.as_mut_ptr(),
                    backend,
                )
            })
            .map_err(tensor_err_to_py)?;

            // SAFETY: `dst_ptr` still points to the tensor owned by Python.
            let tensor = unsafe { dst_ptr.clone_tensor() };
            return Ok(PyTensor { inner: tensor });
        }

        let slice = bias.as_slice();
        let tensor = py
            .allow_threads(|| {
                self.inner.matmul_bias_add_relu_with_backend(
                    &other.inner,
                    slice,
                    &residual.inner,
                    backend,
                )
            })
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor { inner: tensor })
    }

    /// Matrix multiply fused with bias, residual addition, and GELU activation.
    #[pyo3(signature = (other, bias, residual, *, backend=None, out=None))]
    pub fn matmul_bias_add_gelu(
        &self,
        other: &PyTensor,
        bias: &Bound<PyAny>,
        residual: &PyTensor,
        backend: Option<&str>,
        out: Option<&Bound<PyAny>>,
        py: Python<'_>,
    ) -> PyResult<PyTensor> {
        let backend = parse_backend(backend);
        let bias = borrow_f32_argument(bias)?;

        if let Some(cell) = out {
            let mut dst = cell.extract::<PyRefMut<PyTensor>>()?;
            let dst_ptr = TensorOutPtr((&mut dst.inner as *mut Tensor) as usize);
            drop(dst);

            let slice = bias.as_slice();
            // SAFETY: identical reasoning to the other `out=` helpers; the pointer is
            // valid for the duration of the computation and no borrow is held across
            // the GIL release.
            let dst_ptr_closure = dst_ptr;
            py.allow_threads(move || unsafe {
                self.inner.matmul_bias_add_gelu_into_with_backend(
                    &other.inner,
                    slice,
                    &residual.inner,
                    &mut *dst_ptr_closure.as_mut_ptr(),
                    backend,
                )
            })
            .map_err(tensor_err_to_py)?;

            // SAFETY: `dst_ptr` still points to the tensor owned by Python.
            let tensor = unsafe { dst_ptr.clone_tensor() };
            return Ok(PyTensor { inner: tensor });
        }

        let slice = bias.as_slice();
        let tensor = py
            .allow_threads(|| {
                self.inner.matmul_bias_add_gelu_with_backend(
                    &other.inner,
                    slice,
                    &residual.inner,
                    backend,
                )
            })
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor { inner: tensor })
    }

    /// Add (element-wise)
    pub fn add(&self, other: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        let tensor = py
            .allow_threads(|| self.inner.add(&other.inner))
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor { inner: tensor })
    }

    pub fn sub(&self, other: &PyTensor, py: Python<'_>) -> PyResult<Self> {
        let tensor = py
            .allow_threads(|| self.inner.sub(&other.inner))
            .map_err(tensor_err_to_py)?;
        Ok(Self { inner: tensor })
    }

    pub fn scale(&self, value: f32, py: Python<'_>) -> PyResult<Self> {
        let tensor = py
            .allow_threads(|| self.inner.scale(value))
            .map_err(tensor_err_to_py)?;
        Ok(Self { inner: tensor })
    }

    /// Element-wise multiply (Hadamard)
    pub fn hadamard(&self, other: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        let tensor = py
            .allow_threads(|| self.inner.hadamard(&other.inner))
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor { inner: tensor })
    }

    #[pyo3(name = "add_scaled_")]
    pub fn add_scaled_inplace(
        &mut self,
        other: &PyTensor,
        scale: f32,
        py: Python<'_>,
    ) -> PyResult<()> {
        py.allow_threads(|| self.inner.add_scaled(&other.inner, scale))
            .map_err(tensor_err_to_py)
    }

    pub fn add_row_inplace(&mut self, bias: Vec<f32>, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.inner.add_row_inplace(&bias))
            .map_err(tensor_err_to_py)
    }

    /// Transpose
    pub fn transpose(&self, py: Python<'_>) -> PyResult<PyTensor> {
        let tensor = py.allow_threads(|| self.inner.transpose());
        Ok(PyTensor { inner: tensor })
    }

    pub fn reshape(&self, rows: usize, cols: usize, py: Python<'_>) -> PyResult<Self> {
        let tensor = py
            .allow_threads(|| self.inner.reshape(rows, cols))
            .map_err(tensor_err_to_py)?;
        Ok(Self { inner: tensor })
    }

    pub fn sum_axis0(&self) -> Vec<f32> {
        self.inner.sum_axis0()
    }

    pub fn squared_l2_norm(&self) -> f32 {
        self.inner.squared_l2_norm()
    }

    pub fn project_to_poincare(&self, curvature: f32, py: Python<'_>) -> PyResult<Self> {
        let tensor = py
            .allow_threads(|| self.inner.project_to_poincare(curvature))
            .map_err(tensor_err_to_py)?;
        Ok(Self { inner: tensor })
    }

    pub fn hyperbolic_distance(
        &self,
        other: &PyTensor,
        curvature: f32,
        py: Python<'_>,
    ) -> PyResult<f32> {
        py.allow_threads(|| self.inner.hyperbolic_distance(&other.inner, curvature))
            .map_err(tensor_err_to_py)
    }

    #[staticmethod]
    pub fn cat_rows(py: Python<'_>, tensors: &Bound<PyList>) -> PyResult<Self> {
        let mut owned: Vec<Tensor> = Vec::with_capacity(tensors.len());
        for item in tensors.iter() {
            let tensor: PyRef<PyTensor> = item.extract()?;
            owned.push(tensor.inner.clone());
        }
        let tensor = py
            .allow_threads(|| Tensor::cat_rows(&owned))
            .map_err(tensor_err_to_py)?;
        Ok(Self { inner: tensor })
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
