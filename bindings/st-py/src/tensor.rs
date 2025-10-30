use pyo3::exceptions::{PyOverflowError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{
    PyAny, PyByteArray, PyBytes, PyDict, PyList, PyMemoryView, PyModule, PyString, PyTuple,
};
use pyo3::wrap_pyfunction;
use pyo3::{Bound, PyRef, PyRefMut};
#[cfg(feature = "hip")]
use st_backend_hip as hip_backend;
use st_tensor::dlpack::{drop_exported_state, DLManagedTensor, DLPACK_CAPSULE_NAME};
use st_tensor::{
    backend::cpu_dense, AttentionBackend, Layout, MatmulBackend, SoftmaxBackend, Tensor,
    TensorError,
};
use std::ffi::{c_void, CStr};
use std::sync::Arc;
use tracing::warn;

fn parse_backend(label: Option<&str>) -> MatmulBackend {
    match label.unwrap_or("auto") {
        "auto" => MatmulBackend::Auto,
        "faer" => MatmulBackend::CpuFaer,
        "cpu" => MatmulBackend::CpuFaer,
        "simd" => MatmulBackend::CpuSimd,
        "cpu-simd" => MatmulBackend::CpuSimd,
        "naive" => MatmulBackend::CpuNaive,
        #[cfg(feature = "wgpu")]
        "wgpu" => MatmulBackend::GpuWgpu,
        #[cfg(feature = "hip")]
        "hip" => MatmulBackend::GpuHip,
        other => {
            warn!(
                backend = other,
                "unknown backend label, falling back to auto"
            );
            MatmulBackend::Auto
        }
    }
}

fn parse_softmax_backend(label: Option<&str>) -> SoftmaxBackend {
    match label.unwrap_or("auto") {
        "auto" => SoftmaxBackend::Auto,
        "cpu" => SoftmaxBackend::Cpu,
        #[cfg(feature = "wgpu")]
        "wgpu" => SoftmaxBackend::GpuWgpu,
        other => {
            warn!(
                backend = other,
                "unknown softmax backend label, falling back to auto"
            );
            SoftmaxBackend::Auto
        }
    }
}

fn parse_attention_backend(label: Option<&str>) -> AttentionBackend {
    match label.unwrap_or("auto") {
        "auto" => AttentionBackend::Auto,
        "cpu" => AttentionBackend::Cpu,
        #[cfg(feature = "wgpu")]
        "wgpu" => AttentionBackend::GpuWgpu,
        other => {
            warn!(
                backend = other,
                "unknown attention backend label, falling back to auto",
            );
            AttentionBackend::Auto
        }
    }
}

#[derive(Default)]
struct TensorCtorDims {
    rows: Option<usize>,
    cols: Option<usize>,
}

enum TensorDataArg {
    Unspecified,
    ExplicitNone,
    Provided(PyObject),
}

impl TensorDataArg {
    fn is_set(&self) -> bool {
        !matches!(self, Self::Unspecified)
    }

    fn from_object(obj: &Bound<PyAny>) -> Self {
        if obj.is_none() {
            Self::ExplicitNone
        } else {
            Self::Provided(obj.clone().unbind())
        }
    }

    fn into_option(self) -> Option<PyObject> {
        match self {
            Self::Unspecified | Self::ExplicitNone => None,
            Self::Provided(obj) => Some(obj),
        }
    }
}

enum CollectContext {
    Data,
    Row,
}

impl CollectContext {
    fn type_error_message(&self) -> &'static str {
        match self {
            Self::Data => "Tensor data must be an iterable of floats or nested iterables",
            Self::Row => "Tensor rows must be sequences of numbers",
        }
    }
}

enum F32Input {
    Tensor(Tensor),
    Owned(Vec<f32>),
}

unsafe impl Send for F32Input {}

#[derive(Clone, Copy)]
struct TensorOutPtr(usize);

unsafe impl Send for TensorOutPtr {}
unsafe impl Sync for TensorOutPtr {}

impl TensorOutPtr {
    unsafe fn as_mut_ptr(self) -> *mut Tensor {
        self.0 as *mut Tensor
    }

    unsafe fn clone_tensor(self) -> Tensor {
        (*self.as_mut_ptr()).clone()
    }
}

#[pyclass(module = "spiraltorch", name = "CpuSimdPackedRhs")]
#[derive(Clone)]
pub struct PyCpuSimdPackedRhs {
    inner: usize,
    cols: usize,
    data: Arc<[f32]>,
}

impl PyCpuSimdPackedRhs {
    fn new(inner: usize, cols: usize, data: Vec<f32>) -> Self {
        Self {
            inner,
            cols,
            data: Arc::from(data.into_boxed_slice()),
        }
    }
}

#[pymethods]
impl PyCpuSimdPackedRhs {
    #[getter]
    fn inner(&self) -> usize {
        self.inner
    }

    #[getter]
    fn cols(&self) -> usize {
        self.cols
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.data.len())
    }

    fn tolist(&self) -> Vec<f32> {
        self.data.to_vec()
    }
}

fn borrow_f32_argument(any: &Bound<PyAny>) -> PyResult<Vec<f32>> {
    match F32Input::from_py(any)? {
        F32Input::Tensor(tensor) => Ok(tensor.data().to_vec()),
        F32Input::Owned(vec) => Ok(vec),
    }
}

const USED_DLPACK_CAPSULE_NAME: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"used_dltensor\0") };

#[pyclass(module = "spiraltorch", name = "Tensor", subclass)]
#[derive(Clone)]
pub struct PyTensor {
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

fn ensure_dim_matches(slot: &mut Option<usize>, value: usize, label: &str) -> PyResult<()> {
    if let Some(existing) = slot {
        if *existing != value {
            return Err(PyValueError::new_err(format!(
                "Tensor {label} argument conflicts with shape: {existing} != {value}"
            )));
        }
    } else {
        *slot = Some(value);
    }
    Ok(())
}

fn object_repr(any: &Bound<PyAny>) -> String {
    any.repr()
        .map(|s| s.to_string())
        .unwrap_or_else(|_| "<object>".to_owned())
}

fn is_string_like(any: &Bound<PyAny>) -> bool {
    any.downcast::<PyString>().is_ok()
        || any.downcast::<PyBytes>().is_ok()
        || any.downcast::<PyByteArray>().is_ok()
        || any.downcast::<PyMemoryView>().is_ok()
}

fn is_sequence_like(any: &Bound<PyAny>) -> bool {
    if is_string_like(any) {
        return false;
    }
    unsafe { ffi::PySequence_Check(any.as_ptr()) == 1 }
}

fn collect_iterable(any: &Bound<PyAny>, context: CollectContext) -> PyResult<Vec<PyObject>> {
    if is_string_like(any) {
        return Err(PyTypeError::new_err(context.type_error_message()));
    }
    let iter = any
        .iter()
        .map_err(|_| PyTypeError::new_err(context.type_error_message()))?;
    let mut out = Vec::new();
    for item in iter {
        out.push(item?.unbind());
    }
    Ok(out)
}

fn coerce_index(_: Python<'_>, any: &Bound<PyAny>, label: &str) -> PyResult<usize> {
    let repr = object_repr(any);
    let int_obj = match any.call_method0("__int__") {
        Ok(obj) => obj,
        Err(_) => any.call_method0("__index__").map_err(|_| {
            PyTypeError::new_err(format!("Tensor {label} must be an integer, got {repr}"))
        })?,
    };
    let value: i128 = int_obj.extract().map_err(|_| {
        PyTypeError::new_err(format!("Tensor {label} must be an integer, got {repr}"))
    })?;
    if value < 0 {
        return Err(PyValueError::new_err(format!(
            "Tensor {label} must be non-negative, got {value}"
        )));
    }
    usize::try_from(value).map_err(|_| PyOverflowError::new_err("tensor dimension overflowed"))
}

fn coerce_shape(py: Python<'_>, any: &Bound<PyAny>, label: &str) -> PyResult<(usize, usize)> {
    if !is_sequence_like(any) {
        return Err(PyTypeError::new_err(format!(
            "Tensor {label} must be a sequence of two integers"
        )));
    }
    let mut collected = Vec::with_capacity(2);
    for item in any.iter().map_err(|_| {
        PyTypeError::new_err(format!("Tensor {label} must be a sequence of two integers"))
    })? {
        collected.push(item?.unbind());
    }
    if collected.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "Tensor {label} must contain exactly two dimensions, got {}",
            collected.len()
        )));
    }
    let rows = coerce_index(py, &collected[0].bind(py), "shape[0]")?;
    let cols = coerce_index(py, &collected[1].bind(py), "shape[1]")?;
    Ok((rows, cols))
}

fn maybe_shape(py: Python<'_>, any: &Bound<PyAny>) -> PyResult<Option<(usize, usize)>> {
    if !is_sequence_like(any) {
        return Ok(None);
    }
    let mut collected = Vec::with_capacity(2);
    let iter = match any.iter() {
        Ok(iter) => iter,
        Err(_) => return Ok(None),
    };
    for item in iter {
        let obj = match item {
            Ok(obj) => obj,
            Err(_) => return Ok(None),
        };
        collected.push(obj.unbind());
    }
    if collected.len() != 2 {
        return Ok(None);
    }
    let rows = match coerce_index(py, &collected[0].bind(py), "shape[0]") {
        Ok(value) => value,
        Err(_) => return Ok(None),
    };
    let cols = match coerce_index(py, &collected[1].bind(py), "shape[1]") {
        Ok(value) => value,
        Err(_) => return Ok(None),
    };
    Ok(Some((rows, cols)))
}

fn normalize_row(py: Python<'_>, row: &Bound<PyAny>) -> PyResult<Vec<f32>> {
    if let Ok(py_tensor) = row.extract::<PyRef<PyTensor>>() {
        let (rows, cols) = py_tensor.inner.shape();
        if rows > 1 {
            return Err(PyTypeError::new_err(
                "Tensor rows must be sequences of numbers",
            ));
        }
        let data = py_tensor.inner.data();
        let slice = if rows == 0 { &[][..] } else { &data[..cols] };
        return Ok(slice.to_vec());
    }

    if !is_sequence_like(row) && row.hasattr("tolist")? {
        let converted = row.call_method0("tolist")?;
        return normalize_row(py, &converted);
    }

    let items = collect_iterable(row, CollectContext::Row)?;
    let mut values = Vec::with_capacity(items.len());
    for item in items {
        let value = item
            .bind(py)
            .extract::<f32>()
            .map_err(|_| PyTypeError::new_err("Tensor rows must be sequences of numbers"))?;
        values.push(value);
    }
    Ok(values)
}

fn is_row_container(py: Python<'_>, any: &Bound<PyAny>) -> PyResult<bool> {
    if any.extract::<PyRef<PyTensor>>().is_ok() {
        return Ok(true);
    }
    if !is_sequence_like(any) && any.hasattr("tolist")? {
        let converted = any.call_method0("tolist")?;
        return is_row_container(py, &converted);
    }
    if is_sequence_like(any) {
        return Ok(true);
    }
    if is_string_like(any) {
        return Ok(false);
    }
    Ok(any.iter().is_ok())
}

fn flatten_tensor_data(py: Python<'_>, data: &Bound<PyAny>) -> PyResult<(usize, usize, Vec<f32>)> {
    if let Ok(py_tensor) = data.extract::<PyRef<PyTensor>>() {
        let (rows, cols) = py_tensor.inner.shape();
        return Ok((rows, cols, py_tensor.inner.data().to_vec()));
    }

    if !is_sequence_like(data) && data.hasattr("tolist")? {
        let converted = data.call_method0("tolist")?;
        return flatten_tensor_data(py, &converted);
    }

    let items = collect_iterable(data, CollectContext::Data)?;
    if items.is_empty() {
        return Ok((0, 0, Vec::new()));
    }

    let treat_as_rows = is_row_container(py, &items[0].bind(py))?;
    if treat_as_rows {
        let mut cols: Option<usize> = None;
        let mut flat = Vec::new();
        for item in &items {
            let normalized = normalize_row(py, &item.bind(py))?;
            if let Some(expected) = cols {
                if expected != normalized.len() {
                    return Err(PyValueError::new_err(
                        "Tensor rows must all share the same length",
                    ));
                }
            } else {
                cols = Some(normalized.len());
            }
            flat.extend(normalized);
        }
        let cols = cols.unwrap_or(0);
        let rows = if cols == 0 {
            items.len()
        } else {
            flat.len() / cols
        };
        Ok((rows, cols, flat))
    } else {
        let mut flat = Vec::with_capacity(items.len());
        for item in &items {
            let value = item.bind(py).extract::<f32>().map_err(|_| {
                PyTypeError::new_err(
                    "Tensor data must be an iterable of floats or nested iterables",
                )
            })?;
            flat.push(value);
        }
        Ok((1, flat.len(), flat))
    }
}

fn infer_missing_dimension(total: usize, known: usize, label: &str) -> PyResult<usize> {
    if known == 0 {
        if total != 0 {
            return Err(PyValueError::new_err(format!(
                "Tensor data of length {total} cannot fill ({known}) {label}"
            )));
        }
        return Ok(0);
    }
    if total % known != 0 {
        return Err(PyValueError::new_err(format!(
            "Tensor data of length {total} cannot fill ({known}) {label}"
        )));
    }
    Ok(total / known)
}

impl F32Input {
    fn from_py(any: &Bound<PyAny>) -> PyResult<Self> {
        if let Ok(py_tensor) = any.extract::<PyRef<PyTensor>>() {
            let tensor = py_tensor.inner.clone();
            return Ok(Self::Tensor(tensor));
        }

        if let Ok(list) = any.downcast::<PyList>() {
            let mut owned = Vec::with_capacity(list.len());
            for item in list {
                owned.push(item.extract::<f32>()?);
            }
            return Ok(Self::Owned(owned));
        }

        if let Ok(vector) = any.extract::<Vec<f32>>() {
            return Ok(Self::Owned(vector));
        }

        Err(PyTypeError::new_err(
            "expected a PyTensor or iterable of floats",
        ))
    }
}

#[pymethods]
impl PyTensor {
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn new(
        py: Python<'_>,
        args: &Bound<PyTuple>,
        kwargs: Option<&Bound<PyDict>>,
    ) -> PyResult<Self> {
        let mut dims = TensorCtorDims::default();
        let mut data_kw = TensorDataArg::Unspecified;
        let mut shape_kw: Option<PyObject> = None;
        let mut rows_kw: Option<PyObject> = None;
        let mut cols_kw: Option<PyObject> = None;
        let mut backend_kw: Option<PyObject> = None;
        let mut unexpected = Vec::new();

        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs.iter() {
                let key = key
                    .downcast::<PyString>()
                    .map_err(|_| PyTypeError::new_err("Tensor() keyword names must be strings"))?;
                let label = key.to_string();
                match label.as_str() {
                    "data" => {
                        if data_kw.is_set() {
                            return Err(PyTypeError::new_err(
                                "Tensor() got multiple values for data",
                            ));
                        }
                        data_kw = TensorDataArg::from_object(&value);
                    }
                    "shape" => {
                        shape_kw = Some(value.unbind());
                    }
                    "rows" => {
                        rows_kw = Some(value.unbind());
                    }
                    "cols" => {
                        cols_kw = Some(value.unbind());
                    }
                    "backend" => {
                        backend_kw = Some(value.unbind());
                    }
                    other => unexpected.push(other.to_owned()),
                }
            }
        }

        if !unexpected.is_empty() {
            unexpected.sort();
            return Err(PyTypeError::new_err(format!(
                "Tensor() got unexpected keyword arguments: {}",
                unexpected.join(", ")
            )));
        }

        if let Some(shape_obj) = shape_kw {
            let shape_bound = shape_obj.bind(py);
            let (rows, cols) = coerce_shape(py, &shape_bound, "shape")?;
            dims.rows = Some(rows);
            dims.cols = Some(cols);
        }

        if let Some(rows_obj) = rows_kw {
            let rows_bound = rows_obj.bind(py);
            let rows = coerce_index(py, &rows_bound, "rows")?;
            ensure_dim_matches(&mut dims.rows, rows, "rows")?;
        }

        if let Some(cols_obj) = cols_kw {
            let cols_bound = cols_obj.bind(py);
            let cols = coerce_index(py, &cols_bound, "cols")?;
            ensure_dim_matches(&mut dims.cols, cols, "cols")?;
        }

        let mut positional: Vec<PyObject> = args.iter().map(|item| item.unbind()).collect();

        match positional.len() {
            0 => {}
            1 => {
                let candidate = positional.pop().unwrap();
                let candidate_bound = candidate.bind(py);
                if dims.rows.is_none() && dims.cols.is_none() {
                    if let Some((rows, cols)) = maybe_shape(py, &candidate_bound)? {
                        dims.rows = Some(rows);
                        dims.cols = Some(cols);
                    } else {
                        if data_kw.is_set() {
                            return Err(PyTypeError::new_err(
                                "Tensor() got multiple values for data",
                            ));
                        }
                        data_kw = TensorDataArg::from_object(&candidate_bound);
                    }
                } else {
                    if data_kw.is_set() {
                        return Err(PyTypeError::new_err(
                            "Tensor() got multiple values for data",
                        ));
                    }
                    data_kw = TensorDataArg::from_object(&candidate_bound);
                }
            }
            2 => {
                let second = positional.pop().unwrap();
                let first = positional.pop().unwrap();
                let first_bound = first.bind(py);
                if dims.rows.is_none() && dims.cols.is_none() {
                    if let Some((rows, cols)) = maybe_shape(py, &first_bound)? {
                        dims.rows = Some(rows);
                        dims.cols = Some(cols);
                        if data_kw.is_set() {
                            return Err(PyTypeError::new_err(
                                "Tensor() got multiple values for data",
                            ));
                        }
                        data_kw = TensorDataArg::from_object(&second.bind(py));
                    } else {
                        let inferred_rows = coerce_index(py, &first_bound, "rows")?;
                        let inferred_cols = coerce_index(py, &second.bind(py), "cols")?;
                        ensure_dim_matches(&mut dims.rows, inferred_rows, "rows")?;
                        ensure_dim_matches(&mut dims.cols, inferred_cols, "cols")?;
                    }
                } else {
                    let inferred_rows = coerce_index(py, &first_bound, "rows")?;
                    ensure_dim_matches(&mut dims.rows, inferred_rows, "rows")?;
                    let inferred_cols = coerce_index(py, &second.bind(py), "cols")?;
                    ensure_dim_matches(&mut dims.cols, inferred_cols, "cols")?;
                }
            }
            3 => {
                let third = positional.pop().unwrap();
                let second = positional.pop().unwrap();
                let first = positional.pop().unwrap();
                let rows_val = coerce_index(py, &first.bind(py), "rows")?;
                ensure_dim_matches(&mut dims.rows, rows_val, "rows")?;
                let cols_val = coerce_index(py, &second.bind(py), "cols")?;
                ensure_dim_matches(&mut dims.cols, cols_val, "cols")?;
                if data_kw.is_set() {
                    return Err(PyTypeError::new_err(
                        "Tensor() got multiple values for data",
                    ));
                }
                data_kw = TensorDataArg::from_object(&third.bind(py));
            }
            n => {
                return Err(PyTypeError::new_err(format!(
                    "Tensor() takes at most 3 positional arguments but {n} were given",
                )));
            }
        }

        if let Some(obj) = backend_kw {
            let bound = obj.bind(py);
            if !bound.is_none() {
                let label: String = bound
                    .extract()
                    .map_err(|_| PyTypeError::new_err("backend must be a string or None"))?;
                if label != "numpy" && label != "python" {
                    return Err(PyValueError::new_err(
                        "backend must be 'numpy', 'python', or None",
                    ));
                }
                warn!(
                    backend = label.as_str(),
                    "Tensor constructor backend hint ignored; using default storage",
                );
            }
        }

        let tensor = if let Some(data_obj) = data_kw.into_option() {
            let data_bound = data_obj.bind(py);
            let (inferred_rows, inferred_cols, flat) = flatten_tensor_data(py, &data_bound)?;
            let total = flat.len();

            let rows = match dims.rows {
                Some(rows) => rows,
                None => {
                    if let Some(cols) = dims.cols {
                        infer_missing_dimension(total, cols, "columns")?
                    } else {
                        inferred_rows
                    }
                }
            };

            let cols = match dims.cols {
                Some(cols) => cols,
                None => {
                    if let Some(rows) = dims.rows {
                        infer_missing_dimension(total, rows, "rows")?
                    } else {
                        inferred_cols
                    }
                }
            };

            if rows == 0 || cols == 0 {
                if total != 0 {
                    return Err(PyValueError::new_err(format!(
                        "Tensor shape ({rows}, {cols}) is incompatible with {total} data elements",
                    )));
                }
            } else if rows
                .checked_mul(cols)
                .ok_or_else(|| PyOverflowError::new_err("tensor dimensions overflowed"))?
                != total
            {
                return Err(PyValueError::new_err(format!(
                    "Tensor data of length {total} cannot be reshaped to ({rows}, {cols})",
                )));
            }

            Tensor::from_vec(rows, cols, flat).map_err(tensor_err_to_py)?
        } else {
            let (rows, cols) = match (dims.rows, dims.cols) {
                (Some(r), Some(c)) => (r, c),
                _ => {
                    return Err(PyTypeError::new_err(
                        "Tensor() requires a shape when data is omitted",
                    ));
                }
            };
            Tensor::zeros(rows, cols).map_err(tensor_err_to_py)?
        };

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

    /// Return an opaque token representing the underlying storage buffer.
    ///
    /// The token value is derived from the base pointer of the tensor's
    /// storage. It is stable for the lifetime of the allocation and changes
    /// whenever the tensor reallocates or is rebound to a different buffer.
    pub fn storage_token(&self) -> usize {
        self.inner.data().as_ptr() as usize
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

    #[pyo3(signature = (packed, *, out=None))]
    pub fn matmul_simd_prepacked(
        &self,
        packed: &PyCpuSimdPackedRhs,
        out: Option<&Bound<PyAny>>,
        py: Python<'_>,
    ) -> PyResult<PyTensor> {
        if !matches!(self.inner.layout(), Layout::RowMajor) {
            return Err(PyValueError::new_err(
                "cpu-simd matmul expects a row-major left-hand side tensor",
            ));
        }

        let (rows, inner) = self.inner.shape();
        if inner != packed.inner {
            return Err(PyValueError::new_err(format!(
                "lhs inner dimension {} does not match packed rhs inner {}",
                inner, packed.inner
            )));
        }

        let cols = packed.cols;
        let packed_buf = packed.data.clone();
        let lhs = self.inner.clone();

        if let Some(cell) = out {
            let mut dst = cell.extract::<PyRefMut<PyTensor>>()?;
            if dst.inner.shape() != (rows, cols) {
                return Err(PyValueError::new_err(format!(
                    "destination shape {:?} does not match ({}, {})",
                    dst.inner.shape(),
                    rows,
                    cols
                )));
            }
            if !matches!(dst.inner.layout(), Layout::RowMajor) {
                return Err(PyValueError::new_err(
                    "cpu-simd matmul expects a row-major destination tensor",
                ));
            }

            let dst_ptr = TensorOutPtr((&mut dst.inner as *mut Tensor) as usize);
            drop(dst);

            let rows_cl = rows;
            let inner_cl = inner;
            let cols_cl = cols;
            let dst_ptr_closure = dst_ptr;

            py.allow_threads(move || unsafe {
                let tensor = &mut *dst_ptr_closure.as_mut_ptr();
                cpu_dense::matmul_packed_into(
                    tensor.data_mut(),
                    lhs.data(),
                    packed_buf.as_ref(),
                    rows_cl,
                    inner_cl,
                    cols_cl,
                )
            })
            .map_err(|message| PyRuntimeError::new_err(message))?;

            let tensor = unsafe { dst_ptr.clone_tensor() };
            return Ok(PyTensor { inner: tensor });
        }

        let mut dst_tensor = Tensor::zeros(rows, cols).map_err(tensor_err_to_py)?;
        let dst_ptr = TensorOutPtr((&mut dst_tensor as *mut Tensor) as usize);
        let rows_cl = rows;
        let inner_cl = inner;
        let cols_cl = cols;
        let dst_ptr_closure = dst_ptr;

        py.allow_threads(move || unsafe {
            let tensor = &mut *dst_ptr_closure.as_mut_ptr();
            cpu_dense::matmul_packed_into(
                tensor.data_mut(),
                lhs.data(),
                packed_buf.as_ref(),
                rows_cl,
                inner_cl,
                cols_cl,
            )
        })
        .map_err(|message| PyRuntimeError::new_err(message))?;

        Ok(PyTensor { inner: dst_tensor })
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

    /// Matrix multiply fused with bias addition and GELU activation.
    #[pyo3(signature = (other, bias, *, backend=None, out=None))]
    pub fn matmul_bias_gelu(
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
            let bias_owned = bias.clone();
            let tensor = py
                .allow_threads(move || {
                    self.inner.matmul_bias_gelu_with_backend(
                        &other.inner,
                        bias_owned.as_slice(),
                        backend,
                    )
                })
                .map_err(tensor_err_to_py)?;

            {
                let mut dst = cell.extract::<PyRefMut<PyTensor>>()?;
                dst.inner = tensor.clone();
            }

            return Ok(PyTensor { inner: tensor });
        }

        let slice = bias.as_slice();
        let tensor = py
            .allow_threads(|| {
                self.inner
                    .matmul_bias_gelu_with_backend(&other.inner, slice, backend)
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
            let bias_owned = bias.clone();
            let tensor = py
                .allow_threads(move || {
                    self.inner.matmul_bias_add_gelu_with_backend(
                        &other.inner,
                        bias_owned.as_slice(),
                        &residual.inner,
                        backend,
                    )
                })
                .map_err(tensor_err_to_py)?;

            {
                let mut dst = cell.extract::<PyRefMut<PyTensor>>()?;
                dst.inner = tensor.clone();
            }

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

    /// Row-wise softmax with optional backend override.
    #[pyo3(signature = (*, backend=None))]
    pub fn row_softmax(&self, backend: Option<&str>, py: Python<'_>) -> PyResult<PyTensor> {
        let backend = parse_softmax_backend(backend);
        let tensor = py
            .allow_threads(|| self.inner.row_softmax_with_backend(backend))
            .map_err(tensor_err_to_py)?;
        Ok(PyTensor { inner: tensor })
    }

    #[pyo3(signature = (keys, values, *, contexts, sequence, scale, z_bias=None, attn_bias=None, backend=None))]
    pub fn scaled_dot_attention(
        &self,
        keys: &PyTensor,
        values: &PyTensor,
        contexts: usize,
        sequence: usize,
        scale: f32,
        z_bias: Option<&PyTensor>,
        attn_bias: Option<&PyTensor>,
        backend: Option<&str>,
        py: Python<'_>,
    ) -> PyResult<PyTensor> {
        let backend = parse_attention_backend(backend);
        let tensor = py
            .allow_threads(|| {
                self.inner.scaled_dot_attention_with_backend(
                    &keys.inner,
                    &values.inner,
                    contexts,
                    sequence,
                    scale,
                    z_bias.map(|tensor| &tensor.inner),
                    attn_bias.map(|tensor| &tensor.inner),
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

    pub fn sum_axis1(&self) -> Vec<f32> {
        self.inner.sum_axis1()
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
    pub fn cat_rows(tensors: &Bound<PyList>, py: Python<'_>) -> PyResult<Self> {
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
fn tensor_to_dlpack(py: Python<'_>, tensor: PyRef<PyTensor>) -> PyResult<PyObject> {
    tensor.to_dlpack(py)
}

#[pyfunction]
fn cpu_simd_prepack_rhs(py: Python<'_>, rhs: &PyTensor) -> PyResult<PyCpuSimdPackedRhs> {
    if !matches!(rhs.inner.layout(), Layout::RowMajor) {
        return Err(PyValueError::new_err(
            "cpu-simd prepack expects a row-major tensor",
        ));
    }

    let (inner, cols) = rhs.inner.shape();
    let packed = py
        .allow_threads(|| cpu_dense::prepack_rhs(rhs.inner.data(), inner, cols))
        .map_err(PyRuntimeError::new_err)?;

    Ok(PyCpuSimdPackedRhs::new(inner, cols, packed))
}

#[pyfunction]
fn init_backend(label: &str) -> PyResult<bool> {
    match label {
        #[cfg(feature = "hip")]
        "hip" => hip_backend::init()
            .map(|_| true)
            .map_err(|err| PyRuntimeError::new_err(err.to_string())),
        #[cfg(not(feature = "hip"))]
        "hip" => Err(PyRuntimeError::new_err(
            "SpiralTorch was built without HIP support; rebuild with the 'hip' feature",
        )),
        "auto" | "cpu" | "faer" | "simd" | "cpu-simd" | "naive" => Ok(true),
        #[cfg(feature = "wgpu")]
        "wgpu" => Ok(true),
        other => Err(PyValueError::new_err(format!(
            "unknown backend label '{other}'"
        ))),
    }
}

#[cfg(feature = "wgpu")]
#[pyfunction]
fn describe_wgpu_softmax_variants(py: Python<'_>) -> PyResult<Option<Vec<PyObject>>> {
    use st_tensor::backend::wgpu_dense;

    if !wgpu_dense::is_available() {
        return Ok(None);
    }

    let snapshot = match wgpu_dense::softmax_autotune_snapshot() {
        Some(entries) if !entries.is_empty() => entries,
        _ => return Ok(None),
    };

    let mut out = Vec::with_capacity(snapshot.len());
    for entry in snapshot {
        let dict = PyDict::new_bound(py);
        dict.set_item("key", entry.key)?;
        dict.set_item("variant", entry.variant_name())?;
        dict.set_item("score_ms", entry.score_ms)?;
        dict.set_item("samples", entry.samples)?;
        if let Some(summary) = entry.telemetry() {
            let telemetry = PyDict::new_bound(py);
            telemetry.set_item("avg_tflops", summary.avg_tflops)?;
            telemetry.set_item("avg_bandwidth_gbps", summary.avg_bandwidth_gbps)?;
            telemetry.set_item("avg_occupancy", summary.avg_occupancy)?;
            telemetry.set_item("regression_rate", summary.regression_rate)?;
            dict.set_item("telemetry", telemetry)?;
        }
        if let Some(zspace) = entry.zspace() {
            let zspace_dict = PyDict::new_bound(py);
            zspace_dict.set_item("focus", zspace.focus)?;
            zspace_dict.set_item("spiral_flux", zspace.spiral_flux)?;
            let roundtable = PyDict::new_bound(py);
            roundtable.set_item("above", zspace.roundtable.above)?;
            roundtable.set_item("here", zspace.roundtable.here)?;
            roundtable.set_item("beneath", zspace.roundtable.beneath)?;
            roundtable.set_item("drift", zspace.roundtable.drift)?;
            zspace_dict.set_item("roundtable", roundtable)?;
            let golden = PyDict::new_bound(py);
            golden.set_item("ratio_bias", zspace.golden.ratio_bias)?;
            golden.set_item("angle_bias_deg", zspace.golden.angle_bias_deg)?;
            golden.set_item("cooperative_weight", zspace.golden.cooperative_weight)?;
            zspace_dict.set_item("golden", golden)?;
            if let Some(projection) = entry.projection() {
                let projection_dict = PyDict::new_bound(py);
                projection_dict.set_item("focus", projection.focus)?;
                projection_dict.set_item("above", projection.above)?;
                projection_dict.set_item("here", projection.here)?;
                projection_dict.set_item("beneath", projection.beneath)?;
                projection_dict.set_item("swirl", projection.swirl)?;
                projection_dict.set_item("spiral_flux", projection.spiral_flux)?;
                zspace_dict.set_item("projection", projection_dict)?;
            }
            dict.set_item("zspace", zspace_dict)?;
        }
        if let Some(bayesian) = entry.bayesian() {
            let bayes_dict = PyDict::new_bound(py);
            bayes_dict.set_item("posterior_ms", bayesian.posterior_ms)?;
            bayes_dict.set_item("prior_ms", bayesian.prior_ms)?;
            bayes_dict.set_item("uplift_ms", bayesian.uplift_ms)?;
            bayes_dict.set_item("confidence", bayesian.confidence)?;
            bayes_dict.set_item("credible_low_ms", bayesian.credible_low_ms)?;
            bayes_dict.set_item("credible_high_ms", bayesian.credible_high_ms)?;
            dict.set_item("bayesian", bayes_dict)?;
        }
        if let Some(metropolis) = entry.metropolis() {
            let mtm_dict = PyDict::new_bound(py);
            mtm_dict.set_item("acceptance", metropolis.acceptance)?;
            mtm_dict.set_item("expected_ms", metropolis.expected_ms)?;
            mtm_dict.set_item("tries", metropolis.tries)?;
            mtm_dict.set_item("proposal_focus", metropolis.proposal_focus)?;
            mtm_dict.set_item("proposal_flux", metropolis.proposal_flux)?;
            dict.set_item("metropolis", mtm_dict)?;
        }
        if let Some(anneal) = entry.anneal() {
            let anneal_dict = PyDict::new_bound(py);
            anneal_dict.set_item("temperature", anneal.temperature)?;
            anneal_dict.set_item("annealed_ms", anneal.annealed_ms)?;
            anneal_dict.set_item("exploration_mass", anneal.exploration_mass)?;
            anneal_dict.set_item("entropy", anneal.entropy)?;
            anneal_dict.set_item("refreshes", anneal.refreshes)?;
            dict.set_item("anneal", anneal_dict)?;
        }
        out.push(dict.unbind().into());
    }

    Ok(Some(out))
}

#[cfg(not(feature = "wgpu"))]
#[pyfunction]
fn describe_wgpu_softmax_variants(_py: Python<'_>) -> PyResult<Option<Vec<PyObject>>> {
    Ok(None)
}

pub(crate) fn register(py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_class::<PyCpuSimdPackedRhs>()?;
    m.add_function(wrap_pyfunction!(tensor_from_dlpack, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_to_dlpack, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_simd_prepack_rhs, m)?)?;
    m.add_function(wrap_pyfunction!(init_backend, m)?)?;
    m.add_function(wrap_pyfunction!(describe_wgpu_softmax_variants, m)?)?;
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

pub(crate) fn to_dlpack_impl(py: Python<'_>, tensor: &Tensor) -> PyResult<PyObject> {
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

pub(crate) fn tensor_to_torch(py: Python<'_>, tensor: &Tensor) -> PyResult<PyObject> {
    let capsule = to_dlpack_impl(py, tensor)?;
    let torch_dlpack = PyModule::import_bound(py, "torch.utils.dlpack").map_err(|_| {
        PyValueError::new_err("import torch.utils.dlpack before requesting torch tensors")
    })?;
    let torch_tensor = torch_dlpack.call_method1("from_dlpack", (capsule,))?;
    Ok(torch_tensor.into())
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
            let tensor = PyTensor::from_tensor(
                Tensor::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
            );
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
