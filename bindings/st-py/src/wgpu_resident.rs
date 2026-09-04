use pyo3::prelude::*;

#[cfg(feature = "wgpu")]
use crate::tensor::PyTensor;
#[cfg(feature = "wgpu")]
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    types::PyDict,
};
#[cfg(feature = "wgpu")]
use st_backend_wgpu::{
    resident_matmul::{MatmulError, MatmulKernel, MatmulShape, MatmulTile, ResidentMatmul},
    runtime,
};

#[cfg(feature = "wgpu")]
fn error(err: MatmulError) -> PyErr {
    match err {
        MatmulError::Runtime(_) => PyRuntimeError::new_err(err.to_string()),
        _ => PyValueError::new_err(err.to_string()),
    }
}

#[cfg(feature = "wgpu")]
#[pyclass(name = "WgpuMatmul", module = "spiraltorch.wgpu")]
pub(crate) struct PyWgpuMatmul {
    inner: ResidentMatmul,
}

#[cfg(feature = "wgpu")]
#[pymethods]
impl PyWgpuMatmul {
    #[new]
    #[pyo3(signature = (rows, inner, cols, *, tile_mnk=None, kernel=None))]
    fn new(
        py: Python<'_>,
        rows: usize,
        inner: usize,
        cols: usize,
        tile_mnk: Option<(u32, u32, u32)>,
        kernel: Option<&str>,
    ) -> PyResult<Self> {
        let shape = MatmulShape::new(rows, inner, cols).map_err(error)?;
        let tile = match tile_mnk {
            Some((m, n, k)) => MatmulTile::new(m, n, k).map_err(error)?,
            None => MatmulTile::default(),
        };
        let kernel = kernel
            .map(|value| value.parse::<MatmulKernel>())
            .transpose()
            .map_err(PyValueError::new_err)?;
        py.detach(move || {
            let (runtime, _) = runtime::ensure_default_runtime_blocking("python.resident.matmul")
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
            Ok(Self {
                inner: match kernel {
                    Some(kernel) => ResidentMatmul::with_kernel(runtime, shape, tile, kernel),
                    None => ResidentMatmul::with_tile(runtime, shape, tile),
                }
                .map_err(error)?,
            })
        })
    }

    #[getter]
    fn shape(&self) -> (usize, usize, usize) {
        self.inner.shape().dimensions()
    }

    #[getter]
    fn tile_mnk(&self) -> (u32, u32, u32) {
        let [m, n, k] = self.inner.tile().dimensions();
        (m, n, k)
    }

    #[getter]
    fn workgroup_size(&self) -> (u32, u32, u32) {
        let [x, y, z] = self.inner.workgroup_size();
        (x, y, z)
    }

    #[getter]
    fn outputs_per_thread(&self) -> (u32, u32) {
        let [m, n] = self.inner.outputs_per_thread();
        (m, n)
    }

    #[getter]
    fn kernel(&self) -> &'static str {
        self.inner.kernel().as_str()
    }

    #[getter]
    fn generation(&self) -> u64 {
        self.inner.generation()
    }

    #[getter]
    fn output_is_current(&self) -> bool {
        self.inner.output_is_current()
    }

    fn adapter_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let info = self.inner.adapter_info();
        let result = PyDict::new(py);
        result.set_item("name", &info.name)?;
        result.set_item("vendor", info.vendor)?;
        result.set_item("device", info.device)?;
        result.set_item("backend", format!("{:?}", info.backend))?;
        result.set_item("device_type", format!("{:?}", info.device_type))?;
        result.set_item("driver", &info.driver)?;
        result.set_item("driver_info", &info.driver_info)?;
        Ok(result)
    }

    fn upload(&mut self, py: Python<'_>, lhs: &PyTensor, rhs: &PyTensor) -> PyResult<()> {
        let (m, k, n) = self.shape();
        if lhs.inner.shape() != (m, k) || rhs.inner.shape() != (k, n) {
            return Err(PyValueError::new_err(
                "operand shapes must match the workspace",
            ));
        }
        let lhs = lhs
            .inner
            .to_layout(st_tensor::Layout::RowMajor)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let rhs = rhs
            .inner
            .to_layout(st_tensor::Layout::RowMajor)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        py.detach(|| self.inner.upload(lhs.data(), rhs.data()))
            .map_err(error)
    }

    fn upload_rhs(&mut self, py: Python<'_>, rhs: &PyTensor) -> PyResult<()> {
        let (_, k, n) = self.shape();
        if rhs.inner.shape() != (k, n) {
            return Err(PyValueError::new_err("RHS shape must match the workspace"));
        }
        let rhs = rhs
            .inner
            .to_layout(st_tensor::Layout::RowMajor)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        py.detach(|| self.inner.upload_rhs(rhs.data()))
            .map_err(error)
    }

    fn set_lhs_from(&mut self, py: Python<'_>, source: &Self) -> PyResult<()> {
        py.detach(|| self.inner.set_lhs_from(&source.inner))
            .map_err(error)
    }

    #[pyo3(signature = (repetitions=1))]
    fn dispatch(&mut self, py: Python<'_>, repetitions: u32) -> PyResult<u64> {
        py.detach(|| self.inner.dispatch(repetitions))
            .map_err(error)
    }

    fn synchronize(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.synchronize()).map_err(error)
    }

    fn readback(&self, py: Python<'_>) -> PyResult<PyTensor> {
        let (rows, _, cols) = self.shape();
        let snapshot = self.inner.snapshot().map_err(error)?;
        let values = py.detach(|| snapshot.read()).map_err(error)?;
        Ok(PyTensor::from_tensor(
            st_tensor::Tensor::from_vec(rows, cols, values)
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?,
        ))
    }
}

#[cfg(not(feature = "wgpu"))]
#[pyclass(name = "WgpuMatmul", module = "spiraltorch.wgpu")]
pub(crate) struct PyWgpuMatmul;

#[cfg(not(feature = "wgpu"))]
#[pymethods]
impl PyWgpuMatmul {
    #[new]
    #[pyo3(signature = (rows, inner, cols, *, tile_mnk=None, kernel=None))]
    fn new(
        rows: usize,
        inner: usize,
        cols: usize,
        tile_mnk: Option<(u32, u32, u32)>,
        kernel: Option<&str>,
    ) -> PyResult<Self> {
        let _ = (rows, inner, cols, tile_mnk, kernel);
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "WgpuMatmul requires a wheel built with the 'wgpu' feature",
        ))
    }
}
