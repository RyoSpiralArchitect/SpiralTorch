
use pyo3::{prelude::*, types::PyTuple};
use st_core::tensor::Tensor;

#[pyclass]
pub struct TensorPy{ pub inner: Tensor }

#[pymethods]
impl TensorPy {
    #[new]
    fn new() -> Self { TensorPy{ inner: Tensor::zeros(&[1]) } }

    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let v = self.inner.data().into_raw_vec();
        let np = py.import("numpy")?;
        np.getattr("array")?.call1((v,))
    }
}

pub fn register(_py: Python, m: &pyo3::prelude::PyModule) -> PyResult<()> {
    m.add_class::<TensorPy>()?;
    Ok(())
}
