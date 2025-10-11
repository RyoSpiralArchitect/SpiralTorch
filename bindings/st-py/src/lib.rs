
use pyo3::prelude::*;

#[pyfunction]
fn tensor_from_list(py: Python, list: Vec<f32>) -> PyResult<PyObject> {
    let t = st_core::tensor::Tensor::from_array(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[list.len()]), list).unwrap());
    Py::new(py, crate::spiraltorch::TensorPy{ inner: t }).map(|p| p.to_object(py))
}

#[pymodule]
fn spiraltorch(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tensor_from_list, m)?)?;
    crate::spiraltorch::register(py, m)?;
    Ok(())
}

pub mod spiraltorch;
