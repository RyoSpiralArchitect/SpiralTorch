use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Bound;

use st_tensor::fractional::{fracdiff_gl_1d as rust_fracdiff_gl_1d, PadMode};

#[pyfunction]
#[pyo3(signature = (xs, alpha, kernel_len, pad="constant", pad_constant=None))]
fn fracdiff_gl_1d(
    xs: Vec<f32>,
    alpha: f32,
    kernel_len: usize,
    pad: &str,
    pad_constant: Option<f32>,
) -> PyResult<Vec<f32>> {
    let mode = match pad.to_ascii_lowercase().as_str() {
        "constant" => PadMode::Constant(pad_constant.unwrap_or(0.0)),
        "edge" => PadMode::Edge,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown pad mode '{other}', expected 'constant' or 'edge'"
            )))
        }
    };
    rust_fracdiff_gl_1d(&xs, alpha, kernel_len, mode)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (alpha, tol=1e-6, max_len=8192))]
fn gl_coeffs_adaptive(alpha: f32, tol: f32, max_len: usize) -> Vec<f32> {
    st_frac::gl_coeffs_adaptive(alpha, tol, max_len)
}

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "frac")?;
    module.add_function(wrap_pyfunction!(gl_coeffs_adaptive, &module)?)?;
    module.add_function(wrap_pyfunction!(fracdiff_gl_1d, &module)?)?;
    module.add("__doc__", "Fractional differencing (Grünwald–Letnikov)")?;
    module.add("__all__", vec!["gl_coeffs_adaptive", "fracdiff_gl_1d"])?;
    parent.add_submodule(&module)?;
    Ok(())
}
