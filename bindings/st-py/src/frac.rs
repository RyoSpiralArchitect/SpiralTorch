use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Bound;

use st_frac::mellin::MellinLogGrid;
use st_frac::mellin_types::{ComplexScalar, Scalar};
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
fn gl_coeffs_adaptive(alpha: f32, tol: f32, max_len: usize) -> PyResult<Vec<f32>> {
    st_frac::gl_coeffs_adaptive(alpha, tol, max_len)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

fn mellin_err_to_py(err: st_frac::mellin_types::MellinError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[pyclass(module = "spiraltorch.frac", name = "MellinLogGrid")]
#[derive(Clone, Debug)]
struct PyMellinLogGrid {
    inner: MellinLogGrid,
}

#[pymethods]
impl PyMellinLogGrid {
    #[new]
    fn new(log_start: Scalar, log_step: Scalar, samples: Vec<ComplexScalar>) -> PyResult<Self> {
        let inner = MellinLogGrid::new(log_start, log_step, samples).map_err(mellin_err_to_py)?;
        Ok(Self { inner })
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[getter]
    fn log_start(&self) -> Scalar {
        self.inner.log_start()
    }

    #[getter]
    fn log_step(&self) -> Scalar {
        self.inner.log_step()
    }

    #[getter]
    fn samples(&self) -> Vec<ComplexScalar> {
        self.inner.samples().to_vec()
    }

    #[getter]
    fn weights(&self) -> Vec<Scalar> {
        self.inner.weights().to_vec()
    }

    #[getter]
    fn support(&self) -> (Scalar, Scalar) {
        self.inner.support()
    }

    fn weighted_series(&self) -> PyResult<Vec<ComplexScalar>> {
        self.inner.weighted_series().map_err(mellin_err_to_py)
    }

    fn evaluate(&self, s: ComplexScalar) -> PyResult<ComplexScalar> {
        self.inner.evaluate(s).map_err(mellin_err_to_py)
    }

    fn evaluate_many(&self, s_values: Vec<ComplexScalar>) -> PyResult<Vec<ComplexScalar>> {
        self.inner.evaluate_many(&s_values).map_err(mellin_err_to_py)
    }

    fn evaluate_with_series(
        &self,
        s: ComplexScalar,
        weighted: Vec<ComplexScalar>,
    ) -> PyResult<ComplexScalar> {
        self.inner
            .evaluate_with_series(s, &weighted)
            .map_err(mellin_err_to_py)
    }

    fn evaluate_many_with_series(
        &self,
        s_values: Vec<ComplexScalar>,
        weighted: Vec<ComplexScalar>,
    ) -> PyResult<Vec<ComplexScalar>> {
        self.inner
            .evaluate_many_with_series(&s_values, &weighted)
            .map_err(mellin_err_to_py)
    }

    fn evaluate_vertical_line(
        &self,
        real: Scalar,
        imag_values: Vec<Scalar>,
    ) -> PyResult<Vec<ComplexScalar>> {
        self.inner
            .evaluate_vertical_line(real, &imag_values)
            .map_err(mellin_err_to_py)
    }

    fn hilbert_inner_product(&self, other: &PyMellinLogGrid) -> PyResult<ComplexScalar> {
        self.inner
            .hilbert_inner_product(&other.inner)
            .map_err(mellin_err_to_py)
    }

    fn hilbert_norm(&self) -> PyResult<Scalar> {
        self.inner.hilbert_norm().map_err(mellin_err_to_py)
    }

    fn __repr__(&self) -> String {
        format!(
            "MellinLogGrid(len={}, log_start={:.4}, log_step={:.4})",
            self.inner.len(),
            self.inner.log_start(),
            self.inner.log_step()
        )
    }
}

pub(crate) fn register(py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(py, "frac")?;
    module.add_function(wrap_pyfunction!(gl_coeffs_adaptive, &module)?)?;
    module.add_function(wrap_pyfunction!(fracdiff_gl_1d, &module)?)?;
    module.add_class::<PyMellinLogGrid>()?;
    module.add("__doc__", "Fractional differencing + Mellin tooling")?;
    module.add(
        "__all__",
        vec!["gl_coeffs_adaptive", "fracdiff_gl_1d", "MellinLogGrid"],
    )?;
    parent.add_submodule(&module)?;
    Ok(())
}
