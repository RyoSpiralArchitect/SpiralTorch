use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;
use st_frac::mellin_types::ComplexScalar;
use st_frac::zspace::{
    evaluate_weighted_series_many, prepare_weighted_series, trapezoidal_weights,
};

use crate::introspect;

fn pyerr<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Python entry point for evaluating weighted series in Z-space while releasing the GIL.
///
/// Args:
///     real: Real part of the complex samples.
///     imag: Imaginary part of the complex samples.
///     z_re: Real part of the evaluation points.
///     z_im: Imaginary part of the evaluation points.
#[pyfunction]
#[pyo3(signature = (real, imag, z_re, z_im))]
fn zspace_eval(
    py: Python<'_>,
    real: Vec<f32>,
    imag: Vec<f32>,
    z_re: Vec<f32>,
    z_im: Vec<f32>,
) -> PyResult<Vec<(f32, f32)>> {
    if real.len() != imag.len() {
        return Err(PyRuntimeError::new_err("len(real) != len(imag)"));
    }
    if z_re.len() != z_im.len() {
        return Err(PyRuntimeError::new_err("len(z_re) != len(z_im)"));
    }

    let samples: Vec<ComplexScalar> = real
        .into_iter()
        .zip(imag)
        .map(|(r, i)| ComplexScalar::new(r, i))
        .collect();
    let zs: Vec<ComplexScalar> = z_re
        .into_iter()
        .zip(z_im)
        .map(|(r, i)| ComplexScalar::new(r, i))
        .collect();

    let out = py.allow_threads(|| -> Result<_, PyErr> {
        let w = trapezoidal_weights(samples.len()).map_err(pyerr)?;
        let coeff = prepare_weighted_series(&samples, &w).map_err(pyerr)?;
        let vals = evaluate_weighted_series_many(&coeff, &zs).map_err(pyerr)?;
        Ok(vals.into_iter().map(|c| (c.re, c.im)).collect())
    })?;

    Ok(out)
}

pub fn register(py: Python<'_>, module: &Bound<PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(zspace_eval, module)?)?;
    introspect::register_top_level(py, module)?;

    let zspace_module = PyModule::new_bound(py, "zspace")?;
    zspace_module.add_function(wrap_pyfunction!(zspace_eval, &zspace_module)?)?;
    introspect::register_submodule(py, &zspace_module)?;
    module.add_submodule(&zspace_module)?;
    for attr in [
        "ZSpaceSpinBand",
        "ZSpaceRadiusBand",
        "ZSpaceRegionKey",
        "ZSpaceRegionDescriptor",
        "SoftlogicEllipticSample",
        "SoftlogicZFeedback",
    ] {
        if let Ok(value) = zspace_module.getattr(attr) {
            module.add(attr, value)?;
        }
    }
    Ok(())
}
