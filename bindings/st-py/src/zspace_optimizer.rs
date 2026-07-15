use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use pyo3::wrap_pyfunction;
use st_core::runtime::zspace_optimizer::{
    initialize_zspace_meta_optimizer, restore_zspace_meta_optimizer,
    transition_zspace_meta_optimizer, ZSpaceMetaOptimizerConfig, ZSpaceMetaOptimizerRestoreRequest,
    ZSpaceMetaOptimizerStepRequest,
};

fn json_error(context: &str, error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!("{context}: {error}"))
}

fn request_value(request: &Bound<'_, PyAny>, label: &str) -> PyResult<serde_json::Value> {
    let value = crate::json::py_to_json(request)?;
    if !value.is_object() {
        return Err(PyValueError::new_err(format!("{label} must be a mapping")));
    }
    Ok(value)
}

fn response_to_py<T: serde::Serialize>(
    py: Python<'_>,
    response: &T,
    context: &str,
) -> PyResult<PyObject> {
    let value = serde_json::to_value(response).map_err(|error| json_error(context, error))?;
    crate::json::json_to_py(py, &value)
}

#[pyfunction]
fn _zspace_meta_optimizer_init(py: Python<'_>, config: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let config = request_value(config, "Z-space meta-optimizer config")?;
    let config: ZSpaceMetaOptimizerConfig = serde_json::from_value(config)
        .map_err(|error| json_error("invalid Z-space meta-optimizer config", error))?;
    let checkpoint = initialize_zspace_meta_optimizer(config)
        .map_err(|error| json_error("Z-space meta-optimizer initialization failed", error))?;
    response_to_py(
        py,
        &checkpoint,
        "Z-space meta-optimizer checkpoint encoding failed",
    )
}

#[pyfunction]
fn _zspace_meta_optimizer_restore(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let request = request_value(request, "Z-space meta-optimizer restore request")?;
    let request: ZSpaceMetaOptimizerRestoreRequest = serde_json::from_value(request)
        .map_err(|error| json_error("invalid Z-space meta-optimizer restore request", error))?;
    let checkpoint = restore_zspace_meta_optimizer(request)
        .map_err(|error| json_error("Z-space meta-optimizer restore failed", error))?;
    response_to_py(
        py,
        &checkpoint,
        "Z-space meta-optimizer checkpoint encoding failed",
    )
}

#[pyfunction]
fn _zspace_meta_optimizer_step(py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let request = request_value(request, "Z-space meta-optimizer step request")?;
    let request: ZSpaceMetaOptimizerStepRequest = serde_json::from_value(request)
        .map_err(|error| json_error("invalid Z-space meta-optimizer step request", error))?;
    let report = py
        .allow_threads(|| transition_zspace_meta_optimizer(request))
        .map_err(|error| json_error("Z-space meta-optimizer step failed", error))?;
    response_to_py(py, &report, "Z-space meta-optimizer report encoding failed")
}

pub(crate) fn register(_py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_function(wrap_pyfunction!(_zspace_meta_optimizer_init, parent)?)?;
    parent.add_function(wrap_pyfunction!(_zspace_meta_optimizer_restore, parent)?)?;
    parent.add_function(wrap_pyfunction!(_zspace_meta_optimizer_step, parent)?)?;
    Ok(())
}
