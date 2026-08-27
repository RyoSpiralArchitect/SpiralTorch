use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use pyo3::wrap_pyfunction;
use st_core::runtime::zspace_runtime_protocol_catalog::{
    validate_zspace_runtime_protocol_catalog_value, zspace_runtime_protocol_catalog,
    ZSPACE_RUNTIME_PROTOCOL_CATALOG_CONTRACT_VERSION, ZSPACE_RUNTIME_PROTOCOL_CATALOG_ID_RULE,
    ZSPACE_RUNTIME_PROTOCOL_CATALOG_KIND, ZSPACE_RUNTIME_PROTOCOL_CATALOG_SEMANTIC_BACKEND,
    ZSPACE_RUNTIME_PROTOCOL_CATALOG_SEMANTIC_OWNER, ZSPACE_RUNTIME_PROTOCOL_CATALOG_STATUS,
};

const PY_PROTOCOL_CATALOG_MAX_INGRESS_BYTES: u64 = 1_024 * 1_024;
const PY_PROTOCOL_CATALOG_MAX_INGRESS_NODES: u64 = 20_000;
const PY_PROTOCOL_CATALOG_MAX_INGRESS_DEPTH: usize = 16;

fn json_error(context: &str, error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!("{context}: {error}"))
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
fn _zspace_runtime_protocol_catalog(py: Python<'_>) -> PyResult<PyObject> {
    let catalog = py
        .allow_threads(zspace_runtime_protocol_catalog)
        .map_err(|error| json_error("Z-space runtime protocol catalog failed", error))?;
    response_to_py(
        py,
        &catalog,
        "Z-space runtime protocol catalog encoding failed",
    )
}

#[pyfunction]
fn _zspace_runtime_protocol_catalog_validate(
    py: Python<'_>,
    catalog: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let value = crate::json::py_to_json_bounded(
        catalog,
        PY_PROTOCOL_CATALOG_MAX_INGRESS_BYTES,
        PY_PROTOCOL_CATALOG_MAX_INGRESS_NODES,
        PY_PROTOCOL_CATALOG_MAX_INGRESS_DEPTH,
        "Z-space runtime protocol catalog",
    )?;
    let catalog = py
        .allow_threads(|| validate_zspace_runtime_protocol_catalog_value(value))
        .map_err(|error| json_error("Z-space runtime protocol catalog validation failed", error))?;
    response_to_py(
        py,
        &catalog,
        "Z-space runtime protocol catalog encoding failed",
    )
}

pub(crate) fn register(_py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    for (name, value) in [
        (
            "ZSPACE_RUNTIME_PROTOCOL_CATALOG_CONTRACT_VERSION",
            ZSPACE_RUNTIME_PROTOCOL_CATALOG_CONTRACT_VERSION,
        ),
        (
            "ZSPACE_RUNTIME_PROTOCOL_CATALOG_ID_RULE",
            ZSPACE_RUNTIME_PROTOCOL_CATALOG_ID_RULE,
        ),
        (
            "ZSPACE_RUNTIME_PROTOCOL_CATALOG_KIND",
            ZSPACE_RUNTIME_PROTOCOL_CATALOG_KIND,
        ),
        (
            "ZSPACE_RUNTIME_PROTOCOL_CATALOG_SEMANTIC_OWNER",
            ZSPACE_RUNTIME_PROTOCOL_CATALOG_SEMANTIC_OWNER,
        ),
        (
            "ZSPACE_RUNTIME_PROTOCOL_CATALOG_SEMANTIC_BACKEND",
            ZSPACE_RUNTIME_PROTOCOL_CATALOG_SEMANTIC_BACKEND,
        ),
        (
            "ZSPACE_RUNTIME_PROTOCOL_CATALOG_STATUS",
            ZSPACE_RUNTIME_PROTOCOL_CATALOG_STATUS,
        ),
    ] {
        parent.add(name, value)?;
    }
    parent.add_function(wrap_pyfunction!(_zspace_runtime_protocol_catalog, parent)?)?;
    parent.add_function(wrap_pyfunction!(
        _zspace_runtime_protocol_catalog_validate,
        parent
    )?)?;
    Ok(())
}
