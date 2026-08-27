use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use pyo3::wrap_pyfunction;
use st_core::runtime::zspace_stochastic_schrodinger::{
    run_zspace_stochastic_schrodinger_forward, run_zspace_stochastic_schrodinger_vjp,
    validate_zspace_stochastic_schrodinger_forward_value,
    validate_zspace_stochastic_schrodinger_vjp_value, ZSpaceStochasticSchrodingerForwardRequest,
    ZSpaceStochasticSchrodingerVjpRequest, ZSPACE_STOCHASTIC_SCHRODINGER_EVIDENCE_BOUNDARY,
    ZSPACE_STOCHASTIC_SCHRODINGER_FORWARD_CONTRACT_VERSION,
    ZSPACE_STOCHASTIC_SCHRODINGER_FORWARD_KIND, ZSPACE_STOCHASTIC_SCHRODINGER_ID_RULE,
    ZSPACE_STOCHASTIC_SCHRODINGER_MAX_FEATURES, ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_BYTES,
    ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_DEPTH,
    ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_NODES, ZSPACE_STOCHASTIC_SCHRODINGER_MAX_ROWS,
    ZSPACE_STOCHASTIC_SCHRODINGER_MAX_VALUES, ZSPACE_STOCHASTIC_SCHRODINGER_PROTOCOL_OWNER,
    ZSPACE_STOCHASTIC_SCHRODINGER_SEMANTIC_BACKEND, ZSPACE_STOCHASTIC_SCHRODINGER_SEMANTIC_OWNER,
    ZSPACE_STOCHASTIC_SCHRODINGER_VJP_CONTRACT_VERSION, ZSPACE_STOCHASTIC_SCHRODINGER_VJP_KIND,
    ZSPACE_STOCHASTIC_SCHRODINGER_VJP_SEMANTICS,
};

fn json_error(context: &str, error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!("{context}: {error}"))
}

fn mapping_value(value: &Bound<'_, PyAny>, label: &str) -> PyResult<serde_json::Value> {
    let value = crate::json::py_to_json_bounded(
        value,
        ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_BYTES,
        ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_NODES,
        ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_DEPTH as usize,
        label,
    )?;
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
fn _zspace_stochastic_schrodinger_forward(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let request = mapping_value(request, "Z-space stochastic Schrodinger forward request")?;
    let request: ZSpaceStochasticSchrodingerForwardRequest = serde_json::from_value(request)
        .map_err(|error| {
            json_error(
                "invalid Z-space stochastic Schrodinger forward request",
                error,
            )
        })?;
    let receipt = py
        .allow_threads(|| run_zspace_stochastic_schrodinger_forward(request))
        .map_err(|error| json_error("Z-space stochastic Schrodinger forward failed", error))?;
    response_to_py(
        py,
        &receipt,
        "Z-space stochastic Schrodinger forward encoding failed",
    )
}

#[pyfunction]
fn _zspace_stochastic_schrodinger_forward_validate(
    py: Python<'_>,
    receipt: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let receipt = mapping_value(receipt, "Z-space stochastic Schrodinger forward receipt")?;
    let receipt =
        validate_zspace_stochastic_schrodinger_forward_value(receipt).map_err(|error| {
            json_error(
                "Z-space stochastic Schrodinger forward validation failed",
                error,
            )
        })?;
    response_to_py(
        py,
        &receipt,
        "Z-space stochastic Schrodinger forward encoding failed",
    )
}

#[pyfunction]
fn _zspace_stochastic_schrodinger_vjp(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let request = mapping_value(request, "Z-space stochastic Schrodinger VJP request")?;
    let request: ZSpaceStochasticSchrodingerVjpRequest = serde_json::from_value(request)
        .map_err(|error| json_error("invalid Z-space stochastic Schrodinger VJP request", error))?;
    let receipt = py
        .allow_threads(|| run_zspace_stochastic_schrodinger_vjp(request))
        .map_err(|error| json_error("Z-space stochastic Schrodinger VJP failed", error))?;
    response_to_py(
        py,
        &receipt,
        "Z-space stochastic Schrodinger VJP encoding failed",
    )
}

#[pyfunction]
fn _zspace_stochastic_schrodinger_vjp_validate(
    py: Python<'_>,
    receipt: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let receipt = mapping_value(receipt, "Z-space stochastic Schrodinger VJP receipt")?;
    let receipt = validate_zspace_stochastic_schrodinger_vjp_value(receipt).map_err(|error| {
        json_error(
            "Z-space stochastic Schrodinger VJP validation failed",
            error,
        )
    })?;
    response_to_py(
        py,
        &receipt,
        "Z-space stochastic Schrodinger VJP encoding failed",
    )
}

pub(crate) fn register(_py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    for (name, value) in [
        (
            "ZSPACE_STOCHASTIC_SCHRODINGER_FORWARD_CONTRACT_VERSION",
            ZSPACE_STOCHASTIC_SCHRODINGER_FORWARD_CONTRACT_VERSION,
        ),
        (
            "ZSPACE_STOCHASTIC_SCHRODINGER_FORWARD_KIND",
            ZSPACE_STOCHASTIC_SCHRODINGER_FORWARD_KIND,
        ),
        (
            "ZSPACE_STOCHASTIC_SCHRODINGER_VJP_CONTRACT_VERSION",
            ZSPACE_STOCHASTIC_SCHRODINGER_VJP_CONTRACT_VERSION,
        ),
        (
            "ZSPACE_STOCHASTIC_SCHRODINGER_VJP_KIND",
            ZSPACE_STOCHASTIC_SCHRODINGER_VJP_KIND,
        ),
        (
            "ZSPACE_STOCHASTIC_SCHRODINGER_PROTOCOL_OWNER",
            ZSPACE_STOCHASTIC_SCHRODINGER_PROTOCOL_OWNER,
        ),
        (
            "ZSPACE_STOCHASTIC_SCHRODINGER_SEMANTIC_OWNER",
            ZSPACE_STOCHASTIC_SCHRODINGER_SEMANTIC_OWNER,
        ),
        (
            "ZSPACE_STOCHASTIC_SCHRODINGER_SEMANTIC_BACKEND",
            ZSPACE_STOCHASTIC_SCHRODINGER_SEMANTIC_BACKEND,
        ),
        (
            "ZSPACE_STOCHASTIC_SCHRODINGER_ID_RULE",
            ZSPACE_STOCHASTIC_SCHRODINGER_ID_RULE,
        ),
        (
            "ZSPACE_STOCHASTIC_SCHRODINGER_VJP_SEMANTICS",
            ZSPACE_STOCHASTIC_SCHRODINGER_VJP_SEMANTICS,
        ),
        (
            "ZSPACE_STOCHASTIC_SCHRODINGER_EVIDENCE_BOUNDARY",
            ZSPACE_STOCHASTIC_SCHRODINGER_EVIDENCE_BOUNDARY,
        ),
    ] {
        parent.add(name, value)?;
    }
    parent.add(
        "ZSPACE_STOCHASTIC_SCHRODINGER_MAX_VALUES",
        ZSPACE_STOCHASTIC_SCHRODINGER_MAX_VALUES,
    )?;
    parent.add(
        "ZSPACE_STOCHASTIC_SCHRODINGER_MAX_ROWS",
        ZSPACE_STOCHASTIC_SCHRODINGER_MAX_ROWS,
    )?;
    parent.add(
        "ZSPACE_STOCHASTIC_SCHRODINGER_MAX_FEATURES",
        ZSPACE_STOCHASTIC_SCHRODINGER_MAX_FEATURES,
    )?;
    parent.add_function(wrap_pyfunction!(
        _zspace_stochastic_schrodinger_forward,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(
        _zspace_stochastic_schrodinger_forward_validate,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(
        _zspace_stochastic_schrodinger_vjp,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(
        _zspace_stochastic_schrodinger_vjp_validate,
        parent
    )?)?;
    Ok(())
}
