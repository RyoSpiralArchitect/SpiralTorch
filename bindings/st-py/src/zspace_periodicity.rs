use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use pyo3::wrap_pyfunction;
use st_core::runtime::zspace_periodicity::{
    analyze_zspace_periodicity, validate_zspace_periodicity_value, ZSpacePeriodicityRequest,
    ZSPACE_PERIODICITY_ANALYSIS_ID_RULE, ZSPACE_PERIODICITY_CONTRACT_VERSION,
    ZSPACE_PERIODICITY_EVIDENCE_BOUNDARY, ZSPACE_PERIODICITY_KIND,
    ZSPACE_PERIODICITY_MAX_COMPARISON_WORK, ZSPACE_PERIODICITY_MAX_INGRESS_BYTES,
    ZSPACE_PERIODICITY_MAX_INGRESS_DEPTH, ZSPACE_PERIODICITY_MAX_INGRESS_NODES,
    ZSPACE_PERIODICITY_MAX_MINIMUM_REPETITIONS, ZSPACE_PERIODICITY_MAX_PERIOD,
    ZSPACE_PERIODICITY_MAX_SAFE_INTEGER, ZSPACE_PERIODICITY_MAX_TOKENS, ZSPACE_PERIODICITY_RULE,
    ZSPACE_PERIODICITY_SEMANTIC_BACKEND, ZSPACE_PERIODICITY_SEMANTIC_OWNER,
    ZSPACE_PERIODIC_SUFFIX_MAX_PERIOD, ZSPACE_PERIODIC_SUFFIX_MIN_REPETITIONS,
};

fn json_error(context: &str, error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!("{context}: {error}"))
}

fn mapping_value(value: &Bound<'_, PyAny>, label: &str) -> PyResult<serde_json::Value> {
    let value = crate::json::py_to_json_bounded(
        value,
        ZSPACE_PERIODICITY_MAX_INGRESS_BYTES,
        ZSPACE_PERIODICITY_MAX_INGRESS_NODES,
        ZSPACE_PERIODICITY_MAX_INGRESS_DEPTH as usize,
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
fn _zspace_periodicity(py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let request = mapping_value(request, "Z-space periodicity request")?;
    let request: ZSpacePeriodicityRequest = serde_json::from_value(request)
        .map_err(|error| json_error("invalid Z-space periodicity request", error))?;
    let report = py
        .allow_threads(|| analyze_zspace_periodicity(request))
        .map_err(|error| json_error("Z-space periodicity analysis failed", error))?;
    response_to_py(py, &report, "Z-space periodicity encoding failed")
}

#[pyfunction]
fn _zspace_periodicity_validate(py: Python<'_>, report: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let report = mapping_value(report, "Z-space periodicity report")?;
    let report = validate_zspace_periodicity_value(report)
        .map_err(|error| json_error("Z-space periodicity validation failed", error))?;
    response_to_py(py, &report, "Z-space periodicity encoding failed")
}

pub(crate) fn register(_py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add(
        "ZSPACE_PERIODICITY_CONTRACT_VERSION",
        ZSPACE_PERIODICITY_CONTRACT_VERSION,
    )?;
    parent.add("ZSPACE_PERIODICITY_KIND", ZSPACE_PERIODICITY_KIND)?;
    parent.add(
        "ZSPACE_PERIODICITY_SEMANTIC_OWNER",
        ZSPACE_PERIODICITY_SEMANTIC_OWNER,
    )?;
    parent.add(
        "ZSPACE_PERIODICITY_SEMANTIC_BACKEND",
        ZSPACE_PERIODICITY_SEMANTIC_BACKEND,
    )?;
    parent.add("ZSPACE_PERIODICITY_RULE", ZSPACE_PERIODICITY_RULE)?;
    parent.add(
        "ZSPACE_PERIODICITY_ANALYSIS_ID_RULE",
        ZSPACE_PERIODICITY_ANALYSIS_ID_RULE,
    )?;
    parent.add(
        "ZSPACE_PERIODICITY_EVIDENCE_BOUNDARY",
        ZSPACE_PERIODICITY_EVIDENCE_BOUNDARY,
    )?;
    parent.add(
        "ZSPACE_PERIODIC_SUFFIX_MAX_PERIOD",
        ZSPACE_PERIODIC_SUFFIX_MAX_PERIOD,
    )?;
    parent.add(
        "ZSPACE_PERIODIC_SUFFIX_MIN_REPETITIONS",
        ZSPACE_PERIODIC_SUFFIX_MIN_REPETITIONS,
    )?;
    parent.add(
        "ZSPACE_PERIODICITY_MAX_TOKENS",
        ZSPACE_PERIODICITY_MAX_TOKENS,
    )?;
    parent.add(
        "ZSPACE_PERIODICITY_MAX_PERIOD",
        ZSPACE_PERIODICITY_MAX_PERIOD,
    )?;
    parent.add(
        "ZSPACE_PERIODICITY_MAX_MINIMUM_REPETITIONS",
        ZSPACE_PERIODICITY_MAX_MINIMUM_REPETITIONS,
    )?;
    parent.add(
        "ZSPACE_PERIODICITY_MAX_COMPARISON_WORK",
        ZSPACE_PERIODICITY_MAX_COMPARISON_WORK,
    )?;
    parent.add(
        "ZSPACE_PERIODICITY_MAX_SAFE_INTEGER",
        ZSPACE_PERIODICITY_MAX_SAFE_INTEGER,
    )?;
    parent.add_function(wrap_pyfunction!(_zspace_periodicity, parent)?)?;
    parent.add_function(wrap_pyfunction!(_zspace_periodicity_validate, parent)?)?;
    Ok(())
}
