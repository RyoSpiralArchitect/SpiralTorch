use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use pyo3::wrap_pyfunction;
use st_core::runtime::zspace_generation_evidence::{
    summarize_zspace_generation_evidence, validate_zspace_generation_evidence_value,
    ZSpaceGenerationEvidenceRequest, ZSPACE_GENERATION_EVIDENCE_CONTRACT_VERSION,
    ZSPACE_GENERATION_EVIDENCE_KIND, ZSPACE_GENERATION_EVIDENCE_LOOP_SCORE_RULE,
    ZSPACE_GENERATION_EVIDENCE_MAX_INGRESS_BYTES, ZSPACE_GENERATION_EVIDENCE_MAX_INGRESS_DEPTH,
    ZSPACE_GENERATION_EVIDENCE_MAX_INGRESS_NODES, ZSPACE_GENERATION_EVIDENCE_MAX_SAFE_INTEGER,
    ZSPACE_GENERATION_EVIDENCE_MAX_SAMPLES, ZSPACE_GENERATION_EVIDENCE_MAX_TOKENS_PER_SAMPLE,
    ZSPACE_GENERATION_EVIDENCE_MAX_TOTAL_TOKENS, ZSPACE_GENERATION_EVIDENCE_METRIC_RULE,
    ZSPACE_GENERATION_EVIDENCE_NGRAM_ORDERS, ZSPACE_GENERATION_EVIDENCE_PERIODIC_SUFFIX_MAX_PERIOD,
    ZSPACE_GENERATION_EVIDENCE_PERIODIC_SUFFIX_MIN_REPETITIONS,
    ZSPACE_GENERATION_EVIDENCE_SEMANTIC_BACKEND, ZSPACE_GENERATION_EVIDENCE_SEMANTIC_OWNER,
};

fn json_error(context: &str, error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!("{context}: {error}"))
}

fn request_value(request: &Bound<'_, PyAny>, label: &str) -> PyResult<serde_json::Value> {
    let value = crate::json::py_to_json_bounded(
        request,
        ZSPACE_GENERATION_EVIDENCE_MAX_INGRESS_BYTES,
        ZSPACE_GENERATION_EVIDENCE_MAX_INGRESS_NODES,
        ZSPACE_GENERATION_EVIDENCE_MAX_INGRESS_DEPTH as usize,
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
) -> PyResult<Py<PyAny>> {
    let value = serde_json::to_value(response).map_err(|error| json_error(context, error))?;
    crate::json::json_to_py(py, &value)
}

#[pyfunction]
fn _zspace_generation_evidence(py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let request = request_value(request, "Z-space generation evidence request")?;
    let request: ZSpaceGenerationEvidenceRequest = serde_json::from_value(request)
        .map_err(|error| json_error("invalid Z-space generation evidence request", error))?;
    let report = py
        .detach(|| summarize_zspace_generation_evidence(request))
        .map_err(|error| json_error("Z-space generation evidence aggregation failed", error))?;
    response_to_py(py, &report, "Z-space generation evidence encoding failed")
}

#[pyfunction]
fn _zspace_generation_evidence_validate(
    py: Python<'_>,
    report: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let report = request_value(report, "Z-space generation evidence report")?;
    let report = validate_zspace_generation_evidence_value(report)
        .map_err(|error| json_error("Z-space generation evidence validation failed", error))?;
    response_to_py(py, &report, "Z-space generation evidence encoding failed")
}

pub(crate) fn register(_py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add(
        "ZSPACE_GENERATION_EVIDENCE_CONTRACT_VERSION",
        ZSPACE_GENERATION_EVIDENCE_CONTRACT_VERSION,
    )?;
    parent.add(
        "ZSPACE_GENERATION_EVIDENCE_KIND",
        ZSPACE_GENERATION_EVIDENCE_KIND,
    )?;
    parent.add(
        "ZSPACE_GENERATION_EVIDENCE_SEMANTIC_OWNER",
        ZSPACE_GENERATION_EVIDENCE_SEMANTIC_OWNER,
    )?;
    parent.add(
        "ZSPACE_GENERATION_EVIDENCE_SEMANTIC_BACKEND",
        ZSPACE_GENERATION_EVIDENCE_SEMANTIC_BACKEND,
    )?;
    parent.add(
        "ZSPACE_GENERATION_EVIDENCE_METRIC_RULE",
        ZSPACE_GENERATION_EVIDENCE_METRIC_RULE,
    )?;
    parent.add(
        "ZSPACE_GENERATION_EVIDENCE_LOOP_SCORE_RULE",
        ZSPACE_GENERATION_EVIDENCE_LOOP_SCORE_RULE,
    )?;
    parent.add(
        "ZSPACE_GENERATION_EVIDENCE_NGRAM_ORDERS",
        ZSPACE_GENERATION_EVIDENCE_NGRAM_ORDERS,
    )?;
    parent.add(
        "ZSPACE_GENERATION_EVIDENCE_PERIODIC_SUFFIX_MAX_PERIOD",
        ZSPACE_GENERATION_EVIDENCE_PERIODIC_SUFFIX_MAX_PERIOD,
    )?;
    parent.add(
        "ZSPACE_GENERATION_EVIDENCE_PERIODIC_SUFFIX_MIN_REPETITIONS",
        ZSPACE_GENERATION_EVIDENCE_PERIODIC_SUFFIX_MIN_REPETITIONS,
    )?;
    parent.add(
        "ZSPACE_GENERATION_EVIDENCE_MAX_SAFE_INTEGER",
        ZSPACE_GENERATION_EVIDENCE_MAX_SAFE_INTEGER,
    )?;
    parent.add(
        "ZSPACE_GENERATION_EVIDENCE_MAX_SAMPLES",
        ZSPACE_GENERATION_EVIDENCE_MAX_SAMPLES,
    )?;
    parent.add(
        "ZSPACE_GENERATION_EVIDENCE_MAX_TOKENS_PER_SAMPLE",
        ZSPACE_GENERATION_EVIDENCE_MAX_TOKENS_PER_SAMPLE,
    )?;
    parent.add(
        "ZSPACE_GENERATION_EVIDENCE_MAX_TOTAL_TOKENS",
        ZSPACE_GENERATION_EVIDENCE_MAX_TOTAL_TOKENS,
    )?;
    parent.add_function(wrap_pyfunction!(_zspace_generation_evidence, parent)?)?;
    parent.add_function(wrap_pyfunction!(
        _zspace_generation_evidence_validate,
        parent
    )?)?;
    Ok(())
}
