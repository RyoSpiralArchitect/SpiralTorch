use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use pyo3::wrap_pyfunction;
use st_core::runtime::zspace_repetition_unlikelihood::{
    plan_zspace_repetition_unlikelihood, validate_zspace_repetition_unlikelihood_value,
    validate_zspace_repetition_unlikelihood_value_trusted_legacy_replay,
    ZSpaceRepetitionUnlikelihoodRequest, ZSPACE_REPETITION_UNLIKELIHOOD_CANDIDATE_RULE,
    ZSPACE_REPETITION_UNLIKELIHOOD_CONTRACT_VERSION,
    ZSPACE_REPETITION_UNLIKELIHOOD_DIFFERENTIATION_OWNER, ZSPACE_REPETITION_UNLIKELIHOOD_KIND,
    ZSPACE_REPETITION_UNLIKELIHOOD_MATERIALIZED_PLAN_BYTE_RULE,
    ZSPACE_REPETITION_UNLIKELIHOOD_MAX_CANDIDATES_PER_POSITION,
    ZSPACE_REPETITION_UNLIKELIHOOD_MAX_CONTEXT_WINDOW,
    ZSPACE_REPETITION_UNLIKELIHOOD_MAX_MATERIALIZED_PLAN_BYTES,
    ZSPACE_REPETITION_UNLIKELIHOOD_MAX_NGRAM_ORDER,
    ZSPACE_REPETITION_UNLIKELIHOOD_MAX_PROPOSAL_TOP_K,
    ZSPACE_REPETITION_UNLIKELIHOOD_MAX_SAFE_INTEGER, ZSPACE_REPETITION_UNLIKELIHOOD_MAX_SEQUENCES,
    ZSPACE_REPETITION_UNLIKELIHOOD_MAX_STRENGTH,
    ZSPACE_REPETITION_UNLIKELIHOOD_MAX_TOKENS_PER_SEQUENCE,
    ZSPACE_REPETITION_UNLIKELIHOOD_MAX_TOTAL_TOKENS, ZSPACE_REPETITION_UNLIKELIHOOD_MAX_WORK_UNITS,
    ZSPACE_REPETITION_UNLIKELIHOOD_MIN_NGRAM_ORDER, ZSPACE_REPETITION_UNLIKELIHOOD_OBJECTIVE_RULE,
    ZSPACE_REPETITION_UNLIKELIHOOD_PERIODIC_SUFFIX_MAX_PERIOD,
    ZSPACE_REPETITION_UNLIKELIHOOD_PERIODIC_SUFFIX_MIN_REPETITIONS,
    ZSPACE_REPETITION_UNLIKELIHOOD_PROBABILITY_EPSILON,
    ZSPACE_REPETITION_UNLIKELIHOOD_PROPOSAL_OWNER, ZSPACE_REPETITION_UNLIKELIHOOD_PROPOSAL_RULE,
    ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_BACKEND, ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_OWNER,
    ZSPACE_REPETITION_UNLIKELIHOOD_WORK_UNIT_RULE,
};

const PY_REPETITION_UNLIKELIHOOD_MAX_INGRESS_BYTES: u64 = 64 * 1_024 * 1_024;
const PY_REPETITION_UNLIKELIHOOD_MAX_INGRESS_NODES: u64 = 8_000_000;
const PY_REPETITION_UNLIKELIHOOD_MAX_INGRESS_DEPTH: usize = 32;

fn json_error(context: &str, error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!("{context}: {error}"))
}

fn mapping_value(value: &Bound<'_, PyAny>, label: &str) -> PyResult<serde_json::Value> {
    let value = crate::json::py_to_json_bounded(
        value,
        PY_REPETITION_UNLIKELIHOOD_MAX_INGRESS_BYTES,
        PY_REPETITION_UNLIKELIHOOD_MAX_INGRESS_NODES,
        PY_REPETITION_UNLIKELIHOOD_MAX_INGRESS_DEPTH,
        label,
    )?;
    if !value.is_object() {
        return Err(PyValueError::new_err(format!("{label} must be a mapping")));
    }
    Ok(value)
}

fn trusted_legacy_mapping_value(
    value: &Bound<'_, PyAny>,
    label: &str,
) -> PyResult<serde_json::Value> {
    // Keep the historical concrete-container converter for explicit trusted replay.
    let value = crate::json::py_to_json(value)?;
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
fn _zspace_repetition_unlikelihood_plan(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let request = mapping_value(request, "Z-space repetition-unlikelihood request")?;
    let request: ZSpaceRepetitionUnlikelihoodRequest = serde_json::from_value(request)
        .map_err(|error| json_error("invalid Z-space repetition-unlikelihood request", error))?;
    let plan = py
        .allow_threads(|| plan_zspace_repetition_unlikelihood(request))
        .map_err(|error| json_error("Z-space repetition-unlikelihood planning failed", error))?;
    response_to_py(
        py,
        &plan,
        "Z-space repetition-unlikelihood plan encoding failed",
    )
}

#[pyfunction]
fn _zspace_repetition_unlikelihood_validate(
    py: Python<'_>,
    plan: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let plan = mapping_value(plan, "Z-space repetition-unlikelihood plan")?;
    let plan = py
        .allow_threads(|| validate_zspace_repetition_unlikelihood_value(plan))
        .map_err(|error| json_error("Z-space repetition-unlikelihood validation failed", error))?;
    response_to_py(
        py,
        &plan,
        "Z-space repetition-unlikelihood plan encoding failed",
    )
}

#[pyfunction]
fn _zspace_repetition_unlikelihood_validate_trusted_legacy_replay(
    py: Python<'_>,
    plan: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let plan =
        trusted_legacy_mapping_value(plan, "trusted legacy Z-space repetition-unlikelihood plan")?;
    let plan = py
        .allow_threads(|| validate_zspace_repetition_unlikelihood_value_trusted_legacy_replay(plan))
        .map_err(|error| {
            json_error(
                "trusted legacy Z-space repetition-unlikelihood replay failed",
                error,
            )
        })?;
    response_to_py(
        py,
        &plan,
        "Z-space repetition-unlikelihood plan encoding failed",
    )
}

pub(crate) fn register(_py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    for (name, value) in [
        (
            "ZSPACE_REPETITION_UNLIKELIHOOD_CONTRACT_VERSION",
            ZSPACE_REPETITION_UNLIKELIHOOD_CONTRACT_VERSION,
        ),
        (
            "ZSPACE_REPETITION_UNLIKELIHOOD_KIND",
            ZSPACE_REPETITION_UNLIKELIHOOD_KIND,
        ),
        (
            "ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_OWNER",
            ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_OWNER,
        ),
        (
            "ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_BACKEND",
            ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_BACKEND,
        ),
        (
            "ZSPACE_REPETITION_UNLIKELIHOOD_DIFFERENTIATION_OWNER",
            ZSPACE_REPETITION_UNLIKELIHOOD_DIFFERENTIATION_OWNER,
        ),
        (
            "ZSPACE_REPETITION_UNLIKELIHOOD_CANDIDATE_RULE",
            ZSPACE_REPETITION_UNLIKELIHOOD_CANDIDATE_RULE,
        ),
        (
            "ZSPACE_REPETITION_UNLIKELIHOOD_OBJECTIVE_RULE",
            ZSPACE_REPETITION_UNLIKELIHOOD_OBJECTIVE_RULE,
        ),
        (
            "ZSPACE_REPETITION_UNLIKELIHOOD_PROPOSAL_OWNER",
            ZSPACE_REPETITION_UNLIKELIHOOD_PROPOSAL_OWNER,
        ),
        (
            "ZSPACE_REPETITION_UNLIKELIHOOD_PROPOSAL_RULE",
            ZSPACE_REPETITION_UNLIKELIHOOD_PROPOSAL_RULE,
        ),
        (
            "ZSPACE_REPETITION_UNLIKELIHOOD_WORK_UNIT_RULE",
            ZSPACE_REPETITION_UNLIKELIHOOD_WORK_UNIT_RULE,
        ),
        (
            "ZSPACE_REPETITION_UNLIKELIHOOD_MATERIALIZED_PLAN_BYTE_RULE",
            ZSPACE_REPETITION_UNLIKELIHOOD_MATERIALIZED_PLAN_BYTE_RULE,
        ),
    ] {
        parent.add(name, value)?;
    }
    parent.add(
        "ZSPACE_REPETITION_UNLIKELIHOOD_PROBABILITY_EPSILON",
        ZSPACE_REPETITION_UNLIKELIHOOD_PROBABILITY_EPSILON,
    )?;
    parent.add(
        "ZSPACE_REPETITION_UNLIKELIHOOD_MAX_STRENGTH",
        ZSPACE_REPETITION_UNLIKELIHOOD_MAX_STRENGTH,
    )?;
    parent.add(
        "ZSPACE_REPETITION_UNLIKELIHOOD_MIN_NGRAM_ORDER",
        ZSPACE_REPETITION_UNLIKELIHOOD_MIN_NGRAM_ORDER,
    )?;
    parent.add(
        "ZSPACE_REPETITION_UNLIKELIHOOD_MAX_NGRAM_ORDER",
        ZSPACE_REPETITION_UNLIKELIHOOD_MAX_NGRAM_ORDER,
    )?;
    parent.add(
        "ZSPACE_REPETITION_UNLIKELIHOOD_MAX_CONTEXT_WINDOW",
        ZSPACE_REPETITION_UNLIKELIHOOD_MAX_CONTEXT_WINDOW,
    )?;
    parent.add(
        "ZSPACE_REPETITION_UNLIKELIHOOD_MAX_CANDIDATES_PER_POSITION",
        ZSPACE_REPETITION_UNLIKELIHOOD_MAX_CANDIDATES_PER_POSITION,
    )?;
    parent.add(
        "ZSPACE_REPETITION_UNLIKELIHOOD_MAX_PROPOSAL_TOP_K",
        ZSPACE_REPETITION_UNLIKELIHOOD_MAX_PROPOSAL_TOP_K,
    )?;
    parent.add(
        "ZSPACE_REPETITION_UNLIKELIHOOD_PERIODIC_SUFFIX_MAX_PERIOD",
        ZSPACE_REPETITION_UNLIKELIHOOD_PERIODIC_SUFFIX_MAX_PERIOD,
    )?;
    parent.add(
        "ZSPACE_REPETITION_UNLIKELIHOOD_PERIODIC_SUFFIX_MIN_REPETITIONS",
        ZSPACE_REPETITION_UNLIKELIHOOD_PERIODIC_SUFFIX_MIN_REPETITIONS,
    )?;
    parent.add(
        "ZSPACE_REPETITION_UNLIKELIHOOD_MAX_SEQUENCES",
        ZSPACE_REPETITION_UNLIKELIHOOD_MAX_SEQUENCES,
    )?;
    parent.add(
        "ZSPACE_REPETITION_UNLIKELIHOOD_MAX_TOKENS_PER_SEQUENCE",
        ZSPACE_REPETITION_UNLIKELIHOOD_MAX_TOKENS_PER_SEQUENCE,
    )?;
    parent.add(
        "ZSPACE_REPETITION_UNLIKELIHOOD_MAX_TOTAL_TOKENS",
        ZSPACE_REPETITION_UNLIKELIHOOD_MAX_TOTAL_TOKENS,
    )?;
    parent.add(
        "ZSPACE_REPETITION_UNLIKELIHOOD_MAX_SAFE_INTEGER",
        ZSPACE_REPETITION_UNLIKELIHOOD_MAX_SAFE_INTEGER,
    )?;
    parent.add(
        "ZSPACE_REPETITION_UNLIKELIHOOD_MAX_WORK_UNITS",
        ZSPACE_REPETITION_UNLIKELIHOOD_MAX_WORK_UNITS,
    )?;
    parent.add(
        "ZSPACE_REPETITION_UNLIKELIHOOD_MAX_MATERIALIZED_PLAN_BYTES",
        ZSPACE_REPETITION_UNLIKELIHOOD_MAX_MATERIALIZED_PLAN_BYTES,
    )?;
    parent.add_function(wrap_pyfunction!(
        _zspace_repetition_unlikelihood_plan,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(
        _zspace_repetition_unlikelihood_validate,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(
        _zspace_repetition_unlikelihood_validate_trusted_legacy_replay,
        parent
    )?)?;
    Ok(())
}
