use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use pyo3::wrap_pyfunction;
use st_core::runtime::zspace_semantic_review::{
    summarize_zspace_semantic_review_draft, unblind_zspace_semantic_review,
    validate_zspace_semantic_review_draft_receipt_value, validate_zspace_semantic_review_packet,
    validate_zspace_semantic_review_packet_receipt_value,
    validate_zspace_semantic_review_unblind_value, zspace_semantic_review_map_id,
    ZSpaceSemanticReviewDraftRequest, ZSpaceSemanticReviewMapEntry, ZSpaceSemanticReviewPacket,
    ZSpaceSemanticReviewUnblindRequest, ZSPACE_SEMANTIC_REVIEW_CANDIDATE_LABELS,
    ZSPACE_SEMANTIC_REVIEW_DRAFT_CONTRACT_VERSION, ZSPACE_SEMANTIC_REVIEW_DRAFT_KIND,
    ZSPACE_SEMANTIC_REVIEW_DRAFT_SCHEMA, ZSPACE_SEMANTIC_REVIEW_EVIDENCE_BOUNDARY,
    ZSPACE_SEMANTIC_REVIEW_MAP_COMMITMENT_VERSION, ZSPACE_SEMANTIC_REVIEW_MAP_ID_RULE,
    ZSPACE_SEMANTIC_REVIEW_MAP_SCHEMA, ZSPACE_SEMANTIC_REVIEW_MAX_GROUPS,
    ZSPACE_SEMANTIC_REVIEW_PACKET_CONTRACT_VERSION, ZSPACE_SEMANTIC_REVIEW_PACKET_ID_RULE,
    ZSPACE_SEMANTIC_REVIEW_PACKET_KIND, ZSPACE_SEMANTIC_REVIEW_PACKET_SCHEMA,
    ZSPACE_SEMANTIC_REVIEW_PREFERENCE_VALUES, ZSPACE_SEMANTIC_REVIEW_RESPONSE_CONTRACT_VERSION,
    ZSPACE_SEMANTIC_REVIEW_SCORE_DIMENSIONS, ZSPACE_SEMANTIC_REVIEW_SCORE_MAXIMUM,
    ZSPACE_SEMANTIC_REVIEW_SCORE_MINIMUM, ZSPACE_SEMANTIC_REVIEW_SEMANTIC_BACKEND,
    ZSPACE_SEMANTIC_REVIEW_SEMANTIC_OWNER, ZSPACE_SEMANTIC_REVIEW_UNBLIND_CONTRACT_VERSION,
    ZSPACE_SEMANTIC_REVIEW_UNBLIND_KIND,
};

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ZSpaceSemanticReviewMapIdRequest {
    entries: Vec<ZSpaceSemanticReviewMapEntry>,
}

fn json_error(context: &str, error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!("{context}: {error}"))
}

fn mapping_value(value: &Bound<'_, PyAny>, label: &str) -> PyResult<serde_json::Value> {
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
fn _zspace_semantic_review_map_id(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let request = mapping_value(request, "Z-space semantic review map commitment request")?;
    let request: ZSpaceSemanticReviewMapIdRequest = serde_json::from_value(request)
        .map_err(|error| json_error("invalid Z-space semantic review map commitment", error))?;
    let map_id = py
        .allow_threads(|| zspace_semantic_review_map_id(request.entries))
        .map_err(|error| json_error("Z-space semantic review map commitment failed", error))?;
    response_to_py(
        py,
        &map_id,
        "Z-space semantic review map commitment encoding failed",
    )
}

#[pyfunction]
fn _zspace_semantic_review_packet(py: Python<'_>, packet: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let packet = mapping_value(packet, "Z-space semantic review packet")?;
    let packet: ZSpaceSemanticReviewPacket = serde_json::from_value(packet)
        .map_err(|error| json_error("invalid Z-space semantic review packet", error))?;
    let receipt = py
        .allow_threads(|| validate_zspace_semantic_review_packet(packet))
        .map_err(|error| json_error("Z-space semantic review packet validation failed", error))?;
    response_to_py(
        py,
        &receipt,
        "Z-space semantic review packet encoding failed",
    )
}

#[pyfunction]
fn _zspace_semantic_review_packet_validate(
    py: Python<'_>,
    receipt: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let receipt = mapping_value(receipt, "Z-space semantic review packet receipt")?;
    let receipt =
        validate_zspace_semantic_review_packet_receipt_value(receipt).map_err(|error| {
            json_error(
                "Z-space semantic review packet receipt validation failed",
                error,
            )
        })?;
    response_to_py(
        py,
        &receipt,
        "Z-space semantic review packet encoding failed",
    )
}

#[pyfunction]
fn _zspace_semantic_review_draft(py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let request = mapping_value(request, "Z-space semantic review draft request")?;
    let request: ZSpaceSemanticReviewDraftRequest = serde_json::from_value(request)
        .map_err(|error| json_error("invalid Z-space semantic review draft request", error))?;
    let receipt = py
        .allow_threads(|| summarize_zspace_semantic_review_draft(request.packet, request.draft))
        .map_err(|error| json_error("Z-space semantic review draft validation failed", error))?;
    response_to_py(
        py,
        &receipt,
        "Z-space semantic review draft encoding failed",
    )
}

#[pyfunction]
fn _zspace_semantic_review_draft_validate(
    py: Python<'_>,
    receipt: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let receipt = mapping_value(receipt, "Z-space semantic review draft receipt")?;
    let receipt =
        validate_zspace_semantic_review_draft_receipt_value(receipt).map_err(|error| {
            json_error(
                "Z-space semantic review draft receipt validation failed",
                error,
            )
        })?;
    response_to_py(
        py,
        &receipt,
        "Z-space semantic review draft encoding failed",
    )
}

#[pyfunction]
fn _zspace_semantic_review_unblind(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let request = mapping_value(request, "Z-space semantic review unblind request")?;
    let request: ZSpaceSemanticReviewUnblindRequest = serde_json::from_value(request)
        .map_err(|error| json_error("invalid Z-space semantic review unblind request", error))?;
    let report = py
        .allow_threads(|| {
            unblind_zspace_semantic_review(request.packet, request.draft, request.blinding_map)
        })
        .map_err(|error| json_error("Z-space semantic review unblind failed", error))?;
    response_to_py(
        py,
        &report,
        "Z-space semantic review unblind encoding failed",
    )
}

#[pyfunction]
fn _zspace_semantic_review_unblind_validate(
    py: Python<'_>,
    report: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let report = mapping_value(report, "Z-space semantic review unblind report")?;
    let report = validate_zspace_semantic_review_unblind_value(report).map_err(|error| {
        json_error(
            "Z-space semantic review unblind report validation failed",
            error,
        )
    })?;
    response_to_py(
        py,
        &report,
        "Z-space semantic review unblind encoding failed",
    )
}

pub(crate) fn register(_py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    for (name, value) in [
        (
            "ZSPACE_SEMANTIC_REVIEW_PACKET_SCHEMA",
            ZSPACE_SEMANTIC_REVIEW_PACKET_SCHEMA,
        ),
        (
            "ZSPACE_SEMANTIC_REVIEW_MAP_SCHEMA",
            ZSPACE_SEMANTIC_REVIEW_MAP_SCHEMA,
        ),
        (
            "ZSPACE_SEMANTIC_REVIEW_DRAFT_SCHEMA",
            ZSPACE_SEMANTIC_REVIEW_DRAFT_SCHEMA,
        ),
        (
            "ZSPACE_SEMANTIC_REVIEW_PACKET_CONTRACT_VERSION",
            ZSPACE_SEMANTIC_REVIEW_PACKET_CONTRACT_VERSION,
        ),
        (
            "ZSPACE_SEMANTIC_REVIEW_DRAFT_CONTRACT_VERSION",
            ZSPACE_SEMANTIC_REVIEW_DRAFT_CONTRACT_VERSION,
        ),
        (
            "ZSPACE_SEMANTIC_REVIEW_RESPONSE_CONTRACT_VERSION",
            ZSPACE_SEMANTIC_REVIEW_RESPONSE_CONTRACT_VERSION,
        ),
        (
            "ZSPACE_SEMANTIC_REVIEW_MAP_COMMITMENT_VERSION",
            ZSPACE_SEMANTIC_REVIEW_MAP_COMMITMENT_VERSION,
        ),
        (
            "ZSPACE_SEMANTIC_REVIEW_UNBLIND_CONTRACT_VERSION",
            ZSPACE_SEMANTIC_REVIEW_UNBLIND_CONTRACT_VERSION,
        ),
        (
            "ZSPACE_SEMANTIC_REVIEW_PACKET_KIND",
            ZSPACE_SEMANTIC_REVIEW_PACKET_KIND,
        ),
        (
            "ZSPACE_SEMANTIC_REVIEW_DRAFT_KIND",
            ZSPACE_SEMANTIC_REVIEW_DRAFT_KIND,
        ),
        (
            "ZSPACE_SEMANTIC_REVIEW_UNBLIND_KIND",
            ZSPACE_SEMANTIC_REVIEW_UNBLIND_KIND,
        ),
        (
            "ZSPACE_SEMANTIC_REVIEW_SEMANTIC_OWNER",
            ZSPACE_SEMANTIC_REVIEW_SEMANTIC_OWNER,
        ),
        (
            "ZSPACE_SEMANTIC_REVIEW_SEMANTIC_BACKEND",
            ZSPACE_SEMANTIC_REVIEW_SEMANTIC_BACKEND,
        ),
        (
            "ZSPACE_SEMANTIC_REVIEW_PACKET_ID_RULE",
            ZSPACE_SEMANTIC_REVIEW_PACKET_ID_RULE,
        ),
        (
            "ZSPACE_SEMANTIC_REVIEW_MAP_ID_RULE",
            ZSPACE_SEMANTIC_REVIEW_MAP_ID_RULE,
        ),
        (
            "ZSPACE_SEMANTIC_REVIEW_EVIDENCE_BOUNDARY",
            ZSPACE_SEMANTIC_REVIEW_EVIDENCE_BOUNDARY,
        ),
    ] {
        parent.add(name, value)?;
    }
    parent.add(
        "ZSPACE_SEMANTIC_REVIEW_CANDIDATE_LABELS",
        ZSPACE_SEMANTIC_REVIEW_CANDIDATE_LABELS,
    )?;
    parent.add(
        "ZSPACE_SEMANTIC_REVIEW_SCORE_DIMENSIONS",
        ZSPACE_SEMANTIC_REVIEW_SCORE_DIMENSIONS,
    )?;
    parent.add(
        "ZSPACE_SEMANTIC_REVIEW_PREFERENCE_VALUES",
        ZSPACE_SEMANTIC_REVIEW_PREFERENCE_VALUES,
    )?;
    parent.add(
        "ZSPACE_SEMANTIC_REVIEW_SCORE_MINIMUM",
        ZSPACE_SEMANTIC_REVIEW_SCORE_MINIMUM,
    )?;
    parent.add(
        "ZSPACE_SEMANTIC_REVIEW_SCORE_MAXIMUM",
        ZSPACE_SEMANTIC_REVIEW_SCORE_MAXIMUM,
    )?;
    parent.add(
        "ZSPACE_SEMANTIC_REVIEW_MAX_GROUPS",
        ZSPACE_SEMANTIC_REVIEW_MAX_GROUPS,
    )?;
    parent.add_function(wrap_pyfunction!(_zspace_semantic_review_map_id, parent)?)?;
    parent.add_function(wrap_pyfunction!(_zspace_semantic_review_packet, parent)?)?;
    parent.add_function(wrap_pyfunction!(
        _zspace_semantic_review_packet_validate,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(_zspace_semantic_review_draft, parent)?)?;
    parent.add_function(wrap_pyfunction!(
        _zspace_semantic_review_draft_validate,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(_zspace_semantic_review_unblind, parent)?)?;
    parent.add_function(wrap_pyfunction!(
        _zspace_semantic_review_unblind_validate,
        parent
    )?)?;
    Ok(())
}
