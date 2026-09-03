use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use pyo3::wrap_pyfunction;
use st_core::runtime::zspace_semantic_review::{
    seal_zspace_semantic_review_packet, summarize_zspace_semantic_review_draft,
    unblind_zspace_semantic_review, validate_zspace_semantic_review_draft_receipt_value,
    validate_zspace_semantic_review_draft_receipt_value_trusted_legacy_replay,
    validate_zspace_semantic_review_packet, validate_zspace_semantic_review_packet_receipt_value,
    validate_zspace_semantic_review_packet_receipt_value_trusted_legacy_replay,
    validate_zspace_semantic_review_packet_trusted_legacy_replay,
    validate_zspace_semantic_review_unblind_value,
    validate_zspace_semantic_review_unblind_value_trusted_legacy_replay,
    zspace_semantic_review_map_id, zspace_semantic_review_map_id_trusted_legacy_replay,
    ZSpaceSemanticReviewDraftRequest, ZSpaceSemanticReviewMapIdRequest, ZSpaceSemanticReviewPacket,
    ZSpaceSemanticReviewPacketRequest, ZSpaceSemanticReviewUnblindRequest,
    ZSPACE_SEMANTIC_REVIEW_CANDIDATE_LABELS, ZSPACE_SEMANTIC_REVIEW_DRAFT_CONTRACT_VERSION,
    ZSPACE_SEMANTIC_REVIEW_DRAFT_KIND, ZSPACE_SEMANTIC_REVIEW_DRAFT_SCHEMA,
    ZSPACE_SEMANTIC_REVIEW_EVIDENCE_BOUNDARY, ZSPACE_SEMANTIC_REVIEW_MAP_COMMITMENT_VERSION,
    ZSPACE_SEMANTIC_REVIEW_MAP_ID_RULE, ZSPACE_SEMANTIC_REVIEW_MAP_SCHEMA,
    ZSPACE_SEMANTIC_REVIEW_MAP_STATUS, ZSPACE_SEMANTIC_REVIEW_MAX_ARM_NAME_BYTES,
    ZSPACE_SEMANTIC_REVIEW_MAX_CONTINUATION_BYTES, ZSPACE_SEMANTIC_REVIEW_MAX_GROUPS,
    ZSPACE_SEMANTIC_REVIEW_MAX_INGRESS_BYTES, ZSPACE_SEMANTIC_REVIEW_MAX_INGRESS_DEPTH,
    ZSPACE_SEMANTIC_REVIEW_MAX_INGRESS_NODES, ZSPACE_SEMANTIC_REVIEW_MAX_INSTRUCTIONS_BYTES,
    ZSPACE_SEMANTIC_REVIEW_MAX_MAP_ENTRIES, ZSPACE_SEMANTIC_REVIEW_MAX_PACKET_TEXT_BYTES,
    ZSPACE_SEMANTIC_REVIEW_MAX_PROMPT_BYTES, ZSPACE_SEMANTIC_REVIEW_MAX_SAFE_INTEGER,
    ZSPACE_SEMANTIC_REVIEW_PACKET_CONTRACT_VERSION, ZSPACE_SEMANTIC_REVIEW_PACKET_ID_RULE,
    ZSPACE_SEMANTIC_REVIEW_PACKET_KIND, ZSPACE_SEMANTIC_REVIEW_PACKET_SCHEMA,
    ZSPACE_SEMANTIC_REVIEW_PACKET_STATUS, ZSPACE_SEMANTIC_REVIEW_PACKET_TEXT_BYTE_RULE,
    ZSPACE_SEMANTIC_REVIEW_PREFERENCE_VALUES, ZSPACE_SEMANTIC_REVIEW_RESPONSE_CONTRACT_VERSION,
    ZSPACE_SEMANTIC_REVIEW_SCORE_DIMENSIONS, ZSPACE_SEMANTIC_REVIEW_SCORE_MAXIMUM,
    ZSPACE_SEMANTIC_REVIEW_SCORE_MINIMUM, ZSPACE_SEMANTIC_REVIEW_SEMANTIC_BACKEND,
    ZSPACE_SEMANTIC_REVIEW_SEMANTIC_OWNER, ZSPACE_SEMANTIC_REVIEW_UNBLIND_CONTRACT_VERSION,
    ZSPACE_SEMANTIC_REVIEW_UNBLIND_KIND,
};

fn json_error(context: &str, error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!("{context}: {error}"))
}

fn mapping_value(value: &Bound<'_, PyAny>, label: &str) -> PyResult<serde_json::Value> {
    let value = crate::json::py_to_json_bounded(
        value,
        ZSPACE_SEMANTIC_REVIEW_MAX_INGRESS_BYTES,
        ZSPACE_SEMANTIC_REVIEW_MAX_INGRESS_NODES,
        ZSPACE_SEMANTIC_REVIEW_MAX_INGRESS_DEPTH as usize,
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
    let value = crate::json::py_to_json(value)?;
    if !value.is_object() {
        return Err(PyValueError::new_err(format!("{label} must be a mapping")));
    }
    Ok(value)
}

#[pyfunction]
fn _zspace_semantic_review_packet_seal(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let request = mapping_value(request, "Z-space semantic review packet request")?;
    let request: ZSpaceSemanticReviewPacketRequest = serde_json::from_value(request)
        .map_err(|error| json_error("invalid Z-space semantic review packet request", error))?;
    let receipt = py
        .detach(|| seal_zspace_semantic_review_packet(request))
        .map_err(|error| json_error("Z-space semantic review packet sealing failed", error))?;
    response_to_py(
        py,
        &receipt,
        "Z-space semantic review packet encoding failed",
    )
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
fn _zspace_semantic_review_map_id(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let request = mapping_value(request, "Z-space semantic review map commitment request")?;
    let request: ZSpaceSemanticReviewMapIdRequest = serde_json::from_value(request)
        .map_err(|error| json_error("invalid Z-space semantic review map commitment", error))?;
    let map_id = py
        .detach(|| zspace_semantic_review_map_id(request.entries))
        .map_err(|error| json_error("Z-space semantic review map commitment failed", error))?;
    response_to_py(
        py,
        &map_id,
        "Z-space semantic review map commitment encoding failed",
    )
}

#[pyfunction]
fn _zspace_semantic_review_map_id_trusted_legacy_replay(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let request = trusted_legacy_mapping_value(
        request,
        "trusted legacy Z-space semantic review map commitment request",
    )?;
    let request: ZSpaceSemanticReviewMapIdRequest =
        serde_json::from_value(request).map_err(|error| {
            json_error(
                "invalid trusted legacy semantic review map commitment",
                error,
            )
        })?;
    let map_id = py
        .detach(|| zspace_semantic_review_map_id_trusted_legacy_replay(request.entries))
        .map_err(|error| {
            json_error(
                "trusted legacy Z-space semantic review map commitment replay failed",
                error,
            )
        })?;
    response_to_py(
        py,
        &map_id,
        "Z-space semantic review map commitment encoding failed",
    )
}

#[pyfunction]
fn _zspace_semantic_review_packet(
    py: Python<'_>,
    packet: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let packet = mapping_value(packet, "Z-space semantic review packet")?;
    let packet: ZSpaceSemanticReviewPacket = serde_json::from_value(packet)
        .map_err(|error| json_error("invalid Z-space semantic review packet", error))?;
    let receipt = py
        .detach(|| validate_zspace_semantic_review_packet(packet))
        .map_err(|error| json_error("Z-space semantic review packet validation failed", error))?;
    response_to_py(
        py,
        &receipt,
        "Z-space semantic review packet encoding failed",
    )
}

#[pyfunction]
fn _zspace_semantic_review_packet_trusted_legacy_replay(
    py: Python<'_>,
    packet: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let packet =
        trusted_legacy_mapping_value(packet, "trusted legacy Z-space semantic review packet")?;
    let packet: ZSpaceSemanticReviewPacket = serde_json::from_value(packet).map_err(|error| {
        json_error(
            "invalid trusted legacy Z-space semantic review packet",
            error,
        )
    })?;
    let receipt = py
        .detach(|| validate_zspace_semantic_review_packet_trusted_legacy_replay(packet))
        .map_err(|error| {
            json_error(
                "trusted legacy Z-space semantic review packet replay failed",
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
fn _zspace_semantic_review_packet_validate(
    py: Python<'_>,
    receipt: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
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
fn _zspace_semantic_review_packet_validate_trusted_legacy_replay(
    py: Python<'_>,
    receipt: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let receipt = trusted_legacy_mapping_value(
        receipt,
        "trusted legacy Z-space semantic review packet receipt",
    )?;
    let receipt =
        validate_zspace_semantic_review_packet_receipt_value_trusted_legacy_replay(receipt)
            .map_err(|error| {
                json_error(
                    "trusted legacy Z-space semantic review packet receipt replay failed",
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
fn _zspace_semantic_review_draft(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let request = mapping_value(request, "Z-space semantic review draft request")?;
    let request: ZSpaceSemanticReviewDraftRequest = serde_json::from_value(request)
        .map_err(|error| json_error("invalid Z-space semantic review draft request", error))?;
    let receipt = py
        .detach(|| summarize_zspace_semantic_review_draft(request.packet, request.draft))
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
) -> PyResult<Py<PyAny>> {
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
fn _zspace_semantic_review_draft_validate_trusted_legacy_replay(
    py: Python<'_>,
    receipt: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let receipt = trusted_legacy_mapping_value(
        receipt,
        "trusted legacy Z-space semantic review draft receipt",
    )?;
    let receipt =
        validate_zspace_semantic_review_draft_receipt_value_trusted_legacy_replay(receipt)
            .map_err(|error| {
                json_error(
                    "trusted legacy Z-space semantic review draft receipt replay failed",
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
) -> PyResult<Py<PyAny>> {
    let request = mapping_value(request, "Z-space semantic review unblind request")?;
    let request: ZSpaceSemanticReviewUnblindRequest = serde_json::from_value(request)
        .map_err(|error| json_error("invalid Z-space semantic review unblind request", error))?;
    let report = py
        .detach(|| {
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
) -> PyResult<Py<PyAny>> {
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

#[pyfunction]
fn _zspace_semantic_review_unblind_validate_trusted_legacy_replay(
    py: Python<'_>,
    report: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let report = trusted_legacy_mapping_value(
        report,
        "trusted legacy Z-space semantic review unblind report",
    )?;
    let report = validate_zspace_semantic_review_unblind_value_trusted_legacy_replay(report)
        .map_err(|error| {
            json_error(
                "trusted legacy Z-space semantic review unblind report replay failed",
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
            "ZSPACE_SEMANTIC_REVIEW_PACKET_STATUS",
            ZSPACE_SEMANTIC_REVIEW_PACKET_STATUS,
        ),
        (
            "ZSPACE_SEMANTIC_REVIEW_MAP_SCHEMA",
            ZSPACE_SEMANTIC_REVIEW_MAP_SCHEMA,
        ),
        (
            "ZSPACE_SEMANTIC_REVIEW_MAP_STATUS",
            ZSPACE_SEMANTIC_REVIEW_MAP_STATUS,
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
            "ZSPACE_SEMANTIC_REVIEW_PACKET_TEXT_BYTE_RULE",
            ZSPACE_SEMANTIC_REVIEW_PACKET_TEXT_BYTE_RULE,
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
    parent.add(
        "ZSPACE_SEMANTIC_REVIEW_MAX_PROMPT_BYTES",
        ZSPACE_SEMANTIC_REVIEW_MAX_PROMPT_BYTES,
    )?;
    parent.add(
        "ZSPACE_SEMANTIC_REVIEW_MAX_CONTINUATION_BYTES",
        ZSPACE_SEMANTIC_REVIEW_MAX_CONTINUATION_BYTES,
    )?;
    parent.add(
        "ZSPACE_SEMANTIC_REVIEW_MAX_INSTRUCTIONS_BYTES",
        ZSPACE_SEMANTIC_REVIEW_MAX_INSTRUCTIONS_BYTES,
    )?;
    parent.add(
        "ZSPACE_SEMANTIC_REVIEW_MAX_PACKET_TEXT_BYTES",
        ZSPACE_SEMANTIC_REVIEW_MAX_PACKET_TEXT_BYTES,
    )?;
    parent.add(
        "ZSPACE_SEMANTIC_REVIEW_MAX_ARM_NAME_BYTES",
        ZSPACE_SEMANTIC_REVIEW_MAX_ARM_NAME_BYTES,
    )?;
    parent.add(
        "ZSPACE_SEMANTIC_REVIEW_MAX_MAP_ENTRIES",
        ZSPACE_SEMANTIC_REVIEW_MAX_MAP_ENTRIES,
    )?;
    parent.add(
        "ZSPACE_SEMANTIC_REVIEW_MAX_SAFE_INTEGER",
        ZSPACE_SEMANTIC_REVIEW_MAX_SAFE_INTEGER,
    )?;
    parent.add_function(wrap_pyfunction!(_zspace_semantic_review_map_id, parent)?)?;
    parent.add_function(wrap_pyfunction!(
        _zspace_semantic_review_map_id_trusted_legacy_replay,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(
        _zspace_semantic_review_packet_seal,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(_zspace_semantic_review_packet, parent)?)?;
    parent.add_function(wrap_pyfunction!(
        _zspace_semantic_review_packet_trusted_legacy_replay,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(
        _zspace_semantic_review_packet_validate,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(
        _zspace_semantic_review_packet_validate_trusted_legacy_replay,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(_zspace_semantic_review_draft, parent)?)?;
    parent.add_function(wrap_pyfunction!(
        _zspace_semantic_review_draft_validate,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(
        _zspace_semantic_review_draft_validate_trusted_legacy_replay,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(_zspace_semantic_review_unblind, parent)?)?;
    parent.add_function(wrap_pyfunction!(
        _zspace_semantic_review_unblind_validate,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(
        _zspace_semantic_review_unblind_validate_trusted_legacy_replay,
        parent
    )?)?;
    Ok(())
}
