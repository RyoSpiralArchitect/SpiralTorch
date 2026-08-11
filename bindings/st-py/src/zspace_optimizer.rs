use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use pyo3::wrap_pyfunction;
use st_core::runtime::zspace_evidence::{
    summarize_zspace_polarity_evidence, validate_zspace_polarity_evidence_value,
    ZSpacePolarityEvidenceRequest,
};
use st_core::runtime::zspace_optimizer::{
    initialize_zspace_meta_optimizer, plan_zspace_parameter_trajectory,
    plan_zspace_parameter_trajectory_policy_from_value, restore_zspace_meta_optimizer,
    transition_zspace_meta_optimizer, validate_zspace_parameter_trajectory_policy_value,
    validate_zspace_parameter_trajectory_value, zspace_parameter_control_from_value,
    ZSpaceMetaOptimizerConfig, ZSpaceMetaOptimizerRestoreRequest, ZSpaceMetaOptimizerStepRequest,
    ZSpaceParameterTrajectoryPolicy, ZSpaceParameterTrajectoryRequest,
};
use st_core::runtime::zspace_optimizer_feedback::{
    control_zspace_optimizer_feedback, initialize_zspace_optimizer_feedback,
    observe_zspace_optimizer_feedback, restore_zspace_optimizer_feedback,
    ZSpaceOptimizerFeedbackConfig, ZSpaceOptimizerFeedbackControlRequest,
    ZSpaceOptimizerFeedbackObserveRequest, ZSpaceOptimizerFeedbackRestoreRequest,
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

#[pyfunction]
fn _zspace_parameter_control(py: Python<'_>, report: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let report = request_value(report, "Z-space meta-optimizer report")?;
    let control = zspace_parameter_control_from_value(report)
        .map_err(|error| json_error("Z-space parameter-control validation failed", error))?;
    response_to_py(py, &control, "Z-space parameter-control encoding failed")
}

#[pyfunction]
fn _zspace_parameter_trajectory(py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let request = request_value(request, "Z-space parameter-trajectory request")?;
    let request: ZSpaceParameterTrajectoryRequest = serde_json::from_value(request)
        .map_err(|error| json_error("invalid Z-space parameter-trajectory request", error))?;
    let report = py
        .allow_threads(|| plan_zspace_parameter_trajectory(request))
        .map_err(|error| json_error("Z-space parameter-trajectory planning failed", error))?;
    response_to_py(py, &report, "Z-space parameter-trajectory encoding failed")
}

#[pyfunction]
fn _zspace_parameter_trajectory_validate(
    py: Python<'_>,
    report: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let report = request_value(report, "Z-space parameter-trajectory report")?;
    let report = validate_zspace_parameter_trajectory_value(report)
        .map_err(|error| json_error("Z-space parameter-trajectory validation failed", error))?;
    response_to_py(py, &report, "Z-space parameter-trajectory encoding failed")
}

#[pyfunction]
fn _zspace_parameter_trajectory_policy(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let request = request_value(request, "Z-space parameter-trajectory policy request")?;
    let source = request
        .get("source_trajectory")
        .cloned()
        .ok_or_else(|| PyValueError::new_err("source_trajectory is required"))?;
    let policy = request
        .get("policy")
        .cloned()
        .ok_or_else(|| PyValueError::new_err("policy is required"))?;
    let policy: ZSpaceParameterTrajectoryPolicy = serde_json::from_value(policy)
        .map_err(|error| json_error("invalid Z-space parameter-trajectory policy", error))?;
    let report = py
        .allow_threads(|| plan_zspace_parameter_trajectory_policy_from_value(source, policy))
        .map_err(|error| {
            json_error("Z-space parameter-trajectory policy planning failed", error)
        })?;
    response_to_py(
        py,
        &report,
        "Z-space parameter-trajectory policy encoding failed",
    )
}

#[pyfunction]
fn _zspace_parameter_trajectory_policy_validate(
    py: Python<'_>,
    report: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let report = request_value(report, "Z-space parameter-trajectory policy report")?;
    let report = validate_zspace_parameter_trajectory_policy_value(report).map_err(|error| {
        json_error(
            "Z-space parameter-trajectory policy validation failed",
            error,
        )
    })?;
    response_to_py(
        py,
        &report,
        "Z-space parameter-trajectory policy encoding failed",
    )
}

#[pyfunction]
fn _zspace_polarity_evidence(py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let request = request_value(request, "Z-space polarity evidence request")?;
    let request: ZSpacePolarityEvidenceRequest = serde_json::from_value(request)
        .map_err(|error| json_error("invalid Z-space polarity evidence request", error))?;
    let report = py
        .allow_threads(|| summarize_zspace_polarity_evidence(request))
        .map_err(|error| json_error("Z-space polarity evidence aggregation failed", error))?;
    response_to_py(py, &report, "Z-space polarity evidence encoding failed")
}

#[pyfunction]
fn _zspace_polarity_evidence_validate(
    py: Python<'_>,
    report: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let report = request_value(report, "Z-space polarity evidence report")?;
    let report = validate_zspace_polarity_evidence_value(report)
        .map_err(|error| json_error("Z-space polarity evidence validation failed", error))?;
    response_to_py(py, &report, "Z-space polarity evidence encoding failed")
}

#[pyfunction]
fn _zspace_optimizer_feedback_init(
    py: Python<'_>,
    config: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let config = request_value(config, "Z-space optimizer feedback config")?;
    let config: ZSpaceOptimizerFeedbackConfig = serde_json::from_value(config)
        .map_err(|error| json_error("invalid Z-space optimizer feedback config", error))?;
    let checkpoint = initialize_zspace_optimizer_feedback(config)
        .map_err(|error| json_error("Z-space optimizer feedback initialization failed", error))?;
    response_to_py(
        py,
        &checkpoint,
        "Z-space optimizer feedback checkpoint encoding failed",
    )
}

#[pyfunction]
fn _zspace_optimizer_feedback_restore(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let request = request_value(request, "Z-space optimizer feedback restore request")?;
    let request: ZSpaceOptimizerFeedbackRestoreRequest = serde_json::from_value(request)
        .map_err(|error| json_error("invalid Z-space optimizer feedback restore request", error))?;
    let checkpoint = restore_zspace_optimizer_feedback(request)
        .map_err(|error| json_error("Z-space optimizer feedback restore failed", error))?;
    response_to_py(
        py,
        &checkpoint,
        "Z-space optimizer feedback checkpoint encoding failed",
    )
}

#[pyfunction]
fn _zspace_optimizer_feedback_observe(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let request = request_value(request, "Z-space optimizer feedback observation request")?;
    let request: ZSpaceOptimizerFeedbackObserveRequest =
        serde_json::from_value(request).map_err(|error| {
            json_error(
                "invalid Z-space optimizer feedback observation request",
                error,
            )
        })?;
    let report = observe_zspace_optimizer_feedback(request)
        .map_err(|error| json_error("Z-space optimizer feedback observation failed", error))?;
    response_to_py(
        py,
        &report,
        "Z-space optimizer feedback observation encoding failed",
    )
}

#[pyfunction]
fn _zspace_optimizer_feedback_control(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let request = request_value(request, "Z-space optimizer feedback control request")?;
    let request: ZSpaceOptimizerFeedbackControlRequest = serde_json::from_value(request)
        .map_err(|error| json_error("invalid Z-space optimizer feedback control request", error))?;
    let report = control_zspace_optimizer_feedback(request)
        .map_err(|error| json_error("Z-space optimizer feedback control failed", error))?;
    response_to_py(
        py,
        &report,
        "Z-space optimizer feedback control encoding failed",
    )
}

pub(crate) fn register(_py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_function(wrap_pyfunction!(_zspace_meta_optimizer_init, parent)?)?;
    parent.add_function(wrap_pyfunction!(_zspace_meta_optimizer_restore, parent)?)?;
    parent.add_function(wrap_pyfunction!(_zspace_meta_optimizer_step, parent)?)?;
    parent.add_function(wrap_pyfunction!(_zspace_parameter_control, parent)?)?;
    parent.add_function(wrap_pyfunction!(_zspace_parameter_trajectory, parent)?)?;
    parent.add_function(wrap_pyfunction!(
        _zspace_parameter_trajectory_validate,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(
        _zspace_parameter_trajectory_policy,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(
        _zspace_parameter_trajectory_policy_validate,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(_zspace_polarity_evidence, parent)?)?;
    parent.add_function(wrap_pyfunction!(
        _zspace_polarity_evidence_validate,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(_zspace_optimizer_feedback_init, parent)?)?;
    parent.add_function(wrap_pyfunction!(
        _zspace_optimizer_feedback_restore,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(
        _zspace_optimizer_feedback_observe,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(
        _zspace_optimizer_feedback_control,
        parent
    )?)?;
    Ok(())
}
