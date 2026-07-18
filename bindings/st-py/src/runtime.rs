use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use pyo3::wrap_pyfunction;
use st_core::backend::runtime_route::{
    evaluate_runtime_device_route, RuntimeDeviceRoutePayload, RuntimeDeviceRouteRequest,
};
use st_core::runtime::api_llm_route_policy::{
    evaluate_api_llm_route_policy, ApiLlmRoutePolicyEvaluationRequest,
};
use st_core::runtime::topos_route_policy::{
    build_topos_route_rewards, evaluate_topos_route_policy, resolve_topos_route_policy,
    ToposRoutePolicyEvaluationRequest, ToposRoutePolicyResolveRequest, ToposRouteRewardsRequest,
};

#[pyfunction]
fn _api_llm_route_policy_evaluate(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let request: ApiLlmRoutePolicyEvaluationRequest =
        request_from_py(request, "invalid API LLM route-policy evaluation request")?;
    let payload = evaluate_api_llm_route_policy(request)
        .map_err(|error| json_error("API LLM route-policy evaluation failed", error))?;
    payload_to_py(py, payload, "API LLM route-policy contract encoding failed")
}

fn json_error(context: &str, error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!("{context}: {error}"))
}

fn request_from_py<T>(request: &Bound<'_, PyAny>, context: &str) -> PyResult<T>
where
    T: serde::de::DeserializeOwned,
{
    let request = crate::json::py_to_json(request)?;
    serde_json::from_value(request).map_err(|error| json_error(context, error))
}

fn payload_to_py<T>(py: Python<'_>, payload: T, context: &str) -> PyResult<PyObject>
where
    T: serde::Serialize,
{
    let payload = serde_json::to_value(payload).map_err(|error| json_error(context, error))?;
    crate::json::json_to_py(py, &payload)
}

#[pyfunction]
fn _topos_route_policy_evaluate(py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let request: ToposRoutePolicyEvaluationRequest =
        request_from_py(request, "invalid Topos route-policy evaluation request")?;
    let payload = evaluate_topos_route_policy(request)
        .map_err(|error| json_error("Topos route-policy evaluation failed", error))?;
    payload_to_py(py, payload, "Topos route-policy contract encoding failed")
}

#[pyfunction]
fn _topos_route_policy_rewards(py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let request: ToposRouteRewardsRequest =
        request_from_py(request, "invalid Topos route-reward request")?;
    let payload = build_topos_route_rewards(request)
        .map_err(|error| json_error("Topos route-reward projection failed", error))?;
    payload_to_py(py, payload, "Topos route-reward contract encoding failed")
}

#[pyfunction]
fn _topos_route_policy_resolve(py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let request: ToposRoutePolicyResolveRequest =
        request_from_py(request, "invalid Topos route-policy resolution request")?;
    let payload = resolve_topos_route_policy(request)
        .map_err(|error| json_error("Topos route-policy resolution failed", error))?;
    payload_to_py(
        py,
        payload,
        "Topos route-policy resolution contract encoding failed",
    )
}

#[pyfunction]
fn _runtime_device_route_evaluate(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let request: RuntimeDeviceRouteRequest =
        request_from_py(request, "invalid runtime-device route request")?;
    let payload = evaluate_runtime_device_route(request)
        .and_then(|payload| payload.with_execution_client("python"))
        .map_err(|error| json_error("runtime-device route evaluation failed", error))?;
    payload_to_py(py, payload, "runtime-device route contract encoding failed")
}

#[pyfunction]
fn _runtime_device_route_validate(
    py: Python<'_>,
    payload: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let payload: RuntimeDeviceRoutePayload =
        request_from_py(payload, "invalid runtime-device route payload")?;
    payload
        .validate()
        .map_err(|error| json_error("runtime-device route validation failed", error))?;
    payload_to_py(py, payload, "runtime-device route contract encoding failed")
}

#[pyfunction]
fn _runtime_device_route_validate_against(
    py: Python<'_>,
    payload: &Bound<'_, PyAny>,
    request: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let payload: RuntimeDeviceRoutePayload =
        request_from_py(payload, "invalid runtime-device route payload")?;
    let request: RuntimeDeviceRouteRequest =
        request_from_py(request, "invalid runtime-device route replay request")?;
    payload
        .validate_against(request)
        .map_err(|error| json_error("runtime-device route replay failed", error))?;
    payload_to_py(py, payload, "runtime-device route contract encoding failed")
}

pub(crate) fn register(_py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    parent.add_function(wrap_pyfunction!(_api_llm_route_policy_evaluate, parent)?)?;
    parent.add_function(wrap_pyfunction!(_runtime_device_route_evaluate, parent)?)?;
    parent.add_function(wrap_pyfunction!(_runtime_device_route_validate, parent)?)?;
    parent.add_function(wrap_pyfunction!(
        _runtime_device_route_validate_against,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(_topos_route_policy_evaluate, parent)?)?;
    parent.add_function(wrap_pyfunction!(_topos_route_policy_rewards, parent)?)?;
    parent.add_function(wrap_pyfunction!(_topos_route_policy_resolve, parent)?)?;
    Ok(())
}
