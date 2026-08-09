use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use pyo3::wrap_pyfunction;
use st_core::backend::execution_plan::{
    evaluate_runtime_execution_plan, observe_runtime_execution_plan_capabilities,
    validate_tensor_execution_receipt_against_runtime_plan, AcceleratorFallback, BackendPolicy,
    ExecutionConfig, RuntimeExecutionPlanPayload, RuntimeExecutionPlanRequest,
};
use st_core::backend::runtime_probe::{
    observe_runtime_device_probe, RuntimeDeviceProbeObservationRequest, RuntimeDeviceProbePayload,
    RuntimeDeviceProbeRequest,
};
use st_core::backend::runtime_route::{
    evaluate_runtime_device_route, evaluate_runtime_device_route_from_probes,
    RuntimeDeviceRoutePayload, RuntimeDeviceRouteProbeRequest, RuntimeDeviceRouteRequest,
};
use st_core::runtime::api_llm_route_policy::{
    evaluate_api_llm_route_policy, ApiLlmRoutePolicyEvaluationRequest,
};
use st_core::runtime::topos_route_policy::{
    build_topos_route_rewards, evaluate_topos_route_policy, resolve_topos_route_policy,
    ToposRoutePolicyEvaluationRequest, ToposRoutePolicyResolveRequest, ToposRouteRewardsRequest,
};
use st_tensor::TensorExecutionReceipt;

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

fn runtime_device_probe_payload_from_py(
    payload: &Bound<'_, PyAny>,
) -> PyResult<RuntimeDeviceProbePayload> {
    let value = crate::json::py_to_json(payload)?;
    let canonical = value
        .as_object()
        .and_then(|object| object.get("contract"))
        .cloned()
        .unwrap_or(value);
    serde_json::from_value(canonical)
        .map_err(|error| json_error("invalid runtime-device probe payload", error))
}

#[pyfunction]
fn _runtime_device_probe_observe(py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let request: RuntimeDeviceProbeObservationRequest =
        request_from_py(request, "invalid runtime-device observation request")?;
    let payload = observe_runtime_device_probe(request)
        .and_then(|payload| payload.with_execution_client("python"))
        .map_err(|error| json_error("runtime-device observation failed", error))?;
    crate::json::json_to_py(py, &payload.to_transport_value())
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
fn _runtime_device_route_evaluate_probes(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let request: RuntimeDeviceRouteProbeRequest =
        request_from_py(request, "invalid runtime-device probe-route request")?;
    let payload = evaluate_runtime_device_route_from_probes(request)
        .and_then(|payload| payload.with_execution_client("python"))
        .map_err(|error| json_error("runtime-device probe-route evaluation failed", error))?;
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

#[pyfunction]
fn _runtime_device_probe_validate(
    py: Python<'_>,
    payload: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let payload = runtime_device_probe_payload_from_py(payload)?;
    payload
        .validate()
        .map_err(|error| json_error("runtime-device probe validation failed", error))?;
    payload_to_py(py, payload, "runtime-device probe contract encoding failed")
}

#[pyfunction]
fn _runtime_device_probe_validate_against(
    py: Python<'_>,
    payload: &Bound<'_, PyAny>,
    request: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let payload = runtime_device_probe_payload_from_py(payload)?;
    let request: RuntimeDeviceProbeRequest =
        request_from_py(request, "invalid runtime-device probe replay request")?;
    payload
        .validate_against(request)
        .map_err(|error| json_error("runtime-device probe replay failed", error))?;
    payload_to_py(py, payload, "runtime-device probe contract encoding failed")
}

#[pyfunction]
fn _runtime_device_probe_transport(
    py: Python<'_>,
    payload: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let payload = runtime_device_probe_payload_from_py(payload)?;
    payload
        .validate()
        .map_err(|error| json_error("runtime-device probe validation failed", error))?;
    crate::json::json_to_py(py, &payload.to_transport_value())
}

#[pyfunction]
fn _runtime_execution_plan_observe_capabilities(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let request: RuntimeExecutionPlanRequest = request_from_py(
        request,
        "invalid runtime execution-plan observation request",
    )?;
    let request = observe_runtime_execution_plan_capabilities(request)
        .map_err(|error| json_error("runtime component capability observation failed", error))?;
    payload_to_py(
        py,
        request,
        "runtime component capability contract encoding failed",
    )
}

#[pyfunction]
#[pyo3(signature = (*, accelerator_fallback=None, tensor_util_wgpu_min_values=None))]
fn _runtime_execution_config_resolve(
    py: Python<'_>,
    accelerator_fallback: Option<String>,
    tensor_util_wgpu_min_values: Option<usize>,
) -> PyResult<PyObject> {
    let mut config = ExecutionConfig::from_env();
    if let Some(value) = accelerator_fallback {
        config.accelerator_fallback =
            serde_json::from_value::<AcceleratorFallback>(serde_json::Value::String(value))
                .map_err(|error| json_error("invalid accelerator fallback override", error))?;
    }
    if let Some(value) = tensor_util_wgpu_min_values {
        config.tensor_util_wgpu_min_values = value;
    }
    payload_to_py(
        py,
        config,
        "runtime execution configuration encoding failed",
    )
}

#[pyfunction]
fn _runtime_execution_plan_evaluate(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let request: RuntimeExecutionPlanRequest =
        request_from_py(request, "invalid runtime execution-plan request")?;
    let payload = evaluate_runtime_execution_plan(request)
        .and_then(|payload| payload.with_execution_client("python"))
        .map_err(|error| json_error("runtime execution-plan evaluation failed", error))?;
    payload_to_py(
        py,
        payload,
        "runtime execution-plan contract encoding failed",
    )
}

#[pyfunction]
fn _runtime_execution_plan_validate(
    py: Python<'_>,
    payload: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let payload: RuntimeExecutionPlanPayload =
        request_from_py(payload, "invalid runtime execution-plan payload")?;
    payload
        .validate()
        .map_err(|error| json_error("runtime execution-plan validation failed", error))?;
    payload_to_py(
        py,
        payload,
        "runtime execution-plan contract encoding failed",
    )
}

#[pyfunction]
fn _runtime_execution_plan_require_executable(
    py: Python<'_>,
    payload: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let payload: RuntimeExecutionPlanPayload =
        request_from_py(payload, "invalid runtime execution-plan payload")?;
    BackendPolicy::try_from_runtime_plan(&payload)
        .map_err(|error| json_error("runtime execution-plan materialization failed", error))?;
    payload_to_py(
        py,
        payload,
        "runtime execution-plan contract encoding failed",
    )
}

#[pyfunction]
fn _runtime_execution_plan_validate_against(
    py: Python<'_>,
    payload: &Bound<'_, PyAny>,
    request: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let payload: RuntimeExecutionPlanPayload =
        request_from_py(payload, "invalid runtime execution-plan payload")?;
    let request: RuntimeExecutionPlanRequest =
        request_from_py(request, "invalid runtime execution-plan replay request")?;
    payload
        .validate_against(request)
        .map_err(|error| json_error("runtime execution-plan replay failed", error))?;
    payload_to_py(
        py,
        payload,
        "runtime execution-plan contract encoding failed",
    )
}

#[pyfunction]
fn _tensor_execution_receipt_validate(
    py: Python<'_>,
    receipt: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let receipt: TensorExecutionReceipt =
        request_from_py(receipt, "invalid tensor execution receipt")?;
    receipt
        .validate()
        .map_err(|error| json_error("tensor execution receipt validation failed", error))?;
    payload_to_py(
        py,
        receipt,
        "tensor execution receipt contract encoding failed",
    )
}

#[pyfunction]
fn _tensor_execution_receipt_validate_against_runtime_plan(
    py: Python<'_>,
    receipt: &Bound<'_, PyAny>,
    runtime_execution_plan: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let receipt: TensorExecutionReceipt =
        request_from_py(receipt, "invalid tensor execution receipt")?;
    let runtime_execution_plan: RuntimeExecutionPlanPayload = request_from_py(
        runtime_execution_plan,
        "invalid runtime execution-plan payload",
    )?;
    validate_tensor_execution_receipt_against_runtime_plan(&runtime_execution_plan, &receipt)
        .map_err(|error| json_error("tensor execution receipt authorization failed", error))?;
    payload_to_py(
        py,
        receipt,
        "tensor execution receipt contract encoding failed",
    )
}

pub(crate) fn register(_py: Python<'_>, parent: &Bound<PyModule>) -> PyResult<()> {
    parent.add_function(wrap_pyfunction!(_api_llm_route_policy_evaluate, parent)?)?;
    parent.add_function(wrap_pyfunction!(_runtime_device_route_evaluate, parent)?)?;
    parent.add_function(wrap_pyfunction!(
        _runtime_device_route_evaluate_probes,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(_runtime_device_probe_observe, parent)?)?;
    parent.add_function(wrap_pyfunction!(_runtime_device_probe_validate, parent)?)?;
    parent.add_function(wrap_pyfunction!(_runtime_device_probe_transport, parent)?)?;
    parent.add_function(wrap_pyfunction!(
        _runtime_device_probe_validate_against,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(_runtime_device_route_validate, parent)?)?;
    parent.add_function(wrap_pyfunction!(
        _runtime_device_route_validate_against,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(_runtime_execution_config_resolve, parent)?)?;
    parent.add_function(wrap_pyfunction!(_runtime_execution_plan_evaluate, parent)?)?;
    parent.add_function(wrap_pyfunction!(
        _runtime_execution_plan_observe_capabilities,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(_runtime_execution_plan_validate, parent)?)?;
    parent.add_function(wrap_pyfunction!(
        _runtime_execution_plan_require_executable,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(
        _runtime_execution_plan_validate_against,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(
        _tensor_execution_receipt_validate,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(
        _tensor_execution_receipt_validate_against_runtime_plan,
        parent
    )?)?;
    parent.add_function(wrap_pyfunction!(_topos_route_policy_evaluate, parent)?)?;
    parent.add_function(wrap_pyfunction!(_topos_route_policy_rewards, parent)?)?;
    parent.add_function(wrap_pyfunction!(_topos_route_policy_resolve, parent)?)?;
    Ok(())
}
