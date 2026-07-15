use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use pyo3::wrap_pyfunction;
use spiral_safety::{
    AuditEvent, AuditSink, ContentChannel, SafetyPolicy, SafetyVerdict, SafetyViolation,
};
use st_core::inference::concept_diffusion::{apply_concept_diffusion, ConceptDiffusionRequest};
use st_core::inference::generation_control::{
    apply_zspace_generation_control, ZSpaceGenerationControlRequest,
};
use st_core::inference::imaginary_time_schrodinger::{
    apply_imaginary_time_schrodinger, ImaginaryTimeSchrodingerRequest,
};
use st_core::inference::temperature_control::{
    apply_temperature_control, TemperatureControlRequest,
};
use st_core::inference::zspace_coherence::{
    project_zspace_coherence, ZSpaceCoherenceProjectionRequest,
};
use st_core::inference::zspace_posterior::{
    decode_zspace_posterior, project_zspace_posterior, ZSpacePosteriorDecodeRequest,
    ZSpacePosteriorProjectionRequest,
};

fn json_error(context: &str, error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!("{context}: {error}"))
}

#[pyfunction]
fn _zspace_generation_control(py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let request = crate::json::py_to_json(request)?;
    let request_object = request.as_object().ok_or_else(|| {
        PyValueError::new_err("Z-space generation control request must be a mapping")
    })?;
    for field in ["logits", "token_ids", "recent_tokens"] {
        if request_object
            .get(field)
            .is_some_and(|value| !value.is_array())
        {
            return Err(PyValueError::new_err(format!(
                "Z-space generation control '{field}' must be a sequence"
            )));
        }
    }
    if request_object
        .get("config")
        .is_some_and(|value| !value.is_object())
    {
        return Err(PyValueError::new_err(
            "Z-space generation control 'config' must be a mapping",
        ));
    }
    let request: ZSpaceGenerationControlRequest = serde_json::from_value(request)
        .map_err(|error| json_error("invalid Z-space generation control request", error))?;
    let payload = apply_zspace_generation_control(request)
        .map_err(|error| json_error("Z-space generation control failed", error))?;
    let payload = serde_json::to_value(payload)
        .map_err(|error| json_error("Z-space generation control encoding failed", error))?;
    crate::json::json_to_py(py, &payload)
}

#[pyfunction]
fn _zspace_temperature_control(py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let request = crate::json::py_to_json(request)?;
    let request_object = request.as_object().ok_or_else(|| {
        PyValueError::new_err("Z-space temperature control request must be a mapping")
    })?;
    if request_object
        .get("probabilities")
        .is_some_and(|value| !value.is_array())
    {
        return Err(PyValueError::new_err(
            "Z-space temperature control 'probabilities' must be a sequence",
        ));
    }
    for field in ["config", "state"] {
        if request_object
            .get(field)
            .is_some_and(|value| !value.is_object())
        {
            return Err(PyValueError::new_err(format!(
                "Z-space temperature control '{field}' must be a mapping"
            )));
        }
    }
    if request_object
        .get("feedback")
        .is_some_and(|value| !value.is_object() && !value.is_null())
    {
        return Err(PyValueError::new_err(
            "Z-space temperature control 'feedback' must be a mapping",
        ));
    }
    let request: TemperatureControlRequest = serde_json::from_value(request)
        .map_err(|error| json_error("invalid Z-space temperature control request", error))?;
    let payload = apply_temperature_control(request)
        .map_err(|error| json_error("Z-space temperature control failed", error))?;
    let payload = serde_json::to_value(payload)
        .map_err(|error| json_error("Z-space temperature control encoding failed", error))?;
    crate::json::json_to_py(py, &payload)
}

#[pyfunction]
fn _zspace_concept_diffusion(py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let request = crate::json::py_to_json(request)?;
    let request_object = request.as_object().ok_or_else(|| {
        PyValueError::new_err("Z-space concept diffusion request must be a mapping")
    })?;
    for field in ["tags", "state", "affinity", "z_bias"] {
        if request_object
            .get(field)
            .is_some_and(|value| !value.is_array())
        {
            return Err(PyValueError::new_err(format!(
                "Z-space concept diffusion '{field}' must be a sequence"
            )));
        }
    }
    if request_object
        .get("diffusion_tensor")
        .is_some_and(|value| !value.is_array() && !value.is_null())
    {
        return Err(PyValueError::new_err(
            "Z-space concept diffusion 'diffusion_tensor' must be a sequence",
        ));
    }
    if request_object
        .get("observation")
        .is_some_and(|value| !value.is_object() && !value.is_null())
    {
        return Err(PyValueError::new_err(
            "Z-space concept diffusion 'observation' must be a mapping",
        ));
    }
    if request_object
        .get("config")
        .is_some_and(|value| !value.is_object())
    {
        return Err(PyValueError::new_err(
            "Z-space concept diffusion 'config' must be a mapping",
        ));
    }
    let request: ConceptDiffusionRequest = serde_json::from_value(request)
        .map_err(|error| json_error("invalid Z-space concept diffusion request", error))?;
    let payload = apply_concept_diffusion(request)
        .map_err(|error| json_error("Z-space concept diffusion failed", error))?;
    let payload = serde_json::to_value(payload)
        .map_err(|error| json_error("Z-space concept diffusion encoding failed", error))?;
    crate::json::json_to_py(py, &payload)
}

#[pyfunction]
fn _zspace_imaginary_time_schrodinger(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let request = crate::json::py_to_json(request)?;
    let request_object = request.as_object().ok_or_else(|| {
        PyValueError::new_err("Z-space imaginary-time Schrodinger request must be a mapping")
    })?;
    for field in ["tags", "potential", "edges", "initial_amplitude"] {
        if request_object
            .get(field)
            .is_some_and(|value| !value.is_array())
        {
            return Err(PyValueError::new_err(format!(
                "Z-space imaginary-time Schrodinger '{field}' must be a sequence"
            )));
        }
    }
    if request_object
        .get("config")
        .is_some_and(|value| !value.is_object())
    {
        return Err(PyValueError::new_err(
            "Z-space imaginary-time Schrodinger 'config' must be a mapping",
        ));
    }
    let request: ImaginaryTimeSchrodingerRequest = serde_json::from_value(request)
        .map_err(|error| json_error("invalid Z-space imaginary-time Schrodinger request", error))?;
    let payload = apply_imaginary_time_schrodinger(request)
        .map_err(|error| json_error("Z-space imaginary-time Schrodinger failed", error))?;
    let payload = serde_json::to_value(payload)
        .map_err(|error| json_error("Z-space imaginary-time Schrodinger encoding failed", error))?;
    crate::json::json_to_py(py, &payload)
}

#[pyfunction]
fn _zspace_posterior_decode(py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let request = crate::json::py_to_json(request)?;
    let request_object = request.as_object().ok_or_else(|| {
        PyValueError::new_err("Z-space posterior decode request must be a mapping")
    })?;
    if request_object
        .get("z_state")
        .is_some_and(|value| !value.is_array())
    {
        return Err(PyValueError::new_err(
            "Z-space posterior decode 'z_state' must be a sequence",
        ));
    }
    let request: ZSpacePosteriorDecodeRequest = serde_json::from_value(request)
        .map_err(|error| json_error("invalid Z-space posterior decode request", error))?;
    let payload = py
        .allow_threads(|| decode_zspace_posterior(request))
        .map_err(|error| json_error("Z-space posterior decode failed", error))?;
    let payload = serde_json::to_value(payload)
        .map_err(|error| json_error("Z-space posterior decode encoding failed", error))?;
    crate::json::json_to_py(py, &payload)
}

#[pyfunction]
fn _zspace_posterior_project(py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let request = crate::json::py_to_json(request)?;
    let request_object = request.as_object().ok_or_else(|| {
        PyValueError::new_err("Z-space posterior projection request must be a mapping")
    })?;
    if request_object
        .get("z_state")
        .is_some_and(|value| !value.is_array())
    {
        return Err(PyValueError::new_err(
            "Z-space posterior projection 'z_state' must be a sequence",
        ));
    }
    if request_object
        .get("partial")
        .is_some_and(|value| !value.is_object())
    {
        return Err(PyValueError::new_err(
            "Z-space posterior projection 'partial' must be a mapping",
        ));
    }
    if request_object
        .get("telemetry")
        .is_some_and(|value| !value.is_array())
    {
        return Err(PyValueError::new_err(
            "Z-space posterior projection 'telemetry' must be a sequence",
        ));
    }
    let request: ZSpacePosteriorProjectionRequest = serde_json::from_value(request)
        .map_err(|error| json_error("invalid Z-space posterior projection request", error))?;
    let payload = py
        .allow_threads(|| project_zspace_posterior(request))
        .map_err(|error| json_error("Z-space posterior projection failed", error))?;
    let payload = serde_json::to_value(payload)
        .map_err(|error| json_error("Z-space posterior projection encoding failed", error))?;
    crate::json::json_to_py(py, &payload)
}

#[pyfunction]
fn _zspace_coherence_project(py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let request = crate::json::py_to_json(request)?;
    let request_object = request.as_object().ok_or_else(|| {
        PyValueError::new_err("Z-space coherence projection request must be a mapping")
    })?;
    if request_object
        .get("diagnostics")
        .is_some_and(|value| !value.is_object())
    {
        return Err(PyValueError::new_err(
            "Z-space coherence projection 'diagnostics' must be a mapping",
        ));
    }
    if request_object
        .get("coherence")
        .is_some_and(|value| !value.is_array())
    {
        return Err(PyValueError::new_err(
            "Z-space coherence projection 'coherence' must be a sequence",
        ));
    }
    if request_object
        .get("contour")
        .is_some_and(|value| !value.is_object() && !value.is_null())
    {
        return Err(PyValueError::new_err(
            "Z-space coherence projection 'contour' must be a mapping",
        ));
    }
    if request_object
        .get("config")
        .is_some_and(|value| !value.is_object())
    {
        return Err(PyValueError::new_err(
            "Z-space coherence projection 'config' must be a mapping",
        ));
    }
    let request: ZSpaceCoherenceProjectionRequest = serde_json::from_value(request)
        .map_err(|error| json_error("invalid Z-space coherence projection request", error))?;
    let payload = py
        .allow_threads(|| project_zspace_coherence(request))
        .map_err(|error| json_error("Z-space coherence projection failed", error))?;
    let payload = serde_json::to_value(payload)
        .map_err(|error| json_error("Z-space coherence projection encoding failed", error))?;
    crate::json::json_to_py(py, &payload)
}

#[pyclass(module = "spiraltorch.inference", name = "SafetyViolation")]
#[derive(Clone)]
struct SafetyViolationPy {
    #[pyo3(get)]
    category: String,
    #[pyo3(get)]
    offending_term: String,
    #[pyo3(get)]
    message: String,
    #[pyo3(get)]
    risk: String,
    #[pyo3(get)]
    score: f32,
}

impl From<SafetyViolation> for SafetyViolationPy {
    fn from(value: SafetyViolation) -> Self {
        Self {
            category: value.category.to_string(),
            offending_term: value.offending_term,
            message: value.message,
            risk: value.risk.to_string(),
            score: value.score,
        }
    }
}

#[pyclass(module = "spiraltorch.inference", name = "SafetyVerdict")]
#[derive(Clone)]
struct SafetyVerdictPy {
    #[pyo3(get)]
    channel: String,
    #[pyo3(get)]
    score: f32,
    #[pyo3(get)]
    allowed: bool,
    #[pyo3(get)]
    violations: Vec<SafetyViolationPy>,
    #[pyo3(get)]
    dominant_risk: String,
}

impl From<SafetyVerdict> for SafetyVerdictPy {
    fn from(value: SafetyVerdict) -> Self {
        let dominant = value.dominant_risk().to_string();
        Self {
            channel: value.channel.to_string(),
            score: value.score,
            allowed: value.allowed,
            violations: value.violations.into_iter().map(Into::into).collect(),
            dominant_risk: dominant,
        }
    }
}

#[pyclass(module = "spiraltorch.inference", name = "AuditEvent")]
#[derive(Clone)]
struct AuditEventPy {
    #[pyo3(get)]
    timestamp: String,
    #[pyo3(get)]
    channel: String,
    #[pyo3(get)]
    content_preview: String,
    #[pyo3(get)]
    verdict: SafetyVerdictPy,
}

impl From<AuditEvent> for AuditEventPy {
    fn from(value: AuditEvent) -> Self {
        Self {
            timestamp: value.timestamp.to_rfc3339(),
            channel: value.channel.to_string(),
            content_preview: value.content_preview,
            verdict: value.verdict.into(),
        }
    }
}

#[pyclass(module = "spiraltorch.inference", name = "AuditLog")]
#[derive(Clone)]
pub struct AuditLogPy {
    sink: AuditSink,
}

#[pymethods]
impl AuditLogPy {
    fn entries(&self) -> Vec<AuditEventPy> {
        self.sink.snapshot().into_iter().map(Into::into).collect()
    }

    fn clear(&self) {
        self.sink.clear();
    }
}

#[pyclass(module = "spiraltorch.inference", name = "InferenceResult")]
#[derive(Clone)]
pub struct InferenceResultPy {
    #[pyo3(get)]
    accepted: bool,
    #[pyo3(get)]
    response: Option<String>,
    #[pyo3(get)]
    refusal_message: Option<String>,
    #[pyo3(get)]
    prompt_verdict: SafetyVerdictPy,
    #[pyo3(get)]
    response_verdict: Option<SafetyVerdictPy>,
}

impl InferenceResultPy {
    fn success(
        prompt_verdict: SafetyVerdictPy,
        response_verdict: SafetyVerdictPy,
        response: String,
    ) -> Self {
        Self {
            accepted: true,
            response: Some(response),
            refusal_message: None,
            prompt_verdict,
            response_verdict: Some(response_verdict),
        }
    }

    fn refusal(
        prompt_verdict: SafetyVerdictPy,
        response_verdict: Option<SafetyVerdictPy>,
        refusal_message: String,
    ) -> Self {
        Self {
            accepted: false,
            response: None,
            refusal_message: Some(refusal_message),
            prompt_verdict,
            response_verdict,
        }
    }
}

#[pyclass(module = "spiraltorch.inference")]
pub struct InferenceRuntime {
    policy: SafetyPolicy,
}

#[pymethods]
impl InferenceRuntime {
    #[new]
    #[pyo3(signature = (refusal_threshold=None))]
    pub fn new(refusal_threshold: Option<f32>) -> PyResult<Self> {
        let sink = AuditSink::default();
        let mut policy = SafetyPolicy::with_default_terms().with_audit_sink(sink);
        if let Some(threshold) = refusal_threshold {
            if threshold <= 0.0 {
                return Err(PyValueError::new_err("refusal_threshold must be positive"));
            }
            policy = policy.with_refusal_threshold(threshold);
        }
        Ok(Self { policy })
    }

    #[pyo3(signature = (prompt, metadata=None))]
    pub fn generate(
        &self,
        _py: Python<'_>,
        prompt: &str,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<InferenceResultPy> {
        if let Some(meta) = metadata {
            if meta.contains("force_error")? {
                return Err(PyRuntimeError::new_err("forced error via metadata"));
            }
        }

        let prompt_verdict = self.policy.evaluate(prompt, ContentChannel::Prompt);
        if prompt_verdict.should_refuse() {
            let refusal = self.policy.refusal_reason(&prompt_verdict).into_owned();
            return Ok(InferenceResultPy::refusal(
                prompt_verdict.into(),
                None,
                refusal,
            ));
        }

        // Placeholder inference implementation; integrates safety for outputs as well.
        let response = if let Some(meta) = metadata {
            if let Some(candidate) = meta.get_item("candidate")? {
                candidate.extract::<String>()?
            } else {
                format!("Generated response for: {prompt}")
            }
        } else {
            format!("Generated response for: {prompt}")
        };
        let response_verdict = self.policy.evaluate(&response, ContentChannel::Response);
        if response_verdict.should_refuse() {
            let refusal = self.policy.refusal_reason(&response_verdict).into_owned();
            return Ok(InferenceResultPy::refusal(
                prompt_verdict.into(),
                Some(response_verdict.into()),
                refusal,
            ));
        }

        Ok(InferenceResultPy::success(
            prompt_verdict.into(),
            response_verdict.into(),
            response,
        ))
    }

    #[getter]
    pub fn audit_log(&self) -> AuditLogPy {
        AuditLogPy {
            sink: self.policy.audit_sink(),
        }
    }
}

pub fn register(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_zspace_generation_control, m)?)?;
    m.add_function(wrap_pyfunction!(_zspace_temperature_control, m)?)?;
    m.add_function(wrap_pyfunction!(_zspace_concept_diffusion, m)?)?;
    m.add_function(wrap_pyfunction!(_zspace_imaginary_time_schrodinger, m)?)?;
    m.add_function(wrap_pyfunction!(_zspace_posterior_decode, m)?)?;
    m.add_function(wrap_pyfunction!(_zspace_posterior_project, m)?)?;
    m.add_function(wrap_pyfunction!(_zspace_coherence_project, m)?)?;
    m.add_class::<InferenceRuntime>()?;
    m.add_class::<InferenceResultPy>()?;
    m.add_class::<AuditLogPy>()?;
    m.add_class::<SafetyVerdictPy>()?;
    m.add_class::<SafetyViolationPy>()?;
    m.add_class::<AuditEventPy>()?;
    Ok(())
}
