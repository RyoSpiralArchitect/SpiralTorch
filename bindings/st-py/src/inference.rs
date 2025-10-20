use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use spiral_safety::{
    AuditEvent, AuditSink, ContentChannel, SafetyPolicy, SafetyVerdict, SafetyViolation,
};

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
    m.add_class::<InferenceRuntime>()?;
    m.add_class::<InferenceResultPy>()?;
    m.add_class::<AuditLogPy>()?;
    m.add_class::<SafetyVerdictPy>()?;
    m.add_class::<SafetyViolationPy>()?;
    m.add_class::<AuditEventPy>()?;
    Ok(())
}
