//! Canonical runtime-device routing and readiness semantics.
//!
//! Clients collect device observations, while this module alone decides whether
//! a requested backend is directly ready, ready through a surrogate, unavailable,
//! or failed. Python and WASM bindings should expose this contract rather than
//! rebuilding readiness precedence and requirement gates in their host language.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use thiserror::Error;

/// Stable contract identifier shared by Rust, Python, and WASM clients.
pub const RUNTIME_DEVICE_ROUTE_CONTRACT_VERSION: &str = "spiraltorch.runtime_device_route.v4";
/// Payload kind for runtime-device route evaluation.
pub const RUNTIME_DEVICE_ROUTE_KIND: &str = "spiraltorch.runtime_device_route";
/// Crate/module that owns runtime-device route semantics.
pub const RUNTIME_DEVICE_ROUTE_SEMANTIC_OWNER: &str = "st-core::backend::runtime_route";
/// Backend label attached to payloads produced by the canonical implementation.
pub const RUNTIME_DEVICE_ROUTE_SEMANTIC_BACKEND: &str = "rust";

pub const RUNTIME_DEVICE_ROUTE_MAX_REPORTS: usize = 64;
pub const RUNTIME_DEVICE_ROUTE_MAX_BACKENDS: usize = 64;
pub const RUNTIME_DEVICE_ROUTE_MAX_LABEL_BYTES: usize = 64;
pub const RUNTIME_DEVICE_ROUTE_MAX_STATUS_BYTES: usize = 128;
pub const RUNTIME_DEVICE_ROUTE_MAX_DIAGNOSTIC_BYTES: usize = 4_096;

const RUNTIME_DEVICE_ROUTE_REQUEST_DIGEST_DOMAIN: &[u8] =
    b"spiraltorch.runtime_device_route.request.v4\0";
const RUNTIME_DEVICE_ROUTE_OUTPUT_DIGEST_DOMAIN: &[u8] =
    b"spiraltorch.runtime_device_route.output.v4\0";

#[derive(Debug, Error, PartialEq)]
pub enum RuntimeDeviceRouteError {
    #[error("runtime-device report count {actual} exceeds limit {max}")]
    TooManyReports { actual: usize, max: usize },
    #[error("runtime-device backend count for '{field}' is {actual}, exceeding limit {max}")]
    TooManyBackends {
        field: &'static str,
        actual: usize,
        max: usize,
    },
    #[error("runtime-device label '{field}' at index {index} must not be empty")]
    EmptyLabel { field: &'static str, index: usize },
    #[error(
        "runtime-device label '{field}' at index {index} has {actual} bytes, exceeding limit {max}"
    )]
    LabelTooLong {
        field: &'static str,
        index: usize,
        actual: usize,
        max: usize,
    },
    #[error(
        "runtime-device label '{field}' at index {index} contains unsupported byte {byte} at position {position}"
    )]
    InvalidLabelCharacter {
        field: &'static str,
        index: usize,
        position: usize,
        byte: u8,
    },
    #[error("runtime-device label '{label}' appears more than once in '{field}'")]
    DuplicateLabel { field: &'static str, label: String },
    #[error("runtime-device report for requested backend '{backend}' appears more than once")]
    DuplicateReport { backend: String },
    #[error(
        "runtime-device status '{field}' at report {index} has {actual} bytes, exceeding limit {max}"
    )]
    StatusTooLong {
        field: &'static str,
        index: usize,
        actual: usize,
        max: usize,
    },
    #[error(
        "runtime-device status '{field}' at report {index} contains unsupported byte {byte} at position {position}"
    )]
    InvalidStatusCharacter {
        field: &'static str,
        index: usize,
        position: usize,
        byte: u8,
    },
    #[error(
        "runtime-device diagnostic at report {index} has {actual} bytes, exceeding limit {max}"
    )]
    DiagnosticTooLong {
        index: usize,
        actual: usize,
        max: usize,
    },
    #[error(
        "runtime-device report {index} for '{backend}' disagrees on route readiness: runtime_ready={runtime_ready}, effective_backend_runtime_ready={effective_ready}"
    )]
    ConflictingRouteReadiness {
        index: usize,
        backend: String,
        runtime_ready: bool,
        effective_ready: bool,
    },
    #[error(
        "runtime-device report {index} for direct backend '{backend}' disagrees on native readiness: native_ready={native_ready}, route_ready={route_ready}"
    )]
    ConflictingDirectReadiness {
        index: usize,
        backend: String,
        native_ready: bool,
        route_ready: bool,
    },
    #[error(
        "runtime-device report {index} for direct backend '{backend}' disagrees on native evidence: requested_backend_runtime_ready={requested_ready}, available={available}"
    )]
    ConflictingDirectNativeEvidence {
        index: usize,
        backend: String,
        requested_ready: bool,
        available: bool,
    },
    #[error(
        "runtime-device report {index} for '{backend}' cannot be route-ready while its runtime status is 'error'"
    )]
    ErrorStatusMarkedReady { index: usize, backend: String },
    #[error(
        "runtime-device reports disagree on effective backend '{backend}' readiness: '{first_requested_backend}' says {first_readiness:?}, '{second_requested_backend}' says {second_readiness:?}"
    )]
    ConflictingEffectiveBackendReadiness {
        backend: String,
        first_requested_backend: String,
        first_readiness: RuntimeDeviceReadiness,
        second_requested_backend: String,
        second_readiness: RuntimeDeviceReadiness,
    },
    #[error("invalid runtime-device route payload field '{field}': {message}")]
    InvalidPayload {
        field: &'static str,
        message: String,
    },
}

/// One client-observed device report. Fields intentionally mirror the stable
/// readiness subset emitted by `describe_device`; capability details remain
/// transport metadata and do not participate in routing semantics.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct RuntimeDeviceRouteEvidence {
    pub requested_backend: String,
    pub effective_backend: Option<String>,
    pub runtime_ready: Option<bool>,
    pub requested_backend_runtime_ready: Option<bool>,
    pub effective_backend_runtime_ready: Option<bool>,
    /// Legacy direct-backend readiness evidence. Ignored for surrogate routes.
    pub available: Option<bool>,
    pub runtime_status: Option<String>,
    pub requested_backend_runtime_status: Option<String>,
    pub effective_backend_runtime_status: Option<String>,
    /// Legacy status evidence used only when runtime-specific status is absent.
    pub status: Option<String>,
    pub error: Option<String>,
}

/// Runtime-device observations plus the gates a caller wants to enforce.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct RuntimeDeviceRouteRequest {
    pub reports: Vec<RuntimeDeviceRouteEvidence>,
    pub requested_backends: Vec<String>,
    pub required_available_backends: Vec<String>,
    pub required_ready_backends: Vec<String>,
}

/// Evidence state kept separate from the fail-closed boolean projections used
/// by execution gates.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeDeviceReadiness {
    Ready,
    NotReady,
    Unknown,
}

impl From<Option<bool>> for RuntimeDeviceReadiness {
    fn from(value: Option<bool>) -> Self {
        match value {
            Some(true) => Self::Ready,
            Some(false) => Self::NotReady,
            None => Self::Unknown,
        }
    }
}

/// The boolean algebra used for the payload-level execution projection.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeDeviceReadyBasis {
    /// Every explicitly required backend must have a ready route.
    RequiredReadyBackends,
    /// With no explicit requirement, at least one observed route must be ready.
    AnyReadyBackend,
}

/// Structural class of one selected runtime route.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeDeviceRouteClass {
    Direct,
    Surrogate,
    Unavailable,
}

/// Canonical status derived from readiness and probe-error evidence.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeDeviceRouteStatus {
    Ready,
    SurrogateReady,
    NotReady,
    Unknown,
    Error,
}

/// Canonical interpretation of one runtime-device report.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeDeviceRouteRow {
    pub requested_backend: String,
    pub effective_backend: String,
    /// The report was collected without an explicit probe error. This is not
    /// evidence that either the native backend or effective route is ready.
    pub report_available: bool,
    pub native_readiness: RuntimeDeviceReadiness,
    /// Compatibility projection; `None` means native readiness is unknown.
    pub native_ready: Option<bool>,
    pub route_readiness: RuntimeDeviceReadiness,
    /// Fail-closed execution projection; only `Ready` maps to `true`.
    pub route_ready: bool,
    /// The requested and effective backend labels differ, regardless of route
    /// readiness.
    pub fallback: bool,
    pub route: RuntimeDeviceRouteClass,
    pub route_status: RuntimeDeviceRouteStatus,
    pub runtime_status: String,
    pub requested_backend_runtime_status: Option<String>,
    pub effective_backend_runtime_status: Option<String>,
    pub diagnostic: Option<String>,
}

/// Stable result consumed by Python preflight, native Rust callers, and WASM.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeDeviceRoutePayload {
    pub kind: String,
    pub contract_version: String,
    pub semantic_owner: String,
    pub semantic_backend: String,
    /// Transport provenance. This field is excluded from semantic commitments.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_client: Option<String>,
    /// Canonicalized caller observations retained for deterministic replay.
    pub evidence: Vec<RuntimeDeviceRouteEvidence>,
    /// Canonicalized explicit requests before required/report backends are appended.
    pub requested_backends: Vec<String>,
    pub backends: Vec<String>,
    pub report_count: usize,
    pub routes: Vec<RuntimeDeviceRouteRow>,
    pub available_backends: Vec<String>,
    pub native_ready_backends: Vec<String>,
    pub native_not_ready_backends: Vec<String>,
    pub native_readiness_unknown_backends: Vec<String>,
    pub ready_backends: Vec<String>,
    /// Operationally unavailable backends. This remains fail-closed and
    /// therefore includes backends whose route readiness is unknown.
    pub not_ready_backends: Vec<String>,
    pub route_not_ready_backends: Vec<String>,
    pub route_readiness_unknown_backends: Vec<String>,
    /// Backends configured with a distinct effective/surrogate backend. Route
    /// readiness must still be checked separately.
    pub fallback_backends: Vec<String>,
    pub error_backends: Vec<String>,
    pub missing_report_backends: Vec<String>,
    pub status_by_backend: BTreeMap<String, String>,
    pub all_ready: bool,
    pub has_errors: bool,
    /// Evidence-preserving payload-level readiness. Consumers should prefer
    /// this over reconstructing readiness from backend lists.
    pub runtime_readiness: RuntimeDeviceReadiness,
    /// Fail-closed execution projection; only `Ready` maps to `true`.
    pub runtime_ready: bool,
    pub runtime_ready_basis: RuntimeDeviceReadyBasis,
    /// Required routes that are not ready. Empty for the any-ready projection.
    pub runtime_missing_ready_backends: Vec<String>,
    /// Relevant routes whose readiness evidence is unknown.
    pub runtime_unknown_ready_backends: Vec<String>,
    pub required_available_backends: Vec<String>,
    pub required_available_backends_missing: Vec<String>,
    pub required_available_backends_passed: Option<bool>,
    pub required_ready_backends: Vec<String>,
    pub required_ready_backends_missing: Vec<String>,
    pub required_ready_backends_unknown: Vec<String>,
    pub required_ready_backends_passed: Option<bool>,
    pub failures: Vec<String>,
    pub passed: bool,
    pub request_sha256: String,
    pub output_sha256: String,
    pub committed: bool,
}

impl RuntimeDeviceRoutePayload {
    /// Attach transport provenance without changing the Rust-owned commitment.
    pub fn with_execution_client(
        mut self,
        execution_client: impl AsRef<str>,
    ) -> Result<Self, RuntimeDeviceRouteError> {
        self.execution_client = Some(normalized_label(
            execution_client.as_ref(),
            "execution_client",
            0,
        )?);
        self.validate()?;
        Ok(self)
    }

    /// Validate identity, commitments, every derived projection, and replay.
    pub fn validate(&self) -> Result<(), RuntimeDeviceRouteError> {
        for (field, actual, expected) in [
            ("kind", self.kind.as_str(), RUNTIME_DEVICE_ROUTE_KIND),
            (
                "contract_version",
                self.contract_version.as_str(),
                RUNTIME_DEVICE_ROUTE_CONTRACT_VERSION,
            ),
            (
                "semantic_owner",
                self.semantic_owner.as_str(),
                RUNTIME_DEVICE_ROUTE_SEMANTIC_OWNER,
            ),
            (
                "semantic_backend",
                self.semantic_backend.as_str(),
                RUNTIME_DEVICE_ROUTE_SEMANTIC_BACKEND,
            ),
        ] {
            if actual != expected {
                return Err(invalid_payload(
                    field,
                    format!("must be {expected}, got {actual}"),
                ));
            }
        }
        if let Some(execution_client) = self.execution_client.as_deref() {
            let normalized = normalized_label(execution_client, "execution_client", 0)?;
            if normalized != execution_client {
                return Err(invalid_payload(
                    "execution_client",
                    "must already use its canonical lowercase label",
                ));
            }
        }
        for (field, digest) in [
            ("request_sha256", self.request_sha256.as_str()),
            ("output_sha256", self.output_sha256.as_str()),
        ] {
            if !valid_sha256(digest) {
                return Err(invalid_payload(
                    field,
                    "must be a 64-digit lowercase SHA-256",
                ));
            }
        }
        if !self.committed {
            return Err(invalid_payload(
                "committed",
                "runtime-device route payloads must describe a committed decision",
            ));
        }

        let expected = build_runtime_device_route(RuntimeDeviceRouteRequest {
            reports: self.evidence.clone(),
            requested_backends: self.requested_backends.clone(),
            required_available_backends: self.required_available_backends.clone(),
            required_ready_backends: self.required_ready_backends.clone(),
        })?;
        if self.request_sha256 != expected.request_sha256 {
            return Err(invalid_payload(
                "request_sha256",
                "does not bind the embedded canonical evidence and requirements",
            ));
        }
        if self.output_sha256 != expected.output_sha256 {
            return Err(invalid_payload(
                "output_sha256",
                "does not bind the Rust-derived route decision",
            ));
        }
        let mut actual = self.clone();
        actual.execution_client = None;
        if actual != expected {
            return Err(invalid_payload(
                "payload",
                "derived fields do not match replay of the embedded canonical evidence",
            ));
        }
        Ok(())
    }

    /// Replay an external request and require this payload to match it exactly.
    pub fn validate_against(
        &self,
        request: RuntimeDeviceRouteRequest,
    ) -> Result<(), RuntimeDeviceRouteError> {
        self.validate()?;
        let expected = build_runtime_device_route(request)?;
        let mut actual = self.clone();
        actual.execution_client = None;
        if actual != expected {
            return Err(invalid_payload(
                "request",
                "payload does not match replay of the supplied runtime-device request",
            ));
        }
        Ok(())
    }
}

fn normalized_label(
    value: &str,
    field: &'static str,
    index: usize,
) -> Result<String, RuntimeDeviceRouteError> {
    let normalized = value.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return Err(RuntimeDeviceRouteError::EmptyLabel { field, index });
    }
    if normalized.len() > RUNTIME_DEVICE_ROUTE_MAX_LABEL_BYTES {
        return Err(RuntimeDeviceRouteError::LabelTooLong {
            field,
            index,
            actual: normalized.len(),
            max: RUNTIME_DEVICE_ROUTE_MAX_LABEL_BYTES,
        });
    }
    for (position, byte) in normalized.bytes().enumerate() {
        let supported = byte.is_ascii_alphanumeric()
            || (position > 0 && matches!(byte, b'.' | b'_' | b':' | b'+' | b'-' | b'/'));
        if !supported {
            return Err(RuntimeDeviceRouteError::InvalidLabelCharacter {
                field,
                index,
                position,
                byte,
            });
        }
    }
    Ok(normalized)
}

fn normalized_labels(
    values: &[String],
    field: &'static str,
) -> Result<Vec<String>, RuntimeDeviceRouteError> {
    if values.len() > RUNTIME_DEVICE_ROUTE_MAX_BACKENDS {
        return Err(RuntimeDeviceRouteError::TooManyBackends {
            field,
            actual: values.len(),
            max: RUNTIME_DEVICE_ROUTE_MAX_BACKENDS,
        });
    }
    let mut seen = BTreeSet::new();
    let mut normalized = Vec::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        let label = normalized_label(value, field, index)?;
        if !seen.insert(label.clone()) {
            return Err(RuntimeDeviceRouteError::DuplicateLabel { field, label });
        }
        normalized.push(label);
    }
    Ok(normalized)
}

fn normalized_status(
    value: Option<&str>,
    field: &'static str,
    index: usize,
) -> Result<Option<String>, RuntimeDeviceRouteError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let normalized = value.trim().to_ascii_lowercase();
    if normalized.len() > RUNTIME_DEVICE_ROUTE_MAX_STATUS_BYTES {
        return Err(RuntimeDeviceRouteError::StatusTooLong {
            field,
            index,
            actual: normalized.len(),
            max: RUNTIME_DEVICE_ROUTE_MAX_STATUS_BYTES,
        });
    }
    if normalized.is_empty() {
        return Ok(None);
    }
    for (position, byte) in normalized.bytes().enumerate() {
        let supported = byte.is_ascii_alphanumeric()
            || (position > 0 && matches!(byte, b'.' | b'_' | b':' | b'+' | b'-' | b'/'));
        if !supported {
            return Err(RuntimeDeviceRouteError::InvalidStatusCharacter {
                field,
                index,
                position,
                byte,
            });
        }
    }
    Ok(Some(normalized))
}

fn normalized_diagnostic(
    value: Option<&str>,
    index: usize,
) -> Result<Option<String>, RuntimeDeviceRouteError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let normalized = value.trim();
    if normalized.len() > RUNTIME_DEVICE_ROUTE_MAX_DIAGNOSTIC_BYTES {
        return Err(RuntimeDeviceRouteError::DiagnosticTooLong {
            index,
            actual: normalized.len(),
            max: RUNTIME_DEVICE_ROUTE_MAX_DIAGNOSTIC_BYTES,
        });
    }
    Ok((!normalized.is_empty()).then(|| normalized.to_owned()))
}

fn normalize_evidence(
    evidence: &RuntimeDeviceRouteEvidence,
    index: usize,
) -> Result<RuntimeDeviceRouteEvidence, RuntimeDeviceRouteError> {
    let requested_backend = normalized_label(
        &evidence.requested_backend,
        "reports.requested_backend",
        index,
    )?;
    let effective_backend = match evidence.effective_backend.as_deref() {
        Some(value) => normalized_label(value, "reports.effective_backend", index)?,
        None => requested_backend.clone(),
    };
    Ok(RuntimeDeviceRouteEvidence {
        requested_backend,
        effective_backend: Some(effective_backend),
        runtime_ready: evidence.runtime_ready,
        requested_backend_runtime_ready: evidence.requested_backend_runtime_ready,
        effective_backend_runtime_ready: evidence.effective_backend_runtime_ready,
        available: evidence.available,
        runtime_status: normalized_status(
            evidence.runtime_status.as_deref(),
            "reports.runtime_status",
            index,
        )?,
        requested_backend_runtime_status: normalized_status(
            evidence.requested_backend_runtime_status.as_deref(),
            "reports.requested_backend_runtime_status",
            index,
        )?,
        effective_backend_runtime_status: normalized_status(
            evidence.effective_backend_runtime_status.as_deref(),
            "reports.effective_backend_runtime_status",
            index,
        )?,
        status: normalized_status(evidence.status.as_deref(), "reports.status", index)?,
        error: normalized_diagnostic(evidence.error.as_deref(), index)?,
    })
}

fn evaluate_report(
    evidence: &RuntimeDeviceRouteEvidence,
    index: usize,
) -> Result<RuntimeDeviceRouteRow, RuntimeDeviceRouteError> {
    let requested_backend = evidence.requested_backend.clone();
    let effective_backend = evidence
        .effective_backend
        .clone()
        .expect("canonical runtime-device evidence always has an effective backend");
    let direct = requested_backend == effective_backend;

    if direct {
        if let (Some(requested_ready), Some(available)) =
            (evidence.requested_backend_runtime_ready, evidence.available)
        {
            if requested_ready != available {
                return Err(RuntimeDeviceRouteError::ConflictingDirectNativeEvidence {
                    index,
                    backend: requested_backend,
                    requested_ready,
                    available,
                });
            }
        }
    }

    if let (Some(runtime_ready), Some(effective_ready)) = (
        evidence.runtime_ready,
        evidence.effective_backend_runtime_ready,
    ) {
        if runtime_ready != effective_ready {
            return Err(RuntimeDeviceRouteError::ConflictingRouteReadiness {
                index,
                backend: requested_backend,
                runtime_ready,
                effective_ready,
            });
        }
    }
    let route_ready_evidence = evidence
        .runtime_ready
        .or(evidence.effective_backend_runtime_ready)
        .or_else(|| {
            if direct {
                evidence
                    .requested_backend_runtime_ready
                    .or(evidence.available)
            } else {
                None
            }
        });
    let route_ready = route_ready_evidence.unwrap_or(false);
    if direct {
        if let Some(native_ready) = evidence
            .requested_backend_runtime_ready
            .or(evidence.available)
        {
            if native_ready != route_ready {
                return Err(RuntimeDeviceRouteError::ConflictingDirectReadiness {
                    index,
                    backend: requested_backend,
                    native_ready,
                    route_ready,
                });
            }
        }
    }
    let native_ready = evidence
        .requested_backend_runtime_ready
        .or(evidence.available.filter(|_| direct))
        .or(if direct { route_ready_evidence } else { None });
    let native_readiness = native_ready.into();
    let route_readiness = route_ready_evidence.into();

    let runtime_status = evidence.runtime_status.clone();
    let requested_runtime_status = evidence.requested_backend_runtime_status.clone();
    let effective_runtime_status = evidence.effective_backend_runtime_status.clone();
    let legacy_status = evidence.status.clone();
    let runtime_status = runtime_status
        .clone()
        .or_else(|| effective_runtime_status.clone())
        .or_else(|| requested_runtime_status.clone())
        .or(legacy_status)
        .unwrap_or_else(|| "unknown".to_owned());
    let probe_error = runtime_status == "error";
    if probe_error && route_ready {
        return Err(RuntimeDeviceRouteError::ErrorStatusMarkedReady {
            index,
            backend: requested_backend,
        });
    }

    let fallback = !direct;
    let (route, route_status) = if probe_error {
        (
            RuntimeDeviceRouteClass::Unavailable,
            RuntimeDeviceRouteStatus::Error,
        )
    } else {
        match (route_readiness, fallback) {
            (RuntimeDeviceReadiness::Ready, true) => (
                RuntimeDeviceRouteClass::Surrogate,
                RuntimeDeviceRouteStatus::SurrogateReady,
            ),
            (RuntimeDeviceReadiness::Ready, false) => (
                RuntimeDeviceRouteClass::Direct,
                RuntimeDeviceRouteStatus::Ready,
            ),
            (RuntimeDeviceReadiness::NotReady, _) => (
                RuntimeDeviceRouteClass::Unavailable,
                RuntimeDeviceRouteStatus::NotReady,
            ),
            (RuntimeDeviceReadiness::Unknown, _) => (
                RuntimeDeviceRouteClass::Unavailable,
                RuntimeDeviceRouteStatus::Unknown,
            ),
        }
    };

    Ok(RuntimeDeviceRouteRow {
        requested_backend,
        effective_backend,
        report_available: !probe_error,
        native_readiness,
        native_ready,
        route_readiness,
        route_ready,
        fallback,
        route,
        route_status,
        runtime_status,
        requested_backend_runtime_status: requested_runtime_status,
        effective_backend_runtime_status: effective_runtime_status,
        diagnostic: evidence.error.clone(),
    })
}

fn append_missing(backends: &mut Vec<String>, seen: &mut BTreeSet<String>, values: &[String]) {
    for value in values {
        if seen.insert(value.clone()) {
            backends.push(value.clone());
        }
    }
}

fn validate_effective_backend_consistency(
    routes: &[RuntimeDeviceRouteRow],
) -> Result<(), RuntimeDeviceRouteError> {
    let mut observed = BTreeMap::<String, (RuntimeDeviceReadiness, String)>::new();
    for route in routes {
        if route.route_readiness == RuntimeDeviceReadiness::Unknown {
            continue;
        }
        if let Some((first_readiness, first_requested_backend)) =
            observed.get(&route.effective_backend)
        {
            if *first_readiness != route.route_readiness {
                return Err(
                    RuntimeDeviceRouteError::ConflictingEffectiveBackendReadiness {
                        backend: route.effective_backend.clone(),
                        first_requested_backend: first_requested_backend.clone(),
                        first_readiness: *first_readiness,
                        second_requested_backend: route.requested_backend.clone(),
                        second_readiness: route.route_readiness,
                    },
                );
            }
        } else {
            observed.insert(
                route.effective_backend.clone(),
                (route.route_readiness, route.requested_backend.clone()),
            );
        }
    }
    Ok(())
}

/// Evaluate runtime-device observations and requirement gates through the
/// canonical Rust-owned route contract.
pub fn evaluate_runtime_device_route(
    request: RuntimeDeviceRouteRequest,
) -> Result<RuntimeDeviceRoutePayload, RuntimeDeviceRouteError> {
    let payload = build_runtime_device_route(request)?;
    payload.validate()?;
    Ok(payload)
}

fn build_runtime_device_route(
    request: RuntimeDeviceRouteRequest,
) -> Result<RuntimeDeviceRoutePayload, RuntimeDeviceRouteError> {
    if request.reports.len() > RUNTIME_DEVICE_ROUTE_MAX_REPORTS {
        return Err(RuntimeDeviceRouteError::TooManyReports {
            actual: request.reports.len(),
            max: RUNTIME_DEVICE_ROUTE_MAX_REPORTS,
        });
    }
    let requested = normalized_labels(&request.requested_backends, "requested_backends")?;
    let required_available = normalized_labels(
        &request.required_available_backends,
        "required_available_backends",
    )?;
    let required_ready =
        normalized_labels(&request.required_ready_backends, "required_ready_backends")?;

    let mut evidence = Vec::with_capacity(request.reports.len());
    let mut routes = Vec::with_capacity(request.reports.len());
    let mut route_index = BTreeMap::new();
    for (index, raw_evidence) in request.reports.iter().enumerate() {
        let canonical_evidence = normalize_evidence(raw_evidence, index)?;
        let route = evaluate_report(&canonical_evidence, index)?;
        if route_index
            .insert(route.requested_backend.clone(), routes.len())
            .is_some()
        {
            return Err(RuntimeDeviceRouteError::DuplicateReport {
                backend: route.requested_backend,
            });
        }
        evidence.push(canonical_evidence);
        routes.push(route);
    }
    validate_effective_backend_consistency(&routes)?;

    let requested_backends = requested;
    let mut backends = requested_backends.clone();
    let mut seen = backends.iter().cloned().collect::<BTreeSet<_>>();
    append_missing(&mut backends, &mut seen, &required_available);
    append_missing(&mut backends, &mut seen, &required_ready);
    let report_backends = routes
        .iter()
        .map(|route| route.requested_backend.clone())
        .collect::<Vec<_>>();
    append_missing(&mut backends, &mut seen, &report_backends);

    let mut available_backends = Vec::new();
    let mut native_ready_backends = Vec::new();
    let mut native_not_ready_backends = Vec::new();
    let mut native_readiness_unknown_backends = Vec::new();
    let mut ready_backends = Vec::new();
    let mut not_ready_backends = Vec::new();
    let mut route_not_ready_backends = Vec::new();
    let mut route_readiness_unknown_backends = Vec::new();
    let mut fallback_backends = Vec::new();
    let mut error_backends = Vec::new();
    let mut missing_report_backends = Vec::new();
    let mut status_by_backend = BTreeMap::new();

    for backend in &backends {
        let Some(index) = route_index.get(backend).copied() else {
            missing_report_backends.push(backend.clone());
            not_ready_backends.push(backend.clone());
            native_readiness_unknown_backends.push(backend.clone());
            route_readiness_unknown_backends.push(backend.clone());
            status_by_backend.insert(backend.clone(), "missing".to_owned());
            continue;
        };
        let route = &routes[index];
        if route.report_available {
            available_backends.push(backend.clone());
        } else {
            error_backends.push(backend.clone());
        }
        match route.native_ready {
            Some(true) => native_ready_backends.push(backend.clone()),
            Some(false) => native_not_ready_backends.push(backend.clone()),
            None => native_readiness_unknown_backends.push(backend.clone()),
        }
        match route.route_readiness {
            RuntimeDeviceReadiness::Ready => ready_backends.push(backend.clone()),
            RuntimeDeviceReadiness::NotReady => {
                not_ready_backends.push(backend.clone());
                route_not_ready_backends.push(backend.clone());
            }
            RuntimeDeviceReadiness::Unknown => {
                not_ready_backends.push(backend.clone());
                route_readiness_unknown_backends.push(backend.clone());
            }
        }
        if route.fallback {
            fallback_backends.push(backend.clone());
        }
        status_by_backend.insert(backend.clone(), route.runtime_status.clone());
    }

    let available = available_backends.iter().cloned().collect::<BTreeSet<_>>();
    let ready = ready_backends.iter().cloned().collect::<BTreeSet<_>>();
    let route_unknown = route_readiness_unknown_backends
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let required_available_backends_missing = required_available
        .iter()
        .filter(|backend| !available.contains(*backend))
        .cloned()
        .collect::<Vec<_>>();
    let required_ready_backends_missing = required_ready
        .iter()
        .filter(|backend| !ready.contains(*backend))
        .cloned()
        .collect::<Vec<_>>();
    let required_ready_backends_unknown = required_ready_backends_missing
        .iter()
        .filter(|backend| route_unknown.contains(*backend))
        .cloned()
        .collect::<Vec<_>>();
    let required_available_backends_passed =
        (!required_available.is_empty()).then_some(required_available_backends_missing.is_empty());
    let required_ready_backends_passed =
        (!required_ready.is_empty()).then_some(required_ready_backends_missing.is_empty());
    let (
        runtime_readiness,
        runtime_ready_basis,
        runtime_missing_ready_backends,
        runtime_unknown_ready_backends,
    ) = if required_ready.is_empty() {
        let readiness = if !ready_backends.is_empty() {
            RuntimeDeviceReadiness::Ready
        } else if !route_readiness_unknown_backends.is_empty() {
            RuntimeDeviceReadiness::Unknown
        } else {
            RuntimeDeviceReadiness::NotReady
        };
        (
            readiness,
            RuntimeDeviceReadyBasis::AnyReadyBackend,
            Vec::new(),
            route_readiness_unknown_backends.clone(),
        )
    } else {
        let readiness = if required_ready_backends_missing.is_empty() {
            RuntimeDeviceReadiness::Ready
        } else if required_ready_backends_missing
            .iter()
            .any(|backend| !route_unknown.contains(backend))
        {
            RuntimeDeviceReadiness::NotReady
        } else {
            RuntimeDeviceReadiness::Unknown
        };
        (
            readiness,
            RuntimeDeviceReadyBasis::RequiredReadyBackends,
            required_ready_backends_missing.clone(),
            required_ready_backends_unknown.clone(),
        )
    };
    let mut failures = required_available_backends_missing
        .iter()
        .map(|backend| format!("runtime_device_missing:{backend}"))
        .collect::<Vec<_>>();
    failures.extend(required_ready_backends_missing.iter().map(|backend| {
        if route_unknown.contains(backend) {
            format!("runtime_device_readiness_unknown:{backend}")
        } else {
            format!("runtime_device_not_ready:{backend}")
        }
    }));

    let canonical_request = RuntimeDeviceRouteRequest {
        reports: evidence.clone(),
        requested_backends: requested_backends.clone(),
        required_available_backends: required_available.clone(),
        required_ready_backends: required_ready.clone(),
    };
    let request_sha256 = runtime_device_route_request_sha256(&canonical_request);
    let mut payload = RuntimeDeviceRoutePayload {
        kind: RUNTIME_DEVICE_ROUTE_KIND.to_owned(),
        contract_version: RUNTIME_DEVICE_ROUTE_CONTRACT_VERSION.to_owned(),
        semantic_owner: RUNTIME_DEVICE_ROUTE_SEMANTIC_OWNER.to_owned(),
        semantic_backend: RUNTIME_DEVICE_ROUTE_SEMANTIC_BACKEND.to_owned(),
        execution_client: None,
        evidence,
        requested_backends,
        report_count: routes.len(),
        routes,
        all_ready: !backends.is_empty() && not_ready_backends.is_empty(),
        has_errors: !error_backends.is_empty(),
        backends,
        available_backends,
        native_ready_backends,
        native_not_ready_backends,
        native_readiness_unknown_backends,
        ready_backends,
        not_ready_backends,
        route_not_ready_backends,
        route_readiness_unknown_backends,
        fallback_backends,
        error_backends,
        missing_report_backends,
        status_by_backend,
        runtime_readiness,
        runtime_ready: runtime_readiness == RuntimeDeviceReadiness::Ready,
        runtime_ready_basis,
        runtime_missing_ready_backends,
        runtime_unknown_ready_backends,
        required_available_backends: required_available,
        required_available_backends_missing,
        required_available_backends_passed,
        required_ready_backends: required_ready,
        required_ready_backends_missing,
        required_ready_backends_unknown,
        required_ready_backends_passed,
        passed: failures.is_empty(),
        failures,
        request_sha256,
        output_sha256: String::new(),
        committed: true,
    };
    payload.output_sha256 = runtime_device_route_output_sha256(&payload);
    Ok(payload)
}

fn runtime_device_route_request_sha256(request: &RuntimeDeviceRouteRequest) -> String {
    sha256_serialized(RUNTIME_DEVICE_ROUTE_REQUEST_DIGEST_DOMAIN, request)
}

fn runtime_device_route_output_sha256(payload: &RuntimeDeviceRoutePayload) -> String {
    let mut canonical = payload.clone();
    canonical.execution_client = None;
    canonical.output_sha256.clear();
    sha256_serialized(RUNTIME_DEVICE_ROUTE_OUTPUT_DIGEST_DOMAIN, &canonical)
}

fn sha256_serialized<T: Serialize>(domain: &[u8], value: &T) -> String {
    let encoded = serde_json::to_vec(value)
        .expect("runtime-device route commitment values must serialize deterministically");
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.update(encoded);
    let digest = digest.finalize();
    let mut hexadecimal = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut hexadecimal, "{byte:02x}").expect("writing to a String cannot fail");
    }
    hexadecimal
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn invalid_payload(field: &'static str, message: impl ToString) -> RuntimeDeviceRouteError {
    RuntimeDeviceRouteError::InvalidPayload {
        field,
        message: message.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(requested: &str, ready: bool, status: &str) -> RuntimeDeviceRouteEvidence {
        RuntimeDeviceRouteEvidence {
            requested_backend: requested.to_owned(),
            effective_backend: Some(requested.to_owned()),
            runtime_ready: Some(ready),
            requested_backend_runtime_ready: Some(ready),
            effective_backend_runtime_ready: Some(ready),
            available: None,
            runtime_status: Some(status.to_owned()),
            requested_backend_runtime_status: Some(status.to_owned()),
            effective_backend_runtime_status: Some(status.to_owned()),
            status: None,
            error: None,
        }
    }

    #[test]
    fn direct_and_surrogate_readiness_remain_distinct() {
        let mut mps = evidence("mps", true, "kernel_wired");
        mps.effective_backend = Some("wgpu".to_owned());
        mps.requested_backend_runtime_ready = Some(false);
        mps.requested_backend_runtime_status = Some("placeholder".to_owned());
        mps.error = Some("native MPS kernels are not wired".to_owned());

        let payload = evaluate_runtime_device_route(RuntimeDeviceRouteRequest {
            reports: vec![evidence("cpu", true, "cpu"), mps],
            requested_backends: vec!["CPU".to_owned(), " MPS ".to_owned()],
            required_available_backends: vec!["mps".to_owned()],
            required_ready_backends: vec!["mps".to_owned()],
        })
        .expect("valid route request");

        assert_eq!(payload.ready_backends, ["cpu", "mps"]);
        assert_eq!(payload.native_ready_backends, ["cpu"]);
        assert_eq!(payload.native_not_ready_backends, ["mps"]);
        assert_eq!(payload.fallback_backends, ["mps"]);
        assert!(payload.error_backends.is_empty());
        assert_eq!(payload.runtime_readiness, RuntimeDeviceReadiness::Ready);
        assert!(payload.runtime_ready);
        assert_eq!(
            payload.runtime_ready_basis,
            RuntimeDeviceReadyBasis::RequiredReadyBackends
        );
        assert!(payload.runtime_missing_ready_backends.is_empty());
        assert!(payload.passed);
        let mps = &payload.routes[1];
        assert_eq!(mps.route, RuntimeDeviceRouteClass::Surrogate);
        assert_eq!(mps.route_status, RuntimeDeviceRouteStatus::SurrogateReady);
        assert_eq!(mps.native_readiness, RuntimeDeviceReadiness::NotReady);
        assert_eq!(mps.route_readiness, RuntimeDeviceReadiness::Ready);
        assert_eq!(
            mps.diagnostic.as_deref(),
            Some("native MPS kernels are not wired")
        );
    }

    #[test]
    fn requirement_failures_are_projected_by_rust() {
        let payload = evaluate_runtime_device_route(RuntimeDeviceRouteRequest {
            reports: vec![evidence("wgpu", false, "feature_disabled")],
            requested_backends: vec!["wgpu".to_owned()],
            required_available_backends: vec!["cpu".to_owned()],
            required_ready_backends: vec!["wgpu".to_owned()],
        })
        .expect("valid route request");

        assert_eq!(payload.missing_report_backends, ["cpu"]);
        assert_eq!(payload.route_not_ready_backends, ["wgpu"]);
        assert_eq!(payload.route_readiness_unknown_backends, ["cpu"]);
        assert!(payload.required_ready_backends_unknown.is_empty());
        assert_eq!(
            payload.failures,
            [
                "runtime_device_missing:cpu",
                "runtime_device_not_ready:wgpu"
            ]
        );
        assert_eq!(payload.required_available_backends_passed, Some(false));
        assert_eq!(payload.required_ready_backends_passed, Some(false));
        assert_eq!(payload.runtime_readiness, RuntimeDeviceReadiness::NotReady);
        assert!(!payload.runtime_ready);
        assert_eq!(payload.runtime_missing_ready_backends, ["wgpu"]);
        assert!(!payload.passed);
    }

    #[test]
    fn diagnostic_text_does_not_turn_a_surrogate_into_a_probe_error() {
        let mut mps = evidence("mps", true, "kernel_wired");
        mps.effective_backend = Some("wgpu".to_owned());
        mps.requested_backend_runtime_ready = Some(false);
        mps.error = Some("MPS is a placeholder".to_owned());
        let payload = evaluate_runtime_device_route(RuntimeDeviceRouteRequest {
            reports: vec![mps],
            requested_backends: vec!["mps".to_owned()],
            ..RuntimeDeviceRouteRequest::default()
        })
        .expect("diagnostic is not a transport error");

        assert_eq!(payload.available_backends, ["mps"]);
        assert_eq!(payload.ready_backends, ["mps"]);
        assert!(!payload.has_errors);
    }

    #[test]
    fn legacy_available_only_applies_to_direct_routes() {
        let direct = RuntimeDeviceRouteEvidence {
            requested_backend: "webgpu".to_owned(),
            available: Some(true),
            status: Some("available".to_owned()),
            ..RuntimeDeviceRouteEvidence::default()
        };
        let surrogate = RuntimeDeviceRouteEvidence {
            requested_backend: "mps".to_owned(),
            effective_backend: Some("wgpu".to_owned()),
            available: Some(true),
            status: Some("placeholder".to_owned()),
            ..RuntimeDeviceRouteEvidence::default()
        };
        let payload = evaluate_runtime_device_route(RuntimeDeviceRouteRequest {
            reports: vec![direct, surrogate],
            ..RuntimeDeviceRouteRequest::default()
        })
        .expect("valid legacy evidence");

        assert_eq!(payload.ready_backends, ["webgpu"]);
        assert_eq!(payload.not_ready_backends, ["mps"]);
        assert_eq!(payload.routes[0].runtime_status, "available");
        assert_eq!(payload.routes[1].runtime_status, "placeholder");
    }

    #[test]
    fn missing_readiness_stays_unknown_while_execution_fails_closed() {
        let payload = evaluate_runtime_device_route(RuntimeDeviceRouteRequest {
            reports: vec![RuntimeDeviceRouteEvidence {
                requested_backend: "cpu".to_owned(),
                runtime_status: Some("cpu".to_owned()),
                ..RuntimeDeviceRouteEvidence::default()
            }],
            requested_backends: vec!["cpu".to_owned()],
            required_ready_backends: vec!["cpu".to_owned()],
            ..RuntimeDeviceRouteRequest::default()
        })
        .expect("missing evidence is a valid unknown observation");

        let route = &payload.routes[0];
        assert_eq!(route.native_readiness, RuntimeDeviceReadiness::Unknown);
        assert_eq!(route.native_ready, None);
        assert_eq!(route.route_readiness, RuntimeDeviceReadiness::Unknown);
        assert!(!route.route_ready);
        assert_eq!(route.route, RuntimeDeviceRouteClass::Unavailable);
        assert_eq!(route.route_status, RuntimeDeviceRouteStatus::Unknown);
        assert_eq!(payload.native_readiness_unknown_backends, ["cpu"]);
        assert_eq!(payload.route_readiness_unknown_backends, ["cpu"]);
        assert!(payload.route_not_ready_backends.is_empty());
        assert_eq!(payload.not_ready_backends, ["cpu"]);
        assert_eq!(payload.required_ready_backends_passed, Some(false));
        assert_eq!(payload.required_ready_backends_unknown, ["cpu"]);
        assert_eq!(payload.runtime_readiness, RuntimeDeviceReadiness::Unknown);
        assert!(!payload.runtime_ready);
        assert_eq!(payload.runtime_missing_ready_backends, ["cpu"]);
        assert_eq!(payload.runtime_unknown_ready_backends, ["cpu"]);
        assert_eq!(payload.failures, ["runtime_device_readiness_unknown:cpu"]);
        assert!(!payload.passed);
    }

    #[test]
    fn any_ready_projection_is_owned_by_rust() {
        let payload = evaluate_runtime_device_route(RuntimeDeviceRouteRequest {
            reports: vec![
                evidence("wgpu", true, "kernel_wired"),
                RuntimeDeviceRouteEvidence {
                    requested_backend: "cpu".to_owned(),
                    runtime_status: Some("cpu".to_owned()),
                    ..RuntimeDeviceRouteEvidence::default()
                },
            ],
            requested_backends: vec!["wgpu".to_owned(), "cpu".to_owned()],
            ..RuntimeDeviceRouteRequest::default()
        })
        .expect("valid any-ready request");

        assert_eq!(payload.runtime_readiness, RuntimeDeviceReadiness::Ready);
        assert!(payload.runtime_ready);
        assert_eq!(
            payload.runtime_ready_basis,
            RuntimeDeviceReadyBasis::AnyReadyBackend
        );
        assert!(payload.runtime_missing_ready_backends.is_empty());
        assert_eq!(payload.runtime_unknown_ready_backends, ["cpu"]);
    }

    #[test]
    fn any_ready_projection_preserves_unknown_evidence() {
        let payload = evaluate_runtime_device_route(RuntimeDeviceRouteRequest {
            reports: vec![
                evidence("wgpu", false, "feature_disabled"),
                RuntimeDeviceRouteEvidence {
                    requested_backend: "cpu".to_owned(),
                    runtime_status: Some("cpu".to_owned()),
                    ..RuntimeDeviceRouteEvidence::default()
                },
            ],
            requested_backends: vec!["wgpu".to_owned(), "cpu".to_owned()],
            ..RuntimeDeviceRouteRequest::default()
        })
        .expect("valid unknown any-ready request");

        assert_eq!(payload.runtime_readiness, RuntimeDeviceReadiness::Unknown);
        assert!(!payload.runtime_ready);
        assert_eq!(payload.runtime_unknown_ready_backends, ["cpu"]);
    }

    #[test]
    fn required_not_ready_evidence_dominates_unknown_evidence() {
        let payload = evaluate_runtime_device_route(RuntimeDeviceRouteRequest {
            reports: vec![
                evidence("wgpu", false, "feature_disabled"),
                RuntimeDeviceRouteEvidence {
                    requested_backend: "cpu".to_owned(),
                    runtime_status: Some("cpu".to_owned()),
                    ..RuntimeDeviceRouteEvidence::default()
                },
            ],
            required_ready_backends: vec!["wgpu".to_owned(), "cpu".to_owned()],
            ..RuntimeDeviceRouteRequest::default()
        })
        .expect("valid mixed-readiness requirement");

        assert_eq!(payload.runtime_readiness, RuntimeDeviceReadiness::NotReady);
        assert!(!payload.runtime_ready);
        assert_eq!(payload.runtime_missing_ready_backends, ["wgpu", "cpu"]);
        assert_eq!(payload.runtime_unknown_ready_backends, ["cpu"]);
    }

    #[test]
    fn direct_route_evidence_is_also_native_evidence() {
        let payload = evaluate_runtime_device_route(RuntimeDeviceRouteRequest {
            reports: vec![RuntimeDeviceRouteEvidence {
                requested_backend: "cpu".to_owned(),
                runtime_ready: Some(true),
                runtime_status: Some("cpu".to_owned()),
                ..RuntimeDeviceRouteEvidence::default()
            }],
            ..RuntimeDeviceRouteRequest::default()
        })
        .expect("direct route evidence is native evidence");

        let route = &payload.routes[0];
        assert_eq!(route.native_readiness, RuntimeDeviceReadiness::Ready);
        assert_eq!(route.native_ready, Some(true));
        assert_eq!(route.route_readiness, RuntimeDeviceReadiness::Ready);
        assert!(route.route_ready);
        assert!(payload.native_readiness_unknown_backends.is_empty());
        assert!(payload.route_readiness_unknown_backends.is_empty());
    }

    #[test]
    fn native_readiness_cannot_promote_an_unobserved_surrogate() {
        let payload = evaluate_runtime_device_route(RuntimeDeviceRouteRequest {
            reports: vec![RuntimeDeviceRouteEvidence {
                requested_backend: "mps".to_owned(),
                effective_backend: Some("wgpu".to_owned()),
                requested_backend_runtime_ready: Some(true),
                requested_backend_runtime_status: Some("kernel_wired".to_owned()),
                ..RuntimeDeviceRouteEvidence::default()
            }],
            ..RuntimeDeviceRouteRequest::default()
        })
        .expect("valid partial surrogate evidence");

        assert_eq!(payload.native_ready_backends, ["mps"]);
        assert!(payload.ready_backends.is_empty());
        assert_eq!(payload.not_ready_backends, ["mps"]);
        assert_eq!(
            payload.routes[0].route_readiness,
            RuntimeDeviceReadiness::Unknown
        );
        assert_eq!(
            payload.routes[0].route_status,
            RuntimeDeviceRouteStatus::Unknown
        );
        assert_eq!(payload.route_readiness_unknown_backends, ["mps"]);
    }

    #[test]
    fn contradictory_route_readiness_fails_closed() {
        let mut drifted = evidence("wgpu", true, "kernel_wired");
        drifted.effective_backend_runtime_ready = Some(false);
        let error = evaluate_runtime_device_route(RuntimeDeviceRouteRequest {
            reports: vec![drifted],
            ..RuntimeDeviceRouteRequest::default()
        })
        .expect_err("readiness drift must fail closed");

        assert!(matches!(
            error,
            RuntimeDeviceRouteError::ConflictingRouteReadiness { index: 0, .. }
        ));
    }

    #[test]
    fn contradictory_direct_native_evidence_fails_closed() {
        let mut drifted = evidence("cpu", true, "cpu");
        drifted.available = Some(false);
        let error = evaluate_runtime_device_route(RuntimeDeviceRouteRequest {
            reports: vec![drifted],
            ..RuntimeDeviceRouteRequest::default()
        })
        .expect_err("direct native evidence drift must fail closed");

        assert!(matches!(
            error,
            RuntimeDeviceRouteError::ConflictingDirectNativeEvidence { index: 0, .. }
        ));

        let mut legacy_drifted = evidence("cpu", true, "cpu");
        legacy_drifted.requested_backend_runtime_ready = None;
        legacy_drifted.available = Some(false);
        let error = evaluate_runtime_device_route(RuntimeDeviceRouteRequest {
            reports: vec![legacy_drifted],
            ..RuntimeDeviceRouteRequest::default()
        })
        .expect_err("legacy direct readiness drift must fail closed");

        assert!(matches!(
            error,
            RuntimeDeviceRouteError::ConflictingDirectReadiness { index: 0, .. }
        ));
    }

    #[test]
    fn duplicate_reports_are_rejected_after_normalization() {
        let error = evaluate_runtime_device_route(RuntimeDeviceRouteRequest {
            reports: vec![
                evidence("WGPU", true, "kernel_wired"),
                evidence(" wgpu ", true, "kernel_wired"),
            ],
            ..RuntimeDeviceRouteRequest::default()
        })
        .expect_err("duplicate reports must fail closed");

        assert_eq!(
            error,
            RuntimeDeviceRouteError::DuplicateReport {
                backend: "wgpu".to_owned()
            }
        );
    }

    #[test]
    fn payload_is_canonical_committed_and_replay_validated() {
        let request = RuntimeDeviceRouteRequest {
            reports: vec![evidence(" WGPU ", true, " KERNEL_WIRED ")],
            requested_backends: vec![" WGPU ".to_owned()],
            required_ready_backends: vec![" WGPU ".to_owned()],
            ..RuntimeDeviceRouteRequest::default()
        };
        let payload = evaluate_runtime_device_route(request.clone()).expect("valid route request");

        assert_eq!(
            payload.contract_version,
            RUNTIME_DEVICE_ROUTE_CONTRACT_VERSION
        );
        assert_eq!(payload.requested_backends, ["wgpu"]);
        assert_eq!(payload.evidence[0].requested_backend, "wgpu");
        assert_eq!(
            payload.evidence[0].effective_backend.as_deref(),
            Some("wgpu")
        );
        assert_eq!(
            payload.evidence[0].runtime_status.as_deref(),
            Some("kernel_wired")
        );
        assert!(valid_sha256(&payload.request_sha256));
        assert!(valid_sha256(&payload.output_sha256));
        assert!(payload.committed);
        payload
            .validate()
            .expect("self-contained payload validates");
        payload
            .validate_against(request)
            .expect("payload replays against its original request");

        let encoded = serde_json::to_vec(&payload).expect("payload serializes");
        let decoded: RuntimeDeviceRoutePayload =
            serde_json::from_slice(&encoded).expect("payload deserializes");
        decoded.validate().expect("wire round-trip validates");
    }

    #[test]
    fn payload_validation_rejects_tampering_and_wrong_replay_input() {
        let request = RuntimeDeviceRouteRequest {
            reports: vec![evidence("cpu", true, "cpu")],
            requested_backends: vec!["cpu".to_owned()],
            ..RuntimeDeviceRouteRequest::default()
        };
        let payload = evaluate_runtime_device_route(request.clone()).expect("valid route request");

        let mut tampered = payload.clone();
        tampered.routes[0].route_ready = false;
        assert!(matches!(
            tampered.validate(),
            Err(RuntimeDeviceRouteError::InvalidPayload {
                field: "payload",
                ..
            })
        ));

        let different = RuntimeDeviceRouteRequest {
            reports: vec![evidence("cpu", false, "feature_disabled")],
            requested_backends: vec!["cpu".to_owned()],
            ..RuntimeDeviceRouteRequest::default()
        };
        assert!(matches!(
            payload.validate_against(different),
            Err(RuntimeDeviceRouteError::InvalidPayload {
                field: "request",
                ..
            })
        ));
    }

    #[test]
    fn execution_client_is_transport_only_and_commitment_stable() {
        let payload = evaluate_runtime_device_route(RuntimeDeviceRouteRequest {
            reports: vec![evidence("cpu", true, "cpu")],
            ..RuntimeDeviceRouteRequest::default()
        })
        .expect("valid route request");
        let committed = payload.output_sha256.clone();
        let transported = payload
            .with_execution_client(" WASM ")
            .expect("valid transport client");

        assert_eq!(transported.execution_client.as_deref(), Some("wasm"));
        assert_eq!(transported.output_sha256, committed);
        transported
            .validate()
            .expect("transported payload validates");
    }

    #[test]
    fn effective_backend_readiness_must_agree_across_reports() {
        let mut mps = evidence("mps", true, "kernel_wired");
        mps.effective_backend = Some("wgpu".to_owned());
        mps.requested_backend_runtime_ready = Some(false);
        let error = evaluate_runtime_device_route(RuntimeDeviceRouteRequest {
            reports: vec![mps, evidence("wgpu", false, "feature_disabled")],
            ..RuntimeDeviceRouteRequest::default()
        })
        .expect_err("effective-backend drift must fail closed");

        assert!(matches!(
            error,
            RuntimeDeviceRouteError::ConflictingEffectiveBackendReadiness {
                ref backend,
                ..
            } if backend == "wgpu"
        ));
    }

    #[test]
    fn backend_and_status_labels_reject_ambiguous_wire_tokens() {
        let invalid_backend = evaluate_runtime_device_route(RuntimeDeviceRouteRequest {
            reports: vec![evidence("wgpu\nshadow", true, "kernel_wired")],
            ..RuntimeDeviceRouteRequest::default()
        })
        .expect_err("control characters must not enter backend labels");
        assert!(matches!(
            invalid_backend,
            RuntimeDeviceRouteError::InvalidLabelCharacter { .. }
        ));

        let invalid_status = evaluate_runtime_device_route(RuntimeDeviceRouteRequest {
            reports: vec![evidence("wgpu", true, "kernel wired")],
            ..RuntimeDeviceRouteRequest::default()
        })
        .expect_err("status labels must remain machine-readable tokens");
        assert!(matches!(
            invalid_status,
            RuntimeDeviceRouteError::InvalidStatusCharacter { .. }
        ));
    }
}
