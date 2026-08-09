//! Workload-specific runtime capability evidence for tensor execution plans.
//!
//! Capability observation is intentionally separate from plan evaluation.
//! Native Rust observes mutable local runtime state once and commits a typed
//! observation contract. Execution plans embed and replay that contract without
//! querying the receiving process's device again.

use super::execution_plan::{
    RuntimeExecutionComponent, RuntimeTensorBackend, RuntimeTensorBackendPolicy,
};
use super::runtime_probe::{RuntimeDeviceProbeError, RuntimeDeviceProbePayload};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use st_tensor::execution_capability::{
    observe_tensor_execution_capability, TensorExecutionBackend, TensorExecutionCapabilityStatus,
    TensorExecutionReadyProof, TensorExecutionWorkload, TensorExecutionWorkloadKey,
    TensorUtilOperation,
};
use thiserror::Error;

/// Stable contract identifier for Rust-owned component capability observations.
pub const RUNTIME_COMPONENT_CAPABILITY_OBSERVATION_CONTRACT_VERSION: &str =
    "spiraltorch.runtime_component_capability_observation.v3";
/// Payload kind for one committed component capability observation.
pub const RUNTIME_COMPONENT_CAPABILITY_OBSERVATION_KIND: &str =
    "spiraltorch.runtime_component_capability_observation";
/// Crate/module that owns component capability observation semantics.
pub const RUNTIME_COMPONENT_CAPABILITY_OBSERVATION_SEMANTIC_OWNER: &str =
    "st-core::backend::execution_capability";
/// Backend label attached to payloads produced by the canonical observer.
pub const RUNTIME_COMPONENT_CAPABILITY_OBSERVATION_SEMANTIC_BACKEND: &str = "rust";

const RUNTIME_COMPONENT_CAPABILITY_OBSERVATION_REQUEST_DIGEST_DOMAIN: &[u8] =
    b"spiraltorch.runtime_component_capability_observation.request.v3\0";
const RUNTIME_COMPONENT_CAPABILITY_OBSERVATION_OUTPUT_DIGEST_DOMAIN: &[u8] =
    b"spiraltorch.runtime_component_capability_observation.output.v3\0";

#[derive(Debug, Error, PartialEq)]
pub enum RuntimeComponentCapabilityObservationError {
    #[error(transparent)]
    RuntimeProbe(#[from] RuntimeDeviceProbeError),
    #[error("invalid runtime component capability observation request field '{field}': {message}")]
    InvalidRequest {
        field: &'static str,
        message: String,
    },
    #[error("invalid runtime component capability observation payload field '{field}': {message}")]
    InvalidPayload {
        field: &'static str,
        message: String,
    },
    #[error("runtime component capability observation encoding failed: {message}")]
    Encoding { message: String },
}

/// Tensor-utility kernels that can currently be preflighted by an execution plan.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeTensorUtilOperation {
    Scale,
    MaxAxis0,
    MaxAxis0Backward,
}

/// One concrete workload whose native implementation can be observed.
#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(tag = "component", rename_all = "snake_case", deny_unknown_fields)]
pub enum RuntimeComponentWorkload {
    DenseMatmul {
        rows: u64,
        inner: u64,
        cols: u64,
    },
    PrepackedMatmul {
        rows: u64,
        inner: u64,
        cols: u64,
        #[serde(default)]
        bias: bool,
    },
    LayerNorm {
        rows: u64,
        cols: u64,
    },
    Attention {
        contexts: u64,
        sequence: u64,
        head_dim: u64,
        #[serde(default)]
        z_bias: bool,
        #[serde(default)]
        attn_bias: bool,
    },
    /// Row-major row softmax. Other tensor layouts require separate evidence.
    Softmax {
        rows: u64,
        cols: u64,
    },
    TensorUtil {
        operation: RuntimeTensorUtilOperation,
        rows: u64,
        cols: u64,
    },
}

impl RuntimeComponentWorkload {
    pub const fn component(&self) -> RuntimeExecutionComponent {
        match self {
            Self::DenseMatmul { .. } => RuntimeExecutionComponent::DenseMatmul,
            Self::PrepackedMatmul { .. } => RuntimeExecutionComponent::PrepackedMatmul,
            Self::LayerNorm { .. } => RuntimeExecutionComponent::LayerNorm,
            Self::Attention { .. } => RuntimeExecutionComponent::Attention,
            Self::Softmax { .. } => RuntimeExecutionComponent::Softmax,
            Self::TensorUtil { .. } => RuntimeExecutionComponent::TensorUtil,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        match self {
            Self::DenseMatmul { rows, inner, cols }
            | Self::PrepackedMatmul {
                rows, inner, cols, ..
            } => validate_dimensions(
                self.component(),
                &[("rows", *rows), ("inner", *inner), ("cols", *cols)],
                &[
                    ("lhs", &[*rows, *inner]),
                    ("rhs", &[*inner, *cols]),
                    ("output", &[*rows, *cols]),
                ],
            ),
            Self::LayerNorm { rows, cols }
            | Self::Softmax { rows, cols }
            | Self::TensorUtil { rows, cols, .. } => validate_dimensions(
                self.component(),
                &[("rows", *rows), ("cols", *cols)],
                &[("values", &[*rows, *cols])],
            ),
            Self::Attention {
                contexts,
                sequence,
                head_dim,
                z_bias,
                attn_bias,
            } => {
                validate_dimensions(
                    self.component(),
                    &[
                        ("contexts", *contexts),
                        ("sequence", *sequence),
                        ("head_dim", *head_dim),
                    ],
                    &[("values", &[*contexts, *sequence, *head_dim])],
                )?;
                if *z_bias {
                    validate_volume(self.component(), "z_bias", &[*contexts, *sequence])?;
                }
                if *attn_bias {
                    validate_volume(
                        self.component(),
                        "attn_bias",
                        &[*contexts, *sequence, *sequence],
                    )?;
                }
                Ok(())
            }
        }
    }
}

fn validate_dimensions(
    component: RuntimeExecutionComponent,
    dimensions: &[(&str, u64)],
    volumes: &[(&str, &[u64])],
) -> Result<(), String> {
    for (name, value) in dimensions {
        if *value == 0 {
            return Err(format!(
                "{} workload dimension '{name}' must be positive",
                component.as_str()
            ));
        }
    }
    for (name, factors) in volumes {
        validate_volume(component, name, factors)?;
    }
    Ok(())
}

fn validate_volume(
    component: RuntimeExecutionComponent,
    name: &str,
    factors: &[u64],
) -> Result<(), String> {
    if factors
        .iter()
        .try_fold(1_u64, |volume, factor| volume.checked_mul(*factor))
        .is_none()
    {
        return Err(format!(
            "{} workload volume '{name}' exceeds u64 range",
            component.as_str()
        ));
    }
    Ok(())
}

/// Result captured by the Rust local capability observer.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeComponentCapabilityStatus {
    Ready,
    Unavailable,
    NotBuilt,
    Unsupported,
}

/// Proof required before a component capability can be reported as ready.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeComponentReadyProof {
    /// A compiled host implementation with a validated complete shape contract.
    StaticHostContract,
    /// Exact accelerator preflight plus an operation-specific dispatch/readback sentinel.
    RuntimeDispatchSentinel,
}

impl RuntimeComponentCapabilityStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Unavailable => "unavailable",
            Self::NotBuilt => "not_built",
            Self::Unsupported => "unsupported",
        }
    }
}

/// Committed observation for one component workload and selected backend.
///
/// This is deterministic replay evidence, not cryptographic hardware
/// attestation. Local callers should obtain it from the Rust observer and
/// retain the enclosing request commitment.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeComponentCapabilityEvidence {
    pub workload: RuntimeComponentWorkload,
    pub backend: RuntimeTensorBackend,
    pub status: RuntimeComponentCapabilityStatus,
    pub ready_proof: Option<RuntimeComponentReadyProof>,
}

impl RuntimeComponentCapabilityEvidence {
    pub const fn component(&self) -> RuntimeExecutionComponent {
        self.workload.component()
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        self.workload.validate()?;
        match (self.status, self.ready_proof, self.backend) {
            (
                RuntimeComponentCapabilityStatus::Ready,
                Some(RuntimeComponentReadyProof::StaticHostContract),
                RuntimeTensorBackend::Cpu
                | RuntimeTensorBackend::CpuSimd
                | RuntimeTensorBackend::Naive
                | RuntimeTensorBackend::Faer,
            )
            | (
                RuntimeComponentCapabilityStatus::Ready,
                Some(RuntimeComponentReadyProof::RuntimeDispatchSentinel),
                RuntimeTensorBackend::Wgpu | RuntimeTensorBackend::Hip,
            ) => Ok(()),
            (RuntimeComponentCapabilityStatus::Ready, None, _) => {
                Err("ready capability is missing its required proof".to_owned())
            }
            (RuntimeComponentCapabilityStatus::Ready, Some(proof), backend) => Err(format!(
                "ready proof '{proof:?}' is invalid for backend '{}'",
                backend.as_str()
            )),
            (_, None, _) => Ok(()),
            (_, Some(_), _) => Err("non-ready capability must not carry a ready proof".to_owned()),
        }
    }
}

/// Inputs bound to one local component capability observation.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeComponentCapabilityObservationRequest {
    pub runtime_probe: RuntimeDeviceProbePayload,
    /// Component backends resolved by the execution-plan semantic owner.
    pub policy: RuntimeTensorBackendPolicy,
    pub component_workloads: Vec<RuntimeComponentWorkload>,
}

impl RuntimeComponentCapabilityObservationRequest {
    fn canonicalized(mut self) -> Result<Self, RuntimeComponentCapabilityObservationError> {
        self.runtime_probe.validate()?;
        self.runtime_probe.execution_client = None;
        canonicalize_component_workloads(&mut self.component_workloads)
            .map_err(|message| invalid_observation_request("component_workloads", message))?;
        Ok(self)
    }

    fn validate_canonical(&self) -> Result<(), RuntimeComponentCapabilityObservationError> {
        let canonical = self.clone().canonicalized()?;
        if canonical != *self {
            return Err(invalid_observation_request(
                "request",
                "must use canonical operation ordering and omit nested transport provenance",
            ));
        }
        Ok(())
    }
}

/// Replayable Rust-owned observation of workload-specific component readiness.
///
/// The commitments provide deterministic integrity and lineage, not remote
/// hardware attestation. Persisted observations can be replayed without
/// querying the receiving process's mutable device runtime.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeComponentCapabilityObservationPayload {
    pub kind: String,
    pub contract_version: String,
    pub semantic_owner: String,
    pub semantic_backend: String,
    pub request: RuntimeComponentCapabilityObservationRequest,
    pub runtime_probe_output_sha256: String,
    pub capabilities: Vec<RuntimeComponentCapabilityEvidence>,
    pub request_sha256: String,
    pub output_sha256: String,
    pub committed: bool,
}

impl RuntimeComponentCapabilityObservationPayload {
    /// Validate identity, lineage, canonical evidence, and both commitments.
    pub fn validate(&self) -> Result<(), RuntimeComponentCapabilityObservationError> {
        for (field, actual, expected) in [
            (
                "kind",
                self.kind.as_str(),
                RUNTIME_COMPONENT_CAPABILITY_OBSERVATION_KIND,
            ),
            (
                "contract_version",
                self.contract_version.as_str(),
                RUNTIME_COMPONENT_CAPABILITY_OBSERVATION_CONTRACT_VERSION,
            ),
            (
                "semantic_owner",
                self.semantic_owner.as_str(),
                RUNTIME_COMPONENT_CAPABILITY_OBSERVATION_SEMANTIC_OWNER,
            ),
            (
                "semantic_backend",
                self.semantic_backend.as_str(),
                RUNTIME_COMPONENT_CAPABILITY_OBSERVATION_SEMANTIC_BACKEND,
            ),
        ] {
            if actual != expected {
                return Err(invalid_observation_payload(
                    field,
                    format!("must be '{expected}', got '{actual}'"),
                ));
            }
        }
        if !self.committed {
            return Err(invalid_observation_payload(
                "committed",
                "component capability observations must be committed",
            ));
        }
        if !valid_sha256(&self.runtime_probe_output_sha256)
            || !valid_sha256(&self.request_sha256)
            || !valid_sha256(&self.output_sha256)
        {
            return Err(invalid_observation_payload(
                "commitment",
                "all commitment fields must be lowercase SHA-256 values",
            ));
        }
        self.request.validate_canonical()?;
        if self.runtime_probe_output_sha256 != self.request.runtime_probe.output_sha256 {
            return Err(invalid_observation_payload(
                "runtime_probe_output_sha256",
                "must match the committed runtime probe",
            ));
        }
        let mut canonical_capabilities = self.capabilities.clone();
        canonicalize_component_capabilities(&mut canonical_capabilities, &self.request)?;
        if canonical_capabilities != self.capabilities {
            return Err(invalid_observation_payload(
                "capabilities",
                "must use canonical operation ordering",
            ));
        }

        let expected_request_sha256 = observation_digest_json(
            RUNTIME_COMPONENT_CAPABILITY_OBSERVATION_REQUEST_DIGEST_DOMAIN,
            &self.request,
        )?;
        if self.request_sha256 != expected_request_sha256 {
            return Err(invalid_observation_payload(
                "request_sha256",
                "does not bind the canonical observation request",
            ));
        }
        let expected_output_sha256 = observation_output_digest(self)?;
        if self.output_sha256 != expected_output_sha256 {
            return Err(invalid_observation_payload(
                "output_sha256",
                "does not bind the Rust-owned capability observation",
            ));
        }
        Ok(())
    }

    /// Validate this artifact against an explicit replay request.
    pub fn validate_against(
        &self,
        request: RuntimeComponentCapabilityObservationRequest,
    ) -> Result<(), RuntimeComponentCapabilityObservationError> {
        let request = request.canonicalized()?;
        if self.request != request {
            return Err(invalid_observation_payload(
                "request",
                "does not match the supplied replay request",
            ));
        }
        self.validate()
    }
}

/// Capability state projected onto a component route.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeComponentCapabilityState {
    /// The compiled host implementation does not require a mutable runtime probe.
    Static,
    Ready,
    Unobserved,
    Unavailable,
    NotBuilt,
    Unsupported,
    NotApplicable,
}

impl RuntimeComponentCapabilityState {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Static => "static",
            Self::Ready => "ready",
            Self::Unobserved => "unobserved",
            Self::Unavailable => "unavailable",
            Self::NotBuilt => "not_built",
            Self::Unsupported => "unsupported",
            Self::NotApplicable => "not_applicable",
        }
    }

    pub const fn is_ready(self) -> bool {
        matches!(self, Self::Static | Self::Ready)
    }

    /// Returns true when observation proved that the selected implementation
    /// cannot execute the committed workload.
    pub const fn is_known_unready(self) -> bool {
        matches!(self, Self::Unavailable | Self::NotBuilt | Self::Unsupported)
    }
}

impl From<RuntimeComponentCapabilityStatus> for RuntimeComponentCapabilityState {
    fn from(status: RuntimeComponentCapabilityStatus) -> Self {
        match status {
            RuntimeComponentCapabilityStatus::Ready => Self::Ready,
            RuntimeComponentCapabilityStatus::Unavailable => Self::Unavailable,
            RuntimeComponentCapabilityStatus::NotBuilt => Self::NotBuilt,
            RuntimeComponentCapabilityStatus::Unsupported => Self::Unsupported,
        }
    }
}

/// Observe local component kernels once and commit the complete replay contract.
pub fn observe_runtime_component_capabilities(
    request: RuntimeComponentCapabilityObservationRequest,
) -> Result<RuntimeComponentCapabilityObservationPayload, RuntimeComponentCapabilityObservationError>
{
    let request = request.canonicalized()?;
    let capabilities = request
        .component_workloads
        .iter()
        .cloned()
        .map(|workload| {
            let backend = request.policy.backend_for(workload.component());
            observe_component_capability(backend, workload)
        })
        .collect::<Vec<_>>();
    let request_sha256 = observation_digest_json(
        RUNTIME_COMPONENT_CAPABILITY_OBSERVATION_REQUEST_DIGEST_DOMAIN,
        &request,
    )?;
    let runtime_probe_output_sha256 = request.runtime_probe.output_sha256.clone();
    let mut payload = RuntimeComponentCapabilityObservationPayload {
        kind: RUNTIME_COMPONENT_CAPABILITY_OBSERVATION_KIND.to_owned(),
        contract_version: RUNTIME_COMPONENT_CAPABILITY_OBSERVATION_CONTRACT_VERSION.to_owned(),
        semantic_owner: RUNTIME_COMPONENT_CAPABILITY_OBSERVATION_SEMANTIC_OWNER.to_owned(),
        semantic_backend: RUNTIME_COMPONENT_CAPABILITY_OBSERVATION_SEMANTIC_BACKEND.to_owned(),
        request,
        runtime_probe_output_sha256,
        capabilities,
        request_sha256,
        output_sha256: String::new(),
        committed: true,
    };
    payload.output_sha256 = observation_output_digest(&payload)?;
    payload.validate()?;
    Ok(payload)
}

pub(crate) fn observe_component_capability(
    backend: RuntimeTensorBackend,
    workload: RuntimeComponentWorkload,
) -> RuntimeComponentCapabilityEvidence {
    let tensor_capability = observe_tensor_execution_capability(
        tensor_execution_backend(backend),
        tensor_execution_workload(&workload),
    );
    let status = match tensor_capability.status {
        TensorExecutionCapabilityStatus::Ready => RuntimeComponentCapabilityStatus::Ready,
        TensorExecutionCapabilityStatus::Unavailable => {
            RuntimeComponentCapabilityStatus::Unavailable
        }
        TensorExecutionCapabilityStatus::NotBuilt => RuntimeComponentCapabilityStatus::NotBuilt,
        TensorExecutionCapabilityStatus::Unsupported => {
            RuntimeComponentCapabilityStatus::Unsupported
        }
    };
    let ready_proof = tensor_capability.ready_proof.map(|proof| match proof {
        TensorExecutionReadyProof::StaticHostContract => {
            RuntimeComponentReadyProof::StaticHostContract
        }
        TensorExecutionReadyProof::RuntimeDispatchSentinel => {
            RuntimeComponentReadyProof::RuntimeDispatchSentinel
        }
    });
    RuntimeComponentCapabilityEvidence {
        workload,
        backend,
        status,
        ready_proof,
    }
}

fn tensor_execution_backend(backend: RuntimeTensorBackend) -> TensorExecutionBackend {
    match backend {
        RuntimeTensorBackend::Auto => TensorExecutionBackend::Auto,
        RuntimeTensorBackend::Cpu => TensorExecutionBackend::Cpu,
        RuntimeTensorBackend::Faer => TensorExecutionBackend::Faer,
        RuntimeTensorBackend::CpuSimd => TensorExecutionBackend::CpuSimd,
        RuntimeTensorBackend::Naive => TensorExecutionBackend::Naive,
        RuntimeTensorBackend::Wgpu => TensorExecutionBackend::Wgpu,
        RuntimeTensorBackend::Hip => TensorExecutionBackend::Hip,
    }
}

pub(crate) fn tensor_execution_workload(
    workload: &RuntimeComponentWorkload,
) -> TensorExecutionWorkload {
    match workload {
        RuntimeComponentWorkload::DenseMatmul { rows, inner, cols } => {
            TensorExecutionWorkload::DenseMatmul {
                rows: *rows,
                inner: *inner,
                cols: *cols,
            }
        }
        RuntimeComponentWorkload::PrepackedMatmul {
            rows,
            inner,
            cols,
            bias,
        } => TensorExecutionWorkload::PrepackedMatmul {
            rows: *rows,
            inner: *inner,
            cols: *cols,
            bias: *bias,
        },
        RuntimeComponentWorkload::LayerNorm { rows, cols } => TensorExecutionWorkload::LayerNorm {
            rows: *rows,
            cols: *cols,
        },
        RuntimeComponentWorkload::Attention {
            contexts,
            sequence,
            head_dim,
            z_bias,
            attn_bias,
        } => TensorExecutionWorkload::Attention {
            contexts: *contexts,
            sequence: *sequence,
            head_dim: *head_dim,
            z_bias: *z_bias,
            attn_bias: *attn_bias,
        },
        RuntimeComponentWorkload::Softmax { rows, cols } => TensorExecutionWorkload::Softmax {
            rows: *rows,
            cols: *cols,
        },
        RuntimeComponentWorkload::TensorUtil {
            operation,
            rows,
            cols,
        } => TensorExecutionWorkload::TensorUtil {
            operation: match operation {
                RuntimeTensorUtilOperation::Scale => TensorUtilOperation::Scale,
                RuntimeTensorUtilOperation::MaxAxis0 => TensorUtilOperation::MaxAxis0,
                RuntimeTensorUtilOperation::MaxAxis0Backward => {
                    TensorUtilOperation::MaxAxis0Backward
                }
            },
            rows: *rows,
            cols: *cols,
        },
    }
}

pub(crate) fn canonicalize_component_workloads(
    workloads: &mut [RuntimeComponentWorkload],
) -> Result<(), String> {
    if workloads.len() > TensorExecutionWorkloadKey::COUNT {
        return Err(format!(
            "contains {} entries, exceeding the {} canonical operation kinds",
            workloads.len(),
            TensorExecutionWorkloadKey::COUNT
        ));
    }
    for workload in workloads.iter() {
        workload.validate()?;
    }
    workloads.sort_by_key(|workload| tensor_execution_workload(workload).key());
    if let Some(duplicate) = workloads.windows(2).find(|pair| {
        tensor_execution_workload(&pair[0]).key() == tensor_execution_workload(&pair[1]).key()
    }) {
        return Err(format!(
            "contains duplicate '{}' operation workloads",
            tensor_execution_workload(&duplicate[0]).key().as_str()
        ));
    }
    Ok(())
}

fn canonicalize_component_capabilities(
    capabilities: &mut [RuntimeComponentCapabilityEvidence],
    request: &RuntimeComponentCapabilityObservationRequest,
) -> Result<(), RuntimeComponentCapabilityObservationError> {
    if capabilities.len() != request.component_workloads.len() {
        return Err(invalid_observation_payload(
            "capabilities",
            format!(
                "contains {} observations for {} committed workloads",
                capabilities.len(),
                request.component_workloads.len()
            ),
        ));
    }
    for evidence in capabilities.iter() {
        evidence
            .validate()
            .map_err(|message| invalid_observation_payload("capabilities", message))?;
    }
    capabilities.sort_by_key(|evidence| tensor_execution_workload(&evidence.workload).key());
    if let Some(duplicate) = capabilities.windows(2).find(|pair| {
        tensor_execution_workload(&pair[0].workload).key()
            == tensor_execution_workload(&pair[1].workload).key()
    }) {
        return Err(invalid_observation_payload(
            "capabilities",
            format!(
                "contains duplicate '{}' operation observations",
                tensor_execution_workload(&duplicate[0].workload)
                    .key()
                    .as_str()
            ),
        ));
    }
    for (workload, evidence) in request.component_workloads.iter().zip(capabilities.iter()) {
        if workload != &evidence.workload {
            return Err(invalid_observation_payload(
                "capabilities",
                format!(
                    "observation for '{}' does not match its canonical workload",
                    evidence.component().as_str()
                ),
            ));
        }
        let expected_backend = request.policy.backend_for(evidence.component());
        if evidence.backend != expected_backend {
            return Err(invalid_observation_payload(
                "capabilities",
                format!(
                    "observation for '{}' uses backend '{}', expected '{}'",
                    evidence.component().as_str(),
                    evidence.backend.as_str(),
                    expected_backend.as_str()
                ),
            ));
        }
        if expected_backend == RuntimeTensorBackend::Auto
            && evidence.status != RuntimeComponentCapabilityStatus::Unsupported
        {
            return Err(invalid_observation_payload(
                "capabilities",
                format!(
                    "automatic route observation for '{}' must be unsupported",
                    evidence.component().as_str()
                ),
            ));
        }
    }
    Ok(())
}

fn invalid_observation_request(
    field: &'static str,
    message: impl Into<String>,
) -> RuntimeComponentCapabilityObservationError {
    RuntimeComponentCapabilityObservationError::InvalidRequest {
        field,
        message: message.into(),
    }
}

fn invalid_observation_payload(
    field: &'static str,
    message: impl Into<String>,
) -> RuntimeComponentCapabilityObservationError {
    RuntimeComponentCapabilityObservationError::InvalidPayload {
        field,
        message: message.into(),
    }
}

fn observation_digest_json<T: Serialize>(
    domain: &[u8],
    value: &T,
) -> Result<String, RuntimeComponentCapabilityObservationError> {
    let encoded = serde_json::to_vec(value).map_err(|error| {
        RuntimeComponentCapabilityObservationError::Encoding {
            message: error.to_string(),
        }
    })?;
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.update(encoded);
    Ok(format!("{:x}", digest.finalize()))
}

fn observation_output_digest(
    payload: &RuntimeComponentCapabilityObservationPayload,
) -> Result<String, RuntimeComponentCapabilityObservationError> {
    let mut canonical = payload.clone();
    canonical.output_sha256.clear();
    observation_digest_json(
        RUNTIME_COMPONENT_CAPABILITY_OBSERVATION_OUTPUT_DIGEST_DOMAIN,
        &canonical,
    )
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::device_caps::{BackendKind, DeviceCaps};
    #[cfg(feature = "wgpu")]
    use crate::backend::runtime_probe::resolve_backend;
    use crate::backend::runtime_probe::{evaluate_runtime_device_probe, RuntimeDeviceProbeRequest};

    fn cpu_probe() -> RuntimeDeviceProbePayload {
        evaluate_runtime_device_probe(RuntimeDeviceProbeRequest {
            requested_backend: BackendKind::Cpu,
            caps: DeviceCaps::cpu(),
            mps_probe: None,
            requested_workgroup: None,
            cols: None,
            tile_hint: None,
            compaction_hint: None,
        })
        .expect("valid CPU probe")
    }

    fn observation_request() -> RuntimeComponentCapabilityObservationRequest {
        RuntimeComponentCapabilityObservationRequest {
            runtime_probe: cpu_probe(),
            policy: crate::backend::execution_plan::runtime_tensor_policy_for(BackendKind::Cpu),
            component_workloads: vec![
                RuntimeComponentWorkload::Softmax { rows: 2, cols: 4 },
                RuntimeComponentWorkload::DenseMatmul {
                    rows: 2,
                    inner: 3,
                    cols: 4,
                },
            ],
        }
    }

    #[test]
    fn workload_validation_rejects_zero_and_overflow() {
        let zero = RuntimeComponentWorkload::Softmax { rows: 0, cols: 8 };
        assert!(zero.validate().unwrap_err().contains("must be positive"));

        let overflow = RuntimeComponentWorkload::Attention {
            contexts: u64::MAX,
            sequence: 2,
            head_dim: 2,
            z_bias: false,
            attn_bias: false,
        };
        assert!(overflow
            .validate()
            .unwrap_err()
            .contains("exceeds u64 range"));

        let bias_overflow = RuntimeComponentWorkload::Attention {
            contexts: 1,
            sequence: u64::MAX,
            head_dim: 1,
            z_bias: false,
            attn_bias: true,
        };
        assert!(bias_overflow.validate().unwrap_err().contains("attn_bias"));
    }

    #[test]
    fn host_capability_observation_is_ready_evidence() {
        let workload = RuntimeComponentWorkload::DenseMatmul {
            rows: 2,
            inner: 3,
            cols: 4,
        };
        let evidence = observe_component_capability(RuntimeTensorBackend::Faer, workload.clone());
        assert_eq!(evidence.workload, workload);
        assert_eq!(evidence.backend, RuntimeTensorBackend::Faer);
        assert_eq!(evidence.status, RuntimeComponentCapabilityStatus::Ready);
        assert_eq!(
            evidence.ready_proof,
            Some(RuntimeComponentReadyProof::StaticHostContract)
        );
    }

    #[test]
    fn host_capability_rejects_buffers_outside_the_pointer_range() {
        let cols = (isize::MAX as u64 / std::mem::size_of::<f32>() as u64) + 1;
        let workload = RuntimeComponentWorkload::Softmax { rows: 1, cols };

        assert!(workload.validate().is_ok());
        let evidence = observe_component_capability(RuntimeTensorBackend::Cpu, workload);
        assert_eq!(
            evidence.status,
            RuntimeComponentCapabilityStatus::Unavailable
        );
        assert_eq!(evidence.ready_proof, None);
    }

    #[test]
    fn runtime_tensor_util_operations_preserve_rust_semantics_and_wire_names() {
        for (runtime_operation, tensor_operation, wire_name) in [
            (
                RuntimeTensorUtilOperation::Scale,
                TensorUtilOperation::Scale,
                "scale",
            ),
            (
                RuntimeTensorUtilOperation::MaxAxis0,
                TensorUtilOperation::MaxAxis0,
                "max_axis0",
            ),
            (
                RuntimeTensorUtilOperation::MaxAxis0Backward,
                TensorUtilOperation::MaxAxis0Backward,
                "max_axis0_backward",
            ),
        ] {
            let runtime_workload = RuntimeComponentWorkload::TensorUtil {
                operation: runtime_operation,
                rows: 3,
                cols: 2,
            };
            assert_eq!(
                tensor_execution_workload(&runtime_workload),
                TensorExecutionWorkload::TensorUtil {
                    operation: tensor_operation,
                    rows: 3,
                    cols: 2,
                }
            );
            assert_eq!(
                serde_json::to_value(&runtime_workload).unwrap()["operation"],
                wire_name
            );
        }
    }

    #[test]
    fn observation_commits_multiple_operations_for_one_component() {
        let mut request = observation_request();
        request.component_workloads = vec![
            RuntimeComponentWorkload::TensorUtil {
                operation: RuntimeTensorUtilOperation::MaxAxis0Backward,
                rows: 3,
                cols: 2,
            },
            RuntimeComponentWorkload::TensorUtil {
                operation: RuntimeTensorUtilOperation::MaxAxis0,
                rows: 3,
                cols: 2,
            },
        ];

        let observation = observe_runtime_component_capabilities(request)
            .expect("distinct operation workloads share one component observation");

        assert_eq!(observation.capabilities.len(), 2);
        assert_eq!(
            observation
                .request
                .component_workloads
                .iter()
                .map(|workload| tensor_execution_workload(workload).key())
                .collect::<Vec<_>>(),
            vec![
                TensorExecutionWorkloadKey::TensorUtilMaxAxis0,
                TensorExecutionWorkloadKey::TensorUtilMaxAxis0Backward,
            ]
        );
        assert!(observation
            .capabilities
            .iter()
            .all(|evidence| evidence.status == RuntimeComponentCapabilityStatus::Ready));
        observation.validate().expect("multi-operation observation");
    }

    #[test]
    fn observation_rejects_duplicate_operation_workloads() {
        let mut request = observation_request();
        request.component_workloads = vec![
            RuntimeComponentWorkload::TensorUtil {
                operation: RuntimeTensorUtilOperation::MaxAxis0,
                rows: 3,
                cols: 2,
            },
            RuntimeComponentWorkload::TensorUtil {
                operation: RuntimeTensorUtilOperation::MaxAxis0,
                rows: 4,
                cols: 2,
            },
        ];

        assert!(matches!(
            observe_runtime_component_capabilities(request),
            Err(RuntimeComponentCapabilityObservationError::InvalidRequest {
                field: "component_workloads",
                ..
            })
        ));
    }

    #[test]
    fn incompatible_host_backend_cannot_report_ready() {
        let workload = RuntimeComponentWorkload::Softmax { rows: 2, cols: 4 };

        for backend in [
            RuntimeTensorBackend::CpuSimd,
            RuntimeTensorBackend::Naive,
            RuntimeTensorBackend::Faer,
            RuntimeTensorBackend::Hip,
        ] {
            let evidence = observe_component_capability(backend, workload.clone());
            assert_eq!(
                evidence.status,
                RuntimeComponentCapabilityStatus::Unsupported
            );
            assert_eq!(evidence.ready_proof, None);
        }
    }

    #[test]
    fn observation_contract_commits_probe_policy_workloads_and_evidence() {
        let payload = observe_runtime_component_capabilities(observation_request())
            .expect("capability observation");

        payload.validate().expect("valid committed observation");
        assert_eq!(payload.kind, RUNTIME_COMPONENT_CAPABILITY_OBSERVATION_KIND);
        assert_eq!(
            payload.contract_version,
            RUNTIME_COMPONENT_CAPABILITY_OBSERVATION_CONTRACT_VERSION
        );
        assert_eq!(
            payload.semantic_owner,
            RUNTIME_COMPONENT_CAPABILITY_OBSERVATION_SEMANTIC_OWNER
        );
        assert!(payload.committed);
        assert_eq!(payload.request_sha256.len(), 64);
        assert_eq!(payload.output_sha256.len(), 64);
        assert_eq!(
            payload.runtime_probe_output_sha256,
            payload.request.runtime_probe.output_sha256
        );
        assert_eq!(
            payload
                .request
                .component_workloads
                .iter()
                .map(RuntimeComponentWorkload::component)
                .collect::<Vec<_>>(),
            vec![
                RuntimeExecutionComponent::DenseMatmul,
                RuntimeExecutionComponent::Softmax,
            ]
        );
        assert!(payload
            .capabilities
            .iter()
            .all(|evidence| evidence.status == RuntimeComponentCapabilityStatus::Ready));
        assert!(payload.capabilities.iter().all(|evidence| {
            evidence.ready_proof == Some(RuntimeComponentReadyProof::StaticHostContract)
        }));
    }

    #[test]
    fn observation_strips_nested_transport_provenance() {
        let mut request = observation_request();
        request.runtime_probe = request
            .runtime_probe
            .with_execution_client("Python")
            .expect("transport provenance");

        let payload = observe_runtime_component_capabilities(request)
            .expect("capability observation strips provenance");

        assert!(payload.request.runtime_probe.execution_client.is_none());
        payload.validate().expect("canonical observation");
    }

    #[test]
    fn observation_rejects_capability_status_and_backend_tampering() {
        let payload = observe_runtime_component_capabilities(observation_request())
            .expect("capability observation");

        let mut status_tampered = payload.clone();
        status_tampered.capabilities[0].status = RuntimeComponentCapabilityStatus::NotBuilt;
        assert!(matches!(
            status_tampered.validate(),
            Err(RuntimeComponentCapabilityObservationError::InvalidPayload {
                field: "capabilities",
                ..
            })
        ));

        let mut proof_tampered = payload.clone();
        proof_tampered.capabilities[0].ready_proof =
            Some(RuntimeComponentReadyProof::RuntimeDispatchSentinel);
        assert!(matches!(
            proof_tampered.validate(),
            Err(RuntimeComponentCapabilityObservationError::InvalidPayload {
                field: "capabilities",
                ..
            })
        ));

        let mut backend_tampered = payload;
        backend_tampered.capabilities[0].backend = RuntimeTensorBackend::Wgpu;
        assert!(matches!(
            backend_tampered.validate(),
            Err(RuntimeComponentCapabilityObservationError::InvalidPayload {
                field: "capabilities",
                ..
            })
        ));
    }

    #[test]
    fn observation_replay_rejects_different_workloads() {
        let payload = observe_runtime_component_capabilities(observation_request())
            .expect("capability observation");
        let mut replay = observation_request();
        replay.component_workloads[0] = RuntimeComponentWorkload::Softmax { rows: 3, cols: 4 };

        assert!(matches!(
            payload.validate_against(replay),
            Err(RuntimeComponentCapabilityObservationError::InvalidPayload {
                field: "request",
                ..
            })
        ));
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn live_wgpu_observation_commits_dispatch_proofs() {
        if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS")
            .ok()
            .as_deref()
            != Some("1")
        {
            eprintln!(
                "skipping live WGPU observation test; set SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1"
            );
            return;
        }

        let resolution = resolve_backend(BackendKind::Wgpu);
        assert_eq!(resolution.effective_backend, BackendKind::Wgpu);
        let runtime_probe = evaluate_runtime_device_probe(RuntimeDeviceProbeRequest {
            requested_backend: resolution.reported_backend,
            caps: DeviceCaps::wgpu(32, true, 256),
            mps_probe: resolution.mps_probe,
            requested_workgroup: None,
            cols: None,
            tile_hint: None,
            compaction_hint: None,
        })
        .expect("live WGPU probe");
        let request = RuntimeComponentCapabilityObservationRequest {
            runtime_probe,
            policy: crate::backend::execution_plan::runtime_tensor_policy_for(BackendKind::Wgpu),
            component_workloads: vec![
                RuntimeComponentWorkload::DenseMatmul {
                    rows: 2,
                    inner: 3,
                    cols: 4,
                },
                RuntimeComponentWorkload::PrepackedMatmul {
                    rows: 2,
                    inner: 3,
                    cols: 4,
                    bias: true,
                },
                RuntimeComponentWorkload::LayerNorm { rows: 2, cols: 4 },
                RuntimeComponentWorkload::Attention {
                    contexts: 1,
                    sequence: 2,
                    head_dim: 4,
                    z_bias: true,
                    attn_bias: true,
                },
                RuntimeComponentWorkload::Softmax { rows: 2, cols: 4 },
                RuntimeComponentWorkload::TensorUtil {
                    operation: RuntimeTensorUtilOperation::Scale,
                    rows: 2,
                    cols: 4,
                },
            ],
        };

        let observation =
            observe_runtime_component_capabilities(request).expect("WGPU observation");
        observation.validate().expect("committed WGPU observation");
        assert!(observation.capabilities.iter().all(|evidence| {
            evidence.status == RuntimeComponentCapabilityStatus::Ready
                && evidence.ready_proof == Some(RuntimeComponentReadyProof::RuntimeDispatchSentinel)
        }));
    }
}
