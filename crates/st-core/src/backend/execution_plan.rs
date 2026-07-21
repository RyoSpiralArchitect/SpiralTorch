//! Typed tensor execution plans shared by Rust trainers and language bindings.

use super::device_caps::{BackendKind, DeviceCaps};
use super::runtime_probe::{RuntimeDeviceProbeError, RuntimeDeviceProbePayload};
use super::runtime_route::{
    evaluate_runtime_device_route, RuntimeDeviceRouteError, RuntimeDeviceRoutePayload,
    RuntimeDeviceRouteRequest,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
pub use spiral_config::execution::{AcceleratorFallback, ExecutionConfig};
use st_tensor::{
    AttentionBackend, LayerNormBackend, MatmulBackend, SoftmaxBackend, TensorUtilBackend,
};
use thiserror::Error;

/// Stable contract identifier shared by Rust, Python, and WASM clients.
pub const RUNTIME_EXECUTION_PLAN_CONTRACT_VERSION: &str = "spiraltorch.runtime_execution_plan.v1";
/// Payload kind for committed tensor execution plans.
pub const RUNTIME_EXECUTION_PLAN_KIND: &str = "spiraltorch.runtime_execution_plan";
/// Crate/module that owns tensor execution-plan semantics.
pub const RUNTIME_EXECUTION_PLAN_SEMANTIC_OWNER: &str = "st-core::backend::execution_plan";
/// Backend label attached to payloads produced by the canonical implementation.
pub const RUNTIME_EXECUTION_PLAN_SEMANTIC_BACKEND: &str = "rust";

const RUNTIME_EXECUTION_PLAN_REQUEST_DIGEST_DOMAIN: &[u8] =
    b"spiraltorch.runtime_execution_plan.request.v1\0";
const RUNTIME_EXECUTION_PLAN_OUTPUT_DIGEST_DOMAIN: &[u8] =
    b"spiraltorch.runtime_execution_plan.output.v1\0";
const RUNTIME_EXECUTION_PLAN_MAX_CLIENT_BYTES: usize = 64;
const RUNTIME_EXECUTION_PLAN_COMPONENT_COUNT: usize = 6;

#[derive(Debug, Error, PartialEq)]
pub enum RuntimeExecutionPlanError {
    #[error(transparent)]
    RuntimeProbe(#[from] RuntimeDeviceProbeError),
    #[error(transparent)]
    RuntimeRoute(#[from] RuntimeDeviceRouteError),
    #[error("invalid runtime execution-plan request field '{field}': {message}")]
    InvalidRequest {
        field: &'static str,
        message: String,
    },
    #[error("invalid runtime execution-plan payload field '{field}': {message}")]
    InvalidPayload {
        field: &'static str,
        message: String,
    },
    #[error("runtime execution-plan encoding failed: {message}")]
    Encoding { message: String },
    #[error("runtime execution plan is blocked: {blockers:?}")]
    ExecutionBlocked { blockers: Vec<String> },
    #[error(
        "local tensor backend for component '{component}' is '{local}', but the committed plan requires '{planned}'"
    )]
    LocalBackendMismatch {
        component: &'static str,
        planned: String,
        local: String,
    },
}

/// Tensor components whose backend choices are owned by the execution plan.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeExecutionComponent {
    DenseMatmul,
    PrepackedMatmul,
    LayerNorm,
    Attention,
    Softmax,
    TensorUtil,
}

impl RuntimeExecutionComponent {
    pub const ALL: [Self; RUNTIME_EXECUTION_PLAN_COMPONENT_COUNT] = [
        Self::DenseMatmul,
        Self::PrepackedMatmul,
        Self::LayerNorm,
        Self::Attention,
        Self::Softmax,
        Self::TensorUtil,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DenseMatmul => "dense_matmul",
            Self::PrepackedMatmul => "prepacked_matmul",
            Self::LayerNorm => "layer_norm",
            Self::Attention => "attention",
            Self::Softmax => "softmax",
            Self::TensorUtil => "tensor_util",
        }
    }
}

/// Stable, feature-independent tensor backend vocabulary used on the wire.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeTensorBackend {
    Auto,
    Cpu,
    Faer,
    CpuSimd,
    Naive,
    Wgpu,
    Hip,
}

impl RuntimeTensorBackend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Faer => "faer",
            Self::CpuSimd => "cpu_simd",
            Self::Naive => "naive",
            Self::Wgpu => "wgpu",
            Self::Hip => "hip",
        }
    }
}

/// Whether a component choice is concrete, deferred, or threshold-dependent.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeComponentRouteClass {
    Direct,
    Automatic,
    Conditional,
    CpuThresholdFallback,
}

/// Stable policy choices before operation-specific threshold routing.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeTensorBackendPolicy {
    pub dense_matmul: RuntimeTensorBackend,
    pub prepacked_matmul: RuntimeTensorBackend,
    pub layer_norm: RuntimeTensorBackend,
    pub attention: RuntimeTensorBackend,
    pub softmax: RuntimeTensorBackend,
    pub tensor_util: RuntimeTensorBackend,
}

impl RuntimeTensorBackendPolicy {
    pub const fn backend_for(&self, component: RuntimeExecutionComponent) -> RuntimeTensorBackend {
        match component {
            RuntimeExecutionComponent::DenseMatmul => self.dense_matmul,
            RuntimeExecutionComponent::PrepackedMatmul => self.prepacked_matmul,
            RuntimeExecutionComponent::LayerNorm => self.layer_norm,
            RuntimeExecutionComponent::Attention => self.attention,
            RuntimeExecutionComponent::Softmax => self.softmax,
            RuntimeExecutionComponent::TensorUtil => self.tensor_util,
        }
    }
}

/// One component-level route in a committed execution plan.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeComponentRoute {
    pub component: RuntimeExecutionComponent,
    pub requested_backend: RuntimeTensorBackend,
    pub selected_backend: RuntimeTensorBackend,
    pub route: RuntimeComponentRouteClass,
    /// True only when this operation is concretely committed to the effective runtime backend.
    pub native: bool,
    pub fallback: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub values: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<usize>,
}

/// Inputs required to derive one replayable execution plan.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeExecutionPlanRequest {
    pub runtime_probe: RuntimeDeviceProbePayload,
    pub execution_config: ExecutionConfig,
    /// Representative utility-operation size used to report the threshold route.
    /// The installed policy may serve other sizes, so strict WGPU plans also
    /// require a zero threshold before they can be committed.
    #[serde(default)]
    pub tensor_util_values: Option<usize>,
    /// Components the caller requires to resolve directly on the effective backend.
    #[serde(default)]
    pub required_native_components: Vec<RuntimeExecutionComponent>,
}

impl RuntimeExecutionPlanRequest {
    fn canonicalized(mut self) -> Result<Self, RuntimeExecutionPlanError> {
        self.runtime_probe.validate()?;
        self.runtime_probe.execution_client = None;
        if self.required_native_components.len() > RUNTIME_EXECUTION_PLAN_COMPONENT_COUNT {
            return Err(RuntimeExecutionPlanError::InvalidRequest {
                field: "required_native_components",
                message: format!(
                    "contains {} entries, exceeding the {} canonical components",
                    self.required_native_components.len(),
                    RUNTIME_EXECUTION_PLAN_COMPONENT_COUNT
                ),
            });
        }
        self.required_native_components.sort_unstable();
        self.required_native_components.dedup();
        Ok(self)
    }

    fn validate_canonical(&self) -> Result<(), RuntimeExecutionPlanError> {
        let canonical = self.clone().canonicalized()?;
        if canonical != *self {
            return Err(RuntimeExecutionPlanError::InvalidRequest {
                field: "request",
                message:
                    "must use canonical component ordering and omit nested transport provenance"
                        .to_owned(),
            });
        }
        Ok(())
    }
}

/// Readiness state for the complete execution plan.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeExecutionPlanStatus {
    Ready,
    Blocked,
}

/// Rust-owned, committed execution plan consumed by every language surface.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeExecutionPlanPayload {
    pub kind: String,
    pub contract_version: String,
    pub semantic_owner: String,
    pub semantic_backend: String,
    /// Transport provenance. This field is excluded from semantic commitments.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_client: Option<String>,
    pub request: RuntimeExecutionPlanRequest,
    pub requested_backend: BackendKind,
    pub effective_backend: BackendKind,
    pub runtime_probe_output_sha256: String,
    pub runtime_route: RuntimeDeviceRoutePayload,
    pub runtime_route_output_sha256: String,
    pub policy: RuntimeTensorBackendPolicy,
    pub component_routes: Vec<RuntimeComponentRoute>,
    pub native_components: Vec<RuntimeExecutionComponent>,
    pub automatic_components: Vec<RuntimeExecutionComponent>,
    pub conditional_components: Vec<RuntimeExecutionComponent>,
    pub fallback_components: Vec<RuntimeExecutionComponent>,
    pub required_native_components_missing: Vec<RuntimeExecutionComponent>,
    pub all_components_native: bool,
    pub runtime_ready: bool,
    pub surrogate: bool,
    pub execution_allowed: bool,
    pub status: RuntimeExecutionPlanStatus,
    pub blockers: Vec<String>,
    pub request_sha256: String,
    pub output_sha256: String,
    pub committed: bool,
}

/// Tensor backend policy derived from device capabilities and captured runtime configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendPolicy {
    caps: DeviceCaps,
    config: ExecutionConfig,
    matmul_backend: MatmulBackend,
    prepacked_matmul_backend: MatmulBackend,
    layer_norm_backend: LayerNormBackend,
    attention_backend: AttentionBackend,
    softmax_backend: SoftmaxBackend,
    tensor_util_backend: TensorUtilBackend,
    runtime_plan_output_sha256: Option<[u8; 32]>,
}

impl BackendPolicy {
    /// Builds a policy from device capabilities and captures process configuration once.
    pub fn from_device_caps(caps: DeviceCaps) -> Self {
        Self::from_device_caps_with_config(caps, ExecutionConfig::from_env())
    }

    /// Builds a deterministic policy from explicit device capabilities and configuration.
    pub fn from_device_caps_with_config(caps: DeviceCaps, config: ExecutionConfig) -> Self {
        Self {
            caps,
            config,
            matmul_backend: matmul_backend_for(caps.backend),
            prepacked_matmul_backend: prepacked_matmul_backend_for(caps.backend),
            layer_norm_backend: layer_norm_backend_for(caps.backend),
            attention_backend: attention_backend_for(caps.backend),
            softmax_backend: softmax_backend_for(caps.backend),
            tensor_util_backend: tensor_util_backend_for(caps.backend),
            runtime_plan_output_sha256: None,
        }
    }

    /// Creates a policy with explicit tensor backends for focused tests and experiments.
    pub fn explicit(
        caps: DeviceCaps,
        matmul_backend: MatmulBackend,
        prepacked_matmul_backend: MatmulBackend,
        layer_norm_backend: LayerNormBackend,
        attention_backend: AttentionBackend,
        softmax_backend: SoftmaxBackend,
    ) -> Self {
        Self::explicit_with_config(
            caps,
            ExecutionConfig::from_env(),
            matmul_backend,
            prepacked_matmul_backend,
            layer_norm_backend,
            attention_backend,
            softmax_backend,
        )
    }

    /// Creates an explicit policy with a deterministic execution configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn explicit_with_config(
        caps: DeviceCaps,
        config: ExecutionConfig,
        matmul_backend: MatmulBackend,
        prepacked_matmul_backend: MatmulBackend,
        layer_norm_backend: LayerNormBackend,
        attention_backend: AttentionBackend,
        softmax_backend: SoftmaxBackend,
    ) -> Self {
        Self {
            caps,
            config,
            matmul_backend,
            prepacked_matmul_backend,
            layer_norm_backend,
            attention_backend,
            softmax_backend,
            tensor_util_backend: tensor_util_backend_for(caps.backend),
            runtime_plan_output_sha256: None,
        }
    }

    /// Materializes an executable local policy from a validated committed plan.
    ///
    /// Cross-build replay remains valid even when the receiving build lacks a
    /// planned backend. Conversion is the explicit boundary that rejects such
    /// a local feature mismatch instead of silently changing the plan.
    pub fn try_from_runtime_plan(
        plan: &RuntimeExecutionPlanPayload,
    ) -> Result<Self, RuntimeExecutionPlanError> {
        plan.validate()?;
        if !plan.execution_allowed {
            return Err(RuntimeExecutionPlanError::ExecutionBlocked {
                blockers: plan.blockers.clone(),
            });
        }
        let mut policy = Self::from_device_caps_with_config(
            plan.request.runtime_probe.caps(),
            plan.request.execution_config,
        );
        for (component, planned, local) in [
            (
                RuntimeExecutionComponent::DenseMatmul,
                plan.policy.dense_matmul.as_str(),
                policy.matmul_backend_label(),
            ),
            (
                RuntimeExecutionComponent::PrepackedMatmul,
                plan.policy.prepacked_matmul.as_str(),
                policy.prepacked_matmul_backend_label(),
            ),
            (
                RuntimeExecutionComponent::LayerNorm,
                plan.policy.layer_norm.as_str(),
                policy.layer_norm_backend_label(),
            ),
            (
                RuntimeExecutionComponent::Attention,
                plan.policy.attention.as_str(),
                policy.attention_backend_label(),
            ),
            (
                RuntimeExecutionComponent::Softmax,
                plan.policy.softmax.as_str(),
                policy.softmax_backend_label(),
            ),
            (
                RuntimeExecutionComponent::TensorUtil,
                plan.policy.tensor_util.as_str(),
                policy.tensor_util_backend_label(),
            ),
        ] {
            if planned != local {
                return Err(RuntimeExecutionPlanError::LocalBackendMismatch {
                    component: component.as_str(),
                    planned: planned.to_owned(),
                    local: local.to_owned(),
                });
            }
        }
        policy.runtime_plan_output_sha256 = Some(parse_sha256(&plan.output_sha256)?);
        Ok(policy)
    }

    pub const fn device_caps(self) -> DeviceCaps {
        self.caps
    }

    pub const fn execution_config(self) -> ExecutionConfig {
        self.config
    }

    pub const fn matmul_backend(self) -> MatmulBackend {
        self.matmul_backend
    }

    pub const fn prepacked_matmul_backend(self) -> MatmulBackend {
        self.prepacked_matmul_backend
    }

    pub const fn layer_norm_backend(self) -> LayerNormBackend {
        self.layer_norm_backend
    }

    pub const fn attention_backend(self) -> AttentionBackend {
        self.attention_backend
    }

    pub const fn softmax_backend(self) -> SoftmaxBackend {
        self.softmax_backend
    }

    pub const fn tensor_util_backend(self) -> TensorUtilBackend {
        self.tensor_util_backend
    }

    /// Commitment of the canonical runtime plan that produced this policy.
    pub const fn runtime_plan_output_sha256(self) -> Option<[u8; 32]> {
        self.runtime_plan_output_sha256
    }

    pub fn runtime_plan_output_sha256_hex(self) -> Option<String> {
        self.runtime_plan_output_sha256.map(sha256_hex)
    }

    pub const fn device_backend_label(self) -> &'static str {
        self.caps.backend.as_str()
    }

    pub fn matmul_backend_label(self) -> &'static str {
        matmul_backend_label(self.matmul_backend)
    }

    pub fn prepacked_matmul_backend_label(self) -> &'static str {
        matmul_backend_label(self.prepacked_matmul_backend)
    }

    pub fn layer_norm_backend_label(self) -> &'static str {
        layer_norm_backend_label(self.layer_norm_backend)
    }

    pub fn attention_backend_label(self) -> &'static str {
        attention_backend_label(self.attention_backend)
    }

    pub fn softmax_backend_label(self) -> &'static str {
        softmax_backend_label(self.softmax_backend)
    }

    pub fn tensor_util_backend_label(self) -> &'static str {
        tensor_util_backend_label(self.tensor_util_backend)
    }

    /// Resolves the utility-kernel route without consulting mutable global state.
    pub fn tensor_util_route(self, values: usize) -> TensorUtilRoute {
        let requested_backend = self.tensor_util_backend;
        let threshold = self.config.tensor_util_wgpu_min_values;
        let (selected_backend, status) =
            if matches!(requested_backend, TensorUtilBackend::GpuWgpu) && values < threshold {
                (TensorUtilBackend::Cpu, TensorUtilRouteStatus::CpuThreshold)
            } else if matches!(requested_backend, TensorUtilBackend::GpuWgpu) {
                (requested_backend, TensorUtilRouteStatus::Wgpu)
            } else {
                (requested_backend, TensorUtilRouteStatus::Direct)
            };

        TensorUtilRoute {
            requested_backend,
            selected_backend,
            values,
            threshold,
            status,
        }
    }
}

/// Result of applying the typed utility-kernel threshold to one tensor operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorUtilRoute {
    pub requested_backend: TensorUtilBackend,
    pub selected_backend: TensorUtilBackend,
    pub values: usize,
    pub threshold: usize,
    pub status: TensorUtilRouteStatus,
}

impl TensorUtilRoute {
    pub fn requested_backend_label(self) -> &'static str {
        tensor_util_backend_label(self.requested_backend)
    }

    pub fn selected_backend_label(self) -> &'static str {
        tensor_util_backend_label(self.selected_backend)
    }

    pub const fn records_threshold_decision(self) -> bool {
        matches!(
            self.status,
            TensorUtilRouteStatus::Wgpu | TensorUtilRouteStatus::CpuThreshold
        )
    }
}

/// Stable route outcome vocabulary shared by telemetry and language bindings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorUtilRouteStatus {
    Direct,
    Wgpu,
    CpuThreshold,
}

impl TensorUtilRouteStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Wgpu => "wgpu",
            Self::CpuThreshold => "cpu_threshold",
        }
    }
}

impl RuntimeExecutionPlanPayload {
    /// Attach transport provenance without changing the Rust-owned commitment.
    pub fn with_execution_client(
        mut self,
        execution_client: impl AsRef<str>,
    ) -> Result<Self, RuntimeExecutionPlanError> {
        self.execution_client = Some(normalized_execution_client(execution_client.as_ref())?);
        self.validate()?;
        Ok(self)
    }

    /// Validate identity, lineage, every derived projection, commitments, and replay.
    pub fn validate(&self) -> Result<(), RuntimeExecutionPlanError> {
        for (field, actual, expected) in [
            ("kind", self.kind.as_str(), RUNTIME_EXECUTION_PLAN_KIND),
            (
                "contract_version",
                self.contract_version.as_str(),
                RUNTIME_EXECUTION_PLAN_CONTRACT_VERSION,
            ),
            (
                "semantic_owner",
                self.semantic_owner.as_str(),
                RUNTIME_EXECUTION_PLAN_SEMANTIC_OWNER,
            ),
            (
                "semantic_backend",
                self.semantic_backend.as_str(),
                RUNTIME_EXECUTION_PLAN_SEMANTIC_BACKEND,
            ),
        ] {
            if actual != expected {
                return Err(invalid_payload(
                    field,
                    format!("must be '{expected}', got '{actual}'"),
                ));
            }
        }
        if let Some(execution_client) = self.execution_client.as_deref() {
            if normalized_execution_client(execution_client)? != execution_client {
                return Err(invalid_payload(
                    "execution_client",
                    "must already use its canonical lowercase label",
                ));
            }
        }
        if !self.committed {
            return Err(invalid_payload(
                "committed",
                "runtime execution plans must be committed",
            ));
        }
        if !valid_sha256(&self.request_sha256)
            || !valid_sha256(&self.output_sha256)
            || !valid_sha256(&self.runtime_probe_output_sha256)
            || !valid_sha256(&self.runtime_route_output_sha256)
        {
            return Err(invalid_payload(
                "commitment",
                "all commitment fields must be lowercase SHA-256 values",
            ));
        }
        self.request.validate_canonical()?;
        self.runtime_route.validate()?;
        if self.runtime_route.execution_client.is_some() {
            return Err(invalid_payload(
                "runtime_route.execution_client",
                "nested route provenance must be stripped from the semantic plan",
            ));
        }

        let expected = evaluate_runtime_execution_plan(self.request.clone())?;
        let mut actual = self.clone();
        actual.execution_client = None;
        if actual != expected {
            return Err(invalid_payload(
                "payload",
                "derived fields or commitments do not match canonical replay",
            ));
        }
        Ok(())
    }

    /// Validate this artifact against an explicit replay request.
    pub fn validate_against(
        &self,
        request: RuntimeExecutionPlanRequest,
    ) -> Result<(), RuntimeExecutionPlanError> {
        let request = request.canonicalized()?;
        if self.request != request {
            return Err(invalid_payload(
                "request",
                "does not match the supplied replay request",
            ));
        }
        self.validate()
    }
}

/// Derive a committed tensor execution plan from one committed runtime probe.
///
/// The runtime route is rebuilt inside Rust from the probe's exact route
/// evidence. Clients therefore cannot supply an independently interpreted
/// readiness decision alongside the observation.
pub fn evaluate_runtime_execution_plan(
    request: RuntimeExecutionPlanRequest,
) -> Result<RuntimeExecutionPlanPayload, RuntimeExecutionPlanError> {
    let request = request.canonicalized()?;
    evaluate_canonical_runtime_execution_plan(request)
}

fn evaluate_canonical_runtime_execution_plan(
    request: RuntimeExecutionPlanRequest,
) -> Result<RuntimeExecutionPlanPayload, RuntimeExecutionPlanError> {
    request.validate_canonical()?;
    let requested_backend = request.runtime_probe.requested_backend();
    let effective_backend = request.runtime_probe.effective_backend();
    let requested_label = requested_backend.as_str().to_owned();
    let runtime_route = evaluate_runtime_device_route(RuntimeDeviceRouteRequest {
        reports: vec![request.runtime_probe.route_evidence.clone()],
        requested_backends: vec![requested_label.clone()],
        required_available_backends: Vec::new(),
        required_ready_backends: vec![requested_label.clone()],
    })?;
    let route_row = runtime_route
        .routes
        .iter()
        .find(|row| row.requested_backend == requested_label)
        .ok_or_else(|| invalid_payload("runtime_route", "missing requested backend route"))?;
    if route_row.effective_backend != effective_backend.as_str() {
        return Err(invalid_payload(
            "runtime_route.effective_backend",
            "does not match the committed probe",
        ));
    }

    let policy = runtime_tensor_policy_for(effective_backend);
    let component_routes = RuntimeExecutionComponent::ALL
        .into_iter()
        .map(|component| component_route(&request, &policy, effective_backend, component))
        .collect::<Vec<_>>();
    let native_components = component_routes
        .iter()
        .filter(|route| route.native)
        .map(|route| route.component)
        .collect::<Vec<_>>();
    let automatic_components = component_routes
        .iter()
        .filter(|route| route.route == RuntimeComponentRouteClass::Automatic)
        .map(|route| route.component)
        .collect::<Vec<_>>();
    let conditional_components = component_routes
        .iter()
        .filter(|route| route.route == RuntimeComponentRouteClass::Conditional)
        .map(|route| route.component)
        .collect::<Vec<_>>();
    let fallback_components = component_routes
        .iter()
        .filter(|route| route.fallback)
        .map(|route| route.component)
        .collect::<Vec<_>>();
    let required_native_components_missing = request
        .required_native_components
        .iter()
        .copied()
        .filter(|required| {
            component_routes
                .iter()
                .find(|route| route.component == *required)
                .is_none_or(|route| !route.native)
        })
        .collect::<Vec<_>>();

    let runtime_ready = route_row.route_ready && runtime_route.passed;
    let surrogate = route_row.fallback;
    let mut blockers = runtime_route
        .failures
        .iter()
        .map(|failure| format!("runtime_route:{failure}"))
        .collect::<Vec<_>>();
    if !runtime_ready && blockers.is_empty() {
        blockers.push(format!("runtime_route_not_ready:{requested_label}"));
    }
    if surrogate && request.execution_config.accelerator_fallback.is_strict() {
        blockers.push(format!(
            "surrogate_forbidden:{}->{}",
            requested_backend.as_str(),
            effective_backend.as_str()
        ));
    }
    if request.execution_config.accelerator_fallback.is_strict() {
        if policy.tensor_util == RuntimeTensorBackend::Wgpu
            && request.execution_config.tensor_util_wgpu_min_values > 0
        {
            blockers.push(format!(
                "conditional_policy_forbidden:tensor_util_threshold:{}",
                request.execution_config.tensor_util_wgpu_min_values
            ));
        }
        blockers.extend(
            component_routes
                .iter()
                .filter(|route| route.fallback)
                .map(|route| {
                    format!(
                        "component_fallback_forbidden:{}:{}->{}",
                        route.component.as_str(),
                        route.requested_backend.as_str(),
                        route.selected_backend.as_str()
                    )
                }),
        );
        blockers.extend(
            component_routes
                .iter()
                .filter(|route| route.route == RuntimeComponentRouteClass::Automatic)
                .map(|route| format!("automatic_component_forbidden:{}", route.component.as_str())),
        );
        blockers.extend(
            component_routes
                .iter()
                .filter(|route| route.route == RuntimeComponentRouteClass::Conditional)
                .map(|route| {
                    format!(
                        "conditional_component_unresolved:{}",
                        route.component.as_str()
                    )
                }),
        );
    }
    blockers.extend(
        required_native_components_missing
            .iter()
            .map(|component| format!("native_component_unavailable:{}", component.as_str())),
    );
    let execution_allowed = blockers.is_empty();
    let status = if execution_allowed {
        RuntimeExecutionPlanStatus::Ready
    } else {
        RuntimeExecutionPlanStatus::Blocked
    };
    let request_sha256 = digest_json(RUNTIME_EXECUTION_PLAN_REQUEST_DIGEST_DOMAIN, &request)?;
    let runtime_probe_output_sha256 = request.runtime_probe.output_sha256.clone();
    let runtime_route_output_sha256 = runtime_route.output_sha256.clone();
    let mut payload = RuntimeExecutionPlanPayload {
        kind: RUNTIME_EXECUTION_PLAN_KIND.to_owned(),
        contract_version: RUNTIME_EXECUTION_PLAN_CONTRACT_VERSION.to_owned(),
        semantic_owner: RUNTIME_EXECUTION_PLAN_SEMANTIC_OWNER.to_owned(),
        semantic_backend: RUNTIME_EXECUTION_PLAN_SEMANTIC_BACKEND.to_owned(),
        execution_client: None,
        request,
        requested_backend,
        effective_backend,
        runtime_probe_output_sha256,
        runtime_route,
        runtime_route_output_sha256,
        policy,
        component_routes,
        native_components,
        automatic_components,
        conditional_components,
        fallback_components,
        required_native_components_missing,
        all_components_native: false,
        runtime_ready,
        surrogate,
        execution_allowed,
        status,
        blockers,
        request_sha256,
        output_sha256: String::new(),
        committed: true,
    };
    payload.all_components_native =
        payload.native_components.len() == RUNTIME_EXECUTION_PLAN_COMPONENT_COUNT;
    payload.output_sha256 = output_digest(&payload)?;
    Ok(payload)
}

fn runtime_tensor_policy_for(backend: BackendKind) -> RuntimeTensorBackendPolicy {
    match backend {
        BackendKind::Wgpu => RuntimeTensorBackendPolicy {
            dense_matmul: RuntimeTensorBackend::Wgpu,
            prepacked_matmul: RuntimeTensorBackend::Wgpu,
            layer_norm: RuntimeTensorBackend::Wgpu,
            attention: RuntimeTensorBackend::Wgpu,
            softmax: RuntimeTensorBackend::Wgpu,
            tensor_util: RuntimeTensorBackend::Wgpu,
        },
        BackendKind::Cpu => RuntimeTensorBackendPolicy {
            dense_matmul: RuntimeTensorBackend::Faer,
            prepacked_matmul: RuntimeTensorBackend::Faer,
            layer_norm: RuntimeTensorBackend::Cpu,
            attention: RuntimeTensorBackend::Cpu,
            softmax: RuntimeTensorBackend::Cpu,
            tensor_util: RuntimeTensorBackend::Cpu,
        },
        BackendKind::Hip => RuntimeTensorBackendPolicy {
            dense_matmul: RuntimeTensorBackend::Hip,
            prepacked_matmul: RuntimeTensorBackend::Auto,
            layer_norm: RuntimeTensorBackend::Auto,
            attention: RuntimeTensorBackend::Auto,
            softmax: RuntimeTensorBackend::Auto,
            tensor_util: RuntimeTensorBackend::Auto,
        },
        BackendKind::Cuda | BackendKind::Mps => RuntimeTensorBackendPolicy {
            dense_matmul: RuntimeTensorBackend::Auto,
            prepacked_matmul: RuntimeTensorBackend::Auto,
            layer_norm: RuntimeTensorBackend::Auto,
            attention: RuntimeTensorBackend::Auto,
            softmax: RuntimeTensorBackend::Auto,
            tensor_util: RuntimeTensorBackend::Auto,
        },
    }
}

fn component_route(
    request: &RuntimeExecutionPlanRequest,
    policy: &RuntimeTensorBackendPolicy,
    effective_backend: BackendKind,
    component: RuntimeExecutionComponent,
) -> RuntimeComponentRoute {
    let requested_backend = policy.backend_for(component);
    let (selected_backend, route, values, threshold) = if component
        == RuntimeExecutionComponent::TensorUtil
        && requested_backend == RuntimeTensorBackend::Wgpu
    {
        let threshold = request.execution_config.tensor_util_wgpu_min_values;
        match request.tensor_util_values {
            Some(values) if values < threshold => (
                RuntimeTensorBackend::Cpu,
                RuntimeComponentRouteClass::CpuThresholdFallback,
                Some(values),
                Some(threshold),
            ),
            Some(values) => (
                RuntimeTensorBackend::Wgpu,
                RuntimeComponentRouteClass::Direct,
                Some(values),
                Some(threshold),
            ),
            None => (
                RuntimeTensorBackend::Wgpu,
                RuntimeComponentRouteClass::Conditional,
                None,
                Some(threshold),
            ),
        }
    } else if requested_backend == RuntimeTensorBackend::Auto {
        (
            requested_backend,
            RuntimeComponentRouteClass::Automatic,
            None,
            None,
        )
    } else {
        (
            requested_backend,
            RuntimeComponentRouteClass::Direct,
            None,
            None,
        )
    };
    let native = route == RuntimeComponentRouteClass::Direct
        && tensor_backend_is_native(selected_backend, effective_backend);
    let fallback = route == RuntimeComponentRouteClass::CpuThresholdFallback;
    RuntimeComponentRoute {
        component,
        requested_backend,
        selected_backend,
        route,
        native,
        fallback,
        values,
        threshold,
    }
}

fn tensor_backend_is_native(backend: RuntimeTensorBackend, effective: BackendKind) -> bool {
    match effective {
        BackendKind::Wgpu => backend == RuntimeTensorBackend::Wgpu,
        BackendKind::Hip => backend == RuntimeTensorBackend::Hip,
        BackendKind::Cpu => matches!(
            backend,
            RuntimeTensorBackend::Cpu
                | RuntimeTensorBackend::Faer
                | RuntimeTensorBackend::CpuSimd
                | RuntimeTensorBackend::Naive
        ),
        BackendKind::Cuda | BackendKind::Mps => false,
    }
}

fn normalized_execution_client(value: &str) -> Result<String, RuntimeExecutionPlanError> {
    let normalized = value.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return Err(invalid_payload("execution_client", "must not be empty"));
    }
    if normalized.len() > RUNTIME_EXECUTION_PLAN_MAX_CLIENT_BYTES {
        return Err(invalid_payload(
            "execution_client",
            format!(
                "has {} bytes, exceeding limit {}",
                normalized.len(),
                RUNTIME_EXECUTION_PLAN_MAX_CLIENT_BYTES
            ),
        ));
    }
    if let Some((position, byte)) = normalized.bytes().enumerate().find(|(_, byte)| {
        !byte.is_ascii_lowercase() && !byte.is_ascii_digit() && !b"._-".contains(byte)
    }) {
        return Err(invalid_payload(
            "execution_client",
            format!("contains unsupported byte {byte} at position {position}"),
        ));
    }
    Ok(normalized)
}

fn invalid_payload(field: &'static str, message: impl Into<String>) -> RuntimeExecutionPlanError {
    RuntimeExecutionPlanError::InvalidPayload {
        field,
        message: message.into(),
    }
}

fn digest_json<T: Serialize>(
    domain: &[u8],
    value: &T,
) -> Result<String, RuntimeExecutionPlanError> {
    let encoded =
        serde_json::to_vec(value).map_err(|error| RuntimeExecutionPlanError::Encoding {
            message: error.to_string(),
        })?;
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.update(encoded);
    Ok(format!("{:x}", digest.finalize()))
}

fn output_digest(
    payload: &RuntimeExecutionPlanPayload,
) -> Result<String, RuntimeExecutionPlanError> {
    let mut semantic = payload.clone();
    semantic.execution_client = None;
    semantic.output_sha256.clear();
    digest_json(RUNTIME_EXECUTION_PLAN_OUTPUT_DIGEST_DOMAIN, &semantic)
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn parse_sha256(value: &str) -> Result<[u8; 32], RuntimeExecutionPlanError> {
    if !valid_sha256(value) {
        return Err(invalid_payload(
            "output_sha256",
            "must be a lowercase SHA-256 value",
        ));
    }
    let mut bytes = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        bytes[index] = (hex_nibble(pair[0]) << 4) | hex_nibble(pair[1]);
    }
    Ok(bytes)
}

fn hex_nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        _ => unreachable!("validated lowercase hex"),
    }
}

fn sha256_hex(bytes: [u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn matmul_backend_for(kind: BackendKind) -> MatmulBackend {
    match kind {
        BackendKind::Wgpu => wgpu_matmul_backend(),
        BackendKind::Hip => hip_matmul_backend(),
        BackendKind::Cpu => MatmulBackend::CpuFaer,
        BackendKind::Cuda | BackendKind::Mps => MatmulBackend::Auto,
    }
}

fn prepacked_matmul_backend_for(kind: BackendKind) -> MatmulBackend {
    match kind {
        BackendKind::Wgpu => wgpu_matmul_backend(),
        BackendKind::Cpu => MatmulBackend::CpuFaer,
        BackendKind::Cuda | BackendKind::Hip | BackendKind::Mps => MatmulBackend::Auto,
    }
}

fn layer_norm_backend_for(kind: BackendKind) -> LayerNormBackend {
    match kind {
        BackendKind::Wgpu => wgpu_layer_norm_backend(),
        BackendKind::Cpu => LayerNormBackend::Cpu,
        BackendKind::Cuda | BackendKind::Hip | BackendKind::Mps => LayerNormBackend::Auto,
    }
}

fn attention_backend_for(kind: BackendKind) -> AttentionBackend {
    match kind {
        BackendKind::Wgpu => wgpu_attention_backend(),
        BackendKind::Cpu => AttentionBackend::Cpu,
        BackendKind::Cuda | BackendKind::Hip | BackendKind::Mps => AttentionBackend::Auto,
    }
}

fn softmax_backend_for(kind: BackendKind) -> SoftmaxBackend {
    match kind {
        BackendKind::Wgpu => wgpu_softmax_backend(),
        BackendKind::Cpu => SoftmaxBackend::Cpu,
        BackendKind::Cuda | BackendKind::Hip | BackendKind::Mps => SoftmaxBackend::Auto,
    }
}

fn tensor_util_backend_for(kind: BackendKind) -> TensorUtilBackend {
    match kind {
        BackendKind::Wgpu => TensorUtilBackend::GpuWgpu,
        BackendKind::Cpu => TensorUtilBackend::Cpu,
        BackendKind::Cuda | BackendKind::Hip | BackendKind::Mps => TensorUtilBackend::Auto,
    }
}

fn matmul_backend_label(backend: MatmulBackend) -> &'static str {
    match backend {
        MatmulBackend::Auto => "auto",
        MatmulBackend::CpuFaer => "faer",
        MatmulBackend::CpuSimd => "cpu_simd",
        MatmulBackend::CpuNaive => "naive",
        #[cfg(feature = "wgpu")]
        MatmulBackend::GpuWgpu => "wgpu",
        #[cfg(feature = "hip")]
        MatmulBackend::GpuHip => "hip",
        #[allow(unreachable_patterns)]
        _ => "gpu",
    }
}

fn layer_norm_backend_label(backend: LayerNormBackend) -> &'static str {
    match backend {
        LayerNormBackend::Auto => "auto",
        LayerNormBackend::Cpu => "cpu",
        LayerNormBackend::GpuWgpu => "wgpu",
    }
}

fn attention_backend_label(backend: AttentionBackend) -> &'static str {
    match backend {
        AttentionBackend::Auto => "auto",
        AttentionBackend::Cpu => "cpu",
        AttentionBackend::GpuWgpu => "wgpu",
    }
}

fn softmax_backend_label(backend: SoftmaxBackend) -> &'static str {
    match backend {
        SoftmaxBackend::Auto => "auto",
        SoftmaxBackend::Cpu => "cpu",
        #[cfg(feature = "wgpu")]
        SoftmaxBackend::GpuWgpu => "wgpu",
        #[allow(unreachable_patterns)]
        _ => "gpu",
    }
}

fn tensor_util_backend_label(backend: TensorUtilBackend) -> &'static str {
    match backend {
        TensorUtilBackend::Auto => "auto",
        TensorUtilBackend::Cpu => "cpu",
        TensorUtilBackend::GpuWgpu => "wgpu",
    }
}

fn wgpu_matmul_backend() -> MatmulBackend {
    #[cfg(feature = "wgpu")]
    {
        MatmulBackend::GpuWgpu
    }
    #[cfg(not(feature = "wgpu"))]
    {
        MatmulBackend::Auto
    }
}

fn hip_matmul_backend() -> MatmulBackend {
    #[cfg(feature = "hip-real")]
    {
        MatmulBackend::GpuHip
    }
    #[cfg(not(feature = "hip-real"))]
    {
        MatmulBackend::Auto
    }
}

fn wgpu_layer_norm_backend() -> LayerNormBackend {
    #[cfg(feature = "wgpu")]
    {
        LayerNormBackend::GpuWgpu
    }
    #[cfg(not(feature = "wgpu"))]
    {
        LayerNormBackend::Auto
    }
}

fn wgpu_attention_backend() -> AttentionBackend {
    #[cfg(feature = "wgpu")]
    {
        AttentionBackend::GpuWgpu
    }
    #[cfg(not(feature = "wgpu"))]
    {
        AttentionBackend::Auto
    }
}

fn wgpu_softmax_backend() -> SoftmaxBackend {
    #[cfg(feature = "wgpu")]
    {
        SoftmaxBackend::GpuWgpu
    }
    #[cfg(not(feature = "wgpu"))]
    {
        SoftmaxBackend::Auto
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::runtime_probe::{
        evaluate_runtime_device_probe, resolve_backend, RuntimeDeviceProbeRequest,
    };
    use spiral_config::execution::{AcceleratorFallback, ExecutionConfig};

    fn probe_for(backend: BackendKind) -> RuntimeDeviceProbePayload {
        let resolution = resolve_backend(backend);
        evaluate_runtime_device_probe(RuntimeDeviceProbeRequest {
            requested_backend: resolution.reported_backend,
            caps: match resolution.effective_backend {
                BackendKind::Wgpu => DeviceCaps::wgpu(32, true, 256),
                BackendKind::Mps => DeviceCaps::mps(32, true, 256, None),
                BackendKind::Cuda => DeviceCaps::cuda(32, 1024, None),
                BackendKind::Hip => DeviceCaps::hip(64, 1024, None),
                BackendKind::Cpu => DeviceCaps::cpu(),
            },
            mps_probe: resolution.mps_probe,
            requested_workgroup: None,
            cols: None,
            tile_hint: None,
            compaction_hint: None,
        })
        .expect("valid runtime probe")
    }

    fn execution_request(
        probe: RuntimeDeviceProbePayload,
        fallback: AcceleratorFallback,
    ) -> RuntimeExecutionPlanRequest {
        RuntimeExecutionPlanRequest {
            runtime_probe: probe,
            execution_config: ExecutionConfig::new(fallback, 1024),
            tensor_util_values: Some(2048),
            required_native_components: Vec::new(),
        }
    }

    #[test]
    fn cpu_policy_keeps_accelerated_ops_on_cpu_or_auto() {
        let policy = BackendPolicy::from_device_caps_with_config(
            DeviceCaps::cpu(),
            ExecutionConfig::default(),
        );
        assert_eq!(policy.matmul_backend(), MatmulBackend::CpuFaer);
        assert_eq!(policy.prepacked_matmul_backend(), MatmulBackend::CpuFaer);
        assert_eq!(policy.layer_norm_backend(), LayerNormBackend::Cpu);
        assert_eq!(policy.attention_backend(), AttentionBackend::Cpu);
        assert_eq!(policy.softmax_backend(), SoftmaxBackend::Cpu);
        assert_eq!(policy.tensor_util_backend(), TensorUtilBackend::Cpu);
        assert_eq!(policy.device_backend_label(), "cpu");
        assert_eq!(policy.tensor_util_backend_label(), "cpu");
    }

    #[cfg(all(feature = "hip", not(feature = "hip-real")))]
    #[test]
    fn hip_stub_policy_does_not_claim_gpu_matmul() {
        let policy = BackendPolicy::from_device_caps_with_config(
            DeviceCaps::hip(64, 1024, None),
            ExecutionConfig::default(),
        );

        assert_eq!(policy.matmul_backend(), MatmulBackend::Auto);
        assert_eq!(policy.matmul_backend_label(), "auto");
    }

    #[cfg(feature = "hip-real")]
    #[test]
    fn hip_real_policy_commits_gpu_matmul() {
        let policy = BackendPolicy::from_device_caps_with_config(
            DeviceCaps::hip(64, 1024, None),
            ExecutionConfig::default(),
        );

        assert_eq!(policy.matmul_backend(), MatmulBackend::GpuHip);
        assert_eq!(policy.matmul_backend_label(), "hip");
    }

    #[test]
    fn tensor_utility_threshold_is_part_of_the_captured_plan() {
        let config = ExecutionConfig::new(AcceleratorFallback::Forbid, 1024);
        let policy =
            BackendPolicy::from_device_caps_with_config(DeviceCaps::wgpu(32, true, 256), config);

        let small = policy.tensor_util_route(8);
        assert_eq!(small.requested_backend, TensorUtilBackend::GpuWgpu);
        assert_eq!(small.selected_backend, TensorUtilBackend::Cpu);
        assert_eq!(small.status, TensorUtilRouteStatus::CpuThreshold);
        assert_eq!(small.threshold, 1024);

        let large = policy.tensor_util_route(1024);
        assert_eq!(large.selected_backend, TensorUtilBackend::GpuWgpu);
        assert_eq!(large.status, TensorUtilRouteStatus::Wgpu);
        assert_eq!(
            policy.execution_config().accelerator_fallback,
            AcceleratorFallback::Forbid
        );
    }

    #[test]
    fn cpu_runtime_plan_is_committed_replayable_and_executable() {
        let mut request =
            execution_request(probe_for(BackendKind::Cpu), AcceleratorFallback::Allow);
        request.required_native_components = vec![
            RuntimeExecutionComponent::Softmax,
            RuntimeExecutionComponent::DenseMatmul,
            RuntimeExecutionComponent::DenseMatmul,
        ];

        let payload =
            evaluate_runtime_execution_plan(request.clone()).expect("CPU execution plan evaluates");
        assert_eq!(payload.kind, RUNTIME_EXECUTION_PLAN_KIND);
        assert_eq!(
            payload.contract_version,
            RUNTIME_EXECUTION_PLAN_CONTRACT_VERSION
        );
        assert_eq!(payload.requested_backend, BackendKind::Cpu);
        assert_eq!(payload.effective_backend, BackendKind::Cpu);
        assert!(payload.runtime_ready);
        assert!(!payload.surrogate);
        assert!(payload.execution_allowed);
        assert_eq!(payload.status, RuntimeExecutionPlanStatus::Ready);
        assert!(payload.all_components_native);
        assert!(payload.automatic_components.is_empty());
        assert_eq!(
            payload.request.required_native_components,
            vec![
                RuntimeExecutionComponent::DenseMatmul,
                RuntimeExecutionComponent::Softmax
            ]
        );
        assert_eq!(payload.policy.dense_matmul, RuntimeTensorBackend::Faer);
        assert_eq!(payload.policy.softmax, RuntimeTensorBackend::Cpu);
        assert_eq!(payload.request_sha256.len(), 64);
        assert_eq!(payload.output_sha256.len(), 64);
        assert_eq!(
            payload.runtime_probe_output_sha256,
            payload.request.runtime_probe.output_sha256
        );
        assert_eq!(
            payload.runtime_route_output_sha256,
            payload.runtime_route.output_sha256
        );
        payload.validate().expect("committed payload validates");
        payload
            .validate_against(request)
            .expect("non-canonical caller request replays");

        let transported = payload
            .clone()
            .with_execution_client("Python")
            .expect("transport provenance is valid");
        assert_eq!(transported.execution_client.as_deref(), Some("python"));
        assert_eq!(transported.output_sha256, payload.output_sha256);

        let policy = BackendPolicy::try_from_runtime_plan(&transported)
            .expect("local CPU policy materializes");
        assert_eq!(policy.matmul_backend(), MatmulBackend::CpuFaer);
        assert_eq!(
            policy.runtime_plan_output_sha256_hex().as_deref(),
            Some(payload.output_sha256.as_str())
        );
    }

    #[test]
    fn nested_probe_transport_provenance_is_canonicalized_out() {
        let probe = probe_for(BackendKind::Cpu)
            .with_execution_client("wasm")
            .expect("probe transport provenance");
        let payload =
            evaluate_runtime_execution_plan(execution_request(probe, AcceleratorFallback::Allow))
                .expect("plan evaluates");

        assert!(payload.request.runtime_probe.execution_client.is_none());
        assert!(payload.runtime_route.execution_client.is_none());
        payload.validate().expect("canonical payload validates");
    }

    #[test]
    fn tensor_util_threshold_and_native_requirement_are_explicit() {
        let mut request =
            execution_request(probe_for(BackendKind::Wgpu), AcceleratorFallback::Allow);
        request.tensor_util_values = Some(8);
        request.required_native_components = vec![RuntimeExecutionComponent::TensorUtil];
        let payload = evaluate_runtime_execution_plan(request).expect("plan evaluates");
        let route = payload
            .component_routes
            .iter()
            .find(|route| route.component == RuntimeExecutionComponent::TensorUtil)
            .expect("tensor util route");

        assert_eq!(route.requested_backend, RuntimeTensorBackend::Wgpu);
        assert_eq!(route.selected_backend, RuntimeTensorBackend::Cpu);
        assert_eq!(
            route.route,
            RuntimeComponentRouteClass::CpuThresholdFallback
        );
        assert!(route.fallback);
        assert!(!route.native);
        assert_eq!(route.values, Some(8));
        assert_eq!(route.threshold, Some(1024));
        assert_eq!(
            payload.required_native_components_missing,
            vec![RuntimeExecutionComponent::TensorUtil]
        );
        assert!(!payload.execution_allowed);
        assert!(payload
            .blockers
            .contains(&"native_component_unavailable:tensor_util".to_owned()));
    }

    #[test]
    fn strict_execution_blocks_threshold_fallback_without_an_extra_native_gate() {
        let mut request =
            execution_request(probe_for(BackendKind::Wgpu), AcceleratorFallback::Forbid);
        request.tensor_util_values = Some(8);
        let payload = evaluate_runtime_execution_plan(request).expect("plan evaluates");

        assert!(!payload.execution_allowed);
        assert!(payload
            .blockers
            .contains(&"component_fallback_forbidden:tensor_util:wgpu->cpu".to_owned()));
        assert!(payload
            .blockers
            .contains(&"conditional_policy_forbidden:tensor_util_threshold:1024".to_owned()));
    }

    #[test]
    fn strict_wgpu_policy_requires_a_zero_tensor_util_threshold() {
        let mut request =
            execution_request(probe_for(BackendKind::Wgpu), AcceleratorFallback::Forbid);
        request.tensor_util_values = Some(2048);
        let blocked = evaluate_runtime_execution_plan(request.clone()).expect("plan evaluates");

        assert!(
            blocked
                .component_routes
                .iter()
                .find(|route| route.component == RuntimeExecutionComponent::TensorUtil)
                .expect("tensor util route")
                .native
        );
        assert!(!blocked.execution_allowed);
        assert!(blocked
            .blockers
            .contains(&"conditional_policy_forbidden:tensor_util_threshold:1024".to_owned()));

        request.execution_config.tensor_util_wgpu_min_values = 0;
        let threshold_free =
            evaluate_runtime_execution_plan(request).expect("threshold-free plan evaluates");
        assert!(!threshold_free.blockers.iter().any(|blocker| {
            blocker.starts_with("conditional_policy_forbidden:tensor_util_threshold:")
        }));
    }

    #[test]
    fn strict_execution_rejects_an_mps_surrogate_route() {
        let payload = evaluate_runtime_execution_plan(execution_request(
            probe_for(BackendKind::Mps),
            AcceleratorFallback::Forbid,
        ))
        .expect("blocked plan remains inspectable");

        assert!(payload.surrogate);
        assert!(!payload.execution_allowed);
        assert_eq!(payload.status, RuntimeExecutionPlanStatus::Blocked);
        assert!(payload
            .blockers
            .iter()
            .any(|blocker| blocker.starts_with("surrogate_forbidden:mps->")));
        let error = BackendPolicy::try_from_runtime_plan(&payload)
            .expect_err("blocked plan cannot be installed");
        assert!(matches!(
            error,
            RuntimeExecutionPlanError::ExecutionBlocked { .. }
        ));
    }

    #[test]
    fn execution_plan_validation_rejects_tampering() {
        let mut payload = evaluate_runtime_execution_plan(execution_request(
            probe_for(BackendKind::Cpu),
            AcceleratorFallback::Allow,
        ))
        .expect("plan evaluates");
        payload.component_routes[0].selected_backend = RuntimeTensorBackend::Auto;

        let error = payload.validate().expect_err("tampering must fail replay");
        assert!(matches!(
            error,
            RuntimeExecutionPlanError::InvalidPayload {
                field: "payload",
                ..
            }
        ));
    }
}
