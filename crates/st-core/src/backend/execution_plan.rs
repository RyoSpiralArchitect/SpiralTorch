//! Typed tensor execution plans shared by Rust trainers and language bindings.

use super::device_caps::{BackendKind, DeviceCaps};
use super::execution_capability::{
    canonicalize_component_workloads, observe_runtime_component_capabilities,
    tensor_execution_workload,
};
pub use super::execution_capability::{
    RuntimeComponentCapabilityEvidence, RuntimeComponentCapabilityObservationError,
    RuntimeComponentCapabilityObservationPayload, RuntimeComponentCapabilityObservationRequest,
    RuntimeComponentCapabilityState, RuntimeComponentCapabilityStatus, RuntimeComponentReadyProof,
    RuntimeComponentWorkload, RuntimeTensorUtilOperation,
};
use super::runtime_probe::{
    BackendRuntimeState, RuntimeDeviceProbeError, RuntimeDeviceProbePayload,
};
use super::runtime_route::{
    evaluate_runtime_device_route_from_probes, RuntimeDeviceRouteError, RuntimeDeviceRoutePayload,
    RuntimeDeviceRouteProbeRequest,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
pub use spiral_config::execution::{AcceleratorFallback, ExecutionConfig};
use st_tensor::{
    AttentionBackend, LayerNormBackend, MatmulBackend, SoftmaxBackend, TensorExecutionBackend,
    TensorExecutionContractError, TensorExecutionPlanBinding, TensorExecutionReceipt,
    TensorExecutionWorkload, TensorExecutionWorkloadKey, TensorUtilBackend,
};
use thiserror::Error;

/// Stable contract identifier shared by Rust, Python, and WASM clients.
pub const RUNTIME_EXECUTION_PLAN_CONTRACT_VERSION: &str = "spiraltorch.runtime_execution_plan.v8";
/// Payload kind for committed tensor execution plans.
pub const RUNTIME_EXECUTION_PLAN_KIND: &str = "spiraltorch.runtime_execution_plan";
/// Crate/module that owns tensor execution-plan semantics.
pub const RUNTIME_EXECUTION_PLAN_SEMANTIC_OWNER: &str = "st-core::backend::execution_plan";
/// Backend label attached to payloads produced by the canonical implementation.
pub const RUNTIME_EXECUTION_PLAN_SEMANTIC_BACKEND: &str = "rust";

const RUNTIME_EXECUTION_PLAN_REQUEST_DIGEST_DOMAIN: &[u8] =
    b"spiraltorch.runtime_execution_plan.request.v8\0";
const RUNTIME_EXECUTION_PLAN_OUTPUT_DIGEST_DOMAIN: &[u8] =
    b"spiraltorch.runtime_execution_plan.output.v8\0";
const RUNTIME_EXECUTION_PLAN_MAX_CLIENT_BYTES: usize = 64;
pub(crate) const RUNTIME_EXECUTION_PLAN_COMPONENT_COUNT: usize = 6;

#[derive(Debug, Error, PartialEq)]
pub enum RuntimeExecutionPlanError {
    #[error(transparent)]
    RuntimeProbe(#[from] RuntimeDeviceProbeError),
    #[error(transparent)]
    RuntimeRoute(#[from] RuntimeDeviceRouteError),
    #[error(transparent)]
    ComponentCapabilityObservation(#[from] RuntimeComponentCapabilityObservationError),
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
    #[error(
        "local runtime for committed backend '{backend}' is not ready ({status}): {diagnostic}"
    )]
    LocalRuntimeUnavailable {
        backend: String,
        status: String,
        diagnostic: String,
    },
    #[error(
        "local tensor backend for component '{component}' emitted unsupported execution id '{backend}'"
    )]
    UnsupportedLocalBackend {
        component: &'static str,
        backend: String,
    },
}

/// Failure to prove that a tensor receipt was authorized by a runtime plan.
#[derive(Debug, Error, PartialEq)]
pub enum RuntimeExecutionReceiptValidationError {
    #[error(transparent)]
    RuntimePlan(#[from] RuntimeExecutionPlanError),
    #[error(transparent)]
    TensorExecution(#[from] TensorExecutionContractError),
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

/// Whether a plan must prove concrete component capabilities before materialization.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeComponentResolution {
    /// Require workload-specific capability evidence for strict accelerator plans.
    #[default]
    Concrete,
    /// Commit the backend policy now and enforce unobserved capabilities at operation time.
    Deferred,
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
    /// Operation workloads whose mutable accelerator capabilities were observed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub workloads: Vec<RuntimeComponentWorkload>,
    /// Whether the selected implementation is static, observed, or unavailable.
    pub capability_state: RuntimeComponentCapabilityState,
    /// True only when every declared workload is ready on the effective runtime backend.
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
    /// Controls whether unobserved direct component routes may be resolved at operation time.
    #[serde(default)]
    pub component_resolution: RuntimeComponentResolution,
    /// Concrete operation shapes whose native accelerator kernels should be observed.
    #[serde(default)]
    pub component_workloads: Vec<RuntimeComponentWorkload>,
    /// Rust-owned committed observation for every declared component workload.
    #[serde(default)]
    pub component_capability_observation: Option<RuntimeComponentCapabilityObservationPayload>,
    /// Utility-operation size used to resolve the threshold route.
    /// A tensor-util workload supplies this value canonically from `rows * cols`.
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
        canonicalize_component_workloads(&mut self.component_workloads)
            .map_err(|message| invalid_request("component_workloads", message))?;
        canonicalize_tensor_util_values(&mut self)?;
        validate_component_capability_observation(&self)?;
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
                    "must use canonical operation ordering and omit nested transport provenance"
                        .to_owned(),
            });
        }
        Ok(())
    }
}

fn validate_component_capability_observation(
    request: &RuntimeExecutionPlanRequest,
) -> Result<(), RuntimeExecutionPlanError> {
    let Some(observation) = request.component_capability_observation.as_ref() else {
        return Ok(());
    };
    observation.validate_against(RuntimeComponentCapabilityObservationRequest {
        runtime_probe: request.runtime_probe.clone(),
        policy: runtime_tensor_policy_for(request.runtime_probe.effective_backend()),
        component_workloads: request.component_workloads.clone(),
    })?;
    Ok(())
}

fn canonicalize_tensor_util_values(
    request: &mut RuntimeExecutionPlanRequest,
) -> Result<(), RuntimeExecutionPlanError> {
    let mut volumes = request
        .component_workloads
        .iter()
        .filter_map(|workload| match workload {
            RuntimeComponentWorkload::TensorUtil { rows, cols, .. } => {
                Some(rows.checked_mul(*cols).ok_or_else(|| {
                    invalid_request(
                        "component_workloads",
                        "tensor_util workload volume exceeds u64 range",
                    )
                }))
            }
            _ => None,
        });
    let Some(values_u64) = volumes.next() else {
        return Ok(());
    };
    let values_u64 = values_u64?;
    for volume in volumes {
        let volume = volume?;
        if volume != values_u64 {
            return Err(invalid_request(
                "component_workloads",
                format!(
                    "tensor_util operation workloads must share one route volume; got {values_u64} and {volume}"
                ),
            ));
        }
    }
    let values = usize::try_from(values_u64).map_err(|_| {
        invalid_request(
            "component_workloads",
            "tensor_util workload volume exceeds this target's usize range",
        )
    })?;
    match request.tensor_util_values {
        None => request.tensor_util_values = Some(values),
        Some(actual) if actual == values => {}
        Some(actual) => {
            return Err(invalid_request(
                "tensor_util_values",
                format!("must match the tensor_util workload volume {values}, got {actual}"),
            ));
        }
    }
    Ok(())
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
    pub component_capability_observation_output_sha256: Option<String>,
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

/// Validated execution-plan provenance for planning without a local tensor executor.
///
/// This context can derive audit plans, but it cannot authorize tensor execution.
/// Executing clients must materialize a [`BackendPolicy`] instead.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeExecutionPlanningContext {
    device_caps: DeviceCaps,
    execution_config: ExecutionConfig,
    runtime_execution_plan_output_sha256: String,
}

impl RuntimeExecutionPlanningContext {
    pub const fn device_caps(&self) -> DeviceCaps {
        self.device_caps
    }

    pub const fn execution_config(&self) -> ExecutionConfig {
        self.execution_config
    }

    pub fn runtime_execution_plan_output_sha256(&self) -> &str {
        &self.runtime_execution_plan_output_sha256
    }
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
    runtime_plan_workloads: [Option<TensorExecutionWorkload>; TensorExecutionWorkloadKey::COUNT],
    runtime_plan_output_sha256: Option<[u8; 32]>,
}

impl BackendPolicy {
    /// Builds a policy from device capabilities and captures process configuration once.
    pub fn from_device_caps(caps: DeviceCaps) -> Self {
        Self::from_device_caps_with_config(caps, ExecutionConfig::from_env())
    }

    /// Builds a deterministic policy from explicit device capabilities and configuration.
    pub fn from_device_caps_with_config(caps: DeviceCaps, config: ExecutionConfig) -> Self {
        let runtime_policy = runtime_tensor_policy_for(caps.backend);
        Self::from_runtime_tensor_policy(caps, config, &runtime_policy)
    }

    fn from_runtime_tensor_policy(
        caps: DeviceCaps,
        config: ExecutionConfig,
        runtime_policy: &RuntimeTensorBackendPolicy,
    ) -> Self {
        Self {
            caps,
            config,
            matmul_backend: matmul_backend_for(runtime_policy.dense_matmul),
            prepacked_matmul_backend: prepacked_matmul_backend_for(runtime_policy.prepacked_matmul),
            layer_norm_backend: layer_norm_backend_for(runtime_policy.layer_norm),
            attention_backend: attention_backend_for(runtime_policy.attention),
            softmax_backend: softmax_backend_for(runtime_policy.softmax),
            tensor_util_backend: tensor_util_backend_for(runtime_policy.tensor_util),
            runtime_plan_workloads: [None; TensorExecutionWorkloadKey::COUNT],
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
        let runtime_policy = runtime_tensor_policy_for(caps.backend);
        Self {
            caps,
            config,
            matmul_backend,
            prepacked_matmul_backend,
            layer_norm_backend,
            attention_backend,
            softmax_backend,
            tensor_util_backend: tensor_util_backend_for(runtime_policy.tensor_util),
            runtime_plan_workloads: [None; TensorExecutionWorkloadKey::COUNT],
            runtime_plan_output_sha256: None,
        }
    }

    /// Materializes an executable local policy from a validated committed plan.
    ///
    /// Artifact validation remains deterministic across builds. Conversion is
    /// the explicit boundary that rejects a local feature or runtime mismatch
    /// instead of silently changing the committed plan.
    pub fn try_from_runtime_plan(
        plan: &RuntimeExecutionPlanPayload,
    ) -> Result<Self, RuntimeExecutionPlanError> {
        plan.validate()?;
        if !plan.execution_allowed {
            return Err(RuntimeExecutionPlanError::ExecutionBlocked {
                blockers: plan.blockers.clone(),
            });
        }
        ensure_local_runtime_ready(&BackendRuntimeState::observe(plan.effective_backend))?;
        let mut policy = Self::from_runtime_tensor_policy(
            plan.request.runtime_probe.caps(),
            plan.request.execution_config,
            &plan.policy,
        );
        let local_policy = policy.runtime_tensor_policy()?;
        for component in RuntimeExecutionComponent::ALL {
            let planned = plan.policy.backend_for(component);
            let local = local_policy.backend_for(component);
            if planned != local {
                return Err(RuntimeExecutionPlanError::LocalBackendMismatch {
                    component: component.as_str(),
                    planned: planned.as_str().to_owned(),
                    local: local.as_str().to_owned(),
                });
            }
        }
        let mut runtime_plan_workloads = [None; TensorExecutionWorkloadKey::COUNT];
        for workload in &plan.request.component_workloads {
            let workload = tensor_execution_workload(workload);
            runtime_plan_workloads[workload.key().index()] = Some(workload);
        }
        policy.runtime_plan_workloads = runtime_plan_workloads;
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

    /// Typed wire projection of the locally executable backend policy.
    pub fn runtime_tensor_policy(
        self,
    ) -> Result<RuntimeTensorBackendPolicy, RuntimeExecutionPlanError> {
        Ok(RuntimeTensorBackendPolicy {
            dense_matmul: runtime_backend_for_execution_id(
                RuntimeExecutionComponent::DenseMatmul,
                self.matmul_backend.execution_id(),
            )?,
            prepacked_matmul: runtime_backend_for_execution_id(
                RuntimeExecutionComponent::PrepackedMatmul,
                self.prepacked_matmul_backend.execution_id(),
            )?,
            layer_norm: runtime_backend_for_execution_id(
                RuntimeExecutionComponent::LayerNorm,
                self.layer_norm_backend.execution_id(),
            )?,
            attention: runtime_backend_for_execution_id(
                RuntimeExecutionComponent::Attention,
                self.attention_backend.execution_id(),
            )?,
            softmax: runtime_backend_for_execution_id(
                RuntimeExecutionComponent::Softmax,
                self.softmax_backend.execution_id(),
            )?,
            tensor_util: runtime_backend_for_execution_id(
                RuntimeExecutionComponent::TensorUtil,
                self.tensor_util_backend.execution_id(),
            )?,
        })
    }

    /// Commitment of the canonical runtime plan that produced this policy.
    pub const fn runtime_plan_output_sha256(self) -> Option<[u8; 32]> {
        self.runtime_plan_output_sha256
    }

    /// Exact component workloads declared by the committed runtime plan.
    pub(crate) const fn runtime_plan_workloads(
        self,
    ) -> [Option<TensorExecutionWorkload>; TensorExecutionWorkloadKey::COUNT] {
        self.runtime_plan_workloads
    }

    pub fn runtime_plan_output_sha256_hex(self) -> Option<String> {
        self.runtime_plan_output_sha256.map(sha256_hex)
    }

    pub const fn device_backend_label(self) -> &'static str {
        self.caps.backend.as_str()
    }

    pub fn matmul_backend_label(self) -> &'static str {
        self.matmul_backend.execution_id()
    }

    pub fn prepacked_matmul_backend_label(self) -> &'static str {
        self.prepacked_matmul_backend.execution_id()
    }

    pub fn layer_norm_backend_label(self) -> &'static str {
        self.layer_norm_backend.execution_id()
    }

    pub fn attention_backend_label(self) -> &'static str {
        self.attention_backend.execution_id()
    }

    pub fn softmax_backend_label(self) -> &'static str {
        self.softmax_backend.execution_id()
    }

    pub fn tensor_util_backend_label(self) -> &'static str {
        self.tensor_util_backend.execution_id()
    }

    /// Resolves the utility-kernel route without consulting mutable global state.
    pub fn tensor_util_route(self, values: usize) -> TensorUtilRoute {
        let requested_backend = self.tensor_util_backend;
        let threshold = self.config.tensor_util_wgpu_min_values;
        // The threshold is a performance hint, never permission to violate strict execution.
        let (selected_backend, status) = if matches!(requested_backend, TensorUtilBackend::GpuWgpu)
            && values < threshold
            && self.config.accelerator_fallback.allows_fallback()
        {
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

fn ensure_local_runtime_ready(
    state: &BackendRuntimeState,
) -> Result<(), RuntimeExecutionPlanError> {
    if state.runtime_ready {
        return Ok(());
    }
    Err(RuntimeExecutionPlanError::LocalRuntimeUnavailable {
        backend: state.backend.as_str().to_owned(),
        status: state.runtime_status.as_str().to_owned(),
        diagnostic: state
            .runtime_error
            .clone()
            .unwrap_or_else(|| state.recommendation.clone()),
    })
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
        self.requested_backend.execution_id()
    }

    pub fn selected_backend_label(self) -> &'static str {
        self.selected_backend.execution_id()
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
            || self
                .component_capability_observation_output_sha256
                .as_deref()
                .is_some_and(|digest| !valid_sha256(digest))
        {
            return Err(invalid_payload(
                "commitment",
                "all commitment fields must be lowercase SHA-256 values",
            ));
        }
        self.request.validate_canonical()?;
        let expected_capability_observation_sha256 = self
            .request
            .component_capability_observation
            .as_ref()
            .map(|observation| observation.output_sha256.clone());
        if self.component_capability_observation_output_sha256
            != expected_capability_observation_sha256
        {
            return Err(invalid_payload(
                "component_capability_observation_output_sha256",
                "must match the nested committed capability observation",
            ));
        }
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

    /// Bind a committed plan to planning-only consumers.
    ///
    /// This validates the complete Rust-owned commitment and rejects blocked
    /// plans, but deliberately does not inspect the receiving process's tensor
    /// features or runtime. Use [`BackendPolicy::try_from_runtime_plan`] before
    /// executing kernels locally.
    pub fn try_planning_context(
        &self,
    ) -> Result<RuntimeExecutionPlanningContext, RuntimeExecutionPlanError> {
        self.validate()?;
        if !self.execution_allowed {
            return Err(RuntimeExecutionPlanError::ExecutionBlocked {
                blockers: self.blockers.clone(),
            });
        }
        Ok(RuntimeExecutionPlanningContext {
            device_caps: self.request.runtime_probe.caps(),
            execution_config: self.request.execution_config,
            runtime_execution_plan_output_sha256: self.output_sha256.clone(),
        })
    }
}

/// Validate a tensor completion receipt against one explicit committed plan.
///
/// The plan is replayed and must authorize execution, but validation does not
/// require the receiving process to expose the same local accelerator. The
/// exact tensor binding then reapplies dispatch-time workload and route rules.
pub fn validate_tensor_execution_receipt_against_runtime_plan(
    plan: &RuntimeExecutionPlanPayload,
    receipt: &TensorExecutionReceipt,
) -> Result<(), RuntimeExecutionReceiptValidationError> {
    let binding = validated_tensor_execution_plan_binding(plan)?;
    binding.validate_receipt(receipt)?;
    Ok(())
}

fn validated_tensor_execution_plan_binding(
    plan: &RuntimeExecutionPlanPayload,
) -> Result<TensorExecutionPlanBinding, RuntimeExecutionPlanError> {
    plan.try_planning_context()?;
    TensorExecutionPlanBinding::try_new_with_workloads(
        parse_sha256(&plan.output_sha256)?,
        RuntimeExecutionComponent::ALL
            .map(|component| tensor_execution_backend(plan.policy.backend_for(component))),
        plan.request
            .component_workloads
            .iter()
            .map(tensor_execution_workload),
        plan.request.execution_config.accelerator_fallback,
        plan.request.execution_config.tensor_util_wgpu_min_values,
    )
    .map_err(|error| invalid_payload("component_workloads", error.to_string()))
}

/// Observe local component kernels once and return an enriched replay request.
///
/// The observation is a separate Rust-owned committed contract. Validation
/// replays that contract without querying the receiving process's device.
pub fn observe_runtime_execution_plan_capabilities(
    mut request: RuntimeExecutionPlanRequest,
) -> Result<RuntimeExecutionPlanRequest, RuntimeExecutionPlanError> {
    request.component_capability_observation = None;
    let mut request = request.canonicalized()?;
    request.component_capability_observation = Some(observe_runtime_component_capabilities(
        RuntimeComponentCapabilityObservationRequest {
            runtime_probe: request.runtime_probe.clone(),
            policy: runtime_tensor_policy_for(request.runtime_probe.effective_backend()),
            component_workloads: request.component_workloads.clone(),
        },
    )?);
    request.canonicalized()
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
    let runtime_route =
        evaluate_runtime_device_route_from_probes(RuntimeDeviceRouteProbeRequest {
            probes: vec![request.runtime_probe.clone()],
            diagnostic_reports: Vec::new(),
            requested_backends: vec![requested_label.clone()],
            required_available_backends: Vec::new(),
            required_ready_backends: vec![requested_label.clone()],
        })?;
    let route_row = runtime_route.route_for(&requested_label)?;
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

    let runtime_ready = runtime_route
        .selection
        .as_ref()
        .is_some_and(|selection| selection.requested_backend == requested_label);
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
    // `allow` permits an explicit software route; it must not turn evidence that
    // the selected kernel is unusable into a successful preflight.
    blockers.extend(
        component_routes
            .iter()
            .filter(|route| {
                route.route == RuntimeComponentRouteClass::Direct
                    && tensor_backend_is_native(route.selected_backend, effective_backend)
                    && route.capability_state.is_known_unready()
            })
            .map(|route| {
                format!(
                    "component_capability_unready:{}:{}",
                    route.component.as_str(),
                    route.capability_state.as_str()
                )
            }),
    );
    if request.execution_config.accelerator_fallback.is_strict() {
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
        blockers.extend(
            component_routes
                .iter()
                .filter(|route| {
                    route.route == RuntimeComponentRouteClass::Direct
                        && tensor_backend_is_native(route.selected_backend, effective_backend)
                        && !route.capability_state.is_ready()
                        && !route.capability_state.is_known_unready()
                        && (request.component_resolution == RuntimeComponentResolution::Concrete
                            || !route.workloads.is_empty())
                })
                .map(|route| {
                    format!(
                        "component_capability_unready:{}:{}",
                        route.component.as_str(),
                        route.capability_state.as_str()
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
    let component_capability_observation_output_sha256 = request
        .component_capability_observation
        .as_ref()
        .map(|observation| observation.output_sha256.clone());
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
        component_capability_observation_output_sha256,
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

pub(crate) fn runtime_tensor_policy_for(backend: BackendKind) -> RuntimeTensorBackendPolicy {
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

const fn tensor_execution_backend(backend: RuntimeTensorBackend) -> TensorExecutionBackend {
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
        // Strict execution keeps WGPU direct even when the workload is below or lacks a threshold.
        match request.tensor_util_values {
            Some(values)
                if values < threshold
                    && request
                        .execution_config
                        .accelerator_fallback
                        .allows_fallback() =>
            {
                (
                    RuntimeTensorBackend::Cpu,
                    RuntimeComponentRouteClass::CpuThresholdFallback,
                    Some(values),
                    Some(threshold),
                )
            }
            Some(values) => (
                RuntimeTensorBackend::Wgpu,
                RuntimeComponentRouteClass::Direct,
                Some(values),
                Some(threshold),
            ),
            None if request.execution_config.accelerator_fallback.is_strict() => (
                RuntimeTensorBackend::Wgpu,
                RuntimeComponentRouteClass::Direct,
                None,
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
    let workloads = request
        .component_workloads
        .iter()
        .filter(|workload| workload.component() == component)
        .cloned()
        .collect::<Vec<_>>();
    let capability_state = component_capability_state(
        request,
        component,
        selected_backend,
        effective_backend,
        route,
    );
    let native = route == RuntimeComponentRouteClass::Direct
        && tensor_backend_is_native(selected_backend, effective_backend)
        && capability_state.is_ready();
    let fallback = route == RuntimeComponentRouteClass::CpuThresholdFallback;
    RuntimeComponentRoute {
        component,
        requested_backend,
        selected_backend,
        route,
        workloads,
        capability_state,
        native,
        fallback,
        values,
        threshold,
    }
}

fn component_capability_state(
    request: &RuntimeExecutionPlanRequest,
    component: RuntimeExecutionComponent,
    selected_backend: RuntimeTensorBackend,
    effective_backend: BackendKind,
    route: RuntimeComponentRouteClass,
) -> RuntimeComponentCapabilityState {
    if route != RuntimeComponentRouteClass::Direct
        || !tensor_backend_is_native(selected_backend, effective_backend)
    {
        return RuntimeComponentCapabilityState::NotApplicable;
    }
    let workloads = request
        .component_workloads
        .iter()
        .filter(|workload| workload.component() == component)
        .collect::<Vec<_>>();
    if workloads.is_empty()
        && matches!(
            selected_backend,
            RuntimeTensorBackend::Cpu | RuntimeTensorBackend::CpuSimd | RuntimeTensorBackend::Naive
        )
    {
        return RuntimeComponentCapabilityState::Static;
    }
    let Some(observation) = request.component_capability_observation.as_ref() else {
        return RuntimeComponentCapabilityState::Unobserved;
    };
    let mut aggregate = RuntimeComponentCapabilityState::Ready;
    let mut missing = false;
    for workload in workloads {
        let Some(evidence) = observation.capabilities.iter().find(|evidence| {
            evidence.workload == *workload && evidence.backend == selected_backend
        }) else {
            missing = true;
            continue;
        };
        match evidence.status {
            RuntimeComponentCapabilityStatus::Unsupported => {
                return RuntimeComponentCapabilityState::Unsupported;
            }
            RuntimeComponentCapabilityStatus::NotBuilt => {
                aggregate = RuntimeComponentCapabilityState::NotBuilt;
            }
            RuntimeComponentCapabilityStatus::Unavailable
                if aggregate == RuntimeComponentCapabilityState::Ready =>
            {
                aggregate = RuntimeComponentCapabilityState::Unavailable;
            }
            RuntimeComponentCapabilityStatus::Ready
            | RuntimeComponentCapabilityStatus::Unavailable => {}
        }
    }
    if missing && aggregate == RuntimeComponentCapabilityState::Ready {
        RuntimeComponentCapabilityState::Unobserved
    } else {
        aggregate
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

fn invalid_request(field: &'static str, message: impl Into<String>) -> RuntimeExecutionPlanError {
    RuntimeExecutionPlanError::InvalidRequest {
        field,
        message: message.into(),
    }
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
    for (index, pair) in value.as_bytes().as_chunks::<2>().0.iter().enumerate() {
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

fn matmul_backend_for(backend: RuntimeTensorBackend) -> MatmulBackend {
    match backend {
        RuntimeTensorBackend::Auto => MatmulBackend::Auto,
        RuntimeTensorBackend::Cpu => MatmulBackend::CpuNaive,
        RuntimeTensorBackend::Faer if st_tensor::faer_dense::is_available() => {
            MatmulBackend::CpuFaer
        }
        RuntimeTensorBackend::Faer => MatmulBackend::CpuNaive,
        RuntimeTensorBackend::CpuSimd => MatmulBackend::CpuSimd,
        RuntimeTensorBackend::Naive => MatmulBackend::CpuNaive,
        RuntimeTensorBackend::Wgpu => wgpu_matmul_backend(),
        RuntimeTensorBackend::Hip => hip_matmul_backend(),
    }
}

fn prepacked_matmul_backend_for(backend: RuntimeTensorBackend) -> MatmulBackend {
    match backend {
        RuntimeTensorBackend::Auto | RuntimeTensorBackend::Hip => MatmulBackend::Auto,
        RuntimeTensorBackend::Cpu => MatmulBackend::CpuNaive,
        RuntimeTensorBackend::Faer if st_tensor::faer_dense::is_available() => {
            MatmulBackend::CpuFaer
        }
        RuntimeTensorBackend::Faer => MatmulBackend::CpuNaive,
        RuntimeTensorBackend::CpuSimd => MatmulBackend::CpuSimd,
        RuntimeTensorBackend::Naive => MatmulBackend::CpuNaive,
        RuntimeTensorBackend::Wgpu => wgpu_matmul_backend(),
    }
}

fn layer_norm_backend_for(backend: RuntimeTensorBackend) -> LayerNormBackend {
    match backend {
        RuntimeTensorBackend::Cpu => LayerNormBackend::Cpu,
        RuntimeTensorBackend::Wgpu => wgpu_layer_norm_backend(),
        RuntimeTensorBackend::Auto
        | RuntimeTensorBackend::Faer
        | RuntimeTensorBackend::CpuSimd
        | RuntimeTensorBackend::Naive
        | RuntimeTensorBackend::Hip => LayerNormBackend::Auto,
    }
}

fn attention_backend_for(backend: RuntimeTensorBackend) -> AttentionBackend {
    match backend {
        RuntimeTensorBackend::Cpu => AttentionBackend::Cpu,
        RuntimeTensorBackend::Wgpu => wgpu_attention_backend(),
        RuntimeTensorBackend::Auto
        | RuntimeTensorBackend::Faer
        | RuntimeTensorBackend::CpuSimd
        | RuntimeTensorBackend::Naive
        | RuntimeTensorBackend::Hip => AttentionBackend::Auto,
    }
}

fn softmax_backend_for(backend: RuntimeTensorBackend) -> SoftmaxBackend {
    match backend {
        RuntimeTensorBackend::Cpu => SoftmaxBackend::Cpu,
        RuntimeTensorBackend::Wgpu => wgpu_softmax_backend(),
        RuntimeTensorBackend::Auto
        | RuntimeTensorBackend::Faer
        | RuntimeTensorBackend::CpuSimd
        | RuntimeTensorBackend::Naive
        | RuntimeTensorBackend::Hip => SoftmaxBackend::Auto,
    }
}

fn tensor_util_backend_for(backend: RuntimeTensorBackend) -> TensorUtilBackend {
    match backend {
        RuntimeTensorBackend::Cpu => TensorUtilBackend::Cpu,
        RuntimeTensorBackend::Wgpu => wgpu_tensor_util_backend(),
        RuntimeTensorBackend::Auto
        | RuntimeTensorBackend::Faer
        | RuntimeTensorBackend::CpuSimd
        | RuntimeTensorBackend::Naive
        | RuntimeTensorBackend::Hip => TensorUtilBackend::Auto,
    }
}

fn runtime_backend_for_execution_id(
    component: RuntimeExecutionComponent,
    execution_id: &str,
) -> Result<RuntimeTensorBackend, RuntimeExecutionPlanError> {
    let backend = match execution_id {
        "auto" => RuntimeTensorBackend::Auto,
        "cpu" => RuntimeTensorBackend::Cpu,
        "faer" => RuntimeTensorBackend::Faer,
        "cpu_simd" => RuntimeTensorBackend::CpuSimd,
        "naive" => RuntimeTensorBackend::Naive,
        "wgpu" => RuntimeTensorBackend::Wgpu,
        "hip" => RuntimeTensorBackend::Hip,
        unknown => {
            return Err(RuntimeExecutionPlanError::UnsupportedLocalBackend {
                component: component.as_str(),
                backend: unknown.to_owned(),
            });
        }
    };
    Ok(backend)
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

fn wgpu_tensor_util_backend() -> TensorUtilBackend {
    #[cfg(feature = "wgpu")]
    {
        TensorUtilBackend::GpuWgpu
    }
    #[cfg(not(feature = "wgpu"))]
    {
        TensorUtilBackend::Auto
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::runtime_probe::{
        evaluate_runtime_device_probe, resolve_backend, BackendRuntimeStatus,
        RuntimeDeviceProbeRequest,
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
            component_resolution: RuntimeComponentResolution::Concrete,
            component_workloads: Vec::new(),
            component_capability_observation: None,
            tensor_util_values: Some(2048),
            required_native_components: Vec::new(),
        }
    }

    fn softmax_receipt(commitment: &str, rows: u64, cols: u64) -> TensorExecutionReceipt {
        serde_json::from_value(serde_json::json!({
            "kind": "spiraltorch.tensor_execution_receipt",
            "contract_version": "spiraltorch.tensor_execution_receipt.v1",
            "semantic_owner": "st-tensor::execution",
            "component": "softmax",
            "operation": "row_softmax",
            "workload": {"component": "softmax", "rows": rows, "cols": cols},
            "requested_backend": "cpu",
            "selected_backend": "cpu",
            "executed_backend": "cpu",
            "kernel_backend": "cpu",
            "route_status": "direct",
            "runtime_execution_plan_output_sha256": commitment,
        }))
        .expect("typed softmax receipt")
    }

    fn cpu_softmax_plan(rows: u64, cols: u64, threshold: usize) -> RuntimeExecutionPlanPayload {
        let mut request =
            execution_request(probe_for(BackendKind::Cpu), AcceleratorFallback::Allow);
        request.execution_config.tensor_util_wgpu_min_values = threshold;
        request.component_resolution = RuntimeComponentResolution::Deferred;
        request.component_workloads = vec![RuntimeComponentWorkload::Softmax { rows, cols }];
        let request = observe_runtime_execution_plan_capabilities(request)
            .expect("CPU softmax capability is observable");
        evaluate_runtime_execution_plan(request).expect("CPU softmax plan evaluates")
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
        assert_eq!(
            policy
                .runtime_tensor_policy()
                .expect("CPU policy has canonical execution IDs"),
            runtime_tensor_policy_for(BackendKind::Cpu)
        );
    }

    #[test]
    fn explicit_local_policy_projects_to_typed_wire_ids() {
        let policy = BackendPolicy::explicit_with_config(
            DeviceCaps::cpu(),
            ExecutionConfig::default(),
            MatmulBackend::CpuSimd,
            MatmulBackend::CpuNaive,
            LayerNormBackend::Cpu,
            AttentionBackend::Cpu,
            SoftmaxBackend::Cpu,
        );

        assert_eq!(policy.matmul_backend_label(), "cpu_simd");
        assert_eq!(policy.prepacked_matmul_backend_label(), "naive");
        assert_eq!(
            policy
                .runtime_tensor_policy()
                .expect("explicit policy has canonical execution IDs"),
            RuntimeTensorBackendPolicy {
                dense_matmul: RuntimeTensorBackend::CpuSimd,
                prepacked_matmul: RuntimeTensorBackend::Naive,
                layer_norm: RuntimeTensorBackend::Cpu,
                attention: RuntimeTensorBackend::Cpu,
                softmax: RuntimeTensorBackend::Cpu,
                tensor_util: RuntimeTensorBackend::Cpu,
            }
        );
    }

    #[test]
    fn unknown_local_execution_id_fails_closed_with_component_identity() {
        let error = runtime_backend_for_execution_id(
            RuntimeExecutionComponent::Attention,
            "future_accelerator",
        )
        .expect_err("unknown local backend IDs must not enter the wire contract");

        assert_eq!(
            error,
            RuntimeExecutionPlanError::UnsupportedLocalBackend {
                component: "attention",
                backend: "future_accelerator".to_owned(),
            }
        );
    }

    #[test]
    fn executable_plan_materialization_rejects_an_unavailable_local_runtime() {
        let mut unavailable = BackendRuntimeState::observe(BackendKind::Cpu);
        unavailable.backend = BackendKind::Wgpu;
        unavailable.runtime_ready = false;
        unavailable.runtime_status = BackendRuntimeStatus::InitializationFailed;
        unavailable.runtime_error = Some("test adapter is unavailable".to_owned());

        assert_eq!(
            ensure_local_runtime_ready(&unavailable),
            Err(RuntimeExecutionPlanError::LocalRuntimeUnavailable {
                backend: "wgpu".to_owned(),
                status: "initialization_failed".to_owned(),
                diagnostic: "test adapter is unavailable".to_owned(),
            })
        );
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

    #[cfg(feature = "wgpu")]
    #[test]
    fn tensor_utility_threshold_is_part_of_the_captured_plan() {
        let config = ExecutionConfig::new(AcceleratorFallback::Allow, 1024);
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
            AcceleratorFallback::Allow
        );
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn strict_tensor_utility_route_does_not_fall_back_below_the_threshold() {
        let config = ExecutionConfig::new(AcceleratorFallback::Forbid, 1024);
        let policy =
            BackendPolicy::from_device_caps_with_config(DeviceCaps::wgpu(32, true, 256), config);

        let small = policy.tensor_util_route(8);

        assert_eq!(small.requested_backend, TensorUtilBackend::GpuWgpu);
        assert_eq!(small.selected_backend, TensorUtilBackend::GpuWgpu);
        assert_eq!(small.status, TensorUtilRouteStatus::Wgpu);
        assert_eq!(small.threshold, 1024);
    }

    #[cfg(not(feature = "wgpu"))]
    #[test]
    fn feature_disabled_policy_does_not_claim_wgpu_tensor_util_execution() {
        let policy = BackendPolicy::from_device_caps_with_config(
            DeviceCaps::wgpu(32, true, 256),
            ExecutionConfig::default(),
        );

        assert_eq!(
            runtime_tensor_policy_for(BackendKind::Wgpu).tensor_util,
            RuntimeTensorBackend::Wgpu
        );
        assert_eq!(policy.tensor_util_backend(), TensorUtilBackend::Auto);
        assert_eq!(
            policy
                .runtime_tensor_policy()
                .expect("feature-disabled policy has canonical execution IDs")
                .tensor_util,
            RuntimeTensorBackend::Auto
        );
    }

    #[test]
    fn cpu_runtime_plan_is_committed_replayable_and_executable() {
        let mut request =
            execution_request(probe_for(BackendKind::Cpu), AcceleratorFallback::Allow);
        request.component_workloads = representative_component_workloads();
        request.required_native_components = vec![
            RuntimeExecutionComponent::Softmax,
            RuntimeExecutionComponent::DenseMatmul,
            RuntimeExecutionComponent::DenseMatmul,
        ];
        let request = observe_runtime_execution_plan_capabilities(request)
            .expect("CPU component capabilities are observable");

        let payload =
            evaluate_runtime_execution_plan(request.clone()).expect("CPU execution plan evaluates");
        assert_eq!(payload.kind, RUNTIME_EXECUTION_PLAN_KIND);
        assert_eq!(
            payload.contract_version,
            RUNTIME_EXECUTION_PLAN_CONTRACT_VERSION
        );
        assert_eq!(
            payload.runtime_route.contract_version,
            super::super::runtime_route::RUNTIME_DEVICE_ROUTE_CONTRACT_VERSION
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
        let bound_workloads = policy.runtime_plan_workloads();
        for workload in &payload.request.component_workloads {
            let workload = tensor_execution_workload(workload);
            assert_eq!(
                bound_workloads[workload.key().index()],
                Some(workload),
                "materialization must retain the exact declared workload"
            );
        }
        assert_eq!(
            bound_workloads.into_iter().flatten().count(),
            payload.request.component_workloads.len()
        );
    }

    #[test]
    fn runtime_plan_authorizes_only_its_exact_tensor_receipts() {
        let plan = cpu_softmax_plan(2, 3, 1024);
        let receipt = softmax_receipt(&plan.output_sha256, 2, 3);

        validate_tensor_execution_receipt_against_runtime_plan(&plan, &receipt)
            .expect("the committed plan authorizes its exact receipt");

        let foreign_plan = cpu_softmax_plan(2, 3, 2048);
        assert_ne!(foreign_plan.output_sha256, plan.output_sha256);
        assert!(matches!(
            validate_tensor_execution_receipt_against_runtime_plan(&foreign_plan, &receipt),
            Err(RuntimeExecutionReceiptValidationError::TensorExecution(
                TensorExecutionContractError::ReceiptPlanCommitmentMismatch { .. }
            ))
        ));

        let reconstructed_workload = softmax_receipt(&plan.output_sha256, 2, 5);
        reconstructed_workload
            .validate()
            .expect("the reconstructed receipt remains internally consistent");
        assert_eq!(
            validate_tensor_execution_receipt_against_runtime_plan(&plan, &reconstructed_workload,)
                .unwrap_err(),
            RuntimeExecutionReceiptValidationError::TensorExecution(
                TensorExecutionContractError::ReceiptPlanWorkloadMismatch {
                    component: st_tensor::TensorExecutionComponent::Softmax,
                    planned: TensorExecutionWorkload::Softmax { rows: 2, cols: 3 },
                    receipt: TensorExecutionWorkload::Softmax { rows: 2, cols: 5 },
                },
            )
        );
    }

    #[test]
    fn blocked_runtime_plan_cannot_authorize_a_tensor_receipt() {
        let mut request =
            execution_request(probe_for(BackendKind::Cpu), AcceleratorFallback::Forbid);
        request.required_native_components = vec![RuntimeExecutionComponent::DenseMatmul];
        let blocked =
            evaluate_runtime_execution_plan(request).expect("blocked plan remains inspectable");
        assert!(!blocked.execution_allowed);
        let receipt = softmax_receipt(&blocked.output_sha256, 2, 3);

        assert!(matches!(
            validate_tensor_execution_receipt_against_runtime_plan(&blocked, &receipt),
            Err(RuntimeExecutionReceiptValidationError::RuntimePlan(
                RuntimeExecutionPlanError::ExecutionBlocked { .. }
            ))
        ));
    }

    #[test]
    fn planning_context_preserves_provenance_without_materializing_tensor_backends() {
        let payload = evaluate_runtime_execution_plan(execution_request(
            probe_for(BackendKind::Cpu),
            AcceleratorFallback::Allow,
        ))
        .expect("CPU execution plan evaluates");

        let context = payload
            .try_planning_context()
            .expect("ready plan binds to planning provenance");

        assert_eq!(context.device_caps(), payload.request.runtime_probe.caps());
        assert_eq!(context.execution_config(), payload.request.execution_config);
        assert_eq!(
            context.runtime_execution_plan_output_sha256(),
            payload.output_sha256
        );
        let rank = crate::ops::rank_entry::try_plan_rank_with_planning_context(
            crate::backend::unison::RankKind::TopK,
            4,
            128,
            8,
            &context,
        )
        .expect("Rust rank planning accepts validated provenance");
        assert_eq!(
            rank.runtime_execution_plan_output_sha256(),
            Some(payload.output_sha256.as_str())
        );

        let blocked = evaluate_runtime_execution_plan(execution_request(
            probe_for(BackendKind::Wgpu),
            AcceleratorFallback::Forbid,
        ))
        .expect("blocked WGPU plan remains inspectable");
        assert!(matches!(
            blocked.try_planning_context(),
            Err(RuntimeExecutionPlanError::ExecutionBlocked { .. })
        ));
    }

    #[test]
    fn legacy_v7_payload_is_rejected_at_the_contract_boundary() {
        let mut payload = evaluate_runtime_execution_plan(execution_request(
            probe_for(BackendKind::Cpu),
            AcceleratorFallback::Allow,
        ))
        .expect("current execution plan evaluates");
        payload.contract_version = "spiraltorch.runtime_execution_plan.v7".to_owned();

        let error = payload
            .validate()
            .expect_err("v7 and v8 commitments must not share a digest domain");
        assert!(matches!(
            error,
            RuntimeExecutionPlanError::InvalidPayload {
                field: "contract_version",
                ..
            }
        ));
    }

    #[test]
    fn optional_cpu_accelerators_are_unobserved_without_workload_evidence() {
        let request = execution_request(probe_for(BackendKind::Cpu), AcceleratorFallback::Allow);
        let payload = evaluate_runtime_execution_plan(request).expect("CPU plan evaluates");

        for component in [
            RuntimeExecutionComponent::DenseMatmul,
            RuntimeExecutionComponent::PrepackedMatmul,
        ] {
            let route = payload
                .component_routes
                .iter()
                .find(|route| route.component == component)
                .expect("CPU matmul route");
            assert_eq!(
                route.capability_state,
                RuntimeComponentCapabilityState::Unobserved
            );
            assert!(!route.native);
        }
        assert!(!payload.all_components_native);
        assert!(payload.execution_allowed);
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
    fn committed_workload_observation_controls_native_routes() {
        let request = execution_request(probe_for(BackendKind::Cpu), AcceleratorFallback::Allow);
        let unobserved =
            evaluate_runtime_execution_plan(request.clone()).expect("unobserved plan evaluates");

        assert!(!unobserved.all_components_native);
        assert!(unobserved.component_routes.iter().any(|route| {
            route.capability_state == RuntimeComponentCapabilityState::Unobserved
        }));

        let mut observed_request = request;
        observed_request.component_workloads = representative_component_workloads();
        let observed_request = observe_runtime_execution_plan_capabilities(observed_request)
            .expect("Rust capability observation");
        let observed = evaluate_runtime_execution_plan(observed_request.clone())
            .expect("observed plan evaluates");

        assert!(observed.all_components_native);
        assert_eq!(
            observed.native_components,
            RuntimeExecutionComponent::ALL.to_vec()
        );
        assert!(observed.component_routes.iter().all(|route| {
            route.capability_state == RuntimeComponentCapabilityState::Ready
                && !route.workloads.is_empty()
        }));
        observed
            .validate_against(observed_request)
            .expect("capability evidence replays without a live device query");
    }

    #[test]
    fn naked_component_capability_self_reports_are_not_plan_requests() {
        let request = execution_request(probe_for(BackendKind::Wgpu), AcceleratorFallback::Allow);
        let mut value = serde_json::to_value(request).expect("serializable request");
        value.as_object_mut().expect("request object").insert(
            "component_capabilities".to_owned(),
            serde_json::json!([{
                "workload": {"component": "softmax", "rows": 2, "cols": 4},
                "backend": "wgpu",
                "status": "ready"
            }]),
        );

        let error = serde_json::from_value::<RuntimeExecutionPlanRequest>(value)
            .expect_err("naked client capability claims must not deserialize");
        assert!(error.to_string().contains("component_capabilities"));
    }

    #[test]
    fn component_capability_observation_rejects_status_tampering() {
        let mut request =
            execution_request(probe_for(BackendKind::Cpu), AcceleratorFallback::Allow);
        request.component_workloads = vec![RuntimeComponentWorkload::Softmax { rows: 2, cols: 4 }];
        let mut request = observe_runtime_execution_plan_capabilities(request)
            .expect("Rust capability observation");
        request
            .component_capability_observation
            .as_mut()
            .expect("committed observation")
            .capabilities[0]
            .status = RuntimeComponentCapabilityStatus::NotBuilt;

        let error = evaluate_runtime_execution_plan(request)
            .expect_err("tampered capability status must fail before planning");
        assert!(matches!(
            error,
            RuntimeExecutionPlanError::ComponentCapabilityObservation(
                RuntimeComponentCapabilityObservationError::InvalidPayload {
                    field: "capabilities",
                    ..
                }
            )
        ));
    }

    #[test]
    fn capability_observation_cannot_override_the_plan_policy() {
        let mut request =
            execution_request(probe_for(BackendKind::Cpu), AcceleratorFallback::Allow);
        request.component_workloads = vec![RuntimeComponentWorkload::Softmax { rows: 2, cols: 4 }];
        request.component_capability_observation = Some(
            observe_runtime_component_capabilities(RuntimeComponentCapabilityObservationRequest {
                runtime_probe: request.runtime_probe.clone(),
                policy: runtime_tensor_policy_for(BackendKind::Wgpu),
                component_workloads: request.component_workloads.clone(),
            })
            .expect("valid observation for a different policy"),
        );

        let error = evaluate_runtime_execution_plan(request)
            .expect_err("capability observations cannot select the plan policy");
        assert!(matches!(
            error,
            RuntimeExecutionPlanError::ComponentCapabilityObservation(
                RuntimeComponentCapabilityObservationError::InvalidPayload {
                    field: "request",
                    ..
                }
            )
        ));
    }

    #[test]
    fn tensor_util_workload_owns_the_threshold_volume() {
        let mut request =
            execution_request(probe_for(BackendKind::Wgpu), AcceleratorFallback::Allow);
        request.component_workloads = vec![RuntimeComponentWorkload::TensorUtil {
            operation: RuntimeTensorUtilOperation::Scale,
            rows: 2,
            cols: 4,
        }];
        let error = evaluate_runtime_execution_plan(request.clone())
            .expect_err("a contradictory representative value must fail closed");
        assert!(matches!(
            error,
            RuntimeExecutionPlanError::InvalidRequest {
                field: "tensor_util_values",
                ..
            }
        ));

        request.tensor_util_values = None;
        let payload = evaluate_runtime_execution_plan(request)
            .expect("the workload volume is canonical when the alias is omitted");
        assert_eq!(payload.request.tensor_util_values, Some(8));
        let route = payload
            .component_routes
            .iter()
            .find(|route| route.component == RuntimeExecutionComponent::TensorUtil)
            .expect("tensor util route");
        assert_eq!(route.values, Some(8));
        assert_eq!(
            route.route,
            RuntimeComponentRouteClass::CpuThresholdFallback
        );
    }

    #[test]
    fn one_runtime_plan_executes_max_forward_and_backward_operations() {
        let mut request =
            execution_request(probe_for(BackendKind::Cpu), AcceleratorFallback::Allow);
        request.component_resolution = RuntimeComponentResolution::Deferred;
        request.tensor_util_values = None;
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
        let request = observe_runtime_execution_plan_capabilities(request)
            .expect("both tensor utility operations are observable");
        let plan = evaluate_runtime_execution_plan(request)
            .expect("one plan commits both tensor utility operations");
        let route = plan
            .component_routes
            .iter()
            .find(|route| route.component == RuntimeExecutionComponent::TensorUtil)
            .expect("tensor utility route");

        assert_eq!(route.workloads.len(), 2);
        assert_eq!(
            route.capability_state,
            RuntimeComponentCapabilityState::Ready
        );
        assert!(route.native);
        assert_eq!(plan.request.tensor_util_values, Some(6));

        let binding = validated_tensor_execution_plan_binding(&plan).expect("tensor binding");
        let receipts = {
            let _guard = st_tensor::execution::push_execution_plan_binding(binding);
            [
                st_tensor::prepare_tensor_execution(
                    TensorExecutionWorkload::TensorUtil {
                        operation: st_tensor::TensorUtilOperation::MaxAxis0,
                        rows: 3,
                        cols: 2,
                    },
                    "max_axis0",
                    TensorExecutionBackend::Cpu,
                )
                .unwrap()
                .complete(TensorExecutionBackend::Cpu, None)
                .unwrap()
                .receipt(),
                st_tensor::prepare_tensor_execution(
                    TensorExecutionWorkload::TensorUtil {
                        operation: st_tensor::TensorUtilOperation::MaxAxis0Backward,
                        rows: 3,
                        cols: 2,
                    },
                    "max_axis0_backward",
                    TensorExecutionBackend::Cpu,
                )
                .unwrap()
                .complete(TensorExecutionBackend::Cpu, None)
                .unwrap()
                .receipt(),
            ]
        };
        for receipt in receipts {
            validate_tensor_execution_receipt_against_runtime_plan(&plan, &receipt)
                .expect("the plan authorizes each operation receipt");
        }
    }

    #[test]
    fn one_runtime_plan_executes_linear_readout_operations() {
        let operations = [
            (
                RuntimeTensorUtilOperation::AddRow,
                st_tensor::TensorUtilOperation::AddRow,
                "add_row_inplace",
            ),
            (
                RuntimeTensorUtilOperation::SumAxis0,
                st_tensor::TensorUtilOperation::SumAxis0,
                "sum_axis0",
            ),
            (
                RuntimeTensorUtilOperation::SumAxis0Scaled,
                st_tensor::TensorUtilOperation::SumAxis0Scaled,
                "sum_axis0_scaled",
            ),
        ];
        let mut request =
            execution_request(probe_for(BackendKind::Cpu), AcceleratorFallback::Allow);
        request.component_resolution = RuntimeComponentResolution::Deferred;
        request.tensor_util_values = None;
        request.component_workloads = operations
            .iter()
            .map(|(operation, _, _)| RuntimeComponentWorkload::TensorUtil {
                operation: *operation,
                rows: 3,
                cols: 2,
            })
            .collect();
        let request = observe_runtime_execution_plan_capabilities(request)
            .expect("linear readout operations are observable");
        let plan = evaluate_runtime_execution_plan(request)
            .expect("one plan commits every linear readout operation");
        let route = plan
            .component_routes
            .iter()
            .find(|route| route.component == RuntimeExecutionComponent::TensorUtil)
            .expect("tensor utility route");

        assert_eq!(route.workloads.len(), operations.len());
        assert_eq!(
            route.capability_state,
            RuntimeComponentCapabilityState::Ready
        );
        assert!(route.native);
        assert_eq!(plan.request.tensor_util_values, Some(6));

        let binding = validated_tensor_execution_plan_binding(&plan).expect("tensor binding");
        let receipts = {
            let _guard = st_tensor::execution::push_execution_plan_binding(binding);
            operations.map(|(_, operation, name)| {
                st_tensor::prepare_tensor_execution(
                    TensorExecutionWorkload::TensorUtil {
                        operation,
                        rows: 3,
                        cols: 2,
                    },
                    name,
                    TensorExecutionBackend::Cpu,
                )
                .unwrap()
                .complete(TensorExecutionBackend::Cpu, None)
                .unwrap()
                .receipt()
            })
        };
        for receipt in receipts {
            validate_tensor_execution_receipt_against_runtime_plan(&plan, &receipt)
                .expect("the plan authorizes each linear readout operation receipt");
        }
    }

    #[test]
    fn tensor_util_component_route_rejects_mixed_operation_volumes() {
        let mut request =
            execution_request(probe_for(BackendKind::Cpu), AcceleratorFallback::Allow);
        request.tensor_util_values = None;
        request.component_workloads = vec![
            RuntimeComponentWorkload::TensorUtil {
                operation: RuntimeTensorUtilOperation::MaxAxis0,
                rows: 3,
                cols: 2,
            },
            RuntimeComponentWorkload::TensorUtil {
                operation: RuntimeTensorUtilOperation::MaxAxis0Backward,
                rows: 4,
                cols: 2,
            },
        ];

        assert!(matches!(
            evaluate_runtime_execution_plan(request),
            Err(RuntimeExecutionPlanError::InvalidRequest {
                field: "component_workloads",
                ..
            })
        ));
    }

    fn representative_component_workloads() -> Vec<RuntimeComponentWorkload> {
        vec![
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
                rows: 32,
                cols: 64,
            },
            RuntimeComponentWorkload::TensorUtil {
                operation: RuntimeTensorUtilOperation::AddRow,
                rows: 32,
                cols: 64,
            },
            RuntimeComponentWorkload::TensorUtil {
                operation: RuntimeTensorUtilOperation::SumAxis0,
                rows: 32,
                cols: 64,
            },
            RuntimeComponentWorkload::TensorUtil {
                operation: RuntimeTensorUtilOperation::SumAxis0Scaled,
                rows: 32,
                cols: 64,
            },
            RuntimeComponentWorkload::TensorUtil {
                operation: RuntimeTensorUtilOperation::MaxAxis0,
                rows: 32,
                cols: 64,
            },
            RuntimeComponentWorkload::TensorUtil {
                operation: RuntimeTensorUtilOperation::MaxAxis0Backward,
                rows: 32,
                cols: 64,
            },
        ]
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn local_wgpu_observer_commits_ready_kernel_evidence() {
        if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
            eprintln!("skipping runtime WGPU plan test; set SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1");
            return;
        }

        let mut request =
            execution_request(probe_for(BackendKind::Wgpu), AcceleratorFallback::Forbid);
        request.component_workloads = representative_component_workloads();
        let observed = observe_runtime_execution_plan_capabilities(request)
            .expect("local WGPU capabilities are observable");
        assert!(observed
            .component_capability_observation
            .as_ref()
            .expect("committed observation")
            .capabilities
            .iter()
            .all(|evidence| evidence.status == RuntimeComponentCapabilityStatus::Ready));

        let plan = evaluate_runtime_execution_plan(observed).expect("observed plan evaluates");
        assert!(plan.all_components_native);
        assert!(plan.execution_allowed, "plan blockers: {:?}", plan.blockers);
        plan.validate()
            .expect("plan replay uses committed evidence instead of the live device");
    }

    #[cfg(not(feature = "wgpu"))]
    #[test]
    fn feature_disabled_observer_commits_not_built_instead_of_native() {
        let mut request =
            execution_request(probe_for(BackendKind::Wgpu), AcceleratorFallback::Allow);
        request.component_workloads = vec![RuntimeComponentWorkload::Softmax { rows: 2, cols: 4 }];
        let observed = observe_runtime_execution_plan_capabilities(request)
            .expect("feature-disabled observation remains inspectable");
        assert_eq!(
            observed
                .component_capability_observation
                .as_ref()
                .expect("committed observation")
                .capabilities[0]
                .status,
            RuntimeComponentCapabilityStatus::NotBuilt
        );

        let plan = evaluate_runtime_execution_plan(observed).expect("plan evaluates");
        let softmax = plan
            .component_routes
            .iter()
            .find(|route| route.component == RuntimeExecutionComponent::Softmax)
            .expect("softmax route");
        assert_eq!(
            softmax.capability_state,
            RuntimeComponentCapabilityState::NotBuilt
        );
        assert!(!softmax.native);
        assert!(!plan.execution_allowed);
        assert!(plan
            .blockers
            .contains(&"component_capability_unready:softmax:not_built".to_owned()));
    }

    #[test]
    fn strict_execution_never_commits_a_tensor_util_threshold_fallback() {
        let mut request =
            execution_request(probe_for(BackendKind::Wgpu), AcceleratorFallback::Forbid);
        request.tensor_util_values = Some(8);
        let payload = evaluate_runtime_execution_plan(request).expect("plan evaluates");

        let tensor_util = payload
            .component_routes
            .iter()
            .find(|route| route.component == RuntimeExecutionComponent::TensorUtil)
            .expect("tensor util route");

        assert!(!payload.execution_allowed);
        assert_eq!(tensor_util.selected_backend, RuntimeTensorBackend::Wgpu);
        assert_eq!(tensor_util.route, RuntimeComponentRouteClass::Direct);
        assert!(!tensor_util.fallback);
        assert!(payload
            .blockers
            .contains(&"component_capability_unready:tensor_util:unobserved".to_owned()));
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn strict_wgpu_policy_resolves_a_ready_tensor_util_route() {
        if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
            eprintln!("skipping runtime WGPU plan test; set SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1");
            return;
        }
        let mut request =
            execution_request(probe_for(BackendKind::Wgpu), AcceleratorFallback::Forbid);
        request.component_workloads = representative_component_workloads();
        let request = observe_runtime_execution_plan_capabilities(request)
            .expect("local WGPU capabilities are observable");
        let ready = evaluate_runtime_execution_plan(request).expect("plan evaluates");

        let tensor_util = ready
            .component_routes
            .iter()
            .find(|route| route.component == RuntimeExecutionComponent::TensorUtil)
            .expect("tensor util route");
        assert!(tensor_util.native);
        assert_eq!(
            tensor_util.capability_state,
            RuntimeComponentCapabilityState::Ready
        );
        assert!(ready.all_components_native);
        assert!(!ready
            .blockers
            .iter()
            .any(|blocker| blocker.starts_with("conditional_")));
    }

    #[test]
    fn strict_wgpu_concrete_plan_blocks_unobserved_tensor_util_capability() {
        let mut request =
            execution_request(probe_for(BackendKind::Wgpu), AcceleratorFallback::Forbid);
        request.tensor_util_values = None;
        let blocked = evaluate_runtime_execution_plan(request).expect("plan evaluates");

        assert!(!blocked.execution_allowed);
        assert!(blocked
            .blockers
            .contains(&"component_capability_unready:tensor_util:unobserved".to_owned()));
        assert!(!blocked
            .blockers
            .iter()
            .any(|blocker| blocker.starts_with("conditional_component_unresolved:")));
    }

    #[test]
    fn strict_wgpu_deferred_plan_only_retains_runtime_blockers() {
        let mut request =
            execution_request(probe_for(BackendKind::Wgpu), AcceleratorFallback::Forbid);
        request.tensor_util_values = None;
        request.component_resolution = RuntimeComponentResolution::Deferred;

        let deferred = evaluate_runtime_execution_plan(request).expect("plan evaluates");

        assert!(!deferred.all_components_native);
        assert!(deferred.native_components.is_empty());
        assert!(deferred.component_routes.iter().all(|route| {
            route.route == RuntimeComponentRouteClass::Direct
                && route.capability_state == RuntimeComponentCapabilityState::Unobserved
        }));
        assert!(deferred.blockers.iter().all(|blocker| {
            blocker.starts_with("runtime_route:") || blocker.starts_with("runtime_route_not_ready:")
        }));
        if deferred.runtime_ready {
            assert!(deferred.execution_allowed);
            assert_eq!(deferred.status, RuntimeExecutionPlanStatus::Ready);
            assert!(deferred.blockers.is_empty());
        } else {
            assert!(!deferred.execution_allowed);
            assert_eq!(deferred.status, RuntimeExecutionPlanStatus::Blocked);
            assert!(!deferred.blockers.is_empty());
        }
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

    #[test]
    fn execution_plan_binds_the_nested_capability_observation_sha() {
        let mut request =
            execution_request(probe_for(BackendKind::Cpu), AcceleratorFallback::Allow);
        request.component_workloads = vec![RuntimeComponentWorkload::Softmax { rows: 2, cols: 4 }];
        let request =
            observe_runtime_execution_plan_capabilities(request).expect("capability observation");
        let mut payload = evaluate_runtime_execution_plan(request).expect("plan evaluates");
        assert_eq!(
            payload
                .component_capability_observation_output_sha256
                .as_deref(),
            payload
                .request
                .component_capability_observation
                .as_ref()
                .map(|observation| observation.output_sha256.as_str())
        );

        payload.component_capability_observation_output_sha256 = None;
        let error = payload
            .validate()
            .expect_err("missing capability lineage must fail closed");
        assert!(matches!(
            error,
            RuntimeExecutionPlanError::InvalidPayload {
                field: "component_capability_observation_output_sha256",
                ..
            }
        ));
    }
}
