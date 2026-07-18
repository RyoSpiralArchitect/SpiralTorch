// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Canonical runtime-device observation and capability derivation.
//!
//! This module owns what the current Rust build can actually execute and how
//! capability-derived launch hints are calculated. `runtime_route` remains the
//! sole owner of policy gates over these observations. Bindings should expose
//! [`RuntimeDeviceProbePayload::to_transport_value`] instead of reconstructing
//! readiness, surrogate, or capability semantics in their host language.

use super::device_caps::{BackendKind, DeviceCaps, DeviceCapsError};
use super::runtime_route::RuntimeDeviceRouteEvidence;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use st_tensor::{emit_tensor_op, emit_tensor_op_meta};
use std::fmt::Write as _;
use thiserror::Error;

/// Stable contract identifier shared by Rust, Python, and WASM clients.
pub const RUNTIME_DEVICE_PROBE_CONTRACT_VERSION: &str = "spiraltorch.runtime_device_probe.v1";
/// Payload kind for one runtime-device observation.
pub const RUNTIME_DEVICE_PROBE_KIND: &str = "spiraltorch.runtime_device_probe";
/// Crate/module that owns runtime-device observation semantics.
pub const RUNTIME_DEVICE_PROBE_SEMANTIC_OWNER: &str = "st-core::backend::runtime_probe";
/// Backend label attached to payloads produced by the canonical implementation.
pub const RUNTIME_DEVICE_PROBE_SEMANTIC_BACKEND: &str = "rust";

const RUNTIME_DEVICE_PROBE_REQUEST_DIGEST_DOMAIN: &[u8] =
    b"spiraltorch.runtime_device_probe.request.v1\0";
const RUNTIME_DEVICE_PROBE_OUTPUT_DIGEST_DOMAIN: &[u8] =
    b"spiraltorch.runtime_device_probe.output.v1\0";
const RUNTIME_DEVICE_PROBE_MAX_CLIENT_BYTES: usize = 64;

#[derive(Debug, Error, PartialEq)]
pub enum RuntimeDeviceProbeError {
    #[error(transparent)]
    InvalidDeviceCaps(#[from] DeviceCapsError),
    #[error("runtime-device probe field '{field}' must be positive when present")]
    ZeroInput { field: &'static str },
    #[error("runtime-device probe field '{field}' requires 'cols'")]
    HintWithoutColumns { field: &'static str },
    #[error(
        "runtime-device probe caps backend '{caps_backend}' does not match effective backend '{effective_backend}'"
    )]
    CapsBackendMismatch {
        caps_backend: &'static str,
        effective_backend: &'static str,
    },
    #[error("runtime-device probe for '{backend}' has an invalid MPS overlay: {message}")]
    InvalidMpsOverlay {
        backend: &'static str,
        message: String,
    },
    #[error(
        "runtime-device probe selected surrogate '{backend}', but that backend is not runtime-ready in this build"
    )]
    SurrogateNotReady { backend: &'static str },
    #[error("runtime-device probe produced non-finite derived field '{field}'")]
    NonFiniteDerived { field: &'static str },
    #[error("invalid runtime-device probe payload field '{field}': {message}")]
    InvalidPayload {
        field: &'static str,
        message: String,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MpsHostClass {
    NonMacHost,
    AppleSiliconMac,
    IntelMac,
}

impl MpsHostClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NonMacHost => "non-mac-host",
            Self::AppleSiliconMac => "apple-silicon-mac",
            Self::IntelMac => "intel-mac",
        }
    }

    const fn platform_supported(self) -> bool {
        !matches!(self, Self::NonMacHost)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MpsProbeStatus {
    BuildFeatureDisabled,
    UnsupportedHost,
    Placeholder,
}

impl MpsProbeStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BuildFeatureDisabled => "build-feature-disabled",
            Self::UnsupportedHost => "unsupported-host",
            Self::Placeholder => "placeholder",
        }
    }
}

/// Native MPS observations plus the Rust-selected execution surrogate.
///
/// Native MPS remains an honest placeholder. A WGPU surrogate is selected only
/// when `wgpu-rt` is compiled and therefore runtime-ready; merely compiling the
/// planner-side `wgpu` integration is not sufficient.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MpsProbeReport {
    pub feature_enabled: bool,
    pub platform_supported: bool,
    pub host_class: MpsHostClass,
    pub backend_wired: bool,
    pub initialized: bool,
    pub planner_surrogate_backend: BackendKind,
    pub planner_caps: DeviceCaps,
}

impl MpsProbeReport {
    pub const fn status(self) -> MpsProbeStatus {
        if !self.feature_enabled {
            MpsProbeStatus::BuildFeatureDisabled
        } else if !self.platform_supported {
            MpsProbeStatus::UnsupportedHost
        } else {
            MpsProbeStatus::Placeholder
        }
    }

    pub const fn placeholder(self) -> bool {
        !self.backend_wired
    }

    pub const fn available(self) -> bool {
        self.feature_enabled && self.platform_supported && self.backend_wired
    }

    pub const fn planner_route(self) -> &'static str {
        match self.planner_surrogate_backend {
            BackendKind::Wgpu => "metal-via-wgpu",
            _ => "cpu-fallback",
        }
    }

    pub const fn recommended_backend(self) -> BackendKind {
        self.planner_surrogate_backend
    }

    pub const fn recommendation(self) -> &'static str {
        match self.planner_surrogate_backend {
            BackendKind::Wgpu => {
                "use backend='wgpu' today; native MPS kernels are not wired yet"
            }
            _ => {
                "use backend='cpu' today; enable the 'wgpu-rt' feature to route Metal hosts through the WGPU runtime"
            }
        }
    }

    pub const fn error(self) -> &'static str {
        if !self.feature_enabled {
            "mps feature is not enabled in this build"
        } else if !self.platform_supported {
            "mps backend requires a macOS host"
        } else {
            "mps backend is a placeholder; kernels are not wired yet"
        }
    }

    pub const fn host_os(self) -> &'static str {
        std::env::consts::OS
    }

    pub const fn host_arch(self) -> &'static str {
        std::env::consts::ARCH
    }

    pub fn validate(&self) -> Result<(), RuntimeDeviceProbeError> {
        if self.platform_supported != self.host_class.platform_supported() {
            return Err(invalid_mps_overlay(
                "platform_supported disagrees with host_class",
            ));
        }
        if self.backend_wired || self.initialized {
            return Err(invalid_mps_overlay(
                "v1 describes the current unwired, uninitialized native MPS placeholder",
            ));
        }
        if !matches!(
            self.planner_surrogate_backend,
            BackendKind::Wgpu | BackendKind::Cpu
        ) {
            return Err(invalid_mps_overlay(
                "planner surrogate must be either 'wgpu' or 'cpu'",
            ));
        }
        if !self.platform_supported && self.planner_surrogate_backend != BackendKind::Cpu {
            return Err(invalid_mps_overlay(
                "non-macOS hosts must use the CPU planner surrogate",
            ));
        }
        self.planner_caps.validate()?;
        if self.planner_caps.backend != self.planner_surrogate_backend {
            return Err(invalid_mps_overlay(
                "planner_caps.backend must match planner_surrogate_backend",
            ));
        }
        Ok(())
    }

    /// Stable MPS probe projection used by language bindings.
    pub fn to_transport_value(self) -> serde_json::Value {
        serde_json::json!({
            "kind": "spiraltorch.mps_probe",
            "semantic_owner": RUNTIME_DEVICE_PROBE_SEMANTIC_OWNER,
            "semantic_backend": RUNTIME_DEVICE_PROBE_SEMANTIC_BACKEND,
            "backend": "mps",
            "status": self.status().as_str(),
            "feature_enabled": self.feature_enabled,
            "platform_supported": self.platform_supported,
            "host_class": self.host_class,
            "backend_wired": self.backend_wired,
            "placeholder": self.placeholder(),
            "available": self.available(),
            "initialized": self.initialized,
            "host_os": self.host_os(),
            "host_arch": self.host_arch(),
            "planner_surrogate_backend": self.planner_surrogate_backend,
            "planner_route": self.planner_route(),
            "planner_caps": self.planner_caps,
            "recommended_backend": self.recommended_backend(),
            "recommendation": self.recommendation(),
            "devices": [],
            "error": self.error(),
        })
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKernelStatus {
    FeatureDisabled,
    Cpu,
    KernelWired,
    Placeholder,
    StubWithoutHipReal,
}

impl BackendKernelStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FeatureDisabled => "feature_disabled",
            Self::Cpu => "cpu",
            Self::KernelWired => "kernel_wired",
            Self::Placeholder => "placeholder",
            Self::StubWithoutHipReal => "stub_without_hip_real",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendRuntimeStatus {
    FeatureDisabled,
    Cpu,
    Ready,
    InitializationFailed,
    Placeholder,
    StubWithoutHipReal,
}

impl BackendRuntimeStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FeatureDisabled => "feature_disabled",
            Self::Cpu => "cpu",
            Self::Ready => "ready",
            Self::InitializationFailed => "initialization_failed",
            Self::Placeholder => "placeholder",
            Self::StubWithoutHipReal => "stub_without_hip_real",
        }
    }
}

/// Build/runtime facts for one backend. The readiness equation is deliberately
/// explicit so clients do not need to reconstruct it.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BackendRuntimeState {
    pub backend: BackendKind,
    /// Planner or integration code for this backend is present in the build.
    pub integration_compiled: bool,
    /// The feature gate required by the executable runtime is enabled.
    pub feature_enabled: bool,
    pub real_kernels_compiled: bool,
    pub placeholder: bool,
    pub kernel_status: BackendKernelStatus,
    pub runtime_probe_attempted: bool,
    pub runtime_initialized: bool,
    pub runtime_status: BackendRuntimeStatus,
    pub runtime_ready: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_error: Option<String>,
    pub recommendation: String,
}

impl BackendRuntimeState {
    pub fn observe(backend: BackendKind) -> Self {
        let integration_compiled = backend_integration_compiled(backend);
        let feature_enabled = backend_feature_enabled(backend);
        let real_kernels_compiled = backend_real_kernels_compiled(backend);
        let placeholder = backend_placeholder(backend);
        let kernel_status = observed_backend_kernel_status(backend, feature_enabled);
        let build_ready = feature_enabled && real_kernels_compiled && !placeholder;
        let (runtime_probe_attempted, runtime_initialized, runtime_error) =
            probe_backend_initialization(backend, build_ready);
        let runtime_status = observed_backend_runtime_status(kernel_status, runtime_initialized);
        let runtime_ready = build_ready && runtime_initialized;
        Self {
            backend,
            integration_compiled,
            feature_enabled,
            real_kernels_compiled,
            placeholder,
            kernel_status,
            runtime_probe_attempted,
            runtime_initialized,
            runtime_status,
            runtime_ready,
            runtime_error,
            recommendation: backend_runtime_recommendation(backend).to_owned(),
        }
    }

    pub fn validate(&self) -> Result<(), RuntimeDeviceProbeError> {
        if self.real_kernels_compiled && !self.feature_enabled {
            return Err(invalid_payload(
                "real_kernels_compiled",
                "cannot be true while the executable runtime feature is disabled",
            ));
        }
        if self.backend != BackendKind::Wgpu && self.integration_compiled != self.feature_enabled {
            return Err(invalid_payload(
                "integration_compiled",
                "must match feature_enabled for every backend except the two-stage WGPU integration",
            ));
        }
        if self.placeholder != (self.backend == BackendKind::Mps) {
            return Err(invalid_payload(
                "placeholder",
                "must be true exactly for the native MPS placeholder backend",
            ));
        }
        if self.backend == BackendKind::Mps && self.real_kernels_compiled {
            return Err(invalid_payload(
                "real_kernels_compiled",
                "the v1 native MPS placeholder cannot claim real kernels",
            ));
        }
        if self.backend == BackendKind::Cpu
            && (!self.integration_compiled || !self.feature_enabled || !self.real_kernels_compiled)
        {
            return Err(invalid_payload(
                "backend",
                "CPU integration, feature, and kernels must always be available",
            ));
        }
        let expected_kernel_status = if !self.feature_enabled {
            BackendKernelStatus::FeatureDisabled
        } else {
            match self.backend {
                BackendKind::Cpu => BackendKernelStatus::Cpu,
                BackendKind::Mps => BackendKernelStatus::Placeholder,
                BackendKind::Hip if !self.real_kernels_compiled => {
                    BackendKernelStatus::StubWithoutHipReal
                }
                BackendKind::Wgpu | BackendKind::Cuda if !self.real_kernels_compiled => {
                    return Err(invalid_payload(
                        "real_kernels_compiled",
                        "WGPU and CUDA runtime features require real kernels",
                    ));
                }
                BackendKind::Wgpu | BackendKind::Cuda | BackendKind::Hip => {
                    BackendKernelStatus::KernelWired
                }
            }
        };
        if self.kernel_status != expected_kernel_status {
            return Err(invalid_payload(
                "kernel_status",
                format!(
                    "backend '{}' requires status '{}', got '{}'",
                    self.backend.as_str(),
                    expected_kernel_status.as_str(),
                    self.kernel_status.as_str()
                ),
            ));
        }
        let build_ready = self.feature_enabled && self.real_kernels_compiled && !self.placeholder;
        let expected_ready = build_ready && self.runtime_initialized;
        if self.runtime_ready != expected_ready {
            return Err(invalid_payload(
                "runtime_ready",
                "must equal build readiness && runtime_initialized",
            ));
        }
        if self.feature_enabled && !self.integration_compiled {
            return Err(invalid_payload(
                "integration_compiled",
                "cannot be false when the executable runtime feature is enabled",
            ));
        }
        if self.backend == BackendKind::Cpu {
            if self.runtime_probe_attempted
                || !self.runtime_initialized
                || self.runtime_error.is_some()
                || self.runtime_status != BackendRuntimeStatus::Cpu
            {
                return Err(invalid_payload(
                    "runtime_status",
                    "CPU must be ready without an initialization probe",
                ));
            }
        } else if !build_ready {
            let expected_status =
                observed_backend_runtime_status(self.kernel_status, self.runtime_initialized);
            if self.runtime_probe_attempted
                || self.runtime_initialized
                || self.runtime_error.is_some()
                || self.runtime_status != expected_status
            {
                return Err(invalid_payload(
                    "runtime_status",
                    "a backend that is not build-ready cannot claim an initialization probe",
                ));
            }
        } else {
            if !self.runtime_probe_attempted {
                return Err(invalid_payload(
                    "runtime_probe_attempted",
                    "build-ready accelerator backends must be probed",
                ));
            }
            match (self.runtime_initialized, self.runtime_error.as_deref()) {
                (true, None) if self.runtime_status == BackendRuntimeStatus::Ready => {}
                (false, Some(error))
                    if !error.trim().is_empty()
                        && self.runtime_status == BackendRuntimeStatus::InitializationFailed => {}
                _ => {
                    return Err(invalid_payload(
                        "runtime_status",
                        "accelerator initialization state, error, and status disagree",
                    ));
                }
            }
        }
        let expected_recommendation = backend_runtime_recommendation_for_state(
            self.backend,
            self.feature_enabled,
            self.real_kernels_compiled,
        );
        if self.recommendation != expected_recommendation {
            return Err(invalid_payload(
                "recommendation",
                "must match the Rust-derived runtime state recommendation",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendResolution {
    pub effective_backend: BackendKind,
    pub reported_backend: BackendKind,
    pub mps_probe: Option<MpsProbeReport>,
}

impl BackendResolution {
    pub fn requested_backend(&self) -> BackendKind {
        self.reported_backend
    }

    fn to_transport_value(self) -> serde_json::Value {
        let requested = BackendRuntimeState::observe(self.reported_backend);
        let effective = BackendRuntimeState::observe(self.effective_backend);
        let mut payload = serde_json::Map::new();
        payload.insert("backend".into(), self.effective_backend.as_str().into());
        payload.insert(
            "requested_backend".into(),
            self.reported_backend.as_str().into(),
        );
        payload.insert("kind".into(), "st_core_backend_resolution".into());
        payload.insert(
            "effective_backend".into(),
            self.effective_backend.as_str().into(),
        );
        payload.insert(
            "reported_backend".into(),
            self.reported_backend.as_str().into(),
        );
        insert_backend_runtime_meta(&mut payload, "requested", &requested);
        insert_backend_runtime_meta(&mut payload, "effective", &effective);
        payload.insert("has_mps_probe".into(), self.mps_probe.is_some().into());
        if let Some(probe) = self.mps_probe.as_ref() {
            insert_mps_meta(&mut payload, probe);
        }
        serde_json::Value::Object(payload)
    }
}

/// Canonical inputs retained by every committed runtime-device probe.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeDeviceProbeRequest {
    pub requested_backend: BackendKind,
    /// Capability descriptor for the effective execution backend.
    pub caps: DeviceCaps,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mps_probe: Option<MpsProbeReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested_workgroup: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cols: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tile_hint: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compaction_hint: Option<u32>,
}

impl RuntimeDeviceProbeRequest {
    pub fn effective_backend(&self) -> BackendKind {
        self.caps.backend
    }

    pub fn validate(&self) -> Result<(), RuntimeDeviceProbeError> {
        self.caps.validate()?;
        for (field, value) in [
            ("requested_workgroup", self.requested_workgroup),
            ("cols", self.cols),
            ("tile_hint", self.tile_hint),
            ("compaction_hint", self.compaction_hint),
        ] {
            if value == Some(0) {
                return Err(RuntimeDeviceProbeError::ZeroInput { field });
            }
        }
        if self.cols.is_none() {
            if self.tile_hint.is_some() {
                return Err(RuntimeDeviceProbeError::HintWithoutColumns { field: "tile_hint" });
            }
            if self.compaction_hint.is_some() {
                return Err(RuntimeDeviceProbeError::HintWithoutColumns {
                    field: "compaction_hint",
                });
            }
        }

        match (self.requested_backend, self.mps_probe.as_ref()) {
            (BackendKind::Mps, Some(probe)) => {
                probe.validate()?;
                if self.caps.backend != probe.planner_surrogate_backend {
                    return Err(RuntimeDeviceProbeError::CapsBackendMismatch {
                        caps_backend: self.caps.backend.as_str(),
                        effective_backend: probe.planner_surrogate_backend.as_str(),
                    });
                }
            }
            (BackendKind::Mps, None) => {
                return Err(RuntimeDeviceProbeError::InvalidMpsOverlay {
                    backend: BackendKind::Mps.as_str(),
                    message: "MPS requests require the canonical MPS probe".to_owned(),
                });
            }
            (_, Some(_)) => {
                return Err(RuntimeDeviceProbeError::InvalidMpsOverlay {
                    backend: self.requested_backend.as_str(),
                    message: "only MPS requests may carry an MPS probe".to_owned(),
                });
            }
            (_, None) => {
                if self.caps.backend != self.requested_backend {
                    return Err(RuntimeDeviceProbeError::CapsBackendMismatch {
                        caps_backend: self.caps.backend.as_str(),
                        effective_backend: self.requested_backend.as_str(),
                    });
                }
            }
        }
        Ok(())
    }
}

/// Replayable Rust-owned runtime observation. Python and WASM may attach an
/// uncommitted `execution_client`, but every semantic field and derived metric
/// is bound by the two domain-separated commitments.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeDeviceProbePayload {
    pub kind: String,
    pub contract_version: String,
    pub semantic_owner: String,
    pub semantic_backend: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_client: Option<String>,
    pub request: RuntimeDeviceProbeRequest,
    pub requested_runtime: BackendRuntimeState,
    pub effective_runtime: BackendRuntimeState,
    pub aligned_workgroup: Option<u32>,
    pub occupancy_score: Option<f32>,
    pub preferred_tile: Option<u32>,
    pub preferred_compaction_tile: Option<u32>,
    pub route_evidence: RuntimeDeviceRouteEvidence,
    pub request_sha256: String,
    pub output_sha256: String,
    pub committed: bool,
}

/// Compatibility view retained for Rust callers of the original report API.
///
/// New callers that need replay, commitments, or runtime initialization facts
/// should use [`RuntimeDeviceProbePayload`]. This view is always projected from
/// that canonical payload, never evaluated independently.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeviceReport {
    pub reported_backend: BackendKind,
    pub caps: DeviceCaps,
    pub requested_workgroup: Option<u32>,
    pub aligned_workgroup: Option<u32>,
    pub occupancy_score: Option<f32>,
    pub preferred_tile: Option<u32>,
    pub preferred_compaction_tile: Option<u32>,
    pub mps_probe: Option<MpsProbeReport>,
}

impl DeviceReport {
    pub const fn effective_backend(self) -> BackendKind {
        self.caps.backend
    }
}

impl From<&RuntimeDeviceProbePayload> for DeviceReport {
    fn from(payload: &RuntimeDeviceProbePayload) -> Self {
        Self {
            reported_backend: payload.requested_backend(),
            caps: payload.caps(),
            requested_workgroup: payload.request.requested_workgroup,
            aligned_workgroup: payload.aligned_workgroup,
            occupancy_score: payload.occupancy_score,
            preferred_tile: payload.preferred_tile,
            preferred_compaction_tile: payload.preferred_compaction_tile,
            mps_probe: payload.request.mps_probe,
        }
    }
}

impl RuntimeDeviceProbePayload {
    pub fn requested_backend(&self) -> BackendKind {
        self.request.requested_backend
    }

    pub fn reported_backend(&self) -> BackendKind {
        self.requested_backend()
    }

    pub fn effective_backend(&self) -> BackendKind {
        self.request.effective_backend()
    }

    pub fn caps(&self) -> DeviceCaps {
        self.request.caps
    }

    /// Attach transport provenance without changing either semantic commitment.
    pub fn with_execution_client(
        mut self,
        execution_client: impl AsRef<str>,
    ) -> Result<Self, RuntimeDeviceProbeError> {
        self.execution_client = Some(normalized_execution_client(execution_client.as_ref())?);
        self.validate()?;
        Ok(self)
    }

    /// Validate identity, algebraic invariants, commitments, and full replay.
    pub fn validate(&self) -> Result<(), RuntimeDeviceProbeError> {
        for (field, actual, expected) in [
            ("kind", self.kind.as_str(), RUNTIME_DEVICE_PROBE_KIND),
            (
                "contract_version",
                self.contract_version.as_str(),
                RUNTIME_DEVICE_PROBE_CONTRACT_VERSION,
            ),
            (
                "semantic_owner",
                self.semantic_owner.as_str(),
                RUNTIME_DEVICE_PROBE_SEMANTIC_OWNER,
            ),
            (
                "semantic_backend",
                self.semantic_backend.as_str(),
                RUNTIME_DEVICE_PROBE_SEMANTIC_BACKEND,
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
        if !valid_sha256(&self.request_sha256) || !valid_sha256(&self.output_sha256) {
            return Err(invalid_payload(
                "commitment",
                "request_sha256 and output_sha256 must be lowercase SHA-256 values",
            ));
        }
        if !self.committed {
            return Err(invalid_payload(
                "committed",
                "runtime-device probe payloads must describe a committed observation",
            ));
        }
        self.request.validate()?;
        self.requested_runtime.validate()?;
        self.effective_runtime.validate()?;
        if self.requested_runtime.backend != self.requested_backend() {
            return Err(invalid_payload(
                "requested_runtime.backend",
                "must match request.requested_backend",
            ));
        }
        if self.effective_runtime.backend != self.effective_backend() {
            return Err(invalid_payload(
                "effective_runtime.backend",
                "must match request.caps.backend",
            ));
        }
        if self.requested_backend() == self.effective_backend() {
            if self.requested_runtime != self.effective_runtime {
                return Err(invalid_payload(
                    "runtime_state",
                    "direct routes must share one identical requested/effective observation",
                ));
            }
        } else {
            let probe = self
                .request
                .mps_probe
                .as_ref()
                .expect("request validation guarantees an MPS surrogate overlay");
            if self.requested_backend() != BackendKind::Mps
                || self.requested_runtime.feature_enabled != probe.feature_enabled
                || self.requested_runtime.placeholder != probe.placeholder()
                || self.requested_runtime.runtime_initialized != probe.initialized
            {
                return Err(invalid_payload(
                    "runtime_state",
                    "MPS native state must agree with its canonical probe overlay",
                ));
            }
            if !self.effective_runtime.runtime_ready {
                return Err(RuntimeDeviceProbeError::SurrogateNotReady {
                    backend: self.effective_backend().as_str(),
                });
            }
        }
        if self.occupancy_score.is_some_and(|value| !value.is_finite()) {
            return Err(RuntimeDeviceProbeError::NonFiniteDerived {
                field: "occupancy_score",
            });
        }
        if self.route_evidence != route_evidence_for(self) {
            return Err(invalid_payload(
                "route_evidence",
                "must match the Rust-derived runtime observation",
            ));
        }

        let expected_request_sha256 = runtime_device_probe_request_sha256(&self.request);
        if self.request_sha256 != expected_request_sha256 {
            return Err(invalid_payload(
                "request_sha256",
                "does not bind the embedded canonical request",
            ));
        }
        let (aligned, occupancy, tile, compaction_tile) = derive_probe_metrics(&self.request)?;
        if self.aligned_workgroup != aligned
            || self.occupancy_score != occupancy
            || self.preferred_tile != tile
            || self.preferred_compaction_tile != compaction_tile
        {
            return Err(invalid_payload(
                "derived_metrics",
                "workgroup, occupancy, or tile fields do not match the canonical request",
            ));
        }
        let expected_output_sha256 = runtime_device_probe_output_sha256(self);
        if self.output_sha256 != expected_output_sha256 {
            return Err(invalid_payload(
                "output_sha256",
                "does not bind the Rust-derived probe output",
            ));
        }
        Ok(())
    }

    pub fn validate_against(
        &self,
        request: RuntimeDeviceProbeRequest,
    ) -> Result<(), RuntimeDeviceProbeError> {
        self.validate()?;
        request.validate()?;
        if self.request != request {
            return Err(invalid_payload(
                "request",
                "payload does not match replay of the supplied runtime-device probe request",
            ));
        }
        Ok(())
    }

    /// Compatibility-rich wire projection. Canonical contract fields remain
    /// nested and committed; flat fields are aliases derived here in Rust.
    pub fn to_transport_value(&self) -> serde_json::Value {
        let canonical = serde_json::to_value(self)
            .expect("validated runtime-device probes must serialize deterministically");
        let mut value = canonical.clone();
        let object = value
            .as_object_mut()
            .expect("runtime-device probe payload serializes as an object");
        object.insert("contract".into(), canonical);
        object.insert("backend".into(), self.requested_backend().as_str().into());
        object.insert(
            "reported_backend".into(),
            self.requested_backend().as_str().into(),
        );
        object.insert(
            "requested_backend".into(),
            self.requested_backend().as_str().into(),
        );
        object.insert(
            "effective_backend".into(),
            self.effective_backend().as_str().into(),
        );
        insert_caps_meta(object, self.request.caps);
        insert_backend_runtime_meta(object, "requested", &self.requested_runtime);
        insert_backend_runtime_meta(object, "effective", &self.effective_runtime);
        object.insert(
            "runtime_status".into(),
            self.effective_runtime.runtime_status.as_str().into(),
        );
        object.insert(
            "runtime_ready".into(),
            self.effective_runtime.runtime_ready.into(),
        );
        object.insert(
            "runtime_probe_attempted".into(),
            self.effective_runtime.runtime_probe_attempted.into(),
        );
        object.insert(
            "runtime_initialized".into(),
            self.effective_runtime.runtime_initialized.into(),
        );
        object.insert(
            "runtime_error".into(),
            self.effective_runtime.runtime_error.clone().into(),
        );
        object.insert(
            "runtime_recommendation".into(),
            self.effective_runtime.recommendation.clone().into(),
        );
        for (field, value) in [
            ("requested_workgroup", self.request.requested_workgroup),
            ("cols", self.request.cols),
            ("tile_hint", self.request.tile_hint),
            ("compaction_hint", self.request.compaction_hint),
            ("aligned_workgroup", self.aligned_workgroup),
            ("preferred_tile", self.preferred_tile),
            ("preferred_compaction_tile", self.preferred_compaction_tile),
        ] {
            if let Some(value) = value {
                object.insert(field.into(), value.into());
            }
        }
        if let Some(score) = self.occupancy_score {
            object.insert("occupancy_score".into(), (score as f64).into());
        }
        object.insert(
            "has_mps_probe".into(),
            self.request.mps_probe.is_some().into(),
        );
        if let Some(probe) = self.request.mps_probe.as_ref() {
            insert_mps_meta(object, probe);
            object.insert(
                "planner_caps".into(),
                serde_json::to_value(probe.planner_caps)
                    .expect("validated device capabilities must serialize"),
            );
            object.insert("recommendation".into(), probe.recommendation().into());
            object.insert("error".into(), probe.error().into());
        } else if let Some(error) = self.effective_runtime.runtime_error.as_deref() {
            object.insert("error".into(), error.into());
        }
        value
    }
}

pub fn mps_probe() -> MpsProbeReport {
    let planner_surrogate_backend =
        if cfg!(target_os = "macos") && backend_runtime_ready(BackendKind::Wgpu) {
            BackendKind::Wgpu
        } else {
            BackendKind::Cpu
        };

    MpsProbeReport {
        feature_enabled: cfg!(feature = "mps"),
        platform_supported: cfg!(target_os = "macos"),
        host_class: if !cfg!(target_os = "macos") {
            MpsHostClass::NonMacHost
        } else if cfg!(target_arch = "aarch64") {
            MpsHostClass::AppleSiliconMac
        } else {
            MpsHostClass::IntelMac
        },
        backend_wired: false,
        initialized: false,
        planner_surrogate_backend,
        planner_caps: planner_surrogate_backend.default_caps(),
    }
}

/// Whether planner/integration code for a backend is compiled at all.
pub const fn backend_integration_compiled(kind: BackendKind) -> bool {
    match kind {
        BackendKind::Cpu => true,
        BackendKind::Wgpu => cfg!(feature = "wgpu"),
        BackendKind::Mps => cfg!(feature = "mps"),
        BackendKind::Cuda => cfg!(feature = "cuda"),
        BackendKind::Hip => cfg!(feature = "hip"),
    }
}

/// Whether the feature required by the executable backend runtime is enabled.
pub const fn backend_feature_enabled(kind: BackendKind) -> bool {
    match kind {
        BackendKind::Cpu => true,
        BackendKind::Wgpu => cfg!(feature = "wgpu-rt"),
        BackendKind::Mps => cfg!(feature = "mps"),
        BackendKind::Cuda => cfg!(feature = "cuda"),
        BackendKind::Hip => cfg!(feature = "hip"),
    }
}

pub const fn backend_real_kernels_compiled(kind: BackendKind) -> bool {
    match kind {
        BackendKind::Cpu => true,
        BackendKind::Wgpu => cfg!(feature = "wgpu-rt"),
        BackendKind::Mps => false,
        BackendKind::Cuda => cfg!(feature = "cuda"),
        BackendKind::Hip => cfg!(all(feature = "hip", feature = "hip-real")),
    }
}

pub const fn backend_placeholder(kind: BackendKind) -> bool {
    matches!(kind, BackendKind::Mps)
}

const fn observed_backend_kernel_status(
    kind: BackendKind,
    feature_enabled: bool,
) -> BackendKernelStatus {
    if !feature_enabled {
        return BackendKernelStatus::FeatureDisabled;
    }
    match kind {
        BackendKind::Cpu => BackendKernelStatus::Cpu,
        BackendKind::Wgpu | BackendKind::Cuda => BackendKernelStatus::KernelWired,
        BackendKind::Mps => BackendKernelStatus::Placeholder,
        BackendKind::Hip => {
            if cfg!(feature = "hip-real") {
                BackendKernelStatus::KernelWired
            } else {
                BackendKernelStatus::StubWithoutHipReal
            }
        }
    }
}

const fn observed_backend_runtime_status(
    kernel_status: BackendKernelStatus,
    initialized: bool,
) -> BackendRuntimeStatus {
    match kernel_status {
        BackendKernelStatus::FeatureDisabled => BackendRuntimeStatus::FeatureDisabled,
        BackendKernelStatus::Cpu => BackendRuntimeStatus::Cpu,
        BackendKernelStatus::Placeholder => BackendRuntimeStatus::Placeholder,
        BackendKernelStatus::StubWithoutHipReal => BackendRuntimeStatus::StubWithoutHipReal,
        BackendKernelStatus::KernelWired => {
            if initialized {
                BackendRuntimeStatus::Ready
            } else {
                BackendRuntimeStatus::InitializationFailed
            }
        }
    }
}

fn probe_backend_initialization(
    backend: BackendKind,
    build_ready: bool,
) -> (bool, bool, Option<String>) {
    if backend == BackendKind::Cpu {
        return (false, true, None);
    }
    if !build_ready {
        return (false, false, None);
    }

    let result = match backend {
        BackendKind::Wgpu => probe_wgpu_runtime(),
        BackendKind::Cuda => probe_cuda_runtime(),
        BackendKind::Hip => probe_hip_runtime(),
        BackendKind::Cpu => Ok(()),
        BackendKind::Mps => Err("native MPS kernels are not wired".to_owned()),
    };
    match result {
        Ok(()) => (true, true, None),
        Err(error) => (true, false, Some(error)),
    }
}

fn probe_wgpu_runtime() -> Result<(), String> {
    #[cfg(all(feature = "wgpu-rt", not(target_arch = "wasm32")))]
    {
        return crate::backend::wgpu_rt::ensure_default_ctx().map(|_| ());
    }
    #[cfg(all(feature = "wgpu-rt", target_arch = "wasm32"))]
    {
        return crate::backend::wgpu_rt::installed_ctx()
            .is_some()
            .then_some(())
            .ok_or_else(|| "WGPU context is not installed in this WASM runtime".to_owned());
    }
    #[cfg(not(feature = "wgpu-rt"))]
    {
        Err("WGPU runtime feature is not compiled".to_owned())
    }
}

fn probe_cuda_runtime() -> Result<(), String> {
    #[cfg(feature = "cuda")]
    {
        return crate::backend::cuda_loader::probe_device();
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err("CUDA runtime feature is not compiled".to_owned())
    }
}

fn probe_hip_runtime() -> Result<(), String> {
    #[cfg(all(feature = "hip", feature = "hip-real"))]
    {
        let probe = st_backend_hip::probe();
        if probe.available && probe.initialized {
            return Ok(());
        }
        return Err(probe
            .error
            .unwrap_or_else(|| "HIP runtime reported no initialized device".to_owned()));
    }
    #[cfg(not(all(feature = "hip", feature = "hip-real")))]
    {
        Err("HIP real runtime feature is not compiled".to_owned())
    }
}

pub fn backend_runtime_status(kind: BackendKind) -> &'static str {
    BackendRuntimeState::observe(kind).runtime_status.as_str()
}

pub fn backend_runtime_ready(kind: BackendKind) -> bool {
    BackendRuntimeState::observe(kind).runtime_ready
}

pub const fn backend_runtime_recommendation(kind: BackendKind) -> &'static str {
    backend_runtime_recommendation_for_state(
        kind,
        backend_feature_enabled(kind),
        backend_real_kernels_compiled(kind),
    )
}

const fn backend_runtime_recommendation_for_state(
    kind: BackendKind,
    feature_enabled: bool,
    real_kernels_compiled: bool,
) -> &'static str {
    match kind {
        BackendKind::Cpu => "cpu backend is available",
        BackendKind::Wgpu => {
            if feature_enabled {
                "wgpu rank runtime can be initialized by the training entrypoint"
            } else {
                "rebuild with the 'wgpu-rt' feature to enable WGPU training kernels"
            }
        }
        BackendKind::Mps => {
            "mps is a placeholder; follow planner_surrogate_backend from the canonical runtime probe"
        }
        BackendKind::Cuda => {
            if feature_enabled {
                "cuda rank kernels are compiled; verify tensor-kernel trace counters during training"
            } else {
                "rebuild with the cuda feature to request CUDA"
            }
        }
        BackendKind::Hip => {
            if feature_enabled && real_kernels_compiled {
                "hip-real kernels are compiled; verify tensor-kernel trace counters during training"
            } else if feature_enabled {
                "hip feature is compiled as a stub; rebuild with hip-real for real HIP kernels"
            } else {
                "rebuild with the hip feature to request HIP"
            }
        }
    }
}

pub fn resolve_backend(requested_backend: BackendKind) -> BackendResolution {
    let resolution = match requested_backend {
        BackendKind::Mps => {
            let probe = mps_probe();
            BackendResolution {
                effective_backend: probe.planner_surrogate_backend,
                reported_backend: BackendKind::Mps,
                mps_probe: Some(probe),
            }
        }
        _ => BackendResolution {
            effective_backend: requested_backend,
            reported_backend: requested_backend,
            mps_probe: None,
        },
    };
    emit_backend_resolution_meta(&resolution);
    resolution
}

/// Evaluate one canonical probe request and return a committed, replayable payload.
pub fn evaluate_runtime_device_probe(
    request: RuntimeDeviceProbeRequest,
) -> Result<RuntimeDeviceProbePayload, RuntimeDeviceProbeError> {
    let payload = build_runtime_device_probe(request)?;
    payload.validate()?;
    emit_device_report_meta(&payload);
    Ok(payload)
}

/// Compatibility entrypoint retained for existing Rust callers.
///
/// Valid legacy inputs are evaluated by the committed probe first. Invalid
/// capability descriptors now fail loudly rather than being silently
/// normalized; structured ingress should use [`evaluate_runtime_device_probe`]
/// to receive a typed error.
#[allow(clippy::too_many_arguments)]
pub fn build_device_report(
    reported_backend: BackendKind,
    caps: DeviceCaps,
    mps_probe: Option<MpsProbeReport>,
    requested_workgroup: Option<u32>,
    cols: Option<u32>,
    tile_hint: Option<u32>,
    compaction_hint: Option<u32>,
) -> DeviceReport {
    let payload = evaluate_runtime_device_probe(RuntimeDeviceProbeRequest {
        requested_backend: reported_backend,
        caps,
        mps_probe,
        requested_workgroup,
        cols,
        tile_hint,
        compaction_hint,
    })
    .expect("build_device_report requires a valid runtime-device probe request");
    DeviceReport::from(&payload)
}

fn build_runtime_device_probe(
    request: RuntimeDeviceProbeRequest,
) -> Result<RuntimeDeviceProbePayload, RuntimeDeviceProbeError> {
    request.validate()?;
    if request.requested_backend == BackendKind::Mps
        && request.mps_probe.as_ref() != Some(&mps_probe())
    {
        return Err(invalid_mps_overlay(
            "generation requires the current Rust-observed MPS host and surrogate state",
        ));
    }
    let requested_runtime = BackendRuntimeState::observe(request.requested_backend);
    let effective_runtime = if request.requested_backend == request.effective_backend() {
        requested_runtime.clone()
    } else {
        BackendRuntimeState::observe(request.effective_backend())
    };
    if request.requested_backend == BackendKind::Mps && !effective_runtime.runtime_ready {
        return Err(RuntimeDeviceProbeError::SurrogateNotReady {
            backend: request.effective_backend().as_str(),
        });
    }

    let (aligned_workgroup, occupancy_score, preferred_tile, preferred_compaction_tile) =
        derive_probe_metrics(&request)?;

    let request_sha256 = runtime_device_probe_request_sha256(&request);
    let mut payload = RuntimeDeviceProbePayload {
        kind: RUNTIME_DEVICE_PROBE_KIND.to_owned(),
        contract_version: RUNTIME_DEVICE_PROBE_CONTRACT_VERSION.to_owned(),
        semantic_owner: RUNTIME_DEVICE_PROBE_SEMANTIC_OWNER.to_owned(),
        semantic_backend: RUNTIME_DEVICE_PROBE_SEMANTIC_BACKEND.to_owned(),
        execution_client: None,
        request,
        requested_runtime,
        effective_runtime,
        aligned_workgroup,
        occupancy_score,
        preferred_tile,
        preferred_compaction_tile,
        route_evidence: RuntimeDeviceRouteEvidence::default(),
        request_sha256,
        output_sha256: String::new(),
        committed: true,
    };
    payload.route_evidence = route_evidence_for(&payload);
    payload.output_sha256 = runtime_device_probe_output_sha256(&payload);
    Ok(payload)
}

type DerivedProbeMetrics = (Option<u32>, Option<f32>, Option<u32>, Option<u32>);

fn derive_probe_metrics(
    request: &RuntimeDeviceProbeRequest,
) -> Result<DerivedProbeMetrics, RuntimeDeviceProbeError> {
    let (aligned_workgroup, occupancy_score) = if let Some(requested) = request.requested_workgroup
    {
        let score = request.caps.occupancy_score(requested);
        if !score.is_finite() {
            return Err(RuntimeDeviceProbeError::NonFiniteDerived {
                field: "occupancy_score",
            });
        }
        (Some(request.caps.align_workgroup(requested)), Some(score))
    } else {
        (None, None)
    };
    let (preferred_tile, preferred_compaction_tile) = if let Some(cols) = request.cols {
        (
            Some(
                request
                    .caps
                    .preferred_tile(cols, request.tile_hint.unwrap_or(0)),
            ),
            Some(
                request
                    .caps
                    .preferred_compaction_tile(cols, request.compaction_hint.unwrap_or(0)),
            ),
        )
    } else {
        (None, None)
    };
    Ok((
        aligned_workgroup,
        occupancy_score,
        preferred_tile,
        preferred_compaction_tile,
    ))
}

fn route_evidence_for(payload: &RuntimeDeviceProbePayload) -> RuntimeDeviceRouteEvidence {
    let mps_probe = payload.request.mps_probe.as_ref();
    RuntimeDeviceRouteEvidence {
        requested_backend: payload.requested_backend().as_str().to_owned(),
        effective_backend: Some(payload.effective_backend().as_str().to_owned()),
        runtime_ready: Some(payload.effective_runtime.runtime_ready),
        requested_backend_runtime_ready: Some(payload.requested_runtime.runtime_ready),
        effective_backend_runtime_ready: Some(payload.effective_runtime.runtime_ready),
        available: Some(payload.requested_runtime.runtime_ready),
        runtime_status: Some(payload.effective_runtime.runtime_status.as_str().to_owned()),
        requested_backend_runtime_status: Some(
            payload.requested_runtime.runtime_status.as_str().to_owned(),
        ),
        effective_backend_runtime_status: Some(
            payload.effective_runtime.runtime_status.as_str().to_owned(),
        ),
        status: mps_probe.map(|probe| probe.status().as_str().to_owned()),
        error: payload
            .effective_runtime
            .runtime_error
            .clone()
            .or_else(|| mps_probe.map(|probe| probe.error().to_owned())),
    }
}

fn emit_backend_resolution_meta(resolution: &BackendResolution) {
    emit_tensor_op("backend_resolution", &[1], &[1]);
    let value = (*resolution).to_transport_value();
    emit_tensor_op_meta("backend_resolution", move || value);
}

fn emit_device_report_meta(report: &RuntimeDeviceProbePayload) {
    emit_tensor_op("backend_device_report", &[1, 6], &[1, 8]);
    let mut value = report.to_transport_value();
    let object = value
        .as_object_mut()
        .expect("runtime-device probe transport is an object");
    object.insert("contract_kind".into(), RUNTIME_DEVICE_PROBE_KIND.into());
    object.insert("kind".into(), "st_core_backend_device_report".into());
    emit_tensor_op_meta("backend_device_report", move || value);
}

fn insert_backend_runtime_meta(
    payload: &mut serde_json::Map<String, serde_json::Value>,
    prefix: &str,
    state: &BackendRuntimeState,
) {
    payload.insert(
        format!("{prefix}_backend_integration_compiled"),
        state.integration_compiled.into(),
    );
    payload.insert(
        format!("{prefix}_backend_feature_enabled"),
        state.feature_enabled.into(),
    );
    payload.insert(
        format!("{prefix}_backend_real_kernels_compiled"),
        state.real_kernels_compiled.into(),
    );
    payload.insert(
        format!("{prefix}_backend_placeholder"),
        state.placeholder.into(),
    );
    payload.insert(
        format!("{prefix}_backend_kernel_status"),
        state.kernel_status.as_str().into(),
    );
    payload.insert(
        format!("{prefix}_backend_runtime_probe_attempted"),
        state.runtime_probe_attempted.into(),
    );
    payload.insert(
        format!("{prefix}_backend_runtime_initialized"),
        state.runtime_initialized.into(),
    );
    payload.insert(
        format!("{prefix}_backend_runtime_status"),
        state.runtime_status.as_str().into(),
    );
    payload.insert(
        format!("{prefix}_backend_runtime_ready"),
        state.runtime_ready.into(),
    );
    payload.insert(
        format!("{prefix}_backend_runtime_recommendation"),
        state.recommendation.clone().into(),
    );
    payload.insert(
        format!("{prefix}_backend_runtime_error"),
        state.runtime_error.clone().into(),
    );
}

fn insert_caps_meta(payload: &mut serde_json::Map<String, serde_json::Value>, caps: DeviceCaps) {
    payload.insert("subgroup".into(), caps.subgroup.into());
    payload.insert("lane_width".into(), caps.lane_width.into());
    payload.insert("max_workgroup".into(), caps.max_workgroup.into());
    payload.insert(
        "shared_mem_per_workgroup".into(),
        caps.shared_mem_per_workgroup.into(),
    );
}

fn insert_mps_meta(
    payload: &mut serde_json::Map<String, serde_json::Value>,
    probe: &MpsProbeReport,
) {
    payload.insert("mps_status".into(), probe.status().as_str().into());
    payload.insert("mps_host_class".into(), probe.host_class.as_str().into());
    payload.insert("mps_placeholder".into(), probe.placeholder().into());
    payload.insert("mps_available".into(), probe.available().into());
    payload.insert("mps_backend_wired".into(), probe.backend_wired.into());
    payload.insert("mps_initialized".into(), probe.initialized.into());
    payload.insert("mps_planner_route".into(), probe.planner_route().into());
    payload.insert(
        "mps_recommended_backend".into(),
        probe.recommended_backend().as_str().into(),
    );
    payload.insert("host_os".into(), probe.host_os().into());
    payload.insert("host_arch".into(), probe.host_arch().into());

    // Unprefixed aliases preserve the established Python probe surface.
    payload.insert("status".into(), probe.status().as_str().into());
    payload.insert("feature_enabled".into(), probe.feature_enabled.into());
    payload.insert("platform_supported".into(), probe.platform_supported.into());
    payload.insert("host_class".into(), probe.host_class.as_str().into());
    payload.insert("backend_wired".into(), probe.backend_wired.into());
    payload.insert("placeholder".into(), probe.placeholder().into());
    payload.insert("available".into(), probe.available().into());
    payload.insert("initialized".into(), probe.initialized.into());
    payload.insert(
        "planner_surrogate_backend".into(),
        probe.planner_surrogate_backend.as_str().into(),
    );
    payload.insert("planner_route".into(), probe.planner_route().into());
    payload.insert(
        "recommended_backend".into(),
        probe.recommended_backend().as_str().into(),
    );
}

fn runtime_device_probe_request_sha256(request: &RuntimeDeviceProbeRequest) -> String {
    sha256_serialized(RUNTIME_DEVICE_PROBE_REQUEST_DIGEST_DOMAIN, request)
}

fn runtime_device_probe_output_sha256(payload: &RuntimeDeviceProbePayload) -> String {
    let mut canonical = payload.clone();
    canonical.execution_client = None;
    canonical.output_sha256.clear();
    sha256_serialized(RUNTIME_DEVICE_PROBE_OUTPUT_DIGEST_DOMAIN, &canonical)
}

fn sha256_serialized<T: Serialize>(domain: &[u8], value: &T) -> String {
    let encoded = serde_json::to_vec(value)
        .expect("runtime-device probe commitment values must serialize deterministically");
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

fn normalized_execution_client(value: &str) -> Result<String, RuntimeDeviceProbeError> {
    let normalized = value.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return Err(invalid_payload("execution_client", "must not be empty"));
    }
    if normalized.len() > RUNTIME_DEVICE_PROBE_MAX_CLIENT_BYTES {
        return Err(invalid_payload(
            "execution_client",
            format!(
                "has {} bytes, exceeding limit {}",
                normalized.len(),
                RUNTIME_DEVICE_PROBE_MAX_CLIENT_BYTES
            ),
        ));
    }
    if !normalized.bytes().enumerate().all(|(index, byte)| {
        byte.is_ascii_alphanumeric()
            || (index > 0 && matches!(byte, b'.' | b'_' | b':' | b'+' | b'-' | b'/'))
    }) {
        return Err(invalid_payload(
            "execution_client",
            "contains unsupported characters",
        ));
    }
    Ok(normalized)
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn invalid_payload(field: &'static str, message: impl ToString) -> RuntimeDeviceProbeError {
    RuntimeDeviceProbeError::InvalidPayload {
        field,
        message: message.to_string(),
    }
}

fn invalid_mps_overlay(message: impl ToString) -> RuntimeDeviceProbeError {
    RuntimeDeviceProbeError::InvalidMpsOverlay {
        backend: BackendKind::Mps.as_str(),
        message: message.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn request_for(backend: BackendKind) -> RuntimeDeviceProbeRequest {
        let resolution = resolve_backend(backend);
        RuntimeDeviceProbeRequest {
            requested_backend: resolution.reported_backend,
            caps: resolution.effective_backend.default_caps(),
            mps_probe: resolution.mps_probe,
            requested_workgroup: Some(300),
            cols: Some(4096),
            tile_hint: None,
            compaction_hint: None,
        }
    }

    #[test]
    fn mps_probe_reports_honest_runtime_ready_surrogate() {
        let probe = mps_probe();
        probe.validate().unwrap();
        assert!(!probe.backend_wired);
        assert!(!probe.initialized);
        assert!(probe.placeholder());
        assert!(!probe.available());
        assert_eq!(probe.planner_caps.backend, probe.planner_surrogate_backend);
        assert_eq!(probe.recommended_backend(), probe.planner_surrogate_backend);
        assert_eq!(probe.host_os(), std::env::consts::OS);
        assert_eq!(probe.host_arch(), std::env::consts::ARCH);

        if !cfg!(feature = "mps") {
            assert_eq!(probe.status(), MpsProbeStatus::BuildFeatureDisabled);
        } else if !cfg!(target_os = "macos") {
            assert_eq!(probe.status(), MpsProbeStatus::UnsupportedHost);
        } else {
            assert_eq!(probe.status(), MpsProbeStatus::Placeholder);
        }

        let wgpu_ready = BackendRuntimeState::observe(BackendKind::Wgpu).runtime_ready;
        if cfg!(target_os = "macos") && wgpu_ready {
            assert_eq!(probe.planner_surrogate_backend, BackendKind::Wgpu);
            assert_eq!(probe.planner_route(), "metal-via-wgpu");
        } else {
            assert_eq!(probe.planner_surrogate_backend, BackendKind::Cpu);
            assert_eq!(probe.planner_route(), "cpu-fallback");
        }
        assert!(backend_runtime_ready(probe.planner_surrogate_backend));
    }

    #[test]
    fn backend_state_distinguishes_integration_from_executable_runtime() {
        let state = BackendRuntimeState::observe(BackendKind::Wgpu);
        state.validate().unwrap();
        assert_eq!(state.integration_compiled, cfg!(feature = "wgpu"));
        assert_eq!(state.feature_enabled, cfg!(feature = "wgpu-rt"));
        assert_eq!(
            state.kernel_status,
            if cfg!(feature = "wgpu-rt") {
                BackendKernelStatus::KernelWired
            } else {
                BackendKernelStatus::FeatureDisabled
            }
        );
        if cfg!(feature = "wgpu-rt") {
            assert!(state.runtime_probe_attempted);
            assert_eq!(state.runtime_ready, state.runtime_initialized);
            assert_eq!(
                state.runtime_status,
                if state.runtime_initialized {
                    BackendRuntimeStatus::Ready
                } else {
                    BackendRuntimeStatus::InitializationFailed
                }
            );
        } else {
            assert!(!state.runtime_probe_attempted);
            assert!(!state.runtime_initialized);
            assert!(!state.runtime_ready);
            assert_eq!(state.runtime_status, BackendRuntimeStatus::FeatureDisabled);
        }
        assert_eq!(
            backend_runtime_status(BackendKind::Wgpu),
            state.runtime_status.as_str()
        );

        let cpu = BackendRuntimeState::observe(BackendKind::Cpu);
        cpu.validate().unwrap();
        assert!(cpu.integration_compiled);
        assert!(cpu.runtime_ready);
        assert_eq!(cpu.runtime_status, BackendRuntimeStatus::Cpu);
    }

    #[test]
    fn device_probe_is_committed_replayable_and_route_ready() {
        let request = request_for(BackendKind::Mps);
        let payload = evaluate_runtime_device_probe(request.clone()).unwrap();
        payload.validate().unwrap();
        payload.validate_against(request).unwrap();
        assert_eq!(payload.kind, RUNTIME_DEVICE_PROBE_KIND);
        assert_eq!(
            payload.contract_version,
            RUNTIME_DEVICE_PROBE_CONTRACT_VERSION
        );
        assert_eq!(payload.semantic_owner, RUNTIME_DEVICE_PROBE_SEMANTIC_OWNER);
        assert_eq!(payload.requested_backend(), BackendKind::Mps);
        assert!(payload.effective_runtime.runtime_ready);
        assert!(!payload.requested_runtime.runtime_ready);
        assert_eq!(payload.route_evidence.runtime_ready, Some(true));
        assert_eq!(
            payload.route_evidence.requested_backend_runtime_ready,
            Some(false)
        );
        assert_eq!(payload.request_sha256.len(), 64);
        assert_eq!(payload.output_sha256.len(), 64);
        assert!(payload.committed);

        let transport = payload.to_transport_value();
        assert_eq!(transport["backend"], "mps");
        assert_eq!(
            transport["effective_backend"],
            payload.effective_backend().as_str()
        );
        assert_eq!(transport["runtime_ready"], true);
        assert_eq!(transport["requested_backend_runtime_ready"], false);
        assert_eq!(
            transport["aligned_workgroup"],
            payload.aligned_workgroup.unwrap()
        );
        assert!(transport["occupancy_score"].as_f64().unwrap().is_finite());
    }

    #[test]
    fn probe_generation_rejects_a_forged_mps_observation() {
        let mut request = request_for(BackendKind::Mps);
        let probe = request.mps_probe.as_mut().unwrap();
        if probe.platform_supported {
            probe.host_class = match probe.host_class {
                MpsHostClass::AppleSiliconMac => MpsHostClass::IntelMac,
                MpsHostClass::IntelMac => MpsHostClass::AppleSiliconMac,
                MpsHostClass::NonMacHost => unreachable!("supported MPS host must be macOS"),
            };
        } else {
            probe.platform_supported = true;
            probe.host_class = MpsHostClass::IntelMac;
        }
        request
            .validate()
            .expect("forged observation is self-consistent");
        assert!(matches!(
            evaluate_runtime_device_probe(request),
            Err(RuntimeDeviceProbeError::InvalidMpsOverlay { message, .. })
                if message.contains("current Rust-observed MPS host")
        ));
    }

    #[test]
    fn device_probe_rejects_tampering_and_invalid_inputs() {
        let request = request_for(BackendKind::Cpu);
        let payload = evaluate_runtime_device_probe(request.clone()).unwrap();
        let mut tampered = payload.clone();
        tampered.aligned_workgroup = Some(1);
        assert!(tampered.validate().is_err());

        let mut drifted_state = payload.clone();
        drifted_state.requested_runtime.recommendation = "trust the client".to_owned();
        drifted_state.effective_runtime.recommendation = "trust the client".to_owned();
        assert!(matches!(
            drifted_state.validate(),
            Err(RuntimeDeviceProbeError::InvalidPayload {
                field: "recommendation",
                ..
            })
        ));

        let mut different = request.clone();
        different.requested_workgroup = Some(64);
        assert!(payload.validate_against(different).is_err());

        let mut zero = request;
        zero.requested_workgroup = Some(0);
        assert_eq!(
            evaluate_runtime_device_probe(zero),
            Err(RuntimeDeviceProbeError::ZeroInput {
                field: "requested_workgroup"
            })
        );
    }

    #[test]
    fn build_device_report_preserves_metrics_and_compatibility_alias() {
        let resolution = resolve_backend(BackendKind::Cpu);
        let caps = resolution.effective_backend.default_caps();
        let report = build_device_report(
            resolution.reported_backend,
            caps,
            resolution.mps_probe,
            Some(300),
            Some(4096),
            None,
            None,
        );
        assert_eq!(report.reported_backend, BackendKind::Cpu);
        assert_eq!(report.effective_backend(), caps.backend);
        assert_eq!(report.caps, caps);
        assert_eq!(report.aligned_workgroup, Some(caps.align_workgroup(300)));
        assert_eq!(report.occupancy_score, Some(caps.occupancy_score(300)));
        assert_eq!(report.preferred_tile, Some(caps.preferred_tile(4096, 0)));
        assert_eq!(
            report.preferred_compaction_tile,
            Some(caps.preferred_compaction_tile(4096, 0))
        );
    }

    #[test]
    fn backend_availability_reports_feature_and_kernel_truth() {
        assert!(backend_feature_enabled(BackendKind::Cpu));
        assert!(backend_real_kernels_compiled(BackendKind::Cpu));
        assert!(!backend_placeholder(BackendKind::Cpu));
        assert_eq!(backend_runtime_status(BackendKind::Cpu), "cpu");
        assert!(backend_runtime_ready(BackendKind::Cpu));

        assert!(!backend_real_kernels_compiled(BackendKind::Mps));
        assert!(backend_placeholder(BackendKind::Mps));
        assert!(!backend_runtime_ready(BackendKind::Mps));
        assert_eq!(
            backend_feature_enabled(BackendKind::Mps),
            cfg!(feature = "mps")
        );

        assert_eq!(
            backend_real_kernels_compiled(BackendKind::Hip),
            cfg!(all(feature = "hip", feature = "hip-real"))
        );
        let hip = BackendRuntimeState::observe(BackendKind::Hip);
        assert_eq!(
            hip.runtime_ready,
            cfg!(all(feature = "hip", feature = "hip-real")) && hip.runtime_initialized
        );
    }

    #[test]
    fn backend_resolution_and_device_report_emit_committed_meta() {
        let _lock = crate::telemetry::tensor_observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let resolution = resolve_backend(BackendKind::Mps);
        let report = build_device_report(
            resolution.reported_backend,
            resolution.effective_backend.default_caps(),
            resolution.mps_probe,
            Some(300),
            Some(4096),
            None,
            None,
        );
        st_tensor::set_thread_meta_observer(previous);

        let events = events.lock().unwrap();
        let resolution_meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "backend_resolution"
                    && data["kind"] == "st_core_backend_resolution"
                    && data["reported_backend"] == "mps"
            })
            .expect("backend resolution metadata event");
        assert_eq!(resolution_meta.1["requested_backend_runtime_ready"], false);
        assert_eq!(resolution_meta.1["effective_backend_runtime_ready"], true);

        let report_meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "backend_device_report"
                    && data["kind"] == "st_core_backend_device_report"
                    && data["reported_backend"] == "mps"
                    && data["requested_workgroup"] == 300
            })
            .expect("backend device report metadata event");
        assert_eq!(report_meta.1["contract_kind"], RUNTIME_DEVICE_PROBE_KIND);
        assert_eq!(
            report_meta.1["contract_version"],
            RUNTIME_DEVICE_PROBE_CONTRACT_VERSION
        );
        assert_eq!(report_meta.1["committed"], true);
        assert_eq!(report_meta.1["output_sha256"].as_str().unwrap().len(), 64);
        assert_eq!(report_meta.1["lane_width"], report.caps.lane_width);
    }
}
