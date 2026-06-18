// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

use serde::Serialize;
use st_core::backend::device_caps::{BackendKind, DeviceCaps};
use st_core::ops::rank_entry::RankPlan;
use st_nn::{BackendPolicy, RoundtableSchedule, TensorError};

#[derive(Debug, Clone)]
pub struct BackendSelection {
    pub label: String,
    pub caps: DeviceCaps,
}

#[derive(Debug, Clone, Serialize)]
pub struct DeviceCapsMeta {
    pub backend: String,
    pub lane_width: u32,
    pub subgroup: bool,
    pub max_workgroup: u32,
    pub shared_mem_per_workgroup: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
#[allow(dead_code)]
pub struct TensorBackendPolicyMeta {
    pub device_backend: String,
    pub matmul_backend: String,
    pub prepacked_matmul_backend: String,
    pub layer_norm_backend: String,
    pub attention_backend: String,
    pub softmax_backend: String,
    pub tensor_util_backend: String,
}

#[derive(Debug, Clone, Serialize)]
#[allow(dead_code)]
pub struct RankChoiceMeta {
    pub subgroup: bool,
    pub use_2ce: bool,
    pub workgroup: u32,
    pub lanes: u32,
    pub channel_stride: u32,
    pub tile: u32,
    pub compaction_tile: u32,
    pub fft_tile: u32,
    pub fft_radix: u32,
    pub fft_segments: u32,
}

#[derive(Debug, Clone, Serialize)]
#[allow(dead_code)]
pub struct RankPlanBackendAudit {
    pub band: String,
    pub kind: String,
    pub rows: u32,
    pub cols: u32,
    pub k: u32,
    pub choice: RankChoiceMeta,
    pub wgpu_exact_shape_supported: bool,
    pub wgpu_exact_runtime_ready: bool,
    pub wgpu_exact_status: String,
    pub wgpu_exact_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[allow(dead_code)]
pub struct RoundtableBackendAudit {
    pub requested_backend: String,
    pub wgpu_runtime_compiled: bool,
    pub wgpu_runtime_context_installed: bool,
    pub any_wgpu_exact_runtime_ready: bool,
    pub bands: Vec<RankPlanBackendAudit>,
}

#[derive(Debug, Clone, Serialize)]
#[allow(dead_code)]
pub struct BackendRuntimeMeta {
    pub requested_backend: String,
    pub requested_backend_feature_enabled: bool,
    pub requested_backend_kernels_wired: bool,
    pub requested_backend_placeholder: bool,
    pub requested_backend_status: String,
    pub requested_backend_recommendation: String,
    pub wgpu_feature_compiled: bool,
    pub wgpu_rank_runtime_requested: bool,
    pub wgpu_rank_runtime_compiled: bool,
    pub wgpu_rank_runtime_context_installed: bool,
    pub wgpu_rank_runtime_initialized: bool,
    pub wgpu_rank_runtime_error: Option<String>,
    pub cuda_feature_compiled: bool,
    pub cuda_kernels_compiled: bool,
    pub hip_feature_compiled: bool,
    pub hip_real_compiled: bool,
    pub hip_kernels_compiled: bool,
    pub mps_feature_compiled: bool,
    pub mps_placeholder: bool,
}

impl From<DeviceCaps> for DeviceCapsMeta {
    fn from(caps: DeviceCaps) -> Self {
        Self {
            backend: backend_label(caps.backend).to_string(),
            lane_width: caps.lane_width,
            subgroup: caps.subgroup,
            max_workgroup: caps.max_workgroup,
            shared_mem_per_workgroup: caps.shared_mem_per_workgroup,
        }
    }
}

#[allow(dead_code)]
pub fn prepare_backend_runtime(
    selection: &BackendSelection,
) -> Result<BackendRuntimeMeta, TensorError> {
    let wgpu_rank_runtime_requested = matches!(selection.caps.backend, BackendKind::Wgpu);
    let mut wgpu_rank_runtime_initialized = false;
    let mut wgpu_rank_runtime_error = None;
    if wgpu_rank_runtime_requested {
        match ensure_wgpu_rank_runtime() {
            Ok(initialized) => wgpu_rank_runtime_initialized = initialized,
            Err(err) => {
                wgpu_rank_runtime_error = Some(err.to_string());
            }
        }
    }

    Ok(BackendRuntimeMeta {
        requested_backend: selection.label.clone(),
        requested_backend_feature_enabled: backend_feature_enabled(selection.caps.backend),
        requested_backend_kernels_wired: backend_kernels_compiled(selection.caps.backend),
        requested_backend_placeholder: backend_placeholder(selection.caps.backend),
        requested_backend_status: backend_runtime_status(selection.caps.backend).to_string(),
        requested_backend_recommendation: backend_runtime_recommendation(selection.caps.backend)
            .to_string(),
        wgpu_feature_compiled: wgpu_runtime_compiled(),
        wgpu_rank_runtime_requested,
        wgpu_rank_runtime_compiled: wgpu_runtime_compiled(),
        wgpu_rank_runtime_context_installed: wgpu_runtime_context_installed(),
        wgpu_rank_runtime_initialized,
        wgpu_rank_runtime_error,
        cuda_feature_compiled: cuda_feature_compiled(),
        cuda_kernels_compiled: cuda_kernels_compiled(),
        hip_feature_compiled: hip_feature_compiled(),
        hip_real_compiled: hip_real_compiled(),
        hip_kernels_compiled: hip_kernels_compiled(),
        mps_feature_compiled: mps_feature_compiled(),
        mps_placeholder: mps_placeholder(),
    })
}

#[allow(dead_code)]
pub fn tensor_backend_policy_meta(caps: DeviceCaps) -> TensorBackendPolicyMeta {
    let policy = BackendPolicy::from_device_caps(caps);
    TensorBackendPolicyMeta {
        device_backend: policy.device_backend_label().to_string(),
        matmul_backend: policy.matmul_backend_label().to_string(),
        prepacked_matmul_backend: policy.prepacked_matmul_backend_label().to_string(),
        layer_norm_backend: policy.layer_norm_backend_label().to_string(),
        attention_backend: policy.attention_backend_label().to_string(),
        softmax_backend: policy.softmax_backend_label().to_string(),
        tensor_util_backend: policy.tensor_util_backend_label().to_string(),
    }
}

#[allow(dead_code)]
pub fn roundtable_backend_audit(
    caps: DeviceCaps,
    schedule: &RoundtableSchedule,
) -> RoundtableBackendAudit {
    let wgpu_runtime_compiled = wgpu_runtime_compiled();
    let wgpu_runtime_context_installed = wgpu_runtime_context_installed();
    let bands = [
        ("above", schedule.above()),
        ("here", schedule.here()),
        ("beneath", schedule.beneath()),
    ]
    .into_iter()
    .map(|(band, plan)| rank_plan_backend_audit(caps, band, plan))
    .collect::<Vec<_>>();
    let any_wgpu_exact_runtime_ready = bands.iter().any(|band| band.wgpu_exact_runtime_ready);

    RoundtableBackendAudit {
        requested_backend: caps.backend.as_str().to_string(),
        wgpu_runtime_compiled,
        wgpu_runtime_context_installed,
        any_wgpu_exact_runtime_ready,
        bands,
    }
}

#[allow(dead_code)]
fn rank_plan_backend_audit(
    caps: DeviceCaps,
    band: &'static str,
    plan: &RankPlan,
) -> RankPlanBackendAudit {
    let choice = RankChoiceMeta {
        subgroup: plan.choice.subgroup,
        use_2ce: plan.choice.use_2ce,
        workgroup: plan.choice.wg,
        lanes: plan.choice.kl,
        channel_stride: plan.choice.ch,
        tile: plan.choice.tile,
        compaction_tile: plan.choice.ctile,
        fft_tile: plan.choice.fft_tile,
        fft_radix: plan.choice.fft_radix,
        fft_segments: plan.choice.fft_segments,
    };

    let (shape_supported, shape_reason) = wgpu_exact_shape_support(plan);
    let requested_wgpu = matches!(caps.backend, BackendKind::Wgpu);
    let runtime_ready = requested_wgpu
        && shape_supported
        && wgpu_runtime_compiled()
        && wgpu_runtime_context_installed();
    let status = if !requested_wgpu {
        "not_requested"
    } else if !wgpu_runtime_compiled() {
        "uncompiled"
    } else if !shape_supported {
        "fallback_shape"
    } else if !wgpu_runtime_context_installed() {
        "runtime_context_missing"
    } else {
        "exact_runtime_ready"
    };
    let reason = if requested_wgpu {
        shape_reason
    } else {
        Some(format!(
            "requested backend is {}, not wgpu",
            caps.backend.as_str()
        ))
    };

    RankPlanBackendAudit {
        band: band.to_string(),
        kind: plan.kind.as_str().to_string(),
        rows: plan.rows,
        cols: plan.cols,
        k: plan.k,
        choice,
        wgpu_exact_shape_supported: shape_supported,
        wgpu_exact_runtime_ready: runtime_ready,
        wgpu_exact_status: status.to_string(),
        wgpu_exact_reason: reason,
    }
}

#[allow(dead_code)]
fn wgpu_runtime_compiled() -> bool {
    cfg!(feature = "wgpu")
}

#[allow(dead_code)]
fn cuda_feature_compiled() -> bool {
    cfg!(feature = "cuda")
}

#[allow(dead_code)]
fn cuda_kernels_compiled() -> bool {
    st_core::backend::runtime_probe::backend_real_kernels_compiled(BackendKind::Cuda)
}

#[allow(dead_code)]
fn hip_feature_compiled() -> bool {
    cfg!(feature = "hip")
}

#[allow(dead_code)]
fn hip_real_compiled() -> bool {
    st_core::backend::runtime_probe::backend_real_kernels_compiled(BackendKind::Hip)
}

#[allow(dead_code)]
fn hip_kernels_compiled() -> bool {
    st_core::backend::runtime_probe::backend_real_kernels_compiled(BackendKind::Hip)
}

#[allow(dead_code)]
fn mps_feature_compiled() -> bool {
    st_core::backend::runtime_probe::backend_feature_enabled(BackendKind::Mps)
}

#[allow(dead_code)]
fn mps_placeholder() -> bool {
    st_core::backend::runtime_probe::backend_placeholder(BackendKind::Mps)
}

#[allow(dead_code)]
fn backend_feature_enabled(kind: BackendKind) -> bool {
    st_core::backend::runtime_probe::backend_feature_enabled(kind)
}

#[allow(dead_code)]
fn backend_kernels_compiled(kind: BackendKind) -> bool {
    st_core::backend::runtime_probe::backend_real_kernels_compiled(kind)
}

#[allow(dead_code)]
fn backend_placeholder(kind: BackendKind) -> bool {
    st_core::backend::runtime_probe::backend_placeholder(kind)
}

#[allow(dead_code)]
fn backend_runtime_status(kind: BackendKind) -> &'static str {
    st_core::backend::runtime_probe::backend_runtime_status(kind)
}

#[allow(dead_code)]
fn backend_runtime_recommendation(kind: BackendKind) -> &'static str {
    st_core::backend::runtime_probe::backend_runtime_recommendation(kind)
}

#[cfg(feature = "wgpu")]
#[allow(dead_code)]
fn wgpu_runtime_context_installed() -> bool {
    st_core::backend::wgpu_rt::installed_ctx().is_some()
}

#[cfg(not(feature = "wgpu"))]
#[allow(dead_code)]
fn wgpu_runtime_context_installed() -> bool {
    false
}

#[cfg(feature = "wgpu")]
fn ensure_wgpu_rank_runtime() -> Result<bool, TensorError> {
    st_core::backend::wgpu_rt::ensure_default_ctx().map_err(|message| {
        TensorError::Generic(format!("failed to initialize WGPU rank runtime: {message}"))
    })
}

#[cfg(not(feature = "wgpu"))]
fn ensure_wgpu_rank_runtime() -> Result<bool, TensorError> {
    Err(TensorError::Generic(
        "backend=wgpu requested but this build lacks the 'wgpu' feature. Rebuild with `--features wgpu`."
            .to_string(),
    ))
}

#[cfg(feature = "wgpu")]
#[allow(dead_code)]
fn wgpu_exact_shape_support(plan: &RankPlan) -> (bool, Option<String>) {
    match st_core::backend::wgpu_exec::wgpu_rank_exact_support(plan) {
        Ok(()) => (true, None),
        Err(reason) => (false, Some(reason)),
    }
}

#[cfg(not(feature = "wgpu"))]
#[allow(dead_code)]
fn wgpu_exact_shape_support(_plan: &RankPlan) -> (bool, Option<String>) {
    (
        false,
        Some("build lacks st-core/wgpu-rt; rebuild with --features wgpu".to_string()),
    )
}

pub fn parse_backend(name: Option<&str>) -> Result<BackendSelection, TensorError> {
    let raw = name.unwrap_or("auto");
    let kind = parse_backend_kind(raw)?;
    require_backend_available(kind)?;
    let caps = default_caps(kind);
    Ok(BackendSelection {
        label: backend_label(kind).to_string(),
        caps,
    })
}

fn parse_backend_kind(raw: &str) -> Result<BackendKind, TensorError> {
    if raw.eq_ignore_ascii_case("auto") {
        return Ok(default_backend_kind());
    }

    match raw.to_ascii_lowercase().as_str() {
        "wgpu" | "webgpu" => Ok(BackendKind::Wgpu),
        "mps" => Ok(BackendKind::Mps),
        "cuda" => Ok(BackendKind::Cuda),
        "hip" | "rocm" => Ok(BackendKind::Hip),
        "cpu" => Ok(BackendKind::Cpu),
        other => Err(TensorError::Generic(format!(
            "unknown backend '{other}', expected 'auto', 'wgpu', 'mps', 'cuda', 'hip', or 'cpu'"
        ))),
    }
}

fn default_backend_kind() -> BackendKind {
    if cfg!(feature = "wgpu") {
        BackendKind::Wgpu
    } else if cfg!(feature = "cuda") {
        BackendKind::Cuda
    } else if cfg!(feature = "hip") {
        BackendKind::Hip
    } else {
        BackendKind::Cpu
    }
}

fn backend_label(kind: BackendKind) -> &'static str {
    kind.as_str()
}

fn default_caps(kind: BackendKind) -> DeviceCaps {
    kind.default_caps()
}

fn require_backend_available(kind: BackendKind) -> Result<(), TensorError> {
    match kind {
        BackendKind::Cpu => Ok(()),
        BackendKind::Wgpu => {
            if cfg!(feature = "wgpu") {
                Ok(())
            } else {
                Err(TensorError::Generic(
                    "backend=wgpu requested but this build lacks the 'wgpu' feature. Rebuild with `--features wgpu`."
                        .to_string(),
                ))
            }
        }
        BackendKind::Mps => Err(TensorError::Generic(
            "backend=mps is a placeholder today; use backend=wgpu on macOS until native MPS kernels are wired."
                .to_string(),
        )),
        BackendKind::Cuda => {
            if cfg!(feature = "cuda") {
                Ok(())
            } else {
                Err(TensorError::Generic(
                    "backend=cuda requested but this build lacks the 'cuda' feature. Rebuild with `--features cuda`."
                        .to_string(),
                ))
            }
        }
        BackendKind::Hip => {
            if cfg!(feature = "hip") {
                Ok(())
            } else {
                Err(TensorError::Generic(
                    "backend=hip requested but this build lacks the 'hip' feature. Rebuild with `--features hip`."
                        .to_string(),
                ))
            }
        }
    }
}
