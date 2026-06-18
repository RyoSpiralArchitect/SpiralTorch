// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::device_caps::{BackendKind, DeviceCaps};
use st_tensor::{emit_tensor_op, emit_tensor_op_meta};

fn finite_meta_f32(value: f32) -> serde_json::Value {
    serde_json::Value::from(if value.is_finite() { value as f64 } else { 0.0 })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
            BackendKind::Wgpu => "use backend='wgpu' today; native MPS kernels are not wired yet",
            _ => "use backend='cpu' today; enable the 'wgpu' feature to route Metal hosts through WGPU",
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendResolution {
    pub effective_backend: BackendKind,
    pub reported_backend: BackendKind,
    pub mps_probe: Option<MpsProbeReport>,
}

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

pub fn mps_probe() -> MpsProbeReport {
    let planner_surrogate_backend = if cfg!(target_os = "macos") && cfg!(feature = "wgpu") {
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

pub const fn backend_runtime_status(kind: BackendKind) -> &'static str {
    if !backend_feature_enabled(kind) {
        return "feature_disabled";
    }
    match kind {
        BackendKind::Cpu => "cpu",
        BackendKind::Wgpu => "kernel_wired",
        BackendKind::Mps => "placeholder",
        BackendKind::Cuda => "kernel_wired",
        BackendKind::Hip => {
            if cfg!(feature = "hip-real") {
                "kernel_wired"
            } else {
                "stub_without_hip_real"
            }
        }
    }
}

pub const fn backend_runtime_recommendation(kind: BackendKind) -> &'static str {
    match kind {
        BackendKind::Cpu => "cpu backend is available",
        BackendKind::Wgpu => {
            if cfg!(feature = "wgpu-rt") {
                "wgpu rank runtime can be initialized by the training entrypoint"
            } else {
                "rebuild with the wgpu feature to enable WGPU training kernels"
            }
        }
        BackendKind::Mps => {
            "mps backend is a placeholder today; use backend='wgpu' on macOS until native MPS kernels are wired"
        }
        BackendKind::Cuda => {
            if cfg!(feature = "cuda") {
                "cuda rank kernels are compiled; verify tensor-kernel trace counters during training"
            } else {
                "rebuild with the cuda feature to request CUDA"
            }
        }
        BackendKind::Hip => {
            if cfg!(all(feature = "hip", feature = "hip-real")) {
                "hip-real kernels are compiled; verify tensor-kernel trace counters during training"
            } else if cfg!(feature = "hip") {
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
    emit_backend_resolution_meta(resolution);
    resolution
}

fn emit_backend_resolution_meta(resolution: BackendResolution) {
    emit_tensor_op("backend_resolution", &[1], &[1]);
    emit_tensor_op_meta("backend_resolution", || {
        let mut payload = serde_json::Map::new();
        payload.insert(
            "backend".into(),
            resolution.effective_backend.as_str().into(),
        );
        payload.insert(
            "requested_backend".into(),
            resolution.reported_backend.as_str().into(),
        );
        payload.insert("kind".into(), "st_core_backend_resolution".into());
        payload.insert(
            "effective_backend".into(),
            resolution.effective_backend.as_str().into(),
        );
        payload.insert(
            "reported_backend".into(),
            resolution.reported_backend.as_str().into(),
        );
        payload.insert(
            "has_mps_probe".into(),
            resolution.mps_probe.is_some().into(),
        );
        if let Some(probe) = resolution.mps_probe {
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
        }
        serde_json::Value::Object(payload)
    });
}

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
    let (aligned_workgroup, occupancy_score) = if let Some(requested) = requested_workgroup {
        (
            Some(caps.align_workgroup(requested)),
            Some(caps.occupancy_score(requested)),
        )
    } else {
        (None, None)
    };

    let (preferred_tile, preferred_compaction_tile) = if let Some(total_cols) = cols {
        (
            Some(caps.preferred_tile(total_cols, tile_hint.unwrap_or(0))),
            Some(caps.preferred_compaction_tile(total_cols, compaction_hint.unwrap_or(0))),
        )
    } else {
        (None, None)
    };

    let report = DeviceReport {
        reported_backend,
        caps,
        requested_workgroup,
        aligned_workgroup,
        occupancy_score,
        preferred_tile,
        preferred_compaction_tile,
        mps_probe,
    };
    emit_device_report_meta(report);
    report
}

fn emit_device_report_meta(report: DeviceReport) {
    emit_tensor_op("backend_device_report", &[1, 6], &[1, 8]);
    emit_tensor_op_meta("backend_device_report", || {
        let mut payload = serde_json::Map::new();
        payload.insert("backend".into(), report.caps.backend.as_str().into());
        payload.insert(
            "requested_backend".into(),
            report.reported_backend.as_str().into(),
        );
        payload.insert("kind".into(), "st_core_backend_device_report".into());
        payload.insert(
            "reported_backend".into(),
            report.reported_backend.as_str().into(),
        );
        payload.insert(
            "effective_backend".into(),
            report.effective_backend().as_str().into(),
        );
        payload.insert("subgroup".into(), report.caps.subgroup.into());
        payload.insert("lane_width".into(), report.caps.lane_width.into());
        payload.insert("max_workgroup".into(), report.caps.max_workgroup.into());
        payload.insert(
            "has_shared_mem".into(),
            report.caps.shared_mem_per_workgroup.is_some().into(),
        );
        payload.insert(
            "shared_mem_per_workgroup".into(),
            report
                .caps
                .shared_mem_per_workgroup
                .unwrap_or_default()
                .into(),
        );
        payload.insert(
            "requested_workgroup".into(),
            report.requested_workgroup.unwrap_or_default().into(),
        );
        payload.insert(
            "aligned_workgroup".into(),
            report.aligned_workgroup.unwrap_or_default().into(),
        );
        payload.insert(
            "occupancy_score".into(),
            finite_meta_f32(report.occupancy_score.unwrap_or(0.0)),
        );
        payload.insert(
            "preferred_tile".into(),
            report.preferred_tile.unwrap_or_default().into(),
        );
        payload.insert(
            "preferred_compaction_tile".into(),
            report.preferred_compaction_tile.unwrap_or_default().into(),
        );
        payload.insert("has_mps_probe".into(), report.mps_probe.is_some().into());
        if let Some(probe) = report.mps_probe {
            payload.insert("mps_status".into(), probe.status().as_str().into());
            payload.insert("mps_host_class".into(), probe.host_class.as_str().into());
            payload.insert("mps_planner_route".into(), probe.planner_route().into());
            payload.insert("mps_placeholder".into(), probe.placeholder().into());
            payload.insert("mps_available".into(), probe.available().into());
        }
        serde_json::Value::Object(payload)
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn mps_probe_reports_honest_placeholder_surface() {
        let probe = mps_probe();
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

        if cfg!(target_os = "macos") && cfg!(feature = "wgpu") {
            assert_eq!(probe.planner_surrogate_backend, BackendKind::Wgpu);
            assert_eq!(probe.planner_route(), "metal-via-wgpu");
        } else {
            assert_eq!(probe.planner_surrogate_backend, BackendKind::Cpu);
            assert_eq!(probe.planner_route(), "cpu-fallback");
        }
    }

    #[test]
    fn build_device_report_preserves_mps_overlay_and_metrics() {
        let probe = mps_probe();
        let caps = probe.planner_surrogate_backend.default_caps();
        let report = build_device_report(
            BackendKind::Mps,
            caps,
            Some(probe),
            Some(300),
            Some(4096),
            None,
            None,
        );
        assert_eq!(report.reported_backend, BackendKind::Mps);
        assert_eq!(report.effective_backend(), caps.backend);
        assert_eq!(report.mps_probe, Some(probe));
        assert_eq!(report.requested_workgroup, Some(300));
        assert_eq!(report.aligned_workgroup, Some(caps.align_workgroup(300)));
        assert_eq!(report.occupancy_score, Some(caps.occupancy_score(300)));
        assert_eq!(report.preferred_tile, Some(caps.preferred_tile(4096, 0)));
        assert_eq!(
            report.preferred_compaction_tile,
            Some(caps.preferred_compaction_tile(4096, 0))
        );
    }

    #[test]
    fn resolve_backend_routes_mps_through_surrogate_probe() {
        let resolution = resolve_backend(BackendKind::Mps);
        assert_eq!(resolution.reported_backend, BackendKind::Mps);
        assert!(resolution.mps_probe.is_some());
        assert_eq!(
            resolution.effective_backend,
            resolution
                .mps_probe
                .expect("mps probe present")
                .planner_surrogate_backend
        );

        let resolution = resolve_backend(BackendKind::Cuda);
        assert_eq!(resolution.reported_backend, BackendKind::Cuda);
        assert_eq!(resolution.effective_backend, BackendKind::Cuda);
        assert!(resolution.mps_probe.is_none());
    }

    #[test]
    fn backend_availability_reports_feature_and_kernel_truth() {
        assert!(backend_feature_enabled(BackendKind::Cpu));
        assert!(backend_real_kernels_compiled(BackendKind::Cpu));
        assert!(!backend_placeholder(BackendKind::Cpu));
        assert_eq!(backend_runtime_status(BackendKind::Cpu), "cpu");

        assert!(!backend_real_kernels_compiled(BackendKind::Mps));
        assert!(backend_placeholder(BackendKind::Mps));
        assert_eq!(
            backend_feature_enabled(BackendKind::Mps),
            cfg!(feature = "mps")
        );
        assert_eq!(
            backend_runtime_status(BackendKind::Mps),
            if cfg!(feature = "mps") {
                "placeholder"
            } else {
                "feature_disabled"
            }
        );

        assert_eq!(
            backend_feature_enabled(BackendKind::Hip),
            cfg!(feature = "hip")
        );
        assert_eq!(
            backend_real_kernels_compiled(BackendKind::Hip),
            cfg!(all(feature = "hip", feature = "hip-real"))
        );
        if cfg!(feature = "hip") && !cfg!(feature = "hip-real") {
            assert_eq!(
                backend_runtime_status(BackendKind::Hip),
                "stub_without_hip_real"
            );
        }
    }

    #[test]
    fn backend_resolution_and_device_report_emit_meta() {
        let _lock = crate::telemetry::tensor_observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
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
        st_tensor::set_tensor_op_meta_observer(previous);

        assert_eq!(report.reported_backend, BackendKind::Mps);
        let events = events.lock().unwrap();
        let resolution_meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "backend_resolution"
                    && data["kind"] == "st_core_backend_resolution"
                    && data["reported_backend"] == "mps"
            })
            .expect("backend resolution metadata event");
        assert_eq!(resolution_meta.1["requested_backend"], "mps");
        assert_eq!(resolution_meta.1["has_mps_probe"], true);
        assert_eq!(resolution_meta.1["mps_placeholder"], true);

        let report_meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "backend_device_report"
                    && data["kind"] == "st_core_backend_device_report"
                    && data["reported_backend"] == "mps"
                    && data["requested_workgroup"] == 300
            })
            .expect("backend device report metadata event");
        assert_eq!(report_meta.1["has_mps_probe"], true);
        assert_eq!(report_meta.1["lane_width"], report.caps.lane_width);
        assert_eq!(
            report_meta.1["aligned_workgroup"],
            report.aligned_workgroup.unwrap()
        );
        assert!(report_meta.1["occupancy_score"].as_f64().unwrap() >= 0.0);
        assert!(report_meta.1["preferred_tile"].as_u64().unwrap() > 0);
    }
}
