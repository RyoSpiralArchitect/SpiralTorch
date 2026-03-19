// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::device_caps::{BackendKind, DeviceCaps};

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

pub fn resolve_backend(requested_backend: BackendKind) -> BackendResolution {
    match requested_backend {
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
    }
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

    DeviceReport {
        reported_backend,
        caps,
        requested_workgroup,
        aligned_workgroup,
        occupancy_score,
        preferred_tile,
        preferred_compaction_tile,
        mps_probe,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
