// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

use serde::Serialize;
use st_core::backend::device_caps::{BackendKind, DeviceCaps};
use st_nn::TensorError;

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
        "cuda" => Ok(BackendKind::Cuda),
        "hip" | "rocm" => Ok(BackendKind::Hip),
        "cpu" => Ok(BackendKind::Cpu),
        other => Err(TensorError::Generic(format!(
            "unknown backend '{other}', expected 'auto', 'wgpu', 'cuda', 'hip', or 'cpu'"
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
    match kind {
        BackendKind::Wgpu => "wgpu",
        BackendKind::Cuda => "cuda",
        BackendKind::Hip => "hip",
        BackendKind::Cpu => "cpu",
    }
}

fn default_caps(kind: BackendKind) -> DeviceCaps {
    match kind {
        BackendKind::Wgpu => DeviceCaps::wgpu(32, true, 256),
        BackendKind::Cuda => DeviceCaps::cuda(32, 1024, Some(96 * 1024)),
        BackendKind::Hip => DeviceCaps::hip(32, 1024, Some(64 * 1024)),
        BackendKind::Cpu => DeviceCaps::cpu(),
    }
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
