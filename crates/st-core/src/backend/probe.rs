//! Lightweight backend availability probes.
//!
//! The goal is to give higher layers a cheap way to answer:
//! "is this backend even worth attempting to initialise here?" without pulling
//! in full runtimes or toolchains.

use super::device_caps::BackendKind;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendDeviceInfo {
    pub backend: BackendKind,
    pub id: u32,
    pub name: String,
    pub multi_node: bool,
}

fn env_truthy(name: &str) -> Option<bool> {
    let value = std::env::var(name).ok()?;
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    Some(matches!(
        value,
        "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
    ))
}

/// Returns `true` when HIP (ROCm) appears usable for this process.
pub fn hip_available() -> bool {
    #[cfg(feature = "hip")]
    {
        return st_backend_hip::hip_available();
    }
    #[cfg(not(feature = "hip"))]
    {
        false
    }
}

/// Returns any HIP devices that can be inferred from environment hints.
pub fn hip_devices() -> Vec<BackendDeviceInfo> {
    #[cfg(feature = "hip")]
    {
        return st_backend_hip::device_info()
            .into_iter()
            .map(|d| BackendDeviceInfo {
                backend: BackendKind::Hip,
                id: d.id,
                name: d.name.to_string(),
                multi_node: d.multi_node,
            })
            .collect();
    }
    #[cfg(not(feature = "hip"))]
    {
        Vec::new()
    }
}

/// Returns `true` when CUDA tooling/runtime appears usable for this process.
///
/// This is a best-effort probe; it does *not* guarantee the CUDA driver stack
/// is correctly installed.
pub fn cuda_available() -> bool {
    if let Some(flag) = env_truthy("SPIRALTORCH_FORCE_CUDA") {
        return flag;
    }

    let candidates = ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"];
    for var in candidates {
        if let Ok(path) = std::env::var(var) {
            let root = std::path::PathBuf::from(path);
            let nvcc = if cfg!(windows) {
                root.join("bin").join("nvcc.exe")
            } else {
                root.join("bin").join("nvcc")
            };
            if nvcc.exists() {
                return true;
            }
        }
    }

    false
}

pub fn cuda_devices() -> Vec<BackendDeviceInfo> {
    if !cuda_available() {
        return Vec::new();
    }
    vec![BackendDeviceInfo {
        backend: BackendKind::Cuda,
        id: 0,
        name: "cuda".to_string(),
        multi_node: false,
    }]
}

/// Returns `true` when MPS is plausibly usable for this process.
///
/// Today this is primarily a platform gate with an explicit override.
pub fn mps_available() -> bool {
    if let Some(flag) = env_truthy("SPIRALTORCH_FORCE_MPS") {
        return flag;
    }
    cfg!(target_os = "macos")
}

pub fn mps_devices() -> Vec<BackendDeviceInfo> {
    if !mps_available() {
        return Vec::new();
    }
    vec![BackendDeviceInfo {
        backend: BackendKind::Mps,
        id: 0,
        name: "mps".to_string(),
        multi_node: false,
    }]
}
