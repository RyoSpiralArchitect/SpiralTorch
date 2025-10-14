//! HIP backend (ROCm). Default: stubs. Enable `hip-real` for real path.
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HipErr {
    #[error("HIP not enabled (build with feature 'hip-real')")]
    NotEnabled,
    #[error("Other: {0}")]
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeviceInfo {
    pub id: u32,
    pub name: Cow<'static, str>,
    pub multi_node: bool,
}

impl DeviceInfo {
    pub fn new<N: Into<Cow<'static, str>>>(id: u32, name: N, multi_node: bool) -> Self {
        Self {
            id,
            name: name.into(),
            multi_node,
        }
    }
}

#[cfg(feature = "hip-real")]
pub mod rccl_comm;
#[cfg(feature = "hip-real")]
pub mod real;

#[cfg(not(feature = "hip-real"))]
pub mod stub {
    use super::DeviceInfo;
    use std::borrow::Cow;
    use std::env;
    use std::path::PathBuf;

    /// Returns `true` when the process appears to have access to a ROCm runtime.
    ///
    /// The stub checks for explicit opt-in via `SPIRALTORCH_FORCE_HIP`, then
    /// searches common library locations derived from `ROCM_PATH` / `HIP_PATH`.
    pub fn hip_available() -> bool {
        if env::var("SPIRALTORCH_FORCE_HIP")
            .map(|flag| matches!(flag.as_str(), "1" | "true" | "TRUE"))
            .unwrap_or(false)
        {
            return true;
        }

        let mut candidates = Vec::new();
        if let Some(path) = env::var_os("ROCM_PATH") {
            candidates.push(PathBuf::from(path));
        }
        if let Some(path) = env::var_os("HIP_PATH") {
            candidates.push(PathBuf::from(path));
        }

        candidates
            .into_iter()
            .flat_map(|root| {
                [
                    root.join("lib/libamdhip64.so"),
                    root.join("lib/libhiprtc.so"),
                    root.join("bin/hipcc"),
                ]
            })
            .any(|candidate| candidate.exists())
    }

    /// Surface a lightweight view of devices hinted through environment
    /// variables. When no hints are present we still emit a synthetic device so
    /// higher layers can keep their Z-space heuristics engaged while running on
    /// CPU-only development machines.
    pub fn device_info() -> Vec<DeviceInfo> {
        let mut devices = Vec::new();
        if let Some(list) = env::var("ROCM_VISIBLE_DEVICES")
            .ok()
            .or_else(|| env::var("HIP_VISIBLE_DEVICES").ok())
        {
            for (slot, token) in list.split(',').enumerate() {
                let trimmed = token.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let parsed_id = trimmed.parse::<u32>().ok();
                let id = parsed_id.unwrap_or(slot as u32);
                devices.push(DeviceInfo::new(
                    id,
                    Cow::Owned(format!("ROCm-device-{trimmed}")),
                    false,
                ));
            }
        }

        if devices.is_empty() && hip_available() {
            devices.push(DeviceInfo::new(0, Cow::Borrowed("rocm-probe"), false));
        }

        devices
    }
}
