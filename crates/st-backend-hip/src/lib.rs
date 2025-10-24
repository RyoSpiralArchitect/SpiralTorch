// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! HIP backend (ROCm). Default: stubs. Enable `hip-real` for real path.
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
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
pub mod compaction;
#[cfg(feature = "hip-real")]
pub mod rccl_comm;
#[cfg(feature = "hip-real")]
pub mod real;

#[derive(Debug, Clone)]
pub struct HipProbe {
    pub available: bool,
    pub initialized: bool,
    pub devices: Vec<DeviceInfo>,
    pub error: Option<String>,
}

#[derive(Debug)]
pub struct HipRuntime {
    devices: Vec<DeviceInfo>,
}

impl HipRuntime {
    fn new(devices: Vec<DeviceInfo>) -> Self {
        Self { devices }
    }

    pub fn devices(&self) -> &[DeviceInfo] {
        &self.devices
    }

    pub fn device_count(&self) -> usize {
        self.devices.len()
    }
}

fn runtime_slot() -> &'static Mutex<Option<Arc<HipRuntime>>> {
    static RUNTIME: OnceLock<Mutex<Option<Arc<HipRuntime>>>> = OnceLock::new();
    RUNTIME.get_or_init(|| Mutex::new(None))
}

pub fn runtime() -> Option<Arc<HipRuntime>> {
    runtime_slot()
        .lock()
        .ok()
        .and_then(|guard| guard.as_ref().map(Arc::clone))
}

pub fn init() -> Result<Arc<HipRuntime>, HipErr> {
    if let Some(existing) = runtime() {
        return Ok(existing);
    }

    let runtime = Arc::new(build_runtime()?);

    let mut guard = runtime_slot()
        .lock()
        .map_err(|_| HipErr::Other("failed to lock HIP runtime slot".into()))?;
    if let Some(existing) = guard.as_ref() {
        return Ok(existing.clone());
    }
    *guard = Some(runtime.clone());
    Ok(runtime)
}

#[cfg(test)]
fn reset_runtime_for_tests() {
    if let Ok(mut guard) = runtime_slot().lock() {
        guard.take();
    }
}

pub fn probe() -> HipProbe {
    let available = hip_env_available();
    match init() {
        Ok(runtime) => HipProbe {
            available: true,
            initialized: true,
            devices: runtime.devices().to_vec(),
            error: None,
        },
        Err(err) => {
            let devices = finalize_devices(collect_env_devices(), available);
            HipProbe {
                available,
                initialized: false,
                devices,
                error: Some(err.to_string()),
            }
        }
    }
}

#[cfg(not(feature = "hip-real"))]
pub mod stub {
    use super::{collect_env_devices, finalize_devices, hip_env_available, DeviceInfo};

    /// Returns `true` when the process appears to have access to a ROCm runtime.
    ///
    /// The stub checks for explicit opt-in via `SPIRALTORCH_FORCE_HIP`, then
    /// searches common ROCm install locations (including `ROCM_PATH` / `HIP_PATH`,
    /// default `/opt/rocm*` directories, library search paths, and `PATH`
    /// entries for `hipcc`).
    pub fn hip_available() -> bool {
        hip_env_available()
    }

    /// Surface a lightweight view of devices hinted through environment
    /// variables. When no hints are present we still emit a synthetic device so
    /// higher layers can keep their Z-space heuristics engaged while running on
    /// CPU-only development machines.
    pub fn device_info() -> Vec<DeviceInfo> {
        finalize_devices(collect_env_devices(), hip_env_available())
    }
}

fn build_runtime() -> Result<HipRuntime, HipErr> {
    if !hip_env_available() {
        return Err(HipErr::Other(
            "HIP runtime not detected; set SPIRALTORCH_FORCE_HIP=1 or install ROCm".into(),
        ));
    }

    let devices = finalize_devices(collect_env_devices(), true);
    Ok(HipRuntime::new(devices))
}

fn hip_env_available() -> bool {
    if std::env::var("SPIRALTORCH_FORCE_HIP")
        .map(|flag| matches!(flag.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
    {
        return true;
    }

    let mut seen = HashSet::new();
    let mut push_candidate = |candidate: PathBuf| {
        if seen.insert(candidate.clone()) {
            if candidate.exists() {
                return true;
            }
        }
        false
    };

    for root in gather_rocm_roots() {
        if push_rocm_markers(&root, &mut push_candidate) {
            return true;
        }
    }

    for search_path in gather_library_search_paths() {
        for library in ["libamdhip64.so", "libhiprtc.so"] {
            if push_candidate(search_path.join(library)) {
                return true;
            }
        }
    }

    for bin_path in gather_binary_search_paths() {
        for tool in ["hipcc", "rocminfo"] {
            if push_candidate(bin_path.join(tool)) {
                return true;
            }
        }
    }

    false
}

fn gather_rocm_roots() -> Vec<PathBuf> {
    let mut roots = HashSet::new();
    const ENV_ROOT_KEYS: &[&str] = &[
        "ROCM_PATH",
        "ROCM_HOME",
        "ROCM_ROOT",
        "HIP_PATH",
        "HIP_HOME",
        "HIP_ROOT",
    ];

    for key in ENV_ROOT_KEYS {
        if let Some(path) = std::env::var_os(key) {
            roots.insert(PathBuf::from(path));
        }
    }

    for default in ["/opt/rocm", "/usr/local/rocm", "/usr/lib/rocm"] {
        roots.insert(PathBuf::from(default));
    }

    if let Ok(entries) = std::fs::read_dir("/opt") {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            if let Some(name) = entry.file_name().to_str() {
                if name.starts_with("rocm") {
                    roots.insert(path);
                }
            }
        }
    }

    roots.into_iter().collect()
}

fn push_rocm_markers(root: &Path, push: &mut impl FnMut(PathBuf) -> bool) -> bool {
    let lib_dirs = [
        root.join("lib"),
        root.join("lib64"),
        root.join("hip").join("lib"),
        root.join("hip").join("lib64"),
    ];

    for dir in lib_dirs {
        if push(dir.join("libamdhip64.so")) || push(dir.join("libhiprtc.so")) {
            return true;
        }
    }

    let bin_dir = root.join("bin");
    for tool in ["hipcc", "rocminfo"] {
        if push(bin_dir.join(tool)) {
            return true;
        }
    }

    false
}

fn gather_library_search_paths() -> Vec<PathBuf> {
    let mut paths = HashSet::new();
    const LIB_ENV_KEYS: &[&str] = &[
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "HIP_LIBRARY_PATH",
        "HIPLD_LIBRARY_PATH",
        "ROCM_LIBRARY_PATH",
    ];

    for key in LIB_ENV_KEYS {
        if let Some(value) = std::env::var_os(key) {
            for path in std::env::split_paths(&value) {
                paths.insert(path);
            }
        }
    }

    paths.into_iter().collect()
}

fn gather_binary_search_paths() -> Vec<PathBuf> {
    let mut paths = HashSet::new();
    if let Some(value) = std::env::var_os("PATH") {
        for path in std::env::split_paths(&value) {
            paths.insert(path);
        }
    }

    paths.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::Mutex;
    use tempfile::tempdir;

    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    fn clear_force_flag() {
        std::env::remove_var("SPIRALTORCH_FORCE_HIP");
    }

    fn restore_env(key: &str, previous: Option<std::ffi::OsString>) {
        match previous {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }

    #[test]
    fn hip_available_when_forced() {
        let _guard = ENV_MUTEX.lock().unwrap();
        let prev = std::env::var_os("SPIRALTORCH_FORCE_HIP");
        std::env::set_var("SPIRALTORCH_FORCE_HIP", "1");
        assert!(hip_env_available());
        restore_env("SPIRALTORCH_FORCE_HIP", prev);
    }

    #[test]
    fn hip_available_via_rocm_path_marker() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_force_flag();
        let prev_rocm = std::env::var_os("ROCM_PATH");

        let temp = tempdir().expect("tempdir");
        let lib_dir = temp.path().join("lib");
        fs::create_dir(&lib_dir).expect("lib dir");
        fs::write(lib_dir.join("libamdhip64.so"), b"").expect("touch lib");

        std::env::set_var("ROCM_PATH", temp.path());
        assert!(hip_env_available());

        restore_env("ROCM_PATH", prev_rocm);
    }

    #[test]
    fn hip_available_via_path_hipcc() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_force_flag();

        let prev_path = std::env::var_os("PATH");
        let temp = tempdir().expect("tempdir");
        let bin_dir = temp.path().join("bin");
        fs::create_dir(&bin_dir).expect("bin dir");
        fs::write(bin_dir.join("hipcc"), b"").expect("touch hipcc");

        let mut paths = vec![bin_dir];
        if let Some(existing) = prev_path.clone() {
            paths.extend(std::env::split_paths(&existing));
        }
        let joined = std::env::join_paths(paths).expect("join paths");
        std::env::set_var("PATH", &joined);

        assert!(hip_env_available());

        restore_env("PATH", prev_path);
    }

    #[test]
    fn init_requires_detectable_runtime() {
        let _guard = ENV_MUTEX.lock().unwrap();
        super::reset_runtime_for_tests();
        clear_force_flag();

        let prev_rocm = std::env::var_os("ROCM_PATH");
        std::env::remove_var("ROCM_PATH");
        let prev_path = std::env::var_os("PATH");
        std::env::set_var("PATH", "");

        assert!(super::init().is_err());

        restore_env("PATH", prev_path);
        restore_env("ROCM_PATH", prev_rocm);
        super::reset_runtime_for_tests();
    }

    #[test]
    fn init_succeeds_when_forced() {
        let _guard = ENV_MUTEX.lock().unwrap();
        super::reset_runtime_for_tests();
        let prev_force = std::env::var_os("SPIRALTORCH_FORCE_HIP");
        std::env::set_var("SPIRALTORCH_FORCE_HIP", "1");

        let runtime = super::init().expect("runtime should initialise when forced");
        assert!(runtime.device_count() >= 1);
        assert!(runtime.devices().iter().any(|device| device.id == 0));

        restore_env("SPIRALTORCH_FORCE_HIP", prev_force);
        super::reset_runtime_for_tests();
    }

    #[test]
    fn gemm_stub_matches_reference_matmul() {
        let _guard = ENV_MUTEX.lock().unwrap();
        super::reset_runtime_for_tests();
        let prev_force = std::env::var_os("SPIRALTORCH_FORCE_HIP");
        std::env::set_var("SPIRALTORCH_FORCE_HIP", "1");

        let lhs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut out = vec![0.0; 4];
        super::gemm_f32(2, 2, 3, &lhs, &rhs, &mut out).expect("gemm stub should succeed");
        let expected = vec![58.0, 64.0, 139.0, 154.0];
        assert_eq!(out, expected);

        restore_env("SPIRALTORCH_FORCE_HIP", prev_force);
        super::reset_runtime_for_tests();
    }
}

fn collect_env_devices() -> Vec<DeviceInfo> {
    let mut devices = Vec::new();
    if let Some(list) = std::env::var("ROCM_VISIBLE_DEVICES")
        .ok()
        .or_else(|| std::env::var("HIP_VISIBLE_DEVICES").ok())
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
                std::borrow::Cow::Owned(format!("ROCm-device-{trimmed}")),
                false,
            ));
        }
    }
    devices
}

fn finalize_devices(mut devices: Vec<DeviceInfo>, available: bool) -> Vec<DeviceInfo> {
    if devices.is_empty() && available {
        devices.push(DeviceInfo::new(
            0,
            std::borrow::Cow::Borrowed("rocm-probe"),
            false,
        ));
    }

    devices
}

#[cfg(not(feature = "hip-real"))]
pub use stub::{device_info, hip_available};

#[cfg(feature = "hip-real")]
pub fn hip_available() -> bool {
    hip_env_available()
}

fn validate_gemm_dimensions(
    m: usize,
    n: usize,
    k: usize,
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
) -> Result<(), HipErr> {
    let expected_lhs = m
        .checked_mul(k)
        .ok_or_else(|| HipErr::Other("gemm dimensions overflow while validating lhs".into()))?;
    let expected_rhs = k
        .checked_mul(n)
        .ok_or_else(|| HipErr::Other("gemm dimensions overflow while validating rhs".into()))?;
    let expected_out = m
        .checked_mul(n)
        .ok_or_else(|| HipErr::Other("gemm dimensions overflow while validating output".into()))?;

    if lhs.len() != expected_lhs {
        return Err(HipErr::Other(format!(
            "lhs buffer length {} does not match m*k={}",
            lhs.len(),
            expected_lhs
        )));
    }
    if rhs.len() != expected_rhs {
        return Err(HipErr::Other(format!(
            "rhs buffer length {} does not match k*n={}",
            rhs.len(),
            expected_rhs
        )));
    }
    if out.len() != expected_out {
        return Err(HipErr::Other(format!(
            "output buffer length {} does not match m*n={}",
            out.len(),
            expected_out
        )));
    }

    Ok(())
}

fn gemm_stub(m: usize, n: usize, k: usize, lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
    out.fill(0.0);
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for inner in 0..k {
                let lhs_index = row * k + inner;
                let rhs_index = inner * n + col;
                acc += lhs[lhs_index] * rhs[rhs_index];
            }
            out[row * n + col] = acc;
        }
    }
}

pub fn gemm_f32(
    m: usize,
    n: usize,
    k: usize,
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
) -> Result<(), HipErr> {
    init()?;
    validate_gemm_dimensions(m, n, k, lhs, rhs, out)?;
    gemm_stub(m, n, k, lhs, rhs, out);
    Ok(())
}

pub unsafe fn gemm_f32_raw(
    m: usize,
    n: usize,
    k: usize,
    lhs: *const f32,
    rhs: *const f32,
    out: *mut f32,
) -> Result<(), HipErr> {
    if lhs.is_null() || rhs.is_null() || out.is_null() {
        return Err(HipErr::Other(
            "gemm received a null pointer; provide contiguous row-major buffers".into(),
        ));
    }

    let lhs_len = m.checked_mul(k).ok_or_else(|| {
        HipErr::Other("gemm dimensions overflow while constructing lhs slice".into())
    })?;
    let rhs_len = k.checked_mul(n).ok_or_else(|| {
        HipErr::Other("gemm dimensions overflow while constructing rhs slice".into())
    })?;
    let out_len = m.checked_mul(n).ok_or_else(|| {
        HipErr::Other("gemm dimensions overflow while constructing output slice".into())
    })?;

    let lhs_slice = std::slice::from_raw_parts(lhs, lhs_len);
    let rhs_slice = std::slice::from_raw_parts(rhs, rhs_len);
    let out_slice = std::slice::from_raw_parts_mut(out, out_len);
    gemm_f32(m, n, k, lhs_slice, rhs_slice, out_slice)
}

#[cfg(feature = "hip-real")]
pub fn device_info() -> Vec<DeviceInfo> {
    let mut devices = collect_env_devices();
    if !devices.is_empty() {
        return devices;
    }

    if !hip_env_available() {
        return devices;
    }

    match crate::real::enumerate_devices() {
        Ok(found) if !found.is_empty() => found,
        Ok(_) => {
            devices.push(DeviceInfo::new(
                0,
                std::borrow::Cow::Borrowed("hip-runtime"),
                true,
            ));
            devices
        }
        Err(err) => {
            eprintln!("[hip] failed to enumerate devices: {err}");
            devices.push(DeviceInfo::new(
                0,
                std::borrow::Cow::Borrowed("hip-runtime"),
                true,
            ));
            devices
        }
    }
}
