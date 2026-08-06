// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! HIP backend (ROCm). Default: stubs. Enable `hip-real` for real path.
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HipErr {
    #[error("HIP not enabled (build with feature 'hip-real')")]
    NotEnabled,
    #[error("Other: {0}")]
    Other(String),
}

impl From<st_kernel_contracts::compaction::CompactionError> for HipErr {
    fn from(error: st_kernel_contracts::compaction::CompactionError) -> Self {
        Self::Other(format!("compaction contract: {error}"))
    }
}

/// Activation applied by the device-resident GEMM epilogue.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmActivation {
    Relu = 1,
    Gelu = 2,
}

impl GemmActivation {
    #[cfg(feature = "hip-real")]
    pub(crate) const fn hip_code(self) -> i32 {
        self as i32
    }
}

/// Returns whether this build contains the real ROCm implementation.
///
/// The default CPU reference implementation remains useful for validating the
/// public contract, but callers must not advertise it as HIP execution.
pub const fn real_backend_compiled() -> bool {
    cfg!(feature = "hip-real")
}

#[cfg(any(feature = "hip-real", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RcclAllGatherLayout {
    pub receive_count: usize,
    pub send_bytes: usize,
    pub receive_bytes: usize,
}

#[cfg(any(feature = "hip-real", test))]
pub(crate) fn validate_rccl_topology(world: i32, rank: i32) -> Result<usize, HipErr> {
    if world <= 0 || rank < 0 || rank >= world {
        return Err(HipErr::Other(format!(
            "RCCL: invalid topology rank {rank} within world {world}"
        )));
    }
    usize::try_from(world).map_err(|_| HipErr::Other("RCCL: world size does not fit usize".into()))
}

#[cfg(any(feature = "hip-real", test))]
pub(crate) fn rccl_allgather_layout(
    send_count: usize,
    world_size: usize,
) -> Result<RcclAllGatherLayout, HipErr> {
    if world_size == 0 {
        return Err(HipErr::Other(
            "RCCL all-gather world size must be positive".into(),
        ));
    }
    let receive_count = send_count
        .checked_mul(world_size)
        .ok_or_else(|| HipErr::Other("RCCL all-gather receive count overflow".into()))?;
    let send_bytes = send_count
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or_else(|| HipErr::Other("RCCL all-gather send byte length overflow".into()))?;
    let receive_bytes = receive_count
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or_else(|| HipErr::Other("RCCL all-gather receive byte length overflow".into()))?;
    Ok(RcclAllGatherLayout {
        receive_count,
        send_bytes,
        receive_bytes,
    })
}

pub mod compaction_contract;

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

fn lock_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            let guard = poisoned.into_inner();
            mutex.clear_poison();
            guard
        }
    }
}

pub fn runtime() -> Option<Arc<HipRuntime>> {
    lock_recover(runtime_slot()).as_ref().map(Arc::clone)
}

pub fn init() -> Result<Arc<HipRuntime>, HipErr> {
    if let Some(existing) = runtime() {
        return Ok(existing);
    }

    let runtime = Arc::new(build_runtime()?);

    let mut guard = lock_recover(runtime_slot());
    if let Some(existing) = guard.as_ref() {
        return Ok(existing.clone());
    }
    *guard = Some(runtime.clone());
    Ok(runtime)
}

#[cfg(test)]
fn reset_runtime_for_tests() {
    lock_recover(runtime_slot()).take();
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

    #[cfg(feature = "hip-real")]
    let devices = resolve_runtime_devices(collect_env_devices(), crate::real::enumerate_devices)?;
    #[cfg(not(feature = "hip-real"))]
    let devices = finalize_devices(collect_env_devices(), true);

    Ok(HipRuntime::new(devices))
}

#[cfg(any(feature = "hip-real", test))]
fn resolve_runtime_devices<F>(
    environment_devices: Vec<DeviceInfo>,
    enumerate: F,
) -> Result<Vec<DeviceInfo>, HipErr>
where
    F: FnOnce() -> Result<Vec<DeviceInfo>, HipErr>,
{
    if !environment_devices.is_empty() {
        return Ok(environment_devices);
    }

    let runtime_devices = enumerate()?;
    if runtime_devices.is_empty() {
        return Err(HipErr::Other(
            "HIP runtime reported no available devices".into(),
        ));
    }
    Ok(runtime_devices)
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
        if seen.insert(candidate.clone()) && candidate.exists() {
            return true;
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

fn validate_gemm_scale(scale: f32) -> Result<(), HipErr> {
    if scale.is_finite() {
        Ok(())
    } else {
        Err(HipErr::Other(format!(
            "gemm scale must be finite, received {scale}"
        )))
    }
}

fn validate_gemm_epilogue(
    m: usize,
    n: usize,
    bias: &[f32],
    residual: Option<&[f32]>,
) -> Result<(), HipErr> {
    if bias.len() != n {
        return Err(HipErr::Other(format!(
            "bias buffer length {} does not match n={n}",
            bias.len()
        )));
    }
    if let Some(residual) = residual {
        let expected = m.checked_mul(n).ok_or_else(|| {
            HipErr::Other("gemm dimensions overflow while validating residual".into())
        })?;
        if residual.len() != expected {
            return Err(HipErr::Other(format!(
                "residual buffer length {} does not match m*n={expected}",
                residual.len()
            )));
        }
    }
    Ok(())
}

#[cfg(not(feature = "hip-real"))]
fn gemm_scaled_stub(
    m: usize,
    n: usize,
    k: usize,
    scale: f32,
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
) {
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for inner in 0..k {
                let lhs_index = row * k + inner;
                let rhs_index = inner * n + col;
                acc += lhs[lhs_index] * rhs[rhs_index];
            }
            out[row * n + col] = acc * scale;
        }
    }
}

#[cfg(not(feature = "hip-real"))]
fn gemm_lhs_transpose_scaled_stub(
    m: usize,
    n: usize,
    k: usize,
    scale: f32,
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
) {
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for inner in 0..k {
                let lhs_index = inner * m + row;
                let rhs_index = inner * n + col;
                acc += lhs[lhs_index] * rhs[rhs_index];
            }
            out[row * n + col] = acc * scale;
        }
    }
}

#[cfg(not(feature = "hip-real"))]
fn apply_gemm_epilogue_stub(
    m: usize,
    n: usize,
    activation: GemmActivation,
    bias: &[f32],
    residual: Option<&[f32]>,
    out: &mut [f32],
) {
    const GELU_COEFF: f32 = 0.044_715;
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;

    for row in 0..m {
        for (col, &bias_value) in bias.iter().enumerate() {
            let index = row * n + col;
            let mut value = out[index] + bias_value;
            if let Some(residual) = residual {
                value += residual[index];
            }
            out[index] = match activation {
                GemmActivation::Relu => {
                    if value > 0.0 {
                        value
                    } else {
                        0.0
                    }
                }
                GemmActivation::Gelu => {
                    let cubed = value * value * value;
                    0.5 * value * (1.0 + (SQRT_2_OVER_PI * (value + GELU_COEFF * cubed)).tanh())
                }
            };
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
    gemm_scaled_f32(m, n, k, 1.0, lhs, rhs, out)
}

/// Computes `out = (lhs @ rhs) * scale` for contiguous row-major buffers.
pub fn gemm_scaled_f32(
    m: usize,
    n: usize,
    k: usize,
    scale: f32,
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
) -> Result<(), HipErr> {
    init()?;
    validate_gemm_dimensions(m, n, k, lhs, rhs, out)?;
    validate_gemm_scale(scale)?;
    gemm_scaled_backend(m, n, k, scale, lhs, rhs, out)
}

#[cfg(feature = "hip-real")]
fn gemm_scaled_backend(
    m: usize,
    n: usize,
    k: usize,
    scale: f32,
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
) -> Result<(), HipErr> {
    crate::real::gemm_scaled_f32(m, n, k, scale, lhs, rhs, out)
}

#[cfg(not(feature = "hip-real"))]
fn gemm_scaled_backend(
    m: usize,
    n: usize,
    k: usize,
    scale: f32,
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
) -> Result<(), HipErr> {
    gemm_scaled_stub(m, n, k, scale, lhs, rhs, out);
    Ok(())
}

/// Computes `out = (lhs.T @ rhs) * scale` without materializing `lhs.T`.
///
/// `lhs` has shape `k x m`, `rhs` has shape `k x n`, and `out` has shape
/// `m x n`, all in contiguous row-major storage.
pub fn gemm_lhs_transpose_scaled_f32(
    m: usize,
    n: usize,
    k: usize,
    scale: f32,
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
) -> Result<(), HipErr> {
    init()?;
    validate_gemm_dimensions(m, n, k, lhs, rhs, out)?;
    validate_gemm_scale(scale)?;
    gemm_lhs_transpose_scaled_backend(m, n, k, scale, lhs, rhs, out)
}

/// Computes a row-major GEMM followed by bias, optional residual, and activation.
///
/// Real HIP builds keep the GEMM output on-device, execute the epilogue on the
/// same stream, and copy only the final result back to the host.
#[allow(clippy::too_many_arguments)]
pub fn gemm_bias_activation_f32(
    m: usize,
    n: usize,
    k: usize,
    activation: GemmActivation,
    lhs: &[f32],
    rhs: &[f32],
    bias: &[f32],
    residual: Option<&[f32]>,
    out: &mut [f32],
) -> Result<(), HipErr> {
    init()?;
    validate_gemm_dimensions(m, n, k, lhs, rhs, out)?;
    validate_gemm_epilogue(m, n, bias, residual)?;
    gemm_bias_activation_backend(m, n, k, activation, lhs, rhs, bias, residual, out)
}

#[cfg(feature = "hip-real")]
#[allow(clippy::too_many_arguments)]
fn gemm_bias_activation_backend(
    m: usize,
    n: usize,
    k: usize,
    activation: GemmActivation,
    lhs: &[f32],
    rhs: &[f32],
    bias: &[f32],
    residual: Option<&[f32]>,
    out: &mut [f32],
) -> Result<(), HipErr> {
    crate::real::gemm_bias_activation_f32(m, n, k, activation, lhs, rhs, bias, residual, out)
}

#[cfg(not(feature = "hip-real"))]
#[allow(clippy::too_many_arguments)]
fn gemm_bias_activation_backend(
    m: usize,
    n: usize,
    k: usize,
    activation: GemmActivation,
    lhs: &[f32],
    rhs: &[f32],
    bias: &[f32],
    residual: Option<&[f32]>,
    out: &mut [f32],
) -> Result<(), HipErr> {
    gemm_scaled_stub(m, n, k, 1.0, lhs, rhs, out);
    apply_gemm_epilogue_stub(m, n, activation, bias, residual, out);
    Ok(())
}

#[cfg(feature = "hip-real")]
fn gemm_lhs_transpose_scaled_backend(
    m: usize,
    n: usize,
    k: usize,
    scale: f32,
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
) -> Result<(), HipErr> {
    crate::real::gemm_lhs_transpose_scaled_f32(m, n, k, scale, lhs, rhs, out)
}

#[cfg(not(feature = "hip-real"))]
fn gemm_lhs_transpose_scaled_backend(
    m: usize,
    n: usize,
    k: usize,
    scale: f32,
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
) -> Result<(), HipErr> {
    gemm_lhs_transpose_scaled_stub(m, n, k, scale, lhs, rhs, out);
    Ok(())
}

/// Perform a GEMM operation using raw pointers.
///
/// # Safety
/// - `lhs` must be non-null, properly aligned, and point to `m * k` contiguous `f32` values.
/// - `rhs` must be non-null, properly aligned, and point to `k * n` contiguous `f32` values.
/// - `out` must be non-null, properly aligned, and point to `m * n` writable `f32` values.
/// - The referenced memory must be valid for the duration of the call and must not alias in a way
///   that violates Rust's aliasing rules.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::panic::{catch_unwind, AssertUnwindSafe};
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
    fn rccl_topology_contract_rejects_invalid_rank_domains() {
        assert_eq!(validate_rccl_topology(4, 3).unwrap(), 4);
        for (world, rank) in [(0, 0), (2, -1), (2, 2)] {
            let error = validate_rccl_topology(world, rank).unwrap_err();
            assert!(error.to_string().contains("invalid topology"));
        }
    }

    #[test]
    fn rccl_allgather_layout_uses_checked_counts_and_bytes() {
        assert_eq!(
            rccl_allgather_layout(3, 4).unwrap(),
            RcclAllGatherLayout {
                receive_count: 12,
                send_bytes: 24,
                receive_bytes: 96,
            }
        );
        assert_eq!(
            rccl_allgather_layout(0, 4).unwrap(),
            RcclAllGatherLayout {
                receive_count: 0,
                send_bytes: 0,
                receive_bytes: 0,
            }
        );
    }

    #[test]
    fn rccl_allgather_layout_rejects_zero_world_and_every_overflow_stage() {
        assert!(rccl_allgather_layout(1, 0)
            .unwrap_err()
            .to_string()
            .contains("world size"));
        assert!(rccl_allgather_layout(usize::MAX, 2)
            .unwrap_err()
            .to_string()
            .contains("receive count"));
        assert!(
            rccl_allgather_layout(usize::MAX / std::mem::size_of::<u64>() + 1, 1)
                .unwrap_err()
                .to_string()
                .contains("send byte")
        );
        assert!(rccl_allgather_layout(usize::MAX / 16 + 1, 2)
            .unwrap_err()
            .to_string()
            .contains("receive byte"));
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
    fn runtime_survives_poisoned_slot() {
        let _guard = ENV_MUTEX.lock().unwrap();
        super::reset_runtime_for_tests();
        let prev_force = std::env::var_os("SPIRALTORCH_FORCE_HIP");
        std::env::set_var("SPIRALTORCH_FORCE_HIP", "1");
        let initialized = super::init().expect("runtime should initialise when forced");

        let poisoned = catch_unwind(AssertUnwindSafe(|| {
            let _runtime_guard = super::runtime_slot().lock().unwrap();
            panic!("poison HIP runtime slot");
        }));
        assert!(poisoned.is_err());
        assert!(super::runtime_slot().is_poisoned());

        let recovered = super::runtime().expect("poisoned runtime slot should recover");

        assert!(Arc::ptr_eq(&initialized, &recovered));
        assert!(!super::runtime_slot().is_poisoned());
        restore_env("SPIRALTORCH_FORCE_HIP", prev_force);
        super::reset_runtime_for_tests();
    }

    #[test]
    fn init_recovers_after_empty_slot_poison() {
        let _guard = ENV_MUTEX.lock().unwrap();
        super::reset_runtime_for_tests();
        let prev_force = std::env::var_os("SPIRALTORCH_FORCE_HIP");
        std::env::set_var("SPIRALTORCH_FORCE_HIP", "1");

        let poisoned = catch_unwind(AssertUnwindSafe(|| {
            let _runtime_guard = super::runtime_slot().lock().unwrap();
            panic!("poison empty HIP runtime slot");
        }));
        assert!(poisoned.is_err());

        let runtime = super::init().expect("initialization should recover a poisoned slot");

        assert!(runtime.device_count() >= 1);
        assert!(!super::runtime_slot().is_poisoned());
        restore_env("SPIRALTORCH_FORCE_HIP", prev_force);
        super::reset_runtime_for_tests();
    }

    #[test]
    fn concurrent_init_converges_on_one_runtime() {
        let _guard = ENV_MUTEX.lock().unwrap();
        super::reset_runtime_for_tests();
        let prev_force = std::env::var_os("SPIRALTORCH_FORCE_HIP");
        std::env::set_var("SPIRALTORCH_FORCE_HIP", "1");

        let workers = (0..8)
            .map(|_| std::thread::spawn(super::init))
            .collect::<Vec<_>>();
        let runtimes = workers
            .into_iter()
            .map(|worker| {
                worker
                    .join()
                    .expect("initialization worker panicked")
                    .expect("runtime initialization failed")
            })
            .collect::<Vec<_>>();

        assert!(runtimes
            .iter()
            .skip(1)
            .all(|runtime| Arc::ptr_eq(&runtimes[0], runtime)));
        restore_env("SPIRALTORCH_FORCE_HIP", prev_force);
        super::reset_runtime_for_tests();
    }

    #[test]
    fn runtime_device_resolution_uses_real_enumeration_without_hints() {
        let devices = super::resolve_runtime_devices(Vec::new(), || {
            Ok(vec![DeviceInfo::new(2, "gfx-test", false)])
        })
        .expect("real device enumeration should succeed");

        assert_eq!(devices, vec![DeviceInfo::new(2, "gfx-test", false)]);
    }

    #[test]
    fn runtime_device_resolution_prefers_environment_hints() {
        let hinted = vec![DeviceInfo::new(4, "visible-device-4", false)];

        let devices = super::resolve_runtime_devices(hinted.clone(), || {
            panic!("real enumeration should not run when environment hints are present")
        })
        .expect("environment device hints should resolve");

        assert_eq!(devices, hinted);
    }

    #[test]
    fn runtime_device_resolution_rejects_empty_enumeration() {
        let err = super::resolve_runtime_devices(Vec::new(), || Ok(Vec::new()))
            .expect_err("an empty real runtime must not report initialized");

        assert!(err.to_string().contains("no available devices"));
    }

    #[test]
    fn runtime_device_resolution_propagates_enumeration_failure() {
        let err = super::resolve_runtime_devices(Vec::new(), || {
            Err(HipErr::Other("enumeration failed".into()))
        })
        .expect_err("real runtime enumeration failures must remain visible");

        assert!(err.to_string().contains("enumeration failed"));
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

    #[test]
    fn real_backend_capability_matches_build_feature() {
        assert_eq!(super::real_backend_compiled(), cfg!(feature = "hip-real"));
    }

    #[cfg(not(feature = "hip-real"))]
    #[test]
    fn scaled_gemm_stub_fuses_scale_into_reference_result() {
        let _guard = ENV_MUTEX.lock().unwrap();
        super::reset_runtime_for_tests();
        let prev_force = std::env::var_os("SPIRALTORCH_FORCE_HIP");
        std::env::set_var("SPIRALTORCH_FORCE_HIP", "1");

        let lhs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut out = vec![0.0; 4];
        super::gemm_scaled_f32(2, 2, 3, 0.5, &lhs, &rhs, &mut out)
            .expect("scaled GEMM stub should succeed");

        assert_eq!(out, vec![29.0, 32.0, 69.5, 77.0]);
        restore_env("SPIRALTORCH_FORCE_HIP", prev_force);
        super::reset_runtime_for_tests();
    }

    #[cfg(not(feature = "hip-real"))]
    #[test]
    fn lhs_transpose_scaled_gemm_stub_matches_reference() {
        let _guard = ENV_MUTEX.lock().unwrap();
        super::reset_runtime_for_tests();
        let prev_force = std::env::var_os("SPIRALTORCH_FORCE_HIP");
        std::env::set_var("SPIRALTORCH_FORCE_HIP", "1");

        let lhs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut out = vec![0.0; 4];
        super::gemm_lhs_transpose_scaled_f32(2, 2, 3, 0.25, &lhs, &rhs, &mut out)
            .expect("transpose-scaled GEMM stub should succeed");

        assert_eq!(out, vec![22.25, 24.5, 29.0, 32.0]);
        restore_env("SPIRALTORCH_FORCE_HIP", prev_force);
        super::reset_runtime_for_tests();
    }

    #[cfg(not(feature = "hip-real"))]
    #[test]
    fn fused_gemm_epilogue_stub_matches_all_activation_and_residual_modes() {
        let _guard = ENV_MUTEX.lock().unwrap();
        super::reset_runtime_for_tests();
        let prev_force = std::env::var_os("SPIRALTORCH_FORCE_HIP");
        std::env::set_var("SPIRALTORCH_FORCE_HIP", "1");

        let lhs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let bias = vec![-60.0, 1.0];
        let residual = vec![3.0, -70.0, -140.0, 2.0];

        let mut relu = vec![0.0; 4];
        super::gemm_bias_activation_f32(
            2,
            2,
            3,
            GemmActivation::Relu,
            &lhs,
            &rhs,
            &bias,
            None,
            &mut relu,
        )
        .expect("bias+ReLU stub should succeed");
        assert_eq!(relu, vec![0.0, 65.0, 79.0, 155.0]);

        let mut residual_relu = vec![0.0; 4];
        super::gemm_bias_activation_f32(
            2,
            2,
            3,
            GemmActivation::Relu,
            &lhs,
            &rhs,
            &bias,
            Some(&residual),
            &mut residual_relu,
        )
        .expect("bias+residual+ReLU stub should succeed");
        assert_eq!(residual_relu, vec![1.0, 0.0, 0.0, 157.0]);

        let gelu = |value: f32| {
            let cubed = value * value * value;
            0.5 * value * (1.0 + (0.797_884_6 * (value + 0.044_715 * cubed)).tanh())
        };
        for (residual, pre_activation) in [
            (None, vec![-2.0, 65.0, 79.0, 155.0]),
            (Some(residual.as_slice()), vec![1.0, -5.0, -61.0, 157.0]),
        ] {
            let mut actual = vec![0.0; 4];
            super::gemm_bias_activation_f32(
                2,
                2,
                3,
                GemmActivation::Gelu,
                &lhs,
                &rhs,
                &bias,
                residual,
                &mut actual,
            )
            .expect("GELU epilogue stub should succeed");
            for (actual, expected) in actual.iter().zip(pre_activation.into_iter().map(gelu)) {
                assert!((actual - expected).abs() <= 1e-6);
            }
        }

        restore_env("SPIRALTORCH_FORCE_HIP", prev_force);
        super::reset_runtime_for_tests();
    }

    #[cfg(not(feature = "hip-real"))]
    #[test]
    fn fused_gemm_epilogue_validates_inputs_before_mutating_output() {
        let _guard = ENV_MUTEX.lock().unwrap();
        super::reset_runtime_for_tests();
        let prev_force = std::env::var_os("SPIRALTORCH_FORCE_HIP");
        std::env::set_var("SPIRALTORCH_FORCE_HIP", "1");

        let lhs = vec![1.0, 2.0, 3.0, 4.0];
        let rhs = vec![5.0, 6.0, 7.0, 8.0];
        let mut out = vec![13.0; 4];
        let bad_bias = super::gemm_bias_activation_f32(
            2,
            2,
            2,
            GemmActivation::Relu,
            &lhs,
            &rhs,
            &[1.0],
            None,
            &mut out,
        )
        .expect_err("wrong bias length must be rejected");
        assert!(bad_bias.to_string().contains("bias buffer length"));
        assert_eq!(out, vec![13.0; 4]);

        let bad_residual = super::gemm_bias_activation_f32(
            2,
            2,
            2,
            GemmActivation::Gelu,
            &lhs,
            &rhs,
            &[1.0, 2.0],
            Some(&[3.0]),
            &mut out,
        )
        .expect_err("wrong residual length must be rejected");
        assert!(bad_residual.to_string().contains("residual buffer length"));
        assert_eq!(out, vec![13.0; 4]);

        restore_env("SPIRALTORCH_FORCE_HIP", prev_force);
        super::reset_runtime_for_tests();
    }

    #[cfg(not(feature = "hip-real"))]
    #[test]
    fn fused_gemm_epilogue_preserves_zero_inner_semantics() {
        let _guard = ENV_MUTEX.lock().unwrap();
        super::reset_runtime_for_tests();
        let prev_force = std::env::var_os("SPIRALTORCH_FORCE_HIP");
        std::env::set_var("SPIRALTORCH_FORCE_HIP", "1");

        let bias = vec![-1.0, 2.0];
        let residual = vec![0.5, -3.0, 4.0, 5.0];
        let mut out = vec![99.0; 4];
        super::gemm_bias_activation_f32(
            2,
            2,
            0,
            GemmActivation::Relu,
            &[],
            &[],
            &bias,
            Some(&residual),
            &mut out,
        )
        .expect("zero-inner fused GEMM should run its epilogue");

        assert_eq!(out, vec![0.0, 0.0, 3.0, 7.0]);
        restore_env("SPIRALTORCH_FORCE_HIP", prev_force);
        super::reset_runtime_for_tests();
    }

    #[cfg(not(feature = "hip-real"))]
    #[test]
    fn scaled_gemm_rejects_non_finite_scale_without_mutating_output() {
        let _guard = ENV_MUTEX.lock().unwrap();
        super::reset_runtime_for_tests();
        let prev_force = std::env::var_os("SPIRALTORCH_FORCE_HIP");
        std::env::set_var("SPIRALTORCH_FORCE_HIP", "1");

        let lhs = vec![1.0, 2.0, 3.0, 4.0];
        let rhs = vec![5.0, 6.0, 7.0, 8.0];
        let mut out = vec![13.0; 4];
        let error = super::gemm_scaled_f32(2, 2, 2, f32::INFINITY, &lhs, &rhs, &mut out)
            .expect_err("non-finite scale must be rejected");

        assert!(error.to_string().contains("scale must be finite"));
        assert_eq!(out, vec![13.0; 4]);
        restore_env("SPIRALTORCH_FORCE_HIP", prev_force);
        super::reset_runtime_for_tests();
    }
}
