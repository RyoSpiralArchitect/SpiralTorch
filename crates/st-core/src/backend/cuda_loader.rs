// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// crates/st-core/src/backend/cuda_loader.rs

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaFunction};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::compile_ptx;
#[cfg(feature = "cuda")]
use cudarc::nvrtc::Ptx;
#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::panic::{catch_unwind, AssertUnwindSafe};
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex, OnceLock};

// ===== SpiralTorch: CUDA NVRTC compile helpers (feature-gated) =====
#[derive(Debug, thiserror::Error)]
pub enum StCudaNvrtcError {
    #[error("NVRTC unavailable (build without 'nvrtc' feature)")]
    NvrtcUnavailable,
    #[error("NVRTC compile failed: {0}")]
    Nvrtc(String),
    #[error("empty PTX output")]
    EmptyPtx,
}

pub struct NvrtcOptions<'a> {
    pub arch: &'a str,        // e.g. "compute_80"
    pub code: &'a str,        // e.g. "sm_80"
    pub std: &'a str,         // "c++14" / "c++17"
    pub maxrreg: Option<u32>, // optional register cap
}

#[cfg(feature = "cuda")]
pub fn guard_cuda_call<T>(
    context: &str,
    f: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(result) => result.map_err(|err| format!("{context}: {err}")),
        Err(payload) => Err(format!("{context}: {}", panic_payload_to_string(payload))),
    }
}

#[cfg(feature = "cuda")]
pub fn safe_compile_ptx(src: &str) -> Result<Ptx, String> {
    guard_cuda_call("cuda nvrtc compile failed", || {
        compile_ptx(src).map_err(|err| err.to_string())
    })
}

#[cfg(feature = "cuda")]
fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(msg) = payload.downcast_ref::<String>() {
        msg.clone()
    } else if let Some(msg) = payload.downcast_ref::<&'static str>() {
        (*msg).to_string()
    } else {
        "unknown panic payload".to_string()
    }
}

pub fn default_nvrtc_options<'a>() -> NvrtcOptions<'a> {
    NvrtcOptions {
        arch: "compute_80",
        code: "sm_80",
        std: "c++14",
        maxrreg: None,
    }
}

/// Compile CUDA source to PTX with NVRTC (実体はリポの FFI に繋いでください)
pub fn st_compile_with_nvrtc(src: &str, opts: &NvrtcOptions) -> Result<String, StCudaNvrtcError> {
    #[cfg(feature = "cuda")]
    {
        let mut flags = vec![
            format!("--gpu-architecture={}", opts.arch),
            format!("--gpu-code={}", opts.code),
            format!("--std={}", opts.std),
            "--use_fast_math".into(),
            "-default-device".into(),
            "-D__CUDA_NO_HALF_OPERATORS__".into(),
        ];
        if let Some(r) = opts.maxrreg {
            flags.push(format!("--maxrregcount={}", r));
        }
        let ptx = nvrtc_compile_shim(src, &flags).map_err(StCudaNvrtcError::Nvrtc)?;
        if ptx.trim().is_empty() {
            return Err(StCudaNvrtcError::EmptyPtx);
        }
        Ok(ptx)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(StCudaNvrtcError::NvrtcUnavailable)
    }
}

#[cfg(feature = "cuda")]
fn nvrtc_compile_shim(_src: &str, _flags: &[String]) -> Result<String, String> {
    // 実環境ではここを既存の NVRTC FFI 呼び出しに置換
    Err("nvrtc shim not wired".into())
}
// ===== end additions =====

#[cfg(feature = "cuda")]
#[derive(Clone)]
pub struct CudaModule {
    device: Arc<CudaDevice>,
    module_name: &'static str,
    entry: Arc<ModuleEntry>,
}

#[cfg(feature = "cuda")]
impl CudaModule {
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    pub fn module_name(&self) -> &'static str {
        &self.module_name
    }

    pub fn get_func(&self, func_name: &'static str) -> Result<CudaFunction, String> {
        let state = self
            .entry
            .state
            .lock()
            .map_err(|_| "cuda module state poisoned".to_string())?;

        state.functions.get(func_name).cloned().ok_or_else(|| {
            format!(
                "cuda function `{func_name}` not registered in module `{}`",
                self.module_name
            )
        })
    }
}

#[cfg(feature = "cuda")]
struct ModuleState {
    functions: HashMap<&'static str, CudaFunction>,
}

#[cfg(feature = "cuda")]
impl ModuleState {
    fn new() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }
}

#[cfg(feature = "cuda")]
struct ModuleEntry {
    state: Mutex<ModuleState>,
}

#[cfg(feature = "cuda")]
impl ModuleEntry {
    fn new() -> Self {
        Self {
            state: Mutex::new(ModuleState::new()),
        }
    }
}

#[cfg(feature = "cuda")]
static DEVICE: OnceLock<Arc<CudaDevice>> = OnceLock::new();
#[cfg(feature = "cuda")]
static MODULES: OnceLock<Mutex<HashMap<&'static str, Arc<ModuleEntry>>>> = OnceLock::new();

#[cfg(feature = "cuda")]
fn global_device() -> Result<Arc<CudaDevice>, String> {
    if let Some(device) = DEVICE.get() {
        return Ok(Arc::clone(device));
    }

    let created = guard_cuda_call("cuda device initialization failed", || {
        CudaDevice::new(0)
            .map(Arc::new)
            .map_err(|err| err.to_string())
    })?;
    let _ = DEVICE.set(Arc::clone(&created));
    Ok(Arc::clone(DEVICE.get().unwrap_or(&created)))
}

#[cfg(feature = "cuda")]
fn registry() -> &'static Mutex<HashMap<&'static str, Arc<ModuleEntry>>> {
    MODULES.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(feature = "cuda")]
pub fn load_ptx_module(
    ptx: &Ptx,
    module_name: &'static str,
    functions: &[&'static str],
) -> Result<CudaModule, String> {
    let device = global_device()?;

    let entry = {
        let mut modules = registry()
            .lock()
            .map_err(|_| "cuda module registry poisoned".to_string())?;
        modules
            .entry(module_name)
            .or_insert_with(|| Arc::new(ModuleEntry::new()))
            .clone()
    };

    {
        let mut state = entry
            .state
            .lock()
            .map_err(|_| "cuda module state poisoned".to_string())?;

        if !functions
            .iter()
            .all(|name| state.functions.contains_key(name))
        {
            let mut requested: Vec<&'static str> = functions.iter().copied().collect();
            requested.sort_unstable();
            requested.dedup();

            let mut union: Vec<&'static str> = state.functions.keys().copied().collect();
            union.extend(requested);
            union.sort_unstable();
            union.dedup();

            guard_cuda_call("cuda PTX module load failed", || {
                device
                    .load_ptx(ptx.clone(), module_name, &union)
                    .map_err(|err| err.to_string())
            })?;

            state.functions.clear();

            for &name in &union {
                let func = guard_cuda_call("cuda function lookup failed", || {
                    device.get_func(module_name, name).ok_or_else(|| {
                        format!("cuda function `{name}` not registered in module `{module_name}`")
                    })
                })?;
                state.functions.insert(name, func);
            }
        }
    }

    Ok(CudaModule {
        device,
        module_name,
        entry,
    })
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::{guard_cuda_call, panic_payload_to_string};

    #[test]
    fn guard_cuda_call_converts_panics_into_errors() {
        let err = guard_cuda_call("cuda path", || -> Result<(), String> {
            panic!("boom");
        })
        .expect_err("panic should be surfaced as an error");

        assert!(err.contains("cuda path"));
        assert!(err.contains("boom"));
    }

    #[test]
    fn panic_payload_string_handles_unknown_payloads() {
        let payload: Box<dyn std::any::Any + Send> = Box::new(123usize);
        assert_eq!(panic_payload_to_string(payload), "unknown panic payload");
    }
}
