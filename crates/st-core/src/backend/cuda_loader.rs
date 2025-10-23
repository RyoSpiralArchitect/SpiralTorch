// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// crates/st-core/src/backend/cuda_loader.rs

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaFunction};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::Ptx;
#[cfg(feature = "cuda")]
use std::collections::{HashMap, HashSet};
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(feature = "cuda")]
#[derive(Clone)]
pub struct CudaModule {
    device: Arc<CudaDevice>,
    module_name: String,
}

#[cfg(feature = "cuda")]
impl CudaModule {
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    pub fn module_name(&self) -> &str {
        &self.module_name
    }

    pub fn get_func(&self, func_name: &'static str) -> Result<CudaFunction, String> {
        self.device
            .get_func(&self.module_name, func_name)
            .ok_or_else(|| {
                format!(
                    "cuda function `{func_name}` not registered in module `{}`",
                    self.module_name
                )
            })
    }
}

#[cfg(feature = "cuda")]
struct ModuleState {
    registered: HashSet<&'static str>,
}

#[cfg(feature = "cuda")]
impl ModuleState {
    fn new() -> Self {
        Self {
            registered: HashSet::new(),
        }
    }
}

#[cfg(feature = "cuda")]
static DEVICE: OnceLock<Arc<CudaDevice>> = OnceLock::new();
#[cfg(feature = "cuda")]
static MODULES: OnceLock<Mutex<HashMap<String, ModuleState>>> = OnceLock::new();

#[cfg(feature = "cuda")]
fn global_device() -> Result<Arc<CudaDevice>, String> {
    DEVICE
        .get_or_try_init(|| CudaDevice::new(0).map_err(|err| err.to_string()))
        .map(|device| device.clone())
}

#[cfg(feature = "cuda")]
fn registry() -> &'static Mutex<HashMap<String, ModuleState>> {
    MODULES.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(feature = "cuda")]
pub fn load_ptx_module(
    ptx: &Ptx,
    module_name: &str,
    functions: &[&'static str],
) -> Result<CudaModule, String> {
    let device = global_device()?;
    let mut modules = registry()
        .lock()
        .map_err(|_| "cuda module registry poisoned".to_string())?;

    let state = modules
        .entry(module_name.to_string())
        .or_insert_with(ModuleState::new);

    let missing: Vec<&'static str> = functions
        .iter()
        .copied()
        .filter(|name| !state.registered.contains(name))
        .collect();

    if state.registered.is_empty() {
        device
            .load_ptx(ptx.clone(), module_name, functions)
            .map_err(|err| err.to_string())?;
        state.registered.extend(functions.iter().copied());
    } else if !missing.is_empty() {
        let mut names: Vec<&'static str> = state.registered.iter().copied().collect();
        names.extend(missing.iter().copied());
        device
            .load_ptx(ptx.clone(), module_name, &names)
            .map_err(|err| err.to_string())?;
        state.registered.extend(missing);
    }

    Ok(CudaModule {
        device,
        module_name: module_name.to_string(),
    })
}

