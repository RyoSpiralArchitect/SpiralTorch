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
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(feature = "cuda")]
#[derive(Clone)]
pub struct CudaModule {
    device: Arc<CudaDevice>,
    module_name: &'static str,
}

#[cfg(feature = "cuda")]
impl CudaModule {
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    pub fn module_name(&self) -> &'static str {
        &self.module_name
    }

    pub fn get_func(&self, func_name: &'static str) -> Result<Arc<CudaFunction>, String> {
        module_state_get_func(self.module_name, func_name)
    }
}

#[cfg(feature = "cuda")]
struct ModuleState {
    functions: HashMap<&'static str, Arc<CudaFunction>>,
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
    DEVICE
        .get_or_try_init(|| CudaDevice::new(0).map_err(|err| err.to_string()))
        .map(Arc::clone)
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

    let mut state = entry
        .state
        .lock()
        .map_err(|_| "cuda module state poisoned".to_string())?;

    let mut to_register: Vec<&'static str> = if state.functions.is_empty() {
        functions.iter().copied().collect()
    } else {
        functions
            .iter()
            .copied()
            .filter(|name| !state.functions.contains_key(name))
            .collect()
    };

    if !to_register.is_empty() {
        to_register.sort_unstable();
        to_register.dedup();

        device
            .load_ptx(ptx.clone(), module_name, &to_register)
            .map_err(|err| err.to_string())?;

        for &name in &to_register {
            let func = device.get_func(module_name, name).ok_or_else(|| {
                format!("cuda function `{name}` not registered in module `{module_name}`")
            })?;
            state.functions.insert(name, Arc::new(func));
        }
    }

    Ok(CudaModule {
        device,
        module_name,
    })
}

#[cfg(feature = "cuda")]
fn module_state_get_func(
    module_name: &'static str,
    func_name: &'static str,
) -> Result<Arc<CudaFunction>, String> {
    let entry = {
        let modules = registry()
            .lock()
            .map_err(|_| "cuda module registry poisoned".to_string())?;
        modules
            .get(&module_name)
            .cloned()
            .ok_or_else(|| format!("cuda module `{module_name}` not loaded"))?
    };

    let state = entry
        .state
        .lock()
        .map_err(|_| "cuda module state poisoned".to_string())?;

    state.functions.get(func_name).cloned().ok_or_else(|| {
        format!("cuda function `{func_name}` not registered in module `{module_name}`")
    })
}
