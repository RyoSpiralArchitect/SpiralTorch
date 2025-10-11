
use crate::error::Result;
use ndarray::ArrayD;

#[derive(Clone)]
pub enum BackendArrayF32 {
    Cpu{ data: ArrayD<f32> },
    #[cfg(feature="cuda")]
    Cuda{ rows: usize, cols: usize, ptr: std::sync::Arc<cust::memory::DeviceBuffer<f32>> },
    #[cfg(feature="wgpu")]
    Wgpu{ rows: usize, cols: usize, buffer: wgpu::Buffer },
    #[cfg(feature="mps")]
    Mps{ rows: usize, cols: usize, buffer: metal::Buffer },
}

#[cfg(feature="wgpu")]
pub mod wgpu_topk_unified;
#[cfg(feature="wgpu")]
pub mod wgpu_lse_f16;
#[cfg(feature="wgpu")]
pub mod wgpu_ce_f16;

#[cfg(feature="cuda")]
pub mod cuda_where;
#[cfg(feature="mps")]
pub mod mps_where;

#[cfg(feature="wgpu")]
pub mod wgpu_kernels_placeholder {
    // Single placeholder so include_str! in backends can concatenate with `wgpu_kernels_all.wgsl`
    pub const ALL: &str = "// base WGSL library placeholder";
}

#[cfg(feature="mps")]
pub mod mps_pool_autotune { pub struct PoolAutoTune { pub enabled: bool, pub bins: std::collections::BTreeMap<u32, usize>, pub hits: usize, pub misses: usize, pub tune_interval: std::time::Duration, pub window_elapsed: std::time::Duration, pub max_grow: f32 }
impl PoolAutoTune { pub fn new()->Self{ Self{ enabled:false, bins:Default::default(), hits:0, misses:0, tune_interval:std::time::Duration::from_secs(10), window_elapsed:std::time::Duration::from_secs(0), max_grow:2.0 } }
pub fn enable(mut self)->Self{ self.enabled=true; self }
pub fn maybe_tune(&mut self){} }}

#[cfg(feature="mps")]
pub mod mps_pool_window { include!("mps_pool_window.append.rs"); }

#[cfg(feature="mps")]
pub mod mps_pool_auto_explore { include!("mps_pool_auto_explore.append.rs"); }
