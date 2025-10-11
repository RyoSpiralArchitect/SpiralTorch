
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
pub mod wgpu_topk_bigk;
#[cfg(feature="wgpu")]
pub mod wgpu_topk_blockmerge;
#[cfg(feature="wgpu")]
pub mod wgpu_topk_blockmerge_kway;
#[cfg(feature="wgpu")]
pub mod wgpu_lse_f16;
#[cfg(feature="wgpu")]
pub mod wgpu_ce_f16;
#[cfg(feature="wgpu")]
pub mod wgpu_ce_reduce;
#[cfg(feature="wgpu")]
pub mod wgpu_ce_fused_nll;
#[cfg(feature="wgpu")]
pub mod wgpu_ce_full_fused;

#[cfg(feature="cuda")]
pub mod cuda_where;
#[cfg(feature="mps")]
pub mod mps_where;

#[cfg(feature="wgpu")]
pub const WGPU_KERNELS_ALL: &str = include_str!("wgpu_kernels_all.wgsl");

#[cfg(feature="mps")]
pub mod mps_pool_autotune {
    pub struct PoolAutoTune {
        pub enabled: bool,
        pub bins: std::collections::BTreeMap<u32, usize>,
        pub hits: usize, pub misses: usize,
        pub tune_interval: std::time::Duration,
        pub window_elapsed: std::time::Duration,
        pub max_grow: f32,
    }
    impl PoolAutoTune {
        pub fn new()->Self{ Self{ enabled:false, bins:Default::default(), hits:0, misses:0, tune_interval:std::time::Duration::from_secs(10), window_elapsed:std::time::Duration::from_secs(0), max_grow:2.0 } }
        pub fn enable(mut self)->Self{ self.enabled=true; self }
        pub fn maybe_tune(&mut self) {}
        pub fn set_window_secs(&mut self, s:u64){ self.tune_interval = std::time::Duration::from_secs(s); }
        pub fn set_explore_range(&mut self, g:f32){ self.max_grow = g; }
    }
}

#[cfg(feature="mps")]
pub mod mps_pool_window { include!("mps_pool_window.append.rs"); }
#[cfg(feature="mps")]
pub mod mps_pool_auto_explore { include!("mps_pool_auto_explore.append.rs"); }
#[cfg(feature="mps")]
pub mod mps_pool_safety { include!("mps_pool_safety.append.rs"); }
#[cfg(feature="mps")]
pub mod mps_pool_ema { include!("mps_pool_ema.append.rs"); }
#[cfg(feature="mps")]
pub mod mps_pool_dynamic { include!("mps_pool_dynamic_ema_debounce.append.rs"); }
