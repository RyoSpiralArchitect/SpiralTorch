
use ndarray::ArrayD;
use crate::{device::Device, error::Result};

#[cfg(all(feature="mps", target_os="macos"))]
pub mod mps_impl;
#[cfg(feature="wgpu")]
pub mod wgpu_impl;
#[cfg(feature="cuda")]
pub mod cuda_impl;

#[cfg(all(feature="mps", target_os="macos"))] pub use mps_impl::{MpsBackend, BackendArrayF32 as MpsArray, Backend as MpsTrait};
#[cfg(feature="wgpu")] pub use wgpu_impl::{WgpuBackend, BackendArrayF32 as WgpuArray, Backend as WgpuTrait};
#[cfg(feature="cuda")] pub use cuda_impl::{CudaBackend, BackendArrayF32 as CudaArray, Backend as CudaTrait};

/// Unified tagged array handle for device-enabled builds (VERY simplified for this bundle).
pub enum BackendArrayF32 {
    #[cfg(all(feature="mps", target_os="macos"))] Mps { rows: usize, cols: usize, buffer: metal::Buffer },
    #[cfg(feature="wgpu")] Wgpu { rows: usize, cols: usize, buffer: wgpu::Buffer },
    #[cfg(feature="cuda")] Cuda { rows: usize, cols: usize, ptr: std::sync::Arc<cust::memory::DeviceBuffer<f32>> },
    HostStub,
}

pub trait Backend {
    fn name(&self)->&'static str;
    fn device(&self)->Device;
    fn from_host_f32(&self, host:&ArrayD<f32>) -> Result<BackendArrayF32>;
    fn to_host_f32(&self, arr:&BackendArrayF32) -> Result<ArrayD<f32>>;
}
