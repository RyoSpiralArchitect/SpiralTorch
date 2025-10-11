use ndarray::ArrayD;
use crate::{device::Device, error::Result};

#[cfg(all(feature="mps", target_os="macos"))] pub mod mps;
#[cfg(feature="wgpu")] pub mod wgpu;
#[cfg(feature="cuda")] pub mod cuda;

#[cfg(all(feature="mps", target_os="macos"))] pub use mps::MpsBackend;
#[cfg(feature="wgpu")] pub use wgpu::WgpuBackend;
#[cfg(feature="cuda")] pub use cuda::CudaBackend;

/// Unified device buffer handle (very simplified).
pub enum BackendArrayF32 {
    #[cfg(feature="wgpu")] Wgpu { rows: usize, cols: usize, buffer: wgpu::Buffer },
    #[cfg(all(feature="mps", target_os="macos"))] Mps { rows: usize, cols: usize, buffer: metal::Buffer },
    #[cfg(feature="cuda")] Cuda { rows: usize, cols: usize, ptr: std::sync::Arc<cust::memory::DeviceBuffer<f32>> },
    HostStub,
}

pub trait BackendIo {
    fn from_host_f32(&self, host:&ArrayD<f32>) -> Result<BackendArrayF32>;
    fn to_host_f32(&self, arr:&BackendArrayF32) -> Result<ArrayD<f32>>;
}

pub fn ensure_host(arr: &BackendArrayF32, device: Device) -> Result<ArrayD<f32>> {
    match device {
        #[cfg(feature="wgpu")]
        Device::Wgpu => WgpuBackend::new().to_host_f32(arr),
        #[cfg(all(feature="mps", target_os="macos"))]
        Device::Mps => MpsBackend::new().to_host_f32(arr),
        #[cfg(feature="cuda")]
        Device::Cuda => CudaBackend::new().to_host_f32(arr),
        Device::Cpu => Err(crate::error::device("HostStub only on GPU devices")),
    }
}
