use ndarray::ArrayD;
use crate::{device::Device, error::Result};

#[cfg(all(feature="mps", target_os="macos"))]
pub mod mps_impl;

#[cfg(all(feature="mps", target_os="macos"))]
pub use mps_impl::MpsBackend;

#[cfg(all(feature="mps", target_os="macos"))]
pub enum BackendArrayF32 {
    Mps { rows: usize, cols: usize, buffer: metal::Buffer },
}

#[cfg(not(all(feature="mps", target_os="macos")))]
pub enum BackendArrayF32 {
    HostStub,
}

pub trait Backend {
    fn name(&self) -> &'static str;
    fn device(&self) -> Device;
    fn from_host_f32(&self, host: &ArrayD<f32>) -> Result<BackendArrayF32>;
    fn to_host_f32(&self, arr: &BackendArrayF32) -> Result<ArrayD<f32>>;
}
