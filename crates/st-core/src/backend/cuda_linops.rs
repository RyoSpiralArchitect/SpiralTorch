//! CUDA DeviceOps skeleton for unified CG (v1.8.2)
// Implement DeviceOps (dot/axpy/copy) using CUDA buffers & kernels.
// Fill with PTX launches or cudarc kernels; API matches WGPU/HIP siblings.

use crate::ops::hypergrad_gpu::{DeviceOps, DeviceBuf};

pub struct CudaOps { /* keep CUDA device/stream/handles here */ }

impl CudaOps {
    pub fn new() -> Self { Self{} }
}

impl DeviceOps for CudaOps {
    fn dot(&self, _n: usize, _x:&DeviceBuf, _y:&DeviceBuf) -> Result<f32, String> {
        Err("CudaOps::dot not wired in this overlay".into())
    }
    fn axpy(&self, _n: usize, _alpha:f32, _x:&DeviceBuf, _y:&DeviceBuf, _out:&DeviceBuf) -> Result<(), String> {
        Err("CudaOps::axpy not wired in this overlay".into())
    }
    fn copy(&self, _n: usize, _src:&DeviceBuf, _dst:&DeviceBuf) -> Result<(), String> {
        Err("CudaOps::copy not wired in this overlay".into())
    }
}
