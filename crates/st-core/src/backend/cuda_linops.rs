//! CUDA DeviceOps skeleton (v1.8.3)
use crate::ops::hypergrad_gpu::{DeviceOps, DeviceBuf};
pub struct CudaOps;
impl CudaOps { pub fn new() -> Self { Self } }
impl DeviceOps for CudaOps {
    fn dot(&self, _n: usize, _x:&DeviceBuf, _y:&DeviceBuf) -> Result<f32, String> { Err("CudaOps::dot not wired".into()) }
    fn axpy(&self, _n: usize, _alpha:f32, _x:&DeviceBuf, _y:&DeviceBuf, _out:&DeviceBuf) -> Result<(), String> { Err("CudaOps::axpy not wired".into()) }
    fn copy(&self, _n: usize, _src:&DeviceBuf, _dst:&DeviceBuf) -> Result<(), String> { Err("CudaOps::copy not wired".into()) }
}
