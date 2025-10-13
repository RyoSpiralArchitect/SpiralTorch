//! HIP DeviceOps skeleton (v1.8.3)
use crate::ops::hypergrad_gpu::{DeviceOps, DeviceBuf};
pub struct HipOps;
impl HipOps { pub fn new() -> Self { Self } }
impl DeviceOps for HipOps {
    fn dot(&self, _n: usize, _x:&DeviceBuf, _y:&DeviceBuf) -> Result<f32, String> { Err("HipOps::dot not wired".into()) }
    fn axpy(&self, _n: usize, _alpha:f32, _x:&DeviceBuf, _y:&DeviceBuf, _out:&DeviceBuf) -> Result<(), String> { Err("HipOps::axpy not wired".into()) }
    fn copy(&self, _n: usize, _src:&DeviceBuf, _dst:&DeviceBuf) -> Result<(), String> { Err("HipOps::copy not wired".into()) }
}
