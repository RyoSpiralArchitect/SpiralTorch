//! WGPU DeviceOps minimal impl (v1.8.0)
#![allow(unused)]
use super::wgpu_rt;
use super::wgpu_rt::util_reexp::*;

pub struct WgpuOps {
    pub ctx: std::sync::Arc<wgpu_rt::WgpuCtx>,
}
impl WgpuOps {
    pub fn new(ctx: std::sync::Arc<wgpu_rt::WgpuCtx>) -> Self { Self{ ctx } }
    fn layout(&self)->&wgpu::BindGroupLayout { wgpu_rt::ensure_layout_lin(&self.ctx) }
}
use crate::ops::hypergrad_gpu::{DeviceOps, DeviceBuf};

impl DeviceOps for WgpuOps {
    fn dot(&self, n: usize, x:&DeviceBuf, y:&DeviceBuf) -> Result<f32, String> {
        // In real impl, DeviceBuf holds wgpu::Buffer; here we assume we can downcast (omitted).
        Err("wire DeviceBuf<->wgpu::Buffer in binder to use WgpuOps::dot".into())
    }
    fn axpy(&self, _n: usize, _alpha:f32, _x:&DeviceBuf, _y:&DeviceBuf, _out:&DeviceBuf) -> Result<(), String> {
        Err("wire DeviceBuf<->wgpu::Buffer in binder to use WgpuOps::axpy".into())
    }
    fn copy(&self, _n: usize, _src:&DeviceBuf, _dst:&DeviceBuf) -> Result<(), String> {
        Err("wire DeviceBuf<->wgpu::Buffer in binder to use WgpuOps::copy".into())
    }
}
