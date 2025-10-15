// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! HIP DeviceOps skeleton for unified CG (v1.8.2)
// Implement DeviceOps (dot/axpy/copy) using HIP buffers/kernels or rocBLAS/hipBLASLt.

use crate::ops::hypergrad_gpu::{DeviceOps, DeviceBuf};

pub struct HipOps { /* keep HIP device/queue handles here */ }

impl HipOps {
    pub fn new() -> Self { Self{} }
}

impl DeviceOps for HipOps {
    fn dot(&self, _n: usize, _x:&DeviceBuf, _y:&DeviceBuf) -> Result<f32, String> {
        Err("HipOps::dot not wired in this overlay".into())
    }
    fn axpy(&self, _n: usize, _alpha:f32, _x:&DeviceBuf, _y:&DeviceBuf, _out:&DeviceBuf) -> Result<(), String> {
        Err("HipOps::axpy not wired in this overlay".into())
    }
    fn copy(&self, _n: usize, _src:&DeviceBuf, _dst:&DeviceBuf) -> Result<(), String> {
        Err("HipOps::copy not wired in this overlay".into())
    }
}
