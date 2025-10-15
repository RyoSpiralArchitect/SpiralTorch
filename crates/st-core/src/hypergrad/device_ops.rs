// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use once_cell::sync::OnceCell;

pub trait DeviceOps {
    fn dot(n: usize, x: *const f32, y:*const f32) -> f32;
    fn axpy(n: usize, a:f32, x:*const f32, y:*mut f32);
    fn copy(n: usize, x:*const f32, y:*mut f32);
    fn matvec(m: usize, n: usize, a:*const f32, x:*const f32, y:*mut f32);
}

struct CpuOps;
impl DeviceOps for CpuOps {
    fn dot(n:usize, x:*const f32, y:*const f32)->f32{ let xs=unsafe{std::slice::from_raw_parts(x,n)}; let ys=unsafe{std::slice::from_raw_parts(y,n)}; xs.iter().zip(ys).map(|(a,b)|a*b).sum() }
    fn axpy(n:usize, a:f32, x:*const f32, y:*mut f32){ let xs=unsafe{std::slice::from_raw_parts(x,n)}; let ys=unsafe{std::slice::from_raw_parts_mut(y,n)}; for i in 0..n{ ys[i]+=a*xs[i]; } }
    fn copy(n:usize, x:*const f32, y:*mut f32){ let xs=unsafe{std::slice::from_raw_parts(x,n)}; let ys=unsafe{std::slice::from_raw_parts_mut(y,n)}; ys.copy_from_slice(xs); }
    fn matvec(m:usize, n:usize, a:*const f32, x:*const f32, y:*mut f32){ let as_ = unsafe{std::slice::from_raw_parts(a,m*n)}; let xs=unsafe{std::slice::from_raw_parts(x,n)}; let ys=unsafe{std::slice::from_raw_parts_mut(y,m)}; for i in 0..m{ let mut acc=0.0; for j in 0..n{ acc += as_[i*n+j]*xs[j]; } ys[i]=acc; } }
}

static OPS: OnceCell<&'static dyn DeviceOps> = OnceCell::new();

pub fn install_cuda_fused_ops(ops: &'static dyn DeviceOps){ let _=OPS.set(ops); }
pub fn install_hip_fused_ops(ops: &'static dyn DeviceOps){ let _=OPS.set(ops); }
pub fn get_ops() -> &'static dyn DeviceOps { OPS.get().copied().unwrap_or(&CpuOps) }
