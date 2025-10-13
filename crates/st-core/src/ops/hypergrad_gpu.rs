//! Device Linear Operator bridge for CG / implicit hypergrad (v1.7.9)
//! Provide on-device matvec y := A(x). Works with WGPU/CUDA/HIP via trait impls.
//
//! Example (WGPU):
//! struct MyOp { /* hold bind groups & pipeline */ }
//! impl DeviceLinearOp for MyOp {
//!   fn len(&self)->usize { self.n }
//!   fn apply(&self, x_dev:&DeviceBuf, y_dev:&DeviceBuf)->Result<(),String> {
//!     // record commands to compute y := A x
//!     Ok(())
//!   }
//! }
use ndarray::Array1;

#[derive(Clone)]
pub struct DeviceBuf { pub len: usize /* + backend handle in real impl */ }

pub trait DeviceLinearOp: Send + Sync {
    fn len(&self) -> usize;
    fn apply(&self, x:&DeviceBuf, y:&DeviceBuf) -> Result<(), String>;
}

pub struct DeviceCgCfg { pub tol:f32, pub max_iter:usize }
impl Default for DeviceCgCfg { fn default()->Self{ Self{ tol:1e-5, max_iter:256 } } }

/// x0, r, p, Ap are device buffers managed by caller allocator.
pub fn cg_solve_device<L:DeviceLinearOp>(
    lin:&L, b:&DeviceBuf, x:&DeviceBuf, r:&DeviceBuf, p:&DeviceBuf, ap:&DeviceBuf, cfg:&DeviceCgCfg
) -> Result<(), String> {
    // Skeleton: callers should provide fused kernels for dot/axpy to keep data on device.
    // Here we assume those fused ops are embedded in `lin.apply` or in optimizer side.
    // This function defines only the *control* loop shape.
    let _n = lin.len();
    // Pseudocode:
    // r := b - A x
    lin.apply(x, ap)?; /* ap = A x */
    // r = b - ap   (fused axpy kernel outside this skeleton)
    // p = r
    // rs_old = dot(r,r)
    // for i in 0..max_iter { alpha = rs_old / dot(p, A p); x += alpha p; r -= alpha Ap; check tol; beta = rs_new/rs_old; p = r + beta p; }
    Ok(())
}

/// Implicit LR hypergrad on device: solve (I - η H)^T v = g_val, then - v^T g_train
pub fn hypergrad_implicit_lr_device<L:DeviceLinearOp>(
    lin:&L, eta:f32, gval:&DeviceBuf, v:&DeviceBuf, tmp:&DeviceBuf, cfg:&DeviceCgCfg
) -> Result<(), String> {
    // Implement as CG over M x := (I - η H)^T x. User provides HVP inside L.apply via closure specialization.
    // Solve M v = g_val, output in `v`. Then compute - v^T g_train with a separate op.
    let _ = (lin, eta, gval, v, tmp, cfg);
    Ok(())
}
