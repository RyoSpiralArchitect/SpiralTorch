//! Hypergrad device CG (control on host; vectors on device)
use ndarray::Array1;

#[derive(Clone)]
pub struct DeviceBuf { pub len: usize }

pub trait DeviceLinearOp: Send + Sync {
    fn len(&self) -> usize;
    fn apply(&self, x:&DeviceBuf, y:&DeviceBuf) -> Result<(), String>;
}

pub trait DeviceOps: Send + Sync {
    fn dot(&self, n: usize, x:&DeviceBuf, y:&DeviceBuf) -> Result<f32, String>;
    fn axpy(&self, n: usize, alpha:f32, x:&DeviceBuf, y:&DeviceBuf, out:&DeviceBuf) -> Result<(), String>;
    fn copy(&self, n: usize, src:&DeviceBuf, dst:&DeviceBuf) -> Result<(), String>;
}

pub struct DeviceCgCfg { pub tol:f32, pub max_iter:usize }
impl Default for DeviceCgCfg { fn default()->Self{ Self{ tol:1e-5, max_iter:256 } } }

pub fn cg_solve_device<L:DeviceLinearOp, O:DeviceOps>(
    lin:&L, ops:&O, b:&DeviceBuf, x:&DeviceBuf, r:&DeviceBuf, p:&DeviceBuf, ap:&DeviceBuf, cfg:&DeviceCgCfg
) -> Result<(), String> {
    let n = lin.len();
    lin.apply(x, ap)?;
    ops.axpy(n, -1.0, ap, b, r)?;
    ops.copy(n, r, p)?;

    let mut rs_old = ops.dot(n, r, r)?;
    let tol2 = cfg.tol * cfg.tol;
    let mut k = 0usize;
    while k < cfg.max_iter && rs_old > tol2 {
        lin.apply(p, ap)?;
        let pAp = ops.dot(n, p, ap)?;
        let alpha = if pAp != 0.0 { rs_old / pAp.max(1e-30) } else { 0.0 };
        ops.axpy(n,  alpha, p, x, x)?;
        ops.axpy(n, -alpha, ap, r, r)?;
        let rs_new = ops.dot(n, r, r)?;
        if rs_new <= tol2 { break; }
        let beta = if rs_old != 0.0 { rs_new / rs_old } else { 0.0 };
        ops.copy(n, r, p)?;
        ops.axpy(n, beta, p, p, p)?;
        rs_old = rs_new;
        k += 1;
    }
    Ok(())
}
