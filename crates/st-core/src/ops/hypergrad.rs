// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Ameba Hypergrad utilities (ops layer)
//! - Conjugate Gradient solver (matrix-free via `matvec` closure)
//! - Unrolled T-step hypergradient for scalar lambda (e.g., LR)
//! - Implicit differentiation for LR using HVP closure
//!
//! All functions are backend-agnostic; wire into your autograd types as needed.
use ndarray::{Array1, ArrayView1};

pub struct CgCfg { pub tol: f32, pub max_iter: usize }
impl Default for CgCfg { fn default()->Self{ Self{ tol:1e-5, max_iter:256 } } }

/// Solve A x = b by CG with matrix-free matvec.
pub fn cg_solve<F>(matvec: F, b:&Array1<f32>, cfg:&CgCfg) -> Array1<f32>
where F: Fn(&Array1<f32>) -> Array1<f32>
{
    let mut x = Array1::<f32>::zeros(b.len());
    let mut r = b.clone() - matvec(&x);
    let mut p = r.clone();
    let mut rs_old = r.dot(&r);
    for _ in 0..cfg.max_iter {
        let Ap = matvec(&p);
        let denom = p.dot(&Ap).max(1e-20);
        let alpha = rs_old / denom;
        x = &x + &(p.mapv(|v| alpha*v));
        r = &r - &(Ap.mapv(|v| alpha*v));
        let rs_new = r.dot(&r);
        if rs_new.sqrt() < cfg.tol { break; }
        let beta = rs_new / rs_old.max(1e-20);
        p = &r + &(p.mapv(|v| beta*v));
        rs_old = rs_new;
    }
    x
}

/// Unrolled T-step SGD for scalar lambda (e.g., learning rate eta).
/// g_train: ∇_w L_train(w)
/// g_val  : ∇_w L_val(w)
pub fn hypergrad_unrolled<Fg>(w0:&Array1<f32>, eta:f32, steps:usize, g_train:&Fg, g_val:&Fg) -> f32
where Fg: Fn(&Array1<f32>)->Array1<f32>
{
    let mut w = w0.clone();
    // forward unroll
    let mut gs: Vec<Array1<f32>> = Vec::with_capacity(steps);
    for _ in 0..steps {
        let gt = g_train(&w);
        gs.push(gt.clone());
        w = &w - &(gt.mapv(|v| eta*v));
    }
    // reverse accumulation (Pearlmutter-less approximation)
    let mut v = g_val(&w); // dL_val/dw_T
    let mut d_eta = 0.0f32;
    for t in (0..steps).rev() {
        // ∂w_{t+1}/∂η = -∇L_train(w_t)
        d_eta -= v.dot(&gs[t]);
        // Backprop through w_{t} = w_{t+1} + η ∇L_train(w_t)
        // Approximate J^T by identity (omit Hessian-JVP for speed)
        v = v.clone();
    }
    d_eta
}

/// Implicit hypergradient for LR using HVP closure (more accurate).
/// Solve (I - η H)^T v = g_val, then dL/dη = - v^T g_train.
pub fn hypergrad_implicit_lr<Fg,Fh>(w:&Array1<f32>, eta:f32, g_train:&Fg, hvp:&Fh, g_val:&Fg, cfg:&CgCfg) -> f32
where Fg: Fn(&Array1<f32>)->Array1<f32>, Fh: Fn(&Array1<f32>, &Array1<f32>)->Array1<f32>
{
    let gv = g_val(w);
    let matvec = |x:&Array1<f32>| -> Array1<f32> {
        // (I - η H)^T x  ≈  x - η * H x   (H symmetric)
        x - &(hvp(w, x).mapv(|v| eta*v))
    };
    let v = cg_solve(matvec, &gv, cfg);
    let gt = g_train(w);
    - v.dot(&gt)
}
