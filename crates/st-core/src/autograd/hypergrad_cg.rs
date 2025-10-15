// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! CG path for implicit hypergrad: solve (I - J) v = g with Jvp approximation.

use crate::tensor::Tensor;

fn approx_jvp<F>(step_fn: F, w0:&Tensor, hyper:&Tensor, batch:&Tensor, x:&Tensor) -> Tensor
where F: Fn(&Tensor, &Tensor, &Tensor) -> Tensor {
    // J * x ≈ (step(w0 + eps*x) - step(w0)) / eps
    let eps = 1e-3;
    let w_eps = w0.add(&x.mul_scalar(eps));
    let s_eps = step_fn(&w_eps, hyper, batch);
    let s_ref = step_fn(w0, hyper, batch);
    s_eps.sub(&s_ref).mul_scalar(1.0/eps)
}

/// Conjugate Gradient on A v = g where A x = x - J x (J approximated via step_fn).
pub fn implicit_cg<F,G,It>(
    step_fn: F, w0: Tensor, hyper: Tensor, mut data_loader: It, val_loss_fn: G,
    iters: usize, tol: f32
) -> Tensor
where
    F: Fn(&Tensor, &Tensor, &Tensor) -> Tensor + Copy,
    G: Fn(&Tensor) -> Tensor,
    It: Iterator<Item=Tensor>,
{
    let batch = data_loader.next().expect("need one batch");
    // g = ∂ val_loss / ∂ w |_{w0}
    let val = val_loss_fn(&w0);
    val.backward();
    let g = w0.grad().expect("grad w0");

    // define Ax = x - Jx
    let apply_A = |x:&Tensor| -> Tensor {
        let jx = approx_jvp(step_fn, &w0, &hyper, &batch, x);
        x.sub(&jx)
    };

    // CG init
    let mut v = g.zeros_like();      // initial guess 0
    let mut r = g.clone();           // r0 = g - A v0 = g
    let mut p = r.clone();
    let mut rs_old = r.dot(&r);      // assume dot() exists; else reduce sum(r*r)

    for _ in 0..iters {
        let Ap = apply_A(&p);
        let alpha = rs_old.item() / p.dot(&Ap).item().max(1e-12);
        v = v.add(&p.mul_scalar(alpha));
        r = r.sub(&Ap.mul_scalar(alpha));
        let rs_new = r.dot(&r);
        if rs_new.item().sqrt() < tol { break; }
        let beta = rs_new.item() / rs_old.item().max(1e-12);
        p = r.add(&p.mul_scalar(beta));
        rs_old = rs_new;
    }
    // dL/dhyper ≈ ∂L/∂hyper + (∂ step / ∂ hyper)^T * v
    let w1 = step_fn(&w0, &hyper, &batch);
    w1.backward_with_grad(&v).ok();
    hyper.grad().expect("hyper grad via CG")
}
