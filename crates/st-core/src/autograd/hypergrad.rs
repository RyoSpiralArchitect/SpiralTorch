// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Ameba Autograd — Hypergradient utilities with Neumann and CG solvers.
use crate::tensor::Tensor;
use super::super::hypergrad::config::Solver;

pub struct UnrolledOut { pub w_final: Tensor, pub dval_dhyper: Tensor }
pub struct ImplicitOut { pub dval_dhyper: Tensor }

pub fn unrolled<F, G, It>(mut step_fn: F, mut w0: Tensor, hyper: Tensor, mut data_loader: It, steps: usize,
                          val_loss_fn: G) -> UnrolledOut
where
    F: FnMut(&Tensor, &Tensor, &Tensor) -> Tensor,
    G: Fn(&Tensor) -> Tensor,
    It: Iterator<Item=Tensor>,
{
    let mut w = w0.clone();
    for _ in 0..steps {
        if let Some(batch) = data_loader.next() { w = step_fn(&w, &hyper, &batch); }
    }
    let val = val_loss_fn(&w);
    val.backward();
    UnrolledOut{ w_final: w, dval_dhyper: hyper.grad().expect("hyper gradient missing") }
}

/// Implicit hypergrad using either Neumann or CG. `iters` controls truncation/iterations.
pub fn implicit<F, G, It>(
    step_fn: F, w0: Tensor, hyper: Tensor, mut data_loader: It, val_loss_fn: G,
    solve: &str, iters: usize
) -> ImplicitOut
where
    F: Fn(&Tensor, &Tensor, &Tensor) -> Tensor + Copy,
    G: Fn(&Tensor) -> Tensor,
    It: Iterator<Item=Tensor>,
{
    let solver = match solve {
        s if s.eq_ignore_ascii_case("cg") => Solver::Cg,
        _ => Solver::Neumann,
    };

    // one batch for Jacobian linearization
    let batch = data_loader.next().expect("need one batch for Jacobian estimate");
    let w1 = step_fn(&w0, &hyper, &batch);

    // g = ∂ val_loss / ∂ w |_{w0}
    let val = val_loss_fn(&w0);
    val.backward();
    let g = w0.grad().expect("grad w0");

    match solver {
        Solver::Neumann => implicit_neumann(step_fn, w0, hyper, batch, w1, g, iters),
        Solver::Cg      => implicit_cg(step_fn, w0, hyper, batch, w1, g, iters),
    }
}

fn jvp_approx<F>(step_fn:F, w0:&Tensor, hyper:&Tensor, batch:&Tensor, base:&Tensor, x:&Tensor) -> Tensor
where F: Fn(&Tensor, &Tensor, &Tensor)->Tensor {
    let eps = 1e-3;
    let w_eps = w0.add(&x.mul_scalar(eps));
    let s_eps = step_fn(&w_eps, hyper, batch);
    s_eps.sub(base).mul_scalar(1.0/eps)
}

fn implicit_neumann<F>(step_fn:F, w0:Tensor, hyper:Tensor, batch:Tensor, base:Tensor, g:Tensor, iters:usize) -> ImplicitOut
where F: Fn(&Tensor, &Tensor, &Tensor)->Tensor + Copy {
    let mut v = g.clone();
    let mut jt = g.clone();
    for _ in 0..iters {
        jt = jvp_approx(step_fn, &w0, &hyper, &batch, &base, &jt);
        v  = v.add(&jt);
    }
    // seed backward on base with v to pick up ∂ step / ∂ hyper contribution
    base.backward_with_grad(&v).ok();
    let dh = hyper.grad().unwrap_or_else(|| Tensor::zeros_like(&hyper));
    ImplicitOut{ dval_dhyper: dh }
}

fn implicit_cg<F>(step_fn:F, w0:Tensor, hyper:Tensor, batch:Tensor, base:Tensor, g:Tensor, iters:usize) -> ImplicitOut
where F: Fn(&Tensor, &Tensor, &Tensor)->Tensor + Copy {
    // Solve (I - J)^T v = g  via CG on normal equations A = (I - J)(I - J)^T (symmetric PSD approx)
    // We approximate Av with: A x ≈ x - J x - J^T x + J J^T x.
    // Approximate J x with jvp; approximate J^T x by recomputing grad via backward seed trick.
    let mut v = Tensor::zeros_like(&g);
    let mut r = g.clone();           // r = b - A v ; start v=0 => r=b
    let mut p = r.clone();
    let mut rr_old = r.dot(&r).item_f32();

    for _ in 0..iters {
        // A p ≈ p - J p - J^T p + J J^T p
        let jp  = jvp_approx(step_fn, &w0, &hyper, &batch, &base, &p);
        // J^T p via backward seed: seed base with jp, read grad at base
        base.zero_grad();
        base.backward_with_grad(&jp).ok();
        let jtp = base.grad().expect("need base grad");
        // J J^T p via J applied to (J^T p)
        let j_jtp = jvp_approx(step_fn, &w0, &hyper, &batch, &base, &jtp);
        let ap = p.sub(&jp).sub(&jtp).add(&j_jtp);

        let alpha = rr_old / p.dot(&ap).item_f32();
        v = v.add(&p.mul_scalar(alpha));
        r = r.sub(&ap.mul_scalar(alpha));
        let rr_new = r.dot(&r).item_f32();
        if rr_new.sqrt() < 1e-6 { break; }
        let beta = rr_new / rr_old;
        p = r.add(&p.mul_scalar(beta));
        rr_old = rr_new;
    }

    // pick up ∂ step / ∂ hyper contribution via backward seed on base with v
    base.zero_grad();
    base.backward_with_grad(&v).ok();
    let dh = hyper.grad().unwrap_or_else(|| Tensor::zeros_like(&hyper));
    ImplicitOut{ dval_dhyper: dh }
}
