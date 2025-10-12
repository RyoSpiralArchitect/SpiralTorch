//! Ameba Autograd — Hypergradient utilities.
//! - `unrolled`: differentiate through `T` update steps (truncated BPTT)
//! - `implicit`: approximate implicit hypergrad via (I - J)^(-1) vector solve (Neumann / CG)

use crate::tensor::Tensor;

pub struct UnrolledOut { pub w_final: Tensor, pub dval_dhyper: Tensor }
pub struct ImplicitOut { pub dval_dhyper: Tensor }

/// Unrolled hypergradient: differentiate `val_loss_fn(w_T)` wrt `hyper` through T steps.
/// `step_fn(w, hyper, batch) -> w_next` must be differentiable in our graph.
pub fn unrolled<F, G, It>(mut step_fn: F, mut w0: Tensor, hyper: Tensor, mut data_loader: It, steps: usize,
                          val_loss_fn: G) -> UnrolledOut
where
    F: FnMut(&Tensor, &Tensor, &Tensor) -> Tensor,
    G: Fn(&Tensor) -> Tensor,
    It: Iterator<Item=Tensor>,
{
    let mut w = w0.clone();
    for _ in 0..steps {
        if let Some(batch) = data_loader.next() {
            w = step_fn(&w, &hyper, &batch);
        }
    }
    let val = val_loss_fn(&w);
    val.backward(); // seeds d val / d (w, hyper)
    // hyper.grad() is populated by engine; clone for return (or keep ref)
    UnrolledOut{ w_final: w, dval_dhyper: hyper.grad().expect("hyper gradient missing") }
}

/// Implicit hypergradient using Neumann series truncation (approx inverse).
/// Solve v ≈ (I - J)^(-1) * g where J = ∂update/∂w at fixed point; then dL/dhyper ≈ ∂L/∂w · v + ∂L/∂hyper.
pub fn implicit<F, G, It>(
    step_fn: F, w0: Tensor, hyper: Tensor, mut data_loader: It, val_loss_fn: G,
    solve: &str, iters: usize
) -> ImplicitOut
where
    F: Fn(&Tensor, &Tensor, &Tensor) -> Tensor + Copy,
    G: Fn(&Tensor) -> Tensor,
    It: Iterator<Item=Tensor>,
{
    // 1) Build one step around current w0 to get Jvp closure
    let batch = data_loader.next().expect("need one batch for Jacobian estimate");
    let w1 = step_fn(&w0, &hyper, &batch);
    // g = ∂ val_loss / ∂ w |_{w0}
    let val = val_loss_fn(&w0);
    val.backward();
    let mut g = w0.grad().expect("grad w0");

    // v = sum_{t=0..T} J^t g (Neumann)
    let mut v = g.clone();
    let mut jt_g = g.clone();
    for _ in 0..iters {
        // approximate Jvp by one-step linearization: J * x ≈ ∂ step/∂w @ x (vec-Jac product)
        // We simulate via forward-over-reverse trick with a small epsilon.
        let eps = 1e-3;
        let w_eps = w0.add(&jt_g.mul_scalar(eps));
        let w_step = step_fn(&w_eps, &hyper, &batch);
        let jvp_approx = w_step.sub(&w1).mul_scalar(1.0/eps);
        jt_g = jvp_approx;
        v = v.add(&jt_g);
    }

    // dL/dhyper ≈ ∂L/∂hyper + (∂ step / ∂ hyper)^T * v
    // compute via one extra pass:
    let mut hyper_bar = hyper.zeros_like(); // accumulator
    {
        // seed adjoint on output w: v
        w1.backward_with_grad(&v).ok();
        if let Some(hg) = hyper.grad() {
            hyper_bar = hyper_bar.add(&hg);
        }
    }
    ImplicitOut{ dval_dhyper: hyper_bar }
}
