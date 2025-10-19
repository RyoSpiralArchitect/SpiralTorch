// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Ameba Autograd — Hypergradient utilities with Neumann and CG solvers.
use std::error::Error;
use std::fmt;

use crate::tensor::Tensor;
use super::super::hypergrad::config::Solver;

/// Output of the unrolled hypergradient computation.
pub struct UnrolledOut {
    pub w_final: Tensor,
    pub dval_dhyper: Tensor,
}

/// How to approximate Jacobian-vector products with finite differences.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FiniteDiffMode {
    Forward,
    Central,
}

impl Default for FiniteDiffMode {
    fn default() -> Self { Self::Forward }
}

/// Diagnostics for implicit solvers (e.g. CG residuals).
#[derive(Clone, Debug)]
pub struct ImplicitDiagnostics {
    pub solver: Solver,
    pub iterations: usize,
    pub residual: f32,
    pub residual_history: Vec<f32>,
    pub finite_diff_mode: FiniteDiffMode,
}

/// Output of the implicit hypergradient computation.
pub struct ImplicitOut {
    pub dval_dhyper: Tensor,
    pub diagnostics: Option<ImplicitDiagnostics>,
}

/// Controls the behaviour of the implicit hypergradient routines.
#[derive(Clone, Debug)]
pub struct ImplicitOptions {
    pub solver: Solver,
    pub max_iters: usize,
    pub tolerance: f32,
    pub finite_diff_eps: f32,
    pub finite_diff_mode: FiniteDiffMode,
    pub zero_existing_grads: bool,
}

impl Default for ImplicitOptions {
    fn default() -> Self {
        Self {
            solver: Solver::Neumann,
            max_iters: 8,
            tolerance: 1e-6,
            finite_diff_eps: 1e-3,
            finite_diff_mode: FiniteDiffMode::Forward,
            zero_existing_grads: true,
        }
    }
}

impl ImplicitOptions {
    pub fn with_solver(mut self, solver: Solver) -> Self { self.solver = solver; self }
    pub fn with_max_iters(mut self, iters: usize) -> Self { self.max_iters = iters; self }
    pub fn with_tolerance(mut self, tol: f32) -> Self { self.tolerance = tol; self }
    pub fn with_finite_diff_eps(mut self, eps: f32) -> Self { self.finite_diff_eps = eps; self }
    pub fn with_finite_diff_mode(mut self, mode: FiniteDiffMode) -> Self {
        self.finite_diff_mode = mode;
        self
    }
    pub fn keep_existing_grads(mut self) -> Self { self.zero_existing_grads = false; self }
}

/// Errors that can occur during implicit hypergradient computation.
#[derive(Debug)]
pub enum HypergradError {
    MissingBatch,
    MissingGrad(&'static str),
    SolverBreakdown,
    InvalidFiniteDiffStep,
}

impl fmt::Display for HypergradError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HypergradError::MissingBatch => write!(f, "no batch available to estimate Jacobian"),
            HypergradError::MissingGrad(name) => write!(f, "missing gradient for {name}"),
            HypergradError::SolverBreakdown => write!(f, "implicit solver broke down"),
            HypergradError::InvalidFiniteDiffStep => write!(f, "finite difference epsilon must be non-zero"),
        }
    }
}

impl Error for HypergradError {}

pub fn unrolled<F, G, It>(
    mut step_fn: F,
    mut w0: Tensor,
    hyper: Tensor,
    mut data_loader: It,
    steps: usize,
    val_loss_fn: G,
) -> UnrolledOut
where
    F: FnMut(&Tensor, &Tensor, &Tensor) -> Tensor,
    G: Fn(&Tensor) -> Tensor,
    It: Iterator<Item=Tensor>,
{
    hyper.zero_grad();
    w0.zero_grad();
    let mut w = w0.clone();
    for _ in 0..steps {
        if let Some(batch) = data_loader.next() { w = step_fn(&w, &hyper, &batch); }
    }
    let val = val_loss_fn(&w);
    val.backward();
    let dh = hyper.grad().unwrap_or_else(|| Tensor::zeros_like(&hyper));
    UnrolledOut{ w_final: w, dval_dhyper: dh }
}

/// Implicit hypergrad using either Neumann or CG. `iters` controls truncation/iterations.
pub fn implicit<F, G, It>(
    step_fn: F, w0: Tensor, hyper: Tensor, mut data_loader: It, val_loss_fn: G,
    solve: &str, iters: usize
) -> Result<ImplicitOut, HypergradError>
where
    F: Fn(&Tensor, &Tensor, &Tensor) -> Tensor + Copy,
    G: Fn(&Tensor) -> Tensor,
    It: Iterator<Item=Tensor>,
{
    let solver = match solve {
        s if s.eq_ignore_ascii_case("cg") => Solver::Cg,
        _ => Solver::Neumann,
    };

    implicit_with_options(
        step_fn,
        w0,
        hyper,
        data_loader,
        val_loss_fn,
        ImplicitOptions::default().with_solver(solver).with_max_iters(iters),
    )
}

/// Implicit hypergrad with a fully configurable set of options.
pub fn implicit_with_options<F, G, It>(
    step_fn: F,
    mut w0: Tensor,
    mut hyper: Tensor,
    mut data_loader: It,
    val_loss_fn: G,
    opts: ImplicitOptions,
) -> Result<ImplicitOut, HypergradError>
where
    F: Fn(&Tensor, &Tensor, &Tensor) -> Tensor + Copy,
    G: Fn(&Tensor) -> Tensor,
    It: Iterator<Item=Tensor>,
{
    if opts.zero_existing_grads {
        w0.zero_grad();
        hyper.zero_grad();
    }

    if opts.finite_diff_eps.abs() < f32::EPSILON {
        return Err(HypergradError::InvalidFiniteDiffStep);
    }

    // one batch for Jacobian linearization
    let batch = data_loader.next().ok_or(HypergradError::MissingBatch)?;
    let w1 = step_fn(&w0, &hyper, &batch);

    // g = ∂ val_loss / ∂ w |_{w0}
    let val = val_loss_fn(&w0);
    val.backward();
    let g = w0.grad().ok_or(HypergradError::MissingGrad("w0"))?;

    match opts.solver {
        Solver::Neumann => implicit_neumann(step_fn, w0, hyper, batch, w1, g, &opts),
        Solver::Cg      => implicit_cg(step_fn, w0, hyper, batch, w1, g, &opts),
    }
}

fn jvp_approx<F>(
    step_fn: F,
    w0: &Tensor,
    hyper: &Tensor,
    batch: &Tensor,
    base: &Tensor,
    x: &Tensor,
    eps: f32,
    mode: FiniteDiffMode,
) -> Tensor
where
    F: Fn(&Tensor, &Tensor, &Tensor) -> Tensor,
{
    match mode {
        FiniteDiffMode::Forward => {
            let w_eps = w0.add(&x.mul_scalar(eps));
            let s_eps = step_fn(&w_eps, hyper, batch);
            s_eps.sub(base).mul_scalar(1.0 / eps)
        }
        FiniteDiffMode::Central => {
            let offset = x.mul_scalar(eps);
            let w_plus = w0.add(&offset);
            let w_minus = w0.sub(&offset);
            let s_plus = step_fn(&w_plus, hyper, batch);
            let s_minus = step_fn(&w_minus, hyper, batch);
            s_plus.sub(&s_minus).mul_scalar(0.5 / eps)
        }
    }
}

fn implicit_neumann<F>(
    step_fn: F,
    w0: Tensor,
    mut hyper: Tensor,
    batch: Tensor,
    base: Tensor,
    g: Tensor,
    opts: &ImplicitOptions,
) -> Result<ImplicitOut, HypergradError>
where
    F: Fn(&Tensor, &Tensor, &Tensor) -> Tensor + Copy,
{
    let mut v = g.clone();
    let mut jt = g.clone();
    let mut iterations = 0usize;
    let mut residual = jt.dot(&jt).item_f32().sqrt();
    let mut residual_history = Vec::with_capacity(opts.max_iters + 1);
    residual_history.push(residual);

    for _ in 0..opts.max_iters {
        iterations += 1;
        jt = jvp_approx(
            step_fn,
            &w0,
            &hyper,
            &batch,
            &base,
            &jt,
            opts.finite_diff_eps,
            opts.finite_diff_mode,
        );
        residual = jt.dot(&jt).item_f32().sqrt();
        residual_history.push(residual);
        v = v.add(&jt);
        if residual < opts.tolerance {
            break;
        }
    }
    // seed backward on base with v to pick up ∂ step / ∂ hyper contribution
    base.zero_grad();
    base.backward_with_grad(&v).ok();
    let dh = hyper.grad().ok_or(HypergradError::MissingGrad("hyper"))?;
    Ok(ImplicitOut {
        dval_dhyper: dh,
        diagnostics: Some(ImplicitDiagnostics {
            solver: opts.solver,
            iterations,
            residual,
            residual_history,
            finite_diff_mode: opts.finite_diff_mode,
        }),
    })
}

fn implicit_cg<F>(
    step_fn: F,
    w0: Tensor,
    mut hyper: Tensor,
    batch: Tensor,
    base: Tensor,
    g: Tensor,
    opts: &ImplicitOptions,
) -> Result<ImplicitOut, HypergradError>
where
    F: Fn(&Tensor, &Tensor, &Tensor) -> Tensor + Copy,
{
    // Solve (I - J)^T v = g  via CG on normal equations A = (I - J)(I - J)^T (symmetric PSD approx)
    // We approximate Av with: A x ≈ x - J x - J^T x + J J^T x.
    // Approximate J x with jvp; approximate J^T x by recomputing grad via backward seed trick.
    let mut v = Tensor::zeros_like(&g);
    let mut r = g.clone();           // r = b - A v ; start v=0 => r=b
    let mut p = r.clone();
    let mut rr_old = r.dot(&r).item_f32();
    let mut residual = rr_old.sqrt();
    let mut residual_history = Vec::with_capacity(opts.max_iters + 1);
    residual_history.push(residual);
    let mut iterations = 0usize;

    for _ in 0..opts.max_iters {
        iterations += 1;
        // A p ≈ p - J p - J^T p + J J^T p
        let jp = jvp_approx(
            step_fn,
            &w0,
            &hyper,
            &batch,
            &base,
            &p,
            opts.finite_diff_eps,
            opts.finite_diff_mode,
        );
        // J^T p via backward seed: seed base with jp, read grad at base
        base.zero_grad();
        base.backward_with_grad(&jp).ok();
        let jtp = base.grad().ok_or(HypergradError::MissingGrad("base"))?;
        // J J^T p via J applied to (J^T p)
        let j_jtp = jvp_approx(
            step_fn,
            &w0,
            &hyper,
            &batch,
            &base,
            &jtp,
            opts.finite_diff_eps,
            opts.finite_diff_mode,
        );
        let ap = p.sub(&jp).sub(&jtp).add(&j_jtp);

        let denom = p.dot(&ap).item_f32();
        if denom.abs() < 1e-12 { return Err(HypergradError::SolverBreakdown); }
        let alpha = rr_old / denom;
        v = v.add(&p.mul_scalar(alpha));
        r = r.sub(&ap.mul_scalar(alpha));
        let rr_new = r.dot(&r).item_f32();
        residual = rr_new.sqrt();
        residual_history.push(residual);
        if residual < opts.tolerance { break; }
        let beta = rr_new / rr_old;
        p = r.add(&p.mul_scalar(beta));
        rr_old = rr_new;
    }

    // pick up ∂ step / ∂ hyper contribution via backward seed on base with v
    base.zero_grad();
    base.backward_with_grad(&v).ok();
    let dh = hyper.grad().ok_or(HypergradError::MissingGrad("hyper"))?;
    Ok(ImplicitOut {
        dval_dhyper: dh,
        diagnostics: Some(ImplicitDiagnostics {
            solver: opts.solver,
            iterations,
            residual,
            residual_history,
            finite_diff_mode: opts.finite_diff_mode,
        }),
    })
}
