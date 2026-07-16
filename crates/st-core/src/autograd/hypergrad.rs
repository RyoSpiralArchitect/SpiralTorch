// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2025 Ryo SpiralArchitect

//! Unrolled and implicit hypergradients over the `st-tensor` reverse-mode core.

use std::error::Error;
use std::fmt;

use st_tensor::{
    AutogradTensor, Tensor, TensorError, AUTOGRAD_CONTRACT_VERSION, AUTOGRAD_SEMANTIC_OWNER,
};

/// Contract version for higher-order differentiation owned by `st-core`.
pub const HYPERGRAD_CONTRACT_VERSION: &str = "spiraltorch.hypergrad.v1";
/// Module that owns higher-order solver equations and convergence diagnostics.
pub const HYPERGRAD_SEMANTIC_OWNER: &str = "st-core::autograd::hypergrad";

/// Linear solver used for the fixed-point sensitivity equation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Solver {
    Neumann,
    Cg,
}

/// How Jacobian-vector products are approximated numerically.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FiniteDiffMode {
    #[default]
    Forward,
    Central,
}

/// Diagnostics for one implicit hypergradient solve.
#[derive(Clone, Debug, PartialEq)]
pub struct ImplicitDiagnostics {
    pub contract_version: &'static str,
    pub semantic_owner: &'static str,
    pub autograd_contract_version: &'static str,
    pub autograd_semantic_owner: &'static str,
    pub requested_solver: Solver,
    pub effective_solver: Solver,
    pub iterations: usize,
    pub residual: f32,
    pub residual_history: Vec<f32>,
    pub finite_diff_mode: FiniteDiffMode,
    pub finite_diff_step: f32,
    pub converged: bool,
    pub fallback_reason: Option<&'static str>,
}

/// Result of differentiating through an explicitly unrolled optimization path.
#[derive(Clone, Debug)]
pub struct UnrolledOut {
    pub w_final: AutogradTensor,
    pub dval_dhyper: Tensor,
}

/// Result of an implicit fixed-point hypergradient solve.
#[derive(Clone, Debug)]
pub struct ImplicitOut {
    pub dval_dhyper: Tensor,
    pub diagnostics: ImplicitDiagnostics,
}

/// Controls fixed-point sensitivity estimation.
#[derive(Clone, Debug, PartialEq)]
pub struct ImplicitOptions {
    pub solver: Solver,
    pub max_iters: usize,
    pub tolerance: f32,
    pub finite_diff_eps: f32,
    pub finite_diff_mode: FiniteDiffMode,
    pub cg_fallback_to_neumann: bool,
}

#[derive(Clone, Copy)]
struct SolverRoute {
    requested: Solver,
    fallback_reason: Option<&'static str>,
}

impl SolverRoute {
    const fn direct(requested: Solver) -> Self {
        Self {
            requested,
            fallback_reason: None,
        }
    }

    const fn fallback(requested: Solver, reason: &'static str) -> Self {
        Self {
            requested,
            fallback_reason: Some(reason),
        }
    }
}

impl Default for ImplicitOptions {
    fn default() -> Self {
        Self {
            solver: Solver::Neumann,
            max_iters: 32,
            tolerance: 1e-6,
            finite_diff_eps: 1e-3,
            finite_diff_mode: FiniteDiffMode::Central,
            cg_fallback_to_neumann: true,
        }
    }
}

impl ImplicitOptions {
    pub fn with_solver(mut self, solver: Solver) -> Self {
        self.solver = solver;
        self
    }

    pub fn with_max_iters(mut self, iterations: usize) -> Self {
        self.max_iters = iterations;
        self
    }

    pub fn with_tolerance(mut self, tolerance: f32) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub fn with_finite_diff_eps(mut self, epsilon: f32) -> Self {
        self.finite_diff_eps = epsilon;
        self
    }

    pub fn with_finite_diff_mode(mut self, mode: FiniteDiffMode) -> Self {
        self.finite_diff_mode = mode;
        self
    }

    pub fn without_cg_fallback(mut self) -> Self {
        self.cg_fallback_to_neumann = false;
        self
    }

    fn validate(&self) -> HypergradResult<()> {
        if self.max_iters == 0 {
            return Err(HypergradError::InvalidIterations {
                label: "implicit_max_iters",
                iterations: self.max_iters,
            });
        }
        if !self.tolerance.is_finite() || self.tolerance <= 0.0 {
            return Err(HypergradError::InvalidTolerance(self.tolerance));
        }
        if !self.finite_diff_eps.is_finite() || self.finite_diff_eps <= 0.0 {
            return Err(HypergradError::InvalidFiniteDiffStep(self.finite_diff_eps));
        }
        Ok(())
    }
}

/// Failures from higher-order differentiation.
#[derive(Clone, Debug, PartialEq)]
pub enum HypergradError {
    MissingBatch {
        step: usize,
    },
    NonTrainableInput(&'static str),
    InvalidIterations {
        label: &'static str,
        iterations: usize,
    },
    InvalidTolerance(f32),
    InvalidFiniteDiffStep(f32),
    InvalidSolver(String),
    StepShapeMismatch {
        expected: (usize, usize),
        got: (usize, usize),
    },
    SolverBreakdown {
        iteration: usize,
        denominator: f64,
    },
    Tensor(TensorError),
}

impl fmt::Display for HypergradError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingBatch { step } => {
                write!(f, "missing training batch for unrolled step {step}")
            }
            Self::NonTrainableInput(name) => {
                write!(f, "{name} must be an AutogradTensor trainable leaf")
            }
            Self::InvalidIterations { label, iterations } => {
                write!(f, "{label} must be positive, received {iterations}")
            }
            Self::InvalidTolerance(tolerance) => {
                write!(f, "implicit tolerance must be positive and finite, received {tolerance}")
            }
            Self::InvalidFiniteDiffStep(epsilon) => {
                write!(f, "finite-difference step must be positive and finite, received {epsilon}")
            }
            Self::InvalidSolver(solver) => {
                write!(f, "unknown implicit hypergradient solver '{solver}'")
            }
            Self::StepShapeMismatch { expected, got } => {
                write!(f, "optimization step changed parameter shape from {expected:?} to {got:?}")
            }
            Self::SolverBreakdown {
                iteration,
                denominator,
            } => write!(
                f,
                "implicit CG lost a positive direction at iteration {iteration} (p^T A p = {denominator})"
            ),
            Self::Tensor(error) => error.fmt(f),
        }
    }
}

impl Error for HypergradError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Tensor(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TensorError> for HypergradError {
    fn from(error: TensorError) -> Self {
        Self::Tensor(error)
    }
}

pub type HypergradResult<T> = Result<T, HypergradError>;

/// Differentiates validation loss through `steps` explicit optimization steps.
pub fn unrolled<F, G, It>(
    mut step_fn: F,
    w0: AutogradTensor,
    hyper: AutogradTensor,
    mut data_loader: It,
    steps: usize,
    val_loss_fn: G,
) -> HypergradResult<UnrolledOut>
where
    F: FnMut(&AutogradTensor, &AutogradTensor, &AutogradTensor) -> HypergradResult<AutogradTensor>,
    G: Fn(&AutogradTensor) -> HypergradResult<AutogradTensor>,
    It: Iterator<Item = AutogradTensor>,
{
    if steps == 0 {
        return Err(HypergradError::InvalidIterations {
            label: "unrolled_steps",
            iterations: steps,
        });
    }
    ensure_trainable("hyper", &hyper)?;
    let mut weights = w0;
    for step in 0..steps {
        let batch = data_loader
            .next()
            .ok_or(HypergradError::MissingBatch { step })?;
        let next = step_fn(&weights, &hyper, &batch)?;
        ensure_step_shape(weights.shape(), next.shape())?;
        weights = next;
    }

    let validation_loss = val_loss_fn(&weights)?;
    let dval_dhyper = scalar_vjp(&validation_loss, &hyper)?;
    Ok(UnrolledOut {
        w_final: weights,
        dval_dhyper,
    })
}

/// Convenience implicit solve using a named solver.
pub fn implicit<F, G, It>(
    step_fn: F,
    w0: AutogradTensor,
    hyper: AutogradTensor,
    data_loader: It,
    val_loss_fn: G,
    solver: &str,
    iterations: usize,
) -> HypergradResult<ImplicitOut>
where
    F: Fn(&AutogradTensor, &AutogradTensor, &AutogradTensor) -> HypergradResult<AutogradTensor>
        + Copy,
    G: Fn(&AutogradTensor) -> HypergradResult<AutogradTensor>,
    It: Iterator<Item = AutogradTensor>,
{
    let solver = if solver.eq_ignore_ascii_case("cg") {
        Solver::Cg
    } else if solver.eq_ignore_ascii_case("neumann") {
        Solver::Neumann
    } else {
        return Err(HypergradError::InvalidSolver(solver.to_string()));
    };
    implicit_with_options(
        step_fn,
        w0,
        hyper,
        data_loader,
        val_loss_fn,
        ImplicitOptions::default()
            .with_solver(solver)
            .with_max_iters(iterations),
    )
}

/// Solves the fixed-point sensitivity equation with explicit diagnostics.
pub fn implicit_with_options<F, G, It>(
    step_fn: F,
    w0: AutogradTensor,
    hyper: AutogradTensor,
    mut data_loader: It,
    val_loss_fn: G,
    options: ImplicitOptions,
) -> HypergradResult<ImplicitOut>
where
    F: Fn(&AutogradTensor, &AutogradTensor, &AutogradTensor) -> HypergradResult<AutogradTensor>
        + Copy,
    G: Fn(&AutogradTensor) -> HypergradResult<AutogradTensor>,
    It: Iterator<Item = AutogradTensor>,
{
    options.validate()?;
    ensure_trainable("w0", &w0)?;
    ensure_trainable("hyper", &hyper)?;
    let batch = data_loader
        .next()
        .ok_or(HypergradError::MissingBatch { step: 0 })?;
    let base = step_fn(&w0, &hyper, &batch)?;
    ensure_step_shape(w0.shape(), base.shape())?;

    let validation_loss = val_loss_fn(&w0)?;
    let validation_gradient = scalar_vjp(&validation_loss, &w0)?;
    let finite_diff_step = resolve_finite_diff_step(
        options.finite_diff_eps,
        options.finite_diff_mode,
        w0.value(),
        base.value(),
    )?;

    match options.solver {
        Solver::Neumann => solve_neumann(
            &w0,
            &hyper,
            &base,
            validation_gradient,
            &options,
            finite_diff_step,
            SolverRoute::direct(Solver::Neumann),
        ),
        Solver::Cg => solve_cg(
            step_fn,
            &w0,
            &hyper,
            &batch,
            &base,
            validation_gradient,
            &options,
            finite_diff_step,
        ),
    }
}

fn solve_neumann(
    w0: &AutogradTensor,
    hyper: &AutogradTensor,
    base: &AutogradTensor,
    gradient: Tensor,
    options: &ImplicitOptions,
    epsilon: f32,
    route: SolverRoute,
) -> HypergradResult<ImplicitOut> {
    let equation_gradient = gradient.clone();
    let mut sensitivity = gradient.clone();
    let mut transpose_term = gradient;
    let mut residual = stable_l2(&transpose_term);
    let mut residual_history = vec![residual];
    let mut iterations = 0usize;
    let mut converged = residual <= options.tolerance;

    while iterations < options.max_iters && !converged {
        transpose_term = vjp(base, w0, &transpose_term)?;
        sensitivity = sensitivity.add(&transpose_term)?;
        residual = stable_l2(&transpose_term);
        if !residual.is_finite() {
            return Err(HypergradError::Tensor(TensorError::NonFiniteValue {
                label: "implicit_neumann_residual",
                value: residual,
            }));
        }
        residual_history.push(residual);
        iterations += 1;
        converged = residual <= options.tolerance;
    }

    let transpose_sensitivity = vjp(base, w0, &sensitivity)?;
    let equation_residual = sensitivity
        .sub(&transpose_sensitivity)?
        .sub(&equation_gradient)?;
    residual = stable_l2(&equation_residual);
    if !residual.is_finite() {
        return Err(HypergradError::Tensor(TensorError::NonFiniteValue {
            label: "implicit_neumann_equation_residual",
            value: residual,
        }));
    }
    residual_history.push(residual);
    converged = residual <= options.tolerance;
    let dval_dhyper = vjp(base, hyper, &sensitivity)?;
    Ok(ImplicitOut {
        dval_dhyper,
        diagnostics: diagnostics(
            route.requested,
            Solver::Neumann,
            iterations,
            residual,
            residual_history,
            options.finite_diff_mode,
            epsilon,
            converged,
            route.fallback_reason,
        ),
    })
}

#[allow(clippy::too_many_arguments)]
fn solve_cg<F>(
    step_fn: F,
    w0: &AutogradTensor,
    hyper: &AutogradTensor,
    batch: &AutogradTensor,
    base: &AutogradTensor,
    gradient: Tensor,
    options: &ImplicitOptions,
    epsilon: f32,
) -> HypergradResult<ImplicitOut>
where
    F: Fn(&AutogradTensor, &AutogradTensor, &AutogradTensor) -> HypergradResult<AutogradTensor>
        + Copy,
{
    // Solve B B^T v = B g, where B = I - J. This is the normal equation for
    // B^T v = g and keeps CG on a symmetric positive-semidefinite operator.
    let jg = jvp(
        step_fn,
        w0,
        hyper,
        batch,
        base.value(),
        &gradient,
        epsilon,
        options.finite_diff_mode,
    )?;
    let rhs = gradient.sub(&jg)?;
    let mut sensitivity = Tensor::zeros(rhs.shape().0, rhs.shape().1)?;
    let mut residual_vector = rhs;
    let mut direction = residual_vector.clone();
    let mut residual_squared = tensor_dot(&residual_vector, &residual_vector)?;
    let mut residual = residual_squared.sqrt() as f32;
    let mut residual_history = vec![residual];
    let mut iterations = 0usize;
    let mut normal_converged = residual <= options.tolerance;

    while iterations < options.max_iters && !normal_converged {
        let j_direction = jvp(
            step_fn,
            w0,
            hyper,
            batch,
            base.value(),
            &direction,
            epsilon,
            options.finite_diff_mode,
        )?;
        let jt_direction = vjp(base, w0, &direction)?;
        let j_jt_direction = jvp(
            step_fn,
            w0,
            hyper,
            batch,
            base.value(),
            &jt_direction,
            epsilon,
            options.finite_diff_mode,
        )?;
        let applied = direction
            .sub(&j_direction)?
            .sub(&jt_direction)?
            .add(&j_jt_direction)?;
        let denominator = tensor_dot(&direction, &applied)?;
        if !denominator.is_finite() || denominator <= 1e-12 {
            if options.cg_fallback_to_neumann {
                return solve_neumann(
                    w0,
                    hyper,
                    base,
                    gradient,
                    options,
                    epsilon,
                    SolverRoute::fallback(Solver::Cg, "cg_non_positive_direction"),
                );
            }
            return Err(HypergradError::SolverBreakdown {
                iteration: iterations,
                denominator,
            });
        }

        let alpha = residual_squared / denominator;
        let alpha = finite_f32("implicit_cg_alpha", alpha)?;
        sensitivity = sensitivity.add(&direction.scale(alpha)?)?;
        residual_vector = residual_vector.sub(&applied.scale(alpha)?)?;
        let next_residual_squared = tensor_dot(&residual_vector, &residual_vector)?;
        residual = next_residual_squared.sqrt() as f32;
        residual_history.push(residual);
        iterations += 1;
        normal_converged = residual <= options.tolerance;
        if normal_converged {
            break;
        }
        let beta = next_residual_squared / residual_squared;
        let beta = finite_f32("implicit_cg_beta", beta)?;
        direction = residual_vector.add(&direction.scale(beta)?)?;
        residual_squared = next_residual_squared;
    }

    let transpose_sensitivity = vjp(base, w0, &sensitivity)?;
    let equation_residual = sensitivity.sub(&transpose_sensitivity)?.sub(&gradient)?;
    residual = stable_l2(&equation_residual);
    if !residual.is_finite() {
        return Err(HypergradError::Tensor(TensorError::NonFiniteValue {
            label: "implicit_cg_equation_residual",
            value: residual,
        }));
    }
    residual_history.push(residual);
    let converged = residual <= options.tolerance;
    if !converged && options.cg_fallback_to_neumann {
        return solve_neumann(
            w0,
            hyper,
            base,
            gradient,
            options,
            epsilon,
            SolverRoute::fallback(Solver::Cg, "cg_equation_residual"),
        );
    }
    let dval_dhyper = vjp(base, hyper, &sensitivity)?;
    Ok(ImplicitOut {
        dval_dhyper,
        diagnostics: diagnostics(
            Solver::Cg,
            Solver::Cg,
            iterations,
            residual,
            residual_history,
            options.finite_diff_mode,
            epsilon,
            converged,
            None,
        ),
    })
}

#[allow(clippy::too_many_arguments)]
fn jvp<F>(
    step_fn: F,
    w0: &AutogradTensor,
    hyper: &AutogradTensor,
    batch: &AutogradTensor,
    base: &Tensor,
    vector: &Tensor,
    epsilon: f32,
    mode: FiniteDiffMode,
) -> HypergradResult<Tensor>
where
    F: Fn(&AutogradTensor, &AutogradTensor, &AutogradTensor) -> HypergradResult<AutogradTensor>
        + Copy,
{
    if vector.shape() != w0.shape() {
        return Err(HypergradError::Tensor(TensorError::ShapeMismatch {
            left: vector.shape(),
            right: w0.shape(),
        }));
    }
    let hyper = hyper.detach()?;
    let batch = batch.detach()?;
    let offset = vector.scale(epsilon)?;
    match mode {
        FiniteDiffMode::Forward => {
            let perturbed = AutogradTensor::constant(w0.value().add(&offset)?)?;
            let stepped = step_fn(&perturbed, &hyper, &batch)?;
            ensure_step_shape(w0.shape(), stepped.shape())?;
            Ok(stepped.value().sub(base)?.scale(1.0 / epsilon)?)
        }
        FiniteDiffMode::Central => {
            let plus = AutogradTensor::constant(w0.value().add(&offset)?)?;
            let minus = AutogradTensor::constant(w0.value().sub(&offset)?)?;
            let stepped_plus = step_fn(&plus, &hyper, &batch)?;
            let stepped_minus = step_fn(&minus, &hyper, &batch)?;
            ensure_step_shape(w0.shape(), stepped_plus.shape())?;
            ensure_step_shape(w0.shape(), stepped_minus.shape())?;
            Ok(stepped_plus
                .value()
                .sub(stepped_minus.value())?
                .scale(0.5 / epsilon)?)
        }
    }
}

fn vjp(
    output: &AutogradTensor,
    input: &AutogradTensor,
    vector: &Tensor,
) -> HypergradResult<Tensor> {
    output
        .vector_jacobian_product(input, vector)
        .map_err(HypergradError::from)
}

fn scalar_vjp(output: &AutogradTensor, input: &AutogradTensor) -> HypergradResult<Tensor> {
    let (rows, cols) = output.shape();
    if (rows, cols) != (1, 1) {
        return Err(HypergradError::Tensor(TensorError::NonScalarBackward {
            rows,
            cols,
        }));
    }
    vjp(output, input, &Tensor::from_vec(1, 1, vec![1.0])?)
}

fn ensure_trainable(label: &'static str, tensor: &AutogradTensor) -> HypergradResult<()> {
    if tensor.requires_grad() {
        Ok(())
    } else {
        Err(HypergradError::NonTrainableInput(label))
    }
}

fn ensure_step_shape(expected: (usize, usize), got: (usize, usize)) -> HypergradResult<()> {
    if expected == got {
        Ok(())
    } else {
        Err(HypergradError::StepShapeMismatch { expected, got })
    }
}

fn resolve_finite_diff_step(
    initial: f32,
    mode: FiniteDiffMode,
    w0: &Tensor,
    base: &Tensor,
) -> HypergradResult<f32> {
    if !initial.is_finite() || initial <= 0.0 {
        return Err(HypergradError::InvalidFiniteDiffStep(initial));
    }
    let scale = stable_l2(w0).max(stable_l2(base)).max(1.0);
    let roundoff_floor = match mode {
        FiniteDiffMode::Forward => f32::EPSILON.sqrt(),
        FiniteDiffMode::Central => f32::EPSILON.cbrt(),
    };
    let epsilon = initial.abs().max(scale * roundoff_floor);
    if epsilon.is_finite() && epsilon > 0.0 {
        Ok(epsilon)
    } else {
        Err(HypergradError::InvalidFiniteDiffStep(epsilon))
    }
}

fn tensor_dot(lhs: &Tensor, rhs: &Tensor) -> HypergradResult<f64> {
    if lhs.shape() != rhs.shape() {
        return Err(HypergradError::Tensor(TensorError::ShapeMismatch {
            left: lhs.shape(),
            right: rhs.shape(),
        }));
    }
    let mut dot = 0.0f64;
    for (&lhs, &rhs) in lhs.data().iter().zip(rhs.data()) {
        dot += lhs as f64 * rhs as f64;
    }
    if dot.is_finite() {
        Ok(dot)
    } else {
        Err(HypergradError::Tensor(TensorError::NonFiniteValue {
            label: "implicit_tensor_dot",
            value: dot as f32,
        }))
    }
}

fn stable_l2(tensor: &Tensor) -> f32 {
    tensor
        .data()
        .iter()
        .map(|&value| {
            let value = value as f64;
            value * value
        })
        .sum::<f64>()
        .sqrt()
        .min(f32::MAX as f64) as f32
}

fn finite_f32(label: &'static str, value: f64) -> HypergradResult<f32> {
    let narrowed = value as f32;
    if narrowed.is_finite() {
        Ok(narrowed)
    } else {
        Err(HypergradError::Tensor(TensorError::NonFiniteValue {
            label,
            value: narrowed,
        }))
    }
}

#[allow(clippy::too_many_arguments)]
fn diagnostics(
    requested_solver: Solver,
    effective_solver: Solver,
    iterations: usize,
    residual: f32,
    residual_history: Vec<f32>,
    finite_diff_mode: FiniteDiffMode,
    finite_diff_step: f32,
    converged: bool,
    fallback_reason: Option<&'static str>,
) -> ImplicitDiagnostics {
    ImplicitDiagnostics {
        contract_version: HYPERGRAD_CONTRACT_VERSION,
        semantic_owner: HYPERGRAD_SEMANTIC_OWNER,
        autograd_contract_version: AUTOGRAD_CONTRACT_VERSION,
        autograd_semantic_owner: AUTOGRAD_SEMANTIC_OWNER,
        requested_solver,
        effective_solver,
        iterations,
        residual,
        residual_history,
        finite_diff_mode,
        finite_diff_step,
        converged,
        fallback_reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(value: f32, requires_grad: bool) -> AutogradTensor {
        let value = Tensor::from_vec(1, 1, vec![value]).unwrap();
        AutogradTensor::from_tensor(value, requires_grad).unwrap()
    }

    fn sgd_step(
        weights: &AutogradTensor,
        learning_rate: &AutogradTensor,
        gradient: &AutogradTensor,
    ) -> HypergradResult<AutogradTensor> {
        Ok(weights.sub(&learning_rate.hadamard(gradient)?)?)
    }

    fn square_loss(weights: &AutogradTensor) -> HypergradResult<AutogradTensor> {
        Ok(weights.hadamard(weights)?.mean()?)
    }

    fn fixed_point_step(
        weights: &AutogradTensor,
        hyper: &AutogradTensor,
        contraction: &AutogradTensor,
    ) -> HypergradResult<AutogradTensor> {
        Ok(weights.hadamard(contraction)?.add(hyper)?)
    }

    fn coupled_fixed_point_step(
        weights: &AutogradTensor,
        hyper: &AutogradTensor,
        coupling: &AutogradTensor,
    ) -> HypergradResult<AutogradTensor> {
        Ok(weights.matmul(coupling)?.add(hyper)?)
    }

    fn hyper_agnostic_step(
        weights: &AutogradTensor,
        _hyper: &AutogradTensor,
        batch: &AutogradTensor,
    ) -> HypergradResult<AutogradTensor> {
        Ok(weights.add(batch)?)
    }

    fn contraction_only_step(
        weights: &AutogradTensor,
        _hyper: &AutogradTensor,
        contraction: &AutogradTensor,
    ) -> HypergradResult<AutogradTensor> {
        Ok(weights.hadamard(contraction)?)
    }

    fn half_square_loss(weights: &AutogradTensor) -> HypergradResult<AutogradTensor> {
        Ok(weights.hadamard(weights)?.scale(0.5)?.sum()?)
    }

    #[test]
    fn unrolled_hypergradient_matches_closed_form() {
        let weights = scalar(1.0, true);
        let learning_rate = scalar(0.1, true);
        let gradient = scalar(2.0, false);

        let output = unrolled(
            sgd_step,
            weights,
            learning_rate,
            vec![gradient].into_iter(),
            1,
            square_loss,
        )
        .unwrap();

        assert!((output.w_final.item_f32().unwrap() - 0.8).abs() < 1e-6);
        assert!((output.dval_dhyper.data()[0] + 3.2).abs() < 1e-5);
    }

    #[test]
    fn unrolled_disconnected_hypergradient_is_zero_not_stale() {
        let hyper = scalar(0.1, true);
        hyper.scale(7.0).unwrap().backward().unwrap();

        let output = unrolled(
            hyper_agnostic_step,
            scalar(1.0, true),
            hyper.clone(),
            vec![scalar(2.0, false)].into_iter(),
            1,
            square_loss,
        )
        .unwrap();

        assert_eq!(output.dval_dhyper.data(), &[0.0]);
        assert_eq!(hyper.grad().unwrap().data(), &[7.0]);
    }

    #[test]
    fn neumann_implicit_gradient_tracks_linear_fixed_point() {
        let weights = scalar(2.0, true);
        let hyper = scalar(0.5, true);
        let contraction = scalar(0.2, false);
        let options = ImplicitOptions::default()
            .with_solver(Solver::Neumann)
            .with_max_iters(64)
            .with_tolerance(1e-6);

        let output = implicit_with_options(
            fixed_point_step,
            weights,
            hyper,
            vec![contraction].into_iter(),
            half_square_loss,
            options,
        )
        .unwrap();

        assert!((output.dval_dhyper.data()[0] - 2.5).abs() < 2e-4);
        assert!(
            output.diagnostics.converged,
            "diagnostics: {:?}",
            output.diagnostics
        );
        assert_eq!(output.diagnostics.effective_solver, Solver::Neumann);
    }

    #[test]
    fn implicit_disconnected_hypergradient_is_zero_not_stale() {
        let hyper = scalar(0.5, true);
        hyper.scale(9.0).unwrap().backward().unwrap();

        let output = implicit_with_options(
            contraction_only_step,
            scalar(2.0, true),
            hyper.clone(),
            vec![scalar(0.2, false)].into_iter(),
            half_square_loss,
            ImplicitOptions::default(),
        )
        .unwrap();

        assert_eq!(output.dval_dhyper.data(), &[0.0]);
        assert_eq!(hyper.grad().unwrap().data(), &[9.0]);
    }

    #[test]
    fn implicit_disconnected_validation_gradient_is_zero_not_stale() {
        let weights = scalar(2.0, true);
        weights.scale(5.0).unwrap().backward().unwrap();

        let output = implicit_with_options(
            fixed_point_step,
            weights.clone(),
            scalar(0.5, true),
            vec![scalar(0.2, false)].into_iter(),
            |_| Ok(scalar(1.0, false)),
            ImplicitOptions::default(),
        )
        .unwrap();

        assert_eq!(output.dval_dhyper.data(), &[0.0]);
        assert_eq!(weights.grad().unwrap().data(), &[5.0]);
    }

    #[test]
    fn cg_uses_correct_normal_equation_rhs() {
        let weights = scalar(2.0, true);
        let hyper = scalar(0.5, true);
        let contraction = scalar(0.2, false);
        let options = ImplicitOptions::default()
            .with_solver(Solver::Cg)
            .with_max_iters(16)
            .with_tolerance(2e-6)
            .without_cg_fallback();

        let output = implicit_with_options(
            fixed_point_step,
            weights,
            hyper,
            vec![contraction].into_iter(),
            half_square_loss,
            options,
        )
        .unwrap();

        assert!((output.dval_dhyper.data()[0] - 2.5).abs() < 2e-3);
        assert!(
            output.diagnostics.converged,
            "diagnostics: {:?}",
            output.diagnostics
        );
        assert_eq!(output.diagnostics.requested_solver, Solver::Cg);
        assert_eq!(output.diagnostics.effective_solver, Solver::Cg);
        assert_eq!(output.diagnostics.fallback_reason, None);
        assert_eq!(output.diagnostics.semantic_owner, HYPERGRAD_SEMANTIC_OWNER);
        assert_eq!(
            output.diagnostics.autograd_semantic_owner,
            AUTOGRAD_SEMANTIC_OWNER
        );
    }

    #[test]
    fn cg_tracks_a_nonsymmetric_coupled_fixed_point() {
        let weights = AutogradTensor::variable(
            Tensor::from_vec(1, 2, vec![1.0, -0.5]).expect("valid weights"),
        )
        .unwrap();
        let hyper = AutogradTensor::variable(
            Tensor::from_vec(1, 2, vec![0.2, -0.1]).expect("valid hyper parameters"),
        )
        .unwrap();
        let coupling = AutogradTensor::constant(
            Tensor::from_vec(2, 2, vec![0.1, 0.3, 0.0, 0.2]).expect("valid coupling"),
        )
        .unwrap();
        let options = ImplicitOptions::default()
            .with_solver(Solver::Cg)
            .with_max_iters(32)
            .with_tolerance(2e-5)
            .without_cg_fallback();

        let output = implicit_with_options(
            coupled_fixed_point_step,
            weights,
            hyper,
            vec![coupling].into_iter(),
            half_square_loss,
            options,
        )
        .unwrap();

        // v (I - M^T) = [1, -0.5] for M = [[0.1, 0.3], [0, 0.2]].
        let expected = [0.902_777_8, -0.625];
        for (actual, expected) in output.dval_dhyper.data().iter().zip(expected) {
            assert!((actual - expected).abs() < 2e-4, "{actual} != {expected}");
        }
        assert!(output.diagnostics.converged, "{output:?}");
        assert_eq!(output.diagnostics.effective_solver, Solver::Cg);
    }

    #[test]
    fn solver_does_not_claim_convergence_below_f32_residual_floor() {
        let options = ImplicitOptions::default()
            .with_solver(Solver::Neumann)
            .with_max_iters(64)
            .with_tolerance(1e-8);
        let output = implicit_with_options(
            fixed_point_step,
            scalar(2.0, true),
            scalar(0.5, true),
            vec![scalar(0.2, false)].into_iter(),
            half_square_loss,
            options,
        )
        .unwrap();

        assert!(!output.diagnostics.converged);
        assert!(output.diagnostics.residual > 1e-8);
        assert!((output.dval_dhyper.data()[0] - 2.5).abs() < 2e-4);
    }

    #[test]
    fn invalid_options_fail_before_mutating_gradients() {
        let weights = scalar(2.0, true);
        let hyper = scalar(0.5, true);
        let contraction = scalar(0.2, false);
        let options = ImplicitOptions::default().with_tolerance(f32::NAN);

        let error = implicit_with_options(
            fixed_point_step,
            weights.clone(),
            hyper.clone(),
            vec![contraction].into_iter(),
            half_square_loss,
            options,
        )
        .unwrap_err();

        assert!(matches!(error, HypergradError::InvalidTolerance(value) if value.is_nan()));
        assert!(weights.grad().is_none());
        assert!(hyper.grad().is_none());
    }

    #[test]
    fn negative_finite_difference_step_fails_before_graph_evaluation() {
        let weights = scalar(2.0, true);
        let hyper = scalar(0.5, true);
        let options = ImplicitOptions::default().with_finite_diff_eps(-1e-3);

        let error = implicit_with_options(
            fixed_point_step,
            weights.clone(),
            hyper.clone(),
            vec![scalar(0.2, false)].into_iter(),
            half_square_loss,
            options,
        )
        .unwrap_err();

        assert_eq!(error, HypergradError::InvalidFiniteDiffStep(-1e-3));
        assert!(weights.grad().is_none());
        assert!(hyper.grad().is_none());
    }

    #[test]
    fn named_solver_rejects_unknown_values() {
        let error = implicit(
            fixed_point_step,
            scalar(2.0, true),
            scalar(0.5, true),
            vec![scalar(0.2, false)].into_iter(),
            half_square_loss,
            "maybe-cg",
            8,
        )
        .unwrap_err();
        assert_eq!(error, HypergradError::InvalidSolver("maybe-cg".to_string()));
    }

    #[test]
    fn unrolled_requires_every_declared_batch() {
        let error = unrolled(
            sgd_step,
            scalar(1.0, true),
            scalar(0.1, true),
            vec![scalar(2.0, false)].into_iter(),
            2,
            square_loss,
        )
        .unwrap_err();
        assert_eq!(error, HypergradError::MissingBatch { step: 1 });
    }
}
