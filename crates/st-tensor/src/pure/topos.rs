// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Open-cartesian topos guards that keep the pure tensor stack loop-free and
//! numerically stable even in the presence of extreme curvatures.
//!
//! The implementation focuses on three guarantees that were repeatedly flagged
//! as weaknesses in the original stack:
//!
//! * **Numerical safety** – All tensors are validated for finite components and
//!   projected through a saturation window so NaNs and infinities are rewritten
//!   into bounded values.
//! * **Loop freedom** – Traversals through fractal depths are capped by an
//!   "open cartesian" horizon which ensures self-referential rewrites never
//!   re-enter the same stratum.
//! * **Solver determinism** – The conjugate gradient solver exposes explicit
//!   tolerance and iteration limits so hyperbolic Jacobians cannot silently
//!   diverge.
//!
//! The module intentionally stays allocation-light so the new guards can be used
//! from both CPU-only and WASM environments without fighting the borrow checker.

use super::{PureResult, Tensor, TensorError};

/// Maintains safety envelopes for tensors travelling through the pure stack.
#[derive(Clone, Debug)]
pub struct OpenCartesianTopos {
    curvature: f32,
    tolerance: f32,
    saturation: f32,
    max_depth: usize,
    max_volume: usize,
}

impl OpenCartesianTopos {
    /// Builds a new guard. `curvature` must remain negative, `tolerance` and
    /// `saturation` must be positive. `max_depth` and `max_volume` are expressed
    /// in absolute counts rather than logarithms so callers can wire them to
    /// dataset or topology specific limits.
    pub fn new(
        curvature: f32,
        tolerance: f32,
        saturation: f32,
        max_depth: usize,
        max_volume: usize,
    ) -> PureResult<Self> {
        if curvature >= 0.0 {
            return Err(TensorError::NonHyperbolicCurvature { curvature });
        }
        if tolerance <= 0.0 {
            return Err(TensorError::NonPositiveTolerance { tolerance });
        }
        if saturation <= 0.0 {
            return Err(TensorError::NonPositiveSaturation { saturation });
        }
        if max_depth == 0 {
            return Err(TensorError::EmptyInput("topos max depth"));
        }
        if max_volume == 0 {
            return Err(TensorError::EmptyInput("topos max volume"));
        }
        Ok(Self {
            curvature,
            tolerance,
            saturation,
            max_depth,
            max_volume,
        })
    }

    /// Returns the curvature enforced by the topos.
    pub fn curvature(&self) -> f32 {
        self.curvature
    }

    /// Tolerance applied when inverting Jacobians or measuring residuals.
    pub fn tolerance(&self) -> f32 {
        self.tolerance
    }

    /// Returns the saturation limit used to absorb overflows.
    pub fn saturation(&self) -> f32 {
        self.saturation
    }

    /// Maximum permitted traversal depth before the guard considers the topos
    /// closed for the current rewrite.
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Maximum tensor volume allowed inside the topos envelope.
    pub fn max_volume(&self) -> usize {
        self.max_volume
    }

    /// Ensures the provided tensor stays finite and within the permitted volume.
    pub fn guard_tensor(&self, label: &'static str, tensor: &Tensor) -> PureResult<()> {
        let (rows, cols) = tensor.shape();
        let volume = rows.saturating_mul(cols);
        if volume > self.max_volume {
            return Err(TensorError::TensorVolumeExceeded {
                volume,
                max_volume: self.max_volume,
            });
        }
        self.guard_slice(label, tensor.data())
    }

    /// Ensures a buffer remains finite.
    pub fn guard_slice(&self, label: &'static str, slice: &[f32]) -> PureResult<()> {
        for &value in slice {
            if !value.is_finite() {
                return Err(TensorError::NonFiniteValue { label, value });
            }
        }
        Ok(())
    }

    /// Catches runaway recursion depth before it can trigger a feedback loop.
    pub fn ensure_loop_free(&self, depth: usize) -> PureResult<()> {
        if depth >= self.max_depth {
            return Err(TensorError::LoopDetected {
                depth,
                max_depth: self.max_depth,
            });
        }
        Ok(())
    }

    /// Saturates a scalar into the finite window enforced by the topos.
    pub fn saturate(&self, value: f32) -> f32 {
        if !value.is_finite() {
            return 0.0;
        }
        value.clamp(-self.saturation, self.saturation)
    }

    /// Saturates an entire slice in-place.
    pub fn saturate_slice(&self, slice: &mut [f32]) {
        for value in slice.iter_mut() {
            *value = self.saturate(*value);
        }
    }
}

/// Minimal monadic helper that rewrites values through the enclosing topos.
#[derive(Clone, Copy, Debug)]
pub struct RewriteMonad<'a> {
    topos: &'a OpenCartesianTopos,
}

impl<'a> RewriteMonad<'a> {
    /// Wraps a guard for repeated rewrites.
    pub fn new(topos: &'a OpenCartesianTopos) -> Self {
        Self { topos }
    }

    /// Rewrites a scalar by saturating it into the open-cartesian window.
    pub fn rewrite_scalar(&self, value: f32) -> f32 {
        self.topos.saturate(value)
    }

    /// Rewrites a mutable slice and validates the result.
    pub fn rewrite_slice(&self, label: &'static str, slice: &mut [f32]) -> PureResult<()> {
        self.topos.saturate_slice(slice);
        self.topos.guard_slice(label, slice)
    }

    /// Rewrites a tensor and re-validates its envelope.
    pub fn rewrite_tensor(&self, label: &'static str, tensor: &mut Tensor) -> PureResult<()> {
        self.topos.saturate_slice(tensor.data_mut());
        self.topos.guard_tensor(label, tensor)
    }
}

/// Organises tensors rewritten through an open topos into a living "biome".
///
/// The biome behaves like a minimal monad: every tensor absorbed into it is
/// rewritten through the enclosing `OpenCartesianTopos`, saturated into the
/// safety window, and retained as a new shoot.  When the caller is ready to
/// harvest the emergent meaning, the biome collapses all shoots into a guarded
/// canopy tensor that stays within the same topos envelope.
#[derive(Clone, Debug)]
pub struct TensorBiome {
    topos: OpenCartesianTopos,
    shoots: Vec<Tensor>,
    weights: Vec<f32>,
    total_weight: f32,
    shape: Option<(usize, usize)>,
}

impl TensorBiome {
    /// Wraps a biome around an open-cartesian topos.
    pub fn new(topos: OpenCartesianTopos) -> Self {
        Self {
            topos,
            shoots: Vec::new(),
            weights: Vec::new(),
            total_weight: 0.0,
            shape: None,
        }
    }

    /// Returns the guard topos.
    pub fn topos(&self) -> &OpenCartesianTopos {
        &self.topos
    }

    /// Number of shoots currently living inside the biome.
    pub fn len(&self) -> usize {
        self.shoots.len()
    }

    /// Whether the biome is empty.
    pub fn is_empty(&self) -> bool {
        self.shoots.is_empty()
    }

    /// Total accumulated weight across all shoots.
    pub fn total_weight(&self) -> f32 {
        self.total_weight
    }

    /// Returns the individual weights that were assigned to each shoot.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Absorbs a tensor into the biome, rewriting it through the guard topos.
    pub fn absorb(&mut self, label: &'static str, tensor: Tensor) -> PureResult<()> {
        self.absorb_weighted(label, tensor, 1.0)
    }

    /// Absorbs a tensor with an explicit weight that skews the canopy average.
    pub fn absorb_weighted(
        &mut self,
        label: &'static str,
        mut tensor: Tensor,
        weight: f32,
    ) -> PureResult<()> {
        if !weight.is_finite() || weight <= 0.0 {
            return Err(TensorError::NonPositiveWeight { weight });
        }
        let monad = RewriteMonad::new(&self.topos);
        monad.rewrite_tensor(label, &mut tensor)?;
        let shape = tensor.shape();
        if let Some(expected) = self.shape {
            if expected != shape {
                return Err(TensorError::ShapeMismatch {
                    left: expected,
                    right: shape,
                });
            }
        } else {
            self.shape = Some(shape);
        }
        self.shoots.push(tensor);
        self.weights.push(weight);
        self.total_weight += weight;
        Ok(())
    }

    /// Clears all shoots from the biome while preserving the topos.
    pub fn clear(&mut self) {
        self.shoots.clear();
        self.weights.clear();
        self.total_weight = 0.0;
        self.shape = None;
    }

    /// Harvests the biome by averaging all shoots into a guarded canopy tensor.
    pub fn canopy(&self) -> PureResult<Tensor> {
        let (rows, cols) = self.shape.ok_or(TensorError::EmptyInput("tensor_biome"))?;
        if self.is_empty() {
            return Err(TensorError::EmptyInput("tensor_biome"));
        }
        if self.total_weight <= 0.0 {
            return Err(TensorError::NonPositiveWeight {
                weight: self.total_weight,
            });
        }
        let mut acc = Tensor::zeros(rows, cols)?;
        for (shoot, &weight) in self.shoots.iter().zip(self.weights.iter()) {
            acc.add_scaled(shoot, weight)?;
        }
        let mut canopy = acc.scale(1.0 / self.total_weight)?;
        let monad = RewriteMonad::new(&self.topos);
        monad.rewrite_tensor("tensor_biome_canopy", &mut canopy)?;
        Ok(canopy)
    }

    /// Returns a snapshot of the current shoots.
    pub fn shoots(&self) -> &[Tensor] {
        &self.shoots
    }

    /// Stacks all shoots along the row dimension, yielding a dense tensor.
    pub fn stack(&self) -> PureResult<Tensor> {
        let (rows, cols) = self.shape.ok_or(TensorError::EmptyInput("tensor_biome"))?;
        if self.is_empty() {
            return Err(TensorError::EmptyInput("tensor_biome"));
        }
        let mut data = Vec::with_capacity(self.shoots.len() * rows * cols);
        for shoot in &self.shoots {
            data.extend_from_slice(shoot.data());
        }
        Tensor::from_vec(self.shoots.len() * rows, cols, data)
    }
}

/// Deterministic conjugate gradient solver that respects the open-cartesian guard.
pub struct ConjugateGradientSolver<'a> {
    topos: &'a OpenCartesianTopos,
    tolerance: f32,
    max_iterations: usize,
}

impl<'a> ConjugateGradientSolver<'a> {
    /// Creates a solver with an explicit tolerance and iteration cap.
    pub fn new(
        topos: &'a OpenCartesianTopos,
        tolerance: f32,
        max_iterations: usize,
    ) -> PureResult<Self> {
        if tolerance <= 0.0 {
            return Err(TensorError::NonPositiveTolerance { tolerance });
        }
        if max_iterations == 0 {
            return Err(TensorError::EmptyInput("conjugate gradient max iterations"));
        }
        Ok(Self {
            topos,
            tolerance,
            max_iterations,
        })
    }

    /// Returns the solver tolerance.
    pub fn tolerance(&self) -> f32 {
        self.tolerance
    }

    /// Solves a linear system `Ax = b` using repeated matrix-vector products.
    ///
    /// The callback receives the candidate vector and must write `A * src` into
    /// `dst`. The solver enforces explicit tolerances so hyperbolic Jacobians do
    /// not diverge.
    pub fn solve<F>(&self, mut matvec: F, b: &[f32], x: &mut [f32]) -> PureResult<usize>
    where
        F: FnMut(&[f32], &mut [f32]),
    {
        if b.len() != x.len() {
            return Err(TensorError::DataLength {
                expected: b.len(),
                got: x.len(),
            });
        }
        if b.is_empty() {
            return Err(TensorError::EmptyInput("conjugate gradient rhs"));
        }
        self.topos.guard_slice("cg_rhs", b)?;
        self.topos.guard_slice("cg_initial", x)?;
        let mut r = vec![0.0f32; b.len()];
        let mut p = vec![0.0f32; b.len()];
        let mut ap = vec![0.0f32; b.len()];
        matvec(x, &mut ap);
        for ((r_i, p_i), (&b_i, &ap_i)) in
            r.iter_mut().zip(p.iter_mut()).zip(b.iter().zip(ap.iter()))
        {
            *r_i = b_i - ap_i;
            *p_i = *r_i;
        }
        let mut rsold = dot(&r, &r);
        let tol = self.tolerance.max(self.topos.tolerance());
        if rsold.sqrt() <= tol {
            return Ok(0);
        }
        for iter in 0..self.max_iterations {
            matvec(&p, &mut ap);
            let denom = dot(&p, &ap);
            if denom.abs() <= tol {
                return Err(TensorError::ConjugateGradientDiverged {
                    residual: rsold.sqrt(),
                    tolerance: tol,
                });
            }
            let alpha = rsold / denom;
            for (x_i, p_i) in x.iter_mut().zip(p.iter()) {
                *x_i = self.topos.saturate(*x_i + alpha * *p_i);
            }
            for (r_i, ap_i) in r.iter_mut().zip(ap.iter()) {
                *r_i -= alpha * *ap_i;
            }
            let rsnew = dot(&r, &r);
            if rsnew.sqrt() <= tol {
                self.topos.guard_slice("cg_solution", x)?;
                return Ok(iter + 1);
            }
            let beta = rsnew / rsold.max(tol * tol);
            for (p_i, r_i) in p.iter_mut().zip(r.iter()) {
                *p_i = self.topos.saturate(*r_i + beta * *p_i);
            }
            rsold = rsnew;
        }
        Err(TensorError::ConjugateGradientDiverged {
            residual: rsold.sqrt(),
            tolerance: tol,
        })
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo_topos() -> OpenCartesianTopos {
        OpenCartesianTopos::new(-1.0, 1e-5, 10.0, 64, 4096).unwrap()
    }

    #[test]
    fn topos_rejects_non_finite_values() {
        let topos = demo_topos();
        let tensor = Tensor::from_vec(1, 2, vec![1.0, f32::INFINITY]).unwrap();
        let err = topos.guard_tensor("nonfinite", &tensor).unwrap_err();
        matches!(err, TensorError::NonFiniteValue { .. });
    }

    #[test]
    fn biome_absorbs_and_harvests() {
        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos.clone());
        let big = topos.saturation() * 2.0;
        biome
            .absorb(
                "biome_shoot_a",
                Tensor::from_vec(1, 2, vec![big, 0.5]).unwrap(),
            )
            .unwrap();
        biome
            .absorb(
                "biome_shoot_b",
                Tensor::from_vec(1, 2, vec![-big, 1.0]).unwrap(),
            )
            .unwrap();
        let canopy = biome.canopy().unwrap();
        assert_eq!(canopy.shape(), (1, 2));
        let data = canopy.data();
        assert!((data[0] - 0.0).abs() < 1e-6);
        assert!((data[1] - 0.75).abs() < 1e-6);
        assert_eq!(biome.total_weight(), 2.0);
    }

    #[test]
    fn biome_detects_shape_mismatch() {
        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos);
        biome
            .absorb(
                "biome_shape_a",
                Tensor::from_vec(2, 1, vec![0.1, 0.2]).unwrap(),
            )
            .unwrap();
        let err = biome
            .absorb(
                "biome_shape_b",
                Tensor::from_vec(1, 2, vec![0.1, 0.2]).unwrap(),
            )
            .unwrap_err();
        assert!(matches!(err, TensorError::ShapeMismatch { .. }));
    }

    #[test]
    fn biome_weighted_canopy_respects_shoot_weights() {
        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos);
        biome
            .absorb_weighted(
                "weighted_a",
                Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
                1.0,
            )
            .unwrap();
        biome
            .absorb_weighted(
                "weighted_b",
                Tensor::from_vec(1, 1, vec![3.0]).unwrap(),
                3.0,
            )
            .unwrap();
        let canopy = biome.canopy().unwrap();
        assert_eq!(canopy.data(), &[2.5]);
        assert_eq!(biome.weights(), &[1.0, 3.0]);
        assert!((biome.total_weight() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn biome_stack_concatenates_shoots() {
        let topos = demo_topos();
        let mut biome = TensorBiome::new(topos);
        biome
            .absorb("stack_a", Tensor::from_vec(1, 2, vec![0.1, 0.2]).unwrap())
            .unwrap();
        biome
            .absorb("stack_b", Tensor::from_vec(1, 2, vec![0.3, 0.4]).unwrap())
            .unwrap();
        let stacked = biome.stack().unwrap();
        assert_eq!(stacked.shape(), (2, 2));
        assert_eq!(stacked.data(), &[0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn rewrite_monad_saturates_values() {
        let topos = demo_topos();
        let monad = RewriteMonad::new(&topos);
        let mut tensor = Tensor::from_vec(1, 2, vec![20.0, -20.0]).unwrap();
        monad.rewrite_tensor("rewrite", &mut tensor).unwrap();
        assert!(tensor.data().iter().all(|v| v.abs() <= topos.saturation()));
    }

    #[test]
    fn conjugate_gradient_converges_with_guard() {
        let topos = demo_topos();
        let solver = ConjugateGradientSolver::new(&topos, 1e-5, 32).unwrap();
        let matrix = [4.0f32, 1.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 2.0];
        let mut matvec = |src: &[f32], dst: &mut [f32]| {
            dst.fill(0.0);
            for row in 0..3 {
                for col in 0..3 {
                    dst[row] += matrix[row * 3 + col] * src[col];
                }
            }
        };
        let b = [1.0f32, 2.0, 3.0];
        let mut x = [0.0f32; 3];
        let iterations = solver.solve(&mut matvec, &b, &mut x).unwrap();
        assert!(iterations > 0);
        let mut residual = [0.0f32; 3];
        matvec(&x, &mut residual);
        for (res, rhs) in residual.iter_mut().zip(b.iter()) {
            *res -= rhs;
        }
        let norm: f32 = residual.iter().map(|v| v * v).sum();
        assert!(norm.sqrt() <= solver.tolerance().max(topos.tolerance()));
    }
}
