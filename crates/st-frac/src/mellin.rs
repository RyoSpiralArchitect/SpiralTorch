// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::mellin_types::{ComplexScalar, MellinError, MellinResult, Scalar, ZSpaceError};
use crate::zspace::{
    evaluate_weighted_series, evaluate_weighted_series_many, mellin_log_lattice_prefactor,
    prepare_weighted_series, trapezoidal_weights, weighted_z_transform,
};

/// Change-of-variable helper for Mellin integrals.
///
/// Given a positive interval `(a, b)` we operate in the logarithmic domain by
/// setting `x = exp(t)`. The integral bounds in the log domain then become
/// `(ln a, ln b)`, and the Mellin kernel simplifies to `exp(s * t)`.
#[inline]
fn map_range_to_log(range: (Scalar, Scalar)) -> MellinResult<(Scalar, Scalar)> {
    let (a, b) = range;
    if !(a.is_finite() && b.is_finite() && a > 0.0 && b > 0.0 && a < b) {
        return Err(MellinError::InvalidRange);
    }
    Ok((a.ln(), b.ln()))
}

fn float_bits(value: Scalar) -> u128 {
    value.to_bits() as u128
}

/// Sample a function on a log-uniform lattice.
///
/// The returned samples correspond to `x_k = exp(log_start + k * log_step)` for
/// `k = 0..len-1`.
pub fn sample_log_uniform<F>(
    log_start: Scalar,
    log_step: Scalar,
    len: usize,
    f: F,
) -> MellinResult<Vec<ComplexScalar>>
where
    F: Fn(Scalar) -> ComplexScalar,
{
    if len == 0 {
        return Err(MellinError::EmptySamples);
    }
    if !(log_step.is_finite() && log_step > 0.0) {
        return Err(MellinError::InvalidLogStep);
    }
    if !log_start.is_finite() {
        return Err(MellinError::InvalidLogStart);
    }

    let mut samples = Vec::with_capacity(len);
    for idx in 0..len {
        let t = log_start + log_step * idx as Scalar;
        let x = t.exp();
        let val = f(x);
        if !(val.re.is_finite() && val.im.is_finite()) {
            return Err(MellinError::NonFiniteSample { index: idx });
        }
        samples.push(val);
    }
    Ok(samples)
}

/// Convenience sampler for the reference function `f(x) = exp(-x)`.
///
/// This is used by demos and tests because the Mellin transform reduces to the
/// Gamma function on the real axis.
pub fn sample_log_uniform_exp_decay(
    log_start: Scalar,
    log_step: Scalar,
    len: usize,
) -> MellinResult<Vec<ComplexScalar>> {
    sample_log_uniform(log_start, log_step, len, |x| ComplexScalar::new((-x).exp(), 0.0))
}

/// Convenience sampler for `f(x) = exp(-rate * x)`.
///
/// When `rate=1` this matches [`sample_log_uniform_exp_decay`]. The continuous Mellin transform
/// obeys `M{exp(-rate x)}(s) = rate^{-s} Γ(s)` for Re(s) > 0.
pub fn sample_log_uniform_exp_decay_scaled(
    log_start: Scalar,
    log_step: Scalar,
    len: usize,
    rate: Scalar,
) -> MellinResult<Vec<ComplexScalar>> {
    if !(rate.is_finite() && rate > 0.0) {
        return Err(MellinError::InvalidRate);
    }
    sample_log_uniform(log_start, log_step, len, move |x| {
        ComplexScalar::new((-(rate * x)).exp(), 0.0)
    })
}

#[inline]
fn complex_magnitude(value: ComplexScalar) -> Scalar {
    value.norm()
}

/// Precomputed Z-plane evaluation points (prefactor + z) for Mellin lattices.
///
/// This is meant to be re-used across many sample series (e.g. in training loops):
/// precompute the evaluation mesh once, then feed different weighted series.
#[derive(Clone, Debug)]
pub struct MellinEvalPlan {
    log_start: Scalar,
    log_step: Scalar,
    prefactors: Vec<ComplexScalar>,
    z_points: Vec<ComplexScalar>,
    shape: (usize, usize),
}

impl MellinEvalPlan {
    /// Build a plan for arbitrary Mellin evaluation points.
    pub fn many(log_start: Scalar, log_step: Scalar, s_values: &[ComplexScalar]) -> MellinResult<Self> {
        if s_values.is_empty() {
            return Ok(Self {
                log_start,
                log_step,
                prefactors: Vec::new(),
                z_points: Vec::new(),
                shape: (0, 0),
            });
        }

        let mut prefactors = Vec::with_capacity(s_values.len());
        let mut z_points = Vec::with_capacity(s_values.len());
        for &s in s_values {
            let (prefactor, z) = mellin_log_lattice_prefactor(log_start, log_step, s)?;
            prefactors.push(prefactor);
            z_points.push(z);
        }

        Ok(Self {
            log_start,
            log_step,
            prefactors,
            z_points,
            shape: (1, s_values.len()),
        })
    }

    /// Build a plan for `s = real + i * t` along a vertical line.
    pub fn vertical_line(
        log_start: Scalar,
        log_step: Scalar,
        real: Scalar,
        imag_values: &[Scalar],
    ) -> MellinResult<Self> {
        if imag_values.is_empty() {
            return Ok(Self {
                log_start,
                log_step,
                prefactors: Vec::new(),
                z_points: Vec::new(),
                shape: (0, 0),
            });
        }

        let mut prefactors = Vec::with_capacity(imag_values.len());
        let mut z_points = Vec::with_capacity(imag_values.len());
        for &imag in imag_values {
            let s = ComplexScalar::new(real, imag);
            let (prefactor, z) = mellin_log_lattice_prefactor(log_start, log_step, s)?;
            prefactors.push(prefactor);
            z_points.push(z);
        }

        Ok(Self {
            log_start,
            log_step,
            prefactors,
            z_points,
            shape: (1, imag_values.len()),
        })
    }

    /// Build a plan for a 2D mesh `s = real + i * imag`.
    ///
    /// The layout matches [`MellinLogGrid::evaluate_mesh`]:
    /// `out[real_index * imag_len + imag_index]`.
    pub fn mesh(
        log_start: Scalar,
        log_step: Scalar,
        real_values: &[Scalar],
        imag_values: &[Scalar],
    ) -> MellinResult<Self> {
        if real_values.is_empty() || imag_values.is_empty() {
            return Ok(Self {
                log_start,
                log_step,
                prefactors: Vec::new(),
                z_points: Vec::new(),
                shape: (0, 0),
            });
        }

        let rows = real_values.len();
        let cols = imag_values.len();
        let mut prefactors = Vec::with_capacity(rows * cols);
        let mut z_points = Vec::with_capacity(rows * cols);
        for &real in real_values {
            for &imag in imag_values {
                let s = ComplexScalar::new(real, imag);
                let (prefactor, z) = mellin_log_lattice_prefactor(log_start, log_step, s)?;
                prefactors.push(prefactor);
                z_points.push(z);
            }
        }

        Ok(Self {
            log_start,
            log_step,
            prefactors,
            z_points,
            shape: (rows, cols),
        })
    }

    pub fn log_start(&self) -> Scalar {
        self.log_start
    }

    pub fn log_step(&self) -> Scalar {
        self.log_step
    }

    pub fn len(&self) -> usize {
        self.z_points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.z_points.is_empty()
    }

    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    pub fn evaluate(&self, weighted: &[ComplexScalar]) -> MellinResult<Vec<ComplexScalar>> {
        if self.z_points.is_empty() {
            return Ok(Vec::new());
        }
        let series = evaluate_weighted_series_many(weighted, &self.z_points)?;
        Ok(series
            .into_iter()
            .zip(self.prefactors.iter())
            .map(|(value, prefactor)| value * *prefactor)
            .collect())
    }

    pub fn evaluate_magnitude(&self, weighted: &[ComplexScalar]) -> MellinResult<Vec<Scalar>> {
        if self.z_points.is_empty() {
            return Ok(Vec::new());
        }
        let series = evaluate_weighted_series_many(weighted, &self.z_points)?;
        Ok(series
            .into_iter()
            .zip(self.prefactors.iter())
            .map(|(value, prefactor)| complex_magnitude(value) * complex_magnitude(*prefactor))
            .collect())
    }

    pub fn evaluate_log_magnitude(&self, weighted: &[ComplexScalar], epsilon: Scalar) -> MellinResult<Vec<Scalar>> {
        if self.z_points.is_empty() {
            return Ok(Vec::new());
        }
        let series = evaluate_weighted_series_many(weighted, &self.z_points)?;
        Ok(series
            .into_iter()
            .zip(self.prefactors.iter())
            .map(|(value, prefactor)| {
                let mag = complex_magnitude(value) * complex_magnitude(*prefactor);
                if epsilon.is_finite() && epsilon > 0.0 {
                    (mag + epsilon).ln()
                } else {
                    mag.ln_1p()
                }
            })
            .collect())
    }

    #[cfg(feature = "wgpu")]
    pub fn evaluate_gpu(&self, weighted: &[ComplexScalar]) -> MellinResult<Vec<ComplexScalar>> {
        if self.z_points.is_empty() {
            return Ok(Vec::new());
        }
        let series = crate::mellin_wgpu::evaluate_weighted_series_many_gpu(weighted, &self.z_points)?;
        Ok(series
            .into_iter()
            .zip(self.prefactors.iter())
            .map(|(value, prefactor)| value * *prefactor)
            .collect())
    }

    #[cfg(feature = "wgpu")]
    pub fn evaluate_magnitude_gpu(&self, weighted: &[ComplexScalar]) -> MellinResult<Vec<Scalar>> {
        if self.z_points.is_empty() {
            return Ok(Vec::new());
        }
        let series = crate::mellin_wgpu::evaluate_weighted_series_many_gpu(weighted, &self.z_points)?;
        Ok(series
            .into_iter()
            .zip(self.prefactors.iter())
            .map(|(value, prefactor)| complex_magnitude(value) * complex_magnitude(*prefactor))
            .collect())
    }

    #[cfg(feature = "wgpu")]
    pub fn evaluate_log_magnitude_gpu(&self, weighted: &[ComplexScalar], epsilon: Scalar) -> MellinResult<Vec<Scalar>> {
        if self.z_points.is_empty() {
            return Ok(Vec::new());
        }
        let series = crate::mellin_wgpu::evaluate_weighted_series_many_gpu(weighted, &self.z_points)?;
        Ok(series
            .into_iter()
            .zip(self.prefactors.iter())
            .map(|(value, prefactor)| {
                let mag = complex_magnitude(value) * complex_magnitude(*prefactor);
                if epsilon.is_finite() && epsilon > 0.0 {
                    (mag + epsilon).ln()
                } else {
                    mag.ln_1p()
                }
            })
            .collect())
    }
}

/// Pre-sampled log lattice with helpers for Mellin/Hilbert evaluations.
#[derive(Clone, Debug)]
pub struct MellinLogGrid {
    log_start: Scalar,
    log_step: Scalar,
    samples: Vec<ComplexScalar>,
    weights: Vec<Scalar>,
    weighted: Vec<ComplexScalar>,
}

impl MellinLogGrid {
    /// Construct a grid from existing log-uniform samples.
    pub fn new(
        log_start: Scalar,
        log_step: Scalar,
        samples: Vec<ComplexScalar>,
    ) -> MellinResult<Self> {
        if samples.is_empty() {
            return Err(MellinError::EmptySamples);
        }
        if !(log_step.is_finite() && log_step > 0.0) {
            return Err(MellinError::InvalidLogStep);
        }
        if !log_start.is_finite() {
            return Err(MellinError::InvalidLogStart);
        }
        let weights = trapezoidal_weights(samples.len())?;
        let weighted = prepare_weighted_series(&samples, &weights)?;
        Ok(Self {
            log_start,
            log_step,
            samples,
            weights,
            weighted,
        })
    }

    /// Sample a function over a log-uniform lattice and build the grid.
    pub fn from_function<F>(
        log_start: Scalar,
        log_step: Scalar,
        len: usize,
        f: F,
    ) -> MellinResult<Self>
    where
        F: Fn(Scalar) -> ComplexScalar,
    {
        let samples = sample_log_uniform(log_start, log_step, len, f)?;
        Self::new(log_start, log_step, samples)
    }

    /// Evaluate the Mellin transform using the pre-sampled lattice.
    pub fn evaluate(&self, s: ComplexScalar) -> MellinResult<ComplexScalar> {
        let (prefactor, z) = mellin_log_lattice_prefactor(self.log_start, self.log_step, s)?;
        let series = evaluate_weighted_series(&self.weighted, z)?;
        Ok(prefactor * series)
    }

    /// Evaluate the Mellin transform at multiple points sharing the same samples.
    pub fn evaluate_many(&self, s_values: &[ComplexScalar]) -> MellinResult<Vec<ComplexScalar>> {
        if s_values.is_empty() {
            return Ok(Vec::new());
        }
        #[cfg(feature = "wgpu")]
        match self.evaluate_many_gpu(s_values) {
            Ok(values) => return Ok(values),
            Err(MellinError::Gpu(_)) => {}
            Err(other) => return Err(other),
        }

        self.evaluate_many_cpu(s_values)
    }

    fn assert_plan_compatible(&self, plan: &MellinEvalPlan) -> MellinResult<()> {
        if float_bits(self.log_start) != float_bits(plan.log_start())
            || float_bits(self.log_step) != float_bits(plan.log_step())
        {
            return Err(MellinError::LatticeMismatch);
        }
        Ok(())
    }

    /// Precompute Z-plane evaluation points for a mesh.
    pub fn plan_mesh(
        &self,
        real_values: &[Scalar],
        imag_values: &[Scalar],
    ) -> MellinResult<MellinEvalPlan> {
        MellinEvalPlan::mesh(self.log_start, self.log_step, real_values, imag_values)
    }

    /// Precompute Z-plane evaluation points for a vertical line.
    pub fn plan_vertical_line(&self, real: Scalar, imag_values: &[Scalar]) -> MellinResult<MellinEvalPlan> {
        MellinEvalPlan::vertical_line(self.log_start, self.log_step, real, imag_values)
    }

    /// Evaluate the Mellin transform using a precomputed plan.
    pub fn evaluate_plan(&self, plan: &MellinEvalPlan) -> MellinResult<Vec<ComplexScalar>> {
        self.assert_plan_compatible(plan)?;
        #[cfg(feature = "wgpu")]
        match plan.evaluate_gpu(&self.weighted) {
            Ok(values) => return Ok(values),
            Err(MellinError::Gpu(_)) => {}
            Err(other) => return Err(other),
        }
        plan.evaluate(&self.weighted)
    }

    /// Evaluate magnitudes using a precomputed plan.
    pub fn evaluate_plan_magnitude(&self, plan: &MellinEvalPlan) -> MellinResult<Vec<Scalar>> {
        self.assert_plan_compatible(plan)?;
        #[cfg(feature = "wgpu")]
        match plan.evaluate_magnitude_gpu(&self.weighted) {
            Ok(values) => return Ok(values),
            Err(MellinError::Gpu(_)) => {}
            Err(other) => return Err(other),
        }
        plan.evaluate_magnitude(&self.weighted)
    }

    /// Evaluate log-magnitudes using a precomputed plan.
    pub fn evaluate_plan_log_magnitude(
        &self,
        plan: &MellinEvalPlan,
        epsilon: Scalar,
    ) -> MellinResult<Vec<Scalar>> {
        self.assert_plan_compatible(plan)?;
        #[cfg(feature = "wgpu")]
        match plan.evaluate_log_magnitude_gpu(&self.weighted, epsilon) {
            Ok(values) => return Ok(values),
            Err(MellinError::Gpu(_)) => {}
            Err(other) => return Err(other),
        }
        plan.evaluate_log_magnitude(&self.weighted, epsilon)
    }

    fn assert_grid_compatible(&self, other: &MellinLogGrid) -> MellinResult<()> {
        if float_bits(self.log_start) != float_bits(other.log_start)
            || float_bits(self.log_step) != float_bits(other.log_step)
        {
            return Err(MellinError::LatticeMismatch);
        }
        if self.samples.len() != other.samples.len() {
            return Err(MellinError::SampleLengthMismatch {
                expected: self.samples.len(),
                got: other.samples.len(),
            });
        }
        Ok(())
    }

    /// Compute an L2 loss + gradient that matches the Mellin transform of `target`.
    ///
    /// This is intended for training loops: reuse the same `plan` for both grids
    /// and update `self.samples` using the returned gradient.
    ///
    /// The returned loss is the mean squared magnitude of the complex error
    /// `M_self(s) - M_target(s)` over all plan points.
    pub fn l2_loss_grad_plan_match_grid(
        &self,
        plan: &MellinEvalPlan,
        target: &MellinLogGrid,
    ) -> MellinResult<(Scalar, Vec<ComplexScalar>)> {
        self.assert_plan_compatible(plan)?;
        self.assert_grid_compatible(target)?;

        let points = plan.z_points.len();
        let len = self.samples.len();
        let mut grad = vec![ComplexScalar::new(0.0, 0.0); len];
        if points == 0 || len == 0 {
            return Ok((0.0, grad));
        }

        let inv_points = 1.0 / points as Scalar;
        let scale = 2.0 * inv_points;
        let mut loss = 0.0;

        for (z, prefactor) in plan.z_points.iter().zip(plan.prefactors.iter()) {
            let mut series = ComplexScalar::new(0.0, 0.0);
            let mut pow = ComplexScalar::new(1.0, 0.0);
            for idx in 0..len {
                let diff = self.samples[idx] - target.samples[idx];
                let coeff = diff * ComplexScalar::new(self.weights[idx], 0.0);
                series += coeff * pow;
                pow *= *z;
            }

            let error = series * *prefactor;
            loss += error.norm_sqr();

            let factor = error * prefactor.conj();
            let conj_z = z.conj();
            let mut pow_conj = ComplexScalar::new(1.0, 0.0);
            for (idx, grad_slot) in grad.iter_mut().enumerate().take(len) {
                let w = self.weights[idx];
                *grad_slot += factor * pow_conj * ComplexScalar::new(w, 0.0);
                pow_conj *= conj_z;
            }
        }

        loss *= inv_points;
        for g in &mut grad {
            *g *= scale;
        }

        Ok((loss, grad))
    }

    /// Single gradient-descent step that matches the Mellin transform of `target`.
    pub fn train_step_l2_plan_match_grid(
        &mut self,
        plan: &MellinEvalPlan,
        target: &MellinLogGrid,
        lr: Scalar,
    ) -> MellinResult<Scalar> {
        if !(lr.is_finite() && lr > 0.0) {
            return Err(MellinError::InvalidLearningRate);
        }

        let (loss, grad) = self.l2_loss_grad_plan_match_grid(plan, target)?;
        for (sample, grad) in self.samples.iter_mut().zip(grad.iter()) {
            sample.re -= lr * grad.re;
            sample.im -= lr * grad.im;
        }
        self.weighted = prepare_weighted_series(&self.samples, &self.weights)?;
        Ok(loss)
    }

    /// Evaluate the magnitude `|M(s)|`.
    pub fn evaluate_magnitude(&self, s: ComplexScalar) -> MellinResult<Scalar> {
        Ok(complex_magnitude(self.evaluate(s)?))
    }

    /// Evaluate the magnitude `|M(s)|` at multiple points.
    pub fn evaluate_many_magnitude(&self, s_values: &[ComplexScalar]) -> MellinResult<Vec<Scalar>> {
        let values = self.evaluate_many(s_values)?;
        Ok(values.into_iter().map(complex_magnitude).collect())
    }

    /// Evaluate the Mellin transform over a 2D mesh of `s = real + i * imag`.
    ///
    /// The returned values are laid out in row-major order with `real_values`
    /// as the outer dimension:
    /// `out[real_index * imag_values.len() + imag_index]`.
    pub fn evaluate_mesh(
        &self,
        real_values: &[Scalar],
        imag_values: &[Scalar],
    ) -> MellinResult<Vec<ComplexScalar>> {
        if real_values.is_empty() || imag_values.is_empty() {
            return Ok(Vec::new());
        }
        let mut s_values = Vec::with_capacity(real_values.len() * imag_values.len());
        for &real in real_values {
            for &imag in imag_values {
                s_values.push(ComplexScalar::new(real, imag));
            }
        }
        self.evaluate_many(&s_values)
    }

    /// Evaluate a mesh and return magnitudes laid out row-major.
    pub fn evaluate_mesh_magnitude(
        &self,
        real_values: &[Scalar],
        imag_values: &[Scalar],
    ) -> MellinResult<Vec<Scalar>> {
        let plan = MellinEvalPlan::mesh(self.log_start, self.log_step, real_values, imag_values)?;
        #[cfg(feature = "wgpu")]
        match plan.evaluate_magnitude_gpu(&self.weighted) {
            Ok(values) => return Ok(values),
            Err(MellinError::Gpu(_)) => {}
            Err(other) => return Err(other),
        }
        plan.evaluate_magnitude(&self.weighted)
    }

    /// Evaluate a mesh and return log-magnitudes (default uses `ln(1+|M|)`).
    pub fn evaluate_mesh_log_magnitude(
        &self,
        real_values: &[Scalar],
        imag_values: &[Scalar],
        epsilon: Scalar,
    ) -> MellinResult<Vec<Scalar>> {
        let plan = MellinEvalPlan::mesh(self.log_start, self.log_step, real_values, imag_values)?;
        #[cfg(feature = "wgpu")]
        match plan.evaluate_log_magnitude_gpu(&self.weighted, epsilon) {
            Ok(values) => return Ok(values),
            Err(MellinError::Gpu(_)) => {}
            Err(other) => return Err(other),
        }
        plan.evaluate_log_magnitude(&self.weighted, epsilon)
    }

    /// Evaluate the Mellin transform over a 2D mesh using pre-weighted series coefficients.
    pub fn evaluate_mesh_with_series(
        &self,
        real_values: &[Scalar],
        imag_values: &[Scalar],
        weighted: &[ComplexScalar],
    ) -> MellinResult<Vec<ComplexScalar>> {
        if real_values.is_empty() || imag_values.is_empty() {
            return Ok(Vec::new());
        }
        let mut s_values = Vec::with_capacity(real_values.len() * imag_values.len());
        for &real in real_values {
            for &imag in imag_values {
                s_values.push(ComplexScalar::new(real, imag));
            }
        }
        self.evaluate_many_with_series(&s_values, weighted)
    }

    /// Precompute the Z-plane weighted series associated with the grid samples.
    pub fn weighted_series(&self) -> MellinResult<Vec<ComplexScalar>> {
        Ok(self.weighted.clone())
    }

    /// Evaluate the Mellin transform using pre-weighted Z-series coefficients.
    pub fn evaluate_with_series(
        &self,
        s: ComplexScalar,
        weighted: &[ComplexScalar],
    ) -> MellinResult<ComplexScalar> {
        if weighted.len() != self.samples.len() {
            return Err(ZSpaceError::WeightLengthMismatch {
                samples: self.samples.len(),
                weights: weighted.len(),
            }
            .into());
        }
        let (prefactor, z) = mellin_log_lattice_prefactor(self.log_start, self.log_step, s)?;
        let series = evaluate_weighted_series(weighted, z)?;
        Ok(prefactor * series)
    }

    /// Evaluate the Mellin transform at multiple points using pre-weighted coefficients.
    pub fn evaluate_many_with_series(
        &self,
        s_values: &[ComplexScalar],
        weighted: &[ComplexScalar],
    ) -> MellinResult<Vec<ComplexScalar>> {
        if weighted.len() != self.samples.len() {
            return Err(ZSpaceError::WeightLengthMismatch {
                samples: self.samples.len(),
                weights: weighted.len(),
            }
            .into());
        }
        if s_values.is_empty() {
            return Ok(Vec::new());
        }
        let mut prefactors = Vec::with_capacity(s_values.len());
        let mut z_points = Vec::with_capacity(s_values.len());
        for &s in s_values {
            let (prefactor, z) = mellin_log_lattice_prefactor(self.log_start, self.log_step, s)?;
            prefactors.push(prefactor);
            z_points.push(z);
        }
        let series = evaluate_weighted_series_many(weighted, &z_points)?;
        Ok(series
            .into_iter()
            .zip(prefactors)
            .map(|(series, prefactor)| prefactor * series)
            .collect())
    }

    /// Sweep the Mellin transform along a vertical line `s = real + i * t`.
    pub fn evaluate_vertical_line(
        &self,
        real: Scalar,
        imag_values: &[Scalar],
    ) -> MellinResult<Vec<ComplexScalar>> {
        if imag_values.is_empty() {
            return Ok(Vec::new());
        }
        #[cfg(feature = "wgpu")]
        match self.evaluate_vertical_line_gpu(real, imag_values) {
            Ok(values) => return Ok(values),
            Err(MellinError::Gpu(_)) => {}
            Err(other) => return Err(other),
        }

        self.evaluate_vertical_line_cpu(real, imag_values)
    }

    /// Evaluate `|M(real + i·t)|` along a vertical line.
    pub fn evaluate_vertical_line_magnitude(
        &self,
        real: Scalar,
        imag_values: &[Scalar],
    ) -> MellinResult<Vec<Scalar>> {
        let plan = MellinEvalPlan::vertical_line(self.log_start, self.log_step, real, imag_values)?;
        #[cfg(feature = "wgpu")]
        match plan.evaluate_magnitude_gpu(&self.weighted) {
            Ok(values) => return Ok(values),
            Err(MellinError::Gpu(_)) => {}
            Err(other) => return Err(other),
        }
        plan.evaluate_magnitude(&self.weighted)
    }

    /// Evaluate `ln(1 + |M(real + i·t)|)` (or `ln(|M| + epsilon)` when `epsilon>0`).
    pub fn evaluate_vertical_line_log_magnitude(
        &self,
        real: Scalar,
        imag_values: &[Scalar],
        epsilon: Scalar,
    ) -> MellinResult<Vec<Scalar>> {
        let plan = MellinEvalPlan::vertical_line(self.log_start, self.log_step, real, imag_values)?;
        #[cfg(feature = "wgpu")]
        match plan.evaluate_log_magnitude_gpu(&self.weighted, epsilon) {
            Ok(values) => return Ok(values),
            Err(MellinError::Gpu(_)) => {}
            Err(other) => return Err(other),
        }
        plan.evaluate_log_magnitude(&self.weighted, epsilon)
    }

    /// Return the number of log-uniform samples stored in the grid.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check whether the grid has no samples.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Expose the logarithmic start coordinate.
    pub fn log_start(&self) -> Scalar {
        self.log_start
    }

    /// Expose the logarithmic step between samples.
    pub fn log_step(&self) -> Scalar {
        self.log_step
    }

    /// Access the raw log-uniform samples.
    pub fn samples(&self) -> &[ComplexScalar] {
        &self.samples
    }

    /// Access the trapezoidal weights associated with the grid.
    pub fn weights(&self) -> &[Scalar] {
        &self.weights
    }

    /// Report the truncated support in the original domain.
    pub fn support(&self) -> (Scalar, Scalar) {
        let start = self.log_start;
        let end = self.log_start + self.log_step * (self.samples.len() - 1) as Scalar;
        (start.exp(), end.exp())
    }

    fn assert_same_lattice(&self, other: &Self) -> MellinResult<()> {
        if self.len() != other.len()
            || float_bits(self.log_start) != float_bits(other.log_start)
            || float_bits(self.log_step) != float_bits(other.log_step)
        {
            return Err(MellinError::LatticeMismatch);
        }
        Ok(())
    }

    /// Hilbert-space inner product approximated on the log-uniform lattice.
    pub fn hilbert_inner_product(&self, other: &Self) -> MellinResult<ComplexScalar> {
        self.assert_same_lattice(other)?;

        let mut acc = ComplexScalar::new(0.0, 0.0);
        for ((lhs, rhs), &w) in self
            .samples
            .iter()
            .zip(other.samples.iter())
            .zip(self.weights.iter())
        {
            acc += lhs.conj() * *rhs * ComplexScalar::new(w, 0.0);
        }
        Ok(acc * ComplexScalar::new(self.log_step, 0.0))
    }

    /// Hilbert-space norm induced by the Mellin lattice.
    pub fn hilbert_norm(&self) -> MellinResult<Scalar> {
        let ip = self.hilbert_inner_product(self)?;
        if ip.re < -1e-6 {
            return Err(MellinError::NegativeInnerProduct { value: ip });
        }
        Ok(ip.re.max(0.0).sqrt())
    }

    fn evaluate_many_cpu(&self, s_values: &[ComplexScalar]) -> MellinResult<Vec<ComplexScalar>> {
        let plan = MellinEvalPlan::many(self.log_start, self.log_step, s_values)?;
        plan.evaluate(&self.weighted)
    }

    fn evaluate_vertical_line_cpu(
        &self,
        real: Scalar,
        imag_values: &[Scalar],
    ) -> MellinResult<Vec<ComplexScalar>> {
        let plan = MellinEvalPlan::vertical_line(self.log_start, self.log_step, real, imag_values)?;
        plan.evaluate(&self.weighted)
    }

    #[cfg(feature = "wgpu")]
    fn evaluate_many_gpu(&self, s_values: &[ComplexScalar]) -> MellinResult<Vec<ComplexScalar>> {
        let plan = MellinEvalPlan::many(self.log_start, self.log_step, s_values)?;
        plan.evaluate_gpu(&self.weighted)
    }

    #[cfg(feature = "wgpu")]
    fn evaluate_vertical_line_gpu(
        &self,
        real: Scalar,
        imag_values: &[Scalar],
    ) -> MellinResult<Vec<ComplexScalar>> {
        let plan = MellinEvalPlan::vertical_line(self.log_start, self.log_step, real, imag_values)?;
        plan.evaluate_gpu(&self.weighted)
    }
}

/// Numerically integrate a locally integrable function over `(0, ∞)` to obtain
/// its Mellin transform at `s`.
///
/// The caller supplies a truncation range `(a, b)` that captures the support of
/// the function (or the region where it is numerically relevant) together with
/// the number of steps to use for the composite trapezoidal rule in the log
/// domain.  The integral is approximated as
///
/// ```text
/// \int_a^b x^{s-1} f(x) dx = \int_{\ln a}^{\ln b} e^{s t} f(e^{t}) dt
/// ```
///
/// Using the logarithmic domain provides stable behaviour for rapidly decaying
/// functions and mirrors the Hilbert space setting for `L^2((0, \infty), dx/x)`.
pub fn mellin_transform<F>(
    f: F,
    s: ComplexScalar,
    range: (Scalar, Scalar),
    steps: usize,
) -> MellinResult<ComplexScalar>
where
    F: Fn(Scalar) -> ComplexScalar,
{
    if steps < 2 {
        return Err(MellinError::InsufficientSamples);
    }
    let (log_a, log_b) = map_range_to_log(range)?;
    let h = (log_b - log_a) / steps as Scalar;

    let mut acc = ComplexScalar::new(0.0, 0.0);
    for i in 0..=steps {
        let weight = if i == 0 || i == steps { 0.5 } else { 1.0 };
        let t = log_a + h * i as Scalar;
        let x = t.exp();
        let kernel = (s * ComplexScalar::new(t, 0.0)).exp();
        let val = f(x);
        if !(val.re.is_finite() && val.im.is_finite()) {
            return Err(MellinError::NonFiniteFunctionValue { x });
        }
        acc += kernel * val * weight;
    }

    Ok(acc * ComplexScalar::new(h, 0.0))
}

/// Evaluate the Mellin transform from log-spaced samples.
///
/// The samples must correspond to the points `x_k = exp(log_start + k * log_step)`
/// for `k = 0..n-1`.  This layout makes the Mellin integral a standard
/// trapezoidal rule in the logarithmic domain, which aligns with the
/// Parseval identity on the Hilbert space `L^2((0, \infty), dx/x)`.
pub fn mellin_transform_log_samples(
    log_start: Scalar,
    log_step: Scalar,
    samples: &[ComplexScalar],
    s: ComplexScalar,
) -> MellinResult<ComplexScalar> {
    if !(log_step.is_finite() && log_step > 0.0) {
        return Err(MellinError::InvalidLogStep);
    }
    if samples.is_empty() {
        return Err(MellinError::EmptySamples);
    }
    if samples.len() < 2 {
        return Err(MellinError::InsufficientSamples);
    }

    let weights = trapezoidal_weights(samples.len())?;
    let (prefactor, z) = mellin_log_lattice_prefactor(log_start, log_step, s)?;
    for (idx, &sample) in samples.iter().enumerate() {
        if !(sample.re.is_finite() && sample.im.is_finite()) {
            return Err(MellinError::NonFiniteSample { index: idx });
        }
    }
    let series = weighted_z_transform(samples, &weights, z)?;
    Ok(prefactor * series)
}

/// Inner product associated with the Hilbert space `L^2((0, \infty), dx/x)`.
///
/// This is the natural domain for the Mellin transform where it acts as a
/// unitary operator (Plancherel theorem).  Numerically we again employ the log
/// domain so that the weight `dx/x` becomes the standard Lebesgue measure in the
/// `t` variable.
pub fn mellin_l2_inner_product<F, G>(
    f: F,
    g: G,
    range: (Scalar, Scalar),
    steps: usize,
) -> MellinResult<ComplexScalar>
where
    F: Fn(Scalar) -> ComplexScalar,
    G: Fn(Scalar) -> ComplexScalar,
{
    if steps < 2 {
        return Err(MellinError::InsufficientSamples);
    }
    let (log_a, log_b) = map_range_to_log(range)?;
    let h = (log_b - log_a) / steps as Scalar;

    let mut acc = ComplexScalar::new(0.0, 0.0);
    for i in 0..=steps {
        let weight = if i == 0 || i == steps { 0.5 } else { 1.0 };
        let t = log_a + h * i as Scalar;
        let x = t.exp();
        let val = f(x).conj() * g(x);
        if !(val.re.is_finite() && val.im.is_finite()) {
            return Err(MellinError::NonFiniteFunctionValue { x });
        }
        acc += val * weight;
    }

    Ok(acc * ComplexScalar::new(h, 0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zspace::mellin_transform_via_z;
    use libm::tgammaf;

    fn exp_decay(x: Scalar) -> ComplexScalar {
        ComplexScalar::new((-x).exp(), 0.0)
    }

    #[test]
    fn mellin_transform_matches_gamma_on_real_axis() {
        // For f(x) = exp(-x), the Mellin transform on the real axis is Gamma(s).
        let s = ComplexScalar::new(2.5, 0.0);
        let approx = mellin_transform(exp_decay, s, (1e-4, 40.0), 16_384).unwrap();
        let expected = tgammaf(2.5);
        assert!(
            (approx.re - expected).abs() < 1e-3,
            "approx={} expected={}",
            approx,
            expected
        );
        assert!(approx.im.abs() < 1e-3);
    }

    #[test]
    fn log_samples_agree_with_function_integration() {
        let s = ComplexScalar::new(1.3, 0.5);
        let log_start = -5.0f32;
        // sample from x in [exp(log_start), exp(log_start + log_step*(n-1))]
        let log_step = 0.005;
        let n = 2200;
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let t = log_start + log_step * i as f32;
            let x = t.exp();
            samples.push(exp_decay(x));
        }
        let discrete = mellin_transform_log_samples(log_start, log_step, &samples, s).unwrap();
        let continuous = mellin_transform(
            exp_decay,
            s,
            (
                (log_start).exp(),
                (log_start + log_step * (n - 1) as f32).exp(),
            ),
            n - 1,
        )
        .unwrap();
        let diff = (discrete - continuous).norm();
        assert!(
            diff < 5e-3,
            "diff={} discrete={} continuous={}",
            diff,
            discrete,
            continuous
        );
    }

    #[test]
    fn log_samples_match_z_bridge() {
        let s = ComplexScalar::new(0.8, -0.3);
        let log_start = -2.5f32;
        let log_step = 0.1f32;
        let n = 64;
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let t = log_start + log_step * i as f32;
            let x = t.exp();
            samples.push(exp_decay(x));
        }

        let via_direct = mellin_transform_log_samples(log_start, log_step, &samples, s).unwrap();
        let via_z = mellin_transform_via_z(log_start, log_step, &samples, s).unwrap();
        let diff = (via_direct - via_z).norm();
        assert!(
            diff < 1e-6,
            "diff={} direct={} via_z={}",
            diff,
            via_direct,
            via_z
        );
    }

    #[test]
    fn sample_log_uniform_matches_manual_sampling() {
        let log_start = -3.0f32;
        let log_step = 0.2f32;
        let len = 8usize;
        let samples = sample_log_uniform(log_start, log_step, len, exp_decay).unwrap();
        for (idx, sample) in samples.iter().enumerate() {
            let t = log_start + log_step * idx as f32;
            let x = t.exp();
            let expected = exp_decay(x);
            let diff = (*sample - expected).norm();
            assert!(diff < 1e-6, "idx={} diff={}", idx, diff);
        }
    }

    #[test]
    fn mellin_log_grid_matches_free_function_path() {
        let log_start = -4.0f32;
        let log_step = 0.05f32;
        let len = 256usize;
        let grid = MellinLogGrid::from_function(log_start, log_step, len, exp_decay).unwrap();
        let s = ComplexScalar::new(1.1, 0.2);
        let via_grid = grid.evaluate(s).unwrap();
        let via_function =
            mellin_transform_log_samples(log_start, log_step, grid.samples(), s).unwrap();
        let diff = (via_grid - via_function).norm();
        assert!(diff < 1e-6, "diff={}", diff);
    }

    #[test]
    fn mellin_log_grid_many_reuses_samples() {
        let log_start = -3.5f32;
        let log_step = 0.07f32;
        let len = 128usize;
        let grid = MellinLogGrid::from_function(log_start, log_step, len, exp_decay).unwrap();
        let s_values = vec![
            ComplexScalar::new(0.8, 0.3),
            ComplexScalar::new(1.2, -0.1),
            ComplexScalar::new(1.6, 0.4),
        ];
        let batch = grid.evaluate_many(&s_values).unwrap();
        assert_eq!(batch.len(), s_values.len());
        for (idx, &s) in s_values.iter().enumerate() {
            let single = grid.evaluate(s).unwrap();
            let diff = (batch[idx] - single).norm();
            assert!(diff < 1e-6, "idx={} diff={}", idx, diff);
        }
    }

    #[test]
    fn mellin_log_grid_mesh_matches_pointwise() {
        let log_start = -3.0f32;
        let log_step = 0.05f32;
        let len = 256usize;
        let samples = sample_log_uniform_exp_decay(log_start, log_step, len).unwrap();
        let grid = MellinLogGrid::new(log_start, log_step, samples).unwrap();

        let real_values = vec![0.8f32, 1.3];
        let imag_values = vec![-0.5f32, 0.0, 0.4];
        let mesh = grid.evaluate_mesh(&real_values, &imag_values).unwrap();

        assert_eq!(mesh.len(), real_values.len() * imag_values.len());

        for (ri, &real) in real_values.iter().enumerate() {
            for (ii, &imag) in imag_values.iter().enumerate() {
                let idx = ri * imag_values.len() + ii;
                let s = ComplexScalar::new(real, imag);
                let single = grid.evaluate(s).unwrap();
                let diff = (mesh[idx] - single).norm();
                assert!(diff < 1e-5, "idx={} diff={}", idx, diff);
            }
        }
    }

    #[test]
    fn hilbert_inner_product_positive_definite() {
        let range = (1e-4, 60.0);
        let ip = mellin_l2_inner_product(exp_decay, exp_decay, range, 8_192).unwrap();
        assert!(ip.im.abs() < 1e-4);
        assert!(ip.re > 0.0);

        // Symmetry check: <f,g> = conj(<g,f>)
        let g = |x: Scalar| ComplexScalar::new(x.powf(0.5) * (-x).exp(), 0.0);
        let fg = mellin_l2_inner_product(exp_decay, g, range, 8_192).unwrap();
        let gf = mellin_l2_inner_product(g, exp_decay, range, 8_192).unwrap();
        let diff = (fg - gf.conj()).norm();
        assert!(diff < 1e-4, "diff={} fg={} gf={} ", diff, fg, gf);
    }

    #[test]
    fn mellin_log_grid_vertical_line_matches_batch() {
        let log_start = -4.0f32;
        let log_step = 0.05f32;
        let len = 160usize;
        let grid = MellinLogGrid::from_function(log_start, log_step, len, exp_decay).unwrap();

        let real = 1.1f32;
        let imag_values = [-1.0f32, -0.25, 0.0, 0.75, 1.3];
        let vertical = grid.evaluate_vertical_line(real, &imag_values).unwrap();
        let s_values: Vec<ComplexScalar> = imag_values
            .iter()
            .map(|&im| ComplexScalar::new(real, im))
            .collect();
        let batch = grid.evaluate_many(&s_values).unwrap();

        assert_eq!(vertical.len(), batch.len());
        for (idx, (lhs, rhs)) in vertical.iter().zip(batch.iter()).enumerate() {
            let diff = (*lhs - *rhs).norm();
            assert!(diff < 1e-6, "idx={} diff={}", idx, diff);
        }
    }

    #[test]
    fn mellin_log_grid_weighted_series_matches_api() {
        let log_start = -3.0f32;
        let log_step = 0.08f32;
        let len = 120usize;
        let grid = MellinLogGrid::from_function(log_start, log_step, len, exp_decay).unwrap();
        let weighted = grid.weighted_series().unwrap();

        let s = ComplexScalar::new(0.9, 0.4);
        let single_weighted = grid.evaluate_with_series(s, &weighted).unwrap();
        let single_api = grid.evaluate(s).unwrap();
        assert!((single_weighted - single_api).norm() < 1e-6);

        let s_values = vec![
            ComplexScalar::new(0.7, -0.3),
            ComplexScalar::new(1.0, 0.1),
            ComplexScalar::new(1.2, 0.6),
        ];
        let many_weighted = grid
            .evaluate_many_with_series(&s_values, &weighted)
            .unwrap();
        let many_api = grid.evaluate_many(&s_values).unwrap();
        for (idx, (lhs, rhs)) in many_weighted.iter().zip(many_api.iter()).enumerate() {
            let diff = (*lhs - *rhs).norm();
            assert!(diff < 1e-6, "idx={} diff={}", idx, diff);
        }
    }

    #[test]
    fn mellin_log_grid_hilbert_inner_product_matches_closed_form() {
        let log_start = -1.0f32;
        let log_step = 0.01f32;
        let len = 201usize; // covers [-1, 1] in the log domain
        let p = 0.3f32;
        let q = -0.1f32;

        let f_grid = MellinLogGrid::from_function(log_start, log_step, len, |x| {
            ComplexScalar::new(x.powf(p), 0.0)
        })
        .unwrap();
        let g_grid = MellinLogGrid::from_function(log_start, log_step, len, |x| {
            ComplexScalar::new(x.powf(q), 0.0)
        })
        .unwrap();

        let ip = f_grid.hilbert_inner_product(&g_grid).unwrap();
        let (a, b) = f_grid.support();
        let exponent = p + q;
        let expected = (b.powf(exponent) - a.powf(exponent)) / exponent;
        let diff = (ip - ComplexScalar::new(expected, 0.0)).norm();
        assert!(diff < 5e-3, "diff={} ip={} expected={}", diff, ip, expected);

        let norm = f_grid.hilbert_norm().unwrap();
        let norm_sq = (b.powf(2.0 * p) - a.powf(2.0 * p)) / (2.0 * p);
        let expected_norm = norm_sq.sqrt();
        assert!(
            (norm - expected_norm).abs() < 5e-3,
            "norm={} expected={}",
            norm,
            expected_norm
        );
    }

    #[test]
    fn sample_log_uniform_exp_decay_scaled_matches_rate_one() {
        let log_start = -3.0f32;
        let log_step = 0.1f32;
        let len = 32usize;
        let base = sample_log_uniform_exp_decay(log_start, log_step, len).unwrap();
        let scaled = sample_log_uniform_exp_decay_scaled(log_start, log_step, len, 1.0).unwrap();
        assert_eq!(base.len(), scaled.len());
        for (idx, (lhs, rhs)) in base.iter().zip(scaled.iter()).enumerate() {
            let diff = (*lhs - *rhs).norm();
            assert!(diff < 1e-6, "idx={} diff={}", idx, diff);
        }
    }

    #[test]
    fn mellin_eval_plan_mesh_matches_grid_evaluate_mesh() {
        let log_start = -3.0f32;
        let log_step = 0.05f32;
        let len = 256usize;
        let grid = MellinLogGrid::from_function(log_start, log_step, len, exp_decay).unwrap();
        let weighted = grid.weighted_series().unwrap();

        let real_values = vec![0.8f32, 1.3];
        let imag_values = vec![-0.5f32, 0.0, 0.4];
        let plan = MellinEvalPlan::mesh(log_start, log_step, &real_values, &imag_values).unwrap();
        assert_eq!(plan.shape(), (real_values.len(), imag_values.len()));

        let via_plan = plan.evaluate(&weighted).unwrap();
        let via_grid = grid.evaluate_mesh(&real_values, &imag_values).unwrap();
        assert_eq!(via_plan.len(), via_grid.len());
        for (idx, (lhs, rhs)) in via_plan.iter().zip(via_grid.iter()).enumerate() {
            let diff = (*lhs - *rhs).norm();
            assert!(diff < 1e-6, "idx={} diff={}", idx, diff);
        }
    }

    #[test]
    fn mellin_log_grid_mesh_magnitude_matches_complex_norm() {
        let log_start = -3.0f32;
        let log_step = 0.05f32;
        let len = 256usize;
        let grid = MellinLogGrid::from_function(log_start, log_step, len, exp_decay).unwrap();

        let real_values = vec![0.8f32, 1.3];
        let imag_values = vec![-0.5f32, 0.0, 0.4];
        let mags = grid.evaluate_mesh_magnitude(&real_values, &imag_values).unwrap();
        let complex = grid.evaluate_mesh(&real_values, &imag_values).unwrap();
        assert_eq!(mags.len(), complex.len());
        for (idx, (mag, value)) in mags.iter().zip(complex.iter()).enumerate() {
            let diff = (*mag - value.norm()).abs();
            assert!(diff < 1e-6, "idx={} diff={}", idx, diff);
        }
    }
}
