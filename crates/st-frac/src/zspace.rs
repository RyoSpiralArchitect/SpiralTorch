// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg_attr(
    not(test),
    deny(clippy::expect_used, clippy::panic, clippy::unwrap_used)
)]

use crate::mellin_types::{ComplexScalar, MellinError, MellinResult, Scalar, ZSpaceError};
use std::f32::consts::PI;

pub type ZSpaceResult<T> = Result<T, ZSpaceError>;

#[derive(Clone, Copy)]
struct ValidatedSeries<'a> {
    coefficients: &'a [ComplexScalar],
    prefix: &'a [ComplexScalar],
    head: ComplexScalar,
    tail: ComplexScalar,
}

impl<'a> ValidatedSeries<'a> {
    fn new(weighted: &'a [ComplexScalar]) -> MellinResult<Self> {
        let Some((tail, prefix)) = weighted.split_last() else {
            return Err(ZSpaceError::EmptySeries.into());
        };
        for (index, coefficient) in weighted.iter().enumerate() {
            if !complex_is_finite(*coefficient) {
                return Err(MellinError::NonFiniteSample { index });
            }
        }
        let head = prefix.first().copied().unwrap_or(*tail);
        Ok(Self {
            coefficients: weighted,
            prefix,
            head,
            tail: *tail,
        })
    }

    fn direct(self, z: ComplexScalar) -> ComplexScalar {
        let mut value = self.tail;
        for coefficient in self.prefix.iter().rev() {
            value = *coefficient + z * value;
        }
        value
    }

    fn direct_with_derivative(self, z: ComplexScalar) -> (ComplexScalar, ComplexScalar) {
        let mut value = self.tail;
        let mut derivative = ComplexScalar::new(0.0, 0.0);
        for coefficient in self.prefix.iter().rev() {
            derivative = derivative * z + value;
            value = *coefficient + z * value;
        }
        (value, derivative)
    }

    fn inverse(self, z: ComplexScalar) -> ComplexScalar {
        if self.coefficients.len() == 1 || z == ComplexScalar::new(0.0, 0.0) {
            return self.head;
        }

        let inverse_z = ComplexScalar::new(1.0, 0.0) / z;
        let mut value = self.head;
        for coefficient in self.coefficients.iter().skip(1) {
            value = *coefficient + inverse_z * value;
        }

        let mut z_power = ComplexScalar::new(1.0, 0.0);
        for _ in 0..self.coefficients.len() - 1 {
            z_power *= z;
        }
        z_power * value
    }

    fn inverse_with_derivative(self, z: ComplexScalar) -> (ComplexScalar, ComplexScalar) {
        if self.coefficients.len() == 1 {
            return (self.head, ComplexScalar::new(0.0, 0.0));
        }
        if z == ComplexScalar::new(0.0, 0.0) {
            let derivative = self
                .coefficients
                .get(1)
                .copied()
                .unwrap_or_else(|| ComplexScalar::new(0.0, 0.0));
            return (self.head, derivative);
        }

        let inverse_z = ComplexScalar::new(1.0, 0.0) / z;
        let mut value = self.head;
        let mut derivative_inverse = ComplexScalar::new(0.0, 0.0);
        for coefficient in self.coefficients.iter().skip(1) {
            derivative_inverse = derivative_inverse * inverse_z + value;
            value = *coefficient + inverse_z * value;
        }

        let mut z_power = ComplexScalar::new(1.0, 0.0);
        for _ in 0..self.coefficients.len() - 1 {
            z_power *= z;
        }
        let transformed = z_power * value;
        let degree = ComplexScalar::new((self.coefficients.len() - 1) as Scalar, 0.0);
        let derivative = z_power * inverse_z * (degree * value - inverse_z * derivative_inverse);
        (transformed, derivative)
    }
}

#[inline]
fn complex_is_finite(value: ComplexScalar) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

#[inline]
fn validate_z(z: ComplexScalar) -> MellinResult<()> {
    if complex_is_finite(z) {
        Ok(())
    } else {
        Err(MellinError::NonFiniteZ { re: z.re, im: z.im })
    }
}

#[inline]
pub(crate) fn finite_output(
    value: ComplexScalar,
    quantity: &'static str,
    index: usize,
) -> MellinResult<ComplexScalar> {
    if complex_is_finite(value) {
        Ok(value)
    } else {
        Err(MellinError::NonFiniteOutput {
            quantity,
            index,
            re: value.re,
            im: value.im,
        })
    }
}

#[inline]
pub(crate) fn finite_scalar_output(
    value: Scalar,
    quantity: &'static str,
    index: usize,
) -> MellinResult<Scalar> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(MellinError::NonFiniteScalarOutput {
            quantity,
            index,
            value,
        })
    }
}

#[inline]
fn finite_outputs(
    value: ComplexScalar,
    derivative: ComplexScalar,
    index: usize,
) -> MellinResult<(ComplexScalar, ComplexScalar)> {
    Ok((
        finite_output(value, "series value", index)?,
        finite_output(derivative, "series derivative", index)?,
    ))
}

/// Window functions applied to trapezoidal weights on a log lattice.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LogLatticeWindow {
    /// No tapering (rectangular window).
    #[default]
    Rectangular,
    /// Hann taper for endpoint leakage suppression.
    Hann,
    /// Tukey (tapered cosine) window with a fixed alpha=0.5.
    Tukey,
    /// Blackman window.
    Blackman,
}

impl LogLatticeWindow {
    #[inline]
    pub fn apply(self, weights: &mut [Scalar]) {
        match self {
            LogLatticeWindow::Rectangular => {}
            LogLatticeWindow::Hann => {
                if weights.len() <= 2 {
                    return;
                }
                let denom = (weights.len() - 1) as Scalar;
                for (idx, w) in weights.iter_mut().enumerate() {
                    let phase = 2.0 * PI * idx as Scalar / denom;
                    let hann = 0.5 * (1.0 - phase.cos());
                    *w *= hann;
                }
            }
            LogLatticeWindow::Tukey => {
                if weights.len() <= 2 {
                    return;
                }
                let alpha: Scalar = 0.5;
                if alpha <= 0.0 {
                    return;
                }
                if alpha >= 1.0 {
                    LogLatticeWindow::Hann.apply(weights);
                    return;
                }
                let denom = (weights.len() - 1) as Scalar;
                let edge = alpha / 2.0;
                for (idx, w) in weights.iter_mut().enumerate() {
                    let x = idx as Scalar / denom;
                    let tukey = if x < edge {
                        0.5 * (1.0 + (PI * (2.0 * x / alpha - 1.0)).cos())
                    } else if x <= 1.0 - edge {
                        1.0
                    } else {
                        0.5 * (1.0 + (PI * (2.0 * x / alpha - 2.0 / alpha + 1.0)).cos())
                    };
                    *w *= tukey;
                }
            }
            LogLatticeWindow::Blackman => {
                if weights.len() <= 2 {
                    return;
                }
                let denom = (weights.len() - 1) as Scalar;
                for (idx, w) in weights.iter_mut().enumerate() {
                    let phase = 2.0 * PI * idx as Scalar / denom;
                    let blackman = 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos();
                    *w *= blackman;
                }
            }
        }
    }
}

/// Compute trapezoidal-rule weights for a log-uniform lattice.
#[inline]
pub fn trapezoidal_weights(len: usize) -> ZSpaceResult<Vec<Scalar>> {
    if len == 0 {
        return Err(ZSpaceError::EmptySamples);
    }
    if len == 1 {
        return Ok(vec![1.0]);
    }
    let mut weights = vec![1.0; len];
    if let Some(first) = weights.first_mut() {
        *first = 0.5;
    }
    if let Some(last) = weights.last_mut() {
        *last = 0.5;
    }
    Ok(weights)
}

/// Compute trapezoidal-rule weights and optionally apply a window + sum-preserving renormalisation.
///
/// If `preserve_sum=true`, the windowed weights are scaled so that their sum matches the
/// original trapezoidal sum (preserving the constant-function integral in the log domain).
#[inline]
pub fn trapezoidal_weights_windowed(
    len: usize,
    window: LogLatticeWindow,
    preserve_sum: bool,
) -> ZSpaceResult<Vec<Scalar>> {
    let mut weights = trapezoidal_weights(len)?;
    if window == LogLatticeWindow::Rectangular && !preserve_sum {
        return Ok(weights);
    }

    let base_sum: Scalar = weights.iter().sum();
    window.apply(&mut weights);
    if preserve_sum {
        let sum: Scalar = weights.iter().sum();
        if sum > Scalar::EPSILON && base_sum.is_finite() {
            let scale = base_sum / sum;
            for w in weights.iter_mut() {
                *w *= scale;
            }
        }
    }
    Ok(weights)
}

/// Map Mellin abscissa data to the Z-plane representation.
#[inline]
pub fn mellin_log_lattice_prefactor(
    log_start: Scalar,
    log_step: Scalar,
    s: ComplexScalar,
) -> MellinResult<(ComplexScalar, ComplexScalar)> {
    if !(log_step.is_finite() && log_step > 0.0) {
        return Err(MellinError::InvalidLogStep);
    }
    if !log_start.is_finite() {
        return Err(MellinError::InvalidLogStart);
    }
    if !complex_is_finite(s) {
        return Err(MellinError::NonFiniteAbscissa { re: s.re, im: s.im });
    }

    let step = ComplexScalar::new(log_step, 0.0);
    let start = ComplexScalar::new(log_start, 0.0);
    let prefactor = finite_output((s * start).exp() * step, "Mellin prefactor", 0)?;
    let z = finite_output((s * step).exp(), "Z-plane mapping", 0)?;
    Ok((prefactor, z))
}

/// Evaluate a weighted Z-transform at the point `z`.
pub fn weighted_z_transform(
    samples: &[ComplexScalar],
    weights: &[Scalar],
    z: ComplexScalar,
) -> MellinResult<ComplexScalar> {
    let weighted = prepare_weighted_series(samples, weights)?;
    evaluate_weighted_series(&weighted, z)
}

/// Evaluate a weighted Z-transform and its derivative with respect to `z`.
///
/// Returns `(value, d_value_dz)`.
pub fn weighted_z_transform_with_derivative(
    samples: &[ComplexScalar],
    weights: &[Scalar],
    z: ComplexScalar,
) -> MellinResult<(ComplexScalar, ComplexScalar)> {
    let weighted = prepare_weighted_series(samples, weights)?;
    evaluate_weighted_series_with_derivative(&weighted, z)
}

/// Evaluate the weighted Z-transform at multiple `z` points.
pub fn weighted_z_transform_many(
    samples: &[ComplexScalar],
    weights: &[Scalar],
    z_values: &[ComplexScalar],
) -> MellinResult<Vec<ComplexScalar>> {
    let weighted = prepare_weighted_series(samples, weights)?;
    evaluate_weighted_series_many(&weighted, z_values)
}

/// Evaluate the weighted Z-transform and `d/dz` derivative at multiple `z` points.
///
/// Returns `(values, d_values_dz)`.
pub fn weighted_z_transform_many_with_derivative(
    samples: &[ComplexScalar],
    weights: &[Scalar],
    z_values: &[ComplexScalar],
) -> MellinResult<(Vec<ComplexScalar>, Vec<ComplexScalar>)> {
    let weighted = prepare_weighted_series(samples, weights)?;
    evaluate_weighted_series_many_with_derivative(&weighted, z_values)
}

/// Precompute the weighted power-series coefficients for Z-plane evaluations.
#[inline]
pub fn prepare_weighted_series(
    samples: &[ComplexScalar],
    weights: &[Scalar],
) -> ZSpaceResult<Vec<ComplexScalar>> {
    if samples.is_empty() {
        return Err(ZSpaceError::EmptySamples);
    }
    if samples.len() != weights.len() {
        return Err(ZSpaceError::WeightLengthMismatch {
            samples: samples.len(),
            weights: weights.len(),
        });
    }

    for (index, (sample, &weight)) in samples.iter().zip(weights.iter()).enumerate() {
        if !weight.is_finite() {
            return Err(ZSpaceError::NonFiniteWeight {
                index,
                value: weight,
            });
        }
        if !complex_is_finite(*sample) {
            return Err(ZSpaceError::NonFiniteSampleCoeff { index });
        }
    }

    let mut weighted = Vec::with_capacity(samples.len());
    for (index, (sample, &weight)) in samples.iter().zip(weights.iter()).enumerate() {
        let coefficient = *sample * ComplexScalar::new(weight, 0.0);
        if !complex_is_finite(coefficient) {
            return Err(ZSpaceError::NonFiniteWeightedCoeff {
                index,
                re: coefficient.re,
                im: coefficient.im,
            });
        }
        weighted.push(coefficient);
    }
    Ok(weighted)
}

/// Evaluate a weighted power series using the precomputed coefficients.
pub fn evaluate_weighted_series(
    weighted: &[ComplexScalar],
    z: ComplexScalar,
) -> MellinResult<ComplexScalar> {
    let series = ValidatedSeries::new(weighted)?;
    validate_z(z)?;
    finite_output(series.direct(z), "series value", 0)
}

/// Evaluate a weighted power series and its derivative with respect to `z`.
///
/// Returns `(value, d_value_dz)`.
pub fn evaluate_weighted_series_with_derivative(
    weighted: &[ComplexScalar],
    z: ComplexScalar,
) -> MellinResult<(ComplexScalar, ComplexScalar)> {
    let series = ValidatedSeries::new(weighted)?;
    validate_z(z)?;
    let (value, derivative) = series.direct_with_derivative(z);
    finite_outputs(value, derivative, 0)
}

/// Evaluate the weighted power series at multiple `z` values.
pub fn evaluate_weighted_series_many(
    weighted: &[ComplexScalar],
    z_values: &[ComplexScalar],
) -> MellinResult<Vec<ComplexScalar>> {
    if weighted.is_empty() {
        return Err(ZSpaceError::EmptySeries.into());
    }
    if z_values.is_empty() {
        return Err(ZSpaceError::EmptyZValues.into());
    }
    let series = ValidatedSeries::new(weighted)?;

    let mut out = Vec::with_capacity(z_values.len());
    for (index, &z) in z_values.iter().enumerate() {
        validate_z(z)?;
        out.push(finite_output(series.direct(z), "series value", index)?);
    }
    Ok(out)
}

/// Evaluate the weighted power series and `d/dz` derivative at multiple `z` values.
///
/// Returns `(values, d_values_dz)`.
pub fn evaluate_weighted_series_many_with_derivative(
    weighted: &[ComplexScalar],
    z_values: &[ComplexScalar],
) -> MellinResult<(Vec<ComplexScalar>, Vec<ComplexScalar>)> {
    if weighted.is_empty() {
        return Err(ZSpaceError::EmptySeries.into());
    }
    if z_values.is_empty() {
        return Err(ZSpaceError::EmptyZValues.into());
    }
    let series = ValidatedSeries::new(weighted)?;

    let mut values = Vec::with_capacity(z_values.len());
    let mut derivatives = Vec::with_capacity(z_values.len());
    for (index, &z) in z_values.iter().enumerate() {
        validate_z(z)?;
        let (value, derivative) = series.direct_with_derivative(z);
        let (value, derivative) = finite_outputs(value, derivative, index)?;
        values.push(value);
        derivatives.push(derivative);
    }
    Ok((values, derivatives))
}

/// Evaluate a weighted power series using the inverse-polynomial representation.
///
/// This is numerically preferable for `|z| > 1` because it evaluates
/// `Q(1/z)` with `|1/z| < 1` and then rescales by `z^{n-1}`.
pub fn evaluate_weighted_series_inverse(
    weighted: &[ComplexScalar],
    z: ComplexScalar,
) -> MellinResult<ComplexScalar> {
    let series = ValidatedSeries::new(weighted)?;
    validate_z(z)?;
    finite_output(series.inverse(z), "series value", 0)
}

/// Evaluate a weighted power series and its `d/dz` derivative using the inverse-polynomial representation.
///
/// Returns `(value, d_value_dz)`.
pub fn evaluate_weighted_series_with_derivative_inverse(
    weighted: &[ComplexScalar],
    z: ComplexScalar,
) -> MellinResult<(ComplexScalar, ComplexScalar)> {
    let series = ValidatedSeries::new(weighted)?;
    validate_z(z)?;
    let (value, derivative) = series.inverse_with_derivative(z);
    finite_outputs(value, derivative, 0)
}

/// Evaluate a weighted power series using a stable strategy:
/// direct Horner for `|z| <= 1`, inverse representation for `|z| > 1`.
pub fn evaluate_weighted_series_stable(
    weighted: &[ComplexScalar],
    z: ComplexScalar,
) -> MellinResult<ComplexScalar> {
    let series = ValidatedSeries::new(weighted)?;
    validate_z(z)?;
    let value = if z.norm_sqr() <= 1.0 {
        series.direct(z)
    } else {
        series.inverse(z)
    };
    finite_output(value, "series value", 0)
}

/// Evaluate a weighted power series and its derivative using the stable strategy.
///
/// Returns `(value, d_value_dz)`.
pub fn evaluate_weighted_series_with_derivative_stable(
    weighted: &[ComplexScalar],
    z: ComplexScalar,
) -> MellinResult<(ComplexScalar, ComplexScalar)> {
    let series = ValidatedSeries::new(weighted)?;
    validate_z(z)?;
    let (value, derivative) = if z.norm_sqr() <= 1.0 {
        series.direct_with_derivative(z)
    } else {
        series.inverse_with_derivative(z)
    };
    finite_outputs(value, derivative, 0)
}

/// Evaluate the weighted power series at multiple `z` values using the stable strategy.
pub fn evaluate_weighted_series_many_stable(
    weighted: &[ComplexScalar],
    z_values: &[ComplexScalar],
) -> MellinResult<Vec<ComplexScalar>> {
    if weighted.is_empty() {
        return Err(ZSpaceError::EmptySeries.into());
    }
    if z_values.is_empty() {
        return Err(ZSpaceError::EmptyZValues.into());
    }
    let series = ValidatedSeries::new(weighted)?;

    let mut out = Vec::with_capacity(z_values.len());
    for (index, &z) in z_values.iter().enumerate() {
        validate_z(z)?;
        let value = if z.norm_sqr() <= 1.0 {
            series.direct(z)
        } else {
            series.inverse(z)
        };
        out.push(finite_output(value, "series value", index)?);
    }
    Ok(out)
}

/// Evaluate the weighted power series and `d/dz` derivative at multiple `z` values using the stable strategy.
///
/// Returns `(values, d_values_dz)`.
pub fn evaluate_weighted_series_many_with_derivative_stable(
    weighted: &[ComplexScalar],
    z_values: &[ComplexScalar],
) -> MellinResult<(Vec<ComplexScalar>, Vec<ComplexScalar>)> {
    if weighted.is_empty() {
        return Err(ZSpaceError::EmptySeries.into());
    }
    if z_values.is_empty() {
        return Err(ZSpaceError::EmptyZValues.into());
    }
    let series = ValidatedSeries::new(weighted)?;

    let mut values = Vec::with_capacity(z_values.len());
    let mut derivatives = Vec::with_capacity(z_values.len());
    for (index, &z) in z_values.iter().enumerate() {
        validate_z(z)?;
        let (value, derivative) = if z.norm_sqr() <= 1.0 {
            series.direct_with_derivative(z)
        } else {
            series.inverse_with_derivative(z)
        };
        let (value, derivative) = finite_outputs(value, derivative, index)?;
        values.push(value);
        derivatives.push(derivative);
    }

    Ok((values, derivatives))
}

/// Evaluate the Mellin transform on a log-uniform grid via Z-plane series.
pub fn mellin_transform_via_z(
    log_start: Scalar,
    log_step: Scalar,
    samples: &[ComplexScalar],
    s: ComplexScalar,
) -> MellinResult<ComplexScalar> {
    let weights = trapezoidal_weights(samples.len())?;
    let (prefactor, z) = mellin_log_lattice_prefactor(log_start, log_step, s)?;
    let series = weighted_z_transform(samples, &weights, z)?;
    finite_output(prefactor * series, "Mellin value", 0)
}

/// Evaluate the Mellin transform via Z-plane series and return `d/ds`.
///
/// Returns `(M(s), dM/ds)`.
pub fn mellin_transform_via_z_with_derivative(
    log_start: Scalar,
    log_step: Scalar,
    samples: &[ComplexScalar],
    s: ComplexScalar,
) -> MellinResult<(ComplexScalar, ComplexScalar)> {
    let weights = trapezoidal_weights(samples.len())?;
    let (prefactor, z) = mellin_log_lattice_prefactor(log_start, log_step, s)?;

    let weighted = prepare_weighted_series(samples, &weights)?;
    let (series, d_series_dz) = evaluate_weighted_series_with_derivative(&weighted, z)?;

    let start = ComplexScalar::new(log_start, 0.0);
    let step = ComplexScalar::new(log_step, 0.0);

    let value = finite_output(prefactor * series, "Mellin value", 0)?;
    let d_value_ds = finite_output(
        prefactor * (start * series + step * z * d_series_dz),
        "Mellin derivative",
        0,
    )?;
    Ok((value, d_value_ds))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;

    fn assert_non_finite_output<T: Debug>(
        result: MellinResult<T>,
        expected_quantity: &'static str,
        expected_index: usize,
    ) {
        match result {
            Err(MellinError::NonFiniteOutput {
                quantity, index, ..
            }) => {
                assert_eq!(quantity, expected_quantity);
                assert_eq!(index, expected_index);
            }
            other => panic!("expected NonFiniteOutput, got {other:?}"),
        }
    }

    #[test]
    fn trapezoidal_weights_match_manual() {
        let single = trapezoidal_weights(1).unwrap();
        assert_eq!(single, vec![1.0]);

        let weights = trapezoidal_weights(4).unwrap();
        assert_eq!(weights, vec![0.5, 1.0, 1.0, 0.5]);
    }

    #[test]
    fn weighted_z_transform_reduces_to_polynomial() {
        let samples = [
            ComplexScalar::new(1.0, 0.0),
            ComplexScalar::new(2.0, 0.0),
            ComplexScalar::new(-1.0, 0.0),
        ];
        let weights = vec![1.0, 1.0, 1.0];
        let z = ComplexScalar::new(0.5, 0.5);
        let value = weighted_z_transform(&samples, &weights, z).unwrap();
        let expected = samples[0] + samples[1] * z + samples[2] * z * z;
        let diff = (value - expected).norm();
        assert!(diff < 1e-6, "diff={}", diff);
    }

    #[test]
    fn weighted_series_derivative_matches_manual_polynomial() {
        let weighted = [
            ComplexScalar::new(1.0, -0.5),
            ComplexScalar::new(2.0, 0.25),
            ComplexScalar::new(-1.0, 0.75),
        ];
        let z = ComplexScalar::new(0.3, -0.4);

        let (value, deriv) = evaluate_weighted_series_with_derivative(&weighted, z).unwrap();
        let expected_value = weighted[0] + weighted[1] * z + weighted[2] * z * z;
        let expected_deriv = weighted[1] + ComplexScalar::new(2.0, 0.0) * weighted[2] * z;

        assert!(
            (value - expected_value).norm() < 1e-6,
            "value diff={}",
            (value - expected_value).norm()
        );
        assert!(
            (deriv - expected_deriv).norm() < 1e-6,
            "deriv diff={}",
            (deriv - expected_deriv).norm()
        );
    }

    #[test]
    fn weighted_series_derivative_len1_is_zero() {
        let weighted = [ComplexScalar::new(1.0, 2.0)];
        let z = ComplexScalar::new(0.7, -0.1);
        let (value, deriv) = evaluate_weighted_series_with_derivative(&weighted, z).unwrap();
        assert!((value - weighted[0]).norm() < 1e-6);
        assert!(deriv.norm() < 1e-6);
    }

    #[test]
    fn inverse_derivative_at_zero_uses_first_two_coefficients() {
        let weighted = [
            ComplexScalar::new(1.0, 2.0),
            ComplexScalar::new(3.0, 4.0),
            ComplexScalar::new(5.0, 6.0),
        ];
        let (value, derivative) = evaluate_weighted_series_with_derivative_inverse(
            &weighted,
            ComplexScalar::new(0.0, 0.0),
        )
        .unwrap();
        assert_eq!(value, weighted[0]);
        assert_eq!(derivative, weighted[1]);
    }

    #[test]
    fn trapezoidal_weights_windowed_preserve_sum_matches_expected() {
        fn assert_weights_close(actual: &[Scalar], expected: &[Scalar], tol: Scalar) {
            assert_eq!(actual.len(), expected.len());
            for (idx, (&lhs, &rhs)) in actual.iter().zip(expected.iter()).enumerate() {
                let diff = (lhs - rhs).abs();
                assert!(
                    diff < tol,
                    "idx={} lhs={} rhs={} diff={} tol={}",
                    idx,
                    lhs,
                    rhs,
                    diff,
                    tol
                );
            }
        }

        let base = trapezoidal_weights(4).unwrap();
        assert_eq!(base, vec![0.5, 1.0, 1.0, 0.5]);

        let rect = trapezoidal_weights_windowed(4, LogLatticeWindow::Rectangular, true).unwrap();
        assert_eq!(rect, base);

        let hann_no_preserve =
            trapezoidal_weights_windowed(4, LogLatticeWindow::Hann, false).unwrap();
        assert_weights_close(&hann_no_preserve, &[0.0, 0.75, 0.75, 0.0], 1e-6);

        let hann_preserve = trapezoidal_weights_windowed(4, LogLatticeWindow::Hann, true).unwrap();
        assert_weights_close(&hann_preserve, &[0.0, 1.5, 1.5, 0.0], 1e-6);

        let tukey_no_preserve =
            trapezoidal_weights_windowed(5, LogLatticeWindow::Tukey, false).unwrap();
        assert_weights_close(&tukey_no_preserve, &[0.0, 1.0, 1.0, 1.0, 0.0], 1e-6);

        let tukey_preserve =
            trapezoidal_weights_windowed(5, LogLatticeWindow::Tukey, true).unwrap();
        assert_weights_close(
            &tukey_preserve,
            &[0.0, 4.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0, 0.0],
            1e-5,
        );

        let blackman_no_preserve =
            trapezoidal_weights_windowed(5, LogLatticeWindow::Blackman, false).unwrap();
        assert_weights_close(&blackman_no_preserve, &[0.0, 0.34, 1.0, 0.34, 0.0], 1e-5);

        let blackman_preserve =
            trapezoidal_weights_windowed(5, LogLatticeWindow::Blackman, true).unwrap();
        assert_weights_close(
            &blackman_preserve,
            &[0.0, 0.8095238, 2.3809524, 0.8095238, 0.0],
            1e-5,
        );
    }

    #[test]
    fn mellin_via_z_matches_direct_series() {
        let samples = vec![
            ComplexScalar::new(1.0, 0.0),
            ComplexScalar::new(0.5, 0.0),
            ComplexScalar::new(0.25, 0.0),
        ];
        let log_start = -2.0f32;
        let log_step = 0.5f32;
        let s = ComplexScalar::new(1.2, 0.7);

        let direct = {
            let mut acc = ComplexScalar::new(0.0, 0.0);
            for (idx, &sample) in samples.iter().enumerate() {
                let weight = if idx == 0 || idx + 1 == samples.len() {
                    0.5
                } else {
                    1.0
                };
                let t = log_start + log_step * idx as f32;
                let kernel = (s * ComplexScalar::new(t, 0.0)).exp();
                acc += sample * ComplexScalar::new(weight, 0.0) * kernel;
            }
            acc * ComplexScalar::new(log_step, 0.0)
        };

        let via_z = mellin_transform_via_z(log_start, log_step, &samples, s).unwrap();
        let diff = (direct - via_z).norm();
        assert!(diff < 1e-6, "diff={}", diff);
    }

    #[test]
    fn mellin_via_z_derivative_matches_direct_series() {
        let samples = vec![
            ComplexScalar::new(1.0, 0.0),
            ComplexScalar::new(0.5, 0.0),
            ComplexScalar::new(0.25, 0.0),
        ];
        let log_start = -2.0f32;
        let log_step = 0.5f32;
        let s = ComplexScalar::new(1.2, 0.7);

        let (direct, direct_deriv) = {
            let mut value_acc = ComplexScalar::new(0.0, 0.0);
            let mut deriv_acc = ComplexScalar::new(0.0, 0.0);
            for (idx, &sample) in samples.iter().enumerate() {
                let weight = if idx == 0 || idx + 1 == samples.len() {
                    0.5
                } else {
                    1.0
                };
                let t = log_start + log_step * idx as f32;
                let t_c = ComplexScalar::new(t, 0.0);
                let kernel = (s * t_c).exp();
                let term = sample * ComplexScalar::new(weight, 0.0) * kernel;
                value_acc += term;
                deriv_acc += term * t_c;
            }
            let step = ComplexScalar::new(log_step, 0.0);
            (value_acc * step, deriv_acc * step)
        };

        let (via_z, via_z_deriv) =
            mellin_transform_via_z_with_derivative(log_start, log_step, &samples, s).unwrap();

        assert!(
            (direct - via_z).norm() < 1e-6,
            "value diff={}",
            (direct - via_z).norm()
        );
        assert!(
            (direct_deriv - via_z_deriv).norm() < 1e-6,
            "deriv diff={}",
            (direct_deriv - via_z_deriv).norm()
        );
    }

    #[test]
    fn inverse_and_stable_series_match_direct_for_large_z() {
        let weighted = vec![
            ComplexScalar::new(1.0, -0.5),
            ComplexScalar::new(2.0, 0.25),
            ComplexScalar::new(-1.0, 0.75),
            ComplexScalar::new(0.25, 0.1),
        ];
        let z = ComplexScalar::new(1.2, 0.8); // |z| > 1

        let direct = evaluate_weighted_series(&weighted, z).unwrap();
        let inverse = evaluate_weighted_series_inverse(&weighted, z).unwrap();
        let stable = evaluate_weighted_series_stable(&weighted, z).unwrap();

        assert!(
            (direct - inverse).norm() < 1e-5,
            "diff={}",
            (direct - inverse).norm()
        );
        assert!(
            (direct - stable).norm() < 1e-5,
            "diff={}",
            (direct - stable).norm()
        );

        let (direct_v, direct_d) = evaluate_weighted_series_with_derivative(&weighted, z).unwrap();
        let (inv_v, inv_d) =
            evaluate_weighted_series_with_derivative_inverse(&weighted, z).unwrap();
        let (stable_v, stable_d) =
            evaluate_weighted_series_with_derivative_stable(&weighted, z).unwrap();
        assert!((direct_v - inv_v).norm() < 1e-5);
        assert!((direct_v - stable_v).norm() < 1e-5);
        assert!((direct_d - inv_d).norm() < 1e-5);
        assert!((direct_d - stable_d).norm() < 1e-5);
    }

    #[test]
    fn weighted_z_transform_many_matches_single_path() {
        let samples = vec![
            ComplexScalar::new(0.2, 0.1),
            ComplexScalar::new(-0.4, 0.3),
            ComplexScalar::new(0.1, -0.2),
        ];
        let weights = trapezoidal_weights(samples.len()).unwrap();
        let zs = vec![
            ComplexScalar::new(0.7, 0.0),
            ComplexScalar::new(0.4, -0.3),
            ComplexScalar::new(-0.2, 0.5),
        ];

        let batch = weighted_z_transform_many(&samples, &weights, &zs).unwrap();
        for (idx, &z) in zs.iter().enumerate() {
            let single = weighted_z_transform(&samples, &weights, z).unwrap();
            let diff = (batch[idx] - single).norm();
            assert!(diff < 1e-6, "idx={} diff={}", idx, diff);
        }
    }

    #[test]
    fn evaluate_rejects_nonfinite_coeff() {
        let weighted = vec![
            ComplexScalar::new(1.0, 0.0),
            ComplexScalar::new(f32::NAN, 0.0),
        ];
        let z = ComplexScalar::new(0.5, 0.1);
        assert!(evaluate_weighted_series(&weighted, z).is_err());
    }

    #[test]
    fn evaluate_rejects_nonfinite_z() {
        let weighted = vec![ComplexScalar::new(1.0, 0.0)];
        let z = ComplexScalar::new(f32::INFINITY, 0.0);
        assert!(evaluate_weighted_series(&weighted, z).is_err());
    }

    #[test]
    fn every_series_evaluator_rejects_empty_coefficients() {
        let z = ComplexScalar::new(1.0, 0.0);
        assert!(evaluate_weighted_series(&[], z).is_err());
        assert!(evaluate_weighted_series_with_derivative(&[], z).is_err());
        assert!(evaluate_weighted_series_inverse(&[], z).is_err());
        assert!(evaluate_weighted_series_with_derivative_inverse(&[], z).is_err());
        assert!(evaluate_weighted_series_stable(&[], z).is_err());
        assert!(evaluate_weighted_series_with_derivative_stable(&[], z).is_err());
        assert!(evaluate_weighted_series_many(&[], &[z]).is_err());
        assert!(evaluate_weighted_series_many_with_derivative(&[], &[z]).is_err());
        assert!(evaluate_weighted_series_many_stable(&[], &[z]).is_err());
        assert!(evaluate_weighted_series_many_with_derivative_stable(&[], &[z]).is_err());

        let invalid = [ComplexScalar::new(Scalar::NAN, 0.0)];
        assert!(matches!(
            evaluate_weighted_series_many(&invalid, &[]),
            Err(MellinError::ZSpace(ZSpaceError::EmptyZValues))
        ));
        assert!(matches!(
            evaluate_weighted_series_many(&[], &[]),
            Err(MellinError::ZSpace(ZSpaceError::EmptySeries))
        ));
    }

    #[test]
    fn prepare_rejects_nonfinite_inputs() {
        // Non-finite weight
        let samples = vec![ComplexScalar::new(1.0, 0.0), ComplexScalar::new(2.0, 0.0)];
        let bad_weights = vec![1.0, f32::NAN];
        assert!(prepare_weighted_series(&samples, &bad_weights).is_err());

        // Non-finite sample
        let bad_samples = vec![
            ComplexScalar::new(1.0, 0.0),
            ComplexScalar::new(f32::NAN, 0.0),
        ];
        let weights = vec![0.5, 0.5];
        assert!(prepare_weighted_series(&bad_samples, &weights).is_err());
    }

    #[test]
    fn prepare_rejects_finite_inputs_that_overflow_weighted_coefficient() {
        let error =
            prepare_weighted_series(&[ComplexScalar::new(Scalar::MAX, 0.0)], &[2.0]).unwrap_err();
        assert!(matches!(
            error,
            ZSpaceError::NonFiniteWeightedCoeff { index: 0, .. }
        ));

        let input_error = prepare_weighted_series(
            &[
                ComplexScalar::new(Scalar::MAX, 0.0),
                ComplexScalar::new(Scalar::NAN, 0.0),
            ],
            &[2.0, 1.0],
        )
        .unwrap_err();
        assert!(matches!(
            input_error,
            ZSpaceError::NonFiniteSampleCoeff { index: 1 }
        ));
    }

    #[test]
    fn evaluators_reject_finite_inputs_that_overflow_value() {
        let weighted = [
            ComplexScalar::new(Scalar::MAX, 0.0),
            ComplexScalar::new(Scalar::MAX, 0.0),
        ];
        let z = ComplexScalar::new(2.0, 0.0);

        assert_non_finite_output(evaluate_weighted_series(&weighted, z), "series value", 0);
        assert_non_finite_output(
            evaluate_weighted_series_inverse(&weighted, z),
            "series value",
            0,
        );
        assert_non_finite_output(
            evaluate_weighted_series_stable(&weighted, z),
            "series value",
            0,
        );
        assert_non_finite_output(
            evaluate_weighted_series_with_derivative(&weighted, z),
            "series value",
            0,
        );
        assert_non_finite_output(
            evaluate_weighted_series_with_derivative_inverse(&weighted, z),
            "series value",
            0,
        );
        assert_non_finite_output(
            evaluate_weighted_series_with_derivative_stable(&weighted, z),
            "series value",
            0,
        );
    }

    #[test]
    fn derivative_evaluator_rejects_overflow_when_value_cancels() {
        let weighted = [
            ComplexScalar::new(0.0, 0.0),
            ComplexScalar::new(Scalar::MAX, 0.0),
            ComplexScalar::new(0.0, 0.0),
            ComplexScalar::new(-Scalar::MAX, 0.0),
        ];

        assert_non_finite_output(
            evaluate_weighted_series_with_derivative(&weighted, ComplexScalar::new(1.0, 0.0)),
            "series derivative",
            0,
        );
    }

    #[test]
    fn batch_evaluators_report_the_overflowing_z_index() {
        let weighted = [
            ComplexScalar::new(1.0, 0.0),
            ComplexScalar::new(Scalar::MAX, 0.0),
        ];
        let z_values = [ComplexScalar::new(0.0, 0.0), ComplexScalar::new(2.0, 0.0)];

        assert_non_finite_output(
            evaluate_weighted_series_many(&weighted, &z_values),
            "series value",
            1,
        );
        assert_non_finite_output(
            evaluate_weighted_series_many_stable(&weighted, &z_values),
            "series value",
            1,
        );
        assert_non_finite_output(
            evaluate_weighted_series_many_with_derivative(&weighted, &z_values),
            "series value",
            1,
        );
        assert_non_finite_output(
            evaluate_weighted_series_many_with_derivative_stable(&weighted, &z_values),
            "series value",
            1,
        );
    }

    #[test]
    fn mellin_mapping_rejects_non_finite_abscissa_and_overflow() {
        assert!(matches!(
            mellin_log_lattice_prefactor(0.0, 1.0, ComplexScalar::new(Scalar::NAN, 0.0)),
            Err(MellinError::NonFiniteAbscissa { .. })
        ));
        assert_non_finite_output(
            mellin_log_lattice_prefactor(100.0, 1.0, ComplexScalar::new(1.0, 0.0)),
            "Mellin prefactor",
            0,
        );
        assert_non_finite_output(
            mellin_log_lattice_prefactor(0.0, 100.0, ComplexScalar::new(1.0, 0.0)),
            "Z-plane mapping",
            0,
        );
    }

    #[test]
    fn mellin_transform_rejects_finite_final_product_overflow() {
        assert_non_finite_output(
            mellin_transform_via_z(
                1.0,
                1.0,
                &[ComplexScalar::new(Scalar::MAX, 0.0)],
                ComplexScalar::new(1.0, 0.0),
            ),
            "Mellin value",
            0,
        );
    }
}
