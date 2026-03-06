// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::mellin_types::{ComplexScalar, MellinError, MellinResult, Scalar, ZSpaceError};

pub type ZSpaceResult<T> = Result<T, ZSpaceError>;

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

    let step = ComplexScalar::new(log_step, 0.0);
    let start = ComplexScalar::new(log_start, 0.0);
    let prefactor = (s * start).exp() * step;
    let z = (s * step).exp();
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

    // Robustness: validate finiteness of inputs before forming the weighted series
    for (i, (sample, &w)) in samples.iter().zip(weights.iter()).enumerate() {
        if !w.is_finite() {
            return Err(ZSpaceError::NonFiniteWeight { index: i, value: w });
        }
        if !sample.re.is_finite() || !sample.im.is_finite() {
            return Err(ZSpaceError::NonFiniteSampleCoeff { index: i });
        }
    }

    Ok(samples
        .iter()
        .zip(weights.iter())
        .map(|(sample, &w)| *sample * ComplexScalar::new(w, 0.0))
        .collect())
}

/// Evaluate a weighted power series using the precomputed coefficients.
pub fn evaluate_weighted_series(
    weighted: &[ComplexScalar],
    z: ComplexScalar,
) -> MellinResult<ComplexScalar> {
    if weighted.is_empty() {
        return Err(ZSpaceError::EmptySeries.into());
    }

    // Validate coefficients and z are finite
    for (i, coeff) in weighted.iter().enumerate() {
        if !coeff.re.is_finite() || !coeff.im.is_finite() {
            return Err(MellinError::NonFiniteSample { index: i });
        }
    }
    if !z.re.is_finite() || !z.im.is_finite() {
        return Err(MellinError::NonFiniteZ { re: z.re, im: z.im });
    }

    // Horner's method (stable & fewer multiplications):
    let mut acc = *weighted.last().unwrap();
    for coeff in weighted[..weighted.len() - 1].iter().rev() {
        acc = *coeff + z * acc;
    }
    Ok(acc)
}

/// Evaluate a weighted power series and its derivative with respect to `z`.
///
/// Returns `(value, d_value_dz)`.
pub fn evaluate_weighted_series_with_derivative(
    weighted: &[ComplexScalar],
    z: ComplexScalar,
) -> MellinResult<(ComplexScalar, ComplexScalar)> {
    if weighted.is_empty() {
        return Err(ZSpaceError::EmptySeries.into());
    }

    // Validate coefficients and z are finite
    for (i, coeff) in weighted.iter().enumerate() {
        if !coeff.re.is_finite() || !coeff.im.is_finite() {
            return Err(MellinError::NonFiniteSample { index: i });
        }
    }
    if !z.re.is_finite() || !z.im.is_finite() {
        return Err(MellinError::NonFiniteZ { re: z.re, im: z.im });
    }

    if weighted.len() == 1 {
        return Ok((weighted[0], ComplexScalar::new(0.0, 0.0)));
    }

    // Horner's method with derivative accumulation.
    // For P(z) = a_0 + a_1 z + ... + a_{n-1} z^{n-1}:
    //   p  = a_{n-1}
    //   dp = 0
    //   for k = n-2..0:
    //     dp = dp*z + p
    //     p  = p*z + a_k
    let mut p = *weighted.last().expect("weighted is non-empty");
    let mut dp = ComplexScalar::new(0.0, 0.0);
    for coeff in weighted[..weighted.len() - 1].iter().rev() {
        dp = dp * z + p;
        p = *coeff + z * p;
    }
    Ok((p, dp))
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

    // Validate coefficients once; `evaluate_weighted_series` would otherwise
    // perform this check per-`z`.
    for (i, coeff) in weighted.iter().enumerate() {
        if !coeff.re.is_finite() || !coeff.im.is_finite() {
            return Err(MellinError::NonFiniteSample { index: i });
        }
    }

    let mut out = Vec::with_capacity(z_values.len());
    let tail = *weighted.last().expect("checked non-empty");
    for &z in z_values {
        if !z.re.is_finite() || !z.im.is_finite() {
            return Err(MellinError::NonFiniteZ { re: z.re, im: z.im });
        }

        let mut acc = tail;
        for coeff in weighted[..weighted.len() - 1].iter().rev() {
            acc = *coeff + z * acc;
        }
        out.push(acc);
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

    for (i, coeff) in weighted.iter().enumerate() {
        if !coeff.re.is_finite() || !coeff.im.is_finite() {
            return Err(MellinError::NonFiniteSample { index: i });
        }
    }

    let mut values = Vec::with_capacity(z_values.len());
    let mut derivatives = Vec::with_capacity(z_values.len());
    if weighted.len() == 1 {
        let value = weighted[0];
        for &z in z_values {
            if !z.re.is_finite() || !z.im.is_finite() {
                return Err(MellinError::NonFiniteZ { re: z.re, im: z.im });
            }
            values.push(value);
            derivatives.push(ComplexScalar::new(0.0, 0.0));
        }
        return Ok((values, derivatives));
    }

    let tail = *weighted.last().expect("checked non-empty");
    for &z in z_values {
        if !z.re.is_finite() || !z.im.is_finite() {
            return Err(MellinError::NonFiniteZ { re: z.re, im: z.im });
        }

        let mut p = tail;
        let mut dp = ComplexScalar::new(0.0, 0.0);
        for coeff in weighted[..weighted.len() - 1].iter().rev() {
            dp = dp * z + p;
            p = *coeff + z * p;
        }
        values.push(p);
        derivatives.push(dp);
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
    Ok(prefactor * series)
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

    let value = prefactor * series;
    let d_value_ds = prefactor * (start * series + step * z * d_series_dz);
    Ok((value, d_value_ds))
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // New: robustness tests
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
}
