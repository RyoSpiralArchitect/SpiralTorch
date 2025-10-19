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
///
/// The Mellin transform evaluated on a log-uniform grid `t_k = log_start + k * log_step`
/// can be rewritten as a weighted power series on the complex unit `z = exp(s * log_step)`.
/// This helper returns the scalar prefactor and the Z-plane point that together encode
/// that change of variables.
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
///
/// The caller supplies per-sample trapezoidal weights that capture the Hilbert-space
/// measure induced by the Mellin transform. The returned value corresponds to the
/// discrete power series `sum_k w_k x_k z^k`.
pub fn weighted_z_transform(
    samples: &[ComplexScalar],
    weights: &[Scalar],
    z: ComplexScalar,
) -> MellinResult<ComplexScalar> {
    let weighted = prepare_weighted_series(samples, weights)?;
    evaluate_weighted_series(&weighted, z)
}

/// Evaluate the weighted Z-transform at multiple `z` points.
///
/// This helper shares the preweighted samples across all evaluation points,
/// avoiding repeated weight application when sweeping an entire vertical line in
/// the Mellin domain.
pub fn weighted_z_transform_many(
    samples: &[ComplexScalar],
    weights: &[Scalar],
    z_values: &[ComplexScalar],
) -> MellinResult<Vec<ComplexScalar>> {
    let weighted = prepare_weighted_series(samples, weights)?;
    evaluate_weighted_series_many(&weighted, z_values)
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

    let mut acc = ComplexScalar::new(0.0, 0.0);
    let mut power = ComplexScalar::new(1.0, 0.0);

    for coeff in weighted.iter() {
        acc += *coeff * power;
        power *= z;
    }

    Ok(acc)
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

    z_values
        .iter()
        .map(|&z| evaluate_weighted_series(weighted, z))
        .collect()
}

/// Evaluate the Mellin transform on a log-uniform grid by routing the computation
/// through its Z-plane power-series form.
///
/// This provides a numerically stable pathway to couple Mellin-domain analyses with
/// Z-space tooling such as the pulse interfaces in SpiralTorch, without touching the
/// low-level pulse definitions directly.
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
    fn prepared_series_matches_direct_weighting() {
        let samples = vec![
            ComplexScalar::new(1.0, -0.5),
            ComplexScalar::new(-0.7, 0.2),
            ComplexScalar::new(0.3, 0.9),
        ];
        let weights = vec![0.5, 1.0, 0.5];
        let weighted = prepare_weighted_series(&samples, &weights).unwrap();
        for (idx, coeff) in weighted.iter().enumerate() {
            let manual = samples[idx] * ComplexScalar::new(weights[idx], 0.0);
            let diff = (*coeff - manual).norm();
            assert!(diff < 1e-6, "idx={} diff={}", idx, diff);
        }
    }

    #[test]
    fn evaluate_weighted_series_many_shares_coefficients() {
        let samples = vec![
            ComplexScalar::new(0.4, -0.2),
            ComplexScalar::new(-0.1, 0.7),
            ComplexScalar::new(0.3, 0.1),
            ComplexScalar::new(-0.2, -0.5),
        ];
        let weights = trapezoidal_weights(samples.len()).unwrap();
        let weighted = prepare_weighted_series(&samples, &weights).unwrap();
        let zs = [ComplexScalar::new(0.9, 0.1), ComplexScalar::new(-0.3, 0.4)];

        let via_prepare = evaluate_weighted_series_many(&weighted, &zs).unwrap();
        let via_api = weighted_z_transform_many(&samples, &weights, &zs).unwrap();

        for (idx, (lhs, rhs)) in via_prepare.iter().zip(via_api.iter()).enumerate() {
            let diff = (*lhs - *rhs).norm();
            assert!(diff < 1e-6, "idx={} diff={}", idx, diff);
        }
    }
}
