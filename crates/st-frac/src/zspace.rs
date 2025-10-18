// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use num_complex::Complex32;

/// Compute trapezoidal-rule weights for a log-uniform lattice.
#[inline]
pub fn trapezoidal_weights(len: usize) -> Vec<f32> {
    assert!(len > 0, "at least one sample is required");
    if len == 1 {
        return vec![1.0];
    }
    let mut weights = vec![1.0f32; len];
    if let Some(first) = weights.first_mut() {
        *first = 0.5;
    }
    if let Some(last) = weights.last_mut() {
        *last = 0.5;
    }
    weights
}

/// Map Mellin abscissa data to the Z-plane representation.
///
/// The Mellin transform evaluated on a log-uniform grid `t_k = log_start + k * log_step`
/// can be rewritten as a weighted power series on the complex unit `z = exp(s * log_step)`.
/// This helper returns the scalar prefactor and the Z-plane point that together encode
/// that change of variables.
#[inline]
pub fn mellin_log_lattice_prefactor(
    log_start: f32,
    log_step: f32,
    s: Complex32,
) -> (Complex32, Complex32) {
    assert!(
        log_step.is_finite() && log_step > 0.0,
        "log_step must be positive"
    );
    assert!(log_start.is_finite(), "log_start must be finite");

    let step = Complex32::new(log_step, 0.0);
    let start = Complex32::new(log_start, 0.0);
    let prefactor = (s * start).exp() * step;
    let z = (s * step).exp();
    (prefactor, z)
}

/// Evaluate a weighted Z-transform at the point `z`.
///
/// The caller supplies per-sample trapezoidal weights that capture the Hilbert-space
/// measure induced by the Mellin transform. The returned value corresponds to the
/// discrete power series `sum_k w_k x_k z^k`.
pub fn weighted_z_transform(samples: &[Complex32], weights: &[f32], z: Complex32) -> Complex32 {
    assert!(!samples.is_empty(), "samples must not be empty");
    assert_eq!(samples.len(), weights.len(), "weights must match samples");

    let mut acc = Complex32::new(0.0, 0.0);
    let mut power = Complex32::new(1.0, 0.0);

    for (sample, &w) in samples.iter().zip(weights.iter()) {
        let weight = Complex32::new(w, 0.0);
        acc += *sample * weight * power;
        power *= z;
    }

    acc
}

/// Evaluate the Mellin transform on a log-uniform grid by routing the computation
/// through its Z-plane power-series form.
///
/// This provides a numerically stable pathway to couple Mellin-domain analyses with
/// Z-space tooling such as the pulse interfaces in SpiralTorch, without touching the
/// low-level pulse definitions directly.
pub fn mellin_transform_via_z(
    log_start: f32,
    log_step: f32,
    samples: &[Complex32],
    s: Complex32,
) -> Complex32 {
    let weights = trapezoidal_weights(samples.len());
    let (prefactor, z) = mellin_log_lattice_prefactor(log_start, log_step, s);
    let series = weighted_z_transform(samples, &weights, z);
    prefactor * series
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trapezoidal_weights_match_manual() {
        let single = trapezoidal_weights(1);
        assert_eq!(single, vec![1.0]);

        let weights = trapezoidal_weights(4);
        assert_eq!(weights, vec![0.5, 1.0, 1.0, 0.5]);
    }

    #[test]
    fn weighted_z_transform_reduces_to_polynomial() {
        let samples = [
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(-1.0, 0.0),
        ];
        let weights = vec![1.0, 1.0, 1.0];
        let z = Complex32::new(0.5, 0.5);
        let value = weighted_z_transform(&samples, &weights, z);
        let expected = samples[0] + samples[1] * z + samples[2] * z * z;
        let diff = (value - expected).norm();
        assert!(diff < 1e-6, "diff={}", diff);
    }

    #[test]
    fn mellin_via_z_matches_direct_series() {
        let samples = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(0.5, 0.0),
            Complex32::new(0.25, 0.0),
        ];
        let log_start = -2.0f32;
        let log_step = 0.5f32;
        let s = Complex32::new(1.2, 0.7);

        let direct = {
            let mut acc = Complex32::new(0.0, 0.0);
            for (idx, &sample) in samples.iter().enumerate() {
                let weight = if idx == 0 || idx + 1 == samples.len() {
                    0.5
                } else {
                    1.0
                };
                let t = log_start + log_step * idx as f32;
                let kernel = (s * Complex32::new(t, 0.0)).exp();
                acc += sample * Complex32::new(weight, 0.0) * kernel;
            }
            acc * Complex32::new(log_step, 0.0)
        };

        let via_z = mellin_transform_via_z(log_start, log_step, &samples, s);
        let diff = (direct - via_z).norm();
        assert!(diff < 1e-6, "diff={}", diff);
    }
}
