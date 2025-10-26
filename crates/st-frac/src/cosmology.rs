// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::mellin_types::{ComplexScalar, MellinError, Scalar, ZSpaceError};
use crate::zspace::{
    evaluate_weighted_series, evaluate_weighted_series_many, prepare_weighted_series,
    trapezoidal_weights,
};
use std::f32::consts::PI;
use thiserror::Error;

/// Result alias for cosmology utilities.
pub type CosmologyResult<T> = Result<T, CosmologyError>;

/// Error type covering the cosmology helpers.
#[derive(Debug, Error)]
pub enum CosmologyError {
    #[error("sample sequence must not be empty")]
    EmptySamples,
    #[error("log step must be finite and positive")]
    InvalidLogStep,
    #[error("log start must be finite")]
    InvalidLogStart,
    #[error("sample at index {index} is not finite")]
    NonFiniteSample { index: usize },
    #[error("weight normalisation produced zero divisor")]
    ZeroNormalisation,
    #[error("series are not compatible: {reason}")]
    IncompatibleSeries { reason: String },
    #[error("Planck mass must be finite and positive")]
    InvalidPlanckMass,
    #[error("projection for {channel} at index {index} is not finite")]
    NonFiniteProjection { channel: &'static str, index: usize },
    #[error(transparent)]
    Mellin(#[from] MellinError),
    #[error(transparent)]
    ZSpace(#[from] ZSpaceError),
}

/// Window functions applied to the trapezoidal weights before normalisation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowFunction {
    /// No tapering (rectangular window).
    Rectangular,
    /// Hann taper for endpoint leakage suppression.
    Hann,
}

impl Default for WindowFunction {
    fn default() -> Self {
        WindowFunction::Rectangular
    }
}

impl WindowFunction {
    fn apply(self, weights: &mut [Scalar]) {
        match self {
            WindowFunction::Rectangular => {}
            WindowFunction::Hann => {
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
        }
    }
}

/// Normalisation applied to the windowed weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightNormalisation {
    /// No normalisation is applied.
    None,
    /// L1 normalisation (sum of absolute values equals 1).
    L1,
    /// L2 normalisation (Euclidean norm equals 1).
    L2,
}

impl Default for WeightNormalisation {
    fn default() -> Self {
        WeightNormalisation::L1
    }
}

impl WeightNormalisation {
    fn apply(self, weights: &mut [Scalar]) -> CosmologyResult<()> {
        match self {
            WeightNormalisation::None => Ok(()),
            WeightNormalisation::L1 => {
                let norm: Scalar = weights.iter().map(|w| w.abs()).sum();
                if norm <= Scalar::EPSILON {
                    return Err(CosmologyError::ZeroNormalisation);
                }
                for w in weights.iter_mut() {
                    *w /= norm;
                }
                Ok(())
            }
            WeightNormalisation::L2 => {
                let norm: Scalar = weights.iter().map(|w| w * w).sum::<Scalar>().sqrt();
                if norm <= Scalar::EPSILON {
                    return Err(CosmologyError::ZeroNormalisation);
                }
                for w in weights.iter_mut() {
                    *w /= norm;
                }
                Ok(())
            }
        }
    }
}

/// Options controlling how a log-time series is projected into Z-space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeriesOptions {
    pub window: WindowFunction,
    pub normalisation: WeightNormalisation,
}

impl Default for SeriesOptions {
    fn default() -> Self {
        Self {
            window: WindowFunction::default(),
            normalisation: WeightNormalisation::default(),
        }
    }
}

/// Log-time sequence projected into Z-space with pre-weighted coefficients.
#[derive(Clone, Debug)]
pub struct LogZSeries {
    log_start: Scalar,
    log_step: Scalar,
    len: usize,
    samples: Vec<Scalar>,
    weights: Vec<Scalar>,
    weighted_coeffs: Vec<ComplexScalar>,
    options: SeriesOptions,
}

impl LogZSeries {
    /// Build a series from real-valued samples using default options.
    pub fn from_samples(
        log_start: Scalar,
        log_step: Scalar,
        samples: Vec<Scalar>,
    ) -> CosmologyResult<Self> {
        Self::from_samples_with_options(log_start, log_step, samples, SeriesOptions::default())
    }

    /// Build a series from samples with explicit window/normalisation options.
    pub fn from_samples_with_options(
        log_start: Scalar,
        log_step: Scalar,
        samples: Vec<Scalar>,
        options: SeriesOptions,
    ) -> CosmologyResult<Self> {
        if samples.is_empty() {
            return Err(CosmologyError::EmptySamples);
        }
        if !(log_step.is_finite() && log_step > 0.0) {
            return Err(CosmologyError::InvalidLogStep);
        }
        if !log_start.is_finite() {
            return Err(CosmologyError::InvalidLogStart);
        }
        for (idx, value) in samples.iter().enumerate() {
            if !value.is_finite() {
                return Err(CosmologyError::NonFiniteSample { index: idx });
            }
        }

        let mut weights = trapezoidal_weights(samples.len())?;
        options.window.apply(&mut weights);
        options.normalisation.apply(&mut weights)?;

        let complex_samples: Vec<ComplexScalar> = samples
            .iter()
            .map(|&v| ComplexScalar::new(v, 0.0))
            .collect();
        let weighted_coeffs = prepare_weighted_series(&complex_samples, &weights)?;

        Ok(Self {
            log_start,
            log_step,
            len: samples.len(),
            samples,
            weights,
            weighted_coeffs,
            options,
        })
    }

    /// Number of log-time samples.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Starting log-time of the lattice.
    pub fn log_start(&self) -> Scalar {
        self.log_start
    }

    /// Log-time step of the lattice.
    pub fn log_step(&self) -> Scalar {
        self.log_step
    }

    /// Options used to build the series.
    pub fn options(&self) -> SeriesOptions {
        self.options
    }

    /// Underlying real samples.
    pub fn samples(&self) -> &[Scalar] {
        &self.samples
    }

    /// Windowed weights used during the projection.
    pub fn weights(&self) -> &[Scalar] {
        &self.weights
    }

    /// Evaluate the weighted Z-series at a specific point.
    pub fn evaluate_z(&self, z: ComplexScalar) -> CosmologyResult<ComplexScalar> {
        Ok(evaluate_weighted_series(&self.weighted_coeffs, z)?)
    }

    /// Evaluate the weighted Z-series at multiple Z-points.
    pub fn evaluate_many_z(
        &self,
        z_values: &[ComplexScalar],
    ) -> CosmologyResult<Vec<ComplexScalar>> {
        Ok(evaluate_weighted_series_many(
            &self.weighted_coeffs,
            z_values,
        )?)
    }

    /// Ensure another series shares the same lattice configuration.
    pub fn ensure_compatible(&self, other: &Self) -> CosmologyResult<()> {
        if self.len != other.len {
            return Err(CosmologyError::IncompatibleSeries {
                reason: format!("length mismatch {} vs {}", self.len, other.len),
            });
        }
        if (self.log_step - other.log_step).abs() > 1e-6 {
            return Err(CosmologyError::IncompatibleSeries {
                reason: format!("log_step mismatch {} vs {}", self.log_step, other.log_step),
            });
        }
        if (self.log_start - other.log_start).abs() > 1e-6 {
            return Err(CosmologyError::IncompatibleSeries {
                reason: format!(
                    "log_start mismatch {} vs {}",
                    self.log_start, other.log_start
                ),
            });
        }
        Ok(())
    }
}

/// Assemble the curvature perturbation power spectrum from Z-projected background data.
pub fn assemble_pzeta(
    z_points: &[ComplexScalar],
    h_series: &LogZSeries,
    epsilon_series: &LogZSeries,
    planck_mass: Scalar,
) -> CosmologyResult<Vec<Scalar>> {
    if z_points.is_empty() {
        return Ok(Vec::new());
    }
    if !(planck_mass.is_finite() && planck_mass > 0.0) {
        return Err(CosmologyError::InvalidPlanckMass);
    }
    h_series.ensure_compatible(epsilon_series)?;

    let h_proj = h_series.evaluate_many_z(z_points)?;
    let eps_proj = epsilon_series.evaluate_many_z(z_points)?;

    let denom = 8.0 * PI * PI * planck_mass * planck_mass;
    let mut spectrum = Vec::with_capacity(z_points.len());

    for (idx, (h_z, eps_z)) in h_proj.into_iter().zip(eps_proj.into_iter()).enumerate() {
        if !(h_z.re.is_finite() && h_z.im.is_finite()) {
            return Err(CosmologyError::NonFiniteProjection {
                channel: "H",
                index: idx,
            });
        }
        if !(eps_z.re.is_finite() && eps_z.im.is_finite()) {
            return Err(CosmologyError::NonFiniteProjection {
                channel: "epsilon",
                index: idx,
            });
        }
        let eps = eps_z.re.max(1e-20);
        let power = (h_z.re * h_z.re + h_z.im * h_z.im) / (denom * eps);
        spectrum.push(power);
    }

    Ok(spectrum)
}

/// Map Mellin-space abscissa values into the Z-plane for a log-time lattice.
pub fn log_lattice_z_points(
    log_step: Scalar,
    s_values: &[ComplexScalar],
) -> CosmologyResult<Vec<ComplexScalar>> {
    if s_values.is_empty() {
        return Ok(Vec::new());
    }
    if !(log_step.is_finite() && log_step > 0.0) {
        return Err(CosmologyError::InvalidLogStep);
    }
    let step = ComplexScalar::new(log_step, 0.0);
    Ok(s_values.iter().map(|&s| (s * step).exp()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn demo_series(value: Scalar) -> LogZSeries {
        let samples = vec![value; 8];
        LogZSeries::from_samples_with_options(0.0, 0.25, samples, SeriesOptions::default()).unwrap()
    }

    #[test]
    fn series_rejects_invalid_inputs() {
        let err = LogZSeries::from_samples(0.0, 0.25, Vec::new()).unwrap_err();
        assert!(matches!(err, CosmologyError::EmptySamples));

        let err = LogZSeries::from_samples(0.0, -0.5, vec![1.0]).unwrap_err();
        assert!(matches!(err, CosmologyError::InvalidLogStep));

        let err = LogZSeries::from_samples(f32::NAN, 0.5, vec![1.0]).unwrap_err();
        assert!(matches!(err, CosmologyError::InvalidLogStart));

        let err = LogZSeries::from_samples(0.0, 0.5, vec![1.0, f32::INFINITY]).unwrap_err();
        assert!(matches!(err, CosmologyError::NonFiniteSample { .. }));
    }

    #[test]
    fn series_projects_with_expected_weights() {
        let samples = (0..4).map(|i| i as Scalar + 1.0).collect::<Vec<_>>();
        let series = LogZSeries::from_samples_with_options(
            0.0,
            0.5,
            samples.clone(),
            SeriesOptions::default(),
        )
        .unwrap();
        let z = ComplexScalar::new(0.7, -0.3);
        let projected = series.evaluate_z(z).unwrap();

        let weights = series.weights();
        let mut manual = ComplexScalar::new(0.0, 0.0);
        let mut z_power = ComplexScalar::new(1.0, 0.0);
        for (idx, sample) in samples.iter().enumerate() {
            if idx > 0 {
                z_power *= z;
            }
            let term = ComplexScalar::new(*sample, 0.0) * weights[idx];
            manual += term * z_power;
        }
        let diff = (projected - manual).norm();
        assert!(diff < 1e-5, "diff={}", diff);
    }

    #[test]
    fn assemble_pzeta_matches_constant_background() {
        let h_series = demo_series(10.0);
        let eps_series = demo_series(0.05);
        let z_points = vec![
            ComplexScalar::new(0.5, 0.0),
            ComplexScalar::new(0.2, 0.3),
            ComplexScalar::new(-0.1, 0.4),
        ];
        let spectrum = assemble_pzeta(&z_points, &h_series, &eps_series, 1.0).unwrap();

        let weights = h_series.weights();
        for (idx, &z) in z_points.iter().enumerate() {
            let mut accum = ComplexScalar::new(0.0, 0.0);
            let mut z_power = ComplexScalar::new(1.0, 0.0);
            for (k, &w) in weights.iter().enumerate() {
                if k > 0 {
                    z_power *= z;
                }
                accum += ComplexScalar::new(w, 0.0) * z_power;
            }
            let h_z = ComplexScalar::new(10.0, 0.0) * accum;
            let eps_re = (0.05 * accum.re).max(1e-20);
            let h_norm_sq = h_z.re * h_z.re + h_z.im * h_z.im;
            let expected = h_norm_sq / (8.0 * PI * PI * eps_re);
            assert_relative_eq!(spectrum[idx], expected, max_relative = 1e-5);
        }
    }

    #[test]
    fn log_lattice_z_points_matches_manual() {
        let s_values = vec![
            ComplexScalar::new(-0.5, 0.0),
            ComplexScalar::new(-0.1, 0.7),
            ComplexScalar::new(-0.25, -0.3),
        ];
        let log_step = 0.3;
        let mapped = log_lattice_z_points(log_step, &s_values).unwrap();
        for (idx, (&s, &z)) in s_values.iter().zip(mapped.iter()).enumerate() {
            let expected = (s * ComplexScalar::new(log_step, 0.0)).exp();
            let diff = (z - expected).norm();
            assert!(diff < 1e-6, "idx={} diff={}", idx, diff);
        }
    }

    #[test]
    fn assemble_pzeta_rejects_incompatible_series() {
        let h_series = demo_series(1.0);
        let eps_series = LogZSeries::from_samples_with_options(
            0.0,
            0.25,
            vec![0.1; 6],
            SeriesOptions::default(),
        )
        .unwrap();
        let err = assemble_pzeta(&[ComplexScalar::new(0.5, 0.0)], &h_series, &eps_series, 1.0)
            .unwrap_err();
        assert!(matches!(err, CosmologyError::IncompatibleSeries { .. }));

        let eps_series = LogZSeries::from_samples_with_options(
            0.1,
            0.25,
            vec![0.1; 8],
            SeriesOptions::default(),
        )
        .unwrap();
        let err = assemble_pzeta(&[ComplexScalar::new(0.5, 0.0)], &h_series, &eps_series, 1.0)
            .unwrap_err();
        assert!(matches!(err, CosmologyError::IncompatibleSeries { .. }));
    }
}
