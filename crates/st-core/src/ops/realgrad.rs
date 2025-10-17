// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Real gradient projection helpers that blend the Reach lattice sketch with the
//! existing Leech lattice tooling.
//!
//! The helper intentionally stays dependency free so we can drive the julia-like
//! transliteration from notebooks, but now reuses the shared Ramanujan π series
//! and Leech density projector that already live inside the repository.

use core::f32::consts::PI;

use crate::util::math::{ramanujan_pi, LeechProjector};

const DEFAULT_RANK: usize = 24;
const DEFAULT_WEIGHT: f64 = 1.0;
const DEFAULT_THRESHOLD: f32 = 0.005;

/// Result emitted by [`project_realgrad`].
#[derive(Debug, Clone, PartialEq)]
pub struct RealGradProjection {
    /// Scaled gradient that lives in the concrete Z-space tape.
    pub realgrad: Vec<f32>,
    /// Leech-projected Z-space magnitudes.
    pub z_space: Vec<f32>,
    /// Residual magnitudes that were too large to keep and were therefore sent
    /// to the "monad biome" side channel for dedicated treatment.
    pub monad_biome: Vec<f32>,
    /// Lebesgue-style integral (L¹ norm) used to stabilise the projection.
    pub lebesgue_measure: f32,
    /// Ramanujan π estimate used for the projection.
    pub ramanujan_pi: f32,
}

impl RealGradProjection {
    /// Returns `true` when the projection yielded any non-zero residuals.
    pub fn has_residuals(&self) -> bool {
        !self.monad_biome.is_empty()
    }

    /// Returns the total magnitude routed to the monad biome.
    pub fn residual_energy(&self) -> f32 {
        self.monad_biome.iter().copied().sum()
    }

    /// Returns the accumulated energy of the Z-space field.
    pub fn z_energy(&self) -> f32 {
        self.z_space.iter().map(|value| value.abs()).sum()
    }
}

/// Configuration for [`project_realgrad`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RealGradConfig {
    /// Number of iterations used for the Ramanujan π approximation.
    pub ramanujan_iterations: usize,
    /// Target Z-space rank for the Leech projector.
    pub z_rank: usize,
    /// Weight applied when enriching magnitudes with the Leech projector.
    pub z_weight: f64,
    /// Threshold beyond which magnitudes are routed to the monad biome.
    pub residual_threshold: f32,
}

impl Default for RealGradConfig {
    fn default() -> Self {
        Self {
            ramanujan_iterations: 4,
            z_rank: DEFAULT_RANK,
            z_weight: DEFAULT_WEIGHT,
            residual_threshold: DEFAULT_THRESHOLD,
        }
    }
}

impl RealGradConfig {
    /// Ensures the configuration is numerically stable.
    fn sanitised(self) -> Self {
        Self {
            ramanujan_iterations: self.ramanujan_iterations.max(1),
            z_rank: self.z_rank.max(1),
            z_weight: self.z_weight.max(0.0),
            residual_threshold: self.residual_threshold.max(0.0),
        }
    }

    /// Calibrates the configuration against a previous projection and returns
    /// the tuned configuration alongside the tuning diagnostics.
    pub fn calibrate(self, projection: &RealGradProjection) -> (Self, RealGradTuning) {
        let base = self.sanitised();
        let monad_energy = projection.residual_energy();
        let z_energy = projection.z_energy();
        let lebesgue = projection.lebesgue_measure.max(f32::EPSILON);
        let total_energy = (monad_energy + z_energy).max(f32::EPSILON);
        let residual_ratio = monad_energy / total_energy;
        let lebesgue_ratio = monad_energy / lebesgue;
        let pi_multiplier = if PI > 0.0 {
            (projection.ramanujan_pi / PI).max(0.0)
        } else {
            1.0
        };
        let target_ratio = 0.1f32;
        let raw_adjustment = 1.0 + (residual_ratio - target_ratio) * pi_multiplier + lebesgue_ratio;
        let adjustment_factor = raw_adjustment.clamp(0.25, 4.0);
        let suggested_threshold = (base.residual_threshold * adjustment_factor).max(0.0);

        let tuned_config = RealGradConfig {
            residual_threshold: suggested_threshold,
            ..base
        };

        let diagnostics = RealGradTuning {
            monad_energy,
            z_energy,
            residual_ratio,
            lebesgue_ratio,
            pi_multiplier,
            adjustment_factor,
            suggested_threshold,
        };

        (tuned_config, diagnostics)
    }
}

/// Diagnostics emitted by [`RealGradConfig::calibrate`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RealGradTuning {
    /// Total magnitude rerouted to the monad biome in the previous projection.
    pub monad_energy: f32,
    /// Total magnitude retained in the Z-space field.
    pub z_energy: f32,
    /// Share of the overall energy captured by the monad biome.
    pub residual_ratio: f32,
    /// Ratio between the monad biome energy and the Lebesgue integral.
    pub lebesgue_ratio: f32,
    /// Multiplier derived from the Ramanujan π estimate relative to π.
    pub pi_multiplier: f32,
    /// Factor applied to the previous residual threshold.
    pub adjustment_factor: f32,
    /// Suggested residual threshold for the next projection.
    pub suggested_threshold: f32,
}

/// Projects a Euclidean gradient into the Reach lattice "Realgrad" tape.
///
/// The routine performs a discrete Fourier transform so that harmonics can be
/// scheduled independently, scales the resulting real-valued field using a
/// Lebesgue-style normaliser, enriches the magnitudes through the Leech lattice,
/// and finally hands excessively energetic modes to the "monad biome" output
/// vector.
pub fn project_realgrad(values: &[f32], config: RealGradConfig) -> RealGradProjection {
    let len = values.len();
    let config = config.sanitised();
    if len == 0 {
        let ramanujan_pi = ramanujan_pi(config.ramanujan_iterations) as f32;
        return RealGradProjection {
            realgrad: Vec::new(),
            z_space: Vec::new(),
            monad_biome: Vec::new(),
            lebesgue_measure: 0.0,
            ramanujan_pi,
        };
    }

    let ramanujan_pi = ramanujan_pi(config.ramanujan_iterations) as f32;
    let projector = LeechProjector::new(config.z_rank, config.z_weight);
    let scale = 1.0 / len as f32;
    let mut spectrum: Vec<(f32, f32)> = vec![(0.0, 0.0); len];

    for (k, slot) in spectrum.iter_mut().enumerate() {
        let mut acc_re = 0.0f32;
        let mut acc_im = 0.0f32;
        let factor = -2.0 * ramanujan_pi * (k as f32) * scale;
        for (n_idx, &value) in values.iter().enumerate() {
            let angle = factor * (n_idx as f32);
            let (sin, cos) = angle.sin_cos();
            acc_re += value * cos;
            acc_im += value * sin;
        }
        *slot = (acc_re, acc_im);
    }

    let mut z_space = Vec::with_capacity(len);
    let mut monad_biome = Vec::new();
    for &(re, im) in &spectrum {
        let magnitude = (re * re + im * im).sqrt() * scale;
        if magnitude > config.residual_threshold {
            monad_biome.push(magnitude);
        }
        let enriched = projector.enrich(magnitude as f64) as f32;
        z_space.push(enriched);
    }

    let lebesgue_measure = values.iter().map(|v| v.abs()).sum::<f32>();
    let normaliser = if lebesgue_measure > 0.0 {
        ramanujan_pi / lebesgue_measure
    } else {
        0.0
    };

    let realgrad = values
        .iter()
        .zip(z_space.iter())
        .map(|(&value, &z)| value * normaliser + z)
        .collect();

    RealGradProjection {
        realgrad,
        z_space,
        monad_biome,
        lebesgue_measure,
        ramanujan_pi,
    }
}

#[cfg(test)]
mod tests {
    use super::{project_realgrad, RealGradConfig, DEFAULT_THRESHOLD};

    #[test]
    fn projection_handles_empty_input() {
        let projection = project_realgrad(&[], RealGradConfig::default());
        assert!(projection.realgrad.is_empty());
        assert!(projection.z_space.is_empty());
        assert!(projection.monad_biome.is_empty());
        assert_eq!(projection.lebesgue_measure, 0.0);
        assert!(projection.ramanujan_pi > 3.14);
    }

    #[test]
    fn projection_respects_l1_measure() {
        let data = [1.0f32, -2.0, 3.0, -4.0];
        let projection = project_realgrad(&data, RealGradConfig::default());
        assert_eq!(projection.lebesgue_measure, 10.0);
        assert_eq!(projection.realgrad.len(), data.len());
        assert_eq!(projection.z_space.len(), data.len());
        assert!(projection.monad_biome.len() <= data.len());
        assert!(projection.realgrad.iter().any(|v| *v > 0.0));
    }

    #[test]
    fn projection_routes_large_modes_to_monad_biome() {
        let mut data = vec![0.0f32; 8];
        data[0] = DEFAULT_THRESHOLD * 16.0;
        let projection = project_realgrad(&data, RealGradConfig::default());
        assert!(projection.has_residuals());
        assert!(!projection.monad_biome.is_empty());
    }

    #[test]
    fn projection_reports_energy_levels() {
        let data = [0.25f32; 4];
        let projection = project_realgrad(&data, RealGradConfig::default());
        assert!(projection.z_energy() > 0.0);
        assert_eq!(
            projection.residual_energy(),
            projection.monad_biome.iter().sum::<f32>()
        );
    }

    #[test]
    fn calibration_adjusts_threshold_based_on_residual_ratio() {
        let data = vec![DEFAULT_THRESHOLD * 1024.0; 16];
        let config = RealGradConfig::default();
        let projection = project_realgrad(&data, config);
        let (tuned, tuning) = config.calibrate(&projection);
        assert!(tuning.monad_energy > 0.0);
        assert!(tuning.lebesgue_ratio > 0.0);
        assert!(tuning.adjustment_factor > 1.0);
        assert!(tuned.residual_threshold > config.residual_threshold);

        let small_data = vec![DEFAULT_THRESHOLD * 0.1; 16];
        let small_projection = project_realgrad(&small_data, config);
        let (tuned_small, tuning_small) = config.calibrate(&small_projection);
        assert!(tuning_small.adjustment_factor < 1.0);
        assert!(tuned_small.residual_threshold < config.residual_threshold);
    }
}
