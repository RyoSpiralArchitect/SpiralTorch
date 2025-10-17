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
}
