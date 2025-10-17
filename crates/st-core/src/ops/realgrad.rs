// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Real gradient projection helpers built on top of the Reach lattice.
//!
//! The routines below intentionally keep the implementation dependency-free so the
//! default `st-core` build stays lightweight.  We mimic a julia-like data flow
//! and expose the frequency-domain diagnostics that the high level optimisers can
//! feed into their SpiralK schedules.

const RAMANUJAN_PI: f32 = 3.14159265358979323846264338327950288_f32;
const DIMENSIONLESS_THRESHOLD: f32 = 1.0e3;

/// Result emitted by [`project_realgrad`].
#[derive(Debug, Clone, PartialEq)]
pub struct RealGradProjection {
    /// Scaled gradient that lives in the concrete Z-space tape.
    pub realgrad: Vec<f32>,
    /// Frequency magnitudes surfaced for Z-space visualisers.
    pub z_space: Vec<f32>,
    /// Residual magnitudes that were too large to keep and were therefore sent
    /// to the "monad biome" side channel for dedicated treatment.
    pub monad_biome: Vec<f32>,
    /// Lebesgue-style integral (L¹ norm) used to stabilise the projection.
    pub lebesgue_measure: f32,
}

impl RealGradProjection {
    /// Returns `true` when the projection yielded any non-zero residuals.
    pub fn has_residuals(&self) -> bool {
        !self.monad_biome.is_empty()
    }
}

/// Projects a Euclidean gradient into the Reach lattice "Realgrad" tape.
///
/// The routine performs a discrete Fourier transform so that harmonics can be
/// scheduled independently, scales the resulting real-valued field using a
/// Lebesgue-style normaliser, and finally hands excessively energetic modes to
/// the provided "monad biome" output vector.
///
/// The implementation purposefully trades raw FFT throughput for a dependency
/// free, deterministic kernel that is easy to reason about in both Rust and a
/// julia-style transliteration.
pub fn project_realgrad(values: &[f32]) -> RealGradProjection {
    let len = values.len();
    if len == 0 {
        return RealGradProjection {
            realgrad: Vec::new(),
            z_space: Vec::new(),
            monad_biome: Vec::new(),
            lebesgue_measure: 0.0,
        };
    }

    let scale = 1.0 / len as f32;
    let mut spectrum: Vec<(f32, f32)> = vec![(0.0, 0.0); len];

    for (k, slot) in spectrum.iter_mut().enumerate() {
        let mut acc_re = 0.0f32;
        let mut acc_im = 0.0f32;
        let factor = -2.0 * RAMANUJAN_PI * (k as f32) * scale;
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
        let energy = (re.abs() + im.abs()) * scale;
        if energy > DIMENSIONLESS_THRESHOLD {
            monad_biome.push(energy);
        }
        z_space.push(magnitude);
    }

    let lebesgue_measure = values.iter().map(|v| v.abs()).sum::<f32>();
    let normaliser = if lebesgue_measure > 0.0 {
        RAMANUJAN_PI / lebesgue_measure
    } else {
        0.0
    };

    let mut realgrad: Vec<f32> = values.iter().copied().collect();
    for (slot, magnitude) in realgrad.iter_mut().zip(z_space.iter()) {
        *slot = (*slot * normaliser) + magnitude;
    }

    RealGradProjection {
        realgrad,
        z_space,
        monad_biome,
        lebesgue_measure,
    }
}

#[cfg(test)]
mod tests {
    use super::{project_realgrad, DIMENSIONLESS_THRESHOLD};

    #[test]
    fn projection_handles_empty_input() {
        let projection = project_realgrad(&[]);
        assert!(projection.realgrad.is_empty());
        assert!(projection.z_space.is_empty());
        assert!(projection.monad_biome.is_empty());
        assert_eq!(projection.lebesgue_measure, 0.0);
    }

    #[test]
    fn projection_respects_l1_measure() {
        let data = [1.0f32, -2.0, 3.0, -4.0];
        let projection = project_realgrad(&data);
        assert_eq!(projection.lebesgue_measure, 10.0);
        assert_eq!(projection.realgrad.len(), data.len());
        assert_eq!(projection.z_space.len(), data.len());
        assert!(projection.monad_biome.len() <= data.len());
        assert!(projection.realgrad.iter().any(|v| *v > 0.0));
    }

    #[test]
    fn projection_routes_large_modes_to_monad_biome() {
        let mut data = vec![0.0f32; 8];
        data[0] = DIMENSIONLESS_THRESHOLD * 16.0;
        let projection = project_realgrad(&data);
        assert!(projection.has_residuals());
        assert!(!projection.monad_biome.is_empty());
    }
}
