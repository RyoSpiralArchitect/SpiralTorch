// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Fractal noise generators tailored for Z-space lattices.
//!
//! The helpers in this module synthesise self-similar fields on logarithmic
//! lattices so Mellin-aware models can inject narrative-style branching or
//! cosmological texture into their flows. The implementation keeps the output
//! deterministic for a given seed, making it suitable for reproducible research
//! experiments and unit tests alike.

use core::f32::consts::TAU;
use rand::{rngs::StdRng, Rng, SeedableRng};
use thiserror::Error;

/// Errors emitted by the fractal field generator.
#[derive(Debug, Error)]
pub enum FractalFieldError {
    /// The generator was asked to operate on an empty or zero-dimensional field.
    #[error("dimension must be greater than zero")]
    Dimension,
    /// The requested lattice depth must be positive.
    #[error("lattice depth must be greater than zero")]
    Depth,
    /// Encountered a non-finite scale while building the lattice.
    #[error("scale[{index}] is not finite: {value}")]
    NonFiniteScale { index: usize, value: f32 },
}

/// Self-similar samples indexed by `(scale_index, feature_index)`.
#[derive(Clone, Debug)]
pub struct FractalField {
    scales: Vec<f32>,
    dimension: usize,
    samples: Vec<f32>,
}

impl FractalField {
    fn offset(&self, scale_index: usize, feature: usize) -> usize {
        scale_index * self.dimension + feature
    }

    /// Returns the log-scale lattice used to generate the field.
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Number of independent fractal channels carried by the field.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Number of log-lattice samples.
    pub fn depth(&self) -> usize {
        self.scales.len()
    }

    /// Returns the raw amplitude at `(scale_index, feature)`.
    pub fn sample(&self, scale_index: usize, feature: usize) -> Option<f32> {
        self.samples.get(self.offset(scale_index, feature)).copied()
    }

    /// Mean amplitude across all features at the supplied scale.
    pub fn mean_amplitude(&self, scale_index: usize) -> Option<f32> {
        if scale_index >= self.depth() {
            return None;
        }
        let start = self.offset(scale_index, 0);
        let end = start + self.dimension;
        let slice = &self.samples[start..end];
        let sum: f32 = slice.iter().copied().sum();
        Some(sum / self.dimension as f32)
    }

    /// Returns an iterator over the field samples grouped by scale.
    pub fn iter_scales(&self) -> impl Iterator<Item = &[f32]> {
        self.samples.chunks(self.dimension)
    }
}

/// Deterministic fractal generator operating on log-uniform lattices.
#[derive(Clone, Debug)]
pub struct FractalFieldGenerator {
    dimension: usize,
    spectral_exponent: f32,
    octaves: usize,
    persistence: f32,
    seed: u64,
}

impl FractalFieldGenerator {
    /// Creates a generator with the provided spectral exponent and RNG seed.
    pub fn new(
        dimension: usize,
        spectral_exponent: f32,
        seed: u64,
    ) -> Result<Self, FractalFieldError> {
        if dimension == 0 {
            return Err(FractalFieldError::Dimension);
        }
        let octaves = 6usize;
        Ok(Self {
            dimension,
            spectral_exponent: spectral_exponent.max(0.0),
            octaves,
            persistence: 0.65,
            seed,
        })
    }

    /// Configures the number of octaves used when synthesising the field.
    pub fn with_octaves(mut self, octaves: usize) -> Self {
        self.octaves = octaves.max(1);
        self
    }

    /// Adjusts the persistence factor (how quickly the amplitude decays per octave).
    pub fn with_persistence(mut self, persistence: f32) -> Self {
        self.persistence = persistence.clamp(0.0, 1.0);
        self
    }

    /// Generates a fractal field on a log-uniform lattice described by `(log_start, log_step, depth)`.
    pub fn generate(
        &self,
        log_start: f32,
        log_step: f32,
        depth: usize,
    ) -> Result<FractalField, FractalFieldError> {
        if depth == 0 {
            return Err(FractalFieldError::Depth);
        }
        let mut scales = Vec::with_capacity(depth);
        for i in 0..depth {
            scales.push(log_start + log_step * i as f32);
        }
        self.generate_for_scales(&scales)
    }

    /// Generates a fractal field on top of an existing lattice.
    pub fn generate_for_scales(&self, scales: &[f32]) -> Result<FractalField, FractalFieldError> {
        if scales.is_empty() {
            return Err(FractalFieldError::Depth);
        }
        for (index, &scale) in scales.iter().enumerate() {
            if !scale.is_finite() {
                return Err(FractalFieldError::NonFiniteScale {
                    index,
                    value: scale,
                });
            }
        }

        let mut samples = Vec::with_capacity(scales.len() * self.dimension);
        for (s_idx, &scale) in scales.iter().enumerate() {
            for feature in 0..self.dimension {
                let mut rng =
                    StdRng::seed_from_u64(self.seed ^ ((s_idx as u64) << 32) ^ feature as u64);
                samples.push(self.fractal_value(scale, feature, &mut rng));
            }
        }

        Ok(FractalField {
            scales: scales.to_vec(),
            dimension: self.dimension,
            samples,
        })
    }

    fn fractal_value(&self, scale: f32, feature: usize, rng: &mut StdRng) -> f32 {
        let mut frequency = 1.0f32;
        let mut amplitude = 0.0f32;
        let mut weight_sum = 0.0f32;
        let persistence = self.persistence.clamp(1e-3, 0.999);
        let mut octave_gain = 1.0f32;
        for _ in 0..self.octaves {
            let weight = frequency.powf(-self.spectral_exponent) * octave_gain;
            let phase = rng.gen::<f32>() * TAU;
            let drift = ((feature as f32 + 1.0) * scale * frequency).sin();
            amplitude += weight * (phase + drift).sin();
            weight_sum += weight;
            frequency *= 2.0;
            octave_gain *= persistence;
        }
        if weight_sum <= 0.0 {
            0.0
        } else {
            amplitude / weight_sum
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_is_deterministic() {
        let gen = FractalFieldGenerator::new(3, 1.0, 0x5A5A).unwrap();
        let field_a = gen.generate(0.0, 0.25, 5).unwrap();
        let field_b = gen.generate(0.0, 0.25, 5).unwrap();
        assert_eq!(field_a.samples, field_b.samples);
    }

    #[test]
    fn mean_amplitude_matches_manual_average() {
        let gen = FractalFieldGenerator::new(4, 0.5, 42).unwrap();
        let field = gen.generate(-1.0, 0.5, 3).unwrap();
        for scale_idx in 0..field.depth() {
            let mean = field.mean_amplitude(scale_idx).unwrap();
            let mut manual = 0.0;
            for feature in 0..field.dimension() {
                manual += field.sample(scale_idx, feature).unwrap();
            }
            manual /= field.dimension() as f32;
            assert!((mean - manual).abs() < 1e-6);
        }
    }
}
