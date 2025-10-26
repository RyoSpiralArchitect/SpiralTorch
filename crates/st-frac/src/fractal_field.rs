// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Fractal field generators that weave self-similar structure into the Z-lattice.
//!
//! The routines exposed here build deterministic, Mandelbrot-inspired perturbations
//! over the log lattices used by SpiralTorch’s Mellin tooling.  They provide a
//! controllable way to inject “branching cosmos” energy into Mellin grids so that
//! downstream Z-space operators can explore scale-invariant narratives.

use crate::mellin::MellinLogGrid;
use crate::mellin_types::{ComplexScalar, MellinError, Scalar};
use thiserror::Error;

/// Result type produced by the fractal field helpers.
pub type FractalFieldResult<T> = Result<T, FractalFieldError>;

/// Errors emitted while synthesising fractal fields.
#[derive(Debug, Error)]
pub enum FractalFieldError {
    #[error("log_step must be positive and finite")]
    InvalidLogStep,
    #[error("log_start must be finite")]
    InvalidLogStart,
    #[error("at least one lattice sample is required")]
    EmptyLattice,
    #[error("octaves must be at least 1")]
    InvalidOctaves,
    #[error("lacunarity must be > 1.0")]
    InvalidLacunarity,
    #[error("gain must be positive")]
    InvalidGain,
    #[error("iteration count must be at least 1")]
    InvalidIterations,
    #[error(transparent)]
    Mellin(#[from] MellinError),
}

/// Deterministic fractal field synthesiser over the Z-lattice.
#[derive(Clone, Debug)]
pub struct FractalFieldGenerator {
    octaves: u32,
    lacunarity: f32,
    gain: f32,
    iterations: u32,
}

impl FractalFieldGenerator {
    /// Creates a new generator with the provided hyper-parameters.
    pub fn new(
        octaves: u32,
        lacunarity: f32,
        gain: f32,
        iterations: u32,
    ) -> FractalFieldResult<Self> {
        if octaves == 0 {
            return Err(FractalFieldError::InvalidOctaves);
        }
        if !(lacunarity.is_finite() && lacunarity > 1.0) {
            return Err(FractalFieldError::InvalidLacunarity);
        }
        if !(gain.is_finite() && gain > 0.0) {
            return Err(FractalFieldError::InvalidGain);
        }
        if iterations == 0 {
            return Err(FractalFieldError::InvalidIterations);
        }
        Ok(Self {
            octaves,
            lacunarity,
            gain,
            iterations,
        })
    }

    /// Builds the branching field sampled on a log-lattice.
    pub fn branching_field(
        &self,
        log_start: Scalar,
        log_step: Scalar,
        len: usize,
    ) -> FractalFieldResult<Vec<ComplexScalar>> {
        if !(log_start.is_finite()) {
            return Err(FractalFieldError::InvalidLogStart);
        }
        if !(log_step.is_finite() && log_step > 0.0) {
            return Err(FractalFieldError::InvalidLogStep);
        }
        if len == 0 {
            return Err(FractalFieldError::EmptyLattice);
        }
        let mut field = Vec::with_capacity(len);
        for i in 0..len {
            let coord = log_start + log_step * i as Scalar;
            field.push(self.sample(coord));
        }
        Ok(field)
    }

    /// Generates a Mellin grid seeded with a fractal branching field.
    pub fn spawn_grid(
        &self,
        log_start: Scalar,
        log_step: Scalar,
        len: usize,
    ) -> FractalFieldResult<MellinLogGrid> {
        let samples = self.branching_field(log_start, log_step, len)?;
        Ok(MellinLogGrid::new(log_start, log_step, samples)?)
    }

    /// Adds a fractal branching field on top of an existing Mellin grid.
    pub fn weave_with_grid(&self, base: &MellinLogGrid) -> FractalFieldResult<MellinLogGrid> {
        let mut samples = base.samples().to_vec();
        let branch = self.branching_field(base.log_start(), base.log_step(), base.len())?;
        for (sample, perturbation) in samples.iter_mut().zip(branch.iter()) {
            *sample += *perturbation;
        }
        Ok(MellinLogGrid::new(
            base.log_start(),
            base.log_step(),
            samples,
        )?)
    }

    /// Returns the configured number of octaves.
    pub fn octaves(&self) -> u32 {
        self.octaves
    }

    /// Lacunarity parameter used by the generator.
    pub fn lacunarity(&self) -> f32 {
        self.lacunarity
    }

    /// Gain applied between octaves.
    pub fn gain(&self) -> f32 {
        self.gain
    }

    /// Iteration count used for the Mandelbrot-style loop.
    pub fn iterations(&self) -> u32 {
        self.iterations
    }

    fn sample(&self, coord: Scalar) -> ComplexScalar {
        let mut frequency = 1.0f32;
        let mut amplitude = 1.0f32;
        let mut total = ComplexScalar::new(0.0, 0.0);
        let mut normaliser = 0.0f32;

        for _ in 0..self.octaves {
            let angle = coord * frequency;
            let c = ComplexScalar::new(angle.cos(), angle.sin());
            let mut z = ComplexScalar::new(0.0, 0.0);
            let mut escape_it = self.iterations;
            for iter in 0..self.iterations {
                z = z * z + c;
                if z.norm_sqr() > 4.0 {
                    escape_it = iter + 1;
                    break;
                }
            }
            let escape = escape_it as f32 / self.iterations as f32;
            let envelope = (-0.5 * (coord * frequency).powi(2)).exp();
            let contribution = c * (escape * envelope * amplitude);
            total += contribution;
            normaliser += amplitude;
            frequency *= self.lacunarity;
            amplitude *= self.gain;
        }

        if normaliser > 0.0 {
            total * (1.0 / normaliser)
        } else {
            total
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_branching_field() {
        let generator = FractalFieldGenerator::new(3, 2.0, 0.5, 16).unwrap();
        let field = generator.branching_field(-2.0, 0.25, 12).unwrap();
        assert_eq!(field.len(), 12);
        assert!(field.iter().any(|c| c.re.abs() > 0.0));
        assert!(field.iter().any(|c| c.im.abs() > 0.0));
    }

    #[test]
    fn weaves_with_existing_grid() {
        let base = MellinLogGrid::from_function(-1.0, 0.2, 6, |x| {
            let amp = (x.exp()).sin();
            ComplexScalar::new(amp, 0.0)
        })
        .unwrap();
        let generator = FractalFieldGenerator::new(2, 2.0, 0.6, 12).unwrap();
        let woven = generator.weave_with_grid(&base).unwrap();
        assert_eq!(woven.len(), base.len());
        let diff: Vec<_> = woven
            .samples()
            .iter()
            .zip(base.samples().iter())
            .map(|(lhs, rhs)| (*lhs - *rhs).norm())
            .collect();
        assert!(diff.iter().any(|v| *v > 0.0));
    }
}
