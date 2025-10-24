// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 SpiralTorch contributors
// crates/st-nn/src/layers/zspace_projector.rs
//
// ZSpaceProjector
//  - Projects a log-uniform Mellin sampling (complex) into a bank of Z-plane points.
//  - Depends on st-frac's numerically-hardened Z-space primitives (Horner + finiteness checks).
//
// Public API:
//   - ZSpaceProjector::new(log_start, log_step, z_points)
//   - project_series(&self, samples: &[ComplexScalar]) -> Vec<ComplexScalar>
//   - project_batch(&self, batch: &[Vec<ComplexScalar>]) -> Vec<Vec<ComplexScalar>>
//
// NOTE: Adjust the `use` paths if your workspace exposes different crate names.

use st_frac::mellin_types::{Scalar, ComplexScalar};
use st_frac::zspace::{
    trapezoidal_weights,
    prepare_weighted_series,
    evaluate_weighted_series_many,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProjectorError {
    #[error("input samples must not be empty")]
    EmptySamples,
    #[error("z-points must not be empty")]
    EmptyZ,
    #[error("mismatched batch length: got {got}, expected {expected}")]
    BatchMismatch { got: usize, expected: usize },
    #[error(transparent)]
    Mellin(#[from] st_frac::mellin_types::MellinError),
    #[error(transparent)]
    ZSpace(#[from] st_frac::mellin_types::ZSpaceError),
}

pub type ProjectorResult<T> = Result<T, ProjectorError>;

/// Projects Mellin-sampled series onto Z-plane evaluation points.
pub struct ZSpaceProjector {
    pub log_start: Scalar,
    pub log_step: Scalar,
    pub z_points: Vec<ComplexScalar>,
}

impl ZSpaceProjector {
    pub fn new(log_start: Scalar, log_step: Scalar, z_points: Vec<ComplexScalar>) -> ProjectorResult<Self> {
        if z_points.is_empty() { return Err(ProjectorError::EmptyZ); }
        Ok(Self { log_start, log_step, z_points })
    }

    /// Project a single series (complex samples) to all target z-points.
    pub fn project_series(&self, samples: &[ComplexScalar]) -> ProjectorResult<Vec<ComplexScalar>> {
        if samples.is_empty() { return Err(ProjectorError::EmptySamples); }
        // Prepare weighted coefficients once (uses st-frac finiteness checks)
        let weights = trapezoidal_weights(samples.len())?;
        let weighted = prepare_weighted_series(samples, &weights)?;
        // Evaluate at all z points (Horner + checks inside)
        let out = evaluate_weighted_series_many(&weighted, &self.z_points)?;
        Ok(out)
    }

    /// Project a batch: layout = [batch][k]
    pub fn project_batch(&self, batch: &[Vec<ComplexScalar>]) -> ProjectorResult<Vec<Vec<ComplexScalar>>> {
        if batch.is_empty() { return Err(ProjectorError::EmptySamples); }
        let mut out = Vec::with_capacity(batch.len());
        for series in batch.iter() {
            out.push(self.project_series(series)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projector_runs_and_respects_sizes() {
        let log_start = -2.0_f32;
        let log_step = 0.5_f32;
        let z_points = vec![
            ComplexScalar::new(0.7, 0.0),
            ComplexScalar::new(0.4, -0.3),
            ComplexScalar::new(-0.2, 0.5),
        ];
        let p = ZSpaceProjector::new(log_start, log_step, z_points.clone()).unwrap();

        let samples = vec![
            ComplexScalar::new(0.2, 0.1),
            ComplexScalar::new(-0.4, 0.3),
            ComplexScalar::new(0.1, -0.2),
            ComplexScalar::new(0.0, 0.0),
        ];
        let y = p.project_series(&samples).unwrap();
        assert_eq!(y.len(), z_points.len());
    }
}
