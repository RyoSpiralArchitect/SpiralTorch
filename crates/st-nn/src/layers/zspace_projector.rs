// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 SpiralTorch contributors
// Proposed replacement for crates/st-nn/src/layers/zspace_projector.rs
//
// Goal:
//  - Provide a robust Z-space Projector layer that relies on st-frac's
//    numerically stable Z-transform (Horner) and non-finite checks.
//  - Keep the API minimal and self-contained, easy to wire into existing NN graphs.
//
// NOTE: You may need to adjust the import paths depending on the workspace layout.

use st_frac::mellin_types::{Scalar, ComplexScalar};
use st_frac::zspace::{
    trapezoidal_weights,
    mellin_log_lattice_prefactor,
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

/// ZSpaceProjector projects a log-uniform Mellin sampling into a bank of
/// complex Z-points (e.g. along a vertical line in s = σ + iω).
pub struct ZSpaceProjector {
    /// log(t) start for the sampling grid
    pub log_start: Scalar,
    /// constant log-step size
    pub log_step: Scalar,
    /// target points in the Z-plane (typically exp(s * log_step))
    pub z_points: Vec<ComplexScalar>,
}

impl ZSpaceProjector {
    pub fn new(log_start: Scalar, log_step: Scalar, z_points: Vec<ComplexScalar>) -> ProjectorResult<Self> {
        if z_points.is_empty() { return Err(ProjectorError::EmptyZ); }
        Ok(Self { log_start, log_step, z_points })
    }

    /// Project one channel (single series) of complex samples to all z_points.
    ///
    /// `samples` are Mellin-sampled values on t_k = log_start + k * log_step.
    pub fn project_series(&self, samples: &[ComplexScalar]) -> ProjectorResult<Vec<ComplexScalar>> {
        if samples.is_empty() { return Err(ProjectorError::EmptySamples); }

        // 1) Prefactor for Mellin->Z reparam is constant for a fixed s; here we just
        //    compute the series using Z-transform primitives which already include the
        //    correct weighting and stability (Horner).
        let weights = trapezoidal_weights(samples.len())?;
        let weighted = prepare_weighted_series(samples, &weights)?;

        // 2) Evaluate series for all provided z points.
        let out = evaluate_weighted_series_many(&weighted, &self.z_points)?;
        Ok(out)
    }

    /// Project a batch: samples laid out as [batch][k]
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
