// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Covariance reconstruction head used by the Z-RBA module.

use crate::{PureResult, Tensor, TensorError};
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use std::cmp::Ordering;

/// Telemetry exposing key PSD statistics.
#[derive(Clone, Debug)]
pub struct CovHeadTelemetry {
    pub min_eigenvalue: f32,
    pub max_eigenvalue: f32,
    pub condition_number: f32,
    pub stabiliser: f32,
}

/// Covariance output bundle.
#[derive(Clone, Debug)]
pub struct CovHeadOutput {
    pub covariance: Tensor,
    pub telemetry: CovHeadTelemetry,
}

/// Covariance reconstruction using low-rank factors + diagonal correction.
#[derive(Debug)]
pub struct CovHead {
    rank: usize,
    stabiliser: f32,
}

impl CovHead {
    pub fn new(rank: usize) -> Self {
        Self {
            rank: rank.max(1),
            stabiliser: 1e-4,
        }
    }

    pub fn forward(&self, mu: &Tensor, sigma_diag: &Tensor) -> PureResult<CovHeadOutput> {
        let (rows, cols) = mu.shape();
        if sigma_diag.shape() != (rows, cols) {
            return Err(TensorError::ShapeMismatch {
                left: sigma_diag.shape(),
                right: (rows, cols),
            });
        }
        if rows == 0 {
            return Err(TensorError::EmptyInput("cov_head::forward"));
        }

        let mu_data = mu.data();
        let sigma_data = sigma_diag.data();
        let mut mean = vec![0.0f32; cols];
        for r in 0..rows {
            for c in 0..cols {
                mean[c] += mu_data[r * cols + c];
            }
        }
        for value in &mut mean {
            *value /= rows as f32;
        }

        let mut cov = vec![0.0f32; cols * cols];
        for i in 0..cols {
            for j in 0..cols {
                let mut acc = 0.0f32;
                for r in 0..rows {
                    let xi = mu_data[r * cols + i] - mean[i];
                    let xj = mu_data[r * cols + j] - mean[j];
                    acc += xi * xj;
                }
                cov[i * cols + j] = acc / rows as f32;
            }
        }

        let mut diag_variance = vec![0.0f32; cols];
        for r in 0..rows {
            for c in 0..cols {
                diag_variance[c] += sigma_data[r * cols + c];
            }
        }
        for value in &mut diag_variance {
            *value = (*value / rows as f32).max(1e-6);
        }

        let sample_cov = DMatrix::from_vec(cols, cols, cov);
        let truncated = self.low_rank_projection(sample_cov.clone());
        let diag = DMatrix::from_diagonal(&DVector::from_vec(diag_variance));
        let combined = truncated + diag;
        let (psd, telemetry) = self.make_psd(combined);
        let tensor = Tensor::from_vec(cols, cols, psd.iter().copied().collect())?;
        Ok(CovHeadOutput {
            covariance: tensor,
            telemetry,
        })
    }

    fn low_rank_projection(&self, matrix: DMatrix<f32>) -> DMatrix<f32> {
        let eigen = SymmetricEigen::new(matrix.clone());
        let mut pairs: Vec<(usize, f32)> = eigen
            .eigenvalues
            .iter()
            .enumerate()
            .map(|(idx, &value)| (idx, value))
            .collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        let mut result = DMatrix::zeros(matrix.nrows(), matrix.ncols());
        for (rank, (idx, value)) in pairs.into_iter().enumerate() {
            if rank >= self.rank {
                break;
            }
            let eigenvector = eigen.eigenvectors.column(idx);
            let value = value.max(0.0);
            result += value * (&eigenvector * eigenvector.transpose());
        }
        result
    }

    fn make_psd(&self, matrix: DMatrix<f32>) -> (DMatrix<f32>, CovHeadTelemetry) {
        let eigen = SymmetricEigen::new(matrix.clone());
        let mut adjusted = eigen.eigenvalues.clone();
        let mut min_eigen = f32::INFINITY;
        let mut max_eigen = f32::NEG_INFINITY;
        for value in adjusted.iter_mut() {
            min_eigen = min_eigen.min(*value);
            max_eigen = max_eigen.max(*value);
            if *value < self.stabiliser {
                *value = self.stabiliser;
            }
        }
        let diag = DMatrix::from_diagonal(&adjusted);
        let psd = &eigen.eigenvectors * diag * eigen.eigenvectors.transpose();
        let mut min_adj = f32::INFINITY;
        let mut max_adj = f32::NEG_INFINITY;
        for value in adjusted.iter() {
            min_adj = min_adj.min(*value);
            max_adj = max_adj.max(*value);
        }
        let condition = if min_adj > 0.0 {
            max_adj / min_adj
        } else {
            f32::INFINITY
        };
        (
            psd,
            CovHeadTelemetry {
                min_eigenvalue: min_adj,
                max_eigenvalue: max_adj,
                condition_number: condition,
                stabiliser: self.stabiliser,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covariance_head_outputs_psd() {
        let mu = Tensor::from_vec(
            3,
            4,
            vec![
                0.1, 0.2, 0.3, 0.4, 0.0, -0.1, 0.2, 0.3, -0.2, 0.1, 0.0, -0.3,
            ],
        )
        .unwrap();
        let sigma = Tensor::from_vec(
            3,
            4,
            vec![
                0.05, 0.02, 0.03, 0.01, 0.04, 0.03, 0.02, 0.01, 0.06, 0.05, 0.04, 0.03,
            ],
        )
        .unwrap();
        let head = CovHead::new(2);
        let output = head.forward(&mu, &sigma).unwrap();
        assert_eq!(output.covariance.shape(), (4, 4));
        assert!(output.telemetry.min_eigenvalue > 0.0);
        assert!(output.telemetry.condition_number.is_finite());
    }
}
