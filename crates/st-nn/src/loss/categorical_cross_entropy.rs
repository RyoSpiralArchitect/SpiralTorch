// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::Loss;
use crate::{PureResult, Tensor};
use st_tensor::TensorError;

/// Multi-class cross entropy loss (categorical) operating on probability
/// predictions and one-hot targets.
#[derive(Debug, Clone, Copy)]
pub struct CategoricalCrossEntropy {
    epsilon: f32,
}

impl Default for CategoricalCrossEntropy {
    fn default() -> Self {
        Self { epsilon: 1e-9 }
    }
}

impl CategoricalCrossEntropy {
    /// Creates a categorical cross entropy loss with a default numerical
    /// stability epsilon.
    pub fn new() -> Self {
        Self::default()
    }

    /// Overrides the epsilon used to clamp probabilities.
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon.max(1e-12);
        self
    }

    /// Returns the epsilon used for clamping.
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }
}

impl Loss for CategoricalCrossEntropy {
    fn forward(&mut self, prediction: &Tensor, target: &Tensor) -> PureResult<Tensor> {
        if prediction.shape() != target.shape() {
            return Err(TensorError::ShapeMismatch {
                left: prediction.shape(),
                right: target.shape(),
            });
        }
        let (rows, cols) = prediction.shape();
        if rows == 0 || cols == 0 {
            return Tensor::from_vec(1, 1, vec![0.0]);
        }
        let mut sum = 0.0f32;
        for (pred, tgt) in prediction.data().iter().zip(target.data().iter()) {
            let p = pred.clamp(self.epsilon, 1.0);
            sum += -tgt * p.ln();
        }
        Tensor::from_vec(1, 1, vec![sum / rows as f32])
    }

    fn backward(&mut self, prediction: &Tensor, target: &Tensor) -> PureResult<Tensor> {
        if prediction.shape() != target.shape() {
            return Err(TensorError::ShapeMismatch {
                left: prediction.shape(),
                right: target.shape(),
            });
        }
        let (rows, cols) = prediction.shape();
        if rows == 0 || cols == 0 {
            return Tensor::zeros(rows, cols);
        }
        let inv_batch = 1.0 / rows as f32;
        let mut grad = Vec::with_capacity(rows * cols);
        for (pred, tgt) in prediction.data().iter().zip(target.data().iter()) {
            let p = pred.clamp(self.epsilon, 1.0);
            grad.push(-tgt / p * inv_batch);
        }
        Tensor::from_vec(rows, cols, grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categorical_ce_matches_manual() {
        let mut loss = CategoricalCrossEntropy::new().with_epsilon(1e-9);
        let prediction = Tensor::from_vec(2, 3, vec![0.1, 0.6, 0.3, 0.8, 0.1, 0.1]).unwrap();
        let target = Tensor::from_vec(2, 3, vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
        let value = loss.forward(&prediction, &target).unwrap();
        let expected = (-0.6_f32.ln() - 0.8_f32.ln()) / 2.0;
        assert!((value.data()[0] - expected).abs() < 1e-6);

        let grad = loss.backward(&prediction, &target).unwrap();
        assert_eq!(grad.shape(), (2, 3));
        assert!((grad.data()[1] + 1.0 / (0.6 * 2.0)).abs() < 1e-6);
        assert!((grad.data()[3] + 1.0 / (0.8 * 2.0)).abs() < 1e-6);
    }
}

