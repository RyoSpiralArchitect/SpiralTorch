// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::Loss;
use crate::{PureResult, Tensor};
use st_tensor::TensorError;

#[derive(Debug, Clone, Copy)]
pub struct ContrastiveLoss {
    pub margin: f32,
}

impl ContrastiveLoss {
    pub fn new(margin: f32) -> Self {
        Self {
            margin: margin.max(0.0),
        }
    }
}

impl Loss for ContrastiveLoss {
    fn forward(&mut self, prediction: &Tensor, target: &Tensor) -> PureResult<Tensor> {
        if prediction.shape() != target.shape() {
            return Err(TensorError::ShapeMismatch {
                left: prediction.shape(),
                right: target.shape(),
            });
        }
        let mut sum = 0.0f32;
        let (rows, cols) = prediction.shape();
        let inv = 1.0 / (rows * cols) as f32;
        for (dist, label_raw) in prediction.data().iter().zip(target.data().iter()) {
            let label = label_raw.clamp(0.0, 1.0);
            let positive = label * dist.powi(2);
            let negative = (1.0 - label) * (self.margin - dist).max(0.0).powi(2);
            sum += positive + negative;
        }
        Tensor::from_vec(1, 1, vec![sum * inv])
    }

    fn backward(&mut self, prediction: &Tensor, target: &Tensor) -> PureResult<Tensor> {
        if prediction.shape() != target.shape() {
            return Err(TensorError::ShapeMismatch {
                left: prediction.shape(),
                right: target.shape(),
            });
        }
        let (rows, cols) = prediction.shape();
        let inv = 1.0 / (rows * cols) as f32;
        let mut grad = Vec::with_capacity(rows * cols);
        for (dist, label_raw) in prediction.data().iter().zip(target.data().iter()) {
            let label = label_raw.clamp(0.0, 1.0);
            let positive_grad = 2.0 * dist * label;
            let delta = self.margin - dist;
            let negative_grad = if delta > 0.0 {
                -2.0 * delta * (1.0 - label)
            } else {
                0.0
            };
            grad.push((positive_grad + negative_grad) * inv);
        }
        Tensor::from_vec(rows, cols, grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contrastive_loss_handles_positive_and_negative_pairs() {
        let mut loss = ContrastiveLoss::new(1.0);
        let prediction = Tensor::from_vec(1, 2, vec![0.2, 1.3]).unwrap();
        let target = Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap();
        let value = loss.forward(&prediction, &target).unwrap();
        assert!(value.data()[0] > 0.0);
    }
}
