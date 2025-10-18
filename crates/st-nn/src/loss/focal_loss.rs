// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::Loss;
use crate::{PureResult, Tensor};
use st_tensor::TensorError;

#[derive(Debug, Clone, Copy)]
pub struct FocalLoss {
    pub alpha: f32,
    pub gamma: f32,
    epsilon: f32,
}

impl FocalLoss {
    pub fn new(alpha: f32, gamma: f32) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 1.0),
            gamma: gamma.max(0.0),
            epsilon: 1e-6,
        }
    }

    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon.max(1e-9);
        self
    }
}

impl Loss for FocalLoss {
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
        for (pred, tgt_raw) in prediction.data().iter().zip(target.data().iter()) {
            let tgt = tgt_raw.clamp(0.0, 1.0);
            let prob = pred.clamp(self.epsilon, 1.0 - self.epsilon);
            let pt = if tgt > 0.5 { prob } else { 1.0 - prob };
            let alpha = if tgt > 0.5 {
                self.alpha
            } else {
                1.0 - self.alpha
            };
            sum += -alpha * (1.0 - pt).powf(self.gamma) * pt.ln();
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
        for (pred, tgt_raw) in prediction.data().iter().zip(target.data().iter()) {
            let tgt = tgt_raw.clamp(0.0, 1.0);
            let prob = pred.clamp(self.epsilon, 1.0 - self.epsilon);
            let alpha = if tgt > 0.5 {
                self.alpha
            } else {
                1.0 - self.alpha
            };
            let pt = if tgt > 0.5 { prob } else { 1.0 - prob };
            let focal = (1.0 - pt).powf(self.gamma);
            let term = focal * (self.gamma * pt.ln() * (pt - 1.0) + 1.0);
            let sign = if tgt > 0.5 { -1.0 } else { 1.0 };
            grad.push(alpha * term * sign * inv);
        }
        Tensor::from_vec(rows, cols, grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn focal_loss_computes_value() {
        let mut loss = FocalLoss::new(0.25, 2.0);
        let prediction = Tensor::from_vec(1, 2, vec![0.9, 0.2]).unwrap();
        let target = Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap();
        let value = loss.forward(&prediction, &target).unwrap();
        assert!(value.data()[0] > 0.0);
    }
}
