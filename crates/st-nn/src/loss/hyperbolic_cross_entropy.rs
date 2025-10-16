// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::Loss;
use crate::{PureResult, Tensor};
use st_tensor::TensorError;

/// Cross entropy defined on a hyperbolic Poincaré ball.
#[derive(Debug, Clone, Copy)]
pub struct HyperbolicCrossEntropy {
    curvature: f32,
    epsilon: f32,
}

impl HyperbolicCrossEntropy {
    /// Creates the loss with the provided (negative) curvature and the default
    /// stability epsilon.
    pub fn new(curvature: f32) -> PureResult<Self> {
        Self::with_epsilon(curvature, 1e-5)
    }

    /// Creates the loss with explicit epsilon used to clamp targets.
    pub fn with_epsilon(curvature: f32, epsilon: f32) -> PureResult<Self> {
        if curvature >= 0.0 {
            return Err(TensorError::NonHyperbolicCurvature { curvature });
        }
        Ok(Self { curvature, epsilon })
    }

    /// Returns the hyperbolic curvature.
    pub fn curvature(&self) -> f32 {
        self.curvature
    }

    /// Returns the epsilon used for clamping targets.
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }
}

impl Loss for HyperbolicCrossEntropy {
    fn forward(&mut self, prediction: &Tensor, target: &Tensor) -> PureResult<Tensor> {
        if prediction.shape() != target.shape() {
            return Err(TensorError::ShapeMismatch {
                left: prediction.shape(),
                right: target.shape(),
            });
        }
        let (rows, cols) = prediction.shape();
        let scale = (-self.curvature).sqrt();
        let mut sum = 0.0f32;
        let count = (rows * cols) as f32;
        for (pred, tgt_raw) in prediction.data().iter().zip(target.data().iter()) {
            let tgt = tgt_raw.clamp(self.epsilon, 1.0 - self.epsilon);
            let scaled = pred * scale;
            let log_sigmoid_pos = -((1.0 + (-scaled).exp()).ln());
            let log_sigmoid_neg = -((1.0 + scaled.exp()).ln());
            sum += -tgt * log_sigmoid_pos - (1.0 - tgt) * log_sigmoid_neg;
        }
        Tensor::from_vec(1, 1, vec![sum / count])
    }

    fn backward(&mut self, prediction: &Tensor, target: &Tensor) -> PureResult<Tensor> {
        if prediction.shape() != target.shape() {
            return Err(TensorError::ShapeMismatch {
                left: prediction.shape(),
                right: target.shape(),
            });
        }
        let (rows, cols) = prediction.shape();
        let scale = (-self.curvature).sqrt();
        let inv = 1.0 / (rows * cols) as f32;
        let mut data = Vec::with_capacity(rows * cols);
        for (pred, tgt_raw) in prediction.data().iter().zip(target.data().iter()) {
            let tgt = tgt_raw.clamp(self.epsilon, 1.0 - self.epsilon);
            let scaled = pred * scale;
            let sigmoid = 1.0 / (1.0 + (-scaled).exp());
            data.push(scale * (sigmoid - tgt) * inv);
        }
        Tensor::from_vec(rows, cols, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hyperbolic_cross_entropy_matches_binary_ce_shape() {
        let mut loss = HyperbolicCrossEntropy::new(-1.0).unwrap();
        let prediction = Tensor::from_vec(1, 2, vec![0.5, -0.5]).unwrap();
        let target = Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap();
        let value = loss.forward(&prediction, &target).unwrap();
        assert_eq!(value.shape(), (1, 1));

        let grad = loss.backward(&prediction, &target).unwrap();
        assert_eq!(grad.shape(), (1, 2));
        assert!(grad.data()[0].is_finite());
        assert!(grad.data()[1].is_finite());
    }
}
