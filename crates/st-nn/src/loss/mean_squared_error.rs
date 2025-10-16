// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::Loss;
use crate::{PureResult, Tensor};
use st_tensor::TensorError;

/// Classic mean squared error loss with mean reduction.
#[derive(Debug, Default, Clone, Copy)]
pub struct MeanSquaredError;

impl MeanSquaredError {
    /// Creates a new mean squared error loss instance.
    pub fn new() -> Self {
        Self
    }
}

impl Loss for MeanSquaredError {
    fn forward(&mut self, prediction: &Tensor, target: &Tensor) -> PureResult<Tensor> {
        if prediction.shape() != target.shape() {
            return Err(TensorError::ShapeMismatch {
                left: prediction.shape(),
                right: target.shape(),
            });
        }
        let (rows, cols) = prediction.shape();
        let mut sum = 0.0f32;
        for (pred, tgt) in prediction.data().iter().zip(target.data().iter()) {
            let diff = pred - tgt;
            sum += diff * diff;
        }
        let mean = sum / (rows * cols) as f32;
        Tensor::from_vec(1, 1, vec![mean])
    }

    fn backward(&mut self, prediction: &Tensor, target: &Tensor) -> PureResult<Tensor> {
        if prediction.shape() != target.shape() {
            return Err(TensorError::ShapeMismatch {
                left: prediction.shape(),
                right: target.shape(),
            });
        }
        let (rows, cols) = prediction.shape();
        let inv = 2.0f32 / (rows * cols) as f32;
        let mut data = Vec::with_capacity(rows * cols);
        for (pred, tgt) in prediction.data().iter().zip(target.data().iter()) {
            data.push((pred - tgt) * inv);
        }
        Tensor::from_vec(rows, cols, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mse_forward_backward() {
        let mut loss = MeanSquaredError::new();
        let prediction = Tensor::from_vec(1, 3, vec![0.5, -0.5, 1.0]).unwrap();
        let target = Tensor::from_vec(1, 3, vec![0.0, 0.0, 1.5]).unwrap();
        let value = loss.forward(&prediction, &target).unwrap();
        assert!((value.data()[0] - 0.25).abs() < 1e-6);

        let grad = loss.backward(&prediction, &target).unwrap();
        assert_eq!(grad.data().len(), 3);
        assert!(grad.data()[0] > 0.0);
        assert!(grad.data()[1] < 0.0);
    }
}
