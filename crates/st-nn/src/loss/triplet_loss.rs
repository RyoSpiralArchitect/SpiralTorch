// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::Loss;
use crate::{PureResult, Tensor};
use st_tensor::TensorError;

#[derive(Debug, Clone, Copy)]
pub struct TripletLoss {
    pub margin: f32,
}

impl TripletLoss {
    pub fn new(margin: f32) -> Self {
        Self {
            margin: margin.max(0.0),
        }
    }
}

impl Loss for TripletLoss {
    fn forward(&mut self, prediction: &Tensor, target: &Tensor) -> PureResult<Tensor> {
        if prediction.shape() != target.shape() {
            return Err(TensorError::ShapeMismatch {
                left: prediction.shape(),
                right: target.shape(),
            });
        }
        let (rows, cols) = prediction.shape();
        if cols < 2 {
            return Err(TensorError::InvalidDimensions { rows, cols });
        }
        let mut sum = 0.0f32;
        let inv = 1.0 / rows.max(1) as f32;
        for row in 0..rows {
            let offset = row * cols;
            let pos = prediction.data()[offset];
            let neg = prediction.data()[offset + 1];
            let weight = target.data()[offset].abs().max(1.0);
            let loss = (pos - neg + self.margin).max(0.0);
            sum += loss * weight;
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
        if cols < 2 {
            return Err(TensorError::InvalidDimensions { rows, cols });
        }
        let mut grad = vec![0.0f32; rows * cols];
        let inv = 1.0 / rows.max(1) as f32;
        for row in 0..rows {
            let offset = row * cols;
            let pos = prediction.data()[offset];
            let neg = prediction.data()[offset + 1];
            let weight = target.data()[offset].abs().max(1.0);
            let diff = pos - neg + self.margin;
            if diff > 0.0 {
                grad[offset] = weight * inv;
                grad[offset + 1] = -weight * inv;
            }
        }
        Tensor::from_vec(rows, cols, grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triplet_loss_applies_margin() {
        let mut loss = TripletLoss::new(1.0);
        let prediction = Tensor::from_vec(2, 2, vec![0.5, 0.2, 0.3, 0.6]).unwrap();
        let target = Tensor::from_vec(2, 2, vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let value = loss.forward(&prediction, &target).unwrap();
        assert!(value.data()[0] >= 0.0);
    }
}
