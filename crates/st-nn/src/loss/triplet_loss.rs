// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::{
    checked_loss_value, emit_loss_backend_meta, reduce_loss_terms, relabel_loss_non_finite,
    validate_loss_inputs, validate_loss_tensor, Loss,
};
use crate::execution::current_tensor_util_backend_for_values;
use crate::{PureResult, Tensor};
use st_tensor::TensorError;

#[derive(Debug, Clone, Copy)]
pub struct TripletLoss {
    pub margin: f32,
}

impl TripletLoss {
    pub fn new(margin: f32) -> Self {
        let margin = if margin.is_finite() {
            margin.max(0.0)
        } else {
            0.0
        };
        Self { margin }
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
        validate_loss_inputs(prediction, target)?;
        let (rows, cols) = prediction.shape();
        if cols < 2 {
            return Err(TensorError::InvalidDimensions { rows, cols });
        }
        let inv = 1.0 / rows.max(1) as f32;
        let mut terms = Vec::with_capacity(rows);
        for row in 0..rows {
            let offset = row * cols;
            let pos = prediction.data()[offset];
            let neg = prediction.data()[offset + 1];
            let weight = target.data()[offset].abs().max(1.0);
            let diff = checked_loss_value("triplet_pair_delta", pos - neg)?;
            let margin_score = checked_loss_value("triplet_margin_score", diff + self.margin)?;
            let loss = margin_score.max(0.0);
            terms.push(checked_loss_value("triplet_loss_term", loss * weight)?);
        }
        let terms = Tensor::from_vec(rows, 1, terms)?;
        let loss = reduce_loss_terms(
            &terms,
            inv,
            "triplet_loss_term",
            "triplet_loss_column_mean",
            "triplet_loss_value",
        )?;
        emit_loss_backend_meta("triplet_loss_forward", prediction, (1, 1), "row_mean");
        Tensor::from_vec(1, 1, vec![loss])
    }

    fn backward(&mut self, prediction: &Tensor, target: &Tensor) -> PureResult<Tensor> {
        if prediction.shape() != target.shape() {
            return Err(TensorError::ShapeMismatch {
                left: prediction.shape(),
                right: target.shape(),
            });
        }
        validate_loss_inputs(prediction, target)?;
        let (rows, cols) = prediction.shape();
        if cols < 2 {
            return Err(TensorError::InvalidDimensions { rows, cols });
        }
        let mut grad = vec![0.0f32; rows * cols];
        for row in 0..rows {
            let offset = row * cols;
            let pos = prediction.data()[offset];
            let neg = prediction.data()[offset + 1];
            let weight = target.data()[offset].abs().max(1.0);
            let pair_delta = checked_loss_value("triplet_pair_delta", pos - neg)?;
            let margin_score =
                checked_loss_value("triplet_margin_score", pair_delta + self.margin)?;
            if margin_score > 0.0 {
                grad[offset] = checked_loss_value("triplet_positive_gradient", weight)?;
                grad[offset + 1] = checked_loss_value("triplet_negative_gradient", -weight)?;
            }
        }
        let grad = Tensor::from_vec(rows, cols, grad)?;
        validate_loss_tensor("triplet_loss_gradient", &grad)?;
        let grad = relabel_loss_non_finite(
            grad.scale_with_backend(
                1.0 / rows.max(1) as f32,
                current_tensor_util_backend_for_values(grad.data().len()),
            ),
            "triplet_loss_gradient",
        )?;
        validate_loss_tensor("triplet_loss_gradient", &grad)?;
        emit_loss_backend_meta(
            "triplet_loss_backward",
            prediction,
            (rows, cols),
            "row_mean",
        );
        Ok(grad)
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

    #[test]
    fn triplet_loss_zero_sized_prediction_returns_zero_loss_and_empty_grad() {
        let mut loss = TripletLoss::new(1.0);
        let prediction = Tensor::from_vec(0, 2, Vec::new()).unwrap();
        let target = Tensor::from_vec(0, 2, Vec::new()).unwrap();

        let value = loss.forward(&prediction, &target).unwrap();
        assert_eq!(value.shape(), (1, 1));
        assert_eq!(value.data()[0], 0.0);

        let grad = loss.backward(&prediction, &target).unwrap();
        assert_eq!(grad.shape(), (0, 2));
        assert!(grad.data().is_empty());
    }

    #[test]
    fn triplet_loss_sanitizes_non_finite_margin() {
        let loss = TripletLoss::new(f32::INFINITY);

        assert_eq!(loss.margin, 0.0);
    }

    #[test]
    fn triplet_loss_rejects_overflowing_pair_delta() {
        let mut loss = TripletLoss::new(1.0);
        let prediction = Tensor::from_vec(1, 2, vec![f32::MAX, -f32::MAX]).unwrap();
        let target = Tensor::from_vec(1, 2, vec![1.0, 1.0]).unwrap();

        let err = loss.forward(&prediction, &target).unwrap_err();

        assert!(matches!(
            err,
            st_tensor::TensorError::NonFiniteValue {
                label: "triplet_pair_delta",
                value,
            } if value.is_infinite()
        ));
    }
}
