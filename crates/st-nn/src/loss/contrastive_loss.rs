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
pub struct ContrastiveLoss {
    pub margin: f32,
}

impl ContrastiveLoss {
    pub fn new(margin: f32) -> Self {
        let margin = if margin.is_finite() {
            margin.max(0.0)
        } else {
            0.0
        };
        Self { margin }
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
        validate_loss_inputs(prediction, target)?;
        let (rows, cols) = prediction.shape();
        if rows == 0 || cols == 0 {
            emit_loss_backend_meta("contrastive_loss_forward", prediction, (1, 1), "mean");
            return Tensor::from_vec(1, 1, vec![0.0]);
        }
        let inv = 1.0 / (rows * cols) as f32;
        let mut terms = Vec::with_capacity(rows * cols);
        for (dist, label_raw) in prediction.data().iter().zip(target.data().iter()) {
            let label = label_raw.clamp(0.0, 1.0);
            let positive = if label > 0.0 {
                let distance_squared =
                    checked_loss_value("contrastive_distance_squared", dist.powi(2))?;
                checked_loss_value("contrastive_positive_term", label * distance_squared)?
            } else {
                0.0
            };
            let negative_label = 1.0 - label;
            let negative = if negative_label > 0.0 {
                let margin_delta =
                    checked_loss_value("contrastive_margin_delta", self.margin - dist)?;
                if margin_delta > 0.0 {
                    let negative_distance =
                        checked_loss_value("contrastive_negative_distance", margin_delta.powi(2))?;
                    checked_loss_value(
                        "contrastive_negative_term",
                        negative_label * negative_distance,
                    )?
                } else {
                    0.0
                }
            } else {
                0.0
            };
            terms.push(checked_loss_value(
                "contrastive_loss_term",
                positive + negative,
            )?);
        }
        let terms = Tensor::from_vec(rows, cols, terms)?;
        let loss = reduce_loss_terms(
            &terms,
            inv,
            "contrastive_loss_term",
            "contrastive_loss_column_mean",
            "contrastive_loss_value",
        )?;
        emit_loss_backend_meta("contrastive_loss_forward", prediction, (1, 1), "mean");
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
        if rows == 0 || cols == 0 {
            emit_loss_backend_meta(
                "contrastive_loss_backward",
                prediction,
                (rows, cols),
                "mean",
            );
            return Tensor::zeros(rows, cols);
        }
        let mut grad = Vec::with_capacity(rows * cols);
        for (dist, label_raw) in prediction.data().iter().zip(target.data().iter()) {
            let label = label_raw.clamp(0.0, 1.0);
            let positive_grad = if label > 0.0 {
                let coefficient = checked_loss_value("contrastive_positive_gradient", 2.0 * label)?;
                checked_loss_value("contrastive_positive_gradient", coefficient * dist)?
            } else {
                0.0
            };
            let negative_label = 1.0 - label;
            let negative_grad = if negative_label > 0.0 {
                let delta = checked_loss_value("contrastive_margin_delta", self.margin - dist)?;
                if delta > 0.0 {
                    let coefficient =
                        checked_loss_value("contrastive_negative_gradient", -2.0 * negative_label)?;
                    checked_loss_value("contrastive_negative_gradient", coefficient * delta)?
                } else {
                    0.0
                }
            } else {
                0.0
            };
            grad.push(checked_loss_value(
                "contrastive_loss_gradient",
                positive_grad + negative_grad,
            )?);
        }
        let grad = Tensor::from_vec(rows, cols, grad)?;
        validate_loss_tensor("contrastive_loss_gradient", &grad)?;
        let grad = relabel_loss_non_finite(
            grad.scale_with_backend(
                1.0 / (rows * cols) as f32,
                current_tensor_util_backend_for_values(grad.data().len()),
            ),
            "contrastive_loss_gradient",
        )?;
        validate_loss_tensor("contrastive_loss_gradient", &grad)?;
        emit_loss_backend_meta(
            "contrastive_loss_backward",
            prediction,
            (rows, cols),
            "mean",
        );
        Ok(grad)
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

    #[test]
    fn contrastive_loss_zero_sized_prediction_returns_zero_loss_and_empty_grad() {
        let mut loss = ContrastiveLoss::new(1.0);
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
    fn contrastive_loss_sanitizes_non_finite_margin() {
        let loss = ContrastiveLoss::new(f32::NAN);

        assert_eq!(loss.margin, 0.0);
    }

    #[test]
    fn contrastive_loss_rejects_overflowing_distance_square() {
        let mut loss = ContrastiveLoss::new(1.0);
        let prediction = Tensor::from_vec(1, 1, vec![f32::MAX]).unwrap();
        let target = Tensor::from_vec(1, 1, vec![1.0]).unwrap();

        let err = loss.forward(&prediction, &target).unwrap_err();

        assert!(matches!(
            err,
            st_tensor::TensorError::NonFiniteValue {
                label: "contrastive_distance_squared",
                value,
            } if value.is_infinite()
        ));
    }

    #[test]
    fn contrastive_loss_ignores_margin_satisfied_negative_large_distance() {
        let mut loss = ContrastiveLoss::new(1.0);
        let prediction = Tensor::from_vec(1, 1, vec![f32::MAX]).unwrap();
        let target = Tensor::from_vec(1, 1, vec![0.0]).unwrap();

        let value = loss.forward(&prediction, &target).unwrap();
        assert_eq!(value.data()[0], 0.0);

        let grad = loss.backward(&prediction, &target).unwrap();
        assert_eq!(grad.data()[0], 0.0);
    }
}
