// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::{
    binary_probability_epsilon, checked_loss_value, emit_loss_backend_meta, reduce_loss_terms,
    relabel_loss_non_finite, validate_loss_inputs, validate_loss_tensor, Loss,
};
use crate::execution::current_tensor_util_backend_for_values;
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
        let alpha = if alpha.is_finite() {
            alpha.clamp(0.0, 1.0)
        } else {
            0.25
        };
        let gamma = if gamma.is_finite() {
            gamma.max(0.0)
        } else {
            0.0
        };
        Self {
            alpha,
            gamma,
            epsilon: 1e-6,
        }
    }

    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = binary_probability_epsilon(epsilon, 1.0e-6);
        self
    }

    fn derivative_wrt_probability(&self, probability: f32, target: f32) -> PureResult<f32> {
        let prob = probability.clamp(self.epsilon, 1.0 - self.epsilon);
        let positive = target > 0.5;
        let pt = if positive { prob } else { 1.0 - prob };
        let alpha = if positive {
            self.alpha
        } else {
            1.0 - self.alpha
        };
        let one_minus_pt = 1.0 - pt;
        let focal = one_minus_pt.powf(self.gamma);
        let modulation_grad = if self.gamma <= f32::EPSILON {
            0.0
        } else {
            self.gamma * one_minus_pt.powf(self.gamma - 1.0) * pt.ln()
        };
        let d_loss_d_pt = alpha * (modulation_grad - focal / pt);
        let derivative = if positive { d_loss_d_pt } else { -d_loss_d_pt };
        checked_loss_value("focal_loss_gradient", derivative)
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
        validate_loss_inputs(prediction, target)?;
        let (rows, cols) = prediction.shape();
        if rows == 0 || cols == 0 {
            emit_loss_backend_meta("focal_loss_forward", prediction, (1, 1), "mean");
            return Tensor::from_vec(1, 1, vec![0.0]);
        }
        let inv = 1.0 / (rows * cols) as f32;
        let mut terms = Vec::with_capacity(rows * cols);
        for (pred, tgt_raw) in prediction.data().iter().zip(target.data().iter()) {
            let tgt = tgt_raw.clamp(0.0, 1.0);
            let prob = pred.clamp(self.epsilon, 1.0 - self.epsilon);
            let pt = if tgt > 0.5 { prob } else { 1.0 - prob };
            let alpha = if tgt > 0.5 {
                self.alpha
            } else {
                1.0 - self.alpha
            };
            let term = checked_loss_value(
                "focal_loss_term",
                -alpha * (1.0 - pt).powf(self.gamma) * pt.ln(),
            )?;
            terms.push(term);
        }
        let terms = Tensor::from_vec(rows, cols, terms)?;
        let loss = reduce_loss_terms(
            &terms,
            inv,
            "focal_loss_term",
            "focal_loss_column_mean",
            "focal_loss_value",
        )?;
        emit_loss_backend_meta("focal_loss_forward", prediction, (1, 1), "mean");
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
            emit_loss_backend_meta("focal_loss_backward", prediction, (rows, cols), "mean");
            return Tensor::zeros(rows, cols);
        }
        let mut grad = Vec::with_capacity(rows * cols);
        for (pred, tgt_raw) in prediction.data().iter().zip(target.data().iter()) {
            let tgt = tgt_raw.clamp(0.0, 1.0);
            grad.push(self.derivative_wrt_probability(*pred, tgt)?);
        }
        let grad = Tensor::from_vec(rows, cols, grad)?;
        validate_loss_tensor("focal_loss_gradient", &grad)?;
        let grad = relabel_loss_non_finite(
            grad.scale_with_backend(
                1.0 / (rows * cols) as f32,
                current_tensor_util_backend_for_values(grad.data().len()),
            ),
            "focal_loss_gradient",
        )?;
        validate_loss_tensor("focal_loss_gradient", &grad)?;
        emit_loss_backend_meta("focal_loss_backward", prediction, (rows, cols), "mean");
        Ok(grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn focal_loss_value_at(
        mut loss: FocalLoss,
        prediction: &Tensor,
        target: &Tensor,
        index: usize,
        delta: f32,
    ) -> f32 {
        let mut perturbed = prediction.clone();
        perturbed.data_mut()[index] += delta;
        loss.forward(&perturbed, target).unwrap().data()[0]
    }

    fn assert_finite_difference_matches(
        loss: FocalLoss,
        prediction: &Tensor,
        target: &Tensor,
        index: usize,
    ) {
        let mut analytic_loss = loss;
        let analytic = analytic_loss.backward(prediction, target).unwrap().data()[index];
        let eps = 1.0e-4;
        let plus = focal_loss_value_at(loss, prediction, target, index, eps);
        let minus = focal_loss_value_at(loss, prediction, target, index, -eps);
        let numerical = (plus - minus) / (2.0 * eps);
        let tolerance = 2.5e-3_f32.max(numerical.abs() * 0.08);
        assert!(
            (analytic - numerical).abs() <= tolerance,
            "focal gradient mismatch at {index}: analytic={analytic} numerical={numerical} tolerance={tolerance}"
        );
    }

    #[test]
    fn focal_loss_computes_value() {
        let mut loss = FocalLoss::new(0.25, 2.0);
        let prediction = Tensor::from_vec(1, 2, vec![0.9, 0.2]).unwrap();
        let target = Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap();
        let value = loss.forward(&prediction, &target).unwrap();
        assert!(value.data()[0] > 0.0);
    }

    #[test]
    fn focal_loss_backward_matches_forward_finite_difference() {
        let loss = FocalLoss::new(0.35, 1.7).with_epsilon(1.0e-6);
        let prediction = Tensor::from_vec(1, 2, vec![0.23, 0.77]).unwrap();
        let target = Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap();

        assert_finite_difference_matches(loss, &prediction, &target, 0);
        assert_finite_difference_matches(loss, &prediction, &target, 1);
    }

    #[test]
    fn focal_loss_gamma_zero_matches_weighted_binary_ce_gradient() {
        let mut loss = FocalLoss::new(0.25, 0.0);
        let prediction = Tensor::from_vec(1, 2, vec![0.4, 0.7]).unwrap();
        let target = Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap();
        let grad = loss.backward(&prediction, &target).unwrap();

        assert!((grad.data()[0] - (-0.25 / 0.4 / 2.0)).abs() < 1.0e-6);
        assert!((grad.data()[1] - (0.75 / (1.0 - 0.7) / 2.0)).abs() < 1.0e-6);
    }

    #[test]
    fn focal_loss_zero_sized_prediction_returns_zero_loss_and_empty_grad() {
        let mut loss = FocalLoss::new(0.25, 2.0);
        let prediction = Tensor::from_vec(0, 2, Vec::new()).unwrap();
        let target = Tensor::from_vec(0, 2, Vec::new()).unwrap();

        let value = loss.forward(&prediction, &target).unwrap();
        assert_eq!(value.data()[0], 0.0);

        let grad = loss.backward(&prediction, &target).unwrap();
        assert_eq!(grad.shape(), (0, 2));
        assert!(grad.data().is_empty());
    }

    #[test]
    fn focal_loss_sanitizes_extreme_epsilon() {
        let mut loss = FocalLoss::new(0.25, 2.0).with_epsilon(0.9);
        assert!(loss.epsilon < 0.5);
        let prediction = Tensor::from_vec(1, 2, vec![0.0, 1.0]).unwrap();
        let target = Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap();

        let value = loss.forward(&prediction, &target).unwrap();
        assert!(value.data()[0].is_finite());
        let grad = loss.backward(&prediction, &target).unwrap();
        assert!(grad.data().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn focal_loss_sanitizes_non_finite_constructor_scalars() {
        let loss = FocalLoss::new(f32::NAN, f32::INFINITY);

        assert_eq!(loss.alpha, 0.25);
        assert_eq!(loss.gamma, 0.0);
    }

    #[test]
    fn focal_loss_rejects_non_finite_prediction() {
        let mut loss = FocalLoss::new(0.25, 2.0);
        let prediction = Tensor::from_vec(1, 1, vec![f32::NEG_INFINITY]).unwrap();
        let target = Tensor::from_vec(1, 1, vec![1.0]).unwrap();

        let err = loss.backward(&prediction, &target).unwrap_err();

        assert!(matches!(
            err,
            st_tensor::TensorError::NonFiniteValue {
                label: "loss_prediction",
                value,
            } if value.is_infinite()
        ));
    }
}
