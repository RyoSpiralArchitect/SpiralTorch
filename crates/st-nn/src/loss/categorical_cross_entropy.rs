// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::{
    checked_loss_value, emit_loss_backend_meta, emit_loss_backend_meta_with_backend,
    probability_floor, reduce_loss_terms, relabel_loss_non_finite, validate_loss_inputs,
    validate_loss_tensor, Loss,
};
use crate::execution::current_tensor_util_backend_for_values;
use crate::{PureResult, Tensor};
#[cfg(feature = "wgpu")]
use st_tensor::backend::wgpu_dense;
use st_tensor::{TensorError, TensorUtilBackend};

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
        self.epsilon = probability_floor(epsilon, Self::default().epsilon);
        self
    }

    /// Returns the epsilon used for clamping.
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }
}

fn tensor_util_backend_label(backend: TensorUtilBackend) -> &'static str {
    match backend {
        TensorUtilBackend::Auto => "auto",
        TensorUtilBackend::Cpu => "cpu",
        TensorUtilBackend::GpuWgpu => "wgpu",
    }
}

#[cfg(feature = "wgpu")]
fn strict_gpu_path() -> bool {
    std::env::var("SPIRALTORCH_STRICT_GPU")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
}

#[cfg(feature = "wgpu")]
fn categorical_ce_wgpu_error(op_name: &'static str, message: String) -> TensorError {
    TensorError::BackendFailure {
        backend: "wgpu",
        message: format!("{op_name} wgpu path failed ({message}); fallback disabled"),
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
        validate_loss_inputs(prediction, target)?;
        let (rows, cols) = prediction.shape();
        if rows == 0 || cols == 0 {
            emit_loss_backend_meta(
                "categorical_cross_entropy_forward",
                prediction,
                (1, 1),
                "batch_mean",
            );
            return Tensor::from_vec(1, 1, vec![0.0]);
        }
        let route_backend = current_tensor_util_backend_for_values(rows.saturating_mul(cols));
        let requested_backend = tensor_util_backend_label(route_backend);
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(route_backend, TensorUtilBackend::GpuWgpu) && wgpu_dense::is_available() {
                match wgpu_dense::categorical_cross_entropy_forward(
                    prediction.data(),
                    target.data(),
                    rows,
                    cols,
                    self.epsilon,
                ) {
                    Ok(loss) => {
                        let loss = checked_loss_value("categorical_cross_entropy_loss", loss)?;
                        emit_loss_backend_meta_with_backend(
                            "categorical_cross_entropy_forward",
                            prediction,
                            (1, 1),
                            "batch_mean",
                            "wgpu_dense",
                            requested_backend,
                            "loss.categorical_cross_entropy.forward_wgpu",
                            None,
                        );
                        return Tensor::from_vec(1, 1, vec![loss]);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(categorical_ce_wgpu_error(
                            "categorical_cross_entropy_forward",
                            message,
                        ));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let mut terms = Vec::with_capacity(rows * cols);
        for (pred, tgt) in prediction.data().iter().zip(target.data().iter()) {
            let p = pred.clamp(self.epsilon, 1.0);
            let term = checked_loss_value("categorical_cross_entropy_term", -tgt * p.ln())?;
            terms.push(term);
        }
        let terms = Tensor::from_vec(rows, cols, terms)?;
        let loss = reduce_loss_terms(
            &terms,
            1.0 / rows as f32,
            "categorical_cross_entropy_term",
            "categorical_cross_entropy_column_mean",
            "categorical_cross_entropy_loss",
        )?;
        emit_loss_backend_meta_with_backend(
            "categorical_cross_entropy_forward",
            prediction,
            (1, 1),
            "batch_mean",
            "cpu",
            requested_backend,
            "loss.categorical_cross_entropy.forward_scalar",
            {
                #[cfg(feature = "wgpu")]
                {
                    wgpu_failure.as_deref()
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    None
                }
            },
        );
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
                "categorical_cross_entropy_backward",
                prediction,
                (rows, cols),
                "batch_mean",
            );
            return Tensor::zeros(rows, cols);
        }
        let route_backend = current_tensor_util_backend_for_values(rows.saturating_mul(cols));
        let requested_backend = tensor_util_backend_label(route_backend);
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(route_backend, TensorUtilBackend::GpuWgpu) && wgpu_dense::is_available() {
                match wgpu_dense::categorical_cross_entropy_backward(
                    prediction.data(),
                    target.data(),
                    rows,
                    cols,
                    self.epsilon,
                ) {
                    Ok(grad) => {
                        let grad = Tensor::from_vec(rows, cols, grad)?;
                        validate_loss_tensor("categorical_cross_entropy_gradient", &grad)?;
                        emit_loss_backend_meta_with_backend(
                            "categorical_cross_entropy_backward",
                            prediction,
                            (rows, cols),
                            "batch_mean",
                            "wgpu_dense",
                            requested_backend,
                            "loss.categorical_cross_entropy.backward_wgpu",
                            None,
                        );
                        return Ok(grad);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(categorical_ce_wgpu_error(
                            "categorical_cross_entropy_backward",
                            message,
                        ));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let mut grad = Vec::with_capacity(rows * cols);
        for (pred, tgt) in prediction.data().iter().zip(target.data().iter()) {
            let p = pred.clamp(self.epsilon, 1.0);
            grad.push(checked_loss_value(
                "categorical_cross_entropy_gradient",
                -tgt / p,
            )?);
        }
        let grad = Tensor::from_vec(rows, cols, grad)?;
        validate_loss_tensor("categorical_cross_entropy_gradient", &grad)?;
        let grad = relabel_loss_non_finite(
            grad.scale_with_backend(
                1.0 / rows as f32,
                current_tensor_util_backend_for_values(grad.data().len()),
            ),
            "categorical_cross_entropy_gradient",
        )?;
        validate_loss_tensor("categorical_cross_entropy_gradient", &grad)?;
        emit_loss_backend_meta_with_backend(
            "categorical_cross_entropy_backward",
            prediction,
            (rows, cols),
            "batch_mean",
            "cpu",
            requested_backend,
            "loss.categorical_cross_entropy.backward_scalar",
            {
                #[cfg(feature = "wgpu")]
                {
                    wgpu_failure.as_deref()
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    None
                }
            },
        );
        Ok(grad)
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

    #[test]
    fn categorical_ce_zero_sized_prediction_returns_zero_loss_and_empty_grad() {
        let mut loss = CategoricalCrossEntropy::new();
        let prediction = Tensor::from_vec(0, 3, Vec::new()).unwrap();
        let target = Tensor::from_vec(0, 3, Vec::new()).unwrap();

        let value = loss.forward(&prediction, &target).unwrap();
        assert_eq!(value.shape(), (1, 1));
        assert_eq!(value.data()[0], 0.0);

        let grad = loss.backward(&prediction, &target).unwrap();
        assert_eq!(grad.shape(), (0, 3));
        assert!(grad.data().is_empty());
    }

    #[test]
    fn categorical_ce_sanitizes_extreme_epsilon() {
        let mut loss = CategoricalCrossEntropy::new().with_epsilon(2.0);
        assert_eq!(loss.epsilon(), 1.0);
        let prediction = Tensor::from_vec(1, 2, vec![0.0, 1.0]).unwrap();
        let target = Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap();

        let value = loss.forward(&prediction, &target).unwrap();
        assert!(value.data()[0].is_finite());

        let grad = loss.backward(&prediction, &target).unwrap();
        assert!(grad.data().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn categorical_ce_rejects_non_finite_target() {
        let mut loss = CategoricalCrossEntropy::new();
        let prediction = Tensor::from_vec(1, 2, vec![0.4, 0.6]).unwrap();
        let target = Tensor::from_vec(1, 2, vec![0.0, f32::INFINITY]).unwrap();

        let err = loss.forward(&prediction, &target).unwrap_err();

        assert!(matches!(
            err,
            st_tensor::TensorError::NonFiniteValue {
                label: "loss_target",
                value,
            } if value.is_infinite()
        ));
    }
}
