// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::{
    binary_probability_epsilon, checked_loss_value, emit_loss_backend_meta,
    emit_loss_backend_meta_with_backend, reduce_loss_terms, relabel_loss_non_finite,
    validate_loss_inputs, validate_loss_tensor, Loss,
};
use crate::execution::current_tensor_util_backend_for_values;
use crate::{PureResult, Tensor};
#[cfg(feature = "wgpu")]
use st_tensor::backend::wgpu_dense;
use st_tensor::{TensorError, TensorUtilBackend};

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
        if curvature >= 0.0 || !curvature.is_finite() {
            return Err(TensorError::NonHyperbolicCurvature { curvature });
        }
        Ok(Self {
            curvature,
            epsilon: binary_probability_epsilon(epsilon, 1.0e-5),
        })
    }

    /// Returns the hyperbolic curvature.
    pub fn curvature(&self) -> f32 {
        self.curvature
    }

    /// Returns the epsilon used for clamping targets.
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    fn stable_softplus(value: f32) -> PureResult<f32> {
        checked_loss_value("hyperbolic_cross_entropy_softplus_input", value)?;
        let output = if value > 0.0 {
            value + (-value).exp().ln_1p()
        } else {
            value.exp().ln_1p()
        };
        checked_loss_value("hyperbolic_cross_entropy_softplus", output)
    }

    fn stable_sigmoid(value: f32) -> PureResult<f32> {
        checked_loss_value("hyperbolic_cross_entropy_sigmoid_input", value)?;
        let output = if value >= 0.0 {
            1.0 / (1.0 + (-value).exp())
        } else {
            let exp_value = value.exp();
            exp_value / (1.0 + exp_value)
        };
        checked_loss_value("hyperbolic_cross_entropy_sigmoid", output)
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
fn hyperbolic_ce_wgpu_error(op_name: &'static str, message: String) -> TensorError {
    TensorError::BackendFailure {
        backend: "wgpu",
        message: format!("{op_name} wgpu path failed ({message}); fallback disabled"),
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
        validate_loss_inputs(prediction, target)?;
        let (rows, cols) = prediction.shape();
        if rows == 0 || cols == 0 {
            emit_loss_backend_meta(
                "hyperbolic_cross_entropy_forward",
                prediction,
                (1, 1),
                "mean",
            );
            return Tensor::from_vec(1, 1, vec![0.0]);
        }
        let backend = current_tensor_util_backend_for_values(prediction.data().len());
        let requested_backend = tensor_util_backend_label(backend);
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(backend, TensorUtilBackend::GpuWgpu) && wgpu_dense::is_available() {
                match wgpu_dense::hyperbolic_cross_entropy_forward(
                    prediction.data(),
                    target.data(),
                    rows,
                    cols,
                    self.curvature,
                    self.epsilon,
                ) {
                    Ok(loss) => {
                        let loss = checked_loss_value("hyperbolic_cross_entropy_loss", loss)?;
                        emit_loss_backend_meta_with_backend(
                            "hyperbolic_cross_entropy_forward",
                            prediction,
                            (1, 1),
                            "mean",
                            "wgpu_dense",
                            requested_backend,
                            "loss.hyperbolic_cross_entropy.forward_wgpu",
                            None,
                        );
                        return Tensor::from_vec(1, 1, vec![loss]);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(hyperbolic_ce_wgpu_error(
                            "hyperbolic_cross_entropy_forward",
                            message,
                        ));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let scale = (-self.curvature).sqrt();
        let count = (rows * cols) as f32;
        let mut terms = Vec::with_capacity(rows * cols);
        for (pred, tgt_raw) in prediction.data().iter().zip(target.data().iter()) {
            let tgt = tgt_raw.clamp(self.epsilon, 1.0 - self.epsilon);
            let scaled = checked_loss_value("hyperbolic_cross_entropy_scaled_logit", pred * scale)?;
            let log_sigmoid_pos = -Self::stable_softplus(-scaled)?;
            let log_sigmoid_neg = -Self::stable_softplus(scaled)?;
            let term = checked_loss_value(
                "hyperbolic_cross_entropy_term",
                -tgt * log_sigmoid_pos - (1.0 - tgt) * log_sigmoid_neg,
            )?;
            terms.push(term);
        }
        let terms = Tensor::from_vec(rows, cols, terms)?;
        let loss = reduce_loss_terms(
            &terms,
            1.0 / count,
            "hyperbolic_cross_entropy_term",
            "hyperbolic_cross_entropy_column_mean",
            "hyperbolic_cross_entropy_loss",
        )?;
        emit_loss_backend_meta_with_backend(
            "hyperbolic_cross_entropy_forward",
            prediction,
            (1, 1),
            "mean",
            "cpu",
            requested_backend,
            "loss.hyperbolic_cross_entropy.scalar_forward",
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
                "hyperbolic_cross_entropy_backward",
                prediction,
                (rows, cols),
                "mean",
            );
            return Tensor::zeros(rows, cols);
        }
        let backend = current_tensor_util_backend_for_values(prediction.data().len());
        let requested_backend = tensor_util_backend_label(backend);
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(backend, TensorUtilBackend::GpuWgpu) && wgpu_dense::is_available() {
                match wgpu_dense::hyperbolic_cross_entropy_backward(
                    prediction.data(),
                    target.data(),
                    rows,
                    cols,
                    self.curvature,
                    self.epsilon,
                ) {
                    Ok(grad) => {
                        let grad = Tensor::from_vec(rows, cols, grad)?;
                        validate_loss_tensor("hyperbolic_cross_entropy_gradient", &grad)?;
                        emit_loss_backend_meta_with_backend(
                            "hyperbolic_cross_entropy_backward",
                            prediction,
                            (rows, cols),
                            "mean",
                            "wgpu_dense",
                            requested_backend,
                            "loss.hyperbolic_cross_entropy.backward_wgpu",
                            None,
                        );
                        return Ok(grad);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(hyperbolic_ce_wgpu_error(
                            "hyperbolic_cross_entropy_backward",
                            message,
                        ));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let scale = (-self.curvature).sqrt();
        let mut data = Vec::with_capacity(rows * cols);
        for (pred, tgt_raw) in prediction.data().iter().zip(target.data().iter()) {
            let tgt = tgt_raw.clamp(self.epsilon, 1.0 - self.epsilon);
            let scaled = checked_loss_value("hyperbolic_cross_entropy_scaled_logit", pred * scale)?;
            let sigmoid = Self::stable_sigmoid(scaled)?;
            data.push(checked_loss_value(
                "hyperbolic_cross_entropy_gradient",
                scale * (sigmoid - tgt),
            )?);
        }
        let grad = Tensor::from_vec(rows, cols, data)?;
        validate_loss_tensor("hyperbolic_cross_entropy_gradient", &grad)?;
        let grad = relabel_loss_non_finite(
            grad.scale_with_backend(
                1.0 / (rows * cols) as f32,
                current_tensor_util_backend_for_values(grad.data().len()),
            ),
            "hyperbolic_cross_entropy_gradient",
        )?;
        validate_loss_tensor("hyperbolic_cross_entropy_gradient", &grad)?;
        emit_loss_backend_meta_with_backend(
            "hyperbolic_cross_entropy_backward",
            prediction,
            (rows, cols),
            "mean",
            "cpu",
            requested_backend,
            "loss.hyperbolic_cross_entropy.scalar_backward",
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

    #[test]
    fn hyperbolic_cross_entropy_zero_sized_prediction_returns_zero_loss_and_empty_grad() {
        let mut loss = HyperbolicCrossEntropy::new(-1.0).unwrap();
        let prediction = Tensor::from_vec(0, 2, Vec::new()).unwrap();
        let target = Tensor::from_vec(0, 2, Vec::new()).unwrap();

        let value = loss.forward(&prediction, &target).unwrap();
        assert_eq!(value.data()[0], 0.0);

        let grad = loss.backward(&prediction, &target).unwrap();
        assert_eq!(grad.shape(), (0, 2));
        assert!(grad.data().is_empty());
    }

    #[test]
    fn hyperbolic_cross_entropy_sanitizes_extreme_epsilon() {
        let mut loss = HyperbolicCrossEntropy::with_epsilon(-1.0, 0.9).unwrap();
        assert!(loss.epsilon() < 0.5);
        let prediction = Tensor::from_vec(1, 2, vec![0.0, 1.0]).unwrap();
        let target = Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap();

        let value = loss.forward(&prediction, &target).unwrap();
        assert!(value.data()[0].is_finite());
        let grad = loss.backward(&prediction, &target).unwrap();
        assert!(grad.data().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn hyperbolic_cross_entropy_rejects_non_finite_curvature() {
        let err = HyperbolicCrossEntropy::new(f32::NAN).unwrap_err();

        assert!(matches!(
            err,
            st_tensor::TensorError::NonHyperbolicCurvature { curvature } if curvature.is_nan()
        ));
    }

    #[test]
    fn hyperbolic_cross_entropy_handles_large_logits_without_overflow() {
        let mut loss = HyperbolicCrossEntropy::new(-1.0).unwrap();
        let prediction = Tensor::from_vec(1, 2, vec![1000.0, -1000.0]).unwrap();
        let target = Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap();

        let value = loss.forward(&prediction, &target).unwrap();
        assert!(value.data()[0].is_finite());

        let grad = loss.backward(&prediction, &target).unwrap();
        assert!(grad.data().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn hyperbolic_cross_entropy_rejects_overflowing_scaled_logit() {
        let mut loss = HyperbolicCrossEntropy::new(-4.0).unwrap();
        let prediction = Tensor::from_vec(1, 1, vec![f32::MAX]).unwrap();
        let target = Tensor::from_vec(1, 1, vec![1.0]).unwrap();

        let err = loss.forward(&prediction, &target).unwrap_err();

        assert!(matches!(
            err,
            st_tensor::TensorError::NonFiniteValue {
                label: "hyperbolic_cross_entropy_scaled_logit",
                value,
            } if value.is_infinite()
        ));
    }
}
