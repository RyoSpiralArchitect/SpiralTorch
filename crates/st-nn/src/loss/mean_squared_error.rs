// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::{
    checked_loss_value, emit_loss_backend_meta, emit_loss_backend_meta_with_backend,
    relabel_loss_non_finite, validate_loss_inputs, validate_loss_tensor, Loss,
};
use crate::execution::current_tensor_util_backend_for_values;
use crate::{PureResult, Tensor};
#[cfg(feature = "wgpu")]
use st_tensor::backend::wgpu_dense;
use st_tensor::{TensorError, TensorUtilBackend};

/// Classic mean squared error loss with mean reduction.
#[derive(Debug, Default, Clone, Copy)]
pub struct MeanSquaredError;

impl MeanSquaredError {
    /// Creates a new mean squared error loss instance.
    pub fn new() -> Self {
        Self
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
fn mse_wgpu_error(op_name: &'static str, message: String) -> TensorError {
    TensorError::BackendFailure {
        backend: "wgpu",
        message: format!("{op_name} wgpu path failed ({message}); fallback disabled"),
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
        validate_loss_inputs(prediction, target)?;
        let (rows, cols) = prediction.shape();
        if rows == 0 || cols == 0 {
            emit_loss_backend_meta("mse_loss_forward", prediction, (1, 1), "mean");
            return Tensor::from_vec(1, 1, vec![0.0]);
        }
        let backend = current_tensor_util_backend_for_values(prediction.data().len());
        let requested_backend = tensor_util_backend_label(backend);
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(backend, TensorUtilBackend::GpuWgpu) && wgpu_dense::is_available() {
                match wgpu_dense::mse_loss_forward(prediction.data(), target.data(), rows, cols) {
                    Ok(loss) => {
                        let loss = checked_loss_value("mse_loss", loss)?;
                        emit_loss_backend_meta_with_backend(
                            "mse_loss_forward",
                            prediction,
                            (1, 1),
                            "mean",
                            "wgpu_dense",
                            requested_backend,
                            "loss.mse.forward_wgpu",
                            None,
                        );
                        return Tensor::from_vec(1, 1, vec![loss]);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(mse_wgpu_error("mse_loss_forward", message));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let diff =
            relabel_loss_non_finite(prediction.sub_with_backend(target, backend), "mse_diff")?;
        validate_loss_tensor("mse_diff", &diff)?;
        let squared = relabel_loss_non_finite(
            diff.hadamard_with_backend(&diff, backend),
            "mse_squared_diff",
        )?;
        validate_loss_tensor("mse_squared_diff", &squared)?;
        let column_means = relabel_loss_non_finite(
            squared
                .try_sum_axis0_scaled_with_backend(1.0 / prediction.data().len() as f32, backend),
            "mse_loss_column_mean",
        )?;
        let mut mean = 0.0f32;
        for value in column_means {
            let value = checked_loss_value("mse_loss_column_mean", value)?;
            mean = checked_loss_value("mse_loss", mean + value)?;
        }
        emit_loss_backend_meta_with_backend(
            "mse_loss_forward",
            prediction,
            (1, 1),
            "mean",
            "cpu",
            requested_backend,
            "loss.mse.forward_scalar",
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
        Tensor::from_vec(1, 1, vec![mean])
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
            emit_loss_backend_meta("mse_loss_backward", prediction, (rows, cols), "mean");
            return Tensor::zeros(rows, cols);
        }
        let backend = current_tensor_util_backend_for_values(prediction.data().len());
        let requested_backend = tensor_util_backend_label(backend);
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(backend, TensorUtilBackend::GpuWgpu) && wgpu_dense::is_available() {
                match wgpu_dense::mse_loss_backward(prediction.data(), target.data(), rows, cols) {
                    Ok(grad) => {
                        let grad = Tensor::from_vec(rows, cols, grad)?;
                        validate_loss_tensor("mse_gradient", &grad)?;
                        emit_loss_backend_meta_with_backend(
                            "mse_loss_backward",
                            prediction,
                            (rows, cols),
                            "mean",
                            "wgpu_dense",
                            requested_backend,
                            "loss.mse.backward_wgpu",
                            None,
                        );
                        return Ok(grad);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(mse_wgpu_error("mse_loss_backward", message));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let inv = 2.0f32 / (rows * cols) as f32;
        let diff =
            relabel_loss_non_finite(prediction.sub_with_backend(target, backend), "mse_diff")?;
        validate_loss_tensor("mse_diff", &diff)?;
        let grad = relabel_loss_non_finite(
            diff.scale_with_backend(
                inv,
                current_tensor_util_backend_for_values(diff.data().len()),
            ),
            "mse_gradient",
        )?;
        validate_loss_tensor("mse_gradient", &grad)?;
        emit_loss_backend_meta_with_backend(
            "mse_loss_backward",
            prediction,
            (rows, cols),
            "mean",
            "cpu",
            requested_backend,
            "loss.mse.backward_scalar",
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

    #[test]
    fn mse_zero_sized_prediction_returns_zero_loss_and_empty_grad() {
        let mut loss = MeanSquaredError::new();
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
    fn mse_rejects_non_finite_prediction() {
        let mut loss = MeanSquaredError::new();
        let prediction = Tensor::from_vec(1, 1, vec![f32::NAN]).unwrap();
        let target = Tensor::zeros(1, 1).unwrap();

        let err = loss.forward(&prediction, &target).unwrap_err();

        assert!(matches!(
            err,
            st_tensor::TensorError::NonFiniteValue {
                label: "loss_prediction",
                value,
            } if value.is_nan()
        ));
    }

    #[test]
    fn mse_rejects_overflowing_forward_square() {
        let mut loss = MeanSquaredError::new();
        let prediction = Tensor::from_vec(1, 1, vec![f32::MAX]).unwrap();
        let target = Tensor::zeros(1, 1).unwrap();

        let err = loss.forward(&prediction, &target).unwrap_err();

        assert!(matches!(
            err,
            st_tensor::TensorError::NonFiniteValue {
                label: "mse_squared_diff",
                value,
            } if value.is_infinite()
        ));
    }

    #[test]
    fn mse_relabels_overflowing_backward_scale_as_gradient() {
        let mut loss = MeanSquaredError::new();
        let prediction = Tensor::from_vec(1, 1, vec![f32::MAX]).unwrap();
        let target = Tensor::zeros(1, 1).unwrap();

        let err = loss.backward(&prediction, &target).unwrap_err();

        assert!(matches!(
            err,
            st_tensor::TensorError::NonFiniteValue {
                label: "mse_gradient",
                value,
            } if value.is_infinite()
        ));
    }
}
