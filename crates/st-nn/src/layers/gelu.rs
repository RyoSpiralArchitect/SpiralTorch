// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::execution::current_tensor_util_backend_for_values;
use crate::module::Module;
use crate::{PureResult, Tensor};
use st_tensor::{emit_tensor_op, emit_tensor_op_meta, TensorError, TensorUtilBackend};

const SQRT_2_OVER_PI: f32 = std::f32::consts::FRAC_2_SQRT_PI * std::f32::consts::FRAC_1_SQRT_2;
const KAPPA: f32 = 0.044715;

fn validate_finite_value(label: &'static str, value: f32) -> PureResult<()> {
    if !value.is_finite() {
        return Err(TensorError::NonFiniteValue { label, value });
    }
    Ok(())
}

fn validate_finite_slice(label: &'static str, values: &[f32]) -> PureResult<()> {
    for &value in values {
        validate_finite_value(label, value)?;
    }
    Ok(())
}

fn validate_finite_tensor(label: &'static str, tensor: &Tensor) -> PureResult<()> {
    validate_finite_slice(label, tensor.data())
}

fn strict_gpu_path() -> bool {
    std::env::var("SPIRALTORCH_STRICT_GPU")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
}

fn tensor_util_backend_label(backend: TensorUtilBackend) -> &'static str {
    match backend {
        TensorUtilBackend::Auto => "auto",
        TensorUtilBackend::Cpu => "cpu",
        TensorUtilBackend::GpuWgpu => "wgpu",
    }
}

fn emit_empty_gelu_backward_meta(rows: usize, cols: usize, backend: TensorUtilBackend) {
    emit_tensor_op("gelu_backward", &[rows, cols, rows, cols], &[rows, cols]);
    emit_tensor_op_meta("gelu_backward", || {
        serde_json::json!({
            "backend": "identity",
            "requested_backend": tensor_util_backend_label(backend),
            "rows": rows,
            "cols": cols,
            "fallback": {
                "from": null,
                "message": null,
            }
        })
    });
}

/// Gaussian Error Linear Unit with the tanh-based approximation that remains
/// stable inside the Z-space curvature bounds.
#[derive(Debug, Clone, Copy, Default)]
pub struct Gelu;

impl Gelu {
    /// Creates a new GELU activation.
    pub fn new() -> Self {
        Self
    }

    #[cfg(test)]
    fn gelu(value: f32) -> f32 {
        let cubic = value * value * value;
        let inner = SQRT_2_OVER_PI * (value + KAPPA * cubic);
        0.5 * value * (1.0 + inner.tanh())
    }

    fn gelu_checked(value: f32) -> PureResult<f32> {
        validate_finite_value("gelu_input", value)?;
        let square = value * value;
        validate_finite_value("gelu_square", square)?;
        let cubic = square * value;
        validate_finite_value("gelu_cubic", cubic)?;
        let inner_arg = value + KAPPA * cubic;
        validate_finite_value("gelu_inner_arg", inner_arg)?;
        let inner = SQRT_2_OVER_PI * inner_arg;
        validate_finite_value("gelu_inner", inner)?;
        let tanh_inner = inner.tanh();
        validate_finite_value("gelu_tanh", tanh_inner)?;
        let output = 0.5 * value * (1.0 + tanh_inner);
        validate_finite_value("gelu_output", output)?;
        Ok(output)
    }

    #[cfg(test)]
    fn gelu_derivative(value: f32) -> f32 {
        let cubic = value * value * value;
        let inner = SQRT_2_OVER_PI * (value + KAPPA * cubic);
        let tanh_inner = inner.tanh();
        let sech_sq = 1.0 - tanh_inner * tanh_inner;
        let d_inner = SQRT_2_OVER_PI * (1.0 + 3.0 * KAPPA * value * value);
        0.5 * (1.0 + tanh_inner) + 0.5 * value * sech_sq * d_inner
    }

    fn gelu_derivative_checked(value: f32) -> PureResult<f32> {
        validate_finite_value("gelu_backward_input", value)?;
        let square = value * value;
        validate_finite_value("gelu_derivative_square", square)?;
        let cubic = square * value;
        validate_finite_value("gelu_derivative_cubic", cubic)?;
        let inner_arg = value + KAPPA * cubic;
        validate_finite_value("gelu_derivative_inner_arg", inner_arg)?;
        let inner = SQRT_2_OVER_PI * inner_arg;
        validate_finite_value("gelu_derivative_inner", inner)?;
        let tanh_inner = inner.tanh();
        validate_finite_value("gelu_derivative_tanh", tanh_inner)?;
        let sech_sq = 1.0 - tanh_inner * tanh_inner;
        validate_finite_value("gelu_derivative_sech_sq", sech_sq)?;
        let d_inner = SQRT_2_OVER_PI * (1.0 + 3.0 * KAPPA * square);
        validate_finite_value("gelu_derivative_inner_grad", d_inner)?;
        let derivative = 0.5 * (1.0 + tanh_inner) + 0.5 * value * sech_sq * d_inner;
        validate_finite_value("gelu_derivative", derivative)?;
        Ok(derivative)
    }
}

impl Module for Gelu {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        validate_finite_tensor("gelu_input", input)?;
        if rows == 0 || cols == 0 {
            return Tensor::zeros(rows, cols);
        }
        let mut data = Vec::with_capacity(rows * cols);
        for value in input.data() {
            data.push(Self::gelu_checked(*value)?);
        }
        Tensor::from_vec(rows, cols, data)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        if input.shape() != grad_output.shape() {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: grad_output.shape(),
            });
        }
        validate_finite_tensor("gelu_backward_input", input)?;
        validate_finite_tensor("gelu_backward_grad_output", grad_output)?;
        let backend = current_tensor_util_backend_for_values(rows.saturating_mul(cols));
        if rows == 0 || cols == 0 {
            let output = Tensor::zeros(rows, cols)?;
            emit_empty_gelu_backward_meta(rows, cols, backend);
            return Ok(output);
        }

        match input.gelu_backward_with_backend(grad_output, backend) {
            Ok(tensor) => {
                validate_finite_tensor("gelu_backward_output", &tensor)?;
                Ok(tensor)
            }
            Err(err @ TensorError::BackendFailure { .. }) if strict_gpu_path() => Err(err),
            Err(TensorError::BackendFailure {
                backend: fallback_from,
                message: fallback_message,
            }) => {
                let mut data = Vec::with_capacity(rows * cols);
                for (z, g) in input.data().iter().zip(grad_output.data().iter()) {
                    let derivative = Self::gelu_derivative_checked(*z)?;
                    let value = derivative * g;
                    validate_finite_value("gelu_backward_output", value)?;
                    data.push(value);
                }
                let output = Tensor::from_vec(rows, cols, data)?;
                emit_tensor_op("gelu_backward", &[rows, cols, rows, cols], &[rows, cols]);
                emit_tensor_op_meta("gelu_backward", || {
                    serde_json::json!({
                        "backend": "cpu",
                        "requested_backend": tensor_util_backend_label(backend),
                        "rows": rows,
                        "cols": cols,
                        "fallback": {
                            "from": fallback_from,
                            "message": fallback_message,
                        }
                    })
                });
                Ok(output)
            }
            Err(err) => Err(err),
        }
    }

    fn visit_parameters(
        &self,
        _visitor: &mut dyn FnMut(&crate::module::Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        _visitor: &mut dyn FnMut(&mut crate::module::Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex, OnceLock};

    fn observer_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    #[test]
    fn gelu_forward_matches_reference() {
        let layer = Gelu::new();
        let input = Tensor::from_vec(1, 4, vec![-1.0, -0.5, 0.5, 1.0]).unwrap();
        let output = layer.forward(&input).unwrap();
        let expected: Vec<f32> = input.data().iter().map(|&x| Gelu::gelu(x)).collect();
        for (out, exp) in output.data().iter().zip(expected.iter()) {
            assert!((out - exp).abs() < 1e-6);
        }
    }

    #[test]
    fn gelu_backward_uses_derivative() {
        let mut layer = Gelu::new();
        let input = Tensor::from_vec(1, 3, vec![-0.75, 0.0, 0.75]).unwrap();
        let grad_out = Tensor::from_vec(1, 3, vec![0.2, -0.5, 0.3]).unwrap();
        let grad_in = layer.backward(&input, &grad_out).unwrap();
        for i in 0..3 {
            let expected = Gelu::gelu_derivative(input.data()[i]) * grad_out.data()[i];
            assert!((grad_in.data()[i] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn gelu_empty_axes_forward_backward_pass_through() {
        let mut layer = Gelu::new();
        for (rows, cols) in [(0, 4), (3, 0)] {
            let input = Tensor::from_vec(rows, cols, Vec::new()).unwrap();
            let output = layer.forward(&input).unwrap();
            assert_eq!(output.shape(), (rows, cols));
            assert!(output.data().is_empty());

            let grad_out = Tensor::from_vec(rows, cols, Vec::new()).unwrap();
            let grad_in = layer.backward(&input, &grad_out).unwrap();
            assert_eq!(grad_in.shape(), (rows, cols));
            assert!(grad_in.data().is_empty());
        }
    }

    #[test]
    fn gelu_forward_rejects_non_finite_input() {
        let layer = Gelu::new();
        let input = Tensor::from_vec(1, 2, vec![0.25, f32::NAN]).unwrap();

        let err = layer.forward(&input).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "gelu_input",
                value,
            } if value.is_nan()
        ));
    }

    #[test]
    fn gelu_forward_rejects_overflowing_scalar_intermediate() {
        let layer = Gelu::new();
        let input = Tensor::from_vec(1, 1, vec![f32::MAX]).unwrap();

        let err = layer.forward(&input).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "gelu_square",
                value,
            } if value.is_infinite()
        ));
    }

    #[test]
    fn gelu_backward_rejects_non_finite_grad_output() {
        let mut layer = Gelu::new();
        let input = Tensor::from_vec(1, 2, vec![0.25, -0.5]).unwrap();
        let grad_out = Tensor::from_vec(1, 2, vec![0.1, f32::NAN]).unwrap();

        let err = layer.backward(&input, &grad_out).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "gelu_backward_grad_output",
                value,
            } if value.is_nan()
        ));
    }

    #[test]
    fn gelu_backward_rejects_overflowing_backend_output() {
        let _policy_guard = crate::execution::push_backend_policy(
            crate::execution::BackendPolicy::from_device_caps(
                st_core::backend::device_caps::DeviceCaps::cpu(),
            ),
        );
        let mut layer = Gelu::new();
        let input = Tensor::from_vec(1, 1, vec![f32::MAX]).unwrap();
        let grad_out = Tensor::from_vec(1, 1, vec![1.0]).unwrap();

        let err = layer.backward(&input, &grad_out).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "gelu_backward_output",
                value,
            } if !value.is_finite()
        ));
    }

    #[test]
    fn gelu_empty_backward_emits_backend_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let mut layer = Gelu::new();
        let input = Tensor::from_vec(0, 4, Vec::new()).unwrap();
        let grad_out = Tensor::from_vec(0, 4, Vec::new()).unwrap();
        let _ = layer.backward(&input, &grad_out).unwrap();
        st_tensor::set_tensor_op_meta_observer(previous);

        let events = events.lock().unwrap();
        let gelu = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "gelu_backward" && data["rows"] == 0 && data["cols"] == 4
            })
            .expect("empty gelu backward metadata event");
        assert_eq!(gelu.1["backend"], "identity");
        assert_eq!(gelu.1["requested_backend"], "auto");
    }

    #[test]
    fn gelu_backward_respects_cpu_policy() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let _policy_guard = crate::execution::push_backend_policy(
            crate::execution::BackendPolicy::from_device_caps(
                st_core::backend::device_caps::DeviceCaps::cpu(),
            ),
        );
        let mut layer = Gelu::new();
        let input =
            Tensor::from_vec(2, 4, vec![-0.75, 0.0, 0.75, 1.25, -1.0, 0.5, 1.0, -0.25]).unwrap();
        let grad_out =
            Tensor::from_vec(2, 4, vec![0.2, -0.5, 0.3, 0.1, -0.4, 0.6, 0.8, -0.2]).unwrap();
        let _ = layer.backward(&input, &grad_out).unwrap();
        st_tensor::set_tensor_op_meta_observer(previous);

        let events = events.lock().unwrap();
        let gelu = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "gelu_backward" && data["rows"] == 2 && data["cols"] == 4
            })
            .expect("gelu backward metadata event");
        assert_eq!(gelu.1["backend"], "cpu");
        assert_eq!(gelu.1["requested_backend"], "cpu");
    }
}
