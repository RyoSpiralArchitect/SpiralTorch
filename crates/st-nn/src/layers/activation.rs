// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::execution::current_tensor_util_backend_for_values;
use crate::module::Module;
use crate::{PureResult, Tensor};
#[cfg(feature = "wgpu")]
use st_tensor::backend::wgpu_dense;
use st_tensor::{emit_tensor_op, emit_tensor_op_meta, TensorError, TensorUtilBackend};

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

fn emit_relu_meta(
    op_name: &'static str,
    rows: usize,
    cols: usize,
    backend: &'static str,
    requested_backend: &'static str,
    kernel: &'static str,
    backward: bool,
    active_values: usize,
    mask_backend: Option<String>,
    fallback_message: Option<&str>,
) {
    let input_shape = if backward {
        vec![rows, cols, rows, cols]
    } else {
        vec![rows, cols]
    };
    emit_tensor_op(op_name, &input_shape, &[rows, cols]);
    emit_tensor_op_meta(op_name, || {
        let values = rows.saturating_mul(cols);
        let mut data = serde_json::Map::new();
        data.insert("backend".to_string(), serde_json::json!(backend));
        data.insert(
            "requested_backend".to_string(),
            serde_json::json!(requested_backend),
        );
        data.insert("kernel".to_string(), serde_json::json!(kernel));
        data.insert(
            "kind".to_string(),
            serde_json::json!(if backward {
                "activation_backward_mask"
            } else {
                "activation_forward"
            }),
        );
        data.insert("rows".to_string(), serde_json::json!(rows));
        data.insert("cols".to_string(), serde_json::json!(cols));
        data.insert("values".to_string(), serde_json::json!(values));
        data.insert("output_rows".to_string(), serde_json::json!(rows));
        data.insert("output_cols".to_string(), serde_json::json!(cols));
        data.insert("output_values".to_string(), serde_json::json!(values));
        data.insert(
            "active_values".to_string(),
            serde_json::json!(active_values),
        );
        data.insert(
            "inactive_values".to_string(),
            serde_json::json!(values.saturating_sub(active_values)),
        );
        data.insert("mask_backend".to_string(), serde_json::json!(mask_backend));
        data.insert(
            "estimated_mask_values".to_string(),
            serde_json::json!(if backward { values } else { 0 }),
        );
        data.insert(
            "empty".to_string(),
            serde_json::json!(rows == 0 || cols == 0),
        );
        if let Some(message) = fallback_message {
            data.insert(
                "fallback".to_string(),
                serde_json::json!({"from": "wgpu", "message": message}),
            );
        }
        serde_json::Value::Object(data)
    });
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
fn relu_wgpu_error(op_name: &'static str, message: String) -> TensorError {
    TensorError::BackendFailure {
        backend: "wgpu",
        message: format!("{op_name} wgpu path failed ({message}); fallback disabled"),
    }
}

/// Lightweight ReLU activation that keeps gradients aligned with the
/// SpiralTorch module trait. The layer is stateless and therefore does not
/// participate in parameter visits.
#[derive(Debug, Default, Clone, Copy)]
pub struct Relu;

impl Relu {
    /// Creates a new ReLU layer.
    pub fn new() -> Self {
        Self
    }
}

impl Module for Relu {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        validate_finite_tensor("relu_input", input)?;
        let active_values = input.data().iter().filter(|value| **value > 0.0).count();
        let route_backend = current_tensor_util_backend_for_values(rows.saturating_mul(cols));
        let requested_backend = tensor_util_backend_label(route_backend);
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(route_backend, TensorUtilBackend::GpuWgpu) && wgpu_dense::is_available() {
                match wgpu_dense::relu(input.data(), rows, cols) {
                    Ok(buffer) => {
                        validate_finite_slice("relu_output", &buffer)?;
                        let output = Tensor::from_vec(rows, cols, buffer)?;
                        emit_relu_meta(
                            "relu_forward",
                            rows,
                            cols,
                            "wgpu_dense",
                            requested_backend,
                            "relu.forward_wgpu",
                            false,
                            active_values,
                            None,
                            None,
                        );
                        return Ok(output);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(relu_wgpu_error("relu_forward", message));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let data = input
            .data()
            .iter()
            .map(|value| (*value).max(0.0))
            .collect::<Vec<_>>();
        let output = Tensor::from_vec(rows, cols, data)?;
        validate_finite_tensor("relu_output", &output)?;
        emit_relu_meta(
            "relu_forward",
            rows,
            cols,
            "cpu",
            requested_backend,
            "relu.scalar",
            false,
            active_values,
            None,
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
        Ok(output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        if input.shape() != grad_output.shape() {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: grad_output.shape(),
            });
        }
        let (rows, cols) = input.shape();
        validate_finite_tensor("relu_backward_input", input)?;
        validate_finite_tensor("relu_backward_grad_output", grad_output)?;
        let active_values = input.data().iter().filter(|value| **value > 0.0).count();
        let route_backend = current_tensor_util_backend_for_values(rows.saturating_mul(cols));
        let requested_backend = tensor_util_backend_label(route_backend);
        #[cfg(feature = "wgpu")]
        let mut wgpu_failure: Option<String> = None;

        #[cfg(feature = "wgpu")]
        {
            if matches!(route_backend, TensorUtilBackend::GpuWgpu) && wgpu_dense::is_available() {
                match wgpu_dense::relu_backward(input.data(), grad_output.data(), rows, cols) {
                    Ok(buffer) => {
                        validate_finite_slice("relu_backward_output", &buffer)?;
                        let output = Tensor::from_vec(rows, cols, buffer)?;
                        emit_relu_meta(
                            "relu_backward",
                            rows,
                            cols,
                            "wgpu_dense",
                            requested_backend,
                            "relu.backward_wgpu",
                            true,
                            active_values,
                            Some(requested_backend.to_string()),
                            None,
                        );
                        return Ok(output);
                    }
                    Err(message) if strict_gpu_path() => {
                        return Err(relu_wgpu_error("relu_backward", message));
                    }
                    Err(message) => {
                        wgpu_failure = Some(message);
                    }
                }
            }
        }

        let mask_data = input
            .data()
            .iter()
            .map(|value| if *value > 0.0 { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let mask = Tensor::from_vec(rows, cols, mask_data)?;
        let mask_backend = current_tensor_util_backend_for_values(rows.saturating_mul(cols));
        let output = grad_output.hadamard_with_backend(&mask, mask_backend)?;
        validate_finite_tensor("relu_backward_output", &output)?;
        emit_relu_meta(
            "relu_backward",
            rows,
            cols,
            "cpu",
            requested_backend,
            "relu.backward_mask_scalar",
            true,
            active_values,
            Some(mask_backend.to_string()),
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
        Ok(output)
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
    use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

    fn observer_lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    #[test]
    fn relu_forward_backward() {
        let relu = Relu::new();
        let input = Tensor::from_vec(1, 4, vec![-1.0, -0.5, 0.2, 1.5]).unwrap();
        let output = relu.forward(&input).unwrap();
        assert_eq!(output.data(), &[0.0, 0.0, 0.2, 1.5]);

        let mut relu = relu;
        let grad_output = Tensor::from_vec(1, 4, vec![0.3, 0.4, 0.5, 0.6]).unwrap();
        let grad_input = relu.backward(&input, &grad_output).unwrap();
        assert_eq!(grad_input.data(), &[0.0, 0.0, 0.5, 0.6]);
    }

    #[test]
    fn relu_forward_rejects_non_finite_input() {
        let relu = Relu::new();
        let input = Tensor::from_vec(1, 3, vec![-1.0, f32::NAN, 0.5]).unwrap();

        let err = relu.forward(&input).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "relu_input",
                value,
            } if value.is_nan()
        ));
    }

    #[test]
    fn relu_backward_rejects_non_finite_input_without_masking_it() {
        let mut relu = Relu::new();
        let input = Tensor::from_vec(1, 3, vec![-1.0, f32::NAN, 0.5]).unwrap();
        let grad_output = Tensor::from_vec(1, 3, vec![0.3, 0.4, 0.5]).unwrap();

        let err = relu.backward(&input, &grad_output).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "relu_backward_input",
                value,
            } if value.is_nan()
        ));
    }

    #[test]
    fn relu_backward_rejects_non_finite_grad_output() {
        let mut relu = Relu::new();
        let input = Tensor::from_vec(1, 3, vec![-1.0, 0.0, 0.5]).unwrap();
        let grad_output = Tensor::from_vec(1, 3, vec![0.3, f32::INFINITY, 0.5]).unwrap();

        let err = relu.backward(&input, &grad_output).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "relu_backward_grad_output",
                value,
            } if value.is_infinite()
        ));
    }

    #[test]
    fn relu_forward_backward_emit_backend_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let mut relu = Relu::new();
        let input = Tensor::from_vec(1, 4, vec![-1.0, -0.5, 0.2, 1.5]).unwrap();
        let grad_output = Tensor::from_vec(1, 4, vec![0.3, 0.4, 0.5, 0.6]).unwrap();
        let _ = relu.forward(&input).unwrap();
        let _ = relu.backward(&input, &grad_output).unwrap();
        st_tensor::set_tensor_op_meta_observer(previous);

        let events = events.lock().unwrap();
        let forward = events
            .iter()
            .find(|(op_name, data)| *op_name == "relu_forward" && data["cols"] == 4)
            .expect("relu forward metadata event");
        assert_eq!(forward.1["backend"], "cpu");
        assert_eq!(forward.1["requested_backend"], "auto");
        assert_eq!(forward.1["kind"], "activation_forward");
        assert_eq!(forward.1["active_values"], 2);

        let backward = events
            .iter()
            .find(|(op_name, data)| *op_name == "relu_backward" && data["cols"] == 4)
            .expect("relu backward metadata event");
        assert_eq!(backward.1["backend"], "cpu");
        assert_eq!(backward.1["kind"], "activation_backward_mask");
        assert_eq!(backward.1["mask_backend"], "auto");

        let hadamard = events
            .iter()
            .any(|(op_name, data)| *op_name == "hadamard" && data["cols"] == 4);
        assert!(hadamard);
    }
}
