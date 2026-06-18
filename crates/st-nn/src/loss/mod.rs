// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

mod categorical_cross_entropy;
mod contrastive_loss;
mod focal_loss;
mod hyperbolic_cross_entropy;
mod mean_squared_error;
mod triplet_loss;

use crate::execution::current_tensor_util_backend_for_values;
use crate::{PureResult, Tensor};
use st_tensor::{emit_tensor_op, emit_tensor_op_meta, TensorError};

pub use categorical_cross_entropy::CategoricalCrossEntropy;
pub use contrastive_loss::ContrastiveLoss;
pub use focal_loss::FocalLoss;
pub use hyperbolic_cross_entropy::HyperbolicCrossEntropy;
pub use mean_squared_error::MeanSquaredError;
pub use triplet_loss::TripletLoss;

/// Trait implemented by differentiable losses that operate directly on
/// SpiralTorch tensors.
pub trait Loss {
    /// Computes the loss value for the given predictions and targets.
    fn forward(&mut self, prediction: &Tensor, target: &Tensor) -> PureResult<Tensor>;

    /// Returns the gradient of the loss with respect to the predictions.
    fn backward(&mut self, prediction: &Tensor, target: &Tensor) -> PureResult<Tensor>;
}

pub(crate) fn probability_floor(epsilon: f32, fallback: f32) -> f32 {
    let value = if epsilon.is_finite() {
        epsilon
    } else {
        fallback
    };
    value.clamp(1.0e-12, 1.0)
}

pub(crate) fn binary_probability_epsilon(epsilon: f32, fallback: f32) -> f32 {
    let value = if epsilon.is_finite() {
        epsilon
    } else {
        fallback
    };
    value.clamp(1.0e-12, 0.499_999)
}

pub(crate) fn checked_loss_value(label: &'static str, value: f32) -> PureResult<f32> {
    if !value.is_finite() {
        return Err(TensorError::NonFiniteValue { label, value });
    }
    Ok(value)
}

pub(crate) fn validate_loss_tensor(label: &'static str, tensor: &Tensor) -> PureResult<()> {
    for &value in tensor.data() {
        checked_loss_value(label, value)?;
    }
    Ok(())
}

pub(crate) fn relabel_loss_non_finite<T>(
    result: PureResult<T>,
    label: &'static str,
) -> PureResult<T> {
    match result {
        Err(TensorError::NonFiniteValue { value, .. }) => {
            Err(TensorError::NonFiniteValue { label, value })
        }
        other => other,
    }
}

pub(crate) fn validate_loss_inputs(prediction: &Tensor, target: &Tensor) -> PureResult<()> {
    validate_loss_tensor("loss_prediction", prediction)?;
    validate_loss_tensor("loss_target", target)?;
    Ok(())
}

pub(crate) fn emit_loss_backend_meta(
    op_name: &'static str,
    prediction: &Tensor,
    output_shape: (usize, usize),
    reduction: &'static str,
) {
    emit_loss_backend_meta_with_backend(
        op_name,
        prediction,
        output_shape,
        reduction,
        "cpu",
        "auto",
        "scalar",
        None,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_loss_backend_meta_with_backend(
    op_name: &'static str,
    prediction: &Tensor,
    output_shape: (usize, usize),
    reduction: &'static str,
    backend: &'static str,
    requested_backend: &'static str,
    kernel: &'static str,
    fallback_message: Option<&str>,
) {
    let (rows, cols) = prediction.shape();
    emit_tensor_op(op_name, &[rows, cols], &[output_shape.0, output_shape.1]);
    emit_tensor_op_meta(op_name, || {
        let mut data = serde_json::Map::new();
        data.insert("backend".to_string(), serde_json::json!(backend));
        data.insert(
            "requested_backend".to_string(),
            serde_json::json!(requested_backend),
        );
        data.insert("kernel".to_string(), serde_json::json!(kernel));
        data.insert("rows".to_string(), serde_json::json!(rows));
        data.insert("cols".to_string(), serde_json::json!(cols));
        data.insert(
            "values".to_string(),
            serde_json::json!(rows.saturating_mul(cols)),
        );
        data.insert("output_rows".to_string(), serde_json::json!(output_shape.0));
        data.insert("output_cols".to_string(), serde_json::json!(output_shape.1));
        data.insert("reduction".to_string(), serde_json::json!(reduction));
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

pub(crate) fn reduce_loss_terms(
    terms: &Tensor,
    scale: f32,
    term_label: &'static str,
    reduction_label: &'static str,
    loss_label: &'static str,
) -> PureResult<f32> {
    validate_loss_tensor(term_label, terms)?;
    let backend = current_tensor_util_backend_for_values(terms.data().len());
    let reduced = relabel_loss_non_finite(
        terms.try_sum_axis0_scaled_with_backend(scale, backend),
        reduction_label,
    )?;
    let mut total = 0.0f32;
    for value in reduced {
        let value = checked_loss_value(reduction_label, value)?;
        total = checked_loss_value(loss_label, total + value)?;
    }
    Ok(total)
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
    fn probability_epsilon_helpers_keep_clamp_ranges_valid() {
        assert_eq!(probability_floor(f32::NAN, 1.0e-6), 1.0e-6);
        assert_eq!(probability_floor(-1.0, 1.0e-6), 1.0e-12);
        assert_eq!(probability_floor(2.0, 1.0e-6), 1.0);
        assert_eq!(binary_probability_epsilon(f32::INFINITY, 1.0e-5), 1.0e-5);
        assert_eq!(binary_probability_epsilon(-1.0, 1.0e-5), 1.0e-12);
        assert!(binary_probability_epsilon(0.9, 1.0e-5) < 0.5);
    }

    #[test]
    fn mse_loss_emits_cpu_backend_metadata() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&events);
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push((event.op_name, event.data.clone()));
        })));

        let mut loss = MeanSquaredError::new();
        let prediction = Tensor::from_vec(1, 2, vec![0.5, -0.5]).unwrap();
        let target = Tensor::zeros(1, 2).unwrap();
        let _ = loss.forward(&prediction, &target).unwrap();
        let _ = loss.backward(&prediction, &target).unwrap();

        st_tensor::set_tensor_op_meta_observer(previous);
        let events = events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let forward = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "mse_loss_forward" && data["rows"] == 1 && data["cols"] == 2
            })
            .expect("mse forward metadata");
        assert_eq!(forward.1["backend"], "cpu");
        assert_eq!(forward.1["requested_backend"], "auto");
        assert_eq!(forward.1["rows"], 1);
        assert_eq!(forward.1["cols"], 2);
        assert_eq!(forward.1["output_rows"], 1);
        assert_eq!(forward.1["output_cols"], 1);

        let backward = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "mse_loss_backward" && data["rows"] == 1 && data["cols"] == 2
            })
            .expect("mse backward metadata");
        assert_eq!(backward.1["backend"], "cpu");
        assert_eq!(backward.1["output_rows"], 1);
        assert_eq!(backward.1["output_cols"], 2);

        let sub = events
            .iter()
            .find(|(op_name, data)| *op_name == "sub" && data["rows"] == 1 && data["cols"] == 2)
            .expect("mse difference metadata");
        assert!(sub.1["backend"].as_str().is_some());

        let hadamard = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "hadamard" && data["rows"] == 1 && data["cols"] == 2
            })
            .expect("mse forward square metadata");
        assert!(hadamard.1["backend"].as_str().is_some());

        let reduction = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "sum_axis0_scaled" && data["rows"] == 1 && data["cols"] == 2
            })
            .expect("mse forward reduction metadata");
        assert!(reduction.1["backend"].as_str().is_some());

        let scale = events
            .iter()
            .find(|(op_name, data)| *op_name == "scale" && data["rows"] == 1 && data["cols"] == 2)
            .expect("mse backward reduction scale metadata");
        assert!(scale.1["backend"].as_str().is_some());
    }

    #[test]
    fn supervised_loss_forward_reductions_emit_tensor_utility_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&events);
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push((event.op_name, event.data.clone()));
        })));

        let prediction = Tensor::from_vec(2, 2, vec![0.8, 0.2, 0.3, 0.7]).unwrap();
        let target = Tensor::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let mut categorical = CategoricalCrossEntropy::new();
        let _ = categorical.forward(&prediction, &target).unwrap();
        let mut focal = FocalLoss::new(0.25, 2.0);
        let _ = focal.forward(&prediction, &target).unwrap();
        let mut hyperbolic = HyperbolicCrossEntropy::new(-1.0).unwrap();
        let _ = hyperbolic.forward(&prediction, &target).unwrap();
        let mut contrastive = ContrastiveLoss::new(1.0);
        let _ = contrastive.forward(&prediction, &target).unwrap();
        let mut triplet = TripletLoss::new(1.0);
        let _ = triplet.forward(&prediction, &target).unwrap();

        st_tensor::set_tensor_op_meta_observer(previous);
        let events = events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let reductions = events
            .iter()
            .filter(|(op_name, data)| {
                *op_name == "sum_axis0_scaled" && data["backend"].as_str().is_some()
            })
            .count();
        assert!(reductions >= 5);
    }
}
