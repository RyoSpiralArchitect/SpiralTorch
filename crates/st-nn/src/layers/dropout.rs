// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::execution::current_tensor_util_backend_for_values;
use crate::module::Module;
use crate::{PureResult, Tensor};
use rand::rngs::StdRng;
use rand::Rng;
use spiral_config::determinism;
use st_tensor::{emit_tensor_op, emit_tensor_op_meta, TensorError};
use std::cell::RefCell;

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

fn emit_dropout_meta(
    op_name: &'static str,
    rows: usize,
    cols: usize,
    backward: bool,
    training: bool,
    probability: f32,
    keep_probability: f32,
    retained_values: usize,
    mask_backend: Option<String>,
) {
    let input_shape = if backward {
        vec![rows, cols, rows, cols]
    } else {
        vec![rows, cols]
    };
    emit_tensor_op(op_name, &input_shape, &[rows, cols]);
    emit_tensor_op_meta(op_name, || {
        let values = rows.saturating_mul(cols);
        let backend = if training { "composite" } else { "identity" };
        let rng_backend = if training { Some("cpu") } else { None };
        serde_json::json!({
            "backend": backend,
            "requested_backend": "auto",
            "kernel": "dropout.mask",
            "kind": if training {
                if backward { "dropout_backward_mask" } else { "dropout_forward_mask" }
            } else if backward {
                "dropout_backward_identity"
            } else {
                "dropout_forward_identity"
            },
            "rows": rows,
            "cols": cols,
            "values": values,
            "output_rows": rows,
            "output_cols": cols,
            "output_values": values,
            "training": training,
            "probability": probability,
            "keep_probability": keep_probability,
            "keep_scale": if keep_probability > 0.0 { 1.0 / keep_probability } else { 0.0 },
            "retained_values": retained_values,
            "dropped_values": values.saturating_sub(retained_values),
            "rng_backend": rng_backend,
            "mask_backend": mask_backend,
            "estimated_mask_values": if training { values } else { 0 },
            "empty": rows == 0 || cols == 0,
        })
    });
}

/// Stochastic dropout layer that scales retained activations during training
/// and becomes a no-op when evaluation mode is enabled.
#[derive(Debug)]
pub struct Dropout {
    probability: f32,
    training: bool,
    rng: RefCell<StdRng>,
    last_mask: RefCell<Option<Tensor>>,
}

impl Dropout {
    /// Creates a new dropout layer with the provided drop probability.
    ///
    /// `probability` must lie in `[0, 1)`. During training the layer retains
    /// activations with probability `1 - probability` and rescales them by the
    /// inverse keep probability to preserve expectation.
    pub fn new(probability: f32) -> PureResult<Self> {
        Self::with_seed(probability, None)
    }

    /// Creates a new dropout layer with a deterministic seed used for unit
    /// tests and reproducible experiments.
    pub fn with_seed(probability: f32, seed: Option<u64>) -> PureResult<Self> {
        if !(0.0..1.0).contains(&probability) {
            return Err(TensorError::InvalidValue {
                label: "dropout_probability",
            });
        }
        let rng = determinism::rng_from_optional(seed, "st-nn/layers/dropout");
        Ok(Self {
            probability,
            training: true,
            rng: RefCell::new(rng),
            last_mask: RefCell::new(None),
        })
    }

    /// Returns the configured drop probability.
    pub fn probability(&self) -> f32 {
        self.probability
    }

    /// Sets the layer to training (`true`) or evaluation (`false`) mode.
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
        if !training {
            self.last_mask.borrow_mut().take();
        }
    }

    /// Returns whether the layer is currently in training mode.
    pub fn training(&self) -> bool {
        self.training
    }

    /// Convenience helper that switches the layer into training mode and clears
    /// any cached inference mask.
    pub fn train(&mut self) {
        self.set_training(true);
    }

    /// Convenience helper that switches the layer into evaluation mode,
    /// dropping any cached stochastic mask so gradients flow transparently.
    pub fn eval(&mut self) {
        self.set_training(false);
    }

    fn keep_probability(&self) -> f32 {
        1.0 - self.probability
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let keep_probability = self.keep_probability();
        let (rows, cols) = input.shape();
        validate_finite_tensor("dropout_input", input)?;
        if !self.training || keep_probability == 1.0 {
            self.last_mask.borrow_mut().take();
            emit_dropout_meta(
                "dropout_forward",
                rows,
                cols,
                false,
                false,
                self.probability,
                keep_probability,
                rows.saturating_mul(cols),
                None,
            );
            return Ok(input.clone());
        }

        if rows == 0 || cols == 0 {
            return Err(TensorError::EmptyInput("dropout forward input"));
        }

        let mut rng = self.rng.borrow_mut();
        let mut mask_data = Vec::with_capacity(rows * cols);
        let mut retained_values = 0usize;
        for _ in 0..rows * cols {
            let sample: f32 = rng.gen();
            if sample < keep_probability {
                retained_values = retained_values.saturating_add(1);
                mask_data.push(1.0 / keep_probability);
            } else {
                mask_data.push(0.0);
            }
        }
        drop(rng);

        let mask = Tensor::from_vec(rows, cols, mask_data)?;
        validate_finite_tensor("dropout_mask", &mask)?;
        let mask_backend = current_tensor_util_backend_for_values(input.data().len());
        let output = input.hadamard_with_backend(&mask, mask_backend)?;
        validate_finite_tensor("dropout_output", &output)?;
        self.last_mask.borrow_mut().replace(mask);
        emit_dropout_meta(
            "dropout_forward",
            rows,
            cols,
            false,
            true,
            self.probability,
            keep_probability,
            retained_values,
            Some(mask_backend.to_string()),
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
        let keep_probability = self.keep_probability();
        let (rows, cols) = grad_output.shape();
        validate_finite_tensor("dropout_backward_input", input)?;
        validate_finite_tensor("dropout_backward_grad_output", grad_output)?;
        if !self.training || keep_probability == 1.0 {
            emit_dropout_meta(
                "dropout_backward",
                rows,
                cols,
                true,
                false,
                self.probability,
                keep_probability,
                rows.saturating_mul(cols),
                None,
            );
            return Ok(grad_output.clone());
        }

        let mask_guard = self.last_mask.borrow();
        let Some(mask) = mask_guard.as_ref() else {
            return Err(TensorError::InvalidValue {
                label: "dropout_mask_missing",
            });
        };
        if mask.shape() != grad_output.shape() {
            return Err(TensorError::ShapeMismatch {
                left: mask.shape(),
                right: grad_output.shape(),
            });
        }
        validate_finite_tensor("dropout_mask", mask)?;
        let retained_values = mask.data().iter().filter(|value| **value != 0.0).count();
        let mask_backend = current_tensor_util_backend_for_values(grad_output.data().len());
        let output = grad_output.hadamard_with_backend(mask, mask_backend)?;
        validate_finite_tensor("dropout_backward_output", &output)?;
        emit_dropout_meta(
            "dropout_backward",
            rows,
            cols,
            true,
            true,
            self.probability,
            keep_probability,
            retained_values,
            Some(mask_backend.to_string()),
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

    fn set_training(&mut self, training: bool) -> PureResult<()> {
        Dropout::set_training(self, training);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "wgpu")]
    use crate::execution::{push_backend_policy, BackendPolicy};
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;
    #[cfg(feature = "wgpu")]
    use st_core::backend::device_caps::DeviceCaps;
    #[cfg(feature = "wgpu")]
    use st_tensor::backend::wgpu_dense;
    use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

    fn observer_lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    #[test]
    fn dropout_forward_matches_mask() {
        let layer = Dropout::with_seed(0.25, Some(42)).unwrap();
        let input = Tensor::from_vec(1, 6, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let output = layer.forward(&input).unwrap();

        let keep_probability = 1.0 - layer.probability();
        let mut rng = StdRng::seed_from_u64(42);
        let mut expected = Vec::new();
        for value in input.data() {
            let sample: f32 = rng.gen();
            if sample < keep_probability {
                expected.push(value * (1.0 / keep_probability));
            } else {
                expected.push(0.0);
            }
        }
        let expected_tensor = Tensor::from_vec(1, 6, expected).unwrap();
        assert_eq!(output, expected_tensor);
    }

    #[test]
    fn dropout_backward_uses_mask() {
        let mut layer = Dropout::with_seed(0.5, Some(7)).unwrap();
        let input = Tensor::from_vec(1, 4, vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let _ = layer.forward(&input).unwrap();
        let grad_output = Tensor::from_vec(1, 4, vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        let grad_input = layer.backward(&input, &grad_output).unwrap();

        let keep_probability = 1.0 - layer.probability();
        let mut rng = StdRng::seed_from_u64(7);
        let mut expected = Vec::new();
        for grad in grad_output.data() {
            let sample: f32 = rng.gen();
            if sample < keep_probability {
                expected.push(grad * (1.0 / keep_probability));
            } else {
                expected.push(0.0);
            }
        }
        let expected_tensor = Tensor::from_vec(1, 4, expected).unwrap();
        assert_eq!(grad_input, expected_tensor);
    }

    #[test]
    fn dropout_forward_rejects_non_finite_input_without_caching_mask() {
        let layer = Dropout::with_seed(0.5, Some(7)).unwrap();
        let input = Tensor::from_vec(1, 4, vec![1.0, f32::NAN, 2.0, 3.0]).unwrap();

        let err = layer.forward(&input).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "dropout_input",
                value,
            } if value.is_nan()
        ));
        assert!(layer.last_mask.borrow().is_none());
    }

    #[test]
    fn dropout_eval_forward_rejects_non_finite_identity_input() {
        let mut layer = Dropout::with_seed(0.5, Some(7)).unwrap();
        layer.eval();
        let input = Tensor::from_vec(1, 4, vec![1.0, f32::INFINITY, 2.0, 3.0]).unwrap();

        let err = layer.forward(&input).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "dropout_input",
                value,
            } if value.is_infinite()
        ));
        assert!(layer.last_mask.borrow().is_none());
    }

    #[test]
    fn dropout_backward_rejects_non_finite_input_without_masking_it() {
        let mut layer = Dropout::with_seed(0.5, Some(7)).unwrap();
        let forward_input = Tensor::from_vec(1, 4, vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let _ = layer.forward(&forward_input).unwrap();
        let input = Tensor::from_vec(1, 4, vec![1.0, f32::NAN, 1.0, 1.0]).unwrap();
        let grad_output = Tensor::from_vec(1, 4, vec![0.1, 0.2, 0.3, 0.4]).unwrap();

        let err = layer.backward(&input, &grad_output).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "dropout_backward_input",
                value,
            } if value.is_nan()
        ));
    }

    #[test]
    fn dropout_backward_rejects_non_finite_grad_output() {
        let mut layer = Dropout::with_seed(0.5, Some(7)).unwrap();
        let input = Tensor::from_vec(1, 4, vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let _ = layer.forward(&input).unwrap();
        let grad_output = Tensor::from_vec(1, 4, vec![0.1, f32::INFINITY, 0.3, 0.4]).unwrap();

        let err = layer.backward(&input, &grad_output).unwrap_err();

        assert!(matches!(
            err,
            TensorError::NonFiniteValue {
                label: "dropout_backward_grad_output",
                value,
            } if value.is_infinite()
        ));
    }

    #[test]
    fn dropout_backward_rejects_shape_mismatch_before_using_mask() {
        let mut layer = Dropout::with_seed(0.5, Some(7)).unwrap();
        let input = Tensor::from_vec(1, 4, vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let _ = layer.forward(&input).unwrap();
        let bad_grad_output = Tensor::from_vec(2, 2, vec![0.1, 0.2, 0.3, 0.4]).unwrap();

        let err = layer.backward(&input, &bad_grad_output).unwrap_err();

        assert!(matches!(err, TensorError::ShapeMismatch { .. }));
    }

    #[test]
    fn dropout_forward_backward_emit_backend_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let mut layer = Dropout::with_seed(0.5, Some(7)).unwrap();
        let input = Tensor::from_vec(1, 4, vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let grad_output = Tensor::from_vec(1, 4, vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        let _ = layer.forward(&input).unwrap();
        let _ = layer.backward(&input, &grad_output).unwrap();
        st_tensor::set_tensor_op_meta_observer(previous);

        let keep_probability = 1.0 - layer.probability();
        let mut rng = StdRng::seed_from_u64(7);
        let retained = (0..4)
            .filter(|_| {
                let sample: f32 = rng.gen();
                sample < keep_probability
            })
            .count();

        let events = events.lock().unwrap();
        let forward = events
            .iter()
            .find(|(op_name, data)| *op_name == "dropout_forward" && data["cols"] == 4)
            .expect("dropout forward metadata event");
        assert_eq!(forward.1["backend"], "composite");
        assert_eq!(forward.1["kind"], "dropout_forward_mask");
        assert_eq!(forward.1["retained_values"], retained);
        assert_eq!(forward.1["dropped_values"], 4usize.saturating_sub(retained));
        assert_eq!(forward.1["rng_backend"], "cpu");
        assert_eq!(forward.1["mask_backend"], "auto");

        let backward = events
            .iter()
            .find(|(op_name, data)| *op_name == "dropout_backward" && data["cols"] == 4)
            .expect("dropout backward metadata event");
        assert_eq!(backward.1["backend"], "composite");
        assert_eq!(backward.1["kind"], "dropout_backward_mask");
        assert_eq!(backward.1["retained_values"], retained);
        assert_eq!(backward.1["rng_backend"], "cpu");
        assert_eq!(backward.1["mask_backend"], "auto");

        let hadamard_count = events
            .iter()
            .filter(|(op_name, data)| *op_name == "hadamard" && data["cols"] == 4)
            .count();
        assert!(hadamard_count >= 2);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn dropout_forced_wgpu_uses_hadamard_and_marks_mask_application() {
        if !wgpu_dense::is_available() {
            return;
        }
        let _lock = observer_lock();
        let previous_threshold = std::env::var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES").ok();
        std::env::set_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", "1");

        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let input = Tensor::from_fn(3, 4, |row, col| {
            ((row * 13 + col * 5) % 17) as f32 * 0.041 - 0.2
        })
        .unwrap();
        let grad_output = Tensor::from_fn(3, 4, |row, col| {
            ((row * 7 + col * 11) % 19) as f32 * 0.033 - 0.15
        })
        .unwrap();
        let cpu_policy = BackendPolicy::from_device_caps(DeviceCaps::cpu());
        let mut cpu_layer = Dropout::with_seed(0.35, Some(123)).unwrap();
        let cpu_forward = {
            let _guard = push_backend_policy(cpu_policy);
            cpu_layer.forward(&input).unwrap()
        };
        let cpu_grad_input = {
            let _guard = push_backend_policy(cpu_policy);
            cpu_layer.backward(&input, &grad_output).unwrap()
        };

        let wgpu_policy = BackendPolicy::from_device_caps(DeviceCaps::wgpu(32, true, 256));
        let mut wgpu_layer = Dropout::with_seed(0.35, Some(123)).unwrap();
        let (wgpu_forward, wgpu_grad_input) = {
            let _guard = push_backend_policy(wgpu_policy);
            (
                wgpu_layer.forward(&input).unwrap(),
                wgpu_layer.backward(&input, &grad_output).unwrap(),
            )
        };

        st_tensor::set_tensor_op_meta_observer(previous);
        match previous_threshold {
            Some(value) => std::env::set_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", value),
            None => std::env::remove_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES"),
        }

        assert_eq!(cpu_forward.shape(), wgpu_forward.shape());
        for (idx, (&cpu, &wgpu)) in cpu_forward
            .data()
            .iter()
            .zip(wgpu_forward.data().iter())
            .enumerate()
        {
            let delta = (cpu - wgpu).abs();
            assert!(
                delta <= 1e-6,
                "dropout forward mismatch at {idx}: cpu={cpu} wgpu={wgpu} delta={delta}"
            );
        }
        assert_eq!(cpu_grad_input.shape(), wgpu_grad_input.shape());
        for (idx, (&cpu, &wgpu)) in cpu_grad_input
            .data()
            .iter()
            .zip(wgpu_grad_input.data().iter())
            .enumerate()
        {
            let delta = (cpu - wgpu).abs();
            assert!(
                delta <= 1e-6,
                "dropout backward mismatch at {idx}: cpu={cpu} wgpu={wgpu} delta={delta}"
            );
        }

        let events = events.lock().unwrap();
        assert!(events
            .iter()
            .any(|(op_name, data)| *op_name == "hadamard" && data["backend"] == "wgpu_dense"));
        assert!(events.iter().any(|(op_name, data)| {
            *op_name == "dropout_forward"
                && data["backend"] == "composite"
                && data["rng_backend"] == "cpu"
                && data["mask_backend"] == "wgpu"
        }));
        assert!(events.iter().any(|(op_name, data)| {
            *op_name == "dropout_backward"
                && data["backend"] == "composite"
                && data["rng_backend"] == "cpu"
                && data["mask_backend"] == "wgpu"
        }));
    }

    #[test]
    fn dropout_evaluation_is_identity() {
        let mut layer = Dropout::with_seed(0.4, Some(5)).unwrap();
        layer.set_training(false);
        let input = Tensor::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output, input);

        let grad_output = Tensor::from_vec(2, 2, vec![0.5, 0.25, -0.75, 1.25]).unwrap();
        let grad_input = layer.backward(&input, &grad_output).unwrap();
        assert_eq!(grad_input, grad_output);
    }
}
