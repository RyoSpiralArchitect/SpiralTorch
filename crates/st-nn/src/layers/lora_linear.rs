// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::execution::{
    current_matmul_backend, current_prepacked_matmul_backend,
    current_tensor_util_backend_for_values,
};
use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};
use std::collections::HashMap;

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

fn relabel_non_finite<T>(result: PureResult<T>, label: &'static str) -> PureResult<T> {
    match result {
        Err(TensorError::NonFiniteValue { value, .. }) => {
            Err(TensorError::NonFiniteValue { label, value })
        }
        other => other,
    }
}

/// Frozen-base linear layer with trainable low-rank LoRA adapter matrices.
#[derive(Debug)]
pub struct LoraLinear {
    name: String,
    base_weight: Parameter,
    base_bias: Tensor,
    lora_a: Parameter,
    lora_b: Parameter,
    rank: usize,
    alpha: f32,
}

impl LoraLinear {
    /// Creates a LoRA linear layer with deterministic base and adapter values.
    ///
    /// `rank` must fit within the effective rank of the base matrix and `alpha`
    /// must be finite and strictly positive.
    pub fn new(
        name: impl Into<String>,
        input_dim: usize,
        output_dim: usize,
        rank: usize,
        alpha: f32,
    ) -> PureResult<Self> {
        if input_dim == 0 || output_dim == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: input_dim,
                cols: output_dim,
            });
        }
        if rank == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: rank,
                cols: output_dim,
            });
        }
        if rank > input_dim.min(output_dim) {
            return Err(TensorError::InvalidValue { label: "lora_rank" });
        }
        if !alpha.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "lora_alpha",
                value: alpha,
            });
        }
        if alpha <= 0.0 {
            return Err(TensorError::InvalidValue {
                label: "lora_alpha",
            });
        }
        let name = name.into();
        let mut scale = 0.01f32;
        let base_weight = Tensor::from_fn(input_dim, output_dim, |_r, _c| {
            let value = scale;
            scale += 0.01;
            value
        })?;
        let base_bias = Tensor::zeros(1, output_dim)?;
        let lora_a = Tensor::from_fn(input_dim, rank, |r, c| ((r + c + 1) as f32) * 1.0e-3)?;
        let lora_b = Tensor::zeros(rank, output_dim)?;
        Ok(Self {
            name: name.clone(),
            base_weight: Parameter::new(format!("{name}::weight"), base_weight),
            base_bias,
            lora_a: Parameter::new(format!("{name}::lora_a"), lora_a),
            lora_b: Parameter::new(format!("{name}::lora_b"), lora_b),
            rank,
            alpha,
        })
    }

    /// Returns the base frozen weight tensor.
    pub fn base_weight(&self) -> &Tensor {
        self.base_weight.value()
    }

    /// Returns the base frozen bias tensor.
    pub fn base_bias(&self) -> &Tensor {
        &self.base_bias
    }

    /// Returns the LoRA rank.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Returns the LoRA alpha.
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Returns alpha / rank, the scale applied to the adapter path.
    pub fn scale(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    /// Captures frozen base tensors keyed like a regular linear layer.
    pub fn base_state_dict(&self) -> PureResult<HashMap<String, Tensor>> {
        let mut state = HashMap::new();
        state.insert(
            format!("{}::weight", self.name),
            self.base_weight.value().clone(),
        );
        state.insert(format!("{}::bias", self.name), self.base_bias.clone());
        Ok(state)
    }

    /// Restores frozen base tensors from a base-state dictionary.
    pub fn load_base_state_dict(&mut self, state: &HashMap<String, Tensor>) -> PureResult<()> {
        let weight_name = format!("{}::weight", self.name);
        let bias_name = format!("{}::bias", self.name);
        let Some(weight) = state.get(&weight_name) else {
            return Err(TensorError::MissingParameter { name: weight_name });
        };
        let Some(bias) = state.get(&bias_name) else {
            return Err(TensorError::MissingParameter { name: bias_name });
        };
        if weight.shape() != self.base_weight.value().shape() {
            return Err(TensorError::ShapeMismatch {
                left: self.base_weight.value().shape(),
                right: weight.shape(),
            });
        }
        if bias.shape() != self.base_bias.shape() {
            return Err(TensorError::ShapeMismatch {
                left: self.base_bias.shape(),
                right: bias.shape(),
            });
        }
        validate_finite_tensor("lora_base_weight", weight)?;
        validate_finite_tensor("lora_base_bias", bias)?;
        self.base_weight.load_value(weight)?;
        self.base_bias = bias.clone();
        Ok(())
    }

    fn validate_tensors(&self, input: Option<&Tensor>) -> PureResult<()> {
        if let Some(input) = input {
            validate_finite_tensor("lora_linear_input", input)?;
        }
        validate_finite_tensor("lora_base_weight", self.base_weight.value())?;
        validate_finite_tensor("lora_base_bias", &self.base_bias)?;
        validate_finite_tensor("lora_a", self.lora_a.value())?;
        validate_finite_tensor("lora_b", self.lora_b.value())
    }
}

impl Module for LoraLinear {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        if input.shape().1 != self.base_weight.value().shape().0 {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: self.base_weight.value().shape(),
            });
        }
        self.validate_tensors(Some(input))?;
        let base_pack = self.base_weight.ensure_matmul_pack()?;
        let mut output = relabel_non_finite(
            input.matmul_prepacked_bias_with_backend(
                &base_pack,
                self.base_bias.data(),
                current_prepacked_matmul_backend(),
            ),
            "lora_linear_base_output",
        )?;

        let adapter_a_pack = self.lora_a.ensure_matmul_pack()?;
        let hidden = relabel_non_finite(
            input
                .matmul_prepacked_with_backend(&adapter_a_pack, current_prepacked_matmul_backend()),
            "lora_linear_hidden",
        )?;
        let adapter_b_pack = self.lora_b.ensure_matmul_pack()?;
        let adapter = relabel_non_finite(
            hidden
                .matmul_prepacked_with_backend(&adapter_b_pack, current_prepacked_matmul_backend()),
            "lora_linear_adapter_output",
        )?;
        let backend = current_tensor_util_backend_for_values(output.data().len());
        relabel_non_finite(
            output.add_scaled_with_backend(&adapter, self.scale(), backend),
            "lora_linear_output",
        )?;
        validate_finite_tensor("lora_linear_output", &output)?;
        Ok(output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        if input.shape().0 != grad_output.shape().0 {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: grad_output.shape(),
            });
        }
        let (rows, input_cols) = input.shape();
        let (_, output_cols) = grad_output.shape();
        let (base_rows, base_cols) = self.base_weight.value().shape();
        if input_cols != base_rows || output_cols != base_cols {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: (rows, base_cols),
            });
        }
        self.validate_tensors(Some(input))?;
        validate_finite_tensor("lora_linear_grad_output", grad_output)?;
        if rows == 0 {
            return Tensor::zeros(rows, input_cols);
        }

        let batch = rows as f32;
        let scale = self.scale();
        let adapter_a_pack = self.lora_a.ensure_matmul_pack()?;
        let hidden = relabel_non_finite(
            input
                .matmul_prepacked_with_backend(&adapter_a_pack, current_prepacked_matmul_backend()),
            "lora_linear_hidden",
        )?;
        let grad_b = relabel_non_finite(
            hidden.matmul_lhs_transpose_scaled_with_backend(
                grad_output,
                scale / batch,
                current_matmul_backend(),
            ),
            "lora_b_grad",
        )?;

        let adapter_b_transpose_pack = self.lora_b.ensure_matmul_transpose_pack()?;
        let grad_adapter_hidden = relabel_non_finite(
            grad_output.matmul_prepacked_with_backend(
                &adapter_b_transpose_pack,
                current_prepacked_matmul_backend(),
            ),
            "lora_adapter_hidden_grad_unscaled",
        )?;
        let grad_adapter_hidden = relabel_non_finite(
            grad_adapter_hidden.scale_with_backend(
                scale,
                current_tensor_util_backend_for_values(grad_adapter_hidden.data().len()),
            ),
            "lora_adapter_hidden_grad",
        )?;
        let grad_a = relabel_non_finite(
            input.matmul_lhs_transpose_scaled_with_backend(
                &grad_adapter_hidden,
                1.0 / batch,
                current_matmul_backend(),
            ),
            "lora_a_grad",
        )?;

        let base_transpose_pack = self.base_weight.ensure_matmul_transpose_pack()?;
        let mut grad_input = relabel_non_finite(
            grad_output.matmul_prepacked_with_backend(
                &base_transpose_pack,
                current_prepacked_matmul_backend(),
            ),
            "lora_base_input_grad",
        )?;
        let adapter_a_transpose_pack = self.lora_a.ensure_matmul_transpose_pack()?;
        let adapter_input = relabel_non_finite(
            grad_adapter_hidden.matmul_prepacked_with_backend(
                &adapter_a_transpose_pack,
                current_prepacked_matmul_backend(),
            ),
            "lora_adapter_input_grad",
        )?;
        let backend = current_tensor_util_backend_for_values(grad_input.data().len());
        relabel_non_finite(
            grad_input.add_scaled_with_backend(&adapter_input, 1.0, backend),
            "lora_input_grad",
        )?;

        validate_finite_tensor("lora_a_grad", &grad_a)?;
        validate_finite_tensor("lora_b_grad", &grad_b)?;
        validate_finite_tensor("lora_input_grad", &grad_input)?;
        self.lora_a.accumulate_euclidean(&grad_a)?;
        self.lora_b.accumulate_euclidean(&grad_b)?;
        Ok(grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.lora_a)?;
        visitor(&self.lora_b)?;
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.lora_a)?;
        visitor(&mut self.lora_b)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_non_finite_label<T>(result: PureResult<T>, expected: &'static str) {
        match result {
            Err(TensorError::NonFiniteValue { label, .. }) => assert_eq!(label, expected),
            Err(error) => panic!("expected non-finite label {expected}, got {error:?}"),
            Ok(_) => panic!("expected non-finite label {expected}"),
        }
    }

    #[track_caller]
    fn assert_tensor_close(lhs: &Tensor, rhs: &Tensor, tolerance: f32) {
        assert_eq!(lhs.shape(), rhs.shape());
        for (idx, (&left, &right)) in lhs.data().iter().zip(rhs.data()).enumerate() {
            let delta = (left - right).abs();
            assert!(
                delta <= tolerance,
                "tensor mismatch at {idx}: left={left} right={right} delta={delta} tolerance={tolerance}"
            );
        }
    }

    fn set_adapter(layer: &mut LoraLinear, a: Vec<f32>, b: Vec<f32>) {
        let (input_dim, rank) = layer.lora_a.value().shape();
        let (_, output_dim) = layer.lora_b.value().shape();
        layer
            .lora_a
            .load_value(&Tensor::from_vec(input_dim, rank, a).unwrap())
            .unwrap();
        layer
            .lora_b
            .load_value(&Tensor::from_vec(rank, output_dim, b).unwrap())
            .unwrap();
    }

    fn mean_linear_objective(layer: &LoraLinear, input: &Tensor, grad_output: &Tensor) -> f32 {
        let output = layer.forward(input).unwrap();
        output
            .data()
            .iter()
            .zip(grad_output.data())
            .map(|(value, grad)| value * grad)
            .sum::<f32>()
            / input.shape().0 as f32
    }

    #[test]
    fn lora_linear_forward_matches_base_when_adapter_b_is_zero() {
        let layer = LoraLinear::new("head", 3, 2, 2, 4.0).unwrap();
        let input = Tensor::from_vec(1, 3, vec![1.0, -2.0, 0.5]).unwrap();
        let output = layer.forward(&input).unwrap();
        let mut expected = input.matmul(layer.base_weight()).unwrap();
        expected.add_row_inplace(layer.base_bias().data()).unwrap();
        assert_eq!(output, expected);
    }

    #[test]
    fn lora_linear_backward_accumulates_adapter_gradients_only() {
        let mut layer = LoraLinear::new("head", 3, 2, 2, 4.0).unwrap();
        let input = Tensor::from_vec(2, 3, vec![1.0, 0.0, -1.0, 0.5, 0.25, 0.75]).unwrap();
        let grad = Tensor::from_vec(2, 2, vec![0.1, -0.2, 0.3, 0.4]).unwrap();
        let grad_input = layer.backward(&input, &grad).unwrap();
        assert_eq!(grad_input.shape(), input.shape());
        let state = layer.state_dict().unwrap();
        assert!(state.contains_key("head::lora_a"));
        assert!(state.contains_key("head::lora_b"));
        assert!(!state.contains_key("head::weight"));
        assert!(!state.contains_key("head::bias"));
    }

    #[test]
    fn lora_linear_optimizer_step_invalidates_adapter_packs() {
        let mut layer = LoraLinear::new("head", 2, 1, 1, 2.0).unwrap();
        let input = Tensor::from_vec(1, 2, vec![0.5, -0.1]).unwrap();
        let grad_output = Tensor::from_vec(1, 1, vec![0.3]).unwrap();

        let before = layer.forward(&input).unwrap();
        layer.backward(&input, &grad_output).unwrap();
        layer.apply_step(0.05).unwrap();
        let after = layer.forward(&input).unwrap();

        assert_ne!(before, after);
    }

    #[test]
    fn lora_linear_requires_positive_alpha() {
        assert!(matches!(
            LoraLinear::new("head", 3, 2, 2, 0.0),
            Err(TensorError::InvalidValue {
                label: "lora_alpha"
            })
        ));
        assert!(matches!(
            LoraLinear::new("head", 3, 2, 2, -1.0),
            Err(TensorError::InvalidValue {
                label: "lora_alpha"
            })
        ));
        assert_non_finite_label(LoraLinear::new("head", 3, 2, 2, f32::NAN), "lora_alpha");
    }

    #[test]
    fn lora_linear_rejects_rank_above_the_effective_matrix_rank() {
        assert!(matches!(
            LoraLinear::new("head", 3, 2, 3, 3.0),
            Err(TensorError::InvalidValue { label: "lora_rank" })
        ));
    }

    #[test]
    fn lora_linear_backward_matches_adapter_finite_differences() {
        const EPSILON: f32 = 1.0e-3;
        const TOLERANCE: f32 = 2.5e-3;

        let mut layer = LoraLinear::new("head", 2, 2, 2, 2.0).unwrap();
        set_adapter(
            &mut layer,
            vec![0.2, -0.3, 0.4, 0.1],
            vec![0.5, -0.2, 0.3, 0.6],
        );
        let input = Tensor::from_vec(2, 2, vec![0.7, -0.4, -0.2, 0.9]).unwrap();
        let grad_output = Tensor::from_vec(2, 2, vec![0.3, -0.5, 0.8, 0.2]).unwrap();

        let grad_input = layer.backward(&input, &grad_output).unwrap();
        let analytic_a = layer.lora_a.gradient().unwrap().clone();
        let analytic_b = layer.lora_b.gradient().unwrap().clone();

        for idx in 0..layer.lora_a.value().data().len() {
            let original = layer.lora_a.value().data()[idx];
            layer.lora_a.value_mut().data_mut()[idx] = original + EPSILON;
            let plus = mean_linear_objective(&layer, &input, &grad_output);
            layer.lora_a.value_mut().data_mut()[idx] = original - EPSILON;
            let minus = mean_linear_objective(&layer, &input, &grad_output);
            layer.lora_a.value_mut().data_mut()[idx] = original;
            let numeric = (plus - minus) / (2.0 * EPSILON);
            let delta = (analytic_a.data()[idx] - numeric).abs();
            assert!(
                delta <= TOLERANCE,
                "lora_a finite difference mismatch at {idx}: analytic={} numeric={numeric} delta={delta}",
                analytic_a.data()[idx]
            );
        }

        for idx in 0..layer.lora_b.value().data().len() {
            let original = layer.lora_b.value().data()[idx];
            layer.lora_b.value_mut().data_mut()[idx] = original + EPSILON;
            let plus = mean_linear_objective(&layer, &input, &grad_output);
            layer.lora_b.value_mut().data_mut()[idx] = original - EPSILON;
            let minus = mean_linear_objective(&layer, &input, &grad_output);
            layer.lora_b.value_mut().data_mut()[idx] = original;
            let numeric = (plus - minus) / (2.0 * EPSILON);
            let delta = (analytic_b.data()[idx] - numeric).abs();
            assert!(
                delta <= TOLERANCE,
                "lora_b finite difference mismatch at {idx}: analytic={} numeric={numeric} delta={delta}",
                analytic_b.data()[idx]
            );
        }

        let mut expected_input = grad_output
            .matmul(&layer.base_weight().transpose())
            .unwrap();
        let adapter_hidden = grad_output
            .matmul(&layer.lora_b.value().transpose())
            .unwrap()
            .scale(layer.scale())
            .unwrap();
        let adapter_input = adapter_hidden
            .matmul(&layer.lora_a.value().transpose())
            .unwrap();
        expected_input.add_scaled(&adapter_input, 1.0).unwrap();
        assert_tensor_close(&grad_input, &expected_input, 1.0e-6);
    }

    #[test]
    fn lora_linear_backward_rejects_non_finite_input_grad_without_updates() {
        let mut layer = LoraLinear::new("head", 1, 1, 1, 1.0).unwrap();
        let mut state = layer.base_state_dict().unwrap();
        state.insert(
            "head::weight".to_string(),
            Tensor::from_vec(1, 1, vec![f32::MAX]).unwrap(),
        );
        layer.load_base_state_dict(&state).unwrap();
        let input = Tensor::from_vec(1, 1, vec![0.0]).unwrap();
        let grad_output = Tensor::from_vec(1, 1, vec![2.0]).unwrap();

        assert_non_finite_label(layer.backward(&input, &grad_output), "lora_base_input_grad");
        assert!(layer.lora_a.gradient().is_none());
        assert!(layer.lora_b.gradient().is_none());
    }

    #[test]
    fn lora_linear_empty_batch_is_a_noop_for_adapter_gradients() {
        let mut layer = LoraLinear::new("head", 3, 2, 2, 4.0).unwrap();
        let input = Tensor::from_vec(0, 3, Vec::new()).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), (0, 2));
        assert!(output.data().is_empty());

        let grad_output = Tensor::from_vec(0, 2, Vec::new()).unwrap();
        let grad_input = layer.backward(&input, &grad_output).unwrap();
        assert_eq!(grad_input.shape(), (0, 3));
        assert!(grad_input.data().is_empty());
        assert!(layer.lora_a.gradient().is_none());
        assert!(layer.lora_b.gradient().is_none());
    }

    #[test]
    fn lora_linear_base_state_round_trips() {
        let source = LoraLinear::new("head", 3, 2, 2, 4.0).unwrap();
        let mut target = LoraLinear::new("head", 3, 2, 2, 4.0).unwrap();
        let state = source.base_state_dict().unwrap();
        target.load_base_state_dict(&state).unwrap();
        assert_eq!(target.base_state_dict().unwrap(), state);
    }

    #[test]
    fn lora_linear_base_state_load_invalidates_frozen_weight_pack() {
        let mut layer = LoraLinear::new("head", 2, 1, 1, 2.0).unwrap();
        let input = Tensor::from_vec(1, 2, vec![0.5, -0.1]).unwrap();
        let before = layer.forward(&input).unwrap();
        let mut state = layer.base_state_dict().unwrap();
        state.insert(
            "head::weight".to_string(),
            Tensor::from_vec(2, 1, vec![0.4, -0.6]).unwrap(),
        );
        state.insert(
            "head::bias".to_string(),
            Tensor::from_vec(1, 1, vec![0.2]).unwrap(),
        );

        layer.load_base_state_dict(&state).unwrap();
        let after = layer.forward(&input).unwrap();
        let mut expected = input.matmul(layer.base_weight()).unwrap();
        expected.add_row_inplace(layer.base_bias().data()).unwrap();

        assert_ne!(before, after);
        assert_eq!(after, expected);
    }

    #[test]
    fn lora_linear_base_state_load_is_transactional() {
        let mut layer = LoraLinear::new("head", 3, 2, 2, 4.0).unwrap();
        let before = layer.base_state_dict().unwrap();
        let mut invalid = before.clone();
        invalid.insert(
            "head::weight".to_string(),
            Tensor::from_vec(3, 2, vec![0.4; 6]).unwrap(),
        );
        let mut bias = Tensor::from_vec(1, 2, vec![0.1, 0.2]).unwrap();
        bias.data_mut()[1] = f32::NAN;
        invalid.insert("head::bias".to_string(), bias);

        assert_non_finite_label(layer.load_base_state_dict(&invalid), "lora_base_bias");
        assert_eq!(layer.base_state_dict().unwrap(), before);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn lora_linear_forced_wgpu_matches_cpu_and_routes_every_tensor_stage() {
        use crate::execution::{push_backend_policy, BackendPolicy, ExecutionConfig};
        use st_core::backend::device_caps::DeviceCaps;
        use st_core::backend::execution_plan::AcceleratorFallback;
        use std::sync::{Arc, Mutex};

        let _lock = crate::test_global_state_lock();
        let mut cpu_layer = LoraLinear::new("head", 9, 7, 5, 10.0).unwrap();
        let mut wgpu_layer = LoraLinear::new("head", 9, 7, 5, 10.0).unwrap();
        let a = (0..45)
            .map(|idx| ((idx as f32 * 0.17).sin() * 0.08) + 0.01)
            .collect::<Vec<_>>();
        let b = (0..35)
            .map(|idx| ((idx as f32 * 0.11).cos() * 0.07) - 0.015)
            .collect::<Vec<_>>();
        set_adapter(&mut cpu_layer, a.clone(), b.clone());
        set_adapter(&mut wgpu_layer, a, b);
        let input = Tensor::from_fn(17, 9, |row, col| {
            (((row * 19 + col * 7) as f32) * 0.031).sin() * 0.6
        })
        .unwrap();
        let grad_output = Tensor::from_fn(17, 7, |row, col| {
            (((row * 13 + col * 5) as f32) * 0.023).cos() * 0.4
        })
        .unwrap();

        let config = ExecutionConfig::new(AcceleratorFallback::Allow, 1);
        let cpu_policy = BackendPolicy::from_device_caps_with_config(DeviceCaps::cpu(), config);
        let (cpu_output, cpu_grad_input) = {
            let _guard = push_backend_policy(cpu_policy);
            (
                cpu_layer.forward(&input).unwrap(),
                cpu_layer.backward(&input, &grad_output).unwrap(),
            )
        };

        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));
        let wgpu_policy =
            BackendPolicy::from_device_caps_with_config(DeviceCaps::wgpu(32, true, 256), config);
        let (wgpu_output, wgpu_grad_input) = {
            let _guard = push_backend_policy(wgpu_policy);
            (
                wgpu_layer.forward(&input).unwrap(),
                wgpu_layer.backward(&input, &grad_output).unwrap(),
            )
        };
        st_tensor::set_thread_meta_observer(previous);

        assert_tensor_close(&cpu_output, &wgpu_output, 3.0e-4);
        assert_tensor_close(&cpu_grad_input, &wgpu_grad_input, 3.0e-4);
        assert_tensor_close(
            cpu_layer.lora_a.gradient().unwrap(),
            wgpu_layer.lora_a.gradient().unwrap(),
            3.0e-4,
        );
        assert_tensor_close(
            cpu_layer.lora_b.gradient().unwrap(),
            wgpu_layer.lora_b.gradient().unwrap(),
            3.0e-4,
        );

        let events = events.lock().unwrap();
        for op_name in [
            "matmul_prepacked_bias",
            "matmul_prepacked",
            "matmul_lhs_transpose_scaled",
            "scale",
            "add_scaled",
        ] {
            assert!(
                events.iter().any(|(name, data)| {
                    *name == op_name && data["requested_backend"] == "wgpu"
                }),
                "missing forced-WGPU metadata for {op_name}: {events:?}"
            );
            assert!(
                events.iter().all(|(name, data)| {
                    *name != op_name || data["requested_backend"] != "auto"
                }),
                "{op_name} escaped the active backend policy"
            );
        }
    }
}
