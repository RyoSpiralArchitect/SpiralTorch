// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::execution::{current_matmul_backend, current_tensor_util_backend_for_values};
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

/// Frozen-base linear layer with trainable low-rank LoRA adapter matrices.
#[derive(Debug)]
pub struct LoraLinear {
    name: String,
    base_weight: Tensor,
    base_bias: Tensor,
    lora_a: Parameter,
    lora_b: Parameter,
    rank: usize,
    alpha: f32,
}

impl LoraLinear {
    /// Creates a LoRA linear layer with deterministic base and adapter values.
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
        if !alpha.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "lora_alpha",
                value: alpha,
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
            base_weight,
            base_bias,
            lora_a: Parameter::new(format!("{name}::lora_a"), lora_a),
            lora_b: Parameter::new(format!("{name}::lora_b"), lora_b),
            rank,
            alpha,
        })
    }

    /// Returns the base frozen weight tensor.
    pub fn base_weight(&self) -> &Tensor {
        &self.base_weight
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
        state.insert(format!("{}::weight", self.name), self.base_weight.clone());
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
        if weight.shape() != self.base_weight.shape() {
            return Err(TensorError::ShapeMismatch {
                left: self.base_weight.shape(),
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
        self.base_weight = weight.clone();
        self.base_bias = bias.clone();
        Ok(())
    }

    fn validate_tensors(&self, input: Option<&Tensor>) -> PureResult<()> {
        if let Some(input) = input {
            validate_finite_tensor("lora_linear_input", input)?;
        }
        validate_finite_tensor("lora_base_weight", &self.base_weight)?;
        validate_finite_tensor("lora_base_bias", &self.base_bias)?;
        validate_finite_tensor("lora_a", self.lora_a.value())?;
        validate_finite_tensor("lora_b", self.lora_b.value())
    }
}

impl Module for LoraLinear {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        if input.shape().1 != self.base_weight.shape().0 {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: self.base_weight.shape(),
            });
        }
        self.validate_tensors(Some(input))?;
        let mut output = input.matmul(&self.base_weight)?;
        output.add_row_inplace(self.base_bias.data())?;

        let hidden = input.matmul(self.lora_a.value())?;
        let adapter = hidden.matmul(self.lora_b.value())?;
        let backend = current_tensor_util_backend_for_values(output.data().len());
        output.add_scaled_with_backend(&adapter, self.scale(), backend)?;
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
        let (base_rows, base_cols) = self.base_weight.shape();
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
        let hidden = input.matmul(self.lora_a.value())?;
        let grad_b = hidden.matmul_lhs_transpose_scaled_with_backend(
            grad_output,
            scale / batch,
            current_matmul_backend(),
        )?;

        let grad_adapter_hidden = grad_output
            .matmul(&self.lora_b.value().transpose())?
            .scale(scale)?;
        let grad_a = input.matmul_lhs_transpose_scaled_with_backend(
            &grad_adapter_hidden,
            1.0 / batch,
            current_matmul_backend(),
        )?;

        let mut grad_input = grad_output.matmul(&self.base_weight.transpose())?;
        let adapter_input = grad_adapter_hidden.matmul(&self.lora_a.value().transpose())?;
        let backend = current_tensor_util_backend_for_values(grad_input.data().len());
        grad_input.add_scaled_with_backend(&adapter_input, 1.0, backend)?;

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
    fn lora_linear_base_state_round_trips() {
        let source = LoraLinear::new("head", 3, 2, 2, 4.0).unwrap();
        let mut target = LoraLinear::new("head", 3, 2, 2, 4.0).unwrap();
        let state = source.base_state_dict().unwrap();
        target.load_base_state_dict(&state).unwrap();
        assert_eq!(target.base_state_dict().unwrap(), state);
    }
}
