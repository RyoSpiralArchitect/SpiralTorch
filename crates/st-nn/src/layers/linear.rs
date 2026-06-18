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

/// Fully-connected layer with an optional hypergrad tape on its parameters.
#[derive(Debug)]
pub struct Linear {
    weight: Parameter,
    bias: Parameter,
}

impl Linear {
    /// Creates a new linear layer with deterministic small parameters.
    pub fn new(name: impl Into<String>, input_dim: usize, output_dim: usize) -> PureResult<Self> {
        if input_dim == 0 || output_dim == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: input_dim,
                cols: output_dim,
            });
        }
        let name = name.into();
        let mut scale = 0.01f32;
        let weights = Tensor::from_fn(input_dim, output_dim, |_r, _c| {
            let value = scale;
            scale += 0.01;
            value
        })?;
        let bias = Tensor::zeros(1, output_dim)?;
        Ok(Self {
            weight: Parameter::new(format!("{name}::weight"), weights),
            bias: Parameter::new(format!("{name}::bias"), bias),
        })
    }

    /// Returns a reference to the weight parameter.
    pub fn weight(&self) -> &Parameter {
        &self.weight
    }

    /// Returns a reference to the bias parameter.
    pub fn bias(&self) -> &Parameter {
        &self.bias
    }

    fn validate_parameters(&self) -> PureResult<()> {
        validate_finite_tensor("linear_weight", self.weight.value())?;
        validate_finite_tensor("linear_bias", self.bias.value())
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        if input.shape().1 != self.weight.value().shape().0 {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: self.weight.value().shape(),
            });
        }
        validate_finite_tensor("linear_input", input)?;
        self.validate_parameters()?;
        let pack = self.weight.ensure_matmul_pack()?;
        let out = relabel_non_finite(
            input.matmul_prepacked_bias_with_backend(
                &pack,
                self.bias.value().data(),
                current_prepacked_matmul_backend(),
            ),
            "linear_output",
        )?;
        validate_finite_tensor("linear_output", &out)?;
        Ok(out)
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
        let (weight_rows, weight_cols) = self.weight.value().shape();
        if input_cols != weight_rows {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: self.weight.value().shape(),
            });
        }
        if output_cols != weight_cols {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: (rows, weight_cols),
            });
        }
        validate_finite_tensor("linear_backward_input", input)?;
        validate_finite_tensor("linear_backward_grad_output", grad_output)?;
        self.validate_parameters()?;
        if rows == 0 {
            return Tensor::zeros(rows, input_cols);
        }

        let batch = rows as f32;
        let grad_w = relabel_non_finite(
            input.matmul_lhs_transpose_scaled_with_backend(
                grad_output,
                1.0 / batch,
                current_matmul_backend(),
            ),
            "linear_weight_grad",
        )?;
        validate_finite_tensor("linear_weight_grad", &grad_w)?;

        let bias_backend = current_tensor_util_backend_for_values(grad_output.data().len());
        let summed = relabel_non_finite(
            grad_output.try_sum_axis0_scaled_with_backend(1.0 / batch, bias_backend),
            "linear_bias_grad",
        )?;
        validate_finite_slice("linear_bias_grad", &summed)?;
        let grad_b = Tensor::from_vec(1, summed.len(), summed)?;
        validate_finite_tensor("linear_bias_grad", &grad_b)?;

        let pack_t = self.weight.ensure_matmul_transpose_pack()?;
        let grad_input = relabel_non_finite(
            grad_output.matmul_prepacked_with_backend(&pack_t, current_prepacked_matmul_backend()),
            "linear_input_grad",
        )?;
        validate_finite_tensor("linear_input_grad", &grad_input)?;

        self.weight.accumulate_euclidean(&grad_w)?;
        self.bias.accumulate_euclidean(&grad_b)?;
        Ok(grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.weight)?;
        visitor(&self.bias)?;
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.weight)?;
        visitor(&mut self.bias)?;
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

    #[cfg(feature = "wgpu")]
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

    #[test]
    fn linear_forward_matches_manual() {
        let layer = Linear::new("fc", 3, 2).unwrap();
        let input = Tensor::from_vec(1, 3, vec![1.0, -2.0, 0.5]).unwrap();
        let output = layer.forward(&input).unwrap();
        let expected = input.matmul(layer.weight.value()).unwrap();
        let mut expected = expected;
        expected.add_row_inplace(layer.bias.value().data()).unwrap();
        assert_eq!(output, expected);
    }

    #[test]
    fn linear_forward_rejects_non_finite_input() {
        let layer = Linear::new("fc", 3, 2).unwrap();
        let mut input = Tensor::from_vec(1, 3, vec![1.0, -2.0, 0.5]).unwrap();
        input.data_mut()[1] = f32::NAN;

        assert_non_finite_label(layer.forward(&input), "linear_input");
    }

    #[test]
    fn linear_forward_rejects_non_finite_parameter_before_pack() {
        let mut layer = Linear::new("fc", 3, 2).unwrap();
        layer.weight.value_mut().data_mut()[0] = f32::NAN;
        let input = Tensor::from_vec(1, 3, vec![1.0, -2.0, 0.5]).unwrap();

        assert_non_finite_label(layer.forward(&input), "linear_weight");
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn linear_forward_forced_wgpu_prepacked_matches_cpu_reference_on_edge_tiles() {
        use crate::execution::{push_backend_policy, BackendPolicy};
        use st_core::backend::device_caps::{BackendKind, DeviceCaps};
        use st_tensor::{AttentionBackend, LayerNormBackend, MatmulBackend, SoftmaxBackend};

        let layer = Linear::new("fc", 8, 6).unwrap();
        let input = Tensor::from_vec(
            12,
            8,
            (0..96)
                .map(|idx| ((idx as f32 * 0.137).sin() * 0.7) + ((idx % 5) as f32 * 0.01))
                .collect(),
        )
        .unwrap();
        let cpu_policy = BackendPolicy::explicit(
            DeviceCaps::cpu(),
            MatmulBackend::CpuNaive,
            MatmulBackend::CpuNaive,
            LayerNormBackend::Cpu,
            AttentionBackend::Cpu,
            SoftmaxBackend::Cpu,
        );
        let wgpu_policy = BackendPolicy::explicit(
            BackendKind::Wgpu.default_caps(),
            MatmulBackend::GpuWgpu,
            MatmulBackend::GpuWgpu,
            LayerNormBackend::Cpu,
            AttentionBackend::Cpu,
            SoftmaxBackend::Cpu,
        );
        let cpu = {
            let _guard = push_backend_policy(cpu_policy);
            layer.forward(&input).unwrap()
        };
        let wgpu = {
            let _guard = push_backend_policy(wgpu_policy);
            match layer.forward(&input) {
                Ok(value) => value,
                Err(TensorError::BackendFailure { backend, .. }) if backend == "wgpu" => return,
                Err(error) => panic!("forced WGPU Linear::forward failed: {error:?}"),
            }
        };

        assert_tensor_close(&cpu, &wgpu, 1e-4);
    }

    #[test]
    fn linear_backward_streams_hypergrad() {
        let mut layer = Linear::new("fc", 4, 3).unwrap();
        layer
            .attach_hypergrad(-1.0, 0.05)
            .expect("hypergrad attachment");
        let input =
            Tensor::from_vec(2, 4, vec![0.1, 0.2, -0.3, 0.4, -0.5, 0.6, 0.7, -0.8]).unwrap();
        let target = Tensor::zeros(2, 3).unwrap();
        let output = layer.forward(&input).unwrap();
        let diff = output.sub(&target).unwrap();
        let grad = diff.scale(1.0 / input.shape().0 as f32).unwrap();
        let _ = layer.backward(&input, &grad).unwrap();
        let before = layer.weight().value().clone();
        layer.apply_step(0.01).unwrap();
        let after = layer.weight().value();
        assert_ne!(before, *after);
    }

    #[test]
    fn linear_backward_rejects_non_finite_grad_without_updates() {
        let mut layer = Linear::new("fc", 3, 2).unwrap();
        let input = Tensor::from_vec(1, 3, vec![1.0, -2.0, 0.5]).unwrap();
        let mut grad_output = Tensor::from_vec(1, 2, vec![0.2, -0.4]).unwrap();
        grad_output.data_mut()[0] = f32::NAN;

        assert_non_finite_label(
            layer.backward(&input, &grad_output),
            "linear_backward_grad_output",
        );
        assert!(layer.weight().gradient().is_none());
        assert!(layer.bias().gradient().is_none());
    }

    #[test]
    fn linear_backward_rejects_non_finite_input_grad_without_updates() {
        let mut layer = Linear::new("fc", 1, 1).unwrap();
        layer.weight.value_mut().data_mut()[0] = f32::MAX;
        let input = Tensor::from_vec(1, 1, vec![0.0]).unwrap();
        let grad_output = Tensor::from_vec(1, 1, vec![2.0]).unwrap();

        assert_non_finite_label(layer.backward(&input, &grad_output), "linear_input_grad");
        assert!(layer.weight().gradient().is_none());
        assert!(layer.bias().gradient().is_none());
    }

    #[test]
    fn linear_empty_batch_backward_returns_empty_input_grad_without_updates() {
        let mut layer = Linear::new("fc", 3, 2).unwrap();
        let input = Tensor::from_vec(0, 3, Vec::new()).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), (0, 2));
        assert!(output.data().is_empty());

        let grad_output = Tensor::from_vec(0, 2, Vec::new()).unwrap();
        let grad_input = layer.backward(&input, &grad_output).unwrap();
        assert_eq!(grad_input.shape(), (0, 3));
        assert!(grad_input.data().is_empty());
        assert!(layer.weight().gradient().is_none());
        assert!(layer.bias().gradient().is_none());
    }
}
