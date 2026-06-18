// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::conv::Conv1d;
use super::wave_gate::WaveGate;
use crate::execution::{
    current_matmul_backend, current_prepacked_matmul_backend,
    current_tensor_util_backend_for_values,
};
use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};
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

fn relabel_non_finite<T>(result: PureResult<T>, label: &'static str) -> PureResult<T> {
    match result {
        Err(TensorError::NonFiniteValue { value, .. }) => {
            Err(TensorError::NonFiniteValue { label, value })
        }
        other => other,
    }
}

#[derive(Clone, Debug)]
struct WaveRnnCache {
    input: Tensor,
    gating_in: Tensor,
    final_hidden: Tensor,
    batch: usize,
    out_steps: usize,
}

impl WaveRnnCache {
    fn validate(&self) -> PureResult<()> {
        validate_finite_tensor("wave_rnn_cache_input", &self.input)?;
        validate_finite_tensor("wave_rnn_cache_gating_in", &self.gating_in)?;
        validate_finite_tensor("wave_rnn_cache_final_hidden", &self.final_hidden)
    }
}

/// Convolutional recurrent layer that uses a WaveGate to preserve Z-space phase.
#[derive(Debug)]
pub struct WaveRnn {
    conv: Conv1d,
    gate: WaveGate,
    readout: Parameter,
    readout_bias: Parameter,
    hidden_dim: usize,
    cache: RefCell<Option<WaveRnnCache>>,
}

impl WaveRnn {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        in_channels: usize,
        hidden_dim: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        curvature: f32,
        temperature: f32,
    ) -> PureResult<Self> {
        let name = name.into();
        if hidden_dim == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: 1,
                cols: hidden_dim,
            });
        }
        let conv = Conv1d::new(
            format!("{name}::conv"),
            in_channels,
            hidden_dim,
            kernel_size,
            stride,
            padding,
            1,
        )?;
        let gate = WaveGate::new(format!("{name}::gate"), hidden_dim, curvature, temperature)?;
        let mut seed = 0.005f32;
        let readout_weight = Tensor::from_fn(hidden_dim, hidden_dim, |_r, _c| {
            let value = seed;
            seed = (seed * 1.61).rem_euclid(0.2).max(1e-3);
            value
        })?;
        let readout_bias = Tensor::zeros(1, hidden_dim)?;
        Ok(Self {
            conv,
            gate,
            readout: Parameter::new(format!("{name}::readout"), readout_weight),
            readout_bias: Parameter::new(format!("{name}::bias"), readout_bias),
            hidden_dim,
            cache: RefCell::new(None),
        })
    }

    fn validate_readout_parameters(&self) -> PureResult<()> {
        validate_finite_tensor("wave_rnn_readout", self.readout.value())?;
        validate_finite_tensor("wave_rnn_readout_bias", self.readout_bias.value())
    }
}

impl Module for WaveRnn {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        self.cache.borrow_mut().take();
        validate_finite_tensor("wave_rnn_input", input)?;
        self.validate_readout_parameters()?;
        let conv_out = self.conv.forward(input)?;
        validate_finite_tensor("wave_rnn_conv_output", &conv_out)?;
        let (batch, cols) = conv_out.shape();
        if cols % self.hidden_dim != 0 {
            return Err(TensorError::ShapeMismatch {
                left: conv_out.shape(),
                right: (batch, self.hidden_dim),
            });
        }
        let out_steps = cols / self.hidden_dim;
        let gating_in = conv_out.reshape(batch * out_steps, self.hidden_dim)?;
        validate_finite_tensor("wave_rnn_gating_input", &gating_in)?;
        let gating_out = self.gate.forward(&gating_in)?;
        validate_finite_tensor("wave_rnn_gating_output", &gating_out)?;
        let mut final_hidden = Tensor::zeros(batch, self.hidden_dim)?;
        {
            let source = gating_out.data();
            let dest = final_hidden.data_mut();
            for b in 0..batch {
                let src_offset = (b * out_steps + (out_steps - 1)) * self.hidden_dim;
                let dst_offset = b * self.hidden_dim;
                dest[dst_offset..dst_offset + self.hidden_dim]
                    .copy_from_slice(&source[src_offset..src_offset + self.hidden_dim]);
            }
        }
        validate_finite_tensor("wave_rnn_final_hidden", &final_hidden)?;
        let pack = self.readout.ensure_matmul_pack()?;
        let output = relabel_non_finite(
            final_hidden.matmul_prepacked_bias_with_backend(
                &pack,
                self.readout_bias.value().data(),
                current_prepacked_matmul_backend(),
            ),
            "wave_rnn_output",
        )?;
        validate_finite_tensor("wave_rnn_output", &output)?;
        *self.cache.borrow_mut() = Some(WaveRnnCache {
            input: input.clone(),
            gating_in,
            final_hidden,
            batch,
            out_steps,
        });
        Ok(output)
    }

    fn backward(&mut self, _input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        validate_finite_tensor("wave_rnn_backward_grad_output", grad_output)?;
        self.validate_readout_parameters()?;
        let cache = self
            .cache
            .borrow()
            .as_ref()
            .cloned()
            .ok_or(TensorError::EmptyInput("wave_rnn_cache"))?;
        cache.validate()?;
        if grad_output.shape() != (cache.batch, self.hidden_dim) {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: (cache.batch, self.hidden_dim),
            });
        }
        if cache.batch == 0 {
            return Tensor::zeros(cache.batch, cache.input.shape().1);
        }
        let inv_batch = 1.0 / cache.batch as f32;
        let grad_readout = relabel_non_finite(
            cache.final_hidden.matmul_lhs_transpose_scaled_with_backend(
                grad_output,
                inv_batch,
                current_matmul_backend(),
            ),
            "wave_rnn_readout_grad",
        )?;
        validate_finite_tensor("wave_rnn_readout_grad", &grad_readout)?;
        let tensor_util_backend = current_tensor_util_backend_for_values(grad_output.data().len());
        let summed = relabel_non_finite(
            grad_output.try_sum_axis0_scaled_with_backend(inv_batch, tensor_util_backend),
            "wave_rnn_readout_bias_grad",
        )?;
        validate_finite_slice("wave_rnn_readout_bias_grad", &summed)?;
        let grad_bias = Tensor::from_vec(1, summed.len(), summed)?;
        validate_finite_tensor("wave_rnn_readout_bias_grad", &grad_bias)?;
        let pack_t = self.readout.ensure_matmul_transpose_pack()?;
        let grad_final = relabel_non_finite(
            grad_output.matmul_prepacked_with_backend(&pack_t, current_prepacked_matmul_backend()),
            "wave_rnn_grad_final_hidden",
        )?;
        validate_finite_tensor("wave_rnn_grad_final_hidden", &grad_final)?;
        let mut grad_gate_out = Tensor::zeros(cache.batch * cache.out_steps, self.hidden_dim)?;
        {
            let src = grad_final.data();
            let dst = grad_gate_out.data_mut();
            for b in 0..cache.batch {
                let src_offset = b * self.hidden_dim;
                let dst_offset = (b * cache.out_steps + (cache.out_steps - 1)) * self.hidden_dim;
                dst[dst_offset..dst_offset + self.hidden_dim]
                    .copy_from_slice(&src[src_offset..src_offset + self.hidden_dim]);
            }
        }
        validate_finite_tensor("wave_rnn_grad_gate_output", &grad_gate_out)?;
        let grad_gate_in = self.gate.backward_with_parameter_gradient_scale(
            &cache.gating_in,
            &grad_gate_out,
            inv_batch,
        )?;
        validate_finite_tensor("wave_rnn_grad_gate_input", &grad_gate_in)?;
        let grad_conv_out = grad_gate_in.reshape(cache.batch, self.hidden_dim * cache.out_steps)?;
        validate_finite_tensor("wave_rnn_grad_conv_output", &grad_conv_out)?;
        let grad_input = self.conv.backward(&cache.input, &grad_conv_out)?;
        validate_finite_tensor("wave_rnn_grad_input", &grad_input)?;
        self.readout.accumulate_euclidean(&grad_readout)?;
        self.readout_bias.accumulate_euclidean(&grad_bias)?;
        self.cache.borrow_mut().take();
        Ok(grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.conv.visit_parameters(visitor)?;
        self.gate.visit_parameters(visitor)?;
        visitor(&self.readout)?;
        visitor(&self.readout_bias)?;
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.conv.visit_parameters_mut(visitor)?;
        self.gate.visit_parameters_mut(visitor)?;
        visitor(&mut self.readout)?;
        visitor(&mut self.readout_bias)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    #[cfg(feature = "wgpu")]
    use crate::{
        execution::{push_backend_policy, BackendPolicy},
        TensorError,
    };
    #[cfg(feature = "wgpu")]
    use st_core::backend::device_caps::{BackendKind, DeviceCaps};
    #[cfg(feature = "wgpu")]
    use st_tensor::{AttentionBackend, LayerNormBackend, MatmulBackend, SoftmaxBackend};

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

    #[cfg(feature = "wgpu")]
    fn cpu_policy() -> BackendPolicy {
        BackendPolicy::explicit(
            DeviceCaps::cpu(),
            MatmulBackend::CpuNaive,
            MatmulBackend::CpuNaive,
            LayerNormBackend::Cpu,
            AttentionBackend::Cpu,
            SoftmaxBackend::Cpu,
        )
    }

    #[cfg(feature = "wgpu")]
    fn wgpu_policy() -> BackendPolicy {
        BackendPolicy::explicit(
            BackendKind::Wgpu.default_caps(),
            MatmulBackend::GpuWgpu,
            MatmulBackend::GpuWgpu,
            LayerNormBackend::Cpu,
            AttentionBackend::Cpu,
            SoftmaxBackend::Cpu,
        )
    }

    #[test]
    fn wave_rnn_runs_forward_backward() {
        let mut rnn = WaveRnn::new("wrnn", 2, 4, 3, 1, 1, -1.0, 0.5).unwrap();
        let input = Tensor::from_vec(1, 6, vec![0.1, 0.2, 0.3, -0.1, 0.0, 0.05]).unwrap();
        let output = rnn.forward(&input).unwrap();
        assert_eq!(output.shape(), (1, 4));
        let grad_out = Tensor::from_vec(1, 4, vec![0.01, -0.02, 0.03, -0.01]).unwrap();
        let grad_in = rnn.backward(&input, &grad_out).unwrap();
        assert_eq!(grad_in.shape(), input.shape());
    }

    #[test]
    fn wave_rnn_forward_rejects_non_finite_input_and_clears_stale_cache() {
        let rnn = WaveRnn::new("wrnn", 2, 4, 3, 1, 1, -1.0, 0.5).unwrap();
        let input = Tensor::from_vec(1, 6, vec![0.1, 0.2, 0.3, -0.1, 0.0, 0.05]).unwrap();
        let _ = rnn.forward(&input).unwrap();
        assert!(rnn.cache.borrow().is_some());
        let mut bad_input = input.clone();
        bad_input.data_mut()[0] = f32::NAN;

        assert_non_finite_label(rnn.forward(&bad_input), "wave_rnn_input");
        assert!(rnn.cache.borrow().is_none());
    }

    #[test]
    fn wave_rnn_backward_rejects_non_finite_grad_without_consuming_cache_or_readout_updates() {
        let mut rnn = WaveRnn::new("wrnn", 2, 4, 3, 1, 1, -1.0, 0.5).unwrap();
        let input = Tensor::from_vec(1, 6, vec![0.1, 0.2, 0.3, -0.1, 0.0, 0.05]).unwrap();
        let _ = rnn.forward(&input).unwrap();
        let mut grad_out = Tensor::from_vec(1, 4, vec![0.01, -0.02, 0.03, -0.01]).unwrap();
        grad_out.data_mut()[2] = f32::INFINITY;

        assert_non_finite_label(
            rnn.backward(&input, &grad_out),
            "wave_rnn_backward_grad_output",
        );
        assert!(rnn.cache.borrow().is_some());
        assert!(rnn.readout.gradient().is_none());
        assert!(rnn.readout_bias.gradient().is_none());
    }

    #[test]
    fn wave_rnn_backward_rejects_overflowing_readout_transpose_without_updates() {
        let mut rnn = WaveRnn::new("wrnn", 1, 1, 1, 1, 0, -1.0, 0.5).unwrap();
        let input = Tensor::from_vec(1, 2, vec![0.0, 0.0]).unwrap();
        let _ = rnn.forward(&input).unwrap();
        rnn.readout.value_mut().data_mut()[0] = f32::MAX;
        let grad_out = Tensor::from_vec(1, 1, vec![2.0]).unwrap();

        assert_non_finite_label(
            rnn.backward(&input, &grad_out),
            "wave_rnn_grad_final_hidden",
        );
        assert!(rnn.cache.borrow().is_some());
        assert!(rnn.readout.gradient().is_none());
        assert!(rnn.readout_bias.gradient().is_none());
    }

    #[test]
    fn wave_rnn_gate_gradients_are_batch_not_timestep_normalized() {
        let mut rnn = WaveRnn::new("wrnn", 1, 3, 1, 1, 0, -1.0, 0.5).unwrap();
        let input = Tensor::from_vec(1, 4, vec![0.2, -0.1, 0.35, 0.05]).unwrap();
        let _ = rnn.forward(&input).unwrap();
        let cache = rnn.cache.borrow().as_ref().expect("wave rnn cache").clone();
        assert_eq!(cache.batch, 1);
        assert_eq!(cache.out_steps, 4);

        let grad_out = Tensor::from_vec(1, 3, vec![0.08, -0.04, 0.06]).unwrap();
        let grad_final = grad_out.matmul(&rnn.readout.value().transpose()).unwrap();
        let mut grad_gate_out =
            Tensor::zeros(cache.batch * cache.out_steps, rnn.hidden_dim).unwrap();
        {
            let src = grad_final.data();
            let dst = grad_gate_out.data_mut();
            let dst_offset = (cache.out_steps - 1) * rnn.hidden_dim;
            dst[dst_offset..dst_offset + rnn.hidden_dim].copy_from_slice(src);
        }

        let mut default_gate = WaveGate::new("wrnn::gate", 3, -1.0, 0.5).unwrap();
        let _ = default_gate
            .backward(&cache.gating_in, &grad_gate_out)
            .unwrap();
        let mut batch_scaled_gate = WaveGate::new("wrnn::gate", 3, -1.0, 0.5).unwrap();
        let _ = batch_scaled_gate
            .backward_with_parameter_gradient_scale(
                &cache.gating_in,
                &grad_gate_out,
                1.0 / (cache.batch as f32),
            )
            .unwrap();

        let _ = rnn.backward(&input, &grad_out).unwrap();

        let default_gate_grad = default_gate.gate().gradient().expect("default gate grad");
        let batch_gate_grad = batch_scaled_gate
            .gate()
            .gradient()
            .expect("batch gate grad");
        let rnn_gate_grad = rnn.gate.gate().gradient().expect("rnn gate grad");
        let default_bias_grad = default_gate.bias().gradient().expect("default bias grad");
        let batch_bias_grad = batch_scaled_gate
            .bias()
            .gradient()
            .expect("batch bias grad");
        let rnn_bias_grad = rnn.gate.bias().gradient().expect("rnn bias grad");
        let step_scale = cache.out_steps as f32;

        for (idx, ((&default_value, &batch_value), &rnn_value)) in default_gate_grad
            .data()
            .iter()
            .zip(batch_gate_grad.data())
            .zip(rnn_gate_grad.data())
            .enumerate()
        {
            let expected = default_value * step_scale;
            assert!(
                (batch_value - expected).abs() <= 1.0e-6,
                "batch-scaled gate gradient mismatch at {idx}: batch={batch_value} expected={expected}"
            );
            assert!(
                (rnn_value - batch_value).abs() <= 1.0e-6,
                "wave rnn gate gradient mismatch at {idx}: rnn={rnn_value} batch={batch_value}"
            );
        }
        for (idx, ((&default_value, &batch_value), &rnn_value)) in default_bias_grad
            .data()
            .iter()
            .zip(batch_bias_grad.data())
            .zip(rnn_bias_grad.data())
            .enumerate()
        {
            let expected = default_value * step_scale;
            assert!(
                (batch_value - expected).abs() <= 1.0e-6,
                "batch-scaled bias gradient mismatch at {idx}: batch={batch_value} expected={expected}"
            );
            assert!(
                (rnn_value - batch_value).abs() <= 1.0e-6,
                "wave rnn bias gradient mismatch at {idx}: rnn={rnn_value} batch={batch_value}"
            );
        }
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn wave_rnn_forced_wgpu_prepacked_matches_cpu_reference_on_edge_tiles() {
        let cpu_rnn = WaveRnn::new("wrnn", 2, 6, 3, 1, 1, -1.0, 0.5).unwrap();
        let wgpu_rnn = WaveRnn::new("wrnn", 2, 6, 3, 1, 1, -1.0, 0.5).unwrap();
        let input = Tensor::from_vec(
            12,
            16,
            (0..192)
                .map(|idx| ((idx as f32 * 0.071).sin() * 0.4) - ((idx % 9) as f32 * 0.005))
                .collect(),
        )
        .unwrap();
        let cpu = {
            let _guard = push_backend_policy(cpu_policy());
            cpu_rnn.forward(&input).unwrap()
        };
        let wgpu = {
            let _guard = push_backend_policy(wgpu_policy());
            match wgpu_rnn.forward(&input) {
                Ok(value) => value,
                Err(TensorError::BackendFailure { backend, .. }) if backend == "wgpu" => return,
                Err(error) => panic!("forced WGPU WaveRnn forward failed: {error:?}"),
            }
        };

        assert_tensor_close(&cpu, &wgpu, 1e-4);
    }

    #[test]
    fn wave_rnn_empty_batch_backward_returns_empty_grad_without_updates() {
        let mut rnn = WaveRnn::new("wrnn", 2, 4, 3, 1, 1, -1.0, 0.5).unwrap();
        let input = Tensor::from_vec(0, 6, Vec::new()).unwrap();
        let output = rnn.forward(&input).unwrap();
        assert_eq!(output.shape(), (0, 4));
        assert!(output.data().is_empty());

        let grad_out = Tensor::from_vec(0, 4, Vec::new()).unwrap();
        let grad_in = rnn.backward(&input, &grad_out).unwrap();
        assert_eq!(grad_in.shape(), input.shape());
        assert!(grad_in.data().is_empty());
        assert!(rnn.readout.gradient().is_none());
        assert!(rnn.readout_bias.gradient().is_none());
    }
}
