// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::conv::Conv1d;
use super::wave_gate::WaveGate;
use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};
use std::cell::RefCell;

#[derive(Clone, Debug)]
struct WaveRnnCache {
    input: Tensor,
    gating_in: Tensor,
    final_hidden: Tensor,
    batch: usize,
    out_steps: usize,
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
}

impl Module for WaveRnn {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let conv_out = self.conv.forward(input)?;
        let (batch, cols) = conv_out.shape();
        if cols % self.hidden_dim != 0 {
            return Err(TensorError::ShapeMismatch {
                left: conv_out.shape(),
                right: (batch, self.hidden_dim),
            });
        }
        let out_steps = cols / self.hidden_dim;
        let gating_in = conv_out.reshape(batch * out_steps, self.hidden_dim)?;
        let gating_out = self.gate.forward(&gating_in)?;
        let gating_reshaped = gating_out.reshape(batch, self.hidden_dim * out_steps)?;
        let mut final_hidden = Tensor::zeros(batch, self.hidden_dim)?;
        {
            let source = gating_reshaped.data();
            let dest = final_hidden.data_mut();
            for b in 0..batch {
                let src_offset =
                    b * self.hidden_dim * out_steps + (out_steps - 1) * self.hidden_dim;
                let dst_offset = b * self.hidden_dim;
                dest[dst_offset..dst_offset + self.hidden_dim]
                    .copy_from_slice(&source[src_offset..src_offset + self.hidden_dim]);
            }
        }
        let mut output = final_hidden.matmul(self.readout.value())?;
        output.add_row_inplace(self.readout_bias.value().data())?;
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
        let cache = self
            .cache
            .borrow_mut()
            .take()
            .ok_or(TensorError::EmptyInput("wave_rnn_cache"))?;
        if grad_output.shape() != (cache.batch, self.hidden_dim) {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: (cache.batch, self.hidden_dim),
            });
        }
        let inv_batch = 1.0 / cache.batch as f32;
        let grad_readout = cache
            .final_hidden
            .transpose()
            .matmul(grad_output)?
            .scale(inv_batch)?;
        let grad_bias = {
            let summed = grad_output.sum_axis0();
            Tensor::from_vec(1, summed.len(), summed)?.scale(inv_batch)?
        };
        self.readout.accumulate_euclidean(&grad_readout)?;
        self.readout_bias.accumulate_euclidean(&grad_bias)?;
        let readout_t = self.readout.value().transpose();
        let grad_final = grad_output.matmul(&readout_t)?;
        let mut grad_gate_out = Tensor::zeros(cache.batch, self.hidden_dim * cache.out_steps)?;
        {
            let src = grad_final.data();
            let dst = grad_gate_out.data_mut();
            for b in 0..cache.batch {
                let src_offset = b * self.hidden_dim;
                let dst_offset =
                    b * self.hidden_dim * cache.out_steps + (cache.out_steps - 1) * self.hidden_dim;
                for h in 0..self.hidden_dim {
                    dst[dst_offset + h] = src[src_offset + h];
                }
            }
        }
        let grad_gate_out =
            grad_gate_out.reshape(cache.batch * cache.out_steps, self.hidden_dim)?;
        let grad_gate_in = self.gate.backward(&cache.gating_in, &grad_gate_out)?;
        let grad_conv_out = grad_gate_in.reshape(cache.batch, self.hidden_dim * cache.out_steps)?;
        let grad_input = self.conv.backward(&cache.input, &grad_conv_out)?;
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
}
