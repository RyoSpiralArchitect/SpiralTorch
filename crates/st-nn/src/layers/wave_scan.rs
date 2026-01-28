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
struct WaveScanCache {
    input: Tensor,
    gating_in: Tensor,
    batch: usize,
    out_steps: usize,
    features: usize,
}

/// Convolutional scan module that reads a flattened (channels×steps) sequence and emits
/// a single context vector from the final step after Z-space WaveGate projection.
#[derive(Debug)]
pub struct WaveScan {
    conv: Conv1d,
    gate: WaveGate,
    in_channels: usize,
    features: usize,
    cache: RefCell<Option<WaveScanCache>>,
}

impl WaveScan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        in_channels: usize,
        features: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        curvature: f32,
        temperature: f32,
    ) -> PureResult<Self> {
        if in_channels == 0 || features == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: in_channels.max(1),
                cols: features.max(1),
            });
        }
        let name = name.into();
        let conv = Conv1d::new(
            format!("{name}::conv"),
            in_channels,
            features,
            kernel_size,
            stride,
            padding,
            dilation,
        )?;
        let gate = WaveGate::new(format!("{name}::gate"), features, curvature, temperature)?;
        Ok(Self {
            conv,
            gate,
            in_channels,
            features,
            cache: RefCell::new(None),
        })
    }

    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    pub fn features(&self) -> usize {
        self.features
    }

    pub fn gate(&self) -> &WaveGate {
        &self.gate
    }

    pub fn gate_mut(&mut self) -> &mut WaveGate {
        &mut self.gate
    }

    fn infer_steps(&self, cols: usize) -> PureResult<usize> {
        if !cols.is_multiple_of(self.in_channels) {
            return Err(TensorError::ShapeMismatch {
                left: (1, cols),
                right: (1, self.in_channels),
            });
        }
        Ok(cols / self.in_channels)
    }
}

impl Module for WaveScan {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (batch, cols) = input.shape();
        let steps = self.infer_steps(cols)?;
        if steps == 0 {
            return Err(TensorError::InvalidDimensions { rows: batch, cols });
        }

        let conv_out = self.conv.forward(input)?;
        let (rows, conv_cols) = conv_out.shape();
        if rows != batch || conv_cols % self.features != 0 {
            return Err(TensorError::ShapeMismatch {
                left: conv_out.shape(),
                right: (batch, self.features),
            });
        }
        let out_steps = conv_cols / self.features;
        if out_steps == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: batch,
                cols: conv_cols,
            });
        }

        let gating_in = conv_out.reshape(batch * out_steps, self.features)?;
        let gating_out = self.gate.forward(&gating_in)?;
        let gating_reshaped = gating_out.reshape(batch, self.features * out_steps)?;

        let mut final_hidden = Tensor::zeros(batch, self.features)?;
        {
            let source = gating_reshaped.data();
            let dest = final_hidden.data_mut();
            for b in 0..batch {
                let src_offset =
                    b * self.features * out_steps + (out_steps - 1) * self.features;
                let dst_offset = b * self.features;
                dest[dst_offset..dst_offset + self.features]
                    .copy_from_slice(&source[src_offset..src_offset + self.features]);
            }
        }

        *self.cache.borrow_mut() = Some(WaveScanCache {
            input: input.clone(),
            gating_in,
            batch,
            out_steps,
            features: self.features,
        });
        Ok(final_hidden)
    }

    fn backward(&mut self, _input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let cache = self
            .cache
            .borrow_mut()
            .take()
            .ok_or(TensorError::EmptyInput("wave_scan_cache"))?;
        if grad_output.shape() != (cache.batch, cache.features) {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: (cache.batch, cache.features),
            });
        }

        let mut grad_gate_out = Tensor::zeros(cache.batch, cache.features * cache.out_steps)?;
        {
            let src = grad_output.data();
            let dst = grad_gate_out.data_mut();
            for b in 0..cache.batch {
                let src_offset = b * cache.features;
                let dst_offset =
                    b * cache.features * cache.out_steps + (cache.out_steps - 1) * cache.features;
                dst[dst_offset..dst_offset + cache.features]
                    .copy_from_slice(&src[src_offset..src_offset + cache.features]);
            }
        }

        let grad_gate_out =
            grad_gate_out.reshape(cache.batch * cache.out_steps, cache.features)?;
        let grad_gate_in = self.gate.backward(&cache.gating_in, &grad_gate_out)?;
        let grad_conv_out = grad_gate_in.reshape(cache.batch, cache.features * cache.out_steps)?;
        self.conv.backward(&cache.input, &grad_conv_out)
    }

    fn visit_parameters(&self, visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>) -> PureResult<()> {
        self.conv.visit_parameters(visitor)?;
        self.gate.visit_parameters(visitor)?;
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.conv.visit_parameters_mut(visitor)?;
        self.gate.visit_parameters_mut(visitor)?;
        Ok(())
    }
}

/// Multi-dilation WaveScan stack that averages the per-dilation summaries.
#[derive(Debug)]
pub struct WaveScanStack {
    scans: Vec<WaveScan>,
    features: usize,
}

impl WaveScanStack {
    pub fn new(scans: Vec<WaveScan>) -> PureResult<Self> {
        if scans.is_empty() {
            return Err(TensorError::EmptyInput("wave_scan_stack"));
        }
        let features = scans[0].features();
        if scans.iter().any(|scan| scan.features() != features) {
            return Err(TensorError::InvalidDimensions {
                rows: features,
                cols: 0,
            });
        }
        Ok(Self { scans, features })
    }

    pub fn features(&self) -> usize {
        self.features
    }

    pub fn scans(&self) -> &[WaveScan] {
        &self.scans
    }

    pub fn scans_mut(&mut self) -> &mut [WaveScan] {
        &mut self.scans
    }
}

impl Module for WaveScanStack {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (batch, _) = input.shape();
        let mut sum = Tensor::zeros(batch, self.features)?;
        for scan in &self.scans {
            let out = scan.forward(input)?;
            sum.add_scaled(&out, 1.0)?;
        }
        let inv = 1.0 / (self.scans.len() as f32).max(1.0);
        sum.scale(inv)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        if self.scans.is_empty() {
            return Ok(grad_output.clone());
        }
        let inv = 1.0 / (self.scans.len() as f32).max(1.0);
        let grad_scaled = grad_output.scale(inv)?;
        let (batch, cols) = input.shape();
        let mut total = Tensor::zeros(batch, cols)?;
        for scan in &mut self.scans {
            let grad_in = scan.backward(input, &grad_scaled)?;
            total.add_scaled(&grad_in, 1.0)?;
        }
        Ok(total)
    }

    fn visit_parameters(&self, visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>) -> PureResult<()> {
        for scan in &self.scans {
            scan.visit_parameters(visitor)?;
        }
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        for scan in &mut self.scans {
            scan.visit_parameters_mut(visitor)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wave_scan_forward_backward_matches_shapes() {
        let mut scan = WaveScan::new("ws", 4, 4, 3, 1, 1, 2, -1.0, 0.7).unwrap();
        let input = Tensor::from_vec(2, 16, vec![0.1; 32]).unwrap(); // in_channels=4, steps=4
        let out = scan.forward(&input).unwrap();
        assert_eq!(out.shape(), (2, 4));
        let grad_out = Tensor::from_vec(2, 4, vec![0.05; 8]).unwrap();
        let grad_in = scan.backward(&input, &grad_out).unwrap();
        assert_eq!(grad_in.shape(), input.shape());
    }

    #[test]
    fn wave_scan_stack_averages_branches() {
        let scan_a = WaveScan::new("a", 2, 2, 3, 1, 1, 1, -1.0, 1.0).unwrap();
        let scan_b = WaveScan::new("b", 2, 2, 3, 1, 1, 2, -1.0, 1.0).unwrap();
        let mut stack = WaveScanStack::new(vec![scan_a, scan_b]).unwrap();
        let input = Tensor::from_vec(1, 8, vec![0.2; 8]).unwrap(); // in_channels=2, steps=4
        let out = stack.forward(&input).unwrap();
        assert_eq!(out.shape(), (1, 2));
        let grad_out = Tensor::from_vec(1, 2, vec![0.1, -0.2]).unwrap();
        let grad_in = stack.backward(&input, &grad_out).unwrap();
        assert_eq!(grad_in.shape(), input.shape());
    }
}

