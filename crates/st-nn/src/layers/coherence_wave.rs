// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::coherence_scan::ZSpaceCoherenceScan;
use super::wave_gate::WaveGate;
use super::wave_scan::{WaveScan, WaveScanStack};
use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};
use std::cell::RefCell;

#[derive(Clone, Debug)]
struct CoherenceWaveCache {
    fused_pre_gate: Tensor,
}

fn fixed_len_text(text: &str, chars: usize) -> Result<String, TensorError> {
    if chars == 0 {
        return Err(TensorError::EmptyInput("infuse_text_chars"));
    }
    let source: Vec<char> = text.chars().collect();
    if source.is_empty() {
        return Err(TensorError::EmptyInput("infuse_text"));
    }
    let mut out = String::with_capacity(chars);
    for idx in 0..chars {
        out.push(source[idx % source.len()]);
    }
    Ok(out)
}

/// Hybrid block combining:
/// - A: `ZSpaceCoherenceScan` (temporal coherence weights)
/// - B: multi-dilation `WaveScanStack` (dilated conv + WaveGate)
/// - C: a final `WaveGate` resonator (optionally text-infusable)
///
/// Input: flattened embeddings shaped `(batch, dim * steps)`.
/// Output: context vector shaped `(batch, dim)`.
#[derive(Debug)]
pub struct ZSpaceCoherenceWaveBlock {
    dim: usize,
    steps: usize,
    memory: usize,
    scan: ZSpaceCoherenceScan,
    wave: WaveScanStack,
    resonator: WaveGate,
    cache: RefCell<Option<CoherenceWaveCache>>,
}

impl ZSpaceCoherenceWaveBlock {
    /// Builds a new block. `dilations` must not be empty.
    pub fn new(
        dim: usize,
        steps: usize,
        memory: usize,
        curvature: f32,
        temperature: f32,
        kernel_size: usize,
        dilations: Vec<usize>,
    ) -> PureResult<Self> {
        if dim == 0 || steps == 0 || memory == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: steps.max(1),
                cols: dim.max(1),
            });
        }
        if memory > steps {
            return Err(TensorError::InvalidDimensions {
                rows: memory,
                cols: steps,
            });
        }
        if kernel_size == 0 {
            return Err(TensorError::InvalidDimensions { rows: 1, cols: 0 });
        }
        if dilations.is_empty() {
            return Err(TensorError::EmptyInput("wave_dilations"));
        }

        let scan = ZSpaceCoherenceScan::new(dim, steps, memory, curvature, temperature)?;
        let mut scans = Vec::with_capacity(dilations.len());
        for (idx, dilation) in dilations.into_iter().enumerate() {
            let padding = dilation.saturating_mul(kernel_size.saturating_sub(1)) / 2;
            scans.push(WaveScan::new(
                format!("wave_scan_{idx}"),
                dim,
                dim,
                kernel_size,
                1,
                padding,
                dilation,
                curvature,
                temperature,
            )?);
        }
        let wave = WaveScanStack::new(scans)?;
        let resonator = WaveGate::new("coherence_wave_resonator", dim, curvature, temperature)?;

        Ok(Self {
            dim,
            steps,
            memory,
            scan,
            wave,
            resonator,
            cache: RefCell::new(None),
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    pub fn memory(&self) -> usize {
        self.memory
    }

    pub fn resonator(&self) -> &WaveGate {
        &self.resonator
    }

    pub fn resonator_mut(&mut self) -> &mut WaveGate {
        &mut self.resonator
    }

    /// Streams text into the resonator and wave gates via `Parameter::absorb_text`.
    ///
    /// Note: `LanguageWaveEncoder::encode_z_space` yields `2 * char_count` coordinates.
    /// For infusion to work, the block feature dimension must be even; the method will
    /// derive a fixed-length string of `dim / 2` characters.
    pub fn infuse_text(&mut self, text: &str) -> PureResult<()> {
        if self.dim % 2 != 0 {
            return Err(TensorError::InvalidDimensions {
                rows: self.dim,
                cols: 2,
            });
        }
        let required_chars = self.dim / 2;
        let snippet = fixed_len_text(text, required_chars)?;
        let encoder = self.resonator.encoder().clone();

        self.resonator
            .visit_parameters_mut(&mut |param| param.absorb_text(&encoder, &snippet))?;
        for scan in self.wave.scans_mut() {
            scan.gate_mut()
                .visit_parameters_mut(&mut |param| param.absorb_text(&encoder, &snippet))?;
        }
        Ok(())
    }
}

impl Module for ZSpaceCoherenceWaveBlock {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (batch, cols) = input.shape();
        let expected = self.dim * self.steps;
        if cols != expected {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: (batch, expected),
            });
        }

        let scan_ctx = self.scan.forward(input)?;
        let wave_ctx = self.wave.forward(input)?;
        let fused = scan_ctx.add(&wave_ctx)?;
        let out = self.resonator.forward(&fused)?;
        *self.cache.borrow_mut() = Some(CoherenceWaveCache {
            fused_pre_gate: fused,
        });
        Ok(out)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let cache = self
            .cache
            .borrow_mut()
            .take()
            .ok_or(TensorError::EmptyInput("coherence_wave_cache"))?;
        let grad_fused = self
            .resonator
            .backward(&cache.fused_pre_gate, grad_output)?;
        let grad_scan = self.scan.backward(input, &grad_fused)?;
        let grad_wave = self.wave.backward(input, &grad_fused)?;
        grad_scan.add(&grad_wave)
    }

    fn visit_parameters(&self, visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>) -> PureResult<()> {
        self.wave.visit_parameters(visitor)?;
        self.resonator.visit_parameters(visitor)?;
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.wave.visit_parameters_mut(visitor)?;
        self.resonator.visit_parameters_mut(visitor)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coherence_wave_block_shapes_match() {
        let mut block = ZSpaceCoherenceWaveBlock::new(8, 4, 3, -1.0, 1.0, 3, vec![1, 2]).unwrap();
        let input = Tensor::from_vec(2, 32, vec![0.1; 64]).unwrap();
        let out = block.forward(&input).unwrap();
        assert_eq!(out.shape(), (2, 8));
        let grad_out = Tensor::from_vec(2, 8, vec![0.01; 16]).unwrap();
        let grad_in = block.backward(&input, &grad_out).unwrap();
        assert_eq!(grad_in.shape(), input.shape());
    }

    #[test]
    fn coherence_wave_infuses_fixed_length_text() {
        let mut block = ZSpaceCoherenceWaveBlock::new(8, 4, 3, -0.9, 0.8, 3, vec![1]).unwrap();
        block.attach_hypergrad(-0.9, 0.05).unwrap();
        block.infuse_text("hello").unwrap();
        block.apply_step(0.01).unwrap();
    }
}
