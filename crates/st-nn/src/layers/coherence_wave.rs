// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::coherence_scan::ZSpaceCoherenceScan;
use super::wave_gate::WaveGate;
use super::wave_scan::{WaveScan, WaveScanStack};
use crate::execution::current_tensor_util_backend_for_values;
use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};
use st_tensor::{emit_tensor_op, emit_tensor_op_meta, TensorUtilBackend};
use std::cell::RefCell;

#[derive(Clone, Debug)]
struct CoherenceWaveCache {
    fused_pre_gate: Tensor,
}

fn tensor_util_backend_label(backend: TensorUtilBackend) -> &'static str {
    match backend {
        TensorUtilBackend::Auto => "auto",
        TensorUtilBackend::Cpu => "cpu",
        TensorUtilBackend::GpuWgpu => "wgpu",
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_coherence_wave_meta(
    op_name: &'static str,
    kind: &'static str,
    batch: usize,
    dim: usize,
    steps: usize,
    memory: usize,
    wave_branches: usize,
    merge_backend: TensorUtilBackend,
    backward: bool,
) {
    let input_values = batch.saturating_mul(dim).saturating_mul(steps);
    let context_values = batch.saturating_mul(dim);
    emit_tensor_op(op_name, &[batch, steps, dim], &[batch, dim]);
    emit_tensor_op_meta(op_name, || {
        serde_json::json!({
            "backend": "composite",
            "requested_backend": tensor_util_backend_label(merge_backend),
            "merge_backend": tensor_util_backend_label(merge_backend),
            "kernel": "coherence_wave.composite",
            "kind": kind,
            "batch": batch,
            "dim": dim,
            "steps": steps,
            "memory": memory,
            "wave_branches": wave_branches,
            "input_values": input_values,
            "context_values": context_values,
            "scan_context_values": context_values,
            "wave_context_values": context_values,
            "fused_context_values": context_values,
            "estimated_merge_values": context_values,
            "estimated_backward_merge_values": if backward { input_values } else { 0 },
            "backward": backward,
            "empty": input_values == 0 || context_values == 0,
        })
    });
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
        Self::with_output_scales(
            dim,
            steps,
            memory,
            curvature,
            temperature,
            kernel_size,
            dilations,
            1.0,
            0.0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_self_score_scale(
        dim: usize,
        steps: usize,
        memory: usize,
        curvature: f32,
        temperature: f32,
        kernel_size: usize,
        dilations: Vec<usize>,
        self_score_scale: f32,
    ) -> PureResult<Self> {
        Self::with_output_scales(
            dim,
            steps,
            memory,
            curvature,
            temperature,
            kernel_size,
            dilations,
            self_score_scale,
            0.0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_output_scales(
        dim: usize,
        steps: usize,
        memory: usize,
        curvature: f32,
        temperature: f32,
        kernel_size: usize,
        dilations: Vec<usize>,
        self_score_scale: f32,
        query_residual_scale: f32,
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

        let scan = ZSpaceCoherenceScan::with_output_scales(
            dim,
            steps,
            memory,
            curvature,
            temperature,
            self_score_scale,
            query_residual_scale,
        )?;
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
        if !self.dim.is_multiple_of(2) {
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
        let merge_backend = current_tensor_util_backend_for_values(scan_ctx.data().len());
        let fused = scan_ctx.add_with_backend(&wave_ctx, merge_backend)?;
        let out = self.resonator.forward(&fused)?;
        *self.cache.borrow_mut() = Some(CoherenceWaveCache {
            fused_pre_gate: fused,
        });
        emit_coherence_wave_meta(
            "coherence_wave_forward",
            "coherence_wave_forward_composite",
            batch,
            self.dim,
            self.steps,
            self.memory,
            self.wave.scans().len(),
            merge_backend,
            false,
        );
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
        let merge_backend = current_tensor_util_backend_for_values(grad_scan.data().len());
        let grad_input = grad_scan.add_with_backend(&grad_wave, merge_backend)?;
        let (batch, _) = input.shape();
        emit_coherence_wave_meta(
            "coherence_wave_backward",
            "coherence_wave_backward_composite",
            batch,
            self.dim,
            self.steps,
            self.memory,
            self.wave.scans().len(),
            merge_backend,
            true,
        );
        Ok(grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
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

    fn infuse_text(&mut self, text: &str) -> PureResult<()> {
        ZSpaceCoherenceWaveBlock::infuse_text(self, text)
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
    fn coherence_wave_block_emits_composite_backend_meta() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let mut block = ZSpaceCoherenceWaveBlock::new(6, 5, 4, -1.0, 1.0, 3, vec![1, 2]).unwrap();
        let input = Tensor::from_vec(1, 30, vec![0.1; 30]).unwrap();
        let out = block.forward(&input).unwrap();
        let grad_out = Tensor::from_vec(1, 6, vec![0.01; 6]).unwrap();
        let grad_in = block.backward(&input, &grad_out).unwrap();
        st_tensor::set_tensor_op_meta_observer(previous);

        assert_eq!(out.shape(), (1, 6));
        assert_eq!(grad_in.shape(), input.shape());
        let events = events.lock().unwrap();
        for (op_name, kind, backward) in [
            (
                "coherence_wave_forward",
                "coherence_wave_forward_composite",
                false,
            ),
            (
                "coherence_wave_backward",
                "coherence_wave_backward_composite",
                true,
            ),
        ] {
            let event = events
                .iter()
                .find(|(name, data)| {
                    *name == op_name
                        && data["backend"] == "composite"
                        && data["kind"] == kind
                        && data["dim"] == 6
                        && data["steps"] == 5
                        && data["wave_branches"] == 2
                })
                .unwrap_or_else(|| panic!("{op_name} metadata event"));
            assert_eq!(event.1["requested_backend"], "auto");
            assert_eq!(event.1["merge_backend"], "auto");
            assert_eq!(event.1["context_values"], 6);
            assert_eq!(event.1["scan_context_values"], 6);
            assert_eq!(event.1["wave_context_values"], 6);
            assert_eq!(event.1["backward"], backward);
        }
        assert!(
            events
                .iter()
                .any(|(name, data)| *name == "zspace_coherence_scan_forward"
                    && data["dim"] == 6
                    && data["steps"] == 5),
            "coherence wave should still expose scan child metadata"
        );
        assert!(
            events
                .iter()
                .any(|(name, data)| *name == "wave_scan_stack_forward"
                    && data["features"] == 6
                    && data["branch_count"] == 2),
            "coherence wave should still expose wave stack child metadata"
        );
    }

    #[test]
    fn coherence_wave_infuses_fixed_length_text() {
        let mut block = ZSpaceCoherenceWaveBlock::new(8, 4, 3, -0.9, 0.8, 3, vec![1]).unwrap();
        block.attach_hypergrad(-0.9, 0.05).unwrap();
        block.infuse_text("hello").unwrap();
        block.apply_step(0.01).unwrap();
    }
}
