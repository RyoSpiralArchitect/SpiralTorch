// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor};
use st_tensor::topos::OpenCartesianTopos;
use st_tensor::{LanguageWaveEncoder, TensorError};

/// Element-wise gate that keeps a persistent Z-space resonance attached to the
/// hypergrad tape. The resonator can ingest text or complex waves and amplify
/// the incoming signal before passing it downstream.
#[derive(Debug)]
pub struct ToposResonator {
    gate: Parameter,
    encoder: Option<LanguageWaveEncoder>,
}

impl ToposResonator {
    /// Creates a new resonator with an identity gate.
    pub fn new(name: impl Into<String>, rows: usize, cols: usize) -> PureResult<Self> {
        let weights = Tensor::from_vec(rows, cols, vec![1.0; rows * cols])?;
        Ok(Self {
            gate: Parameter::new(name, weights),
            encoder: None,
        })
    }

    /// Provides immutable access to the parameter for external inspection.
    pub fn parameter(&self) -> &Parameter {
        &self.gate
    }

    /// Provides mutable access to the parameter so callers can attach
    /// hypergrad tapes or rename the gate.
    pub fn parameter_mut(&mut self) -> &mut Parameter {
        &mut self.gate
    }

    /// Attaches a dedicated text encoder that can stream descriptions directly
    /// into the hypergrad tape.
    pub fn with_encoder(mut self, encoder: LanguageWaveEncoder) -> Self {
        self.encoder = Some(encoder);
        self
    }

    /// Streams raw text into the resonator when an encoder has been attached.
    pub fn infuse_text(&mut self, text: &str) -> PureResult<()> {
        let encoder = self
            .encoder
            .as_ref()
            .ok_or(TensorError::EmptyInput("topos resonator encoder"))?;
        self.gate.absorb_text(encoder, text)
    }

    /// Connects the gate to a caller supplied open topos.
    pub fn attach_open_topos(
        &mut self,
        curvature: f32,
        learning_rate: f32,
        topos: OpenCartesianTopos,
    ) -> PureResult<()> {
        self.gate
            .attach_hypergrad_with_topos(curvature, learning_rate, topos)
    }
}

impl Module for ToposResonator {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        input.hadamard(self.gate.value())
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let grad_gate = grad_output.hadamard(input)?;
        self.gate.accumulate_euclidean(&grad_gate)?;
        grad_output.hadamard(self.gate.value())
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.gate)
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.gate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resonator_behaves_like_identity_by_default() {
        let resonator = ToposResonator::new("gate", 2, 2).unwrap();
        let input = Tensor::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let out = resonator.forward(&input).unwrap();
        assert_eq!(out.data(), input.data());
    }

    #[test]
    fn backward_accumulates_gate_gradient() {
        let mut resonator = ToposResonator::new("gate", 2, 2).unwrap();
        let input = Tensor::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let grad_output = Tensor::from_vec(2, 2, vec![0.5, 0.25, 0.1, 0.0]).unwrap();
        let grad_input = resonator.backward(&input, &grad_output).unwrap();
        assert_eq!(grad_input.data(), grad_output.data());
        let mut captured: Option<Tensor> = None;
        resonator
            .visit_parameters(&mut |param: &Parameter| {
                captured = param.gradient().cloned();
                Ok(())
            })
            .unwrap();
        let gradient = captured.expect("gradient present");
        assert_eq!(
            gradient.data(),
            grad_output.hadamard(&input).unwrap().data()
        );
    }
}
