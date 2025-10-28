// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor};

/// Stateless identity layer that forwards its input unchanged.
#[derive(Debug, Default, Clone, Copy)]
pub struct Identity;

impl Identity {
    /// Creates a new identity layer.
    pub fn new() -> Self {
        Self
    }
}

impl Module for Identity {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        Ok(input.clone())
    }

    fn backward(&mut self, _input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        Ok(grad_output.clone())
    }

    fn visit_parameters(
        &self,
        _visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        _visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_forward_and_backward_are_noops() {
        let layer = Identity::new();
        let input = Tensor::from_vec(2, 3, vec![0.5, -1.0, 2.0, 3.5, 0.0, -0.25]).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output, input);

        let mut layer = layer;
        let grad_output = Tensor::from_vec(2, 3, vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0]).unwrap();
        let grad_input = layer.backward(&input, &grad_output).unwrap();
        assert_eq!(grad_input, grad_output);
    }
}
