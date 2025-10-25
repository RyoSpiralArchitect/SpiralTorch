// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};

/// Smooth non-linear activation implemented as a scaled hyperbolic tangent.
///
/// The layer is stateless and therefore behaves similarly to other activation
/// modules, yet it conforms to the [`Module`] trait to integrate with any
/// existing training pipelines expecting parameter visitors.
#[derive(Debug, Clone, Copy)]
pub struct NonLiner {
    slope: f32,
}

impl Default for NonLiner {
    fn default() -> Self {
        Self { slope: 1.0 }
    }
}

impl NonLiner {
    /// Creates a new non-linear layer using a configurable slope.
    pub fn new(slope: f32) -> Self {
        Self { slope }
    }

    /// Returns the slope applied to the input before the tanh activation.
    pub fn slope(&self) -> f32 {
        self.slope
    }
}

impl Module for NonLiner {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        let mut data = Vec::with_capacity(rows * cols);
        for value in input.data() {
            data.push((self.slope * *value).tanh());
        }
        Tensor::from_vec(rows, cols, data)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        if input.shape() != grad_output.shape() {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: grad_output.shape(),
            });
        }
        let (rows, cols) = input.shape();
        let mut data = Vec::with_capacity(rows * cols);
        for (input_value, grad_value) in input.data().iter().zip(grad_output.data().iter()) {
            let activation = (self.slope * *input_value).tanh();
            let derivative = self.slope * (1.0 - activation * activation);
            data.push(derivative * *grad_value);
        }
        Tensor::from_vec(rows, cols, data)
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
    fn forward_matches_tanh() {
        let layer = NonLiner::new(0.75);
        let input = Tensor::from_vec(2, 2, vec![-1.0, -0.5, 0.5, 1.0]).unwrap();
        let output = layer.forward(&input).unwrap();
        let expected: Vec<f32> = input
            .data()
            .iter()
            .map(|v| (0.75 * *v).tanh())
            .collect();
        assert_eq!(output.data(), expected.as_slice());
    }

    #[test]
    fn backward_respects_chain_rule() {
        let mut layer = NonLiner::new(1.25);
        let input = Tensor::from_vec(1, 3, vec![-0.4, 0.0, 0.6]).unwrap();
        let grad_output = Tensor::from_vec(1, 3, vec![0.3, -0.1, 0.7]).unwrap();
        let grad_input = layer.backward(&input, &grad_output).unwrap();
        let expected: Vec<f32> = input
            .data()
            .iter()
            .zip(grad_output.data().iter())
            .map(|(x, g)| {
                let activation = (1.25 * *x).tanh();
                let derivative = 1.25 * (1.0 - activation * activation);
                derivative * *g
            })
            .collect();
        assert_eq!(grad_input.data(), expected.as_slice());
    }
}
