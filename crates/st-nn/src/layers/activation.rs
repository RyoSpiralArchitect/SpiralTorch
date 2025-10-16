// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::Module;
use crate::{PureResult, Tensor};

/// Lightweight ReLU activation that keeps gradients aligned with the
/// SpiralTorch module trait. The layer is stateless and therefore does not
/// participate in parameter visits.
#[derive(Debug, Default, Clone, Copy)]
pub struct Relu;

impl Relu {
    /// Creates a new ReLU layer.
    pub fn new() -> Self {
        Self
    }
}

impl Module for Relu {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        let mut data = Vec::with_capacity(rows * cols);
        for value in input.data() {
            data.push(value.max(0.0));
        }
        Tensor::from_vec(rows, cols, data)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        if input.shape() != grad_output.shape() {
            return Err(st_tensor::TensorError::ShapeMismatch {
                left: input.shape(),
                right: grad_output.shape(),
            });
        }
        let (rows, cols) = input.shape();
        let mut data = Vec::with_capacity(rows * cols);
        for (input_value, grad_value) in input.data().iter().zip(grad_output.data().iter()) {
            if *input_value > 0.0 {
                data.push(*grad_value);
            } else {
                data.push(0.0);
            }
        }
        Tensor::from_vec(rows, cols, data)
    }

    fn visit_parameters(
        &self,
        _visitor: &mut dyn FnMut(&crate::module::Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        _visitor: &mut dyn FnMut(&mut crate::module::Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relu_forward_backward() {
        let relu = Relu::new();
        let input = Tensor::from_vec(1, 4, vec![-1.0, -0.5, 0.2, 1.5]).unwrap();
        let output = relu.forward(&input).unwrap();
        assert_eq!(output.data(), &[0.0, 0.0, 0.2, 1.5]);

        let mut relu = relu;
        let grad_output = Tensor::from_vec(1, 4, vec![0.3, 0.4, 0.5, 0.6]).unwrap();
        let grad_input = relu.backward(&input, &grad_output).unwrap();
        assert_eq!(grad_input.data(), &[0.0, 0.0, 0.5, 0.6]);
    }
}
