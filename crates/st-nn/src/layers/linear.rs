// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};

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
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        if input.shape().1 != self.weight.value().shape().0 {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: self.weight.value().shape(),
            });
        }
        let mut out = input.matmul(self.weight.value())?;
        out.add_row_inplace(self.bias.value().data())?;
        Ok(out)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        if input.shape().0 != grad_output.shape().0 {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: grad_output.shape(),
            });
        }
        let batch = input.shape().0 as f32;
        let grad_w = input.transpose().matmul(grad_output)?.scale(1.0 / batch)?;
        self.weight.accumulate_euclidean(&grad_w)?;

        let summed = grad_output.sum_axis0();
        let grad_b = Tensor::from_vec(1, summed.len(), summed)?.scale(1.0 / batch)?;
        self.bias.accumulate_euclidean(&grad_b)?;

        let weight_t = self.weight.value().transpose();
        let grad_input = grad_output.matmul(&weight_t)?;
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
}
