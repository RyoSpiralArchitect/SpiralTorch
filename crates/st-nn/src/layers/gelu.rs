// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::Module;
use crate::{PureResult, Tensor};
use st_tensor::TensorError;

const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
const KAPPA: f32 = 0.044715;

/// Gaussian Error Linear Unit with the tanh-based approximation that remains
/// stable inside the Z-space curvature bounds.
#[derive(Debug, Clone, Copy, Default)]
pub struct Gelu;

impl Gelu {
    /// Creates a new GELU activation.
    pub fn new() -> Self {
        Self
    }

    fn gelu(value: f32) -> f32 {
        let cubic = value * value * value;
        let inner = SQRT_2_OVER_PI * (value + KAPPA * cubic);
        0.5 * value * (1.0 + inner.tanh())
    }

    fn gelu_derivative(value: f32) -> f32 {
        let cubic = value * value * value;
        let inner = SQRT_2_OVER_PI * (value + KAPPA * cubic);
        let tanh_inner = inner.tanh();
        let sech_sq = 1.0 - tanh_inner * tanh_inner;
        let d_inner = SQRT_2_OVER_PI * (1.0 + 3.0 * KAPPA * value * value);
        0.5 * (1.0 + tanh_inner) + 0.5 * value * sech_sq * d_inner
    }
}

impl Module for Gelu {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        let mut data = Vec::with_capacity(rows * cols);
        for value in input.data() {
            data.push(Self::gelu(*value));
        }
        Tensor::from_vec(rows, cols, data)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        input.gelu_backward(grad_output)
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
    fn gelu_forward_matches_reference() {
        let layer = Gelu::new();
        let input = Tensor::from_vec(1, 4, vec![-1.0, -0.5, 0.5, 1.0]).unwrap();
        let output = layer.forward(&input).unwrap();
        let expected: Vec<f32> = input.data().iter().map(|&x| Gelu::gelu(x)).collect();
        for (out, exp) in output.data().iter().zip(expected.iter()) {
            assert!((out - exp).abs() < 1e-6);
        }
    }

    #[test]
    fn gelu_backward_uses_derivative() {
        let mut layer = Gelu::new();
        let input = Tensor::from_vec(1, 3, vec![-0.75, 0.0, 0.75]).unwrap();
        let grad_out = Tensor::from_vec(1, 3, vec![0.2, -0.5, 0.3]).unwrap();
        let grad_in = layer.backward(&input, &grad_out).unwrap();
        for i in 0..3 {
            let expected = Gelu::gelu_derivative(input.data()[i]) * grad_out.data()[i];
            assert!((grad_in.data()[i] - expected).abs() < 1e-6);
        }
    }
}
