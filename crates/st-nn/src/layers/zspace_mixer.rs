// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::{Module, Parameter};
use st_tensor::{PureResult, Tensor, TensorError};

/// Lightweight gating module that modulates Z-space activations column-wise.
///
/// The mixer keeps a single row of parameters and broadcasts it across the
/// incoming batch, performing an element-wise product. This keeps the module
/// compatible with the hypergrad tape while remaining fully deterministic in
/// CPU-only environments.
pub struct ZSpaceMixer {
    gate: Parameter,
}

impl ZSpaceMixer {
    /// Builds a mixer with the provided number of features. Parameters start at
    /// `1.0` so the module initially acts as a pass-through.
    pub fn new(name: impl Into<String>, features: usize) -> PureResult<Self> {
        if features == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: 1,
                cols: features,
            });
        }
        let weights = Tensor::from_fn(1, features, |_, _| 1.0)?;
        Ok(Self {
            gate: Parameter::new(name, weights),
        })
    }

    /// Returns a view into the underlying parameter.
    pub fn gate(&self) -> &Parameter {
        &self.gate
    }

    /// Returns a mutable view into the parameter.
    pub fn gate_mut(&mut self) -> &mut Parameter {
        &mut self.gate
    }

    fn assert_input(&self, input: &Tensor) -> PureResult<()> {
        let (_, cols) = input.shape();
        let gate_shape = self.gate.value().shape();
        if gate_shape.1 != cols {
            return Err(TensorError::ShapeMismatch {
                left: gate_shape,
                right: (1, cols),
            });
        }
        Ok(())
    }

    fn gate_row(&self) -> &[f32] {
        self.gate.value().data()
    }
}

impl Module for ZSpaceMixer {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        self.assert_input(input)?;
        let (rows, cols) = input.shape();
        let gate: Vec<f32> = self.gate_row().to_vec();
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            let offset = r * cols;
            for c in 0..cols {
                data.push(input.data()[offset + c] * gate[c]);
            }
        }
        Tensor::from_vec(rows, cols, data)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        self.assert_input(input)?;
        if input.shape() != grad_output.shape() {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: grad_output.shape(),
            });
        }

        let (rows, cols) = input.shape();
        let gate: Vec<f32> = self.gate_row().to_vec();

        let mut grad_gate = vec![0.0f32; cols];
        for r in 0..rows {
            let offset = r * cols;
            for c in 0..cols {
                let idx = offset + c;
                grad_gate[c] += grad_output.data()[idx] * input.data()[idx];
            }
        }
        let grad_tensor = Tensor::from_vec(1, cols, grad_gate)?;
        self.gate.accumulate_euclidean(&grad_tensor)?;

        let mut grad_input = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            let offset = r * cols;
            for c in 0..cols {
                let idx = offset + c;
                grad_input.push(grad_output.data()[idx] * gate[c]);
            }
        }
        Tensor::from_vec(rows, cols, grad_input)
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
    fn mixer_scales_and_accumulates_gradients() {
        let mut mixer = ZSpaceMixer::new("mixer", 3).unwrap();
        let input = Tensor::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let output = mixer.forward(&input).unwrap();
        assert_eq!(output.data(), input.data());

        let grad_output = Tensor::from_vec(2, 3, vec![0.5, 1.0, -1.0, 0.25, 0.5, -0.5]).unwrap();
        let grad_input = mixer.backward(&input, &grad_output).unwrap();
        assert_eq!(grad_input.data(), grad_output.data());

        let gate = mixer.gate().value();
        let expected_grad = vec![
            1.0 * 0.5 + 4.0 * 0.25,
            2.0 * 1.0 + 5.0 * 0.5,
            3.0 * -1.0 + 6.0 * -0.5,
        ];
        let grads = mixer.gate().gradient().unwrap();
        for (expected, actual) in expected_grad.iter().zip(grads.data()) {
            assert!((expected - actual).abs() < 1e-6);
        }
        assert_eq!(gate.shape(), (1, 3));
    }
}
