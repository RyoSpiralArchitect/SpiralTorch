// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::Module;
use crate::{PureResult, Tensor, TensorError};
use std::f32::consts::TAU;

/// Applies a deterministic spiral-wavelet style transform along the Z-axis to
/// expose local resonance patterns between neighbouring slices.
#[derive(Clone, Debug)]
pub struct SpiralZTransform {
    slices: usize,
    features_per_slice: usize,
    cos_table: Vec<f32>,
    sin_table: Vec<f32>,
}

impl SpiralZTransform {
    /// Builds a transform with the requested number of Z slices and features per slice.
    pub fn new(slices: usize, features_per_slice: usize) -> PureResult<Self> {
        if slices < 2 {
            return Err(TensorError::InvalidDimensions {
                rows: slices,
                cols: features_per_slice,
            });
        }
        if features_per_slice == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: slices,
                cols: features_per_slice,
            });
        }
        let mut cos_table = Vec::with_capacity(slices);
        let mut sin_table = Vec::with_capacity(slices);
        for idx in 0..slices {
            let theta = TAU * idx as f32 / slices as f32;
            cos_table.push(theta.cos());
            sin_table.push(theta.sin());
        }
        Ok(Self {
            slices,
            features_per_slice,
            cos_table,
            sin_table,
        })
    }

    fn assert_input(&self, input: &Tensor) -> PureResult<()> {
        let (_, cols) = input.shape();
        if cols != self.slices * self.features_per_slice {
            return Err(TensorError::ShapeMismatch {
                left: (input.shape().0, cols),
                right: (input.shape().0, self.slices * self.features_per_slice),
            });
        }
        Ok(())
    }
}

impl Module for SpiralZTransform {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        self.assert_input(input)?;
        let (rows, cols) = input.shape();
        let mut data = vec![0.0f32; rows * cols];
        let slice_span = self.features_per_slice;
        for r in 0..rows {
            let row_offset = r * cols;
            for s in 0..self.slices {
                let base = row_offset + s * slice_span;
                let prev = if s == 0 { self.slices - 1 } else { s - 1 };
                let next = (s + 1) % self.slices;
                let prev_offset = row_offset + prev * slice_span;
                let next_offset = row_offset + next * slice_span;
                let cos = self.cos_table[s];
                let sin = self.sin_table[s];
                for f in 0..slice_span {
                    let center = input.data()[base + f];
                    let prev_value = input.data()[prev_offset + f];
                    let next_value = input.data()[next_offset + f];
                    data[base + f] =
                        center * 1.5 + prev_value * 0.25 * cos + next_value * 0.25 * sin;
                }
            }
        }
        Tensor::from_vec(rows, cols, data)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        self.assert_input(input)?;
        self.assert_input(grad_output)?;
        if input.shape() != grad_output.shape() {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: grad_output.shape(),
            });
        }
        let (rows, cols) = grad_output.shape();
        let mut grad_input = vec![0.0f32; rows * cols];
        let slice_span = self.features_per_slice;
        for r in 0..rows {
            let row_offset = r * cols;
            for s in 0..self.slices {
                let base = row_offset + s * slice_span;
                let prev = if s == 0 { self.slices - 1 } else { s - 1 };
                let next = (s + 1) % self.slices;
                let prev_offset = row_offset + prev * slice_span;
                let next_offset = row_offset + next * slice_span;
                let cos = self.cos_table[s];
                let sin = self.sin_table[s];
                for f in 0..slice_span {
                    let grad = grad_output.data()[base + f];
                    grad_input[base + f] += grad * 1.5;
                    grad_input[prev_offset + f] += grad * 0.25 * cos;
                    grad_input[next_offset + f] += grad * 0.25 * sin;
                }
            }
        }
        Tensor::from_vec(rows, cols, grad_input)
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
    fn spiral_transform_blends_neighbouring_slices() {
        let module = SpiralZTransform::new(3, 2).unwrap();
        let input = Tensor::from_vec(1, 6, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let output = module.forward(&input).unwrap();
        assert_eq!(output.shape(), (1, 6));
        // Ensure the centre slice picks up contributions from neighbours.
        assert!(output.data()[2] > input.data()[2]);
        assert!(output.data()[3] > input.data()[3]);
    }

    #[test]
    fn spiral_transform_backward_respects_linear_weights() {
        let mut module = SpiralZTransform::new(4, 1).unwrap();
        let input = Tensor::from_vec(1, 4, vec![0.2, -0.4, 0.6, -0.8]).unwrap();
        let grad_out = Tensor::from_vec(1, 4, vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let grad_in = module.backward(&input, &grad_out).unwrap();
        // Sum of gradients reflects contributions from neighbours and self.
        let total: f32 = grad_in.data().iter().copied().sum();
        assert!((total - 6.0).abs() < 1e-4);
    }
}
