// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};
use std::f32::EPSILON;

/// Attention-style pointer that can move along the Z axis and gate slices with
/// a soft attention profile.
#[derive(Debug)]
pub struct ZNavigationField {
    slices: usize,
    features_per_slice: usize,
    position: Parameter,
    sharpness: Parameter,
}

impl ZNavigationField {
    /// Builds a navigation field with a learnable pointer position and sharpness.
    pub fn new(
        slices: usize,
        features_per_slice: usize,
        init_pos: f32,
        init_sharpness: f32,
    ) -> PureResult<Self> {
        if slices == 0 || features_per_slice == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: slices,
                cols: features_per_slice,
            });
        }
        if init_sharpness <= 0.0 {
            return Err(TensorError::InvalidValue {
                label: "sharpness must be positive",
            });
        }
        let position = Parameter::new(
            "zspace_navigation_position",
            Tensor::from_vec(1, 1, vec![init_pos.clamp(0.0, slices as f32 - 1.0)])?,
        );
        let sharpness = Parameter::new(
            "zspace_navigation_sharpness",
            Tensor::from_vec(1, 1, vec![init_sharpness])?,
        );
        Ok(Self {
            slices,
            features_per_slice,
            position,
            sharpness,
        })
    }

    fn assert_input(&self, tensor: &Tensor) -> PureResult<()> {
        let (_, cols) = tensor.shape();
        if cols != self.slices * self.features_per_slice {
            return Err(TensorError::ShapeMismatch {
                left: tensor.shape(),
                right: (tensor.shape().0, self.slices * self.features_per_slice),
            });
        }
        Ok(())
    }

    fn pointer(&self) -> (f32, f32) {
        (
            self.position.value().data()[0],
            self.sharpness.value().data()[0].max(EPSILON),
        )
    }

    fn compute_weights(&self, position: f32, sharpness: f32) -> Vec<f32> {
        let mut raw = Vec::with_capacity(self.slices);
        for s in 0..self.slices {
            let distance = s as f32 - position;
            raw.push((-sharpness * distance * distance).exp());
        }
        let sum: f32 = raw.iter().copied().sum::<f32>().max(EPSILON);
        raw.into_iter().map(|v| v / sum).collect()
    }

    fn accumulate_parameter_gradients(
        &mut self,
        weights: &[f32],
        dl_dw: &[f32],
        position: f32,
        sharpness: f32,
    ) -> PureResult<()> {
        let mut mean_da_pos = 0.0f32;
        let mut mean_da_sharp = 0.0f32;
        for (s, &w) in weights.iter().enumerate() {
            let delta = s as f32 - position;
            mean_da_pos += w * (2.0 * sharpness * delta);
            mean_da_sharp += w * (-(delta * delta));
        }

        let mut grad_pos = 0.0f32;
        let mut grad_sharp = 0.0f32;
        for (s, (&w, &dl)) in weights.iter().zip(dl_dw.iter()).enumerate() {
            let delta = s as f32 - position;
            let da_pos = 2.0 * sharpness * delta;
            let da_sharp = -(delta * delta);
            grad_pos += dl * w * (da_pos - mean_da_pos);
            grad_sharp += dl * w * (da_sharp - mean_da_sharp);
        }

        let pos_update = Tensor::from_vec(1, 1, vec![grad_pos])?;
        let sharp_update = Tensor::from_vec(1, 1, vec![grad_sharp])?;
        self.position.accumulate_euclidean(&pos_update)?;
        self.sharpness.accumulate_euclidean(&sharp_update)?;
        Ok(())
    }

    /// Returns a shared reference to the position parameter.
    pub fn position(&self) -> &Parameter {
        &self.position
    }

    /// Returns a shared reference to the sharpness parameter.
    pub fn sharpness(&self) -> &Parameter {
        &self.sharpness
    }
}

impl Module for ZNavigationField {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        self.assert_input(input)?;
        let (rows, cols) = input.shape();
        let (position, sharpness) = self.pointer();
        let weights = self.compute_weights(position, sharpness);
        let mut output = vec![0.0f32; rows * cols];
        let span = self.features_per_slice;
        for r in 0..rows {
            let row_offset = r * cols;
            for s in 0..self.slices {
                let base = row_offset + s * span;
                let weight = weights[s];
                for i in 0..span {
                    output[base + i] = input.data()[base + i] * weight;
                }
            }
        }
        Tensor::from_vec(rows, cols, output)
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
        let (position, sharpness) = self.pointer();
        let weights = self.compute_weights(position, sharpness);
        let span = self.features_per_slice;
        let mut grad_input = vec![0.0f32; rows * cols];
        let mut dl_dw = vec![0.0f32; self.slices];
        for r in 0..rows {
            let row_offset = r * cols;
            for s in 0..self.slices {
                let base = row_offset + s * span;
                let weight = weights[s];
                for i in 0..span {
                    let grad_val = grad_output.data()[base + i];
                    grad_input[base + i] += grad_val * weight;
                    dl_dw[s] += grad_val * input.data()[base + i];
                }
            }
        }
        self.accumulate_parameter_gradients(&weights, &dl_dw, position, sharpness)?;
        Tensor::from_vec(rows, cols, grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&crate::module::Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.position)?;
        visitor(&self.sharpness)
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut crate::module::Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.position)?;
        visitor(&mut self.sharpness)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn navigation_field_weights_sum_to_one() {
        let module = ZNavigationField::new(4, 2, 1.0, 0.5).unwrap();
        let (_, sharpness) = module.pointer();
        let weights = module.compute_weights(1.0, sharpness);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn navigation_field_scales_slices() {
        let module = ZNavigationField::new(3, 1, 0.0, 1.0).unwrap();
        let input = Tensor::from_vec(1, 3, vec![1.0, 2.0, 3.0]).unwrap();
        let output = module.forward(&input).unwrap();
        assert_eq!(output.shape(), (1, 3));
        assert!(output.data()[0] >= output.data()[1]);
        assert!(output.data()[0] >= output.data()[2]);
    }

    #[test]
    fn navigation_field_backward_accumulates_parameter_gradients() {
        let mut module = ZNavigationField::new(3, 2, 0.5, 1.5).unwrap();
        let input = Tensor::from_vec(1, 6, vec![0.3, 0.6, 0.9, 0.2, 0.1, 0.4]).unwrap();
        let grad_out = Tensor::from_vec(1, 6, vec![0.5, -0.4, 0.2, -0.1, 0.3, -0.2]).unwrap();
        let grad_in = module.backward(&input, &grad_out).unwrap();
        assert_eq!(grad_in.shape(), (1, 6));
        assert!(module.position().gradient().is_some());
        assert!(module.sharpness().gradient().is_some());
    }
}
