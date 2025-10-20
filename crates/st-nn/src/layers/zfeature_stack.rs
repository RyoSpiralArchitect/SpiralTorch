// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::Module;
use crate::{PureResult, Tensor, TensorError};

/// Built-in feature projections that can be stacked per Z slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureKind {
    /// Pass-through identity channel.
    Identity,
    /// Difference between the current slice and the previous one.
    EdgeContrast,
    /// Absolute magnitude of the slice, capturing resonance energy.
    Resonance,
}

/// Combines multiple feature views per Z slice to build a richer stack of
/// representations (e.g. RGB + edge magnitude).
#[derive(Clone, Debug)]
pub struct ZFeatureStack {
    slices: usize,
    base_channels: usize,
    kinds: Vec<FeatureKind>,
}

impl ZFeatureStack {
    /// Builds a new feature stack.
    pub fn new(slices: usize, base_channels: usize, kinds: Vec<FeatureKind>) -> PureResult<Self> {
        if slices == 0 || base_channels == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: slices,
                cols: base_channels,
            });
        }
        if kinds.is_empty() {
            return Err(TensorError::InvalidValue {
                label: "feature kinds must not be empty",
            });
        }
        Ok(Self {
            slices,
            base_channels,
            kinds,
        })
    }

    fn assert_input(&self, tensor: &Tensor) -> PureResult<()> {
        let (_, cols) = tensor.shape();
        if cols != self.slices * self.base_channels {
            return Err(TensorError::ShapeMismatch {
                left: tensor.shape(),
                right: (tensor.shape().0, self.slices * self.base_channels),
            });
        }
        Ok(())
    }

    fn output_cols(&self) -> usize {
        self.slices * self.base_channels * self.kinds.len()
    }
}

impl Module for ZFeatureStack {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        self.assert_input(input)?;
        let (rows, cols) = input.shape();
        let out_cols = self.output_cols();
        let mut output = vec![0.0f32; rows * out_cols];
        let block = self.base_channels;
        for r in 0..rows {
            let row_in_offset = r * cols;
            let row_out_offset = r * out_cols;
            for s in 0..self.slices {
                let base_in = row_in_offset + s * block;
                let prev = if s == 0 { self.slices - 1 } else { s - 1 };
                let prev_in = row_in_offset + prev * block;
                let slice = &input.data()[base_in..base_in + block];
                let prev_slice = &input.data()[prev_in..prev_in + block];
                let out_base = row_out_offset + s * block * self.kinds.len();
                for (k, kind) in self.kinds.iter().enumerate() {
                    let target = &mut output[out_base + k * block..out_base + (k + 1) * block];
                    match kind {
                        FeatureKind::Identity => {
                            target.copy_from_slice(slice);
                        }
                        FeatureKind::EdgeContrast => {
                            for i in 0..block {
                                target[i] = slice[i] - prev_slice[i];
                            }
                        }
                        FeatureKind::Resonance => {
                            for i in 0..block {
                                target[i] = slice[i].abs();
                            }
                        }
                    }
                }
            }
        }
        Tensor::from_vec(rows, out_cols, output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        self.assert_input(input)?;
        let (rows, cols) = input.shape();
        let expected_cols = self.output_cols();
        if grad_output.shape() != (rows, expected_cols) {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: (rows, expected_cols),
            });
        }
        let mut grad_input = vec![0.0f32; rows * cols];
        let block = self.base_channels;
        for r in 0..rows {
            let row_in_offset = r * cols;
            let row_out_offset = r * expected_cols;
            for s in 0..self.slices {
                let base_in = row_in_offset + s * block;
                let prev = if s == 0 { self.slices - 1 } else { s - 1 };
                let prev_in = row_in_offset + prev * block;
                let slice = &input.data()[base_in..base_in + block];
                let out_base = row_out_offset + s * block * self.kinds.len();
                for (k, kind) in self.kinds.iter().enumerate() {
                    let grad_slice =
                        &grad_output.data()[out_base + k * block..out_base + (k + 1) * block];
                    match kind {
                        FeatureKind::Identity => {
                            for i in 0..block {
                                grad_input[base_in + i] += grad_slice[i];
                            }
                        }
                        FeatureKind::EdgeContrast => {
                            for i in 0..block {
                                grad_input[base_in + i] += grad_slice[i];
                                grad_input[prev_in + i] -= grad_slice[i];
                            }
                        }
                        FeatureKind::Resonance => {
                            for i in 0..block {
                                let sign = if slice[i] > 0.0 {
                                    1.0
                                } else if slice[i] < 0.0 {
                                    -1.0
                                } else {
                                    0.0
                                };
                                grad_input[base_in + i] += grad_slice[i] * sign;
                            }
                        }
                    }
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
    fn feature_stack_builds_multiple_views() {
        let module =
            ZFeatureStack::new(2, 3, vec![FeatureKind::Identity, FeatureKind::EdgeContrast])
                .unwrap();
        let input = Tensor::from_vec(
            1,
            6,
            vec![
                0.1, 0.2, 0.3, // slice 0
                0.4, 0.5, 0.6, // slice 1
            ],
        )
        .unwrap();
        let output = module.forward(&input).unwrap();
        assert_eq!(output.shape(), (1, 12));
        // Identity copy lives in first block.
        assert_eq!(&output.data()[0..3], &input.data()[0..3]);
        // Edge contrast uses slice1 - slice0.
        assert!((output.data()[9] - (0.6 - 0.3)).abs() < 1e-6);
    }

    #[test]
    fn feature_stack_backward_accumulates_gradients() {
        let mut module =
            ZFeatureStack::new(3, 2, vec![FeatureKind::Identity, FeatureKind::Resonance]).unwrap();
        let input = Tensor::from_vec(1, 6, vec![0.2, -0.4, 0.6, -0.8, 1.0, -1.2]).unwrap();
        let grad_out = Tensor::from_vec(1, 12, vec![1.0; 12]).unwrap();
        let grad_in = module.backward(&input, &grad_out).unwrap();
        assert_eq!(grad_in.shape(), (1, 6));
        // Resonance branch contributes sign information (second channel negative entries).
        assert!(grad_in.data()[1] < 1.0);
    }
}
