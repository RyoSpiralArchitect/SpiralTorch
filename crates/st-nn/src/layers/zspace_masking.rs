// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::Module;
use crate::{PureResult, Tensor, TensorError};

/// Applies deterministic Z-slice masking and autoregressive completion driven by
/// a running latent memory.
#[derive(Clone, Debug)]
pub struct ZMasking {
    slices: usize,
    features_per_slice: usize,
    drop_stride: usize,
    drop_phase: usize,
    alpha: f32,
}

impl ZMasking {
    /// Builds a masking module that drops every `drop_stride`-th slice starting
    /// at `drop_phase` and fills it from a running latent memory with decay
    /// factor `alpha`.
    pub fn new(
        slices: usize,
        features_per_slice: usize,
        drop_stride: usize,
        drop_phase: usize,
        alpha: f32,
    ) -> PureResult<Self> {
        if slices == 0 || features_per_slice == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: slices,
                cols: features_per_slice,
            });
        }
        if drop_stride < 2 {
            return Err(TensorError::InvalidValue {
                label: "drop_stride must be at least 2",
            });
        }
        if drop_phase >= drop_stride {
            return Err(TensorError::InvalidValue {
                label: "drop_phase must be less than drop_stride",
            });
        }
        if !(0.0..=1.0).contains(&alpha) {
            return Err(TensorError::InvalidValue {
                label: "alpha must lie in [0, 1]",
            });
        }
        Ok(Self {
            slices,
            features_per_slice,
            drop_stride,
            drop_phase,
            alpha,
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

    #[inline]
    fn is_dropped(&self, slice: usize) -> bool {
        slice % self.drop_stride == self.drop_phase
    }
}

impl Module for ZMasking {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        self.assert_input(input)?;
        let (rows, cols) = input.shape();
        let span = self.features_per_slice;
        let mut output = vec![0.0f32; rows * cols];

        for r in 0..rows {
            let mut running = vec![0.0f32; span];
            let row_offset = r * cols;
            for s in 0..self.slices {
                let base = row_offset + s * span;
                let slice_data = &input.data()[base..base + span];
                let out_slice = &mut output[base..base + span];
                if self.is_dropped(s) && s != 0 {
                    out_slice.copy_from_slice(&running);
                } else {
                    out_slice.copy_from_slice(slice_data);
                }
                for i in 0..span {
                    running[i] = self.alpha * running[i] + (1.0 - self.alpha) * out_slice[i];
                }
            }
        }

        Tensor::from_vec(rows, cols, output)
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

        let (rows, cols) = input.shape();
        let span = self.features_per_slice;
        let mut grad_input = vec![0.0f32; rows * cols];

        for r in 0..rows {
            let row_offset = r * cols;
            let mut grad_running_next = vec![0.0f32; span];
            for s in (0..self.slices).rev() {
                let base = row_offset + s * span;
                let grad_slice = &grad_output.data()[base..base + span];
                let dropped = self.is_dropped(s) && s != 0;

                let mut grad_out_total = vec![0.0f32; span];
                for i in 0..span {
                    grad_out_total[i] = grad_slice[i] + grad_running_next[i] * (1.0 - self.alpha);
                }

                if !dropped {
                    for i in 0..span {
                        grad_input[base + i] += grad_out_total[i];
                    }
                }

                let mut grad_running_prev = vec![0.0f32; span];
                for i in 0..span {
                    grad_running_prev[i] = grad_running_next[i] * self.alpha;
                    if dropped {
                        grad_running_prev[i] += grad_out_total[i];
                    }
                }

                grad_running_next = grad_running_prev;
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
    fn masking_drops_stride_and_fills_with_memory() {
        let module = ZMasking::new(5, 2, 2, 1, 0.6).unwrap();
        let input = Tensor::from_vec(
            1,
            10,
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )
        .unwrap();
        let output = module.forward(&input).unwrap();
        assert_eq!(output.shape(), (1, 10));
        // Ensure dropped slices (index 1,3,...) are no longer raw copies.
        assert_ne!(output.data()[2], input.data()[2]);
        assert_ne!(output.data()[6], input.data()[6]);
    }

    #[test]
    fn masking_backward_zeroes_dropped_slices() {
        let mut module = ZMasking::new(4, 1, 2, 1, 0.5).unwrap();
        let input = Tensor::from_vec(1, 4, vec![0.2, 0.4, 0.6, 0.8]).unwrap();
        let grad_out = Tensor::from_vec(1, 4, vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let grad_in = module.backward(&input, &grad_out).unwrap();
        // Dropped slices (index 1 and 3) receive no direct gradient.
        assert_eq!(grad_in.data()[1], 0.0);
        assert_eq!(grad_in.data()[3], 0.0);
    }
}
