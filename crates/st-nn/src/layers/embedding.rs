// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};

fn token_to_index(value: f32, vocab_size: usize) -> usize {
    if vocab_size == 0 {
        return 0;
    }
    if !value.is_finite() {
        return 0;
    }
    let rounded = value.round();
    if !rounded.is_finite() {
        return 0;
    }
    let idx = rounded as isize;
    if idx <= 0 {
        return 0;
    }
    let max = vocab_size.saturating_sub(1) as isize;
    idx.min(max) as usize
}

/// Simple embedding lookup table.
///
/// Inputs are expected to be integer token IDs stored as floats in a tensor
/// shaped `(batch, steps)`. Outputs are flattened embeddings shaped
/// `(batch, steps * embed_dim)` so they compose with sequence modules that
/// consume flattened steps.
#[derive(Debug)]
pub struct Embedding {
    weight: Parameter,
    vocab_size: usize,
    embed_dim: usize,
}

impl Embedding {
    pub fn new(name: impl Into<String>, vocab_size: usize, embed_dim: usize) -> PureResult<Self> {
        if vocab_size == 0 || embed_dim == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: vocab_size.max(1),
                cols: embed_dim.max(1),
            });
        }
        let name = name.into();
        let mut scale = 0.01f32;
        let weight = Tensor::from_fn(vocab_size, embed_dim, |_r, _c| {
            let value = scale;
            scale = (scale + 0.013).rem_euclid(0.05).max(1e-4);
            value
        })?;
        Ok(Self {
            weight: Parameter::new(format!("{name}::weight"), weight),
            vocab_size,
            embed_dim,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    pub fn weight(&self) -> &Parameter {
        &self.weight
    }
}

impl Module for Embedding {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (batch, steps) = input.shape();
        let output_cols = steps * self.embed_dim;
        if steps == 0 {
            return Tensor::zeros(batch, 0);
        }
        let weights = self.weight.value().data();
        let input_data = input.data();
        let mut out = Vec::with_capacity(batch * output_cols);
        for b in 0..batch {
            let row_offset = b * steps;
            for t in 0..steps {
                let token = input_data[row_offset + t];
                let idx = token_to_index(token, self.vocab_size);
                let start = idx * self.embed_dim;
                out.extend_from_slice(&weights[start..start + self.embed_dim]);
            }
        }
        Tensor::from_vec(batch, output_cols, out)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let (batch, steps) = input.shape();
        let output_cols = steps * self.embed_dim;
        if grad_output.shape() != (batch, output_cols) {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: (batch, output_cols),
            });
        }
        if batch == 0 || steps == 0 {
            return Tensor::zeros(batch, steps);
        }

        let input_data = input.data();
        let grad_data = grad_output.data();
        let mut grad_weight = vec![0.0f32; self.vocab_size * self.embed_dim];
        for b in 0..batch {
            let in_row = b * steps;
            let grad_row = b * output_cols;
            for t in 0..steps {
                let idx = token_to_index(input_data[in_row + t], self.vocab_size);
                let gw_base = idx * self.embed_dim;
                let go_base = grad_row + t * self.embed_dim;
                for c in 0..self.embed_dim {
                    grad_weight[gw_base + c] += grad_data[go_base + c];
                }
            }
        }
        let grad_w = Tensor::from_vec(self.vocab_size, self.embed_dim, grad_weight)?
            .scale(1.0 / batch as f32)?;
        self.weight.accumulate_euclidean(&grad_w)?;

        Tensor::zeros(batch, steps)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.weight)?;
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.weight)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_forward_picks_rows() {
        let layer = Embedding::new("emb", 4, 3).unwrap();
        let input = Tensor::from_vec(2, 3, vec![0.0, 1.0, 3.0, 2.0, 1.0, 0.0]).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), (2, 9));

        let weights = layer.weight().value().data();
        let expect_row = |idx: usize| -> Vec<f32> {
            let start = idx * 3;
            weights[start..start + 3].to_vec()
        };
        let out = output.data();
        assert_eq!(out[0..3], expect_row(0));
        assert_eq!(out[3..6], expect_row(1));
        assert_eq!(out[6..9], expect_row(3));
        assert_eq!(out[9..12], expect_row(2));
        assert_eq!(out[12..15], expect_row(1));
        assert_eq!(out[15..18], expect_row(0));
    }

    #[test]
    fn embedding_backward_updates_weight() {
        let mut layer = Embedding::new("emb", 5, 2).unwrap();
        layer.attach_hypergrad(-1.0, 0.05).unwrap();
        let input = Tensor::from_vec(3, 4, vec![0.0, 1.0, 2.0, 3.0, 1.0, 1.0, 4.0, 0.0, 2.0, 2.0, 2.0, 2.0]).unwrap();
        let output = layer.forward(&input).unwrap();
        let grad_output = Tensor::from_vec(output.shape().0, output.shape().1, vec![1.0; output.data().len()]).unwrap();
        let grad_in = layer.backward(&input, &grad_output).unwrap();
        assert_eq!(grad_in.shape(), input.shape());
        assert!(grad_in.data().iter().all(|v| *v == 0.0));

        let before = layer.weight().value().clone();
        layer.apply_step(0.01).unwrap();
        let after = layer.weight().value();
        assert_ne!(before, *after);
    }
}

