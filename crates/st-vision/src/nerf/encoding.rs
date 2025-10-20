// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use st_tensor::{PureResult, Tensor, TensorError};

/// Classic NeRF-style positional encoding that expands low dimensional inputs
/// using sinusoidal basis functions.
#[derive(Clone, Debug)]
pub struct PositionalEncoding {
    input_dims: usize,
    frequencies: Vec<f32>,
    include_input: bool,
}

impl PositionalEncoding {
    /// Builds a positional encoding for the given dimensionality and number of
    /// frequency bands. Frequencies follow a power-of-two progression matching
    /// the original NeRF formulation.
    pub fn new(input_dims: usize, num_frequencies: usize) -> PureResult<Self> {
        if input_dims == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: input_dims,
                cols: num_frequencies.max(1),
            });
        }
        let mut frequencies = Vec::with_capacity(num_frequencies);
        for idx in 0..num_frequencies {
            let freq = 2f32.powi(idx as i32);
            frequencies.push(freq);
        }
        Ok(Self {
            input_dims,
            frequencies,
            include_input: true,
        })
    }

    /// Disables the residual copy of the original coordinates in the encoded
    /// representation.
    pub fn without_input(mut self) -> Self {
        self.include_input = false;
        self
    }

    /// Returns the dimensionality of the raw coordinates accepted by the encoder.
    pub fn input_dims(&self) -> usize {
        self.input_dims
    }

    /// Returns the dimensionality of the encoded output.
    pub fn output_dims(&self) -> usize {
        let base = if self.include_input {
            self.input_dims
        } else {
            0
        };
        base + self.input_dims * self.frequencies.len() * 2
    }

    /// Returns the number of active frequency bands.
    pub fn num_frequencies(&self) -> usize {
        self.frequencies.len()
    }

    /// Encodes a batch of coordinates.
    pub fn encode(&self, input: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        if cols != self.input_dims {
            return Err(TensorError::ShapeMismatch {
                left: (rows, self.input_dims),
                right: (rows, cols),
            });
        }
        let mut encoded = Vec::with_capacity(rows * self.output_dims());
        let data = input.data();
        for row in 0..rows {
            let offset = row * cols;
            if self.include_input {
                encoded.extend_from_slice(&data[offset..offset + cols]);
            }
            for &freq in &self.frequencies {
                for dim in 0..cols {
                    let value = data[offset + dim] * freq;
                    encoded.push(value.sin());
                    encoded.push(value.cos());
                }
            }
        }
        Tensor::from_vec(rows, self.output_dims(), encoded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding_preserves_expected_dimensions() {
        let encoding = PositionalEncoding::new(3, 4).unwrap();
        assert_eq!(encoding.output_dims(), 3 + 3 * 4 * 2);
        let coords = Tensor::from_vec(2, 3, vec![0.0, 1.0, 2.0, -1.0, 0.5, 3.5]).unwrap();
        let encoded = encoding.encode(&coords).unwrap();
        assert_eq!(encoded.shape(), (2, encoding.output_dims()));
    }
}
