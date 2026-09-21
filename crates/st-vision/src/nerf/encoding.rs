// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::{checked_elements, row_major, validate_finite};
use st_tensor::{PureResult, Tensor, TensorError};

/// Classic NeRF-style positional encoding that expands low dimensional inputs
/// using sinusoidal basis functions.
#[derive(Clone, Debug)]
pub struct PositionalEncoding {
    input_dims: usize,
    frequencies: Vec<f32>,
    include_input: bool,
    output_dims: usize,
}

impl PositionalEncoding {
    /// Builds a positional encoding for the given dimensionality and number of
    /// frequency bands. Frequencies follow a power-of-two progression matching
    /// the original NeRF formulation. At most 128 bands have finite float32
    /// frequencies. Output dimension arithmetic is checked before allocation.
    pub fn new(input_dims: usize, num_frequencies: usize) -> PureResult<Self> {
        if input_dims == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: input_dims,
                cols: num_frequencies.max(1),
            });
        }
        if num_frequencies > 128 {
            return Err(TensorError::InvalidValue {
                label: "nerf_frequency_count",
            });
        }
        let output_dims = input_dims.checked_mul(2 * num_frequencies + 1).ok_or(
            TensorError::InvalidDimensions {
                rows: input_dims,
                cols: num_frequencies,
            },
        )?;
        let mut frequencies = Vec::with_capacity(num_frequencies);
        for idx in 0..num_frequencies {
            let freq = 2f32.powi(idx as i32);
            frequencies.push(freq);
        }
        Ok(Self {
            input_dims,
            frequencies,
            include_input: true,
            output_dims,
        })
    }

    /// Disables the residual copy of the original coordinates in the encoded
    /// representation.
    pub fn without_input(mut self) -> Self {
        if self.include_input {
            self.include_input = false;
            self.output_dims -= self.input_dims;
        }
        self
    }

    /// Returns the dimensionality of the raw coordinates accepted by the encoder.
    pub fn input_dims(&self) -> usize {
        self.input_dims
    }

    /// Returns the dimensionality of the encoded output.
    pub fn output_dims(&self) -> usize {
        self.output_dims
    }

    /// Returns the number of active frequency bands.
    pub fn num_frequencies(&self) -> usize {
        self.frequencies.len()
    }

    /// Encodes logical coordinates into a row-major tensor, preserving inputs.
    /// Rejects non-finite coordinates, even for a zero-feature encoding, and
    /// overflowing frequency-scaled phases before evaluating trigonometry.
    pub fn encode(&self, input: &Tensor) -> PureResult<Tensor> {
        let (rows, cols) = input.shape();
        if cols != self.input_dims {
            return Err(TensorError::ShapeMismatch {
                left: (rows, self.input_dims),
                right: (rows, cols),
            });
        }
        checked_elements(rows, self.output_dims)?;
        validate_finite(input.data(), "nerf_input")?;
        if let Some(&largest) = self.frequencies.last() {
            // Powers of two are ordered and positive: a finite largest phase
            // bounds every band, so no per-band finite checks are needed.
            for &value in input.data() {
                let phase = value * largest;
                if !phase.is_finite() {
                    return Err(TensorError::NonFiniteValue {
                        label: "nerf_phase",
                        value: phase,
                    });
                }
            }
        }
        let mut output = Tensor::zeros(rows, self.output_dims)?;
        if output.is_empty() {
            return Ok(output);
        }
        let input = row_major(input)?;
        for (source, target) in input
            .data()
            .chunks_exact(cols)
            .zip(output.data_mut().chunks_exact_mut(self.output_dims))
        {
            let mut offset = 0;
            if self.include_input {
                target[..cols].copy_from_slice(source);
                offset = cols;
            }
            for &freq in &self.frequencies {
                for &value in source {
                    let (sin, cos) = (value * freq).sin_cos();
                    target[offset] = sin;
                    target[offset + 1] = cos;
                    offset += 2;
                }
            }
        }
        Ok(output)
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
