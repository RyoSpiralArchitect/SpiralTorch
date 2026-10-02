//! Scaled dot-product attention with additive Z-space and pairwise score biases.
//! Causality is structural, not an arbitrarily large negative score. All tensors
//! use explicit batch/head axes; broadcasting belongs to the tensor view layer.

use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionMask {
    None,
    /// Query `q` sees keys `0..=query_offset + q`. A cached decode must specify
    /// its absolute query offset, rather than using a top-left triangular mask.
    Causal {
        query_offset: usize,
    },
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum AttentionError {
    #[error("attention expects Q [B,H,Q,D], K/V [B,H,K,D], with nonzero K and D")]
    Shape,
    #[error("attention bias expects Z [B,H,K] or pairwise [B,H,Q,K]; broadcast explicitly")]
    BiasShape,
    #[error("attention shape product overflows usize")]
    Overflow,
    #[error("attention scale must be finite")]
    Scale,
    #[error("causal query offset and length must fit inside the key sequence")]
    QueryOffset,
    #[error("attention data length does not match its validated shape")]
    Length,
    #[error("attention inputs, scores and computed outputs must be finite")]
    NonFinite,
}

#[derive(Clone, Copy, Debug)]
pub struct AttentionSpec {
    query: [usize; 4],
    key: [usize; 4],
    contexts: usize,
    scale: f32,
    mask: AttentionMask,
}

fn product(shape: &[usize]) -> Result<usize, AttentionError> {
    shape.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d).ok_or(AttentionError::Overflow)
    })
}

impl AttentionSpec {
    /// Empty batch/head/query axes are allowed. No dropout, implicit head
    /// expansion (GQA), padding mask, or backward operation is implied.
    pub fn new(
        query: &[usize],
        key: &[usize],
        value: &[usize],
        scale: f32,
        mask: AttentionMask,
    ) -> Result<Self, AttentionError> {
        let query: [usize; 4] = query.try_into().map_err(|_| AttentionError::Shape)?;
        let key: [usize; 4] = key.try_into().map_err(|_| AttentionError::Shape)?;
        if key.as_slice() != value
            || query[..2] != key[..2]
            || query[3] != key[3]
            || key[2] == 0
            || key[3] == 0
        {
            return Err(AttentionError::Shape);
        }
        if !scale.is_finite() {
            return Err(AttentionError::Scale);
        }
        if let AttentionMask::Causal { query_offset } = mask {
            if query_offset
                .checked_add(query[2])
                .is_none_or(|end| end > key[2])
            {
                return Err(AttentionError::QueryOffset);
            }
        }
        product(&query)?;
        product(&key)?;
        product(&[query[0], query[1], query[2], key[2]])?;
        let contexts = product(&query[..2])?;
        Ok(Self {
            query,
            key,
            contexts,
            scale,
            mask,
        })
    }

    pub fn query_shape(&self) -> [usize; 4] {
        self.query
    }
    pub fn key_shape(&self) -> [usize; 4] {
        self.key
    }
    pub fn contexts(&self) -> usize {
        self.contexts
    }
    pub fn queries(&self) -> usize {
        self.query[2]
    }
    pub fn keys(&self) -> usize {
        self.key[2]
    }
    pub fn head_dim(&self) -> usize {
        self.query[3]
    }
    pub fn scale(&self) -> f32 {
        self.scale
    }
    pub fn mask(&self) -> AttentionMask {
        self.mask
    }
    pub fn z_bias_shape(&self) -> [usize; 3] {
        [self.query[0], self.query[1], self.key[2]]
    }
    pub fn pair_bias_shape(&self) -> [usize; 4] {
        [self.query[0], self.query[1], self.query[2], self.key[2]]
    }
    pub fn validate_bias_shapes(
        &self,
        z_bias: Option<&[usize]>,
        pair_bias: Option<&[usize]>,
    ) -> Result<(), AttentionError> {
        if z_bias.is_some_and(|s| s != self.z_bias_shape())
            || pair_bias.is_some_and(|s| s != self.pair_bias_shape())
        {
            return Err(AttentionError::BiasShape);
        }
        Ok(())
    }
}

fn finite(value: f32) -> Result<f32, AttentionError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(AttentionError::NonFinite)
    }
}

/// CPU oracle, not a routing fallback. Biases are added AFTER scaling QK.
/// Online normalized softmax avoids materializing a quadratic score tensor.
/// Inputs and f32 score/weighted-output arithmetic must remain finite; no
/// bitwise identity between different reduction orders is promised.
pub fn attention_reference(
    spec: AttentionSpec,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    z_bias: Option<&[f32]>,
    pair_bias: Option<&[f32]>,
) -> Result<Vec<f32>, AttentionError> {
    for (values, len) in [
        (Some(query), product(&spec.query)?),
        (Some(key), product(&spec.key)?),
        (Some(value), product(&spec.key)?),
        (z_bias, product(&spec.z_bias_shape())?),
        (pair_bias, product(&spec.pair_bias_shape())?),
    ] {
        if let Some(values) = values {
            if values.len() != len {
                return Err(AttentionError::Length);
            }
            if values.iter().any(|v| !v.is_finite()) {
                return Err(AttentionError::NonFinite);
            }
        }
    }
    let mut output = vec![0.; query.len()];
    let d = spec.head_dim();
    for context in 0..spec.contexts() {
        for q in 0..spec.queries() {
            let row = context * spec.queries() + q;
            let visible = match spec.mask() {
                AttentionMask::None => spec.keys(),
                AttentionMask::Causal { query_offset } => query_offset + q + 1,
            };
            let mut maximum = -f32::MAX;
            let mut sum = 0.;
            for k in 0..visible {
                let key_row = context * spec.keys() + k;
                let mut dot = 0.;
                for i in 0..d {
                    dot = finite(dot + finite(query[row * d + i] * key[key_row * d + i])?)?;
                }
                let mut score = finite(dot * spec.scale())?;
                if let Some(bias) = z_bias {
                    score = finite(score + bias[key_row])?;
                }
                if let Some(bias) = pair_bias {
                    score = finite(score + bias[row * spec.keys() + k])?;
                }
                let next_maximum = maximum.max(score);
                let previous = sum * (maximum - next_maximum).exp();
                let current = (score - next_maximum).exp();
                sum = finite(previous + current)?;
                let alpha = previous / sum;
                let weight = current / sum;
                for i in 0..d {
                    output[row * d + i] =
                        finite(output[row * d + i] * alpha + value[key_row * d + i] * weight)?;
                }
                maximum = next_maximum;
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests;
