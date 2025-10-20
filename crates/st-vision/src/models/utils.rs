// SPDX-License-Identifier: AGPL-3.0-or-later

use st_nn::PureResult;
use st_tensor::{Tensor, TensorError};

pub(crate) fn conv_output_hw(
    input_hw: (usize, usize),
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> PureResult<(usize, usize)> {
    let (h, w) = input_hw;
    if h == 0 || w == 0 {
        return Err(TensorError::InvalidDimensions { rows: h, cols: w });
    }
    let effective_kh = (kernel.0 - 1) * dilation.0 + 1;
    let effective_kw = (kernel.1 - 1) * dilation.1 + 1;
    if h + 2 * padding.0 < effective_kh || w + 2 * padding.1 < effective_kw {
        return Err(TensorError::InvalidDimensions {
            rows: h + 2 * padding.0,
            cols: effective_kh.max(effective_kw),
        });
    }
    let oh = (h + 2 * padding.0 - effective_kh) / stride.0 + 1;
    let ow = (w + 2 * padding.1 - effective_kw) / stride.1 + 1;
    Ok((oh, ow))
}

pub(crate) fn pool_output_hw(
    input_hw: (usize, usize),
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> PureResult<(usize, usize)> {
    let (h, w) = input_hw;
    if h == 0 || w == 0 {
        return Err(TensorError::InvalidDimensions { rows: h, cols: w });
    }
    if h + 2 * padding.0 < kernel.0 || w + 2 * padding.1 < kernel.1 {
        return Err(TensorError::InvalidDimensions {
            rows: h + 2 * padding.0,
            cols: kernel.0.max(kernel.1),
        });
    }
    let oh = (h + 2 * padding.0 - kernel.0) / stride.0 + 1;
    let ow = (w + 2 * padding.1 - kernel.1) / stride.1 + 1;
    Ok((oh, ow))
}

/// Flattens a convolution-style tensor into per-location tokens.
///
/// The input is expected to have shape `(batch, channels * height * width)`. The
/// returned tensor groups spatial locations as rows so that sequence-style
/// modules can operate over them easily.
pub(crate) fn conv_to_tokens(
    tensor: &Tensor,
    channels: usize,
    hw: (usize, usize),
) -> PureResult<Tensor> {
    let (batch, cols) = tensor.shape();
    let expected = channels
        .checked_mul(hw.0)
        .and_then(|value| value.checked_mul(hw.1))
        .ok_or(TensorError::InvalidDimensions {
            rows: hw.0,
            cols: hw.1,
        })?;
    if cols != expected {
        return Err(TensorError::ShapeMismatch {
            left: (batch, cols),
            right: (batch, expected),
        });
    }
    let tokens_per_batch = hw.0 * hw.1;
    let mut data = Vec::with_capacity(batch * tokens_per_batch * channels);
    let spatial = tokens_per_batch;
    for b in 0..batch {
        let row = &tensor.data()[b * cols..(b + 1) * cols];
        for index in 0..spatial {
            for c in 0..channels {
                let offset = c * spatial + index;
                data.push(row[offset]);
            }
        }
    }
    Tensor::from_vec(batch * tokens_per_batch, channels, data)
}

/// Packs token rows back into a flattened convolution layout.
pub(crate) fn tokens_to_conv(
    tokens: &Tensor,
    batch: usize,
    channels: usize,
    hw: (usize, usize),
) -> PureResult<Tensor> {
    let tokens_per_batch = hw.0 * hw.1;
    if tokens.shape().0 != batch * tokens_per_batch || tokens.shape().1 != channels {
        return Err(TensorError::ShapeMismatch {
            left: tokens.shape(),
            right: (batch * tokens_per_batch, channels),
        });
    }
    let mut data = vec![0.0f32; batch * channels * tokens_per_batch];
    for b in 0..batch {
        for index in 0..tokens_per_batch {
            for c in 0..channels {
                let src = (b * tokens_per_batch + index) * channels + c;
                let dst = b * channels * tokens_per_batch + c * tokens_per_batch + index;
                data[dst] = tokens.data()[src];
            }
        }
    }
    Tensor::from_vec(batch, channels * tokens_per_batch, data)
}

/// Averages tokens belonging to the same batch, returning a single embedding
/// per batch element.
pub(crate) fn mean_pool_tokens(
    tokens: &Tensor,
    batch: usize,
    tokens_per_batch: usize,
) -> PureResult<Tensor> {
    if tokens_per_batch == 0 {
        return Err(TensorError::InvalidDimensions {
            rows: batch,
            cols: tokens_per_batch,
        });
    }
    if tokens.shape().0 != batch * tokens_per_batch {
        return Err(TensorError::ShapeMismatch {
            left: tokens.shape(),
            right: (batch * tokens_per_batch, tokens.shape().1),
        });
    }
    let mut pooled = Vec::with_capacity(batch * tokens.shape().1);
    for b in 0..batch {
        let start = b * tokens_per_batch;
        let end = start + tokens_per_batch;
        let mut sums = vec![0.0f32; tokens.shape().1];
        for row in start..end {
            let slice = &tokens.data()[row * tokens.shape().1..(row + 1) * tokens.shape().1];
            for (sum, value) in sums.iter_mut().zip(slice.iter()) {
                *sum += *value;
            }
        }
        for value in sums.iter_mut() {
            *value /= tokens_per_batch as f32;
        }
        pooled.extend_from_slice(&sums);
    }
    Tensor::from_vec(batch, tokens.shape().1, pooled)
}

/// Distributes gradients from a pooled embedding back to the individual tokens.
pub(crate) fn mean_pool_tokens_backward(
    grad_output: &Tensor,
    batch: usize,
    tokens_per_batch: usize,
) -> PureResult<Tensor> {
    if tokens_per_batch == 0 {
        return Err(TensorError::InvalidDimensions {
            rows: batch,
            cols: tokens_per_batch,
        });
    }
    if grad_output.shape().0 != batch {
        return Err(TensorError::ShapeMismatch {
            left: grad_output.shape(),
            right: (batch, grad_output.shape().1),
        });
    }
    let mut data = vec![0.0f32; batch * tokens_per_batch * grad_output.shape().1];
    for b in 0..batch {
        let grad = &grad_output.data()[b * grad_output.shape().1..(b + 1) * grad_output.shape().1];
        for token in 0..tokens_per_batch {
            let offset = (b * tokens_per_batch + token) * grad_output.shape().1;
            for (idx, value) in grad.iter().enumerate() {
                data[offset + idx] = *value / tokens_per_batch as f32;
            }
        }
    }
    Tensor::from_vec(batch * tokens_per_batch, grad_output.shape().1, data)
}
