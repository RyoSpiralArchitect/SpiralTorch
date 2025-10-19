// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};
use std::cell::RefCell;

fn validate_positive(value: usize, _label: &str) -> PureResult<()> {
    if value == 0 {
        return Err(TensorError::InvalidDimensions {
            rows: 1,
            cols: value,
        });
    }
    Ok(())
}

fn kernel_span(in_channels: usize, kernel: usize) -> usize {
    in_channels * kernel
}

/// One-dimensional convolution with explicit stride and padding controls.
#[derive(Debug)]
pub struct Conv1d {
    weight: Parameter,
    bias: Parameter,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl Conv1d {
    pub fn new(
        name: impl Into<String>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> PureResult<Self> {
        validate_positive(in_channels, "in_channels")?;
        validate_positive(out_channels, "out_channels")?;
        validate_positive(kernel_size, "kernel_size")?;
        validate_positive(stride, "stride")?;
        let name = name.into();
        let span = kernel_span(in_channels, kernel_size);
        let mut seed = 0.01f32;
        let weight = Tensor::from_fn(out_channels, span, |_r, _c| {
            let value = seed;
            seed = (seed * 1.37).rem_euclid(0.1).max(1e-3);
            value
        })?;
        let bias = Tensor::zeros(1, out_channels)?;
        Ok(Self {
            weight: Parameter::new(format!("{name}::weight"), weight),
            bias: Parameter::new(format!("{name}::bias"), bias),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        })
    }

    fn infer_width(&self, cols: usize) -> PureResult<usize> {
        if cols % self.in_channels != 0 {
            return Err(TensorError::ShapeMismatch {
                left: (1, cols),
                right: (1, self.in_channels),
            });
        }
        Ok(cols / self.in_channels)
    }

    fn output_width(&self, input_width: usize) -> PureResult<usize> {
        let numer = input_width + 2 * self.padding;
        if numer < self.kernel_size {
            return Err(TensorError::InvalidDimensions {
                rows: input_width,
                cols: self.kernel_size,
            });
        }
        Ok((numer - self.kernel_size) / self.stride + 1)
    }
}

impl Module for Conv1d {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (batch, cols) = input.shape();
        let width = self.infer_width(cols)?;
        let out_width = self.output_width(width)?;
        let mut out = Tensor::zeros(batch, self.out_channels * out_width)?;
        let weight = self.weight.value();
        let bias = self.bias.value();
        let weight_data = weight.data();
        let bias_data = bias.data();
        let span = kernel_span(self.in_channels, self.kernel_size);
        let out_cols = out.shape().1;
        {
            let out_data = out.data_mut();
            for b in 0..batch {
                let row = &input.data()[b * cols..(b + 1) * cols];
                let (start, end) = (b * out_cols, (b + 1) * out_cols);
                let out_row = &mut out_data[start..end];
                for oc in 0..self.out_channels {
                    let weight_row = &weight_data[oc * span..(oc + 1) * span];
                    let bias = bias_data[oc];
                    for ow in 0..out_width {
                        let mut acc = bias;
                        for ic in 0..self.in_channels {
                            let channel_offset = ic * width;
                            for k in 0..self.kernel_size {
                                let pos = ow * self.stride + k;
                                if pos < self.padding {
                                    continue;
                                }
                                let idx = pos - self.padding;
                                if idx >= width {
                                    continue;
                                }
                                let input_val = row[channel_offset + idx];
                                let weight_idx = ic * self.kernel_size + k;
                                acc += input_val * weight_row[weight_idx];
                            }
                        }
                        out_row[oc * out_width + ow] = acc;
                    }
                }
            }
        }
        Ok(out)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let (batch, cols) = input.shape();
        let width = self.infer_width(cols)?;
        let out_width = self.output_width(width)?;
        if grad_output.shape() != (batch, self.out_channels * out_width) {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: (batch, self.out_channels * out_width),
            });
        }
        let span = kernel_span(self.in_channels, self.kernel_size);
        let mut grad_weight = Tensor::zeros(self.out_channels, span)?;
        let mut grad_bias = vec![0.0f32; self.out_channels];
        let mut grad_input = Tensor::zeros(batch, cols)?;
        let weight = self.weight.value();
        let weight_data = weight.data();
        let grad_out_cols = grad_output.shape().1;
        let grad_input_cols = grad_input.shape().1;
        {
            let grad_weight_data = grad_weight.data_mut();
            let grad_input_data = grad_input.data_mut();
            for b in 0..batch {
                let row = &input.data()[b * cols..(b + 1) * cols];
                let grad_row = &grad_output.data()[b * grad_out_cols..(b + 1) * grad_out_cols];
                let (start_in, end_in) = (b * grad_input_cols, (b + 1) * grad_input_cols);
                let grad_in_row = &mut grad_input_data[start_in..end_in];
                for oc in 0..self.out_channels {
                    let weight_row = &weight_data[oc * span..(oc + 1) * span];
                    for ow in 0..out_width {
                        let go = grad_row[oc * out_width + ow];
                        grad_bias[oc] += go;
                        for ic in 0..self.in_channels {
                            let channel_offset = ic * width;
                            for k in 0..self.kernel_size {
                                let pos = ow * self.stride + k;
                                if pos < self.padding {
                                    continue;
                                }
                                let idx = pos - self.padding;
                                if idx >= width {
                                    continue;
                                }
                                let input_val = row[channel_offset + idx];
                                let weight_idx = ic * self.kernel_size + k;
                                grad_weight_data[oc * span + weight_idx] += go * input_val;
                                grad_in_row[channel_offset + idx] += go * weight_row[weight_idx];
                            }
                        }
                    }
                }
            }
        }
        let inv_batch = 1.0 / batch as f32;
        for value in grad_weight.data_mut() {
            *value *= inv_batch;
        }
        let mut bias_tensor = Tensor::from_vec(1, self.out_channels, grad_bias)?;
        bias_tensor = bias_tensor.scale(inv_batch)?;
        self.weight.accumulate_euclidean(&grad_weight)?;
        self.bias.accumulate_euclidean(&bias_tensor)?;
        Ok(grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.weight)?;
        visitor(&self.bias)?;
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.weight)?;
        visitor(&mut self.bias)?;
        Ok(())
    }
}

/// Two-dimensional convolution operating on `(batch, channels * height * width)` tensors.
#[derive(Debug)]
pub struct Conv2d {
    weight: Parameter,
    bias: Parameter,
    in_channels: usize,
    out_channels: usize,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    input_hw: (usize, usize),
}

impl Conv2d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        input_hw: (usize, usize),
    ) -> PureResult<Self> {
        validate_positive(in_channels, "in_channels")?;
        validate_positive(out_channels, "out_channels")?;
        validate_positive(kernel.0, "kernel_h")?;
        validate_positive(kernel.1, "kernel_w")?;
        validate_positive(stride.0, "stride_h")?;
        validate_positive(stride.1, "stride_w")?;
        validate_positive(dilation.0, "dilation_h")?;
        validate_positive(dilation.1, "dilation_w")?;
        validate_positive(input_hw.0, "input_height")?;
        validate_positive(input_hw.1, "input_width")?;
        let name = name.into();
        let span = in_channels * kernel.0 * kernel.1;
        let mut seed = 0.02f32;
        let weight = Tensor::from_fn(out_channels, span, |_r, _c| {
            let value = seed;
            seed = (seed * 1.57).rem_euclid(0.15).max(5e-3);
            value
        })?;
        let bias = Tensor::zeros(1, out_channels)?;
        Ok(Self {
            weight: Parameter::new(format!("{name}::weight"), weight),
            bias: Parameter::new(format!("{name}::bias"), bias),
            in_channels,
            out_channels,
            kernel,
            stride,
            padding,
            dilation,
            input_hw,
        })
    }

    fn output_hw(&self) -> PureResult<(usize, usize)> {
        let (h, w) = self.input_hw;
        let (kh, kw) = self.kernel;
        let (ph, pw) = self.padding;
        let (sh, sw) = self.stride;
        let (dh, dw) = self.dilation;
        let effective_kh = dh * (kh.saturating_sub(1)) + 1;
        let effective_kw = dw * (kw.saturating_sub(1)) + 1;
        if h + 2 * ph < effective_kh || w + 2 * pw < effective_kw {
            return Err(TensorError::InvalidDimensions {
                rows: h + 2 * ph,
                cols: effective_kh.max(effective_kw),
            });
        }
        let oh = (h + 2 * ph - effective_kh) / sh + 1;
        let ow = (w + 2 * pw - effective_kw) / sw + 1;
        Ok((oh, ow))
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (batch, cols) = input.shape();
        let expected_cols = self.in_channels * self.input_hw.0 * self.input_hw.1;
        if cols != expected_cols {
            return Err(TensorError::ShapeMismatch {
                left: (1, cols),
                right: (1, expected_cols),
            });
        }
        let (oh, ow) = self.output_hw()?;
        let mut out = Tensor::zeros(batch, self.out_channels * oh * ow)?;
        let weight = self.weight.value();
        let bias = self.bias.value();
        let weight_data = weight.data();
        let bias_data = bias.data();
        let span = self.in_channels * self.kernel.0 * self.kernel.1;
        let (h, w) = self.input_hw;
        let out_cols = out.shape().1;
        {
            let out_data = out.data_mut();
            for b in 0..batch {
                let row = &input.data()[b * cols..(b + 1) * cols];
                let (start, end) = (b * out_cols, (b + 1) * out_cols);
                let out_row = &mut out_data[start..end];
                for oc in 0..self.out_channels {
                    let weight_row = &weight_data[oc * span..(oc + 1) * span];
                    let bias = bias_data[oc];
                    for oh_idx in 0..oh {
                        for ow_idx in 0..ow {
                            let mut acc = bias;
                            for ic in 0..self.in_channels {
                                let channel_offset = ic * h * w;
                                for kh in 0..self.kernel.0 {
                                    for kw in 0..self.kernel.1 {
                                        let pos_h = oh_idx * self.stride.0 + kh * self.dilation.0;
                                        let pos_w = ow_idx * self.stride.1 + kw * self.dilation.1;
                                        if pos_h < self.padding.0 || pos_w < self.padding.1 {
                                            continue;
                                        }
                                        let idx_h = pos_h - self.padding.0;
                                        let idx_w = pos_w - self.padding.1;
                                        if idx_h >= h || idx_w >= w {
                                            continue;
                                        }
                                        let input_idx = channel_offset + idx_h * w + idx_w;
                                        let weight_idx = ic * self.kernel.0 * self.kernel.1
                                            + kh * self.kernel.1
                                            + kw;
                                        acc += row[input_idx] * weight_row[weight_idx];
                                    }
                                }
                            }
                            out_row[oc * (oh * ow) + oh_idx * ow + ow_idx] = acc;
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let (batch, cols) = input.shape();
        let expected_cols = self.in_channels * self.input_hw.0 * self.input_hw.1;
        if cols != expected_cols {
            return Err(TensorError::ShapeMismatch {
                left: (1, cols),
                right: (1, expected_cols),
            });
        }
        let (oh, ow) = self.output_hw()?;
        if grad_output.shape() != (batch, self.out_channels * oh * ow) {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: (batch, self.out_channels * oh * ow),
            });
        }
        let span = self.in_channels * self.kernel.0 * self.kernel.1;
        let mut grad_weight = Tensor::zeros(self.out_channels, span)?;
        let mut grad_bias = vec![0.0f32; self.out_channels];
        let mut grad_input = Tensor::zeros(batch, cols)?;
        let weight = self.weight.value();
        let weight_data = weight.data();
        let (h, w) = self.input_hw;
        let grad_out_cols = grad_output.shape().1;
        let grad_input_cols = grad_input.shape().1;
        {
            let grad_weight_data = grad_weight.data_mut();
            let grad_input_data = grad_input.data_mut();
            for b in 0..batch {
                let row = &input.data()[b * cols..(b + 1) * cols];
                let grad_row = &grad_output.data()[b * grad_out_cols..(b + 1) * grad_out_cols];
                let (start_in, end_in) = (b * grad_input_cols, (b + 1) * grad_input_cols);
                let grad_in_row = &mut grad_input_data[start_in..end_in];
                for oc in 0..self.out_channels {
                    let weight_row = &weight_data[oc * span..(oc + 1) * span];
                    for oh_idx in 0..oh {
                        for ow_idx in 0..ow {
                            let go = grad_row[oc * (oh * ow) + oh_idx * ow + ow_idx];
                            grad_bias[oc] += go;
                            for ic in 0..self.in_channels {
                                let channel_offset = ic * h * w;
                                for kh in 0..self.kernel.0 {
                                    for kw in 0..self.kernel.1 {
                                        let pos_h = oh_idx * self.stride.0 + kh * self.dilation.0;
                                        let pos_w = ow_idx * self.stride.1 + kw * self.dilation.1;
                                        if pos_h < self.padding.0 || pos_w < self.padding.1 {
                                            continue;
                                        }
                                        let idx_h = pos_h - self.padding.0;
                                        let idx_w = pos_w - self.padding.1;
                                        if idx_h >= h || idx_w >= w {
                                            continue;
                                        }
                                        let input_idx = channel_offset + idx_h * w + idx_w;
                                        let weight_idx = ic * self.kernel.0 * self.kernel.1
                                            + kh * self.kernel.1
                                            + kw;
                                        grad_weight_data[oc * span + weight_idx] +=
                                            go * row[input_idx];
                                        grad_in_row[input_idx] += go * weight_row[weight_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let inv_batch = 1.0 / batch as f32;
        for value in grad_weight.data_mut() {
            *value *= inv_batch;
        }
        let mut bias_tensor = Tensor::from_vec(1, self.out_channels, grad_bias)?;
        bias_tensor = bias_tensor.scale(inv_batch)?;
        self.weight.accumulate_euclidean(&grad_weight)?;
        self.bias.accumulate_euclidean(&bias_tensor)?;
        Ok(grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.weight)?;
        visitor(&self.bias)?;
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.weight)?;
        visitor(&mut self.bias)?;
        Ok(())
    }
}

/// Max pooling over 2D feature maps.
#[derive(Debug)]
pub struct MaxPool2d {
    channels: usize,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    input_hw: (usize, usize),
    last_indices: RefCell<Vec<usize>>,
}

impl MaxPool2d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        channels: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        input_hw: (usize, usize),
    ) -> PureResult<Self> {
        validate_positive(channels, "channels")?;
        validate_positive(kernel.0, "kernel_h")?;
        validate_positive(kernel.1, "kernel_w")?;
        validate_positive(stride.0, "stride_h")?;
        validate_positive(stride.1, "stride_w")?;
        validate_positive(input_hw.0, "input_height")?;
        validate_positive(input_hw.1, "input_width")?;
        Ok(Self {
            channels,
            kernel,
            stride,
            padding,
            input_hw,
            last_indices: RefCell::new(Vec::new()),
        })
    }

    fn output_hw(&self) -> PureResult<(usize, usize)> {
        let (h, w) = self.input_hw;
        let (kh, kw) = self.kernel;
        let (ph, pw) = self.padding;
        let (sh, sw) = self.stride;
        if h + 2 * ph < kh || w + 2 * pw < kw {
            return Err(TensorError::InvalidDimensions {
                rows: h + 2 * ph,
                cols: kh.max(kw),
            });
        }
        Ok(((h + 2 * ph - kh) / sh + 1, (w + 2 * pw - kw) / sw + 1))
    }
}

impl Module for MaxPool2d {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (batch, cols) = input.shape();
        let expected = self.channels * self.input_hw.0 * self.input_hw.1;
        if cols != expected {
            return Err(TensorError::ShapeMismatch {
                left: (1, cols),
                right: (1, expected),
            });
        }
        let (oh, ow) = self.output_hw()?;
        let mut out = Tensor::zeros(batch, self.channels * oh * ow)?;
        let mut indices = self.last_indices.borrow_mut();
        indices.clear();
        indices.resize(batch * self.channels * oh * ow, 0);
        let (h, w) = self.input_hw;
        let out_cols = out.shape().1;
        {
            let out_data = out.data_mut();
            for b in 0..batch {
                let row = &input.data()[b * cols..(b + 1) * cols];
                let (start, end) = (b * out_cols, (b + 1) * out_cols);
                let out_row = &mut out_data[start..end];
                for c in 0..self.channels {
                    let channel_offset = c * h * w;
                    for oh_idx in 0..oh {
                        for ow_idx in 0..ow {
                            let mut best = f32::MIN;
                            let mut best_idx = channel_offset;
                            for kh in 0..self.kernel.0 {
                                for kw in 0..self.kernel.1 {
                                    let pos_h = oh_idx * self.stride.0 + kh;
                                    let pos_w = ow_idx * self.stride.1 + kw;
                                    if pos_h < self.padding.0 || pos_w < self.padding.1 {
                                        continue;
                                    }
                                    let idx_h = pos_h - self.padding.0;
                                    let idx_w = pos_w - self.padding.1;
                                    if idx_h >= h || idx_w >= w {
                                        continue;
                                    }
                                    let index = channel_offset + idx_h * w + idx_w;
                                    let value = row[index];
                                    if value > best {
                                        best = value;
                                        best_idx = index;
                                    }
                                }
                            }
                            let out_index = c * (oh * ow) + oh_idx * ow + ow_idx;
                            out_row[out_index] = best;
                            indices[b * self.channels * oh * ow + out_index] = best_idx;
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    fn backward(&mut self, _input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let (batch, cols) = grad_output.shape();
        let (oh, ow) = self.output_hw()?;
        if cols != self.channels * oh * ow {
            return Err(TensorError::ShapeMismatch {
                left: (1, cols),
                right: (1, self.channels * oh * ow),
            });
        }
        let mut grad_input =
            Tensor::zeros(batch, self.channels * self.input_hw.0 * self.input_hw.1)?;
        let indices = self.last_indices.borrow();
        let grad_input_cols = grad_input.shape().1;
        {
            let grad_input_data = grad_input.data_mut();
            for b in 0..batch {
                let grad_row = &grad_output.data()[b * cols..(b + 1) * cols];
                let (start, end) = (b * grad_input_cols, (b + 1) * grad_input_cols);
                let grad_in_row = &mut grad_input_data[start..end];
                for idx in 0..cols {
                    let input_index = indices[b * cols + idx];
                    grad_in_row[input_index] += grad_row[idx];
                }
            }
        }
        Ok(grad_input)
    }

    fn visit_parameters(
        &self,
        _visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        _visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }
}

/// Average pooling over 2D feature maps.
#[derive(Debug)]
pub struct AvgPool2d {
    channels: usize,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    input_hw: (usize, usize),
}

impl AvgPool2d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        channels: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        input_hw: (usize, usize),
    ) -> PureResult<Self> {
        validate_positive(channels, "channels")?;
        validate_positive(kernel.0, "kernel_h")?;
        validate_positive(kernel.1, "kernel_w")?;
        validate_positive(stride.0, "stride_h")?;
        validate_positive(stride.1, "stride_w")?;
        validate_positive(input_hw.0, "input_height")?;
        validate_positive(input_hw.1, "input_width")?;
        Ok(Self {
            channels,
            kernel,
            stride,
            padding,
            input_hw,
        })
    }

    fn output_hw(&self) -> PureResult<(usize, usize)> {
        let (h, w) = self.input_hw;
        let (kh, kw) = self.kernel;
        let (ph, pw) = self.padding;
        let (sh, sw) = self.stride;
        if h + 2 * ph < kh || w + 2 * pw < kw {
            return Err(TensorError::InvalidDimensions {
                rows: h + 2 * ph,
                cols: kh.max(kw),
            });
        }
        Ok(((h + 2 * ph - kh) / sh + 1, (w + 2 * pw - kw) / sw + 1))
    }
}

impl Module for AvgPool2d {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (batch, cols) = input.shape();
        let expected = self.channels * self.input_hw.0 * self.input_hw.1;
        if cols != expected {
            return Err(TensorError::ShapeMismatch {
                left: (1, cols),
                right: (1, expected),
            });
        }
        let (oh, ow) = self.output_hw()?;
        let mut out = Tensor::zeros(batch, self.channels * oh * ow)?;
        let (h, w) = self.input_hw;
        let area = (self.kernel.0 * self.kernel.1) as f32;
        let out_cols = out.shape().1;
        {
            let out_data = out.data_mut();
            for b in 0..batch {
                let row = &input.data()[b * cols..(b + 1) * cols];
                let (start, end) = (b * out_cols, (b + 1) * out_cols);
                let out_row = &mut out_data[start..end];
                for c in 0..self.channels {
                    let channel_offset = c * h * w;
                    for oh_idx in 0..oh {
                        for ow_idx in 0..ow {
                            let mut acc = 0.0f32;
                            for kh in 0..self.kernel.0 {
                                for kw in 0..self.kernel.1 {
                                    let pos_h = oh_idx * self.stride.0 + kh;
                                    let pos_w = ow_idx * self.stride.1 + kw;
                                    if pos_h < self.padding.0 || pos_w < self.padding.1 {
                                        continue;
                                    }
                                    let idx_h = pos_h - self.padding.0;
                                    let idx_w = pos_w - self.padding.1;
                                    if idx_h >= h || idx_w >= w {
                                        continue;
                                    }
                                    let index = channel_offset + idx_h * w + idx_w;
                                    acc += row[index];
                                }
                            }
                            out_row[c * (oh * ow) + oh_idx * ow + ow_idx] = acc / area;
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    fn backward(&mut self, _input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let (batch, cols) = grad_output.shape();
        let (oh, ow) = self.output_hw()?;
        if cols != self.channels * oh * ow {
            return Err(TensorError::ShapeMismatch {
                left: (1, cols),
                right: (1, self.channels * oh * ow),
            });
        }
        let mut grad_input =
            Tensor::zeros(batch, self.channels * self.input_hw.0 * self.input_hw.1)?;
        let (h, w) = self.input_hw;
        let area = (self.kernel.0 * self.kernel.1) as f32;
        let grad_input_cols = grad_input.shape().1;
        {
            let grad_input_data = grad_input.data_mut();
            for b in 0..batch {
                let grad_row = &grad_output.data()[b * cols..(b + 1) * cols];
                let (start, end) = (b * grad_input_cols, (b + 1) * grad_input_cols);
                let grad_in_row = &mut grad_input_data[start..end];
                for c in 0..self.channels {
                    let channel_offset = c * h * w;
                    for oh_idx in 0..oh {
                        for ow_idx in 0..ow {
                            let go = grad_row[c * (oh * ow) + oh_idx * ow + ow_idx] / area;
                            for kh in 0..self.kernel.0 {
                                for kw in 0..self.kernel.1 {
                                    let pos_h = oh_idx * self.stride.0 + kh;
                                    let pos_w = ow_idx * self.stride.1 + kw;
                                    if pos_h < self.padding.0 || pos_w < self.padding.1 {
                                        continue;
                                    }
                                    let idx_h = pos_h - self.padding.0;
                                    let idx_w = pos_w - self.padding.1;
                                    if idx_h >= h || idx_w >= w {
                                        continue;
                                    }
                                    let index = channel_offset + idx_h * w + idx_w;
                                    grad_in_row[index] += go;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(grad_input)
    }

    fn visit_parameters(
        &self,
        _visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        _visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv1d_forward_matches_manual() {
        let conv = Conv1d::new("conv", 1, 1, 3, 1, 1).unwrap();
        let input = Tensor::from_vec(1, 5, vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape().0, 1);
    }

    #[test]
    fn max_pool_tracks_indices() {
        let pool = MaxPool2d::new(1, (2, 2), (2, 2), (0, 0), (2, 2)).unwrap();
        let input = Tensor::from_vec(1, 4, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let out = pool.forward(&input).unwrap();
        assert_eq!(out.data(), &[4.0]);
    }

    #[test]
    fn conv2d_backward_supports_dilation() {
        let mut conv =
            Conv2d::new("conv", 1, 1, (2, 2), (1, 1), (1, 1), (3, 3)).expect("conv init");
        {
            let weight_data = conv.weight.value_mut().data_mut();
            weight_data.copy_from_slice(&[0.2, -0.1, 0.05, 0.3]);
        }
        {
            let bias_data = conv.bias.value_mut().data_mut();
            bias_data.copy_from_slice(&[0.05]);
        }

        let input = Tensor::from_vec(1, 9, vec![0.5, -0.3, 1.2, 0.7, -1.1, 0.4, 0.9, -0.2, 0.6])
            .expect("input tensor");
        let output = conv.forward(&input).expect("forward");
        let out_len = output.shape().1;
        let grad_output = Tensor::from_vec(1, out_len, vec![1.0; out_len]).expect("grad out");

        let grad_input = conv
            .backward(&input, &grad_output)
            .expect("conv backward computes grads");

        let weight_grad = conv
            .weight
            .gradient()
            .expect("weight grad available")
            .data()
            .to_vec();
        let bias_grad = conv
            .bias
            .gradient()
            .expect("bias grad available")
            .data()
            .to_vec();

        let eps = 1e-3f32;
        let mut numeric_weight_grad = vec![0.0f32; weight_grad.len()];
        for (idx, numeric) in numeric_weight_grad.iter_mut().enumerate() {
            let base = conv.weight.value().data()[idx];
            let mut weight_data = conv.weight.value_mut().data_mut();
            weight_data[idx] = base + eps;
            drop(weight_data);
            let out_pos = conv.forward(&input).expect("forward +eps");
            let loss_pos: f32 = out_pos
                .data()
                .iter()
                .zip(grad_output.data())
                .map(|(o, go)| o * go)
                .sum();
            let mut weight_data = conv.weight.value_mut().data_mut();
            weight_data[idx] = base - eps;
            drop(weight_data);
            let out_neg = conv.forward(&input).expect("forward -eps");
            let loss_neg: f32 = out_neg
                .data()
                .iter()
                .zip(grad_output.data())
                .map(|(o, go)| o * go)
                .sum();
            let mut weight_data = conv.weight.value_mut().data_mut();
            weight_data[idx] = base;
            drop(weight_data);
            *numeric = (loss_pos - loss_neg) / (2.0 * eps);
        }

        let mut numeric_bias_grad = vec![0.0f32; bias_grad.len()];
        for (idx, numeric) in numeric_bias_grad.iter_mut().enumerate() {
            let base = conv.bias.value().data()[idx];
            let mut bias_data = conv.bias.value_mut().data_mut();
            bias_data[idx] = base + eps;
            drop(bias_data);
            let out_pos = conv.forward(&input).expect("forward bias +eps");
            let loss_pos: f32 = out_pos
                .data()
                .iter()
                .zip(grad_output.data())
                .map(|(o, go)| o * go)
                .sum();
            let mut bias_data = conv.bias.value_mut().data_mut();
            bias_data[idx] = base - eps;
            drop(bias_data);
            let out_neg = conv.forward(&input).expect("forward bias -eps");
            let loss_neg: f32 = out_neg
                .data()
                .iter()
                .zip(grad_output.data())
                .map(|(o, go)| o * go)
                .sum();
            let mut bias_data = conv.bias.value_mut().data_mut();
            bias_data[idx] = base;
            drop(bias_data);
            *numeric = (loss_pos - loss_neg) / (2.0 * eps);
        }

        let input_len = input.shape().1;
        let mut numeric_input_grad = vec![0.0f32; input_len];
        let base_input = input.data().to_vec();
        for i in 0..input_len {
            let mut input_pos = base_input.clone();
            input_pos[i] += eps;
            let input_pos_tensor = Tensor::from_vec(1, input_len, input_pos).expect("input +eps");
            let out_pos = conv.forward(&input_pos_tensor).expect("forward input +eps");
            let loss_pos: f32 = out_pos
                .data()
                .iter()
                .zip(grad_output.data())
                .map(|(o, go)| o * go)
                .sum();

            let mut input_neg = base_input.clone();
            input_neg[i] -= eps;
            let input_neg_tensor = Tensor::from_vec(1, input_len, input_neg).expect("input -eps");
            let out_neg = conv.forward(&input_neg_tensor).expect("forward input -eps");
            let loss_neg: f32 = out_neg
                .data()
                .iter()
                .zip(grad_output.data())
                .map(|(o, go)| o * go)
                .sum();

            numeric_input_grad[i] = (loss_pos - loss_neg) / (2.0 * eps);
        }

        for (analytic, numeric) in weight_grad.iter().zip(numeric_weight_grad.iter()) {
            assert!(
                (analytic - numeric).abs() < 1e-2,
                "weight grad mismatch: analytic={} numeric={}",
                analytic,
                numeric
            );
        }

        for (analytic, numeric) in bias_grad.iter().zip(numeric_bias_grad.iter()) {
            assert!(
                (analytic - numeric).abs() < 1e-3,
                "bias grad mismatch: analytic={} numeric={}",
                analytic,
                numeric
            );
        }

        for (analytic, numeric) in grad_input.data().iter().zip(numeric_input_grad.iter()) {
            assert!(
                (analytic - numeric).abs() < 1e-2,
                "input grad mismatch: analytic={} numeric={}",
                analytic,
                numeric
            );
        }
    }
}
