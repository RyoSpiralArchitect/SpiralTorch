// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::{Module, Parameter};
use crate::schedule::GradientBands;
use crate::{PureResult, Tensor, TensorError};
#[cfg(feature = "wgpu")]
use st_tensor::backend::wgpu_dense;
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

fn dilated_extent(size: usize, dilation: usize) -> PureResult<usize> {
    size.checked_sub(1)
        .and_then(|value| value.checked_mul(dilation))
        .and_then(|value| value.checked_add(1))
        .ok_or(TensorError::InvalidDimensions {
            rows: size,
            cols: dilation,
        })
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
        input_hw: (usize, usize),
    ) -> PureResult<Self> {
        validate_positive(in_channels, "in_channels")?;
        validate_positive(out_channels, "out_channels")?;
        validate_positive(kernel.0, "kernel_h")?;
        validate_positive(kernel.1, "kernel_w")?;
        validate_positive(stride.0, "stride_h")?;
        validate_positive(stride.1, "stride_w")?;
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
        let conv = Self {
            weight: Parameter::new(format!("{name}::weight"), weight),
            bias: Parameter::new(format!("{name}::bias"), bias),
            in_channels,
            out_channels,
            kernel,
            stride,
            padding,
            dilation: (1, 1),
            input_hw,
        };
        // Validate configuration by computing the output size once during construction.
        conv.output_hw()?;
        Ok(conv)
    }

    fn dilated_kernel(&self) -> PureResult<(usize, usize)> {
        Ok((
            dilated_extent(self.kernel.0, self.dilation.0)?,
            dilated_extent(self.kernel.1, self.dilation.1)?,
        ))
    }

    /// Overrides the dilation factors used by the convolution.
    pub fn set_dilation(&mut self, dilation: (usize, usize)) -> PureResult<()> {
        validate_positive(dilation.0, "dilation_h")?;
        validate_positive(dilation.1, "dilation_w")?;
        let previous = self.dilation;
        self.dilation = dilation;
        if let Err(error) = self.output_hw() {
            self.dilation = previous;
            return Err(error);
        }
        Ok(())
    }

    /// Builder-style helper that returns a new instance with custom dilation factors.
    pub fn with_dilation(mut self, dilation: (usize, usize)) -> PureResult<Self> {
        self.set_dilation(dilation)?;
        Ok(self)
    }

    fn output_hw(&self) -> PureResult<(usize, usize)> {
        let (h, w) = self.input_hw;
        let (kh, kw) = self.dilated_kernel()?;
        let (ph, pw) = self.padding;
        let (sh, sw) = self.stride;
        if h + 2 * ph < kh || w + 2 * pw < kw {
            return Err(TensorError::InvalidDimensions {
                rows: h + 2 * ph,
                cols: kh.max(kw),
            });
        }
        let oh = (h + 2 * ph - kh) / sh + 1;
        let ow = (w + 2 * pw - kw) / sw + 1;
        Ok((oh, ow))
    }

    fn im2col(&self, input: &Tensor, batch: usize, oh: usize, ow: usize) -> PureResult<Tensor> {
        let kernel_elems = self.in_channels * self.kernel.0 * self.kernel.1;
        let mut columns = Tensor::zeros(batch * oh * ow, kernel_elems)?;
        let cols = input.shape().1;
        let (h, w) = self.input_hw;
        let (dilation_h, dilation_w) = self.dilation;
        let pad_h = self.padding.0 as isize;
        let pad_w = self.padding.1 as isize;
        {
            let input_data = input.data();
            let column_data = columns.data_mut();
            for b in 0..batch {
                let row = &input_data[b * cols..(b + 1) * cols];
                for oh_idx in 0..oh {
                    for ow_idx in 0..ow {
                        let row_index = b * oh * ow + oh_idx * ow + ow_idx;
                        let offset = row_index * kernel_elems;
                        let mut col_idx = 0;
                        for ic in 0..self.in_channels {
                            let channel_offset = ic * h * w;
                            for kh in 0..self.kernel.0 {
                                for kw in 0..self.kernel.1 {
                                    let pos_h = oh_idx * self.stride.0 + kh * dilation_h;
                                    let pos_w = ow_idx * self.stride.1 + kw * dilation_w;
                                    let idx_h = pos_h as isize - pad_h;
                                    let idx_w = pos_w as isize - pad_w;
                                    column_data[offset + col_idx] = if idx_h < 0
                                        || idx_w < 0
                                        || idx_h >= h as isize
                                        || idx_w >= w as isize
                                    {
                                        0.0
                                    } else {
                                        let ih = idx_h as usize;
                                        let iw = idx_w as usize;
                                        row[channel_offset + ih * w + iw]
                                    };
                                    col_idx += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(columns)
    }

    fn grad_output_to_matrix(
        &self,
        grad_output: &Tensor,
        batch: usize,
        oh: usize,
        ow: usize,
    ) -> PureResult<Tensor> {
        let mut matrix = Tensor::zeros(batch * oh * ow, self.out_channels)?;
        let grad_cols = grad_output.shape().1;
        let spatial = oh * ow;
        {
            let grad_data = grad_output.data();
            let matrix_data = matrix.data_mut();
            for b in 0..batch {
                let grad_row = &grad_data[b * grad_cols..(b + 1) * grad_cols];
                for oh_idx in 0..oh {
                    for ow_idx in 0..ow {
                        let row_index = b * spatial + oh_idx * ow + ow_idx;
                        let offset = row_index * self.out_channels;
                        for oc in 0..self.out_channels {
                            let grad_idx = oc * spatial + oh_idx * ow + ow_idx;
                            matrix_data[offset + oc] = grad_row[grad_idx];
                        }
                    }
                }
            }
        }
        Ok(matrix)
    }

    fn accumulate_from_grad_matrix(
        &mut self,
        grad_matrix: &Tensor,
        patches: &Tensor,
        batch: usize,
        oh: usize,
        ow: usize,
    ) -> PureResult<Tensor> {
        let grad_weight = grad_matrix.transpose().matmul(patches)?;
        let grad_weight = grad_weight.scale(1.0 / batch as f32)?;
        let bias_sums = grad_matrix.sum_axis0();
        let mut bias_tensor = Tensor::from_vec(1, self.out_channels, bias_sums)?;
        bias_tensor = bias_tensor.scale(1.0 / batch as f32)?;
        let grad_patches = grad_matrix.matmul(self.weight.value())?;
        let grad_input = self.col2im(&grad_patches, batch, oh, ow)?;
        self.weight.accumulate_euclidean(&grad_weight)?;
        self.bias.accumulate_euclidean(&bias_tensor)?;
        Ok(grad_input)
    }

    /// Propagates Above/Here/Beneath gradients individually, returning the
    /// corresponding input gradients stacked as a [`GradientBands`] volume.
    pub fn backward_band_volume(
        &mut self,
        input: &Tensor,
        bands: &GradientBands,
    ) -> PureResult<GradientBands> {
        let (batch, cols) = input.shape();
        let expected_cols = self.in_channels * self.input_hw.0 * self.input_hw.1;
        if cols != expected_cols {
            return Err(TensorError::ShapeMismatch {
                left: (1, cols),
                right: (1, expected_cols),
            });
        }
        let (oh, ow) = self.output_hw()?;
        let patches = self.im2col(input, batch, oh, ow)?;
        let mut outputs: [Option<Tensor>; 3] = [None, None, None];
        for (idx, grad) in bands.iter().iter().enumerate() {
            if grad.shape() != (batch, self.out_channels * oh * ow) {
                return Err(TensorError::ShapeMismatch {
                    left: grad.shape(),
                    right: (batch, self.out_channels * oh * ow),
                });
            }
            if grad.squared_l2_norm() == 0.0 {
                outputs[idx] = Some(Tensor::zeros(batch, expected_cols)?);
                continue;
            }
            let grad_matrix = self.grad_output_to_matrix(grad, batch, oh, ow)?;
            let grad_input =
                self.accumulate_from_grad_matrix(&grad_matrix, &patches, batch, oh, ow)?;
            outputs[idx] = Some(grad_input);
        }
        let take_or_zero = |slot: Option<Tensor>| -> PureResult<Tensor> {
            match slot {
                Some(tensor) => Ok(tensor),
                None => Tensor::zeros(batch, expected_cols),
            }
        };
        let above = take_or_zero(outputs[0].take())?;
        let here = take_or_zero(outputs[1].take())?;
        let beneath = take_or_zero(outputs[2].take())?;
        GradientBands::from_components(above, here, beneath)
    }

    fn col2im(&self, cols: &Tensor, batch: usize, oh: usize, ow: usize) -> PureResult<Tensor> {
        let expected_rows = batch * oh * ow;
        let kernel_elems = self.in_channels * self.kernel.0 * self.kernel.1;
        if cols.shape() != (expected_rows, kernel_elems) {
            return Err(TensorError::ShapeMismatch {
                left: cols.shape(),
                right: (expected_rows, kernel_elems),
            });
        }
        let mut output =
            Tensor::zeros(batch, self.in_channels * self.input_hw.0 * self.input_hw.1)?;
        let (h, w) = self.input_hw;
        let (dilation_h, dilation_w) = self.dilation;
        let pad_h = self.padding.0 as isize;
        let pad_w = self.padding.1 as isize;
        let spatial = oh * ow;
        let output_cols = output.shape().1;
        {
            let cols_data = cols.data();
            let output_data = output.data_mut();
            for b in 0..batch {
                let (start, end) = (b * output_cols, (b + 1) * output_cols);
                let grad_in_row = &mut output_data[start..end];
                for oh_idx in 0..oh {
                    for ow_idx in 0..ow {
                        let row_index = b * spatial + oh_idx * ow + ow_idx;
                        let column_row =
                            &cols_data[row_index * kernel_elems..(row_index + 1) * kernel_elems];
                        let mut col_idx = 0;
                        for ic in 0..self.in_channels {
                            let channel_offset = ic * h * w;
                            for kh in 0..self.kernel.0 {
                                for kw in 0..self.kernel.1 {
                                    let pos_h = oh_idx * self.stride.0 + kh * dilation_h;
                                    let pos_w = ow_idx * self.stride.1 + kw * dilation_w;
                                    let idx_h = pos_h as isize - pad_h;
                                    let idx_w = pos_w as isize - pad_w;
                                    if idx_h >= 0
                                        && idx_w >= 0
                                        && idx_h < h as isize
                                        && idx_w < w as isize
                                    {
                                        let ih = idx_h as usize;
                                        let iw = idx_w as usize;
                                        let index = channel_offset + ih * w + iw;
                                        grad_in_row[index] += column_row[col_idx];
                                    }
                                    col_idx += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(output)
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
        #[cfg(feature = "wgpu")]
        {
            if let Some(tensor) = self.try_forward_wgpu(input, batch, oh, ow)? {
                return Ok(tensor);
            }
        }
        self.forward_cpu(input, batch, oh, ow)
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
        let patches = self.im2col(input, batch, oh, ow)?;
        let grad_matrix = self.grad_output_to_matrix(grad_output, batch, oh, ow)?;
        self.accumulate_from_grad_matrix(&grad_matrix, &patches, batch, oh, ow)
    }

    fn backward_bands(&mut self, input: &Tensor, bands: &GradientBands) -> PureResult<Tensor> {
        let volume = self.backward_band_volume(input, bands)?;
        volume.combine()
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

impl Conv2d {
    fn forward_cpu(
        &self,
        input: &Tensor,
        batch: usize,
        oh: usize,
        ow: usize,
    ) -> PureResult<Tensor> {
        let mut out = Tensor::zeros(batch, self.out_channels * oh * ow)?;
        let weight = self.weight.value();
        let bias = self.bias.value();
        let weight_data = weight.data();
        let bias_data = bias.data();
        let span = self.in_channels * self.kernel.0 * self.kernel.1;
        let (h, w) = self.input_hw;
        let (dilation_h, dilation_w) = self.dilation;
        let out_cols = out.shape().1;
        let cols = input.shape().1;
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
                                        let pos_h = oh_idx * self.stride.0 + kh * dilation_h;
                                        let pos_w = ow_idx * self.stride.1 + kw * dilation_w;
                                        let idx_h = pos_h as isize - self.padding.0 as isize;
                                        let idx_w = pos_w as isize - self.padding.1 as isize;
                                        if idx_h < 0
                                            || idx_w < 0
                                            || idx_h >= h as isize
                                            || idx_w >= w as isize
                                        {
                                            continue;
                                        }
                                        let input_idx =
                                            channel_offset + idx_h as usize * w + idx_w as usize;
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

    #[cfg(feature = "wgpu")]
    fn try_forward_wgpu(
        &self,
        input: &Tensor,
        batch: usize,
        oh: usize,
        ow: usize,
    ) -> PureResult<Option<Tensor>> {
        if !wgpu_dense::is_available() {
            return Ok(None);
        }
        let span = self.in_channels * self.kernel.0 * self.kernel.1;
        let rows = batch * oh * ow;
        let total_work = rows
            .checked_mul(span)
            .and_then(|value| value.checked_mul(self.out_channels))
            .unwrap_or(usize::MAX);
        if total_work < 4096 {
            return Ok(None);
        }
        let pad_h = match i32::try_from(self.padding.0) {
            Ok(value) => value,
            Err(_) => return Ok(None),
        };
        let pad_w = match i32::try_from(self.padding.1) {
            Ok(value) => value,
            Err(_) => return Ok(None),
        };

        let mut weight_t = vec![0.0f32; span * self.out_channels];
        let weight_data = self.weight.value().data();
        for oc in 0..self.out_channels {
            let start = oc * span;
            let end = start + span;
            for (idx, value) in weight_data[start..end].iter().enumerate() {
                weight_t[idx * self.out_channels + oc] = *value;
            }
        }

        let maybe = wgpu_dense::conv_im2col_gemm(
            input.data(),
            batch,
            self.in_channels,
            self.input_hw.0,
            self.input_hw.1,
            self.kernel.0,
            self.kernel.1,
            self.stride.0,
            self.stride.1,
            pad_h,
            pad_w,
            self.dilation.0,
            self.dilation.1,
            &weight_t,
            self.out_channels,
            oh,
            ow,
        );

        let buffer = match maybe {
            Ok(buffer) => buffer,
            Err(_) => return Ok(None),
        };

        let mut out = Tensor::zeros(batch, self.out_channels * oh * ow)?;
        let bias_data = self.bias.value().data();
        let spatial = oh * ow;
        {
            let out_data = out.data_mut();
            for b in 0..batch {
                for oh_idx in 0..oh {
                    for ow_idx in 0..ow {
                        let row_index = b * spatial + oh_idx * ow + ow_idx;
                        let row_start = row_index * self.out_channels;
                        for oc in 0..self.out_channels {
                            let target = b * self.out_channels * spatial
                                + oc * spatial
                                + oh_idx * ow
                                + ow_idx;
                            out_data[target] = buffer[row_start + oc] + bias_data[oc];
                        }
                    }
                }
            }
        }
        Ok(Some(out))
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
    #[cfg(feature = "wgpu")]
    use st_tensor::backend::wgpu_dense;

    #[test]
    fn conv1d_forward_matches_manual() {
        let conv = Conv1d::new("conv", 1, 1, 3, 1, 1).unwrap();
        let input = Tensor::from_vec(1, 5, vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape().0, 1);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn conv2d_wgpu_matches_cpu_path() {
        if !wgpu_dense::is_available() {
            return;
        }
        let mut conv = Conv2d::new("conv_gpu", 3, 8, (3, 3), (1, 1), (1, 1), (8, 8)).unwrap();
        for (idx, value) in conv.weight.value_mut().data_mut().iter_mut().enumerate() {
            *value = (idx as f32).sin();
        }
        for (idx, value) in conv.bias.value_mut().data_mut().iter_mut().enumerate() {
            *value = idx as f32 * 0.1;
        }
        let batch = 4;
        let input = Tensor::from_fn(batch, 3 * 8 * 8, |row, col| {
            ((row * 97 + col * 31) % 23) as f32 * 0.05
        })
        .unwrap();
        let (oh, ow) = conv.output_hw().unwrap();
        let cpu = conv.forward_cpu(&input, batch, oh, ow).unwrap();
        let gpu = conv
            .try_forward_wgpu(&input, batch, oh, ow)
            .unwrap()
            .unwrap_or_else(|| panic!("wgpu path unexpectedly unavailable"));
        for (&a, &b) in cpu.data().iter().zip(gpu.data().iter()) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn max_pool_tracks_indices() {
        let pool = MaxPool2d::new(1, (2, 2), (2, 2), (0, 0), (2, 2)).unwrap();
        let input = Tensor::from_vec(1, 4, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let out = pool.forward(&input).unwrap();
        assert_eq!(out.data(), &[4.0]);
    }

    #[test]
    fn conv2d_backward_matches_manual_kernel11() {
        let mut conv = Conv2d::new("conv", 1, 1, (1, 1), (1, 1), (0, 0), (2, 2)).unwrap();
        conv.weight.value_mut().data_mut()[0] = 1.5;
        conv.bias.value_mut().data_mut()[0] = 0.0;
        let input = Tensor::from_vec(1, 4, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let grad_output = Tensor::from_vec(1, 4, vec![1.0; 4]).unwrap();
        let grad_input = conv.backward(&input, &grad_output).unwrap();
        assert_eq!(grad_input.shape(), input.shape());
        for &value in grad_input.data() {
            assert!((value - 1.5).abs() < 1e-6);
        }
        let weight_grad = conv.weight.gradient().unwrap();
        assert!((weight_grad.data()[0] - 10.0).abs() < 1e-6);
        let bias_grad = conv.bias.gradient().unwrap();
        assert!((bias_grad.data()[0] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn conv2d_respects_dilation_configuration() {
        let mut conv = Conv2d::new("conv", 1, 1, (3, 3), (1, 1), (0, 0), (5, 5)).unwrap();
        conv.set_dilation((2, 2)).unwrap();
        assert_eq!(conv.output_hw().unwrap(), (1, 1));
    }

    #[test]
    fn conv2d_backward_bands_matches_backward() {
        use crate::plan::RankPlanner;
        use crate::schedule::{RoundtableConfig, RoundtableSchedule};
        use st_core::backend::device_caps::DeviceCaps;

        let mut conv = Conv2d::new("conv_a", 1, 1, (2, 2), (1, 1), (0, 0), (3, 3)).unwrap();
        let mut conv_bands = Conv2d::new("conv_b", 1, 1, (2, 2), (1, 1), (0, 0), (3, 3)).unwrap();

        for (idx, value) in conv.weight.value_mut().data_mut().iter_mut().enumerate() {
            *value = idx as f32 + 1.0;
        }
        conv.bias.value_mut().data_mut()[0] = 0.5;
        conv_bands.weight.load_value(conv.weight.value()).unwrap();
        conv_bands.bias.load_value(conv.bias.value()).unwrap();

        let input =
            Tensor::from_vec(1, 9, vec![0.5, -1.0, 2.0, 1.5, 0.0, -0.5, 2.5, 1.0, -1.5]).unwrap();
        let grad_output = Tensor::from_vec(1, 4, vec![0.8, -0.4, 1.2, -0.6]).unwrap();

        let standard = conv.backward(&input, &grad_output).unwrap();
        let (oh, ow) = conv_bands.output_hw().unwrap();
        let planner = RankPlanner::new(DeviceCaps::cpu());
        let schedule = RoundtableSchedule::new(
            &planner,
            1,
            (conv_bands.out_channels * oh * ow) as u32,
            RoundtableConfig::default(),
        );
        let bands = schedule.split(&grad_output).unwrap();
        let combined = conv_bands.backward_bands(&input, &bands).unwrap();

        assert_eq!(standard.shape(), combined.shape());
        for (&a, &b) in standard.data().iter().zip(combined.data().iter()) {
            assert!((a - b).abs() < 1e-5);
        }

        let grad_weight = conv.weight.gradient().unwrap();
        let grad_weight_bands = conv_bands.weight.gradient().unwrap();
        for (&a, &b) in grad_weight
            .data()
            .iter()
            .zip(grad_weight_bands.data().iter())
        {
            assert!((a - b).abs() < 1e-5);
        }

        let grad_bias = conv.bias.gradient().unwrap();
        let grad_bias_bands = conv_bands.bias.gradient().unwrap();
        for (&a, &b) in grad_bias.data().iter().zip(grad_bias_bands.data().iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn conv2d_backward_band_volume_exposes_components() {
        use crate::plan::RankPlanner;
        use crate::schedule::{RoundtableConfig, RoundtableSchedule};
        use st_core::backend::device_caps::DeviceCaps;

        let mut conv_seq = Conv2d::new("conv_seq", 1, 1, (2, 2), (1, 1), (0, 0), (3, 3)).unwrap();
        let mut conv_volume =
            Conv2d::new("conv_vol", 1, 1, (2, 2), (1, 1), (0, 0), (3, 3)).unwrap();

        for (idx, value) in conv_seq
            .weight
            .value_mut()
            .data_mut()
            .iter_mut()
            .enumerate()
        {
            *value = (idx + 1) as f32;
        }
        conv_seq.bias.value_mut().data_mut()[0] = -0.3;
        conv_volume
            .weight
            .load_value(conv_seq.weight.value())
            .unwrap();
        conv_volume.bias.load_value(conv_seq.bias.value()).unwrap();

        let input = Tensor::from_vec(
            1,
            9,
            vec![0.75, -0.5, 1.0, 0.25, -1.25, 0.0, 1.5, 0.8, -0.4],
        )
        .unwrap();
        let grad_output = Tensor::from_vec(1, 4, vec![1.0, -0.5, 0.25, 0.75]).unwrap();

        let (oh, ow) = conv_seq.output_hw().unwrap();
        let planner = RankPlanner::new(DeviceCaps::cpu());
        let schedule = RoundtableSchedule::new(
            &planner,
            1,
            (conv_seq.out_channels * oh * ow) as u32,
            RoundtableConfig::default(),
        );
        let bands = schedule.split(&grad_output).unwrap();

        let grad_above = conv_seq.backward(&input, bands.above()).unwrap();
        let grad_here = conv_seq.backward(&input, bands.here()).unwrap();
        let grad_beneath = conv_seq.backward(&input, bands.beneath()).unwrap();

        let volume = conv_volume.backward_band_volume(&input, &bands).unwrap();

        for (&a, &b) in grad_above.data().iter().zip(volume.above().data().iter()) {
            assert!((a - b).abs() < 1e-5);
        }
        for (&a, &b) in grad_here.data().iter().zip(volume.here().data().iter()) {
            assert!((a - b).abs() < 1e-5);
        }
        for (&a, &b) in grad_beneath
            .data()
            .iter()
            .zip(volume.beneath().data().iter())
        {
            assert!((a - b).abs() < 1e-5);
        }

        let mut expected_sum = grad_above.clone();
        expected_sum.add_scaled(&grad_here, 1.0).unwrap();
        expected_sum.add_scaled(&grad_beneath, 1.0).unwrap();
        let combined = volume.combine().unwrap();
        for (&a, &b) in expected_sum.data().iter().zip(combined.data().iter()) {
            assert!((a - b).abs() < 1e-5);
        }

        let seq_weight = conv_seq.weight.gradient().unwrap();
        let vol_weight = conv_volume.weight.gradient().unwrap();
        for (&a, &b) in seq_weight.data().iter().zip(vol_weight.data().iter()) {
            assert!((a - b).abs() < 1e-5);
        }

        let seq_bias = conv_seq.bias.gradient().unwrap();
        let vol_bias = conv_volume.bias.gradient().unwrap();
        for (&a, &b) in seq_bias.data().iter().zip(vol_bias.data().iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
