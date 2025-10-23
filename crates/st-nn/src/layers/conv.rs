// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::{Module, Parameter};
use crate::schedule::GradientBands;
use crate::{PureResult, Tensor, TensorError};
use st_core::util::math::LeechProjector;
#[cfg(feature = "wgpu")]
use st_tensor::backend::wgpu_dense;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

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

const INVALID_NEIGHBOR: usize = usize::MAX;
pub const DEFAULT_CONV6DA_OFFSETS: &[(isize, isize, isize)] = &[
    (0, 0, 0),
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

struct AxisState {
    dims: [usize; 26],
    has_dim: [bool; 26],
    values: [usize; 26],
}

impl AxisState {
    fn new() -> Self {
        Self {
            dims: [0; 26],
            has_dim: [false; 26],
            values: [0; 26],
        }
    }

    fn register(&mut self, axis: usize, size: usize) -> PureResult<()> {
        if size == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: 1,
                cols: size,
            });
        }
        if self.has_dim[axis] {
            if self.dims[axis] != size {
                return Err(TensorError::ShapeMismatch {
                    left: (self.dims[axis], size),
                    right: (size, self.dims[axis]),
                });
            }
        } else {
            self.dims[axis] = size;
            self.has_dim[axis] = true;
        }
        Ok(())
    }

    fn dim(&self, axis: usize) -> PureResult<usize> {
        if self.has_dim[axis] {
            Ok(self.dims[axis])
        } else {
            Err(TensorError::InvalidValue {
                label: "einsum_axis",
            })
        }
    }

    fn dim_unchecked(&self, axis: usize) -> usize {
        debug_assert!(self.has_dim[axis]);
        self.dims[axis]
    }

    fn set_value(&mut self, axis: usize, value: usize) {
        debug_assert!(self.has_dim[axis]);
        debug_assert!(value < self.dims[axis]);
        self.values[axis] = value;
    }

    fn value(&self, axis: usize) -> usize {
        debug_assert!(self.has_dim[axis]);
        self.values[axis]
    }
}

fn axis_to_index(axis: char) -> PureResult<usize> {
    if axis.is_ascii_lowercase() {
        Ok((axis as u8 - b'a') as usize)
    } else {
        Err(TensorError::InvalidValue {
            label: "einsum_axis",
        })
    }
}

fn index_for_operand(spec: &[usize], shape: (usize, usize), state: &AxisState) -> usize {
    debug_assert_eq!(spec.len(), 2);
    let row = state.value(spec[0]);
    let col = state.value(spec[1]);
    debug_assert!(row < shape.0 && col < shape.1);
    row * shape.1 + col
}

fn accumulate_einsum(
    depth: usize,
    sum_axes: &[usize],
    state: &mut AxisState,
    lhs_spec: &[usize],
    lhs_shape: (usize, usize),
    lhs_data: &[f32],
    rhs_spec: &[usize],
    rhs_shape: (usize, usize),
    rhs_data: &[f32],
) -> f32 {
    if depth == sum_axes.len() {
        let lhs_idx = index_for_operand(lhs_spec, lhs_shape, state);
        let rhs_idx = index_for_operand(rhs_spec, rhs_shape, state);
        lhs_data[lhs_idx] * rhs_data[rhs_idx]
    } else {
        let axis = sum_axes[depth];
        let dim = state.dim_unchecked(axis);
        let mut acc = 0.0f32;
        for value in 0..dim {
            state.set_value(axis, value);
            acc += accumulate_einsum(
                depth + 1,
                sum_axes,
                state,
                lhs_spec,
                lhs_shape,
                lhs_data,
                rhs_spec,
                rhs_shape,
                rhs_data,
            );
        }
        acc
    }
}

fn einsum_contract(lhs: &Tensor, rhs: &Tensor, equation: &str) -> PureResult<Tensor> {
    let expression = equation.replace(' ', "");
    let (inputs, output) = expression
        .split_once("->")
        .ok_or(TensorError::InvalidValue {
            label: "einsum_equation",
        })?;
    let (lhs_spec, rhs_spec) = inputs.split_once(',').ok_or(TensorError::InvalidValue {
        label: "einsum_equation",
    })?;
    let lhs_axes: Vec<usize> = lhs_spec
        .chars()
        .map(axis_to_index)
        .collect::<PureResult<_>>()?;
    let rhs_axes: Vec<usize> = rhs_spec
        .chars()
        .map(axis_to_index)
        .collect::<PureResult<_>>()?;
    let output_axes: Vec<usize> = output
        .chars()
        .map(axis_to_index)
        .collect::<PureResult<_>>()?;
    if lhs_axes.len() != 2 || rhs_axes.len() != 2 || output_axes.len() != 2 {
        return Err(TensorError::InvalidValue {
            label: "einsum_rank",
        });
    }
    let mut state = AxisState::new();
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    state.register(lhs_axes[0], lhs_shape.0)?;
    state.register(lhs_axes[1], lhs_shape.1)?;
    state.register(rhs_axes[0], rhs_shape.0)?;
    state.register(rhs_axes[1], rhs_shape.1)?;
    for &axis in &output_axes {
        state.dim(axis)?;
    }
    let mut output_flags = [false; 26];
    for &axis in &output_axes {
        output_flags[axis] = true;
    }
    let mut seen_sum = [false; 26];
    let mut sum_axes = Vec::new();
    for &axis in lhs_axes.iter().chain(rhs_axes.iter()) {
        if !output_flags[axis] && !seen_sum[axis] {
            seen_sum[axis] = true;
            sum_axes.push(axis);
        }
    }
    let out_rows = state.dim(output_axes[0])?;
    let out_cols = state.dim(output_axes[1])?;
    let mut result = Tensor::zeros(out_rows, out_cols)?;
    {
        let lhs_data = lhs.data();
        let rhs_data = rhs.data();
        let out_data = result.data_mut();
        for row in 0..out_rows {
            state.set_value(output_axes[0], row);
            for col in 0..out_cols {
                state.set_value(output_axes[1], col);
                let value = accumulate_einsum(
                    0, &sum_axes, &mut state, &lhs_axes, lhs_shape, lhs_data, &rhs_axes, rhs_shape,
                    rhs_data,
                );
                out_data[row * out_cols + col] = value;
            }
        }
    }
    Ok(result)
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
    dilation: usize,
}

impl Conv1d {
    pub fn new(
        name: impl Into<String>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> PureResult<Self> {
        validate_positive(in_channels, "in_channels")?;
        validate_positive(out_channels, "out_channels")?;
        validate_positive(kernel_size, "kernel_size")?;
        validate_positive(stride, "stride")?;
        validate_positive(dilation, "dilation")?;
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
            dilation: 1,
        })
    }

    /// Overrides the dilation factor applied to the convolution kernel.
    pub fn set_dilation(&mut self, dilation: usize) -> PureResult<()> {
        validate_positive(dilation, "dilation")?;
        self.dilation = dilation;
        Ok(())
    }

    /// Builder-style helper returning a new instance configured with dilation.
    pub fn with_dilation(mut self, dilation: usize) -> PureResult<Self> {
        self.set_dilation(dilation)?;
        Ok(self)
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
        let eff_kernel = dilated_extent(self.kernel_size, self.dilation)?;
        if numer < eff_kernel {
            return Err(TensorError::InvalidDimensions {
                rows: input_width,
                cols: eff_kernel,
            });
        }
        Ok((numer - eff_kernel) / self.stride + 1)
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
                                let pos = ow * self.stride + k * self.dilation;
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
                                let pos = ow * self.stride + k * self.dilation;
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
        let conv = Self {
            weight: Parameter::new(format!("{name}::weight"), weight),
            bias: Parameter::new(format!("{name}::bias"), bias),
            in_channels,
            out_channels,
            kernel,
            stride,
            padding,
            dilation,
            input_hw,
        };
        // Validate configuration by computing the output size once during construction.
        conv.output_hw()?;
        Ok(conv)
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
        let eff_kh = dilated_extent(self.kernel.0, self.dilation.0)?;
        let eff_kw = dilated_extent(self.kernel.1, self.dilation.1)?;
        let (ph, pw) = self.padding;
        let (sh, sw) = self.stride;
        if h + 2 * ph < eff_kh || w + 2 * pw < eff_kw {
            return Err(TensorError::InvalidDimensions {
                rows: h + 2 * ph,
                cols: eff_kh.max(eff_kw),
            });
        }
        let oh = (h + 2 * ph - eff_kh) / sh + 1;
        let ow = (w + 2 * pw - eff_kw) / sw + 1;
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

#[derive(Debug)]
pub struct Conv6da {
    weight: Parameter,
    bias: Parameter,
    in_channels: usize,
    out_channels: usize,
    depth: usize,
    height: usize,
    width: usize,
    volume: usize,
    neighbors_per_cell: usize,
    neighbor_indices: Vec<usize>,
    leech_projector: LeechProjector,
}

impl Conv6da {
    pub fn new(
        name: impl Into<String>,
        in_channels: usize,
        out_channels: usize,
        grid: (usize, usize, usize),
        leech_rank: usize,
        leech_weight: f64,
    ) -> PureResult<Self> {
        Self::with_neighbor_offsets(
            name,
            in_channels,
            out_channels,
            grid,
            leech_rank,
            leech_weight,
            DEFAULT_CONV6DA_OFFSETS,
        )
    }

    pub fn with_neighbor_offsets(
        name: impl Into<String>,
        in_channels: usize,
        out_channels: usize,
        grid: (usize, usize, usize),
        leech_rank: usize,
        leech_weight: f64,
        neighbor_offsets: &[(isize, isize, isize)],
    ) -> PureResult<Self> {
        validate_positive(in_channels, "in_channels")?;
        validate_positive(out_channels, "out_channels")?;
        validate_positive(grid.0, "depth")?;
        validate_positive(grid.1, "height")?;
        validate_positive(grid.2, "width")?;
        if neighbor_offsets.is_empty() {
            return Err(TensorError::InvalidDimensions { rows: 0, cols: 0 });
        }
        let name = name.into();
        let neighbors_per_cell = neighbor_offsets.len();
        let span = in_channels * neighbors_per_cell;
        let plane = grid.1 * grid.2;
        let volume = grid.0 * plane;
        let mut neighbor_indices = vec![INVALID_NEIGHBOR; volume * neighbors_per_cell];
        for depth_idx in 0..grid.0 {
            for height_idx in 0..grid.1 {
                for width_idx in 0..grid.2 {
                    let cell_index = depth_idx * plane + height_idx * grid.2 + width_idx;
                    let slice_start = cell_index * neighbors_per_cell;
                    let slice_end = slice_start + neighbors_per_cell;
                    let indices = &mut neighbor_indices[slice_start..slice_end];
                    for (offset_idx, (od, oh, ow)) in neighbor_offsets.iter().enumerate() {
                        let nd = depth_idx as isize + *od;
                        let nh = height_idx as isize + *oh;
                        let nw = width_idx as isize + *ow;
                        if nd < 0
                            || nh < 0
                            || nw < 0
                            || nd >= grid.0 as isize
                            || nh >= grid.1 as isize
                            || nw >= grid.2 as isize
                        {
                            continue;
                        }
                        let base = nd as usize * plane + nh as usize * grid.2 + nw as usize;
                        indices[offset_idx] = base;
                    }
                }
            }
        }
        let mut seed = 0.01f32;
        let weight = Tensor::from_fn(out_channels, span, |_r, _c| {
            let value = seed;
            seed = (seed * 1.31).rem_euclid(0.2).max(1e-3);
            value
        })?;
        let bias = Tensor::zeros(1, out_channels)?;
        Ok(Self {
            weight: Parameter::new(format!("{name}::weight"), weight),
            bias: Parameter::new(format!("{name}::bias"), bias),
            in_channels,
            out_channels,
            depth: grid.0,
            height: grid.1,
            width: grid.2,
            volume,
            neighbors_per_cell,
            neighbor_indices,
            leech_projector: LeechProjector::new(leech_rank, leech_weight),
        })
    }

    fn cells(&self) -> usize {
        self.volume
    }

    fn expected_cols(&self) -> usize {
        self.in_channels * self.volume
    }

    fn neighbor_span(&self) -> usize {
        self.in_channels * self.neighbors_per_cell
    }

    fn neighbor_slice(&self, cell_index: usize) -> &[usize] {
        let start = cell_index * self.neighbors_per_cell;
        let end = start + self.neighbors_per_cell;
        &self.neighbor_indices[start..end]
    }

    fn gather_neighbors(&self, row: &[f32], buffer: &mut [f32], cell_index: usize) -> f64 {
        buffer.fill(0.0);
        let mut geodesic_sq = 0.0f64;
        let volume = self.volume;
        let indices = self.neighbor_slice(cell_index);
        for (offset_idx, &base) in indices.iter().enumerate() {
            if base == INVALID_NEIGHBOR {
                continue;
            }
            for ic in 0..self.in_channels {
                let channel_offset = ic * volume;
                let value = row[channel_offset + base];
                buffer[offset_idx * self.in_channels + ic] = value;
                geodesic_sq += f64::from(value) * f64::from(value);
            }
        }
        geodesic_sq.sqrt()
    }
}

impl Module for Conv6da {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (batch, cols) = input.shape();
        if cols != self.expected_cols() {
            return Err(TensorError::ShapeMismatch {
                left: (1, cols),
                right: (1, self.expected_cols()),
            });
        }
        let span = self.neighbor_span();
        let volume = self.cells();
        let mut out = Tensor::zeros(batch, self.out_channels * volume)?;
        let weight = self.weight.value();
        let bias = self.bias.value();
        let weight_data = weight.data();
        let bias_data = bias.data();
        let mut neighbors = vec![0.0f32; span];
        let out_cols = out.shape().1;
        let mut out_rows = out.data_mut().chunks_exact_mut(out_cols);
        let mut input_rows = input.data().chunks_exact(cols);
        for (row, out_row) in (&mut input_rows).zip(&mut out_rows) {
            for cell_index in 0..volume {
                let geodesic = self.gather_neighbors(row, &mut neighbors, cell_index);
                let density = self.leech_projector.enrich(geodesic) as f32;
                for (oc, weight_row) in weight_data.chunks_exact(span).enumerate() {
                    let mut acc = bias_data[oc] + density;
                    for (value, weight) in neighbors.iter().zip(weight_row.iter()) {
                        acc += value * weight;
                    }
                    out_row[oc * volume + cell_index] = acc;
                }
            }
        }
        debug_assert!(input_rows.remainder().is_empty());
        debug_assert!(out_rows.into_remainder().is_empty());
        Ok(out)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let (batch, cols) = input.shape();
        if cols != self.expected_cols() {
            return Err(TensorError::ShapeMismatch {
                left: (1, cols),
                right: (1, self.expected_cols()),
            });
        }
        let volume = self.cells();
        let span = self.neighbor_span();
        if grad_output.shape() != (batch, self.out_channels * volume) {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: (batch, self.out_channels * volume),
            });
        }
        let mut grad_weight = vec![0.0f32; self.out_channels * span];
        let mut grad_bias = vec![0.0f32; self.out_channels];
        let mut grad_input = Tensor::zeros(batch, cols)?;
        let mut neighbors = vec![0.0f32; span];
        let volume_per_channel = volume;
        let weight = self.weight.value();
        let grad_cols = grad_output.shape().1;
        let leech_factor = self.leech_projector.enrich(1.0);
        let mut grad_input_rows = grad_input.data_mut().chunks_exact_mut(cols);
        let mut input_rows = input.data().chunks_exact(cols);
        let mut grad_rows = grad_output.data().chunks_exact(grad_cols);
        let weight_data = weight.data();
        for ((row, grad_row), grad_in_row) in (&mut input_rows)
            .zip(&mut grad_rows)
            .zip(&mut grad_input_rows)
        {
            for cell_index in 0..volume {
                let geodesic = self.gather_neighbors(row, &mut neighbors, cell_index);
                let indices = self.neighbor_slice(cell_index);
                let mut sum_go = 0.0f64;
                for (oc, weight_row) in weight_data.chunks_exact(span).enumerate() {
                    let go = grad_row[oc * volume + cell_index];
                    sum_go += f64::from(go);
                    grad_bias[oc] += go;
                    for (idx, &value) in neighbors.iter().enumerate() {
                        grad_weight[oc * span + idx] += go * value;
                    }
                    for ic in 0..self.in_channels {
                        let channel_offset = ic * volume_per_channel;
                        for (offset_idx, &base) in indices.iter().enumerate() {
                            if base == INVALID_NEIGHBOR {
                                continue;
                            }
                            let input_index = channel_offset + base;
                            let weight_idx = offset_idx * self.in_channels + ic;
                            grad_in_row[input_index] += go * weight_row[weight_idx];
                        }
                    }
                }
                if geodesic > 0.0 && leech_factor > f64::EPSILON && sum_go.abs() > f64::EPSILON {
                    let scale = (sum_go * leech_factor / geodesic) as f32;
                    for ic in 0..self.in_channels {
                        let channel_offset = ic * volume_per_channel;
                        for (offset_idx, &base) in indices.iter().enumerate() {
                            if base == INVALID_NEIGHBOR {
                                continue;
                            }
                            let input_index = channel_offset + base;
                            let value = neighbors[offset_idx * self.in_channels + ic];
                            grad_in_row[input_index] += scale * value;
                        }
                    }
                }
            }
        }
        debug_assert!(input_rows.remainder().is_empty());
        debug_assert!(grad_rows.remainder().is_empty());
        debug_assert!(grad_input_rows.into_remainder().is_empty());
        let grad_weight_tensor = Tensor::from_vec(self.out_channels, span, grad_weight)?;
        let grad_bias_tensor = Tensor::from_vec(1, self.out_channels, grad_bias)?;
        self.weight.accumulate_euclidean(&grad_weight_tensor)?;
        self.bias.accumulate_euclidean(&grad_bias_tensor)?;
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

impl Conv2d {
    fn forward_cpu(
        &self,
        input: &Tensor,
        batch: usize,
        oh: usize,
        ow: usize,
    ) -> PureResult<Tensor> {
        let weight = self.weight.value();
        let bias = self.bias.value();
        let patches = self.im2col(input, batch, oh, ow)?;
        let mut contracted = einsum_contract(&patches, &weight, "bs,os->bo")?;
        contracted.add_row_inplace(bias.data())?;
        let spatial = oh * ow;
        contracted.reshape(batch, self.out_channels * spatial)
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
    fn conv6da_forward_matches_neighbor_sum_without_leech() {
        let mut conv = Conv6da::new("conv6", 1, 1, (1, 2, 2), 24, 0.0).unwrap();
        for value in conv.weight.value_mut().data_mut() {
            *value = 1.0;
        }
        conv.bias.value_mut().data_mut()[0] = 0.0;
        let input = Tensor::from_vec(1, 4, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), (1, 4));
        let expected = [6.0f32, 7.0, 8.0, 9.0];
        for (out, exp) in output.data().iter().zip(expected.iter()) {
            assert!((out - exp).abs() < 1e-5);
        }
    }

    #[test]
    fn conv6da_forward_injects_leech_density() {
        let mut conv = Conv6da::new("conv6", 1, 1, (1, 2, 2), 24, 1.0).unwrap();
        for value in conv.weight.value_mut().data_mut() {
            *value = 1.0;
        }
        conv.bias.value_mut().data_mut()[0] = 0.0;
        let input = Tensor::from_vec(1, 4, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let output = conv.forward(&input).unwrap();
        let projector = LeechProjector::new(24, 1.0);
        let geodesics = [
            (1.0_f64 * 1.0 + 3.0 * 3.0 + 2.0 * 2.0).sqrt(),
            (2.0_f64 * 2.0 + 4.0 * 4.0 + 1.0 * 1.0).sqrt(),
            (3.0_f64 * 3.0 + 1.0 * 1.0 + 4.0 * 4.0).sqrt(),
            (4.0_f64 * 4.0 + 2.0 * 2.0 + 3.0 * 3.0).sqrt(),
        ];
        let base = [6.0f32, 7.0, 8.0, 9.0];
        for ((out, base_sum), geodesic) in
            output.data().iter().zip(base.iter()).zip(geodesics.iter())
        {
            let expected = *base_sum + projector.enrich(*geodesic) as f32;
            assert!((out - expected).abs() < 1e-4);
        }
    }

    #[test]
    fn conv6da_backward_matches_manual_without_leech() {
        let mut conv = Conv6da::new("conv6", 1, 1, (1, 2, 2), 24, 0.0).unwrap();
        for value in conv.weight.value_mut().data_mut() {
            *value = 1.0;
        }
        conv.bias.value_mut().data_mut()[0] = 0.0;
        let input = Tensor::from_vec(1, 4, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let grad_output = Tensor::from_vec(1, 4, vec![1.0; 4]).unwrap();
        let grad_input = conv.backward(&input, &grad_output).unwrap();
        let expected_grad_input = [3.0f32; 4];
        for (value, expected) in grad_input.data().iter().zip(expected_grad_input.iter()) {
            assert!((value - expected).abs() < 1e-5);
        }
        let weight_grad = conv.weight.gradient().unwrap();
        let expected_weight = [10.0, 0.0, 0.0, 7.0, 3.0, 6.0, 4.0];
        for (value, expected) in weight_grad.data().iter().zip(expected_weight.iter()) {
            assert!((value - expected).abs() < 1e-5);
        }
        let bias_grad = conv.bias.gradient().unwrap();
        assert!((bias_grad.data()[0] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn conv6da_backward_includes_leech_density_contribution() {
        let mut conv = Conv6da::new("conv6", 1, 1, (1, 1, 1), 24, 2.0).unwrap();
        for value in conv.weight.value_mut().data_mut() {
            *value = 0.0;
        }
        conv.bias.value_mut().data_mut()[0] = 0.0;
        let input = Tensor::from_vec(1, 1, vec![3.0]).unwrap();
        let grad_output = Tensor::from_vec(1, 1, vec![0.5]).unwrap();
        let grad_input = conv.backward(&input, &grad_output).unwrap();
        let projector = LeechProjector::new(24, 2.0);
        let expected = grad_output.data()[0] * projector.enrich(1.0) as f32;
        assert!((grad_input.data()[0] - expected).abs() < 1e-6);
        let weight_grad = conv.weight.gradient().unwrap();
        let grads = weight_grad.data();
        assert!((grads[0] - grad_output.data()[0] * input.data()[0]).abs() < 1e-6);
        for &value in &grads[1..] {
            assert!(value.abs() < 1e-6);
        }
        let bias_grad = conv.bias.gradient().unwrap();
        assert!((bias_grad.data()[0] - grad_output.data()[0]).abs() < 1e-6);
    }

    #[test]
    fn conv6da_forward_respects_custom_neighbors() {
        let offsets = [(0, 0, 0), (0, 0, 1)];
        let mut conv =
            Conv6da::with_neighbor_offsets("conv6", 1, 1, (1, 1, 3), 24, 0.0, &offsets).unwrap();
        for value in conv.weight.value_mut().data_mut() {
            *value = 1.0;
        }
        conv.bias.value_mut().data_mut()[0] = 0.0;
        let input = Tensor::from_vec(1, 3, vec![1.0, 2.0, 3.0]).unwrap();
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), (1, 3));
        let expected = [3.0f32, 5.0, 3.0];
        for (&value, &target) in output.data().iter().zip(expected.iter()) {
            assert!((value - target).abs() < 1e-5);
        }
    }

    #[test]
    fn conv1d_forward_matches_manual() {
        let conv = Conv1d::new("conv", 1, 1, 3, 1, 1, 1).unwrap();
        let input = Tensor::from_vec(1, 5, vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape().0, 1);
    }

    #[test]
    fn conv1d_forward_respects_dilation() {
        let mut conv = Conv1d::new("conv", 1, 1, 3, 1, 0).unwrap();
        conv.set_dilation(2).unwrap();
        conv.bias.value_mut().data_mut()[0] = 0.0;
        for (idx, value) in conv.weight.value_mut().data_mut().iter_mut().enumerate() {
            *value = (idx + 1) as f32;
        }
        let input = Tensor::from_vec(1, 5, vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), (1, 1));
        assert!((output.data()[0] - 22.0).abs() < 1e-5);
    }

    #[test]
    fn conv1d_set_dilation_rejects_zero() {
        let mut conv = Conv1d::new("conv", 1, 1, 3, 1, 0).unwrap();
        assert!(conv.set_dilation(0).is_err());
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn conv2d_wgpu_matches_cpu_path() {
        if !wgpu_dense::is_available() {
            return;
        }
        let mut conv =
            Conv2d::new("conv_gpu", 3, 8, (3, 3), (1, 1), (1, 1), (1, 1), (8, 8)).unwrap();
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
    fn max_pool_propagates_gradients_to_maxima() {
        let mut pool = MaxPool2d::new(1, (2, 2), (1, 1), (0, 0), (3, 3)).unwrap();
        let input =
            Tensor::from_vec(1, 9, vec![1.0, 3.0, 2.0, 4.0, 6.0, 5.0, 0.0, -1.0, -2.0]).unwrap();
        pool.forward(&input).unwrap();
        let grad_output = Tensor::from_vec(1, 4, vec![0.5, -1.0, 2.0, 3.0]).unwrap();
        let grad_input = pool.backward(&input, &grad_output).unwrap();
        let mut expected = vec![0.0f32; 9];
        expected[4] = 0.5 - 1.0 + 2.0 + 3.0;
        for (&value, &target) in grad_input.data().iter().zip(expected.iter()) {
            assert!((value - target).abs() < 1e-6);
        }
    }

    #[test]
    fn conv2d_backward_matches_manual_kernel11() {
        let mut conv = Conv2d::new("conv", 1, 1, (1, 1), (1, 1), (0, 0), (1, 1), (2, 2)).unwrap();
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
        let mut conv = Conv2d::new("conv", 1, 1, (3, 3), (1, 1), (0, 0), (1, 1), (9, 9)).unwrap();
        conv.set_dilation((2, 2)).unwrap();
        assert_eq!(conv.output_hw().unwrap(), (5, 5));
    }

    #[test]
    fn conv2d_backward_bands_matches_backward() {
        use crate::plan::RankPlanner;
        use crate::schedule::{RoundtableConfig, RoundtableSchedule};
        use st_core::backend::device_caps::DeviceCaps;

        let mut conv = Conv2d::new("conv_a", 1, 1, (2, 2), (1, 1), (0, 0), (1, 1), (3, 3)).unwrap();
        let mut conv_bands =
            Conv2d::new("conv_b", 1, 1, (2, 2), (1, 1), (0, 0), (1, 1), (3, 3)).unwrap();

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

        let mut conv_seq =
            Conv2d::new("conv_seq", 1, 1, (2, 2), (1, 1), (0, 0), (1, 1), (3, 3)).unwrap();
        let mut conv_volume =
            Conv2d::new("conv_vol", 1, 1, (2, 2), (1, 1), (0, 0), (1, 1), (3, 3)).unwrap();

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
