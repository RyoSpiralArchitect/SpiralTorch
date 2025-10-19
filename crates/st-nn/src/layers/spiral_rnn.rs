// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};
use st_tensor::topos::OpenCartesianTopos;
use std::cell::RefCell;

#[derive(Clone, Debug)]
struct SpiralRnnStepCache {
    input: Tensor,
    state_before: Tensor,
    drive_act: Tensor,
    reset_tanh: Tensor,
    gate: Tensor,
    anchor: Tensor,
}

#[derive(Clone, Debug)]
struct SpiralRnnCache {
    steps: Vec<SpiralRnnStepCache>,
    batch: usize,
    steps_count: usize,
    input_dim: usize,
    hidden_dim: usize,
}

fn tanh_tensor(tensor: &Tensor) -> PureResult<Tensor> {
    let (rows, cols) = tensor.shape();
    let mut data = Vec::with_capacity(rows * cols);
    for &value in tensor.data() {
        data.push(value.tanh());
    }
    Tensor::from_vec(rows, cols, data)
}

fn sigmoid_tensor(tensor: &Tensor) -> PureResult<Tensor> {
    let (rows, cols) = tensor.shape();
    let mut data = Vec::with_capacity(rows * cols);
    for &value in tensor.data() {
        data.push(1.0 / (1.0 + (-value).exp()));
    }
    Tensor::from_vec(rows, cols, data)
}

fn one_minus_tensor(tensor: &Tensor) -> PureResult<Tensor> {
    let (rows, cols) = tensor.shape();
    let mut data = Vec::with_capacity(rows * cols);
    for &value in tensor.data() {
        data.push(1.0 - value);
    }
    Tensor::from_vec(rows, cols, data)
}

fn split_columns(tensor: &Tensor, hidden_dim: usize) -> PureResult<(Tensor, Tensor)> {
    let (rows, cols) = tensor.shape();
    if cols != hidden_dim * 2 {
        return Err(TensorError::ShapeMismatch {
            left: tensor.shape(),
            right: (rows, hidden_dim * 2),
        });
    }
    let mut left = Vec::with_capacity(rows * hidden_dim);
    let mut right = Vec::with_capacity(rows * hidden_dim);
    for r in 0..rows {
        let offset = r * cols;
        left.extend_from_slice(&tensor.data()[offset..offset + hidden_dim]);
        right.extend_from_slice(&tensor.data()[offset + hidden_dim..offset + cols]);
    }
    Ok((
        Tensor::from_vec(rows, hidden_dim, left)?,
        Tensor::from_vec(rows, hidden_dim, right)?,
    ))
}

fn concat_columns(left: &Tensor, right: &Tensor) -> PureResult<Tensor> {
    if left.shape() != right.shape() {
        return Err(TensorError::ShapeMismatch {
            left: left.shape(),
            right: right.shape(),
        });
    }
    let (rows, cols) = left.shape();
    let mut data = Vec::with_capacity(rows * cols * 2);
    for r in 0..rows {
        let offset = r * cols;
        data.extend_from_slice(&left.data()[offset..offset + cols]);
        data.extend_from_slice(&right.data()[offset..offset + cols]);
    }
    Tensor::from_vec(rows, cols * 2, data)
}

const FALLBACK_SATURATION: f32 = 1e6;

fn saturate_slice(values: &mut [f32], topos: Option<&OpenCartesianTopos>) {
    if let Some(topos) = topos {
        topos.saturate_slice(values);
    } else {
        for value in values.iter_mut() {
            if !value.is_finite() {
                *value = 0.0;
                continue;
            }
            if *value > FALLBACK_SATURATION {
                *value = FALLBACK_SATURATION;
            } else if *value < -FALLBACK_SATURATION {
                *value = -FALLBACK_SATURATION;
            }
        }
    }
}

fn guard_slice(
    label: &'static str,
    values: &[f32],
    topos: Option<&OpenCartesianTopos>,
) -> PureResult<()> {
    if let Some(topos) = topos {
        topos.guard_slice(label, values)
    } else {
        for &value in values {
            if !value.is_finite() {
                return Err(TensorError::NonFiniteValue { label, value });
            }
        }
        Ok(())
    }
}

fn stabilise_tensor(
    label: &'static str,
    tensor: &mut Tensor,
    topos: Option<&OpenCartesianTopos>,
) -> PureResult<()> {
    {
        let data = tensor.data_mut();
        saturate_slice(data, topos);
    }
    guard_slice(label, tensor.data(), topos)
}

fn broadcast_row(row: &Tensor, rows: usize) -> PureResult<Tensor> {
    if row.shape().0 != 1 {
        return Err(TensorError::ShapeMismatch {
            left: row.shape(),
            right: (1, row.shape().1),
        });
    }
    let cols = row.shape().1;
    let mut data = Vec::with_capacity(rows * cols);
    let slice = row.data();
    for _ in 0..rows {
        data.extend_from_slice(slice);
    }
    Tensor::from_vec(rows, cols, data)
}

#[derive(Debug)]
pub struct SpiralRnn {
    input_kernel: Parameter,
    state_kernel: Parameter,
    phase_kernel: Parameter,
    bias: Parameter,
    phase_bias: Parameter,
    anchor: Parameter,
    input_dim: usize,
    hidden_dim: usize,
    steps: usize,
    cache: RefCell<Option<SpiralRnnCache>>,
}

impl SpiralRnn {
    pub fn new(
        name: impl Into<String>,
        input_dim: usize,
        hidden_dim: usize,
        steps: usize,
    ) -> PureResult<Self> {
        if input_dim == 0 || hidden_dim == 0 || steps == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: input_dim.max(1),
                cols: hidden_dim.max(1),
            });
        }
        let name = name.into();
        let mut seed = 0.0027f32;
        let input_kernel = Tensor::from_fn(input_dim, hidden_dim * 2, |_r, _c| {
            let value = seed;
            seed = (seed * 1.873).rem_euclid(0.05).max(1e-4);
            value
        })?;
        let mut seed_state = 0.0013f32;
        let state_kernel = Tensor::from_fn(hidden_dim, hidden_dim * 2, |_r, _c| {
            let value = seed_state;
            seed_state = (seed_state * 1.619).rem_euclid(0.04).max(1e-4);
            value
        })?;
        let phase_kernel = Tensor::from_fn(hidden_dim, hidden_dim, |r, c| {
            if r == c {
                0.5
            } else {
                ((r as f32 - c as f32).sin()) * 0.01
            }
        })?;
        let bias = Tensor::zeros(1, hidden_dim * 2)?;
        let phase_bias = Tensor::zeros(1, hidden_dim)?;
        let anchor = Tensor::from_fn(1, hidden_dim, |_r, c| 0.05 * ((c as f32 + 1.0).ln().sin()))?;
        Ok(Self {
            input_kernel: Parameter::new(format!("{name}::input_kernel"), input_kernel),
            state_kernel: Parameter::new(format!("{name}::state_kernel"), state_kernel),
            phase_kernel: Parameter::new(format!("{name}::phase_kernel"), phase_kernel),
            bias: Parameter::new(format!("{name}::bias"), bias),
            phase_bias: Parameter::new(format!("{name}::phase_bias"), phase_bias),
            anchor: Parameter::new(format!("{name}::anchor"), anchor),
            input_dim,
            hidden_dim,
            steps,
            cache: RefCell::new(None),
        })
    }
}

impl Module for SpiralRnn {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let (batch, cols) = input.shape();
        if cols != self.input_dim * self.steps {
            return Err(TensorError::ShapeMismatch {
                left: input.shape(),
                right: (batch, self.input_dim * self.steps),
            });
        }
        let anchor_topos_owned = self.anchor.hypergrad().map(|hyper| hyper.topos().clone());
        let anchor_topos = anchor_topos_owned.as_ref();
        let mut anchor = broadcast_row(self.anchor.value(), batch)?;
        stabilise_tensor("spiral_rnn_anchor", &mut anchor, anchor_topos)?;
        let mut state = anchor.clone();
        stabilise_tensor("spiral_rnn_state_initial", &mut state, anchor_topos)?;
        guard_slice("spiral_rnn_input", input.data(), anchor_topos)?;
        let mut caches = Vec::with_capacity(self.steps);
        for step in 0..self.steps {
            stabilise_tensor("spiral_rnn_state_loop", &mut state, anchor_topos)?;
            let mut step_input = Vec::with_capacity(batch * self.input_dim);
            for b in 0..batch {
                let offset = b * cols + step * self.input_dim;
                step_input.extend_from_slice(&input.data()[offset..offset + self.input_dim]);
            }
            let mut input_step = Tensor::from_vec(batch, self.input_dim, step_input)?;
            stabilise_tensor("spiral_rnn_input_step", &mut input_step, anchor_topos)?;
            let mut input_proj = input_step.matmul(self.input_kernel.value())?;
            stabilise_tensor("spiral_rnn_input_proj", &mut input_proj, anchor_topos)?;
            let mut state_proj = state.matmul(self.state_kernel.value())?;
            stabilise_tensor("spiral_rnn_state_proj", &mut state_proj, anchor_topos)?;
            let mut combined = input_proj.add(&state_proj)?;
            combined.add_row_inplace(self.bias.value().data())?;
            stabilise_tensor("spiral_rnn_combined", &mut combined, anchor_topos)?;
            let (mut drive_pre, mut reset_pre) = split_columns(&combined, self.hidden_dim)?;
            stabilise_tensor("spiral_rnn_drive_pre", &mut drive_pre, anchor_topos)?;
            stabilise_tensor("spiral_rnn_reset_pre", &mut reset_pre, anchor_topos)?;
            let mut drive_act = tanh_tensor(&drive_pre)?;
            stabilise_tensor("spiral_rnn_drive_act", &mut drive_act, anchor_topos)?;
            let mut state_phase = state.matmul(self.phase_kernel.value())?;
            stabilise_tensor("spiral_rnn_state_phase", &mut state_phase, anchor_topos)?;
            let mut gate_pre = state_phase.add(&reset_pre)?;
            gate_pre.add_row_inplace(self.phase_bias.value().data())?;
            stabilise_tensor("spiral_rnn_gate_pre", &mut gate_pre, anchor_topos)?;
            let mut gate = sigmoid_tensor(&gate_pre)?;
            stabilise_tensor("spiral_rnn_gate", &mut gate, anchor_topos)?;
            let mut one_minus = one_minus_tensor(&gate)?;
            stabilise_tensor("spiral_rnn_one_minus", &mut one_minus, anchor_topos)?;
            let mut retained = state.hadamard(&gate)?;
            stabilise_tensor("spiral_rnn_retained", &mut retained, anchor_topos)?;
            let mut injected = drive_act.hadamard(&one_minus)?;
            stabilise_tensor("spiral_rnn_injected", &mut injected, anchor_topos)?;
            let mut reset_tanh = tanh_tensor(&reset_pre)?;
            stabilise_tensor("spiral_rnn_reset_tanh", &mut reset_tanh, anchor_topos)?;
            let mut anchor_mix = anchor.hadamard(&reset_tanh)?;
            stabilise_tensor("spiral_rnn_anchor_mix", &mut anchor_mix, anchor_topos)?;
            let mut new_state = retained.add(&injected)?;
            stabilise_tensor("spiral_rnn_state_pre_anchor", &mut new_state, anchor_topos)?;
            new_state.add_scaled(&anchor_mix, 1.0)?;
            stabilise_tensor("spiral_rnn_state_post_anchor", &mut new_state, anchor_topos)?;
            caches.push(SpiralRnnStepCache {
                input: input_step,
                state_before: state.clone(),
                drive_act,
                reset_tanh,
                gate,
                anchor: anchor.clone(),
            });
            state = new_state;
        }
        stabilise_tensor("spiral_rnn_state_final", &mut state, anchor_topos)?;
        let mut output = state.clone();
        stabilise_tensor("spiral_rnn_output", &mut output, anchor_topos)?;
        *self.cache.borrow_mut() = Some(SpiralRnnCache {
            steps: caches,
            batch,
            steps_count: self.steps,
            input_dim: self.input_dim,
            hidden_dim: self.hidden_dim,
        });
        Ok(output)
    }

    fn backward(&mut self, _input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        let cache = self
            .cache
            .borrow_mut()
            .take()
            .ok_or(TensorError::EmptyInput("spiral_rnn_cache"))?;
        if grad_output.shape() != (cache.batch, cache.hidden_dim) {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: (cache.batch, cache.hidden_dim),
            });
        }
        let anchor_topos_owned = self.anchor.hypergrad().map(|hyper| hyper.topos().clone());
        let anchor_topos = anchor_topos_owned.as_ref();
        let mut grad_state = grad_output.clone();
        stabilise_tensor("spiral_rnn_grad_state_init", &mut grad_state, anchor_topos)?;
        guard_slice("spiral_rnn_grad_output", grad_output.data(), anchor_topos)?;
        let mut grad_input_data = vec![0.0f32; cache.batch * cache.input_dim * cache.steps_count];
        let mut grad_input_kernel = Tensor::zeros(self.input_dim, self.hidden_dim * 2)?;
        let mut grad_state_kernel = Tensor::zeros(self.hidden_dim, self.hidden_dim * 2)?;
        let mut grad_phase_kernel = Tensor::zeros(self.hidden_dim, self.hidden_dim)?;
        let mut grad_bias = Tensor::zeros(1, self.hidden_dim * 2)?;
        let mut grad_phase_bias = Tensor::zeros(1, self.hidden_dim)?;
        let mut grad_anchor = Tensor::zeros(1, self.hidden_dim)?;
        for step_idx in (0..cache.steps_count).rev() {
            let step = &cache.steps[step_idx];
            let mut one_minus = one_minus_tensor(&step.gate)?;
            stabilise_tensor("spiral_rnn_one_minus_back", &mut one_minus, anchor_topos)?;
            let mut grad_retained = grad_state.clone();
            stabilise_tensor("spiral_rnn_grad_retained", &mut grad_retained, anchor_topos)?;
            let mut grad_injected = grad_state.clone();
            stabilise_tensor("spiral_rnn_grad_injected", &mut grad_injected, anchor_topos)?;
            let mut grad_anchor_mix = grad_state.clone();
            stabilise_tensor(
                "spiral_rnn_grad_anchor_mix",
                &mut grad_anchor_mix,
                anchor_topos,
            )?;

            let mut grad_state_from_retained = grad_retained.hadamard(&step.gate)?;
            stabilise_tensor(
                "spiral_rnn_grad_state_from_retained",
                &mut grad_state_from_retained,
                anchor_topos,
            )?;
            let mut grad_gate_from_retained = grad_retained.hadamard(&step.state_before)?;
            stabilise_tensor(
                "spiral_rnn_grad_gate_retained",
                &mut grad_gate_from_retained,
                anchor_topos,
            )?;

            let mut grad_drive_act = grad_injected.hadamard(&one_minus)?;
            stabilise_tensor(
                "spiral_rnn_grad_drive_act",
                &mut grad_drive_act,
                anchor_topos,
            )?;
            let mut grad_one_minus = grad_injected.hadamard(&step.drive_act)?;
            stabilise_tensor(
                "spiral_rnn_grad_one_minus",
                &mut grad_one_minus,
                anchor_topos,
            )?;
            let mut grad_gate_from_injected = grad_one_minus.scale(-1.0)?;
            stabilise_tensor(
                "spiral_rnn_grad_gate_injected",
                &mut grad_gate_from_injected,
                anchor_topos,
            )?;

            let mut grad_anchor_broadcast = grad_anchor_mix.hadamard(&step.reset_tanh)?;
            stabilise_tensor(
                "spiral_rnn_grad_anchor_broadcast",
                &mut grad_anchor_broadcast,
                anchor_topos,
            )?;
            let mut grad_reset_tanh = grad_anchor_mix.hadamard(&step.anchor)?;
            stabilise_tensor(
                "spiral_rnn_grad_reset_tanh",
                &mut grad_reset_tanh,
                anchor_topos,
            )?;
            let mut grad_anchor_step =
                Tensor::from_vec(1, self.hidden_dim, grad_anchor_broadcast.sum_axis0())?;
            stabilise_tensor(
                "spiral_rnn_grad_anchor_step",
                &mut grad_anchor_step,
                anchor_topos,
            )?;
            grad_anchor.add_scaled(&grad_anchor_step, 1.0)?;
            stabilise_tensor(
                "spiral_rnn_grad_anchor_total",
                &mut grad_anchor,
                anchor_topos,
            )?;

            let mut grad_drive_pre = {
                let mut factor = Vec::with_capacity(step.drive_act.data().len());
                for &v in step.drive_act.data() {
                    factor.push(1.0 - v * v);
                }
                Tensor::from_vec(step.drive_act.shape().0, step.drive_act.shape().1, factor)?
                    .hadamard(&grad_drive_act)?
            };
            stabilise_tensor(
                "spiral_rnn_grad_drive_pre",
                &mut grad_drive_pre,
                anchor_topos,
            )?;

            let mut grad_reset_from_anchor = {
                let mut factor = Vec::with_capacity(step.reset_tanh.data().len());
                for &v in step.reset_tanh.data() {
                    factor.push(1.0 - v * v);
                }
                Tensor::from_vec(step.reset_tanh.shape().0, step.reset_tanh.shape().1, factor)?
                    .hadamard(&grad_reset_tanh)?
            };
            stabilise_tensor(
                "spiral_rnn_grad_reset_from_anchor",
                &mut grad_reset_from_anchor,
                anchor_topos,
            )?;

            let mut grad_gate_total = grad_gate_from_retained.add(&grad_gate_from_injected)?;
            stabilise_tensor(
                "spiral_rnn_grad_gate_total",
                &mut grad_gate_total,
                anchor_topos,
            )?;
            let mut grad_gate_pre = {
                let gate = step.gate.clone();
                let slope = gate.hadamard(&one_minus)?;
                slope.hadamard(&grad_gate_total)?
            };
            stabilise_tensor("spiral_rnn_grad_gate_pre", &mut grad_gate_pre, anchor_topos)?;

            let mut grad_reset_pre = grad_gate_pre.add(&grad_reset_from_anchor)?;
            stabilise_tensor(
                "spiral_rnn_grad_reset_pre",
                &mut grad_reset_pre,
                anchor_topos,
            )?;
            let mut grad_state_phase = grad_gate_pre.clone();
            stabilise_tensor(
                "spiral_rnn_grad_state_phase",
                &mut grad_state_phase,
                anchor_topos,
            )?;

            let grad_combined_left = grad_drive_pre.clone();
            let grad_combined_right = grad_reset_pre.clone();
            let mut grad_combined = concat_columns(&grad_combined_left, &grad_combined_right)?;
            stabilise_tensor("spiral_rnn_grad_combined", &mut grad_combined, anchor_topos)?;

            let mut grad_bias_step =
                Tensor::from_vec(1, self.hidden_dim * 2, grad_combined.sum_axis0())?;
            stabilise_tensor(
                "spiral_rnn_grad_bias_step",
                &mut grad_bias_step,
                anchor_topos,
            )?;
            grad_bias.add_scaled(&grad_bias_step, 1.0)?;
            stabilise_tensor("spiral_rnn_grad_bias", &mut grad_bias, anchor_topos)?;

            let mut grad_input_proj =
                grad_combined.matmul(&self.input_kernel.value().transpose())?;
            stabilise_tensor(
                "spiral_rnn_grad_input_proj",
                &mut grad_input_proj,
                anchor_topos,
            )?;
            let mut grad_state_proj =
                grad_combined.matmul(&self.state_kernel.value().transpose())?;
            stabilise_tensor(
                "spiral_rnn_grad_state_proj",
                &mut grad_state_proj,
                anchor_topos,
            )?;

            let mut grad_input_kernel_step = step.input.transpose().matmul(&grad_combined)?;
            stabilise_tensor(
                "spiral_rnn_grad_input_kernel_step",
                &mut grad_input_kernel_step,
                anchor_topos,
            )?;
            grad_input_kernel.add_scaled(&grad_input_kernel_step, 1.0)?;
            stabilise_tensor(
                "spiral_rnn_grad_input_kernel",
                &mut grad_input_kernel,
                anchor_topos,
            )?;

            let mut grad_state_kernel_step =
                step.state_before.transpose().matmul(&grad_combined)?;
            stabilise_tensor(
                "spiral_rnn_grad_state_kernel_step",
                &mut grad_state_kernel_step,
                anchor_topos,
            )?;
            grad_state_kernel.add_scaled(&grad_state_kernel_step, 1.0)?;
            stabilise_tensor(
                "spiral_rnn_grad_state_kernel",
                &mut grad_state_kernel,
                anchor_topos,
            )?;

            let mut grad_phase_kernel_step =
                step.state_before.transpose().matmul(&grad_state_phase)?;
            stabilise_tensor(
                "spiral_rnn_grad_phase_kernel_step",
                &mut grad_phase_kernel_step,
                anchor_topos,
            )?;
            grad_phase_kernel.add_scaled(&grad_phase_kernel_step, 1.0)?;
            stabilise_tensor(
                "spiral_rnn_grad_phase_kernel",
                &mut grad_phase_kernel,
                anchor_topos,
            )?;

            let mut grad_phase_bias_step =
                Tensor::from_vec(1, self.hidden_dim, grad_state_phase.sum_axis0())?;
            stabilise_tensor(
                "spiral_rnn_grad_phase_bias_step",
                &mut grad_phase_bias_step,
                anchor_topos,
            )?;
            grad_phase_bias.add_scaled(&grad_phase_bias_step, 1.0)?;
            stabilise_tensor(
                "spiral_rnn_grad_phase_bias",
                &mut grad_phase_bias,
                anchor_topos,
            )?;

            let mut grad_state_from_phase =
                grad_state_phase.matmul(&self.phase_kernel.value().transpose())?;
            stabilise_tensor(
                "spiral_rnn_grad_state_from_phase",
                &mut grad_state_from_phase,
                anchor_topos,
            )?;

            let mut grad_state_before = grad_state_from_retained;
            grad_state_before.add_scaled(&grad_state_proj, 1.0)?;
            grad_state_before.add_scaled(&grad_state_from_phase, 1.0)?;
            stabilise_tensor(
                "spiral_rnn_grad_state_before",
                &mut grad_state_before,
                anchor_topos,
            )?;

            let grad_input_step = grad_input_proj;
            let grad_input_slice = grad_input_step.data();
            let total_cols = self.input_dim * self.steps;
            for b in 0..cache.batch {
                let dest_offset = b * total_cols + step_idx * self.input_dim;
                let src_offset = b * self.input_dim;
                grad_input_data[dest_offset..dest_offset + self.input_dim]
                    .copy_from_slice(&grad_input_slice[src_offset..src_offset + self.input_dim]);
            }

            grad_state = grad_state_before;
        }
        let inv_batch = 1.0 / cache.batch as f32;
        stabilise_tensor(
            "spiral_rnn_grad_input_kernel_final",
            &mut grad_input_kernel,
            anchor_topos,
        )?;
        self.input_kernel
            .accumulate_euclidean(&grad_input_kernel.scale(inv_batch)?)?;
        stabilise_tensor(
            "spiral_rnn_grad_state_kernel_final",
            &mut grad_state_kernel,
            anchor_topos,
        )?;
        self.state_kernel
            .accumulate_euclidean(&grad_state_kernel.scale(inv_batch)?)?;
        stabilise_tensor(
            "spiral_rnn_grad_phase_kernel_final",
            &mut grad_phase_kernel,
            anchor_topos,
        )?;
        self.phase_kernel
            .accumulate_euclidean(&grad_phase_kernel.scale(inv_batch)?)?;
        stabilise_tensor("spiral_rnn_grad_bias_final", &mut grad_bias, anchor_topos)?;
        self.bias
            .accumulate_euclidean(&grad_bias.scale(inv_batch)?)?;
        stabilise_tensor(
            "spiral_rnn_grad_phase_bias_final",
            &mut grad_phase_bias,
            anchor_topos,
        )?;
        self.phase_bias
            .accumulate_euclidean(&grad_phase_bias.scale(inv_batch)?)?;
        stabilise_tensor(
            "spiral_rnn_grad_anchor_final",
            &mut grad_anchor,
            anchor_topos,
        )?;
        self.anchor
            .accumulate_euclidean(&grad_anchor.scale(inv_batch)?)?;
        saturate_slice(&mut grad_input_data, anchor_topos);
        guard_slice("spiral_rnn_grad_input", &grad_input_data, anchor_topos)?;
        Tensor::from_vec(
            cache.batch,
            cache.input_dim * cache.steps_count,
            grad_input_data,
        )
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.input_kernel)?;
        visitor(&self.state_kernel)?;
        visitor(&self.phase_kernel)?;
        visitor(&self.bias)?;
        visitor(&self.phase_bias)?;
        visitor(&self.anchor)?;
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.input_kernel)?;
        visitor(&mut self.state_kernel)?;
        visitor(&mut self.phase_kernel)?;
        visitor(&mut self.bias)?;
        visitor(&mut self.phase_bias)?;
        visitor(&mut self.anchor)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spiral_rnn_forward_backward() {
        let mut rnn = SpiralRnn::new("spiral", 3, 5, 2).unwrap();
        let input = Tensor::from_vec(
            2,
            6,
            vec![
                0.1, 0.2, -0.1, 0.05, -0.03, 0.04, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7,
            ],
        )
        .unwrap();
        let output = rnn.forward(&input).unwrap();
        assert_eq!(output.shape(), (2, 5));
        let grad_out = Tensor::from_vec(
            2,
            5,
            vec![
                0.01, -0.02, 0.03, -0.01, 0.04, -0.05, 0.02, 0.01, -0.03, 0.02,
            ],
        )
        .unwrap();
        let grad_input = rnn.backward(&input, &grad_out).unwrap();
        assert_eq!(grad_input.shape(), input.shape());
    }
}
