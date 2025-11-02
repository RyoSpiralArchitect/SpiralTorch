// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor};
use st_tensor::TensorError;
use std::cell::RefCell;

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

/// Single-layer LSTM operating on sequences laid out along the batch axis.
#[derive(Debug)]
pub struct Lstm {
    input_dim: usize,
    hidden_dim: usize,
    weight_ih: Parameter,
    weight_hh: Parameter,
    bias_ih: Parameter,
    bias_hh: Parameter,
    hidden_state: RefCell<Tensor>,
    cell_state: RefCell<Tensor>,
    cache: RefCell<Option<LstmCache>>,
}

#[derive(Debug, Clone)]
struct LstmCache {
    inputs: Vec<f32>,
    gates_i: Vec<f32>,
    gates_f: Vec<f32>,
    gates_g: Vec<f32>,
    gates_o: Vec<f32>,
    hidden_states: Vec<f32>,
    cell_states: Vec<f32>,
    timesteps: usize,
    input_dim: usize,
    hidden_dim: usize,
}

impl LstmCache {
    fn new(timesteps: usize, input_dim: usize, hidden_dim: usize, h0: &[f32], c0: &[f32]) -> Self {
        let mut hidden_states = vec![0.0f32; (timesteps + 1) * hidden_dim];
        hidden_states[..hidden_dim].copy_from_slice(h0);
        let mut cell_states = vec![0.0f32; (timesteps + 1) * hidden_dim];
        cell_states[..hidden_dim].copy_from_slice(c0);
        Self {
            inputs: vec![0.0f32; timesteps * input_dim],
            gates_i: vec![0.0f32; timesteps * hidden_dim],
            gates_f: vec![0.0f32; timesteps * hidden_dim],
            gates_g: vec![0.0f32; timesteps * hidden_dim],
            gates_o: vec![0.0f32; timesteps * hidden_dim],
            hidden_states,
            cell_states,
            timesteps,
            input_dim,
            hidden_dim,
        }
    }
}

impl Lstm {
    /// Creates a new LSTM layer with small deterministic parameters.
    pub fn new(name: impl Into<String>, input_dim: usize, hidden_dim: usize) -> PureResult<Self> {
        if input_dim == 0 || hidden_dim == 0 {
            return Err(TensorError::InvalidDimensions {
                rows: input_dim,
                cols: hidden_dim,
            });
        }
        let name = name.into();
        let weight_ih = Tensor::from_fn(input_dim, 4 * hidden_dim, |row, col| {
            (((row * 13 + col * 7) % 17) as f32 + 1.0) * 0.01
        })?;
        let weight_hh = Tensor::from_fn(hidden_dim, 4 * hidden_dim, |row, col| {
            (((row * 11 + col * 5) % 23) as f32 + 1.0) * 0.01
        })?;
        let bias_ih = Tensor::zeros(1, 4 * hidden_dim)?;
        let bias_hh = Tensor::zeros(1, 4 * hidden_dim)?;
        let hidden_state = Tensor::zeros(1, hidden_dim)?;
        let cell_state = Tensor::zeros(1, hidden_dim)?;
        Ok(Self {
            input_dim,
            hidden_dim,
            weight_ih: Parameter::new(format!("{name}::weight_ih"), weight_ih),
            weight_hh: Parameter::new(format!("{name}::weight_hh"), weight_hh),
            bias_ih: Parameter::new(format!("{name}::bias_ih"), bias_ih),
            bias_hh: Parameter::new(format!("{name}::bias_hh"), bias_hh),
            hidden_state: RefCell::new(hidden_state),
            cell_state: RefCell::new(cell_state),
            cache: RefCell::new(None),
        })
    }

    /// Resets the hidden and cell state to zero.
    pub fn reset_state(&self) -> PureResult<()> {
        *self.hidden_state.borrow_mut() = Tensor::zeros(1, self.hidden_dim)?;
        *self.cell_state.borrow_mut() = Tensor::zeros(1, self.hidden_dim)?;
        Ok(())
    }

    /// Loads an explicit initial hidden and cell state.
    pub fn set_state(&self, hidden: &Tensor, cell: &Tensor) -> PureResult<()> {
        if hidden.shape() != (1, self.hidden_dim) || cell.shape() != (1, self.hidden_dim) {
            return Err(TensorError::ShapeMismatch {
                left: hidden.shape(),
                right: (1, self.hidden_dim),
            });
        }
        *self.hidden_state.borrow_mut() = hidden.clone();
        *self.cell_state.borrow_mut() = cell.clone();
        Ok(())
    }

    fn guard_input(&self, input: &Tensor) -> PureResult<()> {
        let (rows, cols) = input.shape();
        if cols != self.input_dim {
            return Err(TensorError::ShapeMismatch {
                left: (rows, cols),
                right: (rows, self.input_dim),
            });
        }
        if rows == 0 {
            return Err(TensorError::EmptyInput("lstm_forward"));
        }
        Ok(())
    }
}

impl Module for Lstm {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        self.guard_input(input)?;
        let (timesteps, _) = input.shape();
        let hidden_dim = self.hidden_dim;
        let input_dim = self.input_dim;
        let mut output = vec![0.0f32; timesteps * hidden_dim];
        let mut hidden_prev = self.hidden_state.borrow().data().to_vec();
        let mut cell_prev = self.cell_state.borrow().data().to_vec();
        let cache_template =
            LstmCache::new(timesteps, input_dim, hidden_dim, &hidden_prev, &cell_prev);
        let mut cache = cache_template;
        let weight_ih = self.weight_ih.value();
        let weight_hh = self.weight_hh.value();
        let bias_ih = self.bias_ih.value();
        let bias_hh = self.bias_hh.value();
        for t in 0..timesteps {
            let input_slice = &input.data()[t * input_dim..(t + 1) * input_dim];
            cache.inputs[t * input_dim..(t + 1) * input_dim].copy_from_slice(input_slice);
            let mut gates = vec![0.0f32; 4 * hidden_dim];
            for gate in 0..4 * hidden_dim {
                let mut value = bias_ih.data()[gate] + bias_hh.data()[gate];
                for idx in 0..input_dim {
                    value += input_slice[idx] * weight_ih.data()[idx * 4 * hidden_dim + gate];
                }
                for idx in 0..hidden_dim {
                    value += hidden_prev[idx] * weight_hh.data()[idx * 4 * hidden_dim + gate];
                }
                gates[gate] = value;
            }
            for unit in 0..hidden_dim {
                let gi = sigmoid(gates[unit]);
                let gf = sigmoid(gates[hidden_dim + unit]);
                let gg = gates[2 * hidden_dim + unit].tanh();
                let go = sigmoid(gates[3 * hidden_dim + unit]);
                let cell = gf * cell_prev[unit] + gi * gg;
                let hidden = go * cell.tanh();
                cache.gates_i[t * hidden_dim + unit] = gi;
                cache.gates_f[t * hidden_dim + unit] = gf;
                cache.gates_g[t * hidden_dim + unit] = gg;
                cache.gates_o[t * hidden_dim + unit] = go;
                cache.cell_states[(t + 1) * hidden_dim + unit] = cell;
                cache.hidden_states[(t + 1) * hidden_dim + unit] = hidden;
                cell_prev[unit] = cell;
                hidden_prev[unit] = hidden;
                output[t * hidden_dim + unit] = hidden;
            }
        }
        *self.hidden_state.borrow_mut() = Tensor::from_vec(1, hidden_dim, hidden_prev.clone())?;
        *self.cell_state.borrow_mut() = Tensor::from_vec(1, hidden_dim, cell_prev.clone())?;
        *self.cache.borrow_mut() = Some(cache);
        Tensor::from_vec(timesteps, hidden_dim, output)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        self.guard_input(input)?;
        if grad_output.shape().0 != input.shape().0 || grad_output.shape().1 != self.hidden_dim {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: (input.shape().0, self.hidden_dim),
            });
        }
        let cache = self
            .cache
            .borrow_mut()
            .take()
            .ok_or(TensorError::InvalidValue {
                label: "lstm_cache_missing",
            })?;
        let timesteps = cache.timesteps;
        let input_dim = cache.input_dim;
        let hidden_dim = cache.hidden_dim;
        let weight_ih = self.weight_ih.value();
        let weight_hh = self.weight_hh.value();
        let mut grad_input = vec![0.0f32; timesteps * input_dim];
        let mut grad_w_ih = vec![0.0f32; input_dim * 4 * hidden_dim];
        let mut grad_w_hh = vec![0.0f32; hidden_dim * 4 * hidden_dim];
        let mut grad_b_ih = vec![0.0f32; 4 * hidden_dim];
        let mut grad_b_hh = vec![0.0f32; 4 * hidden_dim];
        let mut grad_h_next = vec![0.0f32; hidden_dim];
        let mut grad_c_next = vec![0.0f32; hidden_dim];
        for step in (0..timesteps).rev() {
            let grad_hidden_slice = &grad_output.data()[step * hidden_dim..(step + 1) * hidden_dim];
            let prev_hidden = &cache.hidden_states[step * hidden_dim..(step + 1) * hidden_dim];
            let prev_cell = &cache.cell_states[step * hidden_dim..(step + 1) * hidden_dim];
            let curr_cell = &cache.cell_states[(step + 1) * hidden_dim..(step + 2) * hidden_dim];
            let mut gate_grad = vec![0.0f32; 4 * hidden_dim];
            for unit in 0..hidden_dim {
                let dh = grad_hidden_slice[unit] + grad_h_next[unit];
                let o = cache.gates_o[step * hidden_dim + unit];
                let i = cache.gates_i[step * hidden_dim + unit];
                let f = cache.gates_f[step * hidden_dim + unit];
                let g = cache.gates_g[step * hidden_dim + unit];
                let tanh_c = curr_cell[unit].tanh();
                let do_gate = dh * tanh_c * o * (1.0 - o);
                let dc = dh * o * (1.0 - tanh_c * tanh_c) + grad_c_next[unit];
                let di = dc * g * i * (1.0 - i);
                let dg = dc * i * (1.0 - g * g);
                let df = dc * prev_cell[unit] * f * (1.0 - f);
                grad_c_next[unit] = dc * f;
                gate_grad[unit] = di;
                gate_grad[hidden_dim + unit] = df;
                gate_grad[2 * hidden_dim + unit] = dg;
                gate_grad[3 * hidden_dim + unit] = do_gate;
            }
            for gate in 0..4 * hidden_dim {
                grad_b_ih[gate] += gate_grad[gate];
                grad_b_hh[gate] += gate_grad[gate];
            }
            for input_idx in 0..input_dim {
                let mut acc = 0.0f32;
                for gate in 0..4 * hidden_dim {
                    acc += gate_grad[gate] * weight_ih.data()[input_idx * 4 * hidden_dim + gate];
                    grad_w_ih[input_idx * 4 * hidden_dim + gate] +=
                        cache.inputs[step * input_dim + input_idx] * gate_grad[gate];
                }
                grad_input[step * input_dim + input_idx] = acc;
            }
            let mut next_h = vec![0.0f32; hidden_dim];
            for hidden_idx in 0..hidden_dim {
                let mut acc = 0.0f32;
                for gate in 0..4 * hidden_dim {
                    acc += gate_grad[gate] * weight_hh.data()[hidden_idx * 4 * hidden_dim + gate];
                    grad_w_hh[hidden_idx * 4 * hidden_dim + gate] +=
                        prev_hidden[hidden_idx] * gate_grad[gate];
                }
                next_h[hidden_idx] = acc;
            }
            grad_h_next = next_h;
        }
        let grad_w_ih = Tensor::from_vec(input_dim, 4 * hidden_dim, grad_w_ih)?;
        let grad_w_hh = Tensor::from_vec(hidden_dim, 4 * hidden_dim, grad_w_hh)?;
        let grad_b_ih = Tensor::from_vec(1, 4 * hidden_dim, grad_b_ih)?;
        let grad_b_hh = Tensor::from_vec(1, 4 * hidden_dim, grad_b_hh)?;
        self.weight_ih.accumulate_euclidean(&grad_w_ih)?;
        self.weight_hh.accumulate_euclidean(&grad_w_hh)?;
        self.bias_ih.accumulate_euclidean(&grad_b_ih)?;
        self.bias_hh.accumulate_euclidean(&grad_b_hh)?;
        Tensor::from_vec(timesteps, input_dim, grad_input)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&self.weight_ih)?;
        visitor(&self.weight_hh)?;
        visitor(&self.bias_ih)?;
        visitor(&self.bias_hh)
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        visitor(&mut self.weight_ih)?;
        visitor(&mut self.weight_hh)?;
        visitor(&mut self.bias_ih)?;
        visitor(&mut self.bias_hh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lstm_forward_produces_hidden_sequence() {
        let lstm = Lstm::new("lstm", 2, 3).unwrap();
        let input = Tensor::from_vec(4, 2, vec![0.1, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, 0.8]).unwrap();
        let output = lstm.forward(&input).unwrap();
        assert_eq!(output.shape(), (4, 3));
        for value in output.data() {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn lstm_backward_accumulates_gradients() {
        let mut lstm = Lstm::new("lstm", 3, 2).unwrap();
        let input =
            Tensor::from_vec(3, 3, vec![0.2, -0.1, 0.3, 0.4, -0.5, 0.6, -0.2, 0.1, 0.7]).unwrap();
        let grad_out = Tensor::from_vec(3, 2, vec![0.1, -0.2, 0.3, 0.2, -0.4, 0.5]).unwrap();
        let _ = lstm.forward(&input).unwrap();
        let grad_input = lstm.backward(&input, &grad_out).unwrap();
        assert_eq!(grad_input.shape(), (3, 3));
        assert!(lstm.weight_ih.gradient().is_some());
        assert!(lstm.bias_hh.gradient().is_some());
    }
}
