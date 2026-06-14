// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};
use std::collections::HashMap;

/// Sequential container that mirrors `nn.Sequential`.
#[derive(Default)]
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl core::fmt::Debug for Sequential {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Sequential(num_layers={})", self.layers.len())
    }
}

impl Sequential {
    /// Creates an empty container.
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Appends a new layer to the sequence.
    pub fn push<M>(&mut self, layer: M)
    where
        M: Module + 'static,
    {
        self.layers.push(Box::new(layer));
    }

    /// Inserts a new layer at the provided position.
    pub fn insert<M>(&mut self, index: usize, layer: M) -> PureResult<()>
    where
        M: Module + 'static,
    {
        if index > self.layers.len() {
            return Err(TensorError::InvalidDimensions {
                rows: index,
                cols: self.layers.len(),
            });
        }
        self.layers.insert(index, Box::new(layer));
        Ok(())
    }

    /// Appends a pre-boxed module to the sequence.
    pub fn push_boxed(&mut self, layer: Box<dyn Module>) {
        self.layers.push(layer);
    }

    /// Returns the number of layers registered in the container.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Returns `true` when the container does not hold any layers.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        let mut activ = input.clone();
        for layer in &self.layers {
            activ = layer.forward(&activ)?;
        }
        Ok(activ)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        if self.layers.is_empty() {
            return Ok(grad_output.clone());
        }
        let mut activations = Vec::with_capacity(self.layers.len());
        let mut current = input.clone();
        for layer in &self.layers {
            let next = layer.forward(&current)?;
            activations.push(next.clone());
            current = next;
        }
        let mut grad = grad_output.clone();
        for (idx, layer) in self.layers.iter_mut().enumerate().rev() {
            let layer_input = if idx == 0 {
                input
            } else {
                &activations[idx - 1]
            };
            grad = layer.backward(layer_input, &grad)?;
        }
        Ok(grad)
    }

    fn visit_parameters(
        &self,
        visitor: &mut dyn FnMut(&Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        for layer in &self.layers {
            layer.visit_parameters(visitor)?;
        }
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        for layer in &mut self.layers {
            layer.visit_parameters_mut(visitor)?;
        }
        Ok(())
    }

    fn state_dict(&self) -> PureResult<HashMap<String, Tensor>> {
        let mut state = HashMap::new();
        for layer in &self.layers {
            for (name, value) in layer.state_dict()? {
                state.insert(name, value);
            }
        }
        Ok(state)
    }

    fn load_state_dict(&mut self, state: &HashMap<String, Tensor>) -> PureResult<()> {
        for layer in &mut self.layers {
            layer.load_state_dict(state)?;
        }
        Ok(())
    }

    fn infuse_text(&mut self, text: &str) -> PureResult<()> {
        for layer in &mut self.layers {
            layer.infuse_text(text)?;
        }
        Ok(())
    }

    fn set_training(&mut self, training: bool) -> PureResult<()> {
        for layer in &mut self.layers {
            layer.set_training(training)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::linear::Linear;

    #[test]
    fn sequential_forward_and_backward() {
        let mut seq = Sequential::new();
        seq.push(Linear::new("l1", 2, 3).unwrap());
        seq.push(Linear::new("l2", 3, 1).unwrap());
        seq.attach_hypergrad(-1.0, 0.05).unwrap();

        let input = Tensor::from_vec(1, 2, vec![0.5, -0.1]).unwrap();
        let target = Tensor::from_vec(1, 1, vec![0.2]).unwrap();
        let output = seq.forward(&input).unwrap();
        let grad_out = output.sub(&target).unwrap();
        let _ = seq.backward(&input, &grad_out).unwrap();
        seq.apply_step(0.01).unwrap();
        let new_output = seq.forward(&input).unwrap();
        assert_ne!(output, new_output);
    }

    #[test]
    fn sequential_insert_places_layer_and_rejects_out_of_bounds_index() {
        let mut seq = Sequential::new();
        seq.push(Linear::new("head", 2, 1).unwrap());

        assert!(seq.insert(2, Linear::new("bad", 1, 1).unwrap()).is_err());
        seq.insert(0, Linear::new("project", 2, 2).unwrap())
            .unwrap();

        let input = Tensor::from_vec(1, 2, vec![0.5, -0.25]).unwrap();
        let output = seq.forward(&input).unwrap();
        assert_eq!(seq.len(), 2);
        assert_eq!(output.shape(), (1, 1));
    }
}
