// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::{Module, Parameter};
use crate::{PureResult, Tensor, TensorError};
use st_tensor::{Layout, TensorContentStamp};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

struct SavedParameter {
    name: String,
    stamp: Option<TensorContentStamp>,
    exact: Option<Tensor>,
}

impl SavedParameter {
    fn new(parameter: &Parameter) -> PureResult<Self> {
        let value = parameter.value();
        let stamp = value.content_stamp();
        // Foreign aliases need isolated values. Non-row-major parameters may
        // be normalized by gradient accumulation without changing their value.
        let exact = if stamp.is_none() || value.layout() != Layout::RowMajor {
            Some(value.to_layout(Layout::RowMajor)?.into_snapshot())
        } else {
            None
        };
        Ok(Self {
            name: parameter.name().to_owned(),
            stamp,
            exact,
        })
    }

    fn matches(&self, parameter: &Parameter) -> PureResult<bool> {
        if self.name != parameter.name() {
            return Ok(false);
        }
        if self
            .stamp
            .as_ref()
            .is_some_and(|s| s.matches(parameter.value()))
        {
            return Ok(true);
        }
        match &self.exact {
            Some(value) => same_values(value, parameter.value()),
            None => Ok(false),
        }
    }
}

fn same_values(left: &Tensor, right: &Tensor) -> PureResult<bool> {
    if left.shape() != right.shape() {
        return Ok(false);
    }
    let left = left.to_layout(Layout::RowMajor)?;
    let right = right.to_layout(Layout::RowMajor)?;
    Ok(left
        .data()
        .iter()
        .zip(right.data())
        .all(|(a, b)| a.to_bits() == b.to_bits()))
}

struct SavedForward {
    inputs: Vec<Tensor>,
    input_stamp: Option<TensorContentStamp>,
    output_shape: (usize, usize),
    parameters: Vec<SavedParameter>,
}

/// Sequential container that mirrors `nn.Sequential`.
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
    last_forward: RefCell<Option<Arc<SavedForward>>>,
    #[cfg(feature = "wgpu")]
    resident: crate::resident::ResidentForwardCache,
}

impl core::fmt::Debug for Sequential {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Sequential(num_layers={})", self.layers.len())
    }
}

impl Sequential {
    /// Creates an empty container.
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            last_forward: RefCell::new(None),
            #[cfg(feature = "wgpu")]
            resident: Default::default(),
        }
    }

    /// Appends a new layer to the sequence.
    pub fn push<M>(&mut self, layer: M)
    where
        M: Module + 'static,
    {
        self.last_forward.get_mut().take();
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
        self.last_forward.get_mut().take();
        self.layers.insert(index, Box::new(layer));
        Ok(())
    }

    /// Appends a pre-boxed module to the sequence.
    pub fn push_boxed(&mut self, layer: Box<dyn Module>) {
        self.last_forward.get_mut().take();
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

    /// Forward without retaining host backward inputs. This does not change
    /// training/evaluation mode and invalidates any previous host forward tape.
    pub fn forward_untracked(&self, input: &Tensor) -> PureResult<Tensor> {
        self.forward_untracked_owned(input.clone())
    }

    /// Untracked forward with ownership transfer and no container activation
    /// retention, including nested sequences. Per-layer caches may remain.
    pub fn forward_untracked_owned(&self, mut input: Tensor) -> PureResult<Tensor> {
        self.last_forward.borrow_mut().take();
        for layer in &self.layers {
            input = layer.forward_untracked_owned(input)?;
        }
        Ok(input)
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Sequential {
    #[cfg(feature = "wgpu")]
    fn forward_resident(
        &self,
        input: &st_backend_wgpu::resident_tensor::ResidentTensor,
    ) -> Result<st_backend_wgpu::resident_tensor::ResidentTensor, crate::resident::InferenceError>
    {
        self.last_forward.borrow_mut().take();
        self.resident.forward(self.inference_ops()?, input)
    }
    #[cfg(feature = "wgpu")]
    fn forward_resident_snapshot(
        &self,
        input: &st_backend_wgpu::resident_tensor::ResidentTensor,
    ) -> Result<st_backend_wgpu::resident_tensor::TensorReadback, crate::resident::InferenceError>
    {
        self.last_forward.borrow_mut().take();
        self.resident.snapshot(self.inference_ops()?, input)
    }
    #[cfg(feature = "wgpu")]
    fn resident_forward_stats(&self) -> Option<crate::resident::ResidentForwardStats> {
        Some(self.resident.stats())
    }
    #[cfg(feature = "wgpu")]
    fn clear_resident_forward_cache(&self) {
        self.resident.clear();
    }

    fn resident_parameter_bindings(
        &self,
    ) -> Result<Vec<crate::resident::ResidentParameterBinding<'_>>, crate::resident::InferenceError>
    {
        let mut bindings = Vec::new();
        for layer in &self.layers {
            bindings.extend(layer.resident_parameter_bindings()?);
        }
        Ok(bindings)
    }

    fn inference_ops(
        &self,
    ) -> Result<Vec<crate::resident::InferenceOp>, crate::resident::InferenceError> {
        crate::resident::collect_inference_ops(self, self.layers.len())
    }

    fn append_inference_ops(
        &self,
        operations: &mut Vec<crate::resident::InferenceOp>,
    ) -> Result<(), crate::resident::InferenceError> {
        let start = operations.len();
        for layer in &self.layers {
            if let Err(error) = layer.append_inference_ops(operations) {
                operations.truncate(start);
                return Err(error);
            }
        }
        Ok(())
    }

    fn forward(&self, input: &Tensor) -> PureResult<Tensor> {
        self.forward_owned(input.clone())
    }

    fn forward_untracked_owned(&self, input: Tensor) -> PureResult<Tensor> {
        Sequential::forward_untracked_owned(self, input)
    }

    fn forward_owned(&self, mut activ: Tensor) -> PureResult<Tensor> {
        // Even a failed new forward can replace a child's stochastic state.
        self.last_forward.borrow_mut().take();
        if self.layers.is_empty() {
            return Ok(activ);
        }
        let input_stamp = activ.content_stamp();
        let mut parameters = Vec::new();
        self.visit_parameters(&mut |parameter| {
            parameters.push(SavedParameter::new(parameter)?);
            Ok(())
        })?;
        let mut inputs = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            // Snapshot before transferring ownership; an in-place-capable child
            // must not overwrite the input required by this forward's pullback.
            let saved = activ.into_snapshot();
            activ = layer.forward_owned(saved.clone())?;
            inputs.push(saved);
        }
        self.last_forward
            .borrow_mut()
            .replace(Arc::new(SavedForward {
                inputs,
                input_stamp,
                output_shape: activ.shape(),
                parameters,
            }));
        Ok(activ)
    }

    fn backward(&mut self, input: &Tensor, grad_output: &Tensor) -> PureResult<Tensor> {
        if self.layers.is_empty() {
            if input.shape() != grad_output.shape() {
                return Err(TensorError::ShapeMismatch {
                    left: input.shape(),
                    right: grad_output.shape(),
                });
            }
            return Ok(grad_output.clone());
        }
        let saved =
            self.last_forward
                .borrow()
                .as_ref()
                .cloned()
                .ok_or(TensorError::InvalidValue {
                    label: "sequential_forward_missing",
                })?;
        if grad_output.shape() != saved.output_shape {
            return Err(TensorError::ShapeMismatch {
                left: grad_output.shape(),
                right: saved.output_shape,
            });
        }
        if !saved.input_stamp.as_ref().is_some_and(|s| s.matches(input))
            && !same_values(&saved.inputs[0], input)?
        {
            return Err(TensorError::InvalidValue {
                label: "sequential_forward_input_mismatch",
            });
        }
        let mut index = 0;
        self.visit_parameters(&mut |parameter| {
            let Some(expected) = saved.parameters.get(index) else {
                return Err(TensorError::InvalidValue {
                    label: "sequential_forward_parameters_changed",
                });
            };
            if !expected.matches(parameter)? {
                return Err(TensorError::InvalidValue {
                    label: "sequential_forward_parameters_changed",
                });
            }
            index += 1;
            Ok(())
        })?;
        if index != saved.parameters.len() {
            return Err(TensorError::InvalidValue {
                label: "sequential_forward_parameters_changed",
            });
        }
        let result = (|| {
            let mut grad = grad_output.clone();
            for (idx, layer) in self.layers.iter_mut().enumerate().rev() {
                grad = layer.backward_retained(&saved.inputs[idx], &grad)?;
            }
            Ok(grad)
        })();
        if result.is_err() {
            // A child may have consumed state or accumulated partial gradients.
            self.last_forward.get_mut().take();
        }
        result
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

    fn zero_accumulators(&mut self) -> PureResult<()> {
        for layer in &mut self.layers {
            layer.zero_accumulators()?;
        }
        Ok(())
    }

    fn scale_learning_rates(&mut self, factor: f32) -> PureResult<()> {
        crate::optim::validate_module_learning_rate_scale(self, factor)?;
        for layer in &mut self.layers {
            layer.scale_learning_rates(factor)?;
        }
        Ok(())
    }

    fn visit_parameters_mut(
        &mut self,
        visitor: &mut dyn FnMut(&mut Parameter) -> PureResult<()>,
    ) -> PureResult<()> {
        self.last_forward.get_mut().take();
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
        self.last_forward.get_mut().take();
        for layer in &mut self.layers {
            layer.load_state_dict(state)?;
        }
        Ok(())
    }

    fn infuse_text(&mut self, text: &str) -> PureResult<()> {
        self.last_forward.get_mut().take();
        for layer in &mut self.layers {
            layer.infuse_text(text)?;
        }
        Ok(())
    }

    fn set_training(&mut self, training: bool) -> PureResult<()> {
        self.last_forward.get_mut().take();
        for layer in &mut self.layers {
            layer.set_training(training)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "sequential_capture_tests.rs"]
mod capture_tests;

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
