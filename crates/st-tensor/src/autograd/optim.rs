//! Atomic SGD for immutable autodiff leaves, using the existing tensor update kernel.

use super::{backward_lock, lock_gradient, AutogradOperation, AutogradTensor};
use crate::{PureResult, TensorError, TensorUtilBackend};
use std::collections::HashSet;

/// Plain CPU SGD over a collection of immutable trainable leaves.
///
/// Fetch `parameters()` again for each forward pass: a successful step replaces
/// the whole collection with fresh leaves whose gradients are unset. Old handles,
/// graphs and accumulated gradients remain valid and unchanged. This is not an
/// in-place PyTorch-style parameter optimizer or a Z-space tape.
///
/// Every registered parameter must have a gradient, including unused parameters
/// (which require an explicit zero gradient). Missing gradients and overflow fail
/// the entire step; they never silently skip a parameter. No momentum, clipping,
/// decay, or automatic gradient averaging is applied.
#[derive(Debug)]
pub struct AutogradSgd {
    parameters: Vec<AutogradTensor>,
    learning_rate: f32,
}

impl AutogradSgd {
    pub fn new(parameters: Vec<AutogradTensor>, learning_rate: f32) -> PureResult<Self> {
        validate_learning_rate(learning_rate)?;
        let mut seen = HashSet::new();
        for parameter in &parameters {
            validate_parameter(parameter)?;
            if !seen.insert(parameter.id()) {
                return Err(TensorError::Generic(
                    "autograd SGD parameters must be unique leaves".into(),
                ));
            }
        }
        Ok(Self {
            parameters,
            learning_rate,
        })
    }

    /// Registers a borrowed leaf without consuming the caller's graph handle.
    pub fn add_parameter(&mut self, parameter: &AutogradTensor) -> PureResult<usize> {
        validate_parameter(parameter)?;
        if self.parameters.iter().any(|existing| existing == parameter) {
            return Err(TensorError::Generic(
                "autograd SGD parameters must be unique leaves".into(),
            ));
        }
        let index = self.parameters.len();
        self.parameters.push(parameter.clone());
        Ok(index)
    }

    pub fn parameters(&self) -> &[AutogradTensor] {
        &self.parameters
    }

    pub fn parameter(&self, index: usize) -> PureResult<AutogradTensor> {
        self.parameters.get(index).cloned().ok_or_else(|| {
            TensorError::Generic(format!(
                "autograd SGD parameter index {index} is out of bounds"
            ))
        })
    }

    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    pub fn set_learning_rate(&mut self, learning_rate: f32) -> PureResult<()> {
        validate_learning_rate(learning_rate)?;
        self.learning_rate = learning_rate;
        Ok(())
    }

    /// Clears all current leaf gradients together, serialized against backward.
    pub fn zero_grad(&self) {
        let _serial = backward_lock();
        for parameter in &self.parameters {
            *lock_gradient(&parameter.node.gradient) = None;
        }
    }

    /// Snapshots all gradients together and publishes only a complete update.
    ///
    /// Backward calls after the snapshot still belong to the old immutable
    /// leaves. Callers must finish gradient accumulation before stepping.
    pub fn step(&mut self) -> PureResult<()> {
        if self.parameters.is_empty() {
            return Err(TensorError::EmptyInput("autograd SGD parameters"));
        }
        let _deferred_observers = crate::observability::defer_tensor_observers();
        let gradients = {
            let _serial = backward_lock();
            self.parameters
                .iter()
                .enumerate()
                .map(|(index, parameter)| {
                    parameter.gradient_locked().ok_or_else(|| {
                        TensorError::Generic(format!(
                            "autograd SGD parameter {index} has no gradient"
                        ))
                    })
                })
                .collect::<PureResult<Vec<_>>>()?
        };
        let mut prepared = Vec::with_capacity(self.parameters.len());
        for (parameter, gradient) in self.parameters.iter().zip(gradients) {
            let mut value = parameter.value().clone();
            value.add_scaled_with_backend(
                &gradient,
                -self.learning_rate,
                TensorUtilBackend::Cpu,
            )?;
            prepared.push(AutogradTensor::variable(value)?);
        }
        self.parameters = prepared;
        crate::emit_tensor_op(
            "autograd_sgd_step",
            &[self.parameters.len()],
            &[self.parameters.len()],
        );
        crate::emit_tensor_op_meta("autograd_sgd_step", || {
            serde_json::json!({
                "semantic_owner": super::AUTOGRAD_SEMANTIC_OWNER,
                "backend": "cpu",
                "requested_backend": "cpu",
                "parameter_count": self.parameters.len(),
                "learning_rate": self.learning_rate,
                "update_rule": "fresh_leaves;all_or_nothing;old_graphs_unchanged",
            })
        });
        Ok(())
    }
}

fn validate_learning_rate(rate: f32) -> PureResult<()> {
    if !rate.is_finite() || rate <= 0.0 {
        return Err(TensorError::NonPositiveLearningRate { rate });
    }
    Ok(())
}

fn validate_parameter(parameter: &AutogradTensor) -> PureResult<()> {
    if !parameter.requires_grad() || !matches!(parameter.node.operation, AutogradOperation::Leaf) {
        return Err(TensorError::Generic(
            "autograd SGD parameters must be trainable leaves".into(),
        ));
    }
    Ok(())
}
