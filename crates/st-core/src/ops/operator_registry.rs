// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Pluggable operator registry system.
//!
//! This module provides a framework for registering custom tensor operators
//! that integrate seamlessly with the SpiralTorch ecosystem.

use crate::PureResult;
use st_tensor::{Tensor, TensorError};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Signature describing an operator's inputs and outputs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OperatorSignature {
    /// Operator name
    pub name: String,
    /// Number of input tensors
    pub num_inputs: usize,
    /// Number of output tensors
    pub num_outputs: usize,
    /// Whether operator supports in-place execution
    pub supports_inplace: bool,
    /// Whether operator supports gradient computation
    pub differentiable: bool,
}

/// Metadata for a registered operator.
#[derive(Debug, Clone)]
pub struct OperatorMetadata {
    /// Operator signature
    pub signature: OperatorSignature,
    /// Human-readable description
    pub description: String,
    /// Supported backends
    pub backends: Vec<String>,
    /// Custom attributes
    pub attributes: HashMap<String, String>,
}

/// Type alias for operator execution function.
pub type OperatorFn = Arc<dyn Fn(&[&Tensor]) -> PureResult<Vec<Tensor>> + Send + Sync>;

/// Type alias for gradient computation function.
pub type GradientFn = Arc<dyn Fn(&[&Tensor], &[&Tensor], &[&Tensor]) -> PureResult<Vec<Tensor>> + Send + Sync>;

/// A registered operator with its implementation.
pub struct RegisteredOperator {
    metadata: OperatorMetadata,
    forward_fn: OperatorFn,
    backward_fn: Option<GradientFn>,
}

impl RegisteredOperator {
    /// Create a new registered operator.
    pub fn new(
        metadata: OperatorMetadata,
        forward_fn: OperatorFn,
        backward_fn: Option<GradientFn>,
    ) -> Self {
        Self {
            metadata,
            forward_fn,
            backward_fn,
        }
    }

    /// Execute the operator.
    pub fn execute(&self, inputs: &[&Tensor]) -> PureResult<Vec<Tensor>> {
        if inputs.len() != self.metadata.signature.num_inputs {
            return Err(TensorError::Generic(format!(
                "Operator '{}' expects {} inputs, got {}",
                self.metadata.signature.name,
                self.metadata.signature.num_inputs,
                inputs.len()
            )));
        }

        (self.forward_fn)(inputs)
    }

    /// Compute gradients.
    pub fn backward(
        &self,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        grad_outputs: &[&Tensor],
    ) -> PureResult<Vec<Tensor>> {
        if let Some(backward_fn) = &self.backward_fn {
            backward_fn(inputs, outputs, grad_outputs)
        } else {
            Err(TensorError::Generic(format!(
                "Operator '{}' does not support gradient computation",
                self.metadata.signature.name
            )))
        }
    }

    /// Get operator metadata.
    pub fn metadata(&self) -> &OperatorMetadata {
        &self.metadata
    }
}

/// Global registry for custom operators.
pub struct OperatorRegistry {
    operators: RwLock<HashMap<String, Arc<RegisteredOperator>>>,
}

impl OperatorRegistry {
    /// Create a new operator registry.
    pub fn new() -> Self {
        Self {
            operators: RwLock::new(HashMap::new()),
        }
    }

    /// Clear all registered operators (useful for testing).
    #[cfg(test)]
    pub fn clear(&self) {
        self.operators.write().unwrap().clear();
    }

    /// Register a new operator.
    pub fn register(&self, operator: RegisteredOperator) -> PureResult<()> {
        let name = operator.metadata.signature.name.clone();
        let mut operators = self.operators.write().unwrap();

        if operators.contains_key(&name) {
            return Err(TensorError::Generic(format!(
                "Operator '{}' is already registered",
                name
            )));
        }

        operators.insert(name, Arc::new(operator));
        Ok(())
    }

    /// Get a registered operator by name.
    pub fn get(&self, name: &str) -> Option<Arc<RegisteredOperator>> {
        self.operators.read().unwrap().get(name).cloned()
    }

    /// List all registered operators.
    pub fn list_operators(&self) -> Vec<String> {
        self.operators.read().unwrap().keys().cloned().collect()
    }

    /// Find operators by backend support.
    pub fn find_by_backend(&self, backend: &str) -> Vec<Arc<RegisteredOperator>> {
        self.operators
            .read()
            .unwrap()
            .values()
            .filter(|op| op.metadata.backends.contains(&backend.to_string()))
            .cloned()
            .collect()
    }

    /// Execute an operator by name.
    pub fn execute(&self, name: &str, inputs: &[&Tensor]) -> PureResult<Vec<Tensor>> {
        let operator = self.get(name).ok_or_else(|| {
            TensorError::Generic(format!("Operator '{}' not found", name))
        })?;

        operator.execute(inputs)
    }
}

impl Default for OperatorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating operator metadata.
pub struct OperatorBuilder {
    signature: OperatorSignature,
    description: String,
    backends: Vec<String>,
    attributes: HashMap<String, String>,
    forward_fn: Option<OperatorFn>,
    backward_fn: Option<GradientFn>,
}

impl OperatorBuilder {
    /// Create a new operator builder.
    pub fn new(name: impl Into<String>, num_inputs: usize, num_outputs: usize) -> Self {
        Self {
            signature: OperatorSignature {
                name: name.into(),
                num_inputs,
                num_outputs,
                supports_inplace: false,
                differentiable: false,
            },
            description: String::new(),
            backends: Vec::new(),
            attributes: HashMap::new(),
            forward_fn: None,
            backward_fn: None,
        }
    }

    /// Set the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Add a supported backend.
    pub fn with_backend(mut self, backend: impl Into<String>) -> Self {
        self.backends.push(backend.into());
        self
    }

    /// Set whether the operator supports in-place execution.
    pub fn with_inplace(mut self, inplace: bool) -> Self {
        self.signature.supports_inplace = inplace;
        self
    }

    /// Set whether the operator is differentiable.
    pub fn with_differentiable(mut self, diff: bool) -> Self {
        self.signature.differentiable = diff;
        self
    }

    /// Add a custom attribute.
    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// Set the forward function.
    pub fn with_forward(mut self, forward_fn: OperatorFn) -> Self {
        self.forward_fn = Some(forward_fn);
        self
    }

    /// Set the backward function.
    pub fn with_backward(mut self, backward_fn: GradientFn) -> Self {
        self.backward_fn = Some(backward_fn);
        self.signature.differentiable = true;
        self
    }

    /// Build the registered operator.
    pub fn build(self) -> PureResult<RegisteredOperator> {
        let forward_fn = self.forward_fn.ok_or_else(|| {
            TensorError::Generic("Forward function is required".to_string())
        })?;

        let metadata = OperatorMetadata {
            signature: self.signature,
            description: self.description,
            backends: self.backends,
            attributes: self.attributes,
        };

        Ok(RegisteredOperator::new(metadata, forward_fn, self.backward_fn))
    }
}

/// Global operator registry instance.
static GLOBAL_OPERATOR_REGISTRY: std::sync::OnceLock<OperatorRegistry> = std::sync::OnceLock::new();

/// Get the global operator registry.
pub fn global_operator_registry() -> &'static OperatorRegistry {
    GLOBAL_OPERATOR_REGISTRY.get_or_init(OperatorRegistry::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operator_registration() {
        let registry = OperatorRegistry::new();

        let operator = OperatorBuilder::new("test_op", 2, 1)
            .with_description("A test operator")
            .with_backend("CPU")
            .with_forward(Arc::new(|inputs| {
                // Simple identity operation for testing
                Ok(vec![inputs[0].clone()])
            }))
            .build()
            .unwrap();

        assert!(registry.register(operator).is_ok());
        assert!(registry.get("test_op").is_some());
    }

    #[test]
    fn test_find_by_backend() {
        let registry = OperatorRegistry::new();

        let op1 = OperatorBuilder::new("op1", 1, 1)
            .with_backend("CPU")
            .with_forward(Arc::new(|inputs| Ok(vec![inputs[0].clone()])))
            .build()
            .unwrap();

        let op2 = OperatorBuilder::new("op2", 1, 1)
            .with_backend("CUDA")
            .with_forward(Arc::new(|inputs| Ok(vec![inputs[0].clone()])))
            .build()
            .unwrap();

        registry.register(op1).unwrap();
        registry.register(op2).unwrap();

        let cpu_ops = registry.find_by_backend("CPU");
        assert_eq!(cpu_ops.len(), 1);
        assert_eq!(cpu_ops[0].metadata().signature.name, "op1");
    }
}
