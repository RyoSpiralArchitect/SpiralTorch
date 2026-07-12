// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Pluggable operator registry system.
//!
//! This module provides a framework for registering custom tensor operators
//! that integrate seamlessly with the SpiralTorch ecosystem.

use crate::plugin::{global_registry, PluginEvent};
use crate::PureResult;
use st_tensor::{Tensor, TensorError};
use std::collections::HashMap;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

fn read_recover<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    match lock.read() {
        Ok(guard) => guard,
        Err(poisoned) => {
            let guard = poisoned.into_inner();
            lock.clear_poison();
            guard
        }
    }
}

fn write_recover<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    match lock.write() {
        Ok(guard) => guard,
        Err(poisoned) => {
            let guard = poisoned.into_inner();
            lock.clear_poison();
            guard
        }
    }
}

/// Signature describing an operator's inputs and outputs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OperatorSignature {
    /// Operator name
    pub name: String,
    /// Number of input tensors, or `0` to accept a dynamic count.
    pub num_inputs: usize,
    /// Number of output tensors, or `0` to accept a dynamic count.
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
pub type GradientFn =
    Arc<dyn Fn(&[&Tensor], &[&Tensor], &[&Tensor]) -> PureResult<Vec<Tensor>> + Send + Sync>;

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

    /// Execute the operator and enforce its declared nonzero input/output counts.
    pub fn execute(&self, inputs: &[&Tensor]) -> PureResult<Vec<Tensor>> {
        if self.metadata.signature.num_inputs > 0
            && inputs.len() != self.metadata.signature.num_inputs
        {
            return Err(TensorError::Generic(format!(
                "Operator '{}' expects {} inputs, got {}",
                self.metadata.signature.name,
                self.metadata.signature.num_inputs,
                inputs.len()
            )));
        }

        let outputs = (self.forward_fn)(inputs)?;
        if self.metadata.signature.num_outputs > 0
            && outputs.len() != self.metadata.signature.num_outputs
        {
            return Err(TensorError::Generic(format!(
                "Operator '{}' expects {} outputs, got {}",
                self.metadata.signature.name,
                self.metadata.signature.num_outputs,
                outputs.len()
            )));
        }

        let input_shape = inputs
            .first()
            .map(|tensor| {
                let (rows, cols) = tensor.shape();
                vec![rows, cols]
            })
            .unwrap_or_default();
        let output_shape = outputs
            .first()
            .map(|tensor| {
                let (rows, cols) = tensor.shape();
                vec![rows, cols]
            })
            .unwrap_or_default();
        global_registry()
            .event_bus()
            .publish(&PluginEvent::TensorOp {
                op_name: self.metadata.signature.name.clone(),
                input_shape,
                output_shape,
            });

        Ok(outputs)
    }

    /// Compute gradients and enforce the operator's declared nonzero tensor counts.
    pub fn backward(
        &self,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        grad_outputs: &[&Tensor],
    ) -> PureResult<Vec<Tensor>> {
        if self.metadata.signature.num_inputs > 0
            && inputs.len() != self.metadata.signature.num_inputs
        {
            return Err(TensorError::Generic(format!(
                "Operator '{}' backward expects {} inputs, got {}",
                self.metadata.signature.name,
                self.metadata.signature.num_inputs,
                inputs.len()
            )));
        }
        if self.metadata.signature.num_outputs > 0
            && outputs.len() != self.metadata.signature.num_outputs
        {
            return Err(TensorError::Generic(format!(
                "Operator '{}' backward expects {} outputs, got {}",
                self.metadata.signature.name,
                self.metadata.signature.num_outputs,
                outputs.len()
            )));
        }
        if self.metadata.signature.num_outputs > 0
            && grad_outputs.len() != self.metadata.signature.num_outputs
        {
            return Err(TensorError::Generic(format!(
                "Operator '{}' backward expects {} output gradients, got {}",
                self.metadata.signature.name,
                self.metadata.signature.num_outputs,
                grad_outputs.len()
            )));
        }

        if let Some(backward_fn) = &self.backward_fn {
            let gradients = backward_fn(inputs, outputs, grad_outputs)?;
            if self.metadata.signature.num_inputs > 0
                && gradients.len() != self.metadata.signature.num_inputs
            {
                return Err(TensorError::Generic(format!(
                    "Operator '{}' backward expects {} input gradients, got {}",
                    self.metadata.signature.name,
                    self.metadata.signature.num_inputs,
                    gradients.len()
                )));
            }
            Ok(gradients)
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
        let removed = {
            let mut operators = write_recover(&self.operators);
            operators
                .drain()
                .map(|(_, operator)| operator)
                .collect::<Vec<_>>()
        };
        drop(removed);
    }

    /// Register a new operator.
    pub fn register(&self, operator: RegisteredOperator) -> PureResult<()> {
        let name = operator.metadata.signature.name.clone();
        let mut operators = write_recover(&self.operators);

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
        read_recover(&self.operators).get(name).cloned()
    }

    /// List all registered operators in lexicographic order.
    pub fn list_operators(&self) -> Vec<String> {
        let mut operators = read_recover(&self.operators)
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        operators.sort_unstable();
        operators
    }

    /// Unregister an operator by name.
    pub fn unregister(&self, name: &str) -> bool {
        let removed = {
            let mut operators = write_recover(&self.operators);
            operators.remove(name)
        };
        removed.is_some()
    }

    /// Find operators by backend support, ordered lexicographically by name.
    pub fn find_by_backend(&self, backend: &str) -> Vec<Arc<RegisteredOperator>> {
        let mut operators = read_recover(&self.operators)
            .values()
            .filter(|op| {
                op.metadata
                    .backends
                    .iter()
                    .any(|candidate| candidate == backend)
            })
            .cloned()
            .collect::<Vec<_>>();
        operators.sort_unstable_by(|left, right| {
            left.metadata
                .signature
                .name
                .cmp(&right.metadata.signature.name)
        });
        operators
    }

    /// Execute an operator by name.
    pub fn execute(&self, name: &str, inputs: &[&Tensor]) -> PureResult<Vec<Tensor>> {
        let operator = self
            .get(name)
            .ok_or_else(|| TensorError::Generic(format!("Operator '{}' not found", name)))?;

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
        let forward_fn = self
            .forward_fn
            .ok_or_else(|| TensorError::Generic("Forward function is required".to_string()))?;
        if self.signature.differentiable && self.backward_fn.is_none() {
            return Err(TensorError::Generic(format!(
                "Differentiable operator '{}' requires a backward function",
                self.signature.name
            )));
        }

        let metadata = OperatorMetadata {
            signature: self.signature,
            description: self.description,
            backends: self.backends,
            attributes: self.attributes,
        };

        Ok(RegisteredOperator::new(
            metadata,
            forward_fn,
            self.backward_fn,
        ))
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
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Weak;

    fn identity_operator(name: &str) -> RegisteredOperator {
        OperatorBuilder::new(name, 1, 1)
            .with_backend("CPU")
            .with_forward(Arc::new(|inputs| Ok(vec![inputs[0].clone()])))
            .build()
            .unwrap()
    }

    fn poison_registry(registry: &OperatorRegistry) {
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _guard = registry.operators.write().unwrap();
            panic!("poison operator registry");
        }));
        assert!(result.is_err());
        assert!(registry.operators.is_poisoned());
    }

    struct RegistryDropProbe {
        registry: Weak<OperatorRegistry>,
        observed_unlocked: Arc<AtomicBool>,
    }

    impl Drop for RegistryDropProbe {
        fn drop(&mut self) {
            let Some(registry) = self.registry.upgrade() else {
                return;
            };
            self.observed_unlocked
                .store(registry.operators.try_read().is_ok(), Ordering::SeqCst);
        }
    }

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

    #[test]
    fn operator_execution_enforces_declared_output_count() {
        let operator = OperatorBuilder::new("bad_outputs", 1, 2)
            .with_forward(Arc::new(|inputs| Ok(vec![inputs[0].clone()])))
            .build()
            .unwrap();
        let input = Tensor::from_vec(1, 1, vec![1.0]).unwrap();

        let error = operator.execute(&[&input]).unwrap_err();
        assert!(error.to_string().contains("expects 2 outputs, got 1"));
    }

    #[test]
    fn differentiable_operator_requires_backward_function() {
        let error = OperatorBuilder::new("missing_backward", 1, 1)
            .with_differentiable(true)
            .with_forward(Arc::new(|inputs| Ok(vec![inputs[0].clone()])))
            .build()
            .err()
            .expect("a differentiable operator without backward must be rejected");

        assert!(error
            .to_string()
            .contains("Differentiable operator 'missing_backward' requires a backward function"));
    }

    #[test]
    fn operator_backward_enforces_declared_tensor_counts() {
        let operator = OperatorBuilder::new("bad_backward", 1, 1)
            .with_forward(Arc::new(|inputs| Ok(vec![inputs[0].clone()])))
            .with_backward(Arc::new(|_, _, _| Ok(Vec::new())))
            .build()
            .unwrap();
        let tensor = Tensor::from_vec(1, 1, vec![1.0]).unwrap();

        let error = operator.backward(&[], &[&tensor], &[&tensor]).unwrap_err();
        assert!(error
            .to_string()
            .contains("backward expects 1 inputs, got 0"));

        let error = operator.backward(&[&tensor], &[], &[&tensor]).unwrap_err();
        assert!(error
            .to_string()
            .contains("backward expects 1 outputs, got 0"));

        let error = operator.backward(&[&tensor], &[&tensor], &[]).unwrap_err();
        assert!(error
            .to_string()
            .contains("backward expects 1 output gradients, got 0"));

        let error = operator
            .backward(&[&tensor], &[&tensor], &[&tensor])
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("backward expects 1 input gradients, got 0"));
    }

    #[test]
    fn zero_arity_signatures_remain_dynamic() {
        let operator = OperatorBuilder::new("dynamic", 0, 0)
            .with_forward(Arc::new(|inputs| {
                Ok(inputs.iter().map(|input| (*input).clone()).collect())
            }))
            .build()
            .unwrap();
        let first = Tensor::from_vec(1, 1, vec![1.0]).unwrap();
        let second = Tensor::from_vec(1, 1, vec![2.0]).unwrap();

        let outputs = operator.execute(&[&first, &second]).unwrap();
        assert_eq!(outputs.len(), 2);
    }

    #[test]
    fn poisoned_registry_recovers_across_read_and_write_operations() {
        let registry = OperatorRegistry::new();

        poison_registry(&registry);
        registry.register(identity_operator("recoverable")).unwrap();
        assert!(!registry.operators.is_poisoned());

        poison_registry(&registry);
        assert!(registry.get("recoverable").is_some());
        assert!(!registry.operators.is_poisoned());

        poison_registry(&registry);
        assert!(registry.unregister("recoverable"));
        assert!(!registry.operators.is_poisoned());
    }

    #[test]
    fn unregister_drops_operator_callbacks_outside_write_lock() {
        let registry = Arc::new(OperatorRegistry::new());
        let observed_unlocked = Arc::new(AtomicBool::new(false));
        let probe = RegistryDropProbe {
            registry: Arc::downgrade(&registry),
            observed_unlocked: observed_unlocked.clone(),
        };
        let operator = OperatorBuilder::new("drop_probe", 0, 0)
            .with_forward(Arc::new(move |_| {
                let _keep_probe_alive = &probe;
                Ok(Vec::new())
            }))
            .build()
            .unwrap();
        registry.register(operator).unwrap();

        assert!(registry.unregister("drop_probe"));
        assert!(observed_unlocked.load(Ordering::SeqCst));
    }

    #[test]
    fn operator_discovery_is_deterministic() {
        let registry = OperatorRegistry::new();
        registry.register(identity_operator("zeta")).unwrap();
        registry.register(identity_operator("alpha")).unwrap();

        assert_eq!(registry.list_operators(), vec!["alpha", "zeta"]);
        let cpu_names = registry
            .find_by_backend("CPU")
            .into_iter()
            .map(|operator| operator.metadata.signature.name.clone())
            .collect::<Vec<_>>();
        assert_eq!(cpu_names, vec!["alpha", "zeta"]);
    }
}
