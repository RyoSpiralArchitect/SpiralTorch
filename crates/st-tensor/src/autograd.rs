// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2025 Ryo SpiralArchitect

//! Reverse-mode automatic differentiation over [`Tensor`].
//!
//! The graph is immutable, gradients are accumulated explicitly, and a failed
//! backward pass never commits a partial gradient set. Python and WASM clients
//! should expose this contract rather than rebuilding graph semantics.

use crate::{PureResult, Tensor, TensorError};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

/// Versioned semantic contract shared by native and binding surfaces.
pub const AUTOGRAD_CONTRACT_VERSION: &str = "spiraltorch.autograd.v1";
/// Crate that owns graph construction and reverse-mode semantics.
pub const AUTOGRAD_SEMANTIC_OWNER: &str = "st-tensor";

static NEXT_NODE_ID: AtomicU64 = AtomicU64::new(1);
static BACKWARD_SERIALIZER: Mutex<()> = Mutex::new(());

/// Structural summary of one connected autodiff graph.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AutogradGraphSummary {
    pub node_count: usize,
    pub operation_count: usize,
    pub leaf_count: usize,
    pub trainable_leaf_count: usize,
    pub max_depth: usize,
}

/// Receipt emitted after an atomic reverse-mode pass commits.
#[derive(Clone, Debug, PartialEq)]
pub struct AutogradBackwardReport {
    pub graph: AutogradGraphSummary,
    pub gradient_node_count: usize,
    pub leaf_gradient_count: usize,
    pub seed_l2: f32,
}

impl AutogradGraphSummary {
    fn contract_map(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut payload = serde_json::Map::new();
        payload.insert(
            "contract_version".to_owned(),
            AUTOGRAD_CONTRACT_VERSION.into(),
        );
        payload.insert("semantic_owner".to_owned(), AUTOGRAD_SEMANTIC_OWNER.into());
        payload.insert("node_count".to_owned(), self.node_count.into());
        payload.insert("operation_count".to_owned(), self.operation_count.into());
        payload.insert("leaf_count".to_owned(), self.leaf_count.into());
        payload.insert(
            "trainable_leaf_count".to_owned(),
            self.trainable_leaf_count.into(),
        );
        payload.insert("max_depth".to_owned(), self.max_depth.into());
        payload
    }

    /// Returns the canonical binding and telemetry payload for this graph.
    pub fn contract_payload(&self) -> serde_json::Value {
        serde_json::Value::Object(self.contract_map())
    }
}

impl AutogradBackwardReport {
    /// Returns the canonical binding and telemetry payload for this pass.
    pub fn contract_payload(&self) -> serde_json::Value {
        let mut payload = self.graph.contract_map();
        payload.insert(
            "gradient_node_count".to_owned(),
            self.gradient_node_count.into(),
        );
        payload.insert(
            "leaf_gradient_count".to_owned(),
            self.leaf_gradient_count.into(),
        );
        payload.insert("seed_l2".to_owned(), self.seed_l2.into());
        serde_json::Value::Object(payload)
    }
}

/// A tensor value attached to an immutable reverse-mode graph node.
#[derive(Clone)]
pub struct AutogradTensor {
    node: Arc<AutogradNode>,
}

struct AutogradNode {
    id: u64,
    value: Tensor,
    requires_grad: bool,
    gradient: Mutex<Option<Tensor>>,
    operation: AutogradOperation,
}

#[derive(Clone)]
enum AutogradOperation {
    Leaf,
    Add {
        lhs: AutogradTensor,
        rhs: AutogradTensor,
    },
    Sub {
        lhs: AutogradTensor,
        rhs: AutogradTensor,
    },
    Hadamard {
        lhs: AutogradTensor,
        rhs: AutogradTensor,
    },
    Matmul {
        lhs: AutogradTensor,
        rhs: AutogradTensor,
    },
    Scale {
        input: AutogradTensor,
        factor: f32,
    },
    Transpose {
        input: AutogradTensor,
    },
    Sum {
        input: AutogradTensor,
    },
    Mean {
        input: AutogradTensor,
    },
}

impl fmt::Debug for AutogradTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AutogradTensor")
            .field("id", &self.id())
            .field("shape", &self.shape())
            .field("requires_grad", &self.requires_grad())
            .field("operation", &self.operation_name())
            .finish()
    }
}

impl PartialEq for AutogradTensor {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl Eq for AutogradTensor {}

impl AutogradOperation {
    fn name(&self) -> &'static str {
        match self {
            Self::Leaf => "leaf",
            Self::Add { .. } => "add",
            Self::Sub { .. } => "sub",
            Self::Hadamard { .. } => "hadamard",
            Self::Matmul { .. } => "matmul",
            Self::Scale { .. } => "scale",
            Self::Transpose { .. } => "transpose",
            Self::Sum { .. } => "sum",
            Self::Mean { .. } => "mean",
        }
    }

    fn parents(&self) -> Vec<AutogradTensor> {
        match self {
            Self::Leaf => Vec::new(),
            Self::Add { lhs, rhs }
            | Self::Sub { lhs, rhs }
            | Self::Hadamard { lhs, rhs }
            | Self::Matmul { lhs, rhs } => vec![lhs.clone(), rhs.clone()],
            Self::Scale { input, .. }
            | Self::Transpose { input }
            | Self::Sum { input }
            | Self::Mean { input } => vec![input.clone()],
        }
    }

    fn backward(&self, upstream: &Tensor) -> PureResult<Vec<(AutogradTensor, Tensor)>> {
        match self {
            Self::Leaf => Ok(Vec::new()),
            Self::Add { lhs, rhs } => Ok(vec![
                (lhs.clone(), upstream.clone()),
                (rhs.clone(), upstream.clone()),
            ]),
            Self::Sub { lhs, rhs } => Ok(vec![
                (lhs.clone(), upstream.clone()),
                (rhs.clone(), upstream.scale(-1.0)?),
            ]),
            Self::Hadamard { lhs, rhs } => Ok(vec![
                (lhs.clone(), upstream.hadamard(rhs.value())?),
                (rhs.clone(), upstream.hadamard(lhs.value())?),
            ]),
            Self::Matmul { lhs, rhs } => {
                let rhs_t = rhs.value().transpose();
                let lhs_t = lhs.value().transpose();
                Ok(vec![
                    (lhs.clone(), upstream.matmul(&rhs_t)?),
                    (rhs.clone(), lhs_t.matmul(upstream)?),
                ])
            }
            Self::Scale { input, factor } => Ok(vec![(input.clone(), upstream.scale(*factor)?)]),
            Self::Transpose { input } => Ok(vec![(input.clone(), upstream.transpose())]),
            Self::Sum { input } => {
                let seed = scalar_value(upstream, "autograd_sum_seed")?;
                let gradient = Tensor::from_vec(
                    input.shape().0,
                    input.shape().1,
                    vec![seed; input.value().len()],
                )?;
                Ok(vec![(input.clone(), gradient)])
            }
            Self::Mean { input } => {
                let seed = scalar_value(upstream, "autograd_mean_seed")?;
                let len = input.value().len();
                if len == 0 {
                    return Err(TensorError::EmptyInput("autograd mean input"));
                }
                let gradient = Tensor::from_vec(
                    input.shape().0,
                    input.shape().1,
                    vec![seed / len as f32; len],
                )?;
                Ok(vec![(input.clone(), gradient)])
            }
        }
    }
}

impl AutogradTensor {
    /// Creates an immutable leaf. Trainable leaves accumulate gradients.
    pub fn from_tensor(value: Tensor, requires_grad: bool) -> PureResult<Self> {
        validate_finite_tensor("autograd_leaf", &value)?;
        Ok(Self::from_node(
            value,
            requires_grad,
            AutogradOperation::Leaf,
        ))
    }

    /// Creates a trainable leaf.
    pub fn variable(value: Tensor) -> PureResult<Self> {
        Self::from_tensor(value, true)
    }

    /// Creates a non-trainable leaf.
    pub fn constant(value: Tensor) -> PureResult<Self> {
        Self::from_tensor(value, false)
    }

    fn from_node(value: Tensor, requires_grad: bool, operation: AutogradOperation) -> Self {
        Self {
            node: Arc::new(AutogradNode {
                id: NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed),
                value,
                requires_grad,
                gradient: Mutex::new(None),
                operation,
            }),
        }
    }

    fn from_operation(value: Tensor, operation: AutogradOperation) -> PureResult<Self> {
        validate_finite_tensor("autograd_forward", &value)?;
        let requires_grad = operation
            .parents()
            .iter()
            .any(AutogradTensor::requires_grad);
        if requires_grad {
            Ok(Self::from_node(value, true, operation))
        } else {
            Ok(Self::from_node(value, false, AutogradOperation::Leaf))
        }
    }

    #[inline]
    pub fn id(&self) -> u64 {
        self.node.id
    }

    #[inline]
    pub fn value(&self) -> &Tensor {
        &self.node.value
    }

    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        self.value().shape()
    }

    #[inline]
    pub fn requires_grad(&self) -> bool {
        self.node.requires_grad
    }

    #[inline]
    pub fn operation_name(&self) -> &'static str {
        self.node.operation.name()
    }

    /// Returns a detached leaf that shares the immutable tensor buffer.
    pub fn detach(&self) -> PureResult<Self> {
        Self::constant(self.value().clone())
    }

    /// Returns the currently accumulated gradient, if any.
    pub fn grad(&self) -> Option<Tensor> {
        let _serial = backward_lock();
        self.gradient_locked()
    }

    /// Clears only this node's accumulated gradient.
    pub fn zero_grad(&self) {
        let _serial = backward_lock();
        *lock_gradient(&self.node.gradient) = None;
    }

    /// Clears gradients for every node reachable from this output.
    pub fn zero_grad_graph(&self) {
        let _serial = backward_lock();
        for node in self.topological_order() {
            *lock_gradient(&node.node.gradient) = None;
        }
    }

    pub fn add(&self, rhs: &Self) -> PureResult<Self> {
        let value = self.value().add(rhs.value())?;
        Self::from_operation(
            value,
            AutogradOperation::Add {
                lhs: self.clone(),
                rhs: rhs.clone(),
            },
        )
    }

    pub fn sub(&self, rhs: &Self) -> PureResult<Self> {
        let value = self.value().sub(rhs.value())?;
        Self::from_operation(
            value,
            AutogradOperation::Sub {
                lhs: self.clone(),
                rhs: rhs.clone(),
            },
        )
    }

    pub fn hadamard(&self, rhs: &Self) -> PureResult<Self> {
        let value = self.value().hadamard(rhs.value())?;
        Self::from_operation(
            value,
            AutogradOperation::Hadamard {
                lhs: self.clone(),
                rhs: rhs.clone(),
            },
        )
    }

    pub fn matmul(&self, rhs: &Self) -> PureResult<Self> {
        let value = self.value().matmul(rhs.value())?;
        Self::from_operation(
            value,
            AutogradOperation::Matmul {
                lhs: self.clone(),
                rhs: rhs.clone(),
            },
        )
    }

    pub fn scale(&self, factor: f32) -> PureResult<Self> {
        let value = self.value().scale(factor)?;
        Self::from_operation(
            value,
            AutogradOperation::Scale {
                input: self.clone(),
                factor,
            },
        )
    }

    pub fn transpose(&self) -> PureResult<Self> {
        Self::from_operation(
            self.value().transpose(),
            AutogradOperation::Transpose {
                input: self.clone(),
            },
        )
    }

    pub fn sum(&self) -> PureResult<Self> {
        let value = self.value().data().iter().try_fold(0.0f64, |sum, &value| {
            let next = sum + value as f64;
            next.is_finite()
                .then_some(next)
                .ok_or(TensorError::NonFiniteValue {
                    label: "autograd_sum",
                    value: next as f32,
                })
        })?;
        let value = value as f32;
        if !value.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "autograd_sum",
                value,
            });
        }
        Self::from_operation(
            Tensor::from_vec(1, 1, vec![value])?,
            AutogradOperation::Sum {
                input: self.clone(),
            },
        )
    }

    pub fn mean(&self) -> PureResult<Self> {
        if self.value().is_empty() {
            return Err(TensorError::EmptyInput("autograd mean input"));
        }
        let sum = self
            .value()
            .data()
            .iter()
            .map(|&value| value as f64)
            .sum::<f64>();
        let value = (sum / self.value().len() as f64) as f32;
        if !value.is_finite() {
            return Err(TensorError::NonFiniteValue {
                label: "autograd_mean",
                value,
            });
        }
        Self::from_operation(
            Tensor::from_vec(1, 1, vec![value])?,
            AutogradOperation::Mean {
                input: self.clone(),
            },
        )
    }

    pub fn dot(&self, rhs: &Self) -> PureResult<Self> {
        self.hadamard(rhs)?.sum()
    }

    pub fn mean_squared_error(&self, target: &Self) -> PureResult<Self> {
        let residual = self.sub(target)?;
        residual.hadamard(&residual)?.mean()
    }

    pub fn zeros_like(tensor: &Self) -> PureResult<Self> {
        Self::constant(Tensor::zeros(tensor.shape().0, tensor.shape().1)?)
    }

    pub fn item_f32(&self) -> PureResult<f32> {
        scalar_value(self.value(), "autograd_item")
    }

    /// Runs reverse mode with an implicit scalar seed of one.
    pub fn backward(&self) -> PureResult<AutogradBackwardReport> {
        let (rows, cols) = self.shape();
        if (rows, cols) != (1, 1) {
            return Err(TensorError::NonScalarBackward { rows, cols });
        }
        let seed = Tensor::from_vec(1, 1, vec![1.0])?;
        self.backward_with_grad(&seed)
    }

    /// Runs an atomic vector-Jacobian product with an explicit output seed.
    pub fn backward_with_grad(&self, seed: &Tensor) -> PureResult<AutogradBackwardReport> {
        if seed.shape() != self.shape() {
            return Err(TensorError::ShapeMismatch {
                left: seed.shape(),
                right: self.shape(),
            });
        }
        validate_finite_tensor("autograd_backward_seed", seed)?;
        let report = {
            let _serial = backward_lock();
            self.backward_locked(seed)?
        };
        self.emit_backward_report(seed, &report);
        Ok(report)
    }

    fn backward_locked(&self, seed: &Tensor) -> PureResult<AutogradBackwardReport> {
        let topology = self.topological_order();
        let mut local_gradients = HashMap::<u64, Tensor>::new();
        local_gradients.insert(self.id(), seed.clone());

        for node in topology.iter().rev() {
            let Some(upstream) = local_gradients.get(&node.id()).cloned() else {
                continue;
            };
            for (parent, gradient) in node.node.operation.backward(&upstream)? {
                if !parent.requires_grad() {
                    continue;
                }
                if gradient.shape() != parent.shape() {
                    return Err(TensorError::ShapeMismatch {
                        left: gradient.shape(),
                        right: parent.shape(),
                    });
                }
                validate_finite_tensor("autograd_backward_gradient", &gradient)?;
                if let Some(existing) = local_gradients.get(&parent.id()) {
                    let accumulated = existing.add(&gradient)?;
                    validate_finite_tensor("autograd_backward_accumulation", &accumulated)?;
                    local_gradients.insert(parent.id(), accumulated);
                } else {
                    local_gradients.insert(parent.id(), gradient);
                }
            }
        }

        // Build every next value before mutating any node, preserving failure atomicity.
        let mut prepared = Vec::new();
        for node in &topology {
            if !node.requires_grad() {
                continue;
            }
            let Some(delta) = local_gradients.get(&node.id()) else {
                continue;
            };
            let next = match node.gradient_locked() {
                Some(existing) => existing.add(delta)?,
                None => delta.clone(),
            };
            validate_finite_tensor("autograd_committed_gradient", &next)?;
            prepared.push((node.clone(), next));
        }

        for (node, gradient) in &prepared {
            *lock_gradient(&node.node.gradient) = Some(gradient.clone());
        }

        let graph = self.graph_summary();
        let leaf_gradient_count = prepared
            .iter()
            .filter(|(node, _)| matches!(node.node.operation, AutogradOperation::Leaf))
            .count();
        let report = AutogradBackwardReport {
            graph,
            gradient_node_count: prepared.len(),
            leaf_gradient_count,
            seed_l2: stable_l2(seed),
        };
        Ok(report)
    }

    fn emit_backward_report(&self, seed: &Tensor, report: &AutogradBackwardReport) {
        crate::emit_tensor_op(
            "autograd_backward",
            &[seed.shape().0, seed.shape().1],
            &[self.shape().0, self.shape().1],
        );
        crate::emit_tensor_op_meta("autograd_backward", || report.contract_payload());
    }

    pub fn graph_summary(&self) -> AutogradGraphSummary {
        let mut summary = AutogradGraphSummary::default();
        let mut depths = HashMap::<u64, usize>::new();
        let mut stack = vec![(self.clone(), 0usize)];
        while let Some((node, depth)) = stack.pop() {
            if depths.get(&node.id()).is_some_and(|seen| *seen >= depth) {
                continue;
            }
            let first_visit = depths.insert(node.id(), depth).is_none();
            if first_visit {
                summary.node_count += 1;
                if matches!(node.node.operation, AutogradOperation::Leaf) {
                    summary.leaf_count += 1;
                    if node.requires_grad() {
                        summary.trainable_leaf_count += 1;
                    }
                } else {
                    summary.operation_count += 1;
                }
            }
            summary.max_depth = summary.max_depth.max(depth);
            for parent in node.node.operation.parents() {
                stack.push((parent, depth.saturating_add(1)));
            }
        }
        summary
    }

    fn topological_order(&self) -> Vec<AutogradTensor> {
        let mut topology = Vec::new();
        let mut visited = HashSet::<u64>::new();
        let mut stack = vec![(self.clone(), false)];
        while let Some((node, expanded)) = stack.pop() {
            if expanded {
                topology.push(node);
                continue;
            }
            if !visited.insert(node.id()) {
                continue;
            }
            stack.push((node.clone(), true));
            let mut parents = node.node.operation.parents();
            parents.reverse();
            for parent in parents {
                stack.push((parent, false));
            }
        }
        topology
    }

    fn gradient_locked(&self) -> Option<Tensor> {
        lock_gradient(&self.node.gradient).clone()
    }
}

fn backward_lock() -> MutexGuard<'static, ()> {
    BACKWARD_SERIALIZER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn lock_gradient(gradient: &Mutex<Option<Tensor>>) -> MutexGuard<'_, Option<Tensor>> {
    gradient
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn scalar_value(tensor: &Tensor, label: &'static str) -> PureResult<f32> {
    if tensor.shape() != (1, 1) {
        return Err(TensorError::NonScalarBackward {
            rows: tensor.shape().0,
            cols: tensor.shape().1,
        });
    }
    let value = tensor.data()[0];
    if !value.is_finite() {
        return Err(TensorError::NonFiniteValue { label, value });
    }
    Ok(value)
}

fn validate_finite_tensor(label: &'static str, tensor: &Tensor) -> PureResult<()> {
    if let Some(&value) = tensor.data().iter().find(|value| !value.is_finite()) {
        return Err(TensorError::NonFiniteValue { label, value });
    }
    Ok(())
}

fn stable_l2(tensor: &Tensor) -> f32 {
    tensor
        .data()
        .iter()
        .map(|&value| {
            let value = value as f64;
            value * value
        })
        .sum::<f64>()
        .sqrt()
        .min(f32::MAX as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn tensor(rows: usize, cols: usize, values: Vec<f32>) -> Tensor {
        Tensor::from_vec(rows, cols, values).expect("valid test tensor")
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "gradient mismatch at {index}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn backward_requires_explicit_seed_for_non_scalar_output() {
        let variable = AutogradTensor::variable(tensor(1, 2, vec![1.0, 2.0])).unwrap();
        let error = variable.backward().unwrap_err();
        assert_eq!(error, TensorError::NonScalarBackward { rows: 1, cols: 2 });
        assert!(variable.grad().is_none());
    }

    #[test]
    fn branching_graph_accumulates_every_path_once() {
        let variable = AutogradTensor::variable(tensor(1, 3, vec![-2.0, 0.5, 3.0])).unwrap();
        let square = variable.hadamard(&variable).unwrap();
        let linear = variable.scale(3.0).unwrap();
        let loss = square.add(&linear).unwrap().sum().unwrap();

        let report = loss.backward().unwrap();

        assert_close(variable.grad().unwrap().data(), &[-1.0, 4.0, 9.0], 1e-6);
        assert_eq!(report.graph.trainable_leaf_count, 1);
        assert_eq!(report.leaf_gradient_count, 1);
    }

    #[test]
    fn matmul_backward_matches_closed_form() {
        let lhs =
            AutogradTensor::variable(tensor(2, 3, vec![1.0, 2.0, -1.0, 0.5, -2.0, 3.0])).unwrap();
        let rhs =
            AutogradTensor::variable(tensor(3, 2, vec![2.0, -1.0, 0.0, 3.0, 1.5, 2.0])).unwrap();
        let loss = lhs.matmul(&rhs).unwrap().sum().unwrap();

        loss.backward().unwrap();

        assert_close(
            lhs.grad().unwrap().data(),
            &[1.0, 3.0, 3.5, 1.0, 3.0, 3.5],
            1e-6,
        );
        assert_close(
            rhs.grad().unwrap().data(),
            &[1.5, 1.5, 0.0, 0.0, 2.0, 2.0],
            1e-6,
        );
    }

    #[test]
    fn repeated_backward_accumulates_and_graph_zeroing_clears() {
        let variable = AutogradTensor::variable(tensor(1, 2, vec![2.0, -4.0])).unwrap();
        let loss = variable.hadamard(&variable).unwrap().mean().unwrap();
        loss.backward().unwrap();
        loss.backward().unwrap();
        assert_close(variable.grad().unwrap().data(), &[4.0, -8.0], 1e-6);

        loss.zero_grad_graph();
        assert!(variable.grad().is_none());
        assert!(loss.grad().is_none());
    }

    #[test]
    fn backward_failure_does_not_commit_partial_gradients() {
        let variable = AutogradTensor::variable(tensor(1, 1, vec![1e-38])).unwrap();
        let output = variable.scale(f32::MAX).unwrap();
        let seed = tensor(1, 1, vec![2.0]);

        let error = output.backward_with_grad(&seed).unwrap_err();

        assert!(matches!(error, TensorError::NonFiniteValue { .. }));
        assert!(variable.grad().is_none());
        assert!(output.grad().is_none());
    }

    #[test]
    fn mean_squared_error_matches_finite_difference() {
        let prediction_values = vec![0.5, -0.25, 1.5];
        let target_values = vec![0.1, 0.5, -0.5];
        let prediction = AutogradTensor::variable(tensor(1, 3, prediction_values.clone())).unwrap();
        let target = AutogradTensor::constant(tensor(1, 3, target_values.clone())).unwrap();
        let loss = prediction.mean_squared_error(&target).unwrap();
        loss.backward().unwrap();
        let gradient = prediction.grad().unwrap();

        let epsilon = 1e-3;
        let mut finite_difference = Vec::new();
        for index in 0..prediction_values.len() {
            let mut plus = prediction_values.clone();
            let mut minus = prediction_values.clone();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let objective = |values: &[f32]| {
                values
                    .iter()
                    .zip(&target_values)
                    .map(|(prediction, target)| (prediction - target).powi(2))
                    .sum::<f32>()
                    / values.len() as f32
            };
            finite_difference.push((objective(&plus) - objective(&minus)) / (2.0 * epsilon));
        }
        assert_close(gradient.data(), &finite_difference, 2e-4);
    }

    #[test]
    fn concurrent_backward_calls_accumulate_without_lost_updates() {
        let variable = AutogradTensor::variable(tensor(1, 2, vec![1.0, -2.0])).unwrap();
        let loss = variable.hadamard(&variable).unwrap().sum().unwrap();
        let mut threads = Vec::new();
        for _ in 0..4 {
            let loss = loss.clone();
            threads.push(std::thread::spawn(move || loss.backward().unwrap()));
        }
        for thread in threads {
            thread.join().unwrap();
        }
        assert_close(variable.grad().unwrap().data(), &[8.0, -16.0], 1e-6);
    }

    #[test]
    fn backward_emits_versioned_rust_owned_metadata() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&events);
        let previous = crate::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));
        let variable = AutogradTensor::variable(tensor(1, 1, vec![2.0])).unwrap();
        variable.hadamard(&variable).unwrap().backward().unwrap();
        crate::set_thread_meta_observer(previous);

        let events = events.lock().unwrap();
        let event = events
            .iter()
            .find(|(name, _)| *name == "autograd_backward")
            .expect("autograd metadata event");
        assert_eq!(event.1["contract_version"], AUTOGRAD_CONTRACT_VERSION);
        assert_eq!(event.1["semantic_owner"], AUTOGRAD_SEMANTIC_OWNER);
        assert_eq!(event.1["trainable_leaf_count"], 1);
    }

    #[test]
    fn backward_observer_can_reenter_gradient_reads() {
        let variable = AutogradTensor::variable(tensor(1, 1, vec![2.0])).unwrap();
        let observed_gradient = Arc::new(Mutex::new(None));
        let captured_gradient = Arc::clone(&observed_gradient);
        let observed_variable = variable.clone();
        let previous = crate::set_thread_meta_observer(Some(Arc::new(move |event| {
            if event.op_name == "autograd_backward" {
                *captured_gradient.lock().unwrap() = observed_variable
                    .grad()
                    .map(|gradient| gradient.data().to_vec());
            }
        })));

        variable.hadamard(&variable).unwrap().backward().unwrap();
        crate::set_thread_meta_observer(previous);

        assert_eq!(*observed_gradient.lock().unwrap(), Some(vec![4.0]));
    }
}
