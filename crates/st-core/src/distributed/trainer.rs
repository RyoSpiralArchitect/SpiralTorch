// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::collective::{CollectiveArena, CollectiveError};
use crate::backend::execution::current_tensor_util_route;
use st_tensor::{
    emit_tensor_op, emit_tensor_op_meta, tensor_op_meta_observer_installed, Tensor, TensorError,
    TensorUtilBackend,
};

/// Errors emitted by the [`DistributedTrainer`].
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum DistributedTrainerError {
    #[error("trainer requires at least one shard and positive dimension")]
    EmptyTopology,
    #[error("learning rate must be positive, got {0}")]
    NonPositiveLearningRate(f32),
    #[error("gradient dimensionality mismatch: expected {expected}, got {got}")]
    GradientDimension { expected: usize, got: usize },
    #[error("async gradient payload length mismatch: expected {expected}, got {got}")]
    AsyncPayload { expected: usize, got: usize },
    #[error("{label} contained non-finite value at index {index}: {value}")]
    NonFiniteValue {
        label: &'static str,
        index: usize,
        value: f32,
    },
    #[error("tensor operation {operation} failed: {source}")]
    TensorOperation {
        operation: &'static str,
        #[source]
        source: TensorError,
    },
    #[error(transparent)]
    Collective(#[from] CollectiveError),
}

#[derive(Clone, Copy, Debug)]
struct TensorRoute {
    backend: TensorUtilBackend,
    requested_backend: &'static str,
    selected_backend: &'static str,
}

#[derive(Debug)]
struct AsyncMergeResult {
    next_parameters: Vec<Vec<f32>>,
    accumulator_l2: f32,
    parameter_delta_l2: f64,
    reduction_route: TensorRoute,
    update_route: TensorRoute,
    scale: f32,
    payload_len: usize,
}

fn current_tensor_route(values: usize) -> TensorRoute {
    let route = current_tensor_util_route(values);
    TensorRoute {
        backend: route.selected_backend,
        requested_backend: route.requested_backend_label(),
        selected_backend: route.selected_backend_label(),
    }
}

fn finite_meta_f32(value: f32) -> serde_json::Value {
    serde_json::Value::from(if value.is_finite() { value as f64 } else { 0.0 })
}

fn finite_meta_f64(value: f64) -> serde_json::Value {
    serde_json::Value::from(if value.is_finite() { value } else { 0.0 })
}

fn l2_norm(values: &[f32]) -> f32 {
    values
        .iter()
        .filter(|value| value.is_finite())
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
}

fn nested_l2_norm(values: &[Vec<f32>]) -> f32 {
    values
        .iter()
        .flat_map(|row| row.iter())
        .filter(|value| value.is_finite())
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
}

/// Simple synchronous + asynchronous gradient reducer used by distributed trainers.
#[derive(Debug, Clone)]
pub struct DistributedTrainer {
    arena: CollectiveArena,
    parameters: Vec<Vec<f32>>,
    shard_dim: usize,
    learning_rate: f32,
}

impl DistributedTrainer {
    /// Builds a trainer with the provided shard configuration.
    pub fn new(
        shards: usize,
        shard_dim: usize,
        learning_rate: f32,
    ) -> Result<Self, DistributedTrainerError> {
        if shards == 0 || shard_dim == 0 {
            return Err(DistributedTrainerError::EmptyTopology);
        }
        if learning_rate <= 0.0 || !learning_rate.is_finite() {
            return Err(DistributedTrainerError::NonPositiveLearningRate(
                learning_rate,
            ));
        }
        Ok(Self {
            arena: CollectiveArena::new(),
            parameters: vec![vec![0.0; shard_dim]; shards],
            shard_dim,
            learning_rate,
        })
    }

    fn total_params(&self) -> usize {
        self.parameters.len() * self.shard_dim
    }

    fn validate_finite_slice(
        label: &'static str,
        values: &[f32],
    ) -> Result<(), DistributedTrainerError> {
        for (index, value) in values.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(DistributedTrainerError::NonFiniteValue {
                    label,
                    index,
                    value,
                });
            }
        }
        Ok(())
    }

    fn tensor_error(operation: &'static str, source: TensorError) -> DistributedTrainerError {
        DistributedTrainerError::TensorOperation { operation, source }
    }

    fn first_reduction_overflow(rows: &[Vec<f32>], cols: usize) -> Option<(usize, f32)> {
        for col in 0..cols {
            let mut sum = 0.0f32;
            for row in rows {
                sum += row[col];
                if !sum.is_finite() {
                    return Some((col, sum));
                }
            }
        }
        None
    }

    fn map_reduction_error(
        operation: &'static str,
        label: &'static str,
        rows: &[Vec<f32>],
        cols: usize,
        source: TensorError,
    ) -> DistributedTrainerError {
        if matches!(source, TensorError::NonFiniteValue { .. }) {
            if let Some((index, value)) = Self::first_reduction_overflow(rows, cols) {
                return DistributedTrainerError::NonFiniteValue {
                    label,
                    index,
                    value,
                };
            }
        }
        Self::tensor_error(operation, source)
    }

    fn map_collective_reduction_error(
        rows: &[Vec<f32>],
        cols: usize,
        error: CollectiveError,
    ) -> DistributedTrainerError {
        match error {
            CollectiveError::NonFiniteReduction { index, value } => {
                DistributedTrainerError::NonFiniteValue {
                    label: "reduced_gradient",
                    index,
                    value,
                }
            }
            CollectiveError::TensorOperation { source, .. } => Self::map_reduction_error(
                "sync_gradient_reduction",
                "reduced_gradient",
                rows,
                cols,
                source,
            ),
            error => DistributedTrainerError::Collective(error),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn map_update_error(
        operation: &'static str,
        delta_label: &'static str,
        parameter_label: &'static str,
        parameters: &[f32],
        gradients: &[f32],
        scale: f32,
        source: TensorError,
    ) -> DistributedTrainerError {
        if matches!(source, TensorError::NonFiniteValue { .. }) {
            for (index, (&parameter, &gradient)) in parameters.iter().zip(gradients).enumerate() {
                let delta = -scale * gradient;
                if !delta.is_finite() {
                    return DistributedTrainerError::NonFiniteValue {
                        label: delta_label,
                        index,
                        value: delta,
                    };
                }
                let next = parameter + scale * gradient;
                if !next.is_finite() {
                    return DistributedTrainerError::NonFiniteValue {
                        label: parameter_label,
                        index,
                        value: next,
                    };
                }
            }
        }
        Self::tensor_error(operation, source)
    }

    fn rows_tensor(
        operation: &'static str,
        rows: &[Vec<f32>],
        cols: usize,
    ) -> Result<Tensor, DistributedTrainerError> {
        let data = rows.iter().flat_map(|row| row.iter().copied()).collect();
        Tensor::from_vec(rows.len(), cols, data)
            .map_err(|error| Self::tensor_error(operation, error))
    }

    fn split_parameter_tensor(&self, tensor: &Tensor) -> Vec<Vec<f32>> {
        tensor
            .data()
            .chunks(self.shard_dim)
            .map(|shard| shard.to_vec())
            .collect()
    }

    fn prepare_async_merge(
        &self,
        gradients: &[Vec<f32>],
        capture_meta: bool,
    ) -> Result<AsyncMergeResult, DistributedTrainerError> {
        let total = self.total_params();
        let payload_len = gradients.iter().map(Vec::len).sum();
        for gradient in gradients {
            if gradient.len() != total {
                return Err(DistributedTrainerError::AsyncPayload {
                    expected: total,
                    got: gradient.len(),
                });
            }
            Self::validate_finite_slice("async_gradient", gradient)?;
        }

        let contributions = gradients.len();
        let reduction_route = current_tensor_route(payload_len);
        let gradient_tensor = Self::rows_tensor("async_gradient_tensor", gradients, total)?;
        let accumulator = gradient_tensor
            .try_sum_axis0_with_backend(reduction_route.backend)
            .map_err(|error| {
                Self::map_reduction_error(
                    "async_gradient_reduction",
                    "reduced_async_gradient",
                    gradients,
                    total,
                    error,
                )
            })?;
        Self::validate_finite_slice("reduced_async_gradient", &accumulator)?;

        let parameter_values = self.parameters();
        let mut next_parameters = Tensor::from_vec(1, total, parameter_values.clone())
            .map_err(|error| Self::tensor_error("async_parameter_tensor", error))?;
        let accumulator_tensor = Tensor::from_vec(1, total, accumulator.clone())
            .map_err(|error| Self::tensor_error("async_accumulator_tensor", error))?;
        let scale = self.learning_rate / contributions as f32;
        let update_scale = -scale;
        let update_route = current_tensor_route(total);
        next_parameters
            .add_scaled_with_backend(&accumulator_tensor, update_scale, update_route.backend)
            .map_err(|error| {
                Self::map_update_error(
                    "async_parameter_update",
                    "async_parameter_delta",
                    "async_parameter_update",
                    &parameter_values,
                    &accumulator,
                    update_scale,
                    error,
                )
            })?;
        Self::validate_finite_slice("async_parameter_update", next_parameters.data())?;

        let accumulator_l2 = if capture_meta {
            l2_norm(&accumulator)
        } else {
            0.0
        };
        let parameter_delta_l2 = if capture_meta {
            accumulator
                .iter()
                .map(|value| {
                    let delta = *value as f64 * scale as f64;
                    delta * delta
                })
                .sum::<f64>()
                .sqrt()
        } else {
            0.0
        };

        Ok(AsyncMergeResult {
            next_parameters: self.split_parameter_tensor(&next_parameters),
            accumulator_l2,
            parameter_delta_l2,
            reduction_route,
            update_route,
            scale,
            payload_len,
        })
    }

    /// Returns a flattened view over all parameter shards.
    pub fn parameters(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.total_params());
        for shard in &self.parameters {
            flat.extend_from_slice(shard);
        }
        flat
    }

    /// Replaces local parameter shards with the broadcast payload.
    pub fn broadcast_parameters(&mut self, root: &[f32]) -> Result<(), DistributedTrainerError> {
        if root.len() != self.total_params() {
            return Err(DistributedTrainerError::GradientDimension {
                expected: self.total_params(),
                got: root.len(),
            });
        }
        Self::validate_finite_slice("broadcast_parameters", root)?;
        for (idx, shard) in self.parameters.iter_mut().enumerate() {
            let offset = idx * self.shard_dim;
            shard.copy_from_slice(&root[offset..offset + self.shard_dim]);
        }
        Ok(())
    }

    /// Applies a synchronous gradient step after performing an all-reduce sum.
    pub fn apply_step(
        &mut self,
        gradients: &mut [Vec<f32>],
    ) -> Result<(), DistributedTrainerError> {
        if gradients.len() != self.parameters.len() {
            return Err(DistributedTrainerError::GradientDimension {
                expected: self.parameters.len(),
                got: gradients.len(),
            });
        }
        for gradient in gradients.iter() {
            if gradient.len() != self.shard_dim {
                return Err(DistributedTrainerError::GradientDimension {
                    expected: self.shard_dim,
                    got: gradient.len(),
                });
            }
            Self::validate_finite_slice("gradient", gradient)?;
        }

        let capture_meta = tensor_op_meta_observer_installed();
        let gradient_l2_before = if capture_meta {
            nested_l2_norm(gradients)
        } else {
            0.0
        };
        let mut collective_gradients = gradients.to_vec();
        let collective_report = self
            .arena
            .all_reduce_sum(&mut collective_gradients)
            .map_err(|error| {
                Self::map_collective_reduction_error(gradients, self.shard_dim, error)
            })?;
        let reduced_gradient = collective_gradients
            .into_iter()
            .next()
            .ok_or(DistributedTrainerError::EmptyTopology)?;
        let workers = gradients.len() as f32;

        let parameter_values = self.parameters();
        let update_gradient_values = reduced_gradient
            .iter()
            .copied()
            .cycle()
            .take(self.total_params())
            .collect::<Vec<_>>();
        let mut next_parameters = Tensor::from_vec(
            self.parameters.len(),
            self.shard_dim,
            parameter_values.clone(),
        )
        .map_err(|error| Self::tensor_error("sync_parameter_tensor", error))?;
        let update_gradient = Tensor::from_vec(
            gradients.len(),
            self.shard_dim,
            update_gradient_values.clone(),
        )
        .map_err(|error| Self::tensor_error("sync_update_gradient_tensor", error))?;
        let update_scale = -(self.learning_rate / workers);
        let update_route = current_tensor_route(self.total_params());
        next_parameters
            .add_scaled_with_backend(&update_gradient, update_scale, update_route.backend)
            .map_err(|error| {
                Self::map_update_error(
                    "sync_parameter_update",
                    "parameter_delta",
                    "parameter_update",
                    &parameter_values,
                    &update_gradient_values,
                    update_scale,
                    error,
                )
            })?;
        Self::validate_finite_slice("parameter_update", next_parameters.data())?;
        let parameter_delta_l2 = if capture_meta {
            update_gradient_values
                .iter()
                .map(|value| {
                    let delta = *value as f64 * -(update_scale as f64);
                    delta * delta
                })
                .sum::<f64>()
                .sqrt()
        } else {
            0.0
        };
        for gradient in gradients.iter_mut() {
            gradient.copy_from_slice(&reduced_gradient);
        }
        let reduced_gradient_l2 = if capture_meta {
            nested_l2_norm(gradients)
        } else {
            0.0
        };
        self.parameters = self.split_parameter_tensor(&next_parameters);
        emit_tensor_op(
            "distributed_trainer_sync_step",
            &[gradients.len().max(1), self.shard_dim],
            &[self.parameters.len().max(1), self.shard_dim],
        );
        if capture_meta {
            emit_tensor_op_meta("distributed_trainer_sync_step", || {
                serde_json::json!({
                    "backend": "cpu",
                    "requested_backend": collective_report.requested_backend.as_str(),
                    "kind": "st_core_distributed_trainer_sync_step",
                    "control_backend": "cpu_vec_shards",
                    "gradient_reduction_backend": format!("tensor_util_{}", collective_report.selected_backend),
                    "parameter_update_backend": format!("tensor_util_{}", update_route.selected_backend),
                    "gradient_reduction_requested_backend": collective_report.requested_backend.as_str(),
                    "parameter_update_requested_backend": update_route.requested_backend,
                    "gradient_reduction_mode": "collective_arena_tensor_sum_axis0_then_host_broadcast",
                    "parameter_update_mode": "tensor_add_scaled_sharded_sgd_average",
                    "route_blocker": "host_collective_broadcast_and_shard_storage",
                    "numeric_execution_owner": "st-core::backend::execution",
                    "numeric_backend_semantics": "selected_route",
                    "actual_numeric_backend_source": "sum_axis0_and_add_scaled_tensor_op_meta",
                    "collective_contract_version": collective_report.contract_version.as_str(),
                    "collective_input_sha256": collective_report.input_sha256.as_str(),
                    "collective_output_sha256": collective_report.output_sha256.as_str(),
                    "shards": self.parameters.len(),
                    "shard_dim": self.shard_dim,
                    "workers": gradients.len(),
                    "learning_rate": finite_meta_f32(self.learning_rate),
                    "gradient_l2_before": finite_meta_f32(gradient_l2_before),
                    "reduced_gradient_l2": finite_meta_f32(reduced_gradient_l2),
                    "parameter_delta_l2": finite_meta_f64(parameter_delta_l2),
                    "total_params": self.total_params(),
                    "estimated_reduction_values": gradients.len().saturating_mul(self.shard_dim),
                    "estimated_parameter_update_values": self.total_params(),
                })
            });
        }
        Ok(())
    }

    /// Queues a full-parameter gradient for asynchronous merging.
    pub fn submit_async_gradient(&self, gradient: &[f32]) -> Result<(), DistributedTrainerError> {
        if gradient.len() != self.total_params() {
            return Err(DistributedTrainerError::GradientDimension {
                expected: self.total_params(),
                got: gradient.len(),
            });
        }
        Self::validate_finite_slice("async_gradient", gradient)?;
        self.arena.enqueue_gradient(gradient)?;
        let queued_batches = self.arena.queued_gradient_batches();
        emit_tensor_op(
            "distributed_trainer_async_enqueue",
            &[gradient.len().max(1)],
            &[gradient.len().max(1)],
        );
        if tensor_op_meta_observer_installed() {
            emit_tensor_op_meta("distributed_trainer_async_enqueue", || {
                serde_json::json!({
                    "backend": "cpu",
                    "requested_backend": "none",
                    "kind": "st_core_distributed_trainer_async_enqueue",
                    "queue_backend": "cpu_mutex_gradient_batches",
                    "queue_mode": "append_owned_gradient_batch",
                    "route_blocker": "host_mutex_gradient_transport",
                    "gradient_len": gradient.len(),
                    "queued_batches": queued_batches,
                    "total_params": self.total_params(),
                    "gradient_l2": finite_meta_f32(l2_norm(gradient)),
                    "estimated_queue_push_values": gradient.len(),
                })
            });
        }
        Ok(())
    }

    /// Drains all pending asynchronous gradients, applying their averaged update.
    pub fn merge_async_gradients(&mut self) -> Result<bool, DistributedTrainerError> {
        let capture_meta = tensor_op_meta_observer_installed();
        let lease = self.arena.lease_gradient_batches();
        if lease.is_empty() {
            emit_tensor_op(
                "distributed_trainer_async_merge",
                &[0, self.total_params().max(1)],
                &[self.parameters.len().max(1), self.shard_dim],
            );
            if capture_meta {
                emit_tensor_op_meta("distributed_trainer_async_merge", || {
                    serde_json::json!({
                        "backend": "cpu",
                        "requested_backend": "none",
                        "kind": "st_core_distributed_trainer_async_merge",
                        "queue_backend": "cpu_mutex_gradient_batches",
                        "gradient_accumulation_backend": "none",
                        "parameter_update_backend": "none",
                        "merge_mode": "empty_queue",
                        "route_blocker": "host_mutex_gradient_transport",
                        "merged": false,
                        "contributions": 0,
                        "payload_len": 0,
                        "total_params": self.total_params(),
                        "learning_rate": finite_meta_f32(self.learning_rate),
                        "estimated_queue_drain_values": 0,
                        "estimated_accumulation_values": 0,
                        "estimated_parameter_update_values": 0,
                    })
                });
            }
            lease.commit();
            return Ok(false);
        }
        let contributions = lease.batches().len();
        let result = self.prepare_async_merge(lease.batches(), capture_meta)?;
        let total = self.total_params();
        self.parameters = result.next_parameters;
        lease.commit();
        emit_tensor_op(
            "distributed_trainer_async_merge",
            &[contributions.max(1), total],
            &[self.parameters.len().max(1), self.shard_dim],
        );
        if capture_meta {
            emit_tensor_op_meta("distributed_trainer_async_merge", || {
                serde_json::json!({
                    "backend": "cpu",
                    "requested_backend": result.reduction_route.requested_backend,
                    "kind": "st_core_distributed_trainer_async_merge",
                    "control_backend": "cpu_gradient_batch_queue",
                    "queue_backend": "cpu_mutex_gradient_batches",
                    "gradient_accumulation_backend": format!("tensor_util_{}", result.reduction_route.selected_backend),
                    "parameter_update_backend": format!("tensor_util_{}", result.update_route.selected_backend),
                    "gradient_accumulation_requested_backend": result.reduction_route.requested_backend,
                    "parameter_update_requested_backend": result.update_route.requested_backend,
                    "merge_mode": "tensor_sum_axis0_then_tensor_add_scaled",
                    "route_blocker": "host_mutex_gradient_transport_and_shard_storage",
                    "numeric_execution_owner": "st-core::backend::execution",
                    "numeric_backend_semantics": "selected_route",
                    "actual_numeric_backend_source": "sum_axis0_and_add_scaled_tensor_op_meta",
                    "merged": true,
                    "contributions": contributions,
                    "payload_len": result.payload_len,
                    "total_params": total,
                    "shards": self.parameters.len(),
                    "shard_dim": self.shard_dim,
                    "learning_rate": finite_meta_f32(self.learning_rate),
                    "merge_scale": finite_meta_f32(result.scale),
                    "accumulator_l2": finite_meta_f32(result.accumulator_l2),
                    "parameter_delta_l2": finite_meta_f64(result.parameter_delta_l2),
                    "estimated_queue_drain_values": result.payload_len,
                    "estimated_accumulation_values": result.payload_len,
                    "estimated_parameter_update_values": total,
                })
            });
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn rejects_invalid_topology() {
        assert_eq!(
            DistributedTrainer::new(0, 4, 0.1).unwrap_err(),
            DistributedTrainerError::EmptyTopology
        );
        assert_eq!(
            DistributedTrainer::new(2, 0, 0.1).unwrap_err(),
            DistributedTrainerError::EmptyTopology
        );
        assert_eq!(
            DistributedTrainer::new(2, 4, 0.0).unwrap_err(),
            DistributedTrainerError::NonPositiveLearningRate(0.0)
        );
        assert!(matches!(
            DistributedTrainer::new(2, 4, f32::NAN).unwrap_err(),
            DistributedTrainerError::NonPositiveLearningRate(rate) if rate.is_nan()
        ));
    }

    #[test]
    fn synchronous_step_applies_all_reduce() {
        let mut trainer = DistributedTrainer::new(2, 3, 0.5).unwrap();
        let mut gradients = vec![vec![1.0, 2.0, 3.0], vec![2.0, 0.0, -2.0]];
        trainer.apply_step(&mut gradients).unwrap();
        // All-reduce sums: [3,2,-?], average by workers (2)
        let params = trainer.parameters();
        assert!((params[0] + 0.75).abs() <= f32::EPSILON);
        assert!((params[1] + 0.5).abs() <= f32::EPSILON);
        assert!((params[2] + 0.25).abs() <= f32::EPSILON);
        assert!((params[3] + 0.75).abs() <= f32::EPSILON);
        assert!((params[4] + 0.5).abs() <= f32::EPSILON);
        assert!((params[5] + 0.25).abs() <= f32::EPSILON);
    }

    #[test]
    fn trainer_steps_emit_backend_meta() {
        let _lock = crate::telemetry::tensor_observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let mut trainer = DistributedTrainer::new(2, 2, 0.5).unwrap();
        let mut gradients = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        trainer.apply_step(&mut gradients).unwrap();
        trainer
            .submit_async_gradient(&[1.0, 1.0, 1.0, 1.0])
            .unwrap();
        trainer
            .submit_async_gradient(&[3.0, 3.0, 3.0, 3.0])
            .unwrap();
        assert!(trainer.merge_async_gradients().unwrap());
        st_tensor::set_thread_meta_observer(previous);

        let events = events.lock().unwrap().clone();
        let sync = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "distributed_trainer_sync_step"
                    && data["kind"] == "st_core_distributed_trainer_sync_step"
                    && data["total_params"] == 4
                    && data["shards"] == 2
                    && data["shard_dim"] == 2
            })
            .expect("distributed sync step metadata event");
        assert_eq!(sync.1["workers"], 2);
        assert_eq!(sync.1["shards"], 2);
        assert_eq!(sync.1["shard_dim"], 2);
        assert_eq!(sync.1["requested_backend"], "auto");
        assert_eq!(sync.1["control_backend"], "cpu_vec_shards");
        assert_eq!(sync.1["gradient_reduction_backend"], "tensor_util_auto");
        assert_eq!(sync.1["parameter_update_backend"], "tensor_util_auto");
        assert_eq!(sync.1["gradient_reduction_requested_backend"], "auto");
        assert_eq!(sync.1["parameter_update_requested_backend"], "auto");
        assert_eq!(
            sync.1["gradient_reduction_mode"],
            "collective_arena_tensor_sum_axis0_then_host_broadcast"
        );
        assert_eq!(
            sync.1["parameter_update_mode"],
            "tensor_add_scaled_sharded_sgd_average"
        );
        assert_eq!(
            sync.1["route_blocker"],
            "host_collective_broadcast_and_shard_storage"
        );
        assert_eq!(
            sync.1["numeric_execution_owner"],
            "st-core::backend::execution"
        );
        assert_eq!(sync.1["numeric_backend_semantics"], "selected_route");
        assert_eq!(
            sync.1["actual_numeric_backend_source"],
            "sum_axis0_and_add_scaled_tensor_op_meta"
        );
        assert_eq!(sync.1["estimated_reduction_values"], 4);
        assert_eq!(sync.1["estimated_parameter_update_values"], 4);
        assert_eq!(
            sync.1["collective_contract_version"],
            crate::distributed::collective::COLLECTIVE_EXECUTION_REPORT_CONTRACT_VERSION
        );
        assert_eq!(
            sync.1["collective_input_sha256"].as_str().unwrap().len(),
            64
        );
        assert_eq!(
            sync.1["collective_output_sha256"].as_str().unwrap().len(),
            64
        );
        assert!(
            sync.1["reduced_gradient_l2"].as_f64().unwrap()
                > sync.1["gradient_l2_before"].as_f64().unwrap()
        );
        assert!(sync.1["parameter_delta_l2"].as_f64().unwrap() > 0.0);

        let merge = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "distributed_trainer_async_merge"
                    && data["kind"] == "st_core_distributed_trainer_async_merge"
                    && data["merged"] == true
                    && data["total_params"] == 4
                    && data["payload_len"] == 8
            })
            .expect("distributed async merge metadata event");
        assert_eq!(merge.1["contributions"], 2);
        assert_eq!(merge.1["payload_len"], 8);
        assert_eq!(merge.1["total_params"], 4);
        assert_eq!(merge.1["requested_backend"], "auto");
        assert_eq!(merge.1["control_backend"], "cpu_gradient_batch_queue");
        assert_eq!(merge.1["queue_backend"], "cpu_mutex_gradient_batches");
        assert_eq!(merge.1["gradient_accumulation_backend"], "tensor_util_auto");
        assert_eq!(merge.1["parameter_update_backend"], "tensor_util_auto");
        assert_eq!(merge.1["gradient_accumulation_requested_backend"], "auto");
        assert_eq!(merge.1["parameter_update_requested_backend"], "auto");
        assert_eq!(
            merge.1["merge_mode"],
            "tensor_sum_axis0_then_tensor_add_scaled"
        );
        assert_eq!(
            merge.1["route_blocker"],
            "host_mutex_gradient_transport_and_shard_storage"
        );
        assert_eq!(
            merge.1["numeric_execution_owner"],
            "st-core::backend::execution"
        );
        assert_eq!(merge.1["numeric_backend_semantics"], "selected_route");
        assert_eq!(
            merge.1["actual_numeric_backend_source"],
            "sum_axis0_and_add_scaled_tensor_op_meta"
        );
        assert_eq!(merge.1["estimated_queue_drain_values"], 8);
        assert_eq!(merge.1["estimated_accumulation_values"], 8);
        assert_eq!(merge.1["estimated_parameter_update_values"], 4);
        assert!((merge.1["merge_scale"].as_f64().unwrap() - 0.25).abs() < 1e-9);
        assert!(merge.1["parameter_delta_l2"].as_f64().unwrap() > 0.0);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn cpu_and_wgpu_policy_routes_preserve_distributed_updates() {
        use crate::backend::device_caps::DeviceCaps;
        use crate::backend::execution::{
            push_backend_policy, AcceleratorFallback, BackendPolicy, ExecutionConfig,
        };

        fn run_step_sequence(trainer: &mut DistributedTrainer) -> Vec<Vec<f32>> {
            let mut gradients = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
            trainer.apply_step(&mut gradients).unwrap();
            trainer
                .submit_async_gradient(&[1.0, -2.0, 0.5, 0.25])
                .unwrap();
            trainer
                .submit_async_gradient(&[3.0, 2.0, 1.0, -0.25])
                .unwrap();
            assert!(trainer.merge_async_gradients().unwrap());
            gradients
        }

        let initial_parameters = [0.5, -0.25, 1.0, -1.5];
        let execution_config = ExecutionConfig::new(AcceleratorFallback::Allow, 1);
        let mut cpu_trainer = DistributedTrainer::new(2, 2, 0.1).unwrap();
        cpu_trainer
            .broadcast_parameters(&initial_parameters)
            .unwrap();
        let cpu_policy =
            BackendPolicy::from_device_caps_with_config(DeviceCaps::cpu(), execution_config);
        let cpu_gradients = {
            let _guard = push_backend_policy(cpu_policy);
            run_step_sequence(&mut cpu_trainer)
        };

        let _lock = crate::telemetry::tensor_observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&events);
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));
        let mut wgpu_trainer = DistributedTrainer::new(2, 2, 0.1).unwrap();
        wgpu_trainer
            .broadcast_parameters(&initial_parameters)
            .unwrap();
        let wgpu_policy = BackendPolicy::from_device_caps_with_config(
            DeviceCaps::wgpu(32, true, 256),
            execution_config,
        );
        let wgpu_gradients = {
            let _guard = push_backend_policy(wgpu_policy);
            run_step_sequence(&mut wgpu_trainer)
        };
        st_tensor::set_thread_meta_observer(previous);

        for (cpu, wgpu) in cpu_gradients
            .iter()
            .flatten()
            .zip(wgpu_gradients.iter().flatten())
        {
            assert!((cpu - wgpu).abs() <= 3.0e-4, "cpu={cpu}, wgpu={wgpu}");
        }
        for (cpu, wgpu) in cpu_trainer
            .parameters()
            .iter()
            .zip(wgpu_trainer.parameters())
        {
            assert!((cpu - wgpu).abs() <= 3.0e-4, "cpu={cpu}, wgpu={wgpu}");
        }

        let events = events.lock().unwrap();
        for op_name in ["sum_axis0", "add_scaled"] {
            let routed = events
                .iter()
                .filter(|(name, _)| *name == op_name)
                .collect::<Vec<_>>();
            assert_eq!(routed.len(), 2, "expected sync and async {op_name}");
            assert!(routed
                .iter()
                .all(|(_, data)| data["requested_backend"] == "wgpu"));
            assert!(routed
                .iter()
                .all(|(_, data)| matches!(data["backend"].as_str(), Some("cpu" | "wgpu_dense"))));
        }
        let sync = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "distributed_trainer_sync_step"
                    && data["kind"] == "st_core_distributed_trainer_sync_step"
            })
            .expect("distributed sync metadata");
        assert_eq!(sync.1["requested_backend"], "wgpu");
        assert_eq!(sync.1["gradient_reduction_backend"], "tensor_util_wgpu");
        assert_eq!(sync.1["parameter_update_backend"], "tensor_util_wgpu");
        assert_eq!(sync.1["control_backend"], "cpu_vec_shards");
        assert_eq!(
            sync.1["numeric_execution_owner"],
            "st-core::backend::execution"
        );
        let merge = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "distributed_trainer_async_merge"
                    && data["kind"] == "st_core_distributed_trainer_async_merge"
                    && data["merged"] == true
            })
            .expect("distributed async merge metadata");
        assert_eq!(merge.1["requested_backend"], "wgpu");
        assert_eq!(merge.1["gradient_accumulation_backend"], "tensor_util_wgpu");
        assert_eq!(merge.1["parameter_update_backend"], "tensor_util_wgpu");
        assert_eq!(merge.1["control_backend"], "cpu_gradient_batch_queue");
        assert_eq!(
            merge.1["numeric_execution_owner"],
            "st-core::backend::execution"
        );
    }

    #[test]
    fn broadcast_replaces_parameters() {
        let mut trainer = DistributedTrainer::new(2, 2, 0.1).unwrap();
        let payload = vec![0.1, 0.2, 0.3, 0.4];
        trainer.broadcast_parameters(&payload).unwrap();
        assert_eq!(trainer.parameters(), payload);
    }

    #[test]
    fn broadcast_rejects_non_finite_without_mutating_parameters() {
        let mut trainer = DistributedTrainer::new(2, 2, 0.1).unwrap();
        let payload = vec![0.1, 0.2, 0.3, 0.4];
        trainer.broadcast_parameters(&payload).unwrap();
        let err = trainer
            .broadcast_parameters(&[1.0, f32::NAN, 2.0, 3.0])
            .unwrap_err();
        assert!(matches!(
            err,
            DistributedTrainerError::NonFiniteValue {
                label: "broadcast_parameters",
                index: 1,
                value,
            } if value.is_nan()
        ));
        assert_eq!(trainer.parameters(), payload);
    }

    #[test]
    fn synchronous_step_rejects_non_finite_without_mutating_parameters() {
        let mut trainer = DistributedTrainer::new(2, 2, 0.1).unwrap();
        let before = trainer.parameters();
        let mut gradients = vec![vec![1.0, f32::INFINITY], vec![0.5, 0.25]];
        let err = trainer.apply_step(&mut gradients).unwrap_err();
        assert_eq!(
            err,
            DistributedTrainerError::NonFiniteValue {
                label: "gradient",
                index: 1,
                value: f32::INFINITY,
            }
        );
        assert_eq!(trainer.parameters(), before);
    }

    #[test]
    fn synchronous_step_rejects_reduced_overflow_without_mutating_parameters() {
        let mut trainer = DistributedTrainer::new(2, 1, 0.1).unwrap();
        let before = trainer.parameters();
        let mut gradients = vec![vec![f32::MAX], vec![f32::MAX]];
        let err = trainer.apply_step(&mut gradients).unwrap_err();
        assert_eq!(
            err,
            DistributedTrainerError::NonFiniteValue {
                label: "reduced_gradient",
                index: 0,
                value: f32::INFINITY,
            }
        );
        assert_eq!(trainer.parameters(), before);
    }

    #[test]
    fn synchronous_step_rejects_update_delta_overflow_without_mutating_parameters() {
        let mut trainer = DistributedTrainer::new(1, 1, f32::MAX).unwrap();
        let before = trainer.parameters();
        let mut gradients = vec![vec![2.0]];
        let gradients_before = gradients.clone();

        let err = trainer.apply_step(&mut gradients).unwrap_err();

        assert_eq!(
            err,
            DistributedTrainerError::NonFiniteValue {
                label: "parameter_delta",
                index: 0,
                value: f32::INFINITY,
            }
        );
        assert_eq!(trainer.parameters(), before);
        assert_eq!(gradients, gradients_before);
    }

    #[test]
    fn synchronous_step_rejects_parameter_overflow_without_mutating_parameters() {
        let mut trainer = DistributedTrainer::new(1, 1, f32::MAX).unwrap();
        trainer.broadcast_parameters(&[f32::MAX]).unwrap();
        let before = trainer.parameters();
        let mut gradients = vec![vec![-0.5]];
        let gradients_before = gradients.clone();

        let err = trainer.apply_step(&mut gradients).unwrap_err();

        assert_eq!(
            err,
            DistributedTrainerError::NonFiniteValue {
                label: "parameter_update",
                index: 0,
                value: f32::INFINITY,
            }
        );
        assert_eq!(trainer.parameters(), before);
        assert_eq!(gradients, gradients_before);
    }

    #[test]
    fn async_merge_applies_average() {
        let mut trainer = DistributedTrainer::new(1, 3, 0.2).unwrap();
        let grad_a = vec![1.0, 2.0, 3.0];
        let grad_b = vec![3.0, 2.0, 1.0];
        trainer.submit_async_gradient(&grad_a).unwrap();
        trainer.submit_async_gradient(&grad_b).unwrap();
        assert!(trainer.merge_async_gradients().unwrap());
        let params = trainer.parameters();
        // Average gradient is [2,2,2] => params -= 0.2 * avg
        assert!((params[0] - (-0.4)).abs() <= f32::EPSILON);
        assert!((params[1] - (-0.4)).abs() <= f32::EPSILON);
        assert!((params[2] - (-0.4)).abs() <= f32::EPSILON);
        assert_eq!(trainer.arena.queued_gradient_batches(), 0);
        assert!(!trainer.merge_async_gradients().unwrap());
    }

    #[test]
    fn async_gradient_rejects_non_finite_without_enqueueing() {
        let mut trainer = DistributedTrainer::new(1, 2, 0.2).unwrap();
        let err = trainer
            .submit_async_gradient(&[1.0, f32::NEG_INFINITY])
            .unwrap_err();
        assert_eq!(
            err,
            DistributedTrainerError::NonFiniteValue {
                label: "async_gradient",
                index: 1,
                value: f32::NEG_INFINITY,
            }
        );
        assert!(!trainer.merge_async_gradients().unwrap());
    }

    #[test]
    fn async_merge_rejects_non_finite_without_mutating_parameters() {
        let mut trainer = DistributedTrainer::new(1, 2, 0.2).unwrap();
        let before = trainer.parameters();
        trainer
            .arena
            .restore_gradient_batches(vec![vec![1.0, f32::NAN]]);
        let err = trainer.merge_async_gradients().unwrap_err();
        assert!(matches!(
            err,
            DistributedTrainerError::NonFiniteValue {
                label: "async_gradient",
                index: 1,
                value,
            } if value.is_nan()
        ));
        assert_eq!(trainer.parameters(), before);
        assert_eq!(trainer.arena.queued_gradient_batches(), 1);
    }

    #[test]
    fn async_merge_rejects_update_delta_overflow_without_mutating_parameters() {
        let mut trainer = DistributedTrainer::new(1, 1, f32::MAX).unwrap();
        let before = trainer.parameters();
        trainer.submit_async_gradient(&[2.0]).unwrap();

        let err = trainer.merge_async_gradients().unwrap_err();

        assert_eq!(
            err,
            DistributedTrainerError::NonFiniteValue {
                label: "async_parameter_delta",
                index: 0,
                value: f32::INFINITY,
            }
        );
        assert_eq!(trainer.parameters(), before);
        assert_eq!(trainer.arena.queued_gradient_batches(), 1);
    }

    #[test]
    fn async_merge_rejects_parameter_overflow_without_mutating_parameters() {
        let mut trainer = DistributedTrainer::new(1, 1, f32::MAX).unwrap();
        trainer.broadcast_parameters(&[f32::MAX]).unwrap();
        let before = trainer.parameters();
        trainer.submit_async_gradient(&[-0.5]).unwrap();

        let err = trainer.merge_async_gradients().unwrap_err();

        assert_eq!(
            err,
            DistributedTrainerError::NonFiniteValue {
                label: "async_parameter_update",
                index: 0,
                value: f32::INFINITY,
            }
        );
        assert_eq!(trainer.parameters(), before);
        assert_eq!(trainer.arena.queued_gradient_batches(), 1);
    }

    #[test]
    fn async_merge_detects_incomplete_payload() {
        let mut trainer = DistributedTrainer::new(1, 2, 0.2).unwrap();
        trainer.arena.restore_gradient_batches(vec![vec![1.0]]);
        let err = trainer.merge_async_gradients().unwrap_err();
        assert_eq!(
            err,
            DistributedTrainerError::AsyncPayload {
                expected: 2,
                got: 1
            }
        );
        assert_eq!(trainer.arena.queued_gradient_batches(), 1);
    }
}
