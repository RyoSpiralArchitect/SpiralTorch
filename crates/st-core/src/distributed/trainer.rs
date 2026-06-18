// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::collective::CollectiveArena;
use st_tensor::{emit_tensor_op, emit_tensor_op_meta, tensor_op_meta_observer_installed};

/// Errors emitted by the [`DistributedTrainer`].
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum DistributedTrainerError {
    #[error("trainer requires at least one shard and positive dimension")]
    EmptyTopology,
    #[error("learning rate must be positive, got {0}")]
    NonPositiveLearningRate(f32),
    #[error("gradient dimensionality mismatch: expected {expected}, got {got}")]
    GradientDimension { expected: usize, got: usize },
    #[error("async gradient payload length {got} is not a multiple of {expected}")]
    AsyncPayload { expected: usize, got: usize },
    #[error("{label} contained non-finite value at index {index}: {value}")]
    NonFiniteValue {
        label: &'static str,
        index: usize,
        value: f32,
    },
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
    async_buffer: Vec<f32>,
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
            async_buffer: Vec::new(),
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

    fn checked_parameter_update(
        delta_label: &'static str,
        parameter_label: &'static str,
        index: usize,
        parameter: f32,
        delta: f32,
    ) -> Result<f32, DistributedTrainerError> {
        if !delta.is_finite() {
            return Err(DistributedTrainerError::NonFiniteValue {
                label: delta_label,
                index,
                value: delta,
            });
        }
        let next = parameter - delta;
        if !next.is_finite() {
            return Err(DistributedTrainerError::NonFiniteValue {
                label: parameter_label,
                index,
                value: next,
            });
        }
        Ok(next)
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
        self.arena.all_reduce_sum(gradients);
        for gradient in gradients.iter() {
            Self::validate_finite_slice("reduced_gradient", gradient)?;
        }
        let workers = gradients.len() as f32;
        let reduced_gradient_l2 = if capture_meta {
            nested_l2_norm(gradients)
        } else {
            0.0
        };

        let mut next_parameters = self.parameters.clone();
        let mut parameter_delta_l2 = 0.0f64;
        for (shard_idx, (shard, gradient)) in
            next_parameters.iter_mut().zip(gradients.iter()).enumerate()
        {
            for (offset, (param, grad)) in shard.iter_mut().zip(gradient.iter()).enumerate() {
                let delta = self.learning_rate * (*grad / workers);
                if capture_meta {
                    let value = delta as f64;
                    parameter_delta_l2 += value * value;
                }
                let index = shard_idx * self.shard_dim + offset;
                *param = Self::checked_parameter_update(
                    "parameter_delta",
                    "parameter_update",
                    index,
                    *param,
                    delta,
                )?;
            }
        }
        if capture_meta {
            parameter_delta_l2 = parameter_delta_l2.sqrt();
        }
        self.parameters = next_parameters;
        emit_tensor_op(
            "distributed_trainer_sync_step",
            &[gradients.len().max(1), self.shard_dim],
            &[self.parameters.len().max(1), self.shard_dim],
        );
        if capture_meta {
            emit_tensor_op_meta("distributed_trainer_sync_step", || {
                serde_json::json!({
                    "backend": "cpu",
                    "requested_backend": "auto",
                    "kind": "st_core_distributed_trainer_sync_step",
                    "gradient_reduction_backend": "cpu_collective_arena",
                    "parameter_update_backend": "cpu_loop",
                    "gradient_reduction_mode": "all_reduce_sum_in_place",
                    "parameter_update_mode": "sharded_sgd_average",
                    "route_blocker": "host_vec_shards_collective_arena",
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
        self.arena.enqueue_gradient(gradient);
        emit_tensor_op(
            "distributed_trainer_async_enqueue",
            &[gradient.len().max(1)],
            &[gradient.len().max(1)],
        );
        if tensor_op_meta_observer_installed() {
            emit_tensor_op_meta("distributed_trainer_async_enqueue", || {
                serde_json::json!({
                    "backend": "cpu",
                    "requested_backend": "auto",
                    "kind": "st_core_distributed_trainer_async_enqueue",
                    "queue_backend": "cpu_mutex_vec",
                    "queue_mode": "append_flat_gradient",
                    "route_blocker": "host_mutex_gradient_queue",
                    "gradient_len": gradient.len(),
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
        self.async_buffer.clear();
        self.arena.drain_gradients(&mut self.async_buffer);
        if self.async_buffer.is_empty() {
            emit_tensor_op(
                "distributed_trainer_async_merge",
                &[0, self.total_params().max(1)],
                &[self.parameters.len().max(1), self.shard_dim],
            );
            if capture_meta {
                emit_tensor_op_meta("distributed_trainer_async_merge", || {
                    serde_json::json!({
                        "backend": "cpu",
                        "requested_backend": "auto",
                        "kind": "st_core_distributed_trainer_async_merge",
                        "queue_backend": "cpu_mutex_vec",
                        "gradient_accumulation_backend": "cpu_loop",
                        "parameter_update_backend": "cpu_loop",
                        "merge_mode": "empty_queue",
                        "route_blocker": "host_mutex_gradient_queue",
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
            return Ok(false);
        }
        let total = self.total_params();
        if !self.async_buffer.len().is_multiple_of(total) {
            return Err(DistributedTrainerError::AsyncPayload {
                expected: total,
                got: self.async_buffer.len(),
            });
        }
        Self::validate_finite_slice("async_gradient", &self.async_buffer)?;
        let contributions = self.async_buffer.len() / total;
        let mut accumulator = vec![0.0f32; total];
        for (idx, value) in self.async_buffer.iter().enumerate() {
            accumulator[idx % total] += *value;
        }
        Self::validate_finite_slice("reduced_async_gradient", &accumulator)?;
        let scale = self.learning_rate / contributions as f32;
        let accumulator_l2 = if capture_meta {
            l2_norm(&accumulator)
        } else {
            0.0
        };
        let mut next_parameters = self.parameters.clone();
        let mut parameter_delta_l2 = 0.0f64;
        for (idx, value) in accumulator.into_iter().enumerate() {
            let shard_idx = idx / self.shard_dim;
            let offset = idx % self.shard_dim;
            let delta = scale * value;
            if capture_meta {
                let value = delta as f64;
                parameter_delta_l2 += value * value;
            }
            let parameter = next_parameters[shard_idx][offset];
            next_parameters[shard_idx][offset] = Self::checked_parameter_update(
                "async_parameter_delta",
                "async_parameter_update",
                idx,
                parameter,
                delta,
            )?;
        }
        if capture_meta {
            parameter_delta_l2 = parameter_delta_l2.sqrt();
        }
        self.parameters = next_parameters;
        emit_tensor_op(
            "distributed_trainer_async_merge",
            &[contributions.max(1), total],
            &[self.parameters.len().max(1), self.shard_dim],
        );
        if capture_meta {
            emit_tensor_op_meta("distributed_trainer_async_merge", || {
                serde_json::json!({
                    "backend": "cpu",
                    "requested_backend": "auto",
                    "kind": "st_core_distributed_trainer_async_merge",
                    "queue_backend": "cpu_mutex_vec",
                    "gradient_accumulation_backend": "cpu_loop",
                    "parameter_update_backend": "cpu_loop",
                    "merge_mode": "flat_modulo_accumulate_then_sgd",
                    "route_blocker": "host_mutex_gradient_queue_and_flat_cpu_accumulator",
                    "merged": true,
                    "contributions": contributions,
                    "payload_len": self.async_buffer.len(),
                    "total_params": total,
                    "shards": self.parameters.len(),
                    "shard_dim": self.shard_dim,
                    "learning_rate": finite_meta_f32(self.learning_rate),
                    "merge_scale": finite_meta_f32(scale),
                    "accumulator_l2": finite_meta_f32(accumulator_l2),
                    "parameter_delta_l2": finite_meta_f64(parameter_delta_l2),
                    "estimated_queue_drain_values": self.async_buffer.len(),
                    "estimated_accumulation_values": self.async_buffer.len(),
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
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
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
        st_tensor::set_tensor_op_meta_observer(previous);

        let events = events.lock().unwrap();
        let sync = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "distributed_trainer_sync_step"
                    && data["kind"] == "st_core_distributed_trainer_sync_step"
            })
            .expect("distributed sync step metadata event");
        assert_eq!(sync.1["workers"], 2);
        assert_eq!(sync.1["shards"], 2);
        assert_eq!(sync.1["shard_dim"], 2);
        assert_eq!(sync.1["gradient_reduction_backend"], "cpu_collective_arena");
        assert_eq!(sync.1["parameter_update_backend"], "cpu_loop");
        assert_eq!(sync.1["gradient_reduction_mode"], "all_reduce_sum_in_place");
        assert_eq!(sync.1["parameter_update_mode"], "sharded_sgd_average");
        assert_eq!(sync.1["route_blocker"], "host_vec_shards_collective_arena");
        assert_eq!(sync.1["estimated_reduction_values"], 4);
        assert_eq!(sync.1["estimated_parameter_update_values"], 4);
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
            })
            .expect("distributed async merge metadata event");
        assert_eq!(merge.1["contributions"], 2);
        assert_eq!(merge.1["payload_len"], 8);
        assert_eq!(merge.1["total_params"], 4);
        assert_eq!(merge.1["queue_backend"], "cpu_mutex_vec");
        assert_eq!(merge.1["gradient_accumulation_backend"], "cpu_loop");
        assert_eq!(merge.1["parameter_update_backend"], "cpu_loop");
        assert_eq!(merge.1["merge_mode"], "flat_modulo_accumulate_then_sgd");
        assert_eq!(
            merge.1["route_blocker"],
            "host_mutex_gradient_queue_and_flat_cpu_accumulator"
        );
        assert_eq!(merge.1["estimated_queue_drain_values"], 8);
        assert_eq!(merge.1["estimated_accumulation_values"], 8);
        assert_eq!(merge.1["estimated_parameter_update_values"], 4);
        assert!((merge.1["merge_scale"].as_f64().unwrap() - 0.25).abs() < 1e-9);
        assert!(merge.1["parameter_delta_l2"].as_f64().unwrap() > 0.0);
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
    }

    #[test]
    fn synchronous_step_rejects_parameter_overflow_without_mutating_parameters() {
        let mut trainer = DistributedTrainer::new(1, 1, f32::MAX).unwrap();
        trainer.broadcast_parameters(&[f32::MAX]).unwrap();
        let before = trainer.parameters();
        let mut gradients = vec![vec![-0.5]];

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
        trainer.arena.enqueue_gradient(&[1.0, f32::NAN]);
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
    }

    #[test]
    fn async_merge_detects_incomplete_payload() {
        let mut trainer = DistributedTrainer::new(1, 2, 0.2).unwrap();
        trainer.arena.enqueue_gradient(&[1.0]);
        let err = trainer.merge_async_gradients().unwrap_err();
        assert_eq!(
            err,
            DistributedTrainerError::AsyncPayload {
                expected: 2,
                got: 1
            }
        );
    }
}
