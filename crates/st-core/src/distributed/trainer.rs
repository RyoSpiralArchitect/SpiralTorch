// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::collective::CollectiveArena;

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
        if learning_rate <= 0.0 {
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
        }

        self.arena.all_reduce_sum(gradients);
        let workers = gradients.len() as f32;

        for (shard, gradient) in self.parameters.iter_mut().zip(gradients.iter()) {
            for (param, grad) in shard.iter_mut().zip(gradient.iter()) {
                *param -= self.learning_rate * (*grad / workers);
            }
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
        self.arena.enqueue_gradient(gradient);
        Ok(())
    }

    /// Drains all pending asynchronous gradients, applying their averaged update.
    pub fn merge_async_gradients(&mut self) -> Result<bool, DistributedTrainerError> {
        self.async_buffer.clear();
        self.arena.drain_gradients(&mut self.async_buffer);
        if self.async_buffer.is_empty() {
            return Ok(false);
        }
        let total = self.total_params();
        if !self.async_buffer.len().is_multiple_of(total) {
            return Err(DistributedTrainerError::AsyncPayload {
                expected: total,
                got: self.async_buffer.len(),
            });
        }
        let contributions = self.async_buffer.len() / total;
        let mut accumulator = vec![0.0f32; total];
        for (idx, value) in self.async_buffer.iter().enumerate() {
            accumulator[idx % total] += *value;
        }
        let scale = self.learning_rate / contributions as f32;
        for (idx, value) in accumulator.into_iter().enumerate() {
            let shard_idx = idx / self.shard_dim;
            let offset = idx % self.shard_dim;
            self.parameters[shard_idx][offset] -= scale * value;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn broadcast_replaces_parameters() {
        let mut trainer = DistributedTrainer::new(2, 2, 0.1).unwrap();
        let payload = vec![0.1, 0.2, 0.3, 0.4];
        trainer.broadcast_parameters(&payload).unwrap();
        assert_eq!(trainer.parameters(), payload);
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
