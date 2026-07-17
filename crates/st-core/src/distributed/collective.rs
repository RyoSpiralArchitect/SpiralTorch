// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};

/// Simple collective communication fabric that operates on in-memory buffers.
#[derive(Clone, Default, Debug)]
pub struct CollectiveArena {
    buffer: Arc<Mutex<VecDeque<Vec<f32>>>>,
}

impl CollectiveArena {
    pub fn new() -> Self {
        Self {
            buffer: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    fn lock_buffer(&self) -> MutexGuard<'_, VecDeque<Vec<f32>>> {
        match self.buffer.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                let guard = poisoned.into_inner();
                self.buffer.clear_poison();
                guard
            }
        }
    }

    /// Performs an in-place all-reduce sum across the provided slices.
    pub fn all_reduce_sum(&self, tensors: &mut [Vec<f32>]) {
        if tensors.is_empty() {
            return;
        }
        let mut accumulator = vec![0.0f32; tensors[0].len()];
        for tensor in tensors.iter() {
            for (idx, value) in tensor.iter().enumerate() {
                accumulator[idx] += *value;
            }
        }
        for tensor in tensors.iter_mut() {
            for (idx, slot) in tensor.iter_mut().enumerate() {
                *slot = accumulator[idx];
            }
        }
    }

    /// Broadcasts a tensor from the root to all peers.
    pub fn broadcast(&self, root: &[f32], peers: &mut [Vec<f32>]) {
        for peer in peers {
            peer.clear();
            peer.extend_from_slice(root);
        }
    }

    /// Stores the provided gradient slice so asynchronous workers can merge it later.
    pub fn enqueue_gradient(&self, gradient: &[f32]) {
        self.lock_buffer().push_back(gradient.to_vec());
    }

    /// Drains all pending gradients into the supplied buffer.
    pub fn drain_gradients(&self, out: &mut Vec<f32>) {
        for gradient in self.drain_gradient_batches() {
            out.extend_from_slice(&gradient);
        }
    }

    pub(super) fn drain_gradient_batches(&self) -> Vec<Vec<f32>> {
        self.lock_buffer().drain(..).collect()
    }

    pub(super) fn restore_gradient_batches(&self, gradients: Vec<Vec<f32>>) {
        if gradients.is_empty() {
            return;
        }
        let mut guard = self.lock_buffer();
        let mut restored = VecDeque::from(gradients);
        restored.append(&mut guard);
        *guard = restored;
    }

    pub(super) fn queued_gradient_batches(&self) -> usize {
        self.lock_buffer().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_reduce_sums_values() {
        let arena = CollectiveArena::new();
        let mut tensors = vec![vec![1.0, 2.0], vec![0.5, -1.0]];
        arena.all_reduce_sum(&mut tensors);
        assert_eq!(tensors[0], vec![1.5, 1.0]);
        assert_eq!(tensors[1], vec![1.5, 1.0]);
    }

    #[test]
    fn broadcast_clones_root() {
        let arena = CollectiveArena::new();
        let mut peers = vec![vec![], vec![]];
        arena.broadcast(&[1.0, 2.0, 3.0], &mut peers);
        assert_eq!(peers[0], vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn gradient_queue_accumulates_updates() {
        let arena = CollectiveArena::new();
        arena.enqueue_gradient(&[1.0, 2.0]);
        arena.enqueue_gradient(&[3.0]);
        assert_eq!(arena.queued_gradient_batches(), 2);
        let mut out = Vec::new();
        arena.drain_gradients(&mut out);
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
        assert_eq!(arena.queued_gradient_batches(), 0);
    }

    #[test]
    fn restored_batches_precede_gradients_enqueued_during_a_failed_merge() {
        let arena = CollectiveArena::new();
        arena.enqueue_gradient(&[1.0, 2.0]);
        let drained = arena.drain_gradient_batches();
        arena.enqueue_gradient(&[3.0, 4.0]);

        arena.restore_gradient_batches(drained);

        assert_eq!(
            arena.drain_gradient_batches(),
            vec![vec![1.0, 2.0], vec![3.0, 4.0]]
        );
    }

    #[test]
    fn gradient_queue_recovers_after_mutex_poison() {
        let arena = CollectiveArena::new();
        let buffer = arena.buffer.clone();
        let _ = std::thread::spawn(move || {
            let _guard = buffer.lock().unwrap();
            panic!("poison collective queue");
        })
        .join();

        arena.enqueue_gradient(&[1.0, 2.0]);

        assert_eq!(arena.drain_gradient_batches(), vec![vec![1.0, 2.0]]);
    }
}
