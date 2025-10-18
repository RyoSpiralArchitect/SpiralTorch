// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::sync::{Arc, Mutex};

/// Simple collective communication fabric that operates on in-memory buffers.
#[derive(Clone, Default, Debug)]
pub struct CollectiveArena {
    buffer: Arc<Mutex<Vec<f32>>>,
}

impl CollectiveArena {
    pub fn new() -> Self {
        Self {
            buffer: Arc::new(Mutex::new(Vec::new())),
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
        let mut guard = self.buffer.lock().unwrap();
        guard.extend_from_slice(gradient);
    }

    /// Drains all pending gradients into the supplied buffer.
    pub fn drain_gradients(&self, out: &mut Vec<f32>) {
        let mut guard = self.buffer.lock().unwrap();
        out.extend_from_slice(&guard);
        guard.clear();
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
        let mut out = Vec::new();
        arena.drain_gradients(&mut out);
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }
}
