// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::sync::Arc;

use super::distributed::st_distributed;

/// Reduction strategy applied to distributed metrics once synchronized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricReduce {
    /// Keep the sum aggregated across all peers.
    Sum,
    /// Average the aggregated value over the world size.
    Mean,
}

/// Trait implemented by devices capable of participating in self-supervised training.
pub trait TrainingDevice: Send + Sync {
    /// Identifier of the current worker inside the distributed world.
    fn rank(&self) -> usize;
    /// Total number of workers that contribute gradients.
    fn world_size(&self) -> usize;
    /// Synchronizes gradients in-place, applying the device's strategy (e.g. all-reduce).
    fn synchronize_gradients(&self, gradients: &mut [f32]);
    /// Aggregates the provided metrics in-place according to the reduction policy.
    fn aggregate_metrics(&self, metrics: &mut [f32], reduce: MetricReduce);
}

/// CPU-only device that does not perform any synchronization.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuDevice;

impl CpuDevice {
    pub fn new() -> Self {
        Self
    }
}

impl TrainingDevice for CpuDevice {
    fn rank(&self) -> usize {
        0
    }

    fn world_size(&self) -> usize {
        1
    }

    fn synchronize_gradients(&self, _gradients: &mut [f32]) {}

    fn aggregate_metrics(&self, _metrics: &mut [f32], _reduce: MetricReduce) {}
}

/// Distributed device backed by a rendezvous session.
#[derive(Debug, Clone)]
pub struct DistributedDevice {
    session: Arc<st_distributed::RendezvousSession>,
    strategy: SyncStrategy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncStrategy {
    /// Synchronize gradients via an all-reduce sum and average locally.
    AllReduce,
}

impl DistributedDevice {
    /// Creates a new distributed device that connects to the provided rendezvous group.
    pub fn new(group: impl Into<String>, rank: usize, world_size: usize) -> Self {
        let session = st_distributed::rendezvous(group.into(), rank, world_size);
        Self {
            session,
            strategy: SyncStrategy::AllReduce,
        }
    }

    fn all_reduce(&self, buffer: &mut [f32]) {
        st_distributed::all_reduce(&self.session, buffer);
        if self.strategy == SyncStrategy::AllReduce && self.session.world_size() > 0 {
            let scale = 1.0 / self.session.world_size() as f32;
            buffer.iter_mut().for_each(|v| *v *= scale);
        }
    }
}

impl TrainingDevice for DistributedDevice {
    fn rank(&self) -> usize {
        self.session.rank()
    }

    fn world_size(&self) -> usize {
        self.session.world_size()
    }

    fn synchronize_gradients(&self, gradients: &mut [f32]) {
        self.all_reduce(gradients);
    }

    fn aggregate_metrics(&self, metrics: &mut [f32], reduce: MetricReduce) {
        st_distributed::all_reduce(&self.session, metrics);
        if reduce == MetricReduce::Mean {
            let scale = 1.0 / self.world_size() as f32;
            metrics.iter_mut().for_each(|value| *value *= scale);
        }
    }
}
