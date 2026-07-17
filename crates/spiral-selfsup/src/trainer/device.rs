// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::sync::Arc;
use std::time::Duration;

use super::distributed::st_distributed::{self, DistributedError};
use st_core::backend::execution::current_tensor_util_backend_for_values;
use st_core::distributed::{AccumulatorSyncError, AccumulatorSynchronizer};
use st_tensor::{Tensor, TensorError};
use thiserror::Error;

const CPU_ACCUMULATOR_PROVIDER: &str = "spiral-selfsup.cpu_accumulator.v1";
const DISTRIBUTED_ACCUMULATOR_PROVIDER: &str = "spiral-selfsup.distributed_accumulator.v1";

#[derive(Debug, Clone, serde::Deserialize, PartialEq, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct DistributedAccumulatorCheckpointState {
    group_id: String,
    strategy: String,
    collective_timeout_nanos: u64,
}

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
    fn synchronize_gradients(&self, gradients: &mut [f32]) -> Result<(), TrainingDeviceError>;
    /// Aggregates the provided metrics in-place according to the reduction policy.
    fn aggregate_metrics(
        &self,
        metrics: &mut [f32],
        reduce: MetricReduce,
    ) -> Result<(), TrainingDeviceError>;
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

    fn synchronize_gradients(&self, _gradients: &mut [f32]) -> Result<(), TrainingDeviceError> {
        Ok(())
    }

    fn aggregate_metrics(
        &self,
        _metrics: &mut [f32],
        _reduce: MetricReduce,
    ) -> Result<(), TrainingDeviceError> {
        Ok(())
    }
}

impl AccumulatorSynchronizer for CpuDevice {
    fn rank(&self) -> usize {
        TrainingDevice::rank(self)
    }

    fn world_size(&self) -> usize {
        TrainingDevice::world_size(self)
    }

    fn synchronize_accumulators(&self, gradients: &mut [f32]) -> Result<(), AccumulatorSyncError> {
        TrainingDevice::synchronize_gradients(self, gradients)
            .map_err(AccumulatorSyncError::backend)
    }

    fn checkpoint_provider(&self) -> &'static str {
        CPU_ACCUMULATOR_PROVIDER
    }
}

/// Distributed device backed by a rendezvous session.
#[derive(Debug, Clone)]
pub struct DistributedDevice {
    session: Arc<st_distributed::RendezvousSession>,
    strategy: SyncStrategy,
    collective_timeout: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncStrategy {
    /// Synchronize gradients via an all-reduce sum and average locally.
    AllReduce,
}

impl DistributedDevice {
    /// Creates a new distributed device that connects to the provided rendezvous group.
    pub fn new(
        group: impl Into<String>,
        rank: usize,
        world_size: usize,
    ) -> Result<Self, TrainingDeviceError> {
        let session = st_distributed::rendezvous(group.into(), rank, world_size)?;
        Ok(Self {
            session,
            strategy: SyncStrategy::AllReduce,
            collective_timeout: st_distributed::DEFAULT_COLLECTIVE_TIMEOUT,
        })
    }

    /// Overrides the maximum time each collective may wait for every rank.
    pub fn with_collective_timeout(mut self, timeout: Duration) -> Self {
        self.collective_timeout = timeout;
        self
    }

    fn scale_buffer_with_policy(buffer: &mut [f32], scale: f32) -> Result<(), TrainingDeviceError> {
        if buffer.is_empty() {
            return Ok(());
        }
        let backend = current_tensor_util_backend_for_values(buffer.len());
        let tensor = Tensor::from_vec(1, buffer.len(), buffer.to_vec())?;
        let scaled = tensor.scale_with_backend(scale, backend)?;
        buffer.copy_from_slice(scaled.data());
        Ok(())
    }

    fn all_reduce(&self, buffer: &mut [f32]) -> Result<(), TrainingDeviceError> {
        let original = buffer.to_vec();
        st_distributed::all_reduce_with_timeout(&self.session, buffer, self.collective_timeout)?;
        if self.strategy == SyncStrategy::AllReduce && self.session.world_size() > 0 {
            let scale = 1.0 / self.session.world_size() as f32;
            if let Err(error) = Self::scale_buffer_with_policy(buffer, scale) {
                buffer.copy_from_slice(&original);
                return Err(error);
            }
        }
        Ok(())
    }
}

impl TrainingDevice for DistributedDevice {
    fn rank(&self) -> usize {
        self.session.rank()
    }

    fn world_size(&self) -> usize {
        self.session.world_size()
    }

    fn synchronize_gradients(&self, gradients: &mut [f32]) -> Result<(), TrainingDeviceError> {
        self.all_reduce(gradients)
    }

    fn aggregate_metrics(
        &self,
        metrics: &mut [f32],
        reduce: MetricReduce,
    ) -> Result<(), TrainingDeviceError> {
        let original = metrics.to_vec();
        st_distributed::all_reduce_with_timeout(&self.session, metrics, self.collective_timeout)?;
        if reduce == MetricReduce::Mean {
            let scale = 1.0 / TrainingDevice::world_size(self) as f32;
            if let Err(error) = Self::scale_buffer_with_policy(metrics, scale) {
                metrics.copy_from_slice(&original);
                return Err(error);
            }
        }
        Ok(())
    }
}

impl AccumulatorSynchronizer for DistributedDevice {
    fn rank(&self) -> usize {
        TrainingDevice::rank(self)
    }

    fn world_size(&self) -> usize {
        TrainingDevice::world_size(self)
    }

    fn synchronize_accumulators(&self, gradients: &mut [f32]) -> Result<(), AccumulatorSyncError> {
        TrainingDevice::synchronize_gradients(self, gradients)
            .map_err(AccumulatorSyncError::backend)
    }

    fn checkpoint_provider(&self) -> &'static str {
        DISTRIBUTED_ACCUMULATOR_PROVIDER
    }

    fn checkpoint_state(&self) -> Result<Option<serde_json::Value>, AccumulatorSyncError> {
        if self.session.collective_in_flight() {
            return Err(AccumulatorSyncError::backend(
                "cannot checkpoint a distributed collective in flight",
            ));
        }
        let collective_timeout_nanos = u64::try_from(self.collective_timeout.as_nanos())
            .map_err(|_| AccumulatorSyncError::backend("collective timeout does not fit u64"))?;
        let state = DistributedAccumulatorCheckpointState {
            group_id: self.session.group_id().to_owned(),
            strategy: "all_reduce_mean".to_owned(),
            collective_timeout_nanos,
        };
        serde_json::to_value(state)
            .map(Some)
            .map_err(AccumulatorSyncError::backend)
    }

    fn validate_checkpoint_state(
        &self,
        state: Option<&serde_json::Value>,
    ) -> Result<(), AccumulatorSyncError> {
        if self.session.collective_in_flight() {
            return Err(AccumulatorSyncError::backend(
                "cannot restore while a distributed collective is in flight",
            ));
        }
        let state = state.ok_or_else(|| {
            AccumulatorSyncError::backend("distributed checkpoint state is missing")
        })?;
        let state: DistributedAccumulatorCheckpointState =
            serde_json::from_value(state.clone()).map_err(AccumulatorSyncError::backend)?;
        let collective_timeout_nanos = u64::try_from(self.collective_timeout.as_nanos())
            .map_err(|_| AccumulatorSyncError::backend("collective timeout does not fit u64"))?;
        let expected = DistributedAccumulatorCheckpointState {
            group_id: self.session.group_id().to_owned(),
            strategy: "all_reduce_mean".to_owned(),
            collective_timeout_nanos,
        };
        if state != expected {
            return Err(AccumulatorSyncError::backend(format!(
                "distributed checkpoint state mismatch: expected {expected:?}, got {state:?}"
            )));
        }
        Ok(())
    }
}

/// Errors surfaced by a [`TrainingDevice`] implementation.
#[derive(Debug, Error, PartialEq)]
pub enum TrainingDeviceError {
    /// Raised when rendezvous metadata is invalid or inconsistent.
    #[error("distributed rendezvous failed: {0}")]
    Rendezvous(#[from] DistributedError),
    /// Raised when policy-routed tensor post-processing fails.
    #[error("distributed tensor operation failed: {0}")]
    Tensor(#[from] TensorError),
}
