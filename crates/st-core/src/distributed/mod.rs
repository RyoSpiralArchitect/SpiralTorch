// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

pub mod autograd;
pub mod collective;
pub mod prob_params;
pub mod topk3_stage;
pub mod topk_dist;
pub mod trainer;

/// Error returned by accumulator synchronization backends.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum AccumulatorSyncError {
    #[error("accumulator synchronization backend failed: {0}")]
    Backend(String),
}

impl AccumulatorSyncError {
    pub fn backend(error: impl ToString) -> Self {
        Self::Backend(error.to_string())
    }
}

/// Minimal trait shared by trainers that can synchronize parameter accumulators.
pub trait AccumulatorSynchronizer: Send + Sync {
    /// Identifier of the current participant.
    fn rank(&self) -> usize;
    /// Number of participants contributing accumulator buffers.
    fn world_size(&self) -> usize;
    /// Synchronizes one flat accumulator buffer in-place.
    fn synchronize_accumulators(&self, gradients: &mut [f32]) -> Result<(), AccumulatorSyncError>;
}
