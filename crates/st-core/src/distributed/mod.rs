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

pub const ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_KIND: &str =
    "spiraltorch.accumulator_synchronizer_checkpoint";
pub const ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_CONTRACT_VERSION: &str =
    "spiraltorch.accumulator_synchronizer_checkpoint.v1";
pub const ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_SEMANTIC_OWNER: &str =
    "st-core::distributed::AccumulatorSynchronizer";
pub const ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_SEMANTIC_BACKEND: &str = "rust";
pub const ACCUMULATOR_SYNCHRONIZER_OPAQUE_PROVIDER: &str =
    "spiraltorch.accumulator_synchronizer.opaque";
pub const ACCUMULATOR_SYNCHRONIZER_MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

/// Stable identity and optional provider state for an attached collective.
///
/// The checkpoint does not create network resources. Orchestrators must attach
/// a compatible provider before asking Rust to validate and restore it.
#[derive(Clone, Debug, serde::Deserialize, PartialEq, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct AccumulatorSynchronizerCheckpoint {
    pub kind: String,
    pub contract_version: String,
    pub semantic_owner: String,
    pub semantic_backend: String,
    pub provider: String,
    pub rank: u64,
    pub world_size: u64,
    pub state: Option<serde_json::Value>,
}

impl AccumulatorSynchronizerCheckpoint {
    pub fn validate(&self) -> Result<(), AccumulatorSyncError> {
        for (field, actual, expected) in [
            (
                "kind",
                self.kind.as_str(),
                ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_KIND,
            ),
            (
                "contract_version",
                self.contract_version.as_str(),
                ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_CONTRACT_VERSION,
            ),
            (
                "semantic_owner",
                self.semantic_owner.as_str(),
                ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_SEMANTIC_OWNER,
            ),
            (
                "semantic_backend",
                self.semantic_backend.as_str(),
                ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_SEMANTIC_BACKEND,
            ),
        ] {
            if actual != expected {
                return Err(AccumulatorSyncError::backend(format!(
                    "checkpoint {field} must be {expected}, got {actual}"
                )));
            }
        }
        if self.provider.trim().is_empty() {
            return Err(AccumulatorSyncError::backend(
                "checkpoint provider must not be empty",
            ));
        }
        if self.world_size == 0 || self.rank >= self.world_size {
            return Err(AccumulatorSyncError::backend(format!(
                "checkpoint rank {} must be within world size {}",
                self.rank, self.world_size
            )));
        }
        if self.rank > ACCUMULATOR_SYNCHRONIZER_MAX_SAFE_INTEGER
            || self.world_size > ACCUMULATOR_SYNCHRONIZER_MAX_SAFE_INTEGER
        {
            return Err(AccumulatorSyncError::backend(format!(
                "checkpoint rank and world size must not exceed JavaScript's exact integer limit {}",
                ACCUMULATOR_SYNCHRONIZER_MAX_SAFE_INTEGER
            )));
        }
        if self.state.as_ref().is_some_and(|state| !state.is_object()) {
            return Err(AccumulatorSyncError::backend(
                "checkpoint provider state must be a JSON object",
            ));
        }
        Ok(())
    }

    pub fn requires_opaque_reattach(&self) -> bool {
        self.provider == ACCUMULATOR_SYNCHRONIZER_OPAQUE_PROVIDER
    }
}

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

    /// Stable provider identifier used by trainer runtime checkpoints.
    fn checkpoint_provider(&self) -> &'static str {
        ACCUMULATOR_SYNCHRONIZER_OPAQUE_PROVIDER
    }

    /// Optional provider-owned state. The default provider is identity-only.
    fn checkpoint_state(&self) -> Result<Option<serde_json::Value>, AccumulatorSyncError> {
        Ok(None)
    }

    /// Validates provider state against an already attached runtime resource.
    fn validate_checkpoint_state(
        &self,
        state: Option<&serde_json::Value>,
    ) -> Result<(), AccumulatorSyncError> {
        if state.is_some() {
            Err(AccumulatorSyncError::backend(
                "opaque accumulator synchronizer cannot restore provider state",
            ))
        } else {
            Ok(())
        }
    }

    fn checkpoint(&self) -> Result<AccumulatorSynchronizerCheckpoint, AccumulatorSyncError> {
        let rank = u64::try_from(self.rank())
            .map_err(|_| AccumulatorSyncError::backend("rank does not fit u64"))?;
        let world_size = u64::try_from(self.world_size())
            .map_err(|_| AccumulatorSyncError::backend("world size does not fit u64"))?;
        let checkpoint = AccumulatorSynchronizerCheckpoint {
            kind: ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_KIND.to_owned(),
            contract_version: ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_CONTRACT_VERSION.to_owned(),
            semantic_owner: ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_SEMANTIC_OWNER.to_owned(),
            semantic_backend: ACCUMULATOR_SYNCHRONIZER_CHECKPOINT_SEMANTIC_BACKEND.to_owned(),
            provider: self.checkpoint_provider().to_owned(),
            rank,
            world_size,
            state: self.checkpoint_state()?,
        };
        checkpoint.validate()?;
        Ok(checkpoint)
    }

    fn validate_checkpoint(
        &self,
        checkpoint: &AccumulatorSynchronizerCheckpoint,
    ) -> Result<(), AccumulatorSyncError> {
        checkpoint.validate()?;
        if checkpoint.provider != self.checkpoint_provider() {
            return Err(AccumulatorSyncError::backend(format!(
                "checkpoint provider mismatch: expected {}, got {}",
                self.checkpoint_provider(),
                checkpoint.provider
            )));
        }
        let rank = u64::try_from(self.rank())
            .map_err(|_| AccumulatorSyncError::backend("rank does not fit u64"))?;
        let world_size = u64::try_from(self.world_size())
            .map_err(|_| AccumulatorSyncError::backend("world size does not fit u64"))?;
        if checkpoint.rank != rank || checkpoint.world_size != world_size {
            return Err(AccumulatorSyncError::backend(format!(
                "checkpoint topology mismatch: expected rank {rank}/{world_size}, got {}/{}",
                checkpoint.rank, checkpoint.world_size
            )));
        }
        self.validate_checkpoint_state(checkpoint.state.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestSynchronizer {
        rank: usize,
        world_size: usize,
    }

    impl AccumulatorSynchronizer for TestSynchronizer {
        fn rank(&self) -> usize {
            self.rank
        }

        fn world_size(&self) -> usize {
            self.world_size
        }

        fn synchronize_accumulators(
            &self,
            _gradients: &mut [f32],
        ) -> Result<(), AccumulatorSyncError> {
            Ok(())
        }
    }

    #[test]
    fn opaque_synchronizer_checkpoint_roundtrips_and_verifies_topology() {
        let synchronizer = TestSynchronizer {
            rank: 1,
            world_size: 3,
        };
        let encoded = serde_json::to_string(&synchronizer.checkpoint().unwrap()).unwrap();
        let checkpoint: AccumulatorSynchronizerCheckpoint = serde_json::from_str(&encoded).unwrap();

        assert!(checkpoint.requires_opaque_reattach());
        synchronizer.validate_checkpoint(&checkpoint).unwrap();

        let wrong = TestSynchronizer {
            rank: 0,
            world_size: 3,
        };
        assert!(wrong.validate_checkpoint(&checkpoint).is_err());
    }

    #[test]
    fn synchronizer_checkpoint_rejects_unknown_fields() {
        let synchronizer = TestSynchronizer {
            rank: 0,
            world_size: 1,
        };
        let mut payload = serde_json::to_value(synchronizer.checkpoint().unwrap()).unwrap();
        payload
            .as_object_mut()
            .unwrap()
            .insert("commander".to_owned(), serde_json::json!("python"));
        let error =
            serde_json::from_value::<AccumulatorSynchronizerCheckpoint>(payload).unwrap_err();
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn synchronizer_checkpoint_rejects_cross_runtime_integer_overflow() {
        let synchronizer = TestSynchronizer {
            rank: 0,
            world_size: 1,
        };
        let mut checkpoint = synchronizer.checkpoint().unwrap();
        checkpoint.world_size = ACCUMULATOR_SYNCHRONIZER_MAX_SAFE_INTEGER + 1;
        assert!(checkpoint
            .validate()
            .unwrap_err()
            .to_string()
            .contains("exact integer limit"));
    }
}
