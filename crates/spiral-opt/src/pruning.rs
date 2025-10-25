use crate::report::{OptimisationError, StructuredPruningReport};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy)]
struct BlockRecord {
    start: usize,
    len: usize,
    norm_sq: f32,
}

impl BlockRecord {
    #[inline]
    fn new(start: usize, len: usize, norm_sq: f32) -> Self {
        Self {
            start,
            len,
            norm_sq,
        }
    }
}

/// Configuration for channel/block structured pruning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StructuredPruningConfig {
    pub block_size: usize,
    pub target_sparsity: f32,
    /// Minimum L2 norm that a block must maintain to be preserved regardless
    /// of global sparsity targets.  This is useful for preventing important
    /// channels from being accidentally removed when they hold small values.
    pub min_l2_keep: f32,
}

impl Default for StructuredPruningConfig {
    fn default() -> Self {
        Self {
            block_size: 32,
            target_sparsity: 0.5,
            min_l2_keep: 1e-4,
        }
    }
}

/// Stateless helper applying structured pruning decisions to raw weights.
#[derive(Debug, Clone, Copy)]
pub struct StructuredPruner;

impl StructuredPruner {
    pub fn new() -> Self {
        Self
    }

    pub fn apply(
        &self,
        weights: &mut [f32],
        config: StructuredPruningConfig,
    ) -> Result<StructuredPruningReport, OptimisationError> {
        let mut workspace = StructuredPruningWorkspace::default();
        self.apply_with_workspace(weights, config, &mut workspace)
    }

    pub fn apply_with_workspace(
        &self,
        weights: &mut [f32],
        config: StructuredPruningConfig,
        workspace: &mut StructuredPruningWorkspace,
    ) -> Result<StructuredPruningReport, OptimisationError> {
        if config.block_size == 0 {
            return Err(OptimisationError::InvalidBlockSize);
        }
        if !(0.0..=1.0).contains(&config.target_sparsity) {
            return Err(OptimisationError::InvalidSparsity {
                target: config.target_sparsity,
            });
        }
        if weights.is_empty() {
            return Ok(StructuredPruningReport::new(
                config.target_sparsity,
                0.0,
                config.block_size,
                0,
                0,
                0.0,
            ));
        }

        let block_count = (weights.len() + config.block_size - 1) / config.block_size;
        workspace.prepare(block_count);

        let mut offset = 0;
        for chunk in weights.chunks(config.block_size) {
            let norm_sq = chunk.iter().map(|w| *w * *w).sum::<f32>();
            workspace
                .block_norms
                .push(BlockRecord::new(offset, chunk.len(), norm_sq));
            offset += chunk.len();
        }

        workspace
            .block_norms
            .sort_unstable_by(|a, b| a.norm_sq.partial_cmp(&b.norm_sq).unwrap_or(Ordering::Equal));
        let target_blocks =
            ((workspace.block_norms.len() as f32) * config.target_sparsity).floor() as usize;
        let mut pruned_blocks = 0;
        let mut l2_error = 0.0f32;
        let min_keep_sq = config.min_l2_keep * config.min_l2_keep;

        for (idx, block) in workspace.block_norms.iter().enumerate() {
            if idx >= target_blocks && block.norm_sq >= min_keep_sq {
                continue;
            }
            let range = block.start..block.start + block.len;
            l2_error += block.norm_sq;
            weights[range].fill(0.0);
            pruned_blocks += 1;
        }

        let kept_blocks = workspace.block_norms.len().saturating_sub(pruned_blocks);
        let achieved_sparsity = if workspace.block_norms.is_empty() {
            0.0
        } else {
            pruned_blocks as f32 / workspace.block_norms.len() as f32
        };
        Ok(StructuredPruningReport::new(
            config.target_sparsity,
            achieved_sparsity,
            config.block_size,
            pruned_blocks,
            kept_blocks,
            l2_error.sqrt(),
        ))
    }
}

#[derive(Debug, Default)]
pub struct StructuredPruningWorkspace {
    block_norms: Vec<BlockRecord>,
}

impl StructuredPruningWorkspace {
    pub fn new() -> Self {
        Self {
            block_norms: Vec::new(),
        }
    }

    pub fn with_capacity(blocks: usize) -> Self {
        Self {
            block_norms: Vec::with_capacity(blocks),
        }
    }

    fn prepare(&mut self, required_blocks: usize) {
        if self.block_norms.capacity() < required_blocks {
            self.block_norms
                .reserve(required_blocks - self.block_norms.capacity());
        }
        self.block_norms.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pruning_respects_block_structure() {
        let mut weights = vec![1.0f32; 16];
        let pruner = StructuredPruner::new();
        let report = pruner
            .apply(
                &mut weights,
                StructuredPruningConfig {
                    block_size: 4,
                    target_sparsity: 0.5,
                    min_l2_keep: 0.0,
                },
            )
            .unwrap();
        assert_eq!(report.block_size, 4);
        assert_eq!(report.pruned_blocks + report.kept_blocks, 4);
        assert!(report.achieved_sparsity >= 0.25);
    }

    #[test]
    fn workspace_reuse_matches_default() {
        let mut weights_default = vec![1.0f32; 64];
        let mut weights_workspace = weights_default.clone();
        let pruner = StructuredPruner::new();
        let config = StructuredPruningConfig {
            block_size: 8,
            target_sparsity: 0.25,
            min_l2_keep: 0.0,
        };

        let report_default = pruner
            .apply(&mut weights_default, config)
            .expect("default application should succeed");

        let mut workspace = StructuredPruningWorkspace::new();
        let report_workspace = pruner
            .apply_with_workspace(&mut weights_workspace, config, &mut workspace)
            .expect("workspace application should succeed");

        assert_eq!(weights_default, weights_workspace);
        assert_eq!(report_default, report_workspace);
    }
}
