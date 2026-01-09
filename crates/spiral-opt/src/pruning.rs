use crate::{
    ops::block::{compute_block_norms, zero_block, BlockRecord},
    report::{OptimisationError, StructuredPruningReport},
};

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

impl Default for StructuredPruner {
    fn default() -> Self {
        Self
    }
}

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

        let block_count = weights.len().div_ceil(config.block_size);
        workspace.prepare(block_count);
        compute_block_norms(
            weights,
            config.block_size,
            block_count,
            &mut workspace.block_norms,
        );

        let block_total = workspace.block_norms.len();
        let target_blocks = (block_total as f32 * config.target_sparsity).floor() as usize;
        let mut pruned_blocks = 0;
        let mut l2_error = 0.0f32;
        let min_keep_sq = config.min_l2_keep * config.min_l2_keep;

        if target_blocks == 0 {
            for block in &workspace.block_norms {
                if block.norm_sq < min_keep_sq {
                    l2_error += block.norm_sq;
                    zero_block(weights, block);
                    pruned_blocks += 1;
                }
            }
        } else if target_blocks >= block_total {
            for block in &workspace.block_norms {
                l2_error += block.norm_sq;
                zero_block(weights, block);
                pruned_blocks += 1;
            }
        } else {
            // Partition blocks so the `target_blocks` smallest norms occupy the front without the
            // `O(n log n)` cost of a full sort.
            let nth_index = target_blocks - 1;
            workspace
                .block_norms
                .select_nth_unstable_by(nth_index, |a, b| a.norm_sq.total_cmp(&b.norm_sq));
            let (to_prune, remainder) = workspace.block_norms.split_at_mut(target_blocks);

            for block in to_prune.iter() {
                l2_error += block.norm_sq;
                zero_block(weights, block);
                pruned_blocks += 1;
            }

            for block in remainder.iter() {
                if block.norm_sq < min_keep_sq {
                    l2_error += block.norm_sq;
                    zero_block(weights, block);
                    pruned_blocks += 1;
                }
            }
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

    #[test]
    fn l2_error_matches_pruned_energy() {
        let mut weights = vec![1.0f32; 12];
        let original = weights.clone();
        let pruner = StructuredPruner::new();
        let config = StructuredPruningConfig {
            block_size: 4,
            target_sparsity: 0.5,
            min_l2_keep: 0.0,
        };

        let report = pruner
            .apply(&mut weights, config)
            .expect("pruning should succeed");

        let observed_energy = original
            .iter()
            .zip(&weights)
            .filter(|(_, &after)| after == 0.0)
            .map(|(&before, _)| before * before)
            .sum::<f32>()
            .sqrt();

        assert!((report.l2_error - observed_energy).abs() < 1e-6);
    }

    #[test]
    fn min_norm_threshold_preserves_strong_blocks() {
        let mut weights = vec![
            5.0, 5.0, // strong block 0
            4.0, 4.0, // strong block 1
            0.1, 0.1, // weak block 2
            0.2, 0.2, // weak block 3
        ];
        let pruner = StructuredPruner::new();
        let config = StructuredPruningConfig {
            block_size: 2,
            target_sparsity: 0.25,
            min_l2_keep: 5.0,
        };

        let report = pruner
            .apply(&mut weights, config)
            .expect("pruning should succeed");

        assert_eq!(&weights[..4], &[5.0, 5.0, 4.0, 4.0]);
        assert_eq!(&weights[4..], &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(report.pruned_blocks, 2);
        assert_eq!(report.kept_blocks, 2);
        assert!((report.achieved_sparsity - 0.5).abs() < 1e-6);
        let expected_error = (0.1f32 * 0.1 * 2.0 + 0.2 * 0.2 * 2.0).sqrt();
        assert!((report.l2_error - expected_error).abs() < 1e-6);
    }

    #[test]
    fn zero_target_only_prunes_below_threshold() {
        let mut weights = vec![
            0.05, 0.05, // weak block 0
            0.2, 0.2, // weak block 1
            2.0, 2.0, // strong block 2
        ];
        let pruner = StructuredPruner::new();
        let config = StructuredPruningConfig {
            block_size: 2,
            target_sparsity: 0.0,
            min_l2_keep: 0.3,
        };

        let report = pruner
            .apply(&mut weights, config)
            .expect("pruning should succeed");

        assert_eq!(&weights[..4], &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(&weights[4..], &[2.0, 2.0]);
        assert_eq!(report.pruned_blocks, 2);
        assert_eq!(report.kept_blocks, 1);
        assert!((report.achieved_sparsity - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn partitioning_removes_smallest_blocks_first() {
        let mut weights = vec![
            5.0, 5.0, // block 0
            4.0, 4.0, // block 1
            3.0, 3.0, // block 2
            2.0, 2.0, // block 3
            1.0, 1.0, // block 4
        ];
        let pruner = StructuredPruner::new();
        let config = StructuredPruningConfig {
            block_size: 2,
            target_sparsity: 0.4, // prune floor(5 * 0.4) == 2 blocks
            min_l2_keep: 0.0,
        };

        let report = pruner
            .apply(&mut weights, config)
            .expect("pruning should succeed");

        assert_eq!(&weights[..6], &[5.0, 5.0, 4.0, 4.0, 3.0, 3.0]);
        assert_eq!(&weights[6..], &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(report.pruned_blocks, 2);
        assert_eq!(report.kept_blocks, 3);
        assert!((report.achieved_sparsity - 0.4).abs() < 1e-6);
    }
}
