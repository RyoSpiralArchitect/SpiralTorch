use crate::report::{OptimisationError, StructuredPruningReport};

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

        let mut block_norms = Vec::new();
        let mut offset = 0;
        while offset < weights.len() {
            let block_end = (offset + config.block_size).min(weights.len());
            let block = &weights[offset..block_end];
            let l2 = block.iter().map(|w| w * w).sum::<f32>().sqrt();
            block_norms.push((offset, block_end, l2));
            offset = block_end;
        }

        block_norms.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        let target_blocks = ((block_norms.len() as f32) * config.target_sparsity).floor() as usize;
        let mut pruned_blocks = 0;
        let mut l2_error = 0.0;

        for (idx, (start, end, norm)) in block_norms.iter().enumerate() {
            if idx >= target_blocks && *norm >= config.min_l2_keep {
                continue;
            }
            for w in &mut weights[*start..*end] {
                l2_error += *w * *w;
                *w = 0.0;
            }
            pruned_blocks += 1;
        }

        let kept_blocks = block_norms.len().saturating_sub(pruned_blocks);
        let achieved_sparsity = if block_norms.is_empty() {
            0.0
        } else {
            pruned_blocks as f32 / block_norms.len() as f32
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
}
