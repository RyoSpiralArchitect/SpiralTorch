use thiserror::Error;

/// Errors that can be emitted by optimisation utilities.
#[derive(Debug, Error)]
pub enum OptimisationError {
    #[error("invalid sparsity target {target} (expected 0.0..=1.0)")]
    InvalidSparsity { target: f32 },
    #[error("block size must be greater than zero")]
    InvalidBlockSize,
}

/// Compact summary emitted by the quantisation step.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "report-serde", derive(serde::Serialize, serde::Deserialize))]
pub struct QuantizationReport {
    pub bit_width: u8,
    pub observed_min: f32,
    pub observed_max: f32,
    pub scale: f32,
    pub zero_point: f32,
    pub quant_error: f32,
    pub observed_steps: u64,
}

impl QuantizationReport {
    pub(crate) fn new(
        bit_width: u8,
        observed_min: f32,
        observed_max: f32,
        scale: f32,
        zero_point: f32,
        quant_error: f32,
        observed_steps: u64,
    ) -> Self {
        Self {
            bit_width,
            observed_min,
            observed_max,
            scale,
            zero_point,
            quant_error,
            observed_steps,
        }
    }

    pub(crate) fn empty(bit_width: u8) -> Self {
        Self::new(bit_width, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    }
}

/// Summary describing structured pruning results.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "report-serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StructuredPruningReport {
    pub target_sparsity: f32,
    pub achieved_sparsity: f32,
    pub block_size: usize,
    pub pruned_blocks: usize,
    pub kept_blocks: usize,
    pub l2_error: f32,
}

impl StructuredPruningReport {
    pub(crate) fn new(
        target_sparsity: f32,
        achieved_sparsity: f32,
        block_size: usize,
        pruned_blocks: usize,
        kept_blocks: usize,
        l2_error: f32,
    ) -> Self {
        Self {
            target_sparsity,
            achieved_sparsity,
            block_size,
            pruned_blocks,
            kept_blocks,
            l2_error,
        }
    }
}

/// Combined summary for quantisation + pruning pipelines.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "report-serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CompressionReport {
    pub original_params: usize,
    pub remaining_params: usize,
    pub quantization: Option<QuantizationReport>,
    pub pruning: Option<StructuredPruningReport>,
    pub estimated_latency_reduction: f32,
}

impl CompressionReport {
    pub fn new(
        original_params: usize,
        remaining_params: usize,
        quantization: Option<QuantizationReport>,
        pruning: Option<StructuredPruningReport>,
        estimated_latency_reduction: f32,
    ) -> Self {
        Self {
            original_params,
            remaining_params,
            quantization,
            pruning,
            estimated_latency_reduction,
        }
    }
}
