//! Compatibility aliases for the older MidK-named compaction helpers.
//!
//! New code should prefer `ops::compaction`.

pub use super::compaction::{CompactionError, CompactionOut};

pub fn midk_compact_between(
    x: &[f32],
    rows: u32,
    cols: u32,
    row_stride: u32,
    lower: f32,
    upper: f32,
) -> Result<CompactionOut, CompactionError> {
    super::compaction::compact_between(x, rows, cols, row_stride, lower, upper)
}

pub fn bottomk_compact_below(
    x: &[f32],
    rows: u32,
    cols: u32,
    row_stride: u32,
    upper: f32,
) -> Result<CompactionOut, CompactionError> {
    super::compaction::compact_below(x, rows, cols, row_stride, upper)
}

pub fn topk_compact_above(
    x: &[f32],
    rows: u32,
    cols: u32,
    row_stride: u32,
    lower: f32,
) -> Result<CompactionOut, CompactionError> {
    super::compaction::compact_above(x, rows, cols, row_stride, lower)
}
