//! Threshold/mask compaction helpers.
//!
//! This module provides CPU reference implementations for row-wise compaction
//! passes: given a predicate (for example `lower <= x <= upper`), pack the
//! selected values per row into a dense prefix while also emitting the original
//! column indices.
//!
//! The output layout mirrors the WGPU compaction kernels in
//! `backend/wgpu_kernels_rankk.wgsl`: for each row `r`, the first `counts[r]`
//! entries are valid and stable-ordered by increasing column index.

use crate::backend::device_caps::DeviceCaps;
use crate::backend::unison_heuristics::{self, CompactionChoice};

/// Planned threshold compaction dispatch.
#[derive(Clone, Debug)]
pub struct CompactionPlan {
    pub rows: u32,
    pub cols: u32,
    pub choice: CompactionChoice,
}

impl CompactionPlan {
    pub fn to_unison_script(&self) -> String {
        self.choice.to_unison_script()
    }
}

/// Packed compaction output (row-wise).
#[derive(Clone, Debug, PartialEq)]
pub struct CompactionOut {
    /// Packed values. Length is `rows * cols` (row-major). Only the first
    /// `counts[r]` entries of each row are valid.
    pub values: Vec<f32>,
    /// Packed column indices for the corresponding `values`. Length is
    /// `rows * cols`. Invalid entries are set to `u32::MAX`.
    pub indices: Vec<u32>,
    /// Number of selected values for each row. Length is `rows`.
    pub counts: Vec<u32>,
    pub rows: u32,
    pub cols: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompactionError {
    InvalidShape,
    NonFiniteThreshold,
}

pub fn plan_compaction(rows: u32, cols: u32, caps: DeviceCaps) -> CompactionPlan {
    let choice = unison_heuristics::choose_unified_compaction(rows, cols, caps);
    CompactionPlan { rows, cols, choice }
}

fn validate_dense(
    rows: u32,
    cols: u32,
    row_stride: u32,
    x_len: usize,
) -> Result<(), CompactionError> {
    if rows == 0 || cols == 0 {
        return Err(CompactionError::InvalidShape);
    }
    if row_stride < cols {
        return Err(CompactionError::InvalidShape);
    }
    let needed = rows as usize * row_stride as usize;
    if x_len < needed {
        return Err(CompactionError::InvalidShape);
    }
    Ok(())
}

fn compact_where<F>(
    x: &[f32],
    rows: u32,
    cols: u32,
    row_stride: u32,
    mut pred: F,
) -> Result<CompactionOut, CompactionError>
where
    F: FnMut(f32) -> bool,
{
    validate_dense(rows, cols, row_stride, x.len())?;

    let mut values = vec![0.0f32; rows as usize * cols as usize];
    let mut indices = vec![u32::MAX; rows as usize * cols as usize];
    let mut counts = vec![0u32; rows as usize];

    for r in 0..rows as usize {
        let row_base_in = r * row_stride as usize;
        let row_base_out = r * cols as usize;
        let mut out_pos = 0usize;
        for c in 0..cols as usize {
            let v = x[row_base_in + c];
            if pred(v) {
                values[row_base_out + out_pos] = v;
                indices[row_base_out + out_pos] = c as u32;
                out_pos += 1;
            }
        }
        counts[r] = out_pos as u32;
    }

    Ok(CompactionOut {
        values,
        indices,
        counts,
        rows,
        cols,
    })
}

/// Select values where `lower <= x <= upper` (inclusive).
pub fn compact_between(
    x: &[f32],
    rows: u32,
    cols: u32,
    row_stride: u32,
    lower: f32,
    upper: f32,
) -> Result<CompactionOut, CompactionError> {
    if !lower.is_finite() || !upper.is_finite() {
        return Err(CompactionError::NonFiniteThreshold);
    }
    if lower > upper {
        return Ok(CompactionOut {
            values: vec![0.0; rows as usize * cols as usize],
            indices: vec![u32::MAX; rows as usize * cols as usize],
            counts: vec![0u32; rows as usize],
            rows,
            cols,
        });
    }
    compact_where(x, rows, cols, row_stride, |v| v >= lower && v <= upper)
}

/// Select values where `x <= upper` (inclusive).
pub fn compact_below(
    x: &[f32],
    rows: u32,
    cols: u32,
    row_stride: u32,
    upper: f32,
) -> Result<CompactionOut, CompactionError> {
    if !upper.is_finite() {
        return Err(CompactionError::NonFiniteThreshold);
    }
    compact_where(x, rows, cols, row_stride, |v| v <= upper)
}

/// Select values where `x >= lower` (inclusive).
pub fn compact_above(
    x: &[f32],
    rows: u32,
    cols: u32,
    row_stride: u32,
    lower: f32,
) -> Result<CompactionOut, CompactionError> {
    if !lower.is_finite() {
        return Err(CompactionError::NonFiniteThreshold);
    }
    compact_where(x, rows, cols, row_stride, |v| v >= lower)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compacts_between_stably_with_indices() {
        let x = vec![0.0, 2.0, 1.0, 4.0, 3.0, 5.0];
        let out = compact_between(&x, 1, 6, 6, 2.0, 4.0).unwrap();
        assert_eq!(out.counts, vec![3]);
        assert_eq!(&out.values[..6], &[2.0, 4.0, 3.0, 0.0, 0.0, 0.0]);
        assert_eq!(&out.indices[..6], &[1, 3, 4, u32::MAX, u32::MAX, u32::MAX]);
    }

    #[test]
    fn compacts_below_two_rows() {
        let x = vec![
            0.0, 1.0, 2.0, 3.0, // row0
            3.0, 2.0, 1.0, 0.0, // row1
        ];
        let out = compact_below(&x, 2, 4, 4, 1.0).unwrap();
        assert_eq!(out.counts, vec![2, 2]);
        assert_eq!(&out.values[..4], &[0.0, 1.0, 0.0, 0.0]);
        assert_eq!(&out.indices[..4], &[0, 1, u32::MAX, u32::MAX]);
        assert_eq!(&out.values[4..8], &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(&out.indices[4..8], &[2, 3, u32::MAX, u32::MAX]);
    }

    #[test]
    fn validates_shape_and_thresholds() {
        let x = vec![1.0, 2.0, 3.0];
        assert_eq!(
            compact_between(&x, 0, 3, 3, 0.0, 1.0).unwrap_err(),
            CompactionError::InvalidShape
        );
        assert_eq!(
            compact_between(&x, 1, 3, 2, 0.0, 1.0).unwrap_err(),
            CompactionError::InvalidShape
        );
        assert_eq!(
            compact_between(&x, 1, 3, 3, f32::NAN, 1.0).unwrap_err(),
            CompactionError::NonFiniteThreshold
        );
    }

    #[test]
    fn compaction_plan_exposes_dedicated_choice() {
        let plan = plan_compaction(1_024, 65_536, DeviceCaps::wgpu(32, true, 256));
        assert_eq!(plan.rows, 1_024);
        assert_eq!(plan.cols, 65_536);
        assert!(plan.choice.ctile >= 256);
        assert!(plan.to_unison_script().contains("unison compaction"));
    }
}
