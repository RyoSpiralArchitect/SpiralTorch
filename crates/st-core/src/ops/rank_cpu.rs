//! CPU reference Rank-K selection.
//!
//! This module implements a small CPU fallback for per-row TopK/MidK/BottomK
//! selection. The output layout mirrors the WGPU TopK kernels:
//! `values[r*k + t]` / `indices[r*k + t]` for row `r` and rank slot `t`.
//!
//! `RankKind::MidK` is treated as an exact selection over the value-sorted row:
//! sort ascending, take the centered window of width `k`, and emit that window
//! in ascending order. Threshold-based MidK compaction remains available via
//! `ops::midk`.

use crate::ops::rank_entry::{RankKind, RankPlan};
use std::cmp::Ordering;

#[derive(Clone, Debug, PartialEq)]
pub struct RankKSelection {
    pub values: Vec<f32>,
    pub indices: Vec<u32>,
    pub rows: u32,
    pub k: u32,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RankCpuError {
    #[error(
        "invalid shape (rows={rows}, cols={cols}, k={k}, row_stride={row_stride}, x_len={x_len})"
    )]
    InvalidShape {
        rows: u32,
        cols: u32,
        k: u32,
        row_stride: u32,
        x_len: usize,
    },
    #[error("k ({k}) exceeds cols ({cols})")]
    KExceedsCols { k: u32, cols: u32 },
    #[error("output buffer length mismatch (expected={expected}, out_vals={out_vals}, out_idx={out_idx})")]
    OutputLen {
        expected: usize,
        out_vals: usize,
        out_idx: usize,
    },
}

fn checked_len(rows: u32, cols: u32) -> Option<usize> {
    (rows as usize).checked_mul(cols as usize)
}

fn validate_dense(
    x: &[f32],
    rows: u32,
    cols: u32,
    row_stride: u32,
    k: u32,
    out_vals: &[f32],
    out_idx: &[u32],
) -> Result<(), RankCpuError> {
    if rows == 0 || cols == 0 {
        return Err(RankCpuError::InvalidShape {
            rows,
            cols,
            k,
            row_stride,
            x_len: x.len(),
        });
    }
    if row_stride < cols {
        return Err(RankCpuError::InvalidShape {
            rows,
            cols,
            k,
            row_stride,
            x_len: x.len(),
        });
    }
    if k > cols {
        return Err(RankCpuError::KExceedsCols { k, cols });
    }

    let needed_x = checked_len(rows, row_stride).ok_or(RankCpuError::InvalidShape {
        rows,
        cols,
        k,
        row_stride,
        x_len: x.len(),
    })?;
    if x.len() < needed_x {
        return Err(RankCpuError::InvalidShape {
            rows,
            cols,
            k,
            row_stride,
            x_len: x.len(),
        });
    }

    let expected_out = checked_len(rows, k).unwrap_or(usize::MAX);
    if out_vals.len() != expected_out || out_idx.len() != expected_out {
        return Err(RankCpuError::OutputLen {
            expected: expected_out,
            out_vals: out_vals.len(),
            out_idx: out_idx.len(),
        });
    }
    Ok(())
}

fn cmp_topk(a: &(f32, u32), b: &(f32, u32)) -> Ordering {
    let (va, ia) = *a;
    let (vb, ib) = *b;
    match (va.is_nan(), vb.is_nan()) {
        (true, true) => ia.cmp(&ib),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => vb.total_cmp(&va).then_with(|| ia.cmp(&ib)),
    }
}

fn cmp_bottomk(a: &(f32, u32), b: &(f32, u32)) -> Ordering {
    let (va, ia) = *a;
    let (vb, ib) = *b;
    match (va.is_nan(), vb.is_nan()) {
        (true, true) => ia.cmp(&ib),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => va.total_cmp(&vb).then_with(|| ia.cmp(&ib)),
    }
}

fn midk_window_start(cols: usize, k: usize) -> usize {
    cols.saturating_sub(k) / 2
}

pub fn topk_select_into(
    x: &[f32],
    rows: u32,
    cols: u32,
    row_stride: u32,
    k: u32,
    out_vals: &mut [f32],
    out_idx: &mut [u32],
) -> Result<(), RankCpuError> {
    validate_dense(x, rows, cols, row_stride, k, out_vals, out_idx)?;
    if k == 0 {
        return Ok(());
    }

    let cols_usize = cols as usize;
    let k_usize = k as usize;
    let row_stride_usize = row_stride as usize;

    let mut pairs: Vec<(f32, u32)> = Vec::with_capacity(cols_usize);

    for r in 0..rows as usize {
        let in_base = r * row_stride_usize;
        let out_base = r * k_usize;
        pairs.clear();
        for c in 0..cols_usize {
            pairs.push((x[in_base + c], c as u32));
        }
        pairs.sort_by(cmp_topk);
        for t in 0..k_usize {
            let (v, i) = pairs[t];
            out_vals[out_base + t] = v;
            out_idx[out_base + t] = i;
        }
    }

    Ok(())
}

pub fn bottomk_select_into(
    x: &[f32],
    rows: u32,
    cols: u32,
    row_stride: u32,
    k: u32,
    out_vals: &mut [f32],
    out_idx: &mut [u32],
) -> Result<(), RankCpuError> {
    validate_dense(x, rows, cols, row_stride, k, out_vals, out_idx)?;
    if k == 0 {
        return Ok(());
    }

    let cols_usize = cols as usize;
    let k_usize = k as usize;
    let row_stride_usize = row_stride as usize;

    let mut pairs: Vec<(f32, u32)> = Vec::with_capacity(cols_usize);

    for r in 0..rows as usize {
        let in_base = r * row_stride_usize;
        let out_base = r * k_usize;
        pairs.clear();
        for c in 0..cols_usize {
            pairs.push((x[in_base + c], c as u32));
        }
        pairs.sort_by(cmp_bottomk);
        for t in 0..k_usize {
            let (v, i) = pairs[t];
            out_vals[out_base + t] = v;
            out_idx[out_base + t] = i;
        }
    }

    Ok(())
}

pub fn midk_select_into(
    x: &[f32],
    rows: u32,
    cols: u32,
    row_stride: u32,
    k: u32,
    out_vals: &mut [f32],
    out_idx: &mut [u32],
) -> Result<(), RankCpuError> {
    validate_dense(x, rows, cols, row_stride, k, out_vals, out_idx)?;
    if k == 0 {
        return Ok(());
    }

    let cols_usize = cols as usize;
    let k_usize = k as usize;
    let row_stride_usize = row_stride as usize;
    let start = midk_window_start(cols_usize, k_usize);

    let mut pairs: Vec<(f32, u32)> = Vec::with_capacity(cols_usize);

    for r in 0..rows as usize {
        let in_base = r * row_stride_usize;
        let out_base = r * k_usize;
        pairs.clear();
        for c in 0..cols_usize {
            pairs.push((x[in_base + c], c as u32));
        }
        pairs.sort_by(cmp_bottomk);
        for t in 0..k_usize {
            let (v, i) = pairs[start + t];
            out_vals[out_base + t] = v;
            out_idx[out_base + t] = i;
        }
    }

    Ok(())
}

pub fn select_rank_cpu(
    plan: &RankPlan,
    x: &[f32],
    row_stride: u32,
) -> Result<RankKSelection, RankCpuError> {
    let out_len = checked_len(plan.rows, plan.k).ok_or(RankCpuError::InvalidShape {
        rows: plan.rows,
        cols: plan.cols,
        k: plan.k,
        row_stride,
        x_len: x.len(),
    })?;
    let mut values = vec![0.0f32; out_len];
    let mut indices = vec![0u32; out_len];
    select_rank_cpu_into(plan, x, row_stride, &mut values, &mut indices)?;
    Ok(RankKSelection {
        values,
        indices,
        rows: plan.rows,
        k: plan.k,
    })
}

pub fn select_rank_cpu_into(
    plan: &RankPlan,
    x: &[f32],
    row_stride: u32,
    out_vals: &mut [f32],
    out_idx: &mut [u32],
) -> Result<(), RankCpuError> {
    match plan.kind {
        RankKind::TopK => topk_select_into(
            x, plan.rows, plan.cols, row_stride, plan.k, out_vals, out_idx,
        ),
        RankKind::BottomK => bottomk_select_into(
            x, plan.rows, plan.cols, row_stride, plan.k, out_vals, out_idx,
        ),
        RankKind::MidK => midk_select_into(
            x, plan.rows, plan.cols, row_stride, plan.k, out_vals, out_idx,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::device_caps::DeviceCaps;
    use crate::ops::rank_entry::plan_rank;

    #[test]
    fn cpu_topk_is_deterministic_with_ties() {
        let rows = 2u32;
        let cols = 5u32;
        let k = 2u32;
        let x = vec![
            1.0, 3.0, 2.0, 3.0, -1.0, // row0
            0.0, -2.0, 5.0, 4.0, 5.0, // row1
        ];
        let plan = plan_rank(RankKind::TopK, rows, cols, k, DeviceCaps::cpu());
        let sel = select_rank_cpu(&plan, &x, cols).unwrap();
        assert_eq!(sel.values, vec![3.0, 3.0, 5.0, 5.0]);
        assert_eq!(sel.indices, vec![1, 3, 2, 4]);
    }

    #[test]
    fn cpu_bottomk_selects_smallest() {
        let rows = 2u32;
        let cols = 5u32;
        let k = 2u32;
        let x = vec![
            1.0, 3.0, 2.0, 3.0, -1.0, // row0
            0.0, -2.0, 5.0, 4.0, 5.0, // row1
        ];
        let plan = plan_rank(RankKind::BottomK, rows, cols, k, DeviceCaps::cpu());
        let sel = select_rank_cpu(&plan, &x, cols).unwrap();
        assert_eq!(sel.values, vec![-1.0, 1.0, -2.0, 0.0]);
        assert_eq!(sel.indices, vec![4, 0, 1, 0]);
    }

    #[test]
    fn nan_is_ignored_for_selection() {
        let rows = 1u32;
        let cols = 3u32;
        let k = 1u32;
        let x = vec![f32::NAN, 2.0, 3.0];
        let top = plan_rank(RankKind::TopK, rows, cols, k, DeviceCaps::cpu());
        let sel_top = select_rank_cpu(&top, &x, cols).unwrap();
        assert_eq!(sel_top.values, vec![3.0]);
        assert_eq!(sel_top.indices, vec![2]);

        let bottom = plan_rank(RankKind::BottomK, rows, cols, k, DeviceCaps::cpu());
        let sel_bottom = select_rank_cpu(&bottom, &x, cols).unwrap();
        assert_eq!(sel_bottom.values, vec![2.0]);
        assert_eq!(sel_bottom.indices, vec![1]);
    }

    #[test]
    fn cpu_midk_selects_center_window() {
        let plan = plan_rank(RankKind::MidK, 1, 5, 2, DeviceCaps::cpu());
        let x = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        let sel = select_rank_cpu(&plan, &x, 5).unwrap();
        assert_eq!(sel.values, vec![2.0, 3.0]);
        assert_eq!(sel.indices, vec![3, 2]);
    }

    #[test]
    fn cpu_midk_is_stable_with_duplicates() {
        let plan = plan_rank(RankKind::MidK, 1, 6, 3, DeviceCaps::cpu());
        let x = vec![9.0, 2.0, 2.0, 5.0, 7.0, 2.0];
        let sel = select_rank_cpu(&plan, &x, 6).unwrap();
        assert_eq!(sel.values, vec![2.0, 2.0, 5.0]);
        assert_eq!(sel.indices, vec![2, 5, 3]);
    }
}
