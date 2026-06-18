// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! CPU reference implementations that mirror the behaviour of the CUDA/HIP
//! kernels. They let us exercise the rank-k executors in unit tests until the
//! GPU launch paths are fully wired. Each routine expects a `RankPlan` and the
//! slices registered through `rankk_launch`.

use crate::backend::rankk_launch::LaunchSlices;
use crate::ops::rank_entry::RankPlan;
use std::cmp::Ordering;

#[derive(Clone, Copy)]
pub enum Selection {
    Top,
    Mid,
    Bottom,
}

fn compare_rank_candidates(
    row: &[f32],
    lhs_idx: usize,
    rhs_idx: usize,
    descending: bool,
) -> Ordering {
    let lhs = row[lhs_idx];
    let rhs = row[rhs_idx];
    let order = if descending {
        rhs.total_cmp(&lhs)
    } else {
        lhs.total_cmp(&rhs)
    };
    order.then_with(|| lhs_idx.cmp(&rhs_idx))
}

pub fn run_selection(
    selection: Selection,
    plan: &RankPlan,
    buffers: LaunchSlices<'_>,
) -> Result<(), String> {
    if plan.rows != buffers.rows {
        return Err(format!(
            "plan rows {} did not match buffer rows {}",
            plan.rows, buffers.rows
        ));
    }
    if plan.cols != buffers.cols {
        return Err(format!(
            "plan cols {} did not match buffer cols {}",
            plan.cols, buffers.cols
        ));
    }
    if plan.k != buffers.k {
        return Err(format!(
            "plan k {} did not match buffer k {}",
            plan.k, buffers.k
        ));
    }

    let rows = plan.rows as usize;
    let cols = plan.cols as usize;
    let k = plan.k as usize;

    if k == 0 {
        return Ok(());
    }

    let mut workspace: Vec<usize> = (0..cols).collect();

    for row in 0..rows {
        let row_slice = &buffers.input[row * cols..(row + 1) * cols];
        let out_vals_slice = &mut buffers.out_vals[row * k..(row + 1) * k];
        let out_idx_slice = &mut buffers.out_idx[row * k..(row + 1) * k];

        workspace.clear();
        workspace.extend((0..cols).filter(|&idx| row_slice[idx].is_finite()));

        match selection {
            Selection::Top => {
                workspace.sort_unstable_by(|&a, &b| compare_rank_candidates(row_slice, a, b, true))
            }
            Selection::Bottom | Selection::Mid => {
                workspace.sort_unstable_by(|&a, &b| compare_rank_candidates(row_slice, a, b, false))
            }
        }

        let finite_cols = workspace.len();
        let take = usize::min(k, finite_cols);
        let chosen: &[usize] = match selection {
            Selection::Top => &workspace[..take],
            Selection::Bottom => &workspace[..take],
            Selection::Mid => {
                let start = (finite_cols.saturating_sub(take)) / 2;
                &workspace[start..start + take]
            }
        };

        for (slot, &col_idx) in chosen.iter().enumerate() {
            out_vals_slice[slot] = row_slice[col_idx];
            out_idx_slice[slot] = col_idx as i32;
        }

        for slot in chosen.len()..k {
            out_vals_slice[slot] = f32::NAN;
            out_idx_slice[slot] = -1;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::device_caps::DeviceCaps;
    use crate::backend::rankk_launch::LaunchSlices;
    use crate::backend::unison_heuristics::RankKind;
    use crate::ops::rank_entry::plan_rank;

    fn plan(kind: RankKind, rows: u32, cols: u32, k: u32) -> RankPlan {
        plan_rank(kind, rows, cols, k, DeviceCaps::cpu())
    }

    fn run(
        selection: Selection,
        kind: RankKind,
        input: &[f32],
        rows: u32,
        cols: u32,
        k: u32,
    ) -> (Vec<f32>, Vec<i32>) {
        let mut out_vals = vec![0.0; (rows * k) as usize];
        let mut out_idx = vec![0; (rows * k) as usize];
        let plan = plan(kind, rows, cols, k);
        run_selection(
            selection,
            &plan,
            LaunchSlices {
                input,
                out_vals: &mut out_vals,
                out_idx: &mut out_idx,
                rows,
                cols,
                k,
            },
        )
        .expect("rank-k software selection");
        (out_vals, out_idx)
    }

    #[test]
    fn top_selection_ignores_non_finite_candidates() {
        let input = [f32::NAN, 4.0, f32::INFINITY, 3.0, -2.0, 5.0];
        let (vals, idxs) = run(Selection::Top, RankKind::TopK, &input, 1, 6, 3);

        assert_eq!(vals, vec![5.0, 4.0, 3.0]);
        assert_eq!(idxs, vec![5, 1, 3]);
    }

    #[test]
    fn mid_selection_uses_finite_central_band_and_pads_when_short() {
        let input = [f32::NEG_INFINITY, -3.0, f32::NAN, 0.0, 2.0, 8.0];
        let (vals, idxs) = run(Selection::Mid, RankKind::MidK, &input, 1, 6, 5);

        assert_eq!(idxs[..4], [1, 3, 4, 5]);
        assert_eq!(vals[..4], [-3.0, 0.0, 2.0, 8.0]);
        assert!(vals[4].is_nan());
        assert_eq!(idxs[4], -1);
    }

    #[test]
    fn bottom_selection_ignores_non_finite_candidates() {
        let input = [1.5, f32::NAN, -4.0, f32::INFINITY, -1.0, 2.0];
        let (vals, idxs) = run(Selection::Bottom, RankKind::BottomK, &input, 1, 6, 3);

        assert_eq!(vals, vec![-4.0, -1.0, 1.5]);
        assert_eq!(idxs, vec![2, 4, 0]);
    }
}
