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
        workspace.extend(0..cols);

        match selection {
            Selection::Top => {
                workspace.sort_unstable_by(|&a, &b| match row_slice[b].partial_cmp(&row_slice[a]) {
                    Some(order) => order,
                    None => Ordering::Equal,
                })
            }
            Selection::Bottom | Selection::Mid => {
                workspace.sort_unstable_by(|&a, &b| match row_slice[a].partial_cmp(&row_slice[b]) {
                    Some(order) => order,
                    None => Ordering::Equal,
                })
            }
        }

        let take = usize::min(k, cols);
        let chosen: &[usize] = match selection {
            Selection::Top => &workspace[..take],
            Selection::Bottom => &workspace[..take],
            Selection::Mid => {
                let start = (cols.saturating_sub(take)) / 2;
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
