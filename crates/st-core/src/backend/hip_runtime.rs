// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(feature = "hip-real")]

use crate::backend::rankk_launch::LaunchSlices;
use crate::backend::rankk_software::Selection;
use crate::ops::rank_entry::RankPlan;
use std::f32;
use std::ffi::CStr;
use std::os::raw::c_char;

const HIP_SUCCESS: i32 = 0;
const WAVEFRONT: usize = 64;
const PER_LANE_KEEP: usize = 8;
const SUPPORTED_K: usize = WAVEFRONT * PER_LANE_KEEP;

extern "C" {
    fn st_hip_topk_rowwise_launch(
        host_input: *const f32,
        rows: i32,
        cols: i32,
        k: i32,
        host_out_vals: *mut f32,
        host_out_idx: *mut i32,
    ) -> i32;
    fn st_hip_midk_rowwise_launch(
        host_input: *const f32,
        rows: i32,
        cols: i32,
        k: i32,
        host_out_vals: *mut f32,
        host_out_idx: *mut i32,
    ) -> i32;
    fn st_hip_error_string(code: i32) -> *const c_char;
}

pub fn run_selection(
    selection: Selection,
    plan: &RankPlan,
    mut buffers: LaunchSlices<'_>,
) -> Result<(), String> {
    if plan.rows == 0 || plan.k == 0 {
        return Ok(());
    }
    if plan.cols == 0 {
        fill_empty_columns(&mut buffers);
        return Ok(());
    }

    match selection {
        Selection::Top => launch_topk(plan, buffers),
        Selection::Mid => launch_midk(plan, buffers),
        Selection::Bottom => launch_bottomk(plan, buffers),
    }
}

fn launch_topk(plan: &RankPlan, buffers: LaunchSlices<'_>) -> Result<(), String> {
    launch_topk_kernel(plan, buffers.input, buffers.out_vals, buffers.out_idx)
}

fn launch_bottomk(plan: &RankPlan, buffers: LaunchSlices<'_>) -> Result<(), String> {
    let mut neg_input = Vec::with_capacity(buffers.input.len());
    neg_input.extend(buffers.input.iter().map(|v| -*v));

    launch_topk_kernel(plan, &neg_input, buffers.out_vals, buffers.out_idx)?;

    for value in buffers.out_vals.iter_mut() {
        *value = -*value;
    }
    Ok(())
}

fn launch_midk(plan: &RankPlan, buffers: LaunchSlices<'_>) -> Result<(), String> {
    let code = unsafe {
        st_hip_midk_rowwise_launch(
            buffers.input.as_ptr(),
            plan.rows as i32,
            plan.cols as i32,
            plan.k as i32,
            buffers.out_vals.as_mut_ptr(),
            buffers.out_idx.as_mut_ptr(),
        )
    };

    if code == HIP_SUCCESS {
        Ok(())
    } else {
        Err(format!("hip runtime error: {}", hip_error_string(code)))
    }
}

fn launch_topk_kernel(
    plan: &RankPlan,
    input: &[f32],
    out_vals: &mut [f32],
    out_idx: &mut [i32],
) -> Result<(), String> {
    if plan.k as usize > SUPPORTED_K {
        return Err(format!(
            "hip topk kernel only supports k ≤ {SUPPORTED_K}, received {}",
            plan.k
        ));
    }

    let code = unsafe {
        st_hip_topk_rowwise_launch(
            input.as_ptr(),
            plan.rows as i32,
            plan.cols as i32,
            plan.k as i32,
            out_vals.as_mut_ptr(),
            out_idx.as_mut_ptr(),
        )
    };

    if code == HIP_SUCCESS {
        Ok(())
    } else {
        Err(format!("hip runtime error: {}", hip_error_string(code)))
    }
}

fn fill_empty_columns(buffers: &mut LaunchSlices<'_>) {
    let rows = buffers.rows as usize;
    let k = buffers.k as usize;
    if k == 0 {
        return;
    }
    for row in 0..rows {
        let base = row * k;
        for slot in 0..k {
            buffers.out_vals[base + slot] = f32::NAN;
            buffers.out_idx[base + slot] = -1;
        }
    }
}

fn hip_error_string(code: i32) -> String {
    unsafe {
        let ptr = st_hip_error_string(code);
        if ptr.is_null() {
            format!("code {code}")
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::device_caps::DeviceCaps;
    use crate::ops::rank_entry::{plan_rank, RankKind};

    const ROWS: u32 = 2;
    const COLS: u32 = 5;

    fn plan(kind: RankKind, k: u32) -> RankPlan {
        plan_rank(
            kind,
            ROWS,
            COLS,
            k,
            DeviceCaps::hip(64, 256, Some(64 * 1024)),
        )
    }

    fn sample_input() -> Vec<f32> {
        vec![
            1.0, 3.5, -2.0, 0.5, 7.0, // row 0
            -1.0, 4.0, 0.25, -3.0, 2.0, // row 1
        ]
    }

    #[test]
    fn hip_runtime_bottomk_gpu_path_matches_reference() {
        let input = sample_input();
        let plan = plan(RankKind::BottomK, 2);
        let mut out_vals = vec![0.0f32; (ROWS * plan.k) as usize];
        let mut out_idx = vec![0i32; (ROWS * plan.k) as usize];

        run_selection(
            Selection::Bottom,
            &plan,
            LaunchSlices {
                input: &input,
                out_vals: &mut out_vals,
                out_idx: &mut out_idx,
                rows: ROWS,
                cols: COLS,
                k: plan.k,
            },
        )
        .expect("bottomk should run through hip runtime path");

        assert_eq!(out_vals, vec![-2.0, 0.5, -3.0, -1.0]);
        assert_eq!(out_idx, vec![2, 3, 3, 0]);
    }

    #[test]
    fn hip_runtime_midk_gpu_path_matches_reference() {
        let input = sample_input();
        let plan = plan(RankKind::MidK, 2);
        let mut out_vals = vec![0.0f32; (ROWS * plan.k) as usize];
        let mut out_idx = vec![0i32; (ROWS * plan.k) as usize];

        run_selection(
            Selection::Mid,
            &plan,
            LaunchSlices {
                input: &input,
                out_vals: &mut out_vals,
                out_idx: &mut out_idx,
                rows: ROWS,
                cols: COLS,
                k: plan.k,
            },
        )
        .expect("midk should run through hip runtime path");

        assert_eq!(out_vals, vec![0.5, 1.0, -1.0, 0.25]);
        assert_eq!(out_idx, vec![3, 0, 0, 2]);
    }
}
