// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(feature = "hip-real")]

use crate::backend::rankk_launch::LaunchSlices;
use crate::backend::rankk_software::Selection;
use crate::ops::rank_entry::RankPlan;
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
    fn st_hip_error_string(code: i32) -> *const c_char;
}

pub fn run_selection(
    selection: Selection,
    plan: &RankPlan,
    buffers: LaunchSlices<'_>,
) -> Result<(), String> {
    match selection {
        Selection::Top => launch_topk(plan, buffers),
        Selection::Mid | Selection::Bottom => {
            Err("hip selection not implemented for mid/bottom".to_string())
        }
    }
}

fn launch_topk(plan: &RankPlan, mut buffers: LaunchSlices<'_>) -> Result<(), String> {
    if plan.k as usize > SUPPORTED_K {
        return Err(format!(
            "hip topk kernel only supports k ≤ {SUPPORTED_K}, received {}",
            plan.k
        ));
    }

    let code = unsafe {
        st_hip_topk_rowwise_launch(
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
