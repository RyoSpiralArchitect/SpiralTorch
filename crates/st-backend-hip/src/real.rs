// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::{DeviceInfo, HipErr};
use std::ffi::c_void;

pub type HipPtr = *mut c_void;
type hipError_t = i32;
pub type hipStream_t = *mut c_void;

extern "C" {
    // ... existing externs ...
    fn st_kway_merge_shared_heap_real_keepk_u64(
        cand_packed: *const u64,
        rows: i32,
        total: i32,
        k_final: i32,
        out_vals: *mut f32,
        out_idx: *mut i32,
        stream: hipStream_t,
    ) -> i32;
    fn st_kway_merge_warp_coop_keepk_u64(
        cand_packed: *const u64,
        rows: i32,
        total: i32,
        k_final: i32,
        out_vals: *mut f32,
        out_idx: *mut i32,
        stream: hipStream_t,
    ) -> i32;
}

pub fn kway_merge_shared_heap_real_keepk_u64(
    cand_packed: *const u64,
    rows: i32,
    total: i32,
    k_final: i32,
    out_vals: *mut f32,
    out_idx: *mut i32,
    stream: &super::real::HipStream,
) -> Result<(), HipErr> {
    let rc = unsafe {
        st_kway_merge_shared_heap_real_keepk_u64(
            cand_packed,
            rows,
            total,
            k_final,
            out_vals,
            out_idx,
            stream.0,
        )
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(HipErr::Other(format!(
            "st_kway_merge_shared_heap_real_keepk_u64 {}",
            rc
        )))
    }
}
pub fn kway_merge_warp_coop_keepk_u64(
    cand_packed: *const u64,
    rows: i32,
    total: i32,
    k_final: i32,
    out_vals: *mut f32,
    out_idx: *mut i32,
    stream: &super::real::HipStream,
) -> Result<(), HipErr> {
    let rc = unsafe {
        st_kway_merge_warp_coop_keepk_u64(
            cand_packed,
            rows,
            total,
            k_final,
            out_vals,
            out_idx,
            stream.0,
        )
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(HipErr::Other(format!(
            "st_kway_merge_warp_coop_keepk_u64 {}",
            rc
        )))
    }
}
