// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::real::{hip_result, HipStream};
use crate::HipErr;

extern "C" {
    fn st_compaction_1ce(
        vin: *const f32,
        iin: *const i32,
        rows: i32,
        cols: i32,
        low: f32,
        high: f32,
        vout: *mut f32,
        iout: *mut i32,
        stream: crate::real::hipStream_t,
    ) -> i32;
    fn st_compaction_scan(
        vin: *const f32,
        rows: i32,
        cols: i32,
        low: f32,
        high: f32,
        flags: *mut u32,
        tilecnt: *mut u32,
        tiles_per_row: i32,
        stream: crate::real::hipStream_t,
    ) -> i32;
    fn st_compaction_apply(
        vin: *const f32,
        iin: *const i32,
        rows: i32,
        cols: i32,
        low: f32,
        high: f32,
        flags: *const u32,
        tilecnt: *const u32,
        tiles_per_row: i32,
        vout: *mut f32,
        iout: *mut i32,
        stream: crate::real::hipStream_t,
    ) -> i32;
    fn st_compaction_scan_pass(
        vin: *const f32,
        pos: *mut u32,
        rows: i32,
        cols: i32,
        low: f32,
        high: f32,
        tile: i32,
        stream: crate::real::hipStream_t,
    ) -> i32;
    fn st_compaction_apply_pass(
        vin: *const f32,
        iin: *const i32,
        pos: *const u32,
        rows: i32,
        cols: i32,
        low: f32,
        high: f32,
        vout: *mut f32,
        iout: *mut i32,
        stream: crate::real::hipStream_t,
    ) -> i32;
}

pub const COMPACTION_TILE: i32 = 256;

#[inline]
pub fn tiles_per_row(cols: i32) -> i32 {
    if cols <= 0 {
        0
    } else {
        (cols + (COMPACTION_TILE - 1)) / COMPACTION_TILE
    }
}

pub fn compaction_1ce(
    stream: &HipStream,
    vin: *const f32,
    iin: *const i32,
    rows: i32,
    cols: i32,
    low: f32,
    high: f32,
    vout: *mut f32,
    iout: *mut i32,
) -> Result<(), HipErr> {
    hip_result(
        unsafe { st_compaction_1ce(vin, iin, rows, cols, low, high, vout, iout, stream.raw()) },
        "st_compaction_1ce",
    )
}

pub fn compaction_scan(
    stream: &HipStream,
    vin: *const f32,
    rows: i32,
    cols: i32,
    low: f32,
    high: f32,
    flags: *mut u32,
    tile_counts: *mut u32,
    tiles_per_row: i32,
) -> Result<(), HipErr> {
    if rows <= 0 || cols <= 0 || tiles_per_row <= 0 {
        return Ok(());
    }
    hip_result(
        unsafe {
            st_compaction_scan(
                vin,
                rows,
                cols,
                low,
                high,
                flags,
                tile_counts,
                tiles_per_row,
                stream.raw(),
            )
        },
        "st_compaction_scan",
    )
}

pub fn compaction_apply(
    stream: &HipStream,
    vin: *const f32,
    iin: *const i32,
    rows: i32,
    cols: i32,
    low: f32,
    high: f32,
    flags: *const u32,
    tile_counts: *const u32,
    tiles_per_row: i32,
    vout: *mut f32,
    iout: *mut i32,
) -> Result<(), HipErr> {
    if rows <= 0 || cols <= 0 || tiles_per_row <= 0 {
        return Ok(());
    }
    hip_result(
        unsafe {
            st_compaction_apply(
                vin,
                iin,
                rows,
                cols,
                low,
                high,
                flags,
                tile_counts,
                tiles_per_row,
                vout,
                iout,
                stream.raw(),
            )
        },
        "st_compaction_apply",
    )
}

pub fn compaction_scan_pass(
    stream: &HipStream,
    vin: *const f32,
    rows: i32,
    cols: i32,
    low: f32,
    high: f32,
    positions: *mut u32,
    tile: i32,
) -> Result<(), HipErr> {
    if rows <= 0 || cols <= 0 {
        return Ok(());
    }
    let tile = if tile > 0 { tile } else { COMPACTION_TILE };
    hip_result(
        unsafe {
            st_compaction_scan_pass(vin, positions, rows, cols, low, high, tile, stream.raw())
        },
        "st_compaction_scan_pass",
    )
}

pub fn compaction_apply_pass(
    stream: &HipStream,
    vin: *const f32,
    iin: *const i32,
    positions: *const u32,
    rows: i32,
    cols: i32,
    low: f32,
    high: f32,
    vout: *mut f32,
    iout: *mut i32,
) -> Result<(), HipErr> {
    if rows <= 0 || cols <= 0 {
        return Ok(());
    }
    hip_result(
        unsafe {
            st_compaction_apply_pass(
                vin,
                iin,
                positions,
                rows,
                cols,
                low,
                high,
                vout,
                iout,
                stream.raw(),
            )
        },
        "st_compaction_apply_pass",
    )
}
