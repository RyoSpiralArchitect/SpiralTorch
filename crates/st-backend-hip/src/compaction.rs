// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::real::{hip_result, HipStream};
use crate::HipErr;

pub use crate::compaction_contract::{tiles_per_row, CompactionShape, COMPACTION_TILE};

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

/// Raw device buffers for one-block-per-row compaction.
pub struct CompactionOneShotArgs {
    pub values_in: *const f32,
    pub indices_in: *const i32,
    pub values_out: *mut f32,
    pub indices_out: *mut i32,
    pub shape: CompactionShape,
}

/// Enqueues one-block-per-row compaction.
///
/// # Safety
///
/// For an active shape, all pointers must reference device allocations of at
/// least `rows * cols` elements of their respective types and remain valid
/// until `stream` completes. Output buffers must be writable and non-overlapping.
pub unsafe fn compaction_1ce(
    stream: &HipStream,
    args: CompactionOneShotArgs,
) -> Result<(), HipErr> {
    if !args.shape.active()? {
        return Ok(());
    }
    if args.shape.cols > COMPACTION_TILE {
        return Err(HipErr::Other(format!(
            "one-shot compaction supports at most {COMPACTION_TILE} columns, got {}",
            args.shape.cols
        )));
    }
    hip_result(
        st_compaction_1ce(
            args.values_in,
            args.indices_in,
            args.shape.rows,
            args.shape.cols,
            args.shape.low,
            args.shape.high,
            args.values_out,
            args.indices_out,
            stream.raw(),
        ),
        "st_compaction_1ce",
    )
}

/// Raw device buffers for the tiled compaction scan.
pub struct CompactionScanArgs {
    pub values_in: *const f32,
    pub flags: *mut u32,
    pub tile_counts: *mut u32,
    pub shape: CompactionShape,
}

/// Enqueues the tiled compaction prefix scan.
///
/// # Safety
///
/// `values_in` and `flags` must each cover `rows * cols` device elements;
/// `tile_counts` must cover `rows * tiles_per_row(cols)` elements. All buffers
/// must remain valid until `stream` completes and writable outputs must not
/// overlap the input.
pub unsafe fn compaction_scan(stream: &HipStream, args: CompactionScanArgs) -> Result<(), HipErr> {
    if !args.shape.active()? {
        return Ok(());
    }
    let tile_count = tiles_per_row(args.shape.cols);
    hip_result(
        st_compaction_scan(
            args.values_in,
            args.shape.rows,
            args.shape.cols,
            args.shape.low,
            args.shape.high,
            args.flags,
            args.tile_counts,
            tile_count,
            stream.raw(),
        ),
        "st_compaction_scan",
    )
}

/// Raw device buffers for applying a tiled compaction scan.
pub struct CompactionApplyArgs {
    pub values_in: *const f32,
    pub indices_in: *const i32,
    pub flags: *const u32,
    pub tile_counts: *const u32,
    pub values_out: *mut f32,
    pub indices_out: *mut i32,
    pub shape: CompactionShape,
}

/// Enqueues the tiled compaction scatter.
///
/// # Safety
///
/// Value, index, flag, and output buffers must cover `rows * cols` device
/// elements; `tile_counts` must cover `rows * tiles_per_row(cols)` elements.
/// Buffers must remain valid until `stream` completes and outputs must be
/// writable and non-overlapping.
pub unsafe fn compaction_apply(
    stream: &HipStream,
    args: CompactionApplyArgs,
) -> Result<(), HipErr> {
    if !args.shape.active()? {
        return Ok(());
    }
    let tile_count = tiles_per_row(args.shape.cols);
    hip_result(
        st_compaction_apply(
            args.values_in,
            args.indices_in,
            args.shape.rows,
            args.shape.cols,
            args.shape.low,
            args.shape.high,
            args.flags,
            args.tile_counts,
            tile_count,
            args.values_out,
            args.indices_out,
            stream.raw(),
        ),
        "st_compaction_apply",
    )
}

/// Raw device buffers for the sequential multi-tile scan pass.
pub struct CompactionScanPassArgs {
    pub values_in: *const f32,
    pub positions: *mut u32,
    pub shape: CompactionShape,
    pub tile: i32,
}

/// Enqueues the sequential multi-tile scan pass.
///
/// # Safety
///
/// `values_in` and `positions` must cover `rows * cols` device elements,
/// remain valid until `stream` completes, and not overlap.
pub unsafe fn compaction_scan_pass(
    stream: &HipStream,
    args: CompactionScanPassArgs,
) -> Result<(), HipErr> {
    if !args.shape.active()? {
        return Ok(());
    }
    let tile = if args.tile > 0 {
        args.tile
    } else {
        COMPACTION_TILE
    };
    if tile > COMPACTION_TILE {
        return Err(HipErr::Other(format!(
            "compaction scan tile must be at most {COMPACTION_TILE}, got {tile}"
        )));
    }
    hip_result(
        st_compaction_scan_pass(
            args.values_in,
            args.positions,
            args.shape.rows,
            args.shape.cols,
            args.shape.low,
            args.shape.high,
            tile,
            stream.raw(),
        ),
        "st_compaction_scan_pass",
    )
}

/// Raw device buffers for the sequential multi-tile scatter pass.
pub struct CompactionApplyPassArgs {
    pub values_in: *const f32,
    pub indices_in: *const i32,
    pub positions: *const u32,
    pub values_out: *mut f32,
    pub indices_out: *mut i32,
    pub shape: CompactionShape,
}

/// Enqueues the sequential multi-tile scatter pass.
///
/// # Safety
///
/// Every pointer must cover `rows * cols` device elements of its respective
/// type and remain valid until `stream` completes. Outputs must be writable and
/// non-overlapping.
pub unsafe fn compaction_apply_pass(
    stream: &HipStream,
    args: CompactionApplyPassArgs,
) -> Result<(), HipErr> {
    if !args.shape.active()? {
        return Ok(());
    }
    hip_result(
        st_compaction_apply_pass(
            args.values_in,
            args.indices_in,
            args.positions,
            args.shape.rows,
            args.shape.cols,
            args.shape.low,
            args.shape.high,
            args.values_out,
            args.indices_out,
            stream.raw(),
        ),
        "st_compaction_apply_pass",
    )
}
