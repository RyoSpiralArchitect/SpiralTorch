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
