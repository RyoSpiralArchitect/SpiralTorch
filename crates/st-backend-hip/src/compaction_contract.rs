// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::HipErr;

pub use st_kernel_contracts::compaction::{
    tiles_per_row, CompactionError, CompactionLayout, CompactionOutputF32, CompactionShape,
    COMPACTION_TILE,
};

/// Compatibility facade for callers that expect HIP's error surface.
pub fn compact_rows_reference_f32(
    values: &[f32],
    indices: &[i32],
    shape: CompactionShape,
) -> Result<CompactionOutputF32, HipErr> {
    st_kernel_contracts::compaction::compact_rows_reference_f32(values, indices, shape)
        .map_err(HipErr::from)
}

#[cfg(feature = "hip-real")]
pub(crate) fn validate_compaction_inputs(
    values: &[f32],
    indices: &[i32],
    layout: CompactionLayout,
) -> Result<(), HipErr> {
    layout
        .validate_input_storage(values, indices)
        .map_err(HipErr::from)
}

#[cfg(feature = "hip-real")]
pub(crate) fn compaction_active(shape: CompactionShape) -> Result<bool, HipErr> {
    Ok(!shape.layout().map_err(HipErr::from)?.is_empty())
}
