// SPDX-License-Identifier: AGPL-3.0-or-later

//! Tensor-facing row compaction over one shared kernel contract.
//!
//! CPU and HIP remain explicit execution choices here. Runtime policy belongs
//! to the Rust orchestration layer rather than this operation facade.

use std::error::Error;
use std::fmt;

pub use st_kernel_contracts::compaction::{
    compact_rows_reference_f32, tiles_per_row, CompactionError, CompactionLayout,
    CompactionOutputF32, CompactionShape, COMPACTION_TILE,
};

#[non_exhaustive]
#[derive(Debug)]
pub enum TensorCompactionError {
    Contract(CompactionError),
    Backend(String),
}

impl fmt::Display for TensorCompactionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(error) => write!(formatter, "compaction contract: {error}"),
            Self::Backend(error) => write!(formatter, "compaction backend: {error}"),
        }
    }
}

impl Error for TensorCompactionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Contract(error) => Some(error),
            Self::Backend(_) => None,
        }
    }
}

impl From<CompactionError> for TensorCompactionError {
    fn from(error: CompactionError) -> Self {
        Self::Contract(error)
    }
}

/// Runs the canonical CPU oracle through the tensor operation surface.
pub fn compact_rows_cpu_f32(
    values: &[f32],
    indices: &[i32],
    shape: CompactionShape,
) -> Result<CompactionOutputF32, TensorCompactionError> {
    compact_rows_reference_f32(values, indices, shape).map_err(TensorCompactionError::from)
}

/// Runs the owning HIP executor against the same compaction contract.
#[cfg(feature = "hip-real")]
pub fn compact_rows_hip_f32(
    values: &[f32],
    indices: &[i32],
    shape: CompactionShape,
) -> Result<CompactionOutputF32, TensorCompactionError> {
    crate::backend::hip_dense::ensure_runtime().map_err(TensorCompactionError::Backend)?;
    st_backend_hip::compaction::compact_rows_f32(values, indices, shape)
        .map_err(|error| TensorCompactionError::Backend(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_entrypoint_preserves_the_shared_contract_type() {
        let output = compact_rows_cpu_f32(
            &[0.2, 0.7, 0.5, f32::NAN],
            &[1, 2, 3, 4],
            CompactionShape {
                rows: 1,
                cols: 4,
                low: 0.5,
                high: 0.8,
            },
        )
        .unwrap();

        assert_eq!(output.row_counts(), &[2]);
        assert_eq!(output.row_values(0), Some(&[0.7, 0.5][..]));
        assert_eq!(output.row_indices(0), Some(&[2, 3][..]));
    }

    #[test]
    fn cpu_entrypoint_keeps_contract_failures_typed() {
        let error = compact_rows_cpu_f32(
            &[],
            &[],
            CompactionShape {
                rows: -1,
                cols: 0,
                low: 0.0,
                high: 1.0,
            },
        )
        .unwrap_err();

        assert!(matches!(
            error,
            TensorCompactionError::Contract(CompactionError::NegativeDimensions { .. })
        ));
    }
}
