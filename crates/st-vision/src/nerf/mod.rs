// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Lightweight NeRF field definitions built on top of the core SpiralTorch
//! neural network layers.

mod encoding;
mod field;
mod trainer;

pub use encoding::PositionalEncoding;
pub use field::{FieldSampleLayout, NerfField, NerfFieldConfig};
pub use trainer::{NerfTrainer, NerfTrainingConfig, NerfTrainingStats};

use st_tensor::{Layout, PureResult, Tensor, TensorError};
use std::borrow::Cow;

fn row_major(input: &Tensor) -> PureResult<Cow<'_, Tensor>> {
    if input.layout() == Layout::RowMajor {
        Ok(Cow::Borrowed(input))
    } else {
        Ok(Cow::Owned(input.to_layout(Layout::RowMajor)?))
    }
}

fn checked_elements(rows: usize, cols: usize) -> PureResult<usize> {
    rows.checked_mul(cols)
        .filter(|&len| len <= isize::MAX as usize / std::mem::size_of::<f32>())
        .ok_or(TensorError::InvalidDimensions { rows, cols })
}

fn validate_finite(values: &[f32], label: &'static str) -> PureResult<()> {
    if let Some(&value) = values.iter().find(|value| !value.is_finite()) {
        return Err(TensorError::NonFiniteValue { label, value });
    }
    Ok(())
}

#[cfg(test)]
mod contract_tests {
    use super::*;

    #[test]
    fn checks_element_and_byte_overflow_without_allocation() {
        assert_eq!(checked_elements(0, usize::MAX).unwrap(), 0);
        assert!(checked_elements(usize::MAX, 2).is_err());
        assert!(checked_elements(1, isize::MAX as usize / 4 + 1).is_err());
        assert_eq!(checked_elements(2, 6).unwrap(), 12);
    }
}
