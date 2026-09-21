use st_tensor::{Layout, PureResult, Tensor, TensorError};
use std::borrow::Cow;

pub(crate) fn row_major(input: &Tensor) -> PureResult<Cow<'_, Tensor>> {
    if input.layout() == Layout::RowMajor {
        Ok(Cow::Borrowed(input))
    } else {
        Ok(Cow::Owned(input.to_layout(Layout::RowMajor)?))
    }
}

pub(crate) fn checked_elements(rows: usize, cols: usize) -> PureResult<usize> {
    checked_allocation::<f32>(rows, cols)
}

pub(crate) fn checked_allocation<T>(rows: usize, cols: usize) -> PureResult<usize> {
    rows.checked_mul(cols)
        .filter(|&len| len <= isize::MAX as usize / std::mem::size_of::<T>().max(1))
        .ok_or(TensorError::InvalidDimensions { rows, cols })
}

pub(crate) fn validate_finite(values: &[f32], label: &'static str) -> PureResult<()> {
    if let Some(&value) = values.iter().find(|value| !value.is_finite()) {
        return Err(TensorError::NonFiniteValue { label, value });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checks_element_and_byte_overflow_without_allocation() {
        assert_eq!(checked_elements(0, usize::MAX).unwrap(), 0);
        assert!(checked_elements(usize::MAX, 2).is_err());
        assert!(checked_elements(1, isize::MAX as usize / 4 + 1).is_err());
        assert!(checked_allocation::<f64>(1, isize::MAX as usize / 8 + 1).is_err());
        assert_eq!(checked_elements(2, 6).unwrap(), 12);
    }
}
