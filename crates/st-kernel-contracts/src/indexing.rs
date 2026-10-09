//! Exact integer indices and stable, unscaled embedding pullbacks.
use crate::layout::{NdLayout, NdLayoutError};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum IndexingError {
    #[error("row index is outside the table")]
    Bounds,
    #[error("index count does not match its logical shape")]
    Length,
    #[error("embedding requires a rank-two table with the prepared row count")]
    TableShape,
    #[error("embedding cotangent must match the complete output shape")]
    CotangentShape,
    #[error("row index storage size overflows or cannot be allocated")]
    Allocation,
    #[error(transparent)]
    Layout(#[from] NdLayoutError),
}

fn zeros(len: usize) -> Result<Vec<usize>, IndexingError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| IndexingError::Allocation)?;
    values.resize(len, 0);
    Ok(values)
}

/// Stable CSR grouping in O(rows + indices), preserving duplicate token order.
/// Unlike a float transport, exact integer IDs cannot be rounded or clamped.
pub fn grouped_rows(
    indices: &[usize],
    rows: usize,
) -> Result<(Vec<usize>, Vec<usize>), IndexingError> {
    if indices.iter().any(|&index| index >= rows) {
        return Err(IndexingError::Bounds);
    }
    let count = rows.checked_add(1).ok_or(IndexingError::Allocation)?;
    let mut offsets = zeros(count)?;
    for &index in indices {
        offsets[index + 1] += 1;
    }
    for row in 1..count {
        offsets[row] += offsets[row - 1];
    }
    let mut cursor = zeros(rows)?;
    cursor.copy_from_slice(&offsets[..rows]);
    let mut positions = zeros(indices.len())?;
    for (position, &index) in indices.iter().enumerate() {
        positions[cursor[index]] = position;
        cursor[index] += 1;
    }
    Ok((offsets, positions))
}

/// Logical sample axes survive lookup: table [V,C], IDs [B,T] -> [B,T,C].
/// No padding ID, frequency scaling, averaging, or derivative of IDs is implicit.
pub fn embedding_layout(
    table: &[usize],
    index_shape: &[usize],
    rows: usize,
) -> Result<NdLayout, IndexingError> {
    if table.len() != 2 || table[0] != rows {
        return Err(IndexingError::TableShape);
    }
    let mut output = index_shape.to_vec();
    output.push(table[1]);
    Ok(NdLayout::contiguous(&output)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouping_is_stable_exact_and_keeps_unused_rows() {
        let ids = [3, 1, 3, 0, 1, 3];
        let (offsets, positions) = grouped_rows(&ids, 5).unwrap();
        assert_eq!(offsets, [0, 1, 3, 3, 6, 6]);
        assert_eq!(positions, [3, 1, 4, 0, 2, 5]);
        assert_eq!(grouped_rows(&[], 0).unwrap(), (vec![0], vec![]));
        assert!(matches!(grouped_rows(&[5], 5), Err(IndexingError::Bounds)));
        assert!(grouped_rows(&[usize::MAX], 1).is_err());
        assert!(grouped_rows(&[], usize::MAX).is_err());
    }

    #[test]
    fn lookup_preserves_sample_axes_including_scalar_and_empty_ids() {
        assert_eq!(
            embedding_layout(&[256, 8], &[2, 3], 256).unwrap().shape(),
            [2, 3, 8]
        );
        assert_eq!(embedding_layout(&[256, 8], &[], 256).unwrap().shape(), [8]);
        assert_eq!(embedding_layout(&[0, 8], &[0], 0).unwrap().shape(), [0, 8]);
        assert!(embedding_layout(&[256], &[2, 3], 256).is_err());
        assert!(embedding_layout(&[255, 8], &[2, 3], 256).is_err());
    }
}
