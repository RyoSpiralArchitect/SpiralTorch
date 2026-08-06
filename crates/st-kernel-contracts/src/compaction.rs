// SPDX-License-Identifier: AGPL-3.0-or-later

use thiserror::Error;

pub const COMPACTION_TILE: i32 = 256;

#[non_exhaustive]
#[derive(Debug, Error, Clone, PartialEq)]
pub enum CompactionError {
    #[error("compaction dimensions must be non-negative, got rows={rows} cols={cols}")]
    NegativeDimensions { rows: i32, cols: i32 },
    #[error("compaction bounds must be finite and ordered, got [{low}, {high}]")]
    InvalidBounds { low: f32, high: f32 },
    #[error("compaction {field} value {value} does not fit usize")]
    DimensionConversion { field: &'static str, value: i32 },
    #[error("compaction element count overflows usize: {rows} x {cols}")]
    ElementCountOverflow { rows: i32, cols: i32 },
    #[error("compaction element count exceeds the kernel i32 addressing range: {rows} x {cols}")]
    KernelAddressOverflow { rows: i32, cols: i32 },
    #[error("compaction tile count overflows usize: rows={rows} tiles_per_row={tiles_per_row}")]
    TileCountOverflow { rows: usize, tiles_per_row: usize },
    #[error(
        "compaction {storage} storage byte count is not addressable: elements={elements} element_size={element_size}"
    )]
    StorageByteCountOverflow {
        storage: &'static str,
        elements: usize,
        element_size: usize,
    },
    #[error(
        "compaction input storage mismatch: values={values} indices={indices} expected={expected}"
    )]
    InputStorageMismatch {
        values: usize,
        indices: usize,
        expected: usize,
    },
    #[error(
        "compaction output storage mismatch: values={values} indices={indices} expected={expected}"
    )]
    OutputStorageMismatch {
        values: usize,
        indices: usize,
        expected: usize,
    },
    #[error("compaction output element count overflows usize: {rows} x {cols}")]
    OutputElementCountOverflow { rows: usize, cols: usize },
    #[error("compaction row count length mismatch: got {actual} expected {expected}")]
    RowCountLengthMismatch { actual: usize, expected: usize },
    #[error("compaction row {row} count {count} does not fit u32")]
    RowCountConversion { row: usize, count: usize },
    #[error("compaction row {row} count {count} exceeds width {cols}")]
    RowCountExceedsWidth {
        row: usize,
        count: usize,
        cols: usize,
    },
    #[error("compaction row {row} has non-zero padding")]
    NonZeroPadding { row: usize },
}

#[inline]
pub fn tiles_per_row(cols: i32) -> i32 {
    if cols <= 0 {
        0
    } else {
        1 + (cols - 1) / COMPACTION_TILE
    }
}

fn checked_storage_bytes<T>(
    elements: usize,
    storage: &'static str,
) -> Result<usize, CompactionError> {
    let element_size = std::mem::size_of::<T>();
    let bytes =
        elements
            .checked_mul(element_size)
            .ok_or(CompactionError::StorageByteCountOverflow {
                storage,
                elements,
                element_size,
            })?;
    if bytes > isize::MAX as usize {
        return Err(CompactionError::StorageByteCountOverflow {
            storage,
            elements,
            element_size,
        });
    }
    Ok(bytes)
}

/// Validated host and device allocation dimensions for row compaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactionLayout {
    rows: usize,
    cols: usize,
    element_count: usize,
    tiles_per_row: usize,
    tile_count: usize,
    value_storage_bytes: usize,
    index_storage_bytes: usize,
    flag_storage_bytes: usize,
    row_count_storage_bytes: usize,
    tile_count_storage_bytes: usize,
}

impl CompactionLayout {
    pub fn rows(self) -> usize {
        self.rows
    }

    pub fn cols(self) -> usize {
        self.cols
    }

    pub fn element_count(self) -> usize {
        self.element_count
    }

    pub fn tiles_per_row(self) -> usize {
        self.tiles_per_row
    }

    pub fn tile_count(self) -> usize {
        self.tile_count
    }

    pub fn value_storage_bytes(self) -> usize {
        self.value_storage_bytes
    }

    pub fn index_storage_bytes(self) -> usize {
        self.index_storage_bytes
    }

    pub fn flag_storage_bytes(self) -> usize {
        self.flag_storage_bytes
    }

    pub fn row_count_storage_bytes(self) -> usize {
        self.row_count_storage_bytes
    }

    pub fn tile_count_storage_bytes(self) -> usize {
        self.tile_count_storage_bytes
    }

    pub fn is_empty(self) -> bool {
        self.element_count == 0
    }

    pub fn validate_input_storage(
        self,
        values: &[f32],
        indices: &[i32],
    ) -> Result<(), CompactionError> {
        if values.len() != self.element_count || indices.len() != self.element_count {
            return Err(CompactionError::InputStorageMismatch {
                values: values.len(),
                indices: indices.len(),
                expected: self.element_count,
            });
        }
        Ok(())
    }
}

/// Shared row shape and inclusive value window for row compaction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompactionShape {
    pub rows: i32,
    pub cols: i32,
    pub low: f32,
    pub high: f32,
}

impl CompactionShape {
    pub fn layout(self) -> Result<CompactionLayout, CompactionError> {
        if self.rows < 0 || self.cols < 0 {
            return Err(CompactionError::NegativeDimensions {
                rows: self.rows,
                cols: self.cols,
            });
        }
        if !self.low.is_finite() || !self.high.is_finite() || self.low > self.high {
            return Err(CompactionError::InvalidBounds {
                low: self.low,
                high: self.high,
            });
        }
        let rows =
            usize::try_from(self.rows).map_err(|_| CompactionError::DimensionConversion {
                field: "rows",
                value: self.rows,
            })?;
        let cols =
            usize::try_from(self.cols).map_err(|_| CompactionError::DimensionConversion {
                field: "columns",
                value: self.cols,
            })?;
        let element_count =
            rows.checked_mul(cols)
                .ok_or(CompactionError::ElementCountOverflow {
                    rows: self.rows,
                    cols: self.cols,
                })?;
        if element_count > i32::MAX as usize {
            return Err(CompactionError::KernelAddressOverflow {
                rows: self.rows,
                cols: self.cols,
            });
        }
        let tiles_per_row = usize::try_from(tiles_per_row(self.cols)).map_err(|_| {
            CompactionError::DimensionConversion {
                field: "tiles_per_row",
                value: tiles_per_row(self.cols),
            }
        })?;
        let tile_count =
            rows.checked_mul(tiles_per_row)
                .ok_or(CompactionError::TileCountOverflow {
                    rows,
                    tiles_per_row,
                })?;
        let value_storage_bytes = checked_storage_bytes::<f32>(element_count, "value")?;
        let index_storage_bytes = checked_storage_bytes::<i32>(element_count, "index")?;
        let flag_storage_bytes = checked_storage_bytes::<u32>(element_count, "flag")?;
        let row_count_storage_bytes = checked_storage_bytes::<u32>(rows, "row count")?;
        let tile_count_storage_bytes = checked_storage_bytes::<u32>(tile_count, "tile count")?;
        Ok(CompactionLayout {
            rows,
            cols,
            element_count,
            tiles_per_row,
            tile_count,
            value_storage_bytes,
            index_storage_bytes,
            flag_storage_bytes,
            row_count_storage_bytes,
            tile_count_storage_bytes,
        })
    }
}

/// Stable row-wise range compaction with fixed-width zero-padded storage.
///
/// The first `row_counts[row]` entries of each row are meaningful. Values and
/// indices beyond that prefix are deterministic zeros.
#[derive(Debug, Clone, PartialEq)]
pub struct CompactionOutputF32 {
    rows: usize,
    cols: usize,
    values: Vec<f32>,
    indices: Vec<i32>,
    row_counts: Vec<u32>,
}

impl CompactionOutputF32 {
    pub fn zeroed(layout: CompactionLayout) -> Self {
        Self {
            rows: layout.rows,
            cols: layout.cols,
            values: vec![0.0; layout.element_count],
            indices: vec![0; layout.element_count],
            row_counts: vec![0; layout.rows],
        }
    }

    pub fn from_parts(
        layout: CompactionLayout,
        values: Vec<f32>,
        indices: Vec<i32>,
        row_counts: Vec<u32>,
    ) -> Result<Self, CompactionError> {
        let output = Self {
            rows: layout.rows,
            cols: layout.cols,
            values,
            indices,
            row_counts,
        };
        output.validate()?;
        Ok(output)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn values_storage(&self) -> &[f32] {
        &self.values
    }

    pub fn indices_storage(&self) -> &[i32] {
        &self.indices
    }

    pub fn row_counts(&self) -> &[u32] {
        &self.row_counts
    }

    pub fn row_values(&self, row: usize) -> Option<&[f32]> {
        let count = usize::try_from(*self.row_counts.get(row)?).ok()?;
        let start = row.checked_mul(self.cols)?;
        self.values.get(start..start.checked_add(count)?)
    }

    pub fn row_indices(&self, row: usize) -> Option<&[i32]> {
        let count = usize::try_from(*self.row_counts.get(row)?).ok()?;
        let start = row.checked_mul(self.cols)?;
        self.indices.get(start..start.checked_add(count)?)
    }

    pub fn validate(&self) -> Result<(), CompactionError> {
        let expected = self.rows.checked_mul(self.cols).ok_or(
            CompactionError::OutputElementCountOverflow {
                rows: self.rows,
                cols: self.cols,
            },
        )?;
        if self.values.len() != expected || self.indices.len() != expected {
            return Err(CompactionError::OutputStorageMismatch {
                values: self.values.len(),
                indices: self.indices.len(),
                expected,
            });
        }
        if self.row_counts.len() != self.rows {
            return Err(CompactionError::RowCountLengthMismatch {
                actual: self.row_counts.len(),
                expected: self.rows,
            });
        }
        for (row, &count) in self.row_counts.iter().enumerate() {
            let count =
                usize::try_from(count).map_err(|_| CompactionError::RowCountExceedsWidth {
                    row,
                    count: usize::MAX,
                    cols: self.cols,
                })?;
            if count > self.cols {
                return Err(CompactionError::RowCountExceedsWidth {
                    row,
                    count,
                    cols: self.cols,
                });
            }
            let start = row * self.cols + count;
            let end = (row + 1) * self.cols;
            if self.values[start..end].iter().any(|&value| value != 0.0)
                || self.indices[start..end].iter().any(|&index| index != 0)
            {
                return Err(CompactionError::NonZeroPadding { row });
            }
        }
        Ok(())
    }
}

/// CPU reference oracle for stable row compaction.
///
/// Each row preserves input order and retains values inside the inclusive
/// `[low, high]` window. Non-finite values are excluded by the same comparison
/// semantics used by accelerator kernels.
pub fn compact_rows_reference_f32(
    values: &[f32],
    indices: &[i32],
    shape: CompactionShape,
) -> Result<CompactionOutputF32, CompactionError> {
    let layout = shape.layout()?;
    layout.validate_input_storage(values, indices)?;
    let mut output = CompactionOutputF32::zeroed(layout);

    for row in 0..layout.rows {
        let row_start = row * layout.cols;
        let mut count = 0usize;
        for column in 0..layout.cols {
            let source = row_start + column;
            let value = values[source];
            if value >= shape.low && value <= shape.high {
                let destination = row_start + count;
                output.values[destination] = value;
                output.indices[destination] = indices[source];
                count += 1;
            }
        }
        output.row_counts[row] =
            u32::try_from(count).map_err(|_| CompactionError::RowCountConversion { row, count })?;
    }
    output.validate()?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_count_is_exact_at_boundaries_and_cannot_overflow() {
        assert_eq!(tiles_per_row(-1), 0);
        assert_eq!(tiles_per_row(0), 0);
        assert_eq!(tiles_per_row(1), 1);
        assert_eq!(tiles_per_row(COMPACTION_TILE), 1);
        assert_eq!(tiles_per_row(COMPACTION_TILE + 1), 2);
        assert_eq!(
            tiles_per_row(i32::MAX),
            1 + (i32::MAX - 1) / COMPACTION_TILE
        );

        let unaddressable = (isize::MAX as usize / std::mem::size_of::<f32>()) + 1;
        assert!(matches!(
            checked_storage_bytes::<f32>(unaddressable, "test"),
            Err(CompactionError::StorageByteCountOverflow { .. })
        ));
    }

    #[test]
    fn shape_contract_rejects_invalid_dimensions_bounds_and_addressing() {
        for shape in [
            CompactionShape {
                rows: -1,
                cols: 1,
                low: 0.0,
                high: 1.0,
            },
            CompactionShape {
                rows: 1,
                cols: -1,
                low: 0.0,
                high: 1.0,
            },
            CompactionShape {
                rows: 1,
                cols: 1,
                low: f32::NAN,
                high: 1.0,
            },
            CompactionShape {
                rows: 1,
                cols: 1,
                low: 1.0,
                high: 0.0,
            },
            CompactionShape {
                rows: i32::MAX,
                cols: 2,
                low: 0.0,
                high: 1.0,
            },
        ] {
            assert!(shape.layout().is_err(), "shape should fail: {shape:?}");
        }
    }

    #[test]
    fn shape_contract_distinguishes_noop_and_active_work() {
        for shape in [
            CompactionShape {
                rows: 0,
                cols: 3,
                low: -1.0,
                high: 1.0,
            },
            CompactionShape {
                rows: 3,
                cols: 0,
                low: -1.0,
                high: 1.0,
            },
        ] {
            assert!(shape.layout().unwrap().is_empty());
        }
        let active = CompactionShape {
            rows: 2,
            cols: 3,
            low: -1.0,
            high: 1.0,
        }
        .layout()
        .unwrap();
        assert!(!active.is_empty());
        assert_eq!(active.value_storage_bytes(), 6 * std::mem::size_of::<f32>());
        assert_eq!(active.index_storage_bytes(), 6 * std::mem::size_of::<i32>());
        assert_eq!(active.flag_storage_bytes(), 6 * std::mem::size_of::<u32>());
        assert_eq!(
            active.row_count_storage_bytes(),
            2 * std::mem::size_of::<u32>()
        );
        assert_eq!(
            active.tile_count_storage_bytes(),
            2 * std::mem::size_of::<u32>()
        );
    }

    #[test]
    fn reference_compaction_is_stable_inclusive_and_zero_padded() {
        let values = [
            0.1,
            0.5,
            f32::NAN,
            0.8,
            0.75,
            f32::NEG_INFINITY,
            f32::INFINITY,
            0.6,
            0.4,
            0.9,
        ];
        let indices = [10, 11, 12, 13, 14, 20, 21, 22, 23, 24];
        let output = compact_rows_reference_f32(
            &values,
            &indices,
            CompactionShape {
                rows: 2,
                cols: 5,
                low: 0.4,
                high: 0.8,
            },
        )
        .unwrap();

        assert_eq!(output.row_counts(), &[3, 2]);
        assert_eq!(output.row_values(0), Some(&[0.5, 0.8, 0.75][..]));
        assert_eq!(output.row_indices(0), Some(&[11, 13, 14][..]));
        assert_eq!(output.row_values(1), Some(&[0.6, 0.4][..]));
        assert_eq!(output.row_indices(1), Some(&[22, 23][..]));
        assert_eq!(
            output.values_storage(),
            &[0.5, 0.8, 0.75, 0.0, 0.0, 0.6, 0.4, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            output.indices_storage(),
            &[11, 13, 14, 0, 0, 22, 23, 0, 0, 0]
        );
        assert_eq!(output.row_values(2), None);
    }

    #[test]
    fn reference_compaction_validates_exact_input_storage() {
        let shape = CompactionShape {
            rows: 2,
            cols: 2,
            low: -1.0,
            high: 1.0,
        };
        assert!(matches!(
            compact_rows_reference_f32(&[0.0; 3], &[0; 4], shape),
            Err(CompactionError::InputStorageMismatch { .. })
        ));
        assert!(matches!(
            compact_rows_reference_f32(&[0.0; 4], &[0; 3], shape),
            Err(CompactionError::InputStorageMismatch { .. })
        ));
    }

    #[test]
    fn output_validation_rejects_invalid_counts_and_padding() {
        let layout = CompactionShape {
            rows: 1,
            cols: 2,
            low: 0.0,
            high: 1.0,
        }
        .layout()
        .unwrap();

        assert!(matches!(
            CompactionOutputF32::from_parts(layout, vec![0.0; 2], vec![0; 2], vec![3]),
            Err(CompactionError::RowCountExceedsWidth { .. })
        ));
        assert!(matches!(
            CompactionOutputF32::from_parts(layout, vec![0.0, 0.5], vec![0; 2], vec![1]),
            Err(CompactionError::NonZeroPadding { .. })
        ));

        let overflowing = CompactionOutputF32 {
            rows: usize::MAX,
            cols: 2,
            values: Vec::new(),
            indices: Vec::new(),
            row_counts: Vec::new(),
        };
        assert!(matches!(
            overflowing.validate(),
            Err(CompactionError::OutputElementCountOverflow { .. })
        ));
    }

    #[test]
    fn reference_compaction_preserves_zero_sized_row_shapes() {
        let no_rows = compact_rows_reference_f32(
            &[],
            &[],
            CompactionShape {
                rows: 0,
                cols: 3,
                low: 0.0,
                high: 1.0,
            },
        )
        .unwrap();
        assert_eq!(no_rows.rows(), 0);
        assert_eq!(no_rows.cols(), 3);
        assert!(no_rows.row_counts().is_empty());

        let zero_width = compact_rows_reference_f32(
            &[],
            &[],
            CompactionShape {
                rows: 3,
                cols: 0,
                low: 0.0,
                high: 1.0,
            },
        )
        .unwrap();
        assert_eq!(zero_width.rows(), 3);
        assert_eq!(zero_width.cols(), 0);
        assert_eq!(zero_width.row_counts(), &[0, 0, 0]);
    }
}
