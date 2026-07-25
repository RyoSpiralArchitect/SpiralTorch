use crate::HipErr;

pub const COMPACTION_TILE: i32 = 256;

#[inline]
pub fn tiles_per_row(cols: i32) -> i32 {
    if cols <= 0 {
        0
    } else {
        1 + (cols - 1) / COMPACTION_TILE
    }
}

/// Validated host and device allocation dimensions for row compaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactionLayout {
    rows: usize,
    cols: usize,
    element_count: usize,
    tiles_per_row: usize,
    tile_count: usize,
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

    pub fn is_empty(self) -> bool {
        self.element_count == 0
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
    pub fn layout(self) -> Result<CompactionLayout, HipErr> {
        if self.rows < 0 || self.cols < 0 {
            return Err(HipErr::Other(format!(
                "compaction dimensions must be non-negative, got rows={} cols={}",
                self.rows, self.cols
            )));
        }
        if !self.low.is_finite() || !self.high.is_finite() || self.low > self.high {
            return Err(HipErr::Other(format!(
                "compaction bounds must be finite and ordered, got [{}, {}]",
                self.low, self.high
            )));
        }
        let rows = usize::try_from(self.rows)
            .map_err(|_| HipErr::Other("compaction rows do not fit usize".into()))?;
        let cols = usize::try_from(self.cols)
            .map_err(|_| HipErr::Other("compaction columns do not fit usize".into()))?;
        let element_count = rows.checked_mul(cols).ok_or_else(|| {
            HipErr::Other(format!(
                "compaction element count overflows usize: {} x {}",
                self.rows, self.cols
            ))
        })?;
        if element_count > i32::MAX as usize {
            return Err(HipErr::Other(format!(
                "compaction element count exceeds the kernel i32 addressing range: {} x {}",
                self.rows, self.cols
            )));
        }
        let tiles_per_row = usize::try_from(tiles_per_row(self.cols))
            .map_err(|_| HipErr::Other("compaction tile count does not fit usize".into()))?;
        let tile_count = rows.checked_mul(tiles_per_row).ok_or_else(|| {
            HipErr::Other(format!(
                "compaction tile count overflows usize: {} x {}",
                self.rows, self.cols
            ))
        })?;
        Ok(CompactionLayout {
            rows,
            cols,
            element_count,
            tiles_per_row,
            tile_count,
        })
    }

    #[cfg(any(feature = "hip-real", test))]
    pub(crate) fn active(self) -> Result<bool, HipErr> {
        Ok(!self.layout()?.is_empty())
    }
}

/// Stable row-wise range compaction with fixed-width zero-padded storage.
///
/// The first `row_counts[row]` entries of each row are meaningful. Values and
/// indices beyond that prefix are deterministic zeros so results are portable
/// across CPU, HIP, and future execution backends.
#[derive(Debug, Clone, PartialEq)]
pub struct CompactionOutputF32 {
    rows: usize,
    cols: usize,
    values: Vec<f32>,
    indices: Vec<i32>,
    row_counts: Vec<u32>,
}

impl CompactionOutputF32 {
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

    pub fn validate(&self) -> Result<(), HipErr> {
        let expected_elements = self.rows.checked_mul(self.cols).ok_or_else(|| {
            HipErr::Other("compaction output element count overflows usize".into())
        })?;
        if self.values.len() != expected_elements || self.indices.len() != expected_elements {
            return Err(HipErr::Other(format!(
                "compaction output storage mismatch: values={} indices={} expected={expected_elements}",
                self.values.len(),
                self.indices.len()
            )));
        }
        if self.row_counts.len() != self.rows {
            return Err(HipErr::Other(format!(
                "compaction row count length mismatch: got {} expected {}",
                self.row_counts.len(),
                self.rows
            )));
        }
        for (row, &count) in self.row_counts.iter().enumerate() {
            let count = usize::try_from(count)
                .map_err(|_| HipErr::Other("compaction row count does not fit usize".into()))?;
            if count > self.cols {
                return Err(HipErr::Other(format!(
                    "compaction row {row} count {count} exceeds width {}",
                    self.cols
                )));
            }
            let start = row * self.cols + count;
            let end = (row + 1) * self.cols;
            if self.values[start..end].iter().any(|&value| value != 0.0)
                || self.indices[start..end].iter().any(|&index| index != 0)
            {
                return Err(HipErr::Other(format!(
                    "compaction row {row} has non-zero padding"
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn zeroed(layout: CompactionLayout) -> Self {
        Self {
            rows: layout.rows,
            cols: layout.cols,
            values: vec![0.0; layout.element_count],
            indices: vec![0; layout.element_count],
            row_counts: vec![0; layout.rows],
        }
    }

    #[cfg(feature = "hip-real")]
    pub(crate) fn values_mut_ptr(&mut self) -> *mut f32 {
        self.values.as_mut_ptr()
    }

    #[cfg(feature = "hip-real")]
    pub(crate) fn indices_mut_ptr(&mut self) -> *mut i32 {
        self.indices.as_mut_ptr()
    }

    #[cfg(feature = "hip-real")]
    pub(crate) fn set_row_counts(&mut self, row_counts: Vec<u32>) -> Result<(), HipErr> {
        self.row_counts = row_counts;
        self.validate()
    }
}

pub(crate) fn validate_compaction_inputs(
    values: &[f32],
    indices: &[i32],
    layout: CompactionLayout,
) -> Result<(), HipErr> {
    if values.len() != layout.element_count || indices.len() != layout.element_count {
        return Err(HipErr::Other(format!(
            "compaction input storage mismatch: values={} indices={} expected={}",
            values.len(),
            indices.len(),
            layout.element_count
        )));
    }
    Ok(())
}

/// CPU reference for the backend-independent row compaction contract.
///
/// Each row preserves input order and retains values inside the inclusive
/// `[low, high]` window. Non-finite values are excluded by the same comparison
/// semantics used by the device kernels.
pub fn compact_rows_reference_f32(
    values: &[f32],
    indices: &[i32],
    shape: CompactionShape,
) -> Result<CompactionOutputF32, HipErr> {
    let layout = shape.layout()?;
    validate_compaction_inputs(values, indices, layout)?;
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
        output.row_counts[row] = u32::try_from(count)
            .map_err(|_| HipErr::Other("compaction row count exceeds u32".into()))?;
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
            assert!(!shape.active().unwrap());
        }
        assert!(CompactionShape {
            rows: 2,
            cols: 3,
            low: -1.0,
            high: 1.0,
        }
        .active()
        .unwrap());
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
        output.validate().unwrap();
    }

    #[test]
    fn reference_compaction_validates_exact_input_storage() {
        let shape = CompactionShape {
            rows: 2,
            cols: 2,
            low: -1.0,
            high: 1.0,
        };
        assert!(compact_rows_reference_f32(&[0.0; 3], &[0; 4], shape)
            .unwrap_err()
            .to_string()
            .contains("input storage mismatch"));
        assert!(compact_rows_reference_f32(&[0.0; 4], &[0; 3], shape)
            .unwrap_err()
            .to_string()
            .contains("input storage mismatch"));
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

        let mut excessive_count = CompactionOutputF32::zeroed(layout);
        excessive_count.row_counts[0] = 3;
        assert!(excessive_count
            .validate()
            .unwrap_err()
            .to_string()
            .contains("exceeds width"));

        let mut dirty_padding = CompactionOutputF32::zeroed(layout);
        dirty_padding.row_counts[0] = 1;
        dirty_padding.values[1] = 0.5;
        assert!(dirty_padding
            .validate()
            .unwrap_err()
            .to_string()
            .contains("non-zero padding"));
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
        zero_width.validate().unwrap();
    }
}
