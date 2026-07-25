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

/// Shared row shape and inclusive value window for HIP compaction kernels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompactionShape {
    pub rows: i32,
    pub cols: i32,
    pub low: f32,
    pub high: f32,
}

impl CompactionShape {
    pub(crate) fn active(self) -> Result<bool, HipErr> {
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
        if self.rows == 0 || self.cols == 0 {
            return Ok(false);
        }
        self.rows.checked_mul(self.cols).ok_or_else(|| {
            HipErr::Other(format!(
                "compaction element count exceeds the kernel i32 addressing range: {} x {}",
                self.rows, self.cols
            ))
        })?;
        Ok(true)
    }
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
            assert!(shape.active().is_err(), "shape should fail: {shape:?}");
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
}
