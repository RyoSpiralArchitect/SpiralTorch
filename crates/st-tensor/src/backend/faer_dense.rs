// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#[cfg(feature = "faer")]
mod imp {
    use faer::linalg::matmul::matmul as faer_matmul;
    use faer::mat::{MatMut, MatRef};
    use faer::{get_global_parallelism, Accum};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum DenseLayout {
        RowMajor,
        ColMajor,
    }

    fn validate_inputs(
        lhs: &[f32],
        rhs: &[f32],
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> Result<usize, String> {
        let length = |r: usize, c: usize, name: &str| {
            r.checked_mul(c)
                .ok_or_else(|| format!("{name} dimensions overflow: {r}x{c}"))
        };
        let out_len = length(rows, cols, "destination")?;
        let lhs_len = length(rows, inner, "lhs")?;
        let rhs_len = length(inner, cols, "rhs")?;
        for (name, actual, expected) in [("lhs", lhs.len(), lhs_len), ("rhs", rhs.len(), rhs_len)] {
            if actual != expected {
                return Err(format!(
                    "{name} length mismatch: expected {expected} elements, got {actual}"
                ));
            }
        }
        Ok(out_len)
    }

    pub fn is_available() -> bool {
        true
    }

    pub fn should_use(rows: usize, inner: usize, cols: usize) -> bool {
        let volume = rows.saturating_mul(inner).saturating_mul(cols);
        volume >= 8 * 8 * 8 && (rows >= 4 || cols >= 4)
    }

    pub fn matmul(
        lhs: &[f32],
        rhs: &[f32],
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> Result<Vec<f32>, String> {
        let len = validate_inputs(lhs, rhs, rows, inner, cols)?;
        let mut buffer = vec![0.0; len];
        matmul_oriented_into(
            &mut buffer,
            lhs,
            DenseLayout::RowMajor,
            rhs,
            DenseLayout::RowMajor,
            rows,
            inner,
            cols,
        )?;
        Ok(buffer)
    }

    pub fn matmul_into(
        dst: &mut [f32],
        lhs: &[f32],
        rhs: &[f32],
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> Result<(), String> {
        matmul_oriented_into(
            dst,
            lhs,
            DenseLayout::RowMajor,
            rhs,
            DenseLayout::RowMajor,
            rows,
            inner,
            cols,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matmul_oriented_into(
        dst: &mut [f32],
        lhs: &[f32],
        lhs_layout: DenseLayout,
        rhs: &[f32],
        rhs_layout: DenseLayout,
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> Result<(), String> {
        let len = validate_inputs(lhs, rhs, rows, inner, cols)?;
        if dst.len() != len {
            return Err(format!(
                "destination length mismatch: expected {} elements, got {}",
                len,
                dst.len()
            ));
        }

        if rows == 0 || cols == 0 || inner == 0 {
            dst.fill(0.0);
            return Ok(());
        }

        let lhs = match lhs_layout {
            DenseLayout::RowMajor => MatRef::from_row_major_slice(lhs, rows, inner),
            DenseLayout::ColMajor => MatRef::from_column_major_slice(lhs, rows, inner),
        };
        let rhs = match rhs_layout {
            DenseLayout::RowMajor => MatRef::from_row_major_slice(rhs, inner, cols),
            DenseLayout::ColMajor => MatRef::from_column_major_slice(rhs, inner, cols),
        };
        let out = MatMut::from_row_major_slice_mut(dst, rows, cols);
        // Replace does not read the previous destination; no zeroing pass is needed.
        faer_matmul(out, Accum::Replace, lhs, rhs, 1.0, get_global_parallelism());

        Ok(())
    }
}

#[cfg(not(feature = "faer"))]
mod imp {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum DenseLayout {
        RowMajor,
        ColMajor,
    }

    pub fn is_available() -> bool {
        false
    }

    pub fn should_use(_rows: usize, _inner: usize, _cols: usize) -> bool {
        false
    }

    pub fn matmul(
        _lhs: &[f32],
        _rhs: &[f32],
        rows: usize,
        _inner: usize,
        cols: usize,
    ) -> Result<Vec<f32>, String> {
        Err(format!(
            "faer backend disabled at compile time (requested {rows}x{cols} multiply)"
        ))
    }

    pub fn matmul_into(
        _dst: &mut [f32],
        _lhs: &[f32],
        _rhs: &[f32],
        rows: usize,
        _inner: usize,
        cols: usize,
    ) -> Result<(), String> {
        Err(format!(
            "faer backend disabled at compile time (requested {rows}x{cols} multiply)"
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matmul_oriented_into(
        _dst: &mut [f32],
        _lhs: &[f32],
        _lhs_layout: DenseLayout,
        _rhs: &[f32],
        _rhs_layout: DenseLayout,
        rows: usize,
        _inner: usize,
        cols: usize,
    ) -> Result<(), String> {
        Err(format!(
            "faer backend disabled at compile time (requested {rows}x{cols} multiply)"
        ))
    }
}

pub use imp::*;

#[cfg(all(test, feature = "faer"))]
mod tests {
    use super::*;

    fn orient(values: &[f32], rows: usize, cols: usize, layout: DenseLayout) -> Vec<f32> {
        if layout == DenseLayout::RowMajor {
            return values.to_vec();
        }
        let mut output = vec![0.0; values.len()];
        for row in 0..rows {
            for col in 0..cols {
                output[col * rows + row] = values[row * cols + col];
            }
        }
        output
    }

    #[test]
    fn all_layouts_replace_nan_destinations_and_match_independent_products() {
        for (rows, inner, cols) in [
            (0, 0, 0),
            (0, 3, 4),
            (3, 0, 4),
            (3, 4, 0),
            (1, 1, 1),
            (3, 5, 7),
            (17, 33, 65),
        ] {
            let lhs: Vec<_> = (0..rows * inner)
                .map(|i| ((i % 13) as f32 - 6.0) / 8.0)
                .collect();
            let rhs: Vec<_> = (0..inner * cols)
                .map(|i| ((i % 17) as f32 - 8.0) / 8.0)
                .collect();
            let mut expected = vec![0.0f64; rows * cols];
            for row in 0..rows {
                for col in 0..cols {
                    for k in 0..inner {
                        expected[row * cols + col] +=
                            f64::from(lhs[row * inner + k]) * f64::from(rhs[k * cols + col]);
                    }
                }
            }
            for lhs_layout in [DenseLayout::RowMajor, DenseLayout::ColMajor] {
                for rhs_layout in [DenseLayout::RowMajor, DenseLayout::ColMajor] {
                    let mut output = vec![f32::NAN; rows * cols];
                    matmul_oriented_into(
                        &mut output,
                        &orient(&lhs, rows, inner, lhs_layout),
                        lhs_layout,
                        &orient(&rhs, inner, cols, rhs_layout),
                        rhs_layout,
                        rows,
                        inner,
                        cols,
                    )
                    .unwrap();
                    assert!(output
                        .iter()
                        .zip(&expected)
                        .all(|(&a, &b)| f64::from(a) == b));
                }
            }
            let allocating = matmul(&lhs, &rhs, rows, inner, cols).unwrap();
            let mut into = vec![f32::NAN; rows * cols];
            matmul_into(&mut into, &lhs, &rhs, rows, inner, cols).unwrap();
            assert_eq!(allocating, into);
        }
    }

    #[test]
    fn malformed_lengths_are_errors_before_destination_mutation() {
        for (lhs, rhs) in [
            (vec![1.0; 5], vec![1.0; 6]),
            (vec![1.0; 7], vec![1.0; 6]),
            (vec![1.0; 6], vec![1.0; 5]),
            (vec![1.0; 6], vec![1.0; 7]),
        ] {
            assert!(matmul(&lhs, &rhs, 2, 3, 2).is_err());
            let mut dst = vec![42.0; 4];
            assert!(matmul_into(&mut dst, &lhs, &rhs, 2, 3, 2).is_err());
            assert_eq!(dst, vec![42.0; 4]);
            for layout in [DenseLayout::RowMajor, DenseLayout::ColMajor] {
                assert!(
                    matmul_oriented_into(&mut dst, &lhs, layout, &rhs, layout, 2, 3, 2).is_err()
                );
                assert_eq!(dst, vec![42.0; 4]);
            }
        }
        let mut dst = [42.0; 3];
        assert!(matmul_into(&mut dst, &[1.0; 6], &[1.0; 6], 2, 3, 2).is_err());
        assert_eq!(dst, [42.0; 3]);
        // A zero output must not bypass the input shape contract.
        assert!(matmul(&[], &[], 0, 3, 2).is_err());
        assert!(matmul_into(&mut [], &[], &[], 0, 3, 2).is_err());
    }

    #[test]
    fn overflow_is_rejected_before_allocation_or_empty_shortcuts() {
        for (rows, inner, cols) in [(usize::MAX, 0, 2), (usize::MAX, 2, 0), (0, usize::MAX, 2)] {
            let mut dst = [42.0];
            assert!(matmul(&[], &[], rows, inner, cols)
                .unwrap_err()
                .contains("overflow"));
            assert!(matmul_into(&mut dst, &[], &[], rows, inner, cols)
                .unwrap_err()
                .contains("overflow"));
            assert_eq!(dst, [42.0]);
        }
        assert!(should_use(usize::MAX, usize::MAX, usize::MAX));
        assert!(!should_use(usize::MAX, 0, usize::MAX));
    }
}
