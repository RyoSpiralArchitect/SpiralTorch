// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#[cfg(feature = "faer")]
mod imp {
    use faer::get_global_parallelism;
    use faer::linalg::matmul::matmul as faer_matmul;
    use faer::mat;

    pub fn is_available() -> bool {
        true
    }

    pub fn should_use(rows: usize, inner: usize, cols: usize) -> bool {
        let volume = rows * inner * cols;
        volume >= 8 * 8 * 8 && (rows >= 4 || cols >= 4)
    }

    pub fn matmul(
        lhs: &[f32],
        rhs: &[f32],
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> Result<Vec<f32>, String> {
        if rows == 0 || cols == 0 || inner == 0 {
            return Ok(vec![0.0; rows * cols]);
        }

        let lhs_view = mat::from_row_major_slice::<f32>(lhs, rows, inner);
        let rhs_view = mat::from_row_major_slice::<f32>(rhs, inner, cols);
        let mut buffer = vec![0.0; rows * cols];
        let mut out_view = mat::from_row_major_slice_mut::<f32>(&mut buffer, rows, cols);

        faer_matmul(
            out_view.as_mut(),
            lhs_view,
            rhs_view,
            None,
            0.0,
            get_global_parallelism(),
        );

        Ok(buffer)
    }
}

#[cfg(not(feature = "faer"))]
mod imp {
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
}

pub use imp::*;
