// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#[cfg(feature = "faer")]
mod imp {
    use faer::linalg::matmul::matmul as faer_matmul;
    use faer::mat::{from_raw_parts, Mat, MatRef};
    use faer::get_global_parallelism;

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

        unsafe fn row_major_ref<'a>(
            ptr: *const f32,
            rows: usize,
            cols: usize,
            row_stride: isize,
            col_stride: isize,
        ) -> MatRef<'a, f32> {
            from_raw_parts(ptr, rows, cols, row_stride, col_stride)
        }

        let lhs = unsafe { row_major_ref(lhs.as_ptr(), rows, inner, inner as isize, 1) };
        let rhs = unsafe { row_major_ref(rhs.as_ptr(), inner, cols, cols as isize, 1) };
        let mut out = Mat::<f32>::zeros(rows, cols);

        faer_matmul(out.as_mut(), lhs, rhs, None, 1.0, get_global_parallelism());

        let mut buffer = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                buffer[r * cols + c] = out[(r, c)];
            }
        }
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
