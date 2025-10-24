// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#[cfg(feature = "faer")]
mod imp {
    use faer::get_global_parallelism;
    use faer::linalg::matmul::matmul as faer_matmul;
    use faer::mat::{from_raw_parts, from_raw_parts_mut, MatMut, MatRef};

    unsafe fn row_major_ref<'a>(
        ptr: *const f32,
        rows: usize,
        cols: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> MatRef<'a, f32> {
        from_raw_parts(ptr, rows, cols, row_stride, col_stride)
    }

    unsafe fn row_major_mut<'a>(
        ptr: *mut f32,
        rows: usize,
        cols: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> MatMut<'a, f32> {
        from_raw_parts_mut(ptr, rows, cols, row_stride, col_stride)
    }

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
        let mut buffer = vec![0.0; rows * cols];
        matmul_into(lhs, rhs, &mut buffer, rows, inner, cols)?;
        Ok(buffer)
    }

    pub fn matmul_into(
        lhs: &[f32],
        rhs: &[f32],
        out: &mut [f32],
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> Result<(), String> {
        if rows == 0 || cols == 0 || inner == 0 {
            out.fill(0.0);
            return Ok(());
        }
        if lhs.len() != rows * inner {
            return Err(format!(
                "lhs buffer length mismatch: expected {} elements, got {}",
                rows * inner,
                lhs.len()
            ));
        }
        if rhs.len() != inner * cols {
            return Err(format!(
                "rhs buffer length mismatch: expected {} elements, got {}",
                inner * cols,
                rhs.len()
            ));
        }
        if out.len() != rows * cols {
            return Err(format!(
                "output buffer length mismatch: expected {} elements, got {}",
                rows * cols,
                out.len()
            ));
        }

        out.fill(0.0);
        let lhs = unsafe { row_major_ref(lhs.as_ptr(), rows, inner, inner as isize, 1) };
        let rhs = unsafe { row_major_ref(rhs.as_ptr(), inner, cols, cols as isize, 1) };
        let dst = unsafe { row_major_mut(out.as_mut_ptr(), rows, cols, cols as isize, 1) };
        faer_matmul(dst, lhs, rhs, None, 1.0, get_global_parallelism());
        Ok(())
    }

    pub fn matmul_into(
        dst: &mut [f32],
        lhs: &[f32],
        rhs: &[f32],
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> Result<(), String> {
        if dst.len() != rows * cols {
            return Err(format!(
                "destination length mismatch: expected {} elements, got {}",
                rows * cols,
                dst.len()
            ));
        }

        if rows == 0 || cols == 0 || inner == 0 {
            dst.fill(0.0);
            return Ok(());
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

        unsafe fn row_major_mut<'a>(
            ptr: *mut f32,
            rows: usize,
            cols: usize,
            row_stride: isize,
            col_stride: isize,
        ) -> MatMut<'a, f32> {
            from_raw_parts_mut(ptr, rows, cols, row_stride, col_stride)
        }

        let lhs = unsafe { row_major_ref(lhs.as_ptr(), rows, inner, inner as isize, 1) };
        let rhs = unsafe { row_major_ref(rhs.as_ptr(), inner, cols, cols as isize, 1) };

        dst.fill(0.0);
        let out = unsafe { row_major_mut(dst.as_mut_ptr(), rows, cols, cols as isize, 1) };
        faer_matmul(out, lhs, rhs, None, 1.0, get_global_parallelism());

        Ok(())
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
}

pub use imp::*;
