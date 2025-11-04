// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#[cfg(feature = "faer")]
mod imp {
    use faer::{get_global_parallelism, Accum};
    use faer::linalg::matmul::matmul as faer_matmul;
    use faer::mat::{MatMut, MatRef};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum DenseLayout {
        RowMajor,
        ColMajor,
    }

    unsafe fn row_major_ref<'a>(
        ptr: *const f32,
        rows: usize,
        cols: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> MatRef<'a, f32> {
        MatRef::from_raw_parts(ptr, rows, cols, row_stride, col_stride)
    }

    unsafe fn row_major_mut<'a>(
        ptr: *mut f32,
        rows: usize,
        cols: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> MatMut<'a, f32> {
        MatMut::from_raw_parts_mut(ptr, rows, cols, row_stride, col_stride)
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

        let (lhs_row_stride, lhs_col_stride) = match lhs_layout {
            DenseLayout::RowMajor => (inner as isize, 1),
            DenseLayout::ColMajor => (1, rows as isize),
        };
        let (rhs_row_stride, rhs_col_stride) = match rhs_layout {
            DenseLayout::RowMajor => (cols as isize, 1),
            DenseLayout::ColMajor => (1, inner as isize),
        };

        let lhs =
            unsafe { row_major_ref(lhs.as_ptr(), rows, inner, lhs_row_stride, lhs_col_stride) };
        let rhs =
            unsafe { row_major_ref(rhs.as_ptr(), inner, cols, rhs_row_stride, rhs_col_stride) };

        dst.fill(0.0);
        let mut out = unsafe { row_major_mut(dst.as_mut_ptr(), rows, cols, cols as isize, 1) };
        
        // faer 0.23 API: matmul(dst, beta, lhs, rhs, alpha, parallelism)
        // beta: Accum::Replace (overwrite) or Accum::Add (accumulate)
        // Since dst is zero-filled, we use Replace to overwrite with α·A·B
        faer_matmul(
            out.as_mut(),
            Accum::Replace,      // Overwrite dst with result
            lhs.as_ref(),
            rhs.as_ref(),
            1.0,                  // α = 1
            get_global_parallelism(),
        );

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
