// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#[cfg(feature = "faer")]
mod imp {
    use faer::linalg::matmul::matmul;
    use faer::mat::Mat;
    use faer::Parallelism;

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

        let lhs = Mat::from_fn(rows, inner, |r, c| lhs[r * inner + c]);
        let rhs = Mat::from_fn(inner, cols, |r, c| rhs[r * cols + c]);
        let mut out = Mat::<f32>::zeros(rows, cols);

        matmul(
            Parallelism::Rayon,
            out.as_mut(),
            lhs.as_ref(),
            rhs.as_ref(),
            1.0,
            0.0,
        );

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
