// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use rayon::prelude::*;

const TM: usize = 8;
const TN: usize = 12;

#[cfg(feature = "simd")]
use wide::f32x8;

#[inline]
fn pack_b_block(
    rhs: &[f32],
    inner: usize,
    cols: usize,
    col_start: usize,
    width: usize,
) -> Vec<f32> {
    let mut packed = vec![0.0; width * inner];
    for k in 0..inner {
        let src = &rhs[k * cols + col_start..k * cols + col_start + width];
        for (j, value) in src.iter().enumerate() {
            packed[j * inner + k] = *value;
        }
    }
    packed
}

#[inline]
fn scalar_block_with_packed(
    dst: &mut [f32],
    lhs: &[f32],
    inner: usize,
    cols: usize,
    row_start: usize,
    height: usize,
    col_start: usize,
    width: usize,
    packed: &[f32],
) {
    for local_row in 0..height {
        let global_row = row_start + local_row;
        let lhs_row = &lhs[global_row * inner..(global_row + 1) * inner];
        for local_col in 0..width {
            let mut acc = 0.0f32;
            let packed_column = &packed[local_col * inner..(local_col + 1) * inner];
            for k in 0..inner {
                acc += lhs_row[k] * packed_column[k];
            }
            let dst_index = global_row * cols + col_start + local_col;
            dst[dst_index] += acc;
        }
    }
}

#[inline]
fn compute_with_packed_block(
    dst: &mut [f32],
    lhs: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
    col_start: usize,
    width: usize,
    packed_block: &[f32],
) {
    debug_assert_eq!(packed_block.len(), width * inner);

    if width == TN {
        let full_row_blocks = rows / TM;
        if full_row_blocks > 0 {
            let prefix_rows = full_row_blocks * TM;
            let lhs_prefix = &lhs[..prefix_rows * inner];
            let dst_prefix = &mut dst[..prefix_rows * cols];
            dst_prefix
                .par_chunks_mut(cols * TM)
                .zip(lhs_prefix.par_chunks(TM * inner))
                .for_each(|(dst_chunk, lhs_chunk)| unsafe {
                    microkernel_8x12(
                        lhs_chunk.as_ptr(),
                        packed_block.as_ptr(),
                        dst_chunk.as_mut_ptr().add(col_start),
                        inner,
                        inner,
                        cols,
                        inner,
                    );
                });
        }

        let processed_rows = (rows / TM) * TM;
        if processed_rows < rows {
            scalar_block_with_packed(
                dst,
                lhs,
                inner,
                cols,
                processed_rows,
                rows - processed_rows,
                col_start,
                width,
                packed_block,
            );
        }
    } else if width > 0 {
        scalar_block_with_packed(
            dst,
            lhs,
            inner,
            cols,
            0,
            rows,
            col_start,
            width,
            packed_block,
        );
    }
}

#[inline(always)]
unsafe fn microkernel_8x12(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    lda: usize,
    ldb: usize,
    ldc: usize,
    k: usize,
) {
    #[cfg(feature = "simd")]
    {
        microkernel_8x12_simd(a, b, c, lda, ldb, ldc, k);
    }

    #[cfg(not(feature = "simd"))]
    {
        microkernel_8x12_scalar(a, b, c, lda, ldb, ldc, k);
    }
}

#[cfg(feature = "simd")]
#[inline(always)]
unsafe fn microkernel_8x12_simd(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    lda: usize,
    ldb: usize,
    ldc: usize,
    k: usize,
) {
    let mut acc = [f32x8::splat(0.0); TN];

    for p in 0..k {
        let mut a_values = [0.0f32; TM];
        for r in 0..TM {
            a_values[r] = *a.add(r * lda + p);
        }
        let a_vec = f32x8::from(a_values);

        for col in 0..TN {
            let b_val = *b.add(col * ldb + p);
            let b_vec = f32x8::splat(b_val);
            acc[col] = acc[col] + a_vec * b_vec;
        }
    }

    for col in 0..TN {
        let values = acc[col].to_array();
        for row in 0..TM {
            let dst_ptr = c.add(row * ldc + col);
            *dst_ptr += values[row];
        }
    }
}

#[cfg(not(feature = "simd"))]
#[inline(always)]
unsafe fn microkernel_8x12_scalar(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    lda: usize,
    ldb: usize,
    ldc: usize,
    k: usize,
) {
    let mut acc = [[0.0f32; TM]; TN];

    for p in 0..k {
        let mut a_values = [0.0f32; TM];
        for r in 0..TM {
            a_values[r] = *a.add(r * lda + p);
        }

        for col in 0..TN {
            let b_val = *b.add(col * ldb + p);
            for row in 0..TM {
                acc[col][row] += a_values[row] * b_val;
            }
        }
    }

    for col in 0..TN {
        for row in 0..TM {
            let dst_ptr = c.add(row * ldc + col);
            *dst_ptr += acc[col][row];
        }
    }
}

pub fn is_available() -> bool {
    true
}

pub fn should_use(rows: usize, inner: usize, cols: usize) -> bool {
    let volume = rows * inner * cols;
    volume >= TM * TN * 4 && inner >= 4
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

    if lhs.len() != rows * inner {
        return Err(format!(
            "lhs length mismatch: expected {} elements, got {}",
            rows * inner,
            lhs.len()
        ));
    }

    if rhs.len() != inner * cols {
        return Err(format!(
            "rhs length mismatch: expected {} elements, got {}",
            inner * cols,
            rhs.len()
        ));
    }

    if rows == 0 || cols == 0 || inner == 0 {
        dst.fill(0.0);
        return Ok(());
    }

    dst.fill(0.0);

    let full_blocks = cols / TN;
    let tail = cols % TN;

    for block in 0..full_blocks {
        let col_start = block * TN;
        let packed = pack_b_block(rhs, inner, cols, col_start, TN);
        compute_with_packed_block(
            dst,
            lhs,
            rows,
            inner,
            cols,
            col_start,
            TN,
            packed.as_slice(),
        );
    }

    if tail > 0 {
        let col_start = full_blocks * TN;
        let packed = pack_b_block(rhs, inner, cols, col_start, tail);
        compute_with_packed_block(
            dst,
            lhs,
            rows,
            inner,
            cols,
            col_start,
            tail,
            packed.as_slice(),
        );
    }

    Ok(())
}

pub fn matmul_packed_into(
    dst: &mut [f32],
    lhs: &[f32],
    packed_rhs: &[f32],
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

    if lhs.len() != rows * inner {
        return Err(format!(
            "lhs length mismatch: expected {} elements, got {}",
            rows * inner,
            lhs.len()
        ));
    }

    if packed_rhs.len() != inner * cols {
        return Err(format!(
            "packed rhs length mismatch: expected {} elements, got {}",
            inner * cols,
            packed_rhs.len()
        ));
    }

    if rows == 0 || cols == 0 || inner == 0 {
        dst.fill(0.0);
        return Ok(());
    }

    dst.fill(0.0);

    let full_blocks = cols / TN;
    let tail = cols % TN;

    for block in 0..full_blocks {
        let col_start = block * TN;
        let packed_block = &packed_rhs[col_start * inner..(col_start + TN) * inner];
        compute_with_packed_block(dst, lhs, rows, inner, cols, col_start, TN, packed_block);
    }

    if tail > 0 {
        let col_start = full_blocks * TN;
        let packed_block = &packed_rhs[col_start * inner..(col_start + tail) * inner];
        compute_with_packed_block(dst, lhs, rows, inner, cols, col_start, tail, packed_block);
    }

    Ok(())
}
