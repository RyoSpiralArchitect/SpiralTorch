// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use rayon::prelude::*;

const TM: usize = 8;
const TN: usize = 12;
const L2_TARGET_BYTES: usize = 64 * 1024;

#[cfg(feature = "simd")]
use core::simd::Simd;

#[cfg(feature = "simd")]
type Simd8 = Simd<f32, TM>;

#[inline(always)]
fn row_tile_size(rows: usize, inner: usize) -> usize {
    if rows <= TM {
        return TM;
    }

    let row_bytes = inner.saturating_mul(core::mem::size_of::<f32>());
    if row_bytes == 0 {
        return TM;
    }

    let mut tile = (L2_TARGET_BYTES / row_bytes).max(1);
    tile = tile.min(rows);

    let tile = tile / TM * TM;
    if tile == 0 {
        TM
    } else {
        tile
    }
}

#[inline]
fn pack_b_block_into(
    dst: &mut [f32],
    rhs: &[f32],
    inner: usize,
    cols: usize,
    col_start: usize,
    width: usize,
) {
    debug_assert_eq!(dst.len(), width * inner);

    unsafe {
        for col in 0..width {
            let dst_col = dst.as_mut_ptr().add(col * inner);
            let mut rhs_ptr = rhs.as_ptr().add(col_start + col);
            for offset in 0..inner {
                *dst_col.add(offset) = *rhs_ptr;
                rhs_ptr = rhs_ptr.add(cols);
            }
        }
    }
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
fn pack_a_block(src: &[f32], inner: usize, dst: &mut [f32]) {
    debug_assert_eq!(src.len(), TM * inner);
    debug_assert_eq!(dst.len(), TM * inner);

    unsafe {
        for k in 0..inner {
            let dst_col = dst.as_mut_ptr().add(k * TM);
            let mut src_ptr = src.as_ptr().add(k);
            for row in 0..TM {
                *dst_col.add(row) = *src_ptr;
                src_ptr = src_ptr.add(inner);
            }
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
            let row_tile = row_tile_size(prefix_rows, inner);

            dst_prefix
                .par_chunks_mut(cols * row_tile)
                .zip(lhs_prefix.par_chunks(row_tile * inner))
                .for_each(|(dst_chunk, lhs_chunk)| {
                    let local_rows = lhs_chunk.len() / inner;
                    let mut packed_a = vec![0.0f32; TM * inner];

                    for offset in (0..local_rows).step_by(TM) {
                        let lhs_panel = &lhs_chunk[offset * inner..(offset + TM) * inner];
                        pack_a_block(lhs_panel, inner, &mut packed_a);
                        unsafe {
                            microkernel_8x12(
                                packed_a.as_ptr(),
                                packed_block.as_ptr(),
                                dst_chunk.as_mut_ptr().add(offset * cols + col_start),
                                TM,
                                inner,
                                cols,
                                inner,
                            );
                        }
                    }
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
    let mut acc = [Simd8::splat(0.0); TN];

    for p in 0..k {
        let a_slice = core::slice::from_raw_parts(a.add(p * lda), TM);
        let a_vec = Simd8::from_slice(a_slice);

        let mut b_ptr = b.add(p);
        for col in 0..TN {
            let b_vec = Simd8::splat(*b_ptr);
            acc[col] += a_vec * b_vec;
            b_ptr = b_ptr.add(ldb);
        }
    }

    for col in 0..TN {
        let mut dst_ptr = c.add(col);
        let col_vec = acc[col];
        for row in 0..TM {
            *dst_ptr += col_vec[row];
            dst_ptr = dst_ptr.add(ldc);
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
    let mut acc = [[0.0f32; TN]; TM];

    for p in 0..k {
        let mut b_ptr = b.add(p);
        for col in 0..TN {
            let b_val = *b_ptr;
            b_ptr = b_ptr.add(ldb);
            for row in 0..TM {
                let a_val = *a.add(p * lda + row);
                acc[row][col] += a_val * b_val;
            }
        }
    }

    for row in 0..TM {
        let row_acc = acc[row];
        let dst_row = c.add(row * ldc);
        for col in 0..TN {
            *dst_row.add(col) += row_acc[col];
        }
    }
}

pub fn is_available() -> bool {
    true
}

pub fn should_use(rows: usize, inner: usize, cols: usize) -> bool {
    let volume = rows * inner * cols;
    volume >= TM * TN * 4 && rows >= TM && inner >= TM && cols >= TN
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

    let mut packed_panel = vec![0.0f32; TN * inner];

    for block in 0..full_blocks {
        let col_start = block * TN;
        pack_b_block_into(packed_panel.as_mut_slice(), rhs, inner, cols, col_start, TN);
        compute_with_packed_block(
            dst,
            lhs,
            rows,
            inner,
            cols,
            col_start,
            TN,
            packed_panel.as_slice(),
        );
    }

    if tail > 0 {
        let col_start = full_blocks * TN;
        pack_b_block_into(
            &mut packed_panel[..tail * inner],
            rhs,
            inner,
            cols,
            col_start,
            tail,
        );
        compute_with_packed_block(
            dst,
            lhs,
            rows,
            inner,
            cols,
            col_start,
            tail,
            &packed_panel[..tail * inner],
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

pub fn prepack_rhs(rhs: &[f32], inner: usize, cols: usize) -> Result<Vec<f32>, String> {
    if rhs.len() != inner * cols {
        return Err(format!(
            "rhs length mismatch: expected {} elements, got {}",
            inner * cols,
            rhs.len()
        ));
    }

    let mut packed = vec![0.0f32; inner * cols];
    let full_blocks = cols / TN;
    let tail = cols % TN;

    for block in 0..full_blocks {
        let col_start = block * TN;
        let dst = &mut packed[col_start * inner..(col_start + TN) * inner];
        pack_b_block_into(dst, rhs, inner, cols, col_start, TN);
    }

    if tail > 0 {
        let col_start = full_blocks * TN;
        let dst = &mut packed[col_start * inner..(col_start + tail) * inner];
        pack_b_block_into(dst, rhs, inner, cols, col_start, tail);
    }

    Ok(packed)
}
