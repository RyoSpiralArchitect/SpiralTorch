// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json;
use st_kdsl::autotune_store::{load_best_typed, lookup_similar, record_best, AutoTuneMatch};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::lock_recover;
use spiral_config::determinism;

const L2_TARGET_BYTES: usize = 64 * 1024;

#[cfg(feature = "simd")]
use core::simd::Simd;

type KernelFn = unsafe fn(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    lda: usize,
    ldb: usize,
    ldc: usize,
    k: usize,
);

#[derive(Clone, Copy)]
struct MicroKernelSpec {
    name: &'static str,
    tm: usize,
    tn: usize,
    kernel: KernelFn,
    row_major_kernel: KernelFn,
}

const MICROKERNELS: &[MicroKernelSpec] = &[
    MicroKernelSpec {
        name: "m8n12",
        tm: 8,
        tn: 12,
        kernel: microkernel_8x12,
        row_major_kernel: microkernel_row_major::<8, 12>,
    },
    MicroKernelSpec {
        name: "m4n16",
        tm: 4,
        tn: 16,
        kernel: microkernel_4x16,
        row_major_kernel: microkernel_row_major::<4, 16>,
    },
];

const M8N12_TM: usize = 8;
const M8N12_TN: usize = 12;
const M4N16_TM: usize = 4;
const M4N16_TN: usize = 16;

#[cfg(feature = "simd")]
type Simd8 = Simd<f32, M8N12_TM>;

#[cfg(feature = "simd")]
type Simd4 = Simd<f32, M4N16_TM>;

const DEFAULT_KERNEL_INDEX: usize = 0;

static CPU_AUTOTUNE_CACHE: OnceLock<Mutex<HashMap<String, usize>>> = OnceLock::new();

fn cpu_autotune_cache() -> &'static Mutex<HashMap<String, usize>> {
    CPU_AUTOTUNE_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn cached_microkernel_index(key: &str) -> Option<usize> {
    lock_recover(cpu_autotune_cache()).get(key).copied()
}

fn cache_microkernel_index(key: &str, index: usize) {
    lock_recover(cpu_autotune_cache()).insert(key.to_owned(), index);
}

#[inline(always)]
fn row_tile_size(rows: usize, inner: usize, tm: usize) -> usize {
    if tm == 0 {
        return rows.max(1);
    }

    if rows <= tm {
        return tm;
    }

    let row_bytes = inner.saturating_mul(core::mem::size_of::<f32>());
    if row_bytes == 0 {
        return tm;
    }

    let mut tile = (L2_TARGET_BYTES / row_bytes).max(1);
    tile = tile.min(rows);

    let tile = tile / tm * tm;
    if tile == 0 {
        tm
    } else {
        tile
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
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
fn pack_a_block(src: &[f32], inner: usize, tm: usize, dst: &mut [f32]) {
    debug_assert_eq!(src.len(), tm * inner);
    debug_assert_eq!(dst.len(), tm * inner);

    for k in 0..inner {
        for row in 0..tm {
            dst[k * tm + row] = src[row * inner + k];
        }
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn compute_with_packed_block(
    spec: &'static MicroKernelSpec,
    dst: &mut [f32],
    lhs: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
    col_start: usize,
    width: usize,
    packed_block: &[f32],
    serial_scratch: &mut Vec<f32>,
) {
    debug_assert_eq!(packed_block.len(), width * inner);

    let tm = spec.tm;
    let tn = spec.tn;
    let kernel = spec.kernel;

    if width >= tn {
        let full_width = width / tn * tn;
        let full_row_blocks = rows / tm;
        if full_row_blocks > 0 {
            let prefix_rows = full_row_blocks * tm;
            let lhs_prefix = &lhs[..prefix_rows * inner];
            let dst_prefix = &mut dst[..prefix_rows * cols];
            let row_tile = row_tile_size(prefix_rows, inner, tm);

            let apply = |dst_chunk: &mut [f32], lhs_chunk: &[f32], packed_a: &mut [f32]| {
                let local_rows = lhs_chunk.len() / inner;
                debug_assert_eq!(local_rows % tm, 0);

                for offset in (0..local_rows).step_by(tm) {
                    let lhs_panel = &lhs_chunk[offset * inner..(offset + tm) * inner];
                    pack_a_block(lhs_panel, inner, tm, packed_a);
                    for col in (0..full_width).step_by(tn) {
                        // SAFETY: A has tm * inner lanes; each B subpanel has tn * inner
                        // elements. Both output row and column spans stay within this chunk.
                        unsafe {
                            kernel(
                                packed_a.as_ptr(),
                                packed_block.as_ptr().add(col * inner),
                                dst_chunk.as_mut_ptr().add(offset * cols + col_start + col),
                                tm,
                                inner,
                                cols,
                                inner,
                            );
                        }
                    }
                }
            };

            if prefix_rows <= row_tile || determinism::lock_reduction_order() {
                // One bounded panel per call, reused across row tiles and RHS blocks.
                serial_scratch.resize(tm * inner, 0.0);
                for (dst_chunk, lhs_chunk) in dst_prefix
                    .chunks_mut(cols * row_tile)
                    .zip(lhs_prefix.chunks(row_tile * inner))
                {
                    apply(dst_chunk, lhs_chunk, serial_scratch);
                }
            } else {
                dst_prefix
                    .par_chunks_mut(cols * row_tile)
                    .zip(lhs_prefix.par_chunks(row_tile * inner))
                    .for_each(|(dst_chunk, lhs_chunk)| {
                        let mut packed_a = vec![0.0f32; tm * inner];
                        apply(dst_chunk, lhs_chunk, &mut packed_a);
                    });
            }
        }

        let processed_rows = (rows / tm) * tm;
        if full_width < width && processed_rows > 0 {
            scalar_block_with_packed(
                dst,
                lhs,
                inner,
                cols,
                0,
                processed_rows,
                col_start + full_width,
                width - full_width,
                &packed_block[full_width * inner..],
            );
        }
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
    debug_assert_eq!(lda, M8N12_TM);
    debug_assert_eq!(ldb, k);
    debug_assert!(ldc >= M8N12_TN);
    let mut acc = [Simd8::splat(0.0); M8N12_TN];

    for p in 0..k {
        let a_slice = core::slice::from_raw_parts(a.add(p * lda), M8N12_TM);
        let a_vec = Simd8::from_slice(a_slice);

        let mut b_ptr = b.add(p);
        for col in 0..M8N12_TN {
            let b_vec = Simd8::splat(*b_ptr);
            acc[col] += a_vec * b_vec;
            b_ptr = b_ptr.add(ldb);
        }
    }

    for col in 0..M8N12_TN {
        let mut dst_ptr = c.add(col);
        let col_vec = acc[col];
        for row in 0..M8N12_TM {
            *dst_ptr += col_vec[row];
            dst_ptr = dst_ptr.add(ldc);
        }
    }
}

#[cfg(not(feature = "simd"))]
#[allow(clippy::needless_range_loop)]
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
    debug_assert_eq!(lda, M8N12_TM);
    debug_assert_eq!(ldb, k);
    debug_assert!(ldc >= M8N12_TN);
    // Match the packed A lanes: adjacent rows are independent accumulators,
    // so vectorization does not need to reassociate the K reduction.
    let mut acc = [[0.0f32; M8N12_TM]; M8N12_TN];

    for p in 0..k {
        let mut b_ptr = b.add(p);
        for col in 0..M8N12_TN {
            let b_val = *b_ptr;
            b_ptr = b_ptr.add(ldb);
            for row in 0..M8N12_TM {
                let a_val = *a.add(p * lda + row);
                acc[col][row] += a_val * b_val;
            }
        }
    }

    for row in 0..M8N12_TM {
        let dst_row = c.add(row * ldc);
        for col in 0..M8N12_TN {
            *dst_row.add(col) += acc[col][row];
        }
    }
}

#[inline(always)]
unsafe fn microkernel_4x16(
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
        microkernel_4x16_simd(a, b, c, lda, ldb, ldc, k);
    }

    #[cfg(not(feature = "simd"))]
    {
        microkernel_4x16_scalar(a, b, c, lda, ldb, ldc, k);
    }
}

#[cfg(feature = "simd")]
#[inline(always)]
unsafe fn microkernel_4x16_simd(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    lda: usize,
    ldb: usize,
    ldc: usize,
    k: usize,
) {
    debug_assert_eq!(lda, M4N16_TM);
    debug_assert_eq!(ldb, k);
    debug_assert!(ldc >= M4N16_TN);
    let mut acc = [Simd4::splat(0.0); M4N16_TN];

    for p in 0..k {
        let a_slice = core::slice::from_raw_parts(a.add(p * lda), M4N16_TM);
        let a_vec = Simd4::from_slice(a_slice);

        let mut b_ptr = b.add(p);
        for col in 0..M4N16_TN {
            let b_vec = Simd4::splat(*b_ptr);
            acc[col] += a_vec * b_vec;
            b_ptr = b_ptr.add(ldb);
        }
    }

    for col in 0..M4N16_TN {
        let mut dst_ptr = c.add(col);
        let col_vec = acc[col];
        for row in 0..M4N16_TM {
            *dst_ptr += col_vec[row];
            dst_ptr = dst_ptr.add(ldc);
        }
    }
}

#[cfg(not(feature = "simd"))]
#[allow(clippy::needless_range_loop)]
#[inline(always)]
unsafe fn microkernel_4x16_scalar(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    lda: usize,
    ldb: usize,
    ldc: usize,
    k: usize,
) {
    debug_assert_eq!(lda, M4N16_TM);
    debug_assert_eq!(ldb, k);
    debug_assert!(ldc >= M4N16_TN);
    let mut acc = [[0.0f32; M4N16_TM]; M4N16_TN];

    for p in 0..k {
        let mut b_ptr = b.add(p);
        for col in 0..M4N16_TN {
            let b_val = *b_ptr;
            b_ptr = b_ptr.add(ldb);
            for row in 0..M4N16_TM {
                let a_val = *a.add(p * lda + row);
                acc[col][row] += a_val * b_val;
            }
        }
    }

    for row in 0..M4N16_TM {
        let dst_row = c.add(row * ldc);
        for col in 0..M4N16_TN {
            *dst_row.add(col) += acc[col][row];
        }
    }
}

fn default_kernel() -> &'static MicroKernelSpec {
    &MICROKERNELS[DEFAULT_KERNEL_INDEX]
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum CpuRhsLayout {
    RowMajor,
    PrepackedColumns,
}

impl CpuRhsLayout {
    fn tag(self) -> &'static str {
        match self {
            Self::RowMajor => "row_major",
            Self::PrepackedColumns => "prepacked_columns",
        }
    }
}

fn select_microkernel(
    rows: usize,
    inner: usize,
    cols: usize,
    rhs_layout: CpuRhsLayout,
) -> &'static MicroKernelSpec {
    autotune_microkernel(rows, inner, cols, rhs_layout).unwrap_or_else(default_kernel)
}

#[inline(always)]
#[allow(clippy::needless_range_loop)]
unsafe fn microkernel_row_major<const TM: usize, const TN: usize>(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    lda: usize,
    ldb: usize,
    ldc: usize,
    k: usize,
) {
    debug_assert!(lda >= k && ldb >= TN && ldc >= TN);
    // Independent column lanes preserve each output's sequential K reduction.
    let mut acc = [[0.0f32; TN]; TM];
    for p in 0..k {
        for row in 0..TM {
            let a_value = *a.add(row * lda + p);
            for col in 0..TN {
                acc[row][col] += a_value * *b.add(p * ldb + col);
            }
        }
    }
    for row in 0..TM {
        for col in 0..TN {
            *c.add(row * ldc + col) += acc[row][col];
        }
    }
}

fn matmul_with_kernel_spec(
    spec: &'static MicroKernelSpec,
    dst: &mut [f32],
    lhs: &[f32],
    rhs: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
) {
    let full_cols = cols / spec.tn * spec.tn;
    let row_tile = row_tile_size(rows, inner, spec.tm);
    let apply = |dst_chunk: &mut [f32], lhs_chunk: &[f32]| {
        let local_rows = lhs_chunk.len() / inner;
        let full_rows = local_rows / spec.tm * spec.tm;
        for row in (0..full_rows).step_by(spec.tm) {
            for col in (0..full_cols).step_by(spec.tn) {
                // SAFETY: each tile reads TM complete LHS rows and TN RHS columns;
                // the disjoint output chunk contains every addressed row and column.
                unsafe {
                    (spec.row_major_kernel)(
                        lhs_chunk.as_ptr().add(row * inner),
                        rhs.as_ptr().add(col),
                        dst_chunk.as_mut_ptr().add(row * cols + col),
                        inner,
                        cols,
                        cols,
                        inner,
                    );
                }
            }
        }
        for row in 0..local_rows {
            let start = if row < full_rows { full_cols } else { 0 };
            direct_row::<true>(
                &mut dst_chunk[row * cols..(row + 1) * cols],
                &lhs_chunk[row * inner..(row + 1) * inner],
                rhs,
                cols,
                start,
            );
        }
    };
    // A partial final row group alone does not justify starting parallel work.
    if rows / spec.tm * spec.tm <= row_tile || determinism::lock_reduction_order() {
        for (dst_chunk, lhs_chunk) in dst
            .chunks_mut(cols * row_tile)
            .zip(lhs.chunks(inner * row_tile))
        {
            apply(dst_chunk, lhs_chunk);
        }
    } else {
        dst.par_chunks_mut(cols * row_tile)
            .zip(lhs.par_chunks(inner * row_tile))
            .for_each(|(dst_chunk, lhs_chunk)| apply(dst_chunk, lhs_chunk));
    }
}

fn matmul_packed_with_kernel_spec(
    spec: &'static MicroKernelSpec,
    dst: &mut [f32],
    lhs: &[f32],
    packed_rhs: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
) {
    let mut serial_scratch = Vec::new();
    compute_with_packed_block(
        spec,
        dst,
        lhs,
        rows,
        inner,
        cols,
        0,
        cols,
        packed_rhs,
        &mut serial_scratch,
    );
}

pub fn is_available() -> bool {
    true
}

pub fn should_use(rows: usize, inner: usize, cols: usize) -> bool {
    let min_tm = MICROKERNELS
        .iter()
        .map(|kernel| kernel.tm)
        .min()
        .unwrap_or(1);
    let min_tn = MICROKERNELS
        .iter()
        .map(|kernel| kernel.tn)
        .min()
        .unwrap_or(1);
    let volume = rows.saturating_mul(inner).saturating_mul(cols);
    volume >= min_tm * min_tn * 4 && rows >= min_tm && inner >= min_tm && cols >= min_tn
}

#[inline]
fn matrix_len(rows: usize, cols: usize, name: &str) -> Result<usize, String> {
    rows.checked_mul(cols)
        .ok_or_else(|| format!("{name} dimensions overflow: {rows}x{cols}"))
}

#[inline]
fn validate_matmul_lengths(
    dst: &[f32],
    lhs: &[f32],
    rhs: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
    rhs_name: &str,
) -> Result<(), String> {
    for (name, actual, expected) in [
        (
            "destination",
            dst.len(),
            matrix_len(rows, cols, "destination")?,
        ),
        ("lhs", lhs.len(), matrix_len(rows, inner, "lhs")?),
        (rhs_name, rhs.len(), matrix_len(inner, cols, rhs_name)?),
    ] {
        if actual != expected {
            return Err(format!(
                "{name} length mismatch: expected {expected} elements, got {actual}"
            ));
        }
    }
    Ok(())
}

#[inline]
fn direct_panel<const WIDTH: usize, const ACCUMULATE: bool>(
    out: &mut [f32],
    row: &[f32],
    rhs: &[f32],
    cols: usize,
    col: usize,
) {
    // Fixed lanes keep partial sums local; each lane still reduces K in order.
    let mut sums = [0.0f32; WIDTH];
    for (&a, rhs_row) in row.iter().zip(rhs.chunks_exact(cols)) {
        for (sum, &b) in sums.iter_mut().zip(&rhs_row[col..col + WIDTH]) {
            *sum += a * b;
        }
    }
    if ACCUMULATE {
        for (value, sum) in out[col..col + WIDTH].iter_mut().zip(sums) {
            *value += sum;
        }
    } else {
        out[col..col + WIDTH].copy_from_slice(&sums);
    }
}

#[inline]
fn direct_row<const ACCUMULATE: bool>(
    out: &mut [f32],
    row: &[f32],
    rhs: &[f32],
    cols: usize,
    start: usize,
) {
    let end = start + (cols - start) / 8 * 8;
    for col in (start..end).step_by(8) {
        direct_panel::<8, ACCUMULATE>(out, row, rhs, cols, col);
    }
    let tail = if cols - end >= 4 {
        direct_panel::<4, ACCUMULATE>(out, row, rhs, cols, end);
        end + 4
    } else {
        end
    };
    for (col, value) in out.iter_mut().enumerate().skip(tail) {
        let mut sum = 0.0f32;
        for (&a, rhs_row) in row.iter().zip(rhs.chunks_exact(cols)) {
            sum += a * rhs_row[col];
        }
        if ACCUMULATE {
            *value += sum;
        } else {
            *value = sum;
        }
    }
}

fn matmul_direct(dst: &mut [f32], lhs: &[f32], rhs: &[f32], inner: usize, cols: usize) {
    for (out, row) in dst.chunks_exact_mut(cols).zip(lhs.chunks_exact(inner)) {
        direct_row::<false>(out, row, rhs, cols, 0);
    }
}

pub fn matmul_into(
    dst: &mut [f32],
    lhs: &[f32],
    rhs: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<(), String> {
    validate_matmul_lengths(dst, lhs, rhs, rows, inner, cols, "rhs")?;

    if rows == 0 || cols == 0 || inner == 0 {
        dst.fill(0.0);
        return Ok(());
    }

    dst.fill(0.0);

    if !should_use(rows, inner, cols) {
        matmul_direct(dst, lhs, rhs, inner, cols);
        return Ok(());
    }
    let kernel = select_microkernel(rows, inner, cols, CpuRhsLayout::RowMajor);
    if rows < kernel.tm || cols < kernel.tn {
        matmul_direct(dst, lhs, rhs, inner, cols);
    } else {
        matmul_with_kernel_spec(kernel, dst, lhs, rhs, rows, inner, cols);
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
    validate_matmul_lengths(dst, lhs, packed_rhs, rows, inner, cols, "packed rhs")?;

    if rows == 0 || cols == 0 || inner == 0 {
        dst.fill(0.0);
        return Ok(());
    }

    dst.fill(0.0);

    let kernel = select_microkernel(rows, inner, cols, CpuRhsLayout::PrepackedColumns);
    matmul_packed_with_kernel_spec(kernel, dst, lhs, packed_rhs, rows, inner, cols);

    Ok(())
}

pub fn prepack_rhs(rhs: &[f32], inner: usize, cols: usize) -> Result<Vec<f32>, String> {
    let len = matrix_len(inner, cols, "rhs")?;
    if rhs.len() != len {
        return Err(format!(
            "rhs length mismatch: expected {} elements, got {}",
            len,
            rhs.len()
        ));
    }

    let mut packed = vec![0.0f32; len];
    if len == 0 {
        return Ok(packed);
    }
    for col in 0..cols {
        for k in 0..inner {
            packed[col * inner + k] = rhs[k * cols + col];
        }
    }

    Ok(packed)
}

const CPU_AUTOTUNE_REVISION: u64 = 5;
const CPU_AUTOTUNE_MIN_VOLUME: usize = 64 * 64 * 32;
const CPU_AUTOTUNE_SAMPLE_MAX_DIM: usize = 2048;
const CPU_AUTOTUNE_WARMUP_RUNS: usize = 1;
const CPU_AUTOTUNE_SAMPLE_RUNS: usize = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredCpuKernel {
    kernel: String,
}

fn autotune_microkernel(
    rows: usize,
    inner: usize,
    cols: usize,
    rhs_layout: CpuRhsLayout,
) -> Option<&'static MicroKernelSpec> {
    if !should_autotune(rows, inner, cols) {
        return None;
    }

    let (bucket_rows, bucket_inner, bucket_cols) = quantized_problem(rows, inner, cols);
    let (key, path) = cpu_autotune_key(bucket_rows, bucket_inner, bucket_cols, rhs_layout)?;

    if let Some(index) = cached_microkernel_index(&key) {
        return MICROKERNELS.get(index);
    }

    let sample_rows = sample_dimension(bucket_rows);
    let sample_inner = sample_dimension(bucket_inner);
    let sample_cols = sample_dimension(bucket_cols);

    let context = CpuAutotuneContext {
        rows: bucket_rows,
        inner: bucket_inner,
        cols: bucket_cols,
        sample_rows,
        sample_inner,
        sample_cols,
        revision: CPU_AUTOTUNE_REVISION,
        runs: CPU_AUTOTUNE_SAMPLE_RUNS as u32,
        rhs_layout,
    };

    let autotune_enabled = autotune_env_enabled();
    eprintln!("[autotune] key={key} apply={autotune_enabled}");

    let matches = if autotune_enabled {
        lookup_similar(path.as_path(), &key, &context, 4)
    } else {
        Vec::new()
    };

    if autotune_enabled {
        let stored = load_best_typed(path.as_path(), &key, &context, None::<StoredCpuKernel>);
        if let Some(stored) = stored {
            if let Some(index) = MICROKERNELS
                .iter()
                .position(|spec| spec.name == stored.kernel)
            {
                cache_microkernel_index(&key, index);
                return MICROKERNELS.get(index);
            }
        }
    }

    let lhs_len = sample_rows.checked_mul(sample_inner)?;
    let rhs_len = sample_inner.checked_mul(sample_cols)?;
    let out_len = sample_rows.checked_mul(sample_cols)?;

    let lhs = vec![1.0f32; lhs_len];
    let rhs = vec![1.0f32; rhs_len];
    let mut scratch = vec![0.0f32; out_len];

    let mut ordered_indices: Vec<usize> = (0..MICROKERNELS.len()).collect();
    if !matches.is_empty() {
        reorder_kernels(&mut ordered_indices, &matches);
    }

    let mut best: Option<(usize, f64)> = None;
    for index in ordered_indices {
        let spec = &MICROKERNELS[index];
        match microbenchmark_kernel(
            spec,
            (sample_rows, sample_inner, sample_cols),
            rhs_layout,
            &lhs,
            &rhs,
            scratch.as_mut_slice(),
        ) {
            Ok(score) => {
                let update = best
                    .map(|(_, best_score)| score < best_score)
                    .unwrap_or(true);
                if update {
                    best = Some((index, score));
                }
            }
            Err(_) => continue,
        }
    }

    if let Some((index, score)) = best {
        cache_microkernel_index(&key, index);
        if autotune_enabled {
            let stored = StoredCpuKernel {
                kernel: MICROKERNELS[index].name.to_string(),
            };
            let _ = record_best(path.as_path(), &key, &context, score, &stored);
        }
        MICROKERNELS.get(index)
    } else {
        None
    }
}

fn reorder_kernels(order: &mut [usize], matches: &[AutoTuneMatch]) {
    let mut scored: Vec<(f64, usize)> = order
        .iter()
        .copied()
        .map(|index| {
            let name = MICROKERNELS[index].name;
            let score = matches
                .iter()
                .filter_map(|m| {
                    serde_json::from_value::<StoredCpuKernel>(m.entry.params.clone()).ok()
                })
                .find(|stored| stored.kernel == name)
                .and_then(|stored| {
                    matches.iter().find_map(|m| {
                        serde_json::from_value::<StoredCpuKernel>(m.entry.params.clone())
                            .ok()
                            .filter(|candidate| candidate.kernel == stored.kernel)
                            .map(|_| m.entry.score)
                    })
                })
                .unwrap_or(f64::INFINITY);
            (score, index)
        })
        .collect();

    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    for (slot, (_, index)) in scored.into_iter().enumerate() {
        order[slot] = index;
    }
}

#[derive(Serialize)]
struct CpuAutotuneContext {
    rows: usize,
    inner: usize,
    cols: usize,
    sample_rows: usize,
    sample_inner: usize,
    sample_cols: usize,
    revision: u64,
    runs: u32,
    rhs_layout: CpuRhsLayout,
}

fn microbenchmark_kernel(
    spec: &'static MicroKernelSpec,
    (rows, inner, cols): (usize, usize, usize),
    rhs_layout: CpuRhsLayout,
    lhs: &[f32],
    rhs: &[f32],
    scratch: &mut [f32],
) -> Result<f64, String> {
    if rows == 0 || inner == 0 || cols == 0 {
        return Ok(0.0);
    }
    let apply = |scratch: &mut [f32]| match rhs_layout {
        CpuRhsLayout::RowMajor => {
            matmul_with_kernel_spec(spec, scratch, lhs, rhs, rows, inner, cols);
        }
        CpuRhsLayout::PrepackedColumns => {
            matmul_packed_with_kernel_spec(spec, scratch, lhs, rhs, rows, inner, cols);
        }
    };

    for _ in 0..CPU_AUTOTUNE_WARMUP_RUNS {
        scratch.fill(0.0);
        apply(scratch);
    }

    let mut total = Duration::default();
    for _ in 0..CPU_AUTOTUNE_SAMPLE_RUNS {
        scratch.fill(0.0);
        let start = Instant::now();
        apply(scratch);
        total += start.elapsed();
    }

    if CPU_AUTOTUNE_SAMPLE_RUNS == 0 {
        return Ok(0.0);
    }

    Ok(total.as_secs_f64() / CPU_AUTOTUNE_SAMPLE_RUNS as f64)
}

fn should_autotune(rows: usize, inner: usize, cols: usize) -> bool {
    if rows == 0 || inner == 0 || cols == 0 {
        return false;
    }

    rows.checked_mul(inner)
        .and_then(|volume| volume.checked_mul(cols))
        .map(|volume| volume >= CPU_AUTOTUNE_MIN_VOLUME)
        .unwrap_or(false)
}

fn quantized_problem(rows: usize, inner: usize, cols: usize) -> (usize, usize, usize) {
    (
        quantize_dimension(rows),
        quantize_dimension(inner),
        quantize_dimension(cols),
    )
}

fn quantize_dimension(value: usize) -> usize {
    if value == 0 {
        return 0;
    }

    let step = if value <= 64 {
        8
    } else if value <= 256 {
        16
    } else if value <= 1024 {
        32
    } else {
        64
    };

    ((value + step / 2) / step).max(1) * step
}

fn sample_dimension(value: usize) -> usize {
    quantize_dimension(value).clamp(1, CPU_AUTOTUNE_SAMPLE_MAX_DIM)
}

fn cpu_autotune_key(
    rows: usize,
    inner: usize,
    cols: usize,
    rhs_layout: CpuRhsLayout,
) -> Option<(String, PathBuf)> {
    let path = autotune_store_path()?;
    Some((cpu_autotune_signature(rows, inner, cols, rhs_layout), path))
}

fn cpu_autotune_signature(
    rows: usize,
    inner: usize,
    cols: usize,
    rhs_layout: CpuRhsLayout,
) -> String {
    let arch = env::consts::ARCH;
    let os = env::consts::OS;
    let features = cpu_feature_tag();
    let layout = rhs_layout.tag();
    let portable_simd = cfg!(feature = "simd");
    format!(
        "cpu.matmul.v{CPU_AUTOTUNE_REVISION:02}|{arch}|{os}|{features}|simd={portable_simd}|{layout}|{rows}x{inner}x{cols}|runs{CPU_AUTOTUNE_SAMPLE_RUNS}"
    )
}

fn autotune_env_enabled() -> bool {
    env::var("SPIRALTORCH_AUTOTUNE")
        .map(|v| v != "0")
        .unwrap_or(true)
}

fn autotune_store_path() -> Option<PathBuf> {
    if let Some(path) = env::var_os("SPIRALTORCH_AUTOTUNE_STORE") {
        return Some(PathBuf::from(path));
    }
    if let Some(home) = env::var_os("HOME") {
        let mut path = PathBuf::from(home);
        path.push(".spiraltorch");
        path.push("kernels.json");
        return Some(path);
    }
    None
}

#[cfg(target_arch = "x86_64")]
fn cpu_feature_tag() -> String {
    let mut features = Vec::new();
    if std::is_x86_feature_detected!("avx512f") {
        features.push("avx512f");
    }
    if std::is_x86_feature_detected!("avx2") {
        features.push("avx2");
    }
    if std::is_x86_feature_detected!("fma") {
        features.push("fma");
    }
    if std::is_x86_feature_detected!("avx") {
        features.push("avx");
    }
    if std::is_x86_feature_detected!("sse4.2") {
        features.push("sse4_2");
    }
    if features.is_empty() {
        features.push("baseline");
    }
    features.join("+")
}

#[cfg(not(target_arch = "x86_64"))]
fn cpu_feature_tag() -> String {
    "baseline".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    #[test]
    fn autotune_cache_recovers_after_poison() {
        let cache = cpu_autotune_cache();
        let poisoned = catch_unwind(AssertUnwindSafe(|| {
            let _guard = cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            panic!("poison CPU autotune cache");
        }));
        assert!(poisoned.is_err());
        assert!(cache.is_poisoned());

        cache_microkernel_index("poison.recovery", 1);

        assert!(!cache.is_poisoned());
        assert_eq!(cached_microkernel_index("poison.recovery"), Some(1));
        lock_recover(cache).remove("poison.recovery");
    }

    fn reference_matmul(
        lhs: &[f32],
        rhs: &[f32],
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let mut acc = 0.0f32;
                for k in 0..inner {
                    acc += lhs[r * inner + k] * rhs[k * cols + c];
                }
                out[r * cols + c] = acc;
            }
        }
        out
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        const EPS: f32 = 1e-4;
        assert_eq!(actual.len(), expected.len());
        for (idx, (&actual_value, &expected_value)) in
            actual.iter().zip(expected.iter()).enumerate()
        {
            assert!(
                (actual_value - expected_value).abs() < EPS,
                "mismatch at index {idx}: actual={actual_value}, expected={expected_value}"
            );
        }
    }

    #[track_caller]
    fn unwrap_ok<T>(result: Result<T, String>) -> T {
        match result {
            Ok(value) => value,
            Err(error) => panic!("expected Ok(..), got Err({error})"),
        }
    }

    #[test]
    fn matmul_into_matches_reference_m8n12() {
        let rows = 8;
        let inner = 7;
        let cols = 12;
        let lhs: Vec<f32> = (0..rows * inner)
            .map(|idx| ((idx % 13) as f32 * 0.25) - 1.5)
            .collect();
        let rhs: Vec<f32> = (0..inner * cols)
            .map(|idx| ((idx % 17) as f32 * 0.1) - 0.8)
            .collect();
        let expected = reference_matmul(&lhs, &rhs, rows, inner, cols);

        let mut dst = vec![0.0f32; rows * cols];
        unwrap_ok(matmul_into(&mut dst, &lhs, &rhs, rows, inner, cols));
        assert_close(&dst, &expected);
    }

    #[test]
    fn matmul_into_matches_reference_with_tail_and_partial_rows() {
        let rows = 10;
        let inner = 9;
        let cols = 13;
        let lhs: Vec<f32> = (0..rows * inner)
            .map(|idx| ((idx % 11) as f32 * 0.33) - 1.1)
            .collect();
        let rhs: Vec<f32> = (0..inner * cols)
            .map(|idx| ((idx % 19) as f32 * 0.15) - 1.25)
            .collect();
        let expected = reference_matmul(&lhs, &rhs, rows, inner, cols);

        let mut dst = vec![0.0f32; rows * cols];
        unwrap_ok(matmul_into(&mut dst, &lhs, &rhs, rows, inner, cols));
        assert_close(&dst, &expected);
    }

    #[test]
    fn matmul_packed_into_matches_unpacked() {
        let rows = 8;
        let inner = 6;
        let cols = 16;
        let lhs: Vec<f32> = (0..rows * inner)
            .map(|idx| ((idx % 7) as f32 * 0.5) - 1.0)
            .collect();
        let rhs: Vec<f32> = (0..inner * cols)
            .map(|idx| ((idx % 23) as f32 * 0.2) - 2.0)
            .collect();

        let mut dst_unpacked = vec![0.0f32; rows * cols];
        unwrap_ok(matmul_into(
            &mut dst_unpacked,
            &lhs,
            &rhs,
            rows,
            inner,
            cols,
        ));

        let packed_rhs = unwrap_ok(prepack_rhs(&rhs, inner, cols));
        let mut dst_packed = vec![0.0f32; rows * cols];
        unwrap_ok(matmul_packed_into(
            &mut dst_packed,
            &lhs,
            &packed_rhs,
            rows,
            inner,
            cols,
        ));

        assert_close(&dst_packed, &dst_unpacked);
    }

    #[test]
    fn matmul_with_kernel_spec_matches_reference_m4n16() {
        let rows = 4;
        let inner = 5;
        let cols = 16;
        let lhs: Vec<f32> = (0..rows * inner)
            .map(|idx| ((idx % 9) as f32 * 0.4) - 1.75)
            .collect();
        let rhs: Vec<f32> = (0..inner * cols)
            .map(|idx| ((idx % 5) as f32 * 0.6) - 0.2)
            .collect();
        let expected = reference_matmul(&lhs, &rhs, rows, inner, cols);

        let mut dst = vec![0.0f32; rows * cols];
        matmul_with_kernel_spec(&MICROKERNELS[1], &mut dst, &lhs, &rhs, rows, inner, cols);
        assert_close(&dst, &expected);
    }

    #[test]
    fn every_kernel_preserves_sequential_reduction_with_tails() {
        for spec in MICROKERNELS {
            for (rows, inner, cols) in [
                (spec.tm, 1, spec.tn),
                (spec.tm * 2 + 1, 37, spec.tn * 2 + 3),
                (spec.tm + 1, 1025, spec.tn + 1),
                (spec.tm * 3 + 1, 137, spec.tn * 16 + 3),
                (spec.tm + 1, 2049, spec.tn * 2 + 1),
            ] {
                let lhs: Vec<f32> = (0..rows * inner)
                    .map(|i| ((i * 17 % 127) as f32 - 63.0) / 31.0)
                    .collect();
                let rhs: Vec<f32> = (0..inner * cols)
                    .map(|i| ((i * 29 % 131) as f32 - 65.0) / 37.0)
                    .collect();
                let expected = reference_matmul(&lhs, &rhs, rows, inner, cols);
                let packed = unwrap_ok(prepack_rhs(&rhs, inner, cols));
                for use_packed in [false, true] {
                    let mut actual = vec![0.25; rows * cols];
                    if use_packed {
                        matmul_packed_with_kernel_spec(
                            spec,
                            &mut actual,
                            &lhs,
                            &packed,
                            rows,
                            inner,
                            cols,
                        );
                    } else {
                        matmul_with_kernel_spec(spec, &mut actual, &lhs, &rhs, rows, inner, cols);
                    }
                    for (index, (&actual, &expected)) in
                        actual.iter().zip(expected.iter()).enumerate()
                    {
                        assert_eq!(
                            actual.to_bits(),
                            (0.25 + expected).to_bits(),
                            "{} {rows}x{inner}x{cols}, packed={use_packed}, index={index}",
                            spec.name,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn public_paths_preserve_sequential_reduction_and_overwrite_destination() {
        for (rows, inner, cols) in [
            (1, 31, 8),
            (3, 37, 13),
            (4, 5, 16),
            (7, 31, 8),
            (8, 31, 8),
            (8, 7, 12),
            (17, 37, 29),
            (32, 64, 32),
            (96, 128, 96),
            (65, 257, 49),
            (129, 129, 97),
            (129, 1025, 33),
            (1, 4096, 1),
            (16, 0, 12),
            (0, 31, 8),
            (3, 17, 0),
        ] {
            let lhs: Vec<_> = (0..rows * inner)
                .map(|i| ((i * 17 % 127) as f32 - 63.0) / 31.0)
                .collect();
            let rhs: Vec<_> = (0..inner * cols)
                .map(|i| ((i * 29 % 131) as f32 - 65.0) / 37.0)
                .collect();
            let expected = reference_matmul(&lhs, &rhs, rows, inner, cols);
            let packed = unwrap_ok(prepack_rhs(&rhs, inner, cols));
            for use_packed in [false, true] {
                let mut dst = vec![f32::NAN; rows * cols];
                for _ in 0..2 {
                    if use_packed {
                        unwrap_ok(matmul_packed_into(
                            &mut dst, &lhs, &packed, rows, inner, cols,
                        ));
                    } else {
                        unwrap_ok(matmul_into(&mut dst, &lhs, &rhs, rows, inner, cols));
                    }
                    assert_eq!(
                        dst.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "{rows}x{inner}x{cols}, packed={use_packed}"
                    );
                }
            }
        }
    }

    #[test]
    fn invalid_lengths_and_overflow_are_errors_before_writing() {
        for (rows, inner, cols) in [(usize::MAX, 2, 0), (0, usize::MAX, 2), (usize::MAX, 0, 2)] {
            let mut dst = [19.0];
            assert!(matmul_into(&mut dst, &[], &[], rows, inner, cols)
                .unwrap_err()
                .contains("overflow"));
            assert!(matmul_packed_into(&mut dst, &[], &[], rows, inner, cols)
                .unwrap_err()
                .contains("overflow"));
            assert_eq!(dst, [19.0]);
        }
        assert!(prepack_rhs(&[], usize::MAX, 2)
            .unwrap_err()
            .contains("overflow"));
        assert!(should_use(usize::MAX, usize::MAX, usize::MAX));
        assert!(unwrap_ok(prepack_rhs(&[], 0, usize::MAX)).is_empty());
        for (a, b) in [(vec![1.0], vec![1.0; 4]), (vec![1.0; 4], vec![1.0])] {
            let mut dst = [19.0; 4];
            assert!(matmul_into(&mut dst, &a, &b, 2, 2, 2).is_err());
            assert!(matmul_packed_into(&mut dst, &a, &b, 2, 2, 2).is_err());
            assert_eq!(dst, [19.0; 4]);
        }
    }

    #[test]
    fn serial_panel_storage_is_reused_between_rhs_blocks() {
        let (rows, inner, cols) = (8, 7, 24);
        let lhs = vec![1.0; rows * inner];
        let rhs = vec![1.0; 12 * inner];
        let mut dst = vec![0.0; rows * cols];
        let mut scratch = Vec::new();
        compute_with_packed_block(
            default_kernel(),
            &mut dst,
            &lhs,
            rows,
            inner,
            cols,
            0,
            12,
            &rhs,
            &mut scratch,
        );
        let pointer = scratch.as_ptr();
        let capacity = scratch.capacity();
        compute_with_packed_block(
            default_kernel(),
            &mut dst,
            &lhs,
            rows,
            inner,
            cols,
            12,
            12,
            &rhs,
            &mut scratch,
        );
        assert_eq!(scratch.as_ptr(), pointer);
        assert_eq!(scratch.capacity(), capacity);
        assert_eq!(dst, vec![7.0; rows * cols]);
    }

    #[test]
    fn direct_panels_preserve_every_small_width_and_tail() {
        for rows in [1, 3, 7] {
            for inner in [1, 3, 17] {
                for cols in 1..=33 {
                    let lhs: Vec<_> = (0..rows * inner)
                        .map(|i| (i % 19) as f32 / 7.0 - 1.0)
                        .collect();
                    let rhs: Vec<_> = (0..inner * cols)
                        .map(|i| (i % 23) as f32 / 11.0 - 1.0)
                        .collect();
                    let expected = reference_matmul(&lhs, &rhs, rows, inner, cols);
                    let mut dst = vec![f32::NAN; rows * cols];
                    matmul_direct(&mut dst, &lhs, &rhs, inner, cols);
                    assert_eq!(
                        dst.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                }
            }
        }
    }

    #[test]
    fn row_major_kernels_respect_strides_and_output_guards() {
        for spec in MICROKERNELS {
            for inner in [0, 1, 5, 37] {
                let (lda, ldb, ldc) = (inner + 3, spec.tn + 5, spec.tn + 7);
                let lhs: Vec<_> = (0..spec.tm * lda + 1)
                    .map(|i| (i % 19) as f32 / 7.0 - 1.0)
                    .collect();
                let rhs: Vec<_> = (0..inner * ldb + 1)
                    .map(|i| (i % 23) as f32 / 11.0 - 1.0)
                    .collect();
                let mut actual = vec![17.25; (spec.tm + 2) * ldc];
                let mut expected = actual.clone();
                for row in 0..spec.tm {
                    for col in 0..spec.tn {
                        let mut sum = 0.0f32;
                        for k in 0..inner {
                            sum += lhs[1 + row * lda + k] * rhs[1 + k * ldb + col];
                        }
                        expected[(row + 1) * ldc + 2 + col] += sum;
                    }
                }
                // SAFETY: the padded fixtures cover both input tiles and all output
                // rows; surrounding guard elements are deliberately not writable.
                unsafe {
                    (spec.row_major_kernel)(
                        lhs.as_ptr().add(1),
                        rhs.as_ptr().add(1),
                        actual.as_mut_ptr().add(ldc + 2),
                        lda,
                        ldb,
                        ldc,
                        inner,
                    );
                }
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn row_major_tiles_preserve_every_row_and_column_tail() {
        for spec in MICROKERNELS {
            for row_tail in 0..spec.tm {
                for col_tail in 0..spec.tn {
                    let (rows, inner, cols) = (spec.tm * 2 + row_tail, 17, spec.tn * 2 + col_tail);
                    let lhs: Vec<_> = (0..rows * inner)
                        .map(|i| (i % 19) as f32 / 7.0 - 1.0)
                        .collect();
                    let rhs: Vec<_> = (0..inner * cols)
                        .map(|i| (i % 23) as f32 / 11.0 - 1.0)
                        .collect();
                    let expected = reference_matmul(&lhs, &rhs, rows, inner, cols);
                    let mut actual = vec![0.25; rows * cols];
                    matmul_with_kernel_spec(spec, &mut actual, &lhs, &rhs, rows, inner, cols);
                    for (&a, &b) in actual.iter().zip(&expected) {
                        assert_eq!(a.to_bits(), (0.25 + b).to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn autotune_keeps_rhs_layouts_separate_and_benchmarks_the_actual_path() {
        let (rows, inner, cols) = (9, 17, 35);
        let row_key = cpu_autotune_signature(rows, inner, cols, CpuRhsLayout::RowMajor);
        let packed_key = cpu_autotune_signature(rows, inner, cols, CpuRhsLayout::PrepackedColumns);
        assert_ne!(row_key, packed_key);
        assert!(row_key.contains("|row_major|"));
        assert!(packed_key.contains("|prepacked_columns|"));

        let lhs: Vec<_> = (0..rows * inner)
            .map(|i| (i % 19) as f32 / 7.0 - 1.0)
            .collect();
        let rhs: Vec<_> = (0..inner * cols)
            .map(|i| (i % 23) as f32 / 11.0 - 1.0)
            .collect();
        let packed = unwrap_ok(prepack_rhs(&rhs, inner, cols));
        let expected = reference_matmul(&lhs, &rhs, rows, inner, cols);
        for spec in MICROKERNELS {
            for (layout, data) in [
                (CpuRhsLayout::RowMajor, &rhs),
                (CpuRhsLayout::PrepackedColumns, &packed),
            ] {
                let mut actual = vec![f32::NAN; rows * cols];
                let elapsed = unwrap_ok(microbenchmark_kernel(
                    spec,
                    (rows, inner, cols),
                    layout,
                    &lhs,
                    data,
                    &mut actual,
                ));
                assert!(elapsed.is_finite() && elapsed >= 0.0);
                for (&a, &b) in actual.iter().zip(&expected) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
            }
        }
    }
}
