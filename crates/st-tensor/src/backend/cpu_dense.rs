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
}

const MICROKERNELS: &[MicroKernelSpec] = &[
    MicroKernelSpec {
        name: "m8n12",
        tm: 8,
        tn: 12,
        kernel: microkernel_8x12,
    },
    MicroKernelSpec {
        name: "m4n16",
        tm: 4,
        tn: 16,
        kernel: microkernel_4x16,
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
fn pack_a_block(src: &[f32], inner: usize, tm: usize, dst: &mut [f32]) {
    debug_assert_eq!(src.len(), tm * inner);
    debug_assert_eq!(dst.len(), tm * inner);

    unsafe {
        for k in 0..inner {
            let dst_col = dst.as_mut_ptr().add(k * tm);
            let mut src_ptr = src.as_ptr().add(k);
            for row in 0..tm {
                *dst_col.add(row) = *src_ptr;
                src_ptr = src_ptr.add(inner);
            }
        }
    }
}

#[inline]
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
) {
    debug_assert_eq!(packed_block.len(), width * inner);

    let tm = spec.tm;
    let tn = spec.tn;
    let kernel = spec.kernel;

    if width == tn {
        let full_row_blocks = rows / tm;
        if full_row_blocks > 0 {
            let prefix_rows = full_row_blocks * tm;
            let lhs_prefix = &lhs[..prefix_rows * inner];
            let dst_prefix = &mut dst[..prefix_rows * cols];
            let row_tile = row_tile_size(prefix_rows, inner, tm);

            dst_prefix
                .par_chunks_mut(cols * row_tile)
                .zip(lhs_prefix.par_chunks(row_tile * inner))
                .for_each(|(dst_chunk, lhs_chunk)| {
                    let local_rows = lhs_chunk.len() / inner;
                    let mut packed_a = vec![0.0f32; tm * inner];

                    for offset in (0..local_rows).step_by(tm) {
                        let lhs_panel = &lhs_chunk[offset * inner..(offset + tm) * inner];
                        pack_a_block(lhs_panel, inner, tm, &mut packed_a);
                        unsafe {
                            kernel(
                                packed_a.as_ptr(),
                                packed_block.as_ptr(),
                                dst_chunk.as_mut_ptr().add(offset * cols + col_start),
                                tm,
                                inner,
                                cols,
                                inner,
                            );
                        }
                    }
                });
        }

        let processed_rows = (rows / tm) * tm;
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
    let mut acc = [[0.0f32; M8N12_TN]; M8N12_TM];

    for p in 0..k {
        let mut b_ptr = b.add(p);
        for col in 0..M8N12_TN {
            let b_val = *b_ptr;
            b_ptr = b_ptr.add(ldb);
            for row in 0..M8N12_TM {
                let a_val = *a.add(p * lda + row);
                acc[row][col] += a_val * b_val;
            }
        }
    }

    for row in 0..M8N12_TM {
        let row_acc = acc[row];
        let dst_row = c.add(row * ldc);
        for col in 0..M8N12_TN {
            *dst_row.add(col) += row_acc[col];
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
    let mut acc = [[0.0f32; M4N16_TN]; M4N16_TM];

    for p in 0..k {
        let mut b_ptr = b.add(p);
        for col in 0..M4N16_TN {
            let b_val = *b_ptr;
            b_ptr = b_ptr.add(ldb);
            for row in 0..M4N16_TM {
                let a_val = *a.add(p * lda + row);
                acc[row][col] += a_val * b_val;
            }
        }
    }

    for row in 0..M4N16_TM {
        let row_acc = acc[row];
        let dst_row = c.add(row * ldc);
        for col in 0..M4N16_TN {
            *dst_row.add(col) += row_acc[col];
        }
    }
}

fn default_kernel() -> &'static MicroKernelSpec {
    &MICROKERNELS[DEFAULT_KERNEL_INDEX]
}

fn select_microkernel(rows: usize, inner: usize, cols: usize) -> &'static MicroKernelSpec {
    autotune_microkernel(rows, inner, cols).unwrap_or_else(default_kernel)
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
    let tn = spec.tn;
    let mut packed_panel = vec![0.0f32; tn * inner];
    let full_blocks = if tn == 0 { 0 } else { cols / tn };
    let tail = if tn == 0 { cols } else { cols % tn };

    for block in 0..full_blocks {
        let col_start = block * tn;
        pack_b_block_into(packed_panel.as_mut_slice(), rhs, inner, cols, col_start, tn);
        compute_with_packed_block(
            spec,
            dst,
            lhs,
            rows,
            inner,
            cols,
            col_start,
            tn,
            packed_panel.as_slice(),
        );
    }

    if tail > 0 {
        let col_start = full_blocks * tn;
        pack_b_block_into(
            &mut packed_panel[..tail * inner],
            rhs,
            inner,
            cols,
            col_start,
            tail,
        );
        compute_with_packed_block(
            spec,
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
    let tn = spec.tn;
    let full_blocks = if tn == 0 { 0 } else { cols / tn };
    let tail = if tn == 0 { cols } else { cols % tn };

    for block in 0..full_blocks {
        let col_start = block * tn;
        let packed_block = &packed_rhs[col_start * inner..(col_start + tn) * inner];
        compute_with_packed_block(
            spec,
            dst,
            lhs,
            rows,
            inner,
            cols,
            col_start,
            tn,
            packed_block,
        );
    }

    if tail > 0 {
        let col_start = full_blocks * tn;
        let packed_block = &packed_rhs[col_start * inner..(col_start + tail) * inner];
        compute_with_packed_block(
            spec,
            dst,
            lhs,
            rows,
            inner,
            cols,
            col_start,
            tail,
            packed_block,
        );
    }
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
    let volume = rows * inner * cols;
    volume >= min_tm * min_tn * 4 && rows >= min_tm && inner >= min_tm && cols >= min_tn
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

    let kernel = select_microkernel(rows, inner, cols);
    matmul_with_kernel_spec(kernel, dst, lhs, rhs, rows, inner, cols);

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

    let kernel = select_microkernel(rows, inner, cols);
    matmul_packed_with_kernel_spec(kernel, dst, lhs, packed_rhs, rows, inner, cols);

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
    for col in 0..cols {
        for k in 0..inner {
            packed[col * inner + k] = rhs[k * cols + col];
        }
    }

    Ok(packed)
}

const CPU_AUTOTUNE_REVISION: u64 = 1;
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
) -> Option<&'static MicroKernelSpec> {
    if !should_autotune(rows, inner, cols) {
        return None;
    }

    let (bucket_rows, bucket_inner, bucket_cols) = quantized_problem(rows, inner, cols);
    let (key, path) = cpu_autotune_key(bucket_rows, bucket_inner, bucket_cols)?;

    if let Some(index) = CPU_AUTOTUNE_CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .ok()
        .and_then(|guard| guard.get(&key).copied())
    {
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
                if let Ok(mut cache) = CPU_AUTOTUNE_CACHE
                    .get_or_init(|| Mutex::new(HashMap::new()))
                    .lock()
                {
                    cache.insert(key.clone(), index);
                }
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
            sample_rows,
            sample_inner,
            sample_cols,
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
        if let Ok(mut cache) = CPU_AUTOTUNE_CACHE
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
        {
            cache.insert(key.clone(), index);
        }
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
}

fn microbenchmark_kernel(
    spec: &'static MicroKernelSpec,
    rows: usize,
    inner: usize,
    cols: usize,
    lhs: &[f32],
    rhs: &[f32],
    scratch: &mut [f32],
) -> Result<f64, String> {
    if rows == 0 || inner == 0 || cols == 0 {
        return Ok(0.0);
    }

    for _ in 0..CPU_AUTOTUNE_WARMUP_RUNS {
        scratch.fill(0.0);
        matmul_with_kernel_spec(spec, scratch, lhs, rhs, rows, inner, cols);
    }

    let mut total = Duration::default();
    for _ in 0..CPU_AUTOTUNE_SAMPLE_RUNS {
        scratch.fill(0.0);
        let start = Instant::now();
        matmul_with_kernel_spec(spec, scratch, lhs, rhs, rows, inner, cols);
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
    quantize_dimension(value)
        .min(CPU_AUTOTUNE_SAMPLE_MAX_DIM)
        .max(1)
}

fn cpu_autotune_key(rows: usize, inner: usize, cols: usize) -> Option<(String, PathBuf)> {
    let path = autotune_store_path()?;
    let arch = env::consts::ARCH;
    let os = env::consts::OS;
    let features = cpu_feature_tag();
    let key = format!(
        "cpu.matmul.v{CPU_AUTOTUNE_REVISION:02}|{arch}|{os}|{features}|{rows}x{inner}x{cols}|runs{CPU_AUTOTUNE_SAMPLE_RUNS}"
    );
    Some((key, path))
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
