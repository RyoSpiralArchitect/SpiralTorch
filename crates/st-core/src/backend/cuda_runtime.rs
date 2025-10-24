// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(feature = "cuda")]

use crate::backend::cuda_loader::{self, CudaModule};
use crate::backend::rankk_launch::LaunchSlices;
use crate::backend::rankk_software::Selection;
use crate::ops::rank_entry::RankPlan;
use cudarc::driver::{LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::f32;
use std::sync::OnceLock;

const MODULE_NAME: &str = "spiraltorch_rankk";
const TOPK_KERNEL: &str = "topk_warp_heap_rowwise_kernel";
const BOTTOMK_KERNEL: &str = "bottomk_warp_heap_rowwise_kernel";
const TOP_BITONIC_KERNEL: &str = "topk_warp_bitonic_rowwise_kernel";
const BOTTOM_BITONIC_KERNEL: &str = "bottomk_warp_bitonic_rowwise_kernel";
const MODULE_KERNELS: &[&str] = &[
    TOPK_KERNEL,
    BOTTOMK_KERNEL,
    TOP_BITONIC_KERNEL,
    BOTTOM_BITONIC_KERNEL,
];
const CUDA_SOURCE: &str = include_str!("cuda_topk_rankk.cu");
const WARP_LANES: usize = 32;
const BLOCK_WARPS: usize = 4;
const THREADS_PER_BLOCK: usize = WARP_LANES * BLOCK_WARPS;
const PER_THREAD_KEEP: usize = 8;
const SUPPORTED_K: usize = THREADS_PER_BLOCK * PER_THREAD_KEEP;

static COMPILED_PTX: OnceLock<cudarc::nvrtc::Ptx> = OnceLock::new();
static CUDA_MODULE: OnceLock<CudaModule> = OnceLock::new();

/// Attempt to execute the CUDA kernels for the requested selection.
/// Falls back to the caller when the selection is not implemented on GPU.
pub fn run_selection(
    selection: Selection,
    plan: &RankPlan,
    mut buffers: LaunchSlices<'_>,
) -> Result<(), String> {
    if plan.rows == 0 || plan.k == 0 {
        return Ok(());
    }
    if plan.cols == 0 {
        fill_empty_columns(&mut buffers);
        return Ok(());
    }

    match selection {
        Selection::Top if plan.k == 1 => launch_bitonic_kernel(plan, buffers, TOP_BITONIC_KERNEL),
        Selection::Bottom if plan.k == 1 => {
            launch_bitonic_kernel(plan, buffers, BOTTOM_BITONIC_KERNEL)
        }
        Selection::Top => launch_heap_kernel(plan, buffers, TOPK_KERNEL),
        Selection::Bottom => launch_heap_kernel(plan, buffers, BOTTOMK_KERNEL),
        Selection::Mid => Err("cuda selection not implemented for mid".to_string()),
    }
}

fn launch_heap_kernel(
    plan: &RankPlan,
    mut buffers: LaunchSlices<'_>,
    kernel_name: &'static str,
) -> Result<(), String> {
    launch_cuda_kernel(
        plan,
        buffers,
        kernel_name,
        (THREADS_PER_BLOCK as u32, 1, 1),
        heap_shared_bytes(),
        Some(SUPPORTED_K),
    )
}

fn launch_bitonic_kernel(
    plan: &RankPlan,
    buffers: LaunchSlices<'_>,
    kernel: &'static str,
) -> Result<(), String> {
    launch_cuda_kernel(plan, buffers, kernel, (WARP_LANES as u32, 1, 1), 0, Some(1))
}

fn launch_cuda_kernel(
    plan: &RankPlan,
    mut buffers: LaunchSlices<'_>,
    kernel_name: &'static str,
    block_dim: (u32, u32, u32),
    shared_mem_bytes: u32,
    k_limit: Option<usize>,
) -> Result<(), String> {
    if let Some(limit) = k_limit {
        if plan.k as usize > limit {
            return Err(format!(
                "cuda kernel `{kernel_name}` only supports k ≤ {limit}, received {}",
                plan.k
            ));
        }
    }

    let rows = plan.rows as usize;
    let k = plan.k as usize;

    let module = cuda_module()?;
    let device = module.device();
    let func = module.get_func(kernel_name)?;

    let input = device
        .htod_sync_copy(buffers.input)
        .map_err(|err| err.to_string())?;
    let mut out_vals = device
        .alloc_zeros::<f32>(rows * k)
        .map_err(|err| err.to_string())?;
    let mut out_idx = device
        .alloc_zeros::<i32>(rows * k)
        .map_err(|err| err.to_string())?;

    let grid = grid_for_rows(plan.rows);
    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim,
        shared_mem_bytes,
    };

    unsafe {
        func.launch(
            cfg,
            (
                &input,
                plan.rows as i32,
                plan.cols as i32,
                plan.k as i32,
                &mut out_vals,
                &mut out_idx,
            ),
        )
        .map_err(|err| err.to_string())?;
    }

    let host_vals: Vec<f32> = device
        .dtoh_sync_copy(&out_vals)
        .map_err(|err| err.to_string())?;
    let host_idx: Vec<i32> = device
        .dtoh_sync_copy(&out_idx)
        .map_err(|err| err.to_string())?;

    debug_assert_eq!(host_vals.len(), rows * k);
    debug_assert_eq!(host_idx.len(), rows * k);
    buffers.out_vals.copy_from_slice(&host_vals);
    buffers.out_idx.copy_from_slice(&host_idx);

    Ok(())
}

fn cuda_module() -> Result<&'static CudaModule, String> {
    let ptx =
        COMPILED_PTX.get_or_try_init(|| compile_ptx(CUDA_SOURCE).map_err(|err| err.to_string()))?;
    CUDA_MODULE.get_or_try_init(|| cuda_loader::load_ptx_module(ptx, MODULE_NAME, MODULE_KERNELS))
}

fn fill_empty_columns(buffers: &mut LaunchSlices<'_>) {
    let rows = buffers.rows as usize;
    let k = buffers.k as usize;
    if k == 0 {
        return;
    }
    for row in 0..rows {
        let base = row * k;
        for slot in 0..k {
            buffers.out_vals[base + slot] = f32::NAN;
            buffers.out_idx[base + slot] = -1;
        }
    }
}

fn grid_for_rows(rows: u32) -> (u32, u32, u32) {
    const MAX_GRID_X: u32 = 2_147_483_647;
    const MAX_GRID_YZ: u32 = 65_535;

    if rows == 0 {
        return (1, 1, 1);
    }

    if rows <= MAX_GRID_X {
        return (rows, 1, 1);
    }

    let rows64 = rows as u64;
    let x = MAX_GRID_X;
    let y_needed = (rows64 + x as u64 - 1) / x as u64;
    if y_needed <= MAX_GRID_YZ as u64 {
        return (x, y_needed as u32, 1);
    }

    let y = MAX_GRID_YZ;
    let rows_per_xy = x as u64 * y as u64;
    let z_needed = (rows64 + rows_per_xy - 1) / rows_per_xy;
    let z = z_needed.min(MAX_GRID_YZ as u64).max(1) as u32;
    (x, y, z)
}

fn heap_shared_bytes() -> u32 {
    (THREADS_PER_BLOCK * PER_THREAD_KEEP * std::mem::size_of::<f32>()
        + THREADS_PER_BLOCK * PER_THREAD_KEEP * std::mem::size_of::<i32>()) as u32
}
