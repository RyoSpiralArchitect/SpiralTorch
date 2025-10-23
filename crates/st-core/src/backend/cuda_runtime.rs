// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(feature = "cuda")]

use crate::backend::cuda_loader;
use crate::backend::rankk_launch::LaunchSlices;
use crate::backend::rankk_software::Selection;
use crate::ops::rank_entry::RankPlan;
use cudarc::driver::{LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::sync::OnceLock;

const MODULE_NAME: &str = "spiraltorch_rankk";
const TOPK_KERNEL: &str = "topk_warp_heap_rowwise_kernel";
const BOTTOMK_KERNEL: &str = "bottomk_warp_heap_rowwise_kernel";
const BITONIC_KERNEL: &str = "topk_warp_bitonic_rowwise_kernel";
const MODULE_KERNELS: &[&str] = &[TOPK_KERNEL, BOTTOMK_KERNEL, BITONIC_KERNEL];
const CUDA_SOURCE: &str = include_str!("cuda_topk_rankk.cu");
const LANE_COUNT: usize = 32;
const LANE_KEEP: usize = 8;
const SUPPORTED_K: usize = LANE_COUNT * LANE_KEEP;

static COMPILED_PTX: OnceLock<cudarc::nvrtc::Ptx> = OnceLock::new();

/// Attempt to execute the CUDA kernels for the requested selection.
/// Falls back to the caller when the selection is not implemented on GPU.
pub fn run_selection(
    selection: Selection,
    plan: &RankPlan,
    buffers: LaunchSlices<'_>,
) -> Result<(), String> {
    match selection {
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
    if plan.k as usize > SUPPORTED_K {
        return Err(format!(
            "cuda heap kernel only supports k ≤ {SUPPORTED_K}, received {}",
            plan.k
        ));
    }

    let rows = plan.rows as usize;
    let k = plan.k as usize;

    let ptx =
        COMPILED_PTX.get_or_try_init(|| compile_ptx(CUDA_SOURCE).map_err(|err| err.to_string()))?;
    let module = cuda_loader::load_ptx_module(ptx, MODULE_NAME, MODULE_KERNELS)?;
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

    let grid = (1, plan.rows, 1);
    let block = (LANE_COUNT as u32 * 4, 1, 1);
    let shared_bytes = (LANE_COUNT * LANE_KEEP * std::mem::size_of::<f32>()
        + LANE_COUNT * LANE_KEEP * std::mem::size_of::<i32>()) as u32;
    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: block,
        shared_mem_bytes: shared_bytes,
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
