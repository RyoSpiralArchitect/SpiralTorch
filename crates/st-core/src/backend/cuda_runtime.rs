// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(feature = "cuda")]

use crate::backend::cuda_loader::{self, CudaModule};
use crate::backend::device_caps::BackendKind;
use crate::backend::rankk_launch::LaunchSlices;
use crate::backend::rankk_software::Selection;
use crate::ops::rank_entry::RankPlan;
use cudarc::driver::{LaunchAsync, LaunchConfig};
use std::f32;
use std::sync::OnceLock;

const MODULE_NAME: &str = "spiraltorch_rankk";
const TOPK_KERNEL: &str = "topk_warp_heap_rowwise_kernel";
const BOTTOMK_KERNEL: &str = "bottomk_warp_heap_rowwise_kernel";
const MIDK_KERNEL: &str = "midk_shared_odd_even_rowwise_kernel";
const TOP_BITONIC_KERNEL: &str = "topk_warp_bitonic_rowwise_kernel";
const BOTTOM_BITONIC_KERNEL: &str = "bottomk_warp_bitonic_rowwise_kernel";
const TOP_RESCAN_KERNEL: &str = "topk_exact_rescan_rowwise_kernel";
const BOTTOM_RESCAN_KERNEL: &str = "bottomk_exact_rescan_rowwise_kernel";
const MODULE_KERNELS: &[&str] = &[
    TOPK_KERNEL,
    BOTTOMK_KERNEL,
    MIDK_KERNEL,
    TOP_BITONIC_KERNEL,
    BOTTOM_BITONIC_KERNEL,
    TOP_RESCAN_KERNEL,
    BOTTOM_RESCAN_KERNEL,
];
const CUDA_SOURCE: &str = include_str!("cuda_topk_rankk.cu");
const WARP_LANES: usize = 32;
const PER_THREAD_KEEP: usize = 8;
const SUPPORTED_K: usize = 1024;
const MID_MAX_COLS: u32 = 4096;
const CUDA_RANK_VERIFIED_MAX_WORKGROUP: u32 = 256;
const CUDA_RANK_PORTABLE_DYNAMIC_SHARED_BYTES: u32 = 48 * 1024;

static COMPILED_PTX: OnceLock<cudarc::nvrtc::Ptx> = OnceLock::new();
static CUDA_MODULE: OnceLock<CudaModule> = OnceLock::new();

/// Attempt to execute the CUDA kernels for the requested selection.
pub fn run_selection(
    selection: Selection,
    plan: &RankPlan,
    buffers: &mut LaunchSlices<'_>,
) -> Result<(), String> {
    let workgroup = configured_workgroup_for_selection(plan, selection)?;
    if plan.rows == 0 || plan.k == 0 {
        return Ok(());
    }
    let workgroup = workgroup.unwrap_or(WARP_LANES as u32);
    if plan.cols == 0 {
        fill_empty_columns(buffers);
        return Ok(());
    }

    match selection {
        Selection::Top if plan.k == 1 => launch_bitonic_kernel(plan, buffers, TOP_BITONIC_KERNEL),
        Selection::Bottom if plan.k == 1 => {
            launch_bitonic_kernel(plan, buffers, BOTTOM_BITONIC_KERNEL)
        }
        Selection::Top if needs_exact_rescan(plan.cols, plan.k, workgroup) => {
            launch_rescan_kernel(plan, buffers, TOP_RESCAN_KERNEL, workgroup)
        }
        Selection::Bottom if needs_exact_rescan(plan.cols, plan.k, workgroup) => {
            launch_rescan_kernel(plan, buffers, BOTTOM_RESCAN_KERNEL, workgroup)
        }
        Selection::Top => launch_heap_kernel(plan, buffers, TOPK_KERNEL, workgroup),
        Selection::Bottom => launch_heap_kernel(plan, buffers, BOTTOMK_KERNEL, workgroup),
        Selection::Mid => launch_midk_kernel(plan, buffers, workgroup),
    }
}

fn validate_cuda_plan(plan: &RankPlan) -> Result<(), String> {
    plan.validate()
        .map_err(|error| format!("invalid CUDA rank plan: {error}"))?;
    if plan.device_caps.backend != BackendKind::Cuda {
        return Err(format!(
            "CUDA rank executor requires CUDA device caps, received {}",
            plan.device_caps.backend.as_str()
        ));
    }
    Ok(())
}

fn configured_workgroup_for_selection(
    plan: &RankPlan,
    selection: Selection,
) -> Result<Option<u32>, String> {
    validate_cuda_plan(plan)?;
    if plan.k == 1 && matches!(selection, Selection::Top | Selection::Bottom) {
        return Ok(None);
    }
    let workgroup = plan.choice.wg;
    if workgroup < WARP_LANES as u32 || !workgroup.is_multiple_of(WARP_LANES as u32) {
        return Err(format!(
            "CUDA rank workgroup must be a positive multiple of {WARP_LANES}, received {workgroup}"
        ));
    }
    if workgroup > CUDA_RANK_VERIFIED_MAX_WORKGROUP {
        return Err(format!(
            "CUDA rank workgroup {workgroup} exceeds the verified kernel envelope {CUDA_RANK_VERIFIED_MAX_WORKGROUP}"
        ));
    }
    Ok(Some(workgroup))
}

fn needs_exact_rescan(cols: u32, k: u32, workgroup: u32) -> bool {
    // A thread may own every winner. Retaining eight per thread is exact only
    // when k <= eight or the complete row fits in the retained candidates.
    k as usize > PER_THREAD_KEEP && cols as usize > workgroup as usize * PER_THREAD_KEEP
}

fn launch_rescan_kernel(
    plan: &RankPlan,
    buffers: &mut LaunchSlices<'_>,
    kernel: &'static str,
    workgroup: u32,
) -> Result<(), String> {
    launch_cuda_kernel(
        plan,
        buffers,
        kernel,
        (workgroup, 1, 1),
        rescan_shared_bytes(workgroup)?,
        Some(SUPPORTED_K),
    )
}

fn launch_heap_kernel(
    plan: &RankPlan,
    buffers: &mut LaunchSlices<'_>,
    kernel_name: &'static str,
    workgroup: u32,
) -> Result<(), String> {
    launch_cuda_kernel(
        plan,
        buffers,
        kernel_name,
        (workgroup, 1, 1),
        heap_shared_bytes(workgroup)?,
        Some(workgroup as usize * PER_THREAD_KEEP),
    )
}

fn launch_bitonic_kernel(
    plan: &RankPlan,
    buffers: &mut LaunchSlices<'_>,
    kernel: &'static str,
) -> Result<(), String> {
    launch_cuda_kernel(plan, buffers, kernel, (WARP_LANES as u32, 1, 1), 0, Some(1))
}

fn launch_midk_kernel(
    plan: &RankPlan,
    buffers: &mut LaunchSlices<'_>,
    workgroup: u32,
) -> Result<(), String> {
    validate_mid_cols(plan.cols)?;
    launch_cuda_kernel(
        plan,
        buffers,
        MIDK_KERNEL,
        (workgroup, 1, 1),
        mid_shared_bytes(plan.cols)?,
        None,
    )
}

fn launch_cuda_kernel(
    plan: &RankPlan,
    buffers: &mut LaunchSlices<'_>,
    kernel_name: &'static str,
    block_dim: (u32, u32, u32),
    shared_mem_bytes: u32,
    k_limit: Option<usize>,
) -> Result<(), String> {
    validate_dynamic_shared_memory(plan, kernel_name, shared_mem_bytes)?;
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

fn validate_dynamic_shared_memory(
    plan: &RankPlan,
    kernel_name: &str,
    shared_mem_bytes: u32,
) -> Result<(), String> {
    if shared_mem_bytes > CUDA_RANK_PORTABLE_DYNAMIC_SHARED_BYTES {
        return Err(format!(
            "cuda kernel `{kernel_name}` requires {shared_mem_bytes} shared-memory bytes, exceeding the verified no-opt-in envelope {CUDA_RANK_PORTABLE_DYNAMIC_SHARED_BYTES}"
        ));
    }
    if let Some(limit) = plan.device_caps.shared_mem_per_workgroup {
        if shared_mem_bytes > limit {
            return Err(format!(
                "cuda kernel `{kernel_name}` requires {shared_mem_bytes} shared-memory bytes, exceeding the plan limit {limit}"
            ));
        }
    }
    Ok(())
}

fn cuda_module() -> Result<&'static CudaModule, String> {
    if let Some(module) = CUDA_MODULE.get() {
        return Ok(module);
    }

    if COMPILED_PTX.get().is_none() {
        let compiled = cuda_loader::safe_compile_ptx(CUDA_SOURCE)?;
        let _ = COMPILED_PTX.set(compiled);
    }
    let ptx = COMPILED_PTX
        .get()
        .ok_or_else(|| "failed to initialize CUDA PTX cache".to_string())?;

    let module = cuda_loader::load_ptx_module(ptx, MODULE_NAME, MODULE_KERNELS)?;
    let _ = CUDA_MODULE.set(module);
    CUDA_MODULE
        .get()
        .ok_or_else(|| "failed to initialize CUDA module cache".to_string())
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

fn heap_shared_bytes(workgroup: u32) -> Result<u32, String> {
    selection_shared_bytes(workgroup, PER_THREAD_KEEP, "heap")
}

fn rescan_shared_bytes(workgroup: u32) -> Result<u32, String> {
    selection_shared_bytes(workgroup, 1, "rescan")
}

fn selection_shared_bytes(
    workgroup: u32,
    values_per_thread: usize,
    label: &str,
) -> Result<u32, String> {
    let bytes = (workgroup as usize)
        .checked_mul(values_per_thread)
        .and_then(|slots| {
            slots.checked_mul(std::mem::size_of::<f32>() + std::mem::size_of::<i32>())
        })
        .ok_or_else(|| format!("cuda {label} shared memory size overflow"))?;
    u32::try_from(bytes).map_err(|_| format!("cuda {label} shared memory exceeds u32 limit"))
}

fn mid_shared_bytes(cols: u32) -> Result<u32, String> {
    let cols = cols as usize;
    let bytes = cols
        .checked_mul(std::mem::size_of::<f32>() + std::mem::size_of::<i32>())
        .ok_or_else(|| "cuda midk shared memory size overflow".to_string())?;
    u32::try_from(bytes).map_err(|_| "cuda midk shared memory exceeds u32 limit".to_string())
}

fn validate_mid_cols(cols: u32) -> Result<(), String> {
    if cols > MID_MAX_COLS {
        return Err(format!(
            "cuda midk kernel supports cols ≤ {MID_MAX_COLS}, received {cols}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::device_caps::DeviceCaps;
    use crate::backend::unison_heuristics::RankKind;
    use crate::ops::rank_entry::plan_rank;

    #[test]
    fn concentrated_winners_require_rescan_only_beyond_heap_capacity() {
        assert!(!needs_exact_rescan(2048, 8, 128));
        assert!(!needs_exact_rescan(1024, 1024, 128));
        assert!(needs_exact_rescan(1025, 9, 128));
        assert!(needs_exact_rescan(2048, 16, 128));
        assert!(!needs_exact_rescan(2048, 16, 256));
    }

    #[test]
    fn configured_workgroup_comes_from_the_validated_rank_plan() {
        let plan = plan_rank(
            RankKind::TopK,
            2,
            256,
            8,
            DeviceCaps::cuda(32, 1024, Some(64 * 1024)),
        )
        .try_with_choice_overrides(crate::ops::rank_entry::RankPlanChoiceOverrides {
            workgroup: Some(256),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            configured_workgroup_for_selection(&plan, Selection::Top).unwrap(),
            Some(256)
        );
        assert_eq!(heap_shared_bytes(256).unwrap(), 16 * 1024);
        assert_eq!(rescan_shared_bytes(256).unwrap(), 2 * 1024);
    }

    #[test]
    fn configured_workgroup_rejects_unverified_large_blocks_before_cuda_init() {
        let plan = plan_rank(
            RankKind::TopK,
            2,
            256,
            8,
            DeviceCaps::cuda(32, 1024, Some(64 * 1024)),
        )
        .try_with_choice_overrides(crate::ops::rank_entry::RankPlanChoiceOverrides {
            workgroup: Some(512),
            ..Default::default()
        })
        .unwrap();
        let err = configured_workgroup_for_selection(&plan, Selection::Top)
            .expect_err("512 is outside the tested envelope");
        assert!(err.contains("verified kernel envelope 256"));
    }

    #[test]
    fn fixed_warp_k_one_ignores_the_unconsumed_workgroup() {
        let plan = plan_rank(
            RankKind::TopK,
            2,
            256,
            1,
            DeviceCaps::cuda(32, 1024, Some(64 * 1024)),
        )
        .try_with_choice_overrides(crate::ops::rank_entry::RankPlanChoiceOverrides {
            workgroup: Some(512),
            ..Default::default()
        })
        .unwrap();

        assert_eq!(
            configured_workgroup_for_selection(&plan, Selection::Top).unwrap(),
            None
        );
        assert_eq!(
            configured_workgroup_for_selection(&plan, Selection::Bottom).unwrap(),
            None
        );
        assert!(configured_workgroup_for_selection(&plan, Selection::Mid).is_err());
    }

    #[test]
    fn dynamic_shared_memory_rejects_opt_in_sizes_before_cuda_init() {
        let plan = plan_rank(
            RankKind::TopK,
            2,
            256,
            8,
            DeviceCaps::cuda(32, 1024, Some(64 * 1024)),
        );
        let err = validate_dynamic_shared_memory(
            &plan,
            TOPK_KERNEL,
            CUDA_RANK_PORTABLE_DYNAMIC_SHARED_BYTES + 1,
        )
        .expect_err("opt-in shared memory is not configured");
        assert!(err.contains("verified no-opt-in envelope"));
    }

    #[test]
    fn real_cuda_rank_keeps_concentrated_winners_and_ties() {
        if std::env::var("SPIRALTORCH_RUN_CUDA_RUNTIME_TESTS").as_deref() != Ok("1") {
            return;
        }
        for (kind, selection, sign) in [
            (RankKind::TopK, Selection::Top, 1.0),
            (RankKind::BottomK, Selection::Bottom, -1.0),
        ] {
            for (cols, k) in [(256, 8), (2048, 8), (1025, 9), (2048, 16)] {
                let mut input = vec![0.0; cols];
                let winners = cols.div_ceil(128);
                for i in 0..winners {
                    input[i * 128] = sign * (100.0 + (i / 2) as f32);
                }
                let mut expected: Vec<usize> = (0..cols).collect();
                expected.sort_by(|&a, &b| {
                    (sign * input[b])
                        .total_cmp(&(sign * input[a]))
                        .then(a.cmp(&b))
                });
                let mut values = vec![0.0; k];
                let mut indices = vec![0; k];
                let plan = plan_rank(
                    kind,
                    1,
                    cols as u32,
                    k as u32,
                    DeviceCaps::cuda(32, 1024, None),
                );
                for workgroup in [32, 128, 256] {
                    let plan = plan
                        .try_with_choice_overrides(
                            crate::ops::rank_entry::RankPlanChoiceOverrides {
                                workgroup: Some(workgroup),
                                ..Default::default()
                            },
                        )
                        .unwrap();
                    let mut slices = LaunchSlices {
                        input: &input,
                        out_vals: &mut values,
                        out_idx: &mut indices,
                        rows: 1,
                        cols: cols as u32,
                        k: k as u32,
                    };
                    run_selection(selection, &plan, &mut slices).unwrap();
                    assert_eq!(
                        indices,
                        expected[..k].iter().map(|&i| i as i32).collect::<Vec<_>>(),
                        "workgroup={workgroup}"
                    );
                    assert_eq!(
                        values,
                        expected[..k].iter().map(|&i| input[i]).collect::<Vec<_>>(),
                        "workgroup={workgroup}"
                    );
                }
            }
        }
    }

    fn plan_midk(rows: u32, cols: u32, k: u32) -> RankPlan {
        plan_rank(
            RankKind::MidK,
            rows,
            cols,
            k,
            DeviceCaps::cuda(32, 1024, Some(64 * 1024)),
        )
    }

    #[test]
    fn mid_shared_bytes_matches_expected_layout() {
        let expected =
            MID_MAX_COLS * (std::mem::size_of::<f32>() + std::mem::size_of::<i32>()) as u32;
        assert_eq!(mid_shared_bytes(MID_MAX_COLS).unwrap(), expected);
    }

    #[test]
    fn mid_shared_bytes_rejects_u32_overflow() {
        assert!(mid_shared_bytes(u32::MAX).is_err());
    }

    #[test]
    fn launch_midk_rejects_cols_over_limit_before_cuda_init() {
        let rows = 1u32;
        let cols = MID_MAX_COLS + 1;
        let k = 3u32;
        let plan = plan_midk(rows, cols, k);

        let input = vec![0.0f32; (rows * cols) as usize];
        let mut out_vals = vec![0.0f32; (rows * k) as usize];
        let mut out_idx = vec![0i32; (rows * k) as usize];
        let mut slices = LaunchSlices {
            input: &input,
            out_vals: &mut out_vals,
            out_idx: &mut out_idx,
            rows,
            cols,
            k,
        };
        let err = launch_midk_kernel(&plan, &mut slices, plan.choice.wg)
            .expect_err("cols beyond limit should fail before touching CUDA runtime");

        assert!(err.contains("supports cols"));
        assert!(err.contains("4096"));
    }

    #[test]
    fn validate_mid_cols_allows_exact_limit() {
        assert!(validate_mid_cols(MID_MAX_COLS).is_ok());
    }
}
