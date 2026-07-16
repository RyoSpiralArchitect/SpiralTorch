// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::backend::rankk_launch::{with_registered_buffers_wgpu, LaunchSlices};
use crate::backend::rankk_software::{run_selection, Selection};
use crate::backend::unison_heuristics::RankKind;
use crate::ops::rank_entry::{RankKExecutor, RankPlan};

#[cfg(test)]
use crate::backend::rankk_launch::{with_launch_buffers_wgpu, LaunchBuffers};

#[derive(Default)]
pub struct WgpuExecutor;

const WGPU_RANK_EXACT_SMALL_COLS_LIMIT: u32 = 256;

pub fn wgpu_rank_exact_support(plan: &RankPlan) -> Result<(), String> {
    if plan.choice.use_2ce {
        return st_backend_wgpu::ExactRank2CePlan::try_new(
            exact_2ce_kind(plan.kind),
            plan.rows,
            plan.cols,
            plan.k,
            exact_2ce_tile_cols(plan),
        )
        .map(|_| ())
        .map_err(|error| error.to_string());
    }
    wgpu_rank_exact_support_for(plan.kind, plan.rows, plan.cols, plan.k)
}

pub fn wgpu_rank_exact_support_for(
    kind: RankKind,
    rows: u32,
    cols: u32,
    k: u32,
) -> Result<(), String> {
    if rows == 0 || cols == 0 || k == 0 {
        return Ok(());
    }
    if k > cols {
        return Err(format!(
            "wgpu {} exact path requires k <= cols, got k={k} cols={cols}",
            kind.as_str()
        ));
    }

    match kind {
        RankKind::TopK | RankKind::BottomK => {
            if k > 1 && cols > WGPU_RANK_EXACT_SMALL_COLS_LIMIT {
                return Err(format!(
                    "wgpu {} exact path supports k == 1 for wide rows or cols <= {}, got cols={cols} k={k}",
                    kind.as_str(),
                    WGPU_RANK_EXACT_SMALL_COLS_LIMIT
                ));
            }
        }
        RankKind::MidK => {
            if cols > WGPU_RANK_EXACT_SMALL_COLS_LIMIT {
                return Err(format!(
                    "wgpu midk exact path supports cols <= {}, got cols={cols} k={k}",
                    WGPU_RANK_EXACT_SMALL_COLS_LIMIT
                ));
            }
        }
    }

    Ok(())
}

impl RankKExecutor for WgpuExecutor {
    type Error = String;

    fn launch_topk(&self, plan: &RankPlan) -> Result<(), Self::Error> {
        run_wgpu_selection(plan, Selection::Top)
    }

    fn launch_midk(&self, plan: &RankPlan) -> Result<(), Self::Error> {
        run_wgpu_selection(plan, Selection::Mid)
    }

    fn launch_bottomk(&self, plan: &RankPlan) -> Result<(), Self::Error> {
        run_wgpu_selection(plan, Selection::Bottom)
    }
}

fn run_wgpu_selection(plan: &RankPlan, selection: Selection) -> Result<(), String> {
    with_registered_buffers_wgpu(|buffers| {
        if matches!(selection, Selection::Top) {
            return run_wgpu_topk(plan, buffers);
        }
        if matches!(selection, Selection::Mid) {
            return run_wgpu_midk(plan, buffers);
        }
        if matches!(selection, Selection::Bottom) {
            return run_wgpu_bottomk(plan, buffers);
        }

        if plan.accelerator_fallback().is_strict() {
            return Err(format!(
                "wgpu {} host-buffer bridge is not wired to real WGPU runtime; fallback disabled",
                selection_name(selection)
            ));
        }
        run_selection(selection, plan, buffers)
    })
}

fn run_wgpu_topk(plan: &RankPlan, mut buffers: LaunchSlices<'_>) -> Result<(), String> {
    validate_plan_buffers(plan, &buffers)?;

    if crate::backend::wgpu_rt::installed_ctx().is_some() {
        let real_result = wgpu_rank_exact_support(plan)
            .and_then(|()| {
                if plan.choice.use_2ce {
                    dispatch_exact_2ce(plan, buffers.input)
                } else {
                    let k_lane = plan.choice.kl.max(plan.k).max(1);
                    crate::backend::wgpu_rt::dispatch_topk_host(
                        plan.rows,
                        plan.cols,
                        plan.k,
                        k_lane,
                        buffers.input,
                    )
                }
            })
            .and_then(|(values, indices)| copy_rankk_outputs(&mut buffers, &values, &indices));

        match real_result {
            Ok(()) => return Ok(()),
            Err(err) if plan.accelerator_fallback().is_strict() => {
                return Err(format!(
                    "wgpu topk launch failed ({err}); fallback disabled"
                ));
            }
            Err(err) => {
                return run_selection(Selection::Top, plan, buffers).map_err(|soft_err| {
                    format!("wgpu topk launch failed ({err}); software fallback also failed: {soft_err}")
                });
            }
        }
    }

    if plan.accelerator_fallback().is_strict() {
        return Err("wgpu topk runtime context is not installed; fallback disabled".to_string());
    }
    run_selection(Selection::Top, plan, buffers)
}

fn run_wgpu_midk(plan: &RankPlan, mut buffers: LaunchSlices<'_>) -> Result<(), String> {
    validate_plan_buffers(plan, &buffers)?;

    if crate::backend::wgpu_rt::installed_ctx().is_some() {
        let real_result = wgpu_rank_exact_support(plan)
            .and_then(|()| {
                if plan.choice.use_2ce {
                    dispatch_exact_2ce(plan, buffers.input)
                } else {
                    crate::backend::wgpu_rt::dispatch_midk_host(
                        plan.rows,
                        plan.cols,
                        plan.k,
                        buffers.input,
                    )
                }
            })
            .and_then(|(values, indices)| copy_rankk_outputs(&mut buffers, &values, &indices));

        match real_result {
            Ok(()) => return Ok(()),
            Err(err) if plan.accelerator_fallback().is_strict() => {
                return Err(format!(
                    "wgpu midk launch failed ({err}); fallback disabled"
                ));
            }
            Err(err) => {
                return run_selection(Selection::Mid, plan, buffers).map_err(|soft_err| {
                    format!("wgpu midk launch failed ({err}); software fallback also failed: {soft_err}")
                });
            }
        }
    }

    if plan.accelerator_fallback().is_strict() {
        return Err("wgpu midk runtime context is not installed; fallback disabled".to_string());
    }
    run_selection(Selection::Mid, plan, buffers)
}

fn run_wgpu_bottomk(plan: &RankPlan, mut buffers: LaunchSlices<'_>) -> Result<(), String> {
    validate_plan_buffers(plan, &buffers)?;

    if crate::backend::wgpu_rt::installed_ctx().is_some() {
        let real_result = wgpu_rank_exact_support(plan)
            .and_then(|()| {
                if plan.choice.use_2ce {
                    dispatch_exact_2ce(plan, buffers.input)
                } else {
                    let k_lane = plan.choice.kl.max(plan.k).max(1);
                    crate::backend::wgpu_rt::dispatch_bottomk_host(
                        plan.rows,
                        plan.cols,
                        plan.k,
                        k_lane,
                        buffers.input,
                    )
                }
            })
            .and_then(|(values, indices)| copy_rankk_outputs(&mut buffers, &values, &indices));

        match real_result {
            Ok(()) => return Ok(()),
            Err(err) if plan.accelerator_fallback().is_strict() => {
                return Err(format!(
                    "wgpu bottomk launch failed ({err}); fallback disabled"
                ));
            }
            Err(err) => {
                return run_selection(Selection::Bottom, plan, buffers).map_err(|soft_err| {
                    format!("wgpu bottomk launch failed ({err}); software fallback also failed: {soft_err}")
                });
            }
        }
    }

    if plan.accelerator_fallback().is_strict() {
        return Err("wgpu bottomk runtime context is not installed; fallback disabled".to_string());
    }
    run_selection(Selection::Bottom, plan, buffers)
}

fn dispatch_exact_2ce(plan: &RankPlan, input: &[f32]) -> Result<(Vec<f32>, Vec<i32>), String> {
    crate::backend::wgpu_rt::dispatch_exact_rank_2ce_host(
        exact_2ce_kind(plan.kind),
        plan.rows,
        plan.cols,
        plan.k,
        exact_2ce_tile_cols(plan),
        input,
    )
}

fn exact_2ce_kind(kind: RankKind) -> st_backend_wgpu::ExactRank2CeKind {
    match kind {
        RankKind::TopK => st_backend_wgpu::ExactRank2CeKind::TopK,
        RankKind::MidK => st_backend_wgpu::ExactRank2CeKind::MidK,
        RankKind::BottomK => st_backend_wgpu::ExactRank2CeKind::BottomK,
    }
}

fn exact_2ce_tile_cols(plan: &RankPlan) -> u32 {
    match plan.kind {
        RankKind::TopK => plan.choice.tile,
        RankKind::MidK | RankKind::BottomK => plan.choice.ctile,
    }
}

fn copy_rankk_outputs(
    buffers: &mut LaunchSlices<'_>,
    values: &[f32],
    indices: &[i32],
) -> Result<(), String> {
    if values.len() != buffers.out_vals.len() {
        return Err(format!(
            "wgpu rank-k returned {} values for output buffer length {}",
            values.len(),
            buffers.out_vals.len()
        ));
    }
    if indices.len() != buffers.out_idx.len() {
        return Err(format!(
            "wgpu rank-k returned {} indices for output buffer length {}",
            indices.len(),
            buffers.out_idx.len()
        ));
    }
    buffers.out_vals.copy_from_slice(values);
    buffers.out_idx.copy_from_slice(indices);
    Ok(())
}

fn validate_plan_buffers(plan: &RankPlan, buffers: &LaunchSlices<'_>) -> Result<(), String> {
    if plan.rows != buffers.rows {
        return Err(format!(
            "plan rows {} did not match buffer rows {}",
            plan.rows, buffers.rows
        ));
    }
    if plan.cols != buffers.cols {
        return Err(format!(
            "plan cols {} did not match buffer cols {}",
            plan.cols, buffers.cols
        ));
    }
    if plan.k != buffers.k {
        return Err(format!(
            "plan k {} did not match buffer k {}",
            plan.k, buffers.k
        ));
    }
    Ok(())
}

fn selection_name(selection: Selection) -> &'static str {
    match selection {
        Selection::Top => "topk",
        Selection::Mid => "midk",
        Selection::Bottom => "bottomk",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::device_caps::DeviceCaps;
    use crate::backend::execution_plan::{AcceleratorFallback, ExecutionConfig};
    use crate::ops::rank_entry::{plan_rank, plan_rank_with_config};

    const ROWS: u32 = 2;
    const COLS: u32 = 5;

    fn plan(kind: RankKind, k: u32) -> RankPlan {
        plan_for(kind, ROWS, COLS, k)
    }

    fn plan_for(kind: RankKind, rows: u32, cols: u32, k: u32) -> RankPlan {
        plan_rank(kind, rows, cols, k, DeviceCaps::wgpu(32, true, 256))
    }

    fn strict_plan(kind: RankKind, rows: u32, cols: u32, k: u32) -> RankPlan {
        plan_rank_with_config(
            kind,
            rows,
            cols,
            k,
            DeviceCaps::wgpu(32, true, 256),
            ExecutionConfig::new(AcceleratorFallback::Forbid, 1024),
        )
    }

    fn sample_input() -> Vec<f32> {
        vec![
            1.0, 3.5, -2.0, 0.5, 7.0, // row 0
            -1.0, 4.0, 0.25, -3.0, 2.0, // row 1
        ]
    }

    fn launch_buffers<'a>(
        input: &'a [f32],
        out_vals: &'a mut [f32],
        out_idx: &'a mut [i32],
        k: u32,
    ) -> LaunchBuffers<'a> {
        launch_buffers_for(input, ROWS, COLS, out_vals, out_idx, k)
    }

    fn launch_buffers_for<'a>(
        input: &'a [f32],
        rows: u32,
        cols: u32,
        out_vals: &'a mut [f32],
        out_idx: &'a mut [i32],
        k: u32,
    ) -> LaunchBuffers<'a> {
        LaunchBuffers::new(input, rows, cols, k, out_vals, out_idx).expect("valid launch buffers")
    }

    fn software_reference(
        selection: Selection,
        kind: RankKind,
        input: &[f32],
        rows: u32,
        cols: u32,
        k: u32,
    ) -> (Vec<f32>, Vec<i32>) {
        let plan = plan_for(kind, rows, cols, k);
        let mut out_vals = vec![0.0f32; (rows * k) as usize];
        let mut out_idx = vec![0i32; (rows * k) as usize];
        run_selection(
            selection,
            &plan,
            LaunchSlices {
                input,
                out_vals: &mut out_vals,
                out_idx: &mut out_idx,
                rows,
                cols,
                k,
            },
        )
        .expect("software rank-k reference should succeed");
        (out_vals, out_idx)
    }

    fn wgpu_runtime_tests_enabled() -> bool {
        std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS")
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE"))
            .unwrap_or(false)
    }

    fn ensure_wgpu_runtime_ctx_for_test() -> Result<(), String> {
        crate::backend::wgpu_rt::ensure_default_ctx().map(|_| ())
    }

    #[test]
    fn wgpu_rank_executor_uses_software_reference_when_non_strict() {
        let plan = plan(RankKind::TopK, 2);
        let input = sample_input();
        let mut out_vals = vec![0.0f32; (ROWS * plan.k) as usize];
        let mut out_idx = vec![0i32; (ROWS * plan.k) as usize];

        with_launch_buffers_wgpu(
            launch_buffers(&input, &mut out_vals, &mut out_idx, plan.k),
            || {
                WgpuExecutor.launch_topk(&plan).unwrap();
            },
        );

        assert_eq!(out_vals, vec![7.0, 3.5, 4.0, 2.0]);
        assert_eq!(out_idx, vec![4, 1, 1, 4]);
    }

    #[test]
    fn wgpu_rank_executor_rejects_fallback_when_strict() {
        if crate::backend::wgpu_rt::installed_ctx().is_some() {
            return;
        }

        let plan = strict_plan(RankKind::MidK, ROWS, COLS, 2);
        let input = sample_input();
        let mut out_vals = vec![0.0f32; (ROWS * plan.k) as usize];
        let mut out_idx = vec![0i32; (ROWS * plan.k) as usize];

        let err = with_launch_buffers_wgpu(
            launch_buffers(&input, &mut out_vals, &mut out_idx, plan.k),
            || WgpuExecutor.launch_midk(&plan),
        )
        .expect_err("strict mode should reject software fallback");

        assert!(err.contains("fallback disabled"));
    }

    #[test]
    fn wgpu_topk_strict_requires_runtime_context() {
        if crate::backend::wgpu_rt::installed_ctx().is_some() {
            return;
        }

        let plan = strict_plan(RankKind::TopK, ROWS, COLS, 2);
        let input = sample_input();
        let mut out_vals = vec![0.0f32; (ROWS * plan.k) as usize];
        let mut out_idx = vec![0i32; (ROWS * plan.k) as usize];

        let err = with_launch_buffers_wgpu(
            launch_buffers(&input, &mut out_vals, &mut out_idx, plan.k),
            || WgpuExecutor.launch_topk(&plan),
        )
        .expect_err("strict mode should require an installed WGPU runtime context");

        assert!(err.contains("runtime context is not installed"));
        assert!(err.contains("fallback disabled"));
    }

    #[test]
    fn wgpu_topk_host_rejects_known_inexact_shapes_before_context() {
        let wide_input = vec![0.0f32; 257];
        let wide_err =
            crate::backend::wgpu_rt::dispatch_topk_host(1, 257, 2, 2, &wide_input).unwrap_err();
        assert!(wide_err.contains("cols <= 256"));

        let short_input = vec![0.0f32; 4];
        let short_err =
            crate::backend::wgpu_rt::dispatch_topk_host(1, 4, 5, 5, &short_input).unwrap_err();
        assert!(short_err.contains("k <= cols"));

        let bottom_err =
            crate::backend::wgpu_rt::dispatch_bottomk_host(1, 257, 2, 2, &wide_input).unwrap_err();
        assert!(bottom_err.contains("cols <= 256"));

        let mid_err =
            crate::backend::wgpu_rt::dispatch_midk_host(1, 257, 2, &wide_input).unwrap_err();
        assert!(mid_err.contains("cols <= 256"));
    }

    #[test]
    fn wgpu_rank_exact_support_describes_current_real_path() {
        assert!(wgpu_rank_exact_support(&plan_for(RankKind::TopK, 2, 8, 4)).is_ok());
        assert!(wgpu_rank_exact_support_for(RankKind::TopK, 1, 4096, 1).is_ok());
        assert!(wgpu_rank_exact_support_for(RankKind::BottomK, 1, 4096, 1).is_ok());
        assert!(wgpu_rank_exact_support_for(RankKind::MidK, 0, 4096, 128).is_ok());

        let wide_top = wgpu_rank_exact_support_for(RankKind::TopK, 1, 257, 2).unwrap_err();
        assert!(wide_top.contains("k == 1"));
        assert!(wide_top.contains("cols <= 256"));

        let wide_bottom = wgpu_rank_exact_support_for(RankKind::BottomK, 1, 257, 2).unwrap_err();
        assert!(wide_bottom.contains("k == 1"));
        assert!(wide_bottom.contains("cols <= 256"));

        let wide_mid = wgpu_rank_exact_support_for(RankKind::MidK, 1, 257, 2).unwrap_err();
        assert!(wide_mid.contains("midk"));
        assert!(wide_mid.contains("cols <= 256"));

        let too_large_k = wgpu_rank_exact_support_for(RankKind::TopK, 1, 4, 5).unwrap_err();
        assert!(too_large_k.contains("k <= cols"));

        for kind in [RankKind::TopK, RankKind::MidK, RankKind::BottomK] {
            let mut wide = plan_for(kind, 2, 513, 7);
            wide.choice.use_2ce = true;
            wide.choice.tile = 128;
            wide.choice.ctile = 128;
            assert!(
                wgpu_rank_exact_support(&wide).is_ok(),
                "{} should support wide exact 2CE execution",
                kind.as_str()
            );
        }
    }

    #[test]
    fn wgpu_topk_runtime_matches_software_reference_when_enabled() {
        if !wgpu_runtime_tests_enabled() {
            return;
        }
        if let Err(err) = ensure_wgpu_runtime_ctx_for_test() {
            eprintln!("skipping WGPU runtime parity test: {err}");
            return;
        }

        let plan = strict_plan(RankKind::TopK, ROWS, COLS, 2);
        let input = sample_input();
        let mut out_vals = vec![0.0f32; (ROWS * plan.k) as usize];
        let mut out_idx = vec![0i32; (ROWS * plan.k) as usize];

        with_launch_buffers_wgpu(
            launch_buffers(&input, &mut out_vals, &mut out_idx, plan.k),
            || WgpuExecutor.launch_topk(&plan),
        )
        .expect("strict WGPU TopK should use the installed runtime context");

        assert_eq!(out_vals, vec![7.0, 3.5, 4.0, 2.0]);
        assert_eq!(out_idx, vec![4, 1, 1, 4]);
    }

    #[test]
    fn wgpu_bottomk_runtime_matches_software_reference_when_enabled() {
        if !wgpu_runtime_tests_enabled() {
            return;
        }
        if let Err(err) = ensure_wgpu_runtime_ctx_for_test() {
            eprintln!("skipping WGPU runtime parity test: {err}");
            return;
        }

        let plan = strict_plan(RankKind::BottomK, ROWS, COLS, 2);
        let input = sample_input();
        let mut out_vals = vec![0.0f32; (ROWS * plan.k) as usize];
        let mut out_idx = vec![0i32; (ROWS * plan.k) as usize];

        with_launch_buffers_wgpu(
            launch_buffers(&input, &mut out_vals, &mut out_idx, plan.k),
            || WgpuExecutor.launch_bottomk(&plan),
        )
        .expect("strict WGPU BottomK should use the installed runtime context");

        assert_eq!(out_vals, vec![-2.0, 0.5, -3.0, -1.0]);
        assert_eq!(out_idx, vec![2, 3, 3, 0]);
    }

    #[test]
    fn wgpu_midk_runtime_matches_software_reference_when_enabled() {
        if !wgpu_runtime_tests_enabled() {
            return;
        }
        if let Err(err) = ensure_wgpu_runtime_ctx_for_test() {
            eprintln!("skipping WGPU runtime parity test: {err}");
            return;
        }

        let plan = strict_plan(RankKind::MidK, ROWS, COLS, 2);
        let input = sample_input();
        let mut out_vals = vec![0.0f32; (ROWS * plan.k) as usize];
        let mut out_idx = vec![0i32; (ROWS * plan.k) as usize];

        with_launch_buffers_wgpu(
            launch_buffers(&input, &mut out_vals, &mut out_idx, plan.k),
            || WgpuExecutor.launch_midk(&plan),
        )
        .expect("strict WGPU MidK should use the installed runtime context");

        assert_eq!(out_vals, vec![0.5, 1.0, -1.0, 0.25]);
        assert_eq!(out_idx, vec![3, 0, 0, 2]);
    }

    #[test]
    fn wgpu_rank_runtime_matches_reference_for_wider_small_rows_when_enabled() {
        if !wgpu_runtime_tests_enabled() {
            return;
        }
        if let Err(err) = ensure_wgpu_runtime_ctx_for_test() {
            eprintln!("skipping WGPU runtime parity test: {err}");
            return;
        }

        let rows = 2;
        let cols = 8;
        let k = 4;
        let input = vec![
            1.0, -3.0, 5.5, 0.0, 2.25, -1.5, 4.0, 8.0, // row 0
            6.0, -2.0, 3.0, 7.5, -4.0, 0.5, 9.0, 1.25, // row 1
        ];

        for (selection, kind, launch) in [
            (
                Selection::Top,
                RankKind::TopK,
                WgpuExecutor::launch_topk as fn(&WgpuExecutor, &RankPlan) -> Result<(), String>,
            ),
            (Selection::Mid, RankKind::MidK, WgpuExecutor::launch_midk),
            (
                Selection::Bottom,
                RankKind::BottomK,
                WgpuExecutor::launch_bottomk,
            ),
        ] {
            let plan = strict_plan(kind, rows, cols, k);
            let (expected_vals, expected_idx) =
                software_reference(selection, kind, &input, rows, cols, k);
            let mut out_vals = vec![0.0f32; (rows * k) as usize];
            let mut out_idx = vec![0i32; (rows * k) as usize];

            with_launch_buffers_wgpu(
                launch_buffers_for(&input, rows, cols, &mut out_vals, &mut out_idx, k),
                || launch(&WgpuExecutor, &plan),
            )
            .expect("strict WGPU rank-k should use the installed runtime context");

            assert_eq!(out_vals, expected_vals);
            assert_eq!(out_idx, expected_idx);
        }
    }

    #[test]
    fn wgpu_rank_2ce_runtime_matches_wide_reference_when_enabled() {
        if !wgpu_runtime_tests_enabled() {
            return;
        }
        if let Err(err) = ensure_wgpu_runtime_ctx_for_test() {
            eprintln!("skipping WGPU 2CE runtime parity test: {err}");
            return;
        }

        let rows = 2;
        let cols = 513;
        let k = 7;
        let mut input = (0..rows * cols)
            .map(|index| ((index * 37 % 101) as f32 - 50.0) / 7.0)
            .collect::<Vec<_>>();
        input[0] = f32::NAN;
        input[1] = f32::INFINITY;
        input[2] = -0.0;
        input[3] = 0.0;
        input[129] = 9.25;
        input[385] = 9.25;
        for value in &mut input[cols as usize..] {
            *value = f32::NAN;
        }
        input[cols as usize + 2] = -0.0;
        input[cols as usize + 257] = 0.0;
        input[cols as usize + 512] = -4.0;

        for (selection, kind, launch) in [
            (
                Selection::Top,
                RankKind::TopK,
                WgpuExecutor::launch_topk as fn(&WgpuExecutor, &RankPlan) -> Result<(), String>,
            ),
            (Selection::Mid, RankKind::MidK, WgpuExecutor::launch_midk),
            (
                Selection::Bottom,
                RankKind::BottomK,
                WgpuExecutor::launch_bottomk,
            ),
        ] {
            let mut plan = strict_plan(kind, rows, cols, k);
            plan.choice.use_2ce = true;
            plan.choice.tile = 128;
            plan.choice.ctile = 128;
            let (expected_values, expected_indices) =
                software_reference(selection, kind, &input, rows, cols, k);
            let mut actual_values = vec![0.0; (rows * k) as usize];
            let mut actual_indices = vec![0; (rows * k) as usize];

            with_launch_buffers_wgpu(
                launch_buffers_for(
                    &input,
                    rows,
                    cols,
                    &mut actual_values,
                    &mut actual_indices,
                    k,
                ),
                || launch(&WgpuExecutor, &plan),
            )
            .unwrap_or_else(|error| panic!("{} exact 2CE failed: {error}", kind.as_str()));

            assert_eq!(
                actual_indices,
                expected_indices,
                "{} indices",
                kind.as_str()
            );
            for (slot, (actual, expected)) in
                actual_values.iter().zip(expected_values.iter()).enumerate()
            {
                assert!(
                    (actual.is_nan() && expected.is_nan())
                        || actual.to_bits() == expected.to_bits(),
                    "{} value slot {slot}: actual={actual:?} expected={expected:?}",
                    kind.as_str()
                );
            }
        }
    }

    #[test]
    fn wgpu_rank_executor_errors_without_registered_buffers() {
        let err = WgpuExecutor
            .launch_topk(&plan(RankKind::TopK, 2))
            .unwrap_err();
        assert!(err.contains("no launch buffers"));
    }
}
