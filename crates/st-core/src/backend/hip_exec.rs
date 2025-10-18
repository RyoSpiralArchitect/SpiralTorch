// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::backend::rankk_launch::with_registered_buffers_hip;
use crate::backend::rankk_software::{run_selection, Selection};
use crate::ops::rank_entry::{RankKExecutor, RankPlan};

#[cfg(test)]
use crate::backend::rankk_launch::{with_launch_buffers_hip, LaunchBuffers};

#[derive(Default)]
pub struct HipExecutor;

impl RankKExecutor for HipExecutor {
    type Error = String;
    fn launch_topk(&self, plan: &RankPlan) -> Result<(), Self::Error> {
        dispatch_topk(plan)
    }
    fn launch_midk(&self, plan: &RankPlan) -> Result<(), Self::Error> {
        dispatch_midk(plan)
    }
    fn launch_bottomk(&self, plan: &RankPlan) -> Result<(), Self::Error> {
        dispatch_bottomk(plan)
    }
}

fn is_two_ce(plan: &RankPlan) -> bool {
    let c = &plan.choice;
    // Future: if generated has two_ce_hint, weigh it here.
    c.use_2ce || (plan.cols as u64 >= (c.ctile.max(256) as u64) * 64)
}

fn dispatch_topk(plan: &RankPlan) -> Result<(), String> {
    run_hip_selection(plan, Selection::Top)
}

fn dispatch_midk(plan: &RankPlan) -> Result<(), String> {
    let _ = is_two_ce(plan);
    run_hip_selection(plan, Selection::Mid)
}
fn dispatch_bottomk(plan: &RankPlan) -> Result<(), String> {
    let _ = is_two_ce(plan);
    run_hip_selection(plan, Selection::Bottom)
}

fn run_hip_selection(plan: &RankPlan, selection: Selection) -> Result<(), String> {
    with_registered_buffers_hip(|buffers| run_selection(selection, plan, buffers))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::device_caps::DeviceCaps;
    use crate::ops::rank_entry::{plan_rank, RankKind};

    const ROWS: u32 = 2;
    const COLS: u32 = 5;

    fn plan(kind: RankKind, k: u32) -> RankPlan {
        plan_rank(
            kind,
            ROWS,
            COLS,
            k,
            DeviceCaps::hip(32, 1024, Some(64 * 1024)),
        )
    }

    fn sample_input() -> Vec<f32> {
        vec![
            1.0, 3.5, -2.0, 0.5, 7.0,
            -1.0, 4.0, 0.25, -3.0, 2.0,
        ]
    }

    fn launch_buffers<'a>(
        input: &'a [f32],
        out_vals: &'a mut [f32],
        out_idx: &'a mut [i32],
        k: u32,
    ) -> LaunchBuffers<'a> {
        LaunchBuffers::new(input, ROWS, COLS, k, out_vals, out_idx).expect("valid launch buffers")
    }


    #[test]
    fn hip_topk_selects_largest_values() {
        let plan = plan(RankKind::TopK, 2);
        let input = sample_input();
        let mut out_vals = vec![0.0f32; (ROWS * plan.k) as usize];
        let mut out_idx = vec![0i32; (ROWS * plan.k) as usize];

        with_launch_buffers_hip(launch_buffers(&input, &mut out_vals, &mut out_idx, plan.k), || {
            HipExecutor::default().launch_topk(&plan).unwrap();
        });

        assert_eq!(out_vals, vec![7.0, 3.5, 4.0, 2.0]);
        assert_eq!(out_idx, vec![4, 1, 1, 4]);
    }

    #[test]
    fn hip_midk_selects_central_band() {
        let plan = plan(RankKind::MidK, 2);
        let input = sample_input();
        let mut out_vals = vec![0.0f32; (ROWS * plan.k) as usize];
        let mut out_idx = vec![0i32; (ROWS * plan.k) as usize];

        with_launch_buffers_hip(launch_buffers(&input, &mut out_vals, &mut out_idx, plan.k), || {
            HipExecutor::default().launch_midk(&plan).unwrap();
        });

        assert_eq!(out_vals, vec![0.5, 1.0, -1.0, 0.25]);
        assert_eq!(out_idx, vec![3, 0, 0, 2]);
    }

    #[test]
    fn hip_bottomk_selects_smallest_values() {
        let plan = plan(RankKind::BottomK, 2);
        let input = sample_input();
        let mut out_vals = vec![0.0f32; (ROWS * plan.k) as usize];
        let mut out_idx = vec![0i32; (ROWS * plan.k) as usize];

        with_launch_buffers_hip(launch_buffers(&input, &mut out_vals, &mut out_idx, plan.k), || {
            HipExecutor::default().launch_bottomk(&plan).unwrap();
        });

        assert_eq!(out_vals, vec![-2.0, 0.5, -3.0, -1.0]);
        assert_eq!(out_idx, vec![2, 3, 3, 0]);
    }

    #[test]
    fn hip_errors_without_registered_buffers() {
        let err = HipExecutor::default()
            .launch_topk(&plan(RankKind::TopK, 2))
            .unwrap_err();
        assert!(err.contains("no launch buffers"));
    }
}
