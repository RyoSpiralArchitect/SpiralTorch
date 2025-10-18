// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::ops::rank_entry::{RankKExecutor, RankPlan};

#[derive(Default)]
pub struct CudaExecutor;

impl RankKExecutor for CudaExecutor {
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
    let c = &plan.choice;
    match (c.mk, c.mkd) {
        (2, 4) => topk_warp_heap(plan),
        (2, 5) => topk_warp_bitonic(plan),
        (1, 1) => topk_shared_heap(plan),
        (1, 2) => topk_shared_kway(plan),
        (0, 3) => topk_bitonic(plan),
        _ => topk_default(plan),
    }
}

fn dispatch_midk(plan: &RankPlan) -> Result<(), String> {
    if is_two_ce(plan) {
        midk_two_ce(plan)
    } else {
        midk_one_ce(plan)
    }
}
fn dispatch_bottomk(plan: &RankPlan) -> Result<(), String> {
    if is_two_ce(plan) {
        bottomk_two_ce(plan)
    } else {
        bottomk_one_ce(plan)
    }
}

// ---- CUDA kernels are in cuda_topk_rankk.cu ----
// Below stub calls are ready to be replaced with real CUDA runtime dispatches.

fn stub(plan: &RankPlan, kernel: &str) -> Result<(), String> {
    Err(format!(
        "CUDA rank-k stub invoked: {kernel} for {:?} (rows={}, cols={}, k={}) is not implemented yet.",
        plan.kind, plan.rows, plan.cols, plan.k
    ))
}

fn topk_warp_heap(p: &RankPlan) -> Result<(), String> {
    stub(p, "topk_warp_heap")
}
fn topk_warp_bitonic(p: &RankPlan) -> Result<(), String> {
    stub(p, "topk_warp_bitonic")
}
fn topk_shared_heap(p: &RankPlan) -> Result<(), String> {
    stub(p, "topk_shared_heap")
}
fn topk_shared_kway(p: &RankPlan) -> Result<(), String> {
    stub(p, "topk_shared_kway")
}
fn topk_bitonic(p: &RankPlan) -> Result<(), String> {
    stub(p, "topk_bitonic")
}
fn topk_default(p: &RankPlan) -> Result<(), String> {
    stub(p, "topk_default")
}

fn midk_one_ce(p: &RankPlan) -> Result<(), String> {
    stub(p, "midk_one_ce")
}
fn midk_two_ce(p: &RankPlan) -> Result<(), String> {
    stub(p, "midk_two_ce")
}
fn bottomk_one_ce(p: &RankPlan) -> Result<(), String> {
    stub(p, "bottomk_one_ce")
}
fn bottomk_two_ce(p: &RankPlan) -> Result<(), String> {
    stub(p, "bottomk_two_ce")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::device_caps::DeviceCaps;
    use crate::ops::rank_entry::{plan_rank, RankKind};

    fn plan(kind: RankKind) -> RankPlan {
        plan_rank(kind, 8, 16, 4, DeviceCaps::cuda(32, 1024, Some(64 * 1024)))
    }

    #[test]
    fn cuda_topk_stub_reports_error() {
        let err = topk_default(&plan(RankKind::TopK)).unwrap_err();
        assert!(err.contains("CUDA"));
        assert!(err.contains("topk_default"));
        assert!(err.contains("not implemented"));
    }

    #[test]
    fn cuda_midk_stub_reports_error() {
        let err = midk_one_ce(&plan(RankKind::MidK)).unwrap_err();
        assert!(err.contains("midk_one_ce"));
    }

    #[test]
    fn cuda_bottomk_stub_reports_error() {
        let err = bottomk_two_ce(&plan(RankKind::BottomK)).unwrap_err();
        assert!(err.contains("bottomk_two_ce"));
    }
}
