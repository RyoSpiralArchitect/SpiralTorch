use crate::ops::rank_entry::{RankKExecutor, RankPlan};

#[derive(Default)]
pub struct HipExecutor;

impl RankKExecutor for HipExecutor {
    type Error = String;
    fn launch_topk(&mut self, plan: &RankPlan) -> Result<(), Self::Error> {
        dispatch_topk(plan)
    }
    fn launch_midk(&mut self, plan: &RankPlan) -> Result<(), Self::Error> {
        dispatch_midk(plan)
    }
    fn launch_bottomk(&mut self, plan: &RankPlan) -> Result<(), Self::Error> {
        dispatch_bottomk(plan)
    }
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

fn dispatch_midk(_plan: &RankPlan) -> Result<(), String> {
    unsupported_exact_rank("MidK")
}
fn dispatch_bottomk(_plan: &RankPlan) -> Result<(), String> {
    unsupported_exact_rank("BottomK")
}

fn unsupported_exact_rank(kind: &str) -> Result<(), String> {
    Err(format!(
        "hip_exec: exact {kind} selection is not implemented; use `ops::compaction::plan_compaction(...)` plus a backend compaction runtime for threshold compaction"
    ))
}

// ---- HIP kernels are in hip_topk_rankk.hip.cpp ----
// Below stub calls are ready to be replaced with real WGPU runtime dispatches.
fn topk_warp_heap(_p: &RankPlan) -> Result<(), String> {
    Ok(())
}
fn topk_warp_bitonic(_p: &RankPlan) -> Result<(), String> {
    Ok(())
}
fn topk_shared_heap(_p: &RankPlan) -> Result<(), String> {
    Ok(())
}
fn topk_shared_kway(_p: &RankPlan) -> Result<(), String> {
    Ok(())
}
fn topk_bitonic(_p: &RankPlan) -> Result<(), String> {
    Ok(())
}
fn topk_default(_p: &RankPlan) -> Result<(), String> {
    Ok(())
}
