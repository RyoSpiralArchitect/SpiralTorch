use crate::ops::rank_entry::{RankKExecutor, RankPlan};

#[derive(Default)]
pub struct WgpuExecutor;

impl RankKExecutor for WgpuExecutor {
    type Error = String;
    fn launch_topk(&self, plan:&RankPlan) -> Result<(), Self::Error> { dispatch_topk(plan) }
    fn launch_midk(&self, plan:&RankPlan) -> Result<(), Self::Error> { dispatch_midk(plan) }
    fn launch_bottomk(&self, plan:&RankPlan) -> Result<(), Self::Error> { dispatch_bottomk(plan) }
}

fn two_ce_decision(plan:&RankPlan) -> bool {
    let c = &plan.choice;
    if c.two_ce_hint { return true; }
    if c.use_2ce { return true; }
    let tiles = (plan.cols as u64 + c.ctile.max(256) as u64 - 1) / c.ctile.max(256) as u64;
    tiles >= 64
}

fn dispatch_topk(plan:&RankPlan) -> Result<(), String> { topk_1ce(plan) }
fn dispatch_midk(plan:&RankPlan) -> Result<(), String> {
    if two_ce_decision(plan) { midk_two_ce(plan) } else { midk_one_ce(plan) }
}
fn dispatch_bottomk(plan:&RankPlan) -> Result<(), String> {
    if two_ce_decision(plan) { bottomk_two_ce(plan) } else { bottomk_one_ce(plan) }
}

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn topk_1ce(p:&RankPlan)->Result<(),String>{ crate::backend::wgpu_rt::dispatch_topk_1ce(p) }
#[cfg(not(all(feature="wgpu", feature="wgpu-rt")))]
fn topk_1ce(_p:&RankPlan)->Result<(),String>{ Err("wgpu-rt not enabled".into()) }

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn midk_one_ce(p:&RankPlan)->Result<(),String>{ crate::backend::wgpu_rt::dispatch_compaction_1ce(p, 0) }
#[cfg(not(all(feature="wgpu", feature="wgpu-rt")))]
fn midk_one_ce(_:&RankPlan)->Result<(),String>{ Err("wgpu-rt not enabled".into()) }

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn midk_two_ce(p:&RankPlan)->Result<(),String>{ crate::backend::wgpu_rt::dispatch_compaction_2ce(p, 0) }
#[cfg(not(all(feature="wgpu", feature="wgpu-rt")))]
fn midk_two_ce(_:&RankPlan)->Result<(),String>{ Err("wgpu-rt not enabled".into()) }

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn bottomk_one_ce(p:&RankPlan)->Result<(),String>{ crate::backend::wgpu_rt::dispatch_compaction_1ce(p, 1) }
#[cfg(not(all(feature="wgpu", feature="wgpu-rt")))]
fn bottomk_one_ce(_:&RankPlan)->Result<(),String>{ Err("wgpu-rt not enabled".into()) }

#[cfg(all(feature="wgpu", feature="wgpu-rt"))]
fn bottomk_two_ce(p:&RankPlan)->Result<(),String>{ crate::backend::wgpu_rt::dispatch_compaction_2ce(p, 1) }
#[cfg(not(all(feature="wgpu", feature="wgpu-rt")))]
fn bottomk_two_ce(_:&RankPlan)->Result<(),String>{ Err("wgpu-rt not enabled".into()) }
