use crate::ops::rank_entry::{RankKExecutor, RankPlan};
#[derive(Default)]
pub struct WgpuExecutor;

impl RankKExecutor for WgpuExecutor {
    type Error = String;
    fn launch_topk(&self, plan:&RankPlan) -> Result<(), Self::Error> {
        dispatch_topk(plan)
    }
    fn launch_midk(&self, plan:&RankPlan) -> Result<(), Self::Error> {
        dispatch_midk(plan)
    }
    fn launch_bottomk(&self, plan:&RankPlan) -> Result<(), Self::Error> {
        dispatch_bottomk(plan)
    }
}

fn is_two_ce(plan:&RankPlan) -> bool {
    let c = &plan.choice;
    c.use_2ce || (plan.cols as u64 >= (c.ctile.max(256) as u64)*64)
}

// ---- Strategy dispatchers (mk/mkd aware) ----
fn dispatch_topk(plan:&RankPlan) -> Result<(), String> {
    let c = &plan.choice;
    match (c.mk, c.mkd) {
        (2, 4) => topk_warp_heap(plan),
        (2, 5) => topk_warp_bitonic(plan),
        (1, 1) => topk_shared_heap(plan),
        (1, 2) => topk_shared_kway(plan),
        (0, 3) => topk_bitonic(plan),
        (m, d) => { let _ = (m,d); topk_default(plan) }
    }
}
fn dispatch_midk(plan:&RankPlan) -> Result<(), String> {
    if is_two_ce(plan) { midk_two_ce(plan) } else { midk_one_ce(plan) }
}
fn dispatch_bottomk(plan:&RankPlan) -> Result<(), String> {
    if is_two_ce(plan) { bottomk_two_ce(plan) } else { bottomk_one_ce(plan) }
}

// ---- Kernel entry stubs (replace with real backend kernels) ----
fn topk_warp_heap(_p:&RankPlan)->Result<(),String>{ Ok(()) }
fn topk_warp_bitonic(_p:&RankPlan)->Result<(),String>{ Ok(()) }
fn topk_shared_heap(_p:&RankPlan)->Result<(),String>{ Ok(()) }
fn topk_shared_kway(_p:&RankPlan)->Result<(),String>{ Ok(()) }
fn topk_bitonic(_p:&RankPlan)->Result<(),String>{ Ok(()) }
fn topk_default(_p:&RankPlan)->Result<(),String>{ Ok(()) }

fn midk_one_ce(_p:&RankPlan)->Result<(),String>{ Ok(()) }
fn midk_two_ce(_p:&RankPlan)->Result<(),String>{ Ok(()) }
fn bottomk_one_ce(_p:&RankPlan)->Result<(),String>{ Ok(()) }
fn bottomk_two_ce(_p:&RankPlan)->Result<(),String>{ Ok(()) }
