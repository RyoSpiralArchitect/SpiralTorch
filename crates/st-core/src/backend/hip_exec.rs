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
    unsupported_exact_topk(topk_kernel_name(plan))
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

fn unsupported_exact_topk(kernel: &str) -> Result<(), String> {
    Err(format!(
        "hip_exec: exact TopK selection is not implemented; planner chose `{kernel}` but the HIP runtime path is still a stub"
    ))
}

fn topk_kernel_name(plan: &RankPlan) -> &'static str {
    let c = &plan.choice;
    match (c.mk, c.mkd) {
        (2, 4) => "warp_heap",
        (2, 5) => "warp_bitonic",
        (1, 1) => "shared_heap",
        (1, 2) => "shared_kway",
        (0, 3) => "bitonic",
        _ => "default",
    }
}

#[cfg(test)]
mod tests {
    use super::{unsupported_exact_topk, HipExecutor};
    use crate::backend::device_caps::DeviceCaps;
    use crate::ops::rank_entry::{plan_rank, RankKExecutor, RankKind};

    #[test]
    fn exact_topk_is_reported_as_unsupported() {
        let mut exec = HipExecutor::default();
        let plan = plan_rank(
            RankKind::TopK,
            128,
            8_192,
            32,
            DeviceCaps::hip(32, 1_024, Some(64 * 1024)),
        );
        let err = exec.launch_topk(&plan).unwrap_err();
        assert!(err.contains("exact TopK selection is not implemented"));
        assert!(err.contains("planner chose"));
    }

    #[test]
    fn topk_stub_error_mentions_kernel_choice() {
        let err = unsupported_exact_topk("warp_heap").unwrap_err();
        assert!(err.contains("warp_heap"));
        assert!(err.contains("still a stub"));
    }
}
