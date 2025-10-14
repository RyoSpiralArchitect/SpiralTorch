use st_core::backend::device_caps::DeviceCaps;
use st_core::backend::unison_heuristics::RankKind;
use st_core::ops::rank_entry::{plan_rank, RankPlan};

/// Convenience wrapper that keeps SpiralK heuristics close to high level modules.
#[derive(Clone, Copy, Debug)]
pub struct RankPlanner {
    caps: DeviceCaps,
}

impl RankPlanner {
    /// Builds a planner with the provided device capabilities.
    pub fn new(caps: DeviceCaps) -> Self {
        Self { caps }
    }

    /// Returns the backing capability descriptor.
    pub fn device_caps(&self) -> DeviceCaps {
        self.caps
    }

    /// Plans a TopK execution using SpiralK and the unified heuristics.
    pub fn topk(&self, rows: u32, cols: u32, k: u32) -> RankPlan {
        plan_rank(RankKind::TopK, rows, cols, k, self.caps)
    }

    /// Plans a MidK execution.
    pub fn midk(&self, rows: u32, cols: u32, k: u32) -> RankPlan {
        plan_rank(RankKind::MidK, rows, cols, k, self.caps)
    }

    /// Plans a BottomK execution.
    pub fn bottomk(&self, rows: u32, cols: u32, k: u32) -> RankPlan {
        plan_rank(RankKind::BottomK, rows, cols, k, self.caps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_round_trips_caps() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let planner = RankPlanner::new(caps);
        assert_eq!(planner.device_caps(), caps);
        let plan = planner.topk(64, 1024, 16);
        assert_eq!(plan.rows, 64);
        assert_eq!(plan.cols, 1024);
        assert_eq!(plan.k, 16);
    }
}
