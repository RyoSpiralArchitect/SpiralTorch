// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use st_core::backend::device_caps::DeviceCaps;
use st_core::backend::execution_plan::ExecutionConfig;
use st_core::backend::unison_heuristics::RankKind;
use st_core::ops::rank_entry::{plan_rank_with_config, RankPlan};

/// Convenience wrapper that keeps SpiralK heuristics close to high level modules.
#[derive(Clone, Copy, Debug)]
pub struct RankPlanner {
    caps: DeviceCaps,
    execution_config: ExecutionConfig,
}

impl RankPlanner {
    /// Builds a planner with the provided device capabilities.
    pub fn new(caps: DeviceCaps) -> Self {
        Self::with_execution_config(caps, ExecutionConfig::from_env())
    }

    /// Builds a deterministic planner from an explicit execution contract.
    pub const fn with_execution_config(
        caps: DeviceCaps,
        execution_config: ExecutionConfig,
    ) -> Self {
        Self {
            caps,
            execution_config,
        }
    }

    /// Returns the backing capability descriptor.
    pub fn device_caps(&self) -> DeviceCaps {
        self.caps
    }

    /// Returns the execution contract captured when the planner was built.
    pub const fn execution_config(&self) -> ExecutionConfig {
        self.execution_config
    }

    /// Plans a TopK execution using SpiralK and the unified heuristics.
    pub fn topk(&self, rows: u32, cols: u32, k: u32) -> RankPlan {
        plan_rank_with_config(
            RankKind::TopK,
            rows,
            cols,
            k,
            self.caps,
            self.execution_config,
        )
    }

    /// Plans a MidK execution.
    pub fn midk(&self, rows: u32, cols: u32, k: u32) -> RankPlan {
        plan_rank_with_config(
            RankKind::MidK,
            rows,
            cols,
            k,
            self.caps,
            self.execution_config,
        )
    }

    /// Plans a BottomK execution.
    pub fn bottomk(&self, rows: u32, cols: u32, k: u32) -> RankPlan {
        plan_rank_with_config(
            RankKind::BottomK,
            rows,
            cols,
            k,
            self.caps,
            self.execution_config,
        )
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

    #[test]
    fn explicit_config_is_shared_by_every_rank_plan() {
        use st_core::backend::execution_plan::AcceleratorFallback;

        let config = ExecutionConfig::new(AcceleratorFallback::Forbid, 4096);
        let planner = RankPlanner::with_execution_config(DeviceCaps::cpu(), config);

        for plan in [
            planner.topk(2, 8, 2),
            planner.midk(2, 8, 2),
            planner.bottomk(2, 8, 2),
        ] {
            assert_eq!(plan.execution_config, config);
        }
    }
}
