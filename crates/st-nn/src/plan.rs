// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use st_core::backend::device_caps::DeviceCaps;
use st_core::backend::execution_plan::{BackendPolicy, ExecutionConfig};
use st_core::backend::unison_heuristics::RankKind;
use st_core::ops::rank_entry::{plan_rank_with_policy, RankPlan};

/// Convenience wrapper that keeps SpiralK heuristics close to high level modules.
#[derive(Clone, Copy, Debug)]
pub struct RankPlanner {
    backend_policy: BackendPolicy,
}

impl RankPlanner {
    /// Builds a planner with the provided device capabilities.
    pub fn new(caps: DeviceCaps) -> Self {
        Self::from_backend_policy(BackendPolicy::from_device_caps(caps))
    }

    /// Builds a deterministic planner from an explicit execution contract.
    pub fn with_execution_config(caps: DeviceCaps, execution_config: ExecutionConfig) -> Self {
        Self::from_backend_policy(BackendPolicy::from_device_caps_with_config(
            caps,
            execution_config,
        ))
    }

    /// Builds a planner from the complete Rust-owned execution policy.
    pub const fn from_backend_policy(backend_policy: BackendPolicy) -> Self {
        Self { backend_policy }
    }

    /// Returns the backing capability descriptor.
    pub fn device_caps(&self) -> DeviceCaps {
        self.backend_policy.device_caps()
    }

    /// Returns the execution contract captured when the planner was built.
    pub const fn execution_config(&self) -> ExecutionConfig {
        self.backend_policy.execution_config()
    }

    /// Returns the complete policy captured when the planner was built.
    pub const fn backend_policy(&self) -> BackendPolicy {
        self.backend_policy
    }

    /// Returns the committed runtime plan, when this planner was plan-bound.
    pub fn runtime_execution_plan_output_sha256(&self) -> Option<String> {
        self.backend_policy.runtime_plan_output_sha256_hex()
    }

    /// Plans any supported rank selection through this captured execution context.
    pub fn plan(&self, kind: RankKind, rows: u32, cols: u32, k: u32) -> RankPlan {
        plan_rank_with_policy(kind, rows, cols, k, self.backend_policy)
    }

    /// Plans a TopK execution using SpiralK and the unified heuristics.
    pub fn topk(&self, rows: u32, cols: u32, k: u32) -> RankPlan {
        self.plan(RankKind::TopK, rows, cols, k)
    }

    /// Plans a MidK execution.
    pub fn midk(&self, rows: u32, cols: u32, k: u32) -> RankPlan {
        self.plan(RankKind::MidK, rows, cols, k)
    }

    /// Plans a BottomK execution.
    pub fn bottomk(&self, rows: u32, cols: u32, k: u32) -> RankPlan {
        self.plan(RankKind::BottomK, rows, cols, k)
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
