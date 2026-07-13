// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Standard entry for TopK/MidK/BottomK across all backends.
//! This module plans the run (heuristics) and exposes a single surface the backends can implement.
//
//! Typical flow:
//!   let plan = plan_rank(RankKind::TopK, rows, cols, k, caps);
//!   execute_rank(&plan, backend_impl, tensors);
use crate::backend::device_caps::DeviceCaps;
use crate::backend::execution_plan::{AcceleratorFallback, ExecutionConfig};
use crate::backend::spiralk_fft::SpiralKFftPlan;
use crate::backend::unison_heuristics::{self, Choice, RankKind};

#[derive(Clone, Debug)]
pub struct RankPlan {
    pub kind: RankKind,
    pub rows: u32,
    pub cols: u32,
    pub k: u32,
    pub choice: Choice,
    pub execution_config: ExecutionConfig,
}

impl RankPlan {
    /// Build a SpiralK FFT plan that mirrors the heuristic choice associated with this rank plan.
    pub fn fft_plan(&self) -> SpiralKFftPlan {
        SpiralKFftPlan {
            radix: self.choice.fft_radix.clamp(2, 4),
            tile_cols: self.choice.fft_tile.max(1),
            segments: self.choice.fft_segments.max(1),
            subgroup: self.choice.subgroup,
        }
    }

    /// Emit a WGSL kernel derived from the underlying FFT plan.
    pub fn fft_wgsl(&self) -> String {
        self.fft_plan().emit_wgsl()
    }

    /// Emit the SpiralK hint associated with the FFT plan so DSL consumers can record the decision.
    pub fn fft_spiralk_hint(&self) -> String {
        self.fft_plan().emit_spiralk_hint()
    }

    /// Returns the fallback contract captured when this plan was created.
    pub const fn accelerator_fallback(&self) -> AcceleratorFallback {
        self.execution_config.accelerator_fallback
    }
}

pub fn plan_rank(kind: RankKind, rows: u32, cols: u32, k: u32, caps: DeviceCaps) -> RankPlan {
    plan_rank_with_config(kind, rows, cols, k, caps, ExecutionConfig::from_env())
}

/// Plans rank-k with an explicit execution contract instead of consulting process state.
pub fn plan_rank_with_config(
    kind: RankKind,
    rows: u32,
    cols: u32,
    k: u32,
    caps: DeviceCaps,
    execution_config: ExecutionConfig,
) -> RankPlan {
    let choice = unison_heuristics::choose_unified_rank(rows, cols, k, caps, kind);
    RankPlan {
        kind,
        rows,
        cols,
        k,
        choice,
        execution_config,
    }
}

/// Trait that a backend implements to execute rank-k with a given plan.
pub trait RankKExecutor {
    type Error;
    fn launch_topk(&self, plan: &RankPlan) -> Result<(), Self::Error>;
    fn launch_midk(&self, plan: &RankPlan) -> Result<(), Self::Error>;
    fn launch_bottomk(&self, plan: &RankPlan) -> Result<(), Self::Error>;
}

/// Helper to dispatch by kind.
pub fn execute_rank<E: RankKExecutor>(exec: &E, plan: &RankPlan) -> Result<(), E::Error> {
    match plan.kind {
        RankKind::TopK => exec.launch_topk(plan),
        RankKind::MidK => exec.launch_midk(plan),
        RankKind::BottomK => exec.launch_bottomk(plan),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_plan_carries_subgroup_hint() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let plan = plan_rank(RankKind::TopK, 128, 8192, 64, caps);
        let fft = plan.fft_plan();
        assert!(fft.tile_cols >= 1024);
        assert!(matches!(fft.radix, 2 | 4));
        assert!(fft.segments >= 1);
        assert!(fft.subgroup);
    }

    #[test]
    fn fft_helpers_emit_artifacts() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let plan = plan_rank(RankKind::TopK, 64, 4096, 32, caps);
        let wgsl = plan.fft_wgsl();
        assert!(wgsl.contains("@compute"));
        let hint = plan.fft_spiralk_hint();
        assert!(hint.contains("tile_cols"));
    }

    #[test]
    fn explicit_execution_config_is_captured_by_the_plan() {
        let config = ExecutionConfig::new(AcceleratorFallback::Forbid, 4096);
        let plan = plan_rank_with_config(
            RankKind::MidK,
            4,
            128,
            8,
            DeviceCaps::wgpu(32, true, 256),
            config,
        );
        assert_eq!(plan.execution_config, config);
        assert_eq!(plan.accelerator_fallback(), AcceleratorFallback::Forbid);
    }
}
