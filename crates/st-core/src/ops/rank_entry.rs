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
use crate::backend::spiralk_fft::SpiralKFftPlan;
use crate::backend::unison_heuristics::{self, Choice, RankKind};

#[derive(Clone, Debug)]
pub struct RankPlan {
    pub kind: RankKind,
    pub rows: u32,
    pub cols: u32,
    pub k: u32,
    pub choice: Choice,
}

impl RankPlan {
    /// Build a SpiralK FFT plan that mirrors the heuristic choice associated with this rank plan.
    pub fn fft_plan(&self) -> SpiralKFftPlan {
        SpiralKFftPlan {
            radix: self.choice.fft_radix.max(2).min(4),
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
}

pub fn plan_rank(kind: RankKind, rows: u32, cols: u32, k: u32, caps: DeviceCaps) -> RankPlan {
    let choice = unison_heuristics::choose_unified_rank(rows, cols, k, caps, kind);
    RankPlan {
        kind,
        rows,
        cols,
        k,
        choice,
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
}
