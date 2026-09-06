//! Static rank support shared by runtime dispatch and external native clients.

use crate::backend::device_caps::BackendKind;
use crate::backend::unison_heuristics::RankKind;
use crate::ops::rank_entry::{RankPlan, RankPlanError};
use st_kernel_contracts::rank::{Exact2CeGeometry, Kind};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg(any(feature = "kdsl", feature = "wgpu-rt"))]
pub(crate) enum DirectTopKPipeline {
    Workgroup,
    SubgroupHeap,
    SubgroupHeapIntrinsics,
    SubgroupBitonic,
}

#[cfg(any(feature = "kdsl", feature = "wgpu-rt"))]
impl DirectTopKPipeline {
    pub(crate) fn choose(
        has_subgroup: bool,
        algo_hint: u8,
        prefer_intrinsics: bool,
        cols: u32,
        k: u32,
    ) -> Self {
        if !has_subgroup {
            return Self::Workgroup;
        }
        let prefer_heap = match algo_hint {
            1 => true,
            2 => false,
            _ => (k <= 32 && cols >= 2048) || k <= 16,
        };
        match (prefer_heap, prefer_intrinsics) {
            (true, true) => Self::SubgroupHeapIntrinsics,
            (true, false) => Self::SubgroupHeap,
            (false, _) => Self::SubgroupBitonic,
        }
    }

    #[cfg(feature = "kdsl")]
    pub(crate) fn execution_suffix(self, k_lane: u32, fixed_subgroup_width: u32) -> String {
        match self {
            Self::Workgroup => "pipeline=workgroup".into(),
            Self::SubgroupHeap => "pipeline=subgroup_heap".into(),
            Self::SubgroupHeapIntrinsics => "pipeline=subgroup_heap_intrinsics".into(),
            Self::SubgroupBitonic if fixed_subgroup_width > 0 => format!(
                "pipeline=subgroup_bitonic/keep={}",
                k_lane.min(fixed_subgroup_width)
            ),
            Self::SubgroupBitonic => "pipeline=subgroup_bitonic/lane_scope=unresolved".into(),
        }
    }
}

pub(crate) fn exact_kind(kind: RankKind) -> Kind {
    match kind {
        RankKind::TopK => Kind::TopK,
        RankKind::MidK => Kind::MidK,
        RankKind::BottomK => Kind::BottomK,
    }
}

pub(crate) fn exact_tile_cols(plan: &RankPlan) -> u32 {
    match plan.kind {
        RankKind::TopK => plan.choice.tile,
        RankKind::MidK | RankKind::BottomK => plan.choice.ctile,
    }
}

/// Parameters consumed by the persistent exact two-command WGPU workspace.
/// Device limits and runtime readiness are still checked during allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WgpuResidentRankSpec {
    pub kind: Kind,
    pub rows: u32,
    pub cols: u32,
    pub k: u32,
    pub tile_cols: u32,
}

impl WgpuResidentRankSpec {
    /// Does not reinterpret a direct, CPU, or fallback-enabled plan as resident.
    pub fn try_from_plan(plan: &RankPlan) -> Result<Self, RankPlanError> {
        plan.validate()?;
        if !plan.accelerator_fallback().is_strict() {
            return Err(RankPlanError::DeclaredNativeRequiresStrictFallback);
        }
        if plan.device_caps.backend != BackendKind::Wgpu || !plan.choice.use_2ce {
            return Err(RankPlanError::UnsupportedNativeExecution {
                detail: "resident rank requires a WGPU exact two-command plan".into(),
            });
        }
        if plan.rows == 0 || plan.cols == 0 || plan.k == 0 {
            return Err(RankPlanError::UnsupportedNativeExecution {
                detail: "resident rank workspace dimensions must be positive".into(),
            });
        }
        let geometry =
            Exact2CeGeometry::try_new(plan.rows, plan.cols, plan.k, exact_tile_cols(plan))
                .map_err(|error| RankPlanError::UnsupportedNativeExecution {
                    detail: error.to_string(),
                })?;
        Ok(Self {
            kind: exact_kind(plan.kind),
            rows: plan.rows,
            cols: plan.cols,
            k: plan.k,
            tile_cols: geometry.tile_cols,
        })
    }
}

pub fn wgpu_rank_exact_support(plan: &RankPlan) -> Result<(), String> {
    if plan.choice.use_2ce {
        Exact2CeGeometry::try_new(plan.rows, plan.cols, plan.k, exact_tile_cols(plan))
            .map(|_| ())
            .map_err(|error| error.to_string())
    } else {
        wgpu_rank_exact_support_for(plan.kind, plan.rows, plan.cols, plan.k)
    }
}

pub fn wgpu_rank_exact_support_for(
    kind: RankKind,
    rows: u32,
    cols: u32,
    k: u32,
) -> Result<(), String> {
    st_kernel_contracts::rank::validate_wgpu_direct(exact_kind(kind), rows, cols, k)
}
