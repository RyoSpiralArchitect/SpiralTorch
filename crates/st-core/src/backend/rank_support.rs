//! Static rank support shared by runtime dispatch and external native clients.

use crate::backend::unison_heuristics::RankKind;
use crate::ops::rank_entry::RankPlan;
use st_kernel_contracts::rank::{Exact2CeGeometry, Kind};

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
