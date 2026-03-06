use crate::ops::rank_cpu::{bottomk_select_into, midk_select_into, topk_select_into, RankCpuError};
use crate::ops::rank_entry::{RankKExecutor, RankPlan};

/// CPU executor that runs TopK/BottomK selection over a dense row-major slice.
///
/// The executor owns no buffers: callers provide input/output slices and the
/// executor can then be driven via `ops::rank_entry::execute_rank`.
pub struct CpuExecutor<'a> {
    x: &'a [f32],
    row_stride: u32,
    out_vals: &'a mut [f32],
    out_idx: &'a mut [u32],
}

impl<'a> CpuExecutor<'a> {
    pub fn new(
        x: &'a [f32],
        row_stride: u32,
        out_vals: &'a mut [f32],
        out_idx: &'a mut [u32],
    ) -> Self {
        Self {
            x,
            row_stride,
            out_vals,
            out_idx,
        }
    }
}

impl RankKExecutor for CpuExecutor<'_> {
    type Error = RankCpuError;

    fn launch_topk(&mut self, plan: &RankPlan) -> Result<(), Self::Error> {
        topk_select_into(
            self.x,
            plan.rows,
            plan.cols,
            self.row_stride,
            plan.k,
            self.out_vals,
            self.out_idx,
        )
    }

    fn launch_midk(&mut self, plan: &RankPlan) -> Result<(), Self::Error> {
        midk_select_into(
            self.x,
            plan.rows,
            plan.cols,
            self.row_stride,
            plan.k,
            self.out_vals,
            self.out_idx,
        )
    }

    fn launch_bottomk(&mut self, plan: &RankPlan) -> Result<(), Self::Error> {
        bottomk_select_into(
            self.x,
            plan.rows,
            plan.cols,
            self.row_stride,
            plan.k,
            self.out_vals,
            self.out_idx,
        )
    }
}
