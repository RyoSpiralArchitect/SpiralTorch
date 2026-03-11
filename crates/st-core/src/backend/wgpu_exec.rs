use crate::ops::rank_entry::{RankKExecutor, RankPlan};

/// Legacy plan-only WGPU executor.
///
/// This surface does not carry concrete buffers, so actual execution still
/// requires `WgpuBufferExecutor` or direct `wgpu_rt::*_buffers(...)` calls.
#[derive(Default)]
pub struct WgpuExecutor;

impl RankKExecutor for WgpuExecutor {
    type Error = String;

    fn launch_topk(&mut self, plan: &RankPlan) -> Result<(), Self::Error> {
        dispatch_topk(plan)
    }

    fn launch_midk(&mut self, _plan: &RankPlan) -> Result<(), Self::Error> {
        unsupported_exact_rank("MidK")
    }

    fn launch_bottomk(&mut self, plan: &RankPlan) -> Result<(), Self::Error> {
        dispatch_bottomk(plan)
    }
}

/// Buffer-backed WGPU executor usable with `execute_rank(&mut exec, &plan)`.
///
/// The current exact-selection WGPU path is implemented for `TopK` and `BottomK`.
/// Threshold/mask compaction is a separate surface exposed through
/// `wgpu_rt::dispatch_compaction_*_buffers(...)`.
#[cfg(all(feature = "wgpu", feature = "wgpu-rt"))]
pub struct WgpuBufferExecutor<'a> {
    x: &'a wgpu::Buffer,
    row_stride: u32,
    out_vals: &'a wgpu::Buffer,
    out_idx: &'a wgpu::Buffer,
}

#[cfg(all(feature = "wgpu", feature = "wgpu-rt"))]
impl<'a> WgpuBufferExecutor<'a> {
    pub fn rank(
        x: &'a wgpu::Buffer,
        row_stride: u32,
        out_vals: &'a wgpu::Buffer,
        out_idx: &'a wgpu::Buffer,
    ) -> Self {
        Self {
            x,
            row_stride,
            out_vals,
            out_idx,
        }
    }

    pub fn topk(
        x: &'a wgpu::Buffer,
        row_stride: u32,
        out_vals: &'a wgpu::Buffer,
        out_idx: &'a wgpu::Buffer,
    ) -> Self {
        Self::rank(x, row_stride, out_vals, out_idx)
    }

    pub fn bottomk(
        x: &'a wgpu::Buffer,
        row_stride: u32,
        out_vals: &'a wgpu::Buffer,
        out_idx: &'a wgpu::Buffer,
    ) -> Self {
        Self::rank(x, row_stride, out_vals, out_idx)
    }
}

#[cfg(all(feature = "wgpu", feature = "wgpu-rt"))]
impl RankKExecutor for WgpuBufferExecutor<'_> {
    type Error = String;

    fn launch_topk(&mut self, plan: &RankPlan) -> Result<(), Self::Error> {
        crate::backend::wgpu_rt::dispatch_topk_1ce_buffers(
            plan.rows,
            plan.cols,
            plan.k,
            self.row_stride,
            plan.choice.kl.max(1),
            self.x,
            self.out_vals,
            self.out_idx,
        )
    }

    fn launch_midk(&mut self, _plan: &RankPlan) -> Result<(), Self::Error> {
        unsupported_exact_rank("MidK")
    }

    fn launch_bottomk(&mut self, plan: &RankPlan) -> Result<(), Self::Error> {
        crate::backend::wgpu_rt::dispatch_bottomk_1ce_buffers(
            plan.rows,
            plan.cols,
            plan.k,
            self.row_stride,
            self.x,
            self.out_vals,
            self.out_idx,
        )
    }
}

fn unsupported_exact_rank(kind: &str) -> Result<(), String> {
    Err(format!(
        "wgpu_exec: exact {kind} selection is not implemented on WGPU; use `ops::compaction::plan_compaction(...)` and `wgpu_rt::dispatch_compaction_*_buffers(...)` for threshold compaction"
    ))
}

fn dispatch_topk(plan: &RankPlan) -> Result<(), String> {
    topk_1ce(plan)
}

fn dispatch_bottomk(plan: &RankPlan) -> Result<(), String> {
    bottomk_1ce(plan)
}

#[cfg(all(feature = "wgpu", feature = "wgpu-rt"))]
fn topk_1ce(p: &RankPlan) -> Result<(), String> {
    crate::backend::wgpu_rt::dispatch_topk_1ce(p)
}

#[cfg(all(feature = "wgpu", feature = "wgpu-rt"))]
fn bottomk_1ce(p: &RankPlan) -> Result<(), String> {
    crate::backend::wgpu_rt::dispatch_bottomk_1ce(p)
}

#[cfg(not(all(feature = "wgpu", feature = "wgpu-rt")))]
fn topk_1ce(_p: &RankPlan) -> Result<(), String> {
    Err("wgpu-rt not enabled".into())
}

#[cfg(not(all(feature = "wgpu", feature = "wgpu-rt")))]
fn bottomk_1ce(_p: &RankPlan) -> Result<(), String> {
    Err("wgpu-rt not enabled".into())
}

#[cfg(test)]
mod tests {
    use super::{unsupported_exact_rank, WgpuExecutor};
    use crate::backend::device_caps::DeviceCaps;
    use crate::ops::rank_entry::{plan_rank, RankKExecutor, RankKind};

    #[test]
    fn exact_midk_is_reported_as_unsupported() {
        let mut exec = WgpuExecutor::default();
        let plan = plan_rank(RankKind::MidK, 4, 64, 8, DeviceCaps::wgpu(32, false, 256));
        let err = exec.launch_midk(&plan).unwrap_err();
        assert!(err.contains("exact MidK selection is not implemented"));
    }

    #[test]
    fn exact_bottomk_is_reported_as_unsupported() {
        let mut exec = WgpuExecutor::default();
        let plan = plan_rank(
            RankKind::BottomK,
            4,
            64,
            8,
            DeviceCaps::wgpu(32, false, 256),
        );
        let err = exec.launch_bottomk(&plan).unwrap_err();
        if cfg!(all(feature = "wgpu", feature = "wgpu-rt")) {
            assert!(err.contains("plan-only dispatch is not wired"));
        } else {
            assert!(err.contains("wgpu-rt not enabled"));
        }
    }

    #[test]
    fn unsupported_rank_error_points_to_compaction_api() {
        let err = unsupported_exact_rank("MidK").unwrap_err();
        assert!(err.contains("ops::compaction"));
        assert!(err.contains("dispatch_compaction"));
    }
}
