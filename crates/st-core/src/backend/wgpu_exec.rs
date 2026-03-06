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

    fn launch_midk(&mut self, plan: &RankPlan) -> Result<(), Self::Error> {
        dispatch_midk(plan)
    }

    fn launch_bottomk(&mut self, plan: &RankPlan) -> Result<(), Self::Error> {
        dispatch_bottomk(plan)
    }
}

#[cfg(all(feature = "wgpu", feature = "wgpu-rt"))]
#[derive(Clone, Copy)]
enum WgpuBufferBindings<'a> {
    TopK {
        out_vals: &'a wgpu::Buffer,
        out_idx: &'a wgpu::Buffer,
    },
    Compaction {
        mask: &'a wgpu::Buffer,
        out_counts: &'a wgpu::Buffer,
        out_vals: &'a wgpu::Buffer,
        out_idx: &'a wgpu::Buffer,
    },
}

/// Buffer-backed WGPU executor usable with `execute_rank(&mut exec, &plan)`.
///
/// The current WGPU runtime has two output contracts:
/// - `TopK`: `values + indices`
/// - `MidK/BottomK`: `mask -> counts + packed values + packed indices`
///
/// `RankKind::MidK` on this executor still refers to the existing mask-driven
/// compaction path, not the CPU exact centered-window selection semantics.
#[cfg(all(feature = "wgpu", feature = "wgpu-rt"))]
pub struct WgpuBufferExecutor<'a> {
    x: &'a wgpu::Buffer,
    row_stride: u32,
    bindings: WgpuBufferBindings<'a>,
}

#[cfg(all(feature = "wgpu", feature = "wgpu-rt"))]
impl<'a> WgpuBufferExecutor<'a> {
    pub fn topk(
        x: &'a wgpu::Buffer,
        row_stride: u32,
        out_vals: &'a wgpu::Buffer,
        out_idx: &'a wgpu::Buffer,
    ) -> Self {
        Self {
            x,
            row_stride,
            bindings: WgpuBufferBindings::TopK { out_vals, out_idx },
        }
    }

    pub fn compaction(
        x: &'a wgpu::Buffer,
        row_stride: u32,
        mask: &'a wgpu::Buffer,
        out_counts: &'a wgpu::Buffer,
        out_vals: &'a wgpu::Buffer,
        out_idx: &'a wgpu::Buffer,
    ) -> Self {
        Self {
            x,
            row_stride,
            bindings: WgpuBufferBindings::Compaction {
                mask,
                out_counts,
                out_vals,
                out_idx,
            },
        }
    }
}

#[cfg(all(feature = "wgpu", feature = "wgpu-rt"))]
impl RankKExecutor for WgpuBufferExecutor<'_> {
    type Error = String;

    fn launch_topk(&mut self, plan: &RankPlan) -> Result<(), Self::Error> {
        let WgpuBufferBindings::TopK { out_vals, out_idx } = self.bindings else {
            return Err(
                "wgpu_exec: topk launch requires `WgpuBufferExecutor::topk(...)` buffers".into(),
            );
        };
        crate::backend::wgpu_rt::dispatch_topk_1ce_buffers(
            plan.rows,
            plan.cols,
            plan.k,
            self.row_stride,
            plan.choice.kl.max(1),
            self.x,
            out_vals,
            out_idx,
        )
    }

    fn launch_midk(&mut self, plan: &RankPlan) -> Result<(), Self::Error> {
        dispatch_compaction_buffers(self, plan, 0)
    }

    fn launch_bottomk(&mut self, plan: &RankPlan) -> Result<(), Self::Error> {
        dispatch_compaction_buffers(self, plan, 1)
    }
}

fn two_ce_decision(plan: &RankPlan) -> bool {
    let c = &plan.choice;
    if c.use_2ce {
        return true;
    }
    let tiles = (plan.cols as u64 + c.ctile.max(256) as u64 - 1) / c.ctile.max(256) as u64;
    tiles >= 64
}

#[cfg(all(feature = "wgpu", feature = "wgpu-rt"))]
fn dispatch_compaction_buffers(
    exec: &WgpuBufferExecutor<'_>,
    plan: &RankPlan,
    kind: u32,
) -> Result<(), String> {
    let WgpuBufferBindings::Compaction {
        mask,
        out_counts,
        out_vals,
        out_idx,
    } = exec.bindings
    else {
        return Err(
            "wgpu_exec: compaction launch requires `WgpuBufferExecutor::compaction(...)` buffers"
                .into(),
        );
    };

    if two_ce_decision(plan) {
        crate::backend::wgpu_rt::dispatch_compaction_2ce_buffers(
            plan.rows,
            plan.cols,
            exec.row_stride,
            kind,
            exec.x,
            mask,
            out_counts,
            out_vals,
            out_idx,
        )
    } else {
        crate::backend::wgpu_rt::dispatch_compaction_1ce_buffers(
            plan.rows,
            plan.cols,
            exec.row_stride,
            kind,
            exec.x,
            mask,
            out_counts,
            out_vals,
            out_idx,
        )
    }
}

fn dispatch_topk(plan: &RankPlan) -> Result<(), String> {
    topk_1ce(plan)
}

fn dispatch_midk(plan: &RankPlan) -> Result<(), String> {
    if two_ce_decision(plan) {
        midk_two_ce(plan)
    } else {
        midk_one_ce(plan)
    }
}

fn dispatch_bottomk(plan: &RankPlan) -> Result<(), String> {
    if two_ce_decision(plan) {
        bottomk_two_ce(plan)
    } else {
        bottomk_one_ce(plan)
    }
}

#[cfg(all(feature = "wgpu", feature = "wgpu-rt"))]
fn topk_1ce(p: &RankPlan) -> Result<(), String> {
    crate::backend::wgpu_rt::dispatch_topk_1ce(p)
}

#[cfg(not(all(feature = "wgpu", feature = "wgpu-rt")))]
fn topk_1ce(_p: &RankPlan) -> Result<(), String> {
    Err("wgpu-rt not enabled".into())
}

#[cfg(all(feature = "wgpu", feature = "wgpu-rt"))]
fn midk_one_ce(p: &RankPlan) -> Result<(), String> {
    crate::backend::wgpu_rt::dispatch_compaction_1ce(p, 0)
}

#[cfg(not(all(feature = "wgpu", feature = "wgpu-rt")))]
fn midk_one_ce(_: &RankPlan) -> Result<(), String> {
    Err("wgpu-rt not enabled".into())
}

#[cfg(all(feature = "wgpu", feature = "wgpu-rt"))]
fn midk_two_ce(p: &RankPlan) -> Result<(), String> {
    crate::backend::wgpu_rt::dispatch_compaction_2ce(p, 0)
}

#[cfg(not(all(feature = "wgpu", feature = "wgpu-rt")))]
fn midk_two_ce(_: &RankPlan) -> Result<(), String> {
    Err("wgpu-rt not enabled".into())
}

#[cfg(all(feature = "wgpu", feature = "wgpu-rt"))]
fn bottomk_one_ce(p: &RankPlan) -> Result<(), String> {
    crate::backend::wgpu_rt::dispatch_compaction_1ce(p, 1)
}

#[cfg(not(all(feature = "wgpu", feature = "wgpu-rt")))]
fn bottomk_one_ce(_: &RankPlan) -> Result<(), String> {
    Err("wgpu-rt not enabled".into())
}

#[cfg(all(feature = "wgpu", feature = "wgpu-rt"))]
fn bottomk_two_ce(p: &RankPlan) -> Result<(), String> {
    crate::backend::wgpu_rt::dispatch_compaction_2ce(p, 1)
}

#[cfg(not(all(feature = "wgpu", feature = "wgpu-rt")))]
fn bottomk_two_ce(_: &RankPlan) -> Result<(), String> {
    Err("wgpu-rt not enabled".into())
}

#[cfg(test)]
mod tests {
    use super::two_ce_decision;
    use crate::backend::device_caps::DeviceCaps;
    use crate::ops::rank_entry::{plan_rank, RankKind};

    #[test]
    fn two_ce_decision_respects_explicit_choice() {
        let mut plan = plan_rank(RankKind::BottomK, 4, 1024, 8, DeviceCaps::cpu());
        plan.choice.use_2ce = true;
        assert!(two_ce_decision(&plan));
    }

    #[test]
    fn two_ce_decision_uses_tile_pressure_fallback() {
        let plan = plan_rank(RankKind::BottomK, 4, 262_144, 8, DeviceCaps::wgpu(32, false, 256));
        assert!(two_ce_decision(&plan));
    }
}
