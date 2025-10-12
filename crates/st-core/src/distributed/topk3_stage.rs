use crate::distributed::topk_dist::{TopKShard, merge_two_shards_f32};

#[derive(Clone, Debug)]
pub struct StagePlan { pub k_local: usize, pub k_merge: usize }
pub fn stage_plan(k:usize, nodes:usize)->StagePlan { StagePlan{ k_local: (k as f32*1.5).ceil() as usize, k_merge: k } }

pub struct DistCtx { pub nranks: usize, pub rank: usize, pub use_hip: bool }

pub fn run_topk3_stage(ctx:&DistCtx, local:TopKShard<f32>, k:usize)->TopKShard<f32>{
    if ctx.nranks<=1 { return TopKShard{ vals: local.vals.into_iter().take(k).collect(), idxs: local.idxs.into_iter().take(k).collect() }; }
    // TODO: allgather shards → device merge (HIP). skeleton: pairwise CPU merge (control path).
    merge_two_shards_f32(&local, &local, k)
}
