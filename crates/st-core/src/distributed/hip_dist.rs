//! HIP-based distributed hooks (feature-gated).

#[cfg(feature="hip")]
use st-backend-hip as hip;
use crate::distributed::topk_dist::{TopKShard, merge_two_shards_f32};

#[allow(unused)]
pub fn try_allreduce_lane(mut lane:i32) -> i32 {
    #[cfg(feature="hip")]
    {
        let mut buf=[lane];
        if hip::hip_available() && hip::hip_allreduce_i32(&mut buf).is_ok() {
            return buf[0];
        }
    }
    lane
}

#[allow(unused)]
pub fn distributed_topk_merge(a: TopKShard<f32>, b: TopKShard<f32>, k:usize) -> TopKShard<f32> {
    // CPU fallback merge (same logic is used as a control path)
    merge_two_shards_f32(&a, &b, k)
}
