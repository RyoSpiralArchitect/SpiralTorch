//! 3-stage distributed TopK outline with CPU fallback.
//! Stage1: local topK on each device
//! Stage2: inter-node K-way merge (tree/ring), exchanging candidates
//! Stage3: finalize and (optionally) allgather indices

#[derive(Clone, Debug)]
pub struct TopKShard<T>{
    pub vals: Vec<T>,
    pub idxs: Vec<i32>,
}

pub fn merge_two_shards_f32(a:&TopKShard<f32>, b:&TopKShard<f32>, k:usize) -> TopKShard<f32> {
    // Simple k-way merge on CPU (descending by value)
    let mut out_v = Vec::with_capacity(k);
    let mut out_i = Vec::with_capacity(k);
    let mut ia=0usize; let mut ib=0usize;
    while out_v.len()<k && (ia<a.vals.len() || ib<b.vals.len()){
        let choose_a = if ia<a.vals.len() && (ib>=b.vals.len() || a.vals[ia] >= b.vals[ib]) { true } else { false };
        if choose_a {
            out_v.push(a.vals[ia]); out_i.push(a.idxs[ia]); ia+=1;
        } else {
            out_v.push(b.vals[ib]); out_i.push(b.idxs[ib]); ib+=1;
        }
    }
    TopKShard{ vals: out_v, idxs: out_i }
}
