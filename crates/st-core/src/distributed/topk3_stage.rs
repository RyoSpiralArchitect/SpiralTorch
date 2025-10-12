use crate::distributed::topk_dist::{TopKShard, merge_two_shards_f32};

#[derive(Clone, Debug)]
pub struct StagePlan { pub k_local: usize, pub k_merge: usize }
pub fn stage_plan(k:usize, nodes:usize)->StagePlan { let k_local=((k as f32)*1.5).ceil() as usize; StagePlan{ k_local, k_merge:k } }

pub struct DistCtx { pub nranks: usize, pub rank: usize, pub use_hip: bool }

pub fn run_topk3_stage(ctx:&DistCtx, local:TopKShard<f32>, k:usize)->TopKShard<f32>{
    if ctx.nranks<=1 || !ctx.use_hip {
        return TopKShard{ vals: local.vals.into_iter().take(k).collect(), idxs: local.idxs.into_iter().take(k).collect() };
    }
    #[cfg(feature="hip")]
    {
        #[cfg(feature="hip")] use st_backend_hip::rccl_comm::init_rccl_from_env;
#[cfg(feature="hip")] use st_backend_hip::real::{HipStream, HipPtr, malloc, free, memcpy_h2d_async, memcpy_d2h_async,
            allgather_f32_dev, allgather_i32_dev, kway_merge_bitonic_f32};

        let comm = match init_rccl_from_env(){ Ok(c)=>c, Err(_) => { return merge_two_shards_f32(&local, &local, k); } };
        let rows = 1i32;
        let k_local = local.vals.len();
        let total = k_local * (comm.world as usize);

        let stream = HipStream::create().ok().expect("HipStream");
        // Upload local shard to device
        let szv = k_local * std::mem::size_of::<f32>();
        let szi = k_local * std::mem::size_of::<i32>();
        let d_send_vals: HipPtr = malloc(szv).expect("malloc d_send_vals");
        let d_send_idx : HipPtr = malloc(szi).expect("malloc d_send_idx");
        unsafe{
            memcpy_h2d_async(d_send_vals, local.vals.as_ptr() as *const u8, szv, &stream).ok();
            let mut idx_i32 = local.idxs.clone();
            memcpy_h2d_async(d_send_idx,  idx_i32.as_ptr() as *const u8,  szi, &stream).ok();
        }
        // Gather to device
        let d_recv_vals: HipPtr = malloc(total * std::mem::size_of::<f32>()).expect("malloc d_recv_vals");
        let d_recv_idx : HipPtr = malloc(total * std::mem::size_of::<i32>()).expect("malloc d_recv_idx");
        allgather_f32_dev(comm.comm, &stream, d_send_vals, d_recv_vals, k_local).ok();
        allgather_i32_dev(comm.comm, &stream, d_send_idx,  d_recv_idx,  k_local).ok();
        // Merge (Pass2)
        let d_out_vals: HipPtr = malloc(k * std::mem::size_of::<f32>()).expect("malloc d_out_vals");
        let d_out_idx : HipPtr = malloc(k * std::mem::size_of::<i32>()).expect("malloc d_out_idx");
        kway_merge_bitonic_f32(d_recv_vals as *const f32, d_recv_idx as *const i32, rows, total as i32, k as i32, d_out_vals as *mut f32, d_out_idx as *mut i32, &stream).ok();
        // D2H
        let mut out_vals = vec![0f32; k];
        let mut out_idx  = vec![0i32; k];
        unsafe{
            memcpy_d2h_async(out_vals.as_mut_ptr() as *mut u8, d_out_vals, k * std::mem::size_of::<f32>(), &stream).ok();
            memcpy_d2h_async(out_idx .as_mut_ptr()  as *mut u8, d_out_idx,  k * std::mem::size_of::<i32>(),  &stream).ok();
        }
        let _ = st_backend_hip::real::device_synchronize();
        // Free
        free(d_send_vals); free(d_send_idx); free(d_recv_vals); free(d_recv_idx); free(d_out_vals); free(d_out_idx);
        return TopKShard{ vals: out_vals, idxs: out_idx };
    }
    merge_two_shards_f32(&local, &local, k)
}
