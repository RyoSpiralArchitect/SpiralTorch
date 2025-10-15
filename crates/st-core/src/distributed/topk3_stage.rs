// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(all(feature = "hip", feature = "hip-real"))]

// NOTE: This file replaces the previous overlay version with real keep‑k selection.
// It preserves u64 gather and the tile→final path.

use crate::distributed::topk_dist::{merge_two_shards_f32, TopKShard};

#[derive(Clone, Debug)]
pub struct StagePlan {
    pub k_local: usize,
    pub k_merge: usize,
}
pub fn stage_plan(k: usize, _nodes: usize) -> StagePlan {
    let k_local = ((k as f32) * 1.5).ceil() as usize;
    StagePlan {
        k_local,
        k_merge: k,
    }
}

pub struct DistCtx {
    pub nranks: usize,
    pub rank: usize,
    pub use_hip: bool,
    pub merge_kind: Option<u32>,
} // 0=bitonic,1=shared,2=warp

pub fn run_topk3_stage(ctx: &DistCtx, local: TopKShard<f32>, k: usize) -> TopKShard<f32> {
    if ctx.nranks <= 1 || !ctx.use_hip {
        return TopKShard {
            vals: local.vals.into_iter().take(k).collect(),
            idxs: local.idxs.into_iter().take(k).collect(),
        };
    }
    #[cfg(feature = "hip")]
    {
        #[cfg(feature = "hip-real")]
        use st_backend_hip::rccl_comm::init_rccl_from_env;
        #[cfg(feature = "hip-real")]
        use st_backend_hip::real::{
            allgather_u64_dev, device_synchronize, free, kway_merge_bitonic_u64,
            kway_merge_shared_heap_keepk_u64, kway_merge_shared_heap_real_keepk_u64,
            kway_merge_warp_coop_keepk_u64, kway_merge_warp_heap_keepk_u64, malloc,
            memcpy_d2h_async, memcpy_h2d_async, pack_vals_idx_u64, topk_tile_bitonic_u64, HipPtr,
            HipStream,
        };

        let comm = match init_rccl_from_env() {
            Ok(c) => c,
            Err(_) => {
                return merge_two_shards_f32(&local, &local, k);
            }
        };
        let world = comm.world as usize;

        let rows = 1i32;
        let k_local = local.vals.len();
        let total = k_local * world;

        let stream = HipStream::create().ok().expect("HipStream");
        // Upload local shard (vals/idx) → pack u64 → u64 allgather
        let szv = k_local * std::mem::size_of::<f32>();
        let szi = k_local * std::mem::size_of::<i32>();
        let d_vals: HipPtr = malloc(szv).expect("malloc d_vals");
        let d_idx: HipPtr = malloc(szi).expect("malloc d_idx");
        unsafe {
            memcpy_h2d_async(d_vals, local.vals.as_ptr() as *const u8, szv, &stream).ok();
            memcpy_h2d_async(d_idx, local.idxs.as_ptr() as *const u8, szi, &stream).ok();
        }
        let d_packed_send: HipPtr =
            malloc(k_local * std::mem::size_of::<u64>()).expect("malloc d_packed_send");
        pack_vals_idx_u64(
            d_vals as *const f32,
            d_idx as *const i32,
            d_packed_send as *mut u64,
            k_local as i32,
            &stream,
        )
        .ok();
        let d_packed_recv: HipPtr =
            malloc(total * std::mem::size_of::<u64>()).expect("malloc d_packed_recv");
        allgather_u64_dev(comm.comm, &stream, d_packed_send, d_packed_recv, k_local).ok();

        let shared_limit: usize = std::env::var("HIP_SHARED_LIMIT_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(96 * 1024);
        let kernel_env = std::env::var("TOPK_KERNEL").unwrap_or_else(|_| "auto".into()); // auto|bitonic|shared|warp
        let kernel_pref = ctx.merge_kind.unwrap_or_else(|| match kernel_env.as_str() {
            "bitonic" => 0,
            "shared" => 1,
            "warp" => 2,
            _ => 999,
        });

        let mut out_vals = vec![0f32; k];
        let mut out_idx = vec![0i32; k];

        // capacity checks
        let can_shared = k <= 256 * 8;
        let can_warp = k <= 256 * 4;
        let needed = total * 8; // bytes when unpacked

        let select_auto = || -> u32 {
            if needed > shared_limit {
                return 0;
            } // bitonic path
            if k <= 128 {
                2
            } else if k <= 2048 {
                1
            } else {
                0
            }
        };
        let mk = if kernel_pref == 999 {
            select_auto()
        } else {
            kernel_pref
        };

        match mk {
            1 => {
                // shared real keep‑k
                if needed > shared_limit {
                    // fallback to bitonic
                    let d_out_vals = malloc(k * std::mem::size_of::<f32>()).unwrap();
                    let d_out_idx = malloc(k * std::mem::size_of::<i32>()).unwrap();
                    kway_merge_bitonic_u64(
                        d_packed_recv as *const u64,
                        rows,
                        total as i32,
                        k as i32,
                        d_out_vals as *mut f32,
                        d_out_idx as *mut i32,
                        &stream,
                    )
                    .ok();
                    unsafe {
                        memcpy_d2h_async(
                            out_vals.as_mut_ptr() as *mut u8,
                            d_out_vals,
                            k * std::mem::size_of::<f32>(),
                            &stream,
                        )
                        .ok();
                        memcpy_d2h_async(
                            out_idx.as_mut_ptr() as *mut u8,
                            d_out_idx,
                            k * std::mem::size_of::<i32>(),
                            &stream,
                        )
                        .ok();
                    }
                    let _ = device_synchronize();
                    free(d_out_vals);
                    free(d_out_idx);
                } else {
                    let d_out_vals = malloc(k * std::mem::size_of::<f32>()).unwrap();
                    let d_out_idx = malloc(k * std::mem::size_of::<i32>()).unwrap();
                    kway_merge_shared_heap_real_keepk_u64(
                        d_packed_recv as *const u64,
                        rows,
                        total as i32,
                        k as i32,
                        d_out_vals as *mut f32,
                        d_out_idx as *mut i32,
                        &stream,
                    )
                    .ok();
                    unsafe {
                        memcpy_d2h_async(
                            out_vals.as_mut_ptr() as *mut u8,
                            d_out_vals,
                            k * std::mem::size_of::<f32>(),
                            &stream,
                        )
                        .ok();
                        memcpy_d2h_async(
                            out_idx.as_mut_ptr() as *mut u8,
                            d_out_idx,
                            k * std::mem::size_of::<i32>(),
                            &stream,
                        )
                        .ok();
                    }
                    let _ = device_synchronize();
                    free(d_out_vals);
                    free(d_out_idx);
                }
            }
            2 => {
                // warp coop keep‑k
                if needed > shared_limit {
                    // fallback to bitonic 2-stage
                    let tile_total = (shared_limit / 8).saturating_sub(256).max(128);
                    let tiles = (total + tile_total - 1) / tile_total;
                    let k_tile = std::cmp::min(k, tile_total);
                    let d_tile_out = malloc(tiles * k_tile * std::mem::size_of::<u64>()).unwrap();
                    for t in 0..tiles {
                        let start = t * tile_total;
                        let count = std::cmp::min(tile_total, total - start);
                        topk_tile_bitonic_u64(
                            (d_packed_recv as usize + start * std::mem::size_of::<u64>())
                                as *const u64,
                            1,
                            count as i32,
                            k_tile as i32,
                            (d_tile_out as usize + t * k_tile * std::mem::size_of::<u64>())
                                as *mut u64,
                            &stream,
                        )
                        .ok();
                    }
                    let d_final_vals = malloc(k * std::mem::size_of::<f32>()).unwrap();
                    let d_final_idx = malloc(k * std::mem::size_of::<i32>()).unwrap();
                    kway_merge_bitonic_u64(
                        d_tile_out as *const u64,
                        rows,
                        (tiles * k_tile) as i32,
                        k as i32,
                        d_final_vals as *mut f32,
                        d_final_idx as *mut i32,
                        &stream,
                    )
                    .ok();
                    unsafe {
                        memcpy_d2h_async(
                            out_vals.as_mut_ptr() as *mut u8,
                            d_final_vals,
                            k * std::mem::size_of::<f32>(),
                            &stream,
                        )
                        .ok();
                        memcpy_d2h_async(
                            out_idx.as_mut_ptr() as *mut u8,
                            d_final_idx,
                            k * std::mem::size_of::<i32>(),
                            &stream,
                        )
                        .ok();
                    }
                    let _ = device_synchronize();
                    free(d_tile_out);
                    free(d_final_vals);
                    free(d_final_idx);
                } else {
                    let d_out_vals = malloc(k * std::mem::size_of::<f32>()).unwrap();
                    let d_out_idx = malloc(k * std::mem::size_of::<i32>()).unwrap();
                    kway_merge_warp_coop_keepk_u64(
                        d_packed_recv as *const u64,
                        rows,
                        total as i32,
                        k as i32,
                        d_out_vals as *mut f32,
                        d_out_idx as *mut i32,
                        &stream,
                    )
                    .ok();
                    unsafe {
                        memcpy_d2h_async(
                            out_vals.as_mut_ptr() as *mut u8,
                            d_out_vals,
                            k * std::mem::size_of::<f32>(),
                            &stream,
                        )
                        .ok();
                        memcpy_d2h_async(
                            out_idx.as_mut_ptr() as *mut u8,
                            d_out_idx,
                            k * std::mem::size_of::<i32>(),
                            &stream,
                        )
                        .ok();
                    }
                    let _ = device_synchronize();
                    free(d_out_vals);
                    free(d_out_idx);
                }
            }
            _ => {
                // bitonic (single or 2-stage by capacity)
                if needed <= shared_limit {
                    let d_out_vals = malloc(k * std::mem::size_of::<f32>()).unwrap();
                    let d_out_idx = malloc(k * std::mem::size_of::<i32>()).unwrap();
                    kway_merge_bitonic_u64(
                        d_packed_recv as *const u64,
                        rows,
                        total as i32,
                        k as i32,
                        d_out_vals as *mut f32,
                        d_out_idx as *mut i32,
                        &stream,
                    )
                    .ok();
                    unsafe {
                        memcpy_d2h_async(
                            out_vals.as_mut_ptr() as *mut u8,
                            d_out_vals,
                            k * std::mem::size_of::<f32>(),
                            &stream,
                        )
                        .ok();
                        memcpy_d2h_async(
                            out_idx.as_mut_ptr() as *mut u8,
                            d_out_idx,
                            k * std::mem::size_of::<i32>(),
                            &stream,
                        )
                        .ok();
                    }
                    let _ = device_synchronize();
                    free(d_out_vals);
                    free(d_out_idx);
                } else {
                    let tile_total = (shared_limit / 8).saturating_sub(256).max(128);
                    let tiles = (total + tile_total - 1) / tile_total;
                    let k_tile = std::cmp::min(k, tile_total);
                    let d_tile_out = malloc(tiles * k_tile * std::mem::size_of::<u64>()).unwrap();
                    for t in 0..tiles {
                        let start = t * tile_total;
                        let count = std::cmp::min(tile_total, total - start);
                        topk_tile_bitonic_u64(
                            (d_packed_recv as usize + start * std::mem::size_of::<u64>())
                                as *const u64,
                            1,
                            count as i32,
                            k_tile as i32,
                            (d_tile_out as usize + t * k_tile * std::mem::size_of::<u64>())
                                as *mut u64,
                            &stream,
                        )
                        .ok();
                    }
                    let d_final_vals = malloc(k * std::mem::size_of::<f32>()).unwrap();
                    let d_final_idx = malloc(k * std::mem::size_of::<i32>()).unwrap();
                    kway_merge_bitonic_u64(
                        d_tile_out as *const u64,
                        rows,
                        (tiles * k_tile) as i32,
                        k as i32,
                        d_final_vals as *mut f32,
                        d_final_idx as *mut i32,
                        &stream,
                    )
                    .ok();
                    unsafe {
                        memcpy_d2h_async(
                            out_vals.as_mut_ptr() as *mut u8,
                            d_final_vals,
                            k * std::mem::size_of::<f32>(),
                            &stream,
                        )
                        .ok();
                        memcpy_d2h_async(
                            out_idx.as_mut_ptr() as *mut u8,
                            d_final_idx,
                            k * std::mem::size_of::<i32>(),
                            &stream,
                        )
                        .ok();
                    }
                    let _ = device_synchronize();
                    free(d_tile_out);
                    free(d_final_vals);
                    free(d_final_idx);
                }
            }
        }

        free(d_vals);
        free(d_idx);
        free(d_packed_send);
        free(d_packed_recv);
        return TopKShard {
            vals: out_vals,
            idxs: out_idx,
        };
    }
    merge_two_shards_f32(&local, &local, k)
}
