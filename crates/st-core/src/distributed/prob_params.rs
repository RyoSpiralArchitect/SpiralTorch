// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

/// Consensus for lane parameters (Redis-gated) with optional HIP-real sync.
/// This module is the minimal slice that lets `st-core` compile on its own:
/// - Defines `LaneParams` locally to avoid cross-crate dependencies.
/// - Reads Redis only when the `kv-redis` feature is enabled.
/// - Enables HIP-real lane synchronisation only under compile-time gates.

#[derive(Clone, Debug)]
pub struct LaneParams {
    pub lane: i32,
}

#[cfg(feature = "kv-redis")]
use serde_json::Value;

#[cfg(any(feature = "kv-redis", all(feature = "hip", feature = "hip-real")))]
fn aggregate_lanes(lanes: &mut [i32]) -> Option<i32> {
    let policy = std::env::var("SPIRAL_UNISON_AGG")
        .unwrap_or_else(|_| "mean".into())
        .to_ascii_lowercase();
    aggregate_lanes_with_policy(&policy, lanes)
}

#[cfg(any(feature = "kv-redis", all(feature = "hip", feature = "hip-real"), test))]
fn aggregate_lanes_with_policy(policy: &str, lanes: &mut [i32]) -> Option<i32> {
    if lanes.is_empty() {
        return None;
    }

    if policy == "median" {
        lanes.sort_unstable();
        return Some(lanes[lanes.len() / 2]);
    }

    let sum: i64 = lanes.iter().map(|&x| i64::from(x)).sum();
    Some((sum as f64 / lanes.len() as f64).round() as i32)
}

#[cfg(all(feature = "hip", feature = "hip-real"))]
fn maybe_sync_lane(local_lane: i32) -> Option<i32> {
    use st_backend_hip::rccl_comm::init_rccl_from_env;
    use st_backend_hip::real::{
        allgather_u64_dev, free, malloc, memcpy_d2h_async, memcpy_h2d_async, stream_synchronize,
        HipStream,
    };

    let comm = init_rccl_from_env().ok()?;
    let world = comm.world.max(1) as usize;
    if world <= 1 {
        return Some(local_lane);
    }

    let stream = HipStream::create().ok()?;
    let elem_bytes = std::mem::size_of::<u64>();
    let recv_bytes = elem_bytes.checked_mul(world)?;

    let send = malloc(elem_bytes).ok()?;
    let recv = match malloc(recv_bytes) {
        Ok(ptr) => ptr,
        Err(_) => {
            let _ = free(send);
            return None;
        }
    };

    let result = (|| -> Option<i32> {
        let lane_bits = i64::from(local_lane) as u64;
        let mut gathered = vec![0u64; world];

        unsafe {
            memcpy_h2d_async(
                send,
                (&lane_bits as *const u64).cast::<u8>(),
                elem_bytes,
                &stream,
            )
            .ok()?;
            allgather_u64_dev(comm.comm, &stream, send, recv, 1).ok()?;
            memcpy_d2h_async(
                gathered.as_mut_ptr().cast::<u8>(),
                recv,
                recv_bytes,
                &stream,
            )
            .ok()?;
        }
        stream_synchronize(&stream).ok()?;

        let mut lanes: Vec<i32> = gathered
            .into_iter()
            .map(|bits| {
                let lane64 = bits as i64;
                lane64.clamp(i32::MIN as i64, i32::MAX as i64) as i32
            })
            .collect();
        aggregate_lanes(&mut lanes)
    })();

    let _ = free(send);
    let _ = free(recv);
    result
}

#[cfg(not(all(feature = "hip", feature = "hip-real")))]
fn maybe_sync_lane(_local_lane: i32) -> Option<i32> {
    None
}

/// Pull lane suggestions from Redis and aggregate via median/mean.
#[cfg(feature = "kv-redis")]
fn fetch_lane_from_redis() -> Option<i32> {
    let url = std::env::var("REDIS_URL").ok()?;
    let samples = st_kv::redis_lrange(&url, "spiral:heur:lparams", -16, -1).ok()?;

    let mut lanes: Vec<i32> = Vec::new();
    for s in samples {
        if let Ok(v) = serde_json::from_str::<Value>(&s) {
            if let Some(l) = v.get("lane").and_then(|x| x.as_i64()) {
                lanes.push(l as i32);
            }
        }
    }
    if lanes.is_empty() {
        return None;
    }

    aggregate_lanes(&mut lanes)
}

#[cfg(not(feature = "kv-redis"))]
fn fetch_lane_from_redis() -> Option<i32> {
    None
}

/// Apply runtime consensus, optionally synchronise with HIP-real peers, and return.
pub fn consensus_lane_params(mut p: LaneParams) -> LaneParams {
    if let Some(lane) = fetch_lane_from_redis() {
        p.lane = lane;
    }
    if let Some(lane) = maybe_sync_lane(p.lane) {
        p.lane = lane;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_lanes_mean_rounds() {
        let mut lanes = vec![1, 2, 2, 3];
        assert_eq!(aggregate_lanes_with_policy("mean", &mut lanes), Some(2));
    }

    #[test]
    fn aggregate_lanes_median_uses_middle() {
        let mut lanes = vec![9, 1, 7, 3, 5];
        assert_eq!(aggregate_lanes_with_policy("median", &mut lanes), Some(5));
    }
}
