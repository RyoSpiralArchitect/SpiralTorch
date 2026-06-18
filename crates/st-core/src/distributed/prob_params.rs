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

use st_tensor::{emit_tensor_op, emit_tensor_op_meta};

#[cfg(feature = "kv-redis")]
use serde_json::Value;

const LANE_MIN: i32 = 1;
const LANE_MAX: i32 = 4096;

fn sanitize_lane_i64(lane: i64) -> (i32, bool) {
    let clamped = lane.clamp(i64::from(LANE_MIN), i64::from(LANE_MAX)) as i32;
    (clamped, i64::from(clamped) != lane)
}

fn sanitize_lane(lane: i32) -> (i32, bool) {
    sanitize_lane_i64(i64::from(lane))
}

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
    for lane in lanes.iter_mut() {
        *lane = sanitize_lane(*lane).0;
    }

    if policy == "median" {
        lanes.sort_unstable();
        return Some(lanes[lanes.len() / 2]);
    }

    let sum: i128 = lanes.iter().map(|&x| i128::from(x)).sum();
    let mean = (sum as f64 / lanes.len() as f64).round() as i64;
    Some(sanitize_lane_i64(mean).0)
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
                sanitize_lane_i64(lane64).0
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
                lanes.push(sanitize_lane_i64(l).0);
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
    let input_lane = p.lane;
    let mut redis_applied = false;
    let mut hip_sync_applied = false;
    let (local_lane, input_lane_sanitized) = sanitize_lane(p.lane);
    p.lane = local_lane;
    if let Some(lane) = fetch_lane_from_redis() {
        p.lane = lane;
        redis_applied = true;
    }
    if let Some(lane) = maybe_sync_lane(p.lane) {
        p.lane = sanitize_lane(lane).0;
        hip_sync_applied = true;
    }
    let (output_lane, output_lane_sanitized) = sanitize_lane(p.lane);
    p.lane = output_lane;
    emit_tensor_op("distributed_lane_consensus", &[1], &[1]);
    emit_tensor_op_meta("distributed_lane_consensus", || {
        serde_json::json!({
            "backend": "cpu",
            "requested_backend": "auto",
            "kind": "st_core_distributed_lane_consensus",
            "input_lane": input_lane,
            "output_lane": p.lane,
            "changed": input_lane != p.lane,
            "input_lane_sanitized": input_lane_sanitized,
            "output_lane_sanitized": output_lane_sanitized,
            "lane_min": LANE_MIN,
            "lane_max": LANE_MAX,
            "redis_applied": redis_applied,
            "hip_sync_applied": hip_sync_applied,
            "kv_redis_enabled": cfg!(feature = "kv-redis"),
            "hip_real_enabled": cfg!(all(feature = "hip", feature = "hip-real")),
        })
    });
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

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

    #[test]
    fn aggregate_lanes_sanitizes_invalid_lane_suggestions() {
        let mut lanes = vec![-128, 32, i32::MAX];
        assert_eq!(aggregate_lanes_with_policy("median", &mut lanes), Some(32));
        assert_eq!(lanes, vec![1, 32, LANE_MAX]);

        let mut lanes = vec![-10, 0, 2];
        assert_eq!(aggregate_lanes_with_policy("mean", &mut lanes), Some(1));
        assert_eq!(lanes, vec![1, 1, 2]);
    }

    #[test]
    fn consensus_lane_params_emits_backend_meta() {
        let _lock = crate::telemetry::tensor_observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let out = consensus_lane_params(LaneParams { lane: 3 });
        st_tensor::set_tensor_op_meta_observer(previous);

        assert_eq!(out.lane, 3);
        let events = events.lock().unwrap();
        let meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "distributed_lane_consensus"
                    && data["kind"] == "st_core_distributed_lane_consensus"
            })
            .expect("distributed lane consensus metadata event");
        assert_eq!(meta.1["backend"], "cpu");
        assert_eq!(meta.1["input_lane"], 3);
        assert_eq!(meta.1["output_lane"], 3);
        assert_eq!(meta.1["changed"], false);
        assert_eq!(meta.1["input_lane_sanitized"], false);
        assert_eq!(meta.1["output_lane_sanitized"], false);
        assert_eq!(meta.1["lane_min"], LANE_MIN);
        assert_eq!(meta.1["lane_max"], LANE_MAX);
    }

    #[test]
    fn consensus_lane_params_sanitizes_invalid_input_lane_and_emits_meta() {
        let _lock = crate::telemetry::tensor_observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let out = consensus_lane_params(LaneParams { lane: -17 });
        st_tensor::set_tensor_op_meta_observer(previous);

        assert_eq!(out.lane, LANE_MIN);
        let events = events.lock().unwrap();
        let meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "distributed_lane_consensus"
                    && data["kind"] == "st_core_distributed_lane_consensus"
                    && data["input_lane"] == -17
            })
            .expect("distributed lane consensus metadata event");
        assert_eq!(meta.1["output_lane"], LANE_MIN);
        assert_eq!(meta.1["changed"], true);
        assert_eq!(meta.1["input_lane_sanitized"], true);
        assert_eq!(meta.1["output_lane_sanitized"], false);
    }
}
