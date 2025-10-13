//! Consensus for lane parameters (Redis + optional HIP-real sync)

use super::LaneParams;
use serde_json::Value;

#[cfg(all(feature = "hip", feature = "hip-real"))]
use st_backend_hip::rccl_comm::init_rccl_from_env;

#[cfg(all(feature = "hip", feature = "hip-real"))]
fn maybe_sync() {
    // Best-effort RCCL init/sync (実装は将来拡充予定)
    let _ = init_rccl_from_env();
}

#[cfg(not(all(feature = "hip", feature = "hip-real")))]
fn maybe_sync() {}

fn median_i32(v: &mut [i32]) -> i32 {
    v.sort_unstable();
    let n = v.len();
    if n == 0 { return 0; }
    if n % 2 == 1 { v[n/2] } else {
        ((v[n/2 - 1] as i64 + v[n/2] as i64) / 2) as i32
    }
}

pub fn consensus_lane_params(mut p: LaneParams) -> LaneParams {
    // Redis から最近16件を取得
    let url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1/".into());
    if let Ok(samples) = st_kv::redis_lrange(&url, "spiral:heur:lparams", -16, -1) {
        let mut lanes: Vec<i32> = Vec::new();
        for s in samples {
            if let Ok(v) = serde_json::from_str::<Value>(&s) {
                if let Some(l) = v.get("lane").and_then(|x| x.as_i64()) {
                    lanes.push(l as i32);
                }
            }
        }
        if !lanes.is_empty() {
            let agg = std::env::var("SPIRAL_UNISON_AGG").unwrap_or_else(|_| "mean".into());
            let lane = if agg == "median" {
                median_i32(&mut lanes)
            } else {
                let sum: i64 = lanes.iter().map(|&x| x as i64).sum();
                (sum as f64 / lanes.len() as f64).round() as i32
            };
            p.lane = lane;
        }
    }

    // HIP-real が有効なら軽く同期（stub では no-op）
    maybe_sync();
    p
}
