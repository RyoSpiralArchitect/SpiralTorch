// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

/// Consensus for lane parameters (Redis-gated) with optional HIP-real sync.
/// This module is the minimal slice that lets `st-core` compile on its own:
/// - Defines `LaneParams` locally to avoid cross-crate dependencies.
/// - Reads Redis only when the `kv-redis` feature is enabled.
/// - Keeps HIP-real synchronisation behind a compile-time stub.

#[derive(Clone, Debug)]
pub struct LaneParams {
    pub lane: i32,
}

#[cfg(feature = "kv-redis")]
use serde_json::Value;

/// Stub that runs only when HIP-real is enabled (real sync can replace it later).
#[cfg(all(feature = "hip", feature = "hip-real"))]
fn maybe_sync() {
    // Placeholder for RCCL init/synchronisation. Safe no-op for now.
}

#[cfg(not(all(feature = "hip", feature = "hip-real")))]
fn maybe_sync() {}

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

    let agg = std::env::var("SPIRAL_UNISON_AGG").unwrap_or_else(|_| "mean".into());
    let lane = if agg == "median" {
        lanes.sort_unstable();
        lanes[lanes.len() / 2]
    } else {
        let sum: i64 = lanes.iter().map(|&x| x as i64).sum();
        (sum as f64 / lanes.len() as f64).round() as i32
    };
    Some(lane)
}

#[cfg(not(feature = "kv-redis"))]
fn fetch_lane_from_redis() -> Option<i32> {
    None
}

/// Apply runtime consensus, run the HIP-real stub if needed, and return.
pub fn consensus_lane_params(mut p: LaneParams) -> LaneParams {
    if let Some(lane) = fetch_lane_from_redis() {
        p.lane = lane;
    }
    maybe_sync();
    p
}
