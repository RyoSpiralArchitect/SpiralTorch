/// Consensus for lane parameters (Redis-gated) + optional HIP-real sync.
/// このモジュールは st-core 単体でコンパイルできる最小構成です。
/// - LaneParams をここで定義（他モジュールへの依存を断つ）
/// - Redis は feature "kv-redis" のときだけ読む
/// - HIP-real 同期はコンパイル時ガードのみ（stub）

#[derive(Clone, Debug)]
pub struct LaneParams {
    pub lane: i32,
}

#[cfg(feature="kv-redis")]
use serde_json::Value;

/// HIP-real が有効なときだけ呼ばれる“何もしない”stub（実装は将来差し替え）
#[cfg(all(feature="hip", feature="hip-real"))]
fn maybe_sync() {
    // ここに rccl の初期化/同期を入れる予定。現状は no-op で安全。
}

#[cfg(not(all(feature="hip", feature="hip-real")))]
fn maybe_sync() {}

/// Redis から lane 提案のサンプルを読み、agg=median/mean で要約
#[cfg(feature="kv-redis")]
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
    if lanes.is_empty() { return None; }

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

#[cfg(not(feature="kv-redis"))]
fn fetch_lane_from_redis() -> Option<i32> { None }

/// ランタイム合意で lane を上書き（あれば）→ HIP-real 同期（stub）→ 返却
pub fn consensus_lane_params(mut p: LaneParams) -> LaneParams {
    if let Some(lane) = fetch_lane_from_redis() {
        p.lane = lane;
    }
    maybe_sync();
    p
}
