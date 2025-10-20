// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#[cfg(feature = "logic")]
use super::soft_logic::SoftRule;
#[cfg(all(feature = "logic", feature = "kv-redis"))]
use super::wgpu_heuristics::{SOFT_NAME_CH, SOFT_NAME_KL, SOFT_NAME_USE2CE, SOFT_NAME_WG};
#[cfg(all(feature = "logic", feature = "kv-redis"))]
use serde_json::Value;

#[cfg(feature = "logic")]
pub fn kv_consensus_soft_rules(
    rows: u32,
    cols: u32,
    k: u32,
    subgroup: bool,
    _kind: &'static str,
) -> Vec<SoftRule> {
    #[allow(unused_mut)]
    let mut out = Vec::<SoftRule>::new();
    #[cfg(not(feature = "kv-redis"))]
    let _ = (rows, cols, k, subgroup);
    #[cfg(feature = "kv-redis")]
    {
        if let Ok(url) = std::env::var("REDIS_URL") {
            let lg2c = (32 - (cols.max(1) - 1).leading_zeros()) as u32;
            let lg2k = (32 - (k.max(1) - 1).leading_zeros()) as u32;
            let key = format!(
                "spiral:heur:v1:list:sg:{}:c:{}:k:{}",
                if subgroup { 1 } else { 0 },
                lg2c,
                lg2k
            );
            if let Ok(list) = st_kv::redis_lrange(&url, &key, 0, -1) {
                let w_def = std::env::var("SPIRAL_KV_SOFT_W")
                    .ok()
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.08);
                let mut use2: Vec<bool> = vec![];
                let mut wg: Vec<u32> = vec![];
                let mut kl: Vec<u32> = vec![];
                let mut ch: Vec<u32> = vec![];
                let mut wts: Vec<f32> = vec![];
                for js in list.iter() {
                    if let Ok(v) = serde_json::from_str::<Value>(js) {
                        if let Some(b) = v.get("use_2ce").and_then(|x| x.as_bool()) {
                            use2.push(b);
                        }
                        if let Some(u) = v.get("wg").and_then(|x| x.as_u64()) {
                            wg.push(u as u32);
                        }
                        if let Some(u) = v.get("kl").and_then(|x| x.as_u64()) {
                            kl.push(u as u32);
                        }
                        if let Some(u) = v.get("ch").and_then(|x| x.as_u64()) {
                            ch.push(u as u32);
                        }
                        let w = v
                            .get("weight")
                            .and_then(|x| x.as_f64())
                            .map(|f| f as f32)
                            .unwrap_or(w_def);
                        wts.push(w);
                    }
                }
                let w_med = if wts.is_empty() {
                    w_def
                } else {
                    let mut a = wts.clone();
                    a.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    a[a.len() / 2]
                };
                if let Some(b) = majority_bool(&use2) {
                    out.push(SoftRule {
                        name: SOFT_NAME_USE2CE,
                        weight: w_med,
                        score: if b { 1.0 } else { -1.0 },
                    });
                }
                if let Some(u) = median_u32(&wg) {
                    out.push(SoftRule {
                        name: SOFT_NAME_WG,
                        weight: w_med,
                        score: u as f32,
                    });
                }
                if let Some(u) = median_u32(&kl) {
                    out.push(SoftRule {
                        name: SOFT_NAME_KL,
                        weight: w_med,
                        score: u as f32,
                    });
                }
                if let Some(u) = median_u32(&ch) {
                    out.push(SoftRule {
                        name: SOFT_NAME_CH,
                        weight: w_med,
                        score: u as f32,
                    });
                }
            }
        }
    }
    out
}
#[cfg(not(feature = "logic"))]
pub fn kv_consensus_soft_rules(
    _r: u32,
    _c: u32,
    _k: u32,
    _sg: bool,
    _kind: &'static str,
) -> Vec<()> {
    Vec::new()
}
#[cfg(all(feature = "logic", feature = "kv-redis"))]
fn median_u32(v: &[u32]) -> Option<u32> {
    if v.is_empty() {
        None
    } else {
        let mut a = v.to_vec();
        a.sort_unstable();
        Some(a[a.len() / 2])
    }
}
#[cfg(all(feature = "logic", feature = "kv-redis"))]
fn majority_bool(v: &[bool]) -> Option<bool> {
    if v.is_empty() {
        None
    } else {
        let t = v.iter().filter(|&&b| b).count();
        Some(t * 2 >= v.len())
    }
}
