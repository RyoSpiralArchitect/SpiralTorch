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
use st_tensor::{emit_tensor_op, emit_tensor_op_meta};

fn kv_consensus_bucket(value: u32) -> u32 {
    32 - (value.max(1) - 1).leading_zeros()
}

#[allow(
    clippy::too_many_arguments,
    reason = "Consensus metadata mirrors the soft-rule lookup tuple"
)]
fn emit_kv_consensus_soft_rules_meta(
    rows: u32,
    cols: u32,
    k: u32,
    subgroup: bool,
    kind: &'static str,
    status: &'static str,
    redis_url_present: bool,
    list_len: usize,
    parsed_len: usize,
    rule_count: usize,
    median_weight: f32,
) {
    let lg2c = kv_consensus_bucket(cols);
    let lg2k = kv_consensus_bucket(k);
    emit_tensor_op(
        "kv_consensus_soft_rules",
        &[rows as usize, cols as usize, k as usize],
        &[rule_count, list_len, parsed_len],
    );
    emit_tensor_op_meta("kv_consensus_soft_rules", || {
        serde_json::json!({
            "kind": "st_core_kv_consensus_soft_rules",
            "backend": "cpu",
            "requested_backend": "auto",
            "heuristic_kind": kind,
            "rows": rows,
            "cols": cols,
            "k": k,
            "subgroup": subgroup,
            "logic_feature_enabled": cfg!(feature = "logic"),
            "kv_feature_enabled": cfg!(feature = "kv-redis"),
            "redis_url_present": redis_url_present,
            "status": status,
            "key_lg2c": lg2c,
            "key_lg2k": lg2k,
            "list_len": list_len,
            "parsed_len": parsed_len,
            "rule_count": rule_count,
            "median_weight": if median_weight.is_finite() { median_weight as f64 } else { 0.0 },
        })
    });
}

#[cfg(feature = "logic")]
pub fn kv_consensus_soft_rules(
    rows: u32,
    cols: u32,
    k: u32,
    subgroup: bool,
    kind: &'static str,
) -> Vec<SoftRule> {
    #[allow(unused_mut)]
    let mut out = Vec::<SoftRule>::new();
    #[cfg(not(feature = "kv-redis"))]
    {
        emit_kv_consensus_soft_rules_meta(
            rows,
            cols,
            k,
            subgroup,
            kind,
            "feature_disabled",
            std::env::var_os("REDIS_URL").is_some(),
            0,
            0,
            0,
            0.0,
        );
    }
    #[cfg(feature = "kv-redis")]
    {
        if let Ok(url) = std::env::var("REDIS_URL") {
            let lg2c = kv_consensus_bucket(cols);
            let lg2k = kv_consensus_bucket(k);
            let key = format!(
                "spiral:heur:v1:list:sg:{}:c:{}:k:{}",
                if subgroup { 1 } else { 0 },
                lg2c,
                lg2k
            );
            match st_kv::redis_lrange(&url, &key, 0, -1) {
                Ok(list) => {
                    let w_def = std::env::var("SPIRAL_KV_SOFT_W")
                        .ok()
                        .and_then(|s| s.parse::<f32>().ok())
                        .unwrap_or(0.08);
                    let mut use2: Vec<bool> = vec![];
                    let mut wg: Vec<u32> = vec![];
                    let mut kl: Vec<u32> = vec![];
                    let mut ch: Vec<u32> = vec![];
                    let mut wts: Vec<f32> = vec![];
                    let mut parsed_len = 0usize;
                    for js in list.iter() {
                        if let Ok(v) = serde_json::from_str::<Value>(js) {
                            parsed_len += 1;
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
                        a.sort_by(|a, b| a.total_cmp(b));
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
                    let status = if list.is_empty() {
                        "empty"
                    } else if parsed_len == 0 {
                        "invalid_json"
                    } else if out.is_empty() {
                        "no_rules"
                    } else {
                        "rules"
                    };
                    emit_kv_consensus_soft_rules_meta(
                        rows,
                        cols,
                        k,
                        subgroup,
                        kind,
                        status,
                        true,
                        list.len(),
                        parsed_len,
                        out.len(),
                        w_med,
                    );
                }
                Err(_) => {
                    emit_kv_consensus_soft_rules_meta(
                        rows,
                        cols,
                        k,
                        subgroup,
                        kind,
                        "redis_error",
                        true,
                        0,
                        0,
                        0,
                        0.0,
                    );
                }
            }
        } else {
            emit_kv_consensus_soft_rules_meta(
                rows,
                cols,
                k,
                subgroup,
                kind,
                "missing_url",
                false,
                0,
                0,
                0,
                0.0,
            );
        }
    }
    out
}
#[cfg(not(feature = "logic"))]
pub fn kv_consensus_soft_rules(
    rows: u32,
    cols: u32,
    k: u32,
    subgroup: bool,
    kind: &'static str,
) -> Vec<()> {
    emit_kv_consensus_soft_rules_meta(
        rows,
        cols,
        k,
        subgroup,
        kind,
        "logic_disabled",
        std::env::var_os("REDIS_URL").is_some(),
        0,
        0,
        0,
        0.0,
    );
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
    use std::sync::{Arc, Mutex};

    fn with_env_var<T>(name: &str, value: Option<&str>, f: impl FnOnce() -> T) -> T {
        let previous = std::env::var(name).ok();
        match value {
            Some(value) => std::env::set_var(name, value),
            None => std::env::remove_var(name),
        }

        let result = catch_unwind(AssertUnwindSafe(f));
        match previous {
            Some(previous) => std::env::set_var(name, previous),
            None => std::env::remove_var(name),
        }

        match result {
            Ok(value) => value,
            Err(payload) => resume_unwind(payload),
        }
    }

    #[test]
    fn kv_consensus_soft_rules_emit_backend_meta_without_url() {
        let _observer_lock = crate::telemetry::tensor_observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let rules = with_env_var("REDIS_URL", None, || {
            kv_consensus_soft_rules(64, 8192, 32, false, "topk")
        });
        st_tensor::set_tensor_op_meta_observer(previous);

        assert!(rules.is_empty());
        let events = events.lock().unwrap();
        let meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "kv_consensus_soft_rules"
                    && data["kind"] == "st_core_kv_consensus_soft_rules"
                    && data["heuristic_kind"] == "topk"
                    && data["rows"] == 64
                    && data["cols"] == 8192
                    && data["k"] == 32
            })
            .expect("kv consensus soft-rules metadata event");
        assert_eq!(meta.1["backend"], "cpu");
        assert_eq!(meta.1["requested_backend"], "auto");
        assert_eq!(meta.1["logic_feature_enabled"], cfg!(feature = "logic"));
        assert_eq!(meta.1["kv_feature_enabled"], cfg!(feature = "kv-redis"));
        assert_eq!(meta.1["redis_url_present"], false);
        assert_eq!(meta.1["key_lg2c"], 13);
        assert_eq!(meta.1["key_lg2k"], 5);
        assert_eq!(meta.1["list_len"], 0);
        assert_eq!(meta.1["parsed_len"], 0);
        assert_eq!(meta.1["rule_count"], 0);
        let expected_status = if !cfg!(feature = "logic") {
            "logic_disabled"
        } else if !cfg!(feature = "kv-redis") {
            "feature_disabled"
        } else {
            "missing_url"
        };
        assert_eq!(meta.1["status"], expected_status);
    }
}
