// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::soft_logic::SoftRule;
use super::wgpu_heuristics::{Choice, DslOverrides};
#[cfg(feature = "kdsl")]
use super::wgpu_heuristics::{
    SOFT_NAME_ALGO, SOFT_NAME_CH, SOFT_NAME_CTILE, SOFT_NAME_KL, SOFT_NAME_MODE_BOTTOMK,
    SOFT_NAME_MODE_MIDK, SOFT_NAME_RADIX, SOFT_NAME_SEGMENTS, SOFT_NAME_TILE_COLS,
    SOFT_NAME_USE2CE, SOFT_NAME_WG,
};
#[cfg(feature = "kdsl")]
use serde::Deserialize;
use st_tensor::{emit_tensor_op, emit_tensor_op_meta};

#[cfg(feature = "kdsl")]
#[derive(Deserialize)]
struct SweetBands {
    small: u32,
    mid: u32,
    #[allow(dead_code)]
    large: u32,
}
#[cfg(feature = "kdsl")]
#[derive(Deserialize)]
struct SweetFile {
    topk: Option<SweetBands>,
    midk: Option<SweetBands>,
    bottomk: Option<SweetBands>,
}

#[cfg(feature = "kdsl")]
fn sweet_kc(kind: &str, k: u32) -> u32 {
    let path = if let Some(h) = dirs::home_dir() {
        h.join(".spiraltorch").join("sweet.json")
    } else {
        std::path::PathBuf::from("sweet.json")
    };
    if let Ok(s) = std::fs::read_to_string(path) {
        if let Ok(sf) = serde_json::from_str::<SweetFile>(&s) {
            let bands = match kind {
                "topk" => sf.topk,
                "midk" => sf.midk,
                "bottomk" => sf.bottomk,
                _ => None,
            };
            if let Some(b) = bands {
                if k <= b.small {
                    return 1;
                }
                if k <= b.mid {
                    return 2;
                }
                return 3;
            }
        }
    }
    if k <= 1024 {
        1
    } else if k <= 16384 {
        2
    } else {
        3
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "KDSL bridge metadata mirrors the heuristic request tuple"
)]
fn emit_kdsl_env_bridge_meta(
    rows: u32,
    cols: u32,
    k: u32,
    subgroup: bool,
    kind: &'static str,
    kc: u32,
    src: &str,
    status: &'static str,
    hard_present: bool,
    soft_count: usize,
    overrides: DslOverrides,
) {
    emit_tensor_op(
        "kdsl_env_bridge",
        &[rows as usize, cols as usize, k as usize],
        &[usize::from(hard_present), soft_count],
    );
    emit_tensor_op_meta("kdsl_env_bridge", || {
        serde_json::json!({
            "kind": "st_core_kdsl_env_bridge",
            "backend": "cpu",
            "requested_backend": "auto",
            "rows": rows,
            "cols": cols,
            "k": k,
            "subgroup": subgroup,
            "heuristic_kind": kind,
            "kc": kc,
            "source_present": !src.trim().is_empty(),
            "source_len": src.trim().len(),
            "kdsl_feature_enabled": cfg!(feature = "kdsl"),
            "status": status,
            "hard_present": hard_present,
            "soft_count": soft_count,
            "override_algo_topk": overrides.algo_topk,
            "override_mode_midk": overrides.mode_midk,
            "override_mode_bottomk": overrides.mode_bottomk,
            "override_ctile": overrides.ctile,
            "override_tile_cols": overrides.tile_cols,
            "override_radix": overrides.radix,
            "override_segments": overrides.segments,
        })
    });
}

fn kv_bridge_lg2_bucket(value: u32) -> u32 {
    32 - (value.max(1) - 1).leading_zeros()
}

fn emit_kdsl_kv_bridge_meta(
    rows: u32,
    cols: u32,
    k: u32,
    subgroup: bool,
    status: &'static str,
    redis_url_present: bool,
    choice: Option<Choice>,
) {
    let lg2c = kv_bridge_lg2_bucket(cols);
    let lg2k = kv_bridge_lg2_bucket(k);
    emit_tensor_op(
        "kdsl_kv_bridge",
        &[rows as usize, cols as usize, k as usize],
        &[usize::from(choice.is_some()), lg2c as usize, lg2k as usize],
    );
    emit_tensor_op_meta("kdsl_kv_bridge", || {
        let choice = choice.unwrap_or(Choice {
            use_2ce: false,
            wg: 0,
            kl: 0,
            ch: 0,
            algo_topk: 0,
            ctile: 0,
            mode_midk: 0,
            mode_bottomk: 0,
            tile_cols: 0,
            radix: 0,
            segments: 0,
        });
        serde_json::json!({
            "kind": "st_core_kdsl_kv_bridge",
            "backend": "cpu",
            "requested_backend": "auto",
            "rows": rows,
            "cols": cols,
            "k": k,
            "subgroup": subgroup,
            "kv_feature_enabled": cfg!(feature = "kv-redis"),
            "redis_url_present": redis_url_present,
            "status": status,
            "key_lg2c": lg2c,
            "key_lg2k": lg2k,
            "selected": status == "hit",
            "choice_use_2ce": choice.use_2ce,
            "choice_wg": choice.wg,
            "choice_kl": choice.kl,
            "choice_ch": choice.ch,
            "choice_algo_topk": choice.algo_topk,
            "choice_mode_midk": choice.mode_midk,
            "choice_mode_bottomk": choice.mode_bottomk,
            "choice_ctile": choice.ctile,
            "choice_tile_cols": choice.tile_cols,
            "choice_radix": choice.radix,
            "choice_segments": choice.segments,
        })
    });
}

#[allow(unused_variables)]
pub fn parse_env_dsl_plus_kind(
    rows: u32,
    cols: u32,
    k: u32,
    subgroup: bool,
    kind: &'static str,
) -> (Option<Choice>, Vec<SoftRule>, DslOverrides) {
    let src = std::env::var("SPIRAL_HEUR_K").unwrap_or_default();
    #[allow(unused_mut)]
    let kc = {
        #[cfg(feature = "kdsl")]
        {
            sweet_kc(kind, k)
        }
        #[cfg(not(feature = "kdsl"))]
        {
            if k <= 1024 {
                1
            } else if k <= 16_384 {
                2
            } else {
                3
            }
        }
    };
    #[cfg(feature = "kdsl")]
    let mut ov = DslOverrides::default();
    #[cfg(not(feature = "kdsl"))]
    let ov = DslOverrides::default();
    if src.trim().is_empty() {
        emit_kdsl_env_bridge_meta(
            rows, cols, k, subgroup, kind, kc, &src, "empty", false, 0, ov,
        );
        return (None, vec![], ov);
    }
    #[cfg(feature = "kdsl")]
    {
        let kc = sweet_kc(kind, k);
        let ctx = st_kdsl::Ctx {
            r: rows,
            c: cols,
            k,
            sg: subgroup,
            sgc: if subgroup { 8 } else { 1 },
            kc,
            tile_cols: ((cols.max(1) + 255) / 256) as u32,
            radix: if k.is_power_of_two() { 4 } else { 2 },
            segments: if cols > 131_072 {
                4
            } else if cols > 32_768 {
                2
            } else {
                1
            },
        };
        let out = match st_kdsl::eval_program(&src, &ctx) {
            Ok(o) => o,
            Err(_) => {
                emit_kdsl_env_bridge_meta(
                    rows,
                    cols,
                    k,
                    subgroup,
                    kind,
                    kc,
                    &src,
                    "eval_error",
                    false,
                    0,
                    ov,
                );
                return (None, vec![], ov);
            }
        };
        let mut hard = None;
        if out.hard.use_2ce.is_some()
            || out.hard.wg.is_some()
            || out.hard.kl.is_some()
            || out.hard.ch.is_some()
            || out.hard.algo.is_some()
            || out.hard.midk.is_some()
            || out.hard.bottomk.is_some()
            || out.hard.ctile.is_some()
            || out.hard.tile_cols.is_some()
            || out.hard.radix.is_some()
            || out.hard.segments.is_some()
        {
            hard = Some(Choice {
                use_2ce: out.hard.use_2ce.unwrap_or(false),
                wg: out.hard.wg.unwrap_or(if subgroup { 256 } else { 128 }),
                kl: out.hard.kl.unwrap_or(if k >= 64 {
                    32
                } else if k >= 16 {
                    16
                } else {
                    8
                }),
                ch: out.hard.ch.unwrap_or(if cols > 16_384 { 8192 } else { 0 }),
                algo_topk: out.hard.algo.unwrap_or(0),
                ctile: out.hard.ctile.unwrap_or(0),
                mode_midk: out.hard.midk.unwrap_or(0),
                mode_bottomk: out.hard.bottomk.unwrap_or(0),
                tile_cols: out
                    .hard
                    .tile_cols
                    .unwrap_or(cols.max(1).div_ceil(1024) * 1024),
                radix: out
                    .hard
                    .radix
                    .unwrap_or(if k.is_power_of_two() { 4 } else { 2 }),
                segments: out.hard.segments.unwrap_or(if cols > 131_072 {
                    4
                } else if cols > 32_768 {
                    2
                } else {
                    1
                }),
            });
        }
        let mut soft = Vec::<SoftRule>::new();
        for r in out.soft {
            match r {
                st_kdsl::SoftRule::U2 { val, w } => soft.push(SoftRule {
                    name: SOFT_NAME_USE2CE,
                    weight: w,
                    score: if val { 1.0 } else { -1.0 },
                }),
                st_kdsl::SoftRule::Wg { val, w } => soft.push(SoftRule {
                    name: SOFT_NAME_WG,
                    weight: w,
                    score: val as f32,
                }),
                st_kdsl::SoftRule::Kl { val, w } => soft.push(SoftRule {
                    name: SOFT_NAME_KL,
                    weight: w,
                    score: val as f32,
                }),
                st_kdsl::SoftRule::Ch { val, w } => soft.push(SoftRule {
                    name: SOFT_NAME_CH,
                    weight: w,
                    score: val as f32,
                }),
                st_kdsl::SoftRule::Algo { val, w } => soft.push(SoftRule {
                    name: SOFT_NAME_ALGO,
                    weight: w,
                    score: val as f32,
                }),
                st_kdsl::SoftRule::Midk { val, w } => soft.push(SoftRule {
                    name: SOFT_NAME_MODE_MIDK,
                    weight: w,
                    score: val as f32,
                }),
                st_kdsl::SoftRule::Bottomk { val, w } => soft.push(SoftRule {
                    name: SOFT_NAME_MODE_BOTTOMK,
                    weight: w,
                    score: val as f32,
                }),
                st_kdsl::SoftRule::Ctile { val, w } => soft.push(SoftRule {
                    name: SOFT_NAME_CTILE,
                    weight: w,
                    score: val as f32,
                }),
                st_kdsl::SoftRule::TileCols { val, w } => soft.push(SoftRule {
                    name: SOFT_NAME_TILE_COLS,
                    weight: w,
                    score: val as f32,
                }),
                st_kdsl::SoftRule::Radix { val, w } => soft.push(SoftRule {
                    name: SOFT_NAME_RADIX,
                    weight: w,
                    score: val as f32,
                }),
                st_kdsl::SoftRule::Segments { val, w } => soft.push(SoftRule {
                    name: SOFT_NAME_SEGMENTS,
                    weight: w,
                    score: val as f32,
                }),
            }
        }
        if let Some(a) = out.hard.algo {
            ov.algo_topk = a;
        }
        if let Some(m) = out.hard.midk {
            ov.mode_midk = m;
        }
        if let Some(m) = out.hard.bottomk {
            ov.mode_bottomk = m;
        }
        if let Some(t) = out.hard.ctile {
            ov.ctile = t;
        }
        if let Some(t) = out.hard.tile_cols {
            ov.tile_cols = t;
        }
        if let Some(r) = out.hard.radix {
            ov.radix = r;
        }
        if let Some(s) = out.hard.segments {
            ov.segments = s;
        }
        emit_kdsl_env_bridge_meta(
            rows,
            cols,
            k,
            subgroup,
            kind,
            kc,
            &src,
            "evaluated",
            hard.is_some(),
            soft.len(),
            ov,
        );
        return (hard, soft, ov);
    }
    #[cfg(not(feature = "kdsl"))]
    {
        emit_kdsl_env_bridge_meta(
            rows,
            cols,
            k,
            subgroup,
            kind,
            kc,
            &src,
            "feature_disabled",
            false,
            0,
            ov,
        );
        (None, Vec::new(), ov)
    }
}

/// Debug helper that mirrors [`parse_env_dsl_plus_kind`] but also returns a structured KDSL trace.
#[cfg(feature = "kdsl")]
pub fn parse_env_dsl_plus_kind_explain(
    rows: u32,
    cols: u32,
    k: u32,
    subgroup: bool,
    kind: &'static str,
    max_events: usize,
) -> (
    Option<Choice>,
    Vec<SoftRule>,
    DslOverrides,
    Option<st_kdsl::KdslEvaluationTrace>,
) {
    let src = std::env::var("SPIRAL_HEUR_K").unwrap_or_default();
    let kc = sweet_kc(kind, k);
    let mut ov = DslOverrides::default();
    if src.trim().is_empty() {
        emit_kdsl_env_bridge_meta(
            rows, cols, k, subgroup, kind, kc, &src, "empty", false, 0, ov,
        );
        return (None, vec![], ov, None);
    }

    let ctx = st_kdsl::Ctx {
        r: rows,
        c: cols,
        k,
        sg: subgroup,
        sgc: if subgroup { 8 } else { 1 },
        kc,
        tile_cols: ((cols.max(1) + 255) / 256) as u32,
        radix: if k.is_power_of_two() { 4 } else { 2 },
        segments: if cols > 131_072 {
            4
        } else if cols > 32_768 {
            2
        } else {
            1
        },
    };

    let (out, trace) = match st_kdsl::eval_program_with_trace(&src, &ctx, max_events.max(1)) {
        Ok(result) => result,
        Err(_) => {
            emit_kdsl_env_bridge_meta(
                rows,
                cols,
                k,
                subgroup,
                kind,
                kc,
                &src,
                "eval_error",
                false,
                0,
                ov,
            );
            return (None, vec![], ov, None);
        }
    };

    let mut hard = None;
    if out.hard.use_2ce.is_some()
        || out.hard.wg.is_some()
        || out.hard.kl.is_some()
        || out.hard.ch.is_some()
        || out.hard.algo.is_some()
        || out.hard.midk.is_some()
        || out.hard.bottomk.is_some()
        || out.hard.ctile.is_some()
        || out.hard.tile_cols.is_some()
        || out.hard.radix.is_some()
        || out.hard.segments.is_some()
    {
        hard = Some(Choice {
            use_2ce: out.hard.use_2ce.unwrap_or(false),
            wg: out.hard.wg.unwrap_or(if subgroup { 256 } else { 128 }),
            kl: out.hard.kl.unwrap_or(if k >= 64 {
                32
            } else if k >= 16 {
                16
            } else {
                8
            }),
            ch: out.hard.ch.unwrap_or(if cols > 16_384 { 8192 } else { 0 }),
            algo_topk: out.hard.algo.unwrap_or(0),
            ctile: out.hard.ctile.unwrap_or(0),
            mode_midk: out.hard.midk.unwrap_or(0),
            mode_bottomk: out.hard.bottomk.unwrap_or(0),
            tile_cols: out
                .hard
                .tile_cols
                .unwrap_or(cols.max(1).div_ceil(1024) * 1024),
            radix: out
                .hard
                .radix
                .unwrap_or(if k.is_power_of_two() { 4 } else { 2 }),
            segments: out.hard.segments.unwrap_or(if cols > 131_072 {
                4
            } else if cols > 32_768 {
                2
            } else {
                1
            }),
        });
    }

    let mut soft = Vec::<SoftRule>::new();
    for r in out.soft {
        match r {
            st_kdsl::SoftRule::U2 { val, w } => soft.push(SoftRule {
                name: SOFT_NAME_USE2CE,
                weight: w,
                score: if val { 1.0 } else { -1.0 },
            }),
            st_kdsl::SoftRule::Wg { val, w } => soft.push(SoftRule {
                name: SOFT_NAME_WG,
                weight: w,
                score: val as f32,
            }),
            st_kdsl::SoftRule::Kl { val, w } => soft.push(SoftRule {
                name: SOFT_NAME_KL,
                weight: w,
                score: val as f32,
            }),
            st_kdsl::SoftRule::Ch { val, w } => soft.push(SoftRule {
                name: SOFT_NAME_CH,
                weight: w,
                score: val as f32,
            }),
            st_kdsl::SoftRule::Algo { val, w } => soft.push(SoftRule {
                name: SOFT_NAME_ALGO,
                weight: w,
                score: val as f32,
            }),
            st_kdsl::SoftRule::Midk { val, w } => soft.push(SoftRule {
                name: SOFT_NAME_MODE_MIDK,
                weight: w,
                score: val as f32,
            }),
            st_kdsl::SoftRule::Bottomk { val, w } => soft.push(SoftRule {
                name: SOFT_NAME_MODE_BOTTOMK,
                weight: w,
                score: val as f32,
            }),
            st_kdsl::SoftRule::Ctile { val, w } => soft.push(SoftRule {
                name: SOFT_NAME_CTILE,
                weight: w,
                score: val as f32,
            }),
            st_kdsl::SoftRule::TileCols { val, w } => soft.push(SoftRule {
                name: SOFT_NAME_TILE_COLS,
                weight: w,
                score: val as f32,
            }),
            st_kdsl::SoftRule::Radix { val, w } => soft.push(SoftRule {
                name: SOFT_NAME_RADIX,
                weight: w,
                score: val as f32,
            }),
            st_kdsl::SoftRule::Segments { val, w } => soft.push(SoftRule {
                name: SOFT_NAME_SEGMENTS,
                weight: w,
                score: val as f32,
            }),
        }
    }

    if let Some(a) = out.hard.algo {
        ov.algo_topk = a;
    }
    if let Some(m) = out.hard.midk {
        ov.mode_midk = m;
    }
    if let Some(m) = out.hard.bottomk {
        ov.mode_bottomk = m;
    }
    if let Some(t) = out.hard.ctile {
        ov.ctile = t;
    }
    if let Some(t) = out.hard.tile_cols {
        ov.tile_cols = t;
    }
    if let Some(r) = out.hard.radix {
        ov.radix = r;
    }
    if let Some(s) = out.hard.segments {
        ov.segments = s;
    }

    emit_kdsl_env_bridge_meta(
        rows,
        cols,
        k,
        subgroup,
        kind,
        kc,
        &src,
        "evaluated",
        hard.is_some(),
        soft.len(),
        ov,
    );

    (hard, soft, ov, Some(trace))
}

pub fn parse_env_dsl(
    rows: u32,
    cols: u32,
    k: u32,
    subgroup: bool,
) -> (Option<Choice>, Vec<SoftRule>) {
    let (hard, soft, _ov) = parse_env_dsl_plus_kind(rows, cols, k, subgroup, "topk");
    (hard, soft)
}

pub fn choose_from_kv(rows: u32, cols: u32, k: u32, subgroup: bool) -> Option<Choice> {
    #[cfg(not(feature = "kv-redis"))]
    {
        emit_kdsl_kv_bridge_meta(
            rows,
            cols,
            k,
            subgroup,
            "feature_disabled",
            std::env::var_os("REDIS_URL").is_some(),
            None,
        );
    }
    #[cfg(feature = "kv-redis")]
    {
        let url = match std::env::var("REDIS_URL") {
            Ok(url) => url,
            Err(_) => {
                emit_kdsl_kv_bridge_meta(rows, cols, k, subgroup, "missing_url", false, None);
                return None;
            }
        };
        let lg2c = kv_bridge_lg2_bucket(cols);
        let lg2k = kv_bridge_lg2_bucket(k);
        let key = format!(
            "spiral:heur:v1:sg:{}:c:{}:k:{}",
            if subgroup { 1 } else { 0 },
            lg2c,
            lg2k
        );
        match st_kv::redis_get_raw(&url, &key) {
            Ok(Some(js)) => {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&js) {
                    let getb = |n: &str| v.get(n).and_then(|x| x.as_bool());
                    let getu = |n: &str| v.get(n).and_then(|x| x.as_u64()).map(|u| u as u32);
                    let choice = Choice {
                        use_2ce: getb("use_2ce").unwrap_or(false),
                        wg: getu("wg").unwrap_or(if subgroup { 256 } else { 128 }),
                        kl: getu("kl").unwrap_or(if k >= 64 {
                            32
                        } else if k >= 16 {
                            16
                        } else {
                            8
                        }),
                        ch: getu("ch").unwrap_or(if cols > 16_384 { 8192 } else { 0 }),
                        algo_topk: getu("algo_topk").unwrap_or(0) as u8,
                        ctile: getu("ctile").unwrap_or(0),
                        mode_midk: getu("mode_midk").unwrap_or(0) as u8,
                        mode_bottomk: getu("mode_bottomk").unwrap_or(0) as u8,
                        tile_cols: getu("tile_cols").unwrap_or(cols.max(1).div_ceil(1024) * 1024),
                        radix: getu("radix").unwrap_or(if k.is_power_of_two() { 4 } else { 2 }),
                        segments: getu("segments").unwrap_or(if cols > 131_072 {
                            4
                        } else if cols > 32_768 {
                            2
                        } else {
                            1
                        }),
                    };
                    emit_kdsl_kv_bridge_meta(rows, cols, k, subgroup, "hit", true, Some(choice));
                    return Some(choice);
                }
                emit_kdsl_kv_bridge_meta(rows, cols, k, subgroup, "invalid_json", true, None);
            }
            Ok(None) => {
                emit_kdsl_kv_bridge_meta(rows, cols, k, subgroup, "miss", true, None);
            }
            Err(_) => {
                emit_kdsl_kv_bridge_meta(rows, cols, k, subgroup, "redis_error", true, None);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
    use std::sync::{Arc, Mutex, OnceLock};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

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

    fn with_spiral_heur_k<T>(value: Option<&str>, f: impl FnOnce() -> T) -> T {
        with_env_var("SPIRAL_HEUR_K", value, f)
    }

    #[test]
    fn kdsl_env_bridge_emits_backend_meta() {
        let _env_lock = env_lock();
        let _observer_lock = crate::telemetry::tensor_observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let (hard, soft, overrides) = with_spiral_heur_k(Some("not valid kdsl program"), || {
            parse_env_dsl_plus_kind(32, 4096, 128, true, "topk")
        });
        st_tensor::set_tensor_op_meta_observer(previous);

        assert!(hard.is_none());
        assert!(soft.is_empty());
        assert_eq!(overrides.algo_topk, 0);

        let events = events.lock().unwrap();
        let meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "kdsl_env_bridge"
                    && data["kind"] == "st_core_kdsl_env_bridge"
                    && data["rows"] == 32
                    && data["cols"] == 4096
                    && data["k"] == 128
                    && data["heuristic_kind"] == "topk"
            })
            .expect("kdsl env bridge metadata event");
        assert_eq!(meta.1["backend"], "cpu");
        assert_eq!(meta.1["requested_backend"], "auto");
        assert_eq!(meta.1["source_present"], true);
        assert_eq!(meta.1["source_len"], "not valid kdsl program".len());
        assert_eq!(meta.1["kdsl_feature_enabled"], cfg!(feature = "kdsl"));
        assert_eq!(meta.1["hard_present"], false);
        assert_eq!(meta.1["soft_count"], 0);
        assert_eq!(meta.1["override_algo_topk"], 0);
        if cfg!(feature = "kdsl") {
            assert_eq!(meta.1["status"], "eval_error");
        } else {
            assert_eq!(meta.1["status"], "feature_disabled");
        }
    }

    #[test]
    fn kdsl_kv_bridge_emits_backend_meta_without_url() {
        let _env_lock = env_lock();
        let _observer_lock = crate::telemetry::tensor_observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let choice = with_env_var("REDIS_URL", None, || choose_from_kv(64, 8192, 32, false));
        st_tensor::set_tensor_op_meta_observer(previous);

        assert!(choice.is_none());

        let events = events.lock().unwrap();
        let meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "kdsl_kv_bridge"
                    && data["kind"] == "st_core_kdsl_kv_bridge"
                    && data["rows"] == 64
                    && data["cols"] == 8192
                    && data["k"] == 32
            })
            .expect("kdsl kv bridge metadata event");
        assert_eq!(meta.1["backend"], "cpu");
        assert_eq!(meta.1["requested_backend"], "auto");
        assert_eq!(meta.1["kv_feature_enabled"], cfg!(feature = "kv-redis"));
        assert_eq!(meta.1["redis_url_present"], false);
        assert_eq!(meta.1["key_lg2c"], 13);
        assert_eq!(meta.1["key_lg2k"], 5);
        assert_eq!(meta.1["selected"], false);
        assert_eq!(meta.1["choice_wg"], 0);
        if cfg!(feature = "kv-redis") {
            assert_eq!(meta.1["status"], "missing_url");
        } else {
            assert_eq!(meta.1["status"], "feature_disabled");
        }
    }
}
