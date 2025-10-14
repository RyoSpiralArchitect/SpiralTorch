use super::wgpu_heuristics::{Choice, DslOverrides};
#[cfg(feature = "logic")]
pub use st_logic::{Field, SoftRule, Value};
#[cfg(not(feature = "logic"))]
#[derive(Clone, Debug, Default)]
pub struct SoftRule;
#[cfg(not(feature = "logic"))]
#[derive(Clone, Debug)]
pub enum Field {}
#[cfg(not(feature = "logic"))]
#[derive(Clone, Debug)]
pub enum Value {}
#[cfg(feature = "kdsl")]
use serde::Deserialize;

#[cfg(feature = "kdsl")]
#[derive(Deserialize)]
struct SweetBands {
    small: u32,
    mid: u32,
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

#[allow(unused_variables)]
pub fn parse_env_dsl_plus_kind(
    rows: u32,
    cols: u32,
    k: u32,
    subgroup: bool,
    kind: &'static str,
) -> (Option<Choice>, Vec<SoftRule>, DslOverrides) {
    let src = match std::env::var("SPIRAL_HEUR_K") {
        Ok(s) => s,
        Err(_) => String::new(),
    };
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
    let mut ov = DslOverrides::default();
    if src.trim().is_empty() {
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
            Err(_) => return (None, vec![], ov),
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
                    .unwrap_or(((cols.max(1) + 1023) / 1024) as u32 * 1024),
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
                    field: Field::Use2ce,
                    value: Value::B(val),
                    weight: w,
                }),
                st_kdsl::SoftRule::Wg { val, w } => soft.push(SoftRule {
                    field: Field::Wg,
                    value: Value::U(val),
                    weight: w,
                }),
                st_kdsl::SoftRule::Kl { val, w } => soft.push(SoftRule {
                    field: Field::Kl,
                    value: Value::U(val),
                    weight: w,
                }),
                st_kdsl::SoftRule::Ch { val, w } => soft.push(SoftRule {
                    field: Field::Ch,
                    value: Value::U(val),
                    weight: w,
                }),
                st_kdsl::SoftRule::TileCols { val, w } => soft.push(SoftRule {
                    field: Field::Ctile,
                    value: Value::U(val),
                    weight: w,
                }),
                st_kdsl::SoftRule::Radix { val, w } => soft.push(SoftRule {
                    field: Field::Algo,
                    value: Value::U(val),
                    weight: w,
                }),
                st_kdsl::SoftRule::Segments { val, w } => soft.push(SoftRule {
                    field: Field::Kl,
                    value: Value::U(val),
                    weight: w,
                }),
                _ => {} // algo/midk/bottomk/ctile soft are currently consumed by higher-level selection (optional)
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
        return (hard, soft, ov);
    }
    (None, vec![], ov)
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
        let _ = (rows, cols, k, subgroup);
    }
    #[cfg(feature = "kv-redis")]
    {
        let url = std::env::var("REDIS_URL").ok()?;
        let lg2c = (32 - (cols.max(1) - 1).leading_zeros()) as u32;
        let lg2k = (32 - (k.max(1) - 1).leading_zeros()) as u32;
        let key = format!(
            "spiral:heur:v1:sg:{}:c:{}:k:{}",
            if subgroup { 1 } else { 0 },
            lg2c,
            lg2k
        );
        if let Ok(Some(js)) = st_kv::redis_get_raw(&url, &key) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&js) {
                let getb = |n: &str| v.get(n).and_then(|x| x.as_bool());
                let getu = |n: &str| v.get(n).and_then(|x| x.as_u64()).map(|u| u as u32);
                return Some(Choice {
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
                    tile_cols: getu("tile_cols")
                        .unwrap_or(((cols.max(1) + 1023) / 1024) as u32 * 1024),
                    radix: getu("radix").unwrap_or(if k.is_power_of_two() { 4 } else { 2 }),
                    segments: getu("segments").unwrap_or(if cols > 131_072 {
                        4
                    } else if cols > 32_768 {
                        2
                    } else {
                        1
                    }),
                });
            }
        }
    }
    None
}
