// kdsl_bridge.rs (v1.8.5): parse env KDSL with algo
use super::wgpu_heuristics::Choice;
#[cfg(feature="logic")]
pub use st_logic::{SoftRule, Field, Value};

pub fn parse_env_dsl_plus(rows:u32, cols:u32, k:u32, subgroup:bool) -> (Option<Choice>, Vec<SoftRule>, u8) {
    let src = match std::env::var("SPIRAL_HEUR_K"){ Ok(s)=>s, Err(_)=>return (None, vec![], 0) };
    if src.trim().is_empty(){ return (None, vec![], 0); }
    #[cfg(feature="kdsl")]
    {
        let ctx = st_kdsl::Ctx{ r:rows, c:cols, k, sg:subgroup, sgc: if subgroup { 8 } else { 1 } };
        let out = match st_kdsl::eval_program(&src, &ctx){ Ok(o)=>o, Err(_)=> return (None, vec![], 0) };
        let mut hard = None;
        if out.hard.use_2ce.is_some() || out.hard.wg.is_some() || out.hard.kl.is_some() || out.hard.ch.is_some() {
            hard = Some(Choice{
                use_2ce: out.hard.use_2ce.unwrap_or(false),
                wg: out.hard.wg.unwrap_or(if subgroup{256}else{128}),
                kl: out.hard.kl.unwrap_or(if k>=64{32}else if k>=16{16}else{8}),
                ch: out.hard.ch.unwrap_or(if cols>16_384{8192}else{0}),
                algo_topk: 0,
            });
        }
        let mut soft = Vec::<SoftRule>::new();
        for r in out.soft {
            match r {
                st_kdsl::SoftRule::U2{val,w} => soft.push(SoftRule{ field:Field::Use2ce, value:Value::B(val), weight:w }),
                st_kdsl::SoftRule::Wg{val,w} => soft.push(SoftRule{ field:Field::Wg,     value:Value::U(val), weight:w }),
                st_kdsl::SoftRule::Kl{val,w} => soft.push(SoftRule{ field:Field::Kl,     value:Value::U(val), weight:w }),
                st_kdsl::SoftRule::Ch{val,w} => soft.push(SoftRule{ field:Field::Ch,     value:Value::U(val), weight:w }),
            }
        }
        // parse algo from env (simple scan)
        let mut algo: u8 = 0;
        let s = src.to_ascii_lowercase();
        if s.contains("algo:heap") || s.contains("algo=heap") || s.contains("algo:1") { algo = 1; }
        if s.contains("algo:bitonic") || s.contains("algo=bitonic") || s.contains("algo:2") { algo = 2; }
        return (hard, soft, algo);
    }
    (None, vec![], 0)
}

pub fn choose_from_kv(_rows:u32,_cols:u32,_k:u32,_subgroup:bool)->Option<Choice>{ None }
