#[cfg(feature="logic")]
pub use st_logic::{SoftRule, Field, Value};

use super::heuristics::Choice;

pub fn parse_env_dsl(rows:u32, cols:u32, k:u32, subgroup:bool) -> (Option<Choice>, Vec<SoftRule>) {
    let src = match std::env::var("SPIRAL_HEUR_K"){ Ok(s)=>s, Err(_)=>return (None, vec![]) };
    if src.trim().is_empty(){ return (None, vec![]) }
    #[cfg(feature="kdsl")]
    {
        let ctx = st_kdsl::Ctx{ r:rows, c:cols, k, sg:subgroup };
        let out = match st_kdsl::eval_program(&src, &ctx){ Ok(o)=>o, Err(_)=> return (None, vec![]) };
        let hard = if out.hard.use_2ce.is_some() || out.hard.wg.is_some() || out.hard.kl.is_some() || out.hard.ch.is_some() {
            Some(Choice{
                use_2ce: out.hard.use_2ce.unwrap_or(false),
                wg: out.hard.wg.unwrap_or(if subgroup{256}else{128}),
                kl: out.hard.kl.unwrap_or(if k>=64{32}else if k>=16{16}else{8}),
                ch: out.hard.ch.unwrap_or(if cols>16_384{8192}else{0}),
            })
        } else { None };
        let mut soft = Vec::<SoftRule>::new();
        for r in out.soft {
            match r {
                st_kdsl::SoftRule::U2{val,w} => soft.push(SoftRule{ field:Field::Use2ce, value:Value::B(val), weight:w }),
                st_kdsl::SoftRule::Wg{val,w} => soft.push(SoftRule{ field:Field::Wg,     value:Value::U(val), weight:w }),
                st_kdsl::SoftRule::Kl{val,w} => soft.push(SoftRule{ field:Field::Kl,     value:Value::U(val), weight:w }),
                st_kdsl::SoftRule::Ch{val,w} => soft.push(SoftRule{ field:Field::Ch,     value:Value::U(val), weight:w }),
            }
        }
        return (hard, soft);
    }
    #[allow(unreachable_code)]
    (None, vec![])
}

pub fn choose_from_kv(rows:u32, cols:u32, k:u32, subgroup:bool)->Option<Choice>{
    #[cfg(feature="kv-redis")]
    {
        let url = std::env::var("REDIS_URL").ok()?;
        let lg2c = (32 - (cols.max(1)-1).leading_zeros()) as u32;
        let lg2k = (32 - (k.max(1)-1).leading_zeros()) as u32;
        let key = format!("spiral:heur:v1:sg:{}:c:{}:k:{}", if subgroup{1}else{0}, lg2c, lg2k);
        if let Ok(Some(v)) = st_kv::redis_get_choice(&url, &key) {
            return Some(Choice{ use_2ce:v.use_2ce, wg:v.wg, kl:v.kl, ch:v.ch });
        }
    }
    None
}
