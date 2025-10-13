#[cfg(feature="logic")]
use st_logic::{SoftRule, Field, Value};
use serde_json::Value;

#[cfg(feature="logic")]
pub fn kv_consensus_soft_rules(rows:u32, cols:u32, k:u32, subgroup:bool, _kind:&'static str) -> Vec<SoftRule> {
    let mut out=Vec::<SoftRule>::new();
    #[cfg(feature="kv-redis")]
    {
        if let Ok(url)=std::env::var("REDIS_URL"){
            let lg2c=(32-(cols.max(1)-1).leading_zeros()) as u32;
            let lg2k=(32-(k.max(1)-1).leading_zeros()) as u32;
            let key=format!("spiral:heur:v1:list:sg:{}:c:{}:k:{}", if subgroup{1}else{0}, lg2c, lg2k);
            if let Ok(list)=st_kv::redis_lrange(&url,&key,0,-1){
                let w_def=std::env::var("SPIRAL_KV_SOFT_W").ok().and_then(|s|s.parse::<f32>().ok()).unwrap_or(0.08);
                let mut use2:Vec<bool>=vec![]; let mut wg:Vec<u32>=vec![]; let mut kl:Vec<u32>=vec![]; let mut ch:Vec<u32>=vec![]; let mut wts:Vec<f32>=vec![];
                for js in list.iter(){
                    if let Ok(v)=serde_json::from_str::<Value>(js){
                        if let Some(b)=v.get("use_2ce").and_then(|x|x.as_bool()){ use2.push(b); }
                        if let Some(u)=v.get("wg").and_then(|x|x.as_u64()){ wg.push(u as u32); }
                        if let Some(u)=v.get("kl").and_then(|x|x.as_u64()){ kl.push(u as u32); }
                        if let Some(u)=v.get("ch").and_then(|x|x.as_u64()){ ch.push(u as u32); }
                        let w=v.get("weight").and_then(|x|x.as_f64()).map(|f|f as f32).unwrap_or(w_def);
                        wts.push(w);
                    }
                }
                let w_med=if wts.is_empty(){ w_def } else { let mut a=wts.clone(); a.sort_by(|a,b|a.partial_cmp(b).unwrap()); a[a.len()/2] };
                if let Some(b)=majority_bool(&use2){ out.push(SoftRule{ field:Field::Use2ce, value:Value::B(b), weight:w_med }); }
                if let Some(u)=median_u32(&wg){ out.push(SoftRule{ field:Field::Wg, value:Value::U(u), weight:w_med }); }
                if let Some(u)=median_u32(&kl){ out.push(SoftRule{ field:Field::Kl, value:Value::U(u), weight:w_med }); }
                if let Some(u)=median_u32(&ch){ out.push(SoftRule{ field:Field::Ch, value:Value::U(u), weight:w_med }); }
            }
        }
    }
    out
}
#[cfg(not(feature="logic"))]
pub fn kv_consensus_soft_rules(_r:u32,_c:u32,_k:u32,_sg:bool,_kind:&'static str)->Vec<()> { Vec::new() }
#[cfg(feature="logic")]
fn median_u32(v:&[u32])->Option<u32>{ if v.is_empty(){None}else{ let mut a=v.to_vec(); a.sort_unstable(); Some(a[a.len()/2]) } }
#[cfg(feature="logic")]
fn majority_bool(v:&[bool])->Option<bool>{ if v.is_empty(){None}else{ let t=v.iter().filter(|&&b|b).count(); Some(t*2>=v.len()) } }
