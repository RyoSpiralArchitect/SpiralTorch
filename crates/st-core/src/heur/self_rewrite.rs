use std::time::{SystemTime, UNIX_EPOCH};
use serde_json::Value;
use statrs::distribution::{Normal, ContinuousCDF};

fn ablog_path()->std::path::PathBuf{
    if let Some(h)=dirs::home_dir(){ h.join(".spiraltorch").join("ablog.ndjson") }
    else { std::path::PathBuf::from("ablog.ndjson") }
}
fn heur_path()->std::path::PathBuf{
    if let Some(h)=dirs::home_dir(){ h.join(".spiraltorch").join("heur.kdsl") }
    else { std::path::PathBuf::from("heur.kdsl") }
}
fn wilson_lb(wins:u32,total:u32,alpha:f64)->f64{
    if total==0 {return 0.0}
    let p=wins as f64/total as f64;
    let z=Normal::new(0.0,1.0).unwrap().inverse_cdf(1.0-alpha/2.0);
    let n=total as f64; let den=1.0+z*z/n;
    let cen=p+z*z/(2.0*n);
    let adj=z*((p*(1.0-p)+z*z/(4.0*n))/n).sqrt();
    (cen-adj)/den
}

pub fn maybe_self_rewrite(){
    let enabled=std::env::var("ST_SR_ENABLED").ok().map(|v|v=="1").unwrap_or(false);
    if !enabled {return;}
    let min_n=std::env::var("ST_SR_MIN_N").ok().and_then(|s|s.parse::<u32>().ok()).unwrap_or(50);
    let cooldown_s=std::env::var("ST_SR_COOLDOWN_S").ok().and_then(|s|s.parse::<u64>().ok()).unwrap_or(120);
    let alpha=std::env::var("ST_SR_ALPHA").ok().and_then(|s|s.parse::<f64>().ok()).unwrap_or(0.05);
    let data=match std::fs::read_to_string(ablog_path()){Ok(s)=>s,Err(_)=>return};
    let now=SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    let mut agg: std::collections::HashMap<(String,String),(u32,u32)> = std::collections::HashMap::new();
    for line in data.lines().rev().take(10000){
        if let Ok(v)=serde_json::from_str::<Value>(line){
            let ts=v.get("ts").and_then(|x|x.as_u64()).unwrap_or(0);
            if now.saturating_sub(ts)>cooldown_s{ break; }
            let kind=v.get("kind").and_then(|x|x.as_str()).unwrap_or("");
            let choice=v.get("choice").and_then(|x|x.as_str()).unwrap_or("");
            let win=v.get("win").and_then(|x|x.as_bool()).unwrap_or(false);
            let e=agg.entry((kind.to_string(),choice.to_string())).or_insert((0,0));
            e.1+=1; if win {e.0+=1;}
        }
    }
    let mut lines=Vec::<String>::new();
    for ((kind,choice),(wins,total)) in agg{
        if total<min_n {continue;}
        if wilson_lb(wins,total,alpha)<0.6 {continue;}
        let s = match (kind.as_str(), choice.as_str()){
            ("topk","heap")    => r#"soft(algo,1,0.12,true)"#.to_string(),
            ("topk","bitonic") => r#"soft(algo,2,0.12,true)"#.to_string(),
            ("topk","kway")    => r#"soft(algo,3,0.12,true)"#.to_string(),
            ("midk","1ce")     => r#"soft(midk,1,0.10,true)"#.to_string(),
            ("midk","2ce")     => r#"soft(midk,2,0.10,true)"#.to_string(),
            ("bottomk","1ce")  => r#"soft(bottomk,1,0.08,true)"#.to_string(),
            ("bottomk","2ce")  => r#"soft(bottomk,2,0.08,true)"#.to_string(),
            _=>continue
        };
        lines.push(s);
    }
    if lines.is_empty(){return;}
    let _=std::fs::create_dir_all(heur_path().parent().unwrap());
    if let Ok(mut f)=std::fs::OpenOptions::new().create(true).append(true).open(heur_path()){
        use std::io::Write;
        for l in lines { let _=writeln!(f,"{}",l); }
    }
}
