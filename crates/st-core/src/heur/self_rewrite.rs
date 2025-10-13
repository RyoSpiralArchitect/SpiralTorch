use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};

#[derive(Deserialize, Serialize, Default)]
struct SweetBands { small:u32, mid:u32, large:u32 }
#[derive(Deserialize, Serialize, Default)]
struct SweetFile { topk:Option<SweetBands>, midk:Option<SweetBands>, bottomk:Option<SweetBands> }

fn home_dir() -> std::path::PathBuf {
    if let Some(h)=dirs::home_dir(){ h } else { std::path::PathBuf::from(".") }
}
fn heur_path()->std::path::PathBuf { home_dir().join(".spiraltorch").join("heur.kdsl") }
fn sweet_path()->std::path::PathBuf { home_dir().join(".spiraltorch").join("sweet.json") }
fn ablog_path()->std::path::PathBuf { home_dir().join(".spiraltorch").join("ablog.ndjson") }

fn wilson_lower_bound(wins:u32, total:u32, alpha:f64)->f64{
    if total==0 { return 0.0 }
    let p = wins as f64 / total as f64;
    let z = Normal::new(0.0,1.0).unwrap().inverse_cdf(1.0 - alpha/2.0);
    let n = total as f64;
    let denom = 1.0 + z*z/n;
    let centre = p + z*z/(2.0*n);
    let adj = z * ((p*(1.0-p) + z*z/(4.0*n))/n).sqrt();
    (centre - adj) / denom
}

pub fn maybe_self_rewrite() {
    let enabled = std::env::var("ST_SR_ENABLED").ok().map(|v| v=="1").unwrap_or(false);
    if !enabled { return; }
    let min_n = std::env::var("ST_SR_MIN_N").ok().and_then(|s| s.parse::<u32>().ok()).unwrap_or(50);
    let cooldown_s = std::env::var("ST_SR_COOLDOWN_S").ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(120);
    let alpha = std::env::var("ST_SR_ALPHA").ok().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.05);

    let path = ablog_path();
    let data = match std::fs::read_to_string(&path){ Ok(s)=>s, Err(_)=>return };
    // aggregate last window
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    let mut buckets: std::collections::HashMap<(String,String), (u32,u32)> = std::collections::HashMap::new();
    for line in data.lines().rev().take(5000) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            let ts = v.get("ts").and_then(|x|x.as_u64()).unwrap_or(0);
            if now.saturating_sub(ts) > cooldown_s { break; }
            let kind = v.get("kind").and_then(|x|x.as_str()).unwrap_or("");
            let choice = v.get("choice").and_then(|x|x.as_str()).unwrap_or("");
            let win = v.get("win").and_then(|x|x.as_bool()).unwrap_or(false);
            let key=(kind.to_string(), choice.to_string());
            let e = buckets.entry(key).or_insert((0,0));
            e.1 += 1; if win { e.0 += 1; }
        }
    }
    let mut adds=Vec::<String>::new();
    for ((kind, choice),(wins,total)) in buckets.into_iter() {
        if total < min_n { continue; }
        let lb = wilson_lower_bound(wins,total,alpha);
        if lb < 0.6 { continue; } // guard
        // append soft rule
        let line = match (kind.as_str(), choice.as_str()) {
            ("topk","heap")    => r#"soft(algo,1,0.12,true)"#.to_string(),
            ("topk","bitonic") => r#"soft(algo,2,0.12,true)"#.to_string(),
            ("midk","1ce")     => r#"soft(midk,1,0.10,true)"#.to_string(),
            ("midk","2ce")     => r#"soft(midk,2,0.10,true)"#.to_string(),
            ("bottomk","1ce")  => r#"soft(bottomk,1,0.08,true)"#.to_string(),
            ("bottomk","2ce")  => r#"soft(bottomk,2,0.08,true)"#.to_string(),
            _=>continue
        };
        adds.push(line);
    }
    if adds.is_empty(){ return; }
    let _ = std::fs::create_dir_all(heur_path().parent().unwrap());
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(heur_path()).unwrap();
    for s in adds { use std::io::Write; let _ = writeln!(f, "{}", s); }
}
