use std::{fs, path::PathBuf, io::Write, time::{SystemTime, UNIX_EPOCH}};
use serde::{Serialize, Deserialize};

fn home_dir()->Option<PathBuf>{ std::env::var("HOME").ok().map(|h| PathBuf::from(h)) }
fn dir()->Option<PathBuf>{ let mut p = home_dir()?; p.push(".spiraltorch"); let _=fs::create_dir_all(&p); Some(p) }
fn file(name:&str)->Option<PathBuf>{ let mut p = dir()?; p.push(name); Some(p) }

#[derive(Serialize,Deserialize,Clone,Debug)]
pub struct Obs {
    pub ts: u64,            // unix sec
    pub sg: bool,
    pub lg2c: u32,
    pub lg2k: u32,
    pub pass: String,       // "topk" | "bottomk" | "midk"
    pub choice: String,     // e.g., "wg=256" | "u2=true" | "scan_wg=256"
    pub time_ms: f32,
}

fn now()->u64{ SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() }

pub fn record_observation(o: &Obs){
    if std::env::var("ST_SELF_REWRITE").ok().as_deref() != Some("1") { return; }
    if let Some(p)=file("observations.jsonl"){
        if let Ok(mut f)=fs::OpenOptions::new().create(true).append(true).open(&p){
            let _ = writeln!(f, "{}", serde_json::to_string(o).unwrap());
        }
    }
}

fn read_observations()->Vec<Obs>{
    let p = match file("observations.jsonl"){ Some(x)=>x, None=>return vec![] };
    let s = match fs::read_to_string(p){ Ok(v)=>v, Err(_)=>return vec![] };
    s.lines().filter_map(|l| serde_json::from_str::<Obs>(l).ok()).collect()
}

fn med(mut v:Vec<f32>)->Option<f32>{ if v.is_empty(){ return None } v.sort_by(|a,b| a.partial_cmp(b).unwrap()); Some(v[v.len()/2]) }

fn bucket(o:&Obs)->(bool,u32,u32,String){ (o.sg, o.lg2c, o.lg2k, o.pass.clone()) }

fn improve_enough(new:f32, base:f32, pct:f32)->bool{ new < base * (1.0 - pct) }

fn should_cooldown(bucket_key:&(bool,u32,u32,String), cooldown:u64)->bool{
    let p = match file("rewrite_state.json"){ Some(x)=>x, None=>return false };
    let s = fs::read_to_string(&p).unwrap_or_default();
    let mut last: std::collections::HashMap<String,u64> = serde_json::from_str(&s).unwrap_or_default();
    let key = format!("{:?}", bucket_key);
    let now_s = now();
    if let Some(prev) = last.get(&key){
        if now_s.saturating_sub(*prev) < cooldown { return true; }
    }
    last.insert(key, now_s);
    let _ = fs::write(&p, serde_json::to_string(&last).unwrap());
    false
}

fn heur_path()->Option<PathBuf>{ file("heur.kdsl") }

fn clamp_file_lines(path:&PathBuf, max_lines:usize){
    if let Ok(s)=fs::read_to_string(path){
        let mut lines: Vec<&str> = s.lines().collect();
        if lines.len()>max_lines{
            lines.drain(0..(lines.len()-max_lines));
            let _=fs::write(path, lines.join("\n")+"\n");
        }
    }
}

fn prune_expired(path:&PathBuf, expiry_days:u64){
    let ok = fs::read_to_string(path).unwrap_or_default();
    let cutoff = now() - expiry_days*24*3600;
    let kept: Vec<String> = ok.lines().filter(|line|{
        if let Some(pos)=line.find(";; ts="){
            if let Ok(ts) = line[pos+6..].split_whitespace().next().unwrap_or("").parse::<u64>(){
                return ts >= cutoff;
            }
        }
        true
    }).map(|s| s.to_string()).collect();
    let _=fs::write(path, kept.join("\n")+"\n");
}

fn append_soft(line:&str){
    if let Some(p)=heur_path(){
        let mut existing=String::new();
        if let Ok(s)=fs::read_to_string(&p){ existing=s; }
        if existing.contains(line){ return; }
        let mut f=fs::OpenOptions::new().create(true).append(true).open(&p).unwrap();
        let _=writeln!(f, "{} ;; ts={}", line, now());
    }
}

pub fn load_local_kdsl()->Option<String>{
    if let Ok(path) = std::env::var("SPIRAL_HEUR_K_FILE"){
        return std::fs::read_to_string(path).ok();
    }
    let p = file("heur.kdsl")?;
    std::fs::read_to_string(p).ok()
}

/// Synthesize and append a soft rule if (a) min samples met, (b) improvement significant, (c) cooldown passed.
/// choice_key: e.g., ("wg", "256") or ("u2", "true") or ("scan_wg","256").
pub fn maybe_synthesize_rule(pass:&str, sg:bool, lg2c:u32, lg2k:u32, choice_key:(&str,&str), new_time_ms:f32){
    if std::env::var("ST_SELF_REWRITE").ok().as_deref() != Some("1") { return; }
    let min_samples = std::env::var("ST_SR_MIN_SAMPLES").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(5);
    let improve_pct= std::env::var("ST_SR_IMPROVE_PCT").ok().and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.05);
    let cooldown   = std::env::var("ST_SR_COOLDOWN_SEC").ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(3600);
    let max_lines  = std::env::var("ST_SR_MAX_LINES").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(2000);
    let expiry_days= std::env::var("ST_SR_EXPIRY_DAYS").ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(30);

    let obs = read_observations();
    let bkey = (sg, lg2c, lg2k, pass.to_string());
    if should_cooldown(&bkey, cooldown){ return; }

    // median of same bucket
    let mut times = Vec::<f32>::new();
    for o in obs.iter(){
        if bucket(o)==bkey { times.push(o.time_ms); }
    }
    if times.len() < min_samples { return; }
    let base = match med(times){ Some(x)=>x, None=>return };
    if !improve_enough(new_time_ms, base, improve_pct){ return; }

    // produce condition like sg && c>=2^lg2c && k>=2^lg2k
    let cond = format!("{}{}{}",
        if sg {"sg&&"} else {""},
        format!("c>={}", 1u32<<lg2c),
        if lg2k>0 { format!("&&k>={}", 1u32<<lg2k) } else { "".into() }
    );
    let (field, val) = choice_key;
    let weight = ((base - new_time_ms)/base * 0.5 + 0.1).max(0.1); // conservative
    let line = format!("soft({},{}, {:.3}, {});", field, val, weight, cond);
    append_soft(&line);
    if let Some(p)=file("heur.kdsl"){ clamp_file_lines(&p, max_lines); prune_expired(&p, expiry_days); }
}
