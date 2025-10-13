//! self_rewrite.rs (v1.8.5): A/B logs → Wilson CI → heur.kdsl append
use std::{collections::HashMap, fs, io::Write, path::PathBuf, time::{Duration, Instant}};

#[derive(Clone, Debug)]
pub struct ABKey {
    pub kind: &'static str,    // "topk"|"midk"|"bottomk"
    pub rows: u32, pub cols: u32, pub k: u32,
    pub variant: &'static str, // "heap"|"bitonic"|...
}

#[derive(Default, Clone, Debug)]
struct Stat { n:u32, wins:u32 }

fn wilson_lower(p:f64, n:f64, z:f64) -> f64 {
    // Wilson score interval (lower bound)
    let denom = 1.0 + z*z/n;
    let center = p + z*z/(2.0*n);
    let margin = z * ((p*(1.0-p)/n) + (z*z)/(4.0*n*n)).sqrt();
    (center - margin) / denom
}

pub struct SR {
    map: HashMap<String, Stat>,
    last_emit: Instant,
    cooldown: Duration,
    min_n: u32,
    z: f64,
}
impl Default for SR {
    fn default()->Self{ Self{ map:HashMap::new(), last_emit:Instant::now(), cooldown:Duration::from_secs(60), min_n:50, z:1.96 } }
}
impl SR {
    pub fn from_env()->Self{
        let mut sr = SR::default();
        if let Ok(s) = std::env::var("ST_SR_COOLDOWN_S") { if let Ok(v) = s.parse::<u64>(){ sr.cooldown = Duration::from_secs(v); } }
        if let Ok(s) = std::env::var("ST_SR_MIN_N") { if let Ok(v) = s.parse::<u32>(){ sr.min_n = v; } }
        if let Ok(s) = std::env::var("ST_SR_ALPHA") { if let Ok(alpha) = s.parse::<f64>(){ sr.z = statrs::distribution::Normal::new(0.0,1.0).unwrap().inverse_cdf(1.0-alpha/2.0).abs(); } }
        sr
    }
    fn key(&self, k:&ABKey)->String { format!("{}:c{}:k{}", k.kind, ilog2(k.cols), ilog2(k.k)) }
    pub fn log(&mut self, k:&ABKey, win:bool){
        let on = std::env::var("ST_SR_ENABLED").ok().as_deref()==Some("1");
        if !on { return; }
        let bucket = self.key(k);
        let st = self.map.entry(bucket).or_default();
        st.n += 1; if win { st.wins += 1; }
        self.maybe_emit(k.kind);
    }
    fn maybe_emit(&mut self, kind:&str){
        if self.last_emit.elapsed() < self.cooldown { return; }
        self.last_emit = Instant::now();
        // pick strongest bucket
        let mut best_key = String::new(); let mut best_lb = 0.0; let mut best:Stat = Stat::default();
        for (k, st) in self.map.iter() {
            if st.n < self.min_n { continue; }
            let p = st.wins as f64 / st.n as f64;
            let lb = wilson_lower(p, st.n as f64, self.z);
            if lb > best_lb { best_lb = lb; best_key = k.clone(); best = st.clone(); }
        }
        if best_lb > 0.60 { // >60% win with CI safety
            // append a soft rule to ~/.spiraltorch/heur.kdsl
            let rule = format!("// SR {}\nsoft(wg,256,0.10, sg && k<=32 && c>2048)\n", kind);
            let mut path = heur_path();
            if let Some(dir) = path.parent(){ let _=fs::create_dir_all(dir); }
            let mut f = fs::OpenOptions::new().create(true).append(true).open(&path).unwrap();
            let _ = f.write_all(rule.as_bytes());
        }
    }
}

fn heur_path()->PathBuf{
    if let Some(h) = dirs::home_dir() {
        return h.join(".spiraltorch").join("heur.kdsl");
    }
    PathBuf::from("heur.kdsl")
}

fn ilog2(x:u32)->u32{
    if x<=1 { return 0; }
    31 - (x-1).leading_zeros()
}
