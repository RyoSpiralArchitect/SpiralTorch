// crates/st-core/src/heur/self_rewrite.rs  (v1.8.7)
use std::{collections::HashMap, fs, io::Write, path::PathBuf, time::{Duration, Instant}};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub struct ABKey { pub kind: &'static str, pub rows: u32, pub cols: u32, pub k: u32, pub variant: &'static str }

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
struct Stat { n:u32, wins:u32 }

fn wilson_lower(p:f64, n:f64, z:f64) -> f64 {
    let denom = 1.0 + z*z/n;
    let center = p + z*z/(2.0*n);
    let margin = z * ((p*(1.0-p)/n) + (z*z)/(4.0*n*n)).sqrt();
    (center - margin) / denom
}

#[derive(Default, Serialize, Deserialize)]
struct SweetBands { small:u32, mid:u32, large:u32 }
#[derive(Default, Serialize, Deserialize)]
struct SweetFile { topk:Option<SweetBands>, midk:Option<SweetBands>, bottomk:Option<SweetBands> }

pub struct SR {
    map: HashMap<String, Stat>,
    // per-kind histograms of K (win-weighted)
    hist_small: HashMap<&'static str, u64>,
    hist_mid:   HashMap<&'static str, u64>,
    hist_large: HashMap<&'static str, u64>,
    last_emit: Instant, cooldown: Duration, min_n: u32, z: f64,
}
impl Default for SR {
    fn default()->Self{ Self{
        map:HashMap::new(),
        hist_small:HashMap::new(), hist_mid:HashMap::new(), hist_large:HashMap::new(),
        last_emit:Instant::now(), cooldown:Duration::from_secs(60), min_n:50, z:1.96
    } }
}
impl SR {
    pub fn from_env()->Self{
        let mut sr = SR::default();
        if let Ok(s) = std::env::var("ST_SR_COOLDOWN_S") { if let Ok(v) = s.parse::<u64>(){ sr.cooldown = Duration::from_secs(v); } }
        if let Ok(s) = std::env::var("ST_SR_MIN_N") { if let Ok(v) = s.parse::<u32>(){ sr.min_n = v; } }
        if let Ok(s) = std::env::var("ST_SR_ALPHA") { if let Ok(alpha) = s.parse::<f64>(){
            let z = statrs::distribution::Normal::new(0.0,1.0).unwrap().inverse_cdf(1.0-alpha/2.0).abs();
            sr.z = z;
        } }
        sr
    }
    fn key(&self, k:&ABKey)->String { format!("{}:c{}:k{}", k.kind, ilog2(k.cols), ilog2(k.k)) }

    pub fn log(&mut self, k:&ABKey, win:bool){
        if std::env::var("ST_SR_ENABLED").ok().as_deref()!=Some("1") { return; }
        // A/B stat
        let bucket = self.key(k);
        let st = self.map.entry(bucket).or_default();
        st.n += 1; if win { st.wins += 1; }

        // SweetSpot hist (K bands by thirds: <2^10, <2^14, else) as a starting heuristic
        let band = if k.k <= 1024 { "small" } else if k.k <= 16384 { "mid" } else { "large" };
        let dst = match band { "small"=>&mut self.hist_small, "mid"=>&mut self.hist_mid, _=>&mut self.hist_large };
        let e = dst.entry(k.kind).or_insert(0);
        if win { *e += 2 } else { *e += 1 } // win-weighted bump
        self.maybe_emit(k.kind);
    }

    fn maybe_emit(&mut self, kind:&str){
        if self.last_emit.elapsed() < self.cooldown { return; }
        self.last_emit = Instant::now();
        // A/B → Wilson
        let mut best_key = String::new(); let mut best_lb = 0.0; let mut best:Stat = Stat::default();
        for (k, st) in self.map.iter() {
            if !k.starts_with(kind) { continue; }
            if st.n < self.min_n { continue; }
            let p = st.wins as f64 / st.n as f64;
            let lb = wilson_lower(p, st.n as f64, self.z);
            if lb > best_lb { best_lb = lb; best_key = k.clone(); best = st.clone(); }
        }
        if best_lb > 0.60 {
            let rule = format!("// SR {}\nsoft(wg,256,0.08, sg && k<=32 && c>2048)\n", kind);
            let mut path = heur_path();
            if let Some(dir) = path.parent(){ let _=fs::create_dir_all(dir); }
            let mut f = fs::OpenOptions::new().create(true).append(true).open(&path).unwrap();
            let _ = f.write_all(rule.as_bytes());
        }
        // SweetSpot bands → JSON
        self.emit_sweet(kind);
    }

    fn emit_sweet(&self, kind:&str){
        let mut sweet = read_sweet();
        let bands = SweetBands{
            small: 1024, mid: 16384, large: 1_000_000_000, // defaults
        };
        let sel = match kind {
            "topk"    => sweet.topk.get_or_insert(bands),
            "midk"    => sweet.midk.get_or_insert(bands),
            "bottomk" => sweet.bottomk.get_or_insert(bands),
            _ => return,
        };
        // For now we keep thresholds; a future tuner can optimize them using hist_*.
        // Persist
        write_sweet(&sweet);
    }
}

fn heur_path()->PathBuf{
    if let Some(h) = dirs::home_dir() { return h.join(".spiraltorch").join("heur.kdsl"); }
    PathBuf::from("heur.kdsl")
}

fn sweet_path()->PathBuf{
    if let Some(h) = dirs::home_dir() { return h.join(".spiraltorch").join("sweet.json"); }
    PathBuf::from("sweet.json")
}

fn read_sweet()->SweetFile{
    let p = sweet_path();
    if let Ok(s)=fs::read_to_string(&p){
        if let Ok(v)=serde_json::from_str::<SweetFile>(&s){ return v; }
    }
    SweetFile::default()
}
fn write_sweet(v:&SweetFile){
    let p = sweet_path();
    if let Some(dir)=p.parent(){ let _=fs::create_dir_all(dir); }
    let _ = fs::write(p, serde_json::to_string_pretty(v).unwrap());
}

fn ilog2(x:u32)->u32{ if x<=1 { 0 } else { 31 - (x-1).leading_zeros() } }
