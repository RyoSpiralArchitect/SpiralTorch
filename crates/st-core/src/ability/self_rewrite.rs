//! Auto-append SpiralK soft(...) rules after significance checks.
//! - Tracks A/B wins with Wilson interval; appends only when CI lower bound > threshold.
//! - Respects cooldown, max lines, and safe file path.

use std::{fs, path::PathBuf, time::{Duration, SystemTime}};

#[derive(Default, Clone)]
pub struct Stats { pub wins:u32, pub trials:u32 }

fn wilson_lower_bound(p:f64, n:f64, z:f64)->f64 {
    // lower bound of Wilson interval
    let denom = 1.0 + z*z/n;
    let center = p + z*z/(2.0*n);
    let rad = z * ((p*(1.0-p)/n) + (z*z/(4.0*n*n))).sqrt();
    (center - rad) / denom
}

pub struct RewriteCfg {
    pub alpha: f64,
    pub min_trials: u32,
    pub cooldown_sec: u64,
    pub max_lines: usize,
    pub dst: PathBuf,
}

impl Default for RewriteCfg {
    fn default()->Self{
        let path = std::env::var("SPIRAL_HEUR_PATH").unwrap_or(format!("{}/.spiraltorch/heur.kdsl", dirs::home_dir().unwrap().display()));
        Self{
            alpha: std::env::var("SPIRAL_SELF_ALPHA").ok().and_then(|v|v.parse().ok()).unwrap_or(0.05),
            min_trials: std::env::var("SPIRAL_SELF_MIN_TRIALS").ok().and_then(|v|v.parse().ok()).unwrap_or(20),
            cooldown_sec: 600,
            max_lines: 512,
            dst: PathBuf::from(path),
        }
    }
}

pub fn maybe_append_soft(rule_line:&str, ab:Stats, cfg:&RewriteCfg)->bool{
    let use_sw = std::env::var("SPIRAL_SELF_REWRITE").ok().map(|v|v=="1").unwrap_or(false);
    if !use_sw { return false; }
    if ab.trials < cfg.min_trials { return false; }
    let p = (ab.wins as f64)/(ab.trials as f64);
    let z = 1.959963984540054; // ~95%
    let lb = wilson_lower_bound(p, ab.trials as f64, z);
    if lb < 0.5 { return false; } // not significantly > 50%

    // cooldown / file limits
    let dst = &cfg.dst;
    if let Some(parent) = dst.parent(){ let _ = fs::create_dir_all(parent); }
    if let Ok(meta) = fs::metadata(dst){
        if let Ok(mtime) = meta.modified(){
            if let Ok(elapsed) = mtime.elapsed(){
                if elapsed < Duration::from_secs(cfg.cooldown_sec) { return false; }
            }
        }
    }
    // append with cap
    let mut lines = if let Ok(s) = fs::read_to_string(dst){ s.lines().map(|s|s.to_string()).collect::<Vec<_>>() } else { vec![] };
    lines.push(rule_line.to_string());
    if lines.len() > cfg.max_lines { lines.drain(0..(lines.len()-cfg.max_lines)); }
    let out = lines.join("\n") + "\n";
    if fs::write(dst, out).is_ok(){ true } else { false }
}
