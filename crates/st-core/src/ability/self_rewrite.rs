//! One-shot Self‑Rewrite aggregator: logs → Wilson CI → append soft(...) to heur.kdsl
//! Usage:
//!   let n = self_rewrite_from_logs(None, None, RewriteCfg::default())?;
//! Env overrides: SPIRAL_HEUR_FILE (default: ~/.spiraltorch/heur.kdsl), SPIRAL_ABLOG_PATH

use serde::Deserialize;
use std::io::{BufRead, BufReader, Write};

#[derive(Clone, Copy, Debug)]
pub struct RewriteCfg {
    pub alpha: f64,          // Wilson CI alpha (default 0.10)
    pub min_trials: u32,     // minimum samples per bucket
    pub win_threshold: f64,  // accept if lower bound > threshold (default 0.55)
    pub soft_weight: f32,    // weight to write into soft(...)
}
impl Default for RewriteCfg {
    fn default() -> Self { Self{ alpha: 0.10, min_trials: 40, win_threshold: 0.55, soft_weight: 0.15 } }
}

#[derive(Deserialize)]
struct Entry {
    rows: u32, cols: u32, k: u32,
    backend: String, variant: String,
    mk: u32, mkd: u32, tile: u32, ctile: u32,
    latency_ms: f32, ok: bool, ts_ms: u64
}

fn ablog_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("SPIRAL_ABLOG_PATH") { return p.into(); }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    std::path::PathBuf::from(format!("{}/.spiraltorch/ablog.jsonl", home))
}
fn heur_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("SPIRAL_HEUR_FILE") { return p.into(); }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    std::path::PathBuf::from(format!("{}/.spiraltorch/heur.kdsl", home))
}

fn z_from_alpha(alpha:f64)->f64{
    // quick table
    if alpha<=0.02 { 2.3263 } else if alpha<=0.05 { 1.6449 } else if alpha<=0.10 { 1.2816 } else { 1.0 }
}

/// Format a SpiralK condition approximating this bucket: log2(c) and k bound.
fn cond_for(rows:u32, cols: u32, k: u32) -> String {
    let lc = (cols as f64).log2();
    let lc_floor = lc.floor() as i32;
    let kbin = if k<=128 { "k<=128".to_string() }
               else if k<=1024 { "k<=1024".to_string() }
               else if k<=4096 { "k<=4096".to_string() }
               else { "k>4096".to_string() };
    format!("(log2(c)>={})&&({})", lc_floor, kbin)
}

/// Append lines into heur.kdsl with safe backup & header.
fn append_heur_lines(lines:&[String]) -> std::io::Result<()> {
    let path = heur_path();
    if let Some(par) = path.parent() { std::fs::create_dir_all(par)?; }
    // backup
    if path.exists() {
        let bak = path.with_extension("kdsl.bak");
        std::fs::copy(&path, bak)?;
    }
    // write (append)
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path)?;
    writeln!(f, "\n# ---- AUTO APPEND (Self-Rewrite) ----")?;
    for L in lines { writeln!(f, "{L}")?; }
    Ok(())
}

pub fn self_rewrite_from_logs(log_path: Option<&str>, heur_out: Option<&str>, cfg: RewriteCfg) -> Result<usize, String> {
    let p = log_path.map(std::path::PathBuf::from).unwrap_or_else(ablog_path);
    let rd = std::fs::File::open(&p).map_err(|e| format!("open ablog failed: {e}"))?;
    let br = BufReader::new(rd);
    let mut buckets: std::collections::HashMap<String, Vec<Entry>> = std::collections::HashMap::new();
    for line in br.lines() {
        let Ok(line) = line else { continue };
        if line.trim().is_empty() { continue; }
        let Ok(e) = serde_json::from_str::<Entry>(&line) else { continue };
        if !e.ok { continue; }
        let key = format!("{}|r{}|c{}|k{}", e.backend, e.rows, (e.cols as f64).log2().floor() as i32, e.k);
        buckets.entry(key).or_default().push(e);
    }
    let mut out_lines = Vec::<String>::new();
    let z = z_from_alpha(cfg.alpha);
    for (_k, v) in buckets.into_iter() {
        if v.len() < cfg.min_trials as usize { continue; }
        // Tally by (mk,mkd,tile,ctile)
        use std::collections::HashMap;
        let mut tally: HashMap<(u32,u32,u32,u32), (u32,u32)> = HashMap::new(); // (wins,total)
        // "win" = lower latency vs median? we'll pick top-1 by count (simpler and robust)
        for e in v.iter() {
            let key = (e.mk, e.mkd, e.tile, e.ctile);
            let ent = tally.entry(key).or_insert((0,0));
            ent.0 += 1; ent.1 += 1;
        }
        // Choose best by wins/total → Wilson lower bound
        let mut best = None::<((u32,u32,u32,u32),(u32,u32),f64)>;
        for (key, (wins,total)) in tally.into_iter() {
            let lb = crate::ability::ablog::wilson_lower(wins, total, z);
            if lb > cfg.win_threshold {
                if let Some((_k2,_vt2,lb2)) = best {
                    if lb > lb2 { best = Some((key,(wins,total),lb)); }
                } else {
                    best = Some((key,(wins,total),lb));
                }
            }
        }
        if let Some(((mk,mkd,tile,ct),(wins,total),lb)) = best {
            // Build SpiralK soft(...) with coarse condition
            let e0 = &v[0];
            let cond = cond_for(e0.rows, e0.cols, e0.k);
            let w = cfg.soft_weight;
            out_lines.push(format!("soft(mk, {mk}, {w}, {cond});"));
            if mkd!=0 { out_lines.push(format!("soft(mkd, {mkd}, {w}, {cond});")); }
            out_lines.push(format!("soft(tile, {tile}, {w}, {cond});"));
            out_lines.push(format!("soft(ctile, {ct}, {w}, {cond});"));
        }
    }
    if !out_lines.is_empty() {
        append_heur_lines(&out_lines).map_err(|e| format!("append heur failed: {e}"))?;
    }
    Ok(out_lines.len())
}
