// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Lightweight A/B/C dialogue logger to speed up Self‑Rewrite convergence.
//! Writes JSON lines to ~/.spiraltorch/ablog.jsonl (configurable by SPIRAL_ABLOG_PATH).

use std::{fs::{OpenOptions}, io::Write};
use serde::Serialize;

#[derive(Serialize, Clone, Copy, Debug)]
pub struct ABEntry {
    pub rows: u32, pub cols: u32, pub k: u32,
    pub backend: &'static str,
    pub variant: &'static str,   // "A" (SoftLogic/DSL), "B" (DSL-hard), or "C" (generated table)
    pub mk: u32, pub mkd: u32, pub tile: u32, pub ctile: u32,
    pub latency_ms: f32,         // measured latency for the run
    pub ok: bool,                // kernel succeeded
    pub ts_ms: u64,              // epoch ms
}

fn path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("SPIRAL_ABLOG_PATH") { return p.into(); }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    std::path::PathBuf::from(format!("{}/.spiraltorch/ablog.jsonl", home))
}

pub fn record(e: ABEntry) {
    let p = path();
    if let Some(par) = p.parent() { let _ = std::fs::create_dir_all(par); }
    let mut f = match OpenOptions::new().create(true).append(true).open(&p) {
        Ok(f)=>f, Err(_)=>return
    };
    let _ = serde_json::to_writer(&mut f, &e);
    let _ = f.write_all(b"\n");
}

/// Constant-z Wilson CI lower bound (no external deps). alpha=0.10 → z≈1.2816
pub fn wilson_lower(wins:u32, n:u32, z: f64) -> f64 {
    if n==0 { return 0.0; }
    let phat = wins as f64 / n as f64;
    let denom = 1.0 + z*z / (n as f64);
    let center = phat + z*z/(2.0*(n as f64));
    let rad = z*((phat*(1.0-phat)+(z*z)/(4.0*(n as f64)))/(n as f64)).sqrt();
    (center - rad)/denom
}
