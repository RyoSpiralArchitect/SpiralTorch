// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// crates/st-amg/src/sr_learn.rs
//! Example hook that turns an A/B/C dialogue into Wilson anchored rewrites and
//! optional learn_store updates. Call this from your existing `sr.rs` wiring.

use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn heur_path() -> PathBuf {
    if let Ok(p) = std::env::var("SPIRAL_HEUR_FILE") { return PathBuf::from(p); }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(format!("{home}/.spiraltorch/heur.kdsl"))
}

fn roundtable_log_path() -> PathBuf {
    if let Ok(p) = std::env::var("SPIRAL_SR_LOG_FILE") { return PathBuf::from(p); }
    let heur = heur_path();
    let mut log = heur;
    log.set_file_name("roundtable.log");
    log
}

fn normalize_rule(rule_expr: &str) -> Option<String> {
    let trimmed = rule_expr.trim();
    if trimmed.is_empty() { return None; }
    let mut line = trimmed.trim_end_matches(';').trim().to_string();
    if line.is_empty() { return None; }
    if !line.starts_with("soft(") {
        line = format!("soft({line})");
    }
    line.push(';');
    Some(line)
}

fn already_present(path: &PathBuf, line: &str) -> bool {
    if !path.exists() { return false; }
    match fs::File::open(path) {
        Ok(f) => {
            let rdr = BufReader::new(f);
            let needle = line.trim();
            for read in rdr.lines().flatten() {
                if read.trim() == needle { return true; }
            }
            false
        }
        Err(_) => false,
    }
}

pub fn maybe_append_soft(rule_expr: &str) {
    let Some(line) = normalize_rule(rule_expr) else { return; };
    let path = heur_path();
    if already_present(&path, &line) {
        return;
    }

    if let Some(parent) = path.parent() {
        if let Err(err) = fs::create_dir_all(parent) {
            eprintln!("[sr] heur dir create failed: {err}");
            return;
        }
    }

    if path.exists() {
        let bak = path.with_extension("kdsl.bak");
        if let Err(err) = fs::copy(&path, &bak) {
            eprintln!("[sr] heur backup failed: {err}");
        }
    }

    match OpenOptions::new().create(true).append(true).open(&path) {
        Ok(mut f) => {
            if let Err(err) = writeln!(f, "{line}") {
                eprintln!("[sr] append soft write failed: {err}");
            }
        }
        Err(err) => {
            eprintln!("[sr] open heur failed: {err}");
        }
    }
}

fn label_for(idx: usize) -> char {
    ((b'A' + (idx as u8)) as char)
}

fn log_roundtable(
    consensus_rules: &[&str],
    other_proposals: &[&[&str]],
    wins: u32,
    trials: u32,
    lb: f32,
    lb_thresh: f32,
) {
    let path = roundtable_log_path();
    if let Some(parent) = path.parent() {
        if let Err(err) = fs::create_dir_all(parent) {
            eprintln!("[sr] log dir create failed: {err}");
            return;
        }
    }

    let ts = SystemTime::now().duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs()).unwrap_or_default();

    let mut entry = String::new();
    entry.push_str(&format!("[{ts}] wins={wins} trials={trials} lb={:.4} thresh={:.4}\n", lb, lb_thresh));
    if consensus_rules.is_empty() {
        entry.push_str("consensus: (none)\n");
    } else {
        entry.push_str("consensus:\n");
        for rule in consensus_rules { entry.push_str(&format!("  - {rule}\n")); }
    }
    for (idx, group) in other_proposals.iter().enumerate() {
        let label = label_for(idx);
        entry.push_str(&format!("voice {label}:\n"));
        if group.is_empty() {
            entry.push_str("  (no proposals)\n");
        } else {
            for rule in *group { entry.push_str(&format!("  - {rule}\n")); }
        }
    }
    entry.push('\n');

    if let Err(err) = OpenOptions::new().create(true).append(true).open(&path)
        .and_then(|mut f| f.write_all(entry.as_bytes()))
    {
        eprintln!("[sr] roundtable log append failed: {err}");
    }
}

/// Wilson score lower bound (normal approximation, z≈1.96 → 95%).
pub fn wilson_lower_bound(wins: u32, trials: u32, z: f32) -> f32 {
    if trials == 0 { return 0.0; }
    let n = trials as f32;
    let p = (wins as f32) / n;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = p + z2/(2.0*n);
    let margin = (p*(1.0-p)/n + z2/(4.0*n*n)).sqrt() * z;
    (center - margin) / denom
}

/// Feed the agreed rules plus every perspective voiced in the A/B/C roundtable.
pub fn on_abc_conversation(
    consensus_rules: &[&str],
    other_proposals: &[&[&str]],
    wins: u32,
    trials: u32,
    z: f32,
    lb_thresh: f32,
) {
    let lb = wilson_lower_bound(wins, trials, z);
    log_roundtable(consensus_rules, other_proposals, wins, trials, lb, lb_thresh);
    if lb >= lb_thresh && !consensus_rules.is_empty() {
        let gain = (lb - lb_thresh).max(0.0);
        let soft_weight = (0.15 + gain as f32).min(1.0);
        for (idx, rule) in consensus_rules.iter().copied().enumerate() {
            let boost = soft_weight * (1.0 - (idx as f32 * 0.12));
            if boost <= 0.0 { break; }
            maybe_append_soft(&format!("{rule}, {:.3}", boost));
        }
    }

    #[cfg(feature = "learn_store")]
    let dissent_rules: Vec<&str> = other_proposals
        .iter()
        .flat_map(|group| group.iter().copied())
        .collect();

    #[cfg(feature = "learn_store")]
    {
        use st_logic::learn::{load, save, update_bandit};
        let mut sw = load();
        update_bandit(&mut sw, consensus_rules, &dissent_rules);
        if let Err(err) = save(&sw) {
            eprintln!("[sr] learn_store save failed: {err}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wilson_monotonic() {
        let low = wilson_lower_bound(10, 40, 1.96);
        let high = wilson_lower_bound(30, 60, 1.96);
        assert!(high > low);
    }

    #[test]
    fn test_maybe_append_soft_dedup() {
        let mut file = std::env::temp_dir();
        file.push(format!("heur_test_{}.kdsl", std::process::id()));
        let _ = std::fs::remove_file(&file);
        std::env::set_var("SPIRAL_HEUR_FILE", &file);
        maybe_append_soft("soft(tile, 4096, 0.4);");
        maybe_append_soft("soft(tile, 4096, 0.4);");
        maybe_append_soft(" tile , 4096 , 0.4 ");
        let content = std::fs::read_to_string(&file).unwrap();
        let lines: Vec<_> = content.lines().filter(|l| !l.trim().is_empty()).collect();
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0], "soft(tile, 4096, 0.4);");
        let _ = std::fs::remove_file(&file);
        std::env::remove_var("SPIRAL_HEUR_FILE");
    }

    #[test]
    fn test_log_roundtable_writes() {
        let mut log = std::env::temp_dir();
        log.push(format!("roundtable_{}.log", std::process::id()));
        let _ = std::fs::remove_file(&log);
        std::env::set_var("SPIRAL_SR_LOG_FILE", &log);

        log_roundtable(&["soft(tile, 2048, 0.4)"], &[&["soft(tile, 4096, 0.2)"]], 12, 20, 0.66, 0.55);

        let body = std::fs::read_to_string(&log).unwrap();
        assert!(body.contains("wins=12"));
        assert!(body.contains("consensus"));
        assert!(body.contains("voice A"));

        let _ = std::fs::remove_file(&log);
        std::env::remove_var("SPIRAL_SR_LOG_FILE");
    }
}
