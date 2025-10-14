use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use statrs::distribution::{ContinuousCDF, Normal};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct RewriteState {
    active_version: u32,
    active_rules: Vec<String>,
}

#[derive(Debug, Default, Clone)]
struct RuleStats {
    wins: u32,
    total: u32,
    last_ts: u64,
}

fn heur_dir() -> PathBuf {
    if let Some(home) = dirs::home_dir() {
        home.join(".spiraltorch")
    } else {
        PathBuf::from(".")
    }
}

fn ablog_path() -> PathBuf {
    let mut p = heur_dir();
    p.push("ablog.ndjson");
    p
}

fn heur_file(version: u32) -> PathBuf {
    let mut p = heur_dir();
    p.push(format!("heur_v{version}.kdsl"));
    p
}

fn active_file() -> PathBuf {
    let mut p = heur_dir();
    p.push("heur.kdsl");
    p
}

fn state_file() -> PathBuf {
    let mut p = heur_dir();
    p.push("heur_state.json");
    p
}

fn read_state() -> RewriteState {
    let path = state_file();
    if let Ok(bytes) = fs::read(&path) {
        if let Ok(state) = serde_json::from_slice::<RewriteState>(&bytes) {
            return state;
        }
    }
    RewriteState::default()
}

fn write_state(state: &RewriteState) {
    if let Some(parent) = state_file().parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_vec_pretty(state) {
        let _ = fs::write(state_file(), json);
    }
}

fn wilson_lower_bound(wins: u32, total: u32, alpha: f64) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let p = wins as f64 / total as f64;
    let z = Normal::new(0.0, 1.0)
        .unwrap()
        .inverse_cdf(1.0 - alpha / 2.0);
    let n = total as f64;
    let denom = 1.0 + (z * z) / n;
    let centre = p + (z * z) / (2.0 * n);
    let adj = z * ((p * (1.0 - p) + (z * z) / (4.0 * n)) / n).sqrt();
    (centre - adj) / denom
}

fn sprt_accepts(wins: u32, total: u32, p0: f64, p1: f64, alpha: f64, beta: f64) -> bool {
    if total == 0 {
        return false;
    }
    let wins = wins as f64;
    let losses = (total - wins as u32) as f64;
    let llr = wins * (p1 / p0).ln() + losses * ((1.0 - p1) / (1.0 - p0)).ln();
    let a = ((1.0 - beta) / alpha).ln();
    llr >= a
}

fn parse_logs(path: &Path, window_s: u64) -> HashMap<(String, String), RuleStats> {
    let mut out = HashMap::new();
    let file = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return out,
    };
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let reader = BufReader::new(file);
    let lines: Vec<String> = reader.lines().flatten().collect();
    for line in lines.into_iter().rev() {
        if let Ok(value) = serde_json::from_str::<Value>(&line) {
            let ts = value.get("ts").and_then(|v| v.as_u64()).unwrap_or_default();
            if now.saturating_sub(ts) > window_s {
                break;
            }
            let kind = value
                .get("kind")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let choice = value
                .get("choice")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let win = value.get("win").and_then(|v| v.as_bool()).unwrap_or(false);
            let entry = out.entry((kind, choice)).or_default();
            entry.total += 1;
            if win {
                entry.wins += 1;
            }
            entry.last_ts = entry.last_ts.max(ts);
        }
    }
    out
}

fn choice_to_rule(kind: &str, choice: &str) -> Option<String> {
    let rule = match (kind, choice) {
        ("topk", "heap") => "soft(algo,1,0.12,true)",
        ("topk", "bitonic") => "soft(algo,2,0.12,true)",
        ("topk", "kway") => "soft(algo,3,0.12,true)",
        ("midk", "1ce") => "soft(midk,1,0.10,true)",
        ("midk", "2ce") => "soft(midk,2,0.10,true)",
        ("bottomk", "1ce") => "soft(bottomk,1,0.08,true)",
        ("bottomk", "2ce") => "soft(bottomk,2,0.08,true)",
        _ => return None,
    };
    Some(rule.to_string())
}

fn write_rules(path: &Path, lines: &[String]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut content = String::new();
    for line in lines {
        content.push_str(line);
        content.push('\n');
    }
    fs::write(path, content)
}

fn prune_old_versions(retain: usize, keep_latest: u32) {
    let dir = heur_dir();
    if let Ok(entries) = fs::read_dir(&dir) {
        let mut versions = Vec::new();
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if let Some(rest) = name.strip_prefix("heur_v") {
                    if let Some(num) = rest.strip_suffix(".kdsl") {
                        if let Ok(v) = num.parse::<u32>() {
                            versions.push((v, entry.path()));
                        }
                    }
                }
            }
        }
        versions.sort_by_key(|(v, _)| *v);
        if versions.len() > retain {
            for (version, path) in versions.into_iter() {
                if version == keep_latest {
                    continue;
                }
                if retain == 0 || version + retain as u32 <= keep_latest {
                    let _ = fs::remove_file(path);
                }
            }
        }
    }
}

pub fn maybe_self_rewrite() {
    let enabled = std::env::var("ST_SR_ENABLED")
        .ok()
        .map(|v| v == "1")
        .unwrap_or(false);
    if !enabled {
        return;
    }

    let window_s = std::env::var("ST_SR_WINDOW_S")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(600);
    let ttl_s = std::env::var("ST_SR_TTL_S")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(900);
    let min_n = std::env::var("ST_SR_MIN_N")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(40);
    let alpha = std::env::var("ST_SR_ALPHA")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.05);
    let target_lb = std::env::var("ST_SR_WILSON")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.6);
    let p0 = std::env::var("ST_SR_P0")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.55);
    let p1 = std::env::var("ST_SR_P1")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.7);
    let beta = std::env::var("ST_SR_BETA")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.2);
    let retain_versions = std::env::var("ST_SR_RETAIN")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(3);

    let stats = parse_logs(&ablog_path(), window_s);
    if stats.is_empty() {
        return;
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let mut accepted = Vec::new();
    for ((kind, choice), stat) in stats {
        if stat.total < min_n {
            continue;
        }
        if now.saturating_sub(stat.last_ts) > ttl_s {
            continue;
        }
        let wilson = wilson_lower_bound(stat.wins, stat.total, alpha);
        if wilson < target_lb {
            continue;
        }
        if !sprt_accepts(stat.wins, stat.total, p0, p1, alpha, beta) {
            continue;
        }
        if let Some(rule) = choice_to_rule(&kind, &choice) {
            accepted.push(rule);
        }
    }
    accepted.sort();
    accepted.dedup();

    let mut state = read_state();
    if accepted == state.active_rules {
        return;
    }

    let new_version = state.active_version + 1;
    let heur_path = heur_file(new_version);
    if let Err(err) = write_rules(&heur_path, &accepted) {
        eprintln!("[heur] failed to write {:?}: {}", heur_path, err);
        return;
    }
    if let Err(err) = write_rules(&active_file(), &accepted) {
        eprintln!("[heur] failed to update active heuristics: {}", err);
    }

    state.active_version = new_version;
    state.active_rules = accepted;
    write_state(&state);
    prune_old_versions(retain_versions, new_version);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wilson_lower_bound_behaves() {
        let lb = wilson_lower_bound(70, 100, 0.05);
        assert!(lb > 0.6);
        let low = wilson_lower_bound(10, 100, 0.05);
        assert!(low < 0.5);
    }

    #[test]
    fn sprt_detects_improvements() {
        assert!(sprt_accepts(70, 100, 0.5, 0.7, 0.05, 0.2));
        assert!(!sprt_accepts(52, 100, 0.5, 0.7, 0.05, 0.2));
    }
}
