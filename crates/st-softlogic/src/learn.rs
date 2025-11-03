// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(feature = "learn_store")]

use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs, path::PathBuf};

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct BetaStat {
    pub alpha: f32,
    pub beta: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct SoftWeights {
    pub rule_beta: HashMap<String, BetaStat>,
    pub base_coef: HashMap<String, f32>,
}

fn store_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let p = format!("{home}/.spiraltorch/soft_weights.json");
    PathBuf::from(p)
}

pub fn load() -> SoftWeights {
    let p = store_path();
    if let Ok(bytes) = fs::read(&p) {
        serde_json::from_slice(&bytes).unwrap_or_default()
    } else {
        SoftWeights::default()
    }
}

pub fn save(sw: &SoftWeights) -> std::io::Result<()> {
    let p = store_path();
    if let Some(parent) = p.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let s = serde_json::to_vec_pretty(sw).unwrap();
    fs::write(p, s)
}

pub fn update_bandit(sw: &mut SoftWeights, winner_rules: &[&str], loser_rules: &[&str]) {
    for r in winner_rules {
        let e = sw.rule_beta.entry((*r).to_string()).or_insert(BetaStat {
            alpha: 1.0,
            beta: 1.0,
        });
        e.alpha += 1.0;
    }
    for r in loser_rules {
        let e = sw.rule_beta.entry((*r).to_string()).or_insert(BetaStat {
            alpha: 1.0,
            beta: 1.0,
        });
        e.beta += 1.0;
    }
}

pub fn weight_from_bandit(sw: &SoftWeights, rule: &str) -> f32 {
    if let Some(b) = sw.rule_beta.get(rule) {
        b.alpha / (b.alpha + b.beta).max(1e-6)
    } else {
        0.5
    }
}

fn stable_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

pub fn online_logistic_update(
    coefs: &mut HashMap<String, f32>,
    feats: &HashMap<String, f32>,
    y: f32,
    eta: f32,
    l2: f32,
) {
    let mut dot = 0.0;
    for (k, v) in feats {
        dot += *coefs.get(k).unwrap_or(&0.0) * *v;
    }
    let pred = stable_sigmoid(dot);
    for (k, v) in feats {
        let w = coefs.entry(k.clone()).or_insert(0.0);
        *w += eta * ((y - pred) * *v - l2 * *w);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_bandit_tracks_wins_and_losses() {
        let mut weights = SoftWeights::default();
        update_bandit(&mut weights, &["rule_a", "rule_b"], &["rule_c"]);
        update_bandit(&mut weights, &["rule_a"], &["rule_b", "rule_c"]);

        let rule_a = weights.rule_beta.get("rule_a").unwrap();
        assert!(rule_a.alpha > rule_a.beta);
        let rule_c = weights.rule_beta.get("rule_c").unwrap();
        assert!(rule_c.beta > rule_c.alpha);
    }

    #[test]
    fn logistic_update_adjusts_weights() {
        let mut coefs = HashMap::new();
        let feats = HashMap::from([("bias".to_string(), 1.0), ("signal".to_string(), 2.0)]);

        online_logistic_update(&mut coefs, &feats, 1.0, 0.5, 0.1);

        assert!(coefs.get("signal").unwrap() > &0.0);
        assert!(coefs.get("bias").unwrap() > &0.0);

        // A negative label should push the weights back down.
        online_logistic_update(&mut coefs, &feats, 0.0, 0.5, 0.1);
        assert!(coefs.get("signal").unwrap() < &1.0);
        assert!(coefs.get("bias").unwrap() < &1.0);
    }

    #[test]
    fn stable_sigmoid_handles_large_inputs() {
        assert!(stable_sigmoid(80.0).is_finite());
        assert!(stable_sigmoid(-80.0) >= 0.0);
    }
}
