// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#[cfg(feature = "logic")]
#[derive(Clone, Debug)]
pub struct SoftRule {
    pub name: &'static str,
    pub weight: f32,
    pub score: f32,
}

#[cfg(not(feature = "logic"))]
#[derive(Clone, Debug, Default)]
pub struct SoftRule;

#[cfg(feature = "logic-learn")]
pub mod learn {
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;

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
            let entry = sw
                .rule_beta
                .entry((*r).to_string())
                .or_insert(BetaStat { alpha: 1.0, beta: 1.0 });
            entry.alpha += 1.0;
        }
        for r in loser_rules {
            let entry = sw
                .rule_beta
                .entry((*r).to_string())
                .or_insert(BetaStat { alpha: 1.0, beta: 1.0 });
            entry.beta += 1.0;
        }
    }

    pub fn weight_from_bandit(sw: &SoftWeights, rule: &str) -> f32 {
        if let Some(b) = sw.rule_beta.get(rule) {
            b.alpha / (b.alpha + b.beta).max(1e-6)
        } else {
            0.5
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
        let pred = 1.0 / (1.0 + (-dot).exp());
        for (k, v) in feats {
            let w = coefs.entry(k.clone()).or_insert(0.0);
            *w += eta * ((y - pred) * *v - l2 * *w);
        }
    }
}
