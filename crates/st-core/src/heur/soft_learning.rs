use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use serde::Deserialize;

use crate::backend::device_caps::BackendKind;
use crate::backend::unison_heuristics::Choice;

#[derive(Debug, Clone)]
pub struct SoftRuleLearner {
    weights: HashMap<String, f32>,
    bias: f32,
    temperature: f32,
    beam: usize,
}

#[derive(Debug, Clone)]
pub struct SoftContext {
    rows: u32,
    cols: u32,
    k: u32,
    backend: BackendKind,
    subgroup: bool,
}

#[derive(Debug, Deserialize)]
struct StoredWeights {
    #[serde(default)]
    weights: HashMap<String, f32>,
    #[serde(default)]
    bias: f32,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_beam")]
    beam: usize,
}

const fn default_temperature() -> f32 {
    1.0
}

const fn default_beam() -> usize {
    1
}

impl SoftRuleLearner {
    pub fn maybe_load() -> Option<Self> {
        let path = store_path();
        let stored = match fs::read(&path) {
            Ok(bytes) => match serde_json::from_slice::<StoredWeights>(&bytes) {
                Ok(cfg) => cfg,
                Err(err) => {
                    eprintln!(
                        "[heur] failed to parse {:?}: {} (ignoring learned weights)",
                        path, err
                    );
                    StoredWeights {
                        weights: HashMap::new(),
                        bias: 0.0,
                        temperature: default_temperature(),
                        beam: default_beam(),
                    }
                }
            },
            Err(_) => return None,
        };
        Some(Self {
            weights: stored.weights,
            bias: stored.bias,
            temperature: stored.temperature.max(1e-3),
            beam: stored.beam.max(1),
        })
    }

    pub fn learned_bonus(&mut self, ctx: &SoftContext, choice: &Choice) -> f32 {
        let mut dot = self.bias;
        for (name, value) in ctx.features(choice) {
            if let Some(weight) = self.weights.get(name) {
                dot += *weight * value;
            }
        }
        (dot / self.temperature).tanh()
    }

    pub fn rank(
        &mut self,
        ctx: &SoftContext,
        candidates: &[(&'static str, Choice, f32)],
    ) -> (usize, Vec<f32>) {
        let mut scored = Vec::with_capacity(candidates.len());
        for (_, choice, base) in candidates {
            let bonus = self.learned_bonus(ctx, choice);
            scored.push(base + bonus);
        }
        if scored.is_empty() {
            return (0, scored);
        }

        let mut indices: Vec<usize> = (0..scored.len()).collect();
        indices.sort_by(|&a, &b| scored[b].partial_cmp(&scored[a]).unwrap_or(Ordering::Equal));
        if self.beam < indices.len() {
            indices.truncate(self.beam);
        }
        let best = *indices
            .iter()
            .max_by(|&&lhs, &&rhs| {
                scored[lhs]
                    .partial_cmp(&scored[rhs])
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap_or(&0);
        (best, scored)
    }
}

impl SoftContext {
    pub fn new(rows: u32, cols: u32, k: u32, backend: BackendKind, subgroup: bool) -> Self {
        Self {
            rows,
            cols,
            k,
            backend,
            subgroup,
        }
    }

    fn features(&self, choice: &Choice) -> Vec<(&'static str, f32)> {
        let rows = (self.rows.max(1) as f32).ln_1p();
        let cols = (self.cols.max(1) as f32).ln_1p();
        let k = (self.k.max(1) as f32).ln_1p();
        let wg = choice.wg as f32 / 512.0;
        let kl = choice.kl as f32 / 32.0;
        let tile = choice.tile as f32 / 4_096.0;
        let ctile = choice.ctile as f32 / 4_096.0;
        let mut feats = vec![
            ("bias", 1.0),
            ("rows_ln", rows),
            ("cols_ln", cols),
            ("k_ln", k),
            ("wg_ratio", wg),
            ("kl_ratio", kl),
            ("tile_ratio", tile),
            ("ctile_ratio", ctile),
            ("mk_kind", choice.mk as f32 / 2.0),
            ("mkd_kind", choice.mkd as f32 / 5.0),
            ("two_stage", if choice.use_2ce { 1.0 } else { 0.0 }),
            ("ch_norm", choice.ch as f32 / 8_192.0),
        ];
        feats.push(match self.backend {
            BackendKind::Wgpu => ("backend_wgpu", 1.0),
            BackendKind::Cuda => ("backend_cuda", 1.0),
            BackendKind::Hip => ("backend_hip", 1.0),
            BackendKind::Cpu => ("backend_cpu", 1.0),
        });
        feats.push(("subgroup", if self.subgroup { 1.0 } else { 0.0 }));
        feats
    }
}

fn store_path() -> PathBuf {
    if let Some(home) = dirs::home_dir() {
        home.join(".spiraltorch").join("soft_weights.json")
    } else {
        PathBuf::from("soft_weights.json")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learned_bonus_changes_with_weights() {
        let mut learner = SoftRuleLearner {
            weights: HashMap::from([
                ("wg_ratio".to_string(), 0.5),
                ("backend_wgpu".to_string(), 0.2),
            ]),
            bias: 0.1,
            temperature: 1.0,
            beam: 2,
        };
        let ctx = SoftContext::new(1_024, 65_536, 256, BackendKind::Wgpu, true);
        let choice = Choice {
            use_2ce: true,
            wg: 256,
            kl: 32,
            ch: 8_192,
            mk: 2,
            mkd: 4,
            tile: 2_048,
            ctile: 1_024,
        };
        let bonus = learner.learned_bonus(&ctx, &choice);
        assert!(bonus > 0.0);
    }

    #[test]
    fn rank_selects_highest_score() {
        let mut learner = SoftRuleLearner {
            weights: HashMap::new(),
            bias: 0.0,
            temperature: 1.0,
            beam: 2,
        };
        let ctx = SoftContext::new(512, 16_384, 64, BackendKind::Cuda, true);
        let candidates = vec![
            (
                "a",
                Choice {
                    use_2ce: false,
                    wg: 128,
                    kl: 8,
                    ch: 0,
                    mk: 0,
                    mkd: 3,
                    tile: 256,
                    ctile: 256,
                },
                0.1,
            ),
            (
                "b",
                Choice {
                    use_2ce: true,
                    wg: 256,
                    kl: 16,
                    ch: 0,
                    mk: 1,
                    mkd: 2,
                    tile: 512,
                    ctile: 256,
                },
                0.8,
            ),
        ];
        let (idx, scores) = learner.rank(&ctx, &candidates);
        assert_eq!(idx, 1);
        assert!(scores[1] > scores[0]);
    }
}
