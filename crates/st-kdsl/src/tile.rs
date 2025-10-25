// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Enumeration helpers that generate a consistent autotuning tile search space
//! for subgroup-first kernels.  The template holds candidate values for each
//! dimension and produces the Cartesian product without allocating intermediate
//! vectors so it can be used directly by autotuners.

use serde::{Deserialize, Serialize};
use serde_json;
use std::cmp::Ordering;

/// Concrete tiling choice considered by the search.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TileConfig {
    pub workgroup: (u32, u32),
    pub tile_m: u32,
    pub tile_n: u32,
    pub tile_k: u32,
    pub vector: u32,
    pub stages: u32,
}

impl TileConfig {
    pub fn workgroup_size(&self) -> (u32, u32, u32) {
        (self.workgroup.0, self.workgroup.1, 1)
    }
}

/// Builder that stores candidate dimensions for the search space.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileTemplate {
    workgroups: Vec<(u32, u32)>,
    tile_m: Vec<u32>,
    tile_n: Vec<u32>,
    tile_k: Vec<u32>,
    vector: Vec<u32>,
    stages: Vec<u32>,
}

impl Default for TileTemplate {
    fn default() -> Self {
        Self {
            workgroups: vec![(32, 4), (64, 2), (128, 1)],
            tile_m: vec![64, 128],
            tile_n: vec![32, 64],
            tile_k: vec![16, 32],
            vector: vec![1, 2, 4],
            stages: vec![1, 2],
        }
    }
}

impl TileTemplate {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_workgroups<I>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = (u32, u32)>,
    {
        self.workgroups = values.into_iter().collect();
        self
    }

    pub fn with_tile_m<I>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = u32>,
    {
        self.tile_m = values.into_iter().collect();
        self
    }

    pub fn with_tile_n<I>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = u32>,
    {
        self.tile_n = values.into_iter().collect();
        self
    }

    pub fn with_tile_k<I>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = u32>,
    {
        self.tile_k = values.into_iter().collect();
        self
    }

    pub fn with_vector<I>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = u32>,
    {
        self.vector = values.into_iter().collect();
        self
    }

    pub fn with_stages<I>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = u32>,
    {
        self.stages = values.into_iter().collect();
        self
    }

    pub fn push_workgroup(&mut self, workgroup: (u32, u32)) {
        self.workgroups.push(workgroup);
    }

    pub fn push_tile_m(&mut self, value: u32) {
        self.tile_m.push(value);
    }

    pub fn push_tile_n(&mut self, value: u32) {
        self.tile_n.push(value);
    }

    pub fn push_tile_k(&mut self, value: u32) {
        self.tile_k.push(value);
    }

    pub fn push_vector(&mut self, value: u32) {
        self.vector.push(value);
    }

    pub fn push_stages(&mut self, value: u32) {
        self.stages.push(value);
    }

    pub fn workgroups(&self) -> &[(u32, u32)] {
        &self.workgroups
    }

    pub fn iter(&self) -> TileIter<'_> {
        TileIter::new(self, TileKnowledge::default())
    }

    pub fn iter_with_entries<'a>(
        &'a self,
        entries: &[crate::autotune_store::AutoTuneEntry],
    ) -> TileIter<'a> {
        TileIter::new(self, TileKnowledge::from_entries(entries))
    }

    pub fn is_empty(&self) -> bool {
        self.workgroups.is_empty()
            || self.tile_m.is_empty()
            || self.tile_n.is_empty()
            || self.tile_k.is_empty()
            || self.vector.is_empty()
            || self.stages.is_empty()
    }
}

pub struct TileIter<'a> {
    order: Vec<TileConfig>,
    index: usize,
    _template: &'a TileTemplate,
}

impl<'a> TileIter<'a> {
    fn new(template: &'a TileTemplate, knowledge: TileKnowledge) -> Self {
        let order = if template.is_empty() {
            Vec::new()
        } else {
            knowledge.plan(template)
        };
        Self {
            order,
            index: 0,
            _template: template,
        }
    }
}

impl<'a> Iterator for TileIter<'a> {
    type Item = TileConfig;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.order.len() {
            return None;
        }
        let cfg = self.order[self.index];
        self.index += 1;
        Some(cfg)
    }
}

#[derive(Clone, Debug, Default)]
pub struct TileKnowledge {
    seeds: Vec<TileSeed>,
}

#[derive(Clone, Copy, Debug)]
struct TileSeed {
    config: TileConfig,
    score: f64,
}

impl TileKnowledge {
    fn from_entries(entries: &[crate::autotune_store::AutoTuneEntry]) -> Self {
        let mut seeds = Vec::new();
        for entry in entries {
            if let Ok(snapshot) = serde_json::from_value::<TileSnapshot>(entry.params.clone()) {
                seeds.push(TileSeed {
                    config: snapshot.into(),
                    score: entry.score,
                });
            }
        }
        seeds.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(Ordering::Equal));
        seeds.dedup_by(|a, b| a.config == b.config);
        Self { seeds }
    }

    fn plan(&self, template: &TileTemplate) -> Vec<TileConfig> {
        let mut all = enumerate_all(template);
        if self.seeds.is_empty() {
            return all;
        }

        let mut scored: Vec<(f64, TileConfig)> = all
            .drain(..)
            .map(|cfg| (self.score_for(cfg), cfg))
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        scored.into_iter().map(|(_, cfg)| cfg).collect()
    }

    fn score_for(&self, config: TileConfig) -> f64 {
        self.seeds
            .iter()
            .map(|seed| seed.score + distance(config, seed.config))
            .fold(f64::INFINITY, f64::min)
    }
}

fn enumerate_all(template: &TileTemplate) -> Vec<TileConfig> {
    let mut configs = Vec::new();
    for &workgroup in &template.workgroups {
        for &tile_m in &template.tile_m {
            for &tile_n in &template.tile_n {
                for &tile_k in &template.tile_k {
                    for &vector in &template.vector {
                        for &stages in &template.stages {
                            configs.push(TileConfig {
                                workgroup,
                                tile_m,
                                tile_n,
                                tile_k,
                                vector,
                                stages,
                            });
                        }
                    }
                }
            }
        }
    }
    configs
}

fn distance(a: TileConfig, b: TileConfig) -> f64 {
    let mut delta = 0.0;
    delta += rel_abs_diff(a.workgroup.0 as f64, b.workgroup.0 as f64);
    delta += rel_abs_diff(a.workgroup.1 as f64, b.workgroup.1 as f64);
    delta += rel_abs_diff(a.tile_m as f64, b.tile_m as f64);
    delta += rel_abs_diff(a.tile_n as f64, b.tile_n as f64);
    delta += rel_abs_diff(a.tile_k as f64, b.tile_k as f64);
    delta += rel_abs_diff(a.vector as f64, b.vector as f64);
    delta += rel_abs_diff(a.stages as f64, b.stages as f64);
    delta
}

fn rel_abs_diff(a: f64, b: f64) -> f64 {
    let base = a.abs().max(b.abs()).max(1.0);
    (a - b).abs() / base
}

#[derive(Debug, Clone, Deserialize)]
struct TileSnapshot {
    workgroup: (u32, u32),
    tile_m: u32,
    tile_n: u32,
    tile_k: u32,
    vector: u32,
    stages: u32,
}

impl From<TileSnapshot> for TileConfig {
    fn from(snapshot: TileSnapshot) -> Self {
        TileConfig {
            workgroup: snapshot.workgroup,
            tile_m: snapshot.tile_m,
            tile_n: snapshot.tile_n,
            tile_k: snapshot.tile_k,
            vector: snapshot.vector,
            stages: snapshot.stages,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autotune_store::AutoTuneEntry;
    use serde_json::{to_value, Value};
    use std::collections::HashSet;

    #[test]
    fn enumerates_cartesian_product() {
        let template = TileTemplate::new()
            .with_workgroups([(64, 2), (128, 1)])
            .with_tile_m([64, 96])
            .with_tile_n([32])
            .with_tile_k([16, 32])
            .with_vector([1, 2])
            .with_stages([1, 2]);
        let configs: Vec<_> = template.iter().collect();
        let expected = enumerate_all(&template);
        let configs_set: HashSet<_> = configs.iter().copied().collect();
        let expected_set: HashSet<_> = expected.iter().copied().collect();
        assert_eq!(configs.len(), expected.len());
        assert_eq!(configs_set, expected_set);
    }

    #[test]
    fn handles_empty_dimension() {
        let template = TileTemplate::new().with_tile_k(Vec::<u32>::new());
        assert!(template.is_empty());
        assert_eq!(template.iter().count(), 0);
    }

    #[test]
    fn guided_iteration_prioritizes_seed() {
        let template = TileTemplate::new()
            .with_workgroups([(64, 4)])
            .with_tile_m([64, 96])
            .with_tile_n([32, 64])
            .with_tile_k([16, 32])
            .with_vector([1])
            .with_stages([1]);

        let seed = TileConfig {
            workgroup: (64, 4),
            tile_m: 96,
            tile_n: 64,
            tile_k: 32,
            vector: 1,
            stages: 1,
        };
        let entry = AutoTuneEntry {
            updated_unix: 0,
            score: 1.0,
            params: to_value(seed).unwrap(),
            context: Value::Null,
            features: Vec::new(),
        };
        let configs: Vec<_> = template.iter_with_entries(&[entry]).take(2).collect();
        assert_eq!(configs[0], seed);
        assert!(configs[1] != seed);
    }
}
