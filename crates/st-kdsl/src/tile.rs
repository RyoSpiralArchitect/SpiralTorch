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
    pub segments: u32,
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
    segments: Vec<u32>,
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
            segments: vec![1],
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

    pub fn with_segments<I>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = u32>,
    {
        self.segments = values.into_iter().collect();
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

    pub fn push_segments(&mut self, value: u32) {
        self.segments.push(value);
    }

    pub fn workgroups(&self) -> &[(u32, u32)] {
        &self.workgroups
    }

    pub fn segments(&self) -> &[u32] {
        &self.segments
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
            || self.segments.is_empty()
    }
}

pub struct TileIter<'a> {
    order: Vec<TileConfig>,
    index: usize,
    _template: &'a TileTemplate,
}

impl<'a> TileIter<'a> {
    pub fn new(template: &'a TileTemplate, knowledge: TileKnowledge) -> Self {
        let knowledge = knowledge.prepared(template);
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

#[derive(Clone, Copy, Debug)]
pub struct WeightedMetric {
    pub w_wgx: f64,
    pub w_wgy: f64,
    pub w_tm: f64,
    pub w_tn: f64,
    pub w_tk: f64,
    pub w_vec: f64,
    pub w_stg: f64,
    pub w_seg: f64,
    pub p: f64,
}

impl Default for WeightedMetric {
    fn default() -> Self {
        Self {
            w_wgx: 0.5,
            w_wgy: 0.5,
            w_tm: 0.8,
            w_tn: 0.8,
            w_tk: 1.2,
            w_vec: 1.3,
            w_stg: 0.7,
            w_seg: 0.9,
            p: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct TemplateStats {
    rg_wgx: f64,
    rg_wgy: f64,
    rg_tm: f64,
    rg_tn: f64,
    rg_tk: f64,
    rg_vec: f64,
    rg_stg: f64,
    rg_seg: f64,
}

impl TemplateStats {
    fn from_template(template: &TileTemplate) -> Self {
        fn range<I>(iter: I) -> f64
        where
            I: IntoIterator<Item = u32>,
        {
            let mut iter = iter.into_iter();
            let first = iter.next();
            let mut min = first.unwrap_or(0);
            let mut max = min;
            for value in iter {
                if value < min {
                    min = value;
                }
                if value > max {
                    max = value;
                }
            }
            let diff = max.saturating_sub(min) as f64;
            diff.max(1.0)
        }

        Self {
            rg_wgx: range(template.workgroups.iter().map(|&(x, _)| x)),
            rg_wgy: range(template.workgroups.iter().map(|&(_, y)| y)),
            rg_tm: range(template.tile_m.iter().copied()),
            rg_tn: range(template.tile_n.iter().copied()),
            rg_tk: range(template.tile_k.iter().copied()),
            rg_vec: range(template.vector.iter().copied()),
            rg_stg: range(template.stages.iter().copied()),
            rg_seg: range(template.segments.iter().copied()),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TileKnowledge {
    seeds: Vec<TileSeed>,
    metric: WeightedMetric,
    stats: TemplateStats,
}

#[derive(Clone, Copy, Debug)]
struct TileSeed {
    config: TileConfig,
    score: f64,
}

impl TileKnowledge {
    pub fn from_entries(entries: &[crate::autotune_store::AutoTuneEntry]) -> Self {
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
        Self {
            seeds,
            metric: WeightedMetric::default(),
            stats: TemplateStats::default(),
        }
    }

    fn prepared(mut self, template: &TileTemplate) -> Self {
        self.stats = TemplateStats::from_template(template);
        self
    }

    pub fn with_metric(mut self, metric: WeightedMetric) -> Self {
        self.metric = metric;
        self
    }

    fn plan(&self, template: &TileTemplate) -> Vec<TileConfig> {
        let mut all = enumerate_all(template);
        if self.seeds.is_empty() {
            return all;
        }

        let stats = self.stats;
        let metric = self.metric;
        all.sort_by(|a, b| {
            let sa = self.seeded_score(*a, stats, metric);
            let sb = self.seeded_score(*b, stats, metric);
            sa.partial_cmp(&sb).unwrap_or(Ordering::Equal)
        });
        all
    }

    fn seeded_score(
        &self,
        config: TileConfig,
        stats: TemplateStats,
        metric: WeightedMetric,
    ) -> f64 {
        self.seeds
            .iter()
            .map(|seed| seed.score + distance_weighted(config, seed.config, stats, metric))
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
                            for &segments in &template.segments {
                                configs.push(TileConfig {
                                    workgroup,
                                    tile_m,
                                    tile_n,
                                    tile_k,
                                    vector,
                                    stages,
                                    segments,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    configs
}

fn distance_weighted(
    a: TileConfig,
    b: TileConfig,
    stats: TemplateStats,
    weights: WeightedMetric,
) -> f64 {
    fn component(delta: f64, range: f64, weight: f64) -> f64 {
        if range <= 0.0 {
            delta.abs() * weight
        } else {
            (delta.abs() / range.max(1.0)) * weight
        }
    }

    let comps = [
        component(
            a.workgroup.0 as f64 - b.workgroup.0 as f64,
            stats.rg_wgx,
            weights.w_wgx,
        ),
        component(
            a.workgroup.1 as f64 - b.workgroup.1 as f64,
            stats.rg_wgy,
            weights.w_wgy,
        ),
        component(a.tile_m as f64 - b.tile_m as f64, stats.rg_tm, weights.w_tm),
        component(a.tile_n as f64 - b.tile_n as f64, stats.rg_tn, weights.w_tn),
        component(a.tile_k as f64 - b.tile_k as f64, stats.rg_tk, weights.w_tk),
        component(
            a.vector as f64 - b.vector as f64,
            stats.rg_vec,
            weights.w_vec,
        ),
        component(
            a.stages as f64 - b.stages as f64,
            stats.rg_stg,
            weights.w_stg,
        ),
        component(
            a.segments as f64 - b.segments as f64,
            stats.rg_seg,
            weights.w_seg,
        ),
    ];

    let p = weights.p;
    if (p - 1.0).abs() < f64::EPSILON {
        comps.iter().sum()
    } else if (p - 2.0).abs() < f64::EPSILON {
        comps.iter().map(|v| v * v).sum::<f64>().sqrt()
    } else {
        comps.iter().map(|v| v.powf(p)).sum::<f64>().powf(1.0 / p)
    }
}

#[derive(Debug, Clone, Deserialize)]
struct TileSnapshot {
    workgroup: (u32, u32),
    tile_m: u32,
    tile_n: u32,
    tile_k: u32,
    vector: u32,
    stages: u32,
    #[serde(default = "default_segments")]
    segments: u32,
}

fn default_segments() -> u32 {
    1
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
            segments: snapshot.segments,
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
            .with_stages([1, 2])
            .with_segments([1, 2]);
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
            .with_stages([1])
            .with_segments([1, 2]);

        let seed = TileConfig {
            workgroup: (64, 4),
            tile_m: 96,
            tile_n: 64,
            tile_k: 32,
            vector: 1,
            stages: 1,
            segments: 2,
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
