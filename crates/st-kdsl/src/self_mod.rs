// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x1000_0000_01b3;

use crate::auto::{
    AiHintGenerator, AiRewriteConfig, AiRewriteError, AiRewritePrompt, HeuristicHint, WilsonMetrics,
};
use crate::{eval_program, synthesize_program, Ctx, Out, SoftRule};

/// Summary of a successful self-rewrite iteration.
pub struct SelfRewriteEvent {
    pub script: String,
    pub hints: Vec<HeuristicHint>,
    pub eta_bar: f32,
    pub out: Out,
}

/// SpiralK loop that automatically mutates kernels when η̄ drops below a threshold.
pub struct SelfRewriteEngine<G: AiHintGenerator> {
    generator: G,
    config: AiRewriteConfig,
    eta_floor: f32,
    max_history: usize,
    eta_history: VecDeque<f32>,
    cache_limit: usize,
    hint_cache: VecDeque<CachedHint>,
    hint_stats: HashMap<String, HintPerformance>,
    ctx_sensitivity: f32,
    staleness_half_life: u32,
    max_staleness: u32,
    context_clusters: Vec<ContextCluster>,
    cluster_limit: usize,
    cluster_decay: f32,
}

#[derive(Clone)]
struct CachedHint {
    key: String,
    hints: Vec<HeuristicHint>,
    eta: f32,
    ctx_signature: u64,
    staleness: u32,
}

#[derive(Clone, Debug, Default)]
struct HintPerformance {
    uses: u32,
    failures: u32,
    total_gain: f32,
    best_gain: f32,
    last_eta: f32,
    last_affinity: f32,
}

impl HintPerformance {
    fn mean_gain(&self) -> f32 {
        if self.uses == 0 {
            0.0
        } else {
            self.total_gain / self.uses as f32
        }
    }

    fn score(&self) -> f32 {
        let mean = self.mean_gain();
        let penalty = self.failures as f32 * 0.05;
        let affinity = self.last_affinity * 0.2;
        self.best_gain * 0.6 + mean * 0.3 + affinity - penalty
    }
}

#[derive(Clone, Debug)]
struct ContextCluster {
    signature: u64,
    total_gain: f32,
    best_gain: f32,
    momentum: f32,
    hits: u32,
}

impl ContextCluster {
    fn new(signature: u64, gain: f32, affinity: f32) -> Self {
        let gain = gain.max(0.0);
        let affinity = affinity.clamp(0.0, 1.0);
        let momentum = gain * affinity;
        Self {
            signature,
            total_gain: gain,
            best_gain: gain,
            momentum,
            hits: if gain > 0.0 { 1 } else { 0 },
        }
    }

    fn mean_gain(&self) -> f32 {
        if self.hits == 0 {
            0.0
        } else {
            self.total_gain / self.hits as f32
        }
    }

    fn update(&mut self, signature: u64, gain: f32, affinity: f32, decay: f32) {
        let gain = gain.max(0.0);
        let affinity = affinity.clamp(0.0, 1.0);
        self.total_gain = self.total_gain * decay + gain;
        self.best_gain = self.best_gain.max(gain);
        self.momentum = self.momentum * decay + gain * affinity;
        if gain > 0.0 {
            self.hits = self.hits.saturating_add(1);
        }
        self.signature = Self::blend_signature(self.signature, signature, affinity);
    }

    fn blend_signature(base: u64, incoming: u64, affinity: f32) -> u64 {
        if affinity <= 0.0 {
            return base;
        }
        let mix = base ^ incoming;
        let scaled = (mix as f32 * affinity).round() as u64;
        base ^ scaled
    }
}

/// Snapshot of a cached hint's historic performance.
#[derive(Clone, Debug, PartialEq)]
pub struct HintStatSnapshot {
    pub key: String,
    pub uses: u32,
    pub failures: u32,
    pub best_gain: f32,
    pub mean_gain: f32,
    pub last_eta: f32,
    pub last_affinity: f32,
}

/// Snapshot of a context cluster used to steer hint reuse.
#[derive(Clone, Debug, PartialEq)]
pub struct ContextClusterSnapshot {
    pub signature: u64,
    pub hits: u32,
    pub best_gain: f32,
    pub mean_gain: f32,
    pub momentum: f32,
}

impl<G: AiHintGenerator> SelfRewriteEngine<G> {
    /// Build a new engine with the supplied AI generator and configuration.
    pub fn new(generator: G, config: AiRewriteConfig) -> Self {
        let eta_floor = config.eta_floor;
        Self {
            generator,
            config,
            eta_floor,
            max_history: 16,
            eta_history: VecDeque::new(),
            cache_limit: 3,
            hint_cache: VecDeque::new(),
            hint_stats: HashMap::new(),
            ctx_sensitivity: 0.55,
            staleness_half_life: 8,
            max_staleness: 24,
            context_clusters: Vec::new(),
            cluster_limit: 16,
            cluster_decay: 0.92,
        }
    }

    /// Adjust the minimum η̄ below which rewrites are attempted.
    pub fn with_eta_floor(mut self, eta_floor: f32) -> Self {
        if eta_floor.is_finite() && eta_floor > 0.0 {
            self.eta_floor = eta_floor;
        }
        self
    }

    /// Set the number of historic η̄ samples retained for smoothing.
    pub fn with_history(mut self, window: usize) -> Self {
        self.max_history = window.max(1);
        self
    }

    /// Adjust the number of cached hint sets retained for low-latency rewrites.
    pub fn with_cache_limit(mut self, limit: usize) -> Self {
        self.cache_limit = limit.max(1);
        while self.hint_cache.len() > self.cache_limit {
            self.hint_cache.pop_back();
        }
        self
    }

    /// Adjust how strictly cached hints must match the incoming context signature.
    pub fn with_context_sensitivity(mut self, sensitivity: f32) -> Self {
        if sensitivity.is_finite() {
            self.ctx_sensitivity = sensitivity.clamp(0.0, 1.0);
        }
        self
    }

    /// Configure how quickly cached hints lose priority as they age.
    pub fn with_staleness_half_life(mut self, half_life: u32) -> Self {
        self.staleness_half_life = half_life.max(1);
        self
    }

    /// Limit how long stale entries are retained before being discarded.
    pub fn with_max_staleness(mut self, max_steps: u32) -> Self {
        self.max_staleness = max_steps.max(1);
        self
    }

    /// Adjust how many distinct context clusters are tracked for hint steering.
    pub fn with_cluster_limit(mut self, limit: usize) -> Self {
        self.cluster_limit = limit.max(1);
        while self.context_clusters.len() > self.cluster_limit {
            self.context_clusters.pop();
        }
        self
    }

    /// Configure the exponential decay applied to cluster momentum between rewrites.
    pub fn with_cluster_decay(mut self, decay: f32) -> Self {
        if decay.is_finite() && decay > 0.0 {
            self.cluster_decay = decay.min(0.999);
        }
        self
    }

    /// Returns the smoothed η̄ after the most recent rewrite.
    pub fn smoothed_eta(&self) -> Option<f32> {
        if self.eta_history.is_empty() {
            None
        } else {
            Some(self.eta_history.iter().copied().sum::<f32>() / self.eta_history.len() as f32)
        }
    }

    /// Returns the current η̄ floor after any adaptive tuning.
    pub fn eta_floor(&self) -> f32 {
        self.eta_floor
    }

    /// Exposes cached hint statistics for monitoring and debugging.
    pub fn hint_statistics(&self) -> Vec<HintStatSnapshot> {
        let mut stats: Vec<_> = self
            .hint_stats
            .iter()
            .map(|(key, perf)| HintStatSnapshot {
                key: key.clone(),
                uses: perf.uses,
                failures: perf.failures,
                best_gain: perf.best_gain,
                mean_gain: perf.mean_gain(),
                last_eta: perf.last_eta,
                last_affinity: perf.last_affinity,
            })
            .collect();
        stats.sort_by(|a, b| {
            b.best_gain
                .partial_cmp(&a.best_gain)
                .unwrap_or(Ordering::Equal)
        });
        stats
    }

    /// Exposes the tracked context clusters and their steering metrics.
    pub fn context_clusters(&self) -> Vec<ContextClusterSnapshot> {
        let mut clusters: Vec<_> = self
            .context_clusters
            .iter()
            .map(|cluster| ContextClusterSnapshot {
                signature: cluster.signature,
                hits: cluster.hits,
                best_gain: cluster.best_gain,
                mean_gain: cluster.mean_gain(),
                momentum: cluster.momentum,
            })
            .collect();
        clusters.sort_by(|a, b| {
            b.best_gain
                .partial_cmp(&a.best_gain)
                .unwrap_or(Ordering::Equal)
        });
        clusters
    }

    /// Attempt a self-rewrite. Returns `Ok(None)` when no rewrite is required.
    pub fn tick(
        &mut self,
        base_src: &str,
        ctx: &Ctx,
        metrics: Option<WilsonMetrics>,
        observed_eta: f32,
    ) -> Result<Option<SelfRewriteEvent>, AiRewriteError> {
        self.age_cache();
        if observed_eta >= self.eta_floor
            && self.smoothed_eta().unwrap_or(observed_eta) >= self.eta_floor
        {
            return Ok(None);
        }

        let ctx_signature = Self::ctx_signature(ctx);

        if let Some(event) = self.try_cached(base_src, ctx, observed_eta, ctx_signature)? {
            let eta_bar = event.eta_bar;
            self.record_eta(eta_bar);
            self.auto_tune_eta();
            return Ok(Some(event));
        }

        let mut prompt =
            AiRewritePrompt::new(base_src.to_string(), *ctx).with_eta_bar(observed_eta);
        if let Some(metrics) = metrics {
            prompt = prompt.with_metrics(metrics);
        }

        let hints = self.generator.generate_hints(&self.config, &prompt)?;
        if hints.is_empty() {
            return Err(AiRewriteError::Empty);
        }
        if hints.len() > self.config.max_hints {
            return Err(AiRewriteError::TooManyHints(hints.len()));
        }
        let event = self.eval_hints(base_src, ctx, &hints)?;
        self.record_eta(event.eta_bar);
        self.push_cache(event.eta_bar, event.hints.clone(), ctx_signature);
        self.record_hint_success(
            &event.hints,
            observed_eta,
            event.eta_bar,
            1.0,
            ctx_signature,
        );
        self.auto_tune_eta();

        Ok(Some(event))
    }

    fn record_eta(&mut self, eta: f32) {
        self.eta_history.push_back(eta);
        while self.eta_history.len() > self.max_history {
            self.eta_history.pop_front();
        }
    }

    fn try_cached(
        &mut self,
        base_src: &str,
        ctx: &Ctx,
        observed_eta: f32,
        ctx_signature: u64,
    ) -> Result<Option<SelfRewriteEvent>, AiRewriteError> {
        let mut indices: Vec<usize> = (0..self.hint_cache.len()).collect();
        indices.sort_by(|a, b| {
            let sa = self.cache_priority_index(*a, ctx_signature);
            let sb = self.cache_priority_index(*b, ctx_signature);
            sb.partial_cmp(&sa).unwrap_or(Ordering::Equal)
        });

        for idx in indices {
            let Some(entry) = self.hint_cache.get(idx).cloned() else {
                continue;
            };
            let affinity = self.context_affinity(entry.ctx_signature, ctx_signature);
            let bridged = self.cluster_bridge_affinity(entry.ctx_signature, ctx_signature);
            let effective_affinity = if affinity >= self.ctx_sensitivity * 0.5 {
                affinity.max(bridged)
            } else {
                affinity
            };
            if effective_affinity < self.ctx_sensitivity {
                self.penalize_hint(&entry.hints);
                continue;
            }
            if entry.eta <= observed_eta {
                self.penalize_hint(&entry.hints);
                continue;
            }
            let event = self.eval_hints(base_src, ctx, &entry.hints)?;
            if event.eta_bar <= observed_eta {
                self.penalize_hint(&entry.hints);
                continue;
            }
            self.record_hint_success(
                &event.hints,
                observed_eta,
                event.eta_bar,
                effective_affinity,
                ctx_signature,
            );
            self.touch_cache(idx, event.eta_bar, event.hints.clone(), ctx_signature);
            return Ok(Some(event));
        }
        Ok(None)
    }

    fn touch_cache(&mut self, idx: usize, eta: f32, hints: Vec<HeuristicHint>, ctx_signature: u64) {
        if let Some(mut cached) = self.hint_cache.remove(idx) {
            cached.eta = eta;
            cached.hints = hints;
            cached.ctx_signature = ctx_signature;
            cached.key = Self::hint_key(&cached.hints);
            cached.staleness = 0;
            self.hint_cache.push_front(cached);
        } else {
            self.hint_cache.push_front(CachedHint {
                key: Self::hint_key(&hints),
                hints,
                eta,
                ctx_signature,
                staleness: 0,
            });
        }
        while self.hint_cache.len() > self.cache_limit {
            self.hint_cache.pop_back();
        }
    }

    fn push_cache(&mut self, eta: f32, hints: Vec<HeuristicHint>, ctx_signature: u64) {
        self.hint_cache.push_front(CachedHint {
            key: Self::hint_key(&hints),
            hints,
            eta,
            ctx_signature,
            staleness: 0,
        });
        while self.hint_cache.len() > self.cache_limit {
            self.hint_cache.pop_back();
        }
    }

    fn cache_priority_index(&self, idx: usize, ctx_signature: u64) -> f32 {
        self.hint_cache
            .get(idx)
            .map(|entry| self.cache_priority(entry, ctx_signature))
            .unwrap_or(0.0)
    }

    fn cache_priority(&self, entry: &CachedHint, ctx_signature: u64) -> f32 {
        let score = self
            .hint_stats
            .get(&entry.key)
            .map(|perf| perf.score())
            .unwrap_or(entry.eta * 0.5);
        let affinity = self.context_affinity(entry.ctx_signature, ctx_signature);
        let bridge = self.cluster_bridge_affinity(entry.ctx_signature, ctx_signature);
        let decay = self.staleness_decay(entry.staleness);
        let cluster_bonus = 1.0 + self.cluster_prediction(ctx_signature);
        (entry.eta + score) * affinity.max(bridge) * decay * cluster_bonus
    }

    fn eval_hints(
        &mut self,
        base_src: &str,
        ctx: &Ctx,
        hints: &[HeuristicHint],
    ) -> Result<SelfRewriteEvent, AiRewriteError> {
        let script = synthesize_program(base_src, hints);
        let out = eval_program(&script, ctx).map_err(AiRewriteError::Dsl)?;
        let eta_bar = self.eta_from_out(&out);
        Ok(SelfRewriteEvent {
            script,
            hints: hints.to_vec(),
            eta_bar,
            out,
        })
    }

    fn eta_from_out(&self, out: &Out) -> f32 {
        if out.soft.is_empty() {
            return self.eta_floor.max(0.0);
        }
        let mut score = 0.0f32;
        for rule in &out.soft {
            let weight = match rule {
                SoftRule::U2 { w, .. }
                | SoftRule::Wg { w, .. }
                | SoftRule::Kl { w, .. }
                | SoftRule::Ch { w, .. }
                | SoftRule::Algo { w, .. }
                | SoftRule::Midk { w, .. }
                | SoftRule::Bottomk { w, .. }
                | SoftRule::Ctile { w, .. }
                | SoftRule::TileCols { w, .. }
                | SoftRule::Radix { w, .. }
                | SoftRule::Segments { w, .. } => *w,
            };
            score += weight;
        }
        let mean_weight = score / out.soft.len() as f32;
        (self.eta_floor + mean_weight).tanh().abs()
    }

    fn hint_key(hints: &[HeuristicHint]) -> String {
        hints
            .iter()
            .map(|hint| {
                format!(
                    "{}:{}:{}:{}",
                    hint.field, hint.value_expr, hint.weight_expr, hint.condition_expr
                )
            })
            .collect::<Vec<_>>()
            .join("|")
    }

    fn record_hint_success(
        &mut self,
        hints: &[HeuristicHint],
        observed: f32,
        eta: f32,
        affinity: f32,
        ctx_signature: u64,
    ) {
        let gain = (eta - observed).max(0.0);
        let key = Self::hint_key(hints);
        let entry = self.hint_stats.entry(key).or_default();
        entry.uses += 1;
        entry.total_gain += gain;
        entry.best_gain = entry.best_gain.max(gain);
        entry.last_eta = eta;
        entry.last_affinity = affinity;
        self.promote_cluster(ctx_signature, gain, affinity);
    }

    fn penalize_hint(&mut self, hints: &[HeuristicHint]) {
        let key = Self::hint_key(hints);
        let entry = self.hint_stats.entry(key.clone()).or_default();
        entry.failures += 1;
        entry.total_gain *= 0.5;
        entry.last_affinity *= 0.5;
        self.degrade_cache_entry(&key);
    }

    fn auto_tune_eta(&mut self) {
        if self.eta_history.len() < self.max_history || self.max_history < 3 {
            return;
        }
        let Some(mean) = self.smoothed_eta() else {
            return;
        };
        let delta = (mean - self.eta_floor) * 0.2;
        if delta.abs() < 1e-3 {
            return;
        }
        let new_floor = (self.eta_floor + delta).clamp(0.05, 0.995);
        self.eta_floor = new_floor;
    }

    fn staleness_decay(&self, staleness: u32) -> f32 {
        0.5f32.powf(staleness as f32 / self.staleness_half_life.max(1) as f32)
    }

    fn age_cache(&mut self) {
        for entry in self.hint_cache.iter_mut() {
            entry.staleness = entry.staleness.saturating_add(1);
        }
        let max_staleness = self.max_staleness;
        self.hint_cache
            .retain(|entry| entry.staleness <= max_staleness);
        self.decay_clusters();
    }

    fn degrade_cache_entry(&mut self, key: &str) {
        if let Some(entry) = self.hint_cache.iter_mut().find(|entry| entry.key == key) {
            entry.staleness = entry.staleness.saturating_add(2);
        }
    }

    fn promote_cluster(&mut self, ctx_signature: u64, gain: f32, affinity: f32) {
        if !gain.is_finite() {
            return;
        }
        let mut best_idx: Option<usize> = None;
        let mut best_affinity = 0.0f32;
        for (idx, cluster) in self.context_clusters.iter().enumerate() {
            let affinity_to_cluster = self.context_affinity(cluster.signature, ctx_signature);
            if affinity_to_cluster > best_affinity {
                best_affinity = affinity_to_cluster;
                best_idx = Some(idx);
            }
        }

        if let Some(idx) = best_idx {
            if let Some(cluster) = self.context_clusters.get_mut(idx) {
                cluster.update(
                    ctx_signature,
                    gain,
                    affinity * best_affinity,
                    self.cluster_decay,
                );
                return;
            }
        }

        if self.context_clusters.len() >= self.cluster_limit {
            self.context_clusters.sort_by(|a, b| {
                a.best_gain
                    .partial_cmp(&b.best_gain)
                    .unwrap_or(Ordering::Equal)
            });
            self.context_clusters.remove(0);
        }

        self.context_clusters
            .push(ContextCluster::new(ctx_signature, gain, affinity));
    }

    fn decay_clusters(&mut self) {
        let decay = self.cluster_decay;
        for cluster in self.context_clusters.iter_mut() {
            cluster.total_gain *= decay;
            cluster.momentum *= decay;
        }
        self.context_clusters
            .retain(|cluster| cluster.total_gain > 1e-4 || cluster.momentum > 1e-4);
    }

    fn cluster_prediction(&self, ctx_signature: u64) -> f32 {
        let mut best = 0.0f32;
        for cluster in &self.context_clusters {
            let affinity = self.context_affinity(cluster.signature, ctx_signature);
            if affinity < 0.2 {
                continue;
            }
            let energy = (1.0 + cluster.best_gain + cluster.momentum * 0.5).min(1.6);
            best = best.max(affinity * energy);
        }
        best.min(2.0)
    }

    fn cluster_bridge_affinity(&self, cached: u64, current: u64) -> f32 {
        let mut best = 0.0f32;
        for cluster in &self.context_clusters {
            let to_cached = self.context_affinity(cluster.signature, cached);
            if to_cached < 0.4 {
                continue;
            }
            let to_current = self.context_affinity(cluster.signature, current);
            if to_current < 0.2 {
                continue;
            }
            let base = (to_cached.sqrt() * to_current.sqrt()).min(1.0);
            let energy = (1.0 + cluster.best_gain + cluster.momentum * 0.5).min(1.6);
            best = best.max((base * energy).min(1.0));
        }
        best
    }

    fn context_affinity(&self, cached: u64, current: u64) -> f32 {
        let diff = (cached ^ current).count_ones() as f32;
        (1.0 - diff / 64.0).clamp(0.0, 1.0)
    }

    fn ctx_signature(ctx: &Ctx) -> u64 {
        let mut hash = FNV_OFFSET_BASIS;
        hash = Self::fnv_mix(hash, ctx.r as u64);
        hash = Self::fnv_mix(hash, ctx.c as u64);
        hash = Self::fnv_mix(hash, ctx.k as u64);
        hash = Self::fnv_mix(hash, ctx.sg as u64);
        hash = Self::fnv_mix(hash, ctx.sgc as u64);
        hash = Self::fnv_mix(hash, ctx.kc as u64);
        hash = Self::fnv_mix(hash, ctx.tile_cols as u64);
        hash = Self::fnv_mix(hash, ctx.radix as u64);
        Self::fnv_mix(hash, ctx.segments as u64)
    }

    fn fnv_mix(mut hash: u64, val: u64) -> u64 {
        hash ^= val;
        hash = hash.wrapping_mul(FNV_PRIME);
        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auto::{HeuristicHint, TemplateAiGenerator};

    fn sample_ctx() -> Ctx {
        Ctx {
            r: 1024,
            c: 4096,
            k: 512,
            sg: true,
            sgc: 128,
            kc: 256,
            tile_cols: 64,
            radix: 4,
            segments: 2,
        }
    }

    #[test]
    fn skip_when_eta_above_threshold() {
        let config = AiRewriteConfig::new("gpt-test");
        let mut engine =
            SelfRewriteEngine::new(TemplateAiGenerator, config.clone()).with_eta_floor(0.6);
        let result = engine
            .tick("algo: 1;", &sample_ctx(), None, 0.8)
            .expect("tick");
        assert!(result.is_none());
    }

    #[test]
    fn rewrites_when_eta_low() {
        let mut generator = |_: &AiRewriteConfig, _: &AiRewritePrompt| {
            Ok(vec![HeuristicHint::new("radix", "radix", 0.9, "true")])
        };
        let config = AiRewriteConfig::new("mock");
        let mut engine = SelfRewriteEngine::new(&mut generator, config);
        let event = engine
            .tick("algo: 1;", &sample_ctx(), None, 0.2)
            .expect("tick")
            .expect("event");
        assert!(event.script.contains("soft"));
        assert!(event.eta_bar > 0.0);
        assert_eq!(event.hints.len(), 1);
    }

    #[test]
    fn reuses_cached_hints_before_invoking_ai() {
        use std::cell::Cell;

        let calls = Cell::new(0usize);
        let mut generator = |_: &AiRewriteConfig, _: &AiRewritePrompt| {
            calls.set(calls.get() + 1);
            Ok(vec![HeuristicHint::new(
                "tile_cols",
                "tile_cols * 2",
                0.9,
                "true",
            )])
        };
        let ctx = sample_ctx();
        let config = AiRewriteConfig::new("mock");
        let mut engine = SelfRewriteEngine::new(&mut generator, config)
            .with_eta_floor(0.75)
            .with_cache_limit(2);

        let first = engine
            .tick("algo: 1;", &ctx, None, 0.3)
            .expect("tick")
            .expect("event");
        assert!(first.eta_bar > 0.0);
        assert_eq!(calls.get(), 1);

        let second = engine
            .tick("algo: 1;", &ctx, None, 0.4)
            .expect("tick")
            .expect("event");
        assert!(second.eta_bar > 0.0);
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn records_hint_statistics_for_cached_successes() {
        let mut generator = |_: &AiRewriteConfig, _: &AiRewritePrompt| {
            Ok(vec![HeuristicHint::new(
                "tile_cols",
                "tile_cols * 2",
                0.9,
                "true",
            )])
        };
        let ctx = sample_ctx();
        let mut engine = SelfRewriteEngine::new(&mut generator, AiRewriteConfig::new("mock"))
            .with_eta_floor(0.6)
            .with_history(4);

        for _ in 0..3 {
            let event = engine
                .tick("algo: 1;", &ctx, None, 0.2)
                .expect("tick")
                .expect("event");
            assert!(event.eta_bar > 0.0);
        }

        let stats = engine.hint_statistics();
        assert_eq!(stats.len(), 1);
        let stat = &stats[0];
        assert!(stat.uses >= 3);
        assert!(stat.best_gain > 0.0);
        assert!(stat.mean_gain > 0.0);
        assert_eq!(stat.failures, 0);
        assert!(stat.last_affinity >= 0.0);
    }

    #[test]
    fn auto_tunes_eta_floor_when_history_full() {
        let mut generator = |_: &AiRewriteConfig, _: &AiRewritePrompt| {
            Ok(vec![HeuristicHint::new("radix", "radix * 2", 0.85, "true")])
        };
        let ctx = sample_ctx();
        let mut engine = SelfRewriteEngine::new(&mut generator, AiRewriteConfig::new("mock"))
            .with_eta_floor(0.3)
            .with_history(3);

        let baseline = engine.eta_floor();
        for _ in 0..3 {
            let event = engine
                .tick("algo: 1;", &ctx, None, 0.1)
                .expect("tick")
                .expect("event");
            assert!(event.eta_bar > 0.0);
        }

        let tuned = engine.eta_floor();
        assert!(tuned > baseline);
        assert!(tuned <= 0.995);
    }

    #[test]
    fn captures_context_clusters_after_successes() {
        let mut generator = |_: &AiRewriteConfig, _: &AiRewritePrompt| {
            Ok(vec![HeuristicHint::new("radix", "radix * 2", 0.95, "true")])
        };
        let ctx = sample_ctx();
        let mut engine = SelfRewriteEngine::new(&mut generator, AiRewriteConfig::new("mock"))
            .with_eta_floor(0.6);

        let event = engine
            .tick("algo: 1;", &ctx, None, 0.25)
            .expect("tick")
            .expect("event");
        assert!(event.eta_bar > 0.0);

        let clusters = engine.context_clusters();
        assert_eq!(clusters.len(), 1);
        let snapshot = &clusters[0];
        assert!(snapshot.best_gain > 0.0);
        assert!(snapshot.momentum > 0.0);
        assert!(snapshot.mean_gain >= 0.0);
    }

    #[test]
    fn skips_cached_hints_when_context_diverges() {
        use std::cell::Cell;

        let calls = Cell::new(0usize);
        let mut generator = |_: &AiRewriteConfig, _: &AiRewritePrompt| {
            calls.set(calls.get() + 1);
            Ok(vec![HeuristicHint::new("radix", "radix * 2", 0.9, "true")])
        };
        let ctx_a = sample_ctx();
        let mut ctx_b = sample_ctx();
        ctx_b.r = ctx_a.r + 512;
        ctx_b.c = ctx_a.c + 128;
        ctx_b.tile_cols = ctx_a.tile_cols * 2;

        let mut engine = SelfRewriteEngine::new(&mut generator, AiRewriteConfig::new("mock"))
            .with_eta_floor(0.7)
            .with_context_sensitivity(0.95);

        engine
            .tick("algo: 1;", &ctx_a, None, 0.2)
            .expect("tick")
            .expect("event");
        assert_eq!(calls.get(), 1);

        engine
            .tick("algo: 1;", &ctx_b, None, 0.25)
            .expect("tick")
            .expect("event");
        assert_eq!(calls.get(), 2, "generator invoked for divergent context");
    }

    #[test]
    fn cluster_bridging_reuses_cache_for_related_context() {
        use std::cell::Cell;

        let calls = Cell::new(0usize);
        let mut generator = |_: &AiRewriteConfig, _: &AiRewritePrompt| {
            calls.set(calls.get() + 1);
            Ok(vec![HeuristicHint::new(
                "tile_cols",
                "tile_cols + 8",
                0.9,
                "true",
            )])
        };

        let ctx_a = sample_ctx();
        let sig_a = SelfRewriteEngine::<TemplateAiGenerator>::ctx_signature(&ctx_a);

        let mut engine = SelfRewriteEngine::new(&mut generator, AiRewriteConfig::new("mock"))
            .with_eta_floor(0.6)
            .with_context_sensitivity(0.8);

        let mut candidate_ctx = None;
        for delta in 1..32 {
            let ctx_b = Ctx {
                r: ctx_a.r + delta * 8,
                c: ctx_a.c + delta * 4,
                k: ctx_a.k,
                sg: ctx_a.sg,
                sgc: ctx_a.sgc,
                kc: ctx_a.kc,
                tile_cols: ctx_a.tile_cols.saturating_add((delta % 3) as u32 + 1),
                radix: ctx_a.radix.saturating_add(delta as u32 % 5 + 1),
                segments: ctx_a.segments,
            };
            let sig_b = SelfRewriteEngine::<TemplateAiGenerator>::ctx_signature(&ctx_b);
            let affinity = engine.context_affinity(sig_a, sig_b);
            if affinity < 0.8 && affinity > 0.45 {
                candidate_ctx = Some((ctx_b, sig_b, affinity));
                break;
            }
        }

        let (ctx_b, sig_b, base_affinity) = candidate_ctx.expect("related context");
        assert!(base_affinity < 0.8);
        assert!(base_affinity > 0.45);

        engine
            .tick("algo: 1;", &ctx_a, None, 0.25)
            .expect("tick")
            .expect("event");
        assert_eq!(calls.get(), 1);

        let bridged = engine.cluster_bridge_affinity(sig_a, sig_b);
        assert!(bridged >= base_affinity);

        engine
            .tick("algo: 1;", &ctx_b, None, 0.3)
            .expect("tick")
            .expect("event");

        assert_eq!(calls.get(), 1, "cluster bridging reused cached hints");
    }

    #[test]
    fn purges_cache_entries_that_age_out() {
        let mut generator = |_: &AiRewriteConfig, _: &AiRewritePrompt| {
            Ok(vec![HeuristicHint::new(
                "segments",
                "segments + 1",
                0.8,
                "true",
            )])
        };
        let ctx = sample_ctx();
        let mut engine = SelfRewriteEngine::new(&mut generator, AiRewriteConfig::new("mock"))
            .with_eta_floor(0.65)
            .with_cache_limit(2)
            .with_staleness_half_life(1)
            .with_max_staleness(2);

        engine
            .tick("algo: 1;", &ctx, None, 0.2)
            .expect("tick")
            .expect("event");
        assert_eq!(engine.hint_cache.len(), 1);

        // Age the cache without triggering a rewrite so the entry decays naturally.
        engine.tick("algo: 1;", &ctx, None, 0.9).expect("tick");
        engine.tick("algo: 1;", &ctx, None, 0.9).expect("tick");
        engine.tick("algo: 1;", &ctx, None, 0.9).expect("tick");

        assert!(engine.hint_cache.is_empty(), "stale entries were purged");
    }
}
