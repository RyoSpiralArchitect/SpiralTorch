// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};

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
}

#[derive(Clone)]
struct CachedHint {
    hints: Vec<HeuristicHint>,
    eta: f32,
}

#[derive(Clone, Debug, Default)]
struct HintPerformance {
    uses: u32,
    failures: u32,
    total_gain: f32,
    best_gain: f32,
    last_eta: f32,
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
        self.best_gain * 0.7 + mean * 0.3 - penalty
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
            })
            .collect();
        stats.sort_by(|a, b| {
            b.best_gain
                .partial_cmp(&a.best_gain)
                .unwrap_or(Ordering::Equal)
        });
        stats
    }

    /// Attempt a self-rewrite. Returns `Ok(None)` when no rewrite is required.
    pub fn tick(
        &mut self,
        base_src: &str,
        ctx: &Ctx,
        metrics: Option<WilsonMetrics>,
        observed_eta: f32,
    ) -> Result<Option<SelfRewriteEvent>, AiRewriteError> {
        if observed_eta >= self.eta_floor
            && self.smoothed_eta().unwrap_or(observed_eta) >= self.eta_floor
        {
            return Ok(None);
        }

        if let Some(event) = self.try_cached(base_src, ctx, observed_eta)? {
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
        self.push_cache(event.eta_bar, event.hints.clone());
        self.record_hint_success(&event.hints, observed_eta, event.eta_bar);
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
    ) -> Result<Option<SelfRewriteEvent>, AiRewriteError> {
        let mut indices: Vec<usize> = (0..self.hint_cache.len()).collect();
        indices.sort_by(|a, b| {
            let sa = self.cache_priority_index(*a);
            let sb = self.cache_priority_index(*b);
            sb.partial_cmp(&sa).unwrap_or(Ordering::Equal)
        });

        for idx in indices {
            let Some(entry) = self.hint_cache.get(idx).cloned() else {
                continue;
            };
            if entry.eta <= observed_eta {
                self.penalize_hint(&entry.hints);
                continue;
            }
            let event = self.eval_hints(base_src, ctx, &entry.hints)?;
            if event.eta_bar <= observed_eta {
                self.penalize_hint(&entry.hints);
                continue;
            }
            self.record_hint_success(&event.hints, observed_eta, event.eta_bar);
            self.touch_cache(idx, event.eta_bar, event.hints.clone());
            return Ok(Some(event));
        }
        Ok(None)
    }

    fn touch_cache(&mut self, idx: usize, eta: f32, hints: Vec<HeuristicHint>) {
        if let Some(mut cached) = self.hint_cache.remove(idx) {
            cached.eta = eta;
            cached.hints = hints;
            self.hint_cache.push_front(cached);
        } else {
            self.hint_cache.push_front(CachedHint { hints, eta });
        }
        while self.hint_cache.len() > self.cache_limit {
            self.hint_cache.pop_back();
        }
    }

    fn push_cache(&mut self, eta: f32, hints: Vec<HeuristicHint>) {
        self.hint_cache.push_front(CachedHint { hints, eta });
        while self.hint_cache.len() > self.cache_limit {
            self.hint_cache.pop_back();
        }
    }

    fn cache_priority_index(&self, idx: usize) -> f32 {
        self.hint_cache
            .get(idx)
            .map(|entry| self.cache_priority(entry))
            .unwrap_or(0.0)
    }

    fn cache_priority(&self, entry: &CachedHint) -> f32 {
        let key = Self::hint_key(&entry.hints);
        let score = self
            .hint_stats
            .get(&key)
            .map(|perf| perf.score())
            .unwrap_or(entry.eta * 0.5);
        entry.eta + score
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

    fn record_hint_success(&mut self, hints: &[HeuristicHint], observed: f32, eta: f32) {
        let gain = (eta - observed).max(0.0);
        let key = Self::hint_key(hints);
        let entry = self.hint_stats.entry(key).or_default();
        entry.uses += 1;
        entry.total_gain += gain;
        entry.best_gain = entry.best_gain.max(gain);
        entry.last_eta = eta;
    }

    fn penalize_hint(&mut self, hints: &[HeuristicHint]) {
        let key = Self::hint_key(hints);
        let entry = self.hint_stats.entry(key).or_default();
        entry.failures += 1;
        entry.total_gain *= 0.5;
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
}
