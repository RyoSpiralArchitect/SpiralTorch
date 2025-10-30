// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::collections::VecDeque;

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
}

#[derive(Clone)]
struct CachedHint {
    hints: Vec<HeuristicHint>,
    eta: f32,
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
        for idx in 0..self.hint_cache.len() {
            let Some(entry) = self.hint_cache.get(idx).cloned() else {
                continue;
            };
            if entry.eta <= observed_eta {
                continue;
            }
            let event = self.eval_hints(base_src, ctx, &entry.hints)?;
            if event.eta_bar <= observed_eta {
                continue;
            }
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
}
