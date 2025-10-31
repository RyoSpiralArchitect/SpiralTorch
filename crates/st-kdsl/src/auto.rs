// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Helpers for generating SpiralK heuristic snippets automatically.
//!
//! The routines here glue tuner output with the runtime Wilson score based
//! self-rewrite logic.  Given latency metrics and precomputed hints the
//! functions append `soft(...)` rules to the original program and re-run the
//! parser so callers can persist the resulting policy.

use super::{eval_program, Ctx, Err, Out};
use thiserror::Error;

/// Aggregate telemetry required for the Wilson score decision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WilsonMetrics {
    /// Baseline latency in microseconds (before rewrite).
    pub baseline_latency: f32,
    /// Candidate latency measured after applying the tuned parameters.
    pub candidate_latency: f32,
    /// Number of wins recorded for the candidate configuration.
    pub wins: u32,
    /// Number of trials collected.
    pub trials: u32,
}

impl WilsonMetrics {
    /// Returns the fractional speed-up relative to the baseline.
    pub fn gain(&self) -> f32 {
        if self.baseline_latency <= 0.0 {
            return 0.0;
        }
        (self.baseline_latency - self.candidate_latency) / self.baseline_latency
    }
}

/// Human readable rule that becomes a `soft(...)` statement.
#[derive(Clone, Debug, PartialEq)]
pub struct HeuristicHint {
    pub field: &'static str,
    pub value_expr: String,
    pub weight_expr: String,
    pub condition_expr: String,
}

impl HeuristicHint {
    pub fn new(
        field: &'static str,
        value_expr: impl Into<String>,
        weight: f32,
        condition_expr: impl Into<String>,
    ) -> Self {
        Self {
            field,
            value_expr: value_expr.into(),
            weight_expr: format!("{weight}"),
            condition_expr: condition_expr.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct AiRewriteConfig {
    pub model: String,
    pub max_hints: usize,
    pub default_weight: f32,
    pub eta_floor: f32,
}

impl AiRewriteConfig {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            max_hints: 4,
            default_weight: 0.72,
            eta_floor: 0.35,
        }
    }

    pub fn with_max_hints(mut self, max_hints: usize) -> Self {
        self.max_hints = max_hints.max(1);
        self
    }

    pub fn with_default_weight(mut self, weight: f32) -> Self {
        if weight.is_finite() && weight > 0.0 {
            self.default_weight = weight;
        }
        self
    }

    pub fn with_eta_floor(mut self, eta_floor: f32) -> Self {
        if eta_floor.is_finite() && eta_floor > 0.0 {
            self.eta_floor = eta_floor;
        }
        self
    }
}

#[derive(Clone, Debug)]
pub struct AiRewritePrompt {
    pub base_program: String,
    pub ctx: Ctx,
    pub metrics: Option<WilsonMetrics>,
    pub eta_bar: f32,
    pub device_guard: Option<String>,
    pub notes: Vec<String>,
}

impl AiRewritePrompt {
    pub fn new(base_program: impl Into<String>, ctx: Ctx) -> Self {
        Self {
            base_program: base_program.into(),
            ctx,
            metrics: None,
            eta_bar: 0.5,
            device_guard: None,
            notes: Vec::new(),
        }
    }

    pub fn with_metrics(mut self, metrics: WilsonMetrics) -> Self {
        self.metrics = Some(metrics);
        self
    }

    pub fn with_eta_bar(mut self, eta_bar: f32) -> Self {
        if eta_bar.is_finite() && eta_bar > 0.0 {
            self.eta_bar = eta_bar;
        }
        self
    }

    pub fn with_device_guard(mut self, guard: impl Into<String>) -> Self {
        self.device_guard = Some(guard.into());
        self
    }

    pub fn push_note(&mut self, note: impl Into<String>) {
        self.notes.push(note.into());
    }
}

#[derive(Debug, Error)]
pub enum AiRewriteError {
    #[error("AI generator returned empty hint set")]
    Empty,
    #[error("AI generator produced {0} hints exceeding the max allowed")]
    TooManyHints(usize),
    #[error("SpiralK evaluation failed: {0}")]
    Dsl(Err),
    #[error("AI generator failed: {0}")]
    Generator(String),
}

impl From<Err> for AiRewriteError {
    fn from(value: Err) -> Self {
        AiRewriteError::Dsl(value)
    }
}

pub trait AiHintGenerator {
    fn generate_hints(
        &mut self,
        config: &AiRewriteConfig,
        prompt: &AiRewritePrompt,
    ) -> Result<Vec<HeuristicHint>, AiRewriteError>;
}

impl<F> AiHintGenerator for F
where
    F: FnMut(&AiRewriteConfig, &AiRewritePrompt) -> Result<Vec<HeuristicHint>, AiRewriteError>,
{
    fn generate_hints(
        &mut self,
        config: &AiRewriteConfig,
        prompt: &AiRewritePrompt,
    ) -> Result<Vec<HeuristicHint>, AiRewriteError> {
        self(config, prompt)
    }
}

pub struct TemplateAiGenerator;

impl AiHintGenerator for TemplateAiGenerator {
    fn generate_hints(
        &mut self,
        config: &AiRewriteConfig,
        prompt: &AiRewritePrompt,
    ) -> Result<Vec<HeuristicHint>, AiRewriteError> {
        let mut hints = Vec::new();
        let ctx = prompt.ctx;
        let gain = prompt.metrics.as_ref().map(|m| m.gain()).unwrap_or(0.0);
        let tile_cols = ctx.tile_cols.max(32);
        let target_tile = (tile_cols as f32 * (1.0 + gain.max(0.0))).max(tile_cols as f32) as u32;
        let guard = prompt.device_guard.as_deref().unwrap_or("true").to_string();
        hints.push(HeuristicHint::new(
            "tile_cols",
            format!("max({}, {target_tile})", ctx.tile_cols),
            config.default_weight,
            guard.clone(),
        ));

        let radix = if gain > 0.08 { 4 } else { ctx.radix.max(2) };
        hints.push(HeuristicHint::new(
            "radix",
            radix.to_string(),
            config.default_weight,
            "true",
        ));

        let segments = if gain > 0.12 {
            ctx.segments.max(1) * 2
        } else {
            ctx.segments.max(1)
        };
        hints.push(HeuristicHint::new(
            "segments",
            segments.to_string(),
            config.default_weight,
            guard,
        ));

        if hints.is_empty() {
            return Err(AiRewriteError::Empty);
        }
        Ok(hints)
    }
}

/// Computes the Wilson score lower bound for a Bernoulli process.
pub fn wilson_lower_bound(wins: u32, trials: u32, z: f32) -> f32 {
    if trials == 0 {
        return 0.0;
    }
    let n = trials as f32;
    let phat = wins as f32 / n;
    let denom = 1.0 + z * z / n;
    let centre = phat + z * z / (2.0 * n);
    let margin = z * ((phat * (1.0 - phat) + z * z / (4.0 * n)) / n).sqrt();
    (centre - margin) / denom
}

/// Returns `true` if the candidate satisfies the gain and confidence thresholds.
pub fn should_rewrite(metrics: &WilsonMetrics, min_gain: f32, min_confidence: f32) -> bool {
    if metrics.trials == 0 || metrics.gain() < min_gain {
        return false;
    }
    let lb = wilson_lower_bound(metrics.wins, metrics.trials, 1.96);
    lb >= min_confidence
}

/// Generate a SpiralK KDSl snippet with the provided hints appended to the base program.
pub fn synthesize_program(base_src: &str, hints: &[HeuristicHint]) -> String {
    let mut script = base_src.trim().to_owned();
    if !script.is_empty() && !script.trim_end().ends_with(';') {
        script.push(';');
    }
    for hint in hints {
        script.push_str("\nsoft (");
        script.push_str(hint.field);
        script.push_str(", ");
        script.push_str(&hint.value_expr);
        script.push_str(", ");
        script.push_str(&hint.weight_expr);
        script.push_str(", ");
        script.push_str(&hint.condition_expr);
        script.push_str(");");
    }
    script
}

/// Auto rewrites a program if the candidate metrics pass the Wilson check.
pub fn rewrite_with_wilson(
    base_src: &str,
    ctx: &Ctx,
    metrics: WilsonMetrics,
    hints: &[HeuristicHint],
    min_gain: f32,
    min_confidence: f32,
) -> Result<(Out, String), Err> {
    let base_out = eval_program(base_src, ctx)?;
    if !should_rewrite(&metrics, min_gain, min_confidence) {
        return Ok((base_out, base_src.to_string()));
    }

    let script = synthesize_program(base_src, hints);
    let patched = eval_program(&script, ctx)?;
    Ok((patched, script))
}

pub fn rewrite_with_ai<G: AiHintGenerator>(
    base_src: &str,
    ctx: &Ctx,
    config: &AiRewriteConfig,
    prompt: &AiRewritePrompt,
    mut generator: G,
) -> Result<(Out, String, Vec<HeuristicHint>), AiRewriteError> {
    let hints = generator.generate_hints(config, prompt)?;
    if hints.is_empty() {
        return Err(AiRewriteError::Empty);
    }
    if hints.len() > config.max_hints {
        return Err(AiRewriteError::TooManyHints(hints.len()));
    }
    let script = synthesize_program(base_src, &hints);
    let out = eval_program(&script, ctx).map_err(AiRewriteError::Dsl)?;
    Ok((out, script, hints))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> Ctx {
        Ctx {
            r: 1024,
            c: 16384,
            k: 512,
            sg: true,
            sgc: 256,
            kc: 1024,
            tile_cols: 128,
            radix: 4,
            segments: 2,
        }
    }

    #[test]
    fn rejects_low_gain() {
        let metrics = WilsonMetrics {
            baseline_latency: 10.0,
            candidate_latency: 9.9,
            wins: 64,
            trials: 128,
        };
        assert!(!should_rewrite(&metrics, 0.02, 0.5));
    }

    #[test]
    fn synthesizes_program() {
        let hints = vec![HeuristicHint::new("radix", "radix", 0.8, "true")];
        let script = synthesize_program("algo: 1;", &hints);
        assert!(script.contains("soft (radix"));
    }

    #[test]
    fn rewrite_roundtrip() {
        let hints = vec![HeuristicHint::new("tile_cols", "kc", 0.9, "c > 4096")];
        let metrics = WilsonMetrics {
            baseline_latency: 10.0,
            candidate_latency: 6.0,
            wins: 96,
            trials: 128,
        };
        let (out, script) =
            rewrite_with_wilson("wg: 256;", &ctx(), metrics, &hints, 0.05, 0.5).unwrap();
        assert!(script.contains("tile_cols"));
        assert!(out.hard.wg.is_some());
    }

    #[test]
    fn rewrite_with_ai_uses_template_generator() {
        let base = "wg: 256;";
        let ctx = ctx();
        let metrics = WilsonMetrics {
            baseline_latency: 12.0,
            candidate_latency: 7.0,
            wins: 128,
            trials: 256,
        };
        let prompt = AiRewritePrompt::new(base, ctx)
            .with_metrics(metrics)
            .with_eta_bar(0.8)
            .with_device_guard("sgc >= 64");
        let config = AiRewriteConfig::new("grok-template").with_max_hints(8);
        let generator = TemplateAiGenerator;
        let (out, script, hints) =
            rewrite_with_ai(base, &ctx, &config, &prompt, generator).unwrap();
        assert!(!hints.is_empty());
        assert!(script.contains("tile_cols"));
        assert!(out.soft.len() >= hints.len());
    }
}
