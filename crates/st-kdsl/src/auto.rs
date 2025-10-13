//! Helpers for generating SpiralK heuristic snippets automatically.
//!
//! The routines here glue tuner output with the runtime Wilson score based
//! self-rewrite logic.  Given latency metrics and precomputed hints the
//! functions append `soft(...)` rules to the original program and re-run the
//! parser so callers can persist the resulting policy.

use super::{eval_program, Ctx, Err, Out};

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
}
