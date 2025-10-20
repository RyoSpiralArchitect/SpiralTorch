//! Drift-Response Linguistics helpers for Rust callers.
//!
//! This module mirrors the equations used by the Python helper in
//! `tools/python/drift_response_linguistics.py`.  It exposes the existential
//! load, safe radius, and strict-mode latching logic so Rust surfaces inside
//! SpiralTorch can participate in the same governance loop.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::sync::LazyLock;

fn default_timing_scale() -> f32 {
    1.0
}

fn default_timing_signal() -> f32 {
    0.0
}

fn default_base_lambda() -> f32 {
    1.0
}

fn default_beta() -> f32 {
    1.0
}

/// Policy thresholds for a frame.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct FrameThreshold {
    /// Minimum acceptable comprehension drop.
    pub tau: f32,
    /// Maximum tolerated loss for the frame.
    pub rho: f32,
    /// Hazard cutoff used when counting risky frames (CHI in the note).
    #[serde(default = "default_hazard_cut")]
    pub hazard: f32,
}

const fn default_hazard_cut() -> f32 {
    1.0
}

impl FrameThreshold {
    /// Construct a new [`FrameThreshold`].
    pub const fn new(tau: f32, rho: f32, hazard: f32) -> Self {
        Self { tau, rho, hazard }
    }
}

/// Observed or estimated state for a word-frame pair.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FrameState {
    pub phi: f32,
    pub c: f32,
    #[serde(rename = "S")]
    pub s: f32,
    #[serde(rename = "a_den")]
    pub a_den: f32,
    #[serde(rename = "a_con")]
    pub a_con: f32,
    #[serde(rename = "b_den")]
    pub b_den: f32,
    #[serde(rename = "b_con")]
    pub b_con: f32,
    pub kappa: f32,
    #[serde(default = "default_timing_scale")]
    pub timing_scale: f32,
    #[serde(default)]
    pub curvature_a_den: f32,
    #[serde(default)]
    pub curvature_a_con: f32,
    #[serde(default)]
    pub curvature_b_den: f32,
    #[serde(default)]
    pub curvature_b_con: f32,
    #[serde(default)]
    pub kappa_slope: f32,
    #[serde(default)]
    pub directional_axes: BTreeMap<String, DirectionalAxis>,
}

impl FrameState {
    /// Mixture slope for value/benefit under drift.
    #[inline]
    pub fn mix_a(&self) -> f32 {
        (1.0 - self.phi) * self.a_den + self.phi * self.a_con
    }

    /// Mixture slope for risk under drift.
    #[inline]
    pub fn mix_b(&self) -> f32 {
        (1.0 - self.phi) * self.b_den + self.phi * self.b_con
    }

    /// Mixture curvature for value/benefit under drift.
    #[inline]
    pub fn mix_curvature_a(&self) -> f32 {
        (1.0 - self.phi) * self.curvature_a_den + self.phi * self.curvature_a_con
    }

    /// Mixture curvature for risk under drift.
    #[inline]
    pub fn mix_curvature_b(&self) -> f32 {
        (1.0 - self.phi) * self.curvature_b_den + self.phi * self.curvature_b_con
    }
}

impl Default for FrameState {
    fn default() -> Self {
        Self {
            phi: 0.0,
            c: 0.0,
            s: 0.0,
            a_den: 0.0,
            a_con: 0.0,
            b_den: 0.0,
            b_con: 0.0,
            kappa: 0.0,
            timing_scale: default_timing_scale(),
            curvature_a_den: 0.0,
            curvature_a_con: 0.0,
            curvature_b_den: 0.0,
            curvature_b_con: 0.0,
            kappa_slope: 0.0,
            directional_axes: BTreeMap::new(),
        }
    }
}

/// Container for per-word DRL measurements.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct WordState {
    pub name: String,
    pub definition_entropy: f32,
    pub frames: BTreeMap<String, FrameState>,
    #[serde(default = "default_timing_signal")]
    pub timing_signal: f32,
    #[serde(default = "default_base_lambda")]
    pub base_lambda: f32,
    #[serde(default = "default_beta")]
    pub beta: f32,
}

impl WordState {
    /// Helper to build a [`WordState`] from iterators.
    pub fn new(name: impl Into<String>, definition_entropy: f32) -> Self {
        Self {
            name: name.into(),
            definition_entropy,
            frames: BTreeMap::new(),
            timing_signal: default_timing_signal(),
            base_lambda: default_base_lambda(),
            beta: default_beta(),
        }
    }

    /// Insert a frame definition.
    pub fn with_frame(mut self, name: impl Into<String>, frame: FrameState) -> Self {
        self.frames.insert(name.into(), frame);
        self
    }
}

/// Summary of DRL statistics for a word.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DrlMetrics {
    pub word: WordState,
    pub existence_load: f32,
    pub frame_hazards: BTreeMap<String, f32>,
    pub safe_radii: BTreeMap<String, f32>,
    pub chi: u32,
    pub strict_mode: bool,
    pub frame_signatures: BTreeMap<String, FrameSignature>,
    pub direction_signatures: BTreeMap<String, BTreeMap<String, DirectionalSignature>>,
}

/// Backwards compatibility alias for earlier drafts that surfaced DRS.
pub type DrsMetrics = DrlMetrics;

/// Local linear and quadratic response statistics for a frame.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FrameSignature {
    pub value_slope: f32,
    pub risk_slope: f32,
    pub net_slope: f32,
    pub value_curvature: f32,
    pub risk_curvature: f32,
    pub net_curvature: f32,
    pub hazard_multiplier: f32,
    pub timing_elasticity: f32,
    pub safe_radius: Option<f32>,
    pub kappa_slope: f32,
    pub tipping_radius: Option<f32>,
}

/// Basis coefficients for evaluating a directional drift.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DirectionalAxis {
    pub value_components: Vec<f32>,
    pub risk_components: Vec<f32>,
    pub kappa_components: Vec<f32>,
    #[serde(default)]
    pub value_curvature_components: Vec<f32>,
    #[serde(default)]
    pub risk_curvature_components: Vec<f32>,
    #[serde(default)]
    pub kappa_slope_components: Vec<f32>,
}

/// Request for a directional signature.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DirectionQuery {
    pub axis: String,
    pub weights: Vec<f32>,
    #[serde(default)]
    pub label: Option<String>,
}

impl DirectionQuery {
    fn label_or_axis(&self) -> &str {
        self.label.as_deref().unwrap_or(&self.axis)
    }
}

/// Directional response diagnostics for a frame.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DirectionalSignature {
    pub value_slope: f32,
    pub risk_slope: f32,
    pub net_slope: f32,
    pub value_curvature: f32,
    pub risk_curvature: f32,
    pub net_curvature: f32,
    pub hazard_base: f32,
    pub hazard: f32,
    pub hazard_multiplier: f32,
    pub timing_elasticity: f32,
    pub safe_radius: Option<f32>,
    pub kappa_slope: f32,
    pub tipping_radius: Option<f32>,
}

/// Options for analysing a word.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AnalysisOptions {
    #[serde(default)]
    pub hazard_cut: Option<f32>,
    #[serde(default = "default_min_radius")]
    pub min_radius: f32,
    #[serde(default)]
    pub direction_queries: BTreeMap<String, Vec<DirectionQuery>>,
}

const fn default_min_radius() -> f32 {
    0.2
}

impl Default for AnalysisOptions {
    fn default() -> Self {
        Self {
            hazard_cut: None,
            min_radius: default_min_radius(),
            direction_queries: BTreeMap::new(),
        }
    }
}

/// Reasonable defaults that prioritise high-safety frames.
pub static DEFAULT_THRESHOLDS: LazyLock<BTreeMap<String, FrameThreshold>> = LazyLock::new(|| {
    let mut map = BTreeMap::new();
    map.insert("Physical".to_string(), FrameThreshold::new(0.05, 0.1, 0.8));
    map.insert("Normative".to_string(), FrameThreshold::new(0.1, 0.08, 0.9));
    map.insert("Social".to_string(), FrameThreshold::new(0.15, 0.06, 0.7));
    map.insert(
        "Protocol".to_string(),
        FrameThreshold::new(0.02, 0.05, 0.95),
    );
    map.insert(
        "MetaLanguage".to_string(),
        FrameThreshold::new(0.08, 0.07, 0.75),
    );
    map.insert("Mythic".to_string(), FrameThreshold::new(0.2, 0.05, 0.6));
    map
});

/// Clone the default threshold map for caller-side mutation.
pub fn default_thresholds() -> BTreeMap<String, FrameThreshold> {
    (*DEFAULT_THRESHOLDS).clone()
}

fn hazard_multiplier(word: &WordState, frame: &FrameState) -> f32 {
    let timing = (word.timing_signal * frame.timing_scale).max(0.0);
    if timing == 0.0 || word.definition_entropy == 0.0 || frame.phi == 0.0 {
        return 1.0;
    }
    let exponent = (word.beta * word.definition_entropy * frame.phi * timing).clamp(-30.0, 30.0);
    exponent.exp()
}

fn timing_elasticity(word: &WordState, frame: &FrameState, multiplier: f32) -> f32 {
    multiplier * word.beta * word.definition_entropy * frame.phi * frame.timing_scale
}

fn tipping_radius(net_slope: f32, net_curvature: f32) -> Option<f32> {
    if net_curvature.abs() < 1e-9 {
        return None;
    }
    let tipping = -net_slope / net_curvature;
    if tipping > 0.0 {
        Some(tipping)
    } else {
        None
    }
}

fn directional_dot(components: &[f32], weights: &[f32]) -> Option<f32> {
    if components.is_empty() {
        return None;
    }
    if components.len() != weights.len() {
        return None;
    }
    Some(
        components
            .iter()
            .zip(weights.iter())
            .map(|(comp, weight)| comp * weight)
            .sum(),
    )
}

fn directional_signature(
    word: &WordState,
    frame: &FrameState,
    axis: &DirectionalAxis,
    weights: &[f32],
    threshold: Option<&FrameThreshold>,
) -> Option<DirectionalSignature> {
    let value_slope =
        directional_dot(&axis.value_components, weights).unwrap_or_else(|| frame.mix_a());
    let risk_linear =
        directional_dot(&axis.risk_components, weights).unwrap_or_else(|| frame.mix_b());
    let risk_slope = word.base_lambda * risk_linear * frame.s;
    let net_slope = value_slope - risk_slope;
    let value_curvature = directional_dot(&axis.value_curvature_components, weights)
        .unwrap_or_else(|| frame.mix_curvature_a());
    let risk_curvature_base = directional_dot(&axis.risk_curvature_components, weights)
        .unwrap_or_else(|| frame.mix_curvature_b());
    let risk_curvature = word.base_lambda * risk_curvature_base * frame.s;
    let net_curvature = value_curvature - risk_curvature;
    let multiplier = hazard_multiplier(word, frame);
    let hazard_base = -(value_slope - risk_slope);
    let hazard_base = hazard_base.max(0.0);
    let hazard = frame.c * multiplier * hazard_base;
    let kappa_value = directional_dot(&axis.kappa_components, weights).unwrap_or(frame.kappa);
    let kappa_slope =
        directional_dot(&axis.kappa_slope_components, weights).unwrap_or(frame.kappa_slope);
    let safe_radius = threshold.map(|t| {
        let kappa_denom = kappa_value.max(1e-6);
        let comprehension_limit = (1.0 - t.tau) / kappa_denom;
        let risk_denom = (risk_linear.abs() * frame.s).max(1e-9);
        let risk_limit = t.rho / risk_denom;
        comprehension_limit.min(risk_limit)
    });
    Some(DirectionalSignature {
        value_slope,
        risk_slope,
        net_slope,
        value_curvature,
        risk_curvature,
        net_curvature,
        hazard_base,
        hazard,
        hazard_multiplier: multiplier,
        timing_elasticity: timing_elasticity(word, frame, multiplier),
        safe_radius,
        kappa_slope,
        tipping_radius: tipping_radius(net_slope, net_curvature),
    })
}

/// Compute the hazard for a specific frame.
pub fn frame_hazard(word: &WordState, frame: &FrameState) -> f32 {
    let a_mix = frame.mix_a();
    let b_mix = frame.mix_b();
    let base = -(a_mix - word.base_lambda * b_mix * frame.s);
    let base = base.max(0.0);
    if base == 0.0 {
        return 0.0;
    }
    frame.c * hazard_multiplier(word, frame) * base
}

/// Compute the existential load for a word.
pub fn existence_load(word: &WordState) -> f32 {
    let mut total = 0.0;
    for frame in word.frames.values() {
        let a_mix = frame.mix_a();
        let b_mix = frame.mix_b();
        let base = -(a_mix - word.base_lambda * b_mix * frame.s);
        let base = base.max(0.0);
        if base == 0.0 {
            continue;
        }
        let amplifier = 1.0 + word.beta * word.definition_entropy * frame.phi;
        total += frame.c * base * amplifier;
    }
    total
}

/// Compute safe radii for each frame.
pub fn safe_radius(
    word: &WordState,
    thresholds: &BTreeMap<String, FrameThreshold>,
) -> BTreeMap<String, f32> {
    let mut radii = BTreeMap::new();
    for (name, frame) in &word.frames {
        if let Some(threshold) = thresholds.get(name) {
            let kappa = frame.kappa.max(1e-6);
            let comprehension_limit = (1.0 - threshold.tau) / kappa;
            let b_mix = frame.mix_b();
            let risk_denom = (b_mix * frame.s).max(1e-9);
            let risk_limit = threshold.rho / risk_denom;
            radii.insert(name.clone(), comprehension_limit.min(risk_limit));
        }
    }
    radii
}

/// Analyse a word using default hazard cut and radius threshold.
pub fn analyse_word(word: &WordState, thresholds: &BTreeMap<String, FrameThreshold>) -> DrlMetrics {
    analyse_word_with_options(word, thresholds, &AnalysisOptions::default())
}

/// Analyse a word using custom hazard cut and radius threshold.
pub fn analyse_word_with(
    word: &WordState,
    thresholds: &BTreeMap<String, FrameThreshold>,
    hazard_cut: Option<f32>,
    min_radius: f32,
) -> DrlMetrics {
    let mut options = AnalysisOptions::default();
    options.hazard_cut = hazard_cut;
    options.min_radius = min_radius;
    analyse_word_with_options(word, thresholds, &options)
}

/// Analyse a word with explicit options including directional queries.
pub fn analyse_word_with_options(
    word: &WordState,
    thresholds: &BTreeMap<String, FrameThreshold>,
    options: &AnalysisOptions,
) -> DrlMetrics {
    let mut frame_hazards = BTreeMap::new();
    let mut frame_signatures = BTreeMap::new();
    for (name, frame) in &word.frames {
        let hazard = frame_hazard(word, frame);
        frame_hazards.insert(name.clone(), hazard);
        let value_slope = frame.mix_a();
        let risk_slope = word.base_lambda * frame.mix_b() * frame.s;
        let net_slope = value_slope - risk_slope;
        let value_curvature = frame.mix_curvature_a();
        let risk_curvature = word.base_lambda * frame.mix_curvature_b() * frame.s;
        let net_curvature = value_curvature - risk_curvature;
        let multiplier = hazard_multiplier(word, frame);
        let signature = FrameSignature {
            value_slope,
            risk_slope,
            net_slope,
            value_curvature,
            risk_curvature,
            net_curvature,
            hazard_multiplier: multiplier,
            timing_elasticity: timing_elasticity(word, frame, multiplier),
            safe_radius: None,
            kappa_slope: frame.kappa_slope,
            tipping_radius: tipping_radius(net_slope, net_curvature),
        };
        frame_signatures.insert(name.clone(), signature);
    }

    let radii = safe_radius(word, thresholds);
    for (name, radius) in &radii {
        if let Some(signature) = frame_signatures.get_mut(name) {
            signature.safe_radius = Some(*radius);
        }
    }
    let mut hazard_counts = 0u32;
    for (name, hazard) in &frame_hazards {
        if let Some(threshold) = thresholds.get(name) {
            let cut = options.hazard_cut.unwrap_or(threshold.hazard);
            if *hazard >= cut {
                hazard_counts += 1;
            }
        }
    }

    let min_radius_observed = radii.values().fold(f32::INFINITY, |acc, r| acc.min(*r));
    let existence = existence_load(word);
    let strict =
        hazard_counts >= 4 || min_radius_observed <= options.min_radius || existence >= 1.0;

    let mut direction_signatures: BTreeMap<String, BTreeMap<String, DirectionalSignature>> =
        BTreeMap::new();
    if !options.direction_queries.is_empty() {
        for (frame_name, queries) in &options.direction_queries {
            if let Some(frame) = word.frames.get(frame_name) {
                if frame.directional_axes.is_empty() {
                    continue;
                }
                let threshold = thresholds.get(frame_name);
                let mut collected = BTreeMap::new();
                for query in queries {
                    if let Some(axis) = frame.directional_axes.get(&query.axis) {
                        if let Some(signature) =
                            directional_signature(word, frame, axis, &query.weights, threshold)
                        {
                            collected.insert(query.label_or_axis().to_string(), signature);
                        }
                    }
                }
                if !collected.is_empty() {
                    direction_signatures.insert(frame_name.clone(), collected);
                }
            }
        }
    }

    DrlMetrics {
        word: word.clone(),
        existence_load: existence,
        frame_hazards,
        safe_radii: radii,
        chi: hazard_counts,
        strict_mode: strict,
        frame_signatures,
        direction_signatures,
    }
}

/// Convert metrics into a scalar penalty using the default radius.
pub fn trainer_penalty(metrics: &DrlMetrics) -> f32 {
    trainer_penalty_with(metrics, 0.2)
}

/// Convert metrics into a scalar penalty using a custom minimum radius.
pub fn trainer_penalty_with(metrics: &DrlMetrics, min_radius: f32) -> f32 {
    let mut penalty = metrics.existence_load;
    if let Some(&min_radius_observed) = metrics
        .safe_radii
        .values()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
    {
        if min_radius_observed < min_radius {
            let denom = min_radius.max(1e-6);
            penalty += (min_radius - min_radius_observed) / denom;
        }
    }
    penalty += metrics.chi as f32;
    if let Some(min_tipping) = metrics
        .frame_signatures
        .values()
        .filter_map(|sig| {
            sig.tipping_radius
                .and_then(|r| if r > 0.0 { Some(r) } else { None })
        })
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
    {
        if min_tipping < min_radius {
            let denom = min_radius.max(1e-6);
            penalty += (min_radius - min_tipping) / denom;
        }
    }
    if !metrics.direction_signatures.is_empty() {
        if let Some(min_direction_radius) = metrics
            .direction_signatures
            .values()
            .flat_map(|map| map.values().filter_map(|sig| sig.safe_radius))
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        {
            if min_direction_radius < min_radius {
                let denom = min_radius.max(1e-6);
                penalty += (min_radius - min_direction_radius) / denom;
            }
        }
        if let Some(min_direction_tipping) = metrics
            .direction_signatures
            .values()
            .flat_map(|map| {
                map.values().filter_map(|sig| {
                    sig.tipping_radius
                        .and_then(|r| if r > 0.0 { Some(r) } else { None })
                })
            })
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        {
            if min_direction_tipping < min_radius {
                let denom = min_radius.max(1e-6);
                penalty += (min_radius - min_direction_tipping) / denom;
            }
        }
    }
    if metrics.strict_mode {
        penalty *= 1.25;
    }
    penalty
}

/// Aggregate a collection of metrics using the default minimum radius.
pub fn aggregate_penalty<'a, I>(metrics: I) -> f32
where
    I: IntoIterator<Item = &'a DrlMetrics>,
{
    aggregate_penalty_with(metrics, 0.2)
}

/// Aggregate a collection of metrics using a custom minimum radius.
pub fn aggregate_penalty_with<'a, I>(metrics: I, min_radius: f32) -> f32
where
    I: IntoIterator<Item = &'a DrlMetrics>,
{
    metrics.into_iter().fold(0.0, |acc, item| {
        acc + trainer_penalty_with(item, min_radius)
    })
}

/// Produce a hazard-to-radius summary per frame.
pub fn frame_summary(metrics: &DrlMetrics) -> BTreeMap<String, f32> {
    let mut summary = BTreeMap::new();
    for (name, hazard) in &metrics.frame_hazards {
        let value = if let Some(radius) = metrics.safe_radii.get(name) {
            let denom = (*radius).max(1e-6);
            hazard / denom
        } else {
            *hazard
        };
        summary.insert(name.clone(), value);
    }
    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normative_word() -> WordState {
        let mut frames = BTreeMap::new();
        let mut axes = BTreeMap::new();
        axes.insert(
            "metaphor".to_string(),
            DirectionalAxis {
                value_components: vec![0.08],
                risk_components: vec![0.12],
                kappa_components: vec![0.18],
                value_curvature_components: vec![0.0],
                risk_curvature_components: vec![0.0],
                kappa_slope_components: vec![0.0],
            },
        );
        axes.insert(
            "definition_break".to_string(),
            DirectionalAxis {
                value_components: vec![-0.2],
                risk_components: vec![0.9],
                kappa_components: vec![0.6],
                value_curvature_components: vec![0.1],
                risk_curvature_components: vec![0.5],
                kappa_slope_components: vec![0.05],
            },
        );
        frames.insert(
            "Normative".to_string(),
            FrameState {
                phi: 0.65,
                c: 0.9,
                s: 0.8,
                a_den: -0.05,
                a_con: 0.2,
                b_den: 0.4,
                b_con: 0.8,
                kappa: 0.35,
                timing_scale: 1.0,
                directional_axes: axes,
                ..FrameState::default()
            },
        );
        WordState {
            name: "AI".to_string(),
            definition_entropy: 0.72,
            frames,
            timing_signal: 1.4,
            base_lambda: 1.0,
            beta: 1.0,
        }
    }

    #[test]
    fn existence_load_matches_reference() {
        let word = normative_word();
        let existence = existence_load(&word);
        assert!((existence - 0.548_958_6).abs() < 1e-6);
    }

    #[test]
    fn analyse_word_flags_radius_and_penalty() {
        let word = normative_word();
        let thresholds = default_thresholds();
        let metrics = analyse_word(&word, &thresholds);
        let hazard = metrics.frame_hazards.get("Normative").copied().unwrap();
        assert!((hazard - 0.720_051_05).abs() < 1e-6);
        let radius = metrics.safe_radii.get("Normative").copied().unwrap();
        assert!((radius - 0.151_515_16).abs() < 1e-6);
        assert!(metrics.strict_mode);
        assert_eq!(metrics.chi, 0);

        let signature = metrics
            .frame_signatures
            .get("Normative")
            .expect("signature for Normative frame");
        assert!((signature.value_slope - 0.112_5).abs() < 1e-6);
        assert!((signature.risk_slope - 0.528).abs() < 1e-6);
        assert!((signature.net_slope + 0.415_5).abs() < 1e-6);
        let expected_multiplier = (0.72_f32 * 0.65 * 1.4).clamp(-30.0, 30.0).exp();
        assert!((signature.hazard_multiplier - expected_multiplier).abs() < 1e-6);
        let expected_elasticity = expected_multiplier * 0.72_f32 * 0.65 * 1.0;
        assert!((signature.timing_elasticity - expected_elasticity).abs() < 1e-6);
        assert_eq!(signature.safe_radius, Some(radius));
        assert!(signature.tipping_radius.is_none());

        let penalty = trainer_penalty(&metrics);
        assert!((penalty - 0.989_228_55).abs() < 1e-6);
    }

    #[test]
    fn aggregate_penalty_sums_entries() {
        let word = normative_word();
        let thresholds = default_thresholds();
        let metrics = analyse_word(&word, &thresholds);
        let metrics_vec = vec![metrics.clone(), metrics];
        let penalty = aggregate_penalty_with(metrics_vec.iter(), 0.2);
        assert!((penalty - 2.0 * 0.989_228_55).abs() < 1e-6);
    }

    #[test]
    fn directional_queries_drive_extra_penalty() {
        let word = normative_word();
        let thresholds = default_thresholds();
        let mut options = AnalysisOptions::default();
        options.direction_queries.insert(
            "Normative".to_string(),
            vec![
                DirectionQuery {
                    axis: "metaphor".to_string(),
                    weights: vec![1.0],
                    label: Some("metaphor".to_string()),
                },
                DirectionQuery {
                    axis: "definition_break".to_string(),
                    weights: vec![1.0],
                    label: None,
                },
            ],
        );
        let metrics = analyse_word_with_options(&word, &thresholds, &options);
        let base_metrics = analyse_word(&word, &thresholds);
        let normative = metrics
            .direction_signatures
            .get("Normative")
            .expect("normative directional signatures");
        assert!(normative.contains_key("metaphor"));
        assert!(normative.contains_key("definition_break"));
        let definition_break = normative
            .get("definition_break")
            .expect("definition break signature");
        let radius = definition_break
            .safe_radius
            .expect("directional safe radius computed");
        assert!(radius < 0.2);
        let penalty_with = trainer_penalty_with(&metrics, 0.2);
        let penalty_without = trainer_penalty(&base_metrics);
        assert!(penalty_with > penalty_without);
    }
}
