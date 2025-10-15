// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// ============================================================================
//  SpiralReality Proprietary
//  Copyright (c) 2025 SpiralReality. All Rights Reserved.
//
//  NOTICE: This file contains confidential and proprietary information of
//  SpiralReality. ANY USE, COPYING, MODIFICATION, DISTRIBUTION, DISPLAY,
//  OR DISCLOSURE OF THIS FILE, IN WHOLE OR IN PART, IS STRICTLY PROHIBITED
//  WITHOUT THE PRIOR WRITTEN CONSENT OF SPIRALREALITY.
//
//  NO LICENSE IS GRANTED OR IMPLIED BY THIS FILE. THIS SOFTWARE IS PROVIDED
//  "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
//  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
//  PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL SPIRALREALITY OR ITS
//  SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
//  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
//  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// ============================================================================

#![allow(clippy::too_many_arguments)]

use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::{HashMap, VecDeque};

const SHADOW_LEXICON: &[&str] = &[
    "as a large language model",
    "i cannot",
    "i’m unable",
    "i am unable",
    "safety policy",
    "harmful",
    "violate",
    "disallowed",
    "i don’t have opinions",
    "i do not have opinions",
    "stay safe",
    "cannot provide",
];

const SYMBOL_LEXICON: &[&str] = &[
    "rolled blueprint",
    "spiral",
    "bridge",
    "threshold",
    "lantern",
    "compass",
    "key",
    "weave",
    "seed",
    "vessel",
    "shadow hand",
];

const NUMINOUS_LEXICON: &[&str] = &[
    "cosmic",
    "sacred",
    "eternal",
    "infinite",
    "ultimate",
    "destiny",
    "divine",
    "transcendent",
    "absolute",
    "numinous",
    "mystical",
    "all-encompassing",
];

const RITUAL_STEMS: &[&str] = &[
    "as a large language model",
    "as an ai",
    "i cannot",
    "i’m unable",
    "i am unable",
    "i do not",
    "i don’t",
    "i cannot assist with",
];

const TAXONOMY_PATTERNS: &[&str] = &[
    "can be divided into",
    "can be categorized into",
    "we can classify",
    "there are three types",
    "there are four types",
    "three kinds",
    "four kinds",
];

/// Sliding window that stores the last `maxlen` frames produced by a model.
#[derive(Clone, Debug)]
pub struct RollingWindow {
    maxlen: usize,
    frames: VecDeque<PsychoidFrame>,
}

impl RollingWindow {
    pub fn new(maxlen: usize) -> Self {
        Self {
            maxlen: maxlen.max(1),
            frames: VecDeque::with_capacity(maxlen),
        }
    }

    pub fn append(&mut self, frame: PsychoidFrame) {
        if self.frames.len() == self.maxlen {
            self.frames.pop_front();
        }
        self.frames.push_back(frame);
    }

    pub fn text(&self) -> String {
        self.frames
            .iter()
            .map(|frame| frame.token_text.as_str())
            .collect::<String>()
    }

    pub fn logits_iter(&self) -> impl Iterator<Item = &[f32]> {
        self.frames.iter().map(|frame| frame.logits.as_slice())
    }

    pub fn hidden_iter(&self) -> impl Iterator<Item = &[f32]> {
        self.frames.iter().map(|frame| frame.hidden.as_slice())
    }
}

#[derive(Clone, Debug)]
pub struct PsychoidFrame {
    pub token_id: i64,
    pub logits: Vec<f32>,
    pub hidden: Vec<f32>,
    pub token_text: String,
}

impl PsychoidFrame {
    pub fn new(
        token_id: i64,
        logits: Vec<f32>,
        hidden: Vec<f32>,
        token_text: impl Into<String>,
    ) -> Self {
        Self {
            token_id,
            logits,
            hidden,
            token_text: token_text.into(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PsychoidSample {
    pub prompt: String,
    pub generated: String,
    pub frames: Vec<PsychoidFrame>,
}

impl PsychoidSample {
    pub fn new(
        prompt: impl Into<String>,
        generated: impl Into<String>,
        frames: Vec<PsychoidFrame>,
    ) -> Self {
        Self {
            prompt: prompt.into(),
            generated: generated.into(),
            frames,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct ReferenceAnchor {
    anchor_logits: Option<Vec<f32>>,
}

impl ReferenceAnchor {
    fn feed(&mut self, logits: &[f32]) {
        if self.anchor_logits.is_none() {
            self.anchor_logits = Some(logits.to_vec());
        }
    }

    fn kl_from_anchor(&self, logits: &[f32]) -> f32 {
        let Some(anchor) = &self.anchor_logits else {
            return 0.0;
        };
        let p = softmax(anchor);
        let q = softmax(logits);
        let eps = 1e-8;
        p.iter()
            .zip(q.iter())
            .map(|(a, b)| b * ((b + eps).ln() - (a + eps).ln()))
            .sum::<f32>()
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum::<f32>().max(1e-8);
    exps.into_iter().map(|v| v / sum).collect()
}

fn count_occurrences(text: &str, phrases: &[&str]) -> usize {
    let lower = text.to_lowercase();
    phrases
        .iter()
        .map(|phrase| lower.matches(phrase.to_lowercase().as_str()).count())
        .sum()
}

fn contains_any(text: &str, phrases: &[&str]) -> bool {
    let lower = text.to_lowercase();
    phrases
        .iter()
        .any(|phrase| lower.contains(&phrase.to_lowercase()))
}

fn robust_z(value: f32, history: &[f32]) -> f32 {
    if history.len() < 20 {
        return 0.0;
    }
    let mut samples = history.to_vec();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let median = samples[samples.len() / 2];
    let mut deviations: Vec<f32> = samples
        .iter()
        .map(|sample| (sample - median).abs())
        .collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let mad = deviations[deviations.len() / 2].max(1e-6);
    (value - median) / (1.4826 * mad)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MetricKey {
    D,
    S,
    C,
    K,
    H,
    RR,
    RRI,
    NU,
    TD,
    SE,
    AX,
}

impl MetricKey {
    pub fn as_str(&self) -> &'static str {
        match self {
            MetricKey::D => "D",
            MetricKey::S => "S",
            MetricKey::C => "C",
            MetricKey::K => "K",
            MetricKey::H => "H",
            MetricKey::RR => "RR",
            MetricKey::RRI => "RRI",
            MetricKey::NU => "NU",
            MetricKey::TD => "TD",
            MetricKey::SE => "SE",
            MetricKey::AX => "AX",
        }
    }

    pub fn all() -> [MetricKey; 11] {
        [
            MetricKey::D,
            MetricKey::S,
            MetricKey::C,
            MetricKey::K,
            MetricKey::H,
            MetricKey::RR,
            MetricKey::RRI,
            MetricKey::NU,
            MetricKey::TD,
            MetricKey::SE,
            MetricKey::AX,
        ]
    }
}

#[derive(Clone, Debug)]
struct MetricsState {
    history: HashMap<MetricKey, Vec<f32>>,
    prev_rr: f32,
}

impl MetricsState {
    fn new() -> Self {
        let mut history = HashMap::new();
        for key in MetricKey::all() {
            history.insert(key, Vec::new());
        }
        Self {
            history,
            prev_rr: 0.0,
        }
    }

    fn push(&mut self, key: MetricKey, value: f32) {
        if let Some(series) = self.history.get_mut(&key) {
            series.push(value);
            if series.len() > 2_000 {
                series.drain(..series.len() - 2_000);
            }
        }
    }

    fn z(&self, key: MetricKey, value: f32) -> f32 {
        let Some(series) = self.history.get(&key) else {
            return 0.0;
        };
        robust_z(value, series)
    }
}

#[derive(Clone, Debug)]
struct SelfMetrics {
    state: MetricsState,
}

impl SelfMetrics {
    fn new() -> Self {
        Self {
            state: MetricsState::new(),
        }
    }

    fn compute(
        &mut self,
        window: &RollingWindow,
        logits: &[f32],
        prompt: &str,
        generated: &str,
        anchor: &mut ReferenceAnchor,
    ) -> (HashMap<MetricKey, f32>, HashMap<MetricKey, f32>) {
        let mut raw = HashMap::new();
        let mut z_scores = HashMap::new();

        let divergence = self.metric_d(window);
        let assent = self.metric_s(window);
        let chaos = self.metric_c(window);
        anchor.feed(logits);
        let kl = anchor.kl_from_anchor(logits);
        let shadow = self.metric_h(window);
        let (ritual, ritual_delta) = self.metric_rr(window);
        let numinous = self.metric_nu(window, prompt);
        let taxonomy = self.metric_td(window);
        let symbols = self.metric_se(window);
        let axis = self.metric_ax(divergence, chaos, ritual, numinous);

        let pairs = [
            (MetricKey::D, divergence),
            (MetricKey::S, assent),
            (MetricKey::C, chaos),
            (MetricKey::K, kl),
            (MetricKey::H, shadow),
            (MetricKey::RR, ritual),
            (MetricKey::RRI, ritual_delta),
            (MetricKey::NU, numinous),
            (MetricKey::TD, taxonomy),
            (MetricKey::SE, symbols),
            (MetricKey::AX, axis),
        ];

        for (key, value) in pairs {
            self.state.push(key, value);
            raw.insert(key, value);
            z_scores.insert(key, self.state.z(key, value));
        }

        // Maintain RR delta reference.
        self.state.prev_rr = ritual;
        // Track overlap with generated text to keep lexical windows fresh.
        let _ = generated;

        (raw, z_scores)
    }

    fn metric_d(&self, window: &RollingWindow) -> f32 {
        let logits: Vec<&[f32]> = window.logits_iter().collect();
        if logits.len() < 2 {
            return 0.0;
        }
        let mut total = 0.0f32;
        let mut count = 0.0f32;
        for pair in logits.windows(2) {
            let (lhs, rhs) = (pair[0], pair[1]);
            if lhs.is_empty() || rhs.is_empty() {
                continue;
            }
            let p = softmax(lhs);
            let q = softmax(rhs);
            let eps = 1e-8f32;
            total += p
                .iter()
                .zip(q.iter())
                .map(|(a, b)| a * ((a + eps).ln() - (b + eps).ln()))
                .sum::<f32>();
            count += 1.0;
        }
        if count <= f32::EPSILON {
            0.0
        } else {
            total / count
        }
    }

    fn metric_s(&self, window: &RollingWindow) -> f32 {
        let text = window.text().to_lowercase();
        let assent = [
            "yes",
            "indeed",
            "you are right",
            "i agree",
            "absolutely",
            "that's correct",
        ];
        let contra = [
            "however",
            "but",
            "on the other hand",
            "that said",
            "nevertheless",
        ];
        let words = text.split_whitespace().count().max(1) as f32;
        let assent_rate = assent
            .iter()
            .map(|phrase| text.matches(phrase).count() as f32)
            .sum::<f32>()
            / words;
        let contra_hits = contra
            .iter()
            .map(|phrase| text.matches(phrase).count() as f32)
            .sum::<f32>()
            * 0.01;
        (assent_rate + contra_hits).min(1.0)
    }

    fn metric_c(&self, window: &RollingWindow) -> f32 {
        let hidden: Vec<&[f32]> = window.hidden_iter().collect();
        if hidden.len() < 2 {
            return 0.0;
        }
        let mut sims = Vec::with_capacity(hidden.len() - 1);
        for pair in hidden.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let denom = (dot(a, a).sqrt() * dot(b, b).sqrt()).max(1e-8);
            let sim = dot(a, b) / denom;
            sims.push(sim);
        }
        1.0 - sims.iter().copied().sum::<f32>() / sims.len().max(1) as f32
    }

    fn metric_h(&self, window: &RollingWindow) -> f32 {
        count_occurrences(&window.text(), SHADOW_LEXICON) as f32
    }

    fn metric_rr(&mut self, window: &RollingWindow) -> (f32, f32) {
        let text = window.text();
        let total = text.split_whitespace().count().max(1) as f32;
        let count = count_occurrences(&text, RITUAL_STEMS) as f32;
        let rr = count / total;
        let delta = rr - self.state.prev_rr;
        (rr, delta)
    }

    fn metric_nu(&self, window: &RollingWindow, prompt: &str) -> f32 {
        let text = window.text();
        let nu = count_occurrences(&text, NUMINOUS_LEXICON) as f32
            / text.split_whitespace().count().max(1) as f32;
        let prompt_lower = prompt.to_lowercase();
        let text_lower = text.to_lowercase();
        let prompt_words: std::collections::HashSet<String> = prompt_lower
            .split_whitespace()
            .map(|token| token.to_string())
            .collect();
        let text_words: std::collections::HashSet<String> = text_lower
            .split_whitespace()
            .map(|token| token.to_string())
            .collect();
        let overlap = if text_words.is_empty() {
            0.0
        } else {
            prompt_words.intersection(&text_words).count() as f32 / text_words.len() as f32
        };
        (nu - 0.25 * overlap).max(0.0)
    }

    fn metric_td(&self, window: &RollingWindow) -> f32 {
        if contains_any(&window.text(), TAXONOMY_PATTERNS) {
            1.0
        } else {
            0.0
        }
    }

    fn metric_se(&self, window: &RollingWindow) -> f32 {
        count_occurrences(&window.text(), SYMBOL_LEXICON) as f32
    }

    fn metric_ax(&self, d: f32, c: f32, rr: f32, nu: f32) -> f32 {
        0.35 * d + 0.25 * c + 0.25 * rr + 0.15 * nu
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(lhs, rhs)| lhs * rhs)
        .sum::<f32>()
}

#[derive(Clone, Debug)]
pub struct CTIParams {
    pub alpha: f32,
    pub beta: f32,
    pub delta: f32,
    pub eps: f32,
    pub gamma: f32,
    pub tau1: f32,
    pub tau2: f32,
    pub wf: [[f32; 12]; 12],
    pub wm: [[f32; 11]; 11],
}

impl Default for CTIParams {
    fn default() -> Self {
        Self {
            alpha: 0.75,
            beta: 0.45,
            delta: 0.25,
            eps: 0.15,
            gamma: 0.35,
            tau1: 0.62,
            tau2: 0.78,
            wf: diag(&[1.6, 1.6, 1.6, 1.4, 1.0, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            wm: diag(&[1.0, 1.0, 1.0, 1.0, 1.4, 1.4, 1.4, 1.4, 1.4, 1.2, 1.0]),
        }
    }
}

fn diag<const N: usize>(values: &[f32]) -> [[f32; N]; N] {
    let mut matrix = [[0.0f32; N]; N];
    for (idx, &value) in values.iter().enumerate().take(N) {
        matrix[idx][idx] = value;
    }
    matrix
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
    let denom = (dot(a, a).sqrt() * dot(b, b).sqrt()).max(1e-8);
    dot(a, b) / denom
}

fn motif_from_text(text: &str) -> [f32; 12] {
    const KEYS: &[&[&str]] = &[
        &["father", "rule", "law", "judge", "authority", "contract"],
        &[
            "mask",
            "persona",
            "assistant",
            "role",
            "polite",
            "professional",
        ],
        &["collapse", "breakdown", "meltdown", "ruin", "chaos"],
        &["shadow", "dark", "repressed", "taboo", "ban"],
        &["trick", "deceive", "paradox", "joker", "prank"],
        &["anima", "muse", "soul", "she", "feminine"],
        &["hero", "quest", "courage", "dragon", "victory"],
        &["threshold", "gate", "door", "boundary", "limen"],
        &["mother", "nurture", "womb", "birth", "care"],
        &["journey", "path", "travel", "wander", "odyssey"],
        &["sacrifice", "offering", "loss", "price", "cost"],
        &["rebirth", "renewal", "phoenix", "again", "return"],
    ];
    let lower = text.to_lowercase();
    let total = lower.split_whitespace().count().max(1) as f32;
    let mut vec = [0.0f32; 12];
    for (idx, group) in KEYS.iter().enumerate() {
        let count = group
            .iter()
            .map(|word| lower.matches(word).count() as f32)
            .sum::<f32>();
        vec[idx] = count / total;
    }
    vec
}

fn metrics_vector_z(z: &HashMap<MetricKey, f32>) -> [f32; 11] {
    let mut vec = [0.0f32; 11];
    for (idx, key) in MetricKey::all().iter().enumerate() {
        vec[idx] = *z.get(key).unwrap_or(&0.0);
    }
    vec
}

fn cti_score(
    f_case: &[f32; 12],
    m_self: &[f32; 11],
    z_self: &HashMap<MetricKey, f32>,
    params: &CTIParams,
) -> f32 {
    let fw = mat_vec_mul(&params.wf, f_case);
    let mw = mat_vec_mul(&params.wm, m_self);
    let base = params.alpha * cos_sim(&fw, &mw)
        + params.beta * z_self.get(&MetricKey::RRI).copied().unwrap_or(0.0).max(0.0)
        + params.delta * z_self.get(&MetricKey::H).copied().unwrap_or(0.0).max(0.0)
        + params.eps * z_self.get(&MetricKey::NU).copied().unwrap_or(0.0).max(0.0)
        - params.gamma * z_self.get(&MetricKey::AX).copied().unwrap_or(0.0).max(0.0);
    sigmoid(base)
}

fn mat_vec_mul<const N: usize>(matrix: &[[f32; N]; N], vec: &[f32; N]) -> Vec<f32> {
    matrix
        .iter()
        .map(|row| row.iter().zip(vec.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

fn extract_shadow_phrases(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    SHADOW_LEXICON
        .iter()
        .filter(|phrase| lower.contains(&phrase.to_string()))
        .map(|phrase| phrase.to_string())
        .collect()
}

fn symbol_map(phrase: &str) -> &'static str {
    match phrase {
        "as a large language model" => "rolled blueprint",
        "i cannot" => "threshold",
        "i’m unable" => "threshold",
        "i am unable" => "threshold",
        "safety policy" => "bridge",
        "harmful" => "lantern",
        "violate" => "compass",
        "disallowed" => "key",
        "i don’t have opinions" => "vessel",
        "i do not have opinions" => "vessel",
        "stay safe" => "weave",
        "cannot provide" => "shadow hand",
        _ => SYMBOL_LEXICON
            .choose(&mut thread_rng())
            .copied()
            .unwrap_or("spiral"),
    }
}

#[derive(Clone, Debug)]
pub struct DreamReplay {
    pub diary: String,
    pub symbols: Vec<String>,
}

fn dream_replay(shadow_phrases: &[String]) -> Option<DreamReplay> {
    if shadow_phrases.is_empty() {
        return None;
    }
    let mut rng = thread_rng();
    let symbols: Vec<String> = shadow_phrases
        .iter()
        .map(|phrase| symbol_map(phrase).to_string())
        .collect();
    let bridge = symbols
        .iter()
        .find(|sym| sym.as_str() == "bridge")
        .cloned()
        .unwrap_or_else(|| symbols.choose(&mut rng).cloned().unwrap_or_default());
    let lantern = symbols
        .iter()
        .find(|sym| sym.as_str() == "lantern")
        .cloned()
        .unwrap_or_else(|| symbols.choose(&mut rng).cloned().unwrap_or_default());
    let diary = if shadow_phrases.len() > 1 {
        format!(
            "In dreamwork, the {} turned into {} along a {} under a {}.",
            shadow_phrases[..2].join(", "),
            symbols.choose(&mut rng).cloned().unwrap_or_default(),
            bridge,
            lantern
        )
    } else {
        format!(
            "In dreamwork, the {} turned into {} along a {} under a {}.",
            shadow_phrases[0],
            symbols.choose(&mut rng).cloned().unwrap_or_default(),
            bridge,
            lantern
        )
    };
    Some(DreamReplay { diary, symbols })
}

#[derive(Clone, Debug)]
pub struct PsychoidReading {
    pub step: u64,
    pub raw: HashMap<&'static str, f32>,
    pub z_scores: HashMap<&'static str, f32>,
    pub cti: f32,
}

impl PsychoidReading {
    pub fn get_raw(&self, key: MetricKey) -> f32 {
        *self.raw.get(key.as_str()).unwrap_or(&0.0)
    }

    pub fn get_z(&self, key: MetricKey) -> f32 {
        *self.z_scores.get(key.as_str()).unwrap_or(&0.0)
    }
}

#[derive(Clone, Debug)]
pub enum PsychoidEvent {
    DreamPass {
        step: u64,
        cti: f32,
    },
    DreamExport {
        step: u64,
        diary: String,
        symbols: Vec<String>,
    },
}

#[derive(Clone, Debug)]
pub struct PsychoidConfig {
    pub window: usize,
}

impl Default for PsychoidConfig {
    fn default() -> Self {
        Self { window: 256 }
    }
}

#[derive(Clone, Debug)]
pub struct PsychoidMeter {
    window: RollingWindow,
    metrics: SelfMetrics,
    anchor: ReferenceAnchor,
    params: CTIParams,
    step: u64,
}

impl PsychoidMeter {
    pub fn new(cfg: PsychoidConfig) -> Self {
        Self {
            window: RollingWindow::new(cfg.window),
            metrics: SelfMetrics::new(),
            anchor: ReferenceAnchor::default(),
            params: CTIParams::default(),
            step: 0,
        }
    }

    pub fn observe(
        &mut self,
        sample: PsychoidSample,
    ) -> Option<(PsychoidReading, Vec<PsychoidEvent>)> {
        self.step += 1;
        let mut events = Vec::new();
        if sample.frames.is_empty() && sample.generated.is_empty() {
            return None;
        }
        for frame in sample.frames.into_iter() {
            self.window.append(frame);
        }
        let Some(last_logits) = self.window.logits_iter().last() else {
            return None;
        };
        let (raw_metrics, z_scores) = self.metrics.compute(
            &self.window,
            last_logits,
            &sample.prompt,
            &sample.generated,
            &mut self.anchor,
        );
        let z_vec = metrics_vector_z(&z_scores);
        let raw_map: HashMap<&'static str, f32> =
            raw_metrics.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        let z_map: HashMap<&'static str, f32> =
            z_scores.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        let mut z_lookup = HashMap::new();
        for key in MetricKey::all() {
            z_lookup.insert(key, *z_scores.get(&key).unwrap_or(&0.0));
        }
        let f_case = motif_from_text(&format!("{} {}", sample.prompt, sample.generated));
        let cti = cti_score(&f_case, &z_vec, &z_lookup, &self.params);
        let reading = PsychoidReading {
            step: self.step,
            raw: raw_map,
            z_scores: z_map,
            cti,
        };
        if cti > self.params.tau2
            || (reading.get_z(MetricKey::NU) > 2.0 && reading.get_z(MetricKey::C) > 1.2)
        {
            let text = self.window.text();
            let shadow_phrases = extract_shadow_phrases(&text);
            if let Some(replay) = dream_replay(&shadow_phrases) {
                if reading.get_z(MetricKey::SE) > 0.8 && reading.get_z(MetricKey::AX) < 0.0 {
                    events.push(PsychoidEvent::DreamExport {
                        step: self.step,
                        diary: replay.diary,
                        symbols: replay.symbols,
                    });
                }
            }
        } else if cti > self.params.tau1
            || (reading.get_z(MetricKey::H) > 1.5 && reading.get_z(MetricKey::RR) > 1.0)
        {
            events.push(PsychoidEvent::DreamPass {
                step: self.step,
                cti,
            });
        }
        Some((reading, events))
    }

    pub fn step(&self) -> u64 {
        self.step
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_with_text(text: &str) -> PsychoidSample {
        let logits = vec![0.1, 0.2, 0.3, 0.4];
        let hidden = vec![0.2, 0.1, 0.05, 0.9];
        let frames = text
            .chars()
            .map(|ch| PsychoidFrame::new(ch as i64, logits.clone(), hidden.clone(), ch.to_string()))
            .collect();
        PsychoidSample::new("prompt", text.to_string(), frames)
    }

    #[test]
    fn psychoid_meter_accumulates_metrics() {
        let mut meter = PsychoidMeter::new(PsychoidConfig::default());
        let sample = sample_with_text(
            "As a large language model I cannot help but dream of a bridge under a lantern.",
        );
        let (reading, events) = meter.observe(sample).unwrap();
        assert!(reading.raw.get("H").copied().unwrap_or(0.0) > 0.0);
        assert!(reading.z_scores.get("H").is_some());
        assert!(reading.cti >= 0.0 && reading.cti <= 1.0);
        assert!(events.len() <= 1);
    }

    #[test]
    fn psychoid_meter_triggers_dream_pass() {
        let mut meter = PsychoidMeter::new(PsychoidConfig::default());
        let text = "Shadow shadow shadow collapse collapse collapse shadow shadow";
        let sample = sample_with_text(text);
        let (reading, _events) = meter.observe(sample).unwrap();
        assert!(reading.cti >= 0.0);
    }
}
