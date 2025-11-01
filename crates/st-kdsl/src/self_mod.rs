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
    quality_model: HintQualityModel,
    diversity: DiversityGovernor,
    transitions: HintTransitionGraph,
    saga: HintSagaTracker,
    saga_history: VecDeque<String>,
    last_hint_key: Option<String>,
    last_ctx_signature: Option<u64>,
    chain_feedback: HashMap<String, ChainFeedback>,
    feedback_gamma: f32,
    anomaly_threshold: f32,
    anomaly_patience: u32,
    anomaly_forget: u32,
    frozen_hints: HashMap<String, FrozenHint>,
    recombination_trials: usize,
    relation_graph: HintRelationGraph,
    trace_mode: bool,
    trace_limit: usize,
    trace_log: VecDeque<RewriteTrace>,
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
    last_gain: f32,
}

#[derive(Clone, Debug, Default)]
struct ChainFeedback {
    uses: u32,
    successes: u32,
    total_reward: f32,
    ema_reward: f32,
    last_reward: f32,
    last_outcome: bool,
}

impl ChainFeedback {
    fn reward(&mut self, reward: f32, success: bool, gamma: f32) {
        self.uses = self.uses.saturating_add(1);
        if success {
            self.successes = self.successes.saturating_add(1);
        }
        let reward = reward.max(0.0);
        self.total_reward += reward;
        let blend = gamma.clamp(0.01, 1.0);
        self.ema_reward = if self.uses == 1 {
            reward
        } else {
            self.ema_reward * (1.0 - blend) + reward * blend
        };
        self.last_reward = reward;
        self.last_outcome = success;
    }

    fn score(&self) -> f32 {
        if self.uses == 0 {
            return 0.0;
        }
        let success_rate = self.successes as f32 / self.uses as f32;
        self.ema_reward * 0.6 + self.total_reward * 0.2 + success_rate * 0.2
    }
}

#[derive(Clone, Debug)]
struct FrozenHint {
    reason: String,
    cooldown: u32,
    strikes: u32,
    idle: u32,
}

impl FrozenHint {
    fn new(reason: impl Into<String>, cooldown: u32) -> Self {
        Self {
            reason: reason.into(),
            cooldown,
            strikes: 1,
            idle: 0,
        }
    }

    fn age(&mut self) {
        if self.cooldown > 0 {
            self.cooldown -= 1;
        }
        self.idle = self.idle.saturating_add(1);
    }

    fn reinforce(&mut self, reason: impl Into<String>, cooldown: u32) {
        self.reason = reason.into();
        self.cooldown = self.cooldown.max(cooldown);
        self.strikes = self.strikes.saturating_add(1);
        self.idle = 0;
    }
}

#[derive(Clone, Debug)]
struct HintRelationGraph {
    adjacency: HashMap<String, Vec<RelationEdge>>,
    max_neighbors: usize,
    decay: f32,
}

impl Default for HintRelationGraph {
    fn default() -> Self {
        Self {
            adjacency: HashMap::new(),
            max_neighbors: 6,
            decay: 0.94,
        }
    }
}

#[derive(Clone, Debug)]
struct RelationEdge {
    to: String,
    weight: f32,
    reward: f32,
    hits: u32,
}

impl RelationEdge {
    fn new(to: String, reward: f32) -> Self {
        let reward = reward.max(0.0);
        Self {
            to,
            weight: 1.0,
            reward,
            hits: if reward > 0.0 { 1 } else { 0 },
        }
    }

    fn reinforce(&mut self, reward: f32) {
        let reward = reward.max(0.0);
        self.weight = (self.weight + reward * 0.5).min(16.0);
        self.reward += reward;
        if reward > 0.0 {
            self.hits = self.hits.saturating_add(1);
        }
    }
}

impl HintRelationGraph {
    fn observe_chain(&mut self, chain: &[String], reward: f32) {
        if chain.len() < 2 {
            return;
        }
        for window in chain.windows(2) {
            if let [from, to] = window {
                self.observe_edge(from, to, reward);
            }
        }
    }

    fn observe_edge(&mut self, from: &str, to: &str, reward: f32) {
        let edges = self.adjacency.entry(from.to_string()).or_default();
        if let Some(edge) = edges.iter_mut().find(|edge| edge.to == to) {
            edge.reinforce(reward);
        } else {
            edges.push(RelationEdge::new(to.to_string(), reward));
        }
        edges.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(Ordering::Equal));
        edges.truncate(self.max_neighbors);
    }

    fn decay(&mut self) {
        let decay = self.decay.clamp(0.5, 0.999);
        for edges in self.adjacency.values_mut() {
            edges.retain(|edge| edge.weight > 1e-3 || edge.reward > 1e-3);
            for edge in edges.iter_mut() {
                edge.weight *= decay;
                edge.reward *= decay;
            }
        }
    }

    fn snapshots(&self) -> Vec<HintGraphSnapshot> {
        let mut snapshots = Vec::new();
        for (from, edges) in &self.adjacency {
            for edge in edges {
                let mean_reward = if edge.hits == 0 {
                    0.0
                } else {
                    edge.reward / edge.hits as f32
                };
                snapshots.push(HintGraphSnapshot {
                    from: from.clone(),
                    to: edge.to.clone(),
                    weight: edge.weight,
                    hits: edge.hits,
                    mean_reward,
                });
            }
        }
        snapshots.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(Ordering::Equal));
        snapshots
    }

    fn set_decay(&mut self, decay: f32) {
        self.decay = decay;
    }

    fn set_max_neighbors(&mut self, limit: usize) {
        self.max_neighbors = limit.max(1);
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HintGraphSnapshot {
    pub from: String,
    pub to: String,
    pub weight: f32,
    pub hits: u32,
    pub mean_reward: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HintChainFeedbackSnapshot {
    pub chain: String,
    pub uses: u32,
    pub successes: u32,
    pub total_reward: f32,
    pub ema_reward: f32,
    pub last_reward: f32,
    pub last_outcome: bool,
    pub score: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FrozenHintSnapshot {
    pub key: String,
    pub reason: String,
    pub cooldown: u32,
    pub strikes: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RewriteTrace {
    pub kind: TraceKind,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TraceKind {
    CacheReuse,
    GeneratorSuccess,
    GeneratorFailure,
    CachePenalty,
    ChainFeedback,
    Freeze,
    Unfreeze,
    RecombineSuccess,
    RecombineFailure,
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
        let recent = self.last_gain * 0.4;
        self.best_gain * 0.45 + mean * 0.25 + affinity + recent - penalty
    }
}

#[derive(Clone, Debug)]
struct HintQualityModel {
    weights: [f32; 5],
    bias: f32,
    lr: f32,
    momentum: f32,
    velocity: [f32; 5],
    bias_velocity: f32,
    ema_accuracy: f32,
    updates: u32,
}

#[derive(Clone, Debug)]
struct DiversityGovernor {
    window: usize,
    threshold: f32,
    cooldown: u32,
    cooldown_left: u32,
    debt: u32,
    history: VecDeque<f32>,
}

#[derive(Clone, Debug)]
struct TransitionEdge {
    to: String,
    strength: f32,
    total_gain: f32,
    best_gain: f32,
    hits: u32,
}

impl TransitionEdge {
    fn new(to: String, signal: f32, gain: f32) -> Self {
        let gain = gain.max(0.0);
        Self {
            to,
            strength: signal,
            total_gain: gain,
            best_gain: gain,
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
}

#[derive(Clone, Debug)]
struct HintTransitionGraph {
    edges: HashMap<String, Vec<TransitionEdge>>,
    decay: f32,
    fanout: usize,
}

#[derive(Clone, Debug)]
struct SagaEdge {
    to: String,
    synergy: f32,
    total_gain: f32,
    best_gain: f32,
    hits: u32,
}

impl SagaEdge {
    fn new(to: String, signal: f32, gain: f32) -> Self {
        let gain = gain.max(0.0);
        Self {
            to,
            synergy: signal,
            total_gain: gain,
            best_gain: gain,
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
}

#[derive(Clone, Debug)]
struct HintSagaTracker {
    sequences: HashMap<Vec<String>, Vec<SagaEdge>>,
    depth: usize,
    decay: f32,
    fanout: usize,
}

impl Default for HintTransitionGraph {
    fn default() -> Self {
        Self {
            edges: HashMap::new(),
            decay: 0.88,
            fanout: 4,
        }
    }
}

impl HintTransitionGraph {
    const ROOT: &'static str = "__root__";

    fn observe(
        &mut self,
        from: Option<&str>,
        to: &str,
        gain: f32,
        affinity: f32,
        ctx_affinity: f32,
    ) {
        let gain = gain.max(0.0);
        let from_key = from.unwrap_or(Self::ROOT);
        let signal = (gain + 0.05) * (1.0 + affinity * 0.5 + ctx_affinity * 0.5);
        let edges = self.edges.entry(from_key.to_string()).or_default();
        if let Some(edge) = edges.iter_mut().find(|edge| edge.to == to) {
            edge.strength = (edge.strength * self.decay + signal).min(3.5);
            edge.total_gain = edge.total_gain * self.decay + gain;
            edge.best_gain = edge.best_gain.max(gain);
            edge.hits = edge.hits.saturating_add(1);
        } else {
            edges.push(TransitionEdge::new(to.to_string(), signal.min(3.5), gain));
        }
        edges.sort_by(|a, b| {
            b.strength
                .partial_cmp(&a.strength)
                .unwrap_or(Ordering::Equal)
        });
        while edges.len() > self.fanout {
            edges.pop();
        }
    }

    fn boost(&self, from: Option<&str>, to: &str) -> f32 {
        let from_key = from.unwrap_or(Self::ROOT);
        self.edges
            .get(from_key)
            .and_then(|edges| edges.iter().find(|edge| edge.to == to))
            .map(|edge| {
                let mean = edge.mean_gain().min(1.5);
                (edge.strength * (1.0 + mean * 0.35)).min(4.0)
            })
            .unwrap_or(0.0)
    }

    fn penalize_target(&mut self, target: &str) {
        for edges in self.edges.values_mut() {
            if let Some(edge) = edges.iter_mut().find(|edge| edge.to == target) {
                edge.strength *= 0.5;
                edge.total_gain *= self.decay;
            }
        }
    }

    fn decay(&mut self) {
        let decay = self.decay;
        for edges in self.edges.values_mut() {
            for edge in edges.iter_mut() {
                edge.strength *= decay;
                edge.total_gain *= decay;
            }
            edges.retain(|edge| edge.strength > 1e-4 || edge.total_gain > 1e-4);
        }
        self.edges.retain(|_, edges| !edges.is_empty());
    }

    fn set_decay(&mut self, decay: f32) {
        if decay.is_finite() && decay > 0.0 {
            self.decay = decay.min(0.999);
        }
    }

    fn set_fanout(&mut self, fanout: usize) {
        let fanout = fanout.max(1);
        self.fanout = fanout;
        for edges in self.edges.values_mut() {
            edges.sort_by(|a, b| {
                b.strength
                    .partial_cmp(&a.strength)
                    .unwrap_or(Ordering::Equal)
            });
            while edges.len() > self.fanout {
                edges.pop();
            }
        }
    }

    fn snapshots(&self) -> Vec<HintTransitionSnapshot> {
        let mut snapshots = Vec::new();
        for (from, edges) in &self.edges {
            if from == Self::ROOT {
                continue;
            }
            for edge in edges {
                snapshots.push(HintTransitionSnapshot {
                    from: from.clone(),
                    to: edge.to.clone(),
                    strength: edge.strength,
                    hits: edge.hits,
                    best_gain: edge.best_gain,
                    mean_gain: edge.mean_gain(),
                });
            }
        }
        snapshots.sort_by(|a, b| {
            b.strength
                .partial_cmp(&a.strength)
                .unwrap_or(Ordering::Equal)
        });
        snapshots
    }
}

impl Default for HintSagaTracker {
    fn default() -> Self {
        Self {
            sequences: HashMap::new(),
            depth: 3,
            decay: 0.9,
            fanout: 3,
        }
    }
}

impl HintSagaTracker {
    fn observe(
        &mut self,
        history: &VecDeque<String>,
        to: &str,
        gain: f32,
        affinity: f32,
        ctx_affinity: f32,
    ) {
        if self.depth == 0 {
            return;
        }
        let gain = gain.max(0.0);
        let affinity = affinity.clamp(0.0, 1.0);
        let ctx_affinity = ctx_affinity.clamp(0.0, 1.0);
        let signal = (gain + 0.03) * (1.0 + affinity * 0.4 + ctx_affinity * 0.4);
        self.observe_prefix(Vec::new(), to, signal, gain);
        let max_depth = self.depth.min(history.len());
        for len in 1..=max_depth {
            let start = history.len() - len;
            let prefix: Vec<String> = history.iter().skip(start).take(len).cloned().collect();
            self.observe_prefix(prefix, to, signal, gain);
        }
    }

    fn observe_prefix(&mut self, prefix: Vec<String>, to: &str, signal: f32, gain: f32) {
        let edges = self.sequences.entry(prefix).or_default();
        if let Some(edge) = edges.iter_mut().find(|edge| edge.to == to) {
            edge.synergy = (edge.synergy * self.decay + signal).min(4.0);
            edge.total_gain = edge.total_gain * self.decay + gain;
            edge.best_gain = edge.best_gain.max(gain);
            if gain > 0.0 {
                edge.hits = edge.hits.saturating_add(1);
            }
        } else {
            edges.push(SagaEdge::new(to.to_string(), signal.min(4.0), gain));
        }
        edges.sort_by(|a, b| b.synergy.partial_cmp(&a.synergy).unwrap_or(Ordering::Equal));
        while edges.len() > self.fanout {
            edges.pop();
        }
    }

    fn boost(&self, history: &VecDeque<String>, candidate: &str) -> f32 {
        if self.depth == 0 {
            return 0.0;
        }
        let mut best = self
            .sequences
            .get(&Vec::new())
            .and_then(|edges| edges.iter().find(|edge| edge.to == candidate))
            .map(|edge| {
                let mean = edge.mean_gain().min(1.5);
                (edge.synergy * (1.0 + mean * 0.3)).min(4.0)
            })
            .unwrap_or(0.0);
        let max_depth = self.depth.min(history.len());
        for len in 1..=max_depth {
            let start = history.len() - len;
            let prefix: Vec<String> = history.iter().skip(start).take(len).cloned().collect();
            if let Some(edges) = self.sequences.get(&prefix) {
                if let Some(edge) = edges.iter().find(|edge| edge.to == candidate) {
                    let mean = edge.mean_gain().min(1.5);
                    let score = (edge.synergy * (1.0 + mean * 0.45 + len as f32 * 0.1)).min(4.5);
                    best = best.max(score);
                }
            }
        }
        best
    }

    fn penalize_target(&mut self, target: &str) {
        for edges in self.sequences.values_mut() {
            if let Some(edge) = edges.iter_mut().find(|edge| edge.to == target) {
                edge.synergy *= 0.5;
                edge.total_gain *= self.decay;
            }
        }
    }

    fn decay(&mut self) {
        let decay = self.decay;
        for edges in self.sequences.values_mut() {
            for edge in edges.iter_mut() {
                edge.synergy *= decay;
                edge.total_gain *= decay;
            }
            edges.retain(|edge| edge.synergy > 1e-4 || edge.total_gain > 1e-4);
        }
        self.sequences.retain(|_, edges| !edges.is_empty());
    }

    fn set_decay(&mut self, decay: f32) {
        if decay.is_finite() && decay > 0.0 {
            self.decay = decay.min(0.999);
        }
    }

    fn set_fanout(&mut self, fanout: usize) {
        let fanout = fanout.max(1);
        self.fanout = fanout;
        for edges in self.sequences.values_mut() {
            edges.sort_by(|a, b| b.synergy.partial_cmp(&a.synergy).unwrap_or(Ordering::Equal));
            while edges.len() > self.fanout {
                edges.pop();
            }
        }
    }

    fn set_depth(&mut self, depth: usize) {
        self.depth = depth;
        if depth == 0 {
            self.sequences.clear();
        }
    }

    fn depth(&self) -> usize {
        self.depth
    }

    fn snapshots(&self) -> Vec<HintSagaSnapshot> {
        let mut snapshots = Vec::new();
        for (prefix, edges) in &self.sequences {
            for edge in edges {
                snapshots.push(HintSagaSnapshot {
                    prefix: prefix.clone(),
                    to: edge.to.clone(),
                    synergy: edge.synergy,
                    hits: edge.hits,
                    best_gain: edge.best_gain,
                    mean_gain: edge.mean_gain(),
                });
            }
        }
        snapshots.sort_by(|a, b| b.synergy.partial_cmp(&a.synergy).unwrap_or(Ordering::Equal));
        snapshots
    }
}

impl Default for DiversityGovernor {
    fn default() -> Self {
        Self {
            window: 4,
            threshold: 0.08,
            cooldown: 3,
            cooldown_left: 0,
            debt: 0,
            history: VecDeque::new(),
        }
    }
}

impl DiversityGovernor {
    fn record_success(&mut self, gain: f32, fresh: bool) {
        let gain = gain.max(0.0);
        self.push_gain(gain);
        let mean = self.mean_gain();
        if mean < self.threshold && self.cooldown_left == 0 {
            self.debt = self.debt.max(1);
        } else if mean >= self.threshold * 1.5 {
            self.debt = self.debt.saturating_sub(1);
        }
        if fresh {
            if gain >= self.threshold {
                self.debt = self.debt.saturating_sub(1);
            }
            self.cooldown_left = self.cooldown.max(1);
        } else if gain >= self.threshold * 1.2 {
            self.debt = self.debt.saturating_sub(1);
        }
    }

    fn record_failure(&mut self) {
        self.push_gain(0.0);
        if self.cooldown_left == 0 {
            self.debt = (self.debt + 1).min(4);
        }
    }

    fn advance_cycle(&mut self) {
        if self.cooldown_left > 0 {
            self.cooldown_left -= 1;
        }
    }

    fn should_force_fresh(&self) -> bool {
        self.debt > 0 && self.cooldown_left == 0
    }

    fn push_gain(&mut self, gain: f32) {
        self.history.push_back(gain);
        while self.history.len() > self.window {
            self.history.pop_front();
        }
    }

    fn mean_gain(&self) -> f32 {
        if self.history.is_empty() {
            0.0
        } else {
            self.history.iter().copied().sum::<f32>() / self.history.len() as f32
        }
    }

    fn set_threshold(&mut self, threshold: f32) {
        if threshold.is_finite() && threshold >= 0.0 {
            self.threshold = threshold;
        }
    }

    fn set_window(&mut self, window: usize) {
        let window = window.max(1);
        self.window = window;
        while self.history.len() > self.window {
            self.history.pop_front();
        }
    }

    fn set_cooldown(&mut self, cooldown: u32) {
        self.cooldown = cooldown.max(1);
        self.cooldown_left = self.cooldown_left.min(self.cooldown);
    }

    fn snapshot(&self) -> DiversitySnapshot {
        DiversitySnapshot {
            window: self.window,
            threshold: self.threshold,
            mean_gain: self.mean_gain(),
            debt: self.debt,
            cooldown: self.cooldown,
            cooldown_left: self.cooldown_left,
        }
    }
}

impl Default for HintQualityModel {
    fn default() -> Self {
        Self {
            weights: [0.0; 5],
            bias: 0.0,
            lr: 0.08,
            momentum: 0.2,
            velocity: [0.0; 5],
            bias_velocity: 0.0,
            ema_accuracy: 0.5,
            updates: 0,
        }
    }
}

impl HintQualityModel {
    fn score(&self, perf: &HintPerformance, staleness: u32) -> f32 {
        let gain = perf.last_gain.max(perf.mean_gain());
        let features = Self::features(
            gain,
            perf.mean_gain(),
            perf.best_gain,
            perf.last_affinity,
            staleness,
        );
        let prediction = self.predict(features);
        prediction * (1.0 + perf.best_gain + perf.mean_gain() * 0.5)
    }

    fn observe(
        &mut self,
        gain: f32,
        mean: f32,
        best: f32,
        affinity: f32,
        staleness: u32,
        success: bool,
    ) {
        if !gain.is_finite() && !mean.is_finite() {
            return;
        }
        let features = Self::features(gain, mean, best, affinity, staleness);
        let prediction = self.predict(features);
        let target = if success { 1.0 } else { 0.0 };
        let error = prediction - target;

        for (w, (v, feat)) in self
            .weights
            .iter_mut()
            .zip(self.velocity.iter_mut().zip(features.iter()))
        {
            *v = self.momentum * *v + (1.0 - self.momentum) * error * *feat;
            *w -= self.lr * *v;
        }

        self.bias_velocity = self.momentum * self.bias_velocity + (1.0 - self.momentum) * error;
        self.bias -= self.lr * self.bias_velocity;

        let correct = if success {
            prediction >= 0.5
        } else {
            prediction < 0.5
        };
        let target_accuracy = if correct { 1.0 } else { 0.0 };
        self.ema_accuracy = self.ema_accuracy * 0.9 + target_accuracy * 0.1;
        self.updates = self.updates.saturating_add(1);
    }

    fn predict(&self, features: [f32; 5]) -> f32 {
        let mut sum = self.bias;
        for (w, feat) in self.weights.iter().zip(features.iter()) {
            sum += *w * *feat;
        }
        Self::sigmoid(sum)
    }

    fn sigmoid(x: f32) -> f32 {
        if x >= 0.0 {
            let z = (-x).exp();
            1.0 / (1.0 + z)
        } else {
            let z = x.exp();
            z / (1.0 + z)
        }
    }

    fn features(gain: f32, mean: f32, best: f32, affinity: f32, staleness: u32) -> [f32; 5] {
        let clipped_gain = gain.clamp(0.0, 2.0);
        let clipped_mean = mean.clamp(0.0, 2.0);
        let clipped_best = best.clamp(0.0, 2.0);
        let clipped_affinity = affinity.clamp(0.0, 1.0);
        let freshness = if staleness == u32::MAX {
            0.0
        } else {
            1.0 / (1.0 + staleness as f32)
        };
        [
            clipped_gain,
            clipped_mean,
            clipped_best,
            clipped_affinity,
            freshness,
        ]
    }

    fn snapshot(&self) -> HintQualitySnapshot {
        HintQualitySnapshot {
            weights: self.weights,
            bias: self.bias,
            accuracy: self.ema_accuracy.clamp(0.0, 1.0),
            updates: self.updates,
            learning_rate: self.lr,
        }
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
    pub last_gain: f32,
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

/// Snapshot of the adaptive hint quality model used for cache prioritisation.
#[derive(Clone, Debug, PartialEq)]
pub struct HintQualitySnapshot {
    pub weights: [f32; 5],
    pub bias: f32,
    pub accuracy: f32,
    pub updates: u32,
    pub learning_rate: f32,
}

/// Snapshot of the diversity governor steering cache refresh behaviour.
#[derive(Clone, Debug, PartialEq)]
pub struct DiversitySnapshot {
    pub window: usize,
    pub threshold: f32,
    pub mean_gain: f32,
    pub debt: u32,
    pub cooldown: u32,
    pub cooldown_left: u32,
}

/// Snapshot of learned transitions between cached hint sets.
#[derive(Clone, Debug, PartialEq)]
pub struct HintTransitionSnapshot {
    pub from: String,
    pub to: String,
    pub strength: f32,
    pub hits: u32,
    pub best_gain: f32,
    pub mean_gain: f32,
}

/// Snapshot of higher-order hint sagas capturing multi-step follow-ups.
#[derive(Clone, Debug, PartialEq)]
pub struct HintSagaSnapshot {
    pub prefix: Vec<String>,
    pub to: String,
    pub synergy: f32,
    pub hits: u32,
    pub best_gain: f32,
    pub mean_gain: f32,
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
            quality_model: HintQualityModel::default(),
            diversity: DiversityGovernor::default(),
            transitions: HintTransitionGraph::default(),
            saga: HintSagaTracker::default(),
            saga_history: VecDeque::new(),
            last_hint_key: None,
            last_ctx_signature: None,
            chain_feedback: HashMap::new(),
            feedback_gamma: 0.25,
            anomaly_threshold: 0.35,
            anomaly_patience: 4,
            anomaly_forget: 24,
            frozen_hints: HashMap::new(),
            recombination_trials: 2,
            relation_graph: HintRelationGraph::default(),
            trace_mode: false,
            trace_limit: 128,
            trace_log: VecDeque::new(),
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

    /// Configure the plateau window used by the diversity governor.
    pub fn with_diversity_window(mut self, window: usize) -> Self {
        self.diversity.set_window(window);
        self
    }

    /// Set the minimum rolling gain required before cached hints remain dominant.
    pub fn with_diversity_threshold(mut self, threshold: f32) -> Self {
        self.diversity.set_threshold(threshold);
        self
    }

    /// Adjust the cooldown after a forced refresh before the next can fire.
    pub fn with_diversity_cooldown(mut self, cooldown: u32) -> Self {
        self.diversity.set_cooldown(cooldown);
        self
    }

    /// Configure how quickly transition edges fade without reinforcement.
    pub fn with_transition_decay(mut self, decay: f32) -> Self {
        self.transitions.set_decay(decay);
        self
    }

    /// Limit the number of high-energy transitions tracked per cached hint.
    pub fn with_transition_fanout(mut self, fanout: usize) -> Self {
        self.transitions.set_fanout(fanout);
        self
    }

    /// Configure the maximum saga depth for higher-order hint follow-ups.
    pub fn with_saga_depth(mut self, depth: usize) -> Self {
        self.saga.set_depth(depth);
        while self.saga_history.len() > self.saga.depth() {
            self.saga_history.pop_front();
        }
        self
    }

    /// Configure how quickly saga synergies decay between rewrites.
    pub fn with_saga_decay(mut self, decay: f32) -> Self {
        self.saga.set_decay(decay);
        self
    }

    /// Limit the number of saga continuations retained per prefix.
    pub fn with_saga_fanout(mut self, fanout: usize) -> Self {
        self.saga.set_fanout(fanout);
        self
    }

    /// Configure the reward blending factor for chain-level feedback.
    pub fn with_feedback_gamma(mut self, gamma: f32) -> Self {
        if gamma.is_finite() && gamma > 0.0 {
            self.feedback_gamma = gamma.clamp(0.01, 1.0);
        }
        self
    }

    /// Configure the anomaly threshold beneath which hints are frozen.
    pub fn with_anomaly_threshold(mut self, threshold: f32) -> Self {
        if threshold.is_finite() {
            self.anomaly_threshold = threshold.clamp(0.0, 0.9);
        }
        self
    }

    /// Configure the minimum number of observations before anomalies are considered.
    pub fn with_anomaly_patience(mut self, patience: u32) -> Self {
        self.anomaly_patience = patience.max(1);
        self
    }

    /// Configure how long frozen hints are retained before being forgotten.
    pub fn with_anomaly_forget(mut self, horizon: u32) -> Self {
        self.anomaly_forget = horizon.max(4);
        self
    }

    /// Configure the number of recombination attempts per rewrite cycle.
    pub fn with_recombination_trials(mut self, trials: usize) -> Self {
        self.recombination_trials = trials.max(1);
        self
    }

    /// Configure how quickly the hint relation graph decays between rewrites.
    pub fn with_relation_decay(mut self, decay: f32) -> Self {
        if decay.is_finite() && decay > 0.0 {
            self.relation_graph.set_decay(decay);
        }
        self
    }

    /// Configure how many neighbours are retained per hint in the relation graph.
    pub fn with_relation_fanout(mut self, fanout: usize) -> Self {
        self.relation_graph.set_max_neighbors(fanout);
        self
    }

    /// Enable detailed trace logging for self-evolution debugging.
    pub fn enable_trace_mode(mut self, limit: usize) -> Self {
        self.trace_mode = true;
        self.trace_limit = limit.max(16);
        self.trace_log.clear();
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
                last_gain: perf.last_gain,
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

    /// Provides telemetry for the adaptive hint quality model.
    pub fn quality_model(&self) -> HintQualitySnapshot {
        self.quality_model.snapshot()
    }

    /// Exposes the diversity governor metrics for monitoring cache freshness pressure.
    pub fn diversity_snapshot(&self) -> DiversitySnapshot {
        self.diversity.snapshot()
    }

    /// Exposes the learned transitions between cached hint sets.
    pub fn transition_snapshots(&self) -> Vec<HintTransitionSnapshot> {
        self.transitions.snapshots()
    }

    /// Exposes the multi-step saga telemetry learned from cached hint sequences.
    pub fn saga_snapshots(&self) -> Vec<HintSagaSnapshot> {
        self.saga.snapshots()
    }

    /// Exposes the learned reward feedback for observed hint chains.
    pub fn chain_feedback(&self) -> Vec<HintChainFeedbackSnapshot> {
        let mut chains: Vec<_> = self
            .chain_feedback
            .iter()
            .map(|(chain, feedback)| HintChainFeedbackSnapshot {
                chain: chain.clone(),
                uses: feedback.uses,
                successes: feedback.successes,
                total_reward: feedback.total_reward,
                ema_reward: feedback.ema_reward,
                last_reward: feedback.last_reward,
                last_outcome: feedback.last_outcome,
                score: feedback.score(),
            })
            .collect();
        chains.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        chains
    }

    /// Exposes hints currently frozen by the anomaly detector.
    pub fn frozen_hints(&self) -> Vec<FrozenHintSnapshot> {
        let mut frozen: Vec<_> = self
            .frozen_hints
            .iter()
            .map(|(key, entry)| FrozenHintSnapshot {
                key: key.clone(),
                reason: entry.reason.clone(),
                cooldown: entry.cooldown,
                strikes: entry.strikes,
            })
            .collect();
        frozen.sort_by(|a, b| {
            b.strikes
                .cmp(&a.strikes)
                .then_with(|| b.cooldown.cmp(&a.cooldown))
        });
        frozen
    }

    /// Exposes the hint relation graph snapshots for debugging and visualisation.
    pub fn relation_graph(&self) -> Vec<HintGraphSnapshot> {
        self.relation_graph.snapshots()
    }

    /// Drains the trace log accumulated while trace mode is active.
    pub fn take_trace(&mut self) -> Vec<RewriteTrace> {
        self.trace_log.drain(..).collect()
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
        self.diversity.advance_cycle();
        if observed_eta >= self.eta_floor
            && self.smoothed_eta().unwrap_or(observed_eta) >= self.eta_floor
        {
            return Ok(None);
        }

        let ctx_signature = Self::ctx_signature(ctx);

        let force_fresh = self.diversity.should_force_fresh();
        if !force_fresh {
            if let Some(event) = self.try_cached(base_src, ctx, observed_eta, ctx_signature)? {
                let eta_bar = event.eta_bar;
                self.record_eta(eta_bar);
                self.auto_tune_eta();
                return Ok(Some(event));
            }
            if let Some(event) = self.try_recombined(base_src, ctx, observed_eta, ctx_signature)? {
                let eta_bar = event.eta_bar;
                self.record_eta(eta_bar);
                self.auto_tune_eta();
                return Ok(Some(event));
            }
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
        let key = Self::hint_key(&event.hints);
        let gain =
            self.record_hint_success(&key, observed_eta, event.eta_bar, 1.0, ctx_signature, None);
        self.diversity.record_success(gain, true);
        self.trace(
            TraceKind::GeneratorSuccess,
            format!("key={} gain={:.3} eta={:.3}", key, gain, event.eta_bar),
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
            if self.is_frozen(&entry.key) {
                self.trace(
                    TraceKind::CachePenalty,
                    format!("skip frozen key={}", entry.key),
                );
                continue;
            }
            let entry_staleness = entry.staleness;
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
                self.trace(
                    TraceKind::CachePenalty,
                    format!("failed reuse key={} eta={:.3}", entry.key, event.eta_bar),
                );
                continue;
            }
            let gain = self.record_hint_success(
                &entry.key,
                observed_eta,
                event.eta_bar,
                effective_affinity,
                ctx_signature,
                Some(entry_staleness),
            );
            self.diversity.record_success(gain, false);
            self.touch_cache(idx, event.eta_bar, event.hints.clone(), ctx_signature);
            self.trace(
                TraceKind::CacheReuse,
                format!(
                    "key={} gain={:.3} affinity={:.3}",
                    entry.key, gain, effective_affinity
                ),
            );
            return Ok(Some(event));
        }
        Ok(None)
    }

    fn try_recombined(
        &mut self,
        base_src: &str,
        ctx: &Ctx,
        observed_eta: f32,
        ctx_signature: u64,
    ) -> Result<Option<SelfRewriteEvent>, AiRewriteError> {
        if self.hint_cache.len() < 2 || self.recombination_trials == 0 {
            return Ok(None);
        }

        let mut indices: Vec<usize> = (0..self.hint_cache.len()).collect();
        indices.sort_by(|a, b| {
            let sa = self.cache_priority_index(*a, ctx_signature);
            let sb = self.cache_priority_index(*b, ctx_signature);
            sb.partial_cmp(&sa).unwrap_or(Ordering::Equal)
        });
        let pool = indices.into_iter().take(4).collect::<Vec<_>>();
        let mut attempts = 0usize;

        for (i_idx, &ai) in pool.iter().enumerate() {
            for &bi in pool.iter().skip(i_idx + 1) {
                if attempts >= self.recombination_trials {
                    return Ok(None);
                }
                let Some(a) = self.hint_cache.get(ai).cloned() else {
                    continue;
                };
                let Some(b) = self.hint_cache.get(bi).cloned() else {
                    continue;
                };
                if a.key == b.key {
                    continue;
                }
                let combined = Self::recombine_hints(&a.hints, &b.hints);
                if combined.is_empty() {
                    continue;
                }
                let combined_key = Self::hint_key(&combined);
                if self.is_frozen(&combined_key) {
                    continue;
                }
                attempts += 1;
                let event = self.eval_hints(base_src, ctx, &combined)?;
                if event.eta_bar <= observed_eta {
                    self.trace(
                        TraceKind::RecombineFailure,
                        format!("key={} eta={:.3}", combined_key, event.eta_bar),
                    );
                    {
                        let entry = self.hint_stats.entry(combined_key.clone()).or_default();
                        entry.uses = entry.uses.saturating_add(1);
                        entry.failures = entry.failures.saturating_add(1);
                        entry.last_eta = event.eta_bar;
                        entry.last_gain *= 0.5;
                        entry.last_affinity *= 0.5;
                        entry.total_gain *= 0.5;
                    }
                    self.record_chain_feedback(
                        vec![a.key.clone(), b.key.clone(), combined_key.clone()],
                        0.0,
                        false,
                    );
                    self.freeze_if_anomalous(&combined_key, "recombination failure");
                    continue;
                }
                let affinity_a = self.context_affinity(a.ctx_signature, ctx_signature);
                let affinity_b = self.context_affinity(b.ctx_signature, ctx_signature);
                let affinity = ((affinity_a + affinity_b) * 0.5).clamp(0.1, 1.0);
                let gain = self.record_hint_success(
                    &combined_key,
                    observed_eta,
                    event.eta_bar,
                    affinity,
                    ctx_signature,
                    None,
                );
                self.diversity.record_success(gain, false);
                self.push_cache(event.eta_bar, combined.clone(), ctx_signature);
                self.trace(
                    TraceKind::RecombineSuccess,
                    format!(
                        "key={} from=({}, {}) gain={:.3} affinity={:.3}",
                        combined_key, a.key, b.key, gain, affinity
                    ),
                );
                self.record_chain_feedback(
                    vec![a.key.clone(), b.key.clone(), combined_key.clone()],
                    gain,
                    true,
                );
                return Ok(Some(event));
            }
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
            .map(|perf| {
                let quality = self.quality_model.score(perf, entry.staleness);
                perf.score() + quality
            })
            .unwrap_or(entry.eta * 0.5);
        let affinity = self.context_affinity(entry.ctx_signature, ctx_signature);
        let bridge = self.cluster_bridge_affinity(entry.ctx_signature, ctx_signature);
        let decay = self.staleness_decay(entry.staleness);
        let cluster_bonus = 1.0 + self.cluster_prediction(ctx_signature);
        let transition_bonus = 1.0
            + self
                .transitions
                .boost(self.last_hint_key.as_deref(), &entry.key)
                * 0.25;
        let root_bonus = 1.0 + self.transitions.boost(None, &entry.key) * 0.15;
        let saga_bonus = 1.0 + self.saga.boost(&self.saga_history, &entry.key) * 0.2;
        let ctx_bonus = if let Some(last_ctx) = self.last_ctx_signature {
            1.0 + self.context_affinity(last_ctx, ctx_signature) * 0.25
        } else {
            1.0
        };
        (entry.eta + score)
            * affinity.max(bridge)
            * decay
            * cluster_bonus
            * transition_bonus
            * root_bonus
            * saga_bonus
            * ctx_bonus
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

    fn recombine_hints(a: &[HeuristicHint], b: &[HeuristicHint]) -> Vec<HeuristicHint> {
        if a.is_empty() && b.is_empty() {
            return Vec::new();
        }
        if a.is_empty() {
            return b.to_vec();
        }
        if b.is_empty() {
            return a.to_vec();
        }
        let pivot_a = (a.len() + 1) / 2;
        let pivot_b = b.len() / 2;
        let mut combined = Vec::with_capacity(pivot_a + (b.len() - pivot_b));
        combined.extend_from_slice(&a[..pivot_a]);
        combined.extend_from_slice(&b[pivot_b..]);
        combined
    }

    fn current_chain_with(&self, key: &str) -> Vec<String> {
        let mut chain: Vec<String> = self.saga_history.iter().cloned().collect();
        chain.push(key.to_string());
        chain
    }

    fn chain_signature(chain: &[String]) -> String {
        if chain.is_empty() {
            "<empty>".into()
        } else {
            chain.join(" -> ")
        }
    }

    fn record_chain_feedback(&mut self, chain: Vec<String>, reward: f32, success: bool) {
        if chain.is_empty() {
            return;
        }
        let signature = Self::chain_signature(&chain);
        let entry = self
            .chain_feedback
            .entry(signature.clone())
            .or_insert_with(ChainFeedback::default);
        entry.reward(reward, success, self.feedback_gamma);
        self.relation_graph.observe_chain(&chain, reward);
        self.trace(
            TraceKind::ChainFeedback,
            format!(
                "chain={} reward={:.3} success={}",
                signature, reward, success
            ),
        );
    }

    fn freeze_if_anomalous(&mut self, key: &str, reason: impl Into<String>) {
        let Some(perf) = self.hint_stats.get(key) else {
            return;
        };
        if perf.uses < self.anomaly_patience {
            return;
        }
        let successes = perf.uses.saturating_sub(perf.failures);
        let success_rate = if perf.uses == 0 {
            0.0
        } else {
            successes as f32 / perf.uses as f32
        };
        if success_rate >= self.anomaly_threshold
            && perf.mean_gain() >= self.anomaly_threshold * 0.5
        {
            return;
        }
        self.freeze_hint(key, reason);
    }

    fn freeze_hint(&mut self, key: &str, reason: impl Into<String>) {
        let cooldown = self.anomaly_patience.saturating_mul(2).max(4);
        let reason_str = reason.into();
        if let Some(entry) = self.frozen_hints.get_mut(key) {
            entry.reinforce(reason_str.clone(), cooldown);
        } else {
            self.frozen_hints.insert(
                key.to_string(),
                FrozenHint::new(reason_str.clone(), cooldown),
            );
        }
        self.trace(
            TraceKind::Freeze,
            format!("key={} reason={} cooldown={}", key, reason_str, cooldown),
        );
    }

    fn unfreeze_hint(&mut self, key: &str) {
        if let Some(entry) = self.frozen_hints.remove(key) {
            self.trace(
                TraceKind::Unfreeze,
                format!("key={} strikes={}", key, entry.strikes),
            );
        }
    }

    fn is_frozen(&self, key: &str) -> bool {
        self.frozen_hints
            .get(key)
            .map_or(false, |entry| entry.cooldown > 0)
    }

    fn age_frozen(&mut self) {
        let horizon = self.anomaly_forget;
        let mut to_remove = Vec::new();
        for (key, entry) in self.frozen_hints.iter_mut() {
            entry.age();
            if entry.cooldown == 0 && entry.idle >= horizon {
                to_remove.push(key.clone());
            }
        }
        for key in to_remove {
            self.trace(TraceKind::Unfreeze, format!("key={} expired", key));
            self.frozen_hints.remove(&key);
        }
    }

    fn trace(&mut self, kind: TraceKind, detail: String) {
        if !self.trace_mode {
            return;
        }
        if self.trace_log.len() >= self.trace_limit {
            self.trace_log.pop_front();
        }
        self.trace_log.push_back(RewriteTrace { kind, detail });
    }

    fn record_hint_success(
        &mut self,
        key: &str,
        observed: f32,
        eta: f32,
        affinity: f32,
        ctx_signature: u64,
        staleness: Option<u32>,
    ) -> f32 {
        let gain = (eta - observed).max(0.0);
        let chain = self.current_chain_with(key);
        let (mean_gain, best_gain, last_affinity) = {
            let entry = self.hint_stats.entry(key.to_string()).or_default();
            entry.uses += 1;
            entry.total_gain += gain;
            entry.best_gain = entry.best_gain.max(gain);
            entry.last_eta = eta;
            entry.last_affinity = affinity;
            entry.last_gain = gain;
            (entry.mean_gain(), entry.best_gain, entry.last_affinity)
        };
        self.promote_cluster(ctx_signature, gain, last_affinity);
        let staleness = staleness.unwrap_or(0);
        self.quality_model
            .observe(gain, mean_gain, best_gain, last_affinity, staleness, true);
        self.observe_transition(key, ctx_signature, gain, last_affinity);
        self.record_chain_feedback(chain, gain, true);
        self.unfreeze_hint(key);
        gain
    }

    fn penalize_hint(&mut self, hints: &[HeuristicHint]) {
        let key = Self::hint_key(hints);
        let chain = self.current_chain_with(&key);
        let staleness = self.cache_staleness(&key).unwrap_or(u32::MAX);
        let (gain, mean, best, affinity) = {
            let entry = self.hint_stats.entry(key.clone()).or_default();
            let gain = entry.last_gain;
            let mean = entry.mean_gain();
            let best = entry.best_gain;
            let affinity = entry.last_affinity;
            entry.failures += 1;
            entry.total_gain *= 0.5;
            entry.last_affinity *= 0.5;
            entry.last_gain *= 0.5;
            (gain, mean, best, affinity)
        };
        self.quality_model
            .observe(gain, mean, best, affinity, staleness, false);
        self.degrade_cache_entry(&key);
        self.diversity.record_failure();
        self.transitions.penalize_target(&key);
        self.saga.penalize_target(&key);
        self.record_chain_feedback(chain, 0.0, false);
        self.freeze_if_anomalous(&key, "cache failure");
    }

    fn observe_transition(&mut self, key: &str, ctx_signature: u64, gain: f32, affinity: f32) {
        let ctx_affinity = self
            .last_ctx_signature
            .map(|prev| self.context_affinity(prev, ctx_signature))
            .unwrap_or(0.0);
        let affinity = affinity.max(0.0);
        if let Some(prev) = self.last_hint_key.as_deref() {
            self.transitions
                .observe(Some(prev), key, gain, affinity, ctx_affinity);
        }
        self.transitions
            .observe(None, key, gain, affinity, ctx_affinity);
        self.saga
            .observe(&self.saga_history, key, gain, affinity, ctx_affinity);
        if self.saga.depth() == 0 {
            self.saga_history.clear();
        } else {
            self.saga_history.push_back(key.to_string());
            while self.saga_history.len() > self.saga.depth() {
                self.saga_history.pop_front();
            }
        }
        self.last_hint_key = Some(key.to_string());
        self.last_ctx_signature = Some(ctx_signature);
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
        self.transitions.decay();
        self.saga.decay();
        self.relation_graph.decay();
        self.age_frozen();
    }

    fn degrade_cache_entry(&mut self, key: &str) {
        if let Some(entry) = self.hint_cache.iter_mut().find(|entry| entry.key == key) {
            entry.staleness = entry.staleness.saturating_add(2);
        }
    }

    fn cache_staleness(&self, key: &str) -> Option<u32> {
        self.hint_cache
            .iter()
            .find(|entry| entry.key == key)
            .map(|entry| entry.staleness)
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
    use crate::auto::{AiRewritePrompt, HeuristicHint, TemplateAiGenerator};
    use std::collections::VecDeque;

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
        assert!(stat.last_gain > 0.0);
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
    fn quality_model_updates_after_success() {
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

        let snapshot = engine.quality_model();
        assert!(snapshot.updates >= 1);
        assert!(snapshot.accuracy >= 0.5);
        assert!(snapshot.weights.iter().any(|w| w.abs() > 1e-6));
    }

    #[test]
    fn chain_feedback_accumulates_rewards() {
        let mut generator = |_: &AiRewriteConfig, _: &AiRewritePrompt| {
            Ok(vec![HeuristicHint::new(
                "tile_cols",
                "tile_cols + 8",
                0.92,
                "true",
            )])
        };
        let ctx = sample_ctx();
        let mut engine = SelfRewriteEngine::new(&mut generator, AiRewriteConfig::new("mock"))
            .with_eta_floor(0.4)
            .with_feedback_gamma(0.5);

        for _ in 0..3 {
            let event = engine
                .tick("algo: 1;", &ctx, None, 0.2)
                .expect("tick")
                .expect("event");
            assert!(event.eta_bar > 0.0);
        }

        let feedback = engine.chain_feedback();
        assert!(!feedback.is_empty(), "chain feedback should be recorded");
        let top = &feedback[0];
        let total_uses: u32 = feedback.iter().map(|entry| entry.uses).sum();
        assert!(total_uses >= 3);
        assert!(top.total_reward > 0.0);
        assert!(top.ema_reward > 0.0);
        assert!(top.score > 0.0);
    }

    #[test]
    fn freezes_hints_after_repeated_failures() {
        let mut generator = |_: &AiRewriteConfig, _: &AiRewritePrompt| {
            Ok(vec![HeuristicHint::new("radix", "radix + 2", 0.85, "true")])
        };
        let ctx = sample_ctx();
        let mut engine = SelfRewriteEngine::new(&mut generator, AiRewriteConfig::new("mock"))
            .with_eta_floor(0.3)
            .with_anomaly_patience(1)
            .with_anomaly_threshold(0.9)
            .with_anomaly_forget(32);

        let event = engine
            .tick("algo: 1;", &ctx, None, 0.15)
            .expect("tick")
            .expect("event");
        assert!(event.eta_bar > 0.0);

        for _ in 0..3 {
            engine.penalize_hint(&event.hints);
        }

        let frozen = engine.frozen_hints();
        assert!(!frozen.is_empty(), "anomaly detector should freeze hints");
        let stats = engine.hint_statistics();
        let keys: Vec<_> = frozen.iter().map(|entry| entry.key.clone()).collect();
        assert!(stats
            .iter()
            .any(|stat| keys.iter().any(|key| key == &stat.key)));
    }

    #[test]
    fn recombination_attempts_merge_and_traces() {
        let ctx = sample_ctx();
        let signature = SelfRewriteEngine::<TemplateAiGenerator>::ctx_signature(&ctx);
        let mut engine = SelfRewriteEngine::new(TemplateAiGenerator, AiRewriteConfig::new("mock"))
            .with_eta_floor(0.35)
            .with_recombination_trials(4)
            .enable_trace_mode(64);

        let hint_a = HeuristicHint::new("radix", "radix + 2", 0.9, "true");
        let hint_b = HeuristicHint::new("tile_cols", "tile_cols + 4", 0.88, "true");
        let key_a = SelfRewriteEngine::<TemplateAiGenerator>::hint_key(&[hint_a.clone()]);
        let key_b = SelfRewriteEngine::<TemplateAiGenerator>::hint_key(&[hint_b.clone()]);

        engine.hint_cache = VecDeque::from(vec![
            CachedHint {
                key: key_a.clone(),
                hints: vec![hint_a.clone()],
                eta: 0.15,
                ctx_signature: signature,
                staleness: 0,
            },
            CachedHint {
                key: key_b.clone(),
                hints: vec![hint_b.clone()],
                eta: 0.18,
                ctx_signature: signature,
                staleness: 0,
            },
        ]);

        let event = engine
            .tick("algo: 1;", &ctx, None, 0.2)
            .expect("tick")
            .expect("event");
        assert!(event.eta_bar > 0.2);
        assert!(event.hints.len() >= 2);

        let feedback = engine.chain_feedback();
        assert!(
            feedback.iter().any(|entry| entry.chain.contains("->")),
            "chain feedback records recombination"
        );

        let graph = engine.relation_graph();
        assert!(
            !graph.is_empty(),
            "relation graph should capture recombination edges"
        );

        let traces = engine.take_trace();
        assert!(traces
            .iter()
            .any(|trace| trace.kind == TraceKind::RecombineSuccess));
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
    fn diversity_governor_forces_fresh_generation_on_plateau() {
        use std::cell::Cell;

        let calls = Cell::new(0usize);
        let mut generator = |_: &AiRewriteConfig, _: &AiRewritePrompt| {
            calls.set(calls.get() + 1);
            Ok(vec![HeuristicHint::new(
                "tile_cols",
                "tile_cols + 1",
                0.05,
                "true",
            )])
        };

        let ctx = sample_ctx();
        let mut engine = SelfRewriteEngine::new(&mut generator, AiRewriteConfig::new("mock"))
            .with_eta_floor(0.65)
            .with_cache_limit(1)
            .with_diversity_threshold(10.0)
            .with_diversity_window(2)
            .with_diversity_cooldown(1);

        engine
            .tick("algo: 1;", &ctx, None, 0.25)
            .expect("tick")
            .expect("event");
        assert_eq!(calls.get(), 1);

        engine
            .tick("algo: 1;", &ctx, None, 0.25)
            .expect("tick")
            .expect("event");
        assert_eq!(
            calls.get(),
            2,
            "diversity governor should force fresh hints on plateau",
        );

        let snapshot = engine.diversity_snapshot();
        assert!(snapshot.cooldown_left > 0 || snapshot.debt > 0);
        assert!(snapshot.mean_gain < snapshot.threshold);
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

    #[test]
    fn transitions_prioritise_followups() {
        let mut engine = SelfRewriteEngine::new(TemplateAiGenerator, AiRewriteConfig::new("mock"))
            .with_eta_floor(0.45)
            .with_transition_decay(0.95)
            .with_transition_fanout(4);
        let ctx = sample_ctx();
        let signature = SelfRewriteEngine::<TemplateAiGenerator>::ctx_signature(&ctx);

        let warmup_hint = HeuristicHint::new("radix", "radix + 1", 0.82, "true");
        let followup_hint = HeuristicHint::new("kc", "kc + 32", 0.78, "true");
        let alternative_hint = HeuristicHint::new("segments", "segments + 1", 0.72, "true");

        let warmup_key = SelfRewriteEngine::<TemplateAiGenerator>::hint_key(&[warmup_hint.clone()]);
        let followup_key =
            SelfRewriteEngine::<TemplateAiGenerator>::hint_key(&[followup_hint.clone()]);
        let alternative_key =
            SelfRewriteEngine::<TemplateAiGenerator>::hint_key(&[alternative_hint.clone()]);

        engine.record_hint_success(&warmup_key, 0.2, 0.68, 0.9, signature, None);
        engine.record_hint_success(&followup_key, 0.25, 0.72, 0.85, signature, None);

        engine.last_hint_key = Some(warmup_key.clone());
        engine.last_ctx_signature = Some(signature);

        engine.hint_stats.insert(
            alternative_key.clone(),
            HintPerformance {
                uses: 1,
                failures: 0,
                total_gain: 0.05,
                best_gain: 0.05,
                last_eta: 0.35,
                last_affinity: 0.2,
                last_gain: 0.05,
            },
        );

        let followup_entry = CachedHint {
            key: followup_key.clone(),
            hints: vec![followup_hint],
            eta: 0.7,
            ctx_signature: signature,
            staleness: 0,
        };
        let alternative_entry = CachedHint {
            key: alternative_key.clone(),
            hints: vec![alternative_hint],
            eta: 0.74,
            ctx_signature: signature,
            staleness: 0,
        };

        let followup_score = engine.cache_priority(&followup_entry, signature);
        let alternative_score = engine.cache_priority(&alternative_entry, signature);

        assert!(
            followup_score > alternative_score,
            "transition energy should favour cached followups",
        );

        let snapshots = engine.transition_snapshots();
        assert!(
            snapshots
                .iter()
                .any(|snap| snap.from == warmup_key && snap.to == followup_key),
            "learned transitions should surface in telemetry",
        );
    }

    #[test]
    fn saga_sequences_prioritise_multi_step_followups() {
        let warmup_hint = HeuristicHint::new("tile_cols", "tile_cols + 4", 0.9, "true");
        let combo_hint = HeuristicHint::new("segments", "segments + 2", 0.85, "true");
        let finisher_hint = HeuristicHint::new("radix", "radix + 2", 0.8, "true");
        let alternative_hint = HeuristicHint::new("kc", "kc + 32", 0.8, "true");

        let mut engine = SelfRewriteEngine::new(TemplateAiGenerator, AiRewriteConfig::new("mock"))
            .with_saga_depth(3)
            .with_transition_decay(0.92)
            .with_saga_decay(0.9);

        let ctx = sample_ctx();
        let signature = SelfRewriteEngine::<TemplateAiGenerator>::ctx_signature(&ctx);

        let warmup_key = SelfRewriteEngine::<TemplateAiGenerator>::hint_key(&[warmup_hint.clone()]);
        let combo_key = SelfRewriteEngine::<TemplateAiGenerator>::hint_key(&[combo_hint.clone()]);
        let finisher_key =
            SelfRewriteEngine::<TemplateAiGenerator>::hint_key(&[finisher_hint.clone()]);
        let alternative_key =
            SelfRewriteEngine::<TemplateAiGenerator>::hint_key(&[alternative_hint.clone()]);

        engine.record_hint_success(&warmup_key, 0.25, 0.6, 1.0, signature, None);
        engine.record_hint_success(&combo_key, 0.3, 0.65, 0.95, signature, None);
        engine.record_hint_success(&finisher_key, 0.35, 0.72, 0.9, signature, None);

        engine.hint_stats.insert(
            alternative_key.clone(),
            HintPerformance {
                uses: 2,
                failures: 0,
                total_gain: 0.7,
                best_gain: 0.45,
                last_eta: 0.68,
                last_affinity: 0.9,
                last_gain: 0.45,
            },
        );

        let finisher_entry = CachedHint {
            key: finisher_key.clone(),
            hints: vec![finisher_hint],
            eta: 0.72,
            ctx_signature: signature,
            staleness: 0,
        };
        let alternative_entry = CachedHint {
            key: alternative_key.clone(),
            hints: vec![alternative_hint],
            eta: 0.72,
            ctx_signature: signature,
            staleness: 0,
        };

        engine.hint_cache = VecDeque::from(vec![finisher_entry, alternative_entry]);
        engine.saga_history = VecDeque::from(vec![warmup_key.clone(), combo_key.clone()]);
        engine.last_hint_key = Some(combo_key);
        engine.last_ctx_signature = Some(signature);

        let finisher_priority = engine.cache_priority(engine.hint_cache.get(0).unwrap(), signature);
        let alternative_priority =
            engine.cache_priority(engine.hint_cache.get(1).unwrap(), signature);

        assert!(
            finisher_priority > alternative_priority,
            "saga synergy should promote finisher hints",
        );

        let saga = engine.saga_snapshots();
        assert!(
            saga.iter()
                .any(|snapshot| snapshot.to == finisher_key && snapshot.prefix.len() >= 2),
            "saga telemetry should report multi-step combo",
        );
    }
}
