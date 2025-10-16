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

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime};

use st_core::backend::unison_heuristics::RankKind;
use st_core::runtime::blackcat::zmeta::ZMetaParams;
use st_core::runtime::blackcat::{
    bandit::SoftBanditMode, BlackCatRuntime, ChoiceGroups, StepMetrics,
};

/// Mode that dictates how a roundtable node participates in distributed consensus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistMode {
    /// Keep the roundtable local. No summaries are exported.
    LocalOnly,
    /// Periodically push compact summaries to a meta layer for light-weight promotion.
    PeriodicMeta,
    /// Fully participate in the meta layer and accept remote proposals automatically.
    FullyGlobal,
}

impl Default for DistMode {
    fn default() -> Self {
        DistMode::LocalOnly
    }
}

/// Configuration that connects a local roundtable to the distributed coordination fabric.
#[derive(Debug, Clone)]
pub struct DistConfig {
    pub node_id: String,
    pub mode: DistMode,
    pub push_interval: Duration,
    pub meta_endpoints: Vec<String>,
    pub summary_window: usize,
}

impl Default for DistConfig {
    fn default() -> Self {
        Self {
            node_id: "local".to_string(),
            mode: DistMode::LocalOnly,
            push_interval: Duration::from_secs(30),
            meta_endpoints: Vec::new(),
            summary_window: 32,
        }
    }
}

/// Gradient band that won a local negotiation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutcomeBand {
    Above,
    Here,
    Beneath,
}

impl OutcomeBand {
    pub fn from_weights(above: f32, here: f32, beneath: f32) -> Self {
        if above >= here && above >= beneath {
            OutcomeBand::Above
        } else if here >= beneath {
            OutcomeBand::Here
        } else {
            OutcomeBand::Beneath
        }
    }
}

/// Compact representation of a single local decision.
#[derive(Debug, Clone)]
pub struct DecisionEvent {
    pub plan_signature: String,
    pub script_hint: String,
    pub rank_kind: RankKind,
    pub winner: OutcomeBand,
    pub score: f32,
    pub wilson_low: f32,
    pub wilson_high: f32,
    pub psi_total: Option<f32>,
    pub band_energy: (f32, f32, f32),
    pub drift: f32,
    pub z_signal: f32,
    pub timestamp: SystemTime,
    pub sequence: u64,
}

impl DecisionEvent {
    fn reliability(&self) -> f32 {
        let window = (self.wilson_high - self.wilson_low).abs().max(1e-4);
        let psi_scale = self.psi_total.unwrap_or(1.0).max(0.1);
        let z_factor = (1.0 + self.z_signal * 0.5).clamp(0.25, 1.75);
        (self.score / psi_scale).clamp(0.0, 1.0) * (1.0 - window.min(1.0)) * z_factor
    }
}

/// Summary shipped from a worker to the meta layer.
#[derive(Debug, Clone)]
pub struct MetaSummary {
    pub node_id: String,
    pub plan_signature: String,
    pub script_hint: String,
    pub rank_kind: RankKind,
    pub winner: OutcomeBand,
    pub mean_score: f32,
    pub wilson_low: f32,
    pub wilson_high: f32,
    pub mean_psi: f32,
    pub mean_z: f32,
    pub support: f32,
    pub events: usize,
    pub issued_at: SystemTime,
}

impl MetaSummary {
    fn from_events(node_id: &str, events: &[DecisionEvent]) -> Option<Self> {
        if events.is_empty() {
            return None;
        }
        let mut sum_score = 0.0f32;
        let mut sum_psi = 0.0f32;
        let mut support = 0.0f32;
        let mut winner_counts: HashMap<OutcomeBand, usize> = HashMap::new();
        let mut latest = SystemTime::UNIX_EPOCH;
        let mut sum_z = 0.0f32;
        for event in events {
            sum_score += event.score;
            if let Some(psi) = event.psi_total {
                sum_psi += psi;
            }
            support += event.reliability();
            *winner_counts.entry(event.winner).or_insert(0) += 1;
            if event.timestamp > latest {
                latest = event.timestamp;
            }
            sum_z += event.z_signal;
        }
        let dominant = winner_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(band, _)| band)
            .unwrap_or(OutcomeBand::Here);
        let first = &events[0];
        let mean_score = sum_score / events.len() as f32;
        let mean_psi = if events.iter().any(|e| e.psi_total.is_some()) {
            sum_psi / events.len() as f32
        } else {
            0.0
        };
        let mean_z = sum_z / events.len() as f32;
        let (wilson_low, wilson_high) = aggregate_wilson(events);
        Some(Self {
            node_id: node_id.to_string(),
            plan_signature: first.plan_signature.clone(),
            script_hint: first.script_hint.clone(),
            rank_kind: first.rank_kind,
            winner: dominant,
            mean_score,
            wilson_low,
            wilson_high,
            mean_psi,
            mean_z,
            support: support.max(0.0),
            events: events.len(),
            issued_at: latest,
        })
    }
}

fn aggregate_wilson(events: &[DecisionEvent]) -> (f32, f32) {
    if events.is_empty() {
        return (0.0, 0.0);
    }
    let mean = events.iter().map(|e| e.score).sum::<f32>() / events.len() as f32;
    let n = events.len() as f32;
    wilson_bounds(mean.clamp(0.0, 1.0), n.max(1.0), 1.0)
}

fn wilson_bounds(p_hat: f32, n: f32, z: f32) -> (f32, f32) {
    let denom = 1.0 + (z * z) / n;
    let centre = p_hat + (z * z) / (2.0 * n);
    let margin = (p_hat * (1.0 - p_hat) + (z * z) / (4.0 * n))
        .max(0.0)
        .sqrt()
        * z
        / n.sqrt();
    let low = ((centre - margin) / denom).clamp(0.0, 1.0);
    let high = ((centre + margin) / denom).clamp(0.0, 1.0);
    (low, high)
}

/// Operation applied to the heur.kdsl op-log.
#[derive(Debug, Clone)]
pub enum HeurOpKind {
    AppendSoft { script: String, weight: f32 },
    Retract { script_hash: u64 },
    Annotate { script_hash: u64, note: String },
}

#[derive(Debug, Clone)]
pub struct HeurOp {
    pub origin: String,
    pub kind: HeurOpKind,
    pub issued_at: SystemTime,
}

/// Proposal broadcast from the meta layer back to workers.
#[derive(Debug, Clone)]
pub struct GlobalProposal {
    pub proposal_id: String,
    pub ops: Vec<HeurOp>,
    pub evidence: Vec<MetaSummary>,
}

/// Minimal CRDT-style op-log to keep heur.kdsl edits deterministic.
#[derive(Debug, Default, Clone)]
pub struct HeurOpLog {
    entries: Vec<HeurOp>,
}

impl HeurOpLog {
    pub fn append(&mut self, op: HeurOp) {
        self.entries.push(op);
    }

    pub fn merge(&mut self, other: &HeurOpLog) {
        for op in &other.entries {
            if !self
                .entries
                .iter()
                .any(|existing| existing.proposal_fingerprint() == op.proposal_fingerprint())
            {
                self.entries.push(op.clone());
            }
        }
    }

    pub fn entries(&self) -> &[HeurOp] {
        &self.entries
    }
}

impl HeurOp {
    fn proposal_fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.origin.hash(&mut hasher);
        match &self.kind {
            HeurOpKind::AppendSoft { script, weight } => {
                script.hash(&mut hasher);
                weight.to_bits().hash(&mut hasher);
            }
            HeurOpKind::Retract { script_hash } => {
                script_hash.hash(&mut hasher);
            }
            HeurOpKind::Annotate { script_hash, note } => {
                script_hash.hash(&mut hasher);
                note.hash(&mut hasher);
            }
        }
        self.issued_at
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|dur| dur.as_nanos())
            .unwrap_or_default()
            .hash(&mut hasher);
        hasher.finish()
    }
}

/// Local node that buffers decisions and emits summaries when required.
#[derive(Debug)]
pub struct RoundtableNode {
    config: DistConfig,
    pending: VecDeque<DecisionEvent>,
    last_flush: Instant,
    next_sequence: u64,
}

impl RoundtableNode {
    pub fn new(config: DistConfig) -> Self {
        Self {
            config,
            pending: VecDeque::new(),
            last_flush: Instant::now(),
            next_sequence: 0,
        }
    }

    pub fn config(&self) -> &DistConfig {
        &self.config
    }

    pub fn record_decision(
        &mut self,
        plan_signature: String,
        script_hint: String,
        rank_kind: RankKind,
        winner: OutcomeBand,
        score: f32,
        psi_total: Option<f32>,
        band_energy: (f32, f32, f32),
        drift: f32,
        z_signal: f32,
    ) -> Option<MetaSummary> {
        if matches!(self.config.mode, DistMode::LocalOnly) {
            return None;
        }
        let score = score.clamp(0.0, 1.0);
        let (wilson_low, wilson_high) = wilson_bounds(score, 1.0, 1.0);
        let event = DecisionEvent {
            plan_signature,
            script_hint,
            rank_kind,
            winner,
            score,
            wilson_low,
            wilson_high,
            psi_total,
            band_energy,
            drift,
            z_signal,
            timestamp: SystemTime::now(),
            sequence: self.next_sequence,
        };
        self.next_sequence += 1;
        self.pending.push_back(event);
        if self.pending.len() >= self.config.summary_window
            || self.last_flush.elapsed() >= self.config.push_interval
        {
            self.last_flush = Instant::now();
            let mut batch = Vec::new();
            while let Some(event) = self.pending.pop_front() {
                batch.push(event);
            }
            return MetaSummary::from_events(&self.config.node_id, &batch);
        }
        None
    }

    pub fn drain(&mut self) {
        self.pending.clear();
    }

    pub fn retune(&mut self, push_interval: Duration, summary_window: usize) {
        let clamped_interval = push_interval.max(Duration::from_millis(10));
        self.config.push_interval = clamped_interval;
        self.config.summary_window = summary_window.max(1);
        if self.pending.len() >= self.config.summary_window {
            self.last_flush = Instant::now() - self.config.push_interval;
        }
    }
}

/// Meta-layer controller that aggregates summaries and emits proposals.
#[derive(Debug)]
pub struct MetaConductor {
    threshold: f32,
    participation: usize,
    plans: HashMap<String, PlanConsensus>,
}

#[derive(Debug)]
struct PlanConsensus {
    script_hint: String,
    evidence: HashMap<String, MetaSummary>,
}

impl MetaConductor {
    pub fn new(threshold: f32, participation: usize) -> Self {
        Self {
            threshold,
            participation,
            plans: HashMap::new(),
        }
    }

    pub fn ingest(&mut self, summary: MetaSummary) -> Option<GlobalProposal> {
        let entry = self
            .plans
            .entry(summary.plan_signature.clone())
            .or_insert_with(|| PlanConsensus {
                script_hint: summary.script_hint.clone(),
                evidence: HashMap::new(),
            });
        entry.script_hint = summary.script_hint.clone();
        entry
            .evidence
            .insert(summary.node_id.clone(), summary.clone());
        let support: f32 = entry
            .evidence
            .values()
            .map(|s| s.support.max(0.0) * (1.0 + s.mean_z * 0.4).clamp(0.3, 1.7))
            .sum();
        if entry.evidence.len() >= self.participation && support >= self.threshold {
            let ops = vec![HeurOp {
                origin: "meta".to_string(),
                kind: HeurOpKind::AppendSoft {
                    script: entry.script_hint.clone(),
                    weight: (support / self.threshold).min(2.0),
                },
                issued_at: SystemTime::now(),
            }];
            let evidence = entry.evidence.values().cloned().collect::<Vec<_>>();
            entry.evidence.clear();
            return Some(GlobalProposal {
                proposal_id: format!("proposal-{}", ops[0].proposal_fingerprint()),
                ops,
                evidence,
            });
        }
        None
    }
}

/// Simulates the effect of a global proposal before committing it.
pub fn simulate_proposal_locally(
    proposal: &GlobalProposal,
    log: &mut HeurOpLog,
) -> (bool, HashMap<String, f32>) {
    let mut metrics = HashMap::new();
    for op in &proposal.ops {
        match &op.kind {
            HeurOpKind::AppendSoft { script, weight } => {
                metrics.insert(script.clone(), *weight);
            }
            HeurOpKind::Retract { script_hash } => {
                metrics.insert(format!("retract:{}", script_hash), 1.0);
            }
            HeurOpKind::Annotate { script_hash, note } => {
                metrics.insert(format!("annotate:{}", script_hash), note.len() as f32);
            }
        }
    }
    log.append(HeurOp {
        origin: "local-preview".to_string(),
        kind: HeurOpKind::Annotate {
            script_hash: proposal
                .ops
                .get(0)
                .map(|op| op.proposal_fingerprint())
                .unwrap_or(0),
            note: "preview".to_string(),
        },
        issued_at: SystemTime::now(),
    });
    (true, metrics)
}

/// Minutes captured by the Blackcat moderator after each summary.
#[derive(Clone, Debug)]
pub struct ModeratorMinutes {
    pub plan_signature: String,
    pub script_hint: String,
    pub winner: OutcomeBand,
    pub support: f32,
    pub mean_score: f32,
    pub mean_psi: f32,
    pub mean_z: f32,
    pub confidence: (f32, f32),
    pub picks: HashMap<String, String>,
    pub reward: f64,
    pub notes: String,
    pub issued_at: SystemTime,
}

/// Result of the moderator ingest step.
#[derive(Clone, Debug)]
pub struct ModeratorOutcome {
    pub minutes: ModeratorMinutes,
    pub proposal: Option<GlobalProposal>,
}

/// Moderator that seats the Blackcat runtime between workers and the meta layer.
pub struct BlackcatModerator {
    conductor: MetaConductor,
    runtime: BlackCatRuntime,
    history: Vec<ModeratorMinutes>,
    history_limit: usize,
}

impl core::fmt::Debug for BlackcatModerator {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BlackcatModerator")
            .field("history", &self.history.len())
            .field("history_limit", &self.history_limit)
            .finish()
    }
}

impl BlackcatModerator {
    /// Builds a moderator with the supplied runtime.
    pub fn new(runtime: BlackCatRuntime, threshold: f32, participation: usize) -> Self {
        Self {
            conductor: MetaConductor::new(threshold, participation),
            runtime,
            history: Vec::new(),
            history_limit: 64,
        }
    }

    /// Builds a moderator that uses a lightweight, opinionated runtime configuration.
    pub fn with_default_runtime(threshold: f32, participation: usize) -> Self {
        let mut groups = HashMap::new();
        groups.insert(
            "agenda".to_string(),
            vec![
                "focus".to_string(),
                "branch".to_string(),
                "synchronize".to_string(),
            ],
        );
        groups.insert(
            "pace".to_string(),
            vec![
                "steady".to_string(),
                "accelerate".to_string(),
                "reflect".to_string(),
            ],
        );
        groups.insert(
            "integration".to_string(),
            vec![
                "local".to_string(),
                "federated".to_string(),
                "global".to_string(),
            ],
        );
        let runtime = BlackCatRuntime::new(
            ZMetaParams::default(),
            ChoiceGroups { groups },
            8,
            SoftBanditMode::TS,
            None,
        );
        Self::new(runtime, threshold, participation)
    }

    /// Ingests a summary, updates the runtime, and optionally emits a global proposal.
    pub fn ingest(&mut self, summary: MetaSummary) -> ModeratorOutcome {
        let context = self.build_context(&summary);
        let picks = self.runtime.choose(context);
        let metrics = self.build_metrics(&summary);
        let reward = self.runtime.post_step(&metrics);
        let notes = format!(
            "blackcat moderated {} → {:?} (score {:.3}, support {:.3}, z {:.3})",
            summary.plan_signature,
            summary.winner,
            summary.mean_score,
            summary.support,
            summary.mean_z
        );
        let minutes = ModeratorMinutes {
            plan_signature: summary.plan_signature.clone(),
            script_hint: summary.script_hint.clone(),
            winner: summary.winner,
            support: summary.support,
            mean_score: summary.mean_score,
            mean_psi: summary.mean_psi,
            mean_z: summary.mean_z,
            confidence: (summary.wilson_low, summary.wilson_high),
            picks,
            reward,
            notes,
            issued_at: summary.issued_at,
        };
        if self.history.len() >= self.history_limit {
            self.history.remove(0);
        }
        self.history.push(minutes.clone());
        let proposal = self.conductor.ingest(summary);
        ModeratorOutcome { minutes, proposal }
    }

    fn build_context(&self, summary: &MetaSummary) -> Vec<f64> {
        let mut ctx = vec![
            1.0,
            summary.mean_score as f64,
            summary.support as f64,
            summary.mean_psi as f64,
            summary.mean_z as f64,
            (summary.wilson_high - summary.wilson_low) as f64,
            summary.events as f64,
        ];
        ctx.resize(self.runtime.context_dim().max(7), 0.0);
        ctx
    }

    fn build_metrics(&self, summary: &MetaSummary) -> StepMetrics {
        let mut extra = HashMap::new();
        extra.insert("meta_support".to_string(), summary.support as f64);
        extra.insert("meta_mean_score".to_string(), summary.mean_score as f64);
        extra.insert("meta_mean_psi".to_string(), summary.mean_psi as f64);
        extra.insert("meta_mean_z".to_string(), summary.mean_z as f64);
        extra.insert(
            "meta_confidence_width".to_string(),
            (summary.wilson_high - summary.wilson_low) as f64,
        );
        extra.insert("meta_events".to_string(), summary.events as f64);
        StepMetrics {
            step_time_ms: (1.0 - summary.mean_score as f64).abs() * 100.0,
            mem_peak_mb: summary.mean_psi.abs() as f64 * 256.0,
            retry_rate: (summary.wilson_high - summary.wilson_low).abs() as f64,
            extra,
        }
    }

    /// Returns the rolling minutes captured so far.
    pub fn minutes(&self) -> &[ModeratorMinutes] {
        &self.history
    }

    /// Overrides the maximum stored minutes.
    pub fn set_history_limit(&mut self, limit: usize) {
        self.history_limit = limit.max(1);
        self.trim_history();
    }

    /// Absorbs externally captured minutes, keeping the rolling window consistent.
    pub fn absorb_minutes(&mut self, minutes: &[ModeratorMinutes]) {
        for minute in minutes {
            if self.history.iter().any(|existing| {
                existing.plan_signature == minute.plan_signature
                    && existing.issued_at == minute.issued_at
            }) {
                continue;
            }
            self.history.push(minute.clone());
        }
        self.trim_history();
    }

    fn trim_history(&mut self) {
        if self.history.len() > self.history_limit {
            let excess = self.history.len() - self.history_limit;
            self.history.drain(0..excess);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_event(band: OutcomeBand, score: f32, psi: Option<f32>) -> DecisionEvent {
        DecisionEvent {
            plan_signature: "topk:4x8:8".to_string(),
            script_hint: "soft(topk)".to_string(),
            rank_kind: RankKind::TopK,
            winner: band,
            score,
            wilson_low: 0.2,
            wilson_high: 0.8,
            psi_total: psi,
            band_energy: (0.4, 0.3, 0.2),
            drift: 0.01,
            z_signal: 0.0,
            timestamp: SystemTime::now(),
            sequence: 0,
        }
    }

    #[test]
    fn summary_requires_events() {
        assert!(MetaSummary::from_events("node", &[]).is_none());
        let summary =
            MetaSummary::from_events("node", &[sample_event(OutcomeBand::Above, 0.6, Some(0.4))])
                .unwrap();
        assert_eq!(summary.node_id, "node");
        assert!(summary.mean_score > 0.0);
    }

    #[test]
    fn node_flushes_with_window() {
        let mut node = RoundtableNode::new(DistConfig {
            node_id: "n1".to_string(),
            mode: DistMode::PeriodicMeta,
            push_interval: Duration::from_secs(60),
            meta_endpoints: Vec::new(),
            summary_window: 2,
        });
        assert!(node
            .record_decision(
                "plan".to_string(),
                "hint".to_string(),
                RankKind::TopK,
                OutcomeBand::Here,
                0.5,
                Some(0.4),
                (0.2, 0.3, 0.1),
                0.01,
                0.0,
            )
            .is_none());
        assert!(node
            .record_decision(
                "plan".to_string(),
                "hint".to_string(),
                RankKind::TopK,
                OutcomeBand::Above,
                0.7,
                None,
                (0.3, 0.2, 0.1),
                0.02,
                0.4,
            )
            .is_some());
    }

    #[test]
    fn conductor_emits_proposal() {
        let mut conductor = MetaConductor::new(0.5, 2);
        let summary = MetaSummary {
            node_id: "a".into(),
            plan_signature: "plan".into(),
            script_hint: "soft(plan)".into(),
            rank_kind: RankKind::TopK,
            winner: OutcomeBand::Above,
            mean_score: 0.7,
            wilson_low: 0.4,
            wilson_high: 0.9,
            mean_psi: 0.3,
            mean_z: 0.1,
            support: 0.4,
            events: 4,
            issued_at: SystemTime::now(),
        };
        assert!(conductor.ingest(summary.clone()).is_none());
        let mut summary_b = summary.clone();
        summary_b.node_id = "b".into();
        summary_b.support = 0.6;
        let proposal = conductor.ingest(summary_b).unwrap();
        assert_eq!(proposal.ops.len(), 1);
        assert_eq!(proposal.evidence.len(), 2);
    }

    fn demo_runtime() -> BlackCatRuntime {
        let groups = ChoiceGroups {
            groups: HashMap::from([
                (
                    "agenda".to_string(),
                    vec!["focus".to_string(), "branch".to_string()],
                ),
                (
                    "pace".to_string(),
                    vec![
                        "steady".to_string(),
                        "accelerate".to_string(),
                        "reflect".to_string(),
                    ],
                ),
            ]),
        };
        BlackCatRuntime::new(ZMetaParams::default(), groups, 7, SoftBanditMode::TS, None)
    }

    #[test]
    fn moderator_tracks_minutes_and_proposals() {
        let mut moderator = BlackcatModerator::new(demo_runtime(), 0.5, 2);
        let summary = MetaSummary {
            node_id: "a".into(),
            plan_signature: "plan".into(),
            script_hint: "soft(plan)".into(),
            rank_kind: RankKind::TopK,
            winner: OutcomeBand::Above,
            mean_score: 0.7,
            wilson_low: 0.4,
            wilson_high: 0.8,
            mean_psi: 0.3,
            mean_z: 0.2,
            support: 0.3,
            events: 3,
            issued_at: SystemTime::now(),
        };
        let outcome_a = moderator.ingest(summary.clone());
        assert!(outcome_a.proposal.is_none());
        assert_eq!(moderator.minutes().len(), 1);
        let mut summary_b = summary;
        summary_b.node_id = "b".into();
        summary_b.support = 0.4;
        summary_b.mean_z = 0.25;
        let outcome_b = moderator.ingest(summary_b);
        assert!(outcome_b.proposal.is_some());
        assert_eq!(moderator.minutes().len(), 2);
    }

    #[test]
    fn moderator_absorbs_minutes() {
        let mut moderator = BlackcatModerator::with_default_runtime(0.5, 2);
        let base_time = SystemTime::now();
        let mut picks = HashMap::new();
        picks.insert("agenda".to_string(), "focus".to_string());
        let minute = ModeratorMinutes {
            plan_signature: "plan-a".into(),
            script_hint: "soft(plan-a)".into(),
            winner: OutcomeBand::Above,
            support: 0.6,
            mean_score: 0.7,
            mean_psi: 0.4,
            mean_z: 0.1,
            confidence: (0.45, 0.85),
            picks: picks.clone(),
            reward: 0.5,
            notes: "base".into(),
            issued_at: base_time,
        };
        moderator.absorb_minutes(&[minute.clone()]);
        assert_eq!(moderator.minutes().len(), 1);

        moderator.absorb_minutes(&[minute.clone()]);
        assert_eq!(moderator.minutes().len(), 1);

        let mut picks_b = picks;
        picks_b.insert("pace".to_string(), "steady".to_string());
        let later_minute = ModeratorMinutes {
            plan_signature: "plan-b".into(),
            script_hint: "soft(plan-b)".into(),
            winner: OutcomeBand::Here,
            support: 0.55,
            mean_score: 0.65,
            mean_psi: 0.35,
            mean_z: 0.05,
            confidence: (0.4, 0.8),
            picks: picks_b,
            reward: 0.25,
            notes: "later".into(),
            issued_at: base_time + Duration::from_secs(5),
        };
        moderator.absorb_minutes(&[later_minute.clone()]);
        assert_eq!(moderator.minutes().len(), 2);

        moderator.set_history_limit(1);
        assert_eq!(moderator.minutes().len(), 1);

        moderator.absorb_minutes(&[later_minute]);
        assert_eq!(moderator.minutes().len(), 1);
    }
}
