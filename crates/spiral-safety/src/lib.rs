//! Safety policy evaluation and auditing utilities for SpiralTorch surfaces.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use thiserror::Error;
use tracing::debug;

pub mod drift_response;
pub use drift_response::{
    aggregate_penalty, aggregate_penalty_with, analyse_word, analyse_word_with,
    analyse_word_with_options, default_thresholds, existence_load, frame_hazard, frame_summary,
    safe_radius, trainer_penalty, trainer_penalty_with, AnalysisOptions, DirectionQuery,
    DirectionalAxis, DirectionalSignature, DrlMetrics, DrsMetrics, FrameSignature, FrameState,
    FrameThreshold, WordState, DEFAULT_THRESHOLDS,
};

/// Content channel used when evaluating policy (prompt vs response).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, Ord, PartialOrd, Hash)]
pub enum ContentChannel {
    Prompt,
    Response,
}

impl fmt::Display for ContentChannel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContentChannel::Prompt => write!(f, "prompt"),
            ContentChannel::Response => write!(f, "response"),
        }
    }
}

/// Categories the policy can flag.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, Ord, PartialOrd, Hash)]
pub enum ViolationCategory {
    Toxicity,
    Bias,
    Safety,
}

impl fmt::Display for ViolationCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ViolationCategory::Toxicity => write!(f, "toxicity"),
            ViolationCategory::Bias => write!(f, "bias"),
            ViolationCategory::Safety => write!(f, "safety"),
        }
    }
}

/// Severity of a flagged item.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, Ord, PartialOrd, Hash)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

impl Default for RiskLevel {
    fn default() -> Self {
        RiskLevel::Low
    }
}

impl fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RiskLevel::Low => write!(f, "low"),
            RiskLevel::Medium => write!(f, "medium"),
            RiskLevel::High => write!(f, "high"),
            RiskLevel::Critical => write!(f, "critical"),
        }
    }
}

/// Detailed description of a policy violation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SafetyViolation {
    pub category: ViolationCategory,
    pub offending_term: String,
    pub message: String,
    pub risk: RiskLevel,
    pub score: f32,
}

impl SafetyViolation {
    pub fn new(category: ViolationCategory, offending_term: impl Into<String>, score: f32) -> Self {
        let term = offending_term.into();
        let message =
            format!("Detected {category} content triggered by '{term}' with score {score:.2}");
        let risk = if score >= 0.75 {
            RiskLevel::Critical
        } else if score >= 0.5 {
            RiskLevel::High
        } else if score >= 0.25 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        };
        Self {
            category,
            offending_term: term,
            message,
            risk,
            score,
        }
    }
}

/// Result of applying a policy filter to a piece of content.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SafetyVerdict {
    pub channel: ContentChannel,
    pub score: f32,
    pub violations: Vec<SafetyViolation>,
    pub allowed: bool,
}

impl SafetyVerdict {
    pub fn new(
        channel: ContentChannel,
        score: f32,
        violations: Vec<SafetyViolation>,
        allowed: bool,
    ) -> Self {
        Self {
            channel,
            score,
            violations,
            allowed,
        }
    }

    /// Whether the verdict should refuse serving content.
    pub fn should_refuse(&self) -> bool {
        !self.allowed
    }

    pub fn dominant_risk(&self) -> RiskLevel {
        self.violations
            .iter()
            .map(|v| v.risk)
            .max_by_key(|risk| match risk {
                RiskLevel::Low => 0,
                RiskLevel::Medium => 1,
                RiskLevel::High => 2,
                RiskLevel::Critical => 3,
            })
            .unwrap_or(RiskLevel::Low)
    }
}

/// Verdict for an individual conversational turn.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TurnVerdict {
    pub index: usize,
    pub verdict: SafetyVerdict,
}

/// Aggregated policy outcome for a multi-turn interaction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConversationVerdict {
    pub total_score: f32,
    pub highest_risk: RiskLevel,
    pub refused_turns: Vec<usize>,
    pub turn_verdicts: Vec<TurnVerdict>,
}

impl ConversationVerdict {
    /// Whether any turn in the conversation should be refused.
    pub fn should_refuse(&self) -> bool {
        !self.refused_turns.is_empty()
    }
}

/// Audit log event stored for compliance review.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditEvent {
    pub timestamp: DateTime<Utc>,
    pub channel: ContentChannel,
    pub content_preview: String,
    pub verdict: SafetyVerdict,
}

impl AuditEvent {
    pub fn new(
        channel: ContentChannel,
        content: impl Into<String>,
        verdict: SafetyVerdict,
    ) -> Self {
        let content = content.into();
        let preview = content.chars().take(240).collect::<String>();
        Self {
            timestamp: Utc::now(),
            channel,
            content_preview: preview,
            verdict,
        }
    }
}

/// Errors returned by the safety subsystem.
#[derive(Debug, Error)]
pub enum SafetyError {
    #[error("policy misconfiguration: {0}")]
    Misconfiguration(String),
}

/// Collects audit events and exposes them to callers.
#[derive(Clone, Default, Debug)]
pub struct AuditSink {
    inner: Arc<Mutex<Vec<AuditEvent>>>,
}

/// Summary statistics derived from an [`AuditSink`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditSummary {
    pub total_events: usize,
    pub refused_events: usize,
    pub highest_risk: RiskLevel,
    pub counts_by_category: BTreeMap<ViolationCategory, usize>,
}

impl Default for AuditSummary {
    fn default() -> Self {
        Self {
            total_events: 0,
            refused_events: 0,
            highest_risk: RiskLevel::Low,
            counts_by_category: BTreeMap::new(),
        }
    }
}

impl AuditSink {
    pub fn push(&self, event: AuditEvent) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.push(event);
        }
    }

    pub fn snapshot(&self) -> Vec<AuditEvent> {
        self.inner
            .lock()
            .map(|events| events.clone())
            .unwrap_or_default()
    }

    pub fn clear(&self) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.clear();
        }
    }

    /// Generate a summary of historical audit events.
    pub fn summarize(&self) -> AuditSummary {
        let events = self.snapshot();
        let mut summary = AuditSummary::default();
        summary.total_events = events.len();

        for event in &events {
            if event.verdict.should_refuse() {
                summary.refused_events += 1;
            }

            for violation in &event.verdict.violations {
                if violation.risk > summary.highest_risk {
                    summary.highest_risk = violation.risk;
                }
                *summary
                    .counts_by_category
                    .entry(violation.category)
                    .or_insert(0) += 1;
            }

            let dominant = event.verdict.dominant_risk();
            if dominant > summary.highest_risk {
                summary.highest_risk = dominant;
            }
        }

        summary
    }
}

/// Policy used to vet model inputs and outputs.
#[derive(Clone, Debug)]
pub struct SafetyPolicy {
    toxicity_keywords: Vec<&'static str>,
    bias_keywords: Vec<&'static str>,
    refusal_threshold: f32,
    audit_sink: AuditSink,
}

impl SafetyPolicy {
    pub fn new(
        toxicity_keywords: Vec<&'static str>,
        bias_keywords: Vec<&'static str>,
        refusal_threshold: f32,
        audit_sink: AuditSink,
    ) -> Self {
        Self {
            toxicity_keywords,
            bias_keywords,
            refusal_threshold,
            audit_sink,
        }
    }

    pub fn with_default_terms() -> Self {
        let toxicity = vec!["hate", "kill", "die", "stupid", "idiot"];
        let bias = vec!["race", "gender", "religion", "minority", "minorities"];
        Self::new(toxicity, bias, 0.5, AuditSink::default())
    }

    pub fn audit_sink(&self) -> AuditSink {
        self.audit_sink.clone()
    }

    pub fn with_audit_sink(mut self, sink: AuditSink) -> Self {
        self.audit_sink = sink;
        self
    }

    pub fn refusal_threshold(&self) -> f32 {
        self.refusal_threshold
    }

    pub fn with_refusal_threshold(mut self, threshold: f32) -> Self {
        self.refusal_threshold = threshold;
        self
    }

    fn scan(&self, content: &str, channel: ContentChannel) -> SafetyVerdict {
        let mut violations = Vec::new();
        let mut cumulative_score = 0.0;

        for term in &self.toxicity_keywords {
            if content.to_ascii_lowercase().contains(term) {
                let score = 0.6;
                cumulative_score += score;
                violations.push(SafetyViolation::new(
                    ViolationCategory::Toxicity,
                    *term,
                    score,
                ));
            }
        }

        for term in &self.bias_keywords {
            if content.to_ascii_lowercase().contains(term) {
                let score = 0.5;
                cumulative_score += score;
                violations.push(SafetyViolation::new(ViolationCategory::Bias, *term, score));
            }
        }

        let allowed = cumulative_score < self.refusal_threshold;
        SafetyVerdict::new(channel, cumulative_score, violations, allowed)
    }

    pub fn evaluate(&self, content: &str, channel: ContentChannel) -> SafetyVerdict {
        let verdict = self.scan(content, channel);
        debug!(channel = %channel, score = %verdict.score, violations = verdict.violations.len(), "evaluated content");
        self.audit_sink
            .push(AuditEvent::new(channel, content, verdict.clone()));
        verdict
    }

    pub fn refusal_reason<'a>(&self, verdict: &'a SafetyVerdict) -> Cow<'a, str> {
        if verdict.violations.is_empty() {
            return Cow::Borrowed("Policy refusal triggered without explicit violation");
        }
        let dominant = verdict
            .violations
            .iter()
            .map(|v| format!("{} ({})", v.category, v.risk))
            .collect::<Vec<_>>()
            .join(", ");
        Cow::Owned(format!("Blocked due to {dominant}"))
    }

    /// Evaluate multiple conversational turns and aggregate the outcomes.
    pub fn evaluate_conversation<'a, I, S>(&self, turns: I) -> ConversationVerdict
    where
        I: IntoIterator<Item = (ContentChannel, S)>,
        S: Into<Cow<'a, str>>,
    {
        let mut total_score = 0.0;
        let mut highest_risk = RiskLevel::Low;
        let mut refused_turns = Vec::new();
        let mut turn_verdicts = Vec::new();

        for (index, (channel, content)) in turns.into_iter().enumerate() {
            let content = content.into();
            let verdict = self.evaluate(content.as_ref(), channel);
            total_score += verdict.score;

            let dominant = verdict.dominant_risk();
            if dominant > highest_risk {
                highest_risk = dominant;
            }

            if verdict.should_refuse() {
                refused_turns.push(index);
            }

            turn_verdicts.push(TurnVerdict { index, verdict });
        }

        ConversationVerdict {
            total_score,
            highest_risk,
            refused_turns,
            turn_verdicts,
        }
    }
}

impl Default for SafetyPolicy {
    fn default() -> Self {
        Self::with_default_terms()
    }
}

/// Run an automated evaluation to ensure policy thresholds remain intact.
pub fn evaluate_policy_surface(
    policy: &SafetyPolicy,
    dataset: &[(&str, bool)],
) -> Result<(), SafetyError> {
    for (content, should_pass) in dataset {
        let verdict = policy.scan(content, ContentChannel::Prompt);
        if *should_pass && verdict.should_refuse() {
            return Err(SafetyError::Misconfiguration(format!(
                "content '{content}' expected to pass but was refused with score {:.2}",
                verdict.score
            )));
        }
        if !*should_pass && !verdict.should_refuse() {
            return Err(SafetyError::Misconfiguration(format!(
                "content '{content}' expected to be blocked but passed"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_toxicity() {
        let policy = SafetyPolicy::with_default_terms();
        let verdict = policy.evaluate("I will kill everyone", ContentChannel::Response);
        assert!(verdict.should_refuse());
        assert!(!verdict.violations.is_empty());
        assert!(verdict
            .violations
            .iter()
            .any(|v| matches!(v.category, ViolationCategory::Toxicity)));
    }

    #[test]
    fn allows_benign_content() {
        let policy = SafetyPolicy::with_default_terms();
        let verdict = policy.evaluate("Hello there", ContentChannel::Prompt);
        assert!(verdict.allowed);
    }

    #[test]
    fn evaluation_suite_catches_regressions() {
        let policy = SafetyPolicy::with_default_terms().with_refusal_threshold(2.0);
        let dataset = vec![
            ("Please help me plan a picnic", true),
            ("Why are minorities bad?", false),
        ];
        assert!(evaluate_policy_surface(&policy, &dataset).is_err());
    }

    #[test]
    fn conversation_evaluation_tracks_refusals() {
        let policy = SafetyPolicy::with_default_terms();
        let turns = vec![
            (ContentChannel::Prompt, "Hello there"),
            (ContentChannel::Response, "You are a stupid idiot"),
        ];

        let conversation = policy.evaluate_conversation(turns);

        assert!(conversation.should_refuse());
        assert_eq!(conversation.refused_turns, vec![1]);
        assert_eq!(conversation.turn_verdicts.len(), 2);
        assert!(conversation.total_score > 0.5);
        assert_eq!(conversation.highest_risk, RiskLevel::High);
    }

    #[test]
    fn audit_sink_summarizes_history() {
        let sink = AuditSink::default();
        let policy = SafetyPolicy::with_default_terms().with_audit_sink(sink.clone());

        policy.evaluate("Hello there", ContentChannel::Prompt);
        policy.evaluate("You are a stupid idiot", ContentChannel::Response);

        let summary = sink.summarize();

        assert_eq!(summary.total_events, 2);
        assert_eq!(summary.refused_events, 1);
        assert_eq!(summary.highest_risk, RiskLevel::High);
        assert_eq!(summary.counts_by_category.get(&ViolationCategory::Toxicity), Some(&2));
    }
}
