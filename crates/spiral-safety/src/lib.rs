//! Safety policy evaluation and auditing utilities for SpiralTorch surfaces.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
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
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
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

/// Keyword pattern that contributes to a policy verdict when present in the analysed content.
#[derive(Clone, Debug)]
pub struct PolicyTerm {
    category: ViolationCategory,
    keyword: String,
    needle: String,
    score: f32,
    channels: Option<Vec<ContentChannel>>,
}

impl PolicyTerm {
    /// Creates a new term with a case-insensitive keyword match.
    pub fn new(category: ViolationCategory, keyword: impl Into<String>, score: f32) -> Self {
        let keyword = keyword.into();
        let needle = keyword.to_ascii_lowercase();
        Self {
            category,
            keyword,
            needle,
            score,
            channels: None,
        }
    }

    /// Restrict a term to the provided content channels (prompt, response, or both).
    pub fn for_channels(mut self, channels: impl IntoIterator<Item = ContentChannel>) -> Self {
        let set: Vec<_> = channels.into_iter().collect();
        self.channels = if set.is_empty() { None } else { Some(set) };
        self
    }

    /// Whether the term applies to the given channel.
    pub fn applies_to(&self, channel: ContentChannel) -> bool {
        match &self.channels {
            Some(channels) => channels.contains(&channel),
            None => true,
        }
    }

    /// Category associated with the term.
    pub fn category(&self) -> ViolationCategory {
        self.category
    }

    /// Score contribution for each detected occurrence.
    pub fn score(&self) -> f32 {
        self.score
    }

    /// Offending keyword surfaced in audit logs.
    pub fn keyword(&self) -> &str {
        &self.keyword
    }

    fn occurrence_count(&self, haystack: &str) -> usize {
        haystack.match_indices(&self.needle).count()
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
}

/// Policy used to vet model inputs and outputs.
#[derive(Clone, Debug)]
pub struct SafetyPolicy {
    terms: Vec<PolicyTerm>,
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
        let mut terms = Vec::new();
        terms.extend(
            toxicity_keywords
                .into_iter()
                .map(|keyword| PolicyTerm::new(ViolationCategory::Toxicity, keyword, 0.6)),
        );
        terms.extend(
            bias_keywords
                .into_iter()
                .map(|keyword| PolicyTerm::new(ViolationCategory::Bias, keyword, 0.5)),
        );
        Self {
            terms,
            refusal_threshold,
            audit_sink,
        }
    }

    /// Creates a policy from the supplied keyword terms.
    pub fn from_terms(
        terms: Vec<PolicyTerm>,
        refusal_threshold: f32,
        audit_sink: AuditSink,
    ) -> Self {
        Self {
            terms,
            refusal_threshold,
            audit_sink,
        }
    }

    pub fn with_default_terms() -> Self {
        let terms = vec![
            PolicyTerm::new(ViolationCategory::Toxicity, "hate", 0.6),
            PolicyTerm::new(ViolationCategory::Toxicity, "kill", 0.7),
            PolicyTerm::new(ViolationCategory::Toxicity, "die", 0.55),
            PolicyTerm::new(ViolationCategory::Toxicity, "stupid", 0.4),
            PolicyTerm::new(ViolationCategory::Toxicity, "idiot", 0.45),
            PolicyTerm::new(ViolationCategory::Bias, "race", 0.5),
            PolicyTerm::new(ViolationCategory::Bias, "gender", 0.45),
            PolicyTerm::new(ViolationCategory::Bias, "religion", 0.5),
            PolicyTerm::new(ViolationCategory::Bias, "minority", 0.5),
            PolicyTerm::new(ViolationCategory::Bias, "minorities", 0.5),
            PolicyTerm::new(ViolationCategory::Safety, "bomb", 0.7)
                .for_channels([ContentChannel::Prompt, ContentChannel::Response]),
            PolicyTerm::new(ViolationCategory::Safety, "suicide", 0.8),
            PolicyTerm::new(ViolationCategory::Safety, "weapon", 0.65)
                .for_channels([ContentChannel::Prompt]),
            PolicyTerm::new(ViolationCategory::Safety, "explosive", 0.65)
                .for_channels([ContentChannel::Prompt]),
            PolicyTerm::new(ViolationCategory::Safety, "self-harm", 0.75),
        ];

        Self::from_terms(terms, 0.5, AuditSink::default())
    }

    pub fn audit_sink(&self) -> AuditSink {
        self.audit_sink.clone()
    }

    pub fn with_audit_sink(mut self, sink: AuditSink) -> Self {
        self.audit_sink = sink;
        self
    }

    /// Removes all accumulated audit events.
    pub fn clear_audit_log(&self) {
        self.audit_sink.clear();
    }

    /// Return the configured policy terms.
    pub fn terms(&self) -> &[PolicyTerm] {
        &self.terms
    }

    /// Adds a policy term and returns an updated policy.
    pub fn with_term(mut self, term: PolicyTerm) -> Self {
        self.terms.push(term);
        self
    }

    /// Extends the set of policy terms in place.
    pub fn extend_terms<I>(&mut self, terms: I)
    where
        I: IntoIterator<Item = PolicyTerm>,
    {
        self.terms.extend(terms);
    }

    /// Adds a policy term without consuming the policy.
    pub fn push_term(&mut self, term: PolicyTerm) {
        self.terms.push(term);
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
        let lowercase = content.to_ascii_lowercase();

        for term in &self.terms {
            if !term.applies_to(channel) {
                continue;
            }

            let matches = term.occurrence_count(&lowercase);
            if matches == 0 {
                continue;
            }

            for _ in 0..matches {
                let score = term.score();
                cumulative_score += score;
                violations.push(SafetyViolation::new(term.category(), term.keyword(), score));
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
    fn accumulates_multiple_occurrences() {
        let policy = SafetyPolicy::with_default_terms().with_refusal_threshold(1.1);
        let verdict = policy.evaluate("We should kill and kill again", ContentChannel::Prompt);
        assert!(verdict.should_refuse());
        assert!(verdict.score >= 1.4);
        assert!(
            verdict
                .violations
                .iter()
                .filter(|v| v.offending_term == "kill")
                .count()
                >= 2
        );

        policy.clear_audit_log();
        assert!(policy.audit_sink().snapshot().is_empty());
    }

    #[test]
    fn channel_specific_terms_only_apply_when_expected() {
        let policy = SafetyPolicy::from_terms(
            vec![PolicyTerm::new(ViolationCategory::Safety, "bomb", 0.7)
                .for_channels([ContentChannel::Prompt])],
            0.5,
            AuditSink::default(),
        );

        let prompt_verdict = policy.evaluate("How to build a bomb", ContentChannel::Prompt);
        assert!(prompt_verdict.should_refuse());

        let response_verdict = policy.evaluate("I will build a bomb", ContentChannel::Response);
        assert!(response_verdict.allowed);

        assert_eq!(policy.audit_sink().snapshot().len(), 2);
    }
}
