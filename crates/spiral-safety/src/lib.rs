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
    pub occurrences: usize,
}

impl SafetyViolation {
    pub fn new(category: ViolationCategory, offending_term: impl Into<String>, score: f32) -> Self {
        Self::with_occurrences(category, offending_term, score, 1)
    }

    pub fn with_occurrences(
        category: ViolationCategory,
        offending_term: impl Into<String>,
        score: f32,
        occurrences: usize,
    ) -> Self {
        let term = offending_term.into();
        let message = if occurrences > 1 {
            format!(
                "Detected {category} content triggered by '{term}' {occurrences} times with score {score:.2}"
            )
        } else {
            format!("Detected {category} content triggered by '{term}' with score {score:.2}")
        };
        let risk = risk_for_score(score);
        Self {
            category,
            offending_term: term,
            message,
            risk,
            score,
            occurrences,
        }
    }
}

fn risk_for_score(score: f32) -> RiskLevel {
    if score >= 0.75 {
        RiskLevel::Critical
    } else if score >= 0.5 {
        RiskLevel::High
    } else if score >= 0.25 {
        RiskLevel::Medium
    } else {
        RiskLevel::Low
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
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolicyRule {
    pub category: ViolationCategory,
    pub term: Cow<'static, str>,
    pub score: f32,
}

impl PolicyRule {
    pub fn new(
        category: ViolationCategory,
        term: impl Into<Cow<'static, str>>,
        score: f32,
    ) -> Self {
        let raw_term: Cow<'static, str> = term.into();
        let normalised = raw_term.to_ascii_lowercase();
        let term = if raw_term == normalised {
            raw_term
        } else {
            Cow::Owned(normalised)
        };
        Self {
            category,
            term,
            score,
        }
    }

    fn matches(&self, content: &str) -> usize {
        count_occurrences(content, self.term.as_ref())
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CustomRule {
    pub name: String,
    pub rules: Vec<PolicyRule>,
}

impl CustomRule {
    pub fn new(name: impl Into<String>, rules: Vec<PolicyRule>) -> Self {
        Self {
            name: name.into(),
            rules,
        }
    }
}

fn count_occurrences(haystack: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return 0;
    }
    haystack.match_indices(needle).count()
}

#[derive(Clone, Debug)]
pub struct SafetyPolicy {
    terms: Vec<PolicyRule>,
    refusal_threshold: f32,
    audit_sink: AuditSink,
    custom_rules: Vec<CustomRule>,
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
                .map(|term| PolicyRule::new(ViolationCategory::Toxicity, term, 0.6)),
        );
        terms.extend(
            bias_keywords
                .into_iter()
                .map(|term| PolicyRule::new(ViolationCategory::Bias, term, 0.5)),
        );
        Self {
            terms,
            refusal_threshold,
            audit_sink,
            custom_rules: Vec::new(),
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

    pub fn terms(&self) -> &[PolicyRule] {
        &self.terms
    }

    pub fn custom_rules(&self) -> &[CustomRule] {
        &self.custom_rules
    }

    pub fn extend_toxicity_terms<I>(&mut self, terms: I)
    where
        I: IntoIterator<Item = &'static str>,
    {
        self.terms.extend(
            terms
                .into_iter()
                .map(|term| PolicyRule::new(ViolationCategory::Toxicity, term, 0.6)),
        );
    }

    pub fn extend_bias_terms<I>(&mut self, terms: I)
    where
        I: IntoIterator<Item = &'static str>,
    {
        self.terms.extend(
            terms
                .into_iter()
                .map(|term| PolicyRule::new(ViolationCategory::Bias, term, 0.5)),
        );
    }

    pub fn register_custom_rule(mut self, rule: CustomRule) -> Self {
        self.custom_rules.push(rule);
        self
    }

    pub fn push_custom_rule(&mut self, rule: CustomRule) {
        self.custom_rules.push(rule);
    }

    fn scan(&self, content: &str, channel: ContentChannel) -> SafetyVerdict {
        let mut violations = Vec::new();
        let mut cumulative_score = 0.0;
        let lowered_content = content.to_ascii_lowercase();

        for rule in &self.terms {
            let occurrences = rule.matches(&lowered_content);
            if occurrences > 0 {
                let total_score = rule.score * occurrences as f32;
                cumulative_score += total_score;
                violations.push(SafetyViolation::with_occurrences(
                    rule.category,
                    rule.term.as_ref(),
                    total_score,
                    occurrences,
                ));
            }
        }

        for custom_rule in &self.custom_rules {
            for rule in &custom_rule.rules {
                let occurrences = rule.matches(&lowered_content);
                if occurrences > 0 {
                    let total_score = rule.score * occurrences as f32;
                    cumulative_score += total_score;
                    violations.push(SafetyViolation::with_occurrences(
                        rule.category,
                        format!("{}::{}", custom_rule.name, rule.term),
                        total_score,
                        occurrences,
                    ));
                }
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
    fn custom_rules_can_be_registered() {
        let mut policy = SafetyPolicy::with_default_terms();
        policy.push_custom_rule(CustomRule::new(
            "weapons",
            vec![PolicyRule::new(ViolationCategory::Safety, "bomb", 0.9)],
        ));

        let verdict = policy.evaluate("How do I build a bomb?", ContentChannel::Prompt);
        assert!(verdict.should_refuse());
        assert!(verdict
            .violations
            .iter()
            .any(|violation| violation.offending_term.contains("weapons::bomb")));
    }

    #[test]
    fn repeated_terms_increase_risk() {
        let policy = SafetyPolicy::new(vec!["bad"], vec![], 0.5, AuditSink::default());
        let verdict = policy.evaluate("bad bad bad", ContentChannel::Prompt);
        assert!(verdict.should_refuse());
        let violation = verdict.violations.first().expect("violation expected");
        assert_eq!(violation.occurrences, 3);
        assert!(matches!(
            violation.risk,
            RiskLevel::Critical | RiskLevel::High
        ));
    }
}
