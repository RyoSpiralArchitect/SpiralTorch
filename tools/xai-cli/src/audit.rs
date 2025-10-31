use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditEvent {
    pub stage: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<Value>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuditSummary {
    pub total_events: usize,
    pub stages: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuditCheckResult {
    pub name: String,
    pub passed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AuditBundle {
    pub events: Vec<AuditEvent>,
    pub summary: AuditSummary,
    pub self_checks: Vec<AuditCheckResult>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StageDifference {
    pub stage: String,
    pub recorded: usize,
    pub recomputed: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditCheckComparison {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recorded_passed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recorded_message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recomputed_passed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recomputed_message: Option<String>,
    pub matches: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditReviewReport {
    pub observed_events: usize,
    pub summary_matches: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stage_differences: Vec<StageDifference>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub issues: Vec<String>,
    #[serde(default)]
    pub check_comparisons: Vec<AuditCheckComparison>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum AuditAnomalySeverity {
    Info,
    Warning,
    Critical,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuditAnomaly {
    pub severity: AuditAnomalySeverity,
    pub message: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct StageTransitionMetric {
    pub from: Option<String>,
    pub to: String,
    pub count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AuditIntrospection {
    pub total_events: usize,
    pub unique_stages: usize,
    pub entropy: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub loops: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transitions: Vec<StageTransitionMetric>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub anomalies: Vec<AuditAnomaly>,
}

#[derive(Clone, Debug, Default)]
pub struct AuditTrail {
    events: Vec<AuditEvent>,
}

impl AuditTrail {
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    pub fn record<S>(&mut self, stage: S)
    where
        S: Into<String>,
    {
        self.record_with_detail(stage, Option::<Value>::None);
    }

    pub fn record_with_value<S, D>(&mut self, stage: S, detail: D)
    where
        S: Into<String>,
        D: Serialize,
    {
        let value = serde_json::to_value(detail).unwrap_or_else(|_| json!("unserialisable"));
        self.record_with_detail(stage, Some(value));
    }

    pub fn record_with_detail<S>(&mut self, stage: S, detail: Option<Value>)
    where
        S: Into<String>,
    {
        self.events.push(AuditEvent {
            stage: stage.into(),
            detail,
        });
    }

    pub fn review(&self) -> AuditBundle {
        let summary = build_summary(&self.events);
        let self_checks = build_self_checks(&self.events, &summary);
        AuditBundle {
            events: self.events.clone(),
            summary,
            self_checks,
        }
    }

    pub fn finish(self) -> AuditBundle {
        let summary = build_summary(&self.events);
        let self_checks = build_self_checks(&self.events, &summary);
        AuditBundle {
            events: self.events,
            summary,
            self_checks,
        }
    }
}

fn build_summary(events: &[AuditEvent]) -> AuditSummary {
    let mut summary = AuditSummary::default();
    summary.total_events = events.len();
    for event in events {
        *summary.stages.entry(event.stage.clone()).or_default() += 1;
    }
    summary
}

fn build_self_checks(events: &[AuditEvent], summary: &AuditSummary) -> Vec<AuditCheckResult> {
    let mut checks = Vec::new();

    checks.push(require_stage(events, "cli.parsed", "cli_parsed"));
    checks.push(require_stage(events, "io.write.report", "report_scheduled"));

    if events
        .iter()
        .any(|event| event.stage == "overlay.requested")
    {
        checks.push(require_stage(
            events,
            "io.read.overlay_base",
            "overlay_base_loaded",
        ));
    }

    if events
        .iter()
        .any(|event| event.stage == "focus_mask.requested")
    {
        checks.push(require_stage(
            events,
            "focus_mask.generated",
            "focus_mask_generated",
        ));
    }

    if events
        .iter()
        .any(|event| event.stage == "metadata.audit_summary_embedded")
    {
        checks.push(require_stage(
            events,
            "metadata.audit_summary_embedded",
            "audit_summary_embedded",
        ));
    }

    checks.push(AuditCheckResult {
        name: "events_recorded".to_string(),
        passed: summary.total_events > 0,
        message: if summary.total_events > 0 {
            None
        } else {
            Some("no audit events were captured".to_string())
        },
    });

    checks
}

fn require_stage(events: &[AuditEvent], stage: &str, name: &str) -> AuditCheckResult {
    let passed = events.iter().any(|event| event.stage == stage);
    AuditCheckResult {
        name: name.to_string(),
        passed,
        message: if passed {
            None
        } else {
            Some(format!("missing required stage `{stage}`"))
        },
    }
}

pub fn review_bundle(bundle: &AuditBundle) -> AuditReviewReport {
    let recomputed_summary = build_summary(&bundle.events);
    let summary_matches = recomputed_summary == bundle.summary;

    let mut stage_differences = Vec::new();
    let mut issues = Vec::new();

    if recomputed_summary.total_events != bundle.summary.total_events {
        issues.push(format!(
            "summary.total_events recorded {} but recomputed {}",
            bundle.summary.total_events, recomputed_summary.total_events
        ));
    }

    let mut stage_names: BTreeSet<String> = bundle.summary.stages.keys().cloned().collect();
    stage_names.extend(recomputed_summary.stages.keys().cloned());

    for stage in stage_names {
        let recorded = *bundle.summary.stages.get(&stage).unwrap_or(&0);
        let recomputed = *recomputed_summary.stages.get(&stage).unwrap_or(&0);
        if recorded != recomputed {
            stage_differences.push(StageDifference {
                stage: stage.clone(),
                recorded,
                recomputed,
            });
            issues.push(format!(
                "summary stage `{stage}` recorded {recorded} but recomputed {recomputed}"
            ));
        }
    }

    let recomputed_checks = build_self_checks(&bundle.events, &recomputed_summary);
    let mut recomputed_map = BTreeMap::new();
    for check in recomputed_checks {
        recomputed_map.insert(check.name.clone(), check);
    }

    let mut recorded_map = BTreeMap::new();
    for check in &bundle.self_checks {
        recorded_map.insert(check.name.clone(), check.clone());
    }

    let mut check_comparisons = Vec::new();
    let mut names = BTreeSet::new();
    names.extend(recorded_map.keys().cloned());
    names.extend(recomputed_map.keys().cloned());

    for name in names {
        let recorded = recorded_map.get(&name);
        let recomputed = recomputed_map.get(&name);

        let matches = match (recorded, recomputed) {
            (Some(recorded), Some(recomputed)) => {
                recorded.passed == recomputed.passed && recorded.message == recomputed.message
            }
            (None, None) => true,
            _ => false,
        };

        if !matches {
            issues.push(format!("self-check `{name}` did not match recorded state"));
        }

        check_comparisons.push(AuditCheckComparison {
            name: name.clone(),
            recorded_passed: recorded.map(|check| check.passed),
            recorded_message: recorded.and_then(|check| check.message.clone()),
            recomputed_passed: recomputed.map(|check| check.passed),
            recomputed_message: recomputed.and_then(|check| check.message.clone()),
            matches,
        });
    }

    AuditReviewReport {
        observed_events: bundle.events.len(),
        summary_matches,
        stage_differences,
        issues,
        check_comparisons,
    }
}

pub fn introspect_bundle(bundle: &AuditBundle) -> AuditIntrospection {
    let mut unique_stages = BTreeSet::new();
    let mut loops = BTreeSet::new();
    let mut transitions: BTreeMap<(Option<String>, String), usize> = BTreeMap::new();
    let mut anomalies = Vec::new();

    let critical_stages = ["cli.parsed", "cli.command", "io.write.report"];
    for stage in &critical_stages {
        if !bundle.summary.stages.contains_key(*stage) {
            anomalies.push(AuditAnomaly {
                severity: AuditAnomalySeverity::Critical,
                message: format!("missing critical stage `{stage}`"),
            });
        }
    }

    let known_prefixes = [
        "cli.",
        "io.",
        "metadata.",
        "statistics.",
        "overlay.",
        "focus_mask.",
        "postprocess.",
        "grad_cam.",
        "integrated_gradients.",
        "model.",
        "finalise.",
        "audit.",
    ];

    let mut previous: Option<&AuditEvent> = None;
    for event in &bundle.events {
        unique_stages.insert(event.stage.clone());
        if !known_prefixes
            .iter()
            .any(|prefix| event.stage.starts_with(prefix))
        {
            anomalies.push(AuditAnomaly {
                severity: AuditAnomalySeverity::Warning,
                message: format!("unknown stage prefix `{}`", event.stage),
            });
        }

        if let Some(prev) = previous {
            if prev.stage == event.stage {
                loops.insert(event.stage.clone());
            }
            *transitions
                .entry((Some(prev.stage.clone()), event.stage.clone()))
                .or_default() += 1;
        } else {
            *transitions.entry((None, event.stage.clone())).or_default() += 1;
        }
        previous = Some(event);
    }

    let total_events = bundle.events.len();
    let mut entropy = 0.0;
    if total_events > 0 {
        for count in bundle.summary.stages.values() {
            let probability = *count as f64 / total_events as f64;
            entropy -= probability * probability.log2();
        }
    }

    let transitions = transitions
        .into_iter()
        .map(|((from, to), count)| StageTransitionMetric { from, to, count })
        .collect();

    AuditIntrospection {
        total_events,
        unique_stages: unique_stages.len(),
        entropy,
        loops: loops.into_iter().collect(),
        transitions,
        anomalies,
    }
}

pub fn introspect_bundles<'a, I>(bundles: I) -> AuditIntrospection
where
    I: IntoIterator<Item = &'a AuditBundle>,
{
    let mut events = Vec::new();

    for bundle in bundles {
        events.extend(bundle.events.clone());
    }

    let summary = build_summary(&events);
    let self_checks = build_self_checks(&events, &summary);
    let combined = AuditBundle {
        events,
        summary,
        self_checks,
    };

    introspect_bundle(&combined)
}
