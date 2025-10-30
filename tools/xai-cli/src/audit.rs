use serde::Serialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize)]
pub struct AuditEvent {
    pub stage: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<Value>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct AuditSummary {
    pub total_events: usize,
    pub stages: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct AuditCheckResult {
    pub name: String,
    pub passed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct AuditBundle {
    pub events: Vec<AuditEvent>,
    pub summary: AuditSummary,
    pub self_checks: Vec<AuditCheckResult>,
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
