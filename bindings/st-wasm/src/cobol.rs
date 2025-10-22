use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use time::{format_description::well_known::Rfc3339, OffsetDateTime};

const DEFAULT_JOB_PLACEHOLDER: &str = "job";
const PLANNER_INITIALIZED_ANNOTATION: &str = "planner_initialized";
const NARRATOR_METRIC_MIN: f32 = 0.0;
const NARRATOR_METRIC_MAX: f32 = 1.0;
const FALLBACK_TIMESTAMP: &str = "1970-01-01T00:00:00Z";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CobolEnvelope {
    pub job_id: String,
    pub release_channel: String,
    pub created_at: String,
    pub initiators: Vec<InteractionInitiator>,
    pub route: CobolRoute,
    pub payload: CobolNarratorPayload,
    pub metadata: CobolMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct CobolRoute {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mq: Option<CobolMqRoute>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cics: Option<CobolCicsRoute>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CobolMqRoute {
    pub manager: String,
    pub queue: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CobolCicsRoute {
    pub transaction: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub program: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channel: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CobolNarratorPayload {
    pub curvature: f32,
    pub temperature: f32,
    pub encoder: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub locale: Option<String>,
    #[serde(default)]
    pub coefficients: Vec<f32>,
}

impl Default for CobolNarratorPayload {
    fn default() -> Self {
        Self {
            curvature: 0.5,
            temperature: 0.5,
            encoder: "spiraltorch.default".to_string(),
            locale: None,
            coefficients: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CobolMetadata {
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub annotations: Vec<String>,
    #[serde(default = "default_metadata_extra")]
    pub extra: Value,
}

fn default_metadata_extra() -> Value {
    Value::Object(Map::new())
}

impl Default for CobolMetadata {
    fn default() -> Self {
        Self {
            tags: Vec::new(),
            annotations: Vec::new(),
            extra: default_metadata_extra(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InteractionInitiator {
    #[serde(rename = "type")]
    pub kind: InitiatorKind,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub persona: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contact: Option<String>,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum InitiatorKind {
    Human,
    Model,
    Automation,
}

pub struct CobolEnvelopeBuilder {
    envelope: CobolEnvelope,
}

impl CobolEnvelopeBuilder {
    pub fn new(job_id: impl Into<String>) -> Self {
        let mut builder = Self {
            envelope: CobolEnvelope {
                job_id: job_id.into(),
                release_channel: "production".to_string(),
                created_at: now_timestamp(),
                initiators: Vec::new(),
                route: CobolRoute::default(),
                payload: CobolNarratorPayload::default(),
                metadata: CobolMetadata::default(),
            },
        };
        sanitize_envelope(&mut builder.envelope);
        ensure_planner_annotation(&mut builder.envelope);
        builder
    }

    pub fn from_envelope(mut envelope: CobolEnvelope) -> Self {
        sanitize_envelope(&mut envelope);
        ensure_planner_annotation(&mut envelope);
        Self { envelope }
    }

    pub fn load_envelope(&mut self, mut envelope: CobolEnvelope) {
        sanitize_envelope(&mut envelope);
        ensure_planner_annotation(&mut envelope);
        self.envelope = envelope;
    }

    pub fn set_release_channel(&mut self, channel: impl Into<String>) {
        if let Some(clean) = sanitize(channel.into()) {
            self.envelope.release_channel = clean;
        }
    }

    pub fn set_created_at(&mut self, created_at: impl Into<String>) {
        if let Some(clean) = sanitize(created_at.into()) {
            self.envelope.created_at = clean;
        }
    }

    pub fn reset_created_at(&mut self) {
        self.envelope.created_at = now_timestamp();
    }

    pub fn set_narrator_config(
        &mut self,
        curvature: f32,
        temperature: f32,
        encoder: impl Into<String>,
        locale: Option<String>,
    ) {
        self.envelope.payload.curvature = curvature;
        self.envelope.payload.temperature = temperature;
        if let Some(clean) = sanitize(encoder.into()) {
            self.envelope.payload.encoder = clean;
        }
        self.envelope.payload.locale = locale.and_then(sanitize);
    }

    pub fn set_coefficients(&mut self, coefficients: Vec<f32>) {
        self.envelope.payload.coefficients = coefficients;
    }

    pub fn add_initiator(&mut self, initiator: InteractionInitiator) {
        self.envelope.initiators.push(initiator);
    }

    pub fn clear_initiators(&mut self) {
        self.envelope.initiators.clear();
    }

    pub fn set_mq_route(
        &mut self,
        manager: impl Into<String>,
        queue: impl Into<String>,
        commit: Option<String>,
    ) {
        self.envelope.route.mq = Some(CobolMqRoute {
            manager: manager.into(),
            queue: queue.into(),
            commit: commit.and_then(sanitize),
        });
    }

    pub fn clear_mq_route(&mut self) {
        self.envelope.route.mq = None;
    }

    pub fn set_cics_route(
        &mut self,
        transaction: impl Into<String>,
        program: Option<String>,
        channel: Option<String>,
    ) {
        self.envelope.route.cics = Some(CobolCicsRoute {
            transaction: transaction.into(),
            program: program.and_then(sanitize),
            channel: channel.and_then(sanitize),
        });
    }

    pub fn clear_cics_route(&mut self) {
        self.envelope.route.cics = None;
    }

    pub fn set_dataset(&mut self, dataset: Option<String>) {
        self.envelope.route.dataset = dataset.and_then(sanitize);
    }

    pub fn clear_route(&mut self) {
        self.clear_mq_route();
        self.clear_cics_route();
        self.envelope.route.dataset = None;
    }

    pub fn add_tag(&mut self, tag: impl Into<String>) {
        if let Some(clean) = sanitize(tag.into()) {
            self.envelope.metadata.tags.push(clean);
        }
    }

    pub fn add_annotation(&mut self, annotation: impl Into<String>) {
        if let Some(clean) = sanitize(annotation.into()) {
            self.envelope.metadata.annotations.push(clean);
        }
    }

    pub fn merge_metadata_value(&mut self, value: Value) {
        match value {
            Value::Null => {}
            Value::Object(mut incoming) => match &mut self.envelope.metadata.extra {
                Value::Object(existing) => {
                    existing.append(&mut incoming);
                }
                other => {
                    *other = Value::Object(incoming);
                }
            },
            other => {
                self.envelope.metadata.extra = other;
            }
        }
    }

    pub fn clear_metadata(&mut self) {
        self.envelope.metadata.extra = default_metadata_extra();
    }

    pub fn is_valid(&self) -> bool {
        self.envelope.is_valid()
    }

    pub fn validation_issues(&self) -> Vec<String> {
        self.envelope.validation_issues()
    }

    pub fn envelope(&self) -> &CobolEnvelope {
        &self.envelope
    }

    pub fn snapshot(&self) -> CobolEnvelope {
        self.envelope.clone()
    }

    pub fn into_envelope(self) -> CobolEnvelope {
        self.envelope
    }
}

impl CobolEnvelope {
    pub fn to_json_string(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    pub fn to_json_bytes(&self) -> serde_json::Result<Vec<u8>> {
        serde_json::to_vec(self)
    }

    pub fn from_json_str(input: &str) -> serde_json::Result<Self> {
        serde_json::from_str(input)
    }

    pub fn from_json_slice(input: &[u8]) -> serde_json::Result<Self> {
        serde_json::from_slice(input)
    }

    pub fn validation_issues(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if self.job_id == DEFAULT_JOB_PLACEHOLDER {
            issues.push("replace the default job identifier before dispatch".to_string());
        }

        if self.initiators.is_empty() {
            issues.push("add at least one initiator before dispatch".to_string());
        }

        if self.route.mq.is_none() && self.route.cics.is_none() && self.route.dataset.is_none() {
            issues.push("configure a delivery route (MQ, CICS, or dataset)".to_string());
        }

        if !(NARRATOR_METRIC_MIN..=NARRATOR_METRIC_MAX).contains(&self.payload.curvature) {
            issues.push("curvature must be between 0.0 and 1.0".to_string());
        }

        if !(NARRATOR_METRIC_MIN..=NARRATOR_METRIC_MAX).contains(&self.payload.temperature) {
            issues.push("temperature must be between 0.0 and 1.0".to_string());
        }

        issues
    }

    pub fn is_valid(&self) -> bool {
        self.validation_issues().is_empty()
    }
}

fn now_timestamp() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| FALLBACK_TIMESTAMP.to_string())
}

fn sanitize(value: String) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn sanitize_envelope(envelope: &mut CobolEnvelope) {
    sanitize_required(&mut envelope.job_id, || DEFAULT_JOB_PLACEHOLDER.to_string());
    sanitize_required(&mut envelope.release_channel, || "production".to_string());
    sanitize_required(&mut envelope.created_at, now_timestamp);

    sanitize_required(&mut envelope.payload.encoder, || {
        "spiraltorch.default".to_string()
    });
    sanitize_option(&mut envelope.payload.locale);

    if let Some(route) = &mut envelope.route.mq {
        let manager = sanitize(std::mem::take(&mut route.manager));
        let queue = sanitize(std::mem::take(&mut route.queue));
        route.commit = route.commit.take().and_then(sanitize);
        match (manager, queue) {
            (Some(manager), Some(queue)) => {
                route.manager = manager;
                route.queue = queue;
            }
            _ => {
                envelope.route.mq = None;
            }
        }
    }

    if let Some(route) = &mut envelope.route.cics {
        let transaction = sanitize(std::mem::take(&mut route.transaction));
        route.program = route.program.take().and_then(sanitize);
        route.channel = route.channel.take().and_then(sanitize);
        match transaction {
            Some(transaction) => {
                route.transaction = transaction;
            }
            None => {
                envelope.route.cics = None;
            }
        }
    }

    envelope.route.dataset = envelope.route.dataset.take().and_then(sanitize);

    sanitize_vec(&mut envelope.metadata.tags);
    sanitize_vec(&mut envelope.metadata.annotations);

    for initiator in &mut envelope.initiators {
        sanitize_required(&mut initiator.name, || "participant".to_string());
        sanitize_option(&mut initiator.persona);
        sanitize_option(&mut initiator.revision);
        sanitize_option(&mut initiator.contact);
        sanitize_notes(&mut initiator.notes);
    }
}

fn ensure_planner_annotation(envelope: &mut CobolEnvelope) {
    sanitize_vec(&mut envelope.metadata.annotations);
    if !envelope
        .metadata
        .annotations
        .iter()
        .any(|annotation| annotation == PLANNER_INITIALIZED_ANNOTATION)
    {
        envelope
            .metadata
            .annotations
            .push(PLANNER_INITIALIZED_ANNOTATION.to_string());
    }
}

fn sanitize_required<F>(target: &mut String, default: F)
where
    F: FnOnce() -> String,
{
    let current = std::mem::take(target);
    match sanitize(current) {
        Some(value) => *target = value,
        None => *target = default(),
    }
}

fn sanitize_option(target: &mut Option<String>) {
    if let Some(value) = target.take() {
        *target = sanitize(value);
    }
}

fn sanitize_vec(values: &mut Vec<String>) {
    let mut sanitized = Vec::with_capacity(values.len());
    for value in values.drain(..) {
        if let Some(clean) = sanitize(value) {
            sanitized.push(clean);
        }
    }
    *values = sanitized;
}

fn sanitize_notes(notes: &mut Vec<String>) {
    sanitize_vec(notes);
}

pub fn make_initiator(
    kind: InitiatorKind,
    name: impl Into<String>,
    persona: Option<String>,
    revision: Option<String>,
    contact: Option<String>,
    note: Option<String>,
) -> InteractionInitiator {
    let mut initiator = InteractionInitiator {
        kind,
        name: name.into(),
        persona: persona.and_then(sanitize),
        revision: revision.and_then(sanitize),
        contact: contact.and_then(sanitize),
        notes: Vec::new(),
    };
    if let Some(note) = note.and_then(sanitize) {
        initiator.notes.push(note);
    }
    initiator
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_applies_updates() {
        let mut builder = CobolEnvelopeBuilder::new("job-42");
        builder.set_release_channel("shadow");
        builder.set_narrator_config(0.9, 0.2, "custom.encoder", Some("en-US".into()));
        builder.set_coefficients(vec![0.1, 0.2]);
        builder.add_initiator(make_initiator(
            InitiatorKind::Human,
            "Operator",
            Some("guide".into()),
            None,
            Some("ops@example".into()),
            Some("primary".into()),
        ));
        builder.set_mq_route("QM1", "Q1", Some("sync".into()));
        builder.set_cics_route("CX12", Some("PGM".into()), None);
        builder.set_dataset(Some("HLQ.DATA".into()));
        builder.add_tag("browser");
        builder.add_annotation("generated");
        builder.merge_metadata_value(serde_json::json!({"priority": "low"}));

        let envelope = builder.snapshot();
        assert_eq!(envelope.job_id, "job-42");
        assert_eq!(envelope.release_channel, "shadow");
        assert_eq!(envelope.payload.encoder, "custom.encoder");
        assert_eq!(envelope.payload.locale.as_deref(), Some("en-US"));
        assert_eq!(envelope.payload.coefficients, vec![0.1, 0.2]);
        assert_eq!(envelope.initiators.len(), 1);
        assert!(envelope.route.mq.is_some());
        assert!(envelope.route.cics.is_some());
        assert_eq!(envelope.route.dataset.as_deref(), Some("HLQ.DATA"));
        assert!(envelope.metadata.tags.contains(&"browser".to_string()));
        assert!(envelope
            .metadata
            .annotations
            .iter()
            .any(|a| a == "generated"));
        assert_eq!(
            envelope.metadata.extra.get("priority"),
            Some(&serde_json::json!("low"))
        );
    }

    #[test]
    fn sanitize_rejects_empty_values() {
        let mut builder = CobolEnvelopeBuilder::new("job-100");
        builder.set_release_channel("   ");
        builder.set_dataset(Some("   ".into()));
        builder.add_tag("  ");
        builder.add_annotation(" ");
        builder.merge_metadata_value(Value::Null);
        assert_eq!(builder.snapshot().release_channel, "production");
        assert!(builder.snapshot().route.dataset.is_none());
        assert!(builder.snapshot().metadata.tags.is_empty());
    }

    #[test]
    fn clearing_state_resets_routes_and_initiators() {
        let mut builder = CobolEnvelopeBuilder::new("job-303");
        builder.add_initiator(make_initiator(
            InitiatorKind::Automation,
            "bot",
            None,
            None,
            None,
            None,
        ));
        builder.set_mq_route("QM2", "QUEUE", None);
        builder.set_cics_route("TRN1", Some("PGM1".into()), Some("CHAN".into()));
        builder.set_dataset(Some("HLQ.DATA".into()));

        builder.clear_initiators();
        builder.clear_mq_route();
        builder.clear_cics_route();
        builder.set_dataset(None);

        let envelope = builder.snapshot();
        assert!(envelope.initiators.is_empty());
        assert!(envelope.route.mq.is_none());
        assert!(envelope.route.cics.is_none());
        assert!(envelope.route.dataset.is_none());

        builder.set_mq_route("QM2", "QUEUE", None);
        builder.set_cics_route("TRN1", None, None);
        builder.set_dataset(Some("HLQ.DATA".into()));
        builder.clear_route();
        let cleared = builder.snapshot();
        assert!(cleared.route.mq.is_none());
        assert!(cleared.route.cics.is_none());
        assert!(cleared.route.dataset.is_none());
    }

    #[test]
    fn resetting_created_at_restores_current_timestamp() {
        let mut builder = CobolEnvelopeBuilder::new("job-404");
        builder.set_created_at("2020-01-01T00:00:00Z");
        builder.reset_created_at();
        assert_ne!(builder.snapshot().created_at, "2020-01-01T00:00:00Z");
    }

    #[test]
    fn builder_from_envelope_sanitises_imported_state() {
        let envelope = CobolEnvelope {
            job_id: "  ".to_string(),
            release_channel: " shadow ".to_string(),
            created_at: " ".to_string(),
            initiators: vec![InteractionInitiator {
                kind: InitiatorKind::Human,
                name: "".to_string(),
                persona: Some(" guide ".to_string()),
                revision: Some("  ".to_string()),
                contact: Some("  ops@example  ".to_string()),
                notes: vec!["  note  ".to_string(), "  ".to_string()],
            }],
            route: CobolRoute {
                mq: Some(CobolMqRoute {
                    manager: "   ".to_string(),
                    queue: " inbound ".to_string(),
                    commit: Some("   ".to_string()),
                }),
                cics: Some(CobolCicsRoute {
                    transaction: " TRN1 ".to_string(),
                    program: Some("  ".to_string()),
                    channel: Some(" CHAN ".to_string()),
                }),
                dataset: Some("  DATA.SET  ".to_string()),
            },
            payload: CobolNarratorPayload {
                curvature: 0.7,
                temperature: 0.3,
                encoder: "  encoder.custom  ".to_string(),
                locale: Some("  en-US  ".to_string()),
                coefficients: vec![0.12],
            },
            metadata: CobolMetadata {
                tags: vec!["  tag-one  ".to_string(), "  ".to_string()],
                annotations: vec!["  ".to_string(), PLANNER_INITIALIZED_ANNOTATION.to_string()],
                extra: default_metadata_extra(),
            },
        };

        let builder = CobolEnvelopeBuilder::from_envelope(envelope);
        let snapshot = builder.snapshot();

        assert_eq!(snapshot.job_id, "job");
        assert_eq!(snapshot.release_channel, "shadow");
        assert_eq!(
            snapshot
                .metadata
                .annotations
                .iter()
                .filter(|value| value.as_str() == PLANNER_INITIALIZED_ANNOTATION)
                .count(),
            1
        );
        assert_eq!(snapshot.route.mq, None);
        assert_eq!(
            snapshot
                .route
                .cics
                .as_ref()
                .map(|cics| cics.transaction.clone()),
            Some("TRN1".to_string())
        );
        assert_eq!(snapshot.route.dataset.as_deref(), Some("DATA.SET"));
        assert_eq!(snapshot.metadata.tags, vec!["tag-one".to_string()]);
        assert_eq!(
            snapshot.initiators.first().expect("initiator").name,
            "participant"
        );
        assert_eq!(
            snapshot.initiators.first().unwrap().persona.as_deref(),
            Some("guide")
        );
        assert_eq!(
            snapshot.initiators.first().unwrap().contact.as_deref(),
            Some("ops@example")
        );
        assert_eq!(
            snapshot.initiators.first().unwrap().notes,
            vec!["note".to_string()]
        );
        assert_eq!(snapshot.payload.encoder, "encoder.custom");
        assert_eq!(snapshot.payload.locale.as_deref(), Some("en-US"));
        assert_ne!(snapshot.created_at, " ");
    }

    #[test]
    fn builder_load_envelope_replaces_previous_state() {
        let mut builder = CobolEnvelopeBuilder::new("job-a");
        builder.add_tag("alpha");
        let mut replacement = builder.snapshot();
        replacement.job_id = "job-b".to_string();
        replacement.release_channel = " shadow ".to_string();
        builder.load_envelope(replacement);

        let snapshot = builder.snapshot();
        assert_eq!(snapshot.job_id, "job-b");
        assert_eq!(snapshot.release_channel, "shadow");
        assert!(snapshot.metadata.tags.contains(&"alpha".to_string()));
    }

    #[test]
    fn validation_flags_missing_initiators_and_route() {
        let builder = CobolEnvelopeBuilder::new("job-91");
        let issues = builder.validation_issues();
        assert_eq!(issues.len(), 2);
        assert!(issues.contains(&"add at least one initiator before dispatch".to_string()));
        assert!(issues.contains(&"configure a delivery route (MQ, CICS, or dataset)".to_string()));
    }

    #[test]
    fn validation_marks_default_job_identifier() {
        let builder = CobolEnvelopeBuilder::new("   ");
        let issues = builder.validation_issues();
        assert!(issues.contains(&"replace the default job identifier before dispatch".to_string()));
    }

    #[test]
    fn validation_flags_out_of_range_metrics() {
        let mut builder = CobolEnvelopeBuilder::new("job-curve");
        builder.add_initiator(make_initiator(
            InitiatorKind::Human,
            "Operator",
            None,
            None,
            Some("ops@example".into()),
            None,
        ));
        builder.set_mq_route("QM1", "QUEUE", None);
        builder.set_narrator_config(1.2, -0.4, "spiraltorch.default", None);
        let issues = builder.validation_issues();
        assert_eq!(issues.len(), 2);
        assert!(issues.contains(&"curvature must be between 0.0 and 1.0".to_string()));
        assert!(issues.contains(&"temperature must be between 0.0 and 1.0".to_string()));
    }

    #[test]
    fn validation_succeeds_for_complete_envelope() {
        let mut builder = CobolEnvelopeBuilder::new("job-ready");
        builder.add_initiator(make_initiator(
            InitiatorKind::Automation,
            "scheduler",
            None,
            None,
            None,
            None,
        ));
        builder.set_cics_route("TRN1", Some("PROG1".into()), None);
        builder.set_narrator_config(0.6, 0.4, "spiraltorch.default", None);
        assert!(builder.is_valid());
        assert!(builder.validation_issues().is_empty());
    }
}
