use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use time::{format_description::well_known::Rfc3339, OffsetDateTime};

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
        let envelope = CobolEnvelope {
            job_id: sanitize(job_id.into()).unwrap_or_else(|| "job".to_string()),
            release_channel: "production".to_string(),
            created_at: now_timestamp(),
            initiators: Vec::new(),
            route: CobolRoute::default(),
            payload: CobolNarratorPayload::default(),
            metadata: CobolMetadata::default(),
        };
        Self::from_envelope(envelope)
    }

    pub fn from_envelope(envelope: CobolEnvelope) -> Self {
        let mut builder = Self {
            envelope: sanitize_envelope(envelope),
        };
        ensure_initialized_annotation(&mut builder.envelope);
        builder
    }

    pub fn from_json_str(json: &str) -> serde_json::Result<Self> {
        let envelope = CobolEnvelope::from_json_str(json)?;
        Ok(Self::from_envelope(envelope))
    }

    pub fn from_json_bytes(bytes: &[u8]) -> serde_json::Result<Self> {
        let envelope = CobolEnvelope::from_json_bytes(bytes)?;
        Ok(Self::from_envelope(envelope))
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

    pub fn set_dataset(&mut self, dataset: Option<String>) {
        self.envelope.route.dataset = dataset.and_then(sanitize);
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

    pub fn replace_envelope(&mut self, envelope: CobolEnvelope) {
        self.envelope = sanitize_envelope(envelope);
        ensure_initialized_annotation(&mut self.envelope);
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

    pub fn from_json_str(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }

    pub fn from_json_bytes(bytes: &[u8]) -> serde_json::Result<Self> {
        serde_json::from_slice(bytes)
    }
}

fn now_timestamp() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string())
}

fn sanitize(value: String) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn sanitize_vec(values: Vec<String>) -> Vec<String> {
    values
        .into_iter()
        .filter_map(|value| sanitize(value))
        .collect()
}

fn sanitize_required(value: String, fallback: &str) -> String {
    sanitize(value).unwrap_or_else(|| fallback.to_string())
}

fn ensure_initialized_annotation(envelope: &mut CobolEnvelope) {
    if !envelope
        .metadata
        .annotations
        .iter()
        .any(|annotation| annotation == "planner_initialized")
    {
        envelope
            .metadata
            .annotations
            .push("planner_initialized".to_string());
    }
}

fn sanitize_envelope(mut envelope: CobolEnvelope) -> CobolEnvelope {
    envelope.job_id = sanitize_required(envelope.job_id, "job");
    envelope.release_channel = sanitize_required(envelope.release_channel, "production");
    envelope.created_at = sanitize(envelope.created_at).unwrap_or_else(now_timestamp);

    if let Some(ref mut mq) = envelope.route.mq {
        mq.manager = sanitize_required(mq.manager.clone(), "default");
        mq.queue = sanitize_required(mq.queue.clone(), "queue");
        mq.commit = mq.commit.take().and_then(sanitize);
    }

    if let Some(ref mut cics) = envelope.route.cics {
        cics.transaction = sanitize_required(cics.transaction.clone(), "TX");
        cics.program = cics.program.take().and_then(sanitize);
        cics.channel = cics.channel.take().and_then(sanitize);
    }

    envelope.route.dataset = envelope.route.dataset.and_then(sanitize);

    for initiator in &mut envelope.initiators {
        initiator.name = sanitize_required(initiator.name.clone(), "participant");
        initiator.persona = initiator.persona.take().and_then(sanitize);
        initiator.revision = initiator.revision.take().and_then(sanitize);
        initiator.contact = initiator.contact.take().and_then(sanitize);
        initiator.notes = sanitize_vec(std::mem::take(&mut initiator.notes));
    }

    envelope.metadata.tags = sanitize_vec(envelope.metadata.tags);
    envelope.metadata.annotations = sanitize_vec(envelope.metadata.annotations);

    envelope
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
    fn builder_round_trips_json() {
        let mut original = CobolEnvelopeBuilder::new("job-200");
        original.set_release_channel(" staging ");
        original.set_narrator_config(0.7, 0.3, " custom ", Some(" fr-FR ".into()));
        let envelope = original.snapshot();
        let json = envelope.to_json_string().unwrap();

        let imported = CobolEnvelopeBuilder::from_json_str(&json).unwrap();
        assert_eq!(imported.envelope().job_id, "job-200");
        assert_eq!(imported.envelope().release_channel, "staging");
        assert_eq!(imported.envelope().payload.locale.as_deref(), Some("fr-FR"));
        assert!(imported
            .envelope()
            .metadata
            .annotations
            .contains(&"planner_initialized".to_string()));
    }

    #[test]
    fn sanitize_envelope_on_import() {
        let raw = CobolEnvelope {
            job_id: "   ".into(),
            release_channel: "  ".into(),
            created_at: "".into(),
            initiators: vec![InteractionInitiator {
                kind: InitiatorKind::Human,
                name: "  ".into(),
                persona: Some("  ops  ".into()),
                revision: None,
                contact: Some("  contact  ".into()),
                notes: vec!["  note  ".into(), "".into()],
            }],
            route: CobolRoute {
                mq: Some(CobolMqRoute {
                    manager: "   ".into(),
                    queue: "   ".into(),
                    commit: Some("   ".into()),
                }),
                cics: Some(CobolCicsRoute {
                    transaction: "   ".into(),
                    program: Some("  program  ".into()),
                    channel: Some("".into()),
                }),
                dataset: Some("  dataset  ".into()),
            },
            payload: CobolNarratorPayload::default(),
            metadata: CobolMetadata {
                tags: vec!["  a  ".into(), "".into()],
                annotations: vec!["existing".into(), "".into()],
                extra: serde_json::json!({"keep": "me"}),
            },
        };

        let builder = CobolEnvelopeBuilder::from_envelope(raw);
        let envelope = builder.envelope();
        assert_eq!(envelope.job_id, "job");
        assert_eq!(envelope.release_channel, "production");
        assert!(!envelope.created_at.is_empty());
        assert_eq!(envelope.route.dataset.as_deref(), Some("dataset"));
        assert_eq!(
            envelope.route.mq.as_ref().map(|mq| (
                mq.manager.clone(),
                mq.queue.clone(),
                mq.commit.clone()
            )),
            Some(("default".into(), "queue".into(), None))
        );
        assert_eq!(
            envelope.route.cics.as_ref().map(|cics| (
                cics.transaction.clone(),
                cics.program.clone(),
                cics.channel.clone()
            )),
            Some(("TX".into(), Some("program".into()), None))
        );
        assert_eq!(envelope.initiators[0].name, "participant");
        assert_eq!(envelope.initiators[0].persona.as_deref(), Some("ops"));
        assert_eq!(envelope.initiators[0].contact.as_deref(), Some("contact"));
        assert_eq!(envelope.initiators[0].notes, vec!["note".to_string()]);
        assert_eq!(envelope.metadata.tags, vec!["a".to_string()]);
        assert!(envelope
            .metadata
            .annotations
            .contains(&"existing".to_string()));
        assert!(envelope
            .metadata
            .annotations
            .contains(&"planner_initialized".to_string()));
    }
}
