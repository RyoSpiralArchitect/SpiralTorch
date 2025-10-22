use js_sys::{Float32Array, Uint8Array};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use serde_wasm_bindgen as swb;
use wasm_bindgen::prelude::*;

use crate::utils::{js_error, json_to_js_value};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CobolEnvelope {
    job_id: String,
    release_channel: String,
    created_at: String,
    initiators: Vec<InteractionInitiator>,
    route: CobolRoute,
    payload: CobolNarratorPayload,
    metadata: CobolMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct CobolRoute {
    mq: Option<CobolMqRoute>,
    cics: Option<CobolCicsRoute>,
    dataset: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CobolMqRoute {
    manager: String,
    queue: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    commit: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CobolCicsRoute {
    transaction: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    program: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    channel: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct CobolMetadata {
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    annotations: Vec<String>,
    #[serde(default = "default_metadata_extra")]
    extra: Value,
}

fn default_metadata_extra() -> Value {
    Value::Object(Map::default())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CobolNarratorPayload {
    curvature: f32,
    temperature: f32,
    encoder: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    locale: Option<String>,
    #[serde(default)]
    coefficients: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InteractionInitiator {
    #[serde(rename = "type")]
    kind: InitiatorKind,
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    persona: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    revision: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    contact: Option<String>,
    #[serde(default)]
    notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum InitiatorKind {
    Human,
    Model,
    Automation,
}

impl CobolEnvelope {
    fn new(job_id: String, release_channel: String) -> Self {
        let created_at = js_sys::Date::new_0().to_iso_string().into();
        Self {
            job_id,
            release_channel,
            created_at,
            initiators: Vec::new(),
            route: CobolRoute::default(),
            payload: CobolNarratorPayload {
                curvature: 0.5,
                temperature: 0.5,
                encoder: "spiraltorch.default".to_string(),
                locale: None,
                coefficients: Vec::new(),
            },
            metadata: CobolMetadata::default(),
        }
    }
}

fn optional_string(value: Option<String>) -> Option<String> {
    value.and_then(|v| {
        let trimmed = v.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn optional_note(note: Option<String>) -> Option<String> {
    optional_string(note)
}

fn optional_tag(tag: &str) -> Option<String> {
    let trimmed = tag.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

#[wasm_bindgen]
pub struct CobolDispatchPlanner {
    envelope: CobolEnvelope,
}

#[wasm_bindgen]
impl CobolDispatchPlanner {
    /// Create a new planner for dispatching narrations to COBOL estates.
    #[wasm_bindgen(constructor)]
    pub fn new(job_id: &str, release_channel: Option<String>) -> CobolDispatchPlanner {
        let channel = optional_string(release_channel).unwrap_or_else(|| "production".to_string());
        CobolDispatchPlanner {
            envelope: CobolEnvelope::new(job_id.to_string(), channel),
        }
    }

    /// Change the release channel (for example `staging`, `shadow`, or `production`).
    #[wasm_bindgen(js_name = setReleaseChannel)]
    pub fn set_release_channel(&mut self, channel: &str) {
        if let Some(clean) = optional_tag(channel) {
            self.envelope.release_channel = clean;
        }
    }

    /// Install the narrator parameters that COBOL will forward to the shared library.
    #[wasm_bindgen(js_name = setNarratorConfig)]
    pub fn set_narrator_config(
        &mut self,
        curvature: f32,
        temperature: f32,
        encoder: &str,
        locale: Option<String>,
    ) {
        self.envelope.payload.curvature = curvature;
        self.envelope.payload.temperature = temperature;
        if let Some(clean) = optional_tag(encoder) {
            self.envelope.payload.encoder = clean;
        }
        self.envelope.payload.locale = optional_string(locale);
    }

    /// Replace the coefficient buffer with the values provided by the WASM runtime.
    #[wasm_bindgen(js_name = setCoefficients)]
    pub fn set_coefficients(&mut self, coefficients: &Float32Array) {
        let len = coefficients.length() as usize;
        let mut buffer = vec![0.0f32; len];
        coefficients.copy_to(&mut buffer);
        self.envelope.payload.coefficients = buffer;
    }

    /// Append a human operator who curated the narration request.
    #[wasm_bindgen(js_name = addHumanInitiator)]
    pub fn add_human_initiator(
        &mut self,
        name: &str,
        persona: Option<String>,
        contact: Option<String>,
        note: Option<String>,
    ) {
        let mut initiator = InteractionInitiator {
            kind: InitiatorKind::Human,
            name: name.to_string(),
            persona: optional_string(persona),
            revision: None,
            contact: optional_string(contact),
            notes: Vec::new(),
        };
        if let Some(note) = optional_note(note) {
            initiator.notes.push(note);
        }
        self.envelope.initiators.push(initiator);
    }

    /// Append a model initiator (for example an LM agent running inside the browser).
    #[wasm_bindgen(js_name = addModelInitiator)]
    pub fn add_model_initiator(
        &mut self,
        name: &str,
        revision: Option<String>,
        persona: Option<String>,
        note: Option<String>,
    ) {
        let mut initiator = InteractionInitiator {
            kind: InitiatorKind::Model,
            name: name.to_string(),
            persona: optional_string(persona),
            revision: optional_string(revision),
            contact: None,
            notes: Vec::new(),
        };
        if let Some(note) = optional_note(note) {
            initiator.notes.push(note);
        }
        self.envelope.initiators.push(initiator);
    }

    /// Append an automation or workflow initiator.
    #[wasm_bindgen(js_name = addAutomationInitiator)]
    pub fn add_automation_initiator(
        &mut self,
        name: &str,
        persona: Option<String>,
        note: Option<String>,
    ) {
        let mut initiator = InteractionInitiator {
            kind: InitiatorKind::Automation,
            name: name.to_string(),
            persona: optional_string(persona),
            revision: None,
            contact: None,
            notes: Vec::new(),
        };
        if let Some(note) = optional_note(note) {
            initiator.notes.push(note);
        }
        self.envelope.initiators.push(initiator);
    }

    /// Route the payload to an MQ queue on z/OS.
    #[wasm_bindgen(js_name = setMqRoute)]
    pub fn set_mq_route(&mut self, manager: &str, queue: &str, commit_mode: Option<String>) {
        self.envelope.route.mq = Some(CobolMqRoute {
            manager: manager.to_string(),
            queue: queue.to_string(),
            commit: optional_string(commit_mode),
        });
    }

    /// Route the payload to a CICS transaction.
    #[wasm_bindgen(js_name = setCicsRoute)]
    pub fn set_cics_route(
        &mut self,
        transaction: &str,
        program: Option<String>,
        channel: Option<String>,
    ) {
        self.envelope.route.cics = Some(CobolCicsRoute {
            transaction: transaction.to_string(),
            program: optional_string(program),
            channel: optional_string(channel),
        });
    }

    /// Set the dataset or GDG that should persist the narration.
    #[wasm_bindgen(js_name = setDataset)]
    pub fn set_dataset(&mut self, dataset: &str) {
        self.envelope.route.dataset = optional_tag(dataset);
    }

    /// Attach a tag that COBOL orchestration can use for routing.
    #[wasm_bindgen(js_name = addTag)]
    pub fn add_tag(&mut self, tag: &str) {
        if let Some(tag) = optional_tag(tag) {
            self.envelope.metadata.tags.push(tag);
        }
    }

    /// Append an annotation describing the narration.
    #[wasm_bindgen(js_name = addAnnotation)]
    pub fn add_annotation(&mut self, annotation: &str) {
        if let Some(annotation) = optional_tag(annotation) {
            self.envelope.metadata.annotations.push(annotation);
        }
    }

    /// Merge structured metadata (JavaScript objects) into the envelope.
    #[wasm_bindgen(js_name = mergeMetadata)]
    pub fn merge_metadata(&mut self, metadata: &JsValue) -> Result<(), JsValue> {
        if metadata.is_undefined() || metadata.is_null() {
            return Ok(());
        }
        let incoming: Value = swb::from_value(metadata.clone()).map_err(|err| js_error(err))?;
        match incoming {
            Value::Object(mut map) => match self.envelope.metadata.extra {
                Value::Object(ref mut existing) => {
                    existing.append(&mut map);
                }
                _ => {
                    self.envelope.metadata.extra = Value::Object(map);
                }
            },
            other => {
                self.envelope.metadata.extra = other;
            }
        }
        Ok(())
    }

    /// Reset any structured metadata to an empty object.
    #[wasm_bindgen(js_name = clearMetadata)]
    pub fn clear_metadata(&mut self) {
        self.envelope.metadata.extra = default_metadata_extra();
    }

    /// Produce the envelope as a JavaScript object.
    #[wasm_bindgen(js_name = toObject)]
    pub fn to_object(&self) -> Result<JsValue, JsValue> {
        swb::to_value(&self.envelope).map_err(|err| js_error(err))
    }

    /// Produce the envelope as a JSON string.
    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string_pretty(&self.envelope).map_err(js_error)
    }

    /// Produce a Uint8Array JSON payload ready for MQ or dataset uploads.
    #[wasm_bindgen(js_name = toUint8Array)]
    pub fn to_uint8_array(&self) -> Result<Uint8Array, JsValue> {
        let bytes = serde_json::to_vec(&self.envelope).map_err(js_error)?;
        let array = Uint8Array::new_with_length(bytes.len() as u32);
        array.copy_from(&bytes);
        Ok(array)
    }

    /// Emit just the coefficient buffer as little-endian bytes.
    #[wasm_bindgen(js_name = coefficientsAsBytes)]
    pub fn coefficients_as_bytes(&self) -> Uint8Array {
        let mut bytes = Vec::with_capacity(self.envelope.payload.coefficients.len() * 4);
        for value in &self.envelope.payload.coefficients {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let array = Uint8Array::new_with_length(bytes.len() as u32);
        array.copy_from(&bytes);
        array
    }

    /// Return the envelope creation timestamp.
    #[wasm_bindgen(getter, js_name = createdAt)]
    pub fn created_at(&self) -> String {
        self.envelope.created_at.clone()
    }

    /// Convenience accessor for the MQ route as a JavaScript object.
    #[wasm_bindgen(js_name = mqRoute)]
    pub fn mq_route(&self) -> Result<JsValue, JsValue> {
        match &self.envelope.route.mq {
            Some(route) => swb::to_value(route).map_err(|err| js_error(err)),
            None => Ok(JsValue::UNDEFINED),
        }
    }

    /// Convenience accessor for the CICS route as a JavaScript object.
    #[wasm_bindgen(js_name = cicsRoute)]
    pub fn cics_route(&self) -> Result<JsValue, JsValue> {
        match &self.envelope.route.cics {
            Some(route) => swb::to_value(route).map_err(|err| js_error(err)),
            None => Ok(JsValue::UNDEFINED),
        }
    }

    /// Build a JSON payload with COBOL-friendly pointers for cross-checking.
    #[wasm_bindgen(js_name = toCobolPreview)]
    pub fn to_cobol_preview(&self) -> Result<JsValue, JsValue> {
        let payload = CobolPreview {
            job_id: &self.envelope.job_id,
            curvature: self.envelope.payload.curvature,
            temperature: self.envelope.payload.temperature,
            coefficient_count: self.envelope.payload.coefficients.len() as u32,
            dataset: self.envelope.route.dataset.clone(),
            release_channel: &self.envelope.release_channel,
        };
        let json = serde_json::to_string(&payload).map_err(js_error)?;
        json_to_js_value(&json)
    }
}

#[derive(Serialize)]
struct CobolPreview<'a> {
    job_id: &'a str,
    curvature: f32,
    temperature: f32,
    coefficient_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    dataset: Option<String>,
    release_channel: &'a str,
}
