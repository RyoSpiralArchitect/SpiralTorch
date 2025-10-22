use js_sys::{Float32Array, Uint8Array};
use serde_wasm_bindgen as swb;
use wasm_bindgen::prelude::*;

use crate::cobol::{make_initiator, CobolEnvelope, CobolEnvelopeBuilder, InitiatorKind};
use crate::utils::{js_error, json_to_js_value};

#[wasm_bindgen]
pub struct CobolDispatchPlanner {
    builder: CobolEnvelopeBuilder,
}

#[wasm_bindgen]
impl CobolDispatchPlanner {
    #[wasm_bindgen(constructor)]
    pub fn new(job_id: &str, release_channel: Option<String>) -> CobolDispatchPlanner {
        let mut builder = CobolEnvelopeBuilder::new(job_id);
        if let Some(channel) = release_channel {
            builder.set_release_channel(channel);
        }
        CobolDispatchPlanner { builder }
    }

    #[wasm_bindgen(js_name = fromJson)]
    pub fn from_json(json: &str) -> Result<CobolDispatchPlanner, JsValue> {
        let envelope = CobolEnvelope::from_json_str(json).map_err(js_error)?;
        Ok(CobolDispatchPlanner {
            builder: CobolEnvelopeBuilder::from_envelope(envelope),
        })
    }

    #[wasm_bindgen(js_name = fromObject)]
    pub fn from_object(value: &JsValue) -> Result<CobolDispatchPlanner, JsValue> {
        let envelope: CobolEnvelope = swb::from_value(value.clone()).map_err(js_error)?;
        Ok(CobolDispatchPlanner {
            builder: CobolEnvelopeBuilder::from_envelope(envelope),
        })
    }

    #[wasm_bindgen(js_name = setReleaseChannel)]
    pub fn set_release_channel(&mut self, channel: &str) {
        self.builder.set_release_channel(channel.to_string());
    }

    #[wasm_bindgen(js_name = setCreatedAt)]
    pub fn set_created_at(&mut self, timestamp: &str) {
        self.builder.set_created_at(timestamp.to_string());
    }

    #[wasm_bindgen(js_name = resetCreatedAt)]
    pub fn reset_created_at(&mut self) {
        self.builder.reset_created_at();
    }

    #[wasm_bindgen(js_name = setNarratorConfig)]
    pub fn set_narrator_config(
        &mut self,
        curvature: f32,
        temperature: f32,
        encoder: &str,
        locale: Option<String>,
    ) {
        self.builder
            .set_narrator_config(curvature, temperature, encoder.to_string(), locale);
    }

    #[wasm_bindgen(js_name = setCoefficients)]
    pub fn set_coefficients(&mut self, coefficients: &Float32Array) {
        let len = coefficients.length() as usize;
        let mut buffer = vec![0.0f32; len];
        coefficients.copy_to(&mut buffer);
        self.builder.set_coefficients(buffer);
    }

    #[wasm_bindgen(js_name = addHumanInitiator)]
    pub fn add_human_initiator(
        &mut self,
        name: &str,
        persona: Option<String>,
        contact: Option<String>,
        note: Option<String>,
    ) {
        let initiator = make_initiator(
            InitiatorKind::Human,
            name.to_string(),
            persona,
            None,
            contact,
            note,
        );
        self.builder.add_initiator(initiator);
    }

    #[wasm_bindgen(js_name = addModelInitiator)]
    pub fn add_model_initiator(
        &mut self,
        name: &str,
        revision: Option<String>,
        persona: Option<String>,
        note: Option<String>,
    ) {
        let initiator = make_initiator(
            InitiatorKind::Model,
            name.to_string(),
            persona,
            revision,
            None,
            note,
        );
        self.builder.add_initiator(initiator);
    }

    #[wasm_bindgen(js_name = addAutomationInitiator)]
    pub fn add_automation_initiator(
        &mut self,
        name: &str,
        persona: Option<String>,
        note: Option<String>,
    ) {
        let initiator = make_initiator(
            InitiatorKind::Automation,
            name.to_string(),
            persona,
            None,
            None,
            note,
        );
        self.builder.add_initiator(initiator);
    }

    #[wasm_bindgen(js_name = clearInitiators)]
    pub fn clear_initiators(&mut self) {
        self.builder.clear_initiators();
    }

    #[wasm_bindgen(js_name = setMqRoute)]
    pub fn set_mq_route(&mut self, manager: &str, queue: &str, commit: Option<String>) {
        self.builder
            .set_mq_route(manager.to_string(), queue.to_string(), commit);
    }

    #[wasm_bindgen(js_name = clearMqRoute)]
    pub fn clear_mq_route(&mut self) {
        self.builder.clear_mq_route();
    }

    #[wasm_bindgen(js_name = setCicsRoute)]
    pub fn set_cics_route(
        &mut self,
        transaction: &str,
        program: Option<String>,
        channel: Option<String>,
    ) {
        self.builder
            .set_cics_route(transaction.to_string(), program, channel);
    }

    #[wasm_bindgen(js_name = clearCicsRoute)]
    pub fn clear_cics_route(&mut self) {
        self.builder.clear_cics_route();
    }

    #[wasm_bindgen(js_name = setDataset)]
    pub fn set_dataset(&mut self, dataset: &str) {
        self.builder.set_dataset(Some(dataset.to_string()));
    }

    #[wasm_bindgen(js_name = clearDataset)]
    pub fn clear_dataset(&mut self) {
        self.builder.set_dataset(None);
    }

    #[wasm_bindgen(js_name = clearRoute)]
    pub fn clear_route(&mut self) {
        self.builder.clear_route();
    }

    #[wasm_bindgen(js_name = addTag)]
    pub fn add_tag(&mut self, tag: &str) {
        self.builder.add_tag(tag.to_string());
    }

    #[wasm_bindgen(js_name = addAnnotation)]
    pub fn add_annotation(&mut self, annotation: &str) {
        self.builder.add_annotation(annotation.to_string());
    }

    #[wasm_bindgen(js_name = mergeMetadata)]
    pub fn merge_metadata(&mut self, metadata: &JsValue) -> Result<(), JsValue> {
        if metadata.is_null() || metadata.is_undefined() {
            return Ok(());
        }
        let value: serde_json::Value =
            swb::from_value(metadata.clone()).map_err(|err| js_error(err))?;
        self.builder.merge_metadata_value(value);
        Ok(())
    }

    #[wasm_bindgen(js_name = clearMetadata)]
    pub fn clear_metadata(&mut self) {
        self.builder.clear_metadata();
    }

    #[wasm_bindgen(js_name = loadJson)]
    pub fn load_json(&mut self, json: &str) -> Result<(), JsValue> {
        let envelope = CobolEnvelope::from_json_str(json).map_err(js_error)?;
        self.builder.load_envelope(envelope);
        Ok(())
    }

    #[wasm_bindgen(js_name = loadObject)]
    pub fn load_object(&mut self, value: &JsValue) -> Result<(), JsValue> {
        let envelope: CobolEnvelope = swb::from_value(value.clone()).map_err(js_error)?;
        self.builder.load_envelope(envelope);
        Ok(())
    }

    #[wasm_bindgen(js_name = toObject)]
    pub fn to_object(&self) -> Result<JsValue, JsValue> {
        swb::to_value(self.builder.envelope()).map_err(|err| js_error(err))
    }

    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> Result<String, JsValue> {
        self.builder.envelope().to_json_string().map_err(js_error)
    }

    #[wasm_bindgen(js_name = toUint8Array)]
    pub fn to_uint8_array(&self) -> Result<Uint8Array, JsValue> {
        let bytes = self.builder.envelope().to_json_bytes().map_err(js_error)?;
        let array = Uint8Array::new_with_length(bytes.len() as u32);
        array.copy_from(&bytes);
        Ok(array)
    }

    #[wasm_bindgen(js_name = coefficientsAsBytes)]
    pub fn coefficients_as_bytes(&self) -> Uint8Array {
        let envelope = self.builder.envelope();
        let mut bytes = Vec::with_capacity(envelope.payload.coefficients.len() * 4);
        for value in &envelope.payload.coefficients {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let array = Uint8Array::new_with_length(bytes.len() as u32);
        array.copy_from(&bytes);
        array
    }

    #[wasm_bindgen(getter, js_name = createdAt)]
    pub fn created_at(&self) -> String {
        self.builder.envelope().created_at.clone()
    }

    #[wasm_bindgen(js_name = mqRoute)]
    pub fn mq_route(&self) -> Result<JsValue, JsValue> {
        match self.builder.envelope().route.mq.as_ref() {
            Some(route) => swb::to_value(route).map_err(|err| js_error(err)),
            None => Ok(JsValue::UNDEFINED),
        }
    }

    #[wasm_bindgen(js_name = cicsRoute)]
    pub fn cics_route(&self) -> Result<JsValue, JsValue> {
        match self.builder.envelope().route.cics.as_ref() {
            Some(route) => swb::to_value(route).map_err(|err| js_error(err)),
            None => Ok(JsValue::UNDEFINED),
        }
    }

    #[wasm_bindgen(js_name = toCobolPreview)]
    pub fn to_cobol_preview(&self) -> Result<JsValue, JsValue> {
        let envelope = self.builder.envelope();
        let preview = CobolPreview {
            job_id: &envelope.job_id,
            curvature: envelope.payload.curvature,
            temperature: envelope.payload.temperature,
            coefficient_count: envelope.payload.coefficients.len() as u32,
            dataset: envelope.route.dataset.as_deref(),
            release_channel: &envelope.release_channel,
        };
        let json = serde_json::to_string(&preview).map_err(js_error)?;
        json_to_js_value(&json)
    }
}

#[derive(serde::Serialize)]
struct CobolPreview<'a> {
    job_id: &'a str,
    curvature: f32,
    temperature: f32,
    coefficient_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    dataset: Option<&'a str>,
    release_channel: &'a str,
}
