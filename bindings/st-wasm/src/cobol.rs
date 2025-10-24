use serde::de::{self, Deserializer, MapAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{Map, Value};
use time::{format_description::well_known::Rfc3339, OffsetDateTime};

const DEFAULT_JOB_PLACEHOLDER: &str = "job";
const PLANNER_INITIALIZED_ANNOTATION: &str = "planner_initialized";
const NARRATOR_METRIC_MIN: f32 = 0.0;
const NARRATOR_METRIC_MAX: f32 = 1.0;
const FALLBACK_TIMESTAMP: &str = "1970-01-01T00:00:00Z";
const ALLOWED_SPACE_UNITS: &[&str] = &["CYL", "TRK", "MB", "KB"];
const ALLOWED_DATASET_TYPES: &[&str] = &[
    "BASIC", "LARGE", "LIBRARY", "EXTREQ", "EXTPREF", "PDS", "HFS",
];

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
    pub dataset: Option<CobolDatasetRoute>,
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

#[derive(Debug, Clone, PartialEq)]
pub struct CobolDatasetRoute {
    pub dataset: String,
    pub member: Option<String>,
    pub disposition: Option<String>,
    pub volume: Option<String>,
    pub record_format: Option<String>,
    pub record_length: Option<u32>,
    pub block_size: Option<u32>,
    pub data_class: Option<String>,
    pub management_class: Option<String>,
    pub storage_class: Option<String>,
    pub space_primary: Option<u32>,
    pub space_secondary: Option<u32>,
    pub space_unit: Option<String>,
    pub directory_blocks: Option<u32>,
    pub dataset_type: Option<String>,
    pub like_dataset: Option<String>,
}

impl Default for CobolDatasetRoute {
    fn default() -> Self {
        Self {
            dataset: String::new(),
            member: None,
            disposition: None,
            volume: None,
            record_format: None,
            record_length: None,
            block_size: None,
            data_class: None,
            management_class: None,
            storage_class: None,
            space_primary: None,
            space_secondary: None,
            space_unit: None,
            directory_blocks: None,
            dataset_type: None,
            like_dataset: None,
        }
    }
}

impl Serialize for CobolDatasetRoute {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if self.member.is_none()
            && self.disposition.is_none()
            && self.volume.is_none()
            && self.record_format.is_none()
            && self.record_length.is_none()
            && self.block_size.is_none()
            && self.data_class.is_none()
            && self.management_class.is_none()
            && self.storage_class.is_none()
            && self.space_primary.is_none()
            && self.space_secondary.is_none()
            && self.space_unit.is_none()
            && self.directory_blocks.is_none()
            && self.dataset_type.is_none()
            && self.like_dataset.is_none()
        {
            serializer.serialize_str(&self.dataset)
        } else {
            let mut entries = 1;
            if self.member.is_some() {
                entries += 1;
            }
            if self.disposition.is_some() {
                entries += 1;
            }
            if self.volume.is_some() {
                entries += 1;
            }
            if self.record_format.is_some() {
                entries += 1;
            }
            if self.record_length.is_some() {
                entries += 1;
            }
            if self.block_size.is_some() {
                entries += 1;
            }
            if self.data_class.is_some() {
                entries += 1;
            }
            if self.management_class.is_some() {
                entries += 1;
            }
            if self.storage_class.is_some() {
                entries += 1;
            }
            if self.space_primary.is_some() {
                entries += 1;
            }
            if self.space_secondary.is_some() {
                entries += 1;
            }
            if self.space_unit.is_some() {
                entries += 1;
            }
            if self.directory_blocks.is_some() {
                entries += 1;
            }
            if self.dataset_type.is_some() {
                entries += 1;
            }
            if self.like_dataset.is_some() {
                entries += 1;
            }
            let mut map = serializer.serialize_map(Some(entries))?;
            map.serialize_entry("dataset", &self.dataset)?;
            if let Some(member) = &self.member {
                map.serialize_entry("member", member)?;
            }
            if let Some(disposition) = &self.disposition {
                map.serialize_entry("disposition", disposition)?;
            }
            if let Some(volume) = &self.volume {
                map.serialize_entry("volume", volume)?;
            }
            if let Some(record_format) = &self.record_format {
                map.serialize_entry("record_format", record_format)?;
            }
            if let Some(record_length) = &self.record_length {
                map.serialize_entry("record_length", record_length)?;
            }
            if let Some(block_size) = &self.block_size {
                map.serialize_entry("block_size", block_size)?;
            }
            if let Some(data_class) = &self.data_class {
                map.serialize_entry("data_class", data_class)?;
            }
            if let Some(management_class) = &self.management_class {
                map.serialize_entry("management_class", management_class)?;
            }
            if let Some(storage_class) = &self.storage_class {
                map.serialize_entry("storage_class", storage_class)?;
            }
            if let Some(space_primary) = &self.space_primary {
                map.serialize_entry("space_primary", space_primary)?;
            }
            if let Some(space_secondary) = &self.space_secondary {
                map.serialize_entry("space_secondary", space_secondary)?;
            }
            if let Some(space_unit) = &self.space_unit {
                map.serialize_entry("space_unit", space_unit)?;
            }
            if let Some(directory_blocks) = &self.directory_blocks {
                map.serialize_entry("directory_blocks", directory_blocks)?;
            }
            if let Some(dataset_type) = &self.dataset_type {
                map.serialize_entry("dataset_type", dataset_type)?;
            }
            if let Some(like_dataset) = &self.like_dataset {
                map.serialize_entry("like_dataset", like_dataset)?;
            }
            map.end()
        }
    }
}

impl<'de> Deserialize<'de> for CobolDatasetRoute {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct DatasetVisitor;

        impl<'de> Visitor<'de> for DatasetVisitor {
            type Value = CobolDatasetRoute;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a dataset name string or an object with dataset metadata")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(CobolDatasetRoute {
                    dataset: value.to_string(),
                    member: None,
                    disposition: None,
                    volume: None,
                    record_format: None,
                    record_length: None,
                    block_size: None,
                    data_class: None,
                    management_class: None,
                    storage_class: None,
                    space_primary: None,
                    space_secondary: None,
                    space_unit: None,
                    directory_blocks: None,
                    dataset_type: None,
                    like_dataset: None,
                })
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                self.visit_str(&value)
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut dataset: Option<String> = None;
                let mut member: Option<String> = None;
                let mut disposition: Option<String> = None;
                let mut volume: Option<String> = None;
                let mut record_format: Option<String> = None;
                let mut record_length: Option<u32> = None;
                let mut block_size: Option<u32> = None;
                let mut data_class: Option<String> = None;
                let mut management_class: Option<String> = None;
                let mut storage_class: Option<String> = None;
                let mut space_primary: Option<u32> = None;
                let mut space_secondary: Option<u32> = None;
                let mut space_unit: Option<String> = None;
                let mut directory_blocks: Option<u32> = None;
                let mut dataset_type: Option<String> = None;
                let mut like_dataset: Option<String> = None;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "dataset" => {
                            if dataset.is_some() {
                                return Err(de::Error::duplicate_field("dataset"));
                            }
                            dataset = Some(map.next_value()?);
                        }
                        "member" => {
                            member = map.next_value()?;
                        }
                        "disposition" => {
                            disposition = map.next_value()?;
                        }
                        "volume" => {
                            volume = map.next_value()?;
                        }
                        "record_format" => {
                            record_format = map.next_value()?;
                        }
                        "record_length" => {
                            record_length = map.next_value()?;
                        }
                        "block_size" => {
                            block_size = map.next_value()?;
                        }
                        "data_class" => {
                            data_class = map.next_value()?;
                        }
                        "management_class" => {
                            management_class = map.next_value()?;
                        }
                        "storage_class" => {
                            storage_class = map.next_value()?;
                        }
                        "space_primary" => {
                            space_primary = map.next_value()?;
                        }
                        "space_secondary" => {
                            space_secondary = map.next_value()?;
                        }
                        "space_unit" => {
                            space_unit = map.next_value()?;
                        }
                        "directory_blocks" => {
                            directory_blocks = map.next_value()?;
                        }
                        "dataset_type" => {
                            dataset_type = map.next_value()?;
                        }
                        "like_dataset" => {
                            like_dataset = map.next_value()?;
                        }
                        other => {
                            return Err(de::Error::unknown_field(
                                other,
                                &[
                                    "dataset",
                                    "member",
                                    "disposition",
                                    "volume",
                                    "record_format",
                                    "record_length",
                                    "block_size",
                                    "data_class",
                                    "management_class",
                                    "storage_class",
                                    "space_primary",
                                    "space_secondary",
                                    "space_unit",
                                    "directory_blocks",
                                    "dataset_type",
                                    "like_dataset",
                                ],
                            ));
                        }
                    }
                }

                let dataset = dataset.ok_or_else(|| de::Error::missing_field("dataset"))?;
                Ok(CobolDatasetRoute {
                    dataset,
                    member,
                    disposition,
                    volume,
                    record_format,
                    record_length,
                    block_size,
                    data_class,
                    management_class,
                    storage_class,
                    space_primary,
                    space_secondary,
                    space_unit,
                    directory_blocks,
                    dataset_type,
                    like_dataset,
                })
            }
        }

        deserializer.deserialize_any(DatasetVisitor)
    }
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
        match dataset.and_then(sanitize) {
            Some(name) => {
                let mut route = self.envelope.route.dataset.take().unwrap_or_default();
                route.dataset = name;
                self.envelope.route.dataset = Some(route);
            }
            None => {
                self.envelope.route.dataset = None;
            }
        }
        sanitize_dataset_route(&mut self.envelope.route.dataset);
    }

    pub fn set_dataset_member(&mut self, member: Option<String>) {
        let member = member.and_then(sanitize);
        match (self.envelope.route.dataset.as_mut(), member) {
            (Some(route), value) => {
                route.member = value;
            }
            (None, Some(value)) => {
                let mut route = CobolDatasetRoute::default();
                route.member = Some(value);
                self.envelope.route.dataset = Some(route);
            }
            (None, None) => {}
        }
        sanitize_dataset_route(&mut self.envelope.route.dataset);
    }

    pub fn set_dataset_disposition(&mut self, disposition: Option<String>) {
        let disposition = disposition.and_then(sanitize);
        match (self.envelope.route.dataset.as_mut(), disposition) {
            (Some(route), value) => {
                route.disposition = value;
            }
            (None, Some(value)) => {
                let mut route = CobolDatasetRoute::default();
                route.disposition = Some(value);
                self.envelope.route.dataset = Some(route);
            }
            (None, None) => {}
        }
        sanitize_dataset_route(&mut self.envelope.route.dataset);
    }

    pub fn set_dataset_volume(&mut self, volume: Option<String>) {
        let volume = volume.and_then(sanitize);
        match (self.envelope.route.dataset.as_mut(), volume) {
            (Some(route), value) => {
                route.volume = value;
            }
            (None, Some(value)) => {
                let mut route = CobolDatasetRoute::default();
                route.volume = Some(value);
                self.envelope.route.dataset = Some(route);
            }
            (None, None) => {}
        }
        sanitize_dataset_route(&mut self.envelope.route.dataset);
    }

    pub fn set_dataset_record_format(&mut self, record_format: Option<String>) {
        let record_format = record_format.and_then(sanitize);
        match (self.envelope.route.dataset.as_mut(), record_format) {
            (Some(route), value) => {
                route.record_format = value;
            }
            (None, Some(value)) => {
                let mut route = CobolDatasetRoute::default();
                route.record_format = Some(value);
                self.envelope.route.dataset = Some(route);
            }
            (None, None) => {}
        }
        sanitize_dataset_route(&mut self.envelope.route.dataset);
    }

    pub fn set_dataset_record_length(&mut self, record_length: Option<u32>) {
        let record_length = sanitize_positive(record_length);
        match (self.envelope.route.dataset.as_mut(), record_length) {
            (Some(route), value) => {
                route.record_length = value;
            }
            (None, Some(value)) => {
                let mut route = CobolDatasetRoute::default();
                route.record_length = Some(value);
                self.envelope.route.dataset = Some(route);
            }
            (None, None) => {}
        }
        sanitize_dataset_route(&mut self.envelope.route.dataset);
    }

    pub fn set_dataset_block_size(&mut self, block_size: Option<u32>) {
        let block_size = sanitize_positive(block_size);
        match (self.envelope.route.dataset.as_mut(), block_size) {
            (Some(route), value) => {
                route.block_size = value;
            }
            (None, Some(value)) => {
                let mut route = CobolDatasetRoute::default();
                route.block_size = Some(value);
                self.envelope.route.dataset = Some(route);
            }
            (None, None) => {}
        }
        sanitize_dataset_route(&mut self.envelope.route.dataset);
    }

    pub fn set_dataset_data_class(&mut self, data_class: Option<String>) {
        let data_class = data_class.and_then(sanitize);
        match (self.envelope.route.dataset.as_mut(), data_class) {
            (Some(route), value) => {
                route.data_class = value;
            }
            (None, Some(value)) => {
                let mut route = CobolDatasetRoute::default();
                route.data_class = Some(value);
                self.envelope.route.dataset = Some(route);
            }
            (None, None) => {}
        }
        sanitize_dataset_route(&mut self.envelope.route.dataset);
    }

    pub fn set_dataset_management_class(&mut self, management_class: Option<String>) {
        let management_class = management_class.and_then(sanitize);
        match (self.envelope.route.dataset.as_mut(), management_class) {
            (Some(route), value) => {
                route.management_class = value;
            }
            (None, Some(value)) => {
                let mut route = CobolDatasetRoute::default();
                route.management_class = Some(value);
                self.envelope.route.dataset = Some(route);
            }
            (None, None) => {}
        }
        sanitize_dataset_route(&mut self.envelope.route.dataset);
    }

    pub fn set_dataset_storage_class(&mut self, storage_class: Option<String>) {
        let storage_class = storage_class.and_then(sanitize);
        match (self.envelope.route.dataset.as_mut(), storage_class) {
            (Some(route), value) => {
                route.storage_class = value;
            }
            (None, Some(value)) => {
                let mut route = CobolDatasetRoute::default();
                route.storage_class = Some(value);
                self.envelope.route.dataset = Some(route);
            }
            (None, None) => {}
        }
        sanitize_dataset_route(&mut self.envelope.route.dataset);
    }

    pub fn set_dataset_space_primary(&mut self, space_primary: Option<u32>) {
        let space_primary = sanitize_positive(space_primary);
        match (self.envelope.route.dataset.as_mut(), space_primary) {
            (Some(route), value) => {
                route.space_primary = value;
            }
            (None, Some(value)) => {
                let mut route = CobolDatasetRoute::default();
                route.space_primary = Some(value);
                self.envelope.route.dataset = Some(route);
            }
            (None, None) => {}
        }
        sanitize_dataset_route(&mut self.envelope.route.dataset);
    }

    pub fn set_dataset_space_secondary(&mut self, space_secondary: Option<u32>) {
        let space_secondary = sanitize_positive(space_secondary);
        match (self.envelope.route.dataset.as_mut(), space_secondary) {
            (Some(route), value) => {
                route.space_secondary = value;
            }
            (None, Some(value)) => {
                let mut route = CobolDatasetRoute::default();
                route.space_secondary = Some(value);
                self.envelope.route.dataset = Some(route);
            }
            (None, None) => {}
        }
        sanitize_dataset_route(&mut self.envelope.route.dataset);
    }

    pub fn set_dataset_space_unit(&mut self, space_unit: Option<String>) {
        let space_unit = sanitize_uppercase(space_unit);
        match (self.envelope.route.dataset.as_mut(), space_unit) {
            (Some(route), value) => {
                route.space_unit = value;
            }
            (None, Some(value)) => {
                let mut route = CobolDatasetRoute::default();
                route.space_unit = Some(value);
                self.envelope.route.dataset = Some(route);
            }
            (None, None) => {}
        }
        sanitize_dataset_route(&mut self.envelope.route.dataset);
    }

    pub fn set_dataset_directory_blocks(&mut self, directory_blocks: Option<u32>) {
        let directory_blocks = sanitize_positive(directory_blocks);
        match (self.envelope.route.dataset.as_mut(), directory_blocks) {
            (Some(route), value) => {
                route.directory_blocks = value;
            }
            (None, Some(value)) => {
                let mut route = CobolDatasetRoute::default();
                route.directory_blocks = Some(value);
                self.envelope.route.dataset = Some(route);
            }
            (None, None) => {}
        }
        sanitize_dataset_route(&mut self.envelope.route.dataset);
    }

    pub fn set_dataset_type(&mut self, dataset_type: Option<String>) {
        let dataset_type = sanitize_uppercase(dataset_type);
        match (self.envelope.route.dataset.as_mut(), dataset_type) {
            (Some(route), value) => {
                route.dataset_type = value;
            }
            (None, Some(value)) => {
                let mut route = CobolDatasetRoute::default();
                route.dataset_type = Some(value);
                self.envelope.route.dataset = Some(route);
            }
            (None, None) => {}
        }
        sanitize_dataset_route(&mut self.envelope.route.dataset);
    }

    pub fn set_dataset_like(&mut self, like_dataset: Option<String>) {
        let like_dataset = like_dataset.and_then(sanitize);
        match (self.envelope.route.dataset.as_mut(), like_dataset) {
            (Some(route), value) => {
                route.like_dataset = value;
            }
            (None, Some(value)) => {
                let mut route = CobolDatasetRoute::default();
                route.like_dataset = Some(value);
                self.envelope.route.dataset = Some(route);
            }
            (None, None) => {}
        }
        sanitize_dataset_route(&mut self.envelope.route.dataset);
    }

    pub fn clear_route(&mut self) {
        self.clear_mq_route();
        self.clear_cics_route();
        self.clear_dataset();
    }

    pub fn clear_dataset(&mut self) {
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

        let dataset_configured = self
            .route
            .dataset
            .as_ref()
            .map(|route| !route.dataset.is_empty())
            .unwrap_or(false);
        if self.route.mq.is_none() && self.route.cics.is_none() && !dataset_configured {
            issues.push("configure a delivery route (MQ, CICS, or dataset)".to_string());
        }

        if let Some(dataset) = &self.route.dataset {
            if let (Some(record_length), Some(block_size)) =
                (dataset.record_length, dataset.block_size)
            {
                if block_size % record_length != 0 {
                    issues.push(
                        "dataset block size must be a multiple of the record length".to_string(),
                    );
                }
            }
            if let Some(unit) = dataset.space_unit.as_deref() {
                if !ALLOWED_SPACE_UNITS.contains(&unit) {
                    issues.push(format!(
                        "dataset space unit must be one of {}",
                        ALLOWED_SPACE_UNITS.join(", ")
                    ));
                }
                if dataset.space_primary.is_none() && dataset.space_secondary.is_none() {
                    issues.push(
                        "specify primary or secondary space when providing a space unit"
                            .to_string(),
                    );
                }
            }
            if dataset.space_secondary.is_some() && dataset.space_primary.is_none() {
                issues.push("dataset secondary space requires a primary allocation".to_string());
            }
            if let Some(dataset_type) = dataset.dataset_type.as_deref() {
                if !ALLOWED_DATASET_TYPES.contains(&dataset_type) {
                    issues.push(format!(
                        "dataset type must be one of {}",
                        ALLOWED_DATASET_TYPES.join(", ")
                    ));
                }
            }
            if let Some(directory_blocks) = dataset.directory_blocks {
                if directory_blocks > 0
                    && dataset.member.is_none()
                    && !matches!(dataset.dataset_type.as_deref(), Some("LIBRARY" | "PDS"))
                {
                    issues.push(
                        "directory blocks are only valid for partitioned dataset allocations"
                            .to_string(),
                    );
                }
            }
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

fn sanitize_positive(value: Option<u32>) -> Option<u32> {
    value.and_then(|candidate| if candidate > 0 { Some(candidate) } else { None })
}

fn sanitize_uppercase(value: Option<String>) -> Option<String> {
    value.and_then(|candidate| sanitize(candidate).map(|clean| clean.to_ascii_uppercase()))
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

    sanitize_dataset_route(&mut envelope.route.dataset);

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

fn sanitize_dataset_route(target: &mut Option<CobolDatasetRoute>) {
    if let Some(mut dataset) = target.take() {
        let dataset_name = sanitize(dataset.dataset);
        dataset.member = dataset.member.take().and_then(sanitize);
        dataset.disposition = dataset.disposition.take().and_then(sanitize);
        dataset.volume = dataset.volume.take().and_then(sanitize);
        dataset.record_format = dataset.record_format.take().and_then(sanitize);
        dataset.record_length = sanitize_positive(dataset.record_length.take());
        dataset.block_size = sanitize_positive(dataset.block_size.take());
        dataset.data_class = dataset.data_class.take().and_then(sanitize);
        dataset.management_class = dataset.management_class.take().and_then(sanitize);
        dataset.storage_class = dataset.storage_class.take().and_then(sanitize);
        dataset.space_primary = sanitize_positive(dataset.space_primary.take());
        dataset.space_secondary = sanitize_positive(dataset.space_secondary.take());
        dataset.space_unit = sanitize_uppercase(dataset.space_unit.take());
        dataset.directory_blocks = sanitize_positive(dataset.directory_blocks.take());
        dataset.dataset_type = sanitize_uppercase(dataset.dataset_type.take());
        dataset.like_dataset = dataset.like_dataset.take().and_then(sanitize);
        if let Some(name) = dataset_name {
            dataset.dataset = name;
            *target = Some(dataset);
        }
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
        builder.set_dataset_member(Some("PAYLOAD".into()));
        builder.set_dataset_disposition(Some("OLD".into()));
        builder.set_dataset_volume(Some("VOL001".into()));
        builder.set_dataset_record_format(Some("FB".into()));
        builder.set_dataset_record_length(Some(512));
        builder.set_dataset_block_size(Some(4096));
        builder.set_dataset_data_class(Some("NARRATE".into()));
        builder.set_dataset_management_class(Some("GDG".into()));
        builder.set_dataset_storage_class(Some("FASTIO".into()));
        builder.set_dataset_space_primary(Some(15));
        builder.set_dataset_space_secondary(Some(5));
        builder.set_dataset_space_unit(Some("cyl".into()));
        builder.set_dataset_directory_blocks(Some(40));
        builder.set_dataset_type(Some("library".into()));
        builder.set_dataset_like(Some("HLQ.MODEL.DATA".into()));
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
        let dataset_route = envelope.route.dataset.expect("dataset");
        assert_eq!(dataset_route.dataset, "HLQ.DATA");
        assert_eq!(dataset_route.member.as_deref(), Some("PAYLOAD"));
        assert_eq!(dataset_route.disposition.as_deref(), Some("OLD"));
        assert_eq!(dataset_route.volume.as_deref(), Some("VOL001"));
        assert_eq!(dataset_route.record_format.as_deref(), Some("FB"));
        assert_eq!(dataset_route.record_length, Some(512));
        assert_eq!(dataset_route.block_size, Some(4096));
        assert_eq!(dataset_route.data_class.as_deref(), Some("NARRATE"));
        assert_eq!(dataset_route.management_class.as_deref(), Some("GDG"));
        assert_eq!(dataset_route.storage_class.as_deref(), Some("FASTIO"));
        assert_eq!(dataset_route.space_primary, Some(15));
        assert_eq!(dataset_route.space_secondary, Some(5));
        assert_eq!(dataset_route.space_unit.as_deref(), Some("CYL"));
        assert_eq!(dataset_route.directory_blocks, Some(40));
        assert_eq!(dataset_route.dataset_type.as_deref(), Some("LIBRARY"));
        assert_eq!(
            dataset_route.like_dataset.as_deref(),
            Some("HLQ.MODEL.DATA")
        );
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
        builder.set_dataset_member(Some("   ".into()));
        builder.set_dataset_disposition(Some("   ".into()));
        builder.set_dataset_volume(Some("   ".into()));
        builder.add_tag("  ");
        builder.add_annotation(" ");
        builder.merge_metadata_value(Value::Null);
        assert_eq!(builder.snapshot().release_channel, "production");
        assert!(builder.snapshot().route.dataset.is_none());
        assert!(builder.snapshot().metadata.tags.is_empty());
    }

    #[test]
    fn dataset_metadata_survives_partial_updates() {
        let mut builder = CobolEnvelopeBuilder::new("job-dataset");
        builder.set_dataset(Some(" HLQ.DATA ".into()));
        builder.set_dataset_member(Some(" MEMBER ".into()));
        builder.set_dataset_disposition(Some(" MOD ".into()));
        builder.set_dataset_volume(Some(" VOL003 ".into()));
        builder.set_dataset_record_format(Some(" FB ".into()));
        builder.set_dataset_record_length(Some(2048));
        builder.set_dataset_block_size(Some(8192));
        builder.set_dataset_data_class(Some(" PRIME ".into()));
        builder.set_dataset_management_class(Some(" GDG ".into()));
        builder.set_dataset_storage_class(Some(" FASTIO ".into()));
        builder.set_dataset_space_primary(Some(20));
        builder.set_dataset_space_secondary(Some(4));
        builder.set_dataset_space_unit(Some(" trk ".into()));
        builder.set_dataset_directory_blocks(Some(8));
        builder.set_dataset_type(Some(" pdS ".into()));
        builder.set_dataset_like(Some("  HLQ.TEMPLATE.DATA  ".into()));
        builder.set_dataset(Some("HLQ.DATA".into()));

        let dataset = builder.snapshot().route.dataset.expect("dataset");
        assert_eq!(dataset.dataset, "HLQ.DATA");
        assert_eq!(dataset.member.as_deref(), Some("MEMBER"));
        assert_eq!(dataset.disposition.as_deref(), Some("MOD"));
        assert_eq!(dataset.volume.as_deref(), Some("VOL003"));
        assert_eq!(dataset.record_format.as_deref(), Some("FB"));
        assert_eq!(dataset.record_length, Some(2048));
        assert_eq!(dataset.block_size, Some(8192));
        assert_eq!(dataset.data_class.as_deref(), Some("PRIME"));
        assert_eq!(dataset.management_class.as_deref(), Some("GDG"));
        assert_eq!(dataset.storage_class.as_deref(), Some("FASTIO"));
        assert_eq!(dataset.space_primary, Some(20));
        assert_eq!(dataset.space_secondary, Some(4));
        assert_eq!(dataset.space_unit.as_deref(), Some("TRK"));
        assert_eq!(dataset.directory_blocks, Some(8));
        assert_eq!(dataset.dataset_type.as_deref(), Some("PDS"));
        assert_eq!(dataset.like_dataset.as_deref(), Some("HLQ.TEMPLATE.DATA"));
    }

    #[test]
    fn dataset_fields_can_be_cleared_individually() {
        let mut builder = CobolEnvelopeBuilder::new("job-dataset-clear");
        builder.set_dataset(Some("HLQ.DATA".into()));
        builder.set_dataset_member(Some("PAYLOAD".into()));
        builder.set_dataset_disposition(Some("SHR".into()));
        builder.set_dataset_volume(Some("VOL001".into()));
        builder.set_dataset_record_format(Some("FB".into()));
        builder.set_dataset_record_length(Some(1024));
        builder.set_dataset_block_size(Some(4096));
        builder.set_dataset_data_class(Some("NARR".into()));
        builder.set_dataset_management_class(Some("GDG".into()));
        builder.set_dataset_storage_class(Some("FAST".into()));
        builder.set_dataset_space_primary(Some(12));
        builder.set_dataset_space_secondary(Some(4));
        builder.set_dataset_space_unit(Some("cyl".into()));
        builder.set_dataset_directory_blocks(Some(24));
        builder.set_dataset_type(Some("library".into()));
        builder.set_dataset_like(Some("HLQ.TEMPLATE.DATA".into()));

        builder.set_dataset_member(None);
        builder.set_dataset_disposition(None);
        builder.set_dataset_volume(None);
        builder.set_dataset_record_format(None);
        builder.set_dataset_record_length(None);
        builder.set_dataset_block_size(None);
        builder.set_dataset_data_class(None);
        builder.set_dataset_management_class(None);
        builder.set_dataset_storage_class(None);
        builder.set_dataset_space_primary(None);
        builder.set_dataset_space_secondary(None);
        builder.set_dataset_space_unit(None);
        builder.set_dataset_directory_blocks(None);
        builder.set_dataset_type(None);
        builder.set_dataset_like(None);

        let dataset = builder.snapshot().route.dataset.expect("dataset");
        assert_eq!(dataset.dataset, "HLQ.DATA");
        assert!(dataset.member.is_none());
        assert!(dataset.disposition.is_none());
        assert!(dataset.volume.is_none());
        assert!(dataset.record_format.is_none());
        assert!(dataset.record_length.is_none());
        assert!(dataset.block_size.is_none());
        assert!(dataset.data_class.is_none());
        assert!(dataset.management_class.is_none());
        assert!(dataset.storage_class.is_none());
        assert!(dataset.space_primary.is_none());
        assert!(dataset.space_secondary.is_none());
        assert!(dataset.space_unit.is_none());
        assert!(dataset.directory_blocks.is_none());
        assert!(dataset.dataset_type.is_none());
        assert!(dataset.like_dataset.is_none());

        builder.set_dataset(None);
        assert!(builder.snapshot().route.dataset.is_none());
    }

    #[test]
    fn dataset_numeric_fields_ignore_non_positive_values() {
        let mut builder = CobolEnvelopeBuilder::new("job-dataset-numeric");
        builder.set_dataset(Some("HLQ.DATA".into()));
        builder.set_dataset_record_length(Some(0));
        builder.set_dataset_block_size(Some(0));
        assert!(builder
            .snapshot()
            .route
            .dataset
            .expect("dataset")
            .record_length
            .is_none());
        assert!(builder
            .snapshot()
            .route
            .dataset
            .expect("dataset")
            .block_size
            .is_none());

        builder.set_dataset_record_length(Some(256));
        builder.set_dataset_block_size(Some(1024));
        let dataset = builder.snapshot().route.dataset.expect("dataset");
        assert_eq!(dataset.record_length, Some(256));
        assert_eq!(dataset.block_size, Some(1024));

        builder.set_dataset_record_length(Some(0));
        builder.set_dataset_block_size(Some(0));
        let cleared = builder.snapshot().route.dataset.expect("dataset");
        assert!(cleared.record_length.is_none());
        assert!(cleared.block_size.is_none());
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
        builder.set_dataset_member(Some("COEFFS".into()));

        builder.clear_initiators();
        builder.clear_mq_route();
        builder.clear_cics_route();
        builder.clear_dataset();

        let envelope = builder.snapshot();
        assert!(envelope.initiators.is_empty());
        assert!(envelope.route.mq.is_none());
        assert!(envelope.route.cics.is_none());
        assert!(envelope.route.dataset.is_none());

        builder.set_mq_route("QM2", "QUEUE", None);
        builder.set_cics_route("TRN1", None, None);
        builder.set_dataset(Some("HLQ.DATA".into()));
        builder.set_dataset_member(Some("COEFFS".into()));
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
                dataset: Some(CobolDatasetRoute {
                    dataset: "  DATA.SET  ".to_string(),
                    member: Some(" MEMBER  ".to_string()),
                    disposition: Some(" SHR ".to_string()),
                    volume: None,
                    record_format: Some(" FB ".to_string()),
                    record_length: Some(2048),
                    block_size: Some(8192),
                    data_class: Some(" NARR ".to_string()),
                    management_class: Some(" GDG ".to_string()),
                    storage_class: Some(" FASTIO ".to_string()),
                    space_primary: Some(12),
                    space_secondary: Some(4),
                    space_unit: Some(" cyl ".to_string()),
                    directory_blocks: Some(30),
                    dataset_type: Some(" library ".to_string()),
                    like_dataset: Some("  TEMPLATE.DATA  ".to_string()),
                }),
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
                .filter(|value| value.as_str() == "planner_initialized")
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
        let dataset = snapshot.route.dataset.expect("dataset");
        assert_eq!(dataset.dataset, "DATA.SET");
        assert_eq!(dataset.member.as_deref(), Some("MEMBER"));
        assert_eq!(dataset.disposition.as_deref(), Some("SHR"));
        assert_eq!(dataset.record_format.as_deref(), Some("FB"));
        assert_eq!(dataset.record_length, Some(2048));
        assert_eq!(dataset.block_size, Some(8192));
        assert_eq!(dataset.data_class.as_deref(), Some("NARR"));
        assert_eq!(dataset.management_class.as_deref(), Some("GDG"));
        assert_eq!(dataset.storage_class.as_deref(), Some("FASTIO"));
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
    fn validation_catches_inconsistent_dataset_block_size() {
        let mut builder = CobolEnvelopeBuilder::new("job-dataset-validate");
        builder.add_initiator(make_initiator(
            InitiatorKind::Automation,
            "validator",
            None,
            None,
            None,
            None,
        ));
        builder.set_dataset(Some("HLQ.DATA".into()));
        builder.set_dataset_record_length(Some(256));
        builder.set_dataset_block_size(Some(300));

        let issues = builder.validation_issues();
        assert_eq!(issues.len(), 1);
        assert!(issues
            .contains(&"dataset block size must be a multiple of the record length".to_string()));
    }

    #[test]
    fn validation_flags_dataset_space_rules() {
        let mut builder = CobolEnvelopeBuilder::new("job-dataset-space");
        builder.add_initiator(make_initiator(
            InitiatorKind::Automation,
            "allocator",
            None,
            None,
            None,
            None,
        ));
        builder.set_dataset(Some("HLQ.DATA".into()));

        builder.set_dataset_space_unit(Some("trk".into()));
        let issues = builder.validation_issues();
        assert!(issues
            .iter()
            .any(|issue| issue.contains("specify primary or secondary space")));

        builder.set_dataset_space_secondary(Some(3));
        let issues = builder.validation_issues();
        assert!(issues
            .iter()
            .any(|issue| issue.contains("dataset secondary space requires a primary")));

        builder.set_dataset_space_primary(Some(10));
        builder.set_dataset_space_unit(Some("blocks".into()));
        let issues = builder.validation_issues();
        assert!(issues
            .iter()
            .any(|issue| issue.contains("dataset space unit must be one of")));

        builder.set_dataset_space_unit(Some("cyl".into()));
        builder.set_dataset_directory_blocks(Some(12));
        let issues = builder.validation_issues();
        assert!(issues
            .iter()
            .any(|issue| issue.contains("directory blocks are only valid")));

        builder.set_dataset_member(Some("PAYLOAD".into()));
        builder.set_dataset_type(Some("unsupported".into()));
        let issues = builder.validation_issues();
        assert!(issues
            .iter()
            .any(|issue| issue.contains("dataset type must be one of")));
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

    #[test]
    fn dataset_route_serialization_supports_string_and_object() {
        let basic = CobolDatasetRoute {
            dataset: "HLQ.DATA".to_string(),
            member: None,
            disposition: None,
            volume: None,
            record_format: None,
            record_length: None,
            block_size: None,
            data_class: None,
            management_class: None,
            storage_class: None,
            space_primary: None,
            space_secondary: None,
            space_unit: None,
            directory_blocks: None,
            dataset_type: None,
            like_dataset: None,
        };
        let serialized = serde_json::to_string(&basic).expect("serialize");
        assert_eq!(serialized, "\"HLQ.DATA\"");

        let mut enriched = basic.clone();
        enriched.member = Some("MEMBER".into());
        enriched.disposition = Some("SHR".into());
        enriched.record_format = Some("FB".into());
        enriched.record_length = Some(512);
        enriched.block_size = Some(4096);
        enriched.space_primary = Some(15);
        enriched.space_secondary = Some(5);
        enriched.space_unit = Some("CYL".into());
        enriched.directory_blocks = Some(30);
        enriched.dataset_type = Some("LIBRARY".into());
        enriched.like_dataset = Some("HLQ.TEMPLATE".into());
        let serialized_enriched = serde_json::to_string(&enriched).expect("serialize enriched");
        assert!(serialized_enriched.contains("\"dataset\":"));
        assert!(serialized_enriched.contains("\"member\""));
        assert!(serialized_enriched.contains("\"disposition\""));
        assert!(serialized_enriched.contains("\"record_length\""));
        assert!(serialized_enriched.contains("\"space_primary\""));
        assert!(serialized_enriched.contains("\"dataset_type\""));

        let parsed_basic: CobolDatasetRoute = serde_json::from_str("\"USER.DATA\"").expect("parse");
        assert_eq!(parsed_basic.dataset, "USER.DATA");
        assert!(parsed_basic.member.is_none());

        let parsed_enriched: CobolDatasetRoute = serde_json::from_str(
            "{\"dataset\":\"USER.DATA\",\"member\":\"MEMBER\",\"disposition\":\"SHR\",\"volume\":\"VOL001\",\"record_format\":\"FB\",\"record_length\":256,\"block_size\":4096,\"space_primary\":20,\"space_unit\":\"CYL\",\"directory_blocks\":12,\"dataset_type\":\"LIBRARY\",\"like_dataset\":\"USER.MODEL\"}",
        )
        .expect("parse object");
        assert_eq!(parsed_enriched.dataset, "USER.DATA");
        assert_eq!(parsed_enriched.member.as_deref(), Some("MEMBER"));
        assert_eq!(parsed_enriched.disposition.as_deref(), Some("SHR"));
        assert_eq!(parsed_enriched.volume.as_deref(), Some("VOL001"));
        assert_eq!(parsed_enriched.record_format.as_deref(), Some("FB"));
        assert_eq!(parsed_enriched.record_length, Some(256));
        assert_eq!(parsed_enriched.block_size, Some(4096));
        assert_eq!(parsed_enriched.space_primary, Some(20));
        assert!(parsed_enriched.space_secondary.is_none());
        assert_eq!(parsed_enriched.space_unit.as_deref(), Some("CYL"));
        assert_eq!(parsed_enriched.directory_blocks, Some(12));
        assert_eq!(parsed_enriched.dataset_type.as_deref(), Some("LIBRARY"));
        assert_eq!(parsed_enriched.like_dataset.as_deref(), Some("USER.MODEL"));
    }
}
