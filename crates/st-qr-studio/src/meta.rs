use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use st_logic::quantum_reality::ZSpace as LogicZSpace;

use crate::{ConceptWindow, RecordedPulse};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TemporalCausalAnnotation {
    pub event_id: usize,
    pub ordinal: usize,
    pub timestamp: f64,
    pub channel: String,
    pub magnitude: f32,
    pub depth: usize,
    pub parents: Vec<usize>,
    pub parent_channels: Vec<String>,
}

impl TemporalCausalAnnotation {
    pub fn is_root(&self) -> bool {
        self.parents.is_empty()
    }
}

#[derive(Clone, Debug, Default)]
pub struct TemporalLogicEngine {
    causal: CausalSet,
}

impl TemporalLogicEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn observe(
        &mut self,
        record: &RecordedPulse,
        concept: Option<&ConceptWindow>,
    ) -> TemporalCausalAnnotation {
        let weights = concept
            .map(|window| window.weights.clone())
            .unwrap_or_default();
        let magnitude = record.pulse.magnitude();
        let event =
            self.causal
                .insert(record.channel.clone(), record.timestamp, magnitude, weights);
        self.causal.annotation(&event)
    }

    pub fn snapshot(&self) -> Vec<TemporalCausalAnnotation> {
        self.causal
            .events()
            .map(|event| self.causal.annotation(event))
            .collect()
    }

    pub fn temporal_tags(annotation: &TemporalCausalAnnotation) -> Vec<String> {
        let mut tags = Vec::with_capacity(4 + annotation.parent_channels.len());
        tags.push(format!("ordinal:{}", annotation.ordinal));
        tags.push(format!("time:{:.3}", annotation.timestamp));
        tags.push(format!("magnitude:{:.3}", annotation.magnitude));
        tags.push(format!("depth:{}", annotation.depth));
        for channel in &annotation.parent_channels {
            tags.push(format!("after:{channel}"));
        }
        tags
    }
}

#[derive(Clone, Debug, Default)]
pub struct ToposLogicBridge {
    sections: HashMap<String, MeaningSheafSection>,
}

impl ToposLogicBridge {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(
        &mut self,
        record: &RecordedPulse,
        concept: Option<&ConceptWindow>,
        annotation: &TemporalCausalAnnotation,
    ) -> Vec<String> {
        let section = self
            .sections
            .entry(record.channel.clone())
            .or_insert_with(MeaningSheafSection::default);
        section.integrate(record, concept, annotation.depth);
        section.semantic_tags(annotation)
    }

    pub fn meaning_sheaf(&self, channel: &str) -> Option<LogicZSpace> {
        self.sections
            .get(channel)
            .map(|section| section.signature())
    }
}

#[derive(Clone, Debug, Default)]
struct MeaningSheafSection {
    total_weight: f32,
    tokens: HashMap<usize, f32>,
    signature: Vec<f64>,
}

impl MeaningSheafSection {
    fn integrate(&mut self, record: &RecordedPulse, concept: Option<&ConceptWindow>, depth: usize) {
        let signature = signature_from_record(record);
        if self.signature.is_empty() {
            self.signature = signature;
        } else {
            let factor = 1.0 / ((self.total_weight + 1.0) as f64);
            for (dst, src) in self.signature.iter_mut().zip(signature.iter()) {
                *dst = (*dst * (1.0 - factor)) + (*src * factor);
            }
        }

        let mut gain = 1.0 + depth as f32;
        if let Some(window) = concept {
            gain *= window.magnitude.max(1e-3);
            for &(token, weight) in &window.weights {
                *self.tokens.entry(token).or_default() += weight * gain;
            }
        }
        self.total_weight += gain;
    }

    fn semantic_tags(&self, annotation: &TemporalCausalAnnotation) -> Vec<String> {
        let mut tags = Vec::new();
        if !self.signature.is_empty() {
            let curvature = self.signature.get(0).copied().unwrap_or(0.0);
            tags.push(format!("sheaf|curvature:{curvature:+.3}"));
            if let Some(above) = self.signature.get(1) {
                tags.push(format!("sheaf|above:{above:.3}"));
            }
            if let Some(here) = self.signature.get(2) {
                tags.push(format!("sheaf|here:{here:.3}"));
            }
            if let Some(beneath) = self.signature.get(3) {
                tags.push(format!("sheaf|beneath:{beneath:.3}"));
            }
        }
        let mut entries: Vec<_> = self.tokens.iter().collect();
        entries.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(Ordering::Equal));
        let limit = entries.len().min(3);
        let total = self.total_weight.max(1e-6);
        for (token, weight) in entries.into_iter().take(limit) {
            let normalised = (*weight / total).clamp(0.0, 1.0);
            tags.push(format!("sheaf|token#{token}:{normalised:.2}"));
        }
        tags.push(format!("sheaf|depth:{}", annotation.depth));
        tags
    }

    fn signature(&self) -> LogicZSpace {
        LogicZSpace {
            signature: self.signature.clone(),
        }
    }
}

#[derive(Clone, Debug)]
struct CausalEvent {
    id: usize,
    channel: String,
    timestamp: f64,
    magnitude: f32,
    parents: Vec<usize>,
    depth: usize,
}

#[derive(Clone, Debug, Default)]
struct CausalSet {
    events: Vec<CausalEvent>,
    latest_by_channel: HashMap<String, usize>,
    latest_by_token: HashMap<usize, usize>,
}

impl CausalSet {
    fn insert(
        &mut self,
        channel: String,
        timestamp: SystemTime,
        magnitude: f32,
        weights: Vec<(usize, f32)>,
    ) -> CausalEvent {
        let id = self.events.len();
        let ts = timestamp_to_secs(timestamp);
        let mut parents = Vec::new();
        if let Some(prev) = self.latest_by_channel.get(&channel) {
            parents.push(*prev);
        }
        for (token, _) in &weights {
            if let Some(prev) = self.latest_by_token.get(token) {
                parents.push(*prev);
            }
        }
        parents.sort_unstable();
        parents.dedup();
        let depth = parents
            .iter()
            .map(|idx| self.events.get(*idx).map(|event| event.depth).unwrap_or(0))
            .max()
            .unwrap_or(0);
        let depth = depth + if parents.is_empty() { 0 } else { 1 };
        let event = CausalEvent {
            id,
            channel: channel.clone(),
            timestamp: ts,
            magnitude,
            parents,
            depth,
        };
        self.events.push(event.clone());
        self.latest_by_channel.insert(channel, id);
        for (token, _) in &weights {
            self.latest_by_token.insert(*token, id);
        }
        event
    }

    fn annotation(&self, event: &CausalEvent) -> TemporalCausalAnnotation {
        let mut parent_channels = Vec::with_capacity(event.parents.len());
        for parent_idx in &event.parents {
            if let Some(parent) = self.events.get(*parent_idx) {
                if parent.channel != event.channel {
                    parent_channels.push(parent.channel.clone());
                } else {
                    parent_channels.push(format!("{}#{}", parent.channel, parent.id));
                }
            }
        }
        TemporalCausalAnnotation {
            event_id: event.id,
            ordinal: event.id,
            timestamp: event.timestamp,
            channel: event.channel.clone(),
            magnitude: event.magnitude,
            depth: event.depth,
            parents: event.parents.clone(),
            parent_channels,
        }
    }

    fn events(&self) -> impl Iterator<Item = &CausalEvent> {
        self.events.iter()
    }
}

fn signature_from_record(record: &RecordedPulse) -> Vec<f64> {
    vec![
        record.pulse.z_score,
        record.pulse.band_energy.0 as f64,
        record.pulse.band_energy.1 as f64,
        record.pulse.band_energy.2 as f64,
        record.pulse.z_bias as f64,
        record.pulse.mean,
    ]
}

fn timestamp_to_secs(timestamp: SystemTime) -> f64 {
    match timestamp.duration_since(UNIX_EPOCH) {
        Ok(duration) => duration_as_secs(duration),
        Err(err) => {
            let duration = err.duration();
            -(duration_as_secs(duration))
        }
    }
}

fn duration_as_secs(duration: Duration) -> f64 {
    duration.as_secs() as f64 + f64::from(duration.subsec_nanos()) / 1_000_000_000.0
}
