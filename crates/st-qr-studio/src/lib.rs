// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 SpiralTorch Contributors
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Quantum Reality Studio primitives that stitch Maxwell-coded pulses,
//! narrative tags, and immersive overlays together.

use serde::{Deserialize, Serialize};
use st_core::maxwell::MaxwellZPulse;
use st_nn::{ConceptHint, MaxwellDesireBridge, NarrativeHint};
use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

const DEFAULT_HISTORY: usize = 512;

#[derive(Debug, Error)]
pub enum StudioError {
    #[error("channel `{0}` is not registered in the capture config")]
    UnknownChannel(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SignalCaptureConfig {
    pub sample_rate_hz: f32,
    pub channels: Vec<String>,
    pub history_limit: usize,
}

impl SignalCaptureConfig {
    pub fn new(sample_rate_hz: f32) -> Self {
        Self {
            sample_rate_hz: sample_rate_hz.max(1.0),
            channels: Vec::new(),
            history_limit: DEFAULT_HISTORY,
        }
    }

    pub fn with_channels(mut self, channels: Vec<String>) -> Self {
        self.channels = channels;
        self
    }

    pub fn with_history_limit(mut self, limit: usize) -> Self {
        self.history_limit = limit.max(1);
        self
    }

    fn allows(&self, channel: &str) -> bool {
        self.channels.is_empty() || self.channels.iter().any(|c| c == channel)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PulseSnapshot {
    pub blocks: u64,
    pub mean: f64,
    pub standard_error: f64,
    pub z_score: f64,
    pub band_energy: (f32, f32, f32),
    pub z_bias: f32,
}

impl PulseSnapshot {
    pub fn magnitude(&self) -> f32 {
        self.z_score.abs() as f32
    }
}

impl From<&MaxwellZPulse> for PulseSnapshot {
    fn from(pulse: &MaxwellZPulse) -> Self {
        Self {
            blocks: pulse.blocks,
            mean: pulse.mean,
            standard_error: pulse.standard_error,
            z_score: pulse.z_score,
            band_energy: pulse.band_energy,
            z_bias: pulse.z_bias,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecordedPulse {
    pub channel: String,
    pub pulse: PulseSnapshot,
    pub timestamp: SystemTime,
}

impl RecordedPulse {
    pub fn new(channel: impl Into<String>, pulse: PulseSnapshot, timestamp: SystemTime) -> Self {
        Self {
            channel: channel.into(),
            pulse,
            timestamp,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum StudioConceptHint {
    Distribution(Vec<f32>),
    Window(Vec<(usize, f32)>),
}

impl StudioConceptHint {
    fn from_concept(hint: ConceptHint) -> Self {
        match hint {
            ConceptHint::Distribution(dist) => StudioConceptHint::Distribution(dist),
            ConceptHint::Window(window) => StudioConceptHint::Window(window),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OverlayFrame {
    pub channel: String,
    pub glyph: String,
    pub intensity: f32,
    pub timestamp: SystemTime,
}

impl OverlayFrame {
    fn new(record: &RecordedPulse, glyph: String, intensity: f32) -> Self {
        Self {
            channel: record.channel.clone(),
            glyph,
            intensity,
            timestamp: record.timestamp,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StudioFrame {
    pub record: RecordedPulse,
    pub concept: Option<StudioConceptHint>,
    pub narrative: Option<NarrativeHint>,
    pub overlay: OverlayFrame,
}

#[derive(Clone, Debug)]
pub struct SignalCaptureSession {
    config: SignalCaptureConfig,
    history: VecDeque<RecordedPulse>,
}

impl SignalCaptureSession {
    pub fn new(config: SignalCaptureConfig) -> Self {
        Self {
            history: VecDeque::with_capacity(config.history_limit),
            config,
        }
    }

    pub fn ingest(
        &mut self,
        channel: &str,
        pulse: MaxwellZPulse,
        timestamp: SystemTime,
    ) -> Result<RecordedPulse, StudioError> {
        if !self.config.allows(channel) {
            return Err(StudioError::UnknownChannel(channel.to_string()));
        }
        let record =
            RecordedPulse::new(channel.to_string(), PulseSnapshot::from(&pulse), timestamp);
        self.history.push_back(record.clone());
        while self.history.len() > self.config.history_limit {
            self.history.pop_front();
        }
        Ok(record)
    }

    pub fn records(&self) -> impl Iterator<Item = &RecordedPulse> {
        self.history.iter()
    }
}

#[derive(Clone, Debug)]
pub struct SemanticTagger {
    fallback_tags: HashMap<String, Vec<String>>,
}

impl SemanticTagger {
    pub fn from_bridge(bridge: &MaxwellDesireBridge) -> Self {
        let mut fallback_tags = HashMap::new();
        for channel in bridge.channel_names() {
            if let Some(tags) = bridge.narrative_tags(&channel) {
                fallback_tags.insert(channel.clone(), tags.to_vec());
            }
        }
        Self { fallback_tags }
    }

    pub fn fallback_for(&self, channel: &str) -> Option<&[String]> {
        self.fallback_tags.get(channel).map(|tags| tags.as_slice())
    }
}

#[derive(Clone, Debug, Default)]
pub struct OverlayComposer;

impl OverlayComposer {
    pub fn compose(
        &self,
        record: &RecordedPulse,
        narrative: Option<&NarrativeHint>,
        fallback: Option<&[String]>,
    ) -> OverlayFrame {
        let glyph = narrative
            .and_then(|hint| hint.tags().first().cloned())
            .or_else(|| fallback.and_then(|tags| tags.first().cloned()))
            .unwrap_or_else(|| record.channel.clone());
        let intensity = narrative
            .map(|hint| hint.intensity())
            .unwrap_or_else(|| record.pulse.magnitude());
        OverlayFrame::new(record, glyph, intensity)
    }
}

pub struct QuantumRealityStudio {
    capture: SignalCaptureSession,
    tagger: SemanticTagger,
    overlay: OverlayComposer,
}

impl QuantumRealityStudio {
    pub fn new(config: SignalCaptureConfig, bridge: &MaxwellDesireBridge) -> Self {
        Self {
            capture: SignalCaptureSession::new(config),
            tagger: SemanticTagger::from_bridge(bridge),
            overlay: OverlayComposer::default(),
        }
    }

    pub fn ingest(
        &mut self,
        bridge: &MaxwellDesireBridge,
        channel: impl AsRef<str>,
        pulse: MaxwellZPulse,
        timestamp: Option<SystemTime>,
    ) -> Result<StudioFrame, StudioError> {
        let channel = channel.as_ref();
        let emission = bridge.emit(channel, &pulse);
        let (concept, narrative) = match emission {
            Some((concept, narrative)) => {
                (Some(StudioConceptHint::from_concept(concept)), narrative)
            }
            None => (None, None),
        };
        let record =
            self.capture
                .ingest(channel, pulse, timestamp.unwrap_or_else(SystemTime::now))?;
        let overlay = self.overlay.compose(
            &record,
            narrative.as_ref(),
            self.tagger.fallback_for(channel),
        );
        Ok(StudioFrame {
            record,
            concept,
            narrative,
            overlay,
        })
    }

    pub fn records(&self) -> impl Iterator<Item = &RecordedPulse> {
        self.capture.records()
    }

    pub fn export_storyboard(&self) -> serde_json::Value {
        let frames: Vec<serde_json::Value> = self
            .records()
            .enumerate()
            .map(|(ordinal, record)| {
                let ts = record
                    .timestamp
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64();
                serde_json::json!({
                    "ordinal": ordinal,
                    "channel": record.channel,
                    "timestamp": ts,
                    "z_score": record.pulse.z_score,
                    "band_energy": record.pulse.band_energy,
                    "z_bias": record.pulse.z_bias,
                })
            })
            .collect();
        serde_json::json!({ "frames": frames })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_pulse() -> MaxwellZPulse {
        MaxwellZPulse {
            blocks: 12,
            mean: 0.14,
            standard_error: 0.03,
            z_score: 4.2,
            band_energy: (0.5, 0.3, 0.2),
            z_bias: 0.27,
        }
    }

    #[test]
    fn capture_rejects_unknown_channels() {
        let mut session = SignalCaptureSession::new(
            SignalCaptureConfig::new(48000.0).with_channels(vec!["alpha".into()]),
        );
        let err = session
            .ingest("beta", sample_pulse(), SystemTime::now())
            .unwrap_err();
        matches!(err, StudioError::UnknownChannel(_));
    }

    #[test]
    fn overlay_prefers_narrative_tags() {
        let bridge = MaxwellDesireBridge::new()
            .with_channel_and_narrative(
                "alpha",
                vec![(0, 1.0)],
                vec!["glimmer".into(), "braid".into()],
            )
            .unwrap();
        let mut studio = QuantumRealityStudio::new(SignalCaptureConfig::new(48000.0), &bridge);
        let frame = studio
            .ingest(&bridge, "alpha", sample_pulse(), None)
            .expect("ingest");
        assert!(frame.concept.is_some());
        assert!(frame.narrative.is_some());
        assert_eq!(frame.overlay.glyph, "braid");
        assert!(frame.overlay.intensity > 0.0);
    }
}
