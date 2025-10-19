// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 SpiralTorch Contributors
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Quantum Reality Studio primitives that stitch Maxwell-coded pulses,
//! narrative tags, and immersive overlays together.

use serde::{Deserialize, Serialize};
use st_core::maxwell::MaxwellZPulse;
use st_frac::mellin_types::ComplexScalar;
use st_frac::zspace::{
    evaluate_weighted_series, mellin_log_lattice_prefactor, prepare_weighted_series,
    trapezoidal_weights,
};
use st_nn::{ConceptHint, MaxwellDesireBridge, NarrativeHint};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

const DEFAULT_HISTORY: usize = 512;

#[derive(Debug, Error)]
pub enum StudioError {
    #[error("channel `{0}` is not registered in the capture config")]
    UnknownChannel(String),
    #[error(transparent)]
    Outlet(#[from] StudioSinkError),
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

#[derive(Debug, Error)]
pub enum StudioSinkError {
    #[error("sink `{sink}` failed: {reason}")]
    Transmission { sink: String, reason: String },
    #[error("multiple sinks failed: {summary}")]
    Aggregate { summary: String },
}

pub trait StudioSink: Send {
    fn name(&self) -> &str;
    fn send(&mut self, frame: &StudioFrame) -> Result<(), StudioSinkError>;
    fn flush(&mut self) -> Result<(), StudioSinkError> {
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ZSpaceEvaluation {
    pub s: (f32, f32),
    pub z: (f32, f32),
    pub value: (f32, f32),
}

#[derive(Clone, Debug)]
pub struct ZSpaceProjection {
    pub channel: String,
    pub lattice_len: usize,
    pub evaluations: Vec<ZSpaceEvaluation>,
}

#[derive(Clone, Debug)]
pub struct ZSpaceSinkConfig {
    pub log_start: f32,
    pub log_step: f32,
    pub s_values: Vec<ComplexScalar>,
}

impl ZSpaceSinkConfig {
    pub fn vertical_line(log_start: f32, log_step: f32, sigma: f32, tau_samples: &[f32]) -> Self {
        let s_values = tau_samples
            .iter()
            .map(|&tau| ComplexScalar::new(sigma, tau))
            .collect();
        Self {
            log_start,
            log_step,
            s_values,
        }
    }
}

#[derive(Clone)]
pub struct ZSpaceSink {
    shared: Arc<ZSpaceShared>,
}

struct ZSpaceShared {
    name: String,
    config: ZSpaceSinkConfig,
    state: Mutex<ZSpaceSinkState>,
}

#[derive(Default)]
struct ZSpaceSinkState {
    samples: HashMap<String, Vec<ComplexScalar>>,
    projections: Vec<ZSpaceProjection>,
}

pub struct ZSpaceSinkHandle {
    shared: Arc<ZSpaceShared>,
}

impl ZSpaceSink {
    pub fn new(name: impl Into<String>, config: ZSpaceSinkConfig) -> Self {
        Self {
            shared: Arc::new(ZSpaceShared {
                name: name.into(),
                config,
                state: Mutex::new(ZSpaceSinkState::default()),
            }),
        }
    }

    pub fn vertical_line(
        name: impl Into<String>,
        log_start: f32,
        log_step: f32,
        sigma: f32,
        tau_samples: &[f32],
    ) -> Self {
        Self::new(
            name,
            ZSpaceSinkConfig::vertical_line(log_start, log_step, sigma, tau_samples),
        )
    }

    pub fn handle(&self) -> ZSpaceSinkHandle {
        ZSpaceSinkHandle {
            shared: Arc::clone(&self.shared),
        }
    }

    fn failure(&self, reason: impl Into<String>) -> StudioSinkError {
        StudioSinkError::Transmission {
            sink: self.shared.name.clone(),
            reason: reason.into(),
        }
    }
}

impl ZSpaceSinkHandle {
    pub fn take_projections(&self) -> Vec<ZSpaceProjection> {
        let mut guard = self.shared.state.lock().expect("ZSpaceSink mutex poisoned");
        std::mem::take(&mut guard.projections)
    }
}

fn encode_snapshot_for_zspace(snapshot: &PulseSnapshot) -> ComplexScalar {
    ComplexScalar::new(snapshot.mean as f32, snapshot.z_bias)
}

impl StudioSink for ZSpaceSink {
    fn name(&self) -> &str {
        &self.shared.name
    }

    fn send(&mut self, frame: &StudioFrame) -> Result<(), StudioSinkError> {
        let sample = encode_snapshot_for_zspace(&frame.record.pulse);
        let mut guard = self
            .shared
            .state
            .lock()
            .map_err(|_| self.failure("state mutex poisoned"))?;
        guard
            .samples
            .entry(frame.record.channel.clone())
            .or_default()
            .push(sample);
        Ok(())
    }

    fn flush(&mut self) -> Result<(), StudioSinkError> {
        let config = self.shared.config.clone();
        let drained = {
            let mut guard = self
                .shared
                .state
                .lock()
                .map_err(|_| self.failure("state mutex poisoned"))?;
            guard.samples.drain().collect::<Vec<_>>()
        };

        if drained.is_empty() {
            return Ok(());
        }

        let mut projections = Vec::new();
        for (channel, samples) in drained {
            if samples.is_empty() {
                continue;
            }
            let weights =
                trapezoidal_weights(samples.len()).map_err(|err| self.failure(err.to_string()))?;
            let weighted = prepare_weighted_series(&samples, &weights)
                .map_err(|err| self.failure(err.to_string()))?;
            let mut evaluations = Vec::with_capacity(config.s_values.len());
            for &s in &config.s_values {
                let (prefactor, z) =
                    mellin_log_lattice_prefactor(config.log_start, config.log_step, s)
                        .map_err(|err| self.failure(err.to_string()))?;
                let series = evaluate_weighted_series(&weighted, z)
                    .map_err(|err| self.failure(err.to_string()))?;
                let value = prefactor * series;
                evaluations.push(ZSpaceEvaluation {
                    s: (s.re, s.im),
                    z: (z.re, z.im),
                    value: (value.re, value.im),
                });
            }
            projections.push(ZSpaceProjection {
                channel,
                lattice_len: samples.len(),
                evaluations,
            });
        }

        let mut guard = self
            .shared
            .state
            .lock()
            .map_err(|_| self.failure("state mutex poisoned"))?;
        guard.projections.extend(projections);
        Ok(())
    }
}

#[derive(Default)]
struct StudioOutlet {
    sinks: Vec<Box<dyn StudioSink>>,
}

impl StudioOutlet {
    fn register<S: StudioSink + 'static>(&mut self, sink: S) {
        self.sinks.push(Box::new(sink));
    }

    fn len(&self) -> usize {
        self.sinks.len()
    }

    fn broadcast(&mut self, frame: &StudioFrame) -> Result<(), StudioSinkError> {
        let mut failures: Vec<StudioSinkError> = Vec::new();
        for sink in &mut self.sinks {
            if let Err(err) = sink.send(frame) {
                failures.push(err);
            }
        }
        Self::reduce_failures(failures)
    }

    fn flush(&mut self) -> Result<(), StudioSinkError> {
        let mut failures: Vec<StudioSinkError> = Vec::new();
        for sink in &mut self.sinks {
            if let Err(err) = sink.flush() {
                failures.push(err);
            }
        }
        Self::reduce_failures(failures)
    }

    fn reduce_failures(mut failures: Vec<StudioSinkError>) -> Result<(), StudioSinkError> {
        match failures.len() {
            0 => Ok(()),
            1 => Err(failures.pop().unwrap()),
            _ => {
                let summary = failures
                    .into_iter()
                    .map(|err| err.to_string())
                    .collect::<Vec<_>>()
                    .join("; ");
                Err(StudioSinkError::Aggregate { summary })
            }
        }
    }
}

pub struct QuantumRealityStudio {
    capture: SignalCaptureSession,
    tagger: SemanticTagger,
    overlay: OverlayComposer,
    outlet: StudioOutlet,
}

impl QuantumRealityStudio {
    pub fn new(config: SignalCaptureConfig, bridge: &MaxwellDesireBridge) -> Self {
        Self {
            capture: SignalCaptureSession::new(config),
            tagger: SemanticTagger::from_bridge(bridge),
            overlay: OverlayComposer::default(),
            outlet: StudioOutlet::default(),
        }
    }

    pub fn with_sink<S: StudioSink + 'static>(mut self, sink: S) -> Self {
        self.outlet.register(sink);
        self
    }

    pub fn register_sink<S: StudioSink + 'static>(&mut self, sink: S) {
        self.outlet.register(sink);
    }

    pub fn sink_count(&self) -> usize {
        self.outlet.len()
    }

    pub fn flush_sinks(&mut self) -> Result<(), StudioError> {
        self.outlet.flush().map_err(StudioError::from)
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
        let frame = StudioFrame {
            record,
            concept,
            narrative,
            overlay,
        };
        self.outlet.broadcast(&frame)?;
        Ok(frame)
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
    use std::sync::{Arc, Mutex};

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

    struct SharedSink {
        name: String,
        frames: Arc<Mutex<Vec<StudioFrame>>>,
    }

    impl SharedSink {
        fn new(name: impl Into<String>, frames: Arc<Mutex<Vec<StudioFrame>>>) -> Self {
            Self {
                name: name.into(),
                frames,
            }
        }
    }

    impl StudioSink for SharedSink {
        fn name(&self) -> &str {
            &self.name
        }

        fn send(&mut self, frame: &StudioFrame) -> Result<(), StudioSinkError> {
            self.frames
                .lock()
                .expect("mutex poisoned")
                .push(frame.clone());
            Ok(())
        }
    }

    struct FailingSink {
        name: String,
    }

    impl FailingSink {
        fn new(name: impl Into<String>) -> Self {
            Self { name: name.into() }
        }
    }

    impl StudioSink for FailingSink {
        fn name(&self) -> &str {
            &self.name
        }

        fn send(&mut self, _frame: &StudioFrame) -> Result<(), StudioSinkError> {
            Err(StudioSinkError::Transmission {
                sink: self.name.clone(),
                reason: "simulated failure".into(),
            })
        }
    }

    #[test]
    fn studio_broadcasts_frames_to_registered_sinks() {
        let bridge = MaxwellDesireBridge::new()
            .with_channel_and_narrative(
                "alpha",
                vec![(0, 1.0)],
                vec!["glimmer".into(), "braid".into()],
            )
            .unwrap();
        let frames = Arc::new(Mutex::new(Vec::new()));
        let sink = SharedSink::new("shared", Arc::clone(&frames));
        let mut studio = QuantumRealityStudio::new(SignalCaptureConfig::new(48000.0), &bridge);
        studio.register_sink(sink);
        let frame = studio
            .ingest(&bridge, "alpha", sample_pulse(), None)
            .expect("ingest");
        assert_eq!(studio.sink_count(), 1);
        assert_eq!(frame.overlay.glyph, "braid");
        let stored = frames.lock().expect("mutex poisoned");
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].overlay.glyph, "braid");
    }

    #[test]
    fn sink_failures_are_reported() {
        let bridge = MaxwellDesireBridge::new()
            .with_channel_and_narrative(
                "alpha",
                vec![(0, 1.0)],
                vec!["glimmer".into(), "braid".into()],
            )
            .unwrap();
        let mut studio = QuantumRealityStudio::new(SignalCaptureConfig::new(48000.0), &bridge)
            .with_sink(FailingSink::new("fail"));
        let err = studio
            .ingest(&bridge, "alpha", sample_pulse(), None)
            .unwrap_err();
        match err {
            StudioError::Outlet(StudioSinkError::Transmission { sink, reason }) => {
                assert_eq!(sink, "fail");
                assert!(reason.contains("failure"));
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn zspace_sink_collects_mellin_projections() {
        let bridge = MaxwellDesireBridge::new()
            .with_channel_and_narrative(
                "alpha",
                vec![(0, 1.0)],
                vec!["glimmer".into(), "braid".into()],
            )
            .unwrap();
        let sink = ZSpaceSink::vertical_line("zspace", -2.0, 0.5, 0.0, &[-1.5, 0.0, 1.5]);
        let mirror = sink.handle();
        let mut studio = QuantumRealityStudio::new(SignalCaptureConfig::new(48000.0), &bridge);
        studio.register_sink(sink);

        let mut pulse_a = sample_pulse();
        pulse_a.z_score = 3.0;
        pulse_a.band_energy = (0.6, 0.3, 0.1);
        studio
            .ingest(&bridge, "alpha", pulse_a, None)
            .expect("ingest pulse a");

        let mut pulse_b = sample_pulse();
        pulse_b.z_score = -2.2;
        pulse_b.z_bias = 0.4;
        pulse_b.band_energy = (0.2, 0.5, 0.3);
        studio
            .ingest(&bridge, "alpha", pulse_b, None)
            .expect("ingest pulse b");

        studio.flush_sinks().expect("flush zspace sink");

        let projections = mirror.take_projections();
        assert_eq!(projections.len(), 1);
        let projection = &projections[0];
        assert_eq!(projection.channel, "alpha");
        assert_eq!(projection.lattice_len, 2);
        assert_eq!(projection.evaluations.len(), 3);
        assert!(projection
            .evaluations
            .iter()
            .any(|eval| eval.value.0.abs() > 0.0 || eval.value.1.abs() > 0.0));
    }
}
