// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 SpiralTorch Contributors
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Quantum Reality Studio primitives that stitch Maxwell-coded pulses,
//! narrative tags, and immersive overlays together.

use serde::{Deserialize, Serialize};
use st_core::maxwell::MaxwellZPulse;
use st_core::theory::inflaton_zspace::PrimordialProjection;
use st_frac::mellin_types::ComplexScalar;
use st_frac::zspace::{
    evaluate_weighted_series, mellin_log_lattice_prefactor, prepare_weighted_series,
    trapezoidal_weights,
};
use st_logic::meta_layer::MetaNarrativeLayer;
use st_logic::quantum_reality::ZSpace as LogicZSpace;
use st_nn::{ConceptHint, MaxwellDesireBridge, NarrativeHint};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

pub use st_core::maxwell::MaxwellZPulse as MaxwellPulse;

mod meta;

pub use meta::{TemporalCausalAnnotation, TemporalLogicEngine, ToposLogicBridge};

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

impl From<&PulseSnapshot> for MaxwellZPulse {
    fn from(snapshot: &PulseSnapshot) -> Self {
        MaxwellZPulse {
            blocks: snapshot.blocks,
            mean: snapshot.mean,
            standard_error: snapshot.standard_error,
            z_score: snapshot.z_score,
            band_energy: snapshot.band_energy,
            z_bias: snapshot.z_bias,
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

    fn as_weighted_window(&self) -> Vec<(usize, f32)> {
        match self {
            StudioConceptHint::Distribution(dist) => dist
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, weight)| *weight > f32::EPSILON)
                .map(|(idx, weight)| (idx, weight))
                .collect(),
            StudioConceptHint::Window(window) => window
                .iter()
                .copied()
                .filter(|(_, weight)| *weight > f32::EPSILON)
                .collect(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConceptWindow {
    pub channel: String,
    pub timestamp: SystemTime,
    pub weights: Vec<(usize, f32)>,
    pub magnitude: f32,
}

impl ConceptWindow {
    fn from_hint(record: &RecordedPulse, hint: &StudioConceptHint) -> Option<Self> {
        let mut weights = hint.as_weighted_window();
        if weights.is_empty() {
            return None;
        }
        weights.sort_by(|a, b| a.0.cmp(&b.0));
        let magnitude = record.pulse.magnitude();
        Some(Self {
            channel: record.channel.clone(),
            timestamp: record.timestamp,
            weights,
            magnitude,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OverlayGlyph {
    pub glyph: String,
    pub intensity: f32,
}

impl OverlayGlyph {
    pub fn new(glyph: impl Into<String>, intensity: f32) -> Self {
        Self {
            glyph: glyph.into(),
            intensity: intensity.max(0.0),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OverlayFrame {
    pub channel: String,
    pub glyph: String,
    pub intensity: f32,
    pub timestamp: SystemTime,
    pub glyphs: Vec<OverlayGlyph>,
}

impl OverlayFrame {
    fn from_record(record: &RecordedPulse, glyph: String, intensity: f32) -> Self {
        let mut frame = Self {
            channel: record.channel.clone(),
            glyph: glyph.clone(),
            intensity,
            timestamp: record.timestamp,
            glyphs: vec![OverlayGlyph::new(glyph, intensity)],
        };
        frame.refresh_primary();
        frame
    }

    fn refresh_primary(&mut self) {
        if let Some(primary) = self.glyphs.first() {
            self.glyph = primary.glyph.clone();
            self.intensity = primary.intensity;
        } else {
            self.glyph = self.channel.clone();
            self.intensity = 0.0;
        }
    }

    pub fn new(
        channel: impl Into<String>,
        timestamp: SystemTime,
        glyphs: Vec<OverlayGlyph>,
    ) -> Self {
        let channel = channel.into();
        let mut filtered: Vec<OverlayGlyph> = glyphs
            .into_iter()
            .filter(|glyph| !glyph.glyph.trim().is_empty())
            .collect();
        if filtered.is_empty() {
            filtered.push(OverlayGlyph::new(channel.clone(), 0.0));
        }
        let mut frame = Self {
            channel,
            glyph: String::new(),
            intensity: 0.0,
            timestamp,
            glyphs: Vec::new(),
        };
        for glyph in filtered {
            frame.push_glyph(glyph);
        }
        frame.refresh_primary();
        frame
    }

    pub fn from_pairs<I, S>(channel: impl Into<String>, timestamp: SystemTime, pairs: I) -> Self
    where
        I: IntoIterator<Item = (S, f32)>,
        S: Into<String>,
    {
        let glyphs = pairs
            .into_iter()
            .map(|(glyph, intensity)| OverlayGlyph::new(glyph, intensity))
            .collect();
        Self::new(channel, timestamp, glyphs)
    }

    pub fn from_glyphs_and_intensities<G, IG, II>(
        channel: impl Into<String>,
        timestamp: SystemTime,
        glyphs: IG,
        intensities: II,
    ) -> Self
    where
        IG: IntoIterator<Item = G>,
        G: Into<String>,
        II: IntoIterator<Item = f32>,
    {
        let mut glyph_iter = glyphs.into_iter();
        let mut intensity_iter = intensities.into_iter();
        let pairs = std::iter::from_fn(move || {
            let glyph = glyph_iter.next()?;
            let intensity = intensity_iter.next().unwrap_or(0.0);
            Some((glyph, intensity))
        });
        Self::from_pairs(channel, timestamp, pairs)
    }

    pub fn push_glyph(&mut self, glyph: OverlayGlyph) {
        if glyph.glyph.trim().is_empty() {
            return;
        }
        if self
            .glyphs
            .iter()
            .any(|existing| existing.glyph == glyph.glyph)
        {
            return;
        }
        self.glyphs.push(glyph);
        self.refresh_primary();
    }

    pub fn extend_tags<I>(&mut self, tags: I, base_intensity: f32)
    where
        I: IntoIterator,
        I::Item: Into<String>,
    {
        for (idx, tag) in tags.into_iter().enumerate() {
            let glyph = tag.into();
            if glyph.trim().is_empty() {
                continue;
            }
            let falloff = base_intensity * 0.75_f32.powi((idx + 1) as i32);
            self.push_glyph(OverlayGlyph::new(glyph, falloff));
        }
    }

    pub fn glyphs(&self) -> &[OverlayGlyph] {
        &self.glyphs
    }

    pub fn new(
        channel: impl Into<String>,
        timestamp: SystemTime,
        glyphs: Vec<OverlayGlyph>,
    ) -> Self {
        let channel = channel.into();
        let mut filtered: Vec<OverlayGlyph> = glyphs
            .into_iter()
            .filter(|glyph| !glyph.glyph.trim().is_empty())
            .collect();
        if filtered.is_empty() {
            filtered.push(OverlayGlyph::new(channel.clone(), 0.0));
        }
        let mut frame = Self {
            channel,
            glyph: String::new(),
            intensity: 0.0,
            timestamp,
            glyphs: Vec::new(),
        };
        for glyph in filtered {
            frame.push_glyph(glyph);
        }
        frame.refresh_primary();
        frame
    }

    pub fn from_pairs<I, S>(channel: impl Into<String>, timestamp: SystemTime, pairs: I) -> Self
    where
        I: IntoIterator<Item = (S, f32)>,
        S: Into<String>,
    {
        let glyphs = pairs
            .into_iter()
            .map(|(glyph, intensity)| OverlayGlyph::new(glyph, intensity))
            .collect();
        Self::new(channel, timestamp, glyphs)
    }

    pub fn from_glyphs_and_intensities<G, IG, II>(
        channel: impl Into<String>,
        timestamp: SystemTime,
        glyphs: IG,
        intensities: II,
    ) -> Self
    where
        IG: IntoIterator<Item = G>,
        G: Into<String>,
        II: IntoIterator<Item = f32>,
    {
        let mut glyph_iter = glyphs.into_iter();
        let mut intensity_iter = intensities.into_iter();
        let pairs = std::iter::from_fn(move || {
            let glyph = glyph_iter.next()?;
            let intensity = intensity_iter.next().unwrap_or(0.0);
            Some((glyph, intensity))
        });
        Self::from_pairs(channel, timestamp, pairs)
    }

    pub fn push_glyph(&mut self, glyph: OverlayGlyph) {
        if glyph.glyph.trim().is_empty() {
            return;
        }
        if self
            .glyphs
            .iter()
            .any(|existing| existing.glyph == glyph.glyph)
        {
            return;
        }
        self.glyphs.push(glyph);
        self.refresh_primary();
    }

    pub fn extend_tags<I>(&mut self, tags: I, base_intensity: f32)
    where
        I: IntoIterator,
        I::Item: Into<String>,
    {
        for (idx, tag) in tags.into_iter().enumerate() {
            let glyph = tag.into();
            if glyph.trim().is_empty() {
                continue;
            }
            let falloff = base_intensity * 0.75_f32.powi((idx + 1) as i32);
            self.push_glyph(OverlayGlyph::new(glyph, falloff));
        }
    }

    pub fn glyphs(&self) -> &[OverlayGlyph] {
        &self.glyphs
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StudioFrame {
    pub record: RecordedPulse,
    pub concept: Option<StudioConceptHint>,
    pub narrative: Option<NarrativeHint>,
    pub overlay: OverlayFrame,
    pub z_space: StudioZSpace,
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

    fn history_limit(&self) -> usize {
        self.config.history_limit
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
        OverlayFrame::from_record(record, glyph, intensity)
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

impl ZSpaceProjection {
    fn signature_components(&self) -> Vec<f64> {
        let mut signature = Vec::with_capacity(self.evaluations.len() * 6);
        for eval in &self.evaluations {
            signature.push(eval.s.0 as f64);
            signature.push(eval.s.1 as f64);
            signature.push(eval.z.0 as f64);
            signature.push(eval.z.1 as f64);
            signature.push(eval.value.0 as f64);
            signature.push(eval.value.1 as f64);
        }
        signature
    }
}

impl From<&ZSpaceProjection> for LogicZSpace {
    fn from(projection: &ZSpaceProjection) -> Self {
        LogicZSpace {
            signature: projection.signature_components(),
        }
    }
}

impl From<ZSpaceProjection> for LogicZSpace {
    fn from(projection: ZSpaceProjection) -> Self {
        LogicZSpace::from(&projection)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StudioZSpace {
    signature: Vec<f64>,
}

impl StudioZSpace {
    pub fn signature(&self) -> &[f64] {
        &self.signature
    }
}

impl From<&LogicZSpace> for StudioZSpace {
    fn from(space: &LogicZSpace) -> Self {
        Self {
            signature: space.signature.clone(),
        }
    }
}

impl From<LogicZSpace> for StudioZSpace {
    fn from(space: LogicZSpace) -> Self {
        StudioZSpace::from(&space)
    }
}

impl From<StudioZSpace> for LogicZSpace {
    fn from(space: StudioZSpace) -> Self {
        LogicZSpace {
            signature: space.signature,
        }
    }
}

#[derive(Clone)]
pub struct ZSpaceSink {
    name: String,
    inner: Arc<Mutex<ZSpaceSinkState>>,
}

#[derive(Default)]
struct ZSpaceSinkState {
    log_start: f32,
    log_step: f32,
    s_values: Vec<ComplexScalar>,
    samples: HashMap<String, Vec<ComplexScalar>>,
    projections: Vec<ZSpaceProjection>,
}

impl ZSpaceSink {
    pub fn new(
        name: impl Into<String>,
        log_start: f32,
        log_step: f32,
        s_values: Vec<ComplexScalar>,
    ) -> Self {
        let state = ZSpaceSinkState {
            log_start,
            log_step,
            s_values,
            samples: HashMap::new(),
            projections: Vec::new(),
        };
        Self {
            name: name.into(),
            inner: Arc::new(Mutex::new(state)),
        }
    }

    pub fn vertical_line(
        name: impl Into<String>,
        log_start: f32,
        log_step: f32,
        sigma: f32,
        tau_samples: &[f32],
    ) -> Self {
        let s_values = tau_samples
            .iter()
            .map(|&tau| ComplexScalar::new(sigma, tau))
            .collect();
        Self::new(name, log_start, log_step, s_values)
    }

    pub fn take_projections(&self) -> Vec<ZSpaceProjection> {
        let mut guard = self.inner.lock().expect("ZSpaceSink mutex poisoned");
        std::mem::take(&mut guard.projections)
    }

    pub fn take_logic_signatures(&self) -> Vec<(String, LogicZSpace)> {
        self.take_projections()
            .into_iter()
            .map(|projection| {
                let channel = projection.channel.clone();
                let signature = LogicZSpace::from(projection);
                (channel, signature)
            })
            .collect()
    }

    /// Injects a pre-computed slow-roll background projection straight into the sink.
    pub fn ingest_primordial_projection(
        &self,
        channel: impl Into<String>,
        projection: &PrimordialProjection,
    ) -> Result<(), StudioSinkError> {
        if projection.is_empty() {
            return Ok(());
        }

        let mut guard = self
            .inner
            .lock()
            .map_err(|_| self.failure("state mutex poisoned"))?;
        guard.log_start = projection.log_start as f32;
        guard.log_step = projection.log_step as f32;
        guard.s_values = projection
            .s_values
            .iter()
            .map(|value| ComplexScalar::new(value.re as f32, value.im as f32))
            .collect();

        let root_channel = channel.into();
        let mut projections = Vec::new();
        for (suffix, values) in [
            (
                "",
                projection
                    .spectrum
                    .iter()
                    .map(|&v| (v as f32, 0.0))
                    .collect::<Vec<_>>(),
            ),
            (
                "::H",
                projection
                    .h_z
                    .iter()
                    .map(|value| (value.re as f32, value.im as f32))
                    .collect::<Vec<_>>(),
            ),
            (
                "::epsilon",
                projection
                    .epsilon_z
                    .iter()
                    .map(|value| (value.re as f32, value.im as f32))
                    .collect::<Vec<_>>(),
            ),
        ] {
            let channel_name = format!("{}{}", root_channel, suffix);
            let mut evaluations = Vec::with_capacity(projection.len());
            for idx in 0..projection.len() {
                let s = &projection.s_values[idx];
                let z = &projection.z_points[idx];
                let (value_re, value_im) = values[idx];
                evaluations.push(ZSpaceEvaluation {
                    s: (s.re as f32, s.im as f32),
                    z: (z.re as f32, z.im as f32),
                    value: (value_re, value_im),
                });
            }
            projections.push(ZSpaceProjection {
                channel: channel_name,
                lattice_len: projection.lattice_len,
                evaluations,
            });
        }

        guard.projections.extend(projections);
        Ok(())
    }

    fn failure(&self, reason: impl Into<String>) -> StudioSinkError {
        StudioSinkError::Transmission {
            sink: self.name.clone(),
            reason: reason.into(),
        }
    }
}

fn encode_snapshot_for_zspace(snapshot: &PulseSnapshot) -> ComplexScalar {
    let amplitude = snapshot.band_energy.0 + snapshot.band_energy.1 + snapshot.band_energy.2;
    let asymmetry = snapshot.band_energy.0 - snapshot.band_energy.2;
    let real = (snapshot.z_score as f32) * amplitude.max(1e-6);
    let imag = snapshot.z_bias + asymmetry;
    ComplexScalar::new(real, imag)
}

impl StudioSink for ZSpaceSink {
    fn name(&self) -> &str {
        &self.name
    }

    fn send(&mut self, frame: &StudioFrame) -> Result<(), StudioSinkError> {
        let sample = encode_snapshot_for_zspace(&frame.record.pulse);
        let mut guard = self
            .inner
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
        let (log_start, log_step, s_values) = {
            let guard = self
                .inner
                .lock()
                .map_err(|_| self.failure("state mutex poisoned"))?;
            (guard.log_start, guard.log_step, guard.s_values.clone())
        };
        let drained = {
            let mut guard = self
                .inner
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
            let mut evaluations = Vec::new();
            for &s in &s_values {
                let (prefactor, z) = mellin_log_lattice_prefactor(log_start, log_step, s)
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
            .inner
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
    frames: VecDeque<StudioFrame>,
    meta: Option<MetaNarrativeLayer>,
}

impl QuantumRealityStudio {
    pub fn new(config: SignalCaptureConfig, bridge: &MaxwellDesireBridge) -> Self {
        let history_limit = config.history_limit;
        Self {
            capture: SignalCaptureSession::new(config),
            tagger: SemanticTagger::from_bridge(bridge),
            overlay: OverlayComposer::default(),
            outlet: StudioOutlet::default(),
            frames: VecDeque::with_capacity(history_limit),
            meta: None,
        }
    }

    pub fn with_sink<S: StudioSink + 'static>(mut self, sink: S) -> Self {
        self.outlet.register(sink);
        self
    }

    pub fn with_meta_layer(mut self, layer: MetaNarrativeLayer) -> Self {
        self.meta = Some(layer);
        self
    }

    pub fn set_meta_layer(&mut self, layer: MetaNarrativeLayer) {
        self.meta = Some(layer);
    }

    pub fn clear_meta_layer(&mut self) {
        self.meta = None;
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

    pub fn record_pulse(
        &mut self,
        channel: impl AsRef<str>,
        pulse: MaxwellZPulse,
        timestamp: Option<SystemTime>,
    ) -> Result<RecordedPulse, StudioError> {
        self.capture.ingest(
            channel.as_ref(),
            pulse,
            timestamp.unwrap_or_else(SystemTime::now),
        )
    }

    pub fn ingest(
        &mut self,
        bridge: &MaxwellDesireBridge,
        channel: impl AsRef<str>,
        pulse: MaxwellZPulse,
        timestamp: Option<SystemTime>,
    ) -> Result<StudioFrame, StudioError> {
        let channel = channel.as_ref();
        let meta = self
            .meta
            .as_mut()
            .and_then(|layer| layer.next_with_pulse(channel, &pulse));
        let emission = bridge.emit(channel, &pulse);
        let (concept, narrative) = match emission {
            Some((concept, narrative)) => {
                (Some(StudioConceptHint::from_concept(concept)), narrative)
            }
            None => (None, None),
        };
        let record = self.record_pulse(channel, pulse, timestamp)?;
        let mut narrative = narrative;
        let mut frame_z_space = StudioZSpace::from(LogicZSpace::from(pulse.clone()));
        if let Some(resolved) = meta {
            narrative = Some(NarrativeHint::new(
                resolved.channel,
                resolved.tags,
                resolved.intensity,
            ));
            frame_z_space = StudioZSpace::from(&resolved.z_space);
        }
        let record = self.record_pulse(channel, pulse, timestamp)?;
        let concept_window = concept
            .as_ref()
            .and_then(|hint| ConceptWindow::from_hint(&record, hint));
        let annotation = self.meta.observe(&record, concept_window.as_ref());
        let mut overlay = self.overlay.compose(
            &record,
            narrative.as_ref(),
            self.tagger.fallback_for(channel),
        );
        if let Some(narrative) = narrative.as_ref() {
            overlay = self.stitch_narrative_tags(overlay, narrative.tags().iter().cloned());
        } else if let Some(fallback) = self.tagger.fallback_for(channel) {
            overlay = self.stitch_narrative_tags(overlay, fallback.iter().cloned());
        }
        let temporal_tags = TemporalLogicEngine::temporal_tags(&annotation);
        overlay = self.stitch_narrative_tags(overlay, temporal_tags.into_iter());
        let sheaf_tags = self
            .topos
            .update(&record, concept_window.as_ref(), &annotation);
        overlay = self.stitch_narrative_tags(overlay, sheaf_tags.into_iter());
        let frame = StudioFrame {
            record,
            concept,
            narrative,
            overlay,
            z_space: frame_z_space,
        };
        self.outlet.broadcast(&frame)?;
        self.frames.push_back(frame.clone());
        while self.frames.len() > self.capture.history_limit() {
            self.frames.pop_front();
        }
        Ok(frame)
    }

    pub fn records(&self) -> impl Iterator<Item = &RecordedPulse> {
        self.capture.records()
    }

    pub fn frames(&self) -> impl Iterator<Item = &StudioFrame> {
        self.frames.iter()
    }

    pub fn causal_snapshot(&self) -> Vec<TemporalCausalAnnotation> {
        self.meta.snapshot()
    }

    pub fn meaning_sheaf(&self, channel: &str) -> Option<LogicZSpace> {
        self.topos.meaning_sheaf(channel)
    }

    pub fn emit_concept_window(&self, frame: &StudioFrame) -> Option<ConceptWindow> {
        frame
            .concept
            .as_ref()
            .and_then(|hint| ConceptWindow::from_hint(&frame.record, hint))
    }

    pub fn infer_concept_window(
        &self,
        bridge: &MaxwellDesireBridge,
        record: &RecordedPulse,
    ) -> Option<ConceptWindow> {
        let pulse = MaxwellZPulse::from(&record.pulse);
        bridge
            .hint_for(&record.channel, &pulse)
            .map(StudioConceptHint::from_concept)
            .and_then(|hint| ConceptWindow::from_hint(record, &hint))
    }

    pub fn stitch_narrative_tags<I>(&self, mut overlay: OverlayFrame, tags: I) -> OverlayFrame
    where
        I: IntoIterator,
        I::Item: Into<String>,
    {
        let base = overlay
            .intensity
            .max(overlay.glyphs().first().map(|g| g.intensity).unwrap_or(1.0))
            .max(1e-6);
        overlay.extend_tags(tags, base);
        overlay
    }

    pub fn export_storyboard(&self) -> serde_json::Value {
        if !self.frames.is_empty() {
            let frames: Vec<serde_json::Value> = self
                .frames
    fn storyboard_entries(&self) -> Vec<serde_json::Value> {
        if !self.frames.is_empty() {
            self.frames
                .iter()
                .enumerate()
                .map(|(ordinal, frame)| {
                    let ts = frame
                        .record
                        .timestamp
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs_f64();
                    let mut entry = serde_json::Map::new();
                    entry.insert("ordinal".into(), serde_json::json!(ordinal));
                    entry.insert(
                        "channel".into(),
                        serde_json::json!(frame.record.channel.clone()),
                    );
                    entry.insert("timestamp".into(), serde_json::json!(ts));
                    entry.insert(
                        "z_score".into(),
                        serde_json::json!(frame.record.pulse.z_score),
                    );
                    entry.insert(
                        "band_energy".into(),
                        serde_json::json!(frame.record.pulse.band_energy),
                    );
                    entry.insert(
                        "z_bias".into(),
                        serde_json::json!(frame.record.pulse.z_bias),
                    );
                    entry.insert(
                        "overlay".into(),
                        serde_json::json!({
                            "glyph": frame.overlay.glyph.clone(),
                            "intensity": frame.overlay.intensity,
                            "glyphs": frame.overlay.glyphs.clone(),
                        }),
                    );
                    entry.insert(
                        "causal".into(),
                        serde_json::json!({
                            "event_id": frame.causal.event_id,
                            "ordinal": frame.causal.ordinal,
                            "timestamp": frame.causal.timestamp,
                            "depth": frame.causal.depth,
                            "parents": frame.causal.parent_channels.clone(),
                            "magnitude": frame.causal.magnitude,
                        }),
                    );
                    if let Some(narrative) = frame.narrative.as_ref() {
                        entry.insert(
                            "narrative".into(),
                            serde_json::json!({
                                "tags": narrative.tags(),
                                "intensity": narrative.intensity(),
                            }),
                        );
                    }
                    if let Some(window) = frame
                        .concept
                        .as_ref()
                        .and_then(|hint| ConceptWindow::from_hint(&frame.record, hint))
                    {
                        entry.insert(
                            "concept_window".into(),
                            serde_json::json!({
                                "weights": window.weights,
                                "magnitude": window.magnitude,
                            }),
                        );
                    }
                    if let Some(sheaf) = self.meaning_sheaf(&frame.record.channel) {
                        entry.insert(
                            "meaning_sheaf".into(),
                            serde_json::json!({
                                "signature": sheaf.signature,
                            }),
                        );
                    }
                    serde_json::Value::Object(entry)
                })
                .collect()
        } else {
            self.records()
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
                .collect()
        }
    }

    pub fn export_storyboard(&self) -> serde_json::Value {
        let frames = self.storyboard_entries();
        serde_json::json!({ "frames": frames })
    }

    pub fn export_storyboard_grouped(&self) -> serde_json::Value {
        let mut grouped: BTreeMap<String, Vec<serde_json::Value>> = BTreeMap::new();
        for entry in self.storyboard_entries() {
            let channel = entry
                .get("channel")
                .and_then(|value| value.as_str())
                .unwrap_or("unknown")
                .to_string();
            grouped.entry(channel).or_default().push(entry);
        }
        let channels =
            grouped
                .into_iter()
                .fold(serde_json::Map::new(), |mut acc, (channel, frames)| {
                    acc.insert(channel, serde_json::Value::Array(frames));
                    acc
                });
        serde_json::Value::Object({
            let mut root = serde_json::Map::new();
            root.insert("channels".into(), serde_json::Value::Object(channels));
            root
        })
    }

    pub fn export_storyboard_ndjson(&self) -> String {
        self.storyboard_entries()
            .into_iter()
            .map(|value| serde_json::to_string(&value).unwrap_or_else(|_| "{}".into()))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use serde_json::Value;
    use st_core::theory::inflaton_zspace::{z_transform, LogLattice, PrimordialProjection};
    use st_logic::meta_layer::{MeaningSection, MeaningSheaf, MetaNarrativeLayer, NarrativeBeat};
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
        assert!(frame.overlay.glyphs().len() >= 1);
        assert_eq!(frame.z_space.signature().len(), 8);
    }

    #[test]
    fn meta_layer_overrides_bridge_narrative() {
        let bridge = MaxwellDesireBridge::new()
            .with_channel("alpha", vec![(0, 1.0)])
            .unwrap();
        let mut layer = MetaNarrativeLayer::new();
        layer
            .engine_mut()
            .insert_event(
                "beat-1",
                NarrativeBeat::new(Some("alpha".into()), "z-open", vec!["causal".into()])
                    .with_intensity_scale(0.5)
                    .with_floor(0.1)
                    .with_sheaf_threshold(0.05),
            )
            .unwrap();
        layer.bridge_mut().attach(
            "beat-1",
            MeaningSheaf::new().with_section(MeaningSection::for_open(
                "z-open",
                vec!["sheaf".into()],
                0.2,
            )),
        );
        let mut studio = QuantumRealityStudio::new(SignalCaptureConfig::new(48000.0), &bridge)
            .with_meta_layer(layer);
        let frame = studio
            .ingest(&bridge, "alpha", sample_pulse(), None)
            .expect("ingest with meta");
        let narrative = frame.narrative.expect("meta narrative");
        assert!(narrative.tags().iter().any(|tag| tag == "causal"));
        assert!(narrative.tags().iter().any(|tag| tag == "sheaf"));
        assert!(narrative.intensity() > 0.0);
        assert!((frame.z_space.signature()[3] - 4.2).abs() < 1e-6);
    }

    #[test]
    fn meta_layer_overrides_bridge_narrative() {
        let bridge = MaxwellDesireBridge::new()
            .with_channel("alpha", vec![(0, 1.0)])
            .unwrap();
        let mut layer = MetaNarrativeLayer::new();
        layer
            .engine_mut()
            .insert_event(
                "beat-1",
                NarrativeBeat::new(Some("alpha".into()), "z-open", vec!["causal".into()])
                    .with_intensity_scale(0.5)
                    .with_floor(0.1)
                    .with_sheaf_threshold(0.05),
            )
            .unwrap();
        layer.bridge_mut().attach(
            "beat-1",
            MeaningSheaf::new().with_section(MeaningSection::for_open(
                "z-open",
                vec!["sheaf".into()],
                0.2,
            )),
        );
        let mut studio = QuantumRealityStudio::new(SignalCaptureConfig::new(48000.0), &bridge)
            .with_meta_layer(layer);
        let frame = studio
            .ingest(&bridge, "alpha", sample_pulse(), None)
            .expect("ingest with meta");
        let narrative = frame.narrative.expect("meta narrative");
        assert!(narrative.tags().iter().any(|tag| tag == "causal"));
        assert!(narrative.tags().iter().any(|tag| tag == "sheaf"));
        assert!(narrative.intensity() > 0.0);
    }

    #[test]
    fn overlay_from_pairs_rejects_duplicates_and_empty_glyphs() {
        let timestamp = SystemTime::now();
        let overlay = OverlayFrame::from_pairs(
            "alpha",
            timestamp,
            vec![("", 1.0), ("braid", 0.8), ("braid", 0.2), ("tunnel", 0.6)],
        );
        assert_eq!(overlay.channel, "alpha");
        assert_eq!(overlay.glyph, "braid");
        assert_eq!(overlay.glyphs().len(), 2);
        assert!(overlay.glyphs().iter().any(|glyph| glyph.glyph == "tunnel"));
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
        assert!(stored[0].overlay.glyphs().len() >= 1);
    }

    #[test]
    fn storyboard_export_includes_overlay_and_narrative_details() {
        let bridge = MaxwellDesireBridge::new()
            .with_channel_and_narrative(
                "alpha",
                vec![(3, 0.4), (5, 0.6)],
                vec!["glimmer".into(), "braid".into()],
            )
            .unwrap();
        let mut studio = QuantumRealityStudio::new(SignalCaptureConfig::new(48000.0), &bridge);
        let frame = studio
            .ingest(&bridge, "alpha", sample_pulse(), None)
            .expect("ingest");
        assert_eq!(studio.frames().count(), 1);

        let export = studio.export_storyboard();
        let frames = export
            .get("frames")
            .and_then(Value::as_array)
            .expect("frames array");
        assert_eq!(frames.len(), 1);
        let entry = &frames[0];
        assert_eq!(entry.get("channel").and_then(Value::as_str), Some("alpha"));
        assert_eq!(
            entry.get("z_score").and_then(Value::as_f64).unwrap() as f32,
            frame.record.pulse.z_score as f32
        );
        let overlay = entry
            .get("overlay")
            .and_then(Value::as_object)
            .expect("overlay object");
        assert_eq!(overlay.get("glyph").and_then(Value::as_str), Some("braid"));
        let glyphs = overlay
            .get("glyphs")
            .and_then(Value::as_array)
            .expect("glyph list");
        assert!(glyphs.len() >= 1);
        let causal = entry
            .get("causal")
            .and_then(Value::as_object)
            .expect("causal object");
        assert_eq!(
            causal.get("ordinal").and_then(Value::as_u64).unwrap() as usize,
            frame.causal.ordinal
        );
        let narrative = entry
            .get("narrative")
            .and_then(Value::as_object)
            .expect("narrative object");
        let tags = narrative
            .get("tags")
            .and_then(Value::as_array)
            .expect("tags array");
        assert!(tags.iter().any(|tag| tag.as_str() == Some("glimmer")));
        let concept_window = entry
            .get("concept_window")
            .and_then(Value::as_object)
            .expect("concept window object");
        let weights = concept_window
            .get("weights")
            .and_then(Value::as_array)
            .expect("weights array");
        assert!(!weights.is_empty());
        let sheaf = entry
            .get("meaning_sheaf")
            .and_then(Value::as_object)
            .expect("meaning sheaf object");
        assert!(sheaf
            .get("signature")
            .and_then(Value::as_array)
            .map(|values| !values.is_empty())
            .unwrap_or(false));
    }

    #[test]
    fn causal_snapshot_orders_events() {
        let bridge = MaxwellDesireBridge::new()
            .with_channel_and_narrative("alpha", vec![(0, 1.0)], vec!["braid".into()])
            .unwrap();
        let mut studio = QuantumRealityStudio::new(SignalCaptureConfig::new(48000.0), &bridge);
        studio
            .ingest(&bridge, "alpha", sample_pulse(), None)
            .expect("first ingest");
        let mut second = sample_pulse();
        second.z_score = 5.0;
        studio
            .ingest(&bridge, "alpha", second, None)
            .expect("second ingest");

        let snapshot = studio.causal_snapshot();
        assert_eq!(snapshot.len(), 2);
        assert!(snapshot[1].ordinal > snapshot[0].ordinal);
        assert!(snapshot[1].depth >= snapshot[0].depth);
        assert!(!snapshot[1].parents.is_empty());
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
        let mirror = sink.clone();
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

    #[test]
    fn zspace_signatures_flatten_evaluations() {
        let bridge = MaxwellDesireBridge::new()
            .with_channel_and_narrative(
                "alpha",
                vec![(0, 1.0)],
                vec!["glimmer".into(), "braid".into()],
            )
            .unwrap();
        let sink = ZSpaceSink::vertical_line("zspace", -1.0, 0.25, 0.5, &[-2.0, 0.0, 2.0, 3.0]);
        let mirror = sink.clone();
        let mut studio = QuantumRealityStudio::new(SignalCaptureConfig::new(48000.0), &bridge);
        studio.register_sink(sink);

        studio
            .ingest(&bridge, "alpha", sample_pulse(), None)
            .expect("ingest pulse");

        studio.flush_sinks().expect("flush zspace sink");

        let signatures = mirror.take_logic_signatures();
        assert_eq!(signatures.len(), 1);
        let (channel, signature) = &signatures[0];
        assert_eq!(channel, "alpha");
        assert_eq!(signature.signature.len(), 24);
        assert!(signature.signature.iter().any(|value| value.abs() > 0.0));
    }

    #[test]
    fn record_and_infer_concept_window() {
        let bridge = MaxwellDesireBridge::new()
            .with_channel_and_narrative(
                "alpha",
                vec![(0, 0.6), (1, 0.4)],
                vec!["glimmer".into(), "braid".into()],
            )
            .unwrap();
        let mut studio = QuantumRealityStudio::new(SignalCaptureConfig::new(48000.0), &bridge);
        let record = studio
            .record_pulse("alpha", sample_pulse(), None)
            .expect("record");
        let window = studio
            .infer_concept_window(&bridge, &record)
            .expect("window");
        assert_eq!(window.channel, "alpha");
        assert!(window.magnitude > 0.0);
        assert!(!window.weights.is_empty());
    }

    #[test]
    fn emit_window_from_frame() {
        let bridge = MaxwellDesireBridge::new()
            .with_channel_and_narrative(
                "alpha",
                vec![(0, 0.8), (1, 0.2)],
                vec!["glimmer".into(), "braid".into()],
            )
            .unwrap();
        let mut studio = QuantumRealityStudio::new(SignalCaptureConfig::new(48000.0), &bridge);
        let frame = studio
            .ingest(&bridge, "alpha", sample_pulse(), None)
            .expect("ingest");
        let window = studio.emit_concept_window(&frame).expect("window");
        assert_eq!(window.channel, "alpha");
        assert!(window.magnitude > 0.0);
        assert!(window.weights.len() >= 1);
    }

    #[test]
    fn stitching_appends_tags_with_falloff() {
        let bridge = MaxwellDesireBridge::new()
            .with_channel_and_narrative("alpha", vec![(0, 1.0)], vec!["braid".into()])
            .unwrap();
        let studio = QuantumRealityStudio::new(SignalCaptureConfig::new(48000.0), &bridge);
        let overlay = OverlayFrame::new(
            "alpha",
            SystemTime::now(),
            vec![OverlayGlyph::new("braid", 1.0)],
        );
        let stitched = studio.stitch_narrative_tags(
            overlay,
            vec![
                "braid".to_string(),
                "tunnel".to_string(),
                "spire".to_string(),
            ],
        );
        assert_eq!(stitched.glyph, "braid");
        assert!(stitched.glyphs().len() >= 2);
        let intensities: Vec<f32> = stitched.glyphs().iter().map(|g| g.intensity).collect();
        assert!(intensities[0] >= intensities[1]);
    }

    #[test]
    fn zspace_sink_ingests_primordial_projection_bundle() {
        let log_start = 0.0;
        let log_step = 0.25;
        let hubble_samples = vec![12.0, 11.5, 11.0, 10.5];
        let epsilon_samples = vec![0.01, 0.011, 0.012, 0.013];
        let hubble_lattice = LogLattice::from_samples(log_start, log_step, hubble_samples.clone());
        let epsilon_lattice =
            LogLattice::from_samples(log_start, log_step, epsilon_samples.clone());
        let s_values = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.5),
            Complex64::new(1.0, -0.5),
        ];
        let z_points: Vec<Complex64> = s_values
            .iter()
            .map(|s| (s * Complex64::new(log_step, 0.0)).exp())
            .collect();
        let h_z: Vec<Complex64> = z_points
            .iter()
            .map(|&z| z_transform(&hubble_lattice.weights, &hubble_lattice.samples, z))
            .collect();
        let epsilon_z: Vec<Complex64> = z_points
            .iter()
            .map(|&z| z_transform(&epsilon_lattice.weights, &epsilon_lattice.samples, z))
            .collect();
        let projection = PrimordialProjection::new(
            log_start,
            log_step,
            hubble_lattice.len(),
            s_values.clone(),
            z_points.clone(),
            h_z.clone(),
            epsilon_z.clone(),
            2.435e18,
        );

        let sink = ZSpaceSink::new(
            "inflation",
            log_start as f32,
            log_step as f32,
            s_values
                .iter()
                .map(|s| ComplexScalar::new(s.re as f32, s.im as f32))
                .collect(),
        );
        sink.ingest_primordial_projection("inflation", &projection)
            .expect("ingest projection");
        let projections = sink.take_projections();
        assert_eq!(projections.len(), 3);
        let mut channels: Vec<String> = projections.iter().map(|p| p.channel.clone()).collect();
        channels.sort();
        assert_eq!(
            channels,
            vec!["inflation", "inflation::H", "inflation::epsilon"]
        );
        let spectrum_projection = projections
            .into_iter()
            .find(|p| p.channel == "inflation")
            .expect("spectrum channel");
        assert_eq!(spectrum_projection.evaluations.len(), projection.len());
        for eval in spectrum_projection.evaluations {
            assert!(eval.value.0.is_finite());
            assert_eq!(eval.value.1, 0.0);
        }
    }

    #[test]
    fn storyboard_grouped_collects_channels() {
        let mut bridge = MaxwellDesireBridge::new();
        bridge
            .register_channel_with_narrative(
                "alpha",
                vec![(0, 1.0)],
                vec!["braid".into(), "glimmer".into()],
            )
            .unwrap();
        bridge
            .register_channel_with_narrative("beta", vec![(1, 1.0)], vec!["tunnel".into()])
            .unwrap();
        let mut studio = QuantumRealityStudio::new(SignalCaptureConfig::new(48000.0), &bridge);
        studio
            .ingest(&bridge, "alpha", sample_pulse(), None)
            .expect("ingest alpha");
        let mut beta = sample_pulse();
        beta.z_score = 2.1;
        studio
            .ingest(&bridge, "beta", beta, None)
            .expect("ingest beta");

        let export = studio.export_storyboard_grouped();
        let channels = export
            .get("channels")
            .and_then(Value::as_object)
            .expect("channels map");
        assert_eq!(channels.len(), 2);
        for (channel, frames) in channels {
            let frames = frames
                .as_array()
                .unwrap_or_else(|| panic!("frames array missing for channel `{}`", channel));
            assert!(!frames.is_empty());
            let first = frames[0]
                .get("channel")
                .and_then(Value::as_str)
                .expect("channel string");
            assert_eq!(first, channel);
        }
    }

    #[test]
    fn ndjson_export_emits_valid_lines() {
        let bridge = MaxwellDesireBridge::new();
        let mut studio = QuantumRealityStudio::new(SignalCaptureConfig::new(48000.0), &bridge);
        studio
            .record_pulse("alpha", sample_pulse(), None)
            .expect("record alpha");
        let ndjson = studio.export_storyboard_ndjson();
        let lines: Vec<&str> = ndjson.lines().collect();
        assert_eq!(lines.len(), 1);
        let parsed: Value = serde_json::from_str(lines[0]).expect("valid json line");
        assert_eq!(parsed.get("channel").and_then(Value::as_str), Some("alpha"));
    }
}
