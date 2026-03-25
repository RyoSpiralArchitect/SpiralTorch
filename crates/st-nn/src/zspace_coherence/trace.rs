// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Built-in tracing + visualization helpers for the Z-space coherence pipeline.
//!
//! The sequencer already supports plugins via [`ZSpaceSequencerPlugin`]. This module
//! adds a ready-to-use recorder that captures owned, JSON-friendly snapshots of each
//! pipeline stage and exposes helpers to turn coherence weights into canvas-ready
//! tensors.

use super::coherence_engine::DomainLinguisticProfile;
use super::sequencer::{
    CoherenceDiagnostics, CoherenceLabel, PreDiscardTelemetry, ZSpaceSequencerPlugin,
    ZSpaceSequencerStage,
};
use crate::language::{ConceptHint, NarrativeHint};
use crate::{PureResult, Tensor};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use st_core::maxwell::MaxwellZPulse;
use st_core::telemetry::noncollapse::NonCollapseSnapshot;
use st_tensor::TensorError;
use std::collections::VecDeque;
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaxwellZPulseSummary {
    pub blocks: u64,
    pub mean: f64,
    pub standard_error: f64,
    pub z_score: f64,
    pub band_energy: (f32, f32, f32),
    pub z_bias: f32,
}

impl From<&MaxwellZPulse> for MaxwellZPulseSummary {
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
pub struct PreDiscardTelemetrySummary {
    pub dominance_ratio: f32,
    pub energy_floor: f32,
    pub discarded: usize,
    pub preserved: usize,
    pub used_fallback: bool,
    pub survivor_energy: f32,
    pub discarded_energy: f32,
    pub dominant_weight: f32,
    pub preserved_ratio: f32,
    pub survivor_energy_ratio: f32,
}

impl From<&PreDiscardTelemetry> for PreDiscardTelemetrySummary {
    fn from(telemetry: &PreDiscardTelemetry) -> Self {
        Self {
            dominance_ratio: telemetry.dominance_ratio(),
            energy_floor: telemetry.energy_floor(),
            discarded: telemetry.discarded(),
            preserved: telemetry.preserved(),
            used_fallback: telemetry.used_fallback(),
            survivor_energy: telemetry.survivor_energy(),
            discarded_energy: telemetry.discarded_energy(),
            dominant_weight: telemetry.dominant_weight(),
            preserved_ratio: telemetry.preserved_ratio(),
            survivor_energy_ratio: telemetry.survivor_energy_ratio(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CoherenceDiagnosticsSummary {
    pub dominant_channel: Option<usize>,
    pub mean_coherence: f32,
    pub energy_ratio: f32,
    pub entropy: f32,
    pub fractional_order: f32,
    pub z_bias: f32,
    pub preserved_channels: usize,
    pub discarded_channels: usize,
    pub label: String,
}

impl CoherenceDiagnosticsSummary {
    fn from_diagnostics(diagnostics: &CoherenceDiagnostics) -> Self {
        let label: CoherenceLabel = diagnostics.observation().lift_to_label();
        Self {
            dominant_channel: diagnostics.dominant_channel(),
            mean_coherence: diagnostics.mean_coherence(),
            energy_ratio: diagnostics.energy_ratio(),
            entropy: diagnostics.coherence_entropy(),
            fractional_order: diagnostics.fractional_order(),
            z_bias: diagnostics.z_bias(),
            preserved_channels: diagnostics.preserved_channels(),
            discarded_channels: diagnostics.discarded_channels(),
            label: label.to_string(),
        }
    }

    fn noncollapse_snapshot(&self) -> NonCollapseSnapshot {
        NonCollapseSnapshot::new()
            .with_coherence_metrics(
                self.entropy,
                self.preserved_channels,
                self.discarded_channels,
            )
            .with_z_bias(self.z_bias)
            .with_coherence_profile(
                self.dominant_channel,
                self.energy_ratio,
                self.mean_coherence,
            )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinguisticProfileSummary {
    pub concept: String,
    pub emphasis: f32,
    pub harmonic_bias_len: Option<usize>,
    pub descriptor: Option<String>,
}

impl From<&DomainLinguisticProfile> for LinguisticProfileSummary {
    fn from(profile: &DomainLinguisticProfile) -> Self {
        Self {
            concept: profile.concept().label().to_string(),
            emphasis: profile.emphasis(),
            harmonic_bias_len: profile.harmonic_bias_len(),
            descriptor: profile.descriptor().map(|s| s.to_string()),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinguisticContourSummary {
    pub coherence_strength: f32,
    pub articulation_bias: f32,
    pub prosody_index: f32,
    pub timbre_spread: f32,
}

impl From<&super::coherence_engine::LinguisticContour> for LinguisticContourSummary {
    fn from(contour: &super::coherence_engine::LinguisticContour) -> Self {
        Self {
            coherence_strength: contour.coherence_strength(),
            articulation_bias: contour.articulation_bias(),
            prosody_index: contour.prosody_index(),
            timbre_spread: contour.timbre_spread(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinguisticChannelReportSummary {
    pub channel: usize,
    pub weight: f32,
    pub backend: String,
    pub dominant_concept: Option<String>,
    pub emphasis: f32,
    pub descriptor: Option<String>,
}

impl From<&super::coherence_engine::LinguisticChannelReport> for LinguisticChannelReportSummary {
    fn from(report: &super::coherence_engine::LinguisticChannelReport) -> Self {
        Self {
            channel: report.channel(),
            weight: report.weight(),
            backend: report.backend().label().to_string(),
            dominant_concept: report
                .dominant_concept()
                .map(|concept| concept.label().to_string()),
            emphasis: report.emphasis(),
            descriptor: report.descriptor().map(|s| s.to_string()),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ConceptHintSnapshot {
    Distribution(Vec<f32>),
    Window(Vec<(usize, f32)>),
}

impl ConceptHintSnapshot {
    fn from_hint(hint: &ConceptHint, max_len: usize) -> Self {
        match hint {
            ConceptHint::Distribution(dist) => {
                ConceptHintSnapshot::Distribution(dist.iter().copied().take(max_len).collect())
            }
            ConceptHint::Window(window) => {
                ConceptHintSnapshot::Window(window.iter().copied().take(max_len).collect())
            }
        }
    }
}

pub const ZSPACE_TRACE_SCHEMA: &str = "spiraltorch.zspace_trace";
pub const ZSPACE_TRACE_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ZSpaceTraceEvent {
    Projected {
        step: u64,
        input_shape: (usize, usize),
        projected_shape: (usize, usize),
    },
    CoherenceMeasured {
        step: u64,
        coherence: Vec<f32>,
    },
    Aggregated {
        step: u64,
        aggregated_shape: (usize, usize),
        coherence: Vec<f32>,
        diagnostics: CoherenceDiagnosticsSummary,
    },
    PreDiscardApplied {
        step: u64,
        original: Vec<f32>,
        filtered: Vec<f32>,
        telemetry: PreDiscardTelemetrySummary,
        survivors: Vec<usize>,
        discarded: Vec<usize>,
    },
    LanguageBridged {
        step: u64,
        coherence: Vec<f32>,
        concept: ConceptHintSnapshot,
        narrative: Option<NarrativeHint>,
        pulse: MaxwellZPulseSummary,
    },
    BackendConfigured {
        step: u64,
        backend: String,
    },
    LinguisticProfileRegistered {
        step: u64,
        profile: LinguisticProfileSummary,
    },
    LinguisticProfilesCleared {
        step: u64,
    },
    SemanticWindowDerived {
        step: u64,
        window: Vec<(usize, f32)>,
        tokens: usize,
    },
    SemanticDistributionDerived {
        step: u64,
        distribution: Vec<f32>,
        window: Vec<(usize, f32)>,
    },
    CanonicalConceptSelected {
        step: u64,
        concept: String,
        channel: String,
    },
    MaxwellBridgeEmitted {
        step: u64,
        channel: String,
        pulse: MaxwellZPulseSummary,
        hint: ConceptHintSnapshot,
        narrative: Option<NarrativeHint>,
    },
    LinguisticContourEmitted {
        step: u64,
        coherence: Vec<f32>,
        contour: LinguisticContourSummary,
    },
    ChannelsDescribed {
        step: u64,
        coherence: Vec<f32>,
        reports: Vec<LinguisticChannelReportSummary>,
    },
    SemanticWindowFused {
        step: u64,
        concept: ConceptHintSnapshot,
    },
    #[cfg(feature = "psi")]
    PsiTelemetryPublished {
        step: u64,
        pulse: MaxwellZPulseSummary,
        events: usize,
        reading_present: bool,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZSpaceTraceRecord {
    pub schema: String,
    pub schema_version: u32,
    pub kind: String,
    pub step: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub noncollapse: Option<NonCollapseSnapshot>,
    pub payload: Map<String, Value>,
}

impl ZSpaceTraceEvent {
    pub fn kind(&self) -> &'static str {
        match self {
            ZSpaceTraceEvent::Projected { .. } => "Projected",
            ZSpaceTraceEvent::CoherenceMeasured { .. } => "CoherenceMeasured",
            ZSpaceTraceEvent::Aggregated { .. } => "Aggregated",
            ZSpaceTraceEvent::PreDiscardApplied { .. } => "PreDiscardApplied",
            ZSpaceTraceEvent::LanguageBridged { .. } => "LanguageBridged",
            ZSpaceTraceEvent::BackendConfigured { .. } => "BackendConfigured",
            ZSpaceTraceEvent::LinguisticProfileRegistered { .. } => "LinguisticProfileRegistered",
            ZSpaceTraceEvent::LinguisticProfilesCleared { .. } => "LinguisticProfilesCleared",
            ZSpaceTraceEvent::SemanticWindowDerived { .. } => "SemanticWindowDerived",
            ZSpaceTraceEvent::SemanticDistributionDerived { .. } => {
                "SemanticDistributionDerived"
            }
            ZSpaceTraceEvent::CanonicalConceptSelected { .. } => "CanonicalConceptSelected",
            ZSpaceTraceEvent::MaxwellBridgeEmitted { .. } => "MaxwellBridgeEmitted",
            ZSpaceTraceEvent::LinguisticContourEmitted { .. } => "LinguisticContourEmitted",
            ZSpaceTraceEvent::ChannelsDescribed { .. } => "ChannelsDescribed",
            ZSpaceTraceEvent::SemanticWindowFused { .. } => "SemanticWindowFused",
            #[cfg(feature = "psi")]
            ZSpaceTraceEvent::PsiTelemetryPublished { .. } => "PsiTelemetryPublished",
        }
    }

    pub fn step(&self) -> u64 {
        match self {
            ZSpaceTraceEvent::Projected { step, .. }
            | ZSpaceTraceEvent::CoherenceMeasured { step, .. }
            | ZSpaceTraceEvent::Aggregated { step, .. }
            | ZSpaceTraceEvent::PreDiscardApplied { step, .. }
            | ZSpaceTraceEvent::LanguageBridged { step, .. }
            | ZSpaceTraceEvent::BackendConfigured { step, .. }
            | ZSpaceTraceEvent::LinguisticProfileRegistered { step, .. }
            | ZSpaceTraceEvent::LinguisticProfilesCleared { step, .. }
            | ZSpaceTraceEvent::SemanticWindowDerived { step, .. }
            | ZSpaceTraceEvent::SemanticDistributionDerived { step, .. }
            | ZSpaceTraceEvent::CanonicalConceptSelected { step, .. }
            | ZSpaceTraceEvent::MaxwellBridgeEmitted { step, .. }
            | ZSpaceTraceEvent::LinguisticContourEmitted { step, .. }
            | ZSpaceTraceEvent::ChannelsDescribed { step, .. }
            | ZSpaceTraceEvent::SemanticWindowFused { step, .. } => *step,
            #[cfg(feature = "psi")]
            ZSpaceTraceEvent::PsiTelemetryPublished { step, .. } => *step,
        }
    }

    pub fn coherence(&self) -> Option<&[f32]> {
        match self {
            ZSpaceTraceEvent::CoherenceMeasured { coherence, .. }
            | ZSpaceTraceEvent::Aggregated { coherence, .. }
            | ZSpaceTraceEvent::LanguageBridged { coherence, .. }
            | ZSpaceTraceEvent::LinguisticContourEmitted { coherence, .. }
            | ZSpaceTraceEvent::ChannelsDescribed { coherence, .. } => Some(coherence.as_slice()),
            _ => None,
        }
    }

    pub fn stable_record(&self) -> ZSpaceTraceRecord {
        ZSpaceTraceRecord {
            schema: ZSPACE_TRACE_SCHEMA.to_string(),
            schema_version: ZSPACE_TRACE_SCHEMA_VERSION,
            kind: self.kind().to_string(),
            step: self.step(),
            noncollapse: self.noncollapse_snapshot(),
            payload: self.payload_map(),
        }
    }

    fn payload_map(&self) -> Map<String, Value> {
        let Ok(value) = serde_json::to_value(self) else {
            return Map::new();
        };
        let Value::Object(root) = value else {
            return Map::new();
        };
        let Some((_kind, body)) = root.into_iter().next() else {
            return Map::new();
        };
        let Value::Object(mut body) = body else {
            return Map::new();
        };
        self.augment_payload_map(&mut body);
        body
    }

    fn augment_payload_map(&self, body: &mut Map<String, Value>) {
        if let Some(card) = self.noncollapse_card_value() {
            body.insert("noncollapse_card".to_string(), card);
        }

        if let ZSpaceTraceEvent::PreDiscardApplied {
            original, filtered, ..
        } = self
        {
            body.entry("coherence".to_string())
                .or_insert_with(|| json!(filtered));
            body.entry("coherence_before".to_string())
                .or_insert_with(|| json!(original));
        }
    }

    fn noncollapse_snapshot(&self) -> Option<NonCollapseSnapshot> {
        match self {
            ZSpaceTraceEvent::Aggregated { diagnostics, .. } => {
                Some(diagnostics.noncollapse_snapshot())
            }
            ZSpaceTraceEvent::PreDiscardApplied { telemetry, .. } => {
                let mut snapshot = NonCollapseSnapshot::new();
                snapshot.preserved_channels = Some(telemetry.preserved);
                snapshot.discarded_channels = Some(telemetry.discarded);
                snapshot.pre_discard_preserved_ratio = Some(telemetry.preserved_ratio);
                snapshot.pre_discard_survivor_energy_ratio =
                    Some(telemetry.survivor_energy_ratio);
                Some(snapshot)
            }
            ZSpaceTraceEvent::LanguageBridged { pulse, .. }
            | ZSpaceTraceEvent::MaxwellBridgeEmitted { pulse, .. } => Some(
                NonCollapseSnapshot::new()
                    .with_z_bias(pulse.z_bias)
                    .with_band_energy(pulse.band_energy),
            ),
            #[cfg(feature = "psi")]
            ZSpaceTraceEvent::PsiTelemetryPublished { pulse, .. } => Some(
                NonCollapseSnapshot::new()
                    .with_z_bias(pulse.z_bias)
                    .with_band_energy(pulse.band_energy),
            ),
            _ => None,
        }
    }

    fn noncollapse_card_value(&self) -> Option<Value> {
        match self {
            ZSpaceTraceEvent::Aggregated { diagnostics, .. } => Some(json!({
                "stage": "aggregated",
                "title": "Aggregated non-collapse",
                "summary": format!(
                    "{} preserved / {} discarded · label {}",
                    diagnostics.preserved_channels,
                    diagnostics.discarded_channels,
                    diagnostics.label
                ),
                "metrics": {
                    "coherence_entropy": diagnostics.entropy,
                    "preserved_channels": diagnostics.preserved_channels,
                    "discarded_channels": diagnostics.discarded_channels,
                    "z_bias": diagnostics.z_bias,
                    "energy_ratio": diagnostics.energy_ratio,
                    "mean_coherence": diagnostics.mean_coherence,
                    "dominant_channel": diagnostics.dominant_channel,
                }
            })),
            ZSpaceTraceEvent::LanguageBridged { pulse, .. } => Some(json!({
                "stage": "language_bridged",
                "title": "Language pulse non-collapse",
                "summary": format!(
                    "z_bias {:+.3} · bands {:.3}/{:.3}/{:.3}",
                    pulse.z_bias,
                    pulse.band_energy.0,
                    pulse.band_energy.1,
                    pulse.band_energy.2
                ),
                "metrics": {
                    "z_bias": pulse.z_bias,
                    "band_energy_above": pulse.band_energy.0,
                    "band_energy_here": pulse.band_energy.1,
                    "band_energy_beneath": pulse.band_energy.2,
                    "blocks": pulse.blocks,
                    "z_score": pulse.z_score,
                }
            })),
            ZSpaceTraceEvent::PreDiscardApplied { telemetry, .. } => Some(json!({
                "stage": "pre_discard",
                "title": "Pre-discard retention",
                "summary": format!(
                    "{} preserved / {} discarded",
                    telemetry.preserved,
                    telemetry.discarded
                ),
                "metrics": {
                    "preserved_channels": telemetry.preserved,
                    "discarded_channels": telemetry.discarded,
                    "preserved_ratio": telemetry.preserved_ratio,
                    "survivor_energy_ratio": telemetry.survivor_energy_ratio,
                    "dominant_weight": telemetry.dominant_weight,
                    "used_fallback": telemetry.used_fallback,
                }
            })),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZSpaceTrace {
    pub dropped_events: usize,
    pub events: Vec<ZSpaceTraceEvent>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZSpaceTraceConfig {
    pub capacity: usize,
    pub max_vector_len: usize,
    pub publish_plugin_events: bool,
}

impl Default for ZSpaceTraceConfig {
    fn default() -> Self {
        let publish_plugin_events = env::var("SPIRAL_ZSPACE_TRACE_PUBLISH")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "on"
                )
            })
            .unwrap_or(false);
        Self {
            capacity: 256,
            max_vector_len: 256,
            publish_plugin_events,
        }
    }
}

#[derive(Clone)]
pub struct ZSpaceTraceRecorder {
    inner: Arc<Mutex<TraceState>>,
    config: ZSpaceTraceConfig,
}

#[derive(Clone, Debug)]
struct TraceState {
    next_step: u64,
    dropped: usize,
    events: VecDeque<ZSpaceTraceEvent>,
}

impl ZSpaceTraceRecorder {
    pub fn new(config: ZSpaceTraceConfig) -> Self {
        Self {
            inner: Arc::new(Mutex::new(TraceState {
                next_step: 0,
                dropped: 0,
                events: VecDeque::new(),
            })),
            config,
        }
    }

    pub fn snapshot(&self) -> ZSpaceTrace {
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        ZSpaceTrace {
            dropped_events: guard.dropped,
            events: guard.events.iter().cloned().collect(),
        }
    }

    pub fn clear(&self) {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.events.clear();
        guard.dropped = 0;
        guard.next_step = 0;
    }

    pub fn write_jsonl(&self, path: impl AsRef<Path>) -> PureResult<()> {
        let records = self.records();
        let file =
            File::create(path.as_ref()).map_err(|err| TensorError::Generic(err.to_string()))?;
        let mut writer = BufWriter::new(file);
        for record in records {
            let json = serde_json::to_string(&record)
                .map_err(|err| TensorError::Generic(err.to_string()))?;
            writer
                .write_all(json.as_bytes())
                .and_then(|_| writer.write_all(b"\n"))
                .map_err(|err| TensorError::Generic(err.to_string()))?;
        }
        Ok(())
    }

    pub fn records(&self) -> Vec<ZSpaceTraceRecord> {
        let trace = self.snapshot();
        trace.events
            .iter()
            .map(ZSpaceTraceEvent::stable_record)
            .collect()
    }

    fn snapshot_stage(&self, stage: ZSpaceSequencerStage<'_>, step: u64) -> ZSpaceTraceEvent {
        let limit = self.config.max_vector_len.max(1);
        match stage {
            ZSpaceSequencerStage::Projected { input, projected } => ZSpaceTraceEvent::Projected {
                step,
                input_shape: input.shape(),
                projected_shape: projected.shape(),
            },
            ZSpaceSequencerStage::CoherenceMeasured { coherence, .. } => {
                ZSpaceTraceEvent::CoherenceMeasured {
                    step,
                    coherence: coherence.iter().copied().take(limit).collect(),
                }
            }
            ZSpaceSequencerStage::Aggregated {
                aggregated,
                coherence,
                diagnostics,
            } => ZSpaceTraceEvent::Aggregated {
                step,
                aggregated_shape: aggregated.shape(),
                coherence: coherence.iter().copied().take(limit).collect(),
                diagnostics: CoherenceDiagnosticsSummary::from_diagnostics(diagnostics),
            },
            ZSpaceSequencerStage::PreDiscardApplied {
                original,
                filtered,
                telemetry,
                survivors,
                discarded,
            } => ZSpaceTraceEvent::PreDiscardApplied {
                step,
                original: original.iter().copied().take(limit).collect(),
                filtered: filtered.iter().copied().take(limit).collect(),
                telemetry: PreDiscardTelemetrySummary::from(telemetry),
                survivors: survivors.to_vec(),
                discarded: discarded.to_vec(),
            },
            ZSpaceSequencerStage::LanguageBridged {
                coherence,
                concept,
                narrative,
                pulse,
                ..
            } => ZSpaceTraceEvent::LanguageBridged {
                step,
                coherence: coherence.iter().copied().take(limit).collect(),
                concept: ConceptHintSnapshot::from_hint(concept, limit),
                narrative: narrative.cloned(),
                pulse: MaxwellZPulseSummary::from(pulse),
            },
            ZSpaceSequencerStage::BackendConfigured { backend } => ZSpaceTraceEvent::BackendConfigured {
                step,
                backend: backend.label().to_string(),
            },
            ZSpaceSequencerStage::LinguisticProfileRegistered { profile } => {
                ZSpaceTraceEvent::LinguisticProfileRegistered {
                    step,
                    profile: LinguisticProfileSummary::from(profile),
                }
            }
            ZSpaceSequencerStage::LinguisticProfilesCleared => {
                ZSpaceTraceEvent::LinguisticProfilesCleared { step }
            }
            ZSpaceSequencerStage::SemanticWindowDerived { window, tokens } => {
                ZSpaceTraceEvent::SemanticWindowDerived {
                    step,
                    window: window.iter().copied().take(limit).collect(),
                    tokens,
                }
            }
            ZSpaceSequencerStage::SemanticDistributionDerived {
                window,
                distribution,
            } => ZSpaceTraceEvent::SemanticDistributionDerived {
                step,
                distribution: distribution.iter().copied().take(limit).collect(),
                window: window.iter().copied().take(limit).collect(),
            },
            ZSpaceSequencerStage::CanonicalConceptSelected { concept, channel } => {
                ZSpaceTraceEvent::CanonicalConceptSelected {
                    step,
                    concept: concept.label().to_string(),
                    channel: channel.to_string(),
                }
            }
            ZSpaceSequencerStage::MaxwellBridgeEmitted {
                channel,
                pulse,
                hint,
                narrative,
            } => ZSpaceTraceEvent::MaxwellBridgeEmitted {
                step,
                channel: channel.to_string(),
                pulse: MaxwellZPulseSummary::from(pulse),
                hint: ConceptHintSnapshot::from_hint(hint, limit),
                narrative: narrative.cloned(),
            },
            ZSpaceSequencerStage::LinguisticContourEmitted { coherence, contour } => {
                ZSpaceTraceEvent::LinguisticContourEmitted {
                    step,
                    coherence: coherence.iter().copied().take(limit).collect(),
                    contour: LinguisticContourSummary::from(contour),
                }
            }
            ZSpaceSequencerStage::ChannelsDescribed { coherence, reports } => ZSpaceTraceEvent::ChannelsDescribed {
                step,
                coherence: coherence.iter().copied().take(limit).collect(),
                reports: reports
                    .iter()
                    .map(LinguisticChannelReportSummary::from)
                    .collect(),
            },
            ZSpaceSequencerStage::SemanticWindowFused { concept } => ZSpaceTraceEvent::SemanticWindowFused {
                step,
                concept: ConceptHintSnapshot::from_hint(concept, limit),
            },
            #[cfg(feature = "psi")]
            ZSpaceSequencerStage::PsiTelemetryPublished {
                pulse,
                reading,
                events,
                ..
            } => ZSpaceTraceEvent::PsiTelemetryPublished {
                step,
                pulse: MaxwellZPulseSummary::from(pulse),
                events: events.len(),
                reading_present: reading.is_some(),
            },
        }
    }
}

impl Default for ZSpaceTraceRecorder {
    fn default() -> Self {
        Self::new(ZSpaceTraceConfig::default())
    }
}

impl ZSpaceSequencerPlugin for ZSpaceTraceRecorder {
    fn name(&self) -> &'static str {
        "zspace_trace"
    }

    fn on_stage(&self, stage: ZSpaceSequencerStage<'_>) -> PureResult<()> {
        let publish_enabled = self.config.publish_plugin_events;
        let capacity = self.config.capacity.max(1);
        let publish_event = {
            let mut guard = self
                .inner
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let step = guard.next_step;
            guard.next_step = guard.next_step.wrapping_add(1);
            let event = self.snapshot_stage(stage, step);
            let publish_event = publish_enabled.then(|| event.stable_record());
            guard.events.push_back(event);
            while guard.events.len() > capacity {
                guard.events.pop_front();
                guard.dropped = guard.dropped.saturating_add(1);
            }
            publish_event
        };

        if let Some(publish_event) = publish_event {
            use st_core::plugin::{global_registry, PluginEvent};
            let bus = global_registry().event_bus();
            if bus.has_listeners("ZSpaceTrace") {
                if let Ok(payload) = serde_json::to_value(&publish_event) {
                    bus.publish(&PluginEvent::custom("ZSpaceTrace", payload));
                }
            }
        }

        Ok(())
    }
}

/// Turns a coherence slice into a square relation tensor that can be pushed into a canvas projector.
///
/// The returned tensor is `n x n` where `n = coherence.len()`, populated with the
/// normalised outer product of the weights.
pub fn coherence_relation_tensor(coherence: &[f32]) -> PureResult<Tensor> {
    if coherence.is_empty() {
        return Err(TensorError::EmptyInput("coherence"));
    }
    let n = coherence.len();
    let max = coherence
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(0.0f32, f32::max)
        .max(1e-6);

    let mut data = Vec::with_capacity(n * n);
    for &a in coherence {
        let a = if a.is_finite() { (a / max).clamp(0.0, 1.0) } else { 0.0 };
        for &b in coherence {
            let b = if b.is_finite() { (b / max).clamp(0.0, 1.0) } else { 0.0 };
            data.push(a * b);
        }
    }
    Tensor::from_vec(n, n, data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zspace_coherence::ZSpaceCoherenceSequencer;
    use st_tensor::OpenCartesianTopos;

    #[test]
    fn recorder_captures_core_pipeline_events() {
        let topos = OpenCartesianTopos::new(-1.0, 1e-5, 10.0, 256, 8192).unwrap();
        let mut seq = ZSpaceCoherenceSequencer::new(64, 8, -1.0, topos).unwrap();
        let recorder = ZSpaceTraceRecorder::new(ZSpaceTraceConfig {
            capacity: 16,
            max_vector_len: 64,
            publish_plugin_events: false,
        });
        seq.register_plugin(recorder.clone());

        let x = Tensor::from_vec(1, 64, vec![0.05; 64]).unwrap();
        let _ = seq.forward_with_diagnostics(&x).unwrap();

        let trace = recorder.snapshot();
        assert!(!trace.events.is_empty());
        assert!(trace
            .events
            .iter()
            .any(|event| matches!(event, ZSpaceTraceEvent::Projected { .. })));
        assert!(trace
            .events
            .iter()
            .any(|event| matches!(event, ZSpaceTraceEvent::CoherenceMeasured { .. })));
        assert!(trace
            .events
            .iter()
            .any(|event| matches!(event, ZSpaceTraceEvent::Aggregated { .. })));

        let records = recorder.records();
        let aggregated = records
            .iter()
            .find(|record| record.kind == "Aggregated")
            .expect("stable aggregated record missing");
        assert_eq!(aggregated.schema, ZSPACE_TRACE_SCHEMA);
        assert_eq!(aggregated.schema_version, ZSPACE_TRACE_SCHEMA_VERSION);
        assert!(aggregated.noncollapse.is_some());
        assert!(aggregated.payload.contains_key("diagnostics"));
        assert!(aggregated.payload.contains_key("noncollapse_card"));
    }

    #[test]
    fn coherence_relation_tensor_is_square() {
        let rel = coherence_relation_tensor(&[1.0, 0.5, 0.0]).unwrap();
        assert_eq!(rel.shape(), (3, 3));
        assert!(rel.data().iter().any(|v| *v > 0.0));
    }
}
