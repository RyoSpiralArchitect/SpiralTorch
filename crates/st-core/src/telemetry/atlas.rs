// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// =============================================================================
//  SpiralReality Proprietary
//  Copyright (c) 2025 SpiralReality. All Rights Reserved.
//
//  NOTICE: This file contains confidential and proprietary information of
//  SpiralReality. ANY USE, COPYING, MODIFICATION, DISTRIBUTION, DISPLAY,
//  OR DISCLOSURE OF THIS FILE, IN WHOLE OR IN PART, IS STRICTLY PROHIBITED
//  WITHOUT THE PRIOR WRITTEN CONSENT OF SPIRALREALITY.
//
//  NO LICENSE IS GRANTED OR IMPLIED BY THIS FILE. THIS SOFTWARE IS PROVIDED
//  "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
//  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
//  PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL SPIRALREALITY OR ITS
//  SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
//  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
//  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// =============================================================================

//! Atlas projections stitching timeline telemetry with maintainer diagnostics.

use super::chrono::{ChronoHarmonics, ChronoSummary};
use super::maintainer::MaintainerStatus;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Conceptual framing attached to atlas fragments so downstream consumers can
/// keep philosophical language—especially around qualia—in the declared scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConceptSense {
    /// C. I. Lewis' "given" — pre-conceptual sensory texture anchoring epistemic work.
    QualiaLewisGiven,
    /// Thomas Nagel's subjectivity — the "what it is like" perspective gap.
    QualiaNagelSubjectivity,
    /// Frank Jackson's knowledge argument — phenomenal facts beyond the physical story.
    QualiaJacksonKnowledge,
    /// David Chalmers' hard problem — phenomenal experience resisting reduction.
    QualiaChalmersHardProblem,
    /// Giulio Tononi's IIT identity — qualia as maximally irreducible causal structure.
    QualiaTononiIit,
    /// General discourse drift — colloquial qualia as a fuzzy stand-in for feeling.
    QualiaGeneralDiscourse,
}

impl ConceptSense {
    /// Returns a short label usable in telemetry dashboards.
    pub fn label(&self) -> &'static str {
        match self {
            ConceptSense::QualiaLewisGiven => "qualia.lewis_given",
            ConceptSense::QualiaNagelSubjectivity => "qualia.nagel_subjectivity",
            ConceptSense::QualiaJacksonKnowledge => "qualia.jackson_knowledge",
            ConceptSense::QualiaChalmersHardProblem => "qualia.chalmers_hard",
            ConceptSense::QualiaTononiIit => "qualia.tononi_iit",
            ConceptSense::QualiaGeneralDiscourse => "qualia.general_discourse",
        }
    }

    /// Returns a prose description summarising the sense.
    pub fn description(&self) -> &'static str {
        match self {
            ConceptSense::QualiaLewisGiven => {
                "Lewis: qualia as pre-conceptual givens underwriting epistemic grounding."
            }
            ConceptSense::QualiaNagelSubjectivity => {
                "Nagel: qualia as the irreducibly subjective 'what it is like' perspective."
            }
            ConceptSense::QualiaJacksonKnowledge => {
                "Jackson: qualia as phenomenal knowledge unattainable from physical facts alone."
            }
            ConceptSense::QualiaChalmersHardProblem => {
                "Chalmers: qualia as the hard problem's phenomenal core resisting reduction."
            }
            ConceptSense::QualiaTononiIit => {
                "Tononi: qualia identified with IIT's maximally irreducible conceptual structures."
            }
            ConceptSense::QualiaGeneralDiscourse => {
                "General discourse: qualia as a loose synonym for feeling or consciousness."
            }
        }
    }
}

/// Annotation capturing how a fragment is framing sensitive philosophical vocabulary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConceptAnnotation {
    /// Target term receiving the annotation (for example, "qualia").
    pub term: String,
    /// Conceptual sense claimed by the producer.
    pub sense: ConceptSense,
    /// Optional free-form note documenting the rationale.
    pub rationale: Option<String>,
}

impl ConceptAnnotation {
    /// Creates a bare annotation for the provided term and sense.
    pub fn new(term: impl Into<String>, sense: ConceptSense) -> Self {
        Self {
            term: normalise_text(term),
            sense,
            rationale: None,
        }
    }

    /// Creates an annotation with an explicit rationale.
    pub fn with_rationale(
        term: impl Into<String>,
        sense: ConceptSense,
        rationale: impl Into<String>,
    ) -> Self {
        let rationale = normalise_optional_text(Some(rationale.into()));
        Self {
            term: normalise_text(term),
            sense,
            rationale,
        }
    }
}

fn normalise_text(input: impl Into<String>) -> String {
    let text = input.into();
    let trimmed = text.trim();
    if trimmed.is_empty() {
        String::new()
    } else {
        trimmed.to_string()
    }
}

fn normalise_optional_text(input: Option<String>) -> Option<String> {
    input.and_then(|text| {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

/// Named scalar surfaced through the atlas projection.
#[derive(Clone, Debug, PartialEq)]
pub struct AtlasMetric {
    /// Human-readable identifier for the metric.
    pub name: String,
    /// Scalar value carried by the metric.
    pub value: f32,
    /// Optional district categorisation supplied by the producer.
    pub district: Option<String>,
}

impl AtlasMetric {
    /// Creates a new metric when the value is finite.
    pub fn new(name: impl Into<String>, value: f32) -> Option<Self> {
        if value.is_finite() {
            Some(Self {
                name: name.into(),
                value,
                district: None,
            })
        } else {
            None
        }
    }

    /// Creates a new metric with an explicit district when the value is finite.
    pub fn with_district(
        name: impl Into<String>,
        value: f32,
        district: impl Into<String>,
    ) -> Option<Self> {
        if value.is_finite() {
            Some(Self {
                name: name.into(),
                value,
                district: Some(district.into()),
            })
        } else {
            None
        }
    }

    /// Returns the district when supplied by the producer.
    pub fn district(&self) -> Option<&str> {
        self.district.as_deref()
    }
}

/// Fragment of atlas state produced by one subsystem.
#[derive(Clone, Debug, Default)]
pub struct AtlasFragment {
    /// Optional timestamp associated with the fragment.
    pub timestamp: Option<f32>,
    /// Optional chrono summary captured by the fragment.
    pub summary: Option<ChronoSummary>,
    /// Optional harmonic analysis paired with the summary.
    pub harmonics: Option<ChronoHarmonics>,
    /// Support contributed by the fragment's producer.
    pub loop_support: Option<f32>,
    /// Optional collapse total conveyed by the fragment.
    pub collapse_total: Option<f32>,
    /// Optional Z-space control signal routed through the fragment.
    pub z_signal: Option<f32>,
    /// Optional SpiralK script hint attached to the fragment.
    pub script_hint: Option<String>,
    /// Maintainer status, if the fragment carried one.
    pub maintainer_status: Option<MaintainerStatus>,
    /// Human-readable maintainer diagnostic, if present.
    pub maintainer_diagnostic: Option<String>,
    /// Suggested geometry clamp from maintainer analysis, if any.
    pub suggested_max_scale: Option<f32>,
    /// Suggested Leech pressure bump from maintainer analysis, if any.
    pub suggested_pressure: Option<f32>,
    /// Additional scalar metrics emitted alongside the fragment.
    pub metrics: Vec<AtlasMetric>,
    /// Free-form notes describing the fragment provenance.
    pub notes: Vec<String>,
    /// Conceptual annotations describing how sensitive terms are being framed.
    pub concepts: Vec<ConceptAnnotation>,
}

impl AtlasFragment {
    /// Creates a new, empty fragment.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true when the fragment does not contain any payload.
    pub fn is_empty(&self) -> bool {
        self.timestamp.is_none()
            && self.summary.is_none()
            && self.harmonics.is_none()
            && self.loop_support.is_none()
            && self.collapse_total.is_none()
            && self.z_signal.is_none()
            && self.script_hint.is_none()
            && self.maintainer_status.is_none()
            && self.maintainer_diagnostic.is_none()
            && self.suggested_max_scale.is_none()
            && self.suggested_pressure.is_none()
            && self.metrics.is_empty()
            && self.notes.is_empty()
            && self.concepts.is_empty()
    }

    /// Pushes a metric onto the fragment when the value is finite.
    pub fn push_metric(&mut self, name: impl Into<String>, value: f32) {
        if let Some(metric) = AtlasMetric::new(name, value) {
            self.metrics.push(metric);
        }
    }

    /// Pushes a metric with an explicit district when the value is finite.
    pub fn push_metric_with_district(
        &mut self,
        name: impl Into<String>,
        value: f32,
        district: impl Into<String>,
    ) {
        if let Some(metric) = AtlasMetric::with_district(name, value, district) {
            self.metrics.push(metric);
        }
    }

    /// Appends a note to the fragment provenance trail.
    pub fn push_note(&mut self, note: impl Into<String>) {
        if let Some(note) = normalise_optional_text(Some(note.into())) {
            if !note.is_empty() {
                self.notes.push(note);
            }
        }
    }

    /// Appends a conceptual annotation to the fragment.
    pub fn push_concept(&mut self, annotation: ConceptAnnotation) {
        self.concepts.push(annotation);
    }

    /// Convenience helper to attach a qualia annotation with optional rationale.
    pub fn annotate_qualia(&mut self, sense: ConceptSense, rationale: Option<impl Into<String>>) {
        let mut annotation = ConceptAnnotation::new("qualia", sense);
        if let Some(rationale) = rationale {
            annotation.rationale = Some(rationale.into());
        }
        self.push_concept(annotation);
    }
}

/// Aggregated atlas state that can be replayed by clients.
#[derive(Clone, Debug, Default)]
pub struct AtlasFrame {
    /// Latest timestamp captured across all fragments.
    pub timestamp: f32,
    /// Latest chrono summary applied to the frame.
    pub chrono_summary: Option<ChronoSummary>,
    /// Latest harmonic snapshot applied to the frame.
    pub harmonics: Option<ChronoHarmonics>,
    /// Accumulated loop support collected from fragments.
    pub loop_support: f32,
    /// Last collapse total reported into the atlas.
    pub collapse_total: Option<f32>,
    /// Last Z-space control signal reported into the atlas.
    pub z_signal: Option<f32>,
    /// Latest SpiralK script hint stitched into the atlas.
    pub script_hint: Option<String>,
    /// Latest maintainer status routed into the frame.
    pub maintainer_status: Option<MaintainerStatus>,
    /// Latest maintainer diagnostic carried with the frame.
    pub maintainer_diagnostic: Option<String>,
    /// Most recent geometry clamp recommendation.
    pub suggested_max_scale: Option<f32>,
    /// Most recent Leech pressure recommendation.
    pub suggested_pressure: Option<f32>,
    /// Scalar metrics accumulated across fragments.
    pub metrics: Vec<AtlasMetric>,
    /// Provenance notes collected from fragments.
    pub notes: Vec<String>,
    /// Conceptual annotations aggregated across fragments.
    pub concepts: Vec<ConceptAnnotation>,
}

impl AtlasFrame {
    /// Creates a frame initialised at the provided timestamp.
    pub fn new(timestamp: f32) -> Self {
        let timestamp = if timestamp.is_finite() {
            timestamp
        } else {
            0.0
        };
        Self {
            timestamp,
            ..Default::default()
        }
    }

    /// Creates a frame from a fragment when it carries any payload.
    pub fn from_fragment(fragment: AtlasFragment) -> Option<Self> {
        if fragment.is_empty() {
            return None;
        }
        let mut frame = Self::default();
        frame.merge_fragment(fragment);
        if frame.timestamp <= 0.0 {
            None
        } else {
            Some(frame)
        }
    }

    /// Merges a fragment into the frame.
    pub fn merge_fragment(&mut self, fragment: AtlasFragment) {
        if fragment.is_empty() {
            return;
        }
        let timestamp = fragment.timestamp.or_else(|| {
            fragment
                .summary
                .as_ref()
                .map(|summary| summary.latest_timestamp)
        });
        if let Some(ts) = timestamp {
            if ts.is_finite() {
                if self.timestamp <= 0.0 {
                    self.timestamp = ts.max(f32::EPSILON);
                } else {
                    self.timestamp = self.timestamp.max(ts);
                }
            }
        }
        if let Some(summary) = fragment.summary {
            self.chrono_summary = Some(summary);
        }
        if let Some(harmonics) = fragment.harmonics {
            self.harmonics = Some(harmonics);
        }
        if let Some(support) = fragment.loop_support {
            if support.is_finite() {
                self.loop_support += support.max(0.0);
            }
        }
        if let Some(total) = fragment.collapse_total {
            if total.is_finite() {
                self.collapse_total = Some(total);
            }
        }
        if let Some(z) = fragment.z_signal {
            if z.is_finite() {
                self.z_signal = Some(z);
            }
        }
        if let Some(script) = fragment.script_hint {
            if let Some(script) = normalise_optional_text(Some(script)) {
                if !script.is_empty() {
                    self.script_hint = Some(script);
                }
            }
        }
        if let Some(status) = fragment.maintainer_status {
            self.maintainer_status = Some(status);
        }
        if let Some(diagnostic) = fragment.maintainer_diagnostic {
            if let Some(diagnostic) = normalise_optional_text(Some(diagnostic)) {
                if !diagnostic.is_empty() {
                    self.maintainer_diagnostic = Some(diagnostic);
                }
            }
        }
        if let Some(clamp) = fragment.suggested_max_scale {
            if clamp.is_finite() {
                self.suggested_max_scale = Some(clamp);
            }
        }
        if let Some(pressure) = fragment.suggested_pressure {
            if pressure.is_finite() {
                self.suggested_pressure = Some(pressure);
            }
        }
        merge_metrics(&mut self.metrics, fragment.metrics);
        merge_notes(&mut self.notes, fragment.notes);
        merge_concepts(&mut self.concepts, fragment.concepts);
        if self.timestamp <= 0.0 {
            self.timestamp = f32::EPSILON;
        }
    }

    /// Retrieves the metric matching the provided identifier, if present.
    pub fn metric(&self, name: &str) -> Option<&AtlasMetric> {
        self.metrics.iter().find(|metric| metric.name == name)
    }

    /// Retrieves the latest scalar value for a metric identifier, if present.
    pub fn metric_value(&self, name: &str) -> Option<f32> {
        self.metric(name).map(|metric| metric.value)
    }

    /// Returns metrics whose identifiers share the provided prefix sorted by name.
    pub fn metrics_with_prefix(&self, prefix: &str) -> Vec<&AtlasMetric> {
        let mut metrics: Vec<&AtlasMetric> = self
            .metrics
            .iter()
            .filter(|metric| metric.name.starts_with(prefix))
            .collect();
        metrics.sort_by(|a, b| a.name.cmp(&b.name));
        metrics
    }

    /// Groups metrics into named districts following the SpiralTorch atlas map.
    pub fn districts(&self) -> Vec<AtlasDistrict> {
        if self.metrics.is_empty() {
            return Vec::new();
        }
        let mut buckets: BTreeMap<String, Vec<AtlasMetric>> = BTreeMap::new();
        for metric in &self.metrics {
            let district = metric
                .district()
                .map(|name| name.to_string())
                .unwrap_or_else(|| infer_district(&metric.name).to_string());
            buckets.entry(district).or_default().push(metric.clone());
        }
        buckets
            .into_iter()
            .map(|(name, metrics)| AtlasDistrict::from_metrics(name, metrics))
            .filter(|district| !district.metrics.is_empty())
            .collect()
    }
}

fn merge_metrics(dest: &mut Vec<AtlasMetric>, incoming: Vec<AtlasMetric>) {
    if incoming.is_empty() {
        return;
    }
    let mut index: BTreeMap<(String, Option<String>), usize> = dest
        .iter()
        .enumerate()
        .map(|(idx, metric)| ((metric.name.clone(), metric.district.clone()), idx))
        .collect();
    for mut metric in incoming {
        if !metric.value.is_finite() {
            continue;
        }
        metric.name = normalise_text(metric.name);
        if metric.name.is_empty() {
            continue;
        }
        metric.district = metric.district.map(|district| normalise_text(district));
        let key = (metric.name.clone(), metric.district.clone());
        if let Some(position) = index.get(&key).copied() {
            dest[position] = metric;
        } else {
            let position = dest.len();
            dest.push(metric);
            index.insert(key, position);
        }
    }
}

fn merge_notes(dest: &mut Vec<String>, incoming: Vec<String>) {
    if incoming.is_empty() {
        return;
    }
    let mut seen: BTreeSet<String> = dest.iter().cloned().collect();
    for note in incoming {
        if let Some(note) = normalise_optional_text(Some(note)) {
            if note.is_empty() {
                continue;
            }
            if seen.insert(note.clone()) {
                dest.push(note);
            } else if let Some(position) = dest.iter().position(|existing| existing == &note) {
                let note = dest.remove(position);
                dest.push(note);
            }
        }
    }
}

fn merge_concepts(dest: &mut Vec<ConceptAnnotation>, incoming: Vec<ConceptAnnotation>) {
    if incoming.is_empty() {
        return;
    }
    for mut concept in incoming {
        concept.term = normalise_text(concept.term);
        if concept.term.is_empty() {
            continue;
        }
        concept.rationale = normalise_optional_text(concept.rationale);
        if let Some(existing) = dest
            .iter_mut()
            .find(|current| current.term == concept.term && current.sense == concept.sense)
        {
            if concept.rationale.is_some() {
                existing.rationale = concept.rationale.clone();
            }
            continue;
        }
        dest.push(concept);
    }
}

#[derive(Clone, Debug)]
struct NoteAccumulator {
    limit: usize,
    queue: VecDeque<String>,
    seen: BTreeSet<String>,
}

impl NoteAccumulator {
    fn new(limit: usize) -> Self {
        Self {
            limit: limit.max(1),
            queue: VecDeque::new(),
            seen: BTreeSet::new(),
        }
    }

    fn extend(&mut self, notes: &[String]) {
        for note in notes {
            if let Some(note) = normalise_optional_text(Some(note.clone())) {
                if note.is_empty() {
                    continue;
                }
                if self.seen.insert(note.clone()) {
                    self.queue.push_back(note);
                } else if let Some(position) =
                    self.queue.iter().position(|existing| existing == &note)
                {
                    if let Some(existing) = self.queue.remove(position) {
                        self.queue.push_back(existing);
                    }
                }
                self.prune();
            }
        }
    }

    fn prune(&mut self) {
        while self.queue.len() > self.limit {
            if let Some(removed) = self.queue.pop_front() {
                self.seen.remove(&removed);
            }
        }
    }

    fn into_vec(self) -> Vec<String> {
        self.queue.into_iter().collect()
    }
}

/// Aggregated atlas route storing chronological frames for replay.
#[derive(Clone, Debug, Default)]
pub struct AtlasRoute {
    /// Chronological atlas frames retained in the route.
    pub frames: Vec<AtlasFrame>,
}

impl AtlasRoute {
    /// Creates a new, empty route.
    pub fn new() -> Self {
        Self { frames: Vec::new() }
    }

    /// Returns true when the route does not contain any frames.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Returns the number of frames stored in the route.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Returns the latest frame when available.
    pub fn latest(&self) -> Option<&AtlasFrame> {
        self.frames.last()
    }

    /// Appends a frame while keeping the total number of frames bounded.
    pub fn push_bounded(&mut self, frame: AtlasFrame, capacity: usize) {
        if frame.timestamp <= 0.0 {
            return;
        }
        self.frames.push(frame);
        if capacity > 0 && self.frames.len() > capacity {
            let overflow = self.frames.len() - capacity;
            self.frames.drain(0..overflow);
        }
    }

    /// Summarises the stored frames into district-level statistics and loopback hints.
    pub fn summary(&self) -> AtlasRouteSummary {
        let mut summary = AtlasRouteSummary {
            frames: self.frames.len(),
            loop_min: f32::INFINITY,
            ..AtlasRouteSummary::default()
        };
        if self.frames.is_empty() {
            return summary;
        }
        summary.latest_timestamp = self
            .frames
            .last()
            .map(|frame| frame.timestamp)
            .unwrap_or(0.0);
        let mut loop_total = 0.0;
        let mut loop_sq_total = 0.0;
        let mut loop_samples = 0usize;
        let mut first_loop = None;
        let mut last_loop = None;
        let mut district_map: BTreeMap<String, DistrictAccumulator> = BTreeMap::new();
        let mut concept_map: BTreeMap<(String, ConceptSense), ConceptAccumulator> = BTreeMap::new();
        let mut first_collapse = None;
        let mut first_z_signal = None;
        let mut note_accumulator = NoteAccumulator::new(16);
        for frame in &self.frames {
            let support = frame.loop_support.max(0.0);
            loop_total += support;
            loop_sq_total += support * support;
            loop_samples += 1;
            if first_loop.is_none() {
                first_loop = Some(support);
            }
            last_loop = Some(support);
            summary.loop_min = summary.loop_min.min(support);
            summary.loop_max = summary.loop_max.max(support);
            summary.total_notes += frame.notes.len();
            note_accumulator.extend(&frame.notes);
            if let Some(total) = frame.collapse_total {
                if total.is_finite() {
                    if first_collapse.is_none() {
                        first_collapse = Some(total);
                    }
                    summary.latest_collapse_total = Some(total);
                }
            }
            if let Some(z) = frame.z_signal {
                if z.is_finite() {
                    if first_z_signal.is_none() {
                        first_z_signal = Some(z);
                    }
                    summary.latest_z_signal = Some(z);
                }
            }
            if let Some(status) = frame.maintainer_status {
                summary.maintainer_status = Some(status);
            }
            if let Some(diagnostic) = frame.maintainer_diagnostic.as_ref() {
                if !diagnostic.is_empty() {
                    summary.maintainer_diagnostic = Some(diagnostic.clone());
                }
            }
            if let Some(clamp) = frame.suggested_max_scale {
                if clamp.is_finite() {
                    summary.suggested_max_scale = Some(clamp);
                }
            }
            if let Some(pressure) = frame.suggested_pressure {
                if pressure.is_finite() {
                    summary.suggested_pressure = Some(pressure);
                }
            }
            if let Some(script) = frame.script_hint.as_ref() {
                if !script.is_empty() {
                    summary.script_hint = Some(script.clone());
                }
            }
            if !frame.concepts.is_empty() {
                for concept in &frame.concepts {
                    let key = (concept.term.clone(), concept.sense);
                    let entry = concept_map.entry(key).or_default();
                    entry.mentions += 1;
                    if let Some(rationale) = concept.rationale.as_ref() {
                        entry.last_rationale = Some(rationale.clone());
                    }
                }
            }
            for district in frame.districts() {
                let entry = district_map.entry(district.name.clone()).or_default();
                entry.push(&district);
            }
        }
        if summary.frames > 0 {
            summary.mean_loop_support = loop_total / summary.frames as f32;
        }
        if loop_samples > 1 {
            let mean = loop_total / loop_samples as f32;
            let variance = (loop_sq_total / loop_samples as f32) - mean * mean;
            summary.loop_std = variance.max(0.0).sqrt();
        } else if loop_samples == 1 {
            summary.loop_std = 0.0;
        }
        summary.loop_total = loop_total;
        if summary.loop_min.is_infinite() {
            summary.loop_min = 0.0;
        }
        if let (Some(first), Some(last)) = (first_loop, last_loop) {
            summary.loop_trend = Some(last - first);
        }
        if let (Some(first), Some(latest)) = (first_collapse, summary.latest_collapse_total) {
            summary.collapse_trend = Some(latest - first);
        }
        if let (Some(first), Some(latest)) = (first_z_signal, summary.latest_z_signal) {
            summary.z_signal_trend = Some(latest - first);
        }
        summary.latest_notes = note_accumulator.into_vec();
        summary.districts = district_map
            .into_iter()
            .map(|(name, accumulator)| accumulator.into_summary(name))
            .filter(|district| district.coverage > 0)
            .collect();
        summary.districts.sort_by(|a, b| {
            b.coverage
                .cmp(&a.coverage)
                .then_with(|| a.name.cmp(&b.name))
        });
        summary.concept_pulses = concept_map
            .into_iter()
            .map(|((term, sense), accumulator)| ConceptPulse {
                term,
                sense,
                mentions: accumulator.mentions,
                last_rationale: accumulator.last_rationale,
            })
            .filter(|pulse| pulse.mentions > 0)
            .collect();
        summary.concept_pulses.sort_by(|a, b| {
            b.mentions
                .cmp(&a.mentions)
                .then_with(|| a.term.cmp(&b.term))
                .then_with(|| a.sense.label().cmp(b.sense.label()))
        });
        summary
    }
}

#[derive(Clone, Debug, Default)]
pub struct AtlasRouteSummary {
    /// Number of frames retained in the route at the time of the summary.
    pub frames: usize,
    /// Timestamp of the latest frame contributing to the summary.
    pub latest_timestamp: f32,
    /// Mean loop support accumulated across the route.
    pub mean_loop_support: f32,
    /// Standard deviation of loop support across the retained frames.
    pub loop_std: f32,
    /// Minimum loop support observed across the retained frames.
    pub loop_min: f32,
    /// Maximum loop support observed across the retained frames.
    pub loop_max: f32,
    /// Total loop support accumulated across the retained frames.
    pub loop_total: f32,
    /// Signed drift between the first and latest loop support samples.
    pub loop_trend: Option<f32>,
    /// Latest collapse total observed within the route.
    pub latest_collapse_total: Option<f32>,
    /// Signed drift between the first and latest collapse totals.
    pub collapse_trend: Option<f32>,
    /// Latest Z-space control signal observed within the route.
    pub latest_z_signal: Option<f32>,
    /// Signed drift between the first and latest Z-space control signals.
    pub z_signal_trend: Option<f32>,
    /// Last maintainer status routed through the atlas route.
    pub maintainer_status: Option<MaintainerStatus>,
    /// Last maintainer diagnostic routed through the atlas route.
    pub maintainer_diagnostic: Option<String>,
    /// Most recent clamp recommendation encountered in the route.
    pub suggested_max_scale: Option<f32>,
    /// Most recent Leech pressure recommendation encountered in the route.
    pub suggested_pressure: Option<f32>,
    /// Latest SpiralK script hint emitted by the route.
    pub script_hint: Option<String>,
    /// Total number of notes surfaced across the retained frames.
    pub total_notes: usize,
    /// Latest unique notes retained in chronological order.
    pub latest_notes: Vec<String>,
    /// District activity summaries accumulated across the route.
    pub districts: Vec<AtlasDistrictSummary>,
    /// Conceptual pulses aggregated across the retained frames.
    pub concept_pulses: Vec<ConceptPulse>,
}

impl AtlasRouteSummary {
    /// Returns true when the summary did not gather any frames.
    pub fn is_empty(&self) -> bool {
        self.frames == 0
    }

    /// Builds perspectives for all districts captured in the summary.
    pub fn perspectives(&self) -> Vec<AtlasPerspective> {
        self.districts
            .iter()
            .map(AtlasPerspective::from_district)
            .collect()
    }

    /// Builds a district perspective with an optional focus filter.
    pub fn perspective_for(&self, district: &str) -> Option<AtlasPerspective> {
        self.perspective_for_with_focus::<&str>(district, &[])
    }

    /// Builds a district perspective filtered by the provided metric prefixes.
    pub fn perspective_for_with_focus<S>(
        &self,
        district: &str,
        focus_prefixes: &[S],
    ) -> Option<AtlasPerspective>
    where
        S: AsRef<str>,
    {
        let mut perspective = self
            .districts
            .iter()
            .find(|summary| summary.name == district)
            .map(AtlasPerspective::from_district)?;
        perspective.apply_focus_filter(focus_prefixes);
        Some(perspective)
    }

    /// Returns the most recent note captured in the summary, if one exists.
    pub fn latest_note(&self) -> Option<&str> {
        self.latest_notes.last().map(|note| note.as_str())
    }

    /// Highlights the most dynamic metrics across districts as atlas beacons.
    pub fn beacons(&self, limit: usize) -> Vec<AtlasBeacon> {
        let mut beacons: Vec<AtlasBeacon> = self
            .districts
            .iter()
            .flat_map(|district| {
                district
                    .focus
                    .iter()
                    .filter_map(|focus| AtlasBeacon::from_focus(&district.name, focus))
            })
            .collect();
        beacons.sort_by(|a, b| {
            b.intensity
                .partial_cmp(&a.intensity)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.metric.cmp(&b.metric))
        });
        if limit > 0 && beacons.len() > limit {
            beacons.truncate(limit);
        }
        beacons
    }

    /// Fetches the beacon associated with a specific metric identifier.
    pub fn beacon_for(&self, metric: &str) -> Option<AtlasBeacon> {
        self.beacons(0)
            .into_iter()
            .find(|beacon| beacon.metric == metric)
    }
}

/// Perspective describing how a particular district can act on the atlas map.
#[derive(Clone, Debug, Default)]
pub struct AtlasPerspective {
    /// Name of the district represented by the perspective.
    pub district: String,
    /// Number of frames that contributed to the perspective metrics.
    pub coverage: usize,
    /// Mean district activity seen during the sampled frames.
    pub mean: f32,
    /// Latest district activity value.
    pub latest: f32,
    /// Signed change between the latest and earliest district activity.
    pub delta: f32,
    /// Average per-frame delta helping nodes gauge momentum.
    pub momentum: f32,
    /// Volatility derived from the district standard deviation.
    pub volatility: f32,
    /// Stability score (1.0 stable, 0.0 turbulent) derived from volatility.
    pub stability: f32,
    /// Metric focus areas that the district can use as anchors.
    pub focus: Vec<AtlasMetricFocus>,
    /// Human readable guidance tying the numbers back to action.
    pub guidance: String,
}

impl AtlasPerspective {
    fn from_district(district: &AtlasDistrictSummary) -> Self {
        let momentum = if district.coverage > 0 {
            district.delta / district.coverage as f32
        } else {
            0.0
        };
        let volatility = district.std_dev;
        let stability = 1.0 / (1.0 + volatility.abs());
        let mut perspective = Self {
            district: district.name.clone(),
            coverage: district.coverage,
            mean: district.mean,
            latest: district.latest,
            delta: district.delta,
            momentum,
            volatility,
            stability,
            focus: district.focus.clone(),
            guidance: String::new(),
        };
        perspective.rebuild_guidance();
        perspective
    }

    fn apply_focus_filter<S>(&mut self, prefixes: &[S])
    where
        S: AsRef<str>,
    {
        if prefixes.is_empty() {
            return;
        }
        let filtered: Vec<AtlasMetricFocus> = self
            .focus
            .iter()
            .filter(|metric| {
                prefixes
                    .iter()
                    .any(|prefix| metric.name.starts_with(prefix.as_ref()))
            })
            .cloned()
            .collect();
        if !filtered.is_empty() {
            self.focus = filtered;
            self.rebuild_guidance();
        }
    }

    fn rebuild_guidance(&mut self) {
        let orientation = if self.delta.abs() <= 1e-6 {
            "is steady"
        } else if self.delta.is_sign_positive() {
            "is rising"
        } else {
            "is cooling"
        };
        let tone = if self.stability >= 0.75 {
            "stable"
        } else if self.stability >= 0.45 {
            "adaptive"
        } else {
            "turbulent"
        };
        let highlight = if self.focus.is_empty() {
            "baseline flow".to_string()
        } else {
            self.focus
                .iter()
                .take(3)
                .map(|metric| format!("{} ({:+.3})", metric.name, metric.delta))
                .collect::<Vec<_>>()
                .join(", ")
        };
        self.guidance = format!(
            "{} district {} with {} stability (momentum {:+.3}, volatility {:.3}); focus on {}.",
            self.district, orientation, tone, self.momentum, self.volatility, highlight
        );
    }
}

#[derive(Clone, Debug, Default)]
pub struct AtlasMetricFocus {
    /// Metric identifier used within the district.
    pub name: String,
    /// Number of frames contributing to the metric statistics.
    pub coverage: usize,
    /// Average value recorded for the metric.
    pub mean: f32,
    /// Latest recorded value for the metric.
    pub latest: f32,
    /// Signed change between the latest and first observation.
    pub delta: f32,
    /// Per-step momentum computed from the accumulated delta.
    pub momentum: f32,
    /// Standard deviation of the metric across the observed frames.
    pub std_dev: f32,
}

impl AtlasMetricFocus {
    fn is_relevant(&self) -> bool {
        self.coverage > 0
    }
}

/// Direction describing how an atlas beacon is trending.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtlasBeaconTrend {
    /// Metric activity is rising across the retained frames.
    Rising,
    /// Metric activity is declining across the retained frames.
    Falling,
    /// Metric activity is effectively unchanged.
    Steady,
}

/// Highlighted metric that other nodes can treat as a navigational beacon.
#[derive(Clone, Debug)]
pub struct AtlasBeacon {
    /// District that produced the beacon metric.
    pub district: String,
    /// Metric identifier flagged as a beacon.
    pub metric: String,
    /// Number of frames that contributed to the beacon statistics.
    pub coverage: usize,
    /// Average value recorded for the beacon metric.
    pub mean: f32,
    /// Latest recorded value for the beacon metric.
    pub latest: f32,
    /// Signed change between the latest and first observation.
    pub delta: f32,
    /// Per-frame momentum computed from the accumulated delta.
    pub momentum: f32,
    /// Standard deviation recorded for the beacon metric.
    pub volatility: f32,
    /// Strength used to rank beacons (higher means more dynamic).
    pub intensity: f32,
    /// Direction describing how the beacon is trending.
    pub trend: AtlasBeaconTrend,
    /// Narrative describing how the beacon should be interpreted.
    pub narrative: String,
}

impl AtlasBeacon {
    fn from_focus(district: &str, focus: &AtlasMetricFocus) -> Option<Self> {
        if focus.coverage == 0 {
            return None;
        }
        let delta = if focus.delta.is_finite() {
            focus.delta
        } else {
            0.0
        };
        let momentum = if focus.momentum.is_finite() {
            focus.momentum
        } else {
            0.0
        };
        let volatility = if focus.std_dev.is_finite() {
            focus.std_dev.max(0.0)
        } else {
            0.0
        };
        let intensity = delta.abs().max(momentum.abs());
        if intensity <= 1e-6 {
            return None;
        }
        let trend = if delta.abs() <= 1e-6 {
            AtlasBeaconTrend::Steady
        } else if delta.is_sign_positive() {
            AtlasBeaconTrend::Rising
        } else {
            AtlasBeaconTrend::Falling
        };
        let volatility_note = if volatility <= 1e-6 {
            "with calm variance"
        } else if volatility < 0.25 {
            "with mild variance"
        } else if volatility < 0.75 {
            "with adaptive variance"
        } else {
            "with turbulent variance"
        };
        let tempo_note = if momentum.abs() <= 1e-6 {
            "flat momentum"
        } else {
            "momentum"
        };
        let narrative = format!(
            "{district}::{metric} {orientation} (Δ {:+.3}, {tempo} {:+.3}, σ {:.3}) {variance}.",
            delta,
            momentum,
            volatility,
            district = district,
            metric = focus.name,
            orientation = match trend {
                AtlasBeaconTrend::Rising => "is surging",
                AtlasBeaconTrend::Falling => "is easing",
                AtlasBeaconTrend::Steady => "holds steady",
            },
            tempo = tempo_note,
            variance = volatility_note,
        );
        Some(Self {
            district: district.to_string(),
            metric: focus.name.clone(),
            coverage: focus.coverage,
            mean: focus.mean,
            latest: focus.latest,
            delta,
            momentum,
            volatility,
            intensity,
            trend,
            narrative,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct AtlasDistrictSummary {
    /// Human readable name of the district.
    pub name: String,
    /// Number of frames contributing to the district statistics.
    pub coverage: usize,
    /// Average district activity across the covered frames.
    pub mean: f32,
    /// Latest district activity value encountered in the route.
    pub latest: f32,
    /// Minimum district activity observed in the route.
    pub min: f32,
    /// Maximum district activity observed in the route.
    pub max: f32,
    /// Signed change between the latest and first observation.
    pub delta: f32,
    /// Standard deviation of the district activity across observations.
    pub std_dev: f32,
    /// Focus metrics that describe the district activity in more detail.
    pub focus: Vec<AtlasMetricFocus>,
}

#[derive(Clone, Debug, Default)]
struct MetricAccumulator {
    sum: f32,
    sum_sq: f32,
    first: Option<f32>,
    last: f32,
    count: usize,
}

impl MetricAccumulator {
    fn push(&mut self, value: f32) {
        if !value.is_finite() {
            return;
        }
        self.sum += value;
        self.sum_sq += value * value;
        self.count += 1;
        if self.first.is_none() {
            self.first = Some(value);
        }
        self.last = value;
    }

    fn into_focus(self, name: String) -> AtlasMetricFocus {
        if self.count == 0 {
            return AtlasMetricFocus {
                name,
                ..Default::default()
            };
        }
        let mean = self.sum / self.count as f32;
        let variance = (self.sum_sq / self.count as f32) - mean * mean;
        let first = self.first.unwrap_or(self.last);
        let delta = self.last - first;
        let momentum = if self.count > 0 {
            delta / self.count as f32
        } else {
            0.0
        };
        AtlasMetricFocus {
            name,
            coverage: self.count,
            mean,
            latest: self.last,
            delta,
            momentum,
            std_dev: variance.max(0.0).sqrt(),
        }
    }
}

#[derive(Clone, Debug, Default)]
struct DistrictAccumulator {
    sum: f32,
    sum_sq: f32,
    min: f32,
    max: f32,
    first: Option<f32>,
    last: f32,
    count: usize,
    metrics: BTreeMap<String, MetricAccumulator>,
}

impl DistrictAccumulator {
    #[allow(
        dead_code,
        reason = "Factory retained for future incremental telemetry wiring"
    )]
    fn new() -> Self {
        Self {
            sum: 0.0,
            sum_sq: 0.0,
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            first: None,
            last: 0.0,
            count: 0,
            metrics: BTreeMap::new(),
        }
    }

    fn push(&mut self, district: &AtlasDistrict) {
        let value = district.mean;
        if !value.is_finite() {
            return;
        }
        self.sum += value;
        self.sum_sq += value * value;
        self.count += 1;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        if self.first.is_none() {
            self.first = Some(value);
        }
        self.last = value;
        for metric in &district.metrics {
            if !metric.value.is_finite() {
                continue;
            }
            let entry = self.metrics.entry(metric.name.clone()).or_default();
            entry.push(metric.value);
        }
    }

    fn into_summary(self, name: String) -> AtlasDistrictSummary {
        if self.count == 0 {
            return AtlasDistrictSummary {
                name,
                ..Default::default()
            };
        }
        let mean = self.sum / self.count as f32;
        let variance = (self.sum_sq / self.count as f32) - mean * mean;
        let first = self.first.unwrap_or(self.last);
        let mut focus: Vec<AtlasMetricFocus> = self
            .metrics
            .into_iter()
            .map(|(name, accumulator)| accumulator.into_focus(name))
            .filter(AtlasMetricFocus::is_relevant)
            .collect();
        focus.sort_by(|a, b| {
            b.coverage
                .cmp(&a.coverage)
                .then_with(|| {
                    b.latest
                        .partial_cmp(&a.latest)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.name.cmp(&b.name))
        });
        if focus.len() > 8 {
            focus.truncate(8);
        }
        AtlasDistrictSummary {
            name,
            coverage: self.count,
            mean,
            latest: self.last,
            min: if self.min.is_finite() { self.min } else { mean },
            max: if self.max.is_finite() { self.max } else { mean },
            delta: self.last - first,
            std_dev: variance.max(0.0).sqrt(),
            focus,
        }
    }
}

/// Aggregated pulse describing how a concept has been invoked across frames.
#[derive(Clone, Debug)]
pub struct ConceptPulse {
    /// Term that was annotated (for example, "qualia").
    pub term: String,
    /// Conceptual sense associated with the annotations.
    pub sense: ConceptSense,
    /// Number of annotations encountered for the pair.
    pub mentions: usize,
    /// Most recent rationale that accompanied the annotations, when present.
    pub last_rationale: Option<String>,
}

#[derive(Clone, Debug, Default)]
struct ConceptAccumulator {
    mentions: usize,
    last_rationale: Option<String>,
}

impl ConceptAccumulator {
    #[allow(
        dead_code,
        reason = "Factory retained for future incremental telemetry wiring"
    )]
    fn new() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod frame_tests {
    use super::*;

    #[test]
    fn merging_fragments_updates_metric_values() {
        let mut fragment = AtlasFragment::new();
        fragment.timestamp = Some(1.0);
        fragment.push_metric("psi.total", 1.0);
        let mut frame = AtlasFrame::from_fragment(fragment).expect("frame");

        let mut update = AtlasFragment::new();
        update.timestamp = Some(2.0);
        update.push_metric("psi.total", 2.5);
        update.push_metric("psi.aux", 3.0);
        frame.merge_fragment(update);

        assert_eq!(frame.metrics.len(), 2);
        assert_eq!(frame.metric_value("psi.total"), Some(2.5));
        let names: Vec<&str> = frame
            .metrics_with_prefix("psi")
            .into_iter()
            .map(|metric| metric.name.as_str())
            .collect();
        assert_eq!(names, vec!["psi.aux", "psi.total"]);
    }
}

#[cfg(test)]
mod summary_tests {
    use super::*;

    #[test]
    fn route_summary_tracks_loop_envelope_and_notes() {
        let mut first = AtlasFragment::new();
        first.timestamp = Some(1.0);
        first.loop_support = Some(1.0);
        first.push_note("alpha");
        let frame_a = AtlasFrame::from_fragment(first).expect("frame a");

        let mut second = AtlasFragment::new();
        second.timestamp = Some(2.0);
        second.loop_support = Some(3.0);
        second.push_note("beta");
        second.push_note("alpha");
        let frame_b = AtlasFrame::from_fragment(second).expect("frame b");

        let mut route = AtlasRoute::new();
        route.push_bounded(frame_a, 8);
        route.push_bounded(frame_b, 8);

        let summary = route.summary();
        assert_eq!(summary.frames, 2);
        assert!((summary.loop_total - 4.0).abs() <= f32::EPSILON);
        assert_eq!(summary.loop_min, 1.0);
        assert_eq!(summary.loop_max, 3.0);
        assert_eq!(summary.loop_trend, Some(2.0));
        assert_eq!(summary.total_notes, 3);
        assert_eq!(
            summary.latest_notes,
            vec!["beta".to_string(), "alpha".to_string()]
        );
        assert_eq!(summary.latest_note(), Some("alpha"));
    }
}

#[cfg(test)]
mod concept_tests {
    use super::*;

    #[test]
    fn concept_annotations_keep_fragment_alive() {
        let mut fragment = AtlasFragment::new();
        fragment.annotate_qualia(
            ConceptSense::QualiaNagelSubjectivity,
            Some("subjective vantage guard"),
        );
        let frame = AtlasFrame::from_fragment(fragment).expect("frame");
        assert!(frame.timestamp > 0.0);
        assert_eq!(frame.concepts.len(), 1);
        assert_eq!(
            frame.concepts[0].sense,
            ConceptSense::QualiaNagelSubjectivity
        );
        assert_eq!(
            frame.concepts[0].rationale.as_deref(),
            Some("subjective vantage guard")
        );
    }

    #[test]
    fn summary_accumulates_concept_pulses() {
        let mut fragment = AtlasFragment::new();
        fragment.timestamp = Some(1.0);
        fragment.push_concept(ConceptAnnotation::with_rationale(
            "qualia",
            ConceptSense::QualiaChalmersHardProblem,
            "charting the hard problem",
        ));
        let frame = AtlasFrame::from_fragment(fragment).unwrap();
        let mut route = AtlasRoute::new();
        route.push_bounded(frame, 8);
        let summary = route.summary();
        assert_eq!(summary.concept_pulses.len(), 1);
        let pulse = &summary.concept_pulses[0];
        assert_eq!(pulse.term, "qualia");
        assert_eq!(pulse.sense, ConceptSense::QualiaChalmersHardProblem);
        assert_eq!(pulse.mentions, 1);
        assert_eq!(
            pulse.last_rationale.as_deref(),
            Some("charting the hard problem")
        );
        assert_eq!(pulse.sense.label(), "qualia.chalmers_hard");
        assert!(pulse.sense.description().contains("hard problem"));
    }
}

/// Atlas district representing a logical SpiralTorch layer.
#[derive(Clone, Debug, Default)]
pub struct AtlasDistrict {
    /// Human readable name of the district.
    pub name: String,
    /// Metrics that contributed to the district activity.
    pub metrics: Vec<AtlasMetric>,
    /// Mean activity value across the district metrics.
    pub mean: f32,
    /// Range spanned by the district metrics.
    pub span: f32,
}

impl AtlasDistrict {
    /// Builds a district from aggregated metrics.
    fn from_metrics(name: String, metrics: Vec<AtlasMetric>) -> Self {
        let mut mean = 0.0;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let len = metrics.len() as f32;
        if len > 0.0 {
            for metric in &metrics {
                mean += metric.value;
                min = min.min(metric.value);
                max = max.max(metric.value);
            }
            mean /= len;
        }
        let span = if metrics.len() > 1 { max - min } else { 0.0 };
        Self {
            name,
            metrics,
            mean,
            span,
        }
    }
}

fn infer_district(name: &str) -> &'static str {
    let lower = name.to_ascii_lowercase();
    let token = lower.split(['.', ':', '/', '-']).next().unwrap_or("");
    match token {
        "py" | "python" | "bindings" | "session" | "timeline" | "config" | "psychoid" => "Surface",
        "trainer" | "maintainer" | "atlas" | "loop" | "chrono" | "policy" | "resonator"
        | "softlogic" | "psi" | "desire" => "Concourse",
        "tensor" | "backend" | "core" | "z" | "collapse" | "geometry" | "kdsl" | "realgrad" => {
            "Substrate"
        }
        _ => "Unknown",
    }
}
