// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// ============================================================================
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
// ============================================================================

use super::atlas::{AtlasFragment, AtlasFrame, AtlasRoute, AtlasRouteSummary};
use super::dashboard::{DashboardFrame, DashboardRing};
use super::maintainer::MaintainerReport;
use super::xai_report::AttributionReport;
#[cfg(any(feature = "psi", feature = "psychoid"))]
use once_cell::sync::Lazy;
#[cfg(feature = "psi")]
use parking_lot::{ReentrantMutex, ReentrantMutexGuard};
#[cfg(feature = "psi")]
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
#[cfg(feature = "psi")]
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::theory::zpulse::{ZScale, ZSource};

use super::chrono::ChronoLoopSignal;
#[cfg(feature = "psi")]
use super::psi::{PsiComponent, PsiEvent, PsiReading, PsiSpiralAdvisory, PsiSpiralTuning};
#[cfg(feature = "psychoid")]
use super::psychoid::PsychoidReading;
#[cfg(feature = "collapse")]
use crate::engine::collapse_drive::DriveCmd;
use crate::ops::realgrad::GradientSummary;
use std::collections::VecDeque;

#[cfg(feature = "psi")]
static LAST_PSI: Lazy<RwLock<Option<PsiReading>>> = Lazy::new(|| RwLock::new(None));

#[cfg(feature = "psi")]
static LAST_PSI_EVENTS: Lazy<RwLock<Vec<PsiEvent>>> = Lazy::new(|| RwLock::new(Vec::new()));

#[cfg(feature = "psi")]
static LAST_PSI_SPIRAL: Lazy<RwLock<Option<PsiSpiralAdvisory>>> = Lazy::new(|| RwLock::new(None));

#[cfg(feature = "psi")]
static LAST_PSI_SPIRAL_TUNING: Lazy<RwLock<Option<PsiSpiralTuning>>> =
    Lazy::new(|| RwLock::new(None));

#[cfg(feature = "psi")]
const PSI_COMPONENTS: [PsiComponent; 6] = [
    PsiComponent::LOSS,
    PsiComponent::GRAD_NORM,
    PsiComponent::UPDATE_RATIO,
    PsiComponent::ACT_DRIFT,
    PsiComponent::ATTN_ENTROPY,
    PsiComponent::BAND_ENERGY,
];

#[cfg(feature = "psi")]
static PSI_TELEMETRY_LOCK: OnceLock<ReentrantMutex<()>> = OnceLock::new();

#[cfg(feature = "psi")]
fn psi_lock() -> &'static ReentrantMutex<()> {
    PSI_TELEMETRY_LOCK.get_or_init(|| ReentrantMutex::new(()))
}

#[cfg(feature = "psi")]
#[must_use]
pub fn psi_telemetry_guard() -> ReentrantMutexGuard<'static, ()> {
    psi_lock().lock()
}

static CONFIG_DIFF_EVENTS: OnceLock<RwLock<Vec<ConfigDiffEvent>>> = OnceLock::new();

fn config_events_cell() -> &'static RwLock<Vec<ConfigDiffEvent>> {
    CONFIG_DIFF_EVENTS.get_or_init(|| RwLock::new(Vec::new()))
}

/// Trait implemented by observers that want to be notified whenever the
/// aggregated atlas frame changes.
pub trait AtlasFrameObserver: Send + Sync {
    /// Called after the atlas frame has been updated.
    fn on_frame(&self, frame: &AtlasFrame);
}

impl<F> AtlasFrameObserver for F
where
    F: Fn(&AtlasFrame) + Send + Sync,
{
    fn on_frame(&self, frame: &AtlasFrame) {
        (self)(frame);
    }
}

#[derive(Clone)]
struct AtlasFrameObserverEntry {
    id: usize,
    callback: Arc<dyn AtlasFrameObserver>,
}

static ATLAS_FRAME_OBSERVERS: OnceLock<RwLock<Vec<AtlasFrameObserverEntry>>> = OnceLock::new();
static NEXT_ATLAS_OBSERVER_ID: AtomicUsize = AtomicUsize::new(1);

fn atlas_observers_cell() -> &'static RwLock<Vec<AtlasFrameObserverEntry>> {
    ATLAS_FRAME_OBSERVERS.get_or_init(|| RwLock::new(Vec::new()))
}

/// Guard that keeps an atlas frame observer registered until dropped.
#[must_use]
pub struct AtlasFrameSubscription {
    id: usize,
}

impl Drop for AtlasFrameSubscription {
    fn drop(&mut self) {
        if let Ok(mut guard) = atlas_observers_cell().write() {
            guard.retain(|entry| entry.id != self.id);
        }
    }
}

/// Registers a new observer that will be notified after each atlas frame update.
pub fn register_atlas_frame_observer(
    observer: impl AtlasFrameObserver + 'static,
) -> AtlasFrameSubscription {
    let id = NEXT_ATLAS_OBSERVER_ID.fetch_add(1, Ordering::Relaxed);
    let callback: Arc<dyn AtlasFrameObserver> = Arc::new(observer);
    {
        let entry = AtlasFrameObserverEntry {
            id,
            callback: callback.clone(),
        };
        match atlas_observers_cell().write() {
            Ok(mut guard) => guard.push(entry),
            Err(poisoned) => poisoned.into_inner().push(entry),
        }
    }
    if let Some(frame) = get_atlas_frame() {
        callback.on_frame(&frame);
    }
    AtlasFrameSubscription { id }
}

fn notify_atlas_frame_observers(frame: &AtlasFrame) {
    let callbacks: Vec<Arc<dyn AtlasFrameObserver>> = match atlas_observers_cell().read() {
        Ok(guard) => guard.iter().map(|entry| entry.callback.clone()).collect(),
        Err(poisoned) => poisoned
            .into_inner()
            .iter()
            .map(|entry| entry.callback.clone())
            .collect(),
    };
    for callback in callbacks {
        callback.on_frame(frame);
    }
}

/// Configuration layer that produced a diff event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfigLayer {
    Base,
    Site,
    Run,
}

impl fmt::Display for ConfigLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigLayer::Base => write!(f, "base"),
            ConfigLayer::Site => write!(f, "site"),
            ConfigLayer::Run => write!(f, "run"),
        }
    }
}

/// Diff emitted while applying layered configuration files.
#[derive(Clone, Debug)]
pub struct ConfigDiffEvent {
    pub layer: ConfigLayer,
    pub path: String,
    pub previous: Option<Value>,
    pub current: Option<Value>,
}

#[cfg(feature = "psi")]
pub fn set_last_psi(reading: &PsiReading) {
    {
        let _guard = psi_lock().lock();
        if let Ok(mut guard) = LAST_PSI.write() {
            *guard = Some(reading.clone());
        }
    }
    let mut fragment = AtlasFragment::new();
    if let Some(timestamp) = psi_step_timestamp(reading.step) {
        fragment.timestamp = Some(timestamp);
    }
    populate_psi_breakdown(&mut fragment, "psi", &reading.breakdown);
    fragment.push_metric("psi.total", reading.total);
    fragment.push_note(format!("psi.step:{}", reading.step));
    merge_atlas_fragment(fragment);
}

#[cfg(feature = "psi")]
pub fn get_last_psi() -> Option<PsiReading> {
    let _guard = psi_lock().lock();
    LAST_PSI
        .read()
        .ok()
        .and_then(|guard| guard.as_ref().cloned())
}

#[cfg(feature = "psi")]
pub fn clear_last_psi() {
    let _guard = psi_lock().lock();
    if let Ok(mut guard) = LAST_PSI.write() {
        *guard = None;
    }
}

#[cfg(feature = "psi")]
pub fn set_last_psi_events(events: &[PsiEvent]) {
    {
        let _guard = psi_lock().lock();
        if let Ok(mut guard) = LAST_PSI_EVENTS.write() {
            guard.clear();
            guard.extend(events.iter().cloned());
        }
    }
    if events.is_empty() {
        return;
    }
    let mut fragment = AtlasFragment::new();
    if let Some(step) = annotate_psi_events(&mut fragment, "psi", events) {
        fragment.timestamp = psi_step_timestamp(step);
    }
    merge_atlas_fragment(fragment);
}

#[cfg(feature = "psi")]
pub fn get_last_psi_events() -> Vec<PsiEvent> {
    let _guard = psi_lock().lock();
    LAST_PSI_EVENTS
        .read()
        .map(|guard| guard.clone())
        .unwrap_or_default()
}

#[cfg(feature = "psi")]
pub fn clear_last_psi_events() {
    let _guard = psi_lock().lock();
    if let Ok(mut guard) = LAST_PSI_EVENTS.write() {
        guard.clear();
    }
}

#[cfg(feature = "psi")]
pub fn set_last_psi_spiral(advisory: &PsiSpiralAdvisory) {
    {
        let _guard = psi_lock().lock();
        if let Ok(mut guard) = LAST_PSI_SPIRAL.write() {
            *guard = Some(advisory.clone());
        }
    }
    let mut fragment = AtlasFragment::new();
    fragment.push_metric("psi.spiral.mu_eff0", advisory.mu_eff0);
    fragment.push_metric("psi.spiral.alpha3", advisory.alpha3);
    fragment.push_metric(
        "psi.spiral.audit_container_gap",
        advisory.audit_container_gap,
    );
    fragment.push_metric("psi.spiral.audit_cluster", advisory.audit_cluster);
    fragment.push_metric("psi.spiral.container_cluster", advisory.container_cluster);
    fragment.push_metric("psi.spiral.stability_score", advisory.stability_score());
    fragment.push_metric(
        "psi.spiral.audit_overbias",
        bool_as_metric(advisory.audit_overbias()),
    );
    fragment.push_note(format!("psi.spiral.regime:{:?}", advisory.regime));
    merge_atlas_fragment(fragment);
}

#[cfg(feature = "psi")]
pub fn get_last_psi_spiral() -> Option<PsiSpiralAdvisory> {
    let _guard = psi_lock().lock();
    LAST_PSI_SPIRAL
        .read()
        .ok()
        .and_then(|guard| guard.as_ref().cloned())
}

#[cfg(feature = "psi")]
pub fn set_last_psi_spiral_tuning(tuning: &PsiSpiralTuning) {
    {
        let _guard = psi_lock().lock();
        if let Ok(mut guard) = LAST_PSI_SPIRAL_TUNING.write() {
            *guard = Some(tuning.clone());
        }
    }
    let mut fragment = AtlasFragment::new();
    fragment.push_metric("psi.spiral.tuning.stability", tuning.stability_score);
    let mut required = 0u32;
    for &component in PSI_COMPONENTS.iter() {
        if tuning.required_components.contains(component) {
            required += 1;
            fragment.push_note(format!(
                "psi.spiral.tuning.required:{}",
                component.to_string()
            ));
        }
        if let Some(delta) = tuning.weight_increments.get(&component) {
            fragment.push_metric(format!("psi.spiral.tuning.weight.{}", component), *delta);
        }
        if let Some(delta) = tuning.threshold_shifts.get(&component) {
            fragment.push_metric(format!("psi.spiral.tuning.threshold.{}", component), *delta);
        }
    }
    fragment.push_metric("psi.spiral.tuning.required", required as f32);
    fragment.push_metric(
        "psi.spiral.tuning.neutral",
        bool_as_metric(tuning.is_neutral()),
    );
    merge_atlas_fragment(fragment);
}

#[cfg(feature = "psi")]
pub fn get_last_psi_spiral_tuning() -> Option<PsiSpiralTuning> {
    let _guard = psi_lock().lock();
    LAST_PSI_SPIRAL_TUNING
        .read()
        .ok()
        .and_then(|guard| guard.as_ref().cloned())
}

/// Records the most recent configuration diff events produced when loading
/// layered configuration files.
pub fn record_config_events(events: &[ConfigDiffEvent]) {
    if let Ok(mut guard) = config_events_cell().write() {
        guard.clear();
        guard.extend(events.iter().cloned());
    }
    if events.is_empty() {
        return;
    }
    let fragment = fragment_from_config_events(events);
    merge_atlas_fragment(fragment);
}

/// Returns the last recorded configuration diff events.
pub fn get_config_events() -> Vec<ConfigDiffEvent> {
    config_events_cell()
        .read()
        .map(|guard| guard.clone())
        .unwrap_or_default()
}

#[cfg(feature = "psychoid")]
static LAST_PSYCHOID: Lazy<RwLock<Option<PsychoidReading>>> = Lazy::new(|| RwLock::new(None));

#[cfg(feature = "psychoid")]
pub fn set_last_psychoid(reading: &PsychoidReading) {
    if let Ok(mut guard) = LAST_PSYCHOID.write() {
        *guard = Some(reading.clone());
    }
    let fragment = fragment_from_psychoid_reading(reading);
    merge_atlas_fragment(fragment);
}

#[cfg(feature = "psychoid")]
pub fn get_last_psychoid() -> Option<PsychoidReading> {
    LAST_PSYCHOID
        .read()
        .ok()
        .and_then(|guard| guard.as_ref().cloned())
}

/// Latest SoftLogic-derived telemetry that has been fed back into the "Z" control space.
#[derive(Debug, Clone, Default)]
pub struct SoftlogicZFeedback {
    /// Aggregate PSI total used when the sample was recorded.
    pub psi_total: f32,
    /// Weighted loss that triggered the feedback pulse.
    pub weighted_loss: f32,
    /// Above/Here/Beneath energy tuple at the moment of sampling.
    pub band_energy: (f32, f32, f32),
    /// Drift term captured from the gradient bands.
    pub drift: f32,
    /// Normalized control signal in the Z space. Positive values bias Above, negative bias Beneath.
    pub z_signal: f32,
    /// Optional log-scale tag attached to the fused pulse that produced the feedback.
    pub scale: Option<ZScale>,
    /// Event tags emitted while producing the fused control signal.
    pub events: Vec<String>,
    /// Attribution weights per contributing Z source.
    pub attributions: Vec<(ZSource, f32)>,
    /// Optional elliptic geometry summary captured while producing the feedback pulse.
    pub elliptic: Option<SoftlogicEllipticSample>,
}

/// Summary of elliptic curvature sampled while producing a SoftLogic feedback pulse.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SoftlogicEllipticSample {
    pub curvature_radius: f32,
    pub geodesic_radius: f32,
    pub normalized_radius: f32,
    pub spin_alignment: f32,
    pub sheet_index: u32,
    pub sheet_position: f32,
    pub normal_bias: f32,
    pub sheet_count: u32,
    pub topological_sector: u32,
    pub homology_index: u32,
    pub rotor_field: [f32; 3],
    pub flow_vector: [f32; 3],
    pub curvature_tensor: [[f32; 3]; 3],
    pub resonance_heat: f32,
    pub noise_density: f32,
    pub quaternion: [f32; 4],
    pub rotation: [f32; 9],
}

impl SoftlogicZFeedback {
    /// Returns true when the feedback contains the provided event tag.
    pub fn has_event(&self, tag: &str) -> bool {
        self.events.iter().any(|event| event == tag)
    }

    /// Iterates over all event tags contained in the feedback record.
    pub fn event_tags(&self) -> impl Iterator<Item = &str> {
        self.events.iter().map(|event| event.as_str())
    }

    /// Appends an event tag to the feedback.
    pub fn push_event<S: Into<String>>(&mut self, tag: S) {
        self.events.push(tag.into());
    }

    /// Clears the current event tags and replaces them with the provided iterator.
    pub fn set_events<I, S>(&mut self, events: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.events.clear();
        self.events.extend(events.into_iter().map(Into::into));
    }

    /// Replaces the attribution weights with the provided iterator.
    pub fn set_attributions<I>(&mut self, attributions: I)
    where
        I: IntoIterator<Item = (ZSource, f32)>,
    {
        self.attributions.clear();
        self.attributions.extend(
            attributions
                .into_iter()
                .map(|(source, weight)| (source, weight.max(0.0))),
        );
    }

    /// Returns the parsed Z-space region descriptor when elliptic telemetry is present.
    pub fn region_descriptor(
        &self,
    ) -> Option<crate::telemetry::zspace_region::ZSpaceRegionDescriptor> {
        self.elliptic
            .as_ref()
            .map(crate::telemetry::zspace_region::ZSpaceRegionDescriptor::from)
    }
}

static LAST_SOFTLOGIC_Z: OnceLock<RwLock<Option<SoftlogicZFeedback>>> = OnceLock::new();
static LAST_REGION_REPORT: OnceLock<RwLock<Option<AttributionReport>>> = OnceLock::new();

fn softlogic_z_cell() -> &'static RwLock<Option<SoftlogicZFeedback>> {
    LAST_SOFTLOGIC_Z.get_or_init(|| RwLock::new(None))
}

fn region_report_cell() -> &'static RwLock<Option<AttributionReport>> {
    LAST_REGION_REPORT.get_or_init(|| RwLock::new(None))
}

#[cfg(feature = "psi")]
#[derive(Clone, Debug)]
pub struct DesireWeightsTelemetry {
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
    pub lambda: f32,
}

#[cfg(feature = "psi")]
#[derive(Clone, Debug)]
pub enum DesirePhaseTelemetry {
    Observation,
    Injection,
    Integration,
}

#[cfg(feature = "psi")]
#[derive(Clone, Debug)]
pub struct DesireAvoidanceTelemetry {
    pub tokens: Vec<usize>,
    pub scores: Vec<f32>,
}

#[cfg(feature = "psi")]
#[derive(Clone, Debug)]
pub struct DesireTriggerTelemetry {
    pub mean_penalty: f32,
    pub mean_entropy: f32,
    pub temperature: f32,
    pub samples: usize,
}

#[cfg(feature = "psi")]
#[derive(Clone, Debug)]
pub struct DesireStepTelemetry {
    pub timestamp: SystemTime,
    pub entropy: f32,
    pub temperature: f32,
    pub hypergrad_penalty: f32,
    pub avoidance_energy: f32,
    pub logit_energy: f32,
    pub phase: DesirePhaseTelemetry,
    pub weights: DesireWeightsTelemetry,
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
    pub lambda: f32,
    pub avoidance: Option<DesireAvoidanceTelemetry>,
    pub trigger: Option<DesireTriggerTelemetry>,
    pub trigger_emitted: bool,
    pub psi_total: Option<f32>,
    pub psi_breakdown: HashMap<PsiComponent, f32>,
    pub psi_events: Vec<PsiEvent>,
    pub z_feedback: Option<SoftlogicZFeedback>,
}

#[cfg(feature = "psi")]
static LAST_DESIRE_STEP: OnceLock<RwLock<Option<DesireStepTelemetry>>> = OnceLock::new();

#[cfg(feature = "psi")]
fn desire_step_cell() -> &'static RwLock<Option<DesireStepTelemetry>> {
    LAST_DESIRE_STEP.get_or_init(|| RwLock::new(None))
}

static ATLAS_FRAME: OnceLock<RwLock<Option<AtlasFrame>>> = OnceLock::new();
static ATLAS_ROUTE: OnceLock<RwLock<VecDeque<AtlasFrame>>> = OnceLock::new();
static DASHBOARD_FRAMES: OnceLock<RwLock<DashboardRing>> = OnceLock::new();
static LAST_MAINTAINER_REPORT: OnceLock<RwLock<Option<MaintainerReport>>> = OnceLock::new();

fn maintainer_cell() -> &'static RwLock<Option<MaintainerReport>> {
    LAST_MAINTAINER_REPORT.get_or_init(|| RwLock::new(None))
}

fn atlas_cell() -> &'static RwLock<Option<AtlasFrame>> {
    ATLAS_FRAME.get_or_init(|| RwLock::new(None))
}

fn atlas_route_cell() -> &'static RwLock<VecDeque<AtlasFrame>> {
    ATLAS_ROUTE.get_or_init(|| RwLock::new(VecDeque::new()))
}

const ATLAS_ROUTE_CAPACITY: usize = 24;

fn dashboard_ring() -> &'static RwLock<DashboardRing> {
    DASHBOARD_FRAMES.get_or_init(|| RwLock::new(DashboardRing::new(64)))
}

/// Pushes a dashboard frame onto the rolling ring buffer.
pub fn push_dashboard_frame(frame: DashboardFrame) {
    if let Ok(mut guard) = dashboard_ring().write() {
        guard.push(frame);
    }
}

/// Returns the latest dashboard frame if one has been recorded.
pub fn latest_dashboard_frame() -> Option<DashboardFrame> {
    dashboard_ring()
        .read()
        .ok()
        .and_then(|guard| guard.latest().cloned())
}

/// Returns up to `limit` frames from the ring, newest first.
pub fn snapshot_dashboard_frames(limit: usize) -> Vec<DashboardFrame> {
    if limit == 0 {
        return Vec::new();
    }
    dashboard_ring()
        .read()
        .map(|guard| guard.iter().rev().take(limit).cloned().collect())
        .unwrap_or_default()
}

fn push_atlas_route(frame: &AtlasFrame) {
    if frame.timestamp <= 0.0 {
        return;
    }
    if let Ok(mut guard) = atlas_route_cell().write() {
        guard.push_back(frame.clone());
        while guard.len() > ATLAS_ROUTE_CAPACITY {
            guard.pop_front();
        }
    }
}

/// Replaces the stored atlas frame with the provided snapshot.
pub fn set_atlas_frame(frame: AtlasFrame) {
    if let Ok(mut guard) = atlas_cell().write() {
        *guard = Some(frame.clone());
    }
    push_atlas_route(&frame);
    notify_atlas_frame_observers(&frame);
}

/// Clears the stored atlas frame and route history.
pub fn clear_atlas() {
    if let Ok(mut guard) = atlas_cell().write() {
        *guard = None;
    }
    clear_atlas_route();
}

/// Clears the stored atlas route.
pub fn clear_atlas_route() {
    if let Ok(mut guard) = atlas_route_cell().write() {
        guard.clear();
    }
}

/// Returns the latest atlas frame if one has been recorded.
pub fn get_atlas_frame() -> Option<AtlasFrame> {
    atlas_cell()
        .read()
        .ok()
        .and_then(|guard| guard.as_ref().cloned())
}

/// Returns the chronological atlas route up to the requested limit.
pub fn get_atlas_route(limit: Option<usize>) -> AtlasRoute {
    let mut route = AtlasRoute::new();
    if let Ok(guard) = atlas_route_cell().read() {
        let limit = limit.unwrap_or(usize::MAX);
        if limit == 0 {
            return route;
        }
        let mut frames: Vec<AtlasFrame> = guard.iter().cloned().collect();
        if frames.len() > limit {
            let start = frames.len() - limit;
            frames.drain(0..start);
        }
        route.frames = frames;
    }
    route
}

/// Summarises the chronological atlas route up to the requested limit.
pub fn get_atlas_route_summary(limit: Option<usize>) -> AtlasRouteSummary {
    get_atlas_route(limit).summary()
}

/// Merges an atlas fragment into the stored frame, creating it if absent.
pub fn merge_atlas_fragment(fragment: AtlasFragment) {
    if fragment.is_empty() {
        return;
    }
    let mut updated: Option<AtlasFrame> = None;
    if let Ok(mut guard) = atlas_cell().write() {
        if let Some(frame) = guard.as_mut() {
            frame.merge_fragment(fragment);
            push_atlas_route(frame);
            updated = Some(frame.clone());
        } else if let Some(frame) = AtlasFrame::from_fragment(fragment) {
            push_atlas_route(&frame);
            updated = Some(frame.clone());
            *guard = Some(frame);
        }
    }
    if let Some(frame) = updated {
        notify_atlas_frame_observers(&frame);
    }
}

/// Stores the latest maintainer report emitted by the temporal maintainer heuristics.
pub fn set_maintainer_report(report: MaintainerReport) {
    if let Ok(mut guard) = maintainer_cell().write() {
        *guard = Some(report.clone());
    }
    let fragment = fragment_from_maintainer_report(&report);
    merge_atlas_fragment(fragment);
}

/// Returns the latest maintainer report, if one has been recorded.
pub fn get_maintainer_report() -> Option<MaintainerReport> {
    maintainer_cell()
        .read()
        .ok()
        .and_then(|guard| guard.as_ref().cloned())
}

#[cfg(test)]
pub(crate) fn clear_maintainer_report_for_test() {
    if let Ok(mut guard) = maintainer_cell().write() {
        *guard = None;
    }
}

/// Stores the most recent SoftLogic Z feedback sample.
pub fn set_softlogic_z(feedback: SoftlogicZFeedback) {
    #[cfg(feature = "psi")]
    let _psi_guard = psi_lock().lock();

    match softlogic_z_cell().write() {
        Ok(mut guard) => {
            *guard = Some(feedback.clone());
        }
        Err(poisoned) => {
            let mut guard = poisoned.into_inner();
            *guard = Some(feedback.clone());
        }
    }

    #[cfg(feature = "psi")]
    drop(_psi_guard);

    let fragment = fragment_from_softlogic(&feedback);
    merge_atlas_fragment(fragment);
}

/// Stores the most recent region-weighted loss heatmap for explainability.
pub fn set_region_loss_report(report: AttributionReport) {
    match region_report_cell().write() {
        Ok(mut guard) => {
            *guard = Some(report);
        }
        Err(poisoned) => {
            let mut guard = poisoned.into_inner();
            *guard = Some(report);
        }
    }
}

/// Clears the stored region-weighted loss report.
pub fn clear_region_loss_report() {
    if let Ok(mut guard) = region_report_cell().write() {
        guard.take();
    }
}

/// Returns the last stored region heatmap report, if any.
pub fn get_region_loss_report() -> Option<AttributionReport> {
    region_report_cell()
        .read()
        .ok()
        .and_then(|guard| guard.as_ref().cloned())
}

/// Returns the latest SoftLogic Z feedback sample if one has been recorded.
pub fn get_softlogic_z() -> Option<SoftlogicZFeedback> {
    #[cfg(feature = "psi")]
    let _psi_guard = psi_lock().lock();

    let result = match softlogic_z_cell().read() {
        Ok(guard) => guard.as_ref().cloned(),
        Err(poisoned) => poisoned.into_inner().as_ref().cloned(),
    };

    #[cfg(feature = "psi")]
    drop(_psi_guard);

    result
}

#[cfg(feature = "psi")]
pub fn clear_softlogic_z() {
    let _guard = psi_lock().lock();
    match softlogic_z_cell().write() {
        Ok(mut guard) => {
            *guard = None;
        }
        Err(poisoned) => {
            let mut guard = poisoned.into_inner();
            *guard = None;
        }
    }
}

/// Snapshot summarising the latest RealGrad projection applied by the system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RealGradPulse {
    /// L¹ magnitude of the input signal.
    pub lebesgue_measure: f32,
    /// Total magnitude routed into the monad biome.
    pub monad_energy: f32,
    /// Total magnitude retained in the Z-space field.
    pub z_energy: f32,
    /// Share of the total energy routed to the monad biome.
    pub residual_ratio: f32,
    /// Ratio between the monad biome and the Lebesgue measure.
    pub lebesgue_ratio: f32,
    /// Ramanujan π estimate used for the projection.
    pub ramanujan_pi: f32,
    /// Tolerance that bounded the tempered sequence convergence.
    pub tolerance: f32,
    /// Final convergence error produced by the tempered sequence (if any).
    pub convergence_error: f32,
    /// Number of sequence members processed.
    pub iterations: u32,
    /// Whether the sequence remained dominated by its bounding function.
    pub dominated: bool,
    /// Whether the sequence satisfied the configured tolerance.
    pub converged: bool,
    /// Cached gradient norm reported by the RealGrad projection.
    pub gradient_norm: f32,
    /// Ratio of near-zero gradient entries observed by the projection.
    pub gradient_sparsity: f32,
    /// Exponential moving average of the gradient norm reported by the engine.
    pub rolling_gradient_norm: f32,
    /// Exponential moving average of the residual ratio reported by the engine.
    pub rolling_residual_ratio: f32,
}

impl Default for RealGradPulse {
    fn default() -> Self {
        Self {
            lebesgue_measure: 0.0,
            monad_energy: 0.0,
            z_energy: 0.0,
            residual_ratio: 0.0,
            lebesgue_ratio: 0.0,
            ramanujan_pi: 0.0,
            tolerance: 0.0,
            convergence_error: 0.0,
            iterations: 0,
            dominated: false,
            converged: false,
            gradient_norm: 0.0,
            gradient_sparsity: 1.0,
            rolling_gradient_norm: 0.0,
            rolling_residual_ratio: 0.0,
        }
    }
}

impl RealGradPulse {
    /// Returns the gradient summary captured by the pulse.
    pub fn gradient_summary(&self) -> GradientSummary {
        GradientSummary {
            norm: self.gradient_norm.max(0.0),
            sparsity: self.gradient_sparsity.clamp(0.0, 1.0),
        }
    }
}

static LAST_REALGRAD: OnceLock<RwLock<Option<RealGradPulse>>> = OnceLock::new();

fn realgrad_cell() -> &'static RwLock<Option<RealGradPulse>> {
    LAST_REALGRAD.get_or_init(|| RwLock::new(None))
}

/// Stores the latest RealGrad pulse emitted by the engine.
pub fn set_last_realgrad(pulse: &RealGradPulse) {
    if let Ok(mut guard) = realgrad_cell().write() {
        *guard = Some(*pulse);
    }
    let fragment = fragment_from_realgrad(pulse);
    merge_atlas_fragment(fragment);
}

/// Returns the most recent RealGrad pulse, if one has been recorded.
pub fn get_last_realgrad() -> Option<RealGradPulse> {
    realgrad_cell()
        .read()
        .ok()
        .and_then(|guard| guard.as_ref().copied())
}

#[cfg(test)]
pub(crate) fn clear_last_realgrad_for_test() {
    if let Ok(mut guard) = realgrad_cell().write() {
        *guard = None;
    }
}

#[cfg_attr(
    feature = "psi",
    doc = "Stores the latest desire step telemetry snapshot for downstream consumers."
)]
#[cfg(feature = "psi")]
pub fn set_last_desire_step(step: DesireStepTelemetry) {
    if let Ok(mut guard) = desire_step_cell().write() {
        *guard = Some(step.clone());
    }
    let mut fragment = fragment_from_desire_step(&step);
    if let Some(feedback) = &step.z_feedback {
        populate_softlogic_metrics(&mut fragment, "desire.z_feedback", feedback);
        fragment.z_signal = Some(feedback.z_signal);
    }
    populate_psi_breakdown(&mut fragment, "desire.psi", &step.psi_breakdown);
    if let Some(total) = step.psi_total {
        fragment.push_metric("desire.psi.total", total);
    }
    if let Some(step_time) = system_time_seconds(step.timestamp) {
        fragment.timestamp = Some(step_time);
    }
    annotate_psi_events(&mut fragment, "desire.psi", &step.psi_events);
    merge_atlas_fragment(fragment);
}

#[cfg_attr(
    feature = "psi",
    doc = "Clears the cached desire step telemetry snapshot."
)]
#[cfg(feature = "psi")]
pub fn clear_last_desire_step() {
    if let Ok(mut guard) = desire_step_cell().write() {
        *guard = None;
    }
}

#[cfg_attr(
    feature = "psi",
    doc = "Returns the latest desire step telemetry snapshot, if one has been recorded."
)]
#[cfg(feature = "psi")]
pub fn get_last_desire_step() -> Option<DesireStepTelemetry> {
    desire_step_cell()
        .read()
        .ok()
        .and_then(|guard| guard.as_ref().cloned())
}

static LAST_CHRONO_LOOP: OnceLock<RwLock<Option<ChronoLoopSignal>>> = OnceLock::new();

fn chrono_loop_cell() -> &'static RwLock<Option<ChronoLoopSignal>> {
    LAST_CHRONO_LOOP.get_or_init(|| RwLock::new(None))
}

/// Stores the most recent chrono loop signal so other nodes can consume it.
pub fn set_chrono_loop(signal: ChronoLoopSignal) {
    if let Ok(mut guard) = chrono_loop_cell().write() {
        *guard = Some(signal);
    }
}

/// Returns the latest chrono loop signal, if any has been recorded.
pub fn get_chrono_loop() -> Option<ChronoLoopSignal> {
    chrono_loop_cell()
        .read()
        .ok()
        .and_then(|guard| guard.as_ref().cloned())
}

/// Envelope combining chrono loop telemetry with collapse/Z feedback so other nodes can replay it.
#[derive(Clone, Debug)]
pub struct LoopbackEnvelope {
    /// Timestamp associated with the captured loop signal.
    pub timestamp: f32,
    /// Optional identifier describing which node produced the envelope.
    pub source: Option<String>,
    /// Participation/support weight contributed by the source node.
    pub support: f32,
    /// Optional aggregate collapse total associated with the envelope.
    pub collapse_total: Option<f32>,
    /// Optional Z-space control bias produced by the softlogic observer.
    pub z_signal: Option<f32>,
    /// Optional SpiralK script hint that accompanied the telemetry.
    pub script_hint: Option<String>,
    /// Chrono loop signal captured at the timestamp.
    pub loop_signal: ChronoLoopSignal,
}

impl LoopbackEnvelope {
    /// Creates a new envelope from the supplied chrono loop signal.
    pub fn new(loop_signal: ChronoLoopSignal) -> Self {
        let timestamp = loop_signal.summary.latest_timestamp;
        Self {
            timestamp,
            source: None,
            support: 1.0,
            collapse_total: None,
            z_signal: None,
            script_hint: None,
            loop_signal,
        }
    }

    /// Annotates the envelope with a source identifier.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Updates the support weight carried by the envelope.
    pub fn with_support(mut self, support: f32) -> Self {
        self.support = if support.is_finite() { support } else { 1.0 };
        self
    }

    /// Records an optional collapse total associated with the envelope.
    pub fn with_collapse_total(mut self, total: Option<f32>) -> Self {
        self.collapse_total = total.filter(|value| value.is_finite());
        self
    }

    /// Records an optional Z-space control signal.
    pub fn with_z_signal(mut self, z: Option<f32>) -> Self {
        self.z_signal = z.filter(|value| value.is_finite());
        self
    }

    /// Annotates the envelope with a SpiralK script hint.
    pub fn with_script_hint(mut self, script: Option<String>) -> Self {
        self.script_hint = script;
        self
    }
}

static LOOPBACK_BUFFER: OnceLock<RwLock<VecDeque<LoopbackEnvelope>>> = OnceLock::new();

fn loopback_cell() -> &'static RwLock<VecDeque<LoopbackEnvelope>> {
    LOOPBACK_BUFFER.get_or_init(|| RwLock::new(VecDeque::with_capacity(32)))
}

/// Pushes a new loopback envelope into the global queue, keeping the buffer bounded.
pub fn push_loopback_envelope(envelope: LoopbackEnvelope) {
    if let Ok(mut guard) = loopback_cell().write() {
        guard.push_back(envelope.clone());
        while guard.len() > 64 {
            guard.pop_front();
        }
    }
    merge_atlas_fragment(fragment_from_loopback(&envelope));
}

fn fragment_from_loopback(envelope: &LoopbackEnvelope) -> AtlasFragment {
    let mut fragment = AtlasFragment::new();
    fragment.timestamp = Some(envelope.timestamp);
    fragment.summary = Some(envelope.loop_signal.summary.clone());
    fragment.harmonics = envelope.loop_signal.harmonics.clone();
    fragment.loop_support = Some(envelope.support);
    fragment.collapse_total = envelope.collapse_total;
    fragment.z_signal = envelope.z_signal;
    fragment.script_hint = envelope.script_hint.clone();
    if let Some(source) = &envelope.source {
        fragment.push_note(format!("source:{source}"));
    }
    fragment.push_metric("loop.frames", envelope.loop_signal.summary.frames as f32);
    fragment.push_metric("loop.energy", envelope.loop_signal.summary.mean_energy);
    fragment.push_metric("loop.drift", envelope.loop_signal.summary.mean_drift);
    if let Some(total) = envelope.collapse_total {
        fragment.push_metric("collapse.total", total);
    }
    if let Some(z) = envelope.z_signal {
        fragment.push_metric("z.bias", z);
    }
    fragment
}

fn fragment_from_maintainer_report(report: &MaintainerReport) -> AtlasFragment {
    let mut fragment = AtlasFragment::new();
    fragment.maintainer_status = Some(report.status);
    if !report.diagnostic.is_empty() {
        fragment.maintainer_diagnostic = Some(report.diagnostic.clone());
        fragment.push_note(format!("maintainer.diagnostic:{}", report.diagnostic));
    }
    fragment.suggested_max_scale = report.suggested_max_scale.filter(|value| value.is_finite());
    fragment.suggested_pressure = report.suggested_pressure.filter(|value| value.is_finite());
    fragment.push_note(format!("maintainer.status:{}", report.status.as_str()));
    fragment.push_metric("maintainer.average_drift", report.average_drift);
    fragment.push_metric("maintainer.mean_energy", report.mean_energy);
    fragment.push_metric("maintainer.mean_decay", report.mean_decay);
    fragment.push_metric(
        "maintainer.should_rewrite",
        bool_as_metric(report.should_rewrite()),
    );
    if let Some(peak) = &report.drift_peak {
        fragment.push_metric("maintainer.drift_peak.frequency", peak.frequency);
        fragment.push_metric("maintainer.drift_peak.magnitude", peak.magnitude);
        fragment.push_metric("maintainer.drift_peak.phase", peak.phase);
    }
    if let Some(peak) = &report.energy_peak {
        fragment.push_metric("maintainer.energy_peak.frequency", peak.frequency);
        fragment.push_metric("maintainer.energy_peak.magnitude", peak.magnitude);
        fragment.push_metric("maintainer.energy_peak.phase", peak.phase);
    }
    #[cfg(feature = "kdsl")]
    if let Some(script) = report.spiralk_script.as_ref() {
        if !script.is_empty() {
            fragment.script_hint = Some(script.clone());
        }
    }
    fragment
}

fn fragment_from_softlogic(feedback: &SoftlogicZFeedback) -> AtlasFragment {
    let mut fragment = AtlasFragment::new();
    fragment.z_signal = Some(feedback.z_signal);
    populate_softlogic_metrics(&mut fragment, "softlogic", feedback);
    if let Some(scale) = feedback.scale {
        fragment.push_note(format!(
            "softlogic.scale.radius:{:.3}@{:.3}",
            scale.physical_radius, scale.log_radius
        ));
    }
    fragment
}

fn populate_softlogic_metrics(
    fragment: &mut AtlasFragment,
    prefix: &str,
    feedback: &SoftlogicZFeedback,
) {
    let (above, here, beneath) = feedback.band_energy;
    fragment.push_metric(format!("{prefix}.psi_total"), feedback.psi_total);
    fragment.push_metric(format!("{prefix}.weighted_loss"), feedback.weighted_loss);
    fragment.push_metric(format!("{prefix}.band.above"), above);
    fragment.push_metric(format!("{prefix}.band.here"), here);
    fragment.push_metric(format!("{prefix}.band.beneath"), beneath);
    fragment.push_metric(format!("{prefix}.band.total"), above + here + beneath);
    fragment.push_metric(format!("{prefix}.drift"), feedback.drift);
    fragment.push_metric(format!("{prefix}.z_signal"), feedback.z_signal);
    if let Some(scale) = feedback.scale {
        fragment.push_metric(format!("{prefix}.scale.physical"), scale.physical_radius);
        fragment.push_metric(format!("{prefix}.scale.log"), scale.log_radius);
    }
    fragment.push_metric(format!("{prefix}.events"), feedback.events.len() as f32);
    for event in &feedback.events {
        if !event.is_empty() {
            fragment.push_note(format!("{prefix}.event:{}", event));
        }
    }
    if !feedback.attributions.is_empty() {
        let mut total = 0.0;
        for (source, weight) in &feedback.attributions {
            let label = z_source_label(source);
            fragment.push_metric(format!("{prefix}.attribution.{label}"), *weight);
            total += weight.max(0.0);
        }
        fragment.push_metric(format!("{prefix}.attribution.total"), total);
    }
    if let Some(elliptic) = &feedback.elliptic {
        let elliptic_prefix = format!("{prefix}.elliptic");
        fragment.push_metric(
            format!("{elliptic_prefix}.curvature_radius"),
            elliptic.curvature_radius,
        );
        fragment.push_metric(
            format!("{elliptic_prefix}.geodesic_radius"),
            elliptic.geodesic_radius,
        );
        fragment.push_metric(
            format!("{elliptic_prefix}.normalized_radius"),
            elliptic.normalized_radius,
        );
        fragment.push_metric(
            format!("{elliptic_prefix}.spin_alignment"),
            elliptic.spin_alignment,
        );
        fragment.push_metric(
            format!("{elliptic_prefix}.sheet_index"),
            elliptic.sheet_index as f32,
        );
        fragment.push_metric(
            format!("{elliptic_prefix}.sheet_position"),
            elliptic.sheet_position,
        );
        fragment.push_metric(
            format!("{elliptic_prefix}.sheet_count"),
            elliptic.sheet_count as f32,
        );
        fragment.push_metric(
            format!("{elliptic_prefix}.normal_bias"),
            elliptic.normal_bias,
        );
    }
}

fn z_source_label(source: &ZSource) -> String {
    match source {
        ZSource::Microlocal => "microlocal".to_string(),
        ZSource::Maxwell => "maxwell".to_string(),
        ZSource::Graph => "graph".to_string(),
        ZSource::Desire => "desire".to_string(),
        ZSource::GW => "gw".to_string(),
        ZSource::RealGrad => "realgrad".to_string(),
        ZSource::Other(tag) => format!("other.{}", tag.to_ascii_lowercase()),
    }
}

fn fragment_from_realgrad(pulse: &RealGradPulse) -> AtlasFragment {
    let mut fragment = AtlasFragment::new();
    if pulse.iterations > 0 {
        fragment.timestamp = Some(pulse.iterations as f32);
    }
    fragment.push_metric("realgrad.lebesgue_measure", pulse.lebesgue_measure);
    fragment.push_metric("realgrad.monad_energy", pulse.monad_energy);
    fragment.push_metric("realgrad.z_energy", pulse.z_energy);
    fragment.push_metric("realgrad.residual_ratio", pulse.residual_ratio);
    fragment.push_metric("realgrad.lebesgue_ratio", pulse.lebesgue_ratio);
    fragment.push_metric("realgrad.ramanujan_pi", pulse.ramanujan_pi);
    fragment.push_metric("realgrad.tolerance", pulse.tolerance);
    fragment.push_metric("realgrad.convergence_error", pulse.convergence_error);
    fragment.push_metric("realgrad.iterations", pulse.iterations as f32);
    fragment.push_metric("realgrad.dominated", bool_as_metric(pulse.dominated));
    fragment.push_metric("realgrad.converged", bool_as_metric(pulse.converged));
    fragment.push_metric("realgrad.gradient_norm", pulse.gradient_norm);
    fragment.push_metric("realgrad.gradient_sparsity", pulse.gradient_sparsity);
    fragment.push_metric(
        "realgrad.rolling_gradient_norm",
        pulse.rolling_gradient_norm,
    );
    fragment.push_metric(
        "realgrad.rolling_residual_ratio",
        pulse.rolling_residual_ratio,
    );
    fragment
}

fn fragment_from_config_events(events: &[ConfigDiffEvent]) -> AtlasFragment {
    let mut fragment = AtlasFragment::new();
    fragment.push_metric("config.diff.total", events.len() as f32);
    let mut base = 0u32;
    let mut site = 0u32;
    let mut run = 0u32;
    for event in events {
        match event.layer {
            ConfigLayer::Base => base += 1,
            ConfigLayer::Site => site += 1,
            ConfigLayer::Run => run += 1,
        }
        fragment.push_note(format!("config.diff.{}:{}", event.layer, event.path));
    }
    if base > 0 {
        fragment.push_metric("config.diff.base", base as f32);
    }
    if site > 0 {
        fragment.push_metric("config.diff.site", site as f32);
    }
    if run > 0 {
        fragment.push_metric("config.diff.run", run as f32);
    }
    fragment
}

fn bool_as_metric(value: bool) -> f32 {
    if value {
        1.0
    } else {
        0.0
    }
}

#[cfg(feature = "psi")]
fn psi_step_timestamp(step: u64) -> Option<f32> {
    if step == 0 {
        None
    } else {
        Some(step as f32)
    }
}

#[cfg(feature = "psi")]
fn populate_psi_breakdown(
    fragment: &mut AtlasFragment,
    prefix: &str,
    breakdown: &HashMap<PsiComponent, f32>,
) {
    for &component in PSI_COMPONENTS.iter() {
        if let Some(value) = breakdown.get(&component) {
            fragment.push_metric(format!("{prefix}.{}", component), *value);
        }
    }
}

#[cfg(feature = "psi")]
fn annotate_psi_events(
    fragment: &mut AtlasFragment,
    prefix: &str,
    events: &[PsiEvent],
) -> Option<u64> {
    let mut last_step = None;
    let mut rising = 0u32;
    let mut falling = 0u32;
    for event in events {
        match event {
            PsiEvent::ThresholdCross {
                component,
                value,
                threshold,
                up,
                step,
            } => {
                last_step = Some(last_step.map_or(*step, |prev: u64| prev.max(*step)));
                if *up {
                    rising += 1;
                } else {
                    falling += 1;
                }
                let label = component.to_string();
                fragment.push_note(format!("{prefix}.event:{}", event));
                fragment.push_metric(format!("{prefix}.event.value.{label}"), *value);
                fragment.push_metric(format!("{prefix}.event.threshold.{label}"), *threshold);
                let delta = if *up {
                    value - threshold
                } else {
                    threshold - value
                };
                fragment.push_metric(format!("{prefix}.event.delta.{label}"), delta);
            }
        }
    }
    fragment.push_metric(format!("{prefix}.events.total"), events.len() as f32);
    if rising > 0 {
        fragment.push_metric(format!("{prefix}.events.up"), rising as f32);
    }
    if falling > 0 {
        fragment.push_metric(format!("{prefix}.events.down"), falling as f32);
    }
    last_step
}

#[cfg(feature = "psi")]
fn fragment_from_desire_step(step: &DesireStepTelemetry) -> AtlasFragment {
    let mut fragment = AtlasFragment::new();
    fragment.push_metric("desire.entropy", step.entropy);
    fragment.push_metric("desire.temperature", step.temperature);
    fragment.push_metric("desire.hypergrad_penalty", step.hypergrad_penalty);
    fragment.push_metric("desire.avoidance_energy", step.avoidance_energy);
    fragment.push_metric("desire.logit_energy", step.logit_energy);
    fragment.push_metric("desire.weights.alpha", step.weights.alpha);
    fragment.push_metric("desire.weights.beta", step.weights.beta);
    fragment.push_metric("desire.weights.gamma", step.weights.gamma);
    fragment.push_metric("desire.weights.lambda", step.weights.lambda);
    fragment.push_metric("desire.alpha", step.alpha);
    fragment.push_metric("desire.beta", step.beta);
    fragment.push_metric("desire.gamma", step.gamma);
    fragment.push_metric("desire.lambda", step.lambda);
    fragment.push_metric(
        "desire.trigger_emitted",
        bool_as_metric(step.trigger_emitted),
    );
    let phase = match step.phase {
        DesirePhaseTelemetry::Observation => "observation",
        DesirePhaseTelemetry::Injection => "injection",
        DesirePhaseTelemetry::Integration => "integration",
    };
    fragment.push_note(format!("desire.phase:{phase}"));
    if let Some(trigger) = &step.trigger {
        fragment.push_metric("desire.trigger.mean_penalty", trigger.mean_penalty);
        fragment.push_metric("desire.trigger.mean_entropy", trigger.mean_entropy);
        fragment.push_metric("desire.trigger.temperature", trigger.temperature);
        fragment.push_metric("desire.trigger.samples", trigger.samples as f32);
    }
    if let Some(avoidance) = &step.avoidance {
        fragment.push_metric("desire.avoidance.tokens", avoidance.tokens.len() as f32);
        if !avoidance.scores.is_empty() {
            let sum: f32 = avoidance.scores.iter().copied().sum();
            let max = avoidance
                .scores
                .iter()
                .fold(f32::NEG_INFINITY, |acc, value| acc.max(*value));
            let min = avoidance
                .scores
                .iter()
                .fold(f32::INFINITY, |acc, value| acc.min(*value));
            let count = avoidance.scores.len() as f32;
            fragment.push_metric("desire.avoidance.score_mean", sum / count);
            fragment.push_metric("desire.avoidance.score_max", max);
            fragment.push_metric("desire.avoidance.score_min", min);
        }
    }
    fragment
}

#[cfg(feature = "psi")]
fn system_time_seconds(time: SystemTime) -> Option<f32> {
    time.duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_secs_f64())
        .and_then(|secs| {
            if secs.is_finite() {
                Some(secs as f32)
            } else {
                None
            }
        })
}

#[cfg(feature = "psychoid")]
fn fragment_from_psychoid_reading(reading: &PsychoidReading) -> AtlasFragment {
    let mut fragment = AtlasFragment::new();
    if reading.step > 0 {
        fragment.timestamp = Some(reading.step as f32);
    }
    fragment.push_metric("psychoid.cti", reading.cti);
    fragment.push_metric("psychoid.raw.metrics", reading.raw.len() as f32);
    fragment.push_metric("psychoid.z.metrics", reading.z_scores.len() as f32);
    let raw_total: f32 = reading.raw.values().copied().sum();
    fragment.push_metric("psychoid.raw.total", raw_total);
    let mut abs_sum = 0.0;
    let mut max_abs = 0.0;
    let mut top_key = None;
    for (key, value) in &reading.z_scores {
        let magnitude = value.abs();
        abs_sum += magnitude;
        if magnitude > max_abs {
            max_abs = magnitude;
            top_key = Some(*key);
        }
    }
    if !reading.z_scores.is_empty() {
        fragment.push_metric(
            "psychoid.z.abs_mean",
            abs_sum / reading.z_scores.len() as f32,
        );
        fragment.push_metric("psychoid.z.abs_max", max_abs);
    }
    if let Some(key) = top_key {
        fragment.push_note(format!("psychoid.leading_z:{key}"));
    }
    fragment
}

#[cfg(feature = "collapse")]
fn fragment_from_collapse_pulse(pulse: &CollapsePulse) -> AtlasFragment {
    let mut fragment = AtlasFragment::new();
    if pulse.step > 0 {
        fragment.timestamp = Some(pulse.step as f32);
    }
    fragment.collapse_total = Some(pulse.total);
    fragment.push_metric("collapse.total", pulse.total);
    fragment.push_metric("collapse.step", pulse.step as f32);
    fragment.push_note(format!(
        "collapse.command:{}",
        collapse_command_label(&pulse.command)
    ));
    match &pulse.command {
        DriveCmd::Collapse {
            grad_scale,
            max_norm,
            lr_decay,
            due_to_trend,
            due_to_deviation,
        } => {
            fragment.push_metric("collapse.command.grad_scale", *grad_scale);
            if let Some(norm) = max_norm {
                fragment.push_metric("collapse.command.max_norm", *norm);
            }
            if let Some(decay) = lr_decay {
                fragment.push_metric("collapse.command.lr_decay", *decay);
            }
            fragment.push_metric(
                "collapse.command.due_to_trend",
                bool_as_metric(*due_to_trend),
            );
            fragment.push_metric(
                "collapse.command.due_to_deviation",
                bool_as_metric(*due_to_deviation),
            );
        }
        DriveCmd::Bloom {
            lr_mul,
            due_to_trend,
            due_to_deviation,
        } => {
            fragment.push_metric("collapse.command.lr_mul", *lr_mul);
            fragment.push_metric(
                "collapse.command.due_to_trend",
                bool_as_metric(*due_to_trend),
            );
            fragment.push_metric(
                "collapse.command.due_to_deviation",
                bool_as_metric(*due_to_deviation),
            );
        }
        DriveCmd::None => {}
    }
    if let Some(signal) = &pulse.loop_signal {
        fragment.summary = Some(signal.summary.clone());
        fragment.harmonics = signal.harmonics.clone();
        fragment.push_metric("collapse.loop.frames", signal.summary.frames as f32);
        fragment.push_metric("collapse.loop.energy", signal.summary.mean_energy);
        fragment.push_metric("collapse.loop.drift", signal.summary.mean_drift);
    }
    fragment
}

#[cfg(feature = "collapse")]
fn collapse_command_label(cmd: &DriveCmd) -> String {
    match cmd {
        DriveCmd::None => "none".to_string(),
        DriveCmd::Collapse {
            grad_scale,
            max_norm,
            lr_decay,
            due_to_trend,
            due_to_deviation,
        } => {
            let mut parts = vec![format!("scale={grad_scale:.3}")];
            if let Some(norm) = max_norm {
                parts.push(format!("max_norm={norm:.3}"));
            }
            if let Some(decay) = lr_decay {
                parts.push(format!("lr_decay={decay:.3}"));
            }
            if *due_to_trend {
                parts.push("trend".to_string());
            }
            if *due_to_deviation {
                parts.push("deviation".to_string());
            }
            format!("collapse({})", parts.join(","))
        }
        DriveCmd::Bloom {
            lr_mul,
            due_to_trend,
            due_to_deviation,
        } => {
            let mut parts = vec![format!("lr_mul={lr_mul:.3}")];
            if *due_to_trend {
                parts.push("trend".to_string());
            }
            if *due_to_deviation {
                parts.push("deviation".to_string());
            }
            format!("bloom({})", parts.join(","))
        }
    }
}

/// Drains up to `limit` loopback envelopes from the queue in FIFO order.
pub fn drain_loopback_envelopes(limit: usize) -> Vec<LoopbackEnvelope> {
    if limit == 0 {
        return Vec::new();
    }
    if let Ok(mut guard) = loopback_cell().write() {
        let mut drained = Vec::new();
        for _ in 0..limit {
            if let Some(envelope) = guard.pop_front() {
                drained.push(envelope);
            } else {
                break;
            }
        }
        drained
    } else {
        Vec::new()
    }
}

/// Returns up to `limit` envelopes without mutating the queue.
pub fn peek_loopback_envelopes(limit: usize) -> Vec<LoopbackEnvelope> {
    if limit == 0 {
        return Vec::new();
    }
    if let Ok(guard) = loopback_cell().read() {
        guard.iter().take(limit).cloned().collect()
    } else {
        Vec::new()
    }
}

#[cfg(feature = "collapse")]
#[derive(Clone, Debug)]
pub struct CollapsePulse {
    /// Step of the PSI reading that triggered the command.
    pub step: u64,
    /// Aggregate PSI total associated with the command.
    pub total: f32,
    /// Command emitted by the collapse drive.
    pub command: DriveCmd,
    /// Latest chrono loop signal observed when the command was issued.
    pub loop_signal: Option<ChronoLoopSignal>,
}

#[cfg(feature = "collapse")]
static LAST_COLLAPSE: OnceLock<RwLock<Option<CollapsePulse>>> = OnceLock::new();

#[cfg(feature = "collapse")]
fn collapse_cell() -> &'static RwLock<Option<CollapsePulse>> {
    LAST_COLLAPSE.get_or_init(|| RwLock::new(None))
}

#[cfg(feature = "collapse")]
/// Stores the most recent collapse command pulse.
pub fn set_collapse_pulse(pulse: CollapsePulse) {
    if let Ok(mut guard) = collapse_cell().write() {
        *guard = Some(pulse.clone());
    }
    let fragment = fragment_from_collapse_pulse(&pulse);
    merge_atlas_fragment(fragment);
}

#[cfg(feature = "collapse")]
/// Returns the most recent collapse pulse, if any.
pub fn get_collapse_pulse() -> Option<CollapsePulse> {
    collapse_cell()
        .read()
        .ok()
        .and_then(|guard| guard.as_ref().cloned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::chrono::{ChronoHarmonics, ChronoPeak, ChronoSummary};
    use crate::telemetry::dashboard::DashboardMetric;
    use crate::telemetry::maintainer::MaintainerStatus;
    use std::sync::{Mutex, OnceLock};
    use std::time::SystemTime;

    fn atlas_test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn sample_summary(timestamp: f32) -> ChronoSummary {
        ChronoSummary {
            frames: 4,
            duration: 1.0,
            latest_timestamp: timestamp,
            mean_drift: 0.2,
            mean_abs_drift: 0.4,
            drift_std: 0.1,
            mean_energy: 2.0,
            energy_std: 0.3,
            mean_decay: -0.1,
            min_energy: 1.5,
            max_energy: 2.5,
        }
    }

    #[test]
    fn loopback_queue_drains_in_order() {
        let _guard = atlas_test_lock().lock().unwrap();
        // Ensure the buffer starts empty for the test.
        let _ = drain_loopback_envelopes(usize::MAX);
        clear_atlas();
        let signal_a = ChronoLoopSignal::new(sample_summary(1.0), None);
        let mut harmonics = ChronoHarmonics {
            frames: 4,
            duration: 1.0,
            sample_rate: 4.0,
            nyquist: 2.0,
            drift_power: vec![0.1; 4],
            energy_power: vec![0.2; 4],
            dominant_drift: Some(ChronoPeak {
                frequency: 0.5,
                magnitude: 0.3,
                phase: 0.0,
            }),
            dominant_energy: None,
        };
        let signal_b = ChronoLoopSignal::new(sample_summary(2.0), Some(harmonics.clone()));
        harmonics.dominant_energy = Some(ChronoPeak {
            frequency: 0.8,
            magnitude: 0.6,
            phase: 0.1,
        });
        let signal_c = ChronoLoopSignal::new(sample_summary(3.0), Some(harmonics));

        push_loopback_envelope(LoopbackEnvelope::new(signal_a).with_support(1.0));
        push_loopback_envelope(LoopbackEnvelope::new(signal_b).with_support(2.0));
        push_loopback_envelope(
            LoopbackEnvelope::new(signal_c)
                .with_support(3.0)
                .with_collapse_total(Some(1.2))
                .with_z_signal(Some(0.4)),
        );

        let drained = drain_loopback_envelopes(2);
        assert_eq!(drained.len(), 2);
        assert!(drained[0].timestamp <= drained[1].timestamp);
        let remaining = drain_loopback_envelopes(2);
        assert_eq!(remaining.len(), 1);
        assert!(drain_loopback_envelopes(1).is_empty());
    }

    #[test]
    fn loopback_updates_atlas_snapshot() {
        let _guard = atlas_test_lock().lock().unwrap();
        clear_atlas();
        let _ = drain_loopback_envelopes(usize::MAX);
        let signal = ChronoLoopSignal::new(sample_summary(2.5), None);
        let envelope = LoopbackEnvelope::new(signal)
            .with_support(2.5)
            .with_collapse_total(Some(1.1))
            .with_z_signal(Some(0.4));
        push_loopback_envelope(envelope);
        let atlas = get_atlas_frame().expect("atlas frame");
        assert!(atlas.timestamp > 0.0);
        assert!(atlas.loop_support >= 2.5 - f32::EPSILON);
        assert_eq!(atlas.collapse_total, Some(1.1));
        assert_eq!(atlas.z_signal, Some(0.4));
        assert!(atlas
            .metrics
            .iter()
            .any(|metric| metric.name == "loop.energy"));
        let route = get_atlas_route(Some(8));
        assert!(!route.is_empty());
        assert_eq!(route.latest().unwrap().timestamp, atlas.timestamp);
    }

    #[test]
    fn atlas_route_retains_recent_frames() {
        let _guard = atlas_test_lock().lock().unwrap();
        clear_atlas();
        clear_atlas_route();
        for idx in 0..6 {
            let mut frame = AtlasFrame::new((idx + 1) as f32);
            frame.loop_support = idx as f32;
            set_atlas_frame(frame);
        }
        let route = get_atlas_route(Some(4));
        assert_eq!(route.len(), 4);
        assert_eq!(route.frames.first().unwrap().timestamp, 3.0);
        assert_eq!(route.latest().unwrap().loop_support, 5.0);
    }

    #[test]
    fn atlas_route_summary_exposes_recent_activity() {
        let _guard = atlas_test_lock().lock().unwrap();
        clear_atlas();
        clear_atlas_route();
        for idx in 0..3 {
            let mut fragment = AtlasFragment::new();
            fragment.timestamp = Some((idx + 1) as f32);
            fragment.push_metric_with_district(
                format!("session.surface.latency.{}", idx),
                idx as f32,
                "Surface",
            );
            fragment.push_metric_with_district(
                format!("trainer.loop.energy.{}", idx),
                1.0 + idx as f32,
                "Concourse",
            );
            merge_atlas_fragment(fragment);
        }
        let summary = get_atlas_route_summary(Some(3));
        assert_eq!(summary.frames, 3);
        assert!(summary.latest_timestamp >= 3.0 - f32::EPSILON);
        assert!(!summary.districts.is_empty());
        let concourse = summary
            .districts
            .iter()
            .find(|district| district.name == "Concourse")
            .expect("concourse summary");
        assert_eq!(concourse.coverage, 3);
        assert!(concourse.delta > 0.0);
    }

    #[test]
    fn softlogic_feedback_populates_atlas() {
        let _guard = atlas_test_lock().lock().unwrap();
        clear_atlas();
        clear_atlas_route();
        let feedback = SoftlogicZFeedback {
            psi_total: 1.2,
            weighted_loss: 0.4,
            band_energy: (0.3, 0.2, 0.1),
            drift: 0.05,
            z_signal: 0.6,
            scale: Some(ZScale::ONE),
            events: vec!["spike".into()],
            attributions: vec![(ZSource::Microlocal, 0.7), (ZSource::RealGrad, 0.3)],
            elliptic: None,
        };
        let z_signal = feedback.z_signal;
        set_softlogic_z(feedback);
        let atlas = get_atlas_frame().expect("softlogic atlas frame");
        assert_eq!(atlas.z_signal, Some(z_signal));
        assert!(atlas
            .metrics
            .iter()
            .any(|metric| metric.name == "softlogic.psi_total"));
        assert!(atlas
            .notes
            .iter()
            .any(|note| note.contains("softlogic.event")));
    }

    #[test]
    fn realgrad_pulse_enriches_atlas_metrics() {
        let _guard = atlas_test_lock().lock().unwrap();
        clear_atlas();
        clear_atlas_route();
        clear_last_realgrad_for_test();
        let mut pulse = RealGradPulse::default();
        pulse.lebesgue_measure = 5.0;
        pulse.monad_energy = 2.0;
        pulse.z_energy = 3.0;
        pulse.residual_ratio = 0.6;
        pulse.lebesgue_ratio = 0.4;
        pulse.ramanujan_pi = 3.1415;
        pulse.tolerance = 1.0e-3;
        pulse.convergence_error = 5.0e-4;
        pulse.iterations = 4;
        pulse.dominated = true;
        pulse.converged = true;
        pulse.gradient_norm = 1.5;
        pulse.gradient_sparsity = 0.2;
        pulse.rolling_gradient_norm = 1.1;
        pulse.rolling_residual_ratio = 0.3;
        set_last_realgrad(&pulse);
        let atlas = get_atlas_frame().expect("realgrad atlas frame");
        assert!(atlas
            .metrics
            .iter()
            .any(|metric| metric.name == "realgrad.gradient_norm"));
        assert!(atlas
            .metrics
            .iter()
            .any(|metric| metric.name == "realgrad.converged"));
    }

    #[test]
    fn config_diff_events_surface_in_atlas() {
        let _guard = atlas_test_lock().lock().unwrap();
        clear_atlas();
        clear_atlas_route();
        let events = vec![
            ConfigDiffEvent {
                layer: ConfigLayer::Base,
                path: "core.yml".into(),
                previous: None,
                current: Some(serde_json::json!({"learning_rate": 0.1})),
            },
            ConfigDiffEvent {
                layer: ConfigLayer::Run,
                path: "override.toml".into(),
                previous: Some(serde_json::json!({"batch_size": 32})),
                current: Some(serde_json::json!({"batch_size": 64})),
            },
        ];
        record_config_events(&events);
        let atlas = get_atlas_frame().expect("config atlas frame");
        assert!(atlas
            .metrics
            .iter()
            .any(|metric| metric.name == "config.diff.total" && metric.value == 2.0));
        assert!(atlas
            .notes
            .iter()
            .any(|note| note.starts_with("config.diff.base")));
    }

    #[test]
    fn realgrad_pulse_roundtrips_through_cache() {
        clear_last_realgrad_for_test();
        assert!(get_last_realgrad().is_none());
        let mut pulse = RealGradPulse::default();
        pulse.lebesgue_measure = 4.0;
        pulse.monad_energy = 1.0;
        pulse.z_energy = 3.0;
        pulse.residual_ratio = 0.25;
        pulse.lebesgue_ratio = 0.5;
        pulse.ramanujan_pi = 3.1415;
        pulse.tolerance = 1.0e-3;
        pulse.convergence_error = 5.0e-4;
        pulse.iterations = 3;
        pulse.dominated = true;
        pulse.converged = true;
        pulse.gradient_norm = 2.5;
        pulse.gradient_sparsity = 0.75;
        pulse.rolling_gradient_norm = 1.5;
        pulse.rolling_residual_ratio = 0.2;
        set_last_realgrad(&pulse);
        let stored = get_last_realgrad().expect("pulse stored");
        assert_eq!(stored.iterations, 3);
        assert!(stored.converged);
        assert!((stored.residual_ratio - 0.25).abs() < f32::EPSILON);
        assert!((stored.gradient_norm - 2.5).abs() < f32::EPSILON);
        assert!((stored.gradient_sparsity - 0.75).abs() < f32::EPSILON);
        assert!((stored.rolling_gradient_norm - 1.5).abs() < f32::EPSILON);
        assert!((stored.rolling_residual_ratio - 0.2).abs() < f32::EPSILON);
        let summary = stored.gradient_summary();
        assert!((summary.norm - 2.5).abs() < f32::EPSILON);
        assert!((summary.sparsity - 0.75).abs() < f32::EPSILON);
        clear_last_realgrad_for_test();
        assert!(get_last_realgrad().is_none());
    }

    #[test]
    fn dashboard_ring_records_frames() {
        let baseline = snapshot_dashboard_frames(usize::MAX).len();

        let mut frame_a = DashboardFrame::new(SystemTime::now());
        frame_a.push_metric(DashboardMetric::new("loss", 1.0));
        push_dashboard_frame(frame_a.clone());

        let mut frame_b = DashboardFrame::new(SystemTime::now());
        frame_b.push_metric(DashboardMetric::new("loss", 0.5));
        push_dashboard_frame(frame_b.clone());

        let latest = latest_dashboard_frame().expect("latest dashboard frame");
        assert_eq!(latest.metrics.len(), 1);
        assert!((latest.metrics[0].value - 0.5).abs() <= f64::EPSILON);

        let snapshot = snapshot_dashboard_frames(2);
        assert!(snapshot.len() >= 2);
        assert!((snapshot[0].metrics[0].value - 0.5).abs() <= f64::EPSILON);
        assert!((snapshot[1].metrics[0].value - 1.0).abs() <= f64::EPSILON);

        let expanded = snapshot_dashboard_frames(baseline + 2);
        assert!(expanded.len() >= baseline + 2);
        assert!(snapshot_dashboard_frames(0).is_empty());
    }

    #[test]
    fn maintainer_report_merges_into_atlas() {
        let _guard = atlas_test_lock().lock().unwrap();
        clear_atlas();
        clear_atlas_route();
        clear_maintainer_report_for_test();

        #[allow(unused_mut)]
        let mut report = MaintainerReport {
            status: MaintainerStatus::Clamp,
            average_drift: 0.42,
            mean_energy: 1.75,
            mean_decay: -0.08,
            drift_peak: Some(ChronoPeak {
                frequency: 0.33,
                magnitude: 0.55,
                phase: 0.12,
            }),
            energy_peak: Some(ChronoPeak {
                frequency: 0.21,
                magnitude: 0.47,
                phase: -0.04,
            }),
            suggested_max_scale: Some(2.4),
            suggested_pressure: Some(0.18),
            diagnostic: "Clamp geometry around 2.4 while boosting pressure.".into(),
            #[cfg(feature = "kdsl")]
            spiralk_script: None,
        };

        #[cfg(feature = "kdsl")]
        {
            report.spiralk_script = Some("(maintain clamp)".into());
        }

        set_maintainer_report(report.clone());

        let atlas = get_atlas_frame().expect("maintainer atlas frame");
        assert_eq!(atlas.maintainer_status, Some(report.status));
        assert_eq!(
            atlas.maintainer_diagnostic.as_deref(),
            Some(report.diagnostic.as_str())
        );
        assert_eq!(atlas.suggested_max_scale, report.suggested_max_scale);
        assert_eq!(atlas.suggested_pressure, report.suggested_pressure);
        assert!(atlas
            .metrics
            .iter()
            .any(|metric| metric.name == "maintainer.average_drift"));
        assert!(atlas
            .notes
            .iter()
            .any(|note| note.starts_with("maintainer.status:")));

        let summary = get_atlas_route_summary(Some(1));
        assert_eq!(summary.maintainer_status, Some(report.status));
        assert_eq!(summary.suggested_max_scale, report.suggested_max_scale);
        assert_eq!(summary.suggested_pressure, report.suggested_pressure);
        assert_eq!(
            summary.maintainer_diagnostic.as_deref(),
            Some(report.diagnostic.as_str())
        );
        assert!(summary
            .districts
            .iter()
            .flat_map(|district| district.focus.iter())
            .any(|focus| focus.name.starts_with("maintainer.")));
    }
}
