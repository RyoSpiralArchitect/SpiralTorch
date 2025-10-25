// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Multi-branch MetaMEMB synchronisation primitives for Z-space PSI pipelines.
//!
//! This module ports the `MetaMEMB_SynchroMonolith.py` prototype to Rust so that
//! the PSI-capable Z-space coherence engine can synchronise several branches in
//! lockstep.  The original Python version combined an in-process pub/sub bus,
//! MetaMEMB samplers, κ(γ) identification, and Arnold tongue visualisation.  The
//! Rust implementation mirrors that architecture using lightweight threads and
//! channels so the components can run inside the SpiralTorch runtime without
//! Python bindings.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::f64::consts::PI;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

#[cfg(feature = "golden")]
use crate::golden::{GoldenBlackcatPulse, GoldenCooperativeDirective};
use st_core::telemetry::atlas::AtlasFragment;
use st_core::telemetry::hub::merge_atlas_fragment;
#[cfg(feature = "psi")]
use st_core::telemetry::hub::set_last_psi;
#[cfg(feature = "psi")]
use st_core::telemetry::psi::{PsiComponent, PsiReading};
use st_core::theory::zpulse::{ZPulse, ZScale, ZSource, ZSupport};

/// Per-branch dynamical parameters used to synthesise PSI samples.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PsiBranchState {
    pub branch_id: String,
    pub gamma: f64,
    pub lambda: f64,
    pub wd: f64,
    pub omega0: f64,
    pub drift_coupled: f64,
    pub phase0: f64,
}

impl PsiBranchState {
    pub fn new(branch_id: impl Into<String>) -> Self {
        Self {
            branch_id: branch_id.into(),
            gamma: 1.3,
            lambda: 1.0,
            wd: 0.7,
            omega0: 0.72,
            drift_coupled: 1.05,
            phase0: 0.0,
        }
    }

    fn poincare_period(&self) -> f64 {
        let wd = self.wd.max(1e-9);
        2.0 * PI / wd
    }
}

/// Global synchronisation state shared across branches.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SyncState {
    pub tau: f64,
    pub step: f64,
    pub seed: u64,
    pub epoch: u64,
    pub branches: BTreeMap<String, PsiBranchState>,
}

impl Default for SyncState {
    fn default() -> Self {
        Self {
            tau: 0.0,
            step: 0.01,
            seed: 42,
            epoch: 0,
            branches: BTreeMap::new(),
        }
    }
}

/// Events exchanged via the [`SynchroBus`].
#[derive(Clone, Debug)]
pub enum SynchroEvent {
    State(SyncState),
    BranchState(PsiBranchState),
    Tick {
        index: u64,
        tau: f64,
        step: f64,
    },
    PhaseSample {
        branch_id: String,
        sample_index: u64,
        t: f64,
        phi: f64,
        params: PsiBranchState,
    },
    KappaUpdate {
        branch_id: String,
        omega_hat: f64,
        k_hat: f64,
        kappa_hat: f64,
        rmse: f64,
        params: PsiBranchState,
    },
    HeatmapUpdate(HeatmapResult),
    Shutdown,
}

/// Lightweight in-process pub/sub bus for the synchroniser components.
#[derive(Clone)]
pub struct SynchroBus {
    state: Arc<Mutex<SyncState>>,
    subscribers: Arc<Mutex<Vec<Sender<SynchroEvent>>>>,
}

impl SynchroBus {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(SyncState::default())),
            subscribers: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn subscribe(&self) -> Receiver<SynchroEvent> {
        let (tx, rx) = mpsc::channel();
        self.subscribers.lock().unwrap().push(tx);
        rx
    }

    pub fn publish(&self, event: SynchroEvent) {
        let mut failed = Vec::new();
        let mut guard = self.subscribers.lock().unwrap();
        for (idx, sub) in guard.iter().enumerate() {
            if sub.send(event.clone()).is_err() {
                failed.push(idx);
            }
        }
        if !failed.is_empty() {
            for idx in failed.into_iter().rev() {
                guard.remove(idx);
            }
        }
    }

    pub fn update_state<F>(&self, mut f: F)
    where
        F: FnMut(&mut SyncState),
    {
        let snapshot = {
            let mut guard = self.state.lock().unwrap();
            f(&mut guard);
            guard.clone()
        };
        self.publish(SynchroEvent::State(snapshot));
    }

    pub fn insert_branch(&self, branch: PsiBranchState) {
        {
            let mut guard = self.state.lock().unwrap();
            guard
                .branches
                .insert(branch.branch_id.clone(), branch.clone());
        }
        self.publish(SynchroEvent::BranchState(branch));
    }

    pub fn snapshot(&self) -> SyncState {
        self.state.lock().unwrap().clone()
    }
}

/// Periodic ticker that drives the synchronisation bus.
pub struct Ticker {
    handle: Option<JoinHandle<()>>,
}

impl Ticker {
    pub fn spawn(bus: SynchroBus, steps: usize, interval: Option<Duration>) -> Self {
        let handle = thread::spawn(move || {
            for index in 0..steps {
                let (tau, step) = {
                    let mut guard = bus.state.lock().unwrap();
                    let tau = guard.tau;
                    let step = guard.step;
                    guard.tau += step;
                    (tau, step)
                };
                bus.publish(SynchroEvent::Tick {
                    index: index as u64,
                    tau,
                    step,
                });
                if let Some(delay) = interval {
                    thread::sleep(delay);
                }
            }
        });
        Self {
            handle: Some(handle),
        }
    }

    pub fn join(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// MetaMEMB oscillator used to generate φ samples.
#[derive(Debug, Clone)]
pub struct MetaMembConfig {
    pub delta: [f64; 3],
    pub omega: [f64; 3],
    pub theta: [f64; 3],
}

impl Default for MetaMembConfig {
    fn default() -> Self {
        Self {
            delta: [0.03, 0.05, 0.02],
            omega: [1.2, 0.9, 1.6],
            theta: [0.0, 1.0, 2.5],
        }
    }
}

#[derive(Debug, Clone)]
struct MetaMembModel {
    config: MetaMembConfig,
}

impl MetaMembModel {
    fn new(config: MetaMembConfig) -> Self {
        Self { config }
    }

    fn bloom(&self, tau: f64) -> f64 {
        let mut acc = 0.0;
        for i in 0..self.config.delta.len() {
            let delta = self.config.delta[i];
            let omega = self.config.omega[i];
            let theta = self.config.theta[i];
            acc += delta * (omega * tau + theta).sin();
        }
        acc
    }

    fn phi_sample(&self, t: f64, params: &PsiBranchState) -> f64 {
        let bloom = self.bloom(t);
        let psi = params.omega0 * t
            + params.lambda * params.drift_coupled.powf(params.gamma) * (5.0 * bloom).tanh();
        let theta = params.wd * t + params.phase0;
        let mut phi = (psi - theta) / (2.0 * PI);
        phi = (phi.rem_euclid(1.0) + 1.0).rem_euclid(1.0);
        phi
    }
}

struct BranchSamplerState {
    params: PsiBranchState,
    td: f64,
    next_sample_tau: f64,
    sample_index: u64,
}

impl BranchSamplerState {
    fn new(params: PsiBranchState, current_tau: f64) -> Self {
        let td = params.poincare_period();
        Self {
            params,
            td,
            next_sample_tau: current_tau,
            sample_index: 0,
        }
    }

    fn update(&mut self, params: PsiBranchState, current_tau: f64) {
        self.params = params;
        self.td = self.params.poincare_period();
        self.next_sample_tau = current_tau;
    }
}

/// Threaded MetaMEMB sampler that listens to ticks and emits φ samples.
pub struct MetaMembSampler {
    handle: Option<JoinHandle<()>>,
}

impl MetaMembSampler {
    pub fn spawn(bus: SynchroBus, config: MetaMembConfig) -> Self {
        let rx = bus.subscribe();
        let model = Arc::new(MetaMembModel::new(config));
        let bus_clone = bus.clone();
        let handle = thread::spawn(move || {
            let mut sampler_state: HashMap<String, BranchSamplerState> = HashMap::new();
            let mut current_tau = bus_clone.snapshot().tau;
            for (branch_id, params) in bus_clone.snapshot().branches.into_iter() {
                sampler_state.insert(
                    branch_id.clone(),
                    BranchSamplerState::new(params.clone(), current_tau),
                );
            }

            while let Ok(event) = rx.recv() {
                match event {
                    SynchroEvent::State(state) => {
                        current_tau = state.tau;
                        for (branch_id, params) in state.branches.into_iter() {
                            sampler_state
                                .entry(branch_id.clone())
                                .and_modify(|branch| branch.update(params.clone(), current_tau))
                                .or_insert_with(|| {
                                    BranchSamplerState::new(params.clone(), current_tau)
                                });
                        }
                    }
                    SynchroEvent::BranchState(params) => {
                        sampler_state
                            .entry(params.branch_id.clone())
                            .and_modify(|branch| branch.update(params.clone(), current_tau))
                            .or_insert_with(|| {
                                BranchSamplerState::new(params.clone(), current_tau)
                            });
                    }
                    SynchroEvent::Tick { tau, .. } => {
                        for (branch_id, branch_state) in sampler_state.iter_mut() {
                            while tau + 1e-12 >= branch_state.next_sample_tau {
                                let phi = model.phi_sample(tau, &branch_state.params);
                                bus_clone.publish(SynchroEvent::PhaseSample {
                                    branch_id: branch_id.clone(),
                                    sample_index: branch_state.sample_index,
                                    t: tau,
                                    phi,
                                    params: branch_state.params.clone(),
                                });
                                branch_state.sample_index += 1;
                                branch_state.next_sample_tau += branch_state.td;
                            }
                        }
                    }
                    SynchroEvent::Shutdown => break,
                    _ => {}
                }
            }
        });

        Self {
            handle: Some(handle),
        }
    }

    pub fn join(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Summary of a single Arnold tongue peak.
#[derive(Clone, Debug)]
pub struct ArnoldTongueSummary {
    pub ratio_p: i64,
    pub ratio_q: i64,
    pub rotation: f64,
    pub lam: f64,
    pub wd: f64,
    pub strength: f64,
    pub peak_strength: f64,
    pub error: f64,
}

impl ArnoldTongueSummary {
    pub fn ratio(&self) -> f64 {
        if self.ratio_q == 0 {
            0.0
        } else {
            self.ratio_p as f64 / self.ratio_q as f64
        }
    }
}

/// Analytics derived from an Arnold tongue heatmap.
#[derive(Clone, Debug)]
pub struct HeatmapAnalytics {
    pub total_energy: f64,
    pub leading_sum: f64,
    pub central_sum: f64,
    pub trailing_sum: f64,
    pub leading_norm: f32,
    pub central_norm: f32,
    pub trailing_norm: f32,
    pub dominant_lam: f64,
    pub dominant_wd: f64,
    pub peak_value: f64,
    pub peak_ratio: f32,
    pub radius: f32,
    pub log_radius: f32,
    pub bias: f32,
    pub drift: f32,
    pub quality: f32,
    pub stderr: f32,
    pub entropy: f32,
}

/// Resulting heatmap for a particular branch.
#[derive(Clone, Debug)]
pub struct HeatmapResult {
    pub branch_id: String,
    pub gamma: f64,
    pub kappa_hat: f64,
    pub lam_grid: Vec<f64>,
    pub wd_grid: Vec<f64>,
    pub matrix: Vec<Vec<f64>>,
    pub tongues: Vec<ArnoldTongueSummary>,
}

impl HeatmapResult {
    pub fn dominant_tongue(&self) -> Option<&ArnoldTongueSummary> {
        self.tongues
            .iter()
            .max_by(|a, b| match a.peak_strength.partial_cmp(&b.peak_strength) {
                Some(order) => order,
                None => Ordering::Equal,
            })
    }

    /// Computes aggregate analytics derived from the Arnold tongue matrix.
    pub fn analyse(&self) -> Option<HeatmapAnalytics> {
        if self.matrix.is_empty() || self.matrix[0].is_empty() {
            return None;
        }

        let lam_bins = self.matrix.len();
        let mut lam_profile = vec![0.0f64; lam_bins];
        let mut total_energy = 0.0f64;
        let mut max_value = f64::MIN;
        let mut max_pos = (0usize, 0usize);

        for (row_idx, row) in self.matrix.iter().enumerate() {
            let mut row_sum = 0.0f64;
            for (col_idx, value) in row.iter().enumerate() {
                let clamped = if value.is_finite() {
                    value.max(0.0)
                } else {
                    0.0
                };
                row_sum += clamped;
                total_energy += clamped;
                if clamped > max_value {
                    max_value = clamped;
                    max_pos = (row_idx, col_idx);
                }
            }
            lam_profile[row_idx] = row_sum;
        }

        if !total_energy.is_finite() || total_energy <= f64::EPSILON {
            return None;
        }

        let mut leading_sum = 0.0f64;
        let mut central_sum = 0.0f64;
        let mut trailing_sum = 0.0f64;
        if lam_bins < 3 {
            central_sum = total_energy;
        } else {
            let leading_end = lam_bins / 3;
            let trailing_start = lam_bins.saturating_sub(lam_bins / 3);
            for (idx, sum) in lam_profile.iter().copied().enumerate() {
                if idx < leading_end {
                    leading_sum += sum;
                } else if idx >= trailing_start {
                    trailing_sum += sum;
                } else {
                    central_sum += sum;
                }
            }
        }

        let normalise = |value: f64| -> f32 {
            if total_energy <= f64::EPSILON {
                0.0
            } else {
                (value / total_energy).max(0.0).min(1.0) as f32
            }
        };

        let leading_norm = normalise(leading_sum);
        let central_norm = normalise(central_sum);
        let trailing_norm = normalise(trailing_sum);

        let (dominant_lam, dominant_wd, peak_value) = if let Some(tongue) = self.dominant_tongue() {
            (tongue.lam, tongue.wd, tongue.peak_strength)
        } else {
            let lam = self
                .lam_grid
                .get(max_pos.0)
                .copied()
                .or_else(|| self.lam_grid.first().copied())
                .unwrap_or(0.0);
            let wd = self
                .wd_grid
                .get(max_pos.1)
                .copied()
                .or_else(|| self.wd_grid.first().copied())
                .unwrap_or(0.0);
            (lam, wd, max_value)
        };

        let radius = (dominant_lam + 1.0).max(1e-3) as f32;
        let log_radius = radius.ln();

        let bias = if (leading_sum + trailing_sum).abs() <= f64::EPSILON {
            0.0
        } else {
            ((leading_sum - trailing_sum) / (leading_sum + trailing_sum)).clamp(-1.0, 1.0) as f32
        };

        let drift = self.kappa_hat.tanh() as f32;
        let peak_ratio = (peak_value / total_energy).clamp(0.0, 1.0) as f32;
        let quality = peak_ratio;
        let stderr = (1.0 - quality).clamp(0.0, 1.0);

        let mut entropy = 0.0f32;
        let distribution = [leading_norm, central_norm, trailing_norm];
        let normaliser = (distribution.len() as f32).ln().max(f32::EPSILON);
        for &p in distribution.iter() {
            if p > f32::EPSILON {
                entropy -= p * p.ln();
            }
        }
        entropy = (entropy / normaliser).clamp(0.0, 1.0);

        Some(HeatmapAnalytics {
            total_energy,
            leading_sum,
            central_sum,
            trailing_sum,
            leading_norm,
            central_norm,
            trailing_norm,
            dominant_lam,
            dominant_wd,
            peak_value,
            peak_ratio,
            radius,
            log_radius,
            bias,
            drift,
            quality,
            stderr,
            entropy,
        })
    }

    /// Converts the heatmap into a synthetic ZPulse snapshot for downstream Z-space consumers.
    pub fn to_zpulse(&self, ts: u64) -> ZPulse {
        let mut pulse = ZPulse::default();
        pulse.source = ZSource::Other("psi");
        pulse.ts = ts;

        let Some(analytics) = self.analyse() else {
            return pulse;
        };

        pulse.band_energy = (
            analytics.leading_norm,
            analytics.central_norm,
            analytics.trailing_norm,
        );
        pulse.support = ZSupport::new(
            analytics.leading_norm,
            analytics.central_norm,
            analytics.trailing_norm,
        );
        let scale =
            ZScale::from_components(analytics.radius, analytics.log_radius).unwrap_or(ZScale::ONE);
        pulse.scale = Some(scale);
        pulse.tempo = analytics.dominant_wd as f32;
        pulse.z_bias = analytics.bias;
        pulse.drift = analytics.drift;
        pulse.quality = analytics.quality;
        pulse.stderr = analytics.stderr;
        pulse.latency_ms = 0.0;
        pulse
    }

    /// Builds an atlas fragment carrying the synchroniser metrics.
    pub fn to_atlas_fragment(&self, timestamp: Option<f32>) -> Option<AtlasFragment> {
        let analytics = self.analyse()?;
        let mut fragment = AtlasFragment::default();
        fragment.timestamp = timestamp;
        fragment.push_metric_with_district("psi.synchro.gamma", self.gamma as f32, "psi");
        fragment.push_metric_with_district("psi.synchro.kappa_hat", self.kappa_hat as f32, "psi");
        fragment.push_metric_with_district(
            "psi.synchro.energy.leading",
            analytics.leading_norm,
            "psi",
        );
        fragment.push_metric_with_district(
            "psi.synchro.energy.central",
            analytics.central_norm,
            "psi",
        );
        fragment.push_metric_with_district(
            "psi.synchro.energy.trailing",
            analytics.trailing_norm,
            "psi",
        );
        fragment.push_metric_with_district("psi.synchro.bias", analytics.bias, "psi");
        fragment.push_metric_with_district("psi.synchro.drift", analytics.drift, "psi");
        fragment.push_metric_with_district("psi.synchro.quality", analytics.quality, "psi");
        fragment.push_metric_with_district("psi.synchro.entropy", analytics.entropy, "psi");
        fragment.push_metric_with_district("psi.synchro.radius", analytics.radius, "psi");
        fragment.push_metric_with_district("psi.synchro.log_radius", analytics.log_radius, "psi");
        fragment.push_note(format!("psi.synchro.branch:{}", self.branch_id));
        if let Some(tongue) = self.dominant_tongue() {
            fragment.push_metric_with_district(
                "psi.synchro.tongue.peak",
                tongue.peak_strength as f32,
                "psi",
            );
            fragment.push_metric_with_district(
                "psi.synchro.tongue.error",
                tongue.error as f32,
                "psi",
            );
            fragment.push_metric_with_district(
                "psi.synchro.tongue.rotation",
                tongue.rotation as f32,
                "psi",
            );
            fragment.push_note(format!(
                "psi.synchro.tongue:{}:{}/{}",
                self.branch_id, tongue.ratio_p, tongue.ratio_q
            ));
        }
        Some(fragment)
    }

    #[cfg(feature = "psi")]
    pub fn to_psi_reading(&self, step: u64) -> Option<PsiReading> {
        let analytics = self.analyse()?;
        let mut breakdown = HashMap::new();
        breakdown.insert(PsiComponent::BAND_ENERGY, analytics.central_norm);
        breakdown.insert(
            PsiComponent::LOSS,
            (1.0 - analytics.quality).clamp(0.0, 1.0),
        );
        breakdown.insert(PsiComponent::GRAD_NORM, analytics.drift.abs().min(1.0));
        breakdown.insert(PsiComponent::UPDATE_RATIO, analytics.peak_ratio);
        breakdown.insert(PsiComponent::ACT_DRIFT, analytics.bias.abs().min(1.0));
        breakdown.insert(PsiComponent::ATTN_ENTROPY, analytics.entropy);
        let total = breakdown.values().copied().sum::<f32>().max(f32::EPSILON);
        Some(PsiReading {
            total,
            breakdown,
            step,
        })
    }

    #[cfg(feature = "golden")]
    pub fn to_golden_pulse(&self) -> Option<GoldenBlackcatPulse> {
        let analytics = self.analyse()?;
        let coverage =
            self.matrix.len() * self.matrix.first().map(|row| row.len()).unwrap_or_default();
        let heuristics_contributions = self.tongues.len();
        let dominant_plan = self
            .dominant_tongue()
            .map(|tongue| format!("{}:{}/{}", self.branch_id, tongue.ratio_p, tongue.ratio_q));

        Some(GoldenBlackcatPulse {
            exploration_drive: analytics.leading_norm,
            optimization_gain: analytics.central_norm,
            synergy_score: analytics.trailing_norm,
            reinforcement_weight: analytics.quality,
            mean_support: analytics.central_norm,
            mean_reward: analytics.peak_ratio as f64,
            mean_psi: analytics.drift,
            mean_confidence: (1.0 - analytics.stderr).clamp(0.0, 1.0),
            coverage,
            heuristics_contributions,
            append_weight: analytics.radius,
            retract_count: 0,
            annotate_count: heuristics_contributions,
            dominant_plan,
        })
    }
}

/// PSI branch pulse derived from the Arnold tongue heatmap.
#[derive(Clone, Debug)]
pub struct PsiSynchroPulse {
    pub branch_id: String,
    pub pulse: ZPulse,
}

/// Atlas fragment paired with the originating PSI branch.
#[derive(Clone, Debug)]
pub struct BranchAtlasFragment {
    pub branch_id: String,
    pub fragment: AtlasFragment,
}

/// PSI telemetry snapshot tagged with the source branch.
#[cfg(feature = "psi")]
#[derive(Clone, Debug)]
pub struct BranchPsiReading {
    pub branch_id: String,
    pub reading: PsiReading,
}

/// Combined outputs from a multi-branch synchronisation run.
#[derive(Clone, Debug)]
pub struct PsiSynchroResult {
    pub heatmaps: Vec<HeatmapResult>,
    pub pulses: Vec<PsiSynchroPulse>,
    pub atlas_fragments: Vec<BranchAtlasFragment>,
    #[cfg(feature = "psi")]
    pub psi_readings: Vec<BranchPsiReading>,
    #[cfg(feature = "golden")]
    pub golden_baseline_interval: Duration,
    #[cfg(feature = "golden")]
    pub golden_baseline_window: usize,
    #[cfg(feature = "golden")]
    pub golden_telemetry: Vec<PsiGoldenTelemetry>,
}

/// Converts heatmap outputs into ZPulse snapshots tagged with their originating branch.
pub fn heatmaps_to_zpulses(heatmaps: &[HeatmapResult]) -> Vec<PsiSynchroPulse> {
    heatmaps
        .iter()
        .enumerate()
        .map(|(idx, heatmap)| PsiSynchroPulse {
            branch_id: heatmap.branch_id.clone(),
            pulse: heatmap.to_zpulse(idx as u64),
        })
        .collect()
}

#[cfg(feature = "golden")]
#[derive(Clone, Debug)]
pub struct PsiGoldenTelemetry {
    pub branch_id: String,
    pub pulse: GoldenBlackcatPulse,
    pub directive: GoldenCooperativeDirective,
}

#[cfg(feature = "golden")]
pub fn heatmaps_to_golden_telemetry(
    heatmaps: &[HeatmapResult],
    baseline_interval: Duration,
    baseline_window: usize,
) -> Vec<PsiGoldenTelemetry> {
    heatmaps
        .iter()
        .filter_map(|heatmap| {
            let pulse = heatmap.to_golden_pulse()?;
            let directive = pulse.directive(baseline_interval, baseline_window);
            Some(PsiGoldenTelemetry {
                branch_id: heatmap.branch_id.clone(),
                pulse,
                directive,
            })
        })
        .collect()
}

/// Parameters controlling the Arnold tongue heatmap computation.
#[derive(Clone, Debug)]
pub struct CircleLockMapConfig {
    pub lam_min: f64,
    pub lam_max: f64,
    pub lam_bins: usize,
    pub wd_min: f64,
    pub wd_max: f64,
    pub wd_bins: usize,
    pub burn_in: usize,
    pub samples: usize,
    pub qmax: usize,
}

impl Default for CircleLockMapConfig {
    fn default() -> Self {
        Self {
            lam_min: 0.0,
            lam_max: 2.0,
            lam_bins: 60,
            wd_min: 0.3,
            wd_max: 1.2,
            wd_bins: 80,
            burn_in: 200,
            samples: 300,
            qmax: 8,
        }
    }
}

fn wrap_delta(delta: f64) -> f64 {
    let mut d = (delta + 0.5).rem_euclid(1.0);
    d -= 0.5;
    d
}

fn rotation_number(omega: f64, kappa: f64, burn: usize, samples: usize) -> f64 {
    let mut phi = 0.0;
    for _ in 0..burn {
        phi = (phi + omega - (kappa / (2.0 * PI)) * (2.0 * PI * phi).sin()).rem_euclid(1.0);
    }
    let mut acc = 0.0;
    let mut prev = phi;
    for _ in 0..samples {
        phi = (phi + omega - (kappa / (2.0 * PI)) * (2.0 * PI * phi).sin()).rem_euclid(1.0);
        acc += (phi - prev).rem_euclid(1.0);
        prev = phi;
    }
    acc / samples as f64
}

fn nearest_rational(value: f64, qmax: usize) -> (i64, i64, f64) {
    let mut best = (0, 1, f64::MAX);
    for q in 1..=qmax.max(1) as i64 {
        let p = (value * q as f64).round() as i64;
        let err = (value - p as f64 / q as f64).abs();
        if err < best.2 {
            best = (p, q, err);
        }
    }
    best
}

fn identify_branch(phi: &[f64]) -> Option<(f64, f64, f64)> {
    if phi.len() < 2 {
        return None;
    }

    let mut sum_y = 0.0;
    let mut sum_sin = 0.0;
    let mut sum_sin2 = 0.0;
    let mut sum_y_sin = 0.0;
    let mut count = 0.0;

    for window in phi.windows(2) {
        let x = window[0];
        let y = wrap_delta(window[1] - window[0]);
        let s = (2.0 * PI * x).sin();
        count += 1.0;
        sum_y += y;
        sum_sin += s;
        sum_sin2 += s * s;
        sum_y_sin += y * s;
    }

    if count < 2.0 {
        return None;
    }

    let n = count;
    let det = n * sum_sin2 - sum_sin * sum_sin;
    if det.abs() <= f64::EPSILON {
        return None;
    }

    let omega_hat = (sum_sin2 * sum_y - sum_sin * sum_y_sin) / det;
    let a_hat = (n * sum_y_sin - sum_sin * sum_y) / det;

    let mut rmse_acc = 0.0;
    for window in phi.windows(2) {
        let x = window[0];
        let y = wrap_delta(window[1] - window[0]);
        let s = (2.0 * PI * x).sin();
        let pred = omega_hat + a_hat * s;
        rmse_acc += (y - pred).powi(2);
    }
    let rmse = (rmse_acc / n).sqrt();
    let k_hat = -2.0 * PI * a_hat;
    Some((omega_hat, k_hat, rmse))
}

/// Least-squares κ̂(γ) estimator for each branch.
pub struct KappaIdentifier {
    handle: Option<JoinHandle<()>>,
}

impl KappaIdentifier {
    pub fn spawn(bus: SynchroBus, min_points: usize, max_points: usize) -> Self {
        let rx = bus.subscribe();
        let bus_clone = bus.clone();
        let handle = thread::spawn(move || {
            let mut buffers: HashMap<String, VecDeque<f64>> = HashMap::new();
            let mut params_cache: HashMap<String, PsiBranchState> = HashMap::new();
            let capacity = max_points.max(min_points).max(2);

            while let Ok(event) = rx.recv() {
                match event {
                    SynchroEvent::PhaseSample {
                        branch_id,
                        phi,
                        params,
                        ..
                    } => {
                        let buf = buffers
                            .entry(branch_id.clone())
                            .or_insert_with(|| VecDeque::with_capacity(capacity));
                        if buf.len() == capacity {
                            buf.pop_front();
                        }
                        buf.push_back(phi);
                        params_cache.insert(branch_id.clone(), params.clone());
                        if buf.len() >= min_points {
                            if let Some((omega_hat, k_hat, rmse)) =
                                identify_branch(&buf.iter().copied().collect::<Vec<_>>())
                            {
                                if let Some(branch_params) = params_cache.get(&branch_id) {
                                    let lambda = branch_params.lambda.max(1e-9);
                                    let kappa_hat = k_hat / lambda;
                                    bus_clone.publish(SynchroEvent::KappaUpdate {
                                        branch_id,
                                        omega_hat,
                                        k_hat,
                                        kappa_hat,
                                        rmse,
                                        params: branch_params.clone(),
                                    });
                                }
                            }
                        }
                    }
                    SynchroEvent::BranchState(params) => {
                        params_cache.insert(params.branch_id.clone(), params);
                    }
                    SynchroEvent::State(state) => {
                        for (branch_id, params) in state.branches.into_iter() {
                            params_cache.insert(branch_id, params);
                        }
                    }
                    SynchroEvent::Shutdown => break,
                    _ => {}
                }
            }
        });

        Self {
            handle: Some(handle),
        }
    }

    pub fn join(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Arnold tongue estimator that emits heatmap matrices for each branch.
pub struct CircleLockMap {
    handle: Option<JoinHandle<()>>,
}

impl CircleLockMap {
    pub fn spawn(bus: SynchroBus, config: CircleLockMapConfig) -> Self {
        let rx = bus.subscribe();
        let bus_clone = bus.clone();
        let handle = thread::spawn(move || {
            while let Ok(event) = rx.recv() {
                match event {
                    SynchroEvent::KappaUpdate {
                        branch_id,
                        params,
                        kappa_hat,
                        ..
                    } => {
                        let lam_grid = linspace(config.lam_min, config.lam_max, config.lam_bins);
                        let wd_grid = linspace(config.wd_min, config.wd_max, config.wd_bins);
                        let mut matrix = vec![vec![0.0; wd_grid.len()]; lam_grid.len()];
                        #[derive(Clone, Copy)]
                        struct TongueAccumulator {
                            total_strength: f64,
                            peak_strength: f64,
                            error: f64,
                            lam: f64,
                            wd: f64,
                            rotation: f64,
                        }
                        let mut tongues: HashMap<(i64, i64), TongueAccumulator> = HashMap::new();
                        for (il, lam) in lam_grid.iter().copied().enumerate() {
                            let kappa = kappa_hat * lam;
                            for (iw, wd) in wd_grid.iter().copied().enumerate() {
                                let omega = params.omega0 / wd - 1.0;
                                let rho =
                                    rotation_number(omega, kappa, config.burn_in, config.samples);
                                let (p, q, err) = nearest_rational(rho, config.qmax);
                                let strength = 1.0 / (err + 1e-6);
                                matrix[il][iw] = strength;
                                if q != 0 {
                                    let entry =
                                        tongues.entry((p, q)).or_insert(TongueAccumulator {
                                            total_strength: 0.0,
                                            peak_strength: f64::MIN,
                                            error: err,
                                            lam,
                                            wd,
                                            rotation: rho,
                                        });
                                    entry.total_strength += strength;
                                    let is_better_peak = strength > entry.peak_strength
                                        || (strength == entry.peak_strength && err < entry.error);
                                    if is_better_peak {
                                        entry.peak_strength = strength;
                                        entry.error = err;
                                        entry.lam = lam;
                                        entry.wd = wd;
                                        entry.rotation = rho;
                                    }
                                }
                            }
                        }
                        let mut tongues: Vec<ArnoldTongueSummary> = tongues
                            .into_iter()
                            .map(|((p, q), acc)| ArnoldTongueSummary {
                                ratio_p: p,
                                ratio_q: q,
                                rotation: acc.rotation,
                                lam: acc.lam,
                                wd: acc.wd,
                                strength: acc.total_strength,
                                peak_strength: acc.peak_strength,
                                error: acc.error,
                            })
                            .collect();
                        tongues.sort_by(|a, b| {
                            match b.peak_strength.partial_cmp(&a.peak_strength) {
                                Some(order) => order,
                                None => Ordering::Equal,
                            }
                        });
                        bus_clone.publish(SynchroEvent::HeatmapUpdate(HeatmapResult {
                            branch_id,
                            gamma: params.gamma,
                            kappa_hat,
                            lam_grid,
                            wd_grid,
                            matrix,
                            tongues,
                        }));
                    }
                    SynchroEvent::Shutdown => break,
                    _ => {}
                }
            }
        });

        Self {
            handle: Some(handle),
        }
    }

    pub fn join(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn linspace(min: f64, max: f64, bins: usize) -> Vec<f64> {
    if bins <= 1 {
        return vec![min];
    }
    let mut values = Vec::with_capacity(bins);
    let step = (max - min) / (bins as f64 - 1.0);
    for i in 0..bins {
        values.push(min + i as f64 * step);
    }
    values
}

struct HeatmapCollector {
    handle: Option<JoinHandle<()>>,
    results: Arc<Mutex<BTreeMap<String, HeatmapResult>>>,
}

impl HeatmapCollector {
    fn spawn(bus: SynchroBus) -> Self {
        let rx = bus.subscribe();
        let results = Arc::new(Mutex::new(BTreeMap::new()));
        let results_clone = Arc::clone(&results);
        let handle = thread::spawn(move || {
            while let Ok(event) = rx.recv() {
                match event {
                    SynchroEvent::HeatmapUpdate(result) => {
                        results_clone
                            .lock()
                            .unwrap()
                            .insert(result.branch_id.clone(), result);
                    }
                    SynchroEvent::Shutdown => break,
                    _ => {}
                }
            }
        });
        Self {
            handle: Some(handle),
            results,
        }
    }

    fn join(mut self) -> Vec<HeatmapResult> {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        self.results.lock().unwrap().values().cloned().collect()
    }
}

/// Controls how the synchroniser reports atlas/PSI telemetry.
#[derive(Clone, Debug)]
pub struct PsiTelemetryConfig {
    pub emit_atlas: bool,
    pub atlas_timestamp: Option<f32>,
    #[cfg(feature = "psi")]
    pub emit_psi: bool,
    #[cfg(feature = "psi")]
    pub psi_step_base: u64,
    #[cfg(feature = "golden")]
    pub emit_golden: bool,
    #[cfg(feature = "golden")]
    pub golden_baseline_interval: Duration,
    #[cfg(feature = "golden")]
    pub golden_baseline_window: usize,
}

impl Default for PsiTelemetryConfig {
    fn default() -> Self {
        Self {
            emit_atlas: true,
            atlas_timestamp: None,
            #[cfg(feature = "psi")]
            emit_psi: true,
            #[cfg(feature = "psi")]
            psi_step_base: 0,
            #[cfg(feature = "golden")]
            emit_golden: true,
            #[cfg(feature = "golden")]
            golden_baseline_interval: Duration::from_secs(30),
            #[cfg(feature = "golden")]
            golden_baseline_window: 32,
        }
    }
}

/// Orchestration parameters for the multi-branch synchroniser.
#[derive(Clone, Debug)]
pub struct PsiSynchroConfig {
    pub step: f64,
    pub samples: usize,
    pub ticker_interval: Option<Duration>,
    pub min_ident_points: usize,
    pub max_ident_points: usize,
    pub metamemb: MetaMembConfig,
    pub circle_map: CircleLockMapConfig,
    pub telemetry: Option<PsiTelemetryConfig>,
}

impl Default for PsiSynchroConfig {
    fn default() -> Self {
        Self {
            step: 0.01,
            samples: 1_000,
            ticker_interval: None,
            min_ident_points: 600,
            max_ident_points: 2_400,
            metamemb: MetaMembConfig::default(),
            circle_map: CircleLockMapConfig::default(),
            telemetry: None,
        }
    }
}

/// Run the multi-branch synchroniser and return the latest heatmaps per branch.
pub fn run_multibranch_demo(
    config: PsiSynchroConfig,
    branches: Vec<PsiBranchState>,
) -> Vec<HeatmapResult> {
    let result = run_zspace_learning_pass(config, branches);
    result.heatmaps
}

fn execute_multibranch(
    config: &PsiSynchroConfig,
    branches: &[PsiBranchState],
) -> Vec<HeatmapResult> {
    let bus = SynchroBus::new();
    bus.update_state(|state| {
        state.tau = 0.0;
        state.step = config.step;
        state.branches.clear();
    });
    for branch in branches.iter().cloned() {
        bus.insert_branch(branch);
    }
    bus.publish(SynchroEvent::State(bus.snapshot()));

    let mut sampler = MetaMembSampler::spawn(bus.clone(), config.metamemb.clone());
    let mut identifier = KappaIdentifier::spawn(
        bus.clone(),
        config.min_ident_points,
        config.max_ident_points,
    );
    let mut circle_map = CircleLockMap::spawn(bus.clone(), config.circle_map.clone());
    let collector = HeatmapCollector::spawn(bus.clone());

    let total_time = branches
        .iter()
        .map(|branch| branch.poincare_period() * config.samples as f64)
        .fold(0.0, f64::max);
    let steps = (total_time / config.step).ceil() as usize + 1;
    let mut ticker = Ticker::spawn(bus.clone(), steps, config.ticker_interval);
    ticker.join();

    bus.publish(SynchroEvent::Shutdown);

    sampler.join();
    identifier.join();
    circle_map.join();

    collector.join()
}

/// Executes a PSI synchronisation pass and returns telemetry suitable for Z-space learning.
pub fn run_zspace_learning_pass(
    config: PsiSynchroConfig,
    branches: Vec<PsiBranchState>,
) -> PsiSynchroResult {
    let telemetry_cfg = config.telemetry.clone();
    let heatmaps = execute_multibranch(&config, &branches);
    let pulses = heatmaps_to_zpulses(&heatmaps);

    let atlas_fragments: Vec<BranchAtlasFragment> = heatmaps
        .iter()
        .filter_map(|heatmap| {
            heatmap
                .to_atlas_fragment(telemetry_cfg.as_ref().and_then(|cfg| cfg.atlas_timestamp))
                .map(|fragment| BranchAtlasFragment {
                    branch_id: heatmap.branch_id.clone(),
                    fragment,
                })
        })
        .collect();

    #[cfg(feature = "psi")]
    let psi_readings: Vec<BranchPsiReading> = {
        let base = telemetry_cfg
            .as_ref()
            .map(|cfg| cfg.psi_step_base)
            .unwrap_or(0);
        heatmaps
            .iter()
            .enumerate()
            .filter_map(|(idx, heatmap)| {
                heatmap
                    .to_psi_reading(base + idx as u64)
                    .map(|reading| BranchPsiReading {
                        branch_id: heatmap.branch_id.clone(),
                        reading,
                    })
            })
            .collect()
    };

    #[cfg(feature = "golden")]
    let (golden_baseline_interval, golden_baseline_window, golden_telemetry) = {
        let baseline_interval = telemetry_cfg
            .as_ref()
            .map(|cfg| cfg.golden_baseline_interval)
            .unwrap_or_else(|| Duration::from_secs(30));
        let baseline_window = telemetry_cfg
            .as_ref()
            .map(|cfg| cfg.golden_baseline_window)
            .unwrap_or(32);
        let telemetry = heatmaps_to_golden_telemetry(&heatmaps, baseline_interval, baseline_window);
        (baseline_interval, baseline_window, telemetry)
    };

    if let Some(cfg) = telemetry_cfg {
        if cfg.emit_atlas {
            for fragment in &atlas_fragments {
                let atlas_fragment: AtlasFragment = fragment.fragment.clone();
                merge_atlas_fragment(atlas_fragment);
            }
        }
        #[cfg(feature = "psi")]
        if cfg.emit_psi {
            for reading in &psi_readings {
                set_last_psi(&reading.reading);
            }
        }
    }

    PsiSynchroResult {
        heatmaps,
        pulses,
        atlas_fragments,
        #[cfg(feature = "psi")]
        psi_readings,
        #[cfg(feature = "golden")]
        golden_baseline_interval,
        #[cfg(feature = "golden")]
        golden_baseline_window,
        #[cfg(feature = "golden")]
        golden_telemetry,
    }
}
