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

#![cfg(feature = "golden")]

use crate::dataset::DataLoader;
use crate::loss::Loss;
use crate::module::Module;
use crate::roundtable::{HeurOp, HeurOpKind, HeurOpLog, ModeratorMinutes};
use crate::schedule::RoundtableSchedule;
use crate::trainer::{EpochStats, ModuleTrainer};
use crate::PureResult;
use st_core::runtime::golden::{
    GoldenRuntime, GoldenRuntimeConfig, GoldenRuntimeError, SpiralMutex,
};
use st_tensor::pure::TensorError;
use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, SystemTime};

#[derive(Debug, Clone)]
pub struct GoldenRetrieverConfig {
    pub workers: usize,
    pub runtime: Option<GoldenRuntimeConfig>,
    pub sync_blackcat_minutes: bool,
    pub sync_heuristics_log: bool,
    pub coordinate_blackcat: bool,
    pub exploration_bias: f32,
    pub optimization_boost: f32,
    pub synergy_bias: f32,
    pub reinforcement_bias: f32,
    pub self_rewrite: Option<GoldenSelfRewriteConfig>,
}

impl Default for GoldenRetrieverConfig {
    fn default() -> Self {
        Self {
            workers: 0,
            runtime: None,
            sync_blackcat_minutes: false,
            sync_heuristics_log: false,
            coordinate_blackcat: false,
            exploration_bias: 1.0,
            optimization_boost: 1.0,
            synergy_bias: 1.0,
            reinforcement_bias: 1.0,
            self_rewrite: None,
        }
    }
}

impl GoldenRetrieverConfig {
    pub fn with_self_rewrite(mut self, config: GoldenSelfRewriteConfig) -> Self {
        self.self_rewrite = Some(config);
        self
    }

    pub fn rewrite_with_scheduler(
        &mut self,
        schedules: &[RoundtableSchedule],
        pulse: Option<&GoldenBlackcatPulse>,
    ) {
        let Some(rewrite) = self.self_rewrite.clone() else {
            return;
        };
        let schedule_signal = GoldenScheduleSignal::from_slice(schedules);
        let mut council = GoldenSelfRewriteState::new(
            rewrite,
            GoldenBiasVector::from_biases(
                self.exploration_bias,
                self.optimization_boost,
                self.synergy_bias,
                self.reinforcement_bias,
            ),
        );
        let idle;
        let empty_log = HeurOpLog::default();
        let pulse_ref = match pulse {
            Some(pulse) => pulse,
            None => {
                idle = GoldenBlackcatPulse::idle();
                &idle
            }
        };
        let GoldenCouncilResolution { biases, .. } =
            council.negotiate(0, pulse_ref, &schedule_signal, &empty_log);
        self.exploration_bias = biases.exploration;
        self.optimization_boost = biases.optimization;
        self.synergy_bias = biases.synergy;
        self.reinforcement_bias = biases.reinforcement;
        self.self_rewrite = Some(council.into_config());
    }
}

struct GoldenWorker {
    _id: usize,
    trainer: SpiralMutex<ModuleTrainer>,
}

#[derive(Debug, Clone)]
pub struct GoldenSelfRewriteConfig {
    pub negotiation_rate: f32,
    pub schedule_weight: f32,
    pub inertia: f32,
    pub max_bias: f32,
    pub council_memory: usize,
    pub schedule_resonance: f32,
    pub synergy_pressure: f32,
}

impl Default for GoldenSelfRewriteConfig {
    fn default() -> Self {
        Self {
            negotiation_rate: 0.35,
            schedule_weight: 0.5,
            inertia: 0.65,
            max_bias: 12.0,
            council_memory: 6,
            schedule_resonance: 0.75,
            synergy_pressure: 0.4,
        }
    }
}

impl GoldenSelfRewriteConfig {
    pub fn with_negotiation_rate(mut self, rate: f32) -> Self {
        self.negotiation_rate = rate.max(0.0);
        self
    }

    pub fn with_schedule_weight(mut self, weight: f32) -> Self {
        self.schedule_weight = weight.max(0.0);
        self
    }

    pub fn with_inertia(mut self, inertia: f32) -> Self {
        self.inertia = inertia.clamp(0.0, 0.995);
        self
    }

    pub fn with_max_bias(mut self, max_bias: f32) -> Self {
        self.max_bias = max_bias.max(1.0);
        self
    }

    pub fn with_council_memory(mut self, slots: usize) -> Self {
        self.council_memory = slots.max(2);
        self
    }

    pub fn with_schedule_resonance(mut self, resonance: f32) -> Self {
        self.schedule_resonance = resonance.max(0.0);
        self
    }

    pub fn with_synergy_pressure(mut self, pressure: f32) -> Self {
        self.synergy_pressure = pressure.max(0.0);
        self
    }
}

#[derive(Debug, Clone, Copy)]
struct GoldenBiasVector {
    exploration: f32,
    optimization: f32,
    synergy: f32,
    reinforcement: f32,
}

impl GoldenBiasVector {
    fn from_biases(exploration: f32, optimization: f32, synergy: f32, reinforcement: f32) -> Self {
        Self {
            exploration,
            optimization,
            synergy,
            reinforcement,
        }
    }

    fn from_pulse(pulse: &GoldenBlackcatPulse) -> Self {
        Self {
            exploration: pulse.exploration_drive.max(0.0) + pulse.synergy_score.max(0.0) * 0.1,
            optimization: pulse.optimization_gain.max(0.0),
            synergy: pulse.synergy_score.max(0.0),
            reinforcement: pulse.reinforcement_weight.max(0.0),
        }
    }

    fn apply_inertia(self, previous: GoldenBiasVector, inertia: f32) -> Self {
        let keep = inertia.clamp(0.0, 0.995);
        let inject = 1.0 - keep;
        Self {
            exploration: previous.exploration * keep + self.exploration * inject,
            optimization: previous.optimization * keep + self.optimization * inject,
            synergy: previous.synergy * keep + self.synergy * inject,
            reinforcement: previous.reinforcement * keep + self.reinforcement * inject,
        }
    }

    fn clamp(self, min: f32, max: f32) -> Self {
        let (min, max) = (min.max(0.01), max.max(min));
        Self {
            exploration: self.exploration.clamp(min, max),
            optimization: self.optimization.clamp(min, max),
            synergy: self.synergy.clamp(min, max),
            reinforcement: self.reinforcement.clamp(min, max),
        }
    }

    fn stabilise(self) -> Self {
        let mean = (self.exploration + self.optimization + self.synergy + self.reinforcement) / 4.0;
        if mean <= f32::EPSILON {
            return Self::from_biases(1.0, 1.0, 1.0, 1.0);
        }
        let scale = (1.0 / mean).clamp(0.25, 4.0);
        Self {
            exploration: self.exploration * scale,
            optimization: self.optimization * scale,
            synergy: self.synergy * scale,
            reinforcement: self.reinforcement * scale,
        }
    }

    fn delta(self, other: GoldenBiasVector) -> GoldenBiasVector {
        GoldenBiasVector {
            exploration: self.exploration - other.exploration,
            optimization: self.optimization - other.optimization,
            synergy: self.synergy - other.synergy,
            reinforcement: self.reinforcement - other.reinforcement,
        }
    }

    fn magnitude(self) -> f32 {
        (self.exploration.powi(2)
            + self.optimization.powi(2)
            + self.synergy.powi(2)
            + self.reinforcement.powi(2))
        .sqrt()
    }

    fn abs_mean(&self) -> f32 {
        (self.exploration.abs()
            + self.optimization.abs()
            + self.synergy.abs()
            + self.reinforcement.abs())
            / 4.0
    }

    fn into_tuple(self) -> (f32, f32, f32, f32) {
        (
            self.exploration,
            self.optimization,
            self.synergy,
            self.reinforcement,
        )
    }
}

#[derive(Debug, Clone)]
struct GoldenSelfRewriteState {
    config: GoldenSelfRewriteConfig,
    council: GoldenBiasVector,
    history: VecDeque<GoldenBiasVector>,
    last_momentum: f32,
}

impl GoldenSelfRewriteState {
    fn new(config: GoldenSelfRewriteConfig, initial: GoldenBiasVector) -> Self {
        let initial = initial.clamp(0.05, 32.0);
        let mut history = VecDeque::new();
        history.push_back(initial);
        Self {
            config,
            council: initial,
            history,
            last_momentum: 0.0,
        }
    }

    fn negotiate(
        &mut self,
        epoch: u64,
        pulse: &GoldenBlackcatPulse,
        schedule_signal: &GoldenScheduleSignal,
        heuristics: &HeurOpLog,
    ) -> GoldenCouncilResolution {
        let negotiation_rate = self.config.negotiation_rate.max(0.0);
        let schedule_weight = self.config.schedule_weight.max(0.0);
        let base = GoldenBiasVector::from_biases(1.0, 1.0, 1.0, 1.0);
        let pulse_vector = GoldenBiasVector::from_pulse(pulse);
        let schedule_vector = schedule_signal.to_bias_vector();
        let mut target = GoldenBiasVector {
            exploration: base.exploration
                + pulse_vector.exploration * negotiation_rate
                + schedule_vector.exploration * schedule_weight,
            optimization: base.optimization
                + pulse_vector.optimization * negotiation_rate
                + schedule_vector.optimization * schedule_weight,
            synergy: base.synergy
                + pulse_vector.synergy * negotiation_rate
                + schedule_vector.synergy * schedule_weight,
            reinforcement: base.reinforcement
                + pulse_vector.reinforcement * negotiation_rate
                + schedule_vector.reinforcement * schedule_weight,
        };

        let synergy_carry = (target.synergy - 1.0).max(0.0);
        target.exploration += synergy_carry * 0.18;
        target.optimization += synergy_carry * 0.24;
        target.reinforcement += synergy_carry * 0.12;

        let resonance = (pulse_vector.delta(schedule_vector).abs_mean()
            * self.config.schedule_resonance.max(0.0))
        .clamp(0.0, 32.0);
        let synergy_pressure = self.config.synergy_pressure.max(0.0);
        if resonance > 0.0 {
            target.exploration += resonance * 0.35;
            target.optimization += resonance * 0.25;
            target.synergy += resonance * synergy_pressure;
            target.reinforcement += resonance * 0.18;
        }

        let previous = self.council;
        let raw_momentum = target.delta(previous);
        target = target.apply_inertia(previous, self.config.inertia);
        target = target.clamp(0.05, self.config.max_bias.max(1.0));
        target = target.stabilise();
        let momentum = raw_momentum.magnitude();
        self.last_momentum = self.last_momentum * 0.55 + momentum * 0.45;
        self.council = target;
        self.push_history(target);
        let stability = self.history_stability();
        self.tune_inertia(stability);
        let divergence = pulse_vector.delta(schedule_vector).abs_mean();
        let band_energy = (
            schedule_signal.exploration,
            schedule_signal.synergy,
            schedule_signal.reinforcement,
        );
        let geometry = (
            pulse.mean_support,
            pulse.mean_confidence,
            pulse.mean_reward as f32,
        );
        let snapshot = GoldenCouncilSnapshot {
            epoch,
            high_watermark: heuristics.high_watermark(),
            missing_ranges: heuristics.missing_ranges(),
            winners: heuristics.top_winners(3),
            evidence: CouncilEvidence {
                band_energy,
                graph_flow: pulse.synergy_score,
                psi: pulse.mean_psi,
                geometry,
            },
            exploration_bias: target.exploration,
            optimization_bias: target.optimization,
            synergy_bias: target.synergy,
            reinforcement_bias: target.reinforcement,
            resonance,
            stability,
            momentum: self.last_momentum,
            divergence,
            schedule_hint: schedule_vector.into_tuple(),
            pulse_recap: pulse.clone(),
        };
        GoldenCouncilResolution {
            biases: target,
            snapshot,
        }
    }

    fn into_config(self) -> GoldenSelfRewriteConfig {
        self.config
    }

    fn push_history(&mut self, biases: GoldenBiasVector) {
        let limit = self.config.council_memory.max(2);
        self.history.push_back(biases);
        while self.history.len() > limit {
            self.history.pop_front();
        }
    }

    fn history_stability(&self) -> f32 {
        if self.history.len() < 2 {
            return 1.0;
        }
        let len = self.history.len() as f32;
        let mut mean = GoldenBiasVector::from_biases(0.0, 0.0, 0.0, 0.0);
        for bias in &self.history {
            mean.exploration += bias.exploration;
            mean.optimization += bias.optimization;
            mean.synergy += bias.synergy;
            mean.reinforcement += bias.reinforcement;
        }
        mean.exploration /= len;
        mean.optimization /= len;
        mean.synergy /= len;
        mean.reinforcement /= len;
        let mut variance = 0.0f32;
        for bias in &self.history {
            variance += (bias.exploration - mean.exploration).powi(2)
                + (bias.optimization - mean.optimization).powi(2)
                + (bias.synergy - mean.synergy).powi(2)
                + (bias.reinforcement - mean.reinforcement).powi(2);
        }
        variance /= len * 4.0;
        (1.0 / (1.0 + variance.sqrt())).clamp(0.0, 1.0)
    }

    fn tune_inertia(&mut self, stability: f32) {
        if stability < 0.45 {
            self.config.inertia = (self.config.inertia * 0.88).clamp(0.05, 0.92);
            self.config.negotiation_rate = (self.config.negotiation_rate * 1.08).clamp(0.1, 1.5);
        } else if stability > 0.85 {
            self.config.inertia = (self.config.inertia * 1.05).clamp(0.05, 0.98);
            self.config.negotiation_rate = (self.config.negotiation_rate * 0.92).clamp(0.05, 1.5);
        }
        if self.last_momentum > 6.0 {
            self.config.schedule_weight = (self.config.schedule_weight * 0.9).max(0.05);
        } else if self.last_momentum < 1.2 {
            self.config.schedule_weight = (self.config.schedule_weight * 1.05).min(3.0);
        }
    }
}

#[derive(Debug, Clone)]
struct GoldenCouncilResolution {
    biases: GoldenBiasVector,
    snapshot: GoldenCouncilSnapshot,
}

#[derive(Debug, Clone)]
pub struct CouncilEvidence {
    pub band_energy: (f32, f32, f32),
    pub graph_flow: f32,
    pub psi: f32,
    pub geometry: (f32, f32, f32),
}

#[derive(Debug, Clone)]
pub struct GoldenCouncilSnapshot {
    pub epoch: u64,
    pub high_watermark: u64,
    pub missing_ranges: Vec<(u64, u64)>,
    pub winners: Vec<HeurOp>,
    pub evidence: CouncilEvidence,
    pub exploration_bias: f32,
    pub optimization_bias: f32,
    pub synergy_bias: f32,
    pub reinforcement_bias: f32,
    pub resonance: f32,
    pub stability: f32,
    pub momentum: f32,
    pub divergence: f32,
    pub schedule_hint: (f32, f32, f32, f32),
    pub pulse_recap: GoldenBlackcatPulse,
}

#[derive(Debug, Clone)]
pub struct CouncilDigest {
    pub snapshot: GoldenCouncilSnapshot,
}

impl CouncilDigest {
    pub fn epoch(&self) -> u64 {
        self.snapshot.epoch
    }
}

#[derive(Debug, Clone, Copy)]
struct GoldenScheduleSignal {
    exploration: f32,
    optimization: f32,
    synergy: f32,
    reinforcement: f32,
}

impl GoldenScheduleSignal {
    fn neutral() -> Self {
        Self {
            exploration: 0.5,
            optimization: 0.5,
            synergy: 0.5,
            reinforcement: 0.5,
        }
    }

    fn from_slice(schedules: &[RoundtableSchedule]) -> Self {
        if schedules.is_empty() {
            return Self::neutral();
        }
        let mut above = 0.0f32;
        let mut here = 0.0f32;
        let mut beneath = 0.0f32;
        let mut depth = 0.0f32;
        for schedule in schedules {
            let a = schedule.above().k as f32;
            let h = schedule.here().k as f32;
            let b = schedule.beneath().k as f32;
            above += a;
            here += h;
            beneath += b;
            depth += a + h + b;
        }
        let total = depth.max(1.0);
        let above_ratio = (above / total).clamp(0.0, 1.0);
        let here_ratio = (here / total).clamp(0.0, 1.0);
        let beneath_ratio = (beneath / total).clamp(0.0, 1.0);
        let balance =
            1.0 - ((above_ratio - here_ratio).abs() + (here_ratio - beneath_ratio).abs()) * 0.5;
        let depth_mean = depth / schedules.len() as f32;
        Self {
            exploration: (above_ratio * 1.4 + 0.3).clamp(0.2, 1.8),
            optimization: (depth_mean / 48.0 + 0.3).clamp(0.2, 2.4),
            synergy: (balance.clamp(0.0, 1.0) * 1.6 + here_ratio * 0.4).clamp(0.2, 2.2),
            reinforcement: ((beneath_ratio * 1.3) + (1.0 - balance).max(0.0) * 0.4).clamp(0.2, 1.8),
        }
    }

    fn to_bias_vector(&self) -> GoldenBiasVector {
        GoldenBiasVector::from_biases(
            self.exploration,
            self.optimization,
            self.synergy,
            self.reinforcement,
        )
    }
}

/// Distributed trainer that coordinates a fleet of [`ModuleTrainer`] instances
/// without leaking data races. GoldenRetriever delegates blocking work to the
/// SpiralTorch runtime and performs deterministic reductions so we stay in
/// lockstep even when multiple nodes run in parallel.
pub struct GoldenRetriever {
    runtime: GoldenRuntime,
    workers: Vec<GoldenWorker>,
    sync_blackcat_minutes: bool,
    sync_heuristics_log: bool,
    coordinate_blackcat: bool,
    exploration_bias: f32,
    optimization_boost: f32,
    synergy_bias: f32,
    reinforcement_bias: f32,
    self_rewrite: Option<GoldenSelfRewriteState>,
    latest_council: Option<GoldenCouncilSnapshot>,
    epoch: u64,
    digest_subscribers: Arc<Mutex<Vec<mpsc::Sender<CouncilDigest>>>>,
}

impl GoldenRetriever {
    pub fn new(config: GoldenRetrieverConfig, trainers: Vec<ModuleTrainer>) -> PureResult<Self> {
        if trainers.is_empty() {
            return Err(TensorError::EmptyInput("golden retriever trainers"));
        }
        let GoldenRetrieverConfig {
            workers,
            runtime,
            sync_blackcat_minutes,
            sync_heuristics_log,
            coordinate_blackcat,
            exploration_bias,
            optimization_boost,
            synergy_bias,
            reinforcement_bias,
            self_rewrite,
        } = config;
        let worker_count = if workers == 0 {
            trainers.len()
        } else {
            workers
        };
        if worker_count != trainers.len() {
            return Err(TensorError::IoError {
                message: format!(
                    "golden retriever expected {worker_count} trainers but received {}",
                    trainers.len()
                ),
            });
        }
        let runtime_cfg = runtime.unwrap_or_default();
        let runtime = GoldenRuntime::new(runtime_cfg).map_err(runtime_error)?;
        let workers = trainers
            .into_iter()
            .enumerate()
            .map(|(id, trainer)| GoldenWorker {
                _id: id,
                trainer: SpiralMutex::new(trainer),
            })
            .collect();
        let exploration_bias = exploration_bias.max(0.0);
        let optimization_boost = optimization_boost.max(0.0);
        let synergy_bias = synergy_bias.max(0.0);
        let reinforcement_bias = reinforcement_bias.max(0.0);
        let self_rewrite = self_rewrite.map(|cfg| {
            GoldenSelfRewriteState::new(
                cfg,
                GoldenBiasVector::from_biases(
                    exploration_bias,
                    optimization_boost,
                    synergy_bias,
                    reinforcement_bias,
                ),
            )
        });
        Ok(Self {
            runtime,
            workers,
            sync_blackcat_minutes,
            sync_heuristics_log,
            coordinate_blackcat,
            exploration_bias,
            optimization_boost,
            synergy_bias,
            reinforcement_bias,
            self_rewrite,
            latest_council: None,
            epoch: 0,
            digest_subscribers: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub fn workers(&self) -> usize {
        self.workers.len()
    }

    pub fn runtime(&self) -> GoldenRuntime {
        self.runtime.clone()
    }

    pub fn coordination_biases(&self) -> (f32, f32, f32, f32) {
        (
            self.exploration_bias,
            self.optimization_boost,
            self.synergy_bias,
            self.reinforcement_bias,
        )
    }

    pub fn last_council_snapshot(&self) -> Option<&GoldenCouncilSnapshot> {
        self.latest_council.as_ref()
    }

    pub fn last_council(&self) -> Option<GoldenCouncilSnapshot> {
        self.latest_council.clone()
    }

    pub fn subscribe_digest(&self) -> mpsc::Receiver<CouncilDigest> {
        let (sender, receiver) = mpsc::channel();
        if let Ok(mut subscribers) = self.digest_subscribers.lock() {
            subscribers.push(sender);
        }
        receiver
    }

    pub fn absorb_digest(&mut self, digest: CouncilDigest) -> PureResult<()> {
        let should_update = self
            .latest_council
            .as_ref()
            .map(|current| digest.snapshot.epoch > current.epoch)
            .unwrap_or(true);
        if should_update {
            self.latest_council = Some(digest.snapshot.clone());
            for worker in &self.workers {
                let mut guard = worker.trainer.lock();
                guard.record_golden_council(&digest.snapshot);
            }
        }
        Ok(())
    }

    pub fn run_epoch<M, L>(
        &mut self,
        modules: Vec<M>,
        losses: Vec<L>,
        loaders: Vec<DataLoader>,
        schedules: Vec<RoundtableSchedule>,
    ) -> PureResult<GoldenEpochReport>
    where
        M: Module + Send + 'static,
        L: Loss + Send + 'static,
    {
        let expected = self.workers.len();
        if modules.len() != expected
            || losses.len() != expected
            || loaders.len() != expected
            || schedules.len() != expected
        {
            return Err(TensorError::IoError {
                message: format!(
                    "golden retriever requires {expected} modules, losses, loaders, and schedules"
                ),
            });
        }

        self.epoch = self.epoch.wrapping_add(1);
        let current_epoch = self.epoch;
        let schedule_signal = self
            .self_rewrite
            .as_ref()
            .map(|_| GoldenScheduleSignal::from_slice(&schedules));
        let mut handles = Vec::with_capacity(expected);
        let mut module_iter = modules.into_iter();
        let mut loss_iter = losses.into_iter();
        let mut loader_iter = loaders.into_iter();
        let mut schedule_iter = schedules.into_iter();

        for worker in self.workers.iter() {
            let mut module = module_iter.next().expect("module length checked");
            let mut loss = loss_iter.next().expect("loss length checked");
            let loader = loader_iter.next().expect("loader length checked");
            let schedule = schedule_iter.next().expect("schedule length checked");
            let trainer = worker.trainer.clone();
            let handle = self
                .runtime
                .spawn_blocking(move || -> PureResult<EpochStats> {
                    let mut guard = trainer.lock();
                    guard.train_epoch(&mut module, &mut loss, loader, &schedule)
                })
                .map_err(runtime_error)?;
            handles.push(handle);
        }

        let mut stats = Vec::with_capacity(expected);
        for handle in handles {
            match handle.join() {
                Ok(result) => stats.push(result?),
                Err(_) => {
                    return Err(TensorError::IoError {
                        message: "golden retriever worker panicked".into(),
                    });
                }
            }
        }

        let (minutes, heuristics) = self.collect_cooperative_state();
        let pulse = if self.coordinate_blackcat {
            Some(self.compose_blackcat_pulse(&minutes, &heuristics))
        } else {
            None
        };
        if self.sync_blackcat_minutes || self.sync_heuristics_log || self.coordinate_blackcat {
            self.broadcast_cooperative_state(&minutes, &heuristics, pulse.as_ref());
        }
        if self.self_rewrite.is_some() {
            self.maybe_self_rewrite(
                current_epoch,
                schedule_signal.as_ref(),
                pulse.as_ref(),
                &heuristics,
            );
        }

        Ok(GoldenEpochReport::from_stats(
            &self.runtime,
            stats,
            minutes,
            heuristics,
            pulse,
            self.latest_council.clone(),
        ))
    }

    fn collect_cooperative_state(&self) -> (Vec<ModeratorMinutes>, HeurOpLog) {
        let mut minutes = Vec::new();
        let mut heuristics = HeurOpLog::default();
        for worker in &self.workers {
            let guard = worker.trainer.lock();
            minutes.extend(guard.blackcat_minutes());
            heuristics.merge(guard.heuristics_log());
        }
        (dedupe_minutes(minutes), heuristics)
    }

    fn broadcast_cooperative_state(
        &self,
        minutes: &[ModeratorMinutes],
        heuristics: &HeurOpLog,
        pulse: Option<&GoldenBlackcatPulse>,
    ) {
        for worker in &self.workers {
            let mut guard = worker.trainer.lock();
            if self.sync_blackcat_minutes {
                guard.sync_blackcat_minutes(minutes);
            }
            if self.sync_heuristics_log {
                guard.merge_heuristics_log(heuristics);
            }
            #[cfg(feature = "golden")]
            if let Some(pulse) = pulse {
                guard.apply_blackcat_pulse(pulse);
            }
        }
    }

    fn maybe_self_rewrite(
        &mut self,
        epoch: u64,
        schedule_signal: Option<&GoldenScheduleSignal>,
        pulse: Option<&GoldenBlackcatPulse>,
        heuristics: &HeurOpLog,
    ) {
        let Some(state) = self.self_rewrite.as_mut() else {
            return;
        };
        let Some(signal) = schedule_signal else {
            return;
        };
        let idle;
        let pulse = match pulse {
            Some(pulse) => pulse,
            None => {
                idle = GoldenBlackcatPulse::idle();
                &idle
            }
        };
        let GoldenCouncilResolution { biases, snapshot } =
            state.negotiate(epoch, pulse, signal, heuristics);
        self.exploration_bias = biases.exploration;
        self.optimization_boost = biases.optimization;
        self.synergy_bias = biases.synergy;
        self.reinforcement_bias = biases.reinforcement;
        self.latest_council = Some(snapshot.clone());
        self.emit_council_digest(&snapshot);
        for worker in &self.workers {
            let mut guard = worker.trainer.lock();
            guard.record_golden_council(&snapshot);
        }
    }

    fn emit_council_digest(&mut self, snapshot: &GoldenCouncilSnapshot) {
        if let Ok(mut subscribers) = self.digest_subscribers.lock() {
            if subscribers.is_empty() {
                return;
            }
            let digest = CouncilDigest {
                snapshot: snapshot.clone(),
            };
            let mut dropped = Vec::new();
            for (idx, sender) in subscribers.iter().enumerate() {
                if sender.send(digest.clone()).is_err() {
                    dropped.push(idx);
                }
            }
            for idx in dropped.into_iter().rev() {
                subscribers.remove(idx);
            }
        }
    }

    fn compose_blackcat_pulse(
        &self,
        minutes: &[ModeratorMinutes],
        heuristics: &HeurOpLog,
    ) -> GoldenBlackcatPulse {
        if minutes.is_empty() && heuristics.entries().is_empty() {
            return GoldenBlackcatPulse::idle();
        }

        let mut support_sum = 0.0f32;
        let mut psi_sum = 0.0f32;
        let mut reward_sum = 0.0f64;
        let mut confidence_sum = 0.0f32;
        let mut plan_totals: HashMap<&str, (f64, usize, SystemTime)> = HashMap::new();

        for minute in minutes {
            support_sum += minute.support.max(0.0);
            psi_sum += minute.mean_psi;
            reward_sum += minute.reward.max(0.0);
            confidence_sum += minute.confidence.0 + minute.confidence.1;
            let entry = plan_totals
                .entry(minute.plan_signature.as_str())
                .or_insert((0.0, 0, minute.issued_at));
            entry.0 += minute.reward.max(0.0);
            entry.1 += 1;
            if minute.issued_at > entry.2 {
                entry.2 = minute.issued_at;
            }
        }

        let coverage = minutes.len();
        let heuristics_contributions = heuristics.entries().len();
        let mut append_weight = 0.0f32;
        let mut retract_count = 0usize;
        let mut annotate_count = 0usize;
        for entry in heuristics.entries() {
            match &entry.kind {
                HeurOpKind::AppendSoft { weight, .. } => {
                    append_weight += weight.max(0.0);
                }
                HeurOpKind::Retract { .. } => {
                    retract_count += 1;
                }
                HeurOpKind::Annotate { .. } => {
                    annotate_count += 1;
                }
            }
        }
        let coverage_f32 = coverage as f32;
        let mean_support = if coverage > 0 {
            support_sum / coverage_f32
        } else {
            0.0
        };
        let mean_reward = if coverage > 0 {
            reward_sum / coverage as f64
        } else {
            0.0
        };
        let mean_psi = if coverage > 0 {
            psi_sum / coverage_f32
        } else {
            0.0
        };
        let mean_confidence = if coverage > 0 {
            confidence_sum / coverage_f32
        } else {
            0.0
        };
        let heuristic_factor = if coverage > 0 {
            heuristics_contributions as f32 / coverage_f32
        } else {
            heuristics_contributions as f32
        };
        let append_factor = if heuristics_contributions > 0 {
            append_weight / heuristics_contributions as f32
        } else {
            append_weight
        };

        let exploration_drive =
            ((mean_support + mean_psi.abs() * 0.25) * self.exploration_bias).max(0.0);
        let optimization_gain =
            ((mean_reward as f32 * 0.5) + heuristic_factor).max(0.0) * self.optimization_boost;
        let synergy_score = ((mean_support * 0.4 + mean_confidence * 0.35 + mean_psi.abs() * 0.25)
            * self.synergy_bias)
            .max(0.0);
        let reinforcement_weight =
            ((append_factor + heuristic_factor * 0.5 + (coverage_f32 * 0.05))
                * self.reinforcement_bias)
                .max(0.0);

        let dominant_plan = plan_totals
            .into_iter()
            .max_by(|(plan_a, (reward_a, count_a, seen_a)), (plan_b, (reward_b, count_b, seen_b))| {
                let avg_a = if *count_a == 0 {
                    0.0
                } else {
                    *reward_a / *count_a as f64
                };
                let avg_b = if *count_b == 0 {
                    0.0
                } else {
                    *reward_b / *count_b as f64
                };
                avg_a
                    .partial_cmp(&avg_b)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| seen_a.cmp(seen_b))
                    .then_with(|| plan_a.cmp(plan_b))
            })
            .map(|(plan, _)| plan.to_string());

        GoldenBlackcatPulse {
            exploration_drive,
            optimization_gain,
            synergy_score,
            reinforcement_weight,
            mean_support,
            mean_reward,
            mean_psi,
            mean_confidence,
            coverage,
            heuristics_contributions,
            append_weight,
            retract_count,
            annotate_count,
            dominant_plan,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GoldenEpochReport {
    pub workers: usize,
    pub batches: usize,
    pub total_loss: f32,
    pub average_loss: f32,
    pub per_worker: Vec<EpochStats>,
    pub moderator_minutes: Vec<ModeratorMinutes>,
    pub heuristics_log: HeurOpLog,
    pub cooperative_pulse: Option<GoldenBlackcatPulse>,
    pub council_snapshot: Option<GoldenCouncilSnapshot>,
}

impl GoldenEpochReport {
    fn from_stats(
        runtime: &GoldenRuntime,
        per_worker: Vec<EpochStats>,
        moderator_minutes: Vec<ModeratorMinutes>,
        heuristics_log: HeurOpLog,
        cooperative_pulse: Option<GoldenBlackcatPulse>,
        council_snapshot: Option<GoldenCouncilSnapshot>,
    ) -> Self {
        let workers = per_worker.len();
        let (total_loss, total_batches) = runtime.reduce(
            &per_worker,
            |stats| (stats.total_loss, stats.batches),
            |left, right| (left.0 + right.0, left.1 + right.1),
            (0.0f32, 0usize),
        );
        let average_loss = if total_batches > 0 {
            total_loss / total_batches as f32
        } else {
            0.0
        };
        Self {
            workers,
            batches: total_batches,
            total_loss,
            average_loss,
            per_worker,
            moderator_minutes,
            heuristics_log,
            cooperative_pulse,
            council_snapshot,
        }
    }

    pub fn cooperative_directive(
        &self,
        baseline_interval: Duration,
        baseline_window: usize,
    ) -> Option<GoldenCooperativeDirective> {
        self.cooperative_pulse
            .as_ref()
            .map(|pulse| pulse.directive(baseline_interval, baseline_window))
    }

    pub fn council_snapshot(&self) -> Option<&GoldenCouncilSnapshot> {
        self.council_snapshot.as_ref()
    }
}

fn runtime_error(err: GoldenRuntimeError) -> TensorError {
    TensorError::IoError { message: err.0 }
}

fn dedupe_minutes(minutes: Vec<ModeratorMinutes>) -> Vec<ModeratorMinutes> {
    let mut deduped = Vec::new();
    for minute in minutes {
        if deduped.iter().any(|existing: &ModeratorMinutes| {
            existing.plan_signature == minute.plan_signature
                && existing.issued_at == minute.issued_at
        }) {
            continue;
        }
        deduped.push(minute);
    }
    deduped
}

#[derive(Debug, Clone)]
pub struct GoldenBlackcatPulse {
    pub exploration_drive: f32,
    pub optimization_gain: f32,
    pub synergy_score: f32,
    pub reinforcement_weight: f32,
    pub mean_support: f32,
    pub mean_reward: f64,
    pub mean_psi: f32,
    pub mean_confidence: f32,
    pub coverage: usize,
    pub heuristics_contributions: usize,
    pub append_weight: f32,
    pub retract_count: usize,
    pub annotate_count: usize,
    pub dominant_plan: Option<String>,
}

#[derive(Debug, Clone)]
pub struct GoldenCooperativeDirective {
    pub push_interval: Duration,
    pub summary_window: usize,
    pub exploration_priority: f32,
    pub reinforcement_weight: f32,
}

impl GoldenBlackcatPulse {
    pub fn idle() -> Self {
        Self {
            exploration_drive: 0.0,
            optimization_gain: 0.0,
            synergy_score: 0.0,
            reinforcement_weight: 0.0,
            mean_support: 0.0,
            mean_reward: 0.0,
            mean_psi: 0.0,
            mean_confidence: 0.0,
            coverage: 0,
            heuristics_contributions: 0,
            append_weight: 0.0,
            retract_count: 0,
            annotate_count: 0,
            dominant_plan: None,
        }
    }

    pub fn is_idle(&self) -> bool {
        self.coverage == 0 && self.heuristics_contributions == 0
    }

    pub fn directive(
        &self,
        baseline_interval: Duration,
        baseline_window: usize,
    ) -> GoldenCooperativeDirective {
        let baseline_secs = baseline_interval.as_secs_f32().max(0.5);
        let optimization_term =
            (1.0 + self.optimization_gain + self.reinforcement_weight * 0.5).clamp(1.0, 6.0);
        let new_interval = Duration::from_secs_f32(
            (baseline_secs / optimization_term).clamp(baseline_secs * 0.1, baseline_secs * 1.2),
        );
        let exploration_term =
            (1.0 + self.exploration_drive + self.synergy_score * 0.3).clamp(0.5, 8.0);
        let summary_window = (baseline_window as f32 * exploration_term)
            .round()
            .clamp(2.0, 4096.0) as usize;
        GoldenCooperativeDirective {
            push_interval: new_interval,
            summary_window,
            exploration_priority: (self.exploration_drive + self.synergy_score).clamp(0.0, 16.0),
            reinforcement_weight: (self.reinforcement_weight + self.optimization_gain)
                .clamp(0.0, 16.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Dataset;
    use crate::layers::linear::Linear;
    use crate::loss::MeanSquaredError;
    use crate::schedule::RoundtableConfig;
    use st_core::backend::device_caps::DeviceCaps;
    use std::time::Duration;

    #[test]
    fn golden_retriever_trains_in_parallel() {
        let caps = DeviceCaps::wgpu(16, true, 128);
        let trainers = vec![
            ModuleTrainer::new(caps, -1.0, 0.05, 0.01),
            ModuleTrainer::new(caps, -1.0, 0.05, 0.01),
        ];
        let mut modules = Vec::new();
        let mut schedules = Vec::new();
        let mut losses = Vec::new();
        let mut loaders = Vec::new();
        for trainer in trainers.iter() {
            let mut layer = Linear::new("lin", 2, 1).unwrap();
            trainer.prepare(&mut layer).unwrap();
            let schedule = trainer.roundtable(1, 1, RoundtableConfig::default());
            schedules.push(schedule);
            modules.push(layer);
            losses.push(MeanSquaredError::default());
            let dataset = Dataset::from_vec(vec![
                (
                    crate::Tensor::from_vec(1, 2, vec![0.0, 1.0]).unwrap(),
                    crate::Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
                ),
                (
                    crate::Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap(),
                    crate::Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
                ),
            ]);
            loaders.push(dataset.loader().batched(1));
        }

        let mut retriever =
            GoldenRetriever::new(GoldenRetrieverConfig::default(), trainers).unwrap();
        let report = retriever
            .run_epoch(modules, losses, loaders, schedules)
            .unwrap();
        assert_eq!(report.workers, 2);
        assert!(report.total_loss.is_finite());
        assert!(report.average_loss.is_finite());
        assert_eq!(report.per_worker.len(), 2);
        assert!(report.batches >= 2);
        assert!(report.moderator_minutes.is_empty());
        assert!(report.heuristics_log.entries().is_empty());
        assert!(report.cooperative_pulse.is_none());
    }

    #[test]
    fn golden_retriever_reports_cooperative_pulse() {
        let caps = DeviceCaps::wgpu(16, true, 128);
        let trainers = vec![
            ModuleTrainer::new(caps, -1.0, 0.05, 0.01),
            ModuleTrainer::new(caps, -1.0, 0.05, 0.01),
        ];
        let mut modules = Vec::new();
        let mut schedules = Vec::new();
        let mut losses = Vec::new();
        let mut loaders = Vec::new();
        for trainer in trainers.iter() {
            let mut layer = Linear::new("lin", 2, 1).unwrap();
            trainer.prepare(&mut layer).unwrap();
            let schedule = trainer.roundtable(1, 1, RoundtableConfig::default());
            schedules.push(schedule);
            modules.push(layer);
            losses.push(MeanSquaredError::default());
            let dataset = Dataset::from_vec(vec![
                (
                    crate::Tensor::from_vec(1, 2, vec![0.0, 1.0]).unwrap(),
                    crate::Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
                ),
                (
                    crate::Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap(),
                    crate::Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
                ),
            ]);
            loaders.push(dataset.loader().batched(1));
        }

        let config = GoldenRetrieverConfig {
            coordinate_blackcat: true,
            exploration_bias: 1.0,
            optimization_boost: 0.5,
            ..GoldenRetrieverConfig::default()
        };
        let mut retriever = GoldenRetriever::new(config, trainers).unwrap();
        let report = retriever
            .run_epoch(modules, losses, loaders, schedules)
            .unwrap();
        let pulse = report.cooperative_pulse.expect("cooperative pulse");
        assert!(pulse.exploration_drive >= 0.0);
        assert!(pulse.optimization_gain >= 0.0);
        assert!(pulse.synergy_score >= 0.0);
        assert!(pulse.reinforcement_weight >= 0.0);
        assert_eq!(pulse.mean_confidence, 0.0);
        assert_eq!(pulse.coverage, 0);
        assert_eq!(pulse.append_weight, 0.0);
        assert_eq!(pulse.retract_count, 0);
        assert_eq!(pulse.annotate_count, 0);
        assert!(pulse.is_idle());
        let directive = pulse.directive(Duration::from_secs_f32(2.0), 32);
        assert!(directive.summary_window >= 2);
        assert!(directive.push_interval.as_secs_f32() > 0.0);
        assert!(directive.exploration_priority >= 0.0);
        assert!(directive.reinforcement_weight >= 0.0);
    }

    #[test]
    fn golden_retriever_reports_council_snapshot() {
        let caps = DeviceCaps::wgpu(32, true, 128);
        let trainers = vec![
            ModuleTrainer::new(caps, -1.0, 0.05, 0.01),
            ModuleTrainer::new(caps, -1.0, 0.05, 0.01),
        ];
        let mut modules = Vec::new();
        let mut schedules = Vec::new();
        let mut losses = Vec::new();
        let mut loaders = Vec::new();
        for trainer in trainers.iter() {
            let mut layer = Linear::new("lin", 2, 1).unwrap();
            trainer.prepare(&mut layer).unwrap();
            let schedule = trainer.roundtable(1, 1, RoundtableConfig::default());
            schedules.push(schedule);
            modules.push(layer);
            losses.push(MeanSquaredError::default());
            let dataset = Dataset::from_vec(vec![
                (
                    crate::Tensor::from_vec(1, 2, vec![0.0, 1.0]).unwrap(),
                    crate::Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
                ),
                (
                    crate::Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap(),
                    crate::Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
                ),
            ]);
            loaders.push(dataset.loader().batched(1));
        }

        let mut config = GoldenRetrieverConfig::default();
        config.coordinate_blackcat = true;
        config.self_rewrite = Some(
            GoldenSelfRewriteConfig::default()
                .with_council_memory(4)
                .with_schedule_resonance(0.9),
        );
        let mut retriever = GoldenRetriever::new(config, trainers).unwrap();
        let report = retriever
            .run_epoch(modules, losses, loaders, schedules)
            .unwrap();
        let snapshot = report
            .council_snapshot()
            .expect("council snapshot should be emitted");
        assert!(snapshot.exploration_bias.is_finite());
        assert!(snapshot.momentum >= 0.0);
        assert!(snapshot.stability >= 0.0 && snapshot.stability <= 1.0);
        assert!(snapshot.schedule_hint.0 > 0.0);
        assert!(snapshot.epoch >= 1);
        assert!(snapshot.high_watermark >= snapshot.winners.len() as u64);
        assert!(snapshot.evidence.graph_flow >= 0.0);
        assert!(snapshot.evidence.geometry.0 >= 0.0);
        if let Some(last) = retriever.last_council_snapshot() {
            assert_eq!(last.schedule_hint.0, snapshot.schedule_hint.0);
        }
        assert!(retriever.last_council().is_some());
    }

    #[test]
    fn golden_retriever_broadcasts_council_digest() {
        let caps = DeviceCaps::wgpu(32, true, 128);
        let trainers = vec![
            ModuleTrainer::new(caps, -1.0, 0.05, 0.01),
            ModuleTrainer::new(caps, -1.0, 0.05, 0.01),
        ];
        let mut modules = Vec::new();
        let mut schedules = Vec::new();
        let mut losses = Vec::new();
        let mut loaders = Vec::new();
        for trainer in trainers.iter() {
            let mut layer = Linear::new("lin", 2, 1).unwrap();
            trainer.prepare(&mut layer).unwrap();
            let schedule = trainer.roundtable(1, 1, RoundtableConfig::default());
            schedules.push(schedule);
            modules.push(layer);
            losses.push(MeanSquaredError::default());
            let dataset = Dataset::from_vec(vec![
                (
                    crate::Tensor::from_vec(1, 2, vec![0.0, 1.0]).unwrap(),
                    crate::Tensor::from_vec(1, 1, vec![1.0]).unwrap(),
                ),
                (
                    crate::Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap(),
                    crate::Tensor::from_vec(1, 1, vec![0.0]).unwrap(),
                ),
            ]);
            loaders.push(dataset.loader().batched(1));
        }

        let mut config = GoldenRetrieverConfig::default();
        config.coordinate_blackcat = true;
        config.self_rewrite = Some(GoldenSelfRewriteConfig::default());
        let mut retriever = GoldenRetriever::new(config, trainers).unwrap();
        let receiver = retriever.subscribe_digest();
        let _ = retriever
            .run_epoch(modules, losses, loaders, schedules)
            .unwrap();
        let digest = receiver
            .recv_timeout(Duration::from_millis(250))
            .expect("council digest delivered");
        assert!(digest.epoch() >= 1);
        assert!(digest.snapshot.high_watermark >= digest.snapshot.winners.len() as u64);
    }

    #[test]
    fn golden_retriever_self_rewrites_biases() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let trainers = vec![
            ModuleTrainer::new(caps, -1.0, 0.05, 0.01),
            ModuleTrainer::new(caps, -1.0, 0.05, 0.01),
        ];
        let mut modules = Vec::new();
        let mut schedules = Vec::new();
        let mut losses = Vec::new();
        let mut loaders = Vec::new();
        for trainer in trainers.iter() {
            let mut layer = Linear::new("lin", 4, 2).unwrap();
            trainer.prepare(&mut layer).unwrap();
            let schedule = trainer.roundtable(
                1,
                64,
                RoundtableConfig::default()
                    .with_top_k(24)
                    .with_mid_k(8)
                    .with_bottom_k(4),
            );
            schedules.push(schedule);
            modules.push(layer);
            losses.push(MeanSquaredError::default());
            let dataset = Dataset::from_vec(vec![
                (
                    crate::Tensor::from_vec(1, 4, vec![0.0, 1.0, 0.5, -0.5]).unwrap(),
                    crate::Tensor::from_vec(1, 2, vec![1.0, 0.0]).unwrap(),
                ),
                (
                    crate::Tensor::from_vec(1, 4, vec![1.0, 0.0, -0.5, 0.75]).unwrap(),
                    crate::Tensor::from_vec(1, 2, vec![0.0, 1.0]).unwrap(),
                ),
            ]);
            loaders.push(dataset.loader().batched(1));
        }

        let mut config = GoldenRetrieverConfig::default();
        config.coordinate_blackcat = true;
        config.self_rewrite = Some(
            GoldenSelfRewriteConfig::default()
                .with_schedule_weight(0.9)
                .with_negotiation_rate(0.4)
                .with_inertia(0.4),
        );
        let mut retriever = GoldenRetriever::new(config, trainers).unwrap();
        let initial = retriever.coordination_biases();
        let report = retriever
            .run_epoch(modules, losses, loaders, schedules)
            .unwrap();
        let updated = retriever.coordination_biases();
        assert!(report.batches >= 2);
        assert!(
            (updated.0 - initial.0).abs() > 1e-4
                || (updated.1 - initial.1).abs() > 1e-4
                || (updated.2 - initial.2).abs() > 1e-4
                || (updated.3 - initial.3).abs() > 1e-4
        );
    }

    #[test]
    fn golden_config_rewrites_via_scheduler() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        let trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
        let schedule = trainer.roundtable(
            1,
            64,
            RoundtableConfig::default()
                .with_top_k(32)
                .with_mid_k(12)
                .with_bottom_k(6),
        );
        let mut config = GoldenRetrieverConfig::default()
            .with_self_rewrite(GoldenSelfRewriteConfig::default().with_schedule_weight(0.75));
        let mut pulse = GoldenBlackcatPulse::idle();
        pulse.exploration_drive = 1.2;
        pulse.optimization_gain = 0.6;
        pulse.synergy_score = 0.8;
        pulse.reinforcement_weight = 0.5;
        config.rewrite_with_scheduler(&[schedule], Some(&pulse));
        assert!(config.exploration_bias > 0.0);
        assert!(
            (config.exploration_bias - 1.0).abs() > 1e-4
                || (config.optimization_boost - 1.0).abs() > 1e-4
                || (config.synergy_bias - 1.0).abs() > 1e-4
                || (config.reinforcement_bias - 1.0).abs() > 1e-4
        );
    }
}
