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
use crate::roundtable::{HeurOpKind, HeurOpLog, ModeratorMinutes};
use crate::schedule::RoundtableSchedule;
use crate::trainer::{EpochStats, ModuleTrainer};
use crate::PureResult;
use st_core::runtime::golden::{
    GoldenRuntime, GoldenRuntimeConfig, GoldenRuntimeError, SpiralMutex,
};
use st_tensor::pure::TensorError;
use std::cmp::Ordering;
use std::time::Duration;

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
        }
    }
}

struct GoldenWorker {
    _id: usize,
    trainer: SpiralMutex<ModuleTrainer>,
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
}

impl GoldenRetriever {
    pub fn new(config: GoldenRetrieverConfig, trainers: Vec<ModuleTrainer>) -> PureResult<Self> {
        if trainers.is_empty() {
            return Err(TensorError::EmptyInput("golden retriever trainers"));
        }
        let worker_count = if config.workers == 0 {
            trainers.len()
        } else {
            config.workers
        };
        if worker_count != trainers.len() {
            return Err(TensorError::IoError {
                message: format!(
                    "golden retriever expected {worker_count} trainers but received {}",
                    trainers.len()
                ),
            });
        }
        let runtime_cfg = config.runtime.unwrap_or_default();
        let runtime = GoldenRuntime::new(runtime_cfg).map_err(runtime_error)?;
        let workers = trainers
            .into_iter()
            .enumerate()
            .map(|(id, trainer)| GoldenWorker {
                _id: id,
                trainer: SpiralMutex::new(trainer),
            })
            .collect();
        Ok(Self {
            runtime,
            workers,
            sync_blackcat_minutes: config.sync_blackcat_minutes,
            sync_heuristics_log: config.sync_heuristics_log,
            coordinate_blackcat: config.coordinate_blackcat,
            exploration_bias: config.exploration_bias.max(0.0),
            optimization_boost: config.optimization_boost.max(0.0),
            synergy_bias: config.synergy_bias.max(0.0),
            reinforcement_bias: config.reinforcement_bias.max(0.0),
        })
    }

    pub fn workers(&self) -> usize {
        self.workers.len()
    }

    pub fn runtime(&self) -> GoldenRuntime {
        self.runtime.clone()
    }

    pub fn run_epoch<M, L>(
        &self,
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

        Ok(GoldenEpochReport::from_stats(
            &self.runtime,
            stats,
            minutes,
            heuristics,
            pulse,
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
        let mut dominant: Option<&ModeratorMinutes> = None;

        for minute in minutes {
            support_sum += minute.support.max(0.0);
            psi_sum += minute.mean_psi;
            reward_sum += minute.reward.max(0.0);
            confidence_sum += minute.confidence.0 + minute.confidence.1;
            dominant = match dominant {
                Some(current) => match minute.reward.partial_cmp(&current.reward) {
                    Some(Ordering::Greater) => Some(minute),
                    Some(Ordering::Equal) => {
                        let current_conf = current.confidence.0 + current.confidence.1;
                        let candidate_conf = minute.confidence.0 + minute.confidence.1;
                        if candidate_conf > current_conf {
                            Some(minute)
                        } else {
                            Some(current)
                        }
                    }
                    _ => Some(current),
                },
                None => Some(minute),
            };
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
            dominant_plan: dominant.map(|m| m.plan_signature.clone()),
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
}

impl GoldenEpochReport {
    fn from_stats(
        runtime: &GoldenRuntime,
        per_worker: Vec<EpochStats>,
        moderator_minutes: Vec<ModeratorMinutes>,
        heuristics_log: HeurOpLog,
        cooperative_pulse: Option<GoldenBlackcatPulse>,
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

        let retriever = GoldenRetriever::new(GoldenRetrieverConfig::default(), trainers).unwrap();
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
        let retriever = GoldenRetriever::new(config, trainers).unwrap();
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
}
