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
use crate::schedule::RoundtableSchedule;
use crate::trainer::{EpochStats, ModuleTrainer};
use crate::PureResult;
use st_core::runtime::golden::{
    GoldenRuntime, GoldenRuntimeConfig, GoldenRuntimeError, SpiralMutex,
};
use st_tensor::pure::TensorError;

#[derive(Debug, Clone)]
pub struct GoldenRetrieverConfig {
    pub workers: usize,
    pub runtime: Option<GoldenRuntimeConfig>,
}

impl Default for GoldenRetrieverConfig {
    fn default() -> Self {
        Self {
            workers: 0,
            runtime: None,
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
        Ok(Self { runtime, workers })
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

        Ok(GoldenEpochReport::from_stats(&self.runtime, stats))
    }
}

#[derive(Debug, Clone)]
pub struct GoldenEpochReport {
    pub workers: usize,
    pub batches: usize,
    pub total_loss: f32,
    pub average_loss: f32,
    pub per_worker: Vec<EpochStats>,
}

impl GoldenEpochReport {
    fn from_stats(runtime: &GoldenRuntime, per_worker: Vec<EpochStats>) -> Self {
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
        }
    }
}

fn runtime_error(err: GoldenRuntimeError) -> TensorError {
    TensorError::IoError { message: err.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Dataset;
    use crate::layers::linear::Linear;
    use crate::loss::MeanSquaredError;
    use crate::schedule::RoundtableConfig;
    use st_core::backend::device_caps::DeviceCaps;

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
    }
}
