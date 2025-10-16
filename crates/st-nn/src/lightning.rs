// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::highlevel::SpiralSession;
use crate::schedule::{RoundtableConfig, RoundtableSchedule};
use crate::trainer::{EpochStats, ModuleTrainer};
use crate::{Loss, Module, PureResult, Tensor};
use std::collections::HashSet;

/// Configuration bundle used by [`SpiralLightning`].
#[derive(Debug, Clone)]
pub struct LightningConfig {
    rows: u32,
    cols: u32,
    roundtable: RoundtableConfig,
    auto_prepare: bool,
}

impl LightningConfig {
    /// Creates a new configuration using the provided output shape.
    pub fn new(rows: u32, cols: u32) -> Self {
        Self {
            rows: rows.max(1),
            cols: cols.max(1),
            roundtable: RoundtableConfig::default(),
            auto_prepare: true,
        }
    }

    /// Overrides the roundtable configuration.
    pub fn with_roundtable(mut self, roundtable: RoundtableConfig) -> Self {
        self.roundtable = roundtable;
        self
    }

    /// Enables or disables automatic hypergrad preparation.
    pub fn with_auto_prepare(mut self, auto_prepare: bool) -> Self {
        self.auto_prepare = auto_prepare;
        self
    }

    /// Returns the configured output rows.
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// Returns the configured output columns.
    pub fn cols(&self) -> u32 {
        self.cols
    }

    /// Returns the stored roundtable configuration.
    pub fn roundtable(&self) -> RoundtableConfig {
        self.roundtable
    }

    /// Returns whether automatic module preparation is enabled.
    pub fn auto_prepare(&self) -> bool {
        self.auto_prepare
    }
}

impl Default for LightningConfig {
    fn default() -> Self {
        Self::new(1, 1)
    }
}

/// Minimal training harness inspired by PyTorch Lightning.
///
/// `SpiralLightning` keeps the `SpiralSession`, `ModuleTrainer`, and
/// `RoundtableSchedule` bundled together so higher-level callers can focus on
/// data orchestration while the struct takes care of preparing modules and
/// launching epochs.
#[derive(Debug)]
pub struct SpiralLightning {
    session: SpiralSession,
    trainer: ModuleTrainer,
    schedule: RoundtableSchedule,
    config: LightningConfig,
    prepared_modules: HashSet<usize>,
}

impl SpiralLightning {
    /// Builds a new lightning harness using the default roundtable settings.
    pub fn new(session: SpiralSession, rows: u32, cols: u32) -> Self {
        let config = LightningConfig::new(rows, cols);
        Self::with_config(session, config)
    }

    /// Builds a new harness for the provided configuration.
    pub fn with_config(session: SpiralSession, config: LightningConfig) -> Self {
        let trainer = session.trainer();
        let schedule = trainer.roundtable(config.rows, config.cols, config.roundtable);
        Self {
            session,
            trainer,
            schedule,
            config,
            prepared_modules: HashSet::new(),
        }
    }

    /// Returns an immutable reference to the underlying session.
    pub fn session(&self) -> &SpiralSession {
        &self.session
    }

    /// Returns an immutable reference to the underlying trainer.
    pub fn trainer(&self) -> &ModuleTrainer {
        &self.trainer
    }

    /// Returns a mutable reference to the underlying trainer.
    pub fn trainer_mut(&mut self) -> &mut ModuleTrainer {
        &mut self.trainer
    }

    /// Returns the currently active roundtable schedule.
    pub fn schedule(&self) -> &RoundtableSchedule {
        &self.schedule
    }

    /// Returns the stored configuration.
    pub fn config(&self) -> &LightningConfig {
        &self.config
    }

    /// Rebuilds the internal schedule using a new configuration.
    pub fn reconfigure(&mut self, mut config: LightningConfig) {
        // Ensure we reuse the latest trainer state when refreshing the schedule.
        config.rows = config.rows.max(1);
        config.cols = config.cols.max(1);
        self.schedule = self
            .trainer
            .roundtable(config.rows, config.cols, config.roundtable);
        self.config = config;
        self.prepared_modules.clear();
    }

    /// Explicitly prepares the provided module using the session guard.
    pub fn prepare_module<M: Module>(&mut self, module: &mut M) -> PureResult<()> {
        self.session.prepare_module(module)?;
        let key = module as *mut M as usize;
        self.prepared_modules.insert(key);
        Ok(())
    }

    fn ensure_prepared<M: Module>(&mut self, module: &mut M) -> PureResult<()> {
        if !self.config.auto_prepare {
            return Ok(());
        }
        let key = module as *mut M as usize;
        if self.prepared_modules.contains(&key) {
            return Ok(());
        }
        self.prepare_module(module)
    }

    /// Runs a full epoch using the stored trainer and schedule.
    pub fn train_epoch<M, L, I>(
        &mut self,
        module: &mut M,
        loss: &mut L,
        batches: I,
    ) -> PureResult<EpochStats>
    where
        M: Module,
        L: Loss,
        I: IntoIterator<Item = (Tensor, Tensor)>,
    {
        self.ensure_prepared(module)?;
        self.trainer
            .train_epoch(module, loss, batches, &self.schedule)
    }

    /// Convenience helper that iterates over multiple epochs.
    pub fn fit_epochs<M, L, It, I>(
        &mut self,
        module: &mut M,
        loss: &mut L,
        epochs: It,
    ) -> PureResult<Vec<EpochStats>>
    where
        M: Module,
        L: Loss,
        It: IntoIterator<Item = I>,
        I: IntoIterator<Item = (Tensor, Tensor)>,
    {
        let mut results = Vec::new();
        for batches in epochs {
            let stats = self.train_epoch(module, loss, batches)?;
            results.push(stats);
        }
        Ok(results)
    }

    /// Resets the prepared module registry so the next call reattaches tapes.
    pub fn reset_prepared_modules(&mut self) {
        self.prepared_modules.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::linear::Linear;
    use crate::loss::MeanSquaredError;
    use st_core::backend::device_caps::DeviceCaps;

    fn tensor(data: &[f32]) -> Tensor {
        Tensor::from_vec(1, data.len(), data.to_vec()).unwrap()
    }

    #[test]
    fn lightning_runs_single_epoch() {
        let session = SpiralSession::builder(DeviceCaps::cpu())
            .with_curvature(-1.0)
            .with_hyper_learning_rate(0.05)
            .build()
            .unwrap();
        let mut lightning = SpiralLightning::new(session, 1, 1);
        let mut model = Linear::new("demo", 1, 1).unwrap();
        let mut loss = MeanSquaredError::new();
        let data = vec![(tensor(&[1.0]), tensor(&[0.5])); 4];
        let stats = lightning
            .train_epoch(&mut model, &mut loss, data.clone())
            .unwrap();
        assert_eq!(stats.batches, 4);
        assert!(model.weight().hypergrad().is_some());
        assert!(model.bias().hypergrad().is_some());

        // Ensure repeated epochs reuse the prepared module without panicking.
        let stats2 = lightning.train_epoch(&mut model, &mut loss, data).unwrap();
        assert_eq!(stats2.batches, 4);
    }

    #[test]
    fn lightning_fit_epochs_runs_multiple_passes() {
        let session = SpiralSession::builder(DeviceCaps::cpu())
            .with_curvature(-1.0)
            .build()
            .unwrap();
        let config = LightningConfig::new(1, 1).with_auto_prepare(true);
        let mut lightning = SpiralLightning::with_config(session, config);
        let mut model = Linear::new("demo", 1, 1).unwrap();
        let mut loss = MeanSquaredError::new();
        let epoch_a = vec![(tensor(&[0.0]), tensor(&[0.0])); 2];
        let epoch_b = vec![(tensor(&[1.0]), tensor(&[1.0])); 2];
        let reports = lightning
            .fit_epochs(&mut model, &mut loss, vec![epoch_a, epoch_b])
            .unwrap();
        assert_eq!(reports.len(), 2);
    }
}
