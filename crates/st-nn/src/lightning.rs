// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::highlevel::SpiralSession;
use crate::schedule::{RoundtableConfig, RoundtableSchedule};
use crate::trainer::{EpochStats, ModuleTrainer};
use crate::{Loss, Module, PureResult, Tensor};
use std::collections::HashSet;

/// Stable key used to track which modules were prepared for hypergrad tapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ModuleKey(usize);

impl ModuleKey {
    fn new<M: Module>(module: &mut M) -> Self {
        Self(module as *mut M as usize)
    }
}

fn roundtable_changed(left: &RoundtableConfig, right: &RoundtableConfig) -> bool {
    if left.top_k != right.top_k
        || left.mid_k != right.mid_k
        || left.bottom_k != right.bottom_k
        || left.here_tolerance != right.here_tolerance
    {
        return true;
    }
    #[cfg(feature = "psychoid")]
    {
        if left.psychoid_enabled != right.psychoid_enabled
            || left.psychoid_log != right.psychoid_log
        {
            return true;
        }
    }
    #[cfg(feature = "psi")]
    {
        if left.psi_enabled != right.psi_enabled {
            return true;
        }
    }
    #[cfg(feature = "collapse")]
    {
        if left.collapse_enabled != right.collapse_enabled {
            return true;
        }
    }
    false
}

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
        Self::builder(rows, cols).build()
    }

    /// Creates a builder that can be used to refine the configuration.
    pub fn builder(rows: u32, cols: u32) -> LightningConfigBuilder {
        LightningConfigBuilder::new(rows, cols)
    }

    /// Overrides the roundtable configuration.
    pub fn with_roundtable(mut self, roundtable: RoundtableConfig) -> Self {
        self.roundtable = roundtable;
        self
    }

    /// Updates the output shape stored by the configuration.
    pub fn with_output_shape(mut self, rows: u32, cols: u32) -> Self {
        self.rows = rows.max(1);
        self.cols = cols.max(1);
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

    /// Returns an immutable reference to the roundtable configuration.
    pub fn roundtable_config(&self) -> &RoundtableConfig {
        &self.roundtable
    }

    /// Returns whether automatic module preparation is enabled.
    pub fn auto_prepare(&self) -> bool {
        self.auto_prepare
    }

    /// Returns a builder seeded with the current configuration values.
    pub fn to_builder(&self) -> LightningConfigBuilder {
        LightningConfigBuilder {
            rows: self.rows,
            cols: self.cols,
            roundtable: self.roundtable,
            auto_prepare: self.auto_prepare,
        }
    }
}

impl Default for LightningConfig {
    fn default() -> Self {
        Self::new(1, 1)
    }
}

/// Builder used to progressively construct [`LightningConfig`] instances.
#[derive(Debug, Clone)]
pub struct LightningConfigBuilder {
    rows: u32,
    cols: u32,
    roundtable: RoundtableConfig,
    auto_prepare: bool,
}

impl LightningConfigBuilder {
    fn new(rows: u32, cols: u32) -> Self {
        Self {
            rows: rows.max(1),
            cols: cols.max(1),
            roundtable: RoundtableConfig::default(),
            auto_prepare: true,
        }
    }

    /// Overrides the output shape tracked by the builder.
    pub fn output_shape(mut self, rows: u32, cols: u32) -> Self {
        self.rows = rows.max(1);
        self.cols = cols.max(1);
        self
    }

    /// Overrides the roundtable configuration used when building the final config.
    pub fn roundtable(mut self, roundtable: RoundtableConfig) -> Self {
        self.roundtable = roundtable;
        self
    }

    /// Enables or disables automatic module preparation.
    pub fn auto_prepare(mut self, auto_prepare: bool) -> Self {
        self.auto_prepare = auto_prepare;
        self
    }

    /// Applies an arbitrary transformation before finishing the builder.
    pub fn configure_with(self, f: impl FnOnce(Self) -> Self) -> Self {
        f(self)
    }

    /// Finalises the builder into a [`LightningConfig`].
    pub fn build(self) -> LightningConfig {
        LightningConfig {
            rows: self.rows,
            cols: self.cols,
            roundtable: self.roundtable,
            auto_prepare: self.auto_prepare,
        }
    }
}

impl Default for LightningConfigBuilder {
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
    prepared_modules: HashSet<ModuleKey>,
}

impl SpiralLightning {
    /// Builds a new lightning harness using the default roundtable settings.
    pub fn new(session: SpiralSession, rows: u32, cols: u32) -> Self {
        let config = LightningConfig::new(rows, cols);
        Self::with_config(session, config)
    }

    /// Creates a builder that can be used to customise the harness before construction.
    pub fn builder(session: SpiralSession) -> LightningBuilder {
        LightningBuilder::new(session)
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

    /// Enables or disables automatic module preparation.
    pub fn set_auto_prepare(&mut self, auto_prepare: bool) {
        if self.config.auto_prepare == auto_prepare {
            return;
        }
        self.config.auto_prepare = auto_prepare;
        self.prepared_modules.clear();
    }

    /// Rebuilds the internal schedule using a new configuration.
    pub fn reconfigure(&mut self, mut config: LightningConfig) {
        // Ensure we reuse the latest trainer state when refreshing the schedule.
        config.rows = config.rows.max(1);
        config.cols = config.cols.max(1);
        let schedule = self
            .trainer
            .roundtable(config.rows, config.cols, config.roundtable);
        let schedule_changed = self.config.rows != config.rows
            || self.config.cols != config.cols
            || roundtable_changed(&self.config.roundtable, &config.roundtable);
        let auto_changed = self.config.auto_prepare != config.auto_prepare;
        self.schedule = schedule;
        self.config = config;
        if schedule_changed || auto_changed {
            self.prepared_modules.clear();
        }
    }

    /// Explicitly prepares the provided module using the session guard.
    pub fn prepare_module<M: Module>(&mut self, module: &mut M) -> PureResult<()> {
        self.session.prepare_module(module)?;
        let key = ModuleKey::new(module);
        self.prepared_modules.insert(key);
        Ok(())
    }

    fn ensure_prepared<M: Module>(&mut self, module: &mut M) -> PureResult<()> {
        if !self.config.auto_prepare {
            return Ok(());
        }
        let key = ModuleKey::new(module);
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

    #[cfg(test)]
    pub(crate) fn prepared_module_count(&self) -> usize {
        self.prepared_modules.len()
    }
}

/// Builder used to create [`SpiralLightning`] harnesses with customised settings.
#[derive(Debug)]
pub struct LightningBuilder {
    session: SpiralSession,
    config: LightningConfigBuilder,
}

impl LightningBuilder {
    fn new(session: SpiralSession) -> Self {
        Self {
            session,
            config: LightningConfigBuilder::default(),
        }
    }

    /// Overrides the output shape used for the internal roundtable.
    pub fn output_shape(mut self, rows: u32, cols: u32) -> Self {
        self.config = self.config.output_shape(rows, cols);
        self
    }

    /// Overrides the roundtable configuration applied to the harness.
    pub fn roundtable(mut self, config: RoundtableConfig) -> Self {
        self.config = self.config.roundtable(config);
        self
    }

    /// Enables or disables automatic module preparation.
    pub fn auto_prepare(mut self, enabled: bool) -> Self {
        self.config = self.config.auto_prepare(enabled);
        self
    }

    /// Applies an arbitrary transformation over the builder prior to construction.
    pub fn configure_with(
        mut self,
        f: impl FnOnce(LightningConfigBuilder) -> LightningConfigBuilder,
    ) -> Self {
        self.config = f(self.config);
        self
    }

    /// Consumes the builder and returns the constructed [`SpiralLightning`].
    pub fn build(self) -> SpiralLightning {
        SpiralLightning::with_config(self.session, self.config.build())
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

    #[test]
    fn lightning_builder_configures_harness() {
        let session = SpiralSession::builder(DeviceCaps::cpu())
            .with_curvature(-0.5)
            .build()
            .unwrap();
        let roundtable = RoundtableConfig::default().with_top_k(4).with_mid_k(3);
        let lightning = SpiralLightning::builder(session.clone())
            .output_shape(3, 5)
            .roundtable(roundtable)
            .auto_prepare(false)
            .build();
        assert_eq!(lightning.config().rows(), 3);
        assert_eq!(lightning.config().cols(), 5);
        assert_eq!(lightning.config().roundtable().top_k, 4);
        assert_eq!(lightning.config().roundtable().mid_k, 3);
        assert!(!lightning.config().auto_prepare());
        // Builder should not consume the session clone allowing further usage.
        let mut from_config = SpiralLightning::with_config(session, lightning.config().clone());
        let mut model = Linear::new("demo", 1, 1).unwrap();
        let mut loss = MeanSquaredError::new();
        from_config
            .train_epoch(
                &mut model,
                &mut loss,
                vec![(tensor(&[0.0]), tensor(&[0.0]))],
            )
            .unwrap();
    }

    #[test]
    fn reconfigure_clears_prepared_registry_when_schedule_changes() {
        let session = SpiralSession::builder(DeviceCaps::cpu())
            .with_curvature(-1.0)
            .build()
            .unwrap();
        let mut lightning = SpiralLightning::new(session, 1, 1);
        let mut model = Linear::new("demo", 1, 1).unwrap();
        lightning.prepare_module(&mut model).unwrap();
        assert_eq!(lightning.prepared_module_count(), 1);
        let new_config = lightning
            .config()
            .to_builder()
            .roundtable(RoundtableConfig::default().with_top_k(2))
            .build();
        lightning.reconfigure(new_config);
        assert_eq!(lightning.prepared_module_count(), 0);
    }

    #[test]
    fn toggling_auto_prepare_resets_registry() {
        let session = SpiralSession::builder(DeviceCaps::cpu())
            .with_curvature(-1.0)
            .build()
            .unwrap();
        let mut lightning = SpiralLightning::new(session, 1, 1);
        let mut model = Linear::new("demo", 1, 1).unwrap();
        lightning.prepare_module(&mut model).unwrap();
        assert_eq!(lightning.prepared_module_count(), 1);
        lightning.set_auto_prepare(false);
        assert_eq!(lightning.prepared_module_count(), 0);
        lightning.set_auto_prepare(true);
        assert_eq!(lightning.config().auto_prepare(), true);
    }
}
