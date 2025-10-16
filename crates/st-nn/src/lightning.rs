// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::dataset::DataLoaderBatches;
use crate::highlevel::SpiralSession;
use crate::schedule::{RoundtableConfig, RoundtableSchedule};
use crate::trainer::{EpochStats, IntoBatch, ModuleTrainer};
use crate::{Loss, Module, PureResult, Tensor};
use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;

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

impl LightningConfig {
    fn clamp_shape(&mut self) {
        self.rows = self.rows.max(1);
        self.cols = self.cols.max(1);
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

/// Collection of batches that will be executed inside a [`LightningStage`].
pub enum LightningEpoch {
    /// Raw `(input, target)` tuples that will be iterated in-process.
    Dataset(Vec<(Tensor, Tensor)>),
    /// Streaming batches driven by the native [`DataLoader`] surface.
    Loader(DataLoaderBatches),
}

impl fmt::Debug for LightningEpoch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dataset(batches) => f
                .debug_struct("Dataset")
                .field("batches", &batches.len())
                .finish(),
            Self::Loader(_) => f.write_str("Loader(DataLoaderBatches)"),
        }
    }
}

impl LightningEpoch {
    /// Creates an epoch from an eager dataset.
    pub fn from_dataset(batches: Vec<(Tensor, Tensor)>) -> Self {
        Self::Dataset(batches)
    }

    /// Creates an epoch from a [`DataLoaderBatches`] stream.
    pub fn from_loader(loader: DataLoaderBatches) -> Self {
        Self::Loader(loader)
    }
}

/// Stage executed by [`SpiralLightning::fit_plan`] with an optional label.
#[derive(Debug)]
pub struct LightningStage {
    config: LightningConfig,
    epochs: Vec<LightningEpoch>,
    label: Option<String>,
}

impl LightningStage {
    /// Builds an empty stage that inherits the provided configuration.
    pub fn new(config: LightningConfig) -> Self {
        Self {
            config,
            epochs: Vec::new(),
            label: None,
        }
    }

    /// Builds a stage populated with the supplied epochs.
    pub fn with_epochs(config: LightningConfig, epochs: Vec<LightningEpoch>) -> Self {
        Self {
            config,
            epochs,
            label: None,
        }
    }

    /// Adds a descriptive label to the stage.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Sets or replaces the stage label in-place.
    pub fn set_label(&mut self, label: impl Into<String>) {
        self.label = Some(label.into());
    }

    /// Appends an epoch to the stage and returns `self` for chaining.
    pub fn add_epoch(mut self, epoch: LightningEpoch) -> Self {
        self.epochs.push(epoch);
        self
    }

    /// Pushes an epoch into the stage without consuming `self`.
    pub fn push_epoch(&mut self, epoch: LightningEpoch) {
        self.epochs.push(epoch);
    }

    /// Returns the label associated with the stage, if any.
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Returns the configuration used by the stage.
    pub fn config(&self) -> &LightningConfig {
        &self.config
    }

    /// Returns the epochs scheduled for the stage.
    pub fn epochs(&self) -> &[LightningEpoch] {
        &self.epochs
    }

    /// Deconstructs the stage into its components.
    fn into_parts(self) -> (LightningConfig, Vec<LightningEpoch>, Option<String>) {
        (self.config, self.epochs, self.label)
    }
}

/// Aggregated statistics for a stage executed by [`SpiralLightning::fit_plan`].
#[derive(Debug, Clone)]
pub struct LightningStageReport {
    config: LightningConfig,
    label: Option<String>,
    epochs: Vec<EpochStats>,
}

impl LightningStageReport {
    /// Creates a stage report from the executed configuration and epoch metrics.
    pub fn new(config: LightningConfig, label: Option<String>, epochs: Vec<EpochStats>) -> Self {
        Self {
            config,
            label,
            epochs,
        }
    }

    /// Returns the configuration that produced this stage.
    pub fn config(&self) -> &LightningConfig {
        &self.config
    }

    /// Returns the optional stage label.
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Returns a slice of epoch metrics gathered for the stage.
    pub fn epochs(&self) -> &[EpochStats] {
        &self.epochs
    }

    /// Returns the sum of batches processed across all epochs.
    pub fn total_batches(&self) -> usize {
        self.epochs.iter().map(|epoch| epoch.batches).sum()
    }

    /// Returns the best epoch (lowest average loss) observed during the stage.
    pub fn best_epoch(&self) -> Option<EpochStats> {
        self.epochs.iter().copied().min_by(|left, right| {
            match left.average_loss.partial_cmp(&right.average_loss) {
                Some(order) => order,
                None => Ordering::Equal,
            }
        })
    }
}

/// Report containing the ordered stage summaries emitted by [`fit_plan`](SpiralLightning::fit_plan).
#[derive(Debug, Clone)]
pub struct LightningReport {
    stages: Vec<LightningStageReport>,
}

impl LightningReport {
    /// Creates a report from an ordered list of stage summaries.
    pub fn new(stages: Vec<LightningStageReport>) -> Self {
        Self { stages }
    }

    /// Returns the stage summaries in execution order.
    pub fn stages(&self) -> &[LightningStageReport] {
        &self.stages
    }

    /// Returns an iterator over every epoch recorded in the report.
    pub fn epochs(&self) -> impl Iterator<Item = &EpochStats> {
        self.stages.iter().flat_map(|stage| stage.epochs())
    }

    /// Returns the total number of epochs executed across all stages.
    pub fn total_epochs(&self) -> usize {
        self.stages.iter().map(|stage| stage.epochs.len()).sum()
    }

    /// Returns the total number of batches processed across the full plan.
    pub fn total_batches(&self) -> usize {
        self.stages.iter().map(|stage| stage.total_batches()).sum()
    }

    /// Returns the best epoch observed during the plan.
    pub fn best_epoch(&self) -> Option<EpochStats> {
        self.stages
            .iter()
            .filter_map(|stage| stage.best_epoch())
            .min_by(
                |left, right| match left.average_loss.partial_cmp(&right.average_loss) {
                    Some(order) => order,
                    None => Ordering::Equal,
                },
            )
    }

    /// Returns the index of the stage that produced the best epoch (if any).
    pub fn best_stage_index(&self) -> Option<usize> {
        self.stages
            .iter()
            .enumerate()
            .filter_map(|(idx, stage)| stage.best_epoch().map(|best| (idx, best.average_loss)))
            .min_by(|left, right| match left.1.partial_cmp(&right.1) {
                Some(order) => order,
                None => Ordering::Equal,
            })
            .map(|(idx, _)| idx)
    }

    /// Consumes the report and returns the owned stage summaries.
    pub fn into_stages(self) -> Vec<LightningStageReport> {
        self.stages
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
    pub fn with_config(session: SpiralSession, mut config: LightningConfig) -> Self {
        config.clamp_shape();
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
        config.clamp_shape();
        let shape_matches = self.config.rows == config.rows && self.config.cols == config.cols;
        let roundtable_matches = !roundtable_changed(&self.config.roundtable, &config.roundtable);
        let auto_changed = self.config.auto_prepare != config.auto_prepare;
        let schedule_changed = !(shape_matches && roundtable_matches);

        if !schedule_changed && !auto_changed {
            self.config = config;
            return;
        }

        if schedule_changed {
            self.schedule = self
                .trainer
                .roundtable(config.rows, config.cols, config.roundtable);
        }

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
        I: IntoIterator,
        I::Item: IntoBatch,
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
        I: IntoIterator,
        I::Item: IntoBatch,
    {
        let mut results = Vec::new();
        for batches in epochs {
            let stats = self.train_epoch(module, loss, batches)?;
            results.push(stats);
        }
        Ok(results)
    }

    /// Executes a staged training plan, reconfiguring the roundtable between stages.
    pub fn fit_plan<M, L, It>(
        &mut self,
        module: &mut M,
        loss: &mut L,
        stages: It,
    ) -> PureResult<LightningReport>
    where
        M: Module,
        L: Loss,
        It: IntoIterator<Item = LightningStage>,
    {
        let stage_iter = stages.into_iter();
        let (reserve, _) = stage_iter.size_hint();
        let mut reports = Vec::with_capacity(reserve);
        for stage in stage_iter {
            let (mut config, epochs, label) = stage.into_parts();
            config.clamp_shape();
            self.reconfigure(config.clone());
            let mut epoch_stats = Vec::with_capacity(epochs.len());
            for epoch in epochs {
                let stats = match epoch {
                    LightningEpoch::Dataset(data) => self.train_epoch(module, loss, data)?,
                    LightningEpoch::Loader(loader) => self.train_epoch(module, loss, loader)?,
                };
                epoch_stats.push(stats);
            }
            reports.push(LightningStageReport::new(config, label, epoch_stats));
        }
        Ok(LightningReport::new(reports))
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

    #[test]
    fn lightning_fit_plan_reconfigures_between_stages() {
        let session = SpiralSession::builder(DeviceCaps::cpu())
            .with_curvature(-1.0)
            .build()
            .unwrap();
        let mut lightning = SpiralLightning::new(session, 1, 1);
        let mut model = Linear::new("demo", 1, 1).unwrap();
        let mut loss = MeanSquaredError::new();

        let warmup_epoch = vec![(tensor(&[0.0]), tensor(&[0.0]))];
        let warmup_stage = LightningStage::new(lightning.config().clone())
            .with_label("warmup")
            .add_epoch(LightningEpoch::from_dataset(warmup_epoch.clone()))
            .add_epoch(LightningEpoch::from_dataset(warmup_epoch));

        let refine_config = lightning
            .config()
            .to_builder()
            .roundtable(RoundtableConfig::default().with_top_k(2))
            .auto_prepare(false)
            .build();
        let refine_epoch = vec![(tensor(&[1.0]), tensor(&[1.0]))];
        let refine_stage = LightningStage::new(refine_config)
            .with_label("refine")
            .add_epoch(LightningEpoch::from_dataset(refine_epoch));

        let report = lightning
            .fit_plan(&mut model, &mut loss, vec![warmup_stage, refine_stage])
            .unwrap();

        assert_eq!(report.stages().len(), 2);
        assert_eq!(report.total_epochs(), 3);
        assert_eq!(report.total_batches(), 3);
        assert_eq!(report.stages()[0].label(), Some("warmup"));
        assert_eq!(report.stages()[1].label(), Some("refine"));
        assert_eq!(report.stages()[1].config().roundtable().top_k, 2);
        assert_eq!(report.stages()[1].config().auto_prepare(), false);
        assert!(report.best_epoch().is_some());
        assert!(report.best_stage_index().is_some());
        assert_eq!(lightning.config().roundtable().top_k, 2);
        assert!(!lightning.config().auto_prepare());
    }
}
