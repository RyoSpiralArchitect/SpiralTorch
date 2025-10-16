// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::roundtable::RoundtableNode;
use crate::{RoundtableConfig, RoundtableSchedule};
use st_core::ecosystem::{
    ConnectorEvent, DistributionSummary, EcosystemRegistry, HeuristicChoiceSummary,
    HeuristicDecision, HeuristicSource, MetricSample, RankPlanSummary, RoundtableConfigSummary,
    RoundtableSummary,
};
use st_core::ops::rank_entry::RankPlan;
use st_tensor::pure::{ComplexTensor, LanguageWaveEncoder, Tensor, TensorError};
use std::collections::HashMap;
use std::time::{Instant, SystemTime};

#[derive(Debug)]
pub enum PipelineError {
    EncoderMissing { pipeline: String },
    Tensor(TensorError),
}

pub type PipelineResult<T> = Result<T, PipelineError>;

#[derive(Clone)]
pub struct LanguagePipelineBuilder {
    name: String,
    tags: HashMap<String, String>,
    encoder: Option<LanguageWaveEncoder>,
}

#[derive(Clone)]
pub struct LanguagePipeline {
    name: String,
    registry: &'static EcosystemRegistry,
    tags: HashMap<String, String>,
    encoder: Option<LanguageWaveEncoder>,
}

impl LanguagePipelineBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            tags: HashMap::new(),
            encoder: None,
        }
    }

    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    pub fn with_encoder(mut self, encoder: LanguageWaveEncoder) -> Self {
        self.encoder = Some(encoder);
        self
    }

    pub fn build(self) -> LanguagePipeline {
        LanguagePipeline {
            name: self.name,
            registry: EcosystemRegistry::global(),
            tags: self.tags,
            encoder: self.encoder,
        }
    }
}

impl LanguagePipeline {
    pub fn builder(name: impl Into<String>) -> LanguagePipelineBuilder {
        LanguagePipelineBuilder::new(name)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn encoder(&self) -> Option<&LanguageWaveEncoder> {
        self.encoder.as_ref()
    }

    pub fn record_metric(&self, sample: MetricSample) {
        self.registry.record_metric(self.apply_tags(sample, &[]));
    }

    pub fn record_heuristic(
        &self,
        subsystem: impl Into<String>,
        kind: impl Into<String>,
        rows: u32,
        cols: u32,
        k: u32,
        choice: HeuristicChoiceSummary,
        source: HeuristicSource,
        score_hint: Option<f32>,
    ) {
        let decision = HeuristicDecision {
            subsystem: subsystem.into(),
            kind: kind.into(),
            rows,
            cols,
            k,
            choice,
            score_hint,
            source,
            issued_at: SystemTime::now(),
        };
        self.registry.record_heuristic(decision);
    }

    pub fn record_roundtable(
        &self,
        rows: u32,
        cols: u32,
        config: RoundtableConfig,
        schedule: &RoundtableSchedule,
        autopilot_enabled: bool,
        distribution: Option<&RoundtableNode>,
    ) -> RoundtableSummary {
        let cfg_summary = summarise_config(config);
        let plans = vec![
            summarise_rank_plan(schedule.above()),
            summarise_rank_plan(schedule.here()),
            summarise_rank_plan(schedule.beneath()),
        ];
        let distribution_summary = distribution.map(summarise_distribution);
        let summary = RoundtableSummary {
            rows,
            cols,
            config: cfg_summary,
            plans,
            autopilot_enabled,
            distribution: distribution_summary.clone(),
            issued_at: SystemTime::now(),
        };

        let mut extra_tags = vec![("autopilot".to_string(), autopilot_enabled.to_string())];
        if let Some(dist) = &distribution_summary {
            extra_tags.push(("distribution_mode".to_string(), dist.mode.clone()));
        }

        self.registry.record_metric(self.apply_tags(
            MetricSample::new("roundtable.rows", rows as f64).with_unit("rows"),
            &extra_tags,
        ));
        self.registry.record_metric(self.apply_tags(
            MetricSample::new("roundtable.cols", cols as f64).with_unit("cols"),
            &extra_tags,
        ));
        self.registry.record_metric(
            self.apply_tags(
                MetricSample::new(
                    "roundtable.autopilot",
                    if autopilot_enabled { 1.0 } else { 0.0 },
                )
                .with_unit("flag"),
                &extra_tags,
            ),
        );
        self.registry.record_metric(self.apply_tags(
            MetricSample::new("roundtable.config.top_k", config.top_k as f64).with_unit("items"),
            &extra_tags,
        ));
        self.registry.record_metric(self.apply_tags(
            MetricSample::new("roundtable.config.mid_k", config.mid_k as f64).with_unit("items"),
            &extra_tags,
        ));
        self.registry.record_metric(
            self.apply_tags(
                MetricSample::new("roundtable.config.bottom_k", config.bottom_k as f64)
                    .with_unit("items"),
                &extra_tags,
            ),
        );
        self.registry.record_metric(
            self.apply_tags(
                MetricSample::new(
                    "roundtable.config.here_tolerance",
                    config.here_tolerance as f64,
                )
                .with_unit("ratio"),
                &extra_tags,
            ),
        );

        for (band, plan) in [
            ("above", schedule.above()),
            ("here", schedule.here()),
            ("beneath", schedule.beneath()),
        ] {
            let mut band_tags = extra_tags.clone();
            band_tags.push(("band".to_string(), band.to_string()));
            self.registry.record_metric(self.apply_tags(
                MetricSample::new("roundtable.band.rows", plan.rows as f64).with_unit("rows"),
                &band_tags,
            ));
            self.registry.record_metric(self.apply_tags(
                MetricSample::new("roundtable.band.cols", plan.cols as f64).with_unit("cols"),
                &band_tags,
            ));
            self.registry.record_metric(self.apply_tags(
                MetricSample::new("roundtable.band.k", plan.k as f64).with_unit("items"),
                &band_tags,
            ));
            self.registry.record_metric(
                self.apply_tags(
                    MetricSample::new("roundtable.band.workgroup", plan.choice.wg as f64)
                        .with_unit("threads"),
                    &band_tags,
                ),
            );
            self.registry.record_metric(
                self.apply_tags(
                    MetricSample::new("roundtable.band.lanes", plan.choice.kl as f64)
                        .with_unit("lanes"),
                    &band_tags,
                ),
            );
            self.registry.record_metric(
                self.apply_tags(
                    MetricSample::new("roundtable.band.channel_stride", plan.choice.ch as f64)
                        .with_unit("stride"),
                    &band_tags,
                ),
            );
            self.registry.record_metric(
                self.apply_tags(
                    MetricSample::new("roundtable.band.tile", plan.choice.tile as f64)
                        .with_unit("tile"),
                    &band_tags,
                ),
            );
            self.registry.record_metric(
                self.apply_tags(
                    MetricSample::new("roundtable.band.compaction_tile", plan.choice.ctile as f64)
                        .with_unit("tile"),
                    &band_tags,
                ),
            );
            self.registry.record_metric(
                self.apply_tags(
                    MetricSample::new(
                        "roundtable.band.subgroup",
                        if plan.choice.subgroup { 1.0 } else { 0.0 },
                    )
                    .with_unit("flag"),
                    &band_tags,
                ),
            );
            self.registry.record_metric(
                self.apply_tags(
                    MetricSample::new("roundtable.band.fft_tile", plan.choice.fft_tile as f64)
                        .with_unit("tile"),
                    &band_tags,
                ),
            );
            self.registry.record_metric(
                self.apply_tags(
                    MetricSample::new("roundtable.band.fft_radix", plan.choice.fft_radix as f64)
                        .with_unit("radix"),
                    &band_tags,
                ),
            );
            self.registry.record_metric(
                self.apply_tags(
                    MetricSample::new(
                        "roundtable.band.fft_segments",
                        plan.choice.fft_segments as f64,
                    )
                    .with_unit("segments"),
                    &band_tags,
                ),
            );
        }

        self.registry.record_roundtable(summary.clone());

        let mut connector_metadata = vec![
            ("rows".to_string(), rows.to_string()),
            ("cols".to_string(), cols.to_string()),
            ("autopilot".to_string(), autopilot_enabled.to_string()),
            (
                "plans".to_string(),
                summary
                    .plans
                    .iter()
                    .map(|plan| plan.kind.as_str())
                    .collect::<Vec<_>>()
                    .join(","),
            ),
        ];
        if let Some(dist) = &distribution_summary {
            connector_metadata.push(("distribution_mode".to_string(), dist.mode.clone()));
            connector_metadata.push(("node_id".to_string(), dist.node_id.clone()));
        }
        self.record_connector("roundtable", connector_metadata);

        summary
    }

    pub fn encode_wave(&self, text: &str) -> PipelineResult<ComplexTensor> {
        let encoder = self
            .encoder
            .as_ref()
            .ok_or_else(|| PipelineError::EncoderMissing {
                pipeline: self.name.clone(),
            })?;
        let start = Instant::now();
        let wave = encoder.encode_wave(text).map_err(PipelineError::from)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        let chars = text.chars().count() as f64;
        let (_, cols) = wave.shape();

        let extras = vec![("mode".to_string(), "wave".to_string())];
        self.registry.record_metric(self.apply_tags(
            MetricSample::new("language.encode.chars", chars).with_unit("chars"),
            &extras,
        ));
        self.registry.record_metric(self.apply_tags(
            MetricSample::new("language.encode.duration_ms", elapsed_ms).with_unit("ms"),
            &extras,
        ));
        self.registry.record_metric(self.apply_tags(
            MetricSample::new("language.encode.wave.cols", cols as f64).with_unit("cols"),
            &extras,
        ));
        self.registry.record_metric(
            self.apply_tags(
                MetricSample::new("language.encode.curvature", encoder.curvature() as f64)
                    .with_unit("curvature"),
                &extras,
            ),
        );
        self.registry.record_metric(
            self.apply_tags(
                MetricSample::new("language.encode.temperature", encoder.temperature() as f64)
                    .with_unit("temperature"),
                &extras,
            ),
        );

        self.record_connector(
            "encode",
            vec![
                ("mode".to_string(), "wave".to_string()),
                ("chars".to_string(), chars.to_string()),
                ("duration_ms".to_string(), format!("{elapsed_ms:.3}")),
                ("cols".to_string(), cols.to_string()),
            ],
        );

        Ok(wave)
    }

    pub fn encode_z_space(&self, text: &str) -> PipelineResult<Tensor> {
        let encoder = self
            .encoder
            .as_ref()
            .ok_or_else(|| PipelineError::EncoderMissing {
                pipeline: self.name.clone(),
            })?;
        let start = Instant::now();
        let tensor = encoder.encode_z_space(text).map_err(PipelineError::from)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        let chars = text.chars().count() as f64;
        let (_, cols) = tensor.shape();

        let extras = vec![("mode".to_string(), "z_space".to_string())];
        self.registry.record_metric(self.apply_tags(
            MetricSample::new("language.encode.chars", chars).with_unit("chars"),
            &extras,
        ));
        self.registry.record_metric(self.apply_tags(
            MetricSample::new("language.encode.duration_ms", elapsed_ms).with_unit("ms"),
            &extras,
        ));
        self.registry.record_metric(self.apply_tags(
            MetricSample::new("language.encode.zspace.cols", cols as f64).with_unit("cols"),
            &extras,
        ));
        self.registry.record_metric(
            self.apply_tags(
                MetricSample::new("language.encode.curvature", encoder.curvature() as f64)
                    .with_unit("curvature"),
                &extras,
            ),
        );
        self.registry.record_metric(
            self.apply_tags(
                MetricSample::new("language.encode.temperature", encoder.temperature() as f64)
                    .with_unit("temperature"),
                &extras,
            ),
        );

        self.record_connector(
            "encode",
            vec![
                ("mode".to_string(), "z_space".to_string()),
                ("chars".to_string(), chars.to_string()),
                ("duration_ms".to_string(), format!("{elapsed_ms:.3}")),
                ("cols".to_string(), cols.to_string()),
            ],
        );

        Ok(tensor)
    }

    pub fn record_connector(&self, stage: impl Into<String>, metadata: Vec<(String, String)>) {
        let mut map = HashMap::new();
        map.insert("pipeline".to_string(), self.name.clone());
        for (key, value) in self.tags.iter() {
            map.entry(key.clone()).or_insert(value.clone());
        }
        for (key, value) in metadata {
            map.insert(key, value);
        }
        self.registry.record_connector(ConnectorEvent {
            name: self.name.clone(),
            stage: stage.into(),
            metadata: map,
            issued_at: SystemTime::now(),
        });
    }
}

impl From<TensorError> for PipelineError {
    fn from(err: TensorError) -> Self {
        PipelineError::Tensor(err)
    }
}

impl core::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PipelineError::EncoderMissing { pipeline } => {
                write!(f, "language pipeline '{pipeline}' is missing an encoder")
            }
            PipelineError::Tensor(err) => err.fmt(f),
        }
    }
}

impl std::error::Error for PipelineError {}

fn summarise_rank_plan(plan: &RankPlan) -> RankPlanSummary {
    let mut summary = RankPlanSummary::new(plan.kind, plan.rows, plan.cols, plan.k);
    summary.workgroup = plan.choice.wg;
    summary.lanes = plan.choice.kl;
    summary.channel_stride = plan.choice.ch;
    summary.tile = plan.choice.tile;
    summary.compaction_tile = plan.choice.ctile;
    summary.subgroup = plan.choice.subgroup;
    summary.fft_tile = plan.choice.fft_tile;
    summary.fft_radix = plan.choice.fft_radix;
    summary.fft_segments = plan.choice.fft_segments;
    summary
}

fn summarise_distribution(node: &RoundtableNode) -> DistributionSummary {
    let cfg = node.config();
    DistributionSummary {
        node_id: cfg.node_id.clone(),
        mode: cfg.mode.as_str().to_string(),
        summary_window: cfg.summary_window,
        push_interval_ms: cfg.push_interval.as_millis().min(u64::MAX as u128) as u64,
        meta_endpoints: cfg.meta_endpoints.clone(),
    }
}

fn summarise_config(config: RoundtableConfig) -> RoundtableConfigSummary {
    #[allow(unused_mut)]
    let mut summary = RoundtableConfigSummary::new(
        config.top_k,
        config.mid_k,
        config.bottom_k,
        config.here_tolerance,
    );
    #[cfg(feature = "psychoid")]
    {
        summary
            .extras
            .insert("psychoid".to_string(), config.psychoid_enabled);
        if config.psychoid_log {
            summary.extras.insert("psychoid_log".to_string(), true);
        }
    }
    #[cfg(feature = "psi")]
    {
        summary.extras.insert("psi".to_string(), config.psi_enabled);
    }
    #[cfg(feature = "collapse")]
    {
        summary
            .extras
            .insert("collapse".to_string(), config.collapse_enabled);
    }
    summary
}

impl LanguagePipeline {
    fn apply_tags(&self, mut sample: MetricSample, extras: &[(String, String)]) -> MetricSample {
        sample = sample.with_tag("pipeline", self.name.clone());
        for (key, value) in &self.tags {
            sample = sample.with_tag(key.clone(), value.clone());
        }
        for (key, value) in extras {
            sample = sample.with_tag(key.clone(), value.clone());
        }
        sample
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::automation::DesireAutomation;
    use super::super::desire::{constant, warmup, DesireLagrangian};
    use super::super::geometry::{
        ConceptHint, RepressionField, SemanticBridge, SparseKernel, SymbolGeometry,
    };
    use super::super::temperature::TemperatureController;
    use crate::plan::RankPlanner;
    use st_core::config::self_rewrite::SelfRewriteCfg;
    use st_core::backend::device_caps::DeviceCaps;
    use std::collections::HashSet;
    use std::sync::mpsc::channel;
    use std::sync::{Mutex, OnceLock};
    use std::time::{Duration, Instant, SystemTime};
    use tempfile::tempdir;

    fn registry_guard() -> &'static Mutex<()> {
        static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
        GUARD.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn encode_wave_records_metrics_and_connector() {
        let _lock = registry_guard().lock().unwrap();
        let registry = EcosystemRegistry::global();
        registry.drain();
        let encoder = LanguageWaveEncoder::new(-1.0, 0.7).unwrap();
        let pipeline = LanguagePipeline::builder("language-test")
            .with_tag("tenant", "demo")
            .with_encoder(encoder)
            .build();
        let wave = pipeline.encode_wave("spiral torch").unwrap();
        assert_eq!(wave.shape().0, 1);

        let report = registry.drain();
        assert!(!report.metrics.is_empty());
        let mut saw_chars = false;
        for sample in &report.metrics {
            if sample.name == "language.encode.chars" {
                assert_eq!(sample.tags.get("mode"), Some(&"wave".to_string()));
                assert_eq!(
                    sample.tags.get("pipeline"),
                    Some(&"language-test".to_string())
                );
                assert_eq!(sample.tags.get("tenant"), Some(&"demo".to_string()));
                saw_chars = true;
            }
        }
        assert!(saw_chars, "missing language.encode.chars metric");
        assert_eq!(report.connectors.len(), 1);
        let connector = &report.connectors[0];
        assert_eq!(connector.name, "language-test");
        assert_eq!(connector.stage, "encode");
        assert_eq!(
            connector.metadata.get("pipeline"),
            Some(&"language-test".to_string())
        );
    }

    #[test]
    fn roundtable_records_summary_and_metrics() {
        let _lock = registry_guard().lock().unwrap();
        let registry = EcosystemRegistry::global();
        registry.drain();
        let pipeline = LanguagePipeline::builder("trainer").build();
        let planner = RankPlanner::new(DeviceCaps::wgpu(32, true, 256));
        let config = RoundtableConfig::default();
        let schedule = RoundtableSchedule::new(&planner, 16, 32, config);
        let summary = pipeline.record_roundtable(16, 32, config, &schedule, false, None);
        assert_eq!(summary.rows, 16);
        assert_eq!(summary.cols, 32);
        let report = registry.drain();
        assert_eq!(report.roundtables.len(), 1);
        assert!(report.metrics.iter().any(|m| m.name == "roundtable.rows"));
        assert_eq!(report.connectors.len(), 1);
        let connector = &report.connectors[0];
        assert_eq!(connector.stage, "roundtable");
        assert_eq!(connector.metadata.get("rows"), Some(&"16".to_string()));
    }

    fn build_geometry() -> SymbolGeometry {
        let syn = SparseKernel::from_rows(
            vec![vec![(0, 0.6), (1, 0.4)], vec![(0, 0.5), (1, 0.5)]],
            1e-6,
        )
        .unwrap();
        let par = SparseKernel::from_rows(
            vec![vec![(0, 0.7), (1, 0.3)], vec![(0, 0.2), (1, 0.8)]],
            1e-6,
        )
        .unwrap();
        SymbolGeometry::new(syn, par).unwrap()
    }

    fn build_semantics() -> SemanticBridge {
        let log_pi = vec![
            vec![(0, (0.7f32).ln()), (1, (0.3f32).ln())],
            vec![(0, (0.4f32).ln()), (1, (0.6f32).ln())],
        ];
        let row = vec![1.0, 1.0];
        let col = vec![1.0, 1.0];
        let anchors = HashSet::new();
        let concept_kernel =
            SparseKernel::from_rows(vec![vec![(0, 1.0)], vec![(1, 1.0)]], 1e-6).unwrap();
        SemanticBridge::new(log_pi, row, col, anchors, 1e-6, concept_kernel).unwrap()
    }

    fn build_automation() -> DesireAutomation {
        let geometry = build_geometry();
        let repression = RepressionField::new(vec![0.05, 0.15]).unwrap();
        let semantics = build_semantics();
        let controller = TemperatureController::new(1.0, 0.8, 0.4, 0.4, 1.6);
        let desire = DesireLagrangian::new(geometry, repression, semantics, controller)
            .unwrap()
            .with_alpha_schedule(warmup(0.0, 0.2, 1))
            .with_beta_schedule(warmup(0.0, 0.1, 1))
            .with_gamma_schedule(constant(0.04))
            .with_lambda_schedule(constant(0.02))
            .with_observation_horizon(Some(1))
            .with_integration_horizon(Some(2));
        let cfg = SelfRewriteCfg {
            score_thresh: 0.0,
            min_samples: 2,
            cooldown_sec: 0,
        };
        DesireAutomation::new(desire, cfg)
    }

    #[test]
    fn pipeline_threads_logbook_and_triggers() {
        let automation = build_automation();
        let dir = tempdir().unwrap();
        let path = dir.path().join("desire.ndjson");
        let logbook = DesireLogbook::with_flush_every(&path, 1).unwrap();
        let trigger_buffer = DesireTriggerBuffer::new();
        let mut pipeline = DesirePipeline::builder(automation)
            .with_logbook(logbook)
            .with_sink(trigger_buffer.clone())
            .build();

        let logits = vec![2.2, 0.4];
        let concept = ConceptHint::Distribution(vec![0.6, 0.4]);
        let start = Instant::now();
        let anchor = SystemTime::now();
        for step in 0..5 {
            let now = start + Duration::from_secs(step as u64 * 5);
            let timestamp = anchor + Duration::from_secs(step as u64 * 5);
            let result = pipeline
                .step_at(&logits, step % 2, &concept, now, timestamp)
                .unwrap();
            assert_eq!(result.solution.indices.len(), 2);
        }
        pipeline.flush().unwrap();
        assert_eq!(pipeline.sink_count(), 2);

        let replay = DesireLogReplay::open(&path).unwrap();
        let mut records = 0usize;
        for entry in replay {
            let record = entry.unwrap();
            if record.trigger.is_some() {
                assert!(!trigger_buffer.is_empty());
            }
            records += 1;
        }
        assert_eq!(records, 5);
        let drained = trigger_buffer.drain().unwrap();
        assert!(!drained.is_empty());
    }

    #[test]
    fn pipeline_replays_into_new_sinks() {
        let automation = build_automation();
        let dir = tempdir().unwrap();
        let path = dir.path().join("desire.ndjson");
        let logbook = DesireLogbook::with_flush_every(&path, 1).unwrap();
        let mut pipeline = DesirePipeline::builder(automation)
            .with_logbook(logbook)
            .build();
        let logits = vec![1.8, 0.9];
        let concept = ConceptHint::Distribution(vec![0.5, 0.5]);
        let start = Instant::now();
        let anchor = SystemTime::now();
        for step in 0..4 {
            let now = start + Duration::from_secs(step as u64 * 7);
            let timestamp = anchor + Duration::from_secs(step as u64 * 7);
            pipeline
                .step_at(&logits, step % 2, &concept, now, timestamp)
                .unwrap();
        }
        pipeline.flush().unwrap();

        let replay = DesireLogReplay::open(&path).unwrap();
        let collector = DesireTriggerBuffer::new();
        let automation_two = build_automation();
        let mut replay_pipeline = DesirePipeline::builder(automation_two)
            .with_sink(collector.clone())
            .build();
        let count = replay_pipeline.replay(replay).unwrap();
        assert_eq!(count, 4);
        assert_eq!(replay_pipeline.sink_count(), 1);
        assert!(collector.len() >= 1);
    }

    #[test]
    fn channel_sink_broadcasts_events() {
        let automation = build_automation();
        let (sender, receiver) = channel();
        let mut pipeline = DesirePipeline::builder(automation)
            .with_channel(sender)
            .build();

        let logits = vec![2.2, 0.4];
        let concept = ConceptHint::Distribution(vec![0.6, 0.4]);
        let start = Instant::now();
        let anchor = SystemTime::now();
        for step in 0..6 {
            let now = start + Duration::from_secs(step as u64);
            let timestamp = anchor + Duration::from_secs(step as u64);
            let result = pipeline
                .step_at(&logits, step % 2, &concept, now, timestamp)
                .unwrap();
            assert_eq!(result.solution.indices.len(), 2);
        }

        let mut step_events = 0usize;
        let mut trigger_events = 0usize;
        let mut timestamps = Vec::new();
        while let Ok(event) = receiver.try_recv() {
            match event {
                DesirePipelineEvent::Step { step, timestamp } => {
                    step_events += 1;
                    assert!(step.solution.indices.len() >= 2);
                    timestamps.push(timestamp);
                }
                DesirePipelineEvent::Trigger { trigger, timestamp } => {
                    trigger_events += 1;
                    assert!(trigger.samples >= 1);
                    timestamps.push(timestamp);
                }
            }
        }

        assert_eq!(step_events, 6);
        assert!(trigger_events >= 1);
        assert!(!timestamps.is_empty());
        for pair in timestamps.windows(2) {
            let a = pair[0]
                .duration_since(anchor)
                .unwrap_or_else(|_| Duration::from_secs(0));
            let b = pair[1]
                .duration_since(anchor)
                .unwrap_or_else(|_| Duration::from_secs(0));
            assert!(b >= a);
        }
    }

    #[test]
    fn trainer_bridge_collects_summaries() {
        let automation = build_automation();
        let bridge = DesireTrainerBridge::new();
        let mut pipeline = DesirePipeline::builder(automation)
            .with_trainer_bridge(&bridge)
            .build();

        let logits = vec![2.1, 0.5];
        let concept = ConceptHint::Distribution(vec![0.55, 0.45]);
        let start = Instant::now();
        let anchor = SystemTime::now();
        for step in 0..6 {
            let now = start + Duration::from_secs(step as u64);
            let timestamp = anchor + Duration::from_secs(step as u64);
            pipeline
                .step_at(&logits, step % 2, &concept, now, timestamp)
                .unwrap();
        }

        assert!(bridge.len() >= 6);
        let summary = bridge.drain_summary().unwrap().unwrap();
        assert_eq!(summary.total, 6);
        assert!(summary.observation >= 1);
        assert!(summary.integration >= 1);
        assert!(summary.mean_entropy.is_finite());
        assert!(summary.mean_temperature.is_finite());
        assert!(summary.mean_alpha >= 0.0);
        assert!(summary.triggers >= 1);
        assert!(bridge.is_empty());
        assert!(bridge.drain_summary().unwrap().is_none());
    }
}
