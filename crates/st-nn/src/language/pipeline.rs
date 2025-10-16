// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::automation::{DesireAutomatedStep, DesireAutomation, DesireRewriteTrigger};
use super::desire::{DesirePhase, DesireWeights};
use super::geometry::ConceptHint;
use super::logbook::{DesireLogReplay, DesireLogbook};
use crate::roundtable::RoundtableNode;
use crate::{PureResult, RoundtableConfig, RoundtableSchedule};
use st_core::ecosystem::{
    ConnectorEvent, DistributionSummary, EcosystemRegistry, HeuristicChoiceSummary,
    HeuristicDecision, HeuristicSource, MetricSample, RankPlanSummary, RoundtableConfigSummary,
    RoundtableSummary,
};
use st_core::ops::rank_entry::RankPlan;
use st_tensor::pure::{ComplexTensor, LanguageWaveEncoder, Tensor, TensorError};
use std::collections::HashMap;
use std::sync::{mpsc::Sender, Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Sink interface used by [`DesirePipeline`] to braid automation steps into external systems.
pub trait DesirePipelineSink: Send {
    fn on_step(&mut self, step: &DesireAutomatedStep, timestamp: SystemTime) -> PureResult<()>;

    fn on_trigger(
        &mut self,
        _trigger: &DesireRewriteTrigger,
        _timestamp: SystemTime,
    ) -> PureResult<()> {
        Ok(())
    }

    fn flush(&mut self) -> PureResult<()> {
        Ok(())
    }
}

impl DesirePipelineSink for DesireLogbook {
    fn on_step(&mut self, step: &DesireAutomatedStep, timestamp: SystemTime) -> PureResult<()> {
        DesireLogbook::record(self, step, timestamp)
    }

    fn flush(&mut self) -> PureResult<()> {
        DesireLogbook::flush(self)
    }
}

#[derive(Clone, Debug)]
pub enum DesirePipelineEvent {
    Step {
        step: DesireAutomatedStep,
        timestamp: SystemTime,
    },
    Trigger {
        trigger: DesireRewriteTrigger,
        timestamp: SystemTime,
    },
}

impl DesirePipelineEvent {
    pub fn timestamp(&self) -> SystemTime {
        match self {
            DesirePipelineEvent::Step { timestamp, .. }
            | DesirePipelineEvent::Trigger { timestamp, .. } => *timestamp,
        }
    }
}

pub struct DesireChannelSink {
    sender: Sender<DesirePipelineEvent>,
}

impl DesireChannelSink {
    pub fn new(sender: Sender<DesirePipelineEvent>) -> Self {
        Self { sender }
    }

    pub fn sender(&self) -> Sender<DesirePipelineEvent> {
        self.sender.clone()
    }
}

impl DesirePipelineSink for DesireChannelSink {
    fn on_step(&mut self, step: &DesireAutomatedStep, timestamp: SystemTime) -> PureResult<()> {
        self.sender
            .send(DesirePipelineEvent::Step {
                step: step.clone(),
                timestamp,
            })
            .map_err(|_| TensorError::InvalidValue {
                label: "desire channel receiver dropped",
            })
    }

    fn on_trigger(
        &mut self,
        trigger: &DesireRewriteTrigger,
        timestamp: SystemTime,
    ) -> PureResult<()> {
        self.sender
            .send(DesirePipelineEvent::Trigger {
                trigger: trigger.clone(),
                timestamp,
            })
            .map_err(|_| TensorError::InvalidValue {
                label: "desire channel receiver dropped",
            })
    }
}

pub struct DesirePipelineBuilder {
    automation: DesireAutomation,
    sinks: Vec<Box<dyn DesirePipelineSink>>,
}

impl DesirePipelineBuilder {
    pub fn new(automation: DesireAutomation) -> Self {
        Self {
            automation,
            sinks: Vec::new(),
        }
    }

    pub fn with_sink<S>(mut self, sink: S) -> Self
    where
        S: DesirePipelineSink + 'static,
    {
        self.sinks.push(Box::new(sink));
        self
    }

    pub fn with_logbook(mut self, logbook: DesireLogbook) -> Self {
        self.sinks.push(Box::new(logbook));
        self
    }

    pub fn with_channel(mut self, sender: Sender<DesirePipelineEvent>) -> Self {
        self.sinks.push(Box::new(DesireChannelSink::new(sender)));
        self
    }

    pub fn with_trainer_bridge(mut self, bridge: &DesireTrainerBridge) -> Self {
        self.sinks.push(Box::new(bridge.clone()));
        self
    }

    pub fn build(self) -> DesirePipeline {
        DesirePipeline {
            automation: self.automation,
            sinks: self.sinks,
        }
    }
}

pub struct DesirePipeline {
    automation: DesireAutomation,
    sinks: Vec<Box<dyn DesirePipelineSink>>,
}

impl DesirePipeline {
    pub fn new(automation: DesireAutomation) -> Self {
        Self {
            automation,
            sinks: Vec::new(),
        }
    }

    pub fn builder(automation: DesireAutomation) -> DesirePipelineBuilder {
        DesirePipelineBuilder::new(automation)
    }

    pub fn automation(&self) -> &DesireAutomation {
        &self.automation
    }

    pub fn automation_mut(&mut self) -> &mut DesireAutomation {
        &mut self.automation
    }

    pub fn attach_sink<S>(&mut self, sink: S)
    where
        S: DesirePipelineSink + 'static,
    {
        self.sinks.push(Box::new(sink));
    }

    pub fn attach_logbook(&mut self, logbook: DesireLogbook) {
        self.sinks.push(Box::new(logbook));
    }

    pub fn attach_channel(&mut self, sender: Sender<DesirePipelineEvent>) {
        self.sinks.push(Box::new(DesireChannelSink::new(sender)));
    }

    pub fn attach_trainer_bridge(&mut self, bridge: &DesireTrainerBridge) {
        self.sinks.push(Box::new(bridge.clone()));
    }

    pub fn sink_count(&self) -> usize {
        self.sinks.len()
    }

    pub fn step_at(
        &mut self,
        lm_logits: &[f32],
        previous_token: usize,
        concept_hint: &ConceptHint,
        now: Instant,
        timestamp: SystemTime,
    ) -> PureResult<DesireAutomatedStep> {
        let step = self
            .automation
            .step(lm_logits, previous_token, concept_hint, now)?;
        self.dispatch(&step, timestamp)?;
        Ok(step)
    }

    pub fn step_with_weights_at(
        &mut self,
        lm_logits: &[f32],
        previous_token: usize,
        concept_hint: &ConceptHint,
        weights: &DesireWeights,
        now: Instant,
        timestamp: SystemTime,
    ) -> PureResult<DesireAutomatedStep> {
        let step = self.automation.step_with_weights(
            lm_logits,
            previous_token,
            concept_hint,
            weights,
            now,
        )?;
        self.dispatch(&step, timestamp)?;
        Ok(step)
    }

    pub fn step_realtime(
        &mut self,
        lm_logits: &[f32],
        previous_token: usize,
        concept_hint: &ConceptHint,
    ) -> PureResult<DesireAutomatedStep> {
        let now = Instant::now();
        let timestamp = SystemTime::now();
        self.step_at(lm_logits, previous_token, concept_hint, now, timestamp)
    }

    pub fn step_with_weights_realtime(
        &mut self,
        lm_logits: &[f32],
        previous_token: usize,
        concept_hint: &ConceptHint,
        weights: &DesireWeights,
    ) -> PureResult<DesireAutomatedStep> {
        let now = Instant::now();
        let timestamp = SystemTime::now();
        self.step_with_weights_at(
            lm_logits,
            previous_token,
            concept_hint,
            weights,
            now,
            timestamp,
        )
    }

    pub fn replay(&mut self, replay: DesireLogReplay) -> PureResult<usize> {
        let mut count = 0usize;
        for entry in replay {
            let record = entry?;
            let step = DesireAutomatedStep {
                solution: record.solution.clone(),
                trigger: record.trigger.clone(),
            };
            let timestamp = timestamp_from_millis(record.timestamp_ms);
            self.dispatch(&step, timestamp)?;
            count = count.saturating_add(1);
        }
        Ok(count)
    }

    pub fn flush(&mut self) -> PureResult<()> {
        for sink in self.sinks.iter_mut() {
            sink.flush()?;
        }
        Ok(())
    }

    fn dispatch(&mut self, step: &DesireAutomatedStep, timestamp: SystemTime) -> PureResult<()> {
        for sink in self.sinks.iter_mut() {
            sink.on_step(step, timestamp)?;
        }
        if let Some(trigger) = &step.trigger {
            for sink in self.sinks.iter_mut() {
                sink.on_trigger(trigger, timestamp)?;
            }
        }
        Ok(())
    }
}

impl Drop for DesirePipeline {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

fn timestamp_from_millis(ms: u128) -> SystemTime {
    let clamped = ms.min(u64::MAX as u128) as u64;
    UNIX_EPOCH + Duration::from_millis(clamped)
}

#[derive(Clone, Debug)]
pub struct DesireTriggerEvent {
    pub trigger: DesireRewriteTrigger,
    pub timestamp: SystemTime,
}

#[derive(Clone, Default)]
pub struct DesireTriggerBuffer {
    shared: Arc<Mutex<Vec<DesireTriggerEvent>>>,
}

impl DesireTriggerBuffer {
    pub fn new() -> Self {
        Self {
            shared: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn len(&self) -> usize {
        match self.shared.lock() {
            Ok(guard) => guard.len(),
            Err(_) => 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn drain(&self) -> PureResult<Vec<DesireTriggerEvent>> {
        let mut guard = self.shared.lock().map_err(|_| TensorError::InvalidValue {
            label: "desire trigger buffer poisoned",
        })?;
        Ok(std::mem::take(&mut *guard))
    }
}

impl DesirePipelineSink for DesireTriggerBuffer {
    fn on_step(&mut self, _step: &DesireAutomatedStep, _timestamp: SystemTime) -> PureResult<()> {
        Ok(())
    }

    fn on_trigger(
        &mut self,
        trigger: &DesireRewriteTrigger,
        timestamp: SystemTime,
    ) -> PureResult<()> {
        let mut guard = self.shared.lock().map_err(|_| TensorError::InvalidValue {
            label: "desire trigger buffer poisoned",
        })?;
        guard.push(DesireTriggerEvent {
            trigger: trigger.clone(),
            timestamp,
        });
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct DesireTrainerEvent {
    pub timestamp: SystemTime,
    pub phase: DesirePhase,
    pub temperature: f32,
    pub entropy: f32,
    pub hypergrad_penalty: f32,
    pub weights: DesireWeights,
    pub trigger: Option<DesireRewriteTrigger>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DesireTrainerSummary {
    pub total: usize,
    pub observation: usize,
    pub injection: usize,
    pub integration: usize,
    pub triggers: usize,
    pub mean_entropy: f32,
    pub mean_temperature: f32,
    pub mean_penalty: f32,
    pub mean_alpha: f32,
    pub mean_beta: f32,
    pub mean_gamma: f32,
    pub mean_lambda: f32,
    pub trigger_mean_penalty: f32,
    pub trigger_mean_entropy: f32,
    pub trigger_mean_temperature: f32,
    pub trigger_mean_samples: f32,
}

impl DesireTrainerSummary {
    fn from_events(events: &[DesireTrainerEvent]) -> Self {
        if events.is_empty() {
            return Self::default();
        }
        let mut summary = DesireTrainerSummary {
            total: events.len(),
            ..Default::default()
        };
        let mut sum_entropy = 0.0f32;
        let mut sum_temperature = 0.0f32;
        let mut sum_penalty = 0.0f32;
        let mut sum_alpha = 0.0f32;
        let mut sum_beta = 0.0f32;
        let mut sum_gamma = 0.0f32;
        let mut sum_lambda = 0.0f32;
        let mut trig_penalty = 0.0f32;
        let mut trig_entropy = 0.0f32;
        let mut trig_temperature = 0.0f32;
        let mut trig_samples = 0.0f32;

        for event in events {
            match event.phase {
                DesirePhase::Observation => summary.observation += 1,
                DesirePhase::Injection => summary.injection += 1,
                DesirePhase::Integration => summary.integration += 1,
            }
            sum_entropy += event.entropy;
            sum_temperature += event.temperature;
            sum_penalty += event.hypergrad_penalty;
            sum_alpha += event.weights.alpha;
            sum_beta += event.weights.beta;
            sum_gamma += event.weights.gamma;
            sum_lambda += event.weights.lambda;
            if let Some(trigger) = &event.trigger {
                summary.triggers += 1;
                trig_penalty += trigger.mean_penalty;
                trig_entropy += trigger.mean_entropy;
                trig_temperature += trigger.temperature;
                trig_samples += trigger.samples as f32;
            }
        }

        let total = summary.total as f32;
        if total > 0.0 {
            summary.mean_entropy = sum_entropy / total;
            summary.mean_temperature = sum_temperature / total;
            summary.mean_penalty = sum_penalty / total;
            summary.mean_alpha = sum_alpha / total;
            summary.mean_beta = sum_beta / total;
            summary.mean_gamma = sum_gamma / total;
            summary.mean_lambda = sum_lambda / total;
        }

        if summary.triggers > 0 {
            let count = summary.triggers as f32;
            summary.trigger_mean_penalty = trig_penalty / count;
            summary.trigger_mean_entropy = trig_entropy / count;
            summary.trigger_mean_temperature = trig_temperature / count;
            summary.trigger_mean_samples = trig_samples / count;
        }

        summary
    }
}

#[derive(Clone, Default)]
pub struct DesireTrainerBridge {
    shared: Arc<Mutex<Vec<DesireTrainerEvent>>>,
}

impl DesireTrainerBridge {
    pub fn new() -> Self {
        Self {
            shared: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn len(&self) -> usize {
        match self.shared.lock() {
            Ok(guard) => guard.len(),
            Err(_) => 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn drain(&self) -> PureResult<Vec<DesireTrainerEvent>> {
        let mut guard = self.shared.lock().map_err(|_| TensorError::InvalidValue {
            label: "desire trainer bridge poisoned",
        })?;
        Ok(std::mem::take(&mut *guard))
    }

    pub fn drain_summary(&self) -> PureResult<Option<DesireTrainerSummary>> {
        let events = self.drain()?;
        if events.is_empty() {
            return Ok(None);
        }
        Ok(Some(DesireTrainerSummary::from_events(&events)))
    }
}

impl DesirePipelineSink for DesireTrainerBridge {
    fn on_step(&mut self, step: &DesireAutomatedStep, timestamp: SystemTime) -> PureResult<()> {
        let mut guard = self.shared.lock().map_err(|_| TensorError::InvalidValue {
            label: "desire trainer bridge poisoned",
        })?;
        guard.push(DesireTrainerEvent {
            timestamp,
            phase: step.solution.phase,
            temperature: step.solution.temperature,
            entropy: step.solution.entropy,
            hypergrad_penalty: step.solution.hypergrad_penalty,
            weights: step.solution.weights.clone(),
            trigger: step.trigger.clone(),
        });
        Ok(())
    }
}

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
    use super::super::automation::DesireAutomation;
    use super::super::desire::{constant, warmup, DesireLagrangian};
    use super::super::geometry::{
        ConceptHint, RepressionField, SemanticBridge, SparseKernel, SymbolGeometry,
    };
    use super::super::temperature::TemperatureController;
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
    use st_core::config::self_rewrite::SelfRewriteCfg;
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
