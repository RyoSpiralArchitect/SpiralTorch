// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::automation::{DesireAutomatedStep, DesireAutomation, DesireRewriteTrigger};
use super::desire::{DesirePhase, DesireSolution, DesireWeights};
use super::geometry::ConceptHint;
use super::logbook::{DesireLogReplay, DesireLogbook};
use crate::gnn::spiralk::{GraphConsensusBridge, GraphConsensusDigest};
use crate::schedule::BandEnergy;
use crate::PureResult;
use st_tensor::pure::TensorError;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::{mpsc::Sender, Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Sink interface used by [`DesirePipeline`] to braid automation steps into
/// external systems.
pub trait DesirePipelineSink: Send {
    /// Receives every automated step alongside the wall-clock timestamp that
    /// was supplied to the pipeline.
    fn on_step(&mut self, step: &DesireAutomatedStep, timestamp: SystemTime) -> PureResult<()>;

    /// Receives emitted rewrite triggers. Implementations can override this to
    /// react without duplicating persistence if the sink already captured the
    /// step in [`Self::on_step`].
    fn on_trigger(
        &mut self,
        _trigger: &DesireRewriteTrigger,
        _timestamp: SystemTime,
    ) -> PureResult<()> {
        Ok(())
    }

    /// Flushes any buffered side-effects. Called by [`DesirePipeline::flush`]
    /// and when the pipeline is dropped.
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

/// Event broadcast by [`DesirePipeline`] when a step is evaluated or a rewrite
/// trigger fires.
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

/// Simple sink that forwards every pipeline event into a channel so trainers,
/// schedulers, or external observers can subscribe without manual glue.
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

/// Builder used to configure the braid of sinks attached to a
/// [`DesirePipeline`].
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

    pub fn with_graph_bridge(mut self, bridge: &DesireGraphBridge) -> Self {
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

/// Coordinates desire automation, persistence, and trigger routing so the
/// "desire stack" can co-evolve with the rest of SpiralTorch.
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

    pub fn attach_graph_bridge(&mut self, bridge: &DesireGraphBridge) {
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

/// Recorded trigger event captured by [`DesireTriggerBuffer`].
#[derive(Clone, Debug)]
pub struct DesireTriggerEvent {
    pub trigger: DesireRewriteTrigger,
    pub timestamp: SystemTime,
}

/// Shared trigger collector that can be cloned by automation, SpiralK, and
/// analytics stages while all viewing the same underlying buffer.
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

#[derive(Clone, Debug)]
pub struct DesireGraphEvent {
    pub timestamp: SystemTime,
    pub solution: DesireSolution,
    pub trigger: Option<DesireRewriteTrigger>,
    pub digest: Option<GraphConsensusDigest>,
}

#[derive(Clone)]
pub struct DesireGraphBridge {
    bridge: GraphConsensusBridge,
    baseline: BandEnergy,
    shared: Arc<Mutex<Vec<DesireGraphEvent>>>,
}

impl DesireGraphBridge {
    pub fn new(bridge: GraphConsensusBridge, baseline: BandEnergy) -> Self {
        Self {
            bridge,
            baseline,
            shared: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn baseline(&self) -> BandEnergy {
        self.baseline
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

    pub fn drain(&self) -> PureResult<Vec<DesireGraphEvent>> {
        let mut guard = self.shared.lock().map_err(|_| TensorError::InvalidValue {
            label: "desire graph bridge poisoned",
        })?;
        Ok(std::mem::take(&mut *guard))
    }

    pub fn drain_summary(&self) -> PureResult<Option<DesireGraphSummary>> {
        let events = self.drain()?;
        if events.is_empty() {
            return Ok(None);
        }
        Ok(Some(DesireGraphSummary::from_events(&events)))
    }
}

impl DesirePipelineSink for DesireGraphBridge {
    fn on_step(&mut self, step: &DesireAutomatedStep, timestamp: SystemTime) -> PureResult<()> {
        let digest = self.bridge.digest(&self.baseline)?;
        let mut guard = self.shared.lock().map_err(|_| TensorError::InvalidValue {
            label: "desire graph bridge poisoned",
        })?;
        guard.push(DesireGraphEvent {
            timestamp,
            solution: step.solution.clone(),
            trigger: step.trigger.clone(),
            digest,
        });
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct DesireGraphSummary {
    pub steps: usize,
    pub triggers: usize,
    pub total_graph_energy: f32,
    pub mean_entropy: f32,
    pub layer_support: Vec<(String, f32)>,
    pub last_timestamp: SystemTime,
}

impl DesireGraphSummary {
    pub fn from_events(events: &[DesireGraphEvent]) -> Self {
        let mut total_entropy = 0.0f32;
        let mut total_graph_energy = 0.0f32;
        let mut triggers = 0usize;
        let mut digest_count = 0usize;
        let mut layer_accumulator: HashMap<String, f32> = HashMap::new();
        let mut last_timestamp = UNIX_EPOCH;

        for event in events {
            total_entropy += event.solution.entropy;
            if event.trigger.is_some() {
                triggers += 1;
            }
            if event.timestamp > last_timestamp {
                last_timestamp = event.timestamp;
            }
            if let Some(digest) = &event.digest {
                digest_count = digest_count.saturating_add(1);
                total_graph_energy += digest.graph_energy;
                for (layer, share) in &digest.layer_shares {
                    *layer_accumulator.entry(layer.clone()).or_insert(0.0) += *share;
                }
            }
        }

        if digest_count > 0 {
            for value in layer_accumulator.values_mut() {
                *value /= digest_count as f32;
            }
        }

        let mut layer_support: Vec<(String, f32)> = layer_accumulator.into_iter().collect();
        layer_support
            .sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));

        let steps = events.len();
        let mean_entropy = if steps == 0 {
            0.0
        } else {
            total_entropy / steps as f32
        };

        Self {
            steps,
            triggers,
            total_graph_energy,
            mean_entropy,
            layer_support,
            last_timestamp,
        }
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
    use crate::gnn::spiralk::GraphConsensusBridge;
    use crate::schedule::BandEnergy;
    use st_core::config::self_rewrite::SelfRewriteCfg;
    use st_core::telemetry::xai::{GraphFlowTracer, NodeFlowSample};
    use std::collections::HashSet;
    use std::sync::{mpsc::channel, Arc, Mutex};
    use std::time::{Duration, Instant, SystemTime};
    use tempfile::tempdir;

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

    #[test]
    fn graph_bridge_collects_consensus() {
        let automation = build_automation();
        let tracer = Arc::new(Mutex::new(GraphFlowTracer::new()));
        let baseline = BandEnergy {
            above: 0.4,
            here: 0.35,
            beneath: 0.25,
            drift: 0.0,
        };
        let bridge = DesireGraphBridge::new(GraphConsensusBridge::new(tracer.clone()), baseline);
        let mut pipeline = DesirePipeline::builder(automation)
            .with_graph_bridge(&bridge)
            .build();

        let logits = vec![2.0, 0.8];
        let concept = ConceptHint::Distribution(vec![0.65, 0.35]);
        let start = Instant::now();
        let anchor = SystemTime::now();
        for step in 0..3 {
            {
                let mut inner = tracer.lock().unwrap();
                inner.begin_layer(
                    format!("layer-{}", step),
                    -1.0,
                    vec![NodeFlowSample {
                        node_index: 0,
                        incoming_weight: 1.0 + step as f32 * 0.1,
                        aggregated_norm: 0.5 + 0.1 * step as f32,
                    }],
                );
            }
            let now = start + Duration::from_secs(step as u64);
            let timestamp = anchor + Duration::from_secs(step as u64);
            pipeline
                .step_at(&logits, 0, &concept, now, timestamp)
                .unwrap();
        }

        assert_eq!(bridge.len(), 3);
        let summary = bridge.drain_summary().unwrap().unwrap();
        assert_eq!(summary.steps, 3);
        assert!(summary.total_graph_energy > 0.0);
        assert!(summary.mean_entropy.is_finite());
        assert!(!summary.layer_support.is_empty());
        assert!(bridge.drain_summary().unwrap().is_none());
    }
}
