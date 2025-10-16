// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::automation::{DesireAutomatedStep, DesireAutomation, DesireRewriteTrigger};
use super::desire::DesireWeights;
use super::geometry::ConceptHint;
use super::logbook::{DesireLogReplay, DesireLogbook};
use crate::PureResult;
use st_tensor::pure::TensorError;
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

#[cfg(test)]
mod tests {
    use super::super::automation::DesireAutomation;
    use super::super::desire::{constant, warmup, DesireLagrangian};
    use super::super::geometry::{
        ConceptHint, RepressionField, SemanticBridge, SparseKernel, SymbolGeometry,
    };
    use super::super::temperature::TemperatureController;
    use super::*;
    use st_core::config::self_rewrite::SelfRewriteCfg;
    use std::collections::HashSet;
    use std::sync::mpsc::channel;
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
}
