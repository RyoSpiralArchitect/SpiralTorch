use std::collections::{HashMap, VecDeque};
use std::time::SystemTime;

use crate::error::RoboticsError;
use crate::runtime::RuntimeStep;
use crate::vision::VisionFeedbackSnapshot;

/// Snapshot representing energetic and psi telemetry over a runtime step.
#[derive(Debug, Clone)]
pub struct TemporalFeedbackSample {
    pub timestamp: SystemTime,
    pub energy_total: f32,
    pub gravitational_total: f32,
    pub psi_stability: f32,
    pub psi_failsafe: bool,
    pub psi_anomalies: Vec<String>,
    pub per_channel_energy: HashMap<String, f32>,
    pub per_channel_gravity: HashMap<String, f32>,
    pub channel_norms: HashMap<String, f32>,
    pub average_norm: f32,
    pub commands: HashMap<String, f32>,
    pub halted: bool,
    pub vision_energy: Option<f32>,
    pub vision_rms: Option<f32>,
    pub vision_alignment: Option<f32>,
}

impl TemporalFeedbackSample {
    pub fn from_step(
        step: &RuntimeStep,
        vision: Option<&VisionFeedbackSnapshot>,
    ) -> TemporalFeedbackSample {
        let mut channel_norms = HashMap::new();
        for (name, values) in &step.frame.coordinates {
            let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
            channel_norms.insert(name.clone(), norm);
        }
        let average_norm = if channel_norms.is_empty() {
            0.0
        } else {
            channel_norms.values().sum::<f32>() / channel_norms.len() as f32
        };
        TemporalFeedbackSample {
            timestamp: step.frame.timestamp,
            energy_total: step.energy.total,
            gravitational_total: step.energy.gravitational,
            psi_stability: step.telemetry.stability,
            psi_failsafe: step.telemetry.failsafe,
            psi_anomalies: step.telemetry.anomalies.clone(),
            per_channel_energy: step.energy.per_channel.clone(),
            per_channel_gravity: step.energy.gravitational_per_channel.clone(),
            channel_norms,
            average_norm,
            commands: step.commands.clone(),
            halted: step.halted,
            vision_energy: vision.map(|snap| snap.canvas_mean_energy),
            vision_rms: vision.map(|snap| snap.canvas_rms_energy),
            vision_alignment: vision.map(|snap| snap.alignment),
        }
    }

    pub fn anomaly_score(&self) -> f32 {
        let mut score = if self.psi_failsafe { 1.0 } else { 0.0 };
        score += self.psi_anomalies.len() as f32;
        if self.halted {
            score += 1.0;
        }
        score
    }

    fn gradient(&self) -> Vec<f32> {
        let mut entries: Vec<(String, f32)> = self
            .per_channel_energy
            .iter()
            .map(|(name, value)| {
                let gravity = self.per_channel_gravity.get(name).copied().unwrap_or(0.0);
                (name.clone(), value - gravity)
            })
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        let mut gradient: Vec<f32> = entries.into_iter().map(|(_, delta)| delta).collect();
        if let Some(alignment) = self.vision_alignment {
            gradient.push(alignment);
        }
        if let Some(rms) = self.vision_rms {
            gradient.push(rms);
        }
        gradient
    }

    pub fn to_partial(&self) -> ZSpacePartialObservation {
        let mut metrics = HashMap::new();
        metrics.insert("psi.energy".to_string(), self.energy_total);
        metrics.insert("psi.stability".to_string(), self.psi_stability);
        metrics.insert(
            "psi.failsafe".to_string(),
            if self.psi_failsafe { 1.0 } else { 0.0 },
        );
        metrics.insert("psi.anomalies".to_string(), self.psi_anomalies.len() as f32);
        metrics.insert("desire.energy.total".to_string(), self.energy_total);
        metrics.insert("gravity.energy.total".to_string(), self.gravitational_total);
        metrics.insert("sensor.norm.mean".to_string(), self.average_norm);
        for (name, energy) in &self.per_channel_energy {
            metrics.insert(format!("desire.energy.{name}"), *energy);
        }
        for (name, gravity) in &self.per_channel_gravity {
            metrics.insert(format!("gravity.energy.{name}"), *gravity);
        }
        for (name, norm) in &self.channel_norms {
            metrics.insert(format!("sensor.norm.{name}"), *norm);
        }
        if let Some(energy) = self.vision_energy {
            metrics.insert("vision.energy.mean".to_string(), energy);
        }
        if let Some(rms) = self.vision_rms {
            metrics.insert("vision.energy.rms".to_string(), rms);
        }
        if let Some(alignment) = self.vision_alignment {
            metrics.insert("vision.alignment".to_string(), alignment);
        }
        ZSpacePartialObservation {
            metrics,
            commands: self.commands.clone(),
            gradient: self.gradient(),
            weight: 1.0,
        }
    }
}

/// Aggregated view of temporal feedback for ZSpaceTrainer.
#[derive(Debug, Clone)]
pub struct TemporalFeedbackSummary {
    pub discounted_energy: f32,
    pub discounted_gravity: f32,
    pub discounted_stability: f32,
    pub discounted_alignment: Option<f32>,
    pub commands: HashMap<String, f32>,
    pub partial: ZSpacePartialObservation,
    pub latest_sensor_norm: f32,
    pub anomaly_score: f32,
}

/// Sliding window aggregator for time-series control feedback.
#[derive(Debug, Clone)]
pub struct TemporalFeedbackLearner {
    horizon: usize,
    discount: f32,
    buffer: VecDeque<TemporalFeedbackSample>,
}

impl TemporalFeedbackLearner {
    pub fn new(horizon: usize, discount: f32) -> Result<Self, RoboticsError> {
        if horizon == 0 {
            return Err(RoboticsError::Trainer(
                "temporal horizon must be positive".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&discount) || discount == 0.0 {
            return Err(RoboticsError::Trainer(
                "discount factor must be in the range (0, 1]".to_string(),
            ));
        }
        Ok(Self {
            horizon,
            discount,
            buffer: VecDeque::with_capacity(horizon),
        })
    }

    pub fn push(
        &mut self,
        step: &RuntimeStep,
        vision: Option<&VisionFeedbackSnapshot>,
    ) -> TemporalFeedbackSummary {
        let sample = TemporalFeedbackSample::from_step(step, vision);
        if self.buffer.len() == self.horizon {
            self.buffer.pop_front();
        }
        self.buffer.push_back(sample.clone());

        let mut discounted_energy = 0.0;
        let mut discounted_gravity = 0.0;
        let mut discounted_stability = 0.0;
        let mut discounted_alignment = 0.0;
        let mut alignment_weight = 0.0;
        let mut commands = HashMap::new();
        let mut weight = 0.0;
        let mut factor = 1.0;
        for snapshot in self.buffer.iter().rev() {
            discounted_energy += factor * snapshot.energy_total;
            discounted_gravity += factor * snapshot.gravitational_total;
            discounted_stability += factor * snapshot.psi_stability;
            if let Some(alignment) = snapshot.vision_alignment {
                discounted_alignment += factor * alignment;
                alignment_weight += factor;
            }
            for (name, value) in &snapshot.commands {
                *commands.entry(name.clone()).or_insert(0.0) += factor * *value;
            }
            weight += factor;
            factor *= self.discount;
        }
        if weight > 0.0 {
            discounted_energy /= weight;
            discounted_gravity /= weight;
            discounted_stability /= weight;
            for value in commands.values_mut() {
                *value /= weight;
            }
        }
        let alignment_metric = if alignment_weight > 0.0 {
            Some(discounted_alignment / alignment_weight)
        } else {
            None
        };

        let mut partial = sample.to_partial();
        partial.weight = weight.max(1.0);
        partial
            .metrics
            .insert("feedback.energy.discounted".to_string(), discounted_energy);
        partial.metrics.insert(
            "feedback.gravity.discounted".to_string(),
            discounted_gravity,
        );
        partial.metrics.insert(
            "feedback.stability.discounted".to_string(),
            discounted_stability,
        );
        if let Some(alignment) = alignment_metric {
            partial
                .metrics
                .insert("vision.alignment.discounted".to_string(), alignment);
        }
        for (name, value) in &commands {
            partial
                .metrics
                .insert(format!("feedback.command.{name}"), *value);
        }

        TemporalFeedbackSummary {
            discounted_energy,
            discounted_gravity,
            discounted_stability,
            discounted_alignment: alignment_metric,
            commands,
            partial,
            latest_sensor_norm: sample.average_norm,
            anomaly_score: sample.anomaly_score(),
        }
    }

    pub fn horizon(&self) -> usize {
        self.horizon
    }

    pub fn discount(&self) -> f32 {
        self.discount
    }
}

/// Trainer-facing metrics derived from robotics feedback.
#[derive(Debug, Clone)]
pub struct TrainerMetrics {
    pub speed: f32,
    pub memory: f32,
    pub stability: f32,
    pub gradient: Vec<f32>,
    pub drs: f32,
}

/// Bundle returned to Python bindings to feed ZSpaceTrainer.
#[derive(Debug, Clone)]
pub struct ZSpaceTrainerSample {
    pub metrics: TrainerMetrics,
    pub partial: ZSpacePartialObservation,
}

/// Helper bridging robotics runtime feedback into ZSpaceTrainer inputs.
#[derive(Debug, Clone)]
pub struct ZSpaceTrainerBridge {
    learner: TemporalFeedbackLearner,
}

impl ZSpaceTrainerBridge {
    pub fn new(horizon: usize, discount: f32) -> Result<Self, RoboticsError> {
        Ok(Self {
            learner: TemporalFeedbackLearner::new(horizon, discount)?,
        })
    }

    pub fn push(
        &mut self,
        step: &RuntimeStep,
        vision: Option<&VisionFeedbackSnapshot>,
    ) -> Result<ZSpaceTrainerSample, RoboticsError> {
        let summary = self.learner.push(step, vision);
        let mut gradient = summary.partial.gradient.clone();
        if gradient.is_empty() {
            gradient.push(summary.discounted_energy - summary.discounted_gravity);
        }
        let metrics = TrainerMetrics {
            speed: summary.latest_sensor_norm,
            memory: summary.discounted_energy + summary.discounted_gravity,
            stability: summary.discounted_stability,
            gradient,
            drs: summary.anomaly_score,
        };
        Ok(ZSpaceTrainerSample {
            metrics,
            partial: summary.partial,
        })
    }

    pub fn horizon(&self) -> usize {
        self.learner.horizon()
    }

    pub fn discount(&self) -> f32 {
        self.learner.discount()
    }
}

/// Aggregated trainer samples representing a single robotics episode.
#[derive(Debug, Clone)]
pub struct TrainerEpisode {
    pub samples: Vec<ZSpaceTrainerSample>,
    pub average_memory: f32,
    pub average_stability: f32,
    pub average_drs: f32,
    pub length: usize,
}

/// Collects trainer samples across an episode and exposes summary metrics.
#[derive(Debug, Clone)]
pub struct ZSpaceTrainerEpisodeBuilder {
    bridge: ZSpaceTrainerBridge,
    capacity: usize,
    buffer: Vec<ZSpaceTrainerSample>,
    memory_sum: f32,
    stability_sum: f32,
    drs_sum: f32,
}

impl ZSpaceTrainerEpisodeBuilder {
    pub fn new(horizon: usize, discount: f32, capacity: usize) -> Result<Self, RoboticsError> {
        if capacity == 0 {
            return Err(RoboticsError::Trainer(
                "episode capacity must be positive".to_string(),
            ));
        }
        Ok(Self {
            bridge: ZSpaceTrainerBridge::new(horizon, discount)?,
            capacity,
            buffer: Vec::with_capacity(capacity.min(8)),
            memory_sum: 0.0,
            stability_sum: 0.0,
            drs_sum: 0.0,
        })
    }

    pub fn push_step(
        &mut self,
        step: &RuntimeStep,
        vision: Option<&VisionFeedbackSnapshot>,
        end_episode: bool,
    ) -> Result<Option<TrainerEpisode>, RoboticsError> {
        if self.buffer.len() == self.capacity {
            return Err(RoboticsError::Trainer(
                "episode capacity exceeded before flush".to_string(),
            ));
        }
        let sample = self.bridge.push(step, vision)?;
        self.memory_sum += sample.metrics.memory;
        self.stability_sum += sample.metrics.stability;
        self.drs_sum += sample.metrics.drs;
        self.buffer.push(sample);
        if end_episode {
            Ok(Some(self.finish_episode()))
        } else {
            Ok(None)
        }
    }

    pub fn flush(&mut self) -> Option<TrainerEpisode> {
        if self.buffer.is_empty() {
            None
        } else {
            Some(self.finish_episode())
        }
    }

    pub fn horizon(&self) -> usize {
        self.bridge.horizon()
    }

    pub fn discount(&self) -> f32 {
        self.bridge.discount()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    fn finish_episode(&mut self) -> TrainerEpisode {
        let samples = std::mem::take(&mut self.buffer);
        let length = samples.len();
        let normaliser = length.max(1) as f32;
        let episode = TrainerEpisode {
            samples,
            average_memory: self.memory_sum / normaliser,
            average_stability: self.stability_sum / normaliser,
            average_drs: self.drs_sum / normaliser,
            length,
        };
        self.memory_sum = 0.0;
        self.stability_sum = 0.0;
        self.drs_sum = 0.0;
        episode
    }
}

/// Partial observation emitted to Python for integration with ZSpacePartialBundle.
#[derive(Debug, Clone)]
pub struct ZSpacePartialObservation {
    pub metrics: HashMap<String, f32>,
    pub commands: HashMap<String, f32>,
    pub gradient: Vec<f32>,
    pub weight: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desire::Desire;
    use crate::runtime::RoboticsRuntime;
    use crate::sensors::SensorFusionHub;
    use crate::telemetry::PsiTelemetry;
    use crate::vision::VisionFeedbackSynchronizer;
    use crate::DesireLagrangianField;

    use std::collections::HashMap;

    #[test]
    fn episode_builder_flushes() {
        let mut hub = SensorFusionHub::new();
        hub.register_channel("vision", 3).unwrap();
        hub.register_channel("imu", 3).unwrap();
        let desires = DesireLagrangianField::new(HashMap::from([
            (
                "vision".to_string(),
                Desire {
                    target_norm: 0.0,
                    tolerance: 0.0,
                    weight: 0.5,
                },
            ),
            (
                "imu".to_string(),
                Desire {
                    target_norm: 0.0,
                    tolerance: 0.0,
                    weight: 1.0,
                },
            ),
        ]));
        let telemetry = PsiTelemetry::default();
        let mut runtime = RoboticsRuntime::new(hub, desires, telemetry);
        let mut builder = ZSpaceTrainerEpisodeBuilder::new(3, 0.9, 8).unwrap();
        let synchronizer = VisionFeedbackSynchronizer::new("vision".to_string());
        let step_a = runtime
            .step(HashMap::from([
                ("vision".to_string(), vec![0.2, 0.3, 0.4]),
                ("imu".to_string(), vec![0.1, -0.2, 0.05]),
            ]))
            .unwrap();
        let vision_a = synchronizer
            .sync_with_vectors(&step_a, &[[0.2, 0.0, 0.0, 0.0], [0.1, 0.02, 0.01, -0.02]])
            .ok();
        assert!(builder
            .push_step(&step_a, vision_a.as_ref(), false)
            .unwrap()
            .is_none());
        let step_b = runtime
            .step(HashMap::from([
                ("vision".to_string(), vec![0.1, 0.1, 0.2]),
                ("imu".to_string(), vec![0.05, 0.02, -0.01]),
            ]))
            .unwrap();
        let vision_b = synchronizer
            .sync_with_vectors(&step_b, &[[0.1, -0.01, 0.0, 0.01]])
            .ok();
        let episode = builder
            .push_step(&step_b, vision_b.as_ref(), true)
            .unwrap()
            .expect("episode to flush");
        assert_eq!(episode.length, 2);
        assert_eq!(episode.samples.len(), 2);
        assert!(episode.average_memory > 0.0);
        assert!(episode.average_stability >= 0.0);
        assert!(episode.average_drs >= 0.0);
        assert!(builder.flush().is_none());
    }

    #[test]
    fn episode_builder_enforces_capacity() {
        let mut hub = SensorFusionHub::new();
        hub.register_channel("vision", 3).unwrap();
        let desires = DesireLagrangianField::new(HashMap::new());
        let telemetry = PsiTelemetry::default();
        let mut runtime = RoboticsRuntime::new(hub, desires, telemetry);
        let mut builder = ZSpaceTrainerEpisodeBuilder::new(2, 0.9, 1).unwrap();
        let step = runtime
            .step(HashMap::from([("vision".to_string(), vec![0.1, 0.0, 0.0])]))
            .unwrap();
        builder.push_step(&step, None, false).unwrap();
        let err = builder
            .push_step(&step, None, false)
            .err()
            .expect("capacity violation");
        if let RoboticsError::Trainer(message) = err {
            assert!(message.contains("capacity"));
        } else {
            panic!("unexpected error kind");
        }
    }
}
