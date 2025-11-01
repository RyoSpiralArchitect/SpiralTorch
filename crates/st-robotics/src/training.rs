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

/// Partial observation emitted to Python for integration with ZSpacePartialBundle.
#[derive(Debug, Clone)]
pub struct ZSpacePartialObservation {
    pub metrics: HashMap<String, f32>,
    pub commands: HashMap<String, f32>,
    pub gradient: Vec<f32>,
    pub weight: f32,
}
