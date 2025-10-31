//! Robotics utilities for SpiralTorch.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::time::SystemTime;

#[derive(Debug)]
pub enum RoboticsError {
    ChannelExists(String),
    ChannelMissing(String),
    DimensionMismatch {
        channel: String,
        expected: usize,
        actual: usize,
    },
    BiasLengthMismatch {
        channel: String,
        expected: usize,
        actual: usize,
    },
    InvalidSmoothing {
        channel: String,
        smoothing: f32,
    },
}

impl fmt::Display for RoboticsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChannelExists(name) => write!(f, "sensor channel '{name}' already registered"),
            Self::ChannelMissing(name) => write!(f, "sensor channel '{name}' is not registered"),
            Self::DimensionMismatch {
                channel,
                expected,
                actual,
            } => write!(
                f,
                "payload for channel '{channel}' must contain {expected} values (got {actual})",
            ),
            Self::BiasLengthMismatch {
                channel,
                expected,
                actual,
            } => write!(
                f,
                "bias for channel '{channel}' must contain {expected} values (got {actual})",
            ),
            Self::InvalidSmoothing { channel, smoothing } => write!(
                f,
                "smoothing factor for channel '{channel}' must be between 0.0 and 1.0 (got {smoothing})",
            ),
        }
    }
}

impl std::error::Error for RoboticsError {}

#[derive(Debug, Clone)]
struct SensorChannel {
    dimension: usize,
    bias: Vec<f32>,
    scale: f32,
    smoothing: f32,
    last_filtered: Vec<f32>,
}

impl SensorChannel {
    fn new(dimension: usize) -> Self {
        Self {
            dimension,
            bias: vec![0.0; dimension],
            scale: 1.0,
            smoothing: 0.0,
            last_filtered: Vec::new(),
        }
    }

    fn apply(&mut self, values: &[f32]) -> Vec<f32> {
        let mut filtered = Vec::with_capacity(values.len());
        let smoothing = self.smoothing;
        for (idx, value) in values.iter().enumerate() {
            let bias = self.bias.get(idx).copied().unwrap_or(0.0);
            let adjusted = (value - bias) * self.scale;
            let output =
                if smoothing > 0.0 && smoothing < 1.0 && self.last_filtered.len() == self.dimension
                {
                    let previous = self.last_filtered[idx];
                    previous + smoothing * (adjusted - previous)
                } else {
                    adjusted
                };
            filtered.push(output);
        }
        self.last_filtered = filtered.clone();
        filtered
    }
}

#[derive(Debug, Clone)]
pub struct FusedFrame {
    pub coordinates: HashMap<String, Vec<f32>>,
    pub timestamp: SystemTime,
}

impl FusedFrame {
    pub fn norm(&self, channel: &str) -> Option<f32> {
        self.coordinates
            .get(channel)
            .map(|values| values.iter().map(|value| value * value).sum::<f32>().sqrt())
    }
}

#[derive(Debug, Default, Clone)]
pub struct SensorFusionHub {
    channels: HashMap<String, SensorChannel>,
}

impl SensorFusionHub {
    pub fn new() -> Self {
        Self {
            channels: HashMap::new(),
        }
    }

    pub fn register_channel(
        &mut self,
        name: impl Into<String>,
        dimension: usize,
    ) -> Result<(), RoboticsError> {
        let name = name.into();
        if self.channels.contains_key(&name) {
            return Err(RoboticsError::ChannelExists(name));
        }
        self.channels.insert(name, SensorChannel::new(dimension));
        Ok(())
    }

    pub fn calibrate(
        &mut self,
        name: &str,
        bias: Option<Vec<f32>>,
        scale: Option<f32>,
        smoothing: Option<f32>,
    ) -> Result<(), RoboticsError> {
        let channel = self
            .channels
            .get_mut(name)
            .ok_or_else(|| RoboticsError::ChannelMissing(name.to_string()))?;
        if let Some(bias) = bias {
            if bias.len() != channel.dimension {
                return Err(RoboticsError::BiasLengthMismatch {
                    channel: name.to_string(),
                    expected: channel.dimension,
                    actual: bias.len(),
                });
            }
            channel.bias = bias;
            channel.last_filtered.clear();
        }
        if let Some(scale) = scale {
            channel.scale = scale;
            channel.last_filtered.clear();
        }
        if let Some(smoothing) = smoothing {
            if !(0.0..=1.0).contains(&smoothing) {
                return Err(RoboticsError::InvalidSmoothing {
                    channel: name.to_string(),
                    smoothing,
                });
            }
            channel.smoothing = smoothing;
        }
        Ok(())
    }

    pub fn fuse(
        &mut self,
        payloads: &HashMap<String, Vec<f32>>,
    ) -> Result<FusedFrame, RoboticsError> {
        let mut coordinates = HashMap::with_capacity(payloads.len());
        for (name, values) in payloads {
            let channel = self
                .channels
                .get_mut(name)
                .ok_or_else(|| RoboticsError::ChannelMissing(name.clone()))?;
            if values.len() != channel.dimension {
                return Err(RoboticsError::DimensionMismatch {
                    channel: name.clone(),
                    expected: channel.dimension,
                    actual: values.len(),
                });
            }
            coordinates.insert(name.clone(), channel.apply(values));
        }
        Ok(FusedFrame {
            coordinates,
            timestamp: SystemTime::now(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct Desire {
    pub target_norm: f32,
    pub tolerance: f32,
    pub weight: f32,
}

impl Desire {
    pub fn energy(&self, norm: f32) -> f32 {
        let delta = (norm - self.target_norm).abs() - self.tolerance;
        let penalty = delta.max(0.0);
        penalty * self.weight
    }
}

#[derive(Debug, Clone)]
pub struct EnergyReport {
    pub total: f32,
    pub per_channel: HashMap<String, f32>,
}

impl EnergyReport {
    pub fn zero() -> Self {
        Self {
            total: 0.0,
            per_channel: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct DesireLagrangianField {
    desires: HashMap<String, Desire>,
}

impl DesireLagrangianField {
    pub fn new(desires: HashMap<String, Desire>) -> Self {
        Self { desires }
    }

    pub fn energy(&self, frame: &FusedFrame) -> EnergyReport {
        let mut report = EnergyReport::zero();
        for (name, desire) in &self.desires {
            if let Some(norm) = frame.norm(name) {
                let channel_energy = desire.energy(norm);
                report.total += channel_energy;
                report.per_channel.insert(name.clone(), channel_energy);
            }
        }
        report
    }
}

#[derive(Debug, Clone)]
pub struct TelemetryReport {
    pub energy: f32,
    pub stability: f32,
    pub failsafe: bool,
    pub anomalies: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PsiTelemetry {
    window: usize,
    stability_threshold: f32,
    failure_energy: f32,
    norm_limit: f32,
    history: VecDeque<f32>,
}

impl PsiTelemetry {
    pub fn new(
        window: usize,
        stability_threshold: f32,
        failure_energy: f32,
        norm_limit: f32,
    ) -> Self {
        Self {
            window: window.max(1),
            stability_threshold,
            failure_energy,
            norm_limit,
            history: VecDeque::with_capacity(window.max(1)),
        }
    }

    pub fn observe(&mut self, frame: &FusedFrame, energy: &EnergyReport) -> TelemetryReport {
        if self.history.len() == self.window {
            self.history.pop_front();
        }
        self.history.push_back(energy.total);

        let stability = if self.history.len() < 2 {
            1.0
        } else {
            let mean = self.history.iter().copied().sum::<f32>() / self.history.len() as f32;
            let variance = self
                .history
                .iter()
                .map(|value| {
                    let diff = value - mean;
                    diff * diff
                })
                .sum::<f32>()
                / self.history.len() as f32;
            let deviation = variance.sqrt();
            1.0 / (1.0 + deviation)
        };

        let mut anomalies = Vec::new();
        if stability < self.stability_threshold {
            anomalies.push("instability".to_string());
        }
        if energy.total > self.failure_energy {
            anomalies.push("energy_overflow".to_string());
        }
        for values in frame.coordinates.values() {
            let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
            if norm > self.norm_limit {
                anomalies.push("norm_overflow".to_string());
            }
        }
        anomalies.sort();
        anomalies.dedup();

        let failsafe = anomalies.iter().any(|tag| tag.starts_with("norm_overflow"))
            || energy.total > self.failure_energy;

        TelemetryReport {
            energy: energy.total,
            stability,
            failsafe,
            anomalies,
        }
    }
}

impl Default for PsiTelemetry {
    fn default() -> Self {
        Self::new(8, 0.5, 5.0, 10.0)
    }
}

#[derive(Debug, Clone)]
pub struct PolicyGradientController {
    base_learning_rate: f32,
    smoothing: f32,
    gauge: f32,
}

impl PolicyGradientController {
    pub fn new(base_learning_rate: f32, smoothing: f32) -> Self {
        Self {
            base_learning_rate,
            smoothing,
            gauge: 0.0,
        }
    }

    pub fn update(
        &mut self,
        energy: &EnergyReport,
        telemetry: &TelemetryReport,
    ) -> HashMap<String, f32> {
        let effective = self.base_learning_rate / (1.0 + energy.total.max(0.0));
        self.gauge = self.gauge * self.smoothing + telemetry.stability * (1.0 - self.smoothing);
        let mut commands = HashMap::new();
        commands.insert("learning_rate".to_string(), effective);
        commands.insert("gauge".to_string(), self.gauge);
        commands
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeStep {
    pub frame: FusedFrame,
    pub energy: EnergyReport,
    pub telemetry: TelemetryReport,
    pub commands: HashMap<String, f32>,
    pub halted: bool,
}

pub struct RoboticsRuntime {
    sensors: SensorFusionHub,
    desires: DesireLagrangianField,
    telemetry: PsiTelemetry,
    policy: Option<PolicyGradientController>,
}

impl RoboticsRuntime {
    pub fn new(
        sensors: SensorFusionHub,
        desires: DesireLagrangianField,
        telemetry: PsiTelemetry,
    ) -> Self {
        Self {
            sensors,
            desires,
            telemetry,
            policy: None,
        }
    }

    pub fn attach_policy_gradient(&mut self, controller: PolicyGradientController) {
        self.policy = Some(controller);
    }

    pub fn step(
        &mut self,
        payloads: HashMap<String, Vec<f32>>,
    ) -> Result<RuntimeStep, RoboticsError> {
        let frame = self.sensors.fuse(&payloads)?;
        let energy = self.desires.energy(&frame);
        let report = self.telemetry.observe(&frame, &energy);
        let mut commands = HashMap::new();
        if let Some(policy) = &mut self.policy {
            commands.extend(policy.update(&energy, &report));
        }
        let halted = report.failsafe;
        if halted {
            commands.insert("halt".to_string(), 1.0);
        }
        Ok(RuntimeStep {
            frame,
            energy,
            telemetry: report,
            commands,
            halted,
        })
    }

    pub fn sensors(&self) -> &SensorFusionHub {
        &self.sensors
    }

    pub fn desires(&self) -> &DesireLagrangianField {
        &self.desires
    }

    pub fn telemetry(&self) -> &PsiTelemetry {
        &self.telemetry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensor_fusion_applies_calibration() {
        let mut hub = SensorFusionHub::new();
        hub.register_channel("imu", 3).unwrap();
        hub.calibrate("imu", Some(vec![0.1, 0.1, 0.1]), Some(2.0), None)
            .unwrap();
        let frame = hub
            .fuse(&HashMap::from([("imu".to_string(), vec![0.2, 0.4, 0.3])]))
            .unwrap();
        let imu = frame.coordinates.get("imu").unwrap();
        assert_eq!(imu.len(), 3);
        assert!((imu[0] - 0.2).abs() < 1e-6);
        assert!((imu[1] - 0.6).abs() < 1e-6);
        assert!((imu[2] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn sensor_fusion_respects_smoothing() {
        let mut hub = SensorFusionHub::new();
        hub.register_channel("imu", 1).unwrap();
        hub.calibrate("imu", None, None, Some(0.5)).unwrap();

        let first = hub
            .fuse(&HashMap::from([("imu".to_string(), vec![1.0])]))
            .unwrap();
        assert!((first.coordinates["imu"][0] - 1.0).abs() < 1e-6);

        let second = hub
            .fuse(&HashMap::from([("imu".to_string(), vec![0.0])]))
            .unwrap();
        assert!(second.coordinates["imu"][0] > 0.0);
        assert!(second.coordinates["imu"][0] < 1.0);
    }

    #[test]
    fn calibrate_rejects_invalid_smoothing() {
        let mut hub = SensorFusionHub::new();
        hub.register_channel("imu", 1).unwrap();
        let err = hub
            .calibrate("imu", None, None, Some(1.5))
            .expect_err("smoothing above 1.0 should be rejected");
        assert!(matches!(err, RoboticsError::InvalidSmoothing { .. }));
        let err = hub
            .calibrate("imu", None, None, Some(-0.1))
            .expect_err("negative smoothing should be rejected");
        assert!(matches!(err, RoboticsError::InvalidSmoothing { .. }));
    }

    #[test]
    fn desire_field_accumulates_energy() {
        let mut hub = SensorFusionHub::new();
        hub.register_channel("pose", 1).unwrap();
        let frame = hub
            .fuse(&HashMap::from([("pose".to_string(), vec![0.7])]))
            .unwrap();
        let mut desires = HashMap::new();
        desires.insert(
            "pose".to_string(),
            Desire {
                target_norm: 0.2,
                tolerance: 0.05,
                weight: 2.0,
            },
        );
        let field = DesireLagrangianField::new(desires);
        let energy = field.energy(&frame);
        assert!(energy.total > 0.0);
        assert!(energy.per_channel.get("pose").unwrap() > &0.0);
    }

    #[test]
    fn telemetry_detects_overflow() {
        let mut hub = SensorFusionHub::new();
        hub.register_channel("pose", 2).unwrap();
        let mut desires = HashMap::new();
        desires.insert(
            "pose".to_string(),
            Desire {
                target_norm: 0.0,
                tolerance: 0.0,
                weight: 1.0,
            },
        );
        let field = DesireLagrangianField::new(desires);
        let mut telemetry = PsiTelemetry::new(4, 0.8, 10.0, 1.0);
        let frame = hub
            .fuse(&HashMap::from([("pose".to_string(), vec![2.0, 0.0])]))
            .unwrap();
        let energy = field.energy(&frame);
        let report = telemetry.observe(&frame, &energy);
        assert!(report.failsafe);
        assert!(report
            .anomalies
            .iter()
            .any(|tag| tag.contains("norm_overflow")));
    }

    #[test]
    fn runtime_emits_policy_commands() {
        let mut hub = SensorFusionHub::new();
        hub.register_channel("imu", 3).unwrap();
        let desires = DesireLagrangianField::new(HashMap::from([(
            "imu".to_string(),
            Desire {
                target_norm: 0.0,
                tolerance: 0.0,
                weight: 1.0,
            },
        )]));
        let telemetry = PsiTelemetry::default();
        let mut runtime = RoboticsRuntime::new(hub, desires, telemetry);
        runtime.attach_policy_gradient(PolicyGradientController::new(0.05, 0.5));
        let step = runtime
            .step(HashMap::from([("imu".to_string(), vec![0.1, -0.2, 0.05])]))
            .unwrap();
        assert!(!step.halted);
        assert!(step.commands.contains_key("learning_rate"));
        assert!(step.commands.contains_key("gauge"));
    }
}
