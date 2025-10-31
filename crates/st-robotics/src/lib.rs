// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::collections::{HashMap, VecDeque};

use thiserror::Error;

/// Convenience alias for channel identifiers.
pub type ChannelId = String;

/// Error emitted by robotics primitives when sensor metadata is invalid or
/// a runtime contract has been violated.
#[derive(Debug, Error)]
pub enum RoboticsError {
    /// Raised when a requested channel has not been registered.
    #[error("channel '{0}' is not registered")]
    MissingChannel(String),
    /// Raised when calibration metadata is incompatible with incoming
    /// measurements.
    #[error(
        "calibration for channel '{channel}' expected dimension {expected} but received {received}"
    )]
    CalibrationDimension {
        channel: String,
        expected: usize,
        received: usize,
    },
    /// Raised when a policy controller fails to return valid commands.
    #[error("policy controller failed: {0}")]
    Policy(String),
}

#[derive(Clone, Debug)]
struct ChannelConfig {
    bias: Vec<f64>,
    scale: f64,
    dimension: Option<usize>,
}

impl ChannelConfig {
    fn new() -> Self {
        Self {
            bias: Vec::new(),
            scale: 1.0,
            dimension: None,
        }
    }

    fn ensure_dimension(&mut self, dim: usize) {
        match self.dimension {
            Some(existing) if existing == dim => {}
            Some(existing) if existing != dim => {
                // Reset bias so it can be resized lazily.
                self.bias.clear();
                self.dimension = Some(dim);
            }
            None => {
                self.dimension = Some(dim);
            }
            _ => {}
        }
    }

    fn apply(&self, payload: &[f64]) -> Vec<f64> {
        let scale = self.scale;
        if self.bias.is_empty() {
            payload.iter().map(|value| value * scale).collect()
        } else {
            payload
                .iter()
                .enumerate()
                .map(|(idx, value)| {
                    let bias = self.bias.get(idx).copied().unwrap_or(0.0);
                    (value - bias) * scale
                })
                .collect()
        }
    }
}

/// Representation of a fused multi-modal robotics frame.
#[derive(Clone, Debug, Default)]
pub struct FusedFrame {
    coordinates: HashMap<ChannelId, Vec<f64>>,
}

impl FusedFrame {
    /// Constructs a frame from channel coordinates.
    pub fn new(coordinates: HashMap<ChannelId, Vec<f64>>) -> Self {
        Self { coordinates }
    }

    /// Returns the coordinate vector for the requested channel if it exists.
    pub fn channel(&self, name: &str) -> Option<&[f64]> {
        self.coordinates.get(name).map(|values| values.as_slice())
    }

    /// Returns an iterator over channel identifiers and coordinates.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &[f64])> {
        self.coordinates
            .iter()
            .map(|(name, values)| (name.as_str(), values.as_slice()))
    }

    /// Computes the Euclidean norm of the provided channel.
    pub fn norm(&self, name: &str) -> Option<f64> {
        self.channel(name)
            .map(|values| values.iter().map(|value| value * value).sum::<f64>().sqrt())
    }

    /// Returns the largest norm observed across all channels.
    pub fn max_norm(&self) -> f64 {
        self.coordinates
            .values()
            .map(|values| values.iter().map(|value| value * value).sum::<f64>().sqrt())
            .fold(0.0, f64::max)
    }

    /// Exposes raw coordinate storage for consumers that need ownership.
    pub fn into_coordinates(self) -> HashMap<ChannelId, Vec<f64>> {
        self.coordinates
    }
}

/// Hub responsible for ingesting raw sensor payloads and applying
/// calibrations before presenting a fused frame in Z-space coordinates.
#[derive(Clone, Debug, Default)]
pub struct SensorFusionHub {
    channels: HashMap<ChannelId, ChannelConfig>,
}

impl SensorFusionHub {
    /// Constructs an empty fusion hub.
    pub fn new() -> Self {
        Self {
            channels: HashMap::new(),
        }
    }

    /// Registers a new sensor channel. If the channel already exists the
    /// call is a no-op.
    pub fn register_channel(&mut self, name: impl Into<String>) {
        self.channels
            .entry(name.into())
            .or_insert_with(ChannelConfig::new);
    }

    /// Applies bias/scale calibration to the requested channel.
    pub fn calibrate(
        &mut self,
        name: &str,
        bias: impl Into<Vec<f64>>,
        scale: f64,
    ) -> Result<(), RoboticsError> {
        let entry = self
            .channels
            .get_mut(name)
            .ok_or_else(|| RoboticsError::MissingChannel(name.to_string()))?;
        let bias_vec = bias.into();
        if let Some(expected) = entry.dimension {
            if expected != bias_vec.len() && !bias_vec.is_empty() {
                return Err(RoboticsError::CalibrationDimension {
                    channel: name.to_string(),
                    expected,
                    received: bias_vec.len(),
                });
            }
        }
        entry.bias = bias_vec;
        entry.scale = scale;
        Ok(())
    }

    /// Fuses raw payloads into calibrated coordinates.
    pub fn fuse(
        &mut self,
        payloads: &HashMap<ChannelId, Vec<f64>>,
    ) -> Result<FusedFrame, RoboticsError> {
        let mut coordinates = HashMap::new();
        for (name, payload) in payloads {
            let config = self
                .channels
                .get_mut(name)
                .ok_or_else(|| RoboticsError::MissingChannel(name.clone()))?;
            config.ensure_dimension(payload.len());
            if !config.bias.is_empty() && config.bias.len() != payload.len() {
                return Err(RoboticsError::CalibrationDimension {
                    channel: name.clone(),
                    expected: config.bias.len(),
                    received: payload.len(),
                });
            }
            let calibrated = config.apply(payload);
            coordinates.insert(name.clone(), calibrated);
        }
        Ok(FusedFrame::new(coordinates))
    }
}

/// Desire encoded as a Lagrangian potential defined in Z-space.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Desire {
    pub target_norm: f64,
    pub tolerance: f64,
    pub weight: f64,
}

impl Desire {
    pub fn new(target_norm: f64, tolerance: f64, weight: f64) -> Self {
        Self {
            target_norm,
            tolerance: tolerance.max(0.0),
            weight: weight.max(0.0),
        }
    }
}

/// Aggregated energy per channel and the total scalar potential.
#[derive(Clone, Debug, Default)]
pub struct DesireEnergy {
    pub per_channel: HashMap<ChannelId, f64>,
    pub total: f64,
}

impl DesireEnergy {
    pub fn new(per_channel: HashMap<ChannelId, f64>) -> Self {
        let total = per_channel.values().copied().sum();
        Self { per_channel, total }
    }
}

/// Field encoding multiple instincts with channel-specific potentials.
#[derive(Clone, Debug, Default)]
pub struct DesireLagrangianField {
    desires: HashMap<ChannelId, Desire>,
}

impl DesireLagrangianField {
    pub fn new(desires: HashMap<ChannelId, Desire>) -> Self {
        Self { desires }
    }

    pub fn is_empty(&self) -> bool {
        self.desires.is_empty()
    }

    pub fn energy(&self, frame: &FusedFrame) -> DesireEnergy {
        let mut per_channel = HashMap::new();
        for (name, desire) in &self.desires {
            if let Some(values) = frame.channel(name) {
                let norm = values.iter().map(|v| v * v).sum::<f64>().sqrt();
                let deviation = (norm - desire.target_norm).abs();
                let slack = (deviation - desire.tolerance).max(0.0);
                let energy = slack * desire.weight;
                per_channel.insert(name.clone(), energy);
            }
        }
        DesireEnergy::new(per_channel)
    }
}

/// Telemetry report emitted after observing fused state and desire energy.
#[derive(Clone, Debug, Default)]
pub struct TelemetryReport {
    pub stability: f64,
    pub energy: f64,
    pub anomalies: Vec<String>,
    pub failsafe: bool,
}

/// ψ-telemetry controller that monitors stability windows and halting rules.
#[derive(Clone, Debug)]
pub struct PsiTelemetry {
    window: usize,
    stability_threshold: f64,
    failure_energy: f64,
    norm_limit: f64,
    history: VecDeque<f64>,
}

impl Default for PsiTelemetry {
    fn default() -> Self {
        Self {
            window: 8,
            stability_threshold: 0.5,
            failure_energy: 5.0,
            norm_limit: f64::INFINITY,
            history: VecDeque::with_capacity(8),
        }
    }
}

impl PsiTelemetry {
    pub fn new(
        window: usize,
        stability_threshold: f64,
        failure_energy: f64,
        norm_limit: f64,
    ) -> Self {
        let mut controller = Self::default();
        if window > 0 {
            controller.window = window;
            controller.history = VecDeque::with_capacity(window);
        }
        controller.stability_threshold = stability_threshold;
        controller.failure_energy = failure_energy;
        controller.norm_limit = norm_limit;
        controller
    }

    pub fn observe(&mut self, frame: &FusedFrame, energy: &DesireEnergy) -> TelemetryReport {
        let total = energy.total;
        if self.window == 0 {
            self.window = 1;
        }
        if self.history.len() == self.window {
            self.history.pop_front();
        }
        self.history.push_back(total);
        let len = self.history.len() as f64;
        let mean = self.history.iter().copied().sum::<f64>() / len.max(1.0);
        let variance = self
            .history
            .iter()
            .map(|value| {
                let diff = value - mean;
                diff * diff
            })
            .sum::<f64>()
            / len.max(1.0);
        let stability = 1.0 / (1.0 + variance);

        let mut anomalies = Vec::new();
        if stability < self.stability_threshold {
            anomalies.push("low_stability".to_string());
        }
        let max_norm = frame.max_norm();
        if max_norm > self.norm_limit {
            anomalies.push("norm_overflow".to_string());
        }
        if total > self.failure_energy {
            anomalies.push("energy_overflow".to_string());
        }
        let failsafe = stability < self.stability_threshold
            || total > self.failure_energy
            || max_norm > self.norm_limit;

        TelemetryReport {
            stability,
            energy: total,
            anomalies,
            failsafe,
        }
    }
}

/// Result returned after stepping the robotics runtime.
#[derive(Clone, Debug)]
pub struct RuntimeStep {
    pub frame: FusedFrame,
    pub energy: DesireEnergy,
    pub telemetry: TelemetryReport,
    pub commands: HashMap<String, f64>,
    pub halted: bool,
}

/// Trait describing a controllable policy hook capable of producing runtime
/// commands from desire energy statistics.
pub trait PolicyController {
    fn step(&mut self, returns: &[f64]) -> Result<HashMap<String, f64>, RoboticsError>;
}

impl<F> PolicyController for F
where
    F: FnMut(&[f64]) -> HashMap<String, f64>,
{
    fn step(&mut self, returns: &[f64]) -> Result<HashMap<String, f64>, RoboticsError> {
        Ok(self(returns))
    }
}

/// Runtime orchestrator wiring sensor fusion, desires, ψ-telemetry, and
/// optional policy control loops.
pub struct RoboticsRuntime {
    sensors: SensorFusionHub,
    desires: DesireLagrangianField,
    telemetry: PsiTelemetry,
    policy: Option<Box<dyn PolicyController + Send>>, // policy returns control commands
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

    pub fn sensors(&self) -> &SensorFusionHub {
        &self.sensors
    }

    pub fn sensors_mut(&mut self) -> &mut SensorFusionHub {
        &mut self.sensors
    }

    pub fn attach_policy(&mut self, policy: Box<dyn PolicyController + Send>) {
        self.policy = Some(policy);
    }

    pub fn clear_policy(&mut self) {
        self.policy = None;
    }

    pub fn step(
        &mut self,
        payloads: &HashMap<ChannelId, Vec<f64>>,
    ) -> Result<RuntimeStep, RoboticsError> {
        let frame = self.sensors.fuse(payloads)?;
        let energy = self.desires.energy(&frame);
        let report = self.telemetry.observe(&frame, &energy);
        let mut commands = HashMap::new();
        let halted = report.failsafe;
        if let Some(policy) = self.policy.as_mut() {
            let returns: Vec<f64> = if energy.per_channel.is_empty() {
                vec![0.0]
            } else {
                energy.per_channel.values().map(|value| -value).collect()
            };
            let update = policy.step(&returns)?;
            commands.extend(update);
        }
        if halted {
            commands.entry("halt".to_string()).or_insert(1.0);
        }
        Ok(RuntimeStep {
            frame,
            energy,
            telemetry: report,
            commands,
            halted,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_frame(values: &[(&str, &[f64])]) -> FusedFrame {
        let mut map = HashMap::new();
        for (name, coords) in values {
            map.insert((*name).to_string(), coords.to_vec());
        }
        FusedFrame::new(map)
    }

    #[test]
    fn fuse_applies_calibration() {
        let mut hub = SensorFusionHub::new();
        hub.register_channel("imu");
        hub.calibrate("imu", vec![0.1, 0.1, 0.1], 2.0).unwrap();

        let mut payloads = HashMap::new();
        payloads.insert("imu".to_string(), vec![0.2, 0.4, 0.3]);
        let frame = hub.fuse(&payloads).unwrap();

        let imu = frame.channel("imu").unwrap();
        assert_eq!(imu.len(), 3);
        assert!((imu[0] - 0.2).abs() < 1e-6);
        assert!((imu[1] - 0.6).abs() < 1e-6);
        assert!((imu[2] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn desire_energy_accumulates() {
        let frame = build_frame(&[("balance", &[0.7]), ("power", &[0.8])]);
        let field = DesireLagrangianField::new(HashMap::from([
            ("balance".to_string(), Desire::new(0.2, 0.05, 2.0)),
            ("power".to_string(), Desire::new(0.5, 0.0, 1.0)),
        ]));
        let energy = field.energy(&frame);
        assert!(energy.total > 0.0);
        assert!(energy.per_channel["balance"] > energy.per_channel["power"]);
    }

    #[test]
    fn telemetry_detects_instability() {
        let mut telemetry = PsiTelemetry::new(4, 0.8, 10.0, 2.0);
        let field = DesireLagrangianField::new(HashMap::from([(
            "pose".to_string(),
            Desire::new(0.0, 0.0, 1.0),
        )]));

        let frames = [
            build_frame(&[("pose", &[0.1, 0.1])]),
            build_frame(&[("pose", &[0.5, 0.5])]),
            build_frame(&[("pose", &[1.5, 1.5])]),
        ];
        let mut reports = Vec::new();
        for frame in &frames {
            let energy = field.energy(frame);
            reports.push(telemetry.observe(frame, &energy));
        }
        assert!(reports.iter().any(|report| report.failsafe));
        assert!(reports
            .iter()
            .any(|report| report.anomalies.iter().any(|tag| tag == "norm_overflow")));
    }

    #[test]
    fn runtime_emits_policy_commands() {
        let mut hub = SensorFusionHub::new();
        hub.register_channel("imu");
        let field = DesireLagrangianField::new(HashMap::from([(
            "imu".to_string(),
            Desire::new(0.0, 0.0, 1.0),
        )]));
        let mut runtime = RoboticsRuntime::new(hub, field, PsiTelemetry::default());
        runtime.attach_policy(Box::new(|returns: &[f64]| {
            let learning_rate = 1.0 + returns.iter().map(|value| value.abs()).sum::<f64>();
            HashMap::from([
                ("learning_rate".to_string(), learning_rate),
                ("gauge".to_string(), learning_rate * 0.5),
            ])
        }));

        let mut payloads = HashMap::new();
        payloads.insert("imu".to_string(), vec![0.1, -0.2, 0.05]);
        let step = runtime.step(&payloads).unwrap();
        assert!(!step.halted);
        assert!(step.commands.contains_key("learning_rate"));
    }

    #[test]
    fn runtime_halts_on_failsafe() {
        let mut hub = SensorFusionHub::new();
        hub.register_channel("imu");
        let field = DesireLagrangianField::new(HashMap::from([(
            "imu".to_string(),
            Desire::new(0.0, 0.0, 1.0),
        )]));
        let telemetry = PsiTelemetry::new(2, 0.2, 0.01, 0.2);
        let mut runtime = RoboticsRuntime::new(hub, field, telemetry);

        let mut payloads = HashMap::new();
        payloads.insert("imu".to_string(), vec![1.0, 1.0, 1.0]);
        let step = runtime.step(&payloads).unwrap();
        assert!(step.halted);
        assert!(step.commands.contains_key("halt"));
    }
}
