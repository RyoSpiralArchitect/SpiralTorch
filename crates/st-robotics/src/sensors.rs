use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use crate::error::RoboticsError;

#[derive(Debug, Clone)]
struct ExponentialSmoother {
    alpha: f32,
    state: Option<f32>,
}

impl ExponentialSmoother {
    fn new(alpha: f32) -> Self {
        Self { alpha, state: None }
    }

    fn update(&mut self, value: f32) -> f32 {
        let next = match self.state {
            Some(current) => self.alpha * value + (1.0 - self.alpha) * current,
            None => value,
        };
        self.state = Some(next);
        next
    }
}

#[derive(Debug, Clone)]
struct SensorChannel {
    dimension: usize,
    bias: Vec<f32>,
    scale: f32,
    smoothing: Option<Vec<ExponentialSmoother>>,
    optional: bool,
    max_staleness: Option<Duration>,
    last_update: Option<SystemTime>,
}

impl SensorChannel {
    fn new(dimension: usize) -> Self {
        let channel = Self {
            dimension,
            bias: vec![0.0; dimension],
            scale: 1.0,
            smoothing: None,
            optional: false,
            max_staleness: None,
            last_update: None,
        };
        channel
    }

    fn set_smoothing(&mut self, alpha: f32) -> Result<(), RoboticsError> {
        if !(0.0..=1.0).contains(&alpha) || alpha == 0.0 {
            return Err(RoboticsError::InvalidSmoothingCoefficient {
                channel: String::new(),
                alpha,
            });
        }
        let mut filters = Vec::with_capacity(self.dimension);
        for _ in 0..self.dimension {
            filters.push(ExponentialSmoother::new(alpha));
        }
        self.smoothing = Some(filters);
        Ok(())
    }

    fn clear_smoothing(&mut self) {
        self.smoothing = None;
    }

    fn apply(&mut self, values: &[f32], now: SystemTime) -> (Vec<f32>, bool) {
        let mut adjusted = Vec::with_capacity(values.len());
        for (idx, value) in values.iter().enumerate() {
            let bias = self.bias.get(idx).copied().unwrap_or(0.0);
            let scaled = (value - bias) * self.scale;
            let output = match self.smoothing.as_mut() {
                Some(filters) => filters
                    .get_mut(idx)
                    .map(|filter| filter.update(scaled))
                    .unwrap_or(scaled),
                None => scaled,
            };
            adjusted.push(output);
        }
        let stale = match (self.max_staleness, self.last_update) {
            (Some(limit), Some(last)) => now.duration_since(last).unwrap_or_default() > limit,
            _ => false,
        };
        self.last_update = Some(now);
        (adjusted, stale)
    }
}

/// Runtime health metadata emitted per sensor channel.
#[derive(Debug, Clone, Default)]
pub struct ChannelHealth {
    pub stale: bool,
    pub optional: bool,
}

/// Captures fused, calibrated observations across registered modalities.
#[derive(Debug, Clone)]
pub struct FusedFrame {
    pub coordinates: HashMap<String, Vec<f32>>,
    pub timestamp: SystemTime,
    pub health: HashMap<String, ChannelHealth>,
}

impl FusedFrame {
    /// Compute the Euclidean norm of a fused channel if present.
    pub fn norm(&self, channel: &str) -> Option<f32> {
        self.coordinates
            .get(channel)
            .map(|values| values.iter().map(|value| value * value).sum::<f32>().sqrt())
    }
}

/// Registry tracking sensor channels and calibration before fusion.
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
        self.register_channel_with_smoothing(name, dimension, None)
    }

    pub fn register_channel_with_smoothing(
        &mut self,
        name: impl Into<String>,
        dimension: usize,
        smoothing: Option<f32>,
    ) -> Result<(), RoboticsError> {
        self.register_channel_with_options(name, dimension, smoothing, false, None)
    }

    pub fn register_channel_with_options(
        &mut self,
        name: impl Into<String>,
        dimension: usize,
        smoothing: Option<f32>,
        optional: bool,
        max_staleness: Option<f32>,
    ) -> Result<(), RoboticsError> {
        let name = name.into();
        if dimension == 0 {
            return Err(RoboticsError::InvalidDimension {
                channel: name.clone(),
            });
        }
        if self.channels.contains_key(&name) {
            return Err(RoboticsError::ChannelExists(name));
        }
        let mut channel = SensorChannel::new(dimension);
        if let Some(alpha) = smoothing {
            channel.set_smoothing(alpha).map_err(|err| match err {
                RoboticsError::InvalidSmoothingCoefficient { .. } => {
                    RoboticsError::InvalidSmoothingCoefficient {
                        channel: name.clone(),
                        alpha,
                    }
                }
                _ => err,
            })?;
        }
        if let Some(threshold) = max_staleness {
            if threshold <= 0.0 {
                return Err(RoboticsError::InvalidStalenessThreshold {
                    channel: name.clone(),
                    threshold,
                });
            }
            let millis = (threshold * 1000.0) as u64;
            channel.max_staleness = Some(Duration::from_millis(millis.max(1)));
        }
        channel.optional = optional;
        self.channels.insert(name, channel);
        Ok(())
    }

    pub fn calibrate(
        &mut self,
        name: &str,
        bias: Option<Vec<f32>>,
        scale: Option<f32>,
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
        }
        if let Some(scale) = scale {
            channel.scale = scale;
        }
        Ok(())
    }

    pub fn configure_smoothing(
        &mut self,
        name: &str,
        smoothing: Option<f32>,
    ) -> Result<(), RoboticsError> {
        let channel = self
            .channels
            .get_mut(name)
            .ok_or_else(|| RoboticsError::ChannelMissing(name.to_string()))?;
        match smoothing {
            Some(alpha) => channel.set_smoothing(alpha).map_err(|err| match err {
                RoboticsError::InvalidSmoothingCoefficient { .. } => {
                    RoboticsError::InvalidSmoothingCoefficient {
                        channel: name.to_string(),
                        alpha,
                    }
                }
                _ => err,
            }),
            None => {
                channel.clear_smoothing();
                Ok(())
            }
        }
    }

    pub fn fuse(
        &mut self,
        payloads: &HashMap<String, Vec<f32>>,
    ) -> Result<FusedFrame, RoboticsError> {
        let now = SystemTime::now();
        let mut coordinates = HashMap::with_capacity(self.channels.len());
        let mut health = HashMap::with_capacity(self.channels.len());
        for (name, channel) in self.channels.iter_mut() {
            if let Some(values) = payloads.get(name) {
                if values.len() != channel.dimension {
                    return Err(RoboticsError::DimensionMismatch {
                        channel: name.clone(),
                        expected: channel.dimension,
                        actual: values.len(),
                    });
                }
                let (adjusted, stale) = channel.apply(values, now);
                coordinates.insert(name.clone(), adjusted);
                health.insert(
                    name.clone(),
                    ChannelHealth {
                        stale,
                        optional: channel.optional,
                    },
                );
            } else if channel.optional {
                let stale = match (channel.max_staleness, channel.last_update) {
                    (Some(limit), Some(last)) => {
                        now.duration_since(last).unwrap_or_default() > limit
                    }
                    (Some(_), None) => true,
                    _ => false,
                };
                health.insert(
                    name.clone(),
                    ChannelHealth {
                        stale,
                        optional: true,
                    },
                );
            } else {
                return Err(RoboticsError::MissingRequiredPayload {
                    channel: name.clone(),
                });
            }
        }
        for name in payloads.keys() {
            if !self.channels.contains_key(name) {
                return Err(RoboticsError::ChannelMissing(name.clone()));
            }
        }
        Ok(FusedFrame {
            coordinates,
            timestamp: now,
            health,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensor_fusion_applies_calibration() {
        let mut hub = SensorFusionHub::new();
        hub.register_channel("imu", 3).unwrap();
        hub.calibrate("imu", Some(vec![0.1, 0.1, 0.1]), Some(2.0))
            .unwrap();
        let frame = hub
            .fuse(&HashMap::from([("imu".to_string(), vec![0.2, 0.4, 0.3])]))
            .unwrap();
        let imu = frame.coordinates.get("imu").unwrap();
        assert_eq!(imu.len(), 3);
        assert!((imu[0] - 0.2).abs() < 1e-6);
        assert!((imu[1] - 0.6).abs() < 1e-6);
        assert!((imu[2] - 0.4).abs() < 1e-6);
        assert!(frame.health.get("imu").is_some());
    }

    #[test]
    fn smoothing_filters_noise() {
        let mut hub = SensorFusionHub::new();
        hub.register_channel_with_smoothing("imu", 1, Some(0.5))
            .unwrap();
        let _ = hub
            .fuse(&HashMap::from([("imu".to_string(), vec![1.0])]))
            .unwrap();
        let frame = hub
            .fuse(&HashMap::from([("imu".to_string(), vec![0.0])]))
            .unwrap();
        let value = frame.coordinates["imu"][0];
        assert!(value > 0.0 && value < 1.0);
        assert_eq!(frame.health["imu"].stale, false);
    }

    #[test]
    fn optional_channel_reports_staleness() {
        let mut hub = SensorFusionHub::new();
        hub.register_channel_with_options("camera", 2, None, true, Some(0.001))
            .unwrap();
        // No payload yet, should mark as stale because no updates and threshold set.
        let frame = hub.fuse(&HashMap::new()).unwrap();
        assert!(frame.health["camera"].optional);
        assert!(frame.health["camera"].stale);
    }

    #[test]
    fn missing_required_channel_returns_error() {
        let mut hub = SensorFusionHub::new();
        hub.register_channel("imu", 3).unwrap();
        let result = hub.fuse(&HashMap::new());
        assert!(matches!(
            result,
            Err(RoboticsError::MissingRequiredPayload { channel }) if channel == "imu"
        ));
    }
}
