use std::collections::HashMap;
use std::time::SystemTime;

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
}

impl SensorChannel {
    fn new(dimension: usize) -> Self {
        let channel = Self {
            dimension,
            bias: vec![0.0; dimension],
            scale: 1.0,
            smoothing: None,
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

    fn apply(&mut self, values: &[f32]) -> Vec<f32> {
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
        adjusted
    }
}

/// Captures fused, calibrated observations across registered modalities.
#[derive(Debug, Clone)]
pub struct FusedFrame {
    pub coordinates: HashMap<String, Vec<f32>>,
    pub timestamp: SystemTime,
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
        let name = name.into();
        if dimension == 0 {
            return Err(RoboticsError::InvalidDimension { channel: name });
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
            let adjusted = channel.apply(values);
            coordinates.insert(name.clone(), adjusted);
        }
        Ok(FusedFrame {
            coordinates,
            timestamp: SystemTime::now(),
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
    }
}
