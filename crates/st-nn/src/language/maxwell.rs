// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::geometry::ConceptHint;
use crate::PureResult;
use st_core::theory::maxwell::MaxwellZPulse;
use st_tensor::TensorError;
use std::collections::HashMap;

/// Couples Maxwell Z pulses with concept windows so the desire lagrangian can
/// project coded-envelope detections back onto the language manifold.
#[derive(Clone, Debug)]
pub struct MaxwellDesireBridge {
    smoothing: f32,
    magnitude_floor: f32,
    channels: HashMap<String, Vec<(usize, f32)>>,
}

impl MaxwellDesireBridge {
    /// Creates an empty bridge with small default smoothing.
    pub fn new() -> Self {
        Self {
            smoothing: 1e-4,
            magnitude_floor: 0.0,
            channels: HashMap::new(),
        }
    }

    /// Ensures every registered window receives at least this additive weight
    /// before being scaled by the detected Z magnitude.
    pub fn with_smoothing(mut self, smoothing: f32) -> Self {
        self.smoothing = smoothing.max(0.0);
        self
    }

    /// Floors the |Z| magnitude used when scaling concept weights. This keeps
    /// the hint informative even when the sequential statistic is small but
    /// still above threshold.
    pub fn with_magnitude_floor(mut self, floor: f32) -> Self {
        self.magnitude_floor = floor.max(0.0);
        self
    }

    /// Returns the number of channels registered with the bridge.
    pub fn len(&self) -> usize {
        self.channels.len()
    }

    /// Returns true when no channels have been registered yet.
    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }

    /// Registers a concept window for the provided channel name.
    pub fn register_channel(
        &mut self,
        channel: impl Into<String>,
        window: Vec<(usize, f32)>,
    ) -> PureResult<()> {
        let channel = channel.into();
        if channel.trim().is_empty() {
            return Err(TensorError::InvalidValue {
                label: "channel name cannot be empty",
            });
        }
        if window.is_empty() {
            return Err(TensorError::InvalidValue {
                label: "Maxwell concept window cannot be empty",
            });
        }
        let mut sanitized = Vec::with_capacity(window.len());
        for (token, weight) in window.into_iter() {
            if weight.is_nan() || weight.is_infinite() {
                return Err(TensorError::InvalidValue {
                    label: "window weight must be finite",
                });
            }
            if weight < 0.0 {
                return Err(TensorError::InvalidValue {
                    label: "window weight must be non-negative",
                });
            }
            sanitized.push((token, weight));
        }
        self.channels.insert(channel, sanitized);
        Ok(())
    }

    /// Convenience builder that registers a channel and returns the bridge.
    pub fn with_channel(
        mut self,
        channel: impl Into<String>,
        window: Vec<(usize, f32)>,
    ) -> PureResult<Self> {
        self.register_channel(channel, window)?;
        Ok(self)
    }

    /// Returns a concept hint aligned with the provided channel, scaled by the
    /// magnitude of the supplied Maxwell pulse.
    pub fn hint_for(&self, channel: impl AsRef<str>, pulse: &MaxwellZPulse) -> Option<ConceptHint> {
        let window = self.channels.get(channel.as_ref())?;
        if window.is_empty() {
            return None;
        }
        let magnitude = (pulse.magnitude() as f32).max(self.magnitude_floor);
        if magnitude <= f32::EPSILON && self.smoothing <= f32::EPSILON {
            return None;
        }
        let mut scaled = Vec::with_capacity(window.len());
        for &(token, weight) in window {
            let base = weight.max(0.0) + self.smoothing;
            let value = base * magnitude;
            if value > f32::EPSILON {
                scaled.push((token, value));
            }
        }
        if scaled.is_empty() {
            None
        } else {
            Some(ConceptHint::Window(scaled))
        }
    }

    /// Returns true when the bridge has a concept window for the given channel.
    pub fn contains(&self, channel: impl AsRef<str>) -> bool {
        self.channels.contains_key(channel.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_pulse() -> MaxwellZPulse {
        MaxwellZPulse {
            blocks: 6,
            mean: 0.32,
            standard_error: 0.07,
            z_score: 3.4,
            band_energy: (0.5, 0.3, 0.1),
            z_bias: 0.41,
        }
    }

    #[test]
    fn bridge_registers_and_scales() {
        let mut bridge = MaxwellDesireBridge::new()
            .with_smoothing(0.05)
            .with_magnitude_floor(0.2);
        bridge
            .register_channel("alpha", vec![(0, 0.6), (1, 0.4)])
            .unwrap();
        assert!(bridge.contains("alpha"));
        let hint = bridge.hint_for("alpha", &sample_pulse()).unwrap();
        match hint {
            ConceptHint::Window(window) => {
                assert_eq!(window.len(), 2);
                for &(_, weight) in &window {
                    assert!(weight > 0.0);
                }
            }
            _ => panic!("expected window"),
        }
    }

    #[test]
    fn empty_channel_rejected() {
        let mut bridge = MaxwellDesireBridge::new();
        let err = bridge.register_channel(" ", vec![(0, 0.2)]).unwrap_err();
        assert!(matches!(err, TensorError::InvalidValue { .. }));
    }

    #[test]
    fn zero_weights_need_smoothing() {
        let bridge = MaxwellDesireBridge::new()
            .with_smoothing(0.0)
            .with_channel("beta", vec![(2, 0.0)])
            .unwrap();
        assert!(bridge.hint_for("beta", &sample_pulse()).is_none());
        let bridge = bridge.with_smoothing(0.1);
        // With smoothing the same pulse should now yield a hint.
        assert!(bridge.hint_for("beta", &sample_pulse()).is_some());
    }
}
