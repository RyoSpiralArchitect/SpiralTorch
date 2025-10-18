// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::geometry::ConceptHint;
use crate::PureResult;
use serde::{Deserialize, Serialize};
use st_core::theory::maxwell::MaxwellZPulse;
use st_tensor::TensorError;
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NarrativeHint {
    channel: String,
    tags: Vec<String>,
    intensity: f32,
}

impl NarrativeHint {
    pub fn new(channel: impl Into<String>, tags: Vec<String>, intensity: f32) -> Self {
        Self {
            channel: channel.into(),
            tags,
            intensity,
        }
    }

    pub fn channel(&self) -> &str {
        &self.channel
    }

    pub fn tags(&self) -> &[String] {
        &self.tags
    }

    pub fn intensity(&self) -> f32 {
        self.intensity
    }
}

#[derive(Clone, Debug)]
struct ChannelProgram {
    window: Vec<(usize, f32)>,
    tags: Vec<String>,
    narrative_gain: f32,
}

/// Couples Maxwell Z pulses with concept windows so the desire lagrangian can
/// project coded-envelope detections back onto the language manifold.
#[derive(Clone, Debug)]
pub struct MaxwellDesireBridge {
    smoothing: f32,
    magnitude_floor: f32,
    channels: HashMap<String, ChannelProgram>,
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
        self.register_channel_internal(channel.into(), window, Vec::new(), 1.0)
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

    pub fn register_channel_with_narrative(
        &mut self,
        channel: impl Into<String>,
        window: Vec<(usize, f32)>,
        tags: Vec<String>,
    ) -> PureResult<()> {
        self.register_channel_internal(channel.into(), window, tags, 1.0)
    }

    pub fn with_channel_and_narrative(
        mut self,
        channel: impl Into<String>,
        window: Vec<(usize, f32)>,
        tags: Vec<String>,
    ) -> PureResult<Self> {
        self.register_channel_with_narrative(channel, window, tags)?;
        Ok(self)
    }

    pub fn set_narrative_tags(
        &mut self,
        channel: impl AsRef<str>,
        tags: Vec<String>,
    ) -> PureResult<()> {
        let program = self
            .channels
            .get_mut(channel.as_ref())
            .ok_or(TensorError::InvalidValue {
                label: "channel is not registered",
            })?;
        program.tags = sanitise_tags(tags)?;
        Ok(())
    }

    pub fn with_narrative_gain(mut self, channel: impl AsRef<str>, gain: f32) -> Self {
        if let Some(program) = self.channels.get_mut(channel.as_ref()) {
            program.narrative_gain = gain.max(0.0);
        }
        self
    }

    pub fn set_narrative_gain(&mut self, channel: impl AsRef<str>, gain: f32) -> PureResult<()> {
        let program = self
            .channels
            .get_mut(channel.as_ref())
            .ok_or(TensorError::InvalidValue {
                label: "channel is not registered",
            })?;
        program.narrative_gain = gain.max(0.0);
        Ok(())
    }

    /// Returns a concept hint aligned with the provided channel, scaled by the
    /// magnitude of the supplied Maxwell pulse.
    pub fn hint_for(&self, channel: impl AsRef<str>, pulse: &MaxwellZPulse) -> Option<ConceptHint> {
        let program = self.channels.get(channel.as_ref())?;
        if program.window.is_empty() {
            return None;
        }
        let magnitude = (pulse.magnitude() as f32).max(self.magnitude_floor);
        if magnitude <= f32::EPSILON && self.smoothing <= f32::EPSILON {
            return None;
        }
        let mut scaled = Vec::with_capacity(program.window.len());
        for &(token, weight) in &program.window {
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

    pub fn channel_names(&self) -> Vec<String> {
        self.channels.keys().cloned().collect()
    }

    pub fn narrative_tags(&self, channel: impl AsRef<str>) -> Option<&[String]> {
        self.channels
            .get(channel.as_ref())
            .map(|program| program.tags.as_slice())
    }

    pub fn narrative_gain(&self, channel: impl AsRef<str>) -> Option<f32> {
        self.channels
            .get(channel.as_ref())
            .map(|program| program.narrative_gain)
    }

    pub fn narrative_for(
        &self,
        channel: impl AsRef<str>,
        pulse: &MaxwellZPulse,
    ) -> Option<NarrativeHint> {
        let program = self.channels.get(channel.as_ref())?;
        if program.tags.is_empty() {
            return None;
        }
        let magnitude = (pulse.magnitude() as f32).max(self.magnitude_floor);
        if magnitude <= f32::EPSILON {
            return None;
        }
        let intensity = magnitude * program.narrative_gain;
        if intensity <= f32::EPSILON {
            return None;
        }
        Some(NarrativeHint::new(
            channel.as_ref(),
            program.tags.clone(),
            intensity,
        ))
    }

    pub fn emit(
        &self,
        channel: impl AsRef<str>,
        pulse: &MaxwellZPulse,
    ) -> Option<(ConceptHint, Option<NarrativeHint>)> {
        let concept = self.hint_for(channel.as_ref(), pulse)?;
        let narrative = self.narrative_for(channel, pulse);
        Some((concept, narrative))
    }

    fn register_channel_internal(
        &mut self,
        channel: String,
        window: Vec<(usize, f32)>,
        tags: Vec<String>,
        gain: f32,
    ) -> PureResult<()> {
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
        let tags = sanitise_tags(tags)?;
        self.channels.insert(
            channel,
            ChannelProgram {
                window: sanitized,
                tags,
                narrative_gain: gain.max(0.0),
            },
        );
        Ok(())
    }
}

fn sanitise_tags(tags: Vec<String>) -> PureResult<Vec<String>> {
    if tags.is_empty() {
        return Ok(Vec::new());
    }
    let mut sanitized = Vec::with_capacity(tags.len());
    for tag in tags {
        let trimmed = tag.trim();
        if trimmed.is_empty() {
            return Err(TensorError::InvalidValue {
                label: "narrative tag cannot be empty",
            });
        }
        sanitized.push(trimmed.to_string());
    }
    sanitized.sort();
    sanitized.dedup();
    Ok(sanitized)
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

    #[test]
    fn narrative_hint_tracks_tags() {
        let mut bridge = MaxwellDesireBridge::new();
        bridge
            .register_channel_with_narrative(
                "gamma",
                vec![(0, 0.9), (1, 0.1)],
                vec!["glimmer".into(), "braid".into()],
            )
            .unwrap();
        bridge.set_narrative_gain("gamma", 2.0).unwrap();
        let pulse = sample_pulse();
        let narrative = bridge.narrative_for("gamma", &pulse).unwrap();
        assert_eq!(narrative.channel(), "gamma");
        assert_eq!(narrative.tags().len(), 2);
        assert!(narrative.intensity() > 0.0);
    }

    #[test]
    fn narrative_tags_cannot_be_empty_strings() {
        let result = MaxwellDesireBridge::new().register_channel_with_narrative(
            "delta",
            vec![(0, 0.4)],
            vec!["  ".into()],
        );
        assert!(result.is_err());
    }
}
