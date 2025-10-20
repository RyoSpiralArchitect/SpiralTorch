use std::f32::consts::TAU;

use st_vision::ZSpaceVolume;

use st_tensor::{PureResult, TensorError};

/// Configuration for rendering a [`ZSpaceVolume`] into time-aware slices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalRendererConfig {
    /// Number of frames to emit for the animation.
    pub frames: usize,
    /// Sampling rate used to evaluate temporal harmonics.
    pub sample_rate_hz: f32,
}

impl Default for TemporalRendererConfig {
    fn default() -> Self {
        Self {
            frames: 16,
            sample_rate_hz: 24.0,
        }
    }
}

/// A single time-aware slice emitted by the renderer.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalRenderSlice {
    pub frame_index: usize,
    pub time_seconds: f32,
    pub data: Vec<f32>,
}

impl TemporalRenderSlice {
    fn new(frame_index: usize, time_seconds: f32, data: Vec<f32>) -> Self {
        Self {
            frame_index,
            time_seconds,
            data,
        }
    }
}

/// Collection of slices arranged as a 4D `[t, z, y, x]` tensor layout.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalRenderOutput {
    pub frames: usize,
    pub depth: usize,
    pub height: usize,
    pub width: usize,
    pub slices: Vec<TemporalRenderSlice>,
}

impl TemporalRenderOutput {
    /// Returns the flat buffer for a given frame if it exists.
    pub fn frame(&self, index: usize) -> Option<&[f32]> {
        self.slices
            .iter()
            .find(|slice| slice.frame_index == index)
            .map(|slice| slice.data.as_slice())
    }
}

/// CPU-backed renderer that mirrors the GPU pipeline for tests and tooling.
#[derive(Clone, Debug)]
pub struct TemporalRenderer {
    config: TemporalRendererConfig,
}

impl TemporalRenderer {
    pub fn new(config: TemporalRendererConfig) -> Self {
        Self { config }
    }

    pub fn render(&self, volume: &ZSpaceVolume) -> PureResult<TemporalRenderOutput> {
        if self.config.frames == 0 {
            return Err(TensorError::InvalidValue {
                label: "temporal_frames",
            });
        }
        if !self.config.sample_rate_hz.is_finite() || self.config.sample_rate_hz <= 0.0 {
            return Err(TensorError::InvalidValue {
                label: "temporal_sample_rate",
            });
        }
        let voxel_count = volume.voxel_count();
        if voxel_count == 0 {
            return Ok(TemporalRenderOutput {
                frames: 0,
                depth: volume.depth(),
                height: volume.height(),
                width: volume.width(),
                slices: Vec::new(),
            });
        }
        let mut slices = Vec::with_capacity(self.config.frames);
        let harmonic_channels = volume.harmonic_channels().max(1);
        let time_delta = 1.0 / self.config.sample_rate_hz;
        let decay = volume.resonance_decay();
        let voxels = volume.voxels();
        let harmonics = if volume.harmonic_channels() == 0 {
            vec![0.0; voxel_count * harmonic_channels]
        } else {
            volume.temporal_harmonics().to_vec()
        };
        for frame in 0..self.config.frames {
            let t = frame as f32 * time_delta;
            let mut buffer = vec![0.0f32; voxel_count];
            for (voxel_index, value) in buffer.iter_mut().enumerate() {
                let mut base = voxels[voxel_index];
                let decay_coeff = decay
                    .get(voxel_index)
                    .copied()
                    .filter(|v| v.is_finite())
                    .unwrap_or(1.0)
                    .max(0.0);
                let harmonics_slice = &harmonics
                    [voxel_index * harmonic_channels..(voxel_index + 1) * harmonic_channels];
                for (channel, amplitude) in harmonics_slice.iter().enumerate() {
                    if amplitude.abs() <= f32::EPSILON {
                        continue;
                    }
                    let freq = (channel + 1) as f32;
                    base += amplitude * (freq * TAU * t).sin();
                }
                let damped = if decay_coeff <= f32::EPSILON {
                    0.0
                } else {
                    base * (-decay_coeff * t).exp()
                };
                *value = damped;
            }
            slices.push(TemporalRenderSlice::new(frame, t, buffer));
        }
        Ok(TemporalRenderOutput {
            frames: self.config.frames,
            depth: volume.depth(),
            height: volume.height(),
            width: volume.width(),
            slices,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_decay_and_harmonics() {
        let mut volume = ZSpaceVolume::zeros_with_temporal(1, 1, 2, 2).unwrap();
        volume.voxels_mut().copy_from_slice(&[0.5, 1.0]);
        volume
            .temporal_harmonics_mut()
            .copy_from_slice(&[0.25, 0.0, -0.25, 0.5]);
        volume.resonance_decay_mut().copy_from_slice(&[0.5, 0.5]);
        let renderer = TemporalRenderer::new(TemporalRendererConfig {
            frames: 3,
            sample_rate_hz: 10.0,
        });
        let output = renderer.render(&volume).unwrap();
        assert_eq!(output.frames, 3);
        assert_eq!(output.slices.len(), 3);
        assert_eq!(output.depth, 1);
        assert_eq!(output.width, 2);
        let first = output.frame(0).unwrap();
        assert_eq!(first.len(), 2);
        // first frame should match base values (no time evolution yet)
        assert!((first[0] - 0.5).abs() < 1e-4);
        assert!((first[1] - 1.0).abs() < 1e-4);
        let second = output.frame(1).unwrap();
        assert_eq!(second.len(), 2);
        // ensure decay reduces magnitude over time
        assert!(second[0].abs() <= first[0].abs() + 1e-3);
    }
}
